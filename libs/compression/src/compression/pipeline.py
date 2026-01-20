"""Staged compression pipeline combining MRL, QINCo2, RVQ, and BitNet.

Implements the compression strategy from ADR-0008/ADR-0014:

Stage 1: Calibration (baseline)                  → 0.67 @ 1× (current)
Stage 2: Matryoshka MRL                          → 0.85-0.90 @ 2-4×
Stage 3: QINCo2/RVQ                              → 0.90-0.92 @ 4-8×
Stage 4: VSA/Hopfield (optional)                 → 0.92-0.93 @ 8×
Stage 5: BitNet b1.58 (model weights)            → 0.93-0.95+ @ 10×

Target: ≥0.95 fidelity at 10x+ compression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from compression.mrl.matryoshka import MatryoshkaCompressor
from compression.qinco.codebook import QINCo2Compressor
from compression.rvq.quantizer import ResidualVectorQuantizer
from compression.config import CompressionConfig


class StagedCompressionPipeline(nn.Module):
    """Multi-stage compression pipeline targeting ≥0.95 fidelity at 10x compression.

    Combines:
    1. Matryoshka dimension truncation (2-4×)
    2. QINCo2 implicit neural codebooks (additional 2-4×)
    3. Optional: RVQ for further compression
    4. BitNet b1.58 for model weights (10×)

    Attributes:
        config: Compression configuration
        mrl_compressor: Matryoshka compressor
        qinco_compressor: QINCo2 compressor (optional)
        rvq_compressor: RVQ compressor (optional)
    """

    def __init__(
        self,
        embedding_dim: int,
        config: CompressionConfig,
        use_qinco: bool = True,
        use_rvq: bool = False,
    ) -> None:
        """Initialize staged compression pipeline.

        Args:
            embedding_dim: Input embedding dimension
            config: Compression configuration
            use_qinco: Whether to use QINCo2 (default: True)
            use_rvq: Whether to use RVQ (default: False)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.config = config
        self.use_qinco = use_qinco
        self.use_rvq = use_rvq

        # Stage 1: Matryoshka truncation
        target_dim = config.mrl_dimensions[-1]  # Smallest dimension
        self.mrl_compressor = MatryoshkaCompressor(target_dim)

        # Stage 2: QINCo2 (optional)
        if use_qinco:
            self.qinco_compressor = QINCo2Compressor(
                embedding_dim=target_dim,
                num_stages=config.qinco_stages,
                codebook_size=config.qinco_codebook_size,
            )
        else:
            self.qinco_compressor = None

        # Stage 3: RVQ (optional)
        if use_rvq:
            self.rvq_compressor = ResidualVectorQuantizer(
                embedding_dim=target_dim,
                num_stages=config.rvq_stages,
                codebook_size=config.rvq_codebook_size,
            )
        else:
            self.rvq_compressor = None

    def compress(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compress embeddings through all stages.

        Args:
            embeddings: Input embeddings (shape: [batch, embedding_dim])

        Returns:
            Tuple of (final_compressed, stage_outputs) where stage_outputs
            contains intermediate results from each stage
        """
        stage_outputs = {}

        # Stage 1: Matryoshka truncation
        mrl_compressed = self.mrl_compressor.compress(embeddings)
        stage_outputs["mrl"] = mrl_compressed

        # Stage 2: QINCo2 (if enabled)
        if self.qinco_compressor is not None:
            qinco_indices, qinco_reconstructed = self.qinco_compressor.compress(mrl_compressed)
            stage_outputs["qinco_indices"] = qinco_indices
            stage_outputs["qinco_reconstructed"] = qinco_reconstructed
            current = qinco_reconstructed
        else:
            current = mrl_compressed

        # Stage 3: RVQ (if enabled)
        if self.rvq_compressor is not None:
            rvq_indices = self.rvq_compressor.encode(current)
            rvq_reconstructed = self.rvq_compressor.decode(rvq_indices)
            stage_outputs["rvq_indices"] = rvq_indices
            stage_outputs["rvq_reconstructed"] = rvq_reconstructed
            current = rvq_reconstructed

        return current, stage_outputs

    def decompress(
        self, compressed: torch.Tensor, stage_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Decompress embeddings.

        Note: Full decompression to original dimension pads with zeros (lossy).

        Args:
            compressed: Compressed embeddings or indices
            stage_outputs: Intermediate stage outputs from compress()

        Returns:
            Decompressed embeddings (shape: [batch, embedding_dim])
        """
        # If we have stage outputs with indices, reconstruct from them
        if stage_outputs is not None:
            if "rvq_indices" in stage_outputs:
                current = self.rvq_compressor.decode(stage_outputs["rvq_indices"])
            elif "qinco_indices" in stage_outputs:
                current = self.qinco_compressor.decode(stage_outputs["qinco_indices"])
            else:
                current = compressed
        else:
            current = compressed

        # Decompress from MRL (pad to original dimension)
        decompressed = self.mrl_compressor.decompress(current, self.embedding_dim)

        return decompressed

    def measure_fidelity(
        self, original: torch.Tensor, compressed: torch.Tensor
    ) -> Dict[str, float]:
        """Measure compression fidelity across stages.

        Args:
            original: Original embeddings (shape: [batch, embedding_dim])
            compressed: Compressed embeddings (shape: [batch, target_dim])

        Returns:
            Dictionary with fidelity metrics
        """
        # Truncate original to match compressed dimension
        target_dim = compressed.shape[1]
        original_truncated = original[:, :target_dim]

        # Normalize both
        original_norm = F.normalize(original_truncated, p=2, dim=1)
        compressed_norm = F.normalize(compressed, p=2, dim=1)

        # Cosine similarity
        cosine_sim = (original_norm * compressed_norm).sum(dim=1).mean().item()

        # MSE
        mse = F.mse_loss(original_truncated, compressed).item()

        # Compression ratio
        compression_ratio = original.shape[1] / compressed.shape[1]

        return {
            "cosine_similarity": cosine_sim,
            "mse": mse,
            "compression_ratio": compression_ratio,
        }

    def total_compression_ratio(self) -> float:
        """Calculate total compression ratio across all stages.

        Returns:
            Overall compression ratio
        """
        ratio = self.embedding_dim / self.mrl_compressor.target_dim

        if self.qinco_compressor is not None:
            ratio *= self.qinco_compressor.compression_ratio()

        if self.rvq_compressor is not None:
            ratio *= self.rvq_compressor.compression_ratio()

        return ratio


def create_compression_pipeline(
    embedding_dim: int = 2048,
    target_fidelity: float = 0.95,
    target_compression: float = 10.0,
) -> StagedCompressionPipeline:
    """Factory function to create optimized compression pipeline.

    Args:
        embedding_dim: Input embedding dimension
        target_fidelity: Target cosine similarity (default: 0.95)
        target_compression: Target compression ratio (default: 10.0)

    Returns:
        Configured StagedCompressionPipeline
    """
    # Configure for target metrics
    config = CompressionConfig()

    # Adjust MRL dimensions for target compression
    if target_compression <= 4.0:
        config.mrl_dimensions = [embedding_dim // 2, embedding_dim // 4]
        use_qinco = False
    elif target_compression <= 8.0:
        config.mrl_dimensions = [embedding_dim // 2, embedding_dim // 4]
        use_qinco = True
    else:  # > 8× compression
        config.mrl_dimensions = [embedding_dim // 4, embedding_dim // 8]
        use_qinco = True

    config.target_fidelity = target_fidelity
    config.target_compression_ratio = target_compression

    pipeline = StagedCompressionPipeline(
        embedding_dim=embedding_dim,
        config=config,
        use_qinco=use_qinco,
        use_rvq=False,  # Optional enhancement
    )

    return pipeline
