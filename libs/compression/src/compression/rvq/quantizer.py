"""Residual Vector Quantization (RVQ).

RVQ iteratively quantizes residuals from previous stages, achieving high
compression with multi-stage refinement.

Mathematical Foundation:
    x̂ = Σᵢ qᵢ(rᵢ₋₁)
    where r₀ = x, rᵢ = rᵢ₋₁ - qᵢ(rᵢ₋₁)

References:
    - Zeghidour et al. (2021). SoundStream: An End-to-End Neural Audio Codec. IEEE/ACM TASLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VectorQuantizer(nn.Module):
    """Single-stage vector quantizer with learnable codebook."""

    def __init__(self, embedding_dim: int, codebook_size: int) -> None:
        """Initialize vector quantizer.

        Args:
            embedding_dim: Dimension of vectors to quantize
            codebook_size: Number of code vectors (e.g., 256 for 8-bit)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size

        # Learnable codebook
        self.codebook = nn.Parameter(torch.randn(codebook_size, embedding_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize vectors.

        Args:
            x: Input vectors (shape: [batch, embedding_dim])

        Returns:
            Tuple of (quantized, indices, commitment_loss)
        """
        # Compute distances to all codes: [batch, codebook_size]
        distances = torch.cdist(x, self.codebook)

        # Find nearest code
        indices = distances.argmin(dim=1)

        # Get quantized vectors
        quantized = self.codebook[indices]

        # Straight-through estimator: copy gradients from quantized to input
        quantized = x + (quantized - x).detach()

        # Commitment loss (for training)
        commitment_loss = F.mse_loss(x.detach(), quantized)

        return quantized, indices, commitment_loss


class ResidualVectorQuantizer(nn.Module):
    """Multi-stage residual vector quantizer.

    Attributes:
        num_stages: Number of quantization stages
        quantizers: List of vector quantizers
    """

    def __init__(self, embedding_dim: int, num_stages: int = 4, codebook_size: int = 256) -> None:
        """Initialize RVQ.

        Args:
            embedding_dim: Dimension of vectors to compress
            num_stages: Number of residual stages
            codebook_size: Codebook size per stage
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_stages = num_stages
        self.codebook_size = codebook_size

        # Create quantizers for each stage
        self.quantizers = nn.ModuleList(
            [VectorQuantizer(embedding_dim, codebook_size) for _ in range(num_stages)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize with residual stages.

        Args:
            x: Input vectors (shape: [batch, embedding_dim])

        Returns:
            Tuple of (quantized, indices, commitment_loss)
        """
        batch_size = x.shape[0]
        indices = torch.zeros(batch_size, self.num_stages, dtype=torch.long, device=x.device)

        residual = x
        quantized_sum = torch.zeros_like(x)
        total_commitment_loss = 0.0

        for stage, quantizer in enumerate(self.quantizers):
            # Quantize residual
            quantized, idx, commitment_loss = quantizer(residual)

            # Store indices
            indices[:, stage] = idx

            # Accumulate quantized
            quantized_sum = quantized_sum + quantized

            # Update residual
            residual = residual - quantized

            # Accumulate loss
            total_commitment_loss = total_commitment_loss + commitment_loss

        return quantized_sum, indices, total_commitment_loss / self.num_stages

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to indices (compression).

        Args:
            x: Input vectors (shape: [batch, embedding_dim])

        Returns:
            Code indices (shape: [batch, num_stages])
        """
        _, indices, _ = self.forward(x)
        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode indices to vectors (decompression).

        Args:
            indices: Code indices (shape: [batch, num_stages])

        Returns:
            Reconstructed vectors (shape: [batch, embedding_dim])
        """
        batch_size = indices.shape[0]
        reconstructed = torch.zeros(batch_size, self.embedding_dim, device=indices.device)

        for stage, quantizer in enumerate(self.quantizers):
            # Get code vectors for this stage
            codes = quantizer.codebook[indices[:, stage]]
            reconstructed = reconstructed + codes

        return reconstructed

    def compression_ratio(self) -> float:
        """Calculate compression ratio.

        Returns:
            Compression ratio (float32 → 8-bit × num_stages)
        """
        original_bytes = self.embedding_dim * 4  # float32
        compressed_bytes = self.num_stages * 1  # 8-bit per stage
        return original_bytes / compressed_bytes
