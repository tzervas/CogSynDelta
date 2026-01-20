"""QINCo2: Implicit Neural Codebook Compression.

QINCo2 uses learnable neural networks as implicit codebooks for vector quantization.
Achieves 34-44% MSE reduction over standard RVQ with residual conditioning.

Mathematical Foundation:
    x̂_m = f_θ(c_m | x̂_{m-1})

    where f_θ is a neural network that generates code vectors conditioned on
    previous stage residuals.

Performance (ICLR 2025):
    - 34-44% MSE reduction over RVQ
    - 24% retrieval improvement at 8-byte compression
    - Scales to billion-vector datasets

References:
    - Vallaeys et al. (2025). Qinco2: Vector Compression and Search with
      Improved Implicit Neural Codebooks. ICLR.
    - Code: github.com/facebookresearch/Qinco
"""

import torch
import torch.nn as nn
from typing import Tuple


class ImplicitCodebook(nn.Module):
    """Implicit neural codebook.

    Instead of storing explicit code vectors, learns a neural network
    that maps code indices to vectors.
    """

    def __init__(self, embedding_dim: int, codebook_size: int, hidden_dim: int = 256) -> None:
        """Initialize implicit codebook.

        Args:
            embedding_dim: Dimension of embeddings to quantize
            codebook_size: Number of codes (e.g., 256 for 8-bit)
            hidden_dim: Hidden layer dimension for implicit network
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size

        # Implicit codebook: index -> vector via neural network
        self.codebook = nn.Sequential(
            nn.Embedding(codebook_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Generate code vectors from indices.

        Args:
            indices: Code indices (shape: [batch] or [batch, num_stages])

        Returns:
            Code vectors (shape: [batch, embedding_dim] or [batch, num_stages, embedding_dim])
        """
        return self.codebook(indices)

    def lookup(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find nearest code for input vectors.

        Args:
            x: Input vectors (shape: [batch, embedding_dim])

        Returns:
            Tuple of (code_indices, quantized_vectors)
        """
        # Generate all code vectors
        all_indices = torch.arange(self.codebook_size, device=x.device)
        all_codes = self.forward(all_indices)  # [codebook_size, embedding_dim]

        # Compute distances to all codes
        # [batch, embedding_dim] @ [embedding_dim, codebook_size]
        similarities = x @ all_codes.T

        # Find nearest (highest similarity)
        indices = similarities.argmax(dim=1)

        # Get quantized vectors
        quantized = self.forward(indices)

        return indices, quantized


class ResidualConditionedCodebook(nn.Module):
    """Residual-conditioned implicit codebook for QINCo2.

    Conditions code generation on previous stage residuals.
    """

    def __init__(self, embedding_dim: int, codebook_size: int, hidden_dim: int = 256) -> None:
        """Initialize residual-conditioned codebook.

        Args:
            embedding_dim: Dimension of embeddings
            codebook_size: Number of codes
            hidden_dim: Hidden dimension
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size

        # Codebook conditioned on residual
        self.codebook = nn.Sequential(
            nn.Linear(embedding_dim + codebook_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Embedding for code indices
        self.code_embedding = nn.Embedding(codebook_size, codebook_size)

    def forward(self, indices: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Generate code vectors conditioned on residual.

        Args:
            indices: Code indices (shape: [batch])
            residual: Previous stage residual (shape: [batch, embedding_dim])

        Returns:
            Code vectors (shape: [batch, embedding_dim])
        """
        # Embed indices
        code_emb = self.code_embedding(indices)

        # Concatenate with residual
        conditioned_input = torch.cat([residual, code_emb], dim=1)

        # Generate code
        return self.codebook(conditioned_input)


class QINCo2Compressor(nn.Module):
    """QINCo2 multi-stage residual compressor with implicit neural codebooks.

    Attributes:
        num_stages: Number of residual quantization stages
        codebooks: List of residual-conditioned implicit codebooks
    """

    def __init__(
        self,
        embedding_dim: int,
        num_stages: int = 4,
        codebook_size: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        """Initialize QINCo2 compressor.

        Args:
            embedding_dim: Dimension of embeddings to compress
            num_stages: Number of residual stages
            codebook_size: Size of each codebook (default: 256 for 8-bit)
            hidden_dim: Hidden dimension for implicit networks
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_stages = num_stages
        self.codebook_size = codebook_size

        # First stage: standard implicit codebook
        self.first_codebook = ImplicitCodebook(embedding_dim, codebook_size, hidden_dim)

        # Subsequent stages: residual-conditioned
        self.residual_codebooks = nn.ModuleList(
            [
                ResidualConditionedCodebook(embedding_dim, codebook_size, hidden_dim)
                for _ in range(num_stages - 1)
            ]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode vectors to code indices (compression).

        Args:
            x: Input vectors (shape: [batch, embedding_dim])

        Returns:
            Code indices (shape: [batch, num_stages])
        """
        batch_size = x.shape[0]
        indices = torch.zeros(batch_size, self.num_stages, dtype=torch.long, device=x.device)

        residual = x

        # First stage
        idx, quantized = self.first_codebook.lookup(residual)
        indices[:, 0] = idx
        residual = residual - quantized

        # Residual stages
        for stage in range(1, self.num_stages):
            # Find best code conditioned on residual
            all_indices = torch.arange(self.codebook_size, device=x.device)
            # Broadcast for batch
            all_indices_batch = all_indices.unsqueeze(0).expand(batch_size, -1)
            residual_batch = residual.unsqueeze(1).expand(-1, self.codebook_size, -1)

            # Generate all codes for this stage
            # Reshape for batched lookup
            flat_indices = all_indices_batch.reshape(-1)
            flat_residuals = residual_batch.reshape(-1, self.embedding_dim)

            all_codes = self.residual_codebooks[stage - 1](flat_indices, flat_residuals)
            all_codes = all_codes.reshape(batch_size, self.codebook_size, self.embedding_dim)

            # Find nearest
            similarities = (residual.unsqueeze(1) * all_codes).sum(dim=2)
            idx = similarities.argmax(dim=1)

            # Get quantized
            quantized = all_codes[torch.arange(batch_size), idx]

            indices[:, stage] = idx
            residual = residual - quantized

        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode code indices to vectors (decompression).

        Args:
            indices: Code indices (shape: [batch, num_stages])

        Returns:
            Reconstructed vectors (shape: [batch, embedding_dim])
        """
        # First stage
        reconstructed = self.first_codebook(indices[:, 0])

        # Residual stages
        residual = torch.zeros_like(reconstructed)
        for stage in range(1, self.num_stages):
            # Compute residual from previous stages
            prev_reconstructed = reconstructed.clone()

            # Generate code for this stage
            code = self.residual_codebooks[stage - 1](indices[:, stage], residual)

            # Add to reconstruction
            reconstructed = reconstructed + code

            # Update residual for next stage (approximation)
            residual = prev_reconstructed - reconstructed

        return reconstructed

    def compress(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress vectors.

        Args:
            x: Input vectors (shape: [batch, embedding_dim])

        Returns:
            Tuple of (code_indices, reconstructed_vectors)
        """
        indices = self.encode(x)
        reconstructed = self.decode(indices)
        return indices, reconstructed

    def compression_ratio(self) -> float:
        """Calculate compression ratio.

        Returns:
            Compression ratio (e.g., 8.0 for float32 → 8-bit × 4 stages)
        """
        # Original: embedding_dim × 4 bytes (float32)
        # Compressed: num_stages × 1 byte (8-bit indices)
        original_bytes = self.embedding_dim * 4
        compressed_bytes = self.num_stages * 1
        return original_bytes / compressed_bytes
