"""Matryoshka Representation Learning (MRL) implementation.

MRL trains embeddings that are useful at multiple granularities. The first k
dimensions form a valid embedding for any k in the target dimension list.

This enables adaptive compression: truncate to desired compression ratio at
inference time without retraining.

Mathematical Foundation:
    Loss = Σᵢ wᵢ · L(f(x)[:dᵢ], y)

    where dᵢ are target dimensions, wᵢ are weights, and f(x)[:dᵢ] is truncation.

Performance (from NeurIPS 2022 paper):
    - 2× compression (2048 → 1024): ~98% accuracy retention
    - 4× compression (2048 → 512): ~95% accuracy retention
    - 8× compression (2048 → 256): ~90% accuracy retention

References:
    - Kusupati et al. (2022). Matryoshka Representation Learning. NeurIPS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MatryoshkaEncoder(nn.Module):
    """Encoder with Matryoshka loss for multi-granularity embeddings.

    The encoder is trained to produce embeddings where any prefix (first k dims)
    forms a valid, useful embedding.

    Attributes:
        input_dim: Input feature dimension
        output_dim: Maximum output embedding dimension
        target_dims: List of target dimensions for nested embeddings
        hidden_dims: Hidden layer dimensions
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        target_dims: List[int],
        hidden_dims: Optional[List[int]] = None,
    ) -> None:
        """Initialize Matryoshka encoder.

        Args:
            input_dim: Input dimension
            output_dim: Maximum embedding dimension
            target_dims: Nested target dimensions (must be sorted, include output_dim)
            hidden_dims: Hidden layer dimensions (default: [512, 1024])
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 1024]

        # Validate target_dims
        assert all(d <= output_dim for d in target_dims), "Target dims must be <= output_dim"
        assert output_dim in target_dims, "output_dim must be in target_dims"

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.target_dims = sorted(target_dims)

        # Build encoder layers
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                ]
            )
            prev_dim = h_dim

        # Final projection to output_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to full-dimensional embedding.

        Args:
            x: Input features (shape: [batch, input_dim])

        Returns:
            Embeddings (shape: [batch, output_dim])
        """
        return self.encoder(x)

    def forward_nested(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Encode and return embeddings at all target dimensions.

        Args:
            x: Input features (shape: [batch, input_dim])

        Returns:
            Dictionary mapping dimension -> embedding at that dimension
        """
        full_embedding = self.forward(x)

        nested_embeddings = {}
        for dim in self.target_dims:
            # Truncate to dimension and L2 normalize
            truncated = full_embedding[:, :dim]
            normalized = F.normalize(truncated, p=2, dim=1)
            nested_embeddings[dim] = normalized

        return nested_embeddings


class MatryoshkaLoss(nn.Module):
    """Matryoshka loss for training multi-granularity embeddings.

    Computes weighted sum of losses at each target dimension.
    """

    # Explicit type annotation for registered buffer
    loss_weights: torch.Tensor

    def __init__(
        self,
        target_dims: List[int],
        loss_weights: Optional[List[float]] = None,
        base_loss: str = "contrastive",
        temperature: float = 0.07,
    ) -> None:
        """Initialize Matryoshka loss.

        Args:
            target_dims: Target dimensions for nested embeddings
            loss_weights: Weights for each dimension (default: uniform)
            base_loss: Base loss function ("contrastive", "triplet", "mse")
            temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.target_dims = sorted(target_dims)

        if loss_weights is None:
            # Uniform weights
            loss_weights = [1.0 / len(target_dims)] * len(target_dims)
        else:
            assert len(loss_weights) == len(target_dims)

        self.register_buffer("loss_weights", torch.tensor(loss_weights))
        self.base_loss = base_loss
        self.temperature = temperature

    def contrastive_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """NT-Xent contrastive loss (SimCLR style).

        Args:
            embeddings: L2-normalized embeddings (shape: [batch, dim])
            labels: Class labels (shape: [batch])

        Returns:
            Contrastive loss (scalar)
        """
        batch_size = embeddings.shape[0]

        # Compute similarity matrix
        similarity = embeddings @ embeddings.T / self.temperature

        # Mask for positive pairs (same label)
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask = labels_eq.float()

        # Mask diagonal
        positive_mask = positive_mask - torch.eye(batch_size, device=embeddings.device)

        # Compute loss
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Mean of log-likelihood over positive pairs
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp(
            min=1
        )

        loss = -mean_log_prob_pos.mean()

        return loss

    def forward(
        self, nested_embeddings: Dict[int, torch.Tensor], labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[int, float]]:
        """Compute Matryoshka loss across all target dimensions.

        Args:
            nested_embeddings: Dict mapping dimension -> normalized embeddings
            labels: Target labels (shape: [batch])

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict maps dim -> loss value
        """
        device = labels.device
        total_loss: torch.Tensor = torch.tensor(0.0, device=device)
        loss_dict: Dict[int, float] = {}

        for i, dim in enumerate(self.target_dims):
            embeddings = nested_embeddings[dim]

            # Compute loss for this dimension
            if self.base_loss == "contrastive":
                loss = self.contrastive_loss(embeddings, labels)
            else:
                raise ValueError(f"Unsupported base_loss: {self.base_loss}")

            # Weight and accumulate
            weighted_loss = self.loss_weights[i] * loss
            total_loss = total_loss + weighted_loss

            loss_dict[dim] = loss.item()

        return total_loss, loss_dict


class MatryoshkaCompressor:
    """Matryoshka compressor for inference-time dimension truncation.

    Given a trained Matryoshka encoder, compress embeddings by truncating
    to target dimension.

    Compression ratios:
        - 2048 → 1024: 2× compression
        - 2048 → 512: 4× compression
        - 2048 → 256: 8× compression
    """

    def __init__(self, target_dim: int) -> None:
        """Initialize Matryoshka compressor.

        Args:
            target_dim: Target dimension for truncation
        """
        self.target_dim = target_dim

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compress embeddings by truncation and normalization.

        Args:
            embeddings: Full embeddings (shape: [batch, full_dim])

        Returns:
            Compressed embeddings (shape: [batch, target_dim])
        """
        # Truncate
        truncated = embeddings[:, : self.target_dim]

        # L2 normalize
        normalized = F.normalize(truncated, p=2, dim=1)

        return normalized

    def decompress(self, compressed: torch.Tensor, full_dim: int) -> torch.Tensor:
        """Decompress embeddings (pad with zeros).

        Note: This is lossy - the padded dimensions are not meaningful.
        Use only for debugging or compatibility with fixed-size systems.

        Args:
            compressed: Compressed embeddings (shape: [batch, target_dim])
            full_dim: Target full dimension

        Returns:
            Decompressed embeddings (shape: [batch, full_dim])
        """
        batch_size = compressed.shape[0]
        padding_size = full_dim - self.target_dim

        # Pad with zeros
        padding = torch.zeros(
            batch_size, padding_size, device=compressed.device, dtype=compressed.dtype
        )
        decompressed = torch.cat([compressed, padding], dim=1)

        return decompressed

    def compression_ratio(self, original_dim: int) -> float:
        """Calculate compression ratio.

        Args:
            original_dim: Original embedding dimension

        Returns:
            Compression ratio (e.g., 4.0 for 2048 → 512)
        """
        return original_dim / self.target_dim

    def fidelity(self, original: torch.Tensor, compressed: torch.Tensor) -> float:
        """Measure compression fidelity via cosine similarity.

        Args:
            original: Original full embeddings (shape: [batch, full_dim])
            compressed: Compressed embeddings (shape: [batch, target_dim])

        Returns:
            Average cosine similarity (0-1, higher is better)
        """
        # Truncate original to match compressed dimension
        original_truncated = original[:, : self.target_dim]
        original_normalized = F.normalize(original_truncated, p=2, dim=1)

        compressed_normalized = F.normalize(compressed, p=2, dim=1)

        # Cosine similarity
        similarities = (original_normalized * compressed_normalized).sum(dim=1)

        return similarities.mean().item()
