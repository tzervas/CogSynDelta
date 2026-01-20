"""Spatial encoding for VSA."""

import torch
import torchhd
from torch import nn

from vsa.config import VSAConfig


class SpatialEncoder(nn.Module):
    """Spatial position encoding using random hypervectors.

    Encodes discrete spatial positions (e.g., grid coordinates) into
    hyperdimensional space using random projection.

    Attributes:
        config: VSA configuration
        position_vectors: Learned or random position encodings
    """

    def __init__(self, config: VSAConfig, num_positions: int) -> None:
        """Initialize spatial encoder.

        Args:
            config: VSA configuration
            num_positions: Number of discrete positions to encode
        """
        super().__init__()
        self.config = config
        self.num_positions = num_positions

        # Generate random position vectors
        dtype = torch.cfloat if config.model == "FHRR" else torch.float32
        self.register_buffer(
            "position_vectors",
            torchhd.random(num_positions, config.dimension, dtype=dtype, device=config.device),
        )

    def encode(self, positions: torch.Tensor) -> torch.Tensor:
        """Encode spatial positions.

        Args:
            positions: Integer position indices (shape: [batch])

        Returns:
            Encoded hypervectors (shape: [batch, dimension])
        """
        return self.position_vectors[positions]

    def encode_2d(self, x_coords: torch.Tensor, y_coords: torch.Tensor) -> torch.Tensor:
        """Encode 2D coordinates using binding.

        Args:
            x_coords: X coordinates (shape: [batch])
            y_coords: Y coordinates (shape: [batch])

        Returns:
            Encoded 2D positions (shape: [batch, dimension])
        """
        from vsa.operations.binding import bind

        x_encoded = self.position_vectors[x_coords]
        y_encoded = self.position_vectors[y_coords]

        # Bind x and y coordinates
        return bind(x_encoded, y_encoded)
