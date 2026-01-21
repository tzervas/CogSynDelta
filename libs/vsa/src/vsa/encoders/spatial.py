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
        position_vectors: Learned or random position encodings (1D)
        x_position_vectors: Separate x-dimension encodings (2D)
        y_position_vectors: Separate y-dimension encodings (2D)
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

        # Generate random position vectors for 1D encoding
        dtype = torch.cfloat if config.model == "FHRR" else torch.float32
        device = config.device if config.device is not None else "cpu"
        self.register_buffer(
            "position_vectors",
            torchhd.random(num_positions, config.dimension, dtype=dtype, device=device),
        )

        # Separate position vectors for 2D encoding (x and y dimensions)
        # Using independent random vectors ensures x and y coordinates don't share encoding space
        self.register_buffer(
            "x_position_vectors",
            torchhd.random(num_positions, config.dimension, dtype=dtype, device=device),
        )
        self.register_buffer(
            "y_position_vectors",
            torchhd.random(num_positions, config.dimension, dtype=dtype, device=device),
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

        Uses separate position vectors for x and y dimensions to avoid ambiguity
        between coordinates. Position (3, 5) will be encoded differently from (5, 3).

        Args:
            x_coords: X coordinates (shape: [batch])
            y_coords: Y coordinates (shape: [batch])

        Returns:
            Encoded 2D positions (shape: [batch, dimension])
        """
        from vsa.operations.binding import bind

        # Use separate position vector sets for x and y to avoid ambiguity
        x_encoded = self.x_position_vectors[x_coords]
        y_encoded = self.y_position_vectors[y_coords]

        # Bind x and y coordinates
        return bind(x_encoded, y_encoded)
