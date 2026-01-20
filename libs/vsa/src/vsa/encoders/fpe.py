"""Fractional Power Encoding (FPE) for continuous temporal grounding.

This module implements FPE using FHRR (Fourier Holographic Reduced Representations)
for encoding continuous temporal information in hyperdimensional space.

Mathematical Foundation:
    T^t = F⁻¹{F{T}^t}

    For FHRR phasor representations:
    T^t = exp(iθt) where θ ~ Uniform[-π, π]

    Similarity decay (THEORETICAL):
    sim(T^t1, T^t2) ≈ sinc(t1 - t2) = sin(π(t1-t2))/(π(t1-t2))

References:
    - Plate, T. A. (2003). Holographic reduced representation: Distributed
      representation for cognitive structures.
    - Kleyko et al. (2021). Vector symbolic architectures as a computing framework
      for nanoscale hardware. Proceedings of the IEEE, 110(10), 1561-1585.
"""

import torch
import torchhd
from torch import nn

from vsa.config import VSAConfig


class FractionalPowerEncoder(nn.Module):
    """Fractional Power Encoding for continuous temporal grounding.

    This encoder uses fractional exponentiation of complex phasors to encode
    continuous temporal information. Physical locality in time translates to
    representational similarity in hyperdimensional space.

    Attributes:
        config: VSA configuration
        base_vector: Random seed hypervector for FPE (shape: [dimension])

    Example:
        >>> config = VSAConfig(dimension=10000, model="FHRR", device="cuda")
        >>> encoder = FractionalPowerEncoder(config)
        >>> base = encoder.random_vector()
        >>> t0 = encoder.encode(base, 0.0)  # Time 0
        >>> t1 = encoder.encode(base, 1.0)  # Time 1
        >>> sim = encoder.cosine_similarity(t0, t1)  # ≈ sinc(1) ≈ 0
    """

    def __init__(self, config: VSAConfig) -> None:
        """Initialize FPE encoder.

        Args:
            config: VSA configuration with dimension, model, device
        """
        super().__init__()
        self.config = config

        # Validate model type
        if config.model != "FHRR":
            raise ValueError(f"FPE requires FHRR model, got {config.model}")

        # Create random base vector (complex phasors)
        self.register_buffer(
            "base_vector",
            torchhd.random(1, config.dimension, dtype=torch.cfloat, device=config.device),
        )

    def random_vector(self) -> torch.Tensor:
        """Generate a random hypervector.

        Returns:
            Random complex phasor hypervector of shape [dimension]
        """
        return torchhd.random(
            1, self.config.dimension, dtype=torch.cfloat, device=self.config.device
        ).squeeze(0)

    def encode(self, base: torch.Tensor, t: float) -> torch.Tensor:
        """Encode a single timestamp using fractional power.

        Args:
            base: Base hypervector (shape: [dimension])
            t: Continuous timestamp (float)

        Returns:
            Encoded hypervector (shape: [dimension])
        """
        # Fractional power encoding: base^t
        return base**t

    def encode_batch(self, bases: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """Encode a batch of timestamps.

        Args:
            bases: Base hypervectors (shape: [batch, dimension])
            timestamps: Continuous timestamps (shape: [batch])

        Returns:
            Encoded hypervectors (shape: [batch, dimension])
        """
        # Expand timestamps for broadcasting: [batch, 1]
        t_expanded = timestamps.unsqueeze(-1)

        # Fractional power encoding: bases^t
        return bases**t_expanded

    def encode_temporal(
        self, base_vector: torch.Tensor, timestamps: torch.Tensor
    ) -> torch.Tensor:
        """Encode temporal information using FPE (API from technical spec).

        Args:
            base_vector: Base hypervector (shape: [batch, dim]) or [dim]
            timestamps: Continuous timestamps (shape: [batch])

        Returns:
            Temporally encoded hypervectors (shape: [batch, dim])
        """
        if base_vector.ndim == 1:
            # Single base vector, broadcast to batch
            base_vector = base_vector.unsqueeze(0).expand(timestamps.shape[0], -1)

        return self.encode_batch(base_vector, timestamps)

    @staticmethod
    def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between hypervectors.

        For complex vectors: sim = Re(x · conj(y)) / (||x|| ||y||)

        Args:
            x: First hypervector (shape: [dimension] or [batch, dimension])
            y: Second hypervector (shape: [dimension] or [batch, dimension])

        Returns:
            Cosine similarity (scalar or shape: [batch])
        """
        return torchhd.cosine_similarity(x, y)

    def verify_sinc_decay(
        self, base: torch.Tensor, t_values: list[float], tolerance: float = 0.05
    ) -> dict[float, tuple[float, float, bool]]:
        """Verify theoretical sinc decay property.

        Args:
            base: Base hypervector (shape: [dimension])
            t_values: List of time offsets to test
            tolerance: Acceptable deviation from theoretical sinc

        Returns:
            Dictionary mapping t -> (measured_sim, expected_sim, within_tolerance)
        """
        t0 = self.encode(base, 0.0)
        results = {}

        for t in t_values:
            t_encoded = self.encode(base, t)
            measured_sim = self.cosine_similarity(t0, t_encoded).item()

            # Expected: sinc(t) = sin(πt)/(πt) for t≠0, 1 for t=0
            if abs(t) < 1e-6:
                expected_sim = 1.0
            else:
                expected_sim = torch.sinc(torch.tensor(t)).item()

            within_tolerance = abs(measured_sim - expected_sim) < tolerance
            results[t] = (measured_sim, expected_sim, within_tolerance)

        return results

    def capacity_bound(self, k: int, S: float, M: int) -> int:
        """Calculate required dimension for capacity bound.

        Capacity bound: n ≥ k/(1-S²) × log(M)

        Args:
            k: Number of items to store
            S: Target similarity threshold
            M: Number of position/time values

        Returns:
            Required minimum dimension
        """
        import math

        return math.ceil(k / (1 - S**2) * math.log(M))
