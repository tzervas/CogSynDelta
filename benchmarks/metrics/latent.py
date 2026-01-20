"""CogSynDelta latent space metrics.

This module provides metrics specific to CogSynDelta's latent space
operations: engram encoding/decoding, VAE latent operations, and
encoding fidelity measurements.

Classes:
    LatentSpaceMetrics: Engrams/sec, encoding fidelity, latent timing

Why CogSynDelta-specific:
    Latent space operations (engrams, VAE encoding/decoding) are the core
    primitive of CogSynDelta. Standard throughput metrics don't capture
    the semantic richness of latent representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from benchmarks.metrics.base import METRIC_REGISTRY, MetricMixin


@dataclass
class LatentSpaceMetrics(MetricMixin):
    """CogSynDelta latent space performance metrics.

    Measures the efficiency of engram encoding/decoding operations,
    which are the fundamental units of CogSynDelta's memory system.

    Attributes:
        engrams_per_sec: Number of engrams encoded per second.
        engrams_per_watt: Engram encoding efficiency (engrams/watt).
        engrams_per_joule: Energy efficiency (engrams/joule).
        encoding_fidelity: Encoding quality score (0-1).
        reconstruction_fidelity: Decoding quality score (0-1).
        latent_dim: Dimensionality of latent space.
        bits_per_dim: Information density per latent dimension.
        encode_latency_ms: Time to encode one sample.
        decode_latency_ms: Time to decode one engram.
        kl_divergence: KL divergence from prior (VAE).
        reconstruction_loss: Reconstruction error.

    Why engrams:
        Engrams are CogSynDelta's unit of memory representation - compressed
        latent vectors that encode semantic information. Tracking engrams/sec
        gives a true measure of the system's cognitive throughput.
    """

    engrams_per_sec: float | None = None
    engrams_per_watt: float | None = None
    engrams_per_joule: float | None = None
    encoding_fidelity: float | None = None
    reconstruction_fidelity: float | None = None
    latent_dim: int | None = None
    bits_per_dim: float | None = None
    encode_latency_ms: float | None = None
    decode_latency_ms: float | None = None
    kl_divergence: float | None = None
    reconstruction_loss: float | None = None

    @classmethod
    def compute(
        cls,
        num_engrams: int,
        duration_sec: float,
        power_draw_w: float | None = None,
        latent_dim: int | None = None,
        encode_time_ms: float | None = None,
        decode_time_ms: float | None = None,
        encoding_quality: float | None = None,
        reconstruction_quality: float | None = None,
        kl_div: float | None = None,
        recon_loss: float | None = None,
    ) -> LatentSpaceMetrics:
        """Compute latent space metrics from benchmark results.

        Args:
            num_engrams: Number of engrams processed.
            duration_sec: Total processing time in seconds.
            power_draw_w: Average power draw in watts (optional).
            latent_dim: Dimensionality of latent space (optional).
            encode_time_ms: Per-sample encoding time in ms (optional).
            decode_time_ms: Per-engram decoding time in ms (optional).
            encoding_quality: Encoding fidelity 0-1 (optional).
            reconstruction_quality: Reconstruction fidelity 0-1 (optional).
            kl_div: KL divergence from VAE (optional).
            recon_loss: Reconstruction loss (optional).

        Returns:
            LatentSpaceMetrics instance with computed values.

        Example:
            >>> metrics = LatentSpaceMetrics.compute(
            ...     num_engrams=10000,
            ...     duration_sec=5.0,
            ...     power_draw_w=250.0,
            ...     latent_dim=256
            ... )
            >>> print(f"Throughput: {metrics.engrams_per_sec:.0f} engrams/s")
        """
        # Throughput
        engrams_per_sec = num_engrams / duration_sec if duration_sec > 0 else None

        # Power efficiency
        engrams_per_watt = None
        engrams_per_joule = None
        if power_draw_w and power_draw_w > 0:
            engrams_per_watt = (num_engrams / duration_sec) / power_draw_w
            energy_j = power_draw_w * duration_sec
            engrams_per_joule = num_engrams / energy_j if energy_j > 0 else None

        # Information density
        bits_per_dim = None
        if latent_dim and encoding_quality:
            # Estimate bits/dim based on fidelity and dimension
            # Higher fidelity = more information preserved per dimension
            bits_per_dim = 32 * encoding_quality  # Assuming float32 representation

        return cls(
            engrams_per_sec=engrams_per_sec,
            engrams_per_watt=engrams_per_watt,
            engrams_per_joule=engrams_per_joule,
            encoding_fidelity=encoding_quality,
            reconstruction_fidelity=reconstruction_quality,
            latent_dim=latent_dim,
            bits_per_dim=bits_per_dim,
            encode_latency_ms=encode_time_ms,
            decode_latency_ms=decode_time_ms,
            kl_divergence=kl_div,
            reconstruction_loss=recon_loss,
        )

    def total_latency_ms(self) -> float | None:
        """Get total encode + decode latency.

        Returns:
            Combined latency in milliseconds, or None.
        """
        if self.encode_latency_ms is not None and self.decode_latency_ms is not None:
            return self.encode_latency_ms + self.decode_latency_ms
        return None

    def fidelity_efficiency(self) -> float | None:
        """Compute fidelity-weighted efficiency.

        Higher is better: throughput weighted by encoding quality.

        Returns:
            Fidelity-weighted engrams/sec, or None.
        """
        if self.engrams_per_sec and self.encoding_fidelity:
            return self.engrams_per_sec * self.encoding_fidelity
        return None

    def to_equivalent_tokens(
        self,
        engrams_per_token: float = 0.1,
    ) -> float | None:
        """Convert engram throughput to equivalent token throughput.

        For comparison with transformer-based systems, we estimate
        equivalent tokens based on information content ratio.

        Args:
            engrams_per_token: Ratio of engrams to tokens (default: 0.1).
                CogSynDelta engrams are more information-dense than tokens.

        Returns:
            Equivalent tokens per second, or None.
        """
        if self.engrams_per_sec:
            return self.engrams_per_sec / engrams_per_token
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatentSpaceMetrics:
        """Deserialize from dictionary.

        Args:
            data: Dictionary of metric values.

        Returns:
            LatentSpaceMetrics instance.
        """
        return cls(
            engrams_per_sec=data.get("engrams_per_sec"),
            engrams_per_watt=data.get("engrams_per_watt"),
            engrams_per_joule=data.get("engrams_per_joule"),
            encoding_fidelity=data.get("encoding_fidelity"),
            reconstruction_fidelity=data.get("reconstruction_fidelity"),
            latent_dim=data.get("latent_dim"),
            bits_per_dim=data.get("bits_per_dim"),
            encode_latency_ms=data.get("encode_latency_ms"),
            decode_latency_ms=data.get("decode_latency_ms"),
            kl_divergence=data.get("kl_divergence"),
            reconstruction_loss=data.get("reconstruction_loss"),
        )


# Register with global registry
METRIC_REGISTRY.register("latent", LatentSpaceMetrics)
