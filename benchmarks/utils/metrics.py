"""Metric computation utilities for model benchmarks.

Provides statistical validation, timing measurement, and memory tracking
per constitution requirements (100 iterations, warmup, statistical context).

Why this module:
    Centralizes metric computation to ensure consistency across all benchmarks.
    Enforces statistical rigor required by constitution: warmup iterations,
    sufficient sample size, and proper statistical reporting.

Example:
    >>> from benchmarks.utils.metrics import measure_latency, measure_memory
    >>> latency = measure_latency(model.forward, input_tensor)
    >>> print(f"p50: {latency.p50:.2f}ms, p95: {latency.p95:.2f}ms")
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

__all__ = [
    "LatencyMetrics",
    "ThroughputMetrics",
    "MemoryMetrics",
    "QualityMetrics",
    "measure_latency",
    "measure_throughput",
    "measure_memory",
    "measure_quality",
]


# Default statistical parameters per constitution
DEFAULT_WARMUP_ITERATIONS = 10
DEFAULT_TIMING_ITERATIONS = 100


@dataclass
class LatencyMetrics:
    """Latency measurements with statistical context.

    All times are in milliseconds.

    Attributes:
        mean: Mean latency across all iterations.
        std: Standard deviation of latency.
        p50: 50th percentile (median) latency.
        p95: 95th percentile latency.
        p99: 99th percentile latency.
        min: Minimum latency observed.
        max: Maximum latency observed.
        n_iterations: Number of iterations measured.
        n_warmup: Number of warmup iterations (not included in stats).
        raw_times: Raw timing data in ms.
    """

    mean: float
    std: float
    p50: float
    p95: float
    p99: float
    min: float
    max: float
    n_iterations: int
    n_warmup: int
    raw_times: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dict with all metrics (excludes raw_times for compactness).
        """
        return {
            "mean_ms": self.mean,
            "std_ms": self.std,
            "p50_ms": self.p50,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
            "min_ms": self.min,
            "max_ms": self.max,
            "n_iterations": self.n_iterations,
            "n_warmup": self.n_warmup,
        }


@dataclass
class ThroughputMetrics:
    """Throughput measurements with statistical context.

    Attributes:
        samples_per_sec: Mean samples processed per second.
        samples_per_sec_std: Standard deviation.
        batch_size: Batch size used for measurement.
        total_samples: Total samples processed.
        total_time_sec: Total time for all iterations.
        n_iterations: Number of batches processed.
    """

    samples_per_sec: float
    samples_per_sec_std: float
    batch_size: int
    total_samples: int
    total_time_sec: float
    n_iterations: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "samples_per_sec": self.samples_per_sec,
            "samples_per_sec_std": self.samples_per_sec_std,
            "batch_size": self.batch_size,
            "total_samples": self.total_samples,
            "total_time_sec": self.total_time_sec,
            "n_iterations": self.n_iterations,
        }


@dataclass
class MemoryMetrics:
    """Memory usage measurements.

    All sizes in megabytes (MB).

    Attributes:
        peak_allocated_mb: Peak GPU memory allocated during operation.
        peak_reserved_mb: Peak GPU memory reserved (including cache).
        current_allocated_mb: Current GPU memory allocated.
        model_size_mb: Estimated model parameter size.
        device: Device type (cuda, cpu, mps).
    """

    peak_allocated_mb: float
    peak_reserved_mb: float
    current_allocated_mb: float
    model_size_mb: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "peak_allocated_mb": self.peak_allocated_mb,
            "peak_reserved_mb": self.peak_reserved_mb,
            "current_allocated_mb": self.current_allocated_mb,
            "model_size_mb": self.model_size_mb,
            "device": self.device,
        }


@dataclass
class QualityMetrics:
    """Model quality measurements.

    Attributes:
        reconstruction_mse: Mean squared error for reconstruction.
        cosine_similarity: Cosine similarity for reconstruction.
        kl_divergence: KL divergence (for VAE latent space).
        custom_metrics: Additional model-specific metrics.
    """

    reconstruction_mse: float | None = None
    cosine_similarity: float | None = None
    kl_divergence: float | None = None
    custom_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {}
        if self.reconstruction_mse is not None:
            result["reconstruction_mse"] = self.reconstruction_mse
        if self.cosine_similarity is not None:
            result["cosine_similarity"] = self.cosine_similarity
        if self.kl_divergence is not None:
            result["kl_divergence"] = self.kl_divergence
        result.update(self.custom_metrics)
        return result


def _get_torch() -> Any:
    """Lazy import torch to avoid import errors when not available."""
    import torch

    return torch


def measure_latency(
    fn: Callable[[], Any],
    n_iterations: int = DEFAULT_TIMING_ITERATIONS,
    n_warmup: int = DEFAULT_WARMUP_ITERATIONS,
    sync_cuda: bool = True,
) -> LatencyMetrics:
    """Measure function latency with statistical rigor.

    Performs warmup iterations, then measures timing with proper CUDA
    synchronization (if applicable) and statistical analysis.

    Args:
        fn: Zero-argument callable to measure.
        n_iterations: Number of timed iterations.
        n_warmup: Number of warmup iterations (not timed).
        sync_cuda: Whether to synchronize CUDA before timing.

    Returns:
        LatencyMetrics with statistical analysis.

    Example:
        >>> latency = measure_latency(lambda: model(input))
        >>> print(f"Latency: {latency.p50:.2f}ms ± {latency.std:.2f}ms")
    """
    torch = _get_torch()

    # Warmup
    for _ in range(n_warmup):
        fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    # Timed iterations
    times: list[float] = []
    for _ in range(n_iterations):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        fn()

        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    times_arr = np.array(times)
    return LatencyMetrics(
        mean=float(np.mean(times_arr)),
        std=float(np.std(times_arr)),
        p50=float(np.percentile(times_arr, 50)),
        p95=float(np.percentile(times_arr, 95)),
        p99=float(np.percentile(times_arr, 99)),
        min=float(np.min(times_arr)),
        max=float(np.max(times_arr)),
        n_iterations=n_iterations,
        n_warmup=n_warmup,
        raw_times=times,
    )


def measure_throughput(
    fn: Callable[[], Any],
    batch_size: int,
    n_iterations: int = DEFAULT_TIMING_ITERATIONS,
    n_warmup: int = DEFAULT_WARMUP_ITERATIONS,
    sync_cuda: bool = True,
) -> ThroughputMetrics:
    """Measure throughput (samples per second).

    Args:
        fn: Zero-argument callable that processes batch_size samples.
        batch_size: Number of samples per call to fn.
        n_iterations: Number of batches to process.
        n_warmup: Number of warmup iterations.
        sync_cuda: Whether to synchronize CUDA.

    Returns:
        ThroughputMetrics with throughput statistics.

    Example:
        >>> throughput = measure_throughput(lambda: model(batch), batch_size=32)
        >>> print(f"Throughput: {throughput.samples_per_sec:.0f} samples/sec")
    """
    torch = _get_torch()

    # Warmup
    for _ in range(n_warmup):
        fn()
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    # Timed iterations
    throughputs: list[float] = []
    total_time = 0.0

    for _ in range(n_iterations):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        fn()

        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        total_time += elapsed
        throughputs.append(batch_size / elapsed)

    throughputs_arr = np.array(throughputs)
    return ThroughputMetrics(
        samples_per_sec=float(np.mean(throughputs_arr)),
        samples_per_sec_std=float(np.std(throughputs_arr)),
        batch_size=batch_size,
        total_samples=batch_size * n_iterations,
        total_time_sec=total_time,
        n_iterations=n_iterations,
    )


def measure_memory(
    fn: Callable[[], Any],
    model: torch.nn.Module | None = None,
    sync_cuda: bool = True,
) -> MemoryMetrics:
    """Measure memory usage during function execution.

    Resets peak memory stats, runs function, captures peak usage.

    Args:
        fn: Function to measure memory for.
        model: Optional model to calculate parameter size.
        sync_cuda: Whether to synchronize CUDA.

    Returns:
        MemoryMetrics with peak and current usage.

    Example:
        >>> memory = measure_memory(lambda: model(input), model=model)
        >>> print(f"Peak memory: {memory.peak_allocated_mb:.1f}MB")
    """
    torch = _get_torch()

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    else:
        device = "cpu"

    # Calculate model size if provided
    model_size_mb = 0.0
    if model is not None:
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        model_size_mb = param_bytes / (1024 * 1024)

    # Run function
    fn()

    if device == "cuda":
        if sync_cuda:
            torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)
        current_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        # CPU memory tracking is limited
        peak_allocated = 0.0
        peak_reserved = 0.0
        current_allocated = 0.0

    return MemoryMetrics(
        peak_allocated_mb=peak_allocated,
        peak_reserved_mb=peak_reserved,
        current_allocated_mb=current_allocated,
        model_size_mb=model_size_mb,
        device=device,
    )


def measure_quality(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    mu: torch.Tensor | None = None,
    logvar: torch.Tensor | None = None,
) -> QualityMetrics:
    """Measure reconstruction quality metrics.

    Args:
        original: Original input tensor.
        reconstructed: Reconstructed output tensor.
        mu: Optional latent mean for KL divergence.
        logvar: Optional latent log-variance for KL divergence.

    Returns:
        QualityMetrics with MSE, cosine similarity, and KL divergence.

    Example:
        >>> quality = measure_quality(input, output, mu, logvar)
        >>> print(f"Cosine similarity: {quality.cosine_similarity:.4f}")
    """
    torch = _get_torch()
    import torch.nn.functional as F

    # Flatten for metrics
    orig_flat = original.view(original.size(0), -1)
    recon_flat = reconstructed.view(reconstructed.size(0), -1)

    # MSE
    mse = F.mse_loss(recon_flat, orig_flat).item()

    # Cosine similarity (mean across batch)
    cosine = F.cosine_similarity(orig_flat, recon_flat, dim=1).mean().item()

    # KL divergence if latent params provided
    kl_div = None
    if mu is not None and logvar is not None:
        kl_div = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())).item()
        kl_div = kl_div / mu.size(0)  # Normalize by batch size

    return QualityMetrics(
        reconstruction_mse=mse,
        cosine_similarity=cosine,
        kl_divergence=kl_div,
    )
