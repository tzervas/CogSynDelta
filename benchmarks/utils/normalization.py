"""Normalization utilities for fair model comparisons.

Provides industry-standard normalizations for comparing models of different
sizes fairly. All normalizations are explicit and documented.

Why normalization:
    Raw throughput favors smaller models. Normalizing by parameter count,
    FLOPS, or memory enables apples-to-apples comparison across architectures.
    We always show BOTH raw and normalized metrics for transparency.

Example:
    >>> from benchmarks.utils.normalization import normalize_throughput
    >>> raw = 1000  # samples/sec
    >>> norm = normalize_throughput(raw, param_count=7_000_000_000)
    >>> print(f"Raw: {raw}, Normalized: {norm.per_billion_params:.2f}/B params")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

__all__ = [
    "NormalizedThroughput",
    "NormalizedMemory",
    "NormalizedLatency",
    "normalize_throughput",
    "normalize_memory",
    "normalize_latency",
    "count_parameters",
    "estimate_flops",
]


@dataclass
class NormalizedThroughput:
    """Throughput with various normalizations.

    Attributes:
        raw: Raw samples per second.
        per_billion_params: Throughput divided by billion parameters.
        per_tflop: Throughput divided by estimated TFLOPS.
        param_count: Total parameter count used for normalization.
        estimated_tflops: Estimated TFLOPS for the operation.
    """

    raw: float
    per_billion_params: float | None
    per_tflop: float | None
    param_count: int | None
    estimated_tflops: float | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "raw_samples_per_sec": self.raw,
            "per_billion_params": self.per_billion_params,
            "per_tflop": self.per_tflop,
            "param_count": self.param_count,
            "estimated_tflops": self.estimated_tflops,
        }


@dataclass
class NormalizedMemory:
    """Memory usage with various normalizations.

    Attributes:
        raw_mb: Raw memory in megabytes.
        bytes_per_param: Memory bytes per parameter.
        mb_per_million_params: Memory MB per million parameters.
        param_count: Total parameter count.
    """

    raw_mb: float
    bytes_per_param: float | None
    mb_per_million_params: float | None
    param_count: int | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "raw_mb": self.raw_mb,
            "bytes_per_param": self.bytes_per_param,
            "mb_per_million_params": self.mb_per_million_params,
            "param_count": self.param_count,
        }


@dataclass
class NormalizedLatency:
    """Latency with various normalizations.

    Attributes:
        raw_ms: Raw latency in milliseconds.
        ms_per_billion_params: Latency normalized by billion parameters.
        tokens_per_ms: Tokens (or samples) processed per millisecond.
        param_count: Total parameter count.
    """

    raw_ms: float
    ms_per_billion_params: float | None
    tokens_per_ms: float | None
    param_count: int | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "raw_ms": self.raw_ms,
            "ms_per_billion_params": self.ms_per_billion_params,
            "tokens_per_ms": self.tokens_per_ms,
            "param_count": self.param_count,
        }


def count_parameters(model: Any, trainable_only: bool = False) -> int:
    """Count total parameters in a PyTorch model.

    Args:
        model: PyTorch model (nn.Module).
        trainable_only: If True, count only trainable parameters.

    Returns:
        Total parameter count.

    Example:
        >>> params = count_parameters(model)
        >>> print(f"Parameters: {params:,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_flops(
    model: Any,
    input_shape: tuple[int, ...],
    batch_size: int = 1,
) -> float:
    """Estimate FLOPS for a forward pass.

    This is a rough estimate based on common layer types. For accurate
    FLOPS, use profiling tools like fvcore or torch.profiler.

    Args:
        model: PyTorch model.
        input_shape: Shape of input tensor (without batch dimension).
        batch_size: Batch size for estimation.

    Returns:
        Estimated FLOPS (floating point operations).

    Note:
        This is a simplified estimation. For accurate numbers, use:
        - fvcore.nn.FlopCountAnalysis
        - torch.profiler with FLOP counting
    """
    # Rough estimation: 2 * params * batch_size for a forward pass
    # This assumes each parameter is used approximately once per sample
    # Real FLOPS depend heavily on architecture
    param_count = count_parameters(model)
    return 2.0 * param_count * batch_size


def normalize_throughput(
    samples_per_sec: float,
    param_count: int | None = None,
    estimated_tflops: float | None = None,
) -> NormalizedThroughput:
    """Normalize throughput for fair comparison.

    Args:
        samples_per_sec: Raw throughput measurement.
        param_count: Total model parameters (optional).
        estimated_tflops: Estimated TFLOPS for the operation (optional).

    Returns:
        NormalizedThroughput with raw and normalized values.

    Example:
        >>> norm = normalize_throughput(1000, param_count=7_000_000_000)
        >>> print(f"{norm.per_billion_params:.2f} samples/sec per billion params")
    """
    per_billion_params = None
    per_tflop = None

    if param_count is not None and param_count > 0:
        billion_params = param_count / 1_000_000_000
        if billion_params > 0:
            per_billion_params = samples_per_sec / billion_params

    if estimated_tflops is not None and estimated_tflops > 0:
        per_tflop = samples_per_sec / estimated_tflops

    return NormalizedThroughput(
        raw=samples_per_sec,
        per_billion_params=per_billion_params,
        per_tflop=per_tflop,
        param_count=param_count,
        estimated_tflops=estimated_tflops,
    )


def normalize_memory(
    memory_mb: float,
    param_count: int | None = None,
) -> NormalizedMemory:
    """Normalize memory usage for fair comparison.

    Args:
        memory_mb: Raw memory usage in megabytes.
        param_count: Total model parameters (optional).

    Returns:
        NormalizedMemory with raw and normalized values.

    Example:
        >>> norm = normalize_memory(1024, param_count=350_000_000)
        >>> print(f"{norm.bytes_per_param:.2f} bytes per parameter")
    """
    bytes_per_param = None
    mb_per_million_params = None

    if param_count is not None and param_count > 0:
        memory_bytes = memory_mb * 1024 * 1024
        bytes_per_param = memory_bytes / param_count

        million_params = param_count / 1_000_000
        if million_params > 0:
            mb_per_million_params = memory_mb / million_params

    return NormalizedMemory(
        raw_mb=memory_mb,
        bytes_per_param=bytes_per_param,
        mb_per_million_params=mb_per_million_params,
        param_count=param_count,
    )


def normalize_latency(
    latency_ms: float,
    param_count: int | None = None,
    tokens_per_call: int | None = None,
) -> NormalizedLatency:
    """Normalize latency for fair comparison.

    Args:
        latency_ms: Raw latency in milliseconds.
        param_count: Total model parameters (optional).
        tokens_per_call: Number of tokens processed per call (optional).

    Returns:
        NormalizedLatency with raw and normalized values.

    Example:
        >>> norm = normalize_latency(50, param_count=7_000_000_000, tokens_per_call=128)
        >>> print(f"{norm.tokens_per_ms:.1f} tokens/ms")
    """
    ms_per_billion_params = None
    tokens_per_ms = None

    if param_count is not None and param_count > 0:
        billion_params = param_count / 1_000_000_000
        if billion_params > 0:
            ms_per_billion_params = latency_ms / billion_params

    if tokens_per_call is not None and latency_ms > 0:
        tokens_per_ms = tokens_per_call / latency_ms

    return NormalizedLatency(
        raw_ms=latency_ms,
        ms_per_billion_params=ms_per_billion_params,
        tokens_per_ms=tokens_per_ms,
        param_count=param_count,
    )
