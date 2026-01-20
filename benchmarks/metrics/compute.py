"""Compute utilization metrics collection.

This module provides GPU and CPU utilization tracking including
clock speeds, memory bandwidth, and thermal throttling detection.

Classes:
    ComputeUtilizationMetrics: GPU/CPU utilization percentages and states

Why this module:
    Utilization metrics are essential for understanding whether workloads
    are compute-bound, memory-bound, or otherwise constrained. Tracking
    clock speeds helps identify thermal throttling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

try:
    import pynvml

    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import contextlib

from benchmarks.metrics.base import METRIC_REGISTRY, MetricMixin


@dataclass
class ComputeUtilizationMetrics(MetricMixin):
    """GPU and CPU utilization metrics.

    Tracks real-time utilization percentages, clock speeds, and
    memory bandwidth to understand compute bottlenecks.

    Attributes:
        gpu_utilization_pct: GPU compute utilization (0-100%).
        gpu_memory_utilization_pct: GPU memory utilization (0-100%).
        gpu_memory_used_mb: GPU memory currently used.
        gpu_memory_total_mb: Total GPU memory available.
        gpu_clock_mhz: Current GPU clock speed.
        gpu_max_clock_mhz: Maximum GPU clock speed.
        gpu_memory_clock_mhz: GPU memory clock speed.
        cpu_utilization_pct: CPU utilization (0-100%).
        cpu_freq_mhz: Current CPU frequency.
        ram_used_mb: System RAM currently used.
        ram_total_mb: Total system RAM.
        is_throttling: Whether thermal throttling is detected.
        capture_timestamp: When metrics were captured.

    Why utilization tracking:
        High GPU utilization with low memory utilization indicates
        compute-bound workloads. The inverse suggests memory-bound.
        Clock speeds below maximum indicate thermal throttling.
    """

    gpu_utilization_pct: float | None = None
    gpu_memory_utilization_pct: float | None = None
    gpu_memory_used_mb: float | None = None
    gpu_memory_total_mb: float | None = None
    gpu_clock_mhz: int | None = None
    gpu_max_clock_mhz: int | None = None
    gpu_memory_clock_mhz: int | None = None
    cpu_utilization_pct: float | None = None
    cpu_freq_mhz: float | None = None
    ram_used_mb: float | None = None
    ram_total_mb: float | None = None
    is_throttling: bool = False
    capture_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def capture(cls, gpu_index: int = 0) -> ComputeUtilizationMetrics:
        """Capture current utilization metrics from hardware.

        Reads GPU metrics via pynvml and CPU metrics via psutil.

        Args:
            gpu_index: Index of GPU to query (default: 0).

        Returns:
            ComputeUtilizationMetrics instance with current values.

        Example:
            >>> metrics = ComputeUtilizationMetrics.capture()
            >>> if metrics.is_throttling:
            ...     print("Warning: GPU is thermally throttling!")
        """
        gpu_util = None
        gpu_mem_util = None
        gpu_mem_used = None
        gpu_mem_total = None
        gpu_clock = None
        gpu_max_clock = None
        gpu_mem_clock = None
        is_throttling = False

        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

                # Utilization rates
                try:
                    rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = rates.gpu
                    gpu_mem_util = rates.memory
                except pynvml.NVMLError:
                    # GPU utilization API may not be supported on all GPUs; gracefully skip.
                    pass

                # Memory info
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_mem_used = mem_info.used / (1024 * 1024)
                    gpu_mem_total = mem_info.total / (1024 * 1024)
                except pynvml.NVMLError:
                    # Memory info API may not be available on all GPUs; gracefully skip.
                    pass

                # Clock speeds
                try:
                    gpu_clock = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_GRAPHICS
                    )
                    gpu_max_clock = pynvml.nvmlDeviceGetMaxClockInfo(
                        handle, pynvml.NVML_CLOCK_GRAPHICS
                    )
                    gpu_mem_clock = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_MEM
                    )

                    # Detect throttling: current clock < 90% of max
                    if gpu_clock and gpu_max_clock:
                        is_throttling = gpu_clock < gpu_max_clock * 0.9
                except pynvml.NVMLError:
                    # Clock speed APIs may not be supported on all GPUs; gracefully skip.
                    pass

                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                # NVML initialization may fail if driver is missing or misconfigured.
                pass

        # CPU metrics
        cpu_util = None
        cpu_freq = None
        ram_used = None
        ram_total = None

        if HAS_PSUTIL:
            with contextlib.suppress(OSError, AttributeError):
                cpu_util = psutil.cpu_percent(interval=0.1)

            with contextlib.suppress(OSError, AttributeError):
                freq = psutil.cpu_freq()
                if freq:
                    cpu_freq = freq.current

            try:
                mem = psutil.virtual_memory()
                ram_used = mem.used / (1024 * 1024)
                ram_total = mem.total / (1024 * 1024)
            except (OSError, AttributeError):
                # If memory statistics are unavailable (e.g., platform or permission issues),
                # leave RAM metrics as None; callers already handle missing metrics.
                pass

        return cls(
            gpu_utilization_pct=gpu_util,
            gpu_memory_utilization_pct=gpu_mem_util,
            gpu_memory_used_mb=gpu_mem_used,
            gpu_memory_total_mb=gpu_mem_total,
            gpu_clock_mhz=gpu_clock,
            gpu_max_clock_mhz=gpu_max_clock,
            gpu_memory_clock_mhz=gpu_mem_clock,
            cpu_utilization_pct=cpu_util,
            cpu_freq_mhz=cpu_freq,
            ram_used_mb=ram_used,
            ram_total_mb=ram_total,
            is_throttling=is_throttling,
        )

    def gpu_memory_pct(self) -> float | None:
        """Calculate GPU memory usage percentage.

        Returns:
            Percentage of GPU memory used, or None.
        """
        if self.gpu_memory_used_mb and self.gpu_memory_total_mb:
            return (self.gpu_memory_used_mb / self.gpu_memory_total_mb) * 100
        return None

    def ram_pct(self) -> float | None:
        """Calculate system RAM usage percentage.

        Returns:
            Percentage of RAM used, or None.
        """
        if self.ram_used_mb and self.ram_total_mb:
            return (self.ram_used_mb / self.ram_total_mb) * 100
        return None

    def throttle_severity(self) -> float | None:
        """Calculate throttling severity (0-1).

        Returns 0 if not throttling, 1 if severely throttled.

        Returns:
            Throttle severity score, or None if not measurable.
        """
        if self.gpu_clock_mhz and self.gpu_max_clock_mhz:
            ratio = self.gpu_clock_mhz / self.gpu_max_clock_mhz
            return max(0.0, 1.0 - ratio)
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComputeUtilizationMetrics:
        """Deserialize from dictionary.

        Args:
            data: Dictionary of metric values.

        Returns:
            ComputeUtilizationMetrics instance.
        """
        timestamp = data.get("capture_timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            gpu_utilization_pct=data.get("gpu_utilization_pct"),
            gpu_memory_utilization_pct=data.get("gpu_memory_utilization_pct"),
            gpu_memory_used_mb=data.get("gpu_memory_used_mb"),
            gpu_memory_total_mb=data.get("gpu_memory_total_mb"),
            gpu_clock_mhz=data.get("gpu_clock_mhz"),
            gpu_max_clock_mhz=data.get("gpu_max_clock_mhz"),
            gpu_memory_clock_mhz=data.get("gpu_memory_clock_mhz"),
            cpu_utilization_pct=data.get("cpu_utilization_pct"),
            cpu_freq_mhz=data.get("cpu_freq_mhz"),
            ram_used_mb=data.get("ram_used_mb"),
            ram_total_mb=data.get("ram_total_mb"),
            is_throttling=data.get("is_throttling", False),
            capture_timestamp=timestamp or datetime.now(UTC),
        )


# Register with global registry
METRIC_REGISTRY.register("compute", ComputeUtilizationMetrics)
