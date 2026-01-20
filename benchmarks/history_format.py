"""Enhanced Benchmark History Format and Storage.

Provides structured, enriched benchmark data with:
- Proper scale indicators and units
- Granular tagging and metadata
- Industry-standard comparison formatting
- Trace/provenance information
- Optimized storage organization

Why this format:
    Industry benchmarks (MLPerf, Hugging Face Open LLM Leaderboard) use
    standardized formats with clear units, scales, and metadata. This
    module ensures CogSynDelta benchmarks are comparable and trustworthy.

Example:
    >>> from benchmarks.history_format import EnhancedBenchmarkRecord
    >>> record = EnhancedBenchmarkRecord.create(results, metadata)
    >>> record.save()
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "ComputeUtilizationMetrics",
    "DiagnosticMetrics",
    "EfficiencyMetrics",
    "EnhancedBenchmarkRecord",
    "LatentSpaceMetrics",
    "MetricEntry",
    "MetricScale",
    "MetricUnit",
    "ModelSizeConfig",
    "NormalizedMetrics",
    "PerformanceTrend",
    "PowerMetrics",
    "RunMetadata",
    "ScaleIndicator",
    "format_with_scale",
    "get_scale_indicator",
]

# Storage paths
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark_results"
HISTORY_DIR = BENCHMARK_DIR / "history"
ENRICHED_DIR = BENCHMARK_DIR / "enriched"


class MetricUnit(str, Enum):
    """Standard units for benchmark metrics."""

    # Time
    MILLISECONDS = "ms"
    SECONDS = "s"
    MICROSECONDS = "μs"
    NANOSECONDS = "ns"

    # Throughput
    SAMPLES_PER_SEC = "samples/s"
    IMAGES_PER_SEC = "img/s"
    TOKENS_PER_SEC = "tok/s"
    BATCHES_PER_SEC = "batch/s"

    # Latent Space / Engram throughput (CogSynDelta-specific)
    ENGRAMS_PER_SEC = "engrams/s"
    EMBEDDINGS_PER_SEC = "emb/s"
    LATENT_OPS_PER_SEC = "latent-ops/s"
    COMPRESSIONS_PER_SEC = "comp/s"
    RECONSTRUCTIONS_PER_SEC = "recon/s"

    # Memory
    BYTES = "B"
    KILOBYTES = "KB"
    MEGABYTES = "MB"
    GIGABYTES = "GB"
    TERABYTES = "TB"

    # Compute
    FLOPS = "FLOP/s"
    GFLOPS = "GFLOP/s"
    TFLOPS = "TFLOP/s"

    # Power & Energy (eco-minded metrics)
    WATTS = "W"
    KILOWATTS = "kW"
    JOULES = "J"
    KILOJOULES = "kJ"
    WATT_HOURS = "Wh"
    KILOWATT_HOURS = "kWh"

    # Efficiency metrics
    SAMPLES_PER_WATT = "samples/W"
    TOKENS_PER_WATT = "tok/W"
    FLOPS_PER_WATT = "FLOP/W"
    GFLOPS_PER_WATT = "GFLOP/W"
    SAMPLES_PER_JOULE = "samples/J"

    # Latent Space Efficiency (CogSynDelta-specific)
    ENGRAMS_PER_WATT = "engrams/W"
    ENGRAMS_PER_JOULE = "engrams/J"
    EMBEDDINGS_PER_WATT = "emb/W"
    LATENT_OPS_PER_WATT = "latent-ops/W"
    BITS_PER_DIM = "bits/dim"  # Information density

    # Compute utilization
    UTILIZATION_PCT = "util%"
    IPC = "IPC"  # Instructions per cycle
    BANDWIDTH_GBS = "GB/s"
    BANDWIDTH_PCT = "BW%"

    # Temperature
    CELSIUS = "°C"
    FAHRENHEIT = "°F"

    # Quality
    PERCENTAGE = "%"
    RATIO = "ratio"
    SCORE = "score"
    SIMILARITY = "sim"

    # Normalized / Comparative (for cross-architecture comparison)
    NORMALIZED_SCORE = "norm"  # 0-1 normalized
    RELATIVE_PERF = "rel%"  # % of baseline/reference
    SPEEDUP = "x"  # Speedup factor

    # Dimensionless
    COUNT = "count"
    UNITLESS = ""


class MetricScale(str, Enum):
    """Scale prefixes for metric values."""

    NANO = "n"      # 10^-9
    MICRO = "μ"     # 10^-6
    MILLI = "m"     # 10^-3
    UNIT = ""       # 10^0
    KILO = "K"      # 10^3
    MEGA = "M"      # 10^6
    GIGA = "G"      # 10^9
    TERA = "T"      # 10^12


@dataclass
class ScaleIndicator:
    """Visual scale indicator for a metric value.

    Provides context for interpreting raw numbers by showing:
    - The scale/magnitude of the value
    - How it compares to typical ranges
    - Visual bar representation
    """

    value: float
    unit: MetricUnit
    scale: MetricScale
    formatted: str
    bar: str
    percentile_rank: float | None = None  # Where this falls in typical range
    typical_range: tuple[float, float] | None = None
    quality_tier: str = "unknown"  # "excellent", "good", "fair", "poor"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "unit": self.unit.value,
            "scale": self.scale.value,
            "formatted": self.formatted,
            "bar": self.bar,
            "percentile_rank": self.percentile_rank,
            "typical_range": list(self.typical_range) if self.typical_range else None,
            "quality_tier": self.quality_tier,
        }


# Typical ranges for common metrics (for scale context)
TYPICAL_RANGES: dict[str, tuple[float, float, MetricUnit, bool]] = {
    # (min, max, unit, higher_is_better)
    # Performance metrics
    "latency_ms": (0.01, 1000.0, MetricUnit.MILLISECONDS, False),
    "throughput": (100, 1_000_000, MetricUnit.SAMPLES_PER_SEC, True),
    "fidelity": (0.0, 1.0, MetricUnit.RATIO, True),
    "compression_ratio": (1.0, 100.0, MetricUnit.RATIO, True),
    "memory_mb": (1.0, 32000.0, MetricUnit.MEGABYTES, False),
    "cosine_similarity": (-1.0, 1.0, MetricUnit.SIMILARITY, True),
    "gflops": (1.0, 500.0, MetricUnit.GFLOPS, True),
    "tflops": (0.1, 100.0, MetricUnit.TFLOPS, True),
    # Power & Energy (eco-minded)
    "power_watts": (10.0, 700.0, MetricUnit.WATTS, False),  # Lower power = better
    "power_draw": (10.0, 700.0, MetricUnit.WATTS, False),
    "gpu_power": (10.0, 500.0, MetricUnit.WATTS, False),
    "energy_joules": (0.1, 10000.0, MetricUnit.JOULES, False),  # Lower energy = better
    "energy_wh": (0.001, 10.0, MetricUnit.WATT_HOURS, False),
    # Efficiency metrics (higher is better = more work per energy)
    "samples_per_watt": (0.1, 1000.0, MetricUnit.SAMPLES_PER_WATT, True),
    "tokens_per_watt": (1.0, 10000.0, MetricUnit.TOKENS_PER_WATT, True),
    "gflops_per_watt": (0.1, 100.0, MetricUnit.GFLOPS_PER_WATT, True),
    "perf_per_watt": (0.1, 100.0, MetricUnit.GFLOPS_PER_WATT, True),
    # Utilization (higher = better resource usage, but watch for thermal throttling)
    "gpu_utilization": (0.0, 100.0, MetricUnit.UTILIZATION_PCT, True),
    "sm_occupancy": (0.0, 100.0, MetricUnit.PERCENTAGE, True),
    "tensor_core_util": (0.0, 100.0, MetricUnit.UTILIZATION_PCT, True),
    "memory_bandwidth": (0.0, 100.0, MetricUnit.BANDWIDTH_PCT, True),
    "pcie_bandwidth": (0.0, 64.0, MetricUnit.BANDWIDTH_GBS, True),  # PCIe 4.0 x16
    # IPC (instructions per cycle) - varies widely by workload
    "ipc": (0.1, 4.0, MetricUnit.IPC, True),  # Modern CPUs can do 4+ IPC
    # Temperature (lower is better for efficiency/longevity)
    "gpu_temp": (30.0, 95.0, MetricUnit.CELSIUS, False),
    "cpu_temp": (30.0, 100.0, MetricUnit.CELSIUS, False),
    "hotspot_temp": (40.0, 110.0, MetricUnit.CELSIUS, False),
    # ─────────────────────────────────────────────────────────────────────────
    # CogSynDelta Latent Space Metrics
    # ─────────────────────────────────────────────────────────────────────────
    "engrams_per_sec": (100.0, 1_000_000.0, MetricUnit.ENGRAMS_PER_SEC, True),
    "embeddings_per_sec": (100.0, 1_000_000.0, MetricUnit.EMBEDDINGS_PER_SEC, True),
    "latent_ops_per_sec": (1000.0, 10_000_000.0, MetricUnit.LATENT_OPS_PER_SEC, True),
    "compressions_per_sec": (10.0, 100_000.0, MetricUnit.COMPRESSIONS_PER_SEC, True),
    "reconstructions_per_sec": (10.0, 100_000.0, MetricUnit.RECONSTRUCTIONS_PER_SEC, True),
    # Latent space efficiency
    "engrams_per_watt": (1.0, 10000.0, MetricUnit.ENGRAMS_PER_WATT, True),
    "engrams_per_joule": (0.1, 1000.0, MetricUnit.ENGRAMS_PER_JOULE, True),
    "embeddings_per_watt": (1.0, 10000.0, MetricUnit.EMBEDDINGS_PER_WATT, True),
    "latent_ops_per_watt": (10.0, 100000.0, MetricUnit.LATENT_OPS_PER_WATT, True),
    # Latent space quality
    "reconstruction_fidelity": (0.0, 1.0, MetricUnit.RATIO, True),
    "latent_utilization": (0.0, 1.0, MetricUnit.RATIO, True),  # How much of latent dims used
    "information_density": (0.1, 8.0, MetricUnit.BITS_PER_DIM, True),  # bits per dimension
    # Normalized metrics (for competitor comparison)
    "normalized_throughput": (0.0, 1.0, MetricUnit.NORMALIZED_SCORE, True),
    "normalized_efficiency": (0.0, 1.0, MetricUnit.NORMALIZED_SCORE, True),
    "normalized_quality": (0.0, 1.0, MetricUnit.NORMALIZED_SCORE, True),
    "relative_perf": (0.0, 500.0, MetricUnit.RELATIVE_PERF, True),  # % of baseline
    "speedup": (0.1, 100.0, MetricUnit.SPEEDUP, True),
}


def get_scale_indicator(
    value: float,
    metric_name: str,
    unit: MetricUnit | None = None,
    typical_range: tuple[float, float] | None = None,
    higher_is_better: bool = True,
) -> ScaleIndicator:
    """Create a scale indicator for a metric value.

    Args:
        value: The metric value.
        metric_name: Name of the metric (for inferring typical ranges).
        unit: Unit of measurement.
        typical_range: Optional (min, max) typical range.
        higher_is_better: Whether higher values are better.

    Returns:
        ScaleIndicator with formatting and context.
    """
    # Infer from known metrics
    for pattern, (typ_min, typ_max, typ_unit, typ_higher) in TYPICAL_RANGES.items():
        if pattern in metric_name.lower():
            if typical_range is None:
                typical_range = (typ_min, typ_max)
            if unit is None:
                unit = typ_unit
            higher_is_better = typ_higher
            break

    unit = unit or MetricUnit.UNITLESS

    # Determine scale
    abs_val = abs(value) if value != 0 else 1
    if abs_val >= 1e12:
        scale = MetricScale.TERA
        scaled_val = value / 1e12
    elif abs_val >= 1e9:
        scale = MetricScale.GIGA
        scaled_val = value / 1e9
    elif abs_val >= 1e6:
        scale = MetricScale.MEGA
        scaled_val = value / 1e6
    elif abs_val >= 1e3:
        scale = MetricScale.KILO
        scaled_val = value / 1e3
    elif abs_val >= 1:
        scale = MetricScale.UNIT
        scaled_val = value
    elif abs_val >= 1e-3:
        scale = MetricScale.MILLI
        scaled_val = value * 1e3
    elif abs_val >= 1e-6:
        scale = MetricScale.MICRO
        scaled_val = value * 1e6
    else:
        scale = MetricScale.NANO
        scaled_val = value * 1e9

    # Format string
    if abs(scaled_val) >= 100:
        formatted = f"{scaled_val:.0f}{scale.value}{unit.value}"
    elif abs(scaled_val) >= 10:
        formatted = f"{scaled_val:.1f}{scale.value}{unit.value}"
    else:
        formatted = f"{scaled_val:.2f}{scale.value}{unit.value}"

    # Calculate percentile rank within typical range
    percentile_rank = None
    quality_tier = "unknown"
    if typical_range:
        range_min, range_max = typical_range
        if range_max > range_min:
            # Normalize to 0-1
            normalized = (value - range_min) / (range_max - range_min)
            normalized = max(0.0, min(1.0, normalized))
            percentile_rank = normalized * 100

            # Determine quality tier
            if higher_is_better:
                if normalized >= 0.9:
                    quality_tier = "excellent"
                elif normalized >= 0.7:
                    quality_tier = "good"
                elif normalized >= 0.4:
                    quality_tier = "fair"
                else:
                    quality_tier = "poor"
            elif normalized <= 0.1:
                # Lower is better (e.g., latency)
                quality_tier = "excellent"
            elif normalized <= 0.3:
                quality_tier = "good"
            elif normalized <= 0.6:
                quality_tier = "fair"
            else:
                quality_tier = "poor"

    # Create visual bar (10 chars)
    bar_width = 10
    if percentile_rank is not None:
        if higher_is_better:
            filled = int(percentile_rank / 100 * bar_width)
        else:
            filled = int((100 - percentile_rank) / 100 * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
    else:
        bar = "?" * bar_width

    return ScaleIndicator(
        value=value,
        unit=unit,
        scale=scale,
        formatted=formatted,
        bar=bar,
        percentile_rank=percentile_rank,
        typical_range=typical_range,
        quality_tier=quality_tier,
    )


def format_with_scale(value: float, metric_name: str) -> str:
    """Quick format a value with appropriate scale.

    Args:
        value: The metric value.
        metric_name: Name for context.

    Returns:
        Formatted string with scale and unit.
    """
    indicator = get_scale_indicator(value, metric_name)
    return indicator.formatted


@dataclass
class MetricEntry:
    """Single metric entry with full context.

    Attributes:
        name: Metric name (e.g., "throughput.samples_per_sec_bs32").
        value: Raw numeric value.
        unit: Unit of measurement.
        scale_indicator: Visual scale context.
        tags: Classification tags (e.g., ["throughput", "inference", "batch32"]).
        higher_is_better: Whether higher values indicate better performance.
        description: Human-readable description.
        methodology: How this metric was measured.
    """

    name: str
    value: float
    unit: MetricUnit = MetricUnit.UNITLESS
    scale_indicator: ScaleIndicator | None = None
    tags: list[str] = field(default_factory=list)
    higher_is_better: bool = True
    description: str = ""
    methodology: str = ""

    def __post_init__(self) -> None:
        """Generate scale indicator if not provided."""
        if self.scale_indicator is None:
            self.scale_indicator = get_scale_indicator(
                self.value, self.name, self.unit, higher_is_better=self.higher_is_better
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit.value,
            "scale_indicator": self.scale_indicator.to_dict() if self.scale_indicator else None,
            "tags": self.tags,
            "higher_is_better": self.higher_is_better,
            "description": self.description,
            "methodology": self.methodology,
        }


@dataclass
class PowerMetrics:
    """Power consumption and energy metrics.

    Captures eco-minded metrics for understanding the environmental
    and operational cost of compute workloads.

    Why track power:
        - Environmental responsibility (carbon footprint)
        - Operational cost estimation
        - Thermal throttling detection
        - Hardware efficiency comparison
    """

    # Instantaneous power readings
    gpu_power_watts: float | None = None
    cpu_power_watts: float | None = None
    system_power_watts: float | None = None

    # Accumulated energy over benchmark duration
    gpu_energy_joules: float | None = None
    total_energy_joules: float | None = None

    # Power limits and throttling
    gpu_power_limit_watts: float | None = None
    power_throttle_duration_ms: float | None = None

    # Temperature (affects efficiency)
    gpu_temp_celsius: float | None = None
    gpu_hotspot_celsius: float | None = None
    cpu_temp_celsius: float | None = None

    # Derived efficiency metrics (computed in post_init)
    power_efficiency_ratio: float | None = None  # % of power limit used

    def __post_init__(self) -> None:
        """Compute derived metrics."""
        if self.gpu_power_watts and self.gpu_power_limit_watts:
            self.power_efficiency_ratio = (
                self.gpu_power_watts / self.gpu_power_limit_watts * 100
            )

    @classmethod
    def capture(cls, duration_seconds: float | None = None) -> PowerMetrics:
        """Capture current power metrics from system.

        Uses pynvml for NVIDIA GPUs. Falls back gracefully if unavailable.

        Args:
            duration_seconds: Benchmark duration for energy calculation.

        Returns:
            PowerMetrics with available readings.
        """
        import contextlib

        metrics = cls()

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Power readings
            metrics.gpu_power_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

            # Power limit
            with contextlib.suppress(pynvml.NVMLError):
                metrics.gpu_power_limit_watts = (
                    pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
                )

            # Temperature
            with contextlib.suppress(pynvml.NVMLError):
                metrics.gpu_temp_celsius = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )

            # Energy calculation
            if duration_seconds and metrics.gpu_power_watts:
                metrics.gpu_energy_joules = metrics.gpu_power_watts * duration_seconds

            # Throttling info
            with contextlib.suppress(pynvml.NVMLError, AttributeError):
                throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                if throttle_reasons & (
                    pynvml.NVML_ClocksThrottleReasons.SwPowerCap
                    | pynvml.NVML_ClocksThrottleReasons.HwPowerBrakeSlowdown
                ):
                    metrics.power_throttle_duration_ms = 1.0  # Flag as throttled

            pynvml.nvmlShutdown()
        except ImportError:
            # pynvml not available - graceful degradation
            pass
        except Exception:  # noqa: S110 - Catch NVIDIA driver issues gracefully
            # NVIDIA driver may be missing or misconfigured
            pass

        return metrics

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gpu_power_watts": self.gpu_power_watts,
            "cpu_power_watts": self.cpu_power_watts,
            "system_power_watts": self.system_power_watts,
            "gpu_energy_joules": self.gpu_energy_joules,
            "total_energy_joules": self.total_energy_joules,
            "gpu_power_limit_watts": self.gpu_power_limit_watts,
            "power_throttle_duration_ms": self.power_throttle_duration_ms,
            "gpu_temp_celsius": self.gpu_temp_celsius,
            "gpu_hotspot_celsius": self.gpu_hotspot_celsius,
            "cpu_temp_celsius": self.cpu_temp_celsius,
            "power_efficiency_ratio": self.power_efficiency_ratio,
        }


@dataclass
class ComputeUtilizationMetrics:
    """GPU and CPU compute utilization metrics.

    Tracks how effectively the hardware is being utilized.

    Why track utilization:
        - Identify bottlenecks (memory-bound vs compute-bound)
        - Optimize batch sizes and kernels
        - Ensure benchmarks stress the right components
    """

    # GPU utilization
    gpu_utilization_pct: float | None = None
    gpu_memory_utilization_pct: float | None = None
    sm_occupancy_pct: float | None = None
    tensor_core_utilization_pct: float | None = None

    # Memory bandwidth
    memory_bandwidth_utilized_pct: float | None = None
    memory_bandwidth_gbs: float | None = None
    pcie_bandwidth_gbs: float | None = None

    # Clock speeds (for detecting throttling)
    gpu_clock_mhz: int | None = None
    gpu_max_clock_mhz: int | None = None
    memory_clock_mhz: int | None = None

    # CPU metrics
    cpu_utilization_pct: float | None = None
    cpu_freq_mhz: float | None = None

    @classmethod
    def capture(cls) -> ComputeUtilizationMetrics:
        """Capture current utilization metrics.

        Uses pynvml for GPU and psutil for CPU metrics.

        Returns:
            ComputeUtilizationMetrics with available readings.
        """
        import contextlib

        metrics = cls()

        # GPU metrics via pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Utilization
            with contextlib.suppress(pynvml.NVMLError):
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics.gpu_utilization_pct = util.gpu
                metrics.gpu_memory_utilization_pct = util.memory

            # Clock speeds
            with contextlib.suppress(pynvml.NVMLError):
                metrics.gpu_clock_mhz = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_SM
                )
                metrics.gpu_max_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(
                    handle, pynvml.NVML_CLOCK_SM
                )
                metrics.memory_clock_mhz = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_MEM
                )

            pynvml.nvmlShutdown()
        except ImportError:
            # pynvml not available - graceful degradation
            pass
        except Exception:  # noqa: S110 - Catch NVIDIA driver issues gracefully
            # NVIDIA driver may be missing or misconfigured
            pass

        # CPU metrics via psutil
        try:
            import psutil
            metrics.cpu_utilization_pct = psutil.cpu_percent(interval=0.1)
            freq = psutil.cpu_freq()
            if freq:
                metrics.cpu_freq_mhz = freq.current
        except ImportError:
            pass

        return metrics

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "gpu_memory_utilization_pct": self.gpu_memory_utilization_pct,
            "sm_occupancy_pct": self.sm_occupancy_pct,
            "tensor_core_utilization_pct": self.tensor_core_utilization_pct,
            "memory_bandwidth_utilized_pct": self.memory_bandwidth_utilized_pct,
            "memory_bandwidth_gbs": self.memory_bandwidth_gbs,
            "pcie_bandwidth_gbs": self.pcie_bandwidth_gbs,
            "gpu_clock_mhz": self.gpu_clock_mhz,
            "gpu_max_clock_mhz": self.gpu_max_clock_mhz,
            "memory_clock_mhz": self.memory_clock_mhz,
            "cpu_utilization_pct": self.cpu_utilization_pct,
            "cpu_freq_mhz": self.cpu_freq_mhz,
        }


@dataclass
class EfficiencyMetrics:
    """Derived efficiency metrics combining performance with power.

    These are the key eco-minded metrics that show work done per unit energy.

    Why efficiency metrics matter:
        - Compare algorithms fairly (fast but power-hungry vs slow but efficient)
        - Estimate operational costs for production
        - Support green computing initiatives
        - Enable carbon footprint estimation
    """

    # Primary efficiency metrics (higher is better)
    samples_per_watt: float | None = None
    samples_per_joule: float | None = None
    tokens_per_watt: float | None = None
    gflops_per_watt: float | None = None
    tflops_per_watt: float | None = None

    # Throughput (for reference)
    samples_per_second: float | None = None
    tokens_per_second: float | None = None
    achieved_gflops: float | None = None
    achieved_tflops: float | None = None

    # Cost estimation
    estimated_cost_per_1m_samples: float | None = None  # At $0.10/kWh
    estimated_kwh_per_1m_samples: float | None = None
    estimated_co2_kg_per_1m_samples: float | None = None  # At 0.4 kg CO2/kWh

    # Comparative efficiency
    efficiency_vs_theoretical_pct: float | None = None
    perf_per_dollar: float | None = None  # Relative metric

    @classmethod
    def compute(
        cls,
        throughput: float,
        power_watts: float,
        duration_seconds: float,
        flops: float | None = None,
        throughput_unit: str = "samples",
        electricity_rate_per_kwh: float = 0.10,
        co2_kg_per_kwh: float = 0.4,
    ) -> EfficiencyMetrics:
        """Compute efficiency metrics from benchmark results.

        Args:
            throughput: Processing rate (samples/s, tokens/s, etc.).
            power_watts: Average power consumption in watts.
            duration_seconds: Total benchmark duration.
            flops: Achieved FLOPS if known.
            throughput_unit: Unit of throughput ("samples", "tokens", etc.).
            electricity_rate_per_kwh: Cost of electricity in $/kWh.
            co2_kg_per_kwh: CO2 emissions factor in kg/kWh.

        Returns:
            EfficiencyMetrics with computed values.
        """
        metrics = cls()

        if power_watts <= 0:
            return metrics

        # Basic efficiency
        if throughput_unit == "samples":
            metrics.samples_per_second = throughput
            metrics.samples_per_watt = throughput / power_watts
        elif throughput_unit == "tokens":
            metrics.tokens_per_second = throughput
            metrics.tokens_per_watt = throughput / power_watts

        # Energy efficiency
        energy_joules = power_watts * duration_seconds
        if energy_joules > 0 and throughput_unit == "samples":
            total_samples = throughput * duration_seconds
            metrics.samples_per_joule = total_samples / energy_joules

        # FLOPS efficiency
        if flops:
            gflops = flops / 1e9
            tflops = flops / 1e12
            metrics.achieved_gflops = gflops
            metrics.achieved_tflops = tflops
            metrics.gflops_per_watt = gflops / power_watts
            metrics.tflops_per_watt = tflops / power_watts

        # Cost estimation (per 1 million samples)
        if metrics.samples_per_second and metrics.samples_per_second > 0:
            seconds_for_1m = 1_000_000 / metrics.samples_per_second
            kwh_for_1m = (power_watts * seconds_for_1m) / 3_600_000
            metrics.estimated_kwh_per_1m_samples = kwh_for_1m
            metrics.estimated_cost_per_1m_samples = kwh_for_1m * electricity_rate_per_kwh
            metrics.estimated_co2_kg_per_1m_samples = kwh_for_1m * co2_kg_per_kwh

        return metrics

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "samples_per_watt": self.samples_per_watt,
            "samples_per_joule": self.samples_per_joule,
            "tokens_per_watt": self.tokens_per_watt,
            "gflops_per_watt": self.gflops_per_watt,
            "tflops_per_watt": self.tflops_per_watt,
            "samples_per_second": self.samples_per_second,
            "tokens_per_second": self.tokens_per_second,
            "achieved_gflops": self.achieved_gflops,
            "achieved_tflops": self.achieved_tflops,
            "estimated_cost_per_1m_samples": self.estimated_cost_per_1m_samples,
            "estimated_kwh_per_1m_samples": self.estimated_kwh_per_1m_samples,
            "estimated_co2_kg_per_1m_samples": self.estimated_co2_kg_per_1m_samples,
            "efficiency_vs_theoretical_pct": self.efficiency_vs_theoretical_pct,
            "perf_per_dollar": self.perf_per_dollar,
        }


@dataclass
class LatentSpaceMetrics:
    """Metrics for latent space / engram computation.

    CogSynDelta-specific metrics for measuring performance of:
    - Memory encoding (raw → engram)
    - Memory retrieval (engram → reconstruction)
    - Compression operations
    - Latent space manipulations

    Why these metrics:
        - Engrams are the fundamental unit of memory in CogSynDelta
        - Standard throughput metrics don't capture latent space efficiency
        - Need to track both speed AND quality of latent operations
        - Enable comparison with other memory-augmented architectures
    """

    # Throughput (operations per second)
    engrams_per_sec: float | None = None  # Memories encoded per second
    embeddings_per_sec: float | None = None  # Raw embeddings computed per second
    latent_ops_per_sec: float | None = None  # Total latent space operations
    compressions_per_sec: float | None = None  # Compression operations
    reconstructions_per_sec: float | None = None  # Reconstruction operations

    # Quality metrics
    encoding_fidelity: float | None = None  # How well input is preserved (0-1)
    reconstruction_fidelity: float | None = None  # Reconstruction quality (0-1)
    latent_utilization: float | None = None  # % of latent dimensions used effectively
    information_retention: float | None = None  # Bits preserved / bits input

    # Efficiency (work per energy)
    engrams_per_watt: float | None = None
    engrams_per_joule: float | None = None
    embeddings_per_watt: float | None = None
    latent_ops_per_watt: float | None = None

    # Latent space characteristics
    latent_dim: int | None = None  # Dimensionality of latent space
    effective_dim: float | None = None  # Effective dimensionality (PCA-based)
    information_density_bits_per_dim: float | None = None
    sparsity: float | None = None  # % of near-zero activations

    # Timing breakdown (for debugging/tuning)
    encode_latency_ms: float | None = None
    decode_latency_ms: float | None = None
    compress_latency_ms: float | None = None
    retrieve_latency_ms: float | None = None

    @classmethod
    def compute(
        cls,
        num_engrams: int,
        duration_seconds: float,
        power_watts: float | None = None,
        encoding_fidelity: float | None = None,
        reconstruction_fidelity: float | None = None,
        latent_dim: int | None = None,
        encode_time_ms: float | None = None,
        decode_time_ms: float | None = None,
    ) -> LatentSpaceMetrics:
        """Compute latent space metrics from benchmark results.

        Args:
            num_engrams: Number of engrams processed.
            duration_seconds: Total benchmark duration.
            power_watts: Average power consumption.
            encoding_fidelity: Encoding quality (0-1).
            reconstruction_fidelity: Reconstruction quality (0-1).
            latent_dim: Latent space dimensionality.
            encode_time_ms: Encoding latency in ms.
            decode_time_ms: Decoding latency in ms.

        Returns:
            LatentSpaceMetrics with computed values.
        """
        metrics = cls()

        if duration_seconds > 0:
            metrics.engrams_per_sec = num_engrams / duration_seconds
            metrics.latent_ops_per_sec = num_engrams * 2 / duration_seconds  # encode + decode

        if power_watts and power_watts > 0 and metrics.engrams_per_sec:
            metrics.engrams_per_watt = metrics.engrams_per_sec / power_watts
            energy_joules = power_watts * duration_seconds
            if energy_joules > 0:
                metrics.engrams_per_joule = num_engrams / energy_joules

        metrics.encoding_fidelity = encoding_fidelity
        metrics.reconstruction_fidelity = reconstruction_fidelity
        metrics.latent_dim = latent_dim
        metrics.encode_latency_ms = encode_time_ms
        metrics.decode_latency_ms = decode_time_ms

        return metrics

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "engrams_per_sec": self.engrams_per_sec,
            "embeddings_per_sec": self.embeddings_per_sec,
            "latent_ops_per_sec": self.latent_ops_per_sec,
            "compressions_per_sec": self.compressions_per_sec,
            "reconstructions_per_sec": self.reconstructions_per_sec,
            "encoding_fidelity": self.encoding_fidelity,
            "reconstruction_fidelity": self.reconstruction_fidelity,
            "latent_utilization": self.latent_utilization,
            "information_retention": self.information_retention,
            "engrams_per_watt": self.engrams_per_watt,
            "engrams_per_joule": self.engrams_per_joule,
            "embeddings_per_watt": self.embeddings_per_watt,
            "latent_ops_per_watt": self.latent_ops_per_watt,
            "latent_dim": self.latent_dim,
            "effective_dim": self.effective_dim,
            "information_density_bits_per_dim": self.information_density_bits_per_dim,
            "sparsity": self.sparsity,
            "encode_latency_ms": self.encode_latency_ms,
            "decode_latency_ms": self.decode_latency_ms,
            "compress_latency_ms": self.compress_latency_ms,
            "retrieve_latency_ms": self.retrieve_latency_ms,
        }


@dataclass
class NormalizedMetrics:
    """Normalized metrics for cross-architecture comparison.

    Provides apples-to-apples comparison with competitor architectures
    by normalizing CogSynDelta-specific metrics to standard benchmarks.

    Why normalization:
        - CogSynDelta uses engrams; competitors use tokens, embeddings, etc.
        - Different architectures have different memory models
        - Need common ground for fair comparison
        - Enable tracking against industry baselines

    Normalization strategies:
        1. Per-parameter: Metric / total_parameters
        2. Per-FLOP: Metric / compute_flops
        3. Per-memory: Metric / memory_footprint
        4. Relative to baseline: Metric / baseline_metric
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Throughput (normalized to 0-1 or relative to baseline)
    # ─────────────────────────────────────────────────────────────────────────
    normalized_throughput: float | None = None  # 0-1 against theoretical max
    throughput_per_param: float | None = None  # samples/s per million params
    throughput_per_flop: float | None = None  # samples per GFLOP
    relative_throughput: float | None = None  # % of baseline system

    # ─────────────────────────────────────────────────────────────────────────
    # Efficiency (normalized)
    # ─────────────────────────────────────────────────────────────────────────
    normalized_efficiency: float | None = None  # 0-1 against theoretical max
    efficiency_per_param: float | None = None  # samples/W per million params
    perf_per_watt_normalized: float | None = None  # Normalized to reference GPU

    # ─────────────────────────────────────────────────────────────────────────
    # Quality (normalized)
    # ─────────────────────────────────────────────────────────────────────────
    normalized_quality: float | None = None  # 0-1 composite quality score
    fidelity_compression_ratio: float | None = None  # fidelity / compression
    quality_efficiency_score: float | None = None  # quality * efficiency

    # ─────────────────────────────────────────────────────────────────────────
    # Latent space (normalized to standard embedding metrics)
    # ─────────────────────────────────────────────────────────────────────────
    # Maps engram metrics to token/embedding equivalents
    equivalent_tokens_per_sec: float | None = None  # Engrams → token equivalent
    equivalent_embeddings_per_sec: float | None = None  # Standardized embedding rate
    latent_efficiency_score: float | None = None  # Composite latent efficiency

    # ─────────────────────────────────────────────────────────────────────────
    # Comparison baselines
    # ─────────────────────────────────────────────────────────────────────────
    baseline_model: str | None = None  # Reference model (e.g., "BERT-base")
    baseline_throughput: float | None = None
    baseline_efficiency: float | None = None
    speedup_vs_baseline: float | None = None  # x times faster
    efficiency_vs_baseline: float | None = None  # x times more efficient

    @classmethod
    def compute(
        cls,
        raw_throughput: float,
        raw_efficiency: float | None = None,
        raw_quality: float | None = None,
        total_params: int | None = None,
        total_flops: float | None = None,
        engrams_per_sec: float | None = None,
        baseline_model: str = "reference",
        baseline_throughput: float = 1000.0,
        baseline_efficiency: float = 10.0,
        theoretical_max_throughput: float | None = None,
    ) -> NormalizedMetrics:
        """Compute normalized metrics for cross-architecture comparison.

        Args:
            raw_throughput: Raw throughput (samples/s, tokens/s, etc.).
            raw_efficiency: Raw efficiency (samples/W, etc.).
            raw_quality: Raw quality score (0-1).
            total_params: Total model parameters.
            total_flops: Total FLOPS for one forward pass.
            engrams_per_sec: CogSynDelta-specific engram throughput.
            baseline_model: Name of reference model.
            baseline_throughput: Baseline throughput for comparison.
            baseline_efficiency: Baseline efficiency for comparison.
            theoretical_max_throughput: Theoretical maximum for normalization.

        Returns:
            NormalizedMetrics with computed values.
        """
        metrics = cls()
        metrics.baseline_model = baseline_model
        metrics.baseline_throughput = baseline_throughput
        metrics.baseline_efficiency = baseline_efficiency

        # Throughput normalization
        if theoretical_max_throughput and theoretical_max_throughput > 0:
            metrics.normalized_throughput = min(1.0, raw_throughput / theoretical_max_throughput)

        if total_params and total_params > 0:
            metrics.throughput_per_param = raw_throughput / (total_params / 1_000_000)

        if total_flops and total_flops > 0:
            metrics.throughput_per_flop = raw_throughput / (total_flops / 1e9)

        if baseline_throughput > 0:
            metrics.relative_throughput = (raw_throughput / baseline_throughput) * 100
            metrics.speedup_vs_baseline = raw_throughput / baseline_throughput

        # Efficiency normalization
        if raw_efficiency:
            if baseline_efficiency > 0:
                metrics.efficiency_vs_baseline = raw_efficiency / baseline_efficiency
            if total_params and total_params > 0:
                metrics.efficiency_per_param = raw_efficiency / (total_params / 1_000_000)

        # Quality normalization
        if raw_quality is not None:
            metrics.normalized_quality = raw_quality  # Already 0-1
            if raw_efficiency:
                metrics.quality_efficiency_score = raw_quality * raw_efficiency

        # Latent space normalization (engrams → equivalent metrics)
        if engrams_per_sec:
            # Heuristic: 1 engram ≈ 100 tokens worth of semantic content
            metrics.equivalent_tokens_per_sec = engrams_per_sec * 100
            metrics.equivalent_embeddings_per_sec = engrams_per_sec

        return metrics

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "normalized_throughput": self.normalized_throughput,
            "throughput_per_param": self.throughput_per_param,
            "throughput_per_flop": self.throughput_per_flop,
            "relative_throughput": self.relative_throughput,
            "normalized_efficiency": self.normalized_efficiency,
            "efficiency_per_param": self.efficiency_per_param,
            "perf_per_watt_normalized": self.perf_per_watt_normalized,
            "normalized_quality": self.normalized_quality,
            "fidelity_compression_ratio": self.fidelity_compression_ratio,
            "quality_efficiency_score": self.quality_efficiency_score,
            "equivalent_tokens_per_sec": self.equivalent_tokens_per_sec,
            "equivalent_embeddings_per_sec": self.equivalent_embeddings_per_sec,
            "latent_efficiency_score": self.latent_efficiency_score,
            "baseline_model": self.baseline_model,
            "baseline_throughput": self.baseline_throughput,
            "baseline_efficiency": self.baseline_efficiency,
            "speedup_vs_baseline": self.speedup_vs_baseline,
            "efficiency_vs_baseline": self.efficiency_vs_baseline,
        }


@dataclass
class DiagnosticMetrics:
    """Diagnostic metrics for understanding WHY performance is what it is.

    These metrics help explain bottlenecks, inefficiencies, and optimization
    opportunities. Structured for natural debugging and tuning workflows.

    Why diagnostic metrics:
        - Raw numbers don't explain causation
        - Need to identify bottlenecks quickly
        - Support data-driven optimization
        - Enable root cause analysis
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Bottleneck Analysis
    # ─────────────────────────────────────────────────────────────────────────
    primary_bottleneck: str | None = None  # "compute", "memory", "io", "latency"
    bottleneck_severity: float | None = None  # 0-1, how much it limits perf
    bottleneck_explanation: str | None = None  # Human-readable explanation

    # ─────────────────────────────────────────────────────────────────────────
    # Compute Analysis
    # ─────────────────────────────────────────────────────────────────────────
    compute_bound: bool | None = None  # Is compute the limiting factor?
    compute_utilization_pct: float | None = None
    theoretical_peak_flops: float | None = None
    achieved_flops: float | None = None
    compute_efficiency_pct: float | None = None  # achieved / theoretical

    # ─────────────────────────────────────────────────────────────────────────
    # Memory Analysis
    # ─────────────────────────────────────────────────────────────────────────
    memory_bound: bool | None = None  # Is memory the limiting factor?
    memory_utilization_pct: float | None = None
    memory_bandwidth_achieved_gbs: float | None = None
    memory_bandwidth_peak_gbs: float | None = None
    memory_efficiency_pct: float | None = None
    cache_hit_rate: float | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # I/O Analysis
    # ─────────────────────────────────────────────────────────────────────────
    io_bound: bool | None = None
    data_loading_time_pct: float | None = None  # % of time in data loading
    pcie_utilization_pct: float | None = None
    disk_wait_time_pct: float | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Latency Analysis
    # ─────────────────────────────────────────────────────────────────────────
    latency_bound: bool | None = None
    kernel_launch_overhead_pct: float | None = None
    synchronization_overhead_pct: float | None = None
    python_overhead_pct: float | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # Optimization Recommendations
    # ─────────────────────────────────────────────────────────────────────────
    recommendations: list[str] = field(default_factory=list)
    optimization_potential_pct: float | None = None  # How much improvement possible
    suggested_batch_size: int | None = None
    suggested_precision: str | None = None  # "fp32", "fp16", "bf16", "int8"

    @classmethod
    def analyze(
        cls,
        compute_util: float | None = None,
        memory_util: float | None = None,
        io_wait_pct: float | None = None,
        kernel_overhead_pct: float | None = None,
        achieved_flops: float | None = None,
        theoretical_flops: float | None = None,
        memory_bw_achieved: float | None = None,
        memory_bw_peak: float | None = None,
    ) -> DiagnosticMetrics:
        """Analyze metrics to identify bottlenecks and generate recommendations.

        Args:
            compute_util: GPU compute utilization (0-100).
            memory_util: Memory bandwidth utilization (0-100).
            io_wait_pct: Percentage of time waiting on I/O.
            kernel_overhead_pct: Kernel launch overhead percentage.
            achieved_flops: Achieved FLOPS.
            theoretical_flops: Theoretical peak FLOPS.
            memory_bw_achieved: Achieved memory bandwidth (GB/s).
            memory_bw_peak: Peak memory bandwidth (GB/s).

        Returns:
            DiagnosticMetrics with analysis and recommendations.
        """
        metrics = cls()
        recommendations: list[str] = []

        # Store raw values
        metrics.compute_utilization_pct = compute_util
        metrics.memory_utilization_pct = memory_util
        metrics.achieved_flops = achieved_flops
        metrics.theoretical_peak_flops = theoretical_flops
        metrics.memory_bandwidth_achieved_gbs = memory_bw_achieved
        metrics.memory_bandwidth_peak_gbs = memory_bw_peak

        # Compute efficiency
        if achieved_flops and theoretical_flops and theoretical_flops > 0:
            metrics.compute_efficiency_pct = (achieved_flops / theoretical_flops) * 100

        # Memory efficiency
        if memory_bw_achieved and memory_bw_peak and memory_bw_peak > 0:
            metrics.memory_efficiency_pct = (memory_bw_achieved / memory_bw_peak) * 100

        # Identify bottleneck
        bottleneck_scores = {
            "compute": compute_util or 0,
            "memory": memory_util or 0,
            "io": (io_wait_pct or 0) * 100,
            "latency": (kernel_overhead_pct or 0) * 100,
        }

        # The bottleneck is the resource that's most utilized
        primary = max(bottleneck_scores, key=bottleneck_scores.get)
        metrics.primary_bottleneck = primary
        metrics.bottleneck_severity = bottleneck_scores[primary] / 100

        # Set bound flags
        metrics.compute_bound = primary == "compute"
        metrics.memory_bound = primary == "memory"
        metrics.io_bound = primary == "io"
        metrics.latency_bound = primary == "latency"

        # Generate explanation
        if primary == "compute":
            metrics.bottleneck_explanation = (
                f"Compute-bound: GPU SMs are {compute_util:.1f}% utilized. "
                "Performance scales with faster GPU or lower precision."
            )
            if compute_util and compute_util < 80:
                recommendations.append("Consider using larger batch sizes to improve SM utilization")
            if metrics.compute_efficiency_pct and metrics.compute_efficiency_pct < 50:
                recommendations.append("Consider using tensor cores (fp16/bf16) for better efficiency")
        elif primary == "memory":
            metrics.bottleneck_explanation = (
                f"Memory-bound: Memory bandwidth is {memory_util:.1f}% utilized. "
                "Performance scales with HBM bandwidth or better data locality."
            )
            recommendations.append("Consider using memory-efficient attention or gradient checkpointing")
            recommendations.append("Review data layout for coalesced memory access")
        elif primary == "io":
            metrics.bottleneck_explanation = (
                f"I/O-bound: {io_wait_pct:.1f}% of time spent waiting on data. "
                "Performance scales with faster storage or prefetching."
            )
            recommendations.append("Enable data prefetching with multiple workers")
            recommendations.append("Consider caching dataset in memory or using faster storage")
        elif primary == "latency":
            metrics.bottleneck_explanation = (
                f"Latency-bound: {kernel_overhead_pct:.1f}% overhead from kernel launches. "
                "Performance scales with CUDA graphs or kernel fusion."
            )
            recommendations.append("Consider using CUDA graphs for reduced launch overhead")
            recommendations.append("Look into torch.compile() for kernel fusion")

        # Estimate optimization potential
        max_util = max(bottleneck_scores.values())
        if max_util > 0:
            metrics.optimization_potential_pct = ((100 - max_util) / 100) * 50  # Conservative

        metrics.recommendations = recommendations
        return metrics

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_bottleneck": self.primary_bottleneck,
            "bottleneck_severity": self.bottleneck_severity,
            "bottleneck_explanation": self.bottleneck_explanation,
            "compute_bound": self.compute_bound,
            "compute_utilization_pct": self.compute_utilization_pct,
            "theoretical_peak_flops": self.theoretical_peak_flops,
            "achieved_flops": self.achieved_flops,
            "compute_efficiency_pct": self.compute_efficiency_pct,
            "memory_bound": self.memory_bound,
            "memory_utilization_pct": self.memory_utilization_pct,
            "memory_bandwidth_achieved_gbs": self.memory_bandwidth_achieved_gbs,
            "memory_bandwidth_peak_gbs": self.memory_bandwidth_peak_gbs,
            "memory_efficiency_pct": self.memory_efficiency_pct,
            "cache_hit_rate": self.cache_hit_rate,
            "io_bound": self.io_bound,
            "data_loading_time_pct": self.data_loading_time_pct,
            "pcie_utilization_pct": self.pcie_utilization_pct,
            "disk_wait_time_pct": self.disk_wait_time_pct,
            "latency_bound": self.latency_bound,
            "kernel_launch_overhead_pct": self.kernel_launch_overhead_pct,
            "synchronization_overhead_pct": self.synchronization_overhead_pct,
            "python_overhead_pct": self.python_overhead_pct,
            "recommendations": self.recommendations,
            "optimization_potential_pct": self.optimization_potential_pct,
            "suggested_batch_size": self.suggested_batch_size,
            "suggested_precision": self.suggested_precision,
        }


@dataclass
class PerformanceTrend:
    """Track performance trends over time with deltas.

    Captures how metrics change between benchmark runs.
    """

    metric_name: str
    current_value: float
    previous_value: float | None = None
    baseline_value: float | None = None

    # Deltas
    delta_from_previous: float | None = None
    delta_pct_from_previous: float | None = None
    delta_from_baseline: float | None = None
    delta_pct_from_baseline: float | None = None

    # Trend analysis
    trend_direction: str = "stable"  # "improving", "degrading", "stable"
    is_regression: bool = False
    regression_threshold_pct: float = 5.0

    def __post_init__(self) -> None:
        """Compute deltas and trend."""
        if self.previous_value is not None:
            self.delta_from_previous = self.current_value - self.previous_value
            if self.previous_value != 0:
                self.delta_pct_from_previous = (
                    self.delta_from_previous / abs(self.previous_value) * 100
                )

        if self.baseline_value is not None:
            self.delta_from_baseline = self.current_value - self.baseline_value
            if self.baseline_value != 0:
                self.delta_pct_from_baseline = (
                    self.delta_from_baseline / abs(self.baseline_value) * 100
                )

        # Determine trend (assumes higher is better by default)
        if self.delta_pct_from_previous is not None:
            if self.delta_pct_from_previous > 2.0:
                self.trend_direction = "improving"
            elif self.delta_pct_from_previous < -2.0:
                self.trend_direction = "degrading"
                if abs(self.delta_pct_from_previous) > self.regression_threshold_pct:
                    self.is_regression = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "previous_value": self.previous_value,
            "baseline_value": self.baseline_value,
            "delta_from_previous": self.delta_from_previous,
            "delta_pct_from_previous": self.delta_pct_from_previous,
            "delta_from_baseline": self.delta_from_baseline,
            "delta_pct_from_baseline": self.delta_pct_from_baseline,
            "trend_direction": self.trend_direction,
            "is_regression": self.is_regression,
        }


@dataclass
class ModelSizeConfig:
    """Configuration for different model sizes.

    Supports cycling through model variants for comprehensive benchmarking.
    """

    name: str  # e.g., "small", "base", "large"
    parameters: int  # Total parameter count
    embed_dim: int
    num_layers: int
    hidden_dim: int
    description: str = ""

    # Standard sizes for CogSynDelta architecture
    SIZES: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "tiny": {"parameters": 1_000_000, "embed_dim": 128, "num_layers": 2, "hidden_dim": 256},
        "small": {"parameters": 10_000_000, "embed_dim": 256, "num_layers": 4, "hidden_dim": 512},
        "base": {"parameters": 50_000_000, "embed_dim": 512, "num_layers": 8, "hidden_dim": 1024},
        "large": {"parameters": 200_000_000, "embed_dim": 768, "num_layers": 12, "hidden_dim": 2048},
        "xlarge": {"parameters": 1_000_000_000, "embed_dim": 1024, "num_layers": 24, "hidden_dim": 4096},
    })

    @classmethod
    def get_config(cls, size: str) -> ModelSizeConfig:
        """Get configuration for a named size."""
        if size not in cls.SIZES:
            available = list(cls.SIZES.keys())
            raise ValueError(f"Unknown size '{size}'. Available: {available}")

        config = cls.SIZES[size]
        return cls(
            name=size,
            parameters=config["parameters"],
            embed_dim=config["embed_dim"],
            num_layers=config["num_layers"],
            hidden_dim=config["hidden_dim"],
            description=f"CogSynDelta {size} configuration",
        )

    @classmethod
    def all_sizes(cls) -> list[str]:
        """Get all available size names."""
        return list(cls.SIZES.keys())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "parameters": self.parameters,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "description": self.description,
        }


@dataclass
class RunMetadata:
    """Complete metadata for a benchmark run.

    Captures everything needed for reproducibility and comparison.
    """

    # Identification
    run_id: str
    timestamp: str
    git_sha: str
    git_branch: str
    git_dirty: bool

    # Environment
    platform: str
    python_version: str
    pytorch_version: str
    cuda_version: str | None
    gpu_name: str | None
    gpu_memory_gb: float | None
    cpu_model: str
    ram_gb: float

    # Configuration
    model_size: ModelSizeConfig | None
    batch_sizes: list[int]
    num_warmup: int
    num_iterations: int
    seed: int

    # Provenance
    command_line: str
    environment_hash: str  # Hash of key env variables
    config_hash: str  # Hash of benchmark config

    @classmethod
    def capture(
        cls,
        model_size: ModelSizeConfig | None = None,
        batch_sizes: list[int] | None = None,
        num_warmup: int = 3,
        num_iterations: int = 10,
        seed: int = 42,
    ) -> RunMetadata:
        """Capture current environment metadata.

        Args:
            model_size: Model configuration being tested.
            batch_sizes: Batch sizes used.
            num_warmup: Warmup iterations.
            num_iterations: Timed iterations.
            seed: Random seed.

        Returns:
            RunMetadata with all captured information.
        """
        import shutil
        import sys

        # Git info
        git_sha = "unknown"
        git_branch = "unknown"
        git_dirty = False
        git_path = shutil.which("git")  # Resolve full path for security (S607)
        if git_path:
            try:
                git_sha = subprocess.check_output(
                    [git_path, "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                git_branch = subprocess.check_output(
                    [git_path, "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                git_status = subprocess.check_output(
                    [git_path, "status", "--porcelain"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                git_dirty = len(git_status) > 0
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass

        # GPU info
        gpu_name = None
        gpu_memory_gb = None
        cuda_version = None
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            cuda_version = torch.version.cuda

        # CPU info
        cpu_model = platform.processor() or "unknown"

        # RAM
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            ram_gb = 0.0

        # Generate hashes for reproducibility (sha256, truncated for brevity)
        env_str = f"{platform.platform()}|{sys.version}|{torch.__version__}|{cuda_version}"
        env_hash = hashlib.sha256(env_str.encode()).hexdigest()[:8]

        config_str = f"{model_size}|{batch_sizes}|{num_warmup}|{num_iterations}|{seed}"
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]

        # Run ID
        timestamp = datetime.now(UTC).isoformat()
        run_id = f"run_{timestamp.replace(':', '-').replace('.', '-')}_{git_sha}_{env_hash}"

        return cls(
            run_id=run_id,
            timestamp=timestamp,
            git_sha=git_sha,
            git_branch=git_branch,
            git_dirty=git_dirty,
            platform=platform.platform(),
            python_version=sys.version.split()[0],
            pytorch_version=torch.__version__,
            cuda_version=cuda_version,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            cpu_model=cpu_model,
            ram_gb=ram_gb,
            model_size=model_size,
            batch_sizes=batch_sizes or [],
            num_warmup=num_warmup,
            num_iterations=num_iterations,
            seed=seed,
            command_line=" ".join(sys.argv),
            environment_hash=env_hash,
            config_hash=config_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "git_sha": self.git_sha,
            "git_branch": self.git_branch,
            "git_dirty": self.git_dirty,
            "platform": self.platform,
            "python_version": self.python_version,
            "pytorch_version": self.pytorch_version,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": self.gpu_memory_gb,
            "cpu_model": self.cpu_model,
            "ram_gb": self.ram_gb,
            "model_size": self.model_size.to_dict() if self.model_size else None,
            "batch_sizes": self.batch_sizes,
            "num_warmup": self.num_warmup,
            "num_iterations": self.num_iterations,
            "seed": self.seed,
            "command_line": self.command_line,
            "environment_hash": self.environment_hash,
            "config_hash": self.config_hash,
        }


@dataclass
class EnhancedBenchmarkRecord:
    """Complete enhanced benchmark record with all context.

    This is the primary storage format for benchmark history.
    Includes power, efficiency, utilization, latent space, and diagnostic metrics.
    Supports both CogSynDelta-specific and normalized cross-architecture comparison.
    """

    # Metadata
    metadata: RunMetadata

    # Metrics organized by component
    components: dict[str, list[MetricEntry]]

    # Summary statistics
    summary: dict[str, Any]

    # Power & efficiency metrics (eco-minded)
    power_metrics: PowerMetrics | None = None
    efficiency_metrics: EfficiencyMetrics | None = None
    utilization_metrics: ComputeUtilizationMetrics | None = None

    # CogSynDelta-specific latent space metrics
    latent_metrics: LatentSpaceMetrics | None = None

    # Normalized metrics for cross-architecture comparison
    normalized_metrics: NormalizedMetrics | None = None

    # Diagnostic metrics for debugging/tuning
    diagnostic_metrics: DiagnosticMetrics | None = None

    # Performance trends (deltas from previous runs)
    trends: list[PerformanceTrend] = field(default_factory=list)

    # Industry comparison context
    industry_context: dict[str, Any] | None = None

    # Raw results (for backward compatibility)
    raw_results: dict[str, Any] | None = None

    @classmethod
    def from_benchmark_results(
        cls,
        results: dict[str, Any],
        model_size: ModelSizeConfig | None = None,
        duration_seconds: float | None = None,
        throughput: float | None = None,
        throughput_unit: str = "samples",
        capture_power: bool = True,
        previous_record: EnhancedBenchmarkRecord | None = None,
        # Latent space parameters
        num_engrams: int | None = None,
        encoding_fidelity: float | None = None,
        reconstruction_fidelity: float | None = None,
        latent_dim: int | None = None,
        # Normalization parameters
        baseline_model: str = "reference",
        baseline_throughput: float = 1000.0,
        theoretical_max_throughput: float | None = None,
    ) -> EnhancedBenchmarkRecord:
        """Create enhanced record from standard benchmark results.

        Args:
            results: Standard benchmark results dict.
            model_size: Optional model size configuration.
            duration_seconds: Benchmark duration for energy calculations.
            throughput: Throughput for efficiency calculations.
            throughput_unit: Unit of throughput ("samples", "tokens", "engrams").
            capture_power: Whether to capture power metrics.
            previous_record: Previous record for trend analysis.
            num_engrams: Number of engrams processed (for latent metrics).
            encoding_fidelity: Encoding quality (0-1).
            reconstruction_fidelity: Reconstruction quality (0-1).
            latent_dim: Latent space dimensionality.
            baseline_model: Name of baseline for comparison.
            baseline_throughput: Baseline throughput for normalization.
            theoretical_max_throughput: Theoretical max for normalization.

        Returns:
            EnhancedBenchmarkRecord with full enrichment.
        """
        # Capture metadata
        metadata = RunMetadata.capture(model_size=model_size)

        # Capture power and utilization metrics
        power_metrics = None
        utilization_metrics = None
        efficiency_metrics = None
        latent_metrics = None
        normalized_metrics = None
        diagnostic_metrics = None

        if capture_power:
            power_metrics = PowerMetrics.capture(duration_seconds)
            utilization_metrics = ComputeUtilizationMetrics.capture()

            # Compute efficiency if we have throughput and power
            if throughput and power_metrics.gpu_power_watts:
                efficiency_metrics = EfficiencyMetrics.compute(
                    throughput=throughput,
                    power_watts=power_metrics.gpu_power_watts,
                    duration_seconds=duration_seconds or 1.0,
                    throughput_unit=throughput_unit,
                )

        # Convert components to MetricEntry format
        components: dict[str, list[MetricEntry]] = {}

        for comp_name, comp_data in results.get("components", {}).items():
            entries: list[MetricEntry] = []

            for category_name, category_data in comp_data.items():
                if not isinstance(category_data, dict):
                    continue

                for metric_name, value in category_data.items():
                    if not isinstance(value, (int, float)):
                        continue

                    # Determine tags and properties
                    tags = [comp_name, category_name]
                    higher_is_better = True
                    unit = MetricUnit.UNITLESS

                    if "latency" in metric_name.lower():
                        higher_is_better = False
                        unit = MetricUnit.MILLISECONDS
                        tags.append("latency")
                    elif "throughput" in metric_name.lower() or "per_sec" in metric_name.lower():
                        unit = MetricUnit.SAMPLES_PER_SEC
                        tags.append("throughput")
                    elif "memory" in metric_name.lower():
                        higher_is_better = False
                        unit = MetricUnit.MEGABYTES
                        tags.append("memory")
                    elif "fidelity" in metric_name.lower() or "similarity" in metric_name.lower():
                        unit = MetricUnit.RATIO
                        tags.append("quality")
                    elif "power" in metric_name.lower() or "watt" in metric_name.lower():
                        higher_is_better = False
                        unit = MetricUnit.WATTS
                        tags.append("power")
                    elif "efficiency" in metric_name.lower() or "per_watt" in metric_name.lower():
                        unit = MetricUnit.SAMPLES_PER_WATT
                        tags.append("efficiency")

                    full_name = f"{comp_name}.{category_name}.{metric_name}"
                    entries.append(
                        MetricEntry(
                            name=full_name,
                            value=float(value),
                            unit=unit,
                            tags=tags,
                            higher_is_better=higher_is_better,
                        )
                    )

            components[comp_name] = entries

        # Compute trends if previous record provided
        trends: list[PerformanceTrend] = []
        if previous_record:
            # Create metric name -> value map from previous
            prev_values: dict[str, float] = {}
            for entries in previous_record.components.values():
                for entry in entries:
                    prev_values[entry.name] = entry.value

            # Compare with current
            for entries in components.values():
                for entry in entries:
                    if entry.name in prev_values:
                        trends.append(
                            PerformanceTrend(
                                metric_name=entry.name,
                                current_value=entry.value,
                                previous_value=prev_values[entry.name],
                            )
                        )

        # Compute latent space metrics if engram data provided
        if num_engrams and duration_seconds:
            latent_metrics = LatentSpaceMetrics.compute(
                num_engrams=num_engrams,
                duration_seconds=duration_seconds,
                power_watts=power_metrics.gpu_power_watts if power_metrics else None,
                encoding_fidelity=encoding_fidelity,
                reconstruction_fidelity=reconstruction_fidelity,
                latent_dim=latent_dim,
            )

        # Compute normalized metrics for cross-architecture comparison
        if throughput:
            total_params = model_size.parameters if model_size else None
            normalized_metrics = NormalizedMetrics.compute(
                raw_throughput=throughput,
                raw_efficiency=efficiency_metrics.samples_per_watt if efficiency_metrics else None,
                raw_quality=encoding_fidelity or reconstruction_fidelity,
                total_params=total_params,
                engrams_per_sec=latent_metrics.engrams_per_sec if latent_metrics else None,
                baseline_model=baseline_model,
                baseline_throughput=baseline_throughput,
                theoretical_max_throughput=theoretical_max_throughput,
            )

        # Compute diagnostic metrics for bottleneck analysis
        if utilization_metrics:
            diagnostic_metrics = DiagnosticMetrics.analyze(
                compute_util=utilization_metrics.gpu_utilization_pct,
                memory_util=utilization_metrics.gpu_memory_utilization_pct,
            )

        # Generate summary with eco-minded metrics
        total_metrics = sum(len(e) for e in components.values())
        regressions = [t for t in trends if t.is_regression]

        summary = {
            "total_metrics": total_metrics,
            "components_tested": len(components),
            "timestamp": metadata.timestamp,
            "hardware": metadata.gpu_name or metadata.cpu_model,
            "regressions_detected": len(regressions),
        }

        # Add power summary if available
        if power_metrics and power_metrics.gpu_power_watts:
            summary["gpu_power_watts"] = power_metrics.gpu_power_watts
            summary["gpu_temp_celsius"] = power_metrics.gpu_temp_celsius

        # Add efficiency summary if available
        if efficiency_metrics:
            if efficiency_metrics.samples_per_watt:
                summary["samples_per_watt"] = efficiency_metrics.samples_per_watt
            if efficiency_metrics.estimated_cost_per_1m_samples:
                summary["cost_per_1m_samples_usd"] = efficiency_metrics.estimated_cost_per_1m_samples
            if efficiency_metrics.estimated_co2_kg_per_1m_samples:
                summary["co2_kg_per_1m_samples"] = efficiency_metrics.estimated_co2_kg_per_1m_samples

        # Add latent space summary if available
        if latent_metrics:
            if latent_metrics.engrams_per_sec:
                summary["engrams_per_sec"] = latent_metrics.engrams_per_sec
            if latent_metrics.engrams_per_watt:
                summary["engrams_per_watt"] = latent_metrics.engrams_per_watt

        # Add normalized comparison summary
        if normalized_metrics:
            if normalized_metrics.speedup_vs_baseline:
                summary["speedup_vs_baseline"] = normalized_metrics.speedup_vs_baseline
            if normalized_metrics.equivalent_tokens_per_sec:
                summary["equivalent_tokens_per_sec"] = normalized_metrics.equivalent_tokens_per_sec

        # Add diagnostic summary
        if diagnostic_metrics and diagnostic_metrics.primary_bottleneck:
            summary["primary_bottleneck"] = diagnostic_metrics.primary_bottleneck
            summary["optimization_potential_pct"] = diagnostic_metrics.optimization_potential_pct

        return cls(
            metadata=metadata,
            components=components,
            summary=summary,
            power_metrics=power_metrics,
            efficiency_metrics=efficiency_metrics,
            utilization_metrics=utilization_metrics,
            latent_metrics=latent_metrics,
            normalized_metrics=normalized_metrics,
            diagnostic_metrics=diagnostic_metrics,
            trends=trends,
            raw_results=results,
        )

    def save(self, directory: Path | None = None) -> Path:
        """Save record to disk.

        Args:
            directory: Directory to save to (default: enriched/).

        Returns:
            Path to saved file.
        """
        directory = directory or ENRICHED_DIR
        directory.mkdir(parents=True, exist_ok=True)

        filename = f"benchmark_{self.metadata.run_id}.json"
        filepath = directory / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        return filepath

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "components": {
                name: [e.to_dict() for e in entries]
                for name, entries in self.components.items()
            },
            "summary": self.summary,
            "power_metrics": self.power_metrics.to_dict() if self.power_metrics else None,
            "efficiency_metrics": self.efficiency_metrics.to_dict() if self.efficiency_metrics else None,
            "utilization_metrics": self.utilization_metrics.to_dict() if self.utilization_metrics else None,
            "latent_metrics": self.latent_metrics.to_dict() if self.latent_metrics else None,
            "normalized_metrics": self.normalized_metrics.to_dict() if self.normalized_metrics else None,
            "diagnostic_metrics": self.diagnostic_metrics.to_dict() if self.diagnostic_metrics else None,
            "trends": [t.to_dict() for t in self.trends],
            "industry_context": self.industry_context,
            "raw_results": self.raw_results,
        }

    def print_summary(self) -> None:
        """Print human-readable summary with scale indicators."""
        print("=" * 80)
        print("ENHANCED BENCHMARK RECORD")
        print("=" * 80)
        print()
        print(f"Run ID:     {self.metadata.run_id}")
        print(f"Timestamp:  {self.metadata.timestamp}")
        print(f"Git:        {self.metadata.git_sha} ({self.metadata.git_branch})")
        print(f"Hardware:   {self.metadata.gpu_name or self.metadata.cpu_model}")
        print(f"PyTorch:    {self.metadata.pytorch_version}")
        print()

        for comp_name, entries in self.components.items():
            print(f"┌─ {comp_name.upper()} {'─' * (75 - len(comp_name))}")

            for entry in entries:
                si = entry.scale_indicator
                if si:
                    tier_icon = {
                        "excellent": "🟢",
                        "good": "🟡",
                        "fair": "🟠",
                        "poor": "🔴",
                        "unknown": "⚪",
                    }.get(si.quality_tier, "⚪")

                    short_name = entry.name.split(".")[-1]
                    print(
                        f"│ {tier_icon} {short_name:<40} {si.bar} {si.formatted:>15}"
                    )

            print(f"└{'─' * 78}")
            print()

        # Print power and efficiency metrics
        self._print_eco_metrics()

        # Print trends/regressions
        self._print_trends()

    def _print_eco_metrics(self) -> None:
        """Print eco-minded metrics (power, efficiency, utilization)."""
        if not any([self.power_metrics, self.efficiency_metrics, self.utilization_metrics]):
            return

        print("┌─ ECO & EFFICIENCY METRICS " + "─" * 51)

        # Power metrics
        if self.power_metrics:
            pm = self.power_metrics
            if pm.gpu_power_watts:
                bar = self._power_bar(pm.gpu_power_watts, pm.gpu_power_limit_watts)
                print(f"│ ⚡ GPU Power:     {pm.gpu_power_watts:>8.1f}W {bar}")
            if pm.gpu_temp_celsius:
                temp_bar = self._temp_bar(pm.gpu_temp_celsius)
                print(f"│ 🌡️  GPU Temp:      {pm.gpu_temp_celsius:>8.1f}°C {temp_bar}")
            if pm.gpu_energy_joules:
                print(f"│ 🔋 Energy Used:   {pm.gpu_energy_joules:>8.1f}J")

        # Efficiency metrics
        if self.efficiency_metrics:
            em = self.efficiency_metrics
            print("│ " + "─" * 76)
            print("│ EFFICIENCY (higher is better):")
            if em.samples_per_watt:
                print(f"│ 📊 Samples/Watt: {em.samples_per_watt:>10.2f} samples/W")
            if em.gflops_per_watt:
                print(f"│ 💻 GFLOPS/Watt:  {em.gflops_per_watt:>10.2f} GFLOP/W")
            if em.estimated_cost_per_1m_samples:
                print(f"│ 💰 Cost/1M:       ${em.estimated_cost_per_1m_samples:>9.4f}")
            if em.estimated_co2_kg_per_1m_samples:
                print(f"│ 🌍 CO₂/1M:       {em.estimated_co2_kg_per_1m_samples:>10.4f} kg")

        # Utilization metrics
        if self.utilization_metrics:
            um = self.utilization_metrics
            print("│ " + "─" * 76)
            print("│ UTILIZATION:")
            if um.gpu_utilization_pct is not None:
                bar = self._util_bar(um.gpu_utilization_pct)
                print(f"│ 🎮 GPU:          {um.gpu_utilization_pct:>8.1f}% {bar}")
            if um.gpu_memory_utilization_pct is not None:
                bar = self._util_bar(um.gpu_memory_utilization_pct)
                print(f"│ 💾 GPU Mem:      {um.gpu_memory_utilization_pct:>8.1f}% {bar}")
            if um.cpu_utilization_pct is not None:
                bar = self._util_bar(um.cpu_utilization_pct)
                print(f"│ 🖥️  CPU:          {um.cpu_utilization_pct:>8.1f}% {bar}")
            if um.gpu_clock_mhz:
                throttled = ""
                if um.gpu_max_clock_mhz and um.gpu_clock_mhz < um.gpu_max_clock_mhz * 0.9:
                    throttled = " ⚠️ THROTTLED"
                print(f"│ ⏱️  GPU Clock:    {um.gpu_clock_mhz:>8d} MHz{throttled}")

        print(f"└{'─' * 78}")
        print()

        # Print latent space metrics
        self._print_latent_metrics()

        # Print normalized comparison
        self._print_normalized_metrics()

        # Print diagnostic analysis
        self._print_diagnostic_metrics()

    def _print_latent_metrics(self) -> None:
        """Print CogSynDelta latent space metrics."""
        if not self.latent_metrics:
            return

        lm = self.latent_metrics
        print("┌─ 🧠 LATENT SPACE METRICS (CogSynDelta) " + "─" * 38)

        if lm.engrams_per_sec:
            print(f"│ 💾 Engrams/sec:      {lm.engrams_per_sec:>12.1f} engrams/s")
        if lm.engrams_per_watt:
            print(f"│ ⚡ Engrams/Watt:     {lm.engrams_per_watt:>12.2f} engrams/W")
        if lm.encoding_fidelity is not None:
            bar = self._util_bar(lm.encoding_fidelity * 100)
            print(f"│ 📊 Encode Fidelity:  {lm.encoding_fidelity:>12.4f} {bar}")
        if lm.reconstruction_fidelity is not None:
            bar = self._util_bar(lm.reconstruction_fidelity * 100)
            print(f"│ 🔄 Recon Fidelity:   {lm.reconstruction_fidelity:>12.4f} {bar}")
        if lm.latent_dim:
            print(f"│ 📐 Latent Dim:       {lm.latent_dim:>12d}")
        if lm.encode_latency_ms:
            print(f"│ ⏱️  Encode Latency:   {lm.encode_latency_ms:>12.2f} ms")
        if lm.decode_latency_ms:
            print(f"│ ⏱️  Decode Latency:   {lm.decode_latency_ms:>12.2f} ms")

        print(f"└{'─' * 78}")
        print()

    def _print_normalized_metrics(self) -> None:
        """Print normalized metrics for cross-architecture comparison."""
        if not self.normalized_metrics:
            return

        nm = self.normalized_metrics
        print("┌─ 📊 NORMALIZED COMPARISON " + "─" * 51)
        print(f"│ Baseline: {nm.baseline_model or 'reference'}")
        print("│ " + "─" * 76)

        if nm.speedup_vs_baseline:
            icon = "🚀" if nm.speedup_vs_baseline > 1 else "🐢"
            print(f"│ {icon} Speedup vs Baseline:    {nm.speedup_vs_baseline:>8.2f}x")
        if nm.relative_throughput:
            print(f"│ 📈 Relative Throughput:   {nm.relative_throughput:>8.1f}%")
        if nm.efficiency_vs_baseline:
            icon = "🟢" if nm.efficiency_vs_baseline > 1 else "🔴"
            print(f"│ {icon} Efficiency vs Baseline: {nm.efficiency_vs_baseline:>8.2f}x")
        if nm.throughput_per_param:
            print(f"│ 📊 Throughput/M params:   {nm.throughput_per_param:>8.2f}")
        if nm.equivalent_tokens_per_sec:
            print(f"│ 🔤 Equiv. Tokens/sec:     {nm.equivalent_tokens_per_sec:>8.0f}")
        if nm.normalized_quality is not None:
            bar = self._util_bar(nm.normalized_quality * 100)
            print(f"│ ⭐ Normalized Quality:    {nm.normalized_quality:>8.4f} {bar}")

        print(f"└{'─' * 78}")
        print()

    def _print_diagnostic_metrics(self) -> None:
        """Print diagnostic analysis for debugging/tuning."""
        if not self.diagnostic_metrics:
            return

        dm = self.diagnostic_metrics
        print("┌─ 🔧 DIAGNOSTIC ANALYSIS " + "─" * 53)

        # Bottleneck analysis
        if dm.primary_bottleneck:
            icons = {
                "compute": "💻",
                "memory": "💾",
                "io": "📁",
                "latency": "⏱️",
            }
            icon = icons.get(dm.primary_bottleneck, "❓")
            severity_bar = "█" * int((dm.bottleneck_severity or 0) * 10)
            print(f"│ {icon} Primary Bottleneck: {dm.primary_bottleneck.upper()}")
            print(f"│    Severity: {severity_bar}{'░' * (10 - len(severity_bar))}")
            if dm.bottleneck_explanation:
                # Wrap explanation
                words = dm.bottleneck_explanation.split()
                line = "│    "
                for word in words:
                    if len(line) + len(word) > 76:
                        print(line)
                        line = "│    " + word + " "
                    else:
                        line += word + " "
                if line.strip() != "│":
                    print(line)

        # Efficiency metrics
        print("│ " + "─" * 76)
        if dm.compute_efficiency_pct is not None:
            bar = self._util_bar(dm.compute_efficiency_pct)
            print(f"│ 💻 Compute Efficiency: {dm.compute_efficiency_pct:>6.1f}% {bar}")
        if dm.memory_efficiency_pct is not None:
            bar = self._util_bar(dm.memory_efficiency_pct)
            print(f"│ 💾 Memory Efficiency:  {dm.memory_efficiency_pct:>6.1f}% {bar}")

        # Recommendations
        if dm.recommendations:
            print("│ " + "─" * 76)
            print("│ 💡 RECOMMENDATIONS:")
            for rec in dm.recommendations[:3]:  # Top 3
                print(f"│    • {rec}")
            if dm.optimization_potential_pct:
                print(f"│ 📈 Est. Optimization Potential: {dm.optimization_potential_pct:.1f}%")

        print(f"└{'─' * 78}")
        print()

    def _power_bar(self, current: float, limit: float | None) -> str:
        """Generate power consumption bar."""
        if not limit:
            return ""
        pct = min(100, current / limit * 100)
        filled = int(pct / 10)
        if pct > 90:
            return "🔴" + "█" * filled + "░" * (10 - filled)
        if pct > 70:
            return "🟡" + "█" * filled + "░" * (10 - filled)
        return "🟢" + "█" * filled + "░" * (10 - filled)

    def _temp_bar(self, temp: float) -> str:
        """Generate temperature bar."""
        if temp > 85:
            return "🔴 HOT"
        if temp > 70:
            return "🟡 WARM"
        return "🟢 COOL"

    def _util_bar(self, pct: float) -> str:
        """Generate utilization bar."""
        filled = int(pct / 10)
        return "█" * filled + "░" * (10 - filled)

    def _print_trends(self) -> None:
        """Print performance trends and regressions."""
        if not self.trends:
            return

        regressions = [t for t in self.trends if t.is_regression]
        improvements = [t for t in self.trends if t.trend_direction == "improving"]

        if regressions:
            print("┌─ ⚠️  PERFORMANCE REGRESSIONS " + "─" * 48)
            for t in regressions:
                short_name = t.metric_name.split(".")[-1]
                print(
                    f"│ 📉 {short_name:<40} "
                    f"{t.delta_pct_from_previous:>+7.1f}%"
                )
            print(f"└{'─' * 78}")
            print()

        if improvements:
            print("┌─ 📈 IMPROVEMENTS " + "─" * 60)
            for t in improvements[:5]:  # Top 5
                short_name = t.metric_name.split(".")[-1]
                print(
                    f"│ 📈 {short_name:<40} "
                    f"{t.delta_pct_from_previous:>+7.1f}%"
                )
            if len(improvements) > 5:
                print(f"│    ... and {len(improvements) - 5} more")
            print(f"└{'─' * 78}")
            print()


def main() -> None:
    """CLI for history format testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced benchmark history format")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run format test with sample data",
    )
    parser.add_argument(
        "--sizes",
        action="store_true",
        help="List available model sizes",
    )
    parser.add_argument(
        "--power",
        action="store_true",
        help="Test power metrics capture",
    )

    args = parser.parse_args()

    if args.sizes:
        print("Available model sizes:")
        for size in ModelSizeConfig.all_sizes():
            config = ModelSizeConfig.get_config(size)
            print(f"  {size}: {config.parameters:,} params, {config.embed_dim}d embed")
    elif args.power:
        print("Capturing power metrics...")
        power = PowerMetrics.capture(duration_seconds=1.0)
        util = ComputeUtilizationMetrics.capture()
        print("\nPower Metrics:")
        for k, v in power.to_dict().items():
            if v is not None:
                print(f"  {k}: {v}")
        print("\nUtilization Metrics:")
        for k, v in util.to_dict().items():
            if v is not None:
                print(f"  {k}: {v}")
    elif args.test:
        # Test with sample data
        sample_results = {
            "components": {
                "test-component": {
                    "throughput": {"samples_per_sec": 1234.5},
                    "latency_ms": {"forward_bs32": 0.85},
                    "quality": {"fidelity": 0.95},
                }
            }
        }
        record = EnhancedBenchmarkRecord.from_benchmark_results(
            sample_results,
            throughput=1234.5,
            duration_seconds=10.0,
        )
        record.print_summary()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
