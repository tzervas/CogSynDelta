"""Power and eco-efficiency metrics collection.

This module provides hardware-agnostic power metrics with NVIDIA GPU support
and estimated ecological impact calculations.

Classes:
    PowerMetrics: Real-time power draw, temperature, and limits
    EcoMetrics: Energy efficiency, CO2 estimates, and sustainability scoring

Why separate from compute:
    Power metrics require specialized hardware access (pynvml) and have
    different update frequencies than utilization metrics. Keeping them
    separate allows selective import and mocking in tests.
"""

from __future__ import annotations

import platform
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
class PowerMetrics(MetricMixin):
    """GPU and system power consumption metrics.

    Captures real-time power draw using NVIDIA Management Library (pynvml)
    when available. Falls back to estimation for non-NVIDIA hardware.

    Attributes:
        power_draw_w: Current power draw in watts.
        power_limit_w: Maximum power limit setting.
        gpu_temperature_c: GPU temperature in Celsius.
        cpu_temperature_c: CPU package temperature (if available).
        fan_speed_pct: GPU fan speed percentage.
        capture_timestamp: When metrics were captured.

    Why power tracking:
        Modern ML workloads consume significant energy. Tracking power
        enables optimization for efficiency (samples/watt) and
        sustainability reporting.
    """

    power_draw_w: float | None = None
    power_limit_w: float | None = None
    gpu_temperature_c: float | None = None
    cpu_temperature_c: float | None = None
    fan_speed_pct: float | None = None
    capture_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def capture(cls, gpu_index: int = 0) -> PowerMetrics:
        """Capture current power metrics from hardware.

        Attempts to read from NVIDIA GPU via pynvml. Falls back to
        None values if hardware access fails.

        Args:
            gpu_index: Index of GPU to query (default: 0).

        Returns:
            PowerMetrics instance with current values.

        Example:
            >>> metrics = PowerMetrics.capture()
            >>> print(f"Power: {metrics.power_draw_w}W")
        """
        power_w = None
        limit_w = None
        gpu_temp = None
        fan_pct = None

        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

                # Power
                with contextlib.suppress(pynvml.NVMLError):
                    power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0

                # Power limit
                with contextlib.suppress(pynvml.NVMLError):
                    limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0

                # Temperature
                with contextlib.suppress(pynvml.NVMLError):
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )

                # Fan
                with contextlib.suppress(pynvml.NVMLError):
                    fan_pct = pynvml.nvmlDeviceGetFanSpeed(handle)

                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                # NVML errors are non-fatal: GPU metrics are optional and failures
                # should not prevent overall metric collection.
                pass

        # CPU temperature (Linux)
        cpu_temp = None
        if HAS_PSUTIL and platform.system() == "Linux":
            try:
                temps = psutil.sensors_temperatures()
                if "coretemp" in temps:
                    cpu_temp = temps["coretemp"][0].current
                elif "k10temp" in temps:  # AMD
                    cpu_temp = temps["k10temp"][0].current
            except (OSError, KeyError, AttributeError):
                # Temperature sensors are optional/best-effort; ignore failures and keep cpu_temp as None.
                pass

        return cls(
            power_draw_w=power_w,
            power_limit_w=limit_w,
            gpu_temperature_c=gpu_temp,
            cpu_temperature_c=cpu_temp,
            fan_speed_pct=fan_pct,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PowerMetrics:
        """Deserialize from dictionary.

        Args:
            data: Dictionary of metric values.

        Returns:
            PowerMetrics instance.
        """
        timestamp = data.get("capture_timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            power_draw_w=data.get("power_draw_w"),
            power_limit_w=data.get("power_limit_w"),
            gpu_temperature_c=data.get("gpu_temperature_c"),
            cpu_temperature_c=data.get("cpu_temperature_c"),
            fan_speed_pct=data.get("fan_speed_pct"),
            capture_timestamp=timestamp or datetime.now(UTC),
        )

    def power_efficiency_pct(self) -> float | None:
        """Calculate power efficiency as percentage of limit.

        Returns:
            Percentage of power limit being used, or None.
        """
        if self.power_draw_w and self.power_limit_w:
            return (self.power_draw_w / self.power_limit_w) * 100
        return None


@dataclass
class EcoMetrics(MetricMixin):
    """Ecological and energy efficiency metrics.

    Computes sustainability-focused metrics including energy per sample,
    estimated CO2 emissions, and efficiency ratings compared to baselines.

    Attributes:
        energy_joules: Total energy consumed in joules.
        energy_per_sample_j: Energy per sample processed.
        samples_per_watt: Throughput efficiency.
        samples_per_joule: Energy efficiency.
        estimated_co2_grams: Estimated CO2 emissions.
        estimated_co2_kg_per_1m_samples: CO2 per million samples.
        efficiency_rating: Qualitative rating (A-F).

    Why eco-metrics:
        CogSynDelta aims to be a sustainable AI system. These metrics
        enable tracking progress toward eco-efficiency goals and
        comparison against industry baselines.
    """

    energy_joules: float | None = None
    energy_per_sample_j: float | None = None
    samples_per_watt: float | None = None
    samples_per_joule: float | None = None
    estimated_co2_grams: float | None = None
    estimated_co2_kg_per_1m_samples: float | None = None
    efficiency_rating: str | None = None

    # Constants for CO2 estimation
    # Average grid carbon intensity: 400g CO2/kWh (global average)
    # Source: IEA World Energy Outlook
    DEFAULT_CARBON_INTENSITY_G_KWH: float = field(default=400.0, repr=False)

    @classmethod
    def compute(
        cls,
        duration_sec: float,
        power_draw_w: float,
        num_samples: int,
        carbon_intensity_g_kwh: float = 400.0,
    ) -> EcoMetrics:
        """Compute eco-metrics from benchmark results.

        Args:
            duration_sec: Total benchmark duration in seconds.
            power_draw_w: Average power draw in watts.
            num_samples: Number of samples processed.
            carbon_intensity_g_kwh: Grid carbon intensity (default: global avg).

        Returns:
            EcoMetrics instance with computed values.

        Example:
            >>> eco = EcoMetrics.compute(
            ...     duration_sec=60,
            ...     power_draw_w=250,
            ...     num_samples=10000
            ... )
            >>> print(f"CO2: {eco.estimated_co2_grams}g")
        """
        # Energy calculations
        energy_j = power_draw_w * duration_sec
        energy_per_sample = energy_j / num_samples if num_samples > 0 else None

        # Efficiency calculations
        samples_per_watt = num_samples / power_draw_w if power_draw_w > 0 else None
        samples_per_joule = num_samples / energy_j if energy_j > 0 else None

        # CO2 estimation
        # energy_j -> kWh: divide by 3,600,000
        energy_kwh = energy_j / 3_600_000
        co2_grams = energy_kwh * carbon_intensity_g_kwh

        # CO2 per million samples
        co2_per_1m = (co2_grams / num_samples) * 1_000_000 if num_samples > 0 else None

        # Efficiency rating based on samples/watt
        rating = cls._compute_rating(samples_per_watt)

        return cls(
            energy_joules=energy_j,
            energy_per_sample_j=energy_per_sample,
            samples_per_watt=samples_per_watt,
            samples_per_joule=samples_per_joule,
            estimated_co2_grams=co2_grams,
            estimated_co2_kg_per_1m_samples=co2_per_1m / 1000 if co2_per_1m else None,
            efficiency_rating=rating,
        )

    @staticmethod
    def _compute_rating(samples_per_watt: float | None) -> str:
        """Compute qualitative efficiency rating.

        Based on typical ML workload efficiency ranges:
        - A: >100 samples/W (highly efficient)
        - B: 50-100 samples/W (good)
        - C: 20-50 samples/W (average)
        - D: 10-20 samples/W (below average)
        - F: <10 samples/W (inefficient)

        Args:
            samples_per_watt: Throughput efficiency value.

        Returns:
            Single letter rating A-F.
        """
        if samples_per_watt is None:
            return "?"
        if samples_per_watt >= 100:
            return "A"
        if samples_per_watt >= 50:
            return "B"
        if samples_per_watt >= 20:
            return "C"
        if samples_per_watt >= 10:
            return "D"
        return "F"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EcoMetrics:
        """Deserialize from dictionary.

        Args:
            data: Dictionary of metric values.

        Returns:
            EcoMetrics instance.
        """
        return cls(
            energy_joules=data.get("energy_joules"),
            energy_per_sample_j=data.get("energy_per_sample_j"),
            samples_per_watt=data.get("samples_per_watt"),
            samples_per_joule=data.get("samples_per_joule"),
            estimated_co2_grams=data.get("estimated_co2_grams"),
            estimated_co2_kg_per_1m_samples=data.get("estimated_co2_kg_per_1m_samples"),
            efficiency_rating=data.get("efficiency_rating"),
        )


# Register with global registry
METRIC_REGISTRY.register("power", PowerMetrics)
METRIC_REGISTRY.register("eco", EcoMetrics)
