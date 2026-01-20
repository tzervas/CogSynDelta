"""Base metric protocols, units, and registries.

This module provides the foundational building blocks for all metrics:
- BaseMetric: Protocol defining the interface all metrics must implement
- MetricUnit: Enumeration of all supported measurement units
- ScaleRange: Defines typical value ranges for scale indicators
- MetricRegistry: Central registry for metric type discovery

Why this design:
    The protocol-based approach allows metrics to be used polymorphically
    while maintaining full type safety. The registry enables dynamic
    discovery of metric types at runtime for serialization/deserialization.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass


class MetricUnit(Enum):
    """Units of measurement for benchmark metrics.

    Organized by category for clarity:
    - Performance: throughput, latency, FLOPS
    - Memory: bytes, bandwidth
    - Power/Eco: watts, joules, CO2
    - Latent Space: engrams (CogSynDelta-specific)
    - Normalized: relative comparisons
    """

    # Performance
    SAMPLES_PER_SEC = auto()
    MS = auto()
    US = auto()
    NS = auto()
    FLOPS = auto()
    TFLOPS = auto()
    GFLOPS = auto()

    # Memory
    BYTES = auto()
    KB = auto()
    MB = auto()
    GB = auto()
    TB = auto()
    GB_PER_SEC = auto()

    # Power and efficiency
    WATTS = auto()
    JOULES = auto()
    SAMPLES_PER_WATT = auto()
    SAMPLES_PER_JOULE = auto()
    KG_CO2 = auto()
    GRAMS_CO2 = auto()

    # Latent space (CogSynDelta-specific)
    ENGRAMS_PER_SEC = auto()
    EMBEDDINGS_PER_SEC = auto()
    LATENT_OPS_PER_SEC = auto()
    ENGRAMS_PER_WATT = auto()
    ENGRAMS_PER_JOULE = auto()
    BITS_PER_DIM = auto()

    # Normalized/comparative
    NORMALIZED_SCORE = auto()
    RELATIVE_PERF = auto()
    SPEEDUP = auto()
    PERCENT = auto()
    RATIO = auto()

    # Other
    COUNT = auto()
    DIMENSIONLESS = auto()


@dataclass
class ScaleRange:
    """Defines typical value ranges for scale indicators.

    Used to determine magnitude prefixes (k, M, G, etc.) and
    contextual formatting for human-readable output.

    Attributes:
        metric_name: Key used to look up this range.
        min_typical: Lower bound of typical values.
        max_typical: Upper bound of typical values.
        unit: The measurement unit for this metric.
        description: Human-readable explanation.
        warning_threshold: Value above which to show warnings.
        critical_threshold: Value above which to show critical alerts.
    """

    metric_name: str
    min_typical: float
    max_typical: float
    unit: MetricUnit
    description: str = ""
    warning_threshold: float | None = None
    critical_threshold: float | None = None

    def get_scale_indicator(self, value: float) -> str:  # noqa: PLR0911
        """Get magnitude indicator for value.

        Args:
            value: The metric value to format.

        Returns:
            Scale prefix like 'k', 'M', 'G' or empty string.
        """
        if value >= 1e12:
            return "T"
        if value >= 1e9:
            return "G"
        if value >= 1e6:
            return "M"
        if value >= 1e3:
            return "k"
        if value < 0.001:
            return "µ"
        if value < 1:
            return "m"
        return ""

    def format_value(self, value: float) -> str:
        """Format value with appropriate scale.

        Args:
            value: Raw metric value.

        Returns:
            Formatted string with scale suffix.
        """
        scale = self.get_scale_indicator(value)
        scale_factors = {"T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "m": 1e-3, "µ": 1e-6}
        factor = scale_factors.get(scale, 1.0)
        scaled_value = value / factor if factor != 1.0 else value
        return f"{scaled_value:.2f}{scale}"


# Typical ranges for common metrics
TYPICAL_RANGES: dict[str, ScaleRange] = {
    # Performance
    "throughput": ScaleRange(
        metric_name="throughput",
        min_typical=100,
        max_typical=1_000_000,
        unit=MetricUnit.SAMPLES_PER_SEC,
        description="Samples processed per second",
    ),
    "latency_ms": ScaleRange(
        metric_name="latency_ms",
        min_typical=0.1,
        max_typical=1000,
        unit=MetricUnit.MS,
        description="Processing latency in milliseconds",
        warning_threshold=100,
        critical_threshold=500,
    ),
    # Power
    "power_draw_w": ScaleRange(
        metric_name="power_draw_w",
        min_typical=50,
        max_typical=500,
        unit=MetricUnit.WATTS,
        description="GPU power consumption",
        warning_threshold=350,
        critical_threshold=450,
    ),
    "temperature_c": ScaleRange(
        metric_name="temperature_c",
        min_typical=30,
        max_typical=100,
        unit=MetricUnit.DIMENSIONLESS,
        description="GPU temperature in Celsius",
        warning_threshold=80,
        critical_threshold=90,
    ),
    # Latent space
    "engrams_per_sec": ScaleRange(
        metric_name="engrams_per_sec",
        min_typical=100,
        max_typical=1_000_000,
        unit=MetricUnit.ENGRAMS_PER_SEC,
        description="Engrams encoded per second",
    ),
    "engrams_per_watt": ScaleRange(
        metric_name="engrams_per_watt",
        min_typical=1,
        max_typical=10_000,
        unit=MetricUnit.ENGRAMS_PER_WATT,
        description="Engram encoding efficiency",
    ),
    # Normalized
    "normalized_throughput": ScaleRange(
        metric_name="normalized_throughput",
        min_typical=0,
        max_typical=1,
        unit=MetricUnit.NORMALIZED_SCORE,
        description="Throughput normalized to 0-1 range",
    ),
    "speedup": ScaleRange(
        metric_name="speedup",
        min_typical=0.1,
        max_typical=100,
        unit=MetricUnit.SPEEDUP,
        description="Performance relative to baseline",
    ),
}


@runtime_checkable
class BaseMetric(Protocol):
    """Protocol defining the interface all metrics must implement.

    All metric classes must provide:
    - to_dict(): Serialization to dictionary
    - to_json(): Serialization to JSON string
    - from_dict(): Deserialization from dictionary (classmethod)

    Why Protocol:
        Using a Protocol instead of an ABC allows for structural typing,
        meaning any class with these methods can be used as a BaseMetric
        without explicit inheritance.
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert metric to dictionary for serialization."""
        ...

    def to_json(self) -> str:
        """Convert metric to JSON string."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseMetric:
        """Create metric instance from dictionary."""
        ...


@dataclass
class MetricMixin:
    """Mixin providing common serialization methods for metrics.

    Inherit from this class (via composition) to get automatic
    to_dict(), to_json(), and from_dict() implementations.

    Example:
        @dataclass
        class MyMetric(MetricMixin):
            value: float
            unit: MetricUnit = MetricUnit.SAMPLES_PER_SEC
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary with enum handling.

        Returns:
            Dictionary representation of the metric.
        """
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Enum):
                result[key] = value.name
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    def to_json(self) -> str:
        """Serialize to JSON string.

        Returns:
            JSON representation of the metric.
        """
        return json.dumps(self.to_dict(), indent=2)


class MetricRegistry:
    """Central registry for metric type discovery and instantiation.

    Allows registration of metric classes and lookup by name,
    enabling dynamic deserialization without hardcoding types.

    Attributes:
        _metrics: Internal mapping of name to metric class.

    Example:
        >>> registry = MetricRegistry()
        >>> registry.register("power", PowerMetrics)
        >>> power_class = registry.get("power")
        >>> metrics = power_class.capture()
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._metrics: dict[str, type] = {}

    def register(self, name: str, metric_class: type) -> None:
        """Register a metric class.

        Args:
            name: Unique identifier for the metric type.
            metric_class: The class to register.

        Raises:
            ValueError: If name is already registered.
        """
        if name in self._metrics:
            raise ValueError(f"Metric '{name}' already registered")
        self._metrics[name] = metric_class

    def get(self, name: str) -> type:
        """Get a registered metric class.

        Args:
            name: The registered name of the metric.

        Returns:
            The metric class.

        Raises:
            KeyError: If name is not registered.
        """
        return self._metrics[name]

    def list_all(self) -> list[str]:
        """List all registered metric names.

        Returns:
            Sorted list of registered metric names.
        """
        return sorted(self._metrics.keys())

    def deserialize(self, name: str, data: dict[str, Any]) -> Any:
        """Deserialize data into a metric instance.

        Args:
            name: The registered metric name.
            data: Dictionary of metric values.

        Returns:
            Instantiated metric object.
        """
        metric_class = self.get(name)
        if hasattr(metric_class, "from_dict"):
            return metric_class.from_dict(data)
        return metric_class(**data)


# Global registry instance
METRIC_REGISTRY = MetricRegistry()
