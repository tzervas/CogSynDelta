"""Metric display formatting and visualization utilities.

This module provides utilities for formatting and displaying metrics
in human-readable form, including scale indicators, progress bars,
and rich terminal output.

Classes:
    MetricFormatter: Format values with units and scale indicators
    ScaledMetricDisplay: Rich display with sparklines and trends

Why dedicated display module:
    Separating display logic from metric computation enables:
    1. Multiple output formats (terminal, HTML, JSON)
    2. Consistent formatting across the codebase
    3. Easy testing of formatting logic
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.metrics.base import TYPICAL_RANGES, MetricUnit

# Unit labels for human-readable output (module-level constant)
_UNIT_LABELS: dict[MetricUnit, str] = {
    MetricUnit.SAMPLES_PER_SEC: "samples/s",
    MetricUnit.MS: "ms",
    MetricUnit.US: "µs",
    MetricUnit.NS: "ns",
    MetricUnit.FLOPS: "FLOPS",
    MetricUnit.TFLOPS: "TFLOPS",
    MetricUnit.GFLOPS: "GFLOPS",
    MetricUnit.BYTES: "B",
    MetricUnit.KB: "KB",
    MetricUnit.MB: "MB",
    MetricUnit.GB: "GB",
    MetricUnit.TB: "TB",
    MetricUnit.GB_PER_SEC: "GB/s",
    MetricUnit.WATTS: "W",
    MetricUnit.JOULES: "J",
    MetricUnit.SAMPLES_PER_WATT: "samples/W",
    MetricUnit.SAMPLES_PER_JOULE: "samples/J",
    MetricUnit.KG_CO2: "kg CO₂",
    MetricUnit.GRAMS_CO2: "g CO₂",
    MetricUnit.ENGRAMS_PER_SEC: "engrams/s",
    MetricUnit.EMBEDDINGS_PER_SEC: "emb/s",
    MetricUnit.LATENT_OPS_PER_SEC: "latent-ops/s",
    MetricUnit.ENGRAMS_PER_WATT: "engrams/W",
    MetricUnit.ENGRAMS_PER_JOULE: "engrams/J",
    MetricUnit.BITS_PER_DIM: "bits/dim",
    MetricUnit.NORMALIZED_SCORE: "",
    MetricUnit.RELATIVE_PERF: "x",
    MetricUnit.SPEEDUP: "x",
    MetricUnit.PERCENT: "%",
    MetricUnit.RATIO: "",
    MetricUnit.COUNT: "",
    MetricUnit.DIMENSIONLESS: "",
}


@dataclass
class MetricFormatter:
    """Format metric values with units and scale indicators.

    Provides consistent formatting for metric values across the codebase,
    including automatic unit prefixes (k, M, G, etc.) and alignment.

    Attributes:
        precision: Number of decimal places for values.
        align_width: Total width for value alignment.
        show_unit: Whether to append unit to formatted value.

    Example:
        >>> fmt = MetricFormatter(precision=2)
        >>> fmt.format(1234567, MetricUnit.SAMPLES_PER_SEC)
        '1.23M samples/s'
    """

    precision: int = 2
    align_width: int = 12
    show_unit: bool = True

    def format(
        self,
        value: float | None,
        unit: MetricUnit,
        metric_name: str | None = None,
    ) -> str:
        """Format a metric value with appropriate scale and unit.

        Args:
            value: The raw metric value.
            unit: The unit of measurement.
            metric_name: Optional metric name for range lookup.

        Returns:
            Formatted string representation.
        """
        if value is None:
            return f"{'N/A':>{self.align_width}}"

        # Get scale indicator
        scale_range = TYPICAL_RANGES.get(metric_name or "")
        if scale_range:
            scale = scale_range.get_scale_indicator(value)
        else:
            scale = self._auto_scale(value)

        # Scale the value
        scaled_value = self._scale_value(value, scale)

        # Format with precision
        formatted = f"{scaled_value:.{self.precision}f}{scale}"

        # Add unit if requested
        if self.show_unit:
            unit_label = _UNIT_LABELS.get(unit, "")
            if unit_label:
                formatted = f"{formatted} {unit_label}"

        return f"{formatted:>{self.align_width}}"

    def _auto_scale(self, value: float) -> str:  # noqa: PLR0911
        """Determine appropriate scale prefix for value.

        Args:
            value: Raw value.

        Returns:
            Scale prefix string.
        """
        if value >= 1e12:
            return "T"
        if value >= 1e9:
            return "G"
        if value >= 1e6:
            return "M"
        if value >= 1e3:
            return "k"
        if value < 0.001 and value > 0:
            return "µ"
        if value < 1 and value > 0:
            return "m"
        return ""

    def _scale_value(self, value: float, scale: str) -> float:
        """Scale value by prefix factor.

        Args:
            value: Raw value.
            scale: Scale prefix.

        Returns:
            Scaled value.
        """
        factors = {"T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "m": 1e-3, "µ": 1e-6}
        factor = factors.get(scale, 1.0)
        return value / factor if factor != 1.0 else value

    def format_percent(self, value: float | None, width: int = 6) -> str:
        """Format a percentage value.

        Args:
            value: Percentage value (0-100).
            width: Alignment width.

        Returns:
            Formatted percentage string.
        """
        if value is None:
            return f"{'N/A':>{width}}"
        return f"{value:>{width}.1f}%"

    def format_bar(
        self,
        value: float,
        max_value: float = 100,
        width: int = 10,
        filled_char: str = "█",
        empty_char: str = "░",
    ) -> str:
        """Generate a progress bar visualization.

        Args:
            value: Current value.
            max_value: Maximum value (for scaling).
            width: Bar width in characters.
            filled_char: Character for filled portion.
            empty_char: Character for empty portion.

        Returns:
            ASCII progress bar string.
        """
        pct = min(1.0, value / max_value) if max_value > 0 else 0
        filled = int(pct * width)
        return filled_char * filled + empty_char * (width - filled)


@dataclass
class ScaledMetricDisplay:
    """Rich metric display with trends and visual indicators.

    Provides formatted display of metrics with trend sparklines,
    delta indicators, and color-coded status.

    Attributes:
        formatter: MetricFormatter instance for value formatting.
        show_trends: Whether to display trend sparklines.
        show_deltas: Whether to show change from previous.
        unicode_enabled: Whether to use unicode characters.

    Example:
        >>> display = ScaledMetricDisplay()
        >>> print(display.render_metric("throughput", 25000, [20000, 22000, 25000]))
        '📈 throughput:   25.00k samples/s  ▁▃█ (+14.0%)'
    """

    formatter: MetricFormatter | None = None
    show_trends: bool = True
    show_deltas: bool = True
    unicode_enabled: bool = True

    def __post_init__(self) -> None:
        """Initialize formatter if not provided."""
        if self.formatter is None:
            self.formatter = MetricFormatter()

    def render_metric(
        self,
        name: str,
        value: float | None,
        history: list[float] | None = None,
        unit: MetricUnit = MetricUnit.DIMENSIONLESS,
        threshold_warning: float | None = None,
        threshold_critical: float | None = None,
    ) -> str:
        """Render a metric with optional trend and thresholds.

        Args:
            name: Metric name for display.
            value: Current metric value.
            history: Historical values for trend sparkline.
            unit: Unit of measurement.
            threshold_warning: Warning threshold value.
            threshold_critical: Critical threshold value.

        Returns:
            Formatted string with metric display.
        """
        assert self.formatter is not None  # For type checker

        # Status icon
        icon = self._status_icon(value, threshold_warning, threshold_critical)

        # Format value
        formatted_value = self.formatter.format(value, unit, name)

        # Trend sparkline
        trend = ""
        if self.show_trends and history and len(history) >= 2:
            trend = " " + self._sparkline(history)

        # Delta from previous
        delta = ""
        if self.show_deltas and history and len(history) >= 2 and value is not None:
            prev = history[-2]
            if prev > 0:
                pct_change = ((value - prev) / prev) * 100
                sign = "+" if pct_change >= 0 else ""
                delta = f" ({sign}{pct_change:.1f}%)"

        return f"{icon} {name:<20} {formatted_value}{trend}{delta}"

    def _status_icon(
        self,
        value: float | None,
        warning: float | None,
        critical: float | None,
    ) -> str:
        """Get status icon based on thresholds.

        Args:
            value: Current value.
            warning: Warning threshold.
            critical: Critical threshold.

        Returns:
            Status emoji.
        """
        if value is None:
            return "❓"
        if critical is not None and value >= critical:
            return "🔴"
        if warning is not None and value >= warning:
            return "🟡"
        return "🟢"

    def _sparkline(self, values: list[float], width: int = 7) -> str:
        """Generate sparkline visualization of trend.

        Args:
            values: Historical values.
            width: Number of values to show.

        Returns:
            Unicode sparkline string.
        """
        if not self.unicode_enabled:
            return ""

        # Take last N values
        recent = values[-width:]
        if not recent:
            return ""

        # Normalize to 0-7 range (8 sparkline chars)
        min_val = min(recent)
        max_val = max(recent)
        range_val = max_val - min_val if max_val > min_val else 1

        sparkline_chars = "▁▂▃▄▅▆▇█"
        result = ""
        for v in recent:
            idx = int((v - min_val) / range_val * 7)
            result += sparkline_chars[min(7, idx)]

        return result

    def render_comparison_table(
        self,
        metrics: dict[str, float],
        baselines: dict[str, float],
        unit: MetricUnit = MetricUnit.DIMENSIONLESS,
    ) -> str:
        """Render side-by-side comparison table.

        Args:
            metrics: Current metric values.
            baselines: Baseline values for comparison.
            unit: Unit of measurement.

        Returns:
            Multi-line table string.
        """
        lines = ["┌─ COMPARISON TABLE " + "─" * 50]
        lines.append("│ " + f"{'Metric':<20} {'Current':>12} {'Baseline':>12} {'Delta':>10}")
        lines.append("│ " + "─" * 56)

        assert self.formatter is not None  # For type checker

        for name, value in metrics.items():
            baseline = baselines.get(name, 0)
            delta_pct = ((value - baseline) / baseline * 100) if baseline > 0 else 0
            sign = "+" if delta_pct >= 0 else ""

            formatted_current = self.formatter.format(value, unit)
            formatted_baseline = self.formatter.format(baseline, unit)

            icon = "📈" if delta_pct > 0 else "📉" if delta_pct < 0 else "➡️"
            lines.append(
                f"│ {icon} {name:<18} {formatted_current:>12} {formatted_baseline:>12} "
                f"{sign}{delta_pct:>8.1f}%"
            )

        lines.append("└" + "─" * 68)
        return "\n".join(lines)
