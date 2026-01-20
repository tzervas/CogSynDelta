"""Benchmark Visualization and Trend Analysis.

Provides visual representations of benchmark trends over time using
terminal-based charts (ASCII) and optional rich HTML reports.

Why terminal-based first:
    Per constitution, the system must degrade gracefully. Not all
    environments have matplotlib/plotly installed. ASCII charts work
    everywhere and are CI-friendly.

Key features:
    - Sparklines for quick trend visualization
    - ASCII bar charts for metric comparison
    - HTML report generation when matplotlib available
    - Metric heatmaps for multi-dimensional analysis
    - Regression detection with visual indicators

Example:
    >>> from benchmarks.visualization import BenchmarkVisualizer
    >>> viz = BenchmarkVisualizer()
    >>> viz.print_trend_report()
    >>> viz.generate_html_report("benchmark_report.html")
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

__all__ = [
    "AsciiChart",
    "BenchmarkVisualizer",
    "MetricTrend",
    "SparklineRenderer",
    "TrendReport",
]

# Paths
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark_results"
HISTORY_DIR = BENCHMARK_DIR / "history"

# Sparkline characters (Unicode block elements for smooth gradients)
SPARK_CHARS = "▁▂▃▄▅▆▇█"

# Status indicators
STATUS_IMPROVING = "📈"
STATUS_DEGRADING = "📉"
STATUS_STABLE = "➡️"
STATUS_WARNING = "⚠️"
STATUS_OK = "✅"


@dataclass
class MetricTrend:
    """Trend analysis for a single metric.

    Attributes:
        name: Metric name.
        values: List of (timestamp, value) tuples.
        current: Most recent value.
        baseline: First/reference value.
        change_pct: Percentage change from baseline.
        trend: Trend direction string.
        sparkline: ASCII sparkline representation.
        is_improving: Whether trend is positive.
        higher_is_better: Whether higher values are better for this metric.
    """

    name: str
    values: list[tuple[str, float]]
    current: float
    baseline: float
    change_pct: float
    trend: str
    sparkline: str
    is_improving: bool
    higher_is_better: bool = True


@dataclass
class TrendReport:
    """Complete trend report for all tracked metrics.

    Attributes:
        timestamp: When report was generated.
        metrics: List of MetricTrend objects.
        summary: Summary statistics.
        recommendations: Suggested actions based on trends.
    """

    timestamp: str
    metrics: list[MetricTrend]
    summary: dict[str, Any]
    recommendations: list[str]


class SparklineRenderer:
    """Renders sparklines from numerical data.

    Sparklines are compact inline charts that show trends at a glance.
    Uses Unicode block characters for smooth visual representation.

    Why sparklines:
        - Fit in terminal output alongside text
        - Instantly communicate trend direction
        - Work in any terminal/environment
        - CI log-friendly (no external dependencies)
    """

    def __init__(self, chars: str = SPARK_CHARS) -> None:
        """Initialize renderer with character set.

        Args:
            chars: Characters to use for sparkline (lowest to highest).
        """
        self.chars = chars
        self.num_levels = len(chars)

    def render(
        self,
        values: list[float],
        width: int | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> str:
        """Render sparkline from values.

        Args:
            values: Numeric values to visualize.
            width: Maximum width (will sample if exceeded).
            min_val: Minimum value for scaling (default: min of values).
            max_val: Maximum value for scaling (default: max of values).

        Returns:
            Sparkline string.

        Example:
            >>> SparklineRenderer().render([1, 2, 3, 4, 5, 4, 3, 2, 1])
            '▁▂▄▆█▆▄▂▁'
        """
        if not values:
            return ""

        # Sample if too many values
        if width and len(values) > width:
            step = len(values) / width
            values = [values[int(i * step)] for i in range(width)]

        # Determine scale
        min_v = min_val if min_val is not None else min(values)
        max_v = max_val if max_val is not None else max(values)
        range_v = max_v - min_v

        if range_v == 0:
            # All values same - use middle character
            return self.chars[self.num_levels // 2] * len(values)

        # Map values to characters
        result = []
        for v in values:
            normalized = (v - min_v) / range_v
            idx = min(int(normalized * (self.num_levels - 1)), self.num_levels - 1)
            result.append(self.chars[idx])

        return "".join(result)


class AsciiChart:
    """Simple ASCII bar charts for terminal output.

    Why ASCII charts:
        - No dependencies required
        - Works in CI logs and terminals
        - Accessible (screen readers can read)
        - Fast to render
    """

    @staticmethod
    def horizontal_bar(
        label: str,
        value: float,
        max_value: float,
        width: int = 40,
        fill_char: str = "█",
        empty_char: str = "░",
    ) -> str:
        """Render horizontal bar.

        Args:
            label: Bar label.
            value: Current value.
            max_value: Maximum value for scaling.
            width: Bar width in characters.
            fill_char: Character for filled portion.
            empty_char: Character for empty portion.

        Returns:
            Formatted bar string.
        """
        ratio: float
        if max_value <= 0:
            ratio = 0.0
        else:
            ratio = min(value / max_value, 1.0)

        filled = int(ratio * width)
        empty = width - filled
        bar = fill_char * filled + empty_char * empty

        return f"{label:<25} │{bar}│ {value:>10.2f}"

    @staticmethod
    def comparison_chart(
        metrics: list[tuple[str, float, float]],
        title: str = "Comparison",
        width: int = 40,
    ) -> str:
        """Render comparison chart with current vs baseline.

        Args:
            metrics: List of (name, current, baseline) tuples.
            title: Chart title.
            width: Bar width.

        Returns:
            Multi-line chart string.
        """
        if not metrics:
            return ""

        max_val = max(max(m[1], m[2]) for m in metrics)
        lines = [
            f"╔{'═' * (width + 40)}╗",
            f"║ {title:^{width + 38}} ║",
            f"╠{'═' * (width + 40)}╣",
            f"║ {'Metric':<25} │ {'Bar':^{width}} │ {'Value':>10} ║",
            f"╟{'─' * (width + 40)}╢",
        ]

        for name, current, baseline in metrics:
            # Current value bar
            bar = AsciiChart.horizontal_bar(name, current, max_val, width)
            lines.append(f"║ {bar} ║")

            # Baseline indicator (thin line)
            baseline_pos = int((baseline / max_val) * width) if max_val > 0 else 0
            baseline_marker = " " * baseline_pos + "│" + " " * (width - baseline_pos - 1)
            delta_pct = ((current - baseline) / baseline * 100) if baseline != 0 else 0
            delta_str = f"{delta_pct:+.1f}%"
            lines.append(f"║ {'(baseline)':<25} │{baseline_marker}│ {delta_str:>10} ║")
            lines.append(f"╟{'─' * (width + 40)}╢")

        lines[-1] = f"╚{'═' * (width + 40)}╝"
        return "\n".join(lines)


class BenchmarkVisualizer:
    """Comprehensive benchmark visualization and trend reporting.

    Provides:
    - Sparkline trend visualization
    - Comparison charts (current vs baseline)
    - Regression heatmaps
    - HTML report generation
    - Actionable recommendations

    Example:
        >>> viz = BenchmarkVisualizer()
        >>> viz.print_trend_report()
        >>> viz.export_markdown("BENCHMARK_TRENDS.md")
    """

    def __init__(
        self,
        history_dir: Path | None = None,
        baseline_dir: Path | None = None,
    ) -> None:
        """Initialize visualizer.

        Args:
            history_dir: Directory containing historical benchmark JSON files.
            baseline_dir: Directory containing baseline files.
        """
        self.history_dir = history_dir or HISTORY_DIR
        self.baseline_dir = baseline_dir or (BENCHMARK_DIR / "baselines")
        self.sparkline = SparklineRenderer()
        self._history_cache: list[dict[str, Any]] | None = None

    def load_history(self, max_entries: int = 50) -> list[dict[str, Any]]:
        """Load benchmark history, newest first.

        Args:
            max_entries: Maximum entries to load.

        Returns:
            List of benchmark result dictionaries.
        """
        if self._history_cache is not None:
            return self._history_cache[:max_entries]

        if not self.history_dir.exists():
            return []

        files = sorted(
            self.history_dir.glob("benchmark_*.json"),
            key=lambda p: p.name,
            reverse=True,
        )

        results: list[dict[str, Any]] = []
        for filepath in files[:max_entries]:
            try:
                with open(filepath, encoding="utf-8") as f:
                    results.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue

        self._history_cache = results
        return results

    def extract_metric_series(
        self,
        metric_path: str,
        history: list[dict[str, Any]] | None = None,
    ) -> list[tuple[str, float]]:
        """Extract time series for a specific metric.

        Args:
            metric_path: Dot-separated path like "pcn-vae-gan.throughput.samples_per_sec_bs32"
            history: Optional pre-loaded history.

        Returns:
            List of (timestamp, value) tuples, oldest first.
        """
        history = history or self.load_history()
        series: list[tuple[str, float]] = []

        parts = metric_path.split(".")
        if len(parts) < 3:
            return series

        component, category, metric = parts[0], parts[1], ".".join(parts[2:])

        for result in reversed(history):  # Oldest first
            timestamp = result.get("timestamp", "unknown")
            components = result.get("components", {})

            if component in components:
                comp_data = components[component]
                if category in comp_data:
                    cat_data = comp_data[category]
                    if metric in cat_data:
                        value = cat_data[metric]
                        if isinstance(value, (int, float)):
                            series.append((timestamp, float(value)))

        return series

    def get_all_metrics(
        self,
        history: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Get all unique metric paths from history.

        Args:
            history: Optional pre-loaded history.

        Returns:
            List of metric paths.
        """
        history = history or self.load_history()
        metrics: set[str] = set()

        for result in history:
            for comp_name, comp_data in result.get("components", {}).items():
                for cat_name, cat_data in comp_data.items():
                    if isinstance(cat_data, dict):
                        for metric_name in cat_data:
                            metrics.add(f"{comp_name}.{cat_name}.{metric_name}")

        return sorted(metrics)

    def analyze_trend(
        self,
        metric_path: str,
        higher_is_better: bool = True,
    ) -> MetricTrend | None:
        """Analyze trend for a specific metric.

        Args:
            metric_path: Dot-separated metric path.
            higher_is_better: Whether higher values are better.

        Returns:
            MetricTrend analysis or None if insufficient data.
        """
        series = self.extract_metric_series(metric_path)
        if len(series) < 2:
            return None

        values = [v for _, v in series]
        timestamps = [t for t, _ in series]

        current = values[-1]
        baseline = values[0]

        if baseline != 0:
            change_pct = ((current - baseline) / baseline) * 100
        else:
            change_pct = 0

        # Determine trend using linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0

        # Normalize slope relative to mean
        slope_pct = (slope / y_mean * 100) if y_mean != 0 else 0

        if abs(slope_pct) < 2:
            trend = "stable"
        elif slope_pct > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        # Is the trend improving?
        is_improving = (slope > 0) == higher_is_better

        # Generate sparkline
        spark = self.sparkline.render(values, width=20)

        return MetricTrend(
            name=metric_path,
            values=list(zip(timestamps, values)),
            current=current,
            baseline=baseline,
            change_pct=change_pct,
            trend=trend,
            sparkline=spark,
            is_improving=is_improving,
            higher_is_better=higher_is_better,
        )

    def generate_trend_report(self) -> TrendReport:
        """Generate comprehensive trend report for all metrics.

        Returns:
            TrendReport with all analyses and recommendations.
        """
        history = self.load_history()
        if not history:
            return TrendReport(
                timestamp=datetime.now(UTC).isoformat(),
                metrics=[],
                summary={"error": "No benchmark history found"},
                recommendations=["Run benchmarks to generate history"],
            )

        metrics: list[MetricTrend] = []
        all_metric_paths = self.get_all_metrics(history)

        # Determine higher_is_better based on metric name
        for path in all_metric_paths:
            higher_is_better = not any(
                pattern in path.lower()
                for pattern in ["latency", "error", "loss", "mse"]
            )

            trend = self.analyze_trend(path, higher_is_better)
            if trend:
                metrics.append(trend)

        # Generate summary
        improving = sum(1 for m in metrics if m.is_improving)
        degrading = sum(1 for m in metrics if not m.is_improving and m.trend != "stable")
        stable = sum(1 for m in metrics if m.trend == "stable")

        summary = {
            "total_metrics": len(metrics),
            "improving": improving,
            "degrading": degrading,
            "stable": stable,
            "history_depth": len(history),
            "time_range": {
                "oldest": history[-1].get("timestamp", "unknown") if history else "N/A",
                "newest": history[0].get("timestamp", "unknown") if history else "N/A",
            },
        }

        # Generate recommendations
        recommendations: list[str] = []

        if degrading > 0:
            degrading_metrics = [m.name for m in metrics if not m.is_improving and m.trend != "stable"]
            recommendations.append(
                f"⚠️ {degrading} metric(s) showing regression: investigate {', '.join(degrading_metrics[:3])}"
            )

        if len(history) < 5:
            recommendations.append(
                "📊 Run more benchmarks to establish reliable baselines (need 5+ samples)"
            )

        if improving > degrading:
            recommendations.append(
                f"✅ Overall positive trend: {improving} metrics improving"
            )

        if stable > len(metrics) // 2:
            recommendations.append(
                "➡️ System is stable - good time to establish new baselines"
            )

        return TrendReport(
            timestamp=datetime.now(UTC).isoformat(),
            metrics=metrics,
            summary=summary,
            recommendations=recommendations,
        )

    def print_trend_report(self) -> None:
        """Print formatted trend report to terminal."""
        report = self.generate_trend_report()

        print("=" * 80)
        print("BENCHMARK TREND REPORT")
        print("=" * 80)
        print(f"Generated: {report.timestamp}")
        print(f"History: {report.summary.get('history_depth', 0)} benchmark runs")
        print()

        if not report.metrics:
            print("No metrics found in history. Run benchmarks first.")
            return

        # Group by component
        by_component: dict[str, list[MetricTrend]] = {}
        for metric in report.metrics:
            component = metric.name.split(".")[0]
            if component not in by_component:
                by_component[component] = []
            by_component[component].append(metric)

        for component, component_metrics in by_component.items():
            print(f"\n┌─ {component.upper()} {'─' * (75 - len(component))}")

            for m in component_metrics:
                # Status indicator
                if m.trend == "stable":
                    status = STATUS_STABLE
                elif m.is_improving:
                    status = STATUS_IMPROVING
                else:
                    status = STATUS_DEGRADING

                # Short metric name (remove component prefix)
                short_name = ".".join(m.name.split(".")[1:])

                print(
                    f"│ {status} {short_name:<45} "
                    f"{m.sparkline} {m.current:>10.2f} ({m.change_pct:+.1f}%)"
                )

            print(f"└{'─' * 78}")

        # Summary
        print("\n" + "─" * 80)
        print("SUMMARY")
        print("─" * 80)
        print(f"  Total metrics tracked: {report.summary['total_metrics']}")
        print(f"  {STATUS_IMPROVING} Improving: {report.summary['improving']}")
        print(f"  {STATUS_DEGRADING} Degrading: {report.summary['degrading']}")
        print(f"  {STATUS_STABLE} Stable: {report.summary['stable']}")

        # Recommendations
        if report.recommendations:
            print("\n" + "─" * 80)
            print("RECOMMENDATIONS")
            print("─" * 80)
            for rec in report.recommendations:
                print(f"  • {rec}")

        print()

    def export_markdown(self, filepath: str | Path) -> Path:
        """Export trend report as Markdown.

        Args:
            filepath: Output file path.

        Returns:
            Path to written file.
        """
        report = self.generate_trend_report()
        filepath = Path(filepath)

        lines = [
            "# Benchmark Trend Report",
            "",
            f"**Generated**: {report.timestamp}",
            f"**History Depth**: {report.summary.get('history_depth', 0)} runs",
            "",
            "## Summary",
            "",
            f"| Status | Count |",
            f"|--------|-------|",
            f"| 📈 Improving | {report.summary.get('improving', 0)} |",
            f"| 📉 Degrading | {report.summary.get('degrading', 0)} |",
            f"| ➡️ Stable | {report.summary.get('stable', 0)} |",
            "",
            "## Metrics by Component",
            "",
        ]

        # Group by component
        by_component: dict[str, list[MetricTrend]] = {}
        for metric in report.metrics:
            component = metric.name.split(".")[0]
            if component not in by_component:
                by_component[component] = []
            by_component[component].append(metric)

        for component, metrics in by_component.items():
            lines.append(f"### {component}")
            lines.append("")
            lines.append("| Metric | Trend | Current | Change | Sparkline |")
            lines.append("|--------|-------|---------|--------|-----------|")

            for m in metrics:
                short_name = ".".join(m.name.split(".")[1:])
                status = "📈" if m.is_improving else ("📉" if m.trend != "stable" else "➡️")
                lines.append(
                    f"| {short_name} | {status} | {m.current:.2f} | "
                    f"{m.change_pct:+.1f}% | `{m.sparkline}` |"
                )

            lines.append("")

        # Recommendations
        if report.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for rec in report.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        content = "\n".join(lines)
        filepath.write_text(content, encoding="utf-8")
        return filepath

    def generate_html_report(self, filepath: str | Path) -> Path | None:
        """Generate interactive HTML report with charts.

        Requires matplotlib for chart generation.
        Returns None if matplotlib not available.

        Args:
            filepath: Output HTML file path.

        Returns:
            Path to written file, or None if unavailable.
        """
        try:
            import base64
            import io

            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - HTML report generation skipped")
            return None

        report = self.generate_trend_report()
        filepath = Path(filepath)

        # Generate charts as base64-encoded PNGs
        charts: list[str] = []

        # Group by component
        by_component: dict[str, list[MetricTrend]] = {}
        for metric in report.metrics:
            component = metric.name.split(".")[0]
            if component not in by_component:
                by_component[component] = []
            by_component[component].append(metric)

        for component, metrics in by_component.items():
            if not metrics:
                continue

            # Create a figure with subplots for each metric
            n_metrics = min(len(metrics), 6)  # Limit to 6 per component
            fig, axes = plt.subplots(
                n_metrics, 1, figsize=(10, 2 * n_metrics), squeeze=False
            )

            for idx, m in enumerate(metrics[:n_metrics]):
                ax = axes[idx, 0]
                values = [v for _, v in m.values]
                ax.plot(values, marker="o", linewidth=2, markersize=4)
                ax.fill_between(range(len(values)), values, alpha=0.3)
                ax.set_title(f"{m.name} ({m.change_pct:+.1f}%)", fontsize=10)
                ax.set_ylabel("Value")
                ax.grid(True, alpha=0.3)

                # Add trend line
                n = len(values)
                if n >= 2:
                    z = [i for i in range(n)]
                    coeffs = [
                        sum((i - (n - 1) / 2) * (v - sum(values) / n) for i, v in enumerate(values))
                        / sum((i - (n - 1) / 2) ** 2 for i in range(n))
                        if sum((i - (n - 1) / 2) ** 2 for i in range(n)) != 0
                        else 0,
                        sum(values) / n,
                    ]
                    trend_y = [coeffs[0] * i + coeffs[1] for i in range(n)]
                    ax.plot(trend_y, "--", color="red", alpha=0.5, label="Trend")

            plt.suptitle(f"{component.upper()} Metrics", fontsize=14, fontweight="bold")
            plt.tight_layout()

            # Convert to base64
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            charts.append(f'<img src="data:image/png;base64,{img_base64}" alt="{component} metrics">')

        # Build HTML
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Trend Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .chart {{ margin: 20px 0; }}
        .chart img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
        .recommendations {{ background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; }}
        .metric-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .metric-table th, .metric-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .metric-table th {{ background: #f8f9fa; }}
        .improving {{ color: #28a745; }}
        .degrading {{ color: #dc3545; }}
        .stable {{ color: #6c757d; }}
    </style>
</head>
<body>
    <h1>📊 Benchmark Trend Report</h1>
    <p><strong>Generated:</strong> {report.timestamp}</p>
    <p><strong>History Depth:</strong> {report.summary.get('history_depth', 0)} benchmark runs</p>

    <div class="summary">
        <h2>Summary</h2>
        <p>📈 <strong>Improving:</strong> {report.summary.get('improving', 0)} metrics</p>
        <p>📉 <strong>Degrading:</strong> {report.summary.get('degrading', 0)} metrics</p>
        <p>➡️ <strong>Stable:</strong> {report.summary.get('stable', 0)} metrics</p>
    </div>

    {''.join(f'<div class="chart">{chart}</div>' for chart in charts)}

    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
            {''.join(f'<li>{rec}</li>' for rec in report.recommendations)}
        </ul>
    </div>

    <h2>Detailed Metrics</h2>
    <table class="metric-table">
        <thead>
            <tr>
                <th>Metric</th>
                <th>Trend</th>
                <th>Current</th>
                <th>Baseline</th>
                <th>Change</th>
            </tr>
        </thead>
        <tbody>
            {''.join(
                f'''<tr>
                    <td>{m.name}</td>
                    <td class="{'improving' if m.is_improving else 'degrading' if m.trend != 'stable' else 'stable'}">
                        {'📈' if m.is_improving else '📉' if m.trend != 'stable' else '➡️'} {m.trend}
                    </td>
                    <td>{m.current:.4f}</td>
                    <td>{m.baseline:.4f}</td>
                    <td>{m.change_pct:+.2f}%</td>
                </tr>'''
                for m in report.metrics
            )}
        </tbody>
    </table>
</body>
</html>"""

        filepath.write_text(html_content, encoding="utf-8")
        return filepath


def main() -> None:
    """CLI entry point for visualization."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark visualization and trends")
    parser.add_argument(
        "--format",
        choices=["terminal", "markdown", "html"],
        default="terminal",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (for markdown/html)",
    )

    args = parser.parse_args()
    viz = BenchmarkVisualizer()

    if args.format == "terminal":
        viz.print_trend_report()
    elif args.format == "markdown":
        output = args.output or Path("BENCHMARK_TRENDS.md")
        viz.export_markdown(output)
        print(f"Markdown report saved to: {output}")
    elif args.format == "html":
        output = args.output or Path("benchmark_report.html")
        result = viz.generate_html_report(output)
        if result:
            print(f"HTML report saved to: {result}")


if __name__ == "__main__":
    main()
