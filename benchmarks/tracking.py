"""Benchmark Tracking and Delta Analysis.

Tracks benchmark results over time, computes deltas from baseline,
and provides stability/trend analysis.

Per constitution: Enables data-driven development and regression detection.

Example:
    >>> from benchmarks.tracking import BenchmarkTracker
    >>> tracker = BenchmarkTracker()
    >>> comparison = tracker.compare_to_baseline(current_results)
    >>> print(tracker.stability_report())
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "MetricDelta",
    "BaselineComparison",
    "StabilityReport",
    "BenchmarkTracker",
]

# Paths
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark_results"
BASELINE_DIR = BENCHMARK_DIR / "baselines"
HISTORY_DIR = BENCHMARK_DIR / "history"

# Thresholds
REGRESSION_THRESHOLD_PCT = 5.0  # Warn if metric regresses >5%
SIGNIFICANT_CHANGE_PCT = 10.0  # Flag significant changes
STABILITY_MIN_SAMPLES = 5  # Minimum samples for stability analysis


@dataclass
class MetricDelta:
    """Delta between current and baseline metric.

    Attributes:
        metric_name: Name of the metric.
        current: Current measured value.
        baseline: Baseline value.
        delta: Absolute difference (current - baseline).
        delta_pct: Percentage difference.
        is_regression: Whether this represents a regression.
        is_significant: Whether change exceeds significance threshold.
        direction: Visual indicator (↑ better, ↓ worse, ↔ stable).
    """

    metric_name: str
    current: float
    baseline: float
    delta: float
    delta_pct: float
    is_regression: bool
    is_significant: bool
    direction: str
    higher_is_better: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "current": self.current,
            "baseline": self.baseline,
            "delta": self.delta,
            "delta_pct": self.delta_pct,
            "is_regression": self.is_regression,
            "is_significant": self.is_significant,
            "direction": self.direction,
        }


@dataclass
class BaselineComparison:
    """Complete comparison between current results and baseline.

    Attributes:
        timestamp: When comparison was made.
        baseline_timestamp: When baseline was captured.
        baseline_git_sha: Git SHA of baseline.
        current_git_sha: Current git SHA.
        deltas: List of MetricDelta for each metric.
        has_regressions: Whether any regressions detected.
        summary: Summary statistics.
    """

    timestamp: str
    baseline_timestamp: str
    baseline_git_sha: str
    current_git_sha: str
    deltas: list[MetricDelta]
    has_regressions: bool
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "baseline_timestamp": self.baseline_timestamp,
            "baseline_git_sha": self.baseline_git_sha,
            "current_git_sha": self.current_git_sha,
            "deltas": [d.to_dict() for d in self.deltas],
            "has_regressions": self.has_regressions,
            "summary": self.summary,
        }

    def print_report(self) -> None:
        """Print human-readable comparison report."""
        print("=" * 60)
        print("BASELINE COMPARISON REPORT")
        print("=" * 60)
        print(f"Current: {self.current_git_sha} ({self.timestamp})")
        print(f"Baseline: {self.baseline_git_sha} ({self.baseline_timestamp})")
        print()

        if self.has_regressions:
            print("⚠️  REGRESSIONS DETECTED")
            print()

        print(f"{'Metric':<40} {'Current':>12} {'Baseline':>12} {'Delta':>12}")
        print("-" * 80)

        for delta in self.deltas:
            flag = "⚠️" if delta.is_regression else "  "
            print(
                f"{flag}{delta.metric_name:<38} {delta.current:>12.2f} "
                f"{delta.baseline:>12.2f} {delta.direction} {delta.delta_pct:>+.1f}%"
            )

        print()


@dataclass
class StabilityReport:
    """Stability analysis across multiple benchmark runs.

    Attributes:
        metric_name: Name of the metric.
        n_samples: Number of samples analyzed.
        mean: Mean value.
        std: Standard deviation.
        min: Minimum value.
        max: Maximum value.
        cv: Coefficient of variation (std/mean).
        trend: Trend direction based on recent values.
        is_stable: Whether metric is considered stable (CV < 10%).
    """

    metric_name: str
    n_samples: int
    mean: float
    std: float
    min: float
    max: float
    cv: float  # Coefficient of variation
    trend: str  # "improving", "degrading", "stable"
    is_stable: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "n_samples": self.n_samples,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "cv": self.cv,
            "trend": self.trend,
            "is_stable": self.is_stable,
        }


class BenchmarkTracker:
    """Tracks benchmark results over time with delta and stability analysis.

    Provides:
    - Baseline comparison with regression detection
    - Historical trend analysis
    - Stability metrics (mean, std, CV)
    """

    def __init__(
        self,
        baseline_dir: Path | None = None,
        history_dir: Path | None = None,
    ) -> None:
        """Initialize tracker.

        Args:
            baseline_dir: Directory for baseline files.
            history_dir: Directory for historical results.
        """
        self.baseline_dir = baseline_dir or BASELINE_DIR
        self.history_dir = history_dir or HISTORY_DIR

    def load_baseline(self) -> dict[str, Any] | None:
        """Load current baseline.

        Returns:
            Baseline dict if exists, None otherwise.
        """
        baseline_path = self.baseline_dir / "model_baseline.json"
        if not baseline_path.exists():
            return None

        with open(baseline_path, encoding="utf-8") as f:
            return json.load(f)

    def load_history(self, max_entries: int = 50) -> list[dict[str, Any]]:
        """Load historical benchmark results.

        Args:
            max_entries: Maximum number of entries to load.

        Returns:
            List of benchmark result dicts, newest first.
        """
        if not self.history_dir.exists():
            return []

        history_files = sorted(
            self.history_dir.glob("benchmark_*.json"),
            key=lambda p: p.name,
            reverse=True,
        )

        results: list[dict[str, Any]] = []
        for filepath in history_files[:max_entries]:
            try:
                with open(filepath, encoding="utf-8") as f:
                    results.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue

        return results

    def compare_to_baseline(
        self,
        current_results: dict[str, Any],
        higher_is_better_metrics: set[str] | None = None,
    ) -> BaselineComparison | None:
        """Compare current results to baseline.

        Args:
            current_results: Current benchmark results dict.
            higher_is_better_metrics: Set of metric names where higher is better.
                Defaults to throughput-like metrics.

        Returns:
            BaselineComparison if baseline exists, None otherwise.
        """
        baseline = self.load_baseline()
        if baseline is None:
            print("No baseline found. Save current results as baseline first.")
            return None

        if higher_is_better_metrics is None:
            higher_is_better_metrics = {
                "throughput",
                "samples_per_sec",
                "images_per_sec",
                "cosine_similarity",
            }

        deltas: list[MetricDelta] = []
        has_regressions = False

        # Extract and compare metrics from components
        current_components = current_results.get("components", {})
        baseline_components = baseline.get("components", {})

        for component_name, current_metrics in current_components.items():
            baseline_metrics = baseline_components.get(component_name, {})

            # Compare throughput metrics
            for metric_category in ["throughput", "latency_ms", "quality"]:
                current_category = current_metrics.get(metric_category, {})
                baseline_category = baseline_metrics.get(metric_category, {})

                for metric_name, current_value in current_category.items():
                    if metric_name not in baseline_category:
                        continue

                    baseline_value = baseline_category[metric_name]
                    if not isinstance(current_value, (int, float)) or not isinstance(
                        baseline_value, (int, float)
                    ):
                        continue

                    # Determine if higher is better
                    higher_better = any(
                        pattern in metric_name.lower()
                        for pattern in higher_is_better_metrics
                    )

                    # For latency, lower is better
                    if "latency" in metric_name.lower():
                        higher_better = False

                    delta_val = current_value - baseline_value
                    delta_pct = (
                        (delta_val / baseline_value * 100)
                        if baseline_value != 0
                        else 0
                    )

                    # Determine if this is a regression
                    if higher_better:
                        is_regression = delta_pct < -REGRESSION_THRESHOLD_PCT
                    else:
                        is_regression = delta_pct > REGRESSION_THRESHOLD_PCT

                    is_significant = abs(delta_pct) > SIGNIFICANT_CHANGE_PCT

                    # Direction indicator
                    if abs(delta_pct) < REGRESSION_THRESHOLD_PCT:
                        direction = "↔"
                    elif (delta_val > 0) == higher_better:
                        direction = "↑"
                    else:
                        direction = "↓"

                    if is_regression:
                        has_regressions = True

                    full_metric_name = f"{component_name}.{metric_category}.{metric_name}"
                    deltas.append(
                        MetricDelta(
                            metric_name=full_metric_name,
                            current=current_value,
                            baseline=baseline_value,
                            delta=delta_val,
                            delta_pct=delta_pct,
                            is_regression=is_regression,
                            is_significant=is_significant,
                            direction=direction,
                            higher_is_better=higher_better,
                        )
                    )

        # Summary
        summary = {
            "total_metrics": len(deltas),
            "regressions": sum(1 for d in deltas if d.is_regression),
            "improvements": sum(
                1 for d in deltas if d.is_significant and not d.is_regression
            ),
            "stable": sum(1 for d in deltas if not d.is_significant),
        }

        return BaselineComparison(
            timestamp=datetime.now(timezone.utc).isoformat(),
            baseline_timestamp=baseline.get("timestamp", "unknown"),
            baseline_git_sha=baseline.get("git_sha", "unknown"),
            current_git_sha=current_results.get("git_sha", "unknown"),
            deltas=deltas,
            has_regressions=has_regressions,
            summary=summary,
        )

    def stability_report(
        self,
        metric_pattern: str | None = None,
    ) -> list[StabilityReport]:
        """Generate stability report from historical data.

        Args:
            metric_pattern: Optional pattern to filter metrics.

        Returns:
            List of StabilityReport for each metric.
        """
        history = self.load_history()
        if len(history) < STABILITY_MIN_SAMPLES:
            print(
                f"Insufficient history ({len(history)} samples). "
                f"Need at least {STABILITY_MIN_SAMPLES} for stability analysis."
            )
            return []

        # Collect metric values over time
        metric_values: dict[str, list[float]] = {}

        for result in history:
            for component_name, component_data in result.get("components", {}).items():
                for category in ["throughput", "latency_ms", "quality"]:
                    for metric_name, value in component_data.get(category, {}).items():
                        if not isinstance(value, (int, float)):
                            continue

                        full_name = f"{component_name}.{category}.{metric_name}"
                        if metric_pattern and metric_pattern not in full_name:
                            continue

                        if full_name not in metric_values:
                            metric_values[full_name] = []
                        metric_values[full_name].append(value)

        # Generate stability reports
        reports: list[StabilityReport] = []

        for metric_name, values in metric_values.items():
            if len(values) < STABILITY_MIN_SAMPLES:
                continue

            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            cv = (std_val / mean_val * 100) if mean_val != 0 else 0

            # Simple trend analysis: compare first half to second half
            mid = len(values) // 2
            first_half_mean = statistics.mean(values[:mid]) if mid > 0 else mean_val
            second_half_mean = statistics.mean(values[mid:])

            pct_change = (
                (second_half_mean - first_half_mean) / first_half_mean * 100
                if first_half_mean != 0
                else 0
            )

            if pct_change > 5:
                trend = "improving"
            elif pct_change < -5:
                trend = "degrading"
            else:
                trend = "stable"

            reports.append(
                StabilityReport(
                    metric_name=metric_name,
                    n_samples=len(values),
                    mean=mean_val,
                    std=std_val,
                    min=min(values),
                    max=max(values),
                    cv=cv,
                    trend=trend,
                    is_stable=cv < 10,  # CV < 10% is considered stable
                )
            )

        return reports

    def print_stability_report(self) -> None:
        """Print human-readable stability report."""
        reports = self.stability_report()

        if not reports:
            return

        print("=" * 80)
        print("BENCHMARK STABILITY REPORT")
        print("=" * 80)
        print()
        print(f"{'Metric':<50} {'Mean':>10} {'Std':>10} {'CV%':>8} {'Trend':>12}")
        print("-" * 90)

        for report in reports:
            stability_flag = "✓" if report.is_stable else "⚠️"
            print(
                f"{stability_flag} {report.metric_name:<48} {report.mean:>10.2f} "
                f"{report.std:>10.2f} {report.cv:>7.1f}% {report.trend:>12}"
            )

        print()
        stable_count = sum(1 for r in reports if r.is_stable)
        print(f"Stable metrics: {stable_count}/{len(reports)}")


def main() -> None:
    """CLI for benchmark tracking."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark tracking and analysis")
    parser.add_argument(
        "--stability",
        action="store_true",
        help="Generate stability report from history",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        help="Compare a results file to baseline",
    )

    args = parser.parse_args()

    tracker = BenchmarkTracker()

    if args.stability:
        tracker.print_stability_report()
    elif args.compare:
        with open(args.compare, encoding="utf-8") as f:
            results = json.load(f)
        comparison = tracker.compare_to_baseline(results)
        if comparison:
            comparison.print_report()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
