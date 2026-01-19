"""Industry Model Comparison Framework.

Compares CogSynDelta's measured performance against published industry baselines.
All industry numbers are from published papers/benchmarks with citations.

Per constitution: "All performance claims must be backed by evidence."
This module enables honest comparison - we show the data as-is, including
when CogSynDelta performs worse than industry models.

Example:
    $ uv run python -m benchmarks.model_comparisons
    $ uv run python -m benchmarks.model_comparisons --format markdown
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "ComparisonResult",
    "IndustryBaseline",
    "ModelComparisonFramework",
    "generate_comparison_report",
]

# Paths
DATA_DIR = Path(__file__).parent / "data"
BASELINE_FILE = DATA_DIR / "industry_baselines.json"


@dataclass
class IndustryBaseline:
    """Industry model baseline data.

    Attributes:
        name: Model name.
        organization: Organization that created the model.
        parameters: Total parameter count.
        metrics: Dict of metric name to value.
        hardware: Hardware used for benchmark.
        source: Source URL for the data.
        publication_date: When the benchmark was published.
        notes: Additional notes.
    """

    name: str
    organization: str
    parameters: int
    metrics: dict[str, float]
    hardware: str
    source: str
    publication_date: str
    notes: str = ""

    @property
    def parameters_billions(self) -> float:
        """Get parameters in billions."""
        return self.parameters / 1_000_000_000

    @property
    def parameters_millions(self) -> float:
        """Get parameters in millions."""
        return self.parameters / 1_000_000


@dataclass
class ComparisonResult:
    """Result of comparing CogSynDelta to an industry baseline.

    Attributes:
        metric_name: Name of the metric being compared.
        cogsyndelta_value: CogSynDelta's measured value.
        baseline_value: Industry baseline value.
        delta: Absolute difference (cogsyndelta - baseline).
        delta_pct: Percentage difference.
        cogsyndelta_better: Whether CogSynDelta performed better.
        baseline_model: Name of the baseline model.
        notes: Comparison notes (e.g., hardware differences).
    """

    metric_name: str
    cogsyndelta_value: float
    baseline_value: float
    delta: float
    delta_pct: float
    cogsyndelta_better: bool
    baseline_model: str
    notes: str = ""

    @property
    def direction_indicator(self) -> str:
        """Get visual indicator for comparison direction."""
        if abs(self.delta_pct) < 5:
            return "↔"  # Within 5%, essentially equal
        return "↑" if self.cogsyndelta_better else "↓"


class ModelComparisonFramework:
    """Framework for comparing CogSynDelta against industry baselines.

    Loads industry baseline data and provides comparison methods.
    All comparisons include methodology notes for transparency.
    """

    def __init__(self, baseline_file: Path | None = None) -> None:
        """Initialize comparison framework.

        Args:
            baseline_file: Path to industry baselines JSON. Uses default if None.
        """
        self.baseline_file = baseline_file or BASELINE_FILE
        self._baselines: dict[str, IndustryBaseline] = {}
        self._compression_baselines: dict[str, dict[str, Any]] = {}
        self._load_baselines()

    def _load_baselines(self) -> None:
        """Load industry baselines from JSON file."""
        if not self.baseline_file.exists():
            print(f"Warning: Baseline file not found: {self.baseline_file}")
            return

        with open(self.baseline_file, encoding="utf-8") as f:
            data = json.load(f)

        # Load model baselines
        for model_id, model_data in data.get("models", {}).items():
            self._baselines[model_id] = IndustryBaseline(
                name=model_data["name"],
                organization=model_data["organization"],
                parameters=model_data["parameters"],
                metrics=model_data["metrics"],
                hardware=model_data["hardware"],
                source=model_data["source"],
                publication_date=model_data["publication_date"],
                notes=model_data.get("notes", ""),
            )

        # Load compression baselines
        self._compression_baselines = data.get("compression_baselines", {})

    def get_baseline(self, model_id: str) -> IndustryBaseline | None:
        """Get a specific industry baseline.

        Args:
            model_id: Model identifier (e.g., "gpt2-small", "bert-base").

        Returns:
            IndustryBaseline if found, None otherwise.
        """
        return self._baselines.get(model_id)

    def list_baselines(self) -> list[str]:
        """List all available baseline model IDs.

        Returns:
            List of model IDs.
        """
        return list(self._baselines.keys())

    def compare_throughput(
        self,
        cogsyndelta_throughput: float,
        cogsyndelta_params: int,
        higher_is_better: bool = True,
    ) -> list[ComparisonResult]:
        """Compare CogSynDelta throughput against industry baselines.

        Args:
            cogsyndelta_throughput: CogSynDelta's measured throughput (samples/sec).
            cogsyndelta_params: CogSynDelta's parameter count.
            higher_is_better: Whether higher throughput is better.

        Returns:
            List of ComparisonResult for each relevant baseline.
        """
        results: list[ComparisonResult] = []

        for model_id, baseline in self._baselines.items():
            # Find throughput-like metrics
            for metric_name, value in baseline.metrics.items():
                if "throughput" not in metric_name.lower():
                    continue

                delta = cogsyndelta_throughput - value
                delta_pct = (delta / value * 100) if value != 0 else 0

                results.append(
                    ComparisonResult(
                        metric_name=metric_name,
                        cogsyndelta_value=cogsyndelta_throughput,
                        baseline_value=value,
                        delta=delta,
                        delta_pct=delta_pct,
                        cogsyndelta_better=(delta > 0) == higher_is_better,
                        baseline_model=baseline.name,
                        notes=f"Hardware: {baseline.hardware}. CogSynDelta: {cogsyndelta_params / 1e6:.1f}M params vs {baseline.parameters_millions:.1f}M params",
                    )
                )

        return results

    def compare_memory(
        self,
        cogsyndelta_memory_mb: float,
        cogsyndelta_params: int,
    ) -> list[ComparisonResult]:
        """Compare CogSynDelta memory usage against industry baselines.

        Args:
            cogsyndelta_memory_mb: CogSynDelta's memory usage in MB.
            cogsyndelta_params: CogSynDelta's parameter count.

        Returns:
            List of ComparisonResult for each relevant baseline.
        """
        results: list[ComparisonResult] = []

        for model_id, baseline in self._baselines.items():
            # Find memory-like metrics
            for metric_name, value in baseline.metrics.items():
                if "memory" not in metric_name.lower():
                    continue

                delta = cogsyndelta_memory_mb - value
                delta_pct = (delta / value * 100) if value != 0 else 0

                # Lower memory is better
                results.append(
                    ComparisonResult(
                        metric_name=metric_name,
                        cogsyndelta_value=cogsyndelta_memory_mb,
                        baseline_value=value,
                        delta=delta,
                        delta_pct=delta_pct,
                        cogsyndelta_better=delta < 0,
                        baseline_model=baseline.name,
                        notes=f"CogSynDelta: {cogsyndelta_params / 1e6:.1f}M params vs {baseline.parameters_millions:.1f}M params",
                    )
                )

        return results

    def compare_compression(
        self,
        cogsyndelta_ratio: float,
        cogsyndelta_fidelity: float,
    ) -> list[dict[str, Any]]:
        """Compare CogSynDelta compression against baselines.

        Args:
            cogsyndelta_ratio: CogSynDelta's compression ratio.
            cogsyndelta_fidelity: CogSynDelta's reconstruction fidelity (cosine sim).

        Returns:
            List of comparison dicts with baseline data.
        """
        results: list[dict[str, Any]] = []

        for baseline_id, baseline in self._compression_baselines.items():
            comparison: dict[str, Any] = {
                "baseline_name": baseline["name"],
                "baseline_description": baseline.get("description", ""),
                "baseline_source": baseline.get("source", ""),
            }

            # Compare compression ratio
            baseline_ratio = baseline.get("compression_ratio", 0)
            if baseline_ratio > 0:
                comparison["compression_ratio"] = {
                    "cogsyndelta": cogsyndelta_ratio,
                    "baseline": baseline_ratio,
                    "cogsyndelta_better": cogsyndelta_ratio > baseline_ratio,
                }

            # Compare fidelity (recall or cosine similarity)
            baseline_recall = baseline.get("recall_at_1") or baseline.get(
                "reconstruction_cosine_similarity", 0
            )
            if baseline_recall > 0:
                comparison["fidelity"] = {
                    "cogsyndelta": cogsyndelta_fidelity,
                    "baseline": baseline_recall,
                    "cogsyndelta_better": cogsyndelta_fidelity > baseline_recall,
                }

            results.append(comparison)

        return results

    def generate_normalized_comparison_table(
        self,
        cogsyndelta_metrics: dict[str, Any],
    ) -> str:
        """Generate a normalized comparison table in markdown.

        Shows both raw and normalized (per-param, per-FLOP) metrics.

        Args:
            cogsyndelta_metrics: Dict with CogSynDelta measurements.

        Returns:
            Markdown-formatted comparison table.
        """
        lines: list[str] = []

        # Header
        lines.append("## Model Comparison: CogSynDelta vs Industry Baselines")
        lines.append("")
        lines.append(f"*Generated: {datetime.now(UTC).isoformat()}*")
        lines.append("")
        lines.append(
            "> **Methodology**: All industry numbers from published sources (see Source column)."
        )
        lines.append("> Hardware varies by model - direct comparisons should account for this.")
        lines.append("> CogSynDelta measured on: RTX 5080 (16GB)")
        lines.append("")

        # Throughput comparison table
        csd_throughput = cogsyndelta_metrics.get("throughput_samples_per_sec", 0)
        csd_params = cogsyndelta_metrics.get("param_count", 0)

        lines.append("### Throughput Comparison")
        lines.append("")
        lines.append(
            "| Model | Params (M) | Throughput (samples/sec) | Per-Billion-Params | Source |"
        )
        lines.append(
            "|-------|------------|--------------------------|-------------------|--------|"
        )

        # CogSynDelta row
        csd_per_billion = csd_throughput / (csd_params / 1e9) if csd_params > 0 else 0
        lines.append(
            f"| **CogSynDelta** | {csd_params / 1e6:.1f} | {csd_throughput:.0f} | {csd_per_billion:.0f} | This benchmark |"
        )

        # Industry baselines
        for model_id, baseline in self._baselines.items():
            throughput = None
            for metric_name, value in baseline.metrics.items():
                if "throughput" in metric_name.lower() and "sample" in metric_name.lower():
                    throughput = value
                    break

            if throughput is not None:
                per_billion = throughput / baseline.parameters_billions
                lines.append(
                    f"| {baseline.name} | {baseline.parameters_millions:.0f} | {throughput:.0f} | {per_billion:.0f} | [Link]({baseline.source}) |"
                )

        lines.append("")

        # Memory comparison table
        csd_memory = cogsyndelta_metrics.get("memory_mb", 0)

        lines.append("### Memory Comparison")
        lines.append("")
        lines.append("| Model | Params (M) | Memory (MB) | Bytes/Param | Source |")
        lines.append("|-------|------------|-------------|-------------|--------|")

        # CogSynDelta row
        csd_bytes_per = (csd_memory * 1e6) / csd_params if csd_params > 0 else 0
        lines.append(
            f"| **CogSynDelta** | {csd_params / 1e6:.1f} | {csd_memory:.0f} | {csd_bytes_per:.1f} | This benchmark |"
        )

        # Industry baselines
        for model_id, baseline in self._baselines.items():
            memory = baseline.metrics.get("memory_inference_mb")
            if memory is not None:
                bytes_per = (memory * 1e6) / baseline.parameters if baseline.parameters > 0 else 0
                lines.append(
                    f"| {baseline.name} | {baseline.parameters_millions:.0f} | {memory:.0f} | {bytes_per:.1f} | [Link]({baseline.source}) |"
                )

        lines.append("")

        # Compression comparison (if applicable)
        csd_compression_ratio = cogsyndelta_metrics.get("compression_ratio")
        csd_fidelity = cogsyndelta_metrics.get("compression_fidelity")

        if csd_compression_ratio is not None:
            lines.append("### Compression Comparison")
            lines.append("")
            lines.append("| Method | Compression Ratio | Fidelity (cosine/recall@1) | Source |")
            lines.append("|--------|-------------------|---------------------------|--------|")

            lines.append(
                f"| **CogSynDelta** | {csd_compression_ratio}x | {csd_fidelity:.3f} | This benchmark |"
            )

            for baseline_id, baseline in self._compression_baselines.items():
                ratio = baseline.get("compression_ratio", "N/A")
                fidelity = baseline.get("recall_at_1") or baseline.get(
                    "reconstruction_cosine_similarity", "N/A"
                )
                source = baseline.get("source", "N/A")
                lines.append(f"| {baseline['name']} | {ratio}x | {fidelity} | [Link]({source}) |")

            lines.append("")

        # Honest assessment section
        lines.append("### Honest Assessment")
        lines.append("")
        lines.append(
            "> **Note**: This comparison is informational, not competitive. CogSynDelta is a"
        )
        lines.append(
            "> hybrid PCN-VAE-GAN architecture optimized for different use cases than pure"
        )
        lines.append(
            "> language models (GPT, LLaMA) or embedding models (BERT, Sentence-Transformers)."
        )
        lines.append(
            "> Direct comparisons should consider architectural differences and intended use."
        )
        lines.append("")

        return "\n".join(lines)


def generate_comparison_report(
    cogsyndelta_metrics: dict[str, Any],
    output_format: str = "markdown",
    output_path: Path | None = None,
) -> str:
    """Generate a comparison report.

    Args:
        cogsyndelta_metrics: Dict with CogSynDelta measurements.
        output_format: "markdown" or "json".
        output_path: Optional path to save report.

    Returns:
        Report as string.
    """
    framework = ModelComparisonFramework()

    if output_format == "markdown":
        report = framework.generate_normalized_comparison_table(cogsyndelta_metrics)
    else:
        # JSON format with structured comparisons
        throughput_comparisons = framework.compare_throughput(
            cogsyndelta_metrics.get("throughput_samples_per_sec", 0),
            cogsyndelta_metrics.get("param_count", 0),
        )
        memory_comparisons = framework.compare_memory(
            cogsyndelta_metrics.get("memory_mb", 0),
            cogsyndelta_metrics.get("param_count", 0),
        )

        report_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "cogsyndelta_metrics": cogsyndelta_metrics,
            "throughput_comparisons": [
                {
                    "baseline": c.baseline_model,
                    "metric": c.metric_name,
                    "cogsyndelta": c.cogsyndelta_value,
                    "baseline_value": c.baseline_value,
                    "delta_pct": c.delta_pct,
                    "better": c.cogsyndelta_better,
                }
                for c in throughput_comparisons
            ],
            "memory_comparisons": [
                {
                    "baseline": c.baseline_model,
                    "metric": c.metric_name,
                    "cogsyndelta": c.cogsyndelta_value,
                    "baseline_value": c.baseline_value,
                    "delta_pct": c.delta_pct,
                    "better": c.cogsyndelta_better,
                }
                for c in memory_comparisons
            ],
        }
        report = json.dumps(report_data, indent=2)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved report to: {output_path}")

    return report


def main() -> None:
    """CLI entry point for model comparisons."""
    parser = argparse.ArgumentParser(
        description="Compare CogSynDelta against industry model baselines",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path",
    )
    parser.add_argument(
        "--list-baselines",
        action="store_true",
        help="List available industry baselines",
    )

    args = parser.parse_args()

    framework = ModelComparisonFramework()

    if args.list_baselines:
        print("Available industry baselines:")
        for model_id in framework.list_baselines():
            baseline = framework.get_baseline(model_id)
            if baseline:
                print(f"  {model_id}: {baseline.name} ({baseline.parameters_millions:.0f}M params)")
        return

    # Example metrics - in practice, these come from model_benchmarks.py
    # Using placeholder values for demonstration
    example_metrics = {
        "throughput_samples_per_sec": 5000,
        "param_count": 50_000_000,
        "memory_mb": 200,
        "compression_ratio": 2,
        "compression_fidelity": 0.67,
    }

    print("Note: Using example metrics. Run model_benchmarks.py first for real data.")
    print()

    report = generate_comparison_report(
        example_metrics,
        output_format=args.format,
        output_path=args.output,
    )

    print(report)


if __name__ == "__main__":
    main()
