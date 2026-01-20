"""Normalized metrics for cross-architecture comparison.

This module provides normalized metrics that enable apples-to-apples
comparison between CogSynDelta and other ML architectures (transformers,
CNNs, etc.).

Classes:
    NormalizedMetrics: Speedup ratios, normalized throughput, equivalent tokens

Why normalization:
    Raw throughput numbers (samples/sec) are meaningless for comparison
    because architectures process different amounts of information per sample.
    Normalized metrics account for model size, quality, and information content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmarks.metrics.base import METRIC_REGISTRY, MetricMixin

# Reference baselines for normalization
# Updated periodically from industry benchmarks
REFERENCE_BASELINES: dict[str, dict[str, float]] = {
    "gpt2_small": {
        "params_m": 124,
        "throughput_tokens_sec": 50000,
        "latency_ms": 20,
        "quality_score": 0.75,
    },
    "gpt2_medium": {
        "params_m": 355,
        "throughput_tokens_sec": 25000,
        "latency_ms": 40,
        "quality_score": 0.80,
    },
    "gpt2_large": {
        "params_m": 774,
        "throughput_tokens_sec": 12000,
        "latency_ms": 80,
        "quality_score": 0.85,
    },
    "llama_7b": {
        "params_m": 7000,
        "throughput_tokens_sec": 2000,
        "latency_ms": 500,
        "quality_score": 0.90,
    },
    "vit_base": {
        "params_m": 86,
        "throughput_samples_sec": 3000,
        "latency_ms": 10,
        "quality_score": 0.82,
    },
    "resnet50": {
        "params_m": 25,
        "throughput_samples_sec": 5000,
        "latency_ms": 5,
        "quality_score": 0.76,
    },
}


@dataclass
class NormalizedMetrics(MetricMixin):
    """Normalized metrics for cross-architecture comparison.

    All metrics are normalized to enable fair comparison between
    architectures with different sizes, modalities, and information
    representations.

    Attributes:
        speedup_vs_baseline: Performance ratio vs baseline model.
        relative_throughput: Throughput as percentage of baseline.
        efficiency_vs_baseline: Efficiency ratio (throughput/param) vs baseline.
        throughput_per_param: Throughput normalized by model size.
        equivalent_tokens_per_sec: CogSynDelta throughput in transformer-equivalent units.
        normalized_quality: Quality score normalized to 0-1.
        normalized_efficiency: Combined throughput*quality/params score.
        baseline_model: Name of reference model used for normalization.
        normalization_version: Version of normalization constants.

    Why these specific metrics:
        - speedup_vs_baseline: Direct comparison with known systems
        - throughput_per_param: Size-agnostic efficiency
        - equivalent_tokens_per_sec: Cross-modality comparison
        - normalized_quality: Ensures we don't sacrifice quality for speed
    """

    speedup_vs_baseline: float | None = None
    relative_throughput: float | None = None
    efficiency_vs_baseline: float | None = None
    throughput_per_param: float | None = None
    equivalent_tokens_per_sec: float | None = None
    normalized_quality: float | None = None
    normalized_efficiency: float | None = None
    baseline_model: str | None = None
    normalization_version: str = field(default="v1.0")

    @classmethod
    def compute(
        cls,
        throughput: float,
        params_millions: float,
        quality_score: float | None = None,
        baseline: str = "gpt2_small",
        engrams_per_sec: float | None = None,
        engram_to_token_ratio: float = 10.0,
    ) -> NormalizedMetrics:
        """Compute normalized comparison metrics.

        Args:
            throughput: Raw throughput (samples/sec or engrams/sec).
            params_millions: Model size in millions of parameters.
            quality_score: Task quality score 0-1 (optional).
            baseline: Reference model name for comparison.
            engrams_per_sec: CogSynDelta engram throughput (optional).
            engram_to_token_ratio: Engrams-to-tokens conversion ratio.
                Default 10.0 means 1 engram ≈ 10 tokens of information.

        Returns:
            NormalizedMetrics instance with computed values.

        Example:
            >>> metrics = NormalizedMetrics.compute(
            ...     throughput=25000,
            ...     params_millions=100,
            ...     quality_score=0.88,
            ...     baseline="gpt2_small"
            ... )
            >>> print(f"Speedup: {metrics.speedup_vs_baseline:.2f}x")
        """
        baseline_data = REFERENCE_BASELINES.get(baseline, REFERENCE_BASELINES["gpt2_small"])
        baseline_throughput = baseline_data.get(
            "throughput_tokens_sec",
            baseline_data.get("throughput_samples_sec", 10000),
        )
        baseline_params = baseline_data["params_m"]
        baseline_quality = baseline_data.get("quality_score", 0.8)

        # Speedup vs baseline
        speedup = throughput / baseline_throughput if baseline_throughput > 0 else None

        # Relative throughput (percentage)
        relative = (throughput / baseline_throughput) * 100 if baseline_throughput > 0 else None

        # Throughput per million parameters
        throughput_per_param = throughput / params_millions if params_millions > 0 else None

        # Baseline efficiency for comparison
        baseline_efficiency = baseline_throughput / baseline_params if baseline_params > 0 else None
        efficiency_ratio = None
        if throughput_per_param and baseline_efficiency:
            efficiency_ratio = throughput_per_param / baseline_efficiency

        # Convert engrams to equivalent tokens
        equiv_tokens = None
        if engrams_per_sec:
            equiv_tokens = engrams_per_sec * engram_to_token_ratio

        # Normalized quality
        norm_quality = None
        if quality_score is not None:
            norm_quality = min(1.0, quality_score / baseline_quality)

        # Combined efficiency score: throughput * quality / params
        norm_efficiency = None
        if throughput_per_param and quality_score:
            norm_efficiency = throughput_per_param * quality_score

        return cls(
            speedup_vs_baseline=speedup,
            relative_throughput=relative,
            efficiency_vs_baseline=efficiency_ratio,
            throughput_per_param=throughput_per_param,
            equivalent_tokens_per_sec=equiv_tokens,
            normalized_quality=norm_quality,
            normalized_efficiency=norm_efficiency,
            baseline_model=baseline,
        )

    def is_pareto_optimal(
        self,
        other: NormalizedMetrics,
    ) -> bool:
        """Check if this metric set Pareto-dominates another.

        A metric set is Pareto-optimal if it's better in at least one
        dimension and no worse in all others.

        Args:
            other: Another NormalizedMetrics to compare against.

        Returns:
            True if self Pareto-dominates other.
        """
        if not self.speedup_vs_baseline or not other.speedup_vs_baseline:
            return False
        if not self.normalized_quality or not other.normalized_quality:
            return False

        speed_better = self.speedup_vs_baseline >= other.speedup_vs_baseline
        quality_better = self.normalized_quality >= other.normalized_quality

        speed_strictly_better = self.speedup_vs_baseline > other.speedup_vs_baseline
        quality_strictly_better = self.normalized_quality > other.normalized_quality

        return speed_better and quality_better and (speed_strictly_better or quality_strictly_better)

    def comparison_summary(self) -> str:
        """Generate human-readable comparison summary.

        Returns:
            Multi-line summary string.
        """
        lines = [f"Comparison vs {self.baseline_model or 'baseline'}:"]
        if self.speedup_vs_baseline:
            emoji = "🚀" if self.speedup_vs_baseline > 1 else "🐢"
            lines.append(f"  {emoji} Speedup: {self.speedup_vs_baseline:.2f}x")
        if self.efficiency_vs_baseline:
            emoji = "🟢" if self.efficiency_vs_baseline > 1 else "🔴"
            lines.append(f"  {emoji} Efficiency: {self.efficiency_vs_baseline:.2f}x")
        if self.equivalent_tokens_per_sec:
            lines.append(f"  🔤 Equiv. Tokens/s: {self.equivalent_tokens_per_sec:.0f}")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedMetrics:
        """Deserialize from dictionary.

        Args:
            data: Dictionary of metric values.

        Returns:
            NormalizedMetrics instance.
        """
        return cls(
            speedup_vs_baseline=data.get("speedup_vs_baseline"),
            relative_throughput=data.get("relative_throughput"),
            efficiency_vs_baseline=data.get("efficiency_vs_baseline"),
            throughput_per_param=data.get("throughput_per_param"),
            equivalent_tokens_per_sec=data.get("equivalent_tokens_per_sec"),
            normalized_quality=data.get("normalized_quality"),
            normalized_efficiency=data.get("normalized_efficiency"),
            baseline_model=data.get("baseline_model"),
            normalization_version=data.get("normalization_version", "v1.0"),
        )


# Register with global registry
METRIC_REGISTRY.register("normalized", NormalizedMetrics)
