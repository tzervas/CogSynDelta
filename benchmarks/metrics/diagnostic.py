"""Diagnostic metrics for bottleneck analysis and tuning.

This module provides diagnostic metrics that help explain the "how"
behind performance numbers, enabling targeted optimization.

Classes:
    DiagnosticMetrics: Bottleneck analysis, efficiency breakdown, recommendations

Why diagnostic metrics:
    Raw performance numbers tell you "what" but not "why". Diagnostic
    metrics help identify whether you're compute-bound, memory-bound,
    or IO-bound, and provide actionable recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmarks.metrics.base import METRIC_REGISTRY, MetricMixin


@dataclass
class DiagnosticMetrics(MetricMixin):
    """Diagnostic metrics for bottleneck analysis and optimization guidance.

    Analyzes utilization patterns to identify the primary bottleneck
    and provides actionable recommendations for improvement.

    Attributes:
        primary_bottleneck: Main bottleneck ("compute", "memory", "io", "latency").
        bottleneck_severity: Severity score 0-1 (1 = severe).
        bottleneck_explanation: Human-readable explanation.
        compute_efficiency_pct: How efficiently compute resources are used.
        memory_efficiency_pct: How efficiently memory is used.
        io_efficiency_pct: How efficiently IO/bandwidth is used.
        theoretical_max_throughput: Maximum possible throughput.
        achieved_throughput_pct: Percentage of theoretical max achieved.
        optimization_potential_pct: Estimated potential improvement.
        recommendations: List of actionable optimization suggestions.
        is_compute_bound: True if compute is the limiting factor.
        is_memory_bound: True if memory bandwidth is the limiting factor.
        is_io_bound: True if IO/data loading is the limiting factor.

    Why these diagnostics:
        Knowing your bottleneck determines optimization strategy:
        - Compute-bound: Use faster kernels, quantization, pruning
        - Memory-bound: Reduce batch size, use gradient checkpointing
        - IO-bound: Use faster storage, prefetch, cache datasets
    """

    primary_bottleneck: str | None = None
    bottleneck_severity: float | None = None
    bottleneck_explanation: str | None = None
    compute_efficiency_pct: float | None = None
    memory_efficiency_pct: float | None = None
    io_efficiency_pct: float | None = None
    theoretical_max_throughput: float | None = None
    achieved_throughput_pct: float | None = None
    optimization_potential_pct: float | None = None
    recommendations: list[str] = field(default_factory=list)
    is_compute_bound: bool = False
    is_memory_bound: bool = False
    is_io_bound: bool = False

    @classmethod
    def analyze(
        cls,
        gpu_utilization_pct: float | None,
        gpu_memory_utilization_pct: float | None,
        cpu_utilization_pct: float | None = None,
        io_wait_pct: float | None = None,
        throughput: float | None = None,
        theoretical_max: float | None = None,
        batch_size: int | None = None,
        model_params_millions: float | None = None,
    ) -> DiagnosticMetrics:
        """Analyze utilization patterns to diagnose bottlenecks.

        Uses heuristics based on utilization patterns to identify
        the primary bottleneck and generate recommendations.

        Args:
            gpu_utilization_pct: GPU compute utilization 0-100.
            gpu_memory_utilization_pct: GPU memory utilization 0-100.
            cpu_utilization_pct: CPU utilization 0-100 (optional).
            io_wait_pct: IO wait percentage (optional).
            throughput: Achieved throughput (optional).
            theoretical_max: Theoretical maximum throughput (optional).
            batch_size: Current batch size (optional).
            model_params_millions: Model size in millions (optional).

        Returns:
            DiagnosticMetrics instance with analysis results.

        Example:
            >>> diag = DiagnosticMetrics.analyze(
            ...     gpu_utilization_pct=95,
            ...     gpu_memory_utilization_pct=40,
            ... )
            >>> print(diag.primary_bottleneck)  # "compute"
            >>> print(diag.recommendations)
        """
        # Initialize
        bottleneck = None
        explanation = ""
        recommendations: list[str] = []
        is_compute = False
        is_memory = False
        is_io = False

        # Analyze patterns
        gpu_util = gpu_utilization_pct or 0
        mem_util = gpu_memory_utilization_pct or 0
        cpu_util = cpu_utilization_pct or 0
        io_wait = io_wait_pct or 0

        # Compute-bound: high GPU util, low memory util
        if gpu_util > 90 and mem_util < 70:
            bottleneck = "compute"
            severity = (gpu_util - 90) / 10  # 0-1 severity
            explanation = (
                f"GPU compute is saturated ({gpu_util:.1f}%) while memory has headroom "
                f"({mem_util:.1f}%). Workload is compute-bound."
            )
            is_compute = True
            recommendations.extend(
                [
                    "Consider using mixed precision (fp16/bf16) to increase throughput",
                    "Try torch.compile() for kernel fusion",
                    "Explore quantization (INT8/INT4) for inference",
                ]
            )

        # Memory-bound: high memory util, lower GPU util
        elif mem_util > 85:
            bottleneck = "memory"
            severity = (mem_util - 85) / 15  # 0-1 severity
            explanation = (
                f"GPU memory is heavily utilized ({mem_util:.1f}%) indicating "
                f"memory bandwidth or capacity is limiting performance."
            )
            is_memory = True
            recommendations.extend(
                [
                    "Reduce batch size to decrease memory pressure",
                    "Enable gradient checkpointing for training",
                    "Use memory-efficient attention (Flash Attention)",
                    "Consider model sharding across multiple GPUs",
                ]
            )

        # IO-bound: low GPU util, high IO wait
        elif gpu_util < 50 and (io_wait > 20 or cpu_util > 80):
            bottleneck = "io"
            severity = max((50 - gpu_util) / 50, io_wait / 100)
            explanation = (
                f"GPU is underutilized ({gpu_util:.1f}%) while CPU/IO is busy. "
                f"Data loading or preprocessing may be the bottleneck."
            )
            is_io = True
            recommendations.extend(
                [
                    "Increase DataLoader num_workers",
                    "Use pinned memory (pin_memory=True)",
                    "Prefetch data to GPU",
                    "Consider NVMe storage or RAM disk for dataset",
                ]
            )

        # Latency-bound: everything is low
        elif gpu_util < 30 and mem_util < 30:
            bottleneck = "latency"
            severity = (60 - gpu_util - mem_util) / 60
            explanation = (
                f"Both GPU ({gpu_util:.1f}%) and memory ({mem_util:.1f}%) are "
                f"underutilized. Kernel launch overhead or synchronization may dominate."
            )
            recommendations.extend(
                [
                    "Increase batch size to amortize kernel launch overhead",
                    "Use CUDA graphs for fixed-size workloads",
                    "Reduce CPU-GPU synchronization points",
                ]
            )

        else:
            bottleneck = "balanced"
            severity = 0.1
            explanation = "Workload is relatively balanced across compute and memory."

        # Efficiency calculations
        compute_eff = gpu_util if gpu_util else None
        memory_eff = mem_util if mem_util else None
        io_eff = 100 - io_wait if io_wait else None

        # Throughput vs theoretical
        achieved_pct = None
        optimization_potential = None
        if throughput and theoretical_max and theoretical_max > 0:
            achieved_pct = (throughput / theoretical_max) * 100
            optimization_potential = 100 - achieved_pct

        # Add batch size recommendation if applicable
        if batch_size and model_params_millions:
            # Heuristic: optimal batch size scales with model size
            suggested_batch = int(1000 / (model_params_millions / 100))
            if batch_size < suggested_batch * 0.5:
                recommendations.append(
                    f"Consider increasing batch size (current: {batch_size}, "
                    f"suggested: {suggested_batch})"
                )
            elif batch_size > suggested_batch * 2:
                recommendations.append(
                    f"Batch size may be too large (current: {batch_size}, "
                    f"suggested: {suggested_batch})"
                )

        return cls(
            primary_bottleneck=bottleneck,
            bottleneck_severity=min(1.0, max(0.0, severity)),
            bottleneck_explanation=explanation,
            compute_efficiency_pct=compute_eff,
            memory_efficiency_pct=memory_eff,
            io_efficiency_pct=io_eff,
            theoretical_max_throughput=theoretical_max,
            achieved_throughput_pct=achieved_pct,
            optimization_potential_pct=optimization_potential,
            recommendations=recommendations,
            is_compute_bound=is_compute,
            is_memory_bound=is_memory,
            is_io_bound=is_io,
        )

    def top_recommendations(self, n: int = 3) -> list[str]:
        """Get top N recommendations.

        Args:
            n: Number of recommendations to return.

        Returns:
            List of top recommendation strings.
        """
        return self.recommendations[:n]

    def bottleneck_icon(self) -> str:
        """Get emoji icon for primary bottleneck.

        Returns:
            Emoji representing the bottleneck type.
        """
        icons = {
            "compute": "💻",
            "memory": "💾",
            "io": "📁",
            "latency": "⏱️",
            "balanced": "⚖️",
        }
        return icons.get(self.primary_bottleneck or "", "❓")

    def severity_bar(self, width: int = 10) -> str:
        """Generate visual severity bar.

        Args:
            width: Width of bar in characters.

        Returns:
            ASCII bar representation.
        """
        if self.bottleneck_severity is None:
            return "░" * width
        filled = int(self.bottleneck_severity * width)
        return "█" * filled + "░" * (width - filled)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticMetrics:
        """Deserialize from dictionary.

        Args:
            data: Dictionary of metric values.

        Returns:
            DiagnosticMetrics instance.
        """
        return cls(
            primary_bottleneck=data.get("primary_bottleneck"),
            bottleneck_severity=data.get("bottleneck_severity"),
            bottleneck_explanation=data.get("bottleneck_explanation"),
            compute_efficiency_pct=data.get("compute_efficiency_pct"),
            memory_efficiency_pct=data.get("memory_efficiency_pct"),
            io_efficiency_pct=data.get("io_efficiency_pct"),
            theoretical_max_throughput=data.get("theoretical_max_throughput"),
            achieved_throughput_pct=data.get("achieved_throughput_pct"),
            optimization_potential_pct=data.get("optimization_potential_pct"),
            recommendations=data.get("recommendations", []),
            is_compute_bound=data.get("is_compute_bound", False),
            is_memory_bound=data.get("is_memory_bound", False),
            is_io_bound=data.get("is_io_bound", False),
        )


# Register with global registry
METRIC_REGISTRY.register("diagnostic", DiagnosticMetrics)
