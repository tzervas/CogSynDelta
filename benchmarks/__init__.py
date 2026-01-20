"""Benchmarking suite for CogSynDelta.

Provides comprehensive benchmarking, tracking, and visualization tools:
- compression_benchmark: Test compression fidelity and ratios
- gpu_benchmark: Hardware performance metrics
- model_benchmarks: Model-specific benchmarks
- tracking: Baseline comparison and regression detection
- visualization: Trend analysis and reporting

Example:
    >>> from benchmarks.visualization import BenchmarkVisualizer
    >>> viz = BenchmarkVisualizer()
    >>> viz.print_trend_report()
"""

from benchmarks.tracking import BenchmarkTracker
from benchmarks.visualization import BenchmarkVisualizer

__all__ = [
    "BenchmarkTracker",
    "BenchmarkVisualizer",
]
