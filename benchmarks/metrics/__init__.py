"""Reusable metrics collection and analysis modules.

This package provides generalized, configurable metrics collection
suitable for benchmarking any ML workload, with optional CogSynDelta
latent space extensions.

Architecture:
    metrics/
    ├── base.py        - BaseMetric protocol and dataclass mixins
    ├── power.py       - PowerMetrics, EcoMetrics (NVIDIA + CPU)
    ├── compute.py     - ComputeUtilizationMetrics, clock speeds
    ├── latent.py      - LatentSpaceMetrics (CogSynDelta-specific)
    ├── normalized.py  - NormalizedMetrics for cross-arch comparison
    ├── diagnostic.py  - DiagnosticMetrics for bottleneck analysis
    ├── display.py     - ScaledMetricDisplay, formatting utilities
    ├── resources.py   - Process-specific resource tracking with CPU/GPU topology
    ├── config.py      - Hierarchical config with defaults + overrides
    └── history_db.py  - SQLite-based history compaction and archival

Example:
    >>> from benchmarks.metrics import PowerMetrics, DiagnosticMetrics
    >>> power = PowerMetrics.capture()
    >>> diag = DiagnosticMetrics.analyze(utilization_data)
    >>> print(diag.recommendations)

    >>> from benchmarks.metrics import ProcessResourceTracker, BenchmarkConfig
    >>> config = BenchmarkConfig.load()
    >>> tracker = ProcessResourceTracker(sample_interval=config.resources.sample_interval_sec)
    >>> tracker.start()
    >>> # ... run benchmark ...
    >>> usage = tracker.stop()

Why:
    These modules are extracted from history_format.py to enable:
    1. Reuse in other projects (e.g., model_comparisons.py)
    2. Easier testing of individual metric components
    3. Parameterized configuration for different hardware
    4. Extension without bloating the main module
"""

from __future__ import annotations

from benchmarks.metrics.base import (
    TYPICAL_RANGES,
    BaseMetric,
    MetricRegistry,
    MetricUnit,
    ScaleRange,
)
from benchmarks.metrics.compute import (
    ComputeUtilizationMetrics,
)
from benchmarks.metrics.config import (
    BenchmarkConfig,
    ComparisonConfig,
    DisplayConfig,
    HistoryConfig,
    MetricCollectionConfig,
    ResourceTrackingConfig,
    get_config,
    reset_config,
    set_config,
)
from benchmarks.metrics.diagnostic import (
    DiagnosticMetrics,
)
from benchmarks.metrics.display import (
    MetricFormatter,
    ScaledMetricDisplay,
)
from benchmarks.metrics.history_db import (
    ArchiveManager,
    BenchmarkHistoryDB,
    BenchmarkSummary,
)
from benchmarks.metrics.latent import (
    LatentSpaceMetrics,
)
from benchmarks.metrics.normalized import (
    NormalizedMetrics,
)
from benchmarks.metrics.power import (
    EcoMetrics,
    PowerMetrics,
)
from benchmarks.metrics.resources import (
    BenchmarkResourceUsage,
    CPUCoreInfo,
    CPUTopology,
    GPUComputeUnit,
    GPUTopology,
    ProcessResourceTracker,
    ResourceSnapshot,
    ResourceTimeSeries,
)

__all__ = [  # noqa: RUF022 - Grouped by category for readability
    # Base
    "TYPICAL_RANGES",
    "BaseMetric",
    "MetricRegistry",
    "MetricUnit",
    "ScaleRange",
    # Power & Eco
    "EcoMetrics",
    "PowerMetrics",
    # Compute
    "ComputeUtilizationMetrics",
    # Latent space
    "LatentSpaceMetrics",
    # Normalized
    "NormalizedMetrics",
    # Diagnostic
    "DiagnosticMetrics",
    # Display
    "MetricFormatter",
    "ScaledMetricDisplay",
    # Resources
    "BenchmarkResourceUsage",
    "CPUCoreInfo",
    "CPUTopology",
    "GPUComputeUnit",
    "GPUTopology",
    "ProcessResourceTracker",
    "ResourceSnapshot",
    "ResourceTimeSeries",
    # Config
    "BenchmarkConfig",
    "ComparisonConfig",
    "DisplayConfig",
    "HistoryConfig",
    "MetricCollectionConfig",
    "ResourceTrackingConfig",
    "get_config",
    "reset_config",
    "set_config",
    # History DB
    "ArchiveManager",
    "BenchmarkHistoryDB",
    "BenchmarkSummary",
]
