"""Benchmark configuration with sensible defaults and overrides.

This module provides a hierarchical configuration system for benchmarks:
1. Built-in defaults (in this file)
2. Project-level overrides (config/benchmark_config.yaml)
3. Runtime overrides (via BenchmarkConfig.override())

Classes:
    BenchmarkConfig: Main configuration class with all settings
    MetricCollectionConfig: Which metrics to collect
    ResourceTrackingConfig: Resource tracking settings
    HistoryConfig: History/archival settings

Why hierarchical config:
    Different environments need different settings. Development wants
    fast iteration, CI wants comprehensive coverage, production wants
    minimal overhead. This allows sensible defaults with easy overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Default config file location
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "benchmark_config.yaml"


@dataclass
class MetricCollectionConfig:
    """Configuration for which metrics to collect.

    Attributes:
        collect_power: Collect power/energy metrics.
        collect_eco: Collect eco-efficiency metrics.
        collect_latent: Collect latent space metrics.
        collect_normalized: Collect normalized comparison metrics.
        collect_diagnostic: Collect diagnostic/bottleneck metrics.
        collect_resource_usage: Track process-specific resource usage.
        collect_cpu_topology: Capture CPU core topology.
        collect_gpu_topology: Capture GPU SM topology.
    """

    collect_power: bool = True
    collect_eco: bool = True
    collect_latent: bool = True
    collect_normalized: bool = True
    collect_diagnostic: bool = True
    collect_resource_usage: bool = True
    collect_cpu_topology: bool = False  # Expensive, opt-in
    collect_gpu_topology: bool = False  # Expensive, opt-in


@dataclass
class ResourceTrackingConfig:
    """Configuration for resource tracking.

    Attributes:
        enabled: Whether to track resources.
        sample_interval_sec: Seconds between samples.
        include_children: Include child process resources.
        track_io: Track disk IO.
        max_samples: Maximum samples to keep (memory limit).
        gpu_index: GPU device to track.
    """

    enabled: bool = True
    sample_interval_sec: float = 0.5
    include_children: bool = True
    track_io: bool = True
    max_samples: int = 1000  # ~8 minutes at 0.5s interval
    gpu_index: int = 0


@dataclass
class HistoryConfig:
    """Configuration for benchmark history management.

    Attributes:
        enabled: Whether to save history.
        history_dir: Directory for history files.
        archive_dir: Directory for archived/compacted history.
        database_path: Path to SQLite database for compacted history.
        max_recent_runs: Keep this many recent runs as individual files.
        auto_archive: Automatically archive old runs.
        archive_after_days: Archive runs older than this.
        compress_archive: Compress archived data.
        retention_days: Delete archives older than this (0 = keep forever).
    """

    enabled: bool = True
    history_dir: Path = field(default_factory=lambda: Path("benchmark_results/history"))
    archive_dir: Path = field(default_factory=lambda: Path("benchmark_results/archive"))
    database_path: Path = field(default_factory=lambda: Path("benchmark_results/history.db"))
    max_recent_runs: int = 50  # Keep last 50 runs as files
    auto_archive: bool = True
    archive_after_days: int = 7  # Archive runs older than a week
    compress_archive: bool = True
    retention_days: int = 365  # Keep archives for a year


@dataclass
class DisplayConfig:
    """Configuration for benchmark display output.

    Attributes:
        show_sparklines: Show trend sparklines.
        show_power: Show power metrics.
        show_eco: Show eco-efficiency.
        show_topology: Show CPU/GPU topology.
        show_recommendations: Show optimization recommendations.
        max_recommendations: Maximum recommendations to show.
        color_output: Use color in terminal output.
    """

    show_sparklines: bool = True
    show_power: bool = True
    show_eco: bool = True
    show_topology: bool = False  # Default off, verbose
    show_recommendations: bool = True
    max_recommendations: int = 3
    color_output: bool = True


@dataclass
class ComparisonConfig:
    """Configuration for cross-architecture comparison.

    Attributes:
        baseline_model: Default baseline for comparison.
        engram_to_token_ratio: Ratio for engram-to-token conversion.
        include_industry_baselines: Compare against industry benchmarks.
        baselines_file: Path to industry baselines JSON.
    """

    baseline_model: str = "gpt2_small"
    engram_to_token_ratio: float = 10.0  # 1 engram ≈ 10 tokens info
    include_industry_baselines: bool = True
    baselines_file: Path = field(
        default_factory=lambda: Path("benchmarks/data/industry_baselines.json")
    )


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration.

    Provides hierarchical configuration with:
    1. Built-in defaults
    2. Project-level overrides (from YAML)
    3. Runtime overrides

    Example:
        >>> config = BenchmarkConfig.load()
        >>> config.override(metrics={"collect_power": False})
        >>> print(config.metrics.collect_power)  # False
    """

    metrics: MetricCollectionConfig = field(default_factory=MetricCollectionConfig)
    resources: ResourceTrackingConfig = field(default_factory=ResourceTrackingConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)

    # Global settings
    verbose: bool = False
    debug: bool = False
    gpu_index: int = 0

    @classmethod
    def load(cls, config_path: Path | str | None = None) -> BenchmarkConfig:
        """Load configuration from YAML file.

        Loads defaults, then overlays project config if it exists.

        Args:
            config_path: Optional path to config file. Uses default if None.

        Returns:
            BenchmarkConfig instance.
        """
        config = cls()

        # Try to load project config
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if path.exists():
            try:
                with open(path) as f:
                    data = yaml.safe_load(f) or {}

                if "benchmark" in data:
                    data = data["benchmark"]

                config._apply_dict(data)
            except Exception:  # noqa: S110 - Intentional: use defaults on config load error
                pass

        # Check environment overrides
        config._apply_env_overrides()

        return config

    @classmethod
    def full_suite(cls) -> BenchmarkConfig:
        """Get configuration for full benchmark suite with all features.

        Returns:
            BenchmarkConfig with all features enabled.
        """
        return cls(
            metrics=MetricCollectionConfig(
                collect_power=True,
                collect_eco=True,
                collect_latent=True,
                collect_normalized=True,
                collect_diagnostic=True,
                collect_resource_usage=True,
                collect_cpu_topology=True,
                collect_gpu_topology=True,
            ),
            resources=ResourceTrackingConfig(
                enabled=True,
                sample_interval_sec=0.25,  # More frequent for full suite
                include_children=True,
                track_io=True,
                max_samples=2000,
            ),
            history=HistoryConfig(
                enabled=True,
                auto_archive=True,
            ),
            display=DisplayConfig(
                show_sparklines=True,
                show_power=True,
                show_eco=True,
                show_topology=True,
                show_recommendations=True,
            ),
            verbose=True,
        )

    @classmethod
    def minimal(cls) -> BenchmarkConfig:
        """Get minimal configuration for fast iteration.

        Returns:
            BenchmarkConfig with minimal overhead.
        """
        return cls(
            metrics=MetricCollectionConfig(
                collect_power=False,
                collect_eco=False,
                collect_latent=True,  # Keep core metrics
                collect_normalized=False,
                collect_diagnostic=False,
                collect_resource_usage=False,
                collect_cpu_topology=False,
                collect_gpu_topology=False,
            ),
            resources=ResourceTrackingConfig(
                enabled=False,
            ),
            history=HistoryConfig(
                enabled=True,
                auto_archive=False,
            ),
            display=DisplayConfig(
                show_sparklines=False,
                show_power=False,
                show_eco=False,
                show_topology=False,
                show_recommendations=False,
            ),
        )

    def override(self, **kwargs: Any) -> BenchmarkConfig:
        """Apply runtime overrides.

        Args:
            **kwargs: Override values. Nested dicts for sub-configs.
                e.g., metrics={"collect_power": False}

        Returns:
            Self for chaining.
        """
        self._apply_dict(kwargs)
        return self

    def _apply_dict(self, data: dict[str, Any]) -> None:
        """Apply dictionary of values to config."""
        if "metrics" in data and isinstance(data["metrics"], dict):
            for k, v in data["metrics"].items():
                if hasattr(self.metrics, k):
                    setattr(self.metrics, k, v)

        if "resources" in data and isinstance(data["resources"], dict):
            for k, v in data["resources"].items():
                if hasattr(self.resources, k):
                    setattr(self.resources, k, v)

        if "history" in data and isinstance(data["history"], dict):
            for k, v in data["history"].items():
                if hasattr(self.history, k):
                    if k.endswith("_dir") or k.endswith("_path"):
                        v = Path(v)
                    setattr(self.history, k, v)

        if "display" in data and isinstance(data["display"], dict):
            for k, v in data["display"].items():
                if hasattr(self.display, k):
                    setattr(self.display, k, v)

        if "comparison" in data and isinstance(data["comparison"], dict):
            for k, v in data["comparison"].items():
                if hasattr(self.comparison, k):
                    if k.endswith("_file"):
                        v = Path(v)
                    setattr(self.comparison, k, v)

        # Top-level settings
        for key in ("verbose", "debug", "gpu_index"):
            if key in data:
                setattr(self, key, data[key])

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # COGSYNDELTA_BENCHMARK_VERBOSE=1
        if os.environ.get("COGSYNDELTA_BENCHMARK_VERBOSE", "").lower() in ("1", "true"):
            self.verbose = True

        # COGSYNDELTA_BENCHMARK_DEBUG=1
        if os.environ.get("COGSYNDELTA_BENCHMARK_DEBUG", "").lower() in ("1", "true"):
            self.debug = True

        # COGSYNDELTA_BENCHMARK_GPU=0
        gpu_env = os.environ.get("COGSYNDELTA_BENCHMARK_GPU")
        if gpu_env and gpu_env.isdigit():
            self.gpu_index = int(gpu_env)

        # COGSYNDELTA_BENCHMARK_FULL=1 for full suite
        if os.environ.get("COGSYNDELTA_BENCHMARK_FULL", "").lower() in ("1", "true"):
            full = BenchmarkConfig.full_suite()
            self.metrics = full.metrics
            self.resources = full.resources
            self.display = full.display

        # COGSYNDELTA_BENCHMARK_MINIMAL=1 for minimal
        if os.environ.get("COGSYNDELTA_BENCHMARK_MINIMAL", "").lower() in ("1", "true"):
            minimal = BenchmarkConfig.minimal()
            self.metrics = minimal.metrics
            self.resources = minimal.resources
            self.display = minimal.display

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": {
                "collect_power": self.metrics.collect_power,
                "collect_eco": self.metrics.collect_eco,
                "collect_latent": self.metrics.collect_latent,
                "collect_normalized": self.metrics.collect_normalized,
                "collect_diagnostic": self.metrics.collect_diagnostic,
                "collect_resource_usage": self.metrics.collect_resource_usage,
                "collect_cpu_topology": self.metrics.collect_cpu_topology,
                "collect_gpu_topology": self.metrics.collect_gpu_topology,
            },
            "resources": {
                "enabled": self.resources.enabled,
                "sample_interval_sec": self.resources.sample_interval_sec,
                "include_children": self.resources.include_children,
                "track_io": self.resources.track_io,
                "max_samples": self.resources.max_samples,
                "gpu_index": self.resources.gpu_index,
            },
            "history": {
                "enabled": self.history.enabled,
                "history_dir": str(self.history.history_dir),
                "archive_dir": str(self.history.archive_dir),
                "database_path": str(self.history.database_path),
                "max_recent_runs": self.history.max_recent_runs,
                "auto_archive": self.history.auto_archive,
                "archive_after_days": self.history.archive_after_days,
                "compress_archive": self.history.compress_archive,
                "retention_days": self.history.retention_days,
            },
            "display": {
                "show_sparklines": self.display.show_sparklines,
                "show_power": self.display.show_power,
                "show_eco": self.display.show_eco,
                "show_topology": self.display.show_topology,
                "show_recommendations": self.display.show_recommendations,
                "max_recommendations": self.display.max_recommendations,
                "color_output": self.display.color_output,
            },
            "comparison": {
                "baseline_model": self.comparison.baseline_model,
                "engram_to_token_ratio": self.comparison.engram_to_token_ratio,
                "include_industry_baselines": self.comparison.include_industry_baselines,
                "baselines_file": str(self.comparison.baselines_file),
            },
            "verbose": self.verbose,
            "debug": self.debug,
            "gpu_index": self.gpu_index,
        }

    def save(self, path: Path | str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save configuration.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump({"benchmark": self.to_dict()}, f, default_flow_style=False)


# Global default configuration instance
_default_config: BenchmarkConfig | None = None


def get_config() -> BenchmarkConfig:
    """Get the global default configuration.

    Loads from file on first access, caches for subsequent calls.

    Returns:
        BenchmarkConfig instance.
    """
    global _default_config
    if _default_config is None:
        _default_config = BenchmarkConfig.load()
    return _default_config


def set_config(config: BenchmarkConfig) -> None:
    """Set the global default configuration.

    Args:
        config: Configuration to use as default.
    """
    global _default_config
    _default_config = config


def reset_config() -> None:
    """Reset global configuration to reload from file."""
    global _default_config
    _default_config = None
