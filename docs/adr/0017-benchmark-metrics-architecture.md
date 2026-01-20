# ADR-0017: Benchmark Metrics Architecture

## Status

Accepted

## Date

2026-01-19

## Context

CogSynDelta's benchmark infrastructure needed a comprehensive metrics system that:

1. **Tracks latent space performance**: Measures engrams/sec and encoding fidelity, which are CogSynDelta's core primitives
2. **Enables cross-architecture comparison**: Provides normalized metrics for comparing against transformers, CNNs, etc.
3. **Supports eco-conscious development**: Tracks power consumption, CO₂ estimates, and efficiency ratings
4. **Facilitates debugging**: Provides diagnostic metrics that explain bottlenecks and suggest optimizations
5. **Is reusable**: Can be extracted and used in other projects

The existing `history_format.py` module had grown organically and needed decomposition into focused, testable components.

## Decision

We will implement a modular metrics architecture with the following structure:

```
benchmarks/
├── history_format.py      # Enhanced benchmark records (uses metrics/)
└── metrics/               # Reusable metrics collection modules
    ├── __init__.py        # Public API
    ├── base.py            # Protocol, units, registry
    ├── power.py           # PowerMetrics, EcoMetrics
    ├── compute.py         # ComputeUtilizationMetrics
    ├── latent.py          # LatentSpaceMetrics (CogSynDelta-specific)
    ├── normalized.py      # NormalizedMetrics for comparison
    ├── diagnostic.py      # DiagnosticMetrics for bottleneck analysis
    └── display.py         # Formatting and visualization
```

### Key Design Decisions

#### 1. Protocol-Based Metrics

All metric classes implement the `BaseMetric` protocol:
- `to_dict()` - Serialize to dictionary
- `to_json()` - Serialize to JSON
- `from_dict()` - Deserialize (classmethod)

This enables polymorphic handling without inheritance.

#### 2. Metric Units Enumeration

A comprehensive `MetricUnit` enum covers all measurement types:
- Performance: `SAMPLES_PER_SEC`, `MS`, `TFLOPS`
- Power: `WATTS`, `JOULES`, `SAMPLES_PER_WATT`
- Latent Space: `ENGRAMS_PER_SEC`, `ENGRAMS_PER_WATT`, `BITS_PER_DIM`
- Normalized: `SPEEDUP`, `NORMALIZED_SCORE`, `RELATIVE_PERF`

#### 3. Factory Methods

Each metric class uses `capture()` or `compute()` factory methods:
- `PowerMetrics.capture()` - Read from hardware
- `EcoMetrics.compute(duration, power, samples)` - Calculate derived values
- `DiagnosticMetrics.analyze(utilization_data)` - Analyze patterns

#### 4. Latent Space Metrics (CogSynDelta-Specific)

`LatentSpaceMetrics` captures engram-specific measurements:
- `engrams_per_sec` - Encoding throughput
- `engrams_per_watt` - Eco-efficiency
- `encoding_fidelity` - Quality score (0-1)
- `reconstruction_fidelity` - Decoding quality
- `bits_per_dim` - Information density

#### 5. Normalized Metrics for Comparison

`NormalizedMetrics` enables apples-to-apples comparison:
- `speedup_vs_baseline` - Performance ratio vs reference model
- `efficiency_vs_baseline` - Efficiency ratio (throughput/params)
- `equivalent_tokens_per_sec` - Engrams converted to transformer-equivalent
- Includes reference baselines (GPT-2, LLaMA, ViT, ResNet)

#### 6. Diagnostic Metrics

`DiagnosticMetrics` provides actionable insights:
- `primary_bottleneck` - "compute", "memory", "io", "latency"
- `bottleneck_explanation` - Human-readable description
- `recommendations` - List of optimization suggestions
- `optimization_potential_pct` - Estimated improvement available

## Consequences

### Positive

1. **Modularity**: Each metric type can be tested, extended, and used independently
2. **Reusability**: The `metrics/` package can be extracted for other projects
3. **Type Safety**: Full type hints enable mypy strict mode
4. **Extensibility**: New metric types can be added via registry
5. **Sustainability**: Built-in CO₂ tracking supports eco-conscious development
6. **Debuggability**: Diagnostic metrics accelerate performance tuning

### Negative

1. **Complexity**: More files to maintain
2. **Import Overhead**: Multiple imports needed (mitigated by `__init__.py` re-exports)
3. **Duplication Risk**: Some logic exists in both `history_format.py` and `metrics/` during transition

### Neutral

1. **Migration Path**: Existing code continues to work via `history_format.py`
2. **Hardware Dependency**: Power metrics require pynvml/psutil (graceful fallback to None)

## References

- ADR-0007: Hybridized Granular Scoring Metrics
- ADR-0015: Context-Efficient Memory Techniques
- [Green AI Paper](https://arxiv.org/abs/1907.10597) - Motivation for eco-metrics
- [MLPerf Inference Rules](https://github.com/mlcommons/inference_policies) - Industry benchmark standards

## Implementation

- `benchmarks/metrics/__init__.py` - Package entry point
- `benchmarks/metrics/base.py` - 360 lines (units, registry, protocol)
- `benchmarks/metrics/power.py` - 310 lines (PowerMetrics, EcoMetrics)
- `benchmarks/metrics/compute.py` - 250 lines (ComputeUtilizationMetrics)
- `benchmarks/metrics/latent.py` - 200 lines (LatentSpaceMetrics)
- `benchmarks/metrics/normalized.py` - 250 lines (NormalizedMetrics)
- `benchmarks/metrics/diagnostic.py` - 290 lines (DiagnosticMetrics)
- `benchmarks/metrics/display.py` - 370 lines (formatters, sparklines)
