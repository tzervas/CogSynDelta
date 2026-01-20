# Compression Benchmark Results

This directory contains benchmark results from the compression research effort
documented in ADR-0008 and ADR-0014.

## File Naming Convention

- `compression_benchmark_YYYY-MM-DD_HHMMSS.json` - Full benchmark data
- `compression_summary_YYYY-MM-DD_HHMMSS.md` - Human-readable summary

## JSON Schema

Each benchmark result file follows this schema:

```json
{
  "results": [
    {
      "compactor_name": "string",
      "fidelity_mean": "float (0-1, target ≥0.95)",
      "fidelity_std": "float",
      "fidelity_min": "float",
      "fidelity_percentiles": {
        "p50": "float",
        "p90": "float",
        "p95": "float",
        "p99": "float"
      },
      "compression_ratio": "float (target ≥4x for high fidelity)",
      "compress_latency_ms": "float",
      "decompress_latency_ms": "float",
      "total_latency_ms": "float",
      "memory_peak_mb": "float",
      "num_samples": "int",
      "embed_dim": "int",
      "timestamp": "ISO 8601 string",
      "metadata": {}
    }
  ],
  "baseline_fidelity": 1.0,
  "device": "cpu|cuda",
  "torch_version": "string",
  "cuda_available": "boolean",
  "timestamp": "ISO 8601 string"
}
```

## Target Metrics (ADR-0008)

| Compression | Target Fidelity |
|-------------|-----------------|
| 2x | ≥0.95 |
| 4x | ≥0.95 |
| 8x | ≥0.93 |
| 16x | ≥0.90 |

## Running Benchmarks

```bash
# Quick benchmark (for testing)
uv run python -m benchmarks.compression_benchmark --quick

# Full benchmark
uv run python -m benchmarks.compression_benchmark --samples 10000

# With GPU
uv run python -m benchmarks.compression_benchmark --device cuda --samples 10000
```

## Analysis

Use the benchmark results for:
1. Tracking progress toward ADR-0008 targets
2. Comparing compactor implementations
3. Identifying performance regressions
4. Validating compression research experiments

## Related Documentation

- [ADR-0008: Compression Fidelity Recovery](../../docs/adr/0008-compression-fidelity-recovery.md)
- [ADR-0014: MRL + QINCo2 Pipeline](../../docs/adr/0014-matryoshka-qinco2-compression.md)
- [Compression Research Spec](../../specs/compression-research/spec.md)
