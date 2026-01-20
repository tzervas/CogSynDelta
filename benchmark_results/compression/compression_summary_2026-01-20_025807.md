# Compression Benchmark Results

**Date**: 2026-01-20T02:58:07.525594+00:00
**Device**: cuda
**PyTorch**: 2.9.1+cu128

## Results

| Compactor | Fidelity | Compression | Latency (ms) | Target Met | Notes |
|-----------|----------|-------------|--------------|------------|-------|
| HighFidelityCompactor | 1.0000 ± 0.0000 | 1.33x | 0.11 | ✗ | - |
| HybridAdaptiveCompactor | 0.9650 ± 0.0107 | 0.79x | 0.79 | ✗ | - |
| ResidualBoostCompactor | 0.5613 ± 0.0402 | 16.00x | 0.60 | ✗ | - |
| DenseEmbeddingEncoder | 0.0033 ± 0.0411 | 2.67x | 0.34 | ✗ | untrained |
