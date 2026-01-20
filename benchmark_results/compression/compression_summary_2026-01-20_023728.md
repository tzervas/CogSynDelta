# Compression Benchmark Results

**Date**: 2026-01-20T02:37:28.648716+00:00
**Device**: cuda
**PyTorch**: 2.9.1+cu128

## Results

| Compactor | Fidelity | Compression | Latency (ms) | Target Met |
|-----------|----------|-------------|--------------|------------|
| HighFidelityCompactor | 1.0000 ± 0.0000 | 1.33x | 0.12 | ✗ |
| HybridAdaptiveCompactor | 0.9639 ± 0.0123 | 0.79x | 2.66 | ✗ |
| ResidualBoostCompactor | 0.5645 ± 0.0454 | 16.00x | 1.13 | ✗ |
| DenseEmbeddingEncoder | -0.0017 ± 0.0438 | 2.67x | 0.27 | ✗ |
