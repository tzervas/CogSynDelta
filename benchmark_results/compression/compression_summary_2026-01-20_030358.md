# Compression Benchmark Results

**Date**: 2026-01-20T03:03:58.795008+00:00
**Device**: cuda
**PyTorch**: 2.9.1+cu128

## Results

| Compactor | Fidelity | Compression | Latency (ms) | Target Met | Notes |
|-----------|----------|-------------|--------------|------------|-------|
| HighFidelityCompactor | 1.0000 ± 0.0000 | 1.33x | 0.12 | ✗ | - |
| HybridAdaptiveCompactor | 0.9650 ± 0.0107 | 0.79x | 1.15 | ✗ | - |
| ResidualBoostCompactor | 0.5613 ± 0.0402 | 16.00x | 0.64 | ✗ | - |
| DenseEmbeddingEncoder | 0.0033 ± 0.0411 | 2.67x | 0.32 | ✗ | untrained |
