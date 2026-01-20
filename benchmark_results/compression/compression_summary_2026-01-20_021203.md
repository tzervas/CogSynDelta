# Compression Benchmark Results

**Date**: 2026-01-20T02:12:03.541155+00:00
**Device**: cuda
**PyTorch**: 2.9.1+cu128

## Results

| Compactor | Fidelity | Compression | Latency (ms) | Target Met |
|-----------|----------|-------------|--------------|------------|
| HighFidelityCompactor | 1.0000 ± 0.0000 | 1.33x | 0.10 | ✗ |
| HybridAdaptiveCompactor | 0.9650 ± 0.0107 | 0.79x | 0.73 | ✗ |
| ResidualBoostCompactor | 0.5613 ± 0.0402 | 16.00x | 0.66 | ✗ |
| LosslessCompactor | 0.2323 ± 0.0434 | 1.60x | 17.60 | ✗ |
