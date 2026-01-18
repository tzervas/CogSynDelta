# GPU Benchmark Results - CogSynDelta v0.2.0

## Environment

- **Date**: 2026-01-18
- **Host**: akula-prime
- **GPU**: NVIDIA GeForce RTX 5080 (16GB GDDR7)
- **Driver**: 590.48.01
- **CUDA**: 13.1
- **PyTorch**: 2.9.1+cu128
- **Compute Capability**: sm_120

## Matrix Multiplication Performance

| Size | Time (ms) | GFLOPS |
|------|-----------|--------|
| 1024x1024 | 0.08 | 25,743 |
| 2048x2048 | 0.47 | 36,630 |
| 4096x4096 | 3.55 | 38,743 |
| 8192x8192 | 28.38 | 38,739 |

**Peak Performance**: 39,390 GFLOPS (4096x4096)

## CNN Inference Throughput

| Batch Size | Time (ms) | Throughput (samples/sec) |
|------------|-----------|--------------------------|
| 1 | 0.06 | 16,170 |
| 16 | 0.11 | 140,759 |
| 64 | 0.21 | 302,315 |
| 256 | 0.44 | 575,443 |

**Best Throughput**: 575,443 samples/sec (batch 256)

## ResNet-18 Inference

| Batch Size | Time (ms) | Throughput (samples/sec) |
|------------|-----------|--------------------------|
| 8 | 1.08 | 7,436 |
| 16 | 1.79 | 8,928 |
| 32 | 3.62 | 8,829 |
| 64 | 8.51 | 7,520 |

**Best Throughput**: 8,928 samples/sec (batch 16)

## Training Performance

- **MNIST Training (2 epochs)**:
  - Final Accuracy: 98.73%
  - Time per Epoch: ~2.5s
  - Training Throughput: 139,329 samples/sec

## Memory Bandwidth

| Size | Host→Device | Device→Host |
|------|-------------|-------------|
| 1 MB | 8.58 GB/s | 2.60 GB/s |
| 10 MB | 16.46 GB/s | 5.38 GB/s |
| 100 MB | 14.65 GB/s | 6.53 GB/s |
| 500 MB | 15.25 GB/s | 6.81 GB/s |
| 1000 MB | 15.17 GB/s | 6.70 GB/s |

**Average H2D Bandwidth**: 14.02 GB/s
**Average D2H Bandwidth**: 5.60 GB/s

## Memory Compression (PCN-VAE-GAN)

| Compression | Fidelity | Time (ms) | Throughput |
|-------------|----------|-----------|------------|
| 2x | 0.670 | 0.55 | 1.8M samples/sec |
| 4x | 0.462 | 0.09 | 11.0M samples/sec |
| 8x | 0.323 | 0.02 | 52.1M samples/sec |
| 16x | 0.228 | 0.02 | 56.9M samples/sec |

## Summary

The RTX 5080 provides excellent performance for CogSynDelta workloads:

- **Compute**: Near 40 TFLOPS sustained on large matrix operations
- **CNN Inference**: >500k samples/sec for lightweight models
- **Training**: Fast iteration with 139k samples/sec training throughput
- **Memory**: 16GB GDDR7 handles batch sizes up to 1024 comfortably

These benchmarks validate that the RTX 5080 with sm_120 compute capability
is fully supported by PyTorch 2.9+ and provides production-ready performance
for the CogSynDelta PCN-VAE-GAN hybrid system.
