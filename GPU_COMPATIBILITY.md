# GPU Compatibility Notes

## RTX 5080 Status

### Current Limitation

The NVIDIA GeForce RTX 5080 (sm_120 compute capability) is **not yet supported** by the current stable PyTorch releases.

### Technical Details

- **GPU**: NVIDIA GeForce RTX 5080
- **VRAM**: 16 GB
- **Compute Capability**: sm_120
- **Driver Version**: 590.48.01
- **CUDA Version**: 13.1

### PyTorch Compatibility

Current PyTorch 2.5.1 supports CUDA capabilities:
- sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90

The RTX 5080's sm_120 is from the next-generation Blackwell architecture and requires:
- PyTorch nightly builds OR
- Future stable releases (estimated PyTorch 2.6+)

### Workaround

Until official support is available, benchmarks run on CPU:

```bash
python benchmarks/gpu_benchmark.py
```

**CPU Performance Results** (20-core system):
- Matrix Operations: 8.6 GFLOPS (1024x1024)
- Neural Network Inference: 5,152 samples/sec (batch 128)
- Memory Compression: 16x @ 27M samples/sec

### Future GPU Support

To enable GPU acceleration when available:

1. **Install PyTorch Nightly** (when sm_120 support is added):
   ```bash
   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121
   ```

2. **Verify GPU Support**:
   ```python
   import torch
   print(f"CUDA Available: {torch.cuda.is_available()}")
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   ```

3. **Run GPU Benchmarks**:
   ```bash
   python benchmarks/rtx5080_benchmark.py
   ```

### Expected Performance (Estimated)

Based on RTX 5080 specifications:
- **CUDA Cores**: ~10,000
- **Tensor Cores**: 4th Gen
- **Expected TFLOPS**: ~50-80 TFLOPS (FP32)
- **Memory Bandwidth**: ~600 GB/s

This would provide **~100-200x speedup** over CPU for:
- Matrix operations
- Neural network training/inference
- Large-scale memory operations

## Monitoring

Check PyTorch CUDA support:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## References

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [CUDA Compute Capabilities](https://developer.nvidia.com/cuda-gpus)
- [PyTorch Nightly Builds](https://pytorch.org/get-started/locally/#start-locally)
