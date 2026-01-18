# GPU Compatibility Matrix

**Last Verified**: 2026-01-18
**PyTorch Version**: 2.9.1 (stable)
**CUDA Versions**: 12.6, 12.8

## Summary

✅ **PyTorch 2.9.1** is the current stable release
✅ **CUDA 12.6 and 12.8** are officially supported
⚠️ **RTX 5080 (sm_120)** support status requires verification

## PyTorch 2.9.1 Specifications

### Release Information
- **Released**: November 12, 2025
- **Status**: Stable (verified 2026-01-18)
- **GitHub**: https://github.com/pytorch/pytorch/releases/tag/v2.9.1
- **PyPI**: https://pypi.org/project/torch/2.9.1/

### CUDA Compatibility

**Officially Supported CUDA Versions**:
- **CUDA 12.8** (recommended for latest hardware)
- **CUDA 12.6** (stable alternative)

**Installation Commands**:
```bash
# CUDA 12.8 (recommended)
pip3 install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.6
pip3 install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126

# CPU only
pip3 install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
```

### Compute Capability Support

PyTorch 2.9.1 supports compute capabilities:
- **sm_50** - Maxwell (GTX 900 series)
- **sm_60, sm_61** - Pascal (GTX 10 series, P100)
- **sm_70, sm_75** - Volta, Turing (V100, RTX 20 series)
- **sm_80, sm_86** - Ampere (A100, RTX 30 series)
- **sm_89** - Ada Lovelace (RTX 40 series, L40)
- **sm_90, sm_90a** - Hopper (H100)

**RTX 5080 (Blackwell architecture) support**: Requires verification against official PyTorch compatibility matrix

## RTX 5080 Setup (Akula Prime Workstation)

### Hardware Configuration
- **System**: Akula Prime (SSH: `ssh akula-prime`)
- **GPU**: NVIDIA GeForce RTX 5080
- **VRAM**: 16 GB GDDR7
- **Architecture**: Blackwell
- **Compute Capability**: sm_120 (requires verification)

### Recommended Setup

1. **Install CUDA 12.8**:
   ```bash
   # SSH into akula-prime
   ssh akula-prime

   # Install CUDA 12.8 (Ubuntu/Debian)
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-8

   # Verify installation
   nvcc --version
   nvidia-smi
   ```

2. **Install PyTorch 2.9.1 with CUDA 12.8**:
   ```bash
   pip3 install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
   ```

3. **Verify GPU Support**:
   ```python
   import torch

   print(f"PyTorch version: {torch.__version__}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"cuDNN version: {torch.backends.cudnn.version()}")

   if torch.cuda.is_available():
       print(f"GPU count: {torch.cuda.device_count()}")
       print(f"GPU name: {torch.cuda.get_device_name(0)}")
       print(f"GPU compute capability: {torch.cuda.get_device_capability(0)}")
       print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
   ```

4. **Run GPU Benchmarks**:
   ```bash
   # On akula-prime
   python benchmarks/rtx5080_benchmark.py
   python benchmarks/gpu_benchmark.py
   ```

### Expected Performance

Based on RTX 5080 specifications (Blackwell architecture):
- **CUDA Cores**: ~10,240
- **Tensor Cores**: 5th Generation
- **RT Cores**: 4th Generation
- **Base Clock**: ~2.6 GHz
- **Memory**: 16GB GDDR7
- **Memory Bandwidth**: ~560 GB/s
- **TDP**: ~320W
- **FP32 Performance**: ~60+ TFLOPS

**Expected Speedups** (vs CPU):
- Matrix operations: 100-200x
- Neural network training: 50-150x
- Inference: 50-100x
- Mixed precision (FP16): 150-300x

## Dependency Versions (Verified 2026-01-18)

### Core Stack
| Package | Version | Notes |
|---------|---------|-------|
| PyTorch | 2.9.1 | Stable release, Nov 2025 |
| torchvision | 0.24.1 | Requires torch==2.9.1 |
| CUDA | 12.6 / 12.8 | Both supported |
| Python | 3.9 - 3.13 | Tested on 3.13.3 |
| NumPy | 2.4.1 | Latest stable |

### GPU Monitoring

**Check CUDA status**:
```bash
# System info
nvidia-smi

# PyTorch CUDA check
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Device details
python -c "import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('CUDA not available')"
```

**Monitor during training**:
```bash
# Real-time GPU utilization
watch -n 1 nvidia-smi

# Detailed GPU stats
nvidia-smi dmon -s ucmt
```

## Troubleshooting

### Issue: CUDA not available

**Solution**: Verify CUDA installation and PyTorch build
```bash
# Check CUDA toolkit
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip3 uninstall torch torchvision
pip3 install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

### Issue: Out of memory errors

**Solution**: Enable memory optimizations
```python
import torch

# Enable memory efficient attention
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use gradient checkpointing
# (implemented in model code)
```

### Issue: Slow inference on GPU

**Solution**: Use torch.compile() and mixed precision
```python
import torch

model = model.to('cuda')
model = torch.compile(model)  # PyTorch 2.x optimization

# Mixed precision
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)
```

## Dependency Validation

To validate all dependencies and their versions:

```bash
# Run dependency validator
python scripts/validate_dependencies.py

# Query dependency documentation
python scripts/query_dependency_docs.py "What CUDA versions does PyTorch 2.9.1 support?"

# Ingest latest dependency docs
python scripts/ingest_dependency_docs.py --package torch --version 2.9.1
```

## References

- **PyTorch 2.9.1**: https://github.com/pytorch/pytorch/releases/tag/v2.9.1
- **PyTorch Docs**: https://pytorch.org/docs/stable/
- **PyTorch Get Started**: https://pytorch.org/get-started/locally/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **CUDA Compute Capabilities**: https://developer.nvidia.com/cuda-gpus
- **cuDNN Support Matrix**: https://docs.nvidia.com/deeplearning/cudnn/latest/
- **Project Dependencies**: See `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`
