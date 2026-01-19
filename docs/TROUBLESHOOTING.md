# Troubleshooting Guide

## Common Issues and Solutions

This guide covers frequently encountered issues when working with CogSynDelta.

## Installation Issues

### UV Package Manager Issues

**Problem**: `uv` command not found
```bash
uv: command not found
```

**Solution**:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.cargo/bin:$PATH"

# Restart terminal or source profile
source ~/.bashrc
```

**Problem**: UV sync fails with Python version conflicts
```bash
error: The requested Python version is not available
```

**Solution**:
```bash
# Check available Python versions
uv python list

# Install Python 3.14 if needed
uv python install 3.14

# Use specific Python version
uv sync --python 3.14
```

### PyTorch/CUDA Issues

**Problem**: CUDA not detected despite GPU being present
```bash
UserWarning: CUDA is not available
```

**Solution**:
```bash
# Check CUDA installation
nvidia-smi

# Check PyTorch CUDA support
uv run python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Problem**: RTX 5080 not supported by current PyTorch version
```bash
RuntimeError: CUDA error: no kernel image is available for execution
```

**Solution**:
- RTX 5080 requires PyTorch 2.9+ with CUDA 12.8+
- Check [PyTorch installation matrix](https://pytorch.org/get-started/locally/)
- Use CPU fallback for development: `export CUDA_VISIBLE_DEVICES=""`

## Runtime Issues

### Memory Errors

**Problem**: Out of memory during training
```bash
RuntimeError: CUDA out of memory
```

**Solutions**:
```python
# Reduce batch size
batch_size = 8  # Try 4, 2, or 1

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.amp import autocast
with autocast(device_type='cuda', dtype=torch.float16):
    # training code

# Clear cache periodically
torch.cuda.empty_cache()
```

**Problem**: Memory tier not found errors
```python
KeyError: 'Memory not found in any tier'
```

**Solution**: This is intentional graceful degradation. The system continues operating with reduced memory capacity.

### Import Errors

**Problem**: Module not found errors
```python
ModuleNotFoundError: No module named 'cogsyndelta'
```

**Solutions**:
```bash
# Install in development mode
uv pip install -e .

# Check Python path
uv run python -c "import sys; print(sys.path)"

# Verify package structure
uv run python -c "import cogsyndelta; print(cogsyndelta.__file__)"
```

### Quantum Computing Features

**Problem**: Quantum features not available
```python
FutureWarning: Quantum computing features are not yet implemented
```

**Solution**: Quantum features are stubbed for future implementation. They will raise `NotImplementedError` or show warnings.

## Testing Issues

### Pytest Discovery Issues

**Problem**: Tests not discovered
```bash
collected 0 items
```

**Solutions**:
```bash
# Run from project root
cd /path/to/CogSynDelta
uv run pytest tests/

# Check test file naming
ls tests/test_*.py

# Run with verbose output
uv run pytest -v tests/
```

### GPU Test Failures

**Problem**: GPU tests fail on CPU-only systems
```bash
AssertionError: CUDA device not available
```

**Solution**:
```bash
# Skip GPU tests
uv run pytest tests/ -k "not gpu"

# Run CPU-only tests
uv run pytest tests/ --ignore=tests/test_optimization.py
```

### Python 3.14 Compatibility Issues

**Problem**: Pydantic errors with Python 3.14
```python
TypeError: _eval_type is not available in Python 3.14
```

**Solution**: Some tests are skipped on Python 3.14 until Pydantic releases a compatible version.

## Development Issues

### Type Checking Errors

**Problem**: MyPy errors
```bash
error: Library stubs not installed for "torch"
```

**Solutions**:
```bash
# Install type stubs
uv add types-PyYAML types-requests

# Run type checking
uv run mypy src/

# Ignore specific errors in code
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # type-only imports
```

### Linting Errors

**Problem**: Ruff formatting issues
```bash
error: Line too long
```

**Solutions**:
```bash
# Auto-fix formatting
uv run ruff format src/ tests/

# Check for errors
uv run ruff check src/ tests/

# Fix specific issues
uv run ruff check --fix src/ tests/
```

### Git Signing Issues

**Problem**: Commit not GPG signed
```bash
error: gpg failed to sign the data
```

**Solutions**:
```bash
# Configure GPG signing
git config commit.gpgsign true
git config user.signingkey YOUR_KEY_ID

# List GPG keys
gpg --list-secret-keys

# Test signing
echo "test" | gpg --clearsign

# Amend last commit
git commit --amend -S
```

## Performance Issues

### Slow Training/Inference

**Problem**: Model runs slower than expected

**Solutions**:
```python
# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Use DataParallel for multi-GPU
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Profile performance
with torch.profiler.profile() as prof:
    # your code
print(prof.key_averages().table())
```

### Benchmark Inconsistencies

**Problem**: Benchmark results vary between runs

**Solutions**:
- Use fixed random seeds: `torch.manual_seed(42)`
- Warm up GPU before benchmarking
- Run multiple iterations and average results
- Use `torch.cuda.synchronize()` for accurate timing

## Getting Help

If these solutions don't resolve your issue:

1. **Check existing issues**: [GitHub Issues](https://github.com/tzervas/CogSynDelta/issues)
2. **Create a new issue**: Include full error messages, environment details, and reproduction steps
3. **Contact support**: maintainers@vectorweight.com for urgent issues

## Environment Information

When reporting issues, please include:

```bash
# System information
uname -a
python --version
uv --version

# Package versions
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
uv run python -c "import cogsyndelta; print(f'CogSynDelta: {cogsyndelta.__version__}')"

# Hardware
nvidia-smi 2>/dev/null || echo "No NVIDIA GPU detected"
```

---

*Last Updated: January 18, 2026*