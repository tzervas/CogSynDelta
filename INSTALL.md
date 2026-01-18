# CogSynDelta Installation Guide

## Quick Start

### Prerequisites
- Python 3.9 or higher
- NVIDIA GPU with CUDA support (optional, but recommended for RTX 5080 optimizations)
- 16GB RAM minimum (32GB+ recommended)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/tzervas/CogSynDelta.git
cd CogSynDelta

# Install in development mode with all dependencies
pip install -e ".[all]"

# Or install just core dependencies
pip install -e .
```

### Installation Options

```bash
# Core only (minimal dependencies)
pip install -e .

# With quantum computing support
pip install -e ".[quantum]"

# With vision/audio processing
pip install -e ".[vision,audio]"

# Development tools
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

## From PyPI (when published)

```bash
pip install cogsyndelta

# With extras
pip install cogsyndelta[quantum,vision,audio]
```

## Verify Installation

```python
import cogsyndelta
print(f"CogSynDelta version: {cogsyndelta.__version__}")

# Test GPU optimization
from cogsyndelta.optimization.cuda_optimization import get_gpu_optimizer
optimizer = get_gpu_optimizer()
print(optimizer.get_memory_stats())
```

## Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_unit.py

# With coverage
pytest --cov=cogsyndelta --cov-report=html
```

## Running the API Server

```bash
# Using entry point
cogsyndelta-server

# Or directly
python -m cogsyndelta.api.server

# With custom config
cogsyndelta-server --config config/custom_config.yaml
```

## Running Benchmarks

```bash
# Using entry point
cogsyndelta-benchmark

# Or directly
python benchmarks/run.py
```

## GPU Setup (NVIDIA RTX 5080)

### Install CUDA Toolkit

```bash
# For Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# Verify
nvcc --version
nvidia-smi
```

### Install PyTorch with CUDA Support

```bash
# CUDA 12.1+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run code quality checks
black src/ tests/
ruff check src/ tests/
mypy src/

# Run tests
pytest
```

## Troubleshooting

### Import Errors

If you get import errors after installation:

```bash
# Ensure the package is installed
pip list | grep cogsyndelta

# Reinstall in development mode
pip install -e . --force-reinstall
```

### CUDA Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"

# If CUDA not found, reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Memory Issues

```bash
# Reduce batch size in config/config.yaml
# Or set environment variable
export COGSYNDELTA_MAX_BATCH_SIZE=16
```

## Configuration

Configuration files are located in `config/`:
- `config.yaml` - Main configuration
- Custom configs can be created and passed via `--config`

## Next Steps

- See [examples/](examples/) for usage examples
- Read [docs/](docs/) for detailed documentation
- Check [benchmarks/](benchmarks/) for performance evaluation
- Review [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
