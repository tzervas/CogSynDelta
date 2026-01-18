# CogSynDelta Installation Guide

## Quick Start

### Prerequisites
- **Python 3.14+** (managed automatically by uv)
- **uv** package manager (recommended)
- NVIDIA GPU with CUDA 12.8 support (optional, for GPU acceleration)
- 16GB RAM minimum (32GB+ recommended)

### Install uv

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pipx
pipx install uv
```

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/tzervas/CogSynDelta.git
cd CogSynDelta

# Install all dependencies (uv handles Python 3.14 automatically)
uv sync

# Verify installation
uv run python -c "import cogsyndelta; print(f'CogSynDelta ready!')"
```

### Installation Options

```bash
# Core + dev dependencies (default)
uv sync

# Include optional extras
uv sync --extra vision      # Computer vision support
uv sync --extra audio       # Audio processing
uv sync --extra gpu-nvidia  # NVIDIA Triton optimization
uv sync --extra inference-providers  # Cloud AI providers
uv sync --extra all         # Everything

# Include documentation tools
uv sync --group docs
```

## Running Commands

### Using uv run (recommended)

```bash
# Start API server
uv run cogsyndelta-server

# Run benchmarks
uv run cogsyndelta-benchmark

# Run tests
uv run pytest tests/ -v

# Run Python scripts
uv run python examples/basic_training.py
```

### Using uvx for tools

```bash
# Run linting (isolated, doesn't affect project)
uvx ruff check src/

# Format code
uvx black src/ tests/

# Type checking (uses project's mypy config)
uv run mypy src/
```

## GPU Setup (NVIDIA)

### CUDA 12.8 (Recommended for RTX 40/50 series)

The project is pre-configured to use PyTorch with CUDA 12.8 from the PyTorch index.

```bash
# Verify CUDA is available after uv sync
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
uv run python -c "import torch; print(f'Device: {torch.cuda.get_device_name(0)}')"
```

### Alternative CUDA Versions

To use a different CUDA version, edit `pyproject.toml`:

```toml
[tool.uv.sources]
# For CUDA 12.6
torch = { index = "pytorch-cu126" }
torchvision = { index = "pytorch-cu126" }

# For CUDA 12.4
torch = { index = "pytorch-cu124" }
torchvision = { index = "pytorch-cu124" }

# For CPU only
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }
```

Then regenerate the lock file:

```bash
uv lock --upgrade-package torch --upgrade-package torchvision
uv sync
```

### AMD ROCm

```bash
# Edit pyproject.toml to use ROCm index
[tool.uv.sources]
torch = { index = "pytorch-rocm63" }
torchvision = { index = "pytorch-rocm63" }

# Regenerate and sync
uv lock --upgrade-package torch --upgrade-package torchvision
uv sync
```

## Development Setup

```bash
# Clone and sync (includes dev dependencies by default)
git clone https://github.com/tzervas/CogSynDelta.git
cd CogSynDelta
uv sync

# Install pre-commit hooks
uv run pre-commit install

# Run quality checks
uvx ruff check src/ tests/
uvx black --check src/ tests/
uv run mypy src/

# Run tests with coverage
uv run pytest tests/ -v --cov=cogsyndelta --cov-report=html
```

## Verify Installation

```python
# uv run python
import cogsyndelta
print(f"CogSynDelta version: {cogsyndelta.__version__}")

# Test GPU optimization
from cogsyndelta.optimization.cuda_optimization import get_gpu_optimizer
optimizer = get_gpu_optimizer()
print(optimizer.get_memory_stats())

# Test PyTorch CUDA
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
```

## Reproducible Builds

The `uv.lock` file ensures reproducible builds across all environments:

```bash
# Install exact versions from lock file
uv sync --frozen

# Update dependencies and regenerate lock
uv lock --upgrade
uv sync
```

## Troubleshooting

### Import Errors

```bash
# Ensure you're using uv run
uv run python -c "import cogsyndelta"

# If issues persist, recreate environment
rm -rf .venv
uv sync
```

### CUDA Not Found

```bash
# Check PyTorch CUDA status
uv run python -c "import torch; print(torch.cuda.is_available())"

# Switch to CUDA index if using CPU-only build
# Edit pyproject.toml [tool.uv.sources] then:
uv lock --upgrade-package torch
uv sync
```

### Memory Issues

```bash
# Reduce batch size via environment variable
export COGSYNDELTA_MAX_BATCH_SIZE=16
uv run cogsyndelta-server
```

### Lock File Conflicts

```bash
# Regenerate lock file
uv lock

# Force upgrade all packages
uv lock --upgrade

# Upgrade specific package
uv lock --upgrade-package torch
```

## Package Caching

uv automatically caches packages in `~/.cache/uv/`. To manage:

```bash
# View cache location
uv cache dir

# Clear cache (if needed)
uv cache clean
```

## Configuration

Configuration files are located in `config/`:
- `config.yaml` - Main configuration
- Custom configs can be passed via `--config`

```bash
uv run cogsyndelta-server --config config/custom_config.yaml
```

## Next Steps

- See [examples/](examples/) for usage examples
- Read [docs/](docs/) for detailed documentation
- Check [ROADMAP.md](ROADMAP.md) for project roadmap
- Review [CONTRIBUTING.md](docs/CONTRIBUTING.md) for development guidelines
