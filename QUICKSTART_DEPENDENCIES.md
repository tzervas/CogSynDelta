# Quick Start: Dependency Documentation System

## Installation

```bash
# From project root
cd /home/kang/Documents/projects/2026/CogSynDelta

# Install all dependencies (includes LlamaIndex)
pip install -e .
```

## Verify Current Status

### 1. Check Dependency Versions

```bash
# View updated requirements
cat requirements.txt

# View dev requirements
cat requirements-dev.txt

# View pyproject.toml
cat pyproject.toml
```

### 2. Read Documentation

```bash
# Main dependency reference
cat DEPENDENCIES.md

# GPU compatibility matrix
cat GPU_COMPATIBILITY.md

# Installation guide
cat INSTALL.md

# Update summary
cat DEPENDENCY_UPDATE_SUMMARY.md
```

## Using the RAG Documentation System

### Step 1: Ingest PyTorch Documentation

```bash
# Ingest PyTorch 2.9.1 docs (takes 2-3 minutes)
python scripts/ingest_dependency_docs.py --package torch --version 2.9.1

# Expected output:
# ================================================================================
# Ingesting documentation for torch
# ================================================================================
# Fetching PyPI documentation for torch...
# Fetching GitHub release notes for pytorch/pytorch v2.9.1...
# Added X documents for torch v2.9.1 from pypi
# Added X documents for torch v2.9.1 from github
# ✓ Completed ingestion for torch
```

### Step 2: Ingest All Dependencies

```bash
# Ingest all configured dependencies (takes 10-15 minutes)
python scripts/ingest_dependency_docs.py --all

# This will fetch docs for:
# - torch, torchvision, numpy
# - fastapi, uvicorn, pydantic
# - qiskit, pennylane, cirq
# - pytest, black, ruff, mypy
# - sphinx, llama-index
```

### Step 3: Query the Documentation

```bash
# Ask about CUDA support
python scripts/query_dependency_docs.py "What CUDA versions does PyTorch 2.9.1 support?"

# Ask about specific features
python scripts/query_dependency_docs.py "How do I enable mixed precision training in PyTorch?"

# Filter by dependency
python scripts/query_dependency_docs.py --dependency fastapi "How to create a WebSocket endpoint?"

# Get more results
python scripts/query_dependency_docs.py --top-k 10 "quantum computing backends"
```

### Step 4: List and Inspect

```bash
# List all indexed dependencies
python scripts/query_dependency_docs.py --list-deps

# Show RAG system statistics
python scripts/query_dependency_docs.py --stats

# Get info about specific dependency
python scripts/query_dependency_docs.py --info torch
python scripts/query_dependency_docs.py --info fastapi
```

## Validate Dependencies

### Check Imports vs Requirements

```bash
# Run validation
python scripts/validate_dependencies.py

# Expected output:
# ================================================================================
# DEPENDENCY VALIDATION REPORT
# Generated: 2026-01-18T...
# ================================================================================
# 
# requirements.txt
# --------------------------------------------------------------------------------
# Declared dependencies: 14
# Unused dependencies: X
# Untracked imports: X
```

### Generate JSON Report

```bash
# JSON output for automation
python scripts/validate_dependencies.py --json > validation.json

# Save to file
python scripts/validate_dependencies.py --report validation_report.txt
```

## On Akula Prime (GPU Workstation)

### Connect

```bash
# SSH to akula-prime
ssh akula-prime
```

### Install CUDA 12.8

```bash
# On akula-prime
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

# Verify
nvcc --version
nvidia-smi
```

### Install PyTorch 2.9.1 with CUDA 12.8

```bash
# On akula-prime
pip3 install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# Verify
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 
```

### Run GPU Benchmarks

```bash
# On akula-prime
cd /home/kang/Documents/projects/2026/CogSynDelta
python benchmarks/rtx5080_benchmark.py
python benchmarks/gpu_benchmark.py
```

## Monthly Maintenance

### Update Dependencies (Monthly)

```bash
# 1. Check for updates
python -c "
import urllib.request, json
packages = ['torch', 'fastapi', 'numpy', 'pytest']
for pkg in packages:
    with urllib.request.urlopen(f'https://pypi.org/pypi/{pkg}/json') as r:
        data = json.loads(r.read())
        print(f'{pkg}: {data[\"info\"][\"version\"]}')"

# 2. Review release notes on GitHub
# - https://github.com/pytorch/pytorch/releases
# - https://github.com/fastapi/fastapi/releases
# - etc.

# 3. Test updates in dev environment
pip install torch==NEW_VERSION --index-url https://download.pytorch.org/whl/cu128

# 4. Update requirements files if compatible
nano requirements.txt

# 5. Re-ingest documentation
python scripts/ingest_dependency_docs.py --package torch --version NEW_VERSION

# 6. Re-validate
python scripts/validate_dependencies.py

# 7. Commit changes
git add requirements.txt requirements-dev.txt pyproject.toml
git commit -m "deps: update to torch NEW_VERSION (verified 2026-XX-XX)"
```

## Troubleshooting

### Module Not Found: llama_index

```bash
# Install dependencies
pip install -e .

# Or manually
pip install llama-index-core llama-index-embeddings-huggingface llama-index-vector-stores-faiss faiss-cpu
```

### CUDA Not Available

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Reinstall PyTorch with CUDA
pip3 uninstall torch torchvision
pip3 install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### Embedding Model Download

First run downloads embedding model (~100MB):
```bash
# This happens automatically on first use
python scripts/ingest_dependency_docs.py --package torch
# Downloading model: BAAI/bge-small-en-v1.5
```

Model is cached at `~/.cache/huggingface/` for future use.

### RAG System Not Returning Results

```bash
# Check if documents are indexed
python scripts/query_dependency_docs.py --stats

# If total_documents is 0, ingest documentation
python scripts/ingest_dependency_docs.py --all
```

## Example Workflow

### Complete Setup (15-20 minutes)

```bash
# 1. Install dependencies
pip install -e .

# 2. Validate current state
python scripts/validate_dependencies.py

# 3. Ingest core dependency docs
python scripts/ingest_dependency_docs.py --package torch --version 2.9.1
python scripts/ingest_dependency_docs.py --package numpy
python scripts/ingest_dependency_docs.py --package fastapi

# 4. Query documentation
python scripts/query_dependency_docs.py "CUDA support in PyTorch"

# 5. Check stats
python scripts/query_dependency_docs.py --stats
```

### Query Examples

```bash
# PyTorch questions
python scripts/query_dependency_docs.py "What's new in PyTorch 2.9.1?"
python scripts/query_dependency_docs.py "How to use torch.compile?"
python scripts/query_dependency_docs.py "CUDA memory management"

# FastAPI questions
python scripts/query_dependency_docs.py --dependency fastapi "WebSocket streaming"
python scripts/query_dependency_docs.py --dependency fastapi "async endpoints"

# Quantum computing questions
python scripts/query_dependency_docs.py --dependency qiskit "quantum circuit basics"
python scripts/query_dependency_docs.py "compare qiskit pennylane cirq"

# Development tools
python scripts/query_dependency_docs.py --dependency pytest "fixtures"
python scripts/query_dependency_docs.py --dependency black "configuration"
```

## Files Reference

### New Files Created

```
src/cogsyndelta/documentation/
├── __init__.py                    # Module exports
├── rag_system.py                  # DependencyDocsRAG class
├── ingestion.py                   # DocumentIngestionPipeline
├── validators.py                  # DependencyValidator
└── README.md                      # Detailed documentation

scripts/
├── ingest_dependency_docs.py      # Ingest docs into RAG
├── validate_dependencies.py       # Validate imports
└── query_dependency_docs.py       # Query RAG system

DEPENDENCIES.md                    # Comprehensive dependency reference
DEPENDENCY_UPDATE_SUMMARY.md       # Summary of updates
```

### Updated Files

```
requirements.txt                   # Updated to exact versions
requirements-dev.txt              # Updated dev dependencies
pyproject.toml                    # Updated package config
INSTALL.md                        # Updated installation guide
GPU_COMPATIBILITY.md              # Completely rewritten
README.md                         # Updated badges and info
```

## Next Steps

1. ✅ Install dependencies: `pip install -e .`
2. ✅ Validate imports: `python scripts/validate_dependencies.py`
3. ✅ Ingest core docs: `python scripts/ingest_dependency_docs.py --all`
4. ✅ Query documentation: `python scripts/query_dependency_docs.py "your question"`
5. 🔧 Setup akula-prime: Install CUDA 12.8 and PyTorch 2.9.1
6. 🔧 Run GPU benchmarks: `python benchmarks/rtx5080_benchmark.py`
7. 📅 Schedule monthly review: Check for dependency updates

## Support

- **Documentation**: See `DEPENDENCIES.md`, `INSTALL.md`, `GPU_COMPATIBILITY.md`
- **Issues**: https://github.com/tzervas/CogSynDelta/issues
- **Discussions**: https://github.com/tzervas/CogSynDelta/discussions

---

**All dependency versions verified as of 2026-01-18**
