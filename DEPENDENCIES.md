# Dependency Documentation & Validation

**Last Updated**: 2026-01-18  
**Validation Status**: ✅ All versions verified against PyPI

This document provides comprehensive information about all project dependencies, their versions, compatibility requirements, and validation status.

## Quick Reference

| Category | Package | Version | Release Date | Notes |
|----------|---------|---------|--------------|-------|
| **Core** | torch | 2.9.1 | 2025-11-12 | CUDA 12.6/12.8 |
| **Core** | torchvision | 0.24.1 | 2025-11-12 | Requires torch==2.9.1 |
| **Core** | numpy | 2.4.1 | 2026-01-XX | Latest stable |
| **Core** | pyyaml | 6.0.3 | 2024-XX-XX | Stable |
| **API** | fastapi | 0.128.0 | 2026-01-XX | Latest stable |
| **API** | uvicorn | 0.40.0 | 2026-01-XX | ASGI server |
| **API** | pydantic | 2.12.5 | 2026-01-XX | Data validation |
| **API** | python-multipart | 0.0.21 | 2026-01-XX | File uploads |
| **API** | websockets | 16.0 | 2026-01-XX | WebSocket support |
| **RAG** | llama-index-core | 0.14.12 | 2026-01-XX | RAG system |
| **RAG** | llama-index-embeddings-huggingface | 0.14.12 | 2026-01-XX | Embeddings |
| **RAG** | llama-index-vector-stores-faiss | 0.14.12 | 2026-01-XX | Vector store |
| **RAG** | faiss-cpu | 1.9.0.post1 | 2025-XX-XX | Vector search |
| **Quantum** | qiskit | 2.3.0 | 2026-01-XX | IBM Quantum |
| **Quantum** | pennylane | 0.44.0 | 2026-01-XX | Differentiable QC |
| **Quantum** | cirq | 1.6.1 | 2026-01-XX | Google Quantum |
| **Vision** | opencv-python | 4.13.0.90 | 2026-01-XX | Computer vision |
| **Vision** | pillow | 12.1.0 | 2026-01-XX | Image processing |
| **Dev** | pytest | 9.0.2 | 2026-01-XX | Testing framework |
| **Dev** | pytest-cov | 7.0.0 | 2026-01-XX | Coverage reporting |
| **Dev** | pytest-asyncio | 1.3.0 | 2026-01-XX | Async testing |
| **Dev** | black | 26.1.0 | 2026-01-XX | Code formatter |
| **Dev** | ruff | 0.14.13 | 2026-01-XX | Linter |
| **Dev** | mypy | 1.19.1 | 2026-01-XX | Type checker |
| **Dev** | pre-commit | 4.5.1 | 2026-01-XX | Git hooks |
| **Dev** | sphinx | 9.1.0 | 2026-01-XX | Documentation |
| **Dev** | sphinx-rtd-theme | 3.1.0 | 2026-01-XX | Sphinx theme |

## Core Dependencies

### PyTorch 2.9.1

**Official Resources**:
- PyPI: https://pypi.org/project/torch/2.9.1/
- Docs: https://pytorch.org/docs/stable/
- GitHub: https://github.com/pytorch/pytorch/releases/tag/v2.9.1
- Get Started: https://pytorch.org/get-started/locally/

**Key Features**:
- CUDA 12.6 and 12.8 support
- torch.compile() for graph optimization
- Improved memory efficiency
- Enhanced mixed precision training
- Better distributed training support

**CUDA Compatibility**:
```bash
# CUDA 12.8 (recommended for RTX 5080)
pip3 install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.6
pip3 install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126

# CPU only
pip3 install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
```

**Compute Capability Support**:
- sm_50 - Maxwell (GTX 900 series)
- sm_60, sm_61 - Pascal (GTX 10 series)
- sm_70, sm_75 - Volta, Turing (V100, RTX 20 series)
- sm_80, sm_86 - Ampere (A100, RTX 30 series)
- sm_89 - Ada Lovelace (RTX 40 series)
- sm_90, sm_90a - Hopper (H100)

**RTX 5080 (Blackwell) Status**: Requires verification - check official compatibility matrix

### torchvision 0.24.1

**Official Resources**:
- PyPI: https://pypi.org/project/torchvision/0.24.1/
- Docs: https://pytorch.org/vision/stable/
- GitHub: https://github.com/pytorch/vision

**Key Features**:
- Computer vision models and transforms
- Pre-trained model weights
- Dataset utilities
- Image/video processing

**Requirement**: Must use with torch==2.9.1

### NumPy 2.4.1

**Official Resources**:
- PyPI: https://pypi.org/project/numpy/2.4.1/
- Docs: https://numpy.org/doc/stable/
- GitHub: https://github.com/numpy/numpy

**Key Features**:
- Multi-dimensional array operations
- Linear algebra
- Fourier transforms
- Random number generation

**Breaking Changes**: NumPy 2.x has breaking changes from 1.x - see migration guide

## API Dependencies

### FastAPI 0.128.0

**Official Resources**:
- PyPI: https://pypi.org/project/fastapi/0.128.0/
- Docs: https://fastapi.tiangolo.com/
- GitHub: https://github.com/fastapi/fastapi

**Key Features**:
- Modern async Python web framework
- Automatic API documentation
- Type validation with Pydantic
- OpenAPI/Swagger support

### Uvicorn 0.40.0

**Official Resources**:
- PyPI: https://pypi.org/project/uvicorn/0.40.0/
- Docs: https://www.uvicorn.org/
- GitHub: https://github.com/encode/uvicorn

**Key Features**:
- Lightning-fast ASGI server
- WebSocket support
- HTTP/2 support
- Production-ready

### Pydantic 2.12.5

**Official Resources**:
- PyPI: https://pypi.org/project/pydantic/2.12.5/
- Docs: https://docs.pydantic.dev/latest/
- GitHub: https://github.com/pydantic/pydantic

**Key Features**:
- Data validation using Python type hints
- Settings management
- JSON schema generation
- Performance improvements in v2

**Breaking Changes**: Pydantic v2 has significant changes from v1

## RAG System Dependencies

### LlamaIndex 0.14.12

**Official Resources**:
- PyPI: https://pypi.org/project/llama-index/0.14.12/
- Docs: https://docs.llamaindex.ai/
- GitHub: https://github.com/run-llama/llama_index

**Key Features**:
- Document ingestion and indexing
- Vector store integration
- Query engines
- Retrieval-augmented generation

**Packages**:
- `llama-index-core`: Core functionality
- `llama-index-embeddings-huggingface`: HuggingFace embeddings
- `llama-index-vector-stores-faiss`: FAISS vector store

### FAISS 1.9.0

**Official Resources**:
- PyPI: https://pypi.org/project/faiss-cpu/1.9.0.post1/
- Docs: https://faiss.ai/
- GitHub: https://github.com/facebookresearch/faiss

**Key Features**:
- Efficient similarity search
- Clustering of dense vectors
- GPU acceleration (faiss-gpu)

**Note**: Use `faiss-cpu` for CPU-only, `faiss-gpu` for GPU acceleration

## Quantum Computing Dependencies (Optional)

### Qiskit 2.3.0

**Official Resources**:
- PyPI: https://pypi.org/project/qiskit/2.3.0/
- Docs: https://docs.quantum.ibm.com/
- GitHub: https://github.com/Qiskit/qiskit

**Key Features**:
- IBM Quantum platform integration
- Quantum circuit design
- Quantum algorithms
- Hardware backend access

### PennyLane 0.44.0

**Official Resources**:
- PyPI: https://pypi.org/project/pennylane/0.44.0/
- Docs: https://docs.pennylane.ai/
- GitHub: https://github.com/PennyLaneAI/pennylane

**Key Features**:
- Differentiable quantum computing
- PyTorch/TensorFlow integration
- Variational quantum algorithms
- Multiple backend support

### Cirq 1.6.1

**Official Resources**:
- PyPI: https://pypi.org/project/cirq/1.6.1/
- Docs: https://quantumai.google/cirq
- GitHub: https://github.com/quantumlib/Cirq

**Key Features**:
- Google Quantum AI framework
- Quantum circuit simulation
- QAOA and VQE algorithms
- Cloud quantum computer access

## Vision Dependencies (Optional)

### OpenCV 4.13.0.90

**Official Resources**:
- PyPI: https://pypi.org/project/opencv-python/4.13.0.90/
- Docs: https://docs.opencv.org/
- GitHub: https://github.com/opencv/opencv

**Key Features**:
- Computer vision algorithms
- Image/video processing
- Object detection and tracking
- Camera interfacing

### Pillow 12.1.0

**Official Resources**:
- PyPI: https://pypi.org/project/pillow/12.1.0/
- Docs: https://pillow.readthedocs.io/
- GitHub: https://github.com/python-pillow/Pillow

**Key Features**:
- Image file I/O
- Image processing operations
- Format conversion
- Drawing and text

## Development Dependencies

### pytest 9.0.2

**Official Resources**:
- PyPI: https://pypi.org/project/pytest/9.0.2/
- Docs: https://docs.pytest.org/
- GitHub: https://github.com/pytest-dev/pytest

**Key Features**:
- Simple and powerful testing
- Fixtures and parametrization
- Plugin ecosystem
- Coverage integration

### Black 26.1.0

**Official Resources**:
- PyPI: https://pypi.org/project/black/26.1.0/
- Docs: https://black.readthedocs.io/
- GitHub: https://github.com/psf/black

**Key Features**:
- Uncompromising code formatter
- PEP 8 compliant
- Minimal configuration
- Fast

### Ruff 0.14.13

**Official Resources**:
- PyPI: https://pypi.org/project/ruff/0.14.13/
- Docs: https://docs.astral.sh/ruff/
- GitHub: https://github.com/astral-sh/ruff

**Key Features**:
- Extremely fast Python linter
- Replaces flake8, pylint, isort, and more
- Written in Rust
- Auto-fix capabilities

### Mypy 1.19.1

**Official Resources**:
- PyPI: https://pypi.org/project/mypy/1.19.1/
- Docs: https://mypy.readthedocs.io/
- GitHub: https://github.com/python/mypy

**Key Features**:
- Static type checker
- Gradual typing support
- Type inference
- Plugin system

### Sphinx 9.1.0

**Official Resources**:
- PyPI: https://pypi.org/project/sphinx/9.1.0/
- Docs: https://www.sphinx-doc.org/
- GitHub: https://github.com/sphinx-doc/sphinx

**Key Features**:
- Documentation generator
- reStructuredText and Markdown support
- Auto-documentation from docstrings
- Extensive extension system

## Dependency Management Tools

### Validation Scripts

```bash
# Validate all dependencies
python scripts/validate_dependencies.py

# Generate JSON report
python scripts/validate_dependencies.py --json

# Save report to file
python scripts/validate_dependencies.py --report dependency_report.txt
```

### RAG Documentation System

```bash
# Ingest documentation for specific package
python scripts/ingest_dependency_docs.py --package torch --version 2.9.1

# Ingest all configured dependencies
python scripts/ingest_dependency_docs.py --all

# Query documentation
python scripts/query_dependency_docs.py "What CUDA versions does PyTorch 2.9.1 support?"

# List indexed dependencies
python scripts/query_dependency_docs.py --list-deps

# Show RAG statistics
python scripts/query_dependency_docs.py --stats

# Get info about specific dependency
python scripts/query_dependency_docs.py --info torch
```

## Update Cadence

**Manual/Monthly Updates**:
- Review dependency updates monthly
- Check for security advisories
- Test compatibility before updating
- Document breaking changes

**Verification Process**:
1. Check PyPI for latest stable version
2. Review release notes and changelog
3. Verify compatibility with other dependencies
4. Test in development environment
5. Update documentation
6. Commit with detailed message

## Compatibility Notes

### Python Version Support

- **Minimum**: Python 3.9
- **Maximum**: Python 3.13 (tested)
- **Recommended**: Python 3.11 or 3.12

### CUDA Version Support

- **PyTorch 2.9.1**: CUDA 12.6, 12.8
- **Recommended**: CUDA 12.8 for RTX 5080

### Operating System Support

- **Linux**: Primary development platform (Ubuntu 22.04+)
- **Windows**: Supported with WSL2 or native
- **macOS**: Supported (CPU only, or MPS for M1/M2)

## Troubleshooting

### Common Issues

**Issue**: Import errors after installation
```bash
# Solution: Reinstall in development mode
pip install -e . --force-reinstall
```

**Issue**: CUDA not available
```bash
# Solution: Reinstall PyTorch with CUDA
pip3 uninstall torch torchvision
pip3 install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

**Issue**: Dependency conflicts
```bash
# Solution: Create fresh virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e ".[all]"
```

**Issue**: Outdated pip/setuptools
```bash
# Solution: Update pip and setuptools
pip install --upgrade pip setuptools wheel
```

## References

- **PyPI**: https://pypi.org/
- **PyTorch**: https://pytorch.org/
- **CUDA Toolkit**: https://developer.nvidia.com/cuda-toolkit
- **Python Packaging**: https://packaging.python.org/
- **Semantic Versioning**: https://semver.org/

## See Also

- [INSTALL.md](INSTALL.md) - Installation instructions
- [GPU_COMPATIBILITY.md](GPU_COMPATIBILITY.md) - GPU setup and compatibility
- [requirements.txt](requirements.txt) - Core dependencies
- [requirements-dev.txt](requirements-dev.txt) - Development dependencies
- [pyproject.toml](pyproject.toml) - Package configuration
