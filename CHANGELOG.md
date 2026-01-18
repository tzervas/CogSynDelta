# Changelog

All notable changes to CogSynDelta will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project organization cleanup and documentation consolidation
- GitHub Spec-Kit integration (AGENTS.md, templates, spec-driven workflow)
- Architecture Decision Records (ADRs) reconstructed from project history
- Branch protection and GitFlow branching strategy
- License tracking system for dependency compliance
- Conventional commits enforcement via pre-commit
- Google-style docstring standards documentation

### Changed
- Moved legacy documentation to `docs/archive/` with tarball preservation
- Updated project structure for cleaner root directory

## [0.2.0] - 2026-01-18

### Added
- Comprehensive test suite (34/34 tests passing)
  - Agent tests (`test_agents.py`)
  - API tests (`test_api.py`, `test_api_adk.py`)
  - Optimization tests (`test_optimization.py`)
- GitHub Actions CI/CD pipeline with UV package manager
- Code quality tooling (ruff, mypy strict mode)
- Dynamic badges for CI status, coverage, version
- Security workflow with CodeQL and dependency review (gated)

### Changed
- Migrated to UV package manager from pip
- Updated to Python 3.14 with `uv.lock` for reproducible builds
- Fixed memory retrieval similarity thresholds
- Updated `torch.load()` calls for `weights_only=True` (PyTorch 2.6+)
- Fixed import paths across codebase

### Fixed
- Division-by-zero bug in benchmarks
- Type annotations for torch/numpy interop
- Timestamp assignment in dense embeddings
- Self-assignment in loop detection test

### Security
- Added security scanning workflow (CodeQL, Snyk integration ready)
- Dependency review for PRs (when GHAS enabled)

## [0.1.0] - 2026-01-18

### Added
- **Core Architecture**
  - PCN-VAE-GAN hybrid with three-phase self-improvement (exploratory, refinement, culling)
  - VL-JEPA (Vision-Language Joint Embedding Predictive Architecture)
  - mHC (Moderated Hyper Connections) for information flow control
  - Intelligent Interconnect Manager for component coordination
  - Model sectioning with dynamic loading for memory efficiency

- **Memory Systems**
  - Tiered memory architecture (active → short-term → long-term)
  - Dense differential embeddings with 10-100x compression
  - Lossless compaction using SVD-based basis vectors
  - Memory persistence with automatic archiving
  - Unified tools interface for memory operations
  - FAISS-based similarity search

- **Self-Improving Agents**
  - Multi-language code generation (SWE/AIE/SWD/AID paradigms)
  - Skill registration and learning system
  - Loop detection and safeguards against infinite improvement cycles

- **API & Integration**
  - FastAPI REST server with OpenAPI documentation
  - WebSocket streaming support for real-time updates
  - Google ADK compliance (A2A protocol support)

- **GPU Support**
  - PyTorch 2.9.1 with CUDA 12.8/12.6/12.4 support
  - Triton 3.5.1 for kernel optimization
  - Multi-GPU index configuration (NVIDIA/ROCm/CPU fallback)

- **Quantum Computing** (Stubbed)
  - Quantum compute module with classical simulation fallbacks
  - FutureWarning emission when quantum features used in stub mode
  - Architecture ready for quantum backends when Python 3.14 support arrives

### Known Issues
- Quantum libraries (qiskit, pennylane, cirq) blocked by Python 3.14 compatibility
- Fireworks AI integration blocked by ruff version conflict

---

## Release Notes Format

Each release includes:
- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Features to be removed in future
- **Removed**: Features removed in this release
- **Fixed**: Bug fixes
- **Security**: Security-related changes

[Unreleased]: https://github.com/tzervas/CogSynDelta/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/tzervas/CogSynDelta/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/tzervas/CogSynDelta/releases/tag/v0.1.0
