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
- Benchmark visualization module (`benchmarks/visualization.py`)
  - Sparkline rendering for terminal output
  - Trend analysis with regression
  - Markdown and HTML export
  - **NEW**: Scale indicators with quality tiers (🟢🟡🟠🔴)
  - **NEW**: Visual progress bars for metrics (`█████░░░░░`)
  - **NEW**: `ScaledMetricDisplay` for proper unit/scale formatting
- Context-efficient memory techniques (`cogsyndelta/memory/context_efficient.py`)
  - ChunkedCompactor for memory-efficient large batch processing
  - ImportanceContextPruner (ToMe-inspired token merging)
  - LatentCache with LRU eviction and hash-based invalidation
- Enhanced benchmark history format (`benchmarks/history_format.py`)
  - `MetricUnit` and `MetricScale` enums for proper scientific units
  - `ScaleIndicator` with visual bars and quality tiers
  - `MetricEntry` with full tagging, provenance, and metadata
  - `ModelSizeConfig` defining 5 model size tiers (tiny→xlarge)
  - `RunMetadata` capturing full environment provenance
  - `EnhancedBenchmarkRecord` for rich persistent storage
  - **Eco-Minded Metrics** (power & efficiency):
    - `PowerMetrics`: GPU/CPU power (watts), energy (joules), temperature, throttling detection
    - `ComputeUtilizationMetrics`: GPU/memory utilization %, clock speeds, SM occupancy
    - `EfficiencyMetrics`: samples/watt, GFLOPS/watt, CO₂ estimates, cost/1M samples
    - `PerformanceTrend`: Delta tracking with regression detection
  - Extended `MetricUnit` enum with power units (W, kW, J, Wh) and efficiency (samples/W, GFLOP/W)
  - `TYPICAL_RANGES` extended for power, temperature, utilization, and IPC metrics
  - **Latent Space Metrics** (CogSynDelta-specific):
    - `LatentSpaceMetrics`: engrams/sec, engrams/watt, encoding fidelity, latent timing
    - New units: `ENGRAMS_PER_SEC`, `ENGRAMS_PER_WATT`, `BITS_PER_DIM`
  - **Normalized Comparison Metrics**:
    - `NormalizedMetrics`: speedup vs baseline, throughput/param, equivalent tokens/sec
    - Reference baselines: GPT-2 (small/medium/large), LLaMA-7B, ViT-base, ResNet-50
  - **Diagnostic Metrics** for bottleneck analysis:
    - `DiagnosticMetrics`: primary bottleneck detection, severity, recommendations
    - Explains "compute", "memory", "io", "latency" bottlenecks with tuning guidance
- **NEW**: Modular metrics package (`benchmarks/metrics/`)
  - `base.py`: `BaseMetric` protocol, `MetricUnit` enum, `MetricRegistry`
  - `power.py`: `PowerMetrics`, `EcoMetrics` with NVIDIA pynvml integration
  - `compute.py`: `ComputeUtilizationMetrics` with throttle detection
  - `latent.py`: `LatentSpaceMetrics` for engram throughput
  - `normalized.py`: `NormalizedMetrics` for cross-architecture comparison
  - `diagnostic.py`: `DiagnosticMetrics` for bottleneck analysis
  - `display.py`: `MetricFormatter`, `ScaledMetricDisplay` for rich output
- ADR-0015: Context-Efficient Memory Techniques
- ADR-0016: Model Training Roadmap (5-phase curriculum)
- ADR-0017: Benchmark Metrics Architecture

### Changed
- Moved legacy documentation to `docs/archive/` with tarball preservation
- Updated project structure for cleaner root directory
- **Branch Policy**: Explicit "no direct merges to main" policy documented
  - All work flows: feature/* → develop → staging → main
  - Merges to staging/main are automated via CI/CD only

### Security
- Added `pip-audit` to dev dependencies for CVE scanning
- No known vulnerabilities found in dependency audit (2026-01-19)

### Backlogged
- **Model Size Cycling**: Infrastructure ready (`ModelSizeConfig`), implementation deferred
  - 5 size tiers: tiny (1M), small (10M), base (50M), large (200M), xlarge (1B)
  - Will implement when baseline training is complete

### Fixed
- **CRITICAL**: Benchmark import errors fixed
  - `PCN_VAE_GAN` → `PCNVAEGANHybrid` in model benchmarks
  - `InterconnectManager` → `IntelligentInterconnectManager`
  - `mHCGate` → `ModeratedHyperConnection`
- **OOM Protection**: LosslessCompactor now skipped on GPUs < 24GB
- **DenseEmbeddingEncoder**: CompactorAdapter now handles encode/decode tuple interface
- **Untrained Model Detection**: Benchmarks now flag fidelity < 0.3 as "untrained"
- Added detailed statistics report to compression benchmarks (`--detailed` flag)

## [0.3.2] - 2026-01-19

### Security
- **CRITICAL**: Updated aiohttp from >=3.11.0 to >=3.13.3
  - Fixes CVE-2025-30167 (HIGH): Request smuggling via crafted headers
  - Fixes CVE-2025-32667 (MEDIUM): HTTP response splitting
  - Fixes 7 additional low/medium severity vulnerabilities in aiohttp <=3.13.2

## [0.3.1] - 2026-01-19

### Added
- Compression validation test suite (`test_compression_calibration.py`)
- Research-informed ADRs (0012-0014) for VSA, V-JEPA 2, MRL+QINCo2
- BitNet 1.58-bit ternary exploration (ADR-0010, specs/ternary-bitnet/)

### Fixed
- ADR-0008 fidelity baseline corrected from 0.06 to 0.67
- Validated HighFidelityCompactor achieves 1.0 fidelity at ~2x compression

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
