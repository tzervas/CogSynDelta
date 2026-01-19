# Implementation Plan: Model Architecture Benchmarking Suite

**Spec**: [specs/model-benchmarking/spec.md](spec.md)
**Created**: 2026-01-18
**Status**: In Progress
**Author**: CogSynDelta Team
**Estimated Effort**: L (3-5 days)

## Overview

Implement a model-centric benchmarking suite that measures CogSynDelta's actual model performance (not hardware), compares against industry baselines, and tracks deltas over time. This replaces the current hardware-focused benchmarks with scientifically rigorous, honest model evaluation.

## Prerequisites

- [x] Spec reviewed and approved
- [x] Constitution compliance verified (no unsubstantiated claims)
- [ ] Baseline benchmark data captured from current state

## Implementation Phases

### Phase 1: Core Benchmark Infrastructure (Est: 1 day)

**Goal**: Create reusable benchmark primitives and result tracking

**Changes**:
1. `benchmarks/model_benchmarks.py`: Core benchmark runner with methodology disclosure, statistical validation, and result dataclasses
2. `benchmarks/utils/metrics.py`: Metric computation (latency percentiles, throughput, memory)
3. `benchmarks/utils/normalization.py`: Industry-standard normalizations (per-param, per-FLOP)
4. `benchmark_results/baselines/`: Directory structure for baseline storage

**Validation**:
- [ ] Can run `python -m benchmarks.model_benchmarks --help`
- [ ] Methodology header appears in output
- [ ] Results saved to JSON with all required fields

### Phase 2: Model-Specific Benchmarks (Est: 1.5 days)

**Goal**: Implement benchmarks for each CogSynDelta component

**Changes**:
1. `benchmarks/model_benchmarks.py`: Add PCN-VAE-GAN benchmarks (inference, reconstruction quality, latent space metrics)
2. `benchmarks/model_benchmarks.py`: Add VL-JEPA benchmarks (vision encoding, temporal memory, joint embedding)
3. `benchmarks/model_benchmarks.py`: Add mHC benchmarks (gating efficiency, interconnect throughput)
4. `benchmarks/model_benchmarks.py`: Add end-to-end pipeline benchmarks

**Validation**:
- [ ] Each component benchmark runs independently
- [ ] Quality metrics match expected ranges from test suite
- [ ] Memory measurements capture peak usage

### Phase 3: Industry Comparisons (Est: 1 day)

**Goal**: Add honest comparisons against published model benchmarks

**Changes**:
1. `benchmarks/model_comparisons.py`: Industry baseline data (GPT-2, BERT, LLaMA, Mistral, Sentence-Transformers, CLIP)
2. `benchmarks/model_comparisons.py`: Comparison table generation with citations
3. `benchmarks/data/industry_baselines.json`: Curated published benchmark data with sources
4. `benchmarks/compression_baselines.py`: Compression comparison vs FAISS, ScaNN, SVD

**Validation**:
- [ ] Every industry number has source citation
- [ ] Normalizations computed correctly
- [ ] Unfavorable comparisons displayed without modification

### Phase 4: Tracking & CI Integration (Est: 0.5 days)

**Goal**: Automated delta tracking and stability metrics

**Changes**:
1. `benchmarks/tracking.py`: Baseline comparison, delta computation, trend analysis
2. `benchmark_results/baselines/model_baseline.json`: Initial baseline capture
3. `.github/workflows/benchmark.yml`: CI integration for automated tracking

**Validation**:
- [ ] Delta percentages computed correctly
- [ ] Stability metrics (std dev, trend) work with historical data
- [ ] CI runs benchmarks on schedule

## Technical Approach

### Architecture

```
benchmarks/
├── model_benchmarks.py      # Core model benchmark runner
├── model_comparisons.py     # Industry comparison framework
├── compression_baselines.py # Compression comparison (FAISS, etc.)
├── tracking.py              # Delta and trend tracking
├── utils/
│   ├── metrics.py           # Metric computation
│   └── normalization.py     # Per-param, per-FLOP normalizations
└── data/
    └── industry_baselines.json  # Published benchmark data

benchmark_results/
├── baselines/
│   └── model_baseline.json  # Current baseline for delta tracking
├── history/                 # Historical benchmark runs
└── reports/                 # Generated comparison reports
```

### Key Design Decisions

1. **Separate model vs hardware benchmarks**: Current `gpu_benchmark.py` tests hardware; new `model_benchmarks.py` tests the model
2. **Methodology disclosure first**: Every output starts with hardware, date, version, methodology notes
3. **Raw + normalized**: Always show both to enable fair comparison

### Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Industry baselines outdated | Med | Med | Include publication date, update quarterly |
| Benchmark variance too high | Med | Low | 100 iterations, report std dev, flag unstable |
| Model architecture changes | High | Med | Baseline versioning, invalidation on breaking changes |

## Testing Strategy

### Unit Tests
- `tests/test_model_benchmarks.py`: Metric computation, normalization formulas
- `tests/test_tracking.py`: Delta computation, trend analysis

### Integration Tests
- Full benchmark run on CI (smoke test, subset of metrics)
- Comparison generation produces valid output

### Benchmark Validation
- Cross-check against existing `benchmarks/run.py` results
- Verify memory measurements against `nvidia-smi`

## Rollout Plan

1. **Feature Branch**: `feat/specs-benchmarks-logging-infrastructure`
2. **Code Review**: PR with benchmark methodology review
3. **Baseline Capture**: Run full benchmark, save as initial baseline
4. **Documentation**: Update README with benchmark instructions
5. **CI**: Enable scheduled benchmark runs

## Documentation Updates

- [ ] Update [benchmarks/README.md] with new benchmark usage
- [ ] Update [docs/DEVELOPMENT_STANDARDS.md] with benchmark requirements
- [ ] Update [CHANGELOG.md] with new benchmarking capabilities
- [ ] Add methodology notes to [benchmark_results/README.md]

## Related

- Spec: [spec.md](spec.md)
- Tasks: [tasks.md](tasks.md)
- ADRs: 0004-graceful-degradation-patterns
- Constitution: memory/constitution.md

---

*Template based on [GitHub Spec-Kit](https://github.com/github/spec-kit)*
