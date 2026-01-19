# Implementation Tasks: Model Architecture Benchmarking Suite

**Spec**: [spec.md](spec.md)
**Plan**: [plan.md](plan.md)
**Created**: 2026-01-18
**Status**: In Progress

## Task Overview

| ID | Task | Status | Est | Phase |
|----|------|--------|-----|-------|
| T1 | Create benchmark utilities (metrics, normalization) | ⬜ | 2h | 1 |
| T2 | Create core model benchmark runner | ⬜ | 3h | 1 |
| T3 | Add PCN-VAE-GAN specific benchmarks | ⬜ | 2h | 2 |
| T4 | Add VL-JEPA benchmarks | ⬜ | 2h | 2 |
| T5 | Add mHC/Interconnect benchmarks | ⬜ | 2h | 2 |
| T6 | Curate industry baseline data with citations | ⬜ | 3h | 3 |
| T7 | Create industry comparison framework | ⬜ | 2h | 3 |
| T8 | Add compression baselines (FAISS, ScaNN, SVD) | ⬜ | 2h | 3 |
| T9 | Implement delta tracking and trends | ⬜ | 2h | 4 |
| T10 | CI integration and baseline capture | ⬜ | 1h | 4 |

**Legend**: ⬜ Not Started | 🔄 In Progress | ✅ Complete | ❌ Blocked

---

## Phase 1: Core Infrastructure

### T1: Create benchmark utilities

**Status**: ⬜ Not Started
**Estimate**: 2 hours
**Dependencies**: None

**Description**:
Create utility modules for metric computation and normalization that will be used across all benchmarks.

**Acceptance Criteria**:
- [ ] `metrics.py` computes latency percentiles (p50, p95, p99)
- [ ] `metrics.py` computes throughput with proper warmup
- [ ] `metrics.py` measures peak GPU memory accurately
- [ ] `normalization.py` computes per-billion-params metrics
- [ ] `normalization.py` computes per-TFLOP metrics
- [ ] All functions have Google-style docstrings

**Files to Create**:
- `benchmarks/utils/__init__.py`
- `benchmarks/utils/metrics.py`
- `benchmarks/utils/normalization.py`

---

### T2: Create core model benchmark runner

**Status**: ⬜ Not Started
**Estimate**: 3 hours
**Dependencies**: T1

**Description**:
Create the main benchmark runner with methodology disclosure header, result dataclasses, and JSON output.

**Acceptance Criteria**:
- [ ] `ModelBenchmarkResult` dataclass with all required fields
- [ ] Methodology header printed at start of every run
- [ ] Results saved to JSON with timestamp, git SHA, hardware spec
- [ ] CLI interface with `--help`
- [ ] Statistical validation (100 iterations, warmup)

**Files to Create**:
- `benchmarks/model_benchmarks.py`
- `benchmark_results/baselines/.gitkeep`
- `benchmark_results/history/.gitkeep`

---

## Phase 2: Model-Specific Benchmarks

### T3: Add PCN-VAE-GAN benchmarks

**Status**: ⬜ Not Started
**Estimate**: 2 hours
**Dependencies**: T2

**Description**:
Benchmark the PCN-VAE-GAN component: inference latency, reconstruction quality, latent space metrics.

**Acceptance Criteria**:
- [ ] Encoder inference latency at batch sizes [1, 8, 32, 64]
- [ ] Decoder inference latency
- [ ] Full VAE forward pass latency
- [ ] Reconstruction MSE and cosine similarity
- [ ] KL divergence (latent space quality)
- [ ] Memory usage during inference

**Files to Modify**:
- `benchmarks/model_benchmarks.py`: Add `PCNVAEGANBenchmark` class

---

### T4: Add VL-JEPA benchmarks

**Status**: ⬜ Not Started
**Estimate**: 2 hours
**Dependencies**: T2

**Description**:
Benchmark VL-JEPA component: vision encoding, temporal memory, joint embedding alignment.

**Acceptance Criteria**:
- [ ] VisionEncoder throughput (images/sec)
- [ ] TemporalMemoryBank read/write latency
- [ ] JointEmbeddingSpace alignment quality
- [ ] End-to-end JEPA inference latency
- [ ] Temporal context window performance scaling

**Files to Modify**:
- `benchmarks/model_benchmarks.py`: Add `VLJEPABenchmark` class

---

### T5: Add mHC/Interconnect benchmarks

**Status**: ⬜ Not Started
**Estimate**: 2 hours
**Dependencies**: T2

**Description**:
Benchmark mHC gating and interconnect manager: gating efficiency, information flow throughput.

**Acceptance Criteria**:
- [ ] mHCGate forward pass latency
- [ ] Gate activation statistics (mean, variance)
- [ ] InterconnectManager routing throughput
- [ ] Cross-component communication latency
- [ ] Memory overhead of interconnect system

**Files to Modify**:
- `benchmarks/model_benchmarks.py`: Add `InterconnectBenchmark` class

---

## Phase 3: Industry Comparisons

### T6: Curate industry baseline data

**Status**: ⬜ Not Started
**Estimate**: 3 hours
**Dependencies**: None

**Description**:
Research and curate published benchmark data for industry models with proper citations.

**Acceptance Criteria**:
- [ ] GPT-2 (small, medium, large): throughput, memory, latency from OpenAI/HF
- [ ] BERT (base, large): throughput, embedding quality from Google/HF
- [ ] LLaMA (7B, 13B): throughput, memory from Meta paper
- [ ] Mistral-7B: throughput, memory from Mistral paper
- [ ] Sentence-Transformers: embedding throughput from SBERT
- [ ] CLIP: vision-language alignment from OpenAI
- [ ] Each entry includes: source URL, publication date, hardware used

**Files to Create**:
- `benchmarks/data/industry_baselines.json`

---

### T7: Create industry comparison framework

**Status**: ⬜ Not Started
**Estimate**: 2 hours
**Dependencies**: T2, T6

**Description**:
Build comparison table generation that shows CogSynDelta vs industry models with normalizations.

**Acceptance Criteria**:
- [ ] Comparison table with raw values
- [ ] Throughput-per-billion-params column
- [ ] Memory-per-param column
- [ ] Source citation for each industry number
- [ ] Markdown and JSON output formats
- [ ] Honest display of unfavorable comparisons

**Files to Create**:
- `benchmarks/model_comparisons.py`

---

### T8: Add compression baselines

**Status**: ⬜ Not Started
**Estimate**: 2 hours
**Dependencies**: T2

**Description**:
Compare CogSynDelta memory compression against FAISS, ScaNN, and standard methods.

**Acceptance Criteria**:
- [ ] FAISS PQ/OPQ baseline numbers from published benchmarks
- [ ] ScaNN baseline numbers
- [ ] Standard SVD/PCA baseline (computed locally)
- [ ] Metrics: recall@k, compression ratio, latency, fidelity
- [ ] Comparison table with methodology notes

**Files to Create**:
- `benchmarks/compression_baselines.py`

---

## Phase 4: Tracking & CI

### T9: Implement delta tracking

**Status**: ⬜ Not Started
**Estimate**: 2 hours
**Dependencies**: T2

**Description**:
Add baseline comparison, delta computation, and stability trend analysis.

**Acceptance Criteria**:
- [ ] Load baseline from JSON file
- [ ] Compute delta and delta percentage for each metric
- [ ] Direction indicator (↑ better, ↓ worse, ↔ stable)
- [ ] Statistical significance indicator (if change > 2 std dev)
- [ ] Trend analysis over historical runs (mean, std, min, max)

**Files to Create**:
- `benchmarks/tracking.py`

---

### T10: CI integration and baseline capture

**Status**: ⬜ Not Started
**Estimate**: 1 hour
**Dependencies**: T9

**Description**:
Capture initial baseline and add CI workflow for automated benchmark tracking.

**Acceptance Criteria**:
- [ ] Initial baseline captured in `model_baseline.json`
- [ ] CI workflow runs benchmarks on schedule (weekly)
- [ ] Results stored in history directory
- [ ] Warning logged if regression >5%

**Files to Create/Modify**:
- `benchmark_results/baselines/model_baseline.json`: Initial baseline
- `.github/workflows/model-benchmark.yml`: CI workflow

---

## Completion Checklist

### Code Quality
- [ ] All tasks complete (✅)
- [ ] `uv run ruff check benchmarks/` passes
- [ ] `uv run mypy benchmarks/` passes
- [ ] Quality score ≥ 90

### Testing
- [ ] Unit tests for metrics and normalization
- [ ] Integration test for full benchmark run
- [ ] Coverage ≥ 80% for new code

### Documentation
- [ ] Docstrings complete (Google style)
- [ ] CHANGELOG.md updated
- [ ] benchmarks/README.md updated

### Benchmarks
- [ ] Initial baseline captured
- [ ] Industry comparisons validated against sources
- [ ] Methodology header in all outputs

---

*Template based on [GitHub Spec-Kit](https://github.com/github/spec-kit)*
