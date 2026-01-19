# Feature Specification: Semantic Compression Research

**Feature Branch**: Research conducted in separate `feat/compression-*` branches
**Created**: 2026-01-18
**Status**: Research Phase
**Author**: CogSynDelta Team

## Summary

Research and evaluate improved embedding compression methods to close the gap between current measured fidelity (0.67 at 2x) and architecture target (>0.95 at 10-100x). This spec defines the research methodology, candidate techniques, evaluation criteria, and decision process for selecting the best approach via ADR.

## Background

### Current State

Per [ADR-0002](../../docs/adr/0002-tiered-memory-architecture.md), the tiered memory architecture targets "10-100x compression while maintaining reconstruction fidelity." Current benchmarks show:

| Compression | Measured Fidelity | Target |
|-------------|------------------|--------|
| 2x | 0.670 | >0.95 |
| 4x | 0.462 | >0.90 |
| 16x | 0.228 | >0.80 |

This gap requires investigation of improved compression techniques.

### Constraints

- **Python-only**: No Rust/C++ dependencies for core compression
- **Memory-efficient**: Must fit in GPU memory for real-time operation
- **Latency-sensitive**: Compression/decompression must add <10ms overhead
- **VL-JEPA compatible**: Must preserve semantic fidelity for downstream tasks

## Research Tracks

Each track will be developed in a separate feature branch with full benchmark suite.

### Track 1: Residual Vector Quantization (RVQ)

**Branch**: `feat/compression-rvq`
**Priority**: High (most promising for this architecture)

**Approach**:
- Iteratively quantize residuals from previous quantization stage
- 4-8 codebook stages with 256-1024 centroids each
- Expected: 10-20x compression at >0.90 fidelity

**Key research questions**:
1. Optimal number of codebook stages for our embedding dimension (512)
2. Codebook initialization strategy (k-means vs learned)
3. Trade-off between codebook size and reconstruction quality

### Track 2: Product Quantization (PQ)

**Branch**: `feat/compression-pq`
**Priority**: High (proven technique with FAISS reference)

**Approach**:
- Split 512-dim vectors into 8-16 subspaces
- Quantize each subspace independently
- Expected: 16-32x compression at >0.85 fidelity

**Key research questions**:
1. Optimal subspace count for semantic preservation
2. OPQ (Optimized Product Quantization) with rotation matrix
3. Fine-tuning codebooks on our specific embedding distribution

### Track 3: Matryoshka Embeddings

**Branch**: `feat/compression-matryoshka`
**Priority**: Medium (novel, less established)

**Approach**:
- Train embeddings with multi-resolution loss
- Truncate to lower dimensions at inference for compression
- Expected: Flexible compression with graceful quality degradation

**Key research questions**:
1. Integration with existing VAE training loop
2. Downstream task impact of truncation
3. Compatibility with VL-JEPA temporal memory

### Track 4: Neural Compression (VQ-VAE Style)

**Branch**: `feat/compression-neural-codec`
**Priority**: Medium (requires additional training)

**Approach**:
- Learned compression with discrete latent codes
- Hierarchical VQ-VAE with multiple quantization layers
- Expected: Highest quality at given compression, but slowest

**Key research questions**:
1. Training cost vs quality improvement
2. Inference latency impact
3. Generalization to unseen embedding distributions

## Evaluation Criteria

All research tracks will be evaluated on the same benchmarks:

### Primary Metrics (must-pass)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Cosine Fidelity @ 10x** | ≥0.90 | Reconstruction similarity |
| **Recall@10 @ 10x** | ≥0.95 | Retrieval accuracy |
| **Compression Latency** | <5ms | Per-sample encode time |
| **Decompression Latency** | <1ms | Per-sample decode time |

### Secondary Metrics (comparison)

| Metric | Purpose |
|--------|---------|
| **Memory footprint** | Codebook/model size |
| **Training time** | If learned components |
| **VL-JEPA task accuracy** | Downstream impact |
| **mHC routing fidelity** | Interconnect impact |

### Benchmark Protocol

1. Run on same hardware (RTX 5080)
2. Use standardized embedding dataset (10K samples from memory)
3. Report mean ± std across 5 runs
4. Include methodology disclosure header
5. Compare against baselines (FAISS PQ, SVD, current implementation)

## Decision Process

1. **Research Phase** (2-4 weeks per track): Implement and benchmark
2. **Evaluation**: Compare all tracks against criteria
3. **ADR**: Document findings and decision in new ADR
4. **Integration**: Merge winning approach to develop branch

### ADR Template for Decision

The final ADR should include:
- Comparison table of all tracks
- Rationale for selection
- Trade-offs accepted
- Integration plan
- Rollback strategy if issues discovered

## Related

- ADR: [0002-tiered-memory-architecture.md](../../docs/adr/0002-tiered-memory-architecture.md) - defines compression requirements
- Spec: [model-benchmarking/spec.md](../model-benchmarking/spec.md) - benchmark methodology
- Code: `src/cogsyndelta/memory/dense_embeddings.py` - current compression

---

*Template based on [GitHub Spec-Kit](https://github.com/github/spec-kit)*
