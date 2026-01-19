# Dataset Development Specification

> Specification for CogSynDelta training dataset development, acquisition, and management—with critical focus on compression fidelity recovery and adjustable compression ratio system.

## Status

| Attribute | Value |
|-----------|-------|
| **Spec ID** | dataset-development |
| **Status** | Draft |
| **Author** | @tzervas |
| **Created** | 2026-01-18 |
| **Updated** | 2026-01-19 |

## Problem Statement

CogSynDelta's brain-inspired architecture requires specialized datasets for each submodel (brain region) to achieve domain expertise. **CRITICAL**: Current compression fidelity is **0.06** (catastrophic failure) vs target **≥0.95**.

### Integration with embeddenator-core Ecosystem

This specification leverages the **embeddenator** family of component libraries:
- **embeddenator-core**: Core embedding compression primitives and interfaces
- **embeddenator-vsa**: Vector Symbolic Architecture operations (binding, bundling)
- **embeddenator-rvq**: Residual Vector Quantization with adaptive staging
- **embeddenator-calibration**: Adaptive calibration infrastructure
- **embeddenator-hopfield**: Modern Hopfield Networks for active memory

The CogSynDelta memory system builds on these composable primitives.

### Current State (CRITICAL)

| Issue | Current | Target | Status |
|-------|---------|--------|--------|
| Compression fidelity | **0.06** | ≥0.95 | 🔴 CRITICAL |
| Compression ratio | 2x | 10-20x | 🟡 Suboptimal |
| Dataset inventory | None | Systematic | 🔴 Missing |
| Cleaning pipeline | None | Automated | 🔴 Missing |
| Language specialization | None | 7 languages | 🔴 Missing |

### Root Cause Analysis (from exhaustive research)

The 0.06 cosine similarity indicates **catastrophic quantization failure**:
1. **Uncalibrated quantization buckets** - ranges don't match embedding distribution
2. **Insufficient bit-width** for differential signals
3. **Cumulative error propagation** without correction mechanisms
4. **Distribution mismatch** - assumes Gaussian when embeddings may be heavy-tailed

## Goals

1. **PRIORITY 1: Fix compression fidelity** from 0.06 → ≥0.95 (critical path)
2. **Implement adjustable compression ratio** system (4x-20x with fidelity guarantees)
3. **Catalog and acquire** high-quality datasets for each brain region
4. **Design automated pipelines** for cleaning, enrichment, and quality control
5. **Create specialization datasets** for each supported programming language
6. **Establish training curriculum** with phased data introduction
7. **Integrate VSA/holographic techniques** for brain-inspired memory

## Non-Goals

- Building a general-purpose dataset platform
- Supporting languages beyond the core 7 (Python, Rust, Go, TS, Java, C++, C#)
- Real-time data ingestion (batch processing is acceptable)
- Achieving >0.95 fidelity at 100x compression (research frontier, not achievable today)

## Success Criteria

| Metric | Target | Measurement | Priority |
|--------|--------|-------------|----------|
| **Compression fidelity** | **≥0.95** | Cosine similarity / Spearman ρ | 🔴 P0 |
| **Adjustable compression** | 4x-20x | Configurable ratio | 🔴 P0 |
| Dataset coverage | ≥ 10M examples/submodel | Inventory count | 🟡 P1 |
| Data quality score | ≥ 0.9 | Automated assessment | 🟡 P1 |
| Deduplication | ≤ 5% duplicates | Hash-based dedup | 🟢 P2 |
| License compliance | 100% | Audit script | 🟢 P2 |

### Realistic Fidelity Targets by Compression Ratio

Based on state-of-the-art research (QINCo2, Matryoshka, BitNet):

| Compression | Achievable Fidelity | Technique | Status |
|-------------|---------------------|-----------|--------|
| **4x** | 0.97-0.99 | Scalar quantization / Matryoshka | ✅ Achievable |
| **8x** | 0.95-0.98 | OPQ, Matryoshka MRL | ✅ Achievable |
| **16x** | 0.90-0.95 | QINCo2, RVQ, BitNet b1.58 | ✅ Achievable |
| **24x** | 0.85-0.92 | PQ with rescoring | ⚠️ Challenging |
| **32x** | 0.70-0.85 | IVF-PQ, Neural RQ | ⚠️ Research |
| **100x** | <0.70 | Not demonstrated | ❌ Not achievable |

## Technical Approach

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Dataset Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐        │
│  │ Acquisition│───▶│ Cleaning   │───▶│ Enrichment │        │
│  │            │    │            │    │            │        │
│  │ • Public   │    │ • Dedup    │    │ • Tagging  │        │
│  │ • Synthetic│    │ • Quality  │    │ • Metadata │        │
│  │ • Partner  │    │ • License  │    │ • Scoring  │        │
│  └────────────┘    └────────────┘    └────────────┘        │
│                                           │                 │
│                                           ▼                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │              Specialization                         │    │
│  │                                                     │    │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │    │
│  │  │ Py  │ │ Rs  │ │ Go  │ │ TS  │ │Java │ │C++  │ │    │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ │    │
│  └────────────────────────────────────────────────────┘    │
│                          │                                  │
│                          ▼                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │           Training Curriculum                       │    │
│  │                                                     │    │
│  │  Phase 1: Foundation → Phase 2: Specialization     │    │
│  │  Phase 3: Integration → Phase 4: Alignment         │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### CRITICAL: Compression Fidelity Recovery Architecture

```
┌─────────────────────────────────────────────────────────────┐
│            Adjustable Compression System                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Stage 1: CALIBRATION (0.06 → 0.85-0.90)             │  │
│  │  • Compute per-dimension min/max/mean/std on 10K+    │  │
│  │  • Store calibration ranges per embedding type       │  │
│  │  • Use calibrated ranges for quantization buckets    │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Stage 2: MATRYOSHKA MRL (flexible compression)      │  │
│  │  • Train with multi-scale loss at dims {64,128,256}  │  │
│  │  • Early dims = coarse semantics, later = fine       │  │
│  │  • Truncate to target compression ratio dynamically  │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Stage 3: RESIDUAL VECTOR QUANTIZATION (0.90→0.95+)  │  │
│  │  • 4-8 residual stages with 256-1024 codebook        │  │
│  │  • Each stage captures what previous missed          │  │
│  │  • Stop when target fidelity achieved                │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Stage 4: VSA/HOLOGRAPHIC (brain-inspired memory)    │  │
│  │  • Modern Hopfield Networks for active memory        │  │
│  │  • VSA binding for temporal context                  │  │
│  │  • torchhd for GPU-accelerated operations            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Adjustable Compression API Design

```python
class AdaptiveCompressionManager:
    """Adjustable compression with fidelity guarantees."""

    def __init__(
        self,
        target_fidelity: float = 0.95,
        max_compression: float = 20.0,
        min_compression: float = 4.0,
    ):
        self.target_fidelity = target_fidelity
        self.max_compression = max_compression
        self.min_compression = min_compression

    def compress(
        self,
        embeddings: torch.Tensor,
        target_ratio: float | None = None,
    ) -> tuple[CompressedEmbedding, CompressionMetrics]:
        """
        Compress with adjustable ratio while guaranteeing fidelity.

        If target_ratio would violate fidelity, automatically reduce
        compression to maintain ≥target_fidelity.

        Args:
            embeddings: Input embeddings to compress
            target_ratio: Desired compression (None = auto-select)

        Returns:
            (compressed_data, metrics including actual_fidelity)
        """
        ...

    def find_optimal_ratio(
        self,
        embeddings: torch.Tensor,
    ) -> float:
        """Find maximum compression that maintains target fidelity."""
        ...
```

### Data Sources

#### Primary Training Datasets for Compression (License Compatible)

| Dataset | Size | License | Use Case | Relevance |
|---------|------|---------|----------|-----------|
| **AllNLI** | 1M+ pairs | CC-BY-3.0/MIT | Contrastive training | 10/10 |
| **LAION-5B Embeddings** | 5.85B | CC-BY-4.0 | Codebook learning | 10/10 |
| **GIST-1M** (960d) | 1M vectors | CC0 | High-dim PQ training | 10/10 |
| **MS MARCO** | 8.8M passages | MIT | Retrieval training | 9/10 |
| **STS Benchmark** | 8.6K pairs | Permissive | Fidelity evaluation | 10/10 |
| **MTEB Suite** | 58 datasets | Apache-2.0 | Multi-task evaluation | 10/10 |

#### Public Datasets for Submodels
| Dataset | Domain | Size | License | Submodel |
|---------|--------|------|---------|----------|
| The Stack v2 | Code | 3TB | OSI-approved | Language Cortex |
| LAION-5B | Images | 5B pairs | CC-BY | Visual Cortex |
| RedPajama | Text | 1.2T tokens | Apache-2.0 | Prefrontal |
| CodeSearchNet | Code Search | 6M | MIT | Language Cortex |
| GitHub Archive | Commits | Petabytes | varies | All |

#### Synthetic Generation
- LLM-generated code examples with quality filtering
- Procedural problem generation
- Augmentation (variable renaming, formatting)

### Quality Pipeline

```python
class DataQualityPipeline:
    """Automated data quality assessment and filtering."""

    def __init__(self):
        self.deduplicator = MinHashDeduplicator()
        self.quality_scorer = CodeQualityScorer()
        self.license_checker = LicenseAuditor()
        self.enricher = MetadataEnricher()

    def process(self, dataset: Dataset) -> Dataset:
        # 1. Deduplication
        dataset = self.deduplicator.deduplicate(dataset)

        # 2. Quality filtering
        dataset = dataset.filter(
            lambda x: self.quality_scorer.score(x) >= 0.9
        )

        # 3. License validation
        dataset = dataset.filter(
            lambda x: self.license_checker.is_compatible(x)
        )

        # 4. Enrichment
        dataset = self.enricher.enrich(dataset)

        return dataset
```

### Calibration Pipeline (CRITICAL)

```python
def calibrate_quantization_ranges(embeddings: torch.Tensor, n_samples: int = 10000):
    """CRITICAL: Compute calibration statistics before any quantization.

    The 0.06 fidelity is caused by uncalibrated quantization.
    This function computes proper ranges per dimension.
    """
    sample = embeddings[:n_samples]

    return {
        'min': sample.min(dim=0)[0],
        'max': sample.max(dim=0)[0],
        'mean': sample.mean(dim=0),
        'std': sample.std(dim=0),
        'percentile_1': torch.quantile(sample, 0.01, dim=0),
        'percentile_99': torch.quantile(sample, 0.99, dim=0),
    }
```

## Dependencies

### Core Libraries
- DVC for dataset versioning
- Apache Arrow/Parquet for storage
- Ray for distributed processing
- HuggingFace Datasets for loading

### Compression & VSA Libraries (embeddenator-core ecosystem)
- **torchhd** - GPU-accelerated VSA operations (MIT License)
- **vector-quantize-pytorch** - RVQ, FSQ, LFQ (MIT License, 3.7k stars)
- **FAISS** - Similarity search with IVF-PQ
- **sentence-transformers** - Pre-trained Matryoshka models

### Key Pre-trained Models
- `tomaarsen/mpnet-base-nli-matryoshka` - Matryoshka embeddings
- OpenAI `text-embedding-3-large` - Native dimension parameter
- `nomic-embed-text-v1` - Matryoshka support

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Fidelity stays at 0.06** | CRITICAL | Calibration + staged approach |
| License contamination | High | Automated audit + legal review |
| Data quality drift | Medium | Continuous monitoring |
| Storage costs | Medium | Tiered storage + compression |
| Processing bottlenecks | Low | Distributed pipeline |

## Timeline

### Phase 1: Fidelity Emergency Recovery (Week 1-2) 🔴 CRITICAL

| Task | Risk | Expected Fidelity |
|------|------|-------------------|
| Implement calibration-based quantization | Low | 0.06 → 0.85-0.90 |
| Add fidelity monitoring with threshold alerts | Low | Continuous |
| Deploy int8 scalar quantization | Low | Baseline |
| Test with pre-trained Matryoshka model | Low | Validate approach |

### Phase 2: High-Fidelity Compression (Week 3-4)

| Task | Risk | Expected Fidelity |
|------|------|-------------------|
| Train MRL multi-scale loss on AllNLI | Medium | 0.90 → 0.93 |
| Implement RVQ (2-4 residual stages) | Medium | 0.93 → 0.95-0.97 |
| Add adaptive lossy/lossless decision | Medium | Guaranteed ≥0.95 |
| Integrate torchhd for VSA operations | Low | Brain-inspired |

### Phase 3: Dataset Pipeline (Week 5-8)

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Research | 2 weeks | Dataset catalog, pipeline design |
| Implementation | 2 weeks | Cleaning pipelines, enrichment |
| Specialization | 2 weeks | Per-language datasets |
| Integration | 2 weeks | Training curriculum |

### Phase 4: Brain-Inspired Memory (Week 9-12)

| Task | Risk | Description |
|------|------|-------------|
| Modern Hopfield layer | Medium | Exponential storage capacity |
| VSA temporal binding | Medium | Context-aware retrieval |
| FAISS + VSA hybrid | Low | Dual-path architecture |
| Consolidation algorithms | High | Sleep-like memory merging |

## References

- [RESEARCH_PROMPT.md](RESEARCH_PROMPT.md) - External research prompt
- [ADR-0002](../../docs/adr/0002-tiered-memory-architecture.md) - Tiered memory architecture
- [ARCHITECTURE.md](../../docs/ARCHITECTURE.md) - System architecture
- [Compression Fidelity Research](../../docs/CogSynDelta%20Compression%20Fidelity%20-%20Datasets%20and%20Techniques%20for%20Brain-Inspired%20Memory%20Systems.md) - Exhaustive dataset analysis
- [VSA and Holographic Computing](../../docs/Solving%20CogSynDelta%20Compression%20Fidelity%20Crisis%20with%20VSA%20and%20Holographic%20Computing.md) - Implementation techniques

### Key Research Papers
- Kusupati et al. "Matryoshka Representation Learning" (NeurIPS 2022)
- Ma et al. "BitNet b1.58" (arXiv:2402.17764) - 100% fidelity at 16x
- QINCo2 (Meta AI, ICLR 2025) - 34% MSE improvement
- Ramsauer et al. "Hopfield Networks is All You Need" (arXiv:2008.02217)

### Sister Projects (embeddenator-core ecosystem)
- **torchhd** - https://github.com/hyperdimensional-computing/torchhd
- **vector-quantize-pytorch** - https://github.com/lucidrains/vector-quantize-pytorch
- **sentence-transformers** - https://github.com/UKPLab/sentence-transformers

---

*Last Updated: January 18, 2026*
