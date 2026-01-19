# Dataset Development Specification

> Specification for CogSynDelta training dataset development, acquisition, and management.

## Status

| Attribute | Value |
|-----------|-------|
| **Spec ID** | dataset-development |
| **Status** | Draft |
| **Author** | @tzervas |
| **Created** | 2026-01-18 |
| **Updated** | 2026-01-18 |

## Problem Statement

CogSynDelta's brain-inspired architecture requires specialized datasets for each submodel (brain region) to achieve domain expertise. Current state:

- No systematic dataset inventory
- No automated cleaning/enrichment pipeline
- No per-language specialization datasets
- Memory compression falls short of target (0.67 @ 2x vs >0.95 @ 10x)

## Goals

1. **Catalog and acquire** high-quality datasets for each brain region
2. **Design automated pipelines** for cleaning, enrichment, and quality control
3. **Create specialization datasets** for each supported programming language
4. **Establish training curriculum** with phased data introduction
5. **Achieve compression fidelity** target through better training data

## Non-Goals

- Building a general-purpose dataset platform
- Supporting languages beyond the core 7 (Python, Rust, Go, TS, Java, C++, C#)
- Real-time data ingestion (batch processing is acceptable)

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Dataset coverage | ≥ 10M examples/submodel | Inventory count |
| Data quality score | ≥ 0.9 | Automated assessment |
| Deduplication | ≤ 5% duplicates | Hash-based dedup |
| License compliance | 100% | Audit script |
| Compression fidelity | ≥ 0.95 @ 10x | Benchmark suite |

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

### Data Sources

#### Public Datasets
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

## Dependencies

- DVC for dataset versioning
- Apache Arrow/Parquet for storage
- Ray for distributed processing
- HuggingFace Datasets for loading

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| License contamination | High | Automated audit + legal review |
| Data quality drift | Medium | Continuous monitoring |
| Storage costs | Medium | Tiered storage + compression |
| Processing bottlenecks | Low | Distributed pipeline |

## Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Research | 2 weeks | Dataset catalog, pipeline design |
| Implementation | 4 weeks | Cleaning pipelines, enrichment |
| Specialization | 4 weeks | Per-language datasets |
| Integration | 2 weeks | Training curriculum |

## References

- [RESEARCH_PROMPT.md](RESEARCH_PROMPT.md) - External research prompt
- [ADR-0002](../../docs/adr/0002-memory-compression-research.md) - Compression research
- [ARCHITECTURE.md](../../docs/ARCHITECTURE.md) - System architecture

---

*Last Updated: January 18, 2026*
