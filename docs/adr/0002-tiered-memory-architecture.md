# ADR-0002: Tiered Memory Architecture

**Status**: Accepted
**Date**: 2026-01-18
**Decision Makers**: @tzervas
**Technical Story**: Memory system design for self-improving AI

## Context

CogSynDelta is a self-improving AI system that needs to maintain both short-term context and long-term knowledge. Key challenges:

1. **Memory capacity**: Active working memory must be bounded for performance
2. **Persistence**: Knowledge must survive restarts and be retrievable
3. **Efficiency**: Can't keep everything in GPU memory
4. **Retrieval**: Need fast semantic similarity search across large knowledge bases

The system processes continuous streams of information and must decide what to remember, what to forget, and how to organize memories for efficient retrieval.

## Decision

We will implement a **three-tier memory architecture**:

```
┌─────────────────┐
│  Active Memory  │  Hot data, bounded capacity, immediate access
│    (GPU/RAM)    │  ~1000 entries max
└────────┬────────┘
         │ age out / access patterns
         ▼
┌─────────────────┐
│ Short-term Mem  │  Recent memories, fast retrieval
│     (RAM)       │  LRU eviction, ~10000 entries
└────────┬────────┘
         │ compress / archive
         ▼
┌─────────────────┐
│ Long-term Mem   │  Compressed storage, persistent
│  (Disk/FAISS)   │  SVD-based lossless compression
└─────────────────┘
```

## Rationale

### Why Three Tiers

1. **Cognitive analogy**: Mirrors human memory (working → short-term → long-term)
2. **Resource optimization**: Hot data stays fast, cold data stays compressed
3. **Graceful degradation**: System continues even if long-term storage is slow

### Compression Strategy

- **Lossless compaction** using SVD-based basis vectors
- Dense differential embeddings for efficient delta storage
- Target: 10-100x compression with high reconstruction fidelity

> **⚠️ KNOWN LIMITATION (2026-01-18)**: Current benchmarks show actual fidelity
> below target thresholds. Measured on RTX 5080:
> - 2x compression: 0.670 cosine similarity (target: >0.95)
> - 4x compression: 0.462 cosine similarity
> - 16x compression: 0.228 cosine similarity
>
> This gap between architecture potential and current implementation is being
> addressed in `feat/compression-*` research branches. See
> [specs/compression-research/spec.md](../../specs/compression-research/spec.md)
> for improvement plan.
>
> Per constitution: "All performance claims must be backed by evidence."
> We report actual measured values, not theoretical capabilities.

### Alternatives Considered

#### Option 1: Single unified memory

- **Pros**: Simpler implementation
- **Cons**: Either too slow (all on disk) or too expensive (all in GPU)
- **Why Rejected**: Doesn't scale

#### Option 2: Two tiers (hot/cold)

- **Pros**: Simpler than three tiers
- **Cons**: No intermediate caching, frequent disk access
- **Why Rejected**: Performance cliff between tiers too steep

#### Option 3: External vector database (Pinecone, Weaviate)

- **Pros**: Managed service, scales automatically
- **Cons**: Network latency, cost, dependency on external service
- **Why Rejected**: Self-contained system requirement, latency concerns

## Consequences

### Positive

- Bounded active memory prevents OOM
- Fast retrieval for recently accessed memories
- Long-term storage is space-efficient
- Clear eviction policies based on access patterns

### Negative

- Complexity: Three tiers to manage and test
- Mitigation: Clear interfaces between tiers, comprehensive tests
- Temporal continuity requires careful handling (see ADR-0004)

### Neutral

- FAISS used for similarity search (CPU version for broad compatibility)
- LlamaIndex for semantic retrieval pipeline

## Implementation

Key files:
- `src/cogsyndelta/memory/active_memory.py` - Tiered memory manager
- `src/cogsyndelta/memory/dense_embeddings.py` - Compression logic
- `src/cogsyndelta/memory/memory_persistence.py` - Disk persistence

Commits:
- `f72dd62`: "Add memory persistence, dense differential embeddings, and safeguards"
- `58305c0`: "Complete implementation: tools, auto-management, active memory with lossless compaction"

## References

- Dense Differential Embeddings paper concepts
- FAISS documentation for similarity search
- `docs/ARCHITECTURE.md` for full system design
