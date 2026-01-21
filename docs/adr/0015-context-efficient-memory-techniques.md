# ADR-0015: Context-Efficient Memory Techniques for GPU VRAM Management

**Status**: Proposed
**Date**: 2026-01-19
**Decision Makers**: @tzervas
**Technical Story**: Optimize context handling without GPU OOM

## Context

CogSynDelta's memory architecture requires efficient handling of large context windows
and embedding stores. Current approaches can exhaust GPU VRAM when processing:
- Large temporal context windows (1000+ memory entries)
- High-dimensional embeddings (512-dim × batch size)
- Multi-scale compression states

Video generation AI and large language models have developed proven techniques for
handling massive contexts efficiently. We need to adapt these for CogSynDelta's
memory system.

### Current Pain Points

1. **LosslessCompactor OOM**: Requires 31GB for quantization tables on full batches
2. **Temporal context scaling**: Linear memory growth with context window
3. **Batch processing limits**: Large batches exhaust available VRAM
4. **No streaming inference**: Must load all data before processing

### Relevant Techniques from Video/LLM AI

| Technique | Source | Memory Reduction | Trade-off |
|-----------|--------|------------------|-----------|
| **Sliding Window Attention** | Longformer, Mistral | O(n) → O(w) | Limited global context |
| **Ring Attention** | Google (2023) | Linear with devices | Communication overhead |
| **Token Dropping/Merging** | EViT, ToMe | 30-50% | Some information loss |
| **Gradient Checkpointing** | PyTorch native | ~70% training | 20-30% slower |
| **Flash Attention** | Dao et al. | 5-20x | Kernel complexity |
| **Temporal Frame Dropping** | Video diffusion | N × reduction | Interpolation artifacts |
| **Latent Caching** | SD, SDXL | Avoid recomputation | Cache invalidation |
| **Progressive Loading** | LLM inference | Bounded peak | Latency variance |

## Decision

Implement a layered context-efficiency strategy:

### Layer 1: Chunk-Based Processing (Immediate)

Process embeddings in memory-bounded chunks with streaming reconstruction:

```python
class ChunkedCompactor(nn.Module):
    """Memory-efficient compactor using chunked processing.

    Why chunking:
        Video diffusion models process frames in temporal chunks to
        avoid loading entire videos into VRAM. We apply the same
        principle to embedding batches.

    Memory formula:
        peak_vram = chunk_size × embed_dim × dtype_bytes + overhead
        With 1000 samples, 512-dim, fp16: 1000 × 512 × 2 = 1MB per chunk
        vs 100000 × 512 × 2 = 100MB full batch
    """

    def __init__(
        self,
        base_compactor: nn.Module,
        chunk_size: int = 1000,
        overlap: int = 0,
    ):
        super().__init__()
        self.base = base_compactor
        self.chunk_size = chunk_size
        self.overlap = overlap

    @torch.no_grad()
    def compress(self, embeddings: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Compress in chunks, yielding results progressively."""
        results = []
        n = embeddings.size(0)

        for start in range(0, n, self.chunk_size - self.overlap):
            end = min(start + self.chunk_size, n)
            chunk = embeddings[start:end]

            # Process chunk
            compressed = self.base.compact(chunk)
            results.append(compressed)

            # Explicitly free intermediate tensors
            del chunk
            torch.cuda.empty_cache()

        return results
```

### Layer 2: Importance-Based Context Pruning (Short-term)

Adapt token merging from vision transformers for memory entries:

```python
class ImportanceContextPruner(nn.Module):
    """Prune low-importance context entries to fit VRAM budget.

    Inspired by ToMe (Token Merging) and EViT (Token Eviction):
    - Compute importance scores for each memory entry
    - Keep top-k most important, merge or drop rest
    - Maintains semantic coverage with reduced memory

    Why this works:
        Memory entries have varying relevance. Recent entries and
        semantically central entries carry most information. We can
        aggressively prune periphery without significant fidelity loss.
    """

    def __init__(
        self,
        max_context: int = 1000,
        importance_fn: str = "attention",  # or "recency", "centrality"
    ):
        super().__init__()
        self.max_context = max_context
        self.importance_fn = importance_fn

        # Lightweight importance scorer
        self.scorer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def compute_importance(
        self,
        embeddings: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score each embedding's importance."""
        # Base importance from content
        content_scores = self.scorer(embeddings).squeeze(-1)

        # Recency bonus (exponential decay)
        if timestamps is not None:
            max_time = timestamps.max()
            recency = torch.exp(-0.1 * (max_time - timestamps))
            content_scores = content_scores + 0.5 * recency

        # Centrality: similarity to mean embedding
        mean_emb = embeddings.mean(dim=0, keepdim=True)
        centrality = F.cosine_similarity(embeddings, mean_emb)
        content_scores = content_scores + 0.3 * centrality

        return content_scores

    def prune(
        self,
        embeddings: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prune to max_context entries, return (pruned_embeddings, indices)."""
        if embeddings.size(0) <= self.max_context:
            return embeddings, torch.arange(embeddings.size(0))

        scores = self.compute_importance(embeddings, timestamps)
        _, top_indices = scores.topk(self.max_context)

        return embeddings[top_indices], top_indices
```

### Layer 3: Latent Caching with Invalidation (Medium-term)

Cache compressed representations to avoid recomputation:

```python
class LatentCache:
    """LRU cache for compressed representations.

    Why caching:
        Stable Diffusion caches VAE latents to avoid re-encoding.
        We cache compressed memory states since compression is
        computationally expensive but deterministic.

    Invalidation strategy:
        - Hash embedding content for cache keys
        - Invalidate on model weight updates
        - LRU eviction when cache exceeds budget
    """

    def __init__(
        self,
        max_size_mb: float = 100.0,
        device: str = "cpu",  # Cache on CPU to save GPU VRAM
    ):
        self.max_size_mb = max_size_mb
        self.device = device
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.current_size_mb = 0.0
        self._version = 0  # Increment on model changes

    def _hash_tensor(self, tensor: torch.Tensor) -> str:
        """Compute cache key from tensor content."""
        return hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()

    def get(self, embeddings: torch.Tensor) -> torch.Tensor | None:
        """Retrieve cached compression if available."""
        key = f"v{self._version}_{self._hash_tensor(embeddings)}"
        if key in self.cache:
            self.cache.move_to_end(key)  # LRU update
            return self.cache[key].to(embeddings.device)
        return None

    def put(self, embeddings: torch.Tensor, compressed: torch.Tensor) -> None:
        """Cache compressed representation."""
        key = f"v{self._version}_{self._hash_tensor(embeddings)}"

        # Move to cache device
        cached = compressed.to(self.device)
        size_mb = cached.numel() * cached.element_size() / (1024 * 1024)

        # Evict if needed
        while self.current_size_mb + size_mb > self.max_size_mb and self.cache:
            _, old = self.cache.popitem(last=False)
            self.current_size_mb -= old.numel() * old.element_size() / (1024 * 1024)

        self.cache[key] = cached
        self.current_size_mb += size_mb

    def invalidate(self) -> None:
        """Invalidate all cached entries (call on model update)."""
        self._version += 1
        self.cache.clear()
        self.current_size_mb = 0.0
```

### Layer 4: Gradient Checkpointing for Training (Training-specific)

```python
def enable_memory_efficient_training(model: nn.Module) -> None:
    """Enable gradient checkpointing for memory-efficient training.

    Per PyTorch docs, this trades compute for memory by
    recomputing activations during backward pass instead of storing them.

    Typical reduction: 60-70% activation memory
    Typical slowdown: 20-30%
    """
    from torch.utils.checkpoint import checkpoint_sequential

    # Enable for transformer layers
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        model.encoder.layers = checkpoint_sequential(
            model.encoder.layers,
            segments=4,  # Checkpoint every 4 layers
        )
```

## Rationale

### Why This Approach

1. **Layered strategy**: Each layer addresses different scenarios
   - Chunking: Immediate fix for OOM
   - Pruning: Reduces context intelligently
   - Caching: Amortizes cost over time
   - Checkpointing: Training-specific optimization

2. **Proven in production**: All techniques borrowed from deployed systems
   - Chunking: Used in all video diffusion models
   - ToMe: Deployed in production image generation
   - Latent caching: Core to Stable Diffusion pipeline
   - Checkpointing: Default in large model training

3. **Graceful degradation**: Each layer independent, can be disabled

### Alternatives Considered

#### Option 1: Quantization Only (INT8/FP8)

- **Pros**: Simple, hardware-accelerated
- **Cons**: Doesn't address context scaling, potential fidelity loss
- **Why Rejected**: Addresses symptoms not cause

#### Option 2: Offloading to CPU

- **Pros**: Virtually unlimited capacity
- **Cons**: PCIe bandwidth bottleneck, high latency
- **Why Rejected**: Too slow for interactive use

#### Option 3: Model Parallelism

- **Pros**: Scales with devices
- **Cons**: Requires multiple GPUs, communication overhead
- **Why Rejected**: Not available on single-GPU systems

## Consequences

### Positive

- Enables processing larger contexts on 16GB GPUs
- Reduces OOM errors in compression benchmarks
- Improves batch processing throughput
- Prepares system for training workloads

### Negative

- Chunked processing has coordination overhead
- Caching requires CPU memory budget
- Pruning may lose edge-case information (mitigated by importance scoring)

### Neutral

- API changes required for streaming results
- Need to tune chunk sizes per GPU memory tier

## Implementation

### Phase 1 (Immediate - 1 week)

1. Implement `ChunkedCompactor` wrapper
2. Add VRAM budget parameter to compression benchmark
3. Update `LosslessCompactor` to use chunked processing

### Phase 2 (Short-term - 2 weeks)

1. Implement `ImportanceContextPruner`
2. Integrate with `ActiveMemory` tier
3. Add pruning statistics to benchmarks

### Phase 3 (Medium-term - 1 month)

1. Implement `LatentCache` with invalidation
2. Add cache hit rate metrics
3. Optimize cache placement strategy

## References

- [Longformer: Long-Document Transformer](https://arxiv.org/abs/2004.05150) - Sliding window attention
- [Ring Attention](https://arxiv.org/abs/2310.01889) - Blockwise parallel transformers
- [Token Merging (ToMe)](https://arxiv.org/abs/2210.09461) - Vision transformer efficiency
- [EViT: Expediting Vision Transformers](https://arxiv.org/abs/2202.00732) - Token pruning
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Memory-efficient attention
- [Gradient Checkpointing](https://pytorch.org/docs/stable/checkpoint.html) - PyTorch docs
- ADR-0008: Compression Fidelity Recovery
- ADR-0014: Matryoshka + QINCo2 Compression

---

*Follows constitution requirement: "The system must handle failures gracefully"*
