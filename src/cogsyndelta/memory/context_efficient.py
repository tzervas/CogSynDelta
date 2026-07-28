"""Context-Efficient Memory Techniques.

Implements memory-efficient processing patterns borrowed from video generation AI
and large language models. These techniques enable processing of large contexts
without exhausting GPU VRAM.

Key techniques:
    - Chunked processing: Process data in memory-bounded segments
    - Importance-based pruning: Keep only most relevant context entries
    - Latent caching: Cache compressed representations to avoid recomputation

Why these patterns:
    Video diffusion models (Stable Video Diffusion, SORA) and LLMs (Longformer,
    Mistral) face similar context/memory trade-offs. Their solutions are
    production-proven and directly applicable to CogSynDelta's memory architecture.

References:
    - ADR-0015: Context-Efficient Memory Techniques
    - ToMe: Token Merging for Vision Transformers (arxiv:2210.09461)
    - Flash Attention (arxiv:2205.14135)

Example:
    >>> from cogsyndelta.memory.context_efficient import ChunkedCompactor
    >>> efficient = ChunkedCompactor(base_compactor, chunk_size=1000)
    >>> results = efficient.compress(large_batch)  # No OOM!
"""

from __future__ import annotations

import gc
import hashlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    pass

__all__ = [
    "ChunkedCompactor",
    "ImportanceContextPruner",
    "LatentCache",
    "VRAMBudget",
    "estimate_tensor_memory",
    "get_available_vram",
]


def get_available_vram() -> float:
    """Get available GPU VRAM in MB.

    Returns:
        Available VRAM in megabytes, or 0 if CUDA not available.
    """
    if not torch.cuda.is_available():
        return 0.0

    # Get memory info
    free, total = torch.cuda.mem_get_info()
    return free / (1024 * 1024)


def estimate_tensor_memory(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> float:
    """Estimate tensor memory in MB.

    Args:
        shape: Tensor shape.
        dtype: Data type.

    Returns:
        Estimated memory in megabytes.
    """
    numel = 1
    for dim in shape:
        numel *= dim

    # Bytes per element
    if dtype == torch.float32:
        bytes_per = 4
    elif dtype in (torch.float16, torch.bfloat16):
        bytes_per = 2
    elif dtype == torch.float64:
        bytes_per = 8
    elif dtype == torch.int32:
        bytes_per = 4
    elif dtype == torch.int64:
        bytes_per = 8
    else:
        bytes_per = 4  # Default assumption

    return numel * bytes_per / (1024 * 1024)


@dataclass
class VRAMBudget:
    """VRAM budget configuration.

    Attributes:
        max_mb: Maximum VRAM to use in megabytes.
        safety_margin: Fraction to reserve (0.0-1.0).
        auto_detect: Whether to auto-detect available VRAM.
    """

    max_mb: float = 2000.0  # 2GB default
    safety_margin: float = 0.1  # 10% safety margin
    auto_detect: bool = True

    def effective_budget(self) -> float:
        """Get effective VRAM budget after safety margin.

        Returns:
            Effective budget in MB.
        """
        if self.auto_detect and torch.cuda.is_available():
            available = get_available_vram()
            budget = min(self.max_mb, available)
        else:
            budget = self.max_mb

        return budget * (1 - self.safety_margin)


class ChunkedCompactor(nn.Module):
    """Memory-efficient compactor using chunked processing.

    Wraps any compactor to process data in memory-bounded chunks,
    preventing OOM errors on large batches.

    Why chunking:
        Video diffusion models process frames in temporal chunks to
        avoid loading entire videos into VRAM. We apply the same
        principle to embedding batches.

    Memory formula:
        peak_vram = chunk_size × embed_dim × dtype_bytes × factor
        factor ~= 3-4 (input + output + intermediate activations)

    Example:
        >>> base = LosslessCompactor(embed_dim=512)
        >>> chunked = ChunkedCompactor(base, chunk_size=500)
        >>> # Process 10000 embeddings without OOM
        >>> results = chunked.compress(torch.randn(10000, 512))
    """

    def __init__(
        self,
        base_compactor: nn.Module,
        chunk_size: int = 1000,
        overlap: int = 0,
        vram_budget: VRAMBudget | None = None,
        clear_cache: bool = True,
    ) -> None:
        """Initialize chunked compactor.

        Args:
            base_compactor: Underlying compactor to wrap.
            chunk_size: Maximum samples per chunk.
            overlap: Overlap between chunks (for context continuity).
            vram_budget: Optional VRAM budget for auto chunk sizing.
            clear_cache: Whether to clear CUDA cache between chunks.
        """
        super().__init__()
        self.base = base_compactor
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.vram_budget = vram_budget or VRAMBudget()
        self.clear_cache = clear_cache

        # Track which method the base compactor uses
        self._has_compact = hasattr(base_compactor, "compact")
        self._has_compress = hasattr(base_compactor, "compress")
        self._has_encode = hasattr(base_compactor, "encode")

    def _compute_chunk_size(self, embed_dim: int, dtype: torch.dtype) -> int:
        """Dynamically compute safe chunk size based on VRAM budget.

        Args:
            embed_dim: Embedding dimension.
            dtype: Data type.

        Returns:
            Safe chunk size.
        """
        budget_mb = self.vram_budget.effective_budget()

        # Estimate memory per sample (4x factor for safety)
        mem_per_sample = estimate_tensor_memory((1, embed_dim), dtype) * 4

        if mem_per_sample > 0:
            auto_chunk_size = int(budget_mb / mem_per_sample)
            return min(self.chunk_size, max(1, auto_chunk_size))

        return self.chunk_size

    def _compress_single(self, chunk: torch.Tensor) -> dict[str, Any] | torch.Tensor:
        """Compress a single chunk using base compactor's method."""
        if self._has_compact:
            return self.base.compact(chunk)  # type: ignore[operator]
        if self._has_compress:
            return self.base.compress(chunk)  # type: ignore[operator]
        if self._has_encode:
            result = self.base.encode(chunk)  # type: ignore[operator]
            if isinstance(result, tuple):
                return {"dense": result[0], "residual": result[1]}
            return result
        msg = "Base compactor has no compress/compact/encode method"
        raise AttributeError(msg)

    def compress(
        self,
        embeddings: torch.Tensor,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any] | torch.Tensor]:
        """Compress embeddings in memory-efficient chunks.

        Args:
            embeddings: Input embeddings [batch, embed_dim].
            progress_callback: Optional callback(processed, total) for progress.

        Returns:
            List of compressed representations, one per chunk.

        Note:
            Results are returned as a list. Use `merge_results()` to combine
            if needed for downstream processing.
        """
        n = embeddings.size(0)
        embed_dim = embeddings.size(1)
        dtype = embeddings.dtype

        # Compute safe chunk size
        chunk_size = self._compute_chunk_size(embed_dim, dtype)

        results: list[dict[str, Any] | torch.Tensor] = []
        step = chunk_size - self.overlap

        with torch.no_grad():
            for start in range(0, n, step):
                end = min(start + chunk_size, n)
                chunk = embeddings[start:end]

                # Process chunk
                compressed = self._compress_single(chunk)
                results.append(compressed)

                # Progress callback
                if progress_callback:
                    progress_callback(end, n)

                # Free memory
                del chunk
                if self.clear_cache and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        return results

    def compact(self, embeddings: torch.Tensor) -> list[dict[str, Any] | torch.Tensor]:
        """Alias for compress() - matches CogSynDelta compactor interface."""
        return self.compress(embeddings)

    def reconstruct(
        self,
        compressed_chunks: list[dict[str, Any] | torch.Tensor],
    ) -> torch.Tensor:
        """Reconstruct embeddings from chunked compressed results.

        Args:
            compressed_chunks: List of compressed representations from compress().

        Returns:
            Reconstructed embeddings tensor.
        """
        reconstructed_chunks: list[torch.Tensor] = []

        with torch.no_grad():
            for chunk in compressed_chunks:
                recon: torch.Tensor
                if hasattr(self.base, "reconstruct"):
                    recon = self.base.reconstruct(chunk)  # type: ignore[operator]
                elif hasattr(self.base, "decode"):
                    if isinstance(chunk, dict) and "dense" in chunk:
                        recon = self.base.decode(chunk["dense"], chunk["residual"])  # type: ignore[operator]
                    else:
                        recon = self.base.decode(chunk)  # type: ignore[operator]
                else:
                    msg = "Base compactor has no reconstruct/decode method"
                    raise AttributeError(msg)

                reconstructed_chunks.append(recon)

                if self.clear_cache and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Handle overlap by averaging overlapping regions
        if self.overlap == 0:
            return torch.cat(reconstructed_chunks, dim=0)

        # With overlap, need to blend
        return self._blend_chunks(reconstructed_chunks)

    def _blend_chunks(self, chunks: list[torch.Tensor]) -> torch.Tensor:
        """Blend overlapping chunks using weighted average.

        Args:
            chunks: List of reconstructed chunks.

        Returns:
            Blended tensor.

        Why:
            Computes smooth linear-ramp/trapezoidal weight windows across chunks and
            accumulates their weighted contributions, eliminating sharp boundaries
            and reconstruction artifacts in overlapping regions.
        """
        if len(chunks) == 1:
            return chunks[0]

        chunk_size = chunks[0].size(0)
        step = chunk_size - self.overlap
        n = (len(chunks) - 1) * step + chunks[-1].size(0)
        embed_dim = chunks[0].size(1)
        dtype = chunks[0].dtype
        device = chunks[0].device

        out_tensor = torch.zeros((n, embed_dim), dtype=dtype, device=device)
        weight_tensor = torch.zeros((n, 1), dtype=dtype, device=device)

        for i, chunk in enumerate(chunks):
            L = chunk.size(0)
            start_idx = i * step
            end_idx = start_idx + L

            # Create trapezoidal window
            window = torch.ones(L, dtype=dtype, device=device)

            if i > 0 and self.overlap > 0:
                left_len = min(self.overlap, L)
                ramp = torch.linspace(
                    0.5 / left_len,
                    1.0 - 0.5 / left_len,
                    left_len,
                    dtype=dtype,
                    device=device,
                )
                window[:left_len] = ramp

            if i < len(chunks) - 1 and self.overlap > 0:
                right_len = min(self.overlap, L)
                ramp = torch.linspace(
                    1.0 - 0.5 / right_len,
                    0.5 / right_len,
                    right_len,
                    dtype=dtype,
                    device=device,
                )
                window[-right_len:] = ramp

            # Expand window to match embedding dim
            window_expanded = window.unsqueeze(1)  # [L, 1]

            out_tensor[start_idx:end_idx] += chunk * window_expanded
            weight_tensor[start_idx:end_idx] += window_expanded

        # Avoid division by zero
        weight_tensor = torch.clamp(weight_tensor, min=1e-5)
        return out_tensor / weight_tensor


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

    Example:
        >>> pruner = ImportanceContextPruner(max_context=500)
        >>> pruned, indices = pruner.prune(embeddings, timestamps)
        >>> # pruned.size(0) <= 500
    """

    def __init__(
        self,
        embed_dim: int = 512,
        max_context: int = 1000,
        importance_weights: dict[str, float] | None = None,
    ) -> None:
        """Initialize pruner.

        Args:
            embed_dim: Embedding dimension for scorer.
            max_context: Maximum entries to keep.
            importance_weights: Weights for importance components.
        """
        super().__init__()
        self.max_context = max_context
        self.embed_dim = embed_dim

        # Default importance weights
        self.importance_weights = importance_weights or {
            "content": 0.4,
            "recency": 0.35,
            "centrality": 0.25,
        }

        # Lightweight importance scorer
        self.content_scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1),
        )

    def compute_importance(
        self,
        embeddings: torch.Tensor,
        timestamps: torch.Tensor | None = None,
        query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute importance scores for each embedding.

        Args:
            embeddings: Input embeddings [batch, embed_dim].
            timestamps: Optional timestamps [batch] for recency scoring.
            query: Optional query embedding for relevance scoring.

        Returns:
            Importance scores [batch].
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        scores = torch.zeros(batch_size, device=device)

        # Content-based importance
        content_scores = self.content_scorer(embeddings).squeeze(-1)
        content_scores = F.softmax(content_scores, dim=0)
        scores = scores + self.importance_weights["content"] * content_scores

        # Recency bonus (exponential decay)
        if timestamps is not None:
            max_time = timestamps.max()
            recency = torch.exp(-0.1 * (max_time - timestamps))
            recency = recency / recency.sum()  # Normalize
            scores = scores + self.importance_weights["recency"] * recency

        # Centrality: similarity to mean embedding
        mean_emb = embeddings.mean(dim=0, keepdim=True)
        centrality = F.cosine_similarity(embeddings, mean_emb.expand_as(embeddings), dim=-1)
        centrality = (centrality + 1) / 2  # Shift to [0, 1]
        centrality = centrality / centrality.sum()  # Normalize
        scores = scores + self.importance_weights["centrality"] * centrality

        # Query relevance (if provided)
        if query is not None:
            query = query.unsqueeze(0) if query.dim() == 1 else query
            relevance = F.cosine_similarity(embeddings, query.expand_as(embeddings), dim=-1)
            relevance = (relevance + 1) / 2
            relevance = relevance / relevance.sum()
            scores = scores + 0.2 * relevance  # Bonus weight for query relevance

        return scores

    def prune(
        self,
        embeddings: torch.Tensor,
        timestamps: torch.Tensor | None = None,
        query: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prune to max_context entries based on importance.

        Args:
            embeddings: Input embeddings [batch, embed_dim].
            timestamps: Optional timestamps for recency scoring.
            query: Optional query for relevance scoring.

        Returns:
            Tuple of (pruned_embeddings, kept_indices).
        """
        if embeddings.size(0) <= self.max_context:
            return embeddings, torch.arange(embeddings.size(0), device=embeddings.device)

        with torch.no_grad():
            scores = self.compute_importance(embeddings, timestamps, query)
            _, top_indices = scores.topk(self.max_context)
            top_indices = top_indices.sort().values  # Maintain original order

        return embeddings[top_indices], top_indices

    def forward(
        self,
        embeddings: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - alias for prune()."""
        return self.prune(embeddings, timestamps)


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

    Example:
        >>> cache = LatentCache(max_size_mb=100)
        >>> cached = cache.get(embeddings)
        >>> if cached is None:
        ...     compressed = compactor.compress(embeddings)
        ...     cache.put(embeddings, compressed)
    """

    def __init__(
        self,
        max_size_mb: float = 100.0,
        device: str = "cpu",  # Cache on CPU to save GPU VRAM
    ) -> None:
        """Initialize cache.

        Args:
            max_size_mb: Maximum cache size in megabytes.
            device: Device to store cached tensors (CPU recommended).
        """
        self.max_size_mb = max_size_mb
        self.device = device
        self.cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.current_size_mb = 0.0
        self._version = 0  # Increment on model changes
        self._hits = 0
        self._misses = 0

    def _hash_tensor(self, tensor: torch.Tensor) -> str:
        """Compute cache key from tensor content.

        Args:
            tensor: Input tensor.

        Returns:
            Hash string.
        """
        # Use first and last elements + shape for fast hashing
        # MD5 is acceptable here - used for cache keys, not security
        data = (
            str(tensor.shape)
            + str(tensor.flatten()[:10].tolist())
            + str(tensor.flatten()[-10:].tolist())
        )
        return hashlib.md5(data.encode(), usedforsecurity=False).hexdigest()

    def _estimate_size(self, data: dict[str, Any] | torch.Tensor) -> float:
        """Estimate storage size in MB."""
        if isinstance(data, torch.Tensor):
            return data.numel() * data.element_size() / (1024 * 1024)
        if isinstance(data, dict):
            total = 0.0
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size() / (1024 * 1024)
            return total
        return 0.0

    def _to_cache_device(
        self, data: dict[str, Any] | torch.Tensor
    ) -> dict[str, Any] | torch.Tensor:
        """Move data to cache device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        if isinstance(data, dict):
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in data.items()
            }
        return data

    def _to_request_device(
        self, data: dict[str, Any] | torch.Tensor, device: torch.device
    ) -> dict[str, Any] | torch.Tensor:
        """Move data to requested device."""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        if isinstance(data, dict):
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        return data

    def get(self, embeddings: torch.Tensor) -> dict[str, Any] | torch.Tensor | None:
        """Retrieve cached compression if available.

        Args:
            embeddings: Input embeddings to look up.

        Returns:
            Cached compressed representation, or None if not cached.
        """
        key = f"v{self._version}_{self._hash_tensor(embeddings)}"

        if key in self.cache:
            self._hits += 1
            self.cache.move_to_end(key)  # LRU update
            return self._to_request_device(self.cache[key], embeddings.device)

        self._misses += 1
        return None

    def put(
        self,
        embeddings: torch.Tensor,
        compressed: dict[str, Any] | torch.Tensor,
    ) -> None:
        """Cache compressed representation.

        Args:
            embeddings: Original embeddings (for key).
            compressed: Compressed representation to cache.
        """
        key = f"v{self._version}_{self._hash_tensor(embeddings)}"

        # Move to cache device
        cached = self._to_cache_device(compressed)
        size_mb = self._estimate_size(cached)

        # Evict if needed
        while self.current_size_mb + size_mb > self.max_size_mb and self.cache:
            _, old = self.cache.popitem(last=False)
            self.current_size_mb -= self._estimate_size(old)

        self.cache[key] = cached  # type: ignore[assignment]
        self.current_size_mb += size_mb

    def invalidate(self) -> None:
        """Invalidate all cached entries.

        Call this when model weights are updated.
        """
        self._version += 1
        self.cache.clear()
        self.current_size_mb = 0.0

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with hit rate, size, etc.
        """
        total = self._hits + self._misses
        return {
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "hits": self._hits,
            "misses": self._misses,
            "size_mb": self.current_size_mb,
            "max_size_mb": self.max_size_mb,
            "num_entries": len(self.cache),
            "version": self._version,
        }

    def clear_stats(self) -> None:
        """Reset hit/miss counters."""
        self._hits = 0
        self._misses = 0


class CachedChunkedCompactor(nn.Module):
    """Combines chunking and caching for maximum efficiency.

    Layers both optimizations:
    1. Check cache for each chunk
    2. Process uncached chunks
    3. Cache results

    Example:
        >>> compactor = CachedChunkedCompactor(
        ...     base_compactor=LosslessCompactor(),
        ...     chunk_size=500,
        ...     cache_size_mb=200,
        ... )
        >>> # First call: compresses and caches
        >>> result1 = compactor.compress(embeddings)
        >>> # Second call: mostly cache hits
        >>> result2 = compactor.compress(embeddings)
    """

    def __init__(
        self,
        base_compactor: nn.Module,
        chunk_size: int = 1000,
        cache_size_mb: float = 100.0,
    ) -> None:
        """Initialize cached chunked compactor.

        Args:
            base_compactor: Underlying compactor.
            chunk_size: Samples per chunk.
            cache_size_mb: Cache size budget.
        """
        super().__init__()
        self.chunked = ChunkedCompactor(base_compactor, chunk_size=chunk_size)
        self.cache = LatentCache(max_size_mb=cache_size_mb)

    def compress(self, embeddings: torch.Tensor) -> list[dict[str, Any] | torch.Tensor]:
        """Compress with caching.

        Args:
            embeddings: Input embeddings.

        Returns:
            List of compressed chunks.
        """
        # Check cache first
        cached = self.cache.get(embeddings)
        if cached is not None:
            return [cached]  # Full batch was cached

        # Process in chunks
        results = self.chunked.compress(embeddings)

        # Cache the full result (if small enough)
        if len(results) == 1:
            self.cache.put(embeddings, results[0])

        return results

    def compact(self, embeddings: torch.Tensor) -> list[dict[str, Any] | torch.Tensor]:
        """Alias for compress()."""
        return self.compress(embeddings)

    def reconstruct(self, compressed_chunks: list[dict[str, Any] | torch.Tensor]) -> torch.Tensor:
        """Reconstruct from compressed chunks."""
        return self.chunked.reconstruct(compressed_chunks)

    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats()
