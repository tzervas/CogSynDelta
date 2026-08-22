"""Unit tests for context-efficient memory components.

Tests ChunkedCompactor, ImportanceContextPruner, LatentCache, and CachedChunkedCompactor.
"""

from __future__ import annotations

import torch
from torch import nn

from cogsyndelta.memory.context_efficient import (
    CachedChunkedCompactor,
    ChunkedCompactor,
    ImportanceContextPruner,
    LatentCache,
    VRAMBudget,
    estimate_tensor_memory,
    get_available_vram,
)


class DummyCompactor(nn.Module):
    """Simple identity compactor for testing ChunkedCompactor."""

    def __init__(self, embed_dim: int = 64) -> None:
        """Initialize dummy compactor.

        Args:
            embed_dim: Embedding dimension.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress tensor by projecting.

        Args:
            x: Input tensor.

        Returns:
            Projected tensor.
        """
        return self.proj(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct tensor.

        Args:
            x: Compressed tensor.

        Returns:
            Reconstructed tensor.
        """
        return x


def test_vram_and_estimation_helpers() -> None:
    """Test get_available_vram, estimate_tensor_memory, and VRAMBudget."""
    available = get_available_vram()
    assert isinstance(available, float)
    assert available >= 0.0

    mem_fp32 = estimate_tensor_memory((100, 64), dtype=torch.float32)
    assert mem_fp32 > 0.0

    mem_fp16 = estimate_tensor_memory((100, 64), dtype=torch.float16)
    assert abs(mem_fp32 - 2 * mem_fp16) < 1e-4

    budget = VRAMBudget(max_mb=1000.0, safety_margin=0.1, auto_detect=False)
    assert budget.effective_budget() == 900.0


def test_chunked_compactor_non_overlapping() -> None:
    """Test ChunkedCompactor without overlap."""
    base = DummyCompactor(embed_dim=32)
    chunked = ChunkedCompactor(base, chunk_size=10, overlap=0, clear_cache=False)

    data = torch.randn(25, 32)
    compressed = chunked.compress(data)
    assert len(compressed) == 3
    assert compressed[0].shape == (10, 32)
    assert compressed[1].shape == (10, 32)
    assert compressed[2].shape == (5, 32)

    reconstructed = chunked.reconstruct(compressed)
    assert reconstructed.shape == (25, 32)


def test_chunked_compactor_overlapping_trapezoidal_blending() -> None:
    """Test ChunkedCompactor with overlap and trapezoidal linear-ramp blending."""
    base = DummyCompactor(embed_dim=16)
    # Identity projection for exact reconstruction validation
    with torch.no_grad():
        base.proj.weight.copy_(torch.eye(16))
        base.proj.bias.zero_()

    chunked = ChunkedCompactor(base, chunk_size=10, overlap=2, clear_cache=False)

    data = torch.randn(20, 16)
    compressed = chunked.compress(data)
    # chunk_size 10, step 8 -> chunks at 0..10, 8..18, 16..20 (3 chunks)
    assert len(compressed) == 3

    reconstructed = chunked.reconstruct(compressed)
    assert reconstructed.shape == (20, 16)
    assert torch.allclose(reconstructed, data, atol=1e-5)


def test_importance_context_pruner() -> None:
    """Test ImportanceContextPruner scoring and pruning."""
    pruner = ImportanceContextPruner(embed_dim=16, max_context=5)
    embeddings = torch.randn(12, 16)
    timestamps = torch.linspace(0, 10, 12)
    query = torch.randn(16)

    scores = pruner.compute_importance(embeddings, timestamps, query)
    assert scores.shape == (12,)
    assert (scores >= 0).all()

    pruned, indices = pruner.prune(embeddings, timestamps, query)
    assert pruned.shape == (5, 16)
    assert len(indices) == 5
    assert (indices[:-1] <= indices[1:]).all()  # Maintained order

    # Forward pass alias check
    fwd_pruned, fwd_indices = pruner(embeddings, timestamps)
    assert fwd_pruned.shape == (5, 16)
    assert len(fwd_indices) == 5


def test_latent_cache() -> None:
    """Test LatentCache put, get, LRU eviction, and stats."""
    cache = LatentCache(max_size_mb=1.0, device="cpu")
    data = torch.randn(10, 16)
    compressed = {"dense": torch.randn(10, 8)}

    # Miss before put
    assert cache.get(data) is None

    cache.put(data, compressed)
    cached = cache.get(data)
    assert cached is not None
    assert isinstance(cached, dict)
    assert "dense" in cached
    assert torch.allclose(cached["dense"], compressed["dense"])

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["num_entries"] == 1

    cache.invalidate()
    assert cache.get(data) is None
    assert cache.stats()["num_entries"] == 0


def test_cached_chunked_compactor() -> None:
    """Test CachedChunkedCompactor integration."""
    base = DummyCompactor(embed_dim=16)
    cached_compactor = CachedChunkedCompactor(base, chunk_size=10, cache_size_mb=5.0)

    data = torch.randn(10, 16)
    results1 = cached_compactor.compress(data)
    assert len(results1) == 1

    # Second call should hit cache
    results2 = cached_compactor.compress(data)
    assert len(results2) == 1

    stats = cached_compactor.cache_stats()
    assert stats["hits"] >= 1
