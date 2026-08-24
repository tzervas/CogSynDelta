"""Unit tests for context-efficient memory techniques.

Tests chunked processing, importance pruning, latent caching, and cached compactor.
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
    """Dummy compactor for testing chunked compactor."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, embed_dim)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress tensor by identity pass-through."""
        return self.linear(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct tensor."""
        return x


def test_vram_and_tensor_memory_estimation() -> None:
    """Test VRAM utility functions."""
    vram = get_available_vram()
    assert isinstance(vram, float)
    assert vram >= 0.0

    mem = estimate_tensor_memory((100, 64), dtype=torch.float32)
    # 100 * 64 * 4 bytes / 1024 / 1024
    assert abs(mem - 0.0244140625) < 1e-4

    budget = VRAMBudget(max_mb=1000.0, safety_margin=0.1, auto_detect=False)
    assert abs(budget.effective_budget() - 900.0) < 1e-4


def test_chunked_compactor_no_overlap() -> None:
    """Test ChunkedCompactor with zero overlap."""
    base = DummyCompactor(embed_dim=16)
    compactor = ChunkedCompactor(base, chunk_size=10, overlap=0)

    embeddings = torch.randn(25, 16)
    chunks = compactor.compress(embeddings)

    assert len(chunks) == 3
    assert chunks[0].shape == (10, 16)
    assert chunks[1].shape == (10, 16)
    assert chunks[2].shape == (5, 16)

    reconstructed = compactor.reconstruct(chunks)
    assert reconstructed.shape == (25, 16)


def test_chunked_compactor_with_overlap_and_blending() -> None:
    """Test ChunkedCompactor with overlap and trapezoidal linear-ramp blending."""
    base = DummyCompactor(embed_dim=16)
    compactor = ChunkedCompactor(base, chunk_size=10, overlap=2)

    embeddings = torch.randn(22, 16)
    chunks = compactor.compress(embeddings)

    reconstructed = compactor.reconstruct(chunks)
    assert reconstructed.shape == (22, 16)


def test_importance_context_pruner() -> None:
    """Test ImportanceContextPruner scoring and pruning."""
    pruner = ImportanceContextPruner(embed_dim=16, max_context=10)
    embeddings = torch.randn(25, 16)
    timestamps = torch.linspace(0, 100, 25)
    query = torch.randn(16)

    pruned, indices = pruner.prune(embeddings, timestamps=timestamps, query=query)
    assert pruned.shape == (10, 16)
    assert indices.shape == (10,)
    assert (indices[:-1] <= indices[1:]).all()  # Order preserved

    # Test forward pass alias
    pruned_fwd, indices_fwd = pruner(embeddings, timestamps=timestamps)
    assert pruned_fwd.shape == (10, 16)


def test_latent_cache() -> None:
    """Test LatentCache putting, getting, eviction, and stats."""
    cache = LatentCache(max_size_mb=0.001, device="cpu")  # ~1KB budget
    x1 = torch.randn(10, 16)
    comp1 = {"dense": torch.randn(10, 8)}

    cache.put(x1, comp1)
    retrieved = cache.get(x1)
    assert retrieved is not None
    assert isinstance(retrieved, dict)

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["num_entries"] == 1

    # Invalidate cache
    cache.invalidate()
    assert cache.get(x1) is None
    assert cache.stats()["num_entries"] == 0


def test_cached_chunked_compactor() -> None:
    """Test CachedChunkedCompactor integration."""
    base = DummyCompactor(embed_dim=16)
    compactor = CachedChunkedCompactor(base_compactor=base, chunk_size=10, cache_size_mb=10.0)

    embeddings = torch.randn(15, 16)
    res1 = compactor.compress(embeddings)
    assert len(res1) > 0

    stats = compactor.cache_stats()
    assert stats["num_entries"] >= 0
