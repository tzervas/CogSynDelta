"""Unit tests for context-efficient memory components."""

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


class SimpleCompactor(nn.Module):
    """Dummy compactor for testing ChunkedCompactor."""

    def __init__(self, embed_dim: int = 64) -> None:
        """Initialize simple compactor."""
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Linear(embed_dim, embed_dim // 2)
        self.decoder = nn.Linear(embed_dim // 2, embed_dim)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress input tensor."""
        return self.encoder(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input tensor."""
        return self.decoder(x)


def test_vram_and_memory_estimation() -> None:
    """Test get_available_vram, estimate_tensor_memory, and VRAMBudget."""
    vram = get_available_vram()
    assert isinstance(vram, float)
    assert vram >= 0.0

    mem = estimate_tensor_memory((100, 64), torch.float32)
    assert abs(mem - (100 * 64 * 4 / (1024 * 1024))) < 1e-5

    budget = VRAMBudget(max_mb=1000.0, safety_margin=0.1, auto_detect=False)
    assert abs(budget.effective_budget() - 900.0) < 1e-5


def test_chunked_compactor_no_overlap() -> None:
    """Test ChunkedCompactor without chunk overlap."""
    base = SimpleCompactor(embed_dim=32)
    chunked = ChunkedCompactor(base, chunk_size=20, overlap=0)

    data = torch.randn(50, 32)
    compressed_chunks = chunked.compress(data)

    assert len(compressed_chunks) == 3
    assert compressed_chunks[0].shape == (20, 16)
    assert compressed_chunks[1].shape == (20, 16)
    assert compressed_chunks[2].shape == (10, 16)

    reconstructed = chunked.reconstruct(compressed_chunks)
    assert reconstructed.shape == (50, 32)


def test_chunked_compactor_with_overlap() -> None:
    """Test ChunkedCompactor with chunk overlap blending."""
    base = SimpleCompactor(embed_dim=32)
    chunked = ChunkedCompactor(base, chunk_size=20, overlap=4)

    # 2 chunks with overlap 4 on 32 samples total step = 16
    data = torch.randn(32, 32)
    compressed_chunks = chunked.compress(data)

    assert len(compressed_chunks) == 2
    assert compressed_chunks[0].shape == (20, 16)
    assert compressed_chunks[1].shape == (16, 16)

    reconstructed = chunked.reconstruct(compressed_chunks)
    assert reconstructed.shape == (32, 32)


def test_importance_context_pruner() -> None:
    """Test ImportanceContextPruner."""
    pruner = ImportanceContextPruner(embed_dim=32, max_context=10)
    data = torch.randn(25, 32)
    timestamps = torch.linspace(1.0, 25.0, 25)

    scores = pruner.compute_importance(data, timestamps=timestamps)
    assert scores.shape == (25,)

    pruned, indices = pruner.prune(data, timestamps=timestamps)
    assert pruned.shape == (10, 32)
    assert len(indices) == 10
    # Check that indices maintain original order
    assert (indices[:-1] < indices[1:]).all()


def test_latent_cache() -> None:
    """Test LatentCache get, put, and invalidate."""
    cache = LatentCache(max_size_mb=10.0, device="cpu")
    data = torch.randn(10, 32)
    compressed = {"dense": torch.randn(10, 16)}

    # Miss first
    assert cache.get(data) is None

    cache.put(data, compressed)

    # Hit second
    retrieved = cache.get(data)
    assert retrieved is not None
    assert isinstance(retrieved, dict)
    assert torch.allclose(retrieved["dense"], compressed["dense"])

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5

    cache.invalidate()
    assert cache.get(data) is None


def test_cached_chunked_compactor() -> None:
    """Test CachedChunkedCompactor integration."""
    base = SimpleCompactor(embed_dim=32)
    cached_compactor = CachedChunkedCompactor(base, chunk_size=20, cache_size_mb=10.0)

    data = torch.randn(15, 32)
    result1 = cached_compactor.compress(data)
    assert len(result1) == 1

    # Second pass should hit cache
    result2 = cached_compactor.compress(data)
    assert len(result2) == 1

    stats = cached_compactor.cache_stats()
    assert stats["hits"] == 1
