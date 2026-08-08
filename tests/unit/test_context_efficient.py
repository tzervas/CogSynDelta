"""Unit tests for context-efficient memory components."""

import torch
from torch import nn

from cogsyndelta.memory.context_efficient import (
    ChunkedCompactor,
    ImportanceContextPruner,
    LatentCache,
    VRAMBudget,
)


class DummyCompactor(nn.Module):
    """A dummy compactor for testing purposes."""

    def __init__(self, embed_dim: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress dummy logic (halve values)."""
        return x * 0.5

    def reconstruct(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decompress dummy logic (double values)."""
        return compressed * 2.0


def test_vram_budget() -> None:
    """Test VRAMBudget calculations."""
    budget = VRAMBudget(max_mb=100.0, safety_margin=0.2, auto_detect=False)
    assert budget.effective_budget() == 80.0


def test_chunked_compactor_no_overlap() -> None:
    """Test ChunkedCompactor with zero overlap."""
    base = DummyCompactor(embed_dim=8)
    compactor = ChunkedCompactor(base, chunk_size=5, overlap=0)

    # 12 items, should be split into chunks of [5, 5, 2]
    x = torch.randn(12, 8)
    compressed = compactor.compress(x)
    assert len(compressed) == 3
    assert compressed[0].shape == (5, 8)
    assert compressed[1].shape == (5, 8)
    assert compressed[2].shape == (2, 8)

    reconstructed = compactor.reconstruct(compressed)
    assert reconstructed.shape == (12, 8)
    assert torch.allclose(x, reconstructed)


def test_chunked_compactor_with_overlap_and_blend() -> None:
    """Test ChunkedCompactor with overlap and trapezoidal linear blend."""
    base = DummyCompactor(embed_dim=8)
    # chunk_size=5, overlap=2, step=3
    # Chunk 1 covers 0 to 5
    # Chunk 2 covers 3 to 8
    # Chunk 3 covers 6 to 11
    # Chunk 4 covers 9 to 12
    compactor = ChunkedCompactor(base, chunk_size=5, overlap=2)

    x = torch.randn(12, 8)
    compressed = compactor.compress(x)
    assert len(compressed) == 4

    reconstructed = compactor.reconstruct(compressed)
    assert reconstructed.shape == (12, 8)
    # Due to float division/precision and dummy doubling/halving,
    # the linear ramp blend should reconstruct the exact original values!
    assert torch.allclose(x, reconstructed, atol=1e-5)


def test_importance_context_pruner() -> None:
    """Test ImportanceContextPruner."""
    pruner = ImportanceContextPruner(embed_dim=8, max_context=5)
    embeddings = torch.randn(10, 8)
    timestamps = torch.arange(10, dtype=torch.float32)

    pruned, kept_indices = pruner.prune(embeddings, timestamps)
    assert pruned.shape == (5, 8)
    assert len(kept_indices) == 5


def test_latent_cache() -> None:
    """Test LatentCache LRU eviction and hits/misses."""
    cache = LatentCache(max_size_mb=1.0)
    x = torch.randn(4, 8)
    y = torch.randn(4, 4)

    assert cache.get(x) is None
    cache.put(x, y)

    cached = cache.get(x)
    assert cached is not None
    assert torch.allclose(cached, y)

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1

    cache.invalidate()
    assert cache.get(x) is None
