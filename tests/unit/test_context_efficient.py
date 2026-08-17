"""Tests for context-efficient memory components."""

import torch
from torch import nn

from cogsyndelta.memory.context_efficient import (
    ChunkedCompactor,
    ImportanceContextPruner,
    LatentCache,
)


class DummyCompactor(nn.Module):
    """Dummy compactor for testing ChunkedCompactor."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim // 2)
        self.decoder = nn.Linear(embed_dim // 2, embed_dim)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


def test_chunked_compactor_no_overlap() -> None:
    base = DummyCompactor(embed_dim=64)
    compactor = ChunkedCompactor(base_compactor=base, chunk_size=10, overlap=0)

    data = torch.randn(25, 64)
    chunks = compactor.compress(data)
    assert len(chunks) == 3  # 10 + 10 + 5

    reconstructed = compactor.reconstruct(chunks)
    assert reconstructed.shape == (25, 64)


def test_chunked_compactor_with_overlap_blending() -> None:
    base = DummyCompactor(embed_dim=64)
    compactor = ChunkedCompactor(base_compactor=base, chunk_size=10, overlap=4)

    # 100 samples with chunk_size=10 and overlap=4 (step=6)
    data = torch.randn(100, 64)
    chunks = compactor.compress(data)

    reconstructed = compactor.reconstruct(chunks)
    assert reconstructed.shape == (100, 64)


def test_importance_context_pruner() -> None:
    pruner = ImportanceContextPruner(embed_dim=64, max_context=10)
    data = torch.randn(25, 64)
    timestamps = torch.linspace(0, 10, 25)

    pruned, indices = pruner.prune(data, timestamps=timestamps)
    assert pruned.shape == (10, 64)
    assert indices.shape == (10,)


def test_latent_cache() -> None:
    cache = LatentCache(max_size_mb=10.0)
    data = torch.randn(10, 64)
    compressed = {"dense": torch.randn(10, 32)}

    assert cache.get(data) is None
    cache.put(data, compressed)

    cached = cache.get(data)
    assert cached is not None
    assert torch.equal(cached["dense"], compressed["dense"])
