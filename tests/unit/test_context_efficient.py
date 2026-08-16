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
)


class DummyCompactor(nn.Module):
    """Dummy compactor for testing."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, embed_dim // 2)
        self.decoder = nn.Linear(embed_dim // 2, embed_dim)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class IdentityCompactor(nn.Module):
    """Identity compactor for testing exact reconstruction and blending."""

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_estimate_tensor_memory() -> None:
    shape = (1000, 512)
    mb_f32 = estimate_tensor_memory(shape, torch.float32)
    assert abs(mb_f32 - (1000 * 512 * 4 / (1024 * 1024))) < 1e-4

    mb_f16 = estimate_tensor_memory(shape, torch.float16)
    assert abs(mb_f16 - (1000 * 512 * 2 / (1024 * 1024))) < 1e-4


def test_vram_budget() -> None:
    budget = VRAMBudget(max_mb=1000.0, safety_margin=0.2, auto_detect=False)
    assert budget.effective_budget() == 800.0


def test_chunked_compactor_compress_reconstruct() -> None:
    base = DummyCompactor(embed_dim=32)
    compactor = ChunkedCompactor(base_compactor=base, chunk_size=10, overlap=0)

    x = torch.randn(25, 32)
    chunks = compactor.compress(x)
    assert len(chunks) == 3  # 10 + 10 + 5
    assert chunks[0].shape == (10, 16)
    assert chunks[1].shape == (10, 16)
    assert chunks[2].shape == (5, 16)

    recon = compactor.reconstruct(chunks)
    assert recon.shape == (25, 32)


def test_chunked_compactor_overlapping_blend() -> None:
    base = IdentityCompactor()
    overlap = 4
    compactor = ChunkedCompactor(base_compactor=base, chunk_size=10, overlap=overlap)

    # Linearly increasing tensor to verify blending
    x = torch.linspace(0.0, 100.0, 22).unsqueeze(1).repeat(1, 16)
    chunks = compactor.compress(x)
    recon = compactor.reconstruct(chunks)

    assert recon.shape == x.shape
    # Trapezoidal blending of linear sequence with identity should match original sequence
    torch.testing.assert_close(recon, x, rtol=1e-4, atol=1e-4)


def test_importance_context_pruner() -> None:
    pruner = ImportanceContextPruner(embed_dim=32, max_context=10)
    x = torch.randn(25, 32)
    timestamps = torch.arange(25, dtype=torch.float32)

    pruned, indices = pruner.prune(x, timestamps=timestamps)
    assert pruned.shape == (10, 32)
    assert len(indices) == 10
    # Indices should maintain chronological order
    assert (indices[:-1] <= indices[1:]).all()


def test_latent_cache() -> None:
    cache = LatentCache(max_size_mb=1.0, device="cpu")
    x = torch.randn(10, 32)
    z = torch.randn(10, 16)

    assert cache.get(x) is None
    cache.put(x, z)

    cached_z = cache.get(x)
    assert cached_z is not None
    torch.testing.assert_close(cached_z, z)

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1

    cache.invalidate()
    assert cache.get(x) is None


def test_cached_chunked_compactor() -> None:
    base = DummyCompactor(embed_dim=32)
    compactor = CachedChunkedCompactor(base_compactor=base, chunk_size=10, cache_size_mb=1.0)

    x = torch.randn(10, 32)  # Fits in 1 chunk
    res1 = compactor.compress(x)
    res2 = compactor.compress(x)

    assert compactor.cache_stats()["hits"] == 1
    torch.testing.assert_close(res1[0], res2[0])
