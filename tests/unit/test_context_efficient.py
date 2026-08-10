"""Unit tests for Context-Efficient Memory components.

Covers ChunkedCompactor, ImportanceContextPruner, and LatentCache.
"""

from __future__ import annotations

import torch
from torch import nn

from cogsyndelta.memory.context_efficient import (
    ChunkedCompactor,
    ImportanceContextPruner,
    LatentCache,
    VRAMBudget,
    estimate_tensor_memory,
    get_available_vram,
)


class MockCompactor(nn.Module):
    """Simple mock compactor for testing ChunkedCompactor."""

    def __init__(self, embed_dim: int = 512) -> None:
        """Initialize mock compactor.

        Args:
            embed_dim: Embedding dimension.
        """
        super().__init__()
        self.embed_dim = embed_dim

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress by multiplying by 2.

        Args:
            x: Input tensor.

        Returns:
            Compressed tensor.
        """
        return x * 2.0

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct by dividing by 2.

        Args:
            x: Compressed tensor.

        Returns:
            Reconstructed tensor.
        """
        return x / 2.0


def test_vram_estimation_and_budget() -> None:
    """Test VRAM utility functions and budget configuration."""
    vram = get_available_vram()
    assert isinstance(vram, float)
    assert vram >= 0.0

    # Estimate tensor memory
    mem_32 = estimate_tensor_memory((1000, 512), dtype=torch.float32)
    # 1000 * 512 * 4 bytes = 2,048,000 bytes = ~1.953 MB
    assert 1.9 <= mem_32 <= 2.0

    mem_16 = estimate_tensor_memory((1000, 512), dtype=torch.float16)
    assert 0.9 <= mem_16 <= 1.0

    # VRAM budget
    budget = VRAMBudget(max_mb=100.0, safety_margin=0.2, auto_detect=False)
    assert budget.effective_budget() == 80.0


def test_chunked_compactor_no_overlap() -> None:
    """Test ChunkedCompactor without overlap."""
    base = MockCompactor(embed_dim=128)
    compactor = ChunkedCompactor(base, chunk_size=10, overlap=0)

    # 25 samples, should produce chunks of sizes: 10, 10, 5
    x = torch.randn(25, 128)
    compressed = compactor.compress(x)
    assert len(compressed) == 3
    assert compressed[0].shape == (10, 128)
    assert compressed[1].shape == (10, 128)
    assert compressed[2].shape == (5, 128)

    reconstructed = compactor.reconstruct(compressed)
    assert reconstructed.shape == (25, 128)
    assert torch.allclose(x, reconstructed, atol=1e-5)


def test_chunked_compactor_with_overlap() -> None:
    """Test ChunkedCompactor with overlap and trapezoidal blending."""
    base = MockCompactor(embed_dim=128)
    # Overlap of 4, chunk size of 10.
    # Step = 6.
    # Chunk 1: [0:10]
    # Chunk 2: [6:16]
    # Chunk 3: [12:22]
    # Chunk 4: [18:25] (size 7)
    # Chunk 5: [24:25] (size 1)
    compactor = ChunkedCompactor(base, chunk_size=10, overlap=4)

    x = torch.randn(25, 128)
    compressed = compactor.compress(x)
    assert len(compressed) == 5

    reconstructed = compactor.reconstruct(compressed)
    assert reconstructed.shape == (25, 128)
    # Blending should reconstruct the original tensor perfectly
    assert torch.allclose(x, reconstructed, atol=1e-5)


def test_importance_context_pruner() -> None:
    """Test ImportanceContextPruner."""
    pruner = ImportanceContextPruner(embed_dim=128, max_context=15)

    embeddings = torch.randn(30, 128)
    timestamps = torch.linspace(0, 100, 30)

    # Pruning to max_context (15)
    pruned, indices = pruner.prune(embeddings, timestamps)
    assert pruned.shape == (15, 128)
    assert len(indices) == 15
    # Should maintain sorted order of indices
    assert torch.all(indices[:-1] <= indices[1:])

    # Under threshold - should keep all
    small_embeddings = torch.randn(10, 128)
    pruned_small, indices_small = pruner.prune(small_embeddings)
    assert pruned_small.shape == (10, 128)
    assert len(indices_small) == 10


def test_latent_cache() -> None:
    """Test LatentCache LRU behavior, hashing, and versioning."""
    cache = LatentCache(max_size_mb=1.0, device="cpu")

    x1 = torch.randn(100, 128)
    compressed1 = torch.randn(100, 64)

    x2 = torch.randn(100, 128)
    compressed2 = torch.randn(100, 64)

    # Initially empty
    assert cache.get(x1) is None

    # Cache put
    cache.put(x1, compressed1)
    cached = cache.get(x1)
    assert cached is not None
    assert torch.allclose(cached, compressed1)

    # Statistics
    stats = cache.stats()
    assert stats["num_entries"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1

    # Invalidation on version increment
    cache.invalidate()
    assert cache.get(x1) is None

    # Size-based eviction (LRU)
    # Estimate size of 100x64 float32 is ~0.024 MB.
    # Let's set a very tiny limit of 0.03 MB to trigger eviction.
    tiny_cache = LatentCache(max_size_mb=0.03, device="cpu")
    tiny_cache.put(x1, compressed1)
    tiny_cache.put(x2, compressed2)

    # x1 might be evicted because of the limit
    # (combined size ~ 0.048 MB > 0.03 MB)
    assert len(tiny_cache.cache) == 1
    assert tiny_cache.get(x1) is None
    assert tiny_cache.get(x2) is not None
