"""Unit tests for context-efficient memory components in CogSynDelta.

Tests the correctness, memory efficiency, and robustness of ChunkedCompactor,
ImportanceContextPruner, LatentCache, and VRAMBudget configurations.
"""

import pytest
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


class MockCompactor(nn.Module):
    """A mock compactor module for testing ChunkedCompactor.

    Why this mock:
        By using a mathematically reversible mock, we can easily verify
        the numerical reconstruction accuracy of ChunkedCompactor with and
        without overlapping regions.
    """

    def __init__(self, embed_dim: int) -> None:
        """Initialize mock compactor.

        Args:
            embed_dim: Embedding dimensions.
        """
        super().__init__()
        self.embed_dim = embed_dim

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress the input by doubling its values.

        Args:
            x: Input tensor.

        Returns:
            Compressed output.
        """
        return x * 2.0

    def reconstruct(self, compressed: torch.Tensor) -> torch.Tensor:
        """Reconstruct the original tensor by halving.

        Args:
            compressed: Compressed input.

        Returns:
            Reconstructed tensor.
        """
        return compressed / 2.0


def test_vram_estimation_and_utility() -> None:
    """Test memory estimation and utility helper functions."""
    # Estimate memory of a [1000, 512] float32 tensor
    size_mb = estimate_tensor_memory((1000, 512), torch.float32)
    # 1000 * 512 * 4 / (1024 * 1024) = 1.953125 MB
    assert pytest.approx(size_mb, rel=1e-3) == 1.953125

    # Check available VRAM runs without exception
    vram = get_available_vram()
    assert isinstance(vram, float)
    assert vram >= 0.0


def test_vram_budget() -> None:
    """Test VRAM budget configuration parameters and methods."""
    budget = VRAMBudget(max_mb=100.0, safety_margin=0.2, auto_detect=False)
    assert budget.effective_budget() == 80.0  # 100 * 0.8


def test_chunked_compactor_no_overlap() -> None:
    """Test ChunkedCompactor processes and reconstructs without overlap."""
    base = MockCompactor(embed_dim=128)
    compactor = ChunkedCompactor(base, chunk_size=100, overlap=0)

    # Input of size 250
    x = torch.randn(250, 128)
    compressed_chunks = compactor.compress(x)

    # Should split into 3 chunks: 100, 100, 50
    assert len(compressed_chunks) == 3
    assert compressed_chunks[0].shape == (100, 128)
    assert compressed_chunks[1].shape == (100, 128)
    assert compressed_chunks[2].shape == (50, 128)

    # Reconstruct and check exact equality
    reconstructed = compactor.reconstruct(compressed_chunks)
    assert reconstructed.shape == (250, 128)
    assert torch.allclose(x, reconstructed, atol=1e-5)


def test_chunked_compactor_with_overlap() -> None:
    """Test ChunkedCompactor with overlap and trapezoidal linear blending."""
    base = MockCompactor(embed_dim=64)
    # Chunk size 100, overlap 20
    compactor = ChunkedCompactor(base, chunk_size=100, overlap=20)

    x = torch.randn(250, 64)
    compressed_chunks = compactor.compress(x)

    # Step is 80 (100 - 20)
    # Chunk 1: [0:100]
    # Chunk 2: [80:180]
    # Chunk 3: [160:240]
    # Chunk 4: [240:250] (last chunk)
    assert len(compressed_chunks) == 4

    reconstructed = compactor.reconstruct(compressed_chunks)
    assert reconstructed.shape == (250, 64)
    # Check that blending did not distort the reconstructed values
    assert torch.allclose(x, reconstructed, atol=1e-5)


def test_importance_context_pruner() -> None:
    """Test ImportanceContextPruner reduces context length correctly."""
    pruner = ImportanceContextPruner(embed_dim=32, max_context=10)

    embeddings = torch.randn(25, 32)
    timestamps = torch.arange(25, dtype=torch.float32)

    # Prune
    pruned, kept_indices = pruner.prune(embeddings, timestamps)

    # Should prune down to max_context (10)
    assert pruned.shape == (10, 32)
    assert kept_indices.shape == (10,)
    # Original order must be preserved
    assert torch.all(kept_indices[:-1] < kept_indices[1:])


def test_latent_cache() -> None:
    """Test LatentCache LRU eviction and hit/miss reporting."""
    cache = LatentCache(max_size_mb=2.0, device="cpu")

    # [1000, 512] float32 is ~1.95 MB
    x1 = torch.ones(1000, 512)
    compressed1 = torch.ones(1000, 512) * 2.0

    cache.put(x1, compressed1)
    stats = cache.stats()
    assert stats["num_entries"] == 1
    assert stats["size_mb"] > 0.0

    # Retrieve and verify hit
    retrieved = cache.get(x1)
    assert retrieved is not None
    assert torch.allclose(retrieved, compressed1)
    assert cache.stats()["hits"] == 1

    # Add a second entry of same size. Since max_size is 2MB, the first should be evicted.
    x2 = torch.zeros(1000, 512)
    compressed2 = torch.zeros(1000, 512) * 2.0
    cache.put(x2, compressed2)

    # First entry should have been evicted
    assert cache.get(x1) is None
    assert cache.stats()["misses"] == 1
    assert cache.stats()["num_entries"] == 1

    # Invalidation
    cache.invalidate()
    assert cache.stats()["num_entries"] == 0


def test_cached_chunked_compactor() -> None:
    """Test CachedChunkedCompactor integration of chunking and caching."""
    base = MockCompactor(embed_dim=16)
    compactor = CachedChunkedCompactor(base, chunk_size=50, cache_size_mb=1.0)

    x = torch.randn(80, 16)
    res1 = compactor.compress(x)
    assert len(res1) > 0

    # Cache should be populated
    stats = compactor.cache_stats()
    assert stats["num_entries"] == 0  # It won't cache because it split into multiple chunks

    # If it is a single chunk, it should be cached
    x_small = torch.randn(30, 16)
    res2 = compactor.compress(x_small)
    assert len(res2) == 1

    # Second time should hit cache
    res3 = compactor.compress(x_small)
    assert len(res3) == 1
    assert compactor.cache_stats()["hits"] == 1
