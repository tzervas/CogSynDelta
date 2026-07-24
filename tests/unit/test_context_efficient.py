"""Unit tests for context-efficient memory components.

Tests the ChunkedCompactor, ImportanceContextPruner, and LatentCache classes
to verify correct behavior, correctness of trapezoidal blending weights,
relevance-based pruning, and LRU cache statistics.
"""

import torch
from torch import nn

from cogsyndelta.memory.context_efficient import (
    ChunkedCompactor,
    ImportanceContextPruner,
    LatentCache,
    VRAMBudget,
    estimate_tensor_memory,
)


class MockCompactor(nn.Module):
    """Mock compactor for wrapping in ChunkedCompactor."""

    def __init__(self) -> None:
        """Initialize mock compactor."""
        super().__init__()

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Mock compress operation.

        Args:
            embeddings: Input embeddings.

        Returns:
            Embeddings directly.
        """
        return embeddings

    def reconstruct(self, compressed: torch.Tensor) -> torch.Tensor:
        """Mock reconstruct operation.

        Args:
            compressed: Compressed embeddings.

        Returns:
            Reconstructed embeddings.
        """
        return compressed


class TestVRAMBudget:
    """Test suite for VRAMBudget configuration."""

    def test_default_budget(self) -> None:
        """Test default budget values."""
        budget = VRAMBudget()
        assert budget.max_mb == 2000.0
        assert budget.safety_margin == 0.1
        assert budget.effective_budget() == 1800.0


def test_estimate_tensor_memory() -> None:
    """Test tensor memory estimation logic."""
    # float32: 4 bytes per element
    # 1000 * 512 * 4 = 2,048,000 bytes = 1.953125 MB
    size = estimate_tensor_memory((1000, 512), dtype=torch.float32)
    assert 1.9 <= size <= 2.0


class TestChunkedCompactor:
    """Test suite for ChunkedCompactor."""

    def test_chunking_no_overlap(self) -> None:
        """Test chunked compression without overlap."""
        base = MockCompactor()
        compactor = ChunkedCompactor(base, chunk_size=5, overlap=0)

        inputs = torch.randn(12, 16)
        chunks = compactor.compress(inputs)

        # 12 elements with chunk_size 5 -> 3 chunks (5, 5, 2)
        assert len(chunks) == 3
        assert chunks[0].size(0) == 5
        assert chunks[1].size(0) == 5
        assert chunks[2].size(0) == 2

        recon = compactor.reconstruct(chunks)
        assert recon.shape == inputs.shape
        assert torch.allclose(recon, inputs)

    def test_chunking_with_overlap_and_trapezoidal_blending(self) -> None:
        """Test chunked compression and reconstruction with overlap and blending."""
        base = MockCompactor()
        compactor = ChunkedCompactor(base, chunk_size=6, overlap=2)

        inputs = torch.randn(14, 16)
        chunks = compactor.compress(inputs)

        # chunk_size = 6, overlap = 2
        # step = 4
        # range(0, 14, 4) -> start indices: 0, 4, 8, 12
        # Chunk 1: [0:6]
        # Chunk 2: [4:10]
        # Chunk 3: [8:14]
        # Chunk 4: [12:14]
        assert len(chunks) == 4
        assert chunks[0].size(0) == 6
        assert chunks[1].size(0) == 6
        assert chunks[2].size(0) == 6
        assert chunks[3].size(0) == 2

        recon = compactor.reconstruct(chunks)
        assert recon.shape == inputs.shape
        # The trapezoidal blending should perfectly reconstruct the identity mapping
        # because MockCompactor does not distort the values, and the weights sum to 1.
        assert torch.allclose(recon, inputs, atol=1e-5)


class TestImportanceContextPruner:
    """Test suite for ImportanceContextPruner."""

    def test_pruning_relevance(self) -> None:
        """Test that pruner correctly retains top most important embeddings."""
        pruner = ImportanceContextPruner(embed_dim=16, max_context=5)

        embeddings = torch.randn(10, 16)
        timestamps = torch.linspace(0.0, 1.0, steps=10)

        # Run prune
        pruned, indices = pruner.prune(embeddings, timestamps=timestamps)

        assert pruned.shape[0] == 5
        assert indices.shape[0] == 5
        # Order should be preserved (sorted indices)
        assert torch.all(indices[:-1] <= indices[1:])

    def test_forward_alias(self) -> None:
        """Test that forward method works identically to prune."""
        pruner = ImportanceContextPruner(embed_dim=16, max_context=3)
        embeddings = torch.randn(8, 16)

        pruned, indices = pruner(embeddings)
        assert pruned.shape[0] == 3


class TestLatentCache:
    """Test suite for LatentCache."""

    def test_cache_hits_and_misses(self) -> None:
        """Test that cache accurately registers hits, misses, and stats."""
        cache = LatentCache(max_size_mb=10.0, device="cpu")

        embeddings = torch.randn(5, 16)
        compressed = torch.randn(5, 8)

        # Initially not in cache
        cached = cache.get(embeddings)
        assert cached is None

        # Add to cache
        cache.put(embeddings, compressed)

        # Retrieve again -> hit
        cached_hit = cache.get(embeddings)
        assert cached_hit is not None
        assert torch.allclose(cached_hit, compressed)

        # Stats check
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["num_entries"] == 1

    def test_cache_eviction(self) -> None:
        """Test LRU eviction when cache size exceeds max_size_mb."""
        # Set a very small max size (e.g. 1 KB) to force eviction
        cache = LatentCache(max_size_mb=0.001, device="cpu")

        emb1 = torch.randn(10, 100)  # ~4KB (floats)
        comp1 = torch.randn(10, 100)
        emb2 = torch.randn(10, 100)
        comp2 = torch.randn(10, 100)

        cache.put(emb1, comp1)
        # Putting second one should evict first one since 4KB > 1KB
        cache.put(emb2, comp2)

        assert cache.get(emb1) is None  # Evicted
        assert cache.get(emb2) is not None  # Retained

    def test_cache_invalidate(self) -> None:
        """Test cache invalidation on model changes."""
        cache = LatentCache(max_size_mb=1.0, device="cpu")
        embeddings = torch.randn(5, 16)
        compressed = torch.randn(5, 8)

        cache.put(embeddings, compressed)
        assert cache.get(embeddings) is not None

        # Invalidate
        cache.invalidate()
        assert cache.get(embeddings) is None
        assert cache.stats()["num_entries"] == 0
