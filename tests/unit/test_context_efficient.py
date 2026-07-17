"""Tests for context-efficient memory techniques.

Validates the performance, behavior, and correctness of ChunkedCompactor,
ImportanceContextPruner, LatentCache, and CachedChunkedCompactor.
"""

from __future__ import annotations

import torch
from torch import nn

from cogsyndelta.memory.context_efficient import (
    CachedChunkedCompactor,
    ChunkedCompactor,
    ImportanceContextPruner,
    LatentCache,
)


class DummyCompactor(nn.Module):
    """Dummy compactor for testing context-efficient modules."""

    def __init__(self, embed_dim: int = 64) -> None:
        """Initialize dummy compactor.

        Args:
            embed_dim: Embedding dimension.
        """
        super().__init__()
        self.embed_dim = embed_dim

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compress by returning a smaller dimension.

        Args:
            embeddings: Input embeddings.

        Returns:
            Compressed embeddings.
        """
        return embeddings[:, : self.embed_dim // 2]

    def reconstruct(self, compressed: torch.Tensor) -> torch.Tensor:
        """Reconstruct by padding.

        Args:
            compressed: Compressed representation.

        Returns:
            Reconstructed embeddings.
        """
        # Reconstruct by repeating or padding with zeros
        padded = torch.zeros(
            compressed.size(0), self.embed_dim, device=compressed.device, dtype=compressed.dtype
        )
        padded[:, : compressed.size(1)] = compressed
        return padded


class TestChunkedCompactor:
    """Test suite for ChunkedCompactor."""

    def test_chunked_compression_no_overlap(self) -> None:
        """Test compressing and reconstructing without overlap."""
        base_compactor = DummyCompactor(embed_dim=64)
        compactor = ChunkedCompactor(base_compactor, chunk_size=10, overlap=0)

        embeddings = torch.randn(25, 64)
        compressed_chunks = compactor.compress(embeddings)

        # 25 samples with chunk_size 10 -> chunks of size 10, 10, 5
        assert len(compressed_chunks) == 3
        assert compressed_chunks[0].shape == (10, 32)
        assert compressed_chunks[1].shape == (10, 32)
        assert compressed_chunks[2].shape == (5, 32)

        reconstructed = compactor.reconstruct(compressed_chunks)
        assert reconstructed.shape == (25, 64)
        # Verify first 32 dimensions are reconstructed perfectly
        assert torch.allclose(reconstructed[:, :32], embeddings[:, :32])

    def test_chunked_compression_with_overlap(self) -> None:
        """Test compressing and reconstructing with overlap and linear blending."""
        base_compactor = DummyCompactor(embed_dim=64)
        overlap = 4
        compactor = ChunkedCompactor(base_compactor, chunk_size=12, overlap=overlap)

        # 20 samples
        embeddings = torch.randn(20, 64)
        compressed_chunks = compactor.compress(embeddings)

        # For range(0, 20, 8), start index is [0, 8, 16], yielding 3 chunks
        assert len(compressed_chunks) == 3
        assert compressed_chunks[0].shape == (12, 32)
        assert compressed_chunks[1].shape == (12, 32)
        assert compressed_chunks[2].shape == (4, 32)

        reconstructed = compactor.reconstruct(compressed_chunks)
        assert reconstructed.shape == (20, 64)

        # Verify that linear blending doesn't break the shape and is continuous
        assert not torch.isnan(reconstructed).any()


class TestImportanceContextPruner:
    """Test suite for ImportanceContextPruner."""

    def test_pruner_keeps_max_context(self) -> None:
        """Test that pruner correctly limits context size."""
        pruner = ImportanceContextPruner(embed_dim=64, max_context=10)
        embeddings = torch.randn(25, 64)
        timestamps = torch.arange(25, dtype=torch.float32)

        pruned, indices = pruner.prune(embeddings, timestamps=timestamps)
        assert pruned.shape == (10, 64)
        assert len(indices) == 10
        # Check indices are sorted (keeps original temporal order)
        assert torch.equal(indices, indices.sort().values)

    def test_pruner_no_op_when_under_max(self) -> None:
        """Test that pruner does nothing when context is already small."""
        pruner = ImportanceContextPruner(embed_dim=64, max_context=50)
        embeddings = torch.randn(25, 64)

        pruned, indices = pruner.prune(embeddings)
        assert pruned.shape == (25, 64)
        assert len(indices) == 25


class TestLatentCache:
    """Test suite for LatentCache."""

    def test_cache_hits_and_misses(self) -> None:
        """Test that cache stores and retrieves representations correctly."""
        cache = LatentCache(max_size_mb=10.0, device="cpu")

        embeddings = torch.randn(5, 64)
        compressed = torch.randn(5, 32)

        # Cache miss
        assert cache.get(embeddings) is None

        # Put in cache
        cache.put(embeddings, compressed)

        # Cache hit
        cached = cache.get(embeddings)
        assert cached is not None
        assert torch.allclose(cached, compressed)

        # Stats
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["num_entries"] == 1


class TestCachedChunkedCompactor:
    """Test suite for CachedChunkedCompactor."""

    def test_cached_chunked_compactor_flow(self) -> None:
        """Test the integration of chunking and caching."""
        base_compactor = DummyCompactor(embed_dim=64)
        compactor = CachedChunkedCompactor(
            base_compactor=base_compactor,
            chunk_size=10,
            cache_size_mb=5.0,
        )

        embeddings = torch.randn(8, 64)

        # First pass (cache miss, compresses and caches)
        results1 = compactor.compress(embeddings)
        assert len(results1) == 1

        # Second pass (cache hit, retrieved from cache)
        results2 = compactor.compress(embeddings)
        assert len(results2) == 1
        assert torch.allclose(results1[0], results2[0])

        stats = compactor.cache_stats()
        assert stats["hits"] == 1
