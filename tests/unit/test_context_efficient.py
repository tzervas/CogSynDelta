"""Unit tests for context-efficient memory components in cogsyndelta.memory.context_efficient."""

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
    """Dummy compactor module for testing ChunkedCompactor."""

    def __init__(self, embed_dim: int = 64) -> None:
        """Initialize dummy compactor."""
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim // 2)
        self.deconv = nn.Linear(embed_dim // 2, embed_dim)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress input tensor."""
        return self.linear(x)

    def reconstruct(self, compressed: torch.Tensor) -> torch.Tensor:
        """Reconstruct compressed tensor."""
        return self.deconv(compressed)


class TestVRAMBudgetAndMemory:
    """Tests for memory estimation and VRAMBudget functions."""

    def test_estimate_tensor_memory(self) -> None:
        """Test tensor memory estimation for various dtypes."""
        shape = (1000, 512)  # 512,000 floats
        # Float32: 512,000 * 4 bytes / (1024 * 1024) = ~1.953MB
        mem_f32 = estimate_tensor_memory(shape, torch.float32)
        assert abs(mem_f32 - 1.953125) < 1e-3

        mem_f16 = estimate_tensor_memory(shape, torch.float16)
        assert abs(mem_f16 - (mem_f32 / 2)) < 1e-3

    def test_vram_budget_effective(self) -> None:
        """Test effective budget calculation."""
        budget = VRAMBudget(max_mb=1000.0, safety_margin=0.1, auto_detect=False)
        assert budget.effective_budget() == 900.0

    def test_get_available_vram(self) -> None:
        """Test available VRAM helper returns numeric value."""
        available = get_available_vram()
        assert isinstance(available, float)
        assert available >= 0.0


class TestChunkedCompactor:
    """Tests for ChunkedCompactor module."""

    def test_initialization(self) -> None:
        """Test ChunkedCompactor initialization."""
        base = DummyCompactor(embed_dim=64)
        compactor = ChunkedCompactor(base, chunk_size=100, overlap=10)
        assert compactor.chunk_size == 100
        assert compactor.overlap == 10

    def test_compress_and_reconstruct_no_overlap(self) -> None:
        """Test compression and reconstruction without overlap."""
        base = DummyCompactor(embed_dim=64)
        compactor = ChunkedCompactor(base, chunk_size=25, overlap=0)
        embeddings = torch.randn(100, 64)

        compressed = compactor.compress(embeddings)
        assert len(compressed) == 4  # 100 / 25
        assert compressed[0].shape == (25, 32)

        reconstructed = compactor.reconstruct(compressed)
        assert reconstructed.shape == (100, 64)

    def test_compress_and_reconstruct_with_trapezoidal_overlap(self) -> None:
        """Test trapezoidal linear-ramp blending with overlapping chunks."""
        base = DummyCompactor(embed_dim=64)
        overlap = 10
        chunk_size = 30
        compactor = ChunkedCompactor(base, chunk_size=chunk_size, overlap=overlap)

        # 50 total samples with step = 20 (chunk_size - overlap)
        # Chunk 0: 0..30, Chunk 1: 20..50, Chunk 2: 40..50
        embeddings = torch.randn(50, 64)

        compressed = compactor.compress(embeddings)
        assert len(compressed) == 3

        reconstructed = compactor.reconstruct(compressed)
        assert reconstructed.shape == (50, 64)


class TestImportanceContextPruner:
    """Tests for ImportanceContextPruner module."""

    def test_passthrough_when_under_max_context(self) -> None:
        """Test that pruner passes through embeddings when size <= max_context."""
        pruner = ImportanceContextPruner(embed_dim=32, max_context=50)
        embeddings = torch.randn(30, 32)

        pruned, indices = pruner.prune(embeddings)
        assert pruned.shape == (30, 32)
        assert len(indices) == 30

    def test_pruning_to_max_context(self) -> None:
        """Test pruning to max_context entries based on importance."""
        pruner = ImportanceContextPruner(embed_dim=32, max_context=20)
        embeddings = torch.randn(50, 32)
        timestamps = torch.linspace(0.0, 10.0, 50)
        query = torch.randn(32)

        pruned, indices = pruner.prune(embeddings, timestamps=timestamps, query=query)
        assert pruned.shape == (20, 32)
        assert len(indices) == 20
        # Verify indices are sorted (preserving original sequence order)
        assert torch.all(indices[:-1] <= indices[1:])


class TestLatentCache:
    """Tests for LatentCache module."""

    def test_cache_put_get_and_stats(self) -> None:
        """Test basic caching behavior."""
        cache = LatentCache(max_size_mb=10.0)
        embeddings = torch.randn(10, 32)
        compressed = torch.randn(10, 16)

        assert cache.get(embeddings) is None
        assert cache.stats()["misses"] == 1

        cache.put(embeddings, compressed)
        retrieved = cache.get(embeddings)
        assert retrieved is not None
        assert torch.allclose(retrieved, compressed)
        assert cache.stats()["hits"] == 1

    def test_cache_invalidation(self) -> None:
        """Test cache invalidation on version update."""
        cache = LatentCache(max_size_mb=10.0)
        embeddings = torch.randn(10, 32)
        compressed = torch.randn(10, 16)

        cache.put(embeddings, compressed)
        assert cache.get(embeddings) is not None

        cache.invalidate()
        assert cache.get(embeddings) is None


class TestCachedChunkedCompactor:
    """Tests for CachedChunkedCompactor integration."""

    def test_compress_and_cache(self) -> None:
        """Test caching integrated with chunked compaction."""
        base = DummyCompactor(embed_dim=32)
        cached_compactor = CachedChunkedCompactor(base, chunk_size=20, cache_size_mb=10.0)

        embeddings = torch.randn(20, 32)
        res1 = cached_compactor.compress(embeddings)
        assert len(res1) == 1

        # Second compression hit cache
        res2 = cached_compactor.compress(embeddings)
        assert len(res2) == 1
        assert cached_compactor.cache_stats()["hits"] == 1
