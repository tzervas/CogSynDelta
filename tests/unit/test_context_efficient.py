"""Unit tests for context-efficient memory techniques.

Ensures that memory optimization techniques like chunked compactor, context pruner,
and latent cache operate correctly and safely.
"""

from __future__ import annotations

import torch
from torch import nn

from cogsyndelta.memory.context_efficient import (
    ChunkedCompactor,
    ImportanceContextPruner,
    LatentCache,
    estimate_tensor_memory,
    get_available_vram,
)


class MockCompactor(nn.Module):
    """Mock compactor that passes embeddings through unchanged.

    Why:
        Provides a fast, predictable compactor for validating the chunking and
        blending logic in ChunkedCompactor.
    """

    def __init__(self) -> None:
        """Initialize mock compactor."""
        super().__init__()

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compress mock implementation (identity).

        Args:
            embeddings: Input embeddings.

        Returns:
            The input embeddings unchanged.
        """
        return embeddings

    def reconstruct(self, compressed: torch.Tensor) -> torch.Tensor:
        """Reconstruct mock implementation (identity).

        Args:
            compressed: Compressed representations.

        Returns:
            The reconstructed embeddings unchanged.
        """
        return compressed


class TestVRAMEstimations:
    """Test VRAM utility helper functions.

    Why:
        Validates the estimation calculations used for adaptive memory-efficient scaling.
    """

    def test_get_available_vram(self) -> None:
        """Test that get_available_vram returns a float."""
        vram = get_available_vram()
        assert isinstance(vram, (int, float))
        assert vram >= 0.0

    def test_estimate_tensor_memory(self) -> None:
        """Test that estimate_tensor_memory calculates correct sizing."""
        tensor = torch.zeros((1000, 512), dtype=torch.float32)
        # 1000 * 512 * 4 bytes = 2048000 bytes = 1.953125 MB
        mem_mb = estimate_tensor_memory(tensor.shape, tensor.dtype)
        assert abs(mem_mb - 1.953125) < 1e-4


class TestChunkedCompactor:
    """Test ChunkedCompactor processing and blending logic.

    Why:
        Verifies that ChunkedCompactor correctly splits input embeddings, compresses
        and reconstructs them chunk-by-chunk, and blends overlapping boundaries smoothly.
    """

    def test_no_overlap_chunking(self) -> None:
        """Test chunked compaction with zero overlap."""
        base = MockCompactor()
        compactor = ChunkedCompactor(base, chunk_size=10, overlap=0)

        # 25 items, size 16
        inputs = torch.randn(25, 16)
        compressed = compactor.compress(inputs)

        # With chunk_size=10, we expect 3 chunks: [10, 16], [10, 16], [5, 16]
        assert len(compressed) == 3
        assert compressed[0].shape == (10, 16)
        assert compressed[1].shape == (10, 16)
        assert compressed[2].shape == (5, 16)

        reconstructed = compactor.reconstruct(compressed)
        assert reconstructed.shape == (25, 16)
        assert torch.allclose(inputs, reconstructed)

    def test_overlap_blending(self) -> None:
        """Test chunked compaction with overlap and linear-ramp blending."""
        base = MockCompactor()
        compactor = ChunkedCompactor(base, chunk_size=10, overlap=4)

        # 22 items, size 16
        # Chunks should start at:
        # Chunk 0: 0 to 10
        # Chunk 1: step is 10-4=6. 6 to 16
        # Chunk 2: 12 to 22
        # Chunk 3: 18 to 22
        inputs = torch.randn(22, 16)
        compressed = compactor.compress(inputs)

        assert len(compressed) == 4
        assert compressed[0].shape == (10, 16)
        assert compressed[1].shape == (10, 16)
        assert compressed[2].shape == (10, 16)
        assert compressed[3].shape == (4, 16)

        reconstructed = compactor.reconstruct(compressed)
        assert reconstructed.shape == (22, 16)

        # For our mock compactor, the input and reconstructed should be extremely close
        # because the linear ramp blending weights sum to 1.0 everywhere.
        assert torch.allclose(inputs, reconstructed, atol=1e-5)

    def test_progress_callback(self) -> None:
        """Test that progress callback is called with correct indices."""
        base = MockCompactor()
        compactor = ChunkedCompactor(base, chunk_size=10, overlap=0)

        inputs = torch.randn(25, 16)
        called_args: list[tuple[int, int]] = []

        def callback(processed: int, total: int) -> None:
            called_args.append((processed, total))

        compactor.compress(inputs, progress_callback=callback)

        assert len(called_args) == 3
        assert called_args == [(10, 25), (20, 25), (25, 25)]


class TestImportanceContextPruner:
    """Test ImportanceContextPruner for sequence context reduction.

    Why:
        Validates that context pruner reduces input sequences to target lengths
        while preserving top-k items.
    """

    def test_pruning_limits(self) -> None:
        """Test that pruner respects max_context limit."""
        pruner = ImportanceContextPruner(embed_dim=16, max_context=10)

        inputs = torch.randn(25, 16)
        pruned, indices = pruner.prune(inputs)

        assert pruned.shape == (10, 16)
        assert len(indices) == 10
        assert indices.max() < 25
        assert indices.min() >= 0


class TestLatentCache:
    """Test LatentCache LRU and sizing bounds.

    Why:
        Ensures that LatentCache behaves as a strict LRU cache, freeing memory
        correctly when size limit is reached.
    """

    def test_cache_put_get(self) -> None:
        """Test basic put and get operations."""
        cache = LatentCache(max_size_mb=10.0, device="cpu")

        tensor_1 = torch.ones((1000, 512), dtype=torch.float32)
        tensor_2 = torch.zeros((1000, 512), dtype=torch.float32)

        cache.put(tensor_1, {"data": tensor_1})
        cache.put(tensor_2, {"data": tensor_2})

        res_1 = cache.get(tensor_1)
        res_2 = cache.get(tensor_2)

        assert res_1 is not None
        assert torch.allclose(res_1["data"], tensor_1)
        assert res_2 is not None
        assert torch.allclose(res_2["data"], tensor_2)

    def test_cache_lru_eviction(self) -> None:
        """Test cache LRU eviction when size exceeds limit."""
        # Max size 3 MB, each tensor is ~1.95 MB, so only 1 tensor can be cached
        cache = LatentCache(max_size_mb=3.0, device="cpu")

        tensor_1 = torch.ones((1000, 512), dtype=torch.float32)
        tensor_2 = torch.zeros((1000, 512), dtype=torch.float32)

        cache.put(tensor_1, {"data": tensor_1})
        assert cache.get(tensor_1) is not None

        # This should evict tensor_1 because current_size (1.95 MB) + new_size (1.95 MB) = 3.9 MB > 3.0 MB
        cache.put(tensor_2, {"data": tensor_2})

        assert cache.get(tensor_1) is None
        assert cache.get(tensor_2) is not None
