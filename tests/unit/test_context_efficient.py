"""Unit tests for context-efficient memory techniques.

Tests:
    - ChunkedCompactor: Chunked processing with/without overlap and linear-ramp blending
    - ImportanceContextPruner: Content, recency, and centrality-based context pruning
    - LatentCache: LRU cache for compressed states with eviction and invalidation
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cogsyndelta.memory.context_efficient import (
    ChunkedCompactor,
    ImportanceContextPruner,
    LatentCache,
    VRAMBudget,
    estimate_tensor_memory,
)


class ToyCompactor(nn.Module):
    """A toy compactor implementation for testing ChunkedCompactor.

    Why:
        Provides a simple deterministic target with both compression and reconstruction
        capabilities to test the chunking and blending wrapper.
    """

    def __init__(self, embed_dim: int = 512) -> None:
        """Initialize ToyCompactor."""
        super().__init__()
        self.embed_dim = embed_dim

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress tensor by scaling."""
        return x * 0.5

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct tensor by scaling back."""
        return x * 2.0


def test_vram_budget_and_estimation() -> None:
    """Test memory estimation and budget utilities."""
    # Estimate float32 tensor
    size_mb = estimate_tensor_memory((1000, 512), dtype=torch.float32)
    expected = (1000 * 512 * 4) / (1024 * 1024)
    assert abs(size_mb - expected) < 1e-5

    # Estimate float16 tensor
    size_mb_16 = estimate_tensor_memory((1000, 512), dtype=torch.float16)
    assert abs(size_mb_16 - expected / 2.0) < 1e-5

    # Budget configuration
    budget = VRAMBudget(max_mb=100.0, safety_margin=0.2, auto_detect=False)
    assert budget.effective_budget() == 80.0


def test_chunked_compactor_no_overlap() -> None:
    """Test ChunkedCompactor processing without overlap."""
    base = ToyCompactor(embed_dim=128)
    compactor = ChunkedCompactor(base, chunk_size=100, overlap=0, clear_cache=False)

    # 250 samples
    x = torch.randn(250, 128)
    compressed = compactor.compress(x)

    # Should split into 3 chunks: 100, 100, 50
    assert len(compressed) == 3
    assert compressed[0].size(0) == 100
    assert compressed[1].size(0) == 100
    assert compressed[2].size(0) == 50

    # Reconstruct
    reconstructed = compactor.reconstruct(compressed)
    assert reconstructed.shape == x.shape
    assert torch.allclose(reconstructed, x, rtol=1e-5, atol=1e-5)


def test_chunked_compactor_with_overlap() -> None:
    """Test ChunkedCompactor with overlap and trapezoidal linear-ramp blending."""
    base = ToyCompactor(embed_dim=64)
    compactor = ChunkedCompactor(base, chunk_size=100, overlap=20, clear_cache=False)

    # 260 samples
    # Step size is chunk_size - overlap = 80
    # Chunk 0: 0:100
    # Chunk 1: 80:180
    # Chunk 2: 160:260
    # Chunk 3: 240:260
    x = torch.randn(260, 64)
    compressed = compactor.compress(x)

    assert len(compressed) == 4
    assert compressed[0].size(0) == 100
    assert compressed[1].size(0) == 100
    assert compressed[2].size(0) == 100
    assert compressed[3].size(0) == 20

    # Reconstruct
    reconstructed = compactor.reconstruct(compressed)
    assert reconstructed.shape == x.shape
    # Due to linear-ramp blending, reconstruction should be exact!
    assert torch.allclose(reconstructed, x, rtol=1e-5, atol=1e-5)


def test_importance_context_pruner() -> None:
    """Test ImportanceContextPruner scoring and pruning."""
    pruner = ImportanceContextPruner(embed_dim=64, max_context=50)

    # 120 embeddings
    embeddings = torch.randn(120, 64)
    timestamps = torch.arange(120, dtype=torch.float32)

    # Prune
    pruned, kept_indices = pruner.prune(embeddings, timestamps)

    assert pruned.size(0) == 50
    assert kept_indices.size(0) == 50
    # Check that indices are sorted (original temporal order is preserved)
    assert torch.all(kept_indices[:-1] <= kept_indices[1:])

    # Scoring with query
    query = torch.randn(64)
    scores = pruner.compute_importance(embeddings, timestamps, query)
    assert scores.shape == (120,)
    assert torch.all(scores >= 0)


def test_latent_cache_lru_and_invalidation() -> None:
    """Test LatentCache LRU eviction and version invalidation."""
    # Cache on CPU with a small limit of 0.5MB
    cache = LatentCache(max_size_mb=0.5, device="cpu")

    emb1 = torch.randn(500, 512)  # about 1MB original, compressed is about 0.488MB
    compressed1 = torch.randn(500, 256)

    # Put and get
    cache.put(emb1, compressed1)
    retrieved = cache.get(emb1)
    assert retrieved is not None
    assert torch.allclose(retrieved, compressed1)

    # Eviction: put a second item of similar size
    emb2 = torch.randn(500, 512)
    compressed2 = torch.randn(500, 256)
    cache.put(emb2, compressed2)

    # Since capacity is 0.5MB and each item is ~0.488MB, putting emb2 should evict emb1
    assert cache.get(emb2) is not None
    assert cache.get(emb1) is None

    # Invalidation
    cache.put(emb2, compressed2)
    assert cache.get(emb2) is not None
    cache.invalidate()
    assert cache.get(emb2) is None
    assert cache.stats()["num_entries"] == 0
