"""Tests for Modern Hopfield Networks."""

import pytest
import torch

from vsa.memory.hopfield import ModernHopfieldMemory


class TestModernHopfieldMemory:
    """Test Modern Hopfield Network implementation."""

    @pytest.fixture
    def memory(self):
        """Create Hopfield memory on CPU."""
        return ModernHopfieldMemory(dimension=1000, beta=1.0, device="cpu")

    @pytest.fixture
    def memory_small(self):
        """Create small Hopfield memory for fast tests."""
        return ModernHopfieldMemory(dimension=100, beta=1.0, device="cpu")

    def test_initialization(self, memory):
        """Test memory initialization."""
        assert memory.dimension == 1000
        assert memory.beta == 1.0
        assert len(memory) == 0
        assert memory.patterns.shape == (0, 1000)

    def test_store_single_pattern(self, memory_small):
        """Test storing a single pattern."""
        pattern = torch.randn(100)
        memory_small.store(pattern)

        assert len(memory_small) == 1
        assert memory_small.patterns.shape == (1, 100)

    def test_store_batch(self, memory_small):
        """Test storing multiple patterns."""
        patterns = torch.randn(5, 100)
        memory_small.store_batch(patterns)

        assert len(memory_small) == 5
        assert memory_small.patterns.shape == (5, 100)

    def test_retrieve_exact(self, memory_small):
        """Test retrieving exact stored pattern."""
        pattern = torch.randn(100)
        memory_small.store(pattern)

        # Retrieve exact pattern
        retrieved = memory_small.retrieve(pattern)

        # Should be very similar
        similarity = (pattern @ retrieved) / (torch.norm(pattern) * torch.norm(retrieved))
        assert similarity > 0.99, f"Similarity {similarity} < 0.99"

    def test_retrieve_noisy(self, memory):
        """Test retrieving noisy pattern (cleanup)."""
        # Store clean pattern
        pattern = torch.randn(1000)
        pattern = pattern / torch.norm(pattern)
        memory.store(pattern)

        # Add noise
        noise = 0.2 * torch.randn(1000)
        noisy = pattern + noise
        noisy = noisy / torch.norm(noisy)

        # Retrieve should cleanup noise
        retrieved = memory.retrieve(noisy)
        retrieved = retrieved / torch.norm(retrieved)

        # Should be more similar to original than noisy was
        sim_original = (pattern @ retrieved).item()
        sim_noisy = (pattern @ noisy).item()

        assert sim_original > sim_noisy, (
            f"Retrieved similarity {sim_original} not better than noisy {sim_noisy}"
        )
        assert sim_original > 0.9, f"Retrieved similarity {sim_original} < 0.9"

    def test_retrieve_multiple_patterns(self, memory):
        """Test retrieving one of multiple stored patterns."""
        # Store 3 distinct patterns
        torch.manual_seed(42)
        patterns = [torch.randn(1000) / 10 for _ in range(3)]
        for p in patterns:
            memory.store(p / torch.norm(p))

        # Query with pattern 1 + noise
        noise = 0.1 * torch.randn(1000)
        query = patterns[1] + noise
        query = query / torch.norm(query)

        # Should retrieve pattern 1
        retrieved = memory.retrieve(query)
        retrieved = retrieved / torch.norm(retrieved)

        # Check similarity to pattern 1
        sim = (patterns[1] / torch.norm(patterns[1]) @ retrieved).item()
        assert sim > 0.8, f"Similarity to pattern 1: {sim} < 0.8"

    def test_batch_retrieve(self, memory_small):
        """Test batch retrieval."""
        # Store patterns
        patterns = torch.randn(3, 100)
        memory_small.store_batch(patterns)

        # Create noisy versions
        noise = 0.1 * torch.randn(3, 100)
        queries = patterns + noise

        # Batch retrieve
        retrieved = memory_small.retrieve_batch(queries)

        assert retrieved.shape == (3, 100)

    def test_max_patterns_eviction(self):
        """Test that oldest patterns are evicted when exceeding capacity."""
        memory = ModernHopfieldMemory(dimension=100, max_patterns=5, device="cpu")

        # Store 10 patterns
        for i in range(10):
            pattern = torch.randn(100)
            memory.store(pattern)

        # Should only have 5 (most recent)
        assert len(memory) == 5

    def test_complex_patterns(self):
        """Test with complex hypervectors (FHRR)."""
        memory = ModernHopfieldMemory(dimension=500, device="cpu")

        # Store complex patterns
        pattern = torch.randn(500, dtype=torch.cfloat)
        pattern = pattern / torch.abs(pattern).clamp(min=1e-8)
        memory.store(pattern)

        # Retrieve with noise
        noise = 0.1 * torch.randn(500, dtype=torch.cfloat)
        query = pattern + noise
        query = query / torch.abs(query).clamp(min=1e-8)

        retrieved = memory.retrieve(query)

        # Check similarity (real part of inner product)
        similarity = (pattern * retrieved.conj()).real.mean()
        assert similarity > 0.8, f"Similarity {similarity} < 0.8"

    def test_capacity_calculation(self, memory):
        """Test exponential capacity calculation."""
        capacity = memory.capacity()

        # For dimension 1000, capacity should be ~2^500 (huge)
        assert capacity > 1e10, f"Capacity {capacity} unexpectedly low"

    def test_clear(self, memory_small):
        """Test clearing all patterns."""
        patterns = torch.randn(5, 100)
        memory_small.store_batch(patterns)

        assert len(memory_small) == 5

        memory_small.clear()

        assert len(memory_small) == 0
        assert memory_small.patterns.shape == (0, 100)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory(self):
        """Test Hopfield memory on GPU."""
        memory = ModernHopfieldMemory(dimension=1000, device="cuda")

        pattern = torch.randn(1000, device="cuda")
        memory.store(pattern)

        retrieved = memory.retrieve(pattern)

        assert retrieved.device.type == "cuda"

    def test_one_step_convergence(self, memory):
        """Test that modern Hopfield converges in 1 step for clean patterns."""
        pattern = torch.randn(1000)
        memory.store(pattern)

        # Small noise
        query = pattern + 0.05 * torch.randn(1000)

        # Retrieve with 1 iteration
        retrieved_1 = memory.retrieve(query, num_iterations=1)
        # Retrieve with 5 iterations
        retrieved_5 = memory.retrieve(query, num_iterations=5)

        # Should be almost identical (modern Hopfield converges in 1 step)
        diff = torch.norm(retrieved_1 - retrieved_5)
        assert diff < 0.1, f"Difference after 1 vs 5 iterations: {diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
