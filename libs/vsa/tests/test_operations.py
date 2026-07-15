"""Tests for VSA operations (binding, bundling, permutation)."""

import pytest
import torch

from vsa.operations.binding import bind, unbind
from vsa.operations.bundling import bundle, weighted_bundle
from vsa.operations.permutation import create_sequence_encoding, inverse_permute, permute


class TestBinding:
    """Test binding operations."""

    @pytest.fixture
    def complex_vectors(self):
        """Create complex hypervectors for testing."""
        torch.manual_seed(42)
        x = torch.randn(1000, dtype=torch.cfloat)
        x = x / torch.abs(x).clamp(min=1e-8)  # Normalize
        y = torch.randn(1000, dtype=torch.cfloat)
        y = y / torch.abs(y).clamp(min=1e-8)
        return x, y

    def test_bind_unbind_invertibility(self, complex_vectors):
        """Test that unbind reverses bind."""
        x, y = complex_vectors

        # Bind x and y
        z = bind(x, y)

        # Unbind to recover x
        x_recovered = unbind(z, y)

        # Similarity should be high (>0.9 for FHRR)
        similarity = (x * x_recovered.conj()).real.mean()
        assert similarity > 0.9, f"Similarity {similarity} < 0.9"

    def test_bind_dissimilarity(self, complex_vectors):
        """Test that bound result is dissimilar to inputs."""
        x, y = complex_vectors
        z = bind(x, y)

        # z should be dissimilar to both x and y
        sim_x = (z * x.conj()).real.mean()
        sim_y = (z * y.conj()).real.mean()

        assert abs(sim_x) < 0.3, f"z too similar to x: {sim_x}"
        assert abs(sim_y) < 0.3, f"z too similar to y: {sim_y}"

    def test_bind_commutativity(self, complex_vectors):
        """Test that bind(x, y) = bind(y, x)."""
        x, y = complex_vectors

        xy = bind(x, y)
        yx = bind(y, x)

        # Should be identical
        assert torch.allclose(xy, yx, rtol=1e-5)

    def test_bind_batch(self):
        """Test batch binding."""
        torch.manual_seed(42)
        batch_size = 10
        dim = 500

        x = torch.randn(batch_size, dim, dtype=torch.cfloat)
        y = torch.randn(batch_size, dim, dtype=torch.cfloat)

        z = bind(x, y)

        assert z.shape == (batch_size, dim)


class TestBundling:
    """Test bundling operations."""

    @pytest.fixture
    def complex_vectors(self):
        """Create complex hypervectors for testing."""
        torch.manual_seed(42)
        vectors = [torch.randn(1000, dtype=torch.cfloat) for _ in range(5)]
        # Normalize
        vectors = [v / torch.abs(v).clamp(min=1e-8) for v in vectors]
        return vectors

    def test_bundle_similarity(self, complex_vectors):
        """Test that bundle is similar to all inputs."""
        vectors = torch.stack(complex_vectors)
        superposition = bundle(vectors)

        # Superposition should be similar to all inputs
        for vec in complex_vectors:
            similarity = (superposition * vec.conj()).real.mean()
            assert similarity > 0, f"Similarity {similarity} should be positive"

    def test_bundle_commutativity(self, complex_vectors):
        """Test that bundle order doesn't matter."""
        import random

        # Bundle in two different orders
        order1 = complex_vectors[:]
        order2 = complex_vectors[:]
        random.shuffle(order2)

        bundle1 = bundle(torch.stack(order1))
        bundle2 = bundle(torch.stack(order2))

        # Should be very similar (may have small numerical differences)
        similarity = (bundle1 * bundle2.conj()).real.mean()
        assert similarity > 0.99, f"Bundles not commutative: similarity {similarity}"

    def test_weighted_bundle(self):
        """Test weighted bundling."""
        torch.manual_seed(42)
        vectors = torch.stack([torch.randn(500, dtype=torch.cfloat) for _ in range(3)])
        weights = torch.tensor([0.6, 0.3, 0.1])  # First vector most important

        weighted = weighted_bundle(vectors, weights)

        # Weighted bundle should be most similar to first vector
        sims = [(weighted * v.conj()).real.mean() for v in vectors]
        assert sims[0] > sims[1] > sims[2], f"Weights not respected: {sims}"


class TestPermutation:
    """Test permutation operations."""

    @pytest.fixture
    def complex_vector(self):
        """Create a complex hypervector."""
        torch.manual_seed(42)
        vec = torch.randn(1000, dtype=torch.cfloat)
        return vec / torch.abs(vec).clamp(min=1e-8)

    def test_permute_inverse(self, complex_vector):
        """Test that inverse_permute reverses permute."""
        shifted = permute(complex_vector, shifts=5)
        recovered = inverse_permute(shifted, shifts=5)

        # Should recover original
        assert torch.allclose(recovered, complex_vector, rtol=1e-4)

    def test_permute_dissimilarity(self, complex_vector):
        """Test that permutation changes similarity."""
        shifted = permute(complex_vector, shifts=1)

        # Should be dissimilar
        similarity = (complex_vector * shifted.conj()).real.mean()
        assert abs(similarity) < 0.5, f"Permutation too similar: {similarity}"

    def test_sequence_encoding(self):
        """Test sequence encoding preserves order."""
        torch.manual_seed(42)
        # Create three distinct "word" vectors
        words = torch.stack(
            [
                torch.randn(500, dtype=torch.cfloat),
                torch.randn(500, dtype=torch.cfloat),
                torch.randn(500, dtype=torch.cfloat),
            ]
        )

        # Encode sequences "abc" and "cba"
        seq_abc = create_sequence_encoding(words)
        seq_cba = create_sequence_encoding(torch.flip(words, [0]))

        # Different orders should be dissimilar
        similarity = (seq_abc * seq_cba.conj()).real.mean()
        assert abs(similarity) < 0.7, f"Sequences too similar: {similarity}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
