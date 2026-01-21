"""Tests for Fractional Power Encoding (FPE)."""

import pytest
import torch

from vsa.config import VSAConfig
from vsa.encoders.fpe import FractionalPowerEncoder


class TestFractionalPowerEncoding:
    """Test suite for FPE temporal encoding."""

    @pytest.fixture
    def encoder(self):
        """Create FPE encoder on CPU for testing."""
        config = VSAConfig(dimension=10000, model="FHRR", device="cpu")
        return FractionalPowerEncoder(config)

    @pytest.fixture
    def encoder_small(self):
        """Create small FPE encoder for fast tests."""
        config = VSAConfig(dimension=1000, model="FHRR", device="cpu")
        return FractionalPowerEncoder(config)

    def test_initialization(self, encoder):
        """Test FPE encoder initialization."""
        assert encoder.config.dimension == 10000
        assert encoder.config.model == "FHRR"
        assert encoder.base_vector.shape == (1, 10000)
        assert encoder.base_vector.dtype == torch.cfloat

    def test_random_vector_generation(self, encoder):
        """Test random hypervector generation."""
        vec = encoder.random_vector()
        assert vec.shape == (10000,)
        assert vec.dtype == torch.cfloat
        # Check that it's approximately normalized (unit magnitude phasors)
        magnitudes = torch.abs(vec)
        assert magnitudes.mean().item() == pytest.approx(1.0, rel=0.1)

    def test_single_encoding(self, encoder_small):
        """Test encoding a single timestamp."""
        base = encoder_small.random_vector()

        t0 = encoder_small.encode(base, 0.0)
        t1 = encoder_small.encode(base, 1.0)

        # t^0 = identity (should equal base)
        assert torch.allclose(t0, base, rtol=1e-5)

        # t^1 = t (should equal base)
        assert torch.allclose(t1, base, rtol=1e-5)

    def test_batch_encoding(self, encoder_small):
        """Test encoding a batch of timestamps."""
        batch_size = 10
        bases = torch.stack([encoder_small.random_vector() for _ in range(batch_size)])
        timestamps = torch.linspace(0, 10, batch_size)

        encoded = encoder_small.encode_batch(bases, timestamps)

        assert encoded.shape == (batch_size, 1000)
        assert encoded.dtype == torch.cfloat

    def test_sinc_decay_property(self, encoder):
        """THEORETICAL: Test that similarity decays as sinc(Δt)."""
        base = encoder.random_vector()

        # Test various time offsets
        t_values = [0.0, 0.5, 1.0, 2.0, 5.0]
        results = encoder.verify_sinc_decay(base, t_values, tolerance=0.05)

        # All values should be within tolerance
        for t, (measured, expected, within_tol) in results.items():
            assert within_tol, (
                f"At t={t}: measured={measured:.4f}, expected={expected:.4f}, "
                f"diff={abs(measured - expected):.4f} > 0.05"
            )

    def test_temporal_locality(self, encoder_small):
        """Test that nearby timestamps have higher similarity."""
        base = encoder_small.random_vector()

        t0 = encoder_small.encode(base, 0.0)
        t1 = encoder_small.encode(base, 0.1)
        t5 = encoder_small.encode(base, 5.0)

        # Nearby times should be more similar
        sim_near = encoder_small.cosine_similarity(t0, t1).item()
        sim_far = encoder_small.cosine_similarity(t0, t5).item()

        assert sim_near > sim_far, (
            f"Near similarity {sim_near} should be > far similarity {sim_far}"
        )

    def test_encode_temporal_api(self, encoder_small):
        """Test the encode_temporal API from technical spec."""
        base = encoder_small.random_vector()
        timestamps = torch.tensor([0.0, 1.0, 2.0, 3.0])

        # Single base, multiple timestamps
        encoded = encoder_small.encode_temporal(base, timestamps)
        assert encoded.shape == (4, 1000)

        # Batch of bases
        bases = base.unsqueeze(0).expand(4, -1)
        encoded_batch = encoder_small.encode_temporal(bases, timestamps)
        assert encoded_batch.shape == (4, 1000)

    def test_capacity_bound_calculation(self, encoder):
        """Test capacity bound formula."""
        # Example: Store 1000 items at 0.95 similarity threshold with 10000 time values
        required_dim = encoder.capacity_bound(k=1000, S=0.95, M=10000)

        # Should be reasonable (not too large)
        assert required_dim > 0
        assert required_dim < 1000000  # Sanity check

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_encoding(self):
        """Test GPU encoding performance."""
        config = VSAConfig(dimension=10000, model="FHRR", device="cuda")
        encoder = FractionalPowerEncoder(config)

        base = encoder.random_vector()
        timestamps = torch.linspace(0, 100, 1000, device="cuda")

        # Warm-up
        for _ in range(10):
            encoder.encode_batch(base.expand(100, -1), timestamps[:100])

        torch.cuda.synchronize()
        import time

        start = time.perf_counter()

        for _ in range(100):
            encoder.encode_batch(base.expand(100, -1), timestamps[:100])

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        throughput = 100 * 100 / elapsed  # vectors/second
        # Log throughput for informational purposes, don't assert on hardware-specific values
        print(f"FPE throughput: {throughput:.0f} vecs/sec")
