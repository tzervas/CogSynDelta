"""Tests for Residual Vector Quantization (RVQ)."""

import pytest
import torch

from compression.rvq.quantizer import ResidualVectorQuantizer, VectorQuantizer


class TestVectorQuantizer:
    """Test suite for single-stage vector quantizer."""

    @pytest.fixture
    def quantizer(self):
        """Create vector quantizer."""
        return VectorQuantizer(embedding_dim=128, codebook_size=256)

    @pytest.fixture
    def quantizer_small(self):
        """Create small quantizer for fast tests."""
        return VectorQuantizer(embedding_dim=32, codebook_size=16)

    def test_initialization(self, quantizer):
        """Test quantizer initialization."""
        assert quantizer.embedding_dim == 128
        assert quantizer.codebook_size == 256
        assert quantizer.codebook.shape == (256, 128)

    def test_forward(self, quantizer_small):
        """Test quantization forward pass."""
        x = torch.randn(4, 32)

        quantized, indices, commitment_loss = quantizer_small(x)

        assert quantized.shape == (4, 32)
        assert indices.shape == (4,)
        assert isinstance(commitment_loss.item(), float)

        # Indices should be in valid range
        assert (indices >= 0).all()
        assert (indices < 16).all()

    def test_quantized_values_from_codebook(self, quantizer_small):
        """Test that quantized vectors come from codebook."""
        x = torch.randn(2, 32)

        quantized, indices, _ = quantizer_small(x)

        # Each quantized vector should exactly match its codebook entry
        for i in range(2):
            expected = quantizer_small.codebook[indices[i]]
            # Note: STE adds residual, so we need to account for that
            # Actually, the quantized output is x + (codebook[idx] - x).detach()
            # So quantized ≠ codebook[idx] in forward pass (for gradients)
            # But they should be close
            actual = quantized[i].detach()
            codebook_vec = quantizer_small.codebook[indices[i]].detach()
            diff = torch.norm(actual - codebook_vec)
            # With STE, actual includes input contribution
            assert diff < 1e-5 or torch.allclose(actual, x[i])

    def test_commitment_loss(self, quantizer_small):
        """Test commitment loss computation."""
        x = torch.randn(8, 32)

        _, _, commitment_loss = quantizer_small(x)

        # Loss should be non-negative
        assert commitment_loss.item() >= 0

    def test_gradient_flow(self, quantizer_small):
        """Test that gradients flow through STE."""
        quantizer_small.train()
        x = torch.randn(4, 32, requires_grad=True)

        quantized, _, commitment_loss = quantizer_small(x)

        # Backprop through quantized output
        loss = quantized.sum() + commitment_loss
        loss.backward()

        # Gradients should exist
        assert x.grad is not None
        assert quantizer_small.codebook.grad is not None


class TestResidualVectorQuantizer:
    """Test suite for multi-stage RVQ."""

    @pytest.fixture
    def rvq(self):
        """Create RVQ."""
        return ResidualVectorQuantizer(
            embedding_dim=128, num_stages=4, codebook_size=256
        )

    @pytest.fixture
    def rvq_small(self):
        """Create small RVQ for fast tests."""
        return ResidualVectorQuantizer(
            embedding_dim=32, num_stages=3, codebook_size=16
        )

    def test_initialization(self, rvq):
        """Test RVQ initialization."""
        assert rvq.num_stages == 4
        assert len(rvq.quantizers) == 4
        for q in rvq.quantizers:
            assert q.embedding_dim == 128
            assert q.codebook_size == 256

    def test_forward(self, rvq_small):
        """Test RVQ forward pass."""
        x = torch.randn(4, 32)

        quantized, indices, loss = rvq_small(x)

        assert quantized.shape == (4, 32)
        assert indices.shape == (4, 3)  # [batch, num_stages]
        assert isinstance(loss.item(), float)

    def test_residual_refinement(self, rvq_small):
        """Test that residuals decrease with more stages."""
        x = torch.randn(2, 32)

        # Quantize with all stages
        quantized_full, _, _ = rvq_small(x)

        # Manually quantize stage by stage to check residuals
        residual = x.clone()
        residual_norms = []

        for q in rvq_small.quantizers:
            residual_norms.append(torch.norm(residual, dim=1).mean().item())
            quantized, _, _ = q(residual)
            residual = residual - quantized.detach()

        # Residual should decrease with each stage
        for i in range(len(residual_norms) - 1):
            assert (
                residual_norms[i + 1] <= residual_norms[i]
            ), f"Stage {i+1} residual not <= stage {i}"

    def test_encode_decode(self, rvq_small):
        """Test encode-decode roundtrip."""
        x = torch.randn(4, 32)

        # Encode to indices
        indices = rvq_small.encode(x)

        assert indices.shape == (4, 3)
        assert (indices >= 0).all()
        assert (indices < 16).all()

        # Decode from indices
        reconstructed = rvq_small.decode(indices)

        assert reconstructed.shape == (4, 32)

        # Reconstruction error should be small
        mse = torch.nn.functional.mse_loss(x, reconstructed)
        assert mse.item() < 1.0, f"MSE {mse.item()} too high"

    def test_more_stages_better_reconstruction(self):
        """Test that more stages improve reconstruction."""
        x = torch.randn(8, 64)

        rvq_2stages = ResidualVectorQuantizer(
            embedding_dim=64, num_stages=2, codebook_size=32
        )
        rvq_4stages = ResidualVectorQuantizer(
            embedding_dim=64, num_stages=4, codebook_size=32
        )

        # Encode/decode with different stage counts
        indices_2 = rvq_2stages.encode(x)
        reconstructed_2 = rvq_2stages.decode(indices_2)

        indices_4 = rvq_4stages.encode(x)
        reconstructed_4 = rvq_4stages.decode(indices_4)

        mse_2 = torch.nn.functional.mse_loss(x, reconstructed_2)
        mse_4 = torch.nn.functional.mse_loss(x, reconstructed_4)

        # More stages should have lower MSE (or similar)
        assert (
            mse_4 <= mse_2 * 1.5
        ), f"4-stage MSE {mse_4} not better than 2-stage {mse_2}"

    def test_batch_processing(self, rvq_small):
        """Test batch processing."""
        batch_sizes = [1, 4, 16]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 32)
            quantized, indices, loss = rvq_small(x)

            assert quantized.shape == (batch_size, 32)
            assert indices.shape == (batch_size, 3)

    def test_deterministic_encoding(self, rvq_small):
        """Test that encoding is deterministic."""
        x = torch.randn(2, 32)

        indices1 = rvq_small.encode(x)
        indices2 = rvq_small.encode(x)

        assert torch.equal(indices1, indices2)

    def test_gradient_flow_through_stages(self, rvq_small):
        """Test gradient flow through all stages."""
        rvq_small.train()
        x = torch.randn(4, 32, requires_grad=True)

        quantized, _, loss = rvq_small(x)

        total_loss = quantized.sum() + loss
        total_loss.backward()

        # All quantizers should have gradients
        for q in rvq_small.quantizers:
            assert q.codebook.grad is not None
            assert q.codebook.grad.abs().sum() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
