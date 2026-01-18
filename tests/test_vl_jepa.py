"""Unit tests for VL-JEPA components.

Tests the VisionEncoder, TemporalMemoryBank, ModeratedHyperConnection,
JointEmbeddingSpace, and HierarchicalPredictiveCoding components.

Per constitution: All new features require tests.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class TestVisionEncoder:
    """Tests for VisionEncoder component."""

    @pytest.fixture
    def encoder(self) -> nn.Module:
        """Create VisionEncoder instance."""
        from cogsyndelta.core.vl_jepa_extension import VisionEncoder

        return VisionEncoder(
            image_size=224,
            patch_size=16,
            in_channels=3,
            embed_dim=512,
            num_layers=2,  # Small for testing
        ).cuda()

    def test_forward_single_image(self, encoder: nn.Module) -> None:
        """Test forward pass with single image."""
        image = torch.randn(1, 3, 224, 224, device="cuda")
        output = encoder(image)
        assert output.shape == (1, 512)

    def test_forward_batch(self, encoder: nn.Module) -> None:
        """Test forward pass with batch of images."""
        images = torch.randn(4, 3, 224, 224, device="cuda")
        output = encoder(images)
        assert output.shape == (4, 512)

    def test_output_reasonable_norm(self, encoder: nn.Module) -> None:
        """Test that output has reasonable norm (not exploding/vanishing)."""
        image = torch.randn(1, 3, 224, 224, device="cuda")
        output = encoder(image)

        # Output should have reasonable norm
        norm = output.norm()
        assert norm > 0.1, f"Output norm too small: {norm}"
        assert norm < 1000, f"Output norm too large: {norm}"

    def test_different_image_sizes_fail(self, encoder: nn.Module) -> None:
        """Test that incorrect image sizes raise error."""
        wrong_size = torch.randn(1, 3, 128, 128, device="cuda")
        with pytest.raises((RuntimeError, ValueError)):
            encoder(wrong_size)


class TestTemporalMemoryBank:
    """Tests for TemporalMemoryBank component."""

    @pytest.fixture
    def memory_bank(self) -> nn.Module:
        """Create TemporalMemoryBank instance."""
        from cogsyndelta.core.vl_jepa_extension import TemporalMemoryBank

        return TemporalMemoryBank(
            embed_dim=512,
            memory_size=100,
            num_read_heads=4,
        ).cuda()

    def test_write_single(self, memory_bank: nn.Module) -> None:
        """Test writing single embedding to memory."""
        value = torch.randn(1, 512, device="cuda")
        memory_bank.write(value)

        # Memory should have one entry (write_pointer advances)
        assert memory_bank.write_pointer.item() == 1

    def test_write_batch(self, memory_bank: nn.Module) -> None:
        """Test writing batch of embeddings."""
        values = torch.randn(5, 512, device="cuda")
        memory_bank.write(values)

        # write_pointer should advance by batch size
        assert memory_bank.write_pointer.item() == 5

    def test_read_returns_correct_shape(self, memory_bank: nn.Module) -> None:
        """Test that read returns correct shape."""
        # Write some values first
        values = torch.randn(10, 512, device="cuda")
        memory_bank.write(values)

        # Read with query
        query = torch.randn(1, 512, device="cuda")
        result = memory_bank.read(query)

        assert result.shape == (1, 512), f"Expected (1, 512), got {result.shape}"

    def test_read_empty_memory(self, memory_bank: nn.Module) -> None:
        """Test reading from empty memory returns zeros or handles gracefully."""
        query = torch.randn(1, 512, device="cuda")
        result = memory_bank.read(query)

        # Should handle empty memory gracefully
        assert result.shape == (1, 512)

    def test_memory_wrapping(self, memory_bank: nn.Module) -> None:
        """Test that memory wraps when capacity exceeded."""
        # Write more than capacity
        for _ in range(15):  # 150 writes > 100 capacity
            values = torch.randn(10, 512, device="cuda")
            memory_bank.write(values)

        # Memory should wrap around (write_pointer is modulo memory_size)
        assert memory_bank.write_pointer.item() <= memory_bank.memory_size


class TestModeratedHyperConnection:
    """Tests for ModeratedHyperConnection (mHC) component."""

    @pytest.fixture
    def mhc(self) -> nn.Module:
        """Create ModeratedHyperConnection instance."""
        from cogsyndelta.core.vl_jepa_extension import ModeratedHyperConnection

        return ModeratedHyperConnection(embed_dim=512).cuda()

    def test_forward_shape(self, mhc: nn.Module) -> None:
        """Test that mHC output has correct shape."""
        source = torch.randn(4, 512, device="cuda")
        target = torch.randn(4, 512, device="cuda")

        output = mhc(source, target)

        assert output.shape == (4, 512), f"Expected (4, 512), got {output.shape}"

    def test_gate_moderation(self, mhc: nn.Module) -> None:
        """Test that gate moderates information flow."""
        source = torch.ones(1, 512, device="cuda")
        target = torch.zeros(1, 512, device="cuda")

        output = mhc(source, target)

        # Output should be somewhere between source and target
        # due to moderation (alpha * gated_source + (1-alpha) * target)
        assert not torch.allclose(output, source)
        assert not torch.allclose(output, target)

    def test_alpha_parameter(self) -> None:
        """Test alpha residual scaling parameter exists."""
        from cogsyndelta.core.vl_jepa_extension import ModeratedHyperConnection

        mhc = ModeratedHyperConnection(embed_dim=512).cuda()
        assert hasattr(mhc, "alpha")
        # Alpha should be learnable and start at 0.5
        assert 0 <= mhc.alpha.item() <= 1


class TestJointEmbeddingSpace:
    """Tests for JointEmbeddingSpace component."""

    @pytest.fixture
    def joint_embed(self) -> nn.Module:
        """Create JointEmbeddingSpace instance."""
        from cogsyndelta.core.vl_jepa_extension import JointEmbeddingSpace

        return JointEmbeddingSpace(
            embed_dim=512,
            latent_dim=20,
        ).cuda()

    def test_forward_shape(self, joint_embed: nn.Module) -> None:
        """Test joint embedding output shape."""
        vision = torch.randn(2, 512, device="cuda")
        state = torch.randn(2, 20, device="cuda")  # latent_dim=20

        result = joint_embed(vision, state)

        # Returns dict with joint_embedding and similarity
        assert "joint_embedding" in result
        assert result["joint_embedding"].shape == (2, 512)

    def test_similarity_computed(self, joint_embed: nn.Module) -> None:
        """Test that similarity matrix is computed correctly."""
        vision = torch.randn(4, 512, device="cuda")
        state = torch.randn(4, 20, device="cuda")

        result = joint_embed(vision, state)

        assert "similarity" in result
        assert result["similarity"].shape == (4, 4)  # batch x batch


class TestHierarchicalPredictiveCoding:
    """Tests for HierarchicalPredictiveCoding with mHC connections."""

    @pytest.fixture
    def hpc(self) -> nn.Module:
        """Create HierarchicalPredictiveCoding instance."""
        from cogsyndelta.core.vl_jepa_extension import HierarchicalPredictiveCoding

        return HierarchicalPredictiveCoding(
            embed_dim=512,
            num_levels=3,
        ).cuda()

    def test_forward_shape(self, hpc: nn.Module) -> None:
        """Test HPC forward pass shape."""
        x = torch.randn(2, 512, device="cuda")
        output = hpc(x)

        assert output.shape == (2, 512), f"Expected (2, 512), got {output.shape}"

    def test_multi_level_processing(self, hpc: nn.Module) -> None:
        """Test that multiple levels are processed."""
        x = torch.randn(2, 512, device="cuda")

        # The HPC should process through all levels
        output = hpc(x)

        # Output should be different from input due to processing
        assert not torch.allclose(output, x, atol=1e-3)


class TestVLJEPAIntegration:
    """Integration tests for full VL-JEPA pipeline."""

    def test_full_pipeline(self) -> None:
        """Test complete VL-JEPA forward pass."""
        from cogsyndelta.core.vl_jepa_extension import (
            HierarchicalPredictiveCoding,
            ModeratedHyperConnection,
            TemporalMemoryBank,
            VisionEncoder,
        )

        # Create components
        vision_encoder = VisionEncoder(embed_dim=512, num_layers=2).cuda()
        memory_bank = TemporalMemoryBank(embed_dim=512, memory_size=50).cuda()
        mhc = ModeratedHyperConnection(embed_dim=512).cuda()
        hpc = HierarchicalPredictiveCoding(embed_dim=512, num_levels=2).cuda()

        # Process image
        image = torch.randn(1, 3, 224, 224, device="cuda")
        vision_embed = vision_encoder(image)

        # Store in memory
        memory_bank.write(vision_embed)

        # Retrieve from memory
        retrieved = memory_bank.read(vision_embed)

        # Apply mHC connection
        connected = mhc(retrieved, vision_embed)

        # Process through HPC
        result = hpc(connected)

        assert result["final_state"].shape == (1, 512)

    def test_temporal_context_window(self) -> None:
        """Test that temporal context accumulates correctly."""
        from cogsyndelta.core.vl_jepa_extension import TemporalMemoryBank

        memory_bank = TemporalMemoryBank(embed_dim=512, memory_size=100).cuda()

        # Write sequence of embeddings
        for i in range(10):
            embedding = torch.randn(1, 512, device="cuda") * (i + 1)
            memory_bank.write(embedding)

        # Query should retrieve relevant context
        query = torch.randn(1, 512, device="cuda")
        result = memory_bank.read(query)

        # Result should incorporate memory context
        assert result.shape == (1, 512)
        assert memory_bank.write_pointer.item() == 10
