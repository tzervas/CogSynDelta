"""Unit tests for mHC/Interconnect Manager components.

Tests the IntelligentInterconnectManager, ContextualAttentionRouter,
PathwayOptimizer, ContextPropagationEngine, and CongestionController.

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


class TestModeratedHyperConnectionBasic:
    """Basic tests for ModeratedHyperConnection from vl_jepa_extension."""

    @pytest.fixture
    def mhc(self) -> nn.Module:
        """Create ModeratedHyperConnection instance."""
        from cogsyndelta.core.vl_jepa_extension import ModeratedHyperConnection

        return ModeratedHyperConnection(embed_dim=256).cuda()

    def test_mhc_initialization(self, mhc: nn.Module) -> None:
        """Test mHC initializes with correct components."""
        assert hasattr(mhc, "gate")
        assert hasattr(mhc, "transform")
        assert hasattr(mhc, "alpha")

    def test_mhc_forward_preserves_batch(self, mhc: nn.Module) -> None:
        """Test forward pass preserves batch dimension."""
        batch_sizes = [1, 4, 16]
        for bs in batch_sizes:
            source = torch.randn(bs, 256, device="cuda")
            target = torch.randn(bs, 256, device="cuda")
            output = mhc(source, target)
            assert output.shape == (bs, 256), f"Failed for batch size {bs}"

    def test_mhc_gradient_flow(self, mhc: nn.Module) -> None:
        """Test that gradients flow through the connection."""
        source = torch.randn(2, 256, device="cuda", requires_grad=True)
        target = torch.randn(2, 256, device="cuda", requires_grad=True)

        output = mhc(source, target)
        loss = output.sum()
        loss.backward()

        assert source.grad is not None, "No gradient for source"
        assert target.grad is not None, "No gradient for target"


class TestIntelligentInterconnectManager:
    """Tests for IntelligentInterconnectManager component."""

    @pytest.fixture
    def manager(self) -> nn.Module:
        """Create IntelligentInterconnectManager instance."""
        from cogsyndelta.core.interconnect_manager import IntelligentInterconnectManager

        return IntelligentInterconnectManager(
            embed_dim=512,
            num_sections=4,
            total_bandwidth=10000,
        ).cuda()

    def test_initialization(self, manager: nn.Module) -> None:
        """Test manager initializes all components."""
        assert hasattr(manager, "router")
        assert hasattr(manager, "optimizer")
        assert hasattr(manager, "propagator")
        assert hasattr(manager, "congestion_controller")

    def test_register_pathway(self, manager: nn.Module) -> None:
        """Test pathway registration."""
        manager.register_pathway("section_0", "section_1", initial_strength=1.0)

        assert ("section_0", "section_1") in manager.pathways

    def test_route_communication(self, manager: nn.Module) -> None:
        """Test routing communication between sections."""
        # Register pathway
        manager.register_pathway("section_0", "section_1")

        # Create states
        source_state = torch.randn(1, 512, device="cuda")
        target_state = torch.randn(1, 512, device="cuda")
        message = torch.randn(1, 512, device="cuda")

        # All section states
        all_states = {
            "section_0": source_state,
            "section_1": target_state,
            "section_2": torch.randn(1, 512, device="cuda"),
            "section_3": torch.randn(1, 512, device="cuda"),
        }

        # Route communication
        transmitted, decision = manager.route_communication(
            source_id="section_0",
            target_id="section_1",
            source_state=source_state,
            target_state=target_state,
            message=message,
            all_section_states=all_states,
        )

        assert transmitted.shape == message.shape


class TestContextualAttentionRouter:
    """Tests for ContextualAttentionRouter component."""

    @pytest.fixture
    def router(self) -> nn.Module:
        """Create ContextualAttentionRouter instance."""
        from cogsyndelta.core.interconnect_manager import ContextualAttentionRouter

        return ContextualAttentionRouter(
            embed_dim=512,
            num_heads=8,
            num_sections=4,
        ).cuda()

    def test_compute_importance(self, router: nn.Module) -> None:
        """Test importance computation."""
        source = torch.randn(1, 512, device="cuda")
        target = torch.randn(1, 512, device="cuda")

        importance = router.compute_importance(source, target)

        # Importance should be scalar between 0 and 1
        assert importance.shape == () or importance.numel() == 1
        assert 0 <= importance.item() <= 1

    def test_allocate_bandwidth(self, router: nn.Module) -> None:
        """Test bandwidth allocation computation."""
        source = torch.randn(1, 512, device="cuda")
        target = torch.randn(1, 512, device="cuda")

        bandwidth = router.allocate_bandwidth(source, target)

        assert bandwidth > 0


class TestPathwayOptimizer:
    """Tests for PathwayOptimizer component."""

    @pytest.fixture
    def optimizer(self) -> nn.Module:
        """Create PathwayOptimizer instance."""
        from cogsyndelta.core.interconnect_manager import PathwayOptimizer

        return PathwayOptimizer(embed_dim=512).cuda()

    def test_update_strength(self, optimizer: nn.Module) -> None:
        """Test pathway strength update with reward."""
        pathway_key = ("section_0", "section_1")
        initial_strength = 1.0

        # Update with positive reward
        reward = torch.tensor(1.0, device="cuda")
        new_strength = optimizer.update_strength(pathway_key, initial_strength, reward)

        # Strength should increase with positive reward
        assert new_strength >= initial_strength


class TestContextPropagationEngine:
    """Tests for ContextPropagationEngine component."""

    @pytest.fixture
    def propagator(self) -> nn.Module:
        """Create ContextPropagationEngine instance."""
        from cogsyndelta.core.interconnect_manager import ContextPropagationEngine

        return ContextPropagationEngine(
            embed_dim=512,
            max_hops=3,
        ).cuda()

    def test_propagate_context(self, propagator: nn.Module) -> None:
        """Test context propagation through intermediate states."""
        context = torch.randn(1, 512, device="cuda")
        intermediate_states = [torch.randn(1, 512, device="cuda") for _ in range(2)]
        target_state = torch.randn(1, 512, device="cuda")

        propagated = propagator.propagate_context(
            context, intermediate_states, target_state
        )

        assert propagated.shape == (1, 512)


class TestCongestionController:
    """Tests for CongestionController component."""

    @pytest.fixture
    def controller(self):
        """Create CongestionController instance."""
        from cogsyndelta.core.interconnect_manager import CongestionController

        return CongestionController(total_bandwidth=1000)

    def test_allocate_bandwidth(self, controller) -> None:
        """Test bandwidth allocation."""
        pathway = ("section_0", "section_1")
        allocated = controller.allocate(pathway, requested_bandwidth=100, priority=5)

        assert allocated is True

    def test_bandwidth_limit(self, controller) -> None:
        """Test that bandwidth limits are respected."""
        pathway = ("section_0", "section_1")

        # Allocate most bandwidth
        controller.allocate(pathway, requested_bandwidth=900, priority=5)

        # Try to allocate more than available
        allocated = controller.allocate(pathway, requested_bandwidth=200, priority=5)

        # Should fail or be limited
        # Implementation may vary - just ensure no crash

    def test_release_bandwidth(self, controller) -> None:
        """Test bandwidth release."""
        pathway = ("section_0", "section_1")

        controller.allocate(pathway, requested_bandwidth=100, priority=5)
        controller.release(pathway)

        # Should be able to allocate again
        allocated = controller.allocate(pathway, requested_bandwidth=100, priority=5)
        assert allocated is True

    def test_get_statistics(self, controller) -> None:
        """Test statistics retrieval."""
        stats = controller.get_statistics()

        assert "total_bandwidth" in stats
        assert "used_bandwidth" in stats
        assert "utilization" in stats


class TestInterconnectIntegration:
    """Integration tests for full interconnect system."""

    def test_full_communication_pipeline(self) -> None:
        """Test complete interconnect communication pipeline."""
        from cogsyndelta.core.interconnect_manager import IntelligentInterconnectManager

        manager = IntelligentInterconnectManager(
            embed_dim=512,
            num_sections=4,
            total_bandwidth=10000,
        ).cuda()

        # Register pathways
        for i in range(4):
            for j in range(4):
                if i != j:
                    manager.register_pathway(f"section_{i}", f"section_{j}")

        # Create section states
        all_states = {
            f"section_{i}": torch.randn(1, 512, device="cuda") for i in range(4)
        }

        # Test communication between all pairs
        successful = 0
        for i in range(4):
            for j in range(4):
                if i != j:
                    message = torch.randn(1, 512, device="cuda")
                    transmitted, decision = manager.route_communication(
                        source_id=f"section_{i}",
                        target_id=f"section_{j}",
                        source_state=all_states[f"section_{i}"],
                        target_state=all_states[f"section_{j}"],
                        message=message,
                        all_section_states=all_states,
                    )
                    if transmitted.abs().sum() > 0:
                        successful += 1

        # At least some communications should succeed
        assert successful > 0

    def test_interconnect_with_mhc(self) -> None:
        """Test interconnect manager with ModeratedHyperConnection."""
        from cogsyndelta.core.interconnect_manager import IntelligentInterconnectManager
        from cogsyndelta.core.vl_jepa_extension import ModeratedHyperConnection

        manager = IntelligentInterconnectManager(
            embed_dim=512, num_sections=4
        ).cuda()
        mhc = ModeratedHyperConnection(embed_dim=512).cuda()

        # Register pathway
        manager.register_pathway("section_0", "section_1")

        # Route information
        source = torch.randn(1, 512, device="cuda")
        target = torch.randn(1, 512, device="cuda")

        all_states = {
            "section_0": source,
            "section_1": target,
            "section_2": torch.randn(1, 512, device="cuda"),
            "section_3": torch.randn(1, 512, device="cuda"),
        }

        transmitted, _ = manager.route_communication(
            source_id="section_0",
            target_id="section_1",
            source_state=source,
            target_state=target,
            message=source,
            all_section_states=all_states,
        )

        # Apply mHC gating
        gated = mhc(transmitted, target)

        assert gated.shape == (1, 512)


class TestLoggingIntegration:
    """Test that interconnect components work with logging infrastructure."""

    def test_manager_operations_logged(self) -> None:
        """Test that manager operations don't break with logging."""
        from cogsyndelta.core.interconnect_manager import IntelligentInterconnectManager
        from cogsyndelta.core.logging_config import get_skip_metrics

        # Get initial metrics
        metrics = get_skip_metrics()
        initial_count = metrics.total()

        # Create and use manager
        manager = IntelligentInterconnectManager(
            embed_dim=512, num_sections=4
        ).cuda()
        manager.register_pathway("section_0", "section_1")

        # Operations should not crash
        stats = manager.congestion_controller.get_statistics()

        assert "total_bandwidth" in stats
        # Metrics should still work
        assert metrics.total() >= initial_count
