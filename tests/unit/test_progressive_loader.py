"""Tests for Progressive Dynamic Selective Loading."""

import pytest
import torch
import torch.nn as nn

from cogsyndelta.core.progressive_loader import (
    LoadingState,
    ProgressiveLoadingConfig,
    ProgressiveLoaderManager,
    SubmodelInfo,
)


class DummySubmodel(nn.Module):
    """Dummy submodel for testing."""

    def __init__(self, size_mb: float = 100):
        super().__init__()
        # Create dummy parameters to simulate size
        num_params = int(size_mb * 1024 * 1024 / 4)  # 4 bytes per FP32 param
        self.weight = nn.Parameter(torch.randn(num_params))
        self._size_mb = size_mb

    def forward(self, x):
        return x * self.weight[0]  # Dummy operation


class TestProgressiveLoadingConfig:
    """Test suite for ProgressiveLoadingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProgressiveLoadingConfig()

        assert config.max_active_submodels == 8
        assert config.gpu_budget_mb == 10000
        assert config.eager_load_threshold == 0.7
        assert config.lazy_unload_threshold == 0.2

    def test_custom_config(self):
        """Test custom configuration."""
        config = ProgressiveLoadingConfig(
            max_active_submodels=5,
            gpu_budget_mb=5000,
            eager_load_threshold=0.8,
            lazy_unload_threshold=0.1,
        )

        assert config.max_active_submodels == 5
        assert config.gpu_budget_mb == 5000
        assert config.eager_load_threshold == 0.8
        assert config.lazy_unload_threshold == 0.1

    def test_validation_thresholds(self):
        """Test that invalid thresholds raise errors."""
        # Eager threshold must be > lazy threshold
        with pytest.raises(AssertionError):
            ProgressiveLoadingConfig(eager_load_threshold=0.2, lazy_unload_threshold=0.8)


class TestSubmodelInfo:
    """Test suite for SubmodelInfo."""

    def test_initialization(self):
        """Test SubmodelInfo initialization."""
        module = DummySubmodel(size_mb=200)

        info = SubmodelInfo(
            name="test_module",
            module=module,
            size_mb=200,
            submodel_type="vision",
            state=LoadingState.DISK,
            pinned=False,
        )

        assert info.name == "test_module"
        assert info.size_mb == 200
        assert info.submodel_type == "vision"
        assert info.state == LoadingState.DISK
        assert not info.pinned


class TestProgressiveLoaderManager:
    """Test suite for ProgressiveLoaderManager."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProgressiveLoadingConfig(
            max_active_submodels=3,
            gpu_budget_mb=500,
            eager_load_threshold=0.7,
            lazy_unload_threshold=0.3,
            async_loading=False,  # Synchronous for testing
        )

    @pytest.fixture
    def loader(self, config):
        """Create loader manager."""
        return ProgressiveLoaderManager(config=config, device="cpu")

    def test_initialization(self, loader):
        """Test loader initialization."""
        assert loader.config.max_active_submodels == 3
        assert len(loader.submodels) == 0
        assert loader.gpu_memory_used == 0

    def test_register_submodel(self, loader):
        """Test registering a submodel."""
        module = DummySubmodel(size_mb=100)

        loader.register_submodel(
            name="vision_0", module=module, size_mb=100, submodel_type="vision"
        )

        assert "vision_0" in loader.submodels
        assert loader.submodels["vision_0"].size_mb == 100
        assert loader.submodels["vision_0"].state == LoadingState.DISK

    def test_activate_submodel(self, loader):
        """Test activating a submodel."""
        module = DummySubmodel(size_mb=100)
        loader.register_submodel("vision_0", module, size_mb=100)

        # Activate
        loader.activate("vision_0", blocking=True)

        assert loader.is_loaded("vision_0")
        assert "vision_0" in loader.active_names
        assert loader.gpu_memory_used == pytest.approx(100, rel=0.1)

    def test_deactivate_submodel(self, loader):
        """Test deactivating a submodel."""
        module = DummySubmodel(size_mb=100)
        loader.register_submodel("vision_0", module, size_mb=100)

        # Activate then deactivate
        loader.activate("vision_0", blocking=True)
        assert loader.is_loaded("vision_0")

        loader.deactivate("vision_0")
        assert not loader.is_loaded("vision_0")
        assert "vision_0" not in loader.active_names

    def test_max_active_limit(self, loader):
        """Test that max_active_submodels is enforced."""
        # Register 5 submodels (limit is 3)
        for i in range(5):
            module = DummySubmodel(size_mb=100)
            loader.register_submodel(f"module_{i}", module, size_mb=100)

        # Activate all 5
        for i in range(5):
            loader.activate(f"module_{i}", blocking=True)

        # Should only have 3 active (most recent)
        assert len(loader.active_names) <= 3

    def test_pinned_submodels_not_evicted(self, loader):
        """Test that pinned submodels are not evicted."""
        # Register pinned submodel
        module_pinned = DummySubmodel(size_mb=100)
        loader.register_submodel("core", module_pinned, size_mb=100)
        loader.pin_submodel("core")
        loader.activate("core", blocking=True)

        # Register and activate many more to trigger eviction
        for i in range(10):
            module = DummySubmodel(size_mb=100)
            loader.register_submodel(f"module_{i}", module, size_mb=100)
            loader.activate(f"module_{i}", blocking=True)

        # Core should still be active (pinned)
        assert loader.is_loaded("core")

    def test_get_loaded(self, loader):
        """Test retrieving loaded module."""
        module = DummySubmodel(size_mb=100)
        loader.register_submodel("vision_0", module, size_mb=100)

        # Not loaded yet
        assert loader.get_loaded("vision_0") is None

        # Activate
        loader.activate("vision_0", blocking=True)

        # Now should return module
        loaded = loader.get_loaded("vision_0")
        assert loaded is not None
        assert isinstance(loaded, DummySubmodel)

    def test_update_from_routing(self, loader):
        """Test updating from interconnect routing decisions."""
        # Register submodels
        for i in range(5):
            module = DummySubmodel(size_mb=100)
            loader.register_submodel(f"module_{i}", module, size_mb=100)

        # Simulate routing decisions
        routing_decisions = {
            ("section_0", "section_1"): 0.9,  # High importance
            ("section_0", "section_2"): 0.5,  # Medium
            ("section_0", "section_3"): 0.1,  # Low
        }

        section_to_submodel = {
            "section_0": "module_0",
            "section_1": "module_1",
            "section_2": "module_2",
            "section_3": "module_3",
        }

        loader.update_from_routing(routing_decisions, section_to_submodel)

        # High importance should be activated (>= 0.7)
        # Note: activation might be async, but with our config it's synchronous
        # module_1 has importance 0.9
        assert loader.is_loaded("module_1") or loader.is_loading("module_1")

    def test_prefetch_queue(self, loader):
        """Test prefetch queue management."""
        module = DummySubmodel(size_mb=100)
        loader.register_submodel("module_0", module, size_mb=100)

        # Add to prefetch queue
        loader.prefetch_queue.put("module_0")

        # Queue should have item
        assert not loader.prefetch_queue.empty()

    def test_get_stats(self, loader):
        """Test statistics retrieval."""
        module = DummySubmodel(size_mb=100)
        loader.register_submodel("vision_0", module, size_mb=100)
        loader.activate("vision_0", blocking=True)

        stats = loader.get_stats()

        assert "gpu_memory_mb" in stats
        assert "num_loaded" in stats
        assert "max_active" in stats
        assert stats["max_active"] == 3
        assert stats["num_loaded"] >= 1

    def test_clear(self, loader):
        """Test clearing all submodels."""
        for i in range(3):
            module = DummySubmodel(size_mb=100)
            loader.register_submodel(f"module_{i}", module, size_mb=100)
            loader.activate(f"module_{i}", blocking=True)

        # Clear
        loader.clear()

        assert len(loader.active_names) == 0
        assert loader.gpu_memory_used == 0

    def test_memory_budget_enforcement(self, loader):
        """Test that GPU memory budget is enforced."""
        # Config has 500MB budget
        # Try to load 6×100MB = 600MB
        for i in range(6):
            module = DummySubmodel(size_mb=100)
            loader.register_submodel(f"module_{i}", module, size_mb=100)
            loader.activate(f"module_{i}", blocking=True)

        # Should not exceed budget
        assert loader.gpu_memory_used <= loader.config.gpu_budget_mb

    def test_submodel_type_tracking(self, loader):
        """Test that submodel types are tracked."""
        vision_module = DummySubmodel(size_mb=100)
        language_module = DummySubmodel(size_mb=100)

        loader.register_submodel("vision_0", vision_module, size_mb=100, submodel_type="vision")
        loader.register_submodel(
            "language_0", language_module, size_mb=100, submodel_type="language"
        )

        assert loader.submodels["vision_0"].submodel_type == "vision"
        assert loader.submodels["language_0"].submodel_type == "language"


class TestLoadingState:
    """Test LoadingState enum."""

    def test_states(self):
        """Test that all expected states exist."""
        assert hasattr(LoadingState, "DISK")
        assert hasattr(LoadingState, "LOADING")
        assert hasattr(LoadingState, "LOADED")
        assert hasattr(LoadingState, "STAGED")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
