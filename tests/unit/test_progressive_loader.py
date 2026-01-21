"""Tests for Progressive Dynamic Selective Loading.

Tests the progressive loader module for dynamic submodel loading/unloading
to manage GPU memory efficiently for large models.
"""

import pytest
import torch
from torch import nn

from cogsyndelta.core.progressive_loader import (
    LoadingState,
    ProgressiveLoaderManager,
    ProgressiveLoadingConfig,
    SubmodelInfo,
)


class DummySubmodel(nn.Module):
    """Dummy submodel for testing.

    Creates a module with approximately the specified memory footprint
    using parameters of appropriate size.
    """

    def __init__(self, size_mb: float = 100):
        """Initialize dummy submodel.

        Args:
            size_mb: Target size in megabytes for the module parameters.
        """
        super().__init__()
        # Create dummy parameters to simulate size
        num_params = int(size_mb * 1024 * 1024 / 4)  # 4 bytes per FP32 param
        self.weight = nn.Parameter(torch.randn(num_params))
        self._size_mb = size_mb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - dummy operation.

        Args:
            x: Input tensor.

        Returns:
            Scaled input tensor.
        """
        return x * self.weight[0]


class TestProgressiveLoadingConfig:
    """Test suite for ProgressiveLoadingConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ProgressiveLoadingConfig()

        # Default is 10 on this branch
        assert config.max_active_submodels == 10
        assert config.gpu_budget_mb == 10000.0
        assert config.eager_load_threshold == 0.7
        assert config.lazy_unload_threshold == 0.2

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ProgressiveLoadingConfig(
            max_active_submodels=5,
            gpu_budget_mb=5000.0,
            eager_load_threshold=0.8,
            lazy_unload_threshold=0.1,
        )

        assert config.max_active_submodels == 5
        assert config.gpu_budget_mb == 5000.0
        assert config.eager_load_threshold == 0.8
        assert config.lazy_unload_threshold == 0.1

    def test_validation_thresholds(self) -> None:
        """Test that invalid thresholds raise errors.

        The eager_load_threshold must be greater than lazy_unload_threshold
        for the loading logic to work correctly.
        """
        # Eager threshold must be > lazy threshold
        with pytest.raises(AssertionError):
            ProgressiveLoadingConfig(eager_load_threshold=0.2, lazy_unload_threshold=0.8)


class TestLoadingState:
    """Test LoadingState enum."""

    def test_states_exist(self) -> None:
        """Test that all expected states exist in the enum."""
        assert hasattr(LoadingState, "DISK")
        assert hasattr(LoadingState, "LOADING")
        assert hasattr(LoadingState, "LOADED")
        assert hasattr(LoadingState, "STAGED")
        assert hasattr(LoadingState, "EVICTING")

    def test_state_values(self) -> None:
        """Test that state values are strings."""
        assert LoadingState.DISK.value == "disk"
        assert LoadingState.LOADED.value == "loaded"


class TestSubmodelInfo:
    """Test suite for SubmodelInfo dataclass."""

    def test_initialization(self) -> None:
        """Test SubmodelInfo initialization with required fields."""
        module = DummySubmodel(size_mb=1.0)
        param_count = sum(p.numel() for p in module.parameters())

        info = SubmodelInfo(
            name="test_module",
            module=module,
            parameter_count=param_count,
            size_mb=1.0,
            submodel_type="vision",
            state=LoadingState.DISK,
            pinned=False,
        )

        assert info.name == "test_module"
        assert info.module is module
        assert info.parameter_count == param_count
        assert info.size_mb == 1.0
        assert info.submodel_type == "vision"
        assert info.state == LoadingState.DISK
        assert not info.pinned

    def test_default_values(self) -> None:
        """Test that SubmodelInfo has sensible defaults."""
        module = DummySubmodel(size_mb=1.0)

        info = SubmodelInfo(
            name="test",
            module=module,
            parameter_count=1000,
            size_mb=0.1,
        )

        assert info.access_count == 0
        assert info.last_accessed == 0.0
        assert info.load_priority == 5
        assert info.pinned is False

    def test_memory_mb_alias(self) -> None:
        """Test that memory_mb is an alias for size_mb."""
        module = DummySubmodel(size_mb=1.0)

        info = SubmodelInfo(
            name="test",
            module=module,
            parameter_count=1000,
            size_mb=100.0,
        )

        assert info.memory_mb == info.size_mb
        assert info.memory_mb == 100.0


class TestProgressiveLoaderManager:
    """Test suite for ProgressiveLoaderManager."""

    @pytest.fixture
    def config(self) -> ProgressiveLoadingConfig:
        """Create test configuration with reasonable limits."""
        return ProgressiveLoadingConfig(
            max_active_submodels=3,
            gpu_budget_mb=500.0,
            eager_load_threshold=0.7,
            lazy_unload_threshold=0.3,
            async_loading=False,  # Synchronous for predictable testing
        )

    @pytest.fixture
    def loader(self, config: ProgressiveLoadingConfig) -> ProgressiveLoaderManager:
        """Create loader manager for testing."""
        return ProgressiveLoaderManager(config=config, device="cpu")

    def test_initialization(self, loader: ProgressiveLoaderManager) -> None:
        """Test loader initialization sets expected defaults."""
        assert loader.config.max_active_submodels == 3
        assert len(loader.submodels) == 0
        assert loader.gpu_memory_used == 0.0

    def test_register_submodel(self, loader: ProgressiveLoaderManager) -> None:
        """Test registering a submodel."""
        module = DummySubmodel(size_mb=1.0)

        loader.register_submodel(
            name="vision_0", module=module, size_mb=1.0, submodel_type="vision"
        )

        assert "vision_0" in loader.submodels
        info = loader.submodels["vision_0"]
        assert info.name == "vision_0"
        # After registration, submodel is staged to CPU
        assert info.state == LoadingState.STAGED
        assert info.submodel_type == "vision"

    def test_register_pinned(self, loader: ProgressiveLoaderManager) -> None:
        """Test registering a pinned submodel loads it immediately."""
        module = DummySubmodel(size_mb=1.0)

        loader.register_submodel(name="core", module=module, size_mb=1.0, pin_memory=True)

        info = loader.submodels["core"]
        assert info.pinned is True
        # Pinned modules are loaded immediately
        assert info.state == LoadingState.LOADED

    def test_activate_submodel(self, loader: ProgressiveLoaderManager) -> None:
        """Test activating a submodel."""
        module = DummySubmodel(size_mb=1.0)
        loader.register_submodel("vision_0", module=module, size_mb=1.0)

        # Activate
        success = loader.activate("vision_0", blocking=True)

        assert success
        assert loader.is_loaded("vision_0")
        assert "vision_0" in loader.active_names

    def test_deactivate_submodel(self, loader: ProgressiveLoaderManager) -> None:
        """Test deactivating a submodel."""
        module = DummySubmodel(size_mb=1.0)
        loader.register_submodel("vision_0", module=module, size_mb=1.0)

        # Activate then deactivate
        loader.activate("vision_0", blocking=True)
        assert loader.is_loaded("vision_0")

        success = loader.deactivate("vision_0")
        assert success
        assert not loader.is_loaded("vision_0")
        assert "vision_0" not in loader.active_names

    def test_deactivate_pinned_fails(self, loader: ProgressiveLoaderManager) -> None:
        """Test that pinned submodels cannot be deactivated."""
        module = DummySubmodel(size_mb=1.0)
        loader.register_submodel("core", module=module, size_mb=1.0, pin_memory=True)

        # Should fail to deactivate pinned module
        success = loader.deactivate("core")
        assert not success
        assert loader.is_loaded("core")

    def test_max_active_limit(self, loader: ProgressiveLoaderManager) -> None:
        """Test that max_active_submodels is enforced."""
        # Register 5 submodels (limit is 3)
        for i in range(5):
            module = DummySubmodel(size_mb=1.0)
            loader.register_submodel(f"module_{i}", module=module, size_mb=1.0)

        # Activate all 5
        for i in range(5):
            loader.activate(f"module_{i}", blocking=True)

        # Should only have 3 active (most recent)
        assert len(loader.active_names) <= 3

    def test_is_loaded_false_when_not_registered(self, loader: ProgressiveLoaderManager) -> None:
        """Test is_loaded returns False for unregistered submodels."""
        assert not loader.is_loaded("nonexistent")

    def test_is_loading_method(self, loader: ProgressiveLoaderManager) -> None:
        """Test is_loading returns False for staged submodels."""
        module = DummySubmodel(size_mb=1.0)
        loader.register_submodel("test", module=module, size_mb=1.0)

        # Not loading, just staged
        assert not loader.is_loading("test")

    def test_get_loaded(self, loader: ProgressiveLoaderManager) -> None:
        """Test retrieving loaded module."""
        module = DummySubmodel(size_mb=1.0)
        loader.register_submodel("vision_0", module=module, size_mb=1.0)

        # Not loaded yet (only staged)
        assert loader.get_loaded("vision_0") is None

        # Activate
        loader.activate("vision_0", blocking=True)

        # Now should return module
        loaded = loader.get_loaded("vision_0")
        assert loaded is not None
        assert isinstance(loaded, DummySubmodel)

    def test_get_stats(self, loader: ProgressiveLoaderManager) -> None:
        """Test statistics retrieval."""
        module = DummySubmodel(size_mb=1.0)
        loader.register_submodel("vision_0", module=module, size_mb=1.0)
        loader.activate("vision_0", blocking=True)

        stats = loader.get_stats()

        assert "gpu_memory_mb" in stats
        assert "total_submodels" in stats
        assert "active_names" in stats
        assert "vision_0" in stats["active_names"]

    def test_clear(self, loader: ProgressiveLoaderManager) -> None:
        """Test clearing all submodels."""
        for i in range(3):
            module = DummySubmodel(size_mb=1.0)
            loader.register_submodel(f"module_{i}", module=module, size_mb=1.0)
            loader.activate(f"module_{i}", blocking=True)

        # Clear
        loader.clear()

        assert len(loader.active_names) == 0
        assert loader.gpu_memory_used == 0

    def test_memory_budget_enforcement(self, loader: ProgressiveLoaderManager) -> None:
        """Test that GPU memory budget is enforced."""
        # Config has 500MB budget
        # Try to load 6×100MB = 600MB
        for i in range(6):
            module = DummySubmodel(size_mb=100)
            loader.register_submodel(f"module_{i}", module=module, size_mb=100.0)
            loader.activate(f"module_{i}", blocking=True)

        # Should not exceed budget
        assert loader.gpu_memory_used <= loader.config.gpu_budget_mb

    def test_submodel_type_tracking(self, loader: ProgressiveLoaderManager) -> None:
        """Test that submodel types are tracked."""
        vision_module = DummySubmodel(size_mb=1.0)
        language_module = DummySubmodel(size_mb=1.0)

        loader.register_submodel("vision_0", vision_module, size_mb=1.0, submodel_type="vision")
        loader.register_submodel(
            "language_0", language_module, size_mb=1.0, submodel_type="language"
        )

        assert loader.submodels["vision_0"].submodel_type == "vision"
        assert loader.submodels["language_0"].submodel_type == "language"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
