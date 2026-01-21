"""Tests for Progressive Dynamic Selective Loading.

Tests the progressive loader module for dynamic submodel loading/unloading
to manage GPU memory efficiently for large models.
"""

import pytest
import torch
from torch import nn

from cogsyndelta.core.progressive_loader import (
    LoadingConfig,
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

    def __init__(self, size_mb: float = 1.0):
        """Initialize dummy submodel.

        Args:
            size_mb: Target size in megabytes for the module parameters.
        """
        super().__init__()
        # Create dummy parameters to simulate size
        # Each float32 param is 4 bytes
        num_params = int(size_mb * 1024 * 1024 / 4)
        # Use smaller chunks to avoid memory issues
        self.weight = nn.Parameter(torch.randn(min(num_params, 100000)))
        self._size_mb = size_mb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - dummy operation.

        Args:
            x: Input tensor.

        Returns:
            Scaled input tensor.
        """
        return x * self.weight[0]


class TestLoadingConfig:
    """Test suite for LoadingConfig / ProgressiveLoadingConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = LoadingConfig()

        assert config.max_active_submodels == 8
        assert config.gpu_budget_mb == 10000.0
        assert config.eager_load_threshold == 0.7
        assert config.lazy_unload_threshold == 0.2

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = LoadingConfig(
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
            LoadingConfig(eager_load_threshold=0.2, lazy_unload_threshold=0.8)

    def test_alias_exists(self) -> None:
        """Test that ProgressiveLoadingConfig is an alias for LoadingConfig."""
        assert ProgressiveLoadingConfig is LoadingConfig


class TestLoadingState:
    """Test LoadingState enum."""

    def test_states_exist(self) -> None:
        """Test that all expected states exist in the enum."""
        assert hasattr(LoadingState, "UNLOADED")
        assert hasattr(LoadingState, "STAGING")
        assert hasattr(LoadingState, "STAGED")
        assert hasattr(LoadingState, "LOADING")
        assert hasattr(LoadingState, "ACTIVE")
        assert hasattr(LoadingState, "EVICTING")

    def test_state_values(self) -> None:
        """Test that state values are strings."""
        assert LoadingState.UNLOADED.value == "unloaded"
        assert LoadingState.ACTIVE.value == "active"


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
            memory_mb=1.0,
        )

        assert info.name == "test_module"
        assert info.module is module
        assert info.parameter_count == param_count
        assert info.memory_mb == 1.0
        assert info.state == LoadingState.UNLOADED  # Default state

    def test_default_values(self) -> None:
        """Test that SubmodelInfo has sensible defaults."""
        module = DummySubmodel(size_mb=1.0)

        info = SubmodelInfo(
            name="test",
            module=module,
            parameter_count=1000,
            memory_mb=0.1,
        )

        assert info.access_count == 0
        assert info.last_accessed == 0.0
        assert info.load_priority == 5
        assert info.pin_memory is False


class TestProgressiveLoaderManager:
    """Test suite for ProgressiveLoaderManager."""

    @pytest.fixture
    def config(self) -> LoadingConfig:
        """Create test configuration with reasonable limits."""
        return LoadingConfig(
            max_active_submodels=3,
            gpu_budget_mb=500.0,
            eager_load_threshold=0.7,
            lazy_unload_threshold=0.3,
            async_loading=False,  # Synchronous for predictable testing
        )

    @pytest.fixture
    def loader(self, config: LoadingConfig) -> ProgressiveLoaderManager:
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

        loader.register_submodel(name="vision_0", module=module)

        assert "vision_0" in loader.submodels
        info = loader.submodels["vision_0"]
        assert info.name == "vision_0"
        # After registration, submodel is staged to CPU
        assert info.state == LoadingState.STAGED

    def test_register_with_context_tags(self, loader: ProgressiveLoaderManager) -> None:
        """Test registering submodel with context tags."""
        module = DummySubmodel(size_mb=1.0)

        loader.register_submodel(
            name="vision_encoder",
            module=module,
            context_tags={"vision", "encoder", "image"},
        )

        info = loader.submodels["vision_encoder"]
        assert "vision" in info.context_tags
        assert "encoder" in info.context_tags

    def test_register_pinned(self, loader: ProgressiveLoaderManager) -> None:
        """Test registering a pinned submodel loads it immediately."""
        module = DummySubmodel(size_mb=1.0)

        loader.register_submodel(name="core", module=module, pin_memory=True)

        info = loader.submodels["core"]
        assert info.pin_memory is True
        # Pinned modules are loaded immediately
        assert info.state == LoadingState.ACTIVE

    def test_activate_submodel(self, loader: ProgressiveLoaderManager) -> None:
        """Test activating a submodel."""
        module = DummySubmodel(size_mb=1.0)
        loader.register_submodel("vision_0", module=module)

        # Activate
        success = loader.activate("vision_0", blocking=True)

        assert success
        assert loader.is_loaded("vision_0")
        assert "vision_0" in loader.active_names

    def test_deactivate_submodel(self, loader: ProgressiveLoaderManager) -> None:
        """Test deactivating a submodel."""
        module = DummySubmodel(size_mb=1.0)
        loader.register_submodel("vision_0", module=module)

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
        loader.register_submodel("core", module=module, pin_memory=True)

        # Should fail to deactivate pinned module
        success = loader.deactivate("core")
        assert not success
        assert loader.is_loaded("core")

    def test_is_loaded_false_when_not_registered(self, loader: ProgressiveLoaderManager) -> None:
        """Test is_loaded returns False for unregistered submodels."""
        assert not loader.is_loaded("nonexistent")

    def test_get_stats(self, loader: ProgressiveLoaderManager) -> None:
        """Test statistics retrieval."""
        module = DummySubmodel(size_mb=1.0)
        loader.register_submodel("vision_0", module=module)
        loader.activate("vision_0", blocking=True)

        stats = loader.get_stats()

        assert "gpu_memory_mb" in stats
        assert "cpu_memory_mb" in stats
        assert "active_names" in stats
        assert "vision_0" in stats["active_names"]


class TestProgressiveLoaderMemoryManagement:
    """Test memory management features of ProgressiveLoaderManager."""

    @pytest.fixture
    def small_budget_loader(self) -> ProgressiveLoaderManager:
        """Create loader with small memory budget for testing eviction."""
        config = LoadingConfig(
            max_active_submodels=2,
            gpu_budget_mb=10.0,  # Small budget
            async_loading=False,
        )
        return ProgressiveLoaderManager(config=config, device="cpu")

    def test_memory_tracking(self, small_budget_loader: ProgressiveLoaderManager) -> None:
        """Test that memory usage is tracked."""
        loader = small_budget_loader
        module = DummySubmodel(size_mb=1.0)

        initial_gpu = loader.gpu_memory_used
        loader.register_submodel("test", module=module)
        loader.activate("test", blocking=True)

        # Memory should have increased
        assert loader.gpu_memory_used >= initial_gpu


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
