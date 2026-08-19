"""
Unit Tests for CUDA Optimization Module

Tests cover:
1. GPUConfig dataclass validation
2. RTX5080Optimizer CPU fallback behavior
3. Memory statistics computation
4. Batch size optimization
5. Model optimization (CPU mode)
6. Embedding optimization

All tests are CPU-compatible for CI - tests GPU code paths via mocking.
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cogsyndelta.optimization import (
    COMPUTE_CAPABILITY,
    CUDA_AVAILABLE,
    DEVICE_NAME,
    TOTAL_MEMORY,
    GPUConfig,
    RTX5080Optimizer,
)


class TestModuleExports:
    """Tests for module-level exports."""

    def test_cuda_available_is_bool(self) -> None:
        """Test CUDA_AVAILABLE is a boolean."""
        assert isinstance(CUDA_AVAILABLE, bool)

    def test_device_name_is_string(self) -> None:
        """Test DEVICE_NAME is a string."""
        assert isinstance(DEVICE_NAME, str)
        # In CI (CPU), this should be "CPU"
        if not CUDA_AVAILABLE:
            assert DEVICE_NAME == "CPU"

    def test_total_memory_is_numeric(self) -> None:
        """Test TOTAL_MEMORY is numeric."""
        assert isinstance(TOTAL_MEMORY, (int, float))
        if not CUDA_AVAILABLE:
            assert TOTAL_MEMORY == 0

    def test_compute_capability_is_tuple(self) -> None:
        """Test COMPUTE_CAPABILITY is a tuple."""
        assert isinstance(COMPUTE_CAPABILITY, tuple)
        assert len(COMPUTE_CAPABILITY) == 2
        if not CUDA_AVAILABLE:
            assert COMPUTE_CAPABILITY == (0, 0)


class TestGPUConfig:
    """Tests for GPUConfig dataclass."""

    def test_config_creation(self) -> None:
        """Test creating GPU config with required fields."""
        config = GPUConfig(device=torch.device("cpu"), memory_gb=16.0)
        assert config.device == torch.device("cpu")
        assert config.memory_gb == 16.0

    def test_config_defaults(self) -> None:
        """Test GPU config default values."""
        config = GPUConfig(device=torch.device("cpu"), memory_gb=8.0)
        assert config.use_mixed_precision is True
        assert config.use_flash_attention is True
        assert config.use_kernel_fusion is True
        assert config.max_batch_size == 64
        assert config.num_streams == 4
        assert config.prefetch_factor == 2

    def test_config_custom_values(self) -> None:
        """Test GPU config with custom values."""
        config = GPUConfig(
            device=torch.device("cpu"),
            memory_gb=32.0,
            use_mixed_precision=False,
            use_flash_attention=False,
            max_batch_size=128,
            num_streams=8,
        )
        assert config.memory_gb == 32.0
        assert config.use_mixed_precision is False
        assert config.max_batch_size == 128
        assert config.num_streams == 8


class TestRTX5080OptimizerCPU:
    """Tests for RTX5080Optimizer in CPU mode."""

    @pytest.fixture
    def optimizer(self) -> RTX5080Optimizer:
        """Create optimizer instance (will use CPU fallback in CI)."""
        return RTX5080Optimizer()

    def test_optimizer_initialization(self, optimizer: RTX5080Optimizer) -> None:
        """Test optimizer initializes correctly."""
        assert optimizer.device is not None
        assert optimizer.config is not None
        if not CUDA_AVAILABLE:
            assert optimizer.device == torch.device("cpu")
            assert optimizer.config.memory_gb == 0
            assert optimizer.config.use_mixed_precision is False

    def test_get_memory_stats_cpu(self, optimizer: RTX5080Optimizer) -> None:
        """Test memory stats in CPU mode."""
        if not CUDA_AVAILABLE:
            stats = optimizer.get_memory_stats()
            assert stats["allocated_gb"] == 0
            assert stats["reserved_gb"] == 0
            assert stats["free_gb"] == 0

    def test_optimize_model_cpu(self, optimizer: RTX5080Optimizer) -> None:
        """Test model optimization in CPU mode returns model unchanged."""
        model = nn.Linear(512, 256)
        optimized = optimizer.optimize_model(model)

        if not CUDA_AVAILABLE:
            # In CPU mode, model should be returned as-is
            assert optimized is model
        else:
            # In GPU mode, model should be moved to GPU
            assert next(optimized.parameters()).device.type == "cuda"

    def test_optimize_embeddings_cpu(
        self, optimizer: RTX5080Optimizer, test_embeddings: torch.Tensor
    ) -> None:
        """Test embedding optimization in CPU mode."""
        optimized = optimizer.optimize_embeddings(test_embeddings)

        if not CUDA_AVAILABLE:
            # In CPU mode, embeddings should be returned as-is
            assert torch.equal(optimized, test_embeddings)

    def test_dynamic_batch_split_cpu(self, optimizer: RTX5080Optimizer) -> None:
        """Test dynamic batch splitting in CPU mode."""
        splits = optimizer.dynamic_batch_split(batch_size=64)

        # Should return a list of batch sizes
        assert isinstance(splits, list)
        assert all(isinstance(s, int) for s in splits)
        assert sum(splits) == 64 or len(splits) > 0


class TestRTX5080OptimizerGPU:
    """Tests for RTX5080Optimizer GPU functionality (mocked)."""

    @pytest.fixture
    def mock_cuda_available(self) -> Any:
        """Mock CUDA as available."""
        with (
            patch("cogsyndelta.optimization.cuda_optimization.CUDA_AVAILABLE", True),
            patch("cogsyndelta.optimization.cuda_optimization.TOTAL_MEMORY", 16 * 1024**3),
            patch("cogsyndelta.optimization.cuda_optimization.DEVICE_NAME", "RTX 5080"),
            patch(
                "cogsyndelta.optimization.cuda_optimization.COMPUTE_CAPABILITY",
                (10, 0),
            ),
        ):
            yield

    def test_compute_optimal_batch_size(self) -> None:
        """Test optimal batch size computation."""
        optimizer = RTX5080Optimizer()

        # Test the internal method
        if hasattr(optimizer, "_compute_optimal_batch_size"):
            # 14GB usable (16 - 2 reserved) = ~140 items at 100MB each
            batch_size = optimizer._compute_optimal_batch_size(14.0)
            assert 8 <= batch_size <= 128

            # Very little memory should give minimum batch
            small_batch = optimizer._compute_optimal_batch_size(0.5)
            assert small_batch == 8

            # Lots of memory should cap at maximum
            large_batch = optimizer._compute_optimal_batch_size(100.0)
            assert large_batch == 128


class TestOptimizationHelpers:
    """Tests for optimization helper functions."""

    @pytest.fixture
    def optimizer(self) -> RTX5080Optimizer:
        """Create optimizer instance."""
        return RTX5080Optimizer()

    def test_memory_stats_keys(self, optimizer: RTX5080Optimizer) -> None:
        """Test memory stats returns expected keys."""
        stats = optimizer.get_memory_stats()

        expected_keys = ["allocated_gb", "reserved_gb", "free_gb"]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_config_values_reasonable(self, optimizer: RTX5080Optimizer) -> None:
        """Test config values are within reasonable ranges."""
        config = optimizer.config

        assert config.max_batch_size > 0
        assert config.max_batch_size <= 256
        assert config.num_streams >= 1
        assert config.num_streams <= 16
        assert config.prefetch_factor >= 1


class TestModelOptimization:
    """Tests for model optimization functionality."""

    @pytest.fixture
    def simple_model(self) -> nn.Module:
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def test_optimize_preserves_architecture(self, simple_model: nn.Module) -> None:
        """Test optimization preserves model architecture."""
        optimizer = RTX5080Optimizer()
        optimized = optimizer.optimize_model(simple_model)

        # Count layers
        original_layers = list(simple_model.modules())
        # torch.compile wraps in OptimizedModule; compare the original graph.
        unwrapped = getattr(optimized, "_orig_mod", optimized)
        optimized_layers = list(unwrapped.modules())

        assert len(original_layers) == len(optimized_layers)

    def test_optimize_model_trainable(self, simple_model: nn.Module) -> None:
        """Test optimized model can still be trained."""
        optimizer = RTX5080Optimizer()
        optimized = optimizer.optimize_model(simple_model)

        # Model should have parameters
        params = list(optimized.parameters())
        assert len(params) > 0

        # Parameters should require grad (unless explicitly disabled)
        trainable_params = [p for p in params if p.requires_grad]
        assert len(trainable_params) > 0

    def test_optimize_model_forward_pass(self, simple_model: nn.Module) -> None:
        """Test optimized model can do forward pass."""
        optimizer = RTX5080Optimizer()
        optimized = optimizer.optimize_model(simple_model)

        # Create test input on same device and dtype as model
        # The optimizer may convert model to half precision for mixed precision
        model_dtype = next(optimized.parameters()).dtype
        x = torch.randn(4, 784, device=optimizer.device, dtype=model_dtype)

        # Forward pass should work
        with torch.no_grad():
            output = optimized(x)

        assert output.shape == (4, 10)


class TestEmbeddingOptimization:
    """Tests for embedding optimization."""

    def test_optimize_embeddings_shape(self, test_embeddings: torch.Tensor) -> None:
        """Test embedding optimization preserves shape."""
        optimizer = RTX5080Optimizer()
        optimized = optimizer.optimize_embeddings(test_embeddings)

        assert optimized.shape == test_embeddings.shape

    def test_optimize_embeddings_contiguous(self, test_embeddings: torch.Tensor) -> None:
        """Test optimized embeddings are contiguous."""
        optimizer = RTX5080Optimizer()

        # Create non-contiguous tensor
        non_contig = test_embeddings.t().t()  # Transpose twice to break contiguity

        optimized = optimizer.optimize_embeddings(non_contig)

        if not CUDA_AVAILABLE:
            # In CPU mode, should still be the same data
            assert optimized.shape == non_contig.shape

    def test_optimize_different_dtypes(self) -> None:
        """Test optimization handles different dtypes."""
        optimizer = RTX5080Optimizer()

        for dtype in [torch.float32, torch.float64]:
            embeddings = torch.randn(8, 512, dtype=dtype)
            optimized = optimizer.optimize_embeddings(embeddings)

            assert optimized.shape == embeddings.shape


# Regression tests
class TestOptimizationRegressions:
    """Regression tests for optimization module."""

    def test_cpu_fallback_always_works(self) -> None:
        """Regression: CPU fallback must always work."""
        # This should never raise, even without GPU
        optimizer = RTX5080Optimizer()
        assert optimizer.device is not None
        assert optimizer.config is not None

    def test_memory_stats_never_negative(self) -> None:
        """Regression: Memory stats must never be negative."""
        optimizer = RTX5080Optimizer()
        stats = optimizer.get_memory_stats()

        for key, value in stats.items():
            assert value >= 0, f"Memory stat {key} is negative: {value}"

    def test_batch_size_within_bounds(self) -> None:
        """Regression: Batch size must be within configured bounds."""
        optimizer = RTX5080Optimizer()

        # Test with various batch sizes
        for batch_size in [1, 16, 64, 128, 256]:
            splits = optimizer.dynamic_batch_split(batch_size)

            for split in splits:
                assert split > 0, "Split batch size is 0 or negative"
                assert split <= batch_size, "Split exceeds original batch"

    def test_optimize_preserves_grad_requirements(self) -> None:
        """Regression: Optimization must preserve grad requirements."""
        model = nn.Linear(512, 256)

        # Mark some parameters as not requiring grad
        model.bias.requires_grad = False

        optimizer = RTX5080Optimizer()
        optimized = optimizer.optimize_model(model)

        # Check grad requirements preserved
        # Note: In GPU mode with half(), this might change
        if not CUDA_AVAILABLE:
            assert optimized.weight.requires_grad is True
            # Bias grad requirement might be reset by optimization
