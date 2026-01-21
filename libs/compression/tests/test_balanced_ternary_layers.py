"""Tests for Balanced Ternary Neural Network Layers."""

from compression.balanced_ternary.layers import (
    BalancedTernaryConv2d,
    BalancedTernaryLinear,
)

import pytest
import torch
from torch import nn


class TestBalancedTernaryLinear:
    """Test suite for BalancedTernaryLinear layer."""

    @pytest.fixture
    def layer(self):
        """Create balanced ternary linear layer."""
        return BalancedTernaryLinear(in_features=128, out_features=64, trits_per_tryte=9)

    @pytest.fixture
    def layer_small(self):
        """Create small layer for fast tests."""
        return BalancedTernaryLinear(in_features=32, out_features=16, trits_per_tryte=9)

    def test_initialization(self, layer):
        """Test layer initialization."""
        assert layer.in_features == 128
        assert layer.out_features == 64
        assert layer.trits_per_tryte == 9

    def test_forward(self, layer_small):
        """Test forward pass."""
        batch_size = 4
        x = torch.randn(batch_size, 32)

        output = layer_small(x)

        assert output.shape == (batch_size, 16)
        assert not torch.isnan(output).any()

    def test_weight_quantization_training(self, layer_small):
        """Test that weights are quantized during training."""
        layer_small.train()
        layer_small.quantize_weights = True

        x = torch.randn(2, 32)
        _ = layer_small(x)

        # Weights should still be FP (STE), but forward uses quantized
        assert layer_small.weight.dtype in [torch.float32, torch.float16]

    def test_quantize_weights_explicit(self, layer_small):
        """Test explicit weight quantization."""
        # Quantize
        layer_small.quantize_weights_explicit()

        # After quantization, should be {-1, 0, +1}
        unique_vals = set(layer_small.weight.detach().flatten().tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})
        assert len(unique_vals) >= 2  # Should have some variety

    def test_get_weight_distribution(self, layer_small):
        """Test weight distribution analysis."""
        layer_small.quantize_weights_explicit()

        dist = layer_small.get_weight_distribution()

        assert "neg_one" in dist
        assert "zero" in dist
        assert "pos_one" in dist

        # Total should equal number of weights
        total = dist["neg_one"] + dist["zero"] + dist["pos_one"]
        expected = layer_small.in_features * layer_small.out_features
        assert total == expected

    def test_gradient_flow_ste(self, layer_small):
        """Test straight-through estimator gradient flow."""
        layer_small.train()
        layer_small.quantize_weights = True

        x = torch.randn(4, 32, requires_grad=True)
        target = torch.randn(4, 16)

        output = layer_small(x)
        loss = ((output - target) ** 2).mean()
        loss.backward()

        # Gradients should exist
        assert x.grad is not None
        assert layer_small.weight.grad is not None

    def test_bias_optional(self):
        """Test layer without bias."""
        layer_no_bias = BalancedTernaryLinear(in_features=32, out_features=16, bias=False)

        assert layer_no_bias.bias is None

        x = torch.randn(2, 32)
        output = layer_no_bias(x)
        assert output.shape == (2, 16)

    def test_batch_processing(self, layer_small):
        """Test different batch sizes."""
        batch_sizes = [1, 4, 16, 32]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 32)
            output = layer_small(x)
            assert output.shape == (batch_size, 16)

    def test_deterministic_quantization(self, layer_small):
        """Test that quantization is deterministic."""
        layer_small.quantize_weights_explicit()

        weights1 = layer_small.weight.clone()

        layer_small.quantize_weights_explicit()

        weights2 = layer_small.weight.clone()

        assert torch.equal(weights1, weights2)

    def test_comparison_with_standard_linear(self):
        """Compare output shape with standard nn.Linear."""
        bt_layer = BalancedTernaryLinear(64, 32)
        std_layer = nn.Linear(64, 32)

        x = torch.randn(8, 64)

        bt_output = bt_layer(x)
        std_output = std_layer(x)

        # Shapes should match
        assert bt_output.shape == std_output.shape


class TestBalancedTernaryConv2d:
    """Test suite for BalancedTernaryConv2d layer."""

    @pytest.fixture
    def conv(self):
        """Create balanced ternary conv layer."""
        return BalancedTernaryConv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            padding=1,
            trits_per_tryte=9,
        )

    @pytest.fixture
    def conv_small(self):
        """Create small conv for fast tests."""
        return BalancedTernaryConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)

    def test_initialization(self, conv):
        """Test conv layer initialization."""
        assert conv.in_channels == 3
        assert conv.out_channels == 64
        assert conv.kernel_size == (3, 3)

    def test_forward(self, conv_small):
        """Test forward pass."""
        batch_size = 4
        x = torch.randn(batch_size, 8, 32, 32)  # CIFAR-like

        output = conv_small(x)

        assert output.shape == (batch_size, 16, 32, 32)  # Same spatial size (padding=1)
        assert not torch.isnan(output).any()

    def test_weight_quantization_training(self, conv_small):
        """Test weight quantization during training."""
        conv_small.train()
        conv_small.quantize_weights = True

        x = torch.randn(2, 8, 16, 16)
        _ = conv_small(x)

        # Weights should still be FP (STE)
        assert conv_small.weight.dtype in [torch.float32, torch.float16]

    def test_quantize_weights_explicit(self, conv_small):
        """Test explicit weight quantization."""
        conv_small.quantize_weights_explicit()

        # Should be {-1, 0, +1}
        unique_vals = set(conv_small.weight.detach().flatten().tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_gradient_flow(self, conv_small):
        """Test gradient flow through conv layer."""
        conv_small.train()
        conv_small.quantize_weights = True

        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        target = torch.randn(2, 16, 16, 16)

        output = conv_small(x)
        loss = ((output - target) ** 2).mean()
        loss.backward()

        assert x.grad is not None
        assert conv_small.weight.grad is not None

    def test_different_kernel_sizes(self):
        """Test different kernel sizes."""
        kernel_sizes = [1, 3, 5, 7]

        for k in kernel_sizes:
            conv = BalancedTernaryConv2d(
                in_channels=4, out_channels=8, kernel_size=k, padding=k // 2
            )

            x = torch.randn(2, 4, 32, 32)
            output = conv(x)

            # With proper padding, spatial size should be preserved
            assert output.shape == (2, 8, 32, 32)

    def test_stride_and_padding(self):
        """Test different stride and padding configurations."""
        # Stride 2 (downsampling)
        conv_downsample = BalancedTernaryConv2d(
            in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1
        )

        x = torch.randn(2, 4, 32, 32)
        output = conv_downsample(x)

        # Should halve spatial dimensions
        assert output.shape == (2, 8, 16, 16)

    def test_comparison_with_standard_conv(self):
        """Compare with standard nn.Conv2d."""
        bt_conv = BalancedTernaryConv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        std_conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        x = torch.randn(4, 3, 32, 32)

        bt_output = bt_conv(x)
        std_output = std_conv(x)

        # Shapes should match
        assert bt_output.shape == std_output.shape


class TestBalancedTernaryNetwork:
    """Test full network with balanced ternary layers."""

    def test_simple_mlp(self):
        """Test simple MLP with balanced ternary layers."""

        class TernaryMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = BalancedTernaryLinear(784, 128)
                self.relu = nn.ReLU()
                self.fc2 = BalancedTernaryLinear(128, 10)

            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = TernaryMLP()
        x = torch.randn(8, 1, 28, 28)  # MNIST-like

        output = model(x)

        assert output.shape == (8, 10)

    def test_simple_cnn(self):
        """Test simple CNN with balanced ternary layers."""

        class TernaryCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = BalancedTernaryConv2d(3, 32, kernel_size=3, padding=1)
                self.conv2 = BalancedTernaryConv2d(32, 64, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc = BalancedTernaryLinear(64 * 8 * 8, 10)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))  # 32×32 → 16×16
                x = self.pool(self.relu(self.conv2(x)))  # 16×16 → 8×8
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = TernaryCNN()
        x = torch.randn(4, 3, 32, 32)  # CIFAR-like

        output = model(x)

        assert output.shape == (4, 10)

    def test_mixed_precision_network(self):
        """Test network mixing standard and ternary layers."""

        class MixedNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 128)  # Standard FP
                self.fc2 = BalancedTernaryLinear(128, 128)  # Ternary
                self.fc3 = nn.Linear(128, 10)  # Standard FP
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        model = MixedNetwork()
        x = torch.randn(4, 64)

        output = model(x)

        assert output.shape == (4, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
