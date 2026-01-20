"""Balanced Ternary Neural Network Layers.

Drop-in replacements for standard PyTorch layers using balanced ternary weights.

Similar to BitNet b1.58, but with balanced ternary representation:
- Linear layers with {-1, 0, +1} weights
- Convolutional layers with balanced ternary kernels
- Embeddings with balanced ternary lookup

Memory efficiency:
- FP16: 2 bytes per parameter
- Balanced ternary: ~0.2 bytes per parameter
- Compression: ~10×
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from compression.balanced_ternary.quantizer import BalancedTernaryQuantizer


class BalancedTernaryLinear(nn.Module):
    """Linear layer with balanced ternary weights.

    Attributes:
        in_features: Input dimension
        out_features: Output dimension
        weight: Balanced ternary weights {-1, 0, 1}
        bias: Optional bias (kept in FP16 for stability)
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, threshold_mode: str = "mean"
    ):
        """Initialize balanced ternary linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to use bias
            threshold_mode: Quantization threshold mode
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Float weights for training (quantized during forward)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Quantizer
        self.quantizer = BalancedTernaryQuantizer(threshold_mode=threshold_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with balanced ternary weights.

        Args:
            x: Input tensor (shape: [..., in_features])

        Returns:
            Output tensor (shape: [..., out_features])
        """
        # Quantize weights to balanced ternary with STE
        weight_ternary = self.quantizer(self.weight)

        # Standard linear operation
        output = F.linear(x, weight_ternary, self.bias)

        return output

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"quantization=balanced_ternary"
        )


class BalancedTernaryConv2d(nn.Module):
    """2D Convolution with balanced ternary kernels.

    Attributes:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        weight: Balanced ternary convolutional kernels
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        threshold_mode: str = "mean",
    ):
        """Initialize balanced ternary conv layer.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Kernel size
            stride: Stride
            padding: Padding
            bias: Whether to use bias
            threshold_mode: Quantization threshold mode
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Float weights for training
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # Quantizer
        self.quantizer = BalancedTernaryQuantizer(threshold_mode=threshold_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with balanced ternary kernels.

        Args:
            x: Input tensor (shape: [batch, in_channels, H, W])

        Returns:
            Output tensor (shape: [batch, out_channels, H', W'])
        """
        # Quantize weights to balanced ternary
        weight_ternary = self.quantizer(self.weight)

        # Standard conv2d
        output = F.conv2d(x, weight_ternary, self.bias, stride=self.stride, padding=self.padding)

        return output

    def extra_repr(self) -> str:
        """Extra representation."""
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"bias={self.bias is not None}, "
            f"quantization=balanced_ternary"
        )


class BalancedTernaryEmbedding(nn.Module):
    """Embedding layer with balanced ternary lookup table.

    Attributes:
        num_embeddings: Vocabulary size
        embedding_dim: Embedding dimension
        weight: Balanced ternary embedding matrix
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        threshold_mode: str = "mean",
    ):
        """Initialize balanced ternary embedding.

        Args:
            num_embeddings: Vocabulary size
            embedding_dim: Embedding dimension
            padding_idx: Padding index
            threshold_mode: Quantization threshold mode
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Float weights for training
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, mean=0, std=1.0)

        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

        # Quantizer
        self.quantizer = BalancedTernaryQuantizer(threshold_mode=threshold_mode)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Forward pass with balanced ternary embeddings.

        Args:
            indices: Input indices (shape: [...])

        Returns:
            Embeddings (shape: [..., embedding_dim])
        """
        # Quantize embeddings to balanced ternary
        weight_ternary = self.quantizer(self.weight)

        # Standard embedding lookup
        output = F.embedding(indices, weight_ternary, padding_idx=self.padding_idx)

        return output

    def extra_repr(self) -> str:
        """Extra representation."""
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"padding_idx={self.padding_idx}, "
            f"quantization=balanced_ternary"
        )


class BalancedTernaryActivation(nn.Module):
    """Activation function optimized for balanced ternary.

    Uses tanh with ternary discretization for output in {-1, 0, +1}.
    """

    def __init__(self, threshold: float = 0.33):
        """Initialize balanced ternary activation.

        Args:
            threshold: Threshold for ternary discretization
        """
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ternary activation.

        Args:
            x: Input tensor

        Returns:
            Ternary activated tensor
        """
        # Tanh activation
        activated = torch.tanh(x)

        # Discretize to {-1, 0, +1}
        ternary = torch.zeros_like(activated)
        ternary[activated > self.threshold] = 1.0
        ternary[activated < -self.threshold] = -1.0

        # Straight-through estimator
        return activated + (ternary - activated).detach()


# Example: Building a balanced ternary MLP
class BalancedTernaryMLP(nn.Module):
    """Example MLP with balanced ternary weights.

    All weights in {-1, 0, +1} for 10× memory reduction.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        """Initialize balanced ternary MLP.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of hidden layers
        """
        super().__init__()

        layers = []

        # Input layer
        layers.append(BalancedTernaryLinear(input_dim, hidden_dim))
        layers.append(BalancedTernaryActivation())

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(BalancedTernaryLinear(hidden_dim, hidden_dim))
            layers.append(BalancedTernaryActivation())

        # Output layer
        layers.append(BalancedTernaryLinear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.network(x)

    def memory_footprint(self) -> dict:
        """Calculate memory footprint.

        Returns:
            Dict with FP16 and balanced ternary estimates
        """
        total_params = sum(p.numel() for p in self.parameters())

        # FP16: 2 bytes per parameter
        fp16_mb = total_params * 2 / (1024**2)

        # Balanced ternary: ~0.2 bytes per parameter
        ternary_mb = total_params * 0.2 / (1024**2)

        return {
            "total_params": total_params,
            "fp16_mb": fp16_mb,
            "balanced_ternary_mb": ternary_mb,
            "compression_ratio": fp16_mb / ternary_mb,
        }
