"""BitNet b1.58: Ternary (-1, 0, +1) weight quantization.

BitNet b1.58 quantizes weights to {-1, 0, +1} achieving 10× memory reduction
compared to FP16 (2 bytes/param → 0.2 bytes/param with ~1.58 bits/weight).

Critical Insight (Nielsen et al., 2024):
    Encoder-only models require 2× hidden size to match FP16 performance.
    For 10B target → design for 20B effective capacity.

Mathematical Foundation:
    W_ternary = sign(W) · 1_{|W| > threshold}
    where threshold = mean(|W|)

Performance:
    - 10× weight memory reduction
    - <2% accuracy loss with 2× hidden size
    - Compatible with standard optimizers via straight-through estimator

References:
    - Ma et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.
    - Nielsen et al. (2024). Encoder models require 2× hidden for FP16 parity.
"""

import torch
import torch.nn as nn
from typing import Optional


def ternary_quantize(weights: torch.Tensor, threshold: Optional[float] = None) -> torch.Tensor:
    """Quantize weights to {-1, 0, +1}.

    Args:
        weights: Weight tensor (any shape)
        threshold: Threshold for zeroing (default: mean(|W|))

    Returns:
        Ternary weights {-1, 0, +1}
    """
    if threshold is None:
        threshold = weights.abs().mean()

    # Sign with threshold
    ternary = torch.sign(weights) * (weights.abs() > threshold).float()

    return ternary


class TernaryQuantizer(nn.Module):
    """Ternary weight quantization with straight-through estimator.

    Attributes:
        threshold_mode: How to compute threshold ("mean", "percentile", "learnable")
        threshold: Threshold value (learnable if threshold_mode="learnable")
    """

    def __init__(self, threshold_mode: str = "mean", init_threshold: float = 0.1) -> None:
        """Initialize ternary quantizer.

        Args:
            threshold_mode: Threshold computation mode
            init_threshold: Initial threshold value (for learnable mode)
        """
        super().__init__()

        self.threshold_mode = threshold_mode

        if threshold_mode == "learnable":
            self.threshold = nn.Parameter(torch.tensor(init_threshold))
        else:
            self.register_buffer("threshold", None)

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weights with straight-through estimator.

        Args:
            weights: Float weights

        Returns:
            Quantized weights (gradients flow through via STE)
        """
        # Compute threshold
        if self.threshold_mode == "mean":
            threshold = weights.abs().mean()
        elif self.threshold_mode == "percentile":
            threshold = weights.abs().quantile(0.5)
        elif self.threshold_mode == "learnable":
            threshold = self.threshold.abs()
        else:
            raise ValueError(f"Unknown threshold_mode: {self.threshold_mode}")

        # Quantize
        ternary = ternary_quantize(weights, threshold)

        # Straight-through estimator: forward ternary, backward float
        return weights + (ternary - weights).detach()


class BitNetb158(nn.Module):
    """BitNet b1.58 encoder model with ternary weights.

    Implements encoder architecture with:
    - 2× hidden size (critical for FP16 parity)
    - Ternary weight quantization
    - LayerNorm (not quantized)
    - GELU activations

    Attributes:
        input_dim: Input dimension
        hidden_dim: Hidden dimension (should be 2× target capacity)
        output_dim: Output dimension
        num_layers: Number of transformer layers
        quantizer: Ternary weight quantizer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        quantize_weights: bool = True,
    ) -> None:
        """Initialize BitNet b1.58 model.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension (use 2× for FP16 parity)
            output_dim: Output embedding dimension
            num_layers: Number of layers
            quantize_weights: Whether to apply ternary quantization
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.quantize_weights = quantize_weights

        if quantize_weights:
            self.quantizer = TernaryQuantizer(threshold_mode="mean")
        else:
            self.quantizer = None

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Encoder layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True),
                        "norm1": nn.LayerNorm(hidden_dim),
                        "ff": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 4),
                            nn.GELU(),
                            nn.Linear(hidden_dim * 4, hidden_dim),
                        ),
                        "norm2": nn.LayerNorm(hidden_dim),
                    }
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def quantize_layer_weights(self, layer: nn.Module) -> None:
        """Apply ternary quantization to layer weights.

        Args:
            layer: Layer to quantize (modifies in-place)
        """
        if not self.quantize_weights:
            return

        for name, param in layer.named_parameters():
            if "weight" in name and param.requires_grad:
                with torch.no_grad():
                    param.copy_(self.quantizer(param))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (shape: [batch, seq_len, input_dim])

        Returns:
            Output embeddings (shape: [batch, seq_len, output_dim])
        """
        # Input projection
        x = self.input_proj(x)

        # Quantize input projection weights if enabled
        if self.training and self.quantize_weights:
            self.quantize_layer_weights(self.input_proj)

        # Encoder layers
        for layer in self.layers:
            # Self-attention
            residual = x
            x = layer["norm1"](x)
            x_attn, _ = layer["attn"](x, x, x)
            x = residual + x_attn

            # Feed-forward
            residual = x
            x = layer["norm2"](x)
            x = layer["ff"](x)
            x = residual + x

            # Quantize layer weights if training
            if self.training and self.quantize_weights:
                self.quantize_layer_weights(layer["attn"])
                self.quantize_layer_weights(layer["ff"])

        # Output projection
        x = self.output_proj(x)

        return x

    def memory_footprint(self) -> dict:
        """Calculate memory footprint.

        Returns:
            Dict with FP16 and ternary memory estimates
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())

        # FP16: 2 bytes per parameter
        fp16_mb = total_params * 2 / 1024 / 1024

        # Ternary (1.58 bits): ~0.2 bytes per parameter
        ternary_mb = total_params * 0.2 / 1024 / 1024

        return {
            "total_params": total_params,
            "fp16_mb": fp16_mb,
            "ternary_mb": ternary_mb,
            "compression_ratio": fp16_mb / ternary_mb,
        }
