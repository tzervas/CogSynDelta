"""Balanced Ternary Quantization for Neural Network Weights.

Quantizes FP16/FP32 weights to balanced ternary {-1, 0, +1} for 10× memory reduction.

Similar to BitNet b1.58, but uses balanced ternary representation which is:
- More symmetric (natural around zero)
- More brain-like (inhibit/neutral/excite)
- Easier to implement in hardware (no explicit sign bit)

Quantization Strategies:
1. Threshold-based: |w| < threshold → 0, else sign(w)
2. Round-to-nearest: Round to {-1, 0, 1}
3. Learned thresholds: Optimize thresholds during training
"""

import torch
import torch.nn as nn
from typing import Tuple

from compression.balanced_ternary.arithmetic import (
    BalancedTernaryTensor,
    tryte_encode,
)


class BalancedTernaryQuantizer(nn.Module):
    """Quantizes weights to balanced ternary {-1, 0, +1}.

    Attributes:
        threshold_mode: How to determine threshold ("mean", "percentile", "learnable")
        threshold: Threshold value (learnable if mode="learnable")
    """

    def __init__(
        self, threshold_mode: str = "mean", init_threshold: float = 0.1, trits_per_tryte: int = 9
    ):
        """Initialize balanced ternary quantizer.

        Args:
            threshold_mode: Threshold computation mode
            init_threshold: Initial threshold value
            trits_per_tryte: Trits per tryte (default: 9)
        """
        super().__init__()

        self.threshold_mode = threshold_mode
        self.trits_per_tryte = trits_per_tryte

        if threshold_mode == "learnable":
            self.threshold = nn.Parameter(torch.tensor(init_threshold))
        else:
            self.register_buffer("threshold", torch.tensor(init_threshold))

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weights to balanced ternary with STE.

        Args:
            weights: Float weights (any shape)

        Returns:
            Quantized weights {-1, 0, 1} (same shape, with STE gradients)
        """
        # Compute threshold
        if self.threshold_mode == "mean":
            threshold = weights.abs().mean()
        elif self.threshold_mode == "percentile":
            threshold = weights.abs().quantile(0.5)  # Median
        elif self.threshold_mode == "learnable":
            threshold = self.threshold.abs()
        else:
            raise ValueError(f"Unknown threshold_mode: {self.threshold_mode}")

        # Quantize to balanced ternary
        ternary = self._quantize_balanced_ternary(weights, threshold)

        # Straight-through estimator: forward ternary, backward float
        return weights + (ternary - weights).detach()

    @staticmethod
    def _quantize_balanced_ternary(weights: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        """Quantize to balanced ternary {-1, 0, 1}.

        Args:
            weights: Float weights
            threshold: Threshold for zero assignment

        Returns:
            Balanced ternary weights
        """
        # Threshold-based quantization
        # |w| < threshold → 0
        # w >= threshold → +1
        # w <= -threshold → -1

        ternary = torch.zeros_like(weights)
        ternary[weights > threshold] = 1.0
        ternary[weights < -threshold] = -1.0

        return ternary

    def quantize(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weights to balanced ternary (alias for forward without STE).

        Args:
            weights: Float weights (any shape)

        Returns:
            Quantized weights {-1, 0, 1} (same shape)
        """
        return self.forward(weights)

    def quantize_with_ste(self, weights: torch.Tensor) -> torch.Tensor:
        """Quantize weights with straight-through estimator.

        This is the same as forward() but explicitly named for clarity.

        Args:
            weights: Float weights (any shape)

        Returns:
            Quantized weights {-1, 0, 1} with STE gradients
        """
        return self.forward(weights)


    def quantize_to_trytes(self, weights: torch.Tensor) -> BalancedTernaryTensor:
        """Quantize weights to balanced ternary trytes.

        Args:
            weights: Float weights (shape: [...])

        Returns:
            Balanced ternary tensor
        """
        # First quantize to {-1, 0, 1}
        ternary = self.forward(weights)

        # Scale to integer range for tryte encoding
        # For 9 trits: range is -9841 to +9841
        max_value = (3**self.trits_per_tryte) // 2  # 9841 for 9 trits

        # Scale weights to [-max_value, +max_value]
        weight_range = weights.abs().max()
        if weight_range > 0:
            scaled = (ternary / weight_range) * max_value
        else:
            scaled = ternary

        # Encode as trytes
        return tryte_encode(scaled, self.trits_per_tryte)


class BalancedTernaryCompressor:
    """Compressor for storing balanced ternary weights efficiently.

    Memory efficiency:
    - FP16: 2 bytes per weight
    - Balanced ternary: ~1.58 bits per trit (~0.2 bytes per weight)
    - Compression ratio: ~10×
    """

    def __init__(self, trits_per_tryte: int = 9):
        """Initialize compressor.

        Args:
            trits_per_tryte: Trits per tryte
        """
        self.trits_per_tryte = trits_per_tryte

    def compress(self, weights: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compress FP16/FP32 weights to balanced ternary.

        Args:
            weights: Float weights (shape: [...])

        Returns:
            Tuple of (compressed_trits, metadata)
        """
        # Quantize to balanced ternary
        quantizer = BalancedTernaryQuantizer(trits_per_tryte=self.trits_per_tryte)
        bt_tensor = quantizer.quantize_to_trytes(weights)

        # Pack trits efficiently
        # Each trit needs log2(3) ≈ 1.58 bits
        # We can pack 5 trits into 1 byte (2^8 = 256 > 3^5 = 243)
        packed = self._pack_trits(bt_tensor.trits)

        metadata = {
            "original_shape": list(weights.shape),
            "trits_per_tryte": self.trits_per_tryte,
            "original_dtype": str(weights.dtype),
            "weight_scale": weights.abs().max().item(),
        }

        return packed, metadata

    def decompress(self, packed: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Decompress balanced ternary back to float.

        Args:
            packed: Packed trits
            metadata: Compression metadata

        Returns:
            Decompressed float weights
        """
        # Unpack trits
        trits = self._unpack_trits(packed, metadata["original_shape"])

        # Convert to balanced ternary tensor
        bt_tensor = BalancedTernaryTensor(trits, metadata["trits_per_tryte"])

        # Convert to float and scale
        weights = bt_tensor.to_float()
        weights = weights * metadata["weight_scale"]

        return weights

    @staticmethod
    def _pack_trits(trits: torch.Tensor) -> torch.Tensor:
        """Pack trits efficiently (5 trits per byte).

        Args:
            trits: Balanced ternary digits {-1, 0, 1} (shape: [..., num_trits])

        Returns:
            Packed bytes (shape: [..., ceil(num_trits / 5)])
        """
        # Convert {-1, 0, 1} to {0, 1, 2} for packing
        shifted_trits = (trits + 1).to(torch.uint8)  # {0, 1, 2}

        # Pack 5 trits into 1 byte
        # byte = t0 * 3^4 + t1 * 3^3 + t2 * 3^2 + t3 * 3 + t4
        num_trits = shifted_trits.shape[-1]
        num_bytes = (num_trits + 4) // 5

        packed = torch.zeros(
            list(shifted_trits.shape[:-1]) + [num_bytes], dtype=torch.uint8, device=trits.device
        )

        for i in range(num_bytes):
            start_idx = i * 5
            end_idx = min(start_idx + 5, num_trits)

            # Extract 5 trits
            chunk = shifted_trits[..., start_idx:end_idx]

            # Pack into byte
            byte_value = torch.zeros_like(packed[..., i])
            for j, power in enumerate([81, 27, 9, 3, 1]):  # 3^4, 3^3, 3^2, 3, 1
                if j < chunk.shape[-1]:
                    byte_value += chunk[..., j] * power

            packed[..., i] = byte_value

        return packed

    @staticmethod
    def _unpack_trits(packed: torch.Tensor, original_shape: list) -> torch.Tensor:
        """Unpack trits from bytes.

        Args:
            packed: Packed bytes (shape: [..., num_bytes])
            original_shape: Original shape before packing

        Returns:
            Unpacked trits {-1, 0, 1} (shape: original_shape + [num_trits])
        """
        num_trits = (packed.shape[-1] - 1) * 5 + 5  # Upper bound

        unpacked = torch.zeros(original_shape + [num_trits], dtype=torch.int8, device=packed.device)

        for i in range(packed.shape[-1]):
            byte_value = packed[..., i]

            # Unpack 5 trits from byte
            for j, power in enumerate([81, 27, 9, 3, 1]):
                trit_value = byte_value // power
                byte_value = byte_value % power

                start_idx = i * 5 + j
                if start_idx < num_trits:
                    unpacked[..., start_idx] = trit_value

        # Convert {0, 1, 2} back to {-1, 0, 1}
        unpacked = unpacked - 1

        return unpacked

    def compression_ratio(self, original_dtype: torch.dtype = torch.float16) -> float:
        """Calculate compression ratio.

        Args:
            original_dtype: Original data type

        Returns:
            Compression ratio
        """
        # Original bytes per weight
        if original_dtype == torch.float16:
            original_bytes = 2
        elif original_dtype == torch.float32:
            original_bytes = 4
        else:
            original_bytes = 4

        # Balanced ternary: ~1.58 bits per trit
        # For tryte (9 trits): ~14.2 bits ≈ 1.78 bytes
        # With packing (5 trits per byte): ~1.8 bytes per 9 trits
        compressed_bytes = (self.trits_per_tryte + 4) // 5  # Bytes per tryte

        return original_bytes / compressed_bytes
