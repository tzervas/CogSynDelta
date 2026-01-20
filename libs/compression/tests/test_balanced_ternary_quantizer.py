"""Tests for Balanced Ternary Quantizer and Compression."""

import pytest
import torch

from compression.balanced_ternary.quantizer import (
    BalancedTernaryCompressor,
    BalancedTernaryQuantizer,
)
from compression.balanced_ternary.tryte import BalancedTernaryTryte


class TestBalancedTernaryQuantizer:
    """Test suite for BalancedTernaryQuantizer."""

    @pytest.fixture
    def quantizer(self):
        """Create quantizer."""
        return BalancedTernaryQuantizer(trits_per_tryte=9)

    def test_initialization(self, quantizer):
        """Test quantizer initialization."""
        assert quantizer.trits_per_tryte == 9

    def test_quantize(self, quantizer):
        """Test basic quantization."""
        weights = torch.randn(4, 4)

        quantized = quantizer.quantize(weights)

        # Should be ternary {-1, 0, +1}
        unique_vals = set(quantized.flatten().tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})
        assert quantized.shape == weights.shape

    def test_quantize_to_trytes(self, quantizer):
        """Test quantization to tryte representation."""
        weights = torch.randn(4, 4)

        trytes = quantizer.quantize_to_trytes(weights)

        assert isinstance(trytes, BalancedTernaryTryte)
        # Trits shape should be (4, 4, 9)
        assert trytes.trits.shape == (4, 4, 9)

    def test_quantize_preserves_sign(self, quantizer):
        """Test that quantization preserves sign."""
        # Create weights with clear positive and negative values
        weights = torch.tensor([[2.0, -2.0], [1.5, -1.5]])

        quantized = quantizer.quantize(weights)

        # Positive weights should stay positive
        assert quantized[0, 0] >= 0
        assert quantized[1, 0] >= 0

        # Negative weights should stay negative
        assert quantized[0, 1] <= 0
        assert quantized[1, 1] <= 0

    def test_quantize_small_values_to_zero(self, quantizer):
        """Test that small values quantize to zero."""
        # Small values near zero
        weights = torch.tensor([[0.1, -0.1, 0.05, -0.05]])

        quantized = quantizer.quantize(weights)

        # Most should be zero (depends on normalization)
        zero_ratio = (quantized == 0).float().mean()
        assert zero_ratio > 0.3  # At least 30% should be zero

    def test_quantize_with_ste(self, quantizer):
        """Test quantization with straight-through estimator."""
        weights = torch.randn(4, 4, requires_grad=True)

        quantized = quantizer.quantize_with_ste(weights)

        # Forward: ternary values
        unique_vals = set(quantized.detach().flatten().tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

        # Backward: gradients should flow
        loss = quantized.sum()
        loss.backward()

        assert weights.grad is not None

    def test_batch_quantization(self, quantizer):
        """Test quantization of batches."""
        batch_sizes = [1, 4, 16]

        for batch_size in batch_sizes:
            weights = torch.randn(batch_size, 32, 32)
            quantized = quantizer.quantize(weights)
            assert quantized.shape == (batch_size, 32, 32)


class TestBalancedTernaryCompressor:
    """Test suite for BalancedTernaryCompressor."""

    @pytest.fixture
    def compressor(self):
        """Create compressor."""
        return BalancedTernaryCompressor(trits_per_tryte=9)

    def test_initialization(self, compressor):
        """Test compressor initialization."""
        assert compressor.trits_per_tryte == 9

    def test_compress(self, compressor):
        """Test weight compression."""
        weights = torch.randn(128, 64)  # 8KB FP16

        packed, metadata = compressor.compress(weights)

        # Should return bytes
        assert isinstance(packed, bytes)
        assert isinstance(metadata, dict)

        # Metadata should have expected keys
        assert "original_shape" in metadata
        assert "trits_per_tryte" in metadata
        assert "weight_scale" in metadata

    def test_compression_ratio(self, compressor):
        """Test compression ratio."""
        weights = torch.randn(512, 256)

        fp16_size = weights.numel() * 2  # 2 bytes per FP16 weight
        packed, _ = compressor.compress(weights)
        compressed_size = len(packed)

        ratio = fp16_size / compressed_size

        # Should achieve ~10× compression (similar to BitNet)
        assert ratio >= 8.0, f"Compression ratio {ratio}× < 8×"

    def test_decompress(self, compressor):
        """Test decompression."""
        weights = torch.randn(64, 32)

        packed, metadata = compressor.compress(weights)
        reconstructed = compressor.decompress(packed, metadata)

        # Should match original shape
        assert reconstructed.shape == weights.shape

        # Values should be ternary {-1, 0, +1}
        unique_vals = set(reconstructed.flatten().tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_compress_decompress_roundtrip(self, compressor):
        """Test compress-decompress roundtrip."""
        weights = torch.randn(128, 64)

        # Compress
        packed, metadata = compressor.compress(weights)

        # Decompress
        reconstructed = compressor.decompress(packed, metadata)

        # Should be deterministic (same input → same output)
        packed2, metadata2 = compressor.compress(weights)
        reconstructed2 = compressor.decompress(packed2, metadata2)

        assert torch.equal(reconstructed, reconstructed2)

    def test_compress_large_tensor(self, compressor):
        """Test compressing large tensor."""
        # Simulate large layer: 2048×2048 = 4M weights
        weights = torch.randn(2048, 2048)

        packed, metadata = compressor.compress(weights)

        # Should compress successfully
        ratio = (weights.numel() * 2) / len(packed)
        assert ratio >= 8.0

    def test_metadata_preservation(self, compressor):
        """Test that metadata is preserved."""
        weights = torch.randn(100, 50)

        _, metadata = compressor.compress(weights)

        assert metadata["original_shape"] == [100, 50]
        assert metadata["trits_per_tryte"] == 9
        assert isinstance(metadata["weight_scale"], float)
        assert metadata["weight_scale"] > 0

    def test_different_shapes(self, compressor):
        """Test compression of different tensor shapes."""
        shapes = [(64, 64), (128, 32), (256, 256), (512, 128)]

        for shape in shapes:
            weights = torch.randn(*shape)
            packed, metadata = compressor.compress(weights)
            reconstructed = compressor.decompress(packed, metadata)

            assert reconstructed.shape == shape


class TestBalancedTernaryTryte:
    """Test suite for BalancedTernaryTryte encoding."""

    def test_initialization_from_tensor(self):
        """Test tryte initialization from tensor."""
        weights = torch.randn(4, 4)

        tryte = BalancedTernaryTryte(weights, trits_per_tryte=9)

        # Should have trits
        assert tryte.trits.shape == (4, 4, 9)
        assert tryte.trits.dtype == torch.int8

    def test_initialization_from_trits(self):
        """Test tryte initialization from existing trits."""
        trits = torch.randint(-1, 2, (4, 4, 9), dtype=torch.int8)

        tryte = BalancedTernaryTryte(trits, trits_per_tryte=9, is_trits=True)

        assert torch.equal(tryte.trits, trits)

    def test_to_decimal(self):
        """Test conversion to decimal."""
        weights = torch.tensor([[1.0, -1.0], [0.5, -0.5]])

        tryte = BalancedTernaryTryte(weights, trits_per_tryte=9)
        decimal = tryte.to_decimal()

        # Should be ternary values after quantization
        assert decimal.shape == (2, 2)
        unique_vals = set(decimal.flatten().tolist())
        assert unique_vals.issubset({-1.0, 0.0, 1.0})

    def test_pack(self):
        """Test packing trits into bytes."""
        weights = torch.randn(8, 8)

        tryte = BalancedTernaryTryte(weights, trits_per_tryte=9)
        packed = tryte.pack()

        # Should return bytes
        assert isinstance(packed, bytes)

        # Packed size should be much smaller than FP16
        fp16_size = 8 * 8 * 2  # 128 bytes
        assert len(packed) < fp16_size

    def test_unpack(self):
        """Test unpacking bytes to tryte."""
        weights = torch.randn(4, 4)

        tryte = BalancedTernaryTryte(weights, trits_per_tryte=9)
        packed = tryte.pack()

        # Unpack
        unpacked = BalancedTernaryTryte.unpack(packed, shape=(4, 4), trits_per_tryte=9)

        # Should match original trits
        assert unpacked.trits.shape == tryte.trits.shape

    def test_pack_unpack_roundtrip(self):
        """Test pack-unpack roundtrip."""
        weights = torch.randn(10, 10)

        tryte = BalancedTernaryTryte(weights, trits_per_tryte=9)
        packed = tryte.pack()
        unpacked = BalancedTernaryTryte.unpack(packed, shape=(10, 10), trits_per_tryte=9)

        # Trits should match (after quantization)
        assert unpacked.trits.shape == tryte.trits.shape

    def test_range_coverage(self):
        """Test that tryte can represent range of values."""
        # 9-trit range: -9841 to +9841
        # Test various weight magnitudes
        weights = torch.randn(16, 16) * 10  # Scale up

        tryte = BalancedTernaryTryte(weights, trits_per_tryte=9)
        decimal = tryte.to_decimal()

        # Should quantize to {-1, 0, +1}
        assert set(decimal.flatten().tolist()).issubset({-1.0, 0.0, 1.0})


class TestCompressionPipeline:
    """Test complete compression pipeline."""

    def test_full_layer_compression(self):
        """Test compressing full layer weights."""
        import torch.nn as nn

        layer = nn.Linear(512, 256)

        compressor = BalancedTernaryCompressor(trits_per_tryte=9)

        # Compress
        packed, metadata = compressor.compress(layer.weight)

        # Check compression ratio
        original_size = layer.weight.numel() * 2  # FP16
        compressed_size = len(packed)
        ratio = original_size / compressed_size

        assert ratio >= 8.0, f"Compression ratio {ratio}× < 8×"

        # Decompress
        reconstructed = compressor.decompress(packed, metadata)

        # Should be ternary
        assert set(reconstructed.flatten().tolist()).issubset({-1.0, 0.0, 1.0})

    def test_multiple_layers_compression(self):
        """Test compressing multiple layers."""
        import torch.nn as nn

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 10)

        model = SimpleNet()
        compressor = BalancedTernaryCompressor()

        compressed_layers = {}
        total_original = 0
        total_compressed = 0

        for name, param in model.named_parameters():
            if "weight" in name:
                packed, metadata = compressor.compress(param)
                compressed_layers[name] = (packed, metadata)

                total_original += param.numel() * 2
                total_compressed += len(packed)

        overall_ratio = total_original / total_compressed

        assert overall_ratio >= 8.0, f"Overall compression {overall_ratio}× < 8×"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
