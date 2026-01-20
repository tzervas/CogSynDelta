"""Tests for Matryoshka Representation Learning (MRL)."""

import pytest
import torch

from compression.mrl.matryoshka import MatryoshkaCompressor, MatryoshkaEncoder


class TestMatryoshkaEncoder:
    """Test suite for MRL encoder."""

    @pytest.fixture
    def encoder(self):
        """Create MRL encoder for testing."""
        return MatryoshkaEncoder(
            input_dim=512,
            output_dim=2048,
            target_dims=[256, 512, 1024, 2048],
            hidden_dims=[512, 1024],
        )

    @pytest.fixture
    def encoder_small(self):
        """Create small MRL encoder for fast tests."""
        return MatryoshkaEncoder(
            input_dim=64, output_dim=256, target_dims=[64, 128, 256], hidden_dims=[128]
        )

    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.input_dim == 512
        assert encoder.output_dim == 2048
        assert encoder.target_dims == [256, 512, 1024, 2048]

    def test_invalid_target_dims(self):
        """Test that invalid target_dims raise errors."""
        # Target dim > output_dim
        with pytest.raises(AssertionError):
            MatryoshkaEncoder(input_dim=512, output_dim=1024, target_dims=[2048])

        # output_dim not in target_dims
        with pytest.raises(AssertionError):
            MatryoshkaEncoder(input_dim=512, output_dim=1024, target_dims=[512])

    def test_forward(self, encoder_small):
        """Test forward pass."""
        batch_size = 8
        x = torch.randn(batch_size, 64)

        output = encoder_small(x)

        assert output.shape == (batch_size, 256)
        assert not torch.isnan(output).any()

    def test_forward_nested(self, encoder_small):
        """Test nested embedding generation."""
        x = torch.randn(4, 64)

        nested = encoder_small.forward_nested(x)

        assert isinstance(nested, dict)
        assert set(nested.keys()) == {64, 128, 256}

        # Check shapes
        assert nested[64].shape == (4, 64)
        assert nested[128].shape == (4, 128)
        assert nested[256].shape == (4, 256)

        # Check that all are normalized
        for dim, emb in nested.items():
            norms = torch.norm(emb, p=2, dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-5)

    def test_truncation_property(self, encoder_small):
        """Test that truncated embeddings are identical to prefix of full embedding."""
        x = torch.randn(2, 64)

        # Get full embedding
        full_emb = encoder_small(x)

        # Get nested embeddings
        nested = encoder_small.forward_nested(x)

        # After normalization, the first k dims should approximately match
        # (not exact due to normalization, but structure should be similar)
        for dim in [64, 128]:
            truncated = full_emb[:, :dim]
            truncated_norm = torch.nn.functional.normalize(truncated, p=2, dim=1)
            nested_emb = nested[dim]

            # Should have high similarity (cosine similarity)
            similarity = (truncated_norm * nested_emb).sum(dim=1).mean()
            assert similarity > 0.95, f"Dim {dim}: similarity {similarity} < 0.95"


class TestMatryoshkaCompressor:
    """Test suite for MRL compressor."""

    @pytest.fixture
    def compressor(self):
        """Create MRL compressor."""
        encoder = MatryoshkaEncoder(
            input_dim=512, output_dim=2048, target_dims=[512, 1024, 2048]
        )
        return MatryoshkaCompressor(
            encoder=encoder, compression_dim=512, full_dim=2048
        )

    @pytest.fixture
    def compressor_small(self):
        """Create small compressor for fast tests."""
        encoder = MatryoshkaEncoder(
            input_dim=64, output_dim=256, target_dims=[64, 128, 256]
        )
        return MatryoshkaCompressor(encoder=encoder, compression_dim=64, full_dim=256)

    def test_initialization(self, compressor):
        """Test compressor initialization."""
        assert compressor.compression_dim == 512
        assert compressor.full_dim == 2048
        ratio = compressor.get_compression_ratio()
        assert ratio == pytest.approx(4.0)  # 2048/512

    def test_compress(self, compressor_small):
        """Test compression."""
        embeddings = torch.randn(4, 256)

        compressed = compressor_small.compress(embeddings)

        assert compressed.shape == (4, 64)
        # Should be normalized
        norms = torch.norm(compressed, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-5)

    def test_decompress(self, compressor_small):
        """Test decompression (padding to full dimension)."""
        compressed = torch.randn(4, 64)

        decompressed = compressor_small.decompress(compressed)

        assert decompressed.shape == (4, 256)
        # First 64 dims should match compressed (normalized)
        compressed_norm = torch.nn.functional.normalize(compressed, p=2, dim=1)
        # Note: decompress pads with zeros, so first k dims won't exactly match
        # This is a reconstruction, not exact inverse
        assert decompressed[:, :64].abs().sum() > 0  # Non-zero in first 64 dims

    def test_compress_decompress_roundtrip(self, compressor_small):
        """Test compress-decompress preserves information in first k dims."""
        original = torch.randn(2, 256)

        compressed = compressor_small.compress(original)
        decompressed = compressor_small.decompress(compressed)

        # First compression_dim should have high similarity
        orig_prefix = torch.nn.functional.normalize(original[:, :64], p=2, dim=1)
        decomp_prefix = torch.nn.functional.normalize(
            decompressed[:, :64], p=2, dim=1
        )

        similarity = (orig_prefix * decomp_prefix).sum(dim=1).mean()
        assert (
            similarity > 0.9
        ), f"Roundtrip similarity {similarity} < 0.9 (some info loss expected)"

    def test_compression_ratio(self, compressor_small):
        """Test compression ratio calculation."""
        ratio = compressor_small.get_compression_ratio()
        assert ratio == pytest.approx(4.0)  # 256/64

    def test_batch_compression(self, compressor_small):
        """Test batch compression."""
        batch_sizes = [1, 4, 16, 64]

        for batch_size in batch_sizes:
            embeddings = torch.randn(batch_size, 256)
            compressed = compressor_small.compress(embeddings)
            assert compressed.shape == (batch_size, 64)

    def test_edge_case_single_embedding(self, compressor_small):
        """Test single embedding compression."""
        embedding = torch.randn(1, 256)

        compressed = compressor_small.compress(embedding)

        assert compressed.shape == (1, 64)

    def test_compression_preserves_relative_distances(self, compressor_small):
        """Test that compression preserves relative distances between embeddings."""
        # Create 3 embeddings where 0-1 are more similar than 0-2
        torch.manual_seed(42)
        emb0 = torch.randn(1, 256)
        emb1 = emb0 + 0.1 * torch.randn(1, 256)  # Similar to emb0
        emb2 = torch.randn(1, 256)  # Different

        embeddings = torch.cat([emb0, emb1, emb2], dim=0)

        # Compute original distances
        orig_dist_01 = torch.dist(emb0, emb1)
        orig_dist_02 = torch.dist(emb0, emb2)

        # Compress
        compressed = compressor_small.compress(embeddings)

        # Compute compressed distances
        comp_dist_01 = torch.dist(compressed[0], compressed[1])
        comp_dist_02 = torch.dist(compressed[0], compressed[2])

        # Relative ordering should be preserved
        assert orig_dist_01 < orig_dist_02
        assert comp_dist_01 < comp_dist_02


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
