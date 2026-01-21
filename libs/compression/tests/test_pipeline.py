"""Tests for staged compression pipeline."""

import pytest
import torch

from compression.config import CompressionConfig
from compression.mrl.matryoshka import MatryoshkaEncoder
from compression.pipeline import StagedCompressionPipeline


class TestStagedCompressionPipeline:
    """Test suite for staged compression pipeline."""

    @pytest.fixture
    def config(self):
        """Create compression config."""
        return CompressionConfig(
            mrl_dims=[1024, 512, 256],
            qinco_stages=2,
            rvq_stages=2,
            use_bitnet=False,  # Skip BitNet for unit tests
        )

    @pytest.fixture
    def pipeline_small(self):
        """Create small pipeline for testing."""
        # Create components
        mrl_encoder = MatryoshkaEncoder(
            input_dim=128, output_dim=256, target_dims=[64, 128, 256]
        )

        config = CompressionConfig(
            mrl_dims=[256, 128, 64],
            qinco_stages=2,
            qinco_codebook_size=16,
            rvq_stages=2,
            rvq_codebook_size=16,
            use_bitnet=False,
        )

        return StagedCompressionPipeline(config=config, mrl_encoder=mrl_encoder)

    def test_initialization(self, pipeline_small):
        """Test pipeline initialization."""
        assert pipeline_small.config is not None
        assert pipeline_small.mrl_compressor is not None

    def test_compress(self, pipeline_small):
        """Test compression through all stages."""
        embeddings = torch.randn(4, 256)

        compressed, stage_outputs = pipeline_small.compress(embeddings)

        # Should have MRL output
        assert "mrl" in stage_outputs
        assert stage_outputs["mrl"].shape[0] == 4

        # Compressed output should be smaller than input
        assert compressed.shape[1] <= embeddings.shape[1]

    def test_decompress(self, pipeline_small):
        """Test decompression."""
        embeddings = torch.randn(4, 256)

        compressed, stage_outputs = pipeline_small.compress(embeddings)

        # Decompress
        reconstructed = pipeline_small.decompress(stage_outputs)

        # Should match original shape
        assert reconstructed.shape == embeddings.shape

    def test_compress_decompress_roundtrip(self, pipeline_small):
        """Test compress-decompress preserves information."""
        embeddings = torch.randn(4, 256)

        compressed, stage_outputs = pipeline_small.compress(embeddings)
        reconstructed = pipeline_small.decompress(stage_outputs)

        # MSE should be reasonable
        mse = torch.nn.functional.mse_loss(embeddings, reconstructed)
        assert mse.item() < 0.5, f"Roundtrip MSE {mse.item()} too high"

    def test_fidelity_calculation(self, pipeline_small):
        """Test fidelity calculation."""
        original = torch.randn(8, 256)

        compressed, stage_outputs = pipeline_small.compress(original)
        reconstructed = pipeline_small.decompress(stage_outputs)

        fidelity = pipeline_small.calculate_fidelity(original, reconstructed)

        # Fidelity should be between 0 and 1
        assert 0 <= fidelity <= 1

        # Should be reasonably high
        assert fidelity > 0.5, f"Fidelity {fidelity} too low"

    def test_compression_ratio(self, pipeline_small):
        """Test compression ratio calculation."""
        ratio = pipeline_small.get_compression_ratio()

        # Should have some compression
        assert ratio > 1.0

    def test_batch_processing(self, pipeline_small):
        """Test different batch sizes."""
        batch_sizes = [1, 4, 16]

        for batch_size in batch_sizes:
            embeddings = torch.randn(batch_size, 256)
            compressed, stage_outputs = pipeline_small.compress(embeddings)
            assert compressed.shape[0] == batch_size

    def test_stage_outputs(self, pipeline_small):
        """Test that stage outputs are captured."""
        embeddings = torch.randn(4, 256)

        _, stage_outputs = pipeline_small.compress(embeddings)

        # Should have outputs from all stages
        assert "mrl" in stage_outputs

        # If QINCo is enabled
        if pipeline_small.qinco_compressor is not None:
            assert "qinco_indices" in stage_outputs


class TestPipelineWithRVQ:
    """Test pipeline with RVQ stage."""

    @pytest.fixture
    def pipeline_rvq(self):
        """Create pipeline with RVQ."""
        mrl_encoder = MatryoshkaEncoder(
            input_dim=64, output_dim=128, target_dims=[32, 64, 128]
        )

        config = CompressionConfig(
            mrl_dims=[128, 64, 32],
            qinco_stages=0,  # Disable QINCo
            rvq_stages=3,
            rvq_codebook_size=32,
            use_bitnet=False,
        )

        return StagedCompressionPipeline(config=config, mrl_encoder=mrl_encoder)

    def test_rvq_compression(self, pipeline_rvq):
        """Test RVQ compression stage."""
        embeddings = torch.randn(4, 128)

        compressed, stage_outputs = pipeline_rvq.compress(embeddings)

        # Should have RVQ indices
        if pipeline_rvq.rvq_compressor is not None:
            assert "rvq_indices" in stage_outputs
            indices = stage_outputs["rvq_indices"]
            assert indices.shape[0] == 4  # Batch size
            assert indices.shape[1] == 3  # RVQ stages


class TestPipelineWithQINCo:
    """Test pipeline with QINCo stage."""

    @pytest.fixture
    def pipeline_qinco(self):
        """Create pipeline with QINCo."""
        mrl_encoder = MatryoshkaEncoder(
            input_dim=64, output_dim=128, target_dims=[32, 64, 128]
        )

        config = CompressionConfig(
            mrl_dims=[128, 64],
            qinco_stages=2,
            qinco_codebook_size=16,
            rvq_stages=0,  # Disable RVQ
            use_bitnet=False,
        )

        return StagedCompressionPipeline(config=config, mrl_encoder=mrl_encoder)

    def test_qinco_compression(self, pipeline_qinco):
        """Test QINCo compression stage."""
        embeddings = torch.randn(4, 128)

        compressed, stage_outputs = pipeline_qinco.compress(embeddings)

        # Should have QINCo outputs
        if pipeline_qinco.qinco_compressor is not None:
            assert "qinco_indices" in stage_outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
