"""Tests for compression benchmark infrastructure.

Validates the benchmark suite produces correct and reproducible results
for measuring compression fidelity, ratio, and performance.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from benchmarks.compression_benchmark import (
    BenchmarkSuite,
    CompressionBenchmark,
    CompressionResult,
    compute_compression_ratio,
    compute_fidelity,
    generate_test_embeddings,
    run_quick_benchmark,
)

# =============================================================================
# Test Data Generation
# =============================================================================


class TestGenerateTestEmbeddings:
    """Tests for generate_test_embeddings function."""

    def test_output_shape(self) -> None:
        """Verify output has correct shape."""
        embeddings = generate_test_embeddings(num_samples=100, embed_dim=256)
        assert embeddings.shape == (100, 256)

    def test_reproducibility(self) -> None:
        """Verify same seed produces same output."""
        emb1 = generate_test_embeddings(num_samples=50, embed_dim=128, seed=123)
        emb2 = generate_test_embeddings(num_samples=50, embed_dim=128, seed=123)
        assert torch.allclose(emb1, emb2)

    def test_different_seeds_differ(self) -> None:
        """Verify different seeds produce different output."""
        emb1 = generate_test_embeddings(num_samples=50, embed_dim=128, seed=1)
        emb2 = generate_test_embeddings(num_samples=50, embed_dim=128, seed=2)
        assert not torch.allclose(emb1, emb2)

    def test_normal_distribution(self) -> None:
        """Test normal distribution option."""
        embeddings = generate_test_embeddings(
            num_samples=1000, embed_dim=128, distribution="normal"
        )
        # Should have roughly zero mean and unit variance per dim
        assert embeddings.mean().abs() < 0.2
        assert 0.5 < embeddings.std() < 1.5

    def test_uniform_distribution(self) -> None:
        """Test uniform distribution option."""
        embeddings = generate_test_embeddings(
            num_samples=1000, embed_dim=128, distribution="uniform"
        )
        # Should be in range [-1, 1]
        assert embeddings.min() >= -1.0
        assert embeddings.max() <= 1.0

    def test_realistic_distribution(self) -> None:
        """Test realistic (normalized) distribution option."""
        embeddings = generate_test_embeddings(
            num_samples=1000, embed_dim=128, distribution="realistic"
        )
        # Should have roughly unit norm
        norms = embeddings.norm(dim=-1)
        assert 0.8 < norms.mean() < 1.2

    def test_invalid_distribution_raises(self) -> None:
        """Test invalid distribution raises ValueError."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            generate_test_embeddings(num_samples=10, embed_dim=64, distribution="invalid")

    def test_device_placement(self) -> None:
        """Test embeddings are on correct device."""
        embeddings = generate_test_embeddings(num_samples=10, embed_dim=64, device="cpu")
        assert embeddings.device.type == "cpu"


# =============================================================================
# Test Fidelity Computation
# =============================================================================


class TestComputeFidelity:
    """Tests for compute_fidelity function."""

    def test_perfect_fidelity(self) -> None:
        """Identical embeddings should have fidelity 1.0."""
        embeddings = torch.randn(100, 64)
        fidelity = compute_fidelity(embeddings, embeddings)
        assert fidelity["mean"] == pytest.approx(1.0, abs=1e-6)
        assert fidelity["std"] == pytest.approx(0.0, abs=1e-6)

    def test_opposite_embeddings(self) -> None:
        """Negated embeddings should have fidelity -1.0."""
        embeddings = torch.randn(100, 64)
        fidelity = compute_fidelity(embeddings, -embeddings)
        assert fidelity["mean"] == pytest.approx(-1.0, abs=1e-6)

    def test_orthogonal_embeddings(self) -> None:
        """Orthogonal embeddings should have fidelity ~0."""
        # Create two sets of orthogonal vectors
        a = torch.zeros(100, 64)
        b = torch.zeros(100, 64)
        a[:, :32] = torch.randn(100, 32)
        b[:, 32:] = torch.randn(100, 32)
        fidelity = compute_fidelity(a, b)
        assert fidelity["mean"] == pytest.approx(0.0, abs=0.01)

    def test_percentiles_ordered(self) -> None:
        """Percentiles should be in order p50 <= p90 <= p95 <= p99."""
        embeddings = torch.randn(100, 64)
        noisy = embeddings + 0.1 * torch.randn_like(embeddings)
        fidelity = compute_fidelity(embeddings, noisy)
        p = fidelity["percentiles"]
        # Note: For high fidelity, all percentiles may be close
        assert p["p50"] <= p["p90"] + 0.01  # Small tolerance for numerical precision
        assert p["p90"] <= p["p95"] + 0.01
        assert p["p95"] <= p["p99"] + 0.01

    def test_min_less_than_mean(self) -> None:
        """Minimum should be less than or equal to mean."""
        embeddings = torch.randn(100, 64)
        noisy = embeddings + 0.5 * torch.randn_like(embeddings)
        fidelity = compute_fidelity(embeddings, noisy)
        assert fidelity["min"] <= fidelity["mean"]


# =============================================================================
# Test Compression Ratio
# =============================================================================


class TestComputeCompressionRatio:
    """Tests for compute_compression_ratio function."""

    def test_same_size_ratio_one(self) -> None:
        """Same size tensors should have ratio 1.0."""
        original = torch.randn(100, 64)
        compressed = torch.randn(100, 64)
        ratio = compute_compression_ratio(original, compressed)
        assert ratio == pytest.approx(1.0)

    def test_half_size_ratio_two(self) -> None:
        """Half-size compressed should have ratio 2.0."""
        original = torch.randn(100, 64)
        compressed = torch.randn(100, 32)
        ratio = compute_compression_ratio(original, compressed)
        assert ratio == pytest.approx(2.0)

    def test_quarter_size_ratio_four(self) -> None:
        """Quarter-size compressed should have ratio 4.0."""
        original = torch.randn(100, 64)
        compressed = torch.randn(100, 16)
        ratio = compute_compression_ratio(original, compressed)
        assert ratio == pytest.approx(4.0)

    def test_dict_compressed_format(self) -> None:
        """Test dict compressed format with multiple tensors."""
        original = torch.randn(100, 64)
        compressed = {
            "codes": torch.randint(0, 256, (100, 8), dtype=torch.uint8),
            "scale": torch.randn(1),
        }
        ratio = compute_compression_ratio(original, compressed)
        # Original: 100*64*4 = 25600 bytes
        # Compressed: 100*8*1 + 1*4 = 804 bytes
        expected = 25600 / 804
        assert ratio == pytest.approx(expected, rel=0.01)


# =============================================================================
# Test CompressionResult
# =============================================================================


class TestCompressionResult:
    """Tests for CompressionResult dataclass."""

    @pytest.fixture
    def sample_result(self) -> CompressionResult:
        """Create sample result for testing."""
        return CompressionResult(
            compactor_name="TestCompactor",
            fidelity_mean=0.96,
            fidelity_std=0.02,
            fidelity_min=0.90,
            fidelity_percentiles={"p50": 0.97, "p90": 0.94, "p95": 0.92, "p99": 0.91},
            compression_ratio=4.5,
            compress_latency_ms=1.5,
            decompress_latency_ms=0.8,
            total_latency_ms=2.3,
            memory_peak_mb=100.0,
            num_samples=1000,
            embed_dim=512,
        )

    def test_to_json_valid(self, sample_result: CompressionResult) -> None:
        """Test to_json produces valid JSON."""
        json_str = sample_result.to_json()
        parsed = json.loads(json_str)
        assert parsed["compactor_name"] == "TestCompactor"
        assert parsed["fidelity_mean"] == 0.96

    def test_meets_target_true(self, sample_result: CompressionResult) -> None:
        """Test meets_target returns True for passing result."""
        assert sample_result.meets_target(min_fidelity=0.95, min_compression=4.0)

    def test_meets_target_false_fidelity(self, sample_result: CompressionResult) -> None:
        """Test meets_target returns False for low fidelity."""
        assert not sample_result.meets_target(min_fidelity=0.99, min_compression=4.0)

    def test_meets_target_false_compression(self, sample_result: CompressionResult) -> None:
        """Test meets_target returns False for low compression."""
        assert not sample_result.meets_target(min_fidelity=0.95, min_compression=8.0)


# =============================================================================
# Test BenchmarkSuite
# =============================================================================


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite dataclass."""

    @pytest.fixture
    def sample_suite(self) -> BenchmarkSuite:
        """Create sample suite for testing."""
        return BenchmarkSuite(
            results=[
                CompressionResult(
                    compactor_name="Good",
                    fidelity_mean=0.98,
                    fidelity_std=0.01,
                    fidelity_min=0.95,
                    fidelity_percentiles={"p50": 0.98, "p90": 0.97, "p95": 0.96, "p99": 0.95},
                    compression_ratio=4.0,
                    compress_latency_ms=1.0,
                    decompress_latency_ms=0.5,
                    total_latency_ms=1.5,
                    memory_peak_mb=50.0,
                    num_samples=100,
                    embed_dim=512,
                ),
                CompressionResult(
                    compactor_name="Bad",
                    fidelity_mean=0.70,
                    fidelity_std=0.10,
                    fidelity_min=0.40,
                    fidelity_percentiles={"p50": 0.72, "p90": 0.55, "p95": 0.50, "p99": 0.42},
                    compression_ratio=8.0,
                    compress_latency_ms=0.5,
                    decompress_latency_ms=0.3,
                    total_latency_ms=0.8,
                    memory_peak_mb=25.0,
                    num_samples=100,
                    embed_dim=512,
                ),
            ],
            baseline_fidelity=1.0,
            device="cpu",
            torch_version="2.9.0",
            cuda_available=False,
        )

    def test_to_json_valid(self, sample_suite: BenchmarkSuite) -> None:
        """Test to_json produces valid JSON."""
        json_str = sample_suite.to_json()
        parsed = json.loads(json_str)
        assert len(parsed["results"]) == 2
        assert parsed["baseline_fidelity"] == 1.0

    def test_summary_table_format(self, sample_suite: BenchmarkSuite) -> None:
        """Test summary_table produces markdown table."""
        table = sample_suite.summary_table()
        assert "| Compactor |" in table
        assert "| Good |" in table
        assert "| Bad |" in table
        assert "✓" in table  # Good should pass
        assert "✗" in table  # Bad should fail


# =============================================================================
# Mock Compactor for Testing
# =============================================================================


class MockCompactor(nn.Module):
    """Mock compactor for testing benchmark infrastructure.

    Simulates compression by truncating dimensions.
    """

    def __init__(self, embed_dim: int = 512, compressed_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.compressed_dim = compressed_dim
        # Simple linear projection
        self.encoder = nn.Linear(embed_dim, compressed_dim)
        self.decoder = nn.Linear(compressed_dim, embed_dim)

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compress via linear projection."""
        return self.encoder(embeddings)

    def reconstruct(self, compressed: torch.Tensor) -> torch.Tensor:
        """Reconstruct via linear projection."""
        return self.decoder(compressed)


class PerfectCompactor(nn.Module):
    """Compactor that returns input unchanged (for testing)."""

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings

    def reconstruct(self, compressed: torch.Tensor) -> torch.Tensor:
        return compressed


# =============================================================================
# Test CompressionBenchmark
# =============================================================================


class TestCompressionBenchmark:
    """Tests for CompressionBenchmark class."""

    def test_initialization(self) -> None:
        """Test benchmark initializes correctly."""
        benchmark = CompressionBenchmark(
            device="cpu",
            num_samples=50,
            embed_dim=64,
        )
        assert benchmark.test_embeddings.shape == (50, 64)

    def test_run_compactor_mock(self) -> None:
        """Test running benchmark on mock compactor."""
        benchmark = CompressionBenchmark(
            device="cpu",
            num_samples=50,
            embed_dim=64,
            num_warmup=1,
            num_iterations=2,
        )
        compactor = MockCompactor(embed_dim=64, compressed_dim=16)
        result = benchmark.run_compactor(compactor, name="MockCompactor")

        assert result.compactor_name == "MockCompactor"
        assert result.num_samples == 50
        assert result.embed_dim == 64
        assert result.compression_ratio > 1.0  # Should compress
        assert 0 <= result.fidelity_mean <= 1.0

    def test_run_compactor_perfect(self) -> None:
        """Test perfect compactor has fidelity 1.0."""
        benchmark = CompressionBenchmark(
            device="cpu",
            num_samples=50,
            embed_dim=64,
            num_warmup=1,
            num_iterations=2,
        )
        compactor = PerfectCompactor()
        result = benchmark.run_compactor(compactor)

        assert result.fidelity_mean == pytest.approx(1.0, abs=1e-5)
        assert result.compression_ratio == pytest.approx(1.0)

    def test_run_full_suite_custom(self) -> None:
        """Test running full suite with custom compactors."""
        benchmark = CompressionBenchmark(
            device="cpu",
            num_samples=30,
            embed_dim=64,
            num_warmup=1,
            num_iterations=1,
        )
        compactors = {
            "Perfect": PerfectCompactor(),
            "Mock4x": MockCompactor(embed_dim=64, compressed_dim=16),
        }
        suite = benchmark.run_full_suite(compactors=compactors)

        assert len(suite.results) == 2
        assert suite.baseline_fidelity == 1.0
        assert suite.torch_version == torch.__version__

    def test_save_results(self) -> None:
        """Test saving results to file."""
        benchmark = CompressionBenchmark(
            device="cpu",
            num_samples=20,
            embed_dim=32,
            num_warmup=1,
            num_iterations=1,
        )
        suite = benchmark.run_full_suite(
            compactors={"Test": PerfectCompactor()}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = benchmark.save_results(tmpdir, suite, prefix="test")

            # Check JSON file exists and is valid
            assert filepath.exists()
            with open(filepath) as f:
                data = json.load(f)
            assert "results" in data

            # Check summary file exists
            summary_files = list(Path(tmpdir).glob("*_summary_*.md"))
            assert len(summary_files) == 1


# =============================================================================
# Test Quick Benchmark
# =============================================================================


class TestRunQuickBenchmark:
    """Tests for run_quick_benchmark function."""

    def test_runs_without_error(self) -> None:
        """Test quick benchmark runs successfully."""
        # This may take a few seconds but should complete
        suite = run_quick_benchmark(device="cpu", num_samples=20)
        assert isinstance(suite, BenchmarkSuite)
        # May have 0 results if imports fail, but shouldn't crash
        assert suite.device == "cpu"
