"""Compression benchmark suite for CogSynDelta memory compactors.

This module provides a standardized evaluation framework for testing semantic
embedding compression techniques. It measures fidelity, compression ratio,
latency, and memory usage across multiple compactor implementations.

Why this benchmark suite:
    ADR-0008 establishes the target of ≥0.95 fidelity at 10x+ compression.
    This suite provides reproducible measurements to track progress toward
    that goal and compare different compression strategies.

Key metrics:
    - Fidelity: Cosine similarity between original and reconstructed embeddings
    - Compression Ratio: Original size / compressed size
    - Latency: Time for compress + decompress operations
    - Memory: Peak GPU/CPU memory usage during operations

Example:
    >>> from benchmarks.compression_benchmark import CompressionBenchmark
    >>> benchmark = CompressionBenchmark(device="cuda")
    >>> results = benchmark.run_full_suite()
    >>> benchmark.save_results("benchmark_results/compression/")
"""

from __future__ import annotations

import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    pass


__all__ = [
    "CompactorAdapter",
    "CompactorProtocol",
    "CompressionBenchmark",
    "CompressionResult",
    "generate_test_embeddings",
    "run_quick_benchmark",
]


# =============================================================================
# Protocols and Types
# =============================================================================


@runtime_checkable
class CompactorProtocol(Protocol):
    """Protocol for compactor classes compatible with this benchmark.

    Compactors must implement compress() and reconstruct() methods with
    standard signatures for benchmark compatibility.
    """

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor | dict[str, Any]:
        """Compress embeddings to a compact representation."""
        ...

    def reconstruct(
        self, compressed: torch.Tensor | dict[str, Any]
    ) -> torch.Tensor:
        """Reconstruct embeddings from compressed representation."""
        ...


class CompactorAdapter:
    """Adapter to normalize different compactor interfaces.

    CogSynDelta compactors use 'compact()' method, while the benchmark
    protocol uses 'compress()'. This adapter bridges the two.

    Why an adapter:
        Different compactors may have evolved with different method names.
        The adapter allows benchmarking without modifying original code.
    """

    def __init__(self, compactor: nn.Module) -> None:
        """Initialize adapter with a compactor module.

        Args:
            compactor: A compactor module (may use compact() or compress()).
        """
        self._compactor = compactor
        self._has_compress = hasattr(compactor, "compress")
        self._has_compact = hasattr(compactor, "compact")
        self._has_encode = hasattr(compactor, "encode")

    def compress(self, embeddings: torch.Tensor) -> torch.Tensor | dict[str, Any]:
        """Compress embeddings using the compactor's native method."""
        if self._has_compress:
            return self._compactor.compress(embeddings)  # type: ignore[union-attr]
        if self._has_compact:
            return self._compactor.compact(embeddings)  # type: ignore[union-attr]
        if self._has_encode:
            return self._compactor.encode(embeddings)  # type: ignore[union-attr]
        msg = f"Compactor {type(self._compactor).__name__} has no compress/compact/encode method"
        raise AttributeError(msg)

    def reconstruct(
        self, compressed: torch.Tensor | dict[str, Any]
    ) -> torch.Tensor:
        """Reconstruct embeddings from compressed representation."""
        return self._compactor.reconstruct(compressed)  # type: ignore[union-attr]

    def eval(self) -> "CompactorAdapter":
        """Set compactor to eval mode."""
        self._compactor.eval()
        return self

    def to(self, device: str | torch.device) -> "CompactorAdapter":
        """Move compactor to device."""
        self._compactor.to(device)
        return self


# =============================================================================
# Result Data Classes
# =============================================================================


@dataclass
class CompressionResult:
    """Results from a single compactor benchmark run.

    Attributes:
        compactor_name: Name of the compactor being tested.
        fidelity_mean: Mean cosine similarity across all samples.
        fidelity_std: Standard deviation of cosine similarity.
        fidelity_min: Minimum fidelity observed.
        fidelity_percentiles: Dict with p50, p90, p95, p99 fidelity values.
        compression_ratio: Ratio of original to compressed size.
        compress_latency_ms: Mean compression time in milliseconds.
        decompress_latency_ms: Mean decompression time in milliseconds.
        total_latency_ms: Mean total (compress + decompress) time.
        memory_peak_mb: Peak memory usage in megabytes.
        num_samples: Number of samples tested.
        embed_dim: Embedding dimension.
        timestamp: ISO 8601 timestamp of benchmark run.
        metadata: Additional metadata about the run.
    """

    compactor_name: str
    fidelity_mean: float
    fidelity_std: float
    fidelity_min: float
    fidelity_percentiles: dict[str, float]
    compression_ratio: float
    compress_latency_ms: float
    decompress_latency_ms: float
    total_latency_ms: float
    memory_peak_mb: float
    num_samples: int
    embed_dim: int
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert result to JSON string.

        Returns:
            JSON string representation of the result.
        """
        return json.dumps(asdict(self), indent=2, default=str)

    def meets_target(
        self,
        min_fidelity: float = 0.95,
        min_compression: float = 4.0,
    ) -> bool:
        """Check if result meets ADR-0008 targets.

        Args:
            min_fidelity: Minimum required fidelity (default 0.95 per ADR).
            min_compression: Minimum compression ratio (default 4x).

        Returns:
            True if both targets are met.
        """
        return self.fidelity_mean >= min_fidelity and self.compression_ratio >= min_compression


@dataclass
class BenchmarkSuite:
    """Results from a full benchmark suite run.

    Attributes:
        results: List of individual compactor results.
        baseline_fidelity: Fidelity of uncompressed baseline (should be 1.0).
        device: Device used for benchmarking.
        torch_version: PyTorch version.
        cuda_available: Whether CUDA was available.
        timestamp: ISO 8601 timestamp of suite run.
    """

    results: list[CompressionResult]
    baseline_fidelity: float
    device: str
    torch_version: str
    cuda_available: bool
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_json(self) -> str:
        """Convert suite results to JSON string."""
        data = {
            "results": [asdict(r) for r in self.results],
            "baseline_fidelity": self.baseline_fidelity,
            "device": self.device,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "timestamp": self.timestamp,
        }
        return json.dumps(data, indent=2, default=str)

    def summary_table(self) -> str:
        """Generate markdown summary table.

        Returns:
            Markdown-formatted table of results.
        """
        lines = [
            "| Compactor | Fidelity | Compression | Latency (ms) | Target Met |",
            "|-----------|----------|-------------|--------------|------------|",
        ]
        for r in self.results:
            met = "✓" if r.meets_target() else "✗"
            lines.append(
                f"| {r.compactor_name} | {r.fidelity_mean:.4f} ± {r.fidelity_std:.4f} | "
                f"{r.compression_ratio:.2f}x | {r.total_latency_ms:.2f} | {met} |"
            )
        return "\n".join(lines)


# =============================================================================
# Utility Functions
# =============================================================================


def generate_test_embeddings(
    num_samples: int = 1000,
    embed_dim: int = 512,
    device: str = "cpu",
    seed: int = 42,
    distribution: str = "normal",
) -> torch.Tensor:
    """Generate test embeddings for benchmarking.

    Creates synthetic embeddings that approximate the distribution of
    real semantic embeddings from neural networks.

    Args:
        num_samples: Number of embedding vectors to generate.
        embed_dim: Dimension of each embedding.
        device: Device to place tensors on.
        seed: Random seed for reproducibility.
        distribution: Type of distribution ("normal", "uniform", "realistic").

    Returns:
        Tensor of shape [num_samples, embed_dim].

    Why "realistic" distribution:
        Real semantic embeddings from transformers tend to have a specific
        structure: roughly unit norm with concentrated variance. The
        "realistic" option simulates this better than pure normal/uniform.
    """
    torch.manual_seed(seed)

    if distribution == "normal":
        embeddings = torch.randn(num_samples, embed_dim, device=device)
    elif distribution == "uniform":
        embeddings = torch.rand(num_samples, embed_dim, device=device) * 2 - 1
    elif distribution == "realistic":
        # Simulate transformer embedding distribution
        # Start with normal, apply layernorm-like normalization
        embeddings = torch.randn(num_samples, embed_dim, device=device)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        # Add slight variance in norm (real embeddings aren't perfectly normalized)
        norm_scale = 1.0 + 0.1 * torch.randn(num_samples, 1, device=device)
        embeddings = embeddings * norm_scale
    else:
        msg = f"Unknown distribution: {distribution}"
        raise ValueError(msg)

    return embeddings


def compute_fidelity(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict[str, float]:
    """Compute fidelity metrics between original and reconstructed embeddings.

    Args:
        original: Original embeddings [batch, dim].
        reconstructed: Reconstructed embeddings [batch, dim].

    Returns:
        Dict with mean, std, min, and percentile fidelity values.
    """
    # Cosine similarity per sample
    similarities = F.cosine_similarity(original, reconstructed, dim=-1)
    sim_list = similarities.cpu().tolist()

    # Compute statistics
    mean_fid = statistics.mean(sim_list)
    std_fid = statistics.stdev(sim_list) if len(sim_list) > 1 else 0.0
    min_fid = min(sim_list)

    # Percentiles
    sorted_sims = sorted(sim_list)
    n = len(sorted_sims)

    def percentile(p: float) -> float:
        k = (n - 1) * p / 100
        f = int(k)
        c = f + 1
        if c >= n:
            return sorted_sims[-1]
        return sorted_sims[f] * (c - k) + sorted_sims[c] * (k - f)

    return {
        "mean": mean_fid,
        "std": std_fid,
        "min": min_fid,
        "percentiles": {
            "p50": percentile(50),
            "p90": percentile(90),
            "p95": percentile(95),
            "p99": percentile(99),
        },
    }


def compute_compression_ratio(
    original: torch.Tensor,
    compressed: torch.Tensor | dict[str, Any],
) -> float:
    """Compute compression ratio.

    Args:
        original: Original embeddings tensor.
        compressed: Compressed representation (tensor or dict).

    Returns:
        Compression ratio (original_size / compressed_size).

    Why dict handling:
        Some compactors return multiple tensors or codes, packaged as a dict.
        We sum all tensor sizes to get total compressed size.
    """
    original_bytes = original.numel() * original.element_size()

    if isinstance(compressed, torch.Tensor):
        compressed_bytes = compressed.numel() * compressed.element_size()
    elif isinstance(compressed, dict):
        compressed_bytes = 0
        for v in compressed.values():
            if isinstance(v, torch.Tensor):
                compressed_bytes += v.numel() * v.element_size()
    else:
        # Fallback: assume same size (1x compression)
        compressed_bytes = original_bytes

    return original_bytes / compressed_bytes if compressed_bytes > 0 else 1.0


def _measure_memory_peak(device: str) -> float:
    """Measure peak memory in MB.

    Args:
        device: Device string ("cuda" or "cpu").

    Returns:
        Peak memory in megabytes.
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    # CPU memory tracking is more complex; return 0 for now
    return 0.0


def _reset_memory_tracking(device: str) -> None:
    """Reset memory tracking counters.

    Args:
        device: Device string.
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# Main Benchmark Class
# =============================================================================


class CompressionBenchmark:
    """Benchmark suite for compression techniques.

    Provides standardized evaluation of compactors against ADR-0008 targets.

    Attributes:
        device: Device for benchmarking.
        num_samples: Number of test samples.
        embed_dim: Embedding dimension.
        num_warmup: Warmup iterations before timing.
        num_iterations: Timed iterations for latency.

    Example:
        >>> benchmark = CompressionBenchmark(device="cuda", num_samples=10000)
        >>> results = benchmark.run_compactor(my_compactor, name="MyCompactor")
        >>> print(results.meets_target())
    """

    def __init__(
        self,
        device: str = "cpu",
        num_samples: int = 1000,
        embed_dim: int = 512,
        num_warmup: int = 3,
        num_iterations: int = 10,
        seed: int = 42,
    ) -> None:
        """Initialize benchmark suite.

        Args:
            device: Device to run benchmarks on ("cpu" or "cuda").
            num_samples: Number of test embeddings to generate.
            embed_dim: Dimension of test embeddings.
            num_warmup: Warmup iterations before timing.
            num_iterations: Number of timed iterations.
            seed: Random seed for reproducibility.
        """
        self.device = device
        self.num_samples = num_samples
        self.embed_dim = embed_dim
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.seed = seed

        # Generate test data
        self.test_embeddings = generate_test_embeddings(
            num_samples=num_samples,
            embed_dim=embed_dim,
            device=device,
            seed=seed,
            distribution="realistic",
        )

    def run_compactor(
        self,
        compactor: nn.Module | CompactorProtocol,
        name: str | None = None,
    ) -> CompressionResult:
        """Benchmark a single compactor.

        Args:
            compactor: Compactor module to test.
            name: Name for the compactor (defaults to class name).

        Returns:
            CompressionResult with all metrics.
        """
        compactor_name = name or compactor.__class__.__name__

        # Wrap in adapter to normalize interface (compact vs compress)
        if not isinstance(compactor, CompactorAdapter):
            adapted = CompactorAdapter(compactor)
        else:
            adapted = compactor

        adapted.eval()
        adapted = adapted.to(self.device)

        embeddings = self.test_embeddings

        # Warmup
        with torch.no_grad():
            for _ in range(self.num_warmup):
                compressed = adapted.compress(embeddings[:100])
                _ = adapted.reconstruct(compressed)

        # Reset memory tracking
        _reset_memory_tracking(self.device)

        # Timed runs
        compress_times = []
        decompress_times = []

        with torch.no_grad():
            for _ in range(self.num_iterations):
                # Compression timing
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                compressed = adapted.compress(embeddings)
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                compress_times.append(time.perf_counter() - start)

                # Decompression timing
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                reconstructed = adapted.reconstruct(compressed)
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                decompress_times.append(time.perf_counter() - start)

        # Get peak memory
        memory_peak = _measure_memory_peak(self.device)

        # Compute fidelity
        fidelity = compute_fidelity(embeddings, reconstructed)

        # Compute compression ratio
        compression_ratio = compute_compression_ratio(embeddings, compressed)

        return CompressionResult(
            compactor_name=compactor_name,
            fidelity_mean=fidelity["mean"],
            fidelity_std=fidelity["std"],
            fidelity_min=fidelity["min"],
            fidelity_percentiles=fidelity["percentiles"],
            compression_ratio=compression_ratio,
            compress_latency_ms=statistics.mean(compress_times) * 1000,
            decompress_latency_ms=statistics.mean(decompress_times) * 1000,
            total_latency_ms=(
                statistics.mean(compress_times) + statistics.mean(decompress_times)
            )
            * 1000,
            memory_peak_mb=memory_peak,
            num_samples=self.num_samples,
            embed_dim=self.embed_dim,
            metadata={
                "device": self.device,
                "num_warmup": self.num_warmup,
                "num_iterations": self.num_iterations,
                "seed": self.seed,
            },
        )

    def run_full_suite(
        self,
        compactors: dict[str, nn.Module] | None = None,
    ) -> BenchmarkSuite:
        """Run benchmarks on all registered compactors.

        Args:
            compactors: Dict mapping names to compactor instances.
                        If None, uses default CogSynDelta compactors.

        Returns:
            BenchmarkSuite with all results.
        """
        if compactors is None:
            compactors = self._get_default_compactors()

        # Compute baseline (uncompressed) fidelity
        baseline_fidelity = 1.0  # By definition

        results = []
        for name, compactor in compactors.items():
            try:
                result = self.run_compactor(compactor, name=name)
                results.append(result)
            except Exception as e:
                # Log but continue with other compactors
                print(f"Warning: Failed to benchmark {name}: {e}")

        return BenchmarkSuite(
            results=results,
            baseline_fidelity=baseline_fidelity,
            device=self.device,
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
        )

    def _get_default_compactors(self) -> dict[str, nn.Module]:
        """Get default CogSynDelta compactors for testing.

        Returns:
            Dict mapping compactor names to instances.

        Why lazy import:
            Avoids circular import issues and allows benchmark module
            to be used standalone for testing new compactors.
        """
        compactors: dict[str, nn.Module] = {}

        try:
            from cogsyndelta.memory.active_memory import (
                HighFidelityCompactor,
                HybridAdaptiveCompactor,
                LosslessCompactor,
                ResidualBoostCompactor,
            )

            compactors["HighFidelityCompactor"] = HighFidelityCompactor(
                embed_dim=self.embed_dim
            )
            compactors["HybridAdaptiveCompactor"] = HybridAdaptiveCompactor(
                embed_dim=self.embed_dim
            )
            compactors["ResidualBoostCompactor"] = ResidualBoostCompactor(
                embed_dim=self.embed_dim
            )
            compactors["LosslessCompactor"] = LosslessCompactor(
                embed_dim=self.embed_dim
            )
        except ImportError as e:
            print(f"Warning: Could not import default compactors: {e}")

        try:
            from cogsyndelta.memory.dense_embeddings import DenseEmbeddingEncoder

            compactors["DenseEmbeddingEncoder"] = DenseEmbeddingEncoder(
                embed_dim=self.embed_dim
            )
        except ImportError:
            pass

        return compactors

    def save_results(
        self,
        output_dir: str | Path,
        suite: BenchmarkSuite,
        prefix: str = "compression",
    ) -> Path:
        """Save benchmark results to JSON file.

        Args:
            output_dir: Directory to save results.
            suite: BenchmarkSuite to save.
            prefix: Filename prefix.

        Returns:
            Path to saved file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
        filename = f"{prefix}_benchmark_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            f.write(suite.to_json())

        # Also save markdown summary
        summary_path = output_dir / f"{prefix}_summary_{timestamp}.md"
        with open(summary_path, "w") as f:
            f.write("# Compression Benchmark Results\n\n")
            f.write(f"**Date**: {suite.timestamp}\n")
            f.write(f"**Device**: {suite.device}\n")
            f.write(f"**PyTorch**: {suite.torch_version}\n\n")
            f.write("## Results\n\n")
            f.write(suite.summary_table())
            f.write("\n")

        return filepath


# =============================================================================
# Quick Benchmark Function
# =============================================================================


def run_quick_benchmark(
    device: str = "cpu",
    num_samples: int = 100,
) -> BenchmarkSuite:
    """Run a quick benchmark with fewer samples for testing.

    Args:
        device: Device to run on.
        num_samples: Number of test samples (default 100 for speed).

    Returns:
        BenchmarkSuite with results.

    Example:
        >>> results = run_quick_benchmark()
        >>> print(results.summary_table())
    """
    benchmark = CompressionBenchmark(
        device=device,
        num_samples=num_samples,
        num_warmup=1,
        num_iterations=3,
    )
    return benchmark.run_full_suite()


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Run compression benchmarks from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Run compression benchmarks")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run benchmarks on",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of test samples",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results/compression",
        help="Output directory for results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with fewer samples",
    )

    args = parser.parse_args()

    if args.quick:
        suite = run_quick_benchmark(device=args.device)
    else:
        benchmark = CompressionBenchmark(
            device=args.device,
            num_samples=args.samples,
        )
        suite = benchmark.run_full_suite()

    print("\n" + suite.summary_table() + "\n")

    benchmark = CompressionBenchmark(device=args.device)
    filepath = benchmark.save_results(args.output, suite)
    print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    main()
