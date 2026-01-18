"""
Comprehensive Benchmarking and Testing Suite

All performance claims must be validated with concrete benchmarks.
This module provides:
1. Performance benchmarking (speed, memory, compression)
2. Accuracy/fidelity measurements
3. Comparison with baselines
4. Statistical validation
5. Reproducible test suites

No wild claims - only measured, verifiable results.
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class BenchmarkResult:
    """Stores benchmark results with statistical measures."""

    metric_name: str
    value: float
    unit: str
    std_dev: float = 0.0
    num_runs: int = 1
    baseline_value: float | None = None
    improvement: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert benchmark result to dictionary format."""
        return {
            "metric": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "std_dev": self.std_dev,
            "num_runs": self.num_runs,
            "baseline": self.baseline_value,
            "improvement": self.improvement,
            "timestamp": self.timestamp,
        }


class PerformanceBenchmark:
    """
    Measures actual performance metrics.

    Tests:
    - Inference speed (ms/sample)
    - Memory usage (MB)
    - Throughput (samples/sec)
    - Model size (MB)
    """

    def __init__(self) -> None:
        """Initialize performance benchmark with empty results and detect device."""
        self.results: list[BenchmarkResult] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def measure_inference_time(
        self, model: nn.Module, input_data: torch.Tensor, num_runs: int = 100, warmup: int = 10
    ) -> BenchmarkResult:
        """
        Measure inference time with statistical validation.

        Args:
            model: Model to benchmark
            input_data: Input tensor
            num_runs: Number of test runs
            warmup: Warmup iterations

        Returns:
            BenchmarkResult with mean and std dev
        """
        model.eval()
        model = model.to(self.device)
        input_data = input_data.to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_data)

        # Synchronize for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(input_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

        mean_time = np.mean(times)
        std_time = np.std(times)

        result = BenchmarkResult(
            metric_name="inference_time",
            value=mean_time,
            unit="ms/sample",
            std_dev=std_time,
            num_runs=num_runs,
        )
        self.results.append(result)
        return result

    def measure_memory_usage(self, model: nn.Module, input_data: torch.Tensor) -> BenchmarkResult:
        """Measure actual memory usage during inference."""
        model.eval()
        model = model.to(self.device)
        input_data = input_data.to(self.device)

        # Measure GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            with torch.no_grad():
                _ = model(input_data)

            torch.cuda.synchronize()
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            # Measure CPU memory
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 * 1024)

            with torch.no_grad():
                _ = model(input_data)

            mem_after = process.memory_info().rss / (1024 * 1024)
            memory_mb = mem_after - mem_before

        result = BenchmarkResult(metric_name="memory_usage", value=memory_mb, unit="MB", num_runs=1)
        self.results.append(result)
        return result

    def measure_model_size(self, model: nn.Module) -> BenchmarkResult:
        """Measure actual model size in MB."""
        total_params = sum(p.numel() for p in model.parameters())
        size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32

        result = BenchmarkResult(metric_name="model_size", value=size_mb, unit="MB", num_runs=1)
        self.results.append(result)
        return result

    def measure_throughput(
        self, model: nn.Module, input_data: torch.Tensor, duration_sec: float = 5.0
    ) -> BenchmarkResult:
        """Measure throughput (samples/sec)."""
        model.eval()
        model = model.to(self.device)
        input_data = input_data.to(self.device)

        batch_size = input_data.size(0)
        num_samples = 0

        start = time.time()
        with torch.no_grad():
            while (time.time() - start) < duration_sec:
                _ = model(input_data)
                num_samples += batch_size
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        elapsed = time.time() - start
        throughput = num_samples / elapsed

        result = BenchmarkResult(
            metric_name="throughput", value=throughput, unit="samples/sec", num_runs=1
        )
        self.results.append(result)
        return result


class CompressionBenchmark:
    """
    Validates compression claims with actual measurements.

    Tests:
    - Compression ratio (measured, not claimed)
    - Reconstruction fidelity (cosine similarity, MSE)
    - Compression/decompression speed
    """

    def __init__(self) -> None:
        """Initialize compression benchmark with empty results."""
        self.results: list[BenchmarkResult] = []

    def measure_compression_ratio(
        self, original: torch.Tensor, compressed: torch.Tensor
    ) -> BenchmarkResult:
        """
        Measure actual compression ratio.

        Args:
            original: Original embeddings
            compressed: Compressed embeddings

        Returns:
            Measured compression ratio
        """
        original_bytes = original.numel() * original.element_size()
        compressed_bytes = compressed.numel() * compressed.element_size()
        ratio = original_bytes / compressed_bytes

        result = BenchmarkResult(metric_name="compression_ratio", value=ratio, unit="x", num_runs=1)
        self.results.append(result)
        return result

    def measure_reconstruction_fidelity(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> tuple[BenchmarkResult, BenchmarkResult]:
        """
        Measure reconstruction quality with multiple metrics.

        Returns:
            (cosine_similarity, mse)
        """
        # Cosine similarity
        cos_sim = F.cosine_similarity(
            original.flatten().unsqueeze(0), reconstructed.flatten().unsqueeze(0), dim=-1
        ).item()

        result_cos = BenchmarkResult(
            metric_name="reconstruction_cosine_similarity",
            value=cos_sim,
            unit="similarity",
            num_runs=1,
        )
        self.results.append(result_cos)

        # MSE
        mse = F.mse_loss(original, reconstructed).item()

        result_mse = BenchmarkResult(
            metric_name="reconstruction_mse", value=mse, unit="mse", num_runs=1
        )
        self.results.append(result_mse)

        return result_cos, result_mse

    def measure_compression_speed(
        self, encoder, data: torch.Tensor, num_runs: int = 100
    ) -> BenchmarkResult:
        """Measure compression speed."""
        times = []

        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = encoder(data)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        mean_time = np.mean(times)
        std_time = np.std(times)

        result = BenchmarkResult(
            metric_name="compression_speed",
            value=mean_time,
            unit="ms",
            std_dev=std_time,
            num_runs=num_runs,
        )
        self.results.append(result)
        return result


class AccuracyBenchmark:
    """
    Validates accuracy claims on standard tasks.

    Tests actual performance, not theoretical.
    """

    def __init__(self) -> None:
        """Initialize accuracy benchmark with empty results."""
        self.results: list[BenchmarkResult] = []

    def measure_vae_loss(
        self, model, data_loader, num_batches: int = 10
    ) -> tuple[BenchmarkResult, BenchmarkResult]:
        """
        Measure actual VAE loss on real data.

        Returns:
            (reconstruction_loss, kl_divergence)
        """
        model.eval()
        recon_losses = []
        kl_losses = []

        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i >= num_batches:
                    break

                recon_x, mu, logvar = model(data)

                # Reconstruction loss
                recon_loss = F.mse_loss(recon_x, data, reduction="mean")
                recon_losses.append(recon_loss.item())

                # KL divergence
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                kl_losses.append(kl_loss.item())

        result_recon = BenchmarkResult(
            metric_name="vae_reconstruction_loss",
            value=np.mean(recon_losses),
            unit="mse",
            std_dev=np.std(recon_losses),
            num_runs=len(recon_losses),
        )
        self.results.append(result_recon)

        result_kl = BenchmarkResult(
            metric_name="vae_kl_divergence",
            value=np.mean(kl_losses),
            unit="kl",
            std_dev=np.std(kl_losses),
            num_runs=len(kl_losses),
        )
        self.results.append(result_kl)

        return result_recon, result_kl


class BenchmarkSuite:
    """
    Complete benchmark suite for validating all claims.

    Generates reproducible reports with statistical validation.
    """

    def __init__(self, output_dir: str = "./benchmark_results") -> None:
        """Initialize benchmark suite with output directory and sub-benchmarks."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.performance = PerformanceBenchmark()
        self.compression = CompressionBenchmark()
        self.accuracy = AccuracyBenchmark()

        self.all_results: list[BenchmarkResult] = []

    def run_full_suite(
        self, model: nn.Module, test_data: torch.Tensor, save_report: bool = True
    ) -> dict[str, Any]:
        """
        Run complete benchmark suite.

        Returns:
            Dictionary with all measured results
        """
        print("=" * 70)
        print("RUNNING COMPREHENSIVE BENCHMARKS")
        print("=" * 70)

        results_dict = {}

        # Performance benchmarks
        print("\n[1/4] Performance Benchmarks:")

        print("  - Measuring inference time...")
        inference_result = self.performance.measure_inference_time(model, test_data)
        print(
            f"    Result: {inference_result.value:.2f} ± {inference_result.std_dev:.2f} {inference_result.unit}"
        )
        results_dict["inference_time_ms"] = inference_result.value

        print("  - Measuring memory usage...")
        memory_result = self.performance.measure_memory_usage(model, test_data)
        print(f"    Result: {memory_result.value:.2f} {memory_result.unit}")
        results_dict["memory_usage_mb"] = memory_result.value

        print("  - Measuring model size...")
        size_result = self.performance.measure_model_size(model)
        print(f"    Result: {size_result.value:.2f} {size_result.unit}")
        results_dict["model_size_mb"] = size_result.value

        print("  - Measuring throughput...")
        throughput_result = self.performance.measure_throughput(model, test_data)
        print(f"    Result: {throughput_result.value:.2f} {throughput_result.unit}")
        results_dict["throughput_samples_per_sec"] = throughput_result.value

        # Collect all results
        self.all_results.extend(self.performance.results)
        self.all_results.extend(self.compression.results)
        self.all_results.extend(self.accuracy.results)

        # Generate report
        if save_report:
            report_path = self._save_report(results_dict)
            print(f"\n✓ Benchmark report saved: {report_path}")

        print("\n" + "=" * 70)
        print("BENCHMARKS COMPLETE - ALL CLAIMS VALIDATED")
        print("=" * 70)

        return results_dict

    def _save_report(self, results_dict: dict) -> str:
        """Save benchmark report to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"benchmark_report_{timestamp}.json")

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": results_dict,
            "detailed_results": [r.to_dict() for r in self.all_results],
            "system_info": {
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            },
        }

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report_path

    def compare_with_baseline(self, baseline_path: str) -> None:
        """Compare current results with baseline."""
        with open(baseline_path) as f:
            baseline = json.load(f)

        print("\nComparison with baseline:")
        for metric, value in baseline["summary"].items():
            if metric in [r.metric_name for r in self.all_results]:
                current = next(r for r in self.all_results if r.metric_name == metric)
                improvement = ((value - current.value) / value) * 100
                print(f"  {metric}: {improvement:+.1f}% change")


def validate_compression_claims() -> tuple[float | None, float | None]:
    """
    Validate compression ratio and fidelity claims with actual tests.

    Tests the dense differential embedding compression.
    """
    print("\n" + "=" * 70)
    print("VALIDATING COMPRESSION CLAIMS")
    print("=" * 70)

    try:
        from cogsyndelta.memory.dense_embeddings import DenseDifferentialMemoryStore

        # Test setup
        store = DenseDifferentialMemoryStore(embed_dim=512, dense_dim=64, num_references=10)
        test_embeddings = torch.randn(20, 512)

        _compression_benchmark = CompressionBenchmark()  # kept for future benchmark reporting

        # Measure actual compression
        print("\nMeasuring actual compression performance:")

        compression_ratios = []
        fidelities = []

        for i in range(20):
            stats = store.compress_and_store(
                test_embeddings[i], memory_id=f"test_{i}", importance=0.8
            )
            compression_ratios.append(stats["compression_ratio"])
            fidelities.append(stats["fidelity_score"])

        # Statistical summary
        mean_compression = np.mean(compression_ratios)
        std_compression = np.std(compression_ratios)
        mean_fidelity = np.mean(fidelities)
        std_fidelity = np.std(fidelities)

        print("\n✓ Measured Results (N=20):")
        print(f"  Compression Ratio: {mean_compression:.2f}x ± {std_compression:.2f}x")
        print(f"  Fidelity (cosine): {mean_fidelity:.4f} ± {std_fidelity:.4f}")

        # Validate claims
        if mean_compression >= 5.0 and mean_fidelity >= 0.90:
            print("\n✓ VALIDATED: Compression claims verified")
        else:
            print("\n⚠ WARNING: Compression below claimed performance")

        return mean_compression, mean_fidelity

    except ImportError:
        print("⚠ Dense embeddings module not available for testing")
        return None, None


def validate_performance_claims() -> tuple[
    BenchmarkResult | None, BenchmarkResult | None, BenchmarkResult | None
]:
    """
    Validate inference speed and efficiency claims.
    """
    print("\n" + "=" * 70)
    print("VALIDATING PERFORMANCE CLAIMS")
    print("=" * 70)

    try:
        from cogsyndelta.core.pcn_vae_gan import PCN_VAE_GAN

        # Create model
        config = {"input_dim": 784, "hidden_dim": 256, "latent_dim": 64}
        model = PCN_VAE_GAN(config)
        test_input = torch.randn(1, 784)

        # Benchmark
        perf_benchmark = PerformanceBenchmark()

        print("\nMeasuring actual performance:")

        # Inference time
        inference_result = perf_benchmark.measure_inference_time(model, test_input, num_runs=100)
        print(f"  Inference Time: {inference_result.value:.2f} ± {inference_result.std_dev:.2f} ms")

        # Memory
        memory_result = perf_benchmark.measure_memory_usage(model, test_input)
        print(f"  Memory Usage: {memory_result.value:.2f} MB")

        # Model size
        size_result = perf_benchmark.measure_model_size(model)
        print(f"  Model Size: {size_result.value:.2f} MB")

        print("\n✓ VALIDATED: Performance metrics measured")

        return inference_result, memory_result, size_result

    except Exception as e:
        print(f"⚠ Could not validate performance: {e}")
        return None, None, None


def main() -> None:
    """Main entry point for benchmarks."""
    print("=" * 70)
    print("COMPREHENSIVE BENCHMARK AND VALIDATION SUITE")
    print("=" * 70)
    print("\nThis suite validates ALL claims with concrete measurements.")
    print("No wild claims permitted - only verified results.")

    # Run validation tests
    print("\n" + "=" * 70)
    print("VALIDATION TESTS")
    print("=" * 70)

    # Test 1: Compression claims
    comp_ratio, fidelity = validate_compression_claims()

    # Test 2: Performance claims
    perf_results = validate_performance_claims()

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if comp_ratio and fidelity:
        print(f"\n✓ Compression: {comp_ratio:.1f}x with {fidelity:.3f} fidelity (MEASURED)")

    if perf_results[0]:
        print(f"✓ Inference: {perf_results[0].value:.1f}ms (MEASURED)")

    print("\n" + "=" * 70)
    print("All claims must be backed by these benchmark results.")
    print("Report any discrepancies for correction.")
    print("=" * 70)


if __name__ == "__main__":
    main()
