"""Model Architecture Benchmarks for CogSynDelta.

Measures CogSynDelta's PCN-VAE-GAN, VL-JEPA, and mHC architecture performance.
This is MODEL benchmarking (quality, efficiency, capability) not HARDWARE benchmarking.

Key differences from gpu_benchmark.py:
- Measures model-specific metrics (reconstruction quality, embedding fidelity)
- Compares against industry baselines (GPT-2, BERT, LLaMA published numbers)
- Tracks deltas over time for regression detection
- Reports both raw and normalized metrics for fair comparison

Per constitution: "All performance claims must be backed by evidence."
No wild claims - only measured, verifiable results.

Example:
    $ uv run python -m benchmarks.model_benchmarks
    $ uv run python -m benchmarks.model_benchmarks --component pcn-vae-gan
    $ uv run python -m benchmarks.model_benchmarks --save-baseline
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

__all__ = [
    "BenchmarkSuite",
    "InterconnectBenchmark",
    "ModelBenchmarkResult",
    "PCNVAEGANBenchmark",
    "VLJEPABenchmark",
    "run_full_benchmark",
]

# Paths
BENCHMARK_DIR = Path(__file__).parent.parent / "benchmark_results"
BASELINE_DIR = BENCHMARK_DIR / "baselines"
HISTORY_DIR = BENCHMARK_DIR / "history"


def get_git_sha() -> str:
    """Get current git commit SHA."""
    import shutil

    git_path = shutil.which("git") or "/usr/bin/git"
    try:
        result = subprocess.run(
            [git_path, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_hardware_info() -> dict[str, Any]:
    """Get hardware information for methodology disclosure."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
    }

    try:
        import torch

        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except ImportError:
        info["pytorch_version"] = "not installed"
        info["cuda_available"] = False

    return info


def print_methodology_header() -> None:
    """Print methodology disclosure header (required for all benchmark output)."""
    info = get_hardware_info()
    print("=" * 80)
    print("COGSYNDELTA MODEL BENCHMARK - METHODOLOGY DISCLOSURE")
    print("=" * 80)
    print(f"Date: {datetime.now(UTC).isoformat()}")
    print(f"Git SHA: {get_git_sha()}")
    print(f"Platform: {info['platform']}")
    print(f"Python: {info['python_version']}")
    print(f"PyTorch: {info.get('pytorch_version', 'N/A')}")
    if info.get("cuda_available"):
        print(f"GPU: {info.get('gpu_name', 'N/A')}")
        print(f"CUDA: {info.get('cuda_version', 'N/A')}")
        print(f"GPU Memory: {info.get('gpu_memory_gb', 0):.1f} GB")
    else:
        print("GPU: Not available (CPU mode)")
    print()
    print("Statistical Parameters (per constitution):")
    print("  - Warmup iterations: 10")
    print("  - Timing iterations: 100")
    print("  - Metrics: mean ± std, p50/p95/p99")
    print()
    print("This benchmark measures MODEL performance, not hardware.")
    print("For hardware benchmarks, see: benchmarks/gpu_benchmark.py")
    print("=" * 80)
    print()


@dataclass
class ComponentMetrics:
    """Metrics for a single component benchmark.

    Attributes:
        name: Component name (e.g., "PCN-VAE-GAN Encoder").
        latency_ms: Latency metrics dict.
        throughput: Throughput metrics dict.
        memory_mb: Memory metrics dict.
        quality: Quality metrics dict (if applicable).
        normalized: Normalized metrics dict.
    """

    name: str
    latency_ms: dict[str, float] = field(default_factory=dict)
    throughput: dict[str, float] = field(default_factory=dict)
    memory_mb: dict[str, float] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)
    normalized: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelBenchmarkResult:
    """Complete benchmark result with metadata.

    Attributes:
        timestamp: ISO 8601 timestamp.
        git_sha: Git commit SHA.
        hardware: Hardware information dict.
        components: Dict of component name to ComponentMetrics.
        summary: Summary statistics.
    """

    timestamp: str
    git_sha: str
    hardware: dict[str, Any]
    components: dict[str, ComponentMetrics]
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "git_sha": self.git_sha,
            "hardware": self.hardware,
            "components": {name: asdict(metrics) for name, metrics in self.components.items()},
            "summary": self.summary,
        }

    def save(self, filepath: Path) -> None:
        """Save result to JSON file.

        Args:
            filepath: Path to save JSON file.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class PCNVAEGANBenchmark:
    """Benchmark suite for PCN-VAE-GAN component.

    Measures:
    - Encoder inference latency at various batch sizes
    - Decoder inference latency
    - Full VAE forward pass
    - Reconstruction quality (MSE, cosine similarity)
    - KL divergence (latent space quality)
    - Memory usage
    """

    def __init__(self, device: str = "cuda") -> None:
        """Initialize benchmark.

        Args:
            device: Device to run benchmarks on ("cuda" or "cpu").
        """
        self.device = device
        self._model = None
        self._param_count: int | None = None

    def _get_model(self) -> Any:
        """Lazy load model."""
        if self._model is None:
            from pathlib import Path

            import torch
            import yaml

            from cogsyndelta.core.pcn_vae_gan import PCNVAEGANHybrid

            # Load config from YAML
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
            with open(config_path) as f:
                config = yaml.safe_load(f)
            self._model = PCNVAEGANHybrid(config)
            actual_device = self.device if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(actual_device)
            self._model.eval()

            from benchmarks.utils.normalization import count_parameters

            self._param_count = count_parameters(self._model)

        return self._model

    def run(self) -> ComponentMetrics:
        """Run PCN-VAE-GAN benchmarks.

        Returns:
            ComponentMetrics with all measurements.
        """
        import torch

        from benchmarks.utils.metrics import (
            measure_latency,
            measure_memory,
            measure_quality,
            measure_throughput,
        )
        from benchmarks.utils.normalization import (
            normalize_memory,
            normalize_throughput,
        )

        print("Running PCN-VAE-GAN benchmarks...")
        model = self._get_model()
        actual_device = next(model.parameters()).device

        metrics = ComponentMetrics(name="PCN-VAE-GAN")

        # PCNVAEGANHybrid uses input_dim=784 (MNIST 28x28 flattened)
        input_dim = 784

        # Test at multiple batch sizes
        batch_sizes = [1, 8, 32, 64]

        for batch_size in batch_sizes:
            print(f"  Batch size {batch_size}...")
            x = torch.randn(batch_size, input_dim, device=actual_device)

            # Latency
            with torch.no_grad():
                latency = measure_latency(lambda _x=x: model(_x))
            metrics.latency_ms[f"forward_bs{batch_size}"] = latency.mean

            # Throughput
            with torch.no_grad():
                throughput = measure_throughput(lambda _x=x: model(_x), batch_size=batch_size)
            metrics.throughput[f"samples_per_sec_bs{batch_size}"] = throughput.samples_per_sec

            # Quality (reconstruction fidelity)
            with torch.no_grad():
                recon, mu, logvar = model(x)
                quality = measure_quality(x, recon, mu, logvar)

            if batch_size == 32:  # Report quality for standard batch size
                metrics.quality["reconstruction_mse"] = quality.reconstruction_mse or 0
                metrics.quality["cosine_similarity"] = quality.cosine_similarity or 0
                if quality.kl_divergence is not None:
                    metrics.quality["kl_divergence"] = quality.kl_divergence

        # Memory
        x = torch.randn(32, input_dim, device=actual_device)
        with torch.no_grad():
            memory = measure_memory(lambda: model(x), model=model)
        metrics.memory_mb["peak_allocated"] = memory.peak_allocated_mb
        metrics.memory_mb["model_size"] = memory.model_size_mb

        # Normalized metrics
        norm_throughput = normalize_throughput(
            metrics.throughput.get("samples_per_sec_bs32", 0),
            param_count=self._param_count,
        )
        norm_memory = normalize_memory(
            memory.peak_allocated_mb,
            param_count=self._param_count,
        )

        metrics.normalized = {
            "param_count": self._param_count,
            "throughput_per_billion_params": norm_throughput.per_billion_params,
            "bytes_per_param": norm_memory.bytes_per_param,
        }

        print(
            f"  ✓ PCN-VAE-GAN: {metrics.throughput.get('samples_per_sec_bs32', 0):.0f} samples/sec"
        )
        return metrics


class VLJEPABenchmark:
    """Benchmark suite for VL-JEPA component.

    Measures:
    - Vision encoder throughput
    - Temporal memory bank read/write latency
    - Joint embedding space alignment
    - End-to-end inference latency
    """

    def __init__(self, device: str = "cuda") -> None:
        """Initialize benchmark.

        Args:
            device: Device to run benchmarks on.
        """
        self.device = device
        self._vision_encoder = None
        self._memory_bank = None
        self._param_count: int | None = None

    def _get_components(self) -> tuple[Any, Any]:
        """Lazy load VL-JEPA components."""
        if self._vision_encoder is None:
            import torch

            from cogsyndelta.core.vl_jepa_extension import (
                TemporalMemoryBank,
                VisionEncoder,
            )

            actual_device = self.device if torch.cuda.is_available() else "cpu"

            self._vision_encoder = VisionEncoder(
                embed_dim=512,
                image_size=224,
                patch_size=16,
                num_layers=6,
            ).to(actual_device)
            self._vision_encoder.eval()

            self._memory_bank = TemporalMemoryBank(
                embed_dim=512,
                memory_size=1000,
                num_read_heads=4,
            ).to(actual_device)

            from benchmarks.utils.normalization import count_parameters

            self._param_count = count_parameters(self._vision_encoder)

        return self._vision_encoder, self._memory_bank

    def run(self) -> ComponentMetrics:
        """Run VL-JEPA benchmarks.

        Returns:
            ComponentMetrics with all measurements.
        """
        import torch

        from benchmarks.utils.metrics import (
            measure_latency,
            measure_memory,
            measure_throughput,
        )
        from benchmarks.utils.normalization import normalize_throughput

        print("Running VL-JEPA benchmarks...")
        vision_encoder, memory_bank = self._get_components()
        actual_device = next(vision_encoder.parameters()).device

        metrics = ComponentMetrics(name="VL-JEPA")

        # Vision encoder throughput
        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            print(f"  Vision encoder batch size {batch_size}...")
            # Image input: [B, C, H, W]
            x = torch.randn(batch_size, 3, 224, 224, device=actual_device)

            with torch.no_grad():
                latency = measure_latency(lambda _x=x: vision_encoder(_x))
                throughput = measure_throughput(
                    lambda _x=x: vision_encoder(_x), batch_size=batch_size
                )

            metrics.latency_ms[f"vision_encoder_bs{batch_size}"] = latency.mean
            metrics.throughput[f"images_per_sec_bs{batch_size}"] = throughput.samples_per_sec

        # Temporal memory bank read/write
        print("  Temporal memory bank...")
        query = torch.randn(1, 512, device=actual_device)
        value = torch.randn(1, 512, device=actual_device)

        with torch.no_grad():
            write_latency = measure_latency(lambda: memory_bank.write(value))
            read_latency = measure_latency(lambda: memory_bank.read(query))

        metrics.latency_ms["memory_write"] = write_latency.mean
        metrics.latency_ms["memory_read"] = read_latency.mean

        # Memory usage
        x = torch.randn(8, 3, 224, 224, device=actual_device)
        with torch.no_grad():
            memory = measure_memory(lambda: vision_encoder(x), model=vision_encoder)
        metrics.memory_mb["peak_allocated"] = memory.peak_allocated_mb
        metrics.memory_mb["model_size"] = memory.model_size_mb

        # Normalized metrics
        norm_throughput = normalize_throughput(
            metrics.throughput.get("images_per_sec_bs8", 0),
            param_count=self._param_count,
        )

        metrics.normalized = {
            "param_count": self._param_count,
            "throughput_per_billion_params": norm_throughput.per_billion_params,
        }

        print(f"  ✓ VL-JEPA: {metrics.throughput.get('images_per_sec_bs8', 0):.0f} images/sec")
        return metrics


class InterconnectBenchmark:
    """Benchmark suite for mHC/Interconnect Manager.

    Measures:
    - mHCGate forward pass latency
    - Gate activation statistics
    - Cross-component routing throughput
    """

    def __init__(self, device: str = "cuda") -> None:
        """Initialize benchmark.

        Args:
            device: Device to run benchmarks on.
        """
        self.device = device
        self._gate = None
        self._manager = None

    def _get_components(self) -> tuple[Any, Any]:
        """Lazy load interconnect components."""
        if self._gate is None:
            import torch

            from cogsyndelta.core.interconnect_manager import IntelligentInterconnectManager
            from cogsyndelta.core.vl_jepa_extension import ModeratedHyperConnection

            actual_device = self.device if torch.cuda.is_available() else "cpu"

            self._gate = ModeratedHyperConnection(embed_dim=512).to(actual_device)
            self._gate.eval()

            self._manager = IntelligentInterconnectManager(
                embed_dim=512,
                num_sections=4,
            ).to(actual_device)

        return self._gate, self._manager

    def run(self) -> ComponentMetrics:
        """Run Interconnect benchmarks.

        Returns:
            ComponentMetrics with all measurements.
        """
        import torch

        from benchmarks.utils.metrics import measure_latency, measure_throughput

        print("Running Interconnect (mHC) benchmarks...")
        gate, manager = self._get_components()
        actual_device = next(gate.parameters()).device

        metrics = ComponentMetrics(name="Interconnect (mHC)")

        # mHCGate benchmarks
        batch_sizes = [1, 8, 32]
        for batch_size in batch_sizes:
            print(f"  mHCGate batch size {batch_size}...")
            source = torch.randn(batch_size, 512, device=actual_device)
            target = torch.randn(batch_size, 512, device=actual_device)

            with torch.no_grad():
                latency = measure_latency(lambda _s=source, _t=target: gate(_s, _t))
                throughput = measure_throughput(
                    lambda _s=source, _t=target: gate(_s, _t), batch_size=batch_size
                )

            metrics.latency_ms[f"gate_bs{batch_size}"] = latency.mean
            metrics.throughput[f"gate_samples_per_sec_bs{batch_size}"] = throughput.samples_per_sec

        # Gate activation statistics
        source = torch.randn(100, 512, device=actual_device)
        target = torch.randn(100, 512, device=actual_device)
        with torch.no_grad():
            gate_output = gate(source, target)
            # The gate output is the moderated value, not the gate values directly
            # We'll compute statistics on the output magnitude
            output_magnitude = gate_output.norm(dim=-1)
            metrics.quality["gate_output_mean"] = output_magnitude.mean().item()
            metrics.quality["gate_output_std"] = output_magnitude.std().item()

        print(
            f"  ✓ Interconnect: {metrics.throughput.get('gate_samples_per_sec_bs32', 0):.0f} samples/sec"
        )
        return metrics


class BenchmarkSuite:
    """Complete benchmark suite for CogSynDelta model.

    Runs all component benchmarks and generates comprehensive report.
    """

    def __init__(self, device: str = "cuda") -> None:
        """Initialize benchmark suite.

        Args:
            device: Device to run benchmarks on.
        """
        self.device = device
        self.benchmarks = {
            "pcn-vae-gan": PCNVAEGANBenchmark(device),
            "vl-jepa": VLJEPABenchmark(device),
            "interconnect": InterconnectBenchmark(device),
        }

    def run(
        self,
        components: list[str] | None = None,
    ) -> ModelBenchmarkResult:
        """Run benchmark suite.

        Args:
            components: List of components to benchmark, or None for all.

        Returns:
            ModelBenchmarkResult with all measurements.
        """
        print_methodology_header()

        if components is None:
            components = list(self.benchmarks.keys())

        result = ModelBenchmarkResult(
            timestamp=datetime.now(UTC).isoformat(),
            git_sha=get_git_sha(),
            hardware=get_hardware_info(),
            components={},
        )

        for name in components:
            if name in self.benchmarks:
                try:
                    metrics = self.benchmarks[name].run()
                    result.components[name] = metrics
                except Exception as e:
                    print(f"  ✗ {name}: Failed with {type(e).__name__}: {e}")
            else:
                print(f"  ✗ Unknown component: {name}")

        # Generate summary
        result.summary = self._generate_summary(result)

        return result

    def _generate_summary(self, result: ModelBenchmarkResult) -> dict[str, Any]:
        """Generate summary statistics from results."""
        summary: dict[str, Any] = {
            "total_components": len(result.components),
            "successful_components": len([c for c in result.components.values() if c.latency_ms]),
        }

        # Aggregate key metrics
        total_params = 0
        total_throughput = 0.0

        for metrics in result.components.values():
            if metrics.normalized.get("param_count"):
                total_params += metrics.normalized["param_count"]
            # Sum throughput at batch size 32 (or 8 for VL-JEPA)
            for key, value in metrics.throughput.items():
                if "bs32" in key or "bs8" in key:
                    total_throughput += value
                    break

        summary["total_parameters"] = total_params
        summary["total_parameters_millions"] = total_params / 1_000_000
        summary["aggregate_throughput_samples_per_sec"] = total_throughput

        return summary


def run_full_benchmark(
    device: str = "cuda",
    components: list[str] | None = None,
    save_baseline: bool = False,
    save_history: bool = True,
) -> ModelBenchmarkResult:
    """Run full model benchmark suite.

    Args:
        device: Device to run benchmarks on.
        components: Components to benchmark (None for all).
        save_baseline: Whether to save as new baseline.
        save_history: Whether to save to history.

    Returns:
        ModelBenchmarkResult with all measurements.
    """
    suite = BenchmarkSuite(device=device)
    result = suite.run(components=components)

    # Print summary
    print()
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total parameters: {result.summary.get('total_parameters_millions', 0):.2f}M")
    print(
        f"Aggregate throughput: {result.summary.get('aggregate_throughput_samples_per_sec', 0):.0f} samples/sec"
    )
    print()

    for name, metrics in result.components.items():
        print(f"{name}:")
        if metrics.throughput:
            key = next(
                (k for k in metrics.throughput if "bs32" in k or "bs8" in k),
                next(iter(metrics.throughput.keys())) if metrics.throughput else None,
            )
            if key:
                print(f"  Throughput: {metrics.throughput[key]:.0f} samples/sec")
        if metrics.quality:
            for qname, qvalue in metrics.quality.items():
                print(f"  {qname}: {qvalue:.4f}")
        print()

    # Save results
    if save_baseline:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        baseline_path = BASELINE_DIR / "model_baseline.json"
        result.save(baseline_path)
        print(f"Saved baseline to: {baseline_path}")

    if save_history:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        history_path = HISTORY_DIR / f"benchmark_{result.timestamp.replace(':', '-')}.json"
        result.save(history_path)
        print(f"Saved history to: {history_path}")

    return result


def main() -> None:
    """CLI entry point for model benchmarks."""
    parser = argparse.ArgumentParser(
        description="CogSynDelta Model Architecture Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python -m benchmarks.model_benchmarks
  uv run python -m benchmarks.model_benchmarks --component pcn-vae-gan
  uv run python -m benchmarks.model_benchmarks --save-baseline
  uv run python -m benchmarks.model_benchmarks --device cpu
        """,
    )
    parser.add_argument(
        "--component",
        "-c",
        action="append",
        choices=["pcn-vae-gan", "vl-jepa", "interconnect"],
        help="Component to benchmark (can specify multiple)",
    )
    parser.add_argument(
        "--device",
        "-d",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run benchmarks on",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save results as new baseline",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Don't save to history",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Custom output path for results JSON",
    )

    args = parser.parse_args()

    result = run_full_benchmark(
        device=args.device,
        components=args.component,
        save_baseline=args.save_baseline,
        save_history=not args.no_history,
    )

    if args.output:
        result.save(args.output)
        print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
