"""
RTX 5080 GPU Benchmark - Simplified without torchvision
Uses synthetic data to benchmark actual GPU performance
"""

import json
import time
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SimpleCNN(nn.Module):
    """Simple CNN for benchmarking"""

    def __init__(self) -> None:
        """Initialize CNN layers for image classification."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ReLU activations and max pooling."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ResNetBlock(nn.Module):
    """ResNet block"""

    def __init__(self, channels: int) -> None:
        """Initialize ResNet block with given channel count."""
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class LargeResNet(nn.Module):
    """Larger ResNet for benchmarking"""

    def __init__(self) -> None:
        """Initialize LargeResNet with multiple ResNet blocks and transition layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = nn.Sequential(*[ResNetBlock(64) for _ in range(3)])
        self.layer2 = nn.Sequential(*[ResNetBlock(128) for _ in range(4)])
        self.layer3 = nn.Sequential(*[ResNetBlock(256) for _ in range(6)])

        # Transition layers
        self.trans1 = nn.Conv2d(64, 128, 1, 2)
        self.trans2 = nn.Conv2d(128, 256, 1, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet layers with transitions and pooling."""
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.trans1(x)
        x = self.layer2(x)
        x = self.trans2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def run_inference_benchmark(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device,
    batch_sizes: list[int],
    num_iterations: int = 100,
    warmup: int = 20,
) -> list[dict[str, Any]]:
    """Run inference throughput benchmark"""
    model.eval()
    results: list[dict[str, Any]] = []

    for batch_size in batch_sizes:
        inputs = torch.randn(batch_size, *input_shape, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(inputs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        avg_time_ms = np.mean(times) * 1000
        std_time_ms = np.std(times) * 1000
        throughput = batch_size / np.mean(times)

        results.append(
            {
                "batch_size": batch_size,
                "avg_time_ms": round(avg_time_ms, 2),
                "std_time_ms": round(std_time_ms, 2),
                "throughput_samples_sec": round(throughput, 1),
            }
        )

        print(
            f"  Batch {batch_size:3d}: {avg_time_ms:7.2f}±{std_time_ms:5.2f} ms  "
            f"({throughput:10,.1f} samples/sec)"
        )

    return results


def run_training_benchmark(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device,
    batch_size: int = 64,
    num_steps: int = 200,
) -> dict[str, Any]:
    """Benchmark training performance"""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    print(f"\nTraining Benchmark: batch_size={batch_size}, steps={num_steps}")

    times = []

    # Warmup
    for _ in range(10):
        inputs = torch.randn(batch_size, *input_shape, device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    for step in range(num_steps):
        inputs = torch.randn(batch_size, *input_shape, device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)

        start = time.perf_counter()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        times.append(time.perf_counter() - start)

        if (step + 1) % 50 == 0:
            avg_time = np.mean(times[-50:]) * 1000
            throughput = batch_size / np.mean(times[-50:])
            print(
                f"  Step {step + 1}/{num_steps}: {avg_time:.2f} ms/step  ({throughput:,.0f} samples/sec)"
            )

    avg_time_ms = np.mean(times) * 1000
    throughput = batch_size / np.mean(times)

    return {
        "batch_size": batch_size,
        "num_steps": num_steps,
        "avg_time_ms": round(avg_time_ms, 2),
        "throughput_samples_sec": round(throughput, 1),
    }


def benchmark_matrix_ops(
    device: torch.device, sizes: list[int] = [1024, 2048, 4096, 8192]
) -> list[dict[str, Any]]:
    """Benchmark matrix operations"""
    print("\nMatrix Operations Benchmark")
    results: list[dict[str, Any]] = []

    for size in sizes:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # Warmup
        for _ in range(5):
            _ = torch.matmul(a, b)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(20):
            start = time.perf_counter()
            c = torch.matmul(a, b)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time_ms = np.mean(times) * 1000
        gflops = (2 * size**3) / (np.mean(times) * 1e9)

        results.append({"size": size, "time_ms": round(avg_time_ms, 2), "gflops": round(gflops, 1)})

        print(f"  {size}x{size}: {avg_time_ms:8.2f} ms  ({gflops:8.1f} GFLOPS)")

    return results


def main() -> None:
    """Run the RTX 5080 GPU benchmark suite with industry standard tests."""
    print("=" * 80)
    print("RTX 5080 GPU Benchmark - Industry Standard Tests")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check GPU
    print(f"\n{'=' * 80}")
    print("Hardware Configuration")
    print("=" * 80)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n❌ CUDA not available. Exiting...")
        return

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(
        f"Compute Capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}"
    )

    all_results = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {},
    }

    # Benchmark 1: Matrix Operations
    print(f"\n{'=' * 80}")
    print("1. Matrix Multiplication (GEMM) - Core GPU Compute")
    print("=" * 80)
    matrix_results = benchmark_matrix_ops(device, [1024, 2048, 4096, 8192])
    all_results["benchmarks"]["matrix_operations"] = matrix_results

    # Benchmark 2: CNN Inference
    print(f"\n{'=' * 80}")
    print("2. CNN Inference - Image Classification Model")
    print("=" * 80)
    cnn_model = SimpleCNN().to(device)
    cnn_results = run_inference_benchmark(
        cnn_model, (1, 28, 28), device, batch_sizes=[1, 8, 32, 128, 512, 1024]
    )
    all_results["benchmarks"]["cnn_inference"] = cnn_results

    # Benchmark 3: ResNet Inference
    print(f"\n{'=' * 80}")
    print("3. ResNet Inference - Large Scale Model")
    print("=" * 80)
    resnet_model = LargeResNet().to(device)
    resnet_results = run_inference_benchmark(
        resnet_model, (3, 224, 224), device, batch_sizes=[1, 4, 8, 16, 32, 64]
    )
    all_results["benchmarks"]["resnet_inference"] = resnet_results

    # Benchmark 4: Training
    print(f"\n{'=' * 80}")
    print("4. Training Performance - Forward + Backward Pass")
    print("=" * 80)
    training_model = SimpleCNN().to(device)
    training_results = run_training_benchmark(
        training_model, (1, 28, 28), device, batch_size=128, num_steps=200
    )
    all_results["benchmarks"]["training"] = training_results

    # Save results
    with open("rtx5080_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'=' * 80}")
    print("Benchmark Summary")
    print("=" * 80)

    best_matrix = max(matrix_results, key=lambda x: x["gflops"])
    print(
        f"\nBest Matrix Performance: {best_matrix['gflops']} GFLOPS ({best_matrix['size']}x{best_matrix['size']})"
    )

    best_cnn = max(cnn_results, key=lambda x: x["throughput_samples_sec"])
    print(
        f"Best CNN Throughput: {best_cnn['throughput_samples_sec']:,.0f} samples/sec (batch {best_cnn['batch_size']})"
    )

    best_resnet = max(resnet_results, key=lambda x: x["throughput_samples_sec"])
    print(
        f"Best ResNet Throughput: {best_resnet['throughput_samples_sec']:,.0f} samples/sec (batch {best_resnet['batch_size']})"
    )

    print(f"\nTraining: {training_results['throughput_samples_sec']:,.0f} samples/sec")

    print(f"\n{'=' * 80}")
    print("✓ All benchmarks completed successfully!")
    print("Results saved to: rtx5080_benchmark_results.json")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
