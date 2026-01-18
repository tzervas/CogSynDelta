"""
Industry-Standard Benchmarks for CogSynDelta on RTX 5080
Tests include: MNIST, ResNet-like models, and standard ML benchmarks
"""

import json
import time
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms


class MNISTNet(nn.Module):
    """Standard CNN for MNIST (LeNet-5 style)"""

    def __init__(self) -> None:
        """Initialize CNN layers for MNIST classification."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv layers with dropout and softmax output."""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ResNetBlock(nn.Module):
    """ResNet-like block for benchmarking"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Initialize ResNet block with convolutions and optional shortcut."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SmallResNet(nn.Module):
    """Small ResNet for benchmarking"""

    def __init__(self, num_classes: int = 10) -> None:
        """Initialize SmallResNet with configurable output classes."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(
        self, in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Create a layer of ResNet blocks."""
        layers = [ResNetBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet layers with global average pooling."""
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def benchmark_throughput(
    model: nn.Module,
    input_shape: tuple,
    device: torch.device,
    batch_sizes: list[int] = [1, 8, 32, 128],
    num_iterations: int = 100,
) -> list[dict[str, Any]]:
    """Measure inference throughput"""
    model.eval()
    results: list[dict[str, Any]] = []

    print(f"\n{'=' * 70}")
    print(f"Throughput Benchmark - {device}")
    print("=" * 70)

    for batch_size in batch_sizes:
        inputs = torch.randn(batch_size, *input_shape, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
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

        avg_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        throughput = batch_size / (np.mean(times))

        result = {
            "batch_size": batch_size,
            "avg_time_ms": avg_time,
            "std_time_ms": std_time,
            "throughput": throughput,
        }
        results.append(result)

        print(
            f"  Batch {batch_size:3d}: {avg_time:7.2f}±{std_time:5.2f} ms  "
            f"({throughput:8.1f} samples/sec)"
        )

    return results


def benchmark_mnist_training(device: torch.device, num_epochs: int = 2) -> list[dict[str, Any]]:
    """Train on MNIST and measure performance"""
    print(f"\n{'=' * 70}")
    print(f"MNIST Training Benchmark - {device}")
    print("=" * 70)

    # Load MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    print("\nLoading MNIST dataset...")
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Create model
    model = MNISTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    results = []

    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        epoch_loss = 0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(
                    f"  Epoch {epoch + 1}/{num_epochs} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / num_batches

        # Test accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = 100.0 * correct / total

        result = {"epoch": epoch + 1, "time": epoch_time, "loss": avg_loss, "accuracy": accuracy}
        results.append(result)

        print(f"\n  Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Test Accuracy: {accuracy:.2f}%\n")

    return results


def benchmark_resnet(
    device: torch.device, batch_sizes: list[int] = [16, 32, 64, 128], num_iterations: int = 50
) -> list[dict[str, Any]]:
    """Benchmark ResNet-like model"""
    print(f"\n{'=' * 70}")
    print(f"ResNet-18 Style Benchmark - {device}")
    print("=" * 70)

    model = SmallResNet(num_classes=1000).to(device)
    input_shape = (3, 224, 224)

    return benchmark_throughput(model, input_shape, device, batch_sizes, num_iterations)


def benchmark_memory_bandwidth(device: torch.device) -> list[dict[str, float]]:
    """Benchmark memory bandwidth"""
    print(f"\n{'=' * 70}")
    print(f"Memory Bandwidth Benchmark - {device}")
    print("=" * 70)

    sizes_mb = [1, 10, 100, 500, 1000]
    results = []

    for size_mb in sizes_mb:
        num_floats = size_mb * 1024 * 1024 // 4

        # Host to device
        data_cpu = torch.randn(num_floats)

        start = time.perf_counter()
        data_gpu = data_cpu.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        h2d_time = time.perf_counter() - start
        h2d_bandwidth = size_mb / h2d_time / 1024  # GB/s

        # Device to host
        start = time.perf_counter()
        _ = data_gpu.cpu()
        if device.type == "cuda":
            torch.cuda.synchronize()
        d2h_time = time.perf_counter() - start
        d2h_bandwidth = size_mb / d2h_time / 1024  # GB/s

        result = {
            "size_mb": size_mb,
            "h2d_bandwidth_gbs": h2d_bandwidth,
            "d2h_bandwidth_gbs": d2h_bandwidth,
        }
        results.append(result)

        print(f"  {size_mb:4d} MB: H2D: {h2d_bandwidth:6.2f} GB/s, D2H: {d2h_bandwidth:6.2f} GB/s")

    return results


def main() -> None:
    """Run comprehensive industry-standard benchmarks"""
    print("=" * 70)
    print("CogSynDelta - Industry Standard Benchmarks on RTX 5080")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Device info
    print(f"\n{'=' * 70}")
    print("Hardware Configuration")
    print("=" * 70)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        device = torch.device("cuda")
    else:
        print("Running on CPU")
        device = torch.device("cpu")

    all_results = {
        "device": str(device),
        "pytorch_version": torch.__version__,
        "timestamp": datetime.now().isoformat(),
    }

    # Run benchmarks
    print("\n" + "=" * 70)
    print("Running Benchmarks...")
    print("=" * 70)

    # 1. MNIST Training (industry standard)
    try:
        mnist_results = benchmark_mnist_training(device, num_epochs=2)
        all_results["mnist_training"] = mnist_results
    except Exception as e:
        print(f"MNIST benchmark failed: {e}")
        all_results["mnist_training"] = f"Failed: {e!s}"

    # 2. CNN Throughput
    try:
        cnn_model = MNISTNet().to(device)
        cnn_results = benchmark_throughput(
            cnn_model, (1, 28, 28), device, batch_sizes=[1, 16, 64, 256]
        )
        all_results["cnn_throughput"] = cnn_results
    except Exception as e:
        print(f"CNN throughput benchmark failed: {e}")
        all_results["cnn_throughput"] = f"Failed: {e!s}"

    # 3. ResNet-like Model
    try:
        resnet_results = benchmark_resnet(device, batch_sizes=[8, 16, 32, 64])
        all_results["resnet_throughput"] = resnet_results
    except Exception as e:
        print(f"ResNet benchmark failed: {e}")
        all_results["resnet_throughput"] = f"Failed: {e!s}"

    # 4. Memory Bandwidth
    if device.type == "cuda":
        try:
            memory_results = benchmark_memory_bandwidth(device)
            all_results["memory_bandwidth"] = memory_results
        except Exception as e:
            print(f"Memory bandwidth benchmark failed: {e}")
            all_results["memory_bandwidth"] = f"Failed: {e!s}"

    # Save results
    with open("gpu_benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'=' * 70}")
    print("Benchmark Summary")
    print("=" * 70)

    if "mnist_training" in all_results and isinstance(all_results["mnist_training"], list):
        final_mnist = all_results["mnist_training"][-1]
        print("\nMNIST Training (2 epochs):")
        print(f"  Final Accuracy: {final_mnist['accuracy']:.2f}%")
        print(f"  Time per Epoch: {final_mnist['time']:.1f}s")

    if "cnn_throughput" in all_results and isinstance(all_results["cnn_throughput"], list):
        best_cnn = max(all_results["cnn_throughput"], key=lambda x: x["throughput"])
        print("\nCNN Inference:")
        print(
            f"  Best Throughput: {best_cnn['throughput']:.0f} samples/sec (batch {best_cnn['batch_size']})"
        )

    if "resnet_throughput" in all_results and isinstance(all_results["resnet_throughput"], list):
        best_resnet = max(all_results["resnet_throughput"], key=lambda x: x["throughput"])
        print("\nResNet-18 Inference:")
        print(
            f"  Best Throughput: {best_resnet['throughput']:.0f} samples/sec (batch {best_resnet['batch_size']})"
        )

    if "memory_bandwidth" in all_results and isinstance(all_results["memory_bandwidth"], list):
        avg_h2d = np.mean([r["h2d_bandwidth_gbs"] for r in all_results["memory_bandwidth"]])
        avg_d2h = np.mean([r["d2h_bandwidth_gbs"] for r in all_results["memory_bandwidth"]])
        print("\nMemory Bandwidth:")
        print(f"  Host to Device: {avg_h2d:.2f} GB/s (avg)")
        print(f"  Device to Host: {avg_d2h:.2f} GB/s (avg)")

    print(f"\n{'=' * 70}")
    print("✓ All benchmarks completed!")
    print("Results saved to: gpu_benchmark_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
