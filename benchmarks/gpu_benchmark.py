"""
GPU-Ready Benchmark Suite for CogSynDelta
Measures performance on available hardware (CPU or GPU)
"""

import torch
import torch.nn as nn
import time
import psutil
import os
from datetime import datetime
from typing import Dict, List, Any


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices."""
    info: Dict[str, Any] = {
        'cpu_cores': psutil.cpu_count(logical=False),
        'cpu_threads': psutil.cpu_count(logical=True),
        'ram_gb': psutil.virtual_memory().total / (1024**3),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'working_dir': os.getcwd(),
    }
    
    if torch.cuda.is_available():
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return info


def benchmark_matrix_operations(device: torch.device, sizes: List[int] = [512, 1024, 2048, 4096]) -> List[Dict[str, Any]]:
    """Benchmark matrix multiplication operations."""
    print(f"\n{'='*70}")
    print(f"Matrix Multiplication Benchmark ({device})")
    print('='*70)
    
    results = []
    
    for size in sizes:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warmup
        for _ in range(5):
            c = torch.matmul(a, b)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(10):
            c = torch.matmul(a, b)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        
        avg_time = (end - start) / 10 * 1000  # ms
        gflops = (2 * size**3) / (avg_time / 1000) / 1e9
        
        result = {
            'size': size,
            'time_ms': avg_time,
            'gflops': gflops
        }
        results.append(result)
        
        print(f"  {size}x{size}: {avg_time:7.2f} ms  ({gflops:6.1f} GFLOPS)")
    
    return results


def benchmark_neural_network(device: torch.device, batch_sizes: List[int] = [1, 8, 32, 128]) -> List[Dict[str, Any]]:
    """Benchmark simple neural network forward pass."""
    print(f"\n{'='*70}")
    print(f"Neural Network Benchmark ({device})")
    print('='*70)
    
    # Create a simple network similar to CogSynDelta components
    class SimpleNet(nn.Module):
        """Simple feedforward network for benchmarking neural network operations."""
        
        def __init__(self) -> None:
            """Initialize network layers with LayerNorm and ReLU activations."""
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(784, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through the network layers."""
            return self.layers(x)
    
    model = SimpleNet().to(device)
    model.eval()
    
    results = []
    
    for batch_size in batch_sizes:
        input_data = torch.randn(batch_size, 784, device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_data)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(100):
                _ = model(input_data)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
        
        avg_time = (end - start) / 100 * 1000  # ms
        throughput = batch_size * 100 / (end - start)  # samples/sec
        
        result = {
            'batch_size': batch_size,
            'time_ms': avg_time,
            'throughput': throughput
        }
        results.append(result)
        
        print(f"  Batch {batch_size:3d}: {avg_time:6.2f} ms  ({throughput:8.1f} samples/sec)")
    
    return results


def benchmark_memory_operations(device: torch.device) -> List[Dict[str, Any]]:
    """Benchmark memory compression operations."""
    print(f"\n{'='*70}")
    print(f"Memory Compression Benchmark ({device})")
    print('='*70)
    
    # Test different compression ratios
    embed_dim = 512
    num_samples = 1000
    
    embeddings = torch.randn(num_samples, embed_dim, device=device)
    
    results = []
    
    for target_dim in [256, 128, 64, 32]:
        compression_ratio = embed_dim / target_dim
        
        # Simple compression via pooling
        start = time.perf_counter()
        
        compressed = torch.nn.functional.adaptive_avg_pool1d(
            embeddings.unsqueeze(1), target_dim
        ).squeeze(1)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        
        # Measure fidelity by decompression
        decompressed = torch.nn.functional.interpolate(
            compressed.unsqueeze(1), size=embed_dim, mode='linear'
        ).squeeze(1)
        
        fidelity = torch.nn.functional.cosine_similarity(
            embeddings, decompressed
        ).mean().item()
        
        time_ms = (end - start) * 1000
        throughput = num_samples / (end - start)
        
        result = {
            'compression_ratio': compression_ratio,
            'fidelity': fidelity,
            'time_ms': time_ms,
            'throughput': throughput
        }
        results.append(result)
        
        print(f"  {compression_ratio:4.1f}x: fidelity={fidelity:.4f}, "
              f"time={time_ms:6.2f} ms, {throughput:7.1f} samples/sec")
    
    return results


def main() -> None:
    """Run complete benchmark suite."""
    print('='*70)
    print('CogSynDelta GPU Benchmark Suite')
    print('='*70)
    print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    # Get device info
    info = get_device_info()
    
    print(f"\n{'='*70}")
    print('System Information')
    print('='*70)
    print(f"  CPU: {info['cpu_cores']} cores ({info['cpu_threads']} threads)")
    print(f"  RAM: {info['ram_gb']:.1f} GB")
    print(f"  PyTorch: {info['pytorch_version']}")
    print(f"  CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"  GPU Count: {info['gpu_count']}")
        print(f"  GPU: {info['gpu_name']}")
        print(f"  GPU Memory: {info['gpu_memory_gb']:.1f} GB")
    
    # Determine device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Using device: {device}")
    
    # Run benchmarks
    matrix_results = benchmark_matrix_operations(device)
    nn_results = benchmark_neural_network(device)
    memory_results = benchmark_memory_operations(device)
    
    # Summary
    print(f"\n{'='*70}")
    print('Benchmark Summary')
    print('='*70)
    
    # Best matrix performance
    best_matrix = max(matrix_results, key=lambda x: x['gflops'])
    print(f"\n  Best Matrix Performance:")
    print(f"    {best_matrix['size']}x{best_matrix['size']}: {best_matrix['gflops']:.1f} GFLOPS")
    
    # Best NN throughput
    best_nn = max(nn_results, key=lambda x: x['throughput'])
    print(f"\n  Best NN Throughput:")
    print(f"    Batch {best_nn['batch_size']}: {best_nn['throughput']:.1f} samples/sec")
    
    # Best compression
    best_compression = max(memory_results, key=lambda x: x['fidelity'])
    print(f"\n  Best Compression (by fidelity):")
    print(f"    {best_compression['compression_ratio']:.1f}x: fidelity={best_compression['fidelity']:.4f}")
    
    print(f"\n{'='*70}")
    
    if not info['cuda_available']:
        print('\nNote: Benchmarks run on CPU only.')
        print('For GPU acceleration, install CUDA-enabled PyTorch:')
        print('  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121')
    else:
        print('\n✓ GPU benchmarks completed successfully!')
    
    print('='*70)


if __name__ == '__main__':
    main()
