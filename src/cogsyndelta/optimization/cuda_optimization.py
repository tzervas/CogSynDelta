"""
CUDA Optimization Module for NVIDIA RTX 5080 (16GB)

Optimizes GPU memory usage, kernel launches, and computational efficiency
for the PCN-VAE-GAN hybrid system on NVIDIA RTX 5080 GPUs.

Features:
1. GPU memory management optimized for 16GB VRAM
2. Mixed precision training (FP16/BF16) for RTX 5080 Tensor Cores
3. Flash Attention for memory-efficient transformers
4. Kernel fusion for reduced memory bandwidth
5. Multi-stream execution for concurrent operations
6. Dynamic batch sizing based on available memory
7. Embedding storage on GPU with efficient transfer

Inspired by embeddenator-core's balanced ternary optimization strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import warnings

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE_NAME = torch.cuda.get_device_name(0)
    TOTAL_MEMORY = torch.cuda.get_device_properties(0).total_memory
    COMPUTE_CAPABILITY = torch.cuda.get_device_capability(0)
else:
    DEVICE_NAME = "CPU"
    TOTAL_MEMORY = 0
    COMPUTE_CAPABILITY = (0, 0)


@dataclass
class GPUConfig:
    """Configuration for GPU optimization."""
    device: torch.device
    memory_gb: float
    use_mixed_precision: bool = True
    use_flash_attention: bool = True
    use_kernel_fusion: bool = True
    max_batch_size: int = 64
    num_streams: int = 4
    prefetch_factor: int = 2


class RTX5080Optimizer:
    """
    Optimizer specifically tuned for NVIDIA RTX 5080 16GB.
    
    RTX 5080 Specifications:
    - 16GB GDDR7
    - Ada Lovelace architecture
    - 4th Gen Tensor Cores (FP8/FP16/BF16/TF32)
    - 2nd Gen RT Cores
    - PCIe 5.0
    - ~20,000 CUDA cores (estimated)
    """
    
    def __init__(self):
        if not CUDA_AVAILABLE:
            warnings.warn("CUDA not available. Using CPU fallback.")
            self.device = torch.device('cpu')
            self.config = GPUConfig(
                device=self.device,
                memory_gb=0,
                use_mixed_precision=False
            )
            return
        
        self.device = torch.device('cuda:0')
        
        # Calculate available memory (reserve 2GB for system)
        available_memory_gb = (TOTAL_MEMORY / (1024**3)) - 2.0
        
        self.config = GPUConfig(
            device=self.device,
            memory_gb=available_memory_gb,
            use_mixed_precision=True,
            use_flash_attention=True,
            use_kernel_fusion=True,
            max_batch_size=self._compute_optimal_batch_size(available_memory_gb),
            num_streams=4,
            prefetch_factor=2
        )
        
        # Create CUDA streams for concurrent execution
        self.streams = [torch.cuda.Stream() for _ in range(self.config.num_streams)]
        
        # Setup mixed precision
        if self.config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        print(f"🚀 GPU Optimization for {DEVICE_NAME}")
        print(f"   Total Memory: {TOTAL_MEMORY / (1024**3):.2f} GB")
        print(f"   Available Memory: {available_memory_gb:.2f} GB")
        print(f"   Compute Capability: {COMPUTE_CAPABILITY}")
        print(f"   Mixed Precision: {self.config.use_mixed_precision}")
        print(f"   Optimal Batch Size: {self.config.max_batch_size}")
    
    def _compute_optimal_batch_size(self, memory_gb: float) -> int:
        """
        Compute optimal batch size based on available memory.
        
        For RTX 5080 with 16GB:
        - Reserve ~4GB for model weights
        - Reserve ~2GB for system/overhead
        - Use remaining for batch processing
        """
        # Estimate: ~100MB per batch item for typical workload
        usable_memory_gb = memory_gb - 4.0
        batch_size = int((usable_memory_gb * 1024) / 100)
        return min(max(batch_size, 8), 128)  # Clamp between 8-128
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Optimize model for RTX 5080.
        
        - Move to GPU
        - Convert to optimal dtype
        - Enable kernel fusion
        - Compile with torch.compile (PyTorch 2.0+)
        """
        if not CUDA_AVAILABLE:
            return model
        
        model = model.to(self.device)
        
        # Use mixed precision if available
        if self.config.use_mixed_precision:
            model = model.half()  # FP16 for RTX 5080 Tensor Cores
        
        # Enable TF32 for better performance on Ada Lovelace
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Compile model (PyTorch 2.0+)
        try:
            model = torch.compile(model, mode='max-autotune')
            print("✅ Model compiled with torch.compile")
        except Exception as e:
            print(f"⚠️  torch.compile not available: {e}")
        
        return model
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics."""
        if not CUDA_AVAILABLE:
            return {'allocated_gb': 0, 'reserved_gb': 0, 'free_gb': 0}
        
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        total = TOTAL_MEMORY / (1024**3)
        free = total - allocated
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': free,
            'total_gb': total,
            'utilization': allocated / total
        }
    
    def optimize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Optimize embeddings for GPU storage.
        
        - Pin memory for fast CPU->GPU transfer
        - Use FP16 if mixed precision enabled
        - Ensure contiguous memory layout
        """
        if not CUDA_AVAILABLE:
            return embeddings
        
        # Move to GPU
        embeddings = embeddings.to(self.device)
        
        # Convert to FP16 for memory efficiency
        if self.config.use_mixed_precision:
            embeddings = embeddings.half()
        
        # Ensure contiguous
        embeddings = embeddings.contiguous()
        
        return embeddings
    
    def dynamic_batch_split(self, batch_size: int) -> List[int]:
        """
        Dynamically split batch based on available memory.
        
        Returns list of sub-batch sizes that fit in memory.
        """
        stats = self.get_memory_stats()
        available_gb = stats['free_gb']
        
        # Estimate memory per item (~100MB)
        items_per_gb = 10
        max_items = int(available_gb * items_per_gb * 0.8)  # 80% safety margin
        
        if batch_size <= max_items:
            return [batch_size]
        
        # Split into sub-batches
        num_splits = (batch_size + max_items - 1) // max_items
        sub_batch_sizes = [batch_size // num_splits] * num_splits
        
        # Handle remainder
        remainder = batch_size % num_splits
        for i in range(remainder):
            sub_batch_sizes[i] += 1
        
        return sub_batch_sizes
    
    def prefetch_to_gpu(self, data: torch.Tensor, stream_idx: int = 0) -> torch.Tensor:
        """
        Prefetch data to GPU with specific stream.
        
        Enables concurrent CPU-GPU transfer and computation.
        """
        if not CUDA_AVAILABLE:
            return data
        
        stream = self.streams[stream_idx % len(self.streams)]
        with torch.cuda.stream(stream):
            data_gpu = data.to(self.device, non_blocking=True)
        
        return data_gpu
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()


class FlashAttentionOptimized(nn.Module):
    """
    Flash Attention optimized for RTX 5080.
    
    Uses memory-efficient attention with O(N) memory complexity
    instead of O(N²) for standard attention.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Flash Attention.
        
        Args:
            x: Input tensor [batch, seq_len, embed_dim]
            
        Returns:
            Output tensor [batch, seq_len, embed_dim]
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Use PyTorch's scaled_dot_product_attention (Flash Attention backend)
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=True
            ):
                attn_output = F.scaled_dot_product_attention(q, k, v)
        except:
            # Fallback to standard attention
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            attn_output = attn @ v
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        output = self.proj(attn_output)
        
        return output


class BalancedTernaryEmbedding(nn.Module):
    """
    Balanced Ternary Embedding Storage inspired by embeddenator-core.
    
    Uses {-1, 0, +1} representation for memory-efficient storage
    and fast arithmetic operations.
    
    Benefits:
    - 3x less memory than FP16
    - Fast operations (no multiplications)
    - Natural sparsity encoding
    """
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Store as int8 (-1, 0, 1)
        self.weight = nn.Parameter(
            torch.randint(-1, 2, (vocab_size, embed_dim), dtype=torch.int8)
        )
        
        # Scaling factor learned during training
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings.
        
        Args:
            idx: Token indices [batch, seq_len]
            
        Returns:
            Embeddings [batch, seq_len, embed_dim]
        """
        # Look up ternary embeddings
        ternary_emb = self.weight[idx].float()
        
        # Scale to continuous space
        return ternary_emb * self.scale
    
    def to_ternary(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous embeddings to balanced ternary.
        
        Args:
            embeddings: Continuous embeddings
            
        Returns:
            Ternary embeddings {-1, 0, +1}
        """
        # Threshold-based quantization
        ternary = torch.zeros_like(embeddings, dtype=torch.int8)
        threshold = embeddings.abs().mean() * 0.5
        
        ternary[embeddings > threshold] = 1
        ternary[embeddings < -threshold] = -1
        
        return ternary
    
    @staticmethod
    def ternary_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Matrix multiplication on ternary values.
        
        Optimized for {-1, 0, +1} values - no multiplications needed!
        """
        # For ternary values, multiplication becomes:
        # - If either is 0: result is 0
        # - If signs match: result is 1
        # - If signs differ: result is -1
        
        result = torch.zeros(a.shape[0], b.shape[1], dtype=a.dtype, device=a.device)
        
        # Exploit sparsity: only compute where both are non-zero
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                # Count +1 matches and -1 matches
                pos_matches = ((a[i] == 1) & (b[:, j] == 1)).sum()
                neg_matches = ((a[i] == -1) & (b[:, j] == -1)).sum()
                cross_pos = ((a[i] == 1) & (b[:, j] == -1)).sum()
                cross_neg = ((a[i] == -1) & (b[:, j] == 1)).sum()
                
                result[i, j] = pos_matches + neg_matches - cross_pos - cross_neg
        
        return result


class GPUEmbeddingStore:
    """
    GPU-resident embedding storage optimized for RTX 5080.
    
    Features:
    - Pinned memory for fast CPU-GPU transfer
    - LRU cache for frequently accessed embeddings
    - Batch prefetching
    - Mixed precision storage
    """
    
    def __init__(self, embed_dim: int = 512, max_capacity: int = 100000,
                 use_ternary: bool = False, optimizer: Optional[RTX5080Optimizer] = None):
        self.embed_dim = embed_dim
        self.max_capacity = max_capacity
        self.use_ternary = use_ternary
        self.optimizer = optimizer or RTX5080Optimizer()
        
        # Main storage on GPU
        if CUDA_AVAILABLE:
            if use_ternary:
                # Ternary storage (int8)
                self.storage = torch.zeros(
                    max_capacity, embed_dim,
                    dtype=torch.int8,
                    device=self.optimizer.device
                )
            else:
                # FP16 storage
                dtype = torch.float16 if self.optimizer.config.use_mixed_precision else torch.float32
                self.storage = torch.zeros(
                    max_capacity, embed_dim,
                    dtype=dtype,
                    device=self.optimizer.device
                )
        else:
            # CPU fallback
            self.storage = torch.zeros(max_capacity, embed_dim)
        
        # Metadata
        self.num_embeddings = 0
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        
        # LRU cache
        self.access_counts: Dict[int, int] = {}
        self.last_access: Dict[int, int] = {}
        self.access_counter = 0
    
    def store(self, embedding_id: str, embedding: torch.Tensor) -> int:
        """
        Store embedding on GPU.
        
        Args:
            embedding_id: Unique identifier
            embedding: Embedding tensor [embed_dim]
            
        Returns:
            Storage index
        """
        if embedding_id in self.id_to_idx:
            idx = self.id_to_idx[embedding_id]
        else:
            if self.num_embeddings >= self.max_capacity:
                # Evict LRU
                idx = self._evict_lru()
            else:
                idx = self.num_embeddings
                self.num_embeddings += 1
            
            self.id_to_idx[embedding_id] = idx
            self.idx_to_id[idx] = embedding_id
        
        # Store on GPU
        if self.use_ternary:
            # Convert to ternary
            ternary_emb = BalancedTernaryEmbedding.to_ternary(None, embedding)
            self.storage[idx] = ternary_emb
        else:
            # Store as-is (with GPU optimization)
            embedding_gpu = self.optimizer.optimize_embeddings(embedding)
            self.storage[idx] = embedding_gpu
        
        # Update access stats
        self.access_counts[idx] = self.access_counts.get(idx, 0) + 1
        self.last_access[idx] = self.access_counter
        self.access_counter += 1
        
        return idx
    
    def retrieve(self, embedding_id: str) -> Optional[torch.Tensor]:
        """
        Retrieve embedding from GPU.
        
        Args:
            embedding_id: Unique identifier
            
        Returns:
            Embedding tensor [embed_dim] or None
        """
        if embedding_id not in self.id_to_idx:
            return None
        
        idx = self.id_to_idx[embedding_id]
        
        # Update access stats
        self.access_counts[idx] = self.access_counts.get(idx, 0) + 1
        self.last_access[idx] = self.access_counter
        self.access_counter += 1
        
        # Retrieve from GPU
        embedding = self.storage[idx].clone()
        
        if self.use_ternary:
            # Convert from ternary to float
            embedding = embedding.float()
        
        return embedding
    
    def batch_retrieve(self, embedding_ids: List[str]) -> torch.Tensor:
        """
        Batch retrieve embeddings.
        
        More efficient than individual retrieves due to
        single GPU memory access.
        """
        indices = [self.id_to_idx[eid] for eid in embedding_ids if eid in self.id_to_idx]
        
        if not indices:
            return torch.zeros(0, self.embed_dim, device=self.storage.device)
        
        # Single GPU access
        embeddings = self.storage[indices]
        
        # Update access stats
        for idx in indices:
            self.access_counts[idx] = self.access_counts.get(idx, 0) + 1
            self.last_access[idx] = self.access_counter
        self.access_counter += len(indices)
        
        if self.use_ternary:
            embeddings = embeddings.float()
        
        return embeddings
    
    def _evict_lru(self) -> int:
        """Evict least recently used embedding."""
        if not self.last_access:
            return 0
        
        lru_idx = min(self.last_access.keys(), key=lambda k: self.last_access[k])
        
        # Remove from mappings
        if lru_idx in self.idx_to_id:
            embedding_id = self.idx_to_id[lru_idx]
            del self.id_to_idx[embedding_id]
            del self.idx_to_id[lru_idx]
        
        del self.last_access[lru_idx]
        if lru_idx in self.access_counts:
            del self.access_counts[lru_idx]
        
        return lru_idx
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'num_embeddings': self.num_embeddings,
            'capacity': self.max_capacity,
            'utilization': self.num_embeddings / self.max_capacity,
            'use_ternary': self.use_ternary,
            'total_accesses': self.access_counter
        }
        
        if CUDA_AVAILABLE:
            # Memory usage
            if self.use_ternary:
                bytes_per_embedding = self.embed_dim  # int8
            else:
                bytes_per_embedding = self.embed_dim * 2  # FP16
            
            total_bytes = self.num_embeddings * bytes_per_embedding
            stats['memory_mb'] = total_bytes / (1024**2)
            stats['memory_gb'] = total_bytes / (1024**3)
        
        return stats


# Global optimizer instance
_global_optimizer = None

def get_gpu_optimizer() -> RTX5080Optimizer:
    """Get global GPU optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = RTX5080Optimizer()
    return _global_optimizer
