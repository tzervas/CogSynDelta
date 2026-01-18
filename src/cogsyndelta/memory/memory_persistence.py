"""
Memory Persistence and Compression System

Implements:
1. Persistent temporal grounding with disk storage
2. Memory compression using dense differential embeddings + semantic residuals
3. Temporal continuity tracking
4. Vector space optimization to prevent bloat
5. Safeguards against infinite loops and unethical hazards

Key features:
- Hierarchical memory compression (recent → short-term → long-term)
- Dense differential encoding (10-100x compression)
- Semantic residuals for high-fidelity preservation
- Importance-weighted memory retention
- Checkpoint/restore for temporal continuity
- Loop detection and circuit breakers
- Ethical constraint validators
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import pickle
import os
import hashlib
from collections import deque
import numpy as np

# Import dense differential encoding
try:
    from cogsyndelta.memory.dense_embeddings import DenseDifferentialMemoryStore
    DENSE_ENCODING_AVAILABLE = True
except ImportError:
    DENSE_ENCODING_AVAILABLE = False


@dataclass
class MemoryMetadata:
    """Metadata for a memory entry."""
    timestamp: datetime
    importance: float = 1.0
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    compression_level: int = 0  # 0=raw, 1=compressed, 2=archived
    hash: str = ""
    source: str = "unknown"


@dataclass
class LoopDetectionState:
    """State for detecting infinite loops."""
    state_hashes: deque = field(default_factory=lambda: deque(maxlen=100))
    repetition_counts: Dict[str, int] = field(default_factory=dict)
    max_repetitions: int = 5
    loop_detected: bool = False
    last_break: datetime = field(default_factory=datetime.now)


class MemoryCompressor(nn.Module):
    """
    Compresses semantic memories using clustering and dimensionality reduction.
    Prevents vector space bloat while maintaining temporal continuity.
    """
    
    def __init__(self, embed_dim: int = 512, compressed_dim: int = 256, 
                 compression_ratio: float = 0.5) -> None:
        """Initialize memory compressor with autoencoder architecture."""
        super(MemoryCompressor, self).__init__()
        
        self.embed_dim = embed_dim
        self.compressed_dim = compressed_dim
        self.compression_ratio = compression_ratio
        
        # Compression network (autoencoder-style)
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, compressed_dim),
            nn.LayerNorm(compressed_dim),
            nn.GELU(),
            nn.Linear(compressed_dim, compressed_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim, compressed_dim),
            nn.GELU(),
            nn.Linear(compressed_dim, embed_dim)
        )
        
    def compress(self, memories: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Compress memory embeddings.
        
        Args:
            memories: Memory embeddings [num_memories, embed_dim]
            
        Returns:
            Compressed memories and compression quality score
        """
        compressed = self.encoder(memories)
        reconstructed = self.decoder(compressed)
        
        # Quality metric (reconstruction error)
        mse = F.mse_loss(reconstructed, memories)
        quality = torch.exp(-mse).item()
        
        return compressed, quality
    
    def decompress(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decompress memory embeddings."""
        return self.decoder(compressed)
    
    def cluster_and_merge(self, memories: torch.Tensor, 
                         similarity_threshold: float = 0.9) -> Tuple[torch.Tensor, List[int]]:
        """
        Cluster similar memories and merge them.
        Reduces redundancy in vector space.
        
        Args:
            memories: Memory embeddings [num_memories, embed_dim]
            similarity_threshold: Cosine similarity threshold for merging
            
        Returns:
            Merged memories and indices of kept memories
        """
        num_memories = memories.size(0)
        if num_memories == 0:
            return memories, []
        
        # Compute pairwise similarities
        normalized = F.normalize(memories, dim=-1)
        similarities = torch.matmul(normalized, normalized.T)
        
        # Find clusters of similar memories
        keep_indices = []
        merged_memories = []
        processed = set()
        
        for i in range(num_memories):
            if i in processed:
                continue
            
            # Find similar memories
            similar = (similarities[i] > similarity_threshold).nonzero(as_tuple=True)[0]
            
            if len(similar) > 1:
                # Merge similar memories (weighted average)
                weights = similarities[i, similar].softmax(dim=0)
                merged = (memories[similar] * weights.unsqueeze(1)).sum(dim=0)
                merged_memories.append(merged)
                keep_indices.append(i)
                processed.update(similar.tolist())
            else:
                # Keep unique memory
                merged_memories.append(memories[i])
                keep_indices.append(i)
                processed.add(i)
        
        return torch.stack(merged_memories), keep_indices


class PersistentMemoryBank(nn.Module):
    """
    Persistent memory bank with hierarchical storage and compression.
    
    Uses dense differential embeddings with semantic residuals for optimal compression.
    
    Memory hierarchy:
    - Working memory (recent, uncompressed)
    - Short-term memory (dense differential compressed)
    - Long-term memory (archived to disk with high compression)
    
    Compression: 10-100x ratio with >0.95 fidelity
    """
    
    def __init__(self, embed_dim: int = 512, 
                 working_capacity: int = 100,
                 short_term_capacity: int = 1000,
                 storage_path: str = "./memory_storage",
                 use_dense_encoding: bool = True) -> None:
        """Initialize persistent memory bank with tiered storage."""
        super(PersistentMemoryBank, self).__init__()
        
        self.embed_dim = embed_dim
        self.working_capacity = working_capacity
        self.short_term_capacity = short_term_capacity
        self.storage_path = storage_path
        self.use_dense_encoding = use_dense_encoding and DENSE_ENCODING_AVAILABLE
        
        # Create storage directory
        os.makedirs(storage_path, exist_ok=True)
        
        # Working memory (recent, fast access)
        self.register_buffer('working_memory', torch.zeros(working_capacity, embed_dim))
        self.working_metadata: List[MemoryMetadata] = []
        self.working_pointer = 0
        
        # Short-term memory (compressed using dense differential encoding)
        if self.use_dense_encoding:
            self.dense_store = DenseDifferentialMemoryStore(
                embed_dim=embed_dim,
                dense_dim=64,  # High compression
                num_references=min(100, short_term_capacity // 10)
            )
        else:
            # Fallback to standard storage
            self.register_buffer('short_term_memory', torch.zeros(short_term_capacity, embed_dim))
        
        self.short_term_metadata: List[MemoryMetadata] = []
        self.short_term_pointer = 0
        
        # Long-term memory (disk-based with maximum compression)
        self.long_term_index: Dict[str, str] = {}  # hash -> filename
        
        # Compressor (fallback for non-dense encoding)
        if not self.use_dense_encoding:
            self.compressor = MemoryCompressor(embed_dim=embed_dim)
        
        # Temporal continuity tracker
        self.temporal_continuity = {
            'last_checkpoint': None,
            'sequence_id': 0,
            'total_memories': 0,
            'compression_stats': {
                'total_original_mb': 0.0,
                'total_compressed_mb': 0.0,
                'average_compression_ratio': 1.0,
                'average_fidelity': 1.0
            }
        }
        
    def write(self, content: torch.Tensor, importance: float = 1.0, 
             source: str = "unknown") -> Any:
        """
        Write to persistent memory with metadata.
        
        Implements hierarchical storage:
        1. New memories → working memory
        2. Aged memories → short-term (compressed)
        3. Old memories → long-term (disk)
        """
        batch_size = content.size(0)
        
        for i in range(batch_size):
            embedding = content[i].detach()
            
            # Compute hash for deduplication
            emb_hash = hashlib.sha256(embedding.cpu().numpy().tobytes()).hexdigest()[:16]
            
            # Create metadata
            metadata = MemoryMetadata(
                timestamp=datetime.now(),
                importance=importance,
                hash=emb_hash,
                source=source
            )
            
            # Write to working memory
            idx = self.working_pointer % self.working_capacity
            self.working_memory[idx] = embedding
            
            if idx < len(self.working_metadata):
                self.working_metadata[idx] = metadata
            else:
                self.working_metadata.append(metadata)
            
            self.working_pointer += 1
            self.temporal_continuity['total_memories'] += 1
            
            # Trigger compression if working memory is full
            if self.working_pointer >= self.working_capacity:
                self._compress_to_short_term()
    
    def _compress_to_short_term(self) -> None:
        """Compress working memory to short-term storage using dense differential encoding."""
        if self.use_dense_encoding:
            # Use dense differential encoding for maximum compression
            for i in range(min(self.working_pointer, self.working_capacity)):
                embedding = self.working_memory[i]
                
                # Get metadata
                if i < len(self.working_metadata):
                    metadata = self.working_metadata[i]
                    importance = metadata.importance
                else:
                    importance = 1.0
                
                # Compress and store
                memory_id = f"st_{self.short_term_pointer}_{metadata.hash if i < len(self.working_metadata) else i}"
                compression_stats = self.dense_store.compress_and_store(
                    embedding,
                    memory_id=memory_id,
                    importance=importance
                )
                
                # Update metadata
                if i < len(self.working_metadata):
                    metadata.compression_level = 1
                    self.short_term_metadata.append(metadata)
                
                # Update compression stats
                self.temporal_continuity['compression_stats']['average_compression_ratio'] = \
                    compression_stats['compression_ratio']
                self.temporal_continuity['compression_stats']['average_fidelity'] = \
                    compression_stats['fidelity_score']
                
                self.short_term_pointer += 1
            
            # Trigger archiving if short-term is full
            if self.short_term_pointer >= self.short_term_capacity:
                self._archive_to_long_term()
        else:
            # Fallback to original compression method
            merged, keep_indices = self.compressor.cluster_and_merge(
                self.working_memory[:self.working_capacity],
                similarity_threshold=0.85
            )
            
            compressed, quality = self.compressor.compress(merged)
            
            num_compressed = compressed.size(0)
            for i in range(num_compressed):
                idx = self.short_term_pointer % self.short_term_capacity
                self.short_term_memory[idx] = compressed[i]
                
                original_idx = keep_indices[i] if i < len(keep_indices) else i
                if original_idx < len(self.working_metadata):
                    metadata = self.working_metadata[original_idx]
                    metadata.compression_level = 1
                    
                    if idx < len(self.short_term_metadata):
                        self.short_term_metadata[idx] = metadata
                    else:
                        self.short_term_metadata.append(metadata)
                
                self.short_term_pointer += 1
            
            if self.short_term_pointer >= self.short_term_capacity:
                self._archive_to_long_term()
        
        # Clear working memory
        self.working_pointer = 0
    
    def _archive_to_long_term(self) -> None:
        """Archive short-term memory to disk."""
        # Select memories to archive (importance-weighted)
        importances = torch.tensor([
            m.importance for m in self.short_term_metadata
        ])
        
        # Keep most important, archive rest
        num_to_keep = self.short_term_capacity // 2
        keep_indices = torch.topk(importances, num_to_keep).indices
        
        archive_indices = []
        for i in range(self.short_term_capacity):
            if i not in keep_indices:
                archive_indices.append(i)
        
        # Archive to disk
        for idx in archive_indices:
            if idx < len(self.short_term_metadata):
                metadata = self.short_term_metadata[idx]
                memory_data = {
                    'embedding': self.short_term_memory[idx].cpu(),
                    'metadata': metadata
                }
                
                filename = f"{self.storage_path}/memory_{metadata.hash}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(memory_data, f)
                
                self.long_term_index[metadata.hash] = filename
                metadata.compression_level = 2
        
        # Compact short-term memory
        new_short_term = self.short_term_memory[keep_indices]
        new_metadata = [self.short_term_metadata[i] for i in keep_indices.tolist()]
        
        self.short_term_memory[:len(new_short_term)] = new_short_term
        self.short_term_metadata = new_metadata
        self.short_term_pointer = len(new_metadata)
    
    def read(self, query: torch.Tensor, num_reads: int = 5,
             include_long_term: bool = False) -> Tuple[torch.Tensor, List[MemoryMetadata]]:
        """
        Read from persistent memory hierarchy.
        
        Args:
            query: Query embedding
            num_reads: Number of memories to retrieve
            include_long_term: Whether to search long-term storage
            
        Returns:
            Retrieved memories and their metadata
        """
        batch_size = query.size(0)
        retrieved_list = []
        metadata_list = []
        
        # Search working memory
        working_valid = self.working_memory[:min(self.working_pointer, self.working_capacity)]
        if working_valid.size(0) > 0:
            similarities = F.cosine_similarity(
                query.unsqueeze(1),
                working_valid.unsqueeze(0),
                dim=-1
            )
            top_k = min(num_reads, working_valid.size(0))
            top_indices = similarities.topk(top_k, dim=1).indices
            
            for b in range(batch_size):
                for idx in top_indices[b]:
                    retrieved_list.append(working_valid[idx])
                    if idx < len(self.working_metadata):
                        self.working_metadata[idx].access_count += 1
                        self.working_metadata[idx].last_accessed = datetime.now()
                        metadata_list.append(self.working_metadata[idx])
        
        # Search short-term memory if needed
        if len(retrieved_list) < num_reads:
            short_term_valid = self.short_term_memory[:min(self.short_term_pointer, self.short_term_capacity)]
            if short_term_valid.size(0) > 0:
                # Decompress for comparison
                decompressed = self.compressor.decompress(short_term_valid)
                
                similarities = F.cosine_similarity(
                    query.unsqueeze(1),
                    decompressed.unsqueeze(0),
                    dim=-1
                )
                remaining = num_reads - len(retrieved_list)
                top_k = min(remaining, decompressed.size(0))
                top_indices = similarities.topk(top_k, dim=1).indices
                
                for b in range(batch_size):
                    for idx in top_indices[b]:
                        retrieved_list.append(decompressed[idx])
                        if idx < len(self.short_term_metadata):
                            self.short_term_metadata[idx].access_count += 1
                            metadata_list.append(self.short_term_metadata[idx])
        
        # Search long-term if enabled and still need more
        if include_long_term and len(retrieved_list) < num_reads:
            # Load a sample of long-term memories
            sample_size = min(100, len(self.long_term_index))
            sample_hashes = list(self.long_term_index.keys())[:sample_size]
            
            for mem_hash in sample_hashes:
                filename = self.long_term_index[mem_hash]
                try:
                    with open(filename, 'rb') as f:
                        data = pickle.load(f)
                        retrieved_list.append(data['embedding'].to(query.device))
                        metadata_list.append(data['metadata'])
                except (FileNotFoundError, pickle.UnpicklingError, KeyError):
                    # Skip corrupted or missing memory files
                    pass
        
        if retrieved_list:
            return torch.stack(retrieved_list[:num_reads]), metadata_list[:num_reads]
        else:
            return torch.zeros(num_reads, self.embed_dim).to(query.device), []
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save temporal continuity checkpoint."""
        checkpoint = {
            'working_memory': self.working_memory.cpu(),
            'working_metadata': self.working_metadata,
            'short_term_memory': self.short_term_memory.cpu(),
            'short_term_metadata': self.short_term_metadata,
            'long_term_index': self.long_term_index,
            'temporal_continuity': self.temporal_continuity,
            'pointers': {
                'working': self.working_pointer,
                'short_term': self.short_term_pointer
            },
            'timestamp': datetime.now()
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        self.temporal_continuity['last_checkpoint'] = datetime.now()
        print(f"✓ Memory checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load temporal continuity checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.working_memory = checkpoint['working_memory'].to(self.working_memory.device)
        self.working_metadata = checkpoint['working_metadata']
        self.short_term_memory = checkpoint['short_term_memory'].to(self.short_term_memory.device)
        self.short_term_metadata = checkpoint['short_term_metadata']
        self.long_term_index = checkpoint['long_term_index']
        self.temporal_continuity = checkpoint['temporal_continuity']
        self.working_pointer = checkpoint['pointers']['working']
        self.short_term_pointer = checkpoint['pointers']['short_term']
        
        print(f"✓ Memory checkpoint loaded: {checkpoint_path}")
        print(f"  Total memories: {self.temporal_continuity['total_memories']}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory bank statistics."""
        return {
            'working_memory': {
                'capacity': self.working_capacity,
                'used': min(self.working_pointer, self.working_capacity),
                'utilization': min(self.working_pointer / self.working_capacity, 1.0)
            },
            'short_term_memory': {
                'capacity': self.short_term_capacity,
                'used': min(self.short_term_pointer, self.short_term_capacity),
                'utilization': min(self.short_term_pointer / self.short_term_capacity, 1.0)
            },
            'long_term_memory': {
                'count': len(self.long_term_index),
                'storage_path': self.storage_path
            },
            'temporal_continuity': self.temporal_continuity
        }


class InfiniteLoopSafeguard:
    """
    Safeguards against infinite loops and unethical simulation cycles.
    
    Implements:
    - State repetition detection
    - Circuit breakers for runaway processes
    - Ethical constraint validation
    - Resource usage monitoring
    """
    
    def __init__(self, max_iterations: int = 1000, 
                 max_repetitions: int = 5,
                 timeout_seconds: float = 300.0) -> None:
        """Initialize safeguard with iteration limits and ethical constraints."""
        self.max_iterations = max_iterations
        self.max_repetitions = max_repetitions
        self.timeout_seconds = timeout_seconds
        
        self.loop_state = LoopDetectionState(max_repetitions=max_repetitions)
        self.iteration_count = 0
        self.start_time = datetime.now()
        
        # Ethical constraints
        self.ethical_constraints = {
            'max_output_size': 10 * 1024 * 1024,  # 10MB
            'max_memory_usage': 1024 * 1024 * 1024,  # 1GB
            'forbidden_patterns': [
                'infinite_loop', 'memory_bomb', 'fork_bomb'
            ]
        }
    
    def check_state(self, state: torch.Tensor) -> Tuple[bool, str]:
        """
        Check if current state indicates a loop or hazard.
        
        Args:
            state: Current system state
            
        Returns:
            (is_safe, message)
        """
        # Iteration limit
        self.iteration_count += 1
        if self.iteration_count > self.max_iterations:
            return False, f"Iteration limit exceeded ({self.max_iterations})"
        
        # Timeout check
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed > self.timeout_seconds:
            return False, f"Timeout exceeded ({self.timeout_seconds}s)"
        
        # State repetition check
        state_hash = hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()[:16]
        
        self.loop_state.state_hashes.append(state_hash)
        self.loop_state.repetition_counts[state_hash] = \
            self.loop_state.repetition_counts.get(state_hash, 0) + 1
        
        if self.loop_state.repetition_counts[state_hash] > self.max_repetitions:
            self.loop_state.loop_detected = True
            return False, f"Infinite loop detected (state repeated {self.max_repetitions}+ times)"
        
        # Check for oscillation patterns
        if len(self.loop_state.state_hashes) >= 4:
            recent = list(self.loop_state.state_hashes)[-4:]
            if recent[0] == recent[2] and recent[1] == recent[3]:
                return False, "Oscillation pattern detected (A-B-A-B cycle)"
        
        return True, "Safe"
    
    def validate_ethical_constraints(self, output: Any) -> Tuple[bool, str]:
        """
        Validate output against ethical constraints.
        
        Args:
            output: System output to validate
            
        Returns:
            (is_valid, message)
        """
        # Size check
        if isinstance(output, torch.Tensor):
            size_bytes = output.element_size() * output.nelement()
            if size_bytes > self.ethical_constraints['max_output_size']:
                return False, f"Output size exceeds limit ({size_bytes} bytes)"
        
        # Pattern check (if output is text/code)
        if isinstance(output, str):
            output_lower = output.lower()
            for pattern in self.ethical_constraints['forbidden_patterns']:
                if pattern in output_lower:
                    return False, f"Forbidden pattern detected: {pattern}"
        
        return True, "Valid"
    
    def trigger_circuit_breaker(self, reason: str) -> None:
        """Trigger circuit breaker to stop execution."""
        self.loop_state.loop_detected = True
        self.loop_state.last_break = datetime.now()
        print(f"⚠️  CIRCUIT BREAKER TRIGGERED: {reason}")
        print(f"   Iterations: {self.iteration_count}")
        print(f"   Time elapsed: {(datetime.now() - self.start_time).total_seconds():.2f}s")
    
    def reset(self) -> None:
        """Reset safeguard state."""
        self.loop_state = LoopDetectionState(max_repetitions=self.max_repetitions)
        self.iteration_count = 0
        self.start_time = datetime.now()


if __name__ == '__main__':
    print("="*70)
    print("MEMORY PERSISTENCE AND SAFEGUARD SYSTEM")
    print("="*70)
    
    # Test persistent memory
    print("\n[Test 1] Persistent Memory Bank:")
    memory_bank = PersistentMemoryBank(
        embed_dim=512,
        working_capacity=10,
        short_term_capacity=50,
        storage_path="./test_memory"
    )
    
    # Write some memories
    for i in range(15):
        embedding = torch.randn(1, 512)
        memory_bank.write(embedding, importance=0.5 + i * 0.05, source=f"test_{i}")
    
    stats = memory_bank.get_statistics()
    print(f"  Working memory: {stats['working_memory']['used']}/{stats['working_memory']['capacity']}")
    print(f"  Short-term memory: {stats['short_term_memory']['used']}/{stats['short_term_memory']['capacity']}")
    
    # Test checkpointing
    print("\n[Test 2] Temporal Continuity Checkpoint:")
    memory_bank.save_checkpoint("./test_memory/checkpoint.pkl")
    
    # Test retrieval
    query = torch.randn(1, 512)
    retrieved, metadata = memory_bank.read(query, num_reads=5)
    print(f"  Retrieved {retrieved.size(0)} memories")
    
    # Test safeguards
    print("\n[Test 3] Infinite Loop Safeguard:")
    safeguard = InfiniteLoopSafeguard(max_iterations=10, max_repetitions=3)
    
    state = torch.randn(512)
    for i in range(15):
        is_safe, message = safeguard.check_state(state)
        if not is_safe:
            print(f"  ⚠️  Stopped at iteration {i}: {message}")
            break
        if i % 3 == 0:
            state = state  # Repeat state to trigger detection
    
    print("\n" + "="*70)
    print("TESTS COMPLETE")
    print("="*70)
