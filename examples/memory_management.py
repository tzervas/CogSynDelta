"""
Memory Management Example

This example demonstrates:
1. Using the tiered memory system
2. Storing and retrieving embeddings
3. Memory compression and persistence
"""

import torch

from cogsyndelta.memory.active_memory import TieredMemoryManager
from cogsyndelta.memory.dense_embeddings import DenseEmbedding


def main() -> None:
    """Demonstrate memory management capabilities."""
    print("=" * 60)
    print("CogSynDelta Memory Management Example")
    print("=" * 60)

    # Initialize memory manager
    embed_dim = 512
    memory_manager = TieredMemoryManager(embed_dim=embed_dim)
    compressor = DenseEmbedding(embed_dim=embed_dim)

    print(f"\nMemory system initialized with {embed_dim}-dimensional embeddings")

    # Create some sample data
    num_memories = 100
    print(f"\nStoring {num_memories} memory embeddings...")

    for i in range(num_memories):
        # Generate random embedding
        embedding = torch.randn(1, embed_dim)
        memory_id = f"memory_{i}"

        # Store in appropriate tier
        if i < 20:
            # Active memory (most recent/important)
            memory_manager.active_memory[memory_id] = embedding
            print(f"  ✓ Stored {memory_id} in active memory")
        elif i < 50:
            # Short-term memory
            memory_manager.short_term[memory_id] = embedding
            if i % 10 == 0:
                print(f"  ✓ Stored {memory_id} in short-term memory")
        else:
            # Long-term memory (compress it)
            compressed = compressor.compress(embedding)
            memory_manager.long_term[memory_id] = compressed
            if i % 20 == 0:
                print(f"  ✓ Stored {memory_id} in long-term memory (compressed)")

    # Query memories
    print("\n" + "=" * 60)
    print("Querying Memory System")
    print("=" * 60)

    query = torch.randn(1, embed_dim)
    print("\nSearching for similar memories to random query...")

    # Search in active memory
    best_match = None
    best_similarity = -float("inf")

    for mem_id, mem_emb in memory_manager.active_memory.items():
        similarity = torch.cosine_similarity(query, mem_emb).item()
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = mem_id

    print(f"  Best match in active memory: {best_match}")
    print(f"  Similarity score: {best_similarity:.4f}")

    # Memory statistics
    print("\n" + "=" * 60)
    print("Memory Statistics")
    print("=" * 60)

    active_count = len(memory_manager.active_memory)
    short_term_count = len(memory_manager.short_term)
    long_term_count = len(memory_manager.long_term)

    print(f"\nActive Memory: {active_count} items")
    print(f"Short-term Memory: {short_term_count} items")
    print(f"Long-term Memory: {long_term_count} items (compressed)")
    print(f"Total Memories: {active_count + short_term_count + long_term_count}")

    # Demonstrate memory management
    print("\n" + "=" * 60)
    print("Memory Tier Management")
    print("=" * 60)

    print("\nManaging memory tiers...")
    memory_manager.manage_tiers()
    print("  ✓ Memory tiers optimized")

    # Calculate compression ratio
    original_size = num_memories * embed_dim * 4  # 4 bytes per float32
    compressed_size = long_term_count * 64 * 4  # Assuming 64-dim compressed
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

    print("\nCompression Statistics:")
    print(f"  Original size: {original_size / 1024:.2f} KB")
    print(f"  Compressed size: {compressed_size / 1024:.2f} KB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")

    print("\n" + "=" * 60)
    print("Memory management demonstration completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
