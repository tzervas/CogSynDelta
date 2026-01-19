"""
Memory management modules for CogSynDelta.

This module provides tiered memory management with adaptive compression:

Compactor Hierarchy (choose based on needs):
============================================

1. **HighFidelityCompactor** - Guaranteed ≥0.99 fidelity, ~1.33x compression
   - No training required
   - Best for: Active memory, critical data

2. **HybridAdaptiveCompactor** - Trainable, 2-4x compression, ≥0.95 fidelity
   - Learns domain-specific compression
   - Best for: Long-term memory, specialized workloads

3. **LosslessCompactor** - Legacy, requires training for good fidelity
   - Deprecated: Use HybridAdaptiveCompactor instead
"""

from cogsyndelta.memory.active_memory import (
    ActiveMemoryManager,
    HighFidelityCompactor,
    HybridAdaptiveCompactor,
    LosslessCompactor,
    MemoryTier,
    TemporalChainManager,
)

__all__ = [
    "ActiveMemoryManager",
    "HighFidelityCompactor",
    "HybridAdaptiveCompactor",
    "LosslessCompactor",
    "MemoryTier",
    "TemporalChainManager",
]
