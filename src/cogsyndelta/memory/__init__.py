"""
Memory management modules for CogSynDelta.

This module provides tiered memory management with adaptive compression:

Compactor Hierarchy (choose based on needs):
============================================

1. **HighFidelityCompactor** - Guaranteed ≥0.99 fidelity, ~1.33x compression
   - No training required
   - Best for: Active memory, critical data

2. **ResidualBoostCompactor** - Multi-stage refinement, 3-6x compression, ≥0.95 fidelity
   - Uses cascaded residual learning (like neural audio codecs)
   - Best for: High compression with good fidelity

3. **HybridAdaptiveCompactor** - Trainable, 2-4x compression, ≥0.95 fidelity
   - Learns domain-specific compression
   - Best for: Long-term memory, specialized workloads

4. **LosslessCompactor** - Legacy, requires training for good fidelity
   - Deprecated: Use ResidualBoostCompactor instead
"""

from cogsyndelta.memory.active_memory import (
    ActiveMemoryManager,
    HighFidelityCompactor,
    HybridAdaptiveCompactor,
    LosslessCompactor,
    MemoryTier,
    ResidualBoostCompactor,
    TemporalChainManager,
)

__all__ = [
    "ActiveMemoryManager",
    "HighFidelityCompactor",
    "HybridAdaptiveCompactor",
    "LosslessCompactor",
    "MemoryTier",
    "ResidualBoostCompactor",
    "TemporalChainManager",
]
