"""
Memory management modules for CogSynDelta.

This module provides tiered memory management with adaptive compression:

Compactor Hierarchy (choose based on needs):
============================================

NOTE: Fidelity and compression ratios depend on training. See benchmark_results/
for current measured performance. The values below are theoretical targets.

1. **HighFidelityCompactor** - Target: ≥0.99 fidelity, ~1.33x compression
   - No training required
   - Best for: Active memory, critical data
   - Status: ✅ Production-ready (verified by benchmarks)

2. **ResidualBoostCompactor** - Target: ≥0.95 fidelity, 3-6x compression
   - Uses cascaded residual learning (like neural audio codecs)
   - Best for: High compression with good fidelity
   - Status: ⚠️ Requires training (current fidelity ~0.56)

3. **HybridAdaptiveCompactor** - Target: ≥0.95 fidelity, 2-4x compression
   - Learns domain-specific compression
   - Best for: Long-term memory, specialized workloads
   - Status: ✅ Production-ready (~0.96 fidelity measured)

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
