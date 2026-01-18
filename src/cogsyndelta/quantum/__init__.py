"""
Quantum Computing Integration Modules.

**FUTURE FEATURE - BACKLOGGED**

This module provides extensible quantum computing backends for hybrid
quantum-classical neural networks. It is currently backlogged pending
Python 3.14 ecosystem support for quantum computing packages (cirq, qiskit,
pennylane).

Status:
    - cirq: Blocked by typedunits lacking Python 3.14 wheels
    - qiskit: Blocked by typedunits (transitive dependency)
    - pennylane: Blocked by typedunits (transitive dependency)

The module includes stub implementations that allow the code to be imported
and tested without the actual quantum backends. These stubs will be replaced
with full implementations once the quantum ecosystem catches up.

See ROADMAP.md for timeline estimates.
"""

# Feature availability flag
QUANTUM_AVAILABLE = False
QUANTUM_BACKLOG_REASON = (
    "Quantum computing packages (cirq, qiskit, pennylane) require "
    "typedunits which lacks Python 3.14 wheels. "
    "See ROADMAP.md for updates."
)

__all__ = [
    "QUANTUM_AVAILABLE",
    "QUANTUM_BACKLOG_REASON",
]
