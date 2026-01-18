# ADR-0003: Quantum Module Stubbing

**Status**: Accepted
**Date**: 2026-01-18
**Decision Makers**: @tzervas
**Technical Story**: Python 3.14 compatibility for quantum computing packages

## Context

CogSynDelta's architecture includes quantum computing integration for:
- Quantum-enhanced optimization
- Quantum neural network components
- Hybrid classical-quantum algorithms

However, as of January 2026, the major quantum computing libraries have not released Python 3.14 compatible wheels:

| Package | Status | Blocking Issue |
|---------|--------|----------------|
| `qiskit` >= 2.3.0 | ❌ | No cp314 wheels |
| `pennylane` >= 0.44.0 | ❌ | No cp314 wheels |
| `cirq` >= 1.6.0 | ❌ | Depends on `typedunits` which lacks cp314 |

The project requires Python 3.14 for other dependencies and language features.

## Decision

We will **stub the quantum module** to provide classical simulation fallbacks that:

1. Allow the codebase to import and run without quantum backends
2. Emit `FutureWarning` when quantum features are used
3. Provide mathematically equivalent (but slower) classical implementations
4. Enable seamless transition when quantum libraries support Python 3.14

## Rationale

### Why Stub Instead of Downgrade Python

1. **Modern features**: Python 3.14 provides performance improvements and language features we use
2. **Dependency compatibility**: PyTorch 2.9, FastAPI, and other core dependencies target 3.14
3. **Forward-looking**: Quantum libraries will catch up; stubbing is temporary

### Why Not Conditional Import

Conditional imports (`try: import qiskit except: ...`) scattered throughout the codebase would:
- Create maintenance burden
- Make testing harder
- Obscure the intended quantum functionality

Centralized stubs are cleaner.

### Alternatives Considered

#### Option 1: Pin Python to 3.13

- **Pros**: Quantum libraries work
- **Cons**: Lose Python 3.14 features, other dependency issues
- **Why Rejected**: Too many dependencies require 3.14

#### Option 2: Remove quantum features entirely

- **Pros**: Simpler codebase
- **Cons**: Loses planned functionality, significant architectural change
- **Why Rejected**: Quantum integration is a core differentiator

#### Option 3: Use Docker with Python 3.13 for quantum only

- **Pros**: Best of both worlds
- **Cons**: Operational complexity, inter-process communication overhead
- **Why Rejected**: Adds deployment complexity

## Consequences

### Positive

- Clean imports throughout codebase (`from cogsyndelta.quantum import ...`)
- Tests pass on Python 3.14
- Clear documentation of stub mode
- Easy migration path when libraries catch up

### Negative

- Quantum operations run as classical simulation (slower, no quantum advantage)
- Mitigation: Clear warnings, documented as "stub mode"
- Users wanting quantum must use Python 3.13 separately

### Neutral

- Stub implementations are mathematically correct, just classical
- Performance benchmarks clearly indicate "stub mode" results

## Implementation

Location: `src/cogsyndelta/quantum/quantum_compute.py`

Key patterns:
```python
import warnings
warnings.warn(
    "Quantum features running in stub mode (classical simulation). "
    "For actual quantum computation, use Python 3.13 with quantum extras.",
    FutureWarning,
    stacklevel=2
)
```

Commits:
- `cb44e78`: "Add VL-JEPA, mHC, self-improving agents, quantum compute, and OpenAPI"
- Test fixes: "Stub quantum_compute for graceful degradation"

## References

- `ROADMAP.md` - Quantum Computing Integration section
- [Qiskit Python support](https://qiskit.org/documentation/getting_started.html)
- [PennyLane installation](https://pennylane.ai/install.html)
