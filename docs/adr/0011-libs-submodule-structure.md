# ADR-0011: Library Submodule Structure for Extractable Components

**Status**: Proposed  
**Date**: 2026-01-19  
**Authors**: @tzervas  
**Supersedes**: None  
**Related**: ADR-0009 (Algebraic Training)

---

## Context

CogSynDelta contains several components that are:
1. **Self-contained** - No or minimal internal dependencies
2. **Reusable** - Valuable as standalone libraries
3. **Complex** - Deserve dedicated documentation, testing, and versioning

Current candidates for extraction:
- `algebraic_training` - Predict training outcomes without backprop
- `mhc` - Moderated HyperConnections (gated cross-model connections)
- `interconnect` - Cross-model weight transfer mechanisms
- Future: Memory compactors, quantum stubs, etc.

### Problem

The current flat structure in `src/cogsyndelta/` makes it difficult to:
1. Extract components into standalone libraries later
2. Maintain clear dependency boundaries
3. Version and release components independently
4. Allow external projects to use components without full CogSynDelta

### Goals

1. **Clean boundaries**: Each extractable component is self-contained
2. **Minimal friction**: Easy to convert to actual submodules/repos later
3. **Backward compatibility**: Existing imports continue to work
4. **Documentation parity**: Each component is fully documented

---

## Decision

Adopt a **libs/ directory structure** for extractable components:

```
CogSynDelta/
├── libs/                              # Extractable libraries
│   ├── algebraic-training/            # Algebraic training library
│   │   ├── pyproject.toml             # Standalone package definition
│   │   ├── README.md                  # Library documentation
│   │   ├── src/
│   │   │   └── algebraic_training/
│   │   │       ├── __init__.py
│   │   │       ├── ntk.py             # Neural Tangent Kernel
│   │   │       ├── fisher.py          # Fisher Information
│   │   │       ├── spectral.py        # Spectral methods
│   │   │       ├── optimizer.py       # Unified optimizer
│   │   │       ├── trainer.py         # High-level trainer
│   │   │       ├── functional.py      # Functional API
│   │   │       └── py.typed
│   │   └── tests/
│   │       └── test_*.py
│   │
│   └── README.md                      # Libs overview
│
├── src/cogsyndelta/
│   ├── optimization/
│   │   ├── __init__.py                # Re-exports from libs + integration
│   │   ├── _integration.py            # CogSynDelta-specific wrappers
│   │   └── cuda_optimization.py       # GPU-specific (stays here)
│   └── ...
```

### Integration Layer

CogSynDelta-specific components that depend on the core libraries:
- `MHCAlgebraicOptimizer` → Uses `algebraic_training` + mHC concepts
- `InterconnectAlgebraicOptimizer` → Uses `algebraic_training` + interconnect
- `PathwayStrengthPredictor` → Routing-specific, may stay in main

These will be in `src/cogsyndelta/optimization/_integration.py` and import from libs.

### Backward Compatibility

Existing imports continue to work:
```python
# These all work:
from cogsyndelta.optimization import UnifiedAlgebraicTrainer
from cogsyndelta.optimization import NTKPredictor
from algebraic_training import UnifiedAlgebraicTrainer  # Direct lib access
```

---

## Consequences

### Positive
- Clear extraction path to standalone repos
- Components can be tested/developed independently
- External projects can vendor just the libs they need
- Forces clean dependency hygiene

### Negative
- More complex directory structure
- Potential import path confusion during transition
- Need to maintain pyproject.toml in multiple places

### Neutral
- Git history preserved via `git mv`
- CI needs updates for multi-package testing

---

## Implementation Plan

### Phase 1: Directory Structure (This PR)
1. Create `libs/` directory with README
2. Create `libs/algebraic-training/` package structure
3. Move and split `algebraic_training.py` using `git mv`
4. Create integration layer in `src/cogsyndelta/optimization/`
5. Update `__init__.py` exports for backward compatibility
6. Add functional API

### Phase 2: Testing & Validation
1. Add accuracy parity tests (algebraic vs SGD)
2. Ensure all 20 existing tests pass
3. Add library-specific tests in `libs/algebraic-training/tests/`

### Phase 3: Documentation
1. Library README with examples
2. Update API_REFERENCE.md
3. Create ADR for each future extracted component

### Future Phases
- Extract mHC components to `libs/mhc/`
- Extract interconnect to `libs/interconnect/`
- Convert to actual git submodules when ready for external release

---

## Alternatives Considered

### 1. Git Submodules Now
**Rejected**: Premature. Need to stabilize APIs before external repos.

### 2. Monorepo with Workspaces
**Rejected**: Python tooling (uv, pip) doesn't handle workspaces as well as npm/cargo.

### 3. Keep Everything Flat
**Rejected**: Makes extraction painful later, unclear boundaries.

---

## References

- [ADR-0009: Algebraic Training Optimization](0009-algebraic-training-optimization.md)
- [Python Packaging Guide](https://packaging.python.org/)
- [src layout discussion](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
