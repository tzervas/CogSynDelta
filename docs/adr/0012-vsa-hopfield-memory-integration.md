# ADR-0012: Vector Symbolic Architecture and Hopfield Networks for Memory Tiers

## Status

**PROPOSED** - 2026-01-19

## Context

The research document "Latent-Space-Native AI Systems: A Technical Deep Dive" identifies
Vector Symbolic Architectures (VSA) and Modern Hopfield Networks as foundational technologies
for CogSynDelta's brain-inspired memory system. This ADR proposes their integration.

### Research Findings

**Vector Symbolic Architectures (VSA):**
- Provide algebraic structure for multi-modal binding via binding (⊗), bundling (+), permutation (ρ)
- MAP binding: `a ⊗ b` produces quasi-orthogonal vector, self-inverse (`a ⊗ a = 1`)
- HRR binding: Circular convolution, O(n log n) via FFT
- Capacity bound: `n ≥ k/(1-S²) × log(M)` where k=items, S=fidelity, M=codebook, n=dimension
- torchhd library provides GPU-accelerated PyTorch implementations (MIT license)

**Modern Hopfield Networks:**
- Transformer attention IS mathematically equivalent to Hopfield update rule
- Exponential storage capacity: 2^(n/2) patterns (vs classical 0.14n)
- One-step convergence for pattern retrieval
- hopfield-layers library available (MIT license)

### Current State

CogSynDelta has a tiered memory architecture (active → short → long-term) but lacks:
1. Algebraic binding operations for multi-modal composition
2. Explicit capacity guarantees based on VSA theory
3. Hopfield-style associative retrieval for active memory

### Proposed Memory Architecture Alignment

```
ACTIVE MEMORY (Hopfield Networks, ~100 items, minutes)
    ↓ consolidation (binding + bundling)
SHORT-TERM MEMORY (VSA bundling, ~10K items, hours)
    ↓ consolidation (compression)
LONG-TERM MEMORY (FAISS IVF-PQ, millions+, permanent)
```

## Decision

Integrate VSA operations and Modern Hopfield networks into the memory system:

### Phase 1: VSA Operations Module

```python
# src/cogsyndelta/memory/vsa_ops.py

import torch
from torchhd import random_hv, bind, bundle, unbind

class VSAMemory:
    """Vector Symbolic Architecture memory operations.

    Provides algebraic binding for multi-modal composition:
    - bind(): Create associations (image ⊗ text ⊗ context)
    - bundle(): Create superpositions (memory = Σ item_i)
    - unbind(): Recover components from bound vectors
    """

    def __init__(self, dim: int = 10000, algebra: str = "MAP"):
        """
        Args:
            dim: Vector dimension (≥10,000 for reliable operations)
            algebra: VSA algebra type ("MAP", "HRR", "BSC", "FHRR")
        """
        self.dim = dim
        self.algebra = algebra

    def create_hypervector(self, name: str) -> torch.Tensor:
        """Create named atomic hypervector."""
        return random_hv(1, self.dim, dtype=torch.float32)

    def bind_multimodal(
        self,
        image_hv: torch.Tensor,
        text_hv: torch.Tensor,
        context_hv: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Bind multi-modal representations into joint concept.

        Result is quasi-orthogonal to all inputs but recoverable via unbind.
        """
        bound = bind(image_hv, text_hv)
        if context_hv is not None:
            bound = bind(bound, context_hv)
        return bound

    def bundle_memories(self, memories: list[torch.Tensor]) -> torch.Tensor:
        """Bundle multiple memories into superposition vector.

        Capacity follows: n ≥ k/(1-S²) × log(M)
        At dim=10000, S=0.95: ~50 items reliable
        """
        return bundle(torch.stack(memories))
```

### Phase 2: Hopfield Active Memory

```python
# src/cogsyndelta/memory/hopfield_memory.py

from hflayers import Hopfield, HopfieldPooling

class HopfieldActiveMemory:
    """Active memory tier using Modern Hopfield Networks.

    Provides:
    - Exponential storage capacity: 2^(d/2) patterns
    - One-step associative retrieval
    - Energy-based pattern completion

    Why Hopfield for active memory: The transformer-attention equivalence
    means we get theoretically-grounded associative memory with GPU efficiency.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        self.hopfield = Hopfield(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_heads=num_heads,
            scaling=1.0 / (embed_dim ** 0.5)  # β = 1/√d
        )
        self.stored_patterns = []

    def store(self, pattern: torch.Tensor) -> None:
        """Store pattern in associative memory."""
        self.stored_patterns.append(pattern)

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve nearest stored pattern via energy minimization."""
        if not self.stored_patterns:
            return query
        patterns = torch.stack(self.stored_patterns)
        # Hopfield retrieval: ξ_new = X · softmax(β · Xᵀξ)
        return self.hopfield(query.unsqueeze(0), patterns).squeeze(0)
```

### Phase 3: Integration with Tiered Memory

Update `ActiveMemoryManager` to use Hopfield for hot data and VSA for binding:

```python
class ActiveMemoryManager:
    def __init__(
        self,
        embed_dim: int = 512,
        use_hopfield: bool = True,
        use_vsa: bool = True,
    ):
        if use_hopfield:
            self.hopfield_store = HopfieldActiveMemory(embed_dim)
        if use_vsa:
            self.vsa = VSAMemory(dim=10000)  # High dim for reliability
```

## Consequences

### Positive

- **Mathematical guarantees**: VSA capacity bounds provide predictable behavior
- **Multi-modal composition**: Algebraic binding creates recoverable joint representations
- **Exponential storage**: Hopfield networks scale to 2^(d/2) patterns
- **Transformer alignment**: Hopfield ≡ attention means consistency with VL-JEPA
- **MIT-licensed libraries**: torchhd and hopfield-layers are production-ready

### Negative

- **High dimension requirement**: VSA needs d≥10,000 for reliable operations
- **Additional dependencies**: torchhd, hflayers packages
- **Memory overhead**: Hopfield pattern storage vs compressed representations
- **Learning curve**: VSA algebra requires understanding for debugging

### Neutral

- Trade-off between VSA dimension (reliability) and storage cost
- Hopfield β parameter tuning affects retrieval precision vs speed

## Implementation Plan

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 1. VSA Module | Week 1-2 | vsa_ops.py, tests, benchmarks |
| 2. Hopfield Memory | Week 3-4 | hopfield_memory.py, integration |
| 3. Tier Integration | Week 5-6 | Updated ActiveMemoryManager |

## Dependencies

```toml
# pyproject.toml additions
[project.optional-dependencies]
vsa = [
    "torchhd>=0.6.0",    # MIT - VSA operations
    "hflayers>=0.2.0",    # MIT - Hopfield layers
]
```

## Testing Strategy

```python
def test_vsa_capacity_bound():
    """Verify bundling capacity follows n ≥ k/(1-S²) × log(M)."""
    vsa = VSAMemory(dim=10000)
    # At S=0.95, k=50 should be reliable
    memories = [vsa.create_hypervector(f"mem_{i}") for i in range(50)]
    bundled = vsa.bundle_memories(memories)

    # Should retrieve with >0.90 fidelity
    for mem in memories:
        similarity = F.cosine_similarity(mem, bundled, dim=-1)
        assert similarity > 0.90

def test_hopfield_exponential_capacity():
    """Verify Hopfield stores many patterns without catastrophic forgetting."""
    hopfield = HopfieldActiveMemory(embed_dim=512)
    patterns = [torch.randn(512) for _ in range(100)]
    for p in patterns:
        hopfield.store(F.normalize(p, dim=-1))

    # Should retrieve stored patterns accurately
    for p in patterns[:10]:  # Sample test
        retrieved = hopfield.retrieve(p + 0.1 * torch.randn(512))  # Noisy query
        similarity = F.cosine_similarity(p.unsqueeze(0), retrieved.unsqueeze(0))
        assert similarity > 0.85
```

## References

- Ramsauer et al. "Hopfield Networks is All You Need" (arXiv:2008.02217)
- torchhd: https://github.com/hyperdimensional-computing/torchhd
- hopfield-layers: https://github.com/ml-jku/hopfield-layers
- Kanerva "Hyperdimensional Computing" (2009)
- Research doc: docs/Latent-Space-Native AI Systems_A Technical Deep Dive for CogSynDelta.md

---

*Authored: 2026-01-19 | Related: ADR-0002, ADR-0008*
