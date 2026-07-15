# CogSynDelta VSA (Vector Symbolic Architectures)

**Version:** 0.1.0
**Status:** Production-ready implementation
**Dependencies:** torch, torchhd, hopfield-layers

## Overview

This library implements Vector Symbolic Architectures (VSA) for CogSynDelta, providing hyperdimensional computing capabilities for compositional reasoning, temporal grounding, and associative memory.

## Key Features

### 1. Fractional Power Encoding (FPE) for Temporal Grounding

Continuous temporal encoding where physical locality translates to representational similarity.

**Mathematical Foundation:**
```
T^t = F⁻¹{F{T}^t}
For FHRR: T^t = exp(iθt) where θ ~ Uniform[-π, π]
Similarity decay: sim(T^t1, T^t2) ≈ sinc(t1 - t2)
```

**Usage:**
```python
from vsa.encoders.fpe import FractionalPowerEncoder
from vsa.config import VSAConfig

config = VSAConfig(dimension=10000, model="FHRR", device="cuda")
encoder = FractionalPowerEncoder(config)

base = encoder.random_vector()
t0 = encoder.encode(base, 0.0)  # Time 0
t1 = encoder.encode(base, 1.0)  # Time 1
sim = encoder.cosine_similarity(t0, t1)  # ≈ sinc(1) ≈ 0
```

### 2. VSA Operations

**Binding (⊗):** Associative composition via circular convolution
```python
from vsa.operations.binding import bind, unbind

z = bind(x, y)  # Bind x and y
x_recovered = unbind(z, y)  # Extract x from z
```

**Bundling (+):** Superposition of hypervectors
```python
from vsa.operations.bundling import bundle

superposition = bundle(torch.stack([x, y, z]))  # x + y + z
```

**Permutation (ρ):** Ordering and role distinction
```python
from vsa.operations.permutation import permute

ordered_pair = bind(permute(x, 0), permute(y, 1))  # (x, y) ≠ (y, x)
```

### 3. Modern Hopfield Networks

Exponential storage capacity: ~2^(d/2) patterns for d-dimensional vectors.

**Mathematical Foundation:**
```
Energy: E = -log(Σᵢ exp(βξᵢᵀx))
Update: ξ_new = X·softmax(βXᵀξ)
```

**Usage:**
```python
from vsa.memory.hopfield import ModernHopfieldMemory

memory = ModernHopfieldMemory(dimension=10000)
memory.store(pattern_a)
memory.store(pattern_b)

noisy = pattern_a + 0.2 * torch.randn_like(pattern_a)
retrieved = memory.retrieve(noisy)  # Cleans up noise
```

### 4. Resonator Networks

Factorize compositional hypervectors in polynomial time.

**Usage:**
```python
from vsa.memory.resonator import ResonatorNetwork

z = bind(x, y)
resonator = ResonatorNetwork(dimension=10000)
y_estimated, iters = resonator.factorize(z, x)  # Recover y
```

## Performance Characteristics

| Feature | Dimension | Capacity | GPU Throughput |
|---------|-----------|----------|----------------|
| FPE Encoding | 10,000 | N/A | >100K vecs/sec |
| Hopfield Storage | 10,000 | ~2^5000 | <1ms retrieval |
| Resonator Factorization | 10,000 | Polynomial | 100 iters typical |

## Installation

```bash
cd libs/vsa
pip install -e .
```

## Testing

```bash
pytest tests/ -v
```

## References

1. Plate, T. A. (2003). *Holographic reduced representation.*
2. Heddes et al. (2023). *Torchhd: An Open Source Python Library to Support Research on HDC and VSA.* JMLR.
3. Ramsauer et al. (2021). *Hopfield Networks is All You Need.* ICLR.
4. Kent et al. (2020). *Resonator networks for factorizing distributed representations.* Neural Computation.

## Citation

```bibtex
@software{cogsyndelta_vsa,
  title={CogSynDelta VSA: Vector Symbolic Architectures for Brain-Inspired AI},
  author={tzervas},
  year={2026},
  url={https://github.com/tzervas/CogSynDelta}
}
```
