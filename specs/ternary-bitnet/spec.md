# Ternary BitNet Implementation Specification

**Spec ID**: ternary-bitnet
**Status**: Draft
**Created**: 2026-01-19
**Parent ADR**: ADR-0010

---

## 1. Overview

### 1.1 Purpose

This specification defines the implementation approach for exploring ternary (1.58-bit) neural network weights within the CogSynDelta architecture, based on BitNet b1.58 research.

### 1.2 Scope

- Ternary weight layer implementations
- Packed trit storage formats
- Hardware-optimized kernels
- Integration with existing CogSynDelta components

### 1.3 Out of Scope

- Standard floating-point quantization (see ADR-0009)
- Production deployment (experimental only)
- Non-ternary low-bit formats (2-bit, 4-bit)

---

## 2. Technical Requirements

### 2.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Implement ternary weight constraint layer | Must |
| FR-2 | Support forward pass with ternary weights | Must |
| FR-3 | Support backward pass with STE | Must |
| FR-4 | Provide weight conversion FP → Ternary | Must |
| FR-5 | Implement packed trit storage | Should |
| FR-6 | CUDA kernel for ternary matmul | Should |
| FR-7 | CPU fallback implementation | Must |

### 2.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | Memory reduction vs FP16 | ≥8× |
| NFR-2 | Inference latency | ≤1.5× FP16 (software), ≤0.5× (optimized) |
| NFR-3 | Accuracy retention | ≥98% relative to FP16 |
| NFR-4 | Python version | 3.14+ |
| NFR-5 | PyTorch compatibility | 2.9+ |

---

## 3. Architecture

### 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    TernaryModule                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ TernaryLinear   │  │ TernaryConv2d   │                  │
│  │                 │  │                 │                  │
│  │ - weight_scale  │  │ - weight_scale  │                  │
│  │ - ternary_weight│  │ - ternary_weight│                  │
│  └────────┬────────┘  └────────┬────────┘                  │
│           │                    │                            │
│           ▼                    ▼                            │
│  ┌─────────────────────────────────────────┐               │
│  │         TernaryQuantizer                 │               │
│  │                                          │               │
│  │  quantize(W) → {-1, 0, +1}              │               │
│  │  scale = absmean(W)                      │               │
│  │  STE for backward pass                   │               │
│  └─────────────────────────────────────────┘               │
│                       │                                     │
│                       ▼                                     │
│  ┌─────────────────────────────────────────┐               │
│  │         PackedTritStorage                │               │
│  │                                          │               │
│  │  5 trits → 1 byte (243/256 efficiency)  │               │
│  │  Lookup tables for encode/decode         │               │
│  └─────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Class Hierarchy

```python
# Proposed module structure
src/cogsyndelta/optimization/ternary/
├── __init__.py
├── quantizer.py          # TernaryQuantizer, STE implementation
├── layers.py             # TernaryLinear, TernaryConv2d
├── storage.py            # PackedTritStorage, bit manipulation
├── kernels/
│   ├── __init__.py
│   ├── cpu_fallback.py   # Pure Python/NumPy implementation
│   └── cuda_ternary.py   # CUDA kernels (if available)
└── conversion.py         # FP16 → Ternary model conversion
```

---

## 4. Detailed Design

### 4.1 Ternary Quantization

**Algorithm**: Absmean quantization with round-clip

```python
def ternary_quantize(W: Tensor) -> tuple[Tensor, Tensor]:
    """
    Quantize floating-point weights to ternary {-1, 0, +1}.

    Args:
        W: Float weight tensor of any shape.

    Returns:
        W_ternary: Ternary weight tensor (same shape, int8 storage).
        scale: Per-tensor or per-channel scale factor.

    Mathematical formulation:
        γ = (1/nm) Σ|W_ij|  (absmean)
        W̃ = RoundClip(W/γ, -1, +1)
    """
    scale = W.abs().mean()
    W_normalized = W / (scale + 1e-8)
    W_ternary = W_normalized.round().clamp(-1, 1).to(torch.int8)
    return W_ternary, scale
```

### 4.2 Straight-Through Estimator (STE)

For backpropagation through discrete quantization:

```python
class TernaryQuantizeFunction(torch.autograd.Function):
    """STE: Forward uses ternary, backward passes gradient through."""

    @staticmethod
    def forward(ctx, W: Tensor) -> Tensor:
        scale = W.abs().mean()
        W_ternary = (W / (scale + 1e-8)).round().clamp(-1, 1)
        ctx.save_for_backward(W)
        return W_ternary * scale

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        W, = ctx.saved_tensors
        # STE: Pass gradient through unchanged
        # Optional: Clip gradient for weights outside [-1, 1]
        grad_input = grad_output.clone()
        grad_input[W.abs() > 1] = 0  # Optional clipping
        return grad_input
```

### 4.3 Packed Trit Storage

**Encoding scheme**: 5 trits per byte

```python
# Trit values: -1 → 0, 0 → 1, +1 → 2
# 5 trits = 3^5 = 243 states (fits in 8 bits)

def encode_trits(trits: np.ndarray) -> np.ndarray:
    """Encode ternary values to packed bytes."""
    # Shift from {-1,0,+1} to {0,1,2}
    shifted = trits + 1
    # Pack 5 trits per byte: t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4
    packed = (shifted[::5] +
              3 * shifted[1::5] +
              9 * shifted[2::5] +
              27 * shifted[3::5] +
              81 * shifted[4::5])
    return packed.astype(np.uint8)

def decode_trits(packed: np.ndarray) -> np.ndarray:
    """Decode packed bytes to ternary values."""
    # Use lookup table for efficiency
    trits = DECODE_LUT[packed]  # Shape: (n_bytes, 5)
    return trits.flatten() - 1  # Shift back to {-1,0,+1}

# Precomputed lookup table
DECODE_LUT = np.array([
    [(i // (3**j)) % 3 for j in range(5)]
    for i in range(243)
], dtype=np.int8)
```

### 4.4 Ternary Matrix Multiplication

**CPU fallback** (using packed representation):

```python
def ternary_matmul(X: Tensor, W_packed: Tensor, scale: Tensor) -> Tensor:
    """
    Compute X @ W where W is ternary-packed.

    Optimization: Separate positive and negative planes.
    Y = scale * (X @ W_pos - X @ W_neg)

    Where W_pos[i,j] = 1 if W[i,j] = +1, else 0
          W_neg[i,j] = 1 if W[i,j] = -1, else 0
    """
    W_ternary = unpack_trits(W_packed)  # Decode to {-1,0,+1}
    W_pos = (W_ternary == 1).float()
    W_neg = (W_ternary == -1).float()

    # Two sparse matmuls (often faster than one dense for sparse W)
    Y = scale * (X @ W_pos - X @ W_neg)
    return Y
```

---

## 5. Integration Points

### 5.1 With Algebraic Training (ADR-0009)

| Integration | Approach |
|-------------|----------|
| Weight initialization | Use algebraic prediction, then quantize to ternary |
| Fine-tuning | Algebraic delta computation in FP, apply as ternary |
| mHC gates | Keep gates in FP16 (small overhead), only weights ternary |

### 5.2 With CogSynDelta Core

```python
# Example: Convert a submodel to ternary
from cogsyndelta.optimization.ternary import convert_to_ternary

# Original model
pcn_model = PrefrontalCortexNetwork(...)

# Convert (inference only initially)
pcn_ternary = convert_to_ternary(
    pcn_model,
    exclude_layers=["gate", "norm"],  # Keep some layers in FP
    pack_weights=True,
)
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

| Test | Description |
|------|-------------|
| `test_ternary_quantize` | Verify quantization produces {-1,0,+1} |
| `test_ste_gradient` | Verify gradient flows through quantization |
| `test_pack_unpack` | Verify packed trit roundtrip |
| `test_ternary_matmul` | Verify matmul matches FP reference |

### 6.2 Integration Tests

| Test | Description |
|------|-------------|
| `test_mnist_accuracy` | Ternary model achieves ≥98% on MNIST |
| `test_memory_reduction` | Verify ≥8× memory reduction |
| `test_mhc_compatibility` | Ternary weights work with mHC |

### 6.3 Benchmarks

| Benchmark | Metric |
|-----------|--------|
| `bench_inference_latency` | Time per forward pass |
| `bench_memory_footprint` | Peak GPU memory |
| `bench_throughput` | Samples per second |

---

## 7. Milestones

| Milestone | Deliverable | Target |
|-----------|-------------|--------|
| M1 | Basic ternary layer + tests | Week 1 |
| M2 | Packed storage + conversion | Week 2 |
| M3 | MNIST benchmark results | Week 3 |
| M4 | CUDA kernel (optional) | Week 4 |
| M5 | Integration assessment | Week 5 |

---

## 8. Open Questions

1. **Per-channel vs per-tensor scaling**: Which gives better accuracy?
2. **Activation quantization**: Should activations also be quantized?
3. **Mixed precision**: Which layers benefit most from ternary?
4. **Training from scratch**: Is ternary-aware training feasible?

---

## 9. References

- BitNet paper: arXiv:2402.17764
- BitNet 2B4T: arXiv:2504.12285
- Microsoft BitNet repo: github.com/microsoft/BitNet
- Ternary weight networks: arXiv:1605.04711

---

*Specification version 0.1 - Draft*
