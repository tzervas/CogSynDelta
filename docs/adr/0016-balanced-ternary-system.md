# ADR-0016: Balanced Ternary Weight Encoding

**Status**: Experimental
**Date**: 2026-01-20
**Decision Makers**: @tzervas
**Technical Story**: Exploring brain-like ternary representations as alternative to binary quantization

## Context

CogSynDelta v0.2.0 uses BitNet b1.58 for ternary weight quantization {-1, 0, +1}, achieving 10× memory compression over FP16. However, BitNet's ternary representation is **asymmetric**:

- Derived from binary quantization via sign-magnitude encoding
- Zero is "absence of signal" rather than a distinct semantic state
- Training relies on straight-through estimators that don't fully leverage ternary arithmetic

Biological neurons exhibit **three fundamental states**:
1. **Inhibit**: Hyperpolarization, active suppression (-1)
2. **Neutral**: Resting potential, no influence (0)
3. **Excite**: Depolarization, activation (+1)

This suggests a more **symmetric ternary representation** might be more brain-like and potentially more expressive.

### Balanced Ternary Mathematics

Balanced ternary is a **radix-3 numeral system** using digits {-1, 0, +1}:

```
Decimal    Balanced Ternary    Binary
   0            0                0
   1            1                1
   2            1̄1              10      (1̄ = -1)
   3            10              11
   4            11             100
   5            11̄            101
  -1            1̄               -1
  -2            11̄              -10
```

**Key properties**:
- **Symmetric**: Negation is trivial (flip all trits)
- **No carry propagation**: Addition is simpler than binary
- **Unique representation**: Every integer has exactly one representation
- **Algebraic closure**: Addition/subtraction stay in ternary domain

### Trytes for Neural Networks

A **tryte** (ternary byte) uses 9 balanced ternary digits (trits):

```
Range:  -9841 to +9841  (3^9 = 19683 values)
Bits:   ~14.2 bits equivalent (log2(19683))
Compression: Similar to BitNet (~1.58 bits/weight after packing)
```

**Why 9 trits**:
- Fits neatly into common weight precisions (FP16 range ±65,504)
- Efficient packing: 5 trits per byte (3^5 = 243 < 256)
- Powers of 3 align with ternary arithmetic

## Decision

We will implement a **balanced ternary weight encoding system** as an **experimental alternative** to BitNet b1.58, with the following design:

1. **Balanced ternary arithmetic** as foundation:
   - Core operations: addition, multiplication, negation, comparison
   - Tryte encoding: 9 trits per weight value
   - Efficient packing: 5 trits per byte storage

2. **Neural network layers** with ternary weights:
   - Drop-in replacements: `BalancedTernaryLinear`, `BalancedTernaryConv2d`
   - Straight-through estimator (STE) for gradients (same as BitNet)
   - Weight initialization: uniform over {-1, 0, +1}

3. **Quantization and compression**:
   - FP16/FP32 → balanced ternary conversion
   - Packed storage: 5 trits/byte (vs FP16's 2 bytes/weight)
   - Target: 10× compression (same as BitNet b1.58)

4. **Separate feature branch** (`claude/balanced-ternary-5wbyn`):
   - Does NOT replace BitNet (runs in parallel for comparison)
   - Experimental status: Research validation needed
   - Integration into main pipeline if benchmarks show benefits

Architecture:

```
┌─────────────────────────────────────────────────────────┐
│              Neural Network Layer                       │
│  ┌──────────────┐         ┌──────────────┐            │
│  │ FP16 Weights │         │ FP16 Weights │            │
│  │  (training)  │         │  (inference) │            │
│  └──────┬───────┘         └──────┬───────┘            │
│         │                        │                     │
│         │ STE gradient           │ quantize            │
│         │                        │                     │
│  ┌──────▼───────────────────────▼───────┐             │
│  │  Balanced Ternary Quantizer          │             │
│  │  - Round to {-1, 0, +1}              │             │
│  │  - Encode as trytes (9 trits)        │             │
│  └──────┬───────────────────────────────┘             │
│         │                                              │
│         ▼                                              │
│  ┌─────────────────┐                                  │
│  │ Ternary Weights │  {-1, 0, +1}^N                   │
│  │  (computation)  │  Packed: 5 trits/byte            │
│  └─────────────────┘                                  │
└─────────────────────────────────────────────────────────┘
```

## Rationale

### Why Balanced Ternary

1. **Symmetry**: Excitation and inhibition are equally represented, unlike binary where 0/1 are asymmetric
2. **Brain-like**: Three neural states (inhibit/neutral/excite) map naturally to {-1, 0, +1}
3. **Algebraic elegance**: No carry propagation, simpler arithmetic operations
4. **Negation is free**: Flip all trits, vs binary's two's complement

### Why Trytes (9 trits)

1. **Precision**: 19,683 distinct values covers typical neural network weight distributions
2. **Packing efficiency**: 5 trits/byte (3^5 = 243 < 256) minimizes waste
3. **Hardware compatibility**: Byte-aligned storage, no bit-shifting needed for access

### Comparison with BitNet b1.58

| Property | BitNet b1.58 | Balanced Ternary | Notes |
|----------|--------------|------------------|-------|
| **Values** | {-1, 0, +1} | {-1, 0, +1} | Same ternary space |
| **Compression** | ~1.58 bits/weight | ~1.58 bits/weight | Equivalent after packing |
| **Arithmetic** | Binary-derived | Ternary-native | BT potentially more efficient |
| **Symmetry** | Asymmetric zero | Symmetric zero | BT more balanced |
| **Hardware** | Needs custom kernels | Needs custom kernels | Both face same challenge |
| **Training** | STE gradients | STE gradients | Same gradient strategy |

**Hypothesis**: Balanced ternary may provide:
- Better gradient flow due to symmetry
- More expressive zero states (neutral vs absence)
- Simpler hardware implementation (no carry logic)

**Risk**: May perform identically to BitNet in practice, since both use same {-1, 0, +1} space.

### Why Experimental Branch

1. **No proven accuracy benefits**: Hypothesis needs empirical validation
2. **Hardware support**: Requires custom CUDA kernels for efficiency
3. **BitNet is established**: Already proven in literature (Microsoft Research)
4. **Parallel exploration**: Compare both approaches before committing

## Alternatives Considered

### Option 1: Continue with BitNet b1.58 only

- **Pros**: Established, proven results, no additional complexity
- **Cons**: Misses potential symmetry benefits, no exploration of ternary-native arithmetic
- **Why Rejected**: Low-cost exploration (separate branch) with potential for insights

### Option 2: Quaternary (base-4) encoding

- **Pros**: Four states {-2, -1, 0, +1} or {-1, 0, +1, +2} provide more precision
- **Cons**: Less brain-like (neurons aren't quaternary), worse compression (2 bits/weight)
- **Why Rejected**: Loses bio-inspired motivation, worse compression

### Option 3: Mixed precision (FP16 important weights, ternary others)

- **Pros**: Best of both worlds (precision where needed, compression elsewhere)
- **Cons**: Complexity, requires importance scoring, irregular memory access
- **Why Rejected**: Out of scope for this exploration (could be future work)

### Option 4: Posit arithmetic

- **Pros**: Variable precision, better dynamic range than IEEE 754
- **Cons**: Not ternary (loses bio-inspiration), hardware support even worse
- **Why Rejected**: Different research direction, not ternary-focused

## Consequences

### Positive

- **Novel exploration**: Balanced ternary is underexplored in modern deep learning
- **Bio-inspiration**: Closer to neural computation than binary
- **Modular design**: Drop-in layer replacements, easy to compare with BitNet
- **Same compression**: 10× memory reduction (matches BitNet b1.58)

### Negative

- **Unproven accuracy**: No guarantees of FP16 parity or BitNet equivalence
  - **Mitigation**: Experimental branch, benchmark before integration
- **Hardware efficiency**: PyTorch doesn't natively support ternary ops, custom kernels needed
  - **Mitigation**: Start with pure Python (slow but correct), optimize later
- **Maintenance burden**: Two quantization systems (BitNet + BT) to maintain
  - **Mitigation**: Keep BT in separate branch until proven, then choose one

### Neutral

- **Research contribution**: Could publish findings (positive or negative)
- **Community interest**: Balanced ternary has niche following, may attract collaborators
- **Educational value**: Team learns ternary arithmetic, numeral systems

## Implementation

### Key Files

**Balanced Ternary Branch** (`claude/balanced-ternary-5wbyn`):

- `libs/compression/src/compression/balanced_ternary/arithmetic.py` - Core ternary arithmetic and BalancedTernaryTensor
- `libs/compression/src/compression/balanced_ternary/layers.py` - Neural network layers (Linear, Conv2d, Embedding)
- `libs/compression/src/compression/balanced_ternary/quantizer.py` - Quantization, STE, and compression
- `libs/compression/tests/test_balanced_ternary_*.py` - Test suites

Total: **~1,000 lines** of balanced ternary implementation

### API Example

```python
from compression.balanced_ternary import BalancedTernaryLinear, BalancedTernaryQuantizer

# Replace standard PyTorch layer
# layer = nn.Linear(512, 256)
layer = BalancedTernaryLinear(512, 256)  # trits_per_tryte defaults to 9

# Forward pass (weights auto-quantized during training via STE)
output = layer(input)  # Ternary {-1, 0, +1} computation

# For deployment: permanently quantize weights
layer.eval()
layer.quantize_weights_explicit()

# Compress trained model
from compression.balanced_ternary import BalancedTernaryCompressor
compressor = BalancedTernaryCompressor(trits_per_tryte=9)
compressed, metadata = compressor.compress(layer.weight)

# Compressed size
original_size = layer.weight.element_size() * layer.weight.numel()  # FP16
compressed_size = metadata["original_shape"][0] * metadata["original_shape"][1]  # packed bytes
compression_ratio = original_size / compressed_size  # ~10×
```

### Tensor Encoding Example

```python
from compression.balanced_ternary import BalancedTernaryTensor, tryte_encode
import torch

# Encode values as balanced ternary tensor
values = torch.tensor([5, -3, 0, 13])
bt_tensor = tryte_encode(values, trits_per_tryte=9)

# Access ternary representation
print(bt_tensor.trits.shape)  # Shape: [4, 9] (values, trits per value)
# Decimal 5 in balanced ternary: 1*3² - 1*3¹ - 1*3⁰ = 9 - 3 - 1 = 5

# Decode back to decimal
decoded = bt_tensor.to_decimal()  # tensor([5, -3, 0, 13])
```

### Testing Strategy

1. **Arithmetic correctness**: Verify ternary addition, multiplication, conversion
2. **Layer equivalence**: Compare with PyTorch layers at FP16
3. **Gradient flow**: Ensure STE gradients propagate correctly
4. **Compression ratio**: Validate 10× compression target
5. **Accuracy benchmarks**: Train on MNIST/CIFAR-10, compare FP16 vs BT vs BitNet

### Success Criteria

**Minimum Viable (for branch acceptance)**:
- ✅ Correct ternary arithmetic (tested)
- ✅ Layer API compatible with PyTorch (drop-in replacement)
- ✅ 10× compression ratio (matches BitNet)
- ⏳ Gradient flow validation (STE works)

**Research Goals (for main integration)**:
- ⏳ FP16 parity: ≥95% of FP16 accuracy on benchmark tasks
- ⏳ BitNet comparison: ≥ BitNet b1.58 accuracy (or document why not)
- ⏳ Training stability: Converges reliably, no NaN/gradient explosions
- ⏳ Efficiency analysis: Profile speed vs BitNet, FP16

(⏳ = pending empirical validation)

## Research Questions

This experimental branch aims to answer:

1. **Does symmetry matter?** Does balanced ternary's symmetric zero improve gradient flow vs BitNet?
2. **Is ternary-native arithmetic faster?** Can we build more efficient kernels using ternary math?
3. **What is the accuracy ceiling?** Can ternary networks match FP16 on complex tasks?
4. **Where does ternary excel?** Are there domains (sparse, discrete) where ternary outperforms continuous?

## References

- **Historical**: Balanced ternary used in Soviet Setun computer (1958)
- **BitNet b1.58**: "BitNet b1.58: Training 1-bit LLMs from Scratch" (Microsoft Research, 2024)
- **Ternary networks**: "Ternary Weight Networks" (Aleeksiev et al., 2016)
- **Neural arithmetic**: "Neural Arithmetic Logic Units" (Trask et al., 2018)
- **Related ADRs**:
  - ADR-0010: Ternary BitNet Exploration (binary-derived ternary)
  - ADR-0014: Matryoshka and QINCo2 Compression (complementary)

## Rollout Plan

### Phase 1: Experimental Validation (Current)

- [x] Implement balanced ternary arithmetic
- [x] Implement neural network layers
- [x] Implement quantization and compression
- [x] Write unit tests for correctness
- [ ] Train MNIST model (baseline benchmark)
- [ ] Train CIFAR-10 model (harder benchmark)
- [ ] Compare: FP16 vs BitNet vs Balanced Ternary

### Phase 2: Research Publication

- [ ] Analyze results, document findings
- [ ] Publish technical report (positive or negative results)
- [ ] Share with research community (arXiv, blog post)

### Phase 3: Integration Decision

- [ ] If BT ≥ BitNet: Integrate into main compression pipeline
- [ ] If BT < BitNet: Keep as research artifact, document why
- [ ] If BT == BitNet: Choose simpler implementation (likely BitNet)

### Phase 4: Hardware Optimization (if integrated)

- [ ] Write CUDA kernels for ternary operations
- [ ] Benchmark GPU efficiency vs PyTorch overhead
- [ ] Optimize packing/unpacking for inference

## Notes

- **Not a replacement for BitNet**: Experimental exploration, runs in parallel
- **Brain-inspired**: Matches neural inhibit/neutral/excite states
- **Symmetric by design**: Negation is trivial, zero is a distinct semantic state
- **Compression equivalent**: Same ~10× reduction as BitNet b1.58
- **Needs validation**: Empirical benchmarks required before claiming benefits

## Open Questions

1. **Hardware support**: Can we build efficient ternary ALUs? Or is simulation overhead too high?
2. **Hybrid approaches**: Could we mix BitNet (conv layers) and BT (linear layers)?
3. **Sparsity interaction**: Does balanced zero interact better with sparsity patterns?
4. **Multi-ternary**: Could we use multiple ternary bases (3, 9, 27) for variable precision?

## Conclusion

Balanced ternary is a **high-risk, high-reward exploration**:

- **High risk**: May perform identically to BitNet (same value space)
- **High reward**: If symmetry matters, could improve gradient flow and efficiency
- **Low cost**: Separate branch, modular implementation, clear success criteria

We proceed with **experimental status**, benchmark rigorously, and integrate only if evidence supports benefits over BitNet b1.58.
