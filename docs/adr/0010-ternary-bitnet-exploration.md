# ADR-0010: Ternary Neural Network Exploration (BitNet)

**Status**: Proposed
**Date**: 2026-01-19
**Decision Makers**: @tzervas
**Technical Story**: Exploring 1.58-bit ternary representations for extreme efficiency
**Related**: ADR-0009 (Algebraic Training Optimization)

## Context

Modern neural networks use 32-bit (FP32) or 16-bit (FP16/BF16) floating-point weights, requiring substantial memory bandwidth and compute. Recent research demonstrates that **ternary weights {-1, 0, +1}** (1.58 bits per weight) can match full-precision performance for models above 3B parameters while achieving dramatic efficiency gains.

This ADR documents an **isolated exploration** of ternary/BitNet approaches, separate from the primary algebraic training work (ADR-0009) which focuses on standard binary floating-point representations.

### Why Isolate This Work?

1. **Different optimization target**: Ternary requires specialized kernels and hardware considerations
2. **Experimental nature**: Less mature than standard quantization (4-8 bit)
3. **Hardware dependency**: Benefits vary significantly by platform (ARM vs x86, GPU vs CPU)
4. **Risk isolation**: Keeps main algebraic training branch stable

## Decision

We will explore ternary neural network implementations in an **isolated feature branch** (`feat/ternary-implementations`) to evaluate:

1. BitNet b1.58 architecture adaptation for CogSynDelta submodels
2. Packed trit storage formats for memory efficiency
3. Hardware-specific optimizations (RTX 5080 GDDR7, AMD64)
4. Integration pathways with algebraic training methods

## Theoretical Foundation

### BitNet b1.58 Architecture

BitNet constrains weights to ternary values during training:

$$W_{ternary} \in \{-1, 0, +1\}^{m \times n}$$

**Quantization function**:
$$\text{RoundClip}(x, a, b) = \max(a, \min(b, \text{round}(x)))$$

**Weight binarization** (for {-1, +1} subset):
$$\tilde{W} = \text{Sign}(W - \mathbb{E}[W])$$

**Absmean quantization** (for full ternary):
$$\tilde{W} = \text{RoundClip}\left(\frac{W}{\gamma + \epsilon}, -1, 1\right), \quad \gamma = \frac{1}{nm}\sum_{ij}|W_{ij}|$$

### Efficiency Analysis

| Metric | FP16 Baseline | BitNet b1.58 | Improvement |
|--------|---------------|--------------|-------------|
| Memory per weight | 16 bits | 1.58 bits | 10.1× |
| Multiply-accumulate | FP16 MAC | INT8 add | 2-6× faster |
| Energy per operation | ~10 pJ | ~0.24 pJ | 41.3× |
| Model size (7B) | 14 GB | 1.4 GB | 10× |

### Packed Trit Representation

Standard approach: Store each trit as 2 bits (wasteful: 4 states for 3 values)

**Optimized packing**: 5 trits in 8 bits (3⁵ = 243 ≤ 256)

```
┌─────────────────────────────────────┐
│  8-bit byte encodes 5 ternary values │
│  Lookup table: byte → (t₁,t₂,t₃,t₄,t₅) │
│  Compression: 1.6 bits/weight        │
└─────────────────────────────────────┘
```

**Alternative**: Separate positive/negative bit planes
- Positive plane: 1 where W = +1
- Negative plane: 1 where W = -1
- Zero implicit where both are 0

## Research Questions

1. **Accuracy parity**: At what model size does ternary match FP16 for CogSynDelta tasks?
2. **Training dynamics**: How does ternary constraint affect algebraic training predictions?
3. **mHC compatibility**: Can ternary weights work with Moderated HyperConnections?
4. **Hardware mapping**: Optimal kernel implementations for available hardware

## Proposed Exploration

### Phase 1: Baseline Evaluation
- [ ] Implement ternary weight constraint layer
- [ ] Benchmark against FP16 on MNIST/CIFAR
- [ ] Measure actual speedup on RTX 5080

### Phase 2: Integration Study
- [ ] Test ternary with NTK predictor
- [ ] Evaluate spectral properties of ternary matrices
- [ ] Assess mHC gate compatibility

### Phase 3: Optimization
- [ ] Implement packed trit storage
- [ ] Develop CUDA kernels for ternary matmul
- [ ] Profile memory bandwidth vs compute

## Success Criteria

| Metric | Target |
|--------|--------|
| Accuracy vs FP16 | ≥98% relative |
| Memory reduction | ≥8× |
| Inference speedup | ≥2× |
| Training compatibility | Works with algebraic init |

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Accuracy degradation | Medium | High | Fallback to 4-bit SpinQuant |
| Hardware incompatibility | Medium | Medium | Software emulation layer |
| Training instability | Low | Medium | Use algebraic init, minimal fine-tune |

## References

1. Ma, S., et al. (2024). "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" arXiv:2402.17764
2. Wang, H., et al. (2025). "BitNet b1.58 2B4T Technical Report" arXiv:2504.12285
3. Microsoft BitNet Repository: https://github.com/microsoft/BitNet

## Relationship to ADR-0009

This exploration is **complementary but isolated** from ADR-0009:

| Aspect | ADR-0009 | ADR-0010 |
|--------|----------|----------|
| Weight representation | Standard FP16/FP32 | Ternary {-1,0,+1} |
| Focus | Algebraic training methods | Extreme quantization |
| Maturity | Production-ready | Experimental |
| Branch | `feat/algebraic-training` | `feat/ternary-implementations` |

If ternary exploration proves successful, integration pathways will be documented in a future ADR.

---

*This ADR will be updated as exploration progresses. Last updated: 2026-01-19*
