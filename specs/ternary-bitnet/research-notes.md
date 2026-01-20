# Ternary BitNet Research Notes

**Context**: Extracted from ADR-0009 research responses  
**Date**: 2026-01-19  
**Purpose**: Preserve ternary-specific research for isolated exploration

---

## 1. Key Findings from Research

### 1.1 BitNet b1.58 Performance Claims

**Source**: arXiv:2402.17764, arXiv:2504.12285

| Claim | Evidence | Confidence |
|-------|----------|------------|
| Matches FP16 at 3B+ params | Benchmarks on language tasks | High |
| 2.71× inference speedup | Specialized kernels required | Medium |
| 41.3× energy reduction | Hardware-dependent | Medium |
| 10× memory reduction | Mathematical (1.58 vs 16 bits) | High |

**Caveats**:
- Software emulation adds 10-20% overhead without specialized hardware
- Benefits most pronounced for memory-bound workloads
- Smaller models (<1B params) show accuracy degradation

### 1.2 Hardware Considerations

**RTX 5080 (GDDR7) Context**:
- High memory bandwidth may reduce ternary benefits for compute-bound ops
- INT8 tensor cores available but not ternary-optimized
- Packed trit operations require software decode

**Reported Gains by Platform**:
| Platform | Speedup Range | Notes |
|----------|---------------|-------|
| ARM (mobile) | 1.37-6× | Best case for energy/memory |
| x86 (CPU) | 1.5-3× | AVX2 packed ops help |
| NVIDIA GPU | 1.2-2× | Software emulation overhead |

### 1.3 Ternary Embedding Representations

From research-response-A:
> "BitNet's {-1, 0, +1} (1.58 bits) viable for embeddings, but emulation overhead ~10-20% without hardware"

**Considerations for CogSynDelta**:
- VL-JEPA embeddings are dense, high-dimensional
- Ternary may lose subtle semantic distinctions
- Hybrid approach: ternary weights, FP activations

---

## 2. Mathematical Foundations

### 2.1 Quantization Theory

**Absmean scaling** (BitNet approach):
$$\gamma = \frac{1}{nm}\sum_{i,j}|W_{ij}|$$
$$\tilde{W} = \text{RoundClip}\left(\frac{W}{\gamma}, -1, 1\right)$$

**Why absmean over absmax?**
- More robust to outliers
- Better distribution matching for Gaussian weights
- Empirically better accuracy retention

### 2.2 Gradient Estimation

**Straight-Through Estimator (STE)**:
$$\frac{\partial L}{\partial W} \approx \frac{\partial L}{\partial \tilde{W}}$$

Despite theoretical issues (gradient of step function is zero), STE works well in practice because:
1. Gradient direction is approximately preserved
2. Learning dynamics remain stable
3. Final accuracy comparable to differentiable alternatives

### 2.3 Spectral Properties

**Question**: How does ternary constraint affect weight matrix spectrum?

**Hypothesis**: 
- Ternary weights have limited expressiveness in singular value space
- May preserve principal components but lose fine-grained structure
- Could align with algebraic training's spectral predictions

**Research needed**: Compare eigenvalue distributions of FP vs ternary matrices

---

## 3. Integration with Algebraic Training

### 3.1 Potential Synergies

| Algebraic Method | Ternary Integration |
|------------------|---------------------|
| NTK prediction | Predict FP weights, quantize to ternary |
| Fisher Information | Weight importance → ternary vs FP decision |
| Spectral analysis | Predict spectrum, construct ternary approximation |
| Natural gradient | Compute in FP, apply as ternary delta |

### 3.2 Open Problems

1. **NTK with ternary constraint**: Does closed-form solution exist?
2. **Fisher-weighted ternarization**: Preserve high-Fisher weights in FP?
3. **Spectral ternary synthesis**: Construct ternary matrix matching target spectrum?

### 3.3 Proposed Experiments

1. Train model algebraically (ADR-0009)
2. Convert to ternary
3. Measure accuracy delta
4. Compare to: (a) ternary-aware training, (b) direct ternary prediction

---

## 4. Implementation Notes

### 4.1 From Microsoft BitNet Repo

**Key implementation details**:
- Uses RMSNorm before ternary layers
- Applies ternary to weights only, activations use INT8
- Training uses FP16 with STE quantization in forward

### 4.2 Memory Layout

**Naive**: 2 bits per trit (25% waste)
```
[00=−1, 01=0, 10=+1, 11=unused]
```

**Optimized**: 5 trits per byte
```
value = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4
range: 0-242 (fits in uint8)
```

**Alternative**: Bit planes
```
positive_plane: bit[i,j] = 1 if W[i,j] == +1
negative_plane: bit[i,j] = 1 if W[i,j] == -1
zero: implicit when both planes are 0
```

Bit planes enable efficient matmul: `Y = scale * (X @ pos - X @ neg)`

### 4.3 Kernel Optimization Strategies

1. **Lookup tables**: Precompute all 243 5-trit decodings
2. **SIMD**: Process 32+ trits in parallel with AVX2/AVX512
3. **Sparse representation**: If many zeros, use sparse matmul
4. **Fused operations**: Combine unpack + matmul + scale

---

## 5. Risk Assessment

### 5.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Accuracy loss >5% | Medium | High | Hybrid FP/ternary |
| No speedup on target HW | Medium | Medium | Focus on memory savings |
| Training instability | Low | High | Use algebraic init |

### 5.2 Research Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| NTK incompatibility | Medium | Medium | Empirical validation |
| mHC degradation | Low | Medium | Keep gates in FP |
| Limited generalization | Medium | Low | Domain-specific eval |

---

## 6. Literature Summary

### 6.1 Core Papers

1. **BitNet (2024)**: arXiv:2402.17764
   - First 1.58-bit LLM matching FP16
   - Absmean quantization
   - Requires specialized training

2. **BitNet 2B4T (2025)**: arXiv:2504.12285
   - Open-source 2B parameter model
   - Competitive with Gemma-3
   - 0.4GB vs 1.4-4.8GB memory

3. **Ternary Weight Networks (2016)**: arXiv:1605.04711
   - Original ternary quantization
   - Threshold-based {-1, 0, +1}
   - Foundation for BitNet

### 6.2 Related Quantization

- **GPTQ**: 4-bit, Hessian-based (arXiv:2210.17323)
- **AWQ**: Activation-aware 4-bit (arXiv:2306.00978)
- **SpinQuant**: Rotation-based 4-bit (arXiv:2405.16406)

These represent fallback options if ternary proves insufficient.

---

## 7. Next Steps

1. [ ] Implement basic TernaryLinear layer
2. [ ] Validate STE gradient flow
3. [ ] Benchmark on MNIST/CIFAR
4. [ ] Compare to GPTQ 4-bit baseline
5. [ ] Assess algebraic training compatibility
6. [ ] Document findings in ADR-0010 update

---

*Notes compiled: 2026-01-19*
