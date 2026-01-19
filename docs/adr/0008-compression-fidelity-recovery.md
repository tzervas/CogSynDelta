# ADR-0008: Compression Fidelity Recovery Strategy

## Status

**ACCEPTED** - 2026-01-19

## Context

CogSynDelta's dense differential embedding system currently achieves **0.67 cosine similarity fidelity** when compressing and reconstructing embeddings at 2x compression. This is **significantly below target** vs the goal of **≥0.95 fidelity** at 10x compression (benchmark data from RTX 5080 baseline).

### Integration with embeddenator-core

This ADR leverages the **embeddenator** family of component libraries:
- **embeddenator-core**: Core embedding compression primitives
- **embeddenator-vsa**: Vector Symbolic Architecture operations
- **embeddenator-rvq**: Residual Vector Quantization
- **embeddenator-calibration**: Adaptive calibration infrastructure

The fidelity crisis resolution strategy aligns with embeddenator-core's architecture for maximum code reuse and interoperability.

### Root Cause Analysis

The 0.67 fidelity (from RTX 5080 benchmark results) indicates significant quality loss:

1. **Uncalibrated quantization buckets** - Fixed ranges (e.g., [-1, 1]) don't match actual embedding distribution
2. **Insufficient bit-width for differential signals** - Differential updates have different statistics than absolute embeddings
3. **Cumulative error propagation** - Each encoding step compounds error without correction
4. **Distribution mismatch** - Assumes Gaussian when embeddings may be heavy-tailed or multi-modal

### State of the Art

Research shows achievable fidelity by compression ratio:

| Compression | Best Fidelity | Technique | Source |
|-------------|---------------|-----------|--------|
| 4x | 0.97-0.99 | Scalar Quantization / Matryoshka | CoRECT 2024 |
| 8x | 0.95-0.98 | OPQ, Matryoshka MRL | Benchmark studies |
| 16x | 0.90-0.95 | QINCo2, RVQ, BitNet b1.58 | Meta AI (ICLR 2025) |
| 24x | 0.85-0.92 | PQ with rescoring | FAISS documentation |
| 100x | <0.70 | Not demonstrated | Research frontier |

## Decision

Implement a **staged recovery approach** with adjustable compression that guarantees fidelity:

### Stage 1: Emergency Calibration (0.67 → 0.85-0.90)

```python
def calibrate_quantization_ranges(embeddings: torch.Tensor, n_samples: int = 10000):
    """Compute proper calibration ranges per dimension."""
    sample = embeddings[:n_samples]
    return {
        'min': sample.min(dim=0)[0],
        'max': sample.max(dim=0)[0],
        'mean': sample.mean(dim=0),
        'std': sample.std(dim=0),
    }

def calibrated_quantize(values: torch.Tensor, calibration: dict, bits: int = 8):
    """Quantize using calibrated ranges, not fixed [-1, 1]."""
    scale = (calibration['max'] - calibration['min']) / (2**bits - 1)
    zero_point = calibration['min']
    quantized = torch.round((values - zero_point) / scale).clamp(0, 2**bits - 1)
    return quantized.to(torch.uint8), scale, zero_point
```

### Stage 2: Matryoshka Representation Learning (flexible compression)

Train embeddings where early dimensions capture coarse semantics:

```python
# Multi-scale loss at truncation points
Total_Loss = Σ(weight_i × Loss(embedding[:dim_i]))
# dims = [64, 128, 256, 512, 768]
```

Benefits:
- Single model serves multiple compression ratios
- 768→256 (3x): ~0.97 fidelity
- 768→128 (6x): ~0.93 fidelity
- 768→64 (12x): ~0.88 fidelity

### Stage 3: Residual Vector Quantization (0.90 → 0.95+)

Add residual stages to capture what base quantization misses:

```python
class ResidualQuantizer:
    def encode(self, x: torch.Tensor, target_fidelity: float = 0.95):
        codes = []
        residual = x.clone()

        for stage in self.stages:
            code = stage.quantize(residual)
            codes.append(code)
            reconstructed = stage.dequantize(code)
            residual = residual - reconstructed

            # Check fidelity and stop early if achieved
            current_fidelity = F.cosine_similarity(x, self.decode(codes))
            if current_fidelity >= target_fidelity:
                break

        return codes
```

### Stage 4: Adjustable Compression API

```python
class AdaptiveCompressionManager:
    """Compression with fidelity guarantees."""

    def compress(
        self,
        embeddings: torch.Tensor,
        target_ratio: float | None = None,
        min_fidelity: float = 0.95,
    ) -> tuple[CompressedData, Metrics]:
        """
        If target_ratio would violate min_fidelity,
        automatically reduce compression to maintain guarantee.
        """
        ...
```

### Stage 5: VSA/Holographic Integration (brain-inspired)

For the hippocampus memory tier, implement holographic computing with guaranteed capacity bounds:

**Vector Symbolic Architecture (VSA) Operations:**
```python
# MAP-B binding (Hadamard product) - 100% fidelity for atomic vectors
def map_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a * b  # Element-wise multiplication

# HRR binding (circular convolution) - ~95% fidelity at d=10,000
def hrr_bind(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_fft = torch.fft.fft(a, dim=-1)
    b_fft = torch.fft.fft(b, dim=-1)
    return torch.fft.ifft(a_fft * b_fft, dim=-1).real
```

**Modern Hopfield Networks for Active Memory:**
- Storage capacity: ~2^(d/2) patterns (exponential in dimension)
- One-step convergence via transformer attention equivalent
- β parameter controls metastable states vs precise retrieval

**Tiered Memory Architecture:**
```
ACTIVE MEMORY (Hopfield, ~100 items, minutes)
    ↓ consolidation
SHORT-TERM MEMORY (VSA bundling, ~10K items, hours)
    ↓ consolidation
LONG-TERM MEMORY (FAISS IVF-PQ, millions+, permanent)
```

**Capacity Bounds:** For bundling k items from M-item codebook at fidelity S:
```
n ≥ k / (1 - S²) × log(M)
```
At k=20, M=1000, S=0.95: n ≥ 2,050 dimensions (use n≥10,000 for reliability)

### Stage 6: Adjustable Compression with Fidelity Guarantees

```python
@dataclass
class CompressionConfig:
    """Adjustable compression targeting fidelity over ratio."""
    min_fidelity: float = 0.95       # Hard constraint
    target_ratio: float = 10.0       # Soft target
    max_ratio: float = 20.0          # Upper bound
    adaptive_stages: bool = True     # Early-stop RVQ
    fallback_lossless: bool = True   # FP16 if fidelity unachievable

class AdaptiveCompressionManager:
    """Compress with guaranteed fidelity, maximizing compression ratio."""

    def compress(
        self,
        embeddings: torch.Tensor,
        config: CompressionConfig = CompressionConfig(),
    ) -> CompressionResult:
        """
        Strategy:
        1. Try Matryoshka truncation (fastest, ~4-12x, 0.88-0.97)
        2. Add RVQ if fidelity insufficient (~8-16x, 0.90-0.97)
        3. Reduce compression if still below target
        4. Fall back to FP16 if fidelity unachievable
        """
        ...
```

## Consequences

### Positive

- Clear path from 0.67 → ≥0.95 fidelity
- **Adjustable compression** with hard fidelity guarantees
- Brain-inspired architecture alignment (CLS theory)
- Uses proven techniques (Matryoshka, RVQ, VSA, Hopfield)
- Interoperates with embeddenator-core component libraries

### Negative

- Implementation complexity increases
- May need to retrain existing models
- Storage format changes required
- Performance overhead for calibration
- VSA requires high dimensions (≥10,000 for reliability)

### Neutral

- Trade-off between compression ratio and fidelity is explicit
- Calibration data requirements (≥10K representative embeddings)

### Risks

| Risk | Mitigation |
|------|------------|
| Calibration drift | Periodic recalibration, online statistics |
| RVQ codebook overfitting | Train on diverse datasets (AllNLI, LAION) |
| VSA dimension requirements | Use d≥10,000 for reliable operations |
| Fidelity regression | Continuous monitoring, alert at <0.93 |

## Implementation Plan

| Phase | Duration | Target Fidelity | Key Deliverables |
|-------|----------|-----------------|------------------|
| 1. Calibration | Week 1-2 | 0.67 → 0.85-0.90 | calibration.py, diagnostics.py |
| 2. Matryoshka | Week 3-4 | 0.90 → 0.93 | matryoshka.py, AllNLI training |
| 3. RVQ | Week 5-6 | 0.93 → 0.95+ | residual_quantization.py |
| 4. Adjustable API | Week 7-8 | Guaranteed 0.95+ | AdaptiveCompressionManager |
| 5. VSA/Hopfield | Week 9-10 | Brain-inspired tier | vsa_memory.py, hopfield.py |
| 6. Optimization | Week 11-12 | RTX 5080 tuning | Triton kernels, BF16 |

## Testing Strategy

```python
def test_fidelity_guarantee():
    """Verify fidelity never drops below threshold."""
    manager = AdaptiveCompressionManager(min_fidelity=0.95)

    for compression_ratio in [4, 8, 12, 16, 20]:
        result = manager.compress(test_embeddings, target_ratio=compression_ratio)
        assert result.actual_fidelity >= 0.95, f"Fidelity {result.actual_fidelity} < 0.95"
        # Ratio may be lower than target if fidelity constraint active
        print(f"Target {compression_ratio}x → Actual {result.compression_ratio:.1f}x @ {result.actual_fidelity:.3f}")
```

## References

- Kusupati et al. "Matryoshka Representation Learning" (NeurIPS 2022)
- Ma et al. "BitNet b1.58" (arXiv:2402.17764)
- QINCo2 (Meta AI, ICLR 2025, arXiv:2501.03078)
- Ramsauer et al. "Hopfield Networks is All You Need" (arXiv:2008.02217)
- Howard & Kahana "Temporal Context Model" (2002)
- torchhd library: https://github.com/hyperdimensional-computing/torchhd
- vector-quantize-pytorch: https://github.com/lucidrains/vector-quantize-pytorch
- embeddenator-core: Sister project for embedding compression primitives

---

*Authored: 2026-01-19 | Supersedes: None | Related: ADR-0002, ADR-0004*
