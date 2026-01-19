# ADR-0008: Compression Fidelity Recovery Strategy

## Status

**ACCEPTED** - 2026-01-18

## Context

CogSynDelta's dense differential embedding system currently achieves **0.06 cosine similarity fidelity** when compressing and reconstructing embeddings. This is a **catastrophic failure** (essentially orthogonal to original) vs the target of **≥0.95 fidelity** at 10x compression.

### Root Cause Analysis

The 0.06 fidelity indicates fundamental breakdown, not minor quality loss:

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

### Stage 1: Emergency Calibration (0.06 → 0.85-0.90)

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

For the hippocampus memory tier:
- **Modern Hopfield Networks** for active memory (exponential capacity: 2^(d/2))
- **VSA binding** for temporal context (torchhd library)
- **FAISS** for precise similarity search on compressed vectors

## Consequences

### Positive

- Clear path from 0.06 → ≥0.95 fidelity
- Adjustable compression with guarantees
- Brain-inspired architecture alignment
- Uses proven techniques (Matryoshka, RVQ, VSA)

### Negative

- Implementation complexity increases
- May need to retrain existing models
- Storage format changes required
- Performance overhead for calibration

### Risks

| Risk | Mitigation |
|------|------------|
| Calibration drift | Periodic recalibration on new data |
| RVQ codebook overfitting | Train on diverse datasets (AllNLI, LAION) |
| VSA dimension requirements | Use d≥10,000 for reliable operations |

## Implementation Plan

| Phase | Duration | Target |
|-------|----------|--------|
| Calibration | Week 1-2 | 0.06 → 0.85-0.90 |
| Matryoshka | Week 3-4 | 0.90 → 0.93 |
| RVQ | Week 5-6 | 0.93 → 0.95+ |
| VSA Integration | Week 7-8 | Brain-inspired memory |
| Optimization | Week 9-12 | Performance tuning |

## References

- Kusupati et al. "Matryoshka Representation Learning" (NeurIPS 2022)
- Ma et al. "BitNet b1.58" (arXiv:2402.17764)
- QINCo2 (Meta AI, ICLR 2025)
- Ramsauer et al. "Hopfield Networks is All You Need" (arXiv:2008.02217)
- torchhd library: https://github.com/hyperdimensional-computing/torchhd
- vector-quantize-pytorch: https://github.com/lucidrains/vector-quantize-pytorch

---

*Authored: 2026-01-18 | Supersedes: None | Related: ADR-0002*
