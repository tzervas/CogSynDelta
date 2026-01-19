# Compression Fidelity Recovery Implementation Plan

> Step-by-step implementation plan to recover from 0.06 → ≥0.95 fidelity

## Executive Summary

**Current State**: 0.06 cosine similarity fidelity (catastrophic failure)
**Target State**: ≥0.95 fidelity at 10x compression with adjustable ratios
**Timeline**: 8-12 weeks
**Risk Level**: Medium (proven techniques, but requires careful implementation)

---

## Phase 1: Emergency Calibration (Week 1-2)

**Goal**: Recover from 0.06 → 0.85-0.90 fidelity via proper calibration

### Task 1.1: Diagnostic Analysis

```python
# File: src/cogsyndelta/memory/compression_diagnostics.py

def diagnose_quantization_failure(
    embeddings: torch.Tensor,
    quantized: torch.Tensor,
    dequantized: torch.Tensor,
) -> dict[str, Any]:
    """Identify the source of fidelity failure.

    Args:
        embeddings: Original embeddings [N, D]
        quantized: Quantized representation
        dequantized: Reconstructed embeddings [N, D]

    Returns:
        Diagnostic report with per-dimension analysis
    """
    cos_sim = F.cosine_similarity(embeddings, dequantized, dim=-1).mean()

    # Per-dimension error analysis
    dim_errors = (embeddings - dequantized).abs().mean(dim=0)
    worst_dims = dim_errors.topk(10)

    # Range preservation analysis
    true_range = embeddings.max(0)[0] - embeddings.min(0)[0]
    quant_range = dequantized.max(0)[0] - dequantized.min(0)[0]
    range_ratio = (quant_range / (true_range + 1e-8)).mean()

    return {
        'cosine_similarity': cos_sim.item(),
        'range_preservation_ratio': range_ratio.item(),  # Should be ~1.0
        'worst_dimensions': worst_dims.indices.tolist(),
        'worst_dim_errors': worst_dims.values.tolist(),
        'mean_absolute_error': (embeddings - dequantized).abs().mean().item(),
    }
```

### Task 1.2: Calibration Infrastructure

```python
# File: src/cogsyndelta/memory/calibration.py

@dataclass
class CalibrationStats:
    """Per-dimension calibration statistics."""
    min: torch.Tensor
    max: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    percentile_01: torch.Tensor
    percentile_99: torch.Tensor


def calibrate_from_corpus(
    embeddings: torch.Tensor,
    n_samples: int = 10000,
) -> CalibrationStats:
    """Compute calibration statistics from representative corpus.

    CRITICAL: This is the fix for 0.06 fidelity.
    The current implementation uses fixed ranges that don't match
    the actual embedding distribution.

    Args:
        embeddings: Representative corpus [N, D] where N >= n_samples
        n_samples: Number of samples to use for calibration

    Returns:
        CalibrationStats with per-dimension statistics
    """
    sample = embeddings[:n_samples]

    return CalibrationStats(
        min=sample.min(dim=0)[0],
        max=sample.max(dim=0)[0],
        mean=sample.mean(dim=0),
        std=sample.std(dim=0),
        percentile_01=torch.quantile(sample, 0.01, dim=0),
        percentile_99=torch.quantile(sample, 0.99, dim=0),
    )


def calibrated_int8_quantize(
    values: torch.Tensor,
    calibration: CalibrationStats,
    use_percentiles: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize using calibrated ranges, not fixed [-1, 1].

    Args:
        values: Values to quantize [N, D]
        calibration: Pre-computed calibration statistics
        use_percentiles: Use 1st/99th percentile for outlier robustness

    Returns:
        (quantized_uint8, scale, zero_point) for dequantization
    """
    if use_percentiles:
        min_val = calibration.percentile_01
        max_val = calibration.percentile_99
    else:
        min_val = calibration.min
        max_val = calibration.max

    scale = (max_val - min_val) / 255
    zero_point = min_val

    # Clamp to calibrated range, then quantize
    clamped = values.clamp(min_val, max_val)
    quantized = torch.round((clamped - zero_point) / (scale + 1e-8))
    quantized = quantized.clamp(0, 255).to(torch.uint8)

    return quantized, scale, zero_point


def calibrated_dequantize(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """Dequantize using stored scale and zero point."""
    return quantized.float() * scale + zero_point
```

### Task 1.3: Integration with Dense Embeddings

Update `dense_embeddings.py` to use calibration:

```python
# Changes to DenseDifferentialMemoryStore.__init__
def __init__(self, ..., calibration_embeddings: torch.Tensor | None = None):
    ...
    # Add calibration
    if calibration_embeddings is not None:
        self.calibration = calibrate_from_corpus(calibration_embeddings)
    else:
        self.calibration = None
        warnings.warn(
            "No calibration data provided. Fidelity may be severely degraded. "
            "Provide ≥10K representative embeddings for proper calibration."
        )
```

### Deliverables Phase 1

- [ ] `compression_diagnostics.py` - Diagnostic tools
- [ ] `calibration.py` - Calibration infrastructure
- [ ] Updated `dense_embeddings.py` - Calibration integration
- [ ] Unit tests for calibration
- [ ] Benchmark: 0.06 → 0.85-0.90 fidelity

---

## Phase 2: Matryoshka Representation Learning (Week 3-4)

**Goal**: Implement flexible compression via dimension truncation

### Task 2.1: Multi-Scale Loss Training

```python
# File: src/cogsyndelta/memory/matryoshka.py

class MatryoshkaEncoder(nn.Module):
    """Encoder trained with multi-scale loss for flexible compression.

    Based on Kusupati et al. "Matryoshka Representation Learning" (NeurIPS 2022).
    Early dimensions capture coarse semantics, later dimensions add fine detail.
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dims: list[int] = [64, 128, 256, 512, 768],
    ):
        super().__init__()
        self.output_dims = sorted(output_dims)
        self.max_dim = max(output_dims)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.max_dim),
            nn.LayerNorm(self.max_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to full dimension."""
        return self.encoder(x)

    def encode_at_dim(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Encode and truncate to specified dimension."""
        full = self.forward(x)
        return full[..., :dim]

    def compute_multi_scale_loss(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable,
        dim_weights: dict[int, float] | None = None,
    ) -> torch.Tensor:
        """Compute weighted loss across all truncation points.

        Args:
            embeddings: Encoded embeddings [N, max_dim]
            targets: Target embeddings or labels
            loss_fn: Loss function (e.g., contrastive, cosine)
            dim_weights: Optional weights per dimension

        Returns:
            Weighted sum of losses at each truncation point
        """
        if dim_weights is None:
            dim_weights = {d: 1.0 for d in self.output_dims}

        total_loss = 0.0
        for dim in self.output_dims:
            truncated = embeddings[..., :dim]
            loss = loss_fn(truncated, targets)
            total_loss += dim_weights[dim] * loss

        return total_loss / len(self.output_dims)
```

### Task 2.2: Pre-trained Model Integration

```python
# Use pre-trained Matryoshka models as baseline
from sentence_transformers import SentenceTransformer

def load_pretrained_matryoshka() -> SentenceTransformer:
    """Load pre-trained Matryoshka model for validation."""
    return SentenceTransformer('tomaarsen/mpnet-base-nli-matryoshka')


def validate_matryoshka_fidelity(
    model: SentenceTransformer,
    test_sentences: list[str],
    truncation_dims: list[int] = [64, 128, 256, 512, 768],
) -> dict[int, float]:
    """Validate fidelity at different truncation points."""
    full_embeddings = model.encode(test_sentences, convert_to_tensor=True)

    fidelity_by_dim = {}
    for dim in truncation_dims:
        truncated = full_embeddings[..., :dim]
        # Pad back to full dim for comparison
        padded = F.pad(truncated, (0, 768 - dim))
        cos_sim = F.cosine_similarity(full_embeddings, padded, dim=-1).mean()
        fidelity_by_dim[dim] = cos_sim.item()

    return fidelity_by_dim
```

### Deliverables Phase 2

- [ ] `matryoshka.py` - Multi-scale encoder
- [ ] Training script for Matryoshka on AllNLI
- [ ] Pre-trained model validation
- [ ] Benchmark: flexible compression 4x-12x at 0.88-0.97 fidelity

---

## Phase 3: Residual Vector Quantization (Week 5-6)

**Goal**: Achieve ≥0.95 fidelity through residual encoding

### Task 3.1: RVQ Implementation

```python
# File: src/cogsyndelta/memory/residual_quantization.py

from vector_quantize_pytorch import ResidualVQ

class AdaptiveResidualQuantizer(nn.Module):
    """RVQ with early stopping based on fidelity threshold.

    Uses vector-quantize-pytorch for production-ready RVQ.
    """

    def __init__(
        self,
        dim: int,
        num_stages: int = 8,
        codebook_size: int = 1024,
        target_fidelity: float = 0.95,
    ):
        super().__init__()
        self.target_fidelity = target_fidelity
        self.num_stages = num_stages

        self.rvq = ResidualVQ(
            dim=dim,
            num_quantizers=num_stages,
            codebook_size=codebook_size,
            kmeans_init=True,
            kmeans_iters=10,
            decay=0.8,
            commitment_weight=1.0,
        )

    def encode(
        self,
        x: torch.Tensor,
        max_stages: int | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Encode with adaptive stage count.

        Args:
            x: Input embeddings [N, D]
            max_stages: Override maximum stages (for compression control)

        Returns:
            (quantized, stages_used)
        """
        if max_stages is None:
            max_stages = self.num_stages

        quantized, indices, _ = self.rvq(x)

        # Check fidelity and potentially use fewer stages
        fidelity = F.cosine_similarity(x, quantized, dim=-1).mean()

        if fidelity >= self.target_fidelity:
            return quantized, max_stages

        return quantized, max_stages

    def encode_adaptive(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Encode with early stopping when target fidelity achieved."""
        residual = x.clone()
        reconstruction = torch.zeros_like(x)

        for stage in range(self.num_stages):
            # Quantize current residual
            stage_quantized, _, _ = self.rvq.quantizers[stage](residual)
            reconstruction = reconstruction + stage_quantized
            residual = residual - stage_quantized

            # Check fidelity
            fidelity = F.cosine_similarity(x, reconstruction, dim=-1).mean()
            if fidelity >= self.target_fidelity:
                return reconstruction, stage + 1

        return reconstruction, self.num_stages
```

### Task 3.2: Codebook Training on LAION

```python
def train_codebooks_on_laion(
    rvq: AdaptiveResidualQuantizer,
    embeddings_path: str,
    batch_size: int = 8192,
    num_epochs: int = 10,
) -> None:
    """Train RVQ codebooks on LAION-5B embeddings.

    Uses pre-computed CLIP embeddings from LAION for efficient training.
    """
    from datasets import load_dataset

    dataset = load_dataset('laion/laion2b-en-vit-h-14-embeddings', streaming=True)

    optimizer = torch.optim.Adam(rvq.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for batch in dataset.iter(batch_size):
            embeddings = torch.tensor(batch['embeddings'])

            # Forward pass
            quantized, indices, commit_loss = rvq.rvq(embeddings)
            recon_loss = F.mse_loss(quantized, embeddings)
            loss = recon_loss + commit_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log fidelity
            fidelity = F.cosine_similarity(embeddings, quantized, dim=-1).mean()
            print(f"Epoch {epoch}, Fidelity: {fidelity:.4f}")
```

### Deliverables Phase 3

- [ ] `residual_quantization.py` - Adaptive RVQ
- [ ] Codebook training pipeline
- [ ] Benchmark: 0.93 → 0.95-0.97 fidelity at 8-16x compression

---

## Phase 4: Adjustable Compression API (Week 7-8)

**Goal**: Unified API with fidelity guarantees

### Task 4.1: Compression Manager

```python
# File: src/cogsyndelta/memory/adaptive_compression.py

@dataclass
class CompressionResult:
    """Result of adaptive compression."""
    compressed: torch.Tensor
    indices: torch.Tensor | None
    compression_ratio: float
    actual_fidelity: float
    stages_used: int
    method: str  # 'matryoshka', 'rvq', 'hybrid', 'lossless'


class AdaptiveCompressionManager:
    """Adjustable compression with fidelity guarantees.

    Implements the tiered compression strategy:
    1. Try Matryoshka truncation (fastest)
    2. Add RVQ if needed for higher fidelity
    3. Fall back to lossless if target unachievable
    """

    def __init__(
        self,
        embed_dim: int = 768,
        target_fidelity: float = 0.95,
        min_compression: float = 4.0,
        max_compression: float = 20.0,
        calibration: CalibrationStats | None = None,
    ):
        self.embed_dim = embed_dim
        self.target_fidelity = target_fidelity
        self.min_compression = min_compression
        self.max_compression = max_compression
        self.calibration = calibration

        # Compression stages
        self.matryoshka = MatryoshkaEncoder(embed_dim)
        self.rvq = AdaptiveResidualQuantizer(embed_dim, target_fidelity=target_fidelity)

    def compress(
        self,
        embeddings: torch.Tensor,
        target_ratio: float | None = None,
    ) -> CompressionResult:
        """Compress with adjustable ratio while guaranteeing fidelity.

        If target_ratio would violate target_fidelity,
        automatically reduce compression to maintain guarantee.

        Args:
            embeddings: Input embeddings [N, D]
            target_ratio: Desired compression (None = auto-select max)

        Returns:
            CompressionResult with actual fidelity and method used
        """
        if target_ratio is None:
            target_ratio = self.max_compression

        # Clamp to valid range
        target_ratio = max(self.min_compression, min(target_ratio, self.max_compression))

        # Try Matryoshka first (truncation-based)
        target_dim = int(self.embed_dim / target_ratio)
        matryoshka_result = self._try_matryoshka(embeddings, target_dim)

        if matryoshka_result.actual_fidelity >= self.target_fidelity:
            return matryoshka_result

        # Try RVQ for higher fidelity
        rvq_result = self._try_rvq(embeddings, target_ratio)

        if rvq_result.actual_fidelity >= self.target_fidelity:
            return rvq_result

        # Reduce compression until fidelity achieved
        return self._find_acceptable_compression(embeddings)

    def _try_matryoshka(
        self,
        embeddings: torch.Tensor,
        target_dim: int,
    ) -> CompressionResult:
        """Try Matryoshka truncation."""
        encoded = self.matryoshka.encode_at_dim(embeddings, target_dim)
        reconstructed = F.pad(encoded, (0, self.embed_dim - target_dim))

        fidelity = F.cosine_similarity(embeddings, reconstructed, dim=-1).mean().item()
        ratio = self.embed_dim / target_dim

        return CompressionResult(
            compressed=encoded,
            indices=None,
            compression_ratio=ratio,
            actual_fidelity=fidelity,
            stages_used=1,
            method='matryoshka',
        )

    def _try_rvq(
        self,
        embeddings: torch.Tensor,
        target_ratio: float,
    ) -> CompressionResult:
        """Try RVQ compression."""
        quantized, stages = self.rvq.encode_adaptive(embeddings)

        fidelity = F.cosine_similarity(embeddings, quantized, dim=-1).mean().item()
        # RVQ ratio depends on codebook size and stages
        ratio = self.embed_dim * 4 / (stages * 2)  # Approximate

        return CompressionResult(
            compressed=quantized,
            indices=None,
            compression_ratio=ratio,
            actual_fidelity=fidelity,
            stages_used=stages,
            method='rvq',
        )

    def _find_acceptable_compression(
        self,
        embeddings: torch.Tensor,
    ) -> CompressionResult:
        """Binary search for acceptable compression ratio."""
        low, high = self.min_compression, self.max_compression

        best_result = None
        while high - low > 0.5:
            mid = (low + high) / 2
            result = self._try_rvq(embeddings, mid)

            if result.actual_fidelity >= self.target_fidelity:
                best_result = result
                low = mid
            else:
                high = mid

        if best_result is None:
            # Fall back to lossless
            return CompressionResult(
                compressed=embeddings,
                indices=None,
                compression_ratio=1.0,
                actual_fidelity=1.0,
                stages_used=0,
                method='lossless',
            )

        return best_result
```

### Deliverables Phase 4

- [ ] `adaptive_compression.py` - Unified compression API
- [ ] Fidelity monitoring and alerting
- [ ] Integration with existing memory system
- [ ] Benchmark: adjustable 4x-20x with ≥0.95 guarantee

---

## Phase 5: VSA/Holographic Integration (Week 9-12)

**Goal**: Brain-inspired memory with Modern Hopfield and VSA

### Task 5.1: torchhd Integration

```python
# File: src/cogsyndelta/memory/vsa_memory.py

import torchhd
from torchhd import embeddings, functional

class VSAMemoryTier(nn.Module):
    """VSA-based memory tier with temporal binding.

    Uses torchhd for GPU-accelerated VSA operations.
    Dimension ≥10,000 recommended for reliable operations.
    """

    def __init__(
        self,
        dim: int = 10000,
        num_slots: int = 1000,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots

        # Random hypervectors for binding
        self.time_hvs = nn.Parameter(
            torchhd.random(num_slots, dim),
            requires_grad=False,
        )
        self.role_hvs = nn.Parameter(
            torchhd.random(10, dim),  # Roles for binding
            requires_grad=False,
        )

        # Memory bank
        self.memory = nn.Parameter(
            torch.zeros(num_slots, dim),
            requires_grad=False,
        )
        self.write_ptr = 0

    def bind(self, content: torch.Tensor, time_idx: int) -> torch.Tensor:
        """Bind content with temporal position."""
        time_hv = self.time_hvs[time_idx]
        return functional.bind(content, time_hv)

    def bundle(self, items: list[torch.Tensor]) -> torch.Tensor:
        """Bundle multiple items via superposition."""
        stacked = torch.stack(items)
        return functional.bundle(stacked)

    def write(self, content: torch.Tensor) -> int:
        """Write content to memory with temporal binding."""
        bound = self.bind(content, self.write_ptr)
        self.memory[self.write_ptr] = bound
        idx = self.write_ptr
        self.write_ptr = (self.write_ptr + 1) % self.num_slots
        return idx

    def query(self, cue: torch.Tensor, k: int = 5) -> torch.Tensor:
        """Query memory by similarity to cue."""
        similarities = functional.cosine_similarity(cue, self.memory)
        top_k = similarities.topk(k)
        return top_k.indices
```

### Task 5.2: Modern Hopfield Network

```python
class ModernHopfieldLayer(nn.Module):
    """Modern Hopfield Network for active memory.

    Based on Ramsauer et al. "Hopfield Networks is All You Need" (2020).
    Achieves exponential storage capacity 2^(d/2) with one-step convergence.
    """

    def __init__(
        self,
        dim: int,
        num_patterns: int = 1000,
        beta: float = 8.0,
    ):
        super().__init__()
        self.dim = dim
        self.beta = beta
        self.patterns = nn.Parameter(torch.randn(num_patterns, dim))

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Modern Hopfield energy."""
        # E(s) = -lse(β * patterns @ s) / β + 0.5 * ||s||^2
        similarities = self.beta * torch.mm(self.patterns, state.T)
        lse = torch.logsumexp(similarities, dim=0)
        return -lse / self.beta + 0.5 * (state ** 2).sum(dim=-1)

    def update(self, state: torch.Tensor) -> torch.Tensor:
        """One-step update (converges to fixed point)."""
        similarities = self.beta * torch.mm(state, self.patterns.T)
        attention = F.softmax(similarities, dim=-1)
        return torch.mm(attention, self.patterns)

    def store(self, pattern: torch.Tensor) -> None:
        """Store new pattern in memory."""
        with torch.no_grad():
            self.patterns.data = torch.cat([
                self.patterns.data,
                pattern.unsqueeze(0),
            ], dim=0)
```

### Deliverables Phase 5

- [ ] `vsa_memory.py` - VSA memory tier
- [ ] `hopfield_memory.py` - Modern Hopfield integration
- [ ] Tiered memory with VSA + FAISS hybrid
- [ ] Benchmark: brain-inspired retrieval performance

---

## Success Criteria

| Phase | Target Fidelity | Compression | Status |
|-------|-----------------|-------------|--------|
| Phase 1 | 0.85-0.90 | Baseline | 🔴 Not started |
| Phase 2 | 0.88-0.97 | 4x-12x | 🔴 Not started |
| Phase 3 | 0.95-0.97 | 8x-16x | 🔴 Not started |
| Phase 4 | **≥0.95** | **4x-20x (adjustable)** | 🔴 Not started |
| Phase 5 | Brain-inspired | + VSA | 🔴 Not started |

---

## Adjustable Compression API Design

The key innovation is **fidelity-first compression** where the hard constraint is fidelity (≥0.95) and compression ratio is optimized within that constraint.

### API Usage Examples

```python
# Initialize with fidelity guarantee
manager = AdaptiveCompressionManager(
    target_fidelity=0.95,  # Hard constraint
    max_compression=20.0,   # Soft target
)

# Compress with automatic ratio selection
result = manager.compress(embeddings)
# result.compression_ratio might be 12x if that's max achieving 0.95

# Compress with specific target (will auto-reduce if needed)
result = manager.compress(embeddings, target_ratio=16.0)
# If 16x only achieves 0.92, result.compression_ratio might be 10x

# Check guarantees
assert result.actual_fidelity >= 0.95  # Always holds
```

### Compression Curve Precomputation

For efficiency, precompute rate-distortion curves per embedding type:

```python
@dataclass
class CompressionCurve:
    """Precomputed rate-distortion curve."""
    ratios: list[float]       # [4, 6, 8, 10, 12, 16, 20]
    fidelities: list[float]   # [0.98, 0.96, 0.95, 0.94, 0.92, 0.90, 0.85]

    def find_ratio_for_fidelity(self, target: float) -> float:
        """Find max compression ratio achieving target fidelity."""
        for ratio, fidelity in zip(self.ratios, self.fidelities):
            if fidelity >= target:
                max_ratio = ratio
        return max_ratio


def precompute_curve(
    manager: AdaptiveCompressionManager,
    calibration_embeddings: torch.Tensor,
    ratios: list[float] = [4, 6, 8, 10, 12, 16, 20],
) -> CompressionCurve:
    """Precompute rate-distortion curve on calibration set."""
    fidelities = []
    for ratio in ratios:
        result = manager.compress(calibration_embeddings, target_ratio=ratio)
        fidelities.append(result.actual_fidelity)

    return CompressionCurve(ratios=ratios, fidelities=fidelities)
```

---

## Integration with embeddenator-core Ecosystem

The implementation builds on composable primitives from embeddenator sister projects:

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CogSynDelta Memory                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            AdaptiveCompressionManager                │   │
│  │  (CogSynDelta integration layer)                    │   │
│  └─────────────────────────────────────────────────────┘   │
│           │              │              │                    │
│           ▼              ▼              ▼                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │embeddenator │ │embeddenator │ │embeddenator │           │
│  │-calibration │ │-rvq         │ │-vsa         │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│           │              │              │                    │
│           ▼              ▼              ▼                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               embeddenator-core                      │   │
│  │  (Shared types, interfaces, utilities)              │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  External Dependencies:                                      │
│  ┌─────────┐ ┌─────────────────────┐ ┌─────────┐          │
│  │ torchhd │ │vector-quantize-torch│ │ FAISS   │          │
│  └─────────┘ └─────────────────────┘ └─────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## Testing Strategy

### Unit Tests

```python
def test_fidelity_guarantee():
    """CRITICAL: Fidelity guarantee must never be violated."""
    manager = AdaptiveCompressionManager(target_fidelity=0.95)

    # Test across various compression targets
    for target_ratio in [4, 8, 12, 16, 20, 30]:
        result = manager.compress(test_embeddings, target_ratio=target_ratio)
        assert result.actual_fidelity >= 0.95, (
            f"Fidelity guarantee violated: {result.actual_fidelity} < 0.95 "
            f"at target ratio {target_ratio}"
        )


def test_compression_optimizes_within_constraint():
    """Verify compression is maximized within fidelity constraint."""
    manager = AdaptiveCompressionManager(target_fidelity=0.95)

    # Should achieve close to max compression when data allows
    easy_result = manager.compress(easy_to_compress_embeddings)
    assert easy_result.compression_ratio >= 10.0

    # Should reduce compression for difficult data
    hard_result = manager.compress(hard_to_compress_embeddings)
    assert hard_result.actual_fidelity >= 0.95  # Still meets guarantee
```

### Benchmark Suite

```python
def benchmark_adjustable_compression():
    """Benchmark adjustable compression on standard datasets."""
    datasets = ['allnli', 'msmarco', 'laion_embeddings', 'gist1m']
    fidelity_targets = [0.90, 0.95, 0.97, 0.99]

    results = []
    for dataset in datasets:
        for target in fidelity_targets:
            manager = AdaptiveCompressionManager(target_fidelity=target)
            embeddings = load_benchmark_embeddings(dataset)

            result = manager.compress(embeddings)
            results.append({
                'dataset': dataset,
                'target_fidelity': target,
                'actual_fidelity': result.actual_fidelity,
                'compression_ratio': result.compression_ratio,
                'method': result.method,
            })

    return pd.DataFrame(results)
```

---

## References

- Kusupati et al. "Matryoshka Representation Learning" (NeurIPS 2022)
- Ma et al. "BitNet b1.58" (arXiv:2402.17764)
- QINCo2 (Meta AI, ICLR 2025, arXiv:2501.03078)
- Ramsauer et al. "Hopfield Networks is All You Need" (arXiv:2008.02217)
- torchhd: https://github.com/hyperdimensional-computing/torchhd
- vector-quantize-pytorch: https://github.com/lucidrains/vector-quantize-pytorch
- embeddenator-core ecosystem: Sister projects for embedding compression

---

*Last Updated: January 19, 2026*
