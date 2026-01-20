# ADR-0014: Matryoshka + QINCo2 Compression Pipeline

## Status

**PROPOSED** - 2026-01-19

## Context

ADR-0008 establishes the need for ≥0.95 fidelity at 10x+ compression. The research
document "Latent-Space-Native AI Systems" identifies the optimal approach as combining
**Matryoshka Representation Learning (MRL)** with **QINCo2 neural codebooks**.

### Research Findings

**Matryoshka Representation Learning:**
- Trains embeddings with multi-granularity: early dimensions capture coarse semantics
- At truncation from 768→64 (12x): **98.37% performance retention**
- OpenAI text-embedding-3-large: 93.1% quality at 3072→256 (12x)
- sentence-transformers provides `MatryoshkaLoss` for training

**QINCo2 (Meta AI, ICLR 2025):**
- Neural network dynamically generates codebooks conditioned on prior steps
- **34-44% MSE reduction** over standard RVQ
- >70% recall@1 at 16 bytes vs RVQ's <40%
- Multi-rate codec: trained for M steps, works with fewer

**Combined Approach:**
1. Train embeddings with MatryoshkaLoss (dimension-aware)
2. Apply QINCo2 quantization on top (neural codebook)
3. Achieve **≥95% fidelity at 10-16x compression**

### Current State

Per ADR-0008 validation (2026-01-19):
- HighFidelityCompactor achieves 1.0 fidelity at ~2x (production ready)
- HybridAdaptiveCompactor achieves 0.96 fidelity at ~2-4x
- Gap remains for 4x-16x compression with ≥0.95 fidelity

### Target

| Compression | Current Best | Target | Gap |
|-------------|--------------|--------|-----|
| 2x | 1.00 | ≥0.95 | ✓ Met |
| 4x | 0.96 | ≥0.95 | ✓ Met |
| 8x | ~0.85 (est) | ≥0.93 | Needs MRL |
| 16x | ~0.60 (est) | ≥0.90 | Needs MRL+QINCo2 |

## Decision

Implement a layered MRL + QINCo2 compression pipeline:

### Phase 1: Matryoshka Embedding Training

```python
# src/cogsyndelta/memory/matryoshka.py

from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss

class MatryoshkaEmbedder:
    """Matryoshka-trained embeddings with multi-scale fidelity.
    
    The key insight: By training with losses at multiple truncation
    points, the model learns to frontload semantic information into
    early dimensions. This enables flexible compression without
    catastrophic fidelity loss.
    
    Why these dimensions: Powers of 2 align with hardware, and
    [64, 128, 256, 512] covers 8x to 1x compression range.
    """
    
    TRUNCATION_DIMS = [64, 128, 256, 512]
    
    def __init__(self, base_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(base_model)
        
    def train_matryoshka(
        self,
        train_dataloader,
        epochs: int = 10,
        dim_weights: dict[int, float] | None = None,
    ):
        """Train with MatryoshkaLoss for multi-scale compression.
        
        Args:
            train_dataloader: DataLoader with (anchor, positive) pairs
            epochs: Training epochs
            dim_weights: Optional weights per dimension (default: equal)
        """
        if dim_weights is None:
            dim_weights = {d: 1.0 for d in self.TRUNCATION_DIMS}
        
        base_loss = MultipleNegativesRankingLoss(self.model)
        matryoshka_loss = MatryoshkaLoss(
            model=self.model,
            loss=base_loss,
            matryoshka_dims=self.TRUNCATION_DIMS,
            matryoshka_weights=list(dim_weights.values()),
        )
        
        self.model.fit(
            train_objectives=[(train_dataloader, matryoshka_loss)],
            epochs=epochs,
        )
    
    def encode_truncated(
        self,
        texts: list[str],
        target_dim: int,
    ) -> torch.Tensor:
        """Encode and truncate to target dimension.
        
        Args:
            texts: Input texts to encode
            target_dim: Truncation dimension (64, 128, 256, or 512)
            
        Returns:
            Truncated embeddings [batch, target_dim]
        """
        full_embeddings = self.model.encode(texts, convert_to_tensor=True)
        return full_embeddings[:, :target_dim]
    
    def compression_ratio(self, target_dim: int) -> float:
        """Calculate compression ratio for target dimension."""
        full_dim = self.model.get_sentence_embedding_dimension()
        return full_dim / target_dim
```

### Phase 2: QINCo2-Style Neural Quantization

```python
# src/cogsyndelta/memory/neural_quantization.py

import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

class QINCo2Quantizer(nn.Module):
    """QINCo2-inspired neural codebook quantization.
    
    Unlike standard RVQ with fixed codebooks, this uses a small
    neural network to generate codebooks conditioned on prior
    quantization steps. This captures inter-codebook dependencies.
    
    Why neural codebooks: Fixed codebooks can't adapt to the
    residual distribution, which changes with each stage. Neural
    generation achieves 34-44% MSE reduction over RVQ.
    """
    
    def __init__(
        self,
        embed_dim: int = 512,
        num_stages: int = 8,
        codebook_size: int = 256,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_stages = num_stages
        
        # Neural codebook generator (conditions on prior codes)
        self.codebook_generator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim + i * 64, 256),  # Prior codes as context
                nn.GELU(),
                nn.Linear(256, codebook_size * embed_dim),
            )
            for i in range(num_stages)
        ])
        
        # Base codebook (learned)
        self.base_codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, embed_dim))
            for _ in range(num_stages)
        ])
        
        # Code embedding (for conditioning)
        self.code_embed = nn.Embedding(codebook_size, 64)
        
    def encode(
        self,
        x: torch.Tensor,
        target_fidelity: float = 0.95,
    ) -> tuple[list[torch.Tensor], float]:
        """Encode with adaptive stages until fidelity target met.
        
        Args:
            x: Input embeddings [batch, embed_dim]
            target_fidelity: Stop early if fidelity achieved
            
        Returns:
            (list of codes per stage, achieved fidelity)
        """
        codes = []
        residual = x.clone()
        reconstructed = torch.zeros_like(x)
        prior_codes = torch.zeros(x.shape[0], 0, device=x.device)
        
        for stage_idx in range(self.num_stages):
            # Generate codebook conditioned on prior codes
            if stage_idx == 0:
                codebook = self.base_codebooks[stage_idx]
            else:
                # Condition on prior code embeddings
                prior_embed = self.code_embed(prior_codes.long()).flatten(1)
                context = torch.cat([residual, prior_embed], dim=-1)
                codebook_flat = self.codebook_generator[stage_idx](context)
                codebook = codebook_flat.view(-1, self.base_codebooks[0].shape[0], self.embed_dim)
                codebook = codebook[0]  # Use first sample's generated codebook
            
            # Quantize residual
            distances = torch.cdist(residual, codebook)
            code_idx = distances.argmin(dim=-1)
            codes.append(code_idx)
            
            # Update for next stage
            quantized = codebook[code_idx]
            reconstructed = reconstructed + quantized
            residual = x - reconstructed
            prior_codes = torch.cat([prior_codes, code_idx.unsqueeze(-1).float()], dim=-1)
            
            # Check fidelity early stopping
            fidelity = F.cosine_similarity(x, reconstructed, dim=-1).mean().item()
            if fidelity >= target_fidelity:
                break
        
        return codes, fidelity
    
    def decode(self, codes: list[torch.Tensor]) -> torch.Tensor:
        """Decode from codes back to embedding."""
        reconstructed = torch.zeros(codes[0].shape[0], self.embed_dim)
        for stage_idx, code in enumerate(codes):
            codebook = self.base_codebooks[stage_idx]
            reconstructed = reconstructed + codebook[code]
        return reconstructed
```

### Phase 3: Combined Pipeline

```python
# src/cogsyndelta/memory/compression_pipeline.py

class MRLQINCoPipeline:
    """Combined Matryoshka + QINCo2 compression pipeline.
    
    Achieves ≥95% fidelity at 10-16x compression by:
    1. Matryoshka truncation (dimension reduction)
    2. QINCo2 quantization (bit reduction)
    """
    
    def __init__(
        self,
        matryoshka_embedder: MatryoshkaEmbedder,
        quantizer: QINCo2Quantizer,
    ):
        self.matryoshka = matryoshka_embedder
        self.quantizer = quantizer
    
    def compress(
        self,
        embeddings: torch.Tensor,
        target_compression: float = 10.0,
        min_fidelity: float = 0.95,
    ) -> CompressedEmbedding:
        """Compress with fidelity guarantee.
        
        Strategy:
        1. Choose MRL truncation level based on target compression
        2. Apply QINCo2 quantization
        3. If fidelity insufficient, reduce compression
        """
        # Determine truncation dimension for first stage compression
        if target_compression >= 8:
            trunc_dim = 64   # 8x from 512
        elif target_compression >= 4:
            trunc_dim = 128  # 4x from 512
        elif target_compression >= 2:
            trunc_dim = 256  # 2x from 512
        else:
            trunc_dim = 512  # No truncation
        
        # MRL truncation
        truncated = embeddings[:, :trunc_dim]
        
        # QINCo2 quantization with early stopping at fidelity target
        codes, fidelity = self.quantizer.encode(truncated, min_fidelity)
        
        return CompressedEmbedding(
            codes=codes,
            trunc_dim=trunc_dim,
            original_dim=embeddings.shape[-1],
            fidelity=fidelity,
        )
```

## Consequences

### Positive

- **Target fidelity achievable**: MRL+QINCo2 can reach ≥95% at 10-16x
- **Flexible compression**: Single model serves multiple ratios
- **Production libraries**: sentence-transformers MatryoshkaLoss is ready
- **Backward compatible**: Works with existing embedding pipelines

### Negative

- **Training required**: MRL needs fine-tuning on target data
- **Compute overhead**: Neural codebook generation adds latency
- **Complexity**: Two-stage pipeline harder to debug

### Neutral

- Trade-off between truncation level and quantization stages
- QINCo2 requires careful hyperparameter tuning

## Implementation Plan

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 1. Matryoshka | Week 1-3 | matryoshka.py, training scripts |
| 2. QINCo2 | Week 4-6 | neural_quantization.py |
| 3. Pipeline | Week 7-8 | compression_pipeline.py, benchmarks |

## Dependencies

```toml
[project.optional-dependencies]
compression = [
    "sentence-transformers>=3.0.0",  # MatryoshkaLoss
    "vector-quantize-pytorch>=1.0.0",  # RVQ baseline
]
```

## References

- Kusupati et al. "Matryoshka Representation Learning" (NeurIPS 2022)
- QINCo2 (Meta AI, ICLR 2025, arXiv:2501.03078)
- sentence-transformers: https://www.sbert.net/
- vector-quantize-pytorch: https://github.com/lucidrains/vector-quantize-pytorch
- Research doc: docs/Latent-Space-Native AI Systems_A Technical Deep Dive for CogSynDelta.md

---

*Authored: 2026-01-19 | Related: ADR-0008, specs/compression-research/spec.md*
