# ADR-0016: Model and Submodel Training Roadmap

**Status**: Proposed
**Date**: 2026-01-19
**Decision Makers**: @tzervas
**Technical Story**: Establish training strategy for CogSynDelta components

## Context

CogSynDelta consists of multiple interconnected neural components:
- **PCN-VAE-GAN**: Core predictive coding network
- **VL-JEPA**: Vision-language joint embedding architecture
- **mHC (Moderated HyperConnections)**: Cross-layer information flow
- **Memory Compactors**: Compression/reconstruction networks
- **Interconnect Manager**: Inter-section routing

Currently, these components are untrained or use random initialization. To achieve
production-quality performance, we need a systematic training strategy that:
1. Respects component dependencies
2. Maximizes hardware utilization
3. Enables incremental progress
4. Maintains benchmark-verified quality

### Current State Assessment

| Component | Status | Untrained Fidelity | Training Data Needed |
|-----------|--------|-------------------|---------------------|
| PCN-VAE-GAN | Implemented | ~0.5 reconstruction | MNIST, then custom |
| VL-JEPA | Implemented | N/A (vision-only) | Video/image pairs |
| mHC Gates | Implemented | ~0.5 gating | Generated/synthetic |
| HighFidelityCompactor | Implemented | 1.0 (identity) | None (rule-based) |
| HybridAdaptiveCompactor | Implemented | 0.96 | Embedding pairs |
| ResidualBoostCompactor | Implemented | 0.56 | Embedding datasets |
| DenseEmbeddingEncoder | Implemented | -0.001 | Embedding reconstruction |
| LosslessCompactor | Implemented | ~0.45 | Embedding pairs |

### Prerequisites

Before training can begin:
1. ✅ Benchmark infrastructure (ADR-0007)
2. ✅ Memory-efficient techniques (ADR-0015)
3. ✅ Compression targets defined (ADR-0008)
4. 🔄 Training data pipeline
5. 🔄 Distributed training setup
6. 🔄 Experiment tracking (MLflow/W&B)

## Decision

Implement a phased training strategy with clear dependencies:

### Phase 0: Infrastructure (Blocking - 2 weeks)

**Goal**: Establish training foundations before any model training.

```
[ Infrastructure ]
├── Training data loaders
│   ├── MNIST/CIFAR for initial testing
│   ├── Synthetic embedding generator
│   └── Streaming data pipeline
├── Experiment tracking
│   ├── MLflow integration
│   ├── Metric logging
│   └── Checkpoint management
├── Distributed training
│   ├── DDP wrapper
│   ├── Gradient accumulation
│   └── Mixed precision (AMP)
└── Validation harness
    ├── Benchmark integration
    ├── Regression detection
    └── Early stopping criteria
```

**Acceptance Criteria**:
- [ ] Training loop completes on MNIST without OOM
- [ ] Metrics logged to MLflow
- [ ] Checkpoints saved/restored correctly
- [ ] DDP training on 2+ GPUs (if available)

### Phase 1: Compactor Training (Non-blocking - 3 weeks)

**Goal**: Train compression models to meet ADR-0008 targets.

**Training Order** (based on dependencies):

1. **DenseEmbeddingEncoder** (week 1)
   - Simplest architecture
   - Self-supervised reconstruction loss
   - Target: 0.90 fidelity at 8x compression
   
2. **ResidualBoostCompactor** (week 2)
   - Depends on residual learning
   - Supervised with embedding pairs
   - Target: 0.85 fidelity at 16x compression

3. **LosslessCompactor** (week 3)
   - Most complex (basis learning)
   - Requires pretrained basis initialization
   - Target: 0.95 fidelity at 4x compression

```python
# Training configuration per compactor
COMPACTOR_TRAINING_CONFIG = {
    "DenseEmbeddingEncoder": {
        "epochs": 50,
        "batch_size": 256,
        "lr": 1e-3,
        "loss": "mse + cosine",
        "target_fidelity": 0.90,
        "target_compression": 8.0,
    },
    "ResidualBoostCompactor": {
        "epochs": 100,
        "batch_size": 128,
        "lr": 5e-4,
        "loss": "mse + residual_penalty",
        "target_fidelity": 0.85,
        "target_compression": 16.0,
    },
    "LosslessCompactor": {
        "epochs": 200,
        "batch_size": 64,
        "lr": 1e-4,
        "loss": "mse + basis_orthogonality",
        "target_fidelity": 0.95,
        "target_compression": 4.0,
        "requires_warmup": True,
    },
}
```

**Acceptance Criteria**:
- [ ] Each compactor meets fidelity target on validation set
- [ ] Benchmark regression tests pass
- [ ] Training curves logged and analyzed

### Phase 2: PCN-VAE-GAN Training (Blocking for Phase 3 - 4 weeks)

**Goal**: Train core architecture on reconstruction + generation tasks.

**Curriculum**:

1. **Week 1-2: VAE Pretraining**
   - MNIST reconstruction
   - Focus: Stable latent space
   - Loss: ELBO = reconstruction + β×KL

2. **Week 3: GAN Integration**
   - Add discriminator
   - Adversarial training with gradient penalty
   - Loss: VAE_loss + λ×adversarial_loss

3. **Week 4: PCN Integration**
   - Add predictive coding layers
   - Multi-scale prediction loss
   - Fine-tune end-to-end

```python
class PCNVAEGANTrainer:
    """Phased trainer for PCN-VAE-GAN.
    
    Why curriculum:
        Training all components simultaneously leads to instability.
        VAE provides stable latent space for GAN, PCN adds prediction
        capability on top of stable representations.
    """
    
    def train_vae_phase(self, epochs: int = 50):
        """Phase 1: VAE reconstruction."""
        for epoch in range(epochs):
            for batch in self.dataloader:
                recon, mu, logvar = self.model.vae_forward(batch)
                loss = self.vae_loss(batch, recon, mu, logvar)
                self.optimize(loss)
    
    def train_gan_phase(self, epochs: int = 30):
        """Phase 2: Add adversarial training."""
        for epoch in range(epochs):
            for batch in self.dataloader:
                # Generator step
                recon, mu, logvar = self.model.vae_forward(batch)
                g_loss = self.generator_loss(batch, recon, mu, logvar)
                self.optimize_generator(g_loss)
                
                # Discriminator step
                d_loss = self.discriminator_loss(batch, recon)
                self.optimize_discriminator(d_loss)
    
    def train_pcn_phase(self, epochs: int = 20):
        """Phase 3: Predictive coding integration."""
        for epoch in range(epochs):
            for batch, next_batch in self.sequential_dataloader:
                prediction = self.model.predict(batch)
                pcn_loss = self.prediction_loss(prediction, next_batch)
                full_loss = pcn_loss + 0.1 * self.vae_loss(...)
                self.optimize(full_loss)
```

**Acceptance Criteria**:
- [ ] Reconstruction quality: SSIM > 0.9 on MNIST
- [ ] Latent space: Smooth interpolations
- [ ] Generation: FID < 50 (modest goal for initial training)
- [ ] Prediction: Next-frame MSE < 0.1

### Phase 3: VL-JEPA Training (Non-blocking - 6 weeks)

**Goal**: Train vision-language joint embedding.

**Strategy**: Adapt from V-JEPA (Meta) methodology:
- Self-supervised on video frames
- Predict masked regions in latent space
- No reconstruction (latent prediction only)

**Data Requirements**:
- Video dataset (Kinetics-400 or custom)
- ~100K video clips minimum
- Frame sampling: 16 frames @ 2 FPS

**Acceptance Criteria**:
- [ ] Temporal coherence: Adjacent frame similarity > 0.8
- [ ] Throughput: > 1000 images/sec inference
- [ ] Memory: Fits in 16GB GPU during training

### Phase 4: mHC and Interconnect Training (Depends on Phase 2 - 3 weeks)

**Goal**: Train cross-component communication.

**Strategy**: Meta-learning approach
- Use frozen Phase 2 components as "sections"
- Train mHC gates to modulate information flow
- Optimize for end-to-end task performance

```python
class InterconnectTrainer:
    """Train interconnect on end-to-end tasks.
    
    Why meta-learning:
        mHC gates need to learn WHEN to pass information, not WHAT.
        This requires seeing the effect of gating decisions on
        downstream task performance.
    """
    
    def __init__(self, sections: list[nn.Module], interconnect: nn.Module):
        # Freeze section weights
        for section in sections:
            for param in section.parameters():
                param.requires_grad = False
        
        # Only train interconnect
        self.trainable_params = interconnect.parameters()
    
    def train_step(self, batch):
        # Forward through sections with interconnect
        x = batch
        for i, section in enumerate(self.sections):
            x = section(x)
            if i < len(self.sections) - 1:
                # Apply mHC gating between sections
                x = self.interconnect.gate(x, section_id=i)
        
        # Task loss (e.g., classification, reconstruction)
        loss = self.task_loss(x, batch)
        return loss
```

**Acceptance Criteria**:
- [ ] Gate utilization: 30-70% (not always on/off)
- [ ] Communication overhead: < 10% latency increase
- [ ] Task performance: >= baseline without interconnect

### Phase 5: End-to-End Fine-tuning (Final - 2 weeks)

**Goal**: Joint fine-tuning of all components.

**Strategy**:
- Unfreeze all components
- Low learning rate (1e-5)
- Heavy regularization to prevent forgetting
- Validate on held-out benchmark suite

**Acceptance Criteria**:
- [ ] All component benchmarks still pass
- [ ] End-to-end metrics improve
- [ ] No catastrophic forgetting

## Training Infrastructure

### Hardware Requirements

| Phase | Min GPU | Recommended | Est. Time |
|-------|---------|-------------|-----------|
| Phase 0 | 8GB | 16GB | 2 weeks |
| Phase 1 | 16GB | 24GB | 3 weeks |
| Phase 2 | 16GB | 24GB | 4 weeks |
| Phase 3 | 24GB | 48GB or multi | 6 weeks |
| Phase 4 | 16GB | 24GB | 3 weeks |
| Phase 5 | 24GB | 48GB | 2 weeks |

### Experiment Tracking

```yaml
# mlflow_config.yaml
tracking_uri: "mlruns/"
experiment_naming: "cogsyndelta_{component}_{date}"
artifact_logging:
  - model_checkpoints
  - training_curves
  - validation_samples
  - benchmark_results
```

### Checkpoint Strategy

```python
CHECKPOINT_STRATEGY = {
    "save_every_n_epochs": 5,
    "keep_last_n": 3,
    "save_best": True,
    "best_metric": "val_fidelity",  # or component-specific
    "early_stopping_patience": 10,
}
```

## Rationale

### Why Phased Approach

1. **Manages complexity**: Each phase has clear scope
2. **Enables parallelism**: Non-blocking phases can overlap
3. **Reduces risk**: Catch issues early before full training
4. **Provides checkpoints**: Usable models at each phase

### Why This Order

1. **Infrastructure first**: Can't train without it
2. **Compactors early**: Independent, validates memory efficiency
3. **PCN-VAE-GAN before mHC**: Gates need stable sections to gate
4. **VL-JEPA parallel**: Independent, can proceed with video data
5. **End-to-end last**: Requires all components trained

### Alternatives Considered

#### Option 1: End-to-End from Start

- **Pros**: Potentially better final performance
- **Cons**: Slow iteration, hard to debug, high risk
- **Why Rejected**: Too risky for initial development

#### Option 2: Transfer Learning from Large Models

- **Pros**: Faster convergence, better representations
- **Cons**: License issues, architectural mismatch
- **Why Rejected**: Want to validate our architecture

## Consequences

### Positive

- Clear roadmap for training effort
- Each phase produces usable artifacts
- Enables parallel work streams
- Benchmark-verified quality gates

### Negative

- Extended timeline (~20 weeks total)
- Requires sustained compute resources
- Coordination overhead between phases

### Neutral

- May need to revisit phases if architecture changes
- Training configs will need tuning

## Implementation

### Immediate Next Steps

1. Create `src/cogsyndelta/training/` module structure
2. Implement Phase 0 training loop
3. Set up MLflow experiment tracking
4. Create training data generators

### Spec-Kit Integration

Each phase should have:
- Spec document in `specs/training-phase-N/spec.md`
- Task breakdown in `specs/training-phase-N/tasks.md`
- Feature branch: `feat/training-phase-N`

### Success Metrics

| Phase | Key Metric | Target |
|-------|-----------|--------|
| 0 | Training loop runs | Complete without errors |
| 1 | Compactor fidelity | Per ADR-0008 targets |
| 2 | PCN-VAE-GAN reconstruction | SSIM > 0.9 |
| 3 | VL-JEPA throughput | > 1000 img/sec |
| 4 | mHC gate utilization | 30-70% |
| 5 | End-to-end benchmark | All pass |

## References

- [V-JEPA: Video Joint-Embedding Predictive Architecture](https://ai.meta.com/research/publications/v-jepa/)
- [Training GANs with Limited Data](https://arxiv.org/abs/2006.06676)
- [Curriculum Learning for Neural Networks](https://arxiv.org/abs/1904.03626)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- ADR-0008: Compression Fidelity Recovery
- ADR-0015: Context-Efficient Memory Techniques
- Constitution: Test-First Development

---

*Follows constitution: "All performance claims must be backed by evidence" - 
each phase has measurable acceptance criteria*
