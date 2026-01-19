# ADR-0009: Algebraic Training Optimization

**Status**: Accepted
**Date**: 2026-01-19
**Decision Makers**: @tzervas
**Technical Story**: Training optimization for CogSynDelta submodels

## Context

CogSynDelta's architecture includes multiple specialized submodels (PCN, VAE, GAN, VL-JEPA) connected through an interconnect system with Moderated HyperConnections (mHC). Training these models traditionally requires extensive backpropagation, which is:

1. **Computationally expensive**: O(epochs × batches × parameters) per training run
2. **Iterative**: No theoretical guarantees on convergence speed
3. **Non-transferable**: Each model trains independently, ignoring interconnect relationships
4. **Opaque**: No insight into where weights will converge

We needed a way to:
- Predict training outcomes without full backpropagation
- Optimize interconnect pathways algebraically
- Tune mHC gates without gradient descent
- Identify weight distribution patterns (concentration points, outliers)

## Decision

We will implement an **Algebraic Training** system that uses mathematical closed-form solutions to predict and simulate training outcomes. The system combines four theoretical foundations:

1. **Neural Tangent Kernel (NTK)**: Closed-form training dynamics prediction
2. **Fisher Information Matrix (FIM)**: Natural gradient for optimal convergence
3. **Spectral Analysis**: Eigenvalue-based weight prediction via pseudo-inverse
4. **Mean Field Theory**: Statistical weight distribution prediction

The implementation provides:

```python
from cogsyndelta.optimization import UnifiedAlgebraicTrainer

trainer = UnifiedAlgebraicTrainer(model)
results = trainer.train_algebraically(train_x, train_y, target_epochs=100)
```

## Rationale

### Why This Approach

**Mathematical Soundness**: These techniques are grounded in proven mathematical theory:
- NTK provides exact solutions in the infinite-width limit
- Fisher Information captures true loss landscape geometry
- Spectral methods give optimal linear solutions

**Computational Efficiency**:
- Traditional backprop: O(epochs × batches × parameters)
- Algebraic approach: O(parameters²) to O(parameters³) one-time
- For n < 10,000 samples, algebraic methods are often faster

**Integration Benefits**:
- Naturally handles cross-model weight transfer via interconnect
- mHC gate optimization becomes a linear algebra problem
- Pathway strength prediction uses correlation analysis

### Alternatives Considered

#### Option 1: Meta-Learning / MAML

- **Pros**: Learn to optimize efficiently, adaptive
- **Cons**: Still requires gradient computation, additional training overhead
- **Why Rejected**: Adds complexity without theoretical guarantees

#### Option 2: Hyperparameter Optimization (NAS, HPO)

- **Pros**: Can find good configurations automatically
- **Cons**: Optimizes architecture, not weights; expensive search
- **Why Rejected**: Orthogonal problem - doesn't predict weight values

#### Option 3: Knowledge Distillation

- **Pros**: Transfer learning from larger models
- **Cons**: Requires pre-trained teacher, doesn't predict convergence
- **Why Rejected**: Complementary technique, not a replacement for training

## Consequences

### Positive

- **Faster iteration**: Skip lengthy training runs during development
- **Theoretical insight**: Understand where weights converge and why
- **Better initialization**: Use predicted weights as optimal starting points
- **mHC optimization**: Analytically compute gate values
- **Pathway optimization**: Predict optimal interconnect routing

### Negative

- **Approximation quality**: Results are approximations, especially for non-linear deep networks
- **Mitigation**: Use as initialization + fine-tuning, not replacement
- **Memory overhead**: Some methods (NTK) require O(n²) memory for kernel matrices
- **Mitigation**: Use diagonal approximations, batch processing

### Neutral

- Introduces new testing requirements for mathematical correctness
- Requires understanding of linear algebra for maintenance

## Implementation

### Module Structure

```
src/cogsyndelta/optimization/algebraic_training.py
├── NTKPredictor           # Neural Tangent Kernel prediction
├── FisherInformationPredictor  # Natural gradient optimization
├── SpectralWeightPredictor     # Eigenvalue-based weight prediction
├── AlgebraicOptimizer          # Main interface combining methods
├── InterconnectAlgebraicOptimizer  # Cross-model optimization
├── MHCAlgebraicOptimizer       # Gate parameter optimization
├── PathwayStrengthPredictor    # Interconnect pathway optimization
├── WeightDistributionPredictor # Statistical weight prediction
└── UnifiedAlgebraicTrainer     # High-level training interface
```

### Key Formulas

**NTK Training Dynamics**:
$$f(x, t) = f(x, 0) - Θ(x, X)K^{-1}(I - e^{-ηKt})(f(X, 0) - Y)$$

**Fisher Natural Gradient**:
$$Δθ = F^{-1}∇L$$

**Spectral Optimal Weights**:
$$W^* = (X^TX + λI)^{-1}X^TY$$

**mHC Gate Optimization**:
$$g^* = \frac{desired - (1-α) \cdot target}{α \cdot transform(source)}$$

### Usage Example

```python
from cogsyndelta.optimization import (
    UnifiedAlgebraicTrainer,
    MHCAlgebraicOptimizer,
    PathwayStrengthPredictor,
)

# Full algebraic training
trainer = UnifiedAlgebraicTrainer(model)
results = trainer.train_algebraically(train_x, train_y)

# Analyze trainability
analysis = trainer.analyze_trainability(train_x, train_y)
print(f"Trainability score: {analysis['trainability_score']:.3f}")
print(f"Recommended LR: {analysis['recommended_learning_rate']:.6f}")

# mHC gate optimization
mhc_opt = MHCAlgebraicOptimizer(embed_dim=256)
optimal_gates = mhc_opt.compute_optimal_gate_values(source, target, desired)

# Pathway strength prediction
pathway_pred = PathwayStrengthPredictor()
strengths = pathway_pred.predict_all_pathway_strengths(section_states)
```

## Validation

The implementation includes comprehensive tests validating:

1. **Spectral vs Backprop**: For linear regression, algebraic prediction achieves results within 2x of 100 epochs of SGD (and is actually optimal for linear cases)
2. **Natural Gradient Stability**: Gradient clipping prevents explosion
3. **mHC Gate Bounds**: Gates stay in valid sigmoid range [0.01, 0.99]
4. **Pathway Predictions**: All strengths non-negative, respects bandwidth constraints

## References

- Jacot, A., Gabriel, F., & Hongler, C. (2018). "Neural Tangent Kernel: Convergence and Generalization in Neural Networks"
- Martens, J. (2020). "New Insights and Perspectives on the Natural Gradient Method"
- Pennington, J., & Worah, P. (2017). "Nonlinear Random Matrix Theory for Deep Learning"
- Mei, S., Montanari, A., & Nguyen, P. M. (2018). "A Mean Field View of the Landscape of Two-Layer Neural Networks"
