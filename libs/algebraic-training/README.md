# Algebraic Training

**Predict neural network training outcomes without backpropagation.**

Algebraic Training uses mathematical analysis (Neural Tangent Kernels, Fisher Information, Spectral Methods) to predict what gradient descent would converge to—often in closed form.

## Installation

```bash
pip install algebraic-training
```

Or install from source:

```bash
cd libs/algebraic-training
pip install -e .
```

## Quick Start

```python
import torch
from algebraic_training import UnifiedAlgebraicTrainer

# Your PyTorch model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)

# Training data
train_x = torch.randn(1000, 784)
train_y = torch.randn(1000, 10)

# Algebraic training
trainer = UnifiedAlgebraicTrainer(model)
results = trainer.train_algebraically(train_x, train_y, target_epochs=100)

print(f"Predicted loss: {results['final_loss']:.6f}")
print(f"Epochs simulated: {results['epochs_simulated']}")
```

## Functional API

```python
import algebraic_training.functional as F

# One-liner prediction
predictions = F.predict_training(model, train_x, train_y, epochs=100)

# Trainability analysis
analysis = F.analyze_trainability(model, train_x, train_y)
print(f"Recommended LR: {analysis['recommended_learning_rate']}")

# Natural gradient step
updated_params = F.natural_gradient_step(model, train_x, train_y, lr=0.1)
```

## Core Components

### Neural Tangent Kernel (NTK)

In wide networks, training dynamics become linear and predictable:

```python
from algebraic_training import NTKPredictor

ntk = NTKPredictor(model)
predictions = ntk.compute_training_predictions(
    train_x, train_y, 
    learning_rate=0.01, 
    n_steps=100
)
```

### Fisher Information

Natural gradient optimization with curvature awareness:

```python
from algebraic_training import FisherInformationPredictor

fisher = FisherInformationPredictor(model)
natural_grad = fisher.compute_natural_gradient_update(train_x, train_y, lr=0.1)
```

### Spectral Methods

Eigenvalue-based weight prediction:

```python
from algebraic_training import SpectralWeightPredictor

spectral = SpectralWeightPredictor(model)
for layer_name, optimal_weights in spectral.compute_optimal_weights(train_x, train_y):
    print(f"{layer_name}: {optimal_weights.shape}")
```

## Why Algebraic Training?

| Approach | Complexity | Guarantees |
|----------|------------|------------|
| SGD (traditional) | O(epochs × batches × params) | Empirical convergence |
| Algebraic | O(n³ + n²d) one-time | Theoretical bounds |

For small-to-medium datasets where n² < epochs × batches, algebraic methods are:
- **Faster**: One-shot computation vs. iterative
- **Predictable**: Know the outcome before training
- **Analyzable**: Understand *why* training succeeds or fails

## Mathematical Foundations

1. **Neural Tangent Kernel**: In infinite-width limit, f(x,t) evolves linearly
2. **Fisher Information**: F⁻¹∇L gives optimal descent direction
3. **Spectral Analysis**: Weight matrices converge to predictable eigenvalue distributions
4. **Mean Field Theory**: Wide network weights become Gaussian with predictable statistics

## API Reference

### Classes

| Class | Purpose |
|-------|---------|
| `UnifiedAlgebraicTrainer` | High-level training API |
| `NTKPredictor` | Neural Tangent Kernel analysis |
| `FisherInformationPredictor` | Natural gradient computation |
| `SpectralWeightPredictor` | Eigenvalue-based prediction |
| `AlgebraicOptimizer` | Combined optimization interface |
| `WeightDistributionPredictor` | Statistics prediction |

### Functional API

| Function | Purpose |
|----------|---------|
| `F.predict_training()` | Predict training outcome |
| `F.analyze_trainability()` | Analyze model trainability |
| `F.natural_gradient_step()` | Single natural gradient update |
| `F.compute_ntk()` | Compute Neural Tangent Kernel |
| `F.compute_fisher()` | Compute Fisher Information Matrix |

## License

MIT License - see [LICENSE](../../LICENSE) for details.

## Citation

```bibtex
@software{algebraic_training,
  title = {Algebraic Training: Predict Neural Network Training Without Backprop},
  author = {CogSynDelta Team},
  year = {2026},
  url = {https://github.com/tzervas/CogSynDelta/tree/main/libs/algebraic-training}
}
```

## Related Work

- [Neural Tangent Kernel](https://arxiv.org/abs/1806.07572) - Jacot et al., 2018
- [Natural Gradient](https://www.mitpressjournals.org/doi/abs/10.1162/089976698300017746) - Amari, 1998
- [K-FAC](https://arxiv.org/abs/1503.05671) - Martens & Grosse, 2015
