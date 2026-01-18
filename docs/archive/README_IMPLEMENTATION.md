# PCN-VAE-GAN Hybrid Self-Improving AI

A PyTorch implementation of a self-improving AI architecture that combines Predictive Coding Networks (PCN), Variational Autoencoders (VAE), and Generative Adversarial Network (GAN) principles to emulate human cognition for novel problem-solving.

## Architecture Overview

The system implements three phases inspired by human cognitive processes:

### 1. **Exploratory Phase** (Wild Creative Sampling)
- Uses VAE with **configurable σ sampling**: `z = μ + σ * ε`, where `ε ~ N(0,1)`
- Generates `k` diverse samples through high-variance sampling
- Enables creative idea generation and exploration of solution space

### 2. **Culling Phase** (Feasibility Grounding)
- **Bidirectional A* search** for efficient exploration
- **Bayesian inference**: `p(θ|data) ≈ exp(log likelihood + log prior - log Z)`
- Selects feasible outputs based on posterior probabilities
- Grounds creative ideas in practical constraints

### 3. **Meta-Optimization Phase** (Adaptive Learning)
- **MAML (Model-Agnostic Meta-Learning)** gradients on Φ
- Meta-loss: `L_meta = E[L_inner(θ_Φ)]`
- Tunes configurations for validated, beneficial implementations
- Enables rapid adaptation to new tasks

## Mathematical Formulation

### VAE Loss Function
```
L = E[||x - x̂||²] + ½(σ² + μ² - 1 - log σ²)
```

Where:
- **Reconstruction loss**: `E[||x - x̂||²]` (MSE between input and reconstruction)
- **KL divergence**: `½ Σ(σ² + μ² - 1 - log σ²)` (regularization term)
- `μ` = mean of latent distribution
- `σ` = standard deviation (std)
- `σ²` = variance = `exp(logvar)`
- `θ` = model parameters
- `e ≈ 2.718` (exponential base in `exp` operations)

### Reparameterization Trick
```
z = μ + σ * ε, where ε ~ N(0,1)
```

This allows backpropagation through stochastic sampling by treating the randomness as an external input.

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- PyYAML >= 6.0
- NumPy >= 1.24.0

## Configuration

The system uses YAML configuration (`config.yaml`) for all hyperparameters:

```yaml
exploratory:
  k: 10                    # Number of exploratory samples
  sigma_scale: 1.0         # Configurable σ scale for VAE sampling
  latent_dim: 20          # Latent dimension
  hidden_dim: 400         # Hidden layer dimension

culling:
  threshold: 0.5          # Culling threshold
  bidirectional_beam_width: 5
  bayesian_prior_weight: 0.1

meta_optimization:
  inner_lr: 0.01          # Inner loop learning rate
  meta_lr: 0.001          # Meta learning rate
  num_inner_steps: 5      # Inner gradient steps

training:
  batch_size: 128
  epochs: 10
  learning_rate: 0.001

vae_loss:
  reconstruction_weight: 1.0
  kl_weight: 1.0
```

## Usage

### Basic Model Creation

```python
from pcn_vae_gan import create_model
import torch

# Load model from config
model = create_model('config.yaml')

# Forward pass
x = torch.randn(32, 784)  # Batch of 32 MNIST-like images
recon_x, mu, logvar = model(x)

# Compute VAE loss
loss, loss_dict = model.vae_loss(recon_x, x, mu, logvar)
print(f"Total loss: {loss.item():.4f}")
print(f"Reconstruction: {loss_dict['reconstruction'].item():.4f}")
print(f"KL divergence: {loss_dict['kl_divergence'].item():.4f}")
```

### Exploratory Phase

```python
# Generate k diverse samples with configurable σ
k = 10
samples = model.exploratory_phase(x, k=k)
print(f"Generated {k} exploratory samples: {samples.shape}")

# Control exploration variance
model.sigma_scale = 2.0  # Higher variance for more exploration
samples_high_var = model.exploratory_phase(x, k=k)
```

### Culling Phase

```python
# Select best sample using Bayesian inference
best_sample, scores = model.culling_phase(samples, x)
print(f"Best sample selected from {len(samples)} candidates")
print(f"Selection scores: {scores}")
```

### Meta-Optimization

```python
# MAML meta-learning step
x_support = torch.randn(16, 784)  # Support set
x_query = torch.randn(16, 784)    # Query set

loss_dict = model.meta_optimization_step(x_support, x_query)
print(f"Meta loss: {loss_dict['meta_loss'].item():.4f}")
```

## Testing

### Run Unit Tests

```bash
python test_unit.py
```

Tests all components:
- ✓ VAE loss formula validation
- ✓ Exploratory phase sampling
- ✓ Culling phase Bayesian inference
- ✓ Meta-optimization MAML
- ✓ Configuration loading
- ✓ Model components

### Run MNIST Tests (requires internet for dataset download)

```bash
# Full training and testing
python test_mnist.py --epochs 10

# Test only (no training)
python test_mnist.py --test-only

# Custom configuration
python test_mnist.py --config custom_config.yaml --batch-size 64
```

## Project Structure

```
CogSynDelta/
├── config.yaml           # YAML configuration file
├── pcn_vae_gan.py       # Main PCN-VAE-GAN hybrid model
├── test_unit.py         # Unit tests (no dataset required)
├── test_mnist.py        # MNIST training and testing
├── requirements.txt     # Python dependencies
└── README_IMPLEMENTATION.md  # This file
```

## Model Architecture

### Encoder
```
Input (784) → FC (400) → ReLU → [μ (20), log(σ²) (20)]
```

### Decoder
```
z (20) → FC (400) → ReLU → FC (784) → Sigmoid → Output (784)
```

### Total Parameters
652,824 parameters

## Key Features

1. **Configurable Variance**: Control exploration through σ_scale parameter
2. **Bayesian Culling**: Probabilistic selection of best candidates
3. **Meta-Learning**: MAML-based rapid adaptation
4. **YAML Configuration**: Easy hyperparameter tuning
5. **Modular Design**: Separate encoder, decoder, and phase implementations
6. **Mathematical Rigor**: Proper VAE loss with KL divergence

## Implementation Details

### Reparameterization Trick
The model uses the reparameterization trick to enable gradient flow through stochastic sampling:

```python
def reparameterize(self, mu, logvar, sigma_scale=1.0):
    std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log(σ²))
    eps = torch.randn_like(std)     # ε ~ N(0,1)
    return mu + sigma_scale * std * eps  # z = μ + σ * ε
```

### VAE Loss Computation
Combines reconstruction and KL divergence:

```python
def vae_loss(self, recon_x, x, mu, logvar):
    # Reconstruction: E[||x - x̂||²]
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

    # KL: ½ Σ(σ² + μ² - 1 - log σ²)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return recon_loss + kl_div
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cogsynedelta2024,
  title={PCN-VAE-GAN Hybrid Self-Improving AI},
  author={CogSynDelta},
  year={2024},
  url={https://github.com/tzervas/CogSynDelta}
}
```

## License

See LICENSE file for details.

## References

- Kingma & Welling (2013): Auto-Encoding Variational Bayes
- Finn et al. (2017): Model-Agnostic Meta-Learning (MAML)
- Predictive Coding Networks for cognitive modeling
