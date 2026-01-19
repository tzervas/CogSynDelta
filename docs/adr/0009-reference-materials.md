# ADR-0009 Reference Materials

**Document Type**: Reference
**Parent ADR**: ADR-0009 (Algebraic Training Optimization)
**Date**: 2026-01-19
**Version**: 1.0

---

## 1. Quick Reference: Key Formulas

### 1.1 Neural Tangent Kernel (NTK)

**Training dynamics** (infinite-width):
$$f(x, t) = f(x, 0) - \Theta(x, X)K^{-1}(I - e^{-\eta K t})(f(X, 0) - Y)$$

**Converged solution** ($t \to \infty$):
$$f^*(x) = f(x, 0) - \Theta(x, X)K^{-1}(f(X, 0) - Y)$$

**Empirical NTK**:
$$\Theta(x, x') = \nabla_\theta f(x;\theta)^\top \nabla_\theta f(x';\theta)$$

### 1.2 Fisher Information Matrix

**Definition**:
$$F_{ij} = \mathbb{E}\left[\frac{\partial \log p(y|x,\theta)}{\partial \theta_i} \frac{\partial \log p(y|x,\theta)}{\partial \theta_j}\right]$$

**Natural gradient update**:
$$\Delta\theta = -F^{-1}\nabla L$$

**Diagonal approximation** (practical):
$$F_{ii} \approx \mathbb{E}[g_i^2]$$

### 1.3 Spectral Methods

**Ridge regression** (closed-form):
$$W^* = (X^\top X + \lambda I)^{-1}X^\top Y$$

**SVD decomposition**:
$$W = U \Sigma V^\top$$

**Marchenko-Pastur** (random matrix eigenvalue density):
$$\rho(\lambda) = \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{2\pi c \lambda}$$

### 1.4 mHC Gate Optimization

**Optimal gate value**:
$$g^* = \frac{desired - (1-\alpha) \cdot target}{\alpha \cdot transform(source)}$$

### 1.5 Quantization

**GPTQ Hessian update**:
$$\delta_F = \frac{w_q - \text{quant}(w_q)}{H^{-1}_{qq}} \times H^{-1}_{:,q}$$

**Fisher-weighted bit allocation**:
$$\text{bits}_i = \text{round}\left(\log_2\left(\frac{|w_i| \cdot F_{ii}}{\sum_j |w_j| \cdot F_{jj}} \cdot B_{total}\right)\right)$$

---

## 2. Verified Benchmarks

### 2.1 NTK vs Backpropagation

| Dataset | NTK Accuracy | Backprop Accuracy | Gap |
|---------|--------------|-------------------|-----|
| MNIST | >98% | >99% | ~1% |
| CIFAR-10 (enhanced CNTK) | 77-89% | 83-90% | 5-10% |

**Source**: Jacot et al. (2018), Lee et al. (2019)

### 2.2 Quantization Methods

| Method | Precision | LLaMA-2 7B Perplexity Increase | Notes |
|--------|-----------|-------------------------------|-------|
| GPTQ | 4-bit | 0.03 (8.34→8.37) | Hessian-based |
| AWQ | 4-bit | ~0.03 | Activation-aware |
| SpinQuant | 4-bit | 2.9 pt gap on tasks | Rotation-based |
| AQLM | 2.76-bit | Viable | Multi-codebook |

**Source**: Frantar et al. (2023), Lin et al. (2023), Meta (2024)

### 2.3 Dataset Composition

| Method | Improvement | Speedup | Compute Overhead |
|--------|-------------|---------|------------------|
| DoReMi | 6.5% downstream | 2.6× faster | 8% (proxy) |
| Data Mixing Laws | Predictable | N/A | Model fitting |

**Source**: Xie et al. (2023), Ye et al. (2024)

### 2.4 Model Merging

| Method | Performance Retention | Notes |
|--------|----------------------|-------|
| Task Arithmetic | 76-85% | Simple weight addition |
| TIES-Merging | 80-89% | Trim + sign elect |
| DARE | 80-95% | Random dropout 90-99% |
| Git Re-Basin | Near-optimal | Permutation alignment |

**Source**: Ilharco et al. (2023), Yadav et al. (2023)

---

## 3. Implementation Reference

### 3.1 CogSynDelta Module Structure

```
src/cogsyndelta/optimization/algebraic_training.py
├── NTKPredictor              # Neural Tangent Kernel
├── FisherInformationPredictor # Natural gradient
├── SpectralWeightPredictor    # Eigenvalue methods
├── AlgebraicOptimizer         # Combined interface
├── InterconnectAlgebraicOptimizer # Cross-model
├── MHCAlgebraicOptimizer      # Gate optimization
├── PathwayStrengthPredictor   # Pathway routing
├── WeightDistributionPredictor # Statistics
└── UnifiedAlgebraicTrainer    # High-level API
```

### 3.2 Usage Patterns

**Basic algebraic training**:
```python
from cogsyndelta.optimization import UnifiedAlgebraicTrainer

trainer = UnifiedAlgebraicTrainer(model)
results = trainer.train_algebraically(train_x, train_y, target_epochs=100)
```

**Trainability analysis**:
```python
analysis = trainer.analyze_trainability(train_x, train_y)
print(f"Score: {analysis['trainability_score']:.3f}")
print(f"Recommended LR: {analysis['recommended_learning_rate']:.6f}")
```

**mHC optimization**:
```python
from cogsyndelta.optimization import MHCAlgebraicOptimizer

mhc_opt = MHCAlgebraicOptimizer(embed_dim=256)
gates = mhc_opt.compute_optimal_gate_values(source, target, desired)
```

**Natural gradient step**:
```python
from cogsyndelta.optimization import FisherInformationPredictor

fisher = FisherInformationPredictor(model)
updates = fisher.compute_natural_gradient_update(train_x, train_y, lr=0.1)
```

---

## 4. Academic Citations

### 4.1 Core Theory

```bibtex
@inproceedings{jacot2018neural,
  title={Neural tangent kernel: Convergence and generalization in neural networks},
  author={Jacot, Arthur and Gabriel, Franck and Hongler, Cl{\'e}ment},
  booktitle={NeurIPS},
  year={2018}
}

@article{amari1998natural,
  title={Natural gradient works efficiently in learning},
  author={Amari, Shun-Ichi},
  journal={Neural computation},
  year={1998}
}

@inproceedings{martens2015optimizing,
  title={Optimizing neural networks with Kronecker-factored approximate curvature},
  author={Martens, James and Grosse, Roger},
  booktitle={ICML},
  year={2015}
}
```

### 4.2 Quantization

```bibtex
@inproceedings{frantar2023gptq,
  title={GPTQ: Accurate post-training quantization for generative pre-trained transformers},
  author={Frantar, Elias and others},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{lin2023awq,
  title={AWQ: Activation-aware weight quantization for LLM compression and acceleration},
  author={Lin, Ji and others},
  booktitle={MLSys},
  year={2024}
}

@article{liu2024spinquant,
  title={SpinQuant: LLM quantization with learned rotations},
  author={Liu, Zechun and others},
  journal={arXiv:2405.16406},
  year={2024}
}
```

### 4.3 Model Composition

```bibtex
@inproceedings{ilharco2023task,
  title={Editing models with task arithmetic},
  author={Ilharco, Gabriel and others},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{yadav2023ties,
  title={TIES-Merging: Resolving interference when merging models},
  author={Yadav, Prateek and others},
  booktitle={NeurIPS},
  year={2023}
}

@inproceedings{xie2023doremi,
  title={DoReMi: Optimizing data mixtures speeds up language model pretraining},
  author={Xie, Sang Michael and others},
  booktitle={NeurIPS},
  year={2023}
}
```

### 4.4 Feature Learning & Scaling

```bibtex
@article{yang2022tensor,
  title={Tensor programs V: Tuning large neural networks via zero-shot hyperparameter transfer},
  author={Yang, Greg and others},
  journal={arXiv:2203.03466},
  year={2022}
}

@article{yang2024umultiplication,
  title={u-μP: The unit-scaled maximal update parametrization},
  author={Yang, Greg and others},
  journal={arXiv:2407.17465},
  year={2024}
}
```

---

## 5. Glossary

| Term | Definition |
|------|------------|
| **NTK** | Neural Tangent Kernel - kernel defined by gradient inner products |
| **FIM** | Fisher Information Matrix - curvature of log-likelihood |
| **STE** | Straight-Through Estimator - gradient approximation for discrete ops |
| **K-FAC** | Kronecker-Factored Approximate Curvature |
| **mHC** | Moderated HyperConnections - gated cross-model connections |
| **μP** | Maximal Update Parameterization - scaling for feature learning |
| **GPTQ** | Generative Pre-Trained Transformer Quantization |
| **AWQ** | Activation-Aware Weight Quantization |
| **TIES** | Trim, Elect Sign, Merge - model merging method |
| **DARE** | Drop And REscale - sparse model merging |
| **DoReMi** | Domain Reweighting with Minimax Optimization |

---

## 6. External Resources

### 6.1 Libraries

| Library | Purpose | Link |
|---------|---------|------|
| Neural Tangents | JAX NTK implementation | github.com/google/neural-tangents |
| K-FAC | TensorFlow Fisher approx | github.com/tensorflow/kfac |
| GPTQ | Post-training quantization | github.com/IST-DASLab/gptq |
| AWQ | Activation-aware quant | github.com/mit-han-lab/llm-awq |
| MUP | Maximal Update Param | github.com/microsoft/mup |

### 6.2 Reference Implementations

| Component | File | Status |
|-----------|------|--------|
| NTK Predictor | `algebraic_training.py:66` | ✅ Implemented |
| Fisher Predictor | `algebraic_training.py:390` | ✅ Implemented |
| Spectral Predictor | `algebraic_training.py:612` | ✅ Implemented |
| mHC Optimizer | `algebraic_training.py:1282` | ✅ Implemented |
| Unified Trainer | `algebraic_training.py:1830` | ✅ Implemented |

---

## 7. Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-19 | Initial consolidated reference |

---

*Reference materials for ADR-0009 Algebraic Training Optimization*
