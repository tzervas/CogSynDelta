"""
Algebraic Training: Predict neural network training outcomes without backpropagation.

This library implements mathematical techniques to simulate and predict what
gradient descent would converge to, achieving high-fidelity training simulation
through algebraic and analytical principles rather than iterative optimization.

Core Components:
    - NTKPredictor: Neural Tangent Kernel for training dynamics
    - FisherInformationPredictor: Natural gradient optimization  
    - SpectralWeightPredictor: Eigenvalue-based weight prediction
    - AlgebraicOptimizer: Combined optimization interface
    - UnifiedAlgebraicTrainer: High-level training API
    - WeightDistributionPredictor: Statistical weight prediction

Functional API:
    - predict_training(): Predict training outcome
    - analyze_trainability(): Analyze model trainability
    - natural_gradient_step(): Single natural gradient update
    - compute_ntk(): Compute Neural Tangent Kernel
    - compute_fisher(): Compute Fisher Information Matrix

Example:
    >>> from algebraic_training import UnifiedAlgebraicTrainer
    >>> trainer = UnifiedAlgebraicTrainer(model)
    >>> results = trainer.train_algebraically(train_x, train_y, target_epochs=100)
"""

from __future__ import annotations

# Functional API
from algebraic_training import functional
from algebraic_training.fisher import FisherInformationPredictor
from algebraic_training.ntk import NTKPredictor
from algebraic_training.optimizer import AlgebraicOptimizer
from algebraic_training.spectral import SpectralWeightPredictor
from algebraic_training.trainer import UnifiedAlgebraicTrainer
from algebraic_training.weight_distribution import WeightDistributionPredictor

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "NTKPredictor",
    "FisherInformationPredictor", 
    "SpectralWeightPredictor",
    "AlgebraicOptimizer",
    "UnifiedAlgebraicTrainer",
    "WeightDistributionPredictor",
    # Functional API module
    "functional",
    # Version
    "__version__",
]
