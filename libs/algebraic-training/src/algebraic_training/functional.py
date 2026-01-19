"""
Functional API for Algebraic Training.

Provides convenient one-liner functions for common operations.

Example:
    >>> import algebraic_training.functional as F
    >>> predictions = F.predict_training(model, train_x, train_y, epochs=100)
    >>> analysis = F.analyze_trainability(model, train_x, train_y)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from algebraic_training.fisher import FisherInformationPredictor
from algebraic_training.ntk import NTKPredictor
from algebraic_training.spectral import SpectralWeightPredictor
from algebraic_training.trainer import UnifiedAlgebraicTrainer
from algebraic_training.weight_distribution import WeightDistributionPredictor


def predict_training(
    model: nn.Module,
    train_x: Tensor,
    train_y: Tensor,
    epochs: int = 100,
    learning_rate: float = 0.01,
    apply_weights: bool = False,
) -> dict[str, Any]:
    """
    Predict training outcome without running backpropagation.

    Args:
        model: Neural network to train
        train_x: Training inputs
        train_y: Training targets
        epochs: Number of epochs to simulate
        learning_rate: Learning rate to simulate
        apply_weights: Whether to apply predicted weights to model

    Returns:
        Dictionary with predicted training results

    Example:
        >>> results = predict_training(model, train_x, train_y, epochs=100)
        >>> print(f"Predicted loss: {results['predicted_mse']:.6f}")
    """
    trainer = UnifiedAlgebraicTrainer(model)
    return trainer.train_algebraically(
        train_x, train_y,
        target_epochs=epochs,
        learning_rate=learning_rate,
        apply_weights=apply_weights,
        verbose=False,
    )


def analyze_trainability(
    model: nn.Module,
    train_x: Tensor,
    train_y: Tensor,
) -> dict[str, Any]:
    """
    Analyze how trainable a model is on given data.

    Args:
        model: Neural network to analyze
        train_x: Training inputs
        train_y: Training targets

    Returns:
        Trainability analysis including recommended hyperparameters

    Example:
        >>> analysis = analyze_trainability(model, train_x, train_y)
        >>> print(f"Recommended LR: {analysis['recommended_learning_rate']:.6f}")
    """
    trainer = UnifiedAlgebraicTrainer(model)
    return trainer.analyze_trainability(train_x, train_y)


def natural_gradient_step(
    model: nn.Module,
    train_x: Tensor,
    train_y: Tensor,
    lr: float = 0.1,
    apply_updates: bool = True,
) -> dict[str, Tensor]:
    """
    Perform a single natural gradient update.

    Natural gradient is more efficient than regular gradient - one step
    is equivalent to many regular gradient steps.

    Args:
        model: Neural network to update
        train_x: Training inputs
        train_y: Training targets
        lr: Learning rate
        apply_updates: Whether to apply updates to model

    Returns:
        Dictionary mapping parameter names to their updates

    Example:
        >>> updates = natural_gradient_step(model, train_x, train_y, lr=0.1)
    """
    fisher = FisherInformationPredictor(model)
    updates = fisher.compute_natural_gradient_update(train_x, train_y, learning_rate=lr)

    if apply_updates:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in updates:
                    param.add_(updates[name])

    return updates


def compute_ntk(
    model: nn.Module,
    x1: Tensor,
    x2: Tensor | None = None,
) -> Tensor:
    """
    Compute the Neural Tangent Kernel between inputs.

    The NTK captures how the network output at x1 changes when we
    update weights to reduce loss at x2.

    Args:
        model: Neural network
        x1: First set of inputs [n1, ...]
        x2: Second set of inputs [n2, ...] (defaults to x1)

    Returns:
        NTK matrix [n1, n2]

    Example:
        >>> ntk = compute_ntk(model, train_x)
        >>> print(f"NTK shape: {ntk.shape}")
    """
    ntk_predictor = NTKPredictor(model)
    return ntk_predictor.compute_empirical_ntk(x1, x2)


def compute_fisher(
    model: nn.Module,
    train_x: Tensor,
    train_y: Tensor,
) -> dict[str, Tensor]:
    """
    Compute diagonal Fisher Information Matrix.

    The Fisher captures weight importance for the task.

    Args:
        model: Neural network
        train_x: Training inputs
        train_y: Training targets

    Returns:
        Dictionary mapping parameter names to Fisher diagonal

    Example:
        >>> fisher = compute_fisher(model, train_x, train_y)
        >>> for name, f in fisher.items():
        ...     print(f"{name}: importance={f.mean():.6f}")
    """
    from collections.abc import Iterator  # noqa: TC003, PLC0415

    fisher_predictor = FisherInformationPredictor(model)

    def data_gen() -> Iterator[tuple[Tensor, Tensor]]:
        for i in range(0, len(train_x), 32):
            yield train_x[i:i+32], train_y[i:i+32]

    return fisher_predictor.compute_fisher_matrix(data_gen())


def compute_optimal_weights(
    model: nn.Module,
    train_x: Tensor,
    train_y: Tensor,
    method: str = "spectral",
) -> dict[str, Tensor]:
    """
    Compute optimal weights in closed form.

    Args:
        model: Neural network
        train_x: Training inputs
        train_y: Training targets
        method: "spectral", "ntk", or "natural_gradient"

    Returns:
        Dictionary mapping parameter names to optimal weights

    Example:
        >>> weights = compute_optimal_weights(model, train_x, train_y)
        >>> # Apply to model
        >>> for name, param in model.named_parameters():
        ...     if name in weights:
        ...         param.data.copy_(weights[name])
    """
    if method == "spectral":
        spectral = SpectralWeightPredictor(model)
        return spectral.predict_network_weights(train_x, train_y)
    elif method == "ntk":
        ntk = NTKPredictor(model)
        return ntk.predict_optimal_weights(train_x, train_y)
    else:  # natural_gradient
        fisher = FisherInformationPredictor(model)
        return fisher.compute_natural_gradient_update(train_x, train_y, learning_rate=1.0)


def predict_convergence_time(
    model: nn.Module,
    train_x: Tensor,
    learning_rate: float = 0.01,
) -> dict[str, float]:
    """
    Predict how long training will take to converge.

    Args:
        model: Neural network
        train_x: Training inputs
        learning_rate: Learning rate to use

    Returns:
        Dictionary with convergence estimates

    Example:
        >>> conv = predict_convergence_time(model, train_x, lr=0.01)
        >>> print(f"Estimated epochs: {conv['estimated_epochs']:.0f}")
    """
    predictor = WeightDistributionPredictor(model)
    return predictor.predict_convergence_time(train_x, learning_rate)


def quick_train(
    model: nn.Module,
    train_x: Tensor,
    train_y: Tensor,
    num_steps: int = 5,
) -> dict[str, float]:
    """
    Quick training using natural gradient steps.

    Args:
        model: Neural network to train
        train_x: Training inputs
        train_y: Training targets
        num_steps: Number of natural gradient steps

    Returns:
        Training statistics

    Example:
        >>> stats = quick_train(model, train_x, train_y, num_steps=5)
        >>> print(f"Loss reduction: {stats['loss_reduction']:.2f}x")
    """
    trainer = UnifiedAlgebraicTrainer(model)
    return trainer.quick_optimize(train_x, train_y, num_natural_steps=num_steps)
