"""
Algebraic Optimizer - Combined optimization interface.

Combines NTK, Fisher, and Spectral methods for comprehensive
algebraic (non-backprop) optimization.

Example:
    >>> optimizer = AlgebraicOptimizer(model)
    >>> predictions = optimizer.predict_training_outcome(train_x, train_y)
    >>> optimal_weights = optimizer.compute_optimal_weights(train_x, train_y)
    >>> optimizer.apply_weights(model, optimal_weights)
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from algebraic_training.fisher import FisherInformationPredictor
from algebraic_training.ntk import NTKPredictor
from algebraic_training.spectral import SpectralWeightPredictor


class AlgebraicOptimizer:
    """
    Main interface for algebraic (non-backprop) optimization.

    This class combines NTK, Fisher, and Spectral methods to:
    1. Predict training outcomes without running training
    2. Compute optimal weights in closed form
    3. Simulate training dynamics algebraically

    When to Use:
        - Small to medium datasets (< 10K samples)
        - Wide networks (width >> depth)
        - When you need guaranteed convergence
        - For meta-learning / architecture search
        - When compute is limited but memory is available

    Args:
        model: Neural network to optimize
        method: Optimization method ("ntk", "fisher", "spectral", "hybrid")
        device: Computation device

    Example:
        >>> optimizer = AlgebraicOptimizer(model)
        >>> predictions = optimizer.predict_training_outcome(train_x, train_y)
        >>> optimal_weights = optimizer.compute_optimal_weights(train_x, train_y)
        >>> optimizer.apply_weights(model, optimal_weights)
    """

    def __init__(
        self,
        model: nn.Module,
        method: Literal["ntk", "fisher", "spectral", "hybrid"] = "hybrid",
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize algebraic optimizer."""
        self.model = model
        self.method = method
        self.device = torch.device(device)

        self.ntk_predictor = NTKPredictor(model)
        self.fisher_predictor = FisherInformationPredictor(model)
        self.spectral_predictor = SpectralWeightPredictor(model)

    def predict_training_outcome(
        self,
        train_x: Tensor,
        train_y: Tensor,
        learning_rate: float = 0.01,
        num_epochs: int = 100,
    ) -> dict[str, Any]:
        """
        Predict what the model will learn without actually training.

        Returns comprehensive predictions including:
        - Final model outputs on training data
        - Weight distributions (mean, variance, outliers)
        - Convergence timeline
        - Expected final loss

        Args:
            train_x: Training inputs
            train_y: Training targets
            learning_rate: Simulated learning rate
            num_epochs: Simulated number of epochs

        Returns:
            Dictionary with prediction results
        """
        results: dict[str, Any] = {}

        if self.method in ("ntk", "hybrid"):
            ntk_pred = self.ntk_predictor.predict_training_dynamics(
                train_x, train_y, train_x,
                learning_rate=learning_rate,
                training_time=float(num_epochs),
            )
            results["ntk_predictions"] = ntk_pred
            results["predicted_train_outputs"] = ntk_pred["predicted_train_outputs"]
            results["predicted_mse"] = ntk_pred["predicted_mse"]
            results["convergence_rates"] = ntk_pred["convergence_eigenvalues"]

        if self.method in ("fisher", "hybrid"):
            fisher_pred = self.fisher_predictor.predict_weight_distribution(
                train_x, train_y, num_epochs=num_epochs
            )
            results["weight_distributions"] = fisher_pred

            outlier_params = []
            concentrated_params = []
            for name, pred in fisher_pred.items():
                if pred["outlier_mask"].any():
                    outlier_params.append(name)
                if pred["concentration_mask"].sum() > pred["concentration_mask"].numel() * 0.5:
                    concentrated_params.append(name)

            results["outlier_parameters"] = outlier_params
            results["concentrated_parameters"] = concentrated_params

        if self.method in ("spectral", "hybrid"):
            spectrum = self.spectral_predictor.analyze_data_spectrum(train_x)
            results["data_spectrum"] = spectrum
            results["effective_dimensionality"] = spectrum["effective_dimensionality"].item()

        return results

    def compute_optimal_weights(
        self,
        train_x: Tensor,
        train_y: Tensor,
        method: Literal["ntk", "spectral", "natural_gradient"] = "spectral",
    ) -> dict[str, Tensor]:
        """
        Compute optimal weights without iterative training.

        This returns weights that approximate what gradient descent
        would converge to, computed in closed form!

        Args:
            train_x: Training inputs
            train_y: Training targets
            method: Method to use for weight computation

        Returns:
            Dictionary mapping parameter names to optimal weights
        """
        if method == "ntk":
            return self.ntk_predictor.predict_optimal_weights(train_x, train_y)
        if method == "spectral":
            return self.spectral_predictor.predict_network_weights(train_x, train_y)
        # natural_gradient
        return self.fisher_predictor.compute_natural_gradient_update(
            train_x, train_y, learning_rate=1.0
        )

    def apply_weights(
        self,
        model: nn.Module,
        weights: dict[str, Tensor],
        blend_factor: float = 1.0,
    ) -> None:
        """
        Apply predicted weights to model.

        Args:
            model: Model to update
            weights: Predicted optimal weights
            blend_factor: Blend with current weights (1.0 = full replacement)
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in weights:
                    if blend_factor >= 1.0:
                        param.copy_(weights[name])
                    else:
                        param.copy_(
                            blend_factor * weights[name] +
                            (1 - blend_factor) * param.data
                        )

    def fast_train(
        self,
        train_x: Tensor,
        train_y: Tensor,
        num_natural_steps: int = 10,
    ) -> dict[str, float | list[float]]:
        """
        Ultra-fast training using natural gradient steps.

        One natural gradient step ≈ many regular gradient steps.
        This achieves rapid convergence with minimal computation.

        Args:
            train_x: Training inputs
            train_y: Training targets
            num_natural_steps: Number of natural gradient steps

        Returns:
            Training statistics including loss trajectory
        """
        losses = []

        for step in range(num_natural_steps):
            with torch.no_grad():
                outputs = self.model(train_x)
                if outputs.shape != train_y.shape:
                    outputs = outputs.view_as(train_y)
                loss = F.mse_loss(outputs, train_y).item()
                losses.append(loss)

            updates = self.fisher_predictor.compute_natural_gradient_update(
                train_x, train_y, learning_rate=0.1 / (step + 1)
            )

            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in updates:
                        param.add_(updates[name])

        with torch.no_grad():
            outputs = self.model(train_x)
            if outputs.shape != train_y.shape:
                outputs = outputs.view_as(train_y)
            final_loss = F.mse_loss(outputs, train_y).item()

        return {
            "initial_loss": losses[0] if losses else float("inf"),
            "final_loss": final_loss,
            "loss_trajectory": list(losses),
            "improvement_ratio": losses[0] / final_loss if final_loss > 0 else float("inf"),
        }
