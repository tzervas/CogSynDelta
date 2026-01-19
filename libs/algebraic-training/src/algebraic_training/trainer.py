"""
Unified Algebraic Trainer.

High-level system for algebraic (non-backprop) training that combines
all predictors for comprehensive optimization.

Example:
    >>> trainer = UnifiedAlgebraicTrainer(model)
    >>> results = trainer.train_algebraically(train_x, train_y, target_epochs=100)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from algebraic_training.fisher import FisherInformationPredictor
from algebraic_training.ntk import NTKPredictor
from algebraic_training.spectral import SpectralWeightPredictor
from algebraic_training.weight_distribution import WeightDistributionPredictor


class UnifiedAlgebraicTrainer:
    """
    Unified system for algebraic (non-backprop) training.

    Combines all predictors for comprehensive algebraic optimization:
    - NTK predictor: Output-level dynamics
    - Fisher predictor: Weight-level importance
    - Spectral predictor: Weight initialization
    - Weight distribution: Statistical weight prediction

    Args:
        model: Main model to train
        device: Computation device

    Example:
        >>> trainer = UnifiedAlgebraicTrainer(model)
        >>> results = trainer.train_algebraically(train_x, train_y)
        >>> print(f"Predicted loss: {results['predicted_mse']:.6f}")
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize unified algebraic trainer."""
        self.model = model
        self.device = torch.device(device)

        self.ntk_predictor = NTKPredictor(model)
        self.fisher_predictor = FisherInformationPredictor(model)
        self.spectral_predictor = SpectralWeightPredictor(model)
        self.weight_predictor = WeightDistributionPredictor(model)

    def train_algebraically(
        self,
        train_x: Tensor,
        train_y: Tensor,
        target_epochs: int = 100,
        learning_rate: float = 0.01,
        apply_weights: bool = True,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Perform full algebraic training.

        This predicts what the model would learn after target_epochs of
        gradient descent, without actually running gradient descent!

        Args:
            train_x: Training inputs
            train_y: Training targets
            target_epochs: Equivalent number of training epochs
            learning_rate: Equivalent learning rate
            apply_weights: Whether to apply predicted weights to model
            verbose: Print progress messages

        Returns:
            Comprehensive training results
        """
        results: dict[str, Any] = {}

        # 1. Predict training dynamics (NTK)
        if verbose:
            print("  [1/4] Predicting training dynamics via NTK...")
        try:
            ntk_results = self.ntk_predictor.predict_training_dynamics(
                train_x, train_y, train_x,
                learning_rate=learning_rate,
                training_time=float(target_epochs),
            )
            results["ntk_dynamics"] = ntk_results
            results["predicted_outputs"] = ntk_results["predicted_train_outputs"]
            results["predicted_mse"] = ntk_results["predicted_mse"]
        except Exception as e:
            if verbose:
                print(f"    NTK prediction failed: {e}")
            results["ntk_error"] = str(e)

        # 2. Predict weight distributions
        if verbose:
            print("  [2/4] Predicting weight distributions...")
        weight_stats = self.weight_predictor.predict_weight_statistics(train_x, train_y)
        results["weight_statistics"] = weight_stats

        # 3. Generate predicted weights (spectral method)
        if verbose:
            print("  [3/4] Computing optimal weights via spectral analysis...")
        optimal_weights = self.spectral_predictor.predict_network_weights(train_x, train_y)
        results["optimal_weights"] = optimal_weights

        # 4. Predict convergence time
        if verbose:
            print("  [4/4] Analyzing convergence properties...")
        convergence = self.weight_predictor.predict_convergence_time(train_x, learning_rate)
        results["convergence_analysis"] = convergence

        # Compute simulated epochs
        results["epochs_simulated"] = target_epochs
        results["final_loss"] = results.get("predicted_mse", float("inf"))

        # Apply weights if requested
        if apply_weights and optimal_weights:
            if verbose:
                print("  Applying predicted weights to model...")
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in optimal_weights:
                        param.copy_(optimal_weights[name])
            results["weights_applied"] = True

        return results

    def quick_optimize(
        self,
        train_x: Tensor,
        train_y: Tensor,
        num_natural_steps: int = 5,
    ) -> dict[str, float]:
        """
        Quick optimization using natural gradient.

        Natural gradient is much more efficient than regular gradient
        descent - one step ≈ many regular steps.

        Args:
            train_x: Training inputs
            train_y: Training targets
            num_natural_steps: Number of natural gradient steps

        Returns:
            Training statistics
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
                train_x, train_y,
                learning_rate=0.5 / (step + 1),
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
            "loss_reduction": losses[0] / final_loss if final_loss > 0 else float("inf"),
            "num_steps": num_natural_steps,
        }

    def analyze_trainability(
        self,
        train_x: Tensor,
        train_y: Tensor,
    ) -> dict[str, Any]:
        """
        Analyze how trainable the model is on this data.

        Returns diagnostics about:
        - Expected training difficulty
        - Predicted convergence time
        - Weight stability predictions
        - Recommended hyperparameters

        Args:
            train_x: Training inputs
            train_y: Training targets

        Returns:
            Trainability analysis results
        """
        convergence = self.weight_predictor.predict_convergence_time(train_x)

        weight_stats = self.weight_predictor.predict_weight_statistics(train_x, train_y)

        total_params = 0
        outlier_params = 0
        concentrated_params = 0

        for _name, stats in weight_stats.items():
            total_params += stats["outlier_mask"].numel()
            outlier_params += int(stats["outlier_mask"].sum().item())
            concentrated_params += int(stats["concentration_mask"].sum().item())

        spectrum = self.spectral_predictor.analyze_data_spectrum(train_x)

        return {
            "convergence": convergence,
            "trainability_score": 1.0 / (convergence["condition_number"] + 1),
            "total_parameters": total_params,
            "outlier_parameters": outlier_params,
            "outlier_ratio": outlier_params / max(total_params, 1),
            "concentrated_parameters": concentrated_params,
            "data_effective_dimensionality": spectrum["effective_dimensionality"].item(),
            "recommended_learning_rate": convergence["recommended_learning_rate"],
            "estimated_epochs_to_converge": convergence["estimated_epochs"],
            "convergence_epochs": int(convergence["estimated_epochs"]),
        }
