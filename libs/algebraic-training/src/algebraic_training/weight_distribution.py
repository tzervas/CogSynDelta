"""
Weight Distribution Predictor.

Predict the distribution of weights after training using statistical 
mechanics and mean field theory.

Mathematical Foundation:
    In wide networks with MSE loss, weights converge to:
        W ~ N(μ*, Σ*)
    
    where:
        μ* = (XᵀX + λI)⁻¹ XᵀY  (mean = ridge regression solution)
        Σ* = σ² (XᵀX + λI)⁻¹   (covariance = inverse Fisher)

Example:
    >>> predictor = WeightDistributionPredictor(model)
    >>> stats = predictor.predict_weight_statistics(train_x, train_y)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class WeightDistributionPredictor:
    """
    Predict the distribution of weights after training.

    Uses statistical mechanics / mean field theory to predict:
    1. Weight concentration points (cluster centers)
    2. Weight spread (variance)
    3. Outlier weights (will be pruned or become extreme)

    Args:
        model: Neural network to analyze
        regularization: Regularization for numerical stability

    Example:
        >>> predictor = WeightDistributionPredictor(model)
        >>> stats = predictor.predict_weight_statistics(train_x, train_y)
        >>> for name, s in stats.items():
        ...     print(f"{name}: mean={s['predicted_mean'].mean():.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        regularization: float = 1e-4,
    ) -> None:
        """Initialize weight distribution predictor."""
        self.model = model
        self.regularization = regularization

    def predict_weight_statistics(
        self,
        train_x: Tensor,
        train_y: Tensor,
        noise_variance: float = 0.1,
    ) -> dict[str, dict[str, Tensor]]:
        """
        Predict mean and variance of each weight after training.

        Args:
            train_x: Training inputs
            train_y: Training targets
            noise_variance: Assumed noise variance in targets

        Returns:
            Dictionary with predicted statistics per parameter
        """
        predictions: dict[str, dict[str, Tensor]] = {}

        if train_x.dim() > 2:
            train_x = train_x.view(train_x.shape[0], -1)
        if train_y.dim() > 2:
            train_y = train_y.view(train_y.shape[0], -1)

        XtX = train_x.T @ train_x / train_x.shape[0]
        XtY = train_x.T @ train_y / train_x.shape[0]

        reg = self.regularization * torch.eye(XtX.shape[0], device=XtX.device)
        precision = XtX + reg

        mean_weights = torch.linalg.solve(precision, XtY)

        weight_covariance = noise_variance * torch.linalg.inv(precision)
        weight_variance = weight_covariance.diag()

        mean_weights_flat = mean_weights.flatten()
        weight_variance_flat = weight_variance.flatten()

        for name, param in self.model.named_parameters():
            if "weight" in name:
                param_size = param.numel()

                mean_size = mean_weights_flat.shape[0]
                if param_size <= mean_size:
                    pred_mean = mean_weights_flat[:param_size].view(param.shape)
                else:
                    repeats = (param_size // mean_size) + 1
                    pred_mean = mean_weights_flat.repeat(repeats)[:param_size].view(param.shape)

                var_size = weight_variance_flat.shape[0]
                if param_size <= var_size:
                    pred_var = weight_variance_flat[:param_size].view(param.shape)
                else:
                    var_repeats = (param_size // var_size) + 1
                    pred_var = weight_variance_flat.repeat(var_repeats)[:param_size].view(param.shape)

                var_threshold = pred_var.mean() + 2 * pred_var.std()
                outlier_mask = pred_var > var_threshold

                concentration_mask = pred_var < pred_var.mean()

                predictions[name] = {
                    "predicted_mean": pred_mean,
                    "predicted_variance": pred_var,
                    "outlier_mask": outlier_mask,
                    "concentration_mask": concentration_mask,
                    "confidence": 1.0 / (pred_var + self.regularization),
                }

        return predictions

    def predict_convergence_time(
        self,
        train_x: Tensor,
        learning_rate: float = 0.01,
    ) -> dict[str, float]:
        """
        Predict how long training will take to converge.

        Convergence time is determined by the smallest eigenvalue of XᵀX:
            T_converge ≈ 1 / (η × λ_min)

        Args:
            train_x: Training inputs
            learning_rate: Learning rate to simulate

        Returns:
            Dictionary with convergence time estimates
        """
        if train_x.dim() > 2:
            train_x = train_x.view(train_x.shape[0], -1)

        XtX = train_x.T @ train_x / train_x.shape[0]
        eigenvalues = torch.linalg.eigvalsh(XtX)

        eigenvalues = eigenvalues[eigenvalues > self.regularization]

        lambda_min = eigenvalues.min().item()
        lambda_max = eigenvalues.max().item()

        slowest_mode_time = 1.0 / (learning_rate * lambda_min)
        fastest_mode_time = 1.0 / (learning_rate * lambda_max)

        condition_number = lambda_max / lambda_min

        return {
            "estimated_epochs": slowest_mode_time,
            "fastest_convergence": fastest_mode_time,
            "condition_number": condition_number,
            "eigenvalue_min": lambda_min,
            "eigenvalue_max": lambda_max,
            "recommended_learning_rate": 2.0 / (lambda_min + lambda_max),
        }

    def generate_predicted_weights(
        self,
        train_x: Tensor,
        train_y: Tensor,
        sample_from_distribution: bool = False,
    ) -> dict[str, Tensor]:
        """
        Generate predicted final weights.

        Args:
            train_x: Training inputs
            train_y: Training targets
            sample_from_distribution: If True, sample from predicted distribution.
                                     If False, return mean (MAP estimate).

        Returns:
            Dictionary mapping parameter names to predicted weights
        """
        stats = self.predict_weight_statistics(train_x, train_y)

        predicted_weights = {}
        for name, param_stats in stats.items():
            mean = param_stats["predicted_mean"]
            var = param_stats["predicted_variance"]

            if sample_from_distribution:
                noise = torch.randn_like(mean)
                predicted_weights[name] = mean + noise * var.sqrt()
            else:
                predicted_weights[name] = mean

        return predicted_weights
