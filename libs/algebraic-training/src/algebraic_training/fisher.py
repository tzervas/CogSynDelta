"""
Fisher Information Matrix (FIM) Predictor.

Predict training outcomes using Fisher Information Matrix analysis.
The FIM captures the curvature of the loss landscape and enables
natural gradient optimization.

Mathematical Foundation:
    For a probabilistic model p(y|x, θ), the FIM is:
        F = E_{x,y}[∇log p(y|x,θ) ∇log p(y|x,θ)ᵀ]

    Natural gradient: Δθ = F⁻¹ ∇L

    This is equivalent to gradient descent in the space of probability
    distributions, not weight space - much more efficient!

Example:
    >>> fisher = FisherInformationPredictor(model)
    >>> updates = fisher.compute_natural_gradient_update(train_x, train_y, lr=0.1)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Iterator


class FisherInformationPredictor(nn.Module):
    """
    Predict training outcomes using Fisher Information Matrix analysis.

    The Fisher Information Matrix (FIM) captures the curvature of the
    loss landscape and enables:

    1. Natural Gradient: Optimal weight updates considering geometry
    2. Weight Importance: Which weights matter most for the task
    3. Convergence Prediction: How fast training will converge
    4. Outlier Detection: Identify weights that will become extreme

    Args:
        model: Neural network to analyze
        fisher_samples: Number of samples for FIM estimation
        damping: Damping factor for matrix inversion (larger = more stable)
        block_diagonal: Use block-diagonal approximation (faster)

    Example:
        >>> fisher = FisherInformationPredictor(model)
        >>> updates = fisher.compute_natural_gradient_update(train_x, train_y, lr=0.1)
        >>> for name, update in updates.items():
        ...     model.get_parameter(name).data.add_(update)
    """

    def __init__(
        self,
        model: nn.Module,
        fisher_samples: int = 1000,
        damping: float = 0.1,
        block_diagonal: bool = True,
    ) -> None:
        """Initialize Fisher Information predictor."""
        super().__init__()
        self.model = model
        self.fisher_samples = fisher_samples
        self.damping = damping
        self.block_diagonal = block_diagonal

        self._fisher_cache: dict[str, Tensor] = {}

    def compute_fisher_matrix(
        self,
        data_loader: Iterator[tuple[Tensor, Tensor]],
        num_batches: int | None = None,
    ) -> dict[str, Tensor]:
        """
        Compute (block-diagonal) Fisher Information Matrix.

        For efficiency, we use the block-diagonal approximation where
        each layer's Fisher is computed independently.

        Args:
            data_loader: Iterator yielding (input, target) batches
            num_batches: Maximum number of batches to process

        Returns:
            Dictionary mapping parameter names to their Fisher matrices (diagonal)
        """
        fisher_diag: dict[str, Tensor] = {}
        for name, param in self.model.named_parameters():
            fisher_diag[name] = torch.zeros_like(param)

        n_samples = 0
        for batch_idx, (x, y) in enumerate(data_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break

            self.model.zero_grad()
            logits = self.model(x)

            if logits.dim() == 2 and logits.shape[1] > 1:
                probs = F.softmax(logits, dim=-1)
                sampled_labels = torch.multinomial(probs, 1).squeeze(-1)
                loss = F.cross_entropy(logits, sampled_labels)
            else:
                loss = F.mse_loss(logits, y)

            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_diag[name] = fisher_diag[name] + param.grad.data**2

            n_samples += x.shape[0]

        for name, value in fisher_diag.items():
            fisher_diag[name] = value / n_samples

        self._fisher_cache = fisher_diag
        return fisher_diag

    def predict_weight_distribution(
        self,
        train_x: Tensor,
        train_y: Tensor,
        num_epochs: int = 100,
    ) -> dict[str, dict[str, Tensor]]:
        """
        Predict the distribution of weights after training.

        Using Fisher Information, we can predict:
        1. Mean weight values (where they'll converge)
        2. Weight variances (uncertainty/spread)
        3. Concentration points (clusters in weight space)
        4. Outliers (weights that will become extreme)

        Mathematical basis:
            Under natural gradient with FIM F:
            θ_{t+1} = θ_t - η F⁻¹ ∇L

            The weights converge to a distribution:
            θ* ~ N(θ_MAP, F⁻¹)

        Args:
            train_x: Training inputs
            train_y: Training targets
            num_epochs: Number of epochs to simulate

        Returns:
            Dictionary with predicted weight statistics per parameter
        """

        def data_gen() -> Iterator[tuple[Tensor, Tensor]]:
            for i in range(0, len(train_x), 32):
                yield train_x[i : i + 32], train_y[i : i + 32]

        fisher = self.compute_fisher_matrix(data_gen(), num_batches=len(train_x) // 32)

        self.model.zero_grad()
        outputs = self.model(train_x)
        if outputs.shape != train_y.shape:
            outputs = outputs.view_as(train_y)
        loss = F.mse_loss(outputs, train_y)
        loss.backward()

        predictions: dict[str, dict[str, Tensor]] = {}

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            f_diag = fisher[name] + self.damping
            grad = param.grad.data

            natural_grad = grad / f_diag
            predicted_mean = param.data - num_epochs * 0.01 * natural_grad
            predicted_var = 1.0 / f_diag

            concentration_mask = predicted_var < predicted_var.mean()

            predicted_magnitude = predicted_mean.abs()
            outlier_threshold = predicted_magnitude.mean() + 3 * predicted_magnitude.std()
            outlier_mask = predicted_magnitude > outlier_threshold

            predictions[name] = {
                "predicted_mean": predicted_mean,
                "predicted_variance": predicted_var,
                "concentration_mask": concentration_mask,
                "outlier_mask": outlier_mask,
                "importance_score": f_diag / f_diag.max(),
            }

        return predictions

    def compute_natural_gradient_update(
        self,
        train_x: Tensor,
        train_y: Tensor,
        learning_rate: float = 0.1,
    ) -> dict[str, Tensor]:
        """
        Compute a single natural gradient update.

        Natural gradient is the OPTIMAL update direction considering
        the geometry of the parameter space. It's equivalent to many
        steps of regular gradient descent!

        One natural gradient step ≈ Many regular gradient steps

        Args:
            train_x: Training inputs
            train_y: Training targets
            learning_rate: Learning rate for the update

        Returns:
            Dictionary mapping parameter names to their updates
        """
        if not self._fisher_cache:

            def data_gen() -> Iterator[tuple[Tensor, Tensor]]:
                for i in range(0, len(train_x), 32):
                    yield train_x[i : i + 32], train_y[i : i + 32]

            self.compute_fisher_matrix(data_gen())

        self.model.zero_grad()
        outputs = self.model(train_x)
        if outputs.shape != train_y.shape:
            outputs = outputs.view_as(train_y)
        loss = F.mse_loss(outputs, train_y)
        loss.backward()

        updates = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                f_diag = self._fisher_cache[name] + self.damping
                natural_grad = param.grad.data / f_diag
                grad_norm = natural_grad.norm()
                max_norm = 1.0
                if grad_norm > max_norm:
                    natural_grad = natural_grad * (max_norm / grad_norm)
                updates[name] = -learning_rate * natural_grad

        return updates
