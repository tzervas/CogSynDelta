"""
Neural Tangent Kernel (NTK) Predictor.

Predict training dynamics using Neural Tangent Kernel theory. In the
infinite-width (or sufficiently wide) limit, neural network training
becomes a LINEAR dynamics problem with closed-form solution.

Mathematical Foundation:
    In the NTK regime, training dynamics are:

    f(x, t) = f(x, 0) + Θ(x, X_train) @ K⁻¹ @ (1 - e^{-ηKt}) @ (Y - f(X, 0))

    where:
        - Θ(x, X) is the NTK between test point x and training data X
        - K = Θ(X, X) is the NTK Gram matrix on training data
        - η is learning rate
        - t is training time (epochs)

Example:
    >>> ntk = NTKPredictor(model)
    >>> predictions = ntk.predict_training_dynamics(
    ...     train_x, train_y, test_x,
    ...     learning_rate=0.01, training_time=100.0
    ... )
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NTKPredictor(nn.Module):
    """
    Predict training dynamics using Neural Tangent Kernel theory.

    In the infinite-width (or sufficiently wide) limit, neural network
    training becomes a LINEAR dynamics problem with closed-form solution.

    Key insight: Instead of running gradient descent for T steps, we can
    compute the EXACT training trajectory analytically:

        f(x, t) = f(x, 0) + Θ(x, X_train) @ K⁻¹ @ (1 - e^{-ηKt}) @ (Y - f(X, 0))

    Computational Complexity:
        - NTK computation: O(n² × d) where n=samples, d=parameters
        - Matrix exponential: O(n³)
        - Total: O(n³ + n²d) vs O(T × n × d) for gradient descent

    For small datasets (n < 10000), this is MUCH faster than training!

    Args:
        model: Neural network to analyze (should be wide for NTK to be accurate)
        use_empirical_ntk: Use empirical NTK (exact) vs analytical (approximate)
        ntk_batch_size: Batch size for NTK computation (memory vs speed)
        regularization: Tikhonov regularization for numerical stability

    Example:
        >>> ntk = NTKPredictor(model)
        >>> results = ntk.predict_training_dynamics(
        ...     train_x, train_y, test_x,
        ...     learning_rate=0.01, training_time=100.0
        ... )
        >>> print(f"Predicted MSE: {results['predicted_mse']:.6f}")
    """

    def __init__(
        self,
        model: nn.Module,
        use_empirical_ntk: bool = True,
        ntk_batch_size: int = 64,
        regularization: float = 1e-4,
    ) -> None:
        """Initialize NTK predictor."""
        super().__init__()
        self.model = model
        self.use_empirical_ntk = use_empirical_ntk
        self.ntk_batch_size = ntk_batch_size
        self.regularization = regularization

        # Cache for computed NTK matrices
        self._ntk_cache: dict[str, Tensor] = {}

    def compute_empirical_ntk(
        self,
        x1: Tensor,
        x2: Tensor | None = None,
    ) -> Tensor:
        """
        Compute empirical Neural Tangent Kernel between inputs.

        The empirical NTK is defined as:
            Θ(x₁, x₂) = ⟨∇_θ f(x₁), ∇_θ f(x₂)⟩

        This captures how the network output at x₁ changes when we
        update weights to reduce loss at x₂.

        Args:
            x1: First set of inputs [n1, ...]
            x2: Second set of inputs [n2, ...] (defaults to x1)

        Returns:
            NTK matrix [n1, n2] (or [n1, n1] if x2 is None)
        """
        if x2 is None:
            x2 = x1

        n1, n2 = x1.shape[0], x2.shape[0]

        def get_jacobian(x: Tensor) -> Tensor:
            """Compute Jacobian of model output w.r.t. all parameters."""
            self.model.zero_grad()
            output = self.model(x)

            if output.dim() > 2:
                output = output.view(output.shape[0], -1)

            jacobians = []
            for i in range(output.shape[1]):
                grads = torch.autograd.grad(
                    output[:, i].sum(),
                    list(self.model.parameters()),
                    create_graph=False,
                    retain_graph=True,
                )
                flat_grad = torch.cat([g.flatten() for g in grads if g is not None])
                jacobians.append(flat_grad)

            return torch.stack(jacobians, dim=1)

        # Compute Jacobians in batches
        j1_batches = []
        for i in range(0, n1, self.ntk_batch_size):
            batch = x1[i : i + self.ntk_batch_size]
            batch_jacobians = []
            for j in range(batch.shape[0]):
                jac = get_jacobian(batch[j : j + 1])
                batch_jacobians.append(jac)
            j1_batches.append(torch.stack(batch_jacobians))

        j1 = torch.cat(j1_batches, dim=0)

        if x2 is x1:
            j2 = j1
        else:
            j2_batches = []
            for i in range(0, n2, self.ntk_batch_size):
                batch = x2[i : i + self.ntk_batch_size]
                batch_jacobians = []
                for j in range(batch.shape[0]):
                    jac = get_jacobian(batch[j : j + 1])
                    batch_jacobians.append(jac)
                j2_batches.append(torch.stack(batch_jacobians))
            j2 = torch.cat(j2_batches, dim=0)

        j1_flat = j1.view(n1, -1)
        j2_flat = j2.view(n2, -1)

        ntk = torch.mm(j1_flat, j2_flat.T)

        return ntk

    def predict_training_dynamics(
        self,
        train_x: Tensor,
        train_y: Tensor,
        test_x: Tensor,
        learning_rate: float = 0.01,
        training_time: float = 100.0,
    ) -> dict[str, Tensor]:
        """
        Predict what the model outputs will be after training.

        This uses the NTK closed-form solution to predict training
        dynamics WITHOUT actually running gradient descent!

        Mathematical derivation:
            Under gradient flow (continuous-time gradient descent):
            df/dt = -η Θ(x, X) (f(X) - Y)

            Solution: f(x, t) = f(x, 0) - Θ(x, X) K⁻¹ (I - e^{-ηKt}) (f(X, 0) - Y)

        Args:
            train_x: Training inputs [n_train, ...]
            train_y: Training targets [n_train, n_out]
            test_x: Test inputs to predict [n_test, ...]
            learning_rate: Equivalent learning rate
            training_time: Equivalent training epochs

        Returns:
            Dictionary with:
            - predicted_outputs: Predicted model outputs on test_x after training
            - training_dynamics: How outputs evolve during training
            - convergence_eigenvalues: Eigenvalues showing convergence rates
        """
        device = train_x.device

        with torch.no_grad():
            f0_train = self.model(train_x)
            f0_test = self.model(test_x)

        if f0_train.dim() > 2:
            f0_train = f0_train.view(f0_train.shape[0], -1)
            f0_test = f0_test.view(f0_test.shape[0], -1)
            train_y = train_y.view(train_y.shape[0], -1)

        K = self.compute_empirical_ntk(train_x, train_x)
        K_test_train = self.compute_empirical_ntk(test_x, train_x)

        K_reg = K + self.regularization * torch.eye(K.shape[0], device=device)

        eigenvalues, eigenvectors = torch.linalg.eigh(K_reg)

        exp_decay = 1.0 - torch.exp(-learning_rate * eigenvalues * training_time)

        evolution_matrix = eigenvectors @ torch.diag(exp_decay) @ eigenvectors.T

        K_inv = eigenvectors @ torch.diag(1.0 / eigenvalues) @ eigenvectors.T

        residual = f0_train - train_y

        delta_train = -K @ K_inv @ evolution_matrix @ residual
        delta_test = -K_test_train @ K_inv @ evolution_matrix @ residual

        f_final_train = f0_train + delta_train
        f_final_test = f0_test + delta_test

        dynamics = []
        for t in torch.linspace(0, training_time, 10, device=device):
            exp_t = 1.0 - torch.exp(-learning_rate * eigenvalues * t)
            evol_t = eigenvectors @ torch.diag(exp_t) @ eigenvectors.T
            delta_t = -K_test_train @ K_inv @ evol_t @ residual
            dynamics.append(f0_test + delta_t)

        return {
            "predicted_outputs": f_final_test,
            "predicted_train_outputs": f_final_train,
            "initial_outputs": f0_test,
            "training_dynamics": torch.stack(dynamics),
            "convergence_eigenvalues": eigenvalues,
            "predicted_mse": F.mse_loss(f_final_train, train_y).item(),
        }

    def predict_optimal_weights(
        self,
        train_x: Tensor,
        train_y: Tensor,
        current_weights: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        """
        Predict optimal weight configuration using NTK linearization.

        In the NTK regime, the optimal weights can be computed as:
            θ* = θ₀ - J(X)ᵀ K⁻¹ (f(X, θ₀) - Y)

        This gives us the weights we would converge to after infinite training!

        Args:
            train_x: Training inputs
            train_y: Training targets
            current_weights: Current weights (uses model weights if None)

        Returns:
            Dictionary mapping parameter names to predicted optimal weights
        """
        if current_weights is None:
            current_weights = {n: p.data.clone() for n, p in self.model.named_parameters()}

        device = train_x.device

        self.model.zero_grad()
        outputs = self.model(train_x)

        if outputs.dim() > 2:
            outputs = outputs.view(outputs.shape[0], -1)
            train_y = train_y.view(train_y.shape[0], -1)

        K = self.compute_empirical_ntk(train_x, train_x)
        K_reg = K + self.regularization * torch.eye(K.shape[0], device=device)
        K_inv = torch.linalg.inv(K_reg)

        residual = outputs - train_y

        optimal_weights = {}
        for name, param in self.model.named_parameters():
            grad_sum = torch.zeros_like(param)

            for i in range(outputs.shape[1]):
                grads = torch.autograd.grad(
                    outputs[:, i].sum(),
                    param,
                    create_graph=False,
                    retain_graph=True,
                )
                if grads[0] is not None:
                    weight = (K_inv @ residual[:, i]).sum()
                    grad_sum = grad_sum + weight * grads[0]

            optimal_weights[name] = current_weights[name] - grad_sum

        return optimal_weights
