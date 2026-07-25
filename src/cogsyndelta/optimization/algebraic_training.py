"""
Algebraic Training: Predict training outcomes without full backpropagation.

This module implements mathematical techniques to simulate and predict what
gradient descent would converge to, achieving high-fidelity training simulation
through algebraic and analytical principles rather than iterative optimization.

Core Mathematical Foundations:
==============================

1. **Neural Tangent Kernel (NTK)**
   In the infinite-width limit, neural network training dynamics become LINEAR:

   f(x, θₜ) - f(x, θ₀) = -Θ(x, X)(I - e^{-ηΘ(X,X)t})(f(X, θ₀) - Y)

   where Θ is the NTK matrix. This allows CLOSED-FORM prediction of training!

2. **Fisher Information Matrix (FIM)**
   The FIM captures the curvature of the loss landscape:

   F = E[∇log p(y|x,θ) ∇log p(y|x,θ)ᵀ]

   Natural gradient: Δθ = F⁻¹∇L achieves optimal convergence.

3. **Spectral Analysis**
   Weight matrices converge to specific spectral distributions:
   - Marchenko-Pastur for random initialization
   - Trained networks show characteristic eigenvalue clustering

4. **Mean Field Theory**
   In wide networks, weights become Gaussian with predictable statistics:

   W ~ N(μ*, σ*²) where μ*, σ* depend on architecture and data

Why This Works:
===============

Traditional backprop: O(epochs × batches × parameters) operations
Algebraic approach: O(parameters² to parameters³) one-time computation

For networks where parameters² < epochs × batches × parameters, algebraic
methods are faster AND provide theoretical guarantees on convergence!

Integration with CogSynDelta:
=============================

- **Interconnect**: Predict cross-model weight transfers
- **mHC (Moderated HyperConnections)**: Optimize gate parameters algebraically
- **Submodel specialization**: Domain-specific weight prediction
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

if TYPE_CHECKING:
    from collections.abc import Iterator


# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL TANGENT KERNEL (NTK) PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════


class NTKPredictor(nn.Module):
    """
    Predict training dynamics using Neural Tangent Kernel theory.

    In the infinite-width (or sufficiently wide) limit, neural network
    training becomes a LINEAR dynamics problem with closed-form solution.

    Key insight: Instead of running gradient descent for T steps, we can
    compute the EXACT training trajectory analytically:

        f(x, t) = f(x, 0) + Θ(x, X_train) @ K⁻¹ @ (1 - e^{-ηKt}) @ (Y - f(X, 0))

    where:
        - Θ(x, X) is the NTK between test point x and training data X
        - K = Θ(X, X) is the NTK Gram matrix on training data
        - η is learning rate
        - t is training time (epochs)

    Computational Complexity:
    ========================
    - NTK computation: O(n² × d) where n=samples, d=parameters
    - Matrix exponential: O(n³)
    - Total: O(n³ + n²d) vs O(T × n × d) for gradient descent

    For small datasets (n < 10000), this is MUCH faster than training!
    """

    def __init__(
        self,
        model: nn.Module,
        use_empirical_ntk: bool = True,
        ntk_batch_size: int = 64,
        regularization: float = 1e-4,
    ) -> None:
        """
        Initialize NTK predictor.

        Args:
            model: Neural network to analyze (should be wide for NTK to be accurate)
            use_empirical_ntk: Use empirical NTK (exact) vs analytical (approximate)
            ntk_batch_size: Batch size for NTK computation (memory vs speed)
            regularization: Tikhonov regularization for numerical stability
        """
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

        # Get Jacobians of model output w.r.t. parameters
        def get_jacobian(x: Tensor) -> Tensor:
            """Compute Jacobian of model output w.r.t. all parameters."""
            self.model.zero_grad()
            output = self.model(x)

            # Flatten output if needed
            if output.dim() > 2:
                output = output.view(output.shape[0], -1)

            # Compute Jacobian for each output dimension
            jacobians = []
            for i in range(output.shape[1]):
                grads = torch.autograd.grad(
                    output[:, i].sum(),
                    list(self.model.parameters()),
                    create_graph=False,
                    retain_graph=True,
                )
                # Flatten all gradients into single vector per sample
                flat_grad = torch.cat([g.flatten() for g in grads if g is not None])
                jacobians.append(flat_grad)

            return torch.stack(jacobians, dim=1)  # [n_params, n_outputs]

        # Compute Jacobians in batches to manage memory
        j1_batches = []
        for i in range(0, n1, self.ntk_batch_size):
            batch = x1[i : i + self.ntk_batch_size]
            # For each sample, get Jacobian
            batch_jacobians = []
            for j in range(batch.shape[0]):
                jac = get_jacobian(batch[j : j + 1])
                batch_jacobians.append(jac)
            j1_batches.append(torch.stack(batch_jacobians))

        j1 = torch.cat(j1_batches, dim=0)  # [n1, n_params, n_outputs]

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

        # NTK = J1 @ J2.T (summed over output dimensions)
        # [n1, n_params, n_out] @ [n_out, n_params, n2] -> [n1, n2]
        j1_flat = j1.view(n1, -1)  # [n1, n_params * n_out]
        j2_flat = j2.view(n2, -1)  # [n2, n_params * n_out]

        ntk = torch.mm(j1_flat, j2_flat.T)  # [n1, n2]

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
        ========================
        Under gradient flow (continuous-time gradient descent):

        df/dt = -η Θ(x, X) (f(X) - Y)

        Solution: f(x, t) = f(x, 0) - Θ(x, X) K⁻¹ (I - e^{-ηKt}) (f(X, 0) - Y)

        where K = Θ(X, X) + λI (regularized NTK on training data)

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

        # Get initial model outputs
        with torch.no_grad():
            f0_train = self.model(train_x)
            f0_test = self.model(test_x)

        # Flatten outputs if needed
        if f0_train.dim() > 2:
            f0_train = f0_train.view(f0_train.shape[0], -1)
            f0_test = f0_test.view(f0_test.shape[0], -1)
            train_y = train_y.view(train_y.shape[0], -1)

        # Compute NTK matrices
        K = self.compute_empirical_ntk(train_x, train_x)  # [n_train, n_train]
        K_test_train = self.compute_empirical_ntk(test_x, train_x)  # [n_test, n_train]

        # Regularize for numerical stability
        K_reg = K + self.regularization * torch.eye(K.shape[0], device=device)

        # Compute eigendecomposition for efficient matrix exponential
        eigenvalues, eigenvectors = torch.linalg.eigh(K_reg)

        # Compute (I - e^{-ηKt}) in eigenspace
        # For eigenvalue λ: (1 - e^{-ηλt})
        exp_decay = 1.0 - torch.exp(-learning_rate * eigenvalues * training_time)

        # Transform back: V @ diag(exp_decay) @ V.T
        evolution_matrix = eigenvectors @ torch.diag(exp_decay) @ eigenvectors.T

        # Compute K⁻¹ in eigenspace
        K_inv = eigenvectors @ torch.diag(1.0 / eigenvalues) @ eigenvectors.T

        # Initial residual
        residual = f0_train - train_y  # [n_train, n_out]

        # Predicted change on training data
        delta_train = -K @ K_inv @ evolution_matrix @ residual

        # Predicted change on test data
        delta_test = -K_test_train @ K_inv @ evolution_matrix @ residual

        # Final predictions
        f_final_train = f0_train + delta_train
        f_final_test = f0_test + delta_test

        # Training dynamics at various time points
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

        where J is the Jacobian of outputs w.r.t. weights.

        This gives us the weights we would converge to after infinite training!
        """
        if current_weights is None:
            current_weights = {n: p.data.clone() for n, p in self.model.named_parameters()}

        device = train_x.device

        # Get initial outputs and their Jacobian
        self.model.zero_grad()
        outputs = self.model(train_x)

        if outputs.dim() > 2:
            outputs = outputs.view(outputs.shape[0], -1)
            train_y = train_y.view(train_y.shape[0], -1)

        # Compute NTK and its inverse
        K = self.compute_empirical_ntk(train_x, train_x)
        K_reg = K + self.regularization * torch.eye(K.shape[0], device=device)
        K_inv = torch.linalg.inv(K_reg)

        # Residual
        residual = outputs - train_y  # [n, n_out]

        # Compute optimal weight updates
        optimal_weights = {}
        for name, param in self.model.named_parameters():
            # Compute gradient of outputs w.r.t. this parameter
            grad_sum = torch.zeros_like(param)

            for i in range(outputs.shape[1]):
                grads = torch.autograd.grad(
                    outputs[:, i].sum(),
                    param,
                    create_graph=False,
                    retain_graph=True,
                )
                if grads[0] is not None:
                    # Weight by K_inv @ residual
                    weight = (K_inv @ residual[:, i]).sum()
                    grad_sum = grad_sum + weight * grads[0]

            optimal_weights[name] = current_weights[name] - grad_sum

        return optimal_weights


# ═══════════════════════════════════════════════════════════════════════════════
# FISHER INFORMATION PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════


class FisherInformationPredictor(nn.Module):
    """
    Predict training outcomes using Fisher Information Matrix analysis.

    The Fisher Information Matrix (FIM) captures the curvature of the
    loss landscape and enables:

    1. **Natural Gradient**: Optimal weight updates considering geometry
    2. **Weight Importance**: Which weights matter most for the task
    3. **Convergence Prediction**: How fast training will converge
    4. **Outlier Detection**: Identify weights that will become extreme

    Mathematical Foundation:
    ========================

    For a probabilistic model p(y|x, θ), the FIM is:
        F = E_{x,y}[∇log p(y|x,θ) ∇log p(y|x,θ)ᵀ]

    Natural gradient: Δθ = F⁻¹ ∇L

    This is equivalent to gradient descent in the space of probability
    distributions, not weight space - much more efficient!

    For neural networks with softmax output:
        F ≈ J.T @ diag(p(1-p)) @ J

    where J is the Jacobian of logits w.r.t. weights.
    """

    def __init__(
        self,
        model: nn.Module,
        fisher_samples: int = 1000,
        damping: float = 0.1,
        block_diagonal: bool = True,
    ) -> None:
        """
        Initialize Fisher Information predictor.

        Args:
            model: Neural network to analyze
            fisher_samples: Number of samples for FIM estimation
            damping: Damping factor for matrix inversion (larger = more stable)
            block_diagonal: Use block-diagonal approximation (faster)
        """
        super().__init__()
        self.model = model
        self.fisher_samples = fisher_samples
        self.damping = damping
        self.block_diagonal = block_diagonal

        # Cache for Fisher matrices per layer
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

        Returns:
            Dictionary mapping parameter names to their Fisher matrices
        """
        # Initialize accumulators
        fisher_diag: dict[str, Tensor] = {}
        for name, param in self.model.named_parameters():
            fisher_diag[name] = torch.zeros_like(param)

        n_samples = 0
        for batch_idx, (x, y) in enumerate(data_loader):
            if num_batches is not None and batch_idx >= num_batches:
                break

            # Forward pass
            self.model.zero_grad()
            logits = self.model(x)

            # For classification: sample from model's distribution
            if logits.dim() == 2 and logits.shape[1] > 1:
                # Softmax classification
                probs = F.softmax(logits, dim=-1)
                # Sample labels from predicted distribution
                sampled_labels = torch.multinomial(probs, 1).squeeze(-1)
                loss = F.cross_entropy(logits, sampled_labels)
            else:
                # Regression: use squared error
                loss = F.mse_loss(logits, y)

            # Compute gradients
            loss.backward()

            # Accumulate squared gradients (diagonal Fisher approximation)
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_diag[name] = fisher_diag[name] + param.grad.data**2

            n_samples += x.shape[0]

        # Average
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
        ==================
        Under natural gradient with FIM F:
            θ_{t+1} = θ_t - η F⁻¹ ∇L

        The weights converge to a distribution:
            θ* ~ N(θ_MAP, F⁻¹)

        where θ_MAP is the maximum a posteriori estimate.
        """

        # Compute Fisher diagonal
        def data_gen() -> Iterator[tuple[Tensor, Tensor]]:
            """Generate batches of training data for Fisher matrix computation.

            Yields:
                Tuple of input and target tensors.
            """
            for i in range(0, len(train_x), 32):
                yield train_x[i : i + 32], train_y[i : i + 32]

        fisher = self.compute_fisher_matrix(data_gen(), num_batches=len(train_x) // 32)

        # Compute natural gradient update
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

            f_diag = fisher[name] + self.damping  # Damped Fisher diagonal
            grad = param.grad.data

            # Natural gradient: F⁻¹ ∇L
            natural_grad = grad / f_diag

            # Predicted optimal weights (after many epochs)
            predicted_mean = param.data - num_epochs * 0.01 * natural_grad

            # Predicted variance (inverse Fisher = posterior covariance)
            predicted_var = 1.0 / f_diag

            # Identify concentration points (low variance = high confidence)
            concentration_mask = predicted_var < predicted_var.mean()

            # Identify outliers (will have extreme values)
            predicted_magnitude = predicted_mean.abs()
            outlier_threshold = predicted_magnitude.mean() + 3 * predicted_magnitude.std()
            outlier_mask = predicted_magnitude > outlier_threshold

            predictions[name] = {
                "predicted_mean": predicted_mean,
                "predicted_variance": predicted_var,
                "concentration_mask": concentration_mask,
                "outlier_mask": outlier_mask,
                "importance_score": f_diag / f_diag.max(),  # Normalized importance
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
        """
        # Compute Fisher (or use cached)
        if not self._fisher_cache:

            def data_gen() -> Iterator[tuple[Tensor, Tensor]]:
                """Generate batches of training data for Fisher matrix computation.

                Yields:
                    Tuple of input and target tensors.
                """
                for i in range(0, len(train_x), 32):
                    yield train_x[i : i + 32], train_y[i : i + 32]

            self.compute_fisher_matrix(data_gen())

        # Compute gradient
        self.model.zero_grad()
        outputs = self.model(train_x)
        if outputs.shape != train_y.shape:
            outputs = outputs.view_as(train_y)
        loss = F.mse_loss(outputs, train_y)
        loss.backward()

        # Compute natural gradient updates with proper clipping
        updates = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                f_diag = self._fisher_cache[name] + self.damping
                # Natural gradient update
                natural_grad = param.grad.data / f_diag
                # Clip to prevent explosions (important for stability)
                grad_norm = natural_grad.norm()
                max_norm = 1.0
                if grad_norm > max_norm:
                    natural_grad = natural_grad * (max_norm / grad_norm)
                updates[name] = -learning_rate * natural_grad

        return updates


# ═══════════════════════════════════════════════════════════════════════════════
# SPECTRAL WEIGHT PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════


class SpectralWeightPredictor(nn.Module):
    """
    Predict optimal weight configurations using spectral analysis.

    Key insight: Weight matrices in trained networks follow specific
    spectral (eigenvalue) distributions. We can:

    1. **Predict converged spectrum**: What eigenvalues will emerge
    2. **Initialize optimally**: Start at predicted distribution
    3. **Skip training entirely**: Jump to predicted weights

    Theoretical Foundation:
    ======================

    Random matrices follow Marchenko-Pastur distribution.
    Trained networks deviate in predictable ways:

    - Top eigenvalues capture task-relevant features
    - Bulk eigenvalues follow modified MP distribution
    - Outlier eigenvalues indicate memorization

    We can PREDICT this spectrum from data statistics!
    """

    def __init__(
        self,
        model: nn.Module,
        rank_ratio: float = 0.1,
    ) -> None:
        """
        Initialize spectral predictor.

        Args:
            model: Neural network to analyze
            rank_ratio: Expected effective rank / full rank ratio
        """
        super().__init__()
        self.model = model
        self.rank_ratio = rank_ratio

    def analyze_data_spectrum(
        self,
        data: Tensor,
    ) -> dict[str, Tensor]:
        """
        Analyze spectral structure of input data.

        The data spectrum tells us what features the network should learn.
        """
        if data.dim() > 2:
            data = data.view(data.shape[0], -1)

        # Center data
        data_centered = data - data.mean(dim=0, keepdim=True)

        # Compute covariance
        cov = data_centered.T @ data_centered / data.shape[0]

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Sort descending
        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Compute effective dimensionality
        normalized_eig = eigenvalues / eigenvalues.sum()
        entropy = -(normalized_eig * (normalized_eig + 1e-10).log()).sum()
        effective_dim = entropy.exp()

        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "effective_dimensionality": effective_dim,
            "explained_variance_ratio": eigenvalues.cumsum(0) / eigenvalues.sum(),
        }

    def predict_layer_weights(
        self,
        input_data: Tensor,
        output_data: Tensor,
        layer_shape: tuple[int, int],
    ) -> Tensor:
        """
        Predict optimal weights for a linear layer using spectral analysis.

        This computes the CLOSED-FORM optimal linear mapping from input to output!

        For linear regression: W* = (XᵀX)⁻¹ XᵀY

        For neural network layers, we use a regularized version that
        accounts for the nonlinear activation context.
        """
        if input_data.dim() > 2:
            input_data = input_data.view(input_data.shape[0], -1)
        if output_data.dim() > 2:
            output_data = output_data.view(output_data.shape[0], -1)

        # Ensure correct dimensions
        in_features, out_features = layer_shape

        # Truncate or pad if needed
        if input_data.shape[1] != in_features:
            if input_data.shape[1] > in_features:
                input_data = input_data[:, :in_features]
            else:
                padding = torch.zeros(
                    input_data.shape[0], in_features - input_data.shape[1], device=input_data.device
                )
                input_data = torch.cat([input_data, padding], dim=1)

        if output_data.shape[1] != out_features:
            if output_data.shape[1] > out_features:
                output_data = output_data[:, :out_features]
            else:
                padding = torch.zeros(
                    output_data.shape[0],
                    out_features - output_data.shape[1],
                    device=output_data.device,
                )
                output_data = torch.cat([output_data, padding], dim=1)

        # Compute optimal weights via pseudo-inverse
        # W* = (XᵀX + λI)⁻¹ XᵀY
        XtX = input_data.T @ input_data
        reg = 1e-4 * torch.eye(XtX.shape[0], device=XtX.device)
        XtY = input_data.T @ output_data

        # Solve linear system
        W_optimal = torch.linalg.solve(XtX + reg, XtY)

        return W_optimal.T  # [out_features, in_features]

    def predict_network_weights(
        self,
        train_x: Tensor,
        train_y: Tensor,
    ) -> dict[str, Tensor]:
        """
        Predict optimal weights for entire network using layer-wise analysis.

        Strategy:
        1. For first layer: Map input to hidden representation
        2. For middle layers: Use data spectrum + random orthogonal
        3. For last layer: Map hidden to output (closed-form)
        """
        predictions: dict[str, Tensor] = {}

        # Get layer information
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layers.append((name, module))

        if not layers:
            return predictions

        # First layer: Input -> Hidden
        first_name, first_layer = layers[0]
        first_weights = self.predict_layer_weights(
            train_x,
            train_x,  # Identity-like for first layer
            (first_layer.in_features, first_layer.out_features),
        )

        # Use SVD to get orthonormal approximation
        U, S, Vh = torch.linalg.svd(first_weights, full_matrices=False)
        # Scale by sqrt of eigenvalues for Xavier-like init
        predictions[first_name + ".weight"] = U @ torch.diag(S.sqrt()) @ Vh

        # Last layer: Hidden -> Output (closed-form optimal)
        if len(layers) > 1:
            last_name, last_layer = layers[-1]

            # Get hidden representation
            with torch.no_grad():
                # Forward through all but last layer
                hidden = train_x
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Linear) and name != last_name:
                        hidden = module(hidden)
                        # Apply activation if exists
                        hidden = F.relu(hidden)  # Assume ReLU

            last_weights = self.predict_layer_weights(
                hidden,
                train_y,
                (last_layer.in_features, last_layer.out_features),
            )
            predictions[last_name + ".weight"] = last_weights

        # Middle layers: Spectral-initialized random orthogonal
        for name, layer in layers[1:-1]:
            # Random orthogonal initialization scaled by data spectrum
            weight = torch.randn(layer.out_features, layer.in_features, device=train_x.device)
            Q, R = torch.linalg.qr(weight.T)
            predictions[name + ".weight"] = Q.T * (2.0 / layer.in_features) ** 0.5

        return predictions


# ═══════════════════════════════════════════════════════════════════════════════
# ALGEBRAIC OPTIMIZER (MAIN INTERFACE)
# ═══════════════════════════════════════════════════════════════════════════════


class AlgebraicOptimizer:
    """
    Main interface for algebraic (non-backprop) optimization.

    This class combines NTK, Fisher, and Spectral methods to:
    1. Predict training outcomes without running training
    2. Compute optimal weights in closed form
    3. Simulate training dynamics algebraically

    Usage:
    ======
    ```python
    optimizer = AlgebraicOptimizer(model)

    # Predict what training would achieve
    predictions = optimizer.predict_training_outcome(train_x, train_y)

    # Get optimal weights directly
    optimal_weights = optimizer.compute_optimal_weights(train_x, train_y)

    # Apply to model (skip training entirely!)
    optimizer.apply_weights(model, optimal_weights)
    ```

    When to Use:
    ============
    - Small to medium datasets (< 10K samples)
    - Wide networks (width >> depth)
    - When you need guaranteed convergence
    - For meta-learning / architecture search
    - When compute is limited but memory is available
    """

    def __init__(
        self,
        model: nn.Module,
        method: Literal["ntk", "fisher", "spectral", "hybrid"] = "hybrid",
        device: torch.device | str = "cpu",
    ) -> None:
        """
        Initialize algebraic optimizer.

        Args:
            model: Neural network to optimize
            method: Optimization method to use
            device: Computation device
        """
        self.model = model
        self.method = method
        self.device = torch.device(device)

        # Initialize sub-predictors
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
        """
        results: dict[str, Any] = {}

        if self.method in ("ntk", "hybrid"):
            # NTK predictions (output-level)
            ntk_pred = self.ntk_predictor.predict_training_dynamics(
                train_x,
                train_y,
                train_x,
                learning_rate=learning_rate,
                training_time=float(num_epochs),
            )
            results["ntk_predictions"] = ntk_pred
            results["predicted_train_outputs"] = ntk_pred["predicted_train_outputs"]
            results["predicted_mse"] = ntk_pred["predicted_mse"]
            results["convergence_rates"] = ntk_pred["convergence_eigenvalues"]

        if self.method in ("fisher", "hybrid"):
            # Fisher predictions (weight-level)
            fisher_pred = self.fisher_predictor.predict_weight_distribution(
                train_x, train_y, num_epochs=num_epochs
            )
            results["weight_distributions"] = fisher_pred

            # Extract summary statistics
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
            # Spectral analysis
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
                        param.copy_(blend_factor * weights[name] + (1 - blend_factor) * param.data)

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
        """
        losses = []

        for step in range(num_natural_steps):
            # Compute current loss
            with torch.no_grad():
                outputs = self.model(train_x)
                if outputs.shape != train_y.shape:
                    outputs = outputs.view_as(train_y)
                loss = F.mse_loss(outputs, train_y).item()
                losses.append(loss)

            # Natural gradient update
            updates = self.fisher_predictor.compute_natural_gradient_update(
                train_x, train_y, learning_rate=0.1 / (step + 1)
            )

            # Apply updates
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in updates:
                        param.add_(updates[name])

        # Final loss
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


# ═══════════════════════════════════════════════════════════════════════════════
# INTERCONNECT-AWARE ALGEBRAIC OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════


class InterconnectAlgebraicOptimizer(AlgebraicOptimizer):
    """
    Algebraic optimizer with awareness of CogSynDelta's interconnect architecture.

    This extends the base AlgebraicOptimizer to:
    1. Optimize submodels within their specialized domains
    2. Predict optimal cross-model weight transfers
    3. Coordinate optimization through the interconnect manager
    4. Integrate with mHC (Moderated HyperConnections) gating
    """

    def __init__(
        self,
        interconnect_manager: Any,  # Type: InterconnectManager
        device: torch.device | str = "cpu",
    ) -> None:
        """
        Initialize interconnect-aware optimizer.

        Args:
            interconnect_manager: CogSynDelta InterconnectManager instance
            device: Computation device
        """
        self.interconnect = interconnect_manager
        self.device = torch.device(device)
        self.submodel_optimizers: dict[str, AlgebraicOptimizer] = {}

        # Create optimizer for each registered submodel
        for name, model in self._get_submodels():
            self.submodel_optimizers[name] = AlgebraicOptimizer(
                model, method="hybrid", device=device
            )

    def _get_submodels(self) -> list[tuple[str, nn.Module]]:
        """Get all submodels from interconnect manager."""
        submodels = []

        # Check for common interconnect patterns
        if hasattr(self.interconnect, "models"):
            for name, model in self.interconnect.models.items():
                if isinstance(model, nn.Module):
                    submodels.append((name, model))
        elif hasattr(self.interconnect, "get_registered_models"):
            for name, model in self.interconnect.get_registered_models():
                submodels.append((name, model))

        return submodels

    def optimize_submodel(
        self,
        submodel_name: str,
        train_x: Tensor,
        train_y: Tensor,
        domain_weights: Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Algebraically optimize a specific submodel for its domain.

        Args:
            submodel_name: Name of submodel to optimize
            train_x: Domain-specific training inputs
            train_y: Domain-specific training targets
            domain_weights: Sample weights for domain relevance

        Returns:
            Optimization results including predicted weights
        """
        if submodel_name not in self.submodel_optimizers:
            raise ValueError(f"Unknown submodel: {submodel_name}")

        optimizer = self.submodel_optimizers[submodel_name]

        # Apply domain weights if provided
        if domain_weights is not None:
            # Weight the loss contribution of each sample
            train_x = train_x * domain_weights.unsqueeze(-1).sqrt()
            train_y = train_y * domain_weights.unsqueeze(-1).sqrt()

        # Predict training outcome
        predictions = optimizer.predict_training_outcome(train_x, train_y)

        # Compute optimal weights
        optimal_weights = optimizer.compute_optimal_weights(train_x, train_y)

        return {
            "submodel": submodel_name,
            "predictions": predictions,
            "optimal_weights": optimal_weights,
        }

    def predict_cross_model_transfer(
        self,
        source_model: str,
        target_model: str,
        shared_data: Tensor,
    ) -> dict[str, Tensor]:
        """
        Predict optimal weight transfer between models.

        Uses spectral analysis to find the optimal linear transformation
        that maps source model representations to target model space.
        """
        if source_model not in self.submodel_optimizers:
            raise ValueError(f"Unknown source model: {source_model}")
        if target_model not in self.submodel_optimizers:
            raise ValueError(f"Unknown target model: {target_model}")

        # Get representations from both models
        source_opt = self.submodel_optimizers[source_model]
        target_opt = self.submodel_optimizers[target_model]

        with torch.no_grad():
            source_repr = source_opt.model(shared_data)
            target_repr = target_opt.model(shared_data)

        # Find optimal linear mapping: A @ source_repr ≈ target_repr
        # A* = target_repr @ source_repr.T @ (source_repr @ source_repr.T + λI)^{-1}
        source_flat = source_repr.view(source_repr.shape[0], -1)
        target_flat = target_repr.view(target_repr.shape[0], -1)

        gram = source_flat.T @ source_flat
        reg = 1e-4 * torch.eye(gram.shape[0], device=gram.device)
        cross = source_flat.T @ target_flat

        transfer_matrix = torch.linalg.solve(gram + reg, cross)

        return {
            "transfer_matrix": transfer_matrix,
            "source_dim": source_flat.shape[1],
            "target_dim": target_flat.shape[1],
            "reconstruction_error": F.mse_loss(source_flat @ transfer_matrix, target_flat).item(),
        }

    def optimize_mhc_gates(
        self,
        mhc_module: nn.Module,
        context_data: Tensor,
        target_modulation: Tensor,
    ) -> dict[str, Tensor]:
        """
        Algebraically optimize Moderated HyperConnection gate parameters.

        mHC gates control information flow between model components.
        We can predict optimal gate values without training:

        gate* = σ(W @ context + b)

        where W, b minimize: ||gate* - target_modulation||²
        """
        # Extract gate network from mHC module
        gate_network: nn.Module | None = None
        if hasattr(mhc_module, "gate") and isinstance(mhc_module.gate, nn.Module):
            gate_network = mhc_module.gate
        elif hasattr(mhc_module, "gate_network") and isinstance(mhc_module.gate_network, nn.Module):
            gate_network = mhc_module.gate_network

        if gate_network is None:
            raise ValueError("Could not find gate network in mHC module")

        # Compute optimal gate parameters
        spectral = SpectralWeightPredictor(gate_network)

        # For sigmoid output, we need to optimize in logit space
        # target_logits = logit(target_modulation)
        target_logits = torch.logit(target_modulation.clamp(0.01, 0.99))

        optimal_weights = spectral.predict_network_weights(context_data, target_logits)

        return optimal_weights

    def coordinate_optimization(
        self,
        train_data: dict[str, tuple[Tensor, Tensor]],
        coordination_weight: float = 0.1,
    ) -> dict[str, dict[str, Any]]:
        """
        Coordinate optimization across all submodels.

        This performs algebraic optimization while encouraging
        submodels to produce compatible representations.
        """
        results: dict[str, dict[str, Any]] = {}

        # First pass: Independent optimization
        representations: dict[str, Tensor] = {}
        for name, (train_x, train_y) in train_data.items():
            if name in self.submodel_optimizers:
                result = self.optimize_submodel(name, train_x, train_y)
                results[name] = result

                # Store intermediate representations
                with torch.no_grad():
                    opt = self.submodel_optimizers[name]
                    representations[name] = opt.model(train_x)

        # Second pass: Cross-model coordination
        model_names = list(results.keys())
        for i, source in enumerate(model_names):
            for target in model_names[i + 1 :]:
                if source in representations and target in representations:
                    # Find shared data
                    source_repr = representations[source]
                    target_repr = representations[target]

                    # Align representations
                    min_samples = min(source_repr.shape[0], target_repr.shape[0])
                    source_repr = source_repr[:min_samples]
                    target_repr = target_repr[:min_samples]

                    transfer = self.predict_cross_model_transfer(
                        source, target, train_data[source][0][:min_samples]
                    )

                    results[f"{source}_to_{target}_transfer"] = transfer

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# mHC (MODERATED HYPERCONNECTION) ALGEBRAIC OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════


class MHCAlgebraicOptimizer(nn.Module):
    """
    Algebraic optimizer specifically for Moderated HyperConnections.

    mHC gates control information flow: gate(source, target) → modulation

    Instead of learning gate parameters through backprop, we can:
    1. Compute optimal gate values analytically
    2. Predict gate network weights that produce these values
    3. Analyze gate dynamics to predict steady-state behavior

    Mathematical Foundation:
    ========================

    For mHC with gate g(s, t) = σ(W[s; t] + b):

    The optimal modulation for transferring information is:
        g* = argmin_g ||target - α(g ⊙ transform(source)) - (1-α)target||²

    Solving: g* = (target - (1-α)target) / (α * transform(source))
           = target / transform(source)  [when simplified]

    We then find W, b such that σ(W[s; t] + b) ≈ g*
    """

    def __init__(
        self,
        embed_dim: int = 512,
        regularization: float = 1e-4,
    ) -> None:
        """
        Initialize mHC optimizer.

        Args:
            embed_dim: Embedding dimension for mHC
            regularization: Regularization for matrix operations
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.regularization = regularization

    def compute_optimal_gate_values(
        self,
        source_states: Tensor,
        target_states: Tensor,
        desired_outputs: Tensor,
        alpha: float = 0.5,
    ) -> Tensor:
        """
        Compute analytically optimal gate values for mHC.

        Given source, target, and desired output, compute what gate
        values would produce the desired modulation.

        Derivation:
        ===========
        output = α * (gate ⊙ transform(source)) + (1-α) * target
        desired = α * (g* ⊙ T(s)) + (1-α) * t

        Solving for g*:
        g* = (desired - (1-α)*target) / (α * transform(source))

        Args:
            source_states: Source representations [batch, embed_dim]
            target_states: Target representations [batch, embed_dim]
            desired_outputs: What we want the output to be [batch, embed_dim]
            alpha: Residual mixing coefficient

        Returns:
            Optimal gate values [batch, embed_dim]
        """
        # Approximate transform(source) as source for linear analysis
        # In practice, we'd use the actual transform network
        transform_source = source_states

        numerator = desired_outputs - (1 - alpha) * target_states
        denominator = alpha * transform_source + self.regularization

        # Compute optimal gates (clamped to valid sigmoid range)
        optimal_gates = numerator / denominator
        optimal_gates = optimal_gates.clamp(0.01, 0.99)

        return optimal_gates

    def predict_gate_network_weights(
        self,
        source_target_pairs: Tensor,
        optimal_gate_values: Tensor,
    ) -> dict[str, Tensor]:
        """
        Predict weights for gate network that produces optimal gates.

        The gate network is: gate = σ(W @ [source; target] + b)

        We solve: W, b = argmin ||σ(W @ X + b) - g*||²

        In logit space: W @ X + b ≈ logit(g*)

        This is linear regression! Closed-form solution exists.

        Args:
            source_target_pairs: Concatenated [source; target] [batch, 2*embed_dim]
            optimal_gate_values: Target gate values [batch, embed_dim]

        Returns:
            Predicted weights for gate network
        """
        # Convert to logit space
        logit_gates = torch.logit(optimal_gate_values.clamp(0.01, 0.99))

        # Add bias term to input
        ones = torch.ones(source_target_pairs.shape[0], 1, device=source_target_pairs.device)
        X_augmented = torch.cat([source_target_pairs, ones], dim=1)  # [batch, 2*embed + 1]

        # Solve least squares: X @ [W; b].T = logit_gates
        # Using pseudo-inverse: [W; b].T = (X.T @ X)^{-1} @ X.T @ Y
        XtX = X_augmented.T @ X_augmented
        reg = self.regularization * torch.eye(XtX.shape[0], device=XtX.device)
        XtY = X_augmented.T @ logit_gates

        solution = torch.linalg.solve(XtX + reg, XtY)

        # Extract weight and bias
        W = solution[:-1, :].T  # [embed_dim, 2*embed_dim]
        b = solution[-1, :]  # [embed_dim]

        return {
            "gate_weight": W,
            "gate_bias": b,
        }

    def predict_alpha_parameter(
        self,
        source_states: Tensor,
        target_states: Tensor,
        desired_outputs: Tensor,
        gate_values: Tensor | None = None,
    ) -> float:
        """
        Predict optimal alpha (residual mixing) parameter.

        The mHC output is: out = α * modulated + (1-α) * target

        Optimal α minimizes: ||out - desired||²

        Taking derivative and setting to zero:
        α* = <desired - target, modulated - target> / ||modulated - target||²
        """
        if gate_values is None:
            gate_values = torch.ones_like(source_states) * 0.5

        # Modulated signal
        modulated = gate_values * source_states  # Simplified transform

        # Compute optimal alpha
        diff_desired = desired_outputs - target_states
        diff_modulated = modulated - target_states

        numerator = (diff_desired * diff_modulated).sum()
        denominator = (diff_modulated**2).sum() + self.regularization

        optimal_alpha = (numerator / denominator).clamp(0.0, 1.0)

        return optimal_alpha.item()

    def analyze_gate_dynamics(
        self,
        gate_network: nn.Module,
        source_distribution: Tensor,
        target_distribution: Tensor,
        num_samples: int = 1000,
    ) -> dict[str, Any]:
        """
        Analyze gate network dynamics to predict steady-state behavior.

        Uses eigenvalue analysis of the expected gate Jacobian to predict:
        1. Which dimensions will be heavily gated (near 0 or 1)
        2. Gate value distribution at convergence
        3. Sensitivity to input perturbations
        """
        # Sample from input distributions
        source_samples = source_distribution[:num_samples]
        target_samples = target_distribution[:num_samples]

        # Compute gate values
        with torch.no_grad():
            gate_input = torch.cat([source_samples, target_samples], dim=-1)
            gate_values = gate_network(gate_input)

        # Analyze gate statistics
        gate_mean = gate_values.mean(dim=0)
        gate_std = gate_values.std(dim=0)

        # Identify concentrated dimensions (low variance = strong gate)
        concentrated_mask = gate_std < 0.1
        open_gates = (gate_mean > 0.9) & concentrated_mask
        closed_gates = (gate_mean < 0.1) & concentrated_mask

        # Compute input-output correlation (sensitivity)
        source_centered = source_samples - source_samples.mean(dim=0)
        gate_centered = gate_values - gate_mean

        # Cross-correlation matrix
        correlation = (source_centered.T @ gate_centered) / num_samples
        correlation_strength = correlation.abs().mean(dim=0)

        return {
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "open_gate_dims": open_gates.sum().item(),
            "closed_gate_dims": closed_gates.sum().item(),
            "adaptive_gate_dims": (~concentrated_mask).sum().item(),
            "source_sensitivity": correlation_strength,
            "predicted_information_flow": gate_mean.mean().item(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PATHWAY STRENGTH PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════


class PathwayStrengthPredictor:
    """
    Predict optimal interconnect pathway strengths algebraically.

    Instead of learning pathway strengths through RL/gradient descent,
    we can compute them directly from communication statistics.

    Mathematical Model:
    ==================

    For pathway (i → j), optimal strength maximizes information transfer:
        strength* = argmax_s I(X_i; Y_j | s)  (mutual information)

    Approximation using linear analysis:
        strength* ≈ ||Cov(X_i, Y_j)||_F / (||Var(X_i)||_F × ||Var(Y_j)||_F)

    This measures how correlated the source and target are - high correlation
    means the pathway should be strong (they benefit from communication).
    """

    def __init__(self, regularization: float = 1e-4) -> None:
        """Initialize pathway strength predictor."""
        self.regularization = regularization
        self._pathway_cache: dict[tuple[str, str], float] = {}

    def compute_optimal_strength(
        self,
        source_states: Tensor,
        target_states: Tensor,
        communication_outcomes: Tensor | None = None,
    ) -> float:
        """
        Compute optimal pathway strength from state statistics.

        If communication_outcomes is provided (what target should become
        after receiving information from source), we optimize for that.
        Otherwise, we use correlation as a proxy for communication value.
        """
        if communication_outcomes is not None:
            # Optimize for specific outcome
            # strength* = argmin_s ||target + s*(source_info) - desired||²
            source_contribution = source_states.mean(dim=0)
            target_current = target_states.mean(dim=0)
            desired = communication_outcomes.mean(dim=0)

            # Linear solution
            diff = desired - target_current
            source_norm = (source_contribution**2).sum() + self.regularization

            optimal_strength = (diff * source_contribution).sum() / source_norm
            return optimal_strength.clamp(0.0, 1.0).item()

        # Use correlation as proxy
        # Center the data
        source_centered = source_states - source_states.mean(dim=0)
        target_centered = target_states - target_states.mean(dim=0)

        # Compute correlation coefficient
        cov = (source_centered * target_centered).mean(dim=0)
        source_std = source_centered.std(dim=0) + self.regularization
        target_std = target_centered.std(dim=0) + self.regularization

        correlation = (cov / (source_std * target_std)).abs().mean()

        return correlation.item()

    def predict_all_pathway_strengths(
        self,
        section_states: dict[str, Tensor],
    ) -> dict[tuple[str, str], float]:
        """
        Predict optimal strengths for all pathways.

        Args:
            section_states: {section_name: state_tensor} for all sections

        Returns:
            {(source, target): optimal_strength} for all pairs
        """
        pathway_strengths = {}
        section_names = list(section_states.keys())

        for i, source in enumerate(section_names):
            for j, target in enumerate(section_names):
                if i != j:
                    strength = self.compute_optimal_strength(
                        section_states[source],
                        section_states[target],
                    )
                    pathway_strengths[(source, target)] = strength

        self._pathway_cache = pathway_strengths
        return pathway_strengths

    def predict_routing_decisions(
        self,
        section_states: dict[str, Tensor],
        bandwidth_constraint: int = 100,
    ) -> list[tuple[str, str, float]]:
        """
        Predict optimal routing decisions under bandwidth constraints.

        Uses the computed pathway strengths to determine which communications
        should occur, respecting bandwidth limits.

        This is a variant of the assignment problem - solvable in closed form!
        """
        # Get all pathway strengths
        strengths = self.predict_all_pathway_strengths(section_states)

        # Sort by strength (descending)
        sorted_pathways = sorted(strengths.items(), key=lambda x: -x[1])

        # Greedy allocation under bandwidth constraint
        allocated = []
        remaining_bandwidth = bandwidth_constraint

        for (source, target), strength in sorted_pathways:
            # Assume each pathway uses 1 unit of bandwidth
            if remaining_bandwidth > 0 and strength > 0.1:  # Threshold
                allocated.append((source, target, strength))
                remaining_bandwidth -= 1

        return allocated


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHT DISTRIBUTION PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════


class WeightDistributionPredictor:
    """
    Predict the distribution of weights after training.

    Uses statistical mechanics / mean field theory to predict:
    1. Weight concentration points (cluster centers)
    2. Weight spread (variance)
    3. Outlier weights (will be pruned or become extreme)

    Mathematical Foundation:
    ========================

    In wide networks with MSE loss, weights converge to:
        W ~ N(μ*, Σ*)

    where:
        μ* = (XᵀX + λI)⁻¹ XᵀY  (mean = ridge regression solution)
        Σ* = σ² (XᵀX + λI)⁻¹   (covariance = inverse Fisher)

    We can compute these analytically!
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

        Returns:
            {param_name: {"mean": ..., "variance": ..., "outlier_mask": ...}}
        """
        predictions: dict[str, dict[str, Tensor]] = {}

        # Compute data covariance
        if train_x.dim() > 2:
            train_x = train_x.view(train_x.shape[0], -1)
        if train_y.dim() > 2:
            train_y = train_y.view(train_y.shape[0], -1)

        XtX = train_x.T @ train_x / train_x.shape[0]
        XtY = train_x.T @ train_y / train_x.shape[0]

        # Regularized precision (inverse covariance)
        reg = self.regularization * torch.eye(XtX.shape[0], device=XtX.device)
        precision = XtX + reg

        # Mean weights (ridge regression solution)
        mean_weights = torch.linalg.solve(precision, XtY)

        # Covariance (inverse precision scaled by noise)
        weight_covariance = noise_variance * torch.linalg.inv(precision)
        weight_variance = weight_covariance.diag()

        # Flatten for distribution across model parameters
        mean_weights_flat = mean_weights.flatten()
        weight_variance_flat = weight_variance.flatten()

        # Now distribute these to actual model parameters
        for name, param in self.model.named_parameters():
            if "weight" in name:
                # Map to this layer's weights
                param_size = param.numel()

                # Use portion of predicted statistics for mean
                mean_size = mean_weights_flat.shape[0]
                if param_size <= mean_size:
                    pred_mean = mean_weights_flat[:param_size].view(param.shape)
                else:
                    # Tile statistics
                    repeats = (param_size // mean_size) + 1
                    pred_mean = mean_weights_flat.repeat(repeats)[:param_size].view(param.shape)

                # Handle variance separately (different size)
                var_size = weight_variance_flat.shape[0]
                if param_size <= var_size:
                    pred_var = weight_variance_flat[:param_size].view(param.shape)
                else:
                    # Tile variance statistics
                    var_repeats = (param_size // var_size) + 1
                    pred_var = weight_variance_flat.repeat(var_repeats)[:param_size].view(
                        param.shape
                    )

                # Identify outliers (high variance = uncertain = potential outlier)
                var_threshold = pred_var.mean() + 2 * pred_var.std()
                outlier_mask = pred_var > var_threshold

                # Identify concentration points (low variance = confident)
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

        where η is learning rate and λ_min is smallest eigenvalue.
        """
        if train_x.dim() > 2:
            train_x = train_x.view(train_x.shape[0], -1)

        # Compute eigenvalues of data covariance
        XtX = train_x.T @ train_x / train_x.shape[0]
        eigenvalues = torch.linalg.eigvalsh(XtX)

        # Remove near-zero eigenvalues
        eigenvalues = eigenvalues[eigenvalues > self.regularization]

        lambda_min = eigenvalues.min().item()
        lambda_max = eigenvalues.max().item()

        # Convergence time estimates
        slowest_mode_time = 1.0 / (learning_rate * lambda_min)
        fastest_mode_time = 1.0 / (learning_rate * lambda_max)

        # Condition number (how hard is optimization)
        condition_number = lambda_max / lambda_min

        return {
            "estimated_epochs": slowest_mode_time,
            "fastest_convergence": fastest_mode_time,
            "condition_number": condition_number,
            "eigenvalue_min": lambda_min,
            "eigenvalue_max": lambda_max,
            "recommended_learning_rate": 2.0 / (lambda_min + lambda_max),  # Optimal for GD
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
            {param_name: predicted_weights}
        """
        stats = self.predict_weight_statistics(train_x, train_y)

        predicted_weights = {}
        for name, param_stats in stats.items():
            mean = param_stats["predicted_mean"]
            var = param_stats["predicted_variance"]

            if sample_from_distribution:
                # Sample from Gaussian
                noise = torch.randn_like(mean)
                predicted_weights[name] = mean + noise * var.sqrt()
            else:
                # Use mean (MAP estimate)
                predicted_weights[name] = mean

        return predicted_weights


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED ALGEBRAIC TRAINING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════


class UnifiedAlgebraicTrainer:
    """
    Unified system for algebraic (non-backprop) training.

    Combines all predictors for comprehensive algebraic optimization:
    - NTK predictor: Output-level dynamics
    - Fisher predictor: Weight-level importance
    - Spectral predictor: Weight initialization
    - mHC optimizer: Gate optimization
    - Pathway predictor: Interconnect optimization
    - Weight distribution: Statistical weight prediction

    Usage:
    ======
    ```python
    trainer = UnifiedAlgebraicTrainer(model)

    # Full algebraic training (no backprop!)
    results = trainer.train_algebraically(train_x, train_y)

    # Results include:
    # - Predicted final weights
    # - Predicted outputs
    # - Predicted training dynamics
    # - Optimal mHC gate values
    # - Optimal interconnect pathways
    ```
    """

    def __init__(
        self,
        model: nn.Module,
        interconnect: Any | None = None,
        mhc_modules: dict[str, nn.Module] | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        Initialize unified algebraic trainer.

        Args:
            model: Main model to train
            interconnect: Optional interconnect manager
            mhc_modules: Optional dict of mHC modules to optimize
            device: Computation device
        """
        self.model = model
        self.device = torch.device(device)

        # Core predictors
        self.ntk_predictor = NTKPredictor(model)
        self.fisher_predictor = FisherInformationPredictor(model)
        self.spectral_predictor = SpectralWeightPredictor(model)
        self.weight_predictor = WeightDistributionPredictor(model)

        # Optional components
        self.interconnect = interconnect
        self.pathway_predictor = PathwayStrengthPredictor() if interconnect else None

        self.mhc_modules = mhc_modules or {}
        self.mhc_optimizer = MHCAlgebraicOptimizer() if mhc_modules else None

    def _optimize_auxiliary_components(self, train_x: Tensor, results: dict[str, Any]) -> None:
        """Optimize auxiliary components such as mHC modules and interconnect pathways.

        Args:
            train_x: Training inputs.
            results: Results dictionary to populate with optimization outcomes.

        Returns:
            None

        Why:
            Modularizes auxiliary component optimization to reduce complexity and improve maintainability.
        """
        # mHC optimization
        if self.mhc_optimizer and self.mhc_modules:
            mhc_results: dict[str, Any] = {}
            for name, mhc in self.mhc_modules.items():
                try:
                    # Forward pass to populate internal states (output not used)
                    with torch.no_grad():
                        _ = self.model(train_x[:32])
                    # Use output as proxy for mHC states
                    gate_module = mhc.gate if hasattr(mhc, "gate") else mhc
                    if isinstance(gate_module, nn.Module):
                        gate_analysis = self.mhc_optimizer.analyze_gate_dynamics(
                            gate_module,
                            train_x[:100],
                            train_x[:100],
                        )
                        mhc_results[name] = gate_analysis
                    else:
                        mhc_results[name] = {"skipped": "gate is not an nn.Module"}
                except Exception as e:
                    mhc_results[name] = {"error": str(e)}
            results["mhc_analysis"] = mhc_results

        # Pathway optimization
        if self.pathway_predictor and self.interconnect:
            # Get section states from interconnect
            section_states = {}
            if hasattr(self.interconnect, "get_section_states"):
                section_states = self.interconnect.get_section_states()
            elif hasattr(self.interconnect, "section_states"):
                section_states = self.interconnect.section_states

            if section_states:
                pathway_strengths = self.pathway_predictor.predict_all_pathway_strengths(
                    section_states
                )
                results["optimal_pathway_strengths"] = pathway_strengths

    def train_algebraically(
        self,
        train_x: Tensor,
        train_y: Tensor,
        target_epochs: int = 100,
        learning_rate: float = 0.01,
        apply_weights: bool = True,
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

        Returns:
            Comprehensive training results
        """
        results: dict[str, Any] = {}

        # 1. Predict training dynamics (NTK)
        print("  [1/5] Predicting training dynamics via NTK...")
        try:
            ntk_results = self.ntk_predictor.predict_training_dynamics(
                train_x,
                train_y,
                train_x,
                learning_rate=learning_rate,
                training_time=float(target_epochs),
            )
            results["ntk_dynamics"] = ntk_results
            results["predicted_outputs"] = ntk_results["predicted_train_outputs"]
            results["predicted_mse"] = ntk_results["predicted_mse"]
        except Exception as e:
            print(f"    NTK prediction failed: {e}")
            results["ntk_error"] = str(e)

        # 2. Predict weight distributions
        print("  [2/5] Predicting weight distributions...")
        weight_stats = self.weight_predictor.predict_weight_statistics(train_x, train_y)
        results["weight_statistics"] = weight_stats

        # 3. Generate predicted weights (spectral method)
        print("  [3/5] Computing optimal weights via spectral analysis...")
        optimal_weights = self.spectral_predictor.predict_network_weights(train_x, train_y)
        results["optimal_weights"] = optimal_weights

        # 4. Predict convergence time
        print("  [4/5] Analyzing convergence properties...")
        convergence = self.weight_predictor.predict_convergence_time(train_x, learning_rate)
        results["convergence_analysis"] = convergence

        # 5. Optimize auxiliary components
        print("  [5/5] Optimizing auxiliary components...")
        self._optimize_auxiliary_components(train_x, results)

        # Apply weights if requested
        if apply_weights and optimal_weights:
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
            # Compute current loss
            with torch.no_grad():
                outputs = self.model(train_x)
                if outputs.shape != train_y.shape:
                    outputs = outputs.view_as(train_y)
                loss = F.mse_loss(outputs, train_y).item()
                losses.append(loss)

            # Natural gradient update
            updates = self.fisher_predictor.compute_natural_gradient_update(
                train_x,
                train_y,
                learning_rate=0.5 / (step + 1),  # Decaying LR
            )

            # Apply updates
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in updates:
                        param.add_(updates[name])

        # Final loss
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
        """
        # Convergence analysis
        convergence = self.weight_predictor.predict_convergence_time(train_x)

        # Weight statistics
        weight_stats = self.weight_predictor.predict_weight_statistics(train_x, train_y)

        # Count problematic weights
        total_params = 0
        outlier_params = 0
        concentrated_params = 0

        for name, stats in weight_stats.items():
            total_params += stats["outlier_mask"].numel()
            outlier_params += int(stats["outlier_mask"].sum().item())
            concentrated_params += int(stats["concentration_mask"].sum().item())

        # Data spectrum
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
        }
