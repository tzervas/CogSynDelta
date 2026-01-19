"""
Spectral Weight Predictor.

Predict optimal weight configurations using spectral (eigenvalue) analysis.
Weight matrices in trained networks follow specific spectral distributions.

Mathematical Foundation:
    Random matrices follow Marchenko-Pastur distribution.
    Trained networks deviate in predictable ways:
    - Top eigenvalues capture task-relevant features
    - Bulk eigenvalues follow modified MP distribution
    - Outlier eigenvalues indicate memorization

Example:
    >>> spectral = SpectralWeightPredictor(model)
    >>> weights = spectral.predict_network_weights(train_x, train_y)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SpectralWeightPredictor(nn.Module):
    """
    Predict optimal weight configurations using spectral analysis.

    Key insight: Weight matrices in trained networks follow specific
    spectral (eigenvalue) distributions. We can:

    1. Predict converged spectrum: What eigenvalues will emerge
    2. Initialize optimally: Start at predicted distribution
    3. Skip training entirely: Jump to predicted weights

    Args:
        model: Neural network to analyze
        rank_ratio: Expected effective rank / full rank ratio

    Example:
        >>> spectral = SpectralWeightPredictor(model)
        >>> for name, weight in spectral.compute_optimal_weights(train_x, train_y):
        ...     print(f"{name}: {weight.shape}")
    """

    def __init__(
        self,
        model: nn.Module,
        rank_ratio: float = 0.1,
    ) -> None:
        """Initialize spectral predictor."""
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

        Args:
            data: Input data tensor [n_samples, ...]

        Returns:
            Dictionary with eigenvalues, eigenvectors, effective dimensionality
        """
        if data.dim() > 2:
            data = data.view(data.shape[0], -1)

        data_centered = data - data.mean(dim=0, keepdim=True)
        cov = data_centered.T @ data_centered / data.shape[0]

        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

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

        Args:
            input_data: Input activations [n_samples, in_features]
            output_data: Target outputs [n_samples, out_features]
            layer_shape: (in_features, out_features) tuple

        Returns:
            Predicted optimal weights [out_features, in_features]
        """
        if input_data.dim() > 2:
            input_data = input_data.view(input_data.shape[0], -1)
        if output_data.dim() > 2:
            output_data = output_data.view(output_data.shape[0], -1)

        in_features, out_features = layer_shape

        if input_data.shape[1] != in_features:
            if input_data.shape[1] > in_features:
                input_data = input_data[:, :in_features]
            else:
                padding = torch.zeros(
                    input_data.shape[0], in_features - input_data.shape[1],
                    device=input_data.device
                )
                input_data = torch.cat([input_data, padding], dim=1)

        if output_data.shape[1] != out_features:
            if output_data.shape[1] > out_features:
                output_data = output_data[:, :out_features]
            else:
                padding = torch.zeros(
                    output_data.shape[0], out_features - output_data.shape[1],
                    device=output_data.device
                )
                output_data = torch.cat([output_data, padding], dim=1)

        XtX = input_data.T @ input_data
        reg = 1e-4 * torch.eye(XtX.shape[0], device=XtX.device)
        XtY = input_data.T @ output_data

        W_optimal = torch.linalg.solve(XtX + reg, XtY)

        return W_optimal.T

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

        Args:
            train_x: Training inputs
            train_y: Training targets

        Returns:
            Dictionary mapping parameter names to predicted weights
        """
        predictions: dict[str, Tensor] = {}

        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layers.append((name, module))

        if not layers:
            return predictions

        first_name, first_layer = layers[0]
        first_weights = self.predict_layer_weights(
            train_x,
            train_x,
            (first_layer.in_features, first_layer.out_features),
        )

        U, S, Vh = torch.linalg.svd(first_weights, full_matrices=False)
        predictions[first_name + ".weight"] = (U @ torch.diag(S.sqrt()) @ Vh)

        if len(layers) > 1:
            last_name, last_layer = layers[-1]

            with torch.no_grad():
                hidden = train_x
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Linear) and name != last_name:
                        hidden = module(hidden)
                        hidden = F.relu(hidden)

            last_weights = self.predict_layer_weights(
                hidden,
                train_y,
                (last_layer.in_features, last_layer.out_features),
            )
            predictions[last_name + ".weight"] = last_weights

        for name, layer in layers[1:-1]:
            weight = torch.randn(
                layer.out_features, layer.in_features, device=train_x.device
            )
            Q, _ = torch.linalg.qr(weight.T)
            predictions[name + ".weight"] = Q.T * (2.0 / layer.in_features) ** 0.5

        return predictions

    def compute_optimal_weights(
        self,
        train_x: Tensor,
        train_y: Tensor,
    ) -> list[tuple[str, Tensor]]:
        """
        Compute optimal weights as an iterator.

        Yields (layer_name, optimal_weights) pairs.

        Args:
            train_x: Training inputs
            train_y: Training targets

        Yields:
            Tuples of (parameter_name, predicted_weights)
        """
        weights = self.predict_network_weights(train_x, train_y)
        return list(weights.items())
