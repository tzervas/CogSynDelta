"""
CogSynDelta-specific algebraic training integration.

This module provides CogSynDelta-specific wrappers and optimizers that
build on the core algebraic-training library for:
- mHC (Moderated HyperConnections) optimization
- Interconnect cross-model weight transfer
- Pathway strength prediction

These components are specific to CogSynDelta's architecture and may be
extracted to a separate integration package in the future.
"""

from __future__ import annotations

# Import from core library (will be in libs/)
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Add libs to path for development
_libs_path = Path(__file__).parent.parent.parent.parent / "libs" / "algebraic-training" / "src"
if _libs_path.exists() and str(_libs_path) not in sys.path:
    sys.path.insert(0, str(_libs_path))

from algebraic_training import (  # noqa: E402
    AlgebraicOptimizer,
    SpectralWeightPredictor,
)

if TYPE_CHECKING:
    pass


class MHCAlgebraicOptimizer(nn.Module):
    """
    Algebraic optimizer for Moderated HyperConnections (mHC).

    mHC gates control information flow: gate(source, target) → modulation

    Instead of learning gate parameters through backprop, we can:
    1. Compute optimal gate values analytically
    2. Predict gate network weights that produce these values
    3. Analyze gate dynamics to predict steady-state behavior

    Mathematical Foundation:
        For mHC with gate g(s, t) = σ(W[s; t] + b):

        The optimal modulation for transferring information is:
            g* = argmin_g ||target - α(g ⊙ transform(source)) - (1-α)target||²

        Solving: g* = (target - (1-α)target) / (α * transform(source))

    Args:
        embed_dim: Embedding dimension for mHC
        regularization: Regularization for matrix operations
    """

    def __init__(
        self,
        embed_dim: int = 512,
        regularization: float = 1e-4,
    ) -> None:
        """Initialize mHC optimizer."""
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

        Args:
            source_states: Source representations [batch, embed_dim]
            target_states: Target representations [batch, embed_dim]
            desired_outputs: What we want the output to be [batch, embed_dim]
            alpha: Residual mixing coefficient

        Returns:
            Optimal gate values [batch, embed_dim]
        """
        transform_source = source_states

        numerator = desired_outputs - (1 - alpha) * target_states
        denominator = alpha * transform_source + self.regularization

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

        We solve in logit space: W @ X + b ≈ logit(g*)

        Args:
            source_target_pairs: Concatenated [source; target] [batch, 2*embed_dim]
            optimal_gate_values: Target gate values [batch, embed_dim]

        Returns:
            Predicted weights for gate network
        """
        logit_gates = torch.logit(optimal_gate_values.clamp(0.01, 0.99))

        ones = torch.ones(source_target_pairs.shape[0], 1, device=source_target_pairs.device)
        X_augmented = torch.cat([source_target_pairs, ones], dim=1)

        XtX = X_augmented.T @ X_augmented
        reg = self.regularization * torch.eye(XtX.shape[0], device=XtX.device)
        XtY = X_augmented.T @ logit_gates

        solution = torch.linalg.solve(XtX + reg, XtY)

        W = solution[:-1, :].T
        b = solution[-1, :]

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

        Args:
            source_states: Source representations
            target_states: Target representations
            desired_outputs: Desired output
            gate_values: Optional gate values (defaults to 0.5)

        Returns:
            Optimal alpha value
        """
        if gate_values is None:
            gate_values = torch.ones_like(source_states) * 0.5

        modulated = gate_values * source_states

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

        Args:
            gate_network: The gate network module
            source_distribution: Source state samples
            target_distribution: Target state samples
            num_samples: Number of samples to use

        Returns:
            Analysis results including gate statistics
        """
        source_samples = source_distribution[:num_samples]
        target_samples = target_distribution[:num_samples]

        with torch.no_grad():
            gate_input = torch.cat([source_samples, target_samples], dim=-1)
            gate_values = gate_network(gate_input)

        gate_mean = gate_values.mean(dim=0)
        gate_std = gate_values.std(dim=0)

        concentrated_mask = gate_std < 0.1
        open_gates = (gate_mean > 0.9) & concentrated_mask
        closed_gates = (gate_mean < 0.1) & concentrated_mask

        source_centered = source_samples - source_samples.mean(dim=0)
        gate_centered = gate_values - gate_mean

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


class InterconnectAlgebraicOptimizer(AlgebraicOptimizer):
    """
    Algebraic optimizer with awareness of CogSynDelta's interconnect architecture.

    Extends the base AlgebraicOptimizer to:
    1. Optimize submodels within their specialized domains
    2. Predict optimal cross-model weight transfers
    3. Coordinate optimization through the interconnect manager
    4. Integrate with mHC (Moderated HyperConnections) gating

    Args:
        interconnect_manager: CogSynDelta InterconnectManager instance
        device: Computation device
    """

    def __init__(
        self,
        interconnect_manager: Any,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize interconnect-aware optimizer."""
        self.interconnect = interconnect_manager
        self.device = torch.device(device)
        self.submodel_optimizers: dict[str, AlgebraicOptimizer] = {}

        for name, model in self._get_submodels():
            self.submodel_optimizers[name] = AlgebraicOptimizer(
                model, method="hybrid", device=device
            )

    def _get_submodels(self) -> list[tuple[str, nn.Module]]:
        """Get all submodels from interconnect manager."""
        submodels = []

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

        if domain_weights is not None:
            train_x = train_x * domain_weights.unsqueeze(-1).sqrt()
            train_y = train_y * domain_weights.unsqueeze(-1).sqrt()

        predictions = optimizer.predict_training_outcome(train_x, train_y)
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

        Args:
            source_model: Name of source model
            target_model: Name of target model
            shared_data: Data to use for computing transfer

        Returns:
            Transfer matrix and statistics
        """
        if source_model not in self.submodel_optimizers:
            raise ValueError(f"Unknown source model: {source_model}")
        if target_model not in self.submodel_optimizers:
            raise ValueError(f"Unknown target model: {target_model}")

        source_opt = self.submodel_optimizers[source_model]
        target_opt = self.submodel_optimizers[target_model]

        with torch.no_grad():
            source_repr = source_opt.model(shared_data)
            target_repr = target_opt.model(shared_data)

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

        Args:
            mhc_module: mHC module containing gate network
            context_data: Context inputs for gate
            target_modulation: Desired gate outputs

        Returns:
            Optimal gate network weights
        """
        gate_network: nn.Module | None = None
        if hasattr(mhc_module, "gate") and isinstance(mhc_module.gate, nn.Module):
            gate_network = mhc_module.gate
        elif hasattr(mhc_module, "gate_network") and isinstance(mhc_module.gate_network, nn.Module):
            gate_network = mhc_module.gate_network

        if gate_network is None:
            raise ValueError("Could not find gate network in mHC module")

        spectral = SpectralWeightPredictor(gate_network)

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

        Args:
            train_data: {submodel_name: (train_x, train_y)} for each submodel
            coordination_weight: Weight for cross-model coordination

        Returns:
            Optimization results for all submodels
        """
        results: dict[str, dict[str, Any]] = {}

        representations: dict[str, Tensor] = {}
        for name, (train_x, train_y) in train_data.items():
            if name in self.submodel_optimizers:
                result = self.optimize_submodel(name, train_x, train_y)
                results[name] = result

                with torch.no_grad():
                    opt = self.submodel_optimizers[name]
                    representations[name] = opt.model(train_x)

        model_names = list(results.keys())
        for i, source in enumerate(model_names):
            for target in model_names[i + 1 :]:
                if source in representations and target in representations:
                    source_repr = representations[source]
                    target_repr = representations[target]

                    min_samples = min(source_repr.shape[0], target_repr.shape[0])
                    source_repr = source_repr[:min_samples]
                    target_repr = target_repr[:min_samples]

                    transfer = self.predict_cross_model_transfer(
                        source, target, train_data[source][0][:min_samples]
                    )

                    results[f"{source}_to_{target}_transfer"] = transfer

        return results


class PathwayStrengthPredictor:
    """
    Predict optimal interconnect pathway strengths algebraically.

    Instead of learning pathway strengths through RL/gradient descent,
    we can compute them directly from communication statistics.

    Mathematical Model:
        For pathway (i → j), optimal strength maximizes information transfer:
            strength* = argmax_s I(X_i; Y_j | s)  (mutual information)

        Approximation using linear analysis:
            strength* ≈ ||Cov(X_i, Y_j)||_F / (||Var(X_i)||_F × ||Var(Y_j)||_F)

    Args:
        regularization: Regularization for numerical stability
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

        Args:
            source_states: Source section states
            target_states: Target section states
            communication_outcomes: Optional desired outcomes

        Returns:
            Optimal pathway strength [0, 1]
        """
        if communication_outcomes is not None:
            source_contribution = source_states.mean(dim=0)
            target_current = target_states.mean(dim=0)
            desired = communication_outcomes.mean(dim=0)

            diff = desired - target_current
            source_norm = (source_contribution**2).sum() + self.regularization

            optimal_strength = (diff * source_contribution).sum() / source_norm
            return optimal_strength.clamp(0.0, 1.0).item()

        source_centered = source_states - source_states.mean(dim=0)
        target_centered = target_states - target_states.mean(dim=0)

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

        Args:
            section_states: {section_name: state_tensor} for all sections
            bandwidth_constraint: Maximum number of pathways to activate

        Returns:
            List of (source, target, strength) tuples for allocated pathways
        """
        strengths = self.predict_all_pathway_strengths(section_states)

        sorted_pathways = sorted(strengths.items(), key=lambda x: -x[1])

        allocated = []
        remaining_bandwidth = bandwidth_constraint

        for (source, target), strength in sorted_pathways:
            if remaining_bandwidth > 0 and strength > 0.1:
                allocated.append((source, target, strength))
                remaining_bandwidth -= 1

        return allocated
