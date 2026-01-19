"""
Tests for Algebraic Training module.

Tests the ability to predict training outcomes without running backpropagation.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNetwork(nn.Module):
    """Simple test network."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TestAlgebraicOptimizer(unittest.TestCase):
    """Test the main AlgebraicOptimizer class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from cogsyndelta.optimization.algebraic_training import AlgebraicOptimizer

        self.model = SimpleNetwork(input_dim=32, hidden_dim=64, output_dim=8)
        self.optimizer = AlgebraicOptimizer(self.model, method="hybrid")

        # Generate synthetic training data
        self.train_x = torch.randn(100, 32)
        self.train_y = torch.randn(100, 8)

    def test_predict_training_outcome(self) -> None:
        """Test that we can predict training outcomes."""
        predictions = self.optimizer.predict_training_outcome(
            self.train_x, self.train_y,
            learning_rate=0.01,
            num_epochs=10,
        )

        # Should have multiple prediction types
        self.assertIn("weight_distributions", predictions)

        # Weight distributions should cover parameters
        self.assertGreater(len(predictions["weight_distributions"]), 0)

    def test_compute_optimal_weights(self) -> None:
        """Test optimal weight computation."""
        optimal_weights = self.optimizer.compute_optimal_weights(
            self.train_x, self.train_y, method="spectral"
        )

        # Should return weights for at least some parameters
        self.assertGreater(len(optimal_weights), 0)

        # Weights should be tensors
        for name, weight in optimal_weights.items():
            self.assertIsInstance(weight, torch.Tensor)

    def test_fast_train(self) -> None:
        """Test fast training with natural gradient."""
        results = self.optimizer.fast_train(
            self.train_x, self.train_y,
            num_natural_steps=3,
        )

        # Should have loss trajectory
        self.assertIn("initial_loss", results)
        self.assertIn("final_loss", results)
        self.assertIn("improvement_ratio", results)

        # Loss should not explode
        self.assertLess(results["final_loss"], results["initial_loss"] * 10)


class TestSpectralWeightPredictor(unittest.TestCase):
    """Test spectral weight prediction."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from cogsyndelta.optimization.algebraic_training import SpectralWeightPredictor

        self.model = SimpleNetwork(input_dim=32, hidden_dim=64, output_dim=8)
        self.predictor = SpectralWeightPredictor(self.model)
        self.train_x = torch.randn(100, 32)
        self.train_y = torch.randn(100, 8)

    def test_analyze_data_spectrum(self) -> None:
        """Test data spectrum analysis."""
        spectrum = self.predictor.analyze_data_spectrum(self.train_x)

        self.assertIn("eigenvalues", spectrum)
        self.assertIn("eigenvectors", spectrum)
        self.assertIn("effective_dimensionality", spectrum)

        # Eigenvalues should be sorted descending
        eigenvalues = spectrum["eigenvalues"]
        self.assertTrue(torch.all(eigenvalues[:-1] >= eigenvalues[1:]))

    def test_predict_layer_weights(self) -> None:
        """Test layer weight prediction."""
        input_data = torch.randn(100, 32)
        output_data = torch.randn(100, 64)

        weights = self.predictor.predict_layer_weights(
            input_data, output_data, (32, 64)
        )

        self.assertEqual(weights.shape, (64, 32))


class TestFisherInformationPredictor(unittest.TestCase):
    """Test Fisher Information prediction."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from cogsyndelta.optimization.algebraic_training import FisherInformationPredictor

        self.model = SimpleNetwork(input_dim=32, hidden_dim=64, output_dim=8)
        self.predictor = FisherInformationPredictor(self.model)
        self.train_x = torch.randn(100, 32)
        self.train_y = torch.randn(100, 8)

    def test_compute_fisher_matrix(self) -> None:
        """Test Fisher matrix computation."""
        def data_gen():
            for i in range(0, 100, 32):
                yield self.train_x[i:i+32], self.train_y[i:i+32]

        fisher = self.predictor.compute_fisher_matrix(data_gen(), num_batches=3)

        # Should have Fisher for each parameter
        self.assertGreater(len(fisher), 0)

        # Fisher values should be non-negative
        for name, f in fisher.items():
            self.assertTrue(torch.all(f >= 0))

    def test_predict_weight_distribution(self) -> None:
        """Test weight distribution prediction."""
        predictions = self.predictor.predict_weight_distribution(
            self.train_x, self.train_y, num_epochs=10
        )

        # Should have predictions for parameters
        self.assertGreater(len(predictions), 0)

        # Each prediction should have required fields
        for name, pred in predictions.items():
            self.assertIn("predicted_mean", pred)
            self.assertIn("predicted_variance", pred)
            self.assertIn("importance_score", pred)


class TestMHCAlgebraicOptimizer(unittest.TestCase):
    """Test mHC algebraic optimization."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from cogsyndelta.optimization.algebraic_training import MHCAlgebraicOptimizer

        self.optimizer = MHCAlgebraicOptimizer(embed_dim=64)

    def test_compute_optimal_gate_values(self) -> None:
        """Test optimal gate value computation."""
        source = torch.randn(32, 64)
        target = torch.randn(32, 64)
        desired = torch.randn(32, 64)

        gates = self.optimizer.compute_optimal_gate_values(
            source, target, desired, alpha=0.5
        )

        self.assertEqual(gates.shape, (32, 64))
        # Gates should be in valid sigmoid range
        self.assertTrue(torch.all(gates >= 0.01))
        self.assertTrue(torch.all(gates <= 0.99))

    def test_predict_gate_network_weights(self) -> None:
        """Test gate network weight prediction."""
        source_target = torch.randn(100, 128)  # Concatenated [source; target]
        optimal_gates = torch.sigmoid(torch.randn(100, 64))

        weights = self.optimizer.predict_gate_network_weights(
            source_target, optimal_gates
        )

        self.assertIn("gate_weight", weights)
        self.assertIn("gate_bias", weights)
        self.assertEqual(weights["gate_weight"].shape, (64, 128))
        self.assertEqual(weights["gate_bias"].shape, (64,))

    def test_predict_alpha_parameter(self) -> None:
        """Test alpha parameter prediction."""
        source = torch.randn(32, 64)
        target = torch.randn(32, 64)
        desired = torch.randn(32, 64)

        alpha = self.optimizer.predict_alpha_parameter(source, target, desired)

        self.assertIsInstance(alpha, float)
        self.assertGreaterEqual(alpha, 0.0)
        self.assertLessEqual(alpha, 1.0)


class TestPathwayStrengthPredictor(unittest.TestCase):
    """Test pathway strength prediction."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from cogsyndelta.optimization.algebraic_training import PathwayStrengthPredictor

        self.predictor = PathwayStrengthPredictor()

    def test_compute_optimal_strength(self) -> None:
        """Test optimal strength computation."""
        source = torch.randn(100, 64)
        target = torch.randn(100, 64)

        strength = self.predictor.compute_optimal_strength(source, target)

        self.assertIsInstance(strength, float)
        self.assertGreaterEqual(strength, 0.0)

    def test_predict_all_pathway_strengths(self) -> None:
        """Test prediction of all pathway strengths."""
        section_states = {
            "section_a": torch.randn(50, 64),
            "section_b": torch.randn(50, 64),
            "section_c": torch.randn(50, 64),
        }

        strengths = self.predictor.predict_all_pathway_strengths(section_states)

        # Should have n*(n-1) pathways for n sections
        expected_pathways = 3 * 2  # 3 sections, excluding self-loops
        self.assertEqual(len(strengths), expected_pathways)

        # All strengths should be non-negative
        for (src, tgt), strength in strengths.items():
            self.assertGreaterEqual(strength, 0.0)
            self.assertNotEqual(src, tgt)

    def test_predict_routing_decisions(self) -> None:
        """Test routing decision prediction under bandwidth constraints."""
        section_states = {
            "section_a": torch.randn(50, 64),
            "section_b": torch.randn(50, 64),
            "section_c": torch.randn(50, 64),
        }

        decisions = self.predictor.predict_routing_decisions(
            section_states, bandwidth_constraint=3
        )

        # Should respect bandwidth constraint
        self.assertLessEqual(len(decisions), 3)

        # Each decision should have (source, target, strength)
        for decision in decisions:
            self.assertEqual(len(decision), 3)
            src, tgt, strength = decision
            self.assertIn(src, section_states)
            self.assertIn(tgt, section_states)
            self.assertIsInstance(strength, float)


class TestWeightDistributionPredictor(unittest.TestCase):
    """Test weight distribution prediction."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from cogsyndelta.optimization.algebraic_training import WeightDistributionPredictor

        self.model = SimpleNetwork(input_dim=32, hidden_dim=64, output_dim=8)
        self.predictor = WeightDistributionPredictor(self.model)
        self.train_x = torch.randn(100, 32)
        self.train_y = torch.randn(100, 8)

    def test_predict_weight_statistics(self) -> None:
        """Test weight statistics prediction."""
        stats = self.predictor.predict_weight_statistics(self.train_x, self.train_y)

        # Should have predictions for weight parameters
        self.assertGreater(len(stats), 0)

        # Each stat should have required fields
        for name, stat in stats.items():
            self.assertIn("predicted_mean", stat)
            self.assertIn("predicted_variance", stat)
            self.assertIn("outlier_mask", stat)
            self.assertIn("concentration_mask", stat)

    def test_predict_convergence_time(self) -> None:
        """Test convergence time prediction."""
        convergence = self.predictor.predict_convergence_time(
            self.train_x, learning_rate=0.01
        )

        self.assertIn("estimated_epochs", convergence)
        self.assertIn("condition_number", convergence)
        self.assertIn("recommended_learning_rate", convergence)

        # Estimated epochs should be positive
        self.assertGreater(convergence["estimated_epochs"], 0)

    def test_generate_predicted_weights(self) -> None:
        """Test weight generation."""
        # Mean prediction (deterministic)
        weights_mean = self.predictor.generate_predicted_weights(
            self.train_x, self.train_y, sample_from_distribution=False
        )

        # Should have weights for some parameters
        self.assertGreater(len(weights_mean), 0)

        # Sampled prediction (stochastic)
        weights_sample1 = self.predictor.generate_predicted_weights(
            self.train_x, self.train_y, sample_from_distribution=True
        )
        weights_sample2 = self.predictor.generate_predicted_weights(
            self.train_x, self.train_y, sample_from_distribution=True
        )

        # Two samples should be different (probabilistic)
        any_different = False
        for name in weights_sample1:
            if name in weights_sample2:
                if not torch.allclose(weights_sample1[name], weights_sample2[name]):
                    any_different = True
                    break
        self.assertTrue(any_different, "Sampled weights should differ")


class TestUnifiedAlgebraicTrainer(unittest.TestCase):
    """Test the unified algebraic trainer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        from cogsyndelta.optimization.algebraic_training import UnifiedAlgebraicTrainer

        self.model = SimpleNetwork(input_dim=32, hidden_dim=64, output_dim=8)
        self.trainer = UnifiedAlgebraicTrainer(self.model)
        self.train_x = torch.randn(50, 32)
        self.train_y = torch.randn(50, 8)

    def test_train_algebraically(self) -> None:
        """Test full algebraic training."""
        results = self.trainer.train_algebraically(
            self.train_x, self.train_y,
            target_epochs=10,
            apply_weights=True,
        )

        # Should have comprehensive results
        self.assertIn("weight_statistics", results)
        self.assertIn("convergence_analysis", results)

    def test_quick_optimize(self) -> None:
        """Test quick optimization."""
        results = self.trainer.quick_optimize(
            self.train_x, self.train_y,
            num_natural_steps=3,
        )

        self.assertIn("initial_loss", results)
        self.assertIn("final_loss", results)
        self.assertIn("loss_reduction", results)

    def test_analyze_trainability(self) -> None:
        """Test trainability analysis."""
        analysis = self.trainer.analyze_trainability(self.train_x, self.train_y)

        self.assertIn("trainability_score", analysis)
        self.assertIn("total_parameters", analysis)
        self.assertIn("outlier_ratio", analysis)
        self.assertIn("recommended_learning_rate", analysis)

        # Trainability score should be in [0, 1]
        self.assertGreaterEqual(analysis["trainability_score"], 0.0)
        self.assertLessEqual(analysis["trainability_score"], 1.0)


class TestAlgebraicVsBackprop(unittest.TestCase):
    """Compare algebraic optimization to actual backprop training."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.input_dim = 16
        self.hidden_dim = 32
        self.output_dim = 4

        # Simple linear regression problem for best comparison
        torch.manual_seed(42)
        self.train_x = torch.randn(100, self.input_dim)
        # Linear target for optimal comparison
        true_weights = torch.randn(self.input_dim, self.output_dim) * 0.5
        self.train_y = self.train_x @ true_weights

    def test_spectral_approximates_backprop(self) -> None:
        """Test that spectral prediction approximates backprop result."""
        from cogsyndelta.optimization.algebraic_training import SpectralWeightPredictor

        # Simple linear model
        model = nn.Linear(self.input_dim, self.output_dim, bias=False)
        predictor = SpectralWeightPredictor(model)

        # Algebraic prediction
        optimal_weights = predictor.predict_layer_weights(
            self.train_x, self.train_y, (self.input_dim, self.output_dim)
        )

        # Apply predicted weights
        with torch.no_grad():
            model.weight.copy_(optimal_weights)

        # Measure performance
        with torch.no_grad():
            algebraic_output = model(self.train_x)
            algebraic_mse = F.mse_loss(algebraic_output, self.train_y).item()

        # Train with backprop for comparison
        model_backprop = nn.Linear(self.input_dim, self.output_dim, bias=False)
        optimizer = torch.optim.SGD(model_backprop.parameters(), lr=0.01)

        for _ in range(100):
            optimizer.zero_grad()
            output = model_backprop(self.train_x)
            loss = F.mse_loss(output, self.train_y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            backprop_output = model_backprop(self.train_x)
            backprop_mse = F.mse_loss(backprop_output, self.train_y).item()

        # Algebraic should be competitive (within 2x of backprop for linear case)
        # For linear regression, closed-form is actually OPTIMAL
        self.assertLess(algebraic_mse, backprop_mse * 2,
                       f"Algebraic MSE {algebraic_mse:.6f} should be close to "
                       f"backprop MSE {backprop_mse:.6f}")


if __name__ == "__main__":
    unittest.main()
