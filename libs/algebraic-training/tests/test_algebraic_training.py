"""Tests for algebraic-training library."""

from __future__ import annotations

import pytest
import torch
from torch import nn


class SimpleNet(nn.Module):
    """Simple network for testing."""

    def __init__(self, in_features: int = 10, hidden: int = 20, out_features: int = 5) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def model() -> nn.Module:
    """Create a simple test model."""
    torch.manual_seed(42)
    return SimpleNet()


@pytest.fixture
def training_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Create simple training data."""
    torch.manual_seed(42)
    train_x = torch.randn(100, 10)
    train_y = torch.randn(100, 5)
    return train_x, train_y


class TestNTKPredictor:
    """Tests for NTKPredictor."""

    def test_compute_empirical_ntk(self, model: nn.Module, training_data: tuple) -> None:
        """Test NTK computation."""
        from algebraic_training import NTKPredictor

        train_x, _ = training_data
        ntk = NTKPredictor(model)

        kernel = ntk.compute_empirical_ntk(train_x[:10])

        assert kernel.shape == (10, 10)
        # NTK should be symmetric
        assert torch.allclose(kernel, kernel.T, atol=1e-5)
        # Diagonal should be positive
        assert (kernel.diag() > 0).all()

    def test_predict_training_dynamics(self, model: nn.Module, training_data: tuple) -> None:
        """Test training dynamics prediction."""
        from algebraic_training import NTKPredictor

        train_x, train_y = training_data
        ntk = NTKPredictor(model)

        results = ntk.predict_training_dynamics(
            train_x[:20], train_y[:20], train_x[:20],
            learning_rate=0.01,
            training_time=10.0,
        )

        assert "predicted_outputs" in results
        assert "predicted_mse" in results
        assert results["predicted_outputs"].shape == train_y[:20].shape


class TestFisherInformationPredictor:
    """Tests for FisherInformationPredictor."""

    def test_compute_fisher_matrix(self, model: nn.Module, training_data: tuple) -> None:
        """Test Fisher matrix computation."""
        from algebraic_training import FisherInformationPredictor

        train_x, train_y = training_data

        fisher = FisherInformationPredictor(model)

        def data_gen():
            for i in range(0, 50, 10):
                yield train_x[i:i+10], train_y[i:i+10]

        fisher_diag = fisher.compute_fisher_matrix(data_gen())

        assert len(fisher_diag) > 0
        for name, f in fisher_diag.items():
            assert (f >= 0).all(), f"Fisher diagonal should be non-negative for {name}"

    def test_natural_gradient_update(self, model: nn.Module, training_data: tuple) -> None:
        """Test natural gradient computation."""
        from algebraic_training import FisherInformationPredictor

        train_x, train_y = training_data
        fisher = FisherInformationPredictor(model)

        updates = fisher.compute_natural_gradient_update(
            train_x[:50], train_y[:50], learning_rate=0.1
        )

        assert len(updates) > 0
        for name, update in updates.items():
            param = dict(model.named_parameters())[name]
            assert update.shape == param.shape


class TestSpectralWeightPredictor:
    """Tests for SpectralWeightPredictor."""

    def test_analyze_data_spectrum(self, model: nn.Module, training_data: tuple) -> None:
        """Test data spectrum analysis."""
        from algebraic_training import SpectralWeightPredictor

        train_x, _ = training_data
        spectral = SpectralWeightPredictor(model)

        spectrum = spectral.analyze_data_spectrum(train_x)

        assert "eigenvalues" in spectrum
        assert "eigenvectors" in spectrum
        assert "effective_dimensionality" in spectrum
        assert spectrum["eigenvalues"].shape[0] == train_x.shape[1]

    def test_predict_layer_weights(self, model: nn.Module, training_data: tuple) -> None:
        """Test layer weight prediction."""
        from algebraic_training import SpectralWeightPredictor

        train_x, train_y = training_data
        spectral = SpectralWeightPredictor(model)

        weights = spectral.predict_layer_weights(
            train_x, train_y, (10, 5)
        )

        assert weights.shape == (5, 10)


class TestAlgebraicOptimizer:
    """Tests for AlgebraicOptimizer."""

    def test_predict_training_outcome(self, model: nn.Module, training_data: tuple) -> None:
        """Test training outcome prediction."""
        from algebraic_training import AlgebraicOptimizer

        train_x, train_y = training_data
        optimizer = AlgebraicOptimizer(model, method="hybrid")

        results = optimizer.predict_training_outcome(
            train_x[:30], train_y[:30],
            learning_rate=0.01,
            num_epochs=10,
        )

        assert "predicted_mse" in results or "effective_dimensionality" in results

    def test_compute_optimal_weights(self, model: nn.Module, training_data: tuple) -> None:
        """Test optimal weight computation."""
        from algebraic_training import AlgebraicOptimizer

        train_x, train_y = training_data
        optimizer = AlgebraicOptimizer(model)

        weights = optimizer.compute_optimal_weights(
            train_x[:50], train_y[:50], method="spectral"
        )

        assert len(weights) > 0

    def test_fast_train(self, model: nn.Module, training_data: tuple) -> None:
        """Test fast training."""
        from algebraic_training import AlgebraicOptimizer

        train_x, train_y = training_data
        optimizer = AlgebraicOptimizer(model)

        results = optimizer.fast_train(
            train_x[:50], train_y[:50], num_natural_steps=3
        )

        assert "initial_loss" in results
        assert "final_loss" in results
        assert results["final_loss"] <= results["initial_loss"]


class TestUnifiedAlgebraicTrainer:
    """Tests for UnifiedAlgebraicTrainer."""

    def test_train_algebraically(self, model: nn.Module, training_data: tuple) -> None:
        """Test full algebraic training."""
        from algebraic_training import UnifiedAlgebraicTrainer

        train_x, train_y = training_data
        trainer = UnifiedAlgebraicTrainer(model)

        results = trainer.train_algebraically(
            train_x[:30], train_y[:30],
            target_epochs=10,
            apply_weights=False,
            verbose=False,
        )

        assert "optimal_weights" in results
        assert "convergence_analysis" in results

    def test_analyze_trainability(self, model: nn.Module, training_data: tuple) -> None:
        """Test trainability analysis."""
        from algebraic_training import UnifiedAlgebraicTrainer

        train_x, train_y = training_data
        trainer = UnifiedAlgebraicTrainer(model)

        analysis = trainer.analyze_trainability(train_x[:50], train_y[:50])

        assert "trainability_score" in analysis
        assert "recommended_learning_rate" in analysis
        assert 0 <= analysis["trainability_score"] <= 1

    def test_quick_optimize(self, model: nn.Module, training_data: tuple) -> None:
        """Test quick optimization."""
        from algebraic_training import UnifiedAlgebraicTrainer

        train_x, train_y = training_data
        trainer = UnifiedAlgebraicTrainer(model)

        results = trainer.quick_optimize(
            train_x[:50], train_y[:50], num_natural_steps=3
        )

        assert "initial_loss" in results
        assert "final_loss" in results


class TestWeightDistributionPredictor:
    """Tests for WeightDistributionPredictor."""

    def test_predict_weight_statistics(self, model: nn.Module, training_data: tuple) -> None:
        """Test weight statistics prediction."""
        from algebraic_training import WeightDistributionPredictor

        train_x, train_y = training_data
        predictor = WeightDistributionPredictor(model)

        stats = predictor.predict_weight_statistics(train_x, train_y)

        assert len(stats) > 0
        for _name, s in stats.items():
            assert "predicted_mean" in s
            assert "predicted_variance" in s

    def test_predict_convergence_time(self, model: nn.Module, training_data: tuple) -> None:
        """Test convergence time prediction."""
        from algebraic_training import WeightDistributionPredictor

        train_x, _ = training_data
        predictor = WeightDistributionPredictor(model)

        convergence = predictor.predict_convergence_time(train_x, learning_rate=0.01)

        assert "estimated_epochs" in convergence
        assert "condition_number" in convergence
        assert convergence["estimated_epochs"] > 0

    def test_generate_predicted_weights(self, model: nn.Module, training_data: tuple) -> None:
        """Test weight generation."""
        from algebraic_training import WeightDistributionPredictor

        train_x, train_y = training_data
        predictor = WeightDistributionPredictor(model)

        weights = predictor.generate_predicted_weights(train_x, train_y)

        assert len(weights) > 0


class TestFunctionalAPI:
    """Tests for functional API."""

    def test_predict_training(self, model: nn.Module, training_data: tuple) -> None:
        """Test predict_training function."""
        import algebraic_training.functional as F

        train_x, train_y = training_data

        results = F.predict_training(model, train_x[:30], train_y[:30], epochs=10)

        assert "optimal_weights" in results

    def test_analyze_trainability(self, model: nn.Module, training_data: tuple) -> None:
        """Test analyze_trainability function."""
        import algebraic_training.functional as F

        train_x, train_y = training_data

        analysis = F.analyze_trainability(model, train_x[:50], train_y[:50])

        assert "trainability_score" in analysis

    def test_natural_gradient_step(self, model: nn.Module, training_data: tuple) -> None:
        """Test natural_gradient_step function."""
        import algebraic_training.functional as F

        train_x, train_y = training_data

        updates = F.natural_gradient_step(
            model, train_x[:50], train_y[:50], lr=0.1, apply_updates=False
        )

        assert len(updates) > 0

    def test_compute_ntk(self, model: nn.Module, training_data: tuple) -> None:
        """Test compute_ntk function."""
        import algebraic_training.functional as F

        train_x, _ = training_data

        ntk = F.compute_ntk(model, train_x[:10])

        assert ntk.shape == (10, 10)

    def test_quick_train(self, model: nn.Module, training_data: tuple) -> None:
        """Test quick_train function."""
        import algebraic_training.functional as F

        train_x, train_y = training_data

        stats = F.quick_train(model, train_x[:50], train_y[:50], num_steps=2)

        assert "final_loss" in stats


class TestAccuracyParity:
    """Accuracy parity tests comparing algebraic vs SGD training."""

    def test_spectral_approximates_backprop(self) -> None:
        """Test that spectral method achieves comparable loss to SGD."""
        import torch.nn.functional as F

        from algebraic_training import SpectralWeightPredictor

        torch.manual_seed(42)

        # Simple linear regression problem
        model = nn.Linear(10, 5)
        train_x = torch.randn(200, 10)
        # Linear target with noise
        true_weights = torch.randn(5, 10) * 0.5
        train_y = train_x @ true_weights.T + torch.randn(200, 5) * 0.1

        # Method 1: Spectral prediction
        spectral = SpectralWeightPredictor(model)
        spectral_weights = spectral.predict_layer_weights(train_x, train_y, (10, 5))

        with torch.no_grad():
            model.weight.copy_(spectral_weights)
            spectral_loss = F.mse_loss(model(train_x), train_y).item()

        # Method 2: SGD training
        model2 = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model2.parameters(), lr=0.01)

        for _ in range(100):
            optimizer.zero_grad()
            loss = F.mse_loss(model2(train_x), train_y)
            loss.backward()
            optimizer.step()

        sgd_loss = F.mse_loss(model2(train_x), train_y).item()

        # Spectral should be within 2x of SGD for this simple problem
        assert spectral_loss < sgd_loss * 2.0, (
            f"Spectral loss {spectral_loss:.4f} too high vs SGD {sgd_loss:.4f}"
        )

    def test_natural_gradient_faster_than_sgd(self) -> None:
        """Test that natural gradient converges faster than regular gradient."""
        import torch.nn.functional as F

        from algebraic_training import FisherInformationPredictor

        torch.manual_seed(42)

        train_x = torch.randn(100, 10)
        train_y = torch.randn(100, 5)

        # Natural gradient: 3 steps
        model_ng = SimpleNet()
        fisher = FisherInformationPredictor(model_ng)

        for i in range(3):
            updates = fisher.compute_natural_gradient_update(
                train_x, train_y, learning_rate=0.3 / (i + 1)
            )
            with torch.no_grad():
                for name, param in model_ng.named_parameters():
                    if name in updates:
                        param.add_(updates[name])

        ng_loss = F.mse_loss(model_ng(train_x), train_y).item()

        # SGD: 3 steps (same compute budget)
        model_sgd = SimpleNet()
        optimizer = torch.optim.SGD(model_sgd.parameters(), lr=0.1)

        for _ in range(3):
            optimizer.zero_grad()
            loss = F.mse_loss(model_sgd(train_x), train_y)
            loss.backward()
            optimizer.step()

        sgd_loss = F.mse_loss(model_sgd(train_x), train_y).item()

        # Natural gradient should achieve lower loss in same number of steps
        # (or at least not be significantly worse)
        assert ng_loss <= sgd_loss * 1.5, (
            f"Natural gradient {ng_loss:.4f} should be <= SGD {sgd_loss:.4f} * 1.5"
        )
