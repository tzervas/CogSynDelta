"""
Shared pytest fixtures for CogSynDelta test suite.

Provides common fixtures for:
- Device configuration (CPU/GPU)
- Configuration loading
- Mock integrated systems
- Test embeddings and tensors
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Provide torch device - always CPU for CI compatibility."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def config() -> dict[str, Any]:
    """Provide default test configuration."""
    return {
        "embed_dim": 512,
        "latent_dim": 128,
        "num_improvement_iterations": 3,
        "security_enabled": True,
        "qa_enabled": True,
        "multi_language": True,
        "exploratory": {"k": 5},
        "meta_optimization": {"num_inner_steps": 3, "inner_lr": 0.01},
        "training": {"batch_size": 32},
    }


@pytest.fixture
def test_embeddings(device: torch.device) -> torch.Tensor:
    """Provide test embeddings tensor."""
    torch.manual_seed(42)
    return torch.randn(4, 512, device=device)


@pytest.fixture
def test_tokens(device: torch.device) -> torch.Tensor:
    """Provide test token tensor."""
    torch.manual_seed(42)
    return torch.randint(0, 1000, (2, 100), device=device)


@pytest.fixture
def mock_integrated_system() -> MagicMock:
    """Provide mock integrated system for API tests."""
    mock = MagicMock()
    mock.process_visual.return_value = {"status": "success", "features": [0.1, 0.2, 0.3]}
    mock.generate_code.return_value = {"code": "print('hello')", "quality": 0.95}
    mock.query_memory.return_value = {"results": [], "count": 0}
    return mock


@pytest.fixture
def sample_message_payload() -> dict[str, Any]:
    """Provide sample A2A message payload."""
    return {
        "message": "Test message content",
        "context": {"task": "code_generation", "language": "python"},
    }


@pytest.fixture(autouse=True)
def set_random_seeds() -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)


@pytest.fixture
def temp_model_path(tmp_path: Path) -> Path:
    """Provide temporary path for model saving tests."""
    return tmp_path / "test_model.pt"
