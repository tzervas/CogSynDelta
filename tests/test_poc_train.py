"""Train path: one region, loss must decrease."""

from __future__ import annotations

from cogsyndelta.contracts.config import TrainConfig
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.poc.train import train_latent_vae


def test_train_loss_decreases() -> None:
    cfg = TrainConfig(
        steps=40, batch_size=32, hidden_dim=64, latent_dim=8, learning_rate=1e-2
    )
    ctx = DeviceContext.resolve("cpu")
    result = train_latent_vae(cfg, ctx, seed=123)
    assert result["last_loss"] < result["first_loss"], (
        f"first={result['first_loss']:.4f} last={result['last_loss']:.4f}"
    )
