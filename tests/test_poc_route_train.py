"""PoC-3: train softmax router with Switch aux load-balance."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from cogsyndelta.contracts.config import RouteTrainConfig
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.poc.router import switch_aux_load_balance
from cogsyndelta.poc.train_route import train_softmax_router
from cogsyndelta.poc.vae import LatentVAE


def _cpu_train_cfg() -> RouteTrainConfig:
    return RouteTrainConfig(
        steps=24,
        batch_size=8,
        eval_batch_size=8,
        hidden_dim=32,
        stream_dim=16,
        latent_dim=4,
        learning_rate=1e-2,
        aux_coef=1.0,
    )


@pytest.fixture(scope="module")
def trained_cpu() -> dict[str, Any]:
    return train_softmax_router(_cpu_train_cfg(), DeviceContext.resolve("cpu"), seed=42)


def test_switch_aux_uniform_is_one() -> None:
    weights = torch.full((8, 2), 0.5)
    topk = torch.tensor([[0], [1], [0], [1], [0], [1], [0], [1]])
    aux = switch_aux_load_balance(weights, topk)
    assert torch.isfinite(aux)
    assert abs(float(aux.item()) - 1.0) < 1e-5


def test_switch_aux_collapse_is_n() -> None:
    weights = torch.tensor([[1.0, 0.0]] * 8)
    topk = torch.zeros(8, 1, dtype=torch.long)
    aux = switch_aux_load_balance(weights, topk)
    assert abs(float(aux.item()) - 2.0) < 1e-5


def test_train_route_loss_decreases(trained_cpu: dict[str, Any]) -> None:
    assert trained_cpu["last_loss"] < trained_cpu["first_loss"], (
        f"first={trained_cpu['first_loss']:.4f} last={trained_cpu['last_loss']:.4f}"
    )


def test_aux_load_balance_is_finite(trained_cpu: dict[str, Any]) -> None:
    assert math.isfinite(trained_cpu["first_aux_lb"])
    assert math.isfinite(trained_cpu["last_aux_lb"])
    assert all(math.isfinite(v) for v in trained_cpu["aux_lbs"])


def test_load_not_collapsed_batch_8(trained_cpu: dict[str, Any]) -> None:
    for key in ("load", "train_load"):
        load = trained_cpu[key]
        assert set(load) == {"residual_mlp", "stream_vae"}
        values = {round(v, 6) for v in load.values()}
        assert values != {0.0, 1.0}, f"{key}={load}"
        assert all(v > 0.0 for v in load.values()), f"{key}={load}"


def test_latent_vae_forward_still_elbo_tuple() -> None:
    torch.manual_seed(0)
    model = LatentVAE(input_dim=16, hidden_dim=32, latent_dim=4)
    x = torch.randn(5, 16)
    out = model(x)
    assert isinstance(out, tuple)
    assert len(out) == 3
    recon, mu, logvar = out
    assert recon.shape == x.shape
    assert mu.shape == (5, 4)
    assert logvar.shape == (5, 4)
