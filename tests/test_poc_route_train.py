"""PoC-3: train softmax router with Switch aux load-balance.

The DoD assertions (loss falls, aux finite, load not collapsed) are made twice: once on
embeddings of real corpus text, which is the claim, and once on the explicit noise
fallback, which is only what a CI runner with no NFS export can check. The fallback
fixture is named so its numbers cannot be quoted as a result.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any

import pytest
import torch

from cogsyndelta.contracts.config import RouteTrainConfig
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.data.stream import CorpusStream, SyntheticStream, corpus_available
from cogsyndelta.poc.router import switch_aux_load_balance
from cogsyndelta.poc.train_route import train_softmax_router
from cogsyndelta.poc.vae import LatentVAE

needs_corpus = pytest.mark.skipif(
    not (corpus_available() and importlib.util.find_spec("tokenizers")),
    reason="fleet dataset export not mounted, or the `train` dependency group is absent",
)


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
def trained_fallback() -> dict[str, Any]:
    """Noise-stream run. Proves the loop runs where there is no corpus; proves nothing else."""
    cfg = _cpu_train_cfg()
    ctx = DeviceContext.resolve("cpu")
    return train_softmax_router(
        cfg, ctx, seed=42, stream_source=SyntheticStream(cfg.stream_dim, ctx.device, seed=42)
    )


@pytest.fixture(scope="module")
def trained_corpus() -> dict[str, Any]:
    """Real-embedding run. This is the one whose numbers mean something."""
    cfg = _cpu_train_cfg()
    ctx = DeviceContext.resolve("auto")
    return train_softmax_router(
        cfg, ctx, seed=42, stream_source=CorpusStream(cfg.stream_dim, ctx.device, seed=42)
    )


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


def _assert_dod(result: dict[str, Any]) -> None:
    """The PoC-3 definition of done: recon falls, aux stays finite, load does not collapse."""
    assert result["last_loss"] < result["first_loss"], (
        f"first={result['first_loss']:.4f} last={result['last_loss']:.4f}"
    )
    assert math.isfinite(result["first_aux_lb"])
    assert math.isfinite(result["last_aux_lb"])
    assert all(math.isfinite(v) for v in result["aux_lbs"])
    for key in ("load", "train_load"):
        load = result[key]
        assert set(load) == {"residual_mlp", "stream_vae"}
        values = {round(v, 6) for v in load.values()}
        assert values != {0.0, 1.0}, f"{key}={load}"
        assert all(v > 0.0 for v in load.values()), f"{key}={load}"


def test_train_route_dod_synthetic_fallback(trained_fallback: dict[str, Any]) -> None:
    """CI-only path: the loop runs and holds the DoD on the explicit noise fallback."""
    assert trained_fallback["stream"].startswith("synthetic-fallback")
    _assert_dod(trained_fallback)


@needs_corpus
def test_train_route_dod_on_corpus(trained_corpus: dict[str, Any]) -> None:
    """The result that counts: the DoD holds when the routed target is real text."""
    assert trained_corpus["stream"].startswith("corpus:")
    _assert_dod(trained_corpus)


def test_stream_width_mismatch_is_rejected() -> None:
    """A stream narrower than the gate must fail loudly rather than reshape itself."""
    cfg = _cpu_train_cfg()
    ctx = DeviceContext.resolve("cpu")
    with pytest.raises(ValueError, match="stream dim"):
        train_softmax_router(
            cfg, ctx, stream_source=SyntheticStream(cfg.stream_dim + 1, ctx.device)
        )


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
