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


# eval_batch_size was 8 until 2026-09-04. That matched batch_size, but it measured a
# Binomial(n=8, p) statistic at the corpus/synthetic streams' historically-measured
# ~0.125 minority-routing rate (see poc/train_route.py's evaluate_route_load
# docstring): P(zero minority hits in 8 draws) = 0.875**8 ~= 34%. That is not
# "collapsed load" once in three fair runs -- it is this test's DoD assertion having
# no safety margin at all, so it flips on any perturbation that moves even one
# borderline token across the gate's decision boundary: ci_local.sh (the pre-push
# gate) syncs CUDA torch and lets trained_corpus's DeviceContext.resolve("auto") pick
# the GPU, while a plain interactive `pytest` run may stay on CPU -- and torch does
# not promise bit-identical reductions across that difference even at a fixed seed,
# only run-to-run repeatability on a given device with deterministic algorithms
# enabled (which this test does not request, since GPU-vs-CPU divergence is the
# thing being defended against here, not intra-device repeatability).
#
# MEASURED (CPU, seed 42, this config, both streams): eval_batch_size=8 gave exactly
# 0.125/0.875 every run, no variance -- CPU alone is fully deterministic here, so the
# two historical pre-push failures are consistent with the GPU path, not a code bug.
# Widening to 64 settles at a stable ~0.34/0.66 (corpus) / ~0.13/0.87 (synthetic),
# P(zero minority hits) ~= 0.0002 -- see test_eval_batch_size_has_collapse_safety_margin
# below, which pins this reasoning down as a regression guard.
def _cpu_train_cfg() -> RouteTrainConfig:
    return RouteTrainConfig(
        steps=24,
        batch_size=8,
        eval_batch_size=64,
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


def test_eval_batch_size_has_collapse_safety_margin() -> None:
    """Regression guard for the 2026-09-04 flake fix (see _cpu_train_cfg's comment).

    A collapsed eval load can mean the aux term failed, or it can mean an eval batch
    too small to distinguish a real collapse from binomial sampling noise. This pins
    the second failure mode shut: at the ~0.125 measured minority-routing floor (the
    lowest nonzero rate seen across the corpus and synthetic streams), the configured
    eval_batch_size must keep P(an honest run reports zero minority hits) under 1%.
    0.125 is not re-derived here -- that would mean training a router inside this
    guard, defeating its purpose as a fast, static check -- it is the floor recorded in
    poc/train_route.py's evaluate_route_load docstring and reproduced by the probe
    that motivated this branch.
    """
    cfg = _cpu_train_cfg()
    measured_minority_rate_floor = 0.125
    p_false_collapse = (1 - measured_minority_rate_floor) ** cfg.eval_batch_size
    assert p_false_collapse < 0.01, (
        f"eval_batch_size={cfg.eval_batch_size} gives P(false collapse)="
        f"{p_false_collapse:.4f} at the measured {measured_minority_rate_floor} "
        f"minority-routing floor -- too small a sample to tell a real router collapse "
        f"from binomial noise (see test_train_route_dod_on_corpus's flake history)"
    )


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
