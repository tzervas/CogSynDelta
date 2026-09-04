"""Train path: one region, loss must decrease.

Two streams are exercised on purpose. The corpus test is the one that means something --
it measures a VAE against embeddings of real text. The fallback test only proves the code
path runs where there is no NFS export, which is every CI runner, and it is named so that
its number is never mistaken for a result.
"""

from __future__ import annotations

import importlib.util

import pytest

from cogsyndelta.contracts.config import TrainConfig
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.data.stream import CorpusStream, SyntheticStream, corpus_available
from cogsyndelta.poc.train import train_latent_vae

needs_corpus = pytest.mark.skipif(
    not (corpus_available() and importlib.util.find_spec("tokenizers")),
    reason="fleet dataset export not mounted, or the `train` dependency group is absent",
)


def _cfg() -> TrainConfig:
    return TrainConfig(steps=40, batch_size=32, hidden_dim=64, latent_dim=8, learning_rate=1e-2)


def test_train_loss_decreases_synthetic_fallback() -> None:
    """CI-only path: the loop runs and the loss falls on the explicit noise fallback."""
    cfg = _cfg()
    ctx = DeviceContext.resolve("cpu")
    stream = SyntheticStream(cfg.input_dim, ctx.device, seed=123)
    result = train_latent_vae(cfg, ctx, seed=123, stream=stream)
    assert result["stream"].startswith("synthetic-fallback")
    assert result["last_loss"] < result["first_loss"], (
        f"first={result['first_loss']:.4f} last={result['last_loss']:.4f}"
    )


@needs_corpus
def test_train_loss_decreases_on_corpus() -> None:
    """The result that counts: ELBO falls on embeddings of real corpus text."""
    cfg = _cfg()
    ctx = DeviceContext.resolve("auto")
    stream = CorpusStream(cfg.input_dim, ctx.device, seed=123)
    result = train_latent_vae(cfg, ctx, seed=123, stream=stream)
    assert result["stream"].startswith("corpus:")
    assert result["last_loss"] < result["first_loss"], (
        f"first={result['first_loss']:.4f} last={result['last_loss']:.4f}"
    )


def test_stream_width_mismatch_is_rejected() -> None:
    """A stream narrower than the region must fail loudly, not broadcast into silence."""
    cfg = _cfg()
    ctx = DeviceContext.resolve("cpu")
    with pytest.raises(ValueError, match="stream dim"):
        train_latent_vae(cfg, ctx, stream=SyntheticStream(cfg.input_dim // 2, ctx.device))
