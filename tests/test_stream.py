"""Stream sources: real corpus embeddings, and the fallback that must never be silent.

The corpus tests skip when the fleet export is not mounted -- CI runners will never have
it. What CI still checks here is the part that does not need data: that the fallback is
reachable only by name, that it labels itself, and that asking for a corpus which is not
there raises a message pointing at the mount rather than quietly handing back noise.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from cogsyndelta.data import stream as stream_mod
from cogsyndelta.data.stream import (
    CorpusStream,
    StreamSource,
    SyntheticStream,
    corpus_available,
    resolve_stream,
)

needs_corpus = pytest.mark.skipif(
    not (corpus_available() and importlib.util.find_spec("tokenizers")),
    reason="fleet dataset export not mounted, or the `train` dependency group is absent",
)


def test_synthetic_is_labelled_and_shaped() -> None:
    """The fallback announces itself in provenance; nothing else has to remember to."""
    src = SyntheticStream(16, "cpu", seed=1)
    assert isinstance(src, StreamSource)
    assert "synthetic-fallback" in src.provenance
    assert src.sample(4).shape == (4, 16)


def test_synthetic_is_reproducible_from_its_seed() -> None:
    """Same seed, same batch: a fallback that drifts is not even useful as a smoke."""
    a = SyntheticStream(8, "cpu", seed=7).sample(5)
    b = SyntheticStream(8, "cpu", seed=7).sample(5)
    assert torch.equal(a, b)


def test_missing_corpus_names_the_mount_and_the_opt_out(tmp_path, monkeypatch) -> None:
    """No corpus must raise, and the message must say how to ask for noise on purpose."""
    monkeypatch.setattr(stream_mod, "DEFAULT_DATASETS_ROOT", tmp_path / "absent")
    assert not corpus_available()
    with pytest.raises(FileNotFoundError, match="CSD_DATASETS_ROOT"):
        resolve_stream("corpus", 16, "cpu")


def test_unknown_spec_is_rejected() -> None:
    """A typo must not fall through to whichever branch happens to be last."""
    with pytest.raises(ValueError, match="unknown stream spec"):
        resolve_stream("random", 16, "cpu")


@needs_corpus
def test_corpus_stream_shape_and_provenance() -> None:
    """Real text gives a [B, D] batch on the requested device, tagged with its origin."""
    src = CorpusStream(32, "cpu", texts=64, seed=0)
    batch = src.sample(8)
    assert batch.shape == (8, 32)
    assert src.provenance.startswith("corpus:code:docstring:")
    assert "random-init" in src.provenance


@needs_corpus
def test_corpus_embeddings_are_anisotropic_unlike_noise() -> None:
    """The property that makes this worth doing.

    A per-dimension min/max quantiser measured on isotropic noise is measured near its
    best case. Real embeddings share a dominant direction -- mean pairwise cosine an order
    of magnitude above Gaussian -- and that is the regime the codec actually has to work
    in. If this assertion ever fails, the "real" stream has collapsed toward noise and
    every number measured on it is back to being about the RNG.
    """
    pool = torch.nn.functional.normalize(CorpusStream(64, "cpu", texts=128, seed=0).pool, dim=-1)
    off_diagonal = ~torch.eye(pool.size(0), dtype=torch.bool)
    real = float((pool @ pool.T)[off_diagonal].mean())

    noise = torch.nn.functional.normalize(torch.randn(128, 64, generator=torch.Generator()), dim=-1)
    gaussian = float((noise @ noise.T)[off_diagonal].mean())

    assert real > 0.3, f"corpus stream is nearly isotropic ({real:.3f}); it has collapsed"
    assert real > 10 * abs(gaussian), f"real={real:.3f} gaussian={gaussian:.3f}"
