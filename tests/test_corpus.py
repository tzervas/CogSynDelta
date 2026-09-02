"""Corpus pipeline tests.

These skip when the fleet dataset export is not mounted. That is deliberate and it is the
honest option: the corpus is a 649 GB NFS export from another host, so CI runners will
never have it. The alternative -- mocking parquet and the tokenizer -- would assert that
the mocks agree with each other and tell us nothing about whether the real path works.

What CI *does* still check here is that the module imports and that the failure mode for
a missing corpus is a clear error rather than an empty iterator, which is the bug this
guard exists to prevent.
"""

from __future__ import annotations

import pytest
import torch

# These MUST precede the corpus import: cogsyndelta.data.corpus imports pyarrow and
# tokenizers at module scope, so without the train group installed the import below
# raises during collection and the whole file errors instead of skipping.
pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

from cogsyndelta.data import corpus as corpus_mod

_HAVE_CORPUS = corpus_mod.DEFAULT_CORPUS_ROOT.is_dir()
needs_corpus = pytest.mark.skipif(
    not _HAVE_CORPUS,
    reason=f"corpus export not mounted at {corpus_mod.DEFAULT_CORPUS_ROOT}",
)


def test_missing_corpus_root_raises_clearly(tmp_path, monkeypatch) -> None:
    """A missing mount must raise, not yield nothing.

    Globbing a non-existent directory returns an empty list, so without this guard an
    unmounted export presents as "the corpus has no shards" -- which sends you looking at
    the data instead of at `findmnt`.
    """
    monkeypatch.setattr(corpus_mod, "DEFAULT_CORPUS_ROOT", tmp_path / "definitely-absent")
    with pytest.raises(FileNotFoundError, match="not present"):
        corpus_mod.corpus_root()


def test_unknown_names_raise() -> None:
    with pytest.raises(KeyError):
        corpus_mod.shard_paths("no-such-corpus")
    with pytest.raises(KeyError):
        corpus_mod.load_tokenizer("no-such-tokenizer")


@needs_corpus
def test_tokenizer_round_trips() -> None:
    tok = corpus_mod.load_tokenizer("gpt2")
    assert tok.get_vocab_size() > 50_000
    text = "The cat sat on the mat."
    assert tok.decode(tok.encode(text).ids).strip() == text.strip()


@needs_corpus
def test_shards_are_discovered_and_sorted() -> None:
    shards = corpus_mod.shard_paths("tinystories")
    assert shards, "expected parquet shards"
    assert shards == sorted(shards), "shard order must be stable across hosts"


@needs_corpus
def test_windows_have_exact_width() -> None:
    tok = corpus_mod.load_tokenizer("gpt2")
    seq_len = 128
    for window in corpus_mod.iter_token_windows(
        "tinystories", tok, seq_len=seq_len, shuffle_buffer=0, limit_windows=5
    ):
        assert window.shape == (seq_len + 1,)
        assert window.dtype == torch.long


@needs_corpus
def test_batches_are_shifted_by_one() -> None:
    """targets must be inputs shifted one position -- the causal LM contract.

    Getting this off by one trains the model to predict the token it was just given,
    which produces a loss curve that falls beautifully and a model that has learned
    nothing.
    """
    tok = corpus_mod.load_tokenizer("gpt2")
    batches = 0
    for inputs, targets in corpus_mod.iter_batches(
        "tinystories", tok, batch_size=4, seq_len=64, shuffle_buffer=0, limit_windows=12
    ):
        assert inputs.shape == (4, 64)
        assert targets.shape == (4, 64)
        assert torch.equal(inputs[:, 1:], targets[:, :-1])
        batches += 1
    assert batches >= 1


@needs_corpus
def test_same_seed_gives_same_windows() -> None:
    """Reproducibility: a checkpoint is only comparable if its data order is."""
    tok = corpus_mod.load_tokenizer("gpt2")

    def take(seed: int) -> list[torch.Tensor]:
        return list(
            corpus_mod.iter_token_windows(
                "tinystories", tok, seq_len=64, seed=seed, shuffle_buffer=64, limit_windows=8
            )
        )

    a, b, c = take(0), take(0), take(1)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert not all(torch.equal(x, y) for x, y in zip(a, c, strict=True))
