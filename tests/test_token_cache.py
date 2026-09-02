"""The pre-tokenised corpus cache must produce byte-identical batches to `_tokenize`.

A tokenisation cache is a pure speed change with two ways to be silently wrong, and both
produce a run that finishes and writes a plausible receipt:

  - **Different padding.** `_tokenize` pads to the longest item IN THE BATCH. Padding the
    whole corpus to `max_len` up front is the obvious implementation and would change
    every batch the model ever sees -- on `compress` it would take the mean padded width
    from 32.9 to 96 and roughly triple its GPU cost. The batch-equality tests below are
    the real gate on this change; a timing win means nothing if the batches moved.
  - **A stale cache.** Training on ids from a corpus that has since changed cannot be
    seen in the loss curve or the receipt. The key is content-addressed, and the tests
    here check that every input which changes the ids also changes the key, and that a
    payload found under the wrong name is refused rather than adopted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

# MUST precede the imports below: cogsyndelta.regions.pretrain imports tokenizers at
# module scope, so without the train group installed the import errors during collection
# and the whole file fails instead of skipping (same reasoning as
# tests/test_build_splits_shuffle.py).
pytest.importorskip("tokenizers", reason="train group not installed")

from tokenizers import Tokenizer, models, pre_tokenizers

from cogsyndelta.regions._checkpoint import atomic_save
from cogsyndelta.regions._tokencache import (
    SCHEMA,
    RaggedTokens,
    cache_key,
    corpus_token_cache,
    encode_ragged,
)
from cogsyndelta.regions.pretrain import _tokenize

pytestmark = pytest.mark.cpu

_MAX_LEN = 8
_REAL_TOKENIZER = Path("/mnt/fleet-datasets/tritter/gpt2_tokenizer.json")

# Deliberately uneven lengths, and one empty string: `_tokenize`'s `max(1, ...)` and its
# `if seq:` guard are both only exercised by a corpus that has a zero-length encoding in
# it, and a real corpus eventually will.
_TEXTS = [
    "alpha beta gamma delta epsilon zeta eta theta iota",  # longer than _MAX_LEN
    "alpha",
    "beta gamma",
    "",
    "delta epsilon zeta",
    "eta",
    "theta iota alpha beta",
    "gamma",
]


@pytest.fixture(scope="module")
def toy_tokenizer() -> Tokenizer:
    """A whitespace word-level tokenizer, so the test needs no dataset mount.

    Returns:
        A tokenizer over the words used by `_TEXTS`, plus an unknown token.
    """
    words = sorted({word for text in _TEXTS for word in text.split()})
    vocab = {"[UNK]": 0, **{word: i + 1 for i, word in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))  # noqa: S106
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


def _assert_batches_match(
    tok: Tokenizer, texts: list[str], max_len: int, ranges: list[tuple[int, int]]
) -> None:
    """Assert the cache and `_tokenize` agree exactly over each `[lo, hi)` range.

    Args:
        tok: Tokenizer under test.
        texts: Corpus to tokenise.
        max_len: Truncation length.
        ranges: Half-open batch ranges to compare.
    """
    tokens = encode_ragged(tok, texts, max_len)
    device = torch.device("cpu")
    for lo, hi in ranges:
        want_ids, want_mask = _tokenize(tok, texts[lo:hi], max_len, device)
        got_ids, got_mask = tokens.batch(lo, hi, device)
        assert got_ids.dtype == want_ids.dtype
        assert got_mask.dtype == want_mask.dtype
        assert got_ids.shape == want_ids.shape, f"width differs on [{lo}, {hi})"
        assert torch.equal(got_ids, want_ids), f"ids differ on [{lo}, {hi})"
        assert torch.equal(got_mask, want_mask), f"mask differs on [{lo}, {hi})"


def test_batches_are_identical_to_tokenize(toy_tokenizer: Tokenizer) -> None:
    """Every batch the cache serves matches what the in-loop tokenizer produced."""
    _assert_batches_match(
        toy_tokenizer,
        _TEXTS,
        _MAX_LEN,
        [(0, len(_TEXTS)), (0, 2), (1, 4), (3, 6), (2, 3), (len(_TEXTS) - 2, len(_TEXTS))],
    )


@pytest.mark.skipif(not _REAL_TOKENIZER.is_file(), reason="fleet dataset mount not present")
def test_batches_match_tokenize_on_the_real_gpt2_tokenizer() -> None:
    """The equality also holds for the tokenizer the regions actually train with.

    The toy tokenizer above cannot catch a divergence between `encode` and `encode_batch`
    in a real BPE pipeline (normalizer, byte-level pre-tokenizer, merges). This can.
    """
    tok = Tokenizer.from_file(str(_REAL_TOKENIZER))
    texts = [
        "def add(a, b):\n    return a + b",
        "Return the sum of two numbers.",
        "",
        "x" * 4000,
        "Ünïcodé — em dash, curly ‘quotes’, and a tab\there.",
        "short",
        "a much longer docstring that will certainly exceed the truncation length " * 4,
        "one",
    ]
    _assert_batches_match(tok, texts, 96, [(0, len(texts)), (1, 5), (2, 4), (5, 8)])


def test_batch_width_is_the_batch_maximum_not_max_len(toy_tokenizer: Tokenizer) -> None:
    """Padding is per batch. Padding to `max_len` up front is the trap this rules out."""
    tokens = encode_ragged(toy_tokenizer, _TEXTS, _MAX_LEN)
    # `_TEXTS[1:3]` is "alpha" (1 token) and "beta gamma" (2), so the batch is 2 wide.
    ids, mask = tokens.batch(1, 3, torch.device("cpu"))
    assert ids.shape == (2, 2)
    assert mask.tolist() == [[1, 0], [1, 1]]
    # And the stored buffer is ragged, not a padded rectangle.
    assert tokens.n_tokens < len(_TEXTS) * _MAX_LEN


def test_truncation_matches_max_len(toy_tokenizer: Tokenizer) -> None:
    """A sequence longer than `max_len` is cut to it, exactly as `_tokenize` cuts it."""
    tokens = encode_ragged(toy_tokenizer, _TEXTS, 3)
    ids, mask = tokens.batch(0, 1, torch.device("cpu"))
    assert ids.shape == (1, 3)
    assert mask.tolist() == [[1, 1, 1]]


def test_all_empty_batch_matches_tokenize(toy_tokenizer: Tokenizer) -> None:
    """A range whose texts all encode to nothing still produces `_tokenize`'s [n, 1]."""
    _assert_batches_match(toy_tokenizer, ["", "", ""], _MAX_LEN, [(0, 3), (1, 3)])


def test_batch_range_is_validated(toy_tokenizer: Tokenizer) -> None:
    """An out-of-range slice raises instead of silently serving the wrong rows."""
    tokens = encode_ragged(toy_tokenizer, _TEXTS, _MAX_LEN)
    with pytest.raises(ValueError, match="not within"):
        tokens.batch(0, len(_TEXTS) + 1, torch.device("cpu"))
    with pytest.raises(ValueError, match="not within"):
        tokens.batch(2, 2, torch.device("cpu"))


def _key(texts: list[str], **overrides: object) -> str:
    """Compute a cache key for `texts` with the test defaults, overridable.

    Args:
        texts: The corpus.
        overrides: Keyword arguments to `cache_key` to replace.

    Returns:
        The hex key.
    """
    kwargs: dict = {
        "max_len": _MAX_LEN,
        "tokenizer_path": "/mnt/fleet-datasets/tritter/gpt2_tokenizer.json",
        "vocab_size": 50257,
        "key_parts": {"region": "code", "side": "anchor"},
    }
    kwargs.update(overrides)
    return cache_key(texts, **kwargs)  # type: ignore[arg-type]


def test_key_changes_with_every_input_that_changes_the_ids() -> None:
    """Corpus content, order, length, truncation and the discriminators all key in."""
    base = _key(_TEXTS)
    assert base == _key(list(_TEXTS)), "same corpus must give the same key"
    assert base != _key([*_TEXTS[:-1], "different"]), "changed text must change the key"
    assert base != _key([_TEXTS[1], _TEXTS[0], *_TEXTS[2:]]), "order must change the key"
    assert base != _key(_TEXTS[:-1]), "a shorter corpus must change the key"
    assert base != _key(_TEXTS, max_len=_MAX_LEN + 1), "max_len must change the key"
    assert base != _key(_TEXTS, vocab_size=50258), "the tokenizer must change the key"
    assert base != _key(_TEXTS, key_parts={"region": "code", "side": "positive"})


def test_key_is_not_fooled_by_concatenation() -> None:
    """['ab', 'c'] and ['a', 'bc'] are different corpora and must key differently."""
    assert _key(["ab", "c"]) != _key(["a", "bc"])


def test_cache_round_trip_hits_and_matches(toy_tokenizer: Tokenizer, tmp_path: Path) -> None:
    """A second call reads the file back and serves identical batches."""
    kwargs: dict = {
        "max_len": _MAX_LEN,
        "cache_dir": tmp_path,
        "tokenizer_path": str(_REAL_TOKENIZER),
        "key_parts": {"region": "test"},
    }
    first = corpus_token_cache(toy_tokenizer, _TEXTS, **kwargs)
    written = list(tmp_path.glob("*.pt"))
    assert len(written) == 1, "one cache file per corpus"

    second = corpus_token_cache(toy_tokenizer, _TEXTS, **kwargs)
    assert torch.equal(first.flat, second.flat)
    assert torch.equal(first.offsets, second.offsets)
    assert list(tmp_path.glob("*.pt")) == written, "a hit must not rewrite the file"


def test_changed_corpus_writes_a_new_cache_entry(toy_tokenizer: Tokenizer, tmp_path: Path) -> None:
    """A changed corpus misses, re-tokenises, and lands under a different name."""
    kwargs: dict = {
        "max_len": _MAX_LEN,
        "cache_dir": tmp_path,
        "tokenizer_path": str(_REAL_TOKENIZER),
        "key_parts": {"region": "test"},
    }
    corpus_token_cache(toy_tokenizer, _TEXTS, **kwargs)
    changed = [*_TEXTS[:-1], "gamma delta epsilon"]
    tokens = corpus_token_cache(toy_tokenizer, changed, **kwargs)
    assert len(list(tmp_path.glob("*.pt"))) == 2
    expected = encode_ragged(toy_tokenizer, changed, _MAX_LEN)
    assert torch.equal(tokens.flat, expected.flat)


def test_a_payload_under_the_wrong_name_is_refused(
    toy_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    """The key is re-checked from inside the payload, not trusted from the filename.

    This is the file-moved-or-restored case. Without the in-payload check, dropping any
    cache file into the directory under the right name would train the run on whatever
    ids that file happened to hold.
    """
    kwargs: dict = {
        "max_len": _MAX_LEN,
        "cache_dir": tmp_path,
        "tokenizer_path": str(_REAL_TOKENIZER),
        "key_parts": {"region": "test"},
    }
    real = corpus_token_cache(toy_tokenizer, _TEXTS, **kwargs)
    victim = next(iter(tmp_path.glob("*.pt")))
    imposter = encode_ragged(toy_tokenizer, ["totally", "different corpus"], _MAX_LEN)
    atomic_save(
        {
            "schema": SCHEMA,
            "key": "0" * 32,
            "max_len": _MAX_LEN,
            "n_texts": 2,
            "key_parts": {"region": "test"},
            "flat": imposter.flat,
            "offsets": imposter.offsets,
        },
        victim,
    )

    served = corpus_token_cache(toy_tokenizer, _TEXTS, **kwargs)
    assert torch.equal(served.flat, real.flat), "must re-tokenise, not serve the imposter"


def test_corrupt_cache_file_falls_back_to_tokenising(
    toy_tokenizer: Tokenizer, tmp_path: Path
) -> None:
    """An unreadable file is a cache miss, not a crashed training run."""
    kwargs: dict = {
        "max_len": _MAX_LEN,
        "cache_dir": tmp_path,
        "tokenizer_path": str(_REAL_TOKENIZER),
        "key_parts": {"region": "test"},
    }
    real = corpus_token_cache(toy_tokenizer, _TEXTS, **kwargs)
    next(iter(tmp_path.glob("*.pt"))).write_bytes(b"not a torch file")
    served = corpus_token_cache(toy_tokenizer, _TEXTS, **kwargs)
    assert torch.equal(served.flat, real.flat)


def test_no_cache_dir_means_no_disk_writes(toy_tokenizer: Tokenizer, tmp_path: Path) -> None:
    """`cache_dir=None` tokenises in memory and touches nothing."""
    tokens = corpus_token_cache(
        toy_tokenizer,
        _TEXTS,
        max_len=_MAX_LEN,
        cache_dir=None,
        tokenizer_path=str(_REAL_TOKENIZER),
        key_parts={"region": "test"},
    )
    assert len(tokens) == len(_TEXTS)
    assert list(tmp_path.iterdir()) == []


def test_to_device_is_a_copy(toy_tokenizer: Tokenizer) -> None:
    """`to` returns a new store rather than mutating the shared one."""
    tokens = encode_ragged(toy_tokenizer, _TEXTS, _MAX_LEN)
    moved = tokens.to(torch.device("cpu"))
    assert isinstance(moved, RaggedTokens)
    assert torch.equal(moved.flat, tokens.flat)
