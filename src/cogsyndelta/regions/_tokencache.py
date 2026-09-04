"""Tokenise a region's corpus ONCE, ragged, and reuse it for every epoch.

WHY THIS EXISTS
`regions/pretrain.py` used to call `_tokenize` inside the training loop, which runs a
Python list comprehension over single `tok.encode()` calls -- 2 x batch_size texts per
step, on one core. Measured on akula-prime's 3090 Ti at batch 256, max_len 96:
**81.5 ms of the 131.3 ms `code` step (62%) was that tokenizer**, while a 24 GiB GPU sat
idle waiting for it. GPU utilisation sampled during a real run oscillated 38% -> 77%,
which is the signature of a starved device rather than a busy one.

The work was also almost entirely redundant. `code` runs 8,000 steps x 256 pairs over
~215k training pairs -- **~9.5 epochs**, so every pair was tokenized ~9.5 times to
produce byte-identical ids. This module does it once, up front, with
`tokenizers.Tokenizer.encode_batch` (Rust, rayon thread-pool; measured 5.6-6.9x the
single-item path across all four region corpora), and caches the result to disk.

THE TRAP THIS MODULE IS SHAPED AROUND: STORE RAGGED, PAD PER BATCH
`_tokenize` pads to the longest item IN THE BATCH, so the sequence width the GPU sees is
a property of the batch, not of `max_len`. Measured mean padded width: `code` 96.0,
`retrieve` 57-63, `compress` 32.9. Pre-padding the whole corpus to `max_len` would be the
obvious implementation and would take `compress` from 32.9 to 96 -- roughly TRIPLING its
GPU cost and turning a 2.6x win into a loss. So the ids are stored as one flat buffer
plus offsets, and :meth:`RaggedTokens.batch` pads to the batch maximum exactly the way
`_tokenize` does. `tests/test_token_cache.py` asserts the two are `torch.equal`.

THE OTHER FAILURE MODE: A STALE CACHE
Silently training on tokens from a corpus that has since changed is worse than not
caching at all, because nothing downstream would show it -- the loss curve and the
receipt would both look normal. The cache key is therefore CONTENT-ADDRESSED: a hash of
every text that went in, plus `max_len`, plus a hash of the tokenizer file itself, plus
the caller's own discriminators (region, which side of the pair). Change any of them and
the key changes and the cache misses. The key is stored inside the payload as well as in
the filename and re-checked on load, so a file that was renamed or moved into place by
hand is rejected rather than trusted.

Atomic writes come from :mod:`cogsyndelta.regions._checkpoint`; a partially written cache
file would be a truncated ragged buffer, which is exactly the silent-corruption case that
helper already exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from cogsyndelta.regions._checkpoint import atomic_save

SCHEMA = "csd-token-cache/v1"
"""Payload schema tag. A file whose schema does not match this is ignored, not adapted."""

_ENCODE_CHUNK = 20_000
"""Texts handed to `encode_batch` at a time.

`encode_batch` on 430k texts at once materialises 430k `Encoding` objects -- each with
ids, type_ids, tokens, offsets, attention_mask -- before any of them can be consumed,
which is gigabytes of Python objects for a result that is ~165 MB of int32. Chunking
keeps peak resident memory flat without measurably changing throughput: the rayon pool
still gets 20,000 texts of work per call, which is far more than enough to saturate 28
cores.
"""


@dataclass(frozen=True)
class RaggedTokens:
    """Token ids for N sequences, stored flat with offsets rather than padded.

    Attributes:
        flat: ``int32`` ids of every sequence, concatenated.
        offsets: ``int64`` of length ``N + 1``; sequence ``i`` is
            ``flat[offsets[i]:offsets[i + 1]]``.
    """

    flat: torch.Tensor
    offsets: torch.Tensor

    def __len__(self) -> int:
        """Number of sequences held."""
        return self.offsets.numel() - 1

    @property
    def n_tokens(self) -> int:
        """Total tokens stored, i.e. what padding to ``max_len`` would have inflated."""
        return int(self.flat.numel())

    def batch(self, lo: int, hi: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad sequences ``[lo, hi)`` to the widest one among them, as `_tokenize` does.

        The gather is vectorised and runs wherever ``flat`` lives (CPU by default): a
        Python loop over the rows would put a second single-threaded stall back into the
        training step, which is the whole thing this module removes.

        Args:
            lo: First sequence index, inclusive.
            hi: Last sequence index, exclusive.
            device: Where the returned tensors should land.

        Returns:
            ``(ids, mask)``, both ``[hi - lo, width]`` and ``dtype=torch.long``, with
            ``width`` the longest sequence in the range (at least 1) and ``mask`` 1 on
            real tokens and 0 on padding -- byte-identical to
            ``pretrain._tokenize(tok, texts[lo:hi], max_len, device)``.

        Raises:
            ValueError: If the range is empty or falls outside the stored sequences.
        """
        if not 0 <= lo < hi <= len(self):
            raise ValueError(f"batch range [{lo}, {hi}) is not within 0..{len(self)}")

        starts = self.offsets[lo:hi]
        lengths = self.offsets[lo + 1 : hi + 1] - starts
        width = max(1, int(lengths.max().item()))

        positions = torch.arange(width, device=self.flat.device)
        real = positions.unsqueeze(0) < lengths.unsqueeze(1)
        if self.flat.numel() == 0:
            # Every sequence in range is empty. `_tokenize` returns an all-zero [n, 1]
            # here (its `max(1, ...)`), and so must this, without indexing an empty buffer.
            zeros = torch.zeros(hi - lo, width, dtype=torch.long, device=self.flat.device)
            return zeros.to(device), zeros.clone().to(device)
        # Out-of-range positions are clamped to a valid index and then masked out, which
        # keeps the gather in bounds without a second pass over the rows.
        index = (starts.unsqueeze(1) + positions.unsqueeze(0)).clamp_(max=self.flat.numel() - 1)
        ids = self.flat[index].long().masked_fill_(~real, 0)
        return ids.to(device), real.long().to(device)

    def to(self, device: torch.device) -> RaggedTokens:
        """Return a copy with the buffers on `device`.

        Args:
            device: Target device.

        Returns:
            A new :class:`RaggedTokens`; the original is unchanged.
        """
        return RaggedTokens(flat=self.flat.to(device), offsets=self.offsets.to(device))


def encode_ragged(tok: Tokenizer, texts: Sequence[str], max_len: int) -> RaggedTokens:
    """Tokenise `texts` with `encode_batch` and store the ids ragged.

    Args:
        tok: The tokenizer.
        texts: Texts to encode, in the order the caller will index them.
        max_len: Truncation length, matching `pretrain._tokenize`.

    Returns:
        A :class:`RaggedTokens` over `texts`, in the same order.
    """
    lengths: list[int] = []
    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), _ENCODE_CHUNK):
        encodings = tok.encode_batch(list(texts[start : start + _ENCODE_CHUNK]))
        ids: list[int] = []
        for encoding in encodings:
            truncated = encoding.ids[:max_len]
            lengths.append(len(truncated))
            ids.extend(truncated)
        chunks.append(torch.tensor(ids, dtype=torch.int32))

    flat = torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.int32)
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.int64)
    if lengths:
        offsets[1:] = torch.tensor(lengths, dtype=torch.int64).cumsum(0)
    return RaggedTokens(flat=flat, offsets=offsets)


def _hash_texts(texts: Iterable[str]) -> str:
    """Content hash of a text sequence, order-sensitive and length-delimited.

    Length-delimited because concatenating raw text is ambiguous: ``["ab", "c"]`` and
    ``["a", "bc"]`` would otherwise hash identically, and those are different corpora
    that tokenise differently.

    Args:
        texts: The texts, in order.

    Returns:
        A hex digest.
    """
    h = hashlib.blake2b(digest_size=16)
    for text in texts:
        raw = text.encode("utf-8", "replace")
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()


def _hash_file(path: Path) -> str:
    """Content hash of a file, or ``"missing"`` when it is not there.

    Args:
        path: File to hash.

    Returns:
        A hex digest, or ``"missing"``.
    """
    if not path.is_file():
        return "missing"
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_key(
    texts: Sequence[str],
    *,
    max_len: int,
    tokenizer_path: str,
    vocab_size: int,
    key_parts: dict[str, Any],
) -> str:
    """Everything that changes the stored ids, hashed into one value.

    Args:
        texts: The exact texts that will be tokenised.
        max_len: Truncation length.
        tokenizer_path: Path to the tokenizer JSON; its CONTENT is hashed, not its name.
        vocab_size: The tokenizer's vocabulary size, a cheap second signal on identity.
        key_parts: Caller discriminators recorded in the key, e.g. region and pair side.

    Returns:
        A hex digest identifying this exact tokenisation.
    """
    payload = json.dumps(
        {
            "schema": SCHEMA,
            "texts": _hash_texts(texts),
            "n_texts": len(texts),
            "max_len": max_len,
            "tokenizer": _hash_file(Path(tokenizer_path)),
            "vocab_size": vocab_size,
            "parts": key_parts,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


def _load_cached(path: Path, key: str, n_texts: int) -> RaggedTokens | None:
    """Read a cache file back, or None if it is absent, unreadable or not this corpus.

    Args:
        path: Candidate cache file.
        key: The key the caller computed for the corpus it is about to train on.
        n_texts: How many sequences the caller expects.

    Returns:
        The cached tokens, or None -- in which case the caller re-tokenises.
    """
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"    token cache {path} unreadable ({exc}); re-tokenising", flush=True)
        return None
    # The key lives in the payload as well as the filename. A file that was copied,
    # renamed or restored from elsewhere therefore cannot masquerade as this corpus --
    # which is the whole point, since training on the wrong ids is silent.
    if payload.get("schema") != SCHEMA or payload.get("key") != key:
        print(f"    token cache {path} is for a different corpus/config; re-tokenising", flush=True)
        return None
    tokens = RaggedTokens(flat=payload["flat"], offsets=payload["offsets"])
    if len(tokens) != n_texts:
        print(f"    token cache {path} holds {len(tokens)} texts, expected {n_texts}", flush=True)
        return None
    return tokens


def corpus_token_cache(
    tok: Tokenizer,
    texts: Sequence[str],
    *,
    max_len: int,
    cache_dir: Path | None,
    tokenizer_path: str,
    key_parts: dict[str, Any],
    label: str = "corpus",
) -> RaggedTokens:
    """Tokenise `texts` once, reading from / writing to a content-addressed disk cache.

    Args:
        tok: The tokenizer.
        texts: Texts to encode, in the order the training loop will index them.
        max_len: Truncation length, matching `pretrain._tokenize`.
        cache_dir: Where cache files live. None tokenises in memory without touching
            disk -- what the tests and any read-only run want.
        tokenizer_path: Path to the tokenizer JSON, hashed into the key.
        key_parts: Caller discriminators for the key, e.g. ``{"region": "code"}``.
        label: Name used in the progress lines only.

    Returns:
        A :class:`RaggedTokens` over `texts`.
    """
    key = cache_key(
        texts,
        max_len=max_len,
        tokenizer_path=tokenizer_path,
        vocab_size=tok.get_vocab_size(),
        key_parts=key_parts,
    )
    path = None if cache_dir is None else cache_dir / f"{label}-{key}.pt"

    if path is not None:
        cached = _load_cached(path, key, len(texts))
        if cached is not None:
            print(
                f"    {label}: token cache HIT ({len(cached):,} texts, "
                f"{cached.n_tokens:,} tokens) {path}",
                flush=True,
            )
            return cached

    tokens = encode_ragged(tok, texts, max_len)
    padded = len(tokens) * max_len
    print(
        f"    {label}: tokenised {len(tokens):,} texts -> {tokens.n_tokens:,} tokens "
        f"({tokens.n_tokens / max(1, padded):.2f}x the ragged/padded ratio)",
        flush=True,
    )

    if path is not None:
        atomic_save(
            {
                "schema": SCHEMA,
                "key": key,
                "max_len": max_len,
                "n_texts": len(tokens),
                "key_parts": key_parts,
                "flat": tokens.flat,
                "offsets": tokens.offsets,
            },
            path,
        )
    return tokens
