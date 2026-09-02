"""Streaming corpus access: parquet shards on disk to packed token windows.

This is the piece that was missing. The repo previously had no way to read text at all --
the "wikitext" path hashed 66 lines from a fixture, and nothing could open the 1654
parquet shards sitting on the fleet export. Everything downstream (region pretraining on
real data, perplexity, quantization measured against a real task) was blocked on it.

Design notes, all of them chosen because they are what working implementations do:

- **Streaming, never materialised.** FineWeb-Edu-10B alone is 27 GB and Wikipedia 67 GB;
  the corpus does not fit in RAM and must not be asked to. Shards are read row group by
  row group, so memory is bounded by one row group regardless of corpus size.

- **Token packing, not per-document padding.** Documents are concatenated with an EOS
  separator and cut into fixed windows. Padding every short document to the window length
  would waste most of the compute on TinyStories, where the median document is ~200
  tokens. Packing keeps every position a real training target.

- **Deterministic given (shards, seed).** Shard order and the shuffle buffer are seeded,
  so a run is reproducible from its config alone. Without this a checkpoint cannot be
  compared against another checkpoint.

- **No `datasets` dependency.** pyarrow reads these files directly. The HF `datasets`
  library would add a large dependency and a cache layer for no gain here, since the data
  is already local parquet in a stable layout.
"""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import torch

# pyarrow and tokenizers live in the `train` dependency group, which CI does not install.
# They were imported at module scope, which made `import cogsyndelta.data` -- and so
# anything importing a sibling module in this package -- fail outright on a runner that
# has only `dev`. Importing them where they are used keeps the package importable and
# still fails loudly, at the call that actually needs the corpus.
if TYPE_CHECKING:
    from tokenizers import Tokenizer

# The fleet export. Override for a different mount or a local copy.
DEFAULT_CORPUS_ROOT = Path(os.environ.get("CSD_CORPUS_ROOT", "/mnt/fleet-datasets/tritter"))

# Named corpora -> (path relative to root, text column, filenames encode a split).
#
# The third field is NOT cosmetic. An earlier version inferred split-awareness by testing
# whether any shard filename contained the split name, and fell back to ALL shards when
# none did. For every corpus without split-encoded filenames that made
# split="validation" return the entire training set -- a held-out perplexity that is
# actually a train perplexity, lower and better-looking and worthless for comparing two
# architectures. Verified at the time: the first document of wikipedia-en "validation"
# was byte-identical to the first of "train". Declaring the property removes the guess.
CORPORA: dict[str, tuple[str, str, bool]] = {
    "tinystories": ("pretrain/tinystories/data", "text", True),
    "fineweb-edu-10b": ("pretrain/fineweb-edu-10B", "text", False),
    "fineweb-edu-100b": ("pretrain/fineweb-edu-100B", "text", False),
    "wikipedia-en": ("pretrain/wikipedia/20231101.en", "text", False),
    "openwebmath": ("pretrain/openwebmath", "text", False),
    "finemath-4plus": ("pretrain/finemath-4plus", "text", False),
}

TOKENIZERS: dict[str, str] = {
    "gpt2": "gpt2_tokenizer.json",
    "pythia": "pythia_tokenizer.json",
}


def corpus_root() -> Path:
    """Return the corpus root, failing loudly if it is not mounted.

    A missing mount is by far the most likely operational failure here (the export lives
    on another host), and the default error from a glob over a non-existent path is an
    empty list -- which would look like "corpus has no shards" rather than "not mounted".
    """
    root = DEFAULT_CORPUS_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"corpus root {root} is not present. It is an NFS export from homelab; "
            f"check `findmnt {root.parent}`. Override with CSD_CORPUS_ROOT."
        )
    return root


def load_tokenizer(name: str = "gpt2") -> Tokenizer:
    """Load a tokenizer that already lives on the corpus export.

    Args:
        name: Key of ``TOKENIZERS``.

    Returns:
        A ``tokenizers.Tokenizer``.
    """
    from tokenizers import Tokenizer

    if name not in TOKENIZERS:
        raise KeyError(f"unknown tokenizer {name!r}; have {sorted(TOKENIZERS)}")
    path = corpus_root() / TOKENIZERS[name]
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer {name!r} missing at {path}")
    return Tokenizer.from_file(str(path))


def shard_paths(corpus: str) -> list[Path]:
    """Return the sorted parquet shards for a named corpus.

    Sorted so that shard order is stable across hosts and filesystems; directory
    iteration order is not.
    """
    if corpus not in CORPORA:
        raise KeyError(f"unknown corpus {corpus!r}; have {sorted(CORPORA)}")
    rel, _, _ = CORPORA[corpus]
    directory = corpus_root() / rel
    if not directory.is_dir():
        raise FileNotFoundError(f"corpus {corpus!r} not found at {directory}")
    shards = sorted(directory.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {directory}")
    return shards


def iter_documents(
    corpus: str,
    *,
    split: str = "train",
    shards: list[Path] | None = None,
    row_group_batch: int = 1000,
) -> Iterator[str]:
    """Yield raw document strings from a corpus, streaming.

    Args:
        corpus: Key of ``CORPORA``.
        split: Substring the shard filename must contain (``train``/``validation``).
            Corpora that do not encode a split in the filename yield every shard.
        shards: Explicit shard list, overriding discovery.
        row_group_batch: Rows pulled from parquet per batch.

    Yields:
        One document string at a time.
    """
    import pyarrow.parquet as pq

    _, column, has_splits = CORPORA[corpus]
    paths = shards if shards is not None else shard_paths(corpus)

    if shards is None:
        if has_splits:
            matched = [p for p in paths if split in p.name]
            if not matched:
                # Never fall back to every shard. That returns training data under the
                # name of a held-out split, and nothing downstream can detect it.
                raise FileNotFoundError(
                    f"corpus {corpus!r} declares splits but no shard name contains "
                    f"{split!r}; refusing to fall back to all shards, which would "
                    f"return training data"
                )
            paths = matched
        elif split != "train":
            raise ValueError(
                f"corpus {corpus!r} has no split-encoded shards, so {split!r} cannot be "
                f"served. Hold out shards explicitly and pass them via shards=."
            )

    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=row_group_batch, columns=[column]):
            for text in batch.column(column).to_pylist():
                if text:
                    yield text


def iter_token_windows(
    corpus: str,
    tokenizer: Tokenizer,
    *,
    seq_len: int = 512,
    split: str = "train",
    seed: int = 0,
    shuffle_buffer: int = 2048,
    eos_id: int | None = None,
    limit_windows: int | None = None,
) -> Iterator[torch.Tensor]:
    """Yield packed ``[seq_len + 1]`` token windows.

    One extra token so a caller can split into ``inputs = w[:-1]`` and
    ``targets = w[1:]`` without another read.

    Documents are concatenated with ``eos_id`` between them and cut at ``seq_len + 1``.
    A shuffle buffer decorrelates neighbouring windows -- consecutive windows otherwise
    come from the same document, which makes each gradient step see a far narrower slice
    of the distribution than the batch size suggests.

    Args:
        corpus: Key of ``CORPORA``.
        tokenizer: Loaded tokenizer.
        seq_len: Context length.
        split: Passed to :func:`iter_documents`.
        seed: Seeds the shuffle buffer.
        shuffle_buffer: Windows held for shuffling. 0 disables.
        eos_id: Separator token. Defaults to the tokenizer's own ``<|endoftext|>``.
            NOT vocab_size - 1: that is correct for GPT-2 only by coincidence. For the
            pythia tokenizer shipped alongside it, vocab_size - 1 is 50276, which decodes
            to two spaces, while the real EOS is id 0 -- so every packed window would be
            separated by whitespace and the real EOS never emitted.
        limit_windows: Stop after this many windows. Useful for smoke runs.

    Yields:
        ``torch.long`` tensors of shape ``[seq_len + 1]``.
    """
    if eos_id is None:
        eos_id = tokenizer.token_to_id("<|endoftext|>")
        if eos_id is None:
            raise ValueError(
                "tokenizer has no <|endoftext|>; pass eos_id explicitly. "
                "vocab_size - 1 is correct for GPT-2 only by coincidence."
            )

    # S311 is suppressed below: this shuffles a training buffer, not a cryptographic
    # context. A seeded, reproducible PRNG is exactly what is wanted --
    # secrets.SystemRandom would destroy the determinism the docstring promises.
    rng = random.Random(seed)  # noqa: S311
    buffer: list[torch.Tensor] = []
    carry: list[int] = []
    emitted = 0
    width = seq_len + 1

    def drain_one() -> torch.Tensor:
        """Pop a uniformly random window via swap-with-last, which is O(1)."""
        idx = rng.randrange(len(buffer))
        buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
        return buffer.pop()

    for doc in iter_documents(corpus, split=split):
        carry.extend(tokenizer.encode(doc).ids)
        carry.append(eos_id)

        while len(carry) >= width:
            window = torch.tensor(carry[:width], dtype=torch.long)
            del carry[:width]

            if shuffle_buffer <= 0:
                yield window
                emitted += 1
            else:
                buffer.append(window)
                if len(buffer) >= shuffle_buffer:
                    yield drain_one()
                    emitted += 1

            if limit_windows is not None and emitted >= limit_windows:
                return

    while buffer:
        yield drain_one()
        emitted += 1
        if limit_windows is not None and emitted >= limit_windows:
            return


def iter_batches(
    corpus: str,
    tokenizer: Tokenizer,
    *,
    batch_size: int = 8,
    seq_len: int = 512,
    device: torch.device | str = "cpu",
    **kwargs: object,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(inputs, targets)`` batches for causal LM training.

    Both are ``[batch_size, seq_len]``; ``targets`` is ``inputs`` shifted by one.

    Yields:
        Tuples of long tensors on ``device``.
    """
    batch: list[torch.Tensor] = []
    for window in iter_token_windows(corpus, tokenizer, seq_len=seq_len, **kwargs):  # type: ignore[arg-type]
        batch.append(window)
        if len(batch) == batch_size:
            stacked = torch.stack(batch).to(device)
            batch.clear()
            yield stacked[:, :-1].contiguous(), stacked[:, 1:].contiguous()
