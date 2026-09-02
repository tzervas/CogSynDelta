"""Real ``[B, D]`` streams: corpus text, through the text encoder, onto the shared stream.

WHY THIS EXISTS
The PoC train, route and compression benches all drew their ``[B, D]`` stream from
``torch.rand`` / ``torch.randn``. Every number they produced was therefore a property of a
random number generator rather than of this system: a VAE fitted to uniform noise learns
the mean, a gate balanced on noise learns nothing about which region should own which
input, and a per-dimension min/max quantiser measured on isotropic Gaussians is being
measured close to its best case. This module supplies the same ``[B, D]`` surface from
real text so those numbers describe the substrate instead.

WHAT COUNTS AS "REAL" HERE, STATED PLAINLY
Corpus text -> BPE tokens -> :class:`TextEncoder` -> one mean-pooled vector per document.
That is exactly the surface the regions consume, so a bench run on it measures the
distribution the mind will actually see.

The encoder is randomly initialised unless ``encoder_checkpoint`` names trained weights,
and that has to be said rather than glossed: these embeddings are the corpus's lexical
statistics under a fixed projection, not a learned semantic space. They are still nothing
like noise. Measured on CodeSearchNet docstrings at width 128: mean pairwise cosine 0.89
(isotropic Gaussian: 0.000) and an entropy-effective rank of 8.7 out of 128. Anisotropy of
that order is what real embedding spaces look like, and it is precisely the property a
codec is *not* tested against when it is handed isotropic noise.

WHY THE FALLBACK IS A SEPARATE CLASS AND NEVER A DEFAULT
CI runners have no corpus -- the export is NFS from another host. :class:`SyntheticStream`
exists for them and for nothing else. It is never selected automatically: a caller that
wants noise has to name it. Every source carries ``provenance``, and every bench that
consumes one records that string, so a number measured on noise cannot later be mistaken
for a number measured on text.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

DEFAULT_DATASETS_ROOT = Path(os.environ.get("CSD_DATASETS_ROOT", "/mnt/fleet-datasets"))
"""Root of the fleet dataset export. Override with ``CSD_DATASETS_ROOT``."""

TEXT_SOURCES: dict[str, tuple[str, str]] = {
    "code": ("csd/region/code/codesearchnet-python/**/*.parquet", "docstring"),
    "stsb": ("csd/region/compress/stsb/**/*.parquet", "sentence1"),
    "tinystories": ("tritter/pretrain/tinystories/data/*train*.parquet", "text"),
}
"""Name -> (glob under the datasets root, text column)."""

DEFAULT_SOURCE = "code"
TOKENIZER_RELPATH = "tritter/gpt2_tokenizer.json"

# Encoder shape. Fixed rather than exposed: this is a data source, not a model to tune.
_ENCODER_HEADS = 4
_ENCODER_DEPTH = 2
_ENCODER_MIN_WIDTH = 64
_ENCODER_MAX_WIDTH = 256

_NOT_MOUNTED = (
    "text corpus not found under {root}. This is the real-data path and it is the "
    "default; nothing falls back on its own, because a silent substitution turns every "
    "measured number back into a property of torch.rand. Mount the export, point "
    "CSD_DATASETS_ROOT at a copy, or ask for SyntheticStream / --stream synthetic "
    "explicitly -- which is honest only for a CI smoke test."
)


def _encoder_width(dim: int) -> int:
    """Transformer width used to produce a ``dim``-wide stream.

    Clamped to ``[64, 256]`` and rounded up to a multiple of the head count. The clamp is
    a cost bound: a 784-wide encoder over a 50k vocab is 39M embedding parameters and
    minutes of CPU for a smoke test. When the clamp bites, the encoder's output
    projection maps up to ``dim``, which caps the stream's rank at 256 -- immaterial in
    practice, since the measured effective rank of these embeddings is under 10.

    Args:
        dim: Requested stream width.

    Returns:
        Encoder width, divisible by the head count.
    """
    width = max(_ENCODER_MIN_WIDTH, min(_ENCODER_MAX_WIDTH, dim))
    return width + (-width % _ENCODER_HEADS)


def datasets_root(root: str | Path | None = None) -> Path:
    """Return the dataset root, failing loudly when it is absent.

    Args:
        root: Explicit override; defaults to :data:`DEFAULT_DATASETS_ROOT`.

    Returns:
        The root directory.

    Raises:
        FileNotFoundError: When the directory does not exist. Globbing a missing path
            returns an empty list, which presents as "the corpus has no shards" and sends
            you looking at the data instead of at ``findmnt``.
    """
    path = Path(root) if root is not None else DEFAULT_DATASETS_ROOT
    if not path.is_dir():
        raise FileNotFoundError(_NOT_MOUNTED.format(root=path))
    return path


def corpus_available(source: str = DEFAULT_SOURCE, root: str | Path | None = None) -> bool:
    """Report whether ``source`` has shards on disk, without raising.

    Args:
        source: Key of :data:`TEXT_SOURCES`.
        root: Dataset root override.

    Returns:
        True when at least one shard and the tokenizer are present.
    """
    base = Path(root) if root is not None else DEFAULT_DATASETS_ROOT
    if not base.is_dir() or source not in TEXT_SOURCES:
        return False
    if not (base / TOKENIZER_RELPATH).is_file():
        return False
    pattern, _ = TEXT_SOURCES[source]
    return any(base.glob(pattern))


@runtime_checkable
class StreamSource(Protocol):
    """A source of ``[B, D]`` batches for the shared stream."""

    provenance: str
    """Human-readable origin, recorded by every bench that consumes this source."""

    dim: int

    def sample(self, batch_size: int) -> torch.Tensor:
        """Return one ``[batch_size, dim]`` batch on the source's device."""
        ...


def _read_texts(shards: list[Path], column: str, limit: int) -> list[str]:
    """Stream up to ``limit`` non-blank strings out of parquet shards.

    pyarrow is imported here, not at module scope: it lives in the ``train`` dependency
    group, and the PoC modules that import this one must stay importable on a CI runner
    that has only ``dev`` installed.

    Args:
        shards: Parquet paths, in order.
        column: Text column to read.
        limit: Stop once this many texts are collected.

    Returns:
        The collected texts.
    """
    import pyarrow.parquet as pq

    texts: list[str] = []
    for path in shards:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512, columns=[column]):
            for value in batch.column(column).to_pylist():
                if value and value.strip():
                    texts.append(value)
                    if len(texts) >= limit:
                        return texts
    return texts


class CorpusStream:
    """Real ``[B, D]`` stream: corpus text encoded once into a pool, then sampled.

    The pool is built once and held on the device. Encoding per step would dominate the
    runtime of every bench here and would measure the encoder rather than the region
    under test.
    """

    def __init__(
        self,
        dim: int,
        device: torch.device | str = "cpu",
        *,
        source: str = DEFAULT_SOURCE,
        texts: int = 256,
        max_len: int = 64,
        seed: int = 0,
        encoder_checkpoint: str | Path | None = None,
        root: str | Path | None = None,
    ) -> None:
        """Encode a pool of corpus documents into ``[texts, dim]`` embeddings.

        Args:
            dim: Stream width the consumer expects.
            device: Where the pool lives. GPU-first: pass the resolved context device.
            source: Key of :data:`TEXT_SOURCES`.
            texts: Pool size. Must exceed the largest batch any consumer will draw.
            max_len: Token truncation length.
            seed: Seeds both encoder init and batch sampling, so a run is reproducible
                from its config alone.
            encoder_checkpoint: Optional trained :class:`TextEncoder` ``state_dict``. When
                omitted the encoder is randomly initialised and ``provenance`` says so.
            root: Dataset root override.

        Raises:
            KeyError: Unknown ``source``.
            FileNotFoundError: Root, tokenizer or shards missing.
            ValueError: The corpus yielded fewer than two usable texts.
        """
        # Validate BEFORE importing anything heavy. `tokenizers` lives in the `train`
        # dependency group, so on a runner that installed only `dev` an unmounted corpus
        # used to surface as ModuleNotFoundError -- naming a missing package instead of
        # the missing mount, and hiding the message that tells you how to ask for the
        # synthetic fallback on purpose.
        if source not in TEXT_SOURCES:
            raise KeyError(f"unknown text source {source!r}; have {sorted(TEXT_SOURCES)}")
        base = datasets_root(root)
        tokenizer_path = base / TOKENIZER_RELPATH
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"tokenizer missing at {tokenizer_path}")

        pattern, column = TEXT_SOURCES[source]
        shards = sorted(base.glob(pattern))
        if not shards:
            raise FileNotFoundError(_NOT_MOUNTED.format(root=base / pattern))

        # Deferred for the same reason: a top-level import would make every PoC bench
        # unimportable on a runner that never touches a corpus.
        from tokenizers import Tokenizer

        from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig

        raw = _read_texts(shards, column, texts)
        if len(raw) < 2:
            raise ValueError(f"source {source!r} yielded {len(raw)} usable texts; need >= 2")

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.dim = dim
        self.device = torch.device(device)
        self.source = source
        self._generator = torch.Generator().manual_seed(seed)

        torch.manual_seed(seed)
        width = _encoder_width(dim)
        encoder = TextEncoder(
            TextEncoderConfig(
                vocab_size=tokenizer.get_vocab_size(),
                dim=width,
                depth=_ENCODER_DEPTH,
                n_heads=_ENCODER_HEADS,
                max_len=max_len,
                out_dim=dim,
            ),
            name=f"stream_encoder_{source}",
        )
        trained = encoder_checkpoint is not None
        if encoder_checkpoint is not None:
            state = torch.load(encoder_checkpoint, map_location="cpu", weights_only=True)
            encoder.load_state_dict(state.get("model", state))
        encoder = encoder.to(self.device).eval()

        ids, mask = self._tokenize(tokenizer, raw, max_len)
        with torch.no_grad():
            self.pool = encoder(ids.to(self.device), mask.to(self.device)).detach()

        self.provenance = (
            f"corpus:{source}:{column}:{len(raw)}texts:"
            f"encoder={'checkpoint' if trained else 'random-init'}:w{width}->{dim}"
        )

    @staticmethod
    def _tokenize(tokenizer, texts: list[str], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode texts to right-padded ids plus an attention mask.

        The mask is not optional. Mean-pooling over padding drags every short text toward
        the same vector, which reads as structure in the data and is an averaged-in
        constant.

        Args:
            tokenizer: A ``tokenizers.Tokenizer``.
            texts: Documents to encode.
            max_len: Truncation width.

        Returns:
            ``(ids, mask)``, both ``[N, T]`` long tensors on CPU.
        """
        encoded = [tokenizer.encode(text).ids[:max_len] for text in texts]
        width = max(1, *(len(seq) for seq in encoded))
        ids = torch.zeros(len(encoded), width, dtype=torch.long)
        mask = torch.zeros(len(encoded), width, dtype=torch.long)
        for i, seq in enumerate(encoded):
            if seq:
                ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
                mask[i, : len(seq)] = 1
        return ids, mask

    def sample(self, batch_size: int) -> torch.Tensor:
        """Draw ``[batch_size, dim]`` real embeddings, with replacement.

        Args:
            batch_size: Rows to draw.

        Returns:
            A tensor on the source's device.
        """
        idx = torch.randint(0, self.pool.size(0), (batch_size,), generator=self._generator)
        return self.pool[idx.to(self.device)]


class SyntheticStream:
    """FALLBACK ONLY. Uniform noise standing in for a corpus that is not mounted.

    This is the ``torch.rand`` the benches used to call directly, kept so CI can smoke the
    code paths on a runner with no NFS export. It is not a measurement substrate: a
    fidelity, a load split or a loss curve obtained from this describes a random number
    generator. ``provenance`` says so in every record it reaches, which is the whole point
    of it being a named class rather than a hidden default.
    """

    def __init__(self, dim: int, device: torch.device | str = "cpu", *, seed: int = 0) -> None:
        """Configure the noise generator.

        Args:
            dim: Stream width.
            device: Where sampled batches land.
            seed: Seeds the generator, so the fallback is at least reproducible.
        """
        self.dim = dim
        self.device = torch.device(device)
        self._generator = torch.Generator().manual_seed(seed)
        self.provenance = "synthetic-fallback:torch.rand:NOT-A-MEASUREMENT"

    def sample(self, batch_size: int) -> torch.Tensor:
        """Draw ``[batch_size, dim]`` uniform noise.

        Args:
            batch_size: Rows to draw.

        Returns:
            A tensor on the configured device.
        """
        return torch.rand(batch_size, self.dim, generator=self._generator).to(self.device)


def resolve_stream(
    spec: str,
    dim: int,
    device: torch.device | str = "cpu",
    *,
    seed: int = 0,
    texts: int = 256,
    max_len: int = 64,
) -> StreamSource:
    """Build the stream source named by ``spec``.

    ``spec`` is ``"corpus"``, ``"corpus:<source>"``, or ``"synthetic"``. There is no
    "auto": choosing noise is a decision a caller makes out loud, not one this function
    makes on its behalf when a mount is missing.

    Args:
        spec: Source specification.
        dim: Stream width.
        device: Target device.
        seed: Seeds encoder init and sampling.
        texts: Pool size for the corpus path.
        max_len: Token truncation length for the corpus path.

    Returns:
        A :class:`StreamSource`.

    Raises:
        ValueError: ``spec`` is not one of the accepted forms.
    """
    if spec == "synthetic":
        return SyntheticStream(dim, device, seed=seed)
    head, _, source = spec.partition(":")
    if head != "corpus":
        raise ValueError(
            f"unknown stream spec {spec!r}; expected 'corpus', 'corpus:<source>' "
            f"(one of {sorted(TEXT_SOURCES)}), or 'synthetic'"
        )
    return CorpusStream(
        dim,
        device,
        source=source or DEFAULT_SOURCE,
        texts=texts,
        max_len=max_len,
        seed=seed,
    )
