"""Pretrain a `classify` specialist: encoder + task head, trained end to end on labels.

WHY THIS IS NOT regions/pretrain.py's BI-ENCODER
`regions/pretrain.py` trains a bi-encoder with symmetric InfoNCE over (anchor, positive)
TEXT pairs -- code's (docstring, function), compress's (premise, hypothesis), retrieve's
(query, passage). That shape needs both sides to be texts a cosine similarity can compare.

`classify`'s two corpora are (text, LABEL) pairs. A label is a class id, not a text
passage, and forcing it through InfoNCE would mean one of:

  - Encoding the label NAME as the "positive" text (banking77's classes are strings like
    `card_arrival`; go_emotions' are emotion words). This is a real technique -- it is how
    zero-shot classification via sentence embeddings works -- but it optimises the WRONG
    thing for a region that will always see the closed label set at inference: it teaches
    "this utterance's embedding is close to the STRING 'card_arrival'", not "classify this
    into one of 77 known intents". It also cannot express go_emotions at all without a
    different loss again -- go_emotions is MULTI-label (a row can carry >1 emotion, see
    `_load_go_emotions`), and symmetric InfoNCE assumes exactly one positive per anchor.
    Every OTHER item in an InfoNCE batch is treated as a negative; for a multi-label
    corpus that is false whenever two rows in the same batch share a label, and it is
    false on every row that carries 2+ gold labels itself (7,102 of go_emotions' 43,410
    rows, mean 1.18 labels/row -- see the manifest and CORPUS-CONTRACT.md 1.5).

  - Bending the loss into something InfoNCE was never built for, which is exactly the
    forcing this project already refused once: `VL_REGIONS` was kept separate from
    `REGIONS` in `scripts/csd-train-all.py` rather than making the vision region's I-JEPA
    objective and linear-probe metric pretend to be a text pair. This module is that same
    refusal applied to `classify`.

So: a small classification HEAD (`nn.Linear`) on top of the SAME `TextEncoder` used
elsewhere, trained end to end with a loss that matches each corpus's actual label shape --
`CrossEntropyLoss` for banking77 (single-label, 77 mutually exclusive intents),
`BCEWithLogitsLoss` for go_emotions (multi-label, 28 independent yes/no emotions). This is
"training a classifier head instead of a bi-encoder", one of the two non-forced options.

WHY NOT A FROZEN-FEATURES LINEAR PROBE (regions/vl_pretrain.py's pattern)
`vl_pretrain.py` trains an encoder SELF-SUPERVISED (I-JEPA, no labels) and then fits a
probe on FROZEN features to measure whether the representation is any good -- the probe
is a measuring instrument, not the deliverable. `classify` has no self-supervised pretext
task at all: the label IS the only training signal available, on a corpus sized for a
supervised classifier (10,003 / 43,410 rows), not for pretraining a general representation
first. So encoder and head are trained TOGETHER from the label loss, exactly the way
`regions/pretrain.py` trains its encoder end to end from InfoNCE -- the parallel is to
that file's TRAINING SHAPE (seeded split, untrained-baseline-first, resumable checkpoint,
receipt), not to its LOSS.

WHY TWO REGIONS, NOT ONE `classify` REGION WITH TWO SOURCES (unlike `retrieve`)
`retrieve` draws (query, passage) from fiqa, Natural Questions and GooAQ -- three
DIFFERENT DOMAINS under the IDENTICAL objective and IDENTICAL output space (one shared
embedding geometry; recall@k means the same thing regardless of source). banking77 and
go_emotions share neither: 77-way single-label softmax over banking intents is not
commensurable with 28-way independent sigmoid over emotions -- there is no single head
shape, no single loss, and no single metric that describes both, so "one classify region"
would have to be two heads bolted onto one encoder speaking two unrelated label
vocabularies. docs/design/CORPUS-CONTRACT.md 1.5 reaches the same conclusion from the
corpus-balance side (81/19 split, "a go_emotions model wearing a classify name") and
names this exact choice ("Two separate regions ... more defensible than a forced merge").
So this module trains `classify_banking77` and `classify_go_emotions` as two independent
specialists -- own encoder, own head, own receipt, own gate -- catalogued together only by
living in this one file and sharing its scaffolding.

THE GATE, AND WHY THE CHANCE FLOOR IS PART OF THE RECEIPT
`beats_untrained` compares against the SAME model architecture with ZERO optimiser steps
applied -- identical in spirit to `regions/pretrain.py`'s untrained InfoNCE baseline,
which is not a formality (a random-init encoder already scores recall@1 0.40 on
CodeSearchNet from lexical overlap). Here the untrained baseline is a randomly
initialised head on a randomly initialised encoder, evaluated with NO training at all,
which is the honest lower bound a classifier has to clear.

banking77's chance floor is genuinely low (top-1 by chance ~= 1/77 = 1.30%; the classes
are close to uniform, 5.34:1 max:min per CORPUS-CONTRACT.md 1.5 B5, so a random guesser's
expected macro-F1 is close to the same 1/77). go_emotions' is the opposite trap: the
corpus is 32.8% one label ("neutral") and 184.7:1 max:min (fails B5 on both counts,
CORPUS-CONTRACT.md 1.5), so a naive metric like per-label binary ACCURACY has a chance
floor near 99.8% for the rarest label (always predict "no grief" and be right almost every
time) -- a number that LOOKS excellent and says nothing. That is why go_emotions is
measured with per-label AVERAGE PRECISION (rank-based, threshold-free) macro-averaged
across labels rather than accuracy: a random ranking's expected AP equals the label's own
prevalence, so chance for macro-AP is the corpus's mean per-label prevalence (~4-5%, NOT
50%), and that number is computed from the data and recorded alongside every AP so nobody
downstream reads a modest macro-AP as either "near-random" or "impressive" without the
reference point that says which.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import nn

from cogsyndelta.corpus import CORPUS_FINGERPRINT_SCHEME, fingerprint_corpus
from cogsyndelta.eval.metrics import _content_fingerprint, assert_no_contamination
from cogsyndelta.regions._checkpoint import atomic_save, load_resumable, rotate_checkpoints
from cogsyndelta.regions._receipt import trainer_defaults, write_receipt
from cogsyndelta.regions._tokencache import corpus_token_cache
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig


@dataclass
class ClassifyPretrainConfig:
    """One classify-specialist run: one encoder, one head, one label space."""

    region: str
    """e.g. 'classify_banking77'. Names the receipt and the checkpoint directory."""
    shards: list[str]
    text_column: str
    label_column: str
    multi_label: bool
    """False: `label_column` is a single value per row (int id or string name), trained
    with softmax cross-entropy. True: `label_column` is a list of ids per row, trained
    with independent per-label BCE."""
    label_names_from_metadata: bool = False
    """True: resolve class names from the parquet's own HuggingFace ClassLabel schema
    metadata (go_emotions' `labels` column carries the canonical 28 names this way --
    reading them beats re-deriving an order from which ids happen to appear, since a
    label that is never positive in the loaded rows would otherwise vanish or shift
    every other index). False: `label_column` holds string names directly (banking77's
    `category`); the class list is `sorted(set(...))` over every loaded row, which is
    the only stable, data-derived order available since the column carries no ids of
    its own.
    """
    steps: int = 2000
    batch_size: int = 256
    lr: float = 3e-4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    checkpoint_every: int = 500
    max_len: int = 96
    seed: int = 0
    bf16: bool = True
    device: str = "auto"
    eval_every: int = 100
    holdout_rows: int = 512
    min_holdout_positives: int = 10
    """Multi-label only. A label with fewer than this many positives in the holdout has
    an AP too noisy to trust (a handful of examples decides it) and is excluded from the
    macro average rather than silently included -- see the module docstring's go_emotions
    'grief' example (77 positives total; a small holdout can land single digits)."""
    encoder: TextEncoderConfig = field(
        default_factory=lambda: TextEncoderConfig(dim=256, depth=4, n_heads=4)
    )
    tokenizer_path: str = "/mnt/fleet-datasets/tritter/gpt2_tokenizer.json"
    out_dir: str = "receipts"


def _resolve_device(spec: str) -> torch.device:
    """Map 'auto' to cuda when available, else honour the literal string."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _tokenize(
    tok: Tokenizer, texts: list[str], max_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode texts to padded ids plus an attention mask, batch-padded like `_tokenize`
    in `regions/pretrain.py`. Duplicated rather than imported: that function is
    module-private, and `regions/vl_pretrain.py` already sets the precedent of each
    pretrain harness keeping its own small, self-contained copy of shape utilities like
    this rather than importing another harness's underscore-prefixed internals.
    """
    encoded = [tok.encode(t).ids[:max_len] for t in texts]
    width = max(1, *(len(e) for e in encoded))
    ids = torch.zeros(len(encoded), width, dtype=torch.long)
    mask = torch.zeros(len(encoded), width, dtype=torch.long)
    for i, seq in enumerate(encoded):
        if seq:
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, : len(seq)] = 1
    return ids.to(device), mask.to(device)


def _read_hf_classlabel_names(shard: str, label_column: str) -> list[str]:
    """Read a column's canonical class names from a parquet's own HF schema metadata.

    Args:
        shard: One parquet path carrying the metadata (every shard of one dataset
            config carries an identical copy; the first is enough).
        label_column: The column whose ClassLabel (or Sequence[ClassLabel]) names are
            wanted.

    Returns:
        Class names in their original index order -- index i in the data means
        `names[i]`, unconditionally.

    Raises:
        ValueError: The file has no HuggingFace metadata, or the named column is not a
            ClassLabel / Sequence[ClassLabel] feature. Falling back silently to a
            data-derived order here would risk a DIFFERENT order than the one the
            dataset publisher assigned, which corrupts every id this function is asked
            to resolve without any visible symptom until predictions are read back.
    """
    import pyarrow.parquet as pq

    meta = pq.ParquetFile(shard).schema_arrow.metadata or {}
    raw = meta.get(b"huggingface")
    if raw is None:
        raise ValueError(f"{shard} carries no huggingface schema metadata")
    info = json.loads(raw)
    feature = info["info"]["features"].get(label_column)
    if feature is None:
        raise ValueError(f"{shard}: no {label_column!r} feature in huggingface metadata")
    # A Sequence[ClassLabel] (go_emotions' `labels`) nests the ClassLabel under "feature";
    # a bare ClassLabel would carry "names" directly. Try both rather than assume the
    # nesting -- a HF dataset config's exact feature shape isn't itself part of the
    # licence-audited manifest and this function should not silently guess.
    names = feature.get("names") or feature.get("feature", {}).get("names")
    if not names:
        raise ValueError(f"{shard}: {label_column!r} feature has no class names: {feature!r}")
    return list(names)


def load_classify_rows(
    shards: list[str],
    text_column: str,
    label_column: str,
    *,
    multi_label: bool,
    label_names_from_metadata: bool,
) -> tuple[list[str], list[list[int]], list[str]]:
    """Stream (text, label-id-list) rows from parquet shards, resolving class names.

    Every row's labels are normalised to a `list[int]` -- length 1 for a single-label
    corpus, 0+ for multi-label -- so the split/dedup/contamination logic below never
    needs to know which shape it is holding.

    Args:
        shards: Parquet paths for ONE corpus (one label space; do not mix banking77 and
            go_emotions shards in one call -- see the module docstring for why they are
            two regions rather than one multi-source region).
        text_column: The text field.
        label_column: The label field -- a string (single-label) or an int list
            (multi-label).
        multi_label: Whether `label_column` already holds an id list per row.
        label_names_from_metadata: Resolve class names from the parquet's HF ClassLabel
            metadata rather than deriving them from the loaded values (see
            :class:`ClassifyPretrainConfig`).

    Returns:
        ``(texts, label_ids, class_names)``. Rows with empty text, or (multi-label only)
        zero labels, are dropped -- an unlabelled row teaches nothing and a blank text is
        a free win for any loss.
    """
    import pyarrow.parquet as pq

    class_names = (
        _read_hf_classlabel_names(shards[0], label_column) if label_names_from_metadata else []
    )
    texts: list[str] = []
    raw_labels: list[Any] = []
    for path in shards:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=2000, columns=[text_column, label_column]):
            t_col = batch.column(text_column).to_pylist()
            l_col = batch.column(label_column).to_pylist()
            for text, label in zip(t_col, l_col, strict=True):
                if not text or not text.strip():
                    continue
                if multi_label:
                    if not label:
                        continue
                    texts.append(text)
                    raw_labels.append(sorted({int(x) for x in label}))
                else:
                    if not label:
                        continue
                    texts.append(text)
                    raw_labels.append(label)

    if not multi_label:
        if not class_names:
            # Only stable, data-derivable order for a bare string column: sorted unique
            # values. Recorded in the receipt/checkpoint so the mapping is reproducible.
            class_names = sorted({str(v) for v in raw_labels})
        name_to_id = {name: i for i, name in enumerate(class_names)}
        label_ids = [[name_to_id[str(v)]] for v in raw_labels]
    else:
        if not class_names:
            raise ValueError(
                "multi_label=True requires label_names_from_metadata=True (or a future "
                "explicit class_names field) -- there is no safe data-derived order for "
                "ids that may never appear as a positive in the loaded rows"
            )
        label_ids = raw_labels

    return texts, label_ids, class_names


def _text_key(text: str) -> str:
    """Fingerprint of normalised text, matching `regions/pretrain.py`'s `_pair_key`
    normalisation (whitespace-collapsed, lower-cased) so dedup/contamination behave
    identically across the two harnesses."""
    return hashlib.blake2b(
        " ".join(text.split()).lower().encode("utf-8", "replace"), digest_size=16
    ).hexdigest()


def _content_overlap_report(train_texts: list[str], holdout_texts: list[str]) -> dict[str, Any]:
    """Train/holdout overlap keyed on CONTENT WORDS, not exact normalised text.

    WHY THIS EXISTS ALONGSIDE `assert_no_contamination`
    `build_classify_splits` deduplicates rows by `_text_key` -- whitespace/case-normalised
    text -- before splitting, and `assert_no_contamination` below fingerprints EVAL and
    TRAIN text the same way. Dedup already guarantees every text's normalised key is
    unique across the pool a holdout is sliced from, so `contamination.overlap` is 0 BY
    CONSTRUCTION regardless of the data -- the exact flaw fixed in
    `cogsyndelta.eval.metrics.pair_contamination_report` for the (anchor, positive) pair
    harnesses (see that function's docstring: "a guard has to key on something its caller
    has NOT already eliminated, or it is only confirming its own arithmetic"). `classify`
    has no pair-level fix to reuse -- there is one text per row, not two -- so this is the
    single-text analogue: `_content_fingerprint` (imported, not reimplemented, so this
    shares the exact function-word list and tokenisation the pair guard uses rather than a
    second implementation that can drift) reduces each text to its content-word SET,
    catching a paraphrase ("how do you know if X" vs "how to know if X") that exact
    normalisation calls two different texts and dedup therefore does not remove.

    NOT gated, unlike the pair guard's `pair_exact`/`pair_content` channels: those were
    calibrated against an independent 53.7%-leakage survey of `retrieve`; no equivalent
    survey exists yet for either classify corpus, and gating on an uncalibrated threshold
    risks being either toothless or blocking a legitimate run for the wrong reason. This
    is reported so the number exists to calibrate a future gate against, not asserted to
    be safe.

    Returns:
        Counts and fraction of holdout texts whose content-word set also appears in
        training, plus up to five example holdout texts so a hit is chaseable.
    """
    train_keys = {_content_fingerprint(t) for t in train_texts}
    holdout_keyed = [(t, _content_fingerprint(t)) for t in holdout_texts]
    hits = [t for t, k in holdout_keyed if k in train_keys]
    return {
        "train_unique_content_keys": len(train_keys),
        "holdout_rows": len(holdout_texts),
        "overlap": len(hits),
        "holdout_fraction_overlapping": (len(hits) / len(holdout_texts)) if holdout_texts else 0.0,
        "examples": [h[:160] for h in hits[:5]],
    }


def build_classify_splits(
    cfg: ClassifyPretrainConfig,
) -> tuple[
    tuple[list[str], list[list[int]]],
    tuple[list[str], list[list[int]]],
    list[str],
    dict[str, Any],
]:
    """Load, seed-shuffle, dedup and split one classify corpus.

    Mirrors `regions/pretrain.py`'s `build_splits` guard-for-guard: shuffle
    UNCONDITIONALLY (so the holdout is a random slice of the corpus, not whatever order
    the parquet happened to store), dedup BEFORE splitting (by normalised TEXT -- there
    is no "anchor side" here, the text itself is the one field a duplicate could hide
    in), holdout FIRST then train, contamination checked before any training happens.

    Returns:
        ``((holdout_texts, holdout_labels), (train_texts, train_labels), class_names, meta)``.
    """
    import random as _random

    texts, label_ids, class_names = load_classify_rows(
        cfg.shards,
        cfg.text_column,
        cfg.label_column,
        multi_label=cfg.multi_label,
        label_names_from_metadata=cfg.label_names_from_metadata,
    )
    n_loaded = len(texts)

    rows = list(zip(texts, label_ids, strict=True))
    # S311: seeded shuffle of training data, not a cryptographic context -- see
    # regions/pretrain.py's identical comment on the identical line.
    _random.Random(cfg.seed).shuffle(rows)  # noqa: S311
    if len(rows) < cfg.holdout_rows * 2:
        raise ValueError(f"only {len(rows)} rows; need at least {cfg.holdout_rows * 2}")

    seen: set[str] = set()
    deduped: list[tuple[str, list[int]]] = []
    for text, labels in rows:
        key = _text_key(text)
        if key not in seen:
            seen.add(key)
            deduped.append((text, labels))
    duplicates_removed = len(rows) - len(deduped)
    rows = deduped
    if len(rows) < cfg.holdout_rows * 2:
        raise ValueError(f"only {len(rows)} unique rows after dedup; need {cfg.holdout_rows * 2}")

    holdout = rows[: cfg.holdout_rows]
    train = rows[cfg.holdout_rows :]
    holdout_texts = [t for t, _ in holdout]
    holdout_labels = [ls for _, ls in holdout]
    train_texts = [t for t, _ in train]
    train_labels = [ls for _, ls in train]

    # `contamination`'s exact-normalised key is the SAME key dedup already partitioned
    # the pool on above, so `overlap` here is 0 by construction -- kept for the shape
    # every other region's receipt carries, not as evidence. `content_overlap` (below)
    # is the channel that can actually fire; see `_content_overlap_report`'s docstring.
    contamination = assert_no_contamination(train_texts, holdout_texts)
    content_overlap = _content_overlap_report(train_texts, holdout_texts)

    # B5 (docs/design/CORPUS-CONTRACT.md Part 3): within-source label concentration,
    # measured on the TRAIN split (what the model actually sees), reported unconditionally
    # so an imbalance is visible in every receipt rather than only when someone thinks to
    # check -- see the module docstring's go_emotions paragraph on why this matters for
    # which metric is trustworthy.
    counts = [0] * len(class_names)
    n_label_instances = 0
    for ls in train_labels:
        for c in ls:
            counts[c] += 1
            n_label_instances += 1
    max_count = max(counts) if counts else 0
    min_positive = min((c for c in counts if c > 0), default=0)
    balance = {
        "n_classes": len(class_names),
        "max_share_of_label_instances": (max_count / n_label_instances)
        if n_label_instances
        else 0.0,
        "max_to_min_ratio": (max_count / min_positive) if min_positive else float("inf"),
        "zero_support_classes": sum(1 for c in counts if c == 0),
        "per_class_train_count": dict(zip(class_names, counts, strict=True)),
    }

    meta = {
        "n_loaded": n_loaded,
        "duplicates_removed": duplicates_removed,
        "contamination": dict(contamination),
        "content_overlap": content_overlap,
        "train_label_balance": balance,
    }
    return (holdout_texts, holdout_labels), (train_texts, train_labels), class_names, meta


def _lr_at(step: int, cfg: ClassifyPretrainConfig) -> float:
    """Linear warmup then cosine decay, matching `regions/pretrain.py`'s schedule."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _multihot(label_ids: list[list[int]], n_classes: int, device: torch.device) -> torch.Tensor:
    """`[N]` label-id lists -> `[N, n_classes]` float multi-hot."""
    out = torch.zeros(len(label_ids), n_classes)
    for i, ids in enumerate(label_ids):
        for c in ids:
            out[i, c] = 1.0
    return out.to(device)


def _average_precision_binary(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    """Average precision for one binary label: discrete, threshold-free, sklearn-equal.

    Sorts by score descending and averages precision-at-rank over the positive ranks --
    the standard discrete AP (no interpolation). A random ranking's EXPECTED AP equals
    the label's prevalence, which is why this is the metric go_emotions is gated on
    rather than accuracy (see the module docstring).

    Returns:
        AP in [0, 1], or None if `labels` has zero positives -- AP is undefined there,
        and returning 0.0 would silently misreport "learned nothing" as "failed".
    """
    n_pos = int(labels.sum().item())
    if n_pos == 0:
        return None
    order = scores.argsort(descending=True)
    sorted_labels = labels[order].float()
    cum_tp = torch.cumsum(sorted_labels, dim=0)
    ranks = torch.arange(1, scores.numel() + 1, dtype=torch.float32, device=scores.device)
    precision_at_rank = cum_tp / ranks
    return float((precision_at_rank * sorted_labels).sum().item() / n_pos)


@torch.no_grad()
def evaluate_multi_label(
    model: TextEncoder,
    head: nn.Linear,
    tok: Tokenizer,
    texts: list[str],
    label_ids: list[list[int]],
    class_names: list[str],
    max_len: int,
    device: torch.device,
    min_holdout_positives: int,
    batch: int = 128,
) -> dict[str, Any]:
    """Per-label average precision, macro- and micro-averaged, plus per-label detail.

    A label whose holdout carries fewer than `min_holdout_positives` positives has its AP
    excluded from `macro_ap` (too few examples to trust a ranking metric on) but is still
    reported in `per_label`, marked `reliable: false`, so the exclusion is visible rather
    than a silently shrinking denominator.
    """
    was_training = model.training
    model.eval()
    logits_chunks = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        ids, mask = _tokenize(tok, chunk, max_len, device)
        logits_chunks.append(head(model(ids, mask)))
    logits = torch.cat(logits_chunks) if logits_chunks else torch.zeros(0, len(class_names))
    multihot = _multihot(label_ids, len(class_names), device)
    if was_training:
        model.train()

    per_label: dict[str, Any] = {}
    reliable_aps: list[float] = []
    reliable_prevalence: list[float] = []
    n = multihot.size(0)
    for c, name in enumerate(class_names):
        n_pos = int(multihot[:, c].sum().item())
        ap = _average_precision_binary(logits[:, c], multihot[:, c])
        reliable = ap is not None and n_pos >= min_holdout_positives
        per_label[name] = {
            "ap": ap,
            "n_pos": n_pos,
            "prevalence": (n_pos / n) if n else 0.0,
            "reliable": reliable,
        }
        if reliable:
            assert ap is not None
            reliable_aps.append(ap)
            reliable_prevalence.append(n_pos / n)

    macro_ap = sum(reliable_aps) / len(reliable_aps) if reliable_aps else 0.0
    macro_ap_chance = (
        sum(reliable_prevalence) / len(reliable_prevalence) if reliable_prevalence else 0.0
    )
    micro_ap = _average_precision_binary(logits.reshape(-1), multihot.reshape(-1))
    return {
        "n_rows": float(n),
        "macro_ap": macro_ap,
        "macro_ap_chance": macro_ap_chance,
        "macro_ap_chance_note": (
            "Expected macro-AP of a RANDOM ranking, computed from this split's own "
            "per-label prevalence -- NOT 0.5. A macro-AP near this number means the "
            "model has not learned to rank positives above negatives for these labels."
        ),
        "micro_ap": micro_ap if micro_ap is not None else 0.0,
        "n_labels_reliable": len(reliable_aps),
        "n_labels_total": len(class_names),
        "per_label": per_label,
    }


@torch.no_grad()
def evaluate_single_label(
    model: TextEncoder,
    head: nn.Linear,
    tok: Tokenizer,
    texts: list[str],
    label_ids: list[list[int]],
    class_names: list[str],
    max_len: int,
    device: torch.device,
    batch: int = 256,
) -> dict[str, Any]:
    """Top-1/top-5 accuracy and macro-F1 over a single-label holdout.

    Chance levels are recorded alongside every number (`top1_chance` etc.) rather than
    left for a reader to compute, per the module docstring's banking77 paragraph.
    """
    was_training = model.training
    model.eval()
    logits_chunks = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        ids, mask = _tokenize(tok, chunk, max_len, device)
        logits_chunks.append(head(model(ids, mask)))
    logits = torch.cat(logits_chunks) if logits_chunks else torch.zeros(0, len(class_names))
    labels = torch.tensor([ls[0] for ls in label_ids], dtype=torch.long, device=device)
    if was_training:
        model.train()

    n_classes = len(class_names)
    preds = logits.argmax(-1)
    top1 = (preds == labels).float().mean().item() if labels.numel() else 0.0
    k = min(5, n_classes)
    topk = (
        (logits.topk(k, dim=-1).indices == labels.unsqueeze(-1)).any(-1).float().mean().item()
        if labels.numel()
        else 0.0
    )

    f1s = []
    for c in range(n_classes):
        tp = int(((preds == c) & (labels == c)).sum().item())
        fp = int(((preds == c) & (labels != c)).sum().item())
        fn = int(((preds != c) & (labels == c)).sum().item())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "n_rows": float(labels.numel()),
        "top1": top1,
        f"top{k}": topk,
        "macro_f1": macro_f1,
        "top1_chance": 1.0 / n_classes if n_classes else 0.0,
        f"top{k}_chance": k / n_classes if n_classes else 0.0,
        "macro_f1_chance_note": (
            "Not recorded as a single number: macro-F1 chance depends on the predicted "
            "class distribution, not only n_classes. For a near-uniform label set (see "
            "train_label_balance in the receipt) it is close to top1_chance."
        ),
    }


def _evaluate(
    cfg: ClassifyPretrainConfig,
    model: TextEncoder,
    head: nn.Linear,
    tok: Tokenizer,
    texts: list[str],
    label_ids: list[list[int]],
    class_names: list[str],
    device: torch.device,
) -> dict[str, Any]:
    """Dispatch to the multi- or single-label evaluator for this config.

    A thin wrapper rather than a `evaluate_multi_label if cfg.multi_label else
    evaluate_single_label` callable variable: the two functions take different keyword
    arguments (`min_holdout_positives` only makes sense for multi-label), and a variable
    holding either is not something a strict type checker can give one honest signature.
    """
    if cfg.multi_label:
        return evaluate_multi_label(
            model,
            head,
            tok,
            texts,
            label_ids,
            class_names,
            cfg.max_len,
            device,
            cfg.min_holdout_positives,
        )
    return evaluate_single_label(
        model,
        head,
        tok,
        texts,
        label_ids,
        class_names,
        cfg.max_len,
        device,
    )


_CHECKPOINT_KEEP = 3
"""Same figure and justification as regions/pretrain.py's `_CHECKPOINT_KEEP`."""


def _resume_fields(cfg: ClassifyPretrainConfig) -> dict[str, Any]:
    """Config fields that must match for a checkpoint to be a valid continuation.

    Same administrative exclusions as `regions/pretrain.py`'s `_resume_fields`
    (`out_dir`, `checkpoint_every`, `eval_every`, `device`) plus this harness's own:
    `min_holdout_positives` changes only which labels feed a REPORTED average, never
    what is trained, so it is administrative here too.
    """
    return {
        "region": cfg.region,
        "shards": sorted(cfg.shards),
        "text_column": cfg.text_column,
        "label_column": cfg.label_column,
        "multi_label": cfg.multi_label,
        "label_names_from_metadata": cfg.label_names_from_metadata,
        "steps": cfg.steps,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "warmup_steps": cfg.warmup_steps,
        "grad_clip": cfg.grad_clip,
        "max_len": cfg.max_len,
        "seed": cfg.seed,
        "bf16": cfg.bf16,
        "holdout_rows": cfg.holdout_rows,
        "encoder": asdict(cfg.encoder),
        "tokenizer_path": cfg.tokenizer_path,
    }


def _config_fingerprint(cfg: ClassifyPretrainConfig) -> str:
    """Hash the resume-relevant config fields into one comparable value."""
    payload = json.dumps(_resume_fields(cfg), sort_keys=True, default=str)
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


def _checkpoint_payload(
    *,
    step: int,
    model: TextEncoder,
    head: nn.Linear,
    opt: torch.optim.Optimizer,
    encoder_cfg: TextEncoderConfig,
    class_names: list[str],
    fingerprint: str,
    fields: dict[str, Any],
    baseline: dict[str, Any],
    history: list[dict[str, float]],
    elapsed_s: float,
) -> dict[str, Any]:
    """Everything needed to continue training identically to an uninterrupted run.

    `class_names` travels in the checkpoint (not only the receipt) because the head's
    output index i means `class_names[i]` -- a checkpoint without it is a set of weights
    whose predictions cannot be decoded.
    """
    return {
        "schema": "csd-classify-checkpoint/v1",
        "step": step,
        "model": model.state_dict(),
        "head": head.state_dict(),
        "opt": opt.state_dict(),
        "config": asdict(encoder_cfg),
        "class_names": class_names,
        "config_fingerprint": fingerprint,
        "config_fields": fields,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "untrained_baseline": baseline,
        "history": history,
        "elapsed_s": elapsed_s,
    }


def pretrain_classify_region(cfg: ClassifyPretrainConfig) -> dict[str, Any]:
    """Train one classify specialist (encoder + head) in isolation and write a receipt.

    Resumable exactly as `regions/pretrain.py.pretrain_region` is: a checkpoint trained
    under an IDENTICAL config resumes (model, head, optimizer, RNG, history, untrained
    baseline all carry forward); a checkpoint trained under a DIFFERENT config is refused.

    Returns:
        The receipt dict, also written to ``{out_dir}/{region}-{timestamp}.json``.
    """
    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    tok = Tokenizer.from_file(cfg.tokenizer_path)

    (holdout_texts, holdout_labels), (train_texts, train_labels), class_names, split_meta = (
        build_classify_splits(cfg)
    )
    n_classes = len(class_names)

    encoder_cfg = TextEncoderConfig(**{**asdict(cfg.encoder), "vocab_size": tok.get_vocab_size()})
    model = TextEncoder(encoder_cfg, name=cfg.region).to(device)
    head = nn.Linear(model.out_dim, n_classes).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=cfg.lr)
    params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())

    ckpt_dir = Path(cfg.out_dir) / f"{cfg.region}-checkpoints"
    fingerprint = _config_fingerprint(cfg)
    fields = _resume_fields(cfg)
    resume = load_resumable(ckpt_dir, fingerprint, fields)

    if resume is None:
        start_step = 0
        history: list[dict[str, float]] = []
        prior_elapsed = 0.0
        # Untrained baseline: a real lower bound, not a formality -- see the module
        # docstring. Measured BEFORE any optimiser step, and never re-measured on a
        # resumed run (the model is no longer untrained by then).
        baseline = _evaluate(
            cfg, model, head, tok, holdout_texts, holdout_labels, class_names, device
        )
        print(f"    {cfg.region}: no valid checkpoint in {ckpt_dir} -- starting fresh", flush=True)
    else:
        if resume.get("class_names") != class_names:
            raise ValueError(
                f"{cfg.region}: checkpoint's class_names disagree with the current "
                f"split's -- resuming would decode the head's output through the wrong "
                f"label order. Move or delete {ckpt_dir}."
            )
        model.load_state_dict(resume["model"])
        head.load_state_dict(resume["head"])
        opt.load_state_dict(resume["opt"])
        torch.set_rng_state(resume["rng_state"])
        if resume.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
        start_step = resume["step"]
        history = resume.get("history", [])
        prior_elapsed = resume.get("elapsed_s", 0.0)
        baseline = resume["untrained_baseline"]
        print(
            f"    {cfg.region}: RESUMING from {resume['_path']} at step "
            f"{start_step}/{cfg.steps} (prior elapsed {prior_elapsed:.0f}s)",
            flush=True,
        )

    tokenise_start = time.time()
    cache_dir = Path(cfg.out_dir) / "token-cache"
    train_tokens = corpus_token_cache(
        tok,
        train_texts,
        max_len=cfg.max_len,
        cache_dir=cache_dir,
        tokenizer_path=cfg.tokenizer_path,
        key_parts={"region": cfg.region, "side": "train"},
        label=f"{cfg.region}-train",
    )
    tokenise_s = time.time() - tokenise_start

    if cfg.multi_label:
        train_targets = _multihot(train_labels, n_classes, device)
    else:
        train_targets = torch.tensor(
            [ls[0] for ls in train_labels], dtype=torch.long, device=device
        )

    amp = cfg.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp)
    if cfg.bf16 and not amp:
        print(
            f"    {cfg.region}: bf16 requested but unsupported on {device}; using fp32", flush=True
        )

    session_start = time.time()
    model.train()
    head.train()
    for step in range(start_step, cfg.steps):
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, cfg)
        lo = (step * cfg.batch_size) % max(1, len(train_texts) - cfg.batch_size)
        hi = min(lo + cfg.batch_size, len(train_texts))
        if hi - lo < 2:
            continue
        ids, mask = train_tokens.batch(lo, hi, device)
        targets = train_targets[lo:hi]
        with autocast:
            logits = head(model(ids, mask))
            loss = (
                F.binary_cross_entropy_with_logits(logits, targets)
                if cfg.multi_label
                else F.cross_entropy(logits, targets)
            )
        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), cfg.grad_clip
            )
        opt.step()

        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            held = _evaluate(
                cfg, model, head, tok, holdout_texts, holdout_labels, class_names, device
            )
            entry: dict[str, float] = {
                "step": float(step),
                "lr": _lr_at(step, cfg),
                "loss": loss.item(),
            }
            entry.update(
                {"held_macro_ap": held["macro_ap"]}
                if cfg.multi_label
                else {"held_top1": held["top1"], "held_macro_f1": held["macro_f1"]}
            )
            history.append(entry)

        if cfg.checkpoint_every and step and step % cfg.checkpoint_every == 0:
            atomic_save(
                _checkpoint_payload(
                    step=step + 1,
                    model=model,
                    head=head,
                    opt=opt,
                    encoder_cfg=encoder_cfg,
                    class_names=class_names,
                    fingerprint=fingerprint,
                    fields=fields,
                    baseline=baseline,
                    history=history,
                    elapsed_s=prior_elapsed + (time.time() - session_start),
                ),
                ckpt_dir / f"step-{step:06d}.pt",
            )
            rotate_checkpoints(ckpt_dir, _CHECKPOINT_KEEP)

    elapsed = prior_elapsed + (time.time() - session_start)
    final = _evaluate(cfg, model, head, tok, holdout_texts, holdout_labels, class_names, device)

    final_ckpt = ckpt_dir / "final.pt"
    atomic_save(
        _checkpoint_payload(
            step=cfg.steps,
            model=model,
            head=head,
            opt=opt,
            encoder_cfg=encoder_cfg,
            class_names=class_names,
            fingerprint=fingerprint,
            fields=fields,
            baseline=baseline,
            history=history,
            elapsed_s=elapsed,
        ),
        final_ckpt,
    )
    rotate_checkpoints(ckpt_dir, _CHECKPOINT_KEEP)

    if cfg.multi_label:
        beats_untrained = {
            "macro_ap": final["macro_ap"] > baseline["macro_ap"],
            "micro_ap": final["micro_ap"] > baseline["micro_ap"],
        }
        capability_per_param = final["macro_ap"] / (params / 1e6) if params else 0.0
    else:
        beats_untrained = {
            "top1": final["top1"] > baseline["top1"],
            "macro_f1": final["macro_f1"] > baseline["macro_f1"],
        }
        capability_per_param = final["top1"] / (params / 1e6) if params else 0.0

    receipt: dict[str, Any] = {
        "schema": "csd-classify-receipt/v1",
        "region": cfg.region,
        "recorded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": (
            "encoder + independent per-label BCE head (multi-label)"
            if cfg.multi_label
            else "encoder + softmax cross-entropy head (single-label)"
        ),
        "class_names": class_names,
        "corpus": {
            "shards": [Path(s).name for s in cfg.shards],
            "fingerprint": fingerprint_corpus(
                cfg.shards, columns=(cfg.text_column, cfg.label_column)
            ),
            "fingerprint_scheme": CORPUS_FINGERPRINT_SCHEME,
            "text_column": cfg.text_column,
            "label_column": cfg.label_column,
            "multi_label": cfg.multi_label,
            "rows_loaded": split_meta["n_loaded"],
            "train_rows": len(train_texts),
            "holdout_rows": len(holdout_texts),
            "duplicates_removed": split_meta["duplicates_removed"],
        },
        "train_label_balance": split_meta["train_label_balance"],
        "contamination": split_meta["contamination"],
        "content_overlap": split_meta["content_overlap"],
        "eval_declaration": "in-mixture: holdout drawn from the same seeded-shuffled pool "
        "as training, by the same code path (docs/design/CORPUS-CONTRACT.md B3). "
        "`contamination` is a tautological 0 (dedup and this check share a key); "
        "`content_overlap` is the channel that can actually fire -- see "
        "_content_overlap_report's docstring.",
        "config": {
            **{k: v for k, v in asdict(cfg).items() if k not in ("shards", "encoder")},
            "encoder": asdict(encoder_cfg),
            "trainer_defaults": trainer_defaults(cfg),
        },
        "parameters": params,
        "checkpoint": str(final_ckpt),
        "device": str(device),
        "elapsed_s": round(elapsed, 1),
        "precision": {
            "autocast": "bf16" if amp else "fp32",
            "requested_bf16": cfg.bf16,
            "master_weights": "fp32",
            "eval": "fp32",
        },
        "tokenisation": {
            "mode": "pre-tokenised ragged corpus cache",
            "prepare_s": round(tokenise_s, 2),
            "cache_dir": str(cache_dir),
            "train_tokens": train_tokens.n_tokens,
        },
        "resumed": resume is not None,
        "resumed_from_step": (resume["step"] if resume is not None else 0),
        "history": history,
        "untrained_baseline": baseline,
        "held_out": final,
        "beats_untrained": beats_untrained,
        "capability_per_param": capability_per_param,
    }

    write_receipt(
        receipt,
        Path(cfg.out_dir),
        f"{cfg.region}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json",
    )
    return receipt
