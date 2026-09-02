"""Per-region pretraining — curriculum step 1, with receipts.

WHY A RECEIPT AND NOT A LOG
A training run that prints numbers and exits leaves nothing to compare against. This
writes a JSON receipt recording the corpus (with a content fingerprint), the method, the
config, the measured held-out metric, and the contamination check. That is what makes
"no quality loss" checkable later, and what lets a phase gate be evaluated by something
other than a human reading scrollback.

`actual` in the program file is filled FROM these receipts, never by hand. A hand-written
result is a claim.

WHAT IS MEASURED, AND WHY THESE
- **held-out task metric** on a split proven disjoint from training. The contamination
  guard runs before training, not after, so a dirty split stops the run rather than
  producing a number nobody can trust.
- **in-batch accuracy** alongside loss. A collapsed encoder has excellent loss and
  chance-level accuracy; loss alone cannot distinguish them.
- **embedding std**. The collapse signal, shared with JEPA.
- **parameters**. The thesis is capability per parameter, so the denominator is recorded
  with every numerator.
- **graded rank correlation**, for regions that have a human-scored eval set. Retrieval
  asks "is the right neighbour first"; a graded set asks "is the whole ordering right".
  A region whose job is stated as "neighbours stay neighbours" is gated on the second
  question, and recall@k cannot answer it -- every held-out pair is its own positive
  there, so recall@k never sees a pair humans called only half-similar.
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
from tokenizers import Tokenizer

from cogsyndelta.eval import (
    assert_no_pair_contamination,
    contamination_report,
    mean_reciprocal_rank,
    pair_fingerprint,
    recall_at_k,
    spearman_correlation,
)
from cogsyndelta.regions._checkpoint import atomic_save, load_resumable, rotate_checkpoints
from cogsyndelta.regions._tokencache import corpus_token_cache
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig, info_nce


@dataclass
class PretrainConfig:
    """One region pretrain run."""

    region: str
    pair_columns: tuple[str, str]
    shards: list[str]
    extra_sources: list[dict] = field(default_factory=list)
    """Additional corpora with DIFFERENT column names, each optionally capped.

    A region's data rarely comes from one place with one schema: `retrieve` draws
    (query, passage) from FiQA, (query, answer) from Natural Questions and
    (question, answer) from GooAQ. Each entry is
    {"shards": [...], "columns": [left, right], "limit": int}.

    `limit` exists for BALANCE, not speed. GooAQ alone is 3,012,496 pairs -- 96% of
    everything available to `retrieve` -- and training uncapped would produce a GooAQ
    model wearing a retrieval region's name.
    """
    steps: int = 2000
    batch_size: int = 256
    lr: float = 3e-4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    checkpoint_every: int = 500
    """0 disables. Long GPU runs need resumability; a 2h run lost to a transient is
    2h of GPU time that could have been the next experiment."""
    max_len: int = 128
    seed: int = 0
    bf16: bool = True
    """Run the forward and backward under `torch.autocast(dtype=torch.bfloat16)`.

    Measured 1.98x at two independent batch sizes on the 3090 Ti (sm_86), with activation
    memory at 0.573x. Autocast, never `model.to(torch.bfloat16)`: master weights, the
    optimizer, the gradient clip and the saved `state_dict` all stay fp32, so a
    checkpoint written here still loads and evaluates on the fleet's 1080 Ti (sm_61,
    which has no bf16 at all) and post-training quantization sees exactly what it saw
    before. Ignored -- silently, and correctly -- on any device without bf16 support;
    such a run simply trains in fp32 at roughly twice the step time.

    True by default because it is the right choice on every card the fleet trains on
    today, and settable because it is the one change in this file that could plausibly
    cost accuracy: `False` is the A/B arm that separates "bf16 hurt recall" from
    "something else did".
    """
    device: str = "auto"
    eval_every: int = 100
    holdout_pairs: int = 512
    encoder: TextEncoderConfig = field(
        default_factory=lambda: TextEncoderConfig(dim=256, depth=4, n_heads=4)
    )
    """n_heads is stated because it must be: dim 256 is not divisible by the
    TextEncoderConfig default of 6 heads, so leaving it implicit made PretrainConfig()
    raise at construction. 4 heads gives 64 dims per head, the usual width."""
    tokenizer_path: str = "/mnt/fleet-datasets/tritter/gpt2_tokenizer.json"
    out_dir: str = "receipts"

    graded_shards: list[str] = field(default_factory=list)
    """Optional parquet shards holding (left, right, human score) triples.

    Kept separate from `shards` on purpose. A graded set is usually a DIFFERENT corpus
    from the training one -- the compress region trains on AllNLI entailment pairs and is
    measured on STS-B -- and folding it into the training shards would make the held-out
    metric a slice of training, which is the failure this whole harness exists to prevent.
    """

    graded_columns: tuple[str, str, str] = ("sentence1", "sentence2", "score")
    graded_name: str = ""
    """Names the graded corpus in the receipt, e.g. 'stsb'. Cosmetic, but a receipt that
    says `spearman: 0.51` without saying against what is not a measurement."""


def _lr_at(step: int, cfg: PretrainConfig) -> float:
    """Linear warmup then cosine decay.

    Warmup is not decoration here. Measured on CodeSearchNet: a random-init encoder
    already scores recall@1 0.40 from lexical overlap, and stepping straight in at peak
    LR destroys that before anything replaces it -- recall fell to 0.02 and never
    recovered. Warmup lets the model leave that basin gradually.
    """
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _resolve_device(spec: str) -> torch.device:
    """Map 'auto' to cuda when available, else honour the literal string."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _tokenize(tok: Tokenizer, texts: list[str], max_len: int, device: torch.device):
    """Encode a list of texts to padded ids plus an attention mask.

    The mask matters: mean-pooling over padding drags every short text toward the same
    vector, which reads as the model learning similarity while it is averaging in a
    constant.
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


def load_pairs(
    shards: list[str], columns: tuple[str, str], limit: int | None = None
) -> list[tuple[str, str]]:
    """Stream (anchor, positive) pairs from parquet shards.

    Args:
        shards: Parquet paths.
        columns: The two text columns forming a pair.
        limit: Stop after this many pairs.

    Returns:
        Pairs with both sides non-empty. Empty sides are dropped rather than encoded as
        blanks -- a blank positive is a free win for the loss and teaches nothing.
    """
    import pyarrow.parquet as pq

    left, right = columns
    pairs: list[tuple[str, str]] = []
    for path in shards:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1000, columns=list(columns)):
            a_col = batch.column(left).to_pylist()
            b_col = batch.column(right).to_pylist()
            for a, b in zip(a_col, b_col, strict=True):
                if a and b and a.strip() and b.strip():
                    pairs.append((a, b))
                    if limit is not None and len(pairs) >= limit:
                        return pairs
    return pairs


def _pair_key(left: str, right: str) -> str:
    """Order-independent fingerprint of a sentence pair.

    Order-independent because similarity is symmetric: (a, b) in training and (b, a) in
    the eval set is the same leak, and an ordered key would miss half of them.

    Delegates to :func:`cogsyndelta.eval.metrics.pair_fingerprint` so that the graded-set
    filter here and the contamination guard's `pair_exact` channel are provably the same
    key. Two independently-written "unordered pair key" implementations that drift apart
    is exactly how a guard ends up checking something other than what it claims to.

    Args:
        left: One side of the pair.
        right: The other side.

    Returns:
        A hex digest identifying the unordered pair, whitespace- and case-normalised.
    """
    return pair_fingerprint(left, right)


def load_graded_pairs(
    shards: list[str], columns: tuple[str, str, str], limit: int | None = None
) -> list[tuple[str, str, float]]:
    """Stream (left, right, human score) triples from parquet shards.

    Args:
        shards: Parquet paths.
        columns: The two text columns and the score column, in that order.
        limit: Stop after this many triples.

    Returns:
        Triples with both sides non-empty and a non-null score. A null score is dropped
        rather than defaulted to zero -- a defaulted score is a human judgement the human
        never made, and it would move the correlation.
    """
    import pyarrow.parquet as pq

    left, right, score = columns
    graded: list[tuple[str, str, float]] = []
    for path in shards:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1000, columns=list(columns)):
            a_col = batch.column(left).to_pylist()
            b_col = batch.column(right).to_pylist()
            s_col = batch.column(score).to_pylist()
            for a, b, s in zip(a_col, b_col, s_col, strict=True):
                if a and b and a.strip() and b.strip() and s is not None:
                    graded.append((a, b, float(s)))
                    if limit is not None and len(graded) >= limit:
                        return graded
    return graded


def _fingerprint_corpus(shards: list[str]) -> str:
    """Hash shard names and sizes so a receipt identifies the data it used."""
    h = hashlib.blake2b(digest_size=16)
    for path in sorted(shards):
        p = Path(path)
        h.update(p.name.encode())
        h.update(str(p.stat().st_size if p.exists() else 0).encode())
    return h.hexdigest()


@torch.no_grad()
def evaluate(
    model: TextEncoder,
    tok: Tokenizer,
    pairs: list[tuple[str, str]],
    max_len: int,
    device: torch.device,
    batch: int = 64,
) -> dict[str, float]:
    """Retrieval metrics over held-out pairs.

    Scores every anchor against every positive in the eval set, so the candidate pool is
    the whole held-out set rather than one batch. In-batch accuracy during training is
    optimistic by comparison -- reporting only that would flatter the model.
    """
    was_training = model.training
    model.eval()
    anchors, positives = [], []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i : i + batch]
        a_ids, a_mask = _tokenize(tok, [a for a, _ in chunk], max_len, device)
        p_ids, p_mask = _tokenize(tok, [b for _, b in chunk], max_len, device)
        anchors.append(torch.nn.functional.normalize(model(a_ids, a_mask), dim=-1))
        positives.append(torch.nn.functional.normalize(model(p_ids, p_mask), dim=-1))
    a = torch.cat(anchors)
    p = torch.cat(positives)
    scores = a @ p.T
    relevant = torch.arange(a.size(0), device=a.device)
    if was_training:
        model.train()
    return {
        "n_pairs": float(a.size(0)),
        "recall@1": recall_at_k(scores, relevant, 1),
        "recall@10": recall_at_k(scores, relevant, 10),
        "mrr": mean_reciprocal_rank(scores, relevant),
        "emb_std": a.std(dim=0).mean().item(),
    }


@torch.no_grad()
def evaluate_graded(
    model: TextEncoder,
    tok: Tokenizer,
    graded: list[tuple[str, str, float]],
    max_len: int,
    device: torch.device,
    batch: int = 64,
) -> dict[str, float]:
    """Rank correlation between predicted cosine and the human score.

    Args:
        model: The encoder under test.
        tok: Tokenizer.
        graded: (left, right, score) triples.
        max_len: Token truncation length.
        device: Where to run.
        batch: Pairs encoded at a time.

    Returns:
        ``spearman`` plus the diagnostics needed to interpret it. ``cos_std`` is not
        decoration: an encoder can hold a healthy per-feature ``emb_std`` while mapping
        every PAIR to nearly the same cosine, and in that state the correlation is being
        decided by floating-point noise. A near-zero ``cos_std`` next to a plausible
        ``spearman`` means the number is not real.
    """
    was_training = model.training
    model.eval()
    predicted: list[float] = []
    gold: list[float] = []
    left_embeddings: list[torch.Tensor] = []
    for i in range(0, len(graded), batch):
        chunk = graded[i : i + batch]
        a_ids, a_mask = _tokenize(tok, [a for a, _, _ in chunk], max_len, device)
        b_ids, b_mask = _tokenize(tok, [b for _, b, _ in chunk], max_len, device)
        a = torch.nn.functional.normalize(model(a_ids, a_mask), dim=-1)
        b = torch.nn.functional.normalize(model(b_ids, b_mask), dim=-1)
        predicted.extend((a * b).sum(dim=-1).tolist())
        gold.extend(score for _, _, score in chunk)
        left_embeddings.append(a)
    if was_training:
        model.train()
    embeddings = torch.cat(left_embeddings)
    cosines = torch.tensor(predicted)
    return {
        "n_pairs": float(len(predicted)),
        "spearman": spearman_correlation(predicted, gold),
        "emb_std": embeddings.std(dim=0).mean().item(),
        "cos_mean": cosines.mean().item(),
        "cos_std": cosines.std().item(),
    }


def _prepare_graded(
    cfg: PretrainConfig, train_pairs: list[tuple[str, str]]
) -> tuple[list[tuple[str, str, float]], dict[str, Any]]:
    """Load the graded eval set and prove it is not a slice of training.

    Two different overlaps show up here and conflating them is how a number gets waved
    through.

    **Pair-level overlap is contamination.** If the exact pair is already a training
    positive, the model was handed the answer. Measured on STS-B validation against
    AllNLI entailment pairs: 2 of 1497 unique pairs, 0.13%. Small enough to look like
    noise, which is precisely why it is REMOVED rather than tolerated -- raising a
    tolerance makes a number look clean without making it clean.

    **Sentence-level overlap is not contamination, and it is large.** 15.1% of STS-B
    validation sentences appear somewhere in AllNLI, because STS-B was partly built from
    the same SNLI data. No graded score leaks through that: the model has read the
    sentence, never the human judgement about the pair. It is recorded in the receipt
    instead of gated on, because the alternative is refusing the exact corpus pairing the
    program prescribes and measuring nothing.

    Args:
        cfg: The run config.
        train_pairs: Pairs the model will actually train on.

    Returns:
        ``(triples, report)``. Both empty when the region declares no graded set.

    Raises:
        ValueError: If fewer than two graded pairs survive, which leaves nothing to rank.
    """
    if not cfg.graded_shards:
        return [], {}

    loaded = load_graded_pairs(cfg.graded_shards, cfg.graded_columns)
    train_keys = {_pair_key(a, b) for a, b in train_pairs}
    kept = [(a, b, s) for a, b, s in loaded if _pair_key(a, b) not in train_keys]
    if len(kept) < 2:
        raise ValueError(
            f"only {len(kept)} graded pairs survive the training-positive filter "
            f"({len(loaded)} loaded); there is nothing left to correlate"
        )

    sentences = contamination_report(
        (text for pair in train_pairs for text in pair),
        (text for a, b, _ in kept for text in (a, b)),
    )
    report: dict[str, Any] = {
        "name": cfg.graded_name or "graded",
        "shards": [Path(s).name for s in cfg.graded_shards],
        "fingerprint": _fingerprint_corpus(cfg.graded_shards),
        "columns": list(cfg.graded_columns),
        "pairs_loaded": len(loaded),
        "pairs_removed_as_train_positive": len(loaded) - len(kept),
        "pairs_evaluated": len(kept),
        "sentence_overlap_with_train": round(sentences["eval_fraction_contaminated"], 4),
        "sentence_overlap_note": (
            "Shared SENTENCES, not shared pairs. Inherent to the corpus pairing: STS-B "
            "was partly built from the same SNLI data as AllNLI. No graded score leaks "
            "through it. The pairs that did leak were removed and counted above."
        ),
    }
    return kept, report


def build_splits(
    cfg: PretrainConfig,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], dict[str, Any]]:
    """Load, interleave, deduplicate and split a region's corpus.

    This is the ONE definition of a region's held-out set. Post-training quantization has
    to score against exactly the pairs training was judged on, and a second implementation
    of "load, shuffle, dedup, take the first 512" would drift from this one the moment
    either changed -- silently, since both would still produce 512 plausible pairs and a
    plausible recall figure. The comparison would simply stop meaning anything.

    Returns:
        ``(holdout, train_pairs, meta)``.
    """
    budget = cfg.steps * cfg.batch_size + cfg.holdout_pairs
    all_pairs = load_pairs(cfg.shards, cfg.pair_columns, limit=budget)
    source_counts = {"primary": len(all_pairs)}
    for source in cfg.extra_sources:
        cap = source.get("limit") or 0
        got = load_pairs(
            source["shards"],
            tuple(source["columns"]),
            limit=cap if cap else budget,
        )
        source_counts[f"{source['columns'][0]}->{source['columns'][1]}"] = len(got)
        all_pairs.extend(got)
    # Shuffle unconditionally -- not just when there is more than one source. This used
    # to be gated on `len(cfg.extra_sources) > 1 or cfg.extra_sources`, which only ever
    # covers ACROSS-source contiguity and is false for every single-source region. `code`
    # and `compress` have exactly one source each, so that condition never fired and
    # neither region was ever shuffled.
    #
    # Two things break without this, and the single-source case breaks both:
    #   - IN-BATCH NEGATIVES get drawn from a contiguous same-domain block. Concatenating
    #     sources is the multi-source version of this (an InfoNCE batch drawn from one
    #     source has all-same-domain negatives); a single ungrouped-but-clustered source
    #     has the identical problem WITHIN itself. Measured on `code` (CodeSearchNet,
    #     one source, so the old condition never shuffled it): the first 3000 rows of
    #     shard 0 span only 5 repositories, 1626 of them from pandas-dev/pandas alone.
    #     Either way the task gets easier and the model learns less than the metric
    #     implies.
    #   - THE HOLDOUT IS UNREPRESENTATIVE, and this half was the one actually overlooked.
    #     `holdout = all_pairs[:cfg.holdout_pairs]` below takes the HEAD of this exact
    #     list, so an unshuffled corpus does not just make training easier, it makes the
    #     EVAL SET a non-random slice too. Measured on `code`: the 512-pair holdout --
    #     which is the entire eval set -- spanned only 2 repositories out of the corpus's
    #     13,581 (`ageitgey/face_recognition` and `apache/spark`). recall@1 0.9355 against
    #     that holdout was measuring "tell these two repos' docstrings apart", not
    #     "retrieve the right Python function". After this fix the same holdout spans
    #     443 repositories.
    import random as _random

    # S311: a seeded shuffle of training data, not a cryptographic context.
    # secrets.SystemRandom would destroy the reproducibility the receipt promises, and
    # two checkpoints are not comparable if their data order is not.
    _random.Random(cfg.seed).shuffle(all_pairs)  # noqa: S311
    if len(all_pairs) < cfg.holdout_pairs * 2:
        raise ValueError(f"only {len(all_pairs)} pairs; need at least {cfg.holdout_pairs * 2}")

    # Deduplicate BEFORE splitting. Real corpora repeat: CodeSearchNet carries boilerplate
    # docstrings ("Returns the value.") across many repositories, so a naive sequential
    # split put the same anchor on both sides. The contamination guard caught this on the
    # first real run -- 2 of 252 held-out pairs, 0.79% -- which is small enough to look
    # like noise and large enough to inflate recall@1 on a 256-candidate pool.
    #
    # Dedup by anchor fingerprint, keeping first occurrence. Raising the tolerance instead
    # would have been the wrong fix: it makes the number look clean without making it
    # clean.
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for anchor, positive in all_pairs:
        key = hashlib.blake2b(
            " ".join(anchor.split()).lower().encode("utf-8", "replace"), digest_size=16
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            deduped.append((anchor, positive))
    duplicates_removed = len(all_pairs) - len(deduped)
    all_pairs = deduped
    if len(all_pairs) < cfg.holdout_pairs * 2:
        raise ValueError(
            f"only {len(all_pairs)} unique pairs after dedup; need {cfg.holdout_pairs * 2}"
        )

    # Hold out FIRST, then train on the remainder. Splitting after training would let the
    # eval set have already been seen.
    holdout = all_pairs[: cfg.holdout_pairs]
    train_pairs = all_pairs[cfg.holdout_pairs :]

    # Contamination is checked BEFORE any training happens, so a dirty split stops the
    # run rather than producing a number nobody can trust.
    #
    # THIS USED TO BE STRUCTURALLY INCAPABLE OF FIRING. The call was
    # `assert_no_contamination([a for a, _ in train_pairs], [a for a, _ in holdout])`,
    # whose `_fingerprint` is `blake2b(" ".join(text.split()).lower())` -- byte for byte
    # the normalisation and hash the dedup loop above had just made unique, over the same
    # anchors. Unique keys, partitioned by the split, cannot intersect: the guard returned
    # `overlap: 0` for every region on every run BY CONSTRUCTION, and that zero went into
    # every receipt looking like a measurement. Meanwhile an independent survey measured
    # 53.7% near-duplicate holdout leakage in `retrieve`.
    #
    # The replacement keys on things the dedup above has NOT eliminated: the positive
    # side (dedup is anchor-only), the pair in either direction (symmetric InfoNCE trains
    # both), and a content-word normalisation coarser than the exact one dedup used. Only
    # the two PAIR-level channels are gated; see `_GATED_CHANNELS` in eval/metrics.py for
    # why the single-side channels are counted rather than enforced.
    contamination = assert_no_pair_contamination(train_pairs, holdout)
    return (
        holdout,
        train_pairs,
        {
            "source_counts": source_counts,
            "duplicates_removed": duplicates_removed,
            "contamination": contamination,
        },
    )


# ---------------------------------------------------------------------------------------
# Resumable checkpointing.
#
# A checkpoint that omits any of {model, optimizer, step, RNG state, the untrained
# baseline, a config fingerprint} produces a resumed run that either silently diverges
# from an uninterrupted one, or is unjudgeable because `beats_untrained` would be
# comparing a partially-trained model against itself. Everything below exists to make
# sure none of those five is ever missing.
# ---------------------------------------------------------------------------------------

_CHECKPOINT_KEEP = 3
"""Periodic checkpoints kept alongside `final.pt`, oldest deleted first.

At the runner's checkpoint_every=200 (scripts/csd-train-all.py), 3 buys ~600 steps of
rollback headroom -- if the single newest checkpoint were ever suspect (e.g. written
right as something started going wrong), the previous two are still there. At
measured checkpoint size (~184-192MB), that is under 600MB per region: immaterial
against the 417GB free on /akula-data at the time this was written, and small next to
the 8000-step runs' own multi-hundred-MB-per-region footprint either way.
"""


def _resume_fields(cfg: PretrainConfig) -> dict[str, Any]:
    """The config fields that must match for a checkpoint to be a valid continuation.

    Deliberately excludes purely administrative fields that do not change what is being
    trained or measured: `out_dir`, `checkpoint_every`, `eval_every`, `device`. Every
    field kept here -- steps, batch_size, lr, warmup, grad_clip, max_len, seed,
    holdout_pairs, the encoder shape, the pair columns, the shard list, the tokenizer,
    the graded set -- changes the run itself, so a checkpoint trained under a different
    value of any of them is not a continuation of what `cfg` describes.
    """
    return {
        "region": cfg.region,
        "pair_columns": list(cfg.pair_columns),
        "shards": sorted(cfg.shards),
        "extra_sources": cfg.extra_sources,
        "steps": cfg.steps,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "warmup_steps": cfg.warmup_steps,
        "grad_clip": cfg.grad_clip,
        "max_len": cfg.max_len,
        "seed": cfg.seed,
        # Precision is a resume-relevant field, not an administrative one: continuing an
        # fp32-trained checkpoint under bf16 (or the reverse) is a different run from
        # either, and the whole point of this dict is to refuse exactly that rather than
        # produce a model that is neither. The refusal names the field, so an operator
        # who meant it can move the checkpoint aside and start fresh.
        "bf16": cfg.bf16,
        "holdout_pairs": cfg.holdout_pairs,
        "encoder": asdict(cfg.encoder),
        "tokenizer_path": cfg.tokenizer_path,
        "graded_shards": sorted(cfg.graded_shards),
        "graded_columns": list(cfg.graded_columns),
        "graded_name": cfg.graded_name,
    }


def _config_fingerprint(cfg: PretrainConfig) -> str:
    """Hash the resume-relevant config fields into one comparable value."""
    payload = json.dumps(_resume_fields(cfg), sort_keys=True, default=str)
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


def _checkpoint_payload(
    *,
    step: int,
    model: TextEncoder,
    opt: torch.optim.Optimizer,
    encoder_cfg: TextEncoderConfig,
    fingerprint: str,
    fields: dict[str, Any],
    baseline: dict[str, float],
    graded_baseline: dict[str, float],
    history: list[dict[str, float]],
    elapsed_s: float,
) -> dict[str, Any]:
    """Everything needed to continue training identically to an uninterrupted run.

    `step` counts COMPLETED optimizer updates: resuming means training
    `range(step, cfg.steps)`. This redefines what the pre-resume checkpoint format's
    `step` key meant (periodic checkpoints stored the 0-indexed loop variable; the old
    `final.pt` stored `cfg.steps` -- the two were never on the same convention). That is
    safe to redefine because the only external readers of these files
    (regions/retrieve.py, scripts/csd-quantize.py, scripts/csd-benchmark.py) read only
    `model` and `config`, never `step`; those two keys keep their pre-existing names and
    shapes unchanged.

    `rng_state`/`cuda_rng_state` capture the torch RNG so a resumed run consumes
    randomness from exactly where an uninterrupted run would have been, rather than
    restarting the RNG stream from `cfg.seed`. The text harness's own batch order does
    not depend on it (the offset is a deterministic function of `step`, not sampled),
    but nothing here should rely on that staying true of every training loop that ever
    calls this -- so it is saved and restored unconditionally.
    """
    return {
        "schema": "csd-pretrain-checkpoint/v1",
        "step": step,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "config": asdict(encoder_cfg),
        "config_fingerprint": fingerprint,
        "config_fields": fields,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "untrained_baseline": baseline,
        "untrained_graded_baseline": graded_baseline,
        "history": history,
        "elapsed_s": elapsed_s,
    }


def pretrain_region(cfg: PretrainConfig) -> dict[str, Any]:
    """Train one region in isolation and write a receipt.

    Resumable: if ``{out_dir}/{region}-checkpoints`` holds a checkpoint trained under an
    IDENTICAL config (:func:`_config_fingerprint`), training continues from it --
    model, optimizer momentum, RNG state, accumulated history and the untrained baseline
    all carry forward rather than being re-measured or restarted. A checkpoint trained
    under a DIFFERENT config is refused outright (see
    :func:`cogsyndelta.regions._checkpoint.load_resumable`) rather than silently adopted
    or silently ignored.

    Returns:
        The receipt dict, also written to ``{out_dir}/{region}-{timestamp}.json``.
    """
    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    tok = Tokenizer.from_file(cfg.tokenizer_path)

    holdout, train_pairs, split_meta = build_splits(cfg)
    source_counts = split_meta["source_counts"]
    duplicates_removed = split_meta["duplicates_removed"]
    contamination = split_meta["contamination"]
    graded, graded_report = _prepare_graded(cfg, train_pairs)

    encoder_cfg = TextEncoderConfig(**{**asdict(cfg.encoder), "vocab_size": tok.get_vocab_size()})
    model = TextEncoder(encoder_cfg, name=cfg.region).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    params = sum(p.numel() for p in model.parameters())

    ckpt_dir = Path(cfg.out_dir) / f"{cfg.region}-checkpoints"
    fingerprint = _config_fingerprint(cfg)
    fields = _resume_fields(cfg)
    resume = load_resumable(ckpt_dir, fingerprint, fields)

    if resume is None:
        start_step = 0
        history: list[dict[str, float]] = []
        prior_elapsed = 0.0
        # The untrained model is a real baseline, not a formality: lexical overlap alone
        # scores recall@1 ~0.40 here. A trained model that does not beat this has not
        # learned, it has merely rearranged. Recorded so the comparison cannot be
        # skipped -- and, on a RESUMED run, never re-measured (see below): the model is
        # no longer untrained, so re-measuring here would compare it against itself.
        baseline = evaluate(model, tok, holdout, cfg.max_len, device)
        graded_baseline = evaluate_graded(model, tok, graded, cfg.max_len, device) if graded else {}
        print(f"    {cfg.region}: no valid checkpoint in {ckpt_dir} -- starting fresh", flush=True)
    else:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        torch.set_rng_state(resume["rng_state"])
        if resume.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
        start_step = resume["step"]
        history = resume.get("history", [])
        prior_elapsed = resume.get("elapsed_s", 0.0)
        baseline = resume["untrained_baseline"]
        graded_baseline = resume.get("untrained_graded_baseline", {})
        print(
            f"    {cfg.region}: RESUMING from {resume['_path']} at step "
            f"{start_step}/{cfg.steps} (prior elapsed {prior_elapsed:.0f}s)",
            flush=True,
        )

    # Tokenise the training corpus ONCE, before the clock starts. Measured on the 3090
    # Ti, 81.5 ms of a 131.3 ms `code` step was a single-threaded CPU tokenizer running
    # inside the loop, and an 8,000-step run is ~9.5 epochs, so every pair was being
    # tokenized ~9.5 times to produce identical ids. `elapsed_s` still measures the
    # training loop only, so it stays comparable with every receipt written before this;
    # the one-off cost is reported separately as `tokenisation.prepare_s`.
    tokenise_start = time.time()
    cache_dir = Path(cfg.out_dir) / "token-cache"
    anchor_tokens = corpus_token_cache(
        tok,
        [a for a, _ in train_pairs],
        max_len=cfg.max_len,
        cache_dir=cache_dir,
        tokenizer_path=cfg.tokenizer_path,
        key_parts={"region": cfg.region, "side": "anchor"},
        label=f"{cfg.region}-train-anchor",
    )
    positive_tokens = corpus_token_cache(
        tok,
        [b for _, b in train_pairs],
        max_len=cfg.max_len,
        cache_dir=cache_dir,
        tokenizer_path=cfg.tokenizer_path,
        key_parts={"region": cfg.region, "side": "positive"},
        label=f"{cfg.region}-train-positive",
    )
    tokenise_s = time.time() - tokenise_start

    # bf16 for the forward and backward; fp32 for everything that is kept or judged.
    # Gated on the DEVICE rather than assumed: Pascal (sm_61, the fleet's 1080 Ti) has no
    # bf16, and a run there must fall back to fp32 instead of failing or -- worse --
    # emulating it slowly. `torch.autocast` is re-entrant, so one context object is
    # constructed here and re-entered per step rather than rebuilt 8,000 times.
    amp = cfg.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp)
    if cfg.bf16 and not amp:
        print(
            f"    {cfg.region}: bf16 requested but unsupported on {device}; using fp32", flush=True
        )

    session_start = time.time()
    model.train()
    for step in range(start_step, cfg.steps):
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, cfg)
        lo = (step * cfg.batch_size) % max(1, len(train_pairs) - cfg.batch_size)
        hi = min(lo + cfg.batch_size, len(train_pairs))
        if hi - lo < 2:
            continue
        # Ids come from the pre-tokenised corpus, padded to THIS batch's longest
        # sequence -- byte-identical to what `_tokenize` produced here before, asserted
        # in tests/test_token_cache.py. The tokenizer is out of the loop entirely.
        a_ids, a_mask = anchor_tokens.batch(lo, hi, device)
        p_ids, p_mask = positive_tokens.batch(lo, hi, device)
        # `info_nce` casts back to fp32 for `normalize` and the logits matmul; autocast
        # covers the two encoder towers, which is where the FLOPs are. `backward` is
        # deliberately OUTSIDE the context -- autocast is a forward-only decision, and
        # the gradients it produces are already fp32 against fp32 master weights, so no
        # `GradScaler` is needed (that is fp16's problem; bf16 has fp32's exponent range).
        with autocast:
            loss, stats = info_nce(model(a_ids, a_mask), model(p_ids, p_mask))
        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            held = evaluate(model, tok, holdout, cfg.max_len, device)
            graded_now = evaluate_graded(model, tok, graded, cfg.max_len, device) if graded else {}
            history.append(
                {
                    "step": float(step),
                    "lr": _lr_at(step, cfg),
                    **stats,
                    "held_recall@1": held["recall@1"],
                    "held_recall@10": held["recall@10"],
                    **({"held_spearman": graded_now["spearman"]} if graded_now else {}),
                }
            )

        if cfg.checkpoint_every and step and step % cfg.checkpoint_every == 0:
            atomic_save(
                _checkpoint_payload(
                    step=step + 1,
                    model=model,
                    opt=opt,
                    encoder_cfg=encoder_cfg,
                    fingerprint=fingerprint,
                    fields=fields,
                    baseline=baseline,
                    graded_baseline=graded_baseline,
                    history=history,
                    elapsed_s=prior_elapsed + (time.time() - session_start),
                ),
                ckpt_dir / f"step-{step:06d}.pt",
            )
            rotate_checkpoints(ckpt_dir, _CHECKPOINT_KEEP)

    elapsed = prior_elapsed + (time.time() - session_start)
    final = evaluate(model, tok, holdout, cfg.max_len, device)
    graded_final = evaluate_graded(model, tok, graded, cfg.max_len, device) if graded else {}

    # A run that reports numbers but keeps no weights cannot be re-evaluated. The periodic
    # checkpoints stop before the last step, so without this the finished model -- the only
    # one the receipt describes -- is the one artefact the run throws away. Regions with a
    # task-specific evaluation (see regions/retrieve.py, which ranks against the full
    # corpus rather than a held-out batch) need to load exactly these weights.
    final_ckpt = ckpt_dir / "final.pt"
    atomic_save(
        _checkpoint_payload(
            step=cfg.steps,
            model=model,
            opt=opt,
            encoder_cfg=encoder_cfg,
            fingerprint=fingerprint,
            fields=fields,
            baseline=baseline,
            graded_baseline=graded_baseline,
            history=history,
            elapsed_s=elapsed,
        ),
        final_ckpt,
    )
    rotate_checkpoints(ckpt_dir, _CHECKPOINT_KEEP)

    receipt: dict[str, Any] = {
        "schema": "csd-pretrain-receipt/v1",
        "region": cfg.region,
        "recorded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "symmetric InfoNCE over in-batch negatives",
        "corpus": {
            "shards": [Path(s).name for s in cfg.shards],
            "fingerprint": _fingerprint_corpus(cfg.shards),
            "pair_columns": list(cfg.pair_columns),
            "sources": source_counts,
            "train_pairs": len(train_pairs),
            "holdout_pairs": len(holdout),
            "duplicates_removed": duplicates_removed,
        },
        **({"graded_corpus": graded_report} if graded_report else {}),
        "contamination": dict(contamination),
        "config": {
            **{k: v for k, v in asdict(cfg).items() if k not in ("shards", "encoder")},
            "encoder": asdict(encoder_cfg),
        },
        "parameters": params,
        "checkpoint": str(final_ckpt),
        "device": str(device),
        "elapsed_s": round(elapsed, 1),
        # Outside `elapsed_s` on purpose: `elapsed_s` has always meant "the training
        # loop", and folding a one-off corpus pass into it would make every receipt
        # written before pre-tokenisation incomparable with every one written after.
        # `ragged_ratio` is the guard on the padding trap -- it is the fraction of
        # `n_texts * max_len` actually stored, so a value at 1.0 means something padded
        # the corpus up front and the GPU is now chewing padding.
        # Recorded because "what precision was this trained in" is the first question
        # asked of any receipt whose recall moved, and reconstructing it from a config
        # flag plus the device's capabilities is exactly the kind of inference that goes
        # wrong a month later.
        "precision": {
            "autocast": "bf16" if amp else "fp32",
            "requested_bf16": cfg.bf16,
            "master_weights": "fp32",
            "logits": "fp32 (forced in info_nce)",
            "eval": "fp32",
        },
        "tokenisation": {
            "mode": "pre-tokenised ragged corpus cache",
            "prepare_s": round(tokenise_s, 2),
            "cache_dir": str(cache_dir),
            "train_tokens": anchor_tokens.n_tokens + positive_tokens.n_tokens,
            "ragged_ratio": round(
                (anchor_tokens.n_tokens + positive_tokens.n_tokens)
                / max(1, 2 * len(train_pairs) * cfg.max_len),
                4,
            ),
        },
        # Lets an operator reading only the receipt tell a resumed run from a fresh one,
        # and from which step -- without this, a receipt with a suspiciously short
        # elapsed_s for its step count looks like a measurement error rather than what
        # it is.
        "resumed": resume is not None,
        "resumed_from_step": (resume["step"] if resume is not None else 0),
        "history": history,
        "untrained_baseline": baseline,
        "held_out": final,
        **(
            {
                "untrained_graded_baseline": graded_baseline,
                "graded_held_out": graded_final,
            }
            if graded
            else {}
        ),
        "beats_untrained": {
            "recall@1": final["recall@1"] > baseline["recall@1"],
            "recall@10": final["recall@10"] > baseline["recall@10"],
            **(
                {"spearman": graded_final["spearman"] > graded_baseline["spearman"]}
                if graded
                else {}
            ),
        },
        "capability_per_param": final["recall@1"] / (params / 1e6) if params else 0.0,
    }

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{cfg.region}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["receipt_path"] = str(path)
    return receipt
