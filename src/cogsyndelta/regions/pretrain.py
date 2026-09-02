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
    assert_no_contamination,
    contamination_report,
    mean_reciprocal_rank,
    recall_at_k,
    spearman_correlation,
)
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig, info_nce


@dataclass
class PretrainConfig:
    """One region pretrain run."""

    region: str
    pair_columns: tuple[str, str]
    shards: list[str]
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

    Args:
        left: One side of the pair.
        right: The other side.

    Returns:
        A hex digest identifying the unordered pair, whitespace- and case-normalised.
    """
    first, second = sorted((" ".join(left.split()).lower(), " ".join(right.split()).lower()))
    return hashlib.blake2b(
        f"{first}\x00{second}".encode("utf-8", "replace"), digest_size=16
    ).hexdigest()


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


def pretrain_region(cfg: PretrainConfig) -> dict[str, Any]:
    """Train one region in isolation and write a receipt.

    Returns:
        The receipt dict, also written to ``{out_dir}/{region}-{timestamp}.json``.
    """
    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    tok = Tokenizer.from_file(cfg.tokenizer_path)

    all_pairs = load_pairs(
        cfg.shards, cfg.pair_columns, limit=cfg.steps * cfg.batch_size + cfg.holdout_pairs
    )
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
    contamination = assert_no_contamination([a for a, _ in train_pairs], [a for a, _ in holdout])
    graded, graded_report = _prepare_graded(cfg, train_pairs)

    encoder_cfg = TextEncoderConfig(**{**asdict(cfg.encoder), "vocab_size": tok.get_vocab_size()})
    model = TextEncoder(encoder_cfg, name=cfg.region).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    params = sum(p.numel() for p in model.parameters())

    ckpt_dir = Path(cfg.out_dir) / f"{cfg.region}-checkpoints"

    # The untrained model is a real baseline, not a formality: lexical overlap alone
    # scores recall@1 ~0.40 here. A trained model that does not beat this has not learned,
    # it has merely rearranged. Recorded so the comparison cannot be skipped.
    baseline = evaluate(model, tok, holdout, cfg.max_len, device)
    graded_baseline = evaluate_graded(model, tok, graded, cfg.max_len, device) if graded else {}

    history: list[dict[str, float]] = []
    started = time.time()
    model.train()
    for step in range(cfg.steps):
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, cfg)
        lo = (step * cfg.batch_size) % max(1, len(train_pairs) - cfg.batch_size)
        chunk = train_pairs[lo : lo + cfg.batch_size]
        if len(chunk) < 2:
            continue
        a_ids, a_mask = _tokenize(tok, [a for a, _ in chunk], cfg.max_len, device)
        p_ids, p_mask = _tokenize(tok, [b for _, b in chunk], cfg.max_len, device)
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
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "config": asdict(encoder_cfg),
                },
                ckpt_dir / f"step-{step:06d}.pt",
            )

    elapsed = time.time() - started
    final = evaluate(model, tok, holdout, cfg.max_len, device)
    graded_final = evaluate_graded(model, tok, graded, cfg.max_len, device) if graded else {}

    # A run that reports numbers but keeps no weights cannot be re-evaluated. The periodic
    # checkpoints stop before the last step, so without this the finished model -- the only
    # one the receipt describes -- is the one artefact the run throws away. Regions with a
    # task-specific evaluation (see regions/retrieve.py, which ranks against the full
    # corpus rather than a held-out batch) need to load exactly these weights.
    final_ckpt = ckpt_dir / "final.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": cfg.steps,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "config": asdict(encoder_cfg),
        },
        final_ckpt,
    )

    receipt: dict[str, Any] = {
        "schema": "csd-pretrain-receipt/v1",
        "region": cfg.region,
        "recorded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "symmetric InfoNCE over in-batch negatives",
        "corpus": {
            "shards": [Path(s).name for s in cfg.shards],
            "fingerprint": _fingerprint_corpus(cfg.shards),
            "pair_columns": list(cfg.pair_columns),
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
