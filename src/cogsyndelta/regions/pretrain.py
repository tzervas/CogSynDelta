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
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from cogsyndelta.eval import assert_no_contamination, mean_reciprocal_rank, recall_at_k
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig, info_nce


@dataclass
class PretrainConfig:
    """One region pretrain run."""

    region: str
    pair_columns: tuple[str, str]
    shards: list[str]
    steps: int = 500
    batch_size: int = 64
    lr: float = 3e-4
    max_len: int = 128
    seed: int = 0
    device: str = "auto"
    eval_every: int = 100
    holdout_pairs: int = 512
    encoder: TextEncoderConfig = field(default_factory=lambda: TextEncoderConfig(dim=256, depth=4))
    tokenizer_path: str = "/mnt/fleet-datasets/tritter/gpt2_tokenizer.json"
    out_dir: str = "receipts"


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

    encoder_cfg = TextEncoderConfig(**{**asdict(cfg.encoder), "vocab_size": tok.get_vocab_size()})
    model = TextEncoder(encoder_cfg, name=cfg.region).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    params = sum(p.numel() for p in model.parameters())

    history: list[dict[str, float]] = []
    started = time.time()
    model.train()
    for step in range(cfg.steps):
        lo = (step * cfg.batch_size) % max(1, len(train_pairs) - cfg.batch_size)
        chunk = train_pairs[lo : lo + cfg.batch_size]
        if len(chunk) < 2:
            continue
        a_ids, a_mask = _tokenize(tok, [a for a, _ in chunk], cfg.max_len, device)
        p_ids, p_mask = _tokenize(tok, [b for _, b in chunk], cfg.max_len, device)
        loss, stats = info_nce(model(a_ids, a_mask), model(p_ids, p_mask))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            history.append({"step": float(step), **stats})

    elapsed = time.time() - started
    final = evaluate(model, tok, holdout, cfg.max_len, device)

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
        "contamination": dict(contamination),
        "config": {
            **{k: v for k, v in asdict(cfg).items() if k not in ("shards", "encoder")},
            "encoder": asdict(encoder_cfg),
        },
        "parameters": params,
        "device": str(device),
        "elapsed_s": round(elapsed, 1),
        "history": history,
        "held_out": final,
        "capability_per_param": final["recall@1"] / (params / 1e6) if params else 0.0,
    }

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{cfg.region}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["receipt_path"] = str(path)
    return receipt
