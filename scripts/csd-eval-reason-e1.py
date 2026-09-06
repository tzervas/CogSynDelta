#!/usr/bin/env python3
"""Score the E1 corrupted-derivation battery on existing reason checkpoints.

GPU-first: encoder scoring runs on CUDA device 0. CPU is the split loader and the
TF-IDF/BM25 oracles. Refuses if another compute app already holds the GPU.

No training. Arms are b256-s0, b512-s0, and the W2c untrained encoder.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cogsyndelta.corpus import CORPUS_FINGERPRINT_SCHEME
from cogsyndelta.eval.beir_fiqa import encode_texts
from cogsyndelta.eval.corrupted_derivation import (
    BATTERY_ID,
    CHANCE,
    CORRUPTION_SEED,
    K_CORRUPTIONS,
    PREREG_NOTES,
    BatteryItem,
    battery_fingerprint,
    build_corrupted_battery,
    control_gates,
    cosine_hit_rate,
    go_kill,
    score_items_lexical,
    score_wrong_problem_tfidf,
    w2c_region_seed,
)
from cogsyndelta.pipeline.receipt import Producer, Receipt
from cogsyndelta.regions._checkpoint import load_checkpoint, sha256_file
from cogsyndelta.regions.pretrain import PretrainConfig, _tokenize, build_splits
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "docs" / "design" / "evidence" / "g48-reason-e1-2026-09-06"
TOKENIZER = "/mnt/fleet-datasets/tritter/gpt2_tokenizer.json"
EXPECTED_GSM8K = 320
EXPECTED_ELIGIBLE = 299
EXPECTED_FP = "ca364a92d2c6c5fd259404e0ab6f52a1"
EXPECTED_TRAIN_PAIRS = 11811
EXPECTED_DUPES = 132

ARMS: dict[str, dict[str, str | None]] = {
    "b256-s0": {
        "train_receipt": (
            "/akula-data/csd/matrix/reason-b256-s0-7bc2699-20260904/receipts/"
            "reason-20260904T145943Z.json"
        ),
        "kind": "trained",
    },
    "b512-s0": {
        "train_receipt": (
            "/akula-data/csd/matrix/reason-b512-s0-7bc2699-20260904/receipts/"
            "reason-20260904T151527Z.json"
        ),
        "kind": "trained",
    },
    "baseline": {
        "train_receipt": None,
        "kind": "untrained",
    },
}


def _corpus_root() -> Path:
    for candidate in (Path("/bulk/csd-corpus"), Path("/mnt/bulk/csd-corpus")):
        if (candidate / "reason").is_dir():
            return candidate / "reason"
    raise FileNotFoundError("reason corpus not found under /bulk or /mnt/bulk")


def compute_apps() -> list[dict[str, str]]:
    """Return live compute apps from nvidia-smi (empty list = GPU free of compute)."""
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return []
    header = [h.strip() for h in lines[0].split(",")]
    rows = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        rows.append(dict(zip(header, parts, strict=False)))
    return rows


def refuse_if_gpu_busy(our_pid: int) -> None:
    """Stop when a compute app other than this process is on the card."""
    apps = compute_apps()
    others = [a for a in apps if str(a.get("pid", "")) != str(our_pid)]
    if others:
        raise RuntimeError(f"GPU 0 has another compute app; refusing to score. apps={others}")


def load_e0_holdout() -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Rebuild the reason holdout and refuse unless it matches the E0 manifest."""
    corpus = _corpus_root()
    cfg = PretrainConfig(
        region="reason",
        pair_columns=("question", "answer"),
        shards=[str(corpus / "gsm8k-main" / "train.parquet")],
        extra_sources=[
            {
                "shards": [str(corpus / "aqua_rat-raw" / "train.parquet")],
                "columns": ["question", "rationale"],
                "limit": 4982,
            }
        ],
        steps=4000,
        batch_size=256,
        max_len=256,
        holdout_pairs=512,
        seed=0,
        split_seed=0,
        order_seed=0,
        require_split_manifest=True,
        tokenizer_path=TOKENIZER,
        encoder=TextEncoderConfig(dim=256, depth=4, n_heads=4, max_len=256),
    )
    holdout, train_pairs, meta = build_splits(cfg)
    fp = str(meta.get("corpus_fingerprint", ""))
    if fp != EXPECTED_FP:
        raise RuntimeError(f"corpus fingerprint {fp} != E0 {EXPECTED_FP}")
    if len(train_pairs) != EXPECTED_TRAIN_PAIRS:
        raise RuntimeError(f"train_pairs {len(train_pairs)} != E0 {EXPECTED_TRAIN_PAIRS}")
    if int(meta.get("duplicates_removed", -1)) != EXPECTED_DUPES:
        raise RuntimeError(
            f"duplicates_removed {meta.get('duplicates_removed')} != {EXPECTED_DUPES}"
        )
    return holdout, meta


def _bind_tokenize(tok: Any, max_len: int, device: torch.device) -> Any:
    def tokenize(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        return _tokenize(tok, texts, max_len, device)

    return tokenize


def encode_battery(
    model: TextEncoder,
    tok: Any,
    items: list[BatteryItem],
    *,
    max_len: int,
    device: torch.device,
    batch: int,
) -> dict[str, float]:
    """GPU-encode questions and 1+K candidates; rank true at index 0."""
    questions = [item.question for item in items]
    candidates: list[str] = []
    for item in items:
        candidates.append(item.true_derivation)
        candidates.extend(item.corruptions)
    tokenize = _bind_tokenize(tok, max_len, device)
    q_vec = encode_texts(model, tokenize, questions, batch=batch)
    c_vec = encode_texts(model, tokenize, candidates, batch=batch)
    n_items = len(items)
    n_cand = 1 + K_CORRUPTIONS
    return cosine_hit_rate(
        q_vec.detach().cpu().numpy().astype(np.float64),
        c_vec.detach().cpu().numpy().astype(np.float64).reshape(n_items, n_cand, -1),
        k=K_CORRUPTIONS,
        tie_seed=CORRUPTION_SEED,
    )


def load_trained(receipt_path: Path, device: torch.device) -> tuple[TextEncoder, str, dict]:
    """Load a matrix-cell checkpoint through load_checkpoint."""
    receipt = json.loads(receipt_path.read_text())
    expected = receipt.get("checkpoint_sha256")
    sha_out: list[str] = []
    blob = load_checkpoint(
        receipt["checkpoint"],
        expected_sha256=str(expected) if expected else None,
        map_location=device,
        sha256_out=sha_out,
    )
    enc = TextEncoderConfig(**receipt["config"]["encoder"])
    model = TextEncoder(enc, name="reason").to(device).eval()
    model.load_state_dict(blob["model"])
    return model, sha_out[0], receipt


def load_untrained(device: torch.device) -> TextEncoder:
    """W2c region-specific-seed untrained encoder (control c / baseline arm)."""
    torch.manual_seed(w2c_region_seed("reason"))
    enc = TextEncoderConfig(dim=256, depth=4, n_heads=4, max_len=256)
    return TextEncoder(enc, name="reason-untrained").to(device).eval()


def write_eval_receipt(
    *,
    arm: str,
    metrics: dict[str, float],
    baseline: dict[str, float],
    gates: dict[str, bool],
    artifacts: dict[str, Any],
    provenance: dict[str, Any],
    detail: dict[str, Any],
    started_utc: str,
    seconds: float,
    device: str,
    out_dir: Path,
) -> Path:
    """Write a kind=eval receipt with the pre-registered notes already in detail."""
    receipt = Receipt(
        producer=Producer("cogsyndelta", "reason", "dense-transformer"),
        stage="eval",
        kind="eval",
        metrics=metrics,
        baseline=baseline,
        gates=gates,
        artifacts=artifacts,
        provenance=provenance,
        detail=detail,
        started_utc=started_utc,
        seconds=seconds,
        device=device,
    )
    return receipt.write(out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=EVIDENCE,
        help="Directory for kind=eval receipts (default: evidence dir)",
    )
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(PREREG_NOTES, flush=True)
    notes_path = out_dir / "PREREG-NOTES.txt"
    if not notes_path.is_file():
        notes_path.write_text(PREREG_NOTES + "\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"
    refuse_if_gpu_busy(os.getpid())
    if not torch.cuda.is_available():
        raise RuntimeError("E1 scoring is GPU-first; CUDA is not available after pinning 0")
    device = torch.device("cuda:0")

    t_all = time.time()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    holdout, split_meta = load_e0_holdout()
    gsm = [(a, b) for a, b in holdout if "####" in b]
    items = build_corrupted_battery(holdout, corruption_seed=CORRUPTION_SEED)
    print(
        f"holdout={len(holdout)} gsm8k={len(gsm)} eligible={len(items)} "
        f"(expected {EXPECTED_GSM8K}/{EXPECTED_ELIGIBLE})",
        flush=True,
    )
    if len(gsm) != EXPECTED_GSM8K:
        raise RuntimeError(f"gsm8k holdout {len(gsm)} != {EXPECTED_GSM8K}")
    if len(items) != EXPECTED_ELIGIBLE:
        raise RuntimeError(f"eligible items {len(items)} != {EXPECTED_ELIGIBLE}")

    tfidf = score_items_lexical(items, scorer="tfidf")
    bm25 = score_items_lexical(items, scorer="bm25")
    wrong = score_wrong_problem_tfidf(items)
    gates = control_gates(tfidf["recall@1"], bm25["recall@1"], wrong["recall@1"])
    print(
        f"oracles  tfidf={tfidf['recall@1']:.4f} bm25={bm25['recall@1']:.4f} "
        f"wrong_problem={wrong['recall@1']:.4f} gates={gates}",
        flush=True,
    )

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER)
    baseline_metrics: dict[str, float] = {
        "derive.tfidf.recall@1": tfidf["recall@1"],
        "derive.bm25.recall@1": bm25["recall@1"],
        "derive.wrong_problem.tfidf.recall@1": wrong["recall@1"],
        "derive.chance": CHANCE,
    }
    arm_scores: dict[str, float] = {}
    receipt_paths: dict[str, str] = {}
    fp = battery_fingerprint(items)

    for arm, spec in ARMS.items():
        refuse_if_gpu_busy(os.getpid())
        t0 = time.time()
        arm_started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        train_path = spec["train_receipt"]
        artifacts: dict[str, Any] = {}
        if spec["kind"] == "trained":
            assert train_path is not None
            model, ckpt_sha, train_receipt = load_trained(Path(train_path), device)
            artifacts = {
                "checkpoint": train_receipt["checkpoint"],
                "checkpoint_sha256": ckpt_sha,
                "source_training_receipt": {
                    "path": train_path,
                    "sha256": sha256_file(Path(train_path)),
                },
            }
        else:
            model = load_untrained(device)
            artifacts = {
                "checkpoint": "untrained",
                "untrained_seed": w2c_region_seed("reason"),
                "untrained_seed_rule": "sha256('csd-w2c-untrained:reason')[:8] uint32",
            }
        ranked = encode_battery(model, tok, items, max_len=256, device=device, batch=args.batch)
        del model
        torch.cuda.empty_cache()
        arm_scores[arm] = ranked["recall@1"]
        metrics = {
            "derive.recall@1": ranked["recall@1"],
            "derive.mrr": ranked["mrr"],
            "derive.n_items": ranked["n_items"],
            "derive.chance": CHANCE,
            "derive.tfidf.recall@1": tfidf["recall@1"],
            "derive.bm25.recall@1": bm25["recall@1"],
            "derive.wrong_problem.tfidf.recall@1": wrong["recall@1"],
        }
        path = write_eval_receipt(
            arm=arm,
            metrics=metrics,
            baseline=baseline_metrics,
            gates=gates,
            artifacts=artifacts,
            provenance={
                "battery_id": BATTERY_ID,
                "pooling": "matched",
                "k": 1,
                "corruption_seed": CORRUPTION_SEED,
                "eval_target": "fp32" if spec["kind"] == "trained" else "untrained",
                "arm": arm,
                "holdout_pairs": len(holdout),
                "gsm8k_holdout": len(gsm),
                "eligible_items": len(items),
                "split": split_meta.get("split"),
                "corpus_fingerprint": EXPECTED_FP,
                "fingerprint_scheme": CORPUS_FINGERPRINT_SCHEME,
                "battery_fingerprint": fp,
                "notes_written_before_scores": True,
            },
            detail={
                "notes": PREREG_NOTES,
                "arm": arm,
                "mean_annotations": float(
                    sum(it.n_annotations for it in items) / max(1, len(items))
                ),
                "family_split": {
                    "derive": ranked,
                    "tfidf": tfidf,
                    "bm25": bm25,
                    "wrong_problem": wrong,
                },
            },
            started_utc=arm_started,
            seconds=time.time() - t0,
            device=str(device),
            out_dir=out_dir / arm,
        )
        receipt_paths[arm] = str(path)
        print(
            f"arm {arm} derive.recall@1={ranked['recall@1']:.4f} "
            f"mrr={ranked['mrr']:.4f} s={time.time() - t0:.1f} -> {path}",
            flush=True,
        )

    verdict = go_kill(arm_scores)
    gpu_min = (time.time() - t_all) / 60.0
    summary = {
        "started_utc": started_utc,
        "battery_id": BATTERY_ID,
        "corruption_seed": CORRUPTION_SEED,
        "notes": PREREG_NOTES,
        "n_items": len(items),
        "gsm8k_holdout": len(gsm),
        "controls": {
            "tfidf": tfidf,
            "bm25": bm25,
            "wrong_problem": wrong,
            "gates": gates,
        },
        "arms": arm_scores,
        "go_kill": verdict,
        "gpu_minutes": gpu_min,
        "receipts": receipt_paths,
        "device": str(device),
        "split": split_meta.get("split"),
        "corpus_fingerprint": EXPECTED_FP,
        "battery_fingerprint": fp,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"go_kill": verdict, "gpu_minutes": gpu_min}, indent=2), flush=True)
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
