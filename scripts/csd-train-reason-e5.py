#!/usr/bin/env python3
"""E5 pre-registration + training: latent-step prediction on the `reason` trunk.

WHAT THIS RUNS
`cogsyndelta.regions.reason_latent_step` (that module's docstring states the loss and
stop-gradient choices; this script states everything about HOW ONE RUN is set up so it
is comparable to the reason baseline and to its own paired control arm).

Two arms, one seed per invocation:
    --arm latent-step     context = question + steps[:t]      (the E5 objective)
    --arm sequence-blind  context = question alone             (W1d's shortcut control)

Both arms load the SAME split manifest and the SAME order manifest as the reason
`b256-s0` baseline (`config/mind/splits/reason-ca364a92-split0.json` and
`...-order0-s4000-b256.json`) at the SAME default `--steps 4000 --batch-size 256`, and
both start from the SAME `torch.manual_seed(--seed)` a plain `TextEncoder` construction
under that seed would produce -- `torch.manual_seed` is the first statement this script
executes, before `build_splits` (which uses Python's own `random`, not torch's, so it
cannot perturb this), exactly mirroring `pretrain_region`'s own ordering. "Same
initialisation, same order manifest, only the objective changes" (the operator's
requirement) is therefore true by construction, not by convention: change `--steps` or
`--batch-size` away from the defaults and a DIFFERENT order-manifest file is resolved
(`cogsyndelta.splits.order_manifest_path` keys on both), so the two arms silently stop
being comparable -- this script does not special-case that; G26's existing guards do
not either, because `require_split_manifest` only refuses a MISSING/mismatched split,
not merely "a different order file than the one you meant".

WHY THIS SCRIPT NEVER APPLIES THE GO/KILL RULE ITSELF
The diagnosis's go/kill is defined ACROSS a pair of runs at the same seed (arm vs. its
sequence-blind control) and, for "go", across all three seeds at once. One invocation of
this script produces exactly one arm of one seed, with no paired receipt to read yet --
this script does not accept a peer-receipt argument. `reason_latent_step.go_kill_note`
is available and IS called here (`receipt["go_kill_reference"]["this_run"]`), but always
with `blind_recall=None`, so it always reports `"pending"` -- the honest state of a lone
run. Comparing all six receipts once phase 2 has produced them (re-reading each receipt's
`predictor_battery.recall@1` and calling `go_kill_note` with the paired arm's score) is
future work, deliberately out of scope for this pass.

PHASE 2 (not run by this task; GPU 0 must be idle first -- see AGENTS.md / this repo's
GPU policy)
    for seed in 0 1 2; do
      for arm in latent-step sequence-blind; do
        uv run --group train python scripts/csd-train-reason-e5.py --seed "$seed" --arm "$arm"
      done
    done
Six invocations, ~11 GPU-min each per the diagnosis's own estimate (~70 GPU-min total).

--dry-run
Builds the real split (G26-enforced, against the real corpus and tokenizer -- nothing
here is a synthetic fixture; see `tests/test_reason_latent_step.py` for the CPU-only
synthetic-fixture tests), writes the real pre-registration file, and runs exactly 2
optimizer steps on CPU (`CUDA_VISIBLE_DEVICES` unset/irrelevant -- device is forced to
`cpu` regardless of what a caller's environment has, so a dry run can never touch the
GPU policy's exclusive-access window). It still writes a receipt, stamped
`"dry_run": true`, so the receipt-shape tests can assert against a real run's output
rather than a hand-built fixture.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cogsyndelta.corpus import CORPUS_FINGERPRINT_SCHEME
from cogsyndelta.eval.corrupted_derivation import (
    CORRUPTION_SEED as E1_CORRUPTION_SEED,
)
from cogsyndelta.eval.corrupted_derivation import (
    build_corrupted_battery,
    control_gates,
    cosine_hit_rate,
    score_items_lexical,
    score_wrong_problem_tfidf,
)
from cogsyndelta.regions._checkpoint import atomic_save, sha256_file
from cogsyndelta.regions._receipt import write_receipt
from cogsyndelta.regions.pretrain import PretrainConfig, _lr_at, _tokenize, build_splits, evaluate
from cogsyndelta.regions.reason_latent_step import (
    MIN_HOLDOUT_STEPS,
    MIN_STEPS,
    PREREG_BATTERY,
    PREREG_CONTROLS,
    PREREG_GO,
    PREREG_HYPOTHESIS,
    PREREG_KILL,
    PREREG_SCOPE_NOTE,
    LatentStepConfig,
    LatentStepModel,
    battery_fingerprint,
    build_step_battery,
    build_step_examples,
    collapse_stats,
    ema_at,
    enumerate_step_windows,
    go_kill_note,
    score_step_battery,
)
from cogsyndelta.regions.text_encoder import TextEncoderConfig

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER = "/mnt/fleet-datasets/tritter/gpt2_tokenizer.json"

# The reason baseline's own pinned draw (docs/design/evidence/g48-reason-e1-2026-09-06/
# README.md), asserted here for the same reason csd-eval-reason-e1.py asserts them:
# "on the manifested split" must mean THIS split, not merely "a split of some size".
EXPECTED_FP = "ca364a92d2c6c5fd259404e0ab6f52a1"
EXPECTED_TRAIN_PAIRS = 11811
EXPECTED_DUPES = 132
EXPECTED_HOLDOUT = 512

ARMS = ("latent-step", "sequence-blind")
GRAD_CLIP = 1.0
# Only `final.pt` is ever written -- no periodic checkpoints, no resume path (unlike
# `pretrain_region`'s `checkpoint_every`): a toy pre-registration run at ~11 GPU-minutes
# does not need mid-run resumability, and adding it would be unused machinery.


def _corpus_root() -> Path:
    """Resolve the local (not shared-NFS) corpus root, same rule as
    `scripts/csd-eval-reason-e1.py::_corpus_root` and `csd-train-all.py`'s
    `REGION_CORPUS_ROOT["reason"]` -- duplicated rather than imported: scripts cannot
    import each other (hyphenated filenames are not valid module names), and this is a
    three-line existence check, not logic worth a shared module for one caller pair.
    """
    for candidate in (Path("/bulk/csd-corpus"), Path("/mnt/bulk/csd-corpus")):
        if (candidate / "reason").is_dir():
            return candidate / "reason"
    raise FileNotFoundError("reason corpus not found under /bulk or /mnt/bulk")


def compute_apps() -> list[dict[str, str]]:
    """Live GPU compute apps from nvidia-smi (empty = GPU free of compute)."""
    proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return []
    header = [h.strip() for h in lines[0].split(",")]
    return [dict(zip(header, [p.strip() for p in ln.split(",")], strict=False)) for ln in lines[1:]]


def refuse_if_gpu_busy(our_pid: int) -> None:
    """GPU policy: refuse to start the run phase if GPU 0 already has a compute app."""
    others = [a for a in compute_apps() if str(a.get("pid", "")) != str(our_pid)]
    if others:
        raise RuntimeError(f"GPU 0 has another compute app; refusing to start. apps={others}")


def build_prereg_config(*, steps: int, batch_size: int, seed: int) -> PretrainConfig:
    """The `PretrainConfig` this script builds splits with -- identical shape to the
    reason baseline's own (`scripts/csd-eval-reason-e1.py::load_e0_holdout`), so
    `build_splits` resolves the SAME split and order manifests.
    """
    corpus = _corpus_root()
    return PretrainConfig(
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
        steps=steps,
        batch_size=batch_size,
        max_len=256,
        holdout_pairs=EXPECTED_HOLDOUT,
        seed=seed,
        split_seed=0,
        order_seed=0,
        require_split_manifest=True,
        tokenizer_path=TOKENIZER,
        encoder=TextEncoderConfig(dim=256, depth=4, n_heads=4, max_len=256),
        lr=3e-4 * (batch_size / 256.0) ** 0.5,
        warmup_steps=max(50, steps // 15),
    )


def write_prereg(
    *,
    out_dir: Path,
    arm: str,
    seed: int,
    steps: int,
    batch_size: int,
    split_meta: dict[str, Any],
    n_train_examples: int,
    n_holdout_examples: int,
    n_battery_items: int,
    untrained_predictor_battery: dict[str, float],
) -> Path:
    """Write the E5 pre-registration file. MUST be called before the training loop.

    Args:
        out_dir: Directory the run's receipt will also land in.
        arm: `"latent-step"` or `"sequence-blind"`.
        seed: This run's training seed.
        steps, batch_size: Resolved training shape.
        split_meta: `build_splits`'s returned `meta` (carries `split`/`batch_order`/
            `corpus_fingerprint`).
        n_train_examples, n_holdout_examples, n_battery_items: Realised counts, so the
            pre-registration names what it will actually be scored against rather than
            an estimate.
        untrained_predictor_battery: The battery score BEFORE any training step runs --
            a control (diagnosis §4 E5: "an untrained predictor at the region-specific
            seed"), recorded here because it is measured before training starts, same
            as `pretrain_region`'s own `baseline`.

    Returns:
        Path written.
    """
    payload = {
        "schema": "csd-e5-prereg/v1",
        "experiment": "E5",
        "region": "reason",
        "arm": arm,
        "seed": seed,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hypothesis": PREREG_HYPOTHESIS,
        "go": PREREG_GO,
        "kill": PREREG_KILL,
        "controls": PREREG_CONTROLS,
        "battery": PREREG_BATTERY,
        "scope": PREREG_SCOPE_NOTE,
        "config": {"steps": steps, "batch_size": batch_size, "min_steps": MIN_STEPS},
        "split": split_meta.get("split"),
        "batch_order": split_meta.get("batch_order"),
        "corpus_fingerprint": split_meta.get("corpus_fingerprint"),
        "counts": {
            "train_examples": n_train_examples,
            "holdout_examples": n_holdout_examples,
            "battery_items": n_battery_items,
        },
        "untrained_predictor_battery": untrained_predictor_battery,
        "notes_written_before_training": True,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = (
        out_dir
        / f"reason-e5-prereg-{arm}-s{seed}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    )
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _score_e1_regression_guard(
    target_encoder: Any,
    tok: Any,
    holdout: list[tuple[str, str]],
    *,
    device: torch.device,
    eval_batch: int,
) -> dict[str, Any]:
    """E1's corrupted-derivation battery, scored on `target_encoder` (a `TextEncoder`
    -- the EMA target is architecture-identical to the bi-encoder E1 already scores).
    Reuses `cogsyndelta.eval.corrupted_derivation`'s own scorer verbatim, per the
    diagnosis's "via the existing E1 scorer" -- this function does not reimplement
    ranking or corruption, only wires the existing pieces to this run's encoder.
    """
    items = build_corrupted_battery(holdout, corruption_seed=E1_CORRUPTION_SEED)
    if not items:
        return {"recall@1": 0.0, "mrr": 0.0, "n_items": 0.0, "chance": 0.0, "skipped": True}
    questions = [it.question for it in items]
    candidates: list[str] = []
    for it in items:
        candidates.append(it.true_derivation)
        candidates.extend(it.corruptions)

    def tokenize(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        return _tokenize(tok, texts, 256, device)

    import numpy as np
    import torch.nn.functional as F

    was_training = target_encoder.training
    target_encoder.eval()
    with torch.no_grad():
        q_chunks, c_chunks = [], []
        for i in range(0, len(questions), eval_batch):
            ids, mask = tokenize(questions[i : i + eval_batch])
            q_chunks.append(F.normalize(target_encoder(ids, mask), dim=-1))
        for i in range(0, len(candidates), eval_batch):
            ids, mask = tokenize(candidates[i : i + eval_batch])
            c_chunks.append(F.normalize(target_encoder(ids, mask), dim=-1))
    if was_training:
        target_encoder.train()
    q_vec = torch.cat(q_chunks).cpu().numpy().astype(np.float64)
    c_vec = torch.cat(c_chunks).cpu().numpy().astype(np.float64).reshape(len(items), 5, -1)
    ranked = cosine_hit_rate(q_vec, c_vec, k=4, tie_seed=E1_CORRUPTION_SEED)
    tfidf = score_items_lexical(items, scorer="tfidf")
    bm25 = score_items_lexical(items, scorer="bm25")
    wrong = score_wrong_problem_tfidf(items)
    gates = control_gates(tfidf["recall@1"], bm25["recall@1"], wrong["recall@1"])
    return {**ranked, "gates": gates, "n_items": float(len(items))}


def main() -> int:
    """Parse args, build the split, pre-register, train, score, write the receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default="latent-step")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--predictor-hidden", type=int, default=128)
    parser.add_argument("--predictor-depth", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=0, help="0 = steps//6")
    parser.add_argument("--eval-batch", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "receipts")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dry = bool(args.dry_run)
    steps = 2 if dry else args.steps
    eval_every = args.eval_every or max(1, steps // 6)
    blind = args.arm == "sequence-blind"

    torch.manual_seed(args.seed)  # FIRST statement that touches torch RNG -- see docstring.

    device = torch.device("cpu")
    if not dry:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        refuse_if_gpu_busy(os.getpid())
        if not torch.cuda.is_available():
            raise RuntimeError("E5 training is GPU-first; CUDA unavailable after pinning device 0")
        device = torch.device("cuda:0")

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()

    cfg = build_prereg_config(steps=args.steps, batch_size=args.batch_size, seed=args.seed)
    holdout, train_pairs, split_meta = build_splits(cfg)
    fp = str(split_meta.get("corpus_fingerprint", ""))
    if fp != EXPECTED_FP:
        raise RuntimeError(f"corpus fingerprint {fp} != reason baseline {EXPECTED_FP}")
    if len(train_pairs) != EXPECTED_TRAIN_PAIRS:
        raise RuntimeError(f"train_pairs {len(train_pairs)} != {EXPECTED_TRAIN_PAIRS}")
    if int(split_meta.get("duplicates_removed", -1)) != EXPECTED_DUPES:
        raise RuntimeError(f"duplicates_removed != {EXPECTED_DUPES}")

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(cfg.tokenizer_path)

    encoder_cfg = TextEncoderConfig(**{**asdict(cfg.encoder), "vocab_size": tok.get_vocab_size()})
    predictor_cfg = LatentStepConfig(
        dim=encoder_cfg.out_dim or encoder_cfg.dim,
        hidden=args.predictor_hidden,
        depth=args.predictor_depth,
    )
    model = LatentStepModel(encoder_cfg, predictor_cfg).to(device)
    trunk_params = sum(p.numel() for p in model.encoder.parameters())
    predictor_params = sum(p.numel() for p in model.predictor.parameters())

    train_examples = build_step_examples(train_pairs, min_steps=MIN_STEPS)
    holdout_examples = build_step_examples(holdout, min_steps=MIN_HOLDOUT_STEPS)
    battery = build_step_battery(holdout_examples, corruption_seed=0)
    print(
        f"reason-e5 arm={args.arm} seed={args.seed} steps={steps} batch={args.batch_size} "
        f"train_examples={len(train_examples)} holdout_examples={len(holdout_examples)} "
        f"battery_items={len(battery)} dry_run={dry}",
        flush=True,
    )

    def tokenize(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        return _tokenize(tok, texts, cfg.max_len, device)

    untrained_predictor_battery = score_step_battery(
        model, tokenize, battery, batch=args.eval_batch, tie_seed=0
    )
    untrained_collapse = collapse_stats(
        model.encode_target(*tokenize([ex.question for ex in holdout_examples]))
        if holdout_examples
        else torch.zeros(0, encoder_cfg.out_dim or encoder_cfg.dim)
    )
    untrained_e1 = _score_e1_regression_guard(
        model.target_encoder, tok, holdout, device=device, eval_batch=args.eval_batch
    )
    untrained_diagonal = evaluate(model.target_encoder, tok, holdout, cfg.max_len, device)

    prereg_path = write_prereg(
        out_dir=args.out_dir,
        arm=args.arm,
        seed=args.seed,
        steps=steps,
        batch_size=args.batch_size,
        split_meta=split_meta,
        n_train_examples=len(train_examples),
        n_holdout_examples=len(holdout_examples),
        n_battery_items=len(battery),
        untrained_predictor_battery=untrained_predictor_battery,
    )
    print(f"pre-registration written BEFORE training: {prereg_path}", flush=True)

    opt = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()), lr=cfg.lr
    )
    history: list[dict[str, float]] = []
    n_skipped_steps = 0
    model.train()
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, cfg)
        lo = (step * args.batch_size) % max(1, len(train_pairs) - args.batch_size)
        hi = min(lo + args.batch_size, len(train_pairs))
        batch_pairs = train_pairs[lo:hi]
        examples = build_step_examples(batch_pairs, min_steps=MIN_STEPS)
        windows = enumerate_step_windows(examples, blind=blind)
        if len(windows) < 1:
            n_skipped_steps += 1
            continue
        ctx_ids, ctx_mask = tokenize([w.context for w in windows])
        tgt_ids, tgt_mask = tokenize([w.target for w in windows])
        loss, stats = model(ctx_ids, ctx_mask, tgt_ids, tgt_mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.encoder.parameters()) + list(model.predictor.parameters()), GRAD_CLIP
        )
        opt.step()
        model.update_target(ema_at(step, steps, predictor_cfg.ema_base, predictor_cfg.ema_final))
        if step % eval_every == 0 or step == steps - 1:
            history.append(
                {
                    "step": float(step),
                    "lr": _lr_at(step, cfg),
                    "n_windows": float(len(windows)),
                    **stats,
                }
            )

    elapsed = time.time() - t0
    model.eval()
    final_predictor_battery = score_step_battery(
        model, tokenize, battery, batch=args.eval_batch, tie_seed=0
    )
    final_collapse = collapse_stats(
        model.encode_target(*tokenize([ex.question for ex in holdout_examples]))
        if holdout_examples
        else torch.zeros(0, encoder_cfg.out_dim or encoder_cfg.dim)
    )
    final_e1 = _score_e1_regression_guard(
        model.target_encoder, tok, holdout, device=device, eval_batch=args.eval_batch
    )
    final_diagonal = evaluate(model.target_encoder, tok, holdout, cfg.max_len, device)

    ckpt_dir = args.out_dir / "reason-e5-checkpoints" / f"{args.arm}-s{args.seed}"
    final_ckpt = ckpt_dir / "final.pt"
    atomic_save(
        {
            "schema": "csd-reason-e5-checkpoint/v1",
            "model": model.state_dict(),
            "encoder_cfg": asdict(encoder_cfg),
            "predictor_cfg": asdict(predictor_cfg),
            "arm": args.arm,
            "seed": args.seed,
            "step": steps,
        },
        final_ckpt,
    )
    checkpoint_sha256 = sha256_file(final_ckpt)

    receipt: dict[str, Any] = {
        "schema": "csd-reason-e5-receipt/v1",
        "kind": "train",
        "experiment": "E5",
        "region": "reason",
        "arm": args.arm,
        "blind": blind,
        "started_utc": started_utc,
        "recorded": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": (
            "latent-step prediction: predict step t+1's target-encoder latent from "
            f"steps <= t via a shared trunk, smooth-L1 loss, EMA target ({
                'blind: context is the question alone' if blind else 'context is question+steps<=t'
            })"
        ),
        "corpus_fingerprint": fp,
        "corpus_fingerprint_scheme": CORPUS_FINGERPRINT_SCHEME,
        "split": split_meta.get("split"),
        "batch_order": split_meta.get("batch_order"),
        "config": {
            "steps": steps,
            "requested_steps": args.steps,
            "batch_size": args.batch_size,
            "lr": cfg.lr,
            "warmup_steps": cfg.warmup_steps,
            "max_len": cfg.max_len,
            "seed": args.seed,
            "split_seed": cfg.split_seed,
            "order_seed": cfg.order_seed,
            "min_steps": MIN_STEPS,
            "min_holdout_steps": MIN_HOLDOUT_STEPS,
            "encoder": asdict(encoder_cfg),
            "predictor": asdict(predictor_cfg),
        },
        "parameters": {
            "trunk": trunk_params,
            "predictor": predictor_params,
            "total": trunk_params + predictor_params,
        },
        "counts": {
            "train_examples": len(train_examples),
            "holdout_examples": len(holdout_examples),
            "battery_items": len(battery),
            "battery_fingerprint": battery_fingerprint(battery),
            "steps_skipped_no_windows": n_skipped_steps,
        },
        "device": str(device),
        "elapsed_s": round(elapsed, 1),
        "dry_run": dry,
        "history": history,
        "untrained_predictor_battery": untrained_predictor_battery,
        "predictor_battery": final_predictor_battery,
        "untrained_collapse": untrained_collapse,
        "collapse": final_collapse,
        "untrained_e1_regression_guard": untrained_e1,
        "e1_regression_guard": final_e1,
        "untrained_diagonal_recall": {
            "recall@1": untrained_diagonal["recall@1"],
            "recall@10": untrained_diagonal["recall@10"],
        },
        "diagonal_recall_reference_only": {
            "recall@1": final_diagonal["recall@1"],
            "recall@10": final_diagonal["recall@10"],
        },
        "untrained_baseline_seed": args.seed,
        "prereg_path": str(prereg_path),
        "go_kill_reference": {
            "hypothesis": PREREG_HYPOTHESIS,
            "go": PREREG_GO,
            "kill": PREREG_KILL,
            "this_run": go_kill_note(final_predictor_battery["recall@1"], None),
        },
        "checkpoint": str(final_ckpt),
        "checkpoint_sha256": checkpoint_sha256,
    }
    path = write_receipt(
        receipt,
        args.out_dir,
        f"reason-e5-{args.arm}-s{args.seed}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json",
    )
    print(
        f"predictor_battery.recall@1={final_predictor_battery['recall@1']:.4f} "
        f"(untrained {untrained_predictor_battery['recall@1']:.4f}) "
        f"e1_guard.recall@1={final_e1['recall@1']:.4f} "
        f"diagonal_recall@1={final_diagonal['recall@1']:.4f} "
        f"elapsed_s={elapsed:.1f} -> {path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
