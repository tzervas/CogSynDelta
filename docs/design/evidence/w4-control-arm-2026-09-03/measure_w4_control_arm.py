"""W4 control-arm evidence (COMMIT 3, B2): the SAME 50-step, batch_size=512 smoke-run
shape as the original MEASURED_VRAM_AT_BATCH_512 measurement, on the real fleet corpus,
run THREE ways with the now-W1-aligned rank harness (commit "fix(memory): align W4's
final-block rank measurement with W1's pre-committed harness"):

  1. control      -- token_loss_weight=0.0, decorr_weight=0.0 (both off)
  2. token_only   -- token_loss_weight=0.1 (memory_config's own TOKEN_LOSS_WEIGHT), decorr_weight=0.0
  3. both_on      -- token_loss_weight=0.1, decorr_weight=0.1 (memory_config's own defaults)

One arm per argv[1] in {"control", "token_only", "both_on"}, run as a separate process
each so CUDA memory is fully released between arms (the 3090 Ti is shared with a
concurrent quantize job). Calls `pretrain_region(memory_config(...))` directly -- NOT
`run_memory_pretrain` -- to skip the full 57,638-passage BEIR/BM25 eval, which the rank
evidence does not need and which would cost far more GPU time than the 50-step smoke run
itself.

`out_dir` is a scratch path, never `/akula-data/csd/receipts` (the real fleet location) --
these are throwaway smoke-run checkpoints (~192MB each), never meant to be loaded again
after this evidence is recorded.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(
    0, "/home/kang/code/personal/tzervas/csd-worktress/CogSynDelta-wt-w4-memory-merge/src"
)

import torch

from cogsyndelta.regions.memory import memory_config
from cogsyndelta.regions.pretrain import pretrain_region
from cogsyndelta.regions.text_encoder import TextEncoderConfig

OUT_ROOT = Path("/akula-data/session-backup-staging/w4-control-arm-2026-09-03/control-arm-runs")

ARMS = {
    "control": {"token_loss_weight": 0.0, "decorr_weight": 0.0},
    "token_only": {"token_loss_weight": 0.1, "decorr_weight": 0.0},
    "both_on": {"token_loss_weight": 0.1, "decorr_weight": 0.1},
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ARMS:
        raise SystemExit(f"usage: control_arm_run.py {{{','.join(ARMS)}}}")
    name = sys.argv[1]
    weights = ARMS[name]

    assert torch.cuda.is_available(), "this evidence run requires the 3090 Ti"

    cfg = memory_config(
        steps=50,
        batch_size=512,
        max_len=96,
        device="cuda",
        checkpoint_every=0,  # smoke run only; final.pt is always written regardless
        encoder=TextEncoderConfig(dim=256, depth=4, n_heads=4, max_len=96),
        out_dir=str(OUT_ROOT / name),
        **weights,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    receipt = pretrain_region(cfg)
    elapsed = time.time() - t0

    peak_allocated_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_reserved_mib = torch.cuda.max_memory_reserved() / (1024 * 1024)

    rank = receipt["token_aware"]["final_block_rank"]
    ratio = (
        rank["token_global_pr_rank"] / rank["pooled_pr_rank"]
        if rank["pooled_pr_rank"]
        else float("nan")
    )

    summary = {
        "arm": name,
        "token_loss_weight": weights["token_loss_weight"],
        "decorr_weight": weights["decorr_weight"],
        "wall_elapsed_s": round(elapsed, 2),  # includes corpus load + tokenisation
        "training_loop_elapsed_s": receipt["elapsed_s"],  # receipt's own, loop only
        "mean_step_time_ms": round(1000 * receipt["elapsed_s"] / cfg.steps, 2),
        "peak_allocated_mib": round(peak_allocated_mib, 1),
        "peak_reserved_mib": round(peak_reserved_mib, 1),
        "final_block_rank": rank,
        "token_global_over_pooled_ratio": ratio,
        "gate_e_pr_clause_passes_at_2x": bool(ratio >= 2.0) if not math.isnan(ratio) else None,
        "parameters": receipt["parameters"],
        "checkpoint": receipt["checkpoint"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "config_fingerprint": receipt["corpus"]["fingerprint"],
        "seed": cfg.seed,
        "device": receipt["device"],
        "precision": receipt["precision"],
        "held_out_recall@1": receipt["held_out"]["recall@1"],
        "graded_held_out_spearman": receipt.get("graded_held_out", {}).get("spearman"),
        "encoder_config": asdict(cfg.encoder),
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"{name}-summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
