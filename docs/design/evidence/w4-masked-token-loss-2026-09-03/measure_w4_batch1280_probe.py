"""VRAM probe: memory region, 20 steps, batch_size=1280, max_len=96, terms ON
(memory_config()'s own defaults), same measurement shape as
docs/design/evidence/w4-control-arm-2026-09-03/measure_w4_control_arm.py -- called as
pretrain_region(memory_config(...)) directly, not run_memory_pretrain, so this measures
the training step's own VRAM, not the (much heavier) 57,638-passage BEIR full-pool eval
run_memory_pretrain layers on top. A scratch out_dir under
/akula-data/session-backup-staging/w4-masked/probe, never /akula-data/csd/receipts.

GPU_PACK_PROBE is not honoured anywhere in this repo (grep confirmed) -- this is the
"--steps 20 into a scratch state dir" fallback the task specified.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from cogsyndelta.regions.memory import memory_config
from cogsyndelta.regions.pretrain import pretrain_region
from cogsyndelta.regions.text_encoder import TextEncoderConfig

OUT_DIR = Path("/akula-data/session-backup-staging/w4-masked/probe")


def main() -> None:
    assert torch.cuda.is_available(), "this probe requires the 3090 Ti"

    cfg = memory_config(
        steps=20,
        batch_size=1280,
        max_len=96,
        device="cuda",
        checkpoint_every=0,
        eval_every=1000,  # step 0 and step steps-1 still evaluate regardless; no extra evals
        encoder=TextEncoderConfig(dim=256, depth=4, n_heads=4, max_len=96),
        out_dir=str(OUT_DIR),
    )
    assert cfg.token_loss_weight > 0.0 and cfg.decorr_weight > 0.0, "terms must be on"

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    summary: dict = {
        "probe": "w4-masked-token-loss VRAM probe, batch=1280",
        "steps": cfg.steps,
        "batch_size": cfg.batch_size,
        "max_len": cfg.max_len,
        "token_loss_weight": cfg.token_loss_weight,
        "decorr_weight": cfg.decorr_weight,
        "encoder_config": asdict(cfg.encoder),
    }
    try:
        receipt = pretrain_region(cfg)
        elapsed = time.time() - t0
        summary.update(
            {
                "fit": True,
                "wall_elapsed_s": round(elapsed, 2),
                "training_loop_elapsed_s": receipt["elapsed_s"],
                "mean_step_time_ms": round(1000 * receipt["elapsed_s"] / cfg.steps, 2),
                "parameters": receipt["parameters"],
                "config_fingerprint": receipt["corpus"]["fingerprint"],
                "device": receipt["device"],
                "precision": receipt["precision"],
            }
        )
    except torch.OutOfMemoryError as exc:
        elapsed = time.time() - t0
        summary.update(
            {
                "fit": False,
                "wall_elapsed_s_before_oom": round(elapsed, 2),
                "oom_error": str(exc),
            }
        )
    finally:
        # Peak stats reflect the highest successful allocation reached before any OOM,
        # which is exactly the number we want whether the run completed or crashed.
        summary["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        summary["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "batch1280-summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}", flush=True)
    if not summary.get("fit", True):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
