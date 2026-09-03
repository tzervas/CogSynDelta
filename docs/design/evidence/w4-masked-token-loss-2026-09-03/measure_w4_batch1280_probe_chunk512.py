"""VRAM re-probe #2: memory region, 20 steps, batch_size=1280, max_len=96, terms ON
(memory_config()'s own defaults), `token_loss_chunk=512` -- the follow-up to
`measure_w4_batch1280_probe_chunked.py` (chunk=2048, `token_loss_chunk`'s own default)
in this directory, which completed all 20 steps with no OutOfMemoryError but whose
driver-observed peak (22,120 MiB) narrowly exceeded the task's own fits threshold
(20,980 MiB = 23,028 - 2,048) -- a near miss, not a clean fit (see README.md section 4).
This script is IDENTICAL to that one except `TOKEN_LOSS_CHUNK` and `OUT_DIR`, to isolate
the chunk-size change as the only variable. Same measurement shape as
`measure_w4_batch1280_probe.py` and
docs/design/evidence/w4-control-arm-2026-09-03/measure_w4_control_arm.py -- called as
`pretrain_region(memory_config(...))` directly, not `run_memory_pretrain`, so this
measures the training step's own VRAM, not the (much heavier) 57,638-passage BEIR
full-pool eval `run_memory_pretrain` layers on top. A scratch out_dir under
/akula-data/session-backup-staging/w4-chunked/probe512, never /akula-data/csd/receipts,
started from an EMPTY directory so no stale checkpoint can silently resume the run (the
chunk=2048 probe's own README notes this happened once and was discarded).

Run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in the environment (read at
CUDA init, so it must be set before the interpreter starts, not inside this script) and
PYTHONPATH pointed at this worktree's src/ (the shared venv's editable install resolves
`cogsyndelta` to the main repo checkout, which does not have `token_loss_chunk` at all)
-- see the accompanying shell wrapper. A second process samples `nvidia-smi
--query-gpu=memory.used` at 0.5s while this runs, for the DRIVER-reported peak alongside
`torch.cuda.max_memory_allocated`/`max_memory_reserved` (the allocator's own view, which
undercounts anything CUDA itself holds outside PyTorch's caching allocator).
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

OUT_DIR = Path("/akula-data/session-backup-staging/w4-chunked/probe512")

# The value under test in this re-probe -- half of TOKEN_LOSS_CHUNK's own 2048 default,
# named explicitly here (not left implicit) so this probe's config is legible on its own.
TOKEN_LOSS_CHUNK = 512


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
        token_loss_chunk=TOKEN_LOSS_CHUNK,
    )
    assert cfg.token_loss_weight > 0.0 and cfg.decorr_weight > 0.0, "terms must be on"
    assert cfg.token_loss_chunk == TOKEN_LOSS_CHUNK

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    summary: dict = {
        "probe": "w4-masked-token-loss VRAM re-probe #2, batch=1280, CHUNKED (chunk=512)",
        "steps": cfg.steps,
        "batch_size": cfg.batch_size,
        "max_len": cfg.max_len,
        "token_loss_weight": cfg.token_loss_weight,
        "decorr_weight": cfg.decorr_weight,
        "token_loss_chunk": cfg.token_loss_chunk,
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
        summary_path = OUT_DIR / "batch1280-chunk512-memory-summary.txt"
        summary_path.write_text(torch.cuda.memory_summary(abbreviated=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "batch1280-chunk512-summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}", flush=True)
    if not summary.get("fit", True):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
