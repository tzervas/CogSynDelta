"""VRAM re-probe: memory region, 20 steps, batch_size=1280, max_len=96, terms ON
(memory_config()'s own defaults), `token_loss_chunk` set explicitly (see below) --
the follow-up to `measure_w4_batch1280_probe.py` in this directory, which OOM'd at this
batch under the unchunked (`token_loss_chunk=0`-equivalent, pre-this-change) form of
`_mlm_token_loss`. Same measurement shape as that script and as
docs/design/evidence/w4-control-arm-2026-09-03/measure_w4_control_arm.py -- called as
pretrain_region(memory_config(...)) directly, not run_memory_pretrain, so this measures
the training step's own VRAM, not the (much heavier) 57,638-passage BEIR full-pool eval
run_memory_pretrain layers on top. A scratch out_dir under
/akula-data/session-backup-staging/w4-chunked/probe, never /akula-data/csd/receipts.

Run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True in the environment (read at
CUDA init, so it must be set before the interpreter starts, not inside this script) --
see the accompanying shell wrapper. A second process samples `nvidia-smi
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

OUT_DIR = Path("/akula-data/session-backup-staging/w4-chunked/probe")

# 2048 is PretrainConfig.token_loss_chunk's own default (see that field's docstring) --
# named explicitly here rather than left implicit, so this probe's config is legible on
# its own without cross-referencing the dataclass default.
TOKEN_LOSS_CHUNK = 2048


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
        "probe": "w4-masked-token-loss VRAM re-probe, batch=1280, CHUNKED",
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
        # `memory_summary()`'s tables report the SAME historical peak stats
        # `max_memory_allocated`/`max_memory_reserved` return, regardless of when it is
        # called (they are running highs, not live state) -- so capturing it here, after
        # the loop, still reflects the peak even though execution has moved past it. This
        # is what "report where the memory goes" needs whether or not this probe's own
        # `fit` (torch-allocator-OOM-based) ends up true: `torch.cuda.max_memory_reserved`
        # alone says a NUMBER, not WHERE that number's bytes are — the top-allocations
        # breakdown here is what answers that when the driver-observed peak disagrees.
        summary_path = OUT_DIR / "batch1280-chunked-memory-summary.txt"
        summary_path.write_text(torch.cuda.memory_summary(abbreviated=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "batch1280-chunked-summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}", flush=True)
    if not summary.get("fit", True):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
