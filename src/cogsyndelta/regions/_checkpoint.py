"""Shared, model-agnostic checkpoint mechanics for the pretrain harnesses.

Both ``regions/pretrain.py`` (text) and ``regions/vl_pretrain.py`` (I-JEPA) need the same
three things done the same way: an atomic write so a crash mid-save never leaves a
truncated file under the checkpoint's real name, rotation that keeps a bounded number of
recent periodic checkpoints, and a config-fingerprint check that REFUSES to resume from a
checkpoint trained under different settings rather than guessing. None of that logic
touches a specific model, optimizer, or config shape -- it lives here once rather than as
two copies that will eventually drift.

What is deliberately NOT here: anything about what goes INSIDE a checkpoint (model
state_dict, RNG state, the untrained baseline, ...). That is per-harness, built by each
caller into a plain dict before it ever reaches :func:`atomic_save`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from time import time_ns
from typing import Any

import torch

_STEP_RE = re.compile(r"step-(\d+)\.pt$")


def atomic_save(obj: dict[str, Any], path: Path) -> None:
    """Write a checkpoint so a crash mid-write never leaves a truncated file at `path`.

    Writes to a temp file in the SAME directory (so `os.replace` is a same-filesystem
    rename, which POSIX guarantees is atomic) then replaces. `torch.load` on `path`
    therefore always sees either the complete previous checkpoint or the complete new
    one, never a partial write -- the crash-during-a-184MB-write case this mechanism
    exists to survive.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}-{time_ns()}"
    try:
        torch.save(obj, tmp)
        tmp.replace(path)  # same-filesystem rename; POSIX guarantees this is atomic
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def numbered_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """`step-NNNNNN.pt` files under `ckpt_dir`, oldest first.

    Never matches `final.pt` or an in-flight `.tmp-*` file from :func:`atomic_save` --
    both fail the `step-<digits>.pt` pattern.
    """
    numbered = []
    for p in ckpt_dir.glob("step-*.pt"):
        m = _STEP_RE.match(p.name)
        if m:
            numbered.append((int(m.group(1)), p))
    numbered.sort(key=lambda t: t[0])
    return numbered


def latest_checkpoint_path(ckpt_dir: Path) -> Path | None:
    """The checkpoint to try resuming from, or None if there is none.

    `final.pt` means a previous run already completed the whole configured step count --
    strictly more complete than any periodic checkpoint could be -- so it wins outright
    when present.
    """
    if not ckpt_dir.is_dir():
        return None
    final = ckpt_dir / "final.pt"
    if final.is_file():
        return final
    numbered = numbered_checkpoints(ckpt_dir)
    return numbered[-1][1] if numbered else None


def rotate_checkpoints(ckpt_dir: Path, keep: int) -> None:
    """Delete periodic checkpoints beyond the most recent `keep`, oldest first.

    Call this only AFTER the new checkpoint is safely on disk (i.e. after
    :func:`atomic_save` has returned), so the newest file always exists before an older
    one is removed -- a crash here costs one now-superfluous old file, never the run's
    only valid checkpoint. `final.pt` is never a candidate: it is not matched by the
    `step-*.pt` glob :func:`numbered_checkpoints` uses.
    """
    numbered = numbered_checkpoints(ckpt_dir)
    doomed = numbered[:-keep] if keep > 0 else numbered
    for _, p in doomed:
        p.unlink(missing_ok=True)


def describe_config_diff(old: dict[str, Any], new: dict[str, Any]) -> str:
    """Name exactly which fields differ, so a refusal is actionable rather than a hash."""
    lines = [
        f"  {key}: checkpoint={old.get(key)!r}  config={new.get(key)!r}"
        for key in sorted(set(old) | set(new))
        if old.get(key) != new.get(key)
    ]
    return "\n".join(lines) if lines else "  (fingerprints differ; no field-level diff found)"


def load_resumable(
    ckpt_dir: Path, fingerprint: str, fields: dict[str, Any]
) -> dict[str, Any] | None:
    """Load a resumable checkpoint, or return None if there is nothing valid to use.

    Args:
        ckpt_dir: Where a region's periodic checkpoints and `final.pt` live.
        fingerprint: The CURRENT config's fingerprint (see each harness's
            ``_config_fingerprint``).
        fields: The current config's resume-relevant fields, plain-JSON-able, used only
            to build a readable diff on a mismatch.

    Returns:
        The loaded checkpoint dict (with `_path` added, naming the file it came from),
        or None if `ckpt_dir` holds nothing usable -- either it does not exist, or the
        only checkpoint present predates resumable training (no fingerprint recorded).

    Raises:
        ValueError: A checkpoint exists and is well-formed, but was trained under a
            DIFFERENT config. Silently starting fresh anyway would leave a stale
            checkpoint sitting in `ckpt_dir` to confuse the next attempt; silently
            resuming from it would produce a run that is neither the old config nor the
            new one. Either way the only honest move is to say what differs and stop.
    """
    path = latest_checkpoint_path(ckpt_dir)
    if path is None:
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "config_fingerprint" not in ckpt:
        print(
            f"    found {path} but it predates resumable training (no config "
            f"fingerprint recorded) -- starting fresh. Move or delete {ckpt_dir} to "
            f"silence this.",
            flush=True,
        )
        return None
    if ckpt["config_fingerprint"] != fingerprint:
        diff = describe_config_diff(ckpt.get("config_fields", {}), fields)
        raise ValueError(
            f"refusing to resume from {path}: it was trained under a different "
            f"config than the one requested now.\n{diff}\n"
            f"Move or delete {ckpt_dir}, or fix the config, before retrying."
        )
    ckpt["_path"] = path
    return ckpt
