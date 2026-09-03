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

:func:`load_checkpoint` is the fourth thing that lives here once rather than as several
copies (design row W0c, DEC-40): the one production entry point for `torch.load` on a
checkpoint file, verifying a content hash BEFORE the file is ever opened when the caller
has one to check against. See its own docstring for the threat model, and
`tests/test_checkpoint_loader_lint.py` for the guard that keeps every other production
`torch.load` of a checkpoint routed through it instead of called directly.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from time import time_ns
from typing import Any

import torch

_STEP_RE = re.compile(r"step-(\d+)\.pt$")


class ChecksumMismatchError(RuntimeError):
    """A checkpoint's content hash did not match the hash the caller expected.

    Raised by :func:`load_checkpoint` BEFORE `torch.load` ever opens the file -- a
    mismatch never reaches the unpickler, malicious or not.
    """


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Content hash of a file on disk, read in chunks rather than loaded whole."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
    map_location: str | torch.device = "cpu",
    sha256_out: list[str] | None = None,
) -> dict[str, Any]:
    """The one production entry point for `torch.load` on a checkpoint file.

    `weights_only` is deliberately NOT a parameter: the underlying `torch.load` call
    below is hardcoded to `weights_only=True`, so there is no argument a caller (or a
    future edit) can pass to turn it off. bandit's B614 (`Use of unsafe PyTorch load`)
    flags `torch.load` whenever `weights_only` is a variable rather than a literal
    `True` -- correctly, since a variable forwarded from a caller has no guarantee of
    ever being `True` at every call site. Removing the parameter, rather than trusting
    every current and future caller to keep passing `True`, is what actually closes
    that gap; every production caller already passed `True` explicitly (see
    `tests/test_checkpoint_load_security.py`) or relied on this default, so the guard
    was already load-bearing here.

    WHY A HASH CHECK BEFORE `torch.load`, NOT JUST `weights_only=True`
    `tests/test_checkpoint_load_security.py` covers a DIFFERENT threat: a checkpoint path
    read out of a receipt JSON on an NFS export mounted `rw,no_root_squash` could point at
    a hostile pickle payload that runs code the moment it is unpickled --
    `weights_only=True` closes that by refusing to construct anything outside a small
    allow-list of tensor/container/numeric types.

    That guard proves the bytes cannot run code. It does not prove they are the SAME
    bytes a caller who recorded an expected hash (a receipt's `checkpoint_sha256`, most
    often) is expecting. A checkpoint silently overwritten, truncated, or swapped for a
    different -- still `weights_only`-safe -- file after that hash was recorded loads
    without complaint under the old per-call-site pattern; the numbers it produces would
    then describe weights the receipt does not actually name. Hashing the whole file
    before `torch.load` ever opens it closes that: a mismatch is refused outright, before
    a single byte reaches the unpickler.

    Every production `torch.load` of a checkpoint in this project routes through here --
    `load_resumable` below, `scripts/csd-quantize.py`, `scripts/csd-benchmark.py` -- and
    `tests/test_checkpoint_loader_lint.py` greps the source tree to keep it that way: a
    stray `torch.load(` outside this function is a checkpoint that skipped the check, not
    a style violation.

    Args:
        path: The checkpoint file.
        expected_sha256: If given, `path`'s content hash must match BEFORE the file is
            ever handed to `torch.load` -- a mismatch raises without attempting to load
            it. `None` (the default) skips the check: not every caller has a prior hash
            to verify against (`load_resumable`, resuming its own training loop, has no
            independently-recorded hash for the checkpoint it is about to read) -- for
            those callers this function is still the one place `torch.load` is invoked,
            which is what the lint enforces.
        map_location: Forwarded to `torch.load`.
        sha256_out: If given, this function appends the checkpoint's content hash to
            it -- the same hash `expected_sha256` was checked against, when one was
            given, or freshly computed here when it was not. Exists so a caller that
            needs the hash for its own purposes (a receipt recording which exact
            bytes it evaluated, e.g. `scripts/csd-benchmark.py` /
            `scripts/csd-quantize.py`) does not have to hash the file a second time
            with `sha256_file` after this function already did. A list rather than a
            single mutable "out" value because Python has no plain by-reference `str`
            out-param; append-and-read-`[0]` is the idiom. Left `None` (the default)
            costs nothing extra: `expected_sha256 is None` and `sha256_out is None`
            together skip hashing entirely, exactly as before this parameter existed.

    Returns:
        The loaded checkpoint dict, exactly as `torch.load` returns it.

    Raises:
        FileNotFoundError: `path` does not exist.
        ChecksumMismatchError: `expected_sha256` was given and did not match.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"load_checkpoint: no such file: {p}")
    actual: str | None = None
    if expected_sha256 is not None or sha256_out is not None:
        actual = sha256_file(p)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ChecksumMismatchError(
            f"checkpoint {p} sha256 {actual} does not match expected "
            f"{expected_sha256} -- refusing to load. The file may have been "
            f"overwritten, truncated, or replaced since the expected hash was "
            f"recorded."
        )
    if sha256_out is not None:
        assert actual is not None  # computed above whenever sha256_out is not None
        sha256_out.append(actual)
    return torch.load(p, map_location=map_location, weights_only=True)


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
    ckpt_dir: Path,
    fingerprint: str,
    fields: dict[str, Any],
    *,
    allow_unfingerprinted: bool = True,
) -> dict[str, Any] | None:
    """Load a resumable checkpoint, or return None if there is nothing valid to use.

    Args:
        ckpt_dir: Where a region's periodic checkpoints and `final.pt` live.
        fingerprint: The CURRENT config's fingerprint (see each harness's
            ``_config_fingerprint``).
        fields: The current config's resume-relevant fields, plain-JSON-able, used only
            to build a readable diff on a mismatch.
        allow_unfingerprinted: What to do with a checkpoint that predates fingerprinted
            checkpoints entirely (no `config_fingerprint` key at all). `True` (the
            default HERE) reproduces this function's original behaviour -- print a
            notice and start fresh -- which is what `regions/vl_pretrain.py` and
            `regions/classify_pretrain.py` still get by calling this positionally and
            not opting in to the stricter mode. `regions/pretrain.py` passes `False` by
            default (`PretrainConfig.allow_unfingerprinted_resume`): an unfingerprinted
            checkpoint sitting in a region's checkpoint directory is indistinguishable
            from one written by code too old to fingerprint anything, and "start fresh,
            leave it there" is exactly the shape of incident this whole mechanism now
            guards against -- a checkpoint from an unidentifiable vintage, sitting where
            a future resume could still find it. Refusing forces an operator to look at
            the file and either delete it or explicitly accept the fresh start with
            `allow_unfingerprinted_resume=True` (`--allow-unfingerprinted-resume`).

    Returns:
        The loaded checkpoint dict (with `_path` added, naming the file it came from),
        or None if `ckpt_dir` holds nothing usable -- either it does not exist, or the
        only checkpoint present predates resumable training and `allow_unfingerprinted`
        is `True`.

    Raises:
        ValueError: A checkpoint exists and is well-formed, but either (a) was trained
            under a DIFFERENT config -- silently starting fresh would leave a stale
            checkpoint sitting in `ckpt_dir` to confuse the next attempt, silently
            resuming would produce a run that is neither the old config nor the new one
            -- or (b) carries no fingerprint at all and `allow_unfingerprinted` is
            `False`. Either way the only honest move is to say what differs (or that
            nothing on the file distinguishes it) and stop.
    """
    path = latest_checkpoint_path(ckpt_dir)
    if path is None:
        return None
    ckpt = load_checkpoint(path, map_location="cpu")
    if "config_fingerprint" not in ckpt:
        if allow_unfingerprinted:
            print(
                f"    found {path} but it predates resumable training (no config "
                f"fingerprint recorded) -- starting fresh. Move or delete {ckpt_dir} to "
                f"silence this.",
                flush=True,
            )
            return None
        raise ValueError(
            f"refusing to resume from {path}: it has no config_fingerprint recorded at "
            f"all (predates fingerprinted checkpoints, or fingerprinting was bypassed "
            f"when it was written), and this call refuses such a checkpoint by default "
            f"rather than silently starting fresh next to it -- an unfingerprinted "
            f"checkpoint left in place is exactly what let one training vintage keep "
            f"sitting beside another with nothing on disk to tell them apart. Current "
            f"config fingerprint is {fingerprint}. Move or delete {ckpt_dir}, or pass "
            f"allow_unfingerprinted_resume=True (--allow-unfingerprinted-resume) if you "
            f"intend to start fresh and leave it."
        )
    if ckpt["config_fingerprint"] != fingerprint:
        diff = describe_config_diff(ckpt.get("config_fields", {}), fields)
        raise ValueError(
            f"refusing to resume from {path}: it was trained under a different config "
            f"than the one requested now (checkpoint fingerprint "
            f"{ckpt['config_fingerprint']}, current config fingerprint {fingerprint}).\n"
            f"{diff}\n"
            f"Move or delete {ckpt_dir}, or fix the config, before retrying."
        )
    ckpt["_path"] = path
    return ckpt
