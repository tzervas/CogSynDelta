#!/usr/bin/env python3
"""Keep session/temp filesystems from filling with training artifacts.

WHY
Three parallel training runs wrote 184 MB checkpoints each into a Claude Code session
scratchpad on /tmp. The filesystem hit 0 bytes free, and the failure mode was worse than
losing the runs: every subsequent shell command failed with ENOSPC while trying to write
its own stdout, so even the cleanup commands could not report what they had done.

The root cause was a relative default path -- checkpoints landed wherever the process
happened to be. That is fixed at the source (see DURABLE_ROOTS below and
csd-train-all.py). This reaper is the safety net for anything that slips through, and for
transcripts and caches that legitimately live in temp but should not accumulate.

POLICY
- NEVER touch a durable root. Checkpoints and receipts under /akula-data or /bulk are
  results, not garbage.
- Only reap what is regenerable: model weights in a temp dir, agent transcripts, caches.
- Age-gate everything. A file being large is not evidence it is finished; a run in
  progress writes large files.
- Report bytes freed. A cleaner that silently deletes is indistinguishable from a bug.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Results live here. Never reaped, regardless of size or age.
DURABLE_ROOTS = ("/akula-data", "/bulk", "/data", "/home/kang/code")

# (glob, min age in hours, why) -- relative to each temp root.
REAP_RULES: list[tuple[str, float, str]] = [
    ("**/*.pt", 2.0, "model checkpoints; regenerable, and the reason /tmp filled"),
    ("**/*.pth", 2.0, "model checkpoints"),
    ("**/*.safetensors", 2.0, "model weights"),
    ("**/*.npy", 6.0, "embedding dumps"),
    ("**/*.pkl", 6.0, "pickled intermediates"),
    ("**/subagents/**/*.jsonl", 1.0, "agent transcripts; consumed once reported"),
    ("**/tasks/*.output", 1.0, "task stdout captures"),
    ("**/uv-cache-*/**", 12.0, "per-run uv caches"),
]


def is_durable(path: Path) -> bool:
    """True if the path sits under a results root that must never be reaped."""
    resolved = str(path.resolve())
    return any(resolved.startswith(root) for root in DURABLE_ROOTS)


def reap(roots: list[Path], apply: bool, min_free_gb: float) -> int:
    """Delete regenerable artifacts older than their rule's age gate."""
    now = time.time()
    total_bytes = 0
    total_files = 0

    for root in roots:
        if not root.is_dir():
            continue
        usage = shutil.disk_usage(root)
        free_gb = usage.free / 1e9
        print(f"  {root}  free={free_gb:.1f}GB used={100 * usage.used / usage.total:.0f}%")
        if free_gb > min_free_gb:
            print(f"    above --min-free-gb {min_free_gb}; nothing to do")
            continue

        for pattern, min_age_h, why in REAP_RULES:
            for path in root.glob(pattern):
                if not path.is_file() or is_durable(path):
                    continue
                age_h = (now - path.stat().st_mtime) / 3600
                if age_h < min_age_h:
                    continue
                size = path.stat().st_size
                total_bytes += size
                total_files += 1
                if apply:
                    try:
                        path.unlink()
                    except OSError as exc:
                        print(f"    could not remove {path}: {exc}")
                        continue
                if size > 10_000_000:
                    verb = "removed" if apply else "would remove"
                    print(
                        f"    {verb} {size / 1e6:>7.1f}MB  {path.name}  ({why}, {age_h:.1f}h old)"
                    )

    verb = "freed" if apply else "would free"
    print(f"\n  {verb} {total_bytes / 1e9:.2f}GB across {total_files} file(s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    # S108 is suppressed deliberately: reaping temp filesystems is this tool's entire
    # purpose, so a hardcoded /tmp default is the intent rather than an oversight. It
    # never writes there, only measures and unlinks, and refuses durable roots.
    ap.add_argument(
        "--roots",
        default="/tmp/claude-1000,/tmp",  # noqa: S108
        help="comma-separated temp roots",
    )
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=5.0,
        help="only reap when free space is below this; 0 forces a sweep",
    )
    ap.add_argument("--apply", action="store_true", help="delete; otherwise dry-run")
    args = ap.parse_args()

    roots = [Path(r.strip()) for r in args.roots.split(",") if r.strip()]
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"csd-tmp-reaper [{mode}]  min_free={args.min_free_gb}GB")
    return reap(roots, args.apply, args.min_free_gb)


if __name__ == "__main__":
    sys.exit(main())
