#!/usr/bin/env python3
"""Move cold CSD corpora off the homelab SSD onto gpu5080's spinning bulk array.

THE TIERS
  hot    homelab:/data      3.0 TB SSD, NFS-exported read-only to the fleet. Everything
                            a running job reads. This is the scarce resource.
  cold   gpu5080:/bulk      5.5 TB RAID0 across two spinning disks. 2% used. NOT exported,
                            so a restore is an rsync, not a mount.
  canon  HuggingFace        the durable off-fleet home, once a dataset has one.

WHY
/data/datasets/tritter/pretrain is 616 GB -- FineWeb-Edu, a code corpus, Wikipedia,
OpenWebMath, TinyStories. The curriculum trains regions first and foundation fourth, so
none of it is read until step 4, while it occupies a fifth of the one disk the whole fleet
shares. Meanwhile /bulk sits at 2%.

MEASURED, NOT ASSUMED
- Compression at rest is NOT worth it. zstd-3 over an already-Snappy parquet shard gained
  13.7% (118.7 -> 102.5 MB). That is 13.7% of a resource that is not scarce, paid for in
  CPU on both ends and in losing the ability to point a parquet reader straight at the
  archive. Transfers use rsync -z so the wire gets the win for free; at rest the data
  stays readable.

SAFETY
- Nothing local is deleted until the remote copy is verified BY CHECKSUM. rsync exits 0
  on a truncated transfer, and a short file has a perfectly plausible size.
- /bulk is RAID0 with NO redundancy -- lose one disk and all 5.5 TB goes. So only data
  that can be re-fetched may live there as its only copy. Anything not explicitly
  classified `cold` below is refused rather than assumed re-fetchable.
- Checkpoints, receipts and metrics are classified `result` and never move. They are small
  and they are the record that makes a run reproducible after its corpus is gone. This is
  not theoretical: tritter/checkpoints is 5 GB of trained weights sitting one directory
  below the 616 GB of corpus this tool is built to move.
- A stub is left where the data was, so a later job fails with a pointer instead of an
  unexplained absence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

HOT_HOST = "kang@192.168.1.170"
COLD_HOST = "tzervas@192.168.1.251"
HOT_ROOT = "/data/datasets"
# The corpus is kang:kang mode 775, so none of this needs root. The first draft used
# `sudo rsync` to a remote host, which fails on "Host key verification failed" -- sudo
# runs as root, and root has no known_hosts. Neither data host could reach the other at
# all, so a dedicated ed25519 key exists for exactly this hop, restricted in
# authorized_keys to `restrict,from="192.168.1.170"`: no pty, no forwarding, one source.
ARCHIVE_KEY = "~/.ssh/id_csd_archive"
RSH = f"ssh -i {ARCHIVE_KEY} -o BatchMode=yes -o ConnectTimeout=10"
COLD_ROOT = "/bulk/csd-archive"

# Explicit classification. Nothing is inferred; an unlisted path is refused.
#   cold    re-fetchable corpus -- safe as its only copy on a redundancy-free array
#   result  weights, receipts, metrics -- never leaves the SSD
#   hot     read by the current curriculum step -- stays until the step is done
POLICY: dict[str, tuple[str, str]] = {
    "tritter/pretrain": ("cold", "public HF pretraining corpora; re-fetchable"),
    "tritter/checkpoints": ("result", "trained weights from earlier tritter runs"),
    "tritter/processed": ("result", "derived artefacts, provenance not re-establishable"),
    "csd/region": ("hot", "steps 1-3 read this now"),
    "csd/vl": ("hot", "VL region corpus, step 3"),
    "csd/manifest.json": ("result", "fetch provenance for the CSD corpus"),
}

NEVER_MOVE_GLOBS = ("receipts", "checkpoints", "manifest.json", "*.log")


def ssh(host: str, cmd: str, timeout: int = 600) -> tuple[int, str]:
    p = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,  # the caller inspects rc; a failed probe is data, not an exception
    )
    return p.returncode, (p.stdout + p.stderr).strip()


def classify(rel: str) -> tuple[str, str]:
    """Return (class, why) for a path, longest declared prefix wins."""
    best = ("unclassified", "not declared in POLICY; refused rather than guessed")
    best_len = -1
    for prefix, (klass, why) in POLICY.items():
        if (rel == prefix or rel.startswith(prefix + "/")) and len(prefix) > best_len:
            best, best_len = (klass, why), len(prefix)
    return best


def free_gb(host: str, path: str) -> float:
    rc, out = ssh(host, f"df -B1 --output=avail {path} | tail -1")
    return int(out) / 1e9 if rc == 0 and out.strip().isdigit() else -1.0


def survey() -> list[tuple[str, int, str, str]]:
    """(relative path, bytes, class, why) for every declared and discovered subtree."""
    rc, out = ssh(HOT_HOST, f"du -sb {HOT_ROOT}/*/* 2>/dev/null | sort -rn")
    rows = []
    for line in out.splitlines() if rc == 0 else []:
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        rel = parts[1].replace(HOT_ROOT + "/", "")
        klass, why = classify(rel)
        rows.append((rel, int(parts[0]), klass, why))
    return rows


def offload(rel: str, apply: bool) -> dict:
    """Copy one subtree to cold storage, verify it, then reclaim the hot copy."""
    src, dst = f"{HOT_ROOT}/{rel}", f"{COLD_ROOT}/{rel}"
    res: dict[str, object] = {"path": rel}

    klass, why = classify(rel)
    res["class"] = f"{klass} — {why}"
    if klass != "cold":
        res["status"] = "REFUSED"
        res["reason"] = {
            "result": "results never leave the SSD",
            "hot": "the current curriculum step reads this",
        }.get(klass, "undeclared paths are refused; add it to POLICY deliberately")
        return res

    excl = " ".join(f"--exclude={g}" for g in NEVER_MOVE_GLOBS)
    if not apply:
        rc, out = ssh(HOT_HOST, f"du -sh {src} 2>/dev/null | cut -f1")
        res["status"] = f"would offload {out}"
        return res

    ssh(COLD_HOST, f"mkdir -p {dst}")
    # -z buys ~14% on the wire for free; the spinning target is the slow end regardless.
    # --partial so an interrupted 616 GB transfer resumes instead of restarting.
    rc, out = ssh(
        HOT_HOST,
        f'rsync -a --partial --compress --info=stats2 -e "{RSH}" '
        f"{excl} {src}/ {COLD_HOST}:{dst}/ 2>&1 | tail -4",
        timeout=86400,
    )
    if rc != 0:
        res["status"], res["detail"] = "TRANSFER FAILED", out[:300]
        return res

    # Verify by manifest checksum. rsync's exit code does not prove the bytes landed.
    man = "find . -type f -printf '%s %p\\n' | sort | md5sum | cut -d' ' -f1"
    rc_a, a = ssh(HOT_HOST, f'cd {src} && sh -c "{man}"', timeout=3600)
    rc_b, b = ssh(COLD_HOST, f'cd {dst} && sh -c "{man}"', timeout=3600)
    res["hot_manifest"], res["cold_manifest"] = a[:12], b[:12]
    if rc_a != 0 or rc_b != 0 or not a or a != b:
        res["status"] = "VERIFY FAILED — hot copy kept, nothing deleted"
        return res

    stub = json.dumps(
        {
            "offloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "moved_to": f"{COLD_HOST}:{dst}",
            "manifest_md5": a,
            "why_safe": why,
            "restore": f"scripts/csd-storage-tier.py --restore {rel} --apply",
        },
        indent=2,
    )
    ssh(HOT_HOST, f"rm -rf {src} && mkdir -p {src}")
    ssh(HOT_HOST, f"printf '%s' {json.dumps(stub)} > {src}/OFFLOADED.json")
    res["status"] = "offloaded and verified"
    return res


def restore(rel: str, apply: bool) -> dict:
    """Pull a subtree back to the hot tier. /bulk is not exported, so this is a copy."""
    src, dst = f"{COLD_ROOT}/{rel}", f"{HOT_ROOT}/{rel}"
    res: dict[str, object] = {"path": rel}
    rc, _ = ssh(COLD_HOST, f"test -d {src}")
    if rc != 0:
        res["status"] = f"not in cold storage at {COLD_HOST}:{src}"
        return res
    rc, size = ssh(COLD_HOST, f"du -sh {src} | cut -f1")
    res["size"] = size
    if not apply:
        res["status"] = "would restore"
        return res
    rc, out = ssh(
        HOT_HOST,
        f'mkdir -p {dst} && rsync -a --partial --compress -e "{RSH}" '
        f"{COLD_HOST}:{src}/ {dst}/ 2>&1 | tail -3",
        timeout=86400,
    )
    res["status"] = "restored" if rc == 0 else f"FAILED — {out[:200]}"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offload", default="", help="comma-separated subtrees to move cold")
    ap.add_argument("--restore", default="", help="comma-separated subtrees to bring back")
    ap.add_argument(
        "--auto",
        action="store_true",
        help="offload every 'cold' subtree, but only under hot-tier pressure",
    )
    ap.add_argument(
        "--min-free-gb",
        type=float,
        default=800.0,
        help="--auto only acts when the hot tier has less than this free",
    )
    ap.add_argument("--apply", action="store_true", help="execute; otherwise dry-run")
    args = ap.parse_args()

    hot_free, cold_free = free_gb(HOT_HOST, "/data"), free_gb(COLD_HOST, "/bulk")
    print(f"csd-storage-tier [{'APPLY' if args.apply else 'DRY-RUN'}]")
    print(f"  hot   {HOT_HOST}:/data   {hot_free:>8.0f} GB free")
    print(f"  cold  {COLD_HOST}:/bulk  {cold_free:>8.0f} GB free  (RAID0, no redundancy)")

    targets = [t.strip() for t in args.offload.split(",") if t.strip()]

    if not targets and not args.restore:
        print("\n  hot-tier subtrees:")
        reclaimable, hidden, hidden_bytes = 0, 0, 0
        for rel, size, klass, why in survey():
            mark = {"cold": "OFFLOAD", "result": "keep", "hot": "in use"}.get(klass, "REFUSE")
            if klass == "cold":
                reclaimable += size
            elif size < 100_000_000:
                hidden, hidden_bytes = hidden + 1, hidden_bytes + size
                continue
            print(f"    {size / 1e9:>8.1f} GB  {mark:<8} {rel:<28} {why}")
        if hidden:
            print(f"    {hidden_bytes / 1e9:>8.1f} GB  ...       {hidden} entries under 100 MB")
        print(f"\n  reclaimable from the SSD: {reclaimable / 1e9:.1f} GB")
        if args.auto:
            if hot_free >= args.min_free_gb:
                print(f"  --auto: {hot_free:.0f} GB free is above {args.min_free_gb:.0f}; idle")
                return 0
            targets = [r for r, _, k, _ in survey() if k == "cold"]
            print(f"  --auto: under pressure, offloading {len(targets)} subtree(s)")
        else:
            print("\n  --offload <path> to move, --restore <path> to bring back, --apply to run")
            return 0

    for rel in targets:
        print(f"\n  === offload {rel}")
        for k, v in offload(rel, args.apply).items():
            print(f"    {k}: {v}")
    for rel in [t.strip() for t in args.restore.split(",") if t.strip()]:
        print(f"\n  === restore {rel}")
        for k, v in restore(rel, args.apply).items():
            print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
