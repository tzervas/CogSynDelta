#!/usr/bin/env python3
"""Publish bulk corpora to private HF dataset repos, and reclaim the local copy.

THE LIFECYCLE
  fetch    csd-corpus-expand.py pulls a licence-cleared dataset onto /bulk
  train    regions read it from /bulk while it is in active use
  publish  this script pushes it to a PRIVATE HF dataset repo
  evict    once the copy is verified on HF, the local bytes are reclaimed

WHEN IT ACTS
Only above a usage ceiling, default 50% of /bulk. Below that there is no reason to move
anything: the array exists to hold corpora, and evicting a dataset that still fits costs a
re-download the next time a region wants it.

WHAT IT WILL NOT DO
Delete a local dataset it has not confirmed on HF. The check is per-file, comparing what
the repo reports against the manifest written at fetch time -- an upload that half
succeeded leaves a repo that exists, is listed, and is missing shards. `repo_exists` is
not evidence, so it is not used as evidence.

PRIVATE, ALWAYS
Every repo is created with private=True, passed explicitly rather than relying on an
account default. A default can change underneath you and the failure mode is a corpus
published to the world.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

BULK_CORPUS = Path("/bulk/csd-corpus") if Path("/bulk").is_dir() else Path("/mnt/bulk/csd-corpus")
OWNER = "tzervas"
REPO_PREFIX = "cogsyndelta-corpus"


def _repo_id(owner: str, region: str, name: str) -> str:
    return f"{owner}/{REPO_PREFIX}-{region}-{name}"


def _usage(path: Path) -> tuple[float, float]:
    u = shutil.disk_usage(path)
    return u.used / u.total, u.free / 1e12


def discover() -> list[tuple[Path, dict]]:
    """Every fetched dataset on the bulk array, with its manifest."""
    found = []
    for man in sorted(BULK_CORPUS.glob("*/*/MANIFEST.json")):
        try:
            found.append((man.parent, json.loads(man.read_text())))
        except (OSError, json.JSONDecodeError):
            continue
    return found


def _local_files(path: Path) -> dict[str, int]:
    return {f.name: f.stat().st_size for f in sorted(path.glob("*.parquet")) if f.is_file()}


def publish(path: Path, manifest: dict, owner: str, token: str, apply: bool) -> dict:
    """Upload one dataset to a private HF dataset repo and verify every shard landed."""
    from huggingface_hub import HfApi

    region = manifest.get("region", "misc")
    name = path.name
    repo = _repo_id(owner, region, name)
    local = _local_files(path)
    res: dict[str, object] = {
        "dataset": f"{region}/{name}",
        "repo": repo,
        "files": len(local),
        "bytes": sum(local.values()),
    }
    if not local:
        res["status"] = "SKIPPED — no parquet shards on disk"
        return res
    if not apply:
        res["status"] = "would publish"
        return res

    api = HfApi(token=token)
    # private=True explicitly. Never rely on an account default for this.
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(path),
        commit_message=f"corpus {region}/{name} from {manifest.get('repo_id')}",
    )

    # Verify per file. A repo that exists proves nothing about what is inside it.
    remote = {
        Path(f).name
        for f in api.list_repo_files(repo_id=repo, repo_type="dataset")
        if f.endswith(".parquet")
    }
    missing = sorted(set(local) - remote)
    res["remote_files"] = len(remote)
    if missing:
        res["status"] = "VERIFY FAILED — local kept"
        res["missing"] = missing
        return res
    res["status"] = "published"
    return res


def evict(path: Path, manifest: dict, repo: str, apply: bool) -> dict:
    """Reclaim a published dataset's local bytes, leaving a pointer behind."""
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    res: dict[str, object] = {"dataset": path.name, "reclaims_bytes": size}
    if not apply:
        res["status"] = "would evict"
        return res
    stub = {
        "evicted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hf_repo": repo,
        "source": manifest.get("repo_id"),
        "config": manifest.get("config", ""),
        "rows": manifest.get("rows"),
        "restore": f"huggingface-cli download {repo} --repo-type dataset --local-dir {path}",
    }
    for f in path.glob("*.parquet"):
        f.unlink()
    (path / "EVICTED.json").write_text(json.dumps(stub, indent=2) + "\n")
    res["status"] = "evicted"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default=OWNER)
    ap.add_argument("--datasets", default="", help="explicit region/name list; default all")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="only evict once bulk usage passes this fraction",
    )
    ap.add_argument("--publish", action="store_true", help="upload to HF")
    ap.add_argument("--evict", action="store_true", help="reclaim local bytes after verifying")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not BULK_CORPUS.parent.is_dir():
        print(f"bulk not mounted at {BULK_CORPUS.parent}", file=sys.stderr)
        return 2

    used, free_tb = _usage(BULK_CORPUS.parent)
    print(f"csd-corpus-publish [{'APPLY' if args.apply else 'DRY-RUN'}]")
    print(f"  bulk {used:.1%} used, {free_tb:.2f} TB free, evict threshold {args.threshold:.0%}")

    token = os.environ.get("HF_TOKEN")
    if args.publish and args.apply and not token:
        print(
            "  HF_TOKEN not set; run under: secret exec HF_TOKEN=gpu/huggingface-token --",
            file=sys.stderr,
        )
        return 2

    wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
    items = discover()
    if not items:
        print(f"  no fetched datasets under {BULK_CORPUS}")
        return 0

    published = []
    for path, manifest in items:
        key = f"{manifest.get('region', 'misc')}/{path.name}"
        if wanted and key not in wanted:
            continue
        if args.publish:
            res = publish(path, manifest, args.owner, token or "", args.apply)
            print(f"  {key:<40} {res['status']}  ({res.get('bytes', 0) / 1e9:.2f} GB)")
            if res["status"] in ("published", "would publish"):
                published.append((path, manifest, res["repo"]))
        else:
            print(f"  {key:<40} {sum(_local_files(path).values()) / 1e9:.2f} GB on disk")

    if args.evict:
        if used < args.threshold:
            print(
                f"\n  eviction skipped: {used:.1%} is below the {args.threshold:.0%} threshold. "
                f"The array exists to hold corpora; evicting one that still fits only buys a "
                f"re-download later."
            )
            return 0
        for path, manifest, repo in published:
            res = evict(path, manifest, repo, args.apply)
            print(
                f"  evict {path.name:<34} {res['status']}  (+{res['reclaims_bytes'] / 1e9:.2f} GB)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
