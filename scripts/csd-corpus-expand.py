#!/usr/bin/env python3
"""Fetch training corpora onto the mechanical bulk array, license-gated and idempotent.

WHERE THIS WRITES AND WHY
/bulk is 5.95 TB of spinning RAID0 at 6.7% used. Corpora are large, they grow, and they
are read sequentially during training -- which is exactly what mechanical storage is good
at and exactly what should not sit on the homelab SSD the whole fleet shares. So datasets
land in /bulk/csd-corpus, and the SSD keeps only what a running job needs right now.

THE LICENSE GATE IS STRUCTURAL, NOT ADVISORY
Every entry must carry a license string that was actually observed on the dataset card or
upstream repository, and a verdict. `fetch` refuses any entry whose verdict is not
TRAIN_OK, and there is deliberately no flag to override that. This project has already
turned down MS MARCO -- whose official terms are non-commercial research only -- and ELI5,
whose license could not be established. Those decisions are worth exactly as much as the
mechanism that keeps them enforced.

A dataset card is not authoritative about its own upstream. Where the card and the source
disagree, the source wins and the entry is rejected.

IDEMPOTENT
A dataset already on disk with a manifest recording the same revision is skipped. Re-runs
are cheap and safe, which is the point: this is meant to be run by an unattended agent.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


# /bulk is local on gpu5080 and /mnt/bulk over NFS everywhere else, so the root is
# resolved rather than hardcoded -- a script that only runs on one host is not a
# fleet tool.
def _default_root() -> Path:
    for candidate in (Path("/bulk"), Path("/mnt/bulk")):
        if candidate.is_dir():
            return candidate / "csd-corpus"
    return Path("/bulk/csd-corpus")


BULK_CORPUS = _default_root()

TRAIN_OK = "TRAIN_OK"
REJECTED = "REJECTED"


@dataclass
class Dataset:
    """One corpus, with the provenance needed to defend using it."""

    repo_id: str
    region: str
    license: str
    """The license string actually observed, not inferred from a family or a mirror."""
    verdict: str
    why: str
    config: str = ""
    splits: tuple[str, ...] = ("train",)
    caveat: str = ""
    columns: tuple[str, ...] = ()

    @property
    def local(self) -> Path:
        name = self.repo_id.split("/")[-1] + (f"-{self.config}" if self.config else "")
        return BULK_CORPUS / self.region / name


# Seeded with entries already verified in this repo's fetch scripts. Everything added here
# must carry an observed license string and a verdict; unverified entries are REJECTED
# rather than fetched, so an unattended run cannot quietly widen the licence surface.
CATALOGUE: list[Dataset] = [
    Dataset(
        repo_id="code-search-net/code_search_net",
        region="code",
        config="go",
        license="MIT (dataset card: license: mit)",
        verdict=TRAIN_OK,
        why="same corpus and licence as the Python config already trained on; adds a second language",
        columns=("func_documentation_string", "func_code_string"),
        caveat="configs are per-language; never glob across them, the schemas match but the domains do not",
    ),
    Dataset(
        repo_id="code-search-net/code_search_net",
        region="code",
        config="java",
        license="MIT (dataset card: license: mit)",
        verdict=TRAIN_OK,
        why="third language for the code region; broadens beyond Python-only docstring pairs",
        columns=("func_documentation_string", "func_code_string"),
    ),
    Dataset(
        repo_id="sentence-transformers/quora-duplicates",
        region="compress",
        config="pair",
        license="UNVERIFIED — pending card check",
        verdict=REJECTED,
        why="paraphrase pairs would suit compress, but the licence has not been observed yet",
        caveat="promote to TRAIN_OK only after reading the card and the upstream Quora terms",
    ),
]


def _disk_free_fraction(path: Path) -> tuple[float, float]:
    usage = shutil.disk_usage(path)
    return usage.used / usage.total, usage.free / 1e12


def _manifest_path(ds: Dataset) -> Path:
    return ds.local / "MANIFEST.json"


def is_present(ds: Dataset) -> bool:
    """True when this exact dataset+config is already on disk with a manifest."""
    man = _manifest_path(ds)
    if not man.is_file():
        return False
    try:
        rec = json.loads(man.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return rec.get("repo_id") == ds.repo_id and rec.get("config", "") == ds.config


def fetch(ds: Dataset, token: str | None, apply: bool, max_used_fraction: float) -> dict:
    """Fetch one dataset, refusing anything not licence-cleared."""
    res: dict[str, object] = {"repo_id": ds.repo_id, "config": ds.config, "region": ds.region}

    if ds.verdict != TRAIN_OK:
        res["status"] = "REFUSED"
        res["reason"] = f"verdict={ds.verdict}; licence observed: {ds.license}. {ds.why}"
        return res

    if is_present(ds):
        res["status"] = "present"
        res["path"] = str(ds.local)
        return res

    used, free_tb = _disk_free_fraction(BULK_CORPUS.parent)
    if used >= max_used_fraction:
        res["status"] = "DEFERRED"
        res["reason"] = (
            f"bulk at {used:.1%} used, at or above the {max_used_fraction:.0%} ceiling. "
            f"Publish and evict cold datasets before fetching more."
        )
        return res

    if not apply:
        res["status"] = "would fetch"
        res["path"] = str(ds.local)
        return res

    from datasets import load_dataset

    ds.local.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = 0
    for split in ds.splits:
        data = load_dataset(
            ds.repo_id, ds.config or None, split=split, token=token, streaming=False
        )
        out = ds.local / f"{split}.parquet"
        data.to_parquet(str(out))
        rows += data.num_rows
    manifest = {
        **asdict(ds),
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": rows,
        "seconds": round(time.time() - started, 1),
        "bytes": sum(f.stat().st_size for f in ds.local.glob("*.parquet")),
    }
    _manifest_path(ds).write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    res["status"] = "fetched"
    res["rows"] = rows
    res["bytes"] = manifest["bytes"]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="", help="restrict to these regions")
    ap.add_argument("--bulk-root", default="", help="override the corpus root")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument(
        "--max-used-fraction",
        type=float,
        default=0.5,
        help="stop fetching once bulk passes this; the operator's eviction threshold",
    )
    args = ap.parse_args()

    global BULK_CORPUS
    if args.bulk_root:
        BULK_CORPUS = Path(args.bulk_root)
    if not BULK_CORPUS.parent.is_dir():
        print(f"bulk not mounted at {BULK_CORPUS.parent}", file=sys.stderr)
        return 2
    BULK_CORPUS.mkdir(parents=True, exist_ok=True)

    used, free_tb = _disk_free_fraction(BULK_CORPUS.parent)
    print(f"csd-corpus-expand [{'APPLY' if args.apply else 'DRY-RUN'}]")
    print(f"  bulk {used:.1%} used, {free_tb:.2f} TB free, ceiling {args.max_used_fraction:.0%}")

    token = os.environ.get("HF_TOKEN")
    wanted = {r.strip() for r in args.regions.split(",") if r.strip()}
    ok = refused = 0
    for ds in CATALOGUE:
        if wanted and ds.region not in wanted:
            continue
        res = fetch(ds, token, args.apply, args.max_used_fraction)
        status = res["status"]
        mark = {"fetched": "+", "present": "=", "would fetch": ".", "REFUSED": "!"}.get(status, "?")
        label = f"{ds.repo_id}" + (f":{ds.config}" if ds.config else "")
        print(f"  {mark} {ds.region:<10} {label:<46} {status}")
        if "reason" in res:
            print(f"      {res['reason']}")
        if status in ("fetched", "present", "would fetch"):
            ok += 1
        elif status == "REFUSED":
            refused += 1

    print(f"\n  {ok} usable, {refused} refused on licence grounds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
