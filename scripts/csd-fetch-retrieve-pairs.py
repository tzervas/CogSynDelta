#!/usr/bin/env python3
"""Close the `retrieve` region's data gap: ready-made (query, positive) pairs.

WHY THIS EXISTS
Measured recall tracks data volume almost directly across regions:

    code      214,813 pairs -> held-out recall@1 0.94
    compress  277,269 pairs -> held-out recall@1 0.26
    retrieve    4,986 pairs -> held-out recall@1 0.006   <- STARVED

`csd-fetch-fiqa-qrels.py` already fixed FiQA's missing qrels and lifted `retrieve` from
4,986 to 14,131 train pairs -- real progress, but still ~15-30x short of the other two
regions. This script closes the rest of the gap with datasets that ship as ready
`(query, positive)` text pairs and need no separate qrels join at all:

    sentence-transformers/natural-questions  (config: pair)   100,231 rows
    sentence-transformers/gooaq              (config: pair) 3,012,496 rows

LICENSING -- checked against the HF API and upstream sources before fetching, not assumed
    natural-questions: derived from Google's Natural Questions, CC BY-SA 3.0 (confirmed via
        google-research-datasets/natural_questions, tag `license:cc-by-sa-3.0`). Same family
        as BeIR/fiqa's CC BY-SA 4.0. Train-eligible.
    gooaq: derived from github.com/allenai/gooaq, which ships an explicit Apache-2.0
        LICENSE file in the repo root. Train-eligible. Caveat worth naming: the *code* that
        collected the data is Apache-2.0; the underlying question/answer text originates
        from Google "People also ask" boxes, i.e. third-party web content. AI2 has
        redistributed it under that license for years without incident, so this script
        treats it as train-eligible, but that provenance is different in kind from a
        dataset an author wrote from scratch and is worth an operator's eyes if this ever
        matters commercially.

TWO CANDIDATES CHECKED AND DELIBERATELY NOT FETCHED
    MS MARCO (BeIR/msmarco, sentence-transformers/msmarco, microsoft/ms_marco): the
        *official* MS MARCO terms (microsoft.github.io/msmarco, and every mirror's lineage
        traces back to them) state the dataset is "intended for non-commercial research
        purposes only". That is exactly the NC/ToS restriction CSD-BRAIN-REGIONS.md already
        uses to keep SciFact and NFCorpus out of the training tree. Same rule applies here:
        NC-restricted stays out of `region/`, full stop. If eval-only use is ever wanted,
        that is a separate, explicit decision -- this script does not make it.
    sentence-transformers/eli5: HF has moved the *source* `eli5` dataset to its
        `defunct-datasets` namespace (a 307 redirect, not a 404) with `license: unknown` in
        its own card. "Defunct" plus "unknown license" is exactly the "unclear -> don't
        assume it's fine" case this task calls out. Not fetched.

Both skip decisions are also recorded as `status: "skipped"` manifest entries (files=0,
bytes=0, `why` holding the reasoning above) so the manifest documents the negative result,
not just the positive ones.

WHERE IT RUNS
On homelab (192.168.1.170), the corpus host. `/data/datasets` there is exported read-only
to the GPU hosts as `/mnt/fleet-datasets`, so derived training data has to be written on
homelab or it is invisible to anything that trains.

WHAT IT REFUSES TO DO
Write a manifest entry it did not measure. Counts and byte totals are taken by walking the
directory *after* the write, and a path that produced zero files is recorded as FAILED, not
silently dropped. This script also re-walks and re-measures every *existing* manifest entry
before writing, the same discipline `csd-fetch-fiqa-qrels.py` used, so the whole file
reflects live disk state and not accumulated claims.

Usage (from an operator shell that can reach the vault):

    secret exec HF_TOKEN=gpu/huggingface-token -- \\
        bash -c 'printf "%s" "$HF_TOKEN" | ssh kang@192.168.1.170 \\
            "HF_TOKEN=\\$(cat) /data/datasets/csd/.venv/bin/python /tmp/csd-fetch-retrieve-pairs.py"'

The token arrives on stdin, never in argv, so it cannot leak into a process listing. It is
optional: both repos are public and the download succeeds without it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("CSD_DATA_ROOT", "/data/datasets/csd"))
RETRIEVE = ROOT / "region" / "retrieve"

# (manifest name, repo_id, columns to verify as (anchor, positive), why/license note)
FETCH_TARGETS = [
    (
        "natural-questions",
        "sentence-transformers/natural-questions",
        ("query", "answer"),
        (
            "Ready (query, answer) pairs, no qrels needed. CC BY-SA 3.0 (confirmed via "
            "google-research-datasets/natural_questions, tag license:cc-by-sa-3.0 -- same "
            "family as BeIR/fiqa's CC BY-SA 4.0). Closes part of the retrieve-region gap "
            "left after fiqa-pairs (14,131 train pairs, still ~15-30x short of code/"
            "compress)."
        ),
    ),
    (
        "gooaq",
        "sentence-transformers/gooaq",
        ("question", "answer"),
        (
            "Ready (question, answer) pairs, no qrels needed. Apache-2.0 per the upstream "
            "github.com/allenai/gooaq LICENSE file. Underlying Q&A text originates from "
            "Google 'People also ask' boxes (third-party web content); AI2 has "
            "redistributed it under Apache-2.0 for years. Large enough (3.01M rows) to put "
            "retrieve on par with, or ahead of, code (214,813) and compress (277,269)."
        ),
    ),
]

SKIPPED_TARGETS = [
    (
        "msmarco",
        "BeIR/msmarco or sentence-transformers/msmarco",
        (
            "NOT FETCHED. Official MS MARCO terms (microsoft.github.io/msmarco; every HF "
            "mirror's lineage traces back to the same license) state the dataset is "
            "'intended for non-commercial research purposes only'. That is the same NC/ToS "
            "restriction CSD-BRAIN-REGIONS.md already uses to keep SciFact and NFCorpus "
            "eval-only, out of region/. Same rule applied here. If eval-only use is wanted "
            "later that is a separate explicit decision, not a default."
        ),
    ),
    (
        "eli5",
        "sentence-transformers/eli5 (source: eli5, now defunct-datasets/eli5)",
        (
            "NOT FETCHED. The source `eli5` dataset now 307-redirects to HF's "
            "`defunct-datasets/eli5` and its own card lists `license: unknown`. "
            "'Defunct' + 'unknown license' is exactly the unclear-license case this task "
            "says not to assume is fine."
        ),
    ),
]


def dir_stats(path: Path) -> tuple[int, int]:
    """Return (file_count, total_bytes) measured on disk, not declared."""
    if not path.exists():
        return 0, 0
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def download_repo(repo_id: str, dest: Path, token: str | None) -> tuple[str, str]:
    """Snapshot a dataset repo into ``dest``. Returns (status, error)."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest),
            token=token,
            max_workers=4,
        )
    except Exception as exc:  # one failure must not abort the manifest
        return "FAILED", f"{type(exc).__name__}: {exc}"[:300]
    return "ok", ""


def count_usable_pairs(parquet_dir: Path, columns: tuple[str, str]) -> tuple[int, int]:
    """Walk every parquet shard under ``parquet_dir`` and count non-empty (a, b) rows.

    Mirrors the exact filter `cogsyndelta.regions.pretrain.load_pairs` applies -- both
    sides present and non-blank after stripping -- so the number reported here is the
    number a training run would actually see, not the raw row count.

    Returns (usable, total_rows).
    """
    import pyarrow.parquet as pq

    left, right = columns
    shards = sorted(parquet_dir.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {parquet_dir}")

    usable = 0
    total = 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=4096, columns=list(columns)):
            a_col = batch.column(left).to_pylist()
            b_col = batch.column(right).to_pylist()
            for a, b in zip(a_col, b_col):
                total += 1
                if a and str(a).strip() and b and str(b).strip():
                    usable += 1
    return usable, total


def remeasure(entries: list[dict]) -> list[dict]:
    """Re-walk every manifest entry's path so no count in the file is stale.

    Entries with status "skipped" (files=0 by design) are left alone -- only entries that
    claim "ok" get held to the "files>0 or it's FAILED" rule.
    """
    for entry in entries:
        if entry.get("status") == "skipped":
            continue
        n_files, n_bytes = dir_stats(Path(entry["path"]))
        entry["files"], entry["bytes"] = n_files, n_bytes
        if n_files == 0 and entry.get("status") == "ok":
            entry["status"] = "FAILED"
            entry["error"] = "manifest claimed ok but the directory is empty on disk"
    return entries


def upsert(entries: list[dict], new: dict) -> list[dict]:
    """Replace the entry with the same path, else append."""
    kept = [e for e in entries if e["path"] != new["path"]]
    kept.append(new)
    return sorted(kept, key=lambda e: (e["group"], e["name"]))


def main() -> int:
    """Fetch the ready-made pair sets, verify usability, record two deliberate skips."""
    token = os.environ.get("HF_TOKEN") or None
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    entries = remeasure(manifest.get("datasets", []))

    overall_ok = True
    for name, repo_id, columns, why in FETCH_TARGETS:
        dest = RETRIEVE / name
        print(f"=== {repo_id} -> {dest}", flush=True)
        t0 = time.time()
        status, error = download_repo(repo_id, dest, token)
        n_files, n_bytes = dir_stats(dest)
        if status == "ok" and n_files == 0:
            status, error = "FAILED", "download reported success but directory is empty"
        print(f"    {status}  {n_files} files  {n_bytes / 1e6:.2f} MB  {time.time() - t0:.0f}s")

        usable = total = 0
        if status == "ok":
            try:
                usable, total = count_usable_pairs(dest, columns)
                print(f"    usable pairs ({columns[0]}, {columns[1]}): {usable}/{total}")
                if usable == 0:
                    status, error = "FAILED", "downloaded but zero usable (non-empty) pairs"
            except Exception as exc:  # verification failure must also fail loudly
                status = "FAILED"
                error = f"verify: {type(exc).__name__}: {exc}"[:300]
                print(f"    {error}", file=sys.stderr)

        if status != "ok":
            overall_ok = False
            print(f"    {error}", file=sys.stderr)

        entries = upsert(
            entries,
            {
                "group": "region/retrieve",
                "name": name,
                "repo_id": repo_id,
                "why": why,
                "status": status,
                "error": error,
                "files": n_files,
                "bytes": n_bytes,
                "path": str(dest),
                "columns": list(columns),
                "usable_pairs": usable,
                "total_rows": total,
            },
        )

    for name, repo_id, why in SKIPPED_TARGETS:
        entries = upsert(
            entries,
            {
                "group": "region/retrieve",
                "name": name,
                "repo_id": repo_id,
                "why": why,
                "status": "skipped",
                "error": "",
                "files": 0,
                "bytes": 0,
                "path": str(RETRIEVE / name),
            },
        )

    manifest.update(
        {
            "schema": "csd-datasets/v1",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": (
                "Counts are MEASURED on disk after download, not declared. A dataset "
                "listed here with files>0 is genuinely present. status=skipped means "
                "files=0 by design (a licensing decision), not a failed download."
            ),
            "total_files": sum(e["files"] for e in entries),
            "total_bytes": sum(e["bytes"] for e in entries),
            "datasets": entries,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    failed = [e for e in entries if e["status"] == "FAILED"]
    skipped = [e for e in entries if e["status"] == "skipped"]
    print(
        f"\n  manifest: {manifest_path}  ({len(entries)} datasets, "
        f"{len(failed)} failed, {len(skipped)} deliberately skipped)"
    )
    for entry in failed:
        print(f"    FAILED {entry['name']}: {entry['error'][:120]}")
    for entry in skipped:
        print(f"    SKIPPED {entry['name']}: {entry['why'][:100]}")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
