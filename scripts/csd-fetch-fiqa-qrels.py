#!/usr/bin/env python3
"""Fetch FiQA's relevance judgements and build the `retrieve` region's training pairs.

WHY THIS EXISTS
`BeIR/fiqa` ships only `corpus` and `queries`. The relevance judgements -- the qrels that
say which passage answers which question -- live in a *separate* repo, `BeIR/fiqa-qrels`.
Without them the corpus on disk is 57,638 passages and 6,648 questions with nothing
connecting the two, which means zero positive pairs and nothing for a contrastive
objective to train on. The directory looked complete and was not.

So this does two things and refuses to pretend either happened:

1. Downloads `BeIR/fiqa-qrels` (cc-by-sa-4.0, same licence as `BeIR/fiqa`).
2. Joins qrels -> (query text, passage text) and writes one parquet per split, which is
   the shape `cogsyndelta.regions.pretrain.load_pairs` reads.

WHERE IT RUNS
On homelab (192.168.1.170), the corpus host. `/data/datasets` there is exported read-only
to the GPU hosts as `/mnt/fleet-datasets`, so derived training data has to be written on
homelab or it is not visible to anything that trains.

WHAT IT REFUSES TO DO
Write a manifest entry it did not measure. Counts and byte totals are taken by walking the
directory *after* the write, and a path that produced zero files is recorded as FAILED.
The previous dataset effort left a directory holding a single metadata.json describing a
download that never happened; it read as present right up until training needed it.

Usage (from an operator shell that can reach the vault):

    secret exec HF_TOKEN=gpu/huggingface-token -- \\
        bash -c 'printf "%s" "$HF_TOKEN" | ssh kang@192.168.1.170 \\
            "HF_TOKEN=\\$(cat) /data/datasets/csd/.venv/bin/python /tmp/fetch_fiqa_qrels.py"'

The token arrives on stdin, never in argv, so it cannot leak into a process listing. It is
optional: both repos are public and the download succeeds without it.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("CSD_DATA_ROOT", "/data/datasets/csd"))
RETRIEVE = ROOT / "region" / "retrieve"

QRELS_REPO = "BeIR/fiqa-qrels"
QRELS_WHY = (
    "The relevance judgements BeIR/fiqa omits. Without them the FiQA corpus and queries "
    "have nothing linking them and the retrieve region has no positive pairs at all. "
    "cc-by-sa-4.0, same licence as BeIR/fiqa, so it is train-eligible."
)
PAIRS_WHY = (
    "Derived: qrels joined to query and passage text, one parquet per BeIR split. This is "
    "the (anchor, positive) shape regions.pretrain.load_pairs reads. Train on `train`; "
    "`dev` and `test` are the held-out splits and are never trained on."
)

SPLITS = ("train", "dev", "test")


def dir_stats(path: Path) -> tuple[int, int]:
    """Return (file_count, total_bytes) measured on disk, not declared."""
    if not path.exists():
        return 0, 0
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def download_qrels(dest: Path, token: str | None) -> tuple[str, str]:
    """Snapshot BeIR/fiqa-qrels into ``dest``. Returns (status, error)."""
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(
            repo_id=QRELS_REPO,
            repo_type="dataset",
            local_dir=str(dest),
            token=token,
            max_workers=4,
        )
    except Exception as exc:  # one failure must not abort the manifest
        return "FAILED", f"{type(exc).__name__}: {exc}"[:300]

    missing = [s for s in SPLITS if not (dest / f"{s}.tsv").is_file()]
    if missing:
        return "FAILED", f"download completed but these splits are absent: {missing}"
    return "ok", ""


def read_beir_texts(parquet_dir: Path) -> dict[str, str]:
    """Read a BeIR corpus/queries parquet directory into ``{_id: text}``.

    Title and text are concatenated when a title exists. FiQA passages are forum answers
    and almost all have an empty title, but SciFact-shaped sets do not, and silently
    dropping the title would quietly discard the most informative line of a document.
    """
    import pyarrow.parquet as pq

    shards = sorted(parquet_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards under {parquet_dir}")

    out: dict[str, str] = {}
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=4096, columns=["_id", "title", "text"]):
            rows = batch.to_pydict()
            for _id, title, text in zip(rows["_id"], rows["title"], rows["text"]):
                title = (title or "").strip()
                text = (text or "").strip()
                out[_id] = f"{title}\n{text}".strip() if title else text
    return out


def read_qrels(path: Path) -> list[tuple[str, str, int]]:
    """Read a BeIR qrels TSV (``query-id, corpus-id, score``) keeping only score >= 1."""
    rows: list[tuple[str, str, int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            score = int(row["score"])
            if score >= 1:
                rows.append((row["query-id"], row["corpus-id"], score))
    return rows


def build_pairs(qrels_dir: Path, fiqa_dir: Path, out_dir: Path) -> dict[str, int]:
    """Join qrels to text and write one parquet per split. Returns rows written per split.

    Every qrel row is kept, including the second and third positive for a query. Collapsing
    to one positive per query is a *training* decision (in-batch negatives dislike repeated
    anchors) and belongs to the trainer, not to the data on disk -- an evaluator that wants
    all relevant passages cannot recover them once they are thrown away here.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    queries = read_beir_texts(fiqa_dir / "queries")
    corpus = read_beir_texts(fiqa_dir / "corpus")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, int] = {}
    for split in SPLITS:
        qrels = read_qrels(qrels_dir / f"{split}.tsv")
        cols: dict[str, list] = {
            "query_id": [],
            "doc_id": [],
            "score": [],
            "query": [],
            "passage": [],
        }
        dropped = 0
        for qid, did, score in qrels:
            q, p = queries.get(qid), corpus.get(did)
            # A qrel naming an id that is not in the corpus is a broken pair, not a pair
            # with an empty side. Encoding "" as a positive is a free win for the loss.
            if not q or not p:
                dropped += 1
                continue
            cols["query_id"].append(qid)
            cols["doc_id"].append(did)
            cols["score"].append(score)
            cols["query"].append(q)
            cols["passage"].append(p)

        table = pa.table(cols)
        pq.write_table(table, out_dir / f"{split}.parquet")
        written[split] = table.num_rows
        print(
            f"    {split}: {table.num_rows} pairs from {len(qrels)} qrels "
            f"({len(set(cols['query_id']))} unique queries, {dropped} unresolvable)",
            flush=True,
        )
    return written


def remeasure(entries: list[dict]) -> list[dict]:
    """Re-walk every manifest entry's path so no count in the file is stale."""
    for entry in entries:
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
    """Download qrels, build the pair splits, and rewrite the manifest from disk."""
    fiqa_dir = RETRIEVE / "fiqa"
    qrels_dir = RETRIEVE / "fiqa-qrels"
    pairs_dir = RETRIEVE / "fiqa-pairs"

    if not (fiqa_dir / "corpus").is_dir():
        print(f"FiQA corpus absent at {fiqa_dir}; fetch BeIR/fiqa first", file=sys.stderr)
        return 2

    print(f"=== {QRELS_REPO} -> {qrels_dir}", flush=True)
    t0 = time.time()
    status, error = download_qrels(qrels_dir, os.environ.get("HF_TOKEN") or None)
    n_files, n_bytes = dir_stats(qrels_dir)
    if status == "ok" and n_files == 0:
        status, error = "FAILED", "download reported success but directory is empty"
    print(f"    {status}  {n_files} files  {n_bytes / 1e6:.2f} MB  {time.time() - t0:.0f}s")
    if status != "ok":
        print(f"    {error}", file=sys.stderr)
        return 1

    print(f"=== join -> {pairs_dir}", flush=True)
    written = build_pairs(qrels_dir, fiqa_dir, pairs_dir)
    if not written.get("train"):
        print("    FAILED: train split produced zero pairs", file=sys.stderr)
        return 1

    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    entries = remeasure(manifest.get("datasets", []))

    qrels_files, qrels_bytes = dir_stats(qrels_dir)
    pairs_files, pairs_bytes = dir_stats(pairs_dir)
    entries = upsert(
        entries,
        {
            "group": "region/retrieve",
            "name": "fiqa-qrels",
            "repo_id": QRELS_REPO,
            "why": QRELS_WHY,
            "status": "ok",
            "error": "",
            "files": qrels_files,
            "bytes": qrels_bytes,
            "path": str(qrels_dir),
        },
    )
    entries = upsert(
        entries,
        {
            "group": "region/retrieve",
            "name": "fiqa-pairs",
            "repo_id": f"derived:{QRELS_REPO}+BeIR/fiqa",
            "why": PAIRS_WHY,
            "status": "ok",
            "error": "",
            "files": pairs_files,
            "bytes": pairs_bytes,
            "path": str(pairs_dir),
            "rows": written,
        },
    )

    manifest.update(
        {
            "schema": "csd-datasets/v1",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": (
                "Counts are MEASURED on disk after download, not declared. A dataset "
                "listed here with files>0 is genuinely present."
            ),
            "total_files": sum(e["files"] for e in entries),
            "total_bytes": sum(e["bytes"] for e in entries),
            "datasets": entries,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    failed = [e for e in entries if e["status"] != "ok"]
    print(f"\n  manifest: {manifest_path}  ({len(entries)} datasets, {len(failed)} failed)")
    for entry in failed:
        print(f"    FAILED {entry['name']}: {entry['error'][:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
