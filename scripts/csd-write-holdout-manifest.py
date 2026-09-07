#!/usr/bin/env python3
"""Write a reserved-holdout manifest (G41) for one landed corpus split.

WHY A MANIFEST AND NOT "JUST DO NOT LIST THE FILE"
A holdout that is merely intended to be held out is not a holdout. Leaving `test.parquet`
out of a region's shard list is an intention held in a config; one widened glob, one
copied config, one new composite phase and the battery's population is training material,
with nothing to say so. The manifest turns the intention into a checkable fact: every row
becomes an `item_id` (the same ordered sha256 `build_splits` uses for its own holdout), so
`assert_no_reserved_holdout_in_pairs` can refuse a training set BY ROW rather than by
filename.

CPU only. Reads parquet, writes JSON, trains nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cogsyndelta.splits import (
    DEFAULT_SPLITS_DIR,
    build_reserved_holdout_manifest,
    item_id,
    write_json_manifest,
)


def sha256_file(path: Path) -> str:
    """Content digest of a shard.

    Sizes are enough for `fingerprint_corpus` (a corpus fetch does not produce a
    same-name, same-size, different-bytes shard), but a holdout manifest pins ONE file
    and is compared against by hand during review, so it records the real digest.

    Args:
        path: File to hash.

    Returns:
        64-char hex digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pairs(shard: Path, columns: tuple[str, str]) -> list[tuple[str, str]]:
    """Read `(left, right)` pairs from a parquet shard.

    Args:
        shard: Parquet file.
        columns: The two column names, in order.

    Returns:
        One tuple per row, in file order.

    Raises:
        SystemExit: pyarrow is not installed in this interpreter.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - environment, not logic
        raise SystemExit("pyarrow is required to read the shard; install the train group") from None
    table = pq.read_table(str(shard), columns=list(columns))
    left = table.column(columns[0]).to_pylist()
    right = table.column(columns[1]).to_pylist()
    return [(str(a), str(b)) for a, b in zip(left, right, strict=True)]


def main() -> int:
    """Build and write one reserved-holdout manifest.

    Returns:
        Process exit status.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="manifest stem, must end in -holdout")
    ap.add_argument("--shard", required=True, type=Path)
    ap.add_argument("--source", required=True, help="upstream repo id")
    ap.add_argument("--split", required=True, help="upstream split name")
    ap.add_argument("--columns", required=True, help="two column names, comma separated")
    ap.add_argument("--licence", required=True)
    ap.add_argument("--upstream", required=True, help="where the licence was verified")
    ap.add_argument("--notes", required=True, help="why this split is reserved")
    ap.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    args = ap.parse_args()

    if not args.name.endswith("-holdout"):
        print(f"--name must end in -holdout (the guard globs *-holdout.json): {args.name}")
        return 2
    columns = tuple(c.strip() for c in args.columns.split(","))
    if len(columns) != 2:
        print(f"--columns needs exactly two names, got {columns}")
        return 2

    shard: Path = args.shard.resolve()
    pairs = read_pairs(shard, columns)  # type: ignore[arg-type]
    ids = [item_id(a, b) for a, b in pairs]
    if len(set(ids)) != len(ids):
        print(f"warning: {len(ids) - len(set(ids))} duplicate item ids in {shard}")

    manifest = build_reserved_holdout_manifest(
        name=args.name,
        source=args.source,
        split=args.split,
        region_scope="all",
        pair_columns=list(columns),
        shard=str(shard),
        shard_sha256=sha256_file(shard),
        rows=len(pairs),
        item_ids=ids,
        licence=args.licence,
        upstream=args.upstream,
        notes=args.notes,
    )
    out = write_json_manifest(args.splits_dir / f"{args.name}.json", manifest)
    print(json.dumps({"path": str(out), "rows": len(pairs), "sha256": manifest["sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
