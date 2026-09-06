#!/usr/bin/env python3
"""Print row W0's per-region parameter table (embedding / token path / pooling head).

Re-instantiates the five regions that have a real matrix checkpoint
(`code`/`compress`/`retrieve`/`reason`/`visual`, under `/akula-data/csd/matrix/` by
default) and prints the split `cogsyndelta.faculty.param_table` computes for each.
Read-only, CPU-only (sets `CUDA_VISIBLE_DEVICES=""` before importing torch, matching
`csd-lexical-baseline.py`'s convention, so a live GPU job on another lane is never
touched); a checkpoint missing or unreadable on this host is reported as a row saying
so rather than aborting the whole table.

This is NOT `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` §2.3's white-matter
interconnect table (the `27,424,039`-param workspace/controller total) -- that needs
DEC-16's module (workspace, controller), which W0 does not build. See
`cogsyndelta.faculty.param_table`'s module docstring for the full explanation.

Usage:
    python scripts/csd-faculty-param-table.py [--matrix-root PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def main(argv: list[str] | None = None) -> int:
    """Build and print the table; return 1 if any region's checkpoint could not load."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-root",
        default=None,
        help="Override CSD_MATRIX_ROOT (default /akula-data/csd/matrix, read-only).",
    )
    args = parser.parse_args(argv)
    if args.matrix_root:
        os.environ["CSD_MATRIX_ROOT"] = args.matrix_root

    # Imported after CSD_MATRIX_ROOT is set: param_table.py resolves MATRIX_ROOT (and
    # therefore CANONICAL_CHECKPOINTS' paths) at import time.
    from cogsyndelta.faculty.param_table import build_matrix_param_table, format_param_table

    rows = build_matrix_param_table()
    print(format_param_table(rows))

    failed = [r for r in rows if r.counts is None]
    if failed:
        print(
            f"\n{len(failed)} of {len(rows)} region(s) could not be re-instantiated:",
            file=sys.stderr,
        )
        for row in failed:
            print(f"  {row.entry.region}: {row.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
