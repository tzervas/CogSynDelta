#!/usr/bin/env python3
"""One-shot lexical-baseline backfill for existing seed-0 cells (g49).

Writes a sidecar ``receipts/cogsyndelta-<region>-lexical-<ts>.json`` next to the
training receipt. Does **not** mutate train / eval / quant receipts -- those stay
the bytes they were. Future eval receipts get the same field from
``scripts/csd-benchmark.py``; this script is the backfill for cells already on disk.

CPU only. Sets ``CUDA_VISIBLE_DEVICES=""`` before importing torch so a live GPU-0
job (g48) is not disturbed. Reuses ``csd-benchmark.py``'s ``_region_eval_context``
so the holdout is the manifested split the eval battery scores, and
``cogsyndelta.eval.lexical`` so the numbers agree with the 2026-09-06 diagnosis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(REPO_ROOT / "src")
if _SRC in sys.path:
    sys.path.remove(_SRC)
sys.path.insert(0, _SRC)

from cogsyndelta.eval.lexical import build_lexical_baseline  # noqa: E402
from cogsyndelta.pipeline.receipt import Producer, Receipt  # noqa: E402
from cogsyndelta.regions._checkpoint import sha256_file  # noqa: E402


def _load_benchmark() -> Any:
    """Import ``scripts/csd-benchmark.py`` for ``_region_eval_context`` / receipt glob."""
    path = Path(__file__).resolve().parent / "csd-benchmark.py"
    name = "csd_benchmark_for_lexical"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _train_receipt_from_cell(cell_dir: Path) -> Path:
    """``stages.train.receipt`` from ``cell.json``, or refuse.

    Args:
        cell_dir: A matrix cell directory.

    Returns:
        Path to the training receipt.

    Raises:
        FileNotFoundError: cell.json or the train receipt is missing.
        ValueError: train stage is not done / has no receipt pointer.
    """
    cell_path = cell_dir / "cell.json"
    cell = json.loads(cell_path.read_text())
    train = (cell.get("stages") or {}).get("train") or {}
    receipt = train.get("receipt")
    if train.get("state") != "done" or not receipt:
        raise ValueError(f"{cell_path}: train stage is not done with a receipt pointer")
    path = Path(str(receipt))
    if not path.is_file():
        raise FileNotFoundError(f"train receipt {path} does not exist")
    return path


def backfill_one(train_receipt_path: Path, *, out_dir: Path | None = None) -> Path:
    """Score the manifested holdout and write a lexical sidecar.

    Args:
        train_receipt_path: Existing training receipt (not modified).
        out_dir: Directory for the sidecar; defaults to the training receipt's parent.

    Returns:
        Path of the written sidecar.
    """
    bench = _load_benchmark()
    train_receipt = json.loads(train_receipt_path.read_text())
    region = str(train_receipt.get("region") or train_receipt.get("producer", {}).get("component"))
    if not region:
        raise ValueError(f"{train_receipt_path}: no region / producer.component")
    cfg, _enc, holdout, _tok, _device, split_meta = bench._region_eval_context(
        region, train_receipt
    )
    del cfg
    split = split_meta.get("split") if isinstance(split_meta.get("split"), dict) else {}
    lexical = build_lexical_baseline(holdout, str(split.get("sha256") or ""))
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec = Receipt(
        producer=Producer("cogsyndelta", region, "dense-transformer"),
        stage="eval",
        kind="lexical",
        lexical_baseline=lexical,
        artifacts={
            "source_training_receipt": {
                "path": str(train_receipt_path),
                "sha256": sha256_file(train_receipt_path),
            }
        },
        provenance={
            "holdout_pairs": len(holdout),
            "eval_target": "lexical",
            "split": split,
            "scorer_version": lexical["scorer_version"],
        },
        started_utc=started_utc,
        device="cpu",
    )
    dest = out_dir or train_receipt_path.parent
    path = rec.write(dest)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--train-receipt", type=Path, default=None)
    ap.add_argument(
        "--cell",
        type=Path,
        default=None,
        help="matrix cell dir; reads cell.json stages.train.receipt",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="sidecar directory (default: the training receipt's parent)",
    )
    args = ap.parse_args()
    if (args.train_receipt is None) == (args.cell is None):
        print("pass exactly one of --train-receipt or --cell", file=sys.stderr)
        return 2
    train_path = (
        args.train_receipt
        if args.train_receipt is not None
        else _train_receipt_from_cell(args.cell)
    )
    path = backfill_one(train_path, out_dir=args.out_dir)
    print(f"wrote {path} (training receipt {train_path} untouched)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
