#!/usr/bin/env python3
"""Generate committed seed-0 split and batch-order manifests for the text regions.

Membership is the historical seed-0 draw: reservoir + shuffle use split_seed=0, which
is what `cfg.seed=0` used to drive. Existing seed-0 cells stay comparable; seed-1
cells are retired from comparison (E0).

Run with CUDA_VISIBLE_DEVICES="" -- this is a CPU corpus pass, not training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS = ROOT / "config" / "mind" / "splits"
# Unique REGIONS keys plus the `language` alias of `code`, matching the six text
# pair-encoder spellings the trainer will look up.
TEXT_REGIONS = ("code", "language", "compress", "retrieve", "reason", "memory")


def _load_train_all():
    path = ROOT / "scripts" / "csd-train-all.py"
    spec = importlib.util.spec_from_file_location("csd_train_all_splits", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg_for_region(mod, name: str, steps: int, batch: int):
    from cogsyndelta.regions.pretrain import PretrainConfig

    entry = mod.region_spec(name)
    resolved = [(mod._shards(g, entry.root), tuple(c), cap) for g, c, cap in entry.sources]
    missing = [
        g
        for (shards, _c, _cap), (g, _, _) in zip(resolved, entry.sources, strict=True)
        if not shards
    ]
    if missing:
        raise FileNotFoundError(f"region {name!r}: no shards for {missing} under {entry.root}")
    primary_shards, pair_cols, _cap = resolved[0]
    extra = [
        {"shards": shards, "columns": list(cols), "limit": cap}
        for shards, cols, cap in resolved[1:]
    ]
    return PretrainConfig(
        region=name,
        pair_columns=pair_cols,
        shards=primary_shards,
        extra_sources=extra,
        steps=steps,
        batch_size=batch,
        holdout_pairs=512,
        split_seed=0,
        order_seed=0,
        seed=0,
        max_len=entry.default_max_len,
    )


def main(argv: list[str] | None = None) -> int:
    """Write seed-0 manifests for each text region whose corpus is on disk.

    Args:
        argv: CLI args; defaults to sys.argv[1:].

    Returns:
        0 on success, 2 if a region is missing shards.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS))
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--regions", default=",".join(TEXT_REGIONS))
    args = parser.parse_args(argv)

    from cogsyndelta.regions.pretrain import _draw_split, write_split_and_order_manifests

    mod = _load_train_all()
    splits_dir = Path(args.splits_dir)
    report: list[dict] = []
    for name in [r.strip() for r in args.regions.split(",") if r.strip()]:
        cfg = _cfg_for_region(mod, name, args.steps, args.batch)
        # Historical seed-0 draw IS this function with membership_seed=0.
        holdout_legacy, train_legacy, meta_legacy = _draw_split(cfg, membership_seed=0)
        split_path, order_path, meta = write_split_and_order_manifests(cfg, splits_dir=splits_dir)
        holdout_now, train_now, _ = _draw_split(cfg, membership_seed=cfg.split_seed)
        equal = holdout_now == holdout_legacy and train_now == train_legacy
        row = {
            "region": name,
            "corpus_fingerprint": meta["corpus_fingerprint"],
            "split_path": str(split_path.relative_to(ROOT))
            if split_path.is_relative_to(ROOT)
            else str(split_path),
            "order_path": str(order_path.relative_to(ROOT))
            if order_path.is_relative_to(ROOT)
            else str(order_path),
            "split_sha256": meta["split"]["sha256"],
            "holdout_pairs": len(holdout_now),
            "train_pairs": len(train_now),
            "duplicates_removed": meta["duplicates_removed"],
            "seed0_equal": equal,
        }
        report.append(row)
        print(
            f"{name}: fp={row['corpus_fingerprint']} sha={row['split_sha256'][:12]}... "
            f"holdout={row['holdout_pairs']} dups={row['duplicates_removed']} "
            f"seed0_equal={equal} -> {split_path.name}",
            flush=True,
        )
    out = splits_dir / "SEED0-REPORT.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    if not all(r["seed0_equal"] for r in report):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
