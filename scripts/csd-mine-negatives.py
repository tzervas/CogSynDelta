#!/usr/bin/env python3
"""Mine BM25 hard negatives for a region, once, offline, into a guarded manifest.

WHY THIS IS A SEPARATE PASS
The T2 arm of PREREG-RETRIEVAL-NEGATIVES-2026-09-06 (rev 3) needs `m = 8` mined negatives
per anchor over a 782,959-pair training union. Mining is hours of CPU (section 7: FiQA's
57,638-passage index is 2.7 s for 500 queries, and this is ~1,570x more queries over
pools several times larger), it is deterministic, and it must be identical across the
arm's two seeds. So it runs ONCE, writes a manifest, and the training runs verify that
manifest (G39) instead of re-mining -- which also makes the mined negative set an
artefact somebody can inspect rather than a side effect of a training job.

WHAT IT WRITES
`csd-mined-negatives/v1`: per-source pool sha256s, the training union's corpus
fingerprint, the sha256 of the qrels artefact the false-negative audit reads, the pinned
BM25 parameters, the sha256 of the training pair sequence the negatives are indexed
against, the mined pool indices themselves, and the audit(s). `regions/_mining.py` holds
the definitions and both guards; this file is only the entry point.

WHAT IT DOES NOT DO
Run the experiment. Mining a manifest is pre-round work (section 6), and the round runs
later, from reviewed and merged code.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cogsyndelta.eval import beir_fiqa
from cogsyndelta.regions import _mining
from cogsyndelta.regions.memory import memory_config
from cogsyndelta.regions.pretrain import (
    _corpus_content_fingerprint,
    build_splits,
    load_sources,
)


def main(argv: list[str] | None = None) -> int:
    """Mine the region's training union and write the manifest.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 when the manifest was written, 1 when the FiQA false-negative audit breached
        its ceiling AND the pre-registered query_id exclusion did not bring it back under
        -- the manifest is still written, because the round is not abandoned on a breach
        (section 4.1), but the exit code says a human has to read the audit.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", default="memory", choices=["memory"])
    parser.add_argument(
        "--steps",
        type=int,
        default=4000,
        help="Steps the RUN will take. It sets the corpus budget, so it must match the "
        "training run's own value or the pinned union differs and G39 refuses.",
    )
    parser.add_argument("--batch-size", type=int, default=1280)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--holdout-pairs", type=int, default=512)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--m", type=int, default=_mining.NEGATIVES_PER_ANCHOR)
    parser.add_argument(
        "--fiqa-source",
        default="primary",
        help="Source name holding the FiQA pairs -- 'primary' for memory_config().",
    )
    parser.add_argument("--eval-root", default=None, help="FiQA dataset root override.")
    parser.add_argument("--out", required=True, help="Manifest path to write.")
    args = parser.parse_args(argv)

    cfg = memory_config(
        steps=args.steps,
        batch_size=args.batch_size,
        max_len=args.max_len,
        holdout_pairs=args.holdout_pairs,
        split_seed=args.split_seed,
        order_seed=args.order_seed,
        token_loss_weight=0.0,
        decorr_weight=0.0,
    )
    started = time.time()
    _, train_pairs, split_meta = build_splits(cfg)
    sources = _mining.restrict_to_union(load_sources(cfg, cfg.split_seed), train_pairs)
    root = Path(args.eval_root) if args.eval_root else None
    qrels = beir_fiqa.load_split("train", root)
    qrels_path = beir_fiqa.resolve_paths(root).split("train")

    result = _mining.mine_and_audit(
        train_pairs=train_pairs,
        sources=sources,
        qrels=qrels,
        fiqa_source=args.fiqa_source,
        m=args.m,
    )
    manifest = _mining.build_manifest(
        region=cfg.region,
        corpus_fingerprint=split_meta.get("corpus_fingerprint") or _corpus_content_fingerprint(cfg),
        result=result,
        train_pairs=train_pairs,
        qrels_path=qrels_path,
        m=args.m,
    )
    written = _mining.write_manifest(args.out, manifest)
    final = result.audits[-1]
    print(
        json.dumps(
            {
                "manifest": str(written),
                "sha256": manifest["sha256"],
                "train_pairs": len(train_pairs),
                "pools": {name: len(pool.texts) for name, pool in result.pools.items()},
                "audits": [audit.as_dict() for audit in result.audits],
                "elapsed_s": round(time.time() - started, 1),
            },
            indent=2,
        )
    )
    return 1 if final.breached else 0


if __name__ == "__main__":
    raise SystemExit(main())
