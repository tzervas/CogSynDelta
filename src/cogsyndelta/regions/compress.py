"""The ``compress`` region: neighbours stay neighbours in a short latent.

CSD-BRAIN-REGIONS.md states the job in one line -- "Neighbors stay neighbors in a short
latent" -- and the router trigger as "sentence pair / embedding to reconstruct". The P1
gate is ``stsb_spearman > 0.40`` and ``emb_std > 0.01``. This module is the wiring that
makes that gate a measurement rather than an intention.

WHY ALLNLI ENTAILMENT PAIRS FOR TRAINING
InfoNCE needs a POSITIVE: two texts that genuinely belong together. Entailment is the
only NLI label that supplies one. Neutral and contradiction pairs are about the same
subject and are therefore excellent negatives and terrible positives -- and the in-batch
negatives InfoNCE already draws are strictly easier, so training on entailment alone is
the honest version of this objective.

The ``pair`` config on disk is already exactly this: the AllNLI authors filtered
``pair-class`` to label 0 and dropped the label column, leaving 314,315 anchor/positive
rows. Reading that directly beats re-deriving it from the 942,069-row ``pair-class``
file, and removes a filter that could silently be written the wrong way round.

WHY STS-B IS HELD OUT ENTIRELY, TRAIN SPLIT INCLUDED
The task left this open. Measured before deciding:

1. **STS-B train leaks into STS-B validation.** 7 of its pairs appear verbatim in the
   validation split, and 8.6% of validation SENTENCES appear in the train split. Training
   on it would contaminate the exact number the gate reads.
2. **STS-B is graded, and InfoNCE cannot express a grade.** 40% of its train pairs score
   below 0.5; the median is 0.60. Feeding those in as positives teaches the encoder that
   sentences humans called dissimilar belong together -- training directly against the
   thing being measured. Thresholding to score >= 0.8 avoids that but leaves 1,406 pairs
   against AllNLI's 313,904, so the upside is rounding error next to the contamination
   cost.

Keeping STS-B wholly unseen also makes the reported Spearman a clean ZERO-SHOT transfer
number: NLI in, semantic similarity out, no in-domain fitting. That is a stronger claim
about the region than a fine-tuned score, and it is the claim the region's one-line job
description actually makes.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from cogsyndelta.regions.pretrain import PretrainConfig, pretrain_region

COMPRESS_ROOT = Path(os.environ.get("CSD_COMPRESS_ROOT", "/mnt/fleet-datasets/csd/region/compress"))
"""Read-only NFS export. Overridable so a local copy or a test fixture can stand in."""

TRAIN_SHARD = "all-nli/pair/train-00000-of-00001.parquet"
GRADED_SHARD = "stsb/data/validation-00000-of-00001.parquet"

P1_GATE = {"stsb_spearman": 0.40, "emb_std": 0.01}
"""From program/csd-program.json, phase P1, region `compress`. Duplicated here as
numbers rather than read from the JSON on purpose: the gate must be checkable by the run
itself without the run being able to rewrite what it is checked against."""


def compress_config(**overrides: Any) -> PretrainConfig:
    """Build the compress region's pretrain config.

    Args:
        **overrides: Any :class:`PretrainConfig` field, e.g. ``steps`` or ``device``.

    Returns:
        A config pointing at AllNLI entailment pairs for training and STS-B validation
        for the graded held-out metric.

    Raises:
        FileNotFoundError: If either shard is missing, rather than training on nothing.
    """
    train_shard = COMPRESS_ROOT / TRAIN_SHARD
    graded_shard = COMPRESS_ROOT / GRADED_SHARD
    for shard in (train_shard, graded_shard):
        if not shard.is_file():
            raise FileNotFoundError(
                f"compress corpus shard missing: {shard}. It is an NFS export; check the "
                f"mount, or point CSD_COMPRESS_ROOT at a local copy."
            )

    cfg = PretrainConfig(
        region="compress",
        pair_columns=("anchor", "positive"),
        shards=[str(train_shard)],
        graded_shards=[str(graded_shard)],
        graded_columns=("sentence1", "sentence2", "score"),
        graded_name="stsb-validation",
    )
    return replace(cfg, **overrides)


def gate_report(receipt: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the P1 gate against a receipt, and say what it was compared with.

    ``emb_std`` is read from the graded evaluation rather than the retrieval one so both
    gate conditions are measured on the same set of embeddings. A collapse detected on
    AllNLI holdout while the Spearman was computed on STS-B would be two statements about
    two different things.

    Args:
        receipt: The dict returned by :func:`pretrain_region`.

    Returns:
        Per-condition pass/fail, the untrained baseline for each, and an overall verdict.

    Raises:
        KeyError: If the receipt carries no graded evaluation, which means the run was
            not configured to measure this gate at all.
    """
    if "graded_held_out" not in receipt:
        raise KeyError(
            "receipt has no graded_held_out; this run measured no Spearman, so the "
            "compress gate cannot be evaluated against it"
        )
    trained = receipt["graded_held_out"]
    untrained = receipt["untrained_graded_baseline"]

    conditions = {
        "stsb_spearman": {
            "threshold": P1_GATE["stsb_spearman"],
            "untrained": untrained["spearman"],
            "trained": trained["spearman"],
            "passed": trained["spearman"] > P1_GATE["stsb_spearman"],
        },
        "emb_std": {
            "threshold": P1_GATE["emb_std"],
            "untrained": untrained["emb_std"],
            "trained": trained["emb_std"],
            "passed": trained["emb_std"] > P1_GATE["emb_std"],
        },
    }
    passed = all(c["passed"] for c in conditions.values())
    beats_untrained = trained["spearman"] > untrained["spearman"]
    return {
        "gate": "P1/compress",
        "conditions": conditions,
        "passed": passed,
        "beats_untrained": beats_untrained,
        # A gate pass that does not beat the random-init model is a pass the corpus
        # handed out, not one the training earned. Both are reported; neither is hidden
        # behind the other.
        "verdict": (
            "PASS"
            if passed and beats_untrained
            else "PASS (but does not beat untrained)"
            if passed
            else "FAIL"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Run compress-region pretraining and print the gate verdict.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 when the P1 gate passes, 1 when it does not. A failing gate is a non-zero exit
        so a run cannot be scripted past without noticing.
    """
    parser = argparse.ArgumentParser(description="Pretrain the compress region (P1).")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--holdout-pairs", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="receipts")
    args = parser.parse_args(argv)

    receipt = pretrain_region(
        compress_config(
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            max_len=args.max_len,
            device=args.device,
            eval_every=args.eval_every,
            holdout_pairs=args.holdout_pairs,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed,
            out_dir=args.out_dir,
        )
    )
    report = gate_report(receipt)
    print(json.dumps({"receipt": receipt["receipt_path"], "gate": report}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
