"""The `memory` region: one hippocampal trunk, two heads -- consolidation and retrieval.

DEC-02 (docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md §1, row W4): merges `compress`
and `retrieve` into one faculty. "One trunk, two heads" is DELIBERATELY not two separate
projection layers -- both parents already share the identical bi-encoder shape
(`TextEncoder` -> masked-mean `pool()` -> symmetric InfoNCE), so the merge is one InfoNCE
run over the UNION of both corpora, read by two EVALUATIONS of the same embedding space:

  CONSOLIDATION HEAD -- `compress`'s objective: AllNLI entailment pairs in training,
  STS-B Spearman as the graded held-out gate (see `regions/compress.py`'s module
  docstring for why entailment-only and why STS-B is held out entirely). Wired straight
  into `pretrain_region` via `graded_shards`, exactly as `compress_config` does.

  RETRIEVAL HEAD -- `retrieve`'s objective: FiQA/NaturalQuestions/GooAQ pairs in
  training. `pretrain_region`'s own `held_out` (a 512-pair diagonal eval) covers this at
  this module's current scope -- the real full-57,638-passage-pool BEIR gate (the SAME
  upgrade `regions/retrieve.py` used to apply over the diagonal eval, and for the
  identical reason: recall@10 out of 512 and recall@10 out of 57,638 are different
  measurements sharing a name) is layered on top separately, once
  `cogsyndelta.eval.beir_fiqa` exists.

WHY FIQA IS THE PRIMARY SOURCE, NOT ALLNLI
DEC-24 (§6.2): `memory` inherits `retrieve`'s token embedding table, "because it is the
parent whose gate (the FiQA BEIR pool) survives as `memory`'s gate, so its tokenisation
statistics are the ones the surviving eval is calibrated against." `PretrainConfig.shards`
(the primary, uncapped source) is FiQA; AllNLI, NaturalQuestions and GooAQ all ride in
through `extra_sources` -- matching `scripts/csd-train-all.py`'s `REGIONS["memory"]`
entry, which is the union of `region_spec("compress")` and `region_spec("retrieve")`'s own
declared sources under the SAME ordering rule. The inheritance mechanism itself is
`regions/pretrain.py`'s `PretrainConfig.init_embedding_from` -- pass a `retrieve`
checkpoint path as `memory_config(init_embedding_from=...)` to use it; `None` (the
default) trains a fresh table, exactly as before DEC-24, with the receipt's own
`shared_embedding_table` block saying which happened.

TOKEN-AWARE, ON BY DEFAULT FOR THIS REGION ONLY
§4.0's `L_token`/`L_decorr` terms default OFF in `PretrainConfig` so every existing region
trains unchanged. `memory` is different: row W4 is "DESIGNED AS THE FIRST TOKEN-AWARE
RETRAIN", so `memory_config()` turns both terms on by default -- the same way
`compress_config()` pins its own architecture choices rather than leaving them to
`PretrainConfig`'s generic defaults.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from cogsyndelta.regions.pretrain import PretrainConfig, pretrain_region

MEMORY_ROOT = Path(os.environ.get("CSD_MEMORY_ROOT", "/mnt/fleet-datasets/csd"))
"""Read-only NFS export root. `memory`'s corpus is the UNION of `compress`'s
(`region/compress/...`) and `retrieve`'s (`region/retrieve/...`) already-declared
sources under the same mount -- see the module docstring -- so this is the shared parent
`CORPUS` in `scripts/csd-train-all.py`, not a new directory of its own. Overridable
so a local copy or a test fixture can stand in, matching `regions/compress.py`'s
`COMPRESS_ROOT` and `regions/retrieve.py`'s `DEFAULT_RETRIEVE_ROOT`."""

RETRIEVAL_FIQA_SHARD = "region/retrieve/fiqa-pairs/train.parquet"
CONSOLIDATION_TRAIN_GLOB = "region/compress/all-nli/pair/train*.parquet"
CONSOLIDATION_GRADED_SHARD = "region/compress/stsb/data/validation-00000-of-00001.parquet"
RETRIEVAL_NQ_GLOB = "region/retrieve/natural-questions/**/train*.parquet"
RETRIEVAL_GOOAQ_GLOB = "region/retrieve/gooaq/**/train*.parquet"
RETRIEVAL_GOOAQ_CAP = 400_000
"""Matches `scripts/csd-train-all.py`'s `REGIONS["retrieve"]` cap -- GooAQ is 96% of
everything `retrieve` can see; training uncapped would produce a GooAQ model wearing a
retrieval region's name, and that reasoning is unchanged by the merge."""

TOKEN_LOSS_WEIGHT = 0.1
"""ζ, §4.0's `L_token` weight -- modest by design; see the module docstring."""

DECORR_WEIGHT = 0.1
"""γ, §4.0's `L_decorr` weight -- modest by design; see the module docstring."""

P1_GATE = {"stsb_spearman": 0.40, "emb_std": 0.01}
"""The SAME basic sanity floor `regions/compress.py`'s `P1_GATE` uses -- "did the
consolidation head learn anything at all", independent of the much stricter
parent-comparison gate (§4.0's W4 row, condition (1): beat compress's OWN measured
0.7588) that `cogsyndelta.eval.beir_fiqa`'s `gates` block will evaluate separately."""


def _shards(pattern: str) -> list[str]:
    return sorted(str(p) for p in MEMORY_ROOT.glob(pattern))


def memory_config(**overrides: Any) -> PretrainConfig:
    """Build the `memory` region's pretrain config: FiQA primary, AllNLI/NQ/GooAQ as
    `extra_sources`, STS-B as the graded (consolidation) gate, §4.0's terms on.

    Args:
        **overrides: Any :class:`PretrainConfig` field, e.g. ``steps`` or ``device``.

    Returns:
        A config whose corpus is the union `compress` + `retrieve` declared, calibrated
        against FiQA's tokenisation (DEC-24) and graded on STS-B (consolidation).

    Raises:
        FileNotFoundError: If the FiQA pairs, the AllNLI training shard, or the STS-B
            graded shard is missing -- the two REQUIRED sources every existing parent
            region already refuses to start without (see `regions/compress.py`'s
            `compress_config`) -- rather than training on nothing or silently dropping a
            head. NaturalQuestions and GooAQ are supplementary (matching
            `region_spec("memory")`'s ordering, they ride in only if present) and are
            omitted rather than refused when their glob resolves nothing, since `retrieve`
            itself is licence-clean and gate-passing on FiQA alone (see
            `regions/retrieve.py`'s own single-source `run_retrieve_pretrain`).
    """
    fiqa = MEMORY_ROOT / RETRIEVAL_FIQA_SHARD
    graded_shard = MEMORY_ROOT / CONSOLIDATION_GRADED_SHARD
    if not fiqa.is_file():
        raise FileNotFoundError(
            f"memory corpus shard missing: {fiqa} (retrieval head, primary source). It "
            f"is an NFS export; check the mount, or point CSD_MEMORY_ROOT at a local copy."
        )
    if not graded_shard.is_file():
        raise FileNotFoundError(
            f"memory corpus shard missing: {graded_shard} (consolidation head's graded "
            f"gate). It is an NFS export; check the mount, or point CSD_MEMORY_ROOT at a "
            f"local copy."
        )
    allnli = _shards(CONSOLIDATION_TRAIN_GLOB)
    if not allnli:
        raise FileNotFoundError(
            f"memory corpus shard missing: {MEMORY_ROOT / CONSOLIDATION_TRAIN_GLOB} "
            f"(consolidation head, training source). It is an NFS export; check the "
            f"mount, or point CSD_MEMORY_ROOT at a local copy."
        )

    extra_sources: list[dict[str, Any]] = [
        {"shards": allnli, "columns": ["anchor", "positive"], "limit": 0}
    ]
    nq = _shards(RETRIEVAL_NQ_GLOB)
    if nq:
        extra_sources.append({"shards": nq, "columns": ["query", "answer"], "limit": 0})
    gooaq = _shards(RETRIEVAL_GOOAQ_GLOB)
    if gooaq:
        extra_sources.append(
            {"shards": gooaq, "columns": ["question", "answer"], "limit": RETRIEVAL_GOOAQ_CAP}
        )

    cfg = PretrainConfig(
        region="memory",
        pair_columns=("query", "passage"),
        shards=[str(fiqa)],
        extra_sources=extra_sources,
        graded_shards=[str(graded_shard)],
        graded_columns=("sentence1", "sentence2", "score"),
        graded_name="stsb-validation",
        token_loss_weight=TOKEN_LOSS_WEIGHT,
        decorr_weight=DECORR_WEIGHT,
    )
    return replace(cfg, **overrides)


def consolidation_gate_report(receipt: dict[str, Any]) -> dict[str, Any]:
    """The basic P1-style sanity gate on the consolidation head: did it learn ANYTHING.

    Identical shape and thresholds to `regions/compress.py`'s `gate_report` -- see that
    function's own docstring for why `emb_std` is read from the graded evaluation and why
    a passing gate that does not beat the untrained model is reported as such rather than
    as a bare PASS. The much stricter W4 row gate (beat `compress`'s own 0.7588,
    §4.0's rank clause, ...) will live in `cogsyndelta.eval.beir_fiqa`'s `gates` block,
    which also covers the retrieval head; this function covers only the consolidation
    half, and only the "trained at all" question.

    Args:
        receipt: The dict returned by :func:`pretrain_region` for a `memory_config()` run.

    Returns:
        Per-condition pass/fail, the untrained baseline for each, and an overall verdict.

    Raises:
        KeyError: If the receipt carries no graded evaluation.
    """
    if "graded_held_out" not in receipt:
        raise KeyError(
            "receipt has no graded_held_out; this run measured no Spearman, so the "
            "memory region's consolidation gate cannot be evaluated against it"
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
        "gate": "consolidation/memory",
        "conditions": conditions,
        "passed": passed,
        "beats_untrained": beats_untrained,
        "verdict": (
            "PASS"
            if passed and beats_untrained
            else "PASS (but does not beat untrained)"
            if passed
            else "FAIL"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Run memory-region pretraining and print the consolidation gate verdict.

    The retrieval head's own gate (the BEIR full-pool eval and §4.0's five pre-registered
    W4 conditions) is added by `cogsyndelta.eval.beir_fiqa` once that module exists; this
    entry point covers what `pretrain_region` alone produces today.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 when the consolidation gate passes, 1 when it does not.
    """
    parser = argparse.ArgumentParser(description="Pretrain the memory region (row W4).")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-len", type=int, default=96)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--holdout-pairs", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--token-loss-weight", type=float, default=TOKEN_LOSS_WEIGHT)
    parser.add_argument("--decorr-weight", type=float, default=DECORR_WEIGHT)
    parser.add_argument("--out-dir", default="receipts")
    args = parser.parse_args(argv)

    receipt = pretrain_region(
        memory_config(
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
            token_loss_weight=args.token_loss_weight,
            decorr_weight=args.decorr_weight,
            out_dir=args.out_dir,
        )
    )
    report = consolidation_gate_report(receipt)
    print(json.dumps({"receipt": receipt["receipt_path"], "consolidation_gate": report}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
