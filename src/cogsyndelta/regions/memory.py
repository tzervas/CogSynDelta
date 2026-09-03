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
  training. `pretrain_region`'s own `held_out` (a 512-pair diagonal eval) covers the
  in-mixture number; `run_memory_pretrain` below layers `cogsyndelta.eval.beir_fiqa`'s
  full-57,638-passage-pool BEIR ranking on top -- the SAME upgrade `regions/retrieve.py`
  used to apply over the diagonal eval, and for the identical reason (recall@10 out of
  512 and recall@10 out of 57,638 are different measurements sharing a name) -- adding a
  `retrieval`/`gates` block to this region's receipt the same way
  `regions/retrieve.py`'s `run_retrieve_pretrain` used to layer its own `retrieval`
  block over `pretrain_region`'s. `gates` carries the five pre-registered W4 conditions
  from `cogsyndelta.eval.beir_fiqa.w4_gates` (DEC-09).

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

import torch
from tokenizers import Tokenizer

from cogsyndelta.eval import beir_fiqa
from cogsyndelta.regions._checkpoint import load_checkpoint
from cogsyndelta.regions.pretrain import PretrainConfig, pretrain_region
from cogsyndelta.regions.pretrain import _tokenize as tokenize_batch
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig

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

MEASURED_VRAM_AT_BATCH_512 = {
    "measured": "2026-09-03, 3090 Ti (24 GiB), real fiqa/all-nli/natural-questions/gooaq "
    "corpus, 50 steps",
    "config": "batch_size=512, max_len=96, dim=256, depth=4, n_heads=4 (16,021,248 "
    "params), bf16 autocast, token_loss_weight=0.1, decorr_weight=0.1",
    "mean_step_time_ms": 146.0,
    "peak_allocated_mib": 10276.6,
    "peak_reserved_mib": 11678.0,
    "peak_whole_card_mib": 13079,  # nvidia-smi memory.used, polled at 0.5s during the run
    "note": (
        "53.7% of the 3090 Ti's 24,564 MiB at the whole-card figure -- comfortable "
        "headroom for the production 8,000-step run at this batch, matching "
        "DEFAULT_BATCH's own measured-table convention in scripts/csd-train-all.py. "
        "VRAM/timing only -- see EVIDENCE_50_STEP_CONTROL_ARM below for what the "
        "final-block rank actually shows at this step count; the rank claim this note "
        "used to make here ('moved the ratio to 5.74x pooled, well past the 2.0x gate, "
        "the first real-data evidence the terms are not a no-op') was measured on a "
        "harness that did not match W1's pre-committed one (both sides of the held-out "
        "pairs through a subsampling participation ratio, not W1's single-side, "
        "full-surface pr_effective_rank -- see 'fix(memory): align W4's final-block "
        "rank measurement with W1's pre-committed harness') and had no control arm, so "
        "it could not tell 'the terms worked' from 'this statistic doesn't discriminate "
        "at 50 steps' apart. It does not, per EVIDENCE_50_STEP_CONTROL_ARM."
    ),
}
"""Row W4's own smoke-run receipt, condensed -- a SHORT run only (50 steps, no full
training run), matching the constraint this worktree operates under. Measured against the
real fleet corpus with `memory_config()`'s own defaults -- `TOKEN_LOSS_WEIGHT`/
`DECORR_WEIGHT` on, not a hand-tuned arm -- so the numbers describe what an operator
actually launching row W4 gets, not a best case.

VERIFIED by direct measurement, not inferred from `scripts/csd-train-all.py`'s own
`DEFAULT_BATCH` table (measured on `code`/`retrieve` with the token-aware terms OFF,
since those regions never turn them on): the MLM head and the decorrelation loss add
real compute and memory no prior region ever paid, so extrapolating from that table
would have been a guess wearing a measurement's clothes."""

EVIDENCE_50_STEP_CONTROL_ARM = {
    "measured": "2026-09-03, 3090 Ti (24 GiB), real fiqa/all-nli/natural-questions/gooaq "
    "corpus, 50 steps, batch_size=512, max_len=96, dim=256/depth=4/n_heads=4 "
    "(16,021,248 params), seed=0 -- IDENTICAL config across all three arms, only the "
    "two weights below vary",
    "harness": "cogsyndelta.eval.benchmark.pr_effective_rank on the FINAL block's "
    "token-global surface, ONE side of the held-out pairs, full surface, no "
    "subsampling -- the same measurement W1's pre-committed harness "
    "(docs/design/evidence/w1-token-rank-2026-09-02/measure_w1.py) used, not the "
    "mismatched one MEASURED_VRAM_AT_BATCH_512's original note reported against. "
    "Full evidence: docs/design/evidence/w4-control-arm-2026-09-03/",
    "code_revision": "5554fcb5efc3e8453fe09e0cf75b862f5c13519b",
    "arms": {
        "control": {
            "token_loss_weight": 0.0,
            "decorr_weight": 0.0,
            "pooled_pr_rank": 1.9995955920659791,
            "token_global_pr_rank": 4.037415218462928,
            "ratio": 2.0191158824727538,
        },
        "token_only": {
            "token_loss_weight": TOKEN_LOSS_WEIGHT,
            "decorr_weight": 0.0,
            "pooled_pr_rank": 1.9994038259022515,
            "token_global_pr_rank": 4.047204865224132,
            "ratio": 2.0242058221519055,
        },
        "both_on": {
            "token_loss_weight": TOKEN_LOSS_WEIGHT,
            "decorr_weight": DECORR_WEIGHT,
            "pooled_pr_rank": 2.1809807874236706,
            "token_global_pr_rank": 4.887977255410147,
            "ratio": 2.241183087717234,
        },
    },
    "gate_e_clause_1_discriminates_at_50_steps": False,
    "note": (
        "The control arm (both weights OFF) already clears §4.0's `token_global_pr_rank "
        ">= 2.0 * pooled_pr_rank` threshold on its own, at 2.0191x -- 50 steps is short "
        "enough that `pooled_pr_rank` sits barely above the 1.0 floor (InfoNCE has not "
        "yet had the steps to differentiate items) while `token_global_pr_rank` starts "
        "measurably higher for an unrelated reason (a token-position surface has more "
        "inherent local variation than a pooled vector even untrained). Gate (e) clause "
        "1's ratio-vs-2.0x threshold is therefore NOT DISCRIMINATING at this step count, "
        "on this harness -- a `passed: True` here does not distinguish 'the token-aware "
        "terms worked' from 'the terms were never turned on'. The threshold itself is "
        "unchanged (a design-doc matter, not this evidence's to alter); whether it "
        "discriminates at the production 8,000-step scale is not measured here. The "
        "terms' real, attributable effect reads off the RAW token_global_pr_rank "
        "instead: control->both_on moves it by +0.8506, of which +0.0098 (1.2%) is "
        "attributable to token_loss_weight alone (control->token_only, decorr_weight=0.0 "
        "throughout that step) and +0.8408 (98.8%) to decorr_weight "
        "(token_only->both_on). L_decorr accounts for essentially all of the measured "
        "movement at these weights, at this step count, on this corpus -- L_token is "
        "not provably a no-op (see the mutation-verified gradient tests in "
        "tests/test_token_aware_objective.py), but its measured effect on this "
        "statistic here is small next to L_decorr's."
    ),
}
"""Row W4's control-arm evidence (COMMIT 3, reviewer finding B2): the SAME 50-step smoke
run MEASURED_VRAM_AT_BATCH_512 above reports VRAM/timing for, run THREE ways -- both
weights off, token_loss_weight alone, and memory_config()'s own on-by-default weights --
so a reader can tell what the token-aware terms actually changed apart from what a
50-step run of ANYTHING would already show. See
docs/design/evidence/w4-control-arm-2026-09-03/README.md for the full write-up,
per-arm summary JSONs, the reproduction command, and checkpoint sha256s."""

P1_GATE = {"stsb_spearman": 0.40, "emb_std": 0.01}
"""The SAME basic sanity floor `regions/compress.py`'s `P1_GATE` uses -- "did the
consolidation head learn anything at all", independent of the much stricter
parent-comparison gate (§4.0's W4 row, condition (1): beat compress's OWN measured
0.7588) that `cogsyndelta.eval.beir_fiqa`'s `gates` block evaluates separately."""


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
    §4.0's rank clause, ...) lives in `cogsyndelta.eval.beir_fiqa`'s `gates` block, which
    also covers the retrieval head; this function covers only the consolidation half, and
    only the "trained at all" question.

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


def embedding_table_divergence(
    checkpoint_a: str, checkpoint_b: str, device: str | torch.device = "cpu"
) -> dict[str, Any]:
    """DEC-24 (§6.2): "the divergence between the two parents' tables is measured and
    recorded in W4's receipt." Mean cosine similarity and mean L2 distance, row for row,
    between two regions' token embedding tables over the SAME vocabulary -- every text
    region in this project shares one GPT-2 BPE tokenizer, so row `i` names the same
    token in both tables and a row-wise comparison is meaningful without alignment.

    Loaded through `load_checkpoint` (DEC-40/W0c's one sanctioned `torch.load` entry
    point), not a direct call.

    Args:
        checkpoint_a: A `TextEncoder` checkpoint (typically `retrieve`'s).
        checkpoint_b: Another (typically `compress`'s).
        device: Where to run the comparison.

    Returns:
        `{"a", "b", "mean_cosine_similarity", "mean_l2_distance", "n_tokens"}`.

    Raises:
        ValueError: The two tables' shapes disagree -- they cannot have been trained
            under the same tokenizer/width and a row-wise comparison would be meaningless.
    """
    dev = torch.device(device)
    state_a = load_checkpoint(checkpoint_a, map_location=dev)
    state_b = load_checkpoint(checkpoint_b, map_location=dev)
    table_a = state_a["model"]["embed.weight"]
    table_b = state_b["model"]["embed.weight"]
    if tuple(table_a.shape) != tuple(table_b.shape):
        raise ValueError(
            f"cannot compare embedding tables of different shape: {checkpoint_a} is "
            f"{tuple(table_a.shape)}, {checkpoint_b} is {tuple(table_b.shape)}"
        )
    a = torch.nn.functional.normalize(table_a.float(), dim=-1)
    b = torch.nn.functional.normalize(table_b.float(), dim=-1)
    cosine = (a * b).sum(dim=-1)
    l2 = (table_a.float() - table_b.float()).norm(dim=-1)
    return {
        "a": checkpoint_a,
        "b": checkpoint_b,
        "mean_cosine_similarity": cosine.mean().item(),
        "mean_l2_distance": l2.mean().item(),
        "n_tokens": table_a.size(0),
    }


def run_memory_pretrain(
    *,
    steps: int = 2000,
    batch_size: int = 256,
    lr: float = 3e-4,
    warmup_steps: int = 200,
    max_len: int = 96,
    seed: int = 0,
    device: str = "auto",
    holdout_pairs: int = 512,
    eval_every: int = 100,
    checkpoint_every: int = 500,
    token_loss_weight: float = TOKEN_LOSS_WEIGHT,
    decorr_weight: float = DECORR_WEIGHT,
    token_loss_chunk: int = 2048,
    encoder: TextEncoderConfig | None = None,
    tokenizer_path: str = "/mnt/fleet-datasets/tritter/gpt2_tokenizer.json",
    out_dir: str = "receipts",
    eval_split: str = "dev",
    eval_root: Path | None = None,
    retrieve_checkpoint: str | None = None,
    compress_checkpoint: str | None = None,
    skip_lexical: bool = False,
) -> dict[str, Any]:
    """Train `memory` (both heads' training corpus, §4.0's terms on) and layer the
    retrieval head's real gate on top: `cogsyndelta.eval.beir_fiqa`'s full-57,638-passage
    BEIR ranking, a BM25 reference from the same pool/qrels, and the five pre-registered
    W4 conditions -- the same two-stage shape `regions/retrieve.py`'s
    `run_retrieve_pretrain` used to apply to `retrieve` alone, extended to a `gates` block
    (DEC-09).

    Args:
        retrieve_checkpoint: DEC-24 -- if given, `memory`'s trunk inherits this
            checkpoint's token embedding table instead of a random init (forwarded as
            `PretrainConfig.init_embedding_from`).
        compress_checkpoint: If ALSO given (together with `retrieve_checkpoint`), the
            divergence between the two parents' tables is measured
            (`embedding_table_divergence`) and recorded, even though only `retrieve`'s
            table is the one actually inherited.
        token_loss_chunk: Forwarded to `PretrainConfig.token_loss_chunk` (see that
            field's own docstring in `regions/pretrain.py` for the chunk-and-checkpoint
            mechanism this controls). Defaults to that field's own default (2048) so
            existing callers are unaffected; a production launch at a batch size where
            the default overshoots the card (see
            `docs/design/evidence/w4-masked-token-loss-2026-09-03/README.md`) passes a
            smaller value explicitly, memory-only trade, no change to loss or gradients.
        eval_root: Dataset root for the BEIR eval (FiQA pool/qrels); defaults to
            `cogsyndelta.eval.beir_fiqa.DEFAULT_FIQA_ROOT`. Independent of `MEMORY_ROOT`
            (which resolves TRAINING sources) so a test can point them at different
            fixture trees.
        skip_lexical: Skip the BM25 reference (only to save CPU on a rerun, matching
            `regions/retrieve.py`'s own `--skip-lexical`). When set, `gates` is written
            with `passed: None` and a note -- gate (c) needs BM25 and cannot be silently
            treated as passed.

    Returns:
        The pretrain receipt extended with `retrieval` and (unless `skip_lexical`)
        `gates` blocks, re-written to `receipt["receipt_path"]`.
    """
    overrides: dict[str, Any] = {}
    if retrieve_checkpoint is not None:
        overrides["init_embedding_from"] = retrieve_checkpoint

    cfg = memory_config(
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        warmup_steps=warmup_steps,
        max_len=max_len,
        seed=seed,
        device=device,
        holdout_pairs=holdout_pairs,
        eval_every=eval_every,
        checkpoint_every=checkpoint_every,
        token_loss_weight=token_loss_weight,
        decorr_weight=decorr_weight,
        token_loss_chunk=token_loss_chunk,
        encoder=encoder or TextEncoderConfig(dim=256, depth=4, n_heads=4, max_len=max_len),
        tokenizer_path=tokenizer_path,
        out_dir=out_dir,
        **overrides,
    )
    receipt = pretrain_region(cfg)

    if retrieve_checkpoint is not None and compress_checkpoint is not None:
        divergence = embedding_table_divergence(
            retrieve_checkpoint, compress_checkpoint, device=receipt["device"]
        )
        receipt["shared_embedding_table"] = {
            **receipt["shared_embedding_table"],
            "divergence_from_compress": divergence,
        }

    checkpoint = Path(receipt.get("checkpoint", ""))
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"pretrain_region did not leave a final checkpoint at {checkpoint}; without "
            f"the trained weights the retrieval head's full-pool ranking cannot be measured"
        )
    device_t = torch.device(receipt["device"])
    tok = Tokenizer.from_file(tokenizer_path)

    state = load_checkpoint(checkpoint, map_location=device_t)
    trained = TextEncoder(TextEncoderConfig(**state["config"]), name="memory").to(device_t)
    trained.load_state_dict(state["model"])

    # `memory`'s own random-init baseline (gate (4)), at the SAME seed the training run's
    # own `untrained_baseline` used -- see `pretrain_region`, which seeds identically.
    torch.manual_seed(seed)
    untrained = TextEncoder(TextEncoderConfig(**state["config"]), name="memory-untrained").to(
        device_t
    )

    def tokenize(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Bind `tok`/`max_len`/`device_t` into the shape `beir_fiqa.encode_texts`
        expects: `(texts) -> (ids, mask)`."""
        return tokenize_batch(tok, texts, max_len, device_t)

    task = beir_fiqa.build_ranking_task(eval_split, pool="corpus", root=eval_root)
    full_pool_trained = beir_fiqa.encoder_rank_metrics(trained, tokenize, task)
    full_pool_untrained = beir_fiqa.encoder_rank_metrics(untrained, tokenize, task)
    del untrained

    full_pool_bm25: dict[str, float] = {}
    if not skip_lexical:
        full_pool_bm25 = beir_fiqa.bm25_metrics(task)

    receipt["retrieval"] = {
        "eval_split": eval_split,
        "full_pool": {
            "task": task.summary(),
            "trained": full_pool_trained,
            "untrained": full_pool_untrained,
            "lexical_bm25": full_pool_bm25,
        },
        "licence": "BeIR/fiqa + BeIR/fiqa-qrels, both cc-by-sa-4.0.",
    }
    receipt["gates"] = (
        beir_fiqa.w4_gates(
            memory_receipt=receipt,
            full_pool_trained=full_pool_trained,
            full_pool_bm25=full_pool_bm25,
            full_pool_untrained=full_pool_untrained,
        )
        if full_pool_bm25
        else {
            "passed": None,
            "note": "skip_lexical=True; gate (c) needs a BM25 reference and was not "
            "computed, so the gates block is not evaluable",
        }
    )
    Path(receipt["receipt_path"]).write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Run memory-region pretraining, the BEIR full-pool eval, and print the five W4
    gates' verdict alongside the consolidation gate.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        0 when every gate passes, 1 when any does not.
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
    parser.add_argument(
        "--token-loss-chunk",
        type=int,
        default=2048,
        help="PretrainConfig.token_loss_chunk -- masked-position vocab projection chunk "
        "size (memory-only trade, see regions/pretrain.py's field docstring). Defaults "
        "to that field's own default; a larger batch size may need this lowered to fit "
        "the card (see docs/design/evidence/w4-masked-token-loss-2026-09-03/README.md).",
    )
    parser.add_argument("--eval-split", default="dev", choices=["dev", "test"])
    parser.add_argument("--retrieve-checkpoint", default=None)
    parser.add_argument("--compress-checkpoint", default=None)
    parser.add_argument("--skip-lexical", action="store_true")
    parser.add_argument("--out-dir", default="receipts")
    args = parser.parse_args(argv)

    receipt = run_memory_pretrain(
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
        token_loss_chunk=args.token_loss_chunk,
        eval_split=args.eval_split,
        retrieve_checkpoint=args.retrieve_checkpoint,
        compress_checkpoint=args.compress_checkpoint,
        skip_lexical=args.skip_lexical,
        out_dir=args.out_dir,
    )
    consolidation = consolidation_gate_report(receipt)
    gates = receipt["gates"]
    print(
        json.dumps(
            {
                "receipt": receipt["receipt_path"],
                "consolidation_gate": consolidation,
                "gates": gates,
            },
            indent=2,
        )
    )
    return 0 if (consolidation["passed"] and gates.get("passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
