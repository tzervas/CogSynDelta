#!/usr/bin/env python3
"""Benchmark trained regions across ranking, efficiency and representation health.

Emits receipts in the model-agnostic envelope, so the pipeline console renders these
beside training and quantization runs without knowing what a region is.

The representation family is the reason this exists as a separate pass rather than more
numbers bolted onto training. A retrieval score is a relative ordering; it stays high in an
embedding space that has collapsed into a narrow cone, because rotating or crushing every
vector together leaves the ordering intact. Anisotropy, alignment, uniformity and effective
rank are what make that visible, and they have to be measured on the same held-out pairs
the ranking numbers came from or they describe a different model.

TWO EVAL TARGETS, ONE BATTERY, TWO RECEIPT KINDS
The default pass (`kind="eval"`) scores the fp32 checkpoint a training receipt names.
`--quantized PATH` scores a PACKED artifact instead -- `cogsyndelta.quant.ptq`'s
`load_packed_artifact` + `unpack_state_dict` rebuild a plain fp32 state dict from the
sub-byte codes, loaded into the same model class, run through the identical battery. That
receipt gets a distinct `kind` (`"eval-quantized"`) and binds to BOTH the packed file's own
sha256 (computed from the bytes this process actually opened, not trusted from a receipt)
and the fp32 parent checkpoint's sha256 (read from the training receipt, which
`scripts/csd-quantize.py` already verified against the checkpoint on disk when it built the
artifact) -- so "the quantized number" and "the fp32 number" are provably about the same
weights before and after packing, not two runs that happen to share a region name.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cogsyndelta.eval.benchmark import BenchmarkResult, benchmark_embeddings, profile_latency
from cogsyndelta.pipeline.receipt import Producer, Receipt

STATE = Path("/akula-data/csd")


def _regions_spec() -> dict:
    path = Path(__file__).parent / "csd-train-all.py"
    spec = importlib.util.spec_from_file_location("csd_train_all", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {"REGIONS": mod.REGIONS, "_shards": mod._shards, "region_spec": mod.region_spec}


class AmbiguousTrainReceiptError(RuntimeError):
    """More than one training receipt matches, and nothing said which one to use."""


def _find_train_receipt(
    region: str, state: Path, train_receipt_path: Path | None = None
) -> Path | None:
    """Resolve the training receipt for `region`.

    An explicit `train_receipt_path` (A7: a caller running many cells against one shared
    `--state` root names its own cell's receipt) is trusted outright and must exist --
    the whole point of the explicit path is that this function stops guessing. An
    explicit path therefore WINS over a newer sibling in the same directory; DESIGN.v2
    §6.1 names that case specifically and `tests/test_benchmark_train_receipt_binding.py`
    pins it.

    With no explicit path and exactly one candidate, that candidate is used -- a bare
    `--regions foo` against a fresh state root is unaffected. With no explicit path and
    SEVERAL candidates this REFUSES and lists them (M1). The old behaviour was
    `sorted(...)[-1]`: the lexicographically last name, which for a `%Y%m%dT%H%M%SZ`
    stamp is the newest file, chosen silently. In a shared `--state` root -- the exact
    situation A7 exists for, and the one a human running these scripts by hand is
    already in -- that binds the eval to whichever run happened to finish most recently,
    and the receipt records the resulting number as if it had been asked for. Live on
    this fleet tonight: `--state /akula-data/csd --regions memory` picks the V1 receipt,
    while the V2 receipt this branch's tests treat as production lives under a different
    state root entirely.

    Returns `None` when nothing matches, same as the original behaviour: the caller
    prints "no training receipt" and skips the region.
    """
    if train_receipt_path is not None:
        if not train_receipt_path.is_file():
            raise FileNotFoundError(f"--train-receipt {train_receipt_path} does not exist")
        return train_receipt_path
    receipts = sorted(state.glob(f"receipts/{region}-2*.json"))
    if not receipts:
        return None
    if len(receipts) > 1:
        listed = "\n  ".join(str(path) for path in receipts)
        raise AmbiguousTrainReceiptError(
            f"{len(receipts)} training receipts match region {region!r} under "
            f"{state}/receipts and no --train-receipt was given:\n  {listed}\n"
            "Pass --train-receipt PATH to name the one this eval is about."
        )
    return receipts[0]


class UnboundTrainReceiptError(RuntimeError):
    """A training receipt names a checkpoint but records no sha256 for it."""


def expected_checkpoint_sha256(
    train_receipt: dict, receipt_path: Path, *, allow_unbound: bool = False
) -> str | None:
    """The fp32 checkpoint sha this receipt binds to, or a refusal (H2/M4).

    TWO PLACES, NOT ONE. Training receipts written by `regions/pretrain.py` record
    `checkpoint_sha256` at the TOP LEVEL; the same receipt read back through
    `cogsyndelta.pipeline.receipt.adapt` -- and through the matrix harness's own
    `receipts.adapt`, which folds the legacy top-level field into `artifacts` for
    receipt SELECTION -- carries it at `artifacts.checkpoint_sha256`. This function
    previously read only the top-level key, with `or None` behind it, so an
    envelope-shaped receipt (the shape A6 itself writes, and the shape anything
    downstream of `adapt` hands back) silently skipped the checkpoint hash check
    entirely: the guard did not fail, it did not run.

    REFUSAL RATHER THAN SKIP when neither key is present (M4). `or None` also treated
    "" and a missing key as "no check needed", which is right for a genuinely pre-R9
    receipt and wrong for everything else -- and the two are indistinguishable from
    inside this function. Making the caller say so explicitly, with
    `--allow-unbound-train-receipt`, is the difference between a documented exception
    and a hole: an eval whose receipt cannot be tied to the bytes it measured produces
    a number bound to a mutable path, which is exactly what
    `scripts/csd-publish-checkpoint.py` refuses to publish and what DESIGN.v2 §4.4's
    verify step 2 compares against the Hub's own LFS sha.

    Raises:
        ValueError: the two locations disagree -- the receipt describes two different
            checkpoints and there is no safe way to pick one.
        UnboundTrainReceiptError: neither location carries a sha and `allow_unbound`
            is false.
    """
    top = str(train_receipt.get("checkpoint_sha256") or "")
    nested = str((train_receipt.get("artifacts") or {}).get("checkpoint_sha256") or "")
    if top and nested and top != nested:
        raise ValueError(
            f"{receipt_path}: checkpoint_sha256 {top} at the top level disagrees with "
            f"artifacts.checkpoint_sha256 {nested} -- the receipt describes two "
            "different checkpoints"
        )
    sha = top or nested
    if sha:
        return sha
    if allow_unbound:
        print(
            f"    WARNING: {receipt_path} records no checkpoint_sha256; the checkpoint "
            "is loaded UNVERIFIED (--allow-unbound-train-receipt)",
            flush=True,
        )
        return None
    raise UnboundTrainReceiptError(
        f"{receipt_path} records no checkpoint_sha256 at the top level or under "
        "artifacts -- the eval could not be bound to the bytes it measures. Pass "
        "--allow-unbound-train-receipt to score it anyway (pre-R9 receipts predate "
        "checkpoint fingerprinting)."
    )


def _region_eval_context(region: str, train_receipt: dict) -> tuple:
    """Everything a battery pass needs BEFORE it touches a checkpoint: the region's
    `PretrainConfig`, the held-out split, a tokenizer and the device -- rebuilt from the
    training receipt exactly as `scripts/csd-quantize.py`'s `quantize_text_region` does,
    for the identical reason (see that function's own comments): re-globbing the corpus
    instead of trusting the receipt's own recorded shard names would silently widen it,
    changing the holdout, voiding the untrained-baseline comparison the caller's `gates`
    block performs. Shared between the fp32 and quantized eval paths so that comparison
    is provably against the SAME split either way.
    """
    from tokenizers import Tokenizer

    from cogsyndelta.corpus import fingerprint_corpus, verify_corpus_fingerprint
    from cogsyndelta.regions.pretrain import PretrainConfig, build_splits
    from cogsyndelta.regions.text_encoder import TextEncoderConfig

    spec = _regions_spec()
    entry = spec["region_spec"](region)
    sources = entry.sources

    # `entry.root` -- not the `_shards` default -- because REGION_CORPUS_ROOT (`reason`
    # lives at /bulk/csd-corpus, not the shared mount `_shards` defaults to) is resolved
    # by `region_spec` now, and every consumer of `sources` must resolve against the same
    # root `run_region` trained against or this silently globs zero shards for `reason`.
    resolved = [(spec["_shards"](g, entry.root), tuple(c), cap) for g, c, cap in sources]

    wanted = train_receipt.get("corpus", {}).get("shards", [])
    if wanted:
        by_name = {Path(p).name: p for p in resolved[0][0]}
        missing = [n for n in wanted if n not in by_name]
        if missing:
            raise RuntimeError(
                f"{region}: shards recorded in the receipt are no longer on disk: {missing}"
            )
        primary = [by_name[n] for n in wanted]
    else:
        primary = resolved[0][0]

    cfg_d = train_receipt["config"]
    extra_sources = [{"shards": s, "columns": list(c), "limit": cap} for s, c, cap in resolved[1:]]
    fingerprint = fingerprint_corpus(
        primary,
        columns=list(cfg_d["pair_columns"]),
        extra_sources=extra_sources,
    )
    verify_corpus_fingerprint(train_receipt.get("corpus", {}), fingerprint, region)

    enc = TextEncoderConfig(**cfg_d["encoder"])
    cfg = PretrainConfig(
        region=region,
        pair_columns=tuple(cfg_d["pair_columns"]),
        shards=primary,
        extra_sources=extra_sources,
        steps=cfg_d["steps"],
        batch_size=cfg_d["batch_size"],
        max_len=cfg_d["max_len"],
        holdout_pairs=cfg_d["holdout_pairs"],
        seed=cfg_d["seed"],
        tokenizer_path=cfg_d["tokenizer_path"],
        encoder=enc,
    )
    holdout, _t, _m = build_splits(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(cfg.tokenizer_path)
    return cfg, enc, holdout, tok, device


def _run_battery(
    model: torch.nn.Module, tok, holdout: list, cfg, device: torch.device, stored_bytes: int
) -> BenchmarkResult:
    """The one battery every eval receipt reports, run against whatever `model` is --
    fp32 or the dequantized reconstruction of a packed artifact. Identical code path for
    both is the point: a difference between the two receipts is then a fact about the
    weights, never an artefact of measuring them two different ways.
    """
    from cogsyndelta.regions.pretrain import _tokenize

    with torch.no_grad():
        a_ids, a_mask = _tokenize(tok, [a for a, _ in holdout], cfg.max_len, device)
        p_ids, p_mask = _tokenize(tok, [b for _, b in holdout], cfg.max_len, device)
        anchors = model(a_ids, a_mask).float().cpu()
        positives = model(p_ids, p_mask).float().cpu()

    batch = a_ids[: min(64, a_ids.size(0))]
    mask = a_mask[: min(64, a_mask.size(0))]
    lat = profile_latency(lambda: model(batch, mask), warmup=5, runs=40, device=str(device))

    params = sum(p.numel() for p in model.parameters())
    return benchmark_embeddings(anchors, positives, params, stored_bytes, lat)


def _apply_metrics_v2_renames(res: BenchmarkResult) -> None:
    """Rename `effective_rank_ratio` -> `effective_rank_entropy_ratio` in place
    (g7-latent-eval-metrics.md §3.2: "Rename `uses_its_dimensions` to
    `repr.effective_rank_entropy_ratio`" -- the ENTROPY definition, distinguished from
    the participation-ratio one `token_aware.final_block_rank.*_pr_rank` reports;
    MM §9). The 0.05 floor is unchanged ("kept as recorded"), only the name.

    Mutates `res.representation` (and so `res.flat()`, which reads it) IN PLACE rather
    than renaming the field at its source in `cogsyndelta.eval.benchmark.effective_rank`
    / `BenchmarkResult` -- that module is outside this lane's file scope for this
    change. Doing the rename here, at the one place this script turns a
    `BenchmarkResult` into a receipt, keeps `eval/benchmark.py`'s own public dict shape
    untouched for any other caller while still shipping the v2 name on disk.
    """
    if "effective_rank_ratio" in res.representation:
        res.representation["effective_rank_entropy_ratio"] = res.representation.pop(
            "effective_rank_ratio"
        )


def _metric_groups(battery_id: str, seed: int) -> dict:
    """Per-metric-group provenance (g7-latent-eval-metrics.md §3.1/§3.3): every group
    this receipt's `metrics` dict reports gets its own `battery_id`, `pooling` and
    `seed`, because `compare()` refuses to diff two numbers unless (among other things)
    those three agree -- a `rank.recall@1` and a `repr.anisotropy` from the SAME
    receipt were measured over different pools and must never be treated as
    interchangeable just because they share a `started_utc`.

    ONE ENTRY PER *POOL*, NOT PER FIELD-NAME PREFIX. `BenchmarkResult.flat()` groups its
    keys under three prefixes (`rank.`/`eff.`/`repr.`), but two `repr.*` fields are
    measured over a DIFFERENT pool than the rest of that family (MM §3.10(d), §12.5) --
    folding them into a bare `"repr"` entry would make a `MetricIdentity` (MM §14) built
    off that entry read `pooling="pooled_both"` for a field that is actually `anchor`-
    or `matched`-pooled, exactly the mismeasurement `compare()`'s refuse predicate exists
    to catch, and a review caught it doing (an earlier review caught the same defect for
    `quant.artifact_recall@1` and it was fixed by giving IT its own entry -- see the
    `quant` entry `benchmark_region_quantized` adds below; this follows that shape). A
    metric whose true pool differs from its prefix family's dominant one gets a key named
    after its own dotted field name (`"repr.emb_std_anchor"`, `"repr.alignment"`), so a
    reader/caller can always do `groups.get(full_name, groups[prefix])` and get the right
    pool either way.

    - `rank.*`: `recall_at_k`/`mean_reciprocal_rank`/`ndcg_at_k`/`average_precision`/
      `precision_at_k`/`candidates`, MM §3.1-§3.6 -- the closed, single-relevant-item,
      matched-diagonal pool (`relevant[i] = i`) built from THIS holdout
      (`benchmark_embeddings`, `src/cogsyndelta/eval/benchmark.py:355-369`).
    - `eff.*`: MM §3.7/§3.8. Not itself a pooled retrieval quantity, but
      `capability_per_param`/`capability_per_mb` are `rank.recall@1` divided by a
      constant (`src/cogsyndelta/eval/benchmark.py:378-379`) -- inherits `rank.*`'s
      pool/battery/seed rather than invent a "no pooling" value the closed
      `pooling` enum has no slot for.
    - `repr.*`: MM §3.9/§3.11/§3.12 -- `anisotropy`/`uniformity`/`effective_rank`/
      `dimensions`/`effective_rank_entropy_ratio` are computed over
      `cat([anchors, positives])`, i.e. `pooled_both` (`benchmark_embeddings`,
      `src/cogsyndelta/eval/benchmark.py:392-399`), subsample seed 0 by default (MM
      §3.9(e)) -- NOT this battery's corpus/holdout seed, which is why this group
      records `seed=0` rather than the `seed` argument.
    - `repr.emb_std_anchor`: MM §12.5 -- anchors ONLY (`representation_std(a)`,
      `src/cogsyndelta/eval/benchmark.py:455`), never `pooled_both`. Own entry,
      `pooling="anchor"`, same `battery_id`/`seed=0` as the `repr` family it is
      otherwise measured alongside.
    - `repr.alignment`: MM §3.10(d)/§12.4 -- MATCHED pairs (anchor row `i` against
      positive row `i`), never `pooled_both`. Own entry, `pooling="matched"`, same
      `battery_id`/`seed=0` as the `repr` family it is otherwise measured alongside.

    Args:
        battery_id: `"eval_holdout"` for a `kind="eval"` receipt, `"eval_quantized_holdout"`
            for `kind="eval-quantized"` (g7 §3.3's closed `battery_id` set) -- the two
            are never the same battery even though they run the identical code path,
            because one scores the fp32 checkpoint and the other the packed artifact.
        seed: The corpus/holdout-construction seed this battery's split was rebuilt
            from (`cfg.seed`, read back from the training receipt) -- what MM §10 item 7
            calls the seed that "drives the corpus reservoir sample, the shuffle, and
            the untrained model's initial weights". Applied to the `rank`/`eff` groups;
            every `repr*` group's own subsample seed is fixed at 0 regardless (see above).
    """
    return {
        "rank": {"battery_id": battery_id, "pooling": "matched", "seed": seed},
        "eff": {"battery_id": battery_id, "pooling": "matched", "seed": seed},
        "repr": {
            "battery_id": battery_id,
            "pooling": "pooled_both",
            "seed": 0,
        },
        "repr.emb_std_anchor": {
            "battery_id": battery_id,
            "pooling": "anchor",
            "seed": 0,
        },
        "repr.alignment": {
            "battery_id": battery_id,
            "pooling": "matched",
            "seed": 0,
        },
    }


def _print_battery(res: BenchmarkResult, *, size_note: str) -> None:
    r, e, rep = res.ranking, res.efficiency, res.representation
    # `map` is deliberately not printed: metrics-v2 §3.2 retires it as a displayed
    # column on this closed-pool battery (it is always == `mrr` here -- see
    # `eval/benchmark.py`'s `benchmark_embeddings` comment) and `res.ranking` no
    # longer carries a "map" key at all.
    print(
        f"    rank  r@1={r['recall@1']:.4f} ndcg@10={r['ndcg@10']:.4f} mrr={r['mrr']:.4f}",
        flush=True,
    )
    print(
        f"    eff   {e['stored_mb']:.1f}MB ({size_note})  "
        f"p50={e.get('latency_p50_ms', 0):.2f}ms p99={e.get('latency_p99_ms', 0):.2f}ms  "
        f"{e.get('throughput_per_s', 0):.0f}/s",
        flush=True,
    )
    # `effective_rank` -> `effective_rank_entropy`: metrics-v2 §3.1 canonical name
    # (`repr.effective_rank_entropy`); `res.representation` carries only the renamed
    # key (see `eval/benchmark.py`'s `benchmark_embeddings`).
    print(
        f"    repr  anisotropy={rep['anisotropy']:.4f}  eff_rank={rep['effective_rank_entropy']:.1f}"
        f"/{rep['dimensions']:.0f} ({rep['effective_rank_entropy_ratio']:.1%})  "
        f"align={rep['alignment']:.4f} unif={rep['uniformity']:.4f}",
        flush=True,
    )
    print(
        f"    thesis  {e['capability_per_param']:.4f} recall per Mparam, "
        f"{e['capability_per_mb']:.4f} per MB",
        flush=True,
    )


def benchmark_region(
    region: str,
    state: Path,
    train_receipt_path: Path | None = None,
    *,
    allow_unbound_train_receipt: bool = False,
) -> Receipt | None:
    """The fp32 pass: score the checkpoint `region`'s training receipt names."""
    from cogsyndelta.quant.ptq import fp32_reference_bytes
    from cogsyndelta.regions._checkpoint import load_checkpoint, sha256_file
    from cogsyndelta.regions.text_encoder import TextEncoder

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()

    resolved_path = _find_train_receipt(region, state, train_receipt_path)
    if resolved_path is None:
        print(f"    no training receipt for {region}", flush=True)
        return None
    train_receipt = json.loads(resolved_path.read_text())

    cfg, enc, holdout, tok, device = _region_eval_context(region, train_receipt)
    model = TextEncoder(enc, name=region).to(device).eval()
    # load_checkpoint: `train_receipt["checkpoint"]` is a path read out of a receipt
    # JSON on the NFS-exported receipts tree (rw, no_root_squash) -- anyone who can write
    # there can name an arbitrary file, so this load must not execute arbitrary pickle
    # bytecode (load_checkpoint hardcodes weights_only=True internally -- there is no
    # argument here that could turn it off) and must refuse a file that does not hash to
    # what the SAME receipt already recorded (expected_sha256) -- both BEFORE torch.load
    # ever opens it. `expected_checkpoint_sha256` reads BOTH places a receipt can carry
    # that hash and refuses outright when neither has one (H2/M4); it is the only source
    # of `expected_sha256` here precisely so the skip cannot happen by omission.
    ckpt_sha_out: list[str] = []
    ck = load_checkpoint(
        train_receipt["checkpoint"],
        expected_sha256=expected_checkpoint_sha256(
            train_receipt, resolved_path, allow_unbound=allow_unbound_train_receipt
        ),
        map_location=device,
        sha256_out=ckpt_sha_out,
    )
    model.load_state_dict(ck["model"])
    checkpoint_sha256 = ckpt_sha_out[0]

    # The fp32 model's OWN weight bytes -- never a quantized artifact's, and never the
    # checkpoint FILE's `stat().st_size` either. A resumable training checkpoint also
    # carries the Adam optimizer's momentum and variance buffers (`opt.state_dict()`,
    # see `regions/pretrain.py`'s checkpoint dict) -- routinely ~2x the weights
    # themselves -- so the file's size is not "the fp32 model", it is "the fp32 model
    # plus training bookkeeping that never ships". `fp32_reference_bytes` is the exact
    # function `cogsyndelta.quant.ptq.build_plan` uses for `QuantPlan.fp32_bytes`, the
    # denominator of a quant receipt's `compression_ratio` -- calling it here, on the
    # SAME `model` this pass just loaded, is what makes this receipt's `eff.stored_mb`
    # divide into `benchmark_region_quantized`'s `packed_stored_bytes` at the same
    # ratio the quantizer itself measured, rather than a second, incompatible number
    # that happens to also be called "fp32 size" (N5). This receipt's `kind` is "eval"
    # and its `provenance.eval_target` is "fp32"; a reader dividing this receipt's
    # `capability_per_mb` by anything but this figure would be comparing capability
    # against the wrong artifact. The quantized number belongs solely to
    # `benchmark_region_quantized`'s "eval-quantized" receipt, which measures
    # `packed_stored_bytes` on the artifact it actually opened.
    stored = fp32_reference_bytes(model)

    res = _run_battery(model, tok, holdout, cfg, device, stored)
    _apply_metrics_v2_renames(res)
    r, e, rep = res.ranking, res.efficiency, res.representation
    _print_battery(res, size_note="fp32")

    return Receipt(
        producer=Producer("cogsyndelta", region, "dense-transformer"),
        stage="eval",
        kind="eval",
        metrics=res.flat(),
        baseline={"rank.recall@1": train_receipt["untrained_baseline"]["recall@1"]},
        gates={
            # g7 §3.2: "two names because two predicates" -- this receipt's own
            # unmargined `rank.recall@1 > untrained_baseline.recall@1` (MM §3.13(f))
            # is NOT the training receipt's `_beats_untrained_gate` (baseline_sane AND
            # a +0.01 margin, MM §1); `beats_untrained_train` names that one.
            "beats_untrained_eval": r["recall@1"] > train_receipt["untrained_baseline"]["recall@1"],
            # `not_anisotropic` (g7 §3.2): DEMOTED from a gating admission test to a
            # recorded value -- no new bound without a study; `repr.anisotropy` above
            # is still recorded in `metrics`, it is simply no longer read as pass/fail.
            "uses_its_dimensions": rep["effective_rank_entropy_ratio"] > 0.05,
        },
        artifacts={
            "checkpoint": train_receipt["checkpoint"],
            # The sha256 `load_checkpoint` just verified (when the training receipt
            # carried one) or computed (when it did not, e.g. a pre-R9 receipt) --
            # never a second, independent hash of the same bytes. This is what lets
            # `scripts/csd-publish-checkpoint.py` bind this eval receipt to the exact
            # checkpoint it was measured on, not merely the mutable path both name.
            "checkpoint_sha256": checkpoint_sha256,
            "source_training_receipt": {
                "path": str(resolved_path),
                "sha256": sha256_file(resolved_path),
            },
        },
        provenance={
            "holdout_pairs": len(holdout),
            "eval_target": "fp32",
            # See the `stored` comment above: this is `fp32_reference_bytes`'s
            # definition, the same one `quant.compression_ratio`'s denominator uses --
            # never a checkpoint file's raw `stat().st_size`, which includes optimizer
            # state.
            "stored_bytes_definition": "weights-only",
            "metric_groups": _metric_groups("eval_holdout", cfg.seed),
        },
        detail={"family_split": {"ranking": r, "efficiency": e, "representation": rep}},
        started_utc=started_utc,
        seconds=time.time() - t0,
        device=str(device),
    )


def benchmark_region_quantized(
    region: str,
    state: Path,
    quantized_path: Path,
    quant_receipt_path: Path | None = None,
    train_receipt_path: Path | None = None,
    *,
    allow_unbound_train_receipt: bool = False,
) -> Receipt:
    """The quantized pass (A6, closing C1): score the PACKED artifact, not the plan.

    `csd-quantize.py`'s own `quantized_metric` is measured on the in-memory model with
    the quantization plan applied, BEFORE `save_packed_artifact` ever writes a file (see
    that script's module docstring) -- a real number, but about the plan, not about the
    bytes that get published. This function is what closes that gap: it opens the actual
    `.ptq.pt` file, unpacks it into a real `TextEncoder`, and runs the identical battery
    `benchmark_region` runs on the fp32 checkpoint, so the two receipts are comparable by
    construction rather than by two different measurement procedures agreeing by luck.

    Raises:
        ValueError: the packed file's own sha256 does not match what `quant_receipt_path`
            (when given) recorded, or that receipt's fp32 parent sha disagrees with the
            training receipt's -- either means the files on disk are not the ones the
            receipts describe, and scoring them would produce a number bound to nothing.
    """
    from cogsyndelta.quant.ptq import (
        load_packed_artifact,
        packed_stored_bytes,
        packed_width_histogram,
        unpack_state_dict,
    )
    from cogsyndelta.regions._checkpoint import sha256_file
    from cogsyndelta.regions.text_encoder import TextEncoder

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()

    quant_receipt: dict | None = None
    quant_receipt_final: Path | None = quant_receipt_path
    if quant_receipt_path is not None:
        quant_receipt = json.loads(quant_receipt_path.read_text())

    # Resolve the training receipt: an explicit --train-receipt wins; failing that, the
    # quant receipt already names the exact training receipt it was built from (A7's
    # `source_training_receipt`, never a second glob against a value that already came
    # from one); failing THAT, fall back to the state-root latest-glob (single-cell use).
    resolved_train_path = train_receipt_path
    if resolved_train_path is None and quant_receipt is not None:
        src = quant_receipt.get("artifacts", {}).get("source_training_receipt", {}).get("path")
        if src:
            resolved_train_path = Path(src)
    train_receipt_path_final = _find_train_receipt(region, state, resolved_train_path)
    if train_receipt_path_final is None:
        raise FileNotFoundError(
            f"no training receipt found for {region!r} under {state}/receipts; pass "
            "--train-receipt explicitly"
        )
    train_receipt = json.loads(train_receipt_path_final.read_text())

    cfg, enc, holdout, tok, device = _region_eval_context(region, train_receipt)

    quantized_path = Path(quantized_path)
    packed = load_packed_artifact(quantized_path)
    # The sha of the bytes THIS process actually opened -- never trusted from a receipt,
    # per `save_packed_artifact`'s own contract (see quant/ptq.py's module docstring):
    # this is what makes the binding below a fact about what was loaded, not an assumption.
    quantized_sha256 = sha256_file(quantized_path)
    if quant_receipt is not None:
        receipt_sha = quant_receipt.get("artifacts", {}).get("quantized_sha256")
        if receipt_sha and receipt_sha != quantized_sha256:
            raise ValueError(
                f"{quantized_path}: sha256 {quantized_sha256} does not match "
                f"--quant-receipt's recorded quantized_sha256 {receipt_sha} -- the file "
                "on disk is not the one that receipt describes"
            )

    # Same two-location read and same refusal as the fp32 path (H2/M4). `allow_unbound`
    # is widened by a quant receipt that carries the fp32 parent sha itself: the eval is
    # then still BOUND -- to the quantizer's own record of the checkpoint it packed --
    # rather than unbound, and refusing would reject a provably-linked chain.
    qc_sha = ""
    if quant_receipt is not None:
        qc_sha = str(quant_receipt.get("artifacts", {}).get("checkpoint_sha256") or "")
    checkpoint_sha256 = (
        expected_checkpoint_sha256(
            train_receipt,
            train_receipt_path_final,
            allow_unbound=allow_unbound_train_receipt or bool(qc_sha),
        )
        or ""
    )
    if quant_receipt is not None:
        if qc_sha and checkpoint_sha256 and qc_sha != checkpoint_sha256:
            raise ValueError(
                f"--quant-receipt's fp32 parent sha256 {qc_sha} disagrees with "
                f"--train-receipt's checkpoint_sha256 {checkpoint_sha256} -- they do not "
                "describe the same fp32 checkpoint"
            )
        checkpoint_sha256 = checkpoint_sha256 or qc_sha or ""

    model = TextEncoder(enc, name=region).to(device)
    model.load_state_dict(unpack_state_dict(packed))
    model = model.eval()

    stored = packed_stored_bytes(packed)
    res = _run_battery(model, tok, holdout, cfg, device, stored)
    _apply_metrics_v2_renames(res)
    r, e, rep = res.ranking, res.efficiency, res.representation
    _print_battery(res, size_note="quantized artifact")

    metrics = res.flat()
    # `quant.artifact_recall@1` (g7 §3.1): the SAME number as `rank.recall@1` above,
    # under the name the "plan-vs-artifact" sameness guard reads (MM §4's special
    # case, g7 §3.3: `quant.plan_recall@1` vs `quant.artifact_recall@1` may be
    # compared ACROSS battery ids by design, on the same sha/holdout, as a check that
    # the quantizer's in-memory plan and the packed file it wrote agree -- never a
    # general cross-battery compare). Recorded here, not only implied by `rank.*`, so
    # a reader of THIS receipt does not have to know csd-quantize.py's field name to
    # find the number that pairs with it.
    metrics["quant.artifact_recall@1"] = metrics["rank.recall@1"]

    artifacts = {
        "checkpoint": train_receipt.get("checkpoint", ""),
        "checkpoint_sha256": checkpoint_sha256,
        "quantized_path": str(quantized_path),
        "quantized_sha256": quantized_sha256,
        "source_training_receipt": {
            "path": str(train_receipt_path_final),
            "sha256": sha256_file(train_receipt_path_final),
        },
    }
    if quant_receipt_final is not None:
        artifacts["source_quant_receipt"] = {
            "path": str(quant_receipt_final),
            "sha256": sha256_file(quant_receipt_final),
        }

    return Receipt(
        producer=Producer("cogsyndelta", region, "dense-transformer"),
        stage="eval",
        kind="eval-quantized",
        metrics=metrics,
        baseline={"rank.recall@1": train_receipt["untrained_baseline"]["recall@1"]},
        gates={
            # See `benchmark_region`'s identical gate for the g7 §3.2 rename rationale
            # (two predicates, two names) and the `not_anisotropic` demotion.
            "beats_untrained_eval": r["recall@1"] > train_receipt["untrained_baseline"]["recall@1"],
            "uses_its_dimensions": rep["effective_rank_entropy_ratio"] > 0.05,
        },
        artifacts=artifacts,
        provenance={
            "holdout_pairs": len(holdout),
            "quantized_size": True,
            "eval_target": "quantized",
            "width_histogram": packed_width_histogram(packed),
            # `packed_stored_bytes` counts packed codes/scale/zero for quantized
            # tensors and 4 bytes/element for the fp32-kept ones it stores verbatim --
            # weights only, same as the fp32 pass's `fp32_reference_bytes` (see
            # `benchmark_region`'s `stored` comment). The two receipts' `eff.stored_mb`
            # are comparable by this shared definition, not by coincidence.
            "stored_bytes_definition": "weights-only",
            # `eval_quantized_holdout`, never `eval_holdout` -- same code path as the
            # fp32 pass, but a DIFFERENT battery: this one scores the packed artifact
            # read off disk, not the in-memory fp32 checkpoint (g7 §3.3's closed
            # `battery_id` set names both separately for exactly this reason).
            "metric_groups": {
                **_metric_groups("eval_quantized_holdout", cfg.seed),
                # `quant.artifact_recall@1` above is `rank.recall@1` verbatim (MM
                # §12.8's plan-vs-artifact sameness special-case, g7 §3.3) -- not an
                # independent measurement, so it shares the `rank` group's
                # battery_id/pooling/seed rather than inventing its own. Without this
                # entry, a caller building a `MetricIdentity` (MM §14) for
                # `quant.artifact_recall@1` off THIS receipt has no
                # battery_id/pooling/seed to read, even though the field is on
                # `metrics`.
                "quant": {
                    "battery_id": "eval_quantized_holdout",
                    "pooling": "matched",
                    "seed": cfg.seed,
                },
            },
        },
        detail={"family_split": {"ranking": r, "efficiency": e, "representation": rep}},
        started_utc=started_utc,
        seconds=time.time() - t0,
        device=str(device),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="code,compress,retrieve")
    ap.add_argument("--state", default=str(STATE))
    ap.add_argument(
        "--train-receipt",
        default=None,
        help="explicit training receipt to evaluate from (single-region only); bypasses "
        "the --state latest-glob",
    )
    ap.add_argument(
        "--allow-unbound-train-receipt",
        action="store_true",
        help="score a training receipt that records no checkpoint_sha256 in either "
        "location (top level or artifacts). The checkpoint is then loaded UNVERIFIED: "
        "the resulting metric is bound to a mutable path, not to bytes. Only for "
        "pre-R9 receipts, which predate checkpoint fingerprinting.",
    )
    ap.add_argument(
        "--quantized",
        default=None,
        help="score this packed artifact (a csd-quantize.py final.ptq.pt) instead of the "
        "fp32 checkpoint; writes a kind=eval-quantized receipt (single-region only)",
    )
    ap.add_argument(
        "--quant-receipt",
        default=None,
        help="the quant receipt final.ptq.pt was built from; binds and cross-checks the "
        "quantized and fp32-parent sha256 (only meaningful with --quantized)",
    )
    args = ap.parse_args()
    state = Path(args.state)
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]

    if args.quantized is not None and len(regions) != 1:
        print("--quantized names one artifact for one region; pass a single --regions value")
        return 2
    if args.quant_receipt is not None and args.quantized is None:
        print("--quant-receipt is only meaningful together with --quantized")
        return 2

    failures = []
    for region in regions:
        print(f"\n=== {region}", flush=True)
        try:
            if args.quantized is not None:
                rec: Receipt | None = benchmark_region_quantized(
                    region,
                    state,
                    Path(args.quantized),
                    quant_receipt_path=Path(args.quant_receipt) if args.quant_receipt else None,
                    train_receipt_path=Path(args.train_receipt) if args.train_receipt else None,
                    allow_unbound_train_receipt=args.allow_unbound_train_receipt,
                )
            else:
                rec = benchmark_region(
                    region,
                    state,
                    train_receipt_path=Path(args.train_receipt) if args.train_receipt else None,
                    allow_unbound_train_receipt=args.allow_unbound_train_receipt,
                )
        except Exception as exc:
            print(f"    FAILED — {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            failures.append(region)
            continue
        if rec is None:
            continue
        path = rec.write(state / "receipts")
        status = "PASS" if rec.passed else "GATE FAIL"
        print(f"    {status}  {path.name}", flush=True)
        if not rec.passed:
            failures.append(region)
    print(f"\n{len(failures)} failure(s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
