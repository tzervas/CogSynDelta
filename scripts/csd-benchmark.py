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


def _find_train_receipt(
    region: str, state: Path, train_receipt_path: Path | None = None
) -> Path | None:
    """Resolve the training receipt for `region`.

    An explicit `train_receipt_path` (A7: a caller running many cells against one shared
    `--state` root names its own cell's receipt) is trusted outright and must exist --
    the whole point of the explicit path is that this function stops guessing. With no
    explicit path this falls back to the pre-existing latest-glob under `state`, so a
    bare `--regions foo` invocation is unaffected: `None` means "not found", same as the
    original behaviour.
    """
    if train_receipt_path is not None:
        if not train_receipt_path.is_file():
            raise FileNotFoundError(f"--train-receipt {train_receipt_path} does not exist")
        return train_receipt_path
    receipts = sorted(state.glob(f"receipts/{region}-2*.json"))
    return receipts[-1] if receipts else None


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


def _print_battery(res: BenchmarkResult, *, size_note: str) -> None:
    r, e, rep = res.ranking, res.efficiency, res.representation
    print(
        f"    rank  r@1={r['recall@1']:.4f} ndcg@10={r['ndcg@10']:.4f} map={r['map']:.4f}",
        flush=True,
    )
    print(
        f"    eff   {e['stored_mb']:.1f}MB ({size_note})  "
        f"p50={e.get('latency_p50_ms', 0):.2f}ms p99={e.get('latency_p99_ms', 0):.2f}ms  "
        f"{e.get('throughput_per_s', 0):.0f}/s",
        flush=True,
    )
    print(
        f"    repr  anisotropy={rep['anisotropy']:.4f}  eff_rank={rep['effective_rank']:.1f}"
        f"/{rep['dimensions']:.0f} ({rep['effective_rank_ratio']:.1%})  "
        f"align={rep['alignment']:.4f} unif={rep['uniformity']:.4f}",
        flush=True,
    )
    print(
        f"    thesis  {e['capability_per_param']:.4f} recall per Mparam, "
        f"{e['capability_per_mb']:.4f} per MB",
        flush=True,
    )


def benchmark_region(
    region: str, state: Path, train_receipt_path: Path | None = None
) -> Receipt | None:
    """The fp32 pass: score the checkpoint `region`'s training receipt names."""
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
    # ever opens it. Older receipts (pre-R9) have no checkpoint_sha256; `or None` skips
    # the hash check for those.
    ckpt_sha_out: list[str] = []
    ck = load_checkpoint(
        train_receipt["checkpoint"],
        expected_sha256=train_receipt.get("checkpoint_sha256") or None,
        map_location=device,
        sha256_out=ckpt_sha_out,
    )
    model.load_state_dict(ck["model"])
    checkpoint_sha256 = ckpt_sha_out[0]

    # The fp32 checkpoint's OWN bytes on disk -- never a quantized artifact's, however
    # convenient a same-region `*-quant-*.json` glob might be. This receipt's `kind` is
    # "eval" and its `provenance.eval_target` is "fp32"; a reader dividing this receipt's
    # `capability_per_mb` by anything but the fp32 checkpoint's real size would be
    # comparing capability against the wrong artifact (N5). The quantized number belongs
    # solely to `benchmark_region_quantized`'s "eval-quantized" receipt, which measures
    # `packed_stored_bytes` on the artifact it actually opened.
    stored = Path(train_receipt["checkpoint"]).stat().st_size

    res = _run_battery(model, tok, holdout, cfg, device, stored)
    r, e, rep = res.ranking, res.efficiency, res.representation
    _print_battery(res, size_note="fp32")

    return Receipt(
        producer=Producer("cogsyndelta", region, "dense-transformer"),
        stage="eval",
        kind="eval",
        metrics=res.flat(),
        baseline={"rank.recall@1": train_receipt["untrained_baseline"]["recall@1"]},
        gates={
            "beats_untrained": r["recall@1"] > train_receipt["untrained_baseline"]["recall@1"],
            # A space where unrelated items sit at cosine 0.9+ is degenerate even when the
            # ranking metrics look healthy, so it is a gate rather than a note.
            "not_anisotropic": rep["anisotropy"] < 0.9,
            "uses_its_dimensions": rep["effective_rank_ratio"] > 0.05,
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

    checkpoint_sha256 = train_receipt.get("checkpoint_sha256", "") or ""
    if quant_receipt is not None:
        qc_sha = quant_receipt.get("artifacts", {}).get("checkpoint_sha256")
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
    r, e, rep = res.ranking, res.efficiency, res.representation
    _print_battery(res, size_note="quantized artifact")

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
        metrics=res.flat(),
        baseline={"rank.recall@1": train_receipt["untrained_baseline"]["recall@1"]},
        gates={
            "beats_untrained": r["recall@1"] > train_receipt["untrained_baseline"]["recall@1"],
            "not_anisotropic": rep["anisotropy"] < 0.9,
            "uses_its_dimensions": rep["effective_rank_ratio"] > 0.05,
        },
        artifacts=artifacts,
        provenance={
            "holdout_pairs": len(holdout),
            "quantized_size": True,
            "eval_target": "quantized",
            "width_histogram": packed_width_histogram(packed),
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
                )
            else:
                rec = benchmark_region(
                    region,
                    state,
                    train_receipt_path=Path(args.train_receipt) if args.train_receipt else None,
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
