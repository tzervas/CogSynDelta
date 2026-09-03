#!/usr/bin/env python3
"""Quantize trained CSD regions, and prove the result against the training receipt.

WHAT THIS GUARANTEES
A quantization number is only meaningful next to the fp32 number it is compared with, on
the same held-out data. So this does not build its own split: it rebuilds the region's
PretrainConfig, calls the same build_splits() training called, and REFUSES to proceed if
the corpus fingerprint does not match the one recorded in the training receipt. A drifted
shard list would still yield 512 plausible pairs and a plausible recall figure, and the
comparison would quietly mean nothing.

It also recomputes the fp32 metric rather than trusting the receipt's, and reports both.
If the two disagree the split or the checkpoint is not what we think it is, and that is
worth seeing rather than averaging away.

THE OUTPUT
A plan assigning each tensor its own width, the measured metric under that plan, and the
measured stored bytes -- from packed buffers, including per-channel metadata, never from
a nominal bit-width.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

DEFAULT_STATE = Path("/akula-data/csd")


def _load_regions_spec() -> dict:
    """Import REGIONS from the training runner so shard globs have ONE definition."""
    path = Path(__file__).parent / "csd-train-all.py"
    spec = importlib.util.spec_from_file_location("csd_train_all", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {"REGIONS": mod.REGIONS, "_shards": mod._shards, "region_spec": mod.region_spec}


def _latest_receipt(state: Path, region: str) -> tuple[Path, dict]:
    found = sorted(state.glob(f"receipts/{region}-2*.json"))
    if not found:
        raise FileNotFoundError(f"no training receipt for {region!r} under {state}/receipts")
    return found[-1], json.loads(found[-1].read_text())


def quantize_text_region(
    region: str, state: Path, tolerance: float, aggressive: int, max_bits: int
) -> dict:
    """Run sensitivity-driven PTQ on one trained text region."""
    from tokenizers import Tokenizer

    from cogsyndelta.corpus import fingerprint_corpus, verify_corpus_fingerprint
    from cogsyndelta.quant.ptq import build_plan
    from cogsyndelta.regions._checkpoint import load_checkpoint, sha256_file
    from cogsyndelta.regions.pretrain import PretrainConfig, build_splits, evaluate
    from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig

    receipt_path, receipt = _latest_receipt(state, region)
    spec = _load_regions_spec()
    entry = spec["region_spec"](region)
    sources = entry.sources

    # `entry.root` -- not the `_shards` default -- because REGION_CORPUS_ROOT (`reason`
    # lives at /bulk/csd-corpus, not the shared mount `_shards` defaults to) is resolved
    # by `region_spec` now, and every consumer of `sources` must resolve against the same
    # root `run_region` trained against or this silently globs zero shards for `reason`.
    resolved = [(spec["_shards"](g, entry.root), tuple(c), cap) for g, c, cap in sources]

    # Select by the shard NAMES the receipt records, not by re-globbing. A glob returns
    # whatever is on disk now, and training may have used a --shard-limit subset -- as
    # `code` did, three of four shards. Re-globbing silently widens the corpus, which
    # changes the dedup order, which changes the holdout. The receipt is the record of
    # what was actually trained on, so it drives the selection and the fingerprint
    # verifies the result.
    wanted = receipt.get("corpus", {}).get("shards", [])
    if wanted:
        by_name = {Path(p).name: p for p in resolved[0][0]}
        missing = [n for n in wanted if n not in by_name]
        if missing:
            raise RuntimeError(
                f"{region}: shards recorded in the receipt are no longer on disk: {missing}"
            )
        primary_shards = [by_name[n] for n in wanted]
    else:
        primary_shards = resolved[0][0]

    # The fingerprint is the contract. If the corpus is not the one training used, every
    # comparison below is void -- so stop rather than report.
    #
    # Rebuilt from the GLOB, deliberately, not from the receipt's own `extra_sources`.
    # Reading the extras back out of the receipt would make them agree with themselves and
    # turn this check into the same kind of tautology the anchor-level contamination guard
    # was. The glob is what is on disk now; the receipt is what training saw; the point is
    # to compare them.
    cfg_d = receipt["config"]
    extra_sources = [{"shards": s, "columns": list(c), "limit": cap} for s, c, cap in resolved[1:]]
    fingerprint = fingerprint_corpus(
        primary_shards,
        columns=list(cfg_d["pair_columns"]),
        extra_sources=extra_sources,
    )
    verify_corpus_fingerprint(receipt.get("corpus", {}), fingerprint, region)

    enc_d = cfg_d["encoder"]
    cfg = PretrainConfig(
        region=region,
        pair_columns=tuple(cfg_d["pair_columns"]),
        shards=primary_shards,
        extra_sources=extra_sources,
        steps=cfg_d["steps"],
        batch_size=cfg_d["batch_size"],
        max_len=cfg_d["max_len"],
        holdout_pairs=cfg_d["holdout_pairs"],
        seed=cfg_d["seed"],
        tokenizer_path=cfg_d["tokenizer_path"],
        encoder=TextEncoderConfig(**enc_d),
    )
    holdout, _train, _meta = build_splits(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(cfg.tokenizer_path)
    model = TextEncoder(TextEncoderConfig(**enc_d), name=region).to(device)
    # load_checkpoint: `receipt["checkpoint"]` is a path read out of a receipt JSON on
    # the NFS-exported receipts tree (rw, no_root_squash) -- anyone who can write there
    # can name an arbitrary file, so this load must not execute arbitrary pickle bytecode
    # (load_checkpoint hardcodes weights_only=True internally -- there is no argument
    # here that could turn it off) and must refuse a file that does not hash to what the
    # SAME receipt already recorded (expected_sha256) -- both BEFORE torch.load ever
    # opens it. Older receipts (pre-R9) have no checkpoint_sha256; `or None` skips the
    # hash check for those rather than refusing every one of them.
    ckpt_sha_out: list[str] = []
    ck = load_checkpoint(
        receipt["checkpoint"],
        expected_sha256=receipt.get("checkpoint_sha256") or None,
        map_location=device,
        sha256_out=ckpt_sha_out,
    )
    model.load_state_dict(ck["model"])
    model.eval()
    checkpoint_sha256 = ckpt_sha_out[0]

    def eval_fn(m: torch.nn.Module) -> float:
        return evaluate(m, tok, holdout, cfg.max_len, device)["recall@1"]

    fp32_metric = eval_fn(model)
    recorded_metric = receipt["held_out"]["recall@1"]
    print(
        f"    fp32 recall@1 recomputed={fp32_metric:.4f}  receipt={recorded_metric:.4f}", flush=True
    )
    if abs(fp32_metric - recorded_metric) > 0.02:
        print(
            "    WARNING: recomputed fp32 differs from the receipt by more than 2 points; "
            "the split or checkpoint is not what the receipt describes",
            flush=True,
        )

    started = time.time()
    plan = build_plan(
        model,
        eval_fn,
        baseline=fp32_metric,
        tolerance=tolerance,
        aggressive_bits=aggressive,
        max_bits=max_bits,
    )
    by_width: dict[int, int] = {}
    for bits in plan.bits.values():
        by_width[bits] = by_width.get(bits, 0) + 1

    print(
        f"    quantized recall@1={plan.metric:.4f}  drop={fp32_metric - plan.metric:.4f} "
        f"(budget {tolerance})",
        flush=True,
    )
    print(
        f"    {plan.fp32_bytes / 1e6:.1f} MB fp32 -> {plan.stored_bytes / 1e6:.1f} MB "
        f"({plan.ratio:.2f}x)  widths={dict(sorted(by_width.items()))}  "
        f"fp32-kept={len(plan.fp32)} tensors",
        flush=True,
    )
    print(f"    {len(plan.promotions)} promotion(s), {time.time() - started:.0f}s", flush=True)

    return {
        "region": region,
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "sensitivity-greedy-mixed-width",
        "checkpoint": receipt["checkpoint"],
        "artifacts": {
            "checkpoint": receipt["checkpoint"],
            # The sha256 `load_checkpoint` just verified (when the training receipt
            # carried one) or computed (when it did not, e.g. a pre-R9 receipt) --
            # never a second, independent hash of the same bytes. This is what lets
            # `scripts/csd-publish-checkpoint.py` bind this quant receipt to the exact
            # checkpoint it was measured on, not merely the mutable path both name.
            "checkpoint_sha256": checkpoint_sha256,
            "source_training_receipt": {
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
            },
        },
        "corpus_fingerprint": fingerprint,
        "tolerance": tolerance,
        "aggressive_bits": aggressive,
        "max_bits": max_bits,
        "fp32_metric_recomputed": fp32_metric,
        "fp32_metric_receipt": recorded_metric,
        "quantized_metric": plan.metric,
        "drop": fp32_metric - plan.metric,
        "within_budget": (fp32_metric - plan.metric) <= tolerance,
        "fp32_bytes": plan.fp32_bytes,
        "stored_bytes": plan.stored_bytes,
        "compression_ratio": plan.ratio,
        "width_histogram": {str(k): v for k, v in sorted(by_width.items())},
        "bits": plan.bits,
        "fp32_tensors": plan.fp32,
        "promotions": plan.promotions,
    }


def main() -> int:
    from cogsyndelta.regions._receipt import write_receipt

    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(DEFAULT_STATE))
    ap.add_argument("--regions", default="code,compress,retrieve")
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="largest acceptable absolute drop in the region's metric",
    )
    ap.add_argument("--aggressive-bits", type=int, default=3)
    ap.add_argument("--max-bits", type=int, default=8)
    args = ap.parse_args()

    state = Path(args.state)
    out_dir = state / "receipts"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"CSD quantization — tolerance={args.tolerance} ladder {args.aggressive_bits}..{args.max_bits}",
        flush=True,
    )

    results, failures = [], []
    for region in [r.strip() for r in args.regions.split(",") if r.strip()]:
        print(f"\n=== {region}", flush=True)
        try:
            rec = quantize_text_region(
                region, state, args.tolerance, args.aggressive_bits, args.max_bits
            )
        except Exception as exc:
            print(f"    FAILED — {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            failures.append(region)
            continue
        write_receipt(
            rec,
            out_dir,
            f"{region}-quant-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json",
        )
        results.append(rec)
        if not rec["within_budget"]:
            failures.append(region)

    print(
        f"\n{len(results)} region(s) quantized, {len(failures)} outside budget or failed",
        flush=True,
    )
    for r in results:
        print(
            f"  {r['region']:<10} {r['compression_ratio']:.2f}x  "
            f"{r['fp32_metric_recomputed']:.4f} -> {r['quantized_metric']:.4f}  "
            f"{'OK' if r['within_budget'] else 'OVER BUDGET'}",
            flush=True,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
