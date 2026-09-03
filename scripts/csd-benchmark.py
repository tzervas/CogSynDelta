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

from cogsyndelta.eval.benchmark import benchmark_embeddings, profile_latency
from cogsyndelta.pipeline.receipt import Producer, Receipt

STATE = Path("/akula-data/csd")


def _regions_spec() -> dict:
    path = Path(__file__).parent / "csd-train-all.py"
    spec = importlib.util.spec_from_file_location("csd_train_all", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {"REGIONS": mod.REGIONS, "_shards": mod._shards, "region_spec": mod.region_spec}


def benchmark_region(region: str, state: Path) -> Receipt | None:
    from tokenizers import Tokenizer

    from cogsyndelta.regions._checkpoint import load_checkpoint
    from cogsyndelta.regions.pretrain import PretrainConfig, _tokenize, build_splits
    from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig

    receipts = sorted(state.glob(f"receipts/{region}-2*.json"))
    if not receipts:
        print(f"    no training receipt for {region}", flush=True)
        return None
    train_receipt = json.loads(receipts[-1].read_text())

    spec = _regions_spec()
    sources = spec["region_spec"](region).sources
    resolved = [(spec["_shards"](g), tuple(c), cap) for g, c, cap in sources]
    wanted = train_receipt.get("corpus", {}).get("shards", [])
    by_name = {Path(p).name: p for p in resolved[0][0]}
    primary = [by_name[n] for n in wanted if n in by_name] or resolved[0][0]

    cfg_d = train_receipt["config"]
    enc = TextEncoderConfig(**cfg_d["encoder"])
    cfg = PretrainConfig(
        region=region,
        pair_columns=tuple(cfg_d["pair_columns"]),
        shards=primary,
        extra_sources=[
            {"shards": s, "columns": list(c), "limit": cap} for s, c, cap in resolved[1:]
        ],
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
    model = TextEncoder(enc, name=region).to(device).eval()
    # load_checkpoint: `train_receipt["checkpoint"]` is a path read out of a receipt
    # JSON on the NFS-exported receipts tree (rw, no_root_squash) -- anyone who can write
    # there can name an arbitrary file, so this load must not execute arbitrary pickle
    # bytecode (weights_only=True, passed explicitly below and matching
    # load_checkpoint's own default) and must refuse a file that does not hash to what
    # the SAME receipt already recorded (expected_sha256) -- both BEFORE torch.load ever
    # opens it. Older receipts (pre-R9) have no checkpoint_sha256; `or None` skips the
    # hash check for those.
    ck = load_checkpoint(
        train_receipt["checkpoint"],
        expected_sha256=train_receipt.get("checkpoint_sha256") or None,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(ck["model"])

    with torch.no_grad():
        a_ids, a_mask = _tokenize(tok, [a for a, _ in holdout], cfg.max_len, device)
        p_ids, p_mask = _tokenize(tok, [b for _, b in holdout], cfg.max_len, device)
        anchors = model(a_ids, a_mask).float().cpu()
        positives = model(p_ids, p_mask).float().cpu()

    batch = a_ids[: min(64, a_ids.size(0))]
    mask = a_mask[: min(64, a_mask.size(0))]
    lat = profile_latency(lambda: model(batch, mask), warmup=5, runs=40, device=str(device))

    params = sum(p.numel() for p in model.parameters())
    # Prefer the quantized size when one exists: what ships is what should be divided by.
    stored = params * 4
    quant = sorted(state.glob(f"receipts/{region}-quant-*.json"))
    quantized = False
    if quant:
        stored = int(json.loads(quant[-1].read_text())["stored_bytes"])
        quantized = True

    res = benchmark_embeddings(anchors, positives, params, stored, lat)
    r, e, rep = res.ranking, res.efficiency, res.representation
    print(
        f"    rank  r@1={r['recall@1']:.4f} ndcg@10={r['ndcg@10']:.4f} map={r['map']:.4f}",
        flush=True,
    )
    print(
        f"    eff   {e['stored_mb']:.1f}MB{' (quantized)' if quantized else ' (fp32)'}  "
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

    return Receipt(
        producer=Producer("cogsyndelta", region, "dense-transformer"),
        stage="eval",
        metrics=res.flat(),
        baseline={"rank.recall@1": train_receipt["untrained_baseline"]["recall@1"]},
        gates={
            "beats_untrained": r["recall@1"] > train_receipt["untrained_baseline"]["recall@1"],
            # A space where unrelated items sit at cosine 0.9+ is degenerate even when the
            # ranking metrics look healthy, so it is a gate rather than a note.
            "not_anisotropic": rep["anisotropy"] < 0.9,
            "uses_its_dimensions": rep["effective_rank_ratio"] > 0.05,
        },
        artifacts={"checkpoint": train_receipt["checkpoint"]},
        provenance={"holdout_pairs": len(holdout), "quantized_size": quantized},
        detail={"family_split": {"ranking": r, "efficiency": e, "representation": rep}},
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        device=str(device),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="code,compress,retrieve")
    ap.add_argument("--state", default=str(STATE))
    args = ap.parse_args()
    state = Path(args.state)

    failures = []
    for region in [r.strip() for r in args.regions.split(",") if r.strip()]:
        print(f"\n=== {region}", flush=True)
        try:
            rec = benchmark_region(region, state)
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
