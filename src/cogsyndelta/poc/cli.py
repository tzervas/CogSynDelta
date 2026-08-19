"""cogsyndelta-poc CLI: train | compress | bench."""

from __future__ import annotations

import argparse
import json
import sys

from cogsyndelta.contracts.config import PocConfig
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.contracts.metrics import MetricsRecord, MetricsStatus, write_metrics_json
from cogsyndelta.poc.compress import run_compression_bench
from cogsyndelta.poc.train import train_latent_vae


def _add_device(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--seed", type=int, default=42)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cogsyndelta-poc")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train one LatentVAE region")
    _add_device(train)
    train.add_argument("--steps", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--checkpoint", type=str, default="checkpoints/poc_vae.pt")

    compress = sub.add_parser("compress", help="Measure shared-memory compression")
    _add_device(compress)
    compress.add_argument("--bits", type=int, default=8)
    compress.add_argument("--embed-dim", type=int, default=512)
    compress.add_argument("--batch", type=int, default=16)

    bench = sub.add_parser("bench", help="Write metrics JSON")
    _add_device(bench)
    bench.add_argument("--out", type=str, default="benchmark_results/poc_metrics.json")
    bench.add_argument("--steps", type=int, default=30)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ctx = DeviceContext.resolve(args.device)
    cfg = PocConfig(device=args.device, seed=args.seed)

    if args.command == "train":
        cfg.train.steps = args.steps
        cfg.train.batch_size = args.batch_size
        result = train_latent_vae(
            cfg.train, ctx, seed=args.seed, checkpoint_path=args.checkpoint
        )
        print(
            json.dumps(
                {
                    "device": result["device"],
                    "first_loss": result["first_loss"],
                    "last_loss": result["last_loss"],
                    "improved": result["last_loss"] < result["first_loss"],
                    "checkpoint": args.checkpoint,
                },
                indent=2,
            )
        )
        return 0 if result["last_loss"] < result["first_loss"] else 1

    if args.command == "compress":
        cfg.compression.quant_bits = args.bits
        cfg.compression.embed_dim = args.embed_dim
        records = run_compression_bench(
            cfg.compression, ctx, batch=args.batch, seed=args.seed
        )
        print(json.dumps([r.to_dict() for r in records], indent=2))
        return 0 if all(r.status == MetricsStatus.PASS for r in records) else 1

    if args.command == "bench":
        cfg.train.steps = args.steps
        train_result = train_latent_vae(cfg.train, ctx, seed=args.seed)
        records = run_compression_bench(cfg.compression, ctx, seed=args.seed)
        records.append(
            MetricsRecord(
                name="latent_vae_train",
                claim="loss decreases over steps",
                measured={
                    "first_loss": train_result["first_loss"],
                    "last_loss": train_result["last_loss"],
                    "steps": train_result["steps"],
                },
                status=(
                    MetricsStatus.PASS
                    if train_result["last_loss"] < train_result["first_loss"]
                    else MetricsStatus.GAP
                ),
                device=ctx.name,
            )
        )
        write_metrics_json(args.out, records)
        print(f"Wrote {args.out}")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
