"""cogsyndelta-poc CLI: train | compress | bench | route | train-route."""

from __future__ import annotations

import argparse
import json
import math
import sys

from cogsyndelta.contracts.config import PocConfig
from cogsyndelta.contracts.device import DeviceContext
from cogsyndelta.contracts.metrics import MetricsRecord, MetricsStatus, write_metrics_json
from cogsyndelta.data.stream import TEXT_SOURCES, StreamSource, resolve_stream
from cogsyndelta.poc.compress import run_compression_bench
from cogsyndelta.poc.route import run_route
from cogsyndelta.poc.train import train_latent_vae
from cogsyndelta.poc.train_route import train_softmax_router


def _add_device(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    p.add_argument("--seed", type=int, default=42)


def _add_stream(p: argparse.ArgumentParser) -> None:
    """Add ``--stream``, defaulting to the real corpus.

    ``synthetic`` is listed but is never the default and is never chosen for you: it is
    uniform noise, and it exists so a CI runner with no NFS export can smoke the code
    path. Every command prints the resolved provenance, so a number produced under the
    fallback says so on its own line.
    """
    p.add_argument(
        "--stream",
        default="corpus",
        choices=("corpus", *(f"corpus:{name}" for name in sorted(TEXT_SOURCES)), "synthetic"),
        help="Where the [B, D] stream comes from. Default: real corpus text.",
    )


def _stream_for(args: argparse.Namespace, dim: int, ctx: DeviceContext) -> StreamSource:
    """Resolve the CLI's ``--stream`` choice at the width this command needs."""
    return resolve_stream(args.stream, dim, ctx.device, seed=args.seed)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cogsyndelta-poc")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train one LatentVAE region")
    _add_device(train)
    _add_stream(train)
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
    _add_stream(bench)
    bench.add_argument("--out", type=str, default="benchmark_results/poc_metrics.json")
    bench.add_argument("--steps", type=int, default=30)

    route = sub.add_parser("route", help="Softmax-route two stream regions (MoE gate, not mHC)")
    _add_device(route)
    route.add_argument("--top-k", type=int, default=1)
    route.add_argument("--batch", type=int, default=16)
    route.add_argument("--stream-dim", type=int, default=64)
    route.add_argument("--no-compact", action="store_true")

    train_route = sub.add_parser(
        "train-route",
        help="Train softmax gate with Switch aux load-balance (MoE, not mHC)",
    )
    _add_device(train_route)
    _add_stream(train_route)
    train_route.add_argument("--steps", type=int, default=40)
    train_route.add_argument("--batch-size", type=int, default=8)
    train_route.add_argument("--aux-coef", type=float, default=1.0)
    train_route.add_argument(
        "--gate-only",
        action="store_true",
        help="Freeze region params; train the softmax gate only",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Run train, compress, bench, route, or train-route and print JSON.

    Args:
        argv: Optional CLI tokens (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on measured pass).
    """
    args = _build_parser().parse_args(argv)
    ctx = DeviceContext.resolve(args.device)
    cfg = PocConfig(device=args.device, seed=args.seed)

    if args.command == "train":
        cfg.train.steps = args.steps
        cfg.train.batch_size = args.batch_size
        result = train_latent_vae(
            cfg.train,
            ctx,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
            stream=_stream_for(args, cfg.train.input_dim, ctx),
        )
        print(
            json.dumps(
                {
                    "device": result["device"],
                    "stream": result["stream"],
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
        records = run_compression_bench(cfg.compression, ctx, batch=args.batch, seed=args.seed)
        print(json.dumps([r.to_dict() for r in records], indent=2))
        return 0 if all(r.status == MetricsStatus.PASS for r in records) else 1

    if args.command == "bench":
        cfg.train.steps = args.steps
        train_result = train_latent_vae(
            cfg.train, ctx, seed=args.seed, stream=_stream_for(args, cfg.train.input_dim, ctx)
        )
        records = run_compression_bench(cfg.compression, ctx, seed=args.seed)
        records.append(
            MetricsRecord(
                name="latent_vae_train",
                claim="loss decreases over steps",
                measured={
                    "first_loss": train_result["first_loss"],
                    "last_loss": train_result["last_loss"],
                    "steps": train_result["steps"],
                    "stream": train_result["stream"],
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

    if args.command == "route":
        cfg.route.top_k = args.top_k
        cfg.route.batch_size = args.batch
        cfg.route.stream_dim = args.stream_dim
        cfg.route.compact = not args.no_compact
        out = run_route(cfg.route, ctx, seed=args.seed)
        print(
            json.dumps(
                {
                    **out["route"],
                    "records": [r.to_dict() for r in out["records"]],
                },
                indent=2,
            )
        )
        return 0 if all(r.status == MetricsStatus.PASS for r in out["records"]) else 1

    if args.command == "train-route":
        cfg.route_train.steps = args.steps
        cfg.route_train.batch_size = args.batch_size
        cfg.route_train.aux_coef = args.aux_coef
        cfg.route_train.train_regions = not args.gate_only
        result = train_softmax_router(
            cfg.route_train,
            ctx,
            seed=args.seed,
            stream_source=_stream_for(args, cfg.route_train.stream_dim, ctx),
        )
        aux_ok = math.isfinite(result["last_aux_lb"])
        load_vals = list(result["load"].values())
        split = all(v > 0.0 for v in load_vals)
        improved = result["last_loss"] < result["first_loss"]
        print(
            json.dumps(
                {
                    "device": result["device"],
                    "stream": result["stream"],
                    "first_loss": result["first_loss"],
                    "last_loss": result["last_loss"],
                    "improved": improved,
                    "first_aux_lb": result["first_aux_lb"],
                    "last_aux_lb": result["last_aux_lb"],
                    "aux_finite": aux_ok,
                    "load": result["load"],
                    "aux_formula": result["aux_formula"],
                    "steps": result["steps"],
                    "notes": result["notes"],
                },
                indent=2,
            )
        )
        return 0 if improved and aux_ok and split else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
