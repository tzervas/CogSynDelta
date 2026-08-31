#!/usr/bin/env python3
"""Run PoC CUDA checks on a live GPU. No pytest. Receipt JSON to --out.

GPU CI images have torch+CUDA but not pytest. Do not import tests/test_poc_cuda.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _device_cuda_resolve() -> None:
    from cogsyndelta.contracts.device import DeviceContext

    ctx = DeviceContext.resolve("cuda")
    assert ctx.is_cuda
    assert ctx.name.startswith("cuda")
    assert ctx.device.type == "cuda"


def _train_loss_decreases_cuda() -> None:
    from cogsyndelta.contracts.config import TrainConfig
    from cogsyndelta.contracts.device import DeviceContext
    from cogsyndelta.poc.train import train_latent_vae

    cfg = TrainConfig(steps=40, batch_size=32, hidden_dim=64, latent_dim=8, learning_rate=1e-2)
    ctx = DeviceContext.resolve("cuda")
    result = train_latent_vae(cfg, ctx, seed=123)
    assert result["device"].startswith("cuda")
    assert result["last_loss"] < result["first_loss"], (
        f"first={result['first_loss']:.4f} last={result['last_loss']:.4f}"
    )


def _compression_bench_cuda() -> None:
    from cogsyndelta.contracts.config import CompressionConfig
    from cogsyndelta.contracts.device import DeviceContext
    from cogsyndelta.contracts.metrics import MetricsStatus
    from cogsyndelta.poc.compress import run_compression_bench

    cfg = CompressionConfig(embed_dim=128, basis_rank=32, quant_bits=8, min_fidelity=0.85)
    ctx = DeviceContext.resolve("cuda")
    records = run_compression_bench(cfg, ctx, batch=8, seed=7)
    by_name = {r.name: r for r in records}
    assert by_name["basis_residual"].status == MetricsStatus.PASS
    assert by_name["basis_residual"].device.startswith("cuda")
    assert by_name["basis_residual"].measured["fidelity"] >= 0.99
    quant = by_name["calibrated_quant_8bit"]
    assert quant.status == MetricsStatus.PASS
    assert quant.measured["fidelity"] >= 0.85


def _route_bench_cuda() -> None:
    from cogsyndelta.contracts.config import RouteConfig
    from cogsyndelta.contracts.device import DeviceContext
    from cogsyndelta.contracts.metrics import MetricsStatus
    from cogsyndelta.poc.route import run_route

    cfg = RouteConfig(stream_dim=32, hidden_dim=64, latent_dim=8, top_k=1, batch_size=8)
    ctx = DeviceContext.resolve("cuda")
    out = run_route(cfg, ctx, seed=11)
    assert out["route"]["device"].startswith("cuda")
    assert out["route"]["regions"] == ["residual_mlp", "stream_vae"]
    assert out["route"]["gate"] == "softmax_topk"
    by_name = {r.name: r for r in out["records"]}
    assert by_name["softmax_router_weights"].status == MetricsStatus.PASS
    assert by_name["routed_stream_quant8"].status == MetricsStatus.PASS
    assert by_name["routed_stream_quant8"].measured["fidelity"] >= 0.90
    load = out["route"]["load"]
    assert set(load) == {"residual_mlp", "stream_vae"}
    assert all(v > 0.0 for v in load.values())
    assert abs(sum(load.values()) - 1.0) < 1e-5


def _train_route_loss_decreases_cuda() -> None:
    from cogsyndelta.contracts.config import RouteTrainConfig
    from cogsyndelta.contracts.device import DeviceContext
    from cogsyndelta.poc.train_route import train_softmax_router

    cfg = RouteTrainConfig(
        steps=24,
        batch_size=8,
        eval_batch_size=8,
        hidden_dim=32,
        stream_dim=16,
        latent_dim=4,
        learning_rate=1e-2,
        aux_coef=1.0,
    )
    ctx = DeviceContext.resolve("cuda")
    result = train_softmax_router(cfg, ctx, seed=42)
    assert result["device"].startswith("cuda")
    assert result["last_loss"] < result["first_loss"], (
        f"first={result['first_loss']:.4f} last={result['last_loss']:.4f}"
    )
    assert result["last_aux_lb"] == result["last_aux_lb"]
    assert abs(result["last_aux_lb"]) != float("inf")
    load = result["load"]
    assert set(load) == {"residual_mlp", "stream_vae"}
    assert {round(v, 6) for v in load.values()} != {0.0, 1.0}, load
    assert all(v > 0.0 for v in load.values()), load


CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("test_device_cuda_resolve", _device_cuda_resolve),
    ("test_train_loss_decreases_cuda", _train_loss_decreases_cuda),
    ("test_compression_bench_cuda", _compression_bench_cuda),
    ("test_route_bench_cuda", _route_bench_cuda),
    ("test_train_route_loss_decreases_cuda", _train_route_loss_decreases_cuda),
]


def main() -> int:
    """Run CUDA PoC checks; write receipt JSON. Returns 0 iff all pass."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    out: dict[str, Any] = {"ok": False, "cuda": False, "device": None, "tests": []}
    try:
        import torch
    except ImportError as exc:
        out["error"] = f"no torch: {exc}"
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return 1
    out["cuda"] = bool(torch.cuda.is_available())
    if out["cuda"]:
        out["device"] = torch.cuda.get_device_name(0)
        out["torch"] = torch.__version__
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))

    failed = 0
    for name, fn in CHECKS:
        rec: dict[str, Any] = {"name": name, "ok": False}
        try:
            fn()
            rec["ok"] = True
        except Exception as exc:
            failed += 1
            rec["error"] = str(exc)
            rec["tb"] = traceback.format_exc()[-1500:]
        out["tests"].append(rec)
    out["ok"] = failed == 0 and bool(out["cuda"]) and bool(out["tests"])
    out["seconds"] = round(time.time() - t0, 2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2)[:8000])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
