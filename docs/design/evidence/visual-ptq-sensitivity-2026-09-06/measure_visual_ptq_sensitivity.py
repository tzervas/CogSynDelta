#!/usr/bin/env python3
"""Is the visual region's 3-bit PTQ plan measuring a robust encoder, or an insensitive probe?

CONTEXT
Both g22 run-2 cells (`docs/design/evidence/g22-visual-prereg-run2-2026-09-05/README.md`)
quantize all 25 target-encoder weight tensors to the 3-bit floor with a EuroSAT pooled
linear-probe drop of at most 0.0026 -- well inside the 0.01 tolerance. That run's README
flags the open question directly: "Either the encoder is genuinely robust to 3-bit weights
on this probe, or the pooled linear probe is an insensitive quantization metric." This
script answers that by measurement, on the seed-1 cell
(`/akula-data/csd/matrix/visual-b128-s1-3ce18db-20260905/`), reusing the SAME probe
protocol/data the harness already uses wherever one exists, and adding three read-outs the
3-bit plan was never optimized against: a transfer set, a pre-pool token-surface probe, and
latent-space geometry.

WHAT THIS DOES NOT DO
It does not touch `src/` or `scripts/`, does not re-run `build_plan`'s sensitivity search,
and does not write a new packed artifact next to the training checkpoint. Task (3)'s ladder
repacks the SAME 25 tensor names the production plan named, at a FIXED width per rung, using
`cogsyndelta.quant.ptq.quantize_tensor` / `dequantize_tensor` (the exact functions the real
packer uses) purely in memory via `apply_plan`. This is a measurement filed as evidence, not
a change to how the region is trained, quantized, or shipped.

THREE MEASUREMENTS
  (1) Reproduce the receipts: fp32 vs the REAL packed 3-bit artifact on disk
      (`step-4000.ptq.pt`), on the exact EuroSAT probe protocol training and eval used.
      Confirms this harness before trusting anything novel it measures.
  (2) On the SAME fp32 and packed-3-bit models, four read-outs the quant plan's tolerance
      check never looked at: the Fashion-t10k transfer probe (top-1 AND top-5 -- the
      production eval-quantized receipt records only top-1), a token-surface probe (a
      linear probe on 16 sampled pre-pool patch tokens per image, see `_token_probe`),
      and latent-space geometry (per-image cosine, rep_std ratio, kNN@10 rank agreement)
      between the fp32 and 3-bit eval latents.
  (3) A bits ladder -- 3, 4, 5, 6, 8 -- repacking the SAME 25 tensors at a FIXED width per
      rung (no sensitivity search), reporting the same four read-out families at each rung
      so the curve shows where each one starts to move.

WHY REUSE `_linear_probe` / `_measure_visual_probes` / `_visual_latents`
`scripts/csd-benchmark.py` and `src/cogsyndelta/regions/vl_pretrain.py` already implement the
exact probe-fitting procedure (feature standardization, seeded head, AdamW, top-1/top-5) the
production receipts report. Reimplementing it here would risk a second, silently different
definition of "linear probe" answering a question about whether ONE probe is sensitive.
Importing and reusing the same functions makes the fp32-vs-3-bit numbers in step (1) a real
reproduction, not a similar-looking new measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC = REPO_ROOT / "src"
SCRIPTS = REPO_ROOT / "scripts"
sys.path.insert(0, str(SRC))

CELL = Path("/akula-data/csd/matrix/visual-b128-s1-3ce18db-20260905")
TRAIN_RECEIPT_PATH = CELL / "receipts/visual-20260905T222732Z.json"
QUANT_RECEIPT_PATH = CELL / "receipts/visual-quant-20260905T223452Z.json"
EVAL_RECEIPT_PATH = CELL / "receipts/cogsyndelta-visual-eval-20260905T222806Z.json"
EVAL_QUANT_RECEIPT_PATH = CELL / "receipts/cogsyndelta-visual-eval-quantized-20260905T223526Z.json"
PACKED_3BIT_PATH = CELL / "receipts/visual-checkpoints/step-4000.ptq.pt"

TOKEN_PROBE_K = 16  # sampled patches per image (task spec: "say, 16")
LADDER_BITS = (3, 4, 5, 6, 8)
NN_K = 10
MEASURE_SEED = 20260906


def _load_benchmark_module() -> Any:
    """Import `scripts/csd-benchmark.py` by path, exactly as `csd-quantize.py` does."""
    path = SCRIPTS / "csd-benchmark.py"
    spec = importlib.util.spec_from_file_location("csd_benchmark_for_ptq_sensitivity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- token probe


def _sample_patch_indices(n_images: int, n_patches: int, k: int, seed: int) -> torch.Tensor:
    """`k` distinct patch positions per image, reproducible from `seed`.

    Vectorized: one uniform-random key per (image, patch), argsort each row, keep the
    first `k` columns. Every image gets its own random subset of patch positions (not the
    same k positions repeated for every image) so the probe sees the encoder's token
    surface generally, not one fixed spatial location.
    """
    g = torch.Generator().manual_seed(seed)
    keys = torch.rand(n_images, n_patches, generator=g)
    return keys.argsort(dim=1)[:, :k]


def _token_features(
    encoder: Any,
    x_u8: torch.Tensor,
    patch_idx: torch.Tensor,
    device: torch.device,
    bs: int = 128,
) -> torch.Tensor:
    """Sampled pre-pool patch features, flattened to one row per (image, sampled patch).

    `encoder` is a `ViTEncoder` (`DeployedVisualEncoder.target_encoder`); `.tokens()` is
    the region-native pre-pool representation (`vl_jepa.py`'s own docstring: "Region-native
    representation BEFORE pooling"). Returned on CPU, float32, shape
    `[n_images * patch_idx.size(1), D]`.
    """
    from cogsyndelta.regions.vl_pretrain import _to_float

    encoder.eval()
    out = []
    with torch.no_grad():
        for i in range(0, x_u8.size(0), bs):
            xb = _to_float(x_u8[i : i + bs], device)
            h, _mask = encoder.tokens(xb)  # [b, n_patches, D]
            idx_b = patch_idx[i : i + bs].to(device)
            idx_exp = idx_b.unsqueeze(-1).expand(-1, -1, h.size(-1))
            sampled = torch.gather(h, 1, idx_exp)  # [b, k, D]
            out.append(sampled.reshape(-1, h.size(-1)).float().cpu())
    return torch.cat(out)


def _token_probe(
    deployed: Any,
    splits: Any,
    device: torch.device,
    tr_idx: torch.Tensor,
    ev_idx: torch.Tensor,
    n_classes: int,
    probe_steps: int,
    probe_lr: float,
    seed: int,
    linear_probe_fn: Any,
) -> dict[str, float]:
    """Linear probe on sampled pre-pool patch tokens, labelled by their image's class.

    Same probe protocol (`_linear_probe`) as the pooled EuroSAT probe -- standardized
    features, seeded head, same steps/lr -- applied to a token-level dataset instead of a
    pooled one: `TOKEN_PROBE_K` patches per image, each row inheriting its image's label.
    This is a coarser signal than the pooled probe (a single patch rarely carries the whole
    scene) but a real one: if quantization degrades individual patch representations while
    the class-relevant signal survives pooling (averaging across 256 patches), the pooled
    probe would show nothing while this one moves.
    """
    enc = deployed.target_encoder
    feats_tr = _token_features(enc, splits.px_tr, tr_idx, device)
    labels_tr = splits.py_tr.repeat_interleave(tr_idx.size(1))
    feats_ev = _token_features(enc, splits.px_ev, ev_idx, device)
    labels_ev = splits.py_ev.repeat_interleave(ev_idx.size(1))
    result = linear_probe_fn(
        feats_tr, labels_tr, feats_ev, labels_ev, n_classes, device, probe_steps, probe_lr, seed
    )
    result["n_eval_tokens"] = float(labels_ev.numel())
    return result


# --------------------------------------------------------------------------- geometry


def _topk_neighbor_mask(latents: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean `[N, N]` mask: `mask[i, j]` iff `j` is one of `i`'s top-`k` cosine neighbours.

    Self-similarity is excluded before the top-k so a latent is never its own neighbour.
    """
    x = F.normalize(latents, dim=1)
    sim = x @ x.T
    sim.fill_diagonal_(-2.0)
    idx = sim.topk(k, dim=1).indices
    mask = torch.zeros_like(sim, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask


def _geometry_vs_fp32(
    fp32_latents: torch.Tensor, variant_latents: torch.Tensor, k: int
) -> dict[str, float]:
    """Per-image cosine, rep_std ratio input, and kNN@k rank agreement, fp32 vs `variant`.

    `rep_std` itself is reported alongside from the caller (it needs the SAME collapse-batch
    definition `_visual_rep_std` uses, not the eval-latents population) -- this function only
    covers what is defined directly on the eval-latent matrix: per-image cosine similarity,
    the mean-pooled latents' own std ratio (a cheap second collapse signal on this exact
    population), and kNN rank agreement.
    """
    cos = F.cosine_similarity(fp32_latents, variant_latents, dim=1)
    fp32_mask = _topk_neighbor_mask(fp32_latents, k)
    variant_mask = _topk_neighbor_mask(variant_latents, k)
    agreement = (fp32_mask & variant_mask).sum(dim=1).float() / k
    return {
        "mean_cosine": float(cos.mean().item()),
        "min_cosine": float(cos.min().item()),
        "eval_latents_std_ratio": float(
            variant_latents.std(dim=0).mean().item()
            / max(1e-9, fp32_latents.std(dim=0).mean().item())
        ),
        "nn_agreement_at_k": float(agreement.mean().item()),
        "k": float(k),
    }


# --------------------------------------------------------------------------- battery


def _run_battery(
    name: str,
    bits: int | None,
    deployed: Any,
    bench: Any,
    cfg: Any,
    device: torch.device,
    splits: Any,
    tr_idx: torch.Tensor,
    ev_idx: torch.Tensor,
    fp32_eval_latents: torch.Tensor | None,
    stored_bytes: int | None,
    fp32_bytes: int,
) -> dict[str, Any]:
    """Every read-out this evidence file reports, for one model variant.

    `fp32_eval_latents=None` marks the fp32 condition itself (geometry is measured against
    it, not against itself); every other condition is compared to it.
    """
    from cogsyndelta.regions.vl_pretrain import _linear_probe

    t0 = time.time()
    held, xfer = bench._measure_visual_probes(deployed, cfg, device, splits)
    eval_latents = bench._visual_latents(deployed, splits.px_ev, device)
    token = _token_probe(
        deployed,
        splits,
        device,
        tr_idx,
        ev_idx,
        splits.n_classes,
        cfg.probe_steps,
        cfg.probe_lr,
        cfg.seed,
        _linear_probe,
    )
    row: dict[str, Any] = {
        "condition": name,
        "bits": bits,
        "primary": {"top1": held["top1"], "top5": held["top5"], "n_eval": held["n_eval"]},
        "rep_std": held["rep_std"],
        "transfer": (
            {"top1": xfer["top1"], "top5": xfer["top5"], "n_eval": xfer["n_eval"]}
            if xfer is not None
            else None
        ),
        "token_surface": {
            "top1": token["top1"],
            "top5": token["top5"],
            "n_eval": token["n_eval_tokens"],
            "k_per_image": TOKEN_PROBE_K,
        },
        "stored_bytes": stored_bytes,
        "fp32_bytes": fp32_bytes,
        "compression_ratio": (fp32_bytes / stored_bytes) if stored_bytes else None,
        "seconds": time.time() - t0,
    }
    if fp32_eval_latents is not None:
        row["geometry_vs_fp32"] = _geometry_vs_fp32(fp32_eval_latents, eval_latents, NN_K)
    else:
        row["geometry_vs_fp32"] = None
    return row, eval_latents


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None, help="cuda or cpu; default auto-detect")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).parent / "results.json"),
        help="where to write the measurement receipt",
    )
    ap.add_argument(
        "--probe-limit",
        type=int,
        default=None,
        help="override cfg.probe_limit (production default 20000) -- for a fast dry run "
        "only; the real measurement leaves this unset so it matches the production "
        "probe-fitting set exactly",
    )
    ap.add_argument(
        "--ladder-bits",
        default=",".join(str(b) for b in LADDER_BITS),
        help="comma-separated bit widths for task (3)'s ladder",
    )
    args = ap.parse_args()
    ladder_bits = tuple(int(b) for b in args.ladder_bits.split(","))

    from cogsyndelta.model.vl_jepa import IJEPA
    from cogsyndelta.quant.ptq import (
        QuantPlan,
        apply_plan,
        fp32_reference_bytes,
        load_packed_artifact,
        unpack_state_dict,
    )
    from cogsyndelta.regions._checkpoint import load_checkpoint

    bench = _load_benchmark_module()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device={device}", flush=True)

    train_receipt = json.loads(TRAIN_RECEIPT_PATH.read_text())
    quant_receipt = json.loads(QUANT_RECEIPT_PATH.read_text())
    eval_receipt = json.loads(EVAL_RECEIPT_PATH.read_text())
    eval_quant_receipt = json.loads(EVAL_QUANT_RECEIPT_PATH.read_text())

    expected_ckpt_sha = bench.require_bound_visual_train_receipt(train_receipt, TRAIN_RECEIPT_PATH)
    cfg = bench._vl_cfg_from_train_receipt("visual", train_receipt)

    model = IJEPA(cfg.jepa).to(device)
    ck = load_checkpoint(
        train_receipt["checkpoint"], expected_sha256=expected_ckpt_sha, map_location=device
    )
    model.load_state_dict(ck["model"])
    deployed_fp32 = bench.wrap_deployed_visual_encoder(model).to(device).eval()
    fp32_bytes = fp32_reference_bytes(deployed_fp32)

    if args.probe_limit is not None:
        print(
            f"WARNING: --probe-limit={args.probe_limit} overrides production probe_limit "
            f"={cfg.probe_limit} -- DRY RUN ONLY, not comparable to the receipts",
            flush=True,
        )
        cfg.probe_limit = args.probe_limit

    print("loading EuroSAT + Fashion-t10k splits (png_zip decode, CPU-bound)...", flush=True)
    t_load = time.time()
    splits = bench.load_visual_splits(cfg)
    print(
        f"  probe_train={splits.py_tr.numel()} probe_eval={splits.py_ev.numel()} "
        f"transfer={'yes' if splits.transfer else 'no'} n_classes={splits.n_classes} "
        f"({time.time() - t_load:.0f}s)",
        flush=True,
    )

    n_patches = (cfg.jepa.image_size // cfg.jepa.patch_size) ** 2
    tr_idx = _sample_patch_indices(splits.py_tr.numel(), n_patches, TOKEN_PROBE_K, MEASURE_SEED)
    ev_idx = _sample_patch_indices(splits.py_ev.numel(), n_patches, TOKEN_PROBE_K, MEASURE_SEED + 1)

    rows: list[dict[str, Any]] = []

    # --- fp32 baseline: also the reference for every geometry comparison below.
    print("\n=== fp32 baseline ===", flush=True)
    fp32_row, fp32_eval_latents = _run_battery(
        "fp32",
        None,
        deployed_fp32,
        bench,
        cfg,
        device,
        splits,
        tr_idx,
        ev_idx,
        None,
        None,
        fp32_bytes,
    )
    print(
        f"  primary top1={fp32_row['primary']['top1']:.4f} (receipt {eval_receipt['metrics']['probe.top1']:.4f})  "
        f"transfer top1={fp32_row['transfer']['top1']:.4f} top5={fp32_row['transfer']['top5']:.4f}  "
        f"token top1={fp32_row['token_surface']['top1']:.4f} top5={fp32_row['token_surface']['top5']:.4f}  "
        f"({fp32_row['seconds']:.0f}s)",
        flush=True,
    )
    rows.append(fp32_row)

    # --- (1) reproduce the receipts: fp32 vs the REAL packed artifact on disk.
    print("\n=== packed 3-bit artifact (step-4000.ptq.pt), task (1) reproduction ===", flush=True)
    packed = load_packed_artifact(PACKED_3BIT_PATH)
    packed_sha = sha256_of(PACKED_3BIT_PATH)
    deployed_packed = bench.wrap_deployed_visual_encoder(IJEPA(cfg.jepa)).to(device)
    deployed_packed.load_state_dict(unpack_state_dict(packed))
    deployed_packed.eval()
    packed_row, _ = _run_battery(
        "packed_artifact_3bit",
        3,
        deployed_packed,
        bench,
        cfg,
        device,
        splits,
        tr_idx,
        ev_idx,
        fp32_eval_latents,
        PACKED_3BIT_PATH.stat().st_size,
        fp32_bytes,
    )
    print(
        f"  primary top1={packed_row['primary']['top1']:.4f} "
        f"(receipt fp32={eval_receipt['metrics']['probe.top1']:.4f} -> "
        f"quantized={eval_quant_receipt['metrics']['probe.top1']:.4f})  "
        f"transfer top1={packed_row['transfer']['top1']:.4f} top5={packed_row['transfer']['top5']:.4f}  "
        f"token top1={packed_row['token_surface']['top1']:.4f} top5={packed_row['token_surface']['top5']:.4f}  "
        f"cosine={packed_row['geometry_vs_fp32']['mean_cosine']:.4f}  "
        f"nn@{NN_K}={packed_row['geometry_vs_fp32']['nn_agreement_at_k']:.4f}  "
        f"({packed_row['seconds']:.0f}s)",
        flush=True,
    )
    rows.append(packed_row)

    reproduction = {
        "fp32_recomputed": fp32_row["primary"]["top1"],
        "fp32_receipt": eval_receipt["metrics"]["probe.top1"],
        "packed_3bit_recomputed": packed_row["primary"]["top1"],
        "packed_3bit_receipt": eval_quant_receipt["metrics"]["probe.top1"],
        "packed_sha256_matches_receipt": packed_sha
        == eval_quant_receipt["artifacts"]["quantized_sha256"],
    }

    # --- (3) bits ladder: SAME 25 target-encoder tensors, FIXED width per rung, no
    # sensitivity search. `quant_receipt["bits"]` names exactly the tensors the production
    # plan quantized (all 25 at 3-bit here); `quant_receipt["fp32_tensors"]` names the
    # 1-D/small tensors it kept in fp32. Re-used unmodified at every rung.
    quantized_names = list(quant_receipt["bits"].keys())
    fp32_kept_names = list(quant_receipt["fp32_tensors"])
    print(f"\n=== bits ladder {ladder_bits} over {len(quantized_names)} tensors ===", flush=True)
    for bits in ladder_bits:
        plan = QuantPlan(
            bits=dict.fromkeys(quantized_names, bits), baseline=fp32_row["primary"]["top1"]
        )
        plan.fp32 = fp32_kept_names
        plan.fp32_bytes = fp32_bytes
        variant, stored_bytes = apply_plan(deployed_fp32, plan)
        variant = variant.to(device).eval()
        row, _ = _run_battery(
            f"ladder_{bits}bit",
            bits,
            variant,
            bench,
            cfg,
            device,
            splits,
            tr_idx,
            ev_idx,
            fp32_eval_latents,
            stored_bytes,
            fp32_bytes,
        )
        print(
            f"  {bits}-bit  primary top1={row['primary']['top1']:.4f}  "
            f"transfer top1={row['transfer']['top1']:.4f}  "
            f"token top1={row['token_surface']['top1']:.4f}  "
            f"cosine={row['geometry_vs_fp32']['mean_cosine']:.4f}  "
            f"nn@{NN_K}={row['geometry_vs_fp32']['nn_agreement_at_k']:.4f}  "
            f"ratio={row['compression_ratio']:.2f}x  ({row['seconds']:.0f}s)",
            flush=True,
        )
        rows.append(row)
        del variant
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = {
        "kind": "evidence",
        "topic": "visual-ptq-sensitivity-2026-09-06",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "cell": str(CELL),
        "inputs": {
            "train_receipt": {
                "path": str(TRAIN_RECEIPT_PATH),
                "sha256": sha256_of(TRAIN_RECEIPT_PATH),
            },
            "quant_receipt": {
                "path": str(QUANT_RECEIPT_PATH),
                "sha256": sha256_of(QUANT_RECEIPT_PATH),
            },
            "eval_receipt": {
                "path": str(EVAL_RECEIPT_PATH),
                "sha256": sha256_of(EVAL_RECEIPT_PATH),
            },
            "eval_quantized_receipt": {
                "path": str(EVAL_QUANT_RECEIPT_PATH),
                "sha256": sha256_of(EVAL_QUANT_RECEIPT_PATH),
            },
            "checkpoint_sha256": expected_ckpt_sha,
            "packed_3bit_artifact": {"path": str(PACKED_3BIT_PATH), "sha256": packed_sha},
        },
        "protocol": {
            "probe_steps": cfg.probe_steps,
            "probe_lr": cfg.probe_lr,
            "seed": cfg.seed,
            "measure_seed": MEASURE_SEED,
            "token_probe_k_per_image": TOKEN_PROBE_K,
            "nn_agreement_k": NN_K,
            "ladder_bits": list(ladder_bits),
            "quantized_tensor_names": quantized_names,
            "fp32_kept_tensor_names": fp32_kept_names,
        },
        "reproduction_check": reproduction,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {args.out}", flush=True)
    print(
        f"reproduction check: fp32 {reproduction['fp32_recomputed']:.4f} "
        f"(receipt {reproduction['fp32_receipt']:.4f})  packed-3bit "
        f"{reproduction['packed_3bit_recomputed']:.4f} (receipt "
        f"{reproduction['packed_3bit_receipt']:.4f})  sha match="
        f"{reproduction['packed_sha256_matches_receipt']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
