"""Pretrain the visual-latent region with I-JEPA, and measure it honestly.

WHY THIS FILE EXISTS
vl_jepa.py has the whole model -- ViT context encoder, EMA target encoder, predictor,
disjoint mask sampling -- and no training loop whatsoever. The region catalogue therefore
lists vl_latent as live=False while its data has been sitting on disk. This is the harness
that closes that gap, and it mirrors regions/pretrain.py so a visual region produces the
same receipts as a text one.

MEASURING THE RIGHT THING, WHICH FOR JEPA IS NOT THE LOSS
A JEPA loss curve is not evidence of learning. The degenerate solution -- map every image
to the same vector -- drives the loss toward zero and looks like a triumph. So the loss is
recorded but never gated on. Two things are:

  1. A LINEAR PROBE on frozen features against held-out labels, compared to the SAME probe
     on the untrained encoder. A random-init ViT is not a zero baseline; its features are
     genuinely somewhat linearly separable, exactly as a random-init text encoder scored
     recall@1 0.40 on CodeSearchNet from lexical overlap alone. Judging a run without that
     baseline is how you convince yourself an untrained model learned something.
  2. rep_std, the representation spread. Recorded untrained and tracked throughout. A run
     whose spread has collapsed toward zero is reported as collapsed even if its loss is
     excellent, because that is precisely the failure the loss cannot see.

The probe is trained on features from the EMA target encoder with gradients stopped at the
encoder. If gradient reached the encoder the probe would be fine-tuning, and the number
would measure the probe's capacity rather than the representation's quality.

IN-DOMAIN AND TRANSFER
Pretraining is on tiny-imagenet (100k images, natively 64x64 RGB, no resampling). The
probe runs twice: on tiny-imagenet's held-out valid split, and on cifar100, a different
dataset with different classes. The second number is the one that says whether the
representation generalises rather than memorises -- the same reason `retrieve` is scored
on out-of-domain fiqa.
"""

from __future__ import annotations

import io
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cogsyndelta.model.vl_jepa import IJEPA, JEPAConfig


@dataclass
class VLPretrainConfig:
    """Everything one visual-region run needs."""

    region: str = "vl_latent"
    train_shards: list[str] = field(default_factory=list)
    probe_train_shards: list[str] = field(default_factory=list)
    probe_eval_shards: list[str] = field(default_factory=list)
    transfer_shards: list[str] = field(default_factory=list)
    image_column: str = "image"
    label_column: str = "label"
    transfer_image_column: str = "img"
    transfer_label_column: str = "fine_label"

    steps: int = 4000
    batch_size: int = 128
    lr: float = 1.5e-4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    weight_decay: float = 0.05
    eval_every: int = 500
    checkpoint_every: int = 1000
    seed: int = 0
    device: str = "auto"

    train_limit: int = 0  # 0 = all
    probe_limit: int = 20_000
    probe_steps: int = 600
    probe_lr: float = 1e-3

    jepa: JEPAConfig = field(default_factory=JEPAConfig)
    cache_dir: str = "/akula-data/csd/vl-cache"
    out_dir: str = "/akula-data/csd/receipts"


def _resolve_device(want: str) -> torch.device:
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def _decode_split(
    shards: list[str], image_col: str, label_col: str, size: int, limit: int, cache: Path
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a parquet split to uint8 ``[N,3,size,size]`` and int64 labels.

    Cached to disk because decoding 100k JPEGs costs minutes and every rerun of this
    harness would otherwise pay it again. The cache key includes the shard list, the
    requested size and the limit, so changing any of them produces a different file
    rather than silently reusing the wrong pixels.
    """
    import pyarrow.parquet as pq
    from PIL import Image

    key = json.dumps({"shards": sorted(shards), "size": size, "limit": limit, "col": image_col})
    tag = str(abs(hash(key)) % (10**16))
    cache.mkdir(parents=True, exist_ok=True)
    xf, yf = cache / f"{tag}-x.npy", cache / f"{tag}-y.npy"
    if xf.exists() and yf.exists():
        return torch.from_numpy(np.load(xf)), torch.from_numpy(np.load(yf))

    images: list[np.ndarray] = []
    labels: list[int] = []
    for shard in shards:
        table = pq.read_table(shard, columns=[image_col, label_col])
        col_img = table.column(image_col).to_pylist()
        col_lab = table.column(label_col).to_pylist()
        for rec, lab in zip(col_img, col_lab, strict=True):
            raw = rec["bytes"] if isinstance(rec, dict) else rec
            if raw is None:
                continue
            with Image.open(io.BytesIO(raw)) as handle:
                # convert("RGB") is not optional: cifar100 and tiny-imagenet are RGB in
                # every sample checked, but a single greyscale image would otherwise
                # produce a [1,H,W] tensor and break the batch stack at an unrelated line.
                # Bound to a new name because open() yields ImageFile and convert() yields
                # Image -- rebinding one variable across both conceals the change.
                rgb = handle.convert("RGB")
                if rgb.size != (size, size):
                    # Image.BICUBIC is the pre-Pillow-10 spelling and survives only as a
                    # deprecation shim; Resampling is the supported location.
                    rgb = rgb.resize((size, size), Image.Resampling.BICUBIC)
                images.append(np.asarray(rgb, dtype=np.uint8))
            labels.append(int(lab))
            if limit and len(images) >= limit:
                break
        if limit and len(images) >= limit:
            break

    x = np.stack(images).transpose(0, 3, 1, 2)  # NHWC -> NCHW
    y = np.asarray(labels, dtype=np.int64)
    np.save(xf, x)
    np.save(yf, y)
    return torch.from_numpy(x), torch.from_numpy(y)


def _to_float(batch_u8: torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 [B,3,H,W] -> normalised float on device."""
    x = batch_u8.to(device, non_blocking=True).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def _features(
    model: IJEPA, x_u8: torch.Tensor, device: torch.device, bs: int = 256
) -> torch.Tensor:
    """Frozen latents from the EMA target encoder. no_grad is the whole point."""
    model.eval()
    out = []
    for i in range(0, x_u8.size(0), bs):
        out.append(model.encode(_to_float(x_u8[i : i + bs], device)).float().cpu())
    model.train()
    return torch.cat(out)


def _linear_probe(
    feats_tr: torch.Tensor,
    y_tr: torch.Tensor,
    feats_ev: torch.Tensor,
    y_ev: torch.Tensor,
    n_classes: int,
    device: torch.device,
    steps: int,
    lr: float,
    seed: int,
) -> dict[str, float]:
    """Fit a linear classifier on FROZEN features and score it on held-out data.

    Features are standardised first. Without it the probe spends most of its budget
    learning the scale of the feature space rather than the decision boundary, which
    understates a representation that is in fact perfectly separable.
    """
    g = torch.Generator().manual_seed(seed)
    mu, sigma = feats_tr.mean(0, keepdim=True), feats_tr.std(0, keepdim=True).clamp_min(1e-6)
    ftr = ((feats_tr - mu) / sigma).to(device)
    fev = ((feats_ev - mu) / sigma).to(device)
    ytr, yev = y_tr.to(device), y_ev.to(device)

    head = nn.Linear(ftr.size(1), n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    bs = min(1024, ftr.size(0))
    for _ in range(steps):
        idx = torch.randint(0, ftr.size(0), (bs,), generator=g).to(device)
        loss = F.cross_entropy(head(ftr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = head(fev)
        top1 = (logits.argmax(-1) == yev).float().mean().item()
        k = min(5, n_classes)
        top5 = (logits.topk(k, dim=-1).indices == yev.unsqueeze(-1)).any(-1).float().mean().item()
    return {"top1": top1, "top5": top5, "n_eval": float(yev.numel())}


def _lr_at(step: int, cfg: VLPretrainConfig) -> float:
    """Linear warmup then cosine decay, matching the text harness."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _ema_at(step: int, cfg: VLPretrainConfig) -> float:
    """Ramp EMA momentum from base to final over the run, as I-JEPA prescribes."""
    p = min(1.0, step / max(1, cfg.steps))
    return cfg.jepa.ema_base + (cfg.jepa.ema_final - cfg.jepa.ema_base) * p


def pretrain_vl_region(cfg: VLPretrainConfig) -> dict:
    """Train the visual region and return a receipt. Never gates on the loss."""
    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    cache = Path(cfg.cache_dir)
    size = cfg.jepa.image_size

    x_tr, _ = _decode_split(
        cfg.train_shards, cfg.image_column, cfg.label_column, size, cfg.train_limit, cache
    )
    px_tr, py_tr = _decode_split(
        cfg.probe_train_shards, cfg.image_column, cfg.label_column, size, cfg.probe_limit, cache
    )
    px_ev, py_ev = _decode_split(
        cfg.probe_eval_shards, cfg.image_column, cfg.label_column, size, 0, cache
    )
    n_classes = int(max(py_tr.max().item(), py_ev.max().item())) + 1

    transfer = None
    if cfg.transfer_shards:
        tx, ty = _decode_split(
            cfg.transfer_shards,
            cfg.transfer_image_column,
            cfg.transfer_label_column,
            size,
            cfg.probe_limit,
            cache,
        )
        cut = int(tx.size(0) * 0.8)
        transfer = (tx[:cut], ty[:cut], tx[cut:], ty[cut:], int(ty.max().item()) + 1)

    model = IJEPA(cfg.jepa).to(device)
    params = sum(p.numel() for p in model.parameters())

    # Baseline BEFORE a single optimiser step. Everything is judged against this.
    base_feats_tr = _features(model, px_tr, device)
    base_feats_ev = _features(model, px_ev, device)
    baseline = _linear_probe(
        base_feats_tr,
        py_tr,
        base_feats_ev,
        py_ev,
        n_classes,
        device,
        cfg.probe_steps,
        cfg.probe_lr,
        cfg.seed,
    )
    with torch.no_grad():
        probe_batch = _to_float(x_tr[: cfg.batch_size], device)
        baseline["rep_std"] = model.target_encoder(probe_batch).mean(dim=1).std(dim=0).mean().item()

    baseline_transfer = None
    if transfer:
        ttr, tytr, tev, tyev, tn = transfer
        baseline_transfer = _linear_probe(
            _features(model, ttr, device),
            tytr,
            _features(model, tev, device),
            tyev,
            tn,
            device,
            cfg.probe_steps,
            cfg.probe_lr,
            cfg.seed,
        )

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    gen = torch.Generator().manual_seed(cfg.seed)
    ckpt_dir = Path(cfg.out_dir) / f"{cfg.region}-checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    started = time.time()
    model.train()
    for step in range(1, cfg.steps + 1):
        for group in opt.param_groups:
            group["lr"] = _lr_at(step, cfg)
        idx = torch.randint(0, x_tr.size(0), (cfg.batch_size,), generator=gen)
        loss, stats = model(_to_float(x_tr[idx], device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        model.update_target(_ema_at(step, cfg))

        if step % cfg.eval_every == 0 or step == cfg.steps:
            history.append({"step": step, **stats, "lr": _lr_at(step, cfg)})
            print(
                f"    step {step:>5}  loss={stats['loss']:.4f}  rep_std={stats['rep_std']:.4f}",
                flush=True,
            )
        if step % cfg.checkpoint_every == 0 or step == cfg.steps:
            torch.save(
                {"step": step, "model": model.state_dict(), "config": asdict(cfg.jepa)},
                ckpt_dir / f"step-{step}.pt",
            )

    final = _linear_probe(
        _features(model, px_tr, device),
        py_tr,
        _features(model, px_ev, device),
        py_ev,
        n_classes,
        device,
        cfg.probe_steps,
        cfg.probe_lr,
        cfg.seed,
    )
    with torch.no_grad():
        final["rep_std"] = (
            model.target_encoder(_to_float(x_tr[: cfg.batch_size], device))
            .mean(dim=1)
            .std(dim=0)
            .mean()
            .item()
        )

    final_transfer = None
    if transfer:
        ttr, tytr, tev, tyev, tn = transfer
        final_transfer = _linear_probe(
            _features(model, ttr, device),
            tytr,
            _features(model, tev, device),
            tyev,
            tn,
            device,
            cfg.probe_steps,
            cfg.probe_lr,
            cfg.seed,
        )

    # Collapse is judged against this run's own untrained spread, not a magic constant --
    # what counts as "low" depends on the architecture and the data.
    collapse_ratio = final["rep_std"] / max(1e-9, baseline["rep_std"])
    collapsed = collapse_ratio < 0.1

    beats = {
        "probe_top1": final["top1"] > baseline["top1"],
        "probe_top5": final["top5"] > baseline["top5"],
        "not_collapsed": not collapsed,
    }
    if final_transfer and baseline_transfer:
        beats["transfer_top1"] = final_transfer["top1"] > baseline_transfer["top1"]

    receipt = {
        "region": cfg.region,
        "objective": "I-JEPA latent prediction; gated on linear probe, never on loss",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "seconds": round(time.time() - started, 1),
        "device": str(device),
        "parameters": params,
        "train_images": int(x_tr.size(0)),
        "probe_classes": n_classes,
        "config": {"jepa": asdict(cfg.jepa), "steps": cfg.steps, "batch_size": cfg.batch_size},
        "untrained_baseline": baseline,
        "held_out": final,
        "untrained_transfer": baseline_transfer,
        "transfer": final_transfer,
        "collapse_ratio": round(collapse_ratio, 4),
        "collapsed": collapsed,
        "history": history,
        "beats_untrained": beats,
    }
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{cfg.region}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(started))}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["receipt_path"] = str(path)
    return receipt
