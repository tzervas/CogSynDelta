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

import hashlib
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cogsyndelta.corpus import stable_cache_tag
from cogsyndelta.model.vl_jepa import IJEPA, JEPAConfig
from cogsyndelta.regions._checkpoint import atomic_save, load_resumable, rotate_checkpoints


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
    harness would otherwise pay it again. The cache key includes the shard list, both
    column names, the requested size and the limit, so changing any of them produces a
    different file rather than silently reusing the wrong pixels.

    The tag is a real digest, not `hash()`. `abs(hash(key)) % 10**16` was salted by
    PYTHONHASHSEED, so it named a different file in every process and this cache had
    never once been read -- every visual run re-decoded the whole corpus and the miss was
    silent, because a miss is indistinguishable from a first run.
    """
    import pyarrow.parquet as pq
    from PIL import Image

    tag = stable_cache_tag(
        {
            "shards": sorted(shards),
            "size": size,
            "limit": limit,
            "image_col": image_col,
            # `label_col` belongs in the key for the same reason `image_col` does: two
            # splits reading the same images under different label columns are different
            # artefacts, and without it the second would silently load the first's labels.
            "label_col": label_col,
        }
    )
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
    # Same failure mode as an interrupted training checkpoint, at smaller scale: a crash
    # partway through np.save would otherwise leave a truncated .npy under the cache's
    # real name, and every future run would then fail (or worse, silently mis-decode)
    # trying to load it. Write elsewhere, then rename into place once each file is
    # complete -- os.replace on the same filesystem is atomic, so a reader only ever
    # sees a complete file or none at all. Passing an open file OBJECT (rather than a
    # path string) to np.save stops it appending its own ".npy" to the temp name.
    tmp_x = cache / f".{xf.name}.tmp-{os.getpid()}"
    tmp_y = cache / f".{yf.name}.tmp-{os.getpid()}"
    try:
        with open(tmp_x, "wb") as f:
            np.save(f, x)
        with open(tmp_y, "wb") as f:
            np.save(f, y)
        tmp_x.replace(xf)  # same-filesystem rename; POSIX guarantees this is atomic
        tmp_y.replace(yf)
    finally:
        tmp_x.unlink(missing_ok=True)
        tmp_y.unlink(missing_ok=True)
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


# ---------------------------------------------------------------------------------------
# Resumable checkpointing.
#
# Same contract as regions/pretrain.py (see its module comment), same shared mechanics
# (cogsyndelta.regions._checkpoint), but two things genuinely differ here and each gets
# separate handling rather than being forced into the text harness's shape:
#
# 1. BATCH ORDER IS SAMPLED, NOT DETERMINISTIC. The text harness picks batch `step` by
#    `(step * batch_size) % len(train_pairs)` -- a pure function of `step`, so restoring
#    `step` alone reproduces the batch sequence. This harness instead draws
#    `torch.randint(..., generator=gen)` from a LOCAL `torch.Generator` that evolves
#    every call. Re-seeding `gen` from `cfg.seed` on resume would REPLAY the same batch
#    sequence from the beginning rather than continuing it, so `gen`'s state is part of
#    the checkpoint, not just `step`.
# 2. `sample_masks` (in `IJEPA.forward`, called with `generator=None` from the training
#    loop) draws from the GLOBAL default RNG for its per-step mask sampling -- a second,
#    independent randomness source from `gen` above. `_linear_probe`'s `nn.Linear` head
#    also draws its initial weights from the same global RNG, and that call happens
#    again after training for the FINAL probe metrics -- so unlike the text harness
#    (which has no randomness in its forward pass at all and is provably insensitive to
#    `rng_state`), a resumed run's global RNG state has to land in exactly the state an
#    uninterrupted run's would have, or the final probe numbers will not match.
#
# The EMA target encoder needs NO separate handling: `IJEPA.__init__` sets
# `self.target_encoder = copy.deepcopy(self.encoder)` as a plain submodule attribute, so
# PyTorch's normal submodule registration already puts its parameters and buffers into
# `model.state_dict()` under the `target_encoder.` prefix. Saving `model.state_dict()`
# therefore already carries it.
#
# The decoded-image cache (`_decode_split`'s `.npy` files) needs no place in the
# checkpoint at all: it is a deterministic, content-addressed memoisation of the RAW
# INPUT DATA keyed by (shards, size, limit, column), not training state, and any change
# to those inputs already produces a different cache file. It has its own atomicity fix
# above for the same reason `_atomic_save` exists, but it is not part of resumability.
# ---------------------------------------------------------------------------------------

_CHECKPOINT_KEEP = 3
"""Same figure and justification as regions/pretrain.py's `_CHECKPOINT_KEEP`: at the
runner's checkpoint_every=200, this buys ~600 steps of rollback headroom at under 600MB
per region -- immaterial against the measured 417GB free on /akula-data."""


def _resume_fields(cfg: VLPretrainConfig) -> dict[str, Any]:
    """The config fields that must match for a checkpoint to be a valid continuation.

    Excludes administrative fields that do not change what is trained or measured:
    `eval_every`, `checkpoint_every`, `device`, `cache_dir`, `out_dir`.
    """
    return {
        "region": cfg.region,
        "train_shards": sorted(cfg.train_shards),
        "probe_train_shards": sorted(cfg.probe_train_shards),
        "probe_eval_shards": sorted(cfg.probe_eval_shards),
        "transfer_shards": sorted(cfg.transfer_shards),
        "image_column": cfg.image_column,
        "label_column": cfg.label_column,
        "transfer_image_column": cfg.transfer_image_column,
        "transfer_label_column": cfg.transfer_label_column,
        "steps": cfg.steps,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "warmup_steps": cfg.warmup_steps,
        "grad_clip": cfg.grad_clip,
        "weight_decay": cfg.weight_decay,
        "seed": cfg.seed,
        "train_limit": cfg.train_limit,
        "probe_limit": cfg.probe_limit,
        "probe_steps": cfg.probe_steps,
        "probe_lr": cfg.probe_lr,
        "jepa": asdict(cfg.jepa),
    }


def _config_fingerprint(cfg: VLPretrainConfig) -> str:
    """Hash the resume-relevant config fields into one comparable value."""
    payload = json.dumps(_resume_fields(cfg), sort_keys=True, default=str)
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


def _checkpoint_payload(
    *,
    step: int,
    model: IJEPA,
    opt: torch.optim.Optimizer,
    gen: torch.Generator,
    jepa_cfg: JEPAConfig,
    fingerprint: str,
    fields: dict[str, Any],
    baseline: dict[str, float],
    baseline_transfer: dict[str, float] | None,
    history: list[dict],
    elapsed_s: float,
    started_at: float,
) -> dict[str, Any]:
    """Everything needed to continue training identically to an uninterrupted run.

    `step` counts COMPLETED optimiser updates (the training loop here is 1-indexed:
    `for step in range(1, cfg.steps + 1)`, so after the iteration where the loop
    variable equals `step`, exactly `step` updates are done) -- resuming means training
    `range(step + 1, cfg.steps + 1)`. This is a different arithmetic offset from
    regions/pretrain.py's 0-indexed loop even though both `step` fields mean "count of
    completed updates"; see that module's `_checkpoint_payload` docstring for why the
    two harnesses' loops are not on the same convention and are not being unified here.

    `gen_state` and `rng_state`/`cuda_rng_state` are two DIFFERENT randomness sources
    (see the module comment above) and both are required for a resumed run to match an
    uninterrupted one, not just for defence in depth as in the text harness.
    """
    return {
        "schema": "csd-vl-pretrain-checkpoint/v1",
        "step": step,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "config": asdict(jepa_cfg),
        "config_fingerprint": fingerprint,
        "config_fields": fields,
        "gen_state": gen.get_state(),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "untrained_baseline": baseline,
        "untrained_baseline_transfer": baseline_transfer,
        "history": history,
        "elapsed_s": elapsed_s,
        "started_at": started_at,
    }


def pretrain_vl_region(cfg: VLPretrainConfig) -> dict:
    """Train the visual region and return a receipt. Never gates on the loss.

    Resumable on the same contract as :func:`cogsyndelta.regions.pretrain.pretrain_region`
    -- see the module comment above for the two places this harness genuinely differs
    (sampled batch order, a second RNG stream feeding both mask sampling and the linear
    probe's head init) and why the EMA target encoder and the decoded-image cache need no
    special handling.
    """
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
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    gen = torch.Generator().manual_seed(cfg.seed)

    ckpt_dir = Path(cfg.out_dir) / f"{cfg.region}-checkpoints"
    fingerprint = _config_fingerprint(cfg)
    fields = _resume_fields(cfg)
    resume = load_resumable(ckpt_dir, fingerprint, fields)

    if resume is None:
        start_step = 1  # loop is 1-indexed; nothing completed yet.
        history: list[dict] = []
        prior_elapsed = 0.0
        started_at = time.time()
        # Baseline BEFORE a single optimiser step. Everything is judged against this,
        # and -- on a RESUMED run -- never re-measured (see below): the model is no
        # longer untrained, so re-measuring here would compare it against itself.
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
            baseline["rep_std"] = (
                model.target_encoder(probe_batch).mean(dim=1).std(dim=0).mean().item()
            )

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
        print(f"    {cfg.region}: no valid checkpoint in {ckpt_dir} -- starting fresh", flush=True)
    else:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        gen.set_state(resume["gen_state"])
        torch.set_rng_state(resume["rng_state"])
        if resume.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(resume["cuda_rng_state"])
        start_step = resume["step"] + 1
        history = resume.get("history", [])
        prior_elapsed = resume.get("elapsed_s", 0.0)
        started_at = resume.get("started_at", time.time())
        baseline = resume["untrained_baseline"]
        baseline_transfer = resume.get("untrained_baseline_transfer")
        print(
            f"    {cfg.region}: RESUMING from {resume['_path']} at step "
            f"{start_step}/{cfg.steps} (prior elapsed {prior_elapsed:.0f}s)",
            flush=True,
        )

    session_start = time.time()
    model.train()
    for step in range(start_step, cfg.steps + 1):
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
            atomic_save(
                _checkpoint_payload(
                    step=step,
                    model=model,
                    opt=opt,
                    gen=gen,
                    jepa_cfg=cfg.jepa,
                    fingerprint=fingerprint,
                    fields=fields,
                    baseline=baseline,
                    baseline_transfer=baseline_transfer,
                    history=history,
                    elapsed_s=prior_elapsed + (time.time() - session_start),
                    started_at=started_at,
                ),
                ckpt_dir / f"step-{step}.pt",
            )
            rotate_checkpoints(ckpt_dir, _CHECKPOINT_KEEP)

    elapsed_total = prior_elapsed + (time.time() - session_start)

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
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
        # Cumulative TRAINING time across every session, not wall time since
        # `started_at` -- the latter would count a crash-to-resume gap as compute.
        "seconds": round(elapsed_total, 1),
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
        # Lets an operator reading only the receipt tell a resumed run from a fresh one.
        "resumed": resume is not None,
        "resumed_from_step": (resume["step"] if resume is not None else 0),
        "beats_untrained": beats,
    }
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{cfg.region}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    receipt["receipt_path"] = str(path)
    return receipt
