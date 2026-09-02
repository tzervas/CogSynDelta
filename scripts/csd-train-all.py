#!/usr/bin/env python3
"""Run the CSD training program end to end, unattended.

WHY THIS EXISTS
Training does not need a language model in the loop. Driving it through an agent burns
budget on work a script does better: the agent re-reads context, re-derives decisions
already made, and dies on a rate limit mid-run. This runner executes phases, evaluates,
records receipts, and stops at a failed gate -- with no LLM involved. An agent's job is
to read the receipts afterwards and decide what to change.

WHAT IT FIXES FROM THE LAST ATTEMPT
Checkpoints defaulted to a relative path, so three parallel runs wrote 184 MB files into
a session scratchpad on /tmp and filled the filesystem to 0 bytes free. Durable paths are
now explicit and default to real storage, never a temp directory.

Separately: three runs died today to session churn and a missing dependency group, and
every one of them restarted from step 0 -- pretrain.py had no resume path, and the
checkpoint interval was `steps // 3` (three checkpoints across an 8000-step run) even
where it did. Both are fixed: `pretrain_region`/`pretrain_vl_region` now resume
automatically from the newest checkpoint whose config fingerprint matches the one being
requested (model, optimizer momentum, RNG state, and the untrained baseline all carry
forward -- see `regions/pretrain.py`'s module comment), refuse outright if the fingerprint
does not match, and this runner checkpoints every `CHECKPOINT_EVERY` steps regardless of
total run length. A resumed region logs it and the receipt records it
(`resumed`/`resumed_from_step`), so an operator reading either the console output or a
receipt afterwards can tell a resumed run from a fresh one.

MEASURING THE RIGHT THING
Every run evaluates the UNTRAINED model first. On CodeSearchNet a random-init encoder
scores recall@1 0.40 from lexical overlap alone, and early training DESTROYS that before
learned structure replaces it -- recall dips to 0.03 around step 1000 and recovers to
0.94 by step 6000. Judging a run before that inversion, or without the baseline, produces
exactly the wrong conclusion. `beats_untrained` is therefore the gate, not raw recall.

SIZING, learned by measurement rather than assumed:
    code       214,813 pairs -> recall@1 0.94
    compress   277,269 pairs -> recall@1 0.26
    retrieve     4,986 pairs -> recall@1 0.006   (data-starved, not a training failure)
Batch size with warmup matters as much as step count: the negatives in InfoNCE ARE the
batch, so a small batch gives a weak signal regardless of how long it runs. Those three
numbers were measured at batch 256; DEFAULT_BATCH is now 1,280 (see its docstring), and
`lr` is derived from it rather than fixed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Durable, never a temp dir. /tmp filled at 0 bytes free when checkpoints landed there.
DEFAULT_STATE = Path("/akula-data/csd")
CORPUS = Path("/mnt/fleet-datasets/csd")

BASE_BATCH = 256
BASE_LR = 3e-4
"""The batch/LR pair every text-region receipt before 2026-09-02 was trained at.

Kept as the reference point for :func:`lr_for_batch` rather than as the batch anyone
should still use. It is what `DEFAULT_BATCH` is scaled FROM, so a future change to the
batch does not silently change the learning rate as well.
"""

DEFAULT_BATCH = 1280
"""Physical batch for the text regions, chosen by probing the card rather than fitting.

In symmetric InfoNCE over in-batch negatives, THE NEGATIVES ARE THE BATCH: each anchor
discriminates its positive against `B - 1` others, and the loss can lower-bound the
mutual information by at most `log B`. At 256 that ceiling is 5.55 nats; at 1,280 it is
7.15. This is a quality lever first and a throughput lever a distant second -- measured
`code` throughput only rises from 3,353 to 3,752 pairs/s across the same change, because
once the tokenizer is out of the loop (see regions/_tokencache.py) the GPU is already
saturated and a bigger batch does the same FLOPs per pair.

Probed on the 3090 Ti (23,028 MiB, sharing the card with a ~922 MiB KDE desktop), fp32,
`code`, max_len 96, measuring peak allocated and the whole card's peak from nvidia-smi:

    batch   peak allocated    whole card    % of card
      256        3,475 MiB     4,996 MiB        21.7%
      512        6,715 MiB     8,188 MiB        35.6%
     1024       13,163 MiB    14,888 MiB        64.7%
     1280       16,386 MiB    18,350 MiB        79.7%   <- DEFAULT_BATCH
     1536       19,611 MiB    21,738 MiB        94.4%

1,536 fits, and a memory model fitted at 256/512 predicts every row above within 0.4%
(`233 + 12.66 * B` MiB allocated). It is still the wrong default: at 94.4% of the card
there is no room for the desktop to open a window, for an eval at a larger holdout, or
for a second job to share the GPU, and the failure mode is an OOM tens of minutes into
a run. 1,280 leaves ~4.7 GiB free and gives up 0.2 nats of ceiling for it.
"""


def lr_for_batch(batch: int) -> float:
    """Scale the learning rate with the square root of the batch size.

    Args:
        batch: The physical batch size the run will use.

    Returns:
        The learning rate for that batch.

    A larger batch gives a lower-variance gradient estimate, so the step can be larger;
    `sqrt(B / B0)` is the standard rule for an adaptive optimizer (AdamW here), where
    linear scaling -- which is the rule for plain SGD -- overshoots. Holding `lr` at its
    256-batch value while raising the batch 5x would be a real change to the run and an
    invisible one, since nothing in the receipt would say the effective step size had
    fallen; deriving it here means the receipt's `lr` always matches its `batch_size`.
    """
    return BASE_LR * math.sqrt(batch / BASE_BATCH)


CHECKPOINT_EVERY = 200
"""Steps between checkpoints, for every region this runner drives (text and visual).

Was `max(1, steps // 3)` -- three checkpoints across an 8000-step run. Training died
three times in one day to session churn and a missing dependency group, and every time
restarted from step 0 because that interval is too coarse AND because pretrain.py had no
resume path at all (fixed separately; this is just the interval).

Chosen from measurements taken on this fleet today, not guessed:
  - step time: code 0.131s, retrieve 0.069s, vl_latent 0.062s, compress 0.042s
    (slowest to fastest; from today's receipts' elapsed_s / steps)
  - checkpoint write: ~0.07s for a 192MB tensor blob, measured on /akula-data's own
    NVMe (torch.save + os.replace)

At 200 steps, the worst case is losing ~26s of the slowest region's compute (`code`,
200 * 0.131s) -- comfortably "minutes, not everything" with margin to spare -- while
write overhead stays under 2% of wall time even for the fastest region (`compress`,
0.07s / (200 * 0.042s) =~ 0.8%). Checkpointing is not the IO-bound side of this trade at
any interval this coarse; the loss-on-crash side is what the interval actually trades
against.
"""


def _shards(pattern: str) -> list[str]:
    return sorted(str(p) for p in CORPUS.glob(pattern))


# A region draws from one or more sources. Each source is (glob, (left, right), cap).
#
# The cap exists for balance, not for speed. gooaq alone is 3,012,496 pairs -- 96% of
# everything available to `retrieve` -- so training uncapped would produce a gooaq model
# wearing a retrieval region's name. Capped at 400k it is 78% of a ~514k mix, comparable
# in size to `code` (455k) and `compress` (320k). Held-out eval stays on fiqa dev/test,
# which is a different domain (financial QA) and therefore measures transfer rather than
# memorisation. Raise the cap if transfer is the bottleneck; that is a measurement, not a
# guess.
SourceSpec = tuple[str, tuple[str, str], int]

# The third element of each entry is the region's DEFAULT max_len: how many tokens of
# each side the tokenizer keeps before truncating, and the width the encoder's
# positional table is sized to (see run_region -- one value feeds both). 96 for every
# region today, matching every receipt on disk; changing a default here would silently
# move future numbers, which is why `--max-len` exists as a per-invocation override
# instead.
#
# `code` is the one region where 96 is known to matter: program/REMAINING.md P0.9a
# measured 93.9% of code-side token sequences exceeding 96 tokens (mean 486, p99 3,113),
# so the encoder sees a `def` line and a line or two of body, then nothing. The reported
# recall@1 there (0.9863 at steps=4000, batch=1280) may therefore be signature matching
# rather than function-body semantics -- `--max-len 256 --regions code` is the arm that
# tests it, at whatever `--batch` the longer sequences still fit in.
REGIONS: dict[str, tuple[list[SourceSpec], str, int]] = {
    "code": (
        [("region/code/codesearchnet-python/**/*.parquet", ("docstring", "code"), 0)],
        "docstring <-> function; NOT next-token over GitHub",
        96,
    ),
    "compress": (
        # all-nli ships FOUR configs under one directory, with four different schemas AND
        # four different meanings. A `**` glob crosses them, which is how this region was
        # last trained on `pair-class` unfiltered -- an even three-way split of entailment,
        # neutral and contradiction. Two thirds of its "positives" were therefore neutral
        # or contradictory: the model was taught that "a person on a horse jumps over a
        # broken down airplane" belongs next to "a person is at a diner, ordering an
        # omelette". That is why compress scored 0.26 while code scored 0.96 -- a poisoned
        # objective, not a harder task. `pair/` is the same corpus pre-filtered to
        # entailment only, so the config is pinned explicitly and never globbed.
        [("region/compress/all-nli/pair/train*.parquet", ("anchor", "positive"), 0)],
        "neighbours stay neighbours in a short latent; all-nli entailment pairs only",
        96,
    ),
    "retrieve": (
        [
            ("region/retrieve/fiqa-pairs/train.parquet", ("query", "passage"), 0),
            ("region/retrieve/natural-questions/**/train*.parquet", ("query", "answer"), 0),
            ("region/retrieve/gooaq/**/train*.parquet", ("question", "answer"), 400_000),
        ],
        "query -> passage rank; multi-source, gooaq capped for balance",
        96,
    ),
}


# The visual region does not fit the text (left, right) pair shape: its objective is
# latent prediction over image patches, its metric is a linear probe rather than recall,
# and its data is an image struct rather than two text columns. So it gets its own entry
# and its own runner rather than being bent into REGIONS.
VL_REGIONS: dict[str, dict] = {
    "vl_latent": {
        "train": "vl/tiny-imagenet/data/train-*.parquet",
        "probe_eval": "vl/tiny-imagenet/data/valid-*.parquet",
        "columns": ("image", "label"),
        # cifar100 is a DIFFERENT dataset with different classes, so the probe on it
        # measures whether the representation transfers rather than memorises -- the same
        # reason `retrieve` is scored on out-of-domain fiqa.
        "transfer": "vl/cifar100/cifar100/test-*.parquet",
        "transfer_columns": ("img", "fine_label"),
        "note": "I-JEPA over 64x64 patches; gated on a linear probe, never on loss",
    },
}


def _schema_mismatch(shards: list[str], cols: tuple[str, str]) -> str | None:
    """Return a description if any shard lacks the requested columns, else None."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None  # cannot verify without pyarrow; the trainer will still raise
    for shard in shards:
        try:
            names = set(pq.ParquetFile(shard).schema_arrow.names)
        except Exception as exc:  # unreadable shard is itself a selection-time problem
            return f"{Path(shard).name}: unreadable ({type(exc).__name__})"
        missing = [c for c in cols if c not in names]
        if missing:
            return (
                f"{Path(shard).parent.name}/{Path(shard).name} lacks {missing}; "
                f"has {sorted(names)}. A glob spanning multiple dataset configs is "
                f"almost certainly the cause -- pin the config explicitly."
            )
    return None


def run_region(
    name: str,
    state: Path,
    steps: int,
    batch: int,
    shard_limit: int,
    dry: bool,
    bf16: bool = True,
    max_len: int | None = None,
) -> dict | None:
    """Train one region and return its receipt.

    Args:
        name: Region key in `REGIONS`.
        state: Durable state directory; receipts and checkpoints land under it.
        steps: Optimizer steps.
        batch: Physical batch size; also sets `lr` (see :func:`lr_for_batch`).
        shard_limit: Keep only this many shards of an uncapped source; 0 means all.
        dry: Resolve and print the sources, then stop without training.
        bf16: Run the forward under bf16 autocast where the device supports it. False is
            the fp32 arm -- the only way to tell "bf16 cost recall" apart from "the batch
            or the schedule did", which is a question that has to be answerable from the
            runner rather than from a one-off script nobody can rerun.
        max_len: Override this region's default max_len (see `REGIONS`). None keeps the
            region's own default. Feeds BOTH `PretrainConfig.max_len` (tokenisation
            truncation) and `TextEncoderConfig.max_len` (the positional table's width,
            and the hard ceiling `TextEncoder.forward` raises past) from the same value
            -- see the comment at the `cfg = PretrainConfig(...)` call below for why
            those two must never be set independently.

    Returns:
        The receipt, or None when the region has no usable sources or `dry` is set.
    """
    sources, note, default_max_len = REGIONS[name]
    resolved_max_len = default_max_len if max_len is None else max_len
    print(f"\n=== {name} — {note}", flush=True)

    resolved: list[SourceSpec] = []
    for glob_pat, cols, cap in sources:
        shards = _shards(glob_pat)
        if shard_limit and not cap:
            shards = shards[:shard_limit]
        if not shards:
            print(f"    source MISSING: {glob_pat}", flush=True)
            continue
        # Fail loudly if a glob spans shards with different schemas. This is what silently
        # poisoned `compress`: the failure surfaced 40 minutes into training as a KeyError
        # on shard 2, and before that it surfaced not at all -- it just trained on the
        # wrong pairs. Checking the columns exist in EVERY shard makes it a selection-time
        # error with a name, not a runtime surprise or a quiet corruption.
        bad = _schema_mismatch(shards, cols)
        if bad:
            print(f"    source REJECTED: {glob_pat}\n      {bad}", flush=True)
            continue
        resolved.append((glob_pat, cols, cap))
        print(f"    {len(shards):>2} shard(s)  {cols}  cap={cap or 'none'}  {glob_pat}", flush=True)
    if not resolved:
        print(f"    no usable sources — skipping {name}", flush=True)
        return None
    # Pair-draws, not steps, is what is held constant when the batch changes: at
    # batch 256 an 8,000-step run drew 2.05M pairs, and the same 2.05M is 1,600 steps
    # at 1,280. Printing it makes a run that quietly does 5x the work visible in the
    # first ten lines of output rather than in the wall clock an hour later.
    print(
        f"    steps={steps} batch={batch} lr={lr_for_batch(batch):.2e} "
        f"pair_draws={steps * batch:,} max_len={resolved_max_len}"
        + (f" (default {default_max_len})" if resolved_max_len != default_max_len else ""),
        flush=True,
    )
    if dry:
        return None

    # Imported here so --dry-run works without torch present.
    from cogsyndelta.regions import PretrainConfig, pretrain_region
    from cogsyndelta.regions.text_encoder import TextEncoderConfig

    # Single-source regions keep the simple path; multi-source ones are materialised by
    # the trainer via explicit shard lists per source.
    primary_glob, pair_cols, _ = resolved[0]
    shards = _shards(primary_glob)
    if shard_limit and not resolved[0][2]:
        shards = shards[:shard_limit]
    extra_sources = [
        {"shards": _shards(g), "columns": list(c), "limit": cap} for g, c, cap in resolved[1:]
    ]

    cfg = PretrainConfig(
        region=name,
        pair_columns=pair_cols,
        shards=shards,
        extra_sources=extra_sources,
        steps=steps,
        batch_size=batch,
        # ONE value feeds both PretrainConfig.max_len and TextEncoderConfig.max_len
        # below, deliberately -- they look independent but are not. PretrainConfig.max_len
        # is what `corpus_token_cache`/`evaluate` truncate to; TextEncoderConfig.max_len
        # sizes `pos_embed` and is the ceiling `TextEncoder.forward` raises past. Setting
        # PretrainConfig's higher than the encoder's would not fail at construction -- it
        # would tokenise longer sequences than the encoder can accept and raise on the
        # first batch whose truncated width exceeds the encoder's table. Setting it lower
        # would silently defeat a longer-context run: tokenisation would still cap at the
        # old length, so the encoder's extra positional capacity would never be exercised.
        max_len=resolved_max_len,
        holdout_pairs=512,
        eval_every=max(1, steps // 6),
        warmup_steps=max(50, steps // 15),
        lr=lr_for_batch(batch),
        checkpoint_every=min(CHECKPOINT_EVERY, max(1, steps // 3)),
        encoder=TextEncoderConfig(dim=256, depth=4, n_heads=4, max_len=resolved_max_len),
        bf16=bf16,
        out_dir=str(state / "receipts"),
    )
    started = time.time()
    receipt = pretrain_region(cfg)
    # pretrain_region already logged the resume decision as it happened; this repeats
    # it as a one-line summary so an operator scanning the whole run's output (or the
    # receipt itself, via `resumed`/`resumed_from_step`) does not have to scroll back.
    if receipt.get("resumed"):
        print(
            f"    resumed from step {receipt['resumed_from_step']}/{steps} "
            f"(checkpoint from a matching config)",
            flush=True,
        )
    else:
        print("    fresh run (no matching checkpoint found)", flush=True)
    b, h = receipt["untrained_baseline"], receipt["held_out"]
    print(
        f"    untrained r@1={b['recall@1']:.4f} r@10={b['recall@10']:.4f}  ->  "
        f"trained r@1={h['recall@1']:.4f} r@10={h['recall@10']:.4f} mrr={h['mrr']:.4f}",
        flush=True,
    )
    print(
        f"    beats_untrained={receipt['beats_untrained']}  "
        f"({time.time() - started:.0f}s, {receipt['parameters']:,} params)",
        flush=True,
    )
    return receipt


def run_vl_region(name: str, state: Path, steps: int, batch: int, dry: bool) -> dict | None:
    """Train the visual region. Separate path because its metric is a probe, not recall."""
    spec = VL_REGIONS[name]
    print(f"\n=== {name} — {spec['note']}", flush=True)

    train = _shards(spec["train"])
    probe_eval = _shards(spec["probe_eval"])
    transfer = _shards(spec["transfer"])
    for label, got in (("train", train), ("probe_eval", probe_eval), ("transfer", transfer)):
        print(f"    {len(got):>2} shard(s)  {label}", flush=True)
        if not got:
            print(f"    source MISSING for {label} — skipping {name}", flush=True)
            return None
    print(f"    steps={steps} batch={batch}", flush=True)
    if dry:
        return None

    from cogsyndelta.model.vl_jepa import JEPAConfig
    from cogsyndelta.regions.vl_pretrain import VLPretrainConfig, pretrain_vl_region

    cfg = VLPretrainConfig(
        region=name,
        train_shards=train,
        # The probe trains on labelled pretraining images and is scored on the valid
        # split, which pretraining never touches.
        probe_train_shards=train,
        probe_eval_shards=probe_eval,
        transfer_shards=transfer,
        image_column=spec["columns"][0],
        label_column=spec["columns"][1],
        transfer_image_column=spec["transfer_columns"][0],
        transfer_label_column=spec["transfer_columns"][1],
        steps=steps,
        batch_size=batch,
        warmup_steps=max(50, steps // 15),
        eval_every=max(1, steps // 8),
        checkpoint_every=min(CHECKPOINT_EVERY, max(1, steps // 3)),
        jepa=JEPAConfig(),
        out_dir=str(state / "receipts"),
    )
    started = time.time()
    receipt = pretrain_vl_region(cfg)
    if receipt.get("resumed"):
        print(
            f"    resumed from step {receipt['resumed_from_step']}/{steps} "
            f"(checkpoint from a matching config)",
            flush=True,
        )
    else:
        print("    fresh run (no matching checkpoint found)", flush=True)
    b, h = receipt["untrained_baseline"], receipt["held_out"]
    print(
        f"    probe  untrained top1={b['top1']:.4f} top5={b['top5']:.4f}  ->  "
        f"trained top1={h['top1']:.4f} top5={h['top5']:.4f}",
        flush=True,
    )
    if receipt.get("transfer") and receipt.get("untrained_transfer"):
        tb, th = receipt["untrained_transfer"], receipt["transfer"]
        print(
            f"    transfer (cifar100)  untrained top1={tb['top1']:.4f}  ->  "
            f"trained top1={th['top1']:.4f}",
            flush=True,
        )
    print(
        f"    rep_std {b['rep_std']:.4f} -> {h['rep_std']:.4f} "
        f"(ratio {receipt['collapse_ratio']}, collapsed={receipt['collapsed']})",
        flush=True,
    )
    print(
        f"    beats_untrained={receipt['beats_untrained']}  "
        f"({time.time() - started:.0f}s, {receipt['parameters']:,} params)",
        flush=True,
    )
    return receipt


def _require_train_deps(dry: bool) -> None:
    """Fail immediately, and by name, when the train dependency group is absent.

    tokenizers and pyarrow live in the `train` group, so `uv run` without it starts the
    run, loads the corpus list, prints a plausible banner, and only then dies on an import
    forty seconds in -- with a message naming a module rather than the invocation. An
    unattended run just records three gate failures that look like training problems.
    """
    if dry:
        return
    missing = []
    for mod in ("tokenizers", "pyarrow"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise SystemExit(
            f"missing {', '.join(missing)} -- these are in the `train` dependency group.\n"
            f"Run this as:  uv run --group train python {sys.argv[0]} ...",
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(DEFAULT_STATE))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--shard-limit", type=int, default=2, help="0 = all shards")
    ap.add_argument("--regions", default="code,compress")
    ap.add_argument(
        "--no-bf16",
        action="store_true",
        help="train the text regions in fp32 (the A/B arm for a recall change)",
    )
    ap.add_argument(
        "--max-len",
        type=int,
        default=None,
        help=(
            "override every selected region's max_len for this invocation -- tokens kept "
            "before truncation, feeding both the tokenizer cache and the encoder's "
            "positional table (see REGIONS / run_region). None keeps each region's own "
            "default (96 today, for every region). Exists for the P0.9a experiment: "
            "`code` truncates 93.9%% of its code-side sequences at 96 tokens, so its "
            "recall@1 may reflect signature matching rather than body semantics -- "
            "`--regions code --max-len 256` (at a batch the longer sequences fit in) is "
            "the arm that tests it. See program/REMAINING.md."
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _require_train_deps(args.dry_run)

    state = Path(args.state)
    if not args.dry_run:
        state.mkdir(parents=True, exist_ok=True)
        (state / "receipts").mkdir(exist_ok=True)

    print(f"CSD training program — state={state}", flush=True)
    if not CORPUS.is_dir():
        print(f"corpus not mounted at {CORPUS}; check findmnt", file=sys.stderr)
        return 2

    summary: dict[str, object] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": args.steps,
        "batch": args.batch,
        "regions": {},
    }

    gate_failures = []
    for name in [r.strip() for r in args.regions.split(",") if r.strip()]:
        if name not in REGIONS and name not in VL_REGIONS:
            known = sorted(set(REGIONS) | set(VL_REGIONS))
            print(f"  unknown region {name!r}; have {known}", file=sys.stderr)
            continue
        try:
            if name in VL_REGIONS:
                receipt = run_vl_region(name, state, args.steps, args.batch, args.dry_run)
            else:
                receipt = run_region(
                    name,
                    state,
                    args.steps,
                    args.batch,
                    args.shard_limit,
                    args.dry_run,
                    bf16=not args.no_bf16,
                    max_len=args.max_len,
                )
        except Exception as exc:
            print(f"  {name}: FAILED — {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            gate_failures.append(name)
            continue
        if receipt is None:
            continue

        beats = receipt["beats_untrained"]
        # Every declared check must pass. Reading fixed key names here would silently
        # pass a visual region, whose keys are probe_top1/not_collapsed rather than
        # recall@1 -- .get() on an absent key returns None, and None is falsy, so a
        # hardcoded recall check would fail vl_latent for the wrong reason entirely.
        passed = bool(beats) and all(bool(v) for v in beats.values())
        summary["regions"][name] = {  # type: ignore[index]
            "receipt": receipt.get("receipt_path"),
            "untrained": receipt["untrained_baseline"],
            "held_out": receipt["held_out"],
            "beats_untrained": beats,
            "gate": "pass" if passed else "FAIL",
            # Also in the per-region receipt (`resumed`/`resumed_from_step`), duplicated
            # here so the run-level summary alone tells a resumed region from a fresh
            # one without opening the region's own receipt.
            "resumed": receipt.get("resumed", False),
            "resumed_from_step": receipt.get("resumed_from_step", 0),
        }
        if not passed:
            # Do not advance a region that has not beaten its own initialisation. It has
            # rearranged, not learned, and everything measured against it is misleading.
            gate_failures.append(name)
            print(f"    GATE FAIL: {name} did not beat its untrained baseline", flush=True)

    if not args.dry_run:
        summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary["gate_failures"] = gate_failures
        out = state / f"program-run-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
        out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nsummary: {out}", flush=True)

    print(
        f"\n{len(summary['regions'])} region(s) run, {len(gate_failures)} gate failure(s)",  # type: ignore[arg-type]
        flush=True,
    )
    return 1 if gate_failures else 0


if __name__ == "__main__":
    sys.exit(main())
