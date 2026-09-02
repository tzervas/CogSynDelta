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
Batch 256 with warmup matters as much as step count: the negatives in InfoNCE ARE the
batch, so a small batch gives a weak signal regardless of how long it runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Durable, never a temp dir. /tmp filled at 0 bytes free when checkpoints landed there.
DEFAULT_STATE = Path("/akula-data/csd")
CORPUS = Path("/mnt/fleet-datasets/csd")

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

REGIONS: dict[str, tuple[list[SourceSpec], str]] = {
    "code": (
        [("region/code/codesearchnet-python/**/*.parquet", ("docstring", "code"), 0)],
        "docstring <-> function; NOT next-token over GitHub",
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
    ),
    "retrieve": (
        [
            ("region/retrieve/fiqa-pairs/train.parquet", ("query", "passage"), 0),
            ("region/retrieve/natural-questions/**/train*.parquet", ("query", "answer"), 0),
            ("region/retrieve/gooaq/**/train*.parquet", ("question", "answer"), 400_000),
        ],
        "query -> passage rank; multi-source, gooaq capped for balance",
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
    name: str, state: Path, steps: int, batch: int, shard_limit: int, dry: bool
) -> dict | None:
    """Train one region and return its receipt."""
    sources, note = REGIONS[name]
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
    print(f"    steps={steps} batch={batch}", flush=True)
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
        max_len=96,
        holdout_pairs=512,
        eval_every=max(1, steps // 6),
        warmup_steps=max(50, steps // 15),
        lr=3e-4,
        checkpoint_every=min(CHECKPOINT_EVERY, max(1, steps // 3)),
        encoder=TextEncoderConfig(dim=256, depth=4, n_heads=4, max_len=96),
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
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--shard-limit", type=int, default=2, help="0 = all shards")
    ap.add_argument("--regions", default="code,compress")
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
                    name, state, args.steps, args.batch, args.shard_limit, args.dry_run
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
