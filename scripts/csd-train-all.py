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
        checkpoint_every=max(1, steps // 3),
        encoder=TextEncoderConfig(dim=256, depth=4, n_heads=4, max_len=96),
        out_dir=str(state / "receipts"),
    )
    started = time.time()
    receipt = pretrain_region(cfg)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(DEFAULT_STATE))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--shard-limit", type=int, default=2, help="0 = all shards")
    ap.add_argument("--regions", default="code,compress")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

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
        if name not in REGIONS:
            print(f"  unknown region {name!r}; have {sorted(REGIONS)}", file=sys.stderr)
            continue
        try:
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
        passed = bool(beats.get("recall@1")) and bool(beats.get("recall@10"))
        summary["regions"][name] = {  # type: ignore[index]
            "receipt": receipt.get("receipt_path"),
            "untrained": receipt["untrained_baseline"],
            "held_out": receipt["held_out"],
            "beats_untrained": beats,
            "gate": "pass" if passed else "FAIL",
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
