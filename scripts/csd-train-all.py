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
from typing import NamedTuple

# Durable, never a temp dir. /tmp filled at 0 bytes free when checkpoints landed there.
DEFAULT_STATE = Path("/akula-data/csd")
CORPUS = Path("/mnt/fleet-datasets/csd")

LOCAL_CORPUS = Path("/bulk/csd-corpus")
"""Corpora fetched directly to a training host's local disk, not (yet) mirrored to the
shared `/mnt/fleet-datasets/csd` NFS export `CORPUS` reads from. `classify` and `reason`
are the first regions to draw from here: their four datasets were fetched and
licence-verified straight onto gpu5080's `/bulk` array (see
docs/design/LICENCE-FOR-OPEN-WEIGHTS.md), and this constant makes that a declared fact
rather than a path guessed at the call site. `REGION_CORPUS_ROOT` below says which
regions use it; every region not listed there still resolves against `CORPUS`.
"""

REGION_CORPUS_ROOT: dict[str, Path] = {"reason": LOCAL_CORPUS}
"""Per-region override of the glob root `_shards()` resolves against. Absent = `CORPUS`.

A dict rather than a field on `SourceSpec`: every OTHER region's sources already share
one root, and `SourceSpec` is a plain 3-tuple used identically for `code`/`compress`/
`retrieve`'s glob/(cols)/cap -- adding a fourth element there to carry a root only
`reason` needs would change every existing entry's shape for one region's benefit.

Consulted in exactly ONE place: `region_spec()`, which resolves it into `RegionEntry.root`
below. `run_region` used to read this dict directly, and `csd-quantize.py` /
`csd-benchmark.py` did not read it at all -- each called `spec["_shards"](g)` with no
root, so both silently resolved `reason`'s shards against the default `CORPUS` mount
instead of the `/bulk/csd-corpus` array they actually live on, finding nothing there.
Routing every consumer through `region_spec()` makes `.root` the one place this can be
read from, so a future region added to this map is not one `run_region`-only edit away
from a quantize/benchmark script quietly seeing an empty corpus again.
"""

RESERVED_FOR_COMPOSE: dict[str, str] = {
    "apps": "codeparrot/apps (10,000 rows)",
    "code_contests": "deepmind/code_contests (13,328 rows)",
}
"""DEC-23 (docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md §5.2, applied
docs/design/CORPUS-CONTRACT.md §2.4, reserved 2026-09-02): the only licence-clean
paired natural-language-problem <-> implementation source in the tree, which makes them
the only material for the `memory x language_code` cross-faculty bin. Allocation is
irreversible -- once a region trains on a row it is permanently ineligible for the
composed model -- so this must be enforced BEFORE `code`/`language_code` ever resolves a
shard under either directory, not discovered after the fact.

Keyed by the directory name `Dataset.local` (scripts/csd-corpus-expand.py) fetches each
source into -- `<region>/<name>` -- so a resolved shard path is caught by
`_refuse_reserved_shards` regardless of which root (`CORPUS`, `LOCAL_CORPUS`, or a future
`REGION_CORPUS_ROOT` override) it was resolved against.
"""


class ReservedSourceError(RuntimeError):
    """Raised when a region's resolved sources fall under a `compose`-reserved corpus.

    Fails closed: this is checked at shard-resolution time, before any slice/limit and
    before torch import, at the call site each runner uses to resolve its own sources
    (`run_region`'s source loop, `run_vl_region`'s train/probe_eval/transfer resolution,
    and `run_classify_region`'s shard resolution against `LOCAL_CORPUS`) -- so it fires
    for ANY region trained through any of those three entry points (not only `code`
    through `run_region`), for `--dry-run` as well as a real run, and before
    `pretrain_region`/`pretrain_vl_region`/`pretrain_classify_region` is ever imported or
    called. See docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md §5.6: "the gate ... is
    not 'the ledger exists'. It is: a training run seeded with one reserved row REFUSES
    TO START."
    """


class GradedSourceMissingError(RuntimeError):
    """Raised when a region declares a `GradedSpec` but its glob resolves no shards.

    Fails closed, matching `regions/compress.py`'s `compress_config` (which raises
    `FileNotFoundError` on exactly this case -- "rather than training on nothing"). A
    declared graded gate that cannot resolve its shard is not "no graded set", it is a
    broken corpus mount or a stale glob, and letting the run continue with `graded_name`
    set but `graded_shards` empty is what would have reproduced the R4 regression this
    runner exists to close: a receipt written with the gate silently absent. See
    `regions/pretrain.py`'s `_assert_graded_gate_present`, which is the second line of
    defence if a caller ever constructs a `PretrainConfig` this way directly.
    """


def _refuse_reserved_shards(region: str, glob_pat: str, shards: list[str]) -> None:
    """Raise :class:`ReservedSourceError` if any resolved shard is under a reserved corpus."""
    for shard in shards:
        for name, note in RESERVED_FOR_COMPOSE.items():
            if name in Path(shard).parts:
                raise ReservedSourceError(
                    f"region {region!r} source {glob_pat!r} resolved a shard under the "
                    f"reserved directory {name!r} ({note}, allocated `compose` -- see "
                    f"docs/design/CORPUS-CONTRACT.md §2.4). Reserved sources may only be "
                    f"consumed by the composed model. Refusing to start. shard: {shard}"
                )


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


def _shards(pattern: str, root: Path = CORPUS) -> list[str]:
    return sorted(str(p) for p in root.glob(pattern))


# A region draws from one or more sources. Each source is (glob, (left, right), cap).
#
# The cap exists for balance, not for speed. gooaq alone is 3,012,496 pairs -- 96% of
# everything available to `retrieve` -- so training uncapped would produce a gooaq model
# wearing a retrieval region's name. Capped at 400k it is 78% of a ~514k mix, comparable
# in size to `code` (455k) and `compress` (320k). The holdout is NOT a fiqa dev/test
# split: `build_splits` shuffles the concatenated three-source pool and takes a uniform
# sample of that mixture, so at ~79% GooAQ post-cap the expected fiqa content of a
# 512-pair holdout is only ~5.6 items post-dedup -- this is an in-mixture recall number,
# not a transfer measurement (see docs/design/CORPUS-CONTRACT.md Part 3, which this
# comment used to contradict). A real fiqa-only transfer evaluation exists separately in
# `cogsyndelta/regions/retrieve.py`, which trains on fiqa `train` alone and scores against
# the full BEIR-style fiqa corpus. Raise the cap if in-mixture balance is the bottleneck;
# that is a measurement, not a guess.
SourceSpec = tuple[str, tuple[str, str], int]

GradedSpec = tuple[str, tuple[str, str, str], str]
"""A region's optional graded/human-scored held-out set: (glob, (left, right, score)
columns, name-recorded-in-the-receipt). `None` for a region that declares no graded gate.

This is what went missing across the R4 regression: `compress` trained a graded gate
once by hand (receipts/compress-20260902T153612Z.json, graded_held_out.spearman 0.4956,
beats_untrained.spearman true) through regions/compress.py's own `compress_config`, which
sets `PretrainConfig.graded_shards` directly. When this runner (`run_region`, below)
became the production entrypoint for `compress` instead, its `PretrainConfig(...)` call
never set `graded_shards` at all -- the field defaults to `[]`, `pretrain_region` treats
that as "no graded set declared" (see `_prepare_graded`), and the gate silently stopped
appearing in every receipt after, with nothing raising. A `GradedSpec` entry here is what
`run_region` resolves into `PretrainConfig.graded_shards/graded_columns/graded_name`, and
`pretrain_region` now refuses to write a receipt that drops it once declared (see
`regions/pretrain.py`'s `_assert_graded_gate_present`). The glob/columns/name below match
`regions/compress.py`'s `GRADED_SHARD`/`graded_columns`/`graded_name` exactly -- both
paths must agree on what "the compress graded gate" means.
"""

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
#
# The fourth element is this region's `GradedSpec`, or `None` if it declares no graded
# gate. Only `compress` has one today.
REGIONS: dict[str, tuple[list[SourceSpec], str, int, GradedSpec | None]] = {
    "code": (
        [("region/code/codesearchnet-python/**/*.parquet", ("docstring", "code"), 0)],
        "docstring <-> function; NOT next-token over GitHub",
        96,
        None,
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
        # STS-B validation, held out entirely from training (see regions/compress.py's
        # module docstring for why): a graded human-scored set, so the region gets a
        # Spearman gate on top of the retrieval one. Must match regions/compress.py's
        # COMPRESS_ROOT-relative `GRADED_SHARD`, `graded_columns` and `graded_name`.
        (
            "region/compress/stsb/data/validation-00000-of-00001.parquet",
            ("sentence1", "sentence2", "score"),
            "stsb-validation",
        ),
    ),
    "retrieve": (
        [
            ("region/retrieve/fiqa-pairs/train.parquet", ("query", "passage"), 0),
            ("region/retrieve/natural-questions/**/train*.parquet", ("query", "answer"), 0),
            ("region/retrieve/gooaq/**/train*.parquet", ("question", "answer"), 400_000),
        ],
        "query -> passage rank; multi-source, gooaq capped for balance",
        96,
        None,
    ),
    "reason": (
        # Both sources are (question, worked-solution) pairs, so this fits the SAME
        # bi-encoder shape as every region above -- unlike `classify` (see
        # regions/classify_pretrain.py's module docstring for why THAT one does not).
        #
        # aqua_rat is capped to 4,982 via SourceSpec's `limit`, per
        # docs/design/CORPUS-CONTRACT.md Part 3:
        #   B1 (max single-source share <= 0.40): uncapped, aqua_rat is 92.88% of the
        #   staged gsm8k+aqua_rat pool (N_eff 1.15) -- "an aqua_rat model called a
        #   reasoning region", per that document's `reason` section. Capping aqua_rat to
        #   ~4,982 against gsm8k's full 7,473 gives a 60/40 mix, N_eff ~1.92 -- the same
        #   ratio that section's own worked arithmetic (7473/0.60*0.40) lands on.
        #   B4 (a cap must be a SAMPLE, not a prefix): satisfied by `load_pairs`
        #   (regions/pretrain.py) itself as of the reservoir-sampling fix -- a cap here
        #   now draws a uniform sample seeded from `cfg.seed` in one pass, recorded as
        #   `corpus.cap_sampling` in the receipt, rather than the first N rows in file
        #   order. No separate pre-sampled file needed.
        [
            ("reason/gsm8k-main/train.parquet", ("question", "answer"), 0),
            ("reason/aqua_rat-raw/train.parquet", ("question", "rationale"), 4982),
        ],
        "question <-> worked derivation; retrieval of the matching solution, NOT step "
        "generation. Corpus at /bulk/csd-corpus, not the shared mount -- see "
        "REGION_CORPUS_ROOT",
        # 256, not the fleet default 96: gsm8k's answer side truncates at 41.2% of rows
        # at 96 tokens (mean 95.3, p90 153) and aqua_rat's rationale at 21.3% -- both mid
        # derivation, which makes the pair meaningless in a way truncating a code body
        # does not (measured with tokstats against the GPT-2 tokenizer this region
        # trains with; see the reason region's training receipt for the same numbers).
        # At 256 both fall under 1.1% truncated.
        256,
        None,
    ),
}


class RegionEntry(NamedTuple):
    """A `REGIONS[name]` value, typed and named, plus the corpus root it resolves against.

    The first four fields are declared in the SAME order as a `REGIONS` value's tuple
    elements (`sources, note, default_max_len, graded`), so `RegionEntry(*REGIONS[name],
    root=...)` reads as a naming layer over that existing shape, not a new one. `root` is
    the fifth field, appended rather than interleaved, so `RegionEntry(*REGIONS[name])`
    -- without a root -- would still raise a clear arity error instead of silently
    shifting `graded` into `root`'s position.

    `root` is NOT itself part of `REGIONS[name]`'s shape -- it comes from
    `REGION_CORPUS_ROOT`, resolved once here by :func:`region_spec` -- but it belongs on
    this type because every consumer that needs `sources` also needs to know which root
    to resolve them against, and `region_spec()` is the one place that used to answer the
    first question but not the second.
    """

    sources: list[SourceSpec]
    note: str
    default_max_len: int
    graded: GradedSpec | None
    root: Path


def region_spec(name: str) -> RegionEntry:
    """Resolve one `REGIONS` entry through its one typed shape.

    `REGIONS` stays the single source of truth -- this only names its shape once. Every
    consumer (this module's own `run_region`, and every script that loads `REGIONS` out
    of this module -- `scripts/csd-quantize.py`, `scripts/csd-benchmark.py`) must go
    through here instead of unpacking `REGIONS[name]` directly.

    Why this exists: commit 0786a77 added the graded-gate field to every `REGIONS`
    value, turning it from a 3-tuple into a 4-tuple, to fix a DIFFERENT regression (the
    graded gate silently going missing -- see `GradedSpec`'s docstring). That change was
    correct and well-tested for what it touched, but `csd-quantize.py` and
    `csd-benchmark.py` each unpack `REGIONS[region]` themselves with their own fixed
    arity (`sources, _note = ...`), so the new field broke both of them with a bare
    `ValueError: too many values to unpack` before either script did anything useful --
    and nothing caught it, because neither script had a test that actually imported and
    called into `csd-train-all.py`'s `REGIONS`. Centralising the unpack here means the
    shape can only break in ONE place, and that place is tested (see
    tests/test_region_spec_consumers.py) -- including a regression test that constructs
    the OLD shape and asserts this function rejects it rather than silently misreading
    it.

    Also resolves `REGION_CORPUS_ROOT` into `RegionEntry.root` (`CORPUS` for every region
    absent from that map), so every consumer agrees on which mount a region's shards live
    under. Before this, `run_region` read `REGION_CORPUS_ROOT` directly while
    `csd-quantize.py` and `csd-benchmark.py` called `_shards(glob)` with no root at all --
    both defaulting to `CORPUS` regardless of the map, so `reason` (the one region in
    `REGION_CORPUS_ROOT`, mounted at `/bulk/csd-corpus`) resolved zero shards through
    either script even though training itself found them fine.

    Raises:
        KeyError: `name` is not a key in `REGIONS`.
        ValueError: `REGIONS[name]` is not the current 4-tuple
            `(sources, note, default_max_len, graded)` shape -- e.g. a stale 2-tuple or
            3-tuple from before the graded-gate field existed.
    """
    if name not in REGIONS:
        raise KeyError(f"no REGIONS entry for {name!r}; known regions: {sorted(REGIONS)}")
    entry = REGIONS[name]
    if not isinstance(entry, tuple) or len(entry) != 4:
        got = len(entry) if isinstance(entry, tuple) else type(entry).__name__
        raise ValueError(
            f"REGIONS[{name!r}] is not the current (sources, note, default_max_len, "
            f"graded) 4-tuple shape (got {got}): {entry!r}. Every REGIONS entry must "
            f"carry the graded-gate field -- None for a region that declares no graded "
            f"gate -- see GradedSpec's docstring for what silently drops if it is "
            f"missing instead of raising here."
        )
    return RegionEntry(*entry, root=REGION_CORPUS_ROOT.get(name, CORPUS))


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
        # measures whether the representation transfers rather than memorises. Unlike
        # `retrieve`'s holdout -- which is a uniform sample of an in-mixture pool, NOT an
        # out-of-domain fiqa split; see the `SourceSpec` comment above and
        # docs/design/CORPUS-CONTRACT.md Part 3 -- this probe really does train on one
        # shard (`train`) and evaluate on an entirely separate one (`transfer`), so it is
        # actually out-of-domain.
        "transfer": "vl/cifar100/cifar100/test-*.parquet",
        "transfer_columns": ("img", "fine_label"),
        "note": "I-JEPA over 64x64 patches; gated on a linear probe, never on loss",
    },
}


# The classify corpora are (text, LABEL) rows, not (anchor, positive) text pairs -- see
# regions/classify_pretrain.py's module docstring for the full reasoning, including why
# banking77 and go_emotions are two SEPARATE regions here rather than one `classify`
# region with two sources: their label spaces are incompatible (77-way single-label vs
# 28-way multi-label), so there is no one head shape, loss or metric that fits both. So
# this gets its own spec dict and runner, same as VL_REGIONS above, rather than a fourth
# column bolted onto SourceSpec for one region family's benefit.
#
# Both corpora live under LOCAL_CORPUS (`/bulk/csd-corpus`), not the shared `CORPUS`
# mount -- same reason `reason` does (see REGION_CORPUS_ROOT above).
CLASSIFY_REGIONS: dict[str, dict] = {
    "classify_banking77": {
        "shards": "classify/banking77/train.parquet",
        "text_column": "text",
        "label_column": "category",
        "multi_label": False,
        "label_names_from_metadata": False,
        "max_len": 96,
        # 1,500 rather than the text-region default 512: 77 roughly-balanced classes
        # (CORPUS-CONTRACT.md B5: 5.34:1 max:min) need enough held-out rows per class for
        # macro-F1 to mean something -- 512 gives ~6-7/class, this gives ~19/class.
        "holdout_rows": 1500,
        "note": (
            "77 banking intents, single-label, softmax cross-entropy head. Chance top1 "
            "~1.3% (1/77) -- see the receipt's untrained_baseline for the measured "
            "number, not this estimate."
        ),
    },
    "classify_go_emotions": {
        "shards": "classify/go_emotions-simplified/train.parquet",
        "text_column": "text",
        "label_column": "labels",
        "multi_label": True,
        "label_names_from_metadata": True,
        "max_len": 96,
        # 3,000 rather than 512: the rarest label ("grief") is 77 of 43,410 rows
        # (0.18%). At 512 held out, its expected count is under 1 -- too few to compute
        # an average precision that means anything. At 3,000 it is ~5-6, still thin but
        # reported (and excluded from the macro average below that threshold) rather
        # than silently unreliable; see ClassifyPretrainConfig.min_holdout_positives.
        "holdout_rows": 3000,
        "note": (
            "28 emotions, multi-label, per-label BCE head. Gated on macro average "
            "precision, NOT accuracy: the corpus is 32.8% one label and 184.7:1 "
            "max:min (CORPUS-CONTRACT.md B5 fail on both), so per-label binary "
            "accuracy's chance floor is ~99.8% for the rarest label -- see "
            "regions/classify_pretrain.py's module docstring."
        ),
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
        dry: Resolve every source (including the region's graded set, if declared) and
            print the resolved PretrainConfig fields as a plan, then stop without
            starting a run or importing torch.
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
    sources, note, default_max_len, graded_spec, corpus_root = region_spec(name)
    resolved_max_len = default_max_len if max_len is None else max_len
    print(f"\n=== {name} — {note}", flush=True)
    if corpus_root != CORPUS:
        print(f"    corpus root: {corpus_root} (not the shared {CORPUS})", flush=True)

    resolved: list[SourceSpec] = []
    for glob_pat, cols, cap in sources:
        shards = _shards(glob_pat, corpus_root)
        # Fail closed BEFORE anything else looks at these shards: a `compose`-reserved
        # source (see RESERVED_FOR_COMPOSE) must never be trained by any region, so this
        # runs ahead of the shard-limit slice, the MISSING check and dry-run's early
        # return -- every path out of this loop for this source is downstream of it.
        _refuse_reserved_shards(name, glob_pat, shards)
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

    # Single-source regions keep the simple path; multi-source ones are materialised by
    # the trainer via explicit shard lists per source.
    primary_glob, pair_cols, _ = resolved[0]
    shards = _shards(primary_glob, corpus_root)
    if shard_limit and not resolved[0][2]:
        shards = shards[:shard_limit]
    extra_sources = [
        {"shards": _shards(g, corpus_root), "columns": list(c), "limit": cap}
        for g, c, cap in resolved[1:]
    ]

    # Resolve the region's graded/human-scored held-out set, if it declares one (see
    # `GradedSpec`). This is what R4 lost: `compress` declares one, this runner used to
    # never resolve it, `PretrainConfig.graded_shards` stayed `[]`, and the Spearman gate
    # silently stopped appearing in every receipt after the one hand-run at 11:36 on
    # 2026-09-02 (receipts/compress-20260902T153612Z.json). A region with no `graded_spec`
    # (every region but `compress`, today) resolves to `graded_shards=None`, which
    # `pretrain_region` correctly reads as "no graded gate declared" -- see
    # `regions/pretrain.py`'s `_prepare_graded`/`_assert_graded_gate_present`. A region
    # WITH a `graded_spec` whose glob resolves nothing is a different case entirely -- a
    # declared gate that failed to resolve, not an undeclared one -- and is refused
    # outright below, the same way `regions/compress.py`'s `compress_config` raises
    # `FileNotFoundError` "rather than training on nothing" for the identical shard.
    graded_shards: list[str] | None = None
    graded_cols: tuple[str, str, str] | None = None
    graded_name: str | None = None
    if graded_spec is not None:
        graded_glob, graded_cols, graded_name = graded_spec
        graded_shards = _shards(graded_glob, corpus_root)
        if not graded_shards:
            raise GradedSourceMissingError(
                f"region {name!r} declares a graded gate ({graded_name!r}) at "
                f"{graded_glob!r} (root {corpus_root}) but that glob resolved no shards "
                f"-- check the NFS mount, an upstream dataset directory rename, or a typo "
                f"in the glob. Refusing to start rather than writing a receipt with no "
                f"graded_held_out, exactly the R4 regression this runner exists to close."
            )
        print(
            f"    graded  {len(graded_shards):>2} shard(s)  {graded_cols}  "
            f"name={graded_name!r}  {graded_glob}",
            flush=True,
        )

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
        # The resolved PretrainConfig fields, printed without importing PretrainConfig
        # itself (that import pulls in torch/tokenizers -- see the comment below) and
        # without starting a run. `graded_shards`/`graded_columns`/`graded_name` are
        # included deliberately when the region declares a graded gate: a dry run is how
        # `compress`'s graded gate is verified wired without spending GPU time, which is
        # exactly the check R4 had no way to make cheaply. For a region that declares NO
        # graded gate (`graded_spec is None`, i.e. `graded_shards is None` here -- the
        # raise above means a declared-but-unresolved glob never reaches this point with
        # an empty list), the three keys are omitted rather than filled with the STS-B
        # default: printing `graded_columns: ["sentence1","sentence2","score"]` for
        # `code`/`retrieve`/`reason`, which have no graded set at all, would misread as
        # those columns being attached to that region.
        plan = {
            "region": name,
            "pair_columns": list(pair_cols),
            "shards": shards,
            "extra_sources": extra_sources,
            "steps": steps,
            "batch_size": batch,
            "lr": lr_for_batch(batch),
            "max_len": resolved_max_len,
            "holdout_pairs": 512,
            "graded_shards": graded_shards,
            "graded_columns": list(graded_cols) if graded_cols is not None else None,
            "graded_name": graded_name,
        }
        print("    resolved PretrainConfig (dry run, no training started):", flush=True)
        print(json.dumps(plan, indent=2), flush=True)
        return None

    # Imported here so --dry-run works without torch present.
    from cogsyndelta.regions import PretrainConfig, pretrain_region
    from cogsyndelta.regions.text_encoder import TextEncoderConfig

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
        # `graded_shards`/`graded_cols`/`graded_name` are `None` above only when
        # `graded_spec is None` (no graded gate declared); a declared-but-unresolved
        # glob already raised `GradedSourceMissingError` before this point, so `None`
        # here can only mean "this region declares no graded gate" -- fall back to
        # `PretrainConfig`'s own no-gate defaults rather than pass `None` into fields
        # typed `list[str]`/`tuple[str, str, str]`/`str`.
        graded_shards=graded_shards or [],
        graded_columns=graded_cols or ("sentence1", "sentence2", "score"),
        graded_name=graded_name or "",
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
    # Same fail-closed requirement as `run_region` (see RESERVED_FOR_COMPOSE and
    # `ReservedSourceError`): a reserved shard must never train ANY region, and this VL
    # path resolves its own shards independently of `run_region`'s loop, so it needs its
    # own call, ahead of the MISSING check and dry-run's early return below.
    for label, glob_pat, got in (
        ("train", spec["train"], train),
        ("probe_eval", spec["probe_eval"], probe_eval),
        ("transfer", spec["transfer"], transfer),
    ):
        _refuse_reserved_shards(name, glob_pat, got)
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
        cache_dir=str(state / "vl-cache"),
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


def run_classify_region(
    name: str, state: Path, steps: int, batch: int, dry: bool, bf16: bool = True
) -> dict | None:
    """Train one classify specialist and return its receipt.

    Separate path from `run_region`/`run_vl_region`: the corpus is (text, label) rows,
    not (anchor, positive) pairs or an image struct, so it needs its own loss (softmax
    cross-entropy or per-label BCE, per the spec's `multi_label`) and its own metrics
    (top1/top5/macro-F1, or macro/micro average precision). See
    `regions/classify_pretrain.py`'s module docstring for the full reasoning.
    """
    spec = CLASSIFY_REGIONS[name]
    print(f"\n=== {name} — {spec['note']}", flush=True)
    print(f"    corpus root: {LOCAL_CORPUS} (not the shared {CORPUS})", flush=True)

    shards = _shards(spec["shards"], LOCAL_CORPUS)
    # Same fail-closed requirement as `run_region` (see RESERVED_FOR_COMPOSE and
    # `ReservedSourceError`) -- doubly so here, since `LOCAL_CORPUS` (/bulk/csd-corpus) is
    # the exact root `apps` and `code_contests` were fetched under, so a careless future
    # `CLASSIFY_REGIONS` glob is one wildcard away from resolving straight into them.
    _refuse_reserved_shards(name, spec["shards"], shards)
    print(
        f"    {len(shards):>2} shard(s)  text={spec['text_column']!r} "
        f"label={spec['label_column']!r} multi_label={spec['multi_label']}  {spec['shards']}",
        flush=True,
    )
    if not shards:
        print(f"    source MISSING for {name} — skipping", flush=True)
        return None
    print(
        f"    steps={steps} batch={batch} lr={lr_for_batch(batch):.2e} "
        f"max_len={spec['max_len']} holdout_rows={spec['holdout_rows']}",
        flush=True,
    )
    if dry:
        return None

    # Imported here so --dry-run works without torch present.
    from cogsyndelta.regions.classify_pretrain import (
        ClassifyPretrainConfig,
        pretrain_classify_region,
    )
    from cogsyndelta.regions.text_encoder import TextEncoderConfig

    cfg = ClassifyPretrainConfig(
        region=name,
        shards=shards,
        text_column=spec["text_column"],
        label_column=spec["label_column"],
        multi_label=spec["multi_label"],
        label_names_from_metadata=spec["label_names_from_metadata"],
        steps=steps,
        batch_size=batch,
        max_len=spec["max_len"],
        holdout_rows=spec["holdout_rows"],
        eval_every=max(1, steps // 6),
        warmup_steps=max(50, steps // 15),
        lr=lr_for_batch(batch),
        checkpoint_every=min(CHECKPOINT_EVERY, max(1, steps // 3)),
        encoder=TextEncoderConfig(dim=256, depth=4, n_heads=4, max_len=spec["max_len"]),
        bf16=bf16,
        out_dir=str(state / "receipts"),
    )
    started = time.time()
    receipt = pretrain_classify_region(cfg)
    if receipt.get("resumed"):
        print(
            f"    resumed from step {receipt['resumed_from_step']}/{steps} "
            f"(checkpoint from a matching config)",
            flush=True,
        )
    else:
        print("    fresh run (no matching checkpoint found)", flush=True)

    b, h = receipt["untrained_baseline"], receipt["held_out"]
    if spec["multi_label"]:
        print(
            f"    untrained macro_ap={b['macro_ap']:.4f} (chance~={b['macro_ap_chance']:.4f}) "
            f"micro_ap={b['micro_ap']:.4f}  ->  "
            f"trained macro_ap={h['macro_ap']:.4f} (chance~={h['macro_ap_chance']:.4f}) "
            f"micro_ap={h['micro_ap']:.4f}",
            flush=True,
        )
    else:
        print(
            f"    untrained top1={b['top1']:.4f} (chance={b['top1_chance']:.4f}) "
            f"macro_f1={b['macro_f1']:.4f}  ->  "
            f"trained top1={h['top1']:.4f} (chance={h['top1_chance']:.4f}) "
            f"macro_f1={h['macro_f1']:.4f}",
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
        if name not in REGIONS and name not in VL_REGIONS and name not in CLASSIFY_REGIONS:
            known = sorted(set(REGIONS) | set(VL_REGIONS) | set(CLASSIFY_REGIONS))
            print(f"  unknown region {name!r}; have {known}", file=sys.stderr)
            continue
        try:
            if name in VL_REGIONS:
                receipt = run_vl_region(name, state, args.steps, args.batch, args.dry_run)
            elif name in CLASSIFY_REGIONS:
                receipt = run_classify_region(
                    name, state, args.steps, args.batch, args.dry_run, bf16=not args.no_bf16
                )
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
