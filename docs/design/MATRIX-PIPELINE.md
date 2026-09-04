# The matrix pipeline, in one page

**Status:** CSD-side adapters (this branch, `feat/matrix-config`) are implemented and
tested; the harness that actually runs a matrix (`tzervas/model-matrix`) is a separate
repo and is not part of this document's scope. This page tells you how to run a matrix
once that harness exists, how to read what it produces, and what the retention policy
prunes. The full design — every gate, every finding disposition, the wave-scheduling
arithmetic — is `/akula-data/session-backup-staging/matrix-harness/DESIGN.v2.md`; this
page is the short version for someone who wants to run or read a matrix, not implement
the harness.

## What a matrix is

One config file (`program/matrix/csd-matrix.yaml`) names a set of regions, an axis grid
per region (batch size, seed, and region-specific knobs like memory's token-aware
`terms`), and one pipeline: `train -> test -> quantize -> test-quant -> collect ->
publish -> verify`. Each point in the grid is a **cell**; the harness runs every cell,
compares them all in one table, and publishes every variant (fp32 and quantized) to a
private HF repo per region with a versioned identifier. Nothing is decided by the
harness that a receipt's `gates` block did not already decide — the harness reads
verdicts, it does not compute new ones.

## How to run one

1. `run.code.sha` in the config must name a commit both training hosts (`akula-prime`,
   `gpu5080`) actually have checked out — the harness refuses (gate G5) otherwise.
2. `model-matrix validate --config program/matrix/csd-matrix.yaml`, then `plan` (prints
   the wave schedule gpu-pack would launch, no GPU touched), then `run` — launched as a
   `systemd-run --user` unit whose `ExecStart` wraps `secret exec HF_TOKEN=...` so the
   Hub token is materialised inside the unit's own environment, never in argv or a unit
   property. The orchestrator blocks on the run's completion sentinel; it does not poll.
3. `model-matrix status --run <id>` mid-run reads `progress.json` and the current wave's
   sentinel — no GPU access needed, safe from another session.
4. On completion (or with one or more cells `failed`/`gated`), `matrix.md`/`matrix.json`
   land under `docs/design/evidence/matrix-<run_id>/`, on a branch, as a PR — the
   pipeline never pushes to `main`.

## How variants are named

Two-phase, because the corpus fingerprint is only known after training completes:

- **`cell_id`** (exists before training starts; names the local working directory):
  `<region>-<axis values>-<sha7>-<date>`, e.g. `memory-b1280-tokon-c512-s0-f48fd8a-20260904`.
- **`variant_id`** (exists once `train` completes; names the Hub branch and every tag):
  `cell_id + "-c" + <first 8 hex of the training receipt's corpus.fingerprint>`. The
  **host is deliberately not part of the identity** — two hosts running the identical
  code sha against the identical corpus produce the same variant by construction, and
  any difference between their weights is a *finding* (checked post hoc against
  `code_revision`), not a naming axis. Putting the host in the identity would let a
  same-host, different-corpus collision still overwrite a branch silently, which is the
  actual failure this scheme exists to prevent.
- fp32 and the quantized (`.ptq.pt`) artifact **share one identifier** — one variant,
  two artifacts, tagged `<variant_id>--fp32` and `<variant_id>--ptq` on the same commit.
- A branch is always cut from an empty orphan ref (`variants/base`), never from `main`'s
  head, so a variant branch holds only its own files. `verify` checks the branch's file
  set equals the cell's manifest exactly — an inherited file from another variant fails
  verification, it does not ride along quietly.

## How to read the table

One table per region per **comparability group** — cells whose `eval_key` (corpus
fingerprint, holdout size, step count, max_len) all match. A region whose cells fall
into two groups gets two tables and a banner naming what differs; they are never merged.

Within a group, cells differing **only** in `seed` are aggregated into one row,
`mean ± half-range`, labelled *"spread = init + split resample"* — CSD's
`build_splits` uses the seed for both weight init and the holdout draw, so a seed change
is not a free error bar on a fixed eval set, it is a genuine resample. That spread is the
right yardstick for "retrain this config from scratch and it lands here," which is what
promotion margins are measured against.

A delta between two cells that differ in a **non-variance** axis (batch, `terms` on vs
off, parent vs fine-tuned child) is only ever printed at **equal seed**. Where no
matched-seed pair exists, the cell reads `unpaired`, not a number — silently comparing
two different holdout draws is exactly the hazard this rule exists to close.

Two PTQ columns, always both present and always both labelled:

- `ptq_recall@1 (quant plan, in-memory; artifact not re-scored)` — `csd-quantize.py`'s
  own `quantized_metric`: a real post-quantization number, but of the plan applied to
  the in-memory model, measured *before* the packed file is written.
- `ptq_recall@1 (artifact)` — `csd-benchmark.py --quantized`'s number, once
  `stages.test-quant.enabled: true` (see `program/matrix/README.md` for the activation
  step). Reads `n/a` while disabled. `tests/test_benchmark_quantized_artifact.py`
  proves the two numbers agree within 1e-4 on today's production memory-region
  artifact — a `plan_vs_artifact_delta` over 0.005 in a later run is a finding row, not
  a failure; dequantize round-trip and CPU-vs-GPU kernel differences are real, small
  effects worth seeing rather than averaging away.

## What retention prunes

A local copy of a cell's bytes is pruned only after, per cell (never per run — one
failed cell does not strand every other cell's disk space): the Hub branch's `verify`
passed and **re-verifies live** at prune time (a cached "verified" is not trusted); the
receipt trio (train, eval, quant) is present on that branch; the local file's sha256
still equals the verified sha; the path resolves under a declared `retention.prune_roots`
entry. `finals/<region>/current` is a symlink, atomically replaced on promotion — there
is never a moment with two claimed finals, and the *old* final is only prunable after the
*new* one re-verifies. `model-matrix prune --dry-run` always runs first and its plan hash
must still match at real-run time; nothing is deleted from a stale plan.

## Where the corpus fingerprint actually lives

Every training receipt already records it — `corpus.fingerprint` (a manifest digest
over the exact shard files trained on: name and byte size per shard, hashed together;
see `_corpus_content_fingerprint` in `src/cogsyndelta/regions/pretrain.py`) — and as of
this branch, the same value also appears at the top level, `corpus_fingerprint`, on
every receipt kind (train, eval, quant, eval-quantized), matching the shape
`csd-quantize.py`'s quant receipt already used. This is what `variant_id`'s `-c<hex>`
suffix and `config_hash` are built from; a corpus drift between what a pre-flight probe
saw and what training actually used is refused (not merely warned about) before the
cell reaches `collect`/`publish`.

## Open questions this config does not resolve

`shard_limit: 2` in `program/matrix/csd-matrix.yaml`'s region defaults is the script's
own default, unconfirmed against what actually produced today's baselines (measured:
`code` used all 4 shards, `compress`/`retrieve`/`reason` used 1) — the harness's `eval_key`
comparability check (`train:$.config.holdout_pairs`, `.steps`, `.max_len`) does not cover
shard count, so two cells trained under a silently different `shard_limit` could still
land in one comparability group. Confirm before trusting a cross-region comparison that
depends on it. `promote_main` defaults to `manual` for the same reason every other
threshold in this file defaults to a probe rather than a guess: nobody has stated a
promotion margin or a minimum seed count yet, and `promote_main: auto` refuses to load
without both.
