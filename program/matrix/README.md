# CSD matrix config

`csd-matrix.yaml` drives the automated train -> test -> quantize -> test (then
fine-tune -> test -> quantize -> test) matrix: one config file, one run, every variant
compared against every other. The harness that reads this file — waves, gpu-pack
admission, Hub publish/verify, the comparison table — is `tzervas/model-matrix`, a
separate private repo (`tooling-lives-in-its-own-repo`: tooling does not live in the
model repo). This directory holds CSD's half of the contract: the config, its templates,
worked examples, and the stage commands the harness shells out to.

Read `docs/design/MATRIX-PIPELINE.md` first for how to run a matrix and how to read its
output table. This file covers how to add or change what is *in* the config.

## Files here

| file | what it is |
|---|---|
| `csd-matrix.yaml` | the first real matrix: code, compress, retrieve, reason, memory — 23 cells |
| `csd-matrix.template.yaml` | copy this to start a new run; every placeholder marked |
| `templates/region.yaml` | one region block, every field commented |
| `templates/finetune.yaml` | the `finetune` block shape (not yet runnable — needs adapter A4) |
| `examples/memory-terms-finetune.yaml` | worked fine-tune example (also blocked on A4) |

Full schema reference and the reasoning behind every rule below:
`/akula-data/session-backup-staging/matrix-harness/DESIGN.v2.md` (the ratified spec this
config implements). That file is not part of this repo; the rules below are its result,
restated so this directory is self-contained for someone extending the config.

## The rules a new region or axis must follow

**Budgets are keyed by the FULL axis tuple, one provenance.** `budget_axes` names every
axis a region's peak VRAM depends on (default: every axis). Every key under `budgets`
must name that exact set — a partial, superset, or ambiguous key is a config-load error,
not a first-match guess. `budgets.<key>.mib` must be `source: report_peak` — copied from
a real `GPU_PACK_PEAK_MIB=` line `cogsyndelta.util.gpu_budget.report_peak()` printed for
that *exact* axis combination — or it is `source: probe`, which the harness treats
identically to `mib: null`: the cell is measured in wave 0, never guessed from a
docstring table, a different step count, or a hand-read `nvidia-smi`. See
`csd-matrix.yaml`'s `memory` block for the one entry in this file that clears that bar
today (`batch=512,terms=on,chunk=512`, 11,700 MiB) and every other entry, which does not.

**`variance_axes` names axes whose difference is noise, not a comparison.** CSD's
`build_splits` (`src/cogsyndelta/regions/pretrain.py`) uses `cfg.seed` for *both* weight
init and the holdout draw, so two cells differing only in seed are scored on different
eval sets. `seed` belongs in `variance_axes` for every region here; the harness
aggregates such cells into one `mean ± half-range` row instead of printing a delta
between them, and never prints a delta between two cells that differ in both a designed
axis and a variance axis with no equal-variance-axis pair to compare against.

**`test-quant` starts disabled.** `csd-benchmark.py --quantized PATH --quant-receipt PATH
--train-receipt PATH` (adapter A6, this branch) scores the packed artifact itself and
writes a `kind: eval-quantized` receipt bound to both `artifacts.quantized_sha256` (the
file it actually opened) and `artifacts.checkpoint_sha256` (the fp32 parent). Until this
branch merges to `main`, `run.code.sha` in a matrix config cannot point at a commit that
has the flag, so `stages.test-quant.enabled: false` and the interim column
(`quantize:$.quantized_metric`, labelled "quant plan, in-memory; artifact not re-scored")
is what the table prints. Flipping it to `true` is a one-line change once `run.code.sha`
is updated to a commit that carries A6 — see the header comment in `csd-matrix.yaml`.

**Every receipt now carries `kind` and `started_utc`.** `kind` is a shape predicate finer
than `stage` (`train`, `eval`, `quant`, `eval-quantized`) so the harness selects a
receipt by kind, never by "whichever file under `--state` is newest" — the exact defect
DESIGN.v2's C2 names. `started_utc` is the stage's real start time (captured before any
work begins, not the end-of-run timestamp `recorded`/`recorded_utc` already carried), so
a harness classifying "started before this sentinel, decided after" has a real value to
compare against. Both fields are additive: a receipt written before this branch has
neither and is still readable (`kind` falls back to `stage` on read, see
`cogsyndelta.pipeline.receipt.adapt`).

**`corpus_fingerprint` is a top-level field on every receipt kind.** The training
receipt already computed it (`corpus.fingerprint`, a manifest digest over the shard
files actually used — see `_corpus_content_fingerprint` in `regions/pretrain.py`); it
now also appears at the top level (`receipt["corpus_fingerprint"]`), matching the shape
`csd-quantize.py`'s quant receipt already used. A variant's identity (region + code sha
+ corpus fingerprint, DESIGN.v2 §4.1) then reads one path regardless of which stage's
receipt it is holding.

**No stage inside a cell picks a receipt by its own latest-glob.** `csd-benchmark.py`
and `csd-quantize.py` both take `--train-receipt PATH` (and `csd-benchmark.py` additionally
`--quant-receipt PATH`) — adapter A7. `csd-matrix.yaml`'s `commands` templates always
pass these explicitly, rendered from the already-classified predecessor stage in the
harness's own cell manifest; the bare latest-glob only fires for a human running a
script by hand against a `--state` root with one training receipt in it.

## Adding a region

1. Copy `templates/region.yaml`'s block, rename the key, fill in `axes`.
2. Set `budget_axes` (and `budget_axes_why` if you narrow it below every axis) and
   `budgets` with every key `null`/`source: probe` unless you have a real
   `report_peak()` number for that exact tuple.
3. Set `variance_axes` — almost always `[seed]` for a text region trained through
   `pretrain_region`; a region with a different splitting mechanism must check whether
   its own seed also resamples the holdout before copying this blindly.
4. Set `primary_metric`, `eval_key`, `gates_read` from the region's own receipt shape —
   `python3 -m json.tool <a real receipt for this region>` is the fastest way to find
   the right JSON pointers.
5. `model-matrix validate --config program/matrix/csd-matrix.yaml` (once the harness
   repo has a checkout) before `plan`/`run`.

## Adding a fine-tune

Blocked on adapter A4 (`--init-from-checkpoint`) today — see `templates/finetune.yaml`'s
header. Once A4 merges, copy that template's `finetune:` block under the parent region
and add `pipeline.finetune` if the config does not already carry it. `corpus_delta: null`
(same corpus as the parent) is the only shape where `beats_parent` is a clean re-eval
today; a non-null `corpus_delta` needs H7's re-evaluate-the-parent-on-the-child's-split
pass, which the harness runs as one extra `test` invocation, not a second training run.
