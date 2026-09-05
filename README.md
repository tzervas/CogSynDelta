# CogSynDelta

CogSynDelta (CSD) is a small-and-capable-by-architecture composed mind: specialised
faculty regions (language, memory, vision, reasoning, ...) trained individually, then
wired together through a learned interconnect ("white matter") that reasons in a shared
latent workspace. Regions exchange latents, never discrete tokens, between encoding and
the final read-out (the latent-space reasoning invariant, DEC-47). The plan is to
quantize each trained region, then train the composed mind on the quantized regions
(`docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` §1 decision summary, DEC-01–DEC-10 for
the region taxonomy, §2.3 DEC-16 for the interconnect module, "What revision 3.4 changes"
(c) / DEC-56 for the quantize-each-then-train-composed sequencing).

## Status (2026-09-04)

**Trained, with receipts:**

| region | what it is | measured | source |
|---|---|---|---|
| `language` | language centre, code specialisation (`region_alias_of: "code"` on legacy reads) | recall@1 0.9766 | `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`, "What is actually being trained on" (receipt `code-20260902T210830Z.json`) |
| `compress` | memory faculty, consolidation head | recall@1 0.7070; graded (STS-B) spearman 0.7588 | same table; graded figure from `program/REMAINING.md` P0.9c (receipt `compress-20260903T120818Z.json`) |
| `retrieve` | memory faculty, retrieval head | recall@1 0.7480 | `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`, same table |
| `reason` | reasoning centre | retrained under the fixed contamination guard; a further re-run against the corrected clean pool is still open | `program/REMAINING.md` P0.11 (receipt `reason-20260903T123431Z.json`) |

**Memory-region (W4) variants** — `compress` and `retrieve` merged into one hippocampal
faculty per DEC-02. Four production-configuration runs; none clears all five
pre-registered gates yet. Best run (batch 1280, chunked token loss): recall@1 0.8535,
3 of 5 gates passed (a, b, d); gates (c) `c_beats_bm25` and (e) `e_retrain_gate`'s rank
clause still fail. Source: `docs/design/evidence/w4-production-runs-2026-09-03/README.md`.

**Visual region, pre-receipts for release purposes** — `visual` (`vl_latent` legacy name;
`region_alias_of: "vl_latent"` on legacy reads) is trained (probe
top-1 0.0606) but its entire pretraining corpus (`zh-plus/tiny-imagenet`, ImageNet-derived)
carries no licence anywhere in its chain and is **BLOCKING** for an open-weights release;
no subset is clean. Source: `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`, `vl_latent` section
(that document still uses the pre-rename name).

**Not trained / not built:**
- The interconnect ("white matter"). `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` is
  a design draft awaiting operator ratification: "Nothing here is applied. No config,
  script, checkpoint or program file is modified by this document" (file header).
- Whole-mind training (phase 3). `program/REMAINING.md`, "TRAINING PHASES".
- Memory-gate overlays as code — row P5′o, status `todo`. `program/REMAINING.md` §P17.

**Next step:** close the correctness debt gating everything downstream before any
interconnect work starts — `program/REMAINING.md`'s P0 table (P0.1 retrain-with-masking-fix
is `wip`; "Nothing measured before P0.1 lands is trustworthy" per its SESSION HANDOFF).

## How the work is verified

- **Receipts bound by content, not by path.** `scripts/csd-quantize.py` refuses to compare
  a quantized model against its fp32 parent unless the corpus fingerprint it rebuilds
  matches the one in the training receipt; `scripts/csd-benchmark.py`'s quantized-eval
  receipts bind both `artifacts.quantized_sha256` and `artifacts.checkpoint_sha256` to the
  bytes actually opened (both scripts' module docstrings).
- **Pre-registered gates.** The W4 memory-region gates were fixed before the batch-1280
  run that was scored against them: "a document that moves a bar after seeing the number
  it produced has stopped being a pre-registration" (`docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md`,
  "What revision 3.5 changes", DEC-68).
- **Untrained baselines on every run.** Every gate requires beating a measured, not
  assumed, random-init baseline (`program/REMAINING.md` P0.10f; the untrained-baseline
  evidence is `docs/design/evidence/w2c-untrained-baselines-2026-09-03/`).
- **Guards proven to be able to fail.** `tests/test_guards_can_fail.py` constructs the
  case each guard exists to reject and asserts the guard actually rejects it — written
  after a contamination guard was found to be vacuous by construction (same hash on both
  sides) despite passing every prior run (`program/REMAINING.md`, "SESSION HANDOFF" and
  P0.10 table).

## Repository map

Real, present directories only.

| path | what it is |
|---|---|
| `src/cogsyndelta/regions/` | per-faculty training entry points (`language`, `compress`, `retrieve`, `memory`, `visual`; `code`/`vl_latent` still work as aliases via `cogsyndelta.regions.aliases`) and shared pretraining code |
| `src/cogsyndelta/quant/` | post-training quantization (`ptq.py`, `packing.py`) |
| `src/cogsyndelta/eval/` | the benchmark battery `scripts/csd-benchmark.py` runs |
| `src/cogsyndelta/pipeline/` | the shared receipt envelope (`receipt.py`) every stage writes into |
| `src/cogsyndelta/contracts/` | region/config/component contracts and the model registry |
| `src/cogsyndelta/data/`, `util/`, `model/`, `vl/` | corpus loading, shared utilities (incl. `util/gpu_budget.py`, the gpu-pack adapter), model definitions, visual composite encoder |
| `src/cogsyndelta/poc/` | the LatentVAE proof-of-concept CLI (`cogsyndelta-poc`), separate from the region regime above |
| `src/cogsyndelta/api/` | the FastAPI server (`cogsyndelta-server`) and a Google-ADK-shaped adapter module — present as code, not a verified compliance claim |
| `src/cogsyndelta/agents/` | a self-improving-agent PoC predating the region-taxonomy design; not one of the faculties in `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` and not on the training path |
| `src/cogsyndelta/quantum/` | its own module docstring: "FUTURE FEATURE - BACKLOGGED" pending Python 3.14 ecosystem support for cirq/qiskit/pennylane |
| `src/cogsyndelta/core/`, `src/cogsyndelta/memory/` | the pre-region-taxonomy PCN-VAE-GAN and memory-persistence stack; `core/` is marked superseded and kept unimported (DEC-12, `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` §1.6) |
| `program/` | `REMAINING.md` (live task list), `KICKOFF.md` (session bootstrap), `matrix/` (train→quant matrix config), `jobs/` (gpu-pack job specs) |
| `docs/design/` | the design documents — source of truth; `docs/design/evidence/` — receipts backing specific measured claims in them |
| `scripts/` | operational entry points: training, quantizing, publishing, corpus admission, CI, docs |
| `tests/` | the pytest suite, including `tests/test_guards_can_fail.py` |
| `.githooks/`, `.github/workflows/` | git hooks and CI job definitions |

## Getting started

```bash
uv sync                 # core + dev group (default)
uv sync --group train   # add pyarrow/tokenizers for real training runs
```

Before your first commit, install the pre-commit hooks once so formatting, ruff, mypy,
shellcheck, yamllint, and commit-message checks run locally:

```bash
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

**If your clone has `core.hooksPath` set to `.githooks`** (`git config --get
core.hooksPath`; this is the case for clones set up under this project's agent-worktree
workflow, where hook directories are shared across worktrees), the commands above will
fail with "Cowardly refusing to install hooks with `core.hooksPath` set" — pre-commit
refuses to install itself over another hooks manager. In that case the tracked
`.githooks/pre-commit` (ruff check --fix + ruff format + git-secrets on staged Python) and
`.githooks/pre-push` (the full `scripts/ci_local.sh` gate) are already active on clone and
there is nothing further to install; skip straight to running `./scripts/lint.sh` below.
Only run `uv run pre-commit install` if `core.hooksPath` is unset.

```bash
./scripts/lint.sh                 # the same gate CI runs, over the whole repo
./scripts/ci_local.sh             # full local CI: lint + tests, own .venv-ci
uv run pytest tests/ -v           # the test suite alone
uv run cogsyndelta-server         # FastAPI server (src/cogsyndelta/api/server.py)
uv run cogsyndelta-benchmark      # benchmarks/run.py
uv run cogsyndelta-poc            # the LatentVAE PoC CLI (src/cogsyndelta/poc/cli.py)
```

CI runs on self-hosted Forgejo Actions runners against `.github/workflows/*.yml` — Forgejo
Actions resolves these the same way GitHub Actions does (`scripts/docs_to_wiki.py`
docstring). `git.vectorweight.com/tzervas/CogSynDelta` is the working remote; GitHub is a
backup mirror, not where CI or review happens (operator statement).

## Training, quantizing, publishing

- `scripts/csd-train-all.py` — runs the training program end to end, unattended: per-region
  pretraining, evaluation, and a receipt per run, stopping at a failed gate.
- `scripts/csd-benchmark.py` — scores a trained (or, once enabled, quantized) region across
  ranking, efficiency and representation-health metrics into the shared receipt envelope.
- `scripts/csd-quantize.py` — quantizes a region and proves the result against its own
  fp32 training receipt via a matching corpus fingerprint, never a trusted one.
- `scripts/csd-publish-checkpoint.py` — publishes one region's checkpoint, receipts and
  model card to a private Hugging Face model repo; aborts before uploading if the repo does
  not report back `private=True`.
- `program/matrix/csd-matrix.yaml` — the train → test → quantize → test matrix config (23
  cells today: language, compress, retrieve, reason, memory — `region:language` since the
  DEC-01/DEC-78 rename; the runtime cell directories under `/akula-data/csd` are matrix
  DATA and keep their pre-rename names, per the alias-layer compatibility rule). The
  harness that runs it —
  waves, gpu-pack admission, Hub publish/verify — lives in the sibling `tzervas/model-matrix`
  repo (`program/matrix/README.md`).

Eight private Hugging Face region repos exist under `tzervas`, plus the composed
`tzervas/cogsyndelta`: `cogsyndelta-region-{code,compress,retrieve,reason,memory,
stream-vae,residual}` and `cogsyndelta-vl-jepa` — the `<owner>/cogsyndelta-region-<name>`
and `<owner>/cogsyndelta-vl-jepa` naming pattern is `scripts/csd-hf-repos.py`'s own naming
section; the specific eight and their private status are an operator statement.

## Tooling repos

Per the operator's role split, tooling lives outside this repo, one repo per role, each
with its own README:

- **`tzervas/gpu-pack`** — GPU packing and VRAM admission; CSD is a client, not a
  dependency (`program/jobs/README.md`).
- **`tzervas/model-matrix`** — the train→test→quantize→test matrix harness that reads
  `program/matrix/csd-matrix.yaml` (`program/matrix/README.md`).
- **`tzervas/dataset-factory`** — licence-gated corpus ingest with provenance, feeding the
  candidates catalogued in `docs/design/DATASET-FACTORY-CATALOGUE-2026-09-03.md` (operator
  statement).
- **`tzervas/csd-autodev`** — the autonomous dev harness; CSD keeps a pointer here, not a
  specification (`docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md`, "AUTODEV LEAVES THIS
  DOCUMENT", DEC-60).

## Datasets and licences

**Stance** (operator, restated in `docs/design/DATASET-FACTORY-CATALOGUE-2026-09-03.md` §1):
architecture and code are MIT; datasets — and therefore weights — may carry non-MIT terms
that must be tracked; non-commercial (NC) terms are acceptable and change the release tier
rather than blocking it; a mirror's licence tag is not evidence about its upstream, so every
verdict rests on the upstream/primary text.

**Catalogue counts** (`docs/design/DATASET-FACTORY-CATALOGUE-2026-09-03.md` §3 Totals):
159 candidates surveyed across 7 faculties, spanning 106 distinct provenance groups —
43 `PERMISSIVE_OK`, 24 `ATTRIBUTION`, 25 `SHARE_ALIKE`, 14 `NC`, 24 `UNVERIFIED`,
11 `BLOCKING`, 18 `REFUSE`. `scripts/csd-corpus-admit.py` prints the admission checklist
for one factory-fetched dataset against a target region.

**Per-region release tiers, as currently trained** (`docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`,
"The short version" and the per-corpus verdicts): `code` and `compress` are **BLOCKING** —
`code`'s mirror licence tag does not match its unlicensed upstream (repairable to MIT by
filtering to the ~71.3% of rows in permissively-licensed repos, not yet done); `compress` is
41.8% MultiNLI-derived with three genres carrying named commercial copyright holders
(repairable to CC BY-SA 4.0 by re-deriving from `nyu-mll/multi_nli`, not yet done).
`vl_latent` is BLOCKING with no repair path found yet (see Status, above). `retrieve` is
**CC BY-NC-SA 4.0** as trained — GooAQ's non-commercial reading was accepted 2026-09-02
without a corpus change. `reason` is `PERMISSIVE_OK`/MIT as trained (`gsm8k` MIT +
`aqua_rat` Apache-2.0, both catalogued `PERMISSIVE_OK`). `memory`, the `compress`+`retrieve`
merge, inherits CC BY-NC-SA 4.0 from `retrieve` under **Rider 1** — a merge inherits its
most restrictive parent. The composed model, once it exists, carries the single strictest
term across every input — today, CC BY-NC-SA 4.0.

## Documentation

`docs/design/` is the source of truth for the architecture; everything else, including
this README, derives from it and should cite it. On push to `main`, `.github/workflows/docs.yml`'s
`wiki` job mirrors `docs/` into this repo's Forgejo wiki using `scripts/docs_to_wiki.py`.
Three reader-oriented tracks (`docs/plain/`, `docs/technical/`, `docs/foundations/`) are
planned on branch `docs/reader-tracks` and are not yet on `main` (operator statement).

## Licence

`pyproject.toml` declares `license = {text = "MIT"}` for the architecture and code. **The
root `LICENSE` file has not been updated to match — it currently reads "Proprietary
License"; this is a known, unresolved inconsistency, flagged here rather than silently
picked one way.** Datasets, and the weights trained on them, carry their own tiers and do
not inherit MIT automatically: see `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md` for the
per-region matrix and the reasoning behind it.
