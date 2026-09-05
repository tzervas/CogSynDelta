---
track: technical
title: The technical track — specifications, decisions, and code anchors
pairs_with: null
depends_on_foundations: []
grounded_in:
  - docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md (rev 3.5, DEC-01..DEC-77, OD-1..OD-18)
  - docs/design/LICENCE-FOR-OPEN-WEIGHTS.md
  - docs/design/CORPUS-CONTRACT.md
  - docs/design/DATASET-FACTORY-CATALOGUE-2026-09-03.md
  - docs/design/MODEL-MANIFESTS.md
  - docs/design/evidence/ (seven directories)
  - program/REMAINING.md, program/KICKOFF.md
  - src/cogsyndelta/, scripts/, tests/
code_revision: forgejo/main 4f3c098
last_verified: 2026-09-04
---

# The technical track

> **Status 2026-09-04.** This index is ratified. **The chapter files are not written yet** —
> `docs/technical/` currently holds only this README. Rows are **unlinked** until the file
> lands; a linked row means the file exists.

Thirteen documents with the **same stems** as [`../plain/`](../plain/README.md), so twins pair by
filename. This track adds: decision ids (`DEC-nn`), programme row ids (`W*`, `E*`, `P*`, `M0*`),
open-decision titles, `file:line` anchors into `src/`, `scripts/` and `tests/`, and an explicit
statement per section of whether the thing is **built**, **design only**, or **blocked**.

This track does not restate `docs/design/` — it indexes it, anchors it to code, and reports what is
measured. Where it disagrees with `docs/design/`, the design doc wins **except on code state**, where
a live probe wins even against the design doc and the doc reports the probe. Two design claims are
already stale that way, and docs 02 and 12 carry the corrections.

## Read in order

Times are at ~150 words a minute — this is reference prose with anchors you will stop to check.
**Status** is the doc-level banner, so the reader sees before opening that 03 and 04 are almost
entirely specification and that the programme is blocked.

| # | doc | status | read | what it specifies |
|---|---|---|---|---|
| 01 | CogSynDelta: thesis, scope, and the state of the tree | mixed / **BLOCKED (OD-17)** | ~13 min | The capability-per-parameter thesis, the phase structure, the §4.1 row table with real states, and the current blockers. |
| 02 | The region taxonomy: ten verdicts, the faculty test, and the migration that has not happened | DESIGN ONLY (migration) / BUILT (code as it stands) | ~13 min | `DEC-01…DEC-12`, the faculty/probe/primitive distinctions, and the exact diff between designed names and the names in `config/` and `contracts/` — including the probed **eight** files citing the retired doc, against §1.5's stale "six". |
| 03 | The white-matter interconnect: workspace, budgets, scheduler, and DEC-47 | **DESIGN ONLY** | ~15 min | `DEC-13…DEC-21`, `DEC-26`, `DEC-47`, `DEC-52/53`: module shape, `tokens()` contract, the two budget simplexes, scheduler behaviours, write-back, topology agreement, the latent invariant and its detector. |
| 04 | The memory faculty and the episodic store: contracts, capacity, and arbitration | BUILT (the merge) / DESIGN ONLY (store, overlays, gate) | ~13 min | `DEC-02`, `DEC-49`, the five contract gaps closed by `DEC-63…DEC-66`, the overlay design `DEC-33`, and the VRAM arbiter `DEC-70`. |
| 05 | Training: objectives, phases, budgets, and the incremental integration protocol | BUILT (phase 1) / DESIGN ONLY (2–4) / BLOCKED | ~15 min | Every loss with its code path, `DEC-19`'s four phases, `DEC-24` shared embedding, `DEC-34` EMA target, `DEC-50` incremental integration, `DEC-29` parallelism, `§4.3`'s budget — and the L_decorr/L_token decomposition **with its 50-step scale qualifier**. |
| 06 | Measurement: gate ladders, receipts, baselines, controls, and the results so far | **BUILT** | ~15 min | The gate ladder and `DEC-37`'s synergy conjunction, `DEC-30`'s two verdicts, every gate function by line, and W1 / W1d / W2c / W4 exactly as measured — including the two failures and **OD-17**. |
| 07 | Quantisation, packing, and the deployment envelope | BUILT (PTQ, packing) / DESIGN ONLY (`D_sched`, int8-KV, M0d) | ~13 min | The PTQ pipeline as implemented, `DEC-27`'s `D_sched` gate, `DEC-25`/`DEC-26`/`DEC-29` envelope constraints, and the VRAM packing model. |
| 08 | Corpora: construction, admission, provenance groups, and the balance rule | BUILT (fingerprints, sampling) / PROSE ONLY (NSRS) / DESIGN ONLY (factory, synthetic, moral) | ~13 min | `DEC-22/36/42/46/57/58/59/62/73`, **balance rules** B1–B5 with their enforcement points, and the absence of `scripts/csd-corpus-admit.py`. |
| 08b | Fingerprints, keyed splits, and licence propagation to a released checkpoint | BUILT (fingerprints, refusal) / DESIGN ONLY (tiers) / OPEN (OD-8, OD-18) | ~12 min | `DEC-38`/`DEC-39` identity and keyed splits, the strictest-input join `DEC-31/45`, `§7.4`'s obligation union, the release tiers, and what `csd-publish-checkpoint.py` actually refuses. |
| 09 | The variant matrix and the Hub as the archive | BUILT (receipts, publication) / DESIGN ONLY (driver, `schedule` stage) | ~11 min | The receipt envelope as the join key, `DEC-69`'s knob table as receipt fields, one private HF repo per region versioned by git revision, and the missing `schedule` stage. |
| 10 | Repository and module map: live programme, sibling tooling, dormant history | **BUILT** (markers NOT applied) | ~13 min | `DEC-75`/`DEC-60`/`DEC-76`, a one-line map of all 79 `src/` modules split live vs dormant with the import-graph evidence, the scripts inventory, and **the verdict line for every legacy doc and for `docs/adr/`**. |
| 11 | The long tracks: scale, predictive training, differential activations, ternary | **DESIGN ONLY** / gated on phase 2 | ~13 min | `DEC-55/56/71/72`, §6.9's ternary order, the ancestor code for each track, and the gate each must pass. |
| 12 | Operating the programme: running jobs, security boundaries, and landing changes | BUILT (entry points, loader) / **W0c OPEN** | ~13 min | Entry points and their caveats, `DEC-40`'s checkpoint loader **and the exact scope of its lint**, §9.9's five **boundaries** B1–B5, and `DEC-61`/`DEC-77` governance. |

## Conventions in this track

**Verdict markers.** Each section opens with one of `BUILT`, `DESIGN ONLY`, or `BLOCKED (<decision
title>)`, and each **doc** opens with a doc-level banner matching the status column above.
`DESIGN ONLY` claims cite the grep that establishes the absence, tagged `[V-abs]`.

**Evidence tags.** `[V]` read at a named `file:line` · `[V*]` cited and independently reproduced ·
`[I]` inferred over `[V]` facts · `[V-abs]` a named grep returning zero hits · `[OP: file]` a verbatim
operator input.

**Cite open decisions by title, never by number.** `OD-1` names two different items (the release
licence / GooAQ contradiction, and the autodev write scope) and `OD-2` likewise. A bare number is
ambiguous.

**Qualify every `B`.** `B` collides **four ways** inside this one track, and a bare `B2` in doc 06 and
a bare `B2` in doc 04 are different objects. Always write the qualifier:

| write | what it is | defined at | used by |
|---|---|---|---|
| `baseline B0` / `B0d` / `B0u` / `B1` / `B2` / `B2t` / `B3` | the nested baseline ladder | design §2.7.1 (line 1704) | doc 06 |
| `boundary B1`…`B5` | the five adversarial boundaries | design §9.9 (line 5606) | docs 03, 04, 12 |
| `balance rule B1`…`B5` | corpus concentration checks | `CORPUS-CONTRACT.md` Part 3 (line 1132; B1 :1147, B5 :1219) | doc 08 |
| `review finding B2` | the review-pass finding on `MEASURED_VRAM_AT_BATCH_512` | `docs/design/evidence/w4-control-arm-2026-09-03/README.md:3` | doc 05 |

**Anchor the code, and pin the revision you anchored it at.** Every doc's `code_revision` front
matter carries the base sha its `file:line` anchors were verified against (`forgejo/main 4f3c098`
today). `grounded_in` pins the design revision; without `code_revision` a single upstream edit
silently invalidated every anchor, which is the one thing a technical doc cannot survive. This
mirrors `src/cogsyndelta/regions/_receipt.py:58` (`capture_code_revision`), which binds every
receipt to the code that produced it.

**Never cite a legacy doc under `docs/`.** Rule 2 of [the chooser](../README.md) bans them as
sources; the only route to citability is a **still-true** verdict recorded in doc 10 plus a named
exemption in the chooser, and **there are currently none**. `docs/CONTRIBUTING.md` and
`docs/DEVELOPMENT_STANDARDS.md` (last touched `5818781`, 2026-01-18) were cited by doc 12 in an
earlier revision of this index and are dropped; doc 12's commit-hygiene claims are grounded in
`DEC-61`/`DEC-77` and operator memory instead. `docs/adr/` is a separate class and **is** citable
as a decision record — by full filename, never by number, because three files share the `0009-`
prefix.

**Sibling repositories are referenced, not re-documented** (`DEC-75`). `tzervas/gpu-pack` owns the
packer; the model-matrix harness owns the variant driver; the dataset-factory ingest owns corpus
sourcing; `csd-autodev` owns the dev loop. CSD stays the model.

**One standalone reference lives here outside the thirteen-chapter index.**
[`model-card-pipeline.md`](model-card-pipeline.md) documents `scripts/csd-card.py` and
`cogsyndelta.cards` (what renders from what, every refusal) — a piece of tooling, not a
design-derived chapter, so it carries no row above and no `DEC`/`OD` content of its own.

## Four facts to hold before writing or reviewing anything here

1. **The taxonomy is not applied.** `config/mind/csd-regions.json` still carries `residual_mlp`,
   `stream_vae`, `language`, `retrieve` in the old `role`/`live`/`pretrain.available` shape (the
   region-name rename landed ahead of the taxonomy rework — DEC-01/DEC-78 — but the shape itself
   is untouched); §1.4's JSON diff is headed *REVIEW ONLY, DO NOT APPLY*. `contracts/region.py`
   still exposes the `activate(stream) → [B,D]` protocol that `DEC-14` retires.
2. **The interconnect does not exist.** Everything in §2 — workspace, adapters, `a_r`, `b_r`,
   write-back, the `Schedule` — is design only, as are the `episodic_store` and overlays.
3. **The programme is blocked on OD-17.** W4 ran at its pre-registered configuration on 2026-09-03:
   gate (c) FAIL (0.200 vs BM25 0.440), gate (e) FAIL (rank ratio 1.2102 < 2.0). The gates were not
   edited after the fact, and W4n is not licensed until OD-17 is answered.
4. **Two design claims are stale, and this track reports the probe rather than the doc.**
   §1.5's *"six tracked files"* citing the retired `CSD-BRAIN-REGIONS.md`: `git grep -l` returns
   **eight** (doc 02 lists them). Row W0c's gate (iii), *"`grep -rn "torch.load" src/ scripts/`
   returns exactly one hit"*: it returns **nine** real call sites, and
   `tests/test_checkpoint_loader_lint.py` is scoped by design to six paths — so W0c is still open
   and doc 12 says so.

Report drift per [the track chooser](../README.md): name the file and line, the probe you ran, and
the design section or `file:line` that contradicts it.
