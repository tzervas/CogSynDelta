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
  - docs/design/evidence/
  - program/REMAINING.md, program/KICKOFF.md
  - src/cogsyndelta/, scripts/, tests/
last_verified: 2026-09-03
---

# The technical track

Twelve documents with the **same stems** as [`../plain/`](../plain/README.md), so twins pair by
filename. This track adds: decision ids (`DEC-nn`), programme row ids (`W*`, `E*`, `P*`, `M0*`),
open-decision titles, `file:line` anchors into `src/`, `scripts/` and `tests/`, and an explicit
statement per section of whether the thing is **built**, **design only**, or **blocked**.

This track does not restate `docs/design/` — it indexes it, anchors it to code, and reports what is
measured. Where it disagrees with `docs/design/`, the design doc wins.

## Read in order

| # | doc | what it specifies |
|---|---|---|
| 01 | [CogSynDelta: thesis, scope, and the state of the tree](01-what-csd-is.md) | The capability-per-parameter thesis, the phase structure, the §4.1 row table with real states, and the current blockers. |
| 02 | [The region taxonomy: ten verdicts, the faculty test, and the migration that has not happened](02-regions-as-faculties.md) | `DEC-01…DEC-12`, the faculty/probe/primitive distinctions, and the exact diff between designed names and the names in `config/` and `contracts/`. |
| 03 | [The white-matter interconnect: workspace, budgets, scheduler, and DEC-47](03-white-matter-and-the-latent-invariant.md) | `DEC-13…DEC-21`, `DEC-26`, `DEC-47`, `DEC-52/53`: module shape, `tokens()` contract, the two budget simplexes, scheduler behaviours, write-back, topology agreement, the latent invariant and its detector. |
| 04 | [The memory faculty and the episodic store: contracts, capacity, and arbitration](04-memory-and-the-episodic-store.md) | `DEC-02`, `DEC-49`, the five contract gaps closed by `DEC-63…DEC-66`, the overlay design `DEC-33`, and the VRAM arbiter `DEC-70`. |
| 05 | [Training: objectives, phases, budgets, and the incremental integration protocol](05-how-it-is-trained.md) | Every loss with its code path, `DEC-19`'s four phases, `DEC-24` shared embedding, `DEC-34` EMA target, `DEC-50` incremental integration, `DEC-29` parallelism, `§4.3`'s budget. |
| 06 | [Measurement: gate ladders, receipts, baselines, controls, and the results so far](06-how-we-know-it-works.md) | The gate ladder and `DEC-37`'s synergy conjunction, `DEC-30`'s two verdicts, every gate function by line, and W1 / W1d / W2c / W4 exactly as measured — including the two failures and **OD-17**. |
| 07 | [Quantisation, packing, and the deployment envelope](07-quantisation-and-packing.md) | The PTQ pipeline as implemented, `DEC-27`'s `D_sched` gate (design only), `DEC-25`/`DEC-26`/`DEC-29` envelope constraints, and the VRAM packing model. |
| 08 | [Corpora: construction, admission, fingerprints, balance rules, and licence propagation](08-data-provenance-and-licences.md) | `DEC-22/36/38/39/42/46/57/58/59/62/73`, balance rules B1–B5, the strictest-input rule `DEC-31/45`, and the absence of `scripts/csd-corpus-admit.py`. |
| 09 | [The variant matrix and the Hub as the archive](09-matrix-pipeline-and-hub-versioning.md) | The receipt envelope as the join key, `DEC-69`'s knob table as receipt fields, one private HF repo per region versioned by git revision, and the missing `schedule` stage. |
| 10 | [Repository and module map: live programme, sibling tooling, dormant history](10-the-tooling-map.md) | `DEC-75`/`DEC-60`/`DEC-76`, a one-line map of all 79 `src/` modules split live vs dormant with the import-graph evidence, and the scripts inventory. |
| 11 | [The long tracks: scale, predictive training, differential activations, ternary](11-where-it-is-going.md) | `DEC-55/56/71/72`, §6.9's ternary order, the ancestor code for each track, and the gate each must pass. |
| 12 | [Operating the programme: running jobs, security boundaries, and landing changes](12-running-it-and-contributing.md) | Entry points and their caveats, `DEC-40`'s checkpoint loader and its lint, §9.9's five adversarial boundaries B1–B5, and `DEC-61`/`DEC-77` governance. |

## Conventions in this track

**Verdict markers.** Each section opens with one of `BUILT`, `DESIGN ONLY`, or `BLOCKED (<decision
title>)`. `DESIGN ONLY` claims cite the grep that establishes the absence, tagged `[V-abs]`.

**Evidence tags.** `[V]` read at a named `file:line` · `[V*]` cited and independently reproduced ·
`[I]` inferred over `[V]` facts · `[V-abs]` a named grep returning zero hits · `[OP: file]` a verbatim
operator input.

**Cite open decisions by title, never by number.** `OD-1` names two different items (the release
licence / GooAQ contradiction, and the autodev write scope) and `OD-2` likewise. A bare number is
ambiguous.

**Sibling repositories are referenced, not re-documented** (`DEC-75`). `tzervas/gpu-pack` owns the
packer; the model-matrix harness owns the variant driver; the dataset-factory ingest owns corpus
sourcing; `csd-autodev` owns the dev loop. CSD stays the model.

## Three facts to hold before writing or reviewing anything here

1. **The taxonomy is not applied.** `config/mind/csd-regions.json` still carries `residual_mlp`,
   `stream_vae`, `code`, `retrieve` in the old `role`/`live`/`pretrain.available` shape; §1.4's JSON
   diff is headed *REVIEW ONLY, DO NOT APPLY*. `contracts/region.py` still exposes the
   `activate(stream) → [B,D]` protocol that `DEC-14` retires.
2. **The interconnect does not exist.** Everything in §2 — workspace, adapters, `a_r`, `b_r`,
   write-back, the `Schedule` — is design only, as are the `episodic_store` and overlays.
3. **The programme is blocked on OD-17.** W4 ran at its pre-registered configuration on 2026-09-03:
   gate (c) FAIL (0.200 vs BM25 0.440), gate (e) FAIL (rank ratio 1.2102 < 2.0). The gates were not
   edited after the fact, and W4n is not licensed until OD-17 is answered.

Report drift per [the track chooser](../README.md): name the file and line, the probe you ran, and
the design section or `file:line` that contradicts it.
