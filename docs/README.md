---
track: chooser
title: CogSynDelta reader documentation — pick a track
pairs_with: null
depends_on_foundations: []
grounded_in:
  - docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md (rev 3.5)
  - docs/design/LICENCE-FOR-OPEN-WEIGHTS.md
  - docs/design/CORPUS-CONTRACT.md
  - docs/design/DATASET-FACTORY-CATALOGUE-2026-09-03.md
  - docs/design/MODEL-MANIFESTS.md
  - program/REMAINING.md, program/KICKOFF.md
  - src/cogsyndelta/, scripts/
last_verified: 2026-09-03
---

# CogSynDelta documentation

Three tracks. Same programme, three different reading contracts.

| track | who it is for | what it promises |
|---|---|---|
| [**plain/**](plain/README.md) | anyone who wants to understand what CSD is and why, without a maths background | plain language, no unexplained jargon, every hard idea handed off to a foundations unit |
| [**technical/**](technical/README.md) | engineers and reviewers working on or auditing the code | precise specifications, decision ids, `file:line` anchors, and an explicit line between what is built and what is designed |
| [**foundations/**](foundations/README.md) | plain-track readers who hit a prerequisite they do not have | one idea per unit — the concept, why CSD needs it, the real code path, and a runnable CPU "try it" |

## How the tracks fit together

**Plain and technical are twins.** They carry **identical file stems**, so `plain/05-how-it-is-trained.md`
and `technical/05-how-it-is-trained.md` are the same subject at two depths. Each names the other in
its `pairs_with` front matter. Read one, or read both — but the plain doc never contradicts its twin,
it only says less.

**Foundations is a prerequisite library, not a third narrative.** Units are numbered in dependency
order and referenced **by stem** from a plain doc's `depends_on_foundations` list. A plain doc assumes
its listed units and nothing more. If a plain doc needs an idea, either the idea is explained inline
or there is a foundations unit for it — there is no third option.

**Foundations units are not a textbook.** Each is one idea, tightly scoped, ending in a small piece of
runnable Python that finishes on a CPU in under a minute. They teach against the real implementation:
every unit points at a `file:line` in this repository, not at a generic example.

## Where truth lives

`docs/design/` is the **source of truth**. These tracks derive from it.

- `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` (**rev 3.5**) carries `DEC-01…DEC-77` and
  `OD-1…OD-18` and outranks every other document in the repository.
- `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`, `CORPUS-CONTRACT.md`,
  `DATASET-FACTORY-CATALOGUE-2026-09-03.md`, `MODEL-MANIFESTS.md` and the `evidence/` directories
  carry the licence, corpus, manifest and sha-bound receipt detail.
- `program/REMAINING.md` and `program/KICKOFF.md` carry the plan.
- `src/cogsyndelta/` and `scripts/` are the real implementation. **A live probe beats a document.**

Two rules follow, and they are not negotiable:

1. **Where a track doc conflicts with `docs/design/`, the design doc wins and the track doc is a bug.**
2. **Many legacy files directly under `docs/`** — `ARCHITECTURE.md`, `API_REFERENCE.md`,
   `CODE_DOCUMENTATION.md`, `CONFIGURATION.md`, `HANDOFF-ARCHITECTURE.md`, `GPU_COMPATIBILITY.md`,
   the compression-fidelity and VSA essays, and others — are **history or aspiration**, not fact.
   Read them as a record of what was once intended. Never cite them for a claim.

## Design is not implementation

The single most common way to misread this programme is to assume that because the design doc
specifies something, it exists. Much of it does not, yet. Every track doc therefore marks its claims:

- **built** — there is code, and usually a receipt;
- **design only** — specified in `docs/design/`, absent from `src/`;
- **blocked** — waiting on an operator decision, named by title.

As of `last_verified` above: the white-matter interconnect, the episodic store, memory-gate overlays
and the `Schedule` are all **design only**; the region renames in §1.4 are marked *REVIEW ONLY, DO NOT
APPLY* and have **not** been applied to `config/mind/csd-regions.json`; and the programme is
**blocked on OD-17** (W4 failed two of five pre-registered clauses at its pre-registered
configuration). A doc that describes any of these as finished is wrong.

## Front matter convention

Every doc in every track begins with the same block:

```yaml
---
track: plain | technical | foundations
title: <the doc's own title>
pairs_with: <twin stem, or null>
depends_on_foundations: [<unit stems>]
grounded_in:
  - <design file + revision, and/or code paths this doc's claims rest on>
last_verified: 2026-09-03
---
```

`grounded_in` is the audit trail: it names what a reviewer must re-read to re-verify the doc.
`last_verified` is the date someone actually did that. A doc whose `last_verified` is older than the
design revision it cites is **stale until re-checked**.

## Evidence tags

The tracks reuse the design doc's tag vocabulary, so a claim's provenance is visible inline:

- `[V]` — read by the author at a named `file:line`, or produced by a named command
- `[V*]` — cited by another agent **and** independently reproduced
- `[I]` — inferred over `[V]`/`[V*]` facts
- `[V-abs]` — an **absence**: a named grep returning zero hits, where no `file:line` can exist
- `[OP: file]` — a verbatim operator input

## Reporting drift

These docs drift when the code moves. When you find a disagreement:

1. Check `docs/design/` first — if the design changed, the track doc is simply out of date.
2. Check the code with a real probe (`grep`, a run, a receipt). **Do not trust a doc over a probe.**
3. Open an issue naming: the **file and line** in the track doc, the **probe you ran**, and the
   **`file:line` or design section** that contradicts it.
4. Fix it on a branch and open a PR. `main` is PR-only, on principle. One logical change per commit,
   committed **by path**.

If the drift is that a design-only thing became real, the fix is usually one word in the doc and a
new `file:line` in `grounded_in` — say so in the PR and bump `last_verified`.
