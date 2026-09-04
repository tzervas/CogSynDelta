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
code_revision: forgejo/main 4f3c098
last_verified: 2026-09-04
---

# CogSynDelta documentation

**CogSynDelta (CSD) is a small experimental language model.** Rather than one
undifferentiated stack, it is built as a set of specialised **faculties** — memory, vision,
reasoning, language — each trained separately and then composed. It is deliberately
toy-scale right now, and that is the experiment: the thing being tested is **capability per
parameter**, whether a mind assembled from specialised parts can do more per unit of size
than one large general stack. It is a research programme with real code and real receipts,
not a product, not a company, and not a finished system.

**It is also, so far, a programme that has published two failures.** Its central bet was
measured and lost, and the first training run designed to answer that is blocked on an open
decision. Those results are in the docs as results — see `plain/06` or `technical/06`.

Three tracks. Same programme, three different reading contracts.

| track | who it is for | what it promises |
|---|---|---|
| [**plain/**](plain/README.md) | anyone who wants to understand what CSD is and why, without a maths background | plain language, no unexplained jargon, every hard idea handed off to a foundations unit |
| [**technical/**](technical/README.md) | engineers and reviewers working on or auditing the code | precise specifications, decision ids, `file:line` anchors, and an explicit line between what is built and what is designed |
| [**foundations/**](foundations/README.md) | plain-track readers who hit a prerequisite they do not have | one idea per unit — the concept, why CSD needs it, the real code path, and a runnable CPU "try it" |

**New here? Start at [`plain/01`](plain/README.md) and only detour into `foundations/` when a
doc's front matter tells you to.**

> **Status 2026-09-04.** The three track indexes are ratified. **No chapter or unit file is
> drafted yet** — each track directory holds only its `README.md`. Index rows are therefore
> **unlinked**; a linked row means the file exists. That is the invariant, in all three
> indexes.

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
  `DATASET-FACTORY-CATALOGUE-2026-09-03.md`, `MODEL-MANIFESTS.md` and the **seven**
  `evidence/` directories carry the licence, corpus, manifest and sha-bound receipt detail.
- `program/REMAINING.md` and `program/KICKOFF.md` carry the plan.
- `src/cogsyndelta/` and `scripts/` are the real implementation. **A live probe beats a document.**

Three rules follow, and they are not negotiable:

1. **Where a track doc conflicts with `docs/design/`, the design doc wins and the track doc is a bug.**
   The one exception is a **code-state** claim: there, a live probe beats every document
   *including the design doc*, and the track doc reports the probe. Two design claims are
   already stale this way — see `technical/02` (§1.5's "six tracked files"; a probe returns
   eight) and `technical/12` (row W0c's gate (iii); a probe returns nine `torch.load` sites,
   not one).
2. **Many legacy files directly under `docs/`** — `ARCHITECTURE.md`, `API_REFERENCE.md`,
   `CODE_DOCUMENTATION.md`, `CONFIGURATION.md`, `HANDOFF-ARCHITECTURE.md`, `GPU_COMPATIBILITY.md`,
   `CONTRIBUTING.md`, `DEVELOPMENT_STANDARDS.md`, the compression-fidelity and VSA essays, and
   others — are **history or aspiration**, not fact. Read them as a record of what was once
   intended. **Never cite them for a claim**, and never list one in a `grounded_in`. The single
   route to citability is an explicit **still-true** verdict recorded in
   [`technical/10`](technical/README.md), which must also name the file here as an exemption.
   **There are currently no exemptions.**
3. **`docs/adr/` is a separate class and is not covered by rule 2.** ADRs are decision records:
   citable as evidence of *what was decided and when*, never as evidence of current code state.
   **Cite an ADR by full filename, never by number** — three files share the `0009-` prefix
   (`0009-algebraic-training-optimization.md`, `0009-reference-materials.md`,
   `0009-supplement-algebraic-training-research.md`), so "ADR-0009" names three documents.

## Design is not implementation

The single most common way to misread this programme is to assume that because the design doc
specifies something, it exists. Much of it does not, yet. Every track doc therefore carries a
**doc-level status banner** in its first screen — *built* / *mostly design only* / *blocked* —
which is also printed as a column in each track index, and marks its individual claims:

- **built** — there is code, and usually a receipt;
- **design only** — specified in `docs/design/`, absent from `src/`;
- **blocked** — waiting on an operator decision, named by title.

As of `last_verified` above: the white-matter interconnect, the episodic store, memory-gate overlays
and the `Schedule` are all **design only**; the region renames in §1.4 are marked *REVIEW ONLY, DO NOT
APPLY* and have **not** been applied to `config/mind/csd-regions.json`; and the programme is
**blocked on OD-17** (W4 failed two of five pre-registered clauses at its pre-registered
configuration). A doc that describes any of these as finished is wrong.

---

# For contributors and reviewers

Everything above is what a reader needs. Everything below is the schema, the tag vocabulary
and the PR workflow — skip it unless you are writing or reviewing one of these docs.

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
code_revision: forgejo/main 4f3c098
last_verified: 2026-09-04
---
```

`grounded_in` is the audit trail: it names what a reviewer must re-read to re-verify the doc.
Exactly **four** classes of entry are permitted, and which one it is must be visible:

1. a `docs/design/` file **with its revision**;
2. a code path, with `file:line` where the claim is about a specific call;
3. a `program/` plan file;
4. **operator memory** under `~/.claude/projects/.../memory/`, cited by stem and tagged
   `[OP: stem]` at every point of use — exactly as the design doc tags operator inputs.
   `technical/09`, `/10`, `/12` and `plain/08b` all rest on operator memory; without this
   class those claims were indistinguishable from design-sourced ones.

A legacy file directly under `docs/` is **not** a permitted class (rule 2 above).

`code_revision` pins the base sha the doc's `file:line` anchors were verified against. It exists
because `grounded_in` pinned the *design* revision and nothing pinned the *code* revision, so a
single upstream edit silently invalidated every anchor with no mechanical way to tell — and a
technical doc's whole value is its anchors. This mirrors what
`src/cogsyndelta/regions/_receipt.py:58` (`capture_code_revision`) already stamps onto every
receipt: a result is bound to the code that produced it. A doc whose `code_revision` is behind
`git rev-parse HEAD` has anchors that are **unverified**, not necessarily wrong — re-probe
before citing them.

`last_verified` is the date someone actually re-read the sources. A doc whose `last_verified`
is older than the design revision it cites is **stale until re-checked**.

## Evidence tags

The tracks reuse the design doc's tag vocabulary, so a claim's provenance is visible inline:

- `[V]` — read by the author at a named `file:line`, or produced by a named command
- `[V*]` — cited by another agent **and** independently reproduced
- `[I]` — inferred over `[V]`/`[V*]` facts
- `[V-abs]` — an **absence**: a named grep returning zero hits, where no `file:line` can exist
- `[OP: file]` — a verbatim operator input

## The `B` namespace: always qualify it

`B` names **four different things** across these tracks, and a bare `B2` is ambiguous:

| write it as | what it is | where it is defined | who uses it |
|---|---|---|---|
| `baseline B0` / `B0d` / `B0u` / `B1` / `B2` / `B2t` / `B3` | the nested baseline ladder | design §2.7.1 (line 1704) | `technical/06` |
| `boundary B1`…`B5` | the five adversarial boundaries | design §9.9 (line 5606) | `technical/03`, `/04`, `/12` |
| `balance rule B1`…`B5` | the corpus concentration checks | `CORPUS-CONTRACT.md` Part 3 (line 1132; B1 at :1147, B5 at :1219) | `technical/08` |
| `review finding B2` | the threat/review pass finding on `MEASURED_VRAM_AT_BATCH_512` | `docs/design/evidence/w4-control-arm-2026-09-03/README.md:3` | `technical/05` |

**Never write a bare `B<n>`.** As it stood, "B2" in doc 06 and "B2" in doc 04 were different
objects with no marker.

## Reporting drift

These docs drift when the code moves. When you find a disagreement:

1. Check `docs/design/` first — if the design changed, the track doc is simply out of date.
2. Check the code with a real probe (`grep`, a run, a receipt). **Do not trust a doc over a probe**
   — and note that this beats the design doc too, for code-state claims (rule 1).
3. Open an issue naming: the **file and line** in the track doc, the **probe you ran**, and the
   **`file:line` or design section** that contradicts it.
4. Fix it on a branch and open a PR. `main` is PR-only, on principle. One logical change per commit,
   committed **by path**.

If the drift is that a design-only thing became real, the fix is usually one word in the doc and a
new `file:line` in `grounded_in` — say so in the PR, bump `code_revision`, and bump `last_verified`.
