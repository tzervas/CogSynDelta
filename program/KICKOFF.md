# CSD session kickoff

Paste this as your first message. It is written to be read by the model, not by you.

---

You are orchestrating CogSynDelta (CSD) — a composed mind of specialised submodel regions
on a self-hosted 4-host GPU fleet. Target: capability of a far larger model from far fewer
parameters, released as open weights, running on a single ~16 GB consumer card.

## Scale — the current model is a TOY, deliberately

Today ~87M parameters, ~450 MB. **Target: as close to a 30B-class model as reasonably
achievable.** Present size is scaffolding to get the architecture right the first time.
Judge current work on whether the STRUCTURE is correct, not on its numbers.

The arithmetic makes quantisation load-bearing rather than incidental: 30B is 120 GB at
fp32 (impossible), 15 GB at 4 bits (marginal), **12.3 GB at the 3.27 effective bits/param
this project's PTQ receipts already measure** — which fits a 16 GB card with ~4 GB for
activations. That is the deployment target.

At scale with long context and vision, ACTIVATION and KV memory dominate weight memory, so
per-region context budgets and selective activation are the levers, not weight paging.
**Dynamic paging is planned, not built** — design the seam, implement when measured.

Training phases, one of which appears nowhere else:
    1. per-region pretraining        <- current
    2. interconnect training          needs (1) to exist
    3. WHOLE-MIND DYNAMIC TRAINING    everything together WITH the interconnect trained.
                                      Probably where most growth in parameters, layers and
                                      region types happens.
    4. fine-tune, then quantise       in that order

**Long context is a first-class goal and primarily VISUAL** — efficient large context across
discrete tokens but chiefly vision, image and video.

**What the operator optimises for: performance, efficiency, functionality, skill, capability,
tools.** Not benchmark rank. Weigh decisions against those.

## The architecture, in the operator's terms — read this before anything else

Regions are **brain-analogous FACULTIES, not task domains.** The test for a well-formed
region is *what faculty does this provide*, never *what dataset does it train on*. A
banking-intent classifier answers only the second, so it is not a region — at most a probe.

Intended shape: reasoning/logic centre, hippocampus (memory), visual cortex, auditory
cortex, speech/language centre, numeric/math centre, **white matter**, frontal cortex
(unification), plus AI-specific regions where the architecture must differ from a brain.

**White matter IS the interconnect AND the routing model** — and it is the centre of gravity
of the whole architecture, not plumbing. It is what wires the submodels into a cohesive
single mind rather than seven models behind a switchboard. Attention is already a routing
primitive, so cross-region attention gives one mechanism doing both jobs: the attention
weights ARE the connection strengths and routing is emergent, not a discrete dispatch made
outside the representation.

It is a **learned scheduler**, not a router. It must learn from context and scenario: which
regions to activate, with what intensity and priority, how attention is split and spread
across them, how much context window each gets, and the execution topology — what runs
async, what sequentially, and what must run parallel in lockstep. Context is hybrid: sliding
windows over an overarching context, plus latent-reasoning windows, budgeted per region.

It is **late-stage by dependency, not preference** — there is no signal to learn from until
trained faculties exist producing real representations. It is its own substantial research
problem, closer to learned program synthesis than to anything currently in the tree.

Full detail: the REGION TAXONOMY section of program/REMAINING.md.

## Your role

You are **strictly the orchestrator and high-level planner**. Delegate all execution to
agents and workflows. Right-size model and reasoning effort per task, dynamically:

- **haiku** — mechanical enumeration, status sweeps, repo listings, file inventory
- **sonnet** — implementation, infra changes, licence verification, training runs, anything
  touching remotes or hosts
- **opus** — architecture decisions, measurement design, adversarial review, anything where
  being subtly wrong is expensive

Use `Workflow` to fan out independent work; use `Agent` for single tracks. Do not spawn an
agent to *watch* something that reports for itself — a polling agent burned 155k tokens
saying "still fine" four times. Runs write receipts; read those instead.

Do not do execution work yourself. Do the planning, the sequencing, the verification of
agents' claims, and the decisions.

## Read before planning anything

    program/REMAINING.md        the program: P0–P15, every task with the observable that
                                says it is done. Opens with SESSION HANDOFF — read that first.
    docs/design/                six design documents, 7,800 lines:
                                  LICENCE-FOR-OPEN-WEIGHTS.md   per-corpus verdicts, release scenarios
                                  CORPUS-CONTRACT.md            what each region's data must contain
                                  TRAINING-SUPERSET.md          why a shippable superset needs a builder
                                  VISUAL-INGEST-PIPELINE.md     provenance-at-fetch, dedup bootstrap
                                  MODERN-TRAINING-STACK.md      measured performance work
                                  MODEL-MANIFESTS.md            declarative model instantiation

Your memory directory also holds five entries from the prior session. They encode mistakes
worth not repeating.

## The pattern that produced most prior findings

**A guard correct in reasoning and wrong in scope, reporting success it could not have
detected the absence of.** Four instances, none found by reading code — all found by
measuring the world. The worst: `build_splits` dedups anchors on a hash, then
`assert_no_contamination` recomputes the *identical* hash over the same anchors, so overlap
is empty by construction for every region, always. An independent survey measured 53.7%
near-duplicate leakage in a holdout that guard called clean.

**Verify a guard by constructing the case it exists to reject and asserting it fails.**
`tests/test_guards_can_fail.py` exists for exactly this.

**Nearly every defect was data or measurement, not modelling.** `compress` went
0.2578 → 0.4961 → 0.7070 with no modelling change — only fixing what it was fed.

## Non-negotiable practices

- **Every run gates on beating its own UNTRAINED baseline.** A random-init text encoder
  scores recall@1 0.40 on CodeSearchNet from lexical overlap alone; I-JEPA loss falls while
  a representation collapses. Without the baseline both look like success.
- **Never trust a mirror's licence tag.** Ten documented cases where a HuggingFace mirror
  claimed terms its upstream never granted. Verify upstream; refuse on mismatch. And keep
  *may I train on this* separate from *may I redistribute it* — they come apart.
- **`git commit -m "..." -- PATH`**, never `git add X && git commit`. This tree has
  concurrent agents; a bare commit sweeps in whatever else is staged. That happened twice.
- **Forgejo is the work target, GitHub is backup.** Push Forgejo first. Most repos mirror
  under the Forgejo org `aphelion`, not `tzervas`. `git/tzervas-forgejo` is push-only; use
  `git/cabal-forgejo-admin` for read or repo creation.
- **One logical change per commit**, staged by name, gates green before pushing.

## Fleet facts that mislead if unknown

    akula-prime  192.168.1.98   RTX 3090 Ti 24 GiB   sudo via SOPS `akula/sudoer-pass`
    gpu5080      192.168.1.251  RTX 5080 16 GiB      /bulk = 5.9 TB spinning RAID0, NFS-exported
    gpu1080ti    192.168.1.243  GTX 1080 Ti 11 GiB   macvtap guest ON gpu5080 — CANNOT reach its own host
    homelab      192.168.1.170  no GPU               NFS server, VictoriaMetrics :8428, Loki :3100

The two GPUs are on **different hosts at 1 Gb/s**, so cross-host DDP is the wrong tool —
gradient all-reduce costs more than the step it overlaps. Train one whole region per host.
Secrets: `secret exec VAR=path -- cmd`. gpu5080 and gpu1080ti are excluded from `git/*` by
`.sops.yaml` — honour that rather than shipping tokens there.

## Immediate state

Complete: three text regions retrained on a shuffled corpus; `vl_latent` trained but on an
UNRELEASABLE corpus (tiny-imagenet traces to ImageNet's non-commercial terms); PTQ at ~9.8x;
pipeline console serving receipts from two hosts into VictoriaMetrics; 660 GB tiered to
spinning storage with verified copies; GPU utilisation 54.5% sawtoothing -> 98.7% with
recall UP.

The contamination guard is FIXED (it was vacuous — see below) with 21 liveness tests, but
its extended channels were never run against the real corpora. A guard that fires on
everything is as useless as one that fires on nothing; only real data settles which this is.
**That is the first thing to verify.**

Three additional regions were trained and gated, then their receipts were DELETED by a later
iteration. Numbers read before deletion:
    reason                0.0039 -> 0.0801  recall@1                 PASS
    classify_banking77    0.0127 -> 0.8453  top-1, chance 0.0130     PASS
    classify_go_emotions  0.0497 -> 0.3642  macro-AP, chance 0.0466  PASS
The models are fine; the FRAMING is wrong — `classify_banking77` is a domain, not a faculty.
Rename or demote rather than retrain.

Left deliberately UNTRACKED: `src/cogsyndelta/regions/retrieve.py` is a second, divergent
retrieve regime training on FiQA alone with a real BEIR-style eval. Committing it would put
two contradictory retrieve definitions in the tree. It needs a decision.

## Two decisions waiting on the operator — ask early

1. **Release licence.** CC-BY-SA buys ~887k pairs, concentrated entirely in `compress` and
   `retrieve`; `code`/`classify`/`reason` gain zero and `vl_latent` nearly zero. Per-region
   dual licensing is the standing recommendation — MIT where nothing is bought, CC-BY-SA
   only where it is. See LICENCE-FOR-OPEN-WEIGHTS.md §Release licence scenarios.
2. **GooAQ README-vs-LICENSE contradiction.** 3.1M pairs, 77.8% of `retrieve`, the largest
   number at risk in the whole corpus. Unresolved and unresolvable without web search.

## Open architectural decisions, all of which reshape the plan

1. **Region taxonomy.** `compress` and `retrieve` both look hippocampal — two operations of
   ONE memory faculty, not two regions. `code` looks like a domain of the language centre
   rather than a peer of it. Settle before the interconnect, because it dispatches by these
   declarations and a wrong name propagates into every routing decision.
2. **Interconnect design.** (a) cross-region attention as white matter, routing emergent;
   (b) fixed interconnect plus a thalamic gate; (c) both. P4 is currently written as a
   discrete router, which is the ALTERNATIVE to learned connectivity rather than a step
   toward it — building it as specified produces something to replace, not extend.
3. **P5 has no training step.** Composition was framed as "regions plus a router evaluated
   together". If the interconnect is what makes them one mind, the compose phase IS training
   it. P5's gates measure a result, not the thing being built.
4. **Regions were all trained in isolation**, each optimising to solve its task alone. If an
   interconnect unifies them, their objectives may need to account for contributing to a
   shared state. Cheap to test early, expensive to discover late.

## Suggested first actions

Confirm what is running, read the SESSION HANDOFF section, then propose a sequenced plan and
check it against the operator before dispatching. **P2.5d (reserve composed-model material)
must precede P2.3 (train new regions)** — allocation is irreversible, and training first
leaves the composed model with nothing to be evaluated on.
