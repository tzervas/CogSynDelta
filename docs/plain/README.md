---
track: plain
title: The plain track — CogSynDelta without the jargon
pairs_with: null
depends_on_foundations: []
grounded_in:
  - docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md (rev 3.5)
  - docs/README.md (track chooser)
code_revision: forgejo/main 4f3c098
last_verified: 2026-09-04
---

# The plain track

**Owner:** the CSD programme operator (Tyler Zervas). **Index last verified:** 2026-09-04,
against design rev 3.5 at `forgejo/main 4f3c098`. **Contributions:** branch and PR, per
[the track chooser](../README.md).

> **Status 2026-09-04.** This index is ratified. **The chapter files are not written yet** —
> `docs/plain/` currently holds only this README. Rows are **unlinked** until the file lands;
> a linked row means the file exists.

Thirteen documents, in the order the ideas actually arrive. No maths is assumed. Where an idea
needs a prerequisite, the doc names a [foundations](../foundations/README.md) unit by stem and
you can go get it in one sitting.

Every doc here has a technical twin with the **same filename** in [`../technical/`](../technical/README.md).
Same subject, more precision, `file:line` anchors. Read the plain one first; reach for the twin when
you need to check something.

## Read in order

Times are at ~200 words a minute. **"prereq"** is the *transitive* cost of the foundations units
this doc assumes, if you have not read them — that number is why it is printed. **Status** is
the doc-level banner: whether it describes code that exists, a specification, or a decision the
programme is waiting on.

| # | doc | status | read | prereq | what you will be able to do afterwards |
|---|---|---|---|---|---|
| 01 | What CogSynDelta is, and why it is small on purpose | mixed / blocked | ~6 min | ~10 min | Say what CSD is trying to prove, why the current model is deliberately a toy, and what would count as it working — and know, from this doc rather than doc 06, that the central bet lost and the programme is blocked. |
| 02 | Regions are brain faculties, not task domains | design only (the renames) | ~7 min | ~6 min | Read the region list and say which entries are real faculties of the mind, which are frozen measuring instruments, and which name in the code is scheduled to change. |
| 03 | White matter: the shared workspace, and the rule that nothing turns back into words | **design only** | ~7 min | ~15 min | Explain how faculties are meant to talk to each other, what the two budgets buy — how much a faculty may read, and how much room it gets on the shared workspace — and why re-serialising to text between regions is forbidden. |
| 04 | Memory: one shared memory system with two jobs, and a store that can be written to | mixed (region built, store design only) | ~6 min | ~22 min | Tell the parametric memory region that exists today from the writable episodic store that does not yet, and say what the memory gate arbitrates. |
| 05 | How CSD is trained: four phases, and never everything at once | phase 1 built, 2–4 design only | ~7 min | **~34 min** | Name the phases in order, say which one the programme is in, and explain why adding a submodel does not trigger a full retrain. |
| 06 | How we know it works: receipts, gates that can fail, and the untrained number | **built** | ~7 min | **~34 min** | Read a CSD receipt, say what would have falsified the claim, and explain why the most important number is usually the untrained one. |
| 07 | Making it fit: quantisation, real bytes, and the VRAM budget | mixed | ~6 min | ~16 min | Say what quantisation costs and buys here, why a bit-width claim must correspond to bytes on disk, and how jobs are placed across three unequal GPUs. |
| 08 | Where the data comes from: sources, provenance groups, and the balance rules | mixed | ~7 min | **none** | Say where a training row is allowed to come from, why a corpus can be big and still too narrow to prove anything, and what the dataset factory actually found. |
| 08b | What that means for the weights: fingerprints, keyed splits, and the licence you inherit | mixed | ~6 min | ~10 min | Trace one training row from its source to a released checkpoint's licence, and say why the composed mind is currently non-commercial. |
| 09 | Running many variants: one config, a grid of experiments, and every result versioned | mixed | ~6 min | ~20 min | Understand how one config describes a whole grid of runs, and how any published checkpoint traces back to the revision that produced it. |
| 10 | What lives where: the repos, the scripts, and the code that is history | **built** | ~6 min | **none** | Know which repo to open for a given job, which scripts are the real entry points, and which directories are dormant history. |
| 11 | Where it is going: a billion per faculty, cheaper steps, and trits | **design only** | ~6 min | ~30 min | Tell a decided direction from a gated experiment, and say what each long-term track must prove first. |
| 12 | Running it, and contributing without breaking the evidence | built | ~6 min | ~20 min | Run a region end to end, read the receipt it produces, and land a change under the rules that keep the measurements trustworthy. |

Docs **08** and **10** need no foundations unit at all — they are the two cheapest entry
points if you are short on time. Docs **05** and **06** are the expensive ones, and the
prereq column says so before you open them rather than after.

## Which foundations unit each doc leans on

Listed as `depends_on_foundations` in each doc's front matter. Collected here so you can plan a
reading path. The **transitive** column counts the units those units in turn require.

| plain doc | foundations units it names | transitive total |
|---|---|---|
| 01 | `01-vectors-and-embeddings`, `02-tokens-position-latents-and-pooling` | 2 units, 2,000 words |
| 02 | `01-vectors-and-embeddings` | 1 unit, 1,100 words |
| 03 | `01`, `02`, `03-attention-and-a-transformer-block` | 3 units, 3,000 words |
| 04 | `01`, `09-retrieval-metrics-and-bm25`, `13-gpu-memory-arithmetic` | 4 units, 4,300 words |
| 05 | `04-loss-and-gradient-descent`, `05-cross-entropy-and-masked-prediction`, `05a-distillation-and-ema-targets`, `06-contrastive-learning-and-in-batch-negatives`, `07-covariance-and-decorrelation` | **7 units, 6,800 words** |
| 06 | `08-singular-values-and-effective-rank`, `09`, `10-baselines-controls-and-preregistration`, `11-hashing-fingerprints-and-keyed-splits` | **7 units, 6,900 words** |
| 07 | `12-quantisation-arithmetic`, `13` | 3 units, 3,300 words |
| 08 | — | none |
| 08b | `11`, `14-licences-as-a-dataflow-property` | 2 units, 2,000 words |
| 09 | `10`, `11` | 4 units, 4,100 words |
| 10 | — | none |
| 11 | `08`, `12`, `13` | 6 units, 6,100 words |
| 12 | `10`, `11` | 4 units, 4,100 words |

Docs 01–03 need very little. Doc 06 is where the foundations track earns its keep — and it
costs about 6,900 words of prerequisites to get there, which is exactly why the
[foundations index](../foundations/README.md) opens with a minimum-viable path.

## Two habits this track keeps

**It says when something does not exist.** Most of the interconnect is a specification, not code. You
will see **design only** in these pages often — and now at the top of each doc, not only per claim,
because docs 03 and 04 are 2,700 consecutive words about a system that is almost entirely
unbuilt and the reader deserves to know that before paragraph one. That is the honest state, not
a gap in the writing.

**It reports failures as failures.** The programme's central bet was measured and it lost; the first
token-aware retrain failed two of its five pre-registered clauses. Those results are in docs 01 and 06
as results, not as setbacks to be explained away. The measurement discipline is the point of the project.

The source of truth is [`../design/`](../design/). Where these docs disagree with it, they are wrong —
see [the track chooser](../README.md) for how to report drift.
