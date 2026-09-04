---
track: plain
title: The plain track — CogSynDelta without the jargon
pairs_with: null
depends_on_foundations: []
grounded_in:
  - docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md (rev 3.5)
  - docs/README.md (track chooser)
last_verified: 2026-09-03
---

# The plain track

Twelve documents, in the order the ideas actually arrive. No maths is assumed. Where an idea needs a
prerequisite, the doc names a [foundations](../foundations/README.md) unit by stem and you can go get
it in one sitting.

Every doc here has a technical twin with the **same filename** in [`../technical/`](../technical/README.md).
Same subject, more precision, `file:line` anchors. Read the plain one first; reach for the twin when
you need to check something.

## Read in order

| # | doc | what you will be able to do afterwards |
|---|---|---|
| 01 | [What CogSynDelta is, and why it is small on purpose](01-what-csd-is.md) | Say what CSD is trying to prove, why the current model is deliberately a toy, and what would count as it working. |
| 02 | [Regions are brain faculties, not task domains](02-regions-as-faculties.md) | Read the region list and say which name is a faculty, which is a probe, and which name in the code is about to change. |
| 03 | [White matter: the shared workspace, and the rule that nothing turns back into words](03-white-matter-and-the-latent-invariant.md) | Explain how faculties are meant to talk to each other, what a "budget" buys, and why re-serialising to text between regions is forbidden. |
| 04 | [Memory: one hippocampal trunk, two heads, and a store that can be written](04-memory-and-the-episodic-store.md) | Tell the parametric memory region that exists today from the writable episodic store that does not yet, and say what the memory gate arbitrates. |
| 05 | [How CSD is trained: four phases, and never everything at once](05-how-it-is-trained.md) | Name the phases in order, say which one the programme is in, and explain why adding a submodel does not trigger a full retrain. |
| 06 | [How we know it works: receipts, gates that can fail, and the untrained number](06-how-we-know-it-works.md) | Read a CSD receipt, say what would have falsified the claim, and explain why the most important number is usually the untrained one. |
| 07 | [Making it fit: quantisation, real bytes, and the VRAM budget](07-quantisation-and-packing.md) | Say what quantisation costs and buys here, why a bit-width claim must correspond to bytes on disk, and how jobs are placed across three unequal GPUs. |
| 08 | [Where the data comes from, and what that means for the weights](08-data-provenance-and-licences.md) | Trace one training row from its source to a released checkpoint's licence, and say why the composed mind is currently non-commercial. |
| 09 | [The matrix pipeline: many variants, one config, and every one of them versioned](09-matrix-pipeline-and-hub-versioning.md) | Understand how one config describes many runs, and how any published checkpoint traces back to the revision that produced it. |
| 10 | [What lives where: the repos, the scripts, and the code that is history](10-the-tooling-map.md) | Know which repo to open for a given job, which scripts are the real entry points, and which directories are dormant history. |
| 11 | [Where it is going: a billion per faculty, cheaper steps, and trits](11-where-it-is-going.md) | Tell a decided direction from a gated experiment, and say what each long-term track must prove first. |
| 12 | [Running it, and contributing without breaking the evidence](12-running-it-and-contributing.md) | Run a region end to end, read the receipt it produces, and land a change under the rules that keep the measurements trustworthy. |

## Which foundations unit each doc leans on

Listed as `depends_on_foundations` in each doc's front matter. Collected here so you can plan a
reading path:

| plain doc | foundations units it assumes |
|---|---|
| 01 | `01-vectors-and-embeddings`, `02-tokens-position-latents-and-pooling` |
| 02 | `01-vectors-and-embeddings` |
| 03 | `01-vectors-and-embeddings`, `02-tokens-position-latents-and-pooling`, `03-attention-and-a-transformer-block` |
| 04 | `01-vectors-and-embeddings`, `09-retrieval-metrics-and-bm25`, `13-gpu-memory-arithmetic` |
| 05 | `04-loss-and-gradient-descent`, `05-cross-entropy-and-masked-prediction`, `06-contrastive-learning-and-in-batch-negatives`, `07-covariance-and-decorrelation` |
| 06 | `08-singular-values-and-effective-rank`, `09-retrieval-metrics-and-bm25`, `10-baselines-controls-and-preregistration`, `11-hashing-fingerprints-and-keyed-splits` |
| 07 | `12-quantisation-arithmetic`, `13-gpu-memory-arithmetic` |
| 08 | `11-hashing-fingerprints-and-keyed-splits`, `14-licences-as-a-dataflow-property` |
| 09 | `10-baselines-controls-and-preregistration`, `11-hashing-fingerprints-and-keyed-splits` |
| 10 | `13-gpu-memory-arithmetic` |
| 11 | `08-singular-values-and-effective-rank`, `12-quantisation-arithmetic`, `13-gpu-memory-arithmetic` |
| 12 | `10-baselines-controls-and-preregistration`, `11-hashing-fingerprints-and-keyed-splits` |

Docs 01–03 need very little. Doc 06 is where the foundations track earns its keep.

## Two habits this track keeps

**It says when something does not exist.** Most of the interconnect is a specification, not code. You
will see **design only** in these pages often. That is the honest state, not a gap in the writing.

**It reports failures as failures.** The programme's central bet was measured and it lost; the first
token-aware retrain failed two of its five pre-registered clauses. Those results are in doc 06 as
results, not as setbacks to be explained away. The measurement discipline is the point of the project.

The source of truth is [`../design/`](../design/). Where these docs disagree with it, they are wrong —
see [the track chooser](../README.md) for how to report drift.
