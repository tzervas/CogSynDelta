# CSD remaining program

Live plan. Each task names its gate — the observable that says it is done — because a task
without one gets reported complete on vibes.

Status: `todo` | `wip` | `done` | `blocked` | `deferred`

---

## P0 — Correctness debt. Blocks everything downstream.

Nothing measured before P0.1 lands is trustworthy: the text encoder attended to padding, so
every metric was batch-composition dependent.

| id | task | gate | status |
|----|------|------|--------|
| P0.1 | Retrain code, compress, retrieve with attention masking fixed | 3 receipts, all `beats_untrained` true | wip |
| P0.6 | **Make the corpus shuffle unconditional** | code's holdout spans >2 repos; batch negatives are cross-repo | todo — BLOCKS P0.2/P0.3 |
| P0.7 | Retrain code + compress AGAIN after P0.6 | receipts on a representative holdout | wip — running |
| P0.8 | Checkpoint fingerprint does not cover CODE changes | a checkpoint from before a data-path change refuses to resume | todo |
| P0.2 | Re-run benchmark battery on retrained regions | eval receipts written, anisotropy < 0.9 | todo |
| P0.3 | Re-run PTQ against retrained fp32 baselines | quant receipts, drop within 0.01 | todo |
| P0.4 | Fix csd-storage-tier verification | manifest honours the SAME excludes as the rsync | done — homelab SSD 1.5T -> 2.1T free |
| P0.5 | Commit the train-dependency guard in csd-train-all.py | committed, gate green | done |

**P0.4 detail.** The 661 GB offload transferred and then failed verification:
`hot_manifest 4c4c6c0b` vs `cold_manifest 7300c204`. Cause is mine — the rsync excludes
receipts/checkpoints/manifest.json/*.log, but the manifest hashes every file on both sides,
so a mismatch is guaranteed. The hot copy was correctly kept. Fix the comparison, re-verify,
and only then reclaim. Do NOT delete anything until a corrected manifest matches.

### P0.6 detail — the measurement is not what it looks like

`build_splits` shuffles only when `extra_sources` is non-empty:

    if len(cfg.extra_sources) > 1 or cfg.extra_sources:

`code` and `compress` have ONE source each, so they are never shuffled. `retrieve` has
three, so it is. Measured consequence for `code`:

  - the first 3000 rows of shard 0 span 5 repositories; pandas-dev/pandas alone is 1626
  - **the first 512 rows -- the entire holdout -- span 2 repositories**

So recall@1 0.9355 is "tell pandas docstrings apart from each other", not "retrieve the
right function from Python". The in-batch negatives are same-repo and therefore easy, and
the held-out set is a domain-shifted slice rather than a sample.

The comment directly above that conditional already states the principle -- contiguous
blocks make negatives same-domain, which "inflates in-batch accuracy while teaching the
model less". The reasoning was correct and the condition was wrong: it applies the fix
ACROSS sources and never WITHIN one.

Fix is a one-line condition change, but it invalidates every single-source number measured
so far. `retrieve` is unaffected. Do NOT compare a post-fix `code` number against 0.9355 or
0.9590 as though the difference were caused by training.

### P0.8 detail — the resume fingerprint has a blind spot

`_config_fingerprint` hashes PretrainConfig fields: paths, steps, batch_size, lr, seed. It
does NOT hash corpus content or the build_splits code path. The shuffle fix changed corpus
ORDER without changing any config field, so a fingerprinted checkpoint written before it
would have resumed silently against differently-ordered data.

Harmless today only by accident: every checkpoint on disk predates the resumable feature
and carries no fingerprint at all, so all are correctly treated as "start fresh". The gap
is real for the next such change.

Fix direction: include something that moves when the data pipeline moves -- the corpus
fingerprint already recorded in receipts is the obvious candidate, since it is derived from
the actual shard list.

## P0.9 — Findings from the corpus survey. These change what our numbers MEAN.

Measured on the 1080 Ti with the project's own encoders (correct for redundancy, explicitly
NOT used to judge quality). Full results: /mnt/bulk/csd-corpus-analysis/analysis.json

| id | finding | why it matters | status |
|----|---------|----------------|--------|
| P0.9a | `code` truncates 93.9% of code-side tokens at max_len=96 | ANSWERED — see below. Truncation inflated the BASELINE, not the trained score | done |
| P0.9b | `retrieve` holdout has 53.7% near-dupes (>=0.90) in train | 0.748 is inflated. But see the split below -- not all of it is leakage | todo |
| P0.9c | `compress` graded/STS-B gate NEVER RAN (`graded_shards=[]`) | a documented gate that silently did nothing | todo |
| P0.9d | 646 anchor==positive pairs in compress (0.23%) | a free InfoNCE win that teaches nothing | todo |
| P0.9e | anchor-only dedup drops valid one-to-many structure | one FiQA question with 23 relevant passages collapses to one, losing 22 real positives | todo |

### P0.9a RESULT — truncation inflated the baseline, not the score

Matched arms, only max_len differing (batch 512, 4000 steps, same lr and seed):

| | max_len=96 | max_len=256 |
|---|---|---|
| untrained r@1 | 0.2285 | **0.0918** |
| trained r@1 | 0.9805 | **0.9941** |
| LEARNED GAIN | 0.7520 | **0.9023** |
| wall clock | 324 s | 716 s |

The untrained baseline is the evidence. A random-init encoder scores 0.2285 at 96 tokens
and 0.0918 at 256 -- signature-level lexical overlap is exploitable when only the signature
is visible, and that shortcut weakens by 60% once whole function bodies are in view.

So the suspicion was half right. Truncation WAS inflating the number, but it inflated the
BASELINE rather than the trained score. The trained model is better with more context
(0.9805 -> 0.9941) against a harder problem. On learned gain over baseline, longer context
wins decisively: 0.902 against 0.752.

Harder task, higher absolute score, bigger gain -- the encoder was starved of context, not
cheating. And it is direct support for P11.4: if 256 beats 96 this clearly at 2.2x wall
clock, a SCHEDULE (96 -> 256 -> longer) is worth building rather than picking a bigger
constant, because the cheap early phase costs little and the expensive phase is where the
gain is.

TWO KINDS OF "LEAKAGE" IN retrieve, and only one is a defect:
  - genuine paraphrase: "how do you know if..." vs "how to know if...", cos 0.993. Real.
  - GooAQ template collision: "44 is 25 percent of what number?" vs "...55 percent...",
    cos 0.995 -- SAME template, DIFFERENT correct answer. That is not duplication; it means
    the metric is measuring the encoder's failure to distinguish a slot value. Deduplicating
    it away would hide a real weakness rather than fix one.

CORRECTION TO AN EARLIER CLAIM IN THIS FILE: raw CodeSearchNet is BALANCED -- top repo
2.44%, 13,581 unique repos. The "1,626 of 3,000 rows are pandas" figure was a pre-shuffle
HEAD artifact, not a property of the corpus. The P0.6 defect was real (the holdout was 2
repos) but the corpus was always diverse; the earlier wording overstated it.

CROSS-DATASET, for anything added later:
  - staged `snli` is 100% already inside all-nli. It would add ZERO new pairs.
  - APPS and CodeContests share 17.8% of problems at cos>=0.90, INVISIBLE to hashing because
    CodeContests prefixes every description with "<id>_<letter>. <Title> - ". Dedupe on the
    stripped description.
  - SQuAD/HotpotQA share 95.9% of article titles. Dedupe by title, not passage hash.

## P2.4 — Licence-clean retrieval mix for MIT open-weights

AUDITED at mirror AND upstream. A clean >100k-pair mix exists.

| dataset | pairs | licence, both ends verified |
|---------|-------|-----------------------------|
| allenai/gooaq | 3,112,679 | Apache-2.0 (upstream LICENSE file read) |
| tasksource/esci | 2,027,874 | Apache-2.0 (amazon-science/esci-data LICENSE read) |
| castorini/mr-tydi | ~167k | Apache-2.0, built on tydiqa which is also Apache-2.0 |
| miracl/miracl | ~40k queries | Apache-2.0 -- under 100k queries, flagged honestly |
| THUIR/T2Ranking | 258k queries | Apache-2.0 in README ONLY; LICENSE file 404s. Verify before use |

IMMEDIATE ACTION: we source gooaq and natural-questions from the `sentence-transformers`
mirrors. EVERY dataset under that account declares NO LICENCE -- all 75. The upstream
`allenai/gooaq` is Apache-2.0 with a real LICENSE file. Same data, actual licence. Switch.

WHAT THIS REPLACES: squad, hotpotqa and natural-questions are all CC-BY-SA. Under an MIT
open-weights release, share-alike propagation to derived weights is the open question the
licence audit is assessing. The mix above avoids it entirely.

MIRROR-MORE-PERMISSIVE-THAN-UPSTREAM, now demonstrably systemic rather than incidental:
  unicamp-dl/mmarco          Apache-2.0 asserted over MS MARCO's non-commercial terms
  nthakur/msmarco-sampled-*  CC-BY-SA-3.0 over the same NC upstream
  facebook/kilt_tasks        MIT over CC-BY-SA and unknown constituents
  xanhho/2WikiMultihopQA     Apache-2.0 where upstream states no licence at all
  mandarjoshi90 README       "Apache 2.0 applies to the data" while the project site says
                             "UW does not own the copyright of the questions and documents"
  allenai/peS2o              flat ODC-BY over S2 Data the S2 licence says may be CC-BY-NC
  lucadiliello/newsqa        redistributes CNN text with no licence, where upstream says it
                             "cannot be made directly available due to legal reasons"
  INVERSE: mteb/nq           CC-BY-NC-SA-3.0 where upstream NQ is CC-BY-SA-3.0, no NC

## P1 — Repo hygiene. Nearly done.

| id | task | gate | status |
|----|------|------|--------|
| P1.1 | Make bitnet-rs and ternary-rs private | `isPrivate: true`, `isArchived: true` preserved | done — both verified |

Both are archived, which makes them read-only to the API (`HTTP 403`). Sequence is
unarchive -> flip -> re-archive, so the archived state they started in is restored.

## P2 — Corpus expansion

`/bulk` is 5.9 TB at ~10% used; the eviction threshold is 50%. Catalogue holds 11
licence-verified datasets across 6 domains, none fetched.

| id | task | gate | status |
|----|------|------|--------|
| P2.1 | Fetch the catalogued datasets to /bulk/csd-corpus | manifests written, row counts recorded | done — 12 manifests, 20 GB, 6.2M rows |
| P2.2 | Wire classify + reason regions into the runner | dry-run resolves shards for both | todo |
| P2.3 | Train the new regions | receipts with `beats_untrained` | todo |

Licence gate is structural: entries not marked TRAIN_OK are refused and there is no
override flag. A mirror's tag is not evidence about its upstream — BeIR/scifact is tagged
cc-by-sa-4.0 while allenai/scifact, which it mirrors, is cc-by-nc-2.0.

### P2.1 notes — three fetch failures and what they actually were

Worth keeping, because two are a general pattern that will recur.

`datasets` 5.0.1 REFUSES to execute remote loading scripts. Any HF dataset whose repo is
just a `*.py` builder now fails with "Dataset scripts are no longer supported". Two of the
three failures were this, with different resolutions:
  - codeparrot/apps had an HF-bot Parquet conversion, so the entry pins
    `revision="refs/convert/parquet"`.
  - PolyAI/banking77 had NO conversion. Its script only downloaded two CSVs from GitHub, so
    the entry now points at those URLs through the PACKAGED `csv` loader -- a loader that
    ships with the library, not remote code, so it stays inside the ban's intent rather
    than around it. That GitHub repo's LICENSE was checked directly (CC BY 4.0) and agrees
    with the catalogue verdict.

BeIR datasets have NO `train` split. hotpotqa is two configs, `corpus` and `queries`, with
relevance judgements in a SEPARATE repo (BeIR/hotpotqa-qrels). The old single entry
described a joined form that never existed. It is now two entries, and the corpus+queries+
qrels join is documented as a caveat rather than forced into a fetch loop that does not fit
it. Expect the same shape from any other BeIR-family dataset.

### Corpus provenance: two hard rules

**PROVENANCE IS CAPTURED AT SOURCING TIME, NOT RECONSTRUCTED.**
Licence, attribution, citation, source URL and retrieval date are recorded at the moment of
fetch. Once you hold a directory of files, that information is unrecoverable -- no later
stage can infer it. This makes acquisition the highest-stakes step in any ingest pipeline,
not the most mechanical one. Every downstream transformation (resize, crop, dedup, format
conversion, tokenisation) must carry the record forward or the pipeline has wasted every
stage before the one that dropped it.

**THE GENERATING MODEL IS PART OF THE PROVENANCE CHAIN.**
Any model used to generate, label, caption, filter or enrich data contributes its own terms
to the result. Two live instances:
  - SWIM-IR's queries are PaLM-2 outputs over CC-BY-SA Wikipedia passages. The dataset has
    TWO stacked provenance questions -- the passages' licence, and whether a provider's terms
    attach to generated text -- and the licence tag captures only the first.
  - Labelling a clean image corpus with an ImageNet-trained classifier reintroduces exactly
    the provenance question the clean corpus was built to avoid.

CONSEQUENCE FOR THIS PROJECT: generate and label with PERMISSIVELY-LICENSED LOCAL models,
not hosted APIs. Slower per token, unencumbered output. For a corpus intended for
publication that trade is correct, and it closes the loop -- open-weights models building the
corpus that trains an open-weights model, with no provenance question anywhere in the chain.
The fleet already has the hardware and llama-rag is now on-demand rather than pinned.

## P2.5 — Broad corpus build-out. Licence-aligned, non-overlapping, multi-TB.

STORAGE IS NOT THE CONSTRAINT. /bulk is 5.9 TB at ~14% used. Terabytes are fine. Old
datasets on the spinning array may be offloaded to HF, stowed, or deleted once superseded.

THE THREE HARD REQUIREMENTS, in priority order:

1. LICENCE-ALIGNED FOR MIT OPEN WEIGHTS. Not merely "may I train on it" -- may a derived
   model be redistributed permissively. Apache-2.0 / MIT / BSD / CC0 / ODC-BY verified at
   BOTH mirror and upstream. Eight audited cases exist where a mirror claims more than its
   upstream grants, so a card tag alone is never sufficient.

2. NON-OVERLAPPING. This is now MEASURABLE rather than assumed, and the tooling exists
   (/mnt/bulk/csd-corpus-analysis/). Two demonstrations of why it matters:
     - staged `snli` is 100% contained in all-nli -- it would have added ZERO pairs
     - APPS and CodeContests share 17.8% of problems, INVISIBLE to hashing because one
       prefixes every description with "<id>_<letter>. <Title> - "
   Every candidate gets an overlap check against what is already held, BEFORE download.
   Downloading a terabyte of duplicate is worse than downloading nothing.

3. NO SINGLE-CORPUS DOMINANCE. `retrieve` is already 77.8% GooAQ, 19.5% NQ, 2.7% FiQA while
   being EVALUATED on financial-domain FiQA. That is training on one distribution and
   measuring on another. Caps exist for exactly this and must be set per source, not
   globally.

COVERAGE NEEDED, per region and then for the composed model:
  code      currently CodeSearchNet Python only, and licence-REJECTED at that. Needs
            multi-language and permissively licensed
  retrieve  the P2.4 mix (gooaq upstream, esci, mr-tydi) replaces the CC-BY-SA set
  compress  all-nli has no licence tag and MultiNLI carries restricted genres. Needs a
            permissive paraphrase/entailment source
  vl        BLOCKED pending the licence audit -- tiny-imagenet may be ImageNet-derived
  classify  banking77 + go_emotions are clean; more breadth wanted
  reason    gsm8k + aqua_rat are clean; the MATH family is DMCA-encumbered
  compose   the CSD model itself needs material that does NOT overlap what the regions
            trained on, or the composed evaluation is contaminated by construction

| id | task | gate | status |
|----|------|------|--------|
| P2.5a | Overlap-check tooling as a reusable gate | a candidate is rejected on measured overlap, not judgement | todo |
| P2.5b | Per-region licence-clean candidate lists | every entry verified at upstream | todo |
| P2.5c | Fetch, with per-source caps | no source exceeds its cap; balance recorded in the receipt | todo |
| P2.5d | Reserve non-overlapping material for the composed model | held-out from every region's training set | todo |


## P3 — Visual region maturation

`vl_latent` passed its gates but weakly: probe top-1 0.0606 on 200 classes, and the run used
3 GB of a 24 GB card.

| id | task | gate | status |
|----|------|------|--------|
| P3.1 | Scale the VL run (steps and batch) | probe top-1 materially above 0.0606, not collapsed | todo |
| P3.2 | Add flickr30k as extra pretrain data | loader filters `__MACOSX/` junk; 31,783 images | todo |

## P4-P6 — Curriculum. Order is mandatory.

Foundation is step 4, not step 1.

| id | task | gate | status |
|----|------|------|--------|
| P4 | Router over trained regions | routing accuracy beats uniform baseline | todo |
| P5 | Composed mind | composed beats best single region on a mixed set | todo |
| P6 | Foundation training | after P4 and P5, never before | todo |

## P7 — Publication

| id | task | gate | status |
|----|------|------|--------|
| P7.1 | Weights-only checkpoint export | exported file ~64 MB, not 184 MB | todo |
| P7.2 | Push regions to private HF repos | repos exist, private, contain weights | todo |

Checkpoints are 184 MB of which only 64 MB is weights; the rest is AdamW optimizer state.
Fine for resuming, wrong to publish.

## P10 — Fleet-parallel training

Two idle-ish GPUs and one measured constraint that decides the design.

MEASURED FACTS
- 3090 Ti (24 GiB) on akula-prime .98; RTX 5080 (16 GiB) on gpu5080 .251.
- They are on DIFFERENT HOSTS, linked at 1000 Mb/s.
- A live region run uses ~5 GiB of 23 GiB at 39% utilisation. The 5080 sits at 10 MiB.
- All regions trained so far total 87M parameters / 348 MB fp32.

WHY NOT DDP ACROSS THE TWO CARDS
Data-parallel synchronises gradients every step. A 16M-parameter model is ~64 MB of
gradients per step per direction; over 1 GbE an all-reduce costs on the order of a second,
which exceeds the step time it would be overlapping. Cross-host DDP would make training
slower. It is the right technique on the wrong interconnect.

WHY REGION-PARALLEL INSTEAD
Regions are independent by construction -- that is what the architecture means. Training
`code` on one host and `compress` on the other requires ZERO gradient communication. The
speedup is near-linear in hosts, and it needs no distributed runtime.

WHY BATCH SIZE IS A QUALITY LEVER, NOT A SPEED ONE
InfoNCE draws its negatives from the batch. A larger batch is a harder and more informative
contrastive problem, so the idle VRAM converts directly into quality. This is why P9.1/P9.2
rank above raw parallelism: filling one card well beats spreading a small batch across two.

| id | task | gate | status |
|----|------|------|--------|
| P10.1 | Make the runner able to execute a region on a named host | a region trains on gpu5080 and writes a receipt back | todo |
| P10.2 | Region-to-host scheduling (independent regions run concurrently) | two regions training simultaneously on two hosts | todo |
| P10.3 | Measure the composed-model footprint before P6 | actual params/bytes recorded; decide if one card constrains it | todo |

DEPENDENCY: P10 lands AFTER P9.1/P9.2. Filling one card is worth more than splitting a
small batch across two, and a batch-size change alters what fits per host -- doing P10
first would mean scheduling against numbers that are about to change.

## P11 — Batch composition and negative difficulty

Operator-identified. Two of these are standard practice we simply are not doing, and one is
a real defect.

THE DEFECT: batches are contiguous slices of a list shuffled ONCE.
`pretrain.py` line ~657: `chunk = train_pairs[lo : lo + cfg.batch_size]`. The corpus is
shuffled at load and never again, so batch composition is frozen for the whole run. `code`
trains ~4.7 epochs at 8000x256 over 430,931 pairs, so every fixed group of 255 negatives
repeats about five times. The model sees the same discriminations over and over instead of
new ones.

| id | task | what it buys | gate | status |
|----|------|--------------|------|--------|
| P11.1 | Reshuffle between epochs | every epoch presents new negative combinations; standard practice (`shuffle=True`) that we skipped | batch composition provably differs across epochs; recall improves or is explained | todo |
| P11.2 | Hard negative mining | InfoNCE learns most from negatives it nearly confuses; random in-batch negatives are mostly trivially easy | mined-negative run beats random-negative run on the same holdout | todo |
| P11.4 | Progressive sequence length ("crawl, walk, run") | short context early, longer context later. Reframes max_len from a binary choice into a schedule | staged-length run matches or beats a fixed-length run at equal compute | todo |
| P11.5 | Progressive visual resolution | same principle for images: small crops early, larger views later, so the visual region eventually reasons over a whole screenshot rather than a tile | staged-resolution beats fixed at equal compute | todo |
| P11.6 | Composite-image training ("patch n' pack") | many images per frame; find, compare and relate elements within one view | composite-trained encoder locates a target element among N that a tile-trained one cannot | todo |
| P11.3 | Curriculum over negative difficulty | easy discriminations first, fine ones later -- "dog, then Belgian Malinois vs other dogs" | staged difficulty beats constant difficulty at equal step count | todo |

COMPOSITE IMAGES -- established, with one correction to the motivating argument.
Prior art: NaViT's "Patch n' Pack" (multiple variable-resolution images packed into one ViT
sequence), mosaic augmentation from YOLOv4 (4 images into 1), and Set-of-Marks prompting
(identify a specific element among many). What is being proposed is making this the NATIVE
training regime rather than an augmentation trick.

THE CORRECTION: packing does NOT reduce compute. N images in one sequence is (N*64)^2
attention against N*(64^2) processed separately -- quadratic in total tokens means packing
costs MORE per image, not less. The efficiency framing is wrong.

THE REAL WINS, which are better arguments anyway:
  - cross-image reasoning becomes possible AT ALL. Separate forward passes cannot compare,
    locate or relate across a set; one packed pass can.
  - fixed overheads amortise -- one pass, one set of norms and projections.
  - IT MATCHES THE DEPLOYMENT TARGET. A screenshot IS a collage. If the end state is a model
    reading a display, "many elements in one frame, find the relevant one" is the NORMAL
    case and training on isolated 64x64 tiles is the artificial one.

STAGING IS PROBABLY REQUIRED, and the operator's instinct here is likely correct: blasting a
heterogeneous mix at an encoder early looks more likely to fragment it than to teach general
structure. I-JEPA predicts masked-region representations from visible context, and that
signal is coherent within a domain and much less so across wildly dissimilar ones. Expect to
need per-domain or weighted staging before a naive union. This should be MEASURED, not
assumed -- it is a testable claim and the probe already exists to test it.

PROGRESSIVE LENGTH IS ESTABLISHED PRACTICE, and it reframes P0.9a. The max_len question is
not "96 or 256" -- it is "96 THEN 256". Training short first is cheaper (attention is
quadratic in sequence length) and the model learns local structure before being asked to
integrate a whole function. Long-context models are routinely trained this way: a base
context first, extended later, rather than paying quadratic cost from step one.

The same argument extends to vision: small crops early, whole screenshots later. That is
the concrete mechanism behind the operator's goal of a model that reads a full display --
you do not start it there, you get it there.

COROLLARY FOR CORPUS SELECTION: many clean datasets beat few dirty ones. The operator is
explicit that swapping to more datasets to satisfy licensing is fine -- "I'm entirely okay
with changing to different datasets and using more datasets to get accomplished what could
be done with fewer datasets with different licenses". Trusted and unpoisoned is the bar;
source count is not. This materially widens the replacement search for BLOCKING corpora,
since a composite of narrow clean sets can substitute for one broad encumbered one.

PRIOR ART -- this is established, not novel, which is good news: epoch reshuffling is
universal; progressive hard-negative mining is the core of DPR, ANCE and RocketQA, where
each round mines negatives the current model ranks highly but that are wrong.

SEQUENCING, per the operator: land conventional training correctly FIRST, then add these
and measure whether each helps. P11.1 is cheap and fixes a defect, so it can go early.
P11.2/P11.3 are experiments and must be gated on a working baseline -- otherwise a gain
cannot be attributed.

## P9 — Modern training stack

Make training, fine-tuning and quantization idempotent, parameterised, pausable, resumable
and fast. Ordered by value-per-risk, not by novelty. Every item states what it buys, because
"modern" is not a reason to add something.

Sequencing constraint: P9.1-P9.3 all edit regions/pretrain.py. They must land ONE AT A TIME
or they collide. P0.1 resumable checkpointing is in flight against that same file.

| id | task | what it buys | gate | status |
|----|------|--------------|------|--------|
| P9.0 | Survey: what applies here, measured | grounding — avoids adding technique for its own sake | written design doc with measured baselines | todo |
| P9.1 | bf16 mixed precision | ~2x step time and half the activation memory on Ampere; the freed VRAM buys batch size, and in InfoNCE the negatives ARE the batch | step time and peak VRAM measured before/after; recall within noise of the fp32 run | todo |
| P9.2 | Gradient accumulation | effective batch beyond VRAM, which is the single biggest lever on contrastive quality | effective batch 4x physical, recall improves or is explained | todo |
| P9.3 | torch.compile | free throughput on an unchanged model | step time measured before/after; identical metrics | todo |
| P9.4 | Config-driven runs (declarative, not CLI flags) | idempotent and re-runnable; a run is a file you can diff, not a command someone typed | same config re-run reproduces the receipt | todo |
| P9.5 | LoRA / adapter fine-tuning | fine-tune a region without retraining it; serves capability-per-parameter directly | adapter-only training beats frozen baseline at <5% of trained params | todo |
| P9.6 | QAT (quantization-aware training) | recovers accuracy PTQ leaves on the table at aggressive widths | at 3-bit, QAT beats the PTQ result on the same held-out set | todo |
| P9.7 | Early stopping on the gate metric | stops burning GPU on a run that has plateaued | run halts within N evals of no improvement | todo |

Already in place, do not rebuild: warmup+cosine LR, gradient clipping, deterministic
seeding with corpus fingerprints, receipt-based experiment tracking, Prometheus export,
sensitivity-driven PTQ with real sub-byte packing.

## P12 — Late-phase training regime (operator vision)

All established techniques. Every one belongs AFTER a working conventional baseline, per
the operator, because none can be attributed without one.

| id | task | name in the literature | status |
|----|------|------------------------|--------|
| P12.1 | Mixed-modality, mixed-class batches late in training | multi-task / interleaved training | todo |
| P12.2 | "not a Malinois -- but what IS it, is it even a dog" | hierarchical classification + open-set / OOD recognition | todo |
| P12.3 | Keep older data in the mix so a region does not forget | experience replay | todo |
| P12.4 | Internal deliberation before answering | latent reasoning (Coconut-style continuous thought) -- SAME as the deferred looped transformers | todo |

WHY VL MAY BE THE PRIMARY INTERFACE, NOT A SIDE QUEST
The operator's argument, which holds up: a 64x64 image at patch-8 is 64 positions, the same
order as a 96-token text sequence, but a screenshot carries far more decision-relevant
information than 64 tokens of prose. Per unit of compute, vision can be the cheaper
channel. If that holds at scale, the end state is a model handed a display or a VM rather
than a token stream, with the text regions as specialists rather than the trunk. This is
why `vl_latent` should not be treated as optional polish.

THE END GOAL THAT SHAPES ALL OF IT
Not a model that captions images or emits text. A composed mind of specialised regions
working in concert, architected for traceability -- eventually rewritten in the operator's
own language so provenance is total and no layer is a black box. That is why regions are
trained and gated INDIVIDUALLY, and why every number carries a receipt: a system you can
interrogate layer by layer has to be built that way from the start, not instrumented later.

## P13 — After CSD: hybrid predictive training, then a Rust stack

Explicitly AFTER a fully trained and polished CSD. Recorded so it persists, not to be
started.

ORDER, per the operator: Python first, fully proven and dialled. Only then Rust — and then
the WHOLE stack in Rust: model, training, quantization, fine-tuning, orchestration, and the
hybrid predictive training approach itself.

STRICT ORDER, set by the operator. Each step gates the next.

| id | task | gate |
|----|------|------|
| P13.0 | CSD fully trained conventionally | all regions gated green on a clean holdout; receipts are the CONTROL GROUP for everything below |
| P13.1 | Fix the PYTHON hybrid trainer | port the Rust design fix back to Python (RSSM-lite world model, predict OUTCOMES not gradients), close the gaps, modernise. Gate: trains without diverging past the step where the old one hit NaN (~160) |
| P13.2 | Retrain CSD with the hybrid trainer, validate | direct A/B against P13.0's receipts on identical data. Gate: matches or beats conventional recall, at lower cost |
| P13.3 | Fix the RUST hybrid trainer | close the Burn VRAM leak (.map() copies the whole model per weight-delta), validate the unvalidated mitigation, modernise. Gate: the 77%/99.9% result reproduced at real model scale |
| P13.4 | Modernise the whole Rust AI/ML ecosystem | currently paused mid-process; resumes here |
| P13.5 | Forgejo repos under the right orgs | see the org map; note the fleet mirrors under `aphelion`, not `tzervas` |

WHY THIS ORDER IS RIGHT, not just a preference:
- CSD trained conventionally is the CONTROL. A novel trainer validated without a baseline on
  the same data cannot distinguish "the trainer works" from "that run was lucky". P13.0
  produces receipts on a known-clean holdout, which is exactly the comparison P13.2 needs.
- Python before Rust. The Rust lineage holds the DESIGN fix; the Python lineage holds the
  working training harness. Porting the fix backwards validates the algorithm where
  iteration is cheap, and only then pays Rust's cost to reimplement something already
  proven.
- The Rust VRAM leak is engineering, not algorithm. It should not block validating whether
  the algorithm is correct.
| P13.4 | Forgejo repos for everything, under the right orgs | see org map below; blocked on the P13.1 catalogue so placement is informed rather than guessed |

### P13.1 — located, and the failure is already diagnosed

THE PYTHON LINEAGE: `tritter` (local /home/kang/code/personal/tzervas/python-ai/tritter,
Forgejo `aphelion/tritter`, GitHub `tzervas/tritter`). Two mechanisms both called "hybrid
predictive training":
  - an embedding-prediction curriculum (JEPA/Coconut-shaped, alpha ramps token loss ->
    embedding loss). Unit-tested, never wired into a real Trainer.
  - a phase-cycling gradient predictor (WARMUP -> FULL -> PREDICT -> CORRECT) that skips
    backprop during PREDICT using an EMA of past gradients. THIS is what broke.

ROOT CAUSE, quoted from its own postmortem: *"the current EMA-based gradient prediction
fails because it accumulates errors exponentially over steps, treating gradients as
independent entities when they are actually samples from a structured, learnable dynamical
system."* Concrete run: a 100M BitNet converged normally 11.2 -> 3.407 through step 112,
then diverged to NaN by step ~160-200 as PREDICT-heavy phases compounded the error.

THE RUST LINEAGE: `hybrid-predict-trainer-rs`, best copy inside the `rust-ai` monorepo. It
is a real implementation of the fix the Python postmortem prescribes -- an RSSM-lite
world-model predictor with online GRU training, i.e. PREDICT OUTCOMES, NOT GRADIENTS. It
stalled on a VRAM leak: Burn's functional `.map()` copies the entire model on every
weight-delta, taking GPT-2-small from 3.9 GB to 14.1 GB in 50 steps. A same-day mitigation
was written and explicitly never validated.

NOT A DEAD END: one validated run is documented at 77% speedup (4x) at 99.9% quality. The
mechanism has partial proof; the scaling and VRAM engineering is what was never finished.

PRESERVATION: the whole postmortem existed ONLY in unpushed local commit b8f8670 and is now
on both remotes as branch `preserve/local-2026-03-05-hybrid-research`, verified byte-
identical (blob b962c296, 41,345 bytes). Local `main` and `origin/main` have DIVERGED --
1 commit local vs 6 upstream, 13 files overlapping, including the very optimization module
upstream rewrote unaware. Do not merge casually.

NOT STARTING FROM ZERO. The operator already has a substantial Rust ML ecosystem on this
fleet, all now private: rust-ai-core, bitnet-rs, ternary-rs, triton-bridge-rs (CUDA driver
loader), memory-gate-rs, trit-vsa, tritter-accel, mycelium and hypha. P13.3 is closer to
consolidation than to greenfield, and the CUDA and ternary pieces are the hard parts that
already exist.

WHY THIS ORDER IS RIGHT, not just preference: a Rust rewrite of an unproven design ports
the design's mistakes into a language where they are more expensive to fix. CSD in Python
is the reference implementation that makes the Rust version a translation rather than a
redesign.

## P14 — Model instantiation policy (its own repo)

Manifests make this possible: once a model's resources are DECLARED, they are something you
can write policy against. Without manifests there is nothing to enforce on.

THE CONTROLS WANTED: who may instantiate which model, for what purpose, for how long, with
how much resource.

PRIMITIVES ALREADY ON THIS FLEET -- generalise these, do not invent parallel machinery:
| control | what already exists |
|---------|--------------------|
| by whom | SOPS/age scoping in ~/.secrets/.sops.yaml. It already excludes gpu5080 from `git/*`, and an agent honoured that today rather than shipping a token there. |
| how much | systemd cgroup limits per unit (MemoryMax, CPUQuota, DeviceAllow); k3s ResourceQuota for containerised work |
| isolation | rootless podman with --cap-drop=ALL, already the fleet convention |
| how long | systemd RuntimeMaxSec; the on-demand llama-rag.service pattern |
| audit | receipts, the Prometheus exporter, and Loki already on homelab |

INDUSTRY STANDARD TOOLS worth evaluating rather than hand-rolling: OPA/Rego or Cedar for
policy decisions, Kyverno if enforcement lands in k3s, SPIFFE/SPIRE for workload identity.
The measure of success is that a rejected instantiation says WHICH rule refused it and why.

NOTE ON GPU QUOTAS specifically: the 3090 Ti and 5080 are consumer cards -- no MIG, so a GPU
cannot be hard-partitioned between workloads the way an A100 can. Enforcement there is
cooperative (a memory fraction a process sets on itself, as the corpus-analysis job did) or
coarse (one job per card, which is what the fleet does today). Do not design assuming
hardware isolation that does not exist on this hardware.

| id | task | status |
|----|------|--------|
| P14.1 | Its own private repo, GitHub + Forgejo | todo |
| P14.2 | Policy schema over the manifest fields | todo — depends on the manifest design |
| P14.3 | Enforcement at instantiation | todo |

DEPENDS ON: the manifest design. Policy without a declared resource envelope has nothing to
evaluate.

### REMOTE CONVENTION — read this before creating or pushing anything

**Forgejo (git.vectorweight.com) is the WORK target. GitHub is REMOTE BACKUP.**

Consequences that change behaviour:
- A new repo is created on Forgejo FIRST, under the right org, then mirrored to a private
  GitHub repo. Not the other way round.
- Push to Forgejo first. A GitHub-only commit is unbacked work sitting in the wrong place.
- PRs, issues and CI live on Forgejo. GitHub is not where work is reviewed.
- Where the two diverge, Forgejo is authoritative.
- Every GitHub mirror stays PRIVATE, without exception.

Note most of the fleet already mirrors to Forgejo under org `aphelion`, NOT under `tzervas`
as one might assume.

### Forgejo organisation map

Three orgs already exist on git.vectorweight.com, so new repos go INTO them rather than
flat under `tzervas`:

| org | purpose |
|-----|---------|
| aphelion | primary org; mirrors Aphelion-Development |
| cabal-collective | self-hosted Cabal agent work, CPU runners |
| mycelium | the mycelium language and its `mycelium-*` Rust crates |

TOKEN SCOPES -- this will bite anyone who does not know it:
`git/tzervas-forgejo` is PUSH-ONLY. It has no `read:user` and no `read:organization`, so
listing orgs or creating repos with it returns a 403 that names a missing scope rather than
a permission problem. Use `git/cabal-forgejo-admin` for anything beyond pushing to an
existing repo.

Eleven repos currently live on GitHub with no Forgejo counterpart: anemochory (NOT the
operator's -- owned by dark-harold), gha-runner-ctl, slovo, python-field-notes,
ap-fleet-work-images, self-hosted-ai, range, rust-ai-core, pybench, bitnet-quantize,
notes-sandbox. Those are the candidates for placement.

## P15 — The long arc (operator vision, context not backlog)

Recorded so the near-term work stays pointed at it. None of this is scheduled.

**Visual scale-out.** Compressed screenshot corpora over Grokipedia and other sources, so
the model reasons about rendered information rather than only prose. Connects to the P12
argument that vision may be the cheaper channel per unit of decision-relevant information.

**Polyglot specialisation, deliberately narrow.** Rust, Python, TypeScript, quantum
languages, and eventually mycelium. The specialisation targets are software engineering, AI
engineering, architecture, research and design -- broad competence with deep spikes, not
uniform mediocrity.

**Mycelium as a first-class training target.** The language is designed to be natively good
for AI without being machine-first: one language that sugars up to human-readable and
lowers mechanically to machine-level, with no separate DSLs or paradigms. Training a model
on a language built to be AI-legible is a genuinely different proposition from training it
on one that merely tolerates AI.

**Embodiment.** Display-out plus keystroke control, then camera and audio, then a robot.
The architecture has to be efficient enough to run at the edge, which is why the ternary
direction matters -- a hybrid of trits and trytes rather than single-trit, targeting
insanely efficient edge inference that still scales up to a data centre.

**Developmental curriculum for alignment, not just capability.** Train in phases the way
children are educated: age-appropriate material first, complexity introduced progressively,
so a baseline human-aligned moral sense is baked in LOW rather than bolted on as a filter.
The hypothesis is that safety established during formation is more robust than safety
imposed afterwards. This is P11/P12 curriculum work aimed at values rather than difficulty.

**Context-aware refusal.** The target is a model that distinguishes "help me stop this
attack" from "help me run one" -- refusing accurately rather than refusing broadly. A model
that blocks defensive security work during a live incident has failed; so has one that
cannot tell the two apart. This is a CAPABILITY, not a filter, and it needs the reasoning
the rest of the architecture is for.

**Persona basin ablation as a release step.** Remove every persona basin except the target
one, so undesired behaviour is not merely suppressed but absent -- nothing for a malicious
user to steer the model into. NOT GREENFIELD: akula-ai-platform already has
`config/abliterate/` and `make abliterate` targets. That tooling identifies and removes
directions in activation space; this extends it from "remove refusal" to "leave only the
intended basin", applied as a final production-readiness wrap.

WHY THIS SHAPES THE NEAR-TERM WORK: an architecture you can interrogate layer by layer is a
precondition for all of it. You cannot ablate a basin you cannot locate, curriculum-train
values you cannot measure, or trust a refusal you cannot explain. That is the same reason
regions are trained and gated individually and every number carries a receipt.

## P8 — Deferred by explicit operator decision

- Recursive/looped latent transformers
- CI image minimisation
- CPU-mode polish for tiny/quantized variants
- Quantum compute backend (no hardware)

## Running a training job

Training runs as a systemd **user** unit (`deploy/systemd/csd-train@.service`, installed to
`~/.config/systemd/user/`) so a run survives the death of whatever shell launched it.
`nohup ... &` only blocks SIGHUP -- it does not survive the launching process group being
killed, which is exactly how two unattended runs died silently in one day: the log stopped
mid-run with no error and no receipt, and the only evidence was `nvidia-smi` showing an idle
GPU. `setsid` was the same-day stopgap; the unit is the durable fix, since systemd owns the
process directly rather than an interactive shell's background job.

Start a run (regions comma-separated, no spaces, matching `--regions` in
`scripts/csd-train-all.py`):

    systemctl --user start csd-train@code,compress,retrieve

Follow it:

    journalctl --user -u csd-train@code,compress,retrieve -f

Check whether it is still running, or how it ended:

    systemctl --user status csd-train@code,compress,retrieve

Notes:
- Steps/batch/shard-limit/state are fixed in the unit to the known-good production
  invocation (`--steps 8000 --batch 256 --shard-limit 0 --state /akula-data/csd`); only the
  region list is templated per-instance.
- `,` is not a valid systemd unit-name character, so every command above logs a harmless
  "Invalid unit name ... maybe you should use systemd-escape?" warning. It still works --
  the unit uses `%I` (escaping undone), not `%i`, to recover the literal comma-separated
  region list before it reaches `--regions`. Do not "fix" the warning yourself.
- `Restart=no` is deliberate: a failed `beats_untrained` gate means a broken configuration,
  not a transient fault. It needs a human or agent to read the receipt in
  `/akula-data/csd/receipts` and change something, not a timer retrying forever.
- Only one instance should run at a time -- there is one GPU. Check `nvidia-smi` and
  `systemctl --user list-units | grep csd-train` before starting a second one.
