# CSD remaining program

Live plan. Each task names its gate — the observable that says it is done — because a task
without one gets reported complete on vibes.

Status: `todo` | `wip` | `done` | `blocked` | `deferred`

---

## SCALE TARGET AND PHASES — the current model is a TOY, deliberately

Today: measured 3 x 16,021,248 + 22,905,216 = 70.97M params, ~284 MB fp32 for the four
trained regions (450 MB assumed 7 x 16M). **Target: as close to a 30B-class
model as reasonably achievable.** The present size is scaffolding to get the architecture
right the first time, not the deliverable. Judge current work on whether the STRUCTURE is
correct, not on its numbers.

**THE ARITHMETIC MAKES QUANTISATION LOAD-BEARING, not incidental:**
    30B fp32                    120 GB   impossible on consumer hardware
    30B bf16                     60 GB   still impossible
    30B at 8 bits                30 GB   two cards
    30B at 4 bits                15 GB   marginal on a 16 GB card
    30B at 3.27 bits           ~12.3 GB  FITS, with ~4 GB left for activations
3.27 effective bits/param is not aspirational -- it is the measured figure from this
project's own PTQ receipts (9.5-9.8x compression at under 1 point of recall). The
sensitivity-driven mixed-width work is what makes the deployment target reachable at all.

**WHERE MEMORY PRESSURE ACTUALLY LANDS AT SCALE:** at 30B with long context and vision,
ACTIVATION and KV memory dominate WEIGHT memory. The levers there are per-region context
budgets and selective activation, not weight paging. Dynamic paging is planned, not built --
design the seam for it, implement when the constraint is measured rather than anticipated.

**TRAINING PHASES, including one not previously recorded:**
    1. per-region pretraining          faculties trained individually        <- current
    2. interconnect training           learned scheduler; needs (1) to exist
    3. WHOLE-MIND DYNAMIC TRAINING     everything trained together WITH the
                                       interconnect already trained. Not previously
                                       recorded anywhere. This is where the composed
                                       system becomes more than its parts, and it is
                                       probably where most of the growth in parameters,
                                       layers and region types happens.
    4. fine-tune, then quantise        in that order; PTQ is post-training by definition

**LONG CONTEXT IS A FIRST-CLASS GOAL, and primarily VISUAL.** Large context capability, done
efficiently, across discrete tokens but PRIMARILY vision, image and video. That is consistent
with the earlier finding that vision may be the cheaper channel per unit of
decision-relevant information -- a screenshot carries more than 64 tokens of prose.

**WHAT THE OPERATOR ACTUALLY OPTIMISES FOR:** performance, efficiency, functionality, skill,
capability, tools. Not benchmark rank. Weigh design decisions against those.

## REGION TAXONOMY — the architecture is BRAIN-ANALOGOUS, not task-analogous

**The test for a well-formed region: what FACULTY does it provide? Not what dataset does it
train on.** A banking-intent classifier answers the second and not the first, so it is not a
region. At most it is an evaluation probe for whatever language or semantic faculty exists.

INTENDED REGION SHAPE, operator's framing:
  reasoning / logic centre       inference, deduction, working through a problem
  hippocampus                    memory: encoding, consolidation, retrieval
  visual cortex                  vision, latent visual reasoning
  auditory cortex                sound
  speech / language centre       language production and comprehension
  numeric / math centre          quantity, arithmetic, symbolic manipulation
  WHITE MATTER                   high-bandwidth interconnect shunting representations
                                 BETWEEN regions -- not a router, not dispatch
  frontal cortex                 unification of regional outputs into one coherent state
  plus AI-specific regions       the architecture is not required to be a literal brain

WHAT THIS INVALIDATES IN THE CURRENT TREE:
  classify_banking77    NOT A REGION. 77 banking intents is a domain, not a faculty.
                        Trained and gated (0.8453 top-1 vs 0.0130 chance) -- the model is
                        fine, the framing is wrong. Demote to a probe or fold into a
                        language/semantic region.
  classify_go_emotions  Same problem, though affect is closer to a real faculty (limbic).
                        Decide whether it is a region or a probe.
  code                  Is this a faculty, or a domain of the language centre? Probably the
                        latter, which means `code` is a specialisation of language rather
                        than a peer of it.
  compress / retrieve   Both look hippocampal -- consolidation and retrieval are two
                        operations of ONE memory faculty, not two regions. Worth deciding
                        deliberately rather than inheriting from how they were built.
  vl_latent             Correct as-is. Visual cortex.
  reason                Correct as-is. Reasoning centre.

TWO COMPONENTS WITH NO CURRENT ANALOGUE, and both are architectural rather than incidental:
  - WHITE MATTER **IS** THE INTERCONNECT AND THE ROUTING MODEL -- or is supplemented by one.
    Operator's position, and it resolves cleanly because ATTENTION IS ALREADY A ROUTING
    PRIMITIVE. Cross-region attention gives one mechanism doing both jobs: the attention
    weights ARE the connection strengths, and routing is emergent from them rather than a
    discrete dispatch decision made outside the representation.

    That is architecturally different from the router currently planned in P4, which
    classifies a query and picks a region. A discrete picker cannot send a representation to
    two regions at partial strength; learned connectivity can, and that is what an
    interconnect is for.

    The biology agrees: routing in the brain is not a module. What connects to what IS the
    routing, carried by tract topology. Where discrete gating does exist it is THALAMIC --
    the thalamus relays and modulates what reaches cortex. So if dynamic gating is wanted on
    top of learned connectivity, the supplement is thalamus-shaped, not white-matter-shaped.

    SEQUENCING -- the interconnect is LATE STAGE BY NECESSITY. It cannot be trained before
    the regions exist, because it has no signal to learn from: there is nothing to route,
    schedule or weight until there are trained faculties producing representations. This is
    not a preference, it is a dependency. It is also its own substantial problem, not a
    finishing step.

    DESIGN CHOICE TO SETTLE BEFORE P4:
      (a) cross-region attention as white matter; routing emergent, no separate router
      (b) fixed high-bandwidth interconnect PLUS a thalamic gate that modulates what passes
      (c) both: learned connectivity, with gating layered on where it earns its cost
    P4 is currently written as a discrete router, which is option (a)'s alternative rather
    than a step toward it. Revisit before building.

    **THE INTERCONNECT IS WHAT WIRES THE SUBMODELS INTO A COHESIVE SINGLE MIND.** It is not
    plumbing between components -- it is the centre of gravity of the architecture. The
    regions are faculties; the interconnect is what makes them ONE MIND rather than seven
    models behind a switchboard.

    Three consequences, all of which change current plans:
      1. THE COMPOSE PHASE IS TRAINING THE INTERCONNECT, not evaluating regions together.
         P5's gates measure whether composition beats individual regions -- that is a
         RESULT, not the thing being built. P5 currently has no training step at all.
      2. THE RESERVED CORPUS IS THE INTERCONNECT'S TRAINING DATA. P2.5d exists so the
         composed model sees material no region has seen, but its real purpose is sharper
         than "non-overlapping": the interconnect must learn to route and integrate on data
         where NO SINGLE REGION ALREADY HAS THE ANSWER MEMORISED. Otherwise it learns to
         forward to whichever region already knows, which is dispatch, not integration.
      3. THE INTERCONNECT IS A LEARNED SCHEDULER, not a router. It must learn, from context
         and scenario, ALL of:
           - WHICH regions to activate
           - with what INTENSITY and PRIORITY
           - how much ATTENTION to allocate to each, split and spread across them
           - how much CONTEXT WINDOW each gets
           - the EXECUTION TOPOLOGY: what runs asynchronously, what sequentially, and what
             must run in parallel IN LOCKSTEP
         A router picks one thing. This emits a dataflow graph. That is a substantially
         harder learning problem and should not be underestimated by inheriting the word
         "router" from P4.

         CONTEXT MANAGEMENT is part of it: a hybrid of sliding windows over an overarching
         context, plus latent-reasoning windows. Per-region budgets, not one global window.

         MEMORY EFFICIENCY is a stated goal -- bring up only what is needed rather than the
         whole model. HONEST TIMING NOTE: a region today is 16M params (64 MB fp32, 6.5 MB
         quantized), so seven regions is ~450 MB and everything fits on the 16 GB 5080 with
         room to spare. Dynamic paging solves a problem that does not exist YET. It becomes
         real when regions scale up, or when many run concurrently with long contexts and
         ACTIVATION memory dominates weight memory. Selective activation still buys compute
         and attention budget today -- just not memory. Build it when the constraint is
         real, and measure which of the two it is.

         DEPLOYMENT TARGET: the composed mind should run on a SINGLE consumer card (~16 GB).

      4. REGIONS MAY NEED TO BE TRAINED KNOWING THEY WILL BE WIRED. A region trained in
         isolation optimises to solve its task ALONE. If the interconnect unifies them, a
         region's objective arguably should account for contributing to a shared state
         rather than producing a standalone answer. Every region trained so far was trained
         in isolation.
  - FRONTAL-CORTEX UNIFICATION. Composition has been treated as "regions plus a router".
    Unification is a faculty in its own right: integrating several regions' outputs into one
    coherent state. P5's gates were written against the router framing and need revisiting.

ACTION: config/mind/csd-regions.json declares `role` per region. Every role must name a
FACULTY. Any region whose role reads as a dataset or a domain is misframed and should be
demoted, merged, or renamed before the router is built -- the router dispatches BY these
declarations, so a wrong name propagates into every routing decision.

## SESSION HANDOFF — read this first

State is committed here and in docs/design/. Nothing depends on a conversation surviving.

**The pattern that produced most of this session's findings:** a guard correct in reasoning
and wrong in scope, reporting success it could not have detected the absence of. Four
instances, none found by reading code — all found by measuring the world:
  1. the contamination guard cannot fire (dedup and the check use the SAME hash, so overlap
     is empty by construction, always) -- P0.10
  2. the corpus shuffle applied across sources but never within one
  3. the offload manifest hashed files the transfer excluded, so it could never match
  4. `_decode_split`'s cache key uses PYTHONHASHSEED-salted `hash()`, so it never hits

**Verify a guard by constructing the case it exists to reject and asserting it fails.**
tests/test_guards_can_fail.py exists for this.

**Nearly every defect was data or measurement, not modelling.** `compress` went
0.2578 -> 0.4961 -> 0.7070 with no modelling change -- only fixing what it was fed.

**In a shared tree, `git add X && git commit` sweeps in whatever another agent staged.**
Use `git commit -m "..." -- PATH`. That mistake was made twice today.

## P0.10 — Guards that cannot fire

| id | defect | status |
|----|--------|--------|
| P0.10a | contamination guard vacuous -- same hash as dedup | done — c16362c |
| P0.10b | `_fingerprint_corpus` omits extra_sources; covers ~3% of `retrieve` | done — b957ec4 |
| P0.10c | caps truncate rather than sample (prefix, not a sample) | done — c42203c |
| P0.10d | `_decode_split` cache never hits (salted hash) | done — f20734c |
| P0.10e | tests asserting every guard CAN fail | done — landed across c16362c..2f305d2 (test_guards_can_fail.py, 21/21 passing) |

### P0.10 extended channels — measured on real corpora 2026-09-02

Extended channels measured on real corpora 2026-09-02: `code` 0/512 pair_content, `compress`
1/512 (0.195%), `retrieve` 35/512 (6.84%); the guard discriminates -- it fires on real
leakage in `compress`/`retrieve` and correctly reports zero on `code`. Commit `3203952`
("fix(eval): extend the contamination channels, preserved at session end") is the code
change that made the measurement possible; the figures themselves are recorded verbatim in
the `build_splits` comment at `src/cogsyndelta/regions/pretrain.py:562-564` ("code 0 of
512, compress 1 of 512, retrieve 35 of 512 (6.84%)") and reproduced by an independent
`build_splits()` run this session. No training receipt yet carries these. Follow-up tasks
P0.11 and P0.12 are tracked in the main P0 table below (P0 — Correctness debt), not here,
so a reader scanning that table for open work sees them.

## P0 — Correctness debt. Blocks everything downstream.

Nothing measured before P0.1 lands is trustworthy: the text encoder attended to padding, so
every metric was batch-composition dependent.

| id | task | gate | status |
|----|------|------|--------|
| P0.1 | Retrain code, compress, retrieve with attention masking fixed | 3 receipts, all `beats_untrained` true | wip |
| P0.6 | **Make the corpus shuffle unconditional** | code's holdout spans >2 repos; batch negatives are cross-repo | done — 883c9d4 |
| P0.7 | Retrain code + compress AGAIN after P0.6 | receipts on a representative holdout | done — receipts code-20260902T210830Z, compress-20260902T211539Z |
| P0.8 | Checkpoint fingerprint does not cover CODE changes | a checkpoint from before a data-path change refuses to resume | todo |
| P0.2 | Re-run benchmark battery on retrained regions | eval receipts written, anisotropy < 0.9 | todo |
| P0.3 | Re-run PTQ against retrained fp32 baselines | quant receipts, drop within 0.01 | todo |
| P0.4 | Fix csd-storage-tier verification | manifest honours the SAME excludes as the rsync | done — homelab SSD 1.5T -> 2.1T free |
| P0.4b | Sample-verify the cold copy is readable (parquet footers) since it is now the only copy | a stratified sample of at least 5% of parquet files per dataset (17 datasets under `tritter/pretrain`, 1654 parquet files total) opens with pyarrow and reports a row count; every file that fails is listed and re-fetched onto `gpu5080:/bulk`; the paths+sizes digest recomputed with `_manifest_cmd()` equals `7300c204` (self-consistency only, not a content check) | todo |
| P0.5 | Commit the train-dependency guard in csd-train-all.py | committed, gate green | done |
| P0.11 | Retrain code/compress/retrieve with the fixed guard so receipts carry the multi-channel report (see P0.10 extended channels above) | 3 receipts with `contamination.channels` present and `gated_channels` reported | todo |
| P0.12 | Eval/quant receipts must record a checkpoint content hash, not a mutable path | receipt names a sha256 that matches the file it was computed from | todo |

P0.11 evidence for `todo` (not `wip`): none of `code-20260902T210830Z.json`,
`compress-20260902T211539Z.json`, `retrieve-20260902T203759Z.json` carries
`contamination.channels` or `gated_channels`, and `ps -eo pid,etime,cmd` on both
akula-prime .98 and gpu5080 .251 shows no `csd-train`/pretrain process running --
akula-prime shows only `scripts/model-pipeline-console` (pid 2423667), gpu5080 shows
nothing matching. No retrain against the fixed guard has been dispatched anywhere yet.

P0.12 evidence: the code-eval receipt (14:37) and code-quant receipt (14:15) both name
`code-checkpoints/final.pt`, whose mtime is 17:08 -- the file they name was overwritten
hours after the receipts were written.

**P0.4 detail.** First attempt failed verification: `hot_manifest 4c4c6c0b` vs
`cold_manifest 7300c204`. Cause: the rsync excludes `receipts`, `checkpoints`, `manifest.json`
and `*.log` via `NEVER_MOVE_GLOBS`, but the manifest comparison hashed every file on both sides
regardless, so a mismatch was guaranteed even for a correct transfer. The comparison was
fixed (applied to the working tree, committed 2m22s after the successful run as `18abe38`,
2026-09-02 16:07:24 -0400, "fix(scripts): honour NEVER_MOVE_GLOBS in offload verification
manifest"). Re-verification then matched, which `scripts/csd-storage-tier.py` proves by
construction: it returns `"VERIFY FAILED — hot copy kept, nothing deleted"` and exits
*before* writing any stub whenever the hot and cold manifests differ, so the mere existence
of `OFFLOADED.json` is proof the two matched at run time -- the operator confirmed the
deletion followed a successful match. The stub records `manifest_md5 7300c204` -- the hot
digest (the script's variable `a`, not the cold `b`), which the `a != b` guard at
`scripts/csd-storage-tier.py:183-194` only lets through once it equals the cold one. All
times below are given as -0400 (the stub's `offloaded_utc` field is recorded in UTC; its
value `2026-09-02T20:05:02Z` converts to `2026-09-02 16:05:02 -0400`). The hot copy was
reclaimed at 16:05:02 -0400, 2m22s *before* `18abe38`'s 16:07:24 -0400 commit timestamp --
i.e. the corrected tool ran from the working tree before it was committed. `18abe38`'s own
commit message states this directly, not as an inference: "Re-verified manually against the
live hosts with the fix applied: hot_manifest == cold_manifest == 7300c204d2c425cdfef4b535e3ef97bd.
Ran the tool's own --apply path to reclaim the hot copy; homelab /data free space rose by
660.4 GB (of 661.3 GB expected -- the gap is btrfs discard=async trickling in)" -- the same
sequence, recorded independently of the reclaim timestamp itself. The corpus
now lives at `gpu5080:/bulk/csd-archive/tritter/pretrain` (616 GB measured via `du -sh`);
homelab holds only the `OFFLOADED.json` stub. `tests/test_corpus.py`'s four tinystories
tests now skip with an explicit reason instead of failing with `FileNotFoundError` (commit
`bb0815fd`, "test(corpus): skip the tinystories tests while that corpus is tiered off").

### P0.6 detail — the measurement is not what it looks like

`build_splits` shuffles only when `extra_sources` is non-empty:

    if len(cfg.extra_sources) > 1 or cfg.extra_sources:

`code` and `compress` have ONE source each, so they are never shuffled. `retrieve` has
three, so it is. Measured consequence for `code`:

  - the first 3000 rows of shard 0 span 5 repositories; pandas-dev/pandas alone is 1626
    [RETRACTED below: this was a pre-shuffle HEAD artifact, not a property of the corpus --
    see "CORRECTION TO AN EARLIER CLAIM IN THIS FILE" later in this file]
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
| P0.9c | `compress` graded/STS-B gate ran once at 11:36 (spearman 0.4956, `receipts/compress-20260902T153612Z.json`) then silently stopped when `csd-train-all.py` became the runner (`842db5e`); `graded_shards` is set only at `regions/compress.py:88` | a documented gate that silently stopped running, not one that never ran | todo |
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

WHAT THIS SUPPORTS, stated narrowly: truncation at 96 was NOT the mechanism producing the
high number. The authoritative post-shuffle-fix `code` recall@1 is **0.9766**
(`code-20260902T210830Z.json`, untrained baseline 0.2285). **No receipt records 0.9863 as
a code recall@1** -- the only `0.986328125` on disk is a `held_recall@10` at step 3000 of
`code-20260902T200323Z.json` (steps=6000, batch=256), a different metric at a different
step, and the comment at `scripts/csd-train-all.py:183` attributes "0.9863" to a third,
unreceipted config (steps=4000, batch=1280) that matches no receipt in
`/akula-data/csd/receipts`.

WHAT IT DOES NOT SUPPORT -- an earlier version of this entry overclaimed "the encoder was
starved of context". Two corrections:
  - recall was ALREADY above 98% at 96 tokens and rose to 99.4%. That is 10 wrong of 512
    becoming 3 wrong of 512 on a single seed against a 512-pair holdout. Modest and
    directionally consistent, not dramatic.
  - the baseline drop happens because the UNTRAINED model now sees ~2.7x more, noisier
    tokens, diluting the lexical overlap that gives a random-init encoder any recall at all.
    That is a legitimate consequence of more context, not proof that a shortcut was removed.

STILL OPEN: 55.8% of code-side sequences exceed 256 tokens, so this tested "meaningfully
more context", not "whole function bodies". A decisive test of body semantics needs 512+.
And a 512-way retrieval across 443 diverse repos may simply be solvable from name and
signature cues -- this experiment does not rule that out.

Measured memory, correcting the quadratic estimate: this model is shallow (dim 256, depth 4)
so linear-in-T terms matter as much as attention. bf16 at batch 512 measured 3,959 MiB at
max_len 96 and 10,105 MiB at 256 -- about 2.6x, not the ~7x a purely quadratic argument
predicts.

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

`/bulk` is 6.0 TB (5.5 TiB) at 14% used (measured: 741G/5.5T; `df -H` reports 6.0T/795G/14%,
`df -h` reports 5.5T/741G/14% -- same filesystem, TB vs TiB); the eviction threshold is 50%. Catalogue holds 11
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

3. NO SINGLE-CORPUS DOMINANCE. `retrieve` is 79.1% GooAQ / 19.8% NQ / 1.09% FiQA POST-DEDUP.

   CORRECTION -- the "evaluated on financial-domain FiQA" transfer test DOES NOT EXIST on
   the training path. build_splits shuffles the concatenated pool and takes all_pairs[:512],
   a uniform sample of the mixture, so expected FiQA content of the holdout is about 5.6
   items. recall@1 0.7480 is an IN-MIXTURE number on a ~79%-GooAQ holdout. The transfer test
   is described in a code comment and never implemented there. A second path, the untracked
   src/cogsyndelta/regions/retrieve.py, does implement a real BEIR-style FiQA-pool eval --
   but trains on FiQA alone. Two `retrieve` regimes now exist in the tree.

   Worse: dedup is UNDOING the balance cap. FiQA's anchor-level collapse (14,131 -> 5,498)
   accounts for 8,633 of the 8,634 rows build_splits removed, so the rule bites only the
   smallest source.

   And the gooaq cap is a PREFIX, not a sample: load_pairs returns as soon as `limit` is
   reached and the shuffle happens afterwards, so the 400,000 rows are the first 13.3% of
   the file in shard order. Unrepairable downstream.

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
| P2.5c | Fetch, with per-source caps that SAMPLE rather than truncate | no source exceeds its cap; balance recorded in the receipt | todo |
| P2.5d | **RESERVE composed-model material FIRST** | held out from every region before any region trains on it | todo — **BLOCKS P2.3** |


### SEQUENCING ERROR, corrected: P2.5d must PRECEDE P2.3

The clean unallocated staged pool is 244,761 rows, and P2.5's coverage list assigns every
one of them to a region. ALLOCATION IS IRREVERSIBLE -- once a region trains on a row, that
row can never serve the composed model's evaluation without contaminating it.

So training the new regions first (P2.3) and reserving afterwards (P2.5d) leaves the
composed model with NOTHING. The reservation needs about 23% of that pool and is free to
take today. It costs an unrecoverable amount to take later.

Recorded because the earlier ordering in this file had it backwards.

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
- All regions trained so far total: measured 3 x 16,021,248 + 22,905,216 = 70.97M params,
  ~284 MB fp32 for the four trained regions (see the scale-target note at the top of this
  file for how the 450 MB seven-region estimate relates; the earlier 348 MB figure at this
  spot followed from a stale 87M-parameter estimate).

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
