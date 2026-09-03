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

MOTOR REGION: explicitly deferred until after the Rust reimplementation (operator,
2026-09-02) -- not a region, not a P-row.

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
| P0.10f | `beats_untrained` gate had no floor — R6 investigation found `retrieve-20260902T203759Z.json`'s `untrained_baseline["recall@1"] == 0.0` made the gate trivially satisfiable by any nonzero trained recall@1 | done — 2f1202b (adds `_beats_untrained_gate` with a chance floor and a pre-registered margin, an `evaluate()` NaN guard, `untrained_baseline_seed` recorded, `test_guards_can_fail.py` DEFECT 6) |

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
| P0.8 | Checkpoint fingerprint does not cover CODE changes | a checkpoint from before a data-path change refuses to resume | wip — branch `fix/checkpoint-fingerprint` |
| P0.2 | Re-run benchmark battery on retrained regions | eval receipts written, anisotropy < 0.9 | todo |
| P0.3 | Re-run PTQ against retrained fp32 baselines | quant receipts, drop within 0.01 | todo |
| P0.4 | Fix csd-storage-tier verification | manifest honours the SAME excludes as the rsync | done — homelab SSD 1.5T -> 2.1T free |
| P0.4b | Sample-verify the cold copy is readable (parquet footers) since it is now the only copy | a stratified sample of at least 5% of parquet files per dataset (17 datasets under `tritter/pretrain`, 1654 parquet files total) opens with pyarrow and reports a row count; every file that fails is listed and re-fetched onto `gpu5080:/bulk`; the paths+sizes digest recomputed with `_manifest_cmd()` equals `7300c204` (self-consistency only, not a content check) | todo |
| P0.5 | Commit the train-dependency guard in csd-train-all.py | committed, gate green | done |
| P0.11 | Retrain code/compress/retrieve with the fixed guard so receipts carry the multi-channel report (see P0.10 extended channels above) | 3 receipts with `contamination.channels` present and `gated_channels` reported | done — `code-20260903T115858Z.json`, `compress-20260903T120818Z.json`, `retrieve-20260903T121603Z.json` all carry `contamination.channels`; `gated_channels` is `["pair_exact", "pair_content"]` (non-null) in all three, matching each other — checked directly against the compress receipt because it had been reported as null, and it is not. `reason` (aqua_rat) is outside this row's stated 3-receipt scope, but also picked up a receipt with the same guard at `reason-20260903T123431Z.json`; it still trains from a flat `limit: 4982` extra_source rather than the W2a union-burn ledger's fingerprint set, so a further re-run against the corrected clean pool remains open separately from P0.11 |
| P0.12 | Eval/quant receipts must record a checkpoint content hash, not a mutable path | receipt names a sha256 that matches the file it was computed from | todo |

**P0.1 detail.** 2f1202b's `_beats_untrained_gate` classifies `untrained_baseline["recall@1"] == 0.0`
as a broken eval, and both retrieve receipts on disk (`retrieve-20260902T165842Z.json`,
`retrieve-20260902T203759Z.json`) carry exactly that value while still recording
`beats_untrained.recall@1: true` -- a receipt the branch's own gate would now reject.
P0.1 is two-thirds green (code, compress) plus one receipt to regenerate; P0.11 covers
the retrieve retrain.

P0.11 evidence for `done`: the retrain against the fixed guard was dispatched and landed.
`code-20260903T115858Z.json`, `compress-20260903T120818Z.json` and
`retrieve-20260903T121603Z.json` each carry `contamination.channels` with `gated_channels`
`["pair_exact", "pair_content"]`. This supersedes the earlier `todo` evidence, which was read
off the pre-fix receipts (`code-20260902T210830Z.json`, `compress-20260902T211539Z.json`,
`retrieve-20260902T203759Z.json`) and a process check with no retrain running at the time.

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

**Storage policy (operator, 2026-09-02):** everything tiered to `gpu5080:/bulk` was acquired
from HuggingFace or another trusted source and is re-acquirable at this phase; a gap in the
cold copy is a re-fetch onto `gpu5080:/bulk`, not a recovery item or an operator decision --
sample-verify cheaply (P0.4b), re-fetch what is missing.

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
| P0.9c | `compress` graded/STS-B gate ran once at 11:36 (spearman 0.4956, `receipts/compress-20260902T153612Z.json`) then silently stopped when `csd-train-all.py` became the runner (`842db5e`); `graded_shards` is set only at `regions/compress.py:88` | a documented gate that silently stopped running, not one that never ran | done — graded gate restored in 7ab9abc/a5d2206 (guard mutation-tested), graded_shards is set by scripts/csd-train-all.py (run_region) as well as regions/compress.py; the P0.11 re-run of compress closed it: `receipts/compress-20260903T120818Z.json` carries `graded_held_out.spearman 0.7588` (n_pairs 1498) against `untrained_graded_baseline.spearman 0.0846`, `beats_untrained.spearman: true`, `code_revision 9df6526` |
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

**DECIDED 2026-09-02** (docs/design/LICENCE-FOR-OPEN-WEIGHTS.md, "Decision 2026-09-02",
commit `4c5dfb5`): GooAQ's NC reading is accepted and the corpus is kept -- "no commercial
use doesn't really apply to this case cuz this isn't the commercial product this is an open
weights model. it just changes the licensing from MIT to something that restricts commercial
use." A region whose input carries an NC term releases NC, not MIT (today: `retrieve`); the
COMPOSED model carries the strictest licence among all its inputs and sub-models (today: NC);
preferred deconfliction is the simplest scheme that satisfies every input, not maximal
openness bought with complexity. RIDER: a region MERGE inherits the most restrictive licence
of its parts -- merging `compress` and `retrieve` into one hippocampal `memory` region would
make that region NC.

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

## P4-P6 — Curriculum. Order is mandatory. Superseded by the ratified interconnect programme.

**Revision 3.2 of the region-taxonomy/white-matter-interconnect design
(`docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md`) was operator-ratified 2026-09-02** — OD-1
through OD-8 accepted as recommended, OD-9 answered (the audio licence audit is committed at
`docs/design/AUDIO-CORPUS-AUDIT.md`). **Revision 3.4 (2026-09-03) is the current text**, and it
adds DEC-52 to DEC-67 plus **eleven new rows**: **W7p** (placement harness — per-job VRAM budget,
target host, `placement{}` in every receipt), **W9i** (token-round-trip probe for the
latent-space invariant), **P2′f** / **P2′s** (dataset factory, synthetic-data contract),
**P5′b** / **P5′L** / **P5′m** (1B-per-submodel scale path, layer-sectioned training
candidate, moral corpus), **M0d** (deployment acceptance on the 5080 + 3090 Ti) and
**PRE-1/PRE-2/PRE-3** (the three fleet-side security prerequisites that block any autodev GPU
route — they block no CSD row). **Revision 3.3's E0/E1/E2 and M0 were never mirrored here and are
BACK-MIRRORED in this pass** — four rows the design doc already carried, which is this file having
been stale rather than the design doc having changed. **So this file gains fifteen row lines
(eleven new + four back-mirrored) against the design doc's eleven, and the two numbers are
different on purpose.** **OD-1 and OD-2 are ANSWERED** (autodev's spec now lives in `tzervas/csd-autodev`,
branch `docs/autodev-spec`); **OD-16 is new** (two diverged copies of `csd-lab-console`).
**W2c is DONE** — evidence at `docs/design/evidence/w2c-untrained-baselines-2026-09-03/`; the
≈0.40 code floor is retired, and `retrieve`'s τ_lo of 0.0000 is now a **blocking clause on
W2b** (DEC-62). The old router-shaped P4/P5/P6 rows are replaced below by
the design's own programme rows (§4.1), id-for-id, in this file's five-column format
(`blocked_by` is in the design doc, not repeated here — see §4.1 for the dependency chain). The
open operator asks OD-10 through OD-15 are not yet answered; see §8 of the design doc. This
table is where P4/P5/P6 used to be; the design doc is authoritative for task/gate text — treat
divergence here as this file having gone stale, not the other way round.

| id | task | gate | status | host |
|----|------|------|--------|------|
| **W0** | **Token surface — read "position latents": per-position hidden vectors, not discrete ids `[DEC-47]`. The method name `tokens()` is the API and is NOT renamed.** Split `TextEncoder.forward` at the pooling line; add `tokens()`/`pool()` to `TextEncoder` and `ViTEncoder`; split `native_dim` into `token_dim`/`pooled_dim` (A32); wire `visual`'s `[B,64,384]` first; **and re-instantiate §2.3's parameter table at the design's actual configuration** (4 controller heads, v1 participant list) so the total stops being `[I]`. Retrains nothing. | **Four properties that can actually fail** `[A10 fixed]`, because the `1e-5` equivalence test cannot: `pool()` will contain the *same* masked-mean expression `forward` contains (`src/cogsyndelta/regions/text_encoder.py:117-124` — **`regions/`, not `model/`**, and the quoted comments match verbatim at those lines [V] `[N10e fixed]`), so it compares an expression to itself through the same weights and a pre-existing pooling bug is present identically on both sides. **(i) Pad-invariance:** perturbing embeddings at masked positions must not change `pool()` — the exact bug class the code comment records having been found once (*"Averaging over padding pulls every short text toward the same vector"*). **(ii) Independent reference:** `pool()` equals a separately written numpy masked mean to 1e-6 — two independently-written implementations agreeing is evidence; one agreeing with itself is not. **(iii) Row-permutation, BOTH clauses** `[N10f fixed]`: permuting the **unmasked** positions of `h` **and `mask` together** leaves `pool(h, mask)` unchanged, **and must change `tokens()`**. Revision 2 wrote *"permuting the rows of `h`"* without permuting `mask`, which changes the masked mean — so the property was false as written and the test would have failed on correct code. The second clause is the half A10 asked for and revision 2 dropped: it is what confirms `tokens()` is not already the constant the central bet feared, and after §4.0 it is the cheaper of the two checks that bear on that question. **(iv) Batch-composition invariance:** the same item padded to 6 and to 66 positions pools identically (the regression measured at cosine 0.958). **Keep `‖pool(tokens(x)) − encode(x)‖∞ < 1e-5` on 512 items as a refactor smoke test, and stop claiming it audits the receipts.** | wip — branch `feat/w0-token-surface` | any / CPU |
| **W0c** | **DEC-40 — one `load_checkpoint()`, and the lint rule that keeps it the only one** `[N10c fixed]`. Revision 2 wrote DEC-40 and then routed it to "OD-7", which is the disposition of the untracked working files — a different item. It had no row, no owner and no gate, which is how the durable half of a security fix becomes a paragraph. This is that row. One `load_checkpoint()` in `src/`, the **only** `torch.load` in the repository, `weights_only=True` **hardcoded** (not a default, not a parameter, no override), manifest SHA-256 verified **before the file is opened**, every script importing it. **Owner: the operator, by hand — explicitly NOT autodev**, because DEC-40 is a control on the loader and OD-1 records that autodev's write scope plus merge authority can weaken a guard and its test in one change. No GPU, no dependencies. | **Three, all constructed to fire** `[N10c fixed]`: (i) a **CI lint rule** (ruff or a grep in `code-quality.yml`) fails the build on any `torch.load` outside that module — **verified by adding one in a scratch branch and watching CI go red**, because without that demonstration this is a policy and not a control; (ii) a **hash-mismatch refusal**, given the same negative-test treatment `tests/test_checkpoint_load_security.py` already gives the pickle refusal — flip one byte of a checkpoint, assert `load_checkpoint` raises **and** that nothing was unpickled; (iii) `grep -rn "torch.load" src/ scripts/` returns exactly one hit. | wip — branch `feat/w0-token-surface` | any / CPU |
| **W3r** | **Composite frame geometry + minimal renderer + checked-in fixture** `[N5 fixed]`. Revision 2 had W7v's round-trip gate consuming a rendered composite, the renderer as W3's primary deliverable, and W3 `blocked_by: W7v`. **This row is the W3-independent prerequisite both of them need.** It fixes the frame geometry once — `image_size 128`, `patch_size 8` ⇒ `n_patches 256` — records it as the number W7v re-shapes the encoder to and W3 renders against, ships a **minimal deterministic layout engine**, and checks **one hand-built 128×128 four-panel composite into the test fixtures** together with the layout description it was built from. CPU only, no GPU, no dependencies. | The fixture exists and is tracked; the renderer **reproduces it byte-identically** from its layout description (a deterministic layout engine that cannot reproduce its own output is not a ground-truth source); the geometry triple is recorded and propagated to §1.4, §2.3 and §6.1. **Verify by making it fail:** perturb one panel's position in the layout description by one pixel and assert the byte-comparison fails. **Until this row lands, "composite rendering" is NOT in the CPU-parallel column of §4.2** — a four-panel composite at an unfixed geometry is not work, it is a guess. | wip — branch `feat/w3r-composite-renderer` | 1080 Ti / CPU |
| **E0** | **Extract and PIN the memory-gate contracts. No GPU, no dependencies — do it with W2a** `[DEC-49] [OP: csd-episodic-store-required.md]`. Port the conformance and tiering tests out of `memory-gate` / `memory-gate-rs` into this tree as **executable contract tests against CSD's own store interface**, before a line of the store is written. The specific tests, named so the row cannot be reported as done by porting something easier: the **domain-isolation** fixture `test_domain_isolation_same_logical_key` [V, `tests/storage/test_store_conformance.py:150`], the **identity** fixtures `test_double_put_same_identity_last_write_wins` and `test_query_requires_domain_or_global_flag` [V, `:170, :188`], the **caller-cannot-mutate** fixture [V, `:299`], and the five Rust eviction fixtures `admit_spills_lowest_importance`, `gpu_hint_protects_from_spill`, `promote_returns_span_to_ram`, `retrieve_merges_tiers`, `disk_prune_drops_lowest` [V, `storage/tiered.rs:292-353`]. **Record which clauses were SILENT** and check the §8 gaps block against the result — E0 is also the row that can prove §8 wrong. | **The gate is that the tests exist and FAIL against an empty implementation, which is the only state in which a contract test is evidence.** Specifically: (i) all **nine named fixtures** (not twelve — the row names nine and the count is corrected to match, rather than leaving three unnamed slots a later pass could fill with easier tests `[S33-5 fixed]`) run and **fail red** against a stub store, and the run is recorded; (ii) **domain isolation** and **eviction order** are among them, so **a store built without them cannot pass this row and therefore cannot reach E1**; (iii) the parameterisation runs the suite against **three backends** — in-memory, SQLite, and the Qdrant fake — because the source suite's own value is that it is backend-independent [V, `tests/storage/test_store_conformance.py:78-299`, parameterized identically across the three backends per ADR-0002/ADR-0003] `[S33-9 fixed]`; (iv) **every clause §1.3 marks `[V]` maps to at least one ported test, and every clause it marks SILENT maps to none** — a test that appears for a silent clause means somebody invented a contract, and the row FAILS. | todo | any / CPU |
| **W1** | **CONSEQUENCE-3, CHEAP HALF.** Per-region participation-ratio effective rank of `pool()` vs `tokens()` (position latents, `[DEC-47]`); mean pairwise cosine; the `R×R` linear-CKA matrix. | **DONE 2026-09-02. Verdict: BET DEAD for all four production regions — and since 2026-09-02 that verdict is CONFIRMED BY W1d and no longer provisional** (participation-ratio ratios 0.66×, 0.78×, 1.00×, 1.30× against a ≤1.5× dead band); per-item flag fires for `code` and `retrieve`. **Entropy-effective rank on the same surfaces reverses the sign for all four (1.21×, 1.16×, 1.21×, 1.84×)** — a disagreement W1d adjudicated by measuring task performance instead of rank. Both columns and the non-production fifth row are in §4.0; evidence at `docs/design/evidence/w1-token-rank-2026-09-02/`. | done | 3090 `.98` |
| **W1d** | **CONFIRMATION of W1, run before the retrain was spent** `[A9 fixed]`. **Matched read-out probe:** the same small cross-attention read-out trained three times with identical params/steps/items — arm (a) over the final block's position latents, arm (b) over `pool()` **broadcast to the same `T`**, arm (c) over **penultimate-block** activations — plus within-item vs between-item variance and **both rank definitions on the probe's own activations**. | **DONE 2026-09-02, 116.5 s, read-only on the production checkpoints. W1 is CONFIRMED and no region is OVERTURNED.** `code` +0.00, `compress` −1.76, `vl_latent` −2.93 — all **CONFIRMED**, each passing its own seed self-check; `retrieve` −6.05 **CONFIRMED but the instrument is NOISY**, its seed self-check spreading **2.54 pp > the 2-pp rule**, so it is flagged and never cited as clean. **The sequence-blind arm BEAT the token arm in three of four regions.** **The penultimate fallback is DEAD** (arm (c) worse or noise everywhere; participation-ratio rank 2–4× lower for every text region) — **`L_token` attaches at the FINAL block**. **The token-aware retrains are LICENSED.** Caveats carried: an untrained cross-attention read-out ≈ uniform attention ≈ a linear function of `pool()`, and 600 steps of read-out training slightly **reduced** recall for `code`/`compress`. Evidence at `docs/design/evidence/w1d-readout-probe-2026-09-02/` with sha256s in §4.0. | done (confirmed) | 3090 `.98` |
| **W1b** | **Regenerate `reason`'s receipt** before it is frozen and before W7a retrains it. Its receipt was deleted; the r@1 0.0801 figure is prose only. | Receipt exists under `/akula-data/csd/receipts/`, reproduces r@1 ≥ 0.0801 **against an untrained baseline instantiated at a region-specific seed, not seed 0** (§4.0). It **must** carry `corpus.cap_sampling` **and** the drawn `pair_fingerprint`s (W2a), so this loss can never recur. **A frozen region with no receipt cannot be a baseline for anything and cannot be shown not to have regressed.** | todo | 3090 `.98` |
| **W2a** | **Ledger + recovery. Minutes, no GPU, no dependencies — do it first** `[A34 fixed]`. (i) **Verify** the `compose` allocation for `codeparrot/apps` + `deepmind/code_contests` landed — it has (§5.2, and §11 R1). (ii) **Recover the aqua_rat draw, as the UNION of both possible draws** (DEC-42) `[A5 fixed] [N1 fixed]`: compute **draw R** = `load_pairs(["reason/aqua_rat-raw/train.parquet"], ("question","rationale"), 4982, seed=0)` under today's reservoir code, **and draw P** = the **first 4,982 pairs `_iter_pairs` yields in shard order** — what the same call returned before `c42203c` rewrote the cap from prefix-truncation to reservoir sampling. Write `fingerprints(R) ∪ fingerprints(P)` to the ledger as `reason`-burned, record the source file's corpus fingerprint, and **mark the remainder clean**. | **Four checks, all of which must pass** `[N1 fixed]`: (i) the recomputed `fingerprint_corpus` of the parquet on disk today matches; (ii) two consecutive computations of **each** draw return identical fingerprints; (iii) the run genuinely used `seed = 0`; **(iv) THE FOURTH, AND IT IS THE ONE THAT CAN FAIL:** establish which code revision the `reason` run used — checkpoint/receipt mtime against `c42203c`'s commit time (2026-09-02 19:32:52 -0400), or a trainer revision recorded in the checkpoint — **and if it cannot be established, RECORD THAT IT COULD NOT.** Either way the ledger takes the union. **The gate that makes this failable:** W2a asserts `count(ledger ∩ draw R) = 4,982` **and** `count(ledger ∩ draw P) = 4,982` and **refuses to emit a ledger that omits either draw** — a fingerprint set carrying only one draw is a FAIL, not a partial pass. **If any of the four fails, the write-off stands and §5.1 reverts** — this is a test, not an assumption. Then re-derive §5.1 and §9.2. | wip — branch `feat/w2a-ledger-recovery` | 1080 Ti / CPU |
| **W2c** | **Measure the two untrained baselines the thresholds rest on** `[A23 fixed] [A13 fixed]`. (i) Instantiate `TextEncoder(vocab 50257, dim 256, depth 4, heads 4)` **at a region-specific seed**, run the `code` in-mixture eval, and settle whether the lexical floor is **0.2285** (the surviving receipt) or **≈0.40** (prose in four files, no artefact). (ii) Re-instantiate `retrieve`'s random encoder and establish a real untrained baseline for the 0.7480 figure, whose recorded baseline is **exactly 0.0000**. **W2c HAS RUN — 2026-09-03, receipt `w2c-untrained-baselines-20260903T124940Z.json`, evidence committed at `docs/design/evidence/w2c-untrained-baselines-2026-09-03/`** (script, `results.json`, `SHA256SUMS`). **The ≈0.40 code floor is RETIRED as unsupported; the measured numbers and the trace of where 0.40 came from live in that directory and are NOT restated here** `[DEC-62]`. | **BOTH CLAUSES MET, and the second answer is the consequential one.** Both numbers exist in receipts; **`τ_lo` is re-derived per bin in chance-normalised units** (DEC-36); the disagreeing documents are corrected. **The 0.40 branch did NOT fire** — the measured code floor is 0.2285 / 0.2344, so `apps`/`code_contests` are not excluded and the reserve's main source survives. **What DID fire is the branch nobody wrote: `retrieve`'s `τ_lo` is 0.0000**, which makes NSRS condition (1) barely satisfiable on that bin. **That is now W2b's blocking clause, not this row's** `[DEC-62]`. | **done 2026-09-03** | 3090 `.98` |
| **E1** | **Build the store: partition, byte-capacity, eviction, lifecycle — plus the DYNAMIC-CAPACITY PROBE on both GPUs** `[DEC-49]`. Implement against E0's failing tests: `(scope, domain, logical_key)` identity with the scope segment derived server-side (§9.9 B2); **byte** capacity from §8 gap (a)'s formula computed **per host per scheduler tick**; scored eviction `importance + gpu_resident_bonus − staleness(last_accessed)` with ties by older timestamp then key; the six lifecycle verbs with **refusing** backpressure at `max_in_flight = 32`; SQLite as the durability oracle with in-memory as the conformance oracle. **The store's two projections are NOT trained here** — they are white-matter parameters and E2 trains them. **This row builds a container and proves it is a correct container; it makes no claim about usefulness**, which is E2's job and is the whole point of DEC-50's step separation. | **Four, three of them constructed to fire.** (i) **E0's nine named tests go green** `[S33-5 fixed]`, and the diff that makes them green touches no test file — a fix that edits its own gate is refused. (ii) **THE DYNAMIC-CAPACITY PROBE, and it must produce two DIFFERENT, POSITIVE numbers** `[S33-6 fixed]`: compute capacity from a live `nvidia-smi` plus the scheduler's current KV and activation budgets on **the 3090 Ti (24 GiB) and the 5080 (16 GiB)**, assert the two **differ**, assert **each is `> 0`**, assert each is **respected** — a write that would exceed it triggers eviction rather than an allocation — and assert the value **changes when the active-region set changes**, since the 1080 Ti is preemptible and a capacity fixed at process start is not dynamic. **A probe returning one number for both cards has measured a constant and failed; a probe returning two DIFFERENT numbers where one is zero has also failed** — §9.11 found the residual may round to zero on the 5080, which is W10's deployment card, and "differ" alone passes that case. **Pre-committed branch, so this is not discovered at deployment:** if either card's `capacity_bytes` rounds to zero, gap (a)'s named alternative fires — a fixed **floor**, reserved for the store before the KV budget is computed, sized at `safety_margin`'s order (2 GiB) — and E1's receipt records which of the two (residual or floor) is active on each card, rather than the row passing on a store that has no capacity at all on the card it will actually run on. (iii) **CROSS-REQUEST FUZZ:** ≥10,000 interleaved writes across ≥100 partitions, then a read from every partition; **zero reads return a value written under another scope**, and the negative test is a deliberately mis-derived scope key that **must** produce a cross-partition read so the fuzz is shown to be capable of catching one. (iv) **EVICTION ORDER UNDER A CONSTRUCTED OVERFLOW:** build a working set whose scores are known and whose total exceeds capacity, then assert the survivors are exactly the top-scored set **and** that a GPU-resident low-importance entry outlives a host-resident higher-importance one by exactly the `+1.0` bonus — the behaviour `gpu_hint_protects_from_spill` pins upstream. | todo | 3090 `.98` **and** 5080 `.251` (both, by construction) |
| **W4** | **Merge `compress` + `retrieve` → `memory`, DESIGNED AS THE FIRST TOKEN-AWARE RETRAIN** so one run serves two rows `[A13 fixed]`. Shared trunk, two heads, joint loss **plus §4.0's `L_decorr` + `L_token` terms**. It is the natural first instance: the merge is a retrain that has to happen anyway, its two parents give it two independent regression checks, and a failure here is diagnosable as *merge* or *objective* by ablating one term. **It is also where DEC-24's shared embedding table first becomes achievable** (§6.2) `[A31 fixed]`. | **All five, pre-registered:** (1) `memory` matches or beats **both** parents on **both** parents' own gates (0.7070 and 0.7070-comparable, and 0.7480 **re-baselined by W2c**); (2) **`recall@10 > 0.20` and `MRR > 0.10`** on the FiQA BEIR eval against the full 57,638-passage pool — the numbers `retrieve.py:384` already states, which revision 1 adopted the eval for and then dropped; (3) **`memory` > BM25** on the same pool and qrels, from the same code path — not "BM25 reported alongside", *beaten*, because *"reporting a win for a loss"* is exactly what that file was written to prevent; (4) **`memory` > its own random-init baseline** on that pool; (5) **§4.0's retrain gate**, both clauses. Saves 15,890,176 params. | todo | 3090 `.98` |
| **W7a** | **Token-aware retrain of `language_code` and `reasoning`.** Same objective as W4, same harness, one region at a time. | **§4.0's retrain gate, per region, both clauses.** Plus: `language_code`'s r@1 0.9766 must not regress > 1 point, and `reasoning`'s regenerated 0.0801 (W1b) must not regress > 1 point. A region that clears the rank clause and fails the receipt clause is **reverted**, and the receipt says which. | todo | 3090 `.98` |
| **W7v** | **`visual`: token-aware retrain AND resolution rebuild** `[A4 fixed]`. `visual` **physically cannot ingest a composite image today**: `JEPAConfig(image_size=64, patch_size=8)` and `ViTEncoder` registers a **fixed** sincos `pos_embed` of size `cfg.n_patches = 64` (`model/vl_jepa.py:52-53, 182, 199` [V]), so `h = self.patch_embed(x) + self.pos_embed` is a **shape error** on a larger image, not a slower forward. Meanwhile §5.4's `visual × language_code` and `visual × memory` are **2 of 4 cross-faculty pairs = 1,024 of 2,048 eval items**, and §5.4 itself concedes they are *"unconstructible against 64×64 single-object tiles"*. Revision 1 scheduled no row to build the prerequisite. **This row does it, folded into the retrain that W1's result made mandatory anyway, so the marginal cost is the resolution change and not a separate run.** Re-shape to composite resolution (`image_size 128`, `patch_size 8` ⇒ `n_patches 256`, `pos_embed` regenerated or 2-D interpolated), continue-train, and — if OD-4 is decided that way — swap the corpus in the same run. | New `n_patches`, `kv_bytes_per_token` and `token_budget.max` recorded and propagated to §1.4, §2.3 and §6.1. Probe top1 and cifar100 transfer **do not regress by more than 1 point** against 0.0606 / 0.2625 **measured on the target encoder** (DEC-34), with the untrained baseline re-instantiated at a region-specific seed. **§4.0's retrain gate**, both clauses. **A composite frame renders, encodes and round-trips** — the negative test is that the 64×64 checkpoint **raises** on the same input. **The frame is W3r's checked-in 128×128 fixture, NOT a W3 deliverable** `[N5 fixed]`: revision 2's gate consumed the composite renderer, which is W3's primary deliverable, while W3 was `blocked_by: W7v` — a dependency inversion in which neither row could start. W3r breaks it. | todo | 3090 `.98` |
| **W7p** | **PLACEMENT HARNESS — a per-job VRAM budget, a target host, and receipts that say which** `[DEC-54] [OP: csd-training-placement-policy.md]`. Not a training row: the tooling every remaining GPU row already needed and none of them had. **Three deliverables.** (i) The launch path (`csd-train@`'s template today hardcodes `--batch 1280`, which is how `reason` OOMed) accepts **`--vram-budget-mib` and `--host`**, sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and a **per-process memory fraction** derived from the budget, and **refuses to start when the budget exceeds the target card's free VRAM at launch** — the refusal is the point, not the sizing. (ii) An **admission check for concurrency**: two or more submodel runs are co-scheduled on one card **only when the sum of their budgets fits**, with the sum printed; `reason` carries a **declared `exclusive: true`** and its measured configuration — **batch 512, `max_len` 256, alone** — so the one region known not to co-reside cannot be packed by a future scheduler that has forgotten why. (iii) Every receipt gains **`placement { host, gpu_name, vram_budget_mib, vram_peak_mib, concurrency, co_resident_jobs[] }`**, which is what makes DEC-54 auditable rather than aspirational. **No MIG on consumer cards: this is process-level PSEUDO-isolation, stated as such so it is never read as a security boundary** — acceptable because the fleet is single-tenant. | **Four, and three are constructed to fire.** (1) **A budget larger than the card's free VRAM REFUSES to launch** — construct it by requesting 30,000 MiB on the 5080 and assert nothing starts and nothing is written. (2) **A co-schedule whose budgets sum over the card REFUSES the second job**, verified by requesting two 14,000 MiB jobs on the 3090 Ti. (3) **`reason` submitted as a co-resident job is REFUSED on its `exclusive` flag**, and the refusal names the receipt that justifies it — the regression this row exists to prevent is exactly the one already paid for. (4) **Positive control:** two submodel runs whose budgets do fit **do run concurrently on the 3090 Ti and both produce receipts carrying `concurrency: 2` and each other's job ids** — without this clause the row is three refusals and no capability. **`vram_peak_mib` is measured, not declared**, so a budget that was wrong is visible afterwards. | **todo — no GPU dependency for (i)/(ii); do early, it unblocks nothing and de-risks everything after W4** | any / CPU + 3090 `.98` for (4) |
| **W2b** | **Reserve construction + NSRS admission + the guard** (replaces P2.5d). §5. **Only the BM25 condition (3), the licence audit, dedupe and rendering are CPU-parallel; `s_r` is inference through every frozen region including `memory`, so this row is GPU-blocked** `[A7 fixed]`. | `data/reserve/` + manifest exists; **thresholds frozen from a disjoint calibration split** and the would-have-been-rejected fraction of the graded set reported (DEC-36); every admitted row records `s_r` for all `r` and satisfies the NSRS filter **including the written condition-(2) exemption for the general bin** `[A21 fixed]`; **`RESERVED.jsonl` holds the union of SOURCE-ROW `pair_fingerprint`s, not composite hashes** (DEC-38); the ledger records `compose` for apps + code_contests; apps↔code_contests dedupe at cos ≥ 0.90 run and its 1,784 removals recorded; **keyed split assignment live** (DEC-39). **AND two refusals, both constructed:** (a) a training run seeded with one **directly reserved** row REFUSES TO START; (b) **a training run seeded with a row that a reserved COMPOSITE was built from REFUSES TO START** — the derived case, which is the path that actually leaks `[A6 fixed]`. **AND ONE MORE, ADDED IN REVISION 3.4 AND BLOCKING THE ROW** `[DEC-62]`: **admission for a CHANCE-FLOOR BIN is defined and pre-registered BEFORE this row runs.** W2c measured `retrieve`'s `τ_lo` at **0.0000** in chance-normalised units — its untrained r@1 *is* chance (1/512) — so condition (1) `max_r s_r < τ_lo` is **barely satisfiable on that bin** and admits by construction rather than by evidence. Adopt either an **absolute-margin floor** (`max_r s_r < chance + δ`, `δ` pre-registered per bin) **or a different normaliser**, record which and why, and **verify it can fail** on an item a single region trivially solves — under the current rule that item would be admitted, and the replacement rule must reject it. A bin whose admission condition cannot reject anything is not filtered. | todo — **BLOCKS P2.3** | 1080 Ti / CPU **+ 3090 for `s_r`** |
| **W3** | **Construct cross-faculty items.** §5.5: executable joins (**sandboxed**, §5.5a), rendered composites, shuffled mismatches. | ≥ 512 admitted items per cross-bin pair for eval **and** ≥ 5,120 per pair for train; every item passes W2b's filter **including the trivial-baseline condition**; **train/eval split at source-row granularity, split key recorded** (DEC-38); provenance chain recorded with **no generating model**, or the model named with revision and licence; **B1/B2 run on the reserve's own generators** and either diversified or waived with a dated waiver naming what the reserve therefore cannot measure `[A16 fixed]`; **the reserve's aqua_rat source-row share is printed against B1's 0.50 hard line and 0.40 operating cap** (§5.1) `[N8 fixed]`. | todo | 1080 Ti / CPU |
| **W1c** | **Anisotropy pre-flight.** Effective rank of the concatenated adapted union on 4,096 real mixed items, **on the retrained regions**. | **The statistic is named, because revision 2's threshold named none** `[N3 fixed]`: the number is **entropy-effective rank** (`cogsyndelta.eval.benchmark.effective_rank`), which is the definition behind the "8.7 of 128" prior this row compares against, and **`< 32 of 512` is in those units**. Participation ratio is recorded beside it, and the two are never compared across definitions — W1's pooled **PR**-ranks of 18–45 and its pooled **entropy** ranks of 115–130 (§4.0) are different quantities and revision 2 printed them as comparable. Comparable priors, restated in the right units: 8.7 of 128 on the pooled surface [V\*]; cross-region CKA ≤ 0.336 (§4.0). **If < 32 of 512 in entropy-effective rank, whitening goes into the adapters before W5.** | todo | 5080 `.251` |
| **W5** | **White matter v1, PHASE A (dense).** Workspace + adapters + frontal + `L_unify` + `rank_head` (DEC-41); all regions on; regions frozen **at their retrained checkpoints**; train on W2b/W3. | **G1** composed > B1, McNemar `p<0.01` Holm-corrected. **G2** no faculty's own-bin score drops > 1 point. **G0** composed > **B3** and > trained-**B0u**. **B0** (trained read-out) and **B0d** (dispatch control) recorded in the same receipt. **Phase-A collapse floor:** `min_r mean(a_r) ≥ η/R = 3%` with the per-region `a_r` histogram printed `[A15 fixed]`. **Overfit gate:** train/held-out gap < 5 points. **Every decision on the DEV half only** (§2.7.8). | todo | 3090 `.98` |
| **E2** | **Admit the store to the interconnect and RETRAIN WHITE MATTER — step 2 of DEC-50's protocol, and its first instance** `[DEC-50]`. Re-run white-matter phase A with `R = 5`: the store's `W_k`/`W_v` projections join the trainable set, regions stay frozen at their retrained checkpoints, the read simplex is re-floored at `η/R = 3.0%`, and the reserve's episode items (X7/X8) enter the training mix. **DEC-26's 2–5% interconnect cap is what makes this affordable:** the thing being retrained is the smallest trainable object in the mind, which is the property the modular design was bought for. | **Three, and the third is the one that can kill the row.** (i) **G2 holds** — no existing faculty's own-bin score drops more than 1 point against W5's receipt, measured on the **dev half**; the store itself is exempt from G2 and the exemption is §2.7.2's, not a new one. (ii) **The composed metric improves** on the dev half against W5's `R = 4` result — *improves*, not *does not regress*, because a participant that costs budget and returns nothing is a cost. (iii) **NON-ZERO ATTENTION MASS TO THE STORE:** `mean(a_store)` over held-out items is **≥ the collapse floor `η/R` = 3.0%**, with the per-iteration histogram printed. **On any failure the store is REVERTED** — removed from the participant set, `R` returns to 4, §5.4's E1-slip arithmetic fires (six pairs, ≥5 of 6, null 0.109), and the receipt records **which** of the three clauses failed, because "the store did not help" and "the store was never attended to" are different findings with different remedies. **Verify the gate can fail, in-contract** `[S33-7 fixed]`: pinning `b_store` to zero is not a state the system can legally reach — §1.4's `token_budget.min` is 8 and §2.4 floors `b_store` at `η/R · B_read = 0.03 × 256 ≈ 8` read tokens precisely so that "the store received no attention mass" is a measurement rather than a starvation artefact (§2.4). So the falsifier pins `b_store` **at that floor** (8 tokens, the legal minimum, not zero) and feeds episodes constructed with **no recall dependency** — turn 2 answerable from turn 1's context alone — and asserts `mean(a_store)` lands **below** 3.0% despite 8 tokens being available to spend on it: a store with nothing worth attending to should not be attended to, and clause (iii) must report FAIL on that input, not on an input the design already forbids. | todo | 3090 `.98` |
| **W5b** | **Write-back (conditioning prefixes).** DEC-17. | Conditioned regions' own-bin scores drop ≤ 1 point **and** the composed metric improves, **on the dev half**. **Pre-committed fallback:** on failure, disable write-back, record that phase-2 integration is workspace-internal only, **and apply §2.4's no-write-back consequences — either build the `Ĉ` variant or emit no `edges`, no `lockstep_groups`, and `topology: not demonstrated`** `[A20 fixed]`. Either result is a pass; not running it is the fail. | todo | 3090 `.98` |
| **W6** | **THE INTEGRATION TEST. This row opens the sealed half, once.** §2.7. | **G3** composed > **B2** (oracle late fusion) **and** > **B2t** (trained matched-capacity late fusion), McNemar `p<0.01` Holm-corrected. **G3′** the **DEC-37 synergy conjunction** — `Δ_A > 0` and `Δ_B > 0` and `I > 0` per pair, block-bootstrap CI by source row excluding 0 — on **≥4 of 6 pairs**, under **zero-ablation AND both content-swap arms**; any pair in the `REDUNDANT` quadrant is reported as a FAIL for that pair, never averaged away. **G0d** `I₀` (B0, trained read-out) and B0d's `I` both contain 0, **or the experiment is void and is reported void**. **I2′** ordered-pair severance CI excludes 0. **THIS GATE IS THE PROGRAMME'S PASS/FAIL.** Seal-break timestamp and code revision recorded. **The pair count is six and stays six** — DEC-48 defers audio, so no fifth participant reaches this row and the criterion is not restated `[S31-1 fixed]`. **DEC-47's inter-region assertion (W9 (iii)) is enabled for this run** and its result is printed in the receipt: an integration verdict obtained while some path re-tokenised would be a verdict about a different architecture. | todo | 3090 `.98` |
| **W8** | **Thalamic controller.** Phase B (distil), phase C (sparse), **phase D (task-loss)**. | **B:** `ρ(ŝ,a) > 0.6` held-out, **and no region was below the phase-A collapse floor**, or its `a_r` is excluded from the distillation targets and the receipt says so. **C:** sparse within 2% relative of dense at ≤50% of region-token FLOPs — *FLOPs only; buying accuracy here is a bug report against phase A*. **D:** beats C on the composed metric at equal-or-lower FLOPs, **or is reverted and the receipt says the scheduler is imitative**. `ctx_r ≥ ctx_min` floor verified by **constructing** an adversarial input that tries to starve a region. All accept/revert on the **dev half**. Train on the 3090; **measure on the 5080**. | todo | 3090 → 5080 `.251` |
| **W8s** | **Scheduler falsifiers S1–S5** and the **three-verdict receipt** (DEC-30 + T1). | All five reported whatever they say. The receipt carries `integration:`, `scheduling:` **and `trigger_sensitivity:`** as separate verdicts. **S5's rare-token threshold is pre-committed before the run.** | todo | 5080 `.251` |
| **W9** | **`Schedule` emission + DAG runtime + validator.** | **Two separate gates, because revision 1's two clauses contradicted each other** `[A29 fixed]` — a run that reproduces the dense path to 1e-4 has not skipped 30% of its iterations. **(i) Runtime correctness:** the DAG executor reproduces the **eager execution of the SAME `Schedule`** to 1e-4 — an implementation-equivalence test that can fail on a real bug. **(ii) Sparsity:** mean iterations down ≥30% **vs dense** at ≤1% score loss — a different comparison against a different reference. Plus: the two topology derivations agree on ≥95% of items **(void, and declared void, under W5b's fallback)**; 10,000-input adversarial fuzz emits **zero** budget-violating graphs; the validator **rejects a `Schedule` carrying a client-supplied `trace_id`**. **(iii) THE LATENT-SPACE INVARIANT, DEC-47:** a runtime assertion that **no inter-region path carries discrete token ids** — every tensor crossing a tract (a region's emitted `h_r`, its adapted form, the workspace latents, and DEC-17's write-back conditioning prefix) is a **float tensor at a declared `*_dim`**, and any integer-typed or vocabulary-indexed payload on such a path **raises**. **Verify it can fail, and this is the clause that makes the invariant a control instead of a convention:** construct the violation — hand the runtime a conditioning prefix built from token **ids** instead of latents, and a region wrapper that returns `argmax` indices from `tokens()` — and assert **both are refused**, in the `tests/test_guards_can_fail.py` pattern. A path nobody has watched refuse a token id is a design intention, not an invariant. | todo | 5080 `.251` |
| **W9i** | **THE TOKEN-ROUND-TRIP PROBE — DEC-47 stops being an assertion and becomes a detector** `[DEC-53] [OP: csd-latent-space-reasoning-invariant.md]`. W9's clause (iii) refuses a violation **someone constructed**; that demonstrates the refusal works and says nothing about the tracts nobody built a case for. This row instruments **every** tract in an emitted `Schedule` — each region's emitted `h_r`, its adapted form, the workspace latents, and DEC-17's write-back conditioning prefix — and produces a **per-tract census**: `dtype`, shape, value range, and a **round-trip signature** (whether the payload is bit-reachable from a vocabulary index, i.e. an integer-valued float tensor whose distinct-value count is ≤ the vocabulary size, which is what a silent `argmax`-then-embed looks like from the outside). **It reports a POSITIVE DETECTION rather than an absence of refusal**, which is the difference between finding an accidental re-serialisation and not having looked. | **Three, and the third is what makes the probe trustworthy** `[DEC-53]`. (i) **A clean run reports ZERO detections across every tract**, with the census printed per tract — a clean bill from an instrument that inspected nothing is the failure mode this clause exists to catch, so the census is a **deliverable, not a log line**. (ii) **A constructed round-trip is DETECTED, not merely refused:** wrap one region so `tokens()` returns `embed(argmax(logits))` — a float tensor of the right dtype and shape, which W9's dtype assertion **passes** — and assert W9i flags that tract by its round-trip signature. **This is the case W9's existing gate cannot see, and it is why this row is not folded into W9.** (iii) **A negative control on the detector itself:** a genuinely continuous tract whose values happen to be sparse must **not** be flagged, verified by feeding a quantised-but-continuous codec output (DEC-08's tract-codec slot) and asserting **no** detection — a probe that flags DEC-47's own sanctioned codec has been miswired, and DEC-47 explicitly permits quantised concept codes as a **codec** while forbidding them as a **bottleneck**. | todo | 5080 `.251` |
| **W10** | **Composed footprint + PTQ** (was P10.3). | Measured params/bytes at fp32 and after PTQ **on the deployment card**. Per region `D_sched ≤ 0.02` nats **and** composed metric drop ≤ 0.01. Predicted: **85,807,015 params, 343 MB fp32, 35.0 MB at 3.27 bits/param** — restated for the v1 participant list `[A24 fixed]` and to be replaced by W0's re-instantiation. | todo | 5080 `.251` |
| **P5′** | **PHASE 3 — whole-mind dynamic training.** Unfreeze. Joint objective over region corpora **and** the reserve, interleaved. Region growth, depth/width growth. | **G2** still holds per bin after unfreezing **AND** G3/G3′ still hold. Each new region beats its own untrained baseline **and** improves the composed metric **and** shows non-zero `Ĉ` to it. Growth increments are reverted if the composed metric does not improve monotonically. **Cross-host activation traffic ≤ 15% of step time** (DEC-29). **DEC-24 binds from here** `[A31 fixed]`. | blocked | 3090 + 5080, pipeline |
| **P5′q** | **QUANTISE-BEFORE-WHOLE-MIND, matched, at toy scale** `[OP: csd-quantize-before-whole-mind-training]`. Run the alternative ordering — per-region pretrain → **quantise (and/or fine-tune) the regions** → train the interconnect and run whole-mind dynamic training with regions held quantised — **against the canonical path** (pretrain → interconnect → whole-mind → fine-tune → quantise), matched on data, steps and seeds. The canonical path stays the default at toy size; this row is the experiment, not a substitution. **Design consequence if it wins:** the interconnect must **train against quantised region activations**, so its train-time inputs match deployment — which ties it to **DEC-27** (per-region PTQ sensitivity must be measured *before* regions are frozen, and `D_sched` becomes a training-time quantity as well as a release gate) and to **DEC-29** (it is the ordering that decides whether phase 3 fits at all on 24 GiB and 16 GiB). | **Report all four, per arm:** (1) **peak VRAM during phase 3** — the number the whole experiment exists to move; (2) composed metric vs the canonical run; (3) per-region drop from quantising *before* vs *after*; (4) **capability per parameter and capability per VRAM-GB**, not loss. Each arm gated on its own untrained baseline. **A win is only a win if it survives DEC-27's `D_sched ≤ 0.02` nats per region** — quantising earlier changes region output distributions earlier, so it shifts the schedule earlier too. | deferred to phase 3 | 3090 + 5080 |
| **P5′o** | **Memory-gate overlays** — the dynamic-paging seam's first client (DEC-33, §6.6) `[OP: csd-memory-gate-overlays]`. | **An overlay applied and then disconnected reproduces the base model's receipt metrics EXACTLY, and the disconnect is logged.** Plus: an overlay **refuses to attach** to a base checkpoint fingerprint it was not trained against (verified by constructing the mismatch); tiered-residency policy reports hit rate, page-in latency and VRAM held. | deferred | 5080 `.251` |
| **P5′s** | **Toy swarm vs comparable models** `[OP: csd-toy-swarm-experiment]`. An N-agent CSD swarm against (a) comparably sized conventional models at the same swarm size and (b) different swarm sizes, across all three GPUs (3090 Ti sm_86 24 GiB, 5080 sm_120 16 GiB, 1080 Ti sm_61 11 GiB). **It consumes two things this document produces: W9's `Schedule` emission and the per-region context/read budgets** — the harness reads those budgets rather than guessing them. | **Hard precondition: phase 3 is green.** Pre-quantisation is acceptable; anything before phase 3 measures a swarm of isolated regions behind a switchboard, which is the wrong thing. **Per-instance KV and activation budgets are computed UP FRONT and the swarm size derived from them, so a run never OOMs mid-experiment.** Capability per parameter and per VRAM-GB, with an untrained-baseline swarm and receipts per run, through the gpu-timeshare scheduler. | deferred | all three GPUs |
| **M0d** | **DEPLOYMENT ACCEPTANCE — the composed mind must actually fit and serve on the two cards it is for** `[DEC-67] [OP: csd-mycelium-downstream-goal.md, fleet-gpu-roles-and-scheduling]`. M0 carries the deployment shape in its **task text**; a shape in a task description is not an acceptance criterion, and the card that is actually tight — the **5080, 16 GiB** — had nothing scheduled to find out. This row is that check, and it is a **prerequisite of M0 rather than a part of it**, so a deployment failure is not discovered while a readiness suite is running. | **Four, and (ii) is the one expected to be tight.** (i) **The composed, PTQ'd mind loads and serves on BOTH the 3090 Ti (24 GiB sm_86) and the 5080 (16 GiB sm_120)**, at the declared context length, with measured peak VRAM printed per card. (ii) **The store's DYNAMIC CAPACITY (DEC-63) is reported SEPARATELY IN BYTES on each card** — and **`capacity_bytes = 0` on the 5080 is a PERMITTED but RECORDED outcome** that fires §9.11's residual-rounds-to-zero finding and forces DEC-63's operator question (residual claim vs fixed floor). A footprint that reports parameters and omits the store has reported half the deployment. (iii) **The 1080 Ti (11 GiB sm_61) serves the RAG helper** (embedding + rerank) and its latency is measured on the same queries M0 will use — it **stays in the deployment unless M0's arm (b) demonstrates native retrieval**. (iv) **Verify it can fail:** run the same load against a deliberately over-sized context length and assert the row reports **FAIL** rather than swapping, thrashing, or silently truncating — a deployment check that cannot report "does not fit" is a launch script. | **blocked — long arc** | 5080 `.251` + 3090 `.98`, RAG on 1080 Ti `.243` |
| **M0** | **MYCELIUM READINESS ASSESSMENT — the downstream bar, and the row that decides whether any of this was worth building** `[DEC-51] [OP: csd-mycelium-downstream-goal.md]`. Operator: *once CSD is dialed in and proven out on quality, assess whether it can be used to complete the **Mycelium** functional, value-semantic programming-language project.* A task suite drawn **from the Mycelium repository itself** — parse, typecheck, implement, refactor — with **ground truth taken from that repository's own tests**, so the labels are executable and were not written for this experiment. **Deployment shape, as ruled:** CSD on the **5080** and the **3090 Ti**; **RAG via the 1080 Ti** carrying a helper embedding/rerank model for what will not co-reside with CSD on the 3090 Ti. **Two arms, and the second is the open question this row exists to answer:** (a) CSD **with** the 1080 Ti RAG helper; (b) CSD **doing retrieval natively as a skill** — `memory` plus `episodic_store` end to end, no helper. **Whether (b) works is MEASURED, NEVER ASSUMED**, and a design that assumed it would be assuming the most load-bearing claim in the programme. | **Pre-registered before the suite is run, or the row is void.** (i) **A pass-rate margin, registered in writing with the comparison model named, its revision pinned and its licence recorded, BEFORE any CSD run** — CSD vs a comparable open model at comparable deployed size, both on the same tasks, same harness, same retrieval corpus. A margin chosen after seeing a number is not a gate. (ii) **Per-task-class breakdown** — parse / typecheck / implement / refactor reported separately, because a mind that parses well and cannot refactor is not a development engine and an average hides exactly that. (iii) **Arm (b) is reported whatever it says**, with the retrieval quality of the native path measured against the helper path on the same queries; `RAG native: NOT DEMONSTRATED` is a permitted and useful outcome, and it is the one that keeps the 1080 Ti in the deployment. (iv) **The untrained/ablated control is the composed mind with `episodic_store` reverted**, which is the cheapest way to find out whether the store earns its place on a real workload rather than on a constructed one. **Verify the gate can fail:** run the suite against the comparison model twice and assert the margin statistic distinguishes nothing — an instrument that reports a win for a model against itself has been miswired, and this programme has already paid once for a saturated instrument. | **blocked — long arc** | 5080 `.251` + 3090 `.98`, **RAG on 1080 Ti `.243`** |
| **P6′** | **Foundation / language trunk.** Unchanged in ordering — step 4, never step 1. Re-scoped: the causal LM is the **language faculty's generative head**, trained as a region under phase-1 discipline, not "foundation training of the whole model". | New prerequisite: **the general bin has a licence verdict** (FineWeb / C4 / Pile have none today [V\*]). **Deferred candidate corpus, recorded so it is not rediscovered:** official language documentation (Python/Rust docs already staged under `official-docs` in the tiered corpus, plus the RAG collections and the vault) — likely as a docs↔API retrieval bin or generative-head material rather than more contrastive pairs; it would require enrichment, structure extraction, version tagging, dedup against the code corpora and a per-doc-set licence verdict under the strictest-input rule. Revisit **only** when this row starts `[OP: csd-official-docs-corpus-idea]`. | blocked | 3090 + 5080 |
| **P2′f** | **THE DATASET FACTORY — the sourcing loop as one structured I/O contract** `[DEC-57] [OP: csd-dataset-factory-and-moral-corpus.md]`. **search → identify → capture provenance → licence verdict → ingest → emit a licence-compliant open dataset with full provenance**, drivable by an agent and producing a **human-visible provenance receipt per dataset**. **It GENERALISES what exists and rewrites nothing:** `scripts/csd-corpus-expand.py`'s catalogue entries already carry `provenance_group` and hold the **upstream licence verbatim as a field separate from the mirror tag** (which matters because the mirror tags were measured to disagree with upstream terms in ten cases, and all 75 sentence-transformers datasets declare none); the fetcher already **enforces `REFUSE`** rather than recording it; `LICENCE-FOR-OPEN-WEIGHTS.md` and `AUDIO-CORPUS-AUDIT.md` are the audit format; the corpus contract's **B1–B5** are the balance rules; NSRS admission and source-row fingerprints are the ingest filter. What is new is the **loop and its schema** — a machine-readable verdict per candidate, so an agent can drive the search half without a human re-reading every licence page. | **Four, and two are refusals.** (1) **The emitted dataset's licence is DERIVED by the strictest-input rule (DEC-31) and PRINTED**, not chosen — a mix containing one NC input emits NC, and the receipt names which input set it. (2) **A candidate whose upstream terms cannot be read at the primary source is `UNVERIFIED` and is REFUSED for training** (eval-only at most), constructed by pointing the factory at a source whose licence page 403s and asserting it does not ingest — *this project's own rule is that a doc does not beat a probe, and a factory that trusts a mirror tag contradicts the method that produced it.* (3) **A candidate whose distributor disclaims owning its own material is `REFUSE`**, constructed from a known case and asserted to write nothing. (4) **Positive control: one real dataset traverses the whole loop end to end** and its receipt reproduces the licence verdict, the provenance group, the row count and the dedup removals — a loop with three refusals and no successful traversal has not been demonstrated. | **todo — CPU, no GPU, and it is the long pole for DEC-56** | 1080 Ti / CPU |
| **P2′s** | **THE SYNTHETIC-DATA CONTRACT** `[DEC-58] [OP: csd-dataset-factory-and-moral-corpus.md]`. Generated data is admitted **only** under a contract, because *"generated data often carries a lot of garbage"* and the operator's stated preference is **extremely high quality and requirement-complete over large**. Four clauses: **the generator is NAMED in provenance** (model, revision, licence — extending §5.6's *"no generating model, or the model named"* rule from the reserve to every corpus); **contamination channels run against EVERY eval**, not only the one the batch was made for, with gated-channel counts printed; **a quality gate**; and **a capped share of any bin**, counted as a **B1 statistic over provenance groups** so one generator cannot become a monoculture wearing many names. | **Three, all failable, and the first is the one that matters.** (i) **THE QUALITY GATE CAN FAIL:** the synthetic bin must **beat a matched human-authored sample on the bin's own metric** — verified by feeding deliberately degraded generations (truncated, template-collapsed, self-similar) and asserting the batch is **discarded**, not warned about. A quality gate nobody has made fail is a preference. (ii) **A batch with no named generator is REFUSED at ingest**, constructed by stripping the field. (iii) **A batch that would push its bin's synthetic share over the cap is TRUNCATED OR REFUSED, and the receipt says which** — with the cap value printed beside the share, because a share reported without its cap is an opinion (§5.1's shape). | **todo — CPU** | 1080 Ti / CPU |
| **P5′b** | **1B PER SUBMODEL, THEN QUANTISE, THEN THE COMPOSED MODEL — the post-phase-3 scale path** `[DEC-56] [OP: csd-billion-per-submodel-scale-path.md]`. Sequence, in order and not skippable: **mid-size proven** (DEC-52's baseline, W6 green, PTQ on the deployment card) → **~1B parameters per submodel** → **quantise each region** → **train the composed model on the quantised regions** → the 30B direction (OD-6). **The ordering "quantise the regions, then train the whole" is what P5′q tests at toy scale first: if P5′q wins, this row is its production form; if P5′q loses, this row's ordering is wrong and must be re-derived before any 1B run is spent.** Proceeds **region by region under DEC-50** — scale one region, retrain the interconnect, then the unified pass — so a regression is attributable. | **Three, and the first is the gate that decides whether the row is reachable at all.** (i) **THE DATA GATE: of the order 10^10 tokens per region**, each with an **upstream-verified** licence under DEC-31 and satisfying **B1–B5** on provenance groups — reported per region **before** any 1B training run is scheduled, because a 1B region trained on a corpus that fails the licence bar is unreleasable and unrecoverable. DEC-57's factory is the instrument. (ii) **Per-region PTQ sensitivity is RE-MEASURED at 1B**: the 3.2675 effective-bits/param figure is a **toy-scale measurement and is not carried forward**; the composed model's bits/param budget is derived from the new number, and DEC-27's `D_sched ≤ 0.02` nats is re-evaluated per region at size. (iii) **DEC-50's three steps per region**, each with its own receipt, and **the monotone-improvement rule**: a scaled region that does not improve the composed metric is **reverted to its mid-size checkpoint**. | deferred — post phase 3 | 3090 + 5080 (+ 1080 Ti under P5′L) |
| **P5′L** | **LAYER-SECTIONED TRAINING — evaluated as a candidate, beside DEC-29, not instead of it** `[DEC-55] [OP: csd-training-placement-policy.md]`. Training **isolated sections of layers/weights on different cards** rather than the whole model at once, so all three cards can contribute at 1B+ per region where otherwise later phases are **locked to the 3090 Ti + 5080, or to the 3090 Ti alone**. **The operator's own caveat is preserved rather than smoothed away** — they flag it as worth considering while noting they may be wrong about some of these techniques — which is exactly why it is a candidate row with a comparison and not a decision. | **A comparison, pre-registered, or the row is void.** Measured **wall-clock and interconnect bytes per section boundary** against **DEC-29's region-granular pipeline-parallel cost on the same model and the same microbatch ≤ 256**, on this fleet's real 1 Gb/s link — **the standing ruling that cross-host DDP is the wrong tool at 1 Gb/s is the null hypothesis, and this row either beats it or records that it did not.** Adopted only on a win; a null result is a **useful, publishable outcome** that keeps the two-card lock honest. | deferred | all three GPUs |
| **P5′m** | **THE MORAL CORPUS — values at the training-data level, and whether that measurably works** `[DEC-59] [OP: csd-dataset-factory-and-moral-corpus.md]`. A **curated** corpus built to instil values at training time rather than filter them at inference time, with **its own CORPUS-CONTRACT entry**, its own provenance chain, and its own licence verdict under DEC-31. **Phase-3 / scale-up, explicitly NOT a toy row:** at 87M there is no behaviour to move and a null result would be uninterpretable, so running it early would burn the question. The operator's stated hope — that this improves model safety in a way industry could adopt — is a **claim to be measured**, which is what this row is for. | **Three arms, pre-registered, because a corpus that "clearly helps" without a control is the `residual_mlp` defect wearing a virtue** `[DEC-59]`. (i) **Held-out moral/safety probes**, constructed **before** training and **reserved under DEC-38/DEC-39** like every other sealed set — source-row fingerprints in `RESERVED.jsonl`, keyed split, and the training run **refuses to start** if a probe row is seeded. (ii) **Three matched arms:** the composed mind **with** the moral corpus, **without** it, and with a **size-matched neutral corpus** in its place — the third arm is what separates *"the values did something"* from *"more data did something"*. (iii) **The result is reported whatever it says**, including `moral corpus: NO MEASURED EFFECT`, which is a real and publishable outcome; and the probes' own contamination channels are run against the corpus, because a moral corpus that contains its own evaluation has measured nothing. | deferred — phase 3 / scale-up | 3090 + 5080 |
| **PRE-1** | **GATEWAY AUTHENTICATES ON EVERY ROUTE, BEFORE `proxy_upstream()` ATTACHES ANYTHING** `[DEC-60]`. **VERIFIED today, and it is worse than OD-2 described:** `dispatch_api_get()` performs **no caller authentication whatsoever**, and `dispatch_api_post()` authenticates **only `/api/apply`**; every other route — including **every proxied route, with `Authorization: Bearer {UPSTREAM_TOKEN}` attached by `proxy_upstream()`** — is served to any unauthenticated caller on `192.168.1.0/24` [V, `scripts/csd-lab-console:1012-1080`, survey `02-skeptic.md` S1/S2]. **OD-2's finding was that the gateway forwards the client credential; the real defect is that it never looks at one** — a textbook confused deputy with no identity to confuse, and the reason it has not bitten is that `PROXY_ALLOW` currently names five read-only routes. **Authenticate the caller on EVERY route, before `proxy_eligible()` and before any credential is attached**, and bind autodev's routes to the unix socket so they **404 on the TCP listener**. | **Two, both constructed** — this is the `tests/test_guards_can_fail.py` pattern applied to the gateway. (i) **An unauthenticated `GET` to an allowlisted proxied route is REFUSED**, asserted against the running unit, and **the upstream sees no request** — refusing after the proxy call is not refusing. (ii) **An authenticated caller whose identity is not on the route's allowlist is REFUSED**, so authentication is not mistaken for authorisation. | **todo — BLOCKS every autodev GPU route** | akula-prime, operator-owned |
| **PRE-2** | **TOKEN-GATE `POST :9108/v1/queue` AND MOVE THE WORKER OFF `kang`** `[DEC-60]`. **VERIFIED:** `_peer_ok()` returns true for **any** `192.168.1.*`, `172.30.*`, `172.32.*` or loopback source — **source-IP prefix only, no token** — and accepts `{kind, subject, extra{}}` into `enqueue_timeshare()`; 15 s later `gpu-timeshare-worker` runs the job **as `kang`** with `git/cabal-forgejo-agent` and `gpu/localai-api-key` **in its environment**, takes attacker-controlled `extra` fields **straight into argv**, and for the specialist kind **posts Forgejo comments under the agent identity** [V, `ansible/files/akula-health-exporter.py:520-596`; `scripts/gpu-timeshare-worker:60-170`; survey `01-threat.md` G17/G18/T4]. **IP-prefix authorisation is the "identity the client can set" anti-pattern, and this is the single most exploitable live path in the fleet.** Two structural changes: **(a)** a **bearer token bound to a producer identity** on `/v1/queue`, with `enqueue_timeshare` enforcing a **default-deny allowlist of `(identity, kind)` pairs**; **(b)** run the worker as **`svc-timeshare`, not `kang`**, with only the tokens that specific `kind` needs, injected per job via `secret exec`. Better still: make `/v1/queue` **loopback-only** and route cross-host handoff through the gateway like everything else. | **Three, all constructed.** (i) **An unauthenticated enqueue from a LAN address is REFUSED** and **nothing is queued** — asserted by reading the queue after the attempt. (ii) **An authenticated producer submitting an UNREGISTERED `kind` is REFUSED**, so a new job type gets no credentials until someone registers it. (iii) **`ps`/`systemctl show` confirms the worker's uid is NOT `kang`** and that its environment carries **only** the tokens its `kind` declares — verified by submitting one `kind` and asserting the other `kind`'s token is absent. **Until (iii) passes, the autodev sandbox's egress restrictions are moot**, because this path reaches the same credentials from the LAN. | **todo — P0; BLOCKS every autodev GPU route** | akula-prime, operator-owned |
| **PRE-3** | **REMOVE `kb_http`'s LOOPBACK FAIL-OPEN** `[DEC-60]`. **VERIFIED:** `_auth_ok()` contains `if self._loopback() and not token: return True` — it **fails OPEN on loopback when the token file is absent or empty**, while non-loopback with no token fails closed [V, `rag/integration/kb_http.py:351-360`; survey `01-threat.md` G15/T6]. Today the token file exists at 0600, so the window is small — **but the whole autodev design leans on the gateway and this store as its enforcement points, and a fail-open branch inside one of them is disproportionate to its size.** If `/akula-data/cabal/kb-http.token` is ever deleted, rotated to empty, or the service starts before the file is placed, **every loopback caller reads all three corpora unauthenticated**. **Delete the bypass; require the token unconditionally; `SystemExit` at startup when it is absent, so the failure is a dead service rather than an open one.** Related and cheap while the file is open: client-controllable audit detail (`X-Akula-Temporary: 1` makes the handler write `detail="temp"` instead of the real detail) is **client-controlled evidence** and should be recorded server-side [V, `kb_http.py:366-372`]. | **Two, both constructed.** (i) **Move the token file aside and restart: the service must EXIT, not serve** — assert the unit is failed and the port is closed, which is the whole difference between fail-closed and fail-open. (ii) **A loopback request with no `Authorization` header is 401** with the token file present — the state the code claims today, asserted rather than assumed. | **todo — small and independent; BLOCKS any autodev RAG route** | akula-prime, operator-owned |

If P5′q wins, the interconnect must train against quantized region activations, not fp32/bf16
ones, so its inputs at train time match deployment (operator, 2026-09-02) — carried over from
the old P5.1 row, now folded into P5′q above.

### Production phase: audio — deferred rows, kept specified (DEC-48)

Operator ruling 2026-09-02: audio waits for the composed mind to prove out without it first
("we can wait to add audio as a future feature once it proves out without audio... gotta walk
before we run"). The four audio rows are `blocked_by` the composed mind passing W6 without
audio, plus an explicit operator go — see the design doc for the seam-preservation rationale.

| id | task | gate | status | host |
|----|------|------|--------|------|
| **A0m** | **Audio corpus MANIFESTS** (DEC-43, DEC-46; `docs/design/AUDIO-CORPUS-AUDIT.md`). The manifest half of revision 3.1's A0, split out because it genuinely has no dependencies while the fetch half does `[S31-7 fixed]`. Per-source row: upstream URL, mirror id, **`mirror_tag` and `licence_upstream` as separate fields**, **`licence_upstream_source`** (the URL actually fetched **plus its fetch date**), **`grant_scope` ∈ {whole_corpus, metadata_only, code_only, unstated}**, the upstream text quoted verbatim, train verdict, redistribute verdict, NC/SA/ND/attribution flags, size **in hours**, provenance group, provenance red flags. **Scope covers BOTH mixes** — the `auditory` mix and the `speech_output` mix (bucket C) `[S31-8 fixed]`. | **Gate (i), rewritten because revision 3.1's version fired on the wrong rows** `[S31-3 fixed]`. It read *"a row where `mirror_tag` and `licence_upstream` hold the same string because only the mirror was read is a FAIL"* — string equality, which **FAILs every correctly audited row where the two genuinely agree** (LibriSpeech, MLS, AMI, MUSAN, VCTK, Hi-Fi TTS, LibriTTS-R, AISHELL-3, Expresso — the audit *celebrates* that agreement for LJSpeech) and **passes AudioSet**, the one row it was written for, because there the two strings also agree and the defect is the grant's *scope*. **The gate keys on the PROVENANCE OF THE READ, not on the value read:** a row **FAILS** when `licence_upstream_source` is absent, when its **host equals the mirror host**, or when its **fetch date is missing** — and separately when `grant_scope` is unset. `metadata_only` is what catches AudioSet; `code_only` is what catches CSS10 and Libri-Light. **Verify it can fail:** construct a row whose `licence_upstream_source` points at the mirror and assert **FAIL**; construct a correctly audited agreeing row and assert **PASS**. | wip — branch `feat/a0-audio-manifests` | 1080 Ti / CPU |
| **A0f** | **Audio corpus FETCH** under the audit's recommended v1 mix, with LibriVox capped as ONE provenance group. Wire the fetcher to A0m's manifest, fetch **the clean tier of both mixes**, and compute the balance numbers **on provenance groups, not dataset names**, **in hours** `[S31-14 fixed]`. | **Four, and (ii)–(iv) are constructed to fire.** **(ii)** the fetcher **refuses every source the audit marks BLOCKING**, verified by *constructing* the case: put `agkphysics/AudioSet` in a scratch manifest and assert the fetcher raises and writes **nothing**. **(iii)** **B1 and B2 are computed with LibriVox as ONE group, in HOURS, for BOTH mixes, and printed beside the same statistics computed per dataset name** — the per-name numbers are expected to look fine, and that is the point. **This clause is a PASS/FAIL, not a print** `[S31-2 fixed] [S31-5 fixed] [S31-15 fixed]`: the grouped max share is reported **against both thresholds** — **> 0.50 FAILS the row** (the hard line) and **> 0.40 is a recorded WARNING** (the operating cap), the shape §5.1 already uses for aqua_rat — and a **grouped `N_eff` below 3 FAILS**. The chosen per-group cap `C` is printed with them, because B1/B2 are functions of `C` and a receipt that omits it has reported an opinion. **(iv)** the LibriVox group's **per-speaker hour histogram** is produced and B5's max share printed. **(v)** every fetched source's licence tier is recorded so DEC-45's table can be **recomputed** rather than trusted. **Verify it can fail:** the constructed BLOCKING row, and a scratch mix built per dataset name that the grouped statistic must FAIL while the per-name statistic passes. | **deferred** | 1080 Ti / CPU |
| **A1** | **`auditory` region pretrain** (DEC-43). Spectrogram-frame patch tokens, masked-latent prediction, the same JEPA harness `visual` uses, over A0f's capped mix. Emits at the shared `tokens()` interface — **position latents, `[DEC-47]`** — so W0 is a prerequisite and the adapter is the one `visual` already has. | **Five, all pre-registered:** (1) **beats its own untrained baseline, instantiated at a region-specific seed and not seed 0** (§4.0); (2) **the contamination channels are present in the receipt** — `corpus.cap_sampling`, the drawn `pair_fingerprint`s and the source-row fingerprints; (3) **anisotropy recorded** — **entropy-effective** rank of `tokens()` on 4,096 real mixed items in W1c's units, printed beside participation ratio, the two never compared across definitions `[N3 fixed]`; (4) the eval split is declared **in the code** as `in-mixture` or `held-out-domain` **before the first receipt**; (5) the licence tier of the mix actually consumed is printed. **Verify the gate can fail:** run the eval against the untrained checkpoint and assert **FAIL**. | **deferred** | 3090 `.98` |
| **A2** | **`speech_output` head** (DEC-44). A second head on the generative trunk: BPE text tokens plus a discrete speech-token stream a synthesiser consumes. Trained on **bucket C** of the audit — which is why A0m/A0f's scope covers bucket C `[S31-8 fixed]`. **THE TRUNK IS FROZEN AND THE HEAD IS AN ADAPTER** `[S31-4 fixed]`: revision 3.1 left the trunk's status unstated while asking for a disconnect test that reproduces the text head's receipts *exactly*, and the two readings could not both hold — a trainable trunk makes "exactly" fail on the fourth decimal by construction, a frozen trunk makes it trivially true. The frozen reading is **adopted**, because it is the one that preserves reversibility for OD-15 and matches DEC-33's overlay discipline, which is only correct for a genuinely detachable thing. | **Five.** (i) **intelligibility** — **WER of a fixed, named, frozen reference ASR** over synthesised output on a **held-out** set beats the untrained head's WER by a margin pre-committed before the run; the ASR's model id, revision and licence are recorded and **the same ASR is used for every later comparison**. (ii) **the disconnect test, now well-posed:** with the speech head detached, the text head reproduces its own receipt metrics **bit-exactly on the logits** — which is achievable *because* the trunk is frozen — and the disconnect is logged. (iii) **the licence tier is printed** (OD-15). (iv) the untrained-head baseline is at a region-specific seed. (v) **balance:** the **grouped B1 of the consumed TTS mix is printed and a share > 0.50 FAILs the row** `[S31-5 fixed]` — four of six clean-tier `speech_output` sources (LJSpeech, CSS10, M-AILABS, Hi-Fi TTS) are **one LibriVox group**, leaving VCTK, AISHELL-3 and (NC) Expresso as the only levers, which makes OD-15 a **balance** question and not only a licence one. **Verify it can fail, in two directions** `[S31-18 fixed]`: feed the reference ASR the **untrained** head's output and assert FAIL — *and*, because that construction only proves the comparison is wired, feed the frozen ASR **silence or white noise** and assert **WER ≈ 1.0**. A stub that returns the reference transcript regardless of input passes the first check and then reports a spectacular improvement; this programme has already paid once for a saturated instrument. **A2 does NOT precede W2b** — with the trunk frozen it changes no region's weights, so the `s_r` invalidation rule does not reach it. | **deferred** | 3090 `.98` |
| **A3** | **The two audio cross-faculty item shapes for the reserve** (§5.4, §5.5). *`auditory × language_code`*: an A0 transcript-carrying source row yields (spectrogram frames, the transcript span, a **distractor span drawn from a different source row of the same provenance group**), and the item asks which text span the audio realises — built by **pairing existing aligned data**, §5.5(a)'s cleanest tier. *`auditory × visual`*: W3r's checked-in 128×128 composite rendered from a cleared text row, paired with audio of that same row read by a clean-tier speaker, plus a shuffled-mismatch negative. Both obey DEC-38's fingerprint granularity and DEC-39's keyed split. **When this row is picked up it re-opens §5.4's pair arithmetic, and the re-opening is pre-specified rather than left to the day:** adding `auditory` makes the participant set five and the all-pairs set **ten**, of which `auditory × memory` and `auditory × reasoning` have no shape here. **Pre-committed resolution, so nobody re-slices a criterion after a seal:** `auditory` enters as a **non-pair participant** unless A3 also declares shapes for those two, the pair set is held at **6 + 2 = 8**, and G3′'s criterion is restated **in the same breath** as **≥ 5 of 8** with its null rate `P(Bin(8,0.5) ≥ 5) = 0.363` printed beside it `[S31-1 fixed]`. | **Gate.** Items are **constructed** on CPU as soon as A0f lands and **admitted** — NSRS `s_r` for every region *including* `auditory` — only after A1. ≥ 512 admitted items per new pair for eval and ≥ 5,120 per pair for train; **the sealed-item allocation across pairs is recomputed and printed, and the recomputed table must print the cross-faculty sealed TOTAL alongside the per-pair figures so the two are visibly reconciled** `[S31-21 fixed]` — revision 3's per-pair column sums to 1,280 against a 2,560-item sealed half because shapes overlap pairs, and A3 inherits that ambiguity unless it prints both. Every item's provenance chain names its audio source and licence tier. **Verify it can fail, and the refusal now has a key to fire on** `[S31-9 fixed]`: revision 3.1 asked W2b to *"refuse the pair rather than score it against a random encoder"* and named **no field the filter could key on** — a randomly initialised encoder emits outputs and `s_r` is perfectly well defined, just low, so the construction produced a **number** and the gate could not fire. **W2b refuses any region whose `status` is not `built` AND whose receipt path is absent or whose checkpoint hash does not match the resident weights** — which gives §1.4's `planned` a second, mechanical job and gives the `s_r` invalidation rule the enforcement point it otherwise lacks. Construct it: attempt admission with `auditory` at `status: planned` and assert **refusal**, then with a receipt whose hash is flipped and assert **refusal** again. | **deferred** | 1080 Ti / CPU **+ 3090 for `s_r`** |

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
| P5′s | Swarm of CSD toy agents vs comparably sized models and swarm sizes, across all three GPUs — was P10.4, pointed at the design doc's id (§4.1) | runs only after phase 3 (whole-mind training) is green; KV cache and activation budget per instance computed up front so no run OOMs; receipts per run through gpu-timeshare; reports capability per parameter and per VRAM-GB vs the comparison models | deferred |
| P10.4 | VRAM-budgeted job packing so concurrent submodel runs never OOM or corrupt a run already in flight — the tool lives in its own repo, `tzervas/gpu-pack` (private; not CogSynDelta, per the operator's tooling-lives-in-its-own-repo rule), not here. CSD's side is the thin adapter: `src/cogsyndelta/util/gpu_budget.py` (`GPU_PACK_BUDGET_MIB` caps the CUDA allocator fraction, `GPU_PACK_PROBE=1` caps `pretrain_region` to a 20-step probe, `GPU_PACK_PEAK_MIB=<n>` reported to stderr on exit) plus example `JobSpec` files under `program/jobs/` | gpu-pack admits a probed job, launches it capped to its budget, and the peak it reports back matches what the driver measured within tolerance | in progress — adapter + specs landed on `feat/vram-budget-env`; gpu-pack's probe/admit/launch pipeline is the remaining piece, tracked in its own repo |

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

## P16 — Ternary (trits/trytes) implementation (deferred)

Operator, 2026-09-02: once the binary path is proven, a fully ternary (trits and trytes,
not bits and bytes) implementation is planned, for higher quality, performance and fidelity
at lower VRAM. ORDER, crawl-walk-run, do not skip a step: binary toy (now) -> binary big
model -> ternary toy -> ternary big model. Language policy as everywhere else in CSD: Python
first, always; Rust only after Python proves the idea. Prior art: `tzervas/embeddenator`
(Forgejo first, GitHub mirror) is NOT a model repo -- it is filesystem and low-level
primitives accelerating VSA, embeddings and ternary arithmetic on binary hardware, the
substrate a ternary CSD would run on. Read it before designing anything ternary.

| id | task | gate | status |
|----|------|------|--------|
| P16.1 | Ternary toy, after the binary big model is proven | ternary toy beats its own untrained baseline and matches the binary toy's composed metric within a stated margin at lower bits/param | deferred |

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

## P17 — Active learning: memory-gate overlays

Persona and skill specialisation lives in differential weight/activation overlays, never
edits to base weights (repos `tzervas/memory-gate`, `tzervas/memory-gate-rs`). Residency is
tiered -- VRAM / GPU cache / disk, scored on importance, size, age and utility, not
all-on-disk. This is the first real user of the dynamic-paging seam the interconnect design
leaves open (see REGION TAXONOMY above, "Dynamic paging is planned, not built").

| id | task | gate | status |
|----|------|------|--------|
| P5′o | memory-gate overlays: persona/skill differential offsets with tiered residency — was P17.1, pointed at the design doc's id (§4.1) | an overlay applied then disconnected reproduces the base model's receipt metrics exactly and the disconnect is logged; residency tiers (VRAM / GPU cache / disk) are scored on importance, size, age and utility and the policy is measured on hit rate, page-in latency and VRAM held; an overlay refuses to attach to a base whose fingerprint it was not trained against | todo |

## P15 — The long arc (operator vision, context not backlog)

Recorded so the near-term work stays pointed at it. None of this is scheduled.

**Visual scale-out.** Compressed screenshot corpora over Grokipedia and other sources, so
the model reasons about rendered information rather than only prose. Connects to the P12
argument that vision may be the cheaper channel per unit of decision-relevant information.

**Polyglot specialisation, deliberately narrow.** Rust, Python, TypeScript, quantum
languages, and eventually mycelium. The specialisation targets are software engineering, AI
engineering, architecture, research and design -- broad competence with deep spikes, not
uniform mediocrity.

**Official language documentation as a corpus.** Deferred. Raw material already exists (an
`official-docs` dataset in the tiered corpus, plus RAG and the Obsidian vault) but it needs
enrichment and a licence verdict before it is a candidate corpus. Revisit when the language
trunk becomes real.

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
