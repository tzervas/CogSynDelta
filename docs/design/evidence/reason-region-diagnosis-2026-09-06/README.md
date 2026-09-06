# Why the `reason` region is the weakest cell in the matrix, and what to run about it

**Ask.** Run E0 (fix the held-out split) and E1 (a reasoning-sensitivity battery with lexical
ceilings) in one CPU session, then E2 in the first GPU session. **Finding.** The matrix
battery is a bag-of-words instrument on every region. `reason` is the one region far below
its lexical ceiling, it is trained for far more epochs than the other text regions on a tiny
two-shape corpus, and its seed axis is contaminated by the split. E1's outcome routes the
next GPU minutes to E3 or E5.

## Scope and conventions

Date: 2026-09-05. Scope: read-only diagnosis of the CogSynDelta `reason` region (canonical id
`reasoning`, DEC-04). Every number is either copied from a receipt or design document with a
`file:line` reference, or computed on CPU from those receipts and the corpus by the scripts in
`scripts/` (outputs in `data/`). Anything unverified says so.

Tags used on numbers:

- **[C]**: computed by the scripts in this directory.
- **[I]**: an inference.
- **[OP]**: an operator memory-file fact.

Abbreviations used in citations:

| short | file |
|---|---|
| `TAX` | `/home/kang/code/personal/tzervas/CogSynDelta/docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` |
| `CC` | `CORPUS-CONTRACT.md` (same directory) |
| `MM` | `METRICS-METHODOLOGY.md` (same directory) |
| `TRAIN` | `scripts/csd-train-all.py` |
| `PRE` | `src/cogsyndelta/regions/pretrain.py` |
| `TE` | `src/cogsyndelta/regions/text_encoder.py` |
| `MX` | `program/matrix/csd-matrix.yaml` |

The table maps each citation prefix to its file.

## 1. What the region is meant to be, and what it is

### Meant to be

The taxonomy keeps `reason` as the *reasoning / logic* faculty, renamed `reasoning` (`TAX:643`;
DEC-04 `TAX:525`). Its role: *"Relates a problem to the STRUCTURE of its derivation. HONEST
LIMIT, unchanged: it recognises derivations, it does not produce them. Phase 2 keeps this
shape. Phase 3 replaces the objective with latent-step prediction so the region contributes
COMPUTATION to the workspace rather than a lookup"* (`TAX:815-821`).

The corpus contract requires multi-step problems with checkable answers, at least three
reasoning shapes, a difficulty spread and rationales (`CC:685-692`). §9.8 already records the
consequence of weakness: at r@1 0.0801 the region *"may never be attended to"*, falsifier
`a_reasoning` in W5 (`TAX:5797-5800`).

Regions exchange latents only (DEC-47, `TAX:568`). Recursive latent refinement is deferred
and is *"a training change, not an architecture change"* on the existing iteration loop
(`TAX:4612-4614`, `TAX:4654-4660`; `[OP: csd-recursive-latent-transformers]`).

### Trained and measured as

| aspect | what the artefact is | source |
|---|---|---|
| encoder | 16,021,248-parameter mean-pooled bi-encoder, dim 256, depth 4, GPT-2 vocabulary | `TE:18-19`, `TE:131` |
| embedding table | 12.87M of the parameters [C] | |
| objective | symmetric InfoNCE over in-batch negatives at fixed temperature 0.05 | `TE:184`, `TE:231` |
| optimiser | AdamW with warmup-plus-cosine | `PRE:1300`, `PRE:231-248` |
| batching | a sliding window over one shuffled order, never reshuffled | `PRE:1404-1407` |
| pairs | gsm8k `(question, answer)` plus aqua_rat `(question, rationale)` reservoir-capped at 4,982 | `TRAIN:454-486` |
| battery | the closed 512-pair diagonal, chance 1/512 (note below) | `MM:146`, `PRE:513-562` |
| gates | no graded gate, no token-aware terms | `TRAIN:485`, `TRAIN:326-336` |

The table describes the trained artefact and how it is scored. The battery ranks every
held-out anchor against every held-out positive, so the pool *is* the holdout.

The entry's note is *"retrieval of the matching solution, NOT step generation"*
(`TRAIN:475`). The region JSON says *"Does NOT generate reasoning steps"* with a dated B2
waiver for having only two shapes (`config/mind/csd-regions.json:143,149`).

The design intends a faculty that recognises derivation structure. The artefact is a
small-corpus surface-retrieval encoder scored by a diagonal that §2 shows to be a
bag-of-words instrument.

## 2. The measured picture

Five receipts: four matrix cells at code `a769409` (cell dirs `7bc2699`) and the 2026-09-03
baseline at `0026a3d`. All share corpus fingerprint `ca364a92d2c6c5fd259404e0ab6f52a1`,
4,000 steps, `max_len` 256, a 512-pair holdout. The learning rate is
`lr = 3e-4·sqrt(B/256)` (`TRAIN:284-300`).

| cell (train receipt) | B | seed | r@1 | r@10 | MRR | nDCG@10 | untrained r@1 | final loss | in-batch acc | train s |
|---|---|---|---|---|---|---|---|---|---|---|
| b256-s0 (`reason-20260904T145943Z`) | 256 | 0 | **0.1875** | 0.4199 | 0.2610 | 0.2891 | 0.0059 | 0.054 | 0.996 | 311.2 |
| b256-s1 (`…150513Z`) | 256 | 1 | 0.1523 | 0.4121 | 0.2436 | 0.2750 | 0.0078 | 0.034 | 1.000 | 310.5 |
| b512-s0 (`…151527Z`) | 512 | 0 | 0.1270 | 0.2676 | 0.1769 | 0.1915 | 0.0059 | 0.0098 | 1.000 | 599.6 |
| b512-s1 (`…152538Z`) | 512 | 1 | 0.1426 | 0.3359 | 0.2140 | 0.2350 | 0.0078 | 0.0044 | 1.000 | 595.9 |
| baseline `0026a3d` (`/akula-data/csd/receipts/reason-20260903T123431Z`) | 512 | 0 | 0.1152 | 0.2676 | 0.1714 | 0.1873 | 0.0059 | 0.0096 | 0.998 | 586.3 |

The table shows the held-out diagonal metrics and training end state of the five receipts.
Chance is 0.00195. W2c's region-specific-seed untrained baseline is 0.0039, τ_lo 0.0020
(`docs/design/evidence/w2c-untrained-baselines-2026-09-03/README.md`). One holdout item is
0.00195 of r@1.

### The seed axis is not a replication axis today: a harness defect

`cfg.seed` drives the reservoir cap (`PRE:887,895`), the shuffle that defines the holdout
(`PRE:928`), model init (`PRE:1262`) and the untrained-baseline seed (`PRE:1645`). The matrix
declares it: *"a seed change also resamples the holdout split"* (`MX:270`).

| seed | train pairs | duplicates removed | untrained r@1 |
|---|---|---|---|
| 0 | 11,811 | 132 | 0.0059 |
| 1 | 11,798 | 145 | 0.0078 |

The table shows what the receipts record per seed; the two seeds also list different
contamination examples. So the two seed-0 cells share one holdout and the two seed-1 cells
share another, and every cross-seed number mixes split, init and order.

| metric | seed 0 | seed 1 |
|---|---|---|
| r@1 | 31 | 5 |
| r@10 | 78 | 39 |

The table shows what survives: the *within-seed* batch comparison is on identical holdouts,
and the cells are the number of holdout items separating b256 from b512 [C]. The batch effect
is consistent in sign on r@10 and MRR and marginal on r@1. "b256 beat b512" is a
two-paired-sample claim, not a replicated one.

### Geometry and quantisation

| cell | anisotropy | alignment | PTQ drop |
|---|---|---|---|
| b256-s0 | 0.0023 | 1.21 | +0.0098 |
| b256-s1 | 0.0012 | 1.26 | −0.0156 |
| b512-s0 | 0.0008 | 1.59 | +0.0039 |
| b512-s1 | 0.0009 | 1.48 | +0.0059 |

The table shows geometry (eval receipts) and the PTQ drop (quant receipts) per cell, in the
cell order of NOTES.md. Anisotropy was recomputed to 0.0023 and 0.0008 for the seed-0 cells
[C]. PTQ is 3-bit on 15–17 of 17 tensors at 6.5–9.8×.

| quantity | value |
|---|---|
| `emb_std` | 0.062 vs 0.013 untrained |
| effective rank | 84–112 of 256 |
| uniformity | −3.53 to −3.75 |

The table shows the remaining geometry figures across the cells (eval receipts). No collapse.
The b512 cells spread more and align worse.

### Training dynamics

Every run ends at in-batch accuracy 0.996–1.000 with loss 0.004–0.054 against held-out r@1
0.12–0.19. At 11,811 pairs, 4,000 steps are **86.7 epochs at B=256 and 173.4 at B=512** [C].
The other text regions get 4.1–7.4 epochs [C].

| run | metric | peak | at step | final |
|---|---|---|---|---|
| b512-s1 | r@1 | 0.168 | 2,664 | 0.143 |
| baseline | same | 0.135 | 2,664 | 0.115 |
| b512-s0 | r@10 | 0.322 | 1,998 | 0.268 |
| b256-s1 | r@1 | 0.160 | 1,998 | 0.152 |

The table shows the runs whose held-out metric peaks before the final step, four of five
(receipt `history`). Only the last three checkpoints are kept (`PRE:1018`).

### Truncation

| source | mean | p90 | over 96 | over 256 |
|---|---|---|---|---|
| gsm8k answers | 95.3 | 153 | 41.2% | **0.31%** |
| aqua_rat rationales | 72.1 | | 21.3% | **1.14%** |

The table shows GPT-2 token lengths over the full sources and the share over 96 tokens and
over 256 tokens [C], matching `TRAIN:478-483`. Evaluating b256-s0 at `max_len` 96 moves r@1
0.1875 → 0.1836 [C]. Not binding at 256.

### The lexical ceiling

| scorer, reason's 512 holdout | r@1 | r@10 | MRR | r@1 gsm8k (320) | r@1 aqua_rat (192) |
|---|---|---|---|---|---|
| TF-IDF cosine | **0.873** | 0.945 | 0.896 | 0.994 | 0.672 |
| BM25 | 0.850 | 0.916 | 0.876 | 0.988 | 0.620 |
| Jaccard over numbers only | 0.428 | 0.732 | 0.532 | 0.413 | 0.453 |
| trained b256-s0 | 0.1875 | 0.420 | 0.261 | 0.184 | 0.193 |
| trained b512-s0 | 0.127 | 0.268 | 0.177 | 0.103 | 0.167 |
| untrained, seed 0 | 0.0059 | 0.033 | 0.021 | — | — |

The table shows lexical scorers against the trained and untrained encoders on the exact
seed-0 holdout (fingerprint reproduced through `build_splits`) [C].

| region | model r@1 (b512-s0 receipt) | TF-IDF r@1 | BM25 r@1 | model ÷ TF-IDF |
|---|---|---|---|---|
| code | 0.980 | 0.982 | 0.996 | 1.00 |
| compress | 0.713 | 0.713 | 0.701 | 1.00 |
| retrieve | 0.721 | 0.770 | 0.754 | 0.94 |
| reason | 0.127 (b256-s0: 0.1875) | 0.873 | 0.850 | 0.15 (0.21) |

The table shows the same scorers on the other regions' exact seed-0 holdouts (fingerprints
reproduced) [C]. Three regions sit at the bag-of-words ceiling of their own battery. `reason`
sits at 15–21% of it.

Per-item diagnostics on b256-s0 [C]:

- Scrubbing digits from both sides drops r@1 0.1875 → 0.1426 (r@10 unchanged), so about a
  quarter of the hits ride on number matching.
- Ranking within one source only gives 0.194 (gsm8k) / 0.203 (aqua_rat), equal to the full
  pool, and the top-1 is from the anchor's own source 91% of the time, so source confusion is
  not the failure.
- The hit rate is 0.19 on items BM25 gets right and 0.17 on items it gets wrong, so the
  encoder is uniformly weak rather than a worse lexical matcher.
- The median rank of the true positive is 18 (b256-s0) vs 107.5 (b512-s0).

## 3. Ranked hypotheses

### H1 — The battery measures lexical pair retrieval, not reasoning, and `reason` is the one region that has not learned the surface

- *For:* TF-IDF 0.873 / BM25 0.850 on the same items. gsm8k pairs are near-trivial lexically
  (0.994) because the worked answer restates the question's entities and numbers. Every
  other encoder sits at its ceiling, so the matrix's weakness ranking is a ranking of
  progress toward bag-of-words matching [C]. The objective is lookup by the design's own
  words (`TAX:818`, `TRAIN:475`).
- *Against:* none in the receipts.
- It changes what "improve reason" means: the number cannot pass ~0.87, and reaching it would
  not be reasoning.

### H2 — Memorisation of a tiny corpus at 87–173 epochs; the batch effect is an epoch effect

- *For:* train accuracy 1.0 at loss ≤ 0.01 against held-out 0.12–0.14 at b512. Peaks before
  the end in 4/5 runs. Doubling the batch doubles the epochs and lowers the score with lower
  loss. On `memory` (505k pairs) batch 256→512→1280 *raised* r@1 0.600→0.815→0.854 (DEC-68,
  `TAX:589`).
- *Against:* two paired samples, one of 5 items. The seed axis is contaminated (§2).
- Confounded with H6 until E2.

### H3 — Small, two-shaped corpus

- *For:* 11.8k pairs vs 214k–505k. N_eff 1.92 with a waiver. Deduction and multi-hop absent
  (`CC:708-717`). Clean reach 1.8% of target without a generator (`TAX:594`). On all 97,467
  aqua_rat rows 4.0% of rationales are under 8 words, 17.0% of questions and 6.7% of
  rationales are duplicates [C].
- *Against:* the score is equally bad on both sources, so imbalance is not what fails today.
  More of the same shape mostly feeds H1's surface.

### H4 — Lookup by design (DEC-04)

Not a cause of the low number but the reason it is the wrong target. It decides E5.

### H5 — `max_len` truncation

Ruled out at 256 (0.31% / 1.14%; eval at 96 barely moves) [C]. It would bind at the fleet
default 96 (41.2%), which is why the row overrides it.

### H6 — Harder in-batch negatives at τ = 0.05 with larger batch

- *For:* the mechanism exists (`TE:183-245`).
- *Against:* DEC-68's curve runs the other way. The batch doubling also doubles epochs. b512
  models are more uniform and less aligned, the memorisation signature [I].
- Below H2.

### H7 — Too few steps

The opposite: loss is at zero and held-out declines after ~2,600 steps.

### H8 — Encoder too small

The same 16.0M encoder reaches the ceiling on three corpora. Capacity does not separate
reason, and a larger model contradicts the small-and-capable rule
(`[OP: csd-design-philosophy]`).

### H9 — Collapse / anisotropy

Ruled out (§2).

### H10 — No graded gate

True (`TRAIN:485`). A measurement gap, not a cause: reason has no instrument a bag-of-words
scorer cannot solve. It becomes E1.

## 4. Pre-registered experiments

### Design rules applied to every experiment (operator rule, 2026-09-05)

- (a) Every arm of an experiment shares the *same training seed* (init, data order, masking)
  and differs in exactly one variable. Replication across seeds {0, 1, 2} is a separate
  axis, compared *within* seed as paired differences, never pooled across seeds.
- (b) The held-out split is *fixed and independent of the training seed* (E0).
- (c) Parameters stay at 16,021,248. Nothing crosses a region boundary as tokens (DEC-47).
  Untrained baselines use the region-specific seed (W2c).

| item | cost |
|---|---|
| b256 run | 311 s |
| b512 run | 600 s |
| eval | 1.4 s |
| PTQ | 11 s |
| card memory at b256 | ~7.1 GiB |
| card memory at b512 | ~13.4 GiB |

The table shows the per-run costs from the receipts on the 3090 Ti (card memory from
cell.json). So b512 stays the ceiling at `max_len` 256: b1024 extrapolates to ~26 GiB [I],
and the b1280 OOM is `[OP: csd-baseline-receipts-2026-09-03]`. Go/kill margins are set
against 1 item = 0.00195 and a binomial 95% half-width of ~0.034 at r@1 ≈ 0.19.

### E0 — Fix the split (prerequisite, 0 GPU-min)

*Change:* separate the split from the training seed. Either a `split_seed` field (default 0)
that alone drives the reservoir cap and the holdout-defining shuffle, with the *remaining*
training pairs reshuffled by `cfg.seed` for order, or a canonical split file per
`(region, corpus fingerprint)` holding the holdout's pair fingerprints, written once and
hashed into every receipt as `split.sha256`. A guard recomputes membership before train/eval
and refuses on mismatch.

*Acceptance, three clauses:*

- (1) two runs with different training seeds report byte-identical held-out membership and
  `split.sha256`;
- (2) the untrained baseline at the region-specific seed is then identical across training
  seeds;
- (3) the guard is shown to fire on a split with one pair swapped
  (`tests/test_guards_can_fail.py` pattern).

*Comparability choice:* pin the canonical split to today's seed-0 draw (reproduced on CPU
with matching fingerprint, 11,811 / 132 [C]). That keeps the two seed-0 cells, the
2026-09-03 baseline and W2c's baselines comparable on the eval axis. The seed-1 cells are
retired from comparison.

### E1 — Reasoning-sensitivity battery and lexical ceilings (CPU only, 0 GPU-min)

*Change:* for each held-out gsm8k pair with ≥ 2 calculator annotations `<<a op b=c>>` (299 of
320 on the fixed split, mean 3.18 annotations [C]) construct K = 4 corrupted derivations by
editing one operand or result in one annotation and propagating to the `####` line when it
is the final value, deterministic corruption seed. An item is to rank the true derivation
among {true, 4 corruptions}, chance 0.20. Add `lexical_ceiling.{tfidf,bm25}.recall@1` to
every region's eval receipt.

*Controls:*

- (a) wrong-problem control: true derivation vs four other problems' derivations, on which
  TF-IDF must score ≥ 0.90, proving the instrument is solvable where the signal is lexical;
- (b) TF-IDF and BM25 on the corrupted battery must land within ±0.05 of 0.20, proving it is
  lexically unsolvable;
- (c) the untrained encoder at the region-specific seed.

*Arms:* the three comparable existing checkpoints (b256-s0, b512-s0, baseline) under two
corruption seeds. No training seed is involved.

*Go/kill:* any checkpoint ≥ 0.30 means the encoder has derivation sensitivity and E3 is next.
All ≤ 0.25 means the contrastive objective learned nothing about structure and E5 is promoted
ahead of E4. Either way the diagonal r@1 on reason's card is demoted from a gate to a
*lexical-ceiling fraction* (model ÷ TF-IDF).

### E2 — Epoch-matched batch ablation (~53 GPU-min)

*Change:* batch {256, 512} × steps {2,000, 4,000}, `eval_every` 200, on the fixed split.
Within each training seed s ∈ {0, 1, 2} all four arms share s and differ only in
(batch, steps). Existing (256, 4000, s0) and (512, 4000, s0) are reused; new: (256, 4000) and
(512, 4000) at s1, s2 (2×5.2 + 2×10), (256, 2000) ×3 (3×2.6), (512, 2000) ×3 (3×5.0).

*Metric:* diagonal r@1/r@10/MRR at the final step and at the best eval point, plus E1's
battery.

*Predictions:*

- H2: (512, 2000) ≈ (256, 4000) within 0.03 r@1 and 0.04 r@10 (same 87 epochs) and both beat
  (512, 4000), in every seed.
- H6: (512, 2000) trails (256, 4000) by > 0.05 r@10 with the same sign in all three seeds.

*Go/kill:* H2 confirmed means the matrix row moves to an epoch budget and "b256 > b512" is
retired as a batch finding. H6 confirmed means a temperature arm (0.05 vs 0.10) joins E3. If
peak − final > 0.02 r@1 in ≥ 4 of the 6 4,000-step runs, best-checkpoint retention is added
(`PRE:1018`).

### E3 — Structure-sensitive negatives in training (~47 GPU-min)

*Change:* each training gsm8k pair with ≥ 2 annotations (6,675 of 7,153 [C]) gets one
corrupted derivation as an extra candidate. The loss stays symmetric InfoNCE over B positives
plus B corrupted candidates, so the objective rewards telling a right derivation from a nearly
identical wrong one without leaving the latent contrastive shape. aqua_rat rows carry no
annotations and get none, recorded as a B1 caveat on the battery.

*Control (same seed):* the identical 2B-candidate run whose extra candidates are other
problems' derivations, separating "more negatives" from "structure-sensitive negatives". Arm
and control share seed s for s ∈ {0, 1, 2}. Batch 256, steps from E2's winner, ~7.8 min per
run (1.5× the b256 run for the doubled candidate tower [I]) × 6.

*Metric:* E1's battery (primary), diagonal r@1 as regression guard.

*Go:* corrupted acc@1 ≥ control + 0.10 and ≥ 0.35 in every seed with diagonal r@1 ≥
control − 0.03. *Kill:* < control + 0.05 in any seed.

### E4 — A third provenance group under the synthetic contract (~16 GPU-min plus CPU)

*Change:* run the DeepMind `mathematics_dataset` generator, the one settled-grant source
(`TAX:4025-4028`; catalogue `DATASET-FACTORY-CATALOGUE-2026-09-03.md:431`), at category caps
to ≤ 0.40 of the bin under DEC-58's four clauses (`TAX:579`). Its shape is question → exact
answer, not question → derivation, so this arm tests whether numeric-shape volume transfers to
a derivation battery. A clean null is a result. A conditional arm adds StrategyQA
decompositions rendered as rationales (2,780 rows, catalogue `:409`) for the multi-hop shape.

*Control (same seed):* E2's winning two-source configuration on the same fixed split, which
E0 makes possible. Today `build_splits` reshuffles the union (`PRE:886-959`), so adding a
source moves the eval set. Seeds {0, 1, 2}, 5.2 min each.

*Go:* E1 battery ≥ control + 0.05 and diagonal r@1 ≥ control − 0.03 in every seed. *Kill:*
otherwise, recorded as "volume of a different shape does not transfer".

### E5 — Latent-step prediction, the DEC-04 phase-3 objective as a toy pre-registration (~70 GPU-min)

*Change:* split each gsm8k derivation into steps (lines; mean 4.49, all held-out items ≥ 3
[C]), encode (question + steps ≤ t) with the same trunk and predict the latent of step t+1
through a small predictor with an EMA target encoder, the JEPA family `vl_pretrain.py`
already uses. The region then emits a *computed* next-step latent rather than a looked-up
one.

*Battery:* rank the true step-(t+1) latent among {true, one corrupted step, three steps from
other problems} by predictor output, chance 0.20. E1's pooled battery is the regression guard.

*Controls (same seed):*

- E3's contrastive encoder on the same trunk;
- an untrained predictor at the region-specific seed;
- collapse guards (`emb_std`, effective rank);
- a sequence-blind arm predicting step t+1 from the question alone, in W1d's pattern, because
  the g14 toy learned a K-independent shortcut and its falsifier fired
  (`[OP: csd-recursive-latent-transformers]`).

Seeds {0, 1, 2}, ~11 min per run [I], six runs.

*Go:* predictor acc@1 ≥ 0.40 and ≥ sequence-blind + 0.10 in every seed. *Kill:* ≤
sequence-blind + 0.05 in any seed.

*Where recursion enters:* feeding the predictor its own output (predict t+2 from the predicted
t+1 latent, acc@K over K predicted steps) is the deferred loop-in-latent-space direction
(`TAX:4654-4660`). E5 declares the seam and measures K = 1 only.

## 5. What has to change in the corpus contract and dataset factory

1. `CC:680` still reads *"planned"* / *"nothing trained"*; five receipts exist. The section
   should carry the measured battery *and* its lexical ceiling. 0.19 reported without the
   0.87 beside it invites the wrong fix.
2. A **reasoning-sensitivity gate** (E1) as reason's graded gate, as STS-B is `compress`'s,
   with the rule that a bag-of-words scorer must sit at chance on it. Corruptions are
   mechanical edits of MIT-licensed human rows and add no licence term. They are declared in
   provenance anyway.
3. A **fixed split** (E0) hashed into receipts, without which no corpus change is comparable.
4. **Sourcing the missing shapes.** The generator (Apache-2.0) is settled but numeric-shaped.
   StrategyQA (multi-hop, `:409`) and ProofNet (371 pairs, `:405`) are VERIFIED permissive
   and small. PRM800K (MIT, `:408`) gives ~800k human step-correctness labels, natural hard
   negatives for E3, but its problems are MATH's, and the documents disagree: `CC:711-713`
   and `csd-regions.json:149` call the MATH family DMCA-rejected, while the pass-1 catalogue
   (`:403`, `:432`) records `hendrycks_math` PERMISSIVE_OK verified at the primary LICENSE.
   That needs an operator ruling before either is admitted. OpenR1-Math-220k (`:407`) is
   Apache but MODEL-OUTPUT-TERMS (OD-18 item 2).
5. **Synthetic with gates:** generator rows only under DEC-58's four clauses at ≤ 0.40 share
   as one provenance group (`TAX:579`; catalogue `:694`).
6. **Out of scope:** NC sources (0.8 points, `TAX:594`), HotpotQA (share-alike, collides with
   retrieve's Wikipedia lineage, `CC:713-716`), the MATH family until ruled, a larger encoder,
   any decoder or token stream on the region (DEC-47), any interconnect change before W5.

W1b (`TAX:2646`) asks for ≥ 0.0801 with a region-specific-seed baseline. The 2026-09-03
receipt gives 0.1152 with a seed-0 baseline and W2c supplies the region-seed figure
separately. Closing the row by citing both is the operator's call.

## 6. Recommendation

Run **E0 then E1 in one CPU session**. E0 is the prerequisite every later comparison needs
and costs nothing. E1 uses only the existing checkpoints and corpus and settles the question
every GPU run depends on: whether the encoder has learned anything about derivation structure
or only 15–21% of a bag-of-words surface.

E1's two controls make it failable both ways. The wrong-problem control proves the instrument
is solvable, and the TF-IDF-at-chance control proves it is not solvable lexically. Its
outcome routes the next 50–70 GPU-minutes to E3 (structure-sensitive negatives) or E5
(latent-step prediction, which DEC-04 already schedules).

E2 should follow in the same GPU session. It is cheap, identical-seed paired, and decides
whether the matrix row moves to an epoch budget, which halves every later reason cell.

## Reproduction

- `scripts/extract_receipts.py` → `data/reason_receipts.json`.
- `scripts/analyse_reason.py` (token lengths, exact holdout via `build_splits` with
  fingerprint check, lexical baselines, CPU evaluation of the two seed-0 checkpoints) →
  `data/reason_analysis.json`.
- `scripts/cross_region_lexical.py` (the other regions' exact holdouts,
  corruption-feasibility counts) → `data/cross_region_lexical.json`.

All run with `CUDA_VISIBLE_DEVICES=""` from the CogSynDelta `.venv`. Nothing under the
repositories or `/akula-data/csd` was modified.
