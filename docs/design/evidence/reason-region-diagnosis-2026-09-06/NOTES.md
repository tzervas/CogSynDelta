# reason-region diagnosis: working notes

Started 2026-09-05T20:49-04:00. Read-only task. No GPU (CUDA_VISIBLE_DEVICES=""), no git state changes, no secrets.

## Plan
1. Read design docs (taxonomy, reason faculty, corpus contract, METRICS-METHODOLOGY, evidence/).
2. Read REGIONS["reason"] in scripts/csd-train-all.py and the trainer src/cogsyndelta/regions/pretrain.py.
3. Read program/matrix/csd-matrix.yaml regions.reason.
4. Inspect the corpus parquet (rows, pairs, licences).
5. Read every receipt under /akula-data/csd/matrix/reason-b*-*-7bc2699-20260904/receipts/ and /akula-data/csd/receipts/.
6. Compute the measured table; rank hypotheses; write experiments.

## Log

### 20:52 design docs
- Taxonomy verdict #5 (REGION-TAXONOMY-AND-INTERCONNECT.md:643): `reason` KEEP, RENAME -> `reasoning`, faculty "reasoning / logic", "r@1 0.0801 vs 0.0039, receipt deleted, prose only"; W1b (:2646) regenerate the receipt at >= 0.0801 against a region-specific-seed untrained baseline.
- DEC-04 (:525): keep, "honest limit recorded and a phase-3 objective change". The JSON diff role (:815-821): "Relates a problem to the STRUCTURE of its derivation. HONEST LIMIT: it recognises derivations, it does not produce them. Phase 2 keeps this shape. Phase 3 replaces the objective with latent-step prediction so the region contributes COMPUTATION to the workspace rather than a lookup."
- §9.8 (:5797): "`reasoning` is too weak to receive attention mass" — falsifier is a_reasoning in W5 phase A.
- §1.3 numeric/math placeholder: "reasoning's current corpus straddles this faculty; splitting today starves both".
- DEC-54 fold (:131-134): reason runs ALONE at b512/max_len 256; b1280 OOMed 640 MiB short of 22 GiB [OP memory].
- DEC-73 (:594, :302-307): reasoning reach 1.8% of 1e10 tokens without a generator, >=100% with DeepMind mathematics_dataset (Apache-2.0, "the single source in the whole catalogue that can supply arbitrary B1-relief volume under a settled grant", :4025-4028). DEC-58 (:579): synthetic data contract, four failable clauses.
- Recursive/looped latent refinement: deferred (:4612-4614, §6.9 :4654-4660); v1 seam = n_iter workspace loop, max_iters, ACT halting; "the future step is a TRAINING change". Memory: csd-recursive-latent-transformers (g14 toy: acc@K flat, shortcut learned; next pre-reg needs a target genuinely sequential in K).
- CORPUS-CONTRACT.md §1.6 (:680-745): requires multi-step problems with checkable answers, >= 3 reasoning shapes, difficulty spread, rationales. Staged pool aqua_rat 92.88% / gsm8k 7.12%, N_eff 1.15 -> capped to 60/40, ~12,455 rows, N_eff 1.92; B2 threshold >= 3 fails -> dated waiver in config/mind/csd-regions.json (2026-09-02). Deduction and multi-hop shapes have no clean source.
- METRICS-METHODOLOGY.md §2.1 (:129-157): candidate pool IS the 512 holdout; chance 1/512; closed in-holdout ranking, "not a search over an external corpus". §3.9 anisotropy demoted to a recorded value.

### 21:05 code
- csd-train-all.py:454-486 REGIONS["reason"]: gsm8k-main (question, answer) uncapped + aqua_rat-raw (question, rationale) reservoir-capped 4,982; note "retrieval of the matching solution, NOT step generation"; default max_len 256 (gsm8k answers truncate 41.2% at 96); graded None. TOKEN_AWARE_REGIONS (:326-336): only memory; reason is (0,0).
- lr_for_batch (:284-300): 3e-4 * sqrt(B/256) -> b512 = 4.24e-4.
- pretrain.py: symmetric InfoNCE over in-batch negatives (text_encoder.py:183-245), temperature FIXED 0.05, mean pooling (:130-135), AdamW default wd (pretrain.py:1300), warmup+cosine (:231-248), batch = sliding window over a fixed shuffled order, never reshuffled per epoch (:1405-1409). No dropout in the encoder (grep). No best-checkpoint retention (_CHECKPOINT_KEEP=3 = last three). Holdout = first 512 after shuffle+anchor-dedup (build_splits :872-1005).
- Encoder: vocab 50257 x dim 256 = 12.87M of the 16.02M params are the embedding table; the trunk (4 blocks) is ~3.2M.

### 21:15 receipts (extract_receipts.py -> reason_receipts.json)
| cell | lr | r@1 | r@10 | MRR | untrained r@1 | final loss | in-batch acc | aniso | eff rank | align | quant drop | s |
| b256-s0 | 3.0e-4 | 0.1875 | 0.4199 | 0.2610 | 0.0059 | 0.054 | 0.996 | 0.0023 | 84.3 | 1.214 | 0.0098 (6.5x) | 311.2 |
| b256-s1 | 3.0e-4 | 0.1523 | 0.4121 | 0.2436 | 0.0078 | 0.034 | 1.000 | 0.0012 | 91.8 | 1.261 | -0.0156 (9.8x) | 310.5 |
| b512-s0 | 4.24e-4 | 0.1270 | 0.2676 | 0.1769 | 0.0059 | 0.0098 | 1.000 | 0.0008 | 98.5 | 1.585 | 0.0039 (9.8x) | 599.6 |
| b512-s1 | 4.24e-4 | 0.1426 | 0.3359 | 0.2140 | 0.0078 | 0.0044 | 1.000 | 0.0009 | 112.1 | 1.481 | 0.0059 (9.8x) | 595.9 |
| OLD b512-s0 0026a3d (2026-09-03) | 4.24e-4 | 0.1152 | 0.2676 | 0.1714 | 0.0059 | 0.0096 | 0.998 | 0.0007 | 98.6 | 1.606 | 0.0 (9.8x) | 586.3 |
- Held-out peaks BEFORE the final step in 4 of 5 runs: b512-s1 r@1 0.168@2664 -> 0.143 final; OLD 0.135@2664 -> 0.115; b512-s0 r@10 0.322@1998 -> 0.268; b256-s1 r@1 0.160@1998 -> 0.152. b256-s0 monotone.
- Epochs: 4000*256/11811 = 86.7; 4000*512/11811 = 173.4 (code b512 = 4.75, compress 7.4, retrieve 4.1).
- Train-stage peak VRAM (cell.json, whole card): b256 7,079-7,197 MiB; b512 13,287-13,427 MiB. Linear extrapolation ~26 GiB at b1024 -> b512 is the ceiling at max_len 256 on the 24 GB card.
- Noise: 1 holdout item = 0.00195. Seed spread on r@1: b256 18 items, b512 8 items. Batch effect on r@1 (same seed): s0 31 items, s1 5 items -> inconsistent; on r@10: 78 and 39 items -> consistent; MRR consistent.

### 21:30 corpus + holdout analysis (analyse_reason.py -> reason_analysis.json; fingerprint reproduced ca364a92)
- GPT-2 token lengths: gsm8k answer mean 95.3, p90 153, >96: 41.2%, >256: 0.31%; aqua_rat rationale (all 97,467) mean 72.1, >96: 21.3%, >256: 1.14%; questions mean 55 / 40. Matches the csd-train-all.py comment. Truncation at 256 is NOT binding. Eval at max_len 96 barely moves the model (0.1875 -> 0.1836).
- aqua_rat quality (all rows): 4.0% rationales < 8 words; 17.0% duplicate questions; 6.7% duplicate rationales; 75.6% carry an explicit answer letter. gsm8k: 100% "####" final marker, 98.7% calc annotations "<<a op b=c>>".
- Holdout: 320 gsm8k / 192 aqua_rat; train 7,153 / 4,658.
- LEXICAL BASELINES ON THE EXACT HOLDOUT: TF-IDF r@1 0.873 (gsm8k 0.994, aqua 0.672), r@10 0.945, MRR 0.896; BM25 0.850 / 0.916 / 0.876; number-Jaccard 0.428 / 0.732 / 0.532. Trained b256-s0: 0.1875 / 0.420 / 0.261; b512-s0 0.127 / 0.268 / 0.177. Untrained 0.0059.
- Model, numbers scrubbed: b256-s0 0.1875 -> 0.1426 (r@10 unchanged 0.416); b512-s0 0.127 -> 0.109.
- Within-source pools: b256-s0 gsm8k 0.194 (pool 320), aqua 0.203 (pool 192) ~= full-pool -> source confusion is not the failure. top1 same-source 0.91.
- Model hits vs BM25 hits: b256-s0 96 hits, 83 shared with BM25's 435; hit rate 0.19 when BM25 hits, 0.17 when BM25 misses -> the encoder is not simply a worse lexical matcher on the easy items; it is uniformly weak. Median rank 18 (b256) vs 107.5 (b512).
- Anisotropy recomputed on CPU matches the receipts (0.0023 / 0.0008).

### 21:40 cross-region lexical ceiling (cross_region_lexical.py -> cross_region_lexical.json; all three fingerprints reproduced)
| region | model r@1 (b512-s0 receipt) | TF-IDF r@1 | BM25 r@1 | model / TF-IDF |
| code | 0.980 | 0.982 | 0.996 | 1.00 |
| compress | 0.713 | 0.713 | 0.701 | 1.00 |
| retrieve | 0.721 | 0.770 | 0.754 | 0.94 |
| reason | 0.127 (b256-s0: 0.1875) | 0.873 | 0.850 | 0.15 (0.21) |
- The 512-diagonal battery is a lexical-overlap instrument on every region; three regions sit AT the bag-of-words ceiling, reason sits at 15-21% of it. "Weakest" = has not learned the surface, on 20-40x less data, at 12-40x more epochs.
- Corrupted-derivation battery feasibility: 299 of 320 held-out gsm8k positives have >= 2 calc annotations (mean 3.18); 6,675 of 7,153 training gsm8k rows have >= 2.

### Hypothesis ranking (draft)
H1 battery measures lexical retrieval, not reasoning, and the encoder is below the lexical ceiling only on reason. H2 memorisation: 87-173 epochs over 11.8k pairs, train acc 1.0, held-out peaks mid-run; batch effect = epoch effect (DEC-68 shows batch helps when data is plentiful). H3 corpus tiny / two shapes / waiver. H4 objective is lookup by design (DEC-04). H5 truncation: ruled out at 256. H6 temperature x batch: confounded with H2, weaker. H7 too few steps: opposite. H8 encoder size: not discriminating (same encoder reaches the ceiling elsewhere). H9 collapse/anisotropy: ruled out. H10 no graded gate: measurement gap, not cause.

### 21:20 coordinator rule: identical-seed control + fixed split
- Verified the defect: cfg.seed drives the reservoir cap (pretrain.py:887,895), the holdout-defining shuffle (:928), model init (:1262) and untrained_baseline_seed (:1645); csd-matrix.yaml:270 declares it. Receipts: seed 0 = 11,811 train / 132 dups / untrained 0.0059; seed 1 = 11,798 / 145 / 0.0078, different contamination examples -> different holdouts. Within-seed batch comparisons are on identical holdouts (31 and 5 items on r@1); cross-seed numbers mix split+init+order.
- Report rewritten: E0 (split fix, three-clause acceptance incl. the guard shown to fire; canonical split pinned to the seed-0 draw so 3 of 5 receipts stay comparable), E1-E5 re-expressed with same-seed arms and per-seed paired go/kill.
- Deliverable: REASON-REGION-DIAGNOSIS.md (final). Scripts + JSON copied under scripts/ and data/.
