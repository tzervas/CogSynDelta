# g48 E1: reasoning-sensitivity battery (corrupted derivation)

Date: 2026-09-06. Scope: diagnosis §4 E1 only, on the E0 seed-0 reason split
(`config/mind/splits/reason-ca364a92-split0.json`, fingerprint
`ca364a92d2c6c5fd259404e0ab6f52a1`, 11,811 train / 132 duplicates removed). No
training. No change to E0. Existing matrix cells only: b256-s0, b512-s0, plus the
untrained encoder at the W2c region-specific seed. Scoring of the encoder is GPU 0;
CPU is loaders, unit tests, and lexical oracles (TF-IDF / BM25).

Battery id: `eval_corrupted_derivation`. Receipts: `kind=eval`.

## Pre-registered interpretation (written before scoring)

Reasoning as a faculty is NOT lookup. Lookup is what the retrieve faculty does.
Reasoning is ruminating on a problem and exploring solutions, questions and answers.

E1 is not trying to raise recall@1. The closed 512-pair diagonal on reason is a
bag-of-words instrument (diagnosis §2: TF-IDF r@1 0.873, trained b256-s0 0.1875).
E1 measures whether any existing checkpoint is sensitive to a broken derivation
step at all: rank the true gsm8k derivation among {true, 4 corruptions} (chance
0.20), with TF-IDF-at-chance and wrong-problem controls.

E3 (structure-sensitive contrastive negatives) is a lookup improvement, second-class
under this definition. E5 (latent-step prediction, iterative refinement) is the
direction.

No change to E0.

Go / kill (diagnosis §4, frozen before scores):

- any checkpoint `derive.recall@1` ≥ 0.30 → derivation sensitivity, E3 next
- all checkpoints `derive.recall@1` ≤ 0.25 → learned nothing about structure,
  E5 ahead of E4
- either way, diagonal r@1 on reason is demoted from a gate to a
  lexical-ceiling fraction (model ÷ TF-IDF)

A value in (0.25, 0.30) for every arm is neither go nor kill.

This block is copied into every E1 receipt's `detail.notes` before metrics are
written.

## Spec (diagnosis §4; not invented)

For each held-out gsm8k pair with ≥ 2 calculator annotations `<<a op b=c>>`
(299 of 320 on the fixed split, mean 3.18 annotations [C]): construct K = 4
corrupted derivations by editing one operand or result in one annotation and
propagating to the `####` line when it is the final value. Deterministic
corruption seed 0 (not a CSD training seed; seed 2 is not invented). Item =
rank the true derivation among {true, 4 corruptions}, chance 0.20.

Controls:

- (a) wrong-problem: true derivation vs four other problems' derivations;
  TF-IDF must score ≥ 0.90
- (b) TF-IDF and BM25 on the corrupted battery must land within ± 0.05 of 0.20
- (c) untrained encoder at the region-specific seed
  (`sha256("csd-w2c-untrained:reason")[:8]` as uint32, W2c)

Arms: b256-s0, b512-s0, baseline (untrained). No training.

## Results

Scored 2026-09-06T02:11Z on GPU 0 (`cuda:0`). Eligible items: 299 / 320 gsm8k
holdout (mean 3.34 calculator annotations). Chance 0.20. Corruption seed 0.

| arm | `derive.recall@1` | `derive.mrr` | vs go (≥0.30) | vs kill (≤0.25) |
|---|---|---|---|---|
| b256-s0 | 0.1137 | 0.4054 | no | yes |
| b512-s0 | 0.1070 | 0.3925 | no | yes |
| baseline (untrained, W2c region seed) | 0.1171 | 0.4299 | no | yes |

Controls (instrument validity; all held):

| control | value | gate |
|---|---|---|
| (a) wrong-problem TF-IDF r@1 | 0.9799 | ≥ 0.90 PASS |
| (b) TF-IDF on corrupted battery | 0.1538 | 0.20 ± 0.05 PASS |
| (b) BM25 on corrupted battery | 0.1538 | 0.20 ± 0.05 PASS |
| (c) untrained encoder | 0.1171 | reported as baseline arm |

GPU minutes: **0.04** (encoder scoring only; loaders/oracles on CPU).

**Kill fired.** All three arms ≤ 0.25 (all sit *below* chance). The contrastive
objective learned nothing about derivation structure. E5 (latent-step prediction)
is ahead of E4. E3 is second-class under the operator definition (reasoning ≠
lookup).

Diagonal r@1 demoted to lexical-ceiling fraction (diagnosis §2 TF-IDF 0.873 on
the same E0 holdout): b256-s0 0.1875 / 0.873 = **0.21**; b512-s0 0.127 / 0.873
= **0.15**.

Commands:

```
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
uv run --group train python scripts/csd-eval-reason-e1.py
```

## Files

- `src/cogsyndelta/eval/corrupted_derivation.py` — battery, oracles, ranking
- `scripts/csd-eval-reason-e1.py` — GPU scoring + eval receipts
- `tests/test_corrupted_derivation.py` — CPU unit tests
- receipts in this directory after the scoring run
