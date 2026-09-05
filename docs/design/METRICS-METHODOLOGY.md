# Metrics Methodology

**Status:** reference, derived from shipped code. Every formula and `file:line` anchor below
was read from the source on the date in the git log of this file, not written from memory of
what a metric with this name "usually" means. `tests/test_metrics_methodology.py` re-reads
every anchor and fails the build if the cited file no longer has that many lines -- a cheap
drift guard, not a proof the prose still matches, so re-read the anchor before trusting an old
copy of this file.

**Schema stamp: `metrics_schema: "csd-metrics/v2"`** (§14). Sections 1-11 below describe the
receipt field names this project's code wrote **before** the `csd-metrics/v2` migration
landed; most of it is still ground truth for what is on disk **today** (§2's training-receipt
fields are entirely unchanged), but a handful of §3/§9 fields the migration actually renamed,
retired, or demoted are marked inline where they diverge -- check §15's deprecation map before
trusting an unmarked §1-§11 field name against a receipt written after this migration.
Sections 12-19 are the `csd-metrics/v2` layer: the canonical name each v1 field maps to (now
IMPLEMENTED, not only proposed -- `compare()`, §14, is the refuse-function in production,
`src/cogsyndelta/eval/metrics.py:763-843`), a retire list, and the standing statements a card
or harness must not contradict. **v2 is a naming and comparison-discipline layer over the same
measurements, not a formula rewrite** -- a v1 field name in a receipt written before the
migration is not wrong, and nothing in §§12-19 licenses inventing a receipt field this
project's code does not write. Source: `g7-latent-eval-metrics.md` (2026-09-04), read from
`/akula-data/session-backup-staging/tools/grok-jobs/`.

**Why this document exists.** A published model card prints numbers like `recall@1: 0.99`,
`anisotropy: 0.02`, `compression_ratio: 9.83x`. None of those names are standardized across the
field: "recall@1" can mean top-1 accuracy against a 512-item closed pool or against a
57,638-document corpus with several correct answers, "anisotropy" can be averaged over a
whole embedding space or just the query side, "effective rank" has at least two common
definitions that disagree in sign on this project's own regions. A reader who cannot answer
"which formula, over which set, comparable to what" is not reading a measurement, they are
reading a number that looks like one. Every section below answers those three questions for
one metric, points at the exact lines that compute it, and says how two instances of it may
legitimately be compared.

**How to use this document.** Look up the receipt field name in the [field index](#field-index)
at the end, or jump straight to the section for the family it belongs to. Every section has
the same shape: **(a)** the receipt field name, **(b)** the formula as implemented, **(c)** a
`file:line` anchor, **(d)** the evaluation set it is computed on, **(e)** how to compare two
instances of it legitimately, **(f)** known caveats.

---

## Contents

1. [The untrained baseline](#1-the-untrained-baseline)
2. [The training held-out battery (`kind: train` receipts)](#2-the-training-held-out-battery-kind-train-receipts)
3. [The eval battery (`kind: eval` / `eval-quantized` receipts)](#3-the-eval-battery-kind-eval-eval-quantized-receipts)
4. [The battery distinction -- quant vs. eval](#4-the-battery-distinction----quant-vs-eval)
5. [Quantization / compression metrics (`kind: quant` receipts)](#5-quantization-compression-metrics-kind-quant-receipts)
6. [Contamination and corpus-fingerprint fields](#6-contamination-and-corpus-fingerprint-fields)
7. [The external retrieval battery: BEIR-style FiQA + BM25 + the W4 gates](#7-the-external-retrieval-battery-beir-style-fiqa-bm25-the-w4-gates)
8. [The training objective the metrics are measured relative to](#8-the-training-objective-the-metrics-are-measured-relative-to)
9. [Definitional collision: three different "effective rank"s](#9-definitional-collision-three-different-effective-ranks)
10. [How to compare two numbers legitimately](#10-how-to-compare-two-numbers-legitimately)
11. [Project-wide caveats](#11-project-wide-caveats)
12. [`csd-metrics/v2` canonical field names](#12-csd-metricsv2-canonical-field-names)
13. [Retire list (`csd-metrics/v2` §3.2)](#13-retire-list-csd-metricsv2-32)
14. [Schema stamp and refuse predicate](#14-schema-stamp-and-refuse-predicate)
15. [v1 -> v2 deprecation map](#15-v1---v2-deprecation-map)
16. [The anisotropy naming caveat](#16-the-anisotropy-naming-caveat)
17. [W1: participation-ratio and entropy rank disagree in sign](#17-w1-participation-ratio-and-entropy-rank-disagree-in-sign)
18. [The dual-harness principle](#18-the-dual-harness-principle)
19. [Standing statements](#19-standing-statements)
20. [Field index](#field-index)

---

## 1. The untrained baseline

**Why every metric is reported beside it.** A trained model's number means nothing on its own
-- "recall@1 = 0.11" reads as a failure until you learn the untrained floor for that region is
0.006 (a real pair, see §12). Every receipt this project writes carries the untrained
model's own score on the identical held-out set, so a reader never has to trust a claim of
improvement without the comparison that makes it checkable.

**How it is produced.** On a **fresh** run (never a resumed one), `pretrain_region` constructs
the encoder, optionally warm-starts its token-embedding table from another region
(`init_embedding_from`, DEC-24), and calls `evaluate()` / `evaluate_graded()` on the held-out
split **before the first optimizer step** --
`src/cogsyndelta/regions/pretrain.py:1332-1343`. That result is stored as `untrained_baseline`
/ `untrained_graded_baseline` in the checkpoint payload
(`src/cogsyndelta/regions/pretrain.py:1219-1220`) and carried forward unchanged through every
resume (`src/cogsyndelta/regions/pretrain.py:1354-1355`) -- a resumed run's model is no longer
untrained, so re-measuring it there would compare the model against itself, not against a
genuine floor.

**Why an untrained model is not "recall@1 = 0".** A random-init mean-pooled transformer is not
random noise: its sinusoidal position embedding dominates its small token embeddings, so
embeddings cluster and one "attractor" pair can dominate similarity, landing recall@1 near or
sometimes at the `1/eval_pairs` chance floor. That is a real, reproducible number, not a bug --
see the investigation recorded in `src/cogsyndelta/regions/pretrain.py:764-817`.

**The sanity gate.** `beats_untrained` (`recall@1`, `recall@10`, and `spearman` where a graded
set exists) is not a bare `final > baseline` comparison. `_beats_untrained_gate`
(`src/cogsyndelta/regions/pretrain.py:755-869`) requires **all** of:

```text
chance["recall@1"]  = 1 / eval_pairs
chance["recall@10"] = min(10, eval_pairs) / eval_pairs
chance["spearman"]  = 0.0

baseline_sane = baseline["recall@1"] >= chance["recall@1"] / 2
beats[metric] = baseline_sane AND final[metric] > max(baseline[metric], chance[metric]) + 0.01
```

`baseline_sane` on real (quantised) data has exactly one live outcome: it fails only when the
untrained baseline's `recall@1` is **exactly** zero hits out of `eval_pairs` --
`src/cogsyndelta/regions/pretrain.py:796-817` spells out why this is a "the eval produced
literally zero hits" detector and not a graduated quality gate. **Caveat:** `beats_untrained
== True` says the trained model cleared a *sane, non-degenerate* baseline by at least one
recall point (or 0.01 spearman); it does not say the baseline itself was a *good* score, only
that it was not zero.

---

## 2. The training held-out battery (`kind: train` receipts)

Produced by `pretrain_region` (`src/cogsyndelta/regions/pretrain.py:1226`) via
`evaluate()` / `evaluate_graded()`. This is the battery every region's own training receipt
reports, and the one `scripts/csd-quantize.py` reads from for its sensitivity search (§4).

**Evaluation set, every field in this section:** the region's own held-out split -- the FIRST
`cfg.holdout_pairs` (default 512) pairs of the deduplicated, shuffled, contamination-screened
corpus, per `build_splits` (`src/cogsyndelta/regions/pretrain.py:872-1005`). It is the SAME
split every stage (train / quant / eval / eval-quantized) reconstructs, gated on the corpus
fingerprint matching what training recorded (§6) -- never re-globbed or re-sampled.

### 2.1 `held_out.recall@1`, `held_out.recall@10` / `untrained_baseline.recall@1`, `.recall@10`

- **(a)** `held_out.recall@1`, `held_out.recall@10`, and the same two keys under
  `untrained_baseline`.
- **(b)** `recall_at_k(scores, relevant, k)`: the fraction of queries whose single matched
  positive is among the top-`k` scored candidates.

  ```python
  top = scores.topk(k, dim=-1).indices
  recall_at_k = (top == relevant.unsqueeze(-1)).any(dim=-1).float().mean().item()
  ```

- **(c)** formula: `src/cogsyndelta/eval/metrics.py:545-560`. Called from
  `src/cogsyndelta/regions/pretrain.py:558-559` inside `evaluate()`
  (`src/cogsyndelta/regions/pretrain.py:513-562`).
- **(d)** `scores = a @ p.T` where `a`, `p` are L2-normalised encodings of **every** anchor and
  every positive in the held-out split (`src/cogsyndelta/regions/pretrain.py:529-538`), so the
  candidate pool **is the holdout itself** (512 candidates at the project's default), and
  `relevant[i] = i` -- each anchor's positive is the diagonal entry. This is a **closed,
  in-holdout** ranking task, not a search over an external corpus; contrast with §7's BEIR-FiQA
  battery, which ranks against the full 57,638-passage corpus.
- **(e)** Comparable across two receipts only when: same region, same `corpus.fingerprint` +
  `fingerprint_scheme` (§6), same `holdout_pairs`, same `k`. A `recall@1` of 0.99 on a
  512-candidate pool and a `recall@1` of 0.99 on a 57,638-candidate pool (§7) are not the same
  measurement even though they share a name -- see `src/cogsyndelta/eval/beir_fiqa.py:1-21`,
  which states this explicitly as the reason that module exists at all.
- **(f)** Because the pool is exactly the holdout, `recall@1` at this holdout size (512) has a
  chance floor of `1/512 ≈ 0.00195` (see §1) -- not 0.

### 2.2 `held_out.mrr` / `untrained_baseline.mrr`

- **(a)** `held_out.mrr`, `untrained_baseline.mrr`.
- **(b)** Mean reciprocal rank of the single relevant candidate.

  ```python
  order = scores.argsort(dim=-1, descending=True)
  ranks = (order == relevant.unsqueeze(-1)).float().argmax(dim=-1) + 1   # 1-based
  mrr = (1.0 / ranks.float()).mean().item()
  ```

- **(c)** `src/cogsyndelta/eval/metrics.py:563-575`.
- **(d)** Same closed 512-candidate pool as §2.1.
- **(e)** Same rule as §2.1.
- **(f)** None beyond §2.1's pool-size caveat.

### 2.3 `held_out.emb_std` / `untrained_baseline.emb_std` / `graded_held_out.emb_std`

- **(a)** `emb_std` under `held_out`, `untrained_baseline`, `graded_held_out`,
  `untrained_graded_baseline`.
- **(b)** Per-feature standard deviation across the batch dimension, averaged over features --
  the collapse signal: near zero means every input maps to (nearly) the same vector.

  ```python
  emb_std = embeddings.std(dim=0).mean().item()
  ```

- **(c)** Computed **inline**, not via the shared helper: `evaluate()` at
  `src/cogsyndelta/regions/pretrain.py:561` and `evaluate_graded()` at
  `src/cogsyndelta/regions/pretrain.py:612`. The identical formula also exists as a
  standalone, tested function, `representation_std()`
  (`src/cogsyndelta/eval/metrics.py:671-687`) -- since csd-metrics/v2 this IS also a
  production call site: `benchmark_embeddings()` calls `representation_std(a)` for the eval
  battery's `repr.emb_std_anchor` (§12.5, §11.5). `held_out.emb_std` /
  `graded_held_out.emb_std` here still duplicate the formula inline rather than calling the
  shared function, so those two train-receipt fields and the eval-receipt field are the same
  formula from separate call sites, not one calling the other.
- **(d)** `held_out.emb_std` / `untrained_baseline.emb_std`: the **anchor side only** (`a` in
  `evaluate()`, `src/cogsyndelta/regions/pretrain.py:534,561`) of the held-out split.
  `graded_held_out.emb_std`: the **left side only** (`a` in `evaluate_graded()`,
  `src/cogsyndelta/regions/pretrain.py:600,607,612`) of the graded set.
- **(e)** Comparable only across receipts computed on the same side, of the same split, of the
  same region -- and note the pool difference from §3's `repr.*` family below.
- **(f)** **This is anchor-side-only, not anchor+positive.** §3's representation family
  (`repr.anisotropy`, `repr.effective_rank`, etc.) is computed over `torch.cat([anchors,
  positives])` -- both sides pooled together
  (`src/cogsyndelta/eval/benchmark.py:438`). A `held_out.emb_std` and a `repr.*` number
  from the same receipt pair are not measuring the same set of vectors; do not read one as a
  cross-check of the other.

### 2.4 `graded_held_out.spearman` / `untrained_graded_baseline.spearman`

- **(a)** `graded_held_out.spearman`, `untrained_graded_baseline.spearman`. Only present for
  regions that declare a graded corpus (`PretrainConfig.graded_shards`, e.g. `compress`'s
  STS-B validation set, `memory`'s inherited gate) -- `_assert_graded_gate_present`
  (`src/cogsyndelta/regions/pretrain.py:683-742`) refuses to write a receipt that silently
  drops a declared graded gate.
- **(b)** Spearman rank correlation, implemented as Pearson correlation over **average ranks**
  (ties share the mean rank of the tied block) -- not scipy, this project's own
  implementation, because tied values are the majority of a graded set like STS-B (1500
  validation pairs over 64 distinct scores; the largest tie group is 139 pairs at score 0.0).

  ```python
  rank_p, rank_g = average_ranks(predicted_cosines), average_ranks(gold_scores)
  cov   = sum((a - mean(rank_p)) * (b - mean(rank_g)) for a, b in zip(rank_p, rank_g))
  var_p = sum((a - mean(rank_p)) ** 2 for a in rank_p)
  var_g = sum((b - mean(rank_g)) ** 2 for b in rank_g)
  spearman = cov / sqrt(var_p * var_g)     # 0.0 if var_p<=0 or var_g<=0 (a constant side)
  ```

- **(c)** `spearman_correlation()`: `src/cogsyndelta/eval/metrics.py:607-652`.
  `_average_ranks()`: `src/cogsyndelta/eval/metrics.py:578-604`. Called from
  `evaluate_graded()`: `src/cogsyndelta/regions/pretrain.py:565-615`.
- **(d)** `predicted` is the cosine similarity of each graded pair's two encodings; `gold` is
  the corpus's human score. The graded set for a region is a **different corpus** from its
  training pairs by design (e.g. `compress` trains on AllNLI, is graded on STS-B) -- see the
  `graded_shards` field docstring, `src/cogsyndelta/regions/pretrain.py:137-144`, and the
  contamination handling in §6.4.
- **(e)** Comparable only across receipts with the same `graded_corpus.fingerprint` and the
  same `graded_columns`. Two regions graded on different corpora (or the same corpus under a
  different fingerprint scheme) are not comparable by this number alone.
- **(f)** Returns exactly `0.0`, not an error or `nan`, when either side of the pair is
  constant -- "no monotone relationship detectable," which is also what a fully collapsed
  encoder legitimately produces (`src/cogsyndelta/eval/metrics.py:644-651`). **Read `cos_std`
  alongside `spearman`**: a near-zero `cos_std` next to a plausible `spearman` means the
  correlation is being decided by floating-point noise, not a real signal
  (`src/cogsyndelta/regions/pretrain.py:585-589`).

### 2.5 `chance`, `beats_untrained`

Not metrics on their own -- gate/context fields. See §1 for `chance`'s formula and
`_beats_untrained_gate`'s full rule.

### 2.6 `capability_per_param`

- **(a)** `capability_per_param` (training receipt, top level).
- **(b)** `final["recall@1"] / (params / 1e6)` -- §2.1's `recall@1`, per million parameters.
- **(c)** `src/cogsyndelta/regions/pretrain.py:1648`. (The identical name in an eval receipt,
  `eff.capability_per_param`, is a *different* computation over a *different* battery -- §3.7.)
- **(d)** Same held-out split as §2.1; `params` is `sum(p.numel() for p in
  model.parameters())` (`src/cogsyndelta/regions/pretrain.py:1280`), the encoder's raw
  parameter count (fp32, unquantized).
- **(e)** Comparable across regions only when both used the same holdout size (both do, by
  project convention: 512) and the same `k=1` recall definition. This is the project's
  headline thesis metric ("capability per parameter"); see §11.2 for its more
  storage-honest sibling.
- **(f)** Divides by **parameter count**, not by anything storage-related -- two encoders with
  identical architectures but very different quantized sizes have the same
  `capability_per_param`. That is deliberate (see the module docstring,
  `src/cogsyndelta/eval/benchmark.py:28-33`) but easy to misread as an efficiency claim; for a
  storage-normalised version see `eff.capability_per_mb` (§3.7).

### 2.7 `token_aware.final_block_rank`

- **(a)** `token_aware.final_block_rank.{pooled_pr_rank, pooled_entropy_rank,
  token_global_pr_rank, token_global_entropy_rank, n_tokens}`. Recorded on **every** training
  receipt, whether or not `token_loss_weight`/`decorr_weight` are nonzero -- an "off" receipt
  is the comparison arm for an "on" one over the identical corpus/seed
  (`src/cogsyndelta/regions/pretrain.py:1503-1508`).
- **(b)** Two **different** effective-rank definitions (§9 explains why both are recorded, and
  why conflating them is the exact ambiguity this project was previously bitten by), each
  computed twice: once on the mean-**pooled** held-out embeddings, once on the **token-global**
  surface (every real, non-padding token position of the held-out anchors, batch-flattened
  into one `[n_tokens, dim]` matrix). `pr_*` is participation-ratio rank; `*_entropy_rank` is
  Shannon-entropy rank (§9's formulas).
- **(c)** `_final_block_rank_stats()`: `src/cogsyndelta/regions/pretrain.py:273-343`. Calls
  `pr_effective_rank()` (`src/cogsyndelta/eval/benchmark.py:251-293`) for the `pr_*` fields and
  `effective_rank()` (`src/cogsyndelta/eval/benchmark.py:159-186`) for the `*_entropy_rank`
  fields.
- **(d)** The **anchor side only** of the held-out split (index 0 of each pair) -- the same
  side W1's own pre-committed harness measured
  (`src/cogsyndelta/regions/pretrain.py:283-292,324-327`), not both sides. Both rank functions
  are called with `sample` set to the **full** size of the surface being measured
  (`src/cogsyndelta/regions/pretrain.py:338-341`), so **neither is subsampled here** --
  contrast with §3.9/§9's default-`sample=2048` behaviour when these same functions are called
  elsewhere.
- **(e)** Comparable only across receipts for the same region, at the same holdout size,
  measured on the same side of the same corpus fingerprint -- and critically, only `pr_*`
  against `pr_*` or `entropy_*` against `entropy_*`; never one against the other (§9).
- **(f)** This is the pair of numbers the W4/W7 retrain gate reads (`token_global_pr_rank >=
  2.0 * pooled_pr_rank`). That gate is **measured not discriminating** at the harness's current
  (50-step smoke) step count: the control arm, with both token-aware terms fully off, already
  clears the ratio on its own at 2.0191x
  (`src/cogsyndelta/eval/beir_fiqa.py:422-434`, `docs/design/evidence/
  w4-control-arm-2026-09-03/`). A `passed: True` on this clause at 50 steps does not by itself
  distinguish "the token-aware terms worked" from "the terms were never turned on."

---

## 3. The eval battery (`kind: eval` / `eval-quantized` receipts)

Produced by `scripts/csd-benchmark.py` via `benchmark_embeddings()`
(`src/cogsyndelta/eval/benchmark.py:373-457`). This is a **different, wider** battery from §2:
ranking metrics MTEB-family retrieval papers report, efficiency (parameters, size, latency,
throughput), and representation-health diagnostics that a ranking score alone cannot surface.
Written as `Receipt(kind="eval" | "eval-quantized", ...)`
(`src/cogsyndelta/pipeline/receipt.py:58-89`), `metrics` flattened with a family prefix
(`rank.`, `eff.`, `repr.`) by `BenchmarkResult.flat()`
(`src/cogsyndelta/eval/benchmark.py:361-370`).

**Evaluation set, every field in this section:** the same held-out split §2 uses, rebuilt from
the training receipt's own recorded config and cross-checked by corpus fingerprint
(`_region_eval_context()`, `scripts/csd-benchmark.py:462-527`) -- **never** re-globbed. `kind:
"eval"` scores the fp32 checkpoint; `kind: "eval-quantized"` scores the actual packed `.ptq.pt`
artifact loaded back off disk and unpacked to fp32 (`scripts/csd-benchmark.py:886-1080`), not
the in-memory quantization plan (§4). `_run_battery()`
(`scripts/csd-benchmark.py:530-551`) encodes the **whole** holdout as one closed candidate
pool -- identical in shape to §2's `scores = a @ p.T`, `relevant = arange(...)` construction,
which is why `rank.recall@1` in this battery is numerically identical to `held_out.recall@1`
in §2 for the same checkpoint (confirmed against a real receipt pair in §10).

### 3.1 `rank.recall@1`, `rank.recall@5`, `rank.recall@10`

Same `recall_at_k()` formula and closed-pool semantics as §2.1
(`src/cogsyndelta/eval/benchmark.py:409-411`), now also at `k=5`. Same caveats.

### 3.2 `rank.mrr`

Same `mean_reciprocal_rank()` as §2.2 (`src/cogsyndelta/eval/benchmark.py:413`).

### 3.3 `rank.ndcg@10`

- **(a)** `rank.ndcg@10`.
- **(b)** Normalised discounted cumulative gain, single relevant item per query -- with one
  relevant document the ideal DCG is 1, so this reduces to the mean of `1/log2(rank+1)` over
  queries where the item landed in the top `k`.

  ```python
  top = scores.topk(k, dim=1).indices
  hits = top == relevant.unsqueeze(1)
  positions = arange(2, k + 2)                       # rank r -> position r+1 -> log2(r+1)
  ndcg_at_k = (hits.float() / log2(positions)).sum(dim=1).mean()
  ```

- **(c)** `src/cogsyndelta/eval/benchmark.py:40-55`.
- **(d)** Same closed pool as §2.1.
- **(e)** Same rule as §2.1; additionally rank-position sensitive in a way `recall@k` is not
  (a hit at rank 1 scores higher than a hit at rank 10, both count identically for
  `recall@10`).
- **(f)** None beyond the pool-size caveat.

### 3.4 `rank.map`

> **RETIRED as a written receipt field (csd-metrics/v2, §13).** `benchmark_embeddings()`
> deliberately no longer computes or writes `rank.map` -- `average_precision()` stays
> importable only for the sameness-guard test that pins `map == mrr` on this pool
> (`src/cogsyndelta/eval/benchmark.py:403-407`). The formula below still describes what the
> function computes when called directly; it is not on a current receipt.

- **(a)** `rank.map`.
- **(b)** Mean average precision, single relevant item per query -- with one relevant document,
  AP is `1/rank`, so **MAP equals MRR** in this battery.

  ```python
  order = scores.argsort(dim=1, descending=True)
  ranks = (order == relevant.unsqueeze(1)).float().argmax(dim=1) + 1
  average_precision = (1.0 / ranks).mean()
  ```

- **(c)** `src/cogsyndelta/eval/benchmark.py:58-70`.
- **(d)** Same closed pool as §2.1.
- **(e)** Same rule as §2.1.
- **(f)** **Numerically identical to `rank.mrr` at this project's holdout shape** (verified
  against a real receipt in §10) -- reported separately anyway because the two diverge the
  moment a battery introduces graded or multi-positive relevance (as §7's BEIR-FiQA battery
  does), and this project would rather carry a redundant column than a metric whose meaning
  silently changes. Do not read `map != mrr` as a bug signal on THIS battery; on this
  battery they are the same computation by construction.

### 3.5 `rank.precision@10`

> **RETIRED as a written receipt field (csd-metrics/v2, §13).** `benchmark_embeddings()`
> deliberately no longer computes or writes `rank.precision@10` -- `precision_at_k()` stays
> importable only for the sameness-guard test that pins `p@10 == r@10/10` on this pool
> (`src/cogsyndelta/eval/benchmark.py:403-407`). The formula below still describes what the
> function computes when called directly; it is not on a current receipt.

- **(a)** `rank.precision@10`.
- **(b)** Fraction of the top-`k` that is relevant; with exactly one positive this **caps at
  `1/k`**, i.e. `precision@10 = recall@10 / 10` exactly.

  ```python
  top = scores.topk(k, dim=1).indices
  precision_at_k = (top == relevant.unsqueeze(1)).float().sum(dim=1).mean() / k
  ```

- **(c)** `src/cogsyndelta/eval/benchmark.py:73-79`.
- **(d)** Same closed pool as §2.1.
- **(e)** Same rule as §2.1.
- **(f)** **This is not an independent measurement of ranking quality on this battery** -- it
  is `recall@10` rescaled by a constant (`1/10`). A published card printing both
  `rank.recall@10` and `rank.precision@10` is printing the same information twice under two
  names; do not read a gap between them as evidence of anything.

### 3.6 `rank.candidates`

Pool size for this battery -- equal to the held-out split size (`a.size(0)`,
`src/cogsyndelta/eval/benchmark.py:414`), i.e. 512 at this project's default. Not a metric,
context for reading every `rank.*` figure beside it.

### 3.7 `eff.parameters`, `eff.stored_mb`, `eff.capability_per_param`, `eff.capability_per_mb`

- **(a)** All four, under `eff.*`.
- **(b)**

  ```text
  eff.parameters          = raw parameter count (fp32 pass) -- same definition as §2.6's `params`
  eff.stored_mb            = stored_bytes / 1e6
  eff.capability_per_param = rank.recall@1 / (eff.parameters / 1e6)
  eff.capability_per_mb    = rank.recall@1 / eff.stored_mb
  ```

- **(c)** `src/cogsyndelta/eval/benchmark.py:371-380`.
- **(d)** `rank.recall@1` from this same battery (§3.1); `stored_bytes` is
  `fp32_reference_bytes(model)` (`src/cogsyndelta/quant/ptq.py:106-122`) for a `kind: "eval"`
  receipt, or `packed_stored_bytes(packed)`
  (`src/cogsyndelta/quant/ptq.py:488-504`) for a `kind: "eval-quantized"` one -- both
  weights-only counts, recorded as `provenance.stored_bytes_definition: "weights-only"`
  (`scripts/csd-benchmark.py:379,535`), **never** a checkpoint file's raw `stat().st_size`
  (which for a resumable training checkpoint also carries the Adam optimizer's momentum and
  variance buffers, routinely ~2x the weights themselves --
  `scripts/csd-benchmark.py:324-341`).
- **(e)** `eff.capability_per_param` is invariant to quantization (same architecture, same
  `recall@1` if quantization did not move it) -- compare it across regions freely once §10's
  rules hold. `eff.capability_per_mb` is the metric that actually moves with quantization
  (§5) and is the more honest efficiency figure per `src/cogsyndelta/eval/benchmark.py:28-33`
  ("CAPABILITY PER PARAMETER, AND WHY PER-BYTE MATTERS MORE").
- **(f)** `eff.capability_per_param` between a `kind: "eval"` and a `kind: "eval-quantized"`
  receipt for the same checkpoint will usually be numerically identical (parameter count does
  not change under quantization); `eff.capability_per_mb` will differ by roughly the
  compression ratio. If you see the two `capability_per_param` figures differ, that means
  `rank.recall@1` moved under quantization, not that the parameter count changed.

### 3.8 `eff.latency_p50_ms`, `.latency_p95_ms`, `.latency_p99_ms`, `.throughput_per_s`, `.peak_vram_mb`

- **(a)** All five.
- **(b)** `profile_latency()` times `warmup` (default 5, discarded) then `runs` (default 40 in
  this battery's caller) calls of a single forward pass over a fixed batch, synchronising CUDA
  around each timed call.

  ```text
  p50/p95/p99 = the 50th/95th/99th percentile of the sorted per-call wall-clock times (ms)
  throughput_per_s = runs / total_wall_clock_seconds
  peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6   (0 on CPU)
  ```

- **(c)** `src/cogsyndelta/eval/benchmark.py:269-311`. Called from
  `scripts/csd-benchmark.py:249` with `runs=40`, a batch of up to 64 held-out rows.
- **(d)** A fixed-size batch drawn from the held-out split, not the full 512 rows -- this is a
  latency probe, not a ranking measurement, and its size is independent of `rank.candidates`.
- **(e)** Comparable only across runs on the **same physical device** (GPU model, driver, and
  whether anything else was sharing it at measurement time) -- neither the receipt nor this
  document records contention, so a latency figure from a fleet host under load is not
  comparable to one from an idle host even if both name the same GPU.
- **(f)** `peak_vram_mb` is `0.0` unconditionally on a CPU-only run
  (`src/cogsyndelta/eval/benchmark.py:348`) -- absence of GPU memory pressure, not a
  measurement of zero.

### 3.9 `repr.anisotropy`

- **(a)** `repr.anisotropy`.
- **(b)** Mean cosine similarity between **random (off-diagonal) pairs** of embeddings, in
  float64 (a float32 cosine of genuinely identical vectors can return `1.0000001`, which
  silently inverts a threshold test at the collapse boundary).

  ```python
  x = normalize(embeddings.double(), dim=-1)     # optionally subsampled to `sample` rows, seeded
  sim = x @ x.T
  anisotropy = mean(sim[i, j] for i != j)         # off-diagonal only
  ```

- **(c)** `src/cogsyndelta/eval/benchmark.py:88-119`.
- **(d)** **Both anchors and positives pooled together**: `both = torch.cat([anchors,
  positives])` (`src/cogsyndelta/eval/benchmark.py:438`), then subsampled to at most 2048
  rows (seed 0) if the pool exceeds that. At this project's 512-pair holdout, `both` has 1024
  rows -- under the 2048 cap, so **no subsampling actually occurs at the project's current
  holdout size**, though the code path exists and would engage on a larger one.
- **(e)** Comparable across two receipts only when both used the same holdout size (so the
  same subsampling regime applies or does not) and the same sampling seed (fixed at 0 by
  default, not varied per receipt).
- **(f)** **This is a representation-geometry diagnostic, not a quality score.** Higher is not
  better or worse in the abstract: near 0 means unrelated items sit close to orthogonal (a
  healthy, spread-out space); near 1 means the space has collapsed into a narrow cone. The
  project used to gate on `>= 0.9` as unhealthy under the name `not_anisotropic`; that gate is
  now **demoted to a recorded value, not a written pass/fail field** (`repr.anisotropy` is
  still measured and printed, the boolean is not) -- `scripts/csd-benchmark.py:450-453`
  (fp32 pass), `scripts/csd-benchmark.py:630-632` (quantized pass) -- not as "anisotropy
  should be minimised" -- a healthy encoder is not at 0.0 either. Do not rank two models by
  "lower anisotropy is better" without also checking `rank.recall@1` moved the direction you
  expect; anisotropy alone cannot tell you retrieval quality (`src/cogsyndelta/eval/
  benchmark.py:21-26`).

### 3.10 `repr.alignment`

- **(a)** `repr.alignment`.
- **(b)** Expected distance between **matched** pairs (Wang & Isola's alignment term). Lower is
  better -- but only jointly with uniformity (§3.11); a collapsed encoder has perfect
  (near-zero) alignment.

  ```python
  a = normalize(anchors.float(), dim=-1); p = normalize(positives.float(), dim=-1)
  alignment = mean( ||a_i - p_i||_2 ** alpha )        # alpha = 2.0 default
  ```

- **(c)** `src/cogsyndelta/eval/benchmark.py:122-133`.
- **(d)** `anchors`, `positives` from the held-out split -- **matched pairs only** (row `i`
  against row `i`), unlike anisotropy's random-pair pool.
- **(e)** Same rule as §3.9 (holdout size, seed).
- **(f)** Never read alone -- see §3.11.

### 3.11 `repr.uniformity`

- **(a)** `repr.uniformity`.
- **(b)** Log of the mean Gaussian potential over all pairs (Wang & Isola's uniformity term).
  Lower means better spread; a random encoder has excellent uniformity despite carrying no
  learned structure, which is why it is never read without alignment.

  ```python
  x = normalize(embeddings.float(), dim=-1)         # optionally subsampled, seeded
  sq = cdist(x, x) ** 2
  uniformity = log( mean( exp(-t * sq[i, j]) for i != j ) )    # t = 2.0 default
  ```

- **(c)** `src/cogsyndelta/eval/benchmark.py:136-156`.
- **(d)** `both = cat([anchors, positives])`, same pool as anisotropy (§3.9).
- **(e)** Same rule as §3.9.
- **(f)** Read `repr.alignment` and `repr.uniformity` **together**, never one without the
  other: "either alone is gameable... a random one has excellent uniformity"
  (`src/cogsyndelta/eval/benchmark.py:127-128`).

### 3.12 `repr.effective_rank`, `repr.dimensions`, `repr.effective_rank_ratio`

> **RENAMED (csd-metrics/v2, §15).** `repr.effective_rank` -> `repr.effective_rank_entropy`
> and `repr.effective_rank_ratio` -> `repr.effective_rank_entropy_ratio` -- same formulas
> (unchanged by this section), current field names and full entries at §12.1. `repr.dimensions`
> is unrenamed. The names below (`effective_rank`, `effective_rank_ratio`) are the LEGACY
> names for a receipt written before this migration; §15 keeps both mapped.

- **(a)** All three.
- **(b)** `repr.effective_rank` is the **Shannon-entropy** effective rank (see §9 for the
  formula and its participation-ratio counterpart -- **do not conflate the two**).
  `repr.dimensions` is the raw embedding width. `repr.effective_rank_ratio =
  repr.effective_rank / repr.dimensions` -- how much of the available space the encoder
  actually uses.
- **(c)** `effective_rank()`: `src/cogsyndelta/eval/benchmark.py:159-186`. Called with its
  **default** `sample=2048` at `src/cogsyndelta/eval/benchmark.py:445,447` (contrast with
  §2.7, where the same function is called with subsampling disabled).
- **(d)** `both = cat([anchors, positives])`, same pool as anisotropy/uniformity, subsampled to
  2048 rows if larger (not triggered at the project's current 1024-row pool -- see §3.9).
- **(e)** Comparable across receipts at the same embedding dimension directly; across
  different dimensions, compare `effective_rank_ratio` instead of the raw rank.
- **(f)** The project's own gate treats `effective_rank_ratio <= 0.05` as unhealthy
  (`uses_its_dimensions`, `scripts/csd-benchmark.py:453,632`) -- an encoder nominally 256-d but
  effectively 12-d is "paying to store 256"
  (`src/cogsyndelta/eval/benchmark.py:164-165`). **This is the entropy definition, not the
  participation-ratio one §2.7 reports** -- see §9 before comparing a `repr.effective_rank`
  figure against a `token_aware.final_block_rank.pooled_entropy_rank` figure from the SAME
  receipt; they use the same formula but different pools (both-sides-pooled-and-subsampled
  here vs. anchor-only-full-surface there) and are not interchangeable numbers even though
  both are "entropy effective rank."

### 3.13 `gates.beats_untrained`, `gates.not_anisotropic`, `gates.uses_its_dimensions`

> **RENAMED / DEMOTED (csd-metrics/v2, §12.9, §13).** `gates.beats_untrained` (eval,
> unmargined) -> `gate.beats_untrained_eval` -- same predicate, current name and full entry at
> §12.9. `gates.not_anisotropic` is DEMOTED entirely: a current eval / eval-quantized receipt
> no longer writes this gate at all (`repr.anisotropy` stays a recorded metric, just not a
> gate -- §13). `gates.uses_its_dimensions` is unrenamed but now reads the renamed
> `repr.effective_rank_entropy_ratio` (§3.12) at the same `0.05` floor. A current receipt's
> `gates` therefore has exactly two keys, `beats_untrained_eval` and `uses_its_dimensions`, not
> the three below.

- **(a)** All three, under `gates` in an eval / eval-quantized receipt.
- **(b)**

  ```text
  beats_untrained    = rank.recall@1 > train_receipt["untrained_baseline"]["recall@1"]
  not_anisotropic    = repr.anisotropy < 0.9
  uses_its_dimensions = repr.effective_rank_ratio > 0.05
  ```

- **(c)** `scripts/csd-benchmark.py:444-453` (fp32 pass), `scripts/csd-benchmark.py:628-632`
  (quantized pass).
- **(d)** As named above.
- **(e)** Booleans, not compared numerically; compare the underlying figures per their own
  sections instead.
- **(f)** `gates.beats_untrained` here is a **simpler, unmargined** check than §1's
  `beats_untrained` sanity gate on the training receipt (no `+0.01` margin, no
  `baseline_sane` precondition) -- the two fields share a name across receipt kinds but not an
  implementation. Do not assume a training receipt's `beats_untrained: True` and an eval
  receipt's `gates.beats_untrained: True` were decided by the same rule.

---

## 4. The battery distinction -- quant vs. eval

**This is the single most consequential thing to get right when comparing two numbers on a
CogSynDelta card, and it has already produced a wrong comparison once on this project.**

`scripts/csd-quantize.py`'s `quantized_metric` field (and `fp32_metric_recomputed`,
`fp32_metric_receipt`, `drop`) is computed via:

```python
def eval_fn(m):
    return evaluate(m, tok, holdout, cfg.max_len, device)["recall@1"]   # §2's battery
```

-- `scripts/csd-quantize.py:171-172,262`, i.e. **§2's training held-out battery**: a single
`recall@1` figure, measured with the quantization plan applied **in memory**
(`apply_plan()`, `src/cogsyndelta/quant/ptq.py:221-233`), **before `save_packed_artifact` ever
writes a file to disk** (`scripts/csd-quantize.py:187-219`). It is a real number, but a claim
about the *plan*, not about the bytes that end up published.

`scripts/csd-benchmark.py`'s `kind: "eval-quantized"` receipt's `metrics["rank.recall@1"]` (and
every other `rank.*`/`eff.*`/`repr.*` field) is computed via **§3's eval battery**
(`benchmark_embeddings()`), run against a `TextEncoder` rebuilt by `load_packed_artifact()` +
`unpack_state_dict()` from the **actual `.ptq.pt` file read off disk**
(`scripts/csd-benchmark.py:388-420,488-493`). This receipt's own module docstring names the
gap directly:

> "`csd-quantize.py`'s own `quantized_metric` is measured on the in-memory model with the
> quantization plan applied, BEFORE `save_packed_artifact` ever writes a file... a real
> number, but about the plan, not about the bytes that get published."
> -- `scripts/csd-benchmark.py:14-24`

**The rule.** A "plan-vs-artifact delta" -- comparing a quant receipt's `quantized_metric`
against an eval-quantized receipt's `rank.recall@1` -- compares like with like **only** when
both numbers are `recall@1` on the identical checkpoint and holdout, because that is the one
metric name both batteries happen to compute with the same underlying formula
(`recall_at_k()`, §2.1/§3.1). Every OTHER field in the eval-quantized receipt (`rank.mrr`,
every `repr.*`, every `eff.*` beyond `stored_mb`) has **no counterpart at all** in a quant
receipt -- the quant stage never computes them. Do not read "the quant receipt didn't report
X" as "X regressed"; it means X was never measured at that stage.

**Even the one shared name can legitimately diverge.** They are two different code paths
reading two different sources of truth (an in-memory dequantized tensor vs. bytes round-tripped
through `torch.save`/packing/`torch.load`). On the `code` region's most recent matrix cell
(`/akula-data/csd/matrix/code-b512-s0-6614ec2-20260904/receipts/`) they agreed exactly
(`quantized_metric: 0.9921875` == `rank.recall@1: 0.9921875`), but `rank.mrr` moved between the
fp32 and quantized **eval** receipts (`0.9938986301422119` -> `0.9935945868492126`) and
`repr.anisotropy` moved by nearly 7x (`0.00303` -> `0.02075`) -- real, measured effects of
quantization that the quant receipt's single `recall@1` figure cannot see at all.

**Publish-time reconciliation.** `scripts/csd-publish-checkpoint.py` independently
**re-measures** `stored_bytes` and `width_histogram` from the packed artifact file itself
(`verify_quantized_measurements()`, `scripts/csd-publish-checkpoint.py:803-864`, using
`packed_stored_bytes()` / `packed_width_histogram()`, §5.6) and refuses to publish if they
disagree with what the quant receipt claims. It does **not** re-run the eval battery -- a
published card's `stored_bytes`/`compression_ratio`/`width_histogram` are file-verified at
publish time; its `quantized_metric` (if the quant receipt is attached) is not independently
re-checked against the eval-quantized receipt's `rank.recall@1` by the publisher itself. That
cross-check is this document's, and the reader's, to make.

---

## 5. Quantization / compression metrics (`kind: quant` receipts)

Produced by `quantize_text_region()` (`scripts/csd-quantize.py:54-272`), backed by
`src/cogsyndelta/quant/ptq.py`.

### 5.1 `fp32_bytes`

- **(a)** `fp32_bytes`.
- **(b)** Every parameter tensor's element count times 4 bytes -- **weights only**, never
  optimizer state, RNG state, or any other checkpoint bookkeeping `torch.save` happens to
  carry alongside the weights.

  ```python
  fp32_bytes = sum(p.numel() * 4 for p in model.parameters())
  ```

- **(c)** `fp32_reference_bytes()`: `src/cogsyndelta/quant/ptq.py:106-122`.
- **(d)** The trained fp32 checkpoint named by the training receipt.
- **(e)** This is the ONE definition of "fp32 size" this project uses for a compression ratio;
  any other measure (e.g. a checkpoint file's `stat().st_size`, which for a resumable
  checkpoint also carries Adam's momentum/variance buffers, routinely ~2x the weights
  themselves) is not comparable to it -- `src/cogsyndelta/quant/ptq.py:106-121` states this
  explicitly.
- **(f)** None beyond (e).

### 5.2 `stored_bytes`, `compression_ratio`

- **(a)** `stored_bytes`, `compression_ratio`.
- **(b)**

  ```python
  # stored_bytes: apply_plan()'s accounting --
  #   a quantized tensor charges its packed codes + per-channel scale + per-channel zero-point;
  #   an unquantized (fp32-kept) tensor charges numel * 4 bytes.
  compression_ratio = fp32_bytes / max(1, stored_bytes)     # QuantPlan.ratio
  ```

- **(c)** `apply_plan()`: `src/cogsyndelta/quant/ptq.py:221-233`. `QuantPlan.ratio`:
  `src/cogsyndelta/quant/ptq.py:209-212`.
- **(d)** The plan actually selected by `build_plan()`'s greedy sensitivity search (§5.5) for
  this checkpoint.
- **(e)** Comparable across regions once both `fp32_bytes` and `stored_bytes` use this same
  accounting -- true by construction for every CogSynDelta quant receipt, since there is only
  one code path that produces either number.
- **(f)** **This is a payload/storage ratio, not a speed or throughput claim.** A 9.8x
  compression ratio says nothing about inference latency or throughput on its own -- those are
  `eff.latency_*_ms` / `eff.throughput_per_s` in the eval battery (§3.8), measured separately
  and not guaranteed to move proportionally (sub-byte codes still get unpacked to fp32 before a
  forward pass runs, per `dequantize_tensor()`, `src/cogsyndelta/quant/ptq.py:96-103`). A
  `QuantPlan.mean_bits` property also exists (`src/cogsyndelta/quant/ptq.py:214-218`,
  `8.0 * stored_bytes / max(1, fp32_bytes // 4)` -- effective bits per weight, blended across
  quantized and fp32-kept tensors) but is **not currently written to any receipt or card**;
  `compression_ratio` and `width_histogram` (§5.3) are what a reader actually sees.

### 5.3 `width_histogram`, `bits`, `fp32_tensors`, `promotions`

- **(a)** All four.
- **(b)** `bits`: the per-tensor width assignment the greedy search settled on (§5.5).
  `width_histogram`: `bits`'s values, tallied -- `{"3": 17}` means 17 tensors were assigned 3
  bits.  `fp32_tensors`: every parameter `quantizable()` excluded (every 1-D parameter --
  norms, biases -- and any tensor under 4096 elements, `src/cogsyndelta/quant/ptq.py:125-138`)
  plus, structurally, nothing else, since every *eligible* tensor always ends up in `bits`.
  `promotions`: the ordered log of `"{tensor}->{new_bits}b"` moves the greedy search made.
- **(c)** `scripts/csd-quantize.py:195-198` (the receipt's own `by_width` tally, built from
  `plan.bits`, **not** from the packed file -- contrast with §5.6). `bits`/`fp32_tensors`:
  `QuantPlan` fields, assembled by `build_plan()`, `src/cogsyndelta/quant/ptq.py:236-291`.
- **(d)** The plan, not the file (see §5.6 for the file-measured counterpart).
- **(e)** `width_histogram` is directly comparable across two quant receipts for the same
  region/architecture; `bits` (a full tensor-name-keyed dict) is comparable only when both
  checkpoints share the exact same parameter names, i.e. the same encoder architecture.
- **(f)** Width 7 is never assigned -- the ladder is `(2, 3, 4, 5, 6, 8)`
  (`src/cogsyndelta/quant/ptq.py:45`); 7 "costs a byte per eight values more than 6 and almost
  never lands between them in practice." A `width_histogram` will therefore never show a `"7"`
  key by construction, not because nothing needed it.

### 5.4 `fp32_metric_recomputed`, `fp32_metric_receipt`, `quantized_metric`, `drop`, `within_budget`, `tolerance`

- **(a)** All six.
- **(b)** All four metric figures are `recall@1` from §2's training held-out battery
  (`eval_fn`, §4). `fp32_metric_recomputed` is measured fresh on the loaded fp32 checkpoint,
  before quantization; `fp32_metric_receipt` is the training receipt's own recorded
  `held_out.recall@1`, read back for comparison; `quantized_metric` is `plan.metric` after the
  greedy search settles (§5.5); `drop = fp32_metric_recomputed - quantized_metric`;
  `within_budget = drop <= tolerance`.
- **(c)** `scripts/csd-quantize.py:174-190,260-264`.
- **(d)** §2's held-out split, rebuilt and corpus-fingerprint-verified against the training
  receipt (`scripts/csd-quantize.py:96-131`) -- **never** re-globbed.
- **(e)** Same rule as §2.1 -- same region, same corpus fingerprint, same holdout size.
- **(f)** If `fp32_metric_recomputed` and `fp32_metric_receipt` disagree by more than 2 points,
  `csd-quantize.py` prints a warning rather than aborting
  (`scripts/csd-quantize.py:179-184) -- "the split or checkpoint is not what the receipt
  describes." A quant receipt with such a warning in its run log should not be trusted for a
  comparison without checking why.

### 5.5 Sensitivity search methodology

- **(b)** Bits are assigned **greedily, on the task metric, never on weight error.** Every
  quantizable tensor starts at `aggressive_bits` (default 3). While the measured task metric
  (§5.4's `recall@1`) sits outside `tolerance` of the fp32 baseline, the single *most
  sensitive* tensor -- the one whose isolated 3-bit quantization dropped the metric most, per
  `measure_sensitivity()` -- is promoted one rung up the ladder `(2, 3, 4, 5, 6, 8)` and the
  metric is re-measured. Each promotion is bought by an *observed* recovery, not a size-based
  rule of thumb ("big tensors get 8 bits").
- **(c)** `measure_sensitivity()`: `src/cogsyndelta/quant/ptq.py:159-194`. `build_plan()`:
  `src/cogsyndelta/quant/ptq.py:236-291`.
- **(f)** Sensitivity is measured by quantizing **one tensor alone** and recomputing the real
  held-out metric -- "a tensor can absorb large weight error with no task effect, and a tiny
  tensor can carry the decision boundary" (`src/cogsyndelta/quant/ptq.py:9-15`). A tensor's
  sensitivity ranking is therefore specific to the metric it was measured against
  (`recall@1`); it is not a general "how important is this tensor" ranking.

### 5.6 Publish-time re-measurement: `packed_stored_bytes`, `packed_width_histogram`

- **(b)** The same byte accounting as §5.2/§5.3, but read back from the **actual packed file**
  rather than from the in-memory plan:

  ```python
  packed_stored_bytes = sum(codes.nbytes + scale.nbytes + zero.nbytes for every quantized tensor)
                       + sum(numel * 4 for every fp32-kept tensor)
  ```

- **(c)** `src/cogsyndelta/quant/ptq.py:488-504` (`packed_stored_bytes`),
  `src/cogsyndelta/quant/ptq.py:507-518` (`packed_width_histogram`). Invoked at publish time by
  `verify_quantized_measurements()`, `scripts/csd-publish-checkpoint.py:803-864`.
- **(e)/(f)** A quant receipt's own `stored_bytes`/`width_histogram` (§5.2/§5.3) are recorded
  from the **plan** at quantize time; `scripts/csd-publish-checkpoint.py` independently
  re-derives both from the **file** and refuses to publish on any disagreement. A card built by
  the publisher therefore carries file-verified numbers; a raw, unpublished quant receipt's
  numbers are the plan's own claim about itself, not yet cross-checked against any file on
  disk.

---

## 6. Contamination and corpus-fingerprint fields

### 6.1 `contamination` (single-key form)

- **(b)/(c)** `contamination_report()` / `assert_no_contamination()`,
  `src/cogsyndelta/eval/metrics.py:203-256`: overlap between two texts under one
  normalisation (`" ".join(text.split()).lower()`, blake2b-128 hashed). **Historical note, not
  current behaviour for pair splits:** when this single key is the SAME normalisation a caller
  already deduplicated its split on, the intersection is empty by construction and the guard
  reports a zero that is arithmetic, not evidence (`src/cogsyndelta/eval/metrics.py:10-18`,
  `docs/design/evidence/` and `tests/test_guards_can_fail.py`). This is why `build_splits`
  (§6.2) uses the multi-channel guard instead.

### 6.2 `contamination` (training receipt's pair-level form: `train_pairs_seen`, `train_pairs_removed`, `eval_pairs`, `gated_channels`, `channels`, `eval_duplicate_positives`)

- **(a)** `contamination.*` in a `kind: train` receipt.
- **(b)** Six overlap channels between the training pairs and the held-out pairs, each a
  different key over the same two texts: `anchor_exact` / `positive_exact` / `pair_exact`
  (whitespace-and-case-normalised), and `anchor_content` / `positive_content` / `pair_content`
  (content-word SET -- function words, order and repetition discarded). Only the two
  **pair-level** channels (`pair_exact`, `pair_content`) are *gated* -- they alone cause a
  training row to be dropped; the anchor/positive-only channels are reported, not acted on,
  because a shared passage or shared query alone is not necessarily leakage
  (`src/cogsyndelta/eval/metrics.py:259-294`).
- **(c)** `pair_contamination_report()` / `screen_pair_contamination()`:
  `src/cogsyndelta/eval/metrics.py:406-542`. `REMOVAL_CEILING = 0.01`:
  `src/cogsyndelta/eval/metrics.py:467-485` -- caps how much of *training* a repair may delete
  (not how much of the holdout may have leaked) before refusing to train at all. Called from
  `build_splits()`: `src/cogsyndelta/regions/pretrain.py:872-1005`.
- **(d)** The full training corpus (streamed, never fully materialised, per
  `src/cogsyndelta/eval/metrics.py:419-421`) against the held-out split.
- **(e)** `train_pairs_removed` and each channel's `eval_fraction_contaminated` are
  measurements of the split **before** cleanup, so a nonzero value is not itself a defect --
  it is the leak that was found and repaired. Compare two receipts' contamination blocks only
  as a sanity check that the split-building code and corpus are stable, not as a
  ranking metric.
- **(f)** `eval_duplicate_positives` -- held-out pairs whose positive is not unique within the
  holdout -- caps `recall@1` **by construction**, since two identical candidates cannot both be
  ranked first (`src/cogsyndelta/eval/metrics.py:318-321`). A nonzero value here puts a hard
  ceiling below 1.0 on every `recall@1` figure computed from that same holdout.

### 6.3 `corpus.fingerprint`, `corpus.fingerprint_scheme`

- **(a)** `corpus.fingerprint`, `corpus.fingerprint_scheme` (training receipt);
  `corpus_fingerprint` (quant receipt, top level).
- **(b)** A hash over **every** declared source (primary plus every `extra_sources` entry): for
  each, the sorted shard basenames and their byte sizes (not full content -- a full hash of
  ~1,600 parquet shards costs minutes per run), the columns read, and the cap applied. Shard
  order does not affect the value.
- **(c)** `fingerprint_corpus()`: `src/cogsyndelta/corpus.py:74-114`.
  `CORPUS_FINGERPRINT_SCHEME = "csd-corpus-fp/v2"`: `src/cogsyndelta/corpus.py:46-53`.
  `verify_corpus_fingerprint()`: `src/cogsyndelta/corpus.py:116-153`.
- **(e)** **Both** the fingerprint value AND the `fingerprint_scheme` string must match for two
  receipts to describe the same corpus. A scheme mismatch means the two numbers were never
  produced by the same rule (v1 hashed only the primary source's shard names/sizes; v2 hashes
  every source, its columns, and its cap) and says nothing about whether the underlying data
  actually differs -- re-run under the current scheme rather than trying to interpret the
  difference. A same-scheme mismatch means the corpus genuinely changed and every downstream
  comparison is void (`src/cogsyndelta/corpus.py:116-137`).
- **(f)** Byte-size hashing (not content hashing) means a same-name, same-size,
  different-content shard swap would not be detected. This project's own corpus-fetch
  pipeline does not produce that case; a hand-edited shard would defeat the fingerprint
  silently.

---

## 7. The external retrieval battery: BEIR-style FiQA + BM25 + the W4 gates

`src/cogsyndelta/eval/beir_fiqa.py`. Used specifically for the `memory` region's W4 gates
(`docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md`, §4.0, row W4) -- the one battery in this
project that ranks against the **full external corpus**, not a closed in-holdout pool, and
carries a genuine lexical baseline rather than only a random-init one.

### 7.1 Pool construction

- **(b)** `build_ranking_task(split, pool="corpus"|"split")`
  (`src/cogsyndelta/eval/beir_fiqa.py:181-234`) reshapes a BeIR pair split into: deduplicated
  queries, a candidate pool, and a per-query list of pool indices judged relevant.
  `pool="corpus"` ranks every query against **all 57,638 FiQA passages** -- the real task, and
  the one the W4 gates read. `pool="split"` ranks only against the passages judged **within
  that split** -- a far easier pool, present for in-batch comparability, never the headline
  number.
- **(f)** "recall@10 out of 512 and recall@10 out of 57,638 are different measurements that
  happen to share a name" -- `src/cogsyndelta/eval/beir_fiqa.py:1-21`, this module's own stated
  reason for existing separately from §2's `evaluate()`. FiQA judges roughly **2.6 passages
  relevant per question** (`RankingTask.summary()`,
  `src/cogsyndelta/eval/beir_fiqa.py:171-178`) -- multi-relevant, unlike every battery in §2/§3.

### 7.2 `rank_metrics` (recall@1 / recall@10 / recall@100 / mrr, multi-relevant)

- **(a)** `recall@1`, `recall@10`, `recall@100`, `mrr` as computed by this module (distinct
  from the identically-named fields in §2/§3 -- see (f)).
- **(b)** Because a query can have several correct passages, each query's **best-scoring**
  gold index is kept and every OTHER gold index for that query is masked to `-inf` before
  reusing §2's `recall_at_k` / `mean_reciprocal_rank`. This is exact, not an approximation: no
  other gold can outrank the best one, so masking the rest cannot change its rank, and it stops
  a second correct passage from being scored as a wrong document that pushed the right one
  down.

  ```python
  for each query with gold indices G:
      pick = argmax(scores[query, G])
      masked_scores[query, G] = -inf
      masked_scores[query, pick] = scores[query, pick]
  recall@k, mrr = recall_at_k(masked_scores, pick), mean_reciprocal_rank(masked_scores, pick)
  ```

- **(c)** `src/cogsyndelta/eval/beir_fiqa.py:237-276`.
- **(d)** The pool from §7.1 (57,638 passages under `pool="corpus"`); gold from the BeIR qrels.
- **(e)** Comparable only at the same pool mode (`corpus` vs `split`) and the same split
  (`dev`/`test` -- ranking `train` measures memorisation, not generalisation, per
  `src/cogsyndelta/eval/beir_fiqa.py:184-186`).
- **(f)** **Same field names as §2/§3's `recall@k`/`mrr`, different formula and different
  pool.** A `recall@10` from this module is never comparable to a `held_out.recall@10` (§2.1)
  or `rank.recall@10` (§3.1) figure without converting both to the same pool and relevance
  shape first -- they are not the same measurement.

### 7.3 `BM25`

- **(a)** The lexical reference point computed alongside `rank_metrics` (via `bm25_metrics()`).
- **(b)** Okapi BM25, a dependency-free numpy inverted index, with `k1=0.9, b=0.4` --
  **explicitly BEIR's own published settings, not Anserini's defaults (`k1=1.2, b=0.75`)**.

  ```text
  score(query, doc) = sum over query terms t of:
      idf(t) * tf(t, doc) * (k1 + 1) / (tf(t, doc) + k1 * (1 - b + b * len(doc) / avg_len))
  idf(t) = log(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
  ```

- **(c)** `src/cogsyndelta/eval/beir_fiqa.py:279-324` (class `BM25`); `bm25_metrics()`:
  `src/cogsyndelta/eval/beir_fiqa.py:390-398`.
- **(d)** Scored through the **identical** `rank_metrics()` code path as the trained encoder
  (§7.2), over the same pool and qrels -- "reporting a win for a loss" (beating "beats random
  init" while losing to word counting) is what this module exists to make structurally
  impossible to do by accident (`src/cogsyndelta/eval/beir_fiqa.py:16-21`).
- **(e)** Only comparable to an encoder's `rank_metrics()` figures on the exact same pool/qrels
  -- true by construction here, since `w4_gates` always builds both from the same
  `RankingTask`.
- **(f)** Tokenisation is a bare `[a-z0-9]+` regex over lower-cased text
  (`src/cogsyndelta/eval/beir_fiqa.py:57`) -- no stemming, no stopword removal. This is a
  simpler BM25 than a tuned production system might run; it is a floor, not an
  upper bound on lexical retrieval.

### 7.4 The W4 gates (`a_beats_both_parents` .. `e_retrain_gate`)

Five pure functions over already-measured numbers, each independently unit-testable and each
naming one pass/fail condition:

| gate | condition | anchor |
|---|---|---|
| `a_beats_both_parents` | `memory`'s `recall@1` >= max(`compress`, `retrieve` parents' `recall@1`) AND `memory`'s graded spearman >= `compress`'s | `src/cogsyndelta/eval/beir_fiqa.py:612-641` |
| `b_full_pool_thresholds` | full-pool `recall@10 > 0.20` AND `mrr > 0.10` | `src/cogsyndelta/eval/beir_fiqa.py:644-661` |
| `c_beats_bm25` | trained `recall@10` > BM25 `recall@10`, same pool/qrels | `src/cogsyndelta/eval/beir_fiqa.py:664-677` |
| `d_beats_random_init` | trained `recall@10` > this region's own untrained-encoder `recall@10`, same pool | `src/cogsyndelta/eval/beir_fiqa.py:680-690` |
| `e_retrain_gate` | `token_global_pr_rank >= 2.0 * pooled_pr_rank` (§2.7/§9) AND no more than 1-point regression against either parent | `src/cogsyndelta/eval/beir_fiqa.py:693-760` |

**(f)** `e_retrain_gate`'s rank clause is the one flagged in §2.7(f) as measured
not-discriminating at 50 steps -- see that caveat before treating a `passed: True` here as
proof the token-aware objective helped. `e_retrain_gate` also explicitly does **not** check the
`banking77` damage-detector probe §4.0 of the design doc names
(`src/cogsyndelta/eval/beir_fiqa.py:547-550`) -- recorded as `"not measured; out of scope"`
in the gate's own output, not silently omitted.

### 7.5 Explicit field names for `beir.*` and `token.*` surfaces (schema v2)

`g7-latent-eval-metrics.md` §3.1 (2026-09-04) assigns canonical, explicitly-named receipt
fields to two surfaces this section and §2.7/§9 already describe by formula but not, until
now, by a stamped name distinct from every other prefix in this document.

- **`beir.recall@1` / `beir.recall@10` / `beir.recall@100` / `beir.mrr`** -- exactly §7.2's
  `rank_metrics()` numbers, renamed under the `beir.` prefix instead of the bare
  `recall@k`/`mrr` keys that module returns on its own (those bare names are what collide,
  by spelling only, with §3.1/§3.2's closed-pool `rank.recall@k`/`rank.mrr` -- see §7.2(f)
  and §10). `to_beir_metrics(pool, metrics)`
  (`src/cogsyndelta/eval/beir_fiqa.py`) performs the rename and stamps two provenance
  fields alongside it: `pooling` (`fiqa_corpus` for `pool="corpus"`, `fiqa_split` for
  `pool="split"`) and `battery_id` (`beir_fiqa_corpus` / `beir_fiqa_split`) -- the fields a
  schema-v2 comparison must match before treating two numbers as the same measurement
  (§10). **`beir.ndcg@10` is a documented placeholder, not a number**: TREC-style nDCG
  over FiQA's multi-relevant qrels is not implemented anywhere in this project today --
  `rank_metrics()` computes recall@k/MRR only (its own module docstring: "No nDCG"). A
  separate bridge outside this repository is expected to compute it via `pytrec_eval`
  against the identical qrels and write it into a receipt's `detail.external`
  (`g7-latent-eval-metrics.md` §4) -- never into `metrics["rank.ndcg@10"]` (that key
  already names the closed-pool formula, §3.3) and never fabricated here as a stand-in
  value.
- **`token.pooled_pr_rank`, `token.global_pr_rank`, `token.pooled_entropy_rank`,
  `token.global_entropy_rank`, `token.pr_rank_ratio`** -- explicit names for the five
  numbers §2.7/§9 already document as `_final_block_rank_stats`'s `pooled_pr_rank`,
  `token_global_pr_rank`, `pooled_entropy_rank`, `token_global_entropy_rank`, and the W4
  gate's `token_global_pr_rank / pooled_pr_rank` ratio. `token_rank_surfaces()`
  (`src/cogsyndelta/eval/beir_fiqa.py`) performs the rename, grouping the pooled pair
  under `pooling: "anchor_pooled"` and the token-global pair under `pooling:
  "anchor_token_global"` (both `battery_id: "train_token_rank"`), and computes
  `token.pr_rank_ratio` through `refuse_cross_family_rank_ratio()` -- a guard that raises
  rather than divide a `_pr_rank` field by an `_entropy_rank` field or vice versa. This
  guard is local to this module and does **not** replace the project-wide schema
  refuse-predicate (§10; `g7-latent-eval-metrics.md` §3.3): that predicate matches
  `battery_id`/`pooling`/schema/checkpoint-sha, which cannot by itself catch a PR-vs-
  entropy conflation, because a PR field and an entropy field measured on the identical
  holdout legitimately share both. `gate_e_retrain_gate`'s `pr_rank_clause` additionally
  carries `token.pr_rank_ratio` (identical value to its existing `ratio` key) and
  `token.pr_rank_ratio_formula` beside its long-standing fields, so neither name is a
  breaking rename of the other.

---

## 8. The training objective the metrics are measured relative to

Every metric in §§1-3 is measured on a model trained against this objective. Knowing its shape
is what makes "why does `emb_std` matter" or "what does the untrained baseline actually differ
by" answerable.

### 8.1 Symmetric InfoNCE over in-batch negatives

- **(b)** Each anchor's positive is its pair; every other item in the batch is a negative --
  the number of negatives *is* the batch size, with no hard-negative mining. Trained in both
  directions (anchor->positive and positive->anchor) and averaged, because retrieval is used
  both ways.

  ```python
  a, p = normalize(anchors.float()), normalize(positives.float())    # fp32, always
  logits = (a @ p.T) / temperature            # temperature = 0.05 default
  labels = arange(batch_size)
  loss = 0.5 * (cross_entropy(logits, labels) + cross_entropy(logits.T, labels))
  ```

- **(c)** `info_nce()`: `src/cogsyndelta/regions/text_encoder.py:183-244`. The receipt's
  `method` field literally names this: `"symmetric InfoNCE over in-batch negatives"`
  (`src/cogsyndelta/regions/pretrain.py:1556`).
- **(f)** The loss is forced to fp32 even under a bf16 autocast -- at `temperature=0.05` a
  unit-cosine logit lands near ±20, where bf16's ~0.125 quantum would be a real perturbation of
  a 1000+-class softmax (`src/cogsyndelta/regions/text_encoder.py:217-228`). `in_batch_acc`
  (top-1 accuracy within the training batch, chance = `1/batch_size`) is logged in `history`
  alongside `loss` for exactly the collapse-detection reason §2.3 describes: "a collapsed
  encoder has low loss and chance-level accuracy."

### 8.2 `L_token` -- masked-token prediction (opt-in, `token_loss_weight > 0`)

- **(b)** BERT-style masked-token prediction, attached at the **final block only** (a
  penultimate-block variant was measured and lost in every region -- W1d, "FALLBACK -- STRUCK
  IN REVISION 3.2"). A sampled fraction (`token_loss_mask_prob`, default 0.15) of real
  (non-padding) positions have their token embedding replaced by a learned `[MASK]` vector
  before the transformer blocks run; a discardable linear head predicts the original token id
  from the final, post-norm representation at those positions.

  ```python
  mlm_mask = (rand(B, T) < mask_prob) & real_positions
  h = where(mlm_mask, mask_embedding, token_embedding(input_ids))
  h = final_block_output(h)                         # full trunk, masked embedding substituted
  loss = cross_entropy(mlm_head(h[mlm_mask]), input_ids[mlm_mask])
  ```

- **(c)** `_mlm_token_loss()`: `src/cogsyndelta/regions/_token_objective.py:122-205`. Invoked
  from the training loop: `src/cogsyndelta/regions/pretrain.py:1431-1454`.
- **(f)** Chunked processing (`token_loss_chunk`, default 2048) is a **memory** knob only -- the
  chunked and unchunked forms are mathematically identical, proven via
  `torch.utils.checkpoint`'s exact (non-reentrant) backward
  (`src/cogsyndelta/regions/_token_objective.py:80-119`). The MLM head and mask embedding are
  auxiliary parameters, **not** part of `TextEncoder`'s own state dict -- discarded after
  training, exactly as every checkpoint reader already expects.

### 8.3 `L_decorr` -- token-position covariance decorrelation (opt-in, `decorr_weight > 0`)

- **(b)** A VICReg/Barlow-Twins-shaped penalty on the off-diagonal mass of the final block's
  token-position feature covariance, computed over every real token position of the batch
  flattened into one matrix (the same "token-global" surface §2.7's rank stats measure),
  normalised by dimension so the penalty's scale does not grow with encoder width.

  ```python
  flat = token_representations[real_positions] - mean(...)     # [N, dim], centered
  cov = (flat.T @ flat) / (N - 1)
  L_decorr = (sum(cov ** 2) - sum(diagonal(cov) ** 2)) / dim    # off-diagonal squared mass only
  ```

- **(c)** `_token_decorrelation_loss()`: `src/cogsyndelta/regions/_token_objective.py:208-233`.
- **(f)** Zero (with no gradient) when fewer than 2 real positions survive in a batch --
  not an error.

### 8.4 The combined objective and where it is `OFF`

- **(b)** `L = InfoNCE(pool, pool+) + γ·L_decorr(h) + ζ·L_token(h)`, where `γ =
  decorr_weight`, `ζ = token_loss_weight`, both default **0.0** -- every region trained with
  the defaults trains byte-identically to a plain-InfoNCE run; the two auxiliary terms are
  additive and opt-in, never structural.
- **(c)** Weights: `PretrainConfig.token_loss_weight` /
  `.decorr_weight`, `src/cogsyndelta/regions/pretrain.py:168-193`. Loop:
  `src/cogsyndelta/regions/pretrain.py:1421-1461`.
- **(f)** A training receipt's `token_aware.enabled` field
  (`src/cogsyndelta/regions/pretrain.py:1656-1663`) is `True` iff **either** weight is nonzero
  -- check the individual weights, not just `enabled`, before assuming both terms were active
  for a given receipt.

---

## 9. Definitional collision: three different "effective rank"s

This project computes "effective rank" **three different ways**, and two of them disagree in
sign on the project's own production regions. A card, receipt, or design doc that says
"effective rank" without saying which of these it means is not naming a measurement.

| function | formula | subsampling | `<2` rows |
|---|---|---|---|
| `effective_rank()` (**entropy**) | `exp(-sum(p_i * log(p_i)))` where `p = singular_values / sum(singular_values)` | default `sample=2048`, seed 0 | returns `0.0` |
| `participation_ratio()` | `(sum(s_i^2))^2 / sum(s_i^4)` over singular values `s` | default `sample=2048`, seed 0 | returns `0.0` |
| `pr_effective_rank()` | **identical formula** to `participation_ratio()` | **never** subsamples | returns `nan` |

```python
# effective_rank -- Shannon entropy of the normalised singular-value spectrum, exponentiated
x = embeddings - mean(embeddings, dim=0)
sv = svdvals(x)
p = sv / sum(sv)
effective_rank = exp(-sum(p_i * log(p_i) for p_i in p if p_i > 0))

# participation_ratio / pr_effective_rank -- participation-ratio rank
x = embeddings - mean(embeddings, dim=0)
sv = svdvals(x); s2 = sv ** 2
pr_rank = (sum(s2)) ** 2 / sum(s2 ** 2)
```

- **(c)** `effective_rank()`: `src/cogsyndelta/eval/benchmark.py:159-186`.
  `participation_ratio()`: `src/cogsyndelta/eval/benchmark.py:189-248`. `pr_effective_rank()`:
  `src/cogsyndelta/eval/benchmark.py:251-293`.
- **(f)** **Why two formulas exist at all, and why the third exists on top of the second:**
  measured on this project's own four production text checkpoints, participation-ratio rank
  ratios came out **0.66x-1.30x** while entropy rank ratios came out **1.16x-1.84x** for the
  identical checkpoints -- they disagree in **sign** on whether the token-global surface has
  higher or lower rank than the pooled one
  (`src/cogsyndelta/eval/benchmark.py:162-169`, `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md`
  §4.0). "Both ranks are participation ratio, and the receipt says so" is the rule this project
  adopted to close that ambiguity -- **never infer which definition a bare "effective rank" or
  "rank ratio" number uses; the field name or an explicit label must say.**

  `pr_effective_rank()` exists as a **byte-for-byte-identical transplant** of the exact
  function W1's pre-committed evidence script measured
  (`docs/design/evidence/w1-token-rank-2026-09-02/measure_w1.py`'s own `pr_effective_rank`,
  lines 198-212 of that frozen file) -- not merely the same formula as
  `participation_ratio()`, but the same *edge-case behaviour* (no subsampling ever; `nan`
  rather than `0.0` below 2 rows), so that every retrain gate reading W1's own pre-committed
  thresholds (§2.7, §7.4's `e_retrain_gate`) reads the **same measurement** W1 itself took,
  not a same-shaped one that quietly diverges from it under subsampling.
  `tests/test_pr_effective_rank_w1_alignment.py` asserts the two agree to `1e-6`.

  **Practical rule:** `repr.effective_rank` (§3.12, from the eval battery) is **always** the
  entropy definition. `token_aware.final_block_rank.*_pr_rank` (§2.7, from a training receipt)
  is **always** the participation-ratio definition, measured via `pr_effective_rank()` (full
  surface, no subsampling). `token_aware.final_block_rank.*_entropy_rank` is the entropy
  definition, also measured full-surface (subsampling disabled by passing `sample=` the
  surface's own size) -- so even two "entropy rank" numbers from this project can differ in
  whether subsampling applied, depending which call site produced them (§2.7 vs §3.12). Never
  divide one of these by another across functions or across subsampling regimes and call the
  result a ratio.

---

## 10. How to compare two numbers legitimately

Before reading anything into a difference between two CogSynDelta metric values, confirm **all**
of the following hold. If any one fails, the two numbers are not describing the same
measurement and a difference between them says nothing about the model.

1. **Same region.** `region` (training/quant receipts) or `producer.component` (eval
   receipts) match.
2. **Same corpus fingerprint AND scheme.** `corpus.fingerprint` and
   `corpus.fingerprint_scheme` (§6.3) match exactly -- not merely "close," and not compared
   under different scheme versions.
3. **Same battery.** A `held_out.*` figure (§2, `evaluate()`) is not the same measurement as a
   `rank.*`/`repr.*`/`eff.*` figure (§3, `benchmark_embeddings()`) even when the underlying
   formula happens to be identical (§2.1 vs §3.1) -- and neither is comparable to a
   `quantized_metric` (§4) or a BEIR-FiQA `recall@k` (§7.2) without converting pool and
   relevance shape first.
4. **Same `k`** for any `recall@k`/`precision@k`/`ndcg@k` comparison, and same pool size
   (`rank.candidates`, §3.6, or `n_pairs`, §2.1) -- a chance floor of `1/512` and a chance
   floor of `1/57638` are not the same baseline.
5. **Same checkpoint bytes**, when the claim is "this is the published artifact's number":
   check `artifacts.checkpoint_sha256` (and, for a quantized figure,
   `artifacts.quantized_sha256`) rather than trusting that two receipts naming the same path
   describe the same weights -- a region's checkpoint lives at one mutable path, and a later
   training run silently invalidates every earlier receipt that named it
   (`scripts/csd-publish-checkpoint.py:48-72`).
6. **Same code revision**, when the claim is "this is exactly reproducible" -- every receipt
   carries `code_revision.{git_sha,dirty,branch,describe}`
   (`src/cogsyndelta/regions/_receipt.py:58-154`, stamped by `write_receipt()`,
   `src/cogsyndelta/regions/_receipt.py:156-191`); a `dirty: true` receipt was written against
   uncommitted changes and cannot be reproduced from the named sha alone.
7. **Same seed**, for anything sampled: `config.seed` (training) drives the corpus reservoir
   sample, the shuffle, and the untrained model's initial weights;
   `untrained_baseline_seed` (`src/cogsyndelta/regions/pretrain.py:1645`) records the seed the
   untrained model specifically was constructed with.

Two published cards satisfying all seven are the only pair this project considers a
legitimate "before vs. after" comparison.

---

## 11. Project-wide caveats

1. **The PTQ compression ratio is a payload/storage ratio, not a speed or throughput claim.**
   See §5.2(f). A high `compression_ratio` says nothing about `eff.latency_*`/`throughput_per_s`
   on its own.
2. **Anisotropy (and alignment/uniformity/effective-rank) are representation-geometry
   diagnostics, not quality scores.** See §3.9(f)/§3.10(f)/§3.11(f). None of them should be
   read as "lower/higher is better" in isolation from a ranking metric (`recall@k`/`mrr`) --
   they exist to catch the case where a ranking metric looks fine while the space has
   quietly collapsed, per `src/cogsyndelta/eval/benchmark.py:21-26`.
3. **A licence tier follows the corpus, not the metric.** `licence_tier()`
   (`scripts/csd-publish-checkpoint.py:395-420`) is derived entirely from
   `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`'s per-region table (section "Decision
   2026-09-02") -- a region's training data provenance -- and has no dependency on anything in
   this document. A region whose numbers on this page look identical to another's can still
   carry a stricter licence, and a merged region inherits its most restrictive parent's tier
   regardless of either parent's measured metrics
   (`scripts/csd-publish-checkpoint.py:200-203`).
4. **The untrained baseline is not "zero."** See §1 in full before reading a small
   `recall@1`/`spearman` as a failure -- check it against `chance` and
   `untrained_baseline`/`untrained_graded_baseline` first.
5. **`token_weighted_perplexity()` is implemented, tested, and exported, but no production
   training or eval path calls it.** `token_weighted_perplexity()`
   (`src/cogsyndelta/eval/metrics.py:75-97`) is exercised only by `tests/test_eval_metrics.py`
   (confirmed by a repo-wide search for its call sites, 2026-09). No region currently reports a
   `perplexity` field despite the module's own opening docstring framing the project's
   efficiency claim around "without decreasing quality and/or increasing perplexity"
   (`src/cogsyndelta/eval/metrics.py:1-8`) -- if a future card or receipt does show a
   `perplexity` field, its formula is `token_weighted_perplexity()`'s (§2's `emb_std`
   sibling), and it should cite this document's entry for it rather than an assumed standard
   definition. `representation_std()` (`src/cogsyndelta/eval/metrics.py:671-687`) is
   DIFFERENT: since csd-metrics/v2 it IS a production call site --
   `benchmark_embeddings()` calls `representation_std(a)` for the eval battery's
   `repr.emb_std_anchor` (§12.5, §2.3(c)) -- `held_out.emb_std` and `graded_held_out.emb_std`
   remain an inline duplicate of the same formula, not a call to the function, so one region's
   receipt can carry the formula computed through the function (`repr.emb_std_anchor`) beside
   two fields computing it inline (`held_out.emb_std`, `graded_held_out.emb_std`).
6. **`kind: "eval"` and `kind: "eval-quantized"` gates use a different, unmargined
   `beats_untrained` rule than the training receipt's own `beats_untrained`.** See §3.13(f). Do
   not assume the two were decided by the same threshold just because they share a name.
7. **A quant receipt's own `stored_bytes`/`width_histogram` are the plan's claim about itself
   until publish time re-measures them from the file.** See §5.6. Prefer numbers from a
   published card (file-verified) over a bare quant receipt when the two could differ.

---

## 12. `csd-metrics/v2` canonical field names

One canonical receipt-field name per formula. `csd-benchmark.py`'s `benchmark_embeddings()`
writes every §12.1/§12.3/§12.4/§12.5/§12.6 name below to a current eval / eval-quantized
receipt (§15 records the v1 name each superseded, for reading an older receipt); §12.2's
`token.*` names and §12.7's `beir.*` names are NOT yet written by any production receipt --
check §15 before assuming a name below is on disk for a GIVEN battery.

### 12.1 `repr.effective_rank_entropy`, `repr.effective_rank_entropy_ratio`

- **Formula** (Roy & Vetterli, EUSIPCO 2007, Def. 1 + Property 1; logs base *e*, `0 log 0 :=
  0`):

  ```python
  sv = svdvals(embeddings - mean(embeddings, dim=0))    # linear spectrum
  p = sv / sum(sv)
  effective_rank_entropy = exp(-sum(p_i * log(p_i) for p_i in p if p_i > 0))
  ```

- **Pool:** `pooled_both` (anchors + positives concatenated), subsample 2048 seed 0.
- **`battery_id`:** `eval_holdout` / `eval_quantized_holdout`.
- **Unit:** scalar, `[1, dim]`.
- **Citation:** Roy & Vetterli, EUSIPCO 2007, Poznan, Def. 1 + Property 1.
- **Why this metric:** this is the project's own comparable prior (the existing "8.7 of 128"
  figure, §3.12) -- it up-weights the spectral tail, catching a "many small directions" style
  of under-use that participation ratio (§12.2) is comparatively insensitive to.
- **Why this battery:** `pooled_both` is the eval battery's existing anisotropy/alignment
  pool (§3.9-§3.11); reusing it means one receipt's representation-geometry fields all
  describe one fixed set of vectors rather than four different subsamples.
- **Falsifies:** `gate.uses_its_dimensions` -- an encoder nominally wide but effectively
  narrow (§3.12(f)).
- **Comparison rule:** same embedding dimension, same pool, same subsample regime (§10). Do
  **not** compare against `token.pooled_entropy_rank` / `token.global_entropy_rank` (§12.2) --
  same formula, different pool (both-sides-pooled-and-subsampled here vs. anchor-only-full-
  surface there), not interchangeable numbers (§9, carried forward unchanged under v2).
- **Sameness special-case:** none.

**`repr.effective_rank_entropy_ratio`** -- `repr.effective_rank_entropy / repr.dimensions`,
i.e. how much of the available embedding width the encoder actually uses. Renamed from
`repr.effective_rank_ratio` (§3.12, §15) to name which of this project's three "effective
rank" definitions the ratio's numerator is (§9: the entropy one, never the participation-ratio
one §12.2 reports).

- **Formula:** `effective_rank_entropy_ratio = effective_rank_entropy / dimensions`
  (`src/cogsyndelta/eval/benchmark.py:447`).
- **Pool / `battery_id`:** same as `repr.effective_rank_entropy` above (`pooled_both`,
  `eval_holdout` / `eval_quantized_holdout`) -- it is that same number, rescaled.
- **Unit:** `[0, 1]`.
- **Why this metric:** the raw rank alone is not comparable across two encoders of different
  embedding width; the ratio is (§3.12(e)).
- **Falsifies:** `gate.uses_its_dimensions` (`effective_rank_entropy_ratio > 0.05`,
  `scripts/csd-benchmark.py:453,632`, §12.9) -- the `0.05` floor is slack, not a tight bound:
  W1's own production regions sit at `~0.45-0.51` (`csd-benchmark.py:453`), so a value near the
  floor is a genuine "paying to store more than it uses" signal, not measurement noise (§13).
- **Comparison rule:** same as `repr.effective_rank_entropy` above.
- **Sameness special-case:** none.

**Do not invent `repr.effective_rank_pr`.** `participation_ratio()` (the pooled,
default-subsampled PR function) is exercised only by
`tests/test_pr_effective_rank_w1_alignment.py`; no production receipt writer calls it. A card
or receipt field with this name would have no code behind it.

### 12.2 `token.pooled_pr_rank` / `token.global_pr_rank` / `token.pooled_entropy_rank` / `token.global_entropy_rank`

- **Formula, PR pair** (participation ratio, **never** subsampled, `<2` real rows -> `nan`;
  Chun, Canatar, Chung, Lee, arXiv:2509.26560 v1, §2.2 `gamma_0`):

  ```python
  s = svdvals(X - mean_rows(X))
  pr_rank = (sum(s_i ** 2 for s_i in s)) ** 2 / sum(s_i ** 4 for s_i in s)
  ```

- **Formula, entropy pair:** identical to §12.1's `effective_rank_entropy`, called with
  `sample=` the surface's own full size (subsampling disabled).
- **Pool:** `token.pooled_*` = `anchor_pooled` (mean-pooled anchor side of the holdout, one
  vector per item). `token.global_*` = `anchor_token_global` (every real, non-padding token
  position of the anchor side, batch-flattened into one `[n_tokens, dim]` matrix).
- **`battery_id`:** `train_holdout` -- W4/W1's own harness, not the wider eval battery.
- **Unit:** scalar, `[1, n_pool_items]` upper-bounded.
- **Citation:** W1 (`docs/design/evidence/w1-token-rank-2026-09-02/`) / W4
  (`src/cogsyndelta/regions/pretrain.py:_final_block_rank_stats`); PR finite-sample-bias
  caveat per Chun et al. 2025.
- **Why these metrics:** `token.global_pr_rank / token.pooled_pr_rank` is W4's `e_retrain_gate`
  numerator/denominator -- the check for whether a token-aware objective moved the
  token-global surface's structure relative to the pooled one. The entropy pair is recorded
  **beside** the PR pair on every receipt, never instead of it, because the two disagree in
  sign on this project's own regions (§17).
- **Why this battery:** `_final_block_rank_stats` is a byte-for-byte transplant of W1's own
  pre-committed measurement function (`pr_effective_rank()`, no subsampling, `nan` below 2
  rows) so a retrain gate reads the **same** measurement W1 took, not a same-shaped one that
  quietly diverges under a different subsampling regime.
- **Falsifies:** `e_retrain_gate`'s PR-rank clause. **Measured not discriminating at the
  harness's current 50-step smoke count** -- the control arm, with both token-aware terms at
  `0.0`, already clears `token_global_pr_rank >= 2.0 * pooled_pr_rank` on its own at `2.0191x`
  (§17, `docs/design/evidence/w4-control-arm-2026-09-03/`). A `passed: True` on this clause at
  50 steps does not by itself distinguish "the objective worked" from "the objective was never
  turned on."
- **Comparison rule:** `pr_*` against `pr_*` or `entropy_*` against `entropy_*` only, never one
  against the other, and never as a cross-function ratio (§9). Same region, holdout size,
  corpus fingerprint side.
- **Sameness special-case:** none across the PR/entropy pair; ordinary §3.3 refuse-predicate
  rules apply within one family.

### 12.3 `repr.anisotropy`

- **Formula** (mean off-diagonal cosine similarity, float64, after L2 normalisation):

  ```python
  x = normalize(embeddings.double(), dim=-1)     # optionally subsampled, seed 0
  sim = x @ x.T
  anisotropy = mean(sim[i, j] for i != j)
  ```

- **Pool:** `pooled_both`, subsample 2048 seed 0.
- **`battery_id`:** `eval_holdout` / `eval_quantized_holdout`.
- **Unit:** `[0, 1]` (float64 cosine mean).
- **Citation -- named, with a mismatch to flag, see §16:** Ethayarajh, EMNLP 2019, §3.4/§4.1
  (arXiv:1909.00512); Godey et al., 2024, §3.1 (arXiv:2401.12143); LoopFormer §4.3.
- **Why this metric:** catches an unstructured-cone collapse that a ranking score alone can
  hide -- `rank.recall@1` can read fine while the space has quietly narrowed.
- **Why this battery:** the same pool alignment/uniformity/effective-rank-entropy read, so one
  receipt's four representation-geometry numbers describe one fixed set of vectors.
- **Falsifies:** `gate.not_anisotropic` (`< 0.9`) -- a collapse tripwire, **not** a "lower is
  better" ranking rule (§19.2).
- **Comparison rule:** same holdout size and subsampling seed. **Never** compared numerically
  to a number from Ethayarajh/Godey/LoopFordmer or another project's per-token anisotropy --
  see §16 in full before citing this field beside a paper's own figure.
- **Sameness special-case:** none.

### 12.4 `repr.alignment` / `repr.uniformity`

- **Formula** (Wang & Isola, ICML 2020, arXiv:2005.10242; alpha=2, t=2):

  ```python
  alignment  = mean(||f(x_i) - f(y_i)||^2)                       # matched pairs, lower better
  uniformity = log(mean(exp(-2 * ||f(x_i) - f(x_j)||^2) for i != j))   # pooled, lower better
  ```

- **Pool:** `matched` (alignment: anchor row `i` against positive row `i`); `pooled_both`
  (uniformity: same pool as anisotropy).
- **`battery_id`:** `eval_holdout` / `eval_quantized_holdout`.
- **Unit:** alignment `>= 0`; uniformity a log-potential, more negative is more spread.
- **Citation:** Wang & Isola, ICML 2020, Theorem 1, §4.1.
- **Why this metric:** established contrastive-geometry pair; g7 spec L4 -- either alone is
  gameable (collapse yields near-perfect alignment; a random, unstructured cloud yields good
  uniformity), so this project reports them jointly by rule, never singly.
- **Why this battery:** matched pairs (alignment) and `pooled_both` (uniformity) come from the
  identical holdout the anisotropy/effective-rank-entropy fields use.
- **Falsifies:** high `rank.recall@1` from a cone (good alignment, bad uniformity) or from an
  unstructured cloud (good uniformity, bad alignment) -- reading either number alone as
  "healthy" is the exact failure mode this pair exists to catch.
- **Comparison rule:** same rule as §12.3.
- **Sameness special-case:** none.

### 12.5 `repr.emb_std_anchor`

- **Formula:** `embeddings.std(dim=0).mean()` -- mean per-feature standard deviation across
  the batch dimension.
- **Pool:** anchors only, held-out split.
- **`battery_id`:** `train_holdout`.
- **Unit:** scalar, embedding-space units.
- **Citation:** none external -- this project's own collapse signal.
- **Why this metric:** the cheapest full-collapse check in the pipeline -- near zero means
  every input maps to nearly the same vector -- computed inline inside `evaluate()` /
  `evaluate_graded()` before the wider eval battery's SVD-based fields ever run.
- **Why this battery:** it is free (already computed as a side effect of the training
  receipt's own held-out pass); no reason to defer a full-collapse check to the separate eval
  battery.
- **Falsifies:** a receipt reporting `gate.beats_untrained_train: True` while
  `repr.emb_std_anchor` sits at its untrained-baseline floor -- a ranking win a healthy encoder
  cannot legitimately produce alongside a collapsed anchor side.
- **Comparison rule:** same side (anchor, not pooled), same split, same region (§2.3(f),
  unchanged under v2). Not comparable to `repr.anisotropy`/`repr.alignment`/`repr.uniformity`
  (different pool: anchor-only here vs. pooled-both there).
- **Sameness special-case:** none.

### 12.6 `rank.recall@k` / `rank.mrr` / `rank.ndcg@10` (closed pool)

- **Formula:** §2.1/§2.2/§3.3's single-relevant `recall_at_k` / `mean_reciprocal_rank` /
  `ndcg_at_k` -- with exactly one relevant item, ndcg reduces to `mean(1[rank<=k] /
  log2(rank+1))`. Named after Wang et al. 2013 Def. 1, but this is the **degenerate
  single-relevant special case**, not the multi-relevant TREC/BEIR formula (§12.7).
- **Pool:** `biencoder_holdout`, size `rank.candidates` (project default 512).
- **`battery_id`:** `eval_holdout` / `eval_quantized_holdout`.
- **Unit:** `[0, 1]`.
- **Citation:** Wang et al. 2013, arXiv:1304.6480, Def. 1 (formula shape only -- see the
  degenerate-case note above).
- **Why this metric:** the headline closed-pool ranking triad every card prints -- recall@1/@10
  for "did the model find its positive," `mrr`/`ndcg@10` for rank-position sensitivity that
  recall alone cannot show.
- **Why this battery:** encodes the whole holdout as one closed candidate pool, identical in
  shape to §2's `held_out.*` construction -- which is why `rank.recall@1` (eval battery) is
  numerically identical to `held_out.recall@1` (train battery) for the same checkpoint (a
  verified receipt pair, §10).
- **Falsifies:** `gate.beats_untrained_eval`.
- **Comparison rule:** same `k`, same pool size, same fingerprint+scheme, same checkpoint sha
  (§10, in full).
- **Sameness special-case:** `rank.recall@1` (eval battery) vs. `held_out.recall@1` (train
  battery) is a legitimate sameness guard **only** on the same sha/holdout. Never compared to
  `beir.recall@k` (§12.7 -- different pool and multi-relevant formula) or to
  `quant.plan_recall@1`/`quant.artifact_recall@1` (§12.8) except through §14's
  `assert_sameness` special case.

### 12.7 `beir.ndcg@10` / `beir.recall@k` / `beir.mrr`

- **Formula, ndcg:** canonical TREC/BEIR nDCG (multi-relevant), Wang et al. 2013 Def. 1:

  ```python
  DCG@k  = sum(G(rel_i) / log2(i + 1) for i in 1..k)
  IDCG@k = DCG@k of the ideal ranking of the qrels
  nDCG@k = DCG@k / IDCG@k          # 0 if IDCG == 0
  ```

- **Formula, recall/mrr:** this project's `rank_metrics` (§7.2) -- each query's best-scoring
  gold index kept, every other gold masked to `-inf`, then §2's `recall_at_k` /
  `mean_reciprocal_rank` reused exactly (not an approximation -- no other gold can outrank the
  best one).
- **Pool:** `fiqa_corpus` (57,638 passages, the real task, W4's headline pool) or
  `fiqa_split` (in-split only, easier, never the headline number).
- **`battery_id`:** `beir_fiqa_corpus` / `beir_fiqa_split`.
- **Unit:** `[0, 1]`.
- **Citation:** Thakur et al., BEIR, NeurIPS 2021 Datasets, arXiv:2104.08663 v4, §3.3.
- **Why this metric:** the one battery in this project that ranks against the full external
  corpus with a genuine lexical (BM25) baseline, not a closed in-holdout pool whose only floor
  is a random init.
- **Why this battery:** the `memory` region's W4 gates read from here specifically because a
  512-item closed pool cannot show whether a region beats word-counting on a real 57,638-
  document retrieval task; FiQA judges ~2.6 passages relevant per question, multi-relevant
  unlike every closed-pool battery in §2/§3.
- **Falsifies:** W4 gates `b_full_pool_thresholds`, `c_beats_bm25`, `d_beats_random_init`
  (§7.4).
- **Comparison rule:** same pool mode (`corpus` vs. `split`), same split (`dev`/`test`, never
  `train` -- ranking `train` measures memorisation, §7.1(f)).
- **Sameness special-case:** none across `beir.*` and `rank.*` -- same field-family names
  (`recall@k`, `mrr`), disjoint pool and relevance shape (§7.2(f), unchanged under v2).
- **CRITICAL -- never alias:** `beir.ndcg@10` is **not implemented** in CSD today (§1.1's
  gap). A card that prints it before the adapter (§18) exists is printing a field with no
  producer. When it ships, it must **never** be written to `metrics["rank.ndcg@10"]` -- that
  would collide with the closed-pool number under one key (§13's "do not retire
  `rank.ndcg@10`" rule exists for exactly this reason).

### 12.8 `quant.plan_recall@1` / `quant.artifact_recall@1` / `quant.drop_recall@1` / `quant.compression_ratio`

- **Formula:** all four `recall@1` figures are §2's `recall_at_k(k=1)`, on §2's training
  held-out battery. `plan_recall@1` is written directly under that name today
  (`scripts/csd-quantize.py:270`; v1 name `quantized_metric`, retired -- §15), measured
  in-memory, `apply_plan()` applied, before any file is written. `artifact_recall@1` =
  `rank.recall@1` from a `kind:
  "eval-quantized"` receipt (the actual packed `.ptq.pt` file read back off disk, unpacked to
  fp32). `drop_recall@1 = fp32_metric_recomputed - plan_recall@1` (plan side) or the
  fp32-eval-vs-eval-quantized delta (artifact side) -- always **one named metric, one battery**
  (MM §4-5), never a unitless "drop." `compression_ratio = fp32_bytes / max(1, stored_bytes)`.
- **Pool:** §2's held-out split, in-memory (plan) vs. the same holdout via the unpacked
  artifact (artifact side).
- **`battery_id`:** `quant_plan` / `eval_quantized_holdout`.
- **Unit:** `recall@1` figures `[0, 1]`; `compression_ratio` a unitless multiplier.
- **Citation:** none external -- this project's own PTQ pipeline (`src/cogsyndelta/quant/
  ptq.py`).
- **Why these metrics:** `plan_recall@1` is cheap (in-memory, before a file is written) and
  drives the greedy sensitivity search during quantization; `artifact_recall@1` is the only
  number reflecting the bytes actually published -- the plan-vs-artifact gap this project has
  already been bitten by once (§4).
- **Why this battery:** `quant_plan` reuses §2's `eval_fn` for search speed (a single
  `recall@1`, nothing wider); `eval_quantized_holdout` reruns the **full** eval battery
  (`rank.*`/`repr.*`/`eff.*`) against the loaded packed artifact, because a compression ratio
  alone says nothing about `eff.latency_*` or `repr.anisotropy` movement under quantization.
- **Falsifies:** `within_budget` (`drop <= tolerance`), measured on the plan -- a `True` here
  can still diverge from the published bytes, which is exactly what an independent
  `artifact_recall@1` reading catches.
- **Comparison rule:** `quant.compression_ratio` is a payload/storage ratio, never a latency
  claim (§19.1).
- **Sameness special-case (the one explicitly licensed cross-`battery_id` comparison, MM §4):**
  `quant.plan_recall@1` vs. `quant.artifact_recall@1` on the **identical checkpoint sha and
  holdout** is a legitimate `assert_sameness` check. Refusing this comparison violates MM §4
  (§14's schema-falsifier #2) -- "never compare across `battery_id`" is stricter than MM
  actually requires and must not be over-applied here.

### 12.8.1 Visual: `quant.plan_probe_top1` / `quant.artifact_probe_top1` / `quant.drop_probe_top1`

Visual receipts do **not** write `quant.*_recall@1`. The measured quantity is EuroSAT
linear-probe top-1 (`probe.top1`), not closed-pool `recall_at_k(k=1)`.

- **Formula:** `plan_probe_top1` = in-memory plan's EuroSAT probe top-1
  (`quantize_visual_region()`, `scripts/csd-quantize.py:340-460`). `artifact_probe_top1` = packed
  EMA-target-encoder artifact's EuroSAT probe top-1
  (`benchmark_visual_region_quantized()`, `scripts/csd-benchmark.py:805-883`).
  `drop_probe_top1 = fp32_metric_recomputed - plan_probe_top1`. `quant.compression_ratio`
  is the same byte-accounting name as the text receipts (`fp32_bytes / stored_bytes`).
- **Battery:** EuroSAT official test linear probe, `n_eval=5400`, 10-way, chance 0.1.
  Fashion t10k transfer is a different named metric (`transfer.top1`), not these fields.
- **`battery_id` / pooling:** `quant_plan` / `linear_probe` (plan); `eval_quantized_holdout`
  / `linear_probe` (artifact).
- **Unit:** top-1 accuracy `[0, 1]`.
- **Sameness special-case:** `quant.plan_probe_top1` vs. `quant.artifact_probe_top1` on
  the identical checkpoint sha is the visual counterpart of §12.8's text pair. Do not
  compare either to `quant.plan_recall@1` or `rank.recall@1`.

### 12.9 `gate.beats_untrained_train` / `gate.beats_untrained_eval`

- **Formula:**

  ```text
  gate.beats_untrained_train = baseline_sane AND final[m] > max(baseline[m], chance[m]) + 0.01
  gate.beats_untrained_eval  = rank.recall@1 > untrained_baseline.recall@1        # no margin
  ```

- **Pool / `battery_id`:** `train_holdout` (train form, §1) / `eval_holdout` +
  `eval_quantized_holdout` (eval form, §3.13).
- **Unit:** boolean.
- **Citation:** none external -- both are this project's own gates.
- **Why two canonical names:** §3.13(f) already flags that the **train** and **eval** receipts
  use the same English field name (`beats_untrained`) for two different predicates -- one
  margined and preconditioned on `baseline_sane`, one a bare inequality. v2 gives each its own
  name specifically so a reader cannot conflate a `passed: True` under one with the other by
  field name alone.
- **Falsifies:** §1's `_beats_untrained_gate` (train form) / §3.13's unmargined check (eval
  form), respectively.
- **Comparison rule:** never compare a `gate.beats_untrained_train` boolean to a
  `gate.beats_untrained_eval` boolean as if they tested the same thing.
- **Sameness special-case:** none.

---

## 13. Retire list (`csd-metrics/v2` §3.2)

- **`rank.map` on the 512-pair closed pool -- retire as an independently displayed column.**
  Single relevant item per query means MAP = MRR by construction (§3.4, `benchmark.py:58-70`).
  Keep it only as a **sameness guard** (`map == mrr`). On FiQA (§12.7) MAP and MRR genuinely
  diverge -- there, both names are legitimate under the `beir.` prefix, not `rank.`.
- **`rank.precision@10` on that pool -- retire as an independently displayed column.**
  `precision@10 == recall@10 / 10` exactly with one positive (§3.5). Keep only as a sameness
  guard (`p@10 == r@10/10`).
- **Bare `effective_rank` -- forbid the name.** Three formulas exist in this codebase
  (§9/§12.1/§12.2); entropy-effective-rank and participation-ratio rank **disagree in sign** on
  this project's own production regions (PR ratios `0.66x-1.30x` vs. entropy ratios
  `1.16x-1.84x`, §17). A field, card cell, or design-doc sentence that says "effective rank"
  without saying which formula is not naming a measurement.
- **`repr.anisotropy < 0.9` as a discriminating admission test -- retire from gates, keep
  recording.** Untrained W1 pooled mean-pairwise cosine sits at `0.84-0.98`; trained text sits
  at `0.015-0.123` (W1). Do not ship a tighter or looser bound on this field without a study.
- **Rename `uses_its_dimensions` to `repr.effective_rank_entropy_ratio`.** The `0.05` floor is
  slack, not a tight bound -- W1's own production regions sit at `~0.45-0.51`
  (`csd-benchmark.py:453`).
- **A single `beats_untrained` spanning train and eval -- retire.** Split into
  `gate.beats_untrained_train` / `gate.beats_untrained_eval` (§12.9).
- **Untrained CKA on seed-0 twins -- never a baseline.** W1's cross-region CKA is `1.0` **by
  construction** when regions share `seed=0` and an identical config; it says the twins are
  literally the same weights, not that anything was measured.

**Reject or retire -- latent-reasoning candidates (g7 §2.3), for context, not native to any
receipt above:**

| Candidate | Verdict | Reason |
|---|---|---|
| Mean successive KL (`loop.kl_succ_mean`) as a quality ranking | Reject as gate | Established only as Huginn's early-exit heuristic (§19.4), not as evidence of reasoning quality. |
| `tau = 5e-4` copied onto CSD | Reject | Huginn's exit threshold is tuned for a 65,536-way readout; not portable onto a 2-/10-way toy without re-derivation. |
| Tokens/s as the reasoning-efficiency headline | Reject for latent loops | Depth/thoughts is the right unit for a recurrent latent core, not sequence length. |
| Mutual information between layers | Reject | Kornblith Sec. 4: an invertible network makes MI equal to `H(input)` regardless of what the layers do. |
| A blended "quality score" | Reject | `compare()` names regressions per-field and refuses a blend by design (`metrics.py:763-843`). |
| Coconut extra sequence slots as CSD's `K` | Reject | KV grows with thoughts (Coconut is horizontal); CSD's latent loop is vertical, no sequence growth. |
| Control-task accuracy alone, or alignment/uniformity alone | Reject as ranking | Selectivity and joint geometry are the point (§12.4, Hewitt & Liang). |
| `g6-switch-toy/` as an `acc(K)` host | Reject | The routing-load-balance toy has no `K` -- not a latent-reasoning metric. |

---

## 14. Schema stamp and refuse predicate

```text
metrics_schema: "csd-metrics/v2"
# v1 = today's mixed names (effective_rank, map, precision@10, unitless drop)
# v2 = table in §12; aliases for v1 names allowed in a deprecation map (§15), not in gates
```

A harness **refuses** to `compare()` two numbers unless **all** of the following hold --
reproduced from `g7-latent-eval-metrics.md` §3.3, which is itself §10 of this document plus a
`battery_id` and a `pooling` axis:

```text
same metrics_schema
same corpus.fingerprint AND fingerprint_scheme
same battery_id  in {
  train_holdout, eval_holdout, eval_quantized_holdout,
  train_graded, train_token_rank,
  quant_plan,
  beir_fiqa_corpus, beir_fiqa_split,
  mteb_<task>,            # future adapter; never alias into rank.*
  latent_loop             # future; not a native battery today
}
same k                       # for @k metrics; else None == None
same pooling  in {
  anchor, matched, pooled_both,
  anchor_pooled, anchor_token_global, graded_left,
  fiqa_corpus, fiqa_split
}
same checkpoint sha256   (or a documented successor via train-receipt binding)
same region / producer.component          # this doc's §10 item 1
same code_revision.git_sha                # this doc's §10 item 6
same seed                                 # this doc's §10 item 7
```

**IMPLEMENTED as a function, not a comment.** `compare()`
(`src/cogsyndelta/eval/metrics.py:763-843`) IS this refuse-predicate: it walks every
`MetricIdentity` field in the declared order above and returns a `ComparisonRefusal` (not a
diff) the moment one differs, before ever touching `values`
(`src/cogsyndelta/eval/metrics.py:806-830`) -- see `tests/test_eval_metrics.py`'s
per-identity-key parametrisation and `tests/test_guards_can_fail.py`'s DEFECT 7 mutation
proofs. The v1 version this replaced diffed only shared keys and checked none of schema,
fingerprint, `battery_id`, pooling, or sha; shipping v2 field names without this predicate
would have re-created the exact plan-vs-artifact misread §4 already documents once. Production
callers of `compare()` are still limited to tests, though: no training, eval, or publish
script invokes it today (confirmed by a repo-wide search for its call sites outside
`tests/` and its own docstrings, 2026-09), so the predicate is correct and proven but not yet
wired into a comparison a human or CI job would actually run.

**Special case, not a generic `compare()`:** `assert_sameness` for `map == mrr`, `p@10 ==
r@10/10` (§13), `held_out.recall@1` vs. `rank.recall@1` on the same sha (§10 item 3), and
`quant.plan_recall@1` vs. `quant.artifact_recall@1` on the same sha/holdout (§12.8). These
*cross* `battery_id` by design. If a refuse-function rejects the last pair, it has violated MM
§4 -- see the schema falsifiers below.

`train_holdout` and `eval_holdout` reconstruct the **same split** (fingerprint-gated
`build_splits`) via different code paths. A harness may compare the two `recall@1`s as a
sameness guard; it may **not** compare `held_out.emb_std` to `repr.anisotropy` (different
pools -- anchor-only vs. pooled-both, §12.5).

**Schema falsifiers -- write these down before the `csd-metrics/v2` patch ships, and check
them against the first real patch, not only against this design:**

1. If two receipts with different `battery_id` still delta a shared `"recall@1"` after the
   patch -> the refuse-function is theatre -> **stop**.
2. If `quant.plan_recall@1` vs. `quant.artifact_recall@1` on the same sha/holdout is refused ->
   the predicate over-copied "never compare across battery_id" past the one case MM §4
   explicitly licenses.
3. If `repr.effective_rank_pr` appears as a written field anywhere -> the patch implemented the
   old parent design's table, not what §12.1 actually says (that name is forbidden -- §12.1's
   "do not invent" note).

---

## 15. v1 -> v2 deprecation map

This maps each v1 (pre-migration) field name to the v2 name that replaced it. "Written
today" varies by row: `repr.effective_rank`/`.effective_rank_ratio`, `quantized_metric`,
`drop`, `compression_ratio`, `gates.beats_untrained`, `rank.map` and `rank.precision@10`
are **fully renamed** -- the v1 name is no longer written by any production receipt, only
the v2 name in the next column is. The one exception is `rank.recall@1`, whose row's own
note explains why it stays written under that name for its own eval purposes and only
takes the `quant.` name in the one plan-vs-artifact comparison §12.8 licenses.

| v1 field (pre-v2 name) | receipt `kind` | v2 canonical name | note |
|---|---|---|---|
| `repr.effective_rank` | eval, eval-quantized | `repr.effective_rank_entropy` | same formula (§9/§12.1), name now says which one |
| `repr.effective_rank_ratio` | eval, eval-quantized | `repr.effective_rank_entropy_ratio` | rename of the `uses_its_dimensions` ratio (§13) |
| `token_aware.final_block_rank.pooled_pr_rank` | train | `token.pooled_pr_rank` | §12.2 |
| `token_aware.final_block_rank.token_global_pr_rank` | train | `token.global_pr_rank` | §12.2 |
| `token_aware.final_block_rank.pooled_entropy_rank` | train | `token.pooled_entropy_rank` | §12.2 |
| `token_aware.final_block_rank.token_global_entropy_rank` | train | `token.global_entropy_rank` | §12.2 |
| `quantized_metric` | quant | `quant.plan_recall@1` | §12.8 |
| `rank.recall@1` (read specifically for a plan-vs-artifact delta) | eval-quantized | `quant.artifact_recall@1` | context-dependent -- stays `rank.recall@1` for its own eval purposes; take the `quant.` name only when the comparison in play is the plan-vs-artifact one (§12.8's sameness special-case) |
| `drop` | quant | `quant.drop_recall@1` | one named metric, one battery (MM §4-5) |
| `quant.plan_recall@1` (on a **visual** receipt) | visual quant | `quant.plan_probe_top1` | §12.8.1 — the quantity was never recall@1 |
| `quant.artifact_recall@1` (on a **visual** eval-quantized receipt) | visual eval-quantized | `quant.artifact_probe_top1` | §12.8.1 |
| `quant.drop_recall@1` (on a **visual** receipt) | visual quant | `quant.drop_probe_top1` | §12.8.1 |
| `compression_ratio` | quant | `quant.compression_ratio` | §12.8 |
| `gates.beats_untrained` (train receipt, margined) | train | `gate.beats_untrained_train` | §12.9 |
| `gates.beats_untrained` (eval receipt, unmargined) | eval, eval-quantized | `gate.beats_untrained_eval` | §12.9 -- same v1 name as the row above, different receipt kind, different predicate |
| `rank.map` | eval, eval-quantized | *(retired as an independent name)* | keep as sameness guard `map == mrr` only (§13) |
| `rank.precision@10` | eval, eval-quantized | *(retired as an independent name)* | keep as sameness guard `p@10 == r@10/10` only (§13) |
| bare "effective rank" (undifferentiated prose) | any | *(forbidden -- name one of `repr.effective_rank_entropy`, `token.pooled_pr_rank`, `token.global_pr_rank`, `token.pooled_entropy_rank`, `token.global_entropy_rank` explicitly)* | §13 |

Fields **not** in this table (`rank.recall@k`, `rank.mrr`, `rank.ndcg@10`, `rank.candidates`,
`repr.anisotropy`, `repr.alignment`, `repr.uniformity`, `eff.*`, `held_out.*`,
`untrained_baseline.*`, `graded_held_out.*`, `corpus.*`, `contamination.*`, `code_revision.*`)
are **already** their own v2 canonical name -- v2 does not rename them, §12's entries for them
exist to add the missing why/falsifies/comparison-rule columns, not to change the string.

---

## 16. The anisotropy naming caveat

`repr.anisotropy` (§3.9/§12.3) is named after three papers that measure a **different
surface**: Ethayarajh (EMNLP 2019, §3.4/§4.1, arXiv:1909.00512) and Godey et al. (2024, §3.1,
arXiv:2401.12143) measure mean cosine similarity over **token embeddings drawn from a full
pretraining corpus**; LoopFormer §4.3 reuses the same construction for a recurrent core's
per-layer states. This project's `repr.anisotropy` measures mean cosine similarity over
**pooled, matched-pair holdout embeddings**, subsampled to at most 2048 rows, seed 0 (§3.9(d)).

**Do not compare CSD's `repr.anisotropy` numerically against a number reported in those
papers, or against another project's per-token anisotropy figure.** Same citation, same
underlying "mean off-diagonal cosine" idea, **different surface** -- a pooled-holdout number
and a token-in-corpus number are not the same measurement even when both are called
"anisotropy" and both cite the same paper. This is the identical rule §7's BEIR-FiQA module
states for `recall@k` ("two measurements that happen to share a name"), applied to
`anisotropy`.

---

## 17. W1: participation-ratio and entropy rank disagree in sign

Measured on this project's own four production text checkpoints
(`docs/design/evidence/w1-token-rank-2026-09-02/results.json`, `generated_utc`
2026-09-03T01:34:22Z), token-global vs. pooled surfaces:

| region | PR pooled | PR token | PR ratio | H pooled | H token | H ratio | n_tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| `code` | 42.63 | 28.09 | **0.66x** | 114.98 | 138.87 | **1.21x** | 26757 |
| `compress` | 45.33 | 35.57 | **0.78x** | 126.86 | 147.49 | **1.16x** | 9636 |
| `retrieve` | 40.71 | 40.65 | **1.00x** | 117.72 | 142.86 | **1.21x** | 5114 |
| `vl_latent` | 18.07 | 23.54 | **1.30x** | 129.83 | 239.09 | **1.84x** | 32768 |

**Three of four production PR ratios are below 1.0; all four entropy ratios are above 1.0.**
This is not a bug in either function -- it is what the two formulas are built to weight
differently:

- **Participation ratio down-weights the tail.** `(sum(s_i^2))^2 / sum(s_i^4)` is dominated by
  a few large singular values; a token surface with even a slightly heavier tail than its
  pooled counterpart pulls the PR ratio below 1.
- **Entropy rank up-weights the tail.** `exp(-sum(p_i * log(p_i)))` gives every nonzero
  singular value a log-linear contribution; the same token surfaces read as **higher** rank
  than pooled under this formula on all four regions.

Never compare entropy-rank to PR-rank, including as a ratio of mixed formulas (§9/§12.2's
comparison rule, restated here because this is the table that makes the disagreement
concrete). Citation for the PR side's finite-sample-bias caveat: Chun, Canatar, Chung, Lee,
arXiv:2509.26560 v1, §2.2 -- "CSD applies no bias correction. Unmeasured on W1 surfaces."

**Downstream consequence for the W4 retrain gate.** The aligned harness's control arm (both
token-aware terms at `0.0`) already clears `token_global_pr_rank >= 2.0 * pooled_pr_rank` at
50 steps: control `2.0191x`, `token_only` `2.0242x`, `both_on` `2.2412x`
(`docs/design/evidence/w4-control-arm-2026-09-03/`). The gate does **not discriminate** at this
step count -- a `passed: True` here is not by itself evidence the token-aware objective moved
anything (§12.2's falsifies entry, restated).

---

## 18. The dual-harness principle

**Native receipts stay the source of truth for gates.** Industry harnesses (MTEB, BEIR,
lm-eval, bigcode, HELM, MLPerf) are **adapters** that write **their own** JSON, cross-
referenced against a native receipt, **never translated into or aliased onto a native field
name.** Dual-harness remains **design-only today**: no MTEB/BEIR adapter exists in the opened
CSD tree, and `pyproject.toml`'s core dependencies carry no `mteb`/`beir`/`datasets`/`sklearn`/
`lm-eval`/`helm` entry.

```text
                    +- csd-benchmark.py  ->  Receipt (rank.* / repr.* / eff.*)
checkpoint.pt  --+
                    +- csd-eval-bridge (separate repo)
                         +- EncoderProtocol.encode(DataLoader, *, task_metadata, ...)
                         |     -> ResultCache / BenchmarkResults JSON  (never ModelResult)
                         +- (later) CsdLM(LM) -> lm-eval / bigcode JSON
                              only after a coda maps s to p in the simplex Delta^(|V|-1)
```

**Where the bridge lives -- not inside `tzervas/CogSynDelta`.** Adding MTEB/BEIR/lm-eval as a
project dependency would be a new heavy dependency in the model repo for what is fundamentally
a harness concern. Candidates: extend `tzervas/model-matrix` (which already consumes
`metrics["rank.recall@1"]` and, per its own `DESIGN.md` §1, **must not** import a project
package), or a thin new `csd-eval-bridge` repo. CogSynDelta keeps `TextEncoder.encode` and its
own receipts; the bridge depends on CSD as a library, not the reverse. This follows the
operator's `tooling-lives-in-its-own-repo` rule the same way the matrix harness itself does
(§P7.3, `program/REMAINING.md`).

**Cross-reference shape (`detail.external`, never `metrics`):**

```json
{
  "harness": "mteb",
  "harness_version": "<pkg version>",
  "benchmark": "task:FiQA2018",
  "task": "FiQA2018",
  "main_metric": "ndcg_at_10",
  "main_value": 0.0,
  "result_sha256": "...",
  "checkpoint_sha256": "...",
  "corpus_fingerprint": "...",
  "metrics_schema": "csd-metrics/v2",
  "pooling": "masked_mean_then_proj",
  "similarity": "cosine",
  "max_len": 96,
  "normalised": true,
  "battery_id": "mteb_FiQA2018"
}
```

Putting an industry nDCG@10 at `metrics["rank.ndcg@10"]` would collide with the closed-pool
number this project already writes there (§12.7's CRITICAL note). Industry blobs go in
`detail`, never in `metrics` -- `passed` (a receipt's own top-level gate summary) requires a
non-empty `gates` block and all entries true; it does not, and must not, read `detail`.

**Cross-reference key** a bridge uses to bind an industry result back to the native
checkpoint it scored: `(checkpoint_sha256, corpus_fingerprint, metrics_schema, battery_id,
pooling, k, region, git_sha, seed)` -- the same nine axes §14's refuse predicate checks.

---

## 19. Standing statements

A card, receipt, or design doc must not contradict any of these without a new measurement
that specifically revisits it:

1. **The PTQ compression ratio is a payload/storage ratio, not a speed or throughput claim.**
   `quant.compression_ratio` (§12.8/§5.2(f)) says nothing about `eff.latency_*_ms` /
   `eff.throughput_per_s` on its own -- those are measured separately (§3.8) and are not
   guaranteed to move proportionally, since sub-byte codes still get unpacked to fp32 before a
   forward pass runs.
2. **Anisotropy (and alignment/uniformity/effective-rank) are diagnostics, not scores.**
   None of `repr.anisotropy`, `repr.alignment`, `repr.uniformity`,
   `repr.effective_rank_entropy` should be read as "lower/higher is better" in isolation from
   a ranking metric (`rank.recall@k`/`rank.mrr`) -- they exist to catch a collapsed
   representation space that a ranking score alone can miss (§11.2, §12.3-§12.5).
3. **Per-token metrics apply only to token-mappable surfaces.** There is **no** token-mappable
   composed read-out today: `CausalLM` exists unplugged (`forward` -> logits, an optional CE
   loss, **no** `generate`, no coda, no adapter). Do not run lm-eval, bigcode-evaluation-
   harness, or HELM on a region that only emits `[B, D]` latents (g7 §1.2/§1.3). No CSD
   fingerprint scheme exists for HumanEval/HellaSwag/GSM8K today -- `corpus.fingerprint`
   (§6.3) is parquet-shard hashing for pair corpora, not a benchmark-item fingerprint.
4. **The latent-space metrics are logged-only, not gates, until a pre-registered validation
   study licenses them.** `loop.acc@k`, `loop.kl_succ_mean` (with `readout_id`), `loop.nonmono`
   (log only), `probe.{acc_ling,acc_ctrl,sel}`, and `route.{aux,aux_coef,H,load.*}` (g7 §2.2)
   are recorded, never gated on, until that study exists. Never compare `loop.acc@k` to
   `rank.recall@1` -- different objects entirely (test-time latent-compute accuracy vs. a
   closed-pool retrieval score). Never compare `route.aux` to a Switch Transformers run that
   folded in `alpha` without dividing it back out (Fedus, Zoph, Shazeer, JMLR 23(120) 2022,
   arXiv:2101.03961, eqs. 4-6) -- CSD's own aux helper is unweighted
   (`N * sum(f_i * P_i)`; `aux_coef` multiplies separately, default `1.0`), so a raw `aux`
   value from CSD and a raw `aux` value that already folded in `alpha=1e-2` are not the same
   scale.

---

## Field index

Receipt field -> section. `kind` is the receipt this field is written into
(`train` = `regions/pretrain.py`, `quant` = `scripts/csd-quantize.py`, `eval`/`eval-quantized`
= `scripts/csd-benchmark.py`).

| field | `kind` | section |
|---|---|---|
| `held_out.recall@1`, `.recall@10`, `untrained_baseline.recall@1`, `.recall@10` | train | [§2.1](#21-held_outrecall1-held_outrecall10-untrained_baselinerecall1-recall10) |
| `held_out.mrr`, `untrained_baseline.mrr` | train | [§2.2](#22-held_outmrr-untrained_baselinemrr) |
| `held_out.emb_std`, `untrained_baseline.emb_std`, `graded_held_out.emb_std` | train | [§2.3](#23-held_outemb_std-untrained_baselineemb_std-graded_held_outemb_std) |
| `graded_held_out.spearman`, `untrained_graded_baseline.spearman` | train | [§2.4](#24-graded_held_outspearman-untrained_graded_baselinespearman) |
| `chance`, `beats_untrained` (training receipt) | train | [§1](#1-the-untrained-baseline), [§2.5](#25-chance-beats_untrained) |
| `capability_per_param` (training receipt, top level) | train | [§2.6](#26-capability_per_param) |
| `token_aware.final_block_rank.*` | train | [§2.7](#27-token_awarefinal_block_rank) |
| `contamination.*` (training receipt) | train | [§6.2](#62-contamination-training-receipts-pair-level-form-train_pairs_seen-train_pairs_removed-eval_pairs-gated_channels-channels-eval_duplicate_positives) |
| `corpus.fingerprint`, `corpus.fingerprint_scheme` | train | [§6.3](#63-corpusfingerprint-corpusfingerprint_scheme) |
| `code_revision.*` | train, quant, eval | [§10](#10-how-to-compare-two-numbers-legitimately) item 6 |
| `metrics["rank.recall@1"]` .. `["rank.recall@10"]` | eval, eval-quantized | [§3.1](#31-rankrecall1-rankrecall5-rankrecall10) |
| `metrics["rank.mrr"]` | eval, eval-quantized | [§3.2](#32-rankmrr) |
| `metrics["rank.ndcg@10"]` | eval, eval-quantized | [§3.3](#33-rankndcg10) |
| `metrics["rank.map"]` | *(retired -- not written by v2, see §3.4)* | [§3.4](#34-rankmap) |
| `metrics["rank.precision@10"]` | *(retired -- not written by v2, see §3.5)* | [§3.5](#35-rankprecision10) |
| `metrics["rank.candidates"]` | eval, eval-quantized | [§3.6](#36-rankcandidates) |
| `metrics["eff.parameters"]`, `["eff.stored_mb"]`, `["eff.capability_per_param"]`, `["eff.capability_per_mb"]` | eval, eval-quantized | [§3.7](#37-effparameters-effstored_mb-effcapability_per_param-effcapability_per_mb) |
| `metrics["eff.latency_p50_ms"]` .. `["eff.peak_vram_mb"]` | eval, eval-quantized | [§3.8](#38-efflatency_p50_ms-latency_p95_ms-latency_p99_ms-throughput_per_s-peak_vram_mb) |
| `metrics["repr.anisotropy"]` | eval, eval-quantized | [§3.9](#39-repranisotropy) |
| `metrics["repr.alignment"]` | eval, eval-quantized | [§3.10](#310-repralignment) |
| `metrics["repr.uniformity"]` | eval, eval-quantized | [§3.11](#311-repruniformity) |
| `metrics["repr.effective_rank"]`, `["repr.dimensions"]`, `["repr.effective_rank_ratio"]` | eval, eval-quantized | [§3.12](#312-repreffective_rank-reprdimensions-repreffective_rank_ratio) |
| `gates.beats_untrained`, `.not_anisotropic`, `.uses_its_dimensions` | eval, eval-quantized | [§3.13](#313-gatesbeats_untrained-gatesnot_anisotropic-gatesuses_its_dimensions) |
| `provenance.stored_bytes_definition` | eval, eval-quantized | [§3.7](#37-effparameters-effstored_mb-effcapability_per_param-effcapability_per_mb) |
| `fp32_bytes` | quant | [§5.1](#51-fp32_bytes) |
| `stored_bytes`, `compression_ratio` | quant | [§5.2](#52-stored_bytes-compression_ratio) |
| `width_histogram`, `bits`, `fp32_tensors`, `promotions` | quant | [§5.3](#53-width_histogram-bits-fp32_tensors-promotions) |
| `fp32_metric_recomputed`, `fp32_metric_receipt`, `quantized_metric`, `drop`, `within_budget`, `tolerance` | quant | [§5.4](#54-fp32_metric_recomputed-fp32_metric_receipt-quantized_metric-drop-within_budget-tolerance), [§4](#4-the-battery-distinction----quant-vs-eval) |
| `corpus_fingerprint` (quant receipt, top level) | quant | [§6.3](#63-corpusfingerprint-corpusfingerprint_scheme) |
| BEIR-FiQA `recall@1`/`recall@10`/`recall@100`/`mrr` | (evidence/gate scripts, not a receipt `kind` above) | [§7.2](#72-rank_metrics-recall1-recall10-recall100-mrr-multi-relevant) |
| W4 gates `a_beats_both_parents` .. `e_retrain_gate` | (evidence/gate scripts) | [§7.4](#74-the-w4-gates-a_beats_both_parents-e_retrain_gate) |

### `csd-metrics/v2` canonical names (§12)

A current eval / eval-quantized receipt writes `repr.effective_rank_entropy`,
`.effective_rank_entropy_ratio`, `repr.anisotropy`, `repr.alignment`, `repr.uniformity`, and
`repr.emb_std_anchor` (§12.1, §12.3, §12.4, §12.5) under these exact names -- NOT "not yet
written." `quant.*` (§12.8) is written by production code too, and directly, not through
the publish script's read-time normalisation: `csd-quantize.py` writes
`quant.plan_recall@1`, `quant.drop_recall@1`, and `quant.compression_ratio` into the quant
receipt, and `csd-benchmark.py` writes `quant.artifact_recall@1` into the eval-quantized
receipt's `metrics` (`scripts/csd-quantize.py:270,271,275`,
`scripts/csd-benchmark.py:1013`). Only `token.*` (§12.2) and `beir.*` (§12.7) remain
unwritten by any production receipt; check §15 before assuming an unmarked name below is on
disk for a given battery.

| v2 canonical field | maps from (v1, if renamed) | section |
|---|---|---|
| `repr.effective_rank_entropy` | `repr.effective_rank` | [§12.1](#121-repreffective_rank_entropy-repreffective_rank_entropy_ratio) |
| `repr.effective_rank_entropy_ratio` | `repr.effective_rank_ratio` | [§12.1](#121-repreffective_rank_entropy-repreffective_rank_entropy_ratio) |
| `token.pooled_pr_rank`, `.global_pr_rank`, `.pooled_entropy_rank`, `.global_entropy_rank` | `token_aware.final_block_rank.*` | [§12.2](#122-tokenpooled_pr_rank--tokenglobal_pr_rank--tokenpooled_entropy_rank--tokenglobal_entropy_rank) |
| `repr.anisotropy` | (same name) | [§12.3](#123-repranisotropy), [§16](#16-the-anisotropy-naming-caveat) |
| `repr.alignment`, `repr.uniformity` | (same names) | [§12.4](#124-repralignment--repruniformity) |
| `repr.emb_std_anchor` | `held_out.emb_std` (train, anchor side) | [§12.5](#125-repremb_std_anchor) |
| `rank.recall@k`, `rank.mrr`, `rank.ndcg@10` (closed pool) | (same names) | [§12.6](#126-rankrecallk--rankmrr--rankndcg10-closed-pool) |
| `beir.ndcg@10`, `beir.recall@k`, `beir.mrr` | BEIR-FiQA `recall@k`/`mrr` (§7.2); `ndcg@10` unimplemented | [§12.7](#127-beirndcg10--beirrecallk--beirmrr) |
| `quant.plan_recall@1`, `.artifact_recall@1`, `.drop_recall@1`, `.compression_ratio` | `quantized_metric`, `rank.recall@1` (eval-quantized), `drop`, `compression_ratio` | [§12.8](#128-quantplan_recall1--quantartifact_recall1--quantdrop_recall1--quantcompression_ratio) |
| `quant.plan_probe_top1`, `.artifact_probe_top1`, `.drop_probe_top1` | (visual; never `quant.*_recall@1` — the quantity is EuroSAT probe top-1) | [§12.8.1](#1281-visual-quantplan_probe_top1--quantartifact_probe_top1--quantdrop_probe_top1) |
| `gate.beats_untrained_train`, `gate.beats_untrained_eval` | `beats_untrained` / `gates.beats_untrained` | [§12.9](#129-gatebeats_untrained_train--gatebeats_untrained_eval) |

See [§15](#15-v1---v2-deprecation-map) for the full v1 -> v2 deprecation map and
[§14](#14-schema-stamp-and-refuse-predicate) for the `metrics_schema` stamp and refuse
predicate every comparison across these names must pass.

## 21. Visual probe metrics (`kind: eval` on the I-JEPA region)

These names are written by `scripts/csd-benchmark.py` `benchmark_visual_region` for
`--regions visual`. They are **not** closed-pool `rank.*` scores. The battery is the
same linear probe training used (`src/cogsyndelta/regions/vl_pretrain.py` `_linear_probe`
on frozen `IJEPA.encode` features — the EMA target encoder).

| field | battery | formula | file |
|---|---|---|---|
| `probe.top1` / `probe.top5` | EuroSAT official test (probe role `primary`) | linear-probe accuracy on frozen target-encoder latents | `scripts/csd-benchmark.py` `benchmark_visual_region`; probe fit `vl_pretrain.py` `_linear_probe` |
| `transfer.top1` / `transfer.top5` | Fashion-MNIST t10k (probe role `transfer`) | same probe, different labelled set | same |
| `repr.rep_std` | mixed I-JEPA train batch | std of target-encoder latents | `vl_pretrain.py` `_rep_std_on_mixed_batch` |
| `gates.beats_untrained_eval` | same EuroSAT split | `probe.top1 > train_receipt.untrained_baseline.top1` | `benchmark_visual_region` |
| `gates.not_collapsed` | same mixed batch | `rep_std / untrained.rep_std >= 0.1` | same predicate as the train receipt |

Do not compare `probe.top1` to `rank.recall@1`. Visual quantize writes
`quant.plan_probe_top1` / `quant.drop_probe_top1` / `quant.artifact_probe_top1`
(§12.8.1), not `quant.*_recall@1`.

### 21.1 Deployed module is the EMA target encoder

Visual quantize and eval-quantized pack **only** `target_encoder.*`
(`DeployedVisualEncoder` in `scripts/csd-benchmark.py`). `IJEPA.encode` is
`target_encoder.embed` (`src/cogsyndelta/model/vl_jepa.py:581-587`). The online
context encoder and the predictor are training-only: they have zero probe
sensitivity, so a full-IJEPA plan would drive them to the floor for free and
`fp32_reference_bytes` / `provenance.parameters` would describe an artifact
nobody deploys. The packed file is still sha-bound to the full training
checkpoint. `--quantized` eval rebuilds the encoder-only wrapper; it does not
`load_state_dict` onto a full IJEPA.
