# Metrics Methodology

**Status:** reference, derived from shipped code. Every formula and `file:line` anchor below
was read from the source on the date in the git log of this file, not written from memory of
what a metric with this name "usually" means. `tests/test_metrics_methodology.py` re-reads
every anchor and fails the build if the cited file no longer has that many lines -- a cheap
drift guard, not a proof the prose still matches, so re-read the anchor before trusting an old
copy of this file.

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
12. [Field index](#field-index)

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
  (`src/cogsyndelta/eval/metrics.py:655-672`), but no production call site invokes it (see
  §11.5) -- the receipt field and the shared function compute the same thing without one
  calling the other.
- **(d)** `held_out.emb_std` / `untrained_baseline.emb_std`: the **anchor side only** (`a` in
  `evaluate()`, `src/cogsyndelta/regions/pretrain.py:534,561`) of the held-out split.
  `graded_held_out.emb_std`: the **left side only** (`a` in `evaluate_graded()`,
  `src/cogsyndelta/regions/pretrain.py:600,607,612`) of the graded set.
- **(e)** Comparable only across receipts computed on the same side, of the same split, of the
  same region -- and note the pool difference from §3's `repr.*` family below.
- **(f)** **This is anchor-side-only, not anchor+positive.** §3's representation family
  (`repr.anisotropy`, `repr.effective_rank`, etc.) is computed over `torch.cat([anchors,
  positives])` -- both sides pooled together
  (`src/cogsyndelta/eval/benchmark.py:392-394`). A `held_out.emb_std` and a `repr.*` number
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
  `src/cogsyndelta/eval/benchmark.py:22-27`) but easy to misread as an efficiency claim; for a
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
  `pr_effective_rank()` (`src/cogsyndelta/eval/benchmark.py:212-254`) for the `pr_*` fields and
  `effective_rank()` (`src/cogsyndelta/eval/benchmark.py:138-155`) for the `*_entropy_rank`
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
(`src/cogsyndelta/eval/benchmark.py:334-401`). This is a **different, wider** battery from §2:
ranking metrics MTEB-family retrieval papers report, efficiency (parameters, size, latency,
throughput), and representation-health diagnostics that a ranking score alone cannot surface.
Written as `Receipt(kind="eval" | "eval-quantized", ...)`
(`src/cogsyndelta/pipeline/receipt.py:58-89`), `metrics` flattened with a family prefix
(`rank.`, `eff.`, `repr.`) by `BenchmarkResult.flat()`
(`src/cogsyndelta/eval/benchmark.py:322-331`).

**Evaluation set, every field in this section:** the same held-out split §2 uses, rebuilt from
the training receipt's own recorded config and cross-checked by corpus fingerprint
(`_region_eval_context()`, `scripts/csd-benchmark.py:163-228`) -- **never** re-globbed. `kind:
"eval"` scores the fp32 checkpoint; `kind: "eval-quantized"` scores the actual packed `.ptq.pt`
artifact loaded back off disk and unpacked to fp32 (`scripts/csd-benchmark.py:388-541`), not
the in-memory quantization plan (§4). `_run_battery()`
(`scripts/csd-benchmark.py:231-252`) encodes the **whole** holdout as one closed candidate
pool -- identical in shape to §2's `scores = a @ p.T`, `relevant = arange(...)` construction,
which is why `rank.recall@1` in this battery is numerically identical to `held_out.recall@1`
in §2 for the same checkpoint (confirmed against a real receipt pair in §10).

### 3.1 `rank.recall@1`, `rank.recall@5`, `rank.recall@10`

Same `recall_at_k()` formula and closed-pool semantics as §2.1
(`src/cogsyndelta/eval/benchmark.py:360-363`), now also at `k=5`. Same caveats.

### 3.2 `rank.mrr`

Same `mean_reciprocal_rank()` as §2.2 (`src/cogsyndelta/eval/benchmark.py:364`).

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
`src/cogsyndelta/eval/benchmark.py:368`), i.e. 512 at this project's default. Not a metric,
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
  (§5) and is the more honest efficiency figure per `src/cogsyndelta/eval/benchmark.py:22-27`
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
  (`src/cogsyndelta/eval/benchmark.py:309`) -- absence of GPU memory pressure, not a
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

- **(c)** `src/cogsyndelta/eval/benchmark.py:82-103`.
- **(d)** **Both anchors and positives pooled together**: `both = torch.cat([anchors,
  positives])` (`src/cogsyndelta/eval/benchmark.py:392-394`), then subsampled to at most 2048
  rows (seed 0) if the pool exceeds that. At this project's 512-pair holdout, `both` has 1024
  rows -- under the 2048 cap, so **no subsampling actually occurs at the project's current
  holdout size**, though the code path exists and would engage on a larger one.
- **(e)** Comparable across two receipts only when both used the same holdout size (so the
  same subsampling regime applies or does not) and the same sampling seed (fixed at 0 by
  default, not varied per receipt).
- **(f)** **This is a representation-geometry diagnostic, not a quality score.** Higher is not
  better or worse in the abstract: near 0 means unrelated items sit close to orthogonal (a
  healthy, spread-out space); near 1 means the space has collapsed into a narrow cone. The
  project's own gate treats `>= 0.9` as unhealthy
  (`not_anisotropic: anisotropy < 0.9`, `scripts/csd-benchmark.py:357,521`), not as "anisotropy
  should be minimised" -- a healthy encoder is not at 0.0 either. Do not rank two models by
  "lower anisotropy is better" without also checking `rank.recall@1` moved the direction you
  expect; anisotropy alone cannot tell you retrieval quality (`src/cogsyndelta/eval/
  benchmark.py:15-20`).

### 3.10 `repr.alignment`

- **(a)** `repr.alignment`.
- **(b)** Expected distance between **matched** pairs (Wang & Isola's alignment term). Lower is
  better -- but only jointly with uniformity (§3.11); a collapsed encoder has perfect
  (near-zero) alignment.

  ```python
  a = normalize(anchors.float(), dim=-1); p = normalize(positives.float(), dim=-1)
  alignment = mean( ||a_i - p_i||_2 ** alpha )        # alpha = 2.0 default
  ```

- **(c)** `src/cogsyndelta/eval/benchmark.py:106-115`.
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

- **(c)** `src/cogsyndelta/eval/benchmark.py:118-135`.
- **(d)** `both = cat([anchors, positives])`, same pool as anisotropy (§3.9).
- **(e)** Same rule as §3.9.
- **(f)** Read `repr.alignment` and `repr.uniformity` **together**, never one without the
  other: "either alone is gameable... a random one has excellent uniformity"
  (`src/cogsyndelta/eval/benchmark.py:110-111`).

### 3.12 `repr.effective_rank`, `repr.dimensions`, `repr.effective_rank_ratio`

- **(a)** All three.
- **(b)** `repr.effective_rank` is the **Shannon-entropy** effective rank (see §9 for the
  formula and its participation-ratio counterpart -- **do not conflate the two**).
  `repr.dimensions` is the raw embedding width. `repr.effective_rank_ratio =
  repr.effective_rank / repr.dimensions` -- how much of the available space the encoder
  actually uses.
- **(c)** `effective_rank()`: `src/cogsyndelta/eval/benchmark.py:138-155`. Called with its
  **default** `sample=2048` at `src/cogsyndelta/eval/benchmark.py:397,399` (contrast with
  §2.7, where the same function is called with subsampling disabled).
- **(d)** `both = cat([anchors, positives])`, same pool as anisotropy/uniformity, subsampled to
  2048 rows if larger (not triggered at the project's current 1024-row pool -- see §3.9).
- **(e)** Comparable across receipts at the same embedding dimension directly; across
  different dimensions, compare `effective_rank_ratio` instead of the raw rank.
- **(f)** The project's own gate treats `effective_rank_ratio <= 0.05` as unhealthy
  (`uses_its_dimensions`, `scripts/csd-benchmark.py:358,522`) -- an encoder nominally 256-d but
  effectively 12-d is "paying to store 256"
  (`src/cogsyndelta/eval/benchmark.py:141-143`). **This is the entropy definition, not the
  participation-ratio one §2.7 reports** -- see §9 before comparing a `repr.effective_rank`
  figure against a `token_aware.final_block_rank.pooled_entropy_rank` figure from the SAME
  receipt; they use the same formula but different pools (both-sides-pooled-and-subsampled
  here vs. anchor-only-full-surface there) and are not interchangeable numbers even though
  both are "entropy effective rank."

### 3.13 `gates.beats_untrained`, `gates.not_anisotropic`, `gates.uses_its_dimensions`

- **(a)** All three, under `gates` in an eval / eval-quantized receipt.
- **(b)**

  ```text
  beats_untrained    = rank.recall@1 > train_receipt["untrained_baseline"]["recall@1"]
  not_anisotropic    = repr.anisotropy < 0.9
  uses_its_dimensions = repr.effective_rank_ratio > 0.05
  ```

- **(c)** `scripts/csd-benchmark.py:354-358` (fp32 pass), `scripts/csd-benchmark.py:520-522`
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
(`verify_quantized_measurements()`, `scripts/csd-publish-checkpoint.py:631-692`, using
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
  `verify_quantized_measurements()`, `scripts/csd-publish-checkpoint.py:631-692`.
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
| `a_beats_both_parents` | `memory`'s `recall@1` >= max(`compress`, `retrieve` parents' `recall@1`) AND `memory`'s graded spearman >= `compress`'s | `src/cogsyndelta/eval/beir_fiqa.py:437-466` |
| `b_full_pool_thresholds` | full-pool `recall@10 > 0.20` AND `mrr > 0.10` | `src/cogsyndelta/eval/beir_fiqa.py:469-486` |
| `c_beats_bm25` | trained `recall@10` > BM25 `recall@10`, same pool/qrels | `src/cogsyndelta/eval/beir_fiqa.py:489-502` |
| `d_beats_random_init` | trained `recall@10` > this region's own untrained-encoder `recall@10`, same pool | `src/cogsyndelta/eval/beir_fiqa.py:505-515` |
| `e_retrain_gate` | `token_global_pr_rank >= 2.0 * pooled_pr_rank` (§2.7/§9) AND no more than 1-point regression against either parent | `src/cogsyndelta/eval/beir_fiqa.py:518-579` |

**(f)** `e_retrain_gate`'s rank clause is the one flagged in §2.7(f) as measured
not-discriminating at 50 steps -- see that caveat before treating a `passed: True` here as
proof the token-aware objective helped. `e_retrain_gate` also explicitly does **not** check the
`banking77` damage-detector probe §4.0 of the design doc names
(`src/cogsyndelta/eval/beir_fiqa.py:547-550`) -- recorded as `"not measured; out of scope"`
in the gate's own output, not silently omitted.

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

- **(c)** `effective_rank()`: `src/cogsyndelta/eval/benchmark.py:138-155`.
  `participation_ratio()`: `src/cogsyndelta/eval/benchmark.py:158-209`. `pr_effective_rank()`:
  `src/cogsyndelta/eval/benchmark.py:212-254`.
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
   quietly collapsed, per `src/cogsyndelta/eval/benchmark.py:15-20`.
3. **A licence tier follows the corpus, not the metric.** `licence_tier()`
   (`scripts/csd-publish-checkpoint.py:263-278`) is derived entirely from
   `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`'s per-region table (section "Decision
   2026-09-02") -- a region's training data provenance -- and has no dependency on anything in
   this document. A region whose numbers on this page look identical to another's can still
   carry a stricter licence, and a merged region inherits its most restrictive parent's tier
   regardless of either parent's measured metrics
   (`scripts/csd-publish-checkpoint.py:200-203`).
4. **The untrained baseline is not "zero."** See §1 in full before reading a small
   `recall@1`/`spearman` as a failure -- check it against `chance` and
   `untrained_baseline`/`untrained_graded_baseline` first.
5. **`token_weighted_perplexity()` and `representation_std()` are implemented, tested, and
   exported, but no production training or eval path calls either one today.**
   `token_weighted_perplexity()` (`src/cogsyndelta/eval/metrics.py:75-97`) and
   `representation_std()` (`src/cogsyndelta/eval/metrics.py:655-672`) are exercised only by
   `tests/test_eval_metrics.py` (confirmed by a repo-wide search for their call sites, 2026-09).
   No region currently reports a `perplexity` field despite the module's own opening docstring
   framing the project's efficiency claim around "without decreasing quality and/or increasing
   perplexity" (`src/cogsyndelta/eval/metrics.py:1-8`) -- if a future card or receipt does show
   a `perplexity` field, its formula is `token_weighted_perplexity()`'s
   (§2's `emb_std` sibling), and it should cite this document's entry for it rather than an
   assumed standard definition. `representation_std()`'s formula is used in practice (§2.3),
   just not through this function -- `held_out.emb_std` and `graded_held_out.emb_std` are an
   inline duplicate of it, not a call to it.
6. **`kind: "eval"` and `kind: "eval-quantized"` gates use a different, unmargined
   `beats_untrained` rule than the training receipt's own `beats_untrained`.** See §3.13(f). Do
   not assume the two were decided by the same threshold just because they share a name.
7. **A quant receipt's own `stored_bytes`/`width_histogram` are the plan's claim about itself
   until publish time re-measures them from the file.** See §5.6. Prefer numbers from a
   published card (file-verified) over a bare quant receipt when the two could differ.

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
| `metrics["rank.map"]` | eval, eval-quantized | [§3.4](#34-rankmap) |
| `metrics["rank.precision@10"]` | eval, eval-quantized | [§3.5](#35-rankprecision10) |
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
