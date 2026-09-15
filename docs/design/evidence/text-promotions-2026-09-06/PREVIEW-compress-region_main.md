---
language: en
library_name: cogsyndelta
license: cc-by-sa-4.0
pipeline_tag: feature-extraction
tags:
- cogsyndelta
- region:compress
- csd-ptq-v1
model-index:
- name: cogsyndelta-region-compress
  results:
  - task:
      type: feature-extraction
    dataset:
      name: cogsyndelta-compress-holdout
      type: cogsyndelta-compress-holdout
    metrics:
    - type: eff.capability_per_mb
      value: 0.012129887603013199
      name: eff.capability_per_mb
      verified: false
    - type: eff.capability_per_param
      value: 0.048519550412052796
      name: eff.capability_per_param
      verified: false
    - type: eff.latency_p50_ms
      value: 3.6059620033483952
      name: eff.latency_p50_ms
      verified: false
    - type: eff.latency_p95_ms
      value: 3.6270570126362145
      name: eff.latency_p95_ms
      verified: false
    - type: eff.latency_p99_ms
      value: 3.6293920129537582
      name: eff.latency_p99_ms
      verified: false
    - type: eff.parameters
      value: 16021248.0
      name: eff.parameters
      verified: false
    - type: eff.peak_vram_mb
      value: 693.832704
      name: eff.peak_vram_mb
      verified: false
    - type: eff.stored_mb
      value: 64.084992
      name: eff.stored_mb
      verified: false
    - type: eff.throughput_per_s
      value: 277.26443808090903
      name: eff.throughput_per_s
      verified: false
    - type: rank.candidates
      value: 512.0
      name: rank.candidates
      verified: false
    - type: rank.mrr
      value: 0.8330286145210266
      name: rank.mrr
      verified: false
    - type: rank.ndcg@10
      value: 0.8527729511260986
      name: rank.ndcg@10
      verified: false
    - type: rank.recall@1
      value: 0.77734375
      name: rank.recall@1
      verified: false
    - type: rank.recall@10
      value: 0.923828125
      name: rank.recall@10
      verified: false
    - type: rank.recall@5
      value: 0.90234375
      name: rank.recall@5
      verified: false
    - type: repr.alignment
      value: 0.8236081600189209
      name: repr.alignment
      verified: false
    - type: repr.anisotropy
      value: 0.0035395718941428167
      name: repr.anisotropy
      verified: false
    - type: repr.dimensions
      value: 256.0
      name: repr.dimensions
      verified: false
    - type: repr.effective_rank
      value: 174.6697540283203
      name: repr.effective_rank
      verified: false
    - type: repr.effective_rank_ratio
      value: 0.6823037266731262
      name: repr.effective_rank_ratio
      verified: false
    - type: repr.uniformity
      value: -3.8742072582244873
      name: repr.uniformity
      verified: false
---

# CogSynDelta -- compress

`cc-by-sa-4.0` · release `v0.1.0` · code revision `a7694090903664bc256b4b96d998b37cacd316cf` · metrics schema `csd-metrics/v1 (not recorded)` · repo `tzervas/cogsyndelta-region-compress`

One faculty region of the composed CogSynDelta mind -- an encoder that maps
any input to the shared latent stream, **not a chat model and not usable on
its own for open-ended generation**. This is the **promoted release** for the
`compress` region: the harness cell selected from the training matrix as the one
this region ships under `main.pt` / the region's Hub repo default revision.

**Parameters:** 16.021 M. **Role:** Neighbors stay neighbors in a short latent.

## How this was chosen

main_rule best_primary_metric_all_gates_pass on train:$.held_out.recall@1, applied within the same code pin (7bc2699), the same corpus fingerprint (65a993819) and the production batch (1280), among the compress cells whose train, test, quantize and test-quant stages all passed. The seed axis resamples the held-out split (untrained baselines 0.0371 at seed 0 vs 0.0410 at seed 1); seed 0 is the canonical split and is also the maximum: compress-b1280-s0 (0.7754) over seed 1 (0.7715). Batch-512 cells (0.7402, 0.7129) are below on both seeds. Receipts are csd-metrics/v1. Promoted variant: compress-b1280-s0-7bc2699-20260904-c65a99381.

## Key features

- **Faculty, not a general model.** Triggered by: Sentence pair or embedding to reconstruct.
- **Every metric is reported beside its untrained baseline** -- see the Evaluation
  results tables below; a number with no baseline column next to it is not on this
  card.
- **The quantization ratio is a storage ratio, not a speed claim.** See Sizes.
- **Anisotropy is a representation-geometry diagnostic, not a quality score** -- read
  it only together with the ranking metrics in the same table.
- **Every printed metric carries a methodology footnote** into
  `docs/design/METRICS-METHODOLOGY.md` -- formula, battery, pooling and `file:line`.
- **This is the promoted release**, not a harness sweep cell -- see "How this was
  chosen" above for the selection criterion against the other cells in the matrix.

## Model overview

| field | value |
|---|---|
| type | latent_vae |
| parameters | 16.021 M |
| stream_dim | 512 |
| hidden_dim | 1024 |
| modality | any |
| precision | fp32, csd-ptq-v1 (packed, sub-byte) |
| release | `v0.1.0` |

## Evaluation results

this variant reaches 1.09 of the bag-of-words ceiling on 512 items (`csd-lexical/v1`).

### Training held-out battery

| metric | this variant | untrained baseline |
|---|---|---|
| `emb_std`[^1] | **0.0622036** | 0.0141413 |
| `mrr`[^2] | **0.832052** | 0.0534239 |
| `n_pairs`[^3] | 512 | 512 |
| `recall@1`[^4] | **0.775391** | 0.0371094 |
| `recall@10`[^5] | **0.923828** | 0.0683594 |

### Gates

| metric | this variant | untrained baseline |
|---|---|---|
| `beats_untrained_eval`[^6] ^v1^ | yes | _(n/a)_ |
| `not_anisotropic`[^7] | yes | _(n/a)_ |
| `uses_its_dimensions`[^8] | yes | _(n/a)_ |

### Retrieval (fp32)

| metric | this variant | untrained baseline | lexical baseline (TF-IDF)[^35] |
|---|---|---|---|
| `rank.candidates`[^9] | 512 | _(n/a)_ | _(n/a)_ |
| `rank.mrr`[^2] | 0.833029 | _(n/a)_ | 0.779907 |
| `rank.ndcg@10`[^10] | 0.852773 | _(n/a)_ | 0.804967 |
| `rank.recall@1`[^4] | 0.777344 | _(n/a)_ | 0.712891 |
| `rank.recall@10`[^5] | 0.923828 | _(n/a)_ | 0.892578 |
| `rank.recall@5`[^11] | 0.902344 | _(n/a)_ | 0.855469 |

### Efficiency (fp32)

| metric | this variant | untrained baseline |
|---|---|---|
| `eff.capability_per_mb`[^12] | 0.0121299 | _(n/a)_ |
| `eff.capability_per_param`[^13] | 0.0485196 | _(n/a)_ |
| `eff.latency_p50_ms`[^14] | 3.60596 | _(n/a)_ |
| `eff.latency_p95_ms`[^15] | 3.62706 | _(n/a)_ |
| `eff.latency_p99_ms`[^16] | 3.62939 | _(n/a)_ |
| `eff.parameters`[^17] | 1.60212e+07 | _(n/a)_ |
| `eff.peak_vram_mb`[^18] | 693.833 | _(n/a)_ |
| `eff.stored_mb`[^19] | 64.085 | _(n/a)_ |
| `eff.throughput_per_s`[^20] | 277.264 | _(n/a)_ |

### Representation (fp32)

| metric | this variant | untrained baseline |
|---|---|---|
| `repr.alignment`[^21] | 0.823608 | _(n/a)_ |
| `repr.anisotropy`[^22] | 0.00353957 | _(n/a)_ |
| `repr.dimensions`[^23] | 256 | _(n/a)_ |
| `repr.effective_rank_entropy`[^24] ^v1^ | 174.67 | _(n/a)_ |
| `repr.effective_rank_entropy_ratio`[^25] ^v1^ | 0.682304 | _(n/a)_ |
| `repr.uniformity`[^26] | -3.87421 | _(n/a)_ |

### Retrieval (quantized artifact)

| metric | this variant | untrained baseline | lexical baseline (TF-IDF)[^35] |
|---|---|---|---|
| `rank.candidates`[^9] | 512 | _(n/a)_ | _(n/a)_ |
| `rank.mrr`[^2] | 0.824454 | _(n/a)_ | 0.779907 |
| `rank.ndcg@10`[^10] | 0.847172 | _(n/a)_ | 0.804967 |
| `rank.recall@1`[^4] | 0.769531 | _(n/a)_ | 0.712891 |
| `rank.recall@10`[^5] | 0.927734 | _(n/a)_ | 0.892578 |
| `rank.recall@5`[^11] | 0.896484 | _(n/a)_ | 0.855469 |

### Efficiency (quantized artifact)

| metric | this variant | untrained baseline |
|---|---|---|
| `eff.capability_per_mb`[^12] | 0.0678378 | _(n/a)_ |
| `eff.capability_per_param`[^13] | 0.0480319 | _(n/a)_ |
| `eff.latency_p50_ms`[^14] | 3.60142 | _(n/a)_ |
| `eff.latency_p95_ms`[^15] | 3.61928 | _(n/a)_ |
| `eff.latency_p99_ms`[^16] | 3.62077 | _(n/a)_ |
| `eff.parameters`[^17] | 1.60212e+07 | _(n/a)_ |
| `eff.peak_vram_mb`[^18] | 498.653 | _(n/a)_ |
| `eff.stored_mb`[^19] | 11.3437 | _(n/a)_ |
| `eff.throughput_per_s`[^20] | 277.676 | _(n/a)_ |

### Representation (quantized artifact)

| metric | this variant | untrained baseline |
|---|---|---|
| `repr.alignment`[^21] | 0.813118 | _(n/a)_ |
| `repr.anisotropy`[^22] | 0.0310283 | _(n/a)_ |
| `repr.dimensions`[^23] | 256 | _(n/a)_ |
| `repr.effective_rank_entropy`[^24] ^v1^ | 178.344 | _(n/a)_ |
| `repr.effective_rank_entropy_ratio`[^25] ^v1^ | 0.696655 | _(n/a)_ |
| `repr.uniformity`[^26] | -3.76556 | _(n/a)_ |

### Quantization (quant_plan battery)

| metric | this variant | untrained baseline |
|---|---|---|
| `fp32_metric_recomputed`[^27] | 0.775391 | _(n/a)_ |
| `quant.plan_recall@1`[^28] | 0.767578 | _(n/a)_ |
| `quant.drop_recall@1`[^29] | 0.0078125 | _(n/a)_ |
| `tolerance`[^30] | 0.01 | _(n/a)_ |
| `within_budget`[^31] | yes | _(n/a)_ |
| `quant.compression_ratio`[^32] | 5.6494 | _(n/a)_ |
| `fp32_bytes`[^33] | 64084992 | _(n/a)_ |
| `stored_bytes`[^34] | 11343688 | _(n/a)_ |

^v1^ v1 receipt; names mapped to csd-metrics/v2 (see `docs/design/METRICS-METHODOLOGY.md` §15, the v1 -> v2 deprecation map).

[^1]: per-feature embedding std, averaged over features, anchor side only (the collapse signal) -- battery_id=`train_holdout`, pooling=`anchor`, `src/cogsyndelta/regions/pretrain.py`.
[^2]: mean reciprocal rank of the matched positive over the closed held-out pool -- battery_id=`train_holdout`, pooling=`matched`, `src/cogsyndelta/eval/metrics.py`.
[^3]: size of the closed held-out pool this row's numbers were computed over -- battery_id=`train_holdout`, pooling=`matched`, `src/cogsyndelta/regions/pretrain.py`.
[^4]: recall@k (k=1): fraction of queries whose matched positive is the top-scored candidate in the closed held-out pool -- battery_id=`train_holdout`, pooling=`matched`, `src/cogsyndelta/eval/metrics.py`.
[^5]: recall@k (k=10): fraction of queries whose matched positive is in the top-10 of the closed held-out pool -- battery_id=`train_holdout`, pooling=`matched`, `src/cogsyndelta/eval/metrics.py`.
[^6]: rank.recall@1 (eval battery) > the training receipt's untrained_baseline recall@1, unmargined -- a different, simpler predicate than the training receipt's own beats_untrained_train gate, which is why g7 gives the two separate names instead of sharing 'beats_untrained' across receipt kinds -- battery_id=`eval_holdout`, pooling=`matched`, `scripts/csd-benchmark.py`.
[^7]: LEGACY: repr.anisotropy < 0.9, on a receipt written before this was DEMOTED from a gating admission test to a recorded value only (g7 §3.2 -- no bound was ever backed by a study; see repr.anisotropy's own entry for the recorded number). A receipt written after the demotion no longer prints this key. -- battery_id=`eval_holdout`, pooling=`pooled_both`, `scripts/csd-benchmark.py`.
[^8]: repr.effective_rank_entropy_ratio > 0.05 -- the 0.05 floor is unchanged; only the metric name changed (g7 §3.2: was repr.effective_rank_ratio, renamed to disambiguate from the participation-ratio rank ratio a training receipt's token_aware.final_block_rank reports, METRICS-METHODOLOGY.md §9) -- battery_id=`eval_holdout`, pooling=`pooled_both`, `scripts/csd-benchmark.py`.
[^9]: size of the closed pool this eval-battery pass ranked against -- the eval receipt's own count, independent of the training receipt's held_out.n_pairs -- battery_id=`eval_holdout`, pooling=`matched`, `src/cogsyndelta/eval/benchmark.py`.
[^10]: normalised discounted cumulative gain at 10, single relevant item per query -- battery_id=`eval_holdout`, pooling=`matched`, `src/cogsyndelta/eval/benchmark.py`.
[^11]: recall@k (k=5): fraction of queries whose matched positive is in the top-5 of the closed eval pool -- battery_id=`eval_holdout`, pooling=`matched`, `src/cogsyndelta/eval/benchmark.py`.
[^12]: rank.recall@1 / max(1e-9, stored_bytes / 1e6) -- capability per MB actually shipped; the more honest sibling of capability_per_param, since parameters are not what ships -- battery_id=`eval_holdout`, pooling=_(n/a)_, `src/cogsyndelta/eval/benchmark.py`.
[^13]: rank.recall@1 / max(1e-9, parameters / 1e6) -- the thesis metric: capability per million parameters, not capability per byte actually shipped (see capability_per_mb for that) -- battery_id=`eval_holdout`, pooling=_(n/a)_, `src/cogsyndelta/eval/benchmark.py`.
[^14]: median wall-clock latency per encode call, from profile_latency() -- battery_id=`eval_holdout`, pooling=_(n/a)_, `src/cogsyndelta/eval/benchmark.py`.
[^15]: 95th-percentile wall-clock latency per encode call, from profile_latency() -- battery_id=`eval_holdout`, pooling=_(n/a)_, `src/cogsyndelta/eval/benchmark.py`.
[^16]: 99th-percentile wall-clock latency per encode call, from profile_latency() -- battery_id=`eval_holdout`, pooling=_(n/a)_, `src/cogsyndelta/eval/benchmark.py`.
[^17]: parameter count, as passed into benchmark_embeddings() -- the same count a training receipt's top-level `parameters` field reports, re-stated alongside the eval battery -- battery_id=`eval_holdout`, pooling=_(n/a)_, `src/cogsyndelta/eval/benchmark.py`.
[^18]: torch.cuda.max_memory_allocated() / 1e6 over the profiled encode calls; 0.0 on a CPU-only run (this project's tests and CI run CPU-only -- a 0.0 here is the profiler never having seen a CUDA device, not a measured zero) -- battery_id=`eval_holdout`, pooling=_(n/a)_, `src/cogsyndelta/eval/benchmark.py`.
[^19]: stored_bytes / 1e6 -- what the model actually occupies AT THIS EVAL PASS (post-quantization for an eval-quantized receipt, fp32 for a plain eval receipt); the disk-footprint half of quant.compression_ratio, not a separate measurement -- battery_id=`eval_holdout`, pooling=_(n/a)_, `src/cogsyndelta/eval/benchmark.py`.
[^20]: encode calls per wall-clock second, from profile_latency() -- inverse of mean latency over the profiled runs, not of any single percentile above -- battery_id=`eval_holdout`, pooling=_(n/a)_, `src/cogsyndelta/eval/benchmark.py`.
[^21]: mean squared distance between MATCHED pairs (Wang & Isola); read only together with uniformity, never alone -- battery_id=`eval_holdout`, pooling=`matched`, `src/cogsyndelta/eval/benchmark.py`.
[^22]: mean cosine similarity between random (off-diagonal) pairs, anchors+positives pooled -- a representation-geometry diagnostic, NOT a quality score -- battery_id=`eval_holdout`, pooling=`pooled_both`, `src/cogsyndelta/eval/benchmark.py`.
[^23]: raw embedding width -- battery_id=`eval_holdout`, pooling=`pooled_both`, `src/cogsyndelta/eval/benchmark.py`.
[^24]: Shannon entropy of the normalised singular-value spectrum, exponentiated -- the ENTROPY definition, not the participation-ratio one training receipts report under token_aware.final_block_rank (see METRICS-METHODOLOGY.md §9). Renamed from effective_rank to name which of this project's three 'effective rank' definitions it is. -- battery_id=`eval_holdout`, pooling=`pooled_both`, `src/cogsyndelta/eval/benchmark.py`.
[^25]: effective_rank / dimensions -- how much of the available space is actually used. Renamed from effective_rank_ratio to name which of this project's three 'effective rank' definitions it is (METRICS-METHODOLOGY.md §9: the entropy one, never the participation-ratio one). -- battery_id=`eval_holdout`, pooling=`pooled_both`, `src/cogsyndelta/eval/benchmark.py`.
[^26]: log mean Gaussian potential over all pairs (Wang & Isola); read only together with alignment, never alone -- battery_id=`eval_holdout`, pooling=`pooled_both`, `src/cogsyndelta/eval/benchmark.py`.
[^27]: recall@1 measured fresh on the loaded fp32 checkpoint -- the training held-out battery, NOT the eval battery's rank.recall@1 (see METRICS-METHODOLOGY.md §4) -- battery_id=`train_holdout`, pooling=`matched`, `scripts/csd-quantize.py`.
[^28]: recall@1 measured on the IN-MEMORY dequantized plan, before the packed artifact is ever written to disk -- a claim about the plan, not about the published bytes (see METRICS-METHODOLOGY.md §4). Compare against quant.artifact_recall@1 ONLY as the plan-vs-artifact sameness guard on the same checkpoint sha/holdout (g7 §3.3's special case) -- never against rank.* or repr.* from the eval battery. -- battery_id=`quant_plan`, pooling=`matched`, `scripts/csd-quantize.py`.
[^29]: fp32_metric_recomputed - quant.plan_recall@1, one named metric on one named battery (g7 §3.1) -- battery_id=`quant_plan`, pooling=`matched`, `scripts/csd-quantize.py`.
[^30]: largest acceptable absolute drop in the task metric -- a configured input, not a measurement -- battery_id=`quant_plan`, pooling=_(n/a)_, `scripts/csd-quantize.py`.
[^31]: quant.drop_recall@1 <= tolerance -- battery_id=`quant_plan`, pooling=`matched`, `scripts/csd-quantize.py`.
[^32]: fp32_bytes / stored_bytes -- a PAYLOAD/STORAGE ratio, NOT a speed or throughput claim (renamed from compression_ratio, g7 §3.1) -- battery_id=`quant_plan`, pooling=_(n/a)_, `src/cogsyndelta/quant/ptq.py`.
[^33]: sum(parameter.numel() * 4) -- weights only, never optimizer or RNG state -- battery_id=`quant_plan`, pooling=_(n/a)_, `src/cogsyndelta/quant/ptq.py`.
[^34]: packed codes + per-channel scale/zero-point for quantized tensors, plus 4 bytes/element for fp32-kept tensors -- battery_id=`quant_plan`, pooling=_(n/a)_, `src/cogsyndelta/quant/ptq.py`.
[^35]: TF-IDF cosine / BM25 (csd-lexical/v1: diagnosis word regex, log-tf * smoothed idf cosine, BM25 k1=1.5 b=0.75, seed-0 1e-9 tie-break) over the identical closed holdout; split_sha256 must equal split.sha256 (G26) -- battery_id=`eval_holdout`, pooling=`matched`, `src/cogsyndelta/eval/lexical.py`.

- **Metrics schema:** `csd-metrics/v1 (not recorded)`

## Sizes

| quantity | value | label |
|---|---|---|
| parameters | 16.021 M | measured, `parameters` |
| weights on disk, fp32 | 64.085 MB | measured, `fp32_bytes` |
| weights on disk, csd-ptq-v1 | 11.344 MB | measured, `stored_bytes` |
| compression ratio (storage, not speed) | 5.65x | measured, `quant.compression_ratio` |
| bit-width histogram | 3-bit: 16, 6-bit: 1 | measured, `width_histogram` |
| eval peak VRAM, fp32 | 693.833 MB | MEASURED, eval peak, eval batch 512, max_len 96 |
| eval peak VRAM, quantized | 498.653 MB | MEASURED, eval peak, eval batch 512, max_len 96 |
| training peak VRAM | 14611.000 MiB | MEASURED, training peak, batch 1280, max_len 96, cell compress-b1280-s1-7bc2699-20260904 |

## How to use

```bash
git clone https://git.vectorweight.com/tzervas/CogSynDelta.git
cd CogSynDelta
git checkout a7694090903664bc256b4b96d998b37cacd316cf
uv sync --group dev
```

<!-- exec -->
```python
import torch
from cogsyndelta.regions.text_encoder import TextEncoder  # or the region's own loader

state = torch.load("final.pt", map_location="cpu", weights_only=True)
# verify: sha256(final.pt) == "e456293ffe39a719f22b57c521c0f27d3e0e1ee127b84e25300922c24da99f42"
```

<!-- exec -->
```python
from cogsyndelta.quant.ptq import load_packed_artifact, unpack_state_dict

packed = load_packed_artifact("final.ptq.pt")
state_dict = unpack_state_dict(packed)
```

Reproduce the numbers on this card:

```bash
python3 scripts/csd-benchmark.py --region compress --checkpoint final.pt
```


## Limitations and out-of-scope use

PoC-scale, single-region checkpoint. `visual_pairs: NOT MEASURED` (no cross-faculty
pairing has been run against this checkpoint). Not evaluated on any corpus outside its
own held-out pool; do not treat the numbers above as generalising to a different
distribution. "Promoted" describes selection among this region's own harness cells,
not a claim of readiness for a downstream product.

## Licence

`cc-by-sa-4.0` -- SNLI (+ government + fiction) repaired corpus is share-alike; no NC-tagged input identified. See `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`, "Decision 2026-09-02".

## Provenance

- **Corpus fingerprint:** `65a9938192b5040beda5e92eb4078dbc` (scheme `csd-corpus-fp/v2`)
- **Seed:** `0`
- **Code revision:** `a7694090903664bc256b4b96d998b37cacd316cf`
- **Checkpoint sha256:** `e456293ffe39a719f22b57c521c0f27d3e0e1ee127b84e25300922c24da99f42`
- **Metrics schema:** `csd-metrics/v1 (not recorded)`
- **Files:**
  - `final.pt`: sha256 `e456293ffe39a719f22b57c521c0f27d3e0e1ee127b84e25300922c24da99f42`

Published by `scripts/csd-publish-checkpoint.py` / `cogsyndelta.cards`. Repo is
private.

## Citation

```bibtex
@misc{cogsyndelta-compress,
  title = {CogSynDelta -- compress region},
  author = {Zervas, Tyler},
  year = {2026},
  note = {release v0.1.0, code revision a7694090903664bc256b4b96d998b37cacd316cf}
}
```

Contact: see the CogSynDelta repository.
