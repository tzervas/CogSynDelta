---
language: en
library_name: cogsyndelta
license: mit
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
    - type: rank.recall@1
      value: 0.49
      name: rank.recall@1
      verified: false
    - type: repr.anisotropy
      value: 0.1306
      name: repr.anisotropy
      verified: false
    - type: repr.effective_rank_ratio
      value: 0.459
      name: repr.effective_rank_ratio
      verified: false
---

# CogSynDelta -- compress (harness cell)

`mit` · code revision `(none recorded)` · metrics schema `csd-metrics/v1 (not recorded)`

One faculty region of the composed CogSynDelta mind -- an encoder that maps
text input to the shared latent stream, **not a chat model and not usable on
its own for open-ended generation**. This is one harness cell (one training/eval/quant
run) of the `compress` region, not the promoted release -- see the region's
`region_main` card for "how this was chosen" among the harness's cells.

**Role:** Docstring <-> function, search and generation.

## Key features

- **Faculty, not a general model.** Triggered by: language=python and a docstring is present..
- **Every metric is reported beside its untrained baseline** -- see the Evaluation
  results tables below; a number with no baseline column next to it is not on this
  card.
- **The quantization ratio is a storage ratio, not a speed claim.** See Sizes.
- **Anisotropy is a representation-geometry diagnostic, not a quality score** -- read
  it only together with the ranking metrics in the same table.
- **Every printed metric carries a methodology footnote** into
  `docs/design/METRICS-METHODOLOGY.md` -- formula, battery, pooling and `file:line`.

## Model overview

| field | value |
|---|---|
| type | contrastive_encoder |
| parameters | (not recorded) |
| stream_dim | 512 |
| hidden_dim | 1024 |
| modality | text |
| precision | fp32, csd-ptq-v1 (packed, sub-byte) |

## Evaluation results

### Training held-out battery

| metric | this variant | untrained baseline |
|---|---|---|
| `n_pairs`[^1] | **512** | 512 |
| `recall@1`[^2] | **0.707** | 0.037 |
| `recall@10`[^3] | **0.9199** | 0.068 |

### Gates

| metric | this variant | untrained baseline |
|---|---|---|
| `beats_untrained_eval`[^4] ^v1^ | yes | _(n/a)_ |
| `not_anisotropic`[^5] | yes | _(n/a)_ |
| `uses_its_dimensions`[^6] | yes | _(n/a)_ |

### Retrieval

| metric | this variant | untrained baseline |
|---|---|---|
| `rank.recall@1`[^2] | 0.49 | _(n/a)_ |

### Representation

| metric | this variant | untrained baseline |
|---|---|---|
| `repr.anisotropy`[^7] | 0.1306 | _(n/a)_ |
| `repr.effective_rank_entropy_ratio`[^8] ^v1^ | 0.459 | _(n/a)_ |

### Quantization (quant_plan battery)

| metric | this variant | untrained baseline |
|---|---|---|
| `fp32_metric_recomputed`[^9] | 0.496 | _(n/a)_ |
| `quant.plan_recall@1`[^10] | 0.488 | _(n/a)_ |
| `quant.drop_recall@1`[^11] | 0.0078 | _(n/a)_ |
| `tolerance`[^12] | 0.01 | _(n/a)_ |
| `within_budget`[^13] | yes | _(n/a)_ |
| `quant.compression_ratio`[^14] | 4.28205 | _(n/a)_ |
| `fp32_bytes`[^15] | 37408 | _(n/a)_ |
| `stored_bytes`[^16] | 8736 | _(n/a)_ |

^v1^ v1 receipt; names mapped to csd-metrics/v2 (see `docs/design/METRICS-METHODOLOGY.md` §15, the v1 -> v2 deprecation map).


[^1]: size of the closed held-out pool this row's numbers were computed over -- battery_id=`train_holdout`, pooling=`matched`, `src/cogsyndelta/regions/pretrain.py`.
[^2]: recall@k (k=1): fraction of queries whose matched positive is the top-scored candidate in the closed held-out pool -- battery_id=`train_holdout`, pooling=`matched`, `src/cogsyndelta/eval/metrics.py`.
[^3]: recall@k (k=10): fraction of queries whose matched positive is in the top-10 of the closed held-out pool -- battery_id=`train_holdout`, pooling=`matched`, `src/cogsyndelta/eval/metrics.py`.
[^4]: rank.recall@1 (eval battery) > the training receipt's untrained_baseline recall@1, unmargined -- a different, simpler predicate than the training receipt's own beats_untrained_train gate, which is why g7 gives the two separate names instead of sharing 'beats_untrained' across receipt kinds -- battery_id=`eval_holdout`, pooling=`matched`, `scripts/csd-benchmark.py`.
[^5]: LEGACY: repr.anisotropy < 0.9, on a receipt written before this was DEMOTED from a gating admission test to a recorded value only (g7 §3.2 -- no bound was ever backed by a study; see repr.anisotropy's own entry for the recorded number). A receipt written after the demotion no longer prints this key. -- battery_id=`eval_holdout`, pooling=`pooled_both`, `scripts/csd-benchmark.py`.
[^6]: repr.effective_rank_entropy_ratio > 0.05 -- the 0.05 floor is unchanged; only the metric name changed (g7 §3.2: was repr.effective_rank_ratio, renamed to disambiguate from the participation-ratio rank ratio a training receipt's token_aware.final_block_rank reports, METRICS-METHODOLOGY.md §9) -- battery_id=`eval_holdout`, pooling=`pooled_both`, `scripts/csd-benchmark.py`.
[^7]: mean cosine similarity between random (off-diagonal) pairs, anchors+positives pooled -- a representation-geometry diagnostic, NOT a quality score -- battery_id=`eval_holdout`, pooling=`pooled_both`, `src/cogsyndelta/eval/benchmark.py`.
[^8]: effective_rank / dimensions -- how much of the available space is actually used. Renamed from effective_rank_ratio to name which of this project's three 'effective rank' definitions it is (METRICS-METHODOLOGY.md §9: the entropy one, never the participation-ratio one). -- battery_id=`eval_holdout`, pooling=`pooled_both`, `src/cogsyndelta/eval/benchmark.py`.
[^9]: recall@1 measured fresh on the loaded fp32 checkpoint -- the training held-out battery, NOT the eval battery's rank.recall@1 (see METRICS-METHODOLOGY.md §4) -- battery_id=`train_holdout`, pooling=`matched`, `scripts/csd-quantize.py`.
[^10]: recall@1 measured on the IN-MEMORY dequantized plan, before the packed artifact is ever written to disk -- a claim about the plan, not about the published bytes (see METRICS-METHODOLOGY.md §4). Compare against quant.artifact_recall@1 ONLY as the plan-vs-artifact sameness guard on the same checkpoint sha/holdout (g7 §3.3's special case) -- never against rank.* or repr.* from the eval battery. -- battery_id=`quant_plan`, pooling=`matched`, `scripts/csd-quantize.py`.
[^11]: fp32_metric_recomputed - quant.plan_recall@1, one named metric on one named battery (g7 §3.1) -- battery_id=`quant_plan`, pooling=`matched`, `scripts/csd-quantize.py`.
[^12]: largest acceptable absolute drop in the task metric -- a configured input, not a measurement -- battery_id=`quant_plan`, pooling=_(n/a)_, `scripts/csd-quantize.py`.
[^13]: quant.drop_recall@1 <= tolerance -- battery_id=`quant_plan`, pooling=`matched`, `scripts/csd-quantize.py`.
[^14]: fp32_bytes / stored_bytes -- a PAYLOAD/STORAGE ratio, NOT a speed or throughput claim (renamed from compression_ratio, g7 §3.1) -- battery_id=`quant_plan`, pooling=_(n/a)_, `src/cogsyndelta/quant/ptq.py`.
[^15]: sum(parameter.numel() * 4) -- weights only, never optimizer or RNG state -- battery_id=`quant_plan`, pooling=_(n/a)_, `src/cogsyndelta/quant/ptq.py`.
[^16]: packed codes + per-channel scale/zero-point for quantized tensors, plus 4 bytes/element for fp32-kept tensors -- battery_id=`quant_plan`, pooling=_(n/a)_, `src/cogsyndelta/quant/ptq.py`.

- **Metrics schema:** `csd-metrics/v1 (not recorded)`

## Sizes

| quantity | value | label |
|---|---|---|
| weights on disk, fp32 | 0.037 MB | measured, `fp32_bytes` |
| weights on disk, csd-ptq-v1 | 0.009 MB | measured, `stored_bytes` |
| compression ratio (storage, not speed) | 4.28x | measured |
| bit-width histogram | 3-bit: 1 | measured, `width_histogram` |

## How to use

```bash
git clone https://git.vectorweight.com/tzervas/CogSynDelta.git
cd CogSynDelta
git checkout (none recorded)
uv sync --group dev
```

```python
import torch
from cogsyndelta.regions.text_encoder import TextEncoder  # or the region's own loader

state = torch.load("final.pt", map_location="cpu", weights_only=True)
# verify: sha256(final.pt) == "20f37e64607e9ca73e98a873012f43bafac8ee33fdb00de9d1d5551ee4f4f54d"
```

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

PoC-scale, single-region harness cell. `visual_pairs: NOT MEASURED` (no cross-faculty
pairing has been run against this checkpoint). Not evaluated on any corpus outside its
own held-out pool; do not treat the numbers above as generalising to a different
distribution.

## Licence

`mit` -- no NC or share-alike input in the catalogue. See `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`, "Decision 2026-09-02".

## Provenance

- **Corpus fingerprint:** `5de8a340c37137554824578a0040604d` (scheme `(scheme not recorded)`)
- **Seed:** `(none recorded)`
- **Code revision:** `(none recorded)`
- **Checkpoint sha256:** `20f37e64607e9ca73e98a873012f43bafac8ee33fdb00de9d1d5551ee4f4f54d`
- **Metrics schema:** `csd-metrics/v1 (not recorded)`
- **Files:**
  - `final.pt`: sha256 `20f37e64607e9ca73e98a873012f43bafac8ee33fdb00de9d1d5551ee4f4f54d`

Published by `scripts/csd-publish-checkpoint.py` / `cogsyndelta.cards`. Repo is
private.

## Citation

```bibtex
@misc{cogsyndelta-compress,
  title = {CogSynDelta -- compress region},
  author = {Zervas, Tyler},
  year = {2026},
  note = {code revision (none recorded)}
}
```

Contact: see the CogSynDelta repository.