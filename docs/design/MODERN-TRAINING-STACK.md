# Modernising the CSD training, fine-tuning and quantization stack

Program item **P9.0**. This is the survey the other P9 tasks are gated on: what actually
applies here, measured, so that nothing gets added because it is modern.

Every recommendation below is judged against one criterion, taken from the project's own
thesis: **does it improve capability per parameter, capability per byte, or the cost of
executing and recovering a run?** Novelty is not a reason. Three of the candidates the
program lists are rejected here, and two of the highest-value items are not on the P9 list
at all — they surfaced from measurement.

Measured `2026-09-02T20:00Z`-`21:00Z` on akula-prime, **while a real training run was
executing on the 3090 Ti**. That constrains what could be measured and is called out at
each point. Nothing here wrote to `/akula-data/csd/receipts`.

Reproduce with:

```bash
# CPU + Prometheus only. Safe beside a live run.
uv run --group train python scripts/csd-measure-training-cost.py --json /tmp/measure.json

# Adds a GPU probe. Caps itself at 35% of the card so it cannot OOM a live run.
uv run --group train python scripts/csd-measure-training-cost.py \
    --gpu-probe --probe-fraction 0.35 --json /tmp/measure.json
```

---

## 1. The measured baseline

### 1.1 Step time and throughput

Derived from the real receipts in `/akula-data/csd/receipts`. `elapsed_s` covers the
training loop only — it starts after the corpus split and the untrained baseline eval, and
includes the in-loop evals and checkpoint writes.

| region | receipt | steps | batch | elapsed_s | **ms/step** | steps/s | pairs/s | sequences/s |
|---|---|---|---|---|---|---|---|---|
| code | `code-…T162544Z` | 8000 | 256 | 1050.4 | **131.3** | 7.62 | 1,950 | 3,899 |
| code | `code-…T200323Z` | 6000 | 256 | 792.7 | **132.1** | 7.57 | 1,938 | 3,875 |
| compress | `compress-…T164930Z` | 8000 | 256 | 338.3 | **42.3** | 23.65 | 6,054 | 12,108 |
| retrieve | `retrieve-…T165842Z` | 8000 | 256 | 548.8 | **68.6** | 14.58 | 3,732 | 7,464 |
| vl_latent | `vl_latent-…T174836Z` | 8000 | 256 | 496.7 | **62.1** | 16.11 | 4,123 | — |

Two independent `code` runs agree to 0.6% (131.3 vs 132.1 ms/step). The step time is stable
and is a usable baseline.

**Step time varies 3.1x across text regions on an identical model and batch size.** That is
not noise and it is the thread that unravels the rest of this document: `_tokenize` pads to
the longest item in the batch, so the sequence length the GPU sees is a property of the
corpus, not the config. Measured mean padded width: `code` 96.0 (every batch saturates the
`max_len` ceiling), `retrieve` 57-63, `compress` 32.9.

Wall-clock outside the training loop is **negligible**, which rules out a whole class of
"optimise the data pipeline" work:

| runner summary | wall clock | sum of `elapsed_s` | overhead |
|---|---|---|---|
| `program-run-…T162545Z` (code) | 1057 s | 1050.4 s | 6.6 s (0.6%) |
| `program-run-…T165842Z` (compress+retrieve) | 893 s | 887.1 s | 5.9 s (0.7%) |
| `program-run-…T175652Z` (vl_latent) | 529 s | 496.7 s | 32.3 s (6.1%) |
| `program-run-…T200323Z` (code) | 797 s | 792.7 s | 4.3 s (0.5%) |

Parquet loading, dedup and the contamination check cost ~6 s on a 227k-pair corpus. Leave
them alone.

### 1.2 Where the step actually goes

This is the measurement that reorders the whole P9 list.

`regions/pretrain.py` tokenizes inside the training loop, synchronously, with a Python list
comprehension over single `tok.encode()` calls — 512 texts per step (256 anchors + 256
positives). Measured on real corpus text at batch 256, `max_len` 96:

| corpus sample | serial (current) | `encode_batch` | speedup | mean padded width |
|---|---|---|---|---|
| `code` / codesearchnet-python | **81.48 ms/step** | 14.46 ms | 5.64x | 96.0 |
| `compress` / all-nli pair | **9.30 ms/step** | 1.66 ms | 5.59x | 32.9 |
| `retrieve` / fiqa-pairs | **61.82 ms/step** | 8.93 ms | 6.92x | 63.1 |
| `retrieve` / gooaq | **20.02 ms/step** | 3.40 ms | 5.89x | 57.2 |

Subtracting from the receipt step time gives the decomposition:

| region | total ms/step | tokenizer | GPU + optimizer |
|---|---|---|---|
| **code** | 131.3 | **81.5 (62%)** | 49.8 (38%) |
| **compress** | 42.3 | 9.3 (22%) | 33.0 (78%) |
| **retrieve** | 68.6 | ~21 (31%, derived) | ~48 (69%) |

*The `retrieve` tokenizer figure is derived, not measured directly*: its corpus is a
mixture — 400,000 gooaq + 100,231 natural-questions + 14,131 fiqa pairs — and only the
gooaq and fiqa shards were timed. Weighting by pair count gives ~21 ms/step. Natural
Questions was not timed.

**On the region that costs the most GPU-hours, 62% of every training step is a
single-threaded CPU tokenizer holding a 24 GiB GPU idle.** Every GPU-side optimisation in
the P9 list — bf16, `torch.compile` — is bounded by the remaining 38%.

### 1.3 Peak VRAM, and how idle the card is

Two independent instruments agree, which is why the number can be trusted.

**Instrument 1 — Prometheus** (`akula_gpu_memory_used_mib`, `192.168.1.98:9108`, 15 s step),
queried over each receipt's own wall-clock window. akula-prime runs a KDE desktop on the
same card; the idle floor is a measured **922 MiB**, and it is subtracted.

| run | peak total | median total | **peak training** | % of 23,028 MiB |
|---|---|---|---|---|
| code (8000 steps) | 5,055 | 5,055 | **4,133 MiB** | 17.9% |
| compress + retrieve (one process) | 6,619 | 6,619 | **5,697 MiB** | 24.7% |
| vl_latent | 3,038 | 3,037 | **2,116 MiB** | 9.2% |
| code (6000 steps) | 9,186* | 5,056 | 4,134 MiB | 17.9% |

\* Window-edge artifact: an unrelated 9,182 MiB process occupied the card until 19:50:15Z,
and the window opens at 19:50:10Z. The median, 5,056 MiB, is the real plateau. It is worth
recording anyway as the **largest CSD GPU footprint ever observed on this card — 39.9%**.

**Instrument 2 — direct probe** (`--gpu-probe`, synthetic ids, all positions real, so this
is the `code` worst case). `torch.cuda.max_memory_allocated` is per-process and immune to
the neighbouring run.

| dtype | batch | peak allocated | peak reserved | ms/step (contended) |
|---|---|---|---|---|
| fp32 | 256 | 3,474.3 MiB | 3,736 MiB | 123.94 |
| fp32 | 512 | 6,713.9 MiB | 6,928 MiB | 237.66 |
| fp32 | 1024 | — | — | OOM under the self-imposed cap |
| bf16 | 256 | 2,102.4 MiB | 2,298 MiB | 62.73 |
| bf16 | 512 | 3,958.7 MiB | 4,172 MiB | 120.27 |
| bf16 | 1024 | 7,644.2 MiB | 7,994 MiB | 169.97 † |
| bf16 | 1536 | 11,331.4 MiB | 11,786 MiB | 207.13 † |

† Separate invocation under different contention. Memory rows are comparable across
invocations; timing rows are not.

Fitting the two 256/512 points and testing against the 1024/1536 points:

```
fp32 peak allocated (MiB) ≈  233 + 12.66·B
bf16 peak allocated (MiB) ≈  245 +  7.25·B
```

Predicted bf16 @1024 = 7,669 vs measured 7,644 (−0.3%); @1536 = 11,381 vs 11,331 (−0.4%).
The model holds over a 6x range in batch size.

The intercept is a check on the whole thing: 233-245 MiB against a predicted
61.11 (weights) + 61.11 (grads) + 122.2 (AdamW `exp_avg` + `exp_avg_sq`) = **244.4 MiB**.

And the cross-instrument check: probe reserved at fp32/256 is 3,736 MiB; add the ~400 MiB
CUDA context that `max_memory_allocated` does not count and you get 4,136 MiB against
Prometheus's 4,133 MiB — **0.07% apart**. Both instruments are measuring the same thing.

#### The headline

During a `code` run the 3090 Ti holds 5,055 MiB of 23,028 MiB.
**17,973 MiB — 78.0% of the card — is idle for 17 minutes.**

Solving the memory model for the largest batch that fits, allowing 922 MiB for the desktop,
400 MiB for the CUDA context, ~4% allocator slack and a 1,024 MiB safety margin:

| precision | largest batch that fits | vs today | negatives per anchor |
|---|---|---|---|
| today | 256 | 1x | 255 |
| fp32 | **~1,536** | 6.0x | 1,535 |
| bf16 | **~2,560** (fit to 2,696) | 10.0x | 2,559 |

This is the single biggest lever available, because in symmetric InfoNCE over in-batch
negatives **the negatives are the batch**. The loss lower-bounds the mutual information by
`log B − L`: at B=256 the ceiling is 5.55 nats; at B=2560 it is 7.85 nats. The objective
today cannot express more than 5.55 nats of alignment no matter how long it runs.

### 1.4 Model size: parameters and real stored bytes

All three text regions are the same shape (`dim` 256, `depth` 4, `n_heads` 4, `max_len` 96,
vocab 50,257).

| component | parameters | share |
|---|---|---|
| `embed.weight` (50,257 x 256) | 12,865,792 | **80.3%** |
| 4 x ViTBlock (qkv, proj, mlp, norms) | 3,154,944 | 19.7% |
| final `norm` | 512 | <0.1% |
| **total** | **16,021,248** | |

| artifact | bytes | note |
|---|---|---|
| fp32 weights | 64,084,992 (61.11 MiB) | matches `fp32_bytes` in every quant receipt |
| checkpoint `final.pt` on disk | 192,298,179 (183.4 MiB) | **3.00x** the weight bytes |
| quantized, `code` | 6,543,592 | 9.79x, effective **3.27 bits/param** |
| quantized, `compress` | 6,519,016 | 9.83x, all 17 tensors at 3 bits |
| quantized, `retrieve` | 6,732,008 | 9.52x, one 5-bit + one 8-bit promotion |
| vl_latent checkpoint | 91,681,547 (87.4 MiB) | 22,905,216 params x 4 B = 91,620,864; **1.00x** |

P7.1 is confirmed exactly: the text checkpoints are 3x parameter bytes — weights plus
AdamW's two moments — so a weights-only export drops 183.4 MiB to 61.1 MiB. The visual
checkpoint already stores 1x and needs no such fix.

Quality per unit size, from the eval receipts:

| region | recall@1 | capability/Mparam | capability/MB | p50 latency | eval peak VRAM |
|---|---|---|---|---|---|
| code | 0.9590 | 0.0599 | 0.1466 | 3.31 ms | 693.9 MB |
| compress | 0.4922 | 0.0307 | 0.0755 | 1.73 ms | 481.8 MB |
| retrieve | 0.6836 | 0.0427 | 0.1015 | 1.14 ms | 387.7 MB |

One caution that belongs next to every one of these numbers: quantization here is
*simulated* — `apply_plan` dequantizes to fp32 and runs fp32 matmuls. The 9.8x is a
**storage** claim. The `latency_p50_ms` above is fp32 latency, and no part of this stack
gets faster from quantizing. `capability_per_mb` is honest about this; a summary that pairs
"9.8x compression" with a latency number would not be.

### 1.5 Two things the receipts do not record, and both matter

**(a) The InfoNCE negatives for `code` come from one repository.**

`build_splits` shuffles only when a region declares `extra_sources`:

```python
if len(cfg.extra_sources) > 1 or cfg.extra_sources:
    _random.Random(cfg.seed).shuffle(all_pairs)
```

`code` and `compress` are single-source, so that condition is false and **they train on raw
corpus order**. The training loop then slices contiguous windows,
`train_pairs[lo:lo+batch]`.

CodeSearchNet-Python is ordered by repository. Measured distinct `repo` values in
consecutive 256-row windows over the first 4,096 rows:

```
[2, 1, 3, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 4, 2, 6]     median = 1
windows of 1024: [4, 1, 2, 12]
```

**The median `code` training batch draws all 255 of its negatives from a single
repository.** The file's own comment warns about exactly this for the multi-source case —
"an InfoNCE batch is drawn from ONE source, so its negatives are all same-domain — an easier
task that inflates in-batch accuracy while teaching the model less" — and the guard does not
fire for the regions that need it most.

The same ordering decides the held-out set: `holdout = all_pairs[:512]`, i.e. the first
~2-3 repositories in the corpus. `code`'s headline recall@1 of 0.9590 is measured against a
512-candidate pool drawn from 2-3 projects, not from Python at large. It is not wrong, but
it does not mean what a MTEB-style number means, and `capability_per_param` 0.0599 for
`code` is not comparable to 0.0307 for `compress` on that basis.

This has a direct consequence for the batch-size argument in §1.3: raising the batch to
2,560 on an unshuffled corpus buys 2,559 negatives that are still ~5-12 repositories. The
shuffle must land *first*, or the batch experiment measures the wrong thing.

**(b) The 50,257-row embedding is mostly unused, per region.**

Measured over 200,000 real texts per corpus (28,262 for fiqa, which is all it has):

| corpus | distinct tokens seen | coverage | tokens covering 99.9% of uses |
|---|---|---|---|
| `code` | 27,519 | 54.8% | **20,950** |
| `compress` | 16,278 | 32.4% | **14,041** |
| `retrieve` / fiqa | 23,487 | 46.7% | 22,064 |
| `retrieve` / gooaq | 46,459 | 92.4% | 41,718 |

80.3% of each region's parameters are an embedding table, and `compress` never indexes
two-thirds of it. That is dead weight sitting in the denominator of the project's own
thesis metric. See §2.11.

### 1.6 A data-dependent GPU sync in the shared attention block

`ViTBlock.forward` (`model/vl_jepa.py`, shared by the text encoder) contains:

```python
empty = ~keep.any(dim=1)
if empty.any():          # tensor -> bool  =>  device synchronise
    keep = keep.clone()
    keep[empty, 0] = True
```

`if` on a CUDA tensor calls `.item()` and blocks the host until the GPU drains. That is 4
syncs per forward, 8 per training step (two towers).

Because `_tokenize` right-pads, position 0 is a real token in every non-empty row, so
`keep[:, 0] = True` unconditionally is *identical* — measured max absolute output difference
**0.0**, including a deliberately all-padding row.

A/B/A/B interleaved to defeat drifting contention, batch 256, ragged mask, 20 steps x 3:

```
current (branch + sync)   150.25 / 148.53 / 149.38 ms    median 149.38
patched (no sync)         142.82 / 144.08 / 140.24 ms    median 142.82
                                                          -6.57 ms  (-4.4%)
```

Under contention. Scaled by the contention factor derived in §1.7 this is ~2.6 ms/step
uncontended — ~5% of the GPU portion, ~2% of a `code` step. Small, free, bit-identical, and
a hard prerequisite for `torch.compile` (§2.3).

### 1.7 What could not be measured, stated plainly

- **Uncontended GPU step time.** Every probe ran alongside a live training job holding ~75%
  of the card. Ratios taken within one invocation are sound; absolute milliseconds are
  upper bounds. The contention factor is derived once, from the one uncontended anchor
  available — the real `code` run's GPU portion of 49.8 ms/step against the probe's 123.94
  ms/step at the same batch, dtype and width — giving **2.49x**. Where an uncontended
  absolute is quoted below it is that division, and it is labelled *derived*.
- **Whether bf16 changes final recall.** That needs a full run against a full run. It is
  Experiment 3, not a measurement.
- **The 5080 and the 1080 Ti.** Neither was probed; no CSD training has run on either. The
  bf16 statements about them in §2.1 come from compute capability, which is a spec fact.
- **Natural Questions tokenizer cost**, and therefore `retrieve`'s exact tokenizer share.
- **Batch homogeneity for `compress` and `retrieve`.** Neither corpus carries a grouping
  column, so the `code` finding could not be checked against them. `retrieve` is at least
  shuffled, since it declares extra sources; `compress` is not, and is unverified.

---

## 2. The candidates

### 2.1 bf16 / AMP mixed precision — **ADOPT**

**What it buys, measured.** At batch 256, width 96, back to back under identical
contention: fp32 123.94 ms/step, bf16 62.73 ms/step. At batch 512: 237.66 vs 120.27.
**1.98x and 1.98x** — two independent batch sizes agreeing, and consistent with the ~2x
fp32-to-bf16 tensor-core ratio on Ampere (sm_86). Memory: the activation slope falls from
12.66 to 7.25 MiB per pair, **0.573x**, which raises the largest fitting batch from ~1,536
to ~2,560.

Applied to the real `code` step, whose GPU portion is 49.8 ms of 131.3 *(derived)*:

| tokenizer state | fp32 step | bf16 step | end-to-end gain from bf16 |
|---|---|---|---|
| today (serial, 81.5 ms) | 131.3 ms | ~106.7 ms | **1.23x** |
| `encode_batch` (14.5 ms) | 64.3 ms | ~39.7 ms | **1.62x** |
| pre-tokenized (0 ms) | 49.8 ms | ~25.2 ms | **1.98x** |

*bf16's end-to-end value is 1.6x larger once the tokenizer is out of the loop*, because
Amdahl's law applies to the 62% the tokenizer owns. That ordering is the single most
important thing in §3.

**What it costs.**
- **The correctness trap is the loss, not the model.** Under `torch.autocast`, `a @ p.T` in
  `info_nce` runs in bf16. bf16 carries 8 mantissa bits; the logits are divided by
  `temperature=0.05`, i.e. multiplied by 20, so a unit-cosine logit lands near ±20 where the
  bf16 quantum is ~0.125. Softmax over 2,560 classes on logits that coarse is a real
  perturbation of the ranking, and it would show up as a quiet recall loss that looks like
  variance. `F.cross_entropy` is on autocast's fp32 promote-list and will upcast itself; the
  **matmul will not**. `info_nce` must cast its inputs to fp32 before `normalize` and the
  matmul. The cost is nil — at batch 2,560 that matmul is 1.7 GFLOP against a step that
  already does hundreds.
- Use `torch.autocast`, never `model.to(torch.bfloat16)`. Autocast keeps fp32 master
  weights, so the optimizer, the gradient clip and the saved `state_dict` stay fp32.
- No `GradScaler`. That is fp16's problem; bf16 has fp32's exponent range.

**Interactions.** Master weights stay fp32, so PTQ (`quant/ptq.py`) sees exactly what it
sees today and needs no change. The receipt's `emb_std` collapse signal is computed under
`no_grad` in `info_nce` and would be bf16 unless the fp32 cast above is applied — another
reason to apply it.

**Does the 1080 Ti constrain this?** GP102 is sm_61: no bf16, and fp16 at 1/64 throughput.
It constrains **nothing here, provided autocast is used rather than a dtype conversion**,
because autocast is a CUDA-context-local decision and the artifact on disk stays fp32. Two
rules make it safe: gate it on `torch.cuda.is_bf16_supported()` rather than assuming, and
never write a bf16 `state_dict`. The quantized regions are already Pascal-safe for the same
reason — `dequantize_tensor` returns fp32 and the matmuls are fp32. If the fleet ever
*trains* on the 1080 Ti, that run simply runs fp32 at ~2x the step time; nothing breaks.

---

### 2.2 Gradient accumulation — **REJECT** (for this objective, on this hardware)

This is the item the program most needs to get right, because P9.2 states it buys "effective
batch beyond VRAM, which is the single biggest lever on contrastive quality". **For InfoNCE
over in-batch negatives, that is not what gradient accumulation does.**

**The precise distinction.** `info_nce` computes `logits = (a @ p.T) / temperature` over
whatever tensor it is handed. Under accumulation of K micro-batches of size B, it is called
K times on B x B matrices. Each anchor discriminates its positive against **B−1 negatives,
K times over** — never against KB−1. What accumulates is the *gradient*, averaged over K
independent B-way classification problems. The variance of the gradient estimate falls by
~K; the difficulty of the task, and the `log B` mutual-information ceiling the objective can
express, do not move at all.

Contrast a true batch of KB: one softmax over KB−1 negatives, ceiling `log KB`. At B=256,
K=4: accumulation leaves the ceiling at 5.55 nats. A real batch of 1024 raises it to 6.93.
These are different objectives, and only the second is the lever P9.2 wants.

**And the premise does not hold here anyway.** Accumulation exists to reach a batch that
does not fit. On this card the batch *does* fit: 78% of it is idle, and the measured memory
model says fp32 reaches 1,536 and bf16 reaches 2,560 with no trickery. Adding accumulation
to reach an effective batch you can simply allocate is complexity bought for nothing, plus a
new correctness surface (scaling the loss by 1/K, clipping once per accumulation window and
not per micro-step, and getting `lr` scheduling right per optimizer step rather than per
micro-step).

**What to do instead.** Raise the physical batch (§3, step 5). If the model later grows
enough that the batch becomes memory-bound, the technique that genuinely buys negatives
without memory is **GradCache** (Gao et al., 2021): forward every micro-batch under
`no_grad` to build the full `[KB, D]` embedding matrix, compute the true KB-way loss and its
gradient with respect to those embeddings, then re-forward each micro-batch with grad and
backpropagate the cached embedding gradients. Mathematically identical to a true KB batch,
at the activation memory of one micro-batch and ~2x the forward compute. That is the item
worth writing down for later — not accumulation.

**Verdict: REJECT.** Revisit as GradCache only if a measured run is memory-bound at the
batch the InfoNCE curve actually wants. Record the reason in `program/REMAINING.md`, because
"more gradient signal" and "more negatives" are easy to conflate and the conflation will
come back.

---

### 2.3 `torch.compile` — **ADOPT LATER**

**Ceiling, measured.** It can only touch the GPU portion: 38% of a `code` step today, ~78%
of a `compress` step. On a 4-block, dim-256 transformer the realistic gain is kernel fusion
of the LayerNorm/GELU/residual chain plus reduced launch overhead — 10-30% of GPU time. On
`code` that is 5-15 ms of 131.3, i.e. **4-11% end-to-end**, against 62% available from the
tokenizer and 24% from bf16. Third-order today. After the tokenizer fix and bf16 it becomes
proportionally more interesting, which is why it is ADOPT LATER rather than REJECT.

**What it costs — two blockers, both concrete.**
1. **The data-dependent branch of §1.6** is a guaranteed graph break every block, every
   step. `torch.compile` would either break the graph 8 times per step (deleting most of the
   benefit) or, with `fullgraph=True`, refuse to compile. Fix §1.6 first; it is free and
   bit-identical.
2. **Dynamic shapes.** `_tokenize` pads to the longest item in the batch, so the sequence
   width varies per step: `compress` was measured between 17 and 62, `retrieve` between 24
   and 96. Dynamo guards on shape; each distinct width triggers a recompilation until
   `torch._dynamo.config.cache_size_limit` (default 8) is exceeded, after which it silently
   falls back to eager — a slower run that reports success. Either `dynamic=True` (which
   gives up some fusion) or bucket the padding to a small set of widths.

   `code` is the exception and the natural first target: its mean padded width is **96.0**
   with a min of 96 — every batch already saturates `max_len`, so its shape is static today.

**Interactions.** Compile after bf16, not before: compiling then changing dtype invalidates
the compiled artifact and doubles the warmup you pay. Warmup itself is real — 30-60 s on
first call — which is 3-6% of a 17-minute `code` run and would be most of a short
experiment. Gate on run length.

---

### 2.4 Gradient checkpointing — **REJECT**

It trades ~30% more compute for less activation memory. Measured activation cost is 12.66
MiB per pair in fp32 and 7.25 in bf16; at the current batch of 256 that is 3,474 MiB on a card
with 17,973 MiB measured free during a real run. **It solves a problem that does not exist**, and adopting it would make
every run slower to reclaim memory nobody is short of.

The threshold is explicit from the memory model: it becomes relevant only above batch ~1,536
fp32 / ~2,700 bf16 at this depth. And at that point the better move is usually not
checkpointing — InfoNCE's return on batch size is logarithmic, so going from 2,560 to 5,120
buys 0.69 nats of ceiling for a doubled cost, while depth or width bought with the same
memory may buy more. Revisit only with a measured curve of recall against batch size that is
still climbing at the memory limit.

---

### 2.5 EMA of weights — **ADOPT**

**What it buys, and the evidence it is needed here.** The receipts show the final iterate is
noisy. `code` (6000-step run), last 3,000 steps: loss 0.106 → 0.327 → 0.574 → 0.088;
in-batch accuracy 0.984 → 0.926 → 0.863 → 0.980. `compress`, last 2,700 steps: in-batch
accuracy 0.883 → 0.594 → 0.617 → 0.535 while held-out recall@1 *rose* 0.418 → 0.475 → 0.496.
The saved `final.pt` is whichever of those nearby iterates step N happened to land on. An
EMA (decay 0.999-0.9999) evaluated alongside the raw weights is the standard, cheap fix, and
contrastive bi-encoders are exactly where it pays.

**What it costs.** One fp32 copy of the weights: 61.11 MiB, **0.27% of the card**. One
lerp over 16M parameters per step — call it a fraction of a millisecond against a 131 ms
step. No change to the objective, the schedule, or the corpus.

**Interactions and the trap.**
- The mechanism already exists in the codebase: `vl_pretrain`/`vl_jepa` run an EMA target
  encoder with `ema_base` 0.996 → `ema_final` 1.0. Reuse that schedule shape rather than
  inventing a second one.
- Partly redundant with cosine-to-zero, which already averages implicitly as the LR
  vanishes. Expect a smaller gain than in a constant-LR setting — that is a reason to
  measure it, not to skip it.
- **The trap is the gate.** If the receipt reports the EMA metric where it used to report
  the raw one, `beats_untrained` silently changes meaning and every historical comparison
  breaks. Record **both** (`held_out` and `held_out_ema`), pick the artifact explicitly, and
  say in the receipt which one `final.pt` holds.

---

### 2.6 Early stopping on the gate metric — **REJECT as written; ADOPT a divergence abort**

P9.7 says it "stops burning GPU on a run that has plateaued". Measured against the actual
histories, it would not have:

| region | metric at 2/3 through | at the end | bought by the last third |
|---|---|---|---|
| code (8000) | 0.9531 @ 6665 | 0.9590 @ 7998 | +0.0059 recall@1 for ~350 s |
| retrieve (8000) | 0.6816 @ 6665 | 0.6855 @ 7998 | +0.0039 for ~91 s |
| compress (8000) | 0.4746 @ 6665 | **0.4961** @ 7998 | **+0.0215 — still climbing hard** |

Two of three regions were still improving at the final step, and on `compress` early
stopping would have cost accuracy. The available saving is ~0-10% of run time on one region.

There is also a structural conflict: `_lr_at` is cosine-to-zero over `cfg.steps`. Stopping
early stops at a non-zero LR and forfeits the annealing, so an "early stop" is not the same
model as a shorter run — it is a worse one. Early stopping and cosine-to-zero want different
schedules, and changing the schedule to accommodate a feature with a measured 0-10% payoff
is a bad trade.

**What is worth having is the opposite failure.** A run whose configuration is broken
currently burns the full 17 minutes and then fails its gate. A **divergence abort** — stop
if the held-out metric is still below the untrained baseline after a set fraction of the run
— catches that in minutes.

The threshold has to be set from the data, and the data has a trap in it. `code` at step
1333 of 8000 scored recall@1 **0.357 against an untrained baseline of 0.400**: healthy runs
go *below* baseline early, because a random-init encoder was measured, at the time, to
score 0.40 from lexical overlap and training destroys that before learned structure
replaces it. That 0.400 baseline (and the run this table is drawn from) predates the
2026-09-02 shuffle fix and was measured against `code`'s then-unshuffled,
two-repository-confined holdout; the measured floor against the current (shuffled)
holdout is **0.2285**, not 0.40 (`docs/design/evidence/w2c-untrained-baselines-2026-09-03/
README.md`). Both `csd-train-all.py` and `pretrain.py` document the (now corrected) figure
in their module docstrings. An abort threshold sized off this pre-fix run should be
re-validated against a post-fix run before being relied on; directionally, an abort
threshold before ~40% of steps would kill healthy runs. Measured recovery points from that
same pre-fix run: `code` crosses back above baseline between step 1333 (0.357) and 2666
(0.883); `compress` and `retrieve` start from near-zero baselines and never dip.

**Verdict: REJECT** the plateau-stopper. **ADOPT** an abort at >=50% of steps if the metric
is still below `untrained_baseline`, recorded in the receipt as an abort rather than a gate
failure so the two are distinguishable. Its enabler is denser evaluation: `eval_every` is
`steps // 6` today (6-8 points across an entire run), which is far too coarse to decide
anything, and each eval costs on the order of 0.2 s *(derived: 1,024 texts tokenized plus 32 forward
passes at batch 64; too small to isolate inside `elapsed_s`)* — 40 evals would cost ~9 s of a
1,050 s run.

---

### 2.7 LoRA / adapters for fine-tuning — **ADOPT LATER**, on the right justification

**The usual justification does not apply here.** LoRA exists so you do not have to
materialise optimizer state for a model that will not fit. Full fine-tuning of this encoder
costs 122.2 MiB of AdamW state — **0.5% of the card**. On memory grounds LoRA buys nothing,
and a design doc that claimed otherwise would be repeating a slogan.

**The justification that does apply is capability per byte at deployment.** Adapter sizes on
this exact architecture, LoRA on the body only (`qkv`, `proj`, `mlp.0`, `mlp.2` over 4
blocks): `4 blocks x r x (1024 + 512 + 1280 + 1280) = 16,384·r` parameters.

| rank | adapter params | % of 16.02M | % of the 3.15M body | fp32 size |
|---|---|---|---|---|
| 8 | 131,072 | 0.82% | 4.2% | 524 KB |
| 16 | 262,144 | 1.64% | 8.3% | 1.05 MB |
| 32 | 524,288 | 3.27% | 16.6% | 2.10 MB |

P9.5's gate — "beats frozen baseline at <5% of trained params" — is met at every rank up to
32. Shipping N task variants as one 6.5 MB quantized base plus N x 0.5 MB adapters, instead
of N x 6.5 MB models, is a real capability-per-byte win and it is the honest reason to
build this.

**Costs and traps.**
- **LoRA and PTQ interact badly if merged in the wrong order.** `ptq.py` measures
  sensitivity per tensor on the task metric. Merging `B@A` into `W` and *then* quantizing
  re-runs the whole sensitivity search per adapter, which defeats the point of a shared
  base. Keep the adapters fp32 and unmerged on top of the quantized base — 524 KB of fp32
  against 6.5 MB of packed weights is affordable, and the base's plan stays valid for all of
  them.
- 80.3% of the parameters are the embedding, which LoRA does not touch. On a region whose
  fine-tuning task shifts vocabulary distribution, a body-only adapter has less headroom
  than the parameter count suggests.
- There is no fine-tuning workload in the program yet. P4 (router) and P5 (composed mind)
  are what would create one. Building the adapter machinery before there is a second task to
  adapt to is speculative.

**Verdict: ADOPT LATER**, after P4/P5 produce a real fine-tuning task, and justified in the
program by artifact economics rather than by training memory.

---

### 2.8 QAT versus the existing PTQ — **REJECT for now; run the cheap prerequisite first**

**PTQ has not hit its wall, and that is measurable.** All three regions are within a 0.01
tolerance at a 3-bit floor, with drops of 0.0039 (`code`), 0.0078 (`compress`), 0.0078
(`retrieve`). `compress` needed **zero** promotions — all 17 quantizable tensors sat at 3
bits. `code` needed one (`blocks.0.qkv` → 4 bits). The greedy allocator barely used its
ladder.

QAT is a large change: a fake-quantize forward path, straight-through estimators, a second
training loop, and a new class of silent failure (a QAT run that trains beautifully against
its own fake-quant and degrades under the real packer). It earns that only if PTQ is the
binding constraint. Nothing measured says it is.

**The prerequisite measurement, which costs almost nothing.** `LADDER` already contains 2.
`build_plan` starts at `aggressive_bits=3` because that is what it was called with. Re-run
PTQ against the existing checkpoints with `aggressive_bits=2` — evaluation only, no
training, ~30 evals per region on a 512-pair holdout, single-digit minutes. Two outcomes,
both decisive:

- **2-bit PTQ lands within tolerance** → QAT is unjustified. Stored bytes fall from ~6.5 MB
  to ~4.5 MB (2.25 effective bits/param including the per-channel scale and zero-point,
  which grow in relative terms as the codes shrink — for `embed` the metadata is 402 KB
  against 3.22 MB of 2-bit codes, a 12% overhead). `capability_per_mb` for `code` rises from
  0.147 to ~0.21, **1.44x**.
- **2-bit PTQ blows the tolerance and the greedy allocator promotes everything back to 3-4
  bits** → *then* QAT has a measured job, and the receipt showing the promotion cascade is
  the justification for building it.

**Verdict: REJECT** QAT until that experiment has been run. Note in `program/REMAINING.md`
that P9.6's gate ("at 3-bit, QAT beats the PTQ result") is testing at a width where PTQ is
already comfortably inside budget — the gate should be restated at 2 bits.

---

### 2.9 Tokenizer cost — **ADOPT. This is the largest single win and it is not in P9.**

Two changes, in increasing order of value.

**(a) `encode_batch` instead of a Python loop.** `tokenizers` is Rust with a rayon
thread-pool, and `[tok.encode(t) for t in texts]` pins it to one core. Measured **5.6-6.9x**
across all four corpora (§1.2). For `code`: 81.5 → 14.5 ms/step, i.e. **131.3 → 64.3
ms/step, 2.04x end-to-end**, from replacing one list comprehension. Risk is close to zero:
`encode_batch` returns the same `Encoding` objects in the same order.

**(b) Pre-tokenize the corpus once.** The stronger version. `code` runs 8,000 steps x 256 =
2,048,000 pair draws over 214,813 training pairs — **9.5 epochs**, so every pair is
tokenized ~9.5 times to produce identical ids. Tokenizing once up front costs 429,626 texts
x 28.2 µs (the measured `encode_batch` rate, 14.46 ms per 512 texts) = **~12 s**, and removes tokenization
from the step loop entirely:

| | ms/step | run time (8000 steps) | speedup |
|---|---|---|---|
| today | 131.3 | 1,050 s | 1.00x |
| (a) `encode_batch` | 64.3 | 514 s | 2.04x |
| (b) pre-tokenized | 49.8 | 398 s + 12 s | **2.56x** |
| (b) + bf16 | ~25.2 *(derived)* | ~202 s + 12 s | **~4.9x** |

Memory for the cache: 214,813 pairs x 2 sides x <=96 tokens x 4 bytes = **165 MB** worst
case on a 46 GB machine, and less in practice since it should be stored ragged.

**The trap, and it is the whole thing.** Store the token ids **ragged** and pad per batch at
collate time. Padding everything to `max_len` up front would take `compress`'s mean width
from 32.9 to 96 and roughly triple its GPU cost — turning a 2x win into a loss. The current
`_tokenize` semantics (pad to the longest item *in this batch*) must be preserved exactly.

**Interactions.** Everything downstream gets cheaper to measure: with tokenization out of
the loop, a bf16 or `torch.compile` A/B is a clean GPU measurement instead of a 38%-of-the-
step measurement. It also makes the `code` step's shape static, which is `torch.compile`'s
other precondition. Do this first.

---

### 2.10 Unconditional data shuffling — **ADOPT. One line, and it gates the batch experiment.**

From §1.5(a): the median `code` training batch draws all 255 negatives from one repository,
because `build_splits` only shuffles when `extra_sources` is non-empty. Making the shuffle
unconditional is a one-line change to an existing seeded, reproducible shuffle.

**What it buys.** Negatives that are actually negatives. It also makes the held-out set a
random sample of the corpus rather than the first 2-3 repositories, which is what
`capability_per_param` needs if it is ever to be compared across regions or against a
published baseline.

**What it costs.** The comparison. `code`'s recall@1 0.9590 was measured on a homogeneous
pool; a shuffled holdout is a different, probably harder pool, and the number will move —
possibly down. **That is a corrected measurement, not a regression**, and the receipt must
say so explicitly or the next reader will treat it as one. It also invalidates the existing
`code` and `compress` quant receipts, whose `fp32_metric_receipt` is tied to the old
holdout.

**Ordering.** This must land **before** any batch-size change, or the batch experiment
attributes to batch size what is really the interaction between batch size and corpus
ordering.

---

### 2.11 Factorized or pruned embedding — **ADOPT LATER**, and it is the biggest thesis lever

80.3% of the parameters and 74% of the stored bytes are an embedding table (§1.4), which no
technique in the P9 list touches. Two options, measured against §1.5(b).

**Vocabulary pruning** to the tokens covering 99.9% of a region's uses: `code` 20,950 rows
(12.87M → 5.36M params, total 16.02M → **8.52M, −47%**), `compress` 14,041 (total →
**6.75M, −58%**), `retrieve` 41,718 because gooaq touches 92.4% of the vocabulary (total →
13.83M, −14%). Effective per region, weak for `retrieve`, and it introduces a per-region
id-remapping table plus an OOV path — a new contract between the tokenizer and every
consumer, right before P5/P6 want the regions composable.

**Factorized embedding** (ALBERT-style, `vocab x r` then `r x dim`) needs no vocabulary
contract change at all. At r=64: 50,257 x 64 + 64 x 256 = 3.23M against 12.87M, so
16.02M → **6.39M parameters, −60%**, uniformly across regions. Carried through the existing
quantizer at today's measured 3.27 effective bits/param, stored bytes fall from 6.54 MB to
~2.60 MB and `capability_per_mb` for `code` rises from 0.147 to **~0.37, a 2.5x gain** — more
than anything quantization can add on top of what PTQ already achieves.

**Costs.** It is an architecture change, so it requires retraining and invalidates every
existing checkpoint and quant receipt. Recall may fall: a rank-64 bottleneck on the input
side is a real capacity reduction, and whether 256-dim token embeddings on a 4-block encoder
have 64 dimensions of usable structure is an empirical question. The evidence that it might:
the eval receipts report `effective_rank_ratio` of 0.38 / 0.46 / 0.48 on the **output**
embeddings — these models already use well under half their declared dimensions.

**Verdict: ADOPT LATER.** Not in the P9.1-P9.3 sequence, because it changes what is being
trained rather than how. But it belongs in the program: it is the largest measured
capability-per-byte lever available, and P6 (foundation training) is the natural place to
adopt it, before a shared embedding is baked into a composed model.

---

### 2.12 Declarative config-driven runs — **ADOPT**

**What it buys, and what is already there.** More than half of this exists: the receipt
already embeds the complete run config (`receipt["config"]` is `asdict(PretrainConfig)` plus
the resolved `TextEncoderConfig`), a corpus fingerprint, and a shard list. A receipt is
already a config in every respect but one — **it cannot be replayed**, because
`pretrain_region` writes the config with `shards` excluded:

```python
"config": {**{k: v for k, v in asdict(cfg).items() if k not in ("shards", "encoder")}, ...}
```

`corpus.shards` records basenames only (`"train-00000-of-00004-….parquet"`), not paths, so
reconstructing the exact input requires knowing `csd-train-all.py`'s glob table. That is the
gap, and it is small: record the resolved absolute shard list, add a `--from-receipt PATH`
that rebuilds a `PretrainConfig` and re-runs it, and P9.4's gate — "same config re-run
reproduces the receipt" — is checkable. Seeding is already deterministic and the corpus
fingerprint already detects input drift.

**What it costs.** Almost nothing in risk: the loader is a new module, and the entry point
is `scripts/csd-train-all.py`, which is a *different file* from the training loop. It is the
one P9 item that can proceed **in parallel** with everything in §2.1-§2.10 without a merge
collision. Do not replace the CLI flags — keep them as overrides, or unattended
`systemd`-driven runs (`deploy/systemd/csd-train@.service`) lose the ability to vary the
region list per instance.

**Interaction.** This is worth landing early for a reason beyond tidiness: every experiment
in §4 is an A/B between two runs that must differ in exactly one field. Doing that with CLI
flags and remembering which flags were used is how a speed optimisation quietly costs
accuracy and nobody notices.

---

### 2.13 Removing the data-dependent GPU sync — **ADOPT**

§1.6. Measured −6.57 ms/step (−4.4%) under contention, ~2.6 ms/step derived uncontended,
output **bit-identical** (max abs diff 0.0, including an all-padding row). One file
(`model/vl_jepa.py`), one statement, no gate movement, and it is the precondition for
`torch.compile`. Land it first because it collides with nothing.

---

## 3. Recommended sequence

Ordered by measured value per unit of risk. The collision constraint is real:
`regions/pretrain.py` is being edited right now for P0.1 resumable checkpointing, and steps
2, 3, 4 and 6 below all touch it.

| # | change | file(s) | expected gain | depends on |
|---|---|---|---|---|
| 0 | measurement script (this doc's instrument) | `scripts/csd-measure-training-cost.py` | — | — |
| 1 | remove the `if empty.any()` sync | `model/vl_jepa.py` | −2.6 ms/step, bit-identical | — |
| 2 | unconditional corpus shuffle | `regions/pretrain.py` | correctness; enables 5 | — |
| 3 | pre-tokenized ragged corpus cache | new module + `pretrain.py` | **131.3 → 49.8 ms/step (2.64x)** | — |
| 4 | bf16 autocast + fp32 logits in `info_nce` | `pretrain.py`, `text_encoder.py` | **49.8 → ~25.2 ms/step (1.98x)** | 3 (for clean measurement) |
| 5 | batch 256 → 1536 (fp32) or 2560 (bf16), LR rescaled | config only | **10x negatives**; `log B` 5.55 → 7.85 nats | 2, 4 |
| 6 | EMA of weights, both metrics in the receipt | new module + `pretrain.py` | stabilises a noisy endpoint | — (measure after 5) |
| 7 | config-driven runs, replayable receipts | `csd-train-all.py` + new loader | reproducibility; A/B hygiene | — (**parallel-safe**) |
| 8 | divergence abort + denser `eval_every` | `pretrain.py` | kills a broken run in minutes | 7 |
| 9 | `torch.compile` on `code` | `pretrain.py` | 4-11% of what remains | 1, 3, 4 |
| 10 | 2-bit PTQ probe (eval only) | `scripts/csd-quantize.py` | decides §2.8 | — (**parallel-safe**) |
| 11 | LoRA adapters | new module | capability/byte at deploy | P4/P5 |
| 12 | factorized embedding | `text_encoder.py` | **params −60%, capability/MB ~2.5x** | P6 |

**Collision risk, explicitly.** Steps 2, 3, 4, 6, 8 and 9 all edit `regions/pretrain.py`,
and P0.1 is in flight against the same file. **They must land one at a time**, each with its
own receipt, each rebased on the last. Steps 1, 7, 10 and 11 touch disjoint files and can run
concurrently with the `pretrain.py` queue. Step 4 also touches `text_encoder.py`, which
nothing else in this list does.

**Cumulative projection for `code` at 8,000 steps** (steps 1+3+4; batch held at 256 so the
comparison is like-for-like): 1,050 s → **~214 s, ~4.9x**. Steps 4 and 5 together take the
run to batch 2,560 at ~233 ms/step *(derived)*; the same 2.05M pair-draws then take 800
steps and **~186 s** — 5.6x faster than today *and* with 2,559 negatives per anchor instead
of 255.

---

## 4. Experiments that would prove the top three

This project gates every run against an untrained baseline precisely so that a speed
optimisation cannot quietly cost accuracy. Each experiment below therefore fixes the seed,
the corpus fingerprint and the holdout, and changes exactly one thing.

Noise floor. `recall@1` on a 512-pair holdout moves in quanta of 1/512 = **0.00195**. The
two `code` runs on record landed at 0.9590 and 0.9531 — a spread of 0.0059, three quanta —
but they differ in **both** step count (8000 vs 6000) **and** corpus (322,744 vs 214,813
train pairs, different fingerprints), so that is not a seed-to-seed variance estimate. **A
true noise floor has not been measured**, and measuring it is cheap: two runs of the same
config at `seed` 0 and 1. Until then, treat **±0.006** (three quanta, the observed spread)
as the working threshold, and treat any result inside it as unproven rather than as
equal.

### Experiment 1 — pre-tokenized corpus cache (§2.9)

*The largest claimed gain and the one with the most ways to be silently wrong.*

- **Baseline.** `code`, 8,000 steps, batch 256, `seed=0`, corpus fingerprint
  `5e632e98b632c65407030324a438f736`. Receipt `code-20260902T200323Z.json` rescaled to 8,000
  steps, or better, one fresh run on the current corpus.
- **Change.** Pre-tokenize ragged, once, up front; pad per batch at collate.
- **Measure.** (i) `elapsed_s` and derived ms/step; (ii) `held_out.recall@1` and `recall@10`;
  (iii) **an exact-equality assertion on the ids**: for 100 sampled batches, the ids and mask
  produced by the cache must be `torch.equal` to what `_tokenize` produces. This is the real
  test — a timing win is meaningless if the batches changed.
- **Success.** ms/step 131.3 → 45-55; recall@1 within ±0.006 of baseline; ids identical.
- **It did not work if.** Ids differ anywhere — most likely cause is padding to `max_len`
  instead of to the batch maximum, which will show up as `compress`'s mean width jumping from
  32.9 to 96 and its step time *rising*. Also watch resident memory: >1 GB for the cache
  means it was stored padded or as Python lists of ints rather than a ragged int32 buffer.

### Experiment 2 — batch size, the InfoNCE negatives lever (§2.10 + §1.3)

*The headline. Must be run after the shuffle, and the shuffle must be measured separately
first or the two effects are inseparable.*

- **Run 2a (isolate the shuffle).** `code`, 8,000 steps, batch 256, unconditional shuffle,
  everything else identical. Expect recall@1 to **move, plausibly downward**, because the
  holdout becomes a random sample rather than 2-3 repositories. Record the new number as the
  baseline for 2b and mark the receipt as a re-baselining, not a regression.
- **Run 2b (batch sweep).** From 2a's baseline, batch ∈ {256, 512, 1024, 2560} with steps
  scaled inversely to hold pair-draws at 2.05M (8000 / 4000 / 2000 / 800) and LR scaled by
  `sqrt(B/256)` — 3.0e-4 / 4.2e-4 / 6.0e-4 / 9.5e-4 — with `warmup_steps` held at
  `steps // 15`.
- **Measure.** `held_out.recall@1` and `mrr` against batch; `elapsed_s`; peak VRAM from
  Prometheus against the §1.3 model; and `repr.anisotropy` and `repr.effective_rank_ratio`
  from the benchmark battery, because more negatives should spread the space and this is
  where that shows.
- **Success.** recall@1 rises monotonically with batch, or plateaus with wall-clock falling;
  `effective_rank_ratio` rises above the current 0.38; peak VRAM matches `233 + 12.66·B`
  (fp32) within 5%.
- **It did not work if.** recall@1 is flat or falls with batch. Two causes to separate before
  concluding anything: (i) the LR rule is wrong — retry batch 2560 at linear scaling and at
  unscaled LR before blaming batch size; (ii) 800 optimizer steps is simply too few for
  cosine annealing to do its work, which would show as a training loss that is still falling
  at the final step. If recall rises but `anisotropy` also rises toward 1, the space is
  collapsing and the recall number is not real — that is exactly what `eval/benchmark.py`'s
  representation family exists to catch.

### Experiment 3 — bf16 autocast (§2.1)

*The one most likely to cost accuracy invisibly, because bf16 has 8 mantissa bits and the
loss divides logits by 0.05.*

- **Baseline.** Run 2a's fp32 result at the chosen batch, same seed, same corpus.
- **Change.** `torch.autocast("cuda", dtype=torch.bfloat16)` around the forward, with
  `info_nce` casting to fp32 before `normalize` and the logits matmul.
- **Measure.** ms/step; peak VRAM; `held_out.recall@1`, `recall@10` and `mrr`; the receipt's
  `emb_std`; and `repr.anisotropy` and `repr.alignment` from the battery.
- **Success.** ms/step ~0.5x; peak allocated matches `245 + 7.25·B` within 5%; **recall@1
  within ±0.006 of the fp32 run**.
- **It did not work if.** recall@1 drops by more than 0.006 while the loss curve looks
  normal. Before accepting that as "bf16 costs accuracy", **re-run with the fp32 cast in
  `info_nce` removed and then restored** — a drop that disappears when the logits matmul is
  forced to fp32 is the temperature-scaling precision trap of §2.1, not a property of bf16,
  and it is fixable rather than a reason to reject. A second signature of the same fault:
  `in_batch_acc` at the end of training being materially below the fp32 run's while the loss
  matches.
- **Also check.** `final.pt` must load and evaluate identically on a machine without bf16.
  The 1080 Ti (sm_61) is the fleet's Pascal card; a `state_dict` that came out bf16 rather
  than fp32 would be the failure, and this is the cheapest place to catch it.

---

## 5. Summary of verdicts

| technique | verdict | one-line reason |
|---|---|---|
| Pre-tokenized corpus cache | **ADOPT** | 62% of a `code` step is a single-threaded CPU tokenizer; 2.6x |
| Unconditional corpus shuffle | **ADOPT** | median `code` batch draws all 255 negatives from one repo |
| Remove the `if empty.any()` sync | **ADOPT** | 8 device syncs/step; bit-identical to remove |
| bf16 autocast | **ADOPT** | 1.98x measured twice; 0.573x activation memory; cast logits to fp32 |
| Raise physical batch to ~2,560 | **ADOPT** | 78% of the card is idle; in InfoNCE the negatives *are* the batch |
| EMA of weights | **ADOPT** | measured late-run oscillation; costs 0.27% of the card |
| Config-driven, replayable runs | **ADOPT** | the receipt is already a config; it just cannot be replayed |
| Divergence abort | **ADOPT** | (replaces plateau early-stopping) fails a broken run in minutes |
| `torch.compile` | **ADOPT LATER** | ceiling is 4-11% end-to-end; needs the sync fix and static shapes |
| LoRA / adapters | **ADOPT LATER** | buys artifact bytes, not training memory; needs P4/P5 first |
| Factorized embedding | **ADOPT LATER** | −60% params, ~2.5x capability/MB — the biggest thesis lever |
| Early stopping on the gate | **REJECT** | 2 of 3 regions still improving at the final step; fights cosine-to-zero |
| Gradient accumulation | **REJECT** | gives more gradient signal, **not more negatives**; and the batch fits |
| Gradient checkpointing | **REJECT** | trades compute for memory that is 78% free |
| QAT | **REJECT for now** | PTQ is inside budget at 3 bits; run the 2-bit PTQ probe first |
