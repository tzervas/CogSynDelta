# W1 — token-vs-pooled effective rank, measured 2026-09-02

Evidence for §4.0 of `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md`. Copied into the repo
so a ratifier reading from the tree can reach the artefact behind the document's largest
measured claim; the originals lived in a session scratchpad, which is not a citation.

**Nothing here is regenerated on read.** These two files are the run's own outputs, byte for
byte. `measure_w1.py` is read-only by construction: it never writes into `src/`, into the repo,
or into `/akula-data/csd/receipts/`.

## What was measured

For each region, on 512 held-out items:

- **pooled surface** — `pool(tokens(x))`, one vector per item, `[512, D]`.
- **token surface** — `tokens(x)` before pooling, all valid (unmasked) positions of all items
  stacked, `[Σ t_valid, D]`.

and, on each surface, two different effective-rank definitions plus mean pairwise cosine. The
pre-pool capture was verified **bit-exact** against the deployed forward pass
(`TextEncoder.forward()`, `IJEPA.encode()`) on an 8-item check per region, `max abs 0.00e+00`;
those checks are recorded in `results.json` under `verified_notes`.

## Both rank definitions, and why there are two

`results.json` records both for every surface. They are **different statistics and they do not
agree on this data.**

- **`*_pr_rank` — participation ratio.** `(Σ s_i²)² / Σ s_i⁴` over the singular values of the
  column-centred matrix. `measure_w1.py:197-211` (`pr_effective_rank`). This is the definition
  W1's rule was pre-committed against in revision 1 of the design doc, before the measurement.
  It weights the **spread** of the spectrum: a few dominant directions pull it down hard.
- **`*_entropy_rank` — entropy-effective rank.** `exp(H)` of the Shannon entropy of the
  linearly-normalised singular-value spectrum, i.e. `cogsyndelta.eval.benchmark.effective_rank`
  called with `sample=N` to disable its default subsampling. `measure_w1.py:214-222`. This is
  the definition behind the tree's existing "8.7 of 128" figure, so it is the project's
  comparable prior. It weights the **tail**: many small directions raise it.

Trained production checkpoints, token-global vs pooled:

| region | PR pooled | PR token | PR ratio | H pooled | H token | H ratio |
|---|---:|---:|---:|---:|---:|---:|
| `code` | 42.63 | 28.09 | **0.66×** | 114.98 | 138.87 | **1.21×** |
| `compress` | 45.33 | 35.57 | **0.78×** | 126.86 | 147.49 | **1.16×** |
| `retrieve` | 40.71 | 40.65 | **1.00×** | 117.72 | 142.86 | **1.21×** |
| `vl_latent` | 18.07 | 23.54 | **1.30×** | 129.83 | 239.09 | **1.84×** |
| `compress_repo_local` (not production) | 7.12 | 14.21 | **2.00×** | 55.23 | 92.42 | **1.67×** |

**The sign reverses.** Under participation ratio three of four production regions fall below
1.0 and all four sit inside the pre-committed ≤1.5× dead band ("the bet is dead"). Under
entropy-effective rank all four exceed 1.0 and `vl_latent` at 1.84× is outside the dead band
entirely. The design doc reports both and treats the W1 verdict as **provisional pending W1d**,
the matched read-out probe, whose rule does not depend on either definition.

## Checkpoints, by sha256

| region | checkpoint | sha256 | step | production |
|---|---|---|---|---|
| `code` | `/akula-data/csd/receipts/code-checkpoints/final.pt` | `dac38d099072481e3f958b33cbca3e965c36b3ca210cc261654b29819f380b03` | 8000 | yes |
| `compress` | `/akula-data/csd/receipts/compress-checkpoints/final.pt` | `c71cb3cd4048dd2b723e511af8b132f06d7a4d69c8b775a5d34e1c18932b7ad6` | 8000 | yes |
| `retrieve` | `/akula-data/csd/receipts/retrieve-checkpoints/final.pt` | `24059d2ad2e2806be4128195efedd72a9583f09c43a0f2d59693eb9fd401dd46` | 8000 | yes |
| `vl_latent` | `/akula-data/csd/receipts/vl_latent-checkpoints/step-8000.pt` | `6c5412bf8cde19f472c683dbe885c453f46c1ba213afdb639c7e2c5b1612a930` | 8000 | yes |
| `compress_repo_local` | `<repo>/receipts/compress-checkpoints/final.pt` | `75880e03b42e03c153f0da42ebe4eb8c4e3337ae26f920d1eed4cd6c8185bf16` | 2000 | **no** |

`compress_repo_local` is an undertrained, non-production checkpoint (step 2,000 against 8,000,
and `max_len` 256 against the production 96). It is reported because it is the same architecture
as production `compress` and lands at 2.00×, four thousandths under the pre-committed "≥2× ⇒ no
retrain" bar, against production `compress`'s 0.78× — a 2.5× spread between two checkpoints of
one region on the statistic that licenses retraining every region. That spread is the argument
for W1d, not evidence against the retrain.

Encoder shape (`dim`, `depth`, `n_heads`, `max_len`, `vocab_size`) is read from each
checkpoint's own saved `config` field, never hardcoded, so it cannot drift from what was trained.

## The command

From the CogSynDelta repo root, on akula-prime `.98` (RTX 3090 Ti), with the corpus mounted at
`/mnt/fleet-datasets/csd` and the tokenizer at `/mnt/fleet-datasets/tritter/gpt2_tokenizer.json`:

```
uv run --no-sync python measure_w1.py
```

`generated_utc` in `results.json`: **2026-09-03T01:34:22Z**. The script writes `results.json`
next to itself (its `OUT_DIR` is the session scratchpad path it ran from); re-running it from
this directory will overwrite the copy here, so re-run it elsewhere and diff.

## Collateral results recorded in the same file

- `cross_region_cka` — pooled and token-aligned linear CKA between the three text regions.
  Trained max **0.336**. Untrained is `1.0` **by construction**, because the three text regions
  share `seed = 0` and an identical `TextEncoderConfig`, so their untrained models are literally
  the same weights. It must never be cited as a baseline.
- `pooled_mean_pairwise_cosine` — falls from 0.84–0.98 untrained to 0.02–0.36 trained, i.e.
  training worked; the verdict is about token-vs-pooled, not about whether the regions learned.
