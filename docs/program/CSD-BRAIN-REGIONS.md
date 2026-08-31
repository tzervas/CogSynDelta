# CSD brain regions — specialized submodels of one mind

Not `STATUS.md`. Map of **live** PoC modules vs **intended** specialists.
Regions are MoE-adjacent experts inside **one** CogSynDelta mind, not a
swarm of agents. Measured numbers stay in `STATUS.md` (PoC-1/2/3).

Do **not** claim a named `code` / `retrieve` / `compress` expert is live.
Do **not** claim mHC, VL-JEPA, quantum, or 10× compression. Autodev
implements. Hosted Grok researches. Never GitHub bot push. Branch
`feat/agent-harness`. Never dual 14B.

Catalog of public sets: [CSD-REGION-DATASETS.md](CSD-REGION-DATASETS.md).
Ladder starts at region-pretrain: [CSD-SCALE-LADDER.md](CSD-SCALE-LADDER.md).

## Why this file exists

A region that never sees its own job-shaped data before the whole mind
is trained becomes dead weight: the softmax gate collapses onto whoever
moved first. Curriculum is **pretrain each region → train the router →
train the assembled mind → scale**. Skipping step 1 is not a shortcut.

## Live (STATUS.md, 2026-08-19)

PoC-1: one LatentVAE region. PoC-2: two regions + softmax top-k mix.
PoC-3: trains that gate with Switch aux `N * sum(f_i * P_i)` so load is
not `{1.0, 0.0}` on a batch of 8. Still not mHC.

| Name | Module | Job | Pretrain-before-assembly | Router trigger | Live? |
|---|---|---|---|---|---|
| `residual_mlp` | `ResidualMLPRegion` (`poc/regions.py`) | Residual update on the shared stream `[B, D]` | Synthetic tokens only (`train-route`). No public corpus yet | Stream that wants a residual MLP step | **yes** (PoC-2/3) |
| `stream_vae` | `LatentVAE` registered as `stream_vae` (`poc/route.py`; class default `latent_vae`) | Encode / reparameterize / decode; compress analog | Synthetic tokens only. Honest stored-byte ratios ~1.3–2.7× in `STATUS.md`, **not** 10× | Reconstruct / quantize the stream | **yes** (PoC-1 substrate; PoC-2/3 second region) |
| `SoftmaxRouter` | `poc/router.py` — **gate, not a region** | Linear `D → N` → softmax → top-k mix of `activate()` | PoC-3 trains the gate on synthetic; real tagged mix is later-assembly | Ambiguous / mix tokens; aux keeps both regions alive | **yes** as the gate (PoC-2/3) |

Protocol: `CognitiveRegion.activate(stream) → [B, D]`.
`LatentVAE.forward` stays the ELBO tuple and is not the routing surface.

## Intended specialists (not live)

Named experts are Phase 3 **after** memory-gate `P1-16` / `G-LIFE`.
They are extra modules of the same mind, not new agents.

| Name | Job | Pretrain-before-assembly | Router trigger | Live? |
|---|---|---|---|---|
| `code` | Docstring ↔ function (search + gen), not next-token over all GitHub | CodeSearchNet Python + `sentence-transformers/codesearchnet` (`pretrain` rows). MBPP / HumanEval are eval | `language=python` + docstring present | **no** |
| `retrieve` | Query / claim → passage rank (P1-09 analog, later P1-15) | FiQA queries if license allows. SciFact / NFCorpus are **eval-only** (NC / ToS) | Query + candidate passage | **no** |
| `compress` | Neighbors stay neighbors in a short latent | STS-B + AllNLI-10k. WikiText-2 is later-assembly recon, not first pretrain | Sentence pair / embedding to reconstruct | **no** as a named expert; `stream_vae` is the PoC analog |
| residual stream | Keep the shared `[B, D]` residual path competent | Evolves `residual_mlp`; no extra public dump required for PoC | Tokens not tagged code/retrieve/compress | **yes** as `residual_mlp` only |
| `route` | Selective activation of the specialists above | **Not** a pretrain corpus. Build `region-route` v0 from already-pretrained catalogs | Label `region_id`; loss = CE + Switch aux | PoC-3 synthetic analog only |

Idle regions must not run `activate()`. `top_k=1` until load is measured
non-collapsed on a labeled batch of 8+. Foundation `common` (C4 / The
Pile / FineWeb) is curriculum step 4, not step 1.

## Curriculum

Mandatory order. Do **not** skip (1). Do **not** start at (4).

1. **R0 per-region pretrain** — each specialist fits **only** its own
   `pretrain` split until it can do its job (isolated; others frozen or
   absent). PoC stand-in today: `python -m cogsyndelta.poc.cli train`
   (LatentVAE) on synthetic tokens. First public split: WikiText-2-raw **train** (region-pretrain now).
2. **R1 router** — train `SoftmaxRouter` (+ Switch aux) so regions
   activate **selectively**. Frozen regions first (`--gate-only`), then
   joint. Tagged mix donated from those pretrained catalogs. PoC
   stand-in: `train-route` on synthetic. Idle experts skip `activate()`.
   Interconnect / mHC is later.
3. **R2 assembled whole-model** — only then joint train of regions +
   gate on the tagged mix (tiny whole-mind). Not a 14B. Not C4 Switch
   pretrain.
4. **R3+ scale** — larger param counts after the tiny assembled mind.
   Private Hub `tzervas/cogsyndelta-region-<name>-<size>` **then**
   `tzervas/cogsyndelta-<size>` **when** `hf/autodev` exists.
   See [CSD-SCALE-LADDER.md](CSD-SCALE-LADDER.md).

## Lab placement (do not fight the pool)

| Work | Where |
|---|---|
| Autodev implement / region-pretrain | 3090 `local/code` (`192.168.1.98`). LocalAI stays loaded. Never dual 14B |
| Tiny CUDA / later region smoke | 5080 `192.168.1.251` exclusive-seq. Comfy stays masked |
| Retrieve-index-light / RAG | 1080 Ti guest `192.168.1.243` **live** (`nvidia-smi` GTX 1080 Ti). No train / PoC CUDA on Pascal |

## Hugging Face (`hf/autodev`)

CSD vault `/akula-data/cabal/csd-vault` (checked 2026-08-31):
`csd/apply-token`, `git/autodev`, `gpu/localai-api-key` only. `hf/`
exists **empty**. Hugging Face has **no** API to mint a fine-grained
write token.

Operator mint (do **not** copy `gpu/huggingface-token`):

```bash
# huggingface.co/settings/tokens → fine-grained write on
# tzervas/cogsyndelta, tzervas/cogsyndelta-eval, tzervas/cogsyndelta-data only
printf '%s' 'hf_…' | SECRET_VAULT=/akula-data/cabal/csd-vault \
  SOPS_AGE_KEY_FILE=$SECRET_VAULT/age.txt secret set hf/autodev
```

Until that exists: **no** Hub publish. Public sets stay at canonical
URLs.

## Next closeable increment (autodev)

**`region-pretrain`:** one existing region (LatentVAE) on
`Salesforce/wikitext` `wikitext-2-raw-v1` **train**, failing pytest
first. Pack: [CONTEXT-PACK-REGION-PRETRAIN.md](CONTEXT-PACK-REGION-PRETRAIN.md).
Not a 14B. Not `G-TRAIN`. P1-09 stays on the Phase 1 board but is not
this steer. 5080 only if tiny CUDA is required later.

`G-TRAIN` bedrock/foundation remains blocked until `G-LIFE`. This file
is a map. One-region public-split pretrain is the closeable slice.
