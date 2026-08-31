# CSD region datasets — tiny experts (not giant pretrain)

Not `STATUS.md`. Catalog of **public** corpora for four CogSynDelta
cognitive **regions** (code, retrieve, compress, route). Regions are
MoE-adjacent specialists of one mind, not agents. Measured PoC numbers
stay in `STATUS.md` (PoC-1/2/3). Do **not** train CSD weights until
`G-LIFE` / Phase 3 (`SELF-HOSTED-DRIVE-TARGETS.md`). Do **not** claim
VL-JEPA, quantum, or 10× compression from this file.

Region map + mandatory curriculum:
[CSD-BRAIN-REGIONS.md](CSD-BRAIN-REGIONS.md).
Scale after assembly: [CSD-SCALE-LADDER.md](CSD-SCALE-LADDER.md).

Private Hub (when a row actually copies data): `tzervas/cogsyndelta-data`
configs `region-<id>` + `common`, eval in `tzervas/cogsyndelta-eval`.
Requires CSD vault `hf/autodev` (not minted; see mint steps below).
Never copy `gpu/huggingface-token`.

**Split column** (every catalog row): `pretrain` = that region's isolated
fit **before** the whole mind trains; `later-assembly` = tagged mix /
joint train / scale **after** those specialists can do their jobs;
`eval-only` = never in any train mix. Do not skip pretrain.

## Why these sets (not The Stack / C4)

PoC-3 already trains a two-region softmax gate on a **synthetic** stream
(`python -m cogsyndelta.poc.cli train-route`). Phase 3 needs **small,
job-shaped** public sets so a tiny expert can learn one job without a
foundation-scale dump.

| Need | Fit | Reject |
|---|---|---|
| Fits a 5080 exclusive-seq smoke / CPU pytest | tens of thousands of rows, not billions of tokens | The Stack 3–6 TB, C4, The Pile, LAION-5B |
| One region learns one job | comment↔code, query↔passage, pair similarity, tagged route mix | undifferentiated “all GitHub” |
| License compatible with later private copy | MIT / Apache-2.0 / BSD / CC-BY / CC-BY-SA; per-file filter | GPL dump, NC as **train** mix, unknown commercial |
| Router can **selectively activate** | explicit domain/task tags or disjoint formats | one blob with no gate signal |

Checked against live PoC (`src/cogsyndelta/poc/{regions,router,route,train_route}.py`):
`SoftmaxRouter` is a linear gate → softmax → top-k mix of
`CognitiveRegion.activate`; Switch aux is `N * sum(f_i * P_i)`
(Fedus et al., 2021). That is MoE gating, **not** mHC.

## Softmax router — how it should gate

Selective activation like Switch/MoE. Idle regions do not run
`activate()`. No 10× claim.

1. **Encode** each example into the shared stream `[B, D]` (same surface
   as PoC ResidualMLP + LatentVAE).
2. **Score** `logits = Linear(D → N_regions)`; `P = softmax(logits)`.
3. **Dispatch** `top_k=1` for tiny specialists (Switch-style). `top_k=2`
   only after load is measured non-collapsed on a labeled batch of 8+.
4. **Activate** only selected rows: `region.activate(stream[selected])`,
   mix with renormalized top-k weights (already in `SoftmaxRouter.route`).
5. **Balance** with Switch aux `N * sum(f_i * P_i)` so the gate cannot
   collapse to `{1.0, 0.0}` (PoC-3 DoD). Uniform routing → aux ≈ 1.0.
6. **Optional label loss** on the first **tagged** mix: cross-entropy
   of `P` vs `region_id` ∈ `{code, retrieve, compress}` **plus** aux.
   Do not drop aux; label-only routing still collapses.
7. **Isolation**: a `region-code` recipe must not silently ingest
   `common` or retrieve qrels. Foundation (`common`) is Phase 3 step 4,
   after bedrock.

Suggested gate prior (not measured):

| Stream cue | Region that should win top-1 |
|---|---|
| Source / identifier / docstring↔body | `code` |
| Query + candidate passage / claim | `retrieve` |
| Embedding or latent to reconstruct / quantize | `compress` |
| Ambiguous / mix tokens | split by softmax; aux keeps all three alive |

PoC today has two regions (`residual_mlp`, `stream_vae`). Named
code/retrieve/compress experts are Phase 3, **after** memory-gate
`P1-16`. Next closeable work is **not** a 14B train.

## Region catalog

Sizes are public cards / papers, not lab downloads. Pin `dataset_id` +
`config` + `revision` before any copy. Filter CodeSearchNet / Stack rows
to MIT / Apache-2.0 / BSD at ingest.

**Split** is the pretrain vs later-assembly cut for that row. A
`pretrain` set may donate a **tagged sample** to `region-route` v0 only
after that region can do its job. Eval rows never enter a train mix.

### `code` — function + NL, not a corpus LM

| Dataset | License | Size | Why it fits | Gate | Split |
|---|---|---|---|---|---|
| [CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net) (Husain et al., 2019) | Per-repo (`other`); keep MIT/Apache/BSD rows only | ~2M comment–code pairs; Python zip ~941 MB; HF train 1.88M / val 89k / test 101k | Docstring↔function is the code **region** job (search + gen), not next-token over all GitHub | Tokens with `language=python` + docstring present → `code`. Leave bare config/HTML to idle. | **pretrain** (`code`, Python train/val); later-assembly: tagged sample into `region-route`. Test held out. |
| [sentence-transformers/codesearchnet](https://huggingface.co/datasets/sentence-transformers/codesearchnet) | Same upstream per-file | 1,375,067 pairs, 492 MB | Ready `(comment, code)` pairs for a tiny embedding/code head | Same as CodeSearchNet; pair encoder, not a 14B. | **pretrain** (`code`); later-assembly: tagged pair sample. |
| [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) (Austin et al., 2021) | **CC-BY-4.0** | 974 problems (`full`); sanitized 427; train 374 / test 500 / val 90 | Tiny exec-supervised Python; fits one pytest + one 5080 smoke | Prompt+code → `code`. Eval, not pretrain. | **eval-only** / later-assembly eval. Sanitized train 374 is a later smoke, **not** a substitute for CodeSearchNet pretrain. |
| [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) (Chen et al., 2021) | **MIT** | 164 tasks, test split only | Held-out functional eval. **Do not train** on it | Never in the train mix. Router must not see HumanEval prompts at train time. | **eval-only** (later-assembly held-out). Never pretrain. |

**Do not start with** `bigcode/the-stack` (3–6 TB, gated ToU). Optional
later sample: `bigcode/the-stack-smol` (10k files/language, 300k rows,
~2.6–3 GB text) **only** after per-file license filter; still larger
than MBPP/CodeSearchNet-Python for a first expert.

### `retrieve` — query/passage + qrels, not MS MARCO-scale

Tiny BEIR slices match P1-09 domain isolation and later P1-15 golden
recall. Copy into private `tzervas/cogsyndelta-eval` only when P1-15
runs; pin revision + license on the card.

| Dataset | License | Size | Why it fits | Gate | Split |
|---|---|---|---|---|---|
| BEIR **SciFact** ([BeIR/scifact](https://huggingface.co/datasets/BeIR/scifact), [allenai/scifact](https://huggingface.co/datasets/allenai/scifact); Wadden et al.) | **CC-BY-NC-2.0** (NC) | ~300 test queries, ~5k docs, ~1.1 rel/q; HF 1k–10k | Scientific claim→abstract retrieval; CPU-rankable | Query/claim text → `retrieve`. **Eval/research only** until a non-NC twin exists. | **eval-only** (later-assembly). NC — not pretrain, not commercial train. |
| BEIR **NFCorpus** ([BeIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus); Boteva et al., 2016) | Academic / NutritionFacts ToS | 323 test queries, 3.6k docs, ~38 rel/q; ~30 MB orig | Small medical IR; graded qrels | Nutrition query → `retrieve`. Do not mix into `code`. | **eval-only** (later-assembly). Academic ToS — not commercial pretrain. |
| BEIR **FiQA-2018** ([BeIR/fiqa](https://huggingface.co/datasets/BeIR/fiqa)) | Redistributed in BEIR (CC-BY-SA on some generated-query mirrors) | 648 test queries, 57k docs, ~2.6 rel/q | Finance QA, still tiny vs MS MARCO | Question → `retrieve`. Domain tag `fiqa` ≠ SciFact (P1-09 isolation analog). | **pretrain** (`retrieve`) if the pinned revision is permissive; later-assembly: tagged queries. Else eval-only. |

BEIR overview: Thakur et al., 2021
([beir-cellar/beir](https://github.com/beir-cellar/beir)).

**Too big for a tiny retrieve expert:** MS MARCO (8.84M passages /
530k train queries), Natural Questions (2.68M docs), HotpotQA (5.23M).
Those wait for foundation `common`, not `region-retrieve`.

P1-09 (now): in-memory oracle, mandatory/global domain, deterministic
rank — **no Hub download**. This catalog is the later P1-15 / Phase 3
source list.

### `compress` — fidelity of a short latent, not 10× marketing

PoC-1/3 already measure LatentVAE + calibrated 8-bit on synthetic
streams. Honest ratios in `STATUS.md` are ~1.3–2.7× stored-byte
(original/stored), **not** 10×. Use small **semantic** sets so a
compress region learns “neighbors stay neighbors,” not ImageNet-scale
codebooks.

| Dataset | License | Size | Why it fits | Gate | Split |
|---|---|---|---|---|---|
| [sentence-transformers/stsb](https://huggingface.co/datasets/sentence-transformers/stsb) (Cer et al., 2017 STS-B) | SemEval / research; pin HF revision | 8,628 pairs (train 5,749 / val 1,500 / test 1,379), 725 kB | Continuous similarity (0–1). Spearman of cosine(compressed) vs gold is the fidelity metric | Sentence pair → `compress`. Do not route to `code`. | **pretrain** (`compress`, train/val); later-assembly: tagged pairs. Test held out. |
| [sentence-transformers/all-nli](https://huggingface.co/datasets/sentence-transformers/all-nli) triplet **sample 10k** (SNLI Bowman et al. 2015 + MultiNLI) | SNLI **CC-BY-SA-4.0**; MultiNLI OANC-style mix | Full card ~2.86M rows / 213 MB; **use 10k train / 1k dev** like the ST tutorial | Anchor/positive/negative for contrastive reconstruction of embeddings | NLI triplet → `compress`. Full 2.8M is not the first mix. | **pretrain** (`compress`, 10k/1k); later-assembly: tagged triplets. Full 2.8M is later-assembly only. |
| [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) `wikitext-2-raw-v1` (Merity et al., 2016) | **CC-BY-SA-4.0** | 36,718 train / 3,760 val / 4,358 test lines; ~5 MB download | Tiny LM reconstruction; Wikipedia articles, not C4 | Wiki line → `compress` (recon) **or** idle if the gate is retrieve/code tagged. Prefer STS/AllNLI for embedding fidelity. | **later-assembly** recon (not first pretrain). Prefer STS-B / AllNLI-10k for step 1. |

Fashion-MNIST (MIT, 60k×784) is a **vision VAE** smoke only; CSD stream
is text/latent first. Do not pull LAION-5B.

### `route` — tagged mix, not C4 Switch pretrain

There is no honest public “router dataset” at tiny scale. Switch
Transformers pre-trained on **C4** (Fedus et al., 2021) — that is giant
pretrain and out of scope. Build `region-route` from the three catalogs
above **after** those regions have completed step-1 pretrain.

| Mix | License | Size | Why it fits | Gate | Split |
|---|---|---|---|---|---|
| **CSD `region-route` v0** (construct; do not train this week) | Intersection of source licenses; drop NC from **train** | ~8–32k tagged rows: MBPP+CodeSearchNet sample, FiQA/NFCorpus queries, STS-B+AllNLI-10k | Teaches the linear gate the three cues in the table above | Label `region_id`; loss = CE + `aux_coef * Switch_aux`. Target: load not `{1,0,0}` on a batch of 8. | **later-assembly** (curriculum step 2). Not a region-pretrain set. Drop NC/eval-only rows from the train mix. |
| Super-NaturalInstructions sample ([allenai/natural-instructions](https://github.com/allenai/natural-instructions), Wang et al., 2022) | **Apache-2.0** | 1,616 tasks; take a few hundred defs + 1–3 examples each | Optional later: task-type routing (QA vs code vs NLI) | Map task category → region; still top-k + aux. | **later-assembly** optional (after `region-route` v0). Not step-1 pretrain. |

PoC-3 already proves recon↓ and aux finite on **synthetic** tokens
(`STATUS.md`). A labeled-stream unit test is the first *CSD* router
increment **after** P1-09 — still not a 14B.

## Rejected for tiny experts

| Corpus | Why rejected now | Split |
|---|---|---|
| The Stack / StarCoderData / full CodeSearchNet-all without license filter | Giant and/or copyleft mix | rejected-now — not pretrain |
| C4, The Pile, FineWeb | Foundation `common` only, Phase 3 step 4 | later-assembly **scale** only, never step 1 |
| MS MARCO / NQ / Hotpot full | Retrieve foundation, not a 5k-doc expert | later-assembly foundation, not `region-retrieve` pretrain |
| LAION-5B, ImageNet-1k as first compress mix | Wrong modality/scale | rejected-now — not pretrain |
| HumanEval, MBPP test, SciFact test | Eval; training on them is leakage | **eval-only** (later-assembly held-out) |
| SciFact / NFCorpus as **commercial train** | NC / academic ToS — eval or wait for a permissive twin | **eval-only** — not pretrain |

## Private Hub + `hf/autodev` (not minted)

Checked `docs/program/CSD-SECRETS.md` (2026-08-31): CSD vault has
`csd/apply-token`, `git/autodev`, `gpu/localai-api-key` only. `hf/`
dir exists empty. Hugging Face has **no** API to mint a fine-grained
write token.

Operator mint (do **not** copy `gpu/huggingface-token`):

```bash
# huggingface.co/settings/tokens → fine-grained write on
# tzervas/cogsyndelta, tzervas/cogsyndelta-eval, tzervas/cogsyndelta-data only
printf '%s' 'hf_…' | SECRET_VAULT=/akula-data/cabal/csd-vault \
  SOPS_AGE_KEY_FILE=$SECRET_VAULT/age.txt secret set hf/autodev
```

Until that exists: **no** Hub publish. Public sets stay at their
canonical URLs. P1-15 copies into `tzervas/cogsyndelta-eval` only after
mint + pin.

## Next closeable increment (autodev, not hosted Grok)

**Still `P1-09`.** Board: gateway retrieve + mandatory/global domain +
deterministic in-memory rank (`feat/gateway-retrieve-domain`). Depends
on P1-04 (met). P1-08 is merged. Do **not** start `G-TRAIN`, dual 14B,
or a full region fit.

Autodev tick: **one failing pytest + one function** in
`memory-gate-wt-p1-09` (`src/` `tests/` only). Not this CogSynDelta
docs tree. Not 14B. Not chroma P1-05.

Later CSD (after P1-09 lands, still not train): one **router or region
unit test** — e.g. tagged batch of 8 with `{code, retrieve, compress}`
cues, `top_k=1`, aux finite, load not collapsed — on CPU. 5080
exclusive-seq only if CUDA is required. 1080 Ti guest is
retrieve-index-light; do **not** ham CUDA/train there.

## Citations (public sources)

1. Husain, Wu, Gazit, Allamanis, Brockschmidt. *CodeSearchNet Challenge:
   Evaluating the State of Semantic Code Search*. arXiv:1909.09436, 2019.
   https://arxiv.org/abs/1909.09436 ·
   https://huggingface.co/datasets/code-search-net/code_search_net
2. Thakur, Reimers, Rücklé, Srivastava, Gurevych. *BEIR: A Heterogeneous
   Benchmark for Zero-shot Evaluation of Information Retrieval Models*.
   NeurIPS Datasets and Benchmarks, 2021.
   https://github.com/beir-cellar/beir
3. Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion
   Parameter Models with Simple and Efficient Sparsity*.
   arXiv:2101.03961, 2021. https://arxiv.org/abs/2101.03961
4. Austin et al. *Program Synthesis with Large Language Models*
   (MBPP). arXiv:2108.07732, 2021.
   https://huggingface.co/datasets/google-research-datasets/mbpp
5. Chen et al. *Evaluating Large Language Models Trained on Code*
   (HumanEval). arXiv:2107.03374, 2021.
   https://huggingface.co/datasets/openai/openai_humaneval
6. Cer et al. *SemEval-2017 Task 1: Semantic Textual Similarity* (STS-B).
   https://huggingface.co/datasets/sentence-transformers/stsb
7. Merity, Xiong, Bradbury, Socher. *Pointer Sentinel Mixture Models*
   (WikiText). arXiv:1609.07843, 2016.
   https://huggingface.co/datasets/Salesforce/wikitext
8. Wang et al. *Super-NaturalInstructions*. arXiv:2204.07705, 2022.
   https://github.com/allenai/natural-instructions
9. Bowman et al. *A large annotated corpus for learning natural language
   inference* (SNLI). 2015. Combined in
   https://huggingface.co/datasets/sentence-transformers/all-nli

## Lab placement (do not fight the pool)

| Work | Where |
|---|---|
| P1-09 implement | 3090 `local/code` (stay loaded; never dual 14B) |
| Later CUDA measure | 5080 `192.168.1.251` exclusive-seq; Comfy stays masked |
| RAG index/retrieve prefer | 1080 Ti guest `192.168.1.243` **if** guest `nvidia-smi` lists the card; no train/PoC CUDA |
| Dataset **download** | Gatekeeper/homelab; 5080 has no WAN |

Horizon training remains blocked until `G-LIFE`. This file is a shopping
list, not a green light.

## Pretrain-before-assembly

Regions **must** pretrain on their own datasets **before** whole-model
train. A mixed dump into an untrained assembly is how a softmax gate
collapses to one expert. Idle regions must not run `activate()`.

This is the same order as [CSD-BRAIN-REGIONS.md](CSD-BRAIN-REGIONS.md).
Do not skip step 1. Do not start at step 4. Do not dual-load 14B.

| Split | When it may enter a recipe |
|---|---|
| **pretrain** | Curriculum (1) only: that region's isolated fit. Others frozen or absent. |
| **later-assembly** | Curriculum (2)–(4): tagged router mix, joint whole-model, or scale. A pretrain row may donate a tagged sample **after** that region can do its job. |
| **eval-only** | Never in any train mix (HumanEval, MBPP test, SciFact, NFCorpus as commercial train). |

Live PoC (`STATUS.md`): two regions (`residual_mlp`, `stream_vae`) +
`SoftmaxRouter` on **synthetic** tokens. Named code / retrieve /
compress experts are Phase 3 after memory-gate `P1-16`. Next closeable
work is still `P1-09`, not a 14B.

## Curriculum

Mandatory order. Autodev implements; hosted Grok does not train.

1. **Per-region pretrain** — each specialist sees **only** its catalog
   `pretrain` rows until it can do its job.
   - `code`: CodeSearchNet Python (license-filtered) +
     `sentence-transformers/codesearchnet`.
   - `retrieve`: FiQA if the pinned revision is permissive.
   - `compress`: STS-B train/val + AllNLI **10k / 1k**.
   - PoC stand-in today: `uv run python -m cogsyndelta.poc.cli train`
     (LatentVAE / `stream_vae`) on synthetic tokens — CPU or 5080
     exclusive-seq. 1080 Ti guest is retrieve-index-light only.
2. **Router** — train `SoftmaxRouter` (+ Switch aux) on a **tagged** mix
   (`region-route` v0) built from already-pretrained catalogs so regions
   activate selectively. `top_k=1` until load is measured non-collapsed
   on a labeled batch of 8+. Drop NC / eval-only rows from the train
   mix. PoC stand-in: `train-route` on synthetic.
3. **Assembled whole-model** — only then joint train of regions + gate
   on the tagged mix. Foundation `common` is still later. Not C4 Switch
   pretrain. Not a 14B.
4. **Scale** — tiny (CPU) → small (5080 CUDA) → medium when measured,
   per [CSD-SCALE-LADDER.md](CSD-SCALE-LADDER.md). Private Hub
   `tzervas/cogsyndelta-<size>` **when** `hf/autodev` exists (mint
   documented above; never copy `gpu/huggingface-token`). WikiText-2,
   Super-NaturalInstructions, The Stack-smol, and C4/Pile/FineWeb wait
   for this step or stay rejected-now.
