# CSD region datasets — tiny experts (not giant pretrain)

Not `STATUS.md`. Catalog of **public** corpora for intended CogSynDelta
cognitive **regions** (code, retrieve, compress, residual, route). Regions
are MoE-adjacent specialists of **one mind**, not a swarm of agents.
Measured PoC numbers stay in `STATUS.md` (PoC-1 LatentVAE; PoC-2 ResidualMLP
+ LatentVAE + softmax top-k; PoC-3 trains the gate). Do **not** claim
mHC, VL-JEPA, quantum, or 10× compression from this file. Do **not**
start `G-TRAIN` bedrock / foundation / 14B. One live PoC region
(LatentVAE or ResidualMLP) may region-pretrain on a pinned public
**train** split with a failing test first.

PoC live modules today are only `residual_mlp` and `stream_vae` on a
**synthetic** `[B, D]` stream (`src/cogsyndelta/poc/`). Named specialists
below are later shopping lists. Next closeable work is **region-pretrain**
of **LatentVAE** on WikiText-2-raw train — not a 14B, not dual 14B.
Never GitHub as a bot. Working branch: `feat/agent-harness`.

Private Hub (when a row actually copies data): `tzervas/cogsyndelta-data`
configs `region-<id>` + `common`, eval in `tzervas/cogsyndelta-eval`.
Requires CSD vault `hf/autodev` (not minted; see mint steps below).
Never copy `gpu/huggingface-token`.

Region map: [CSD-BRAIN-REGIONS.md](CSD-BRAIN-REGIONS.md).
Scale ladder starts at **region-pretrain** (R0):
[CSD-SCALE-LADDER.md](CSD-SCALE-LADDER.md).
Regions **must** pretrain on their own datasets **before** whole-model
train. See **Pretrain-before-assembly** below. Do not skip.

## Why these sets (not The Stack / C4)

PoC-3 already trains a two-region softmax gate on a **synthetic** stream
(`python -m cogsyndelta.poc.cli train-route`). Phase 3 needs **small,
job-shaped** public sets so a tiny expert can learn one job without a
foundation-scale dump.

| Need | Fit | Reject |
|---|---|---|
| Fits a 5080 exclusive-seq smoke / CPU pytest | tens of thousands of rows, not billions of tokens | The Stack 3–6 TB, C4, The Pile, LAION-5B |
| One region learns one job | comment↔code, query↔passage, pair similarity, residual LM, tagged route mix | undifferentiated “all GitHub” |
| License compatible with later private copy | MIT / Apache-2.0 / BSD / CC-BY / CC-BY-SA; per-file filter | GPL dump, **NC as train mix**, unknown commercial |
| Router can **selectively activate** | explicit domain/task tags or disjoint formats | one blob with no gate signal |

Checked against live PoC (`src/cogsyndelta/poc/{regions,router,route,train_route}.py`):
`SoftmaxRouter` is a linear gate → softmax → top-k mix of
`CognitiveRegion.activate`; Switch aux is `N * sum(f_i * P_i)`
(Fedus et al., 2021). That is MoE gating, **not** mHC.

## Training order (data, not weights)

Do not skip. Do not mix `common` into a region that has not learned its
job. Scale **after** the assembled mind works. Matches
[CSD-BRAIN-REGIONS.md](CSD-BRAIN-REGIONS.md) and
[SELF-HOSTED-DRIVE-TARGETS.md](SELF-HOSTED-DRIVE-TARGETS.md).

| Step | Ladder | Who trains | Public mix | Private config (when copied) |
|---|---|---|---|---|
| 1. **Region-pretrain** | **R0** | One specialist at a time | That region's **pretrain** split only | `tzervas/cogsyndelta-data` `region-<id>` → weights `tzervas/cogsyndelta-region-<name>-tiny` |
| 2. **Router / interconnect** | **R1** | Softmax top-k + Switch aux (PoC-3 analog, tagged). Frozen regions, then joint. Not mHC. | Constructed `region-route` from tagged **train** rows. Drop NC. | `region-route` → `tzervas/cogsyndelta-region-route-tiny` |
| 3. **Assembled model (bedrock)** | **R2** | Quantized regions + gate, **no** overall pretrain | Held-out **val** of each specialist; no `common` | `tzervas/cogsyndelta-tiny` (later `tzervas/cogsyndelta-bedrock`) |
| 4. **Scale + foundation** | **R3+** | Same jobs, more rows / params; then overall pretrain | Scale splits below, then `common` + **named** leaks only | `tzervas/cogsyndelta-<size>` then `tzervas/cogsyndelta` |

Tiny-then-scale recipe (first **one specialist**, then router, then the
assembled tiny mind, then more params — never a 14B jump). Authoritative
rungs: [CSD-SCALE-LADDER.md](CSD-SCALE-LADDER.md).

| Ladder | Rows per region | Device | When |
|---|---|---|---|
| R0 tiny region-pretrain | 8k–32k of **that** specialist only | CPU / PoC or 5080 smoke | After `G-LIFE`; PoC analog is `poc.cli train` |
| R1 router (frozen→joint) | Tagged mix 8–32k from **already pretrained** catalogs | CPU or 5080 smoke | After each region can do its job |
| R2 assembled tiny whole-mind | Same tiny tagged mix; no `common` | CPU or 5080 smoke | After R1 load is non-collapsed |
| R3+ larger param counts | Full tiny sets, or 32k–100k, then named scale / `common` | 5080 exclusive-seq | After R2 recon↓ + aux finite |

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
   of `P` vs `region_id` ∈ `{code, retrieve, compress, residual}`
   **plus** aux. Do not drop aux; label-only routing still collapses.
7. **Isolation**: a `region-code` recipe must not silently ingest
   `common` or retrieve qrels. Foundation (`common`) is step 4,
   after bedrock.

Suggested gate prior (not measured):

| Stream cue | Region that should win top-1 |
|---|---|
| Source / identifier / docstring↔body | `code` |
| Query + candidate passage / claim | `retrieve` |
| Embedding or latent to reconstruct / quantize | `compress` |
| Ambiguous / mix / generic language | `residual` |
| Tagged mix (train the **gate**, not a fifth `activate` expert) | `route` labels |

PoC today has two regions (`residual_mlp`, `stream_vae`). Named
code/retrieve/compress/residual experts are Phase 3, **after**
memory-gate `P1-16`.

## Region catalog

Sizes are public cards / papers, not lab downloads. Pin `dataset_id` +
`config` + `revision` before any copy. Filter CodeSearchNet rows to
MIT / Apache-2.0 / BSD at ingest. **NC / academic-ToS sets are eval
only** — they never enter a train mix.

**Split columns:**

- **Pretrain** — region-pretrain only (`region-<id>`). Never HumanEval,
  never test splits, never `common`.
- **Assembly** — router tagged mix, bedrock val, later foundation leak
  (default **no** leak unless named).

### `code` — function + NL, not a corpus LM

**Why this specialist:** one mind still needs a center that maps
docstring ↔ function body and can be exec-checked on tiny Python.
Not next-token over all of GitHub. PoC has no code region yet.

| Dataset | URL | License | Size | Why this specialist | Pretrain | Assembly / eval |
|---|---|---|---|---|---|---|
| CodeSearchNet (Husain et al., 2019) | https://huggingface.co/datasets/code-search-net/code_search_net · paper https://arxiv.org/abs/1909.09436 | Per-repo (`other`); **keep MIT / Apache-2.0 / BSD rows only** | ~2M comment–code pairs; Python zip ~941 MB; HF all-lang train 1.88M / val 89k / test 101k. Python-with-docs ~412k / 23k / 22k (paper) | Docstring↔function is the code **region** job (search + gen) | Tiny: Python + permissive license + docstring present, **sample 8–32k train**. Scale: full filtered Python train. Never test. | Val → bedrock / router tagged `code`. Test → `cogsyndelta-eval` only. No leak into `common` by default. |
| sentence-transformers/codesearchnet | https://huggingface.co/datasets/sentence-transformers/codesearchnet | Same upstream per-file | 1,375,067 pairs, 492 MB | Ready `(comment, code)` pairs for a tiny embedding/code head | Same filter; sample 8–32k for tiny | Same as CodeSearchNet; pair encoder, not a 14B |
| MBPP (Austin et al., 2021) | https://huggingface.co/datasets/google-research-datasets/mbpp · paper https://arxiv.org/abs/2108.07732 | **CC-BY-4.0** | `full`: train **374** / test 500 / val 90 / prompt 10 (~467 kB). `sanitized`: train 120 / test 257 / val 43 / prompt 7 | Tiny exec-supervised Python; fits one pytest + one 5080 smoke | `full` **train** (IDs 601–974) + optional val (511–600). Prompt IDs 1–10 are few-shot, not SGD | **Never** test (IDs 11–510). Copy test to `cogsyndelta-eval`. Router tag `code` on train prompts only |
| HumanEval (Chen et al., 2021) | https://huggingface.co/datasets/openai/openai_humaneval · paper https://arxiv.org/abs/2107.03374 | **MIT** | 164 tasks, test split only | Held-out functional eval | **Do not train** | Eval only. Router must not see HumanEval prompts at train time |
| DS-1000 (Lai et al., 2023) | https://huggingface.co/datasets/xlangai/DS-1000 · https://github.com/xlang-ai/DS-1000 | **CC-BY-SA-4.0** | 1,000 data-science problems, `test` only, ~3.4 MB | Later data-science code eval (NumPy/Pandas), still tiny | **Do not train** | Eval only after MBPP/HumanEval; never in `region-code` train |

**Do not start with** `bigcode/the-stack` (3–6 TB, gated ToU). Optional
later sample: `bigcode/the-stack-smol` (10k files/language, 300k rows,
~2.6–3 GB text) **only** after per-file license filter; still larger
than MBPP/CodeSearchNet-Python for a first expert.

### `retrieve` — query/passage + qrels, not MS MARCO-scale

**Why this specialist:** query + candidate passage is a different cue
from source/docstring. Memory-gate P1-09/P1-15 is the Python retrieve
**backend**; this row is a later CSD region **head**. Tiny Wikipedia
QA / claim sets, not 8.8M MS MARCO passages.

NC / ToS corpora stay **eval/research only** (see rejected + SciFact
row). Train mix is CC-BY / CC-BY-SA / MIT / Apache only.

| Dataset | URL | License | Size | Why this specialist | Pretrain | Assembly / eval |
|---|---|---|---|---|---|---|
| SQuAD 1.1 (Rajpurkar et al., 2016) | https://huggingface.co/datasets/rajpurkar/squad · homepage https://rajpurkar.github.io/SQuAD-explorer/ · paper https://arxiv.org/abs/1606.05250 | **CC-BY-SA-4.0** | train **87,599** / validation **10,570**; ~16–35 MB | Wikipedia question → paragraph span. CPU-rankable query/passage pairs; canonical tiny retrieve job | Tiny: `train[:8000]` (or 8–32k). Scale: full train 87,599 | Validation → eval / bedrock. Hidden test is not public. Router tag `retrieve`. Named leak into `common` only if the foundation spec lists SQuAD |
| HotpotQA (Yang et al., 2018) | https://huggingface.co/datasets/hotpotqa/hotpot_qa · https://hotpotqa.github.io · paper https://arxiv.org/abs/1809.09600 | **CC-BY-SA-4.0** | ~113k QA; HF ~1.27 GB download. BEIR fullwiki corpus 5.23M docs is **too big** for the first expert | Multi-hop question + supporting Wikipedia paragraphs; still CC-BY-SA | Tiny: `distractor` **train sample 8k** (use gold + supporting contexts, not the 5.23M dump). Scale: full distractor train (~90k) | Dev distractor → eval. Do not pull BEIR `hotpotqa` 5.23M corpus into `region-retrieve`. Foundation `common` may add a **named** sample later |
| FEVER (Thorne et al., 2018) | https://huggingface.co/datasets/fever/fever · https://fever.ai · paper ACL 2018 | **CC-BY-SA-3.0** (Wikipedia-derived claims; pin card, ignore spurious `gpl-3.0` on the Hub **code** tag) | 185,445 claims; train / labelled_dev / paper_dev / paper_test | Claim → Wikipedia evidence retrieval (fact-checking cue, still `retrieve`) | Tiny: train sample 8k Supported/Refuted with evidence. Scale: full train minus NotEnoughInfo if desired | `paper_test` / labelled_dev → eval. Do not mix into `code` |
| BEIR ArguAna (Stab & Gurevych / BEIR) | https://huggingface.co/datasets/BeIR/arguana · zip https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip | **CC-BY-SA-4.0** (BIRCO / Hub card; original UKP argument corpus) | **test only**: 1,406 queries, 8.67k docs, 1.0 rel/q | Counterargument retrieval; tiny enough for CPU | **No train split** — not region-pretrain | Eval / P1-15 analog. Router must not train on these queries |
| BEIR SciFact (Wadden et al.) | https://huggingface.co/datasets/BeIR/scifact · https://huggingface.co/datasets/allenai/scifact | **CC-BY-NC-2.0** (NC) | ~300 test queries, ~5k docs | Scientific claim→abstract | **Never train** | Eval/research only until a non-NC twin exists |
| BEIR FiQA-2018 | https://huggingface.co/datasets/BeIR/fiqa · original https://sites.google.com/view/fiqa/ | Original: **non-commercial** (source page). Hub `cc-by-sa-4.0` is a **redistribution mismatch** — do not trust it for train | 648 test queries, 57k docs | Finance QA | **Never train** (NC source) | Eval/research only |
| BEIR NFCorpus (Boteva et al., 2016) | https://huggingface.co/datasets/BeIR/nfcorpus | Academic / NutritionFacts ToS | 323 test queries, 3.6k docs, ~30 MB | Small medical IR | **Never train** | Eval/research only |

BEIR overview: Thakur et al., 2021
(https://github.com/beir-cellar/beir, Apache-2.0 **toolkit**; datasets
keep their own licenses).

**Too big for a tiny retrieve expert:** MS MARCO (8.84M passages /
530k train queries), Natural Questions (2.68M docs), HotpotQA
**fullwiki 5.23M**. Those wait for foundation `common`, not
`region-retrieve`.

P1-09 (now): in-memory oracle, mandatory/global domain, deterministic
rank — **no Hub download**. This catalog is the later P1-15 / Phase 3
source list.

### `compress` — fidelity of a short latent, not 10× marketing

**Why this specialist:** neighbors must stay neighbors after a short
latent / 8-bit compact. PoC-1/3 already measure LatentVAE + calibrated
8-bit on **synthetic** streams. Honest ratios in `STATUS.md` are
~1.3–2.7× stored-byte (original/stored), **not** 10×. Use small
**semantic** sets, not ImageNet-scale codebooks.

| Dataset | URL | License | Size | Why this specialist | Pretrain | Assembly / eval |
|---|---|---|---|---|---|---|
| SNLI (Bowman et al., 2015) | https://huggingface.co/datasets/stanfordnlp/snli · https://nlp.stanford.edu/projects/snli/ · paper https://arxiv.org/abs/1508.05326 | **CC-BY-SA-4.0** | train **550,152** / val 10,000 / test 10,000; ~20 MB download | Premise/hypothesis labels (entail / contradict / neutral) → contrastive reconstruction: entailments stay close, contradictions far | Tiny: **10k train / 1k val** triplets (entail = positive, contradict = negative). Scale: full train | Test 10k → eval Spearman/accuracy of compressed cosine. Router tag `compress`. Default no `common` leak |
| PAWS-Wiki (Zhang et al., 2019) | https://huggingface.co/datasets/google-research-datasets/paws (`labeled_final`) · https://github.com/google-research-datasets/paws · paper https://arxiv.org/abs/1904.01130 | Google: **free use** (acknowledgement requested) + Wikipedia **CC-BY-SA**. **Do not** use PAWS-QQP (Quora ToS) | ~49k train / 8k dev / 8k test labeled Wiki pairs | Paraphrase vs scrambled non-paraphrase; fidelity of compressed neighbors on hard word-order pairs | Tiny: train 8–16k Wiki pairs. Scale: full `labeled_final` train | Dev/test → eval. Skip QQP subset |
| sentence-transformers/stsb (Cer et al., 2017 STS-B) | https://huggingface.co/datasets/sentence-transformers/stsb · SemEval-2017 Task 1 https://arxiv.org/abs/1708.00055 | SemEval / **research redistribution**; pin HF revision. Prefer SNLI/PAWS for **train** | 8,628 pairs (train 5,749 / val 1,500 / test 1,379), 725 kB | Continuous similarity (0–1). Spearman of cosine(compressed) vs gold is the **fidelity metric** | Optional tiny train if operator accepts SemEval terms; otherwise **eval only** | Test 1,379 is the compress-region metric. Never route to `code` |
| sentence-transformers/all-nli triplet (SNLI + MultiNLI) | https://huggingface.co/datasets/sentence-transformers/all-nli | SNLI **CC-BY-SA-4.0**; MultiNLI OANC-permissive + some CC-BY/CC-BY-SA fiction | Full card ~2.86M rows / 213 MB | Ready triplets. **Second** mix after SNLI-only 10k is measured | Tiny: `triplet` **train[:10000]** / `dev[:1000]`. Scale: 100k, not 2.8M first | Dev/test → eval. Full 2.8M is not the first mix |
| WikiText-2 raw (Merity et al., 2016) | https://huggingface.co/datasets/Salesforce/wikitext config `wikitext-2-raw-v1` · paper https://arxiv.org/abs/1609.07843 | **CC-BY-SA-4.0** (Wikipedia; Hub also notes CC-BY-SA-3.0 / GFDL upstream) | 36,718 train / 3,760 val / 4,358 test lines; ~5 MB | Optional **line reconstruction** for the VAE path | Prefer SNLI/PAWS for embedding fidelity. If used: train lines only | Val/test recon. Prefer `residual` for generic Wiki LM (see below) |

Fashion-MNIST (MIT, 60k×784) is a **vision VAE** smoke only; CSD stream
is text/latent first. Do not pull LAION-5B or ImageNet-1k as first mix.

### `residual` — identity / generic language path (scaled ResidualMLP)

**Why this specialist:** softmax needs a real expert that is **not**
code, retrieve, or compress. PoC `residual_mlp` is the analog (stream +
GELU MLP, same dim in/out). Intended job: keep a residual path alive
for ambiguous / mix tokens so specialists do not eat the whole batch.
Not a foundation LM. Not mHC.

| Dataset | URL | License | Size | Why this specialist | Pretrain | Assembly / eval |
|---|---|---|---|---|---|---|
| WikiText-2 raw | https://huggingface.co/datasets/Salesforce/wikitext `wikitext-2-raw-v1` · https://arxiv.org/abs/1609.07843 | **CC-BY-SA-4.0** | 36,718 / 3,760 / 4,358 lines; ~5 MB download | Tiny Wikipedia LM so the residual expert learns generic language without code/query cues | **All train lines** (tiny rung). Identity-preserving: reconstruct or residual-block the encoded line | Val → bedrock. Test → eval perplexity / recon. Router tag `residual` on untagged Wiki lines. Named leak into `common` is natural (same domain) — still list it |
| WikiText-103 raw | same card, config `wikitext-103-raw-v1` | **CC-BY-SA-4.0** | 1,801,350 train / 3,760 val / 4,358 test; ~190 MB download | Scale residual after WikiText-2 loss falls | Scale rung only. Do not start here | Same val/test articles as WT-2 (60+60). Do not dump into `region-code` |
| CNN/DailyMail v1.0.0 (See et al., 2017; Hermann et al.) | https://huggingface.co/datasets/abisee/cnn_dailymail · https://huggingface.co/datasets/ccdv/cnn_dailymail · paper https://arxiv.org/abs/1704.04368 | **Apache-2.0** (v1.0.0 card) | ~287k train / 13k val / 11k test articles; ~300–534 MB | Optional later: news article as generic residual stream (highlights are **not** the compress job) | Tiny: **sample 8k** articles (body only). Scale: full train | Val/test → eval. Do not treat highlights as a summarization region (not in the intended set) |

TinyStories (`roneneldan/TinyStories`) is **CDLA-Sharing-1.0**, not
MIT/Apache/CC — **out** of the first mix.

### `route` — tagged mix, not C4 Switch pretrain

**Why this specialist:** there is no honest public “router dataset” at
tiny scale. Switch Transformers pre-trained on **C4** (Fedus et al.,
2021) — giant pretrain, out of scope. `route` is **not** a fourth
`activate()` expert; it is the **gate** over the four specialists.
Build `region-route` from the catalogs above.

| Mix | URL | License | Size | Why this specialist | Pretrain | Assembly / eval |
|---|---|---|---|---|---|---|
| **CSD `region-route` v0** (construct; do not train this week) | Derived from rows above; pin each source revision | Intersection of source licenses; **drop NC / FiQA / NFCorpus / SciFact from train** | ~8–32k tagged rows, balanced: MBPP+CodeSearchNet sample, SQuAD (and optional FEVER/Hotpot 8k), SNLI-10k + PAWS, WikiText-2 lines | Teaches the linear gate the four cues in the table above | This **is** step 2 (router train), after each region has a tiny pretrain | Loss = CE(`region_id`) + `aux_coef * Switch_aux`. Target: load not `{1,0,0,0}` on a batch of 8. Bedrock uses the same tags, no `common` |
| Super-NaturalInstructions sample (Wang et al., 2022) | https://github.com/allenai/natural-instructions · paper https://arxiv.org/abs/2204.07705 | Task **defs** **Apache-2.0**; **instances inherit the original dataset license** — keep only MIT/Apache/CC instance tasks | 1,616 tasks; take a few hundred defs + 1–3 examples each | Optional later: task-type routing (QA vs code vs NLI vs generic) | After v0 mix is non-collapsed | Map task category → `{code, retrieve, compress, residual}`; still top-k + aux. Do not ingest NC instance licenses |

PoC-3 already proves recon↓ and aux finite on **synthetic** tokens
(`STATUS.md`). A labeled-stream unit test is the first *CSD* router
increment **after** P1-09 — still not a 14B.

## Rejected for tiny experts

| Corpus | Why rejected now |
|---|---|
| The Stack / StarCoderData / full CodeSearchNet-all without license filter | Giant and/or copyleft mix |
| C4, The Pile, FineWeb | Foundation `common` only, Phase 3 step 4 |
| MS MARCO / NQ / Hotpot **fullwiki 5.23M** | Retrieve foundation, not a 5k–90k-doc expert |
| LAION-5B, ImageNet-1k as first compress mix | Wrong modality/scale |
| HumanEval, MBPP **test**, DS-1000, SQuAD hidden test, SciFact test | Eval; training on them is leakage |
| SciFact / FiQA / NFCorpus as **train** | NC / non-commercial source / academic ToS |
| TriviaQA | UW does **not** own copyright; Hub Apache tag is a mismatch |
| TinyStories | CDLA-Sharing-1.0, not MIT/Apache/CC |
| PAWS-QQP | Quora license; Wiki subset only |
| Dual 14B GGUF / GitHub.com dumps as train | Lab / forge policy, not a dataset |

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

**`region-pretrain`.** Pretrain **one existing region** — LatentVAE
(`stream_vae` / `train_latent_vae`) — on a real public **train** split,
failing pytest first. Not whole-model 14B. Not `G-TRAIN` bedrock /
foundation. ResidualMLP is the fallback (no isolated train loop yet).

Pinned split: `Salesforce/wikitext` config `wikitext-2-raw-v1` **train**
(36,718 lines, ~7.75 MB). Revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. Hub license `cc-by-sa-3.0`
+ `gfdl` (Wikipedia-derived). Never test split. Never WikiText-103.
SNLI-10k waits until a pair encoder exists.

Autodev tick: **one failing pytest** in this CogSynDelta tree
(`feat/agent-harness`): `tests/test_poc_region_pretrain.py`. Pack:
[CONTEXT-PACK-REGION-PRETRAIN.md](CONTEXT-PACK-REGION-PRETRAIN.md).
Worktree is CogSynDelta, **not** `memory-gate-wt-p1-09`. P1-09 stays
open on the Phase 1 board but is **not** this steer. Not chroma P1-05.

5080 exclusive-seq only if tiny CUDA is required later. 1080 Ti guest
`192.168.1.243` is retrieve-index-light **and live** (`nvidia-smi` GTX
1080 Ti `GPU-4df3ba11-fd12-3550-bb97-ad00b0b00569`); do **not** ham
CUDA/train there. 3090 `local/code` implements; LocalAI stays loaded.

## Citations (public sources)

1. Husain, Wu, Gazit, Allamanis, Brockschmidt. *CodeSearchNet Challenge:
   Evaluating the State of Semantic Code Search*. arXiv:1909.09436, 2019.
   https://arxiv.org/abs/1909.09436 ·
   https://huggingface.co/datasets/code-search-net/code_search_net
2. Austin et al. *Program Synthesis with Large Language Models*
   (MBPP). arXiv:2108.07732, 2021.
   https://huggingface.co/datasets/google-research-datasets/mbpp
3. Chen et al. *Evaluating Large Language Models Trained on Code*
   (HumanEval). arXiv:2107.03374, 2021.
   https://huggingface.co/datasets/openai/openai_humaneval
4. Lai et al. *DS-1000: A Natural and Reliable Benchmark for Data Science
   Code Generation*. ICML 2023. arXiv:2211.11501.
   https://huggingface.co/datasets/xlangai/DS-1000
5. Rajpurkar, Zhang, Lopyrev, Liang. *SQuAD: 100,000+ Questions for
   Machine Comprehension of Text*. EMNLP 2016. arXiv:1606.05250.
   https://huggingface.co/datasets/rajpurkar/squad
   (CC-BY-SA-4.0; train 87,599 / validation 10,570)
6. Yang, Qi, Zhang, Bengio, Cohen, Salakhutdinov, Manning. *HotpotQA: A
   Dataset for Diverse, Explainable Multi-hop Question Answering*. EMNLP
   2018. https://arxiv.org/abs/1809.09600 ·
   https://huggingface.co/datasets/hotpotqa/hotpot_qa (CC-BY-SA-4.0)
7. Thorne, Vlachos, Christodoulopoulos, Mittal. *FEVER: a Large-scale
   Dataset for Fact Extraction and VERification*. NAACL 2018.
   https://huggingface.co/datasets/fever/fever (CC-BY-SA-3.0)
8. Thakur, Reimers, Rücklé, Srivastava, Gurevych. *BEIR: A Heterogeneous
   Benchmark for Zero-shot Evaluation of Information Retrieval Models*.
   NeurIPS Datasets and Benchmarks, 2021. https://arxiv.org/abs/2104.08663 ·
   https://github.com/beir-cellar/beir
9. Bowman, Angeli, Potts, Manning. *A large annotated corpus for learning
   natural language inference* (SNLI). EMNLP 2015. arXiv:1508.05326.
   https://huggingface.co/datasets/stanfordnlp/snli (CC-BY-SA-4.0;
   550,152 / 10,000 / 10,000)
10. Zhang, Baldridge, He. *PAWS: Paraphrase Adversaries from Word
    Scrambling*. NAACL 2019. arXiv:1904.01130.
    https://huggingface.co/datasets/google-research-datasets/paws
11. Cer et al. *SemEval-2017 Task 1: Semantic Textual Similarity* (STS-B).
    https://arxiv.org/abs/1708.00055 ·
    https://huggingface.co/datasets/sentence-transformers/stsb
12. Merity, Xiong, Bradbury, Socher. *Pointer Sentinel Mixture Models*
    (WikiText). arXiv:1609.07843, 2016.
    https://huggingface.co/datasets/Salesforce/wikitext
13. See, Liu, Manning. *Get To The Point: Summarization with
    Pointer-Generator Networks* (CNN/DailyMail). ACL 2017.
    arXiv:1704.04368. https://huggingface.co/datasets/abisee/cnn_dailymail
    (Apache-2.0)
14. Wang et al. *Super-NaturalInstructions*. EMNLP 2022.
    arXiv:2204.07705. https://github.com/allenai/natural-instructions
    (task defs Apache-2.0; instances keep source licenses)
15. Fedus, Zoph, Shazeer. *Switch Transformers: Scaling to Trillion
    Parameter Models with Simple and Efficient Sparsity*.
    arXiv:2101.03961, 2021. https://arxiv.org/abs/2101.03961

## Lab placement (do not fight the pool)

| Work | Where |
|---|---|
| Region-pretrain implement (this steer) | 3090 `local/code` (stay loaded; never dual 14B) |
| P1-09 memory-gate (not this steer) | 3090 `local/code`; worktree `memory-gate-wt-p1-09` |
| Later CUDA measure | 5080 `192.168.1.251` exclusive-seq; Comfy stays masked |
| RAG index/retrieve prefer | 1080 Ti guest `192.168.1.243` **live**; no train/PoC CUDA |
| Dataset **download** | Gatekeeper/homelab; 5080 has no WAN |

Horizon training remains blocked until `G-LIFE`. This file is a shopping
list, not a green light.

## Pretrain-before-assembly

Regions are specialized **submodels of one CSD mind**. They **must**
pretrain on their own datasets **before** the whole model trains. A
mixed dump into an untrained assembly is how a softmax gate collapses
to one expert. Idle regions must not run `activate()`.

This is the same order as [CSD-BRAIN-REGIONS.md](CSD-BRAIN-REGIONS.md)
and rungs R0→R3+ in [CSD-SCALE-LADDER.md](CSD-SCALE-LADDER.md). Do not
skip R0. Do not start at R3. Do not dual-load 14B. Not mHC / VL-JEPA /
10×. Autodev implements. Hosted Grok does not train.

Live PoC (`STATUS.md` 2026-08-19): two regions (`residual_mlp`,
`stream_vae`) + `SoftmaxRouter` on **synthetic** tokens. Named
code / retrieve / compress / residual experts remain Phase 3 shopping
lists after memory-gate `P1-16`. **Not** mHC / VL-JEPA / 10×.
**Not** 14B assembly. Next closeable CSD work is **region-pretrain**
of one live PoC region (see **Next autodev tick** below).

| Split | When it may enter a recipe |
|---|---|
| **pretrain** | Curriculum (1) / **R0** only: that region's isolated fit. Others frozen or absent. |
| **later-assembly** | Curriculum (2)–(4) / **R1–R3+**: tagged router mix, joint whole-model, or scale. A pretrain row may donate a tagged sample **after** that region can do its job. |
| **eval-only** | Never in any train mix (HumanEval, MBPP test, DS-1000, SciFact, FiQA, NFCorpus as commercial train, SQuAD hidden test). |

Private Hub (when `hf/autodev` exists; mint documented above; never copy
`gpu/huggingface-token`): region checkpoints
`tzervas/cogsyndelta-region-<name>-<size>` **then** assembled
`tzervas/cogsyndelta-<size>`. Data stays in `tzervas/cogsyndelta-data`
configs `region-<id>` / `common`.

### Curriculum order (do not delete the catalog)

Mandatory. Catalog rows above stay cited; this is only the **order**.

1. **R0 — per-region pretrain** — each specialist sees **only** its
   catalog **pretrain** rows until it can do its job.
   - `code`: CodeSearchNet Python (license-filtered, 8–32k) +
     `sentence-transformers/codesearchnet`. MBPP `full` train 374 is a
     later smoke, **not** a substitute. HumanEval / DS-1000 / MBPP test
     stay eval-only.
   - `retrieve`: SQuAD 1.1 `train[:8000]` (CC-BY-SA). Optional tiny:
     FEVER 8k, HotpotQA distractor 8k (not the 5.23M fullwiki dump).
     SciFact / FiQA / NFCorpus / ArguAna stay **eval-only**.
   - `compress`: SNLI 10k/1k, then PAWS-Wiki 8–16k. STS-B is the
     fidelity **metric** (eval; optional train only if SemEval terms
     allow). AllNLI-10k is the second mix. WikiText-2 recon is later.
   - `residual`: WikiText-2 raw train lines. CNN/DailyMail 8k body-only
     is later. WikiText-103 is R3+ scale only.
   - PoC stand-in today: `uv run python -m cogsyndelta.poc.cli train`
     (LatentVAE / `stream_vae`) on synthetic tokens — CPU or 5080
     exclusive-seq smoke. 1080 Ti guest is retrieve-index-light only.
2. **R1 — router / interconnect** — train `SoftmaxRouter` (+ Switch
   aux) on a **tagged** mix (`region-route` v0) built from
   **already-pretrained** catalogs so regions activate selectively.
   Frozen regions first (`train-route --gate-only`), then joint.
   `top_k=1` until load is measured non-collapsed on a labeled batch
   of 8+. Drop NC / eval-only rows. Super-NaturalInstructions is
   optional **after** v0. Interconnect / mHC is later, not this rung.
   PoC stand-in: `train-route` on synthetic.
3. **R2 — assembled tiny whole-mind** — only then joint train of
   regions + gate on the tagged mix (bedrock analog). Foundation
   `common` is still later. Not C4 Switch pretrain. Not a 14B.
4. **R3+ — scale** — same jobs, larger param counts / more rows, then
   overall pretrain. WikiText-103, full SQuAD train, AllNLI 100k, The
   Stack-smol (license-filtered), C4/Pile/FineWeb wait here or stay
   rejected-now. Private Hub `tzervas/cogsyndelta-<size>` **when**
   `hf/autodev` exists.

### Next autodev tick (R0 analog — failing test first)

Steer `next_goal` = **region-pretrain**. Autodev implements on
`feat/agent-harness`. Hosted Grok does not write the trainer. This is
**not** R2 assembly and **not** a 14B.

| Field | Value |
|---|---|
| Region | **LatentVAE** (`stream_vae` / `train_latent_vae`). ResidualMLP fallback (no isolated `poc.cli train` loop yet) |
| Matching public split | `Salesforce/wikitext` config `wikitext-2-raw-v1` **train** (36,718 lines). Residual catalog pretrain. Never test. Never WikiText-103 |
| Revision | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| This tick | **Failing** pytest `tests/test_poc_region_pretrain.py` importing `cogsyndelta.poc.train.train_latent_vae_on_public_split` (does not exist). CPU, seed 42, ≥20 steps, batch 8. Assert `last_loss < first_loss` and `split == "train"` |
| Not this tick | Trainer impl, SNLI-10k, router/`train-route` assembly, 14B, `G-TRAIN`, dual 14B, GitHub |
| Worktree | CogSynDelta `feat/agent-harness`. Never `memory-gate-wt-p1-09`. Never `kang-main-wip` |
| GPU | 3090 `local/code` (LocalAI stays). 5080 exclusive-seq later CUDA. 1080 Ti guest retrieve-index-light only |
| Hub | `hf/autodev` not minted — operator mint documented above. Never copy `gpu/huggingface-token` |

Pack: [CONTEXT-PACK-REGION-PRETRAIN.md](CONTEXT-PACK-REGION-PRETRAIN.md).
Catalog rows above stay cited; this tick only **orders** the first live
PoC region onto its matching public **train** split.
