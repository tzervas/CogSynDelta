# CogSynDelta: state versus plan, 2026-09-06

Synthesised from six read-only surveys (regions, composed, features, experiments, corpus, scale) of `main @ 3bc8c3e`, the matrix dir `/akula-data/csd/matrix/`, and a `csd-corpus-expand.py` dry run. Where surveys disagree the disagreement is recorded in §8, not resolved here.

## 1. Summary

**Done.** The per-region pipeline is real and proven: five regions have trained matrix cells with receipts, the guard and metric machinery (split manifests, G26 fail-closed guards, lexical and untrained baselines, corpus fingerprints, card rendering) has tests that demonstrate the guards fire, and the gating experiments W1, W1d, W2c, W4-control-arm, g22 and g48-E1 carry verdicts (`docs/design/evidence/`, Table 1).

**Missing.** Nothing of the composed model exists as code. There is no faculty protocol, no thalamic controller or workspace, no episodic store, no compose trainer, no composed evals, and no test file for any of them (`find tests -iname "*interconnect*|*workspace*|*episodic*|*compose*"` → none). The `memory` region, a v1 participant, has no matrix cell and its corpus is marked not on disk (`config/mind/csd-regions.json:112`). `affect` and a learned router are design-only.

**Single biggest blocker to composed training.** The plan's own prerequisite chain has not reached the region-set freeze (W2b): `memory` has no matrix checkpoint and its token-aware retrain W4 is "BLOCKED ON OD-17", an operator decision on the gate's `>` versus `≥` and pivot-versus-amend (`REGION-TAXONOMY-AND-INTERCONNECT.md:589`). Even with a frozen set there is no interconnect code to train, so the two blockers are serial: decide OD-17 and train `memory`, and in parallel build the interconnect.

**Table 1: headline counts.**

| quantity | value | source |
|---|---|---|
| regions declared in config | 10 | `config/mind/csd-regions.json` |
| regions with a trained matrix cell | 5 (code, compress, retrieve, reason, visual) | `/akula-data/csd/matrix/` |
| regions with train+quant+eval+eval-quantized receipts | 4 (visual is train-only in the matrix dir) | matrix `receipts/` |
| v1 composed participants per plan | 5 (language, memory, reasoning, visual, episodic_store) | taxonomy DEC-16/DEC-49 |
| v1 participants with a matrix checkpoint | 2 (language-as-`code`, visual); reasoning's cell is KILLed by g48-E1 | §2, §5 |
| composed-model modules with code | 0 (legacy `core/` stack superseded per DEC-12) | composed survey |
| tests covering composed components | 0 | composed survey |
| gating experiments with a verdict | 6 (W1, W1d, W2c, W4-control, g22 ×2, g48-E1) | §5 |
| interconnect rows still `todo` | W0, W2b, W3, W5, W5b, W6, W8, W9, W10, E0, E1, E2 | taxonomy §4 table |
| operator decisions holding the chain | OD-17 (W4 gate), OD-4 (W7v corpus), six episodic-store contract gaps | taxonomy `:589`, `:604`, `:4853` |

## 2. Regions

Each region's code, tests, trained cells and corpus were checked independently by the regions, corpus and scale surveys; the status column is the join.

**Table 2: region status (cells under `/akula-data/csd/matrix/`, config `config/mind/csd-regions.json`).**

| region | plan role | module | tests | latest matrix cell | receipts in matrix | corpus (licence) | status |
|---|---|---|---|---|---|---|---|
| residual_mlp | residual update on the shared stream (`csd-regions.json:11-25`) | `src/cogsyndelta/poc/regions.py:11` (PoC) | none by name | none | none | synthetic only | code-only; config says `live: true` |
| stream_vae | encode/reparameterise/decode (`:26-40`) | none in `regions/`; PoC `poc/vae.py`, `core/pcn_vae_gan.py` | none | none | none | synthetic only | code-only, PoC tier |
| language (alias `code`) | language centre, docstring↔function (`:41-56`) | `regions/pretrain.py` via `region_spec`; `regions/aliases.py` | alias + spec tests | `code-b1280-s{0,1}-7bc2699-20260904` | train, quant, eval, eval-quantized | codesearchnet-python, licence unresolved | trained; cells still named `code` |
| retrieve | query/claim→passage (`:57-73`) | `regions/pretrain.py` | spec tests | `retrieve-b1280-s{0,1}-7bc2699-20260904` | full set | fiqa + NQ + gooaq, NC tier | trained; config says `merged_into: memory` (§8) |
| compress | neighbours stay neighbours (`:74-90`) | `regions/pretrain.py`, `regions/compress.py` | `test_region_compress.py` | `compress-b1280-s{0,1}-7bc2699-20260904` | full set | all-nli, BLOCKING (SNLI CC BY-SA) | trained; config says `merged_into: memory` (§8) |
| memory | hippocampus, DEC-02 union of compress+retrieve (`:91-107`) | `regions/memory.py` | `test_region_memory.py`, `test_region_memory_beir.py` | none | none | not on disk (`available: false`) | code-only; v1 participant; W4 ran it outside the matrix (§8) |
| visual (alias `vl_latent`) | I-JEPA EMA target encoder (`:108-124`) | `regions/vl_pretrain.py` | 10 test files | `visual-b128-s{0,1}-3ce18db-20260905` | train only | visual-clean-v1, PERMISSIVE_OK | trained; eval/quant absent from matrix dir (§8) |
| reason | question↔derivation retrieval, not step generation (`:125-141`) | `regions/pretrain.py` | spec tests | `reason-b512-s{0,1}-7bc2699-20260904` | full set + lexical probe | gsm8k + aqua_rat, dated waiver | trained; g48-E1 verdict KILL (§5) |
| classify_banking77 | 77-way intent head (`:142-158`) | `regions/classify_pretrain.py` | `test_classify_pretrain.py` | none | none | banking77 CC BY, waiver | code-only |
| classify_go_emotions | 28-label multi-label head (`:159-175`) | same module | same test | none | none | go_emotions Apache, waiver | code-only |
| affect | separate z_affect faculty (memory `csd-affect-isolation-contract.md`) | none (`grep z_affect src/` → none) | none | none | none | none | design-only; not in config |
| episodic_store | v1 participant, DEC-49 (taxonomy `:76-90`) | none (`grep EpisodicStore` → none) | none | n/a | none | n/a | design-only |
| interconnect (white matter) | thalamic controller + workspace, DEC-16 (taxonomy `:1217-1391`) | none current; legacy `core/interconnect_manager.py` | `test_mhc.py` (legacy only) | n/a | none | n/a | missing; legacy superseded per DEC-12 |
| router | learned `router_trigger` (`contracts/region_spec.py:90-93`) | string field only | `test_region_spec.py` | n/a | n/a | n/a | declared surface, no model |

## 3. Missing modules

Each entry states what the plan says the module is, then what exists. Sizes the plan commits to are in Table 3.

**Faculty protocol and W0 split.** The plan requires every region to expose `tokens()` (pre-pool position latents) and `pool()` (post-pool vector), with "tokens" always meaning latents (taxonomy `:1087-1217`, DEC-14/15/47). No `Protocol` exists under `src/cogsyndelta`; `TextEncoder`/`ViTEncoder` have a pooling point at `regions/text_encoder.py:130-150` but row W0 (split plus parameter-table re-instantiation) is `todo`.

**Thalamic controller, workspace, adapters, K/V bank, frontal read-out.** The plan's interconnect is a controller emitting context and read budgets, an admission matrix and a halt signal; top-k region tokens pass through per-region adapters into a K/V bank; workspace latents iterate cross-attention, self-attention and MLP; attention weights are the connection strengths; a frontal read-out chooses output modality (taxonomy `:1217-1391`, `:2113-2163`). No module named interconnect, workspace, thalam* or schedule exists; the legacy `core/interconnect_manager.py` and `integrated_system.py` are non-differentiable (`.item()` in `compute_importance`/`allocate_bandwidth`) and superseded per DEC-12 (`:1026`).

**Episodic store.** A non-parametric store with two projections `W_k`, `W_v` inside white matter and a floored read budget, reinstated as a v1 participant by DEC-49 (taxonomy §8, `:4853`). `src/cogsyndelta/memory/*` holds active-memory and persistence code but nothing named episodic_store and no `W_k`/`W_v`; six contract gaps await operator deliberation.

**Compose trainer and composed evals.** The plan trains the interconnect with regions frozen, then runs whole-mind training (DEC-50 three-step protocol, taxonomy §2.6-2.7). `scripts/csd-train-all.py` trains per region only and merely reserves corpus for a future `compose` consumer (`:159`, `:170`, `:240-242`, `:825`); `pipeline/receipt.py` lists a `"compose"` stage that nothing writes (`src/cogsyndelta/pipeline/receipt.py:78`). `cards/templates/composed.md.j2` exists and says whole-mind training has not run.

**Memory region checkpoint.** Not a module but a missing artifact: `regions/memory.py` is implemented and tested, yet no `memory-*` cell exists in the matrix and the config marks the corpus absent (`csd-regions.json:112`); the matrix planning table lists memory cell ids with every metric `n/a` (`matrix/tables/csd-m1-20260904/matrix.md`).

**Affect faculty and learned router.** Affect is a required separate faculty with tagged `z_affect`, memory-velocity learning and a leak guard (Grok S11; gates G0-G5 explicitly unrun, `g6-ternary-memory-gate-2026-09-04/README.md:22` — untracked evidence directory on the main checkout, not in git). The router is a per-region string field described as "learned, not hand-written" with no learning code (`region_spec.py:90-93`).

**stream_vae production module.** Declared in config with a compress-analog role; only PoC-tier VAE code exists and nothing binds to the region name.

**Overlap-check gate (P2.5a) and manifest enforcement.** P2.5a is "unstarted" (`TRAINING-SUPERSET.md:462`) while the current near-dup measure misses its target by a wide margin (`:442`); the manifest `status` schema lives only in markdown (`MODEL-MANIFESTS.md:557`) with no validating module found in `src/`.

**Table 3: sizes the plan commits to for the missing modules.**

| item | planned value | source |
|---|---|---|
| white matter parameters | 27,424,039 (`[I]`, not re-instantiated) | taxonomy `:1307` |
| white matter share of composed mind | 31.8% of ~86.3M | taxonomy `:1310` |
| workspace latents | `z ∈ [B,64,512]`, `n_iter` 4 | taxonomy §2.3 |
| episodic projections | `W_k`, `W_v` 512×512 = 524,288 params | taxonomy §8 |
| episodic read-budget floor | ≈8 tokens (`η/R·B_read`) | taxonomy §8 |
| write-back gate (W5b) | ≤1 pt own-bin drop | DEC-17 |
| topology agreement (W9) | ≥95% between admission matrix and measured Ĉ | DEC-18 |
| E2 attention-mass floor | ≥3.0% | taxonomy §4.1 |

## 4. Planned features and proof status

"Proven" means a test or evidence directory demonstrates the behaviour, not that a doc claims it.

**Table 4: feature proof status.**

| feature | planned | implemented | tested | evidence | status |
|---|---|---|---|---|---|
| matrix harness train→test→quantize→publish→verify | `MATRIX-PIPELINE.md:1-8` | CSD adapters only; harness is `tzervas/model-matrix` | `test_matrix_config.py` | none in repo | partial (out of scope here) |
| `promote_main: auto` | `MATRIX-PIPELINE.md:122-124` | refuses to load without margin/seed count | none | none | missing by design |
| quantized artifact vs in-memory plan agreement | `MATRIX-PIPELINE.md:79-85` | `csd-quantize.py`, `csd-benchmark.py` | `test_benchmark_quantized_artifact.py` | doc claims 1e-4 on memory artifact | proven (doc claim + test) |
| split manifests, G26 fail-closed guards | `CORPUS-CONTRACT.md:1234` | `splits.py`, `pretrain.py:1056-1128,1676-1682` | `test_guards_can_fail.py:1810-1823` | guards shown to fire | proven |
| lexical baselines TF-IDF/BM25 with split-sha guard | `METRICS-METHODOLOGY.md` | `eval/lexical.py` | `test_eval_lexical.py` | card golden `:132` | proven |
| model cards via Jinja2 | `docs/technical/model-card-pipeline.md` | `cards/render.py` et al. | 4 test files | golden fixtures | proven |
| `rank.ndcg@10` | `METRICS-METHODOLOGY.md:2016` | `eval/benchmark.py:46,414` | `test_benchmark_metrics_v2_fields.py:126` | — | proven |
| `beir.ndcg@10` | same row | deliberately absent (`beir_fiqa.py:432`) | `test_beir_fiqa.py:394-397` asserts absence | — | missing by design (agreed) |
| `token.*` v2 receipt fields | `METRICS-METHODOLOGY.md:1320` | mapped from `token_aware.final_block_rank.*` | none under v2 name | W1/W1d dirs | implemented-unproven |
| token-aware retrain W4 (`memory`) | taxonomy `:1178-1179` | `pretrain.py` `token_loss_weight` | `test_token_aware_objective.py` | `w4-production-runs`, `w4-control-arm` | ran; row BLOCKED ON OD-17 |
| token-aware retrains W1b/W7a (reason, language) | taxonomy `:1178-1179` "mandatory" | none | none | none | missing |
| token-aware retrain W7v (visual 128px) | taxonomy `:1179`, `:2379-2380` | config half only (DEC-83) | none | none | missing; blocked on OD-4 |
| W1/W1d rank measurement | DEC-35 `:556` | `measure_w1.py`, `measure_w1d.py` | — | `results.json` | proven |
| untrained-baseline gate | `METRICS-METHODOLOGY.md:2019` | `eval/metrics.py` | `test_benchmark_metrics_v2_*` | `w2c-untrained-baselines` | proven |
| overlap-check gate P2.5a | `TRAINING-SUPERSET.md:449,462` | none | none | none | missing ("unstarted") |
| reason/classify wired into runner P2.2 | `TRAINING-SUPERSET.md:1349` | doc says todo; reason cells exist | none | `g48-reason-e1` | contradictory (§8) |
| manifest `status` field enforcement | `MODEL-MANIFESTS.md:151,557` | schema in markdown only | none | `:165` example | not found |
| `corpus_fingerprint` on every receipt | `MATRIX-PIPELINE.md:105-112` | `pretrain.py:1286` | 2 test files | — | proven |
| flickr30k P3.2 | `TRAINING-SUPERSET.md:121` | BLOCKING, no licensor | — | — | excluded |
| gooaq→allenai/gooaq swap | `TRAINING-SUPERSET.md:117` | pending AI2 | — | — | missing |
| `shard_limit` default vs baselines | `MATRIX-PIPELINE.md:117-121` | config default 2; baselines used 4 or 1 | none | doc self-flags | discrepancy |

## 5. Experiments

Decided experiments carry a pre-registered criterion and a verdict; pending ones have a criterion and a cost but no run on `main`.

**Table 5: decided experiments (`docs/design/evidence/`).**

| id | question | criterion | result | verdict | consequence |
|---|---|---|---|---|---|
| W1 | token-surface rank ≥ pooled by 2×? | ≥2.0× pass, <1.5× dead band | PR ratio 0.66-1.30×, 3/4 regions <1.0; entropy ratio reverses sign | central bet dead under PR | licensed W1d |
| W1d | does a read-out probe confirm token signal? | 2.0 pt rule | code, compress, vl_latent CONFIRMED; retrieve noisy (seed spread ≥2 pp, sequence-blind pool beat tokens by 6.05 pp) | PASS (3 clean) | token-aware retrains licensed |
| W2c | true untrained floors? | measure at 2 seeds | code 0.2285/0.2344; retrieve 0.0020 (chance); reason 0.0059/0.0039 | settled | the 0.40 code floor is unsupported |
| W4 masked-token-loss | is MLM loss masked-gathered? | equivalence 1e-6/1e-2 | 12/12 pass; 728.9 vs 1541.4 MiB (2.11×) | PASS | licensed b1280 measurement |
| W4 production (memory, b1280) | gates a-e | 5 gates | a, b, d pass; c fail (r@10 0.20 vs BM25 0.44); e fail (rank ratio 1.21 < 2.0) | PARTIAL, 3/5 | row BLOCKED ON OD-17 |
| W4 control-arm | does the 2.0× gate discriminate? | control must fail | control 2.0191×, token_only 2.0242×, both 2.2412× at 50 steps; at 4,000 steps 1.0851 vs 1.3475 | KILL (gate invalid at 50 steps) | no replacement gate stated |
| g22 visual run 1 | EMA pooled probe beats untrained + 0.01? | both seeds, uncollapsed | seed0 .6246→.7191, seed1 .6391→.7298 | PASS | superseded by run 2 |
| g22 visual run 2 | replicate under fixed pipeline | same | seed0 .6252→.7224, seed1 .6337→.7237 | PASS, replicated | visual production-passing (H1) |
| g48 E1 | is any reason checkpoint step-sensitive? | go ≥0.30, kill ≤0.25 | b256-s0 0.1137, b512-s0 0.1070, untrained 0.1171 (chance 0.20) | KILL | E5 promoted, E3 demoted |
| reason diagnosis | why is reason weakest? | diagnostic | train acc→1.0 vs held-out r@1 0.12-0.19; TF-IDF ceiling 0.873; seed axis resamples split | n/a | defines E0-E5 |

**Table 6: pending experiments and rows.**

| id | question | cost | prerequisite | source |
|---|---|---|---|---|
| E0 | split independent of seed | 0 GPU-min | none | diagnosis README `:83` |
| E2 (reason) | epoch-matched batch ablation, 2 batches × 2 step counts × 3 seeds | ~53 GPU-min | E0 | `:87` |
| E3 | structure-sensitive negatives (demoted) | ~47 GPU-min | E0 | `:89` |
| E4 | third provenance group via `mathematics_dataset` | ~16 GPU-min | E0, corpus admission | `:91` |
| E5 | latent-step prediction, K=1 toy | ~70 GPU-min | E0 | `:93`; reportedly running in a separate worktree, not on main |
| W0 | tokens()/pool() split | 0 GPU | none | taxonomy table |
| W1b, W7a, W7v | token-aware retrains | see Table 12 | W0; W7v needs OD-4 | taxonomy `:1178-1179` |
| W2b | admission gate, freezes region set | 0 GPU | all retrains done | "BLOCKS P2.3" |
| E0/E1/E2 (episodic) | contract, build+probe, admission retrain | unknown | six contract gaps decided | taxonomy §4.1 |
| W5, W5b, W6 | interconnect train, write-back gate, composed metric | unmeasured | W2b, W0 | taxonomy §2.6 |
| W8, W9, W10 | quantize, topology agreement, final | unmeasured | W6 | taxonomy §4 |
| S16 G0-G5 | affect isolation gates | unknown | affect module | g6 README `:22` |

## 6. Corpus and scale gaps

The corpus lives under `/mnt/bulk/csd-corpus/` (staging) and `/mnt/fleet-datasets/csd/` (training root), not `/akula-data/csd/`, which holds only receipts (`scripts/csd-train-all.py:118,416`).

**Table 7: corpus state per region versus contract target.**

| region | trained corpus | rows | licence | contract target | gap |
|---|---|---|---|---|---|
| language/code | codesearchnet-python | 455,243; 1 source; N_eff 1.00 | unresolved (mirrors disagree) | 6 languages, 165,600 rows, N_eff 6.00 | monolingual; 93.9% truncated at max_len 96 (`CORPUS-CONTRACT.md:118`) |
| compress | all-nli | 277,269 train; N_eff 1.95 | BLOCKING (SNLI CC BY-SA); MIT repair via SynCSE | entailment ≥50%, graded ≥5% | no long-form entailment source; graded gate never ran |
| retrieve | fiqa + NQ + gooaq | 505,216 train; N_eff 1.50 | NC tier (GooAQ contested) | 5 sources, 150k cap each, N_eff 4.47 | 79% GooAQ; holdout 53.71% near-dup of train |
| memory | union spec | none trained | inherits above | STS-B + BEIR FiQA heads | not on disk |
| visual | visual-clean-v1, 7 sources | 581,280 | PERMISSIVE_OK | 6 domains ≤20% each, N_eff 5.41 | contract stale: still calls vl_latent unreleasable |
| reason | gsm8k + aqua_rat | 12,455; N_eff ≈1.92 (config) vs 1.15 (contract) | PERMISSIVE_OK; MATH DMCA-rejected | ≥4 shapes ≥20% each | 2 shapes; deduction/multi-hop have no clean source; waiver |
| classify_banking77 | banking77 | 10,003 | CC BY | ≥4 label spaces | N_eff 1; waiver |
| classify_go_emotions | go_emotions | 43,410 | Apache | max share ≤25%, max:min ≤20:1 | neutral 32.8%, ratio 184.7:1; waiver |
| classify (both) | — | max share 81.27%, N_eff 1.44 | — | ≥10% out-of-label-space | 0% supplied |
| residual_mlp, stream_vae | synthetic | n/a | n/a | n/a | no public corpus bound |

**Table 8: `csd-corpus-expand.py` dry run (12 usable, 3 refused, 0 errored).**

| region | present, not in mix | refused |
|---|---|---|
| code | `codeparrot/apps`, `deepmind/code_contests` | `code_search_net:go` (licence `other`) |
| retrieve | `rajpurkar/squad`, `BeIR/hotpotqa` corpus+queries | `BeIR/scifact` (CC BY-SA) |
| reason | already-trained only | `hendrycks/competition_math` (access disabled) |
| vl | `timm/oxford-iiit-pet`, `zalando-datasets/fashion_mnist` | — |

**Table 9: trained scale versus declared scale (latest matrix receipts).**

| region | params | dim/depth/heads/max_len | steps × batch | wall | config declares |
|---|---|---|---|---|---|
| code | 16,021,248 | 256/4/4/96 | 4000 × 1280 | 627.9 s | stream_dim 512, hidden 1024 |
| compress | 16,021,248 | 256/4/4/96 | 4000 × 1280 | 458.2 s | same |
| retrieve | 16,021,248 | 256/4/4/96 | 4000 × 1280 | 425.5 s | same |
| reason | 16,021,248 | 256/4/4/256 | 4000 × 512 | 595.9 s | same |
| visual | 22,905,216 | 384/6/6, image 128, patch 8 | 4000 × 128 | 540.4 s | same |

The dims are hardcoded at `scripts/csd-train-all.py:1007,1478`, and no receipt names the GPU model or host (`"device": "cuda"` only).

**Table 10: measured training cost, 3090 Ti, batch 256, 2026-09-02 (`MODERN-TRAINING-STACK.md`).**

| measure | value | line |
|---|---|---|
| code step, tokenizer share | 131.3 ms, of which 81.5 ms (62%) single-threaded CPU tokenizer | `:84-97` |
| step time spread across text regions | 3.1× (padded width 96.0 vs 32.9) | `:48-52` |
| peak VRAM in a code run | 4,133 of 23,028 MiB (17.9%); 78% idle | `:107-116`, `:155` |
| embed.weight share of params | 80.3% | `:176-190` |
| quantized code artifact | 6,543,592 bytes, 3.27 effective bits/param | `:176-190` |
| bf16 batch headroom fit | `peak_MiB ≈ 245 + 7.25·B`; fits ~2,560 | `:135-143`, `:160-169` |
| pairs/s (code, compress, retrieve, vl) | 1,950; 6,054; 3,732; 4,123 | throughput table |

**Table 11: the 1B-per-submodel step and its data gate (DEC-56, DEC-73, DEC-74).**

| item | value | source |
|---|---|---|
| data gate per region | ~10^10 tokens, later per-faculty (DEC-73) | taxonomy `:577` |
| reach of target after factory pass 1 | visual 0.3%, memory 1.3-5%, reasoning 1.8%, language 15%; language_trunk 60× over | taxonomy `:577` |
| placement | composed CSD on 5080 + 3090 Ti; 1080 Ti is RAG helper | DEC-67 `:224-225,588` |
| 1080 Ti torch | pinned `torch 2.11.0+cu128` cannot run (sm_61 absent) | DEC-74 `:595` |
| layer-sectioned training across 3 cards | candidate only (DEC-55), not adopted | `:576` |
| PTQ sensitivity carry-forward | not carried to 1B; re-measure at size | DEC-56 (ii) |

## 7. Ranked gap list

Ordered by the operator's direction: prove planned features, then missing modules, then composed training/evals/experiments, then corpus and parameter expansion. GPU-hour estimates use Table 9 (a text cell at b1280 ≈ 0.12-0.18 GPU-h; a visual cell ≈ 0.15 GPU-h) and the diagnosis README's per-experiment budgets; interconnect costs are unmeasured guesses and marked so.

**Table 12: ranked gaps.**

| # | gap | why it blocks | lane shape | prerequisite | GPU-h |
|---|---|---|---|---|---|
| 1 | OD-17: decide W4 gate `>` vs `≥`, pivot vs amend | W4 row is BLOCKED; memory retrain and W2b freeze wait on it (taxonomy `:589`) | operator decision, then single agent (mid) to record DEC | none | 0 |
| 2 | E0: make the split independent of seed; guard that fires on a swapped pair | every seed comparison since is contaminated (diagnosis README `:83`; memory `identical-seeds-as-control`) | single agent (mid) with `test_guards_can_fail` pattern | none | 0 |
| 3 | W0: `tokens()`/`pool()` protocol, re-instantiate parameter table | first `todo` on the interconnect chain; nothing composed can be typed without it | single agent (high) | none | 0 |
| 4 | replacement for the invalidated 2.0× rank gate | W4 control arm KILLed it at 50 steps; only the 4,000-step arm discriminates; no gate means no retrain can be judged | workflow (high design, mid run): pre-register, run control + treatment at 4,000 steps | #2 | ~1 |
| 5 | visual: land eval/quant receipts in the matrix dir, then W7v 128px token-aware retrain (OD-4) | visual is train-only in matrix; W7v is mandatory for the freeze (DEC-83 `:604`) | workflow (mid) | OD-4 decision; #3 | ~0.5 |
| 6 | W1b/W7a token-aware retrains of reason and language, 2 seeds each | mandatory since W1 (taxonomy `:1178-1179`); freeze cannot happen without them | workflow (mid) over existing matrix config | #2, #3, #4 | ~1 |
| 7 | E2 reason batch ablation | settles whether b256>b512 is an epoch effect before any reason retrain is judged | workflow (mid), 12 cells | #2 | ~0.9 |
| 8 | E5 latent-step prediction, verify on main | the promoted reasoning objective after the g48 KILL; a worktree run exists but main shows nothing | single agent (high) to review the worktree result, then merge | #2 | ~1.2 |
| 9 | memory region: put the union corpus on disk, run the first matrix cell with full receipts | v1 participant with no checkpoint; `available: false`; W4 evidence runs are outside the matrix | workflow (mid) | #1, corpus admission | ~0.5 |
| 10 | interconnect module: controller, workspace, adapters, K/V bank, read-out, `Schedule` emission, tests | zero code, zero tests; the composed model cannot be trained | workflow (high): design-to-code with adversarial review per AGENTS.md | #3 | ~0.1 smoke |
| 11 | episodic store: operator decides the six contract gaps, then E0/E1 build and probe | DEC-49 makes it a v1 participant; E2 admission cannot run without it | operator deliberation, then workflow (high) | #10 | ~0.3 probe |
| 12 | hygiene that proves the contract: mark `core/` stack superseded in docstrings, fix or remove the `MindSpec` stream_dim check, fold §7.2/§7.3 edits into `TRAINING-SUPERSET.md` and `MODEL-MANIFESTS.md` | docs and code contradict each other on what the composed model is (§8) | single agent (mid) | #3 | 0 |
| 13 | compose stage in `csd-train-all.py`, composed eval writer, populate `composed.md.j2` | STAGES lists `compose` but nothing writes it; no composed receipt can exist | workflow (high) | #10 | 0 |
| 14 | W2b freeze → W5 phases A-D → W5b → W6 | the composed training itself | workflow (high), emitter-wired per `emitters-not-polling` | #5-#11, #13 | ~4-12 (unmeasured) |
| 15 | episodic E2 admission retrain, then W8/W9/W10 | DEC-50 step 2 and the final gates | workflow (mid) | #14 | ~2-4 (unmeasured) |
| 16 | licence resolution: code zero-delta, compress SynCSE swap, retrieve NC/GooAQ, MATH vs PRM800K DMCA conflict, composed repo `mit` tag | three of five trained regions are not open-weights clean; composed licence inherits the strictest tier | operator rulings, then single agent (mid) per region | none | 0 (compress retrain ~0.2) |
| 17 | code corpus: 6-language mix, raise max_len past the 93.9% truncation, parallelise the tokenizer | single-source N_eff 1.00; 62% of each step is CPU tokenizer | workflow (mid) | #16 | ~0.5 |
| 18 | reason E4 third provenance group; classify out-of-label-space population; retrieve rebalance to 5 sources | reason has 2 of 4 required shapes; classify supplies 0% of the required OOL rows; retrieve is 79% GooAQ | workflow (mid) with the dataset-factory verifier | #16 | ~0.3 |
| 19 | 1B-per-submodel path: dataset-factory pass 2 against DEC-73 per-faculty targets; decide DEC-55 layer-sectioning | data is the stated long pole; visual reach is 0.3% | workflow (high) survey lane, then operator | #14 green | 0 until data exists |

## 8. Discrepancies between docs and reality

Where two surveys disagree, both readings are given; none is chosen here.

**Table 13: discrepancies.**

| # | reading A | reading B | note |
|---|---|---|---|
| 1 | `csd-regions.json:59-90`: compress and retrieve `merged_into: memory`, not retrained standalone | standalone cells with full receipts dated 2026-09-04 exist in the matrix | the cells predate or ignore the merge note; config is the stale side or the runs were pre-merge baselines |
| 2 | composed and experiments surveys: W4 trained the memory region at b1280 (`w4-production-runs-2026-09-03`) | regions, corpus, scale surveys: no `memory-*` cell in the matrix; corpus `available: false` | both hold if W4 ran outside the matrix pipeline; the matrix has no memory receipt either way |
| 3 | g22 run 2 reports visual eval results under a "fixed pipeline (visual eval + quantize stages)" | matrix dirs for visual hold train receipts only; config says `quantization.skip: true` (`:122`) | eval evidence lives in `docs/design/evidence/`, not the matrix |
| 4 | DEC-83 (`:604`): W7v `corpus_source` is `None`, the run refuses to start | corpus survey: visual trained on visual-clean-v1 (581,280 rows) | W7v (128px token-aware rebuild) may be distinct from the g22 pooled-probe runs; unresolved |
| 5 | `TRAINING-SUPERSET.md:1349`: reason and classify "not yet wired into the runner" (P2.2 todo) | reason cells at b256 and b512 with full receipts exist | doc stale, or reason ran through a path the doc does not count |
| 6 | `csd-regions.json` declares stream_dim 512, hidden 1024 | `csd-train-all.py:1007,1478` hardcode dim 256, depth 4, heads 4; every receipt agrees | config does not describe what is trained |
| 7 | taxonomy §2.2 (~`:1194`): the `MindSpec` stream_dim uniformity check "is replaced" | `region_spec.py:163-171` still raises on mismatch | code behind doc |
| 8 | DEC-12 (`:1026`): mark `core/interconnect_manager.py` and `integrated_system.py` superseded in their docstrings | no such marker in either file | code behind doc |
| 9 | composed card and DEC-31/§5.7: composed licence unresolved, inherits strictest tier (NC via GooAQ) | live `tzervas/cogsyndelta` Hub repo tagged `license:mit` | Hub metadata contradicts the design |
| 10 | taxonomy §7.2/§7.3: phase order and manifest fields to be folded into `TRAINING-SUPERSET.md`, `MODEL-MANIFESTS.md` | `TRAINING-SUPERSET.md:268-331` still states the old 4-phase curriculum; manifests doc lacks `faculty`, `token_budget`, `overlay` | edits documented as pending, not applied |
| 11 | `CORPUS-CONTRACT.md` §§1.4-1.6: six-region pre-DEC-02 model; classify/reason have no role; vl_latent unreleasable | config: memory region, roles and waivers for reason/classify, visual renamed and trained on a permissive mix | contract stale |
| 12 | task brief: corpus under `/akula-data/csd/` | `/akula-data/csd/` holds receipts only; corpus at `/mnt/bulk/csd-corpus/` and `/mnt/fleet-datasets/csd/` | path assumption wrong |
| 13 | prose: code untrained lexical floor ≈0.40 | W2c measured 0.2285/0.2344 | no artefact backs 0.40 |
| 14 | recorded retrieve untrained r@1 0.0000 | W2c measured 0.0020 (chance) | superseded measurement, not a bug |
| 15 | `CORPUS-CONTRACT.md:711-713`, `csd-regions.json:149`: MATH family DMCA-rejected | pass-1 catalogue: PERMISSIVE_OK verified at primary LICENSE | blocks PRM800K and MATH-500 admission until ruled |
| 16 | `MATRIX-PIPELINE.md:117`: `shard_limit` default 2 | baselines used 4 shards (code) or 1 (compress, retrieve, reason) | doc self-flags; comparability check ignores shard count |
| 17 | survey ask: wall time and GPU per cell | receipts carry `"device": "cuda"` only, no GPU model or host | provenance gap in the receipt schema |
| 18 | taxonomy treats the 5080 (sm_120) as a settled deployment target | `docs/GPU_COMPATIBILITY.md:11,49,58`: sm_120 support "requires verification" | unresolved |
| 19 | `matrix/tables/csd-m1-20260904/matrix.md` lists memory/language/compress/retrieve/reason cells at `3ce18db` | no such directories exist; `selection.json` lists only `visual` | planning table, not runs |
| 20 | W4 status "proven (ran, has receipts + control arm)" (features survey) | W4 row "BLOCKED ON OD-17" (composed survey) | different senses: executed versus closed |
| 21 | composed survey names the retrain rows W1b (reason), W4 (memory), W7a (rest), W7v (visual) | features survey names them W7a/W7b (language, reasoning) | row ids differ between readings of the same table; confirm against the taxonomy table before scheduling |
| 22 | `csd-regions.json:11-25`: residual_mlp `live: true` | only a PoC class, no cell, synthetic corpus | live flag does not mean trained |
| 23 | DEC-78 rename code→language landed in aliases and `MindSpec` | no `language-*` cell exists; all cells and receipts are named `code` | alias compatibility is doing the work; artifacts not regenerated (memory `early-alpha-no-legacy-compat`) |
