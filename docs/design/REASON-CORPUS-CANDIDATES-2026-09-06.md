# Reasoning-process corpora for the `reason` region — candidates for E4 / E5

**Status:** research note, no code. Written 2026-09-05 for the reason-region diagnosis
(`docs/design/evidence/reason-region-diagnosis-2026-09-06/README.md`, cited as `DIAG`),
sections 3–5. It informs the operator rulings that DIAG §5 asks for; it admits nothing.

**Ground rules applied.** The operator's definition of reasoning for this region is a
*process* — ruminating on a problem, exploring solutions, questions and answers — not a
lookup. A candidate scores on whether it carries derivations with steps, not on whether it
carries answers. Every licence below was read at the **primary source** (a GitHub `LICENSE`,
a first-party card, a Zenodo record), never at a Hugging Face mirror tag; where the mirror
tag disagrees it is said. Sizes are what the primary or first-party page states; token
counts are estimates and say so. Every fact fetched in this pass is dated 2026-09-05; facts
carried from the 2026-09-03 factory pass cite `docs/design/evidence/dataset-factory-2026-09-03/`
(`SURVEY` = `10-survey-reason.md`, `VERIFY` = `30-verify-reason.md`) or the catalogue
`docs/design/datasets/catalogue-2026-09-03.json` (`CAT`). `[I]` marks an inference or
estimate; **unverified** marks a claim this pass could not check.

Dataset-factory conventions honoured (`dataset-factory/README.md`, `docs/DESIGN.md`,
`src/dataset_factory/catalogue.py:108`): admission is verdict ∈ {PERMISSIVE_OK, ATTRIBUTION,
SHARE_ALIKE, NC} with `verification_status == VERIFIED`, `grant_scope` a full-content grant,
no unresolved provenance red flag, no ND; every emit carries exactly one landing `role` of
`train | probe | aux | refuse | held_seed` (DEC-85); **official test splits and carved
held-out slices land under `probe/`, never `train/`**; a derived set inherits the
strictest input licence (`enrich.py::strictest_output_verdict`, `attribution.py::_most_restrictive`).

---

## 1. What is already in the corpus

| source | landed as | rows in training | licence at primary (this pass) | landed manifest says |
|---|---|---|---|---|
| `openai/gsm8k` (`main`) | `/mnt/bulk/csd-corpus/reason/gsm8k-main/train.parquet` | 7,473 (all of `train`) | **MIT**, `github.com/openai/grade-school-math` `LICENSE`, "Copyright (c) 2021 OpenAI" | `"mit (verified via HF dataset_info tags)"` — i.e. verified at the *mirror only*; this pass closes that at the primary |
| `deepmind/aqua_rat` (`raw`) | `/mnt/bulk/csd-corpus/reason/aqua_rat-raw/train.parquet`, `derived/sample-4982-seed0.parquet` | 4,982 reservoir-capped of 97,467 | **Apache-2.0**, `github.com/google-deepmind/AQuA` `LICENSE` | `"apache-2.0 (verified via HF dataset_info tags)"` — same mirror-only verification, closed here |

Neither source's **official test split** was landed (`splits: ["train"]` in both
`MANIFEST.json`). The fixed holdout (`config/mind/splits/reason-ca364a92-split0.json`,
sha256 `77d2c0e1ac02db40c49fd1258785c005b202da184622acaccc7c3f73144baa13`, 512 pairs = 320
gsm8k + 192 aqua_rat) is therefore carved from **gsm8k `train` and aqua_rat `train`**. That
matters for contamination below: any dataset built on gsm8k `train` (OpenMathInstruct,
MetaMathQA) contains the holdout's own problems.

Corpus fingerprint today: `ca364a92d2c6c5fd259404e0ab6f52a1` (DIAG §2). Adding any source
changes it; §4 says what that forces.

---

## 2. Candidate table

Fit is scored 0–5 separately for **E4** (a third provenance group under the synthetic /
multi-shape contract, judged on the E1 derivation-sensitivity battery, DIAG §4) and **E5**
(latent next-step prediction over a derivation split into steps, DIAG §4). A 5 means the
data has the step structure E5 consumes natively; 0 means answer lookup only.

### 2.1 Arithmetic and algebra word problems (the shapes the corpus has)

| # | candidate | primary source and licence verified there | mirror tag | size (items) | tokens, est. | step-level structure | official test split | contamination vs the existing reason corpus | fit E4 / E5 and why |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **gsm8k** | `github.com/openai/grade-school-math/LICENSE` → **MIT** | `mit` (agrees) | 7,473 train / 1,319 test; `socratic` config adds sub-questions | ~1.1M GPT-2 tokens for train [I]: answers mean 95.3 tokens (DIAG §2 [C]) + questions ~55 [I] | **Yes**: one line per step, calculator annotations `<<a op b=c>>`, `####` final answer; `socratic` config gives an explicit sub-question per step | Yes, 1,319 — **not landed** | **It is the corpus.** Its `train` supplies 320 of the 512 fixed holdout pairs | control / **5** — it is the substrate E1's corruptions and E5's step split are built on; the only action is landing `test` under `probe/` |
| 2 | **aqua_rat** | `github.com/google-deepmind/AQuA/LICENSE` → **Apache-2.0** | `apache-2.0` (agrees) | 97,467 train / 254 val / 254 test (`raw`) | ~14M tokens full [I] (rationales mean 72.1 tokens, DIAG §2 [C]); ~0.7M at the 4,982 cap | Partial: free-text rationale, no annotations; 4.0% of rationales under 8 words, 17.0% duplicate questions (DIAG §3 [C]) | Yes, 254 + 254 val — **not landed** | **It is the corpus** (192 holdout pairs). `allenai/math_qa` is the same problems (row 3) | control / **3** — steps exist only as sentences with no checkable intermediate values, so no per-step corruption is possible on this half (DIAG E3's B1 caveat) |
| 3 | **MathQA** (`allenai/math_qa`) | first-party AI2 HF card: "licensed under the Apache License, Version 2.0" (`VERIFY` #6); no GitHub LICENSE found by this pass | `apache-2.0` (agrees) | 37,297 | ~5M tokens [I] | **Yes, and executable**: each AQuA-RAT problem re-annotated with a "fully-specified operational program" (`CAT`) | Yes (train/validation/test) | **Same provenance group as aqua_rat** (`CAT` group `aqua-rat`): it is the same problem text. Zero B1/B2 gain. Its programs, however, give the aqua half exactly the checkable step structure it lacks | **2 / 4** — no new shape (E4 null by construction); for E5 and E3 it is the cheapest way to make aqua_rat rows corruptible: execute the program, perturb one operation, and the corrupted derivation is verifiable. Admit as a *swap* for raw aqua_rat, not an addition (`SURVEY` #6) |
| 4 | **DeepMind `mathematics_dataset`** (generator) | `github.com/google-deepmind/mathematics_dataset` `LICENSE` → **Apache-2.0** (`CAT`, 30-verify-other-sources #6; re-read this pass) | n/a (GitHub) | generator; pre-generated v1.0 has "2 million (question, answer) pairs per module", 8 categories, train-easy/medium/hard + interpolate/extrapolate tests; questions ≤160 chars, answers ≤30 chars | ~45 tokens/item [I] → any volume the caps allow | **No** — "final answers only … without solution steps" (README, this pass) | Yes: interpolate and extrapolate test sets per module | None with gsm8k / aqua_rat (templated, not scraped). DEC-58 clause 2 still requires the contamination channels to run against **every** eval | **4 / 1** — this is E4's designated arm (DIAG §4): the one settled-grant source, numeric-shaped, and a clean null ("volume of a different shape does not transfer") is a result. Useless for E5 as shipped: there is no step to predict. Only the **generator's internal composition tree** could yield steps, which would be a CSD-side enrichment and a new synthetic row |
| 5 | **MATH** (Hendrycks; `hendrycks/math` on GitHub, `EleutherAI/hendrycks_math` on HF) | `raw.githubusercontent.com/hendrycks/math/main/LICENSE` → **MIT**, "Copyright (c) 2021 Dan Hendrycks" (two independent fetches, `VERIFY` #1, 30-verify-other-sources #7) | `EleutherAI/hendrycks_math`: `mit`, accessible. **`hendrycks/competition_math`: "Access to this dataset has been disabled" — DMCA takedown notice** (HF, this pass) | 12,500 = 7,500 train / 5,000 test, 7 subjects | ~4–5M tokens [I] (LaTeX solutions ~150–300 tokens each [I]) | Partial: full worked solutions in prose/LaTeX, `\boxed{}` final answer checkable; steps not delimited | Yes, 5,000; the 500-problem MATH-500 subset is PRM800K's `test.jsonl` | None with gsm8k / aqua_rat. **Upstream of** PRM800K, MetaMathQA, OpenMathInstruct-1/2, MathInstruct, NuminaMath — count problem text once | **3 / 4** — the richest human-written derivation set in the permissive tier; but DIAG §5 item 4 stands: `CC:711-713` and `csd-regions.json:149` call it DMCA-rejected while `CAT` records PERMISSIVE_OK at the primary LICENSE. **Blocked on an operator ruling.** The takedown's claimant and grounds are **unverified** by this pass |
| 6 | **PRM800K** (`github.com/openai/prm800k`) | `raw.githubusercontent.com/openai/prm800k/main/LICENSE` → **MIT**, "Copyright (c) 2023 OpenAI" (`VERIFY` #2) | `tasksource/PRM800K`: `mit` (agrees); `openai/prm800k` is not an HF repo | 800,000 step labels; ~12k MATH problems, ~75k solutions [I, paper]; two phases; `train.jsonl` / `test.jsonl` | ~100M tokens across all sampled solutions [I] | **Yes — the only human step-level correctness labels on this list**: per step −1 / 0 / +1, alternative steps written by labelers in phase 1 | `test.jsonl` = 500 MATH test problems (MATH-500); **`train.jsonl` includes 4,500 MATH test problems** (README, this pass) | None with gsm8k / aqua_rat. Problem text is MATH's (row 5). Solutions were **sampled from an OpenAI model and human-graded** — `SURVEY` #2's "not GPT output" is wrong on the solutions, right on the labels; OpenAI is the rights holder releasing its own model's output under MIT, so no third-party output-terms tension | **4 / 5** — natively supplies (question, steps ≤ t, true step t+1, labelled-wrong alternative) tuples: E5's battery and E3's structure-sensitive negatives without any corruption heuristic. Rides on the MATH ruling; if MATH is ever a probe, PRM800K `train` contaminates 4,500 of its 5,000 test problems, so **MATH-500 is the only admissible MATH probe** once PRM800K is in |

### 2.2 Synthetic derivations over gsm8k / MATH (provenance chains)

| # | candidate | primary source and licence verified there | mirror tag | size | tokens, est. | step-level structure | official test split | contamination | fit E4 / E5 and why |
|---|---|---|---|---|---|---|---|---|---|
| 7 | **OpenMathInstruct-1** (`nvidia/OpenMathInstruct-1`) | first-party card + `LICENSE` file: **"NVIDIA License"** — perpetual, royalty-free grant to "use, reproduce, prepare derivative works of … sublicense and distribute"; retain notices; patent-retaliation and trademark clauses; no NC, no ND (this pass) | `other` / `license_name: nvidia-license` (agrees) | 6.08M rows (4.95M train / 1.13M validation); 8.94 GB | ~2B tokens [I] | **Yes**: text reasoning interleaved with executable Python; `is_correct` per solution | No (evaluates on gsm8k / MATH test) | **Chain:** gsm8k `train` (MIT) + MATH `train` (MIT, DMCA-disputed) → solutions by Mixtral-8x7B (Apache-2.0 model; no output-terms tension) → NVIDIA License. **Contains many solutions for every gsm8k `train` problem, including the 320 holdout anchors** — must be scrubbed on anchor fingerprint before any use, and the MATH half inherits the MATH ruling | **3 / 3** — real volume of code-checked derivations with a clean generator, but DEC-58 caps it at ≤ 0.40 of the bin as one provenance group, the quality gate must be shown to fail, and it adds no new *problem* shape: it is more solutions to the corpus's own questions. Its verdict class is ATTRIBUTION (notice retention), not PERMISSIVE_OK |
| 8 | **OpenMathInstruct-2** (`nvidia/OpenMathInstruct-2`) | first-party card: **CC BY 4.0** (this pass) | `cc-by-4.0` (agrees) | 14M train (22M rows incl. 1M/2M/5M fair downsamples) | ~5B tokens [I] | Yes (text solutions, no code) | No; ships a contamination explorer against gsm8k / MATH / AMC / AIME / Omni-MATH | Same chain as row 7 plus **new synthetic problems**; **generator is Llama-3.1-405B-Instruct**, whose licence states verbatim: "If you use the Llama Materials or any outputs or results of the Llama Materials to create, train, fine tune, or otherwise improve an AI model, which is distributed or made available, you shall also include 'Llama' at the beginning of any such AI model name" and requires "Built with Llama" display (developer.meta.com, this pass) | **2 / 2** — the CC BY grant is clean; the **model-naming obligation on CSD's own weights** is a provenance red flag of exactly the MODEL-OUTPUT-TERMS class (OD-18) and needs an operator ruling before the row can be VERIFIED. Volume is not what `reason` lacks |
| 9 | **MetaMathQA** (`meta-math/MetaMathQA`) | mirror tag only; card prose: "All MetaMathQA data are augmented from the training sets of GSM8K and MATH. None of the augmented data is from the testing set."; **rephrasing by GPT-3.5 per the paper (arXiv:2309.12284), not stated on the card** (`VERIFY`) | `mit` | 395,000 (MATH_AnsAug, GSM_Rephrased, GSM_SV, GSM_FOBAR) | ~150M tokens [I] | Yes (free-text rationale) | No | **Worst on this list**: GSM_Rephrased / SV / FOBAR are **paraphrases and inversions of gsm8k `train` questions — i.e. of the holdout's own anchors**. Only a Stage-2 semantic screen (CC §2.3) could catch it; Stage-1 exact overlap will not | **1 / 1** — held at NC in `CAT` pending the model-output-terms ruling, and everything it adds is a paraphrase of what the corpus already holds. Not recommended for either experiment |
| 10 | **OpenR1-Math-220k** (`open-r1/OpenR1-Math-220k`) | first-party card: **Apache-2.0**; DeepSeek-R1 traces over NuminaMath 1.5 problems (`VERIFY` #7) | `apache-2.0` (agrees) | ~220k verified-correct traces | ~1B tokens [I] (long CoT) | Yes (long free-form traces, Math-Verify-checked answers) | No | Problem text is NuminaMath's (aops_forum chain open; orca_math GPT-4-Turbo) — `CAT` red flag; MODEL-OUTPUT-TERMS OD-18 item 2 (DIAG §5) | **2 / 3** — long derivations, but the *problem* provenance is the open question, not the trace licence; parked with OD-18 |

### 2.3 Formal and proof-shaped

| # | candidate | primary source and licence verified there | mirror tag | size | tokens, est. | step-level structure | official test split | contamination | fit E4 / E5 and why |
|---|---|---|---|---|---|---|---|---|---|
| 11 | **ProofNet** (`hoskinson-center/proofnet`) | `raw.githubusercontent.com/zhangir-azerbayev/ProofNet/main/LICENSE` → **MIT** (`VERIFY` #5) | `mit` (agrees) | 371 = 185 validation / 186 test; `nl_statement`, `nl_proof` (LaTeX, present on all 371), `formal_statement` (Lean 3) | ~0.2M tokens [I] | Partial: informal proofs, unsegmented, no checkable intermediate values | Yes — the whole set is validation + test; **there is no train split** | None | **1 / 2** — a genuine proof shape, but 371 items with no train split is a **probe**, not a training source; land under `probe/` as the deduction-shape held-out set |
| 12 | **miniF2F** (`github.com/openai/miniF2F`) | README on `master` (the `LICENSE` file 404s on both `main` and `master` this pass — **licence verified from the README only**): "`lean` folder … Apache License", "`metamath` folder … MIT", "`isabelle` folder … Apache", "`hollight` folder … FreeBSD" | n/a | 244 valid / 244 test per system (165 HOL Light) | ~0.05M tokens [I] | No — formal *statements*; proofs largely absent | Yes (valid / test only, no train) | Problems are AMC / AIME / MATH-derived [I] — overlaps MATH's competition sources | **0 / 1** — statement lookup; EVAL-ONLY at best, and its competition provenance carries the same question as MATH |
| 13 | **LeanDojo Benchmark / Benchmark 4** (`github.com/lean-dojo/LeanDojo`; Zenodo 10.5281/zenodo.8016385 and 8040109) | Zenodo record 8016385: **"Creative Commons Attribution 2.0 Generic"** (this pass). Code: MIT. Underlying mathlib: Apache-2.0 [I, not fetched] | n/a | 98,734 theorems and proofs (arXiv:2306.15626 abstract) with premise annotations; 42.0 GB total record, 47.3 MB core file | ~50M tokens of Lean [I] | **Yes, and machine-checkable**: tactic-by-tactic proof states; every step verifiable by Lean | Yes — `random` and `novel_premises` splits [I, from the paper; not re-verified this pass] | None with gsm8k / aqua_rat; none with Wikipedia | **3 / 3** — the one candidate whose steps are *exactly* verifiable and whose shape (formal deduction) is missing from the corpus. Costs: **ATTRIBUTION** verdict (CC BY 2.0 propagates a notice into the region's tier), the token surface is Lean not English, and the E1 battery does not read it — E5 would need its own step-corruption battery (swap one tactic; Lean rejects it). Best as the deduction-shape row *after* E5 has a positive result on gsm8k |
| 14 | **Lean Workbook** (`internlm/Lean-Workbook`) | first-party card: **Apache-2.0** (this pass) | `apache-2.0` (agrees) | 25,214; NL statement + Lean 4 formal statement + partial proofs; `status` proved/disproved | ~5M tokens [I] | Partial; auto-formalised by InternLM (synthetic, DEC-58) | No | Contest-problem sources unstated → same open chain as NuminaMath [I] | **1 / 1** — synthetic formalisation with an unstated problem chain; not recommended |

### 2.4 Multi-hop, deduction and commonsense QA

| # | candidate | primary source and licence verified there | mirror tag | size | tokens, est. | step-level structure | official test split | contamination | fit E4 / E5 and why |
|---|---|---|---|---|---|---|---|---|---|
| 15 | **StrategyQA** (`github.com/eladsegal/strategyqa`; mirror `wics/strategy-qa`) | GitHub `LICENSE` → **MIT**, "Copyright (c) 2021 Elad Segal" (`VERIFY` #8) | **`other` — disagrees; primary wins** | 2,780 (arXiv:2101.02235); 2,290 train with decompositions and evidence, ~490 test with hidden answers [I — the `wics` mirror shows a "test" of 2,290 rows that looks like a duplicate of train; **official test size unverified** this pass, allenai.org/data/strategyqa now redirects to Semantic Scholar] | ~0.4M tokens [I] | **Yes**: gold `decomposition` into sub-questions, `facts`, and Wikipedia `evidence` paragraphs per step | Yes (hidden answers, leaderboard) | None with gsm8k / aqua_rat. **Red flag:** the `evidence` paragraphs are Wikipedia text (CC BY-SA), which the MIT grant does not reach — `grant_scope` is heterogeneous. Landing questions + decompositions + facts only stays MIT; landing evidence pulls in `retrieve`'s Wikipedia lineage (CC §2) and a share-alike term | **4 / 3** — DIAG E4's conditional multi-hop arm, verbatim: render decompositions as rationales (`CAT`). Steps are questions, not checkable values, so E5's corruption is "swap one sub-question for another problem's" — weaker than gsm8k's but real. Small: a minority stratum, never a B1 mover |
| 16 | **HotpotQA** (`hotpotqa.github.io`) | site: "HotpotQA is distributed under a CC BY-SA 4.0 License" (`CAT`, re-read this pass) → **SHARE_ALIKE** | `cc-by-sa-4.0` (agrees) | 90,447 train / 7,405 dev / 7,405 test (fullwiki, hidden) | ~60M tokens with contexts [I] | Partial: sentence-level supporting facts (2 hops), no derivation text | Yes | **Wikipedia lineage collides with `retrieve`** (CC §1.6, DIAG §5 item 6); share-alike would flip `reason`'s tier from permissive to SHARE_ALIKE | **1 / 1** — out of scope per DIAG §5; listed because the brief asks. Supporting-fact pairs are lookup, not derivation |
| 17 | **ARC** (`allenai/ai2_arc`; allenai.org/data/arc 302-redirects to this card, so the AI2 card *is* the primary) | first-party card tag **`cc-by-sa-4.0`**, no licence prose (this pass) → SHARE_ALIKE | (is the primary) | 7,787 MCQ: Challenge 1,119 / 299 / 1,172; Easy 2,251 / 570 / 2,376 | ~0.5M tokens [I] | **No** — question, four choices, answer key; no rationale | Yes | None | **0 / 0** — answer lookup by construction, and share-alike. Not a reasoning-process corpus under the operator's definition |
| 18 | **LogiQA** (`github.com/lgw863/LogiQA-dataset`) and **LogiQA 2.0** (`github.com/csitfun/LogiQA2.0`) | LogiQA 1: **no LICENSE file at the primary** (this pass; the paper says only "freely available") → **UNVERIFIED, refuse-by-default**. LogiQA 2.0: README states **CC BY-NC-SA 4.0** (no LICENSE file) | `lucasmccabe/logiqa`: "[More Information Needed]" (agrees that none is stated) | 8,678 = 7,376 / 651 / 651 MCQ | ~1.5M tokens [I] | **No** rationales (context, question, four options, answer index) | Yes | None. Source is translated **Chinese civil-service examination questions** — third-party authorship the distributor does not own [I] | **1 / 0** — no steps, no licence (v1) or NC-SA (v2), and a provenance chain of the "distributor never owned the base corpus" class |
| 19 | **FOLIO** (`github.com/Yale-LILY/FOLIO`; mirror `yale-nlp/FOLIO`) | GitHub `LICENSE` → **CC BY-SA 4.0** (this pass) → SHARE_ALIKE | **`mit` on `yale-nlp/FOLIO` — disagrees; primary wins** (the HF card is gated behind a contact-info agreement) | 1,430 examples over 487 premise sets (arXiv:2209.00840); train + validation public, **test unreleased** | ~0.3M tokens [I] | Partial: premises and conclusion each with a first-order-logic annotation verified by an inference engine; **the engine's derivation is not shipped** — label + formulas only | Test hidden; validation is the usable held-out | None | **2 / 2** — the cleanest *deduction* shape on the list, but share-alike and derivation-free as distributed: what it offers E5 is premises→conclusion, not steps. A CSD-side enrichment (run a prover over the FOL, keep its proof trace) would produce steps — and a CC BY-SA derivative |
| 20 | **BIG-bench subsets** (`github.com/google/BIG-bench`) | repo `LICENSE` → **Apache-2.0** (raw, `VERIFY`); contributions under a CLA; **per-task licence heterogeneity unverified** — the tasks README does not say | `maveriq/bigbenchhard` card asserts "MIT" for the upstream — **wrong** (`VERIFY`) | 200+ tasks; reasoning-shaped JSON tasks include `logical_deduction`, `logic_grid_puzzle`, `tracking_shuffled_objects`, `multistep_arithmetic`, `list_functions`; BBH = 23 tasks with 3-shot CoT exemplars only | ~0.2M tokens per task [I] | Mostly **no** (answers only); BBH's CoT lives in a handful of prompt exemplars, not per item | Test only — no train splits | `strategyqa` is itself a BIG-bench task (row 15 duplicated) | **1 / 1** — `CAT` marks it **EVAL-ONLY**; a `probe/` landing for a deduction battery, never train |
| 21 | **bAbI** (`facebookarchive/bAbI-tasks`) | `LICENSE.md` → **BSD-3-Clause** (30-verify-other-sources) | n/a | 20 tasks, 1k / 10k per task (toy) | ~1M tokens [I] | Partial: supporting-fact indices per answer | Yes | None; a Lua generator exists → DEC-58 synthetic | **2 / 1** — a controllable deduction shape for an E4-style transfer null; too toy for E5 |

Rows the brief did not name but the catalogue holds for `reasoning`, unchanged from `CAT`
and out of scope here: `camel-ai/math` (NC, GPT-4), `orca-math` (NC, GPT-4-Turbo),
`TIGER-Lab/MathInstruct` (NC-if-whole), `NuminaMath-CoT` (aops chain open), `proof-pile-2`
(arXiv bulk rights), `Nemotron-Math-Proofs-v1` (CC BY-SA), `proofwriter` (no licence found),
`open-web-math` (ODC-By, database-rights scope, not derivation-shaped).

---

## 3. Ranked shortlist (five)

Ranking is E5 value × admissibility today, with E4's designated arms first because they are
pre-registered.

| rank | candidate | role(s) | verdict / tier effect on `reason` | what it buys | gate before landing |
|---|---|---|---|---|---|
| **1** | **DeepMind `mathematics_dataset`** (row 4) | `train` (capped, one provenance group `deepmind-mathematics`) + interpolate/extrapolate tests → `probe` | PERMISSIVE_OK; tier unchanged | E4's arm exactly as pre-registered; settled grant; a clean null is a result | DEC-58's four clauses: generator named with revision and licence in provenance; contamination channels against **every** eval; a failable quality gate demonstrated on degraded output; share ≤ 0.40 of the bin. Category is the B5 stratum key; caps by reservoir sampling with the seed recorded (B4) |
| **2** | **StrategyQA** (row 15) | `train` (questions + decompositions + facts, MIT) ; `evidence` paragraphs → `aux` or omitted | PERMISSIVE_OK **only without the Wikipedia evidence**; with it, SHARE_ALIKE and a lineage collision | E4's conditional multi-hop arm; the one new *shape* with gold steps | Catalogue row must carry `grant_scope: "whole_corpus (heterogeneous per file)"` and a red-flag resolution naming the evidence exclusion; official test size confirmed at the primary (unverified here) |
| **3** | **MathQA** (row 3) | `train` as a **swap** for raw aqua_rat; its `test` → `probe` | PERMISSIVE_OK; group `aqua-rat` unchanged, so B1/B2 unchanged | Executable operation programs on the aqua half: removes E3's B1 caveat and makes E5's step corruption checkable on 40% of the corpus | Stage-1 fingerprint proof that MathQA's problems map 1:1 onto the aqua_rat rows already in the fixed holdout, so the holdout is re-expressed, not moved (E0 / G26) |
| **4** | **PRM800K** (row 6) — with **MATH** (row 5) as the same ruling | `train.jsonl` → `train`; `test.jsonl` (MATH-500) → `probe`; MATH `test` minus MATH-500 → `refuse` (contaminated by PRM800K train) | PERMISSIVE_OK (MIT, first-party); tier unchanged | The only human step-level labels: E5's battery and E3's negatives natively; the largest human-derivation source in the permissive tier | **Operator ruling on the MATH DMCA** (DIAG §5 item 4): `CC:711-713` / `csd-regions.json:149` vs `CAT`'s PERMISSIVE_OK at the primary LICENSE. Until ruled, both rows stay out; the takedown's claimant is unverified |
| **5** | **LeanDojo Benchmark 4** (row 13) | `train` (`random` split) ; `novel_premises` test → `probe` | **ATTRIBUTION** (CC BY 2.0) — moves `reason` from permissive to attribution tier; the composed model inherits it | Machine-checkable steps and the missing formal-deduction shape | Fetch the Zenodo record's licence file into `licence_upstream_verbatim`; confirm the split names at the primary; declare a Lean-side step-corruption battery before training (E1's battery does not read Lean) |

Not shortlisted and why, in one line each: OpenMathInstruct-1 (more solutions to the
corpus's own questions; holdout anchors inside it; DEC-58 cap) — OpenMathInstruct-2 (Llama
naming obligation on CSD weights, unruled) — MetaMathQA (paraphrases of the holdout;
GPT-3.5 chain) — ProofNet (probe only, no train split) — FOLIO / ARC / HotpotQA (share-alike,
derivation-free as shipped) — LogiQA (no licence at the primary) — BIG-bench (EVAL-ONLY).

---

## 4. Admission steps under the corpus contract

For every shortlisted row, in order. Paths follow what the factory writes today
(`/mnt/bulk/csd-corpus/factory-2026-09-05/visual/google-deepmind__3d-shapes/` is the live
example of the layout) and what `scripts/csd-corpus-admit.py` reads.

1. **Catalogue row.** Add the entry to a dated catalogue under `docs/design/datasets/`
   (`csd-dataset-factory-catalogue/…` schema, `templates/catalogue-entry.template.json`) with
   `faculty: reasoning`, its `provenance_group`, `licence_mirror_tag`, **`licence_upstream_verbatim`
   quoted from the primary URL in `licence_upstream_source`**, `licence_fetch_date`, a
   full-content `grant_scope`, `verdict`, `verification_status: VERIFIED`, and **`role`** set
   per landing (`train` / `probe` / `aux` / `refuse`). A source with an official test split is
   **two rows** or one row with `allow_patterns` scoping, so the test split lands with
   `role: probe` and never under `train`. Any red flag (StrategyQA's evidence, LeanDojo's
   attribution) goes in `provenance_red_flags` with a `provenance_red_flags_resolution`
   entry citing file and line — a flag with no resolution refuses the row.
2. **Plan, then fetch** with the factory, never by hand:
   `dataset-factory plan <catalogue> --dest /mnt/bulk/csd-corpus/factory-2026-09-06 --max-bytes … --max-datasets …`
   and read every refusal before `dataset-factory fetch …`. The landing is
   `/mnt/bulk/csd-corpus/factory-2026-09-06/reasoning/<org__name>/` with `data/` (pinned
   `resolved_revision` sha), `provenance.json` (carrying the `admission` block at policy
   `2026-09-03.r3`), `LICENSE.upstream.txt`, `attribution.json`, `ATTRIBUTION.md`, `SHA256SUMS`.
   The generator (rank 1) is an *action*: its emitted rows land the same way with the
   generator's repo sha, category caps, sampling method and seed in `provenance.json`.
3. **Process stamp.** Emit `processed/<ISO-timestamp>/` with `train.zip` (or `train/`),
   `probe.zip` (or `probe/`), `train_indices.json`, `probe_indices.json`, `manifest.jsonl`,
   `operations.json`, `DISJOINTNESS.json` and a stamp-level `provenance.json`. **Official test
   splits and carved held-out slices are written only to `probe/`.** A superseded stamp gets
   `SUPERSEDED.json`; nothing is deleted.
4. **Adapter gate.** `scripts/csd-corpus-admit.py <landing>/provenance.json reason` and read
   all five checks: licence tier vs `reason`'s declared tier (LeanDojo will report the
   attribution flip), B1 post-admission share ≤ 0.40 by provenance group over `train` roles
   only, `VERIFIED`, the independent structural-refusal read, policy-constant agreement.
5. **Contamination, CC §2.3 stages 0–3**, against the fixed holdout: Stage 1 exact overlap on
   the blake2b-128 whitespace-collapsed lowercased anchor fingerprint against the 512
   `holdout_ids` in `config/mind/splits/reason-ca364a92-split0.json`; Stage 2 the semantic
   near-duplicate screen (mandatory for anything derived from gsm8k `train`); Stage 3 re-run
   against what actually trained. Print the gated-channel counts (DEC-58 clause 2).
6. **Fingerprint and split.** Adding a source changes `corpus.fingerprint`, so a new manifest
   `config/mind/splits/reason-<new fp8>-split0.json` is drawn with `split_seed 0` under
   `csd-split-draw/v1` — **pinning the existing 512 holdout ids** so E4's control and arm score
   the same items (DIAG E0; CC B3.1) — and the batch-order manifest
   `reason-<fp8>-order0-s<steps>-b<batch>.json` is drawn once. G26 refuses training if the split
   sha does not match the fingerprint or a holdout item appears in a batch.
7. **Receipt.** The training receipt stamps `split.manifest`, `split.sha256`, `split.seed`,
   `corpus.cap_sampling.seed`, `train_source_shares` and `eval_source_shares` per source (B3),
   the generator's identity for synthetic rows (DEC-58 clause 1), and the E1 lexical-ceiling
   fields beside the diagonal (DIAG §4 E1).
8. **Licence tier.** After admission, `scripts/csd-publish-checkpoint.py`'s `LICENCE_TIER`
   for `reason` must equal the most restrictive admitted input (`_most_restrictive`); the
   composed model's licence field inherits it. The test
   `test_licence_tier_matches_publish_checkpoint` is the tripwire.

---

## 5. What this pass could not verify

- The MATH DMCA takedown's claimant and grounds (only the HF notice was read).
- StrategyQA's official test-set size (the `wics` mirror's 2,290-row "test" looks duplicated;
  `allenai.org/data/strategyqa` redirects to Semantic Scholar; the leaderboard host did not
  resolve).
- miniF2F's `LICENSE` file (404 on `main` and `master`); its licences are quoted from the README.
- LeanDojo Benchmark's split names and mathlib's Apache-2.0 (taken from the paper and common
  knowledge, not re-fetched).
- BIG-bench per-task licence heterogeneity.
- PRM800K's problem and solution counts (12k / 75k are from the paper, not the README).
- All token counts are estimates from stated character or row counts; only gsm8k and
  aqua_rat token lengths are measured (DIAG §2 [C]).
