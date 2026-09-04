# REASON faculty — dataset survey

Surveyor pass, 2026-09-03. Ground shape follows `00-ground.md` §(b). Method: HF Hub public API
(`/api/datasets?search=`, `/api/datasets/<id>` for `cardData.license`) fetched live this
session, cross-checked against primary sources (GitHub `LICENSE` files, org GitHub pages, HF
dataset cards' own prose) via WebFetch. No dataset content was downloaded — cards, API JSON and
licence pages only. HF token was not needed; the public API answered every query.

`reasoning` today (per ground truth): gsm8k + aqua_rat, 29M on disk, receipt deleted, both
"spent" (already the sole two sources, both already at B1/B2 ceiling for N_eff). This survey
is aimed at breaking that two-source monoculture without adding a NC/BLOCKING term the merged
`memory` faculty hasn't already forced onto the composed model (composed release is already
CC BY-NC-SA per DEC-31, so NC candidates cost nothing marginal — but ND/BLOCKING candidates are
still refused outright per the strictest-input rule, since they'd forbid release entirely
rather than just add NC).

---

## Provenance-group map (important — several candidates below share lineage)

- **Group `openai-gsm8k`**: `openai/gsm8k` only. Already consumed.
- **Group `hendrycks-math`**: `EleutherAI/hendrycks_math` (MATH). Distinct author (Dan
  Hendrycks) and distinct construction (competition-math scrape + human solutions) from GSM8K
  — separate provenance group despite shared author identity, per DEC-46's "shared lineage"
  test (construction pipeline, not surname).
- **Group `aqua-rat`**: `deepmind/aqua_rat` (already consumed) + `allenai/math_qa` (MathQA is
  explicitly built by re-annotating AQuA-RAT — VERIFIED, HF card: "gathered by using a new
  representation language to annotate over the AQuA-RAT dataset") + the `aqua_rat` subset
  inside `TIGER-Lab/MathInstruct`. All three count as ONE source for B1/B2, not three.
- **Group `numina-aops`**: `AI-MO/NuminaMath-CoT` (aops_forum, olympiads, cn_k12, amc_aime,
  synthetic_math subsets) + `open-r1/OpenR1-Math-220k` (DeepSeek-R1 traces generated **over**
  NuminaMath-1.5 problems) + NuminaMath's own `orca_math` subset (which is itself
  `microsoft/orca-math-word-problems-200k`, a fourth entanglement). Treat all of these as one
  group when computing shares against `reasoning`'s existing gsm8k/aqua_rat sources, since
  NuminaMath's own `orca_math` and synthetic subsets already re-include GSM8K/MATH-derived
  problems by the curators' own description.
- **Group `openai-prm`**: `openai/prm800k` only (OpenAI's own first-party release, distinct
  from GSM8K/MATH provenance despite grading MATH problems — it's a separate human-labeling
  effort, own repo, own licence grant).
- **Group `camel-gpt4`**: `camel-ai/math` only, but shares "GPT-4-generated" construction
  method with camel-ai's other subject datasets (physics, chemistry, biology) — if more than
  one camel-ai subject set is ever admitted, they collapse to one group.
- **Group `proofnet`**: `hoskinson-center/proofnet` only.
- **Group `strategyqa`**: `wics/strategy-qa` / `eladsegal/strategyqa` (same underlying AI2
  release, mirror vs. primary of the same artifact — one group by definition).

---

## Candidate catalogue (20 entries)

Format follows `00-ground.md` §(b). `licence_upstream_source` fetch dates are 2026-09-03
(today) for every VERIFIED row below.

### 1. `EleutherAI/hendrycks_math` (MATH)
- upstream: `github.com/hendrycks/math`
- mirror: `huggingface.co/datasets/EleutherAI/hendrycks_math` (also `HuggingFaceH4/MATH-500`,
  `DigitalLearningGmbH/MATH-lighteval` as re-splits — same provenance group)
- `licence_upstream` **VERIFIED** (raw.githubusercontent.com/hendrycks/math/main/LICENSE, fetched
  2026-09-03): `"MIT License / Copyright (c) 2021 Dan Hendrycks / Permission is hereby granted,
  free of charge, ... to deal in the Software without restriction..."`
- mirror_tag: `mit` (matches — no mismatch here)
- verdict: **PERMISSIVE_OK**
- grant_scope: whole_corpus
- provenance_group: `hendrycks-math` (own)
- size: 12,500 problems (7,500 train / 5,000 test), competition-level, full worked solutions
- quality: human-written competition problems + human solutions, well-known contamination
  risk (MATH-500/AIME-style evals are widely trained on — check against `reasoning`'s own eval
  set before admission, per B3)
- consume/emit fit: direct fit — problem+rationale text in, reasoning latents out
- enrichment: none needed structurally; would want de-dup against any eval split reused
  elsewhere in the fleet
- caveat: none beyond standard eval-leakage hygiene

### 2. `openai/prm800k` (mirrored as `tasksource/PRM800K`, `Birchlabs/openai-prm800k-*`)
- upstream: `github.com/openai/prm800k`
- `licence_upstream` **VERIFIED** (raw.githubusercontent.com/openai/prm800k/main/LICENSE, fetched
  2026-09-03): `"MIT License ... Copyright (c) 2023 OpenAI"` — OpenAI's own first-party release
  of its own human-labeling effort, not GPT output, so none of the "OpenAI ToS forbids training
  competing models on API output" concern applies (there is no API output here — every step
  label is human-graded).
- mirror_tag: `mit` (matches)
- verdict: **PERMISSIVE_OK**
- grant_scope: whole_corpus
- provenance_group: `openai-prm` (own)
- size: ~800K step-level human correctness labels over MATH-problem solution steps
- quality: highest — human-graded, step-level, purpose-built for process supervision
- consume/emit fit: excellent for training a REASON-internal step-verifier / rationale-quality
  signal, not just next-step generation
- enrichment: reshape step labels into (problem, step, label) triples; no licence consequence
  (still PERMISSIVE_OK downstream)
- caveat: solutions graded are MATH-derived (candidate #1's provenance group) — the step labels
  are OpenAI's own artifact but the underlying problems overlap `hendrycks-math`; count problem
  text once for B1 purposes if both are admitted

### 3. `open-web-math/open-web-math`
- upstream: project page / paper (Paster et al., "OpenWebMath")
- `licence_upstream` **VERIFIED** (huggingface.co/datasets/open-web-math/open-web-math card,
  fetched 2026-09-03): `"OpenWebMath is made available under an ODC-By 1.0 license; users
  should also abide by the CommonCrawl ToU"`
- mirror_tag: none set on `cardData.license` (blank on the API — mismatch: the licence lives
  only in card prose, not the structured field a fetcher would read; flag this as exactly the
  kind of mirror-lies gap the ground doc warns about)
- verdict: **ATTRIBUTION** (ODC-By requires attribution; ND/SA not present, but it is not
  unconditional PERMISSIVE_OK either)
- grant_scope: whole_corpus, but sourced from CommonCrawl — each underlying page's own
  copyright is untouched; ODC-By licenses OpenWebMath's *compilation/database rights*, not a
  blanket grant over every crawled page's prose. This is a real nuance, not a mirror lie:
  flag as `provenance_red_flags: "database-right licence over crawled pages of unknown
  individual-page copyright; standard for CC-derived pretraining corpora but worth naming"`
- provenance_group: `open-web-math` (own — CommonCrawl-mined, distinct from every olympiad/
  competition source above)
- size: ~6.3M documents, ~14.7B tokens — this is the volume candidate; nothing else on this
  list is remotely this size
- quality: filtered/deduped math-heavy CommonCrawl subset (LaTeX-aware extraction), mixed
  quality (forum posts, course notes, wikis) vs. curated competition sets — good for the
  `language`-adjacent math-fluency floor, weaker for rationale supervision specifically since
  most documents are prose, not Q→A pairs
- consume/emit fit: best as pretraining/vocabulary material for `numeric/math` (still
  PLACEHOLDER) or the language trunk's deferred official-docs-style corpus, secondary fit for
  `reasoning` itself (would need QA-pair extraction, not raw use)
- enrichment: mining (problem, solution) pairs out of raw documents requires either heuristic
  extraction or model-assisted parsing — attribution obligation propagates to the derived set
  (ODC-By is not extinguished by extraction)
- caveat: `grant_scope` nuance above; large enough that a cap/sample (B4) is mandatory, never a
  prefix

### 4. `camel-ai/math`
- upstream: `github.com/camel-ai/camel` (framework repo; dataset card itself is the primary
  source for the data-specific terms, since the code repo's Apache-2.0 tag covers the
  *framework*, not the generated data)
- `licence_upstream` **VERIFIED** (huggingface.co/datasets/camel-ai/math card, fetched
  2026-09-03): `cc-by-nc-4.0`, with card prose: `"This data was synthetically generated by
  GPT4 and might contain incorrect information. The dataset is there only for research
  purposes."`
- mirror_tag: `cc-by-nc-4.0` (matches — this is a case where mirror and upstream agree, unlike
  #3)
- verdict: **NC** (per DEC-31, NC does not block; the "research purposes" sentence is the
  dataset card's own quality disclaimer, not a separate legal term beyond the stated CC
  BY-NC-4.0 grant — treat the licence field as controlling, but log the sentence as a
  provenance red flag for a human to re-read before commit)
- grant_scope: whole_corpus
- provenance_group: `camel-gpt4` (own; would merge with any other camel-ai subject set later)
- size: ~50K GPT-4-generated math problems with worked solutions
- quality: GPT-4-generated (not human), unverified correctness at scale ("might contain
  incorrect information" is the rights holder's own words) — usable but should not be treated
  as ground-truth-clean without a correctness pass
- consume/emit fit: rationale style/diversity source, not a primary-truth source
- enrichment: would want an automated correctness filter (e.g. execute/verify against a
  symbolic solver) before training use; NC propagates to any enriched derivative (strictest-
  input rule)
- caveat: GPT-4-generated — flagged per task instructions as "derived from a restricted
  source" in the sense that OpenAI's usage policy nominally restricts using its outputs to
  train competing models; camel-ai (a third party, not OpenAI) chose to release this under an
  explicit NC licence of their own, which is the operative grant for redistribution purposes,
  but the underlying OpenAI-ToS question is a real unresolved tension for the *generation*
  step, not the *redistribution* step — same shape as MetaMathQA/orca-math below, logged
  consistently

### 5. `hoskinson-center/proofnet`
- upstream: `github.com/zhangir-azerbayev/ProofNet` (Hoskinson Center for Formal Mathematics)
- `licence_upstream` **VERIFIED** (raw.githubusercontent.com/zhangir-azerbayev/ProofNet/main/
  LICENSE, fetched 2026-09-03): `"MIT License ... Permission is hereby granted, free of
  charge..."`
- mirror_tag: `mit` (matches)
- verdict: **PERMISSIVE_OK**
- grant_scope: whole_corpus
- provenance_group: `proofnet` (own)
- size: 371 problems (informal-formal pairs) from undergrad pure-math textbooks, small
- quality: human-written formal (Lean) + informal statement/proof pairs — the highest-rigor,
  lowest-volume item on this list; directly answers the task's "proof corpora" ask
- consume/emit fit: too small to move B1/B2 shares by itself but valuable as a distinct
  reasoning *style* (formal proof, not word-problem arithmetic) — good NSRS reserve material
  (cross-faculty: math + formal-language) rather than primary training volume
- enrichment: none needed; could pair informal statement with CoT-style informal proof as a
  rationale-style target
- caveat: eval-set risk — ProofNet's test split is a known formal-math benchmark, check for
  leakage before any training use

### 6. `allenai/math_qa` (MathQA)
- upstream: `math-qa.github.io` (Amini et al., ACL 2019) / AI2
- `licence_upstream` **VERIFIED** (huggingface.co/datasets/allenai/math_qa card, fetched
  2026-09-03): `"The dataset is licensed under the Apache License, Version 2.0"`
- mirror_tag: `apache-2.0` (matches)
- verdict: **PERMISSIVE_OK**, but see provenance note
- grant_scope: whole_corpus
- provenance_group: **`aqua-rat`** — NOT a new source. MathQA is AQuA-RAT re-annotated with
  operation programs (VERIFIED, card: "gathered by using a new representation language to
  annotate over the AQuA-RAT dataset"). Admitting this alongside the already-consumed
  `deepmind/aqua_rat` does **not** improve N_eff — it's the same provenance group wearing a
  different annotation layer.
- size: 37,297 problems
- quality: adds explicit operation-program annotations (program-of-thought shape) that raw
  AQuA-RAT lacks — the annotation is the value-add, not new problem text
- consume/emit fit: strong fit specifically for program-of-thought supervision (the task's PoT
  ask), weak fit as a B1/B2 diversifier
- enrichment: none — annotations are already present
- caveat: **do not double-count** against `aqua_rat` in any B1/B2 receipt; if `reasoning`
  wants PoT signal, prefer swapping MathQA in *for* raw aqua_rat rather than adding both

### 7. `open-r1/OpenR1-Math-220k`
- upstream: `huggingface.co/open-r1` org (Hugging Face's own open-r1 reproduction project)
- `licence_upstream` **VERIFIED** (huggingface.co/datasets/open-r1/OpenR1-Math-220k card,
  fetched 2026-09-03): `"The dataset is licensed under Apache 2.0"`
- mirror_tag: `apache-2.0` (matches)
- verdict: **PERMISSIVE_OK**
- grant_scope: whole_corpus
- provenance_group: **`numina-aops`** (VERIFIED, card: "reasoning traces generated by DeepSeek
  R1 for problems from NuminaMath 1.5") — the *problems* are NuminaMath's, the *solutions* are
  DeepSeek-R1-generated
- size: ~220K (400K generated, filtered to verified-correct subset per the card)
- quality: long CoT traces from a strong open-weights reasoner (DeepSeek-R1, not a
  closed-API model — no OpenAI-ToS tension here, unlike MetaMathQA/camel/orca-math), filtered
  for verified correctness — genuinely strong rationale quality
- consume/emit fit: excellent for long-chain reasoning-latent supervision
- enrichment: none needed
- caveat: problem-text provenance overlaps NuminaMath/aops_forum group (see provenance map) —
  if NuminaMath-CoT itself is ever admitted, only count problem-text share once; the R1 traces
  are a distinct annotation layer analogous to MathQA vs. AQuA-RAT

### 8. `wics/strategy-qa` (mirror of AI2's StrategyQA, primary repo `eladsegal/strategyqa`)
- upstream: `github.com/eladsegal/strategyqa` (AI2 / Tel Aviv University, Geva et al. 2021)
- `licence_upstream` **VERIFIED** (github.com/eladsegal/strategyqa repo licence badge, fetched
  2026-09-03): `MIT license`
- mirror_tag: `other` (on the `wics/strategy-qa` HF card) — **mismatch**: mirror says `other`,
  primary source says MIT. Logged exactly per the dataset-mirror-licences-lie precedent; the
  upstream text controls.
- verdict: **PERMISSIVE_OK** (per upstream, overriding the vaguer mirror tag)
- grant_scope: whole_corpus
- provenance_group: `strategyqa` (own)
- size: 2,780 examples with decomposition + implicit multi-hop reasoning steps + supporting
  Wikipedia paragraphs
- quality: human-written multi-hop *implicit*-reasoning questions (the "did Aristotle use a
  laptop"-style task) with gold decomposition annotations — directly answers the task's
  "multi-hop QA rationales" ask, and it's a genuinely different reasoning *shape* (implicit
  strategy inference) than the arithmetic-heavy math sets above
- consume/emit fit: good diversifier away from pure math, small enough to be a minority-share
  stratum rather than a scaling source
- enrichment: decomposition steps could be rendered as CoT-style rationale text
- caveat: small; won't move B1/B2 shares alone, best paired with a larger diversifier

---

## Notable non-top-8 entries (surveyed, not recommended as-is)

- **`AI-MO/NuminaMath-CoT`** — Apache-2.0 VERIFIED at the mirror card itself, but its own
  subset breakdown re-includes `orca_math` (= Microsoft's GPT-4o-generated set, see below) and
  a 30,201-row `aops_forum` subset scraped from Art of Problem Solving community posts, whose
  underlying copyright ownership by individual forum posters is not addressed in anything
  fetched this session — flagged `provenance_red_flags: "aops_forum subset's underlying
  authorship/licence chain to individual forum posters not established by anything surveyed"`.
  Not refused outright (Apache-2.0 is a real grant from a legitimate distributor, AI-MO), but
  held at **NC/PERMISSIVE_OK-with-caveat** pending a dedicated pass on the aops_forum subset
  specifically, rather than admitted whole.
- **`meta-math/MetaMathQA`** — mirror tag `mit`, but HF card prose states augmentation is
  "sourced from ChatGPT 3.5" over GSM8K/MATH problems — this is the clearest "GPT-generated
  data under OpenAI terms" case the task asked to flag. MIT is meta-math's own grant over
  *their* compiled dataset, but the OpenAI usage-policy tension (no training competing models
  on API output) attaches to the generation step, same shape as camel-ai/orca-math but without
  even camel-ai's NC acknowledgment — meta-math asserts MIT outright over ChatGPT-3.5 output.
  Held at **NC** (not REFUSE — no ND/no-redistribution term actually appears anywhere in
  anything fetched; the OpenAI-ToS tension is a policy risk to log, not a licence grant this
  survey found blocking) pending an explicit operator call on whether "asserts MIT over
  GPT-3.5-derived text" clears the bar alone.
- **`microsoft/orca-math-word-problems-200k`** — same shape as MetaMathQA: `mit` tag, but
  generated via Azure OpenAI GPT4 Turbo. Microsoft is itself an OpenAI commercial partner
  asserting MIT over the output, which is a stronger position than a third party doing the
  same, but the underlying tension is identical. Held at **NC**, same reasoning as above.
- **`TIGER-Lab/MathInstruct`** — mirror tag `mit` **for the whole aggregate**, but its own HF
  card lists its component licences individually and names `camel_math` as
  `"Attribution-NonCommercial 4.0 International"` within an otherwise MIT/Apache-2.0 mix. This
  is a live instance of exactly the "mirror lies" pattern the ground doc names: a single
  blanket `license: mit` tag on the aggregate masks an NC component. Per the strictest-input
  rule, consuming MathInstruct whole makes the derived set NC, not MIT, regardless of the tag.
  **Verdict: NC-if-whole, PERMISSIVE_OK-if-camel_math-subset-excluded.** Not placed in the
  top 8 because it is entirely reducible to its already-listed components (gsm8k, aqua_rat/
  MathQA, MATH, TheoremQA, camel-ai/math, NumGLUE) — better to admit those directly with
  correct per-source licence bookkeeping than to admit the aggregate and inherit its blended
  tag.
- **`EleutherAI/proof-pile-2`** — `cardData.license` is **unset** on the mirror (another blank
  structured-field case). Composed of OpenWebMath (ODC-By, candidate #3, already counted),
  AlgebraicStack (mixed per-repo code licences, unaudited this session) and an ArXiv subset
  (bulk arXiv full-text redistribution carries its own constraints beyond a simple licence
  string — arXiv's non-exclusive licence from authors does not straightforwardly grant
  third-party bulk redistribution rights). **UNVERIFIED**, held out of the top 8 pending a
  dedicated arXiv-bulk-rights check; recommend sourcing OpenWebMath directly (#3) instead of
  through this composite.
- **`nvidia/Nemotron-Math-Proofs-v1`** — mirror tag `cc-by-sa-4.0`, not verified against a
  primary NVIDIA source this session. **UNVERIFIED**, plausible SHARE_ALIKE candidate for a
  future pass.
- **`tasksource/proofwriter`** — `cardData.license` unset; primary AI2 ProofWriter repo URL
  guessed this session (`github.com/allenai/proofwriter`) 404'd — the correct primary URL
  wasn't found in this pass. **UNVERIFIED**, not fetched successfully; needs a follow-up
  search rather than a guessed URL.
- **`maveriq/bigbenchhard`** (BIG-Bench-Hard) — repo-level licence for `google/BIG-bench`
  **VERIFIED** (raw.githubusercontent.com/google/BIG-bench/main/LICENSE, fetched 2026-09-03):
  Apache License 2.0. BBH itself is a curated 23-task hard subset of BIG-bench maintained by
  Suzgun et al., distributed inside the same repo — treated as covered by the same Apache-2.0
  grant (VERIFIED at repo level, not fetched at the BBH-subdirectory level specifically).
  **Verdict: PERMISSIVE_OK.** Not in the top 8 only because BBH is an eval suite (23 curated
  hard tasks) more than a training corpus — high B3 leakage risk if used for anything but
  held-out evaluation; flagged as an **EVAL-ONLY** candidate for `reasoning`'s own held-out
  benchmark, not as training volume.

## REFUSED

- **Multi-provider synthetic aggregates** (pattern seen in this session's search results,
  e.g. `AMAImedia/NOESIS-*-reasoning-router-code-math-psych-opus47-deepseek4-qwen36-gemini31-
  r1-gpt54`) — filenames alone declare outputs blended from at least five proprietary/hosted
  model APIs (Claude Opus, DeepSeek, Qwen, Gemini, GPT). **REFUSE-TERMS**: no single upstream
  licence page was fetched (none exists to fetch — this is a downstream aggregator, not a
  rights holder), and blending outputs from five different providers' terms of service
  compounds rather than resolves the redistribution question; no verdict short of REFUSE is
  defensible without a per-provider ToS review this survey did not attempt.
- **Any dataset with an empty/`None` `cardData.license` AND no primary-source page fetched
  successfully this session** (`tasksource/proofwriter` above is the concrete instance) —
  per the operator's REFUSE-by-default posture for "no licence grant exists", these are logged
  UNVERIFIED and treated as refused-for-now, not silently admitted on the strength of download
  count. Re-surveying with the correct primary URL is a follow-up task, not a licence finding.
- **`zh-plus/tiny-imagenet`-style "distributor never owned the base corpus" pattern** — not
  hit directly in this REASON-faculty pass (no math/reasoning candidate surveyed shares that
  specific shape), but the `aops_forum` subset flag under NuminaMath-CoT above is the closest
  analogue found and is held at the same caution level rather than a clean admit.

---

## Summary table — top 8

| # | id | verdict | provenance_group | size | licence status |
|---|---|---|---|---|---|
| 1 | EleutherAI/hendrycks_math (MATH) | PERMISSIVE_OK | hendrycks-math | 12,500 | VERIFIED |
| 2 | openai/prm800k | PERMISSIVE_OK | openai-prm | ~800K step labels | VERIFIED |
| 3 | open-web-math/open-web-math | ATTRIBUTION | open-web-math | 6.3M docs / 14.7B tok | VERIFIED |
| 4 | camel-ai/math | NC | camel-gpt4 | ~50K | VERIFIED |
| 5 | hoskinson-center/proofnet | PERMISSIVE_OK | proofnet | 371 | VERIFIED |
| 6 | allenai/math_qa (MathQA) | PERMISSIVE_OK (same group as aqua_rat — no B2 gain) | aqua-rat | 37,297 | VERIFIED |
| 7 | open-r1/OpenR1-Math-220k | PERMISSIVE_OK | numina-aops | ~220K | VERIFIED |
| 8 | wics/strategy-qa | PERMISSIVE_OK (upstream overrides mismatched mirror tag) | strategyqa | 2,780 | VERIFIED |

Composed-set consequence if all 8 are admitted alongside the existing gsm8k+aqua_rat: strictest
input is `camel-ai/math`'s NC — no change to the composed model's release tier, since `memory`
already forces CC BY-NC-SA per DEC-31. Distinct provenance groups rise from 2
(`openai-gsm8k`, `aqua-rat`) to 7 (`openai-gsm8k`, `aqua-rat`, `hendrycks-math`, `openai-prm`,
`open-web-math`, `camel-gpt4`, `proofnet`, `numina-aops`, `strategyqa` — 9 counting all;
`allenai/math_qa` doesn't add one), which clears B2's `N_eff ≥ 3` floor with real headroom
provided no single group is allowed to dominate share (B1 ≤ 0.40 still needs enforcing at
mixing time, not assumed from group count alone).
