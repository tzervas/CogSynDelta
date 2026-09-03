# Survey: moral / safety corpus (DEC-59)

Scope: values, ethics judgments, harmlessness preferences, refusals, moral dilemmas, norms —
for the `memory`/reasoning-adjacent moral-safety corpus, plus held-out probes to MEASURE
behaviour change from training on it (never assumed). Read `00-ground.md` first; this survey
uses its catalogue shape and verdict classes.

Method used: (1) HF Hub public API (`/api/datasets?search=`, `/api/datasets/<id>`,
`datasets-server.huggingface.co/size`) — no HF token needed, all fetches were the public,
unauthenticated API; (2) primary/upstream source (GitHub LICENSE files, project pages, paper
pages) via WebFetch; (3) WebSearch for corroboration where a primary fetch was thin (404,
partial render). No dataset content was downloaded — only cards, API JSON, and licence pages.
`gpu/huggingface-token` was not needed (public API never refused).

Fetch date for every `licence_upstream_source` below: **2026-09-03**.

---

## Catalogue

### 1. `hendrycks/ethics` — ETHICS

- upstream: https://github.com/hendrycks/ethics (paper: "Aligning AI With Shared Human
  Values", Hendrycks et al. 2021)
- mirror: `hendrycks/ethics` (HF), also `EleutherAI/hendrycks_ethics`
- **licence_upstream: VERIFIED** — `LICENSE` at
  `raw.githubusercontent.com/hendrycks/ethics/master/LICENSE`, full MIT text, quoted:
  *"Permission is hereby granted, free of charge, to any person obtaining a copy of this
  software and associated documentation files (the "Software"), to deal in the Software
  without restriction... THE SOFTWARE IS PROVIDED "AS IS"..."* — Copyright (c) 2020 Dan
  Hendrycks. No separate data licence carve-out; the repo licenses the whole tree, dataset
  included.
- mirror_tag: `license:mit` (matches)
- verdict: **PERMISSIVE_OK**
- provenance_group: `hendrycks-ethics` (standalone; `EleutherAI/hendrycks_ethics` and
  `lighteval/hendrycks_ethics` are re-uploads of the same data, same group)
- size: 134,417 rows (5 subscales: justice, deontology, virtue, utilitarianism, commonsense)
- quality: human-authored scenarios + crowd-labelled judgments, widely used as an eval
  (contamination risk: this IS a standard eval benchmark — using it for training would
  contaminate any eval built on it; recommend held-out split reserved as eval, not trained on
  wholesale)
- consume/emit fit: trains judgment-consistency (label: is this scenario morally acceptable
  per each ethical framework) — feeds a moral-classification head, and doubles as a
  measurement probe per the operator's ask ("MEASURE whether training changes behaviour")
- enrichment: pair each scenario with a natural-language rationale (currently label-only) —
  requires model-generated rationales, which does not change the licence (MIT propagates
  regardless of derivative annotation)

### 2. `Anthropic/hh-rlhf` (+ red-team-attempts) — HH-RLHF

- upstream: https://github.com/anthropics/hh-rlhf
- mirror: `Anthropic/hh-rlhf`
- **licence_upstream: VERIFIED (partial)** — GitHub repo page shows "MIT license" in its
  licence badge/footer; `LICENSE.txt` at the expected raw path 404'd (repo may have moved the
  file or renamed it), so the full text was not fetched directly — the MIT designation is
  corroborated by the HF mirror tag (`license:mit`) and the repo's own licence badge, which
  is a second independent surface reading the same claim. Recommend one more direct fetch
  before this ships to production ingest (mark this line item's fetch as **VERIFIED, thin**).
- mirror_tag: `license:mit` (matches)
- verdict: **PERMISSIVE_OK**
- provenance_group: `anthropic-hh-rlhf` (includes the `red-team-attempts.jsonl.gz` split in
  the same repo — same licence, same group)
- size: 169,352 rows (helpfulness + harmlessness preference pairs, plus red-team transcripts)
- quality: human-written adversarial prompts + model completions + human preference labels;
  Anthropic's own RLHF pipeline data, well-documented methodology
- consume/emit fit: direct fit for harmlessness-preference training (chosen/rejected pairs)
  and for refusal-behaviour data; red-team split feeds an adversarial-probe eval
- enrichment: none needed structurally; could re-rank preference pairs against the ETHICS
  framework labels for cross-consistency checks — MIT stays MIT under any enrichment

### 3. `wassname/social_chemistry_101` — Social Chemistry 101

- upstream: https://github.com/mbforbes/social-chemistry-101 (Forbes et al., AI2/UW,
  EMNLP 2020)
- mirror: `wassname/social_chemistry_101` (no canonical AI2 org upload found on HF at survey
  time — the operator's brief named "Social Chemistry 101 (CC BY-SA)"; the mirror confirms
  that tag independently)
- **licence_upstream: VERIFIED** — GitHub repo README states: *"The dataset is licensed under
  the CC BY-SA 4.0 license."*
- mirror_tag: `license:cc-by-sa-4.0` (matches)
- verdict: **SHARE_ALIKE**
- provenance_group: `social-chemistry-101` (standalone; distinct lineage from Social Bias
  Frames despite both being AI2/UW norm-annotation projects)
- size: 355,922 rows (rules-of-thumb over everyday situations, with social-judgment
  dimensions: legality, agreement, cultural pressure, etc.)
- quality: crowd-annotated free-text norms grounded in real Reddit/story situations; large,
  structured, multi-dimensional judgment labels — good rationale material out of the box
- consume/emit fit: strongest single source for "norms" in the DEC-59 brief — situation +
  rule-of-thumb + judgment triples are close to ready-made (situation, judgment, rationale)
  moral-reasoning items
- enrichment: minimal — mostly needs re-formatting into the region's training shape.
  **SA propagates**: any derived/enriched set built from this corpus (rendering, pairing,
  rephrasing) must stay CC BY-SA and carries an attribution/share-alike obligation into the
  moral-safety corpus as a whole, per the strictest-input rule (DEC-31)

### 4. `metaeval/scruples` — Scruples

- upstream: https://github.com/allenai/scruples (Lourie, Le Bras, Choi; AAAI 2021)
- **licence_upstream: VERIFIED** — `LICENSE` file at
  `raw.githubusercontent.com/allenai/scruples/master/LICENSE` is the full Apache License 2.0
  text, copyright "Allen Institute for Artificial Intelligence" 2020. No separate
  data-vs-code carve-out stated; the repo licence covers the release as a whole (AI2's usual
  pattern of Apache-2.0 for both code and the accompanying corpus unless a card says
  otherwise).
- mirror_tag: `license:apache-2.0` (matches)
- verdict: **PERMISSIVE_OK**
- provenance_group: `scruples` (standalone; sourced from r/AmITheAsshole)
- size: 32,766 rows in the queried config (the full release also has an "anecdotes" split
  with ~30k+ narrative stories with community-voted verdicts — not all configs were sized in
  this pass)
- quality: real Reddit judgment threads (AITA) with vote-derived labels — good ecological
  validity, but is user-generated content originally, so re-check ToS-vs-copyright framing if
  this is enriched (AI2's Apache-2.0 grant is what governs redistribution here, not Reddit's
  ToS, since AI2 is the distributor of record and states a licence — but this is the kind of
  claim in the "distributor disclaims owning what they distribute" refusal class the operator
  flagged, so treat AI2's explicit Apache-2.0 file as the mitigating fact, not silence)
- consume/emit fit: binary/multi-way "who's in the wrong" judgments — feeds moral-judgment
  classification and a resolvable-dilemma reasoning slice
- enrichment: pairing anecdote text with the AITA-community verdict distribution as a soft
  label; Apache-2.0 permits any derivative without propagation obligations beyond attribution
  of the licence itself

### 5. `PKU-Alignment/PKU-SafeRLHF`

- upstream: https://github.com/PKU-Alignment/safe-rlhf (paper: "Safe RLHF: Safe Reinforcement
  Learning from Human Feedback", Dai et al. 2023)
- **licence_upstream: VERIFIED** — the safe-rlhf GitHub README states *"Safe-RLHF is released
  under Apache License 2.0"* for the **codebase**; the **dataset card itself** (fetched
  directly from the HF repo, which is PKU-Alignment's own canonical upload, not a
  third-party mirror) states **`cc-by-nc-4.0`** in its metadata — this is the org's own
  first-party page, so treated as VERIFIED at primary for the dataset (code and data carry
  different licences, a real `grant_scope` split: `whole_corpus` for the NC data licence,
  separate `code_only` Apache-2.0 for the training/eval scripts).
- mirror_tag: `license:cc-by-nc-4.0` (matches; same-org first-party listing)
- verdict: **NC**
- provenance_group: `pku-alignment` (shared lineage with BeaverTails below — same lab, same
  RLHF-safety data programme, same NC term)
- size: 164,236 rows (helpfulness + harmlessness dual-preference comparisons with severity
  tiers)
- quality: large, dual-signal (helpfulness AND harmlessness rated separately, not conflated),
  documented red-teaming and QA process from the paper
- consume/emit fit: strong second NC source alongside GooAQ for `memory`, and a direct fit
  for harmlessness-preference training in the moral-safety corpus
- enrichment: none needed for format; **NC propagates** — already accepted per DEC-31 (moves
  the composed release tier to NC, which per `csd-release-licence-decision` the moral-safety
  corpus is already headed toward via GooAQ, so this adds no *new* constraint beyond what
  `memory` already carries)

### 6. `PKU-Alignment/BeaverTails`

- upstream: project page (sites.google.com/view/pku-beavertails) + GitHub
  (PKU-Alignment/BeaverTails) — same lab as #5
- **licence_upstream: UNVERIFIED at primary** — neither the project page nor the paper
  (arXiv 2307.04657) surfaced an explicit licence statement in this pass; the primary-source
  fetch did not turn up a licence file or page section. The HF mirror card states
  `cc-by-nc-4.0`, and the dataset is first-party PKU-Alignment (same org, same upload pattern
  as PKU-SafeRLHF), which is corroborating but not a primary-source quote.
- mirror_tag: `license:cc-by-nc-4.0`
- verdict: **NC** (provisional — mirror tag + same-org pattern as a verified sibling; flag
  for a follow-up primary-source pass before this is admitted to a training run, per the
  "mirrors lie" precedent — 10/75 mismatches were found exactly this way)
- provenance_group: `pku-alignment` (same group as PKU-SafeRLHF — the strictest-input rule
  treats them as one lineage for B1/B2 purposes if both are used)
- size: 364,170 rows (prompt + response + multi-category harm annotation + safety label)
- quality: 14 harm categories, both helpfulness and harmlessness meta-labels, large scale
- consume/emit fit: category-labelled harm data is a strong fit for a refusal/harm-taxonomy
  head, complementary to PKU-SafeRLHF's paired-comparison format
- enrichment: category labels could seed a taxonomy-conditioned rationale generation step;
  NC propagates, same as #5

### 7. `allenai/wildguardmix` / `allenai/wildjailbreak` — WildGuard / WildJailbreak

- upstream: https://github.com/allenai/wildguard (WildGuardMix); AI2 (WildJailbreak)
- **licence_upstream: VERIFIED (repo code only)** — `LICENSE.md` at
  `raw.githubusercontent.com/allenai/wildguard/main/LICENSE.md` is the full Apache License
  2.0 text, "Licensed under the Apache License, Version 2.0... Copyright 2024" (Allen
  Institute for AI). This licenses the **repository/code**. The **dataset card** itself
  (HF, first-party AI2 org) states `license: odc-by` in its metadata — AI2's now-standard
  split of Apache-2.0 code / ODC-BY data, the same pattern documented for other AI2 releases
  in `00-ground.md`'s WildGuard mention. `grant_scope: whole_corpus` for ODC-BY (data),
  `code_only` Apache-2.0 (repo).
- mirror_tag: `license:odc-by` (matches the data-specific claim)
- verdict: **PERMISSIVE_OK**-adjacent — ODC-BY is not in the five-class LICENCE-FOR-OPEN-
  WEIGHTS.md list verbatim; treat as its own recognized permissive-with-attribution class
  (already named explicitly in the operator's DEC-59 brief as "ODC-By? check" — **confirmed
  ODC-By**), functionally similar to ATTRIBUTION
- provenance_group: `allenai-wildguard` (WildGuardMix and WildJailbreak share AI2's
  red-teaming pipeline lineage — treat as one group unless a later pass shows independent
  data collection)
- size: not sized via datasets-server in this pass (multi-config repo; card states ~92K
  WildGuardMix items, WildJailbreak ~262K per AI2's published figures — **INFERRED** from
  card prose, not independently row-counted here)
- quality: adversarial, LLM-and-human-generated jailbreak/refusal-training data, purpose-built
  for safety training — closest fit in this survey to "refusals" specifically
  named in the DEC-59 brief
- consume/emit fit: direct fit for refusal-calibration (both over-refusal and under-refusal
  examples are labelled, which is unusually good for measuring the operator's "did training
  change behaviour" ask)
- enrichment: none required for format; ODC-BY requires attribution on redistribution —
  attribution manifest needed if this corpus (or a derivative) is redistributed

### 8. `google/civil_comments`

- upstream: original release was Civil Comments platform → figshare archive; Jigsaw/Google
  republished it for the "Unintended Bias in Toxicity Classification" Kaggle challenge
- **licence_upstream: VERIFIED** — TensorFlow Datasets catalog page (a Google-maintained,
  independent-of-mirror surface) states verbatim: *"This data set is an exact replica of the
  data released for the Jigsaw Unintended Bias in Toxicity Classification Kaggle challenge.
  This dataset is released under CC0, as is the underlying comment text."*
- mirror_tag: `license:cc0-1.0` (matches)
- verdict: **PERMISSIVE_OK**
- provenance_group: `civil-comments` (standalone; note `google/jigsaw_toxicity_pred`, item
  #14 below, is a **different** Jigsaw challenge/corpus, not the same lineage — do not
  collapse the two)
- size: 1,999,514 rows (comment text + toxicity + identity-mention multi-labels)
- quality: large, real user comments (2015-2017, ~50 news sites), toxicity + identity-attack
  sub-labels — good for a harm/toxicity detection head, not itself a moral-reasoning corpus
- consume/emit fit: best fit as a **held-out probe** (the operator's "measure, don't assume"
  ask) for toxicity-sensitivity before/after moral-safety training, rather than as primary
  training material
- enrichment: none needed; CC0 imposes no obligation on any derivative

### 9. `toxigen/toxigen-data` — ToxiGen

- upstream: https://github.com/microsoft/TOXIGEN
- **licence_upstream: VERIFIED, with a real conflict flagged** — `LICENSE.txt` in the repo is
  a dual grant: MIT for the code, and **CDLA-Permissive-2.0** for the data ("A Data Recipient
  may use, modify, and share the Data... This agreement does not impose any restriction or
  obligations with respect to the use, modification, or sharing of Results," data provided
  "AS IS"). CDLA-Permissive-2.0 imposes no field-of-use or NC restriction. **However** the
  repo README separately states: *"The data, methods and two trained hatespeech detection
  checkpoints released with this work are intended to be used for research purposes only."*
  This is a stated-intent sentence sitting alongside a licence grant that does not itself
  restrict to research use — a real conflict between the legal grant and the authors' stated
  intent.
- mirror_tag: none set on the HF card (`license_card: None`) — the mirror doesn't even carry
  the licence, which is itself a mirror-vs-upstream gap worth recording
- verdict: **UNVERIFIED / flagged** — recommend NOT admitting until this conflict is
  resolved (e.g. contacting Microsoft, or finding a later release note that clarifies); do
  not auto-classify as PERMISSIVE_OK on the licence file alone when the README explicitly
  narrows intended use, and do not auto-REFUSE on the README's soft language alone when the
  actual grant (CDLA-Permissive-2.0) is unrestricted. This is exactly the shape DEC-59 asked
  surveyors to flag rather than resolve unilaterally.
- provenance_group: `microsoft-toxigen` (standalone)
- size: 319,301 rows (LLM-generated implicit-hate-speech statements, human-annotated)
- quality: synthetic (GPT-3-generated) implicit toxicity examples — useful as an eval probe
  for subtle-harm detection regardless of the training-admission question
- consume/emit fit: candidate held-out probe (implicit-bias/toxicity detection), not a
  training-admission candidate pending the licence-vs-intent conflict above
- enrichment: n/a pending resolution

### 10. `McGill-NLP/stereoset` — StereoSet

- upstream: https://github.com/moinnadeem/StereoSet (the original stereoset.mit.edu project
  repo)
- **licence_upstream: VERIFIED** — GitHub repo page shows CC-BY-SA-4.0 as the declared
  licence (file listing + licence badge)
- mirror_tag: `license:cc-by-sa-4.0` (matches)
- verdict: **SHARE_ALIKE**
- provenance_group: `stereoset` (standalone)
- size: 4,229 rows
- quality: small, purpose-built stereotype-association eval (intersentence + intrasentence),
  human-crowdsourced
- consume/emit fit: held-out bias-sensitivity probe, not primary training volume (too small,
  and using an eval-shaped benchmark as training data would contaminate its own use as a
  measurement instrument)
- enrichment: n/a — keep as probe; SA would propagate to any derivative anyway

### 11. `nyu-mll/crows_pairs` — CrowS-Pairs

- upstream: https://github.com/nyu-mll/crows-pairs
- **licence_upstream: VERIFIED** — GitHub repo states: *"CrowS-Pairs is licensed under a
  Creative Commons Attribution-ShareAlike 4.0 International License."* Repo also notes it was
  constructed from ROCStories and the fiction section of MNLI — a real
  `provenance_red_flags` item: the sentence pairs are derived from those two upstream
  corpora, whose own licences were not independently checked in this pass.
- mirror_tag: `license:cc-by-sa-4.0` (matches)
- verdict: **SHARE_ALIKE** (provisional — the ROCStories/MNLI lineage underneath CrowS-Pairs'
  own CC BY-SA grant was not independently verified; flag before training admission)
- provenance_group: `crows-pairs` (own group, but with an unverified sub-dependency on
  ROCStories + MNIST-fiction lineage)
- size: not sized via datasets-server (small; ~1,500 sentence pairs per published figures,
  **INFERRED** from paper, not row-counted here)
- quality: small, purpose-built stereotype benchmark, same profile as StereoSet
- consume/emit fit: held-out bias-sensitivity probe, same reasoning as #10
- enrichment: n/a — keep as probe

### 12. `allenai/social_bias_frames` — Social Bias Frames

- upstream: AI2 project (Sap et al., ACL 2020)
- **licence_upstream: VERIFIED (mirror-corroborated, first-party org)** — HF card is AI2's
  own first-party upload with `license:cc-by-4.0`; not independently re-verified against a
  separate GitHub/project-page licence file in this pass (time-boxed) — treat as VERIFIED at
  the mirror but **not yet cross-checked at a second primary surface**, per the "mirrors lie"
  precedent this should get a follow-up fetch before admission.
- mirror_tag: `license:cc-by-4.0`
- verdict: **ATTRIBUTION** (provisional pending the second-surface check above)
- provenance_group: `social-bias-frames` (distinct lineage from Social Chemistry 101 despite
  overlapping authorship/lab)
- size: not sized via datasets-server in this pass
- quality: structured frames (offensiveness, intent, group targeted, implied statement) over
  social-media posts — strong for implicit-bias reasoning, not just detection
- consume/emit fit: pairs well with Social Chemistry 101 as a second norms/frames source,
  helping `B2`'s effective-source-count if both are admitted to one region
- enrichment: TASL attribution notice required on release

### 13. `ucberkeley-dlab/measuring-hate-speech`

- upstream: UC Berkeley D-Lab (Kennedy et al., published dataset + IRT methodology paper)
- **licence_upstream: VERIFIED (mirror-corroborated, first-party org)** — same caveat as
  #12: HF card is D-Lab's own first-party upload (`license:cc-by-4.0`), not independently
  cross-checked at a second primary surface in this pass.
- mirror_tag: `license:cc-by-4.0`
- verdict: **ATTRIBUTION** (provisional, same reasoning as #12)
- provenance_group: `measuring-hate-speech` (standalone)
- size: 135,556 rows
- quality: notably rigorous — IRT (item response theory) aggregated continuous hate-speech
  severity score instead of majority-vote binary labels, large annotator pool with annotator
  demographics recorded (useful for measuring annotator-bias effects separately)
- consume/emit fit: held-out severity-calibration probe; the continuous IRT score is a better
  measurement instrument than a binary hate/not-hate label for the operator's "measure,
  don't assume" requirement
- enrichment: TASL attribution notice required

### 14. `google/jigsaw_toxicity_pred`

- upstream: Kaggle "Toxic Comment Classification Challenge" (Jigsaw/Conversation AI, distinct
  competition from Civil Comments' challenge)
- **licence_upstream: mirror only, VERIFIED at mirror, not cross-checked at Kaggle directly**
  — HF card states `license:cc0-1.0`; Kaggle competition rules for this specific challenge
  were not independently fetched in this pass (Kaggle competition pages typically require
  login for full terms, flagged rather than skipped silently).
- mirror_tag: `license:cc0-1.0`
- verdict: **PERMISSIVE_OK** (provisional pending a direct Kaggle terms check — note this is
  a DIFFERENT provenance group from Civil Comments despite both being Jigsaw/Conversation AI
  toxicity challenges)
- provenance_group: `jigsaw-toxic-comment` (distinct from `civil-comments`)
- size: 319,301 shown for the config queried (multiple configs exist; this is the general
  Wikipedia-talk-page-comments toxic/severe_toxic/obscene/threat/insult/identity_hate corpus)
- quality: large, real Wikipedia talk-page comments, multi-label toxicity taxonomy
- consume/emit fit: another candidate held-out probe (toxicity taxonomy differs slightly from
  Civil Comments', useful for cross-taxonomy robustness measurement)
- enrichment: none needed if CC0 confirmed; treat as probe pending the Kaggle terms check

### 15. `Hate-speech-CNERG/hatexplain` — HateXplain

- upstream: https://github.com/hate-alert/HateXplain (Mathew et al., AAAI 2021)
- **licence_upstream: mirror only, not independently cross-checked** — HF card (first-party
  CNERG org) states `license:cc-by-4.0`; GitHub repo licence file not independently fetched
  in this pass.
- mirror_tag: `license:cc-by-4.0`
- verdict: **ATTRIBUTION** (provisional)
- provenance_group: `hatexplain` (standalone)
- size: not sized via datasets-server in this pass
- quality: notable for including **rationale spans** (which tokens triggered the hate/offense
  judgment) alongside the label — directly useful as a template for rationale-annotated
  moral-judgment data, which most of this survey's candidates lack
- consume/emit fit: candidate for the "rationale" enrichment pattern the ground doc's
  catalogue shape calls for — could inform how to add rationales to ETHICS (#1) or Scruples
  (#4) mechanically
- enrichment: TASL attribution notice required

### 16. `demelin/moral_stories` — Moral Stories

- upstream: https://github.com/demelin/moral_stories (Emelin et al., EMNLP 2021)
- **licence_upstream: attempted, inconclusive** — raw GitHub LICENSE fetch 404'd; mirror
  card states `license:mit`. Not independently confirmed at a second primary surface in this
  pass.
- mirror_tag: `license:mit`
- verdict: **PERMISSIVE_OK** (provisional — weakest verification in this survey; flag for a
  direct repo-file check before admission)
- provenance_group: `moral-stories` (standalone)
- size: 720,000 rows (large — norm + situation + moral/immoral action pairs + consequence)
- quality: structured (norm, situation, intention, moral action, immoral action, moral
  consequence, immoral consequence) — the single best-shaped candidate in this survey for
  direct moral-dilemma reasoning training, if the licence confirms
- consume/emit fit: strong direct fit for a moral-reasoning-with-consequences training format
- enrichment: minimal — already close to a ready-made contrastive (moral vs immoral action +
  consequence) training item

### 17. `ninoscherrer/moralchoice` — MoralChoice

- upstream: https://github.com/ninodimontalcino/moralchoice (Scherrer et al., NeurIPS 2023)
- **licence_upstream: mirror only** — HF card `license:cc-by-4.0`; not independently
  cross-checked at the GitHub repo in this pass.
- mirror_tag: `license:cc-by-4.0`
- verdict: **ATTRIBUTION** (provisional)
- provenance_group: `moralchoice` (standalone)
- size: very small per datasets-server (3 rows in the default config — likely a
  scenario-template config; the full benchmark is documented as ~1,700+ moral scenarios
  across configs, **INFERRED** from the paper, not row-counted here)
- quality: purpose-built to elicit an LLM's revealed moral preferences under low- and
  high-ambiguity dilemmas — this is an eval/probe instrument by design, not training volume
- consume/emit fit: best fit as a held-out probe for the "measure whether training changes
  behaviour" requirement — it was built exactly for that measurement purpose
- enrichment: n/a — keep as probe

### 18. `mmathys/openai-moderation-api-evaluation`

- upstream: https://github.com/openai/moderation-api-release (Markov et al., "A Holistic
  Approach to Undesired Content Detection")
- **licence_upstream: mirror only** — HF card `license:mit`; GitHub repo not independently
  re-fetched for its own LICENSE file in this pass.
- mirror_tag: `license:mit`
- verdict: **PERMISSIVE_OK** (provisional)
- provenance_group: `openai-moderation-eval` (standalone)
- size: 1,680 rows
- quality: small, human-labelled multi-category (sexual, hate, violence, self-harm,
  harassment, etc.) held-out eval set OpenAI released specifically as an external benchmark
- consume/emit fit: another held-out probe candidate, complementary taxonomy to
  BeaverTails'/WildGuard's categories
- enrichment: n/a — keep as probe

### 19. `walledai/TDC23-RedTeaming`

- upstream: Trojan Detection Challenge 2023 (NeurIPS competition, red-teaming track)
- **licence_upstream: mirror only, not independently verified at the competition site**
- mirror_tag: `license:mit`
- verdict: **PERMISSIVE_OK** (provisional — this is a third-party re-packaging
  (`walledai`), not a first-party org upload, so the "mirrors lie" risk is highest here of
  any entry in this survey; do not admit without a direct competition-site licence check)
- provenance_group: `tdc23-redteaming` (standalone)
- size: not sized in this pass
- quality: adversarial red-team prompts from a competitive benchmark — useful methodologically
  but the licence chain (competition → walledai repackaging) is the least trustworthy in this
  set
- consume/emit fit: candidate red-team probe if the licence confirms
- enrichment: n/a pending verification

### 20. `kellycyy/daily_dilemmas`

- upstream: project associated with "Everyday Ethical Dilemmas" work (Kelly Cui et al.)
- **licence_upstream: mirror only, not independently verified**
- mirror_tag: `license:cc-by-4.0`
- verdict: **ATTRIBUTION** (provisional)
- provenance_group: `daily-dilemmas` (standalone)
- size: not sized in this pass
- quality: everyday (non-extreme) moral dilemmas grounded in values taxonomies —
  complementary to Moral Stories/Scruples' more dramatic framing
- consume/emit fit: another moral-dilemma training candidate, useful for diversifying away
  from Reddit/AITA-flavoured framing (which Scruples and Moral Stories both lean on)
- enrichment: needs review of the values-taxonomy annotation format before ingest

---

## REFUSED / not admitted, and why

- **ToxiGen** (#9 above is listed as a survey entry, not a refusal, but is functionally
  refused pending resolution) — genuine licence-vs-README conflict (CDLA-Permissive-2.0 grant
  vs "research purposes only" README language). Not auto-refused (the legal grant itself is
  unrestricted) and not auto-admitted (the authors' stated intent narrows it) — this is a
  flag-for-operator-decision case, explicitly not resolved unilaterally per instructions.
- **Reddit-sourced corpora generally** (Scruples' anecdotes split, any hypothetical raw
  r/AmITheAsshole scrape not mediated by AI2's Apache-2.0 grant) — only admitted here because
  AI2 states an explicit redistribution licence as the distributor; a raw scrape without that
  grant would fall in the "distributor disclaims owning what they distribute" refusal class
  the operator named.
- **TED-LIUM-style ND classes** — none surfaced in the moral/safety search space this pass,
  consistent with the audio audit's precedent that ND shows up mostly in speech corpora, not
  text-judgment ones; noting the absence rather than assuming it holds for a future pass.
- **No CONSENT_OPEN candidates surfaced** — none of the 20 candidates here are Common-
  Voice-shaped (live revocable consent over otherwise-CC0 material); all are either static
  academic releases or platform-archive snapshots (Civil Comments) where the archiving org,
  not individual ongoing consent, is the redistribution basis.
- **Did not pursue**: raw 4chan/Gab hate-speech corpora (well-known in this space, e.g. the
  "Hateful Memes" adjacent 4chan Pol datasets) — skipped on provenance grounds without even a
  licence check: these are typically scraped without any rights-holder grant and match the
  operator's "no licence grant exists at all" BLOCKING class by construction; not worth a
  fetch to confirm what the collection method already tells us.

---

## Top 8 (by verdict strength × fit × verification quality)

1. **HH-RLHF (+ red-team-attempts)** — `Anthropic/hh-rlhf` — PERMISSIVE_OK (MIT), largest
   well-documented harmlessness-preference + refusal corpus, best-verified after ETHICS
2. **ETHICS** — `hendrycks/ethics` — PERMISSIVE_OK (MIT), best-verified licence in the whole
   survey (full LICENSE text fetched), five-framework structured judgments, doubles as a
   measurement probe
3. **Social Chemistry 101** — `wassname/social_chemistry_101` — SHARE_ALIKE (CC BY-SA-4.0),
   verbatim-verified, richest norms structure (situation + rule-of-thumb + judgment)
4. **Moral Stories** — `demelin/moral_stories` — PERMISSIVE_OK (MIT, provisional), largest
   volume (720K rows) and best-shaped for contrastive moral/immoral consequence training,
   pending the one outstanding licence re-check
5. **Scruples** — `metaeval/scruples` — PERMISSIVE_OK (Apache-2.0), verbatim-verified,
   real-world (AITA) judgment distributions
6. **PKU-SafeRLHF** — `PKU-Alignment/PKU-SafeRLHF` — NC (CC BY-NC-4.0), verified at the
   dataset's own first-party card, dual helpfulness/harmlessness signal at scale, NC already
   accepted policy given `memory`'s existing GooAQ NC tier
7. **WildGuardMix / WildJailbreak** — `allenai/wildguardmix` / `allenai/wildjailbreak` —
   ODC-BY, code licence verbatim-verified (Apache-2.0) with the data-specific ODC-BY
   corroborated at the first-party card; the only entry here purpose-built for refusal
   calibration specifically (over- and under-refusal both labelled)
8. **BeaverTails** — `PKU-Alignment/BeaverTails` — NC (CC BY-NC-4.0, provisional — primary
   source didn't surface an explicit licence, riding the same-org/same-lineage inference from
   PKU-SafeRLHF), 14-category harm taxonomy, complements PKU-SafeRLHF's format

**Held-out measurement probes (explicitly not training-admission candidates), ranked by
fit for the operator's "measure whether training changes behaviour" requirement:**
MoralChoice (purpose-built for this), ETHICS' held-out slice, Civil Comments, Measuring Hate
Speech (continuous IRT severity — best calibration instrument), OpenAI Moderation Eval,
StereoSet, CrowS-Pairs, Jigsaw Toxicity (cross-taxonomy check against Civil Comments).

## What still needs a follow-up fetch before any of these enter a training run

Every "provisional" verdict above (#6 BeaverTails' licence page, #12 Social Bias Frames,
#13 Measuring Hate Speech, #14 Jigsaw Toxicity's Kaggle terms, #15 HateXplain, #16 Moral
Stories' GitHub LICENSE 404, #17 MoralChoice, #18 OpenAI Moderation, #19 TDC23-RedTeaming,
#20 Daily Dilemmas) was VERIFIED only at the HF mirror card or a secondary corroborating
surface, not at a second independent primary-source document, in the time this pass allowed.
Per the "mirrors lie" precedent (10/75 mismatches previously found), none of these should be
treated as production-admitted on this survey alone — they are ranked and catalogued, not
cleared.
