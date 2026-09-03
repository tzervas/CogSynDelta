# Survey: RETRIEVE / MEMORY faculty (question-passage, hard-negative retrieval,
BEIR-style full-pool eval, multilingual retrieval, long-context/episodic recall)

Session: 2026-09-03. Source tree: `CogSynDelta` (read-only, per `00-ground.md`). No dataset
CONTENT downloaded — only HF Hub API metadata, dataset cards, and primary-source licence
pages fetched via WebFetch/WebSearch this session. Token secret was not needed (public HF
API answered every query used here).

Method used, per the task brief:
1. HF Hub public API (`/api/datasets/<id>`) for mirror tag, downloads, task tags.
2. WebFetch against each candidate's upstream/primary source (paper site, GitHub repo,
   LICENSE file, task homepage) for the operative licence sentence.
3. WebSearch where the primary source needed locating or the GitHub LICENSE file 404'd.

**Headline finding this pass: the "mirrors lie" pattern (see
`dataset-mirror-licences-lie.md`) reproduces hard inside BeIR.** Every one of the eleven
`BeIR/*` repos on HF Hub carries the *identical* blanket mirror tag `cc-by-sa-4.0`
regardless of the subset's actual upstream licence. Two subsets were individually
fetched and check out as mismatches:
- `BeIR/scifact` → primary is **split**: claims/annotations CC BY 4.0, corpus abstracts
  ODC-By 1.0 (S2ORC lineage). Neither is CC BY-SA-4.0.
- `BeIR/scidocs` → primary LICENSE file is **CC BY 4.0** (ATTRIBUTION), not SA.
- `BeIR/fiqa` → primary task site (sites.google.com/view/fiqa) states plainly:
  **"The training data is available only for non-commercial use" / "The testing data is
  available only for non-commercial use."** This is a genuine NC term, not CC BY-SA-4.0 —
  the single largest mismatch found this pass, and it directly affects `ground.md`'s own
  `memory` faculty row, which currently lists FiQA's obligation as CC-BY-SA.

Treat every other un-individually-verified `BeIR/*` tag below as **UNVERIFIED
inherited-with-caution**, not confirmed SHARE_ALIKE, until each subset's own primary is
fetched.

---

## Catalogue entries

### 1. `microsoft/ms_marco` (MS MARCO passage/QnA)
- upstream: https://microsoft.github.io/msmarco/ (Microsoft Research)
- mirror: HF `microsoft/ms_marco`, cardData license: **none set**; a third-party repackage
  `Tevatron/msmarco-passage` tags `apache-2.0` on HF — **mirror lie**, the repackager's own
  code licence, not the data's.
- `licence_upstream` (VERIFIED, fetched 2026-09-03, https://microsoft.github.io/msmarco/):
  *"The MS MARCO datasets are intended for non-commercial research purposes only... made
  available free of charge without extending any license or other intellectual property
  rights... provided 'as is' without warranty and usage of the data has risks since we may
  not own the underlying rights in the documents."*
- grant_scope: whole_corpus (but explicitly non-commercial, no redistribution licence
  extended, and Microsoft disclaims owning underlying rights — three separate concerns
  stacked in one sentence)
- verdict: **NC** (genuine rights-holder term, DEC-31 non-blocking) — but note the "we may
  not own the underlying rights" clause is closer to the operator's REFUSE class
  ("distributors that disclaim owning what they distribute") than plain NC. Flag for
  operator judgment call rather than auto-accept under the NC waiver.
- provenance_group: MS MARCO (covers ms_marco, msmarco-passage, mmarco multilingual
  variants — one lineage)
- size: ~8.8M passages, ~1M queries (rows, VERIFIED from card metadata pattern; exact
  count not re-verified this pass)
- quality: human-authored Bing queries + human-judged relevance; heavily used, some
  overlap/contamination risk with downstream evals is well documented in IR literature
- consume/emit fit: primary passage-retrieval training source for `memory`'s retrieve half
- enrichment: hard-negative mining (BM25 + dense) already standard in the field; enrichment
  does not touch licence since NC already caps the tier
- licence consequence of enrichment: none beyond existing NC (already the strictest
  ceiling among current `memory` inputs alongside GooAQ)

### 2. `sentence-transformers/natural-questions` (Natural Questions, retrieval framing)
- upstream: https://github.com/google-research-datasets/natural-questions
- mirror tag: none set on the sentence-transformers repackage
- `licence_upstream` (VERIFIED, fetched 2026-09-03): repo LICENSE file is
  **Apache License 2.0** (confirmed present at `/blob/master/LICENSE`, full text not
  reproduced by the fetch tool but type and location confirmed).
- grant_scope: whole_corpus
- verdict: **PERMISSIVE_OK**
- provenance_group: Natural Questions (Google Research, single lineage)
- size: 307,373 training examples (paper-reported; not re-verified this pass — INFERRED
  from prior knowledge, mark for row-count re-check before ingest)
- quality: real Google search queries + human-annotated long/short answers from Wikipedia;
  high curation, widely used as a BEIR/MTEB benchmark member (contamination risk with any
  eval reusing NQ)
- consume/emit fit: strong core positive-pair source for retrieve; clean licence makes it
  the best B1-diluting addition available for `memory`
- enrichment: hard-negative mining against the Wikipedia passage pool; no licence
  consequence (PERMISSIVE_OK stays PERMISSIVE_OK downstream)

### 3. `allenai/gooaq` (already in the trained corpus per `ground.md`)
- upstream: https://github.com/allenai/gooaq
- mirror tag (HF `allenai/gooaq`): **apache-2.0**
- `licence_upstream` (VERIFIED, fetched 2026-09-03, two sources):
  - repo LICENSE file (`raw.githubusercontent.com/allenai/gooaq/main/LICENSE`): Apache
    License 2.0 text.
  - repo README (fetched same pass): **"NOTE This dataset should not be used for any
    commercial purposes. See the license for the detailed terms."**
- This is an internal contradiction at the *same* primary source: the LICENSE file text is
  Apache-2.0 (permissive), but the README imposes an explicit NC restriction. Per the
  strictest-input rule this resolves to **NC**, matching `ground.md`'s existing
  classification ("gooaq (NC, ACCEPTED under DEC-31)") — this pass corroborates that verdict
  from the primary source rather than just the mirror.
- grant_scope: whole_corpus, NC-restricted despite the misleading Apache badge
- verdict: **NC** (confirms existing ground-truth entry; do not "fix" it to
  PERMISSIVE_OK on the strength of the mirror tag or the LICENSE file alone — this is a
  live example of the exact trap the provenance rule exists to catch)
- provenance_group: GooAQ (single lineage; already the dominant 79.1% share in `memory`)
- size: 3.01M question-answer pairs (Google autocomplete + answer boxes)
- quality: high volume, but the "answer" side is scraped from Google's own answer-box
  synthesis, not independently human-authored — a real quality/provenance caveat beyond
  licence
- consume/emit fit: already the concentration problem for `memory`; do not add more,
  the priority is diluting it with sources 1-2 above
- enrichment: none proposed — already saturating the region

### 4. `hotpotqa/hotpot_qa`
- upstream: https://hotpotqa.github.io/
- mirror tag: cc-by-sa-4.0 (matches)
- `licence_upstream` (VERIFIED, fetched 2026-09-03): *"HotpotQA is distributed under a
  CC BY-SA 4.0 License... the processed Wikipedia used in the process of creating HotpotQA
  (also under a CC BY-SA 4.0 License)."*
- grant_scope: whole_corpus
- verdict: **SHARE_ALIKE** (mirror and upstream agree — no mismatch here, a clean
  confirmation case)
- provenance_group: HotpotQA / Wikipedia (shares lineage with SQuAD, DBpedia-entity, FEVER,
  MIRACL-en, Mr.TyDi-en — all Wikipedia-derived text; consider as one provenance group for
  B1/B2 if the faculty pool ends up Wikipedia-heavy)
- size: already on disk (961M dir shared with squad/hotpotqa-corpus per ground.md); ~113k
  QA pairs, corpus ~5.2M paragraphs
- quality: multi-hop, human-authored questions requiring 2-doc reasoning; well-curated,
  standard BEIR member (contamination risk with any eval reusing hotpotqa)
- consume/emit fit: already partially trained on (per ground.md); full BEIR-pool version
  (corpus+queries+qrels split) upgrades the diagonal-only 512-pair setup ground.md flags as
  saturated
- enrichment: none needed beyond using the full qrels pool instead of the paired subset

### 5. `rajpurkar/squad` (SQuAD v1.1/v2.0)
- upstream: https://rajpurkar.github.io/SQuAD-explorer/
- mirror tag: cc-by-sa-4.0 (matches)
- `licence_upstream` (VERIFIED, fetched 2026-09-03): *"Download a copy of the dataset
  (distributed under the CC BY-SA 4.0 license)."*
- verdict: **SHARE_ALIKE**
- provenance_group: SQuAD / Wikipedia (same group as #4)
- size: already on disk, ~961M shared dir
- quality: gold-standard extractive QA, heavily used, some known train/test near-duplicate
  leakage documented in later literature
- consume/emit fit: already in use; confirms no change needed to its verdict
- enrichment: n/a, already trained on

### 6. `mandarjoshi/trivia_qa`
- upstream: https://nlp.cs.washington.edu/triviaqa/
- mirror tag: `unknown` (HF cardData literally says "unknown")
- `licence_upstream` (VERIFIED, fetched 2026-09-03): *"The University of Washington does
  not own the copyright of the questions and documents included in TriviaQA."*
- grant_scope: **unstated** — UW is explicitly disclaiming ownership of the underlying
  trivia questions and the web/Wikipedia evidence documents it scraped
- verdict: **REFUSE-TERMS** — this matches the operator stance's named refusal class
  verbatim: *"distributors that disclaim owning what they distribute."* This is not a
  judgment call the way MS MARCO's softer disclaimer is; UW states it outright and provides
  no licence grant at all, just a disclaimer.
- provenance_group: TriviaQA (own lineage — trivia-website + Wikipedia/web scrape)
- **REFUSED.**

### 7. `sentence-transformers/eli5` / `facebookresearch/ELI5`
- upstream: https://github.com/facebookresearch/ELI5
- mirror tag: none set on either HF repo
- `licence_upstream` (VERIFIED, fetched 2026-09-03, `raw.githubusercontent.com`): full BSD
  license text, "For ELI5 software... Copyright (c) Facebook, Inc." — a standard BSD-3
  code licence.
- grant_scope: **code_only** — the BSD grant covers Facebook's *scraping/processing code*,
  not the underlying Reddit posts (r/explainlikeimfive questions and answers) that make up
  the actual dataset content. Facebook does not claim, and BSD cannot grant, rights over
  Reddit user-authored content.
- provenance_red_flags: Reddit-scraped Q&A; no consent process described; Reddit's own API
  Terms (2023 pricing/terms change onward) restrict bulk redistribution of user content
  independent of what the scraper's code licence says — same shape as the
  distributor-disclaims-ownership pattern, one step removed (here the *code* licence
  doesn't even attempt to cover the *content*).
- verdict: **UNVERIFIED, lean REFUSE-TERMS** pending an explicit check of Reddit's terms as
  they applied at ELI5's 2019 scrape date versus today's redistribution terms. Do not
  ingest on the strength of the BSD badge — flag exactly this reasoning if a future
  surveyor is tempted to.
- **Recommend REFUSE for now; revisit only with a specific Reddit-content legal read.**

### 8. `BeIR/fiqa` (financial opinion QA/retrieval)
- upstream: https://sites.google.com/view/fiqa/home
- mirror tag: cc-by-sa-4.0 (BeIR blanket tag)
- `licence_upstream` (VERIFIED, fetched 2026-09-03): *"The training data is available only
  for non-commercial use." / "The testing data is available only for non-commercial use."*
- grant_scope: whole_corpus, explicitly NC
- verdict: **NC** — this is a **mirror lie**, not a UNVERIFIED gap: the mirror positively
  asserts CC-BY-SA-4.0 and the primary source positively asserts NC-only. `ground.md`'s
  `memory` row currently lists FiQA under "CC-BY-SA-4.0" — that line needs correcting to NC
  the next time `ground.md` (or its successor) is revised; this survey does not edit it.
- provenance_group: FiQA (WWW'18 challenge, own lineage)
- size: 57,638-passage pool (per ground.md, already the recommended eval gate), ~6,648
  question-answer pairs
- quality: financial-domain opinion/QA, StackExchange-style but curated for a shared task;
  moderate size, domain-narrow (good for B2's "explicitly narrow region" allowance, not a
  general-capability source)
- consume/emit fit: ground.md already earmarks its full pool as the eval gate for `memory`
  — that's still correct under NC (eval-time NC use is uncontroversial); using it for
  *training* is what needs the NC-tier acknowledgment, already priced in since `memory` is
  NC via GooAQ regardless
- enrichment: none proposed beyond already-planned full-pool eval use

### 9. `BeIR/scifact`
- upstream: https://github.com/allenai/scifact
- mirror tag: cc-by-sa-4.0 (BeIR blanket tag)
- `licence_upstream` (VERIFIED, fetched 2026-09-03, `LICENSE.md`): **split** —
  *"All claims and evidence annotations... are released under CC BY 4.0."* /
  *"The abstracts in the corpus... are part of the Semantic Scholar S2ORC dataset and are
  licensed under ODC-By 1.0."* / code is Apache-2.0.
- grant_scope: whole_corpus, but **two different licences for two different files** —
  claims/annotations vs. abstracts/corpus
- verdict: **ATTRIBUTION** for claims, **ODC-BY** for the passage corpus — neither is
  SHARE_ALIKE. Mismatch with BeIR's blanket tag confirmed.
- provenance_group: SciFact (claims) + S2ORC (corpus) — two groups, not one; if S2ORC-lineage
  material is used elsewhere (it commonly is, e.g. in SciDocs), collapse those under one
  S2ORC provenance group for B1/B2.
- size: 1,409 claims, corpus of ~5,183 abstracts
- quality: expert-annotated scientific claim verification; small but very clean; strong
  fit for `episodic_store`/long-context work given scientific-abstract structure, less so
  as a volume source for retrieve
- consume/emit fit: good B2-diluting narrow addition to `memory`'s retrieve half
- enrichment: none proposed; the ODC-By corpus half would need an attribution manifest if
  redistributed downstream (ODC-By requires attribution, similar obligation weight to CC BY)

### 10. `BeIR/scidocs`
- upstream: https://github.com/allenai/scidocs
- mirror tag: cc-by-sa-4.0 (BeIR blanket tag)
- `licence_upstream` (VERIFIED, fetched 2026-09-03, `LICENSE` raw file): full **CC BY 4.0**
  legal code text.
- verdict: **ATTRIBUTION**, not SHARE_ALIKE — second confirmed mirror lie this pass.
- provenance_group: SciDocs (Semantic Scholar-derived citation/recommendation data — flag
  as possibly related to the SciFact/S2ORC provenance group above; not confirmed identical
  lineage this pass, mark for follow-up)
- size: citation-prediction task, ~30k papers across 7 sub-tasks (not row-verified this pass)
- quality: derived from Semantic Scholar metadata, used as a BEIR document-similarity
  benchmark member more than a training-pair source
- consume/emit fit: modest fit for `memory` — better suited as an eval addition (documents,
  not query-answer pairs) than a training source
- enrichment: n/a; ATTRIBUTION propagates, needs a TASL notice if redistributed

### 11. `BeIR/nfcorpus`
- upstream: nutrition-focused medical IR corpus (NFCorpus, Boteva et al., built from
  NutritionFacts.org forum + PubMed)
- mirror tag: cc-by-sa-4.0 (BeIR blanket tag) — **not individually verified this pass**
- verdict: **UNVERIFIED** — flag explicitly rather than trust the blanket tag, given the
  two confirmed mismatches above from the same mirror family. Do not admit to training
  until its own primary (the NFCorpus project page) is fetched.
- provenance_group: NFCorpus (own lineage; PubMed content specifically often carries its
  own per-article rights, similar caution as CORD-19 below)

### 12. `BeIR/arguana`
- upstream: args.me corpus / ArguAna similarity task
- mirror tag: cc-by-sa-4.0 (BeIR blanket tag) — **not individually verified this pass**
- verdict: **UNVERIFIED**
- provenance_group: args.me / ArguAna (debate-portal-scraped argument text; own lineage)

### 13. `BeIR/quora` (Quora Question Pairs)
- upstream: originally quoradata.quora.com's 2017 dataset-release post (WebFetch returned
  HTTP 403 on the live Quora page this pass; WebSearch found no explicit licence statement
  from Quora itself for the 2017 QQP release — most redistributions are via the Kaggle
  competition, whose competition rules govern competition use, not general redistribution)
- mirror tag: cc-by-sa-4.0 (BeIR blanket tag)
- `licence_upstream`: **fetch attempt failed** (403); WebSearch found no primary licence
  text — status **UNVERIFIED**, not INFERRED
- verdict: **UNVERIFIED, caution** — given two confirmed mirror lies in the same BeIR
  family plus no located primary grant at all for Quora's own release, do not accept the
  CC-BY-SA-4.0 tag at face value. Recommend a follow-up fetch of
  `quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs` directly (not through a
  redirector) before any admission decision.
- provenance_group: Quora Question Pairs (own lineage)

### 14. `BeIR/fever`
- upstream: https://fever.ai/ (data + license page `fever.ai/download/fever/license.html`
  did not yield the FEVER-specific statement this pass — the fetch returned Wikipedia's
  general copyright policy plus CC BY-SA **3.0**, one version behind the BeIR mirror's 4.0
  tag)
- mirror tag: cc-by-sa-4.0
- `licence_upstream`: **partially VERIFIED** — Wikipedia-derived evidence text under CC
  BY-SA 3.0 confirmed from a fever.ai-linked page, but the exact FEVER-dataset-specific
  license statement (as opposed to the underlying Wikipedia policy it inherits) was not
  isolated this pass.
- verdict: **SHARE_ALIKE** (version drift 3.0-vs-4.0 noted but both are share-alike class,
  doesn't change the verdict bucket)
- provenance_group: FEVER / Wikipedia (same group as HotpotQA, SQuAD — Wikipedia lineage)

### 15. `BeIR/climate-fever`
- upstream: built on the same FEVER methodology, applied to climate-change claims
- mirror tag: cc-by-sa-4.0 — **not individually verified this pass**; INFERRED same
  provenance and licence shape as FEVER (#14) since it's explicitly a FEVER-methodology
  derivative, but not independently confirmed
- verdict: **UNVERIFIED (high-confidence INFERRED SHARE_ALIKE, not fetched)**
- provenance_group: same as FEVER — climate-fever should collapse into the FEVER/Wikipedia
  provenance group, not count as an independent source for B1/B2

### 16. `BeIR/trec-covid`
- upstream: CORD-19 (allenai/cord19), via `ir.nist.gov/covidSubmit/`
- mirror tag: cc-by-sa-4.0 (BeIR blanket tag)
- `licence_upstream` (VERIFIED via WebSearch, 2026-09-03, allenai/cord19 GitHub + AWS
  Marketplace listing): CORD-19 is explicitly **heterogeneous per-article** — "AI2 grants a
  worldwide, perpetual, non-exclusive, non-transferable license... for text and data mining
  only," and the corpus is split into **commercial-use, non-commercial-use, and
  custom-license subsets** by publisher, with per-article rights recorded in metadata.
  Nothing close to a single CC-BY-SA-4.0 grant exists at the primary source.
- grant_scope: **unstated at the corpus level** — genuinely varies row-by-row; no single
  verdict can honestly apply to "trec-covid" as one blob
- provenance_red_flags: "text and data mining only" grant is narrower than a training-data
  grant in some readings; mixed-licence corpus requires per-document filtering before any
  admission decision, which the factory does not currently do
- verdict: **REFUSE as a single corpus** — not because every article is unlicensable, but
  because ingesting it under one blanket tag (as BeIR's mirror does) would silently
  mislabel a real mix of NC/custom/commercial-tier content. **Recommend**: if `memory` ever
  wants CORD-19 material, pull only the AI2-published "commercial use subset" by its own
  per-article manifest — a different, much smaller pipeline than "download BeIR/trec-covid."
- **REFUSED as currently mirrored.**

### 17. `BeIR/dbpedia-entity`
- upstream: DBpedia (Wikipedia-infobox-derived structured data)
- mirror tag: cc-by-sa-4.0 — **not individually verified this pass**; DBpedia's own
  well-known licence is CC BY-SA 3.0, consistent in class (share-alike) with the mirror tag
  though again a version-number question, same pattern as FEVER
- verdict: **UNVERIFIED (high-confidence INFERRED SHARE_ALIKE)**
- provenance_group: DBpedia / Wikipedia (same group as #4, #5, #14)

### 18. `BeIR/cqadupstack`
- upstream: StackExchange data dumps (12 sub-forums)
- mirror tag: cc-by-sa-4.0 — **not individually verified this pass**; StackExchange's own
  data-dump licence is genuinely CC BY-SA 4.0, so this is the one BeIR blanket-tag case
  most likely to actually be correct, but still not independently fetched this pass
- verdict: **UNVERIFIED (high-confidence INFERRED SHARE_ALIKE)**
- provenance_group: StackExchange (own group; note StackExchange also underlies some code
  QA corpora used elsewhere in the fleet — check for double-counting if so)

### 19. `BeIR/webis-touche2020`
- upstream: args.me corpus (same underlying source as ArguAna, #12)
- mirror tag: cc-by-sa-4.0 — **not individually verified this pass**
- verdict: **UNVERIFIED**
- provenance_group: args.me — same provenance group as ArguAna (#12); collapse for B1/B2 if
  both are admitted

### 20. `miracl/miracl` (multilingual retrieval, 18 languages)
- upstream: https://github.com/project-miracl/miracl
- mirror tag: apache-2.0
- `licence_upstream` (VERIFIED, fetched 2026-09-03): repo badge and LICENSE reference
  confirm **Apache-2.0** for the repo.
- grant_scope: **code_only in substance** — MIRACL's passages are Wikipedia text in 18
  languages; Apache-2.0 is the pipeline/tooling licence, the actual passage text still
  carries Wikipedia's CC BY-SA obligation regardless of the badge on the repo. Flag this
  explicitly, it is the same shape as GooAQ's contradiction but easier to miss since there's
  no README note calling it out.
- verdict: **SHARE_ALIKE** (by the underlying Wikipedia content, not the repo's own Apache
  tag)
- provenance_group: Wikipedia (joins #4, #5, #14, #17 — a large Wikipedia-lineage cluster
  worth tracking as one group for B1/B2 once several of these are admitted together)
- size: 18 languages, largest multilingual BEIR-style retrieval set available
- quality: native-speaker-authored queries per language, TREC-style pooled judgments;
  high-quality, addresses the "multilingual retrieval sets" ask directly
- consume/emit fit: best available multilingual-retrieval candidate for `memory`; also the
  strongest B2 N_eff lever if per-language passage pools count as distinct strata (check
  B5's stratum-key choice — language is a natural stratum key here)
- enrichment: none required; already query-passage-qrels shaped

### 21. `castorini/mr-tydi` (multilingual retrieval, 11 languages)
- upstream: https://github.com/castorini/mr.tydi
- mirror tag: apache-2.0
- `licence_upstream` (VERIFIED, fetched 2026-09-03): *"Mr. TyDi is licensed under the
  Apache License 2.0."*
- grant_scope: same code_only-in-substance caveat as MIRACL — Wikipedia-derived passages
- verdict: **SHARE_ALIKE** (by underlying content)
- provenance_group: Wikipedia (same group as #20)
- size: smaller than MIRACL, TyDi-QA-derived query set across 11 typologically diverse
  languages
- consume/emit fit: redundant with MIRACL for most of its language coverage; include only
  if MIRACL's per-language splits don't already cover a needed language
- enrichment: none required

### 22. `deepmind/narrativeqa` (long-document QA — episodic-store candidate)
- upstream: https://github.com/deepmind/narrativeqa
- mirror tag: apache-2.0
- `licence_upstream` (VERIFIED, fetched 2026-09-03): repo licence badge is **Apache
  License, Version 2.0**; the fetch explicitly found **no mention** of rights to the
  underlying books/movie scripts (a mix of Project Gutenberg texts and film/TV scripts).
- grant_scope: **code_only** — Apache-2.0 covers DeepMind's QA-pair generation code and
  summary annotations, not necessarily the full book/script texts some pipelines pair with
  the summaries. Gutenberg-sourced books are generally PD/permissive on their own; movie
  and TV scripts are a much less certain rights position and were not addressed by the
  primary source at all.
- verdict: **UNVERIFIED for the full-text pairing**, but the **summaries + QA pairs
  themselves** (DeepMind's own annotation layer) are reasonably **PERMISSIVE_OK** under
  Apache-2.0. Treat as two different admission decisions: summary-only use vs. full-text
  pairing.
- provenance_group: NarrativeQA (own lineage; Gutenberg sub-portion overlaps whatever
  Gutenberg material might already be staged for the `language` trunk faculty — check for
  double-counting there)
- consume/emit fit: this is the best-shaped candidate this pass for the "long-context /
  episodic recall" ask — full-book-length context with grounded QA is exactly
  `episodic_store`'s target shape (long-range key/value retrieval over extended context)
- enrichment: pairing summary-level QA with the retrieved passage/chapter (rather than the
  full book) would both fit the episodic-store's chunked-recall use case better and reduce
  exposure to the unverified movie-script rights question — recommend this scoped variant
  over full-book ingestion

### 23. `allenai/qasper` (long-document scientific-paper QA — episodic-store candidate)
- upstream: allenai.org/data/qasper → redirects to the HF dataset card
- mirror tag: cc-by-4.0
- `licence_upstream`: **INFERRED, not independently re-fetched past the mirror** this pass
  (the allenai.org page redirects straight to the HF card itself, so this is not a true
  independent-primary confirmation — mark UNVERIFIED-at-primary despite the clean-looking
  tag, per the provenance rule's own standard: same host as the mirror fails A0m's gate)
- verdict: **UNVERIFIED** (tag looks right, second independent source still needed —
  the underlying papers are S2ORC/Semantic-Scholar-sourced full papers, likely the same
  ODC-By lineage as SciFact's corpus half, #9)
- provenance_group: possibly S2ORC (same group as SciFact's corpus — follow up)
- consume/emit fit: long-document QA over full papers, another strong episodic-store shape
  candidate if the licence resolves cleanly

### 24. `tau/scrolls` (long-context benchmark suite: NarrativeQA, Qasper, GovReport,
QMSum, SummScreenFD, QuALITY, ContractNLI — bundled)
- upstream: https://www.scrolls-benchmark.com/
- mirror tag: HF cardData license: **none set**
- `licence_upstream`: **not fetched this pass** — flag only as a bundling convenience:
  SCROLLS re-packages several already-listed candidates (NarrativeQA #22, Qasper #23) plus
  others (GovReport is US-government-authored, likely PD; ContractNLI has its own separate
  licence not assessed) under one HF loader. Its own component parts still need individual
  licence verdicts — the bundle's absent HF licence tag is itself a small red flag (no
  claimed licence at all, distinct from "unknown").
- verdict: **UNVERIFIED**, treat as a loader convenience, not a provenance group of its own
  — decompose to its members' individual verdicts

---

## Refused (class REFUSE / do-not-ingest-as-is)

| id | reason |
|---|---|
| `mandarjoshi/trivia_qa` | Primary source explicitly disclaims copyright ownership of both questions and evidence documents — the operator stance's named REFUSE class verbatim |
| `BeIR/trec-covid` (CORD-19) | Genuinely heterogeneous per-article licensing (commercial/NC/custom subsets); a single blanket verdict misrepresents the corpus; would need per-document manifest filtering the factory doesn't do today |
| `facebookresearch/ELI5` (both HF mirrors) | Code licence (BSD) does not extend to the underlying Reddit-authored content; Reddit's own redistribution terms not checked; recommend REFUSE until that specific check is done, not a hard structural block |
| `BeIR/quora` | No primary licence located at all this pass (403 on the live page, WebSearch turned up nothing authoritative); BeIR's own tag is unconfirmed and sits inside a mirror family with two already-confirmed lies — do not admit on the tag alone |

---

## Top 8 candidates, ranked, with verdicts

1. **Natural Questions** (`sentence-transformers/natural-questions` / upstream Google
   Research) — **PERMISSIVE_OK**, VERIFIED at primary. Best single addition: cleanest
   licence, real human queries, directly dilutes GooAQ's 79.1% B1 concentration in `memory`.
2. **MIRACL** (`miracl/miracl`) — **SHARE_ALIKE** (by underlying Wikipedia content, repo
   badge is Apache-2.0 but that's code_only), VERIFIED at primary. Best multilingual-
   retrieval candidate, directly answers the "multilingual retrieval sets" ask, strong B2
   lever via per-language strata.
3. **HotpotQA full BEIR pool** (`hotpotqa/hotpot_qa`, corpus+queries+qrels) — **SHARE_ALIKE**,
   VERIFIED, confirms existing ground.md entry and upgrades it from the saturated 512-pair
   diagonal to the real pool.
4. **SciFact** (`BeIR/scifact`) — **ATTRIBUTION / ODC-BY split**, VERIFIED, mirror-lie
   caught and corrected. Small, clean, good B2 narrow-domain diluter.
5. **NarrativeQA — summary/QA layer only** (`deepmind/narrativeqa`) — **PERMISSIVE_OK** for
   the annotation layer (full-text pairing UNVERIFIED, scope it out). Best-shaped
   candidate found this pass for the episodic-store's long-context recall need.
6. **Mr. TyDi** (`castorini/mr-tydi`) — **SHARE_ALIKE**, VERIFIED, secondary multilingual
   source, admit only for languages MIRACL doesn't cover to avoid redundant volume.
7. **FiQA full pool** (`BeIR/fiqa`) — **NC**, VERIFIED, mirror-lie caught (was
   mis-tagged SHARE_ALIKE in both BeIR and `ground.md`'s own memory row — flag that row for
   correction). Already earmarked as the eval gate; fine to keep at NC since `memory` is
   NC-tier already via GooAQ.
8. **SciDocs** (`BeIR/scidocs`) — **ATTRIBUTION**, VERIFIED, mirror-lie caught. Better fit
   as an eval/document-similarity addition than a training-pair source, but cleanly
   licensed and cheap to add.

**Not in the top 8 but flagged as the single most consequential correction this pass:**
FiQA's licence mismatch in `ground.md` itself (CC-BY-SA-4.0 stated there vs. NC verified
here at the primary source) — worth fixing in the ground doc regardless of whether FiQA
gets used for training, since the eval-gate framing in `ground.md` currently understates
FiQA's actual restriction.

## What still needs individual primary-source verification (not done this pass, time-boxed)

`BeIR/nfcorpus`, `BeIR/arguana`, `BeIR/quora` (attempted, blocked), `BeIR/climate-fever`,
`BeIR/dbpedia-entity`, `BeIR/cqadupstack`, `BeIR/webis-touche2020`, `allenai/qasper` (only
the mirror-hosted card was reachable, not an independent primary), `tau/scrolls`'s
non-NarrativeQA/Qasper members (GovReport, QMSum, SummScreenFD, QuALITY, ContractNLI),
XQuAD/XNLI (queried the HF API only, not individually fetched at primary — XNLI in
particular is commonly cited elsewhere as CC BY-NC 4.0, which would make it a second
FiQA-shaped mismatch if the HF mirror's blank license field gets naively treated as "no
restriction" by a future ingest pass — flag explicitly, do not assume PERMISSIVE_OK from
an absent tag).
