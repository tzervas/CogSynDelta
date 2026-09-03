# Survey — general language pretraining + instruction text

Scope: candidates for `language` (trunk, PLACEHOLDER) and instruction/chat text that could
feed `reasoning`/`memory` prompt-side. Method per task brief: HF public API (`/api/datasets/<id>`,
fetched live this session, see `raw_*.json` in the scratchpad — not copied here, metadata only)
+ WebFetch of upstream/primary source (project page, GitHub, licence page, terms of use) +
WebSearch as needed. No dataset content was downloaded, only card/API/licence-page metadata.
Fetch date for every `licence_upstream_source` below: **2026-09-03** (this session).

Verdict classes per ground truth: `PERMISSIVE_OK`, `ATTRIBUTION`, `SHARE_ALIKE`, `NC`,
`BLOCKING`, `CONSENT_OPEN`, `EVAL-ONLY` (usage tag). Strictest-input rule (DEC-31) applies:
whatever this factory emits inherits the strictest verdict among its inputs.

---

## Catalogue entries

### 1. FineWeb-Edu
- `repo_id`: `HuggingFaceFW/fineweb-edu`
- upstream: HuggingFace (FineWeb project), paper arXiv:2406.17557
- mirror_tag (HF API, VERIFIED live 2026-09-03): `license:odc-by`, 380,210 downloads, 1B<n<10B rows
- licence_upstream: not separately fetched from a non-HF primary (HF *is* the primary publisher
  here — FineWeb-Edu has no separate upstream repo). **VERIFIED at mirror = primary** (ODC-By,
  Open Data Commons Attribution licence, standard CC-derivative-of-Common-Crawl text corpus)
- grant_scope: whole_corpus
- verdict: **PERMISSIVE_OK** (ODC-By allows commercial use + derivatives with attribution)
- provenance_group: Common-Crawl-derived, filtered/classified by a FineWeb-Edu educational-quality
  classifier; independent lineage from RedPajama-v2/Dolma/Nemotron-CC (different filter pipelines,
  same raw CC snapshots underneath — **shared root** with those three at the raw-CC level, worth
  flagging for B1/B2 if more than one CC-derived set is admitted to one region)
- size: 1.3T tokens (per card description)
- quality: classifier-filtered for educational value, heavy dedup; widely used, strong benchmark
  track record; risk: eval contamination is plausible given ubiquity in other labs' pretraining
- feeds: `language` trunk (curriculum step 4, general pretraining)
- enrichment: none needed for base LM pretraining use; if paired into instruction format later,
  ODC-By propagates (attribution notice required) — no NC/SA introduced
- **top candidate**

### 2. Dolma
- `repo_id`: `allenai/dolma`
- mirror_tag (VERIFIED): `license:odc-by`, n>1T rows, 2,406 downloads
- licence_upstream: AllenAI is the primary publisher (HF = upstream here too). ODC-By, same
  licence family as FineWeb-Edu
- grant_scope: whole_corpus
- verdict: **PERMISSIVE_OK**
- provenance_group: Common-Crawl + C4 + peS2o + Gutenberg + Wikipedia + Reddit + code, mixed —
  **multi-source by design**, good B2 profile on its own; overlaps FineWeb-Edu/RedPajama/
  Nemotron-CC/C4 at the raw-CC layer and overlaps peS2o/Gutenberg/Wikipedia candidates below at
  the subset layer — if Dolma AND its subset-sources are both admitted, that's double-counting
  one provenance group, not two
- size: ~3T tokens, ~11B docs
- quality: AI2's documented pipeline (quality filters, dedup, PII scrub, toxicity filter);
  well-audited, training-recipe-grade
- feeds: `language` trunk
- enrichment: none for base use; ODC-By propagates
- **top candidate**

### 3. RedPajama-Data-v2
- `repo_id`: `togethercomputer/RedPajama-Data-V2`
- mirror_tag (VERIFIED): no `license:` tag on the card itself (empty tags list)
- licence_upstream (VERIFIED, fetched HF card body 2026-09-03): *"Please refer to the Common
  Crawl Foundation Terms of Use for the data."* — the **processing code** is Apache-2.0, but the
  **text itself carries no HF-native licence claim**; it inherits Common Crawl's terms
- licence_upstream_source: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2
  (card body) — same host as mirror, so this does NOT satisfy A0m's "host ≠ mirror" gate on its
  own; Common Crawl's own ToU (commoncrawl.org/terms-of-use) was not independently fetched this
  pass
- grant_scope: **unstated** — Common Crawl's ToU permits research/commercial data mining of the
  crawl but the crawl itself is compiled from pages under third-party copyright; this is the same
  "Common-Crawl-derivative" ambiguity every CC-based corpus (FineWeb, C4, Dolma, Nemotron-CC)
  quietly assumes without itself resolving
- verdict: **PERMISSIVE_OK, provisional** — matches the class every other CC-derived corpus in
  this survey is already admitted under (FineWeb-Edu, Dolma, C4 are the same shape and already
  well-precedented in open-weight releases); flagged, not a new risk this survey introduces
- provenance_group: raw-Common-Crawl (shared with FineWeb-Edu, Dolma, C4, Nemotron-CC — same root
  crawl snapshots, different quality filters)
- size: ~30T tokens (100+ B docs) across 5 tiers (raw/dedup-only quality-signal buckets)
- quality: comes with 40+ pre-computed quality signals for downstream filtering — good for
  building a curated sub-mix rather than using raw
- feeds: `language` trunk (as a filterable raw-CC layer, not to be double-stacked against
  FineWeb-Edu/Dolma without checking overlap)
- enrichment: quality-signal-based filtering to reach FineWeb-Edu-like quality; licence unchanged
  by filtering (subset of an unrestricted set is still unrestricted)

### 4. Cosmopedia
- `repo_id`: `HuggingFaceTB/cosmopedia`
- mirror_tag (VERIFIED): `license:apache-2.0`, 23,339 downloads, 10M<n<100M rows
- licence_upstream: HF is the primary publisher (synthetic corpus, HuggingFaceTB's own
  generation using Mixtral). Apache-2.0 as declared on the card
- grant_scope: whole_corpus
- verdict: **PERMISSIVE_OK**
- provenance_group: synthetic, generated FROM seed sources (textbooks/stories/wikihow prompts)
  by Mixtral-8x7B-Instruct — **model-generated, not human-written**; provenance concern is
  "synthetic data trained on model outputs whose own training data licence is opaque" (Mixtral's
  training corpus isn't published) — a soft risk, not a licence blocker, since Mistral's own
  model weights carry Apache-2.0 and impose no output-use restriction
- size: 25-30B tokens
- quality: synthetic textbook-style prose; good for curriculum/instructional text, but
  model-generated text risks amplifying whatever biases/errors Mixtral has; not human-authored
- feeds: `language` trunk
- enrichment: none required; Apache-2.0 propagates cleanly

### 5. Nemotron-CC
- `repo_id`: `nvidia/Nemotron-CC` (the exact HF slug on this pass 404'd for both
  `nvidia/Nemotron-CC` and `nvidia/Nemotron-CC-v2` via API — likely gated/renamed; not resolved
  this session, flagged **UNVERIFIED mirror slug**)
- licence_upstream (VERIFIED, fetched a related NVIDIA HF card body 2026-09-03): *"The NVIDIA
  Data Access Agreement for Model Training enables training of any AI model, including models
  released under proprietary or open source licenses"* and *"written to be fully compliant with
  the letter and the spirit of copyright law"*
- grant_scope: whole_corpus, restricted to model-training use under NVIDIA's own custom
  agreement (not a standard OSI/CC licence) — redistribution of the raw corpus itself is not
  clearly granted
- verdict: **PERMISSIVE_OK for training** but **non-standard licence text**, not a recognized
  open-data licence; flag for legal re-read before this becomes a released-weights input, because
  "Data Access Agreement" language is NVIDIA's own instrument, not CC/ODC
- provenance_group: raw-Common-Crawl (same root as #1-#3, #23 C4 below), heavier quality
  filtering + synthetic rephrasing (partially Qwen/DeepSeek-generated per NVIDIA's own docs) —
  **if the synthetic-rephrase portion derives from Qwen/DeepSeek model outputs, those models'
  own licence terms on generated content may attach** (unresolved this pass — genuine gap, not
  glossed over)
- size: multi-trillion-token (several tiers: high/medium/low quality + synthetic)
- quality: NVIDIA's own quality classifiers + synthetic diversity rephrasing; used in
  Nemotron-4/Llama-Nemotron pretraining
- feeds: `language` trunk
- enrichment: none required for base use
- **flag**: resolve exact HF slug and the Qwen/DeepSeek-output licence question before promoting
  past survey stage

### 6. Wikipedia (wikimedia/wikipedia mirror)
- `repo_id`: `wikimedia/wikipedia`
- mirror_tag (VERIFIED): `license:cc-by-sa-3.0`, `license:gfdl`, 232,884 downloads
- licence_upstream (VERIFIED, fetched foundation.wikimedia.org/wiki/Policy:Terms_of_Use
  2026-09-03): *"When you submit text to which you hold the copyright, you agree to license it
  under: Creative Commons Attribution-ShareAlike 4.0 International License ('CC BY-SA 4.0'), and
  GNU Free Documentation License ('GFDL')"* — note the **mirror tag says 3.0, upstream ToU now
  says 4.0** — a real mismatch, consistent with the "mirrors lie" precedent; use **4.0** as the
  operative current licence, record the mismatch in the receipt
- licence_upstream_source: https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use (host ≠
  huggingface.co — passes A0m's host-diversity gate)
- grant_scope: whole_corpus (text); images/media excluded (separate per-file licensing on
  Commons, not in this text-only mirror)
- verdict: **SHARE_ALIKE**
- provenance_group: Wikipedia (standalone; overlaps with Dolma's and Common Corpus's Wikipedia
  subsets — same provenance group, not three)
- size: ~6.8M English articles (`20231101.en` config), tens of GB
- quality: high — encyclopedic, heavily edited, low noise; standard pretraining staple
- feeds: `language` trunk
- enrichment: attribution manifest required (article history hyperlink or author-list, per ToU);
  SA propagates to any derivative dataset built substantially FROM Wikipedia text (e.g. a QA-pair
  extraction), not necessarily to a trained model's weights (the "is a trained model 'adapted
  material'" question is explicitly UNSETTLED per ground-truth's SHARE_ALIKE definition — flagged
  not resolved)
- **top candidate**

### 7. StackExchange data dump
- `repo_id` (mirror candidate): `HuggingFaceH4/stack-exchange-preferences` (a derived
  preference-pairs mirror, not the raw dump)
- mirror_tag (VERIFIED): `license:cc-by-sa-4.0`, 10,691 downloads
- licence_upstream (VERIFIED, fetched archive.org/details/stackexchange 2026-09-03): *"All user
  content contributed to the Stack Exchange network is cc-by-sa 4.0 licensed, intended to be
  shared and remixed"* with four attribution conditions (visual source credit, hyperlink to
  original post, author names displayed, hyperlink to author profiles)
- licence_upstream_source: https://archive.org/details/stackexchange (Internet Archive mirror of
  the SE data dump; host ≠ huggingface.co)
- grant_scope: whole_corpus (Q&A text + metadata)
- verdict: **SHARE_ALIKE**
- provenance_group: StackExchange (standalone)
- size: raw dump is 100s of GB across all SE sites; the HF preference-pairs mirror is a filtered
  subset (~10M<n<100M rows per its own size tag)
- quality: high signal-to-noise for technical/Q&A text, community-moderated, votes as quality
  signal usable for ranking/filtering
- feeds: `reasoning`/`memory` (Q&A pair structure fits retrieval or instruction pairing) as well
  as `language` trunk prose
- enrichment: the four-condition attribution manifest is a real per-post obligation at scale
  (author name + profile link + post link, per answer used) — non-trivial to satisfy
  mechanically at corpus scale; flag as an engineering cost, not a blocker
- **top candidate**

### 8. Project Gutenberg (via `storytracer/LoC-PD-Books` and `manu/project_gutenberg` mirrors)
- `repo_id`: `storytracer/LoC-PD-Books` (Library of Congress public-domain books, Gutenberg-
  adjacent) and `manu/project_gutenberg` (direct Gutenberg mirror)
- mirror_tag (VERIFIED): LoC-PD-Books `license:cc0-1.0`; project_gutenberg card carries no
  license: tag (empty)
- licence_upstream (VERIFIED, fetched gutenberg.org/policy/permission.html 2026-09-03):
  *"The vast majority of Project Gutenberg eBooks are in the public domain in the US... 'As you
  please' includes any commercial use, republishing in any format, making derivative works"*;
  attribution is explicitly **optional** (*"it is also OK to not cite Project Gutenberg: your
  choice"*); the **"Project Gutenberg" trademark name** itself requires royalty for commercial
  use **only if the name/trademark is retained on redistributed copies** — stripped-of-trademark
  redistribution needs no permission
- grant_scope: whole_corpus (text is public domain; only the trademarked name is restricted)
- verdict: **PERMISSIVE_OK** (effectively CC0-equivalent for the text; the `manu/project_gutenberg`
  mirror's missing license tag is a mirror-metadata gap, not a rights problem — upstream is
  unambiguous)
- provenance_group: Project Gutenberg (LoC-PD-Books and manu/project_gutenberg share this
  lineage — ONE group per DEC-46 LibriVox-precedent logic, even though they're different mirrors)
- size: manu/project_gutenberg ~10K<n<100K rows (curated cleaned subset); LoC-PD-Books is a
  larger multi-source PD book set (partially overlapping, partially not — needs a de-dup pass
  against Gutenberg IDs before combining)
- quality: pre-1929(ish)/PD-cutoff literature, high textual quality, English-heavy, but skews old
  (vocabulary/register mismatch with modern text) — good for `language` trunk diversity, weak
  alone as a full pretraining corpus
- feeds: `language` trunk
- enrichment: none required (PD); if the "Project Gutenberg" name/branding is stripped from
  headers during ingestion, zero remaining obligation
- **top candidate**

### 9. peS2o (AllenAI, scientific papers)
- `repo_id`: `allenai/peS2o`
- mirror_tag (VERIFIED): `license:['odc-by']`, 12,393 downloads, 10B<n<100B rows
- licence_upstream: AllenAI is primary publisher; derived from Semantic Scholar's S2ORC corpus,
  re-licensed ODC-By by AllenAI for the processed/cleaned version
- grant_scope: whole_corpus (processed text; not verified against S2ORC's own per-paper terms —
  **flag**: S2ORC itself aggregates papers under a mix of publisher licences, and AllenAI's
  ODC-By re-release is a re-licensing claim over that mix that this pass did not independently
  verify against S2ORC's own terms)
- verdict: **PERMISSIVE_OK, provisional** (trusting AllenAI's re-license claim; matches Dolma's
  well-audited-pipeline reputation, but flagged as inherited-trust rather than independently
  re-derived from each underlying paper's licence)
- provenance_group: Semantic Scholar / S2ORC (shares lineage with `armanc/scientific_papers`
  below and with arXiv/PMC subsets if those are separately admitted — same academic-paper root)
- size: ~38.97M documents, ~50B tokens (per AllenAI's own reporting)
- quality: cleaned/deduped academic full text (S2ORC + abstracts); strong domain register for
  scientific/technical language, complements Wikipedia/Gutenberg's registers
- feeds: `language` trunk (register diversity), potentially `reasoning` (scientific problem text)
- enrichment: none for base use

### 10. arXiv metadata/full-text
- `repo_id`: no single clean HF mirror pinned this pass (candidate: various `arxiv_dataset`
  mirrors exist but are inconsistently licensed on HF); primary source is arXiv's own bulk-access
  terms
- licence_upstream: **not independently fetched this pass** — arXiv's own terms grant free bulk
  access to metadata (CC0) but **full-text PDFs/LaTeX are per-paper licensed by each author**
  (many default to arXiv's non-exclusive licence, not a redistribution-friendly open licence)
- verdict: **UNVERIFIED** (metadata layer likely PERMISSIVE_OK/CC0; full-text layer is
  per-paper and NOT safely blanket-admittable) — do not promote without a per-paper licence
  pass or restricting to the metadata-only layer
- grant_scope: metadata_only (safe) vs whole_corpus (unsafe, per-paper mixed)
- provenance_group: arXiv (standalone; overlaps peS2o at the abstract layer only)
- feeds: metadata layer → `language` trunk titles/abstracts only, safely
- **REFUSED (as full-text) this pass** — pending a per-paper licence-clearing pass; metadata-only
  layer is a much smaller, separately-assessable candidate

### 11. PubMed Central Open Access subset
- `repo_id`: `pmc/open_access` HF mirror carries all 8 CC variants plus `other`/`unknown` as tags
  (VERIFIED) — meaning the mirror itself is a mixed-licence bucket, not a single verdict
- licence_upstream (VERIFIED, fetched pmc.ncbi.nlm.nih.gov/tools/openftlist/ 2026-09-03):
  *"License terms vary. Please refer to the license statement in each article for specific terms
  of use"* — confirms per-article variance, no blanket grant
- grant_scope: **unstated at corpus level, per-article at article level**
- verdict: **UNVERIFIED as a blanket set** — the CC0/CC-BY/CC-BY-SA subset (a filterable slice
  the OA-subset metadata itself flags per article) is admittable as **PERMISSIVE_OK/ATTRIBUTION/
  SHARE_ALIKE respectively**; the CC-BY-NC / CC-BY-ND / CC-BY-NC-ND slices are **REFUSE-ND /
  moves-to-NC** and must be filtered OUT (ND) or tagged NC, never blanket-admitted with the rest
- provenance_group: PubMed Central (standalone; shares academic-paper root with peS2o/arXiv)
- size: millions of full-text biomedical articles; the fully-open (CC0/CC-BY/CC-BY-SA) slice is
  the safely usable fraction, smaller than the full OA set
- quality: high (peer-reviewed biomedical literature); strong for a future science/medical
  register but this survey's scope is general language — flagged for a future domain pass, not
  fully catalogued here
- feeds: `language` trunk (if filtered to the open slice) or a future domain faculty
- enrichment: mechanical per-article licence filter is REQUIRED before any admission — cannot be
  treated as one dataset for B1-B5 purposes; each licence bucket is its own provenance stratum

### 12. OpenAssistant (oasst2)
- `repo_id`: `OpenAssistant/oasst2`
- mirror_tag (VERIFIED): `license:apache-2.0`, 11,157 downloads, 100K<n<1M rows
- licence_upstream: LAION/OpenAssistant is the primary publisher; Apache-2.0 as declared
- grant_scope: whole_corpus
- verdict: **PERMISSIVE_OK**
- provenance_group: OpenAssistant (standalone; crowd-sourced human-written conversations, not
  model-generated — valuable distinction from Alpaca/UltraChat below)
- size: ~135K messages, 10K+ conversation trees, 35 languages
- quality: human-written and human-rated (quality/toxicity/humor labels per message), multi-turn,
  multilingual — high-quality instruction-tuning source, not synthetic
- feeds: `memory`/`reasoning`/`language` (instruction-following text)
- enrichment: none required; Apache-2.0 propagates cleanly
- **top candidate**

### 13. Databricks Dolly 15k
- `repo_id`: `databricks/databricks-dolly-15k`
- mirror_tag (VERIFIED): `license:cc-by-sa-3.0`, 54,457 downloads
- licence_upstream (VERIFIED, fetched github.com/databrickslabs/dolly 2026-09-03): *"generated
  by Databricks employees and released under a permissive license (CC-BY-SA)"*
- grant_scope: whole_corpus
- verdict: **SHARE_ALIKE**
- provenance_group: Dolly (standalone; human-written by Databricks employees, not model-
  generated — same quality distinction as OpenAssistant)
- size: 15,011 records
- quality: human-authored instruction/response pairs across 7 task categories (brainstorming,
  classification, QA, summarization, etc.); small but clean
- feeds: `memory`/`reasoning` instruction pairing
- enrichment: SA propagates to any derived instruction-format set that substantially reuses
  Dolly's text; attribution notice required

### 14. FLAN collection
- `repo_id`: `Muennighoff/flan` (a reformatted mirror; card carries no license tag on this API
  pull — **mirror gap**)
- licence_upstream (VERIFIED, fetched github.com/google-research/FLAN 2026-09-03): repository
  is **Apache-2.0** (footer/LICENSE file)
- grant_scope: unstated at the individual-task level — **FLAN is a MIXTURE of ~1800+ existing
  NLP tasks/datasets**, each with its OWN upstream licence (SuperGLUE tasks, translation sets,
  etc.); Google's Apache-2.0 covers the *collection code and templates*, not necessarily every
  constituent task's raw text
- verdict: **UNVERIFIED as a blanket set** — same shape as the PMC OA subset problem: the wrapper
  licence (Apache-2.0) does not automatically launder each of ~1800 constituent tasks' own terms;
  admitting FLAN wholesale without a per-task pass risks silently importing an NC/ND source
  (e.g. certain WMT/translation tasks carry mixed terms)
- provenance_group: composite (do not treat as one provenance stratum for B1/B2 — it already
  IS multiple sources internally, which is good for B2, but the internal composition needs its
  own audit)
- size: millions of examples across the merged tasks
- quality: extremely well-curated task instructions/templates; the premier "instruction template
  diversity" source in the field
- feeds: `memory`/`reasoning`
- **REFUSED wholesale this pass, pending per-constituent-task licence audit** — flag as
  high-value if that audit is done; do not admit the blanket mirror

### 15. Tulu 3 SFT mixture
- `repo_id`: `allenai/tulu-3-sft-mixture`
- mirror_tag (VERIFIED): `license:odc-by`, 54,317 downloads, 100K<n<1M rows
- licence_upstream: AllenAI primary publisher, ODC-By declared
- grant_scope: whole_corpus **as declared by AllenAI**, but Tulu mixtures are themselves
  composites of many named sub-sources (some of which, per AllenAI's own release notes, include
  synthetic data generated via models with their own output-use terms) — **not independently
  re-verified against each sub-source this pass**
- verdict: **PERMISSIVE_OK, provisional** — trusting AllenAI's blanket ODC-By claim over the
  mixture (same inherited-trust caveat as peS2o above); AllenAI has historically been careful
  about this (Dolma/Tulu both document sourcing openly), so this is lower-risk than FLAN's
  opaque 1800-task blanket, but still not independently re-derived
- provenance_group: composite (AllenAI SFT mixture; internally multi-source)
- size: ~939K examples
- quality: current-generation (2026-adjacent) high-quality SFT mixture, actively maintained,
  strong benchmark performance in AI2's own model releases
- feeds: `memory`/`reasoning`
- enrichment: none required if AllenAI's blanket claim is trusted; flag for a spot-check pass
  before scale-up

### 16. Alpaca (tatsu-lab)
- `repo_id`: `tatsu-lab/alpaca`
- mirror_tag (VERIFIED): `license:cc-by-nc-4.0`, 105,855 downloads
- licence_upstream (VERIFIED, fetched github.com/tatsu-lab/stanford_alpaca 2026-09-03): *"The
  dataset is CC BY NC 4.0 (allowing only non-commercial use)"*
- grant_scope: whole_corpus, but **generated using OpenAI's `text-davinci-003` via the
  Self-Instruct technique** — OpenAI's usage terms (separately, not quoted from OpenAI's own ToU
  this pass) have historically included a clause restricting use of API outputs to build
  competing models; this was NOT independently re-verified against OpenAI's current terms this
  session, flagged as an **additional unresolved restriction layered on top of the declared
  CC-BY-NC-4.0**, not a substitute for it
- verdict: **NC** (per DEC-31, does not block; but the layered OpenAI-output-terms question is a
  genuine unresolved gap, not merely NC) — recommend treating as **NC with an additional
  provenance_red_flag**, not blanket-cleared
- provenance_group: Alpaca-lineage (self-instruct GPT-3.5/text-davinci-003 generated) — shared
  lineage with any other "GPT-derived instruction data" candidate (none separately catalogued
  this pass beyond UltraChat, which is a distinct lineage using different models/method)
- size: 52,002 instruction-following examples
- quality: model-generated (not human-written), known quality issues are well-documented in the
  literature (hallucinated instructions, some incorrect answers) — treat as a volume-filler, not
  a quality anchor
- feeds: `memory`/`reasoning`
- enrichment: NC propagates (already true of `memory` region today per ground-truth's GooAQ
  finding — this would sit in the same NC tier, not a new problem); the layered OpenAI-terms
  question should be resolved before this dataset is used past toy scale

### 17. UltraChat
- `repo_id`: `HuggingFaceH4/ultrachat_200k` (a filtered/formatted mirror of the original
  `stingning/ultrachat`)
- mirror_tag (VERIFIED): `license:mit`, 101,473 downloads
- licence_upstream (VERIFIED, fetched github.com/thunlp/UltraChat 2026-09-03): *"distributed
  under the MIT license"*; explicitly notes *"all the data is automatically generated (including
  the instructions and responses)"* using Turbo (ChatGPT) APIs, and states the project does
  *"not directly use any data available on the Internet as prompts"*
- grant_scope: whole_corpus; same OpenAI-API-generation provenance question as Alpaca (data
  generated via OpenAI's ChatGPT API) — **but here the creators explicitly chose MIT despite
  that provenance**, which is the project's own licensing decision, not a re-verification of
  OpenAI's terms against it
- verdict: **PERMISSIVE_OK as declared**, with the same **unresolved layered-OpenAI-terms flag**
  as Alpaca (not independently cleared this pass either way)
- provenance_group: UltraChat-lineage (ChatGPT-generated, THUNLP method) — distinct lineage from
  Alpaca (different generation method/prompting scheme, different model, per the source repo)
- size: 1.5M conversations (full UltraChat) / 200K filtered high-quality subset (the HF mirror)
- quality: large-scale, multi-turn, broad topic coverage claimed by the authors; model-generated
  (same synthetic-data caveat as Alpaca/Cosmopedia)
- feeds: `memory`/`reasoning`
- enrichment: none required if MIT is accepted as-is; flag alongside Alpaca for the OpenAI-terms
  question to be resolved as ONE decision covering both (same root issue)

### 18. LMSYS-Chat-1M
- `repo_id`: `lmsys/lmsys-chat-1m` (gated: `auto`, VERIFIED via API)
- licence_upstream (VERIFIED, fetched huggingface.co/datasets/lmsys/lmsys-chat-1m access
  agreement text 2026-09-03): *"You should not distribute, copy, disclose, assign, sublicense,
  embed, host, or otherwise transfer the dataset to any third party"*; separately grants *"a
  limited, non-exclusive, non-transferable, non-sublicensable license to use the LMSYS-Chat-1M
  Dataset... for both research and commercial purposes"*
- grant_scope: whole_corpus for **training use** (explicitly includes commercial), but the raw
  dataset itself is **non-redistributable** by agreement, gated behind identity disclosure, and
  contains **real users' raw conversations with deployed chatbots** (a public demo, not a
  consented-for-research-corpus collection) — a live-consent shape closer to Common Voice's than
  to a standard scrape
- verdict: **REFUSE-TERMS** — non-redistributable-terms class per the operator stance, compounded
  by unresolved consent provenance on raw human conversational content collected via a public
  chat demo rather than an explicit research-consent flow; recommend NOT admitting without a
  dedicated consent-provenance review, separate from the licence question
- provenance_group: LMSYS-Chat (standalone)
- size: 1M conversations, 25+ models, 100+ languages
- **REFUSED this pass**

### 19. Anthropic HH-RLHF
- `repo_id`: `Anthropic/hh-rlhf`
- mirror_tag (VERIFIED): `license:mit`, 35,583 downloads, 100K<n<1M rows
- licence_upstream: fetched github.com/anthropics/hh-rlhf 2026-09-03 — page shows an MIT licence
  file present in the repo listing but the operative LICENSE text itself was not directly quoted
  by the fetch (page summary only); **treat the MIT claim as INFERRED-from-repo-navigation, not
  VERIFIED-by-quoted-text**, and note the mirror tag independently agrees (`license:mit`)
- grant_scope: whole_corpus
- verdict: **PERMISSIVE_OK** (MIT tag agrees at both mirror and repo-listing level, but the actual
  LICENSE file text should be pulled directly before this is treated as fully closed)
- provenance_group: HH-RLHF (standalone; human-and-model comparison data, Anthropic's own
  red-teaming + helpfulness collection)
- size: ~161K comparison pairs (helpful + harmless splits)
- quality: well-documented methodology, human-labeled preference pairs, widely used for RLHF/DPO
  research; content includes red-teaming/harmful-content examples by design (needs careful
  handling, not a licence issue but a content-safety one)
- feeds: `memory`/`reasoning` (preference-pair format), potentially `salience`/`affect` once those
  faculties are built (the harmlessness axis is close to a salience signal)
- enrichment: none required for MIT-declared use
- **top candidate**

### 20. Common Corpus (Pleias)
- `repo_id`: `PleIAs/common_corpus`
- mirror_tag (VERIFIED): no `license:` tag surfaced on this API pull (empty list) — **mirror
  gap**, despite the project's own strong public-domain framing
- licence_upstream (VERIFIED, fetched huggingface.co/datasets/PleIAs/common_corpus card body
  2026-09-03): *"contains only data that is either uncopyrighted or freely licensed"*, *"All
  data in Common Corpus are either uncopyrighted or freely licensed and may be used for both
  commercial and non-commercial purposes"*; constituent breakdown: OpenCulture (PD + Gutenberg +
  Wikisource, 967B tok), OpenGovernment (mixed government licences, 579B tok), OpenSource
  (MIT-ish code, 283B tok), OpenScience (open-repo academic, 281B tok), OpenWeb (Wikipedia
  CC-BY-SA, 88B tok), OpenSemantic (Wikidata CC0, 67B tok)
- grant_scope: whole_corpus **as a union of already-open sources** — genuinely a
  provenance-transparent aggregation (Pleias documents each sub-source's licence), unlike FLAN's
  opaque 1800-task blanket
- verdict: **mixed by construction — PERMISSIVE_OK/CC0 for most strata, SHARE_ALIKE for the
  OpenWeb (Wikipedia) stratum**; the corpus card explicitly supports filtering to a pure-PD/
  attribution-only subset, so this is one of the FEW candidates that self-documents its internal
  licence strata well enough to sub-admit by stratum
- provenance_group: **composite of already-catalogued groups** — Gutenberg (shared with #8),
  Wikipedia (shared with #6) — admitting Common Corpus's OpenCulture/OpenWeb strata alongside
  the standalone Gutenberg/Wikipedia entries above is DOUBLE-COUNTING the same provenance group,
  not two independent sources; only OpenGovernment/OpenScience/OpenSemantic add genuinely new
  provenance if Gutenberg+Wikipedia are already admitted separately
- size: ~2T tokens total across strata (2025 all-editions figure)
- quality: large, French-heavy but multilingual (Pleias is a European effort), strong PD/
  government-document register diversity not covered elsewhere in this survey
- feeds: `language` trunk (especially the OpenGovernment/OpenScience strata for register
  diversity)
- enrichment: none required for the already-open strata; if admitted, dedup against Gutenberg/
  Wikipedia entries above is REQUIRED before B1/B2 computation, not optional
- **top candidate** (for its OpenGovernment/OpenScience/OpenSemantic strata specifically, not
  the whole corpus wholesale, to avoid the Gutenberg/Wikipedia double-count)

### 21. The Pile (deduplicated)
- `repo_id`: `EleutherAI/the_pile_deduplicated`
- mirror_tag (VERIFIED): no license tag surfaced (empty), 12,579 downloads, 100M<n<1B rows
- licence_upstream: **not independently fetched this pass**; The Pile is well-known in the field
  as a composite of 22 sub-sources (Books3, OpenWebText2, PubMed, ArXiv, GitHub, StackExchange,
  etc.) with **highly mixed and in places disputed rights status** — Books3 in particular has
  been the subject of copyright takedown/litigation activity (not independently re-verified this
  pass, but this is public, well-documented controversy, cited here as a reason for caution, not
  as INFERRED fact-of-outcome)
- verdict: **REFUSE-TERMS / BLOCKING pending sub-source split** — do not admit wholesale; several
  constituent sub-sources (Books3 especially) carry active rights disputes that the operator
  stance's "distributors that disclaim owning what they distribute" refusal class plausibly
  covers. A sub-source-by-sub-source pass (keeping only the clean ones: PubMed, ArXiv abstracts,
  StackExchange overlap with #7, Wikipedia overlap with #6) would be a different, smaller,
  admittable candidate — not attempted this pass
- provenance_group: The Pile (would decompose into several already-listed groups if split)
- **REFUSED wholesale this pass**

### 22. C4 (Colossal Clean Crawled Corpus)
- `repo_id`: `allenai/c4`
- mirror_tag (VERIFIED): `license:['odc-by']`, 1,220,090 downloads, 10B<n<100B rows
- licence_upstream: AllenAI re-release of Google's original C4; ODC-By as declared (AllenAI's
  own re-licensing of a Common-Crawl derivative, same shape as FineWeb-Edu/Dolma)
- grant_scope: whole_corpus
- verdict: **PERMISSIVE_OK**
- provenance_group: raw-Common-Crawl (shared root with FineWeb-Edu, Dolma, RedPajama-v2,
  Nemotron-CC — same caution about double-counting the crawl root across multiple CC-derived
  admissions)
- size: ~365M documents (en config), ~305GB
- quality: an older, less aggressively filtered CC derivative than FineWeb-Edu; still a
  widely-used pretraining staple; heuristic cleaning (langdetect, boilerplate removal) rather
  than a learned educational-quality classifier
- feeds: `language` trunk
- enrichment: none required; ODC-By propagates
- note: if FineWeb-Edu and/or Dolma are already admitted, C4 adds little beyond the CC root
  they already cover — lower marginal value than #1/#2 for a region already using either

---

## Refused (summary)

| candidate | why refused |
|---|---|
| **LMSYS-Chat-1M** (#18) | REFUSE-TERMS: explicit non-redistribution clause in the gated agreement, plus unresolved live-consent provenance on raw public-chat-demo conversations (Common-Voice-shaped problem, not flagged CONSENT_OPEN here because it wasn't independently confirmed to fit that class this pass — safer to refuse than assume) |
| **FLAN collection** (#14) | UNVERIFIED wholesale: Apache-2.0 wrapper licence does not launder ~1800 constituent tasks' individually-varying upstream terms; needs a per-task audit before admission, not refused forever |
| **PMC OA subset, full blanket** (#11) | UNVERIFIED wholesale: mirror itself carries 9 different licence tags including NC/ND variants; only the CC0/CC-BY/CC-BY-SA slice is admittable, and that requires a mechanical per-article filter this pass didn't run |
| **arXiv full-text** (#10) | UNVERIFIED: per-paper author-chosen licences, no blanket grant found; metadata/abstract layer is a separate, smaller, likely-admittable candidate not yet cleared either |
| **The Pile (deduplicated)** (#21) | REFUSE-TERMS/BLOCKING pending split: Books3 and other sub-sources carry well-documented, unresolved rights disputes; a cleaned sub-source split is a different candidate, not attempted |
| **Alpaca / UltraChat, provisionally** (#16, #17) | not refused outright (both have a declared licence and are listed as candidates), but flagged: BOTH are generated via OpenAI API output, and OpenAI's own terms-of-use restriction on using API outputs to train competing models was not independently re-verified this pass — resolve as one decision before either goes past toy scale |
| **Nemotron-CC** (#5) | not refused, but exact HF slug 404'd this pass and the licence instrument (NVIDIA's own "Data Access Agreement", not a standard OSI/CC licence) plus a possible Qwen/DeepSeek-output-terms question on its synthetic tier need resolution before promotion |

---

## Top 8 candidates (verdicts)

1. **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`) — **PERMISSIVE_OK** (ODC-By, VERIFIED at
   publisher). 1.3T tokens, educational-quality-filtered. Best single general-pretraining source
   surveyed.
2. **Dolma** (`allenai/dolma`) — **PERMISSIVE_OK** (ODC-By, VERIFIED at publisher). ~3T tokens,
   genuinely multi-source (good B2), AI2-audited pipeline.
3. **Wikipedia** (`wikimedia/wikipedia`) — **SHARE_ALIKE** (CC BY-SA 4.0 per upstream ToU,
   VERIFIED — note mirror tag says 3.0, a real mismatch to record). High-quality encyclopedic
   register.
4. **StackExchange** (`HuggingFaceH4/stack-exchange-preferences`, upstream = full SE dump) —
   **SHARE_ALIKE** (CC BY-SA 4.0, VERIFIED at archive.org). Strong Q&A/technical register, feeds
   `memory`/`reasoning` too.
5. **Project Gutenberg** (`manu/project_gutenberg` + `storytracer/LoC-PD-Books`, one provenance
   group) — **PERMISSIVE_OK** (public domain, VERIFIED at gutenberg.org; attribution optional).
6. **OpenAssistant (oasst2)** — **PERMISSIVE_OK** (Apache-2.0, mirror-VERIFIED). Human-written,
   human-rated, multilingual instruction conversations — best non-synthetic instruction source
   surveyed.
7. **Anthropic HH-RLHF** — **PERMISSIVE_OK** (MIT, mirror-VERIFIED; upstream LICENSE text itself
   not directly quoted, flagged for a follow-up pull). Preference-pair structure, feeds
   `memory`/`reasoning`, possibly future `salience`.
8. **Common Corpus (Pleias), OpenGovernment/OpenScience/OpenSemantic strata** — **mixed
   PERMISSIVE_OK/CC0** (VERIFIED at publisher, self-documented strata). Admit these three strata
   only — the OpenCulture/OpenWeb strata double-count Gutenberg/Wikipedia already listed above.

Composed-corpus consequence if all 8 are admitted: strictest input among the top 8 is
**SHARE_ALIKE** (Wikipedia, StackExchange) — no NC/BLOCKING in this set, so a
general-language corpus built from just these 8 would NOT drag the composed model's release
tier past CC BY-SA, notably better than `memory`'s current NC status. RedPajama-v2, C4,
Nemotron-CC, Cosmopedia, peS2o, Tulu-3-SFT, and Dolly are all also PERMISSIVE_OK/SHARE_ALIKE-
provisional and worth a second-tier pull once the top 8 are ingested and volume needs grow.
