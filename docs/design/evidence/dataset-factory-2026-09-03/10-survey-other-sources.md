# Non-HF dataset survey — sweep of Kaggle, Zenodo, OpenML, data.gov/EU, Common Crawl
# derivatives, academic mirrors, GitHub, Papers with Code, Wikimedia, StackExchange,
# Internet Archive

Session date: 2026-09-03. Method: WebSearch to find the candidate, then WebFetch **the
primary/upstream page itself** (Zenodo record, GitHub LICENSE file, project's own legal
page) for the licence — never trusted a mirror description alone. No dataset content was
downloaded; metadata/licence pages only. `secret ls | grep -i kaggle` returned no entry, so
Kaggle candidates were checked via public dataset pages, not the API.

Verdict classes and fields per `00-ground.md` §(b)/(c). Composed-model rule still applies:
**strictest input wins** — an NC or SHARE_ALIKE addition does not block, ND/BLOCKING does.

---

## 1. EuroSAT

- **source**: Zenodo record 7711810 (https://zenodo.org/records/7711810)
- **faculty fit**: `visual` — the most data-starved faculty (tiny-imagenet and flickr30k
  both BLOCKING). Sentinel-2 satellite RGB, 27,000 labelled images, 10 land-use classes.
- **licence_upstream (VERIFIED, fetched from the Zenodo record itself)**: "MIT License" —
  "A short and simple permissive license with conditions only requiring preservation of
  copyright and license notices."
- **licence_upstream_source**: https://zenodo.org/records/7711810 — fetched 2026-09-03
- **mirror_tag**: also mirrored on HF (`tanganke/eurosat`, `nielsr/eurosat` etc.) — prefer
  this Zenodo primary per DEC-57.
- **verdict**: PERMISSIVE_OK
- **grant_scope**: whole_corpus
- **provenance_group**: EuroSAT (single group; RGB and multispectral variants share
  lineage)
- **redistribute**: none required beyond copyright/licence notice preservation
- **why**: clean MIT grant at the primary source, directly on-target for the faculty with
  the least headroom today; small enough (27k images) to fully re-derive/re-check.
- **caveat**: single narrow domain (land use classes) — would need pairing with a second
  visual source to clear B2 (N_eff ≥ 3) on its own.
- **enrichment plan**: patch-token pipeline is already built for `visual` (JEPA-style);
  just needs class-balance check against B5 (10 classes, roughly balanced by construction —
  low risk).

## 2. Visual Genome

- **source**: Stanford, https://homes.cs.washington.edu/~ranjay/visualgenome/index.html
- **faculty fit**: `visual` — region descriptions + dense annotations, richer than
  classification-only sets; also usable as image/text pairs for cross-faculty (reserve
  corpus) candidates later.
- **licence_upstream (VERIFIED)**: page states "Creative Commons Attribution 4.0
  International License," linking http://creativecommons.org/licenses/by/4.0/.
- **licence_upstream_source**: https://homes.cs.washington.edu/~ranjay/visualgenome/index.html
  — fetched 2026-09-03
- **verdict**: ATTRIBUTION
- **grant_scope**: whole_corpus (annotations); images are drawn from MS-COCO/YFCC100M
  sources with their own per-image terms — same "annotations vs. images" split as COCO
  below, so **grant_scope should be recorded as `metadata_only` for the images layer** even
  though the annotation layer is a clean CC BY 4.0 grant.
- **provenance_group**: shares image lineage with MS-COCO (overlapping image sets) —
  should be one provenance group with COCO for B1/B2 purposes if both are admitted.
- **redistribute**: attribution (TASL) required
- **why**: strong second `visual` source distinct in kind (dense region graphs, not just
  class labels) from EuroSAT — helps B2 without duplicating EuroSAT's single-domain
  narrowness.
- **caveat**: image-licence layer is unverified per-image (Flickr/YFCC-sourced); only the
  annotation text (region descriptions, relationships) is confidently ATTRIBUTION-clean —
  treat image pixels as `metadata_only` grant until per-image licence audit is done.

## 3. COCO (Common Objects in Context)

- **source**: https://cocodataset.org/#termsofuse
- **faculty fit**: `visual`
- **licence_upstream (VERIFIED via search of the terms-of-use page + corroborating
  secondary discussion, primary page content itself did not render fetchable text)**:
  annotations licensed CC BY 4.0 by the COCO Consortium; **images are not owned by the
  Consortium** — they are the original Flickr photos, "use of the images must abide by the
  Flickr Terms of Use," and downstream analysis found the majority of person-class images
  with CC licences requiring attribution that COCO redistribution does not provide.
- **licence_upstream_source**: https://cocodataset.org/#termsofuse — attempted fetch
  2026-09-03 (page returned nav shell only, terms confirmed via GitHub issue
  cocodataset/cocoapi#551 discussion of the same terms page)
- **verdict**: mixed — annotations ATTRIBUTION, images **grant_scope: metadata_only /
  unstated** (per-photo Flickr terms, unverified at scale)
- **provenance_group**: COCO (shared with Visual Genome's image overlap, see above)
- **why**: same-category candidate as Visual Genome, larger (330k images) but the image
  layer's per-photo licence status is the weaker of the two — do not treat as a clean
  whole-corpus grant.
- **caveat**: this is the shape DEC-57's `grant_scope` field exists to catch — "annotations
  clean, images unstated" is exactly AudioSet's `metadata_only` problem transposed to
  vision. Needs a full fetch of the terms page (the fetch tool could not render the anchor
  section) before treating as anything beyond provisional.

## 4. Wikidata

- **source**: https://www.wikidata.org/wiki/Wikidata:Licensing
- **faculty fit**: `memory` (structured facts as retrieval/consolidation targets),
  `numeric/math` placeholder (many statements are literally numeric), `reasoning`
  (entity-relation reasoning items)
- **licence_upstream (VERIFIED)**: "All structured data in the main, property and lexeme
  namespaces is made available under the Creative Commons CC0 License." Other namespace
  text is CC BY-SA 4.0.
- **licence_upstream_source**: https://www.wikidata.org/wiki/Wikidata:Licensing — fetched
  2026-09-03
- **verdict**: PERMISSIVE_OK (structured data proper); the CC BY-SA layer for
  prose/lexeme-gloss text is SHARE_ALIKE
- **grant_scope**: whole_corpus for the CC0 structured layer
- **provenance_group**: Wikidata (single group; distinct from Wikipedia/Wiktionary despite
  shared foundation)
- **redistribute**: none for CC0 layer
- **why**: best-in-class permissiveness and breadth (100M+ items); could seed the
  PLACEHOLDER `numeric/math` faculty (quantity statements, units, dates) without touching
  the reasoning/gsm8k corpus that is currently "spent."
- **enrichment plan**: needs a values-not-BPE-pieces extraction pass to match `numeric`'s
  design (tokens over values); dump via Wikimedia's own dumps.wikimedia.org, not scraped
  live.

## 5. Stack Exchange Data Dump

- **source**: Internet Archive, https://archive.org/details/stackexchange (official dumps,
  mirrored from Stack Exchange Inc.)
- **faculty fit**: `memory` (retrieve/compress) — direct replacement/supplement for the
  79.1%-concentrated, NC-tagged GooAQ share
- **licence_upstream (VERIFIED)**: "Creative Commons Attribution-ShareAlike 4.0
  International (CC-BY-SA 4.0)," with explicit attribution requirements: visually indicate
  Stack Exchange Network as source, hyperlink to the original question, display author
  names, and hyperlink each author name to their profile page — "in standard HTML... without
  any 'nofollow' command."
- **licence_upstream_source**: https://archive.org/details/stackexchange — fetched
  2026-09-03
- **verdict**: SHARE_ALIKE
- **grant_scope**: whole_corpus
- **provenance_group**: Stack Exchange Network (single group — covers StackOverflow,
  math.SE, physics.SE, etc.; all one lineage per the LibriVox precedent)
- **redistribute**: attribution + share-alike + live-hyperlink-back requirement — this last
  clause is unusual (a functioning hyperlink back to a live profile page) and should be
  flagged as a **provenance_red_flag**: "attribution clause requires a live outbound
  hyperlink per answer/author, which a static training corpus cannot satisfy structurally —
  needs a legal read on whether inference-time weights trigger this at all, distinct from a
  redistributed dataset."
- **why**: this is the standout candidate for de-concentrating `memory` away from GooAQ.
  Millions of Q&A pairs, permissively licensed relative to NC, real retrieval structure
  (question/accepted-answer/vote signal).
- **caveat**: the SHARE_ALIKE "is a trained model adapted material" question is explicitly
  unsettled per `LICENCE-FOR-OPEN-WEIGHTS.md` — same open question SNLI's `compress` half
  already carries, so this doesn't introduce a new category of risk, just more volume in an
  existing one. NOT MIT-tier; would keep the faculty in ATTRIBUTION/SHARE_ALIKE tier at
  best, doesn't reach PERMISSIVE_OK.

## 6. DeepMind Mathematics Dataset

- **source**: GitHub, https://github.com/google-deepmind/mathematics_dataset
- **faculty fit**: `numeric/math` PLACEHOLDER — directly unblocks it without cannibalizing
  `reasoning`'s gsm8k/aqua_rat
- **licence_upstream (VERIFIED — repo licence badge/footer; LICENSE file itself not
  rendered by the fetch, standard Apache-2.0 boilerplate)**: Apache License 2.0
- **licence_upstream_source**:
  https://github.com/google-deepmind/mathematics_dataset (LICENSE file path
  /blob/master/LICENSE) — fetched 2026-09-03
- **verdict**: PERMISSIVE_OK
- **grant_scope**: whole_corpus (it's a generator, not a scrape — infinite procedurally
  generated math problems with exact answers)
- **provenance_group**: DeepMind Mathematics Dataset (synthetic, single group)
- **why**: procedurally generated means B1/B2/B5 balance is controllable by construction
  (choose the category mix), and it's an obvious fit for `numeric/math`'s "tokens over
  values, not BPE pieces" design brief since the generator can emit numeric ground truth
  directly.
- **enrichment plan**: run the generator at controlled category caps; treat "category" as
  the B5 stratum key.

## 7. MATH (Hendrycks et al.)

- **source**: GitHub, https://github.com/hendrycks/math
- **faculty fit**: `numeric/math` / `reasoning` (competition mathematics, step-by-step
  solutions — could feed the reserve's "executable items" idea only loosely; better fit is
  numeric)
- **licence_upstream (VERIFIED — fetched the raw LICENSE file directly)**: full MIT License
  text, copyright Dan Hendrycks 2021, standard permissive grant.
- **licence_upstream_source**:
  https://raw.githubusercontent.com/hendrycks/math/main/LICENSE — fetched 2026-09-03
- **verdict**: PERMISSIVE_OK
- **grant_scope**: whole_corpus
- **provenance_group**: MATH (single group; distinct from DeepMind Mathematics Dataset
  despite overlapping subject)
- **why**: cleanest possible licence text (verbatim MIT, fetched raw), high quality
  (competition-sourced, human-authored step solutions) — a strong pair with the DeepMind
  generator for `numeric/math`'s N_eff ≥ 3 requirement since it's a genuinely different
  source and generation method (human-authored vs. procedural).
- **caveat**: original test set was later shown partially contaminated in some public
  model evals — irrelevant to licence, relevant to eval design if used as a held-out
  probe rather than training-only.

## 8. bAbI tasks

- **source**: GitHub (Facebook Archive), https://github.com/facebookarchive/bAbI-tasks
- **faculty fit**: `reasoning` — synthetic multi-hop reasoning tasks (20 categories:
  induction, deduction, counting, path-finding, etc.)
- **licence_upstream (VERIFIED — footer/badge confirms BSD; full LICENSE.md text not
  rendered by the fetch due to a length constraint in the fetch call, re-fetch recommended
  before ingest)**: BSD License, standard clause requiring "Redistributions of source code
  must retain the above copyright notice, this list of conditions and the following
  disclaimer."
- **licence_upstream_source**:
  https://raw.githubusercontent.com/facebookarchive/bAbI-tasks/master/LICENSE.md — fetched
  2026-09-03 (partial; get the full text before ingest, this entry's verbatim text is
  incomplete)
- **verdict**: PERMISSIVE_OK (pending full-text re-fetch to confirm no additional clauses)
- **grant_scope**: whole_corpus
- **why**: synthetic and licence-clean, could bolster `reasoning`'s N_eff after
  gsm8k/aqua_rat were marked "spent" — but tasks are toy-scale and templated, so it's a
  volume/diversity patch, not a quality upgrade.
- **caveat**: archived repo (Meta stopped maintaining it); templated generation means B5
  stratum balance needs checking per task-type, not assumed.

## 9. Project Gutenberg

- **source**: https://www.gutenberg.org/policy/permission.html
- **faculty fit**: `language` (trunk) PLACEHOLDER — foundation/curriculum text corpus
- **licence_upstream (VERIFIED)**: "The vast majority of Project Gutenberg eBooks are in
  the public domain in the US. This means that nobody can grant, or withhold, permission to
  do with this item as you please," including commercial use, republishing, derivative
  works. Explicit: "No permission is needed to use quotes from Project Gutenberg items...
  This applies for all use, including commercial use."
- **licence_upstream_source**: https://www.gutenberg.org/policy/permission.html — fetched
  2026-09-03
- **verdict**: PERMISSIVE_OK for the public-domain majority
- **grant_scope**: whole_corpus, **conditional per-item** — the policy itself flags that
  "some copyrighted works exist in the collection" needing separate permission, so
  `grant_scope` should really be tracked per-book, not corpus-wide.
- **provenance_group**: Project Gutenberg (but should be split by public-domain-confirmed
  vs. not, if a fine-grained pass is done)
- **redistribute**: none for the PD majority; PG's own trademark/header rules apply to
  their specific file headers, not the underlying text
- **why**: classic, large, clean source for the language trunk's foundation-corpora need;
  no rights-holder consent risk on the PD-confirmed set.
- **caveat**: works skew old (pre-1929 US PD cutoff dominates) — stylistically dated
  relative to modern text; would need pairing with a contemporary permissive source for
  register diversity.

## 10. EU Open Data Portal (data.europa.eu)

- **source**: https://data.europa.eu/en/legal-notice
- **faculty fit**: `language` (trunk), general enrichment across faculties (structured
  government/statistical data touches `numeric` too)
- **licence_upstream (VERIFIED)**: editorial content owned by the EU: reuse allowed "provided
  appropriate credit is given and any changes are indicated" — CC BY 4.0. Metadata: CC0 1.0
  waived. Individual published resources: "Most of the resources published display a
  specific reference to the licence under which the owner has chosen to release them" — i.e.
  **the portal is an aggregator, not a uniform licence**, per-dataset check required.
- **licence_upstream_source**: https://data.europa.eu/en/legal-notice — fetched 2026-09-03
- **verdict**: ATTRIBUTION for EU-owned editorial content and CC0 for portal metadata; every
  individual dataset needs its own verdict (this entry is a **source class**, not a single
  dataset)
- **why**: government open-data portals are a reliable low-risk category (public-sector
  reuse mandates are common), good hunting ground for the `language` trunk's need for
  register diversity beyond literary/Gutenberg text.
- **caveat**: this is an index, not a dataset — record it as a **source**, not admit it
  directly; each pull needs its own per-resource licence capture.

## 11. data.gov (US)

- **source**: https://www.data.gov/privacy-policy/ (data policy section)
- **faculty fit**: `language` trunk, general
- **licence_upstream (VERIFIED)**: "Data and content created by government employees within
  the scope of their employment are not subject to domestic copyright protection under 17
  U.S.C. § 105" — public domain by default for US federal works; page flags to check each
  dataset's "Access & Use Information" for exceptions (state/local/contractor-authored data
  on the portal may carry different terms).
- **licence_upstream_source**: https://www.data.gov/privacy-policy/ — fetched 2026-09-03
- **verdict**: PERMISSIVE_OK for confirmed-federal-authored content; **per-dataset check
  required** for the rest (state, local, or third-party-contributed catalog entries are not
  automatically public domain)
- **why**: same class of value as the EU portal — public-domain-by-default government text
  (Congressional Record, agency reports, statistical releases) is essentially licence-risk-
  free once federal authorship is confirmed.
- **caveat**: also an index/aggregator, not a single dataset — record as a source class.

## 12. GDELT Project

- **source**: https://www.gdeltproject.org/about.html
- **faculty fit**: `reasoning` (event/causal structure), `language` trunk (news-derived
  text, though GDELT itself is mostly structured event records, not full article text)
- **licence_upstream (VERIFIED)**: "all datasets released by the GDELT Project are available
  for unlimited and unrestricted use for any academic, commercial, or governmental use of
  any kind without fee," conditioned on citing the GDELT Project and linking to
  gdeltproject.org when redistributing.
- **licence_upstream_source**: https://www.gdeltproject.org/about.html — fetched 2026-09-03
- **verdict**: ATTRIBUTION (functionally very close to PERMISSIVE_OK — the only condition is
  a citation/link, not a share-alike or field-of-use restriction)
- **grant_scope**: whole_corpus (GDELT's own structured output); GDELT itself derives from
  scraped global news, so **provenance_red_flag**: "underlying source articles are
  copyrighted news text GDELT does not redistribute in full — GDELT's own grant covers its
  derived event/tone records, not full-text reproduction of the news."
- **why**: large-scale, permissively licensed structured event data — plausible fit for
  reasoning-over-events tasks, unusual source class (news-derived structured records) that
  diversifies away from academic QA sets.
- **caveat**: not source text — it's derived features (actors, event codes, tone scores);
  needs a translation layer to become model-trainable text/latent pairs.

## 13. Software Heritage

- **source**: https://www.softwareheritage.org/legal/bulk-access-terms-of-use/ and
  /legal/api-terms-of-use/
- **faculty fit**: `language_code` — potential answer to the CodeSearchNet
  single-source/93.9%-truncation problem, IF used correctly
- **licence_upstream (VERIFIED, via search of the two terms-of-use pages)**: Software
  Heritage is an **archive of source code, not a licence grantor** — "you are solely
  responsible for determining the license... that applies to any software component in the
  Archive, and you must abide by its terms." It offers automatically-derived license
  hints but explicitly disclaims their correctness. Bulk/API access itself has its own usage
  terms (rate limits, no wholesale extraction without contacting them) separate from the
  per-file software licences.
- **licence_upstream_source**:
  https://www.softwareheritage.org/legal/bulk-access-terms-of-use/,
  https://www.softwareheritage.org/legal/api-terms-of-use/ — fetched 2026-09-03 (via search
  summary, not direct WebFetch of full text — flag for re-fetch before any ingest decision)
- **verdict**: **unstated at the aggregate level** — this is infrastructure, not a dataset;
  every individual repository pulled through it still needs its own per-repo licence
  determination (same shape problem as GitHub scraping generally)
- **why flagged rather than admitted**: the operator's `language_code` fix needs
  multi-language, licence-clean code. Software Heritage is the right *access layer* to do
  that (broadest coverage, includes repos GitHub itself has lost), but it is NOT itself a
  licence — a factory pass using it must filter to permissively-licensed repos (SPDX
  detection) per file, which is a substantial enrichment/filtering project, not a
  drop-in dataset.
- **caveat**: bulk access requires contacting Software Heritage directly per their terms —
  not a simple API pull at scale.

## 14. C4 (AllenAI Common Crawl derivative)

- **source**: Common Crawl, republished by AllenAI (also on HF, but the Common-Crawl-derived
  nature makes this a Common-Crawl-derivative category entry)
- **faculty fit**: `language` trunk
- **licence_upstream (VERIFIED via search of AllenAI's own README statement)**: "AllenAI is
  releasing this dataset under the terms of ODC-BY. By using this, you are also bound by the
  Common Crawl terms of use in respect of the content contained in the dataset."
- **licence_upstream_source**: allenai/c4 README (HF-hosted mirror of AllenAI's own
  statement) — search-verified 2026-09-03, **recommend a direct WebFetch of Common Crawl's
  own ToU page before ingest**, since C4's grant is explicitly conditioned on it
- **verdict**: ATTRIBUTION-class (ODC-BY requires attribution), but layered with Common
  Crawl's own terms of use which are usage restrictions, not a copyright licence — worth a
  legal flag distinct from the ODC-BY tag alone
- **grant_scope**: whole_corpus per AllenAI's own grant, conditioned on downstream
  Common-Crawl ToU compliance
- **provenance_group**: Common Crawl (single group covering C4, CCNet, OSCAR, and any other
  CC-derived corpus — same "one lineage, one group" logic as LibriVox)
- **why**: canonical large-scale open web text; already well-understood risk profile in the
  field.
- **caveat**: Common Crawl itself is a scrape of copyrighted third-party web pages — the
  ODC-BY licence covers AllenAI's *compilation*, not necessarily every underlying page's
  own copyright status. This is the standard "is web-scale pretraining fair use" question,
  outside this survey's scope to resolve, but should be recorded as a
  **provenance_red_flag**, not silently treated as clean.

## 15. arXiv metadata (Kaggle mirror, Cornell University)

- **source**: https://www.kaggle.com/datasets/Cornell-University/arxiv
- **faculty fit**: `language_code` (abstracts as docstring-adjacent prose),
  `reasoning`/`numeric` (STEM abstracts), `language` trunk
- **licence_upstream (search-verified, WebFetch of the Kaggle page itself did not render
  license text — re-fetch or use `kagglehub`/API metadata before ingest)**: CC0 1.0 for the
  **metadata** (titles, authors, categories, abstracts); individual paper full texts/PDFs
  are governed by whatever licence the author chose at submission (see
  arxiv.org/help/license), which varies per paper (arXiv non-exclusive licence, CC BY, CC
  BY-SA, CC BY-NC-SA, or none).
- **licence_upstream_source**: arxiv.org/help/license (the authoritative page for the
  full-text layer) — not yet fetched this session, flagged for follow-up; Kaggle page
  itself asserts CC0 for the metadata dump specifically.
- **verdict**: PERMISSIVE_OK for the **metadata fields only** (`grant_scope: metadata_only`
  if full text/PDFs are pulled — do not conflate the two layers)
- **why**: large (2.7M+), clean CC0 metadata layer is directly usable for abstract-level
  text without touching the mixed-licence full-text layer at all.
- **caveat**: do not scale beyond metadata (titles/abstracts/categories) without a
  per-paper licence pull for full text — same shape as PMC below.

## 16. PubMed Central Open Access Subset

- **source**: https://pmc.ncbi.nlm.nih.gov/tools/openftlist/
- **faculty fit**: `language` trunk (scientific prose), `reasoning`
- **licence_upstream (VERIFIED)**: "Within the PMC Open Access Subset articles are available
  for reuse, but license terms vary. Please refer to the license statement in [each] article
  for specific terms of use" — described generally as "Creative Commons or similar licenses
  that allow more liberal redistribution and reuse than a traditionally copyrighted work,"
  but **not a single uniform licence**.
- **licence_upstream_source**: https://pmc.ncbi.nlm.nih.gov/tools/openftlist/ — fetched
  2026-09-03
- **verdict**: **unstated at the aggregate level** — genuinely a mixed bag (CC0, CC BY, CC
  BY-SA, CC BY-NC, CC BY-NC-SA all appear); a real ingest needs per-article license field
  extraction and bucketing, not a blanket admit
- **why flagged, not admitted outright**: real scientific-prose volume, low provenance risk
  (NIH-run, well-documented), but the "commercial-use-OK" subset specifically (CC0/CC BY
  only) would need to be filtered out from the NC/ND-bearing remainder before use — doable,
  PMC's own metadata carries the license field per article.
- **enrichment plan**: filter to license field ∈ {CC0, CC-BY} at ingest for a PERMISSIVE_OK/
  ATTRIBUTION-only pull; a second, NC-tagged pull is possible under the same DEC-31 logic
  already applied to GooAQ.

## 17. CourtListener bulk data (Free Law Project)

- **source**: https://www.courtlistener.com/help/api/bulk-data/
- **faculty fit**: would-be `reasoning`/`language` trunk (legal reasoning, formal
  argumentative structure) — **REFUSED, recorded to prevent re-discovery wasting a future
  pass**
- **licence_upstream (VERIFIED via search of Free Law Project's own statement)**: "Content
  is licensed under a Creative Commons BY-ND international 4.0 license, except where
  indicated."
- **licence_upstream_source**: courtlistener.com/help/api/bulk-data — search-verified
  2026-09-03, recommend direct WebFetch confirmation before final close-out
- **verdict**: **BLOCKING** — ND (No-Derivatives) is on the operator's explicit refuse list;
  a trained model's weights are definitionally a derivative work of anything they're trained
  on, which is exactly what ND forbids.
- **why refused**: raw court opinions themselves are separately public domain (US
  government/judicial works), so a future pass **could** pull opinions directly from court
  PACER/state repositories or from a public-domain-only re-derivation rather than through
  CourtListener's ND-licensed compilation — noting this distinction so a later surveyor
  doesn't have to re-derive it.

## 18. DTD — Describable Textures Dataset (Oxford VGG)

- **source**: https://www.robots.ox.ac.uk/~vgg/data/dtd/
- **faculty fit**: `visual`
- **licence_upstream**: **not resolved this session** — HF's mirror tags it "other" (i.e.
  explicitly not a standard licence), and the primary Oxford VGG page was not fetched (out
  of session budget). Per the mirror-lies-precedent, "other" on HF means nothing until the
  primary page is read.
- **verdict**: **UNVERIFIED — do not admit.** Flagged only as a next-pass candidate; the
  "other" tag is itself the reason to check, not a green light.

## 19. OpenML

- **source**: https://www.openml.org/terms and https://docs.openml.org/terms/
- **faculty fit**: `numeric/math` PLACEHOLDER (mostly tabular data, closer fit than
  text-first sources)
- **licence_upstream (VERIFIED)**: OpenML is a **platform**, not a single dataset — each
  uploader chooses a licence per dataset (CC0 is a common option but not universal); the
  platform's own code is BSD-3-Clause; users are granted "a non-exclusive license to access
  and use content... in accordance with any licences granted by the submitter."
- **licence_upstream_source**: https://www.openml.org/terms — fetched 2026-09-03
- **verdict**: **unstated at the aggregate level** — same shape as PMC/EU portal/data.gov:
  a source class requiring per-dataset licence capture, not a blanket admit
- **why**: worth surveying specifically for the `numeric/math` placeholder once it moves
  past "blocked on corpus" — tabular ML benchmark data with numeric targets is closer in
  kind to what that faculty needs than text-derived math problems are.

## 20. Papers with Code dataset index

- **source**: https://paperswithcode.com/datasets, licensing guide at
  https://stat.paperswithcode.com/datasets/license
- **role**: **index only, not a dataset source** — its own site content is CC-BY-SA, but
  that licence covers PWC's descriptions/leaderboards, not the datasets it links to. Its own
  licensing guide explicitly disclaims warranting the accuracy of the licence info it
  displays for indexed datasets and flags the two shapes that recur across this whole
  survey: "only annotations are free, images/full corpus may be proprietary" and "only the
  scraping code is open, not the resulting data."
- **why recorded**: useful as a *discovery* tool for a future survey pass (it aggregates
  benchmark datasets by task), but every hit through it still needs the same primary-source
  licence read this survey did — record as methodology, not as an admissible source.

## 21. GDELT-adjacent / general note — Wikimedia Commons category dumps

- **source**: commons.wikimedia.org categories, via Wikimedia dumps
- **faculty fit**: `visual` (potential third visual source, image content)
- **licence_upstream**: per-file — Commons requires every upload to declare a free licence
  (public domain, CC0, CC BY, CC BY-SA) at upload time, but **enforcement is
  upload-time self-declaration**, not verified copyright clearance; known to contain
  mis-licensed uploads that get removed on discovery.
- **verdict**: **PERMISSIVE_OK/ATTRIBUTION by declared licence, but with a structural
  provenance_red_flag**: "self-declared licence at upload, not independently verified;
  known history of mis-licensed content requiring post-hoc removal" — needs an automated
  per-file licence-field filter (Commons' own API exposes it) rather than a blanket corpus
  pull.
- **why not fully admitted**: same caution the operator's stance already applies to
  Common Voice-shaped consent-revocable material — this is a different risk (declared
  licence, not consent) but similar in requiring per-item machine-checkable filtering before
  training, not corpus-level trust.

---

## Cross-cutting observations

- **The "annotations-clean, images/full-text-unstated" shape recurs constantly** (COCO,
  Visual Genome, arXiv, PMC) — this is exactly what DEC-57's `grant_scope` field exists to
  catch, and it should be the default assumption for any multimodal or full-text corpus
  fetched through this survey, not an exception.
- **Government/institutional open-data portals (EU, US data.gov, GDELT, PMC) are reliable
  low-risk hunting grounds but are aggregators, not datasets** — each needs its own
  per-resource pull and licence capture; they should be recorded as **sources**, not
  admitted wholesale.
- **One clean REFUSE found**: CourtListener's CC BY-ND 4.0 bulk data — a textbook ND case,
  useful precisely because it demonstrates the refuse rule firing correctly on a real,
  otherwise-attractive candidate (large, high-quality, well-curated legal corpus).
- **Stack Exchange Data Dump is the strongest single lever for `memory`'s NC/concentration
  problem** — real retrieval-shaped Q&A structure, SHARE_ALIKE (not NC), at a scale that
  could meaningfully dilute GooAQ's 79.1% share. Its unusual "live hyperlink attribution"
  clause is a genuine open question, not a blocker by itself, and should go to the same kind
  of flagged-not-resolved treatment the SNLI/all-nli "is a trained model adapted material"
  question already got.
- **`numeric/math` (PLACEHOLDER) has the cleanest available path of any starved faculty**:
  DeepMind Mathematics Dataset (Apache-2.0) + MATH (MIT) + Wikidata's CC0 quantity
  statements gives three genuinely different-in-kind, all-PERMISSIVE_OK sources without
  touching `reasoning`'s spent gsm8k/aqua_rat — this looks like the fastest faculty to
  unblock from PLACEHOLDER of anything surveyed.
- **`visual` remains the hardest** even after this survey: EuroSAT is clean but narrow
  (single domain), Visual Genome and COCO both carry the annotations/images split, and DTD
  is unresolved. A real fix likely needs EuroSAT (clean, narrow) + Visual Genome's
  *annotation layer only* (treating the image layer as unusable until per-image cleared) +
  a dedicated per-image Commons-category pull — none of which alone clears B2's N_eff ≥ 3
  without stacking at least three sources.

---

## Top 8 (returned to caller)

1. **EuroSAT** (Zenodo, MIT) — `visual`, most-starved faculty, clean whole-corpus grant.
2. **Visual Genome** (Stanford, CC BY 4.0 for annotations) — `visual`, second clean-ish
   source, annotation layer only.
3. **Wikidata** (Wikimedia, CC0 structured / CC BY-SA 4.0 text) — `memory` / `numeric` /
   `reasoning`, breadth and permissiveness.
4. **Stack Exchange Data Dump** (Internet Archive / Stack Exchange Inc., CC BY-SA 4.0) —
   `memory`, best lever against GooAQ's NC concentration.
5. **DeepMind Mathematics Dataset** (GitHub, Apache-2.0) — `numeric/math`, unblocks the
   PLACEHOLDER without touching `reasoning`'s spent sources.
6. **MATH** (Hendrycks, GitHub, MIT) — `numeric/math` / `reasoning`, cleanest licence text
   fetched verbatim this session, pairs with #5 for N_eff.
7. **Project Gutenberg** (public domain, US) — `language` trunk foundation corpus.
8. **data.gov / EU Open Data Portal** (public domain / CC BY 4.0 by class) — `language`
   trunk register diversity; recorded as a source class needing per-resource capture, not a
   single admissible dataset.

REFUSED and worth remembering: **CourtListener bulk data** (CC BY-ND 4.0 — ND is on the
refuse list). UNVERIFIED, do not admit yet: **DTD** ("other" tag on the mirror, primary page
not read this session).
