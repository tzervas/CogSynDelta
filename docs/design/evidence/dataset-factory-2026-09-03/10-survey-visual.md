# Visual faculty survey — image-caption / image-QA candidates

Surveyor pass, 2026-09-03. Ground truth read first: `00-ground.md`. Scope per the task: the
`visual` faculty is currently **BLOCKING** (tiny-imagenet built on unlicensed ImageNet;
flickr30k also BLOCKING). This survey targets **image-text pairing** — captioning and
visual-QA datasets — which is a different shape of corpus from the pure-image classification/
pretraining composite already surveyed in `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`
("Replacement vision corpora" section, ~25 datasets, VERIFIED at commit `18f0176` of
`CogSynDelta`). That composite (pxhere, PatchCamelyon, Shapes3D, CLEVR, Fashion-MNIST,
EuroSAT-rgb, Quick Draw, Caltech-101/256 — all `PERMISSIVE_OK`/`ATTRIBUTION`, ≈497k images) is
**not re-derived here**; it is cited as background and is still the strongest available I-JEPA
pretrain source. This survey's job is the caption/QA layer on top: 20 new candidates, each
with per-image rights considered separately from annotation rights, since that split is
exactly where this class of dataset breaks.

**No dataset content was downloaded.** Everything below is metadata: HF API responses, dataset
cards, and licence/terms pages fetched live this session. VERIFIED = licence text read this
session from a primary or HF-card source (quoted). INFERRED = this document's synthesis,
corroborated but not independently re-fetched from the true primary source. Two candidates
(COCO's own terms-of-use page, ChartQA's underlying-image-source list) could not be rendered
via WebFetch after repeated attempts — those gaps are flagged explicitly rather than papered
over.

---

## The recurring trap this faculty keeps hitting

Every image-captioning dataset built on **scraped web/social photography** inherits the same
defect already found for `tiny-imagenet` (ImageNet), `TreeOfLife-200M`, `red_caps`, `STL-10`,
`Food-101`, and `Open Images` in the prior audit: **the dataset's own licence tag covers the
annotation/caption text, never the underlying photograph**, and the photographs come from many
individual rightsholders under mixed, unverifiable-at-scale terms. Google's own Open Images
page states this candidly: *"we make no representations or warranties regarding the license
status of each image and you should verify the license for each image yourself."* COCO,
Visual Genome, TextVQA, GQA, NoCaps, OK-VQA/A-OKVQA, VQAv2, SBU Captions, and Localized
Narratives are **all** downstream of this same per-photographer problem — they either reuse
COCO/Flickr30k images directly or reuse Open Images/YFCC100M images directly. A `cc-by-4.0` tag
on the *annotations* does not make the *pixels* clean. This is the single fact that decides
most of the verdicts below, so it is stated once here rather than re-derived nine times.

The second recurring trap: **Conceptual Captions and CC12M are URL lists, not images** —
`grant_scope: metadata_only`, structurally identical to LAION (which the task names explicitly
as the negative example). Google's licence text ("freely used for any purpose... Google
disclaims all liability") covers the URL/caption CSV Google itself compiled — it says nothing
about, and cannot grant, rights over the images living at those URLs, most of which are
third-party web photography with no clean chain at all.

---

## Catalogue entries

Shape per `00-ground.md` §(b). `mirror_url` is the top HF hit by downloads unless noted.

### 1. `HuggingFaceM4/COCO` (and equivalent COCO-caption mirrors: `jxie/coco_captions`, `Multimodal-Fatima/COCO_captions_*`)
- **upstream URL:** cocodataset.org (Microsoft/COCO Consortium)
- **mirror URL:** `huggingface.co/datasets/HuggingFaceM4/COCO`
- **mirror_tag:** `cc-by-4.0`
- **licence_upstream:** **UNVERIFIED this session** — `cocodataset.org/#termsofuse` could not
  be rendered by WebFetch after three attempts (returns only the nav shell; JS-gated content).
  Well-documented public record (widely quoted, not independently re-fetched here) states the
  page distinguishes: *"annotations belong to the COCO Consortium and are licensed under a
  Creative Commons Attribution 4.0 License"* while *"use of the images must abide by the Flickr
  Terms of Use"* — i.e. the CC BY 4.0 grant is annotation-only, images are individually-rights
  Flickr photographs. **Flag this INFERRED, not VERIFIED**, and re-fetch before any admission
  decision — do not accept the `cc-by-4.0` mirror tag as the pixel licence.
- **grant_scope:** `metadata_only` (annotations) if the widely-quoted text is accurate; images
  are third-party
- **verdict:** **BLOCKING** (per-photographer trap, same shape as Open Images/red_caps, both
  already ruled out for this project) — pending a working re-fetch of the primary terms page
- **provenance_group:** `coco-image-lineage` — shared with VQAv2, OK-VQA, A-OKVQA, GQA (partial,
  via Visual Genome), Localized Narratives (COCO subset), NoCaps (uses COCO val for some splits)
- **size:** 123,287 images / ≈616k captions (2017 split)
- **quality signal:** human-written captions (5/image, MTurk), high curation, heavily used as
  an eval benchmark across the field — **contamination risk is real** if used for training:
  COCO captions/images appear inside most public VLM pretraining sets already
- **feeds:** would be the single highest-value image-caption source by curation quality, if
  clean
- **enrichment:** none needed beyond format conversion; licence consequence of the BLOCKING
  verdict is that this poisons any composite it's mixed into (strictest-input rule)
- **why refused:** image rights unverified/likely non-uniform; treat as the tiny-imagenet
  pattern until COCO's terms page is actually re-fetched successfully

### 2. `google-research-datasets/conceptual_captions` (CC3M) / `laion/conceptual-captions-12m-webdataset` (CC12M)
- **upstream URL:** `github.com/google-research-datasets/conceptual-captions`,
  `ai.google.com/research/ConceptualCaptions`
- **mirror URL:** `huggingface.co/datasets/google-research-datasets/conceptual_captions`
- **mirror_tag:** `license:other`
- **licence_upstream:** **VERIFIED**, fetched `raw.githubusercontent.com/.../LICENSE` this
  session, quoted verbatim: *"The dataset may be freely used for any purpose, although
  acknowledgement of Google LLC ('Google') as the data source would be appreciated. The
  dataset is provided 'AS IS' without any warranty..."*
- **grant_scope:** `metadata_only` — this is a URL+caption list, not pixels (LAION-shape). The
  Google grant covers the CSV Google compiled; it is silent on, and cannot grant, rights over
  the linked images (harvested from arbitrary web pages with alt-text)
- **verdict:** **REFUSE (LAION-class)** — structurally the exact negative example named in the
  task brief. Do not admit despite the clean-looking Google licence text; the licence covers
  metadata, not content, and most links have since rotted or point to unverifiable sources
- **provenance_group:** `laion-style-url-list` (shared shape with CC12M, WIT's some subsets,
  RedCaps)
- **size:** CC3M ≈3.3M pairs, CC12M ≈12M pairs (URL rows, not images)
- **quality signal:** captions are Alt-text-derived, hypernymed/normalized by Google's own
  pipeline — reasonably clean text, irrelevant given the image-rights block
- **feeds/enrichment:** N/A — refused at the grant_scope gate before any enrichment question

### 3. `Fhrozen/sbucaptions` (SBU Captions)
- **upstream URL:** `www.cs.virginia.edu/~vicente/sbucaptions/` (Ordonez, Kulkarni, Berg 2011)
- **licence_upstream:** **UNVERIFIED this session** — page not fetched; SBU Captions is a
  scrape of ~1M Flickr photos with user-written captions, same per-photographer Flickr shape
  as COCO/Flickr30k. No known uniform licence grant.
- **grant_scope:** unstated
- **verdict:** **BLOCKING (unresolved)** — same class as GTSRB/EMNIST in the prior audit: no
  formal grant located, "free to use for research" is the historical framing for this vintage
  of scrape
- **provenance_group:** `flickr-scrape-lineage` (independent of COCO's Flickr subset, but same
  underlying rights problem)
- **size:** ≈1M image-caption pairs
- **quality signal:** noisy (single caption/image, user-written, pre-2011 vintage, known to be
  weaker than COCO's curated captions)
- **why refused:** no grant found; low priority to chase given COCO is already blocking on the
  identical defect at higher quality

### 4. Visual Genome (`ranjaykrishna/visual_genome`, `jn12/VisualGenome`)
- **upstream URL:** `visualgenome.org`
- **mirror_tag:** `cc-by-4.0` (on some re-uploads)
- **licence_upstream:** **INFERRED** (not independently re-fetched this session) — Visual
  Genome's own site states annotations are CC BY 4.0; images are **the COCO + YFCC100M
  image pools** — inherits both the COCO per-photographer problem above and YFCC100M's own
  (Flickr, mixed CC licences, never uniformly cleared for this kind of reuse)
- **grant_scope:** `metadata_only` (region descriptions, relationships, QA pairs); images
  third-party
- **verdict:** **BLOCKING** — same shape, higher information density (relationships/attributes,
  not just captions) makes this a bigger loss than COCO alone
- **provenance_group:** `coco-image-lineage` (partial overlap) + `yfcc100m-lineage`
- **size:** 108,077 images, 5.4M region descriptions, 1.7M QA pairs
- **quality signal:** very high (dense scene-graph annotation, widely used foundation for
  GQA/A-OKVQA), the annotation *text* is exactly the kind of structured relational data the
  `visual` faculty's cross-attention read-out would want — wasted on unclean images

### 5. `HuggingFaceM4/VQAv2` and `HuggingFaceM4/A-OKVQA` / `lmms-lab-encoder/OK-VQA`
- **upstream URL:** `visualqa.org` (VQAv2), `okvqa.allenai.org` (OK-VQA), `allenai.org/data/a-okvqa`
- **licence_upstream:** VQAv2/OK-VQA/A-OKVQA all use **COCO images** as their image pool
  (questions/answers are new annotations layered on COCO). Inherits COCO's per-photographer
  problem in full — this was not independently re-verified per-dataset since the shared
  dependency was already established for entry 1.
- **grant_scope:** `metadata_only` (Q&A pairs); images = COCO
- **verdict:** **BLOCKING** — downstream of entry 1
- **provenance_group:** `coco-image-lineage`
- **size:** VQAv2 ≈1.1M QA pairs / 204,721 images; A-OKVQA ≈25k QA pairs (harder, requires
  outside knowledge — higher per-item quality, same image problem)
- **quality signal:** VQAv2 QA is crowd-sourced, well-balanced by design (answer-distribution
  balancing was VQAv2's whole contribution over VQAv1); A-OKVQA requires external commonsense
  knowledge, closer to genuine reasoning-over-vision than template QA

### 6. `lmms-lab-encoder/GQA`
- **upstream URL:** `cs.stanford.edu/people/dorarad/gqa/`
- **licence_upstream:** **UNVERIFIED this session.** GQA's questions are programmatically
  generated from Visual Genome scene graphs; images are the Visual Genome/COCO image pool.
- **grant_scope:** `metadata_only`; images = Visual Genome (entry 4)
- **verdict:** **BLOCKING** — downstream of entry 4
- **provenance_group:** `coco-image-lineage`
- **size:** 22M questions over 113,018 images
- **quality signal:** questions are template/graph-generated (structurally regular, less
  linguistically diverse than human-written QA — a real quality caveat independent of licence)

### 7. `lmms-lab-encoder/textvqa` (TextVQA) and `lmms-lab-encoder/NoCaps`
- **upstream URL:** `textvqa.org`, `nocaps.org`
- **licence_upstream:** **UNVERIFIED this session.** Both are built on **Open Images**
  photographs. Open Images' own page (already quoted in the prior audit, re-confirmed here by
  citation) explicitly declines to assert a uniform image licence: *"we make no representations
  or warranties regarding the license status of each image."*
- **grant_scope:** `metadata_only`; images = Open Images
- **verdict:** **BLOCKING** — same defect already on record for Open Images itself
- **provenance_group:** `open-images-lineage`
- **size:** TextVQA 28,408 images / 45,336 questions; NoCaps 15,100 images / 166,100 captions
  (val+test, held out from training by design — it's meant as an eval-only zero-shot probe)
- **quality signal:** TextVQA specifically requires OCR-in-the-wild reasoning (reads text in
  the scene) — exactly the kind of thing a diagrams/charts-capable `visual` faculty would want;
  the licence defect is the only blocker, not quality

### 8. `HuggingFaceM4/LocalizedNarratives`
- **upstream URL:** `google.github.io/localized-narratives/`
- **licence_upstream:** **UNVERIFIED this session.** Built as an overlay (synchronized voice +
  mouse-trace narration, transcribed to text) on **COCO, ADE20K, Open Images, and Flickr30k**
  images — a union of every already-blocking image pool in this survey.
- **grant_scope:** `metadata_only`; images = union of entries 1/7/Flickr30k
- **verdict:** **BLOCKING**
- **provenance_group:** `coco-image-lineage` + `open-images-lineage` + `flickr30k-lineage`
- **size:** ≈849k narratives across the four source sets
- **quality signal:** unusually rich grounding signal (word-level timing + mouse trace →
  region correspondence) — the richest annotation type in this survey, entirely wasted on
  unclean image pools; worth remembering as a *format* to imitate on a clean corpus later

### 9. `nlphuji/flickr30k`
- Already catalogued in `00-ground.md` as **BLOCKING** (P3.2 planned addition). Re-confirmed
  here rather than re-derived: `flickr30k` is a direct Flickr photo scrape, same
  per-photographer shape, no uniform grant. Listed for completeness of the provenance-group
  map (entry 8 depends on it).
- **provenance_group:** `flickr30k-lineage`

### 10. `wikimedia/wit_base` (WIT — Wikipedia-based Image Text)
- **upstream URL:** `github.com/google-research-datasets/wit`
- **mirror URL:** `huggingface.co/datasets/wikimedia/wit_base`
- **mirror_tag / card-stated licence:** **VERIFIED** this session via HF card fetch:
  **CC BY-SA 4.0** stated at the dataset level.
- **licence_upstream:** **gap identified, not resolved.** The card asserts CC BY-SA 4.0 for the
  whole dataset but — as fetched this session — **does not address per-image licensing**, and
  Wikimedia Commons images are individually tagged with a **mix** of CC0, CC BY, CC BY-SA, and
  (for some historical/PD material) no-licence-required public domain. A dataset-level tag
  cannot be trusted to be the per-image floor without checking Commons' own per-file licence,
  which this session did not do at scale.
- **grant_scope:** `unstated` pending that check — this is a genuinely different case from
  COCO/Open Images: Wikimedia Commons is **itself** a licensed-image repository (every upload
  requires a real licence, unlike an arbitrary web scrape), so the underlying pool is much more
  likely to resolve to CC0/CC BY/CC BY-SA than COCO's Flickr pool is. The floor is probably
  **SHARE_ALIKE at worst** rather than BLOCKING, but this needs a per-file audit before
  admission, not an assumption.
- **verdict:** **UNVERIFIED** (leaning SHARE_ALIKE, not BLOCKING) — the most promising
  candidate in this survey precisely because Commons enforces licensing at upload time; flag
  for a follow-up per-file sampling pass before treating as clean
- **provenance_group:** `wikimedia-commons-lineage`
- **size:** ≈37.6M image-text pairs across 108 languages (English subset alone ≈5.5M)
- **quality signal:** captions/descriptions are real Wikipedia article context (alt text, page
  title, reference description, attribution) — structured and multi-field, genuinely different
  quality profile from scrape captions
- **feeds:** best available caption-scale candidate if the per-file audit clears it; would need
  language filtering (English or the fleet's target set) and a licence-stratification pass
  (keep CC0/CC BY, quarantine or drop CC BY-SA/NC if the region wants to stay out of the NC
  tier `memory` is already in)
- **enrichment:** filter to public-domain + CC BY + CC0 files only for a non-SA subset; SA
  files usable under the already-NC composed-model tier without new cost, per the strictest-
  input rule already in effect for `memory`

### 11. `docvqa` (RRC/DocVQA, task 1) — `lmms-lab-encoder/DocVQA`
- **upstream URL:** `www.docvqa.org`, source images from `industrydocuments.ucsf.edu`
- **licence_upstream:** **VERIFIED (partial)** this session — the docvqa.org page states
  registration/login is required and terms live on the download portal, not the public page
  (could not be read further without an account, which is out of scope for a metadata-only
  survey). What was independently established: the **document images themselves are pages
  from the UCSF Industry Documents Library**, a public archive of documents produced under
  U.S. litigation discovery (principally tobacco-industry document disclosures), which UCSF
  makes available for public research use — a materially different rights posture from a
  Flickr/web photo scrape (these are litigation-disclosure records, not creative photography,
  and UCSF's own library exists specifically to host and serve them publicly).
- **grant_scope:** `unstated` pending reading the actual RRC portal terms (gated behind login)
- **verdict:** **UNVERIFIED** — promising (industry-documents rights posture is unusually
  favorable) but the operative RRC licence text was not read this session; do not admit
  without reading it
- **provenance_group:** `ucsf-industry-documents-lineage`
- **size:** 12,767 images / 50,000 questions (task 1)
- **quality signal:** real scanned business documents (forms, letters, tables) — exactly the
  "diagrams/charts" document-understanding shape the task asked to survey; human-annotated
  QA pairs

### 12. `HuggingFaceM4/ChartQA` / `ahmed-masry/ChartQA`
- **upstream URL:** `github.com/vis-nlp/ChartQA`
- **mirror_tag:** `gpl-3.0`
- **licence_upstream:** **VERIFIED** this session — fetched the repo's actual `LICENSE` file
  content: full **GPLv3** text (software copyleft, not a data licence per se — an odd choice
  for a dataset repo, but it is what the maintainers filed).
- **grant_scope:** `unstated` for the underlying chart images specifically — the WebFetch this
  session found one direct textual clue: *"The Pew Research Centre chart images didn't have
  any SVG files when we crawled them"* — confirming charts were **crawled from Pew Research
  and other statistics sites (Statista is the commonly cited second source in ChartQA's paper,
  not independently re-confirmed here)**, not authored by the ChartQA team. GPLv3 on the repo
  cannot grant rights the crawlers never held over Pew/Statista's own chart images.
- **verdict:** **BLOCKING (unresolved)** — same shape as GTSRB: a licence string exists but
  covers the wrong thing (crawler code and QA-pair generation code), while the chart images'
  actual source (Pew, Statista, and others) retains its own separate, uncleared rights
- **provenance_group:** `pew-statista-chart-crawl-lineage`
- **size:** 21,899 charts / 32,719 QA pairs
- **quality signal:** real published statistical charts, both human-written and template-
  generated QA — genuinely useful diagram/chart shape, blocked purely on image provenance

### 13. `vikhyatk/figureqa` (FigureQA, Microsoft Research)
- **upstream URL:** `github.com/Maluuba/FigureQA` (later `microsoft/FigureQA`)
- **licence_upstream:** **VERIFIED (partial)** — WebFetch this session confirmed FigureQA is
  **entirely synthetic**: charts are rendered programmatically via Bokeh from synthetic
  numeric data, not crawled from any real publisher. This sidesteps the per-photographer /
  per-publisher trap structurally, the same way CLEVR/Shapes3D do for photography. The
  repository's specific LICENSE file text was not retrieved (404/gated); Microsoft Research
  dataset releases of this vintage are typically MS-provided under a permissive research
  licence, but that is **INFERRED**, not read verbatim this session.
- **grant_scope:** `whole_corpus` if the inference above holds (synthetic generation → MS holds
  the rights outright, no third party involved)
- **verdict:** **UNVERIFIED, leaning PERMISSIVE_OK** — best-shaped candidate in the chart/QA
  class precisely because it's synthetic; needs the actual LICENSE file read before admission
- **provenance_group:** `synthetic-chart-render-lineage` (shares the render-not-scrape shape
  with CLEVR/Shapes3D/dSprites already admitted in the prior audit, and with CSD's own W3r
  composite-renderer path — same class of solution, different tool)
- **size:** ≈180,000 synthetic figures (100k train + val/test), ~1.3M QA pairs
- **quality signal:** template QA over synthetic charts (bar/line/pie/scatter), answer space is
  yes/no — limited linguistic diversity but zero provenance risk if the licence confirms clean

### 14. PlotQA (`Dodon/plotqa-dataset`, `achang/plot_qa`)
- **upstream URL:** `github.com/NiteshMethani/PlotQA` (repo access attempted, 404'd this
  session on the `vis-nlp` org mirror path tried — the correct org may differ; not resolved)
- **licence_upstream:** **NOT VERIFIED this session** (fetch failed). PlotQA's charts are
  synthetically rendered (same render-not-scrape shape as FigureQA) but from **real-world
  statistics sourced from the World Bank Open Data and similar government/NGO open-data
  portals** (per the paper's own description, not re-verified here) — meaning even though the
  chart *image* is synthetic, the underlying *data values* plotted may carry their own source
  licence (World Bank Open Data is itself CC BY 4.0, a favorable sign, not independently
  confirmed)
- **grant_scope:** `unstated`
- **verdict:** **UNVERIFIED** — needs the actual repo/LICENSE re-fetched from the correct URL
- **provenance_group:** `synthetic-chart-render-lineage` (shares underlying-data sourcing
  question with real-world statistics, distinct sub-group from FigureQA's pure-synthetic data)
- **size:** ≈224,000 charts / 28M QA pairs (by far the largest chart-QA set found)
- **quality signal:** large scale, real statistical distributions (not toy synthetic numbers)
  makes this higher-fidelity training signal than FigureQA if the licence clears

### 15. AI2D (`lmms-lab-encoder/ai2d`, Allen Institute for AI)
- **upstream URL:** `prior.allenai.org/projects/diagram-understanding` (formerly)
- **licence_upstream:** **NOT VERIFIED this session** — neither the AI2 project page nor the
  HF card exposed an explicit licence in what WebFetch returned. AI2 (Allen Institute) releases
  most of its research datasets under permissive terms (Apache-2.0/ODC-BY is typical for AI2
  corpora, e.g. ScienceQA components, C4, Dolma), but that is a **pattern-based INFERENCE**
  about the publisher, not a licence text read this session.
- **grant_scope:** `unstated`
- **verdict:** **UNVERIFIED** — diagrams are AI2's own commissioned/curated illustrative
  science diagrams (not a scrape of third-party photography), which is structurally favorable,
  but nothing here substitutes for reading the actual grant
- **provenance_group:** `ai2-diagram-lineage`
- **size:** 4,903 diagrams / ≈15,000 questions (multiple-choice)
- **quality signal:** hand-selected textbook-style science diagrams with dense diagram
  elements + relationships (arrows, labels) annotated separately — exactly the "diagrams"
  category the task named, high curation, small scale

### 16. `derek-thomas/ScienceQA`
- **upstream URL:** `scienceqa.github.io`
- **mirror_tag:** **VERIFIED** this session via HF card fetch: `cc-by-sa-4.0`
- **licence_upstream:** **NOT independently verified** — ScienceQA is a compilation across
  multiple source curricula (IXL Learning-licensed educational content is the commonly cited
  origin, not re-confirmed here); a compiled multi-source educational QA set has the same
  "declared licence may not match every constituent source" risk already seen with
  TreeOfLife-200M
- **grant_scope:** `unstated` for image sources specifically (only ~48% of ScienceQA items have
  an associated image/diagram at all — text-only reasoning is the majority of the set)
- **verdict:** **UNVERIFIED (mirror tag SHARE_ALIKE, upstream not confirmed)**
- **provenance_group:** `sciqa-curriculum-lineage`
- **size:** 21,208 questions, ≈10,332 with an image
- **quality signal:** genuinely reasoning-shaped (multi-hop science QA with lecture/explanation
  fields, not just answer extraction) — closest thing in this survey to the `reasoning`
  faculty's shape, useful either way if `visual` and `reasoning` ever share input format

### 17. Segment Anything / SA-1B (Meta)
- **upstream URL:** `ai.meta.com/datasets/segment-anything/`
- **licence_upstream:** **VERIFIED** this session, fetched live: *"Research purposes only"*
  stated as the intended use; images are described as *"licensed from a large photo company"*
  (not open-licensed originals — a commercial licence Meta itself holds, not one that transfers
  to third-party redistribution or training).
- **grant_scope:** `unstated`/restricted — "research purposes only" is a research-only term,
  which the operator's stance explicitly REFUSEs as a class (non-redistributable,
  training-forbidding-in-effect terms)
- **verdict:** **REFUSE-TERMS** — matches the operator's explicit refusal criteria
  ("research-only / non-redistributable terms... licences forbidding model training or weight
  release")
- **provenance_group:** `sa-1b-lineage` (self-contained; images licensed by Meta from an
  unnamed commercial photo provider, not shared lineage with anything else here)
- **size:** 11M images / 1.1B masks
- **quality signal:** very high (SAM's own training data, exhaustive automatic + human-refined
  masks) — irrelevant given the refusal
- **why refused:** explicit research-only term; no redistribution/training grant to third
  parties beyond research use

### 18. `SincereX/ChartBench`, `ChrisFan/ChartDQA`, `ahmed-masry/ChartQAPro` (second-generation chart-QA sets, MIT/Apache/CC-BY tagged mirrors)
- **mirror_tags observed:** `mit` (ChartBench), `apache-2.0` (ChartDQA), `mit` (ChartQAPro)
- **licence_upstream:** **NOT VERIFIED this session** — these are newer (2024-2025), smaller,
  often themselves *derived from* ChartQA/PlotQA/CharXiv (per `1fanj/Chartographer`'s explicit
  `source_datasets` field listing `HuggingFaceM4/ChartQA` and `princeton-nlp/CharXiv`) — i.e.
  they inherit ChartQA's own unresolved chart-image provenance (entry 12) even where the
  *derivative work's own* annotation licence is permissively tagged. A permissive tag on a
  derivative does not clear the base images' rights.
- **grant_scope:** `unstated`, and structurally can't be cleaner than its source
- **verdict:** **BLOCKING (inherited)** for any set built by re-annotating ChartQA/CharXiv chart
  images; **UNVERIFIED** for any that generate genuinely new chart images (none confirmed in
  this quick pass — would need per-repo checking before use)
- **provenance_group:** `pew-statista-chart-crawl-lineage` (via ChartQA) for the ones that
  reuse its images; separate groups for any that don't (not resolved here)
- **size:** ChartBench ≈100K-1M rows (band), ChartDQA/ChartQAPro smaller (1K-10K band)
- **why refused (as a class):** provenance inheritance not checked per-repo; flagged as a
  systematic risk in the fast-moving "ChartQA derivative" space rather than individually cleared

### 19. `mm-eval/InfographicVQA`, `Ryoo72/InfographicsVQA` (InfographicVQA / DocVQA task 3)
- **upstream URL:** `www.docvqa.org` (same RRC portal family as entry 11)
- **licence_upstream:** **NOT VERIFIED this session** — same gated-portal problem as DocVQA;
  infographics are scraped from the open web (per the paper's stated methodology, not
  re-verified here), which is a **worse** provenance shape than DocVQA's UCSF archive — these
  are commercially-designed marketing infographics from arbitrary publishers, not a public
  litigation-disclosure archive
- **grant_scope:** `unstated`
- **verdict:** **BLOCKING (unresolved)**, leaning worse than DocVQA specifically because the
  image *source* (web-scraped infographics vs. a public-interest document archive) is less
  favorable even though both sit behind the same RRC portal terms
- **provenance_group:** `rrc-docvqa-lineage` (shares the portal but not the image source with
  entry 11 — kept as a separate provenance group deliberately)
- **size:** 5,485 infographic images / 30,035 QA pairs
- **quality signal:** rich mixed text+chart+layout documents — exactly the "diagrams/charts"
  shape wanted, but the worst-verified provenance in the document-QA cluster

### 20. `HuggingFaceM4/A-OKVQA` — see entry 5 (kept as its own row per ground-truth shape, cross-referenced rather than duplicated in full)

### 21. Wikimedia Commons directly (not WIT — the raw Commons corpus, e.g. via `jinaai/wikimedia-commons-*_beir` collections or a fresh Commons API pull)
- **upstream URL:** `commons.wikimedia.org`
- **licence_upstream:** **VERIFIED by platform design**, not fetched as a single document —
  Commons requires every upload to carry an explicit, machine-readable licence (CC0, CC BY,
  CC BY-SA, or verified-PD), enforced at upload time and auditable per-file via the API
  (`imageinfo` + `extmetadata`). This is the **only** candidate in this survey where the
  per-image licence is structurally guaranteed to exist and be checkable, rather than assumed
  or absent.
- **grant_scope:** `whole_corpus`, but **heterogeneous per file** — a real ingestion pipeline
  must read each file's licence, not assume a corpus-wide tag
- **verdict:** **PERMISSIVE_OK / SHARE_ALIKE (mixed, resolvable per-file)** — this is not a
  ready-made caption dataset (no paired natural-language descriptions beyond what WIT already
  extracts, see entry 10), but as a **source of licence-clean images** to pair with a separately
  sourced captioning pipeline (model-generated captions over verified-clean images, an
  enrichment path the ground doc's "how it would be enriched" column exists for), this is the
  strongest single image *source* in the whole survey — better than any scrape.
- **provenance_group:** `wikimedia-commons-lineage` (same group as entry 10 — WIT is a
  processed view over this same corpus)
- **size:** Commons hosts >100M media files total; a filtered "photographic, CC0/CC-BY-only,
  no-NC" subset would need to be built, size TBD by the filter
- **enrichment path:** pull a CC0/CC-BY-filtered image subset via the Commons API (licence
  read per-file, recorded in the receipt), then generate captions with an already-licensed
  captioning enrichment step (e.g. a permissively-licensed VLM, not a scrape) — this converts
  a pure image source into a caption pair without inheriting any of the per-photographer risk
  every scraped set in this survey carries. This is the single most promising path found this
  session, not a ready-to-admit dataset.

### 22. LAION-2B / LAION-400M (named explicitly in the task brief as the negative example)
- **verdict:** **REFUSE** — restated for completeness of the catalogue, not surveyed fresh.
  URL-only, `metadata_only` grant scope, no image rights of any kind conveyed. Same class as
  entry 2 (Conceptual Captions/CC12M); listed separately because the task brief named it by
  name as the thing to contrast against actually-licensed corpora.
- **provenance_group:** `laion-style-url-list`

---

## Summary table — top 8 by verdict quality (best candidates first)

| # | dataset | verdict | why it ranks here |
|---|---|---|---|
| 21 | Wikimedia Commons (direct, filtered) | PERMISSIVE_OK/SHARE_ALIKE, per-file resolvable | only source with structurally-guaranteed per-image licensing; needs a build step (filter + caption enrichment), not a ready dataset |
| 10 | `wikimedia/wit_base` | UNVERIFIED, leaning SHARE_ALIKE | already-paired captions at 37.6M scale; same underlying image pool as #21, needs the per-file audit before trusting the dataset-level CC BY-SA tag as the pixel floor |
| 13 | FigureQA (Microsoft) | UNVERIFIED, leaning PERMISSIVE_OK | fully synthetic charts, sidesteps the per-photographer trap entirely; needs the actual LICENSE file read |
| 11 | DocVQA (task 1) | UNVERIFIED, leaning favorable | UCSF Industry Documents Library has an unusually clean public-archive rights posture vs. any web scrape; RRC portal terms not yet read |
| 14 | PlotQA | UNVERIFIED | largest chart-QA set found (224k charts/28M QA); synthetic render + likely-CC-BY-4.0 World Bank source data, repo fetch failed this session, needs retry |
| 15 | AI2D | UNVERIFIED, leaning favorable | AI2-authored diagrams (not a third-party scrape), AI2's typical release pattern is permissive, but no licence text actually read |
| 16 | ScienceQA | UNVERIFIED (mirror SA) | mirror tag CC BY-SA 4.0 confirmed; multi-source curriculum compilation risk not checked per-constituent |
| 12 | ChartQA | BLOCKING (unresolved) | GPLv3 tag covers the wrong layer (crawler/QA-gen code); Pew/Statista chart images' own rights never granted to the crawler — ranked here (not lower) only because its scale/quality is otherwise excellent and it is one repo-fetch away from a possible partial-source clearance (some contributing sources may be CC-licensed even if others aren't) |

**Everything else surveyed (COCO and its entire downstream tree — VQAv2, OK-VQA, A-OKVQA, GQA,
TextVQA, NoCaps, Localized Narratives, Visual Genome, SBU Captions, Flickr30k; Conceptual
Captions/CC12M/LAION; Segment Anything; InfographicVQA; the ChartQA-derivative cluster) is
**REFUSE or BLOCKING**, all for one of exactly three reasons: (a) the per-photographer scrape
trap already established for tiny-imagenet/Open Images/red_caps/TreeOfLife-200M, (b) explicit
research-only terms (SA-1B), or (c) URL-only metadata_only grant scope with no image rights at
all (Conceptual Captions/CC12M/LAION).**

## What was refused, and why (roll-up)

- **Every COCO-derived set** (COCO Captions, VQAv2, OK-VQA, A-OKVQA, GQA, NoCaps, Localized
  Narratives' COCO subset) — per-photographer Flickr image pool, no uniform grant, same defect
  already on record for the project.
- **Every Open-Images-derived set** (TextVQA, NoCaps' Open-Images-sourced portion, Localized
  Narratives' Open Images subset) — Google's own page disclaims a uniform image licence.
- **Visual Genome and GQA** — built on the COCO+YFCC100M image pool, same defect.
- **Flickr30k, SBU Captions, Localized Narratives (Flickr30k subset)** — direct Flickr scrapes,
  no uniform grant located.
- **Conceptual Captions, CC12M, LAION** — URL-list/metadata_only grant scope; the clean-looking
  licence text (Google's, in CC3M/CC12M's case) covers metadata the compiler holds rights to,
  not the linked images.
- **Segment Anything / SA-1B** — explicit "research purposes only" term, refused as a class per
  operator stance (non-redistributable, training-forbidding-in-effect).
- **InfographicVQA** — web-scraped commercial infographics behind a gated portal; worse
  provenance shape than DocVQA even before the portal terms are read.
- **The ChartQA-derivative cluster** (ChartBench, ChartDQA, ChartQAPro, Chartographer) —
  permissive licence tags on the *derivative annotation*, inherited unresolved image provenance
  from ChartQA itself.

## What remains genuinely open (needs a follow-up pass, not a verdict yet)

1. Re-fetch `cocodataset.org/#termsofuse` through a rendering path that survives the JS gate
   (or find the terms mirrored verbatim in a paper/secondary source this session didn't check)
   — this single fetch would either confirm or overturn entry 1's INFERRED status and, by
   extension, five downstream entries.
2. Read the actual RRC portal terms for DocVQA/InfographicVQA (requires an account — outside
   this session's scope, flagged for whoever holds one).
3. Locate and read FigureQA's and PlotQA's actual LICENSE files at their correct current repo
   locations (both fetch attempts this session hit dead links/404s — the org/path likely moved).
4. Read AI2D's and ScienceQA's actual licence grants rather than inferring from publisher
   pattern / mirror tag alone.
5. Run a real per-file sampling audit of Wikimedia Commons / WIT licence heterogeneity (entries
   10 and 21) before treating either as admitted — this is the highest-value follow-up in the
   whole survey, since it's the only path in this document that doesn't dead-end at BLOCKING.
