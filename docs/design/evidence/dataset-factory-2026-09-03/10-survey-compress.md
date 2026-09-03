# Survey — `compress` faculty (semantic compression / similarity)

Scope: paraphrase pairs, STS graded pairs, NLI, summarisation pairs, sentence-pair datasets
for `memory`'s compress half. Method per task: HF Hub public API + dataset cards, primary/
upstream source fetched directly (WebFetch), WebSearch for off-HF sets. No dataset content
downloaded — cards, licence pages, and API metadata only.

Fetch status legend: **VERIFIED** = licence text read this session at a primary (non-mirror)
source, quoted; **VERIFIED (mirror)** = licence tag read this session from the HF card itself
(still a primary read of the card, but the mirror is the host, not the rights holder — flagged
`grant_scope` accordingly); **INFERRED** = this document's synthesis, not a fetched quote;
**PRIOR-AUDIT VERIFIED** = fetched at primary source by this program's own earlier
`LICENCE-FOR-OPEN-WEIGHTS.md` audit (cited, not re-fetched this session, to avoid re-spending
budget on an already-settled primary-source read — the quote below is copied from that doc).

---

## Top 8 (ranked for near-term action)

| rank | id | verdict | why it's near the top |
|---|---|---|---|
| 1 | `hkust-nlp/SynCSE-scratch-NLI` | **PERMISSIVE_OK** | The one like-for-like replacement for the currently-trained all-nli pair shape that reaches clean, not just repaired |
| 2 | `google-research-datasets/paws` (+ `paws-x`) | **PERMISSIVE_OK** (soft attribution ask, not a licence condition) | Broad "freely used for any purpose" grant, VERIFIED this session at the GitHub LICENSE file; hard negatives by construction |
| 3 | `stanfordnlp/snli` (direct, not via all-nli) | **SHARE_ALIKE** | Already 78% of the repair path the internal audit recommends; re-deriving from SNLI+MultiNLI directly (not the flattened all-nli mirror) restores genre filtering |
| 4 | `nyu-mll/multi_nli` (genre-filtered: government + fiction only) | **SHARE_ALIKE / mixed by genre** | Same lineage group as SNLI; the *genre* column (destroyed in the all-nli mirror) is what makes filtering to defensible genres possible at all |
| 5 | `HuggingFaceM4/COCO` (captions, not the untagged ST mirror) | **ATTRIBUTION** | Two independent human captions of one image is a clean anchor/positive pair with a real CC BY 4.0 grant on the annotations |
| 6 | `ltg/en-wiki-paraphrased` | **ATTRIBUTION (unsettled)** | Apache-2.0 tag on the *paraphrase generation*, but `original` column is raw Wikipedia text — card is silent on Wikipedia's CC BY-SA; usable only if the `original` column is dropped or Wikipedia's own licence is honoured |
| 7 | `community-datasets/tapaco` | **ATTRIBUTION** | `cc-by-2.0` VERIFIED on the HF card; Tatoeba-lineage paraphrase pairs across many languages, small but genuinely clean provenance (Tatoeba itself is a long-running CC-licensed community corpus) |
| 8 | `mteb/sickr-sts` / SICK (any mirror) | **NC** | CC BY-NC-SA 3.0 VERIFIED at the primary SICK homepage; usable under the operator's NC-tolerant stance, but pushes `compress` further into the NC tier it's already in via `memory`'s GooAQ |

---

## Full candidate table

### NLI / entailment (compress's current objective shape)

**1. `stanfordnlp/snli`**
- Upstream: https://nlp.stanford.edu/projects/snli/ (Stanford NLP)
- Mirror: `stanfordnlp/snli` (HF, 60,957 downloads)
- Licence upstream: **CC BY-SA 4.0** — VERIFIED (PRIOR-AUDIT, this program's own LICENCE-FOR-OPEN-WEIGHTS.md §"upstream A"). That audit also flags the grant is "a Stanford assertion no Flickr30k source corroborates" — the underlying image captions trace to Flickr30k, whose own licence chain is unclear.
- Mirror tag: `cc-by-sa-4.0` (matches upstream claim)
- Verdict: **SHARE_ALIKE** (whether trained latents count as "adapted material" under SA is the project's open, unresolved question — flagged, not resolved, per the ground doc)
- Provenance group: **SNLI/MultiNLI/all-nli lineage** — same group as candidates 2, 3, and the currently-trained `sentence-transformers/all-nli`
- Size: 550,152 pairs (train+dev+test), currently 100% already inside all-nli per the internal audit ("staged snli is 100% already inside all-nli — it would add ZERO new pairs")
- Quality: human-annotated entailment/contradiction/neutral over Flickr30k captions; well-studied, some acknowledged annotation artifacts (hypothesis-only bias literature)
- Consume/emit fit: direct entailment-pair training source for `compress`'s current anchor/positive objective
- Enrichment: none needed structurally; the fix is *subtractive* — drop non-SNLI rows from the flattened mirror to restore attributability
- Licence consequence: SA propagates to any derived pair set that keeps SNLI rows

**2. `nyu-mll/multi_nli`**
- Upstream: https://cims.nyu.edu/~sbowman/multinli/ (NYU)
- Mirror: `nyu-mll/multi_nli` (HF, 26,492 downloads)
- Licence upstream: mixed **by genre** — VERIFIED (PRIOR-AUDIT): government genre "asserted public domain (unverified — no holder named)"; fiction genre "mixed CC BY-SA 3.0 / CC BY 3.0 / public domain"; 3 of 5 genres (telephone, letters, 9/11 report) have named commercial copyright holders and are the ones that make the flattened all-nli mirror BLOCKING today
- Mirror tag: none declared
- Verdict: **SHARE_ALIKE for government+fiction genres only**; **BLOCKING for the other 3 genres** (named commercial rights holders)
- Provenance group: SNLI/MultiNLI/all-nli lineage (same as #1)
- Size: 392,702 rows total; government+fiction subset ≈ 51,567 pairs
- Quality: standard NLI benchmark, well-cited; genre diversity is the actual value MultiNLI adds over SNLI
- Consume/emit fit: same entailment-pair shape as SNLI
- Enrichment: genre-filter at ingest (the `genre` column the all-nli mirror destroys); this filtering step IS the licence fix
- Licence consequence: filtered subset is SHARE_ALIKE; unfiltered is not usable

**3. `hkust-nlp/SynCSE-scratch-NLI`**
- Upstream/repo: https://github.com/hkust-nlp/SynCSE (same as HF repo — this one is not mirrored elsewhere)
- Mirror: `hkust-nlp/SynCSE-scratch-NLI`
- Licence: **MIT** — PRIOR-AUDIT VERIFIED, at both the dataset repo and GitHub
- Mirror tag: `mit`
- Verdict: **PERMISSIVE_OK**
- Provenance group: standalone (synthetic; GPT-4/GPT-3.5-generated from genre+topic prompts) — **no NLI corpus text carried through**, so it is NOT in the SNLI/MultiNLI group
- Size: 275,579 rows (`sent0`, `sent1`, `nli_hard`) — near-identical to the 277,269 pairs `compress` trains on today
- Quality: synthetic (model-generated), not human-written — a different kind of unknown (contamination/quality of synthetic generation), not an absence of one. No known eval contamination but unverified against `compress`'s own eval set
- Consume/emit fit: exact drop-in replacement shape for the current training regime (anchor/positive/hard-negative)
- Enrichment: none — ingest as-is
- Licence consequence: none — MIT is the weakest possible constraint. **This is the recommended near-term swap.**
- **Trap, explicitly avoid**: `hkust-nlp/SynCSE-**partial**-NLI` looks identical but its sentences come from SimCSE's SNLI+MNLI — carries the SA obligation through despite the MIT tag on the wrapper repo.

**4. `facebook/anli` (Adversarial NLI)**
- Upstream/repo: https://github.com/facebookresearch/anli
- Mirror: `facebook/anli` (HF, 98,645 downloads)
- Licence upstream: **CC BY-NC 4.0** — VERIFIED this session, HF card licence tag `cc-by-nc-4.0`, citing the repo's own LICENSE file
- Mirror tag: matches (`cc-by-nc-4.0`)
- Verdict: **NC**
- Provenance group: standalone lineage, BUT — per the internal audit's existing flag — its R3 split "re-imports OANC anyway," so treat R3 as inheriting whatever OANC's terms are (not independently checked this session)
- Size: ~162,865 pairs across R1/R2/R3
- Quality: adversarially collected (human-and-model-in-the-loop), harder than SNLI/MultiNLI by construction — useful as a hard-negative/robustness supplement, not a volume source
- Consume/emit fit: entailment-shaped; could supplement SynCSE-scratch for a harder tail
- Enrichment: none structural
- Licence consequence: NC — same tier `memory` is already in via GooAQ, so admitting this costs nothing marginal under the strictest-input rule already in force

### Paraphrase pairs

**5. `google-research-datasets/paws` and `paws-x`**
- Upstream/repo: https://github.com/google-research-datasets/paws
- Licence: VERIFIED this session, fetched the repo's `LICENSE` file directly (not the HF mirror):
  > "The dataset may be freely used for any purpose, although acknowledgement of Google LLC ("Google") as the data source would be appreciated. The dataset is provided "AS IS" without any warranty, express or implied. Google disclaims all liability for any damages, direct or indirect, resulting from the use of the dataset."
- Mirror tag (HF): `other` (correctly not auto-mapped to a standard SPDX id, since the grant text above is bespoke)
- Verdict: **PERMISSIVE_OK** — the grant is unconditional on redistribution/commercial use; attribution is requested, not required
- Provenance group: standalone (PAWS is built from Wikipedia + Quora sentence pairs via controlled word-swapping/back-translation — worth noting the underlying Wikipedia/Quora sentences are themselves separately licensed, but Google's grant covers the *dataset as released*, which is the operative fact for a downstream trainer)
- Size: PAWS ~108k labeled pairs (Wikipedia); PAWS-X adds 6 more languages via the same construction, ~49k pairs/language
- Quality: hard negatives by construction (word-order/entity-swap near-duplicates) — a narrow notion of similarity, good for teaching the model *not* to conflate lexical overlap with semantic identity
- Consume/emit fit: complements NLI-shaped positives with contrastive hard negatives; not a source of true paraphrase positives on its own (~50% of pairs are labeled non-paraphrase by design)
- Enrichment: filter to `label==1` (paraphrase) rows if positives-only is wanted; PAWS-X gives multilingual coverage cheaply
- Licence consequence: none — PERMISSIVE_OK does not propagate any obligation

**6. `ltg/en-wiki-paraphrased`**
- Mirror: `ltg/en-wiki-paraphrased` (HF)
- Licence tag: **Apache-2.0** (declared on the paraphrase-generation side)
- Fetch status: **INFERRED / UNVERIFIED at primary** — not independently re-fetched this session; relying on the PRIOR-AUDIT note that flags the exact defect: "the `original` column *is* Wikipedia text and the card is silent on Wikipedia's own CC BY-SA"
- Verdict: **ATTRIBUTION, conditionally** — the `paraphrase` column (model-generated) may genuinely be Apache-2.0 if that's what the generator's licence covers, but the `original` column is Wikipedia prose and inherits **CC BY-SA 4.0** regardless of what tag sits on the row
- Provenance group: Wikipedia-derived (shares lineage with any other Wikipedia-sourced set in the catalogue)
- Size: 5,145,408 rows — largest permissive-tagged paraphrase set found
- Quality: machine-paraphrased (quality/faithfulness of the paraphrase step unverified), largest volume of any candidate here
- Consume/emit fit: paraphrase-pair positives at scale, IF the licence question resolves
- Enrichment: two options — (a) drop the `original` column and use only `paraphrase`-to-`paraphrase` sentence rewrites if that's structurally possible, or (b) accept CC BY-SA propagation from `original` and treat the whole set as SHARE_ALIKE
- Licence consequence: **unresolved — do not admit until the `original`-column question is settled**; provisionally file as UNVERIFIED rather than ATTRIBUTION

**7. `community-datasets/tapaco`**
- Upstream: Tatoeba (https://tatoeba.org), via the TaPaCo paper/corpus
- Mirror: `community-datasets/tapaco` (HF, 689 downloads)
- Licence: VERIFIED this session, HF card licence tag `cc-by-2.0`
- Fetch status: **VERIFIED (mirror)** — tag read directly off the card; Tatoeba's own site-wide licence (CC BY 2.0 FR for user-submitted sentences) is well-documented externally and consistent with this tag, but the Tatoeba site's own licence page was not independently re-fetched this session — treat as **VERIFIED (mirror) leaning VERIFIED**, not fully primary
- Verdict: **ATTRIBUTION**
- Provenance group: standalone (Tatoeba community-sentence lineage — would collapse with any other Tatoeba-derived set the catalogue later admits, per the LibriVox-precedent rule)
- Size: paraphrase clusters across 73 languages, English subset in the low hundreds of thousands of sentence pairs (exact row count not pulled this session)
- Quality: crowd-submitted short sentences, uneven register/domain, but genuinely multilingual and clean provenance
- Consume/emit fit: paraphrase positives, short-sentence register (contrasts with SNLI's longer captions)
- Enrichment: cluster-to-pair expansion (TaPaCo ships as paraphrase *clusters*; pairing requires a sampling strategy — this is a B4 sampling-method decision, not just a licence one)
- Licence consequence: CC BY propagates an attribution requirement to the release manifest

**8. `mteb/OpusparcusPC` / `GEM/opusparcus`**
- Upstream: Language Bank of Finland / OPUS (OpenSubtitles2016 lineage)
- Mirror: `GEM/opusparcus` (HF, 667 downloads)
- Licence: VERIFIED this session, HF card licence tag `cc-by-nc-4.0`, explicit "non-commercial use only" statement on the card
- Fetch status: **VERIFIED (mirror)**; upstream OPUS/OpenSubtitles.org's own terms not independently re-fetched — subtitle-scrape corpora in this family have a documented history of contested rights (subtitle authors vs. distributors), so treat the CC BY-NC-4.0 tag as the operative claim but flag `provenance_red_flags`
- Verdict: **NC**, with a **provenance red flag**: OpenSubtitles-lineage corpora have disputed chain-of-title in the broader literature (subtitle authorship vs. distribution rights) — worth a deeper check before admission, not just accepting the tag
- Provenance group: OpenSubtitles/OPUS lineage — would collapse with any other subtitle-derived candidate
- Size: paraphrase pairs mined from parallel subtitle alignments, six languages, English subset in the hundreds of thousands
- Quality: colloquial/conversational register (subtitles), a different register than SNLI/PAWS's written-prose register — genuine diversity value
- Consume/emit fit: paraphrase positives, conversational register
- Enrichment: none structural beyond standard cap/sample
- Licence consequence: NC tier (already the tier `memory` sits in)

**9. `sentence-transformers/wikianswers-duplicates`**
- Upstream: WikiAnswers / PPDB lineage, via `embedding-data/WikiAnswers`
- Licence: **not found on the HF card** — checked this session, no licence field or statement present
- Fetch status: UNVERIFIED — the underlying WikiAnswers site's own terms were not located this session; PPDB (Paraphrase Database, which WikiAnswers-duplicate mining commonly traces to) itself redistributes under mixed licences depending on the PPDB size tier, adding a second unresolved layer
- Verdict: **UNVERIFIED** — exactly the sentence-transformers-mirror trap this survey was tasked to watch for; do not admit without a primary-source licence read
- Provenance group: WikiAnswers/PPDB lineage
- Size: ~24M duplicate-question pairs (large, per catalogue norms for this family)
- Quality: user-generated Q&A site content, large volume, unknown curation/dedup quality
- Consume/emit fit: would be a large paraphrase-positive source if cleared
- Enrichment: n/a pending licence
- Licence consequence: n/a pending licence — **REFUSE pending verification**, not admitted provisionally

### STS graded pairs

**10. `sentence-transformers/stsb` / `mteb/stsbenchmark-sts`**
- Upstream: STS Benchmark (Cer et al., SemEval STS 2012-2017 aggregation), hosted historically at ixa2.si.ehu.eus
- Licence: **HF card tag is `unknown`** — VERIFIED this session (both mirrors checked; neither declares a licence). Direct fetch of the ixa2.si.ehu.eus primary page failed this session (TLS certificate error on the host) — could not independently confirm upstream terms
- Verdict: **UNVERIFIED**
- Provenance group: SemEval-STS lineage (shared with #11-14 below — all draw from the same STS shared-task series 2012-2017)
- Size: 8,628 graded pairs (train+dev+test), the standard STS-B split
- Quality: gold-standard human similarity judgments (0-5 continuous scale), the field's reference graded-similarity benchmark — high value if the licence clears
- Consume/emit fit: this is the **graded-similarity gate the ground doc flags as "fetched and not trained on... the graded gate never ran"** — filling this gap is a named, already-recognized need for `compress`, independent of this survey
- Enrichment: none needed — it's eval-shaped as-is
- Licence consequence: n/a pending verification — **do not train on it under a `TRAIN_OK`-only fetcher until the licence is resolved**; usable as EVAL-ONLY is a separate, weaker claim that also needs the primary source checked (SemEval shared-task data is traditionally "free for research," which is close to but not identical to a training-permissive grant)

**11-14. `mteb/sts12-sts` through `mteb/sts17-crosslingual-sts`, `mteb/sickr-sts`**
- Same SemEval-STS-series provenance group as #10; same `unknown` mirror-tag pattern observed on the `sts12`/`sickr` cards during this survey's earlier API pass
- Fetch status: **INFERRED** to share #10's unresolved status — not independently re-fetched per-dataset this session (time-boxed); flagging as a **block of candidates needing one primary-source resolution that clears all of them at once**, since they're one provenance group
- Verdict: **UNVERIFIED** (block)
- Note: `mteb/sickr-sts` specifically re-mirrors SICK (see #15) — that one *does* have a resolved upstream licence (NC), so it should NOT be lumped with the true-unknown SemEval-STS subset; listed here only because the HF card search surfaced it under the STS query

**15. SICK (`RobZamp/sick`, `mteb/sickr-sts`, any mirror)**
- Upstream: http://marcobaroni.org/composes/sick.html
- Licence: VERIFIED this session at the primary source — **Creative Commons Attribution-NonCommercial-ShareAlike 3.0**
- Mirror tag: `cc-by-nc-sa-3.0` (matches upstream, unusually — this is a case where the mirror tag is NOT lying)
- Verdict: **NC** (with SA also attached — a composed model using SICK inherits both the NC and the SA obligation)
- Provenance group: standalone (SICK is a purpose-built compositional-semantics benchmark, not derived from SNLI/MultiNLI/PAWS)
- Size: 9,840 sentence pairs (relatedness score 1-5 + entailment label) — small
- Quality: carefully constructed for compositional/systematic entailment phenomena, widely used as a robustness probe; small enough that it's a good graded-eval supplement, not a volume source
- Consume/emit fit: graded-similarity eval (relatedness score), same gap as STS-B (#10) but with a cleared licence
- Enrichment: none — small, ready to use as-is
- Licence consequence: NC+SA — again within the tier `memory` already occupies, so marginal cost is low, but it is the **NC+SA-cleared candidate to reach for before spending more effort chasing the SemEval-STS unknowns in #10-14**, since it fills the same graded-gate gap the ground doc already flags as missing

**16. `mteb/biosses-sts`**
- Upstream: BIOSSES (biomedical STS benchmark, Turkish institution-authored)
- Fetch status: **not fetched this session** — flagged as a candidate worth a follow-up primary-source check (biomedical-domain graded pairs would diversify `compress`'s domain coverage, currently all general-prose), not yet a verdict
- Verdict: **UNVERIFIED**

### Summarisation pairs (sentence/paragraph-level compression signal)

**17. `abisee/cnn_dailymail`**
- Mirror licence tag: `apache-2.0` — VERIFIED this session, HF card states "released under the Apache-2.0 License"
- Fetch status: **flagged, not accepted** — this is a live instance of the mirror-lying pattern the task brief called out. The Apache-2.0 tag, per the repo's own README (checked this session), covers the *dataset-construction code* (See et al.'s `cnn-dailymail` GitHub repo, itself MIT-licensed for the scripts). The actual article text is fetched from `cs.nyu.edu/~kcho/DMQA/`, itself a mirror of the original DeepMind Q&A dataset (Hermann et al. 2015), which is built by scraping CNN.com and DailyMail.co.uk — copyrighted news-organization content with **no located licence grant on the text itself** anywhere in the chain checked this session
- Verdict: **BLOCKING** (grant_scope: `code_only` — exactly the CSS10/Libri-Light shape the ground doc's field list names) — a code licence is being read as a data licence
- Provenance group: DeepMind-QA / CNN-DailyMail news-scrape lineage
- Consume/emit fit: would be excellent long-document-to-summary compression pairs if clean
- Enrichment/licence consequence: n/a — REFUSE as currently tagged; would need a genuine rights grant from CNN/DailyMail (unlikely) or a different summarisation source entirely

**18. `EdinburghNLP/xsum`**
- Mirror licence tag: **`unknown`** — VERIFIED this session, HF card declares no licence
- Source: BBC News articles — BBC's own terms of use are well-documented as restricting reuse/redistribution of article text outside BBC properties (not independently re-fetched this session, but this is consistent with the `unknown` tag rather than contradicting it)
- Verdict: **BLOCKING**
- Provenance group: standalone (BBC lineage)
- Consume/emit fit: would be single-document summary pairs, high quality, but not usable
- REFUSE, no partial fix available (same shape as `visual`'s tiny-imagenet problem: the whole thing is one restricted-source scrape)

**19. `alexfabbri/multi_news`**
- Mirror licence tag: `other` — VERIFIED this session, HF card quotes the LILY Lab "Dataset Usage Agreement": *"The Dataset is intended for non-commercial research and educational purposes only, and is made available free of charge"*
- Verdict: **REFUSE-TERMS** — this is explicitly research/educational-only language, which the operator stance names as a REFUSE class distinct from ordinary NC ("research-only / non-redistributable terms" is listed as a refusal trigger, not absorbed by the NC-tolerant policy)
- Provenance group: standalone (newser.com-sourced)
- Consume/emit fit: would be multi-document summarisation pairs
- REFUSE — terms forbid the redistribution/training use this factory needs regardless of the NC-tolerant stance on ordinary commercial-use restrictions

### Quora-lineage

**20. Quora Question Pairs (`AlekseyKorshuk/quora-question-pairs` and similar re-mirrors)**
- Upstream: Quora's 2017 dataset release (via a Kaggle competition; original Quora blog post)
- Fetch status: **UNVERIFIED** — this session's WebFetch/WebSearch passes did not surface a verbatim primary licence statement; several re-mirrors (`AlekseyKorshuk/...`, `Heliosoph/...`, `CCRss/...`) carry no licence field at all. Quora's Kaggle competition rules are known (from general knowledge, not fetched this session) to restrict the data to the competition/non-commercial research context, but this was not independently confirmed this session and should be treated as INFERRED, not VERIFIED
- Verdict: **UNVERIFIED, leaning BLOCKING** pending a primary fetch of the actual Kaggle competition rules page or Quora's original release terms
- Provenance group: standalone (Quora-lineage; PAWS's Quora-derived half, see #5, is a *separate* release under Google's own grant, not this one — do not conflate the two even though both touch Quora sentences)
- Note: `google-research-datasets/paws`'s "freely used for any purpose" grant is Google's own re-release grant over its constructed pairs, and does NOT retroactively license the raw Quora QQP dataset itself — keep these provenance groups distinct in the catalogue

---

## What was REFUSED and why

| id | reason |
|---|---|
| `abisee/cnn_dailymail` (and `ccdv/cnn_dailymail` mirror) | Apache-2.0 tag covers construction code, not the underlying CNN/DailyMail article text (copyrighted news content, no located grant) — `grant_scope: code_only` mismatch, the same class of error the ground doc names for CSS10/Libri-Light |
| `EdinburghNLP/xsum` | No licence anywhere in the chain; BBC-sourced restricted content |
| `alexfabbri/multi_news` (and `tau/multi_news`) | LILY Lab's own Dataset Usage Agreement states "non-commercial research and educational purposes only" — explicit research-only language, a REFUSE class under the operator stance, not absorbed by the NC-tolerant policy |
| `hkust-nlp/SynCSE-**partial**-NLI` | Named trap: MIT tag on the wrapper repo, but sentences are drawn from SimCSE's SNLI+MNLI pipeline — SA obligation carried through despite the tag |
| `lxyuan/synthetic-nli-triplet`, `SeanLee97/all_nli_angle_format_b` | Named traps (PRIOR-AUDIT): permissive tags applied by repackagers over all-nli-derived text that never carried a clean licence |
| `facebook/xnli`, `nyu-mll/glue` (mnli config) | Same MultiNLI text as candidate #2, without the genre column that makes filtering possible — redundant with a worse-provenance copy |
| `sentence-transformers/wikianswers-duplicates` | No licence field on the card; underlying WikiAnswers/PPDB licence not located this session — REFUSE pending verification, not admitted provisionally |
| Quora Question Pairs (all re-mirrors surveyed) | No verbatim primary licence text located this session; provisional REFUSE pending a direct fetch of Kaggle's competition rules or Quora's own release terms |
| `mteb/sts12-sts` … `mteb/sts17-crosslingual-sts`, `sentence-transformers/stsb` | `unknown` licence tag on every mirror checked; primary-source fetch (ixa2.si.ehu.eus) failed this session (TLS error) — REFUSE pending a successful primary fetch, despite being the field's reference graded-STS benchmark |
| `ltg/en-wiki-paraphrased` | Apache-2.0 tag applies to the paraphrase-generation step only; `original` column is unlicensed-for-this-purpose Wikipedia text — REFUSE pending a column-level resolution (drop `original`, or accept CC BY-SA propagation) |

---

## Cross-cutting notes for the catalogue/factory shape

- **Provenance groups identified this session**: (1) SNLI/MultiNLI/all-nli/SynCSE-partial-NLI, (2) Wikipedia-lineage (en-wiki-paraphrased's `original` column, and would collapse with any other Wikipedia-sourced text elsewhere in the catalogue), (3) Tatoeba (tapaco), (4) OpenSubtitles/OPUS (opusparcus), (5) SemEval-STS-series (sts12-17, stsb), (6) SICK (standalone), (7) PAWS/PAWS-X (standalone, Google's own re-release grant, distinct from raw Quora QQP), (8) Quora QQP raw (standalone, distinct from #7), (9) DeepMind-QA/CNN-DailyMail news-scrape (standalone), (10) BBC/XSum (standalone), (11) LILY-Lab/Multi-News (standalone).
- **The sentence-transformers-mirror problem recurs exactly as flagged**: `sentence-transformers/stsb` and `wikianswers-duplicates` both carry no independently-verifiable licence, consistent with the "0 of 96 sentence-transformers/* declare a licence" finding already on record. Every ST-namespace mirror in this survey was cross-checked against a non-ST mirror or the true upstream where one exists (`mteb/*` for STS, direct `google-research-datasets/*` for PAWS, primary homepages for SNLI/SICK).
- **`compress`'s graded-similarity gap is real and still open**: the ground doc already flags that the graded gate "never ran." Of the graded candidates surveyed, SICK (#15) is the only one with a **cleared** licence (NC+SA); the SemEval-STS series (#10-14), which is larger and more standard, remains UNVERIFIED pending a successful primary fetch (this session's attempt hit a TLS error on the historical host and was not retried through an alternate route).
- **Enrichment/licence-propagation pattern across accepted candidates**: PERMISSIVE_OK sources (PAWS, SynCSE-scratch) impose nothing on the composed release; SHARE_ALIKE sources (SNLI, filtered MultiNLI) propagate SA to any derived pair set that retains their rows; NC sources (ANLI, Opusparcus, SICK) sit inside the NC tier `memory` already occupies via GooAQ, so admitting more NC material is free under the strictest-input rule already in force — the real lever for improving `compress`'s B1/B2 numbers is adding **more independent PERMISSIVE_OK/ATTRIBUTION sources** (raising `N_eff`), not chasing NC volume.
