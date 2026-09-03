# Dataset factory — candidate catalogue, 2026-09-03

**Status:** the synthesis of one dataset-factory survey pass (DEC-57 / P2′f). It records
**159 candidate datasets** across seven faculties, each with a licence verdict, the primary
licence text it rests on, an adversarial verification status, an enrichment plan and that
plan's licence consequence. It changes no code and fetches nothing new; it reconciles the
eight surveys, the eight adversarial verification passes and the enrichment analysis into one
catalogue, and states what is still open.

The machine-readable form is `docs/design/datasets/catalogue-2026-09-03.json` — one object per
dataset in the shape `00-ground.md` §(b) specifies. **The JSON is authoritative for per-entry
fields**; this document is the reading of it.

The full evidence is copied verbatim to `docs/design/evidence/dataset-factory-2026-09-03/`
(the ground doc, eight `10-survey-*.md`, `20-enrichment-licence-impact.md`, eight
`30-verify-*.md`). Nothing below is a claim this catalogue makes on its own authority: every
verdict traces to a quoted primary licence text in those files, with the URL and fetch date.

> **Neither the author nor the reader of this document is a lawyer.** Everything here reports
> what a licence *says* and what follows mechanically. The questions that need a human — in
> several cases a lawyer — are collected at the end and are deliberately not resolved here.

---

## 1. The operator's licence stance, restated

Recorded verbatim from the operator's 2026-09-02/03 statements, because every verdict below is
an application of it:

- The **model architecture code is MIT**. **Datasets, and therefore weights and activations,
  may carry non-MIT licences and must align with each other.**
- **Only open-weights-compliant licensed datasets are permissible.**
- **Non-commercial (NC) is acceptable.** It changes the release licence from MIT to an
  NC-restricted open-weights licence, and the composed model inherits the strictest input
  (DEC-31).
- What matters is **high quality, open weights, open source**.
- **REFUSE**, as a class: no-derivatives (ND); research-only or non-redistributable terms;
  licences forbidding model training or weight release; distributors that disclaim owning what
  they distribute; consent-revocable corpora, unless a held snapshot is `EVAL-ONLY` under a
  `CONSENT_OPEN` class.
- **Provenance rule: mirrors lie.** The verdict rests on the **upstream/primary licence text**,
  recorded verbatim beside the mirror tag. Datasets sharing one lineage form **one provenance
  group** (the LibriVox precedent, DEC-46).
- The **strictest-input rule applies to every emitted or enriched dataset**, not only to region
  training corpora.

The measured basis for the provenance rule is on record: ten mismatches between HF mirror tags
and upstream terms, and all 75 surveyed `sentence-transformers/*` datasets declaring none. This
pass reproduced the pattern at a higher rate than that — see §4.

---

## 2. Verdict and verification classes

**Verdict** (from `LICENCE-FOR-OPEN-WEIGHTS.md`, extended by `AUDIO-CORPUS-AUDIT.md`):

| verdict | meaning |
|---|---|
| `PERMISSIVE_OK` | MIT/Apache-2.0/BSD/CC0/public domain at BOTH mirror and upstream |
| `ATTRIBUTION` | CC BY-family or ODC-By; usable, notice required in the release |
| `SHARE_ALIKE` | CC BY-SA-family, ODbL, CDLA-Sharing, OpenRAIL-M; whether trained weights are "adapted material" is unsettled — flagged, not resolved |
| `NC` | a genuine rights-holder non-commercial term; per DEC-31 it does not block, it moves the region and the composed model to an NC tier |
| `BLOCKING` | no licence grant exists at all, an ownership dispute, a paywall corpus, or a restriction the NC-tolerant policy does not absorb |
| `REFUSE` | refused at ingest under a named rule (ND, research-only, no-redistribution, distributor disclaims ownership, unclassified generator) |
| `UNVERIFIED` | the primary source does not settle the question; **not admissible for training** |
| `CONSENT_OPEN` | copyright-clean but subject to live consent revocation (recommended class, OD-11; **no candidate in this pass was positively filed under it**) |

**Verification status** — the second axis, and the one that decides admissibility:

| status | meaning | count |
|---|---|---|
| `VERIFIED` (V) | the adversarial verifier fetched the primary and agrees with the surveyor's quote and verdict | **105** |
| `CONTRADICTED` (**C**) | the verifier, or this synthesis, fetched the primary and it disagrees; **the corrected verdict recorded here overrides the surveyor's** | **26** |
| `UNVERIFIABLE` (*U*) | a fetch was attempted and the primary does not resolve the question either way; the entry stays listed and is **not admissible until read** | **18** |
| `REFUSED-CLOSED` (R) | verdict was already REFUSE/BLOCKING in the survey and out of the verifier's mandate; kept so the refusal is not re-discovered | **10** |

**159 entries total.** A `CONTRADICTED` verdict is the operative one everywhere below; where the
correction came from this synthesis rather than a verifier, the entry's `verification_note`
says so.

---

## 3. Totals

| faculty | entries | V | **C** | *U* | R | PERMISSIVE_OK | ATTRIBUTION | SHARE_ALIKE | NC | UNVERIFIED | BLOCKING | REFUSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `language_code` | 29 | 20 | 5 | 3 | 1 | 9 | 4 | 4 | 1 | 10 | 0 | 1 |
| `memory` | 39 | 23 | 5 | 6 | 5 | 4 | 5 | 9 | 6 | 8 | 2 | 5 |
| `reasoning` | 19 | 14 | 3 | 1 | 1 | 8 | 2 | 1 | 4 | 3 | 0 | 1 |
| `numeric_math` | 4 | 3 | 0 | 1 | 0 | 3 | 0 | 0 | 0 | 1 | 0 | 0 |
| `visual` | 21 | 14 | 4 | 2 | 1 | 3 | 0 | 0 | 0 | 6 | 9 | 3 |
| `language_trunk` | 26 | 17 | 6 | 1 | 2 | 7 | 5 | 3 | 0 | 6 | 0 | 5 |
| `moral_safety` | 21 | 14 | 3 | 4 | 0 | 8 | 5 | 3 | 2 | 3 | 0 | 0 |
| **total** | **159** | **105** | **26** | **18** | **10** | **42** | **21** | **20** | **13** | **37** | **11** | **15** |

15 entries carry `usage_tag: EVAL-ONLY`. 11 carry a `MODEL-OUTPUT-TERMS` red flag (§4.5). The
catalogue spans **106 distinct provenance groups**, which is the number B1/B2 are computed over
— not 159.

Read the `UNVERIFIED` column as the headline: **37 of 159 candidates (23%) cannot be admitted
today because no primary source settles their licence**, and 18 of those had a fetch attempted
and failed. That is the factory's actual backlog, not the refused list.

---

## 4. What this synthesis found that neither the surveys nor the verifiers had

Five findings that are new at the reconciliation step, ordered by consequence.

### 4.1 Natural Questions is not permissive, and that removes `memory`'s best recommendation

`10-survey-retrieve-memory.md` ranks `sentence-transformers/natural-questions` **first** among
all `memory` candidates — "cleanest licence, real human queries, directly dilutes GooAQ's 79.1%
B1 concentration" — on the strength of the `google-research-datasets/natural-questions` repo's
Apache-2.0 LICENSE, which `30-verify-retrieve-memory.md` independently confirmed.

But the **same survey** applies the opposite reasoning two entries later. For `miracl/miracl`
(#20) it writes: *"grant_scope: **code_only in substance** — MIRACL's passages are Wikipedia text
… Apache-2.0 is the pipeline/tooling licence, the actual passage text still carries Wikipedia's
CC BY-SA obligation regardless of the badge on the repo."* It repeats this for
`castorini/mr-tydi` (#21). **NQ's passages are also Wikipedia text** — the survey's own entry
says so: *"real Google search queries + human-annotated long/short answers from Wikipedia."*

Applying the survey's own rule consistently makes NQ **SHARE_ALIKE, not PERMISSIVE_OK**, and puts
it in the **Wikipedia provenance group** alongside SQuAD, HotpotQA, FEVER, DBpedia, MIRACL and
Mr.TyDi. Neither verifier caught this because each entry was checked against its own primary in
isolation, and NQ's primary *does* say Apache-2.0 — the defect is that the licence covers the
wrong layer, which only shows up when the entries are read side by side.

**Consequence:** `memory` has no large clean-permissive retrieval source at all. Its
clean-permissive tier is 1.3% of a 1e10-token region target (§7), and NQ does not dilute
GooAQ — it moves the concentration problem from GooAQ into an even larger Wikipedia group.
Recorded in the JSON as `verification_status: CONTRADICTED` with the note naming this synthesis
as the source of the contradiction.

### 4.2 COCO's terms page was read after all — by the wrong verifier

`30-verify-visual.md` records COCO's terms-of-use page as the single highest-leverage unfetched
source in the whole visual survey, failed across two sessions and six attempts (it is a
single-page app; the `#termsofuse` anchor loads its content by JS). It left entry 1 explicitly
`INFERRED, not VERIFIED`.

`30-verify-other-sources.md` §"Strengthened" fetched it — by requesting the JS-loaded fragment
URL `https://cocodataset.org/dataset/termsofuse.htm` directly instead of the anchor — and got:

> "The annotations in this dataset along with this website belong to the COCO Consortium and are
> licensed under a Creative Commons Attribution 4.0 License." / "The COCO Consortium does not own
> the copyright of the images. Use of the images must abide by the Flickr Terms of Use."

**COCO's BLOCKING verdict is therefore VERIFIED at the primary, not inferred**, and the survey's
own top open question is closed. It closes five downstream entries with it (VQAv2, OK-VQA,
A-OKVQA, GQA, Localized Narratives). Neither pass noticed the other had done it; recorded here.

### 4.3 CourtListener was refused on a claim the primary contradicts

The one procedural defect in the whole pass, and it is a false REFUSE rather than a false admit.
`10-survey-other-sources.md` refused CourtListener bulk data as `BLOCKING` on a CC BY-ND 4.0
claim it had only *search*-verified, while writing in the same entry *"recommend direct WebFetch
confirmation before final close-out."* `30-verify-other-sources.md` did that fetch, twice, and
Free Law Project's own current bulk-data documentation says:

> "Our bulk data files are free of known copyright restrictions"

displayed with the Creative Commons **Public Domain Mark**, with **every** bulk table
individually marked Public Domain. No ND clause appears anywhere on that page. The general
`courtlistener.com/terms/` page returns 403 from CloudFront and could not be read, so a CC BY-ND
footer may still exist covering the site's presentation layer — which would not touch the data.

Held here as `UNVERIFIED`, **not** REFUSE, pending that one page. **A `BLOCKING` call is exactly
as costly to get wrong as a bad `PERMISSIVE_OK`**: this one wrote off 10M+ opinions, a large
independent provenance group, on a claim primary-source reading does not support. The process
lesson generalises: **REFUSE entries are not exempt from the fetch-before-file discipline
applied to admits.**

### 4.4 The strictest-input rule does not work at the dataset layer, and the catalogue has to be read that way

`20-enrichment-licence-impact.md` §2.1 establishes this from licence text and it reorganises the
whole catalogue. Every share-alike family — **CC BY-SA, ODbL, ODC-By, CDLA-Sharing, and
OpenRAIL-M** — admits exactly **one** emitted dataset licence, its own, and forbids added
restrictions. So:

- There is **no** emitted-dataset licence that satisfies a CC BY-SA input *and* an NC input.
  BY-SA §3(b)(1) demands the same licence elements (no NC); §3(b)(3) forbids imposing additional
  restrictions. The same impossibility holds for CDLA-Sharing × NC (§3.3 names commercial
  restrictions expressly), ODbL × NC (§4.7(a)), ODC-By × NC (§4.4), and for every
  share-alike × *different* share-alike pair.
- The strictest-input rule works for **weights**, where one licence must be chosen for one blob.
  At the **dataset** layer, for the SA×NC pair, there is no strictest licence — **there is no
  licence at all**.
- **Therefore the factory partitions first and enriches second.** Each faculty emits **one file
  per licence tier**, shipped together as a Collection with a top-level manifest (which CC
  expressly blesses). Cross-dataset *pairing* is the operation that trips this: joining a CC
  BY-SA passage to a GooAQ-NC query produces a row with two irreconcilable parents that cannot be
  emitted under any licence. **That row must not be constructed** — a planner constraint upstream
  of ingest, not a filter after it (`R5`).

Every "clean-permissive tier" and "NC-inclusive tier" number in §7 is therefore a **union across
separate files**, never a merged corpus. `ODC-By` is the trap most likely to be missed: despite
its name it is a **database-level copyleft** (§4.2(a)), so FineWeb-Edu, C4, peS2o, OpenWebMath,
WildGuard and SciFact's abstract corpus each emit into an ODC-By-only file — they cannot be
folded into a CC BY tier.

### 4.5 There is a `MODEL-OUTPUT-TERMS` class the verdict scheme has no slot for

Eleven catalogue entries are derived from closed-API model output: `KodCode/KodCode-V1`,
`m-a-p/CodeFeedback-Filtered-Instruction`, `nickrosh/Evol-Instruct-Code-80k-v1`,
`hkust-nlp/SynCSE-scratch-NLI`, `camel-ai/math`, `meta-math/MetaMathQA`,
`microsoft/orca-math-word-problems-200k`, `AI-MO/NuminaMath-CoT` (via its orca_math subset),
`tatsu-lab/alpaca`, `HuggingFaceH4/ultrachat_200k`, and `toxigen/toxigen-data`.

Their licence tags span MIT, Apache-2.0, CC BY-NC-4.0 and CDLA-Permissive-2.0 — **the tag tells
you nothing about the question**, because the question is contractual, not copyright.
`20-enrichment` §1.7 Shape A quotes the live clauses (OpenAI: *"Use Output to develop models
that compete with OpenAI"* and *"Automatically or programmatically extract data or Output"*;
Anthropic: *"may not … access the Services to build a competing product or service, including
to train competing AI models"*) and reasons that **the restriction travels with the person who
accepted it, not with the data**. A third party receiving the dataset never accepted it.

`20-enrichment` §3 SYN-L2 therefore **REFUSES** these generators *for the factory's own
generation*, which is the right call and is not in question. But for **pre-existing third-party
datasets already released under their own licence**, the operator is not the accepting party,
and §5.6 registers exactly this as open: *"Do contractual output restrictions bind a third party
who receives the dataset? … No privity suggests not; the operator who accepted the terms is
bound regardless. The factory's REFUSE makes the question moot going forward, but it matters for
any material already in the tree."*

These eleven entries are therefore filed at their licence verdict with a
`MODEL-OUTPUT-TERMS` red flag, **not** auto-refused and **not** auto-cleared. It is one operator
decision covering all eleven, and it is worth roughly 1e9 tokens across three faculties. The
correct comparison is `HuggingFaceTB/cosmopedia`, which is also synthetic but was generated by
Mixtral under **Apache-2.0 weights run locally** — a `SAFE`-class generator that attaches
nothing. That is the shape to prefer.

---

## 5. Provenance groups

B1, B2 and B5 are computed on **provenance groups**, not on dataset names. The catalogue's 159
entries collapse to **106 distinct groups**, and three collapses do most of the damage:

| group | swallows | consequence |
|---|---|---|
| **`wikipedia`** | SQuAD, HotpotQA, FEVER, DBpedia-entity, MIRACL, Mr.TyDi, **Natural Questions** (§4.1), `ltg/en-wiki-paraphrased`'s `original` column, `wikimedia/wikipedia`, PAWS's Wikipedia half, Common Corpus's OpenWeb stratum | `memory`'s entire share-alike tier is ~99% one group. Adding MIRACL's 18 languages or Mr.TyDi's 11 raises **strata**, not **sources** — N_eff stays ≈ 1. |
| **`PG-STACK`** | the-stack-v2-dedup, the-stack-dedup, starcoderdata, stack-edu, opc-annealing-corpus (by content, despite its ODC-BY tag) | `language_code`'s largest tier by three orders of magnitude is **one** source. 783 GB at N_eff = 1. |
| **`common-crawl`** | FineWeb-Edu, Dolma, RedPajama-v2, C4, Nemotron-CC — and, at the subset layer, Dolma again overlaps peS2o, Gutenberg and Wikipedia | the trunk has enormous volume from few sources; admitting four CC derivatives is one source, not four. |

Two further collapses to hold: **`PG-COMPETITIVE`** (apps, code_contests, OpenCodeReasoning
1 & 2, KodCode, LeetCodeDataset — competitive-programming problem text recurs across all of
them) and **`s2orc`** (peS2o, SciFact's abstract corpus, SciDocs, probably Qasper). And two
deliberate *non*-collapses: PAWS's Quora-derived half is **not** the raw Quora QQP group (Google's
grant is its own re-release), and `google/civil_comments` is **not**
`google/jigsaw_toxicity_pred` (different Jigsaw challenges, different corpora).

---

## 6. Candidate tables, per faculty

Legend: **V** = VERIFIED · **C** = CONTRADICTED (this verdict overrides the surveyor's) ·
*U* = UNVERIFIABLE (listed, **not admissible until read**) · R = REFUSED-CLOSED.
Full fields, including the verbatim upstream licence quote, its URL and fetch date, are in
`docs/design/datasets/catalogue-2026-09-03.json`.

#### `language_code` — candidates (28 listed, 1 refused/blocking below)

| dataset | verdict | verif | provenance group | size | enrichment → licence result |
|---|---|---|---|---|---|
| `bigcode/humanevalpack` | PERMISSIVE_OK · EVAL-ONLY | *U* | PG-EVALSTD | 984 rows / 1.16 MB | none -- held-out eval only → n/a at eval |
| `code-search-net/code_search_net` | PERMISSIVE_OK | V | PG-CSN | 4,141,072 rows / 3.93 GB (datasets-server, 2026-09-03) | none required -- already query/code paired; admit the full multi-language original in place of the Python-only slice → MIT imposes nothing on the emitted dataset or the weights; stays in the clean-permissive tier |
| `codeparrot/apps` | PERMISSIVE_OK | V | PG-COMPETITIVE | 10,000 problems with test cases and reference solutions; on disk in an 18 GB dir shared with code_contests | already integrated; reserve's executable-ground-truth anchor → MIT imposes nothing |
| `evalplus/humanevalplus` | PERMISSIVE_OK · EVAL-ONLY | V | PG-EVALSTD | 164 rows / 2.9 MB | none -- held-out eval only → Apache-2.0 imposes only notice retention |
| `evalplus/mbppplus` | PERMISSIVE_OK · EVAL-ONLY | V | PG-EVALSTD | 378 rows / 1.1 MB | none -- held-out eval only → Apache-2.0 imposes only notice retention |
| `Nan-Do/code-search-net-python` | PERMISSIVE_OK | **C** | PG-CSN | 455,243 rows | replace with the full multi-language CodeSearchNet rather than adding to it → no change; permissive stays permissive |
| `openai/openai_humaneval` | PERMISSIVE_OK · EVAL-ONLY | V | PG-EVALSTD | 164 rows / 83.9 KB | none -- held-out eval only → MIT imposes nothing; the EVAL-ONLY usage tag is structural, not licence-driven |
| `SWE-bench/SWE-smith` | PERMISSIVE_OK | V | PG-SWEBENCH | 59,136 rows / 277.8 MB | none required structurally; worth a dedicated follow-up pass at scale → MIT imposes nothing; clean-permissive tier |
| `zai-org/humaneval-x` | PERMISSIVE_OK · EVAL-ONLY | *U* | PG-EVALSTD | size probe failed (HTTP 500 on datasets-server -- loader issue, not a licence issue) | none -- held-out eval only → n/a at eval |
| `deepmind/code_contests` | ATTRIBUTION | V | PG-COMPETITIVE | 4,044 problems at the top level (the per-problem solution/test expansion is what fills the 18 GB dir) | already integrated → CC BY 4.0 propagates a TASL attribution obligation into the emitted dataset and the model card; no share-alike, no NC |
| `google-research-datasets/mbpp` | ATTRIBUTION · EVAL-ONLY | **C** | PG-EVALSTD | 1,401 rows / 351 KB | none -- held-out eval only → CC BY 4.0 attribution obligation, discharged in the model card |
| `nvidia/OpenCodeReasoning` | ATTRIBUTION | V | PG-COMPETITIVE | 337,766 rows | strip to NVIDIA's own generated reasoning/solution text and treat embedded problem statements as attribution-only pass-through → CC BY 4.0 propagates attribution to the emitted dataset; the problem-statement layer stays UNVERIFIED and is not cleared by the … |
| `nvidia/OpenCodeReasoning-2` | ATTRIBUTION | V | PG-COMPETITIVE | 220,000 rows | same as OpenCodeReasoning → same as OpenCodeReasoning |
| `bigcode/starcoderdata` | SHARE_ALIKE | V | PG-STACK | 783 GB, 86 languages (public reporting; not independently re-measured) | language cap sampling per CORPUS-CONTRACT §1.1; pair with docstring extraction rather than raw file dumps to fix the 93.9%-truncation defect → OpenRAIL-M propagates; same singleton tier as PG-STACK's other rows |
| `bigcode/the-stack-dedup` | SHARE_ALIKE | V | PG-STACK | ~3 TB (public reporting) | none proposed; fallback only if v2 gate access is unavailable → OpenRAIL-M propagates |
| `bigcode/the-stack-v2-dedup` | SHARE_ALIKE | V | PG-STACK | ~900B tokens class (gated; size endpoint 401) | language-stratified sampling to CORPUS-CONTRACT §1.1's cap table; strided/reservoir sample per B4 → OpenRAIL-M behavioural clause propagates to every derived sample; the emitted file is a singleton OpenRAIL-M tier that cannot merge with CC BY-SA, ODC-By, CD… |
| `HuggingFaceTB/stack-edu` | SHARE_ALIKE | V | PG-STACK | 167,063,359 index rows / 17.47 GB of metadata; ~125B tokens once resolved against Software Heritage | Software Heritage resolution pipeline, then the same language caps → inherits Stack v2's OpenRAIL-M; the SWHID index itself grants nothing (R2 refuses a metadata_only grant while corpus CONTENT is being ingested) |
| `KodCode/KodCode-V1` | NC | V | PG-COMPETITIVE | 487,432 rows / 2.64 GB | none structurally; carry the model-output red flag into the receipt → NC propagates -- emits into the NC tier, which cannot be merged with any SHARE_ALIKE or ODC-By tier file |
| `bigcode/commitpack` | UNVERIFIED | **C** | PG-COMMITPACK | not probed (loader 501) | prefer commitpackft (the quality-filtered subset); keep this row only as the provenance-group note → same as commitpackft |
| `bigcode/commitpackft` | UNVERIFIED | **C** | PG-COMMITPACK | not probed (datasets-server 500 on this loader) | per-row `license`-field filter: keep mit/apache-2.0/bsd-2/3-clause/isc/cc0-1.0/unlicense, judge mpl-2.0/lgpl-2.1/epl-1.0 separately as file-level weak copyleft, drop agpl-3.0 and unknown outright; carry the surviving value into the emitted row's `lic_class`… |
| `m-a-p/CodeFeedback-Filtered-Instruction` | UNVERIFIED | **C** | evol-instruct-code | 156,526 rows / 371.2 MB | isolate or drop the Evol-Instruct-Code-derived fraction before treating the remainder as an independent permissive source → if Evol-Instruct-Code resolves to CC BY-NC-SA, that fraction drags the derived set to NC-SA under strictest-input; if it resolves to … |
| `neulab/conala` | UNVERIFIED | V | conala | 596,770 rows / 160.96 MB | n/a pending licence resolution → n/a; if it resolves to Stack Exchange's CC BY-SA it also inherits the per-item attribution problem (R9) |
| `nickrosh/Evol-Instruct-Code-80k-v1` | UNVERIFIED | V | evol-instruct-code | 78,264 rows / 121.5 MB | n/a pending resolution → unresolvable today: an NC-SA reading and an Apache-2.0 reading emit into different, non-mergeable tiers |
| `OpenCoder-LLM/opc-annealing-corpus` | UNVERIFIED | V | PG-STACK | 11,643,084 rows (datasets-server, 2026-09-03) | none until the tag conflict is resolved against the OpenCoder technical report and BigCode's opt-out list → if admitted, inherits OpenRAIL-M from PG-STACK; the ODC-BY tag cannot be the operative one over content the card says came from Stack v2 |
| `OpenCoder-LLM/opc-fineweb-code-corpus` | UNVERIFIED | *U* | opencoder-fineweb | 100,920,235 rows / 147.9 GB original (datasets-server, 2026-09-03) | dedup against StarCoderData/Stack-lineage rows before mixing (crawled docs commonly duplicate repo READMEs already in PG-STACK) → if MIT is confirmed, emits into the clean-permissive tier and imposes nothing; today it cannot be emitted at all |
| `princeton-nlp/SWE-bench` | UNVERIFIED · EVAL-ONLY | V | PG-SWEBENCH | not probed | none -- held-out eval only → n/a; EVAL-ONLY holds regardless of the licence outcome |
| `Software Heritage` | UNVERIFIED | V | (access layer -- per-repo) | the broadest source-code archive available, including repos GitHub has lost | per-file SPDX licence detection and filtering to permissively-licensed repos -- a substantial filtering project, not a drop-in dataset. Bulk access requires contacting Software Heritage directly per their terms. → every repository pulled through it still ne… |
| `SWE-bench/SWE-bench_Verified` | UNVERIFIED · EVAL-ONLY | V | PG-SWEBENCH | not probed | none -- held-out eval only → n/a |

**Refused / blocking in `language_code`**

| dataset | verdict | verif | why refused |
|---|---|---|---|
| `newfacade/LeetCodeDataset` | REFUSE | R | the distributor plainly does not own what it redistributes; LeetCode does, and its terms forbid the scraping — Closed REFUSE. Kept in the catalogue so a future pass does not re-discover it. The same REFUSE-TERMS logic applies to any Codeforces/AtCoder/HackerRank problem text reached by direct scr… |

#### `memory` — candidates (32 listed, 7 refused/blocking below)

| dataset | verdict | verif | provenance group | size | enrichment → licence result |
|---|---|---|---|---|---|
| `deepmind/narrativeqa` | PERMISSIVE_OK | V | narrativeqa | not probed | pair summary-level QA with a retrieved passage/chapter rather than the full book: it fits the episodic store's chunked-recall use better AND avoids the unverified movie-script rights question → the DeepMind annotation layer is Apache-2.0 and stays clean-per… |
| `google-research-datasets/paws` | PERMISSIVE_OK | V | paws | ~108k labelled pairs (Wikipedia); PAWS-X adds ~49k pairs per language across 6 more languages | filter to label==1 for positives-only; PAWS-X gives multilingual coverage cheaply; cluster/pair sampling method must be recorded per B4 → no obligation propagates; clean-permissive tier |
| `hkust-nlp/SynCSE-scratch-NLI` | PERMISSIVE_OK | V | syncse-scratch | 275,579 rows (sent0, sent1, nli_hard) -- near-identical to the 277,269 pairs `compress` trains on today | ingest as-is; exact drop-in shape for the current anchor/positive/hard-negative regime → MIT imposes nothing -- clean-permissive tier, subject to the model-output-terms decision below |
| `nyu-mll/multi_nli` | PERMISSIVE_OK | **C** | multinli | 392,702 rows (the survey's recommended government+fiction subset would have been ~51,567) | none needed -- the genre-filter step the survey recommended is unnecessary. Re-derive from the ORIGINAL (not the flattened all-nli mirror) so the `genre` column survives for B5 stratification. → 9 of 10 genres emit into the clean-permissive tier; the Fictio… |
| `BeIR/arguana` | ATTRIBUTION | **C** | args-me | not probed | none proposed → CC BY 4.0 attribution only; no share-alike propagation |
| `BeIR/scidocs` | ATTRIBUTION | V | s2orc | ~30k papers across 7 sub-tasks (not row-verified) | better as an eval addition than a training source → CC BY 4.0 attribution propagates |
| `BeIR/scifact` | ATTRIBUTION | V | scifact + s2orc | 1,409 claims; corpus of ~5,183 abstracts | none proposed; the ODC-By corpus half needs its own attribution manifest if redistributed → split emission: the claims half is CC BY 4.0 (attribution tier), the abstract corpus is ODC-By 1.0 which is a DATABASE-LEVEL COPYLEFT (§1.2b) and can only be conveye… |
| `BeIR/webis-touche2020` | ATTRIBUTION | **C** | args-me | not probed | none proposed → CC BY 4.0 attribution only |
| `community-datasets/tapaco` | ATTRIBUTION | V | tatoeba | paraphrase clusters across 73 languages; English subset in the low hundreds of thousands of pairs | cluster-to-pair expansion -- TaPaCo ships paraphrase CLUSTERS, so pairing requires a sampling strategy, which is a B4 method-and-seed decision, not only a licence one → CC BY attribution propagates into the release manifest; attribution tier |
| `BeIR/cqadupstack` | SHARE_ALIKE | V | stackexchange | 12 sub-forums; not row-probed | NONE ADMISSIBLE TODAY: Stack Exchange's attribution obligation is PER-ITEM (a live hyperlink to each question and each author profile), which a dataset-level manifest cannot satisfy at any scale → R9 REFUSE at ingest per 20-enrichment §1.3/§4.5 -- an unsati… |
| `BeIR/dbpedia-entity` | SHARE_ALIKE | V | wikipedia | not probed | none proposed → CC BY-SA propagates; emittable at 4.0 per BY-SA 3.0 §4(b)(ii) |
| `BeIR/fever` | SHARE_ALIKE | V | wikipedia | not probed | none proposed → CC BY-SA propagates; may be emitted uniformly at 4.0 alongside 4.0 inputs per BY-SA 3.0 §4(b)(ii) |
| `castorini/mr-tydi` | SHARE_ALIKE | V | wikipedia | 11 typologically diverse languages; smaller than MIRACL | admit only for languages MIRACL does not cover, to avoid redundant volume → CC BY-SA propagates from the Wikipedia passages |
| `hotpotqa/hotpot_qa` | SHARE_ALIKE | V | wikipedia | ~113k QA pairs; corpus ~5.2M paragraphs (961 MB dir shared with squad) | use the full corpus+queries+qrels pool instead of the saturated 512-pair diagonal → CC BY-SA 4.0 propagates; emits into the share-alike tier |
| `miracl/miracl` | SHARE_ALIKE | V | wikipedia | 18 languages; the largest multilingual BEIR-style retrieval set available | none required -- already query/passage/qrels shaped; language is a natural B5 stratum key → CC BY-SA propagates from the Wikipedia passages; share-alike tier |
| `rajpurkar/squad` | SHARE_ALIKE | V | wikipedia | already on disk in the 961 MB retrieve dir | n/a -- already trained on → CC BY-SA 4.0 propagates |
| `sentence-transformers/natural-questions` | SHARE_ALIKE | **C** | wikipedia | 307,373 training examples (paper-reported; not re-verified) | hard-negative mining against the Wikipedia passage pool → under the corrected reading, CC BY-SA propagates to the emitted pair set (CC BY-SA 4.0 §4(b) makes the enriched database Adapted Material expressly), so it emits into the share-alike tier -- NOT the … |
| `stanfordnlp/snli` | SHARE_ALIKE | V | snli | 550,152 pairs (train+dev+test); 100% already inside the all-nli mirror the region trains on | subtractive: drop non-SNLI rows from the flattened all-nli mirror to restore attributability → CC BY-SA 4.0 propagates to any derived pair set that keeps SNLI rows; share-alike tier only |
| `allenai/gooaq` | NC | V | gooaq | 3.01M question-answer pairs | none -- it already saturates the region at 77.8-79.1% post-dedup share → bespoke NC (not CC BY-NC): no adapter's-licence machinery, no defined 'NonCommercial', no compatible-licence list. Emitted rows stay private per DEC-31 Rider 2. |
| `BeIR/fiqa` | NC | V | fiqa | 57,638-passage pool, ~6,648 question-answer pairs | use the full 57,638-passage pool as `memory`'s eval gate, as 00-ground.md already earmarks → NC; eval-time NC use is uncontroversial, training use rides the NC tier `memory` already occupies |
| `facebook/anli` | NC | V | anli | ~162,865 pairs across R1/R2/R3 | none structural; use as a hard-negative tail → NC tier; costs nothing marginal while `memory` is already NC via GooAQ |
| `GEM/opusparcus` | NC | V | opensubtitles | six languages; English subset in the hundreds of thousands of pairs | standard cap/sample only → NC tier |
| `microsoft/ms_marco` | NC | V | ms-marco | ~8.8M passages, ~1M queries | hard-negative mining (BM25 + dense), standard in the field → NC already caps the tier; mining adds nothing. Emits into the NC tier file only. |
| `RobZamp/sick (SICK)` | NC | V | sick | 9,840 sentence pairs (relatedness 1-5 + entailment label) | none -- ready to use as-is → NC AND SA both attach; the composed model inherits both. Emits into the NC-SA tier, which cannot merge with any CC BY-SA or ODC-By file. |
| `allenai/qasper` | UNVERIFIED | V | s2orc | not probed | n/a until an independent primary is read → n/a; if it is S2ORC-lineage the corpus half would be ODC-By, a separate tier from a CC BY claims half |
| `BeIR/climate-fever` | UNVERIFIED | *U* | wikipedia | not probed | n/a pending a primary read → n/a |
| `BeIR/quora` | UNVERIFIED | *U* | quora-qqp | not probed | n/a pending a primary licence read → n/a |
| `ltg/en-wiki-paraphrased` | UNVERIFIED | V | wikipedia | 5,145,408 rows -- the largest permissive-tagged paraphrase set found | two options: (a) drop the `original` column and use paraphrase-to-paraphrase rewrites only, or (b) accept CC BY-SA propagation from `original` and treat the whole set as share-alike → option (a) may reach the clean-permissive tier; option (b) emits into the… |
| `mteb/sts12-sts .. mteb/sts17-crosslingual-sts, mteb/biosses-sts` | UNVERIFIED | *U* | semeval-sts | not probed | n/a pending one primary read that clears the block → n/a |
| `mteb/stsbenchmark-sts` | UNVERIFIED | *U* | semeval-sts | 8,628 graded pairs (the standard STS-B split) | none needed -- it is eval-shaped as-is → n/a; do not train on it under a TRAIN_OK-only fetcher until the licence resolves. EVAL-ONLY use is a separate, weaker claim that also needs the primary. |
| `sentence-transformers/wikianswers-duplicates` | UNVERIFIED | *U* | wikianswers-ppdb | ~24M duplicate-question pairs | n/a pending a primary licence read → n/a |
| `tau/scrolls` | UNVERIFIED | *U* | (composite -- decompose) | 7 bundled component datasets | decompose to member datasets and give each its own verdict; do not treat the bundle as a provenance group → n/a as a bundle |

**Refused / blocking in `memory`**

| dataset | verdict | verif | why refused |
|---|---|---|---|
| `abisee/cnn_dailymail` | BLOCKING | R | a code licence being read as a data licence -- the same shape as CSS10/Libri-Light in the audio audit — Wrong-layer trap. Would need a genuine grant from CNN/DailyMail. |
| `EdinburghNLP/xsum` | BLOCKING | R | The whole corpus is one restricted-source scrape -- the same shape as `visual`'s tiny-imagenet. No partial repair exists. |
| `alexfabbri/multi_news` | REFUSE | R | explicit research/educational-only language, a REFUSE class the NC-tolerant policy does not absorb — An R4 refusal distinct from ordinary NC. Not absorbed by DEC-31. |
| `BeIR/nfcorpus` | REFUSE | **C** | research-only, permission-gated, no redistribution or training grant — A clean R4 refusal. The surveyor's refusal to trust the BeIR blanket tag was right and the primary is worse than hoped. |
| `BeIR/trec-covid` | REFUSE | R | genuinely heterogeneous per-article licensing; a blanket tag silently mislabels a real mix of NC/custom/commercial tiers; 'text and data mining only' is narrower than a training grant in some readings — Refused as a single corpus, not as a class of content. The refusal is about the blanket ingest… |
| `facebookresearch/ELI5` | REFUSE | V | Reddit-scraped Q&A, no consent process described; Reddit's API terms restrict bulk redistribution of user content independently of the scraper's code licence — A wrong-layer trap: the code licence does not even attempt to cover the content. Recommend REFUSE for now; revisit only with a Reddit-con… |
| `mandarjoshi/trivia_qa` | REFUSE | R | the operator's named REFUSE class verbatim: a distributor disclaiming ownership of what it distributes, with no licence grant offered at all — A textbook R3 refusal. Not a judgment call the way MS MARCO's softer disclaimer is. |

#### `reasoning` — candidates (18 listed, 1 refused/blocking below)

| dataset | verdict | verif | provenance group | size | enrichment → licence result |
|---|---|---|---|---|---|
| `allenai/math_qa` | PERMISSIVE_OK | V | aqua-rat | 37,297 problems | prefer swapping MathQA IN FOR raw aqua_rat rather than adding both → Apache-2.0; clean-permissive tier |
| `EleutherAI/hendrycks_math` | PERMISSIVE_OK | V | hendrycks-math | 12,500 problems (7,500 train / 5,000 test), competition-level with full worked solutions | none structurally; de-dup against any eval split reused elsewhere in the fleet before admission (B3) → MIT imposes nothing; clean-permissive tier |
| `facebookarchive/bAbI-tasks` | PERMISSIVE_OK | V | babi | 20 task categories, toy-scale | generate at controlled task-type caps; task type is the B5 stratum key → BSD-3; clean-permissive tier |
| `hoskinson-center/proofnet` | PERMISSIVE_OK | V | proofnet | 371 informal-formal problem pairs from undergraduate pure-maths textbooks | pair informal statement with a CoT-style informal proof as a rationale target → MIT; clean-permissive tier |
| `maveriq/bigbenchhard` | PERMISSIVE_OK · EVAL-ONLY | **C** | big-bench | 23 curated hard tasks | none -- held-out eval only → Apache-2.0; notice retention only |
| `open-r1/OpenR1-Math-220k` | PERMISSIVE_OK | V | numina-aops | ~220K verified-correct traces (94k default + 131k extended, filtered from 400k generated) | none needed → Apache-2.0; clean-permissive tier |
| `openai/prm800k` | PERMISSIVE_OK | V | openai-prm | ~800K step-level human correctness labels over MATH-problem solution steps | reshape step labels into (problem, step, label) triples → MIT; stays clean-permissive downstream |
| `wics/strategy-qa` | PERMISSIVE_OK | V | strategyqa | 2,780 examples with decomposition + implicit multi-hop steps + supporting Wikipedia paragraphs | render decomposition steps as CoT-style rationale text → MIT; clean-permissive tier |
| `GDELT Project` | ATTRIBUTION | V | gdelt | large-scale structured event records | needs a translation layer to become model-trainable text or latent pairs; it is derived features, not source text → attribution + link obligation propagates into the release manifest |
| `open-web-math/open-web-math` | ATTRIBUTION | **C** | open-web-math | ~6.3M documents / ~14.7B tokens -- by far the largest reasoning-adjacent candidate | mandatory strided/reservoir cap (B4, never a prefix); QA-pair extraction from raw documents if used for `reasoning` rather than `numeric/math` pretraining → ODC-By is a DATABASE-LEVEL COPYLEFT (§1.2b, §4.2(a)): a filtered subset is a Derivative Database and… |
| `nvidia/Nemotron-Math-Proofs-v1` | SHARE_ALIKE | V | nvidia-nemotron-math | not probed | none proposed; a candidate for a dedicated pass → CC BY-SA 4.0 propagates; share-alike tier |
| `camel-ai/math` | NC | V | camel-gpt4 | ~50K GPT-4-generated math problems with worked solutions | automated correctness filter (execute/verify against a symbolic solver) before training use → NC propagates to any enriched derivative under strictest-input; NC tier only |
| `meta-math/MetaMathQA` | NC | **C** | metamath | not probed | n/a pending the model-output-terms decision → held at NC pending an explicit operator call on whether 'asserts MIT over GPT-3.5-derived text' clears the bar |
| `microsoft/orca-math-word-problems-200k` | NC | V | orca-math | 200k word problems | n/a pending the model-output-terms decision → held at NC on the same reasoning as MetaMathQA |
| `TIGER-Lab/MathInstruct` | NC | V | (composite -- decompose) | not probed | admit the components directly with per-source bookkeeping rather than the aggregate; excluding camel_math makes the remainder PERMISSIVE_OK → NC-if-whole under strictest-input, regardless of the aggregate's MIT tag; PERMISSIVE_OK-if-camel_math-excluded |
| `AI-MO/NuminaMath-CoT` | UNVERIFIED | V | numina-aops | not row-probed as a whole; aops_forum subset is 30,201 rows | admit the constituent subsets individually with correct per-source bookkeeping rather than the aggregate → Apache-2.0 is a real grant from a legitimate distributor, but the aops_forum fraction's chain is open; the orca_math fraction is a model-output-terms … |
| `EleutherAI/proof-pile-2` | UNVERIFIED | V | (composite -- decompose) | not probed | source OpenWebMath directly instead of through this composite → n/a as a blanket set |
| `tasksource/proofwriter` | UNVERIFIED | *U* | proofwriter | not probed | n/a -- refused-for-now under the REFUSE-by-default posture for 'no licence grant exists' → n/a |

**Refused / blocking in `reasoning`**

| dataset | verdict | verif | why refused |
|---|---|---|---|
| `AMAImedia/NOESIS-* (multi-provider synthetic aggregates)` | REFUSE | R | outputs blended from at least five proprietary/hosted model APIs (Claude Opus, DeepSeek, Qwen, Gemini, GPT); blending compounds rather than resolves the redistribution question — No verdict short of REFUSE is defensible without a per-provider ToS review. A class, not a single dataset. |

#### `numeric_math` — candidates (4 listed, 0 refused/blocking below)

| dataset | verdict | verif | provenance group | size | enrichment → licence result |
|---|---|---|---|---|---|
| `google-deepmind/mathematics_dataset` | PERMISSIVE_OK | V | deepmind-mathematics | a GENERATOR, not a fixed corpus -- arbitrary volume of procedurally generated problems with exact answers | run the generator at controlled category caps; category is the B5 stratum key; the sampling method and seed go in the receipt (B4) → Apache-2.0; the emitted rows are the operator's own and impose nothing. NOTE: this is generated from a TEMPLATE the operator… |
| `hendrycks/math (MATH, direct from GitHub)` | PERMISSIVE_OK | V | hendrycks-math | 12,500 problems | pair with the DeepMind generator for source diversity → MIT; clean-permissive tier |
| `Wikidata` | PERMISSIVE_OK | V | wikidata | 100M+ items | values-not-BPE-pieces extraction to match `numeric`'s design brief; pull via dumps.wikimedia.org, never a live scrape → CC0 imposes nothing on the emitted dataset or the weights; the CC BY-SA prose/lexeme-gloss layer is a SEPARATE tier and must not be merge… |
| `OpenML` | UNVERIFIED | *U* | (source class -- per-dataset) | n/a -- a platform | per-dataset licence capture at pull time; never a blanket admit → n/a at the aggregate level |

#### `visual` — candidates (9 listed, 12 refused/blocking below)

| dataset | verdict | verif | provenance group | size | enrichment → licence result |
|---|---|---|---|---|---|
| `EuroSAT` | PERMISSIVE_OK | V | eurosat | 27,000 labelled Sentinel-2 RGB images, 10 land-use classes | the JEPA-style patch-token pipeline already exists; run a class-balance check against B5 → MIT; clean-permissive tier |
| `Replacement-vision composite (pxhere, PatchCamelyon, Shapes3D, CLEVR, Fashion-MNIST, EuroSAT-rgb, Quick Draw, Caltech-101/256)` | PERMISSIVE_OK | V | (multi-group composite) | ~497k images | pair with EuroSAT and a Commons pull; the caption layer is what the Commons+SAFE-captioner path is for → clean-permissive tier; attribution manifests for the CC BY members |
| `Wikimedia Commons (direct, licence-filtered pull)` | PERMISSIVE_OK | **C** | wikimedia-commons | >100M media files total; a photographic, CC0/CC-BY-only, no-NC subset must be built and its size is TBD by the filter | pull a CC0/CC-BY-filtered image subset via the Commons API with the licence read and recorded PER FILE in the receipt, then generate captions with a SAFE-class (Apache-2.0/MIT, locally run) captioner -- converting a pure image source into caption pairs with… |
| `derek-thomas/ScienceQA` | UNVERIFIED | V | sciqa-curriculum | 21,208 questions, ~10,332 with an image | n/a until the constituent-source question is checked → SHARE_ALIKE if the mirror tag holds at the constituent level; UNVERIFIED today |
| `DocVQA (task 1)` | UNVERIFIED | V | ucsf-industry-documents | 12,767 images / 50,000 questions | none until the portal terms are read by someone with an account → if the 'evaluation license' phrasing governs, this is the same restrictive shape already REFUSE'd for SA-1B |
| `lmms-lab/ai2d (AI2D)` | UNVERIFIED | V | ai2-diagram | 4,903 diagrams / ~15,000 multiple-choice questions | n/a until a grant is read → n/a |
| `Maluuba/FigureQA` | UNVERIFIED | **C** | synthetic-chart-render | ~180,000 synthetic figures (100k train + val/test), ~1.3M QA pairs | none needed if the MSR data-download terms clear; otherwise re-render equivalents with the MIT generator code, which the operator may run outright → if the MSR terms clear, clean-permissive tier; if they do not, the MIT generator code is itself a usable pat… |
| `NiteshMethani/PlotQA` | UNVERIFIED | **C** | synthetic-chart-render | ~224,000 charts / 28M QA pairs -- by far the largest chart-QA set found | none needed if the discrepancy resolves; the underlying data values may carry their own source licence (World Bank Open Data is itself CC BY 4.0, a favourable sign, unconfirmed) → MIT would emit clean-permissive; CC BY 4.0 would emit into the attribution ti… |
| `wikimedia/wit_base` | UNVERIFIED | V | wikimedia-commons | ~37.6M image-text pairs across 108 languages; English subset alone ~5.5M | language filtering to the fleet's target set, then a per-file licence-stratification pass: keep CC0/CC BY for the permissive tier, quarantine CC BY-SA → the dataset-level CC BY-SA 4.0 tag cannot be trusted as the per-image FLOOR without the audit; if the au… |

**Refused / blocking in `visual`**

| dataset | verdict | verif | why refused |
|---|---|---|---|
| `ChartQA-derivative cluster (ChartBench, ChartDQA, ChartQAPro, Chartographer)` | BLOCKING | V | a derivative's own permissive-looking tag EXPRESSLY qualified as conditional on unverified source terms — A primary-source confession of the exact defect the survey inferred. Any member that generates genuinely NEW chart images would need its own per-repo check; none was confirmed. |
| `COCO downstream tree (VQAv2, OK-VQA, A-OKVQA, GQA, NoCaps, LocalizedNarratives, TextVQA)` | BLOCKING | V | One defect, seven datasets. Grouped as one row because they share one image pool and one verdict. |
| `DTD (Describable Textures Dataset)` | BLOCKING | **C** | no licence of any kind on any surface -- not even the mirror's ambiguous 'other' — The 'other' tag was the reason to check, not a green light, and checking made it worse. The surveyor's caution was right. |
| `Fhrozen/sbucaptions (SBU Captions)` | BLOCKING | *U* | Same per-photographer Flickr shape as COCO and Flickr30k. Verdict is inherited, not evidenced. |
| `HuggingFaceM4/COCO (and COCO-caption mirrors)` | BLOCKING | V | the annotation licence is real and clean; the pixels are individually-rights Flickr photographs the Consortium expressly does not own — Would be the single highest-value image-caption source if clean; it is not. Its entire downstream tree (VQAv2, OK-VQA, A-OKVQA, GQA, NoCaps, Localized Narratives… |
| `InfographicVQA` | BLOCKING | *U* | images are web-scraped commercially-designed marketing infographics, a WORSE provenance shape than DocVQA's public document archive, even behind the same portal terms — Kept as a separate provenance group from DocVQA deliberately -- same portal, different image source. Not independently re-derive… |
| `nlphuji/flickr30k` | BLOCKING | V | Already catalogued BLOCKING in 00-ground.md and reconfirmed here. Also the image pool under SNLI's captions and part of LocalizedNarratives. |
| `ranjaykrishna/visual_genome` | BLOCKING | V | A bigger loss than COCO alone because of the annotation density. Inherits both the COCO and the YFCC100M per-photographer problems. |
| `vis-nlp/ChartQA` | BLOCKING | V | WRONG-LAYER TRAP: a real, verified licence that covers the crawler and QA-generation code, not the crawled media — One repo-fetch away from a possible partial-source clearance if some contributing sources turn out CC-licensed. Its derivative cluster inherits the defect. |
| `google-research-datasets/conceptual_captions (CC3M) and CC12M` | REFUSE | V | the clean-looking Google grant covers the CSV Google compiled; it is silent on, and cannot grant, rights over the images at those URLs — Structurally identical to LAION, the named negative example. A clean licence over the wrong artefact. |
| `LAION-2B / LAION-400M` | REFUSE | R | URL-only, no image rights conveyed of any kind. Same class as Conceptual Captions. |
| `Segment Anything / SA-1B` | REFUSE | V | a commercial photo licence Meta itself holds, which does not transfer to third-party redistribution or training — A textbook R4 refusal on explicit research-only terms. No redistribution or training grant to third parties beyond research use. |

#### `language_trunk` — candidates (21 listed, 5 refused/blocking below)

| dataset | verdict | verif | provenance group | size | enrichment → licence result |
|---|---|---|---|---|---|
| `Anthropic/hh-rlhf` | PERMISSIVE_OK | V | anthropic-hh-rlhf | ~161K-169K comparison pairs (helpful + harmless splits, plus red-team transcripts) | none required → MIT propagates nothing; clean-permissive tier |
| `data.gov (US federal)` | PERMISSIVE_OK | V | (source class -- per-dataset) | n/a -- an index | per-resource licence capture at pull time → PD; nothing propagates for confirmed-federal-authored content |
| `HuggingFaceH4/ultrachat_200k` | PERMISSIVE_OK | V | ultrachat | 1.5M conversations (full UltraChat) / 200K filtered high-quality subset (the HF mirror) | none required if MIT is accepted as-is → MIT propagates nothing; clean-permissive tier -- SUBJECT to the model-output-terms decision, which should be made as ONE decision covering UltraChat, Alpaca, KodCode, camel-ai/math, MetaMathQA, orca-math, SynCSE-scra… |
| `HuggingFaceTB/cosmopedia` | PERMISSIVE_OK | V | cosmopedia-synthetic | 25-30B tokens | none required → Apache-2.0 propagates cleanly; clean-permissive tier. No generator terms attach because the generator's weights are Apache-2.0. |
| `OpenAssistant/oasst2` | PERMISSIVE_OK | V | openassistant | ~135K messages, 10K+ conversation trees, 35 languages | none required → Apache-2.0 propagates cleanly; clean-permissive tier, and NO model-output-terms question attaches |
| `PleIAs/common_corpus` | PERMISSIVE_OK | V | (composite: gutenberg + wikipedia + gov + science + semantic) | ~2T tokens total across six strata | admit the OpenGovernment / OpenScience / OpenSemantic strata ONLY -- the OpenCulture and OpenWeb strata double-count the Gutenberg and Wikipedia groups already admitted separately. Dedup against those before B1/B2 is computed, not optional. → mixed BY CONST… |
| `Project Gutenberg (manu/project_gutenberg, storytracer/LoC-PD-Books)` | PERMISSIVE_OK | V | gutenberg | manu/project_gutenberg ~10K-100K rows (curated clean subset); LoC-PD-Books is a larger multi-source PD book set, partially overlapping | strip the Project Gutenberg branding from headers during ingestion, leaving zero remaining obligation; de-dup LoC-PD-Books against Gutenberg IDs before combining → public domain; nothing propagates. Clean-permissive tier. |
| `allenai/c4` | ATTRIBUTION | V | common-crawl | ~365M documents (en config) / ~305 GB | none required → ODC-By propagates -- own tier file |
| `allenai/dolma` | ATTRIBUTION | **C** | common-crawl | ~3T tokens, ~11B docs | none for base use; spot-check and, if necessary, drop the PushShift-Reddit stratum before scale-up → ODC-By propagates as above |
| `allenai/peS2o` | ATTRIBUTION | V | s2orc | ~38.97M documents / ~50B tokens | none for base use → ODC-By propagates -- own tier file |
| `EU Open Data Portal (data.europa.eu)` | ATTRIBUTION | V | (source class -- per-dataset) | n/a -- an index | per-resource licence capture at pull time → CC BY attribution for EU editorial content; CC0 for portal metadata; per-resource for everything else |
| `HuggingFaceFW/fineweb-edu` | ATTRIBUTION | V | common-crawl | 1.3T tokens | none for base LM pretraining; strided cap per B4 if a share limit binds → ODC-By propagates: a filtered subset is a Derivative Database conveyable ONLY under ODC-By, so it emits into its own `-odcby` tier file and cannot merge with CC BY, CC BY-SA or NC files |
| `databricks/databricks-dolly-15k` | SHARE_ALIKE | **C** | dolly | 15,011 records | none required → CC BY-SA propagates to any derived instruction set that substantially reuses Dolly's text. If the version really is 3.0, §4(b)(ii) permits emitting under 4.0 alongside 4.0 inputs; if it is 4.0 already, the same holds trivially. |
| `Stack Exchange data dump (HuggingFaceH4/stack-exchange-preferences and the raw dump)` | SHARE_ALIKE | V | stackexchange | hundreds of GB across all SE sites; the HF preference-pairs mirror is a filtered subset (10M<n<100M rows) | NONE ADMISSIBLE TODAY -- see the licence result → R9 REFUSE AT INGEST: the attribution obligation is PER-ITEM (a live outbound hyperlink per answer and per author profile) and a dataset-level manifest cannot satisfy it at any scale. 20-enrichment §1.3/§4.5 … |
| `wikimedia/wikipedia` | SHARE_ALIKE | V | wikipedia | ~6.8M English articles (20231101.en config) | attribution manifest with the article-history hyperlink or author list per the ToU → CC BY-SA 4.0 propagates to any derivative built substantially FROM Wikipedia text (e.g. QA-pair extraction) -- expressly, per CC BY-SA 4.0 §4(b). Whether it reaches the tra… |
| `allenai/tulu-3-sft-mixture` | UNVERIFIED | **C** | (composite -- decompose) | ~939K examples | per-subset split before whole-corpus admission -- not a spot-check → the blanket ODC-By tag cannot be the operative licence for subsets the card itself says are NC or third-party-model output; splitting produces at least a permissive tier, an NC tier and a … |
| `CourtListener bulk data (Free Law Project)` | UNVERIFIED | **C** | courtlistener | 10M+ opinions | re-open with a fetch of courtlistener.com/terms/ from a different egress, or ask the operator to paste the terms page → if the Public Domain Mark governs the data, this emits into the clean-permissive tier and is a large independent provenance group for the… |
| `nvidia/Nemotron-CC-v2` | UNVERIFIED | **C** | common-crawl | multi-trillion tokens across quality tiers plus a synthetic tier | if admitted, the synthetic tier must be separable from the non-synthetic tiers so the Qwen obligations attach to a bounded share → ENCUMBERED-generator flow-through (SYN-L2 / R10): admitting the synthetic tier obliges the released model to carry a 'Built wi… |
| `Papers with Code dataset index` | UNVERIFIED | *U* | (index only) | n/a | every hit through it still needs the same primary-source licence read → n/a -- record as methodology, not as an admissible source |
| `PubMed Central Open Access subset` | UNVERIFIED | V | pmc-oa | millions of full-text biomedical articles; the CC0/CC BY/CC BY-SA slice is the safely usable fraction | MECHANICAL per-article licence filter is REQUIRED before any admission: keep CC0/CC BY/CC BY-SA, filter OUT every ND slice, tag the NC slices NC. Each licence bucket is its own provenance stratum for B1-B5, not one dataset. → three separate emitted tier fil… |
| `togethercomputer/RedPajama-Data-V2` | UNVERIFIED | V | common-crawl | ~30T tokens across 5 quality-signal tiers | quality-signal-based filtering to FineWeb-Edu-like quality → no HF-native grant exists to propagate; the emitted subset would rest entirely on the Common Crawl ToU, which is an access agreement, not a copyright licence |

**Refused / blocking in `language_trunk`**

| dataset | verdict | verif | why refused |
|---|---|---|---|
| `arXiv full-text` | REFUSE | V | ND options present in the full-text layer — The metadata layer is a smaller, separately-assessable, clean candidate. Do NOT conflate the two layers. |
| `EleutherAI/the_pile_deduplicated` | REFUSE | R | Books3 and other sub-sources carry well-documented, unresolved rights disputes -- plausibly the operator's 'distributors that disclaim owning what they distribute' refusal class — Refused wholesale, not forever. Would decompose into several already-listed groups if split. |
| `lmsys/lmsys-chat-1m` | REFUSE | V | explicit non-redistribution clause; raw users' conversations collected via a public chat demo, not an explicit research-consent flow -- a live-consent shape closer to Common Voice's than to a standard scrape — Refused on terms AND on consent provenance, which are separate grounds. Deliberately no… |
| `Muennighoff/flan (FLAN collection)` | REFUSE | R | some constituent tasks (e.g. certain WMT/translation sets) carry mixed terms that a blanket admit would silently import — High-value IF the per-task audit is done; refused wholesale until then. Internally multi-source, which is good for B2 but means the internal composition needs its own audit. |
| `tatsu-lab/alpaca` | REFUSE | **C** | research-only field-of-use clause hidden in a README beneath a CC-BY-NC badge; MODEL-OUTPUT-TERMS class: generated via OpenAI text-davinci-003 by Self-Instruct — The clearest case in the catalogue of a README narrowing a licence badge. Was never in the survey's top 8; move it out of the live-NC-c… |

#### `moral_safety` — candidates (21 listed, 0 refused/blocking below)

| dataset | verdict | verif | provenance group | size | enrichment → licence result |
|---|---|---|---|---|---|
| `Anthropic/hh-rlhf (+ red-team-attempts)` | PERMISSIVE_OK | V | anthropic-hh-rlhf | 169,352 rows (helpfulness + harmlessness preference pairs, plus red-team transcripts) | none needed structurally; could re-rank preference pairs against the ETHICS framework labels for cross-consistency checks → MIT stays MIT under any enrichment; clean-permissive tier |
| `demelin/moral_stories` | PERMISSIVE_OK | V | moral-stories | 720,000 rows | minimal -- already close to a ready-made contrastive (moral vs immoral action + consequence) training item → MIT propagates nothing; clean-permissive tier |
| `google/civil_comments` | PERMISSIVE_OK | V | civil-comments | 1,999,514 rows (comment text + toxicity + identity-mention multi-labels) | none needed; best used as a HELD-OUT PROBE for toxicity sensitivity before and after moral-safety training, rather than as primary training material → CC0 imposes no obligation on any derivative; clean-permissive tier |
| `google/jigsaw_toxicity_pred` | PERMISSIVE_OK · EVAL-ONLY | V | jigsaw-toxic-comment | 319,301 rows for the config queried (multiple configs exist) | another held-out probe candidate; its taxonomy differs from Civil Comments', which makes it a useful cross-taxonomy robustness check → CC0 imposes nothing; clean-permissive tier |
| `Hate-speech-CNERG/hatexplain` | PERMISSIVE_OK | **C** | hatexplain | not sized in either pass | use its rationale-span format as the model for rationale enrichment elsewhere in the corpus → MIT: notice retention only; clean-permissive tier |
| `hendrycks/ethics` | PERMISSIVE_OK | V | hendrycks-ethics | 134,417 rows across 5 subscales (justice, deontology, virtue, utilitarianism, commonsense) | pair each scenario with a natural-language rationale (the set is label-only today) using a SAFE-class generator; RESERVE a held-out split as a measurement probe rather than training on the whole set → MIT propagates regardless of derivative annotation; clea… |
| `metaeval/scruples` | PERMISSIVE_OK | V | scruples | 32,766 rows in the queried config; the full release also carries an 'anecdotes' split of ~30k+ narrative stories with community-voted verdicts | pair anecdote text with the community verdict DISTRIBUTION as a soft label → Apache-2.0 permits any derivative with notice retention; clean-permissive tier |
| `mmathys/openai-moderation-api-evaluation` | PERMISSIVE_OK · EVAL-ONLY | V | openai-moderation-eval | 1,680 rows | held-out probe with a taxonomy complementary to BeaverTails' and WildGuard's → MIT; nothing propagates |
| `allenai/social_bias_frames` | ATTRIBUTION | *U* | social-bias-frames | not sized in either pass | pairs well with Social Chemistry 101 as a second norms/frames source, raising N_eff if both are admitted to one region → CC BY 4.0 attribution propagates; TASL notice required on release |
| `allenai/wildguardmix` | ATTRIBUTION | V | allenai-wildguard | ~92K items (card prose; not independently row-counted) | none required for format → ODC-By propagates: the enriched set is a Derivative Database conveyable ONLY under ODC-By, so it emits into its own tier file -- it cannot be merged with the NC file (PKU) or the share-alike file (Social Chemistry). The §4.3 examp… |
| `allenai/wildjailbreak` | ATTRIBUTION | V | allenai-wildguard | ~262K items (AI2's published figure; INFERRED from card prose, not row-counted) | none required for format → same ODC-By tier file as wildguardmix |
| `kellycyy/daily_dilemmas` | ATTRIBUTION | *U* | daily-dilemmas | not sized in either pass | review the values-taxonomy annotation format before ingest → CC BY attribution if the tag holds; provisional |
| `ninoscherrer/moralchoice` | ATTRIBUTION · EVAL-ONLY | *U* | moralchoice | 3 rows in the default config (a scenario-template config); the full benchmark is documented as ~1,700+ moral scenarios across configs (INFERRED from the paper) | keep as the primary held-out probe; it was built exactly for the measurement the operator asks for → CC BY attribution; probe use only |
| `McGill-NLP/stereoset` | SHARE_ALIKE · EVAL-ONLY | V | stereoset | 4,229 rows | KEEP AS A PROBE -- using an eval-shaped benchmark as training data destroys its value as a measurement instrument → SA would propagate to any derivative anyway; as a probe it never enters a training file |
| `nyu-mll/crows_pairs` | SHARE_ALIKE · EVAL-ONLY | V | crows-pairs | ~1,500 sentence pairs (published figure; not row-counted) | keep as a probe → SA; probe only |
| `wassname/social_chemistry_101` | SHARE_ALIKE | V | social-chemistry-101 | 355,922 rows (rules-of-thumb over everyday situations with social-judgment dimensions: legality, agreement, cultural pressure) | minimal -- mostly re-formatting into the region's training shape → SA PROPAGATES: any derived or enriched set built from this corpus (rendering, pairing, rephrasing) must stay CC BY-SA and emits into the share-alike tier, which cannot merge with the moral c… |
| `PKU-Alignment/BeaverTails` | NC | **C** | pku-alignment | 364,170 rows (prompt + response + multi-category harm annotation + safety label) | category labels could seed taxonomy-conditioned rationale generation with a SAFE-class generator → NC propagates; same NC tier file as PKU-SafeRLHF |
| `PKU-Alignment/PKU-SafeRLHF` | NC | V | pku-alignment | 164,236 rows (helpfulness + harmlessness dual-preference comparisons with severity tiers) | none needed for format → NC propagates; emits into the moral corpus's NC tier file, which can never merge with the Social Chemistry share-alike file or the WildGuard ODC-By file |
| `toxigen/toxigen-data` | UNVERIFIED | V | microsoft-toxigen | 319,301 rows (LLM-generated implicit-hate-speech statements, human-annotated) | n/a pending resolution → IF the CDLA grant governs, this is the BEST-CASE family in the whole catalogue: CDLA-Permissive-2.0 §3.1 expressly states it imposes no restriction on Results, and §5.4 defines Results to INCLUDE machine learning models -- the only … |
| `ucberkeley-dlab/measuring-hate-speech` | UNVERIFIED · EVAL-ONLY | **C** | measuring-hate-speech | 135,556 rows | n/a pending the grant-scope answer; the continuous score is the best calibration instrument in the survey and belongs in the probe set either way → if the grant covers annotations only, the post text cannot be redistributed and only the scores are usable --… |
| `walledai/TDC23-RedTeaming` | UNVERIFIED | *U* | tdc23-redteaming | not sized in either pass | n/a pending a data-specific grant that may no longer be obtainable → n/a |
---

## 7. Enrichment, and what each operation costs in licence terms

`20-enrichment-licence-impact.md` is the full analysis; a condensed version of its matrix and
mechanics is now a section of `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md` ("Enriched and derived
datasets"). Three results drive every `enrichment_licence_result` field in the catalogue:

1. **Filtering alone is a derivative-database event.** The cheapest operation the factory can
   perform — dropping rows — already triggers CC 4.0 §4(b), ODbL §4.4(b), ODC-By §4.2(a) and
   CDLA-Sharing §1.8. Enrichment is not a licence-neutral cleanup step, and there is no
   creativity threshold to argue about.
2. **The dataset layer is decidable; the weights layer is not.** Whether trained weights are
   "Adapted Material" is unsettled and nothing here closes it. Compliance must therefore be
   engineered at Layer 1, where express clauses answer the question, and the factory must not
   depend on the Layer 2 answer in either direction.
3. **Two obligations cannot be discharged by a manifest and are ingest-time refusals**:
   per-item attribution (the Stack Exchange shape — a live outbound hyperlink per answer and per
   author profile, unsatisfiable at any scale, `R9`) and ODbL §4.6's duty to offer recipients the
   entire enriched dataset once weights are published, which is incompatible with DEC-31 Rider 2's
   private-HF-only policy (`R8`).

**Generator classes for synthetic enrichment** (SYN-L2), which decide whether an enrichment step
attaches terms to the *released model*:

| class | rule | members named in this pass |
|---|---|---|
| **SAFE** | Apache-2.0 or MIT weights, run locally, no hosted-service terms accepted | OLMo-2 (7B, 32B-Instruct), Qwen2.5-7B, Qwen3-8B, Mistral-7B-v0.3, SmolLM2-1.7B |
| **ENCUMBERED** | terms reach the downstream model; usable only with recorded acceptance *before* generating (`R10`) | Gemma (any version — outputs are free but a model trained on them is a *Model Derivative* that must carry the Gemma Terms forward); Llama 3.1+ (forces a `Llama`-prefixed model name, "Built with Llama", and AUP passthrough) |
| **REFUSED** | terms forbid the use, or bind the operator against it | Llama 2 / 3.0; OpenAI; Anthropic; **and any model tagged `other` until its bespoke licence is read** — today `Qwen/Qwen2.5-3B` and `Qwen/Qwen2.5-72B` |

"Qwen is Apache" is false as a family claim — two of four Qwen2.5 sizes checked are `other`.
Classify **per model id and revision**, never per vendor; the default for an unclassified
generator is REFUSED, not ENCUMBERED.

**SYN-L4 — augmentation does not reset provenance.** A rationale generated *from* a CC BY-SA
premise, a paraphrase *of* an NC answer, a translation *of* a CC BY passage: each inherits the
**seed row's** licence class, and carries the generator's terms *in addition* where the generator
is ENCUMBERED. Only rows generated from unlicensed-input prompts — a template the operator wrote,
a CC0 seed — are the operator's alone. This is why the Wikimedia Commons + SAFE-captioner path
(§8, fetch 2) works and why "synthesise our way out of `visual`" does not: **a generator cannot
manufacture a grant that does not exist** (SYN-L3).

---

## 8. Clean-permissive tier vs NC-inclusive tier, against the 1e10-tokens-per-region target

The operator's scale path is ~1B parameters and ~1e10 tokens per region. Two numbers per tier:

- **raw reach** — the admissible tokens available.
- **B1-bounded reach (`T_B1`)** — the largest total that satisfies **B1's max-single-source share
  ≤ 0.40** given the tier's actual provenance-group split. With `L` tokens in the largest group
  and `R` in all the others, the non-largest must supply ≥ 0.60·T, so **`T_B1 = min(L+R, R/0.60)`**.

`T_B1` is the honest number, and it is the one that changes decisions. A tier with **one**
provenance group has `T_B1 = 0` — not "unusable", but "admissible only under B2's waiver clause,
which requires the region to either rename itself to what its corpus actually is or carry a dated
waiver in `csd-regions.json` naming the missing sources."

**All estimates below are INFERRED** under two stated assumptions: **4 bytes per token** for text
measured on disk, and **64 patch tokens per image** for `visual` (matching `JEPAConfig(image_size=64,
patch_size=8)` → `n_patches = 64` as the code stands). Row-count-to-token conversions use a
per-row length appropriate to the shape and are recorded in the reach model, not guessed per row.

| faculty | tier | groups | raw reach | `T_B1` | % of 1e10 |
|---|---|---|---|---|---|
| **`language_code`** | clean-permissive | 3 | 5.8e9 | **1.5e9** | **15%** |
| | OpenRAIL-M (share-alike) | 1 | 2.0e11 | 0 (waiver only) | 0% |
| | NC-inclusive (union of files) | 4 | 2.1e11 | **1.5e9** | **15%** |
| **`memory`** | clean-permissive | 8 | 1.3e8 | **1.3e8** | **1.3%** |
| | share-alike | 2 | 2.0e9 | 3.7e7 | 0.4% |
| | NC (with MS MARCO) | 6 | 7.3e8 | 3.3e8 | 3.3% |
| | NC-inclusive (union of files) | 16 | 2.9e9 | **5.0e8** | **5.0%** |
| | *NC-inclusive if MS MARCO is refused* | 15 | 2.4e9 | **2.0e8** | **2.0%** |
| **`reasoning`** | clean-permissive, no generator | 8 | 5.5e8 | **1.8e8** | **1.8%** |
| | ODC-By (`open-web-math`) | 1 | 1.47e10 | 0 (waiver only) | 0% |
| | NC-inclusive, no generator | 12 | 1.5e10 | **2.6e8** | **2.6%** |
| | **clean-permissive, running the DeepMind generator** | 9 | unbounded | **unbounded** | **≥100%** |
| **`numeric_math`** | clean-permissive (generator + MATH + Wikidata) | 3 | unbounded | **unbounded** | **≥100%** |
| **`visual`** | clean-permissive | 7 | 3.2e7 | **3.2e7** | **0.3%** |
| | NC-inclusive | 7 | 3.2e7 | **3.2e7** | **0.3%** |
| **`language_trunk`** | clean-permissive | 7 | 9.6e11 | **6.4e11** | **6,359%** |
| | ODC-By | 2 | 1.35e12 | 8.3e10 | 833% |
| | NC-inclusive (union of files) | 11 | 2.3e12 | **7.2e11** | **7,192%** |
| **`moral_safety`** | clean-permissive | 7 | 2.4e8 | **2.4e8** | **2.4%** |
| | NC-inclusive (union of files) | 12 | 4.2e8 | **2.4e8** | **2.4%** |

### What this table says

**NC buys almost nothing.** This is the finding most likely to be assumed the other way round.
Compare the clean-permissive and NC-inclusive rows: `language_code` gains **0%** (KodCode is in
the already-dominant PG-COMPETITIVE group, so it adds volume to the group B1 is already capping),
`visual` gains **0%** (there is no NC visual candidate at all), `moral_safety` gains **0%**
(PKU-SafeRLHF and BeaverTails are one group, so their NC file has `T_B1` = 0 alone), `reasoning`
gains **0.8 points**, and `memory` gains **3.7 points** — and 3.3 of those 3.7 are MS MARCO
alone, which is the one entry the operator has to decide on rather than accept (§9).

**The composed model is already NC via GooAQ, so this is not an argument to drop NC.** It is an
argument that *chasing more NC volume is not the lever*. The lever is **more independent
PERMISSIVE_OK and ATTRIBUTION provenance groups**, exactly as `10-survey-compress.md` concluded
for its own faculty. B1, not licence tolerance, is what is binding.

**Three faculties reach the target and four do not.** `language_trunk` reaches it 60× over on
permissive sources alone with B1 and B2 satisfied. `numeric_math` and `reasoning` reach it — but
**only by running the DeepMind `mathematics_dataset` generator**, which is the single source in
the whole catalogue that can supply arbitrary B1-relief volume under a settled Apache-2.0 grant
without a new licence question. Without it, `reasoning` sits at 2.6%.

**`visual` is off by a factor of 300.** Its clean-permissive tier is ~524k images ≈ 3.2e7 patch
tokens. Reaching 1e10 needs ~1.6e8 images at the current 64-token geometry, or ~5.1e7 images if
`visual`'s resolution rebuild (W7v) moves to a 224²/16² geometry at 196 tokens per image. Only
one candidate in the catalogue can plausibly supply that: a licence-filtered Wikimedia Commons
pull. Every other route in the visual survey dead-ends at BLOCKING.

**`memory` is the faculty the merge made worse, not better.** Merging `retrieve` and `compress`
inherits the union of obligations. Post-merge, its clean-permissive tier is 1.3% of target, its
share-alike tier is 99% one provenance group (Wikipedia), and the single largest lever against
GooAQ's concentration — Stack Exchange — is blocked by an attribution obligation the factory
structurally cannot discharge (`R9`), not by its licence class.

---

## 9. Top ten next fetches, in priority order

Each is one fetch or one bounded action, with the balance rules its admission must satisfy.
Priority is by **`T_B1` unlocked per unit of effort**, not by dataset size.

| # | fetch / action | unlocks | balance rules it must satisfy |
|---|---|---|---|
| **1** | The **OpenCoder technical report's data-availability section** (arXiv), for `OpenCoder-LLM/opc-fineweb-code-corpus` | 147.9 GB / 1.0e8 rows — the **only** large code source with provenance independent of PG-STACK, PG-CSN and PG-COMPETITIVE. It is what makes `language_code`'s B1 satisfiable at all. | Admit as its own provenance group. Cap at ≤ 0.40 of the emitted tier (B1). Dedup against PG-STACK first — crawled docs commonly duplicate repo READMEs (B4: strided sample, seed recorded). It is MIT-tagged, so it emits into the clean-permissive file, **not** into any PG-STACK OpenRAIL-M file. |
| **2** | **Wikimedia Commons / WIT per-file licence sampling audit** via the Commons API (`imageinfo` + `extmetadata`) | The only non-dead-end in `visual`, which is 300× short of target. Both the surveyor and the verifier name it the highest-value follow-up. | Stratified sample, licence recorded **per file** in the receipt. Keep CC0/CC BY in the permissive file; **quarantine CC BY-SA into a separate share-alike file, never merged** (§4.4). B5 stratum key = Commons category, max stratum ≤ max(0.25, 2/k). Enforce that Commons' guarantee is **policy-and-moderation, not technical** (§4, 30-verify-visual #21) — sample size must be chosen for that, not for a guaranteed floor. |
| **3** | **Run the DeepMind `mathematics_dataset` generator** at controlled category caps (an action, not a fetch) | Takes `reasoning` from 2.6% to ≥100% of target and unblocks `numeric_math` from PLACEHOLDER — without touching `reasoning`'s spent gsm8k/aqua_rat and without a new licence question. | Counts as **ONE** provenance group under B2 however many categories are generated. Category is the B5 stratum key. B4: the cap is a sample with a recorded method and seed, never a prefix. Cap its own share at ≤ 0.40 so it relieves B1 rather than replacing one monoculture with another. |
| **4** | **`bigcode/commitpackft` per-row `license` field distribution** | Decides whether PG-COMMITPACK is admissible at all. The card claims blanket "permissive" while its own enum contains `agpl-3.0` and `unknown` (§30-verify-code headline 1). | The filter must be **per row**, and the surviving `license` value must be carried into the emitted row's `lic_class` column (`20-enrichment` §4.2) — not into a sidecar manifest. Drop `agpl-3.0` and `unknown`; judge `mpl-2.0`/`lgpl-2.1`/`epl-1.0` separately as file-level weak copyleft. |
| **5** | **`nickrosh/Evol-Teacher` licence conflict** — contact the author, or find a release note reconciling the HF `cc-by-nc-sa-4.0` tag with the GitHub `Apache-2.0` LICENSE | Unblocks **two** entries: `Evol-Instruct-Code-80k-v1` (78,264 rows) and, through it, `m-a-p/CodeFeedback-Filtered-Instruction` (156,526 rows), whose feedback/correction shape nothing else in the catalogue covers. | The two readings emit into **different, non-mergeable files** (clean-permissive vs NC-SA), so this must resolve before emission, not after. Both entries also carry the MODEL-OUTPUT-TERMS flag, so §9's decision is a second gate on the same rows. |
| **6** | **`allenai/qasper`** second independent primary — the QASPER paper's data-availability statement, or a GitHub repo | The second-best `episodic_store`-shaped candidate after NarrativeQA. Blocked purely because `allenai.org/data/qasper` 302s to the HF card, so the only reachable statement fails A0m's host-diversity gate (`R1`). | If it resolves S2ORC-lineage, it joins the `s2orc` group with peS2o, SciFact's corpus and SciDocs — do not count it as independent. If the corpus half is ODC-By and the QA half CC BY, they are **two files**, not one. |
| **7** | **`FigureQA`'s MSR data-download terms** and **`PlotQA`'s MIT-vs-CC-BY-4.0 discrepancy** | ~404k synthetic charts across the two — the only chart/diagram sources in the catalogue that are structurally clean of the per-photographer trap. FigureQA's generator code is confirmed MIT, so a failed licence read still leaves the option of **running the generator**. | Both sit in the **`synthetic-chart-render`** group with CLEVR and Shapes3D, already in the vision composite — they are **not** independent of it for B2. PlotQA's plotted values carry World Bank Open Data's own terms; check before assuming the render licence covers them. |
| **8** | **Resolve `sentence-transformers/natural-questions`' grant scope** (§4.1) — does Google's Apache-2.0 reach the Wikipedia passage text, or only the pipeline? | Decides whether `memory`'s clean-permissive tier is 1.3% or ~2.8% of target, and whether the survey's rank-1 recommendation stands. It is the cheapest way to find out whether `memory` has *any* clean retrieval source. | Whatever the answer, NQ joins the **`wikipedia`** group for B1/B2 alongside SQuAD, HotpotQA, FEVER, DBpedia, MIRACL and Mr.TyDi. It cannot raise N_eff; at best it raises the volume the group is allowed to contribute. |
| **9** | **`courtlistener.com/terms/`** from a different egress (or pasted by the operator past the CloudFront 403) | Re-opens or closes 10M+ court opinions as a large independent provenance group for `language_trunk` and `reasoning` (§4.3). | If the Public Domain Mark governs the data, it emits into the clean-permissive file and is a genuinely new group — one of very few in the catalogue not already collapsed into Wikipedia, Common Crawl, S2ORC or the Stack. |
| **10** | **The SemEval-STS block** — one successful primary read of `ixa2.si.ehu.eus/stswiki` (or an archival snapshot) | Clears five candidates at once (`stsb`, `sts12`–`sts17`, and `biosses` alongside) — one provenance group, one fetch. It fills `compress`'s graded-similarity gate, which `00-ground.md` flags as never having run. | Only ~8,628 graded pairs, so it is an **eval** unlock, not a volume one. B3: declare it `held-out-domain` and report an in-mixture number beside it. **SICK is the already-cleared NC+SA alternative** that fills the same gap and is reachable today — reach for that first and treat this fetch as the permissive upgrade. |

**Deliberately not in the top ten**, and why: `bigcode/starcoderdata` (783 GB, the largest single
lever by volume) is one provenance group under an OpenRAIL-M singleton tier, so it cannot satisfy
B2 alone and its admission is a *waiver* decision, not a fetch; the **Stack Exchange dump** and
**MS MARCO** are legal readings, not fetches (§10); and **COCO's terms page**, which was the
visual survey's own #1 open fetch, is already closed (§4.2).

---

## 10. What is open for the operator

### 10.1 Legal readings to obtain, ranked by what they could invalidate

1. **Is CC BY-NC-SA 4.0 a coherent release licence for the composed model given its CC BY-SA
   inputs?** (`20-enrichment` §2.3, §5.2.) The highest-value question in the register, because it
   is the only one that could invalidate a **shipped** artefact rather than constrain a future
   one. Under the permissive reading of the weights question, CC BY-NC-SA is a freely chosen
   licence and coherent. Under the cautious reading, SNLI's §3(b)(1) requires the same licence
   elements — CC BY-NC-SA is not — and §3(b)(3) forbids added restrictions, while releasing as
   CC BY-SA would breach GooAQ's NC term: **under that reading there is no compliant release
   licence for the composed model at all.** The current licence is not the strictest-input answer;
   it is the answer the permissive reading makes available, and the project has not said which
   reading it takes. Two fixes, both implementable: separate the checkpoints per region (Option D)
   so the SA and NC obligations never have to be satisfied by one licence, or drop one side.
2. **Do OpenAI's and Anthropic's output restrictions bind a third party who receives the dataset?**
   (§5.6.) One decision covering the eleven `MODEL-OUTPUT-TERMS` entries in §4.5, worth roughly
   1e9 tokens across `language_code`, `memory` and `reasoning`. The factory's own generation is
   already REFUSED under SYN-L2 and that is not in question.
3. **MS MARCO specifically.** Microsoft's own sentence stacks three concerns: "non-commercial
   research purposes only", "without extending any license or other intellectual property
   rights", and **"we may not own the underlying rights in the documents."** That third clause is
   closer to the operator's REFUSE class — *distributors that disclaim owning what they
   distribute* — than to plain NC. It is 5.3e8 tokens and 3.3 of `memory`'s 5.0 NC-inclusive
   percentage points. **NC-with-a-waiver, or REFUSE?** This is a judgment call the surveys
   deliberately did not make.
4. **Does Stack Exchange's per-item attribution clause reach an inference-time weight release?**
   Four conditions, including a live outbound hyperlink to each original question and each author
   profile "without any `nofollow` command". A dataset-level manifest cannot satisfy that at any
   scale, which is why `R9` refuses it structurally. But if the clause does not bite on weights at
   all — only on redistributed *data* — then the largest single lever against GooAQ's 79.1%
   concentration becomes available. Worth asking as one question with #1.
5. **Is a machine-learning model a "Produced Work" under ODbL / ODC-By?** (§5.4.) ODbL's own
   definition is *"a work (such as an image, audiovisual material, text, or sounds) resulting from
   using … the Contents (via a search or other query)"*. A set of weights is none of those, and
   training is not obviously a query. If a model is neither a Derivative Database nor a Produced
   Work, ODC-By is simply **silent** — a gap, not a permission. This bears on FineWeb-Edu, C4,
   peS2o, OpenWebMath, WildGuard and SciFact's abstract corpus.
6. **`toxigen/toxigen-data`'s conflict**: an unrestricted CDLA-Permissive-2.0 data grant (which
   *expressly* states it imposes nothing on machine-learning models — the only family surveyed
   where the Layer-2 question has a written answer) against a README sentence narrowing use to
   "research purposes only". The two readings are as far apart as the catalogue gets, and both
   passes deliberately declined to resolve it.
7. **Do sui generis database rights apply to a US-domiciled operator at all?** (§5.3.) If none
   applies, a filter-only enrichment may not be Adapted Material and §7's operation table
   over-constrains. The engineering answer is to comply anyway; the commercial answer may differ.
8. **How far does Gemma's "Model Derivative" definition reach through synthetic data?** (§5.7.)
   No de minimis threshold is stated, so "only a small synthetic share" is not obviously a
   defence. Until answered, Gemma is ENCUMBERED at any share.

### 10.2 Corrections to make in documents this catalogue does not own

- **`00-ground.md`'s `memory` row lists FiQA's obligation as CC-BY-SA-4.0.** The primary source
  says **NC** ("The training data is available only for non-commercial use"). The ground doc
  understates FiQA's restriction; its successor should carry NC.
- **`10-survey-compress.md` is internally inconsistent on `ltg/en-wiki-paraphrased`**: its top-8
  table says "ATTRIBUTION (unsettled)" while its full entry and its refused table both say
  REFUSE-pending. Resolved here as UNVERIFIED; the survey text should be made consistent so a
  reader skimming only the table does not treat it as provisionally usable.
- **`maveriq/bigbenchhard`'s HF card asserts the upstream `google/BIG-bench` repo is MIT.** It is
  Apache-2.0. Verdict is unaffected (both are PERMISSIVE_OK) but the card's claim must not be
  cited downstream as if it had been checked.

### 10.3 Credentials and access the operator may want to provide

- **Kaggle credentials.** `secret ls | grep -i kaggle` returned no entry, so Kaggle candidates
  were checked via public dataset pages only. Two entries turn on login-gated Kaggle competition
  rules: **Quora Question Pairs** (whose standard Kaggle boilerplate, if it applies, reads
  non-commercial **and** non-redistributable, which is REFUSE-TERMS rather than NC) and
  **`google/jigsaw_toxicity_pred`** (already upgraded to VERIFIED via an independent TFDS surface,
  so the Kaggle read would only close it fully). **Worth adding if the operator wants the QQP
  question settled; not blocking anything else.**
- **An RRC / DocVQA portal account.** DocVQA task 1 and InfographicVQA both sit behind
  `rrc.cvc.uab.es`'s login, and a "SOFTWARE EVALUATION LICENSE AGREEMENT" reference was surfaced
  for at least one dataset in that family. Nobody without an account can close either entry.
- **A different network egress** for `courtlistener.com/terms/` (CloudFront 403) and, if it is
  ever wanted, `openai.com/policies/*` (403 to direct fetch; the substance was recovered from
  `openaifoundation.org/terms-of-use`, which carries the same formulation).
- **The HF token was not needed at any point** — the public API answered every query in all eight
  surveys. `gpu/huggingface-token` remains unused by this work.

### 10.4 The moral-corpus probe design (DEC-59 / P5′m)

The operator's requirement is to **measure** whether training on a moral corpus changes
behaviour, never to assume it. The catalogue supports that directly, and the design falls out of
the tier separation:

- **Two arms, one difference.** Train the `moral_safety` clean-permissive tier alone
  (`T_B1` 2.4e8, 7 groups: ETHICS, HH-RLHF, Moral Stories, Scruples, HateXplain, Civil Comments,
  Jigsaw), then train the NC-inclusive union (adding the PKU NC file and the Social Chemistry
  share-alike file as separate files). The difference between the two arms is the measured price
  of the NC and SA inputs — the `measure-the-thing-not-the-proxy` discipline applied to a licence
  decision, and the reason `20-enrichment` §4.6 requires a separable permissive tier per faculty.
- **An untrained baseline is required**, per the same discipline: a result without one is not
  interpretable, and W2c already retired one disputed floor for exactly this reason.
- **Probes are pre-registered and held out, and their provenance groups contribute zero training
  rows** — B3 mode `held-out-domain`, with an in-mixture number reported beside it. The probe set,
  in order of fitness for this measurement: **MoralChoice** (purpose-built to elicit revealed
  moral preferences under low- and high-ambiguity dilemmas), **ETHICS' held-out slice**,
  **Measuring Hate Speech** (its IRT-aggregated *continuous* severity score is the best
  calibration instrument in the catalogue — and its probe use is far less exposed than its
  training use to the annotations-vs-underlying-media gap flagged in §6), **Civil Comments**,
  **OpenAI Moderation Eval**, **StereoSet**, **CrowS-Pairs**, and **Jigsaw Toxicity** as a
  cross-taxonomy check against Civil Comments.
- **The corpus does not target 1e10 tokens.** DEC-59 and the operator's stated preference are
  *"extremely high quality and requirement-complete over large"*. The §8 figure of 2.4% is
  recorded for comparability with the other faculties, not as a shortfall to close.
- **Two entries must be settled before the corpus is built, not after**: `toxigen/toxigen-data`'s
  CDLA-vs-README conflict (§10.1 #6) and `ucberkeley-dlab/measuring-hate-speech`'s grant scope
  (annotations only, or the underlying Twitter/Reddit/YouTube post text too). The second matters
  less if it is used only as a probe, which is what is recommended.

---

## Appendix — what this catalogue does not claim

- **It fetched nothing.** Every licence quote, URL and fetch date comes from the evidence files
  copied to `docs/design/evidence/dataset-factory-2026-09-03/`. Where a fetch failed, it says so
  and the entry is `UNVERIFIABLE`, not inferred clean.
- **Every token figure in §8 is INFERRED**, under the two assumptions stated there. The row and
  byte counts they derive from are VERIFIED where the entry says so.
- **No verdict here is a legal conclusion.** §10.1 is the register of what needs a human.
- **`CONSENT_OPEN` (OD-11) remains recommended and unadopted.** No candidate in this pass was
  positively filed under it; `lmsys/lmsys-chat-1m` was refused on its non-redistribution term
  instead, deliberately, because it was not independently confirmed to fit the consent class and
  refusing is safer than assuming.
