# Pass 2 verification — Claude's independent read of Grok's 17 resolutions

Session date: 2026-09-03. Analyst: Claude (Sonnet 5), same-day second read of
`40-grok-verify-unverified.{md,json}`. Scope: the 17 entries Grok proposed a definite verdict
for, plus a pass over the 20 it left `STILL-UNVERIFIED`. Method: re-fetch the URL Grok cites
(WebFetch); where the fetch failed for a reason specific to this environment (a blocked domain,
a TLS trust-store gap, a gated repo), try an alternate primary or a WebSearch corroboration of
the same page, and record the discrepancy rather than silently substituting. Never adopt a
verdict without independently reading a primary text.

Operator stance and verdict classes: `docs/design/DATASET-FACTORY-CATALOGUE-2026-09-03.md` §1–§2.
Full per-entry JSON fields (the authoritative record): `docs/design/datasets/catalogue-2026-09-03.json`.

**Result: 13 of 17 adopted, 4 declined.** Of the 4 declines, two share one defect (Grok
auto-applied the `MODEL-OUTPUT-TERMS` REFUSE reading the catalogue's own §4.5 explicitly reserves
for a single pending operator decision), one rests on a citation from the wrong HF repo
(`nvidia/Nemotron-CC-v2`), and one declines to override an existing "operator/legal decision, not
a surveyor's call" reservation on primary texts that were already correctly quoted
(`toxigen/toxigen-data`).

## Part 1 — the 17 Grok resolved

| id | Grok's URL | re-fetched by Claude | quote confirmed | Claude's verdict | reason |
|---|---|---|---|---|---|
| `OpenCoder-LLM/opc-annealing-corpus` | huggingface.co/datasets/OpenCoder-LLM/opc-annealing-corpus/raw/main/README.md | yes, same URL | yes, verbatim | **SHARE_ALIKE** (adopted) | Confirms Stack-v2 sourcing; formalises what §5's provenance-group synthesis already implied (PG-STACK "swallows" this entry). |
| `OpenCoder-LLM/opc-fineweb-code-corpus` | huggingface.co/datasets/HuggingFaceFW/fineweb/raw/main/README.md (parent only) | yes, both the parent AND this entry's own card | yes, both verbatim | **ATTRIBUTION** (adopted) | Claude additionally fetched this entry's OWN card (Grok did not) and found it confirms FineWeb sourcing directly: "This code-related data from Fineweb was specifically used in OpenCoder pre-training." Closes the prior pass's "not confirmable" gap. This is the entry the catalogue's own §9 #1 already called "what makes `language_code`'s B1 satisfiable at all" — see §8 arithmetic. |
| `neulab/conala` | stackoverflow.com/help/licensing | **no** — "Claude Code is unable to fetch from stackoverflow.com" (confirmed on 3 attempts: https, http, a web.archive.org route, all domain-blocked in this environment) | not on Grok's URL; **yes** on an alternate primary (ia600508.us.archive.org/30/items/stackexchange/license.txt, the licence file bundled with the actual SE data dump CoNaLa is mined from) | **SHARE_ALIKE** (adopted) | Alternate primary confirms the same substantive clause (CC BY-SA, per-item attribution), corroborated by a WebSearch snippet of Grok's exact page. Matches the existing catalogue's own prior finding (30-verify-code #16). DISCREPANCY logged: Claude could not independently read Grok's cited page. |
| `m-a-p/CodeFeedback-Filtered-Instruction` | huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction/raw/main/README.md | yes, same URL | yes, verbatim (OpenAI usage-policy warning, ShareGPT/Evol-Instruct-Code lineage) | **UNVERIFIED** (declined) | Quote confirmed; verdict declined. This entry is one of the 11 named in §4.5 as `MODEL-OUTPUT-TERMS`, explicitly reserved as "not auto-refused … one operator decision covering all eleven." Also still coupled to nickrosh's unresolved mirror conflict below. |
| `nickrosh/Evol-Instruct-Code-80k-v1` | raw.githubusercontent.com/nickrosh/Evol-Teacher/main/README.md | yes, same URL | yes, verbatim (API-call generation process only) | **UNVERIFIED** (declined) | Grok's fetch did not touch the actual blocking issue: the GitHub `LICENSE` file (Apache-2.0) vs the HF mirror tag (`cc-by-nc-sa-4.0`), both already independently VERIFIED and in direct conflict (30-verify-code #18) — a different file from the one Grok fetched. REFUSE-TERMS also falls under the same §4.5 reservation as CodeFeedback. |
| `allenai/qasper` | arxiv.org/html/2105.03011 | yes, same URL | yes, verbatim ("restricted ourselves to arXiv papers released under a CC-BY-* license") | **ATTRIBUTION** (adopted) | arxiv.org is independent of huggingface.co, resolving the prior pass's R1 host-diversity gate. Paper header also states "License: CC BY 4.0." |
| `ltg/en-wiki-paraphrased` | huggingface.co/datasets/ltg/en-wiki-paraphrased/raw/main/README.md | yes, same URL | yes, verbatim (English Wikipedia source, Mistral-7B paraphrase, `license: apache-2.0`, no CC BY-SA mention) | **SHARE_ALIKE** (adopted) | English Wikipedia is CC BY-SA; paraphrases of BY-SA material are adapted material under the same family. Applies the existing note's own "option (b)" to the corpus as distributed (the `original` column present). |
| `wikimedia/wit_base` | huggingface.co/datasets/wikimedia/wit_base/raw/main/README.md | yes, same URL | yes, verbatim ("CC BY-SA 4.0 international license") | **SHARE_ALIKE** (adopted) | Dataset-level declared licence confirmed. Per-image audit caveat unchanged (individual Commons files may carry a stricter original licence). See §8 for the reach consequence — `visual`'s first share-alike candidate. |
| `NiteshMethani/PlotQA` | raw.githubusercontent.com/NiteshMethani/PlotQA/master/README.md | yes, same URL | yes, verbatim ("datasets … CC-BY-4.0 … models & code … MIT") | **ATTRIBUTION** (adopted) | This is the README, a different file from the repo-root LICENSE the first pass checked (bare MIT, no data/code split). Resolves the "not confirmable in any primary text this pass" gap. |
| `derek-thomas/ScienceQA` | raw.githubusercontent.com/lupantech/ScienceQA/main/README.md | yes, same URL | yes, verbatim ("Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License") | **NC** (adopted) | Upstream (original authors, independent of the HF mirror host) states CC BY-NC-SA 4.0; the HF mirror tag `cc-by-sa-4.0` omits the NC element — a mirror lie. Multi-curriculum constituent-source risk (IXL Learning claim) stays open, not resolved by this pass. |
| `togethercomputer/RedPajama-Data-V2` | huggingface.co/datasets/togethercomputer/RedPajama-Data-V2/raw/main/README.md | yes, both this URL AND commoncrawl.org/terms-of-use | yes, both verbatim | **REFUSE** (adopted) | Together grants no independent database-rights instrument; Common Crawl's own ToU licenses access to the crawl SERVICE, not a copyright grant, and states "BY USING THE CRAWLED CONTENT, YOU AGREE TO RESPECT THE COPYRIGHTS … OF THIRD PARTIES" — the operator's "distributor disclaims ownership" REFUSE class. Contrast with FineWeb (above), which states an explicit ODC-By grant over the same underlying crawl. |
| `nvidia/Nemotron-CC-v2` | huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample/raw/main/LICENSE.md | attempted this entry's OWN card instead (Grok's URL is a **different** HF repo) | **no** — this entry's own card (huggingface.co/datasets/nvidia/Nemotron-CC-v2/raw/main/README.md) returned 401 Unauthorized (gated) | **UNVERIFIED** (declined) | Grok's quoted clauses ("Company may not … Sell, rent, sublicense …") come from a DIFFERENT repo's licence file, in second-person "Company" B2B template language. This catalogue's own prior verify pass had already quoted THIS entry's own card describing the same-named "NVIDIA Data Access Agreement for Model Training" as enabling "training of any AI model, including models released under proprietary or open source licenses" — a materially different framing. Whether the two documents are the same instrument is unconfirmed. |
| `Software Heritage` | softwareheritage.org/legal/bulk-access-terms-of-use/ | attempted, **failed**: "unable to get local issuer certificate" (same TLS trust-store gap the prior pass hit with `curl -k`) | not by direct fetch; **yes** via a WebSearch snippet reproducing the identical clause from this same URL | **REFUSE** (adopted) | Substance unchanged from the prior pass ("makes no claim of correctness … You are solely responsible for determining the license"). Promotes the prior pass's hedge ("adjacent to … the R3 refuse class") to the formal REFUSE class the operator's stance names explicitly. |
| `CourtListener bulk data (Free Law Project)` | wiki.free.law/c/courtlistener/help/api/bulk-data/bulk-legal-data | yes, same URL | yes, verbatim ("free of known copyright restrictions," Public Domain Mark, every bulk table individually marked) | **PERMISSIVE_OK** (adopted) | Matches the catalogue's own §4.3 finding exactly. Closes the call: the remaining gap (courtlistener.com/terms/, 403) covers only the site's presentation layer per that same prior note. Not yet folded into §8 (no byte/token size on record for this entry). |
| `Papers with Code dataset index` | github.com/paperswithcode/paperswithcode-data | yes, same URL | yes, verbatim ("All data is licenced under CC-BY-SA") | **SHARE_ALIKE** (adopted) | Applies to the index dump itself. Admissibility unchanged — remains a discovery tool per its own existing "why"/"caveat", not a training source. |
| `toxigen/toxigen-data` | raw.githubusercontent.com/microsoft/TOXIGEN/master/LICENSE.txt | yes, same URL | yes, verbatim (matches the prior verify pass exactly) | **UNVERIFIED** (declined) | Both primary texts (CDLA-Permissive-2.0 grant vs README's "research purposes only" sentence) were already correctly quoted before this pass — there is no new discrepancy, only a reading. The existing entry explicitly reserves this as "an explicit operator/legal decision, not a surveyor's call." Declined to override that reservation. |
| `ucberkeley-dlab/measuring-hate-speech` | arxiv.org/html/2009.10277 | yes, same URL | yes, substance confirmed (platform-scraped sourcing, YouTube/Reddit/Twitter, MTurk annotation, 50,070 comments); exact "open source dataset" phrasing not reproduced verbatim by the fetch tool's summary | **REFUSE** (adopted, for training; `EVAL-ONLY` unchanged) | Matches the existing catalogue's own "annotations-vs-underlying-media gap" finding exactly (30-verify-moral-safety #13). D-Lab's CC BY 4.0 plausibly covers its own annotations, not the underlying scraped post text. |

## Part 2 — the 20 Grok left `STILL-UNVERIFIED`

No new fetching was done for these; Grok's "what would settle it" line is recorded verbatim in
each entry's new `notes` field in the JSON. Listed here for completeness, matching
`40-grok-verify-unverified.md`'s own table exactly:

`bigcode/commitpackft`, `bigcode/commitpack`, `princeton-nlp/SWE-bench`,
`SWE-bench/SWE-bench_Verified`, `BeIR/quora`, `BeIR/climate-fever`, `tau/scrolls`,
`sentence-transformers/wikianswers-duplicates`, `mteb/stsbenchmark-sts`,
`mteb/sts12-sts .. mteb/sts17-crosslingual-sts, mteb/biosses-sts`, `AI-MO/NuminaMath-CoT`,
`EleutherAI/proof-pile-2`, `tasksource/proofwriter`, `OpenML`, `Maluuba/FigureQA`,
`DocVQA (task 1)`, `lmms-lab/ai2d (AI2D)`, `PubMed Central Open Access subset`,
`allenai/tulu-3-sft-mixture`, `walledai/TDC23-RedTeaming`.

## Notes on method, this pass

- Three domains were unreachable to WebFetch in this environment regardless of the specific page
  requested: `stackoverflow.com` (all of https, http and a `web.archive.org` route to it),
  `web.archive.org` generally, and TLS to `softwareheritage.org` ("unable to get local issuer
  certificate", matching the prior pass's own note that it needed `curl -k`). Where this happened,
  an alternate primary or a WebSearch corroboration was used and the substitution is logged in the
  table above rather than silently treated as equivalent to a direct fetch.
- `nvidia/Nemotron-Pretraining-Dataset-sample` returned content freely; `nvidia/Nemotron-CC-v2`
  (this catalogue's actual entry) returned 401 Unauthorized on direct re-fetch — the entry appears
  to have become gated between the prior verify pass and this one, or access varies by session.
  Either way, Grok's citation of the `-sample` repo's licence file cannot be verified as the same
  instrument this entry's own card names, so the entry stays `UNVERIFIED`.
- No CogSynDelta file outside `docs/design/` was touched, and no dataset content was downloaded.

**VERIFIED vs INFERRED, this pass's own claims:** every quote-confirmation and every URL in the
table above is VERIFIED (Claude read it, or explicitly could not and says so). The §8 token
counts derived from those quotes (e.g. "147.9 GB → 3.7e10 tokens") are INFERRED under the
catalogue's stated 4-bytes/token assumption, exactly as the rest of §8 already is.
