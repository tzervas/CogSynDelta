# Adversarial verification — 10-survey-language-general.md

Method: for every candidate whose verdict is not a plain REFUSE, fetched the primary
licence source myself this session (2026-09-03) — publisher HF card, project page,
GitHub LICENSE/README, or the original rights-holder's terms page — independent of the
surveyor's quotes, plus the public HF API for tag ground-truth. arXiv (#10) and PMC OA
(#11) are UNVERIFIED-not-REFUSE in the survey and were checked too, since they are
exactly the mixed-licence/per-item-variance shape this audit is looking for. LMSYS (#18)
and FLAN (#14) are flat REFUSE calls; I spot-checked LMSYS's quotes only, for confidence.
No dataset content was downloaded — HF public API + WebFetch/WebSearch of licence/card
pages only. No content matched the LAION/scraped-URL-list, ND, or consent-revocable
patterns beyond what's logged below.

Verdict key: **VERIFIED-AGREES** / **CONTRADICTED** (corrected verdict + quoted clause) /
**UNVERIFIABLE** (what was missing).

---

### 1. FineWeb-Edu — VERIFIED-AGREES
HF card confirms `license: odc-by` and ODC-By framing at publisher = primary, matching
the survey exactly. **Added detail the survey didn't have**: fetched Common Crawl's own
Terms of Use directly (`commoncrawl.org/terms-of-use`) — its licence to *use* Crawled
Content is explicitly **"non-sublicensable"**, and it carries an indemnification clause
naming AI training directly: *"use of Crawled Content in connection with artificial
intelligence, machine learning, or other similar technologies, including... large
language models... "* requires the user to indemnify CC for resulting third-party claims.
This is real tension with any CC-derived corpus's own re-licence (ODC-By, applied by
HuggingFaceFW/AllenAI/Google to their own processed selection, not by CC itself) — but it
is the same shape the survey already flagged as "an ambiguity every CC-based corpus
quietly assumes," now backed by the actual clause text instead of an assumption. Verdict
unchanged; flag strengthened, not new.

### 2. Dolma — CONTRADICTED (partial — inconsistent rigor, not a verdict flip)
ODC-By confirmed at the HF card. **Gap**: the survey gives peS2o and Tulu-3-SFT a
"provisional / inherited-trust" caveat for being AllenAI composites of many sub-sources,
but gives Dolma none, despite Dolma being the largest, most heterogeneous composite in
the survey (Common Crawl + C4 + peS2o + Gutenberg + Wikipedia + **Reddit via PushShift**
+ code) and still calling it a flat **top candidate**. I confirmed via AllenAI's own
sources (Dolma README, the Ai2 "Making a switch — Dolma moves to ODC-BY" blog post, and
`arxiv.org/pdf/2402.00159`) that the ODC-By re-licence is AllenAI's own blanket claim over
the whole mixture, same shape as peS2o/Tulu-3 — and that the Reddit component is derived
from the PushShift dump, a source whose own standing is publicly contested: Reddit
revoked PushShift's platform access in 2023 specifically over bulk-scraping/AI-training
concerns, and individual Reddit users — not Reddit or AllenAI — hold copyright on their
own posts. This doesn't rise to a REFUSE call on the evidence gathered this pass, but it
is exactly the "inherited-trust re-license claim over a mixed-rights sub-source" pattern
the survey itself uses to caveat peS2o and Tulu-3. **Recommend**: apply the same
"provisional, trusting AllenAI's blanket claim" caveat to Dolma that peS2o/Tulu-3 already
carry, and flag the Reddit/PushShift stratum specifically for a spot-check before scale-up
— do not keep it un-caveated as a flat top candidate while peer composites are caveated.

### 3. RedPajama-Data-v2 — VERIFIED-AGREES (fills a gap the survey named)
Card body confirms the exact quoted line (*"Please refer to the Common Crawl Foundation
Terms of Use for the data"*) and the Apache-2.0 code-only licence. The survey explicitly
said Common Crawl's own ToU "was not independently fetched this pass" — I fetched it (see
#1 above): non-sublicensable license, explicit AI/ML-training indemnification clause. This
resolves the survey's named gap in the direction of "same risk shape as every other
CC-derived candidate here," not a new one. Verdict (PERMISSIVE_OK, provisional) unchanged.

### 4. Cosmopedia — VERIFIED-AGREES
Card confirms `license: apache-2.0` and Mixtral-8x7B-Instruct generation, matching the
survey. Mistral's Apache-2.0 model licence imposes no output-use restriction (unlike the
OpenAI case below), so the survey's "soft risk, not a blocker" framing holds.

### 5. Nemotron-CC — CONTRADICTED (two points)
**(a) Slug**: the survey flagged `nvidia/Nemotron-CC`/`-v2` as 404'ing and "UNVERIFIED
mirror slug." I fetched `huggingface.co/datasets/nvidia/Nemotron-CC-v2` directly this pass
and it resolves fine (also `-v2.1`, `-Code-v1`, `-Math-v1` all exist per WebSearch). Minor
factual correction, not a licence issue — the slug question is resolved, not open.
**(b) The Qwen/DeepSeek flag is CONFIRMED, not speculative.** The survey wrote "if the
synthetic-rephrase portion derives from Qwen/DeepSeek model outputs, those models' own
licence terms... may attach (unresolved this pass)." I fetched the actual NVIDIA dataset
card: it states outright that the synthetic tier was **"created using"** Qwen3-30B-A3B,
Qwen2.5 variants, DeepSeek-V3, and DeepSeek-R1, and that *"such AI model may be subject to
redistribution and use requirements in the Qwen License Agreement... and the DeepSeek
License Agreement."* I then read the Qwen License Agreement's actual terms (via
GitHub/HF/community sources): it caps free commercial use at 100M MAU (above that,
Alibaba's separate licence is required), restricts building "competitive AI services,"
and requires a "Built with Qwen"/"Improved using Qwen" notice on any model trained on its
outputs. DeepSeek's model licence is broader (MIT-like redistribution rights, no output
claim) but still carries a use-restriction flow-through clause. **This is a real,
confirmed additional constraint on the composed model's release licence** if the
Nemotron-CC synthetic tier is admitted — not a hypothetical to "flag for legal re-read,"
but a term to actually satisfy (attribution notice + the 100M-MAU commercial cap) per the
DEC-31 strictest-input rule. Recommend upgrading this from a soft flag to a concrete
release-licence obligation before promotion.

### 6. Wikipedia — VERIFIED-AGREES
`foundation.wikimedia.org/wiki/Policy:Terms_of_Use` §7.1 confirmed verbatim: *"you agree
to license it under: Creative Commons Attribution-ShareAlike 4.0 International License
('CC BY-SA 4.0'), and GNU Free Documentation License ('GFDL')."* The survey's catch of the
mirror-says-3.0-upstream-says-4.0 mismatch is correct and independently reproduced.

### 7. StackExchange — VERIFIED-AGREES
`archive.org/details/stackexchange` confirmed verbatim: *"All user content contributed to
the Stack Exchange network is cc-by-sa 4.0 licensed"* plus all four attribution conditions
quoted in the survey (visual source credit, hyperlink to original, author names, hyperlink
to author profiles), reproduced exactly. `stackoverflow.com/help/licensing` was not
independently fetchable this pass (tool-blocked), so the archive.org primary source
remains the sole confirmation — consistent with what the survey already used.

### 8. Project Gutenberg — VERIFIED-AGREES
`gutenberg.org/policy/permission.html` confirmed: PD, "as you please" for commercial use
and derivatives, attribution explicitly optional (*"it is also OK to not cite Project
Gutenberg: your choice"*), and the trademark-only royalty condition on retained branding —
all reproduced verbatim, matching the survey exactly.

### 9. peS2o — VERIFIED-AGREES
`license: odc-by` confirmed on the card, S2ORC/Semantic Scholar sourcing confirmed, and —
matching the survey's own flag — the card does **not** independently address underlying
publisher licences for the ~40M constituent papers. The survey's "provisional,
inherited-trust" framing is accurate and not overstated.

### 10. arXiv (full-text) — VERIFIED-AGREES (REFUSE-as-full-text correctly cautious)
`info.arxiv.org/help/license/index.html` confirmed: metadata is CC0 (*"A Creative Commons
CC0 1.0 Universal Public Domain Dedication will apply to all metadata"*), while full-text
carries **five different author-selectable options** including CC BY-NC-SA 4.0 and CC
BY-NC-ND 4.0 (non-commercial, and non-commercial+no-derivatives) alongside the arXiv
perpetual non-exclusive licence — i.e. a materially mixed bag including ND terms the
operator stance explicitly refuses. The survey's "REFUSED (as full-text) this pass,
metadata layer separately assessable" call is well-supported, not overcautious.

### 11. PubMed Central Open Access subset — VERIFIED-AGREES (one miscount, immaterial)
`pmc.ncbi.nlm.nih.gov/tools/openftlist/` confirmed the exact quoted line twice: *"License
terms vary. Please refer to the license statement in each article for specific terms of
use."* HF API tag cross-check: `pmc/open_access` actually carries **7** CC licence
variants (cc0-1.0, cc-by-4.0, cc-by-sa-4.0, cc-by-nd-4.0, cc-by-nc-4.0,
cc-by-nc-sa-4.0, cc-by-nc-nd-4.0) plus `other`/`unknown` = 9 tags total — the survey said
"all 8 CC variants plus other/unknown"; it's 7, not 8. Trivial miscount, verdict
(UNVERIFIED-as-blanket, admit only the clean CC0/BY/BY-SA slice) unaffected — if anything
the ND and NC-ND variants confirmed present make the "filter OUT the ND slices, tag NC for
the NC slices" instruction more clearly necessary, not less.

### 12. OpenAssistant (oasst2) — VERIFIED-AGREES
`license: apache-2.0` confirmed on the card; human-written, human-reviewed conversation
trees confirmed (review/label/detoxify fields), matching the survey's "not
model-generated" distinction.

### 13. Databricks Dolly 15k — VERIFIED-AGREES, one UNVERIFIABLE sub-point
GitHub README confirms *"generated by Databricks employees and released under a
permissive license (CC-BY-SA)"* verbatim. **Note**: neither the GitHub README nor the
repo's own LICENSE file (which is Apache-2.0 — that covers the *code*, not the dataset)
states a CC-BY-SA *version number*; "3.0" comes only from the HF mirror tag and was never
independently confirmed at a primary source, unlike Wikipedia where the upstream ToU page
explicitly names 4.0. Mark the version number **UNVERIFIABLE** this pass (mirror-only),
not confirmed-matching as the entry's clean "SHARE_ALIKE, VERIFIED" framing implies.

### 15. Tulu 3 SFT mixture — CONTRADICTED (verdict undersells a confirmed self-admission)
Card confirms `license: odc-by`, but the **README itself** states, in the surveyor's own
words territory: *"different licenses apply to subsets of the data. Some portions of the
dataset are non-commercial"* and separately that some material is *"output data generated
from third party models that are subject to separate terms governing their use."* This is
not an inference or a risk this pass introduces — it is AllenAI's **own card text
admitting** the blanket ODC-By tag does not cover the whole mixture. That is the identical
shape the survey used to justify **REFUSE-wholesale** for FLAN and **UNVERIFIED-as-blanket**
for PMC OA, but here the survey called it "PERMISSIVE_OK, provisional... lower-risk than
FLAN's opaque 1800-task blanket" and recommended only "a spot-check pass." Given the card
explicitly confirms NC subsets exist inside the ODC-By claim, **recommend correcting the
verdict class to UNVERIFIED-as-blanket** (matching PMC OA/FLAN treatment) rather than
provisional-PERMISSIVE_OK, and requiring the same per-subset split before whole-corpus
admission.

### 16. Alpaca (tatsu-lab) — CONTRADICTED (verdict undersells a confirmed research-only clause)
Mirror tag and CC BY-NC 4.0 confirmed. But the primary README (fetched verbatim via raw
GitHub) contains language the survey's quote **omitted**: *"Alpaca is intended and
licensed for research use only"* and, separately, *"models trained using the dataset
should not be used outside of research purposes."* This is exactly the operator's named
audit trigger — *"research only" clauses hidden in a README* — and it is a materially
narrower restriction than plain CC-BY-NC-4.0 (NC permits any non-commercial use — hobbyist,
nonprofit-operational, etc.; "research purposes only" is a subset of that). The survey's
verdict, **NC (does not block per DEC-31)**, treats this as ordinary NC and misses the
additional field-of-use language entirely. **Recommend correcting the verdict** to at
minimum an escalated NC-plus-research-only-restriction class requiring explicit legal
sign-off on whether that clause is a binding field-of-use limit beyond the stated licence
badge, and per the operator's REFUSE criteria for research-only terms, treat this as a
REFUSE candidate rather than a plain-NC admit until that's resolved.
Separately, the layered-OpenAI-terms flag the survey raised as "unresolved" is now
**confirmed live**: WebSearch of OpenAI's current Terms of Use (dated Jan 1 2026) found the
clause *"Use Output to develop models that compete with OpenAI"* still present as of this
session (direct WebFetch of `openai.com/policies/row-terms-of-use/` was blocked with a 403
this pass, so this is corroborated via search summary of the primary page, not a direct
quote-in-hand — flag as VERIFIED-AGREES-via-secondary-corroboration, not a first-hand
quote). This confirms, rather than resolves, the survey's flag — a real, current
contractual restriction sits on top of Alpaca's declared licence.

### 17. UltraChat — VERIFIED-AGREES (MIT + generation method), OpenAI flag now confirmed
GitHub README confirmed verbatim: *"distributed under the MIT license"* and *"all the data
is automatically generated... using [ChatGPT/Turbo APIs]"* with *"we do not directly use
any data available on the internet as prompts."* No "research only" language equivalent to
Alpaca's was found here — UltraChat's own README doesn't narrow MIT the way Alpaca's README
narrows CC-BY-NC. The OpenAI-output-terms flag applies here too and is now confirmed live
(see #16) rather than merely unresolved — same "resolve as one decision covering both"
recommendation the survey already made stands, now with the underlying clause confirmed
rather than assumed.

### 18. LMSYS-Chat-1M (REFUSE, spot-checked for confidence) — VERIFIED-AGREES
HF API confirms `gated: auto`. Card text confirmed both quoted clauses verbatim: *"You
should not distribute, copy, disclose, assign, sublicense, embed, host, or otherwise
transfer the dataset to any third party"* and the *"for both research and commercial
purposes"* training-use grant. The REFUSE-TERMS call (non-redistributable, plus
unresolved consent provenance on raw public-chat-demo conversations) is well-supported.

### 19. Anthropic HH-RLHF — VERIFIED-AGREES, upgraded from INFERRED to VERIFIED
The survey explicitly flagged the MIT claim as "INFERRED-from-repo-navigation, not
VERIFIED-by-quoted-text." I fetched the actual LICENSE file
(`raw.githubusercontent.com/anthropics/hh-rlhf/master/LICENSE`) directly this pass: full
MIT text, copyright line *"Copyright (c) 2022 Anthropic."* This closes the survey's named
gap — the entry can now read fully VERIFIED rather than partially inferred.

### 20. Common Corpus (Pleias) — VERIFIED-AGREES
Card confirms the general framing verbatim (*"All data in Common Corpus are either
uncopyrighted or freely licensed and may be used for both commercial and non-commercial
purposes"*) and the six-stratum breakdown (OpenCulture/OpenGovernment/OpenSource/
OpenScience/OpenWeb/OpenSemantic) matching the survey's token counts closely (OpenWeb
89B vs. the survey's 88B — trivial rounding, immaterial). No contradiction found; the
survey's "admit strata, not the whole corpus, to avoid double-counting Gutenberg/
Wikipedia" recommendation is sound and unaffected.

### 22. C4 — VERIFIED-AGREES
Card confirms `license: odc-by` and the exact bind-to-Common-Crawl-ToU clause quoted by
the survey. Same CC-derived caveat as #1/#2/#3 applies identically; no new issue.

---

## REFUSE-class entries not independently re-verified beyond #18

**FLAN (#14)** and **The Pile deduplicated (#21)** were left as the survey's own REFUSE
calls (per task scope: verify entries with a verdict *other than* REFUSE). Both refusal
rationales read as well-supported from the survey text itself (FLAN: Apache-2.0 wrapper
over ~1800 unaudited constituent tasks; Pile: Books3's well-documented, public rights
dispute) and nothing found while checking adjacent entries (Tulu-3's self-admitted NC
subsets, Dolma's PushShift-Reddit component) undermines those two refusals — if anything
it reinforces that "one blanket tag over many sub-sources" is a real, recurring failure
mode across several of the survey's PERMISSIVE_OK-provisional calls too, not just the two
it already refused for that reason.

---

## Summary — corrected verdicts

| # | Candidate | Survey verdict | Verification result | Corrected verdict (if changed) |
|---|---|---|---|---|
| 1 | FineWeb-Edu | PERMISSIVE_OK | VERIFIED-AGREES | — |
| 2 | Dolma | PERMISSIVE_OK, top candidate | CONTRADICTED (rigor gap) | PERMISSIVE_OK, **provisional** (Reddit/PushShift stratum unaudited — same caveat as peS2o/Tulu-3) |
| 3 | RedPajama-v2 | PERMISSIVE_OK, provisional | VERIFIED-AGREES | — |
| 4 | Cosmopedia | PERMISSIVE_OK | VERIFIED-AGREES | — |
| 5 | Nemotron-CC | PERMISSIVE_OK for training, flag | CONTRADICTED | PERMISSIVE_OK for training **with confirmed** Qwen-License + DeepSeek-License flow-through obligations (attribution notice, 100M-MAU commercial cap) — no longer merely "unresolved" |
| 6 | Wikipedia | SHARE_ALIKE | VERIFIED-AGREES | — |
| 7 | StackExchange | SHARE_ALIKE | VERIFIED-AGREES | — |
| 8 | Project Gutenberg | PERMISSIVE_OK | VERIFIED-AGREES | — |
| 9 | peS2o | PERMISSIVE_OK, provisional | VERIFIED-AGREES | — |
| 10 | arXiv full-text | UNVERIFIED / REFUSED-as-full-text | VERIFIED-AGREES | — |
| 11 | PMC OA subset | UNVERIFIED as blanket | VERIFIED-AGREES (7 not 8 CC tags — immaterial) | — |
| 12 | OpenAssistant | PERMISSIVE_OK | VERIFIED-AGREES | — |
| 13 | Dolly 15k | SHARE_ALIKE | VERIFIED-AGREES (version UNVERIFIABLE) | — |
| 15 | Tulu-3 SFT | PERMISSIVE_OK, provisional | CONTRADICTED | **UNVERIFIED as blanket** — card admits NC subsets + third-party-model output subsets exist; needs per-subset split like PMC OA/FLAN, not a spot-check |
| 16 | Alpaca | NC (does not block) | CONTRADICTED | **REFUSE pending legal read** — primary README states "research use only" / "should not be used outside of research purposes," matching the operator's research-only refusal trigger; OpenAI competing-model-output clause confirmed live |
| 17 | UltraChat | PERMISSIVE_OK as declared | VERIFIED-AGREES | — (OpenAI-terms flag now confirmed live, not new verdict) |
| 19 | HH-RLHF | PERMISSIVE_OK (MIT, INFERRED) | VERIFIED-AGREES | MIT now fully VERIFIED (LICENSE text quoted directly), no longer inferred |
| 20 | Common Corpus | mixed PERMISSIVE_OK/SHARE_ALIKE | VERIFIED-AGREES | — |
| 22 | C4 | PERMISSIVE_OK | VERIFIED-AGREES | — |

**Net effect on the composed-corpus consequence**: the survey's "Top 8" list (#1, #2, #6,
#7, #8, #12, #19, #20) does not include Alpaca or Tulu-3-SFT, so the corrected verdicts
above do not change the Top-8 strictest-input conclusion (still SHARE_ALIKE, no
NC/BLOCKING). They do matter for the "second-tier pull" list the survey names at the very
end, which explicitly includes Tulu-3-SFT — that entry should not be pulled at second-tier
without the per-subset split first. Alpaca was never in the Top 8 either, but is listed as
a live NC candidate elsewhere in the survey and should be moved to REFUSE-pending-review
rather than treated as a routine NC admit.
