# Verification of `42-model-card-licence-reading-drafts.md`

**Status:** independent read-only verification of Grok 4.6's unattended draft. Does not modify
`42-...md` or any git repository. Written 2026-09-03.

> **Neither the author of this document nor Grok 4.6, the author of `42-...md`, is a lawyer.**
> Nothing below is a legal conclusion or counsel. It reports whether quoted text matches
> primary sources fetched independently this session, and whether the drafts' internal logic
> is consistent with the project's own ratified documents. The operator should still obtain
> independent legal counsel before shipping any of the three drafts.

**Method.** Every `[V]`-marked quote in `42-...md` was checked against a fresh, independent
fetch of the primary source it cites (live `curl` + HTML-strip for creativecommons.org and
opensource.org pages; `pdftotext -layout` on the CC PDF; raw GitHub fetch for GooAQ). Local
cached copies in `licence-texts/` were used only to locate section numbers quickly, then
cross-checked against a live fetch of the same page. No claim below rests on the cached copies
alone.

---

## 1. Quote-by-quote table

Legend: **CONFIRMED** = found verbatim (or the draft's own quotation marks around a fragment
match the fragment exactly) at the cited URL/section. **MISATTRIBUTED** = the quoted text is
verbatim correct but is not on the page the draft cites it to. **CONDENSED** = the draft puts
quotation marks around text that is a paraphrase/compression of the source, not an exact
substring. **CONFIRMED, CAVEAT** = verbatim correct, but only in one of two co-cited sources.

| # | Draft's claim (short) | Cited source | Verdict | Detail |
|---|---|---|---|---|
| 1 | "is not a law firm and does not provide legal advice" | BY-SA 4.0 legalcode preamble | CONFIRMED | Live page, "About the license and Creative Commons": *"Creative Commons Corporation ("Creative Commons") is not a law firm and does not provide legal services or legal advice."* Draft's phrasing is a faithful compression, not in quote marks. |
| 2 | "Creative Commons does not provide legal advice" | CC FAQ | CONFIRMED | Verbatim, live fetch: *"Creative Commons does not provide legal advice. This FAQ is for informational purposes and is not a substitute for legal advice."* |
| 3 | §3(b): "if You Share Adapted Material You produce" | BY-SA 4.0 §3(b) | CONFIRMED | Verbatim, both cached and live legalcode. |
| 4 | "are triggered only when works or adaptations of works are publicly shared" | cited as **CC FAQ**, `faq/#artificial-intelligence-and-cc-licenses` | **MISATTRIBUTED** | The exact sentence does not appear anywhere on the live FAQ page (checked in full, including its "Artificial intelligence and CC licenses" section, which only discusses new-technology permission and privacy law — no "triggered only" language at all). The sentence **is** verbatim on **`using-cc-licensed-works-for-ai-training-2/`**, "Step 2 – When Do CC License Conditions Apply?": *"The attribution (BY) and ShareAlike (SA) conditions, and NoDerivatives (ND) restriction are triggered only when works or adaptations of works are publicly shared."* Quote text is right; URL/source label is wrong. |
| 5 | §4(b): Adapted Material "including for purposes of Section 3(b)" | BY-SA 4.0 §4(b) | CONFIRMED | Verbatim. |
| 6 | §1(a) Adapted Material definition | BY-SA 4.0 §1(a) | CONFIRMED | Verbatim, full paragraph matches. |
| 7 | §2(a)(4): "simply making modifications authorized by this Section 2(a)(4) never produces Adapted Material" | BY-SA 4.0 §2(a)(4) | CONFIRMED | Verbatim. |
| 8 | §2(a)(5)(A)(B)(C) downstream recipients | BY-SA 4.0 §2(a)(5) | CONFIRMED | Content verbatim on live legalcode; only the (A)/(B)/(C) vs the source page's unlettered bold-lead-in rendering differs, which is a formatting artifact of how the page marks sub-items, not a wording change. |
| 9 | §3(b)(1)(2)(3) full ShareAlike text | BY-SA 4.0 §3(b) | CONFIRMED | Verbatim, all three numbered conditions. |
| 10 | §1(g): BY-SA License Elements = "Attribution and ShareAlike" | BY-SA 4.0 §1(g) | CONFIRMED | Verbatim. |
| 11 | §1(g): BY-NC-SA License Elements = "Attribution, NonCommercial, and ShareAlike" | BY-NC-SA 4.0 §1(g) | CONFIRMED | Verbatim. |
| 12 | BY-SA-compatible list = Free Art License 1.3, GPLv3 (one-way) | CC compatible-licences page | CONFIRMED | Verbatim, live fetch: *"Free Art license 1.3 was declared a 'BY-SA–Compatible License' for version 4.0 on 21 October 2014"*; *"GNU General Public License version 3 was declared a 'BY-SA–Compatible License'... Note that compatibility with the GPLv3 is one-way only."* |
| 13 | "No non-CC licence is designated compatible with BY-NC-SA 4.0" | same page | CONFIRMED | Verbatim: *"Currently, no non-CC licenses have been designated as compatible with BY-NC-SA 4.0."* |
| 14 | "Whether a modification…depends primarily on the applicable copyright law…manifests sufficient new creativity to be copyrightable" | CC FAQ | CONFIRMED | Verbatim, live fetch. |
| 15 | Sui generis database rights → "Adapted Material" under 4.0 | CC FAQ | CONFIRMED | Verbatim. |
| 16 | "If you create a remix with material licensed under a ShareAlike license, you need to make sure…" | CC FAQ | CONFIRMED | Verbatim. |
| 17 | "…you cannot incorporate material released under one of the NonCommercial licenses" | CC FAQ | CONFIRMED | Verbatim, same paragraph as #16. |
| 18 | "In general, when remixing ShareAlike content, your adapter's license must be the same license as the license on the material you are adapting" | CC FAQ | CONFIRMED | Verbatim. |
| 19 | ShareAlike conservative-approach paragraph ("model outputs and the model itself, if shared publicly, should be made available under the same CC license…") | *Using CC-licensed Works for AI Training*, May 2025 PDF + HTML | **CONFIRMED, CAVEAT** | Verbatim **in the PDF** (`pdftotext` extraction matches word-for-word). **Not present on the live HTML page** at the co-cited URL, which was rewritten to a shorter "quick look" summary: *"If AI models or outputs are based on ShareAlike content and they will be shared publicly, following the ShareAlike condition would require AI developers to use the same CC license as the original works."* Same meaning, different wording — a reader who only opens the HTML link will not find the quoted sentence. |
| 20 | "It is worth noting that following this guidance will almost certainly lead to overcompliance…" | same PDF | **CONFIRMED, CAVEAT** | Verbatim in PDF; HTML page instead reads *"following this guidance is likely to lead to overcompliance with both copyright law and CC license terms."* Same caveat as #19. |
| 21 | "This guidance…is not intended to take a position on whether and when copyright applies" | same PDF | **CONFIRMED, CAVEAT** | Verbatim in PDF (page 1); not present on the current HTML page. |
| 22 | "Although, in many cases, neither the AI model nor its outputs would be considered to be derivative works of training data under copyright law" | same PDF | **CONFIRMED, CAVEAT** | Verbatim in PDF. Sits in the **NoDerivatives** paragraph of the primer, not the ShareAlike paragraph — the draft itself discloses this ("stated in the ND paragraph, applying to models generally"), which is an accurate and honest framing, not a mis-citation. |
| 23 | NonCommercial paragraph ("complying with the CC license requires that your use not be 'primarily intended for or directed toward commercial advantage or monetary compensation'…") | same PDF | **CONFIRMED, CAVEAT** | Verbatim in PDF; HTML page's shorter paraphrase covers the same point in different words. |
| 24 | OSAID: "Use the system for any purpose and without having to ask for permission" | OSI Open Source AI Definition 1.0 | CONFIRMED | Verbatim, live fetch. |
| 25 | OSAID: "Data Information: Sufficiently detailed information about the data used to train the system so that a skilled person can build a substantially equivalent system" | same | CONFIRMED | Verbatim. |
| 26 | OSAID: "'Open Source models' and 'Open Source weights' must include the data information and code used to derive those parameters" | same | CONFIRMED | Verbatim. |
| 27 | OSAID: "does not require a specific legal mechanism for assuring that the model parameters are freely available to all" | same | CONFIRMED | Verbatim. |
| 28 | CDLA-Permissive-2.0 §3.1/§5.4 exempt "machine learning models" from the data licence | CDLA-Permissive-2.0 | CONFIRMED | §3.1: *"This agreement does not impose any restriction or obligations with respect to the use, modification, or sharing of Results."* §5.4: *"'Results' means any outcome obtained by computational analysis of Data, including for example machine learning models and models' insights."* Phrase is verbatim. |
| 29 | O-UDA §5.4 exempts "machine learning models" the same way | O-UDA 1.0 | **CONDENSED / imprecise** | O-UDA §5.4 actually reads: *"Artificial intelligence models trained on Data (and which do not include more than a de minimis portion of Data) are Results."* O-UDA's own term is **"Artificial intelligence models,"** not "machine learning models" — the draft applies CDLA-Permissive-2.0's phrase to O-UDA too. Substance (both exempt trained models from the data licence) is correct; the quoted phrase is not O-UDA's actual wording. |
| 30 | GooAQ README: "This dataset should not be used for any commercial purposes" | raw GooAQ README | CONFIRMED | Verbatim, fresh fetch: *"**NOTE** This dataset should not be used for any commercial purposes. See the [license](LICENSE) for the detailed terms."* |
| 31 | GooAQ GitHub `LICENSE` file is stock Apache-2.0 | raw GooAQ LICENSE | CONFIRMED | Fresh fetch confirms `Apache License / Version 2.0, January 2004`. |
| 32 | §3(a) "indicate if You modified" | BY-SA 4.0 §3(a)(1)(b) | CONFIRMED | Verbatim: *"indicate if You modified the Licensed Material and retain an indication of any previous modifications."* |
| 33 | BY-SA 3.0 §4(b)(ii): "later version with the same License Elements" | BY-SA 3.0 legalcode | **CONDENSED** | Actual text: *"(ii) a later version of this License **with the same License Elements as this License**."* The draft's quoted fragment drops "of this License" and "as this License." Meaning is preserved; the quotation marks imply exactness that isn't there. |
| 34 | CC FAQ collections: "You may choose a license for the collection, however this does not change the license applicable to the original material" | CC FAQ | CONFIRMED | Verbatim, live fetch. |
| 35 | FiQA reclassified NC, "The training data is available only for non-commercial use" | catalogue §10.2 / `BeIR/fiqa` entry | CONFIRMED against the catalogue | `DATASET-FACTORY-CATALOGUE-2026-09-03.md` §10.2 does say this, in these words, about `00-ground.md`'s FiQA row. **Not independently traced further**: the live `BeIR/fiqa` Hugging Face card I fetched shows license tag `cc-by-sa-4.0` with no NC field visible; the NC sentence likely comes from the original FiQA-2018 task page, which I did not fetch. The draft already treats this reading as open and states its own conclusion does not depend on which way it resolves — so this is a pre-existing open item, not a new problem the draft introduced. |
| 36 | *Getty Images v. Stability AI* [2025] EWHC 2863 (Ch) ¶599–600, UK secondary infringement, not CC ShareAlike, not US law | `LICENCE-FOR-OPEN-WEIGHTS.md` | CONFIRMED | `LICENCE-FOR-OPEN-WEIGHTS.md` line ~695-704 carries the same citation, same paragraph numbers, same "read narrowly" framing. Internally consistent; I did not independently re-verify the UK judgment text itself (outside this task's primary-source list). |

**Net result:** 32 of 36 checked items are verbatim-confirmed against a fresh independent fetch.
One (item 4) is a real mis-citation (right words, wrong source page). Two (items 29, 33) put
quotation marks around condensed paraphrases rather than exact substrings. Five (items 19–23)
are verbatim only in the PDF version of a document the draft co-cites with an HTML URL that has
since been rewritten to different wording — worth a footnote fix, not a retraction.

---

## 2. Logic checks

### 2.1 Does §1(a) Adapted Material + §3(b) ShareAlike support the cautious reading as stated?

**Yes.** §1(a)'s definition — "derived from or based upon the Licensed Material… in a manner
requiring permission under the Copyright and Similar Rights held by the Licensor" — genuinely
does not resolve whether a trained model is "derived from" its inputs "in a manner requiring
permission." §3(b) is confirmed to fire only "if You Share Adapted Material You produce," so
the cautious reading's conclusions are correctly stated as conditional on that antecedent. The
draft does not overstate this: it presents Reading C as "if weights are Adapted Material, then
…" throughout, never as a flat assertion that they are.

### 2.2 Is the claim that CC BY-NC-SA is not a "BY-SA Compatible License" for adaptations correct?

**Yes, confirmed by direct fetch.** The compatible-licences page's list for adaptations of
BY-SA 4.0 material is closed and enumerated: BY-SA 4.0 or later, ported BY-SA versions, Free
Art License 1.3, and GPLv3 (one-way). BY-NC-SA does not appear on it in any form, and the
page's own structure (separate "BY-SA" and "BY-NC-SA" sections, each with its own closed list)
makes clear these are not interchangeable. This also follows independently from §3(b)(1)
(same License Elements required) and §3(b)(3) (no additional restricting terms) — adding NC to
a BY-SA adaptation's licence does both of the things §3(b) forbids. The draft's claim holds.

### 2.3 Does the permissive reading's premise match what CC itself says about AI training?

**Yes, and the draft represents this honestly rather than cherry-picking.** CC's May 2025
primer states both positions verbatim and explicitly declines to choose ("This guidance…is not
intended to take a position on whether and when copyright applies"). The "many cases…would not
be considered derivative works" sentence the permissive reading leans on is real and verbatim,
but — as confirmed in §1 item 22 above — it sits in the primer's NoDerivatives paragraph, not
its ShareAlike paragraph. The draft discloses this placement itself ("stated in the ND
paragraph, applying to models generally") rather than presenting it as a ShareAlike-specific
statement. That is the correct way to use a quote that is topically adjacent but not exactly
on-point, and it is what the draft does.

---

## 3. Internal consistency against `CogSynDelta/docs/design/LICENCE-FOR-OPEN-WEIGHTS.md` (main)

- **Strictest-input rule.** Confirmed faithful. `LICENCE-FOR-OPEN-WEIGHTS.md`'s own "Decision
  2026-09-02" section contains an unlabeled finding (its item 4, immediately following the
  compatibility matrix) that says, almost word-for-word what the draft's Reading-C analysis
  says: *"The composed model's ratified CC BY-NC-SA 4.0 is coherent only under the permissive
  reading of the weights question… if the position is taken it should be taken explicitly, in
  the model card, as a stated position rather than a silence. That position is currently
  silent."* The draft's whole purpose — writing that stated position into a card — is not just
  consistent with the ratified document, it is the ratified document's own outstanding TODO,
  traceable through `20-enrichment-licence-impact.md` §2.3 → catalogue §10.1 #1 →
  `LICENCE-FOR-OPEN-WEIGHTS.md` item 4, all three of which I read directly and which restate
  the same finding in increasingly compressed form.

- **Rider 1 — minor conflation, no effect on conclusions.** In `LICENCE-FOR-OPEN-WEIGHTS.md`,
  "Rider 1" is specifically about a **future taxonomy merge of regions** (e.g., merging
  `compress` and `retrieve` into one hippocampal region) inheriting the more restrictive
  licence. The rule the draft actually needs for its composed-model contradiction — "the
  composed model's release licence is the strictest term among every dataset, submodel, and the
  composed model itself" — is a separate, adjacent sentence in the same "Decision 2026-09-02"
  section's "Composed model" paragraph, not Rider 1 itself. The draft cites "Rider 1" for both.
  This doesn't change any conclusion (both rules produce the same "most restrictive wins"
  outcome), but a careful reader of `LICENCE-FOR-OPEN-WEIGHTS.md` will notice the citation is
  imprecise.

- **Rider 2 (private-HF-only).** Confirmed correctly cited and applied: the draft's "Datasets
  stay private-HF-only (Rider 2)" matches `LICENCE-FOR-OPEN-WEIGHTS.md` verbatim in substance
  ("training and redistribution stay separate questions… Dataset publication remains
  private-HF-only, unchanged by this decision").

- **The `retrieve`/`memory` region's CC BY-NC-SA tier.** Confirmed: the ratified per-region
  table gives `retrieve` CC BY-NC-SA 4.0 "because GooAQ (NC) and Natural Questions/FiQA (CC
  BY-SA) are both present," which is exactly what the draft states for `retrieve`/`memory`.

- **Per-region repos — the one finding worth flagging clearly.** `LICENCE-FOR-OPEN-WEIGHTS.md`
  (2026-09-02) uses a six-region taxonomy: `code`, `classify`, `reason`, `vl_latent`,
  `compress`, `retrieve`, and treats a `compress`+`retrieve` merge as a hypothetical **not yet
  made** ("an open taxonomy decision, not made here"). The draft (`42-...md`, later the same
  day) follows this taxonomy and keeps `compress` and `retrieve`/`memory` as two separate
  standalone-checkpoint tiers (`compress` → CC BY-SA 4.0, `retrieve`/`memory` → CC BY-NC-SA
  4.0) in both Draft A and Draft B-(i)'s tables.

  But `DATASET-FACTORY-CATALOGUE-2026-09-03.md`, in the **same repository, same day**, already
  uses a **different, seven-faculty** taxonomy (`language_code`, `memory`, `reasoning`,
  `numeric_math`, `visual`, `language_trunk`, `moral_safety`) with no `compress` entry at all,
  and says explicitly: *"`memory` is the faculty the merge made worse, not better. Merging
  `retrieve` and `compress` inherits the union of obligations."* That reads as the taxonomy
  merge — hypothetical in `LICENCE-FOR-OPEN-WEIGHTS.md` — already having happened, at least in
  the catalogue's own accounting. If that merge is real (not just the catalogue's own working
  assumption ahead of a formal decision), then under Rider 1 the merged `memory` region
  inherits the **more** restrictive of the two component licences (CC BY-NC-SA 4.0) for its
  **entire** contents, and there may no longer be a standalone `compress`-only checkpoint that
  can ship at CC BY-SA 4.0 at all. I did not find, in either document, an explicit statement of
  whether this merge has been formally decided or is still catalogue-only bookkeeping. **This
  should be confirmed before either Draft A's or Draft B-(i)'s per-region table is pasted
  anywhere** — the compress/retrieve split those tables depend on may already be stale.

---

## 4. Corrections the drafts need before use

1. **Fix the mis-sourced quote (item 4).** "…are triggered only when works or adaptations of
   works are publicly shared" is not on the CC FAQ page. Re-cite it to
   `https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/` ("Step 2 – When Do
   CC License Conditions Apply?"), or drop the FAQ anchor.
2. **Flag the PDF/HTML split for the AI-training-primer quotes (items 19–23).** All five are
   verbatim in the May 2025 PDF; the co-cited HTML page has been rewritten and no longer carries
   the same sentences. Either cite the PDF alone for these five, or note the HTML has since
   diverged.
3. **Fix the O-UDA attribution (item 29).** O-UDA §5.4 says "Artificial intelligence models,"
   not "machine learning models" — that phrase is CDLA-Permissive-2.0's, not O-UDA's. Quote
   each licence's own term, or drop the quotation marks and paraphrase generically.
4. **Drop the quotation marks around the BY-SA 3.0 §4(b)(ii) fragment (item 33)**, or quote it
   in full: "a later version of this License with the same License Elements as this License."
5. **Confirm current region/faculty taxonomy before pasting Draft A's or Draft B-(i)'s
   per-region table.** `DATASET-FACTORY-CATALOGUE-2026-09-03.md` already treats `compress` and
   `retrieve` as merged into one `memory` faculty; if that merge is decided, `compress` no
   longer exists as an independently-releasable CC BY-SA 4.0 checkpoint under Rider 1, and both
   tables need a row removed or relabeled.
6. **Tighten the "Rider 1" citations** that describe the composed-model strictest-term rule —
   that rule lives in Decision 2026-09-02's "Composed model" paragraph, not Rider 1, which is
   specifically about a future region-taxonomy merge. Same outcome either way; different clause.
7. *(No action needed, informational only.)* The FiQA "NC" quote traces correctly to catalogue
   §10.2, but I did not chase that quote past the catalogue to its own primary source (the
   `BeIR/fiqa` Hugging Face card I fetched shows `cc-by-sa-4.0` with no visible NC field — the
   NC sentence probably comes from the original FiQA-2018 task page, not the HF mirror). The
   draft already treats this as unresolved and states its conclusion doesn't depend on it, so
   this doesn't block anything, but it's not fully closed either.

None of these corrections change which reading is more defensible, and none affects the
draft's central, correctly-sourced finding: the ratified CC BY-NC-SA 4.0 composed-model licence
is coherent only under the permissive reading, and the project's own `LICENCE-FOR-OPEN-WEIGHTS.md`
already says this needs to be stated in the model card rather than left silent.

---

## 5. One-paragraph summary for the operator

Draft A assumes the **permissive** reading (trained weights are not Adapted Material of the
training data) and pastes cleanly against the currently-shipped mix — it forces **nothing** out
of the corpus, only requires the card to say plainly that this reading is being taken. Draft
B-(i) assumes the **cautious** reading (weights are Adapted Material) and forces **dropping
every NC-tagged input** — GooAQ for certain, FiQA if it's read as NC too — before the composed
model can ship as CC BY-SA 4.0; Draft B-(iii) sidesteps the question entirely but forces
dropping **both** the NC and the CC BY-SA inputs (GooAQ, SNLI, Natural Questions, FiQA, SQuAD,
HotpotQA), leaving only the permissive tier, which several faculties don't yet reach training
volume on. Whichever reading is picked, the one sentence every version of the card must contain
is the same: an explicit statement of which reading the release relies on, because the
project's own ratified `LICENCE-FOR-OPEN-WEIGHTS.md` already flags that its current CC
BY-NC-SA 4.0 tag is silently assuming the permissive reading and says that silence should be
closed. Every primary-source quote I independently re-fetched from creativecommons.org,
opensource.org and GooAQ's own GitHub checks out as accurate, with one mis-sourced citation
(right words, wrong URL) and two quotation-marked paraphrases that aren't quite verbatim — all
listed above, none of them changes the drafts' conclusions. The one substantive gap I found is
that the project's own same-day dataset catalogue has apparently already merged the `compress`
and `retrieve` regions into a single `memory` faculty, which may make the drafts' per-region
tables (which still ship `compress` standalone at CC BY-SA 4.0) out of date — worth confirming
before either card is pasted anywhere. **Neither I nor Grok is counsel; nothing in this document
or in `42-...md` is legal advice.**
