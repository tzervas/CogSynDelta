# The licence impact of ENRICHING datasets

**Status:** analysis, for the dataset factory (DEC-57 P2′f, DEC-58 P2′s). It recommends
mechanics and states verdicts about licence *text*; it makes no legal conclusion and
changes no catalogue entry.

**Scope:** what happens to licence obligations when the factory takes a licence-compatible
dataset and *enriches* it — cleans, deduplicates, reformats, pairs rows across datasets,
adds rationales or labels, augments synthetically with a named generator, renders text to
images, or translates — and then trains a submodel on the result and releases weights.

> **Neither the author nor the reader of this document is a lawyer.** Everything below
> reports what a licence *says*, quoted from the primary text, and what follows
> mechanically. Section 5 is a register of the questions that need a human — in several
> cases a lawyer — and it is deliberately long.

**Marking convention, held to throughout.**
**VERIFIED** = a licence text fetched from its primary source on **2026-09-03** and quoted
from the file, or a live API response read this session. The fetched texts are kept beside
this document at `/akula-data/session-backup-staging/dataset-factory/licence-texts/`.
**INFERRED** = this document's reasoning over those texts. Every conclusion that combines
two clauses is INFERRED even when both clauses are VERIFIED.

**Ground this rests on:** `00-ground.md` (this directory),
`CogSynDelta/docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`,
`CogSynDelta/docs/design/AUDIO-CORPUS-AUDIT.md`, DEC-31 / DEC-46 / DEC-57 / DEC-58,
and the operator's licence stance of 2026-09-02/03.

---

## Primary sources fetched this session (all 2026-09-03)

| licence / terms | URL fetched | local copy |
|---|---|---|
| CC BY 4.0 | `https://creativecommons.org/licenses/by/4.0/legalcode.txt` | `cc-by-4.0.plain.txt` |
| CC BY-SA 4.0 | `https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt` | `cc-by-sa-4.0.plain.txt` |
| CC BY-SA 3.0 | `https://creativecommons.org/licenses/by-sa/3.0/legalcode.txt` | `cc-by-sa-3.0.plain.txt` |
| CC BY-NC 4.0 | `https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt` | `cc-by-nc-4.0.plain.txt` |
| CC BY-NC-SA 4.0 | `https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt` | `cc-by-nc-sa-4.0.plain.txt` |
| CC0 1.0 | `https://creativecommons.org/publicdomain/zero/1.0/legalcode.txt` | `cc0-1.0.plain.txt` |
| CC FAQ (adaptation / collection / AI / datasets) | `https://creativecommons.org/faq/` | `cc-faq.txt` |
| CC BY-SA compatible licences | `https://creativecommons.org/share-your-work/licensing-considerations/compatible-licenses/` | (WebFetch, quoted below) |
| ODC-By 1.0 | `https://opendatacommons.org/licenses/by/1-0/` | `odc-by-1.0.txt` |
| ODbL 1.0 | `https://opendatacommons.org/licenses/odbl/1-0/` | `odbl-1.0.txt` |
| CDLA-Permissive-2.0 | `https://cdla.dev/permissive-2-0/` | `cdla-permissive-2.0.txt` |
| CDLA-Permissive-1.0 | `https://cdla.dev/permissive-1-0/` | `cdla-permissive-1.0.txt` |
| CDLA-Sharing-1.0 | `https://cdla.dev/sharing-1-0/` | `cdla-sharing-1.0.txt` |
| O-UDA 1.0 | `https://cdla.dev/open-use-of-data-agreement-v1-0/` | `o-uda-1.0.txt` |
| Apache-2.0 | `https://www.apache.org/licenses/LICENSE-2.0.txt` | `apache-2.0.txt` |
| GPL-3.0 | `https://www.gnu.org/licenses/gpl-3.0.txt` | `gpl-3.0.txt` |
| Gemma Terms of Use | `https://ai.google.dev/gemma/terms` | (WebFetch, quoted below) |
| Llama 3 Community Licence | `https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3/LICENSE` | (WebFetch, quoted below) |
| Llama 3.1 Community Licence | `https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE` | (WebFetch, quoted below) |
| Anthropic Commercial Terms | `https://www.anthropic.com/legal/commercial-terms` | `anthropic-commercial-tos.txt` |
| OpenAI Foundation Terms of Use | `https://openaifoundation.org/terms-of-use` | (WebFetch, quoted below) |

**Two fetch failures, recorded honestly.** `openai.com/policies/row-terms-of-use/`,
`/terms-of-use/`, `/business-terms/` and `/education-terms/` returned **HTTP 403** to direct
fetch this session (and the HTML retrieved by `curl` was a JavaScript/cookie interstitial,
not the terms). The OpenAI clause quoted in §3 is therefore taken from
`openaifoundation.org/terms-of-use`, which **was** fetched successfully and carries the same
formulation with "OpenAI Foundation" substituted for "OpenAI". Treat the exact corporate
name in that clause as **INFERRED** and the substance as VERIFIED. `meta-llama/*` on
huggingface.co returned **401** (gated repo), so the Llama texts come from Meta's own GitHub
mirror — a primary source, but not the `llama.com` canonical page, which is
JavaScript-rendered and produced no text.

---

## 0. The frame: enrichment splits the problem into two layers, and only one of them is unsettled

`LICENCE-FOR-OPEN-WEIGHTS.md` established one open question and returned to it repeatedly:
**are trained weights "Adapted Material" / a derivative work of the training data?** That
question is still open and nothing in this document closes it.

**Enrichment introduces a second, separate question that sits *below* it, and that one is
largely answered by licence text rather than by litigation:**

```
  input dataset(s)
        │
        │  ── LAYER 1: enrichment ──►  emitted (enriched) dataset
        │        clean, dedup, reformat, pair, label, augment, render, translate
        │        ANSWERED by express clauses in CC 4.0 §4(b), ODbL §4.4(b),
        │        ODC-By §4.2(a), CDLA-Sharing §1.8/§3.1
        │
        └──────────────────────────►  ── LAYER 2: training ──►  weights
                                          UNSETTLED for CC/GPL;
                                          ANSWERED favourably by CDLA §3.x,
                                          O-UDA/C-UDA §5.4; ambiguous for ODC/ODbL
```

**This is the single most useful thing in this document, so it is stated first.** The
factory's compliance must be **decidable at Layer 1**, because Layer 1 *is* decidable and
Layer 2 is not. Concretely:

- A share-alike obligation on an **enriched dataset** is not a legal opinion; several
  licences say in terms that a filtered, re-arranged or joined database *is* the adapted
  or derivative database. There is no creativity threshold to argue about.
- A share-alike obligation on **weights** depends on the open question. The factory should
  never depend on the answer; it should make both answers survivable by keeping tiers
  separable (§4.6).

**Corollary that bites immediately:** the cheapest enrichment operation the factory can
perform — dropping rows — already triggers the derivative-database clause of every
share-alike data licence surveyed. Enrichment is not a licence-neutral cleanup step.

---

## 1. Per licence family

For each family: (a) is the enriched dataset a derivative/adapted work; (b) what licence the
enriched dataset must carry; (c) whether it can be mixed with other families; (d) whether
training on it constrains the weights; (e) attribution obligations and how to discharge them
mechanically.

### 1.0 First, the enrichment operations, since "is it a derivative work" is per-operation

**VERIFIED, CC BY 4.0 §1(a) (identical in BY-SA/BY-NC/BY-NC-SA 4.0):**

> *"Adapted Material means material subject to Copyright and Similar Rights that is derived
> from or based upon the Licensed Material and in which the Licensed Material is translated,
> altered, arranged, transformed, or otherwise modified in a manner requiring permission
> under the Copyright and Similar Rights held by the Licensor."*

**VERIFIED, CC BY 4.0 §4 (Sui Generis Database Rights) — the clause that does the real work
for a dataset factory:**

> *"Where the Licensed Rights include Sui Generis Database Rights that apply to Your use of
> the Licensed Material: … b. if You include all or a substantial portion of the database
> contents in a database in which You have Sui Generis Database Rights, then the database in
> which You have Sui Generis Database Rights (but not its individual contents) is Adapted
> Material; and c. You must comply with the conditions in Section 3(a) if You Share all or a
> substantial portion of the contents of the database."*

**VERIFIED, CC BY-SA 4.0 §4(b)** is the same sentence with five extra words that decide the
question: *"…is Adapted Material, **including for purposes of Section 3(b)**"* — §3(b) being
ShareAlike. So CC BY-SA's own text says an enriched database of BY-SA content is subject to
ShareAlike. No inference needed.

**VERIFIED, CC FAQ (`creativecommons.org/faq/`, "When is my use considered an adaptation?"):**

> *"Note that all CC licenses allow the user to exercise the rights permitted under the
> license in any format or medium. Those changes are not considered adaptations even if
> applicable law would suggest otherwise."*

and

> *"Also, under version 4.0, certain uses of databases restricted by sui generis database
> rights also constitute adaptations (called 'Adapted Material' in the 4.0 licenses),
> whether or not they would be considered adaptations under copyright law."*

**Operation × "is the output a derivative work" (INFERRED from the clauses above):**

| enrichment operation | copyright-law adaptation? | CC 4.0 sui-generis DB adaptation? | ODbL / ODC-By "Derivative Database"? | CDLA-Sharing "Enhanced Data"? |
|---|---|---|---|---|
| **filter / clean / dedup / cap-sample (B4)** | probably not — deletion adds no creativity | **yes**, if a substantial portion is retained (§4(b)) | **yes** — ODbL §4.4(b) *"Extraction or Re-utilisation of the whole or a Substantial part of the Contents into a new database is a Derivative Database"* (VERIFIED) | **yes** — §1.8 *"'Modify' means to delete, erase, correct or re-arrange Data"* (VERIFIED) |
| **reformat / re-column / re-serialise** | no — "any format or medium" (VERIFIED FAQ) | yes, same as above | yes | yes (§1.8 "re-arrange") |
| **pair rows across two datasets** (query⋈passage, premise→hypothesis) | **yes** — a remix; CC FAQ treats combining as the case where "you must pay attention to the particular license" | yes | yes | yes, plus §1.1 "Add"/"Additions" |
| **add rationales, labels, annotations** | **yes** — new copyrightable contribution over the original | yes | yes (Additions) | yes (Additions) |
| **synthetic augmentation from a named generator** | **yes**, and a *second* licence stack (the generator's) now attaches — see §3 | yes | yes | yes |
| **render text to images** | ambiguous — a medium change (not an adaptation per the FAQ) but typography/layout are creative choices | yes | yes | yes |
| **translate** | **yes, expressly** — §1(a) names *"translated"* first | yes | yes | yes |

**Practical rule (INFERRED):** for licence purposes the factory should treat **every**
emitted dataset as a derivative/adapted database of every input that contributed a
substantial share of rows. The one place the distinction still matters is jurisdiction (§5.3):
CC 4.0 §4 applies only *"Where the Licensed Rights include Sui Generis Database Rights that
apply to Your use"*, and a US-only operator may have none — which would make a
filter-only output arguably **not** Adapted Material. That is a lawyer question, and the
cheap engineering answer is not to depend on it.

---

### 1.1 CC0 / public domain

**(a) Derivative?** Irrelevant — CC0 waives the rights that would make it matter.
**(b) Output licence:** anything, including MIT, CC BY, CC BY-NC-SA, or a closed dataset.
**(c) Mixing:** unrestricted; CC0 rows never constrain the emitted licence.
**(d) Weights:** unconstrained.
**(e) Attribution:** none required. Citation is a norm, not an obligation.

**The trap, and it is a real one — VERIFIED, CC0 §4:**

> *"a. No trademark or patent rights held by Affirmer are waived, abandoned, surrendered,
> licensed or otherwise affected by this document."*
> *"c. Affirmer disclaims responsibility for clearing rights of other persons that may apply
> to the Work or any use thereof, including without limitation any person's Copyright and
> Related Rights in the Work. Further, Affirmer disclaims responsibility for obtaining any
> necessary consents, permissions or other rights required for any use of the Work."*

**(INFERRED)** §4(c) is a *disclaimer of ownership in the CC0 wrapper itself*. A CC0 tag on a
mirror of somebody else's content is therefore **not evidence that the content is clear** —
it is the same failure mode as the AudioSet family in `AUDIO-CORPUS-AUDIT.md`, wearing the
most permissive tag in the ecosystem. This is the textual basis for the operator's REFUSE
rule against *"distributors that disclaim owning what they distribute"*, and it means a CC0
mirror still requires the upstream check, exactly like any other mirror. It is also the
textual root of the recommended `CONSENT_OPEN` class (OD-11): CC0 addresses copyright and
says nothing about consent, privacy or personality rights, which is precisely the shape of a
revocable-consent corpus.

---

### 1.2 Permissive: MIT / Apache-2.0 / BSD / ODC-By / CDLA-Permissive

These are three different animals wearing one label, and the factory must not treat them
alike.

#### 1.2a MIT / Apache-2.0 / BSD applied to *data*

**(a) Derivative?** These are software licences; applied to a dataset they carry their own
notion of "Derivative Works". Apache-2.0 §1 defines it for the Work; a filtered corpus is
plainly within it.
**(b) Output licence:** anything, provided the notices ride along.
**(c) Mixing:** unrestricted — permissive rows never force the emitted licence.
**(d) Weights:** unconstrained. Neither licence says anything about outputs or models.
**(e) Attribution — VERIFIED, Apache-2.0 §4:**

> *"(b) You must cause any modified files to carry prominent notices stating that You changed
> the files; and (c) You must retain, in the Source form of any Derivative Works that You
> distribute, all copyright, patent, trademark, and attribution notices from the Source form
> of the Work…; and (d) If the Work includes a 'NOTICE' text file as part of its
> distribution, then any Derivative Works that You distribute must include a readable copy of
> the attribution notices contained within such NOTICE file…"*

**Mechanically:** copy each input's `NOTICE` (verbatim, if one exists) into the emitted
dataset's attribution manifest; set a `modified: true` flag with the operation chain (§4.2).
§4(b) — "prominent notices stating that You changed the files" — is satisfied by the
`operations[]` list in the manifest plus a header line in each emitted shard.

#### 1.2b ODC-By 1.0 — **not** as permissive as its name, at the dataset layer

**VERIFIED, ODC-By §4.2:**

> *"Notices. If You Publicly Convey this Database, any Derivative Database, or the Database
> as part of a Collective Database, then You must: a. **Do so only under the terms of this
> License**; b. Include a copy of this License or its Uniform Resource Identifier (URI) with
> the Database or Derivative Database…; c. Keep intact any copyright or Database Right
> notices…"*

**VERIFIED, ODC-By §4.4:** *"…You may not impose any further restrictions on the exercise of
the rights granted or affirmed under this License."*

**(a) Derivative?** Yes — filtering alone makes a Derivative Database.
**(b) Output licence:** **ODC-By 1.0.** (INFERRED from §4.2(a).) Despite the name,
ODC-By is a *database-level copyleft*: a derivative database may be conveyed only under
ODC-By. This is regularly missed and it means ODC-By cannot be silently rolled into a
CC BY tier.
**(c) Mixing:** **cannot** be conveyed as part of an NC dataset (§4.4 "no further
restrictions"), and cannot be conveyed under CC BY-SA (different licence, not a designated
compatible licence in either direction). It *can* sit beside those tiers as a separate file
in a Collection.
**(d) Weights — VERIFIED, ODC-By §4.3:**

> *"Notice for using output (Contents). Creating and Using a Produced Work does not require
> the notice in Section 4.2. However, if you Publicly Use a Produced Work, You must include a
> notice associated with the Produced Work reasonably calculated to make any Person that uses,
> views, accesses, interacts with, or is otherwise exposed to the Produced Work aware that
> Content was obtained from the Database … and that it is available under this License."*
> Example notice: *"Contains information from DATABASE NAME which is made available under the
> ODC Attribution License."*

If a model counts as a Produced Work, a **notice in the model card is the whole obligation**
— no licence propagation. See §5.4 for why "if" is doing real work in that sentence.

**(e) The grant-scope trap — VERIFIED, ODC-By §2.4:**

> *"The individual items of the Contents contained in this Database may be covered by other
> rights, including copyright, patent, data protection, privacy, or personality rights, and
> this License does not cover any rights (other than Database Rights or in contract) in
> individual Contents contained in the Database."*
> *"For example, if used on a Database of images (the Contents), this License would not apply
> to copyright over individual images, which could have their own separate licenses…"*

**(INFERRED, and it matters most for `visual`.)** An `odc-by` tag on an image or text corpus
licenses the *collection*, not the *contents*. The factory's `grant_scope` enum must
therefore carry a value the current design does not have: **`database_rights_only`**, sitting
alongside `whole_corpus / metadata_only / code_only / unstated`. A corpus tagged ODC-By whose
contents are third-party works is, at the content layer, exactly as unlicensed as
tiny-imagenet. This is the same class of error as AudioSet's `metadata_only`, and ODC-By
makes it explicit in the licence text, so the fetcher can be made to catch it.

#### 1.2c CDLA-Permissive-2.0 — the best-case family, and the model question is *settled* in it

**VERIFIED, CDLA-Permissive-2.0 §2.1, §3.1, §5.4:**

> *"2.1. A Data Recipient may share Data, with or without modifications, so long as the Data
> Recipient makes available the text of this agreement with the shared Data."*
> *"3. No Restrictions on Results — 3.1. This agreement does not impose any restriction or
> obligations with respect to the use, modification, or sharing of Results."*
> *"5.4. 'Results' means any outcome obtained by computational analysis of Data, including for
> example **machine learning models** and models' insights."*

(CDLA-Permissive-**1.0** carries the same substance in different numbering — VERIFIED §1.10
defines Results, §3.4: *"This Agreement imposes no obligations or restrictions on Your Use or
Publication of Results."*)

**(a) Derivative?** Yes, but the licence does not care — modifications may be shared.
**(b) Output licence:** any, provided the agreement text travels with the data.
**(c) Mixing:** unrestricted.
**(d) Weights:** **expressly unconstrained.** This is the only family surveyed where the
Layer-2 question has a written answer.
**(e) Attribution:** ship the agreement text; no TASL required.

**The same is true of the O-UDA and C-UDA family — VERIFIED, O-UDA 1.0 §5.4:**

> *"'Result' means anything that you develop or improve from your use of Data that does not
> include more than a de minimis portion of the Data on which the use is based … **Artificial
> intelligence models trained on Data (and which do not include more than a de minimis
> portion of Data) are Results.**"*
> §2.1: *"…this agreement does not impose any restriction or obligations with respect to … 2.1.2.
> the use, modification, or distribution of Results."*

**Sourcing consequence (INFERRED, and worth acting on):** when the factory has a choice of
mirrors or of comparable corpora, a CDLA-Permissive / O-UDA / C-UDA source is *strictly
better* than a CC BY one, because it removes a legal uncertainty rather than merely being
cheap to comply with. That should be a tiebreak rule in the surveyor's ranking, not an
afterthought.

---

### 1.3 CC BY (attribution)

**(a) Derivative?** Yes, by §1(a) and/or §4(b) as tabulated in §1.0.
**(b) Output licence:** CC BY 4.0, CC BY-SA 4.0, CC BY-NC 4.0 or CC BY-NC-SA 4.0. **Not**
CC0 — you cannot waive rights you do not hold. **(INFERRED** from CC BY §3(a)(4) plus the CC
FAQ; the FAQ's adapter's-licence *chart* is an image and could not be read this session, so
the per-cell colour coding is INFERRED from the prose.)

**VERIFIED, CC BY 4.0 §3(a)(4):**

> *"If You Share Adapted Material You produce, the Adapter's License You apply must not
> prevent recipients of the Adapted Material from complying with this Public License."*

**VERIFIED, CC FAQ:**

> *"When remixing BY or BY-NC material, it is generally recommended that your adapter's
> license include at least the same license elements as the license applied to the original
> material. This eases reuse for downstream users because they are able to satisfy both
> licenses by complying with the adapter's license."*

**(c) Mixing:** freely mixable into any tier. CC BY is the universal donor of the
non-permissive families.
**(d) Weights:** attribution attaches *if* weights are Adapted Material — unsettled — but the
obligation is a notice, so the cost of complying anyway is a paragraph. Comply anyway.
**(e) Attribution — VERIFIED, CC BY 4.0 §3(a)(1):**

> *"…You must: a. retain the following if it is supplied by the Licensor with the Licensed
> Material: i. identification of the creator(s)…; ii. a copyright notice; iii. a notice that
> refers to this Public License; iv. a notice that refers to the disclaimer of warranties;
> v. a URI or hyperlink to the Licensed Material to the extent reasonably practicable;
> b. **indicate if You modified the Licensed Material and retain an indication of any previous
> modifications**; and c. indicate the Licensed Material is licensed under this Public
> License, and include the text of, or the URI or hyperlink to, this Public License."*

and §3(a)(2), which is what makes a manifest workable:

> *"You may satisfy the conditions in Section 3(a)(1) in any reasonable manner based on the
> medium, means, and context in which You Share the Licensed Material. For example, it may be
> reasonable to satisfy the conditions by providing a URI or hyperlink to a resource that
> includes the required information."*

**Confirmed directly for the dataset case — VERIFIED, CC FAQ, "What attribution obligations
exist when CC-licensed images are included in a published dataset?":**

> *"Where a CC-licensed work is distributed as part of a database or dataset, and assuming
> copyright (or in the European Union, copyright or sui generis database rights) is
> triggered, then the license conditions must be respected. This means providing the required
> attribution information in a way that is reasonable under the circumstances. Our licenses
> allow for some flexibility, and in some cases that may be as simple as providing a link to
> the website where the relevant attribution information is provided."*

**Mechanically:** an `attribution.json` + `ATTRIBUTION.md` pair per emitted dataset, linked
from the model card, discharges §3(a)(1) under §3(a)(2). **The `b` clause —
"indicate if You modified" — is the one the existing draft notices in
`LICENCE-FOR-OPEN-WEIGHTS.md` do not yet satisfy** (that document flags this itself), and it
is exactly what the manifest's `operations[]` list is for. Enrichment *increases* this
obligation: the more the factory transforms, the more there is to declare.

**One obligation the factory must be able to refuse (INFERRED, restating the audit's own
finding):** where a source's attribution requirement is **per-item** — Stack Exchange's
requirement of a hyperlink to each original question and each author's profile is the
worked example in `LICENCE-FOR-OPEN-WEIGHTS.md` — a dataset-level manifest cannot satisfy it,
at any scale, because there is no per-row link. That should be a structural REFUSE at ingest
(§4.5 R9), not a caveat recorded beside an emitted dataset.

---

### 1.4 Share-alike: CC BY-SA / ODbL / CDLA-Sharing

**These three do not interoperate with each other.** That is the finding; everything else in
this subsection is detail.

#### 1.4a CC BY-SA 4.0

**(a) Derivative?** **Yes, expressly, for databases** — §4(b), quoted in §1.0, says the
enriched database *is* Adapted Material *"including for purposes of Section 3(b)"*.
**(b) Output licence — VERIFIED, CC BY-SA 4.0 §3(b):**

> *"In addition to the conditions in Section 3(a), if You Share Adapted Material You produce,
> the following conditions also apply. 1. The Adapter's License You apply must be a Creative
> Commons license with the same License Elements, this version or later, or a BY-SA
> Compatible License. 2. You must include the text of, or the URI or hyperlink to, the
> Adapter's License You apply… 3. **You may not offer or impose any additional or different
> terms or conditions on, or apply any Effective Technological Measures to, Adapted Material
> that restrict exercise of the rights granted under the Adapter's License You apply.**"*

So: **CC BY-SA 4.0** (or a later BY-SA version, or a designated BY-SA Compatible License).

**Version mixing is settled, and this closes an open item in `LICENCE-FOR-OPEN-WEIGHTS.md`.**
That document flagged the CC BY-SA 3.0 → 4.0 upgrade as *"should be confirmed against CC's
current compatible-licences list before being relied on, not assumed from this document."*
It can now be relied on, from the 3.0 text itself rather than from a list —
**VERIFIED, CC BY-SA 3.0 §4(b):**

> *"You may Distribute or Publicly Perform an Adaptation only under the terms of: (i) this
> License; (ii) **a later version of this License with the same License Elements as this
> License**; (iii) a Creative Commons jurisdiction license (either this or a later license
> version) that contains the same License Elements as this License…; (iv) a Creative Commons
> Compatible License."*

**(INFERRED, high confidence):** an enriched dataset mixing CC BY-SA 3.0 inputs (Natural
Questions) with CC BY-SA 4.0 inputs (SNLI, SQuAD, HotpotQA, FiQA, oxford-iiit-pet) may be
emitted uniformly under **CC BY-SA 4.0**, by 3.0 §4(b)(ii)'s own "later version" permission.
The reverse — 4.0 content into a 3.0 dataset — is not permitted.

**The designated compatible licences are a very short list — VERIFIED**
(`creativecommons.org/share-your-work/licensing-considerations/compatible-licenses/`, fetched
2026-09-03): **Free Art License 1.3** (declared 2014-10-21) and **GNU GPL v3** (declared
2015-10-08), and for GPLv3 the page states:

> *"compatibility with the GPLv3 is one-way only, which means you may license your
> contributions to adaptations of BY-SA 4.0 materials under GPLv3, but you may not license
> your contributions to adaptations of GPLv3 projects under BY-SA 4.0."*

The same page states that **no non-CC licence has been designated compatible with BY-SA 3.0.**

**(c) Mixing:** BY-SA can absorb CC0, PD, MIT/Apache/BSD and CC BY. It **cannot** be mixed
into a dataset emitted under CC BY, CC BY-NC, CC BY-NC-SA, ODbL, ODC-By or CDLA-Sharing.
**(d) Weights:** unsettled — unchanged from `LICENCE-FOR-OPEN-WEIGHTS.md`. Note the asymmetry
that enrichment exposes: CC's own text settles the **database** question (§4(b)) and is
silent on the **model** question. Do not let the settled half be read as settling the other.
**(e) Attribution:** §3(a) as for CC BY, plus §3(b)(2) (include the adapter's licence).

#### 1.4b ODbL 1.0 — the model question is answered favourably, and then a different clause bites

**VERIFIED, ODbL §4.4(a)-(c):**

> *"a. Any Derivative Database that You Publicly Use must be only under the terms of: i. This
> License; ii. A later version of this License similar in spirit to this License; or iii. A
> compatible license."*
> *"b. For the avoidance of doubt, Extraction or Re-utilisation of the whole or a Substantial
> part of the Contents into a new database is a Derivative Database and must comply with
> Section 4.4."*
> *"c. Derivative Databases and Produced Works. A Derivative Database is Publicly Used and so
> must comply with Section 4.4. **if a Produced Work created from the Derivative Database is
> Publicly Used.**"*

**VERIFIED, ODbL §4.5(b)-(c) — the carve-out:**

> *"b. Using this Database, a Derivative Database, or this Database as part of a Collective
> Database to create a Produced Work does not create a Derivative Database for purposes of
> Section 4.4; and c. Use of a Derivative Database internally within an organisation is not to
> the public and therefore does not fall under the requirements of Section 4.4."*

**VERIFIED, ODbL §4.6 — the clause that must not be missed:**

> *"Access to Derivative Databases. If You Publicly Use a Derivative Database or a Produced
> Work from a Derivative Database, You must also offer to recipients of the Derivative
> Database or Produced Work a copy in a machine readable form of: a. The entire Derivative
> Database; or b. A file containing all of the alterations made to the Database or the method
> of making the alterations to the Database (such as an algorithm), including any additional
> Contents, that make up all the differences between the Database and the Derivative
> Database. The Derivative Database (under a.) or alteration file (under b.) must be available
> at no more than a reasonable production cost for physical distributions and free of charge
> if distributed over the internet."*

**(INFERRED, and this is a hard operational conflict with existing project policy.)** Chain
the clauses: the factory enriches an ODbL input → §4.4(b) that is a Derivative Database →
the operator publishes weights → *if* the weights are a Produced Work, §4.4(c) makes the
Derivative Database "Publicly Used" → §4.4(a) it must be licensed ODbL **and** §4.6 the
operator must **offer recipients the entire enriched dataset, or a machine-readable
alteration file / the algorithm, free of charge over the internet.**

DEC-31's Rider 2 records that **"dataset publication remains private-HF-only."** An ODbL
input makes those two positions mutually exclusive. **The factory must therefore treat ODbL
as a decision, not an ingest:** either the enriched ODbL-derived dataset is published openly
under ODbL, or ODbL inputs are refused. There is no third option that ships weights. This
belongs in the REFUSE path as a structural check (§4.5 R8), because it is invisible at the
row level and only appears when publication intent is combined with input licence.

**(b) Output licence:** ODbL 1.0. **(c) Mixing:** with CC0/PD/permissive only; not with CC
BY-SA, not with NC (§4.7(a) forbids terms that restrict the granted rights). **(d) Weights:**
notice only under §4.3, *conditioned on a model being a "Produced Work"* — see §5.4.
**(e) Attribution:** §4.3's example notice, verbatim, per input.

#### 1.4c CDLA-Sharing-1.0 — share-alike that is *conditional on publishing*

**VERIFIED, CDLA-Sharing-1.0 §3.1, §3.3, §3.5:**

> *"3.1 If You Publish Data You Receive or Enhanced Data: (a) The Data (including the Enhanced
> Data) must be Published under this Agreement in accordance with this Section 3; and (b) You
> must cause any Data files containing Enhanced Data to carry prominent notices that You have
> changed those files; and (c) If You Publish Data You Receive, You must preserve all credit
> or attribution to the Data Provider(s)…"*
> *"3.3 … You may not modify this Agreement or impose any further restrictions on the exercise
> of the rights granted under this Agreement, **including by adding any restriction on
> commercial or non-commercial Use of Data (including Your Enhanced Data)** or by limiting
> permitted Use of such Data to any particular platform, technology or field of endeavor."*
> *"3.5 This Agreement imposes no obligations or restrictions on Your Use or Publication of
> Results."*

with **VERIFIED §1.5**: *"'Enhanced Data' means the subset of Data that You Publish and that
is composed of (a) Your Additions and/or (b) Modifications to Data You have received"*, and
**§1.11**: *"'Results' means the outcomes or outputs that You obtain from Your Computational
Use of Data."*

**(a) Derivative?** Yes — filtering is "Modify" (§1.8).
**(b) Output licence:** CDLA-Sharing-1.0, unmodified, **but only if the enriched dataset is
published**. Private enrichment triggers nothing (§3.1 opens with *"If You Publish"*).
**(c) Mixing:** §3.3 is explicit — **CDLA-Sharing data can never be placed in an NC dataset**,
even the operator's own Enhanced Data. Nor in a CC BY-SA one (different agreement).
**(d) Weights:** **expressly unconstrained** (§3.5).
**(e) Attribution:** §3.1(b) file-level "changed" notices + §3.1(c) preserve provider credit.

**(INFERRED)** CDLA-Sharing is the *friendliest* share-alike family for this project's shape:
open weights, private datasets. Its obligations attach only on publication of data, and it
disclaims any reach into the model. It is a better sourcing target than CC BY-SA, and the
surveyor should rank it accordingly.

---

### 1.5 Non-commercial: CC BY-NC / CC BY-NC-SA, and bespoke NC terms

**VERIFIED, CC BY-NC 4.0 §1(i):**

> *"NonCommercial means not primarily intended for or directed towards commercial advantage
> or monetary compensation. For purposes of this Public License, the exchange of the Licensed
> Material for other material subject to Copyright and Similar Rights by digital file-sharing
> or similar means is NonCommercial provided there is no payment of monetary compensation in
> connection with the exchange."*

**(a) Derivative?** Same analysis as CC BY / CC BY-SA per §1.0.
**(b) Output licence:** CC BY-NC 4.0 (from a BY-NC input) or **CC BY-NC-SA 4.0** (from a
BY-NC-SA input — §3(b) of BY-NC-SA has the same "same License Elements" requirement as BY-SA).
**(c) Mixing:** NC absorbs CC0/PD/permissive/CC BY. It **cannot** be mixed with CC BY-SA,
ODbL, ODC-By or CDLA-Sharing — each of those forbids adding restrictions.
**(d) Weights:** per DEC-31, an NC input moves the region and the composed model to an
NC-family release. That decision is unchanged here.
**(e) One point enrichment makes sharper (INFERRED):** the NC condition restricts **use**,
not merely sharing. Exercising the Licensed Rights — reproducing rows into an enriched
dataset, running a training pass over them — must itself be non-commercial. For an
open-weights research project that is satisfied on its face (and is the operator's stated
reasoning in DEC-31), but it is a different statement from "the release is NC", and the
model card should say the training was non-commercial, not only that the licence is.

**Bespoke NC (the GooAQ shape) is not CC BY-NC and must not be recorded as if it were.**
GooAQ's restriction is a sentence in a README over a stock Apache-2.0 LICENSE file. There is
no adapter's-licence machinery, no defined "NonCommercial", no compatible-licence list.
**(INFERRED)** The factory should keep `licence_upstream` as the verbatim contradictory pair
and set `verdict: NC` with a `redistribute` flag of `NC`, while recording that the emitted
dataset's own CC BY-NC-SA tag is **the operator's chosen expression of a term whose scope is
uncertain**, not a licence the rights-holder granted. Rider 2 already keeps redistribution of
GooAQ-derived rows private; that stays right and this analysis strengthens the reason for it.

---

### 1.6 GPL-family source code as dataset content

**(a) Derivative?** A corpus of GPL'd source files is a distribution of the programs
themselves. Filtering and re-pairing them does not change that.
**(b) Output licence:** for those rows, GPL. **VERIFIED, GPL-3.0 §5 (aggregation):**

> *"…which are not by their nature extensions of the covered work, and which are not combined
> with it such as to form a larger program, in or on a volume of a storage or distribution
> medium, is called an 'aggregate' if the compilation and its resulting copyright are not
> used to limit the access or legal rights of the compilation's users beyond what the
> individual works permit. Inclusion of a covered work in an aggregate does not cause this
> License to apply to the other parts of the aggregate."*

**(INFERRED)** The aggregation clause helps a *distribution medium* containing unrelated
works; it does not help a corpus in which the GPL'd code **is** the payload of the rows being
learned from. A dataset row whose content is a GPL function is not "aggregated with" the
dataset — it is the dataset.
**(c) Mixing:** GPLv3 is a one-way BY-SA-compatible licence (VERIFIED above): contributions
to adaptations of BY-SA 4.0 may be GPLv3, **not** the reverse. So GPL rows cannot be folded
into a CC BY-SA tier. They cannot be folded into an NC tier either (GPL forbids added
restrictions).
**(d) Weights:** the same unsettled question as CC BY-SA, in a stronger form — see the SFC
position quoted in `LICENCE-FOR-OPEN-WEIGHTS.md`.
**(e) The only mechanism that actually works** is the one the audit already recommends and
BigCode already executed: **per-row licence filtering to permissive repositories**, using a
surviving per-row licence/`repo` column. This is why §4.2's per-row provenance is not
bureaucracy — for `language_code` it is the *only* compliance mechanism available, and the
`all-nli` genre-column loss is the worked example of what happens when it is dropped.

---

### 1.7 Model-output terms (OpenAI / Anthropic / Meta Llama / Google Gemma)

**These are not copyright licences.** They are contract terms binding whoever accepted them,
and they operate whether or not copyright subsists in the outputs. Two distinct shapes, and
the difference decides whether the factory may use the generator at all.

#### Shape A — the term binds *the operator*, and does not reach the artefact

**VERIFIED, Anthropic Commercial Terms §B and §D.4** (`anthropic.com/legal/commercial-terms`,
fetched 2026-09-03):

> *"Anthropic agrees that Customer (a) retains all rights to its Inputs, and (b) owns its
> Outputs. Anthropic disclaims any rights it receives to the Customer Content under these
> Terms."*
> *"D.4. Use Restrictions. Customer may not and must not attempt to (a) access the Services to
> build a competing product or service, **including to train competing AI models** or resell
> the Services except as expressly approved by Anthropic; (b) reverse engineer or duplicate the
> Services; or (c) support any third party's attempt at any of the conduct restricted in this
> sentence."*

**VERIFIED, OpenAI Foundation Terms of Use** (`openaifoundation.org/terms-of-use`, fetched
2026-09-03; the openai.com pages returned 403 — see the fetch-failure note above):

> *"You may not use our Services for any illegal, harmful, or abusive activity. For example,
> you may not: … Automatically or programmatically extract data or Output (defined below). …
> **Use Output to develop models that compete with OpenAI Foundation.**"*
> *"You (a) retain your ownership rights in Input and (b) own the Output. We hereby assign to
> you all our right, title, and interest, if any, in and to Output."*

**(INFERRED)** Both grant the outputs and then restrict what the accepting party may do with
them. The restriction travels with the *person*, not with the data — a third party who
receives the dataset never accepted it. But that is cold comfort: the operator would be the
one in breach, the dataset could not be honestly described, and "is a self-hosted open-weights
CSD a competing model?" is a question the operator should not want to argue.
**Factory rule: REFUSE.** Not because the licence blocks the release, but because the term
blocks the operator. (Note the second OpenAI bullet independently forbids programmatic bulk
extraction of Output, which is what a dataset factory does by definition.)

#### Shape B — the term reaches the *downstream model*

**VERIFIED, Gemma Terms of Use §1.1 and §3** (`ai.google.dev/gemma/terms`, fetched
2026-09-03):

> *"'Model Derivatives' … (i) modifications to Gemma, (ii) works based on Gemma, or (iii) any
> other machine learning model which is created by transfer of patterns of the weights,
> parameters, operations, or Output of Gemma"* — including distillation and **synthetic data
> methods**; *"Outputs are not deemed Model Derivatives."*
> *"3.3 Generated Output. Google claims no rights in Outputs you generate using Gemma."*
> *"3.1 … You must include the use restrictions referenced in Section 3.2 as an enforceable
> provision in any agreement … governing the use and/or distribution of Gemma or Model
> Derivatives and you must provide notice to subsequent users you Distribute to that Gemma or
> Model Derivatives are subject to the use restrictions in Section 3.2."*
> *"3.2 Use Restrictions. You must not use any of the Gemma Services: (1) for the restricted
> uses set forth in the Gemma Prohibited Use Policy…"*

**(INFERRED, and this is the subtle one.)** Gemma's *outputs* are free of Google's claim — so
the **dataset** is clean. But a model *trained on those outputs* falls inside the "Model
Derivatives" definition via "transfer of patterns of … Output … [including] synthetic data
methods", and §3.1 then requires the Gemma use restrictions to be carried forward as an
enforceable provision on anyone the operator distributes to. A CSD region trained on
Gemma-generated rows therefore **cannot** be released under a bare MIT or CC BY-NC-SA notice;
it would have to ship the Gemma Terms alongside. No de minimis threshold is stated in the
definition, so "only a small synthetic share" is not obviously a defence (§5.7).

**VERIFIED, Meta Llama 3 Community License**
(`raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3/LICENSE`):

> *"You will not use the Llama Materials or any output or results of the Llama Materials to
> improve any other large language model (excluding Meta Llama 3 or derivative works thereof)."*

**VERIFIED, Meta Llama 3.1 Community License** (same repo, `llama3_1/LICENSE`) — the
prohibition is replaced by a naming condition:

> *"prominently display 'Built with Llama' on a related website, user interface, blogpost,
> about page, or product documentation. If you use the Llama Materials or any outputs or
> results of the Llama Materials to create, train, fine tune, or otherwise improve an AI
> model, which is distributed or made available, you shall also include 'Llama' at the
> beginning of any such AI model name."*

**(INFERRED)** Llama 2 / Llama 3.0 outputs are a **REFUSE** — an express prohibition on
exactly the factory's use. Llama 3.1+ outputs are *permitted* but force the released model to
be named `Llama-<something>` and to carry "Built with Llama" and the AUP. For a project whose
identity is `CogSynDelta`, that is a naming constraint the operator should decline rather than
discover after training.

#### Safe generators — the class, and the per-model check that still has to happen

**(INFERRED, from the licence structure.)** Apache-2.0 and MIT are software licences over the
Work (weights + code). Neither says anything about outputs, and neither has an acceptable-use
or model-derivative clause. So **a locally-run model whose weights ship under Apache-2.0 or
MIT, used without accepting any hosted-service terms, imposes no term on the generated rows
and no term on a model trained from them.** That is the operator's "which generators are
safe" answer, and it holds for the reason that the licence is silent, not because anyone
granted permission.

**VERIFIED live via `huggingface.co/api/models/*`, 2026-09-03** (`cardData.license`, `gated`):

| model | licence tag | gated | class |
|---|---|---|---|
| `allenai/OLMo-2-1124-7B` | `apache-2.0` | no | **SAFE** |
| `allenai/OLMo-2-0325-32B-Instruct` | `apache-2.0` | no | **SAFE** |
| `Qwen/Qwen2.5-7B` | `apache-2.0` | no | **SAFE** |
| `Qwen/Qwen3-8B` | `apache-2.0` | no | **SAFE** |
| `mistralai/Mistral-7B-v0.3` | `apache-2.0` | no | **SAFE** |
| `HuggingFaceTB/SmolLM2-1.7B` | `apache-2.0` | no | **SAFE** |
| `Qwen/Qwen2.5-3B` | **`other`** | no | **NOT cleared** — bespoke Qwen licence, needs its own reading |
| `Qwen/Qwen2.5-72B` | **`other`** | no | **NOT cleared** — same |
| `google/gemma-2-9b` | `gemma` | manual | **ENCUMBERED** (Shape B) |
| `meta-llama/Llama-3.1-8B` | `llama3.1` | manual | **ENCUMBERED** (naming + AUP) |

**The Qwen rows are the point of the table.** "Qwen is Apache" is false as a family
statement: two of the four Qwen2.5 sizes checked are `other`. The factory must classify
**per model id and revision**, never per vendor — the same rule as "a mirror's tag is not
evidence about its upstream", applied to generators.

---

## 2. The compatibility matrix

### 2.1 Rows = input licence class, columns = the licence the **emitted enriched dataset** may carry

`Y` = permitted; `Y*` = permitted but CC recommends carrying at least the same licence
elements; `—` = not permitted; `n/a` = would be incoherent.

| input ↓ / emitted dataset licence → | CC0 | MIT/Apache (data) | CDLA-Perm-2.0 | CC BY 4.0 | ODC-By 1.0 | CC BY-SA 4.0 | ODbL 1.0 | CDLA-Sharing-1.0 | CC BY-NC 4.0 | CC BY-NC-SA 4.0 | private / unpublished |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **CC0 / PD** | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| **MIT / Apache-2.0 / BSD** | — | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| **CDLA-Permissive 1.0/2.0** | — | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| **O-UDA / C-UDA** | — | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| **CC BY 4.0** | — | — | — | Y | — | Y* | — | — | Y* | Y* | Y |
| **ODC-By 1.0** | — | — | — | — | **Y (only)** | — | — | — | — | — | Y |
| **CC BY-SA 3.0** | — | — | — | — | — | **Y (4.0 via §4(b)(ii))** | — | — | — | — | Y |
| **CC BY-SA 4.0** | — | — | — | — | — | **Y (only)** | — | — | — | — | Y |
| **ODbL 1.0** | — | — | — | — | — | — | **Y (only)** | — | — | — | Y (but see §4.6 trigger) |
| **CDLA-Sharing-1.0** | — | — | — | — | — | — | — | **Y (only)** | — | — | Y |
| **CC BY-NC 4.0** | — | — | — | — | — | — | — | — | Y | Y | Y |
| **CC BY-NC-SA 4.0** | — | — | — | — | — | — | — | — | — | **Y (only)** | Y |
| **bespoke NC (GooAQ shape)** | — | — | — | — | — | — | — | — | Y (chosen) | Y (chosen) | **Y — required by Rider 2** |
| **GPL / AGPL source rows** | — | — | — | — | — | — | — | — | — | — | Y; else GPL-only tier |
| **ND / research-only / no-redistribution / distributor disclaims ownership** | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | **REFUSE at ingest** |

**Reading the matrix (INFERRED, from the clauses quoted in §1):**

1. **Every share-alike family is a singleton column.** CC BY-SA, ODbL, ODC-By and
   CDLA-Sharing each admit exactly one emitted licence — their own. None is on any of the
   others' compatible lists.
2. **The impossible cells are the design constraint.** There is **no** emitted dataset
   licence that satisfies a CC BY-SA input *and* an NC input simultaneously. BY-SA §3(b)(1)
   demands the same licence elements (no NC) and §3(b)(3) forbids imposing additional
   restrictions; BY-NC-SA §3(b)(1) demands NC-SA. The same impossibility holds for
   **CDLA-Sharing × NC** (§3.3 expressly names commercial/non-commercial restrictions),
   **ODbL × NC** (§4.7(a)), **ODC-By × NC** (§4.4), and for every share-alike × different
   share-alike pair.
3. **Therefore the factory may not merge across tiers.** The "strictest-input rule" works for
   *weights*, where one licence must be chosen for one blob. It **fails at the dataset layer**,
   because for the SA×NC pair there is no strictest licence — there is no licence at all.
   The rule the factory needs instead is: **partition first, then enrich.**
4. **Cross-dataset pairing is the operation that trips this**, and it is on the operator's
   enrichment list. Joining a CC BY-SA passage to a GooAQ-NC query produces a row with two
   irreconcilable parents. That row cannot be emitted under any licence. It must not be
   constructed — which is a *planner* constraint, upstream of ingest, not a filter after it.
5. **Collections are the escape hatch, and CC blesses them — VERIFIED, CC FAQ:**
   > *"All Creative Commons licenses (including the version 4.0 licenses) allow licensed
   > material to be included in collections such as anthologies, encyclopedias, and
   > broadcasts. You may choose a license for the collection, however this does not change the
   > license applicable to the original material."*

   So the factory emits **one dataset file per licence tier**, shipped together as a
   collection with a top-level manifest. Each file carries its own licence; nothing merges.

### 2.2 Resulting composed-model release licence, for the three realistic mixes

Under DEC-31's strictest-input rule, applied to the *weights*.

| mix | emitted datasets (separate files) | region weights, standalone | **composed model** | what is load-bearing |
|---|---|---|---|---|
| **Tier P — permissive only**<br>CC0/PD + MIT/Apache/BSD + ODC-By + CDLA-Permissive + O-UDA/C-UDA + CC BY | `<faculty>-permissive` (CC BY 4.0 or CDLA-Permissive-2.0) + `<faculty>-odcby` if ODC-By rows are present | **MIT** | **MIT**, with an attribution manifest | Nothing unsettled. CDLA/O-UDA rows are *expressly* clear at the model layer; CC BY costs a notice. **This is the only tier with no open legal question in it.** |
| **Tier P+SA — add CC BY-SA** | Tier-P files **plus** `<faculty>-sharealike` (CC BY-SA 4.0; 3.0 inputs upgraded per §1.4a) | permissive regions MIT; SA-fed regions **MIT under the "weights are not Adapted Material" reading, CC BY-SA 4.0 under the cautious reading** | **MIT or CC BY-SA 4.0** — the project must *state which reading it takes* | The Layer-2 question, unchanged. Option D's per-region checkpoints + "Collection, not Adapted Material" argument is what keeps the MIT surface. |
| **Tier P+SA+NC — add NC (today's CSD)** | Tier-P + SA files **plus** `<faculty>-nc` (CC BY-NC-SA 4.0). **The SA file and the NC file may never be merged.** | `language_code`/`reasoning`/`classify` MIT; `memory` NC; a purely-SA region CC BY-SA | **CC BY-NC-SA 4.0** (matches DEC-31) | **See the finding immediately below — this row is only coherent under one of the two readings, and the project has not said which.** |

### 2.3 A finding that the enrichment analysis surfaces about the *existing* decision

**(INFERRED, from VERIFIED clauses. Flagged prominently because it changes what the project
must say, not what it must do.)**

The ratified composed-model licence is **CC BY-NC-SA 4.0**, justified as "the strictest term
among every input". Its inputs include SNLI, Natural Questions, SQuAD, HotpotQA and FiQA —
all **CC BY-SA**. Now put the two readings side by side:

- **If weights are NOT Adapted Material** (the field's norm, and the reading the audit records
  as near-universal practice): CC BY-SA's ShareAlike condition never triggers on the weights.
  The NC term — a bespoke README sentence, not CC BY-NC — is honoured as a matter of policy.
  **CC BY-NC-SA 4.0 is a coherent, freely chosen release licence.** ✅
- **If weights ARE Adapted Material** (the cautious reading): SNLI's §3(b)(1) requires the
  adapter's licence to be *"a Creative Commons license with the same License Elements, this
  version or later, or a BY-SA Compatible License."* **CC BY-NC-SA is not** — it adds the NC
  element, and §3(b)(3) separately forbids imposing *"additional or different terms or
  conditions … that restrict exercise of the rights granted."* Releasing as CC BY-NC-SA would
  breach SNLI's condition; releasing as CC BY-SA 4.0 would breach GooAQ's NC term. **Under
  this reading there is no compliant release licence for the composed model at all.** ❌

**So the current licence is not the "strictest-input" answer — it is the answer that the
permissive reading makes available.** The project should say so explicitly in the model card
rather than let CC BY-NC-SA read as the cautious choice. `LICENCE-FOR-OPEN-WEIGHTS.md`
already warns *"asserting that weights are not derivative works because it would be
convenient"* is not a mitigation and that if the position is taken it should be *"taken
explicitly, in the model card, as a stated position rather than a silence."* This is that
position, and it is currently silent.

**Two ways out, both of which the factory can implement:**
- **(i) Separate the checkpoints** (Option D, already recommended): if the SA-fed and NC-fed
  regions ship as distinct files, the SA obligation and the NC obligation never have to be
  satisfied by one licence. This is exactly why "distribute per-region checkpoints as separate
  files, not merged into one weights blob" was recommended, and this analysis raises it from
  *nice* to *the mechanism that makes the cautious reading survivable*.
- **(ii) Remove one side.** Either drop the SA inputs from any region that also carries NC
  input, or drop the NC input. The tier-separable corpus of §4.6 is what makes the cost of
  either measurable rather than guessed.

**This document does not recommend changing the release licence.** It recommends (a) stating
the reading, and (b) building the factory so the choice stays reversible.

---

## 3. DEC-58's synthetic-data contract, restated as licence rules

DEC-58's four clauses are quality and integrity controls. Their licence content is separable,
and the factory needs it as explicit rules because a generator brings *two* licence questions
(the generator's terms, and the seed row's terms) that nothing else in the pipeline has.

**SYN-L1 — The generator is named, or the batch is refused.** `{model_id, revision (exact
commit sha), licence tag, licence source URL + fetch date, the operative output/derivative
clause quoted verbatim}`. This extends DEC-58 clause (1) with the *clause text*, not just the
licence name — because as §1.7 shows, two Apache-adjacent-looking model licences (Gemma,
Llama) differ entirely in whether they reach the downstream model, and the licence name alone
does not tell you. Verified by construction: strip the field, assert refusal (DEC-58's own
test (ii)).

**SYN-L2 — Generators are classified, per model id and revision, into three classes.**

| class | rule | today's members (VERIFIED via HF API 2026-09-03 unless noted) |
|---|---|---|
| **SAFE** | weights under Apache-2.0 or MIT, run locally, **no hosted-service terms accepted**, no acceptable-use or model-derivative clause | `allenai/OLMo-2-1124-7B`, `allenai/OLMo-2-0325-32B-Instruct`, `Qwen/Qwen2.5-7B`, `Qwen/Qwen3-8B`, `mistralai/Mistral-7B-v0.3`, `HuggingFaceTB/SmolLM2-1.7B` |
| **ENCUMBERED** — usable only if the operator accepts the encumbrance *before* generating | terms reach the downstream model | **Gemma** (any version): outputs free, but a model trained on them is a *Model Derivative* and must carry the Gemma Terms + Prohibited Use Policy forward (§3.1). **Llama 3.1+**: permitted, but forces `Llama`-prefixed model name + "Built with Llama" + AUP passthrough |
| **REFUSED** | terms forbid the use, or bind the operator against it | **Llama 2 / Llama 3.0** — *"You will not use the Llama Materials or any output or results … to improve any other large language model"*; **OpenAI** — *"Use Output to develop models that compete with OpenAI…"* and *"Automatically or programmatically extract data or Output"*; **Anthropic** — *"may not … access the Services to build a competing product or service, including to train competing AI models"*; **any model tagged `other`** until its bespoke licence is read (today: `Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-72B`) |

The default for an unclassified generator is **REFUSED**, not ENCUMBERED — the same
fail-closed shape as the existing `fetch()` gate.

**SYN-L3 — Synthetic generation is not a licence launderer, and the factory must say so.**
**(INFERRED, but flatly.)** Generating rows from a model whose *own* training corpus is
BLOCKING does not produce clean rows. The generator's training-data licence does not flow
through to the operator (that is the unsettled Layer-2 question, one level up, and the
operator is not the one who can resolve it), and equally it does not get washed away. The
manifest records the generator's corpus-disclosure status as one of
`{disclosed, partially_disclosed, undisclosed}` and **`undisclosed` is recorded, never
inferred to be clean.** Concretely: no amount of synthetic augmentation repairs `visual`.
tiny-imagenet's problem is a missing grant, and a generator cannot manufacture one — the same
sentence `LICENCE-FOR-OPEN-WEIGHTS.md` already applies to relicensing, applied to generation.

**SYN-L4 — Augmentation does not reset provenance; the seed row's licence survives.**
**(INFERRED, from CC 4.0 §1(a).)** A rationale generated *from* a CC BY-SA premise, a
paraphrase *of* an NC answer, a translation *of* a CC BY passage — each is derived from and
based upon the licensed material, and §1(a) names "translated, altered, arranged,
transformed" explicitly. The synthetic row therefore inherits the **seed row's licence class**
and, where the generator is ENCUMBERED, carries the generator's terms **in addition**. This
is the clause most likely to be skipped, because "we generated it, so we own it" is
intuitive and wrong for anything conditioned on licensed input. Only rows generated from
**unlicensed-input prompts** (a template the operator wrote, a CC0 seed) are the operator's
alone.

**SYN-L5 — A generator is a provenance group.** Unchanged from DEC-58 clause (4): the cap on
synthetic share is a B1 statistic over provenance groups, so one generator cannot become a
monoculture under several batch names. The licence reason to keep it: an ENCUMBERED
generator's share is also the share of the emitted dataset that carries an extra terms
stack, and the receipt must print share **beside its cap**, per §5.1's shape.

**SYN-L6 — The quality gate is unchanged** (DEC-58 clause (3)); it is not a licence rule, but
the licence receipt must record that it ran and passed, because a discarded batch must also
disappear from the attribution manifest and the provenance ids must not dangle.

---

## 4. What the factory must implement so compliance is emitted *with* the dataset

The design principle: **compliance is an artefact, not a claim.** Every obligation identified
in §1 must be either (a) discharged by a file the factory writes, or (b) refused at ingest.
Nothing is left to be remembered at release time.

### 4.1 Manifest fields (extending the existing `Dataset` dataclass and `00-ground.md` §(b))

The current `Dataset` dataclass is
`repo_id, region, license, verdict, why, upstream, config, splits, caveat, columns, revision,
data_files, builder` (VERIFIED, `scripts/csd-corpus-expand.py`). `00-ground.md` already
specifies the *input* catalogue additions (`provenance_group`, `mirror_tag`,
`licence_upstream`, `licence_upstream_source` + fetch date, `grant_scope`, `verdict`,
`usage_tag`, `size`, `redistribute`, `provenance_red_flags`). Those stand. **What follows are
the additional fields the *emitted* dataset needs**, which have no counterpart today because
today nothing is emitted.

```jsonc
{
  "dataset_id": "memory-sharealike-v3",
  "faculty": "memory",
  "tier": "share_alike",            // permissive | attribution | share_alike | nc | refused
  "emitted_at": "2026-09-03T14:00:00Z",
  "factory_commit": "<sha of the factory code that produced this>",

  "licence_out": "CC-BY-SA-4.0",
  "licence_out_basis": [            // WHICH input forced it — never just the answer
    {"input": "stanfordnlp/snli", "clause": "CC BY-SA 4.0 §3(b)(1) + §4(b)"}
  ],
  "licence_out_alternatives_rejected": [
    {"licence": "CC-BY-4.0", "why": "BY-SA §3(b)(1) requires same License Elements"},
    {"licence": "CC-BY-NC-SA-4.0", "why": "BY-SA §3(b)(3) forbids added restrictions"}
  ],

  "inputs": [ {
      "repo_id": "...", "provenance_group": "snli",
      "mirror_tag": "...", "licence_upstream": "<verbatim quote>",
      "licence_upstream_source": "https://...", "licence_fetch_date": "2026-09-03",
      "licence_text_sha256": "<hash of the fetched text, for drift detection>",
      "grant_scope": "whole_corpus",   // + database_rights_only  (NEW, see §1.2b)
      "verdict": "SHARE_ALIKE", "usage_tag": null,
      "redistribute": {"nc": false, "sa": true, "nd": false, "attribution": true},
      "rows_in": 183416, "rows_out": 171204, "share_post_dedup": 0.62
  } ],

  "operations": [                    // ordered; satisfies CC BY §3(a)(1)(b),
    {"op": "dedup", "params": {"method": "minhash", "threshold": 0.85},
     "code_sha": "...", "seed": 1729, "rows_removed": 12212},
    {"op": "pair", "params": {"left": "premise", "right": "hypothesis"}},
    {"op": "cap", "params": {"method": "reservoir", "n": 120000}, "seed": 1729}
  ],                                 // Apache §4(b), CDLA-Sharing §3.1(b),
                                     // and ODbL §4.6(b) "the method of making the alterations"

  "generators": [ {
      "model_id": "allenai/OLMo-2-1124-7B", "revision": "<sha>",
      "licence": "apache-2.0", "licence_source": "https://...", "fetch_date": "2026-09-03",
      "output_clause_verbatim": "<none — Apache-2.0 is silent on outputs>",
      "class": "SAFE", "corpus_disclosure": "disclosed",
      "rows_generated": 8000, "share": 0.062, "cap": 0.10,
      "quality_gate": {"ran": true, "passed": true, "metric": "...", "human_baseline": "..."}
  } ],

  "attribution": [ {                 // TASL, one per attributable input
      "title": "SNLI", "author": "Bowman et al., Stanford NLP",
      "source": "https://nlp.stanford.edu/projects/snli/",
      "licence": "CC BY-SA 4.0", "licence_uri": "https://creativecommons.org/licenses/by-sa/4.0/",
      "modified": true, "modification_summary": "filtered, deduplicated, re-paired, capped",
      "warranty_disclaimer_notice": true, "notice_verbatim": "<if the licence supplies one>"
  } ],

  "obligations": [                   // machine-checkable, asserted at release time
    {"kind": "share_alike",  "licence": "CC-BY-SA-4.0", "scope": "dataset"},
    {"kind": "attribution",  "scope": "dataset+model_card", "manifest": "attribution.json"},
    {"kind": "indicate_modifications", "satisfied_by": "operations[]"},
    {"kind": "carry_licence_text", "files": ["LICENSE-CC-BY-SA-4.0.txt"]}
    // e.g. also: {"kind":"publish_derivative_database","licence":"ODbL-1.0",
    //             "trigger":"public use of a Produced Work","satisfied_by":null}  <- blocks release
  ],

  "balance": { "b1_max_share": 0.38, "b1_cap": 0.40,
               "b2_n_eff": 3.4, "b2_floor": 3.0,
               "b3_mode": "held-out-domain", "b4_method": "reservoir", "b4_seed": 1729,
               "b5_max_stratum": 0.21, "b5_max_min_ratio": 8.4 },

  "refusals": [                      // the REFUSE path is a field, not a silence
    {"candidate": "<repo_id>", "reason": "R3 distributor disclaims ownership",
     "evidence": "<verbatim quote>", "source": "https://...", "date": "2026-09-03"}
  ],

  "row_provenance": {"scheme": "v1", "columns": ["src_group", "src_fp", "op_chain", "gen_id", "lic_class"]}
}
```

**Three fields carry most of the weight, and they are the ones a hurried implementation
drops:** `licence_out_basis` (an answer without its reason cannot be re-checked when an input
changes), `operations[]` (it is simultaneously the CC BY "indicate if you modified"
compliance, the Apache §4(b) changed-files notice, the CDLA-Sharing §3.1(b) notice, and the
ODbL §4.6(b) alteration method), and `refusals[]` (a factory that silently drops a candidate
is indistinguishable from one that never looked).

### 4.2 Per-row provenance

Every emitted row carries, **as columns in the shard, not as a sidecar file**:

| column | content | why it must exist |
|---|---|---|
| `src_group` | provenance group id (DEC-46 grouping, not repo id) | B1/B2/B5 are computed on groups post-dedup; the LibriVox and Wikipedia lineage collisions are invisible at repo granularity |
| `src_fp` | fingerprint of the originating source row | §5.6/DEC-38-39 poisoning defence; also the join key for a licence-driven deletion |
| `op_chain` | id into `operations[]` | which transforms produced this row |
| `gen_id` | generator id, or null | SYN-L1/L5; lets a generator's share be recomputed rather than trusted |
| `lic_class` | the row's licence class | **the compliance column** — makes tier separation, filtering and the per-row GPL filter mechanical |

**The `lic_class` column is the lesson of `all-nli` made structural.** `all-nli` is
BLOCKING-as-trained *only* because the mirror dropped `genre`, `promptID` and `pairID`, so
rows that are individually attributable became collectively unattributable. The same fate
awaits any emitted dataset that carries its licence only in a manifest: one filtering pass by
a future agent and the provenance is gone. It must be in the row.

**Cost (INFERRED):** a 64-bit group id + 128-bit row fingerprint + small ints ≈ 24-32 bytes
per row. At the ~1e10-token / ~1e8-row scale path that is roughly 2.4-3.2 GB — negligible
against 18 GB of `code` alone, and it is the only thing that makes a licence-driven deletion
(consent revocation, a licence changing upstream, a REFUSE discovered late) executable
instead of catastrophic.

### 4.3 Licence verdict receipts

Two receipts, both human-readable, both emitted at ingest/emit time rather than written later.

**Per input (`receipts/inputs/<repo_id>.md`)** — reproduces the mirror tag, the verbatim
upstream licence text with its URL and fetch date, the `sha256` of that text, the grant scope,
the verdict and the one-line why. **Keep the fetched licence text itself** beside the receipt
(as this session did under `licence-texts/`), so a later re-fetch can *diff* rather than
re-argue. An upstream licence that changes silently is a real failure mode — GooAQ's README
has been untouched since 2021, but nothing guarantees that for the next source.

**Per emitted dataset (`receipts/emitted/<dataset_id>.md`)** — the manifest rendered for a
human, plus: rows in → rows out with dedup removals per source; per-source share against the
B1 cap and `N_eff` against the B2 floor, **each printed beside its threshold** (a share
reported without its cap is an opinion — §5.1's existing rule); the generator share beside its
cap; the emitted licence **and which input forced it**; the full attribution block ready to
paste into a model card; and the refusal list.

### 4.4 The attribution manifest, and how each family's obligation is discharged

One `attribution.json` (machine) + `ATTRIBUTION.md` (human, generated from it) per emitted
dataset. The model card links to it — permitted by CC BY §3(a)(2) and confirmed for the
dataset case by the CC FAQ, both quoted verbatim in §1.3.

| input family | mechanical discharge |
|---|---|
| CC0 / PD | nothing required; record the citation the source requests as a courtesy field |
| MIT / BSD | copy the licence text + copyright line into `ATTRIBUTION.md` |
| Apache-2.0 | copy the `NOTICE` file verbatim (§4(d)); set `modified: true` + `operations[]` (§4(b)) |
| CC BY / CC BY-SA | TASL row + licence URI + warranty-disclaimer notice + `modified` + modification summary (§3(a)(1) a-c) |
| CC BY-SA | additionally include the adapter's licence text/URI (§3(b)(2)) |
| ODC-By / ODbL | emit the licence's own example notice verbatim, per input (§4.3 of each), into both the dataset README and the model card |
| CDLA-Permissive / CDLA-Sharing | ship the agreement text with the data (§2.1 / §3.3); Sharing additionally needs the per-file "changed" notice (§3.1(b)) and preserved provider credit (§3.1(c)) |
| CC BY-NC / bespoke NC | TASL row + an explicit statement that the training use was non-commercial (§1.5) |
| ENCUMBERED generator | the generator's terms carried forward as the terms require (Gemma §3.1: as an *enforceable provision*, not a mention) |

**Two obligations that cannot be discharged by a manifest, and are therefore ingest-time
refusals:** per-item attribution (Stack Exchange shape, §1.3) and ODbL's §4.6 dataset-access
duty when the enriched dataset is to stay private (§1.4b).

### 4.5 The REFUSE path

The existing gate — *"`fetch` refuses any entry whose verdict is not `TRAIN_OK`, and there is
deliberately no flag to override that"* (VERIFIED, `csd-corpus-expand.py` docstring) — is the
right shape and should be extended, not replaced. Additional structural refusals, all
fail-closed, none overridable:

| id | refuse when | grounded in |
|---|---|---|
| **R1** | `licence_upstream_source` is absent, its host equals the mirror host, or `licence_fetch_date` is missing | A0m's gate, `00-ground.md` §(b) |
| **R2** | `grant_scope ∈ {metadata_only, code_only, database_rights_only, unstated}` while corpus **content** is being ingested | AudioSet/CSS10 precedent; ODC-By §2.4 (§1.2b) |
| **R3** | the distributor disclaims owning what it distributes | operator stance; CC0 §4(c) shows even a permissive tag can carry this |
| **R4** | ND, research-only, academic-only, no-redistribution EULA, paywalled/membership | operator stance; `BLOCKING` definition |
| **R5** | the input set for one emitted dataset has **no legal common output licence** (any SA×NC or SA×different-SA pair) | §2.1 — **refuse the merge, not the inputs**; the planner must split into tiers |
| **R6** | a synthetic batch has no named generator, or names a REFUSED-class generator, or an unclassified one | SYN-L1/L2, DEC-58 (ii) |
| **R7** | a `CONSENT_OPEN` input is proposed for **training** | OD-11; admit only as `usage_tag: EVAL-ONLY` unless the operator accepts the revocation duty *and* the row-level delete path has been demonstrated end-to-end |
| **R8** | an ODbL input is combined with intent to publish weights **and** keep the enriched dataset private | ODbL §4.4(c) + §4.6 (§1.4b) — the two intents are incompatible |
| **R9** | the input's attribution obligation is **per-item** and per-row links are unavailable | §1.3; unsatisfiable at any scale |
| **R10** | a generator is ENCUMBERED and the operator has not recorded acceptance of the encumbrance *before* generation | §1.7 Shape B; discovering Gemma's Model-Derivative reach after training is unrecoverable |

**Each must be verified by making it fire** — per `verify-guards-by-making-them-fail`: three
CSD guards were previously structurally incapable of firing, and reading them confirmed intent
rather than behaviour. Construct a refusing fixture per rule (strip the field, forge the
grant scope, hand R5 a deliberate SA+NC pair) and assert refusal in the test suite.

### 4.6 A separable clean-permissive tier, per faculty

**The rule:** for every faculty the factory emits **at minimum** `<faculty>-permissive`, and
one additional file per further tier actually used. `<faculty>-permissive` contains only
CC0/PD, MIT/Apache/BSD, ODC-By, CDLA-Permissive, O-UDA/C-UDA and CC BY rows. **Each tier must
independently satisfy B1-B5 or carry a dated waiver naming the missing sources** — so the
permissive tier is a *trainable corpus on its own*, not a residue that happens to be clean.

Why this specific mechanic and not a flag:

1. **It is the only thing that keeps an MIT standalone-region release reachable.** Option D's
   value is entirely in regions that have no restrictive input; once a region's corpus is
   merged across tiers, that option is gone and cannot be recovered without retraining.
2. **It makes the price of an NC or SA input measurable rather than argued.** Train the
   permissive tier, train the full tier, compare on the same eval. That is the
   `measure-the-thing-not-the-proxy` discipline applied to a licence decision, and it turns
   "GooAQ is 77.8% of `retrieve`" from a share into a capability delta.
3. **It is the structural fix for §2.3.** If SA-fed and NC-fed material never share a
   checkpoint, the impossible cell never has to be resolved.
4. **It makes the current starvation honest.** Applied to today's fleet (INFERRED from
   `00-ground.md`): `language_code`'s permissive tier is the ~71.3% permissive-repo filter of
   CodeSearchNet plus `apps` (MIT) and `code_contests` (CC BY) — real, if still B1/B2-failing.
   `reasoning`'s is gsm8k (MIT) + aqua_rat (Apache) — real. **`memory`'s is very nearly
   empty**: GooAQ is NC, SQuAD/HotpotQA/NQ/FiQA are SA, SNLI is SA. **`visual`'s is empty
   today** and its replacement composite is the thing that fills it. Emitting the tier makes
   those four facts appear as file sizes rather than as prose in an audit.

---

## 5. Legal uncertainty register

Where the operator should get a reading, and what to ask. Each entry names the question
precisely enough to be billable.

**5.1 — Are trained weights "Adapted Material" / a derivative work of the training data?**
Unchanged and unresolved; `LICENCE-FOR-OPEN-WEIGHTS.md` surveys the positions (CC's own May
2025 primer declining to choose; *Getty v. Stability AI* [2025] EWHC 2863 (Ch) ¶599-600 on UK
secondary infringement; SFC's contrary view; `timm`/torchvision telling users to assume the
dataset licence reaches the weights). **This analysis does not depend on the answer at Layer 1
and the factory should not be built to depend on it at Layer 2.**

**5.2 — Is the composed model's CC BY-NC-SA 4.0 licence coherent given CC BY-SA inputs?**
The §2.3 finding. Ask: *"Given CC BY-SA 4.0 §3(b)(1) and §3(b)(3), can a model trained on
CC BY-SA data be released under CC BY-NC-SA 4.0 under the cautious reading? If not, is the
per-region-checkpoint 'Collection, not Adapted Material' structure sufficient to keep the
obligations separate?"* This is the highest-value question in the register because it is the
only one that could invalidate a shipped artefact rather than merely constrain a future one.

**5.3 — Do sui generis database rights apply to this operator at all?**
CC 4.0 §4 opens *"Where the Licensed Rights include Sui Generis Database Rights that apply to
Your use"*. The EU/UK right has no US equivalent. If none applies, a filter-only enrichment
may not be Adapted Material and §1.0's table over-constrains. Ask: *"For a US-domiciled
operator publishing globally, does §4 bite, and does the answer change on publication into
the EU?"* The engineering answer is to comply anyway; the commercial answer may differ.

**5.4 — Is a machine learning model a "Produced Work" under ODbL / ODC-By?**
**This weakens the favourable reading in §1.2b and §1.4b and must not be papered over.**
VERIFIED, ODbL §1: *"'Produced Work' – a work (such as an image, audiovisual material, text,
or sounds) resulting from using the whole or a Substantial part of the Contents (**via a
search or other query**) from this Database…"*. A set of weights is not an image, audio, text
or sound, and training is not obviously "a search or other query". If a model is **neither** a
Derivative Database **nor** a Produced Work, ODbL is simply *silent* — which is a gap, not a
permission. Ask: *"Does training fall inside ODbL's Produced Work definition; if not, what
governs?"*

**5.5 — Does a bespoke non-commercial statement in a README create an enforceable term, and
against whom?** The GooAQ shape. DEC-31 accepts the restrictive reading as policy, which makes
this question non-blocking — but it stays open, and it decides whether GooAQ-derived rows may
ever be redistributed (Rider 2 currently says no).

**5.6 — Do contractual output restrictions bind a third party who receives the dataset?**
The OpenAI/Anthropic shape. No privity suggests not; the operator who accepted the terms is
bound regardless. The factory's REFUSE (§3, SYN-L2) makes the question moot going forward, but
it matters for any material already in the tree.

**5.7 — How far does Gemma's "Model Derivative" definition reach through synthetic data?**
The definition covers a model *"created by transfer of patterns of the weights, parameters,
operations, or Output of Gemma"* including synthetic-data methods, and states no de minimis
threshold. Ask: *"If 5% of a from-scratch model's training rows were Gemma-generated, is the
model a Model Derivative, and must it therefore carry the Gemma Terms?"* Until answered, treat
Gemma as ENCUMBERED at any share.

**5.8 — GPL/AGPL and datasets of source code.** Whether distributing a corpus of GPL functions
is distributing the programs; whether the §5 aggregation carve-out reaches a training corpus;
whether weights are a "work based on" the code. The BigCode filter-instead-of-argue approach
sidesteps all three and remains the recommendation.

**5.9 — Does an NC term restrict the training run itself, not just the release?**
CC BY-NC's NC condition binds the exercise of the Licensed Rights, and reproducing rows for a
training pass is such an exercise where copyright is implicated. Probably satisfied here on
the facts; worth confirming before any change in how the project is funded or hosted.

**5.10 — Is CC BY-NC-SA the right combined designation at all?**
`LICENCE-FOR-OPEN-WEIGHTS.md` marks this INFERRED and not independently checked. This session
did not resolve it either; §2.3 shows the harder problem sits underneath it. Ask both
questions together.

---

## Appendix A — What changed, versus the documents this builds on

**New in this analysis (not in `LICENCE-FOR-OPEN-WEIGHTS.md` or `AUDIO-CORPUS-AUDIT.md`):**

1. **The two-layer frame (§0).** The dataset question is decidable; the weights question is
   not. Compliance should be engineered at the layer that has answers.
2. **Filtering alone triggers every share-alike data licence** (§1.0), from express text in
   CC 4.0 §4(b), ODbL §4.4(b) and CDLA-Sharing §1.8.
3. **The SA×NC cell is empty** (§2.1) — the strictest-input rule has no dataset-layer solution
   for that pair, and cross-dataset pairing is exactly the operation that creates it.
4. **§2.3** — today's composed-model licence is coherent only under the permissive reading of
   the weights question, and the project has not stated that it takes that reading.
5. **CC BY-SA 3.0 → 4.0 is settled** from BY-SA 3.0 §4(b)(ii) itself, closing an item
   `LICENCE-FOR-OPEN-WEIGHTS.md` explicitly left open.
6. **ODC-By is a database-level copyleft** despite its name (§1.2b), and its §2.4 licenses the
   database and not the contents — a new `grant_scope` value, `database_rights_only`.
7. **ODbL §4.6 forces publication of the enriched dataset** once weights are released,
   contradicting DEC-31 Rider 2's private-HF-only policy (§1.4b, R8).
8. **CDLA-Permissive / CDLA-Sharing / O-UDA / C-UDA settle the model question in writing**
   (§1.2c, §1.4c) — a sourcing tiebreak the surveyor should apply.
9. **Gemma's Model-Derivative definition reaches models trained on Gemma outputs** even though
   Google claims no rights in the outputs themselves (§1.7 Shape B) — the trap that looks like
   permission.
10. **"Qwen is Apache" is false as a family claim** (§1.7) — verified per model id.

**Unchanged and reaffirmed:** DEC-31's NC-tolerant policy; the strictest-input rule *for
weights*; Option D's per-region checkpoints (this analysis raises its importance);
`BLOCKING ≠ NC` and relicensing moves neither; mirrors lie, and only a primary-source licence
text with a fetch date counts.

## Appendix B — Files written this session

- `/akula-data/session-backup-staging/dataset-factory/20-enrichment-licence-impact.md` — this document
- `/akula-data/session-backup-staging/dataset-factory/licence-texts/` — the primary licence
  texts quoted above, as fetched on 2026-09-03, plus `detag.py` (the HTML→text helper used to
  read the ones served only as web pages). Keeping these is not archival tidiness: §4.3's
  drift check needs the bytes that were relied on, not a URL that may say something else later.

**No dataset content was downloaded. No repository was modified.**
