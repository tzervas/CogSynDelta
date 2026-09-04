# Model-card licence drafts — two readings of the weights question

**Status:** drafts for the operator. Not a decision. Not legal advice.
**Written:** 2026-09-03.
**Does not modify any git repository.** Ready-to-paste blocks are in §2 and §3.

> **Neither the author nor the reader of this document is a lawyer.** Nothing here is a
> legal conclusion, a licence grant, or counsel. It reports what primary licence texts
> *say*, quoted verbatim with URLs, and what follows mechanically if one reading or the
> other is taken. The operator chooses a reading and may obtain independent legal counsel
> before shipping. Creative Commons itself is not a law firm and does not provide legal
> advice. [V] https://creativecommons.org/licenses/by-sa/4.0/legalcode (preamble);
> [V] https://creativecommons.org/faq/ (“Creative Commons does not provide legal advice”).

**Marking.** **[V]** = quoted from a primary text fetched this session, with URL.
**[I]** = this document’s inference over those texts, or over the project’s own ratified
notes. Every conclusion that combines two clauses is **[I]** even when both clauses are
**[V]**.

**Ground.**
- Ratified decision (2026-09-02): `CogSynDelta/docs/design/LICENCE-FOR-OPEN-WEIGHTS.md`
  (Rider 1, strictest-input rule, composed-model tier CC BY-NC-SA 4.0).
- Licence-impact analysis: `docs/design/evidence/dataset-factory-2026-09-03/20-enrichment-licence-impact.md` §2.3.
- Catalogue item 1: `docs/design/DATASET-FACTORY-CATALOGUE-2026-09-03.md` §10.1 #1.
- Primary texts fetched 2026-09-03: CC BY-SA 4.0 legalcode, CC BY-NC-SA 4.0 legalcode,
  CC FAQ, CC *Using CC-licensed Works for AI Training* (May 2025), CC compatible-licences
  list, OSI Open Source AI Definition 1.0.

**What this document does not do.** It does not change the 2026-09-02 release licence. It
does not resolve whether weights are Adapted Material. It writes the two model-card texts
the project would paste once the operator picks a reading.

---

## 1. The two readings, in one page

The unsettled question is whether **trained weights are “Adapted Material”** (CC’s term)
of the training data. ShareAlike and the adapter’s-licence rules fire **only if they are**,
and **only when the weights are Shared**. [V] CC BY-SA 4.0 §3(b) opens “if You Share
Adapted Material You produce”; [V] CC FAQ, “Artificial intelligence and CC licenses”:
the BY and SA conditions “are triggered only when works or adaptations of works are
publicly shared” — https://creativecommons.org/faq/#artificial-intelligence-and-cc-licenses.
Creative Commons states both halves of the question and declines to choose (quoted below).
The project’s current CC BY-NC-SA 4.0 composed-model tag is coherent under one reading
and impossible under the other. [I] `20-enrichment-licence-impact.md` §2.3; catalogue §10.1 #1.

This is **Layer 2** (training → weights). **Layer 1** (enrichment → emitted dataset) is a
different question, already answered by licence text: a filtered or joined BY-SA database
**is** Adapted Material “including for purposes of Section 3(b)”. [V] CC BY-SA 4.0 §4(b),
https://creativecommons.org/licenses/by-sa/4.0/legalcode. Nothing below re-opens Layer 1.
Datasets stay private-HF-only (Rider 2); these drafts are about **weights**.

### Primary quotes both readings rest on

**Adapted Material — CC BY-SA 4.0 §1(a)** (identical in BY-NC-SA 4.0 §1(a)).
[V] https://creativecommons.org/licenses/by-sa/4.0/legalcode

> Adapted Material means material subject to Copyright and Similar Rights that is derived
> from or based upon the Licensed Material and in which the Licensed Material is translated,
> altered, arranged, transformed, or otherwise modified in a manner requiring permission
> under the Copyright and Similar Rights held by the Licensor.

**Technical modifications are not adaptations — §2(a)(4).**
[V] same URL.

> For purposes of this Public License, simply making modifications authorized by this
> Section 2(a)(4) never produces Adapted Material.

§2(a)(4) covers “all media and formats” and “technical modifications necessary” to
exercise the Licensed Rights. [I] Training a model is not a format conversion of the
dataset; this clause is quoted because the permissive reading sometimes leans on it, and
it does not obviously reach weights.

**Downstream recipients — §2(a)(5).**
[V] same URL.

> (A) Offer from the Licensor — Licensed Material. Every recipient of the Licensed Material
> automatically receives an offer from the Licensor to exercise the Licensed Rights under
> the terms and conditions of this Public License.
> (B) Additional offer from the Licensor — Adapted Material. Every recipient of Adapted
> Material from You automatically receives an offer from the Licensor to exercise the
> Licensed Rights in the Adapted Material under the conditions of the Adapter’s License
> You apply.
> (C) No downstream restrictions. You may not offer or impose any additional or different
> terms or conditions on, or apply any Effective Technological Measures to, the Licensed
> Material if doing so restricts exercise of the Licensed Rights by any recipient of the
> Licensed Material.

**ShareAlike — CC BY-SA 4.0 §3(b).**
[V] same URL.

> In addition to the conditions in Section 3(a), if You Share Adapted Material You produce,
> the following conditions also apply.
> 1. The Adapter’s License You apply must be a Creative Commons license with the same
> License Elements, this version or later, or a BY-SA Compatible License.
> 2. You must include the text of, or the URI or hyperlink to, the Adapter’s License You apply. …
> 3. You may not offer or impose any additional or different terms or conditions on, or
> apply any Effective Technological Measures to, Adapted Material that restrict exercise of
> the rights granted under the Adapter’s License You apply.

BY-SA 4.0’s License Elements are **Attribution and ShareAlike** — not NonCommercial.
[V] §1(g), same URL. BY-NC-SA 4.0’s License Elements are **Attribution, NonCommercial,
and ShareAlike**. [V] https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode §1(g).
BY-NC-SA is **not** a BY-SA Compatible License. The designated BY-SA 4.0 compatibles are
Free Art License 1.3 and GPLv3 (one-way). [V]
https://creativecommons.org/share-your-work/licensing-considerations/compatible-licenses/.
No non-CC licence is designated compatible with BY-NC-SA 4.0. [V] same page.

**CC FAQ — when is a use an adaptation; combining BY-SA with NC.**
[V] https://creativecommons.org/faq/

> Whether a modification of licensed material is considered an adaptation for the purpose
> of CC licenses depends primarily on the applicable copyright law. … Generally, a
> modification rises to the level of an adaptation under copyright law when the modified
> work is based on the prior work but manifests sufficient new creativity to be
> copyrightable… Also, under version 4.0, certain uses of databases restricted by sui
> generis database rights also constitute adaptations (called “Adapted Material” in the
> 4.0 licenses), whether or not they would be considered adaptations under copyright law.

> If you create a remix with material licensed under a ShareAlike license, you need to
> make sure that all of the material contributed to the remix is licensed under the same
> license or one that CC has named as compatible… Similarly, if you want to use a remix
> for commercial purposes, you cannot incorporate material released under one of the
> NonCommercial licenses.

> In general, when remixing ShareAlike content, your adapter’s license must be the same
> license as the license on the material you are adapting.

**CC’s own position on AI training (both halves, unchosen).**
[V] Creative Commons, *Using CC-licensed Works for AI Training*, May 2025,
https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/ and the PDF
https://creativecommons.org/wp-content/uploads/2025/05/Using-CC-licensed-Works-for-AI-Training.pdf

Conservative / over-compliance half:

> When training data is subject to the ShareAlike condition, model outputs and the model
> itself, if shared publicly, should be made available under the same CC license as the
> original works when taking this conservative approach.

> It is worth noting that following this guidance will almost certainly lead to
> overcompliance as a matter of copyright law and the Creative Commons licenses themselves.
> It assumes the most restrictive version of the facts and the law in order to put forth
> the most conservative approach.

> This guidance that follows is designed for those scenarios and is not intended to take a
> position on whether and when copyright applies.

Permissive / copyright-law half (stated in the ND paragraph, applying to models generally):

> Although, in many cases, neither the AI model nor its outputs would be considered to be
> derivative works of training data under copyright law…

NC, if the licence applies at all:

> When training data is subject to the NonCommercial restriction, complying with the CC
> license requires that your use not be “primarily intended for or directed toward
> commercial advantage or monetary compensation.” In other words, making copies of the
> work in the process of training a model, as well as subsequent use and distribution of
> the trained model, would need to be for noncommercial purposes.

**“Open weights” is not “Open Source AI”.**
[V] OSI, *The Open Source AI Definition – 1.0*, https://opensource.org/ai/open-source-ai-definition

> An *Open Source AI* is an AI system made available under terms and in a way that grant
> the freedoms to: **Use** the system for any purpose and without having to ask for
> permission. …

> **Data Information:** Sufficiently detailed information about the data used to train the
> system so that a skilled person can build a substantially equivalent system. …

> “Open Source models” and “Open Source weights” must include the data information and
> code used to derive those parameters.

> The Open Source AI Definition does not require a specific legal mechanism for assuring
> that the model parameters are freely available to all.

[I] An NC restriction on the weights is compatible with “open weights” as this project
uses the phrase (Decision 2026-09-02: NC “just changes the licensing from MIT to something
that restricts commercial use”). It is **not** compatible with OSAID’s “use for any
purpose”. OSAID also treats **data information** and **weights** as separately licensed
elements — which is the framing the permissive reading uses, and which OSAID does not
itself decide.

### Reading P — permissive: weights are not Adapted Material of the training data

**Rests on:** §1(a)’s “in a manner requiring permission under the Copyright and Similar
Rights” gate (if training is fair use / TDM / otherwise not a copyright-restricted
adaptation, the CC conditions never fire — [V] §2(a)(2) Exceptions and Limitations;
[V] CC FAQ “Do Creative Commons licenses affect exceptions and limitations”); CC’s
“in many cases, neither the AI model nor its outputs would be considered to be derivative
works”; OSI’s split between data information and parameters; the field’s near-universal
practice of shipping MIT/Apache weights over BY-SA corpora. [I] `LICENCE-FOR-OPEN-WEIGHTS.md`,
“The share-alike question”.

**What it implies.**

| artefact | implication under Reading P |
|---|---|
| **(a) Composed CogSynDelta model** (GooAQ NC + CC BY-SA inputs such as SNLI, Natural Questions, SQuAD, HotpotQA, and — on the 2026-09-02 reading — FiQA, plus permissive sets) | ShareAlike never triggers on the weights. The operator **chooses** a licence. The 2026-09-02 decision chose **CC BY-NC-SA 4.0 as policy**, to honour GooAQ’s README NC sentence and the strictest-input rule. That choice is coherent: it is not discharging SNLI’s §3(b), because §3(b) has not fired. Attribution manifests are still supplied for every input (cheap; §3(a)(2) allows a URI). [I] |
| **(b) Submodel, permissive inputs only** (MIT/Apache/BSD/CC0/CDLA-Permissive/O-UDA, optionally CC BY) | Operator’s choice; the 2026-09-02 table ships these **MIT**. CDLA-Permissive-2.0 §3.1 / §5.4 and O-UDA §5.4 expressly put “machine learning models” outside the data licence — the one family where Layer 2 is written, not inferred. [V] https://cdla.dev/permissive-2-0/ |
| **(c) Submodel with CC BY-SA inputs** (e.g. `compress` on SNLI) | ShareAlike does not reach the weights. Standalone release may be MIT (field norm) or CC BY-SA 4.0 (policy caution). The 2026-09-02 Option D table ships `compress` as **CC BY-SA 4.0 anyway**, as a policy overlay, not as a legal necessity under this reading. [I] |
| **(d) Submodel with an NC input** (e.g. `retrieve` / `memory` on GooAQ) | NC is honoured as **policy**, not as a fired ShareAlike adapter’s-licence. The 2026-09-02 table ships it **CC BY-NC-SA 4.0** because that region also has SA-tagged inputs; a region with NC and no SA could be CC BY-NC 4.0. GooAQ’s NC is a README sentence over an Apache-2.0 LICENSE file, not CC BY-NC; the CC tag is the operator’s chosen expression of a term whose scope is uncertain. [V] https://raw.githubusercontent.com/allenai/gooaq/main/README.md ; [I] `20-enrichment` §1.5. |

**Cost of this reading, stated plainly.** It is the reading that makes the already-ratified
composed-model licence **true rather than silent**. If a court later holds that weights
**are** Adapted Material, a CC BY-NC-SA release of a model trained on BY-SA inputs is the
incoherent case in Reading C below. The mitigation is to **say the reading in the card**,
not to pretend CC BY-NC-SA is the cautious choice. [I] `LICENCE-FOR-OPEN-WEIGHTS.md`:
“If that position is taken, take it explicitly, in the model card, as a stated position
rather than a silence.”

### Reading C — cautious: weights are Adapted Material of the training data

**Rests on:** §1(a) read to include a trained model as “derived from or based upon” the
Licensed Material; CC’s conservative-approach sentence that “the model itself, if shared
publicly, should be made available under the same CC license”; `timm` / torchvision
practice of telling users to assume the dataset licence reaches the weights; SFC’s
position that a trained model “probably is” a work based on the input. [I] citations in
`LICENCE-FOR-OPEN-WEIGHTS.md`. §3(b) then fires on any Shared weights trained on BY-SA
data.

**What it implies.**

| artefact | implication under Reading C |
|---|---|
| **(a) Composed CogSynDelta model** with **both** CC BY-SA inputs **and** an NC input | **No coherent release licence.** SNLI (and any other BY-SA parent) requires the adapter’s licence to have the **same License Elements** (BY + SA, no NC) [V] §3(b)(1), and forbids additional terms that restrict the granted rights [V] §3(b)(3). CC BY-NC-SA adds the NC element and is not a BY-SA Compatible License [V] compatible-licences list. Releasing as CC BY-SA 4.0 would drop GooAQ’s NC term, which the 2026-09-02 decision accepted and kept. Rider 1 (a merge inherits the most restrictive parent) cannot break the tie: there is no “most restrictive” licence that is both BY-SA and NC-SA. [I] `20-enrichment` §2.1–§2.3. The card must pick **one side**: (i) drop NC inputs, release CC BY-SA 4.0; (ii) drop CC BY-SA inputs, release CC BY-NC-SA 4.0; or (iii) ship a permissive-only tier. Option D (separate per-region checkpoint files, Collection not adaptation) is the architectural escape if the composed blob is **not** shipped. [V] CC FAQ collections: “You may choose a license for the collection, however this does not change the license applicable to the original material.” https://creativecommons.org/faq/ |
| **(b) Submodel, permissive inputs only** | Unconstrained at Layer 2. Ship MIT (or Apache-2.0 / CC BY 4.0 with a TASL notice). This is the only tier with no open legal question in it. [I] `20-enrichment` §2.2 Tier P. |
| **(c) Submodel with CC BY-SA inputs** (no NC) | Adapter’s licence **must** be CC BY-SA 4.0 (or later, or FAL 1.3 / GPLv3). Not MIT. Not CC BY. Not CC BY-NC-SA. Attribution + “indicate if You modified” still attach [V] §3(a). BY-SA 3.0 inputs (Natural Questions) may be emitted/adapted under 4.0 via 3.0 §4(b)(ii) “later version with the same License Elements”. [V] https://creativecommons.org/licenses/by-sa/3.0/legalcode |
| **(d) Submodel with an NC input** (no BY-SA) | Adapter’s licence **must** carry NC. CC BY-NC 4.0 or, if a BY-NC-SA parent is present, CC BY-NC-SA 4.0 [V] BY-NC-SA §3(b)(1) same License Elements. Training itself must be NonCommercial if the NC licence is the one being relied on [V] CC primer, NC paragraph. A submodel that also has a BY-SA parent collapses into (a): **impossible** as one blob. |

**FiQA footnote.** The 2026-09-02 decision treated FiQA as CC BY-SA 4.0. The 2026-09-03
catalogue reclassified it **NC** (“The training data is available only for non-commercial
use”). [V] catalogue §10.2; `BeIR/fiqa` entry. Either way the composed model still has
**both** an NC parent (GooAQ, and FiQA on the catalogue reading) **and** BY-SA parents
(SNLI, Natural Questions, SQuAD, HotpotQA). The contradiction in (a) does not depend on
which of the two FiQA readings is right.

---

## 2. DRAFT A — model-card text under the permissive reading

*Paste into a README.md / Hugging Face model card. This is the text that matches the
already-ratified 2026-09-02 licence (CC BY-NC-SA 4.0 on the composed model) **once the
reading is stated**. It does not change that licence. Replace angle-bracket placeholders
at publication. Attribution URIs must point at a real `ATTRIBUTION.md` / `attribution.json`
the factory emits.*

````markdown
## License

**Code** (architecture, training scripts, this repository): MIT.

**Composed model weights** (the merged CogSynDelta checkpoint):
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

**Standalone region checkpoints**, when shipped as separate files rather than as the
composed blob:

| checkpoint | licence |
|---|---|
| `code`, `classify`, `reason`, and `vl_latent` once its corpus is a clean-permissive replacement | MIT |
| `compress` (SNLI / share-alike inputs) | CC BY-SA 4.0 (policy overlay; see Reading taken) |
| `retrieve` / `memory` (GooAQ NC + share-alike inputs) | CC BY-NC-SA 4.0 |
| composed model | CC BY-NC-SA 4.0 |

This is a **non-commercial** release of the weights. “NonCommercial” means not primarily
intended for or directed towards commercial advantage or monetary compensation
([CC BY-NC-SA 4.0 §1](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)).
Training on the NC-tagged inputs was itself non-commercial research. Dataset rows are
**not** redistributed; dataset publication is private (project Rider 2).

GooAQ, the input that moves the composed model into the NC family, is **not** CC BY-NC.
Its GitHub LICENSE file is Apache-2.0; its README states “This dataset should not be used
for any commercial purposes.” This project accepted the restrictive reading on 2026-09-02
and expresses it by releasing these weights under CC BY-NC-SA 4.0. That tag is the
operator’s chosen licence on **our weights**, not a licence AI2 granted on the dataset.

A CC BY-NC-SA 4.0 weights release is **open weights with an NC restriction**. It is not
an [Open Source AI](https://opensource.org/ai/open-source-ai-definition) artefact under
the OSI definition, which requires the freedom to use the system for any purpose.

## Training data licences

Weights were trained on a mix of third-party datasets. Datasets themselves are not
shipped with this card. Attribution for every attributable input is collected at
`<URI to ATTRIBUTION.md>` (Title, Author, Source, Licence, whether modified), which
is how we discharge CC §3(a) in a manner reasonable for this medium
([CC BY-SA 4.0 §3(a)(2)](https://creativecommons.org/licenses/by-sa/4.0/legalcode)).

Input **licence families** (not an exhaustive file list; the manifest is the list):

| family | typical tags | role in this release |
|---|---|---|
| Permissive | MIT, Apache-2.0, BSD, CC0, CDLA-Permissive, O-UDA | no restriction on weights; notices retained |
| Attribution | CC BY 4.0, ODC-By 1.0 | notice in the manifest; ODC-By is a database-level copyleft on **data** we do not ship |
| Share-alike | CC BY-SA 3.0 / 4.0 (SNLI, Natural Questions / Wikipedia passages, SQuAD, HotpotQA, and others) | see Reading taken — we do **not** treat weights as Adapted Material of these sets |
| Non-commercial | GooAQ (bespoke README NC over Apache-2.0); other NC-family inputs in the catalogue | honoured as policy by the CC BY-NC-SA 4.0 weights tag |

Every input that contributed rows was **modified** in the ordinary training sense
(filtered, formatted, batched). The operations list in the dataset-factory receipt is
the “indicate if You modified” record.

**Not in this release.** Corpora with no licence grant (including ImageNet-derived
tiny-imagenet), distributors that disclaim owning what they distribute, ND /
research-only / no-redistribution terms, and generators whose terms forbid the use,
are refused at ingest and are not in these weights.

## Reading taken

**We take the permissive reading of the weights question, and we say so.**

Creative Commons defines Adapted Material as material “derived from or based upon the
Licensed Material and in which the Licensed Material is translated, altered, arranged,
transformed, or otherwise modified in a manner requiring permission under the Copyright
and Similar Rights held by the Licensor”
([CC BY-SA 4.0 §1(a)](https://creativecommons.org/licenses/by-sa/4.0/legalcode)).
ShareAlike applies only “if You Share Adapted Material You produce” (§3(b)).

We do **not** treat these trained weights as Adapted Material of the training data.
On that reading, CC BY-SA’s ShareAlike condition does not trigger on the checkpoint,
and the licence on the weights is ours to choose. We choose **CC BY-NC-SA 4.0** as
policy, because one training input (GooAQ) carries a non-commercial term we accepted
on 2026-09-02, and because the project’s strictest-input rule assigns the composed
blob the strictest term among its inputs.

Creative Commons’ own May 2025 primer states both that “in many cases, neither the AI
model nor its outputs would be considered to be derivative works of training data under
copyright law” and that, **when taking a conservative approach**, “the model itself, if
shared publicly, should be made available under the same CC license as the original
works”
([Using CC-licensed Works for AI Training](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/)).
We are taking the first of those, not the second, and we are not silent about it.

This reading is a stated position, not a finding of law. It is the reading that makes
a single CC BY-NC-SA 4.0 tag on a model trained on both CC BY-SA and NC inputs
**coherent**. The cautious reading — that weights **are** Adapted Material — would make
that tag incoherent, because CC BY-NC-SA is not a BY-SA Compatible License
([compatible licences](https://creativecommons.org/share-your-work/licensing-considerations/compatible-licenses/))
and BY-SA §3(b)(3) forbids adding terms that restrict the granted rights. We do not
take that reading here. A reader who requires the cautious reading should not use the
composed checkpoint; they should use a standalone permissive-only region, or wait for
a BY-SA-only or permissive-only composed artefact.

**Not legal advice.** Neither this card nor its authors are a lawyer. If you need
certainty, read the upstream licences and obtain counsel.
````

---

## 3. DRAFT B — model-card text under the cautious reading

*Under Reading C the composed model **as currently mixed** cannot be licensed. The card
must drop a side. Drafts below are for **option (i)** (drop NC, release CC BY-SA 4.0)
and **option (iii)** (permissive-only). Option (ii) is stated in prose, not as a
paste-ready card, because it is the expensive corpus rewrite and the operator has not
chosen it.*

### What option (ii) would require (not drafted as a card)

Drop **every CC BY-SA (and other share-alike) input** from any region that contributes
tensors to the released blob, then release the remainder under **CC BY-NC-SA 4.0**.

Concretely, for the **currently trained** mix that would mean:

- **Drop from `compress`:** SNLI (CC BY-SA 4.0; 58% of `all-nli` as trained). The
  repaired SNLI+government+fiction path is still majority BY-SA, so `compress` would
  need a non-SA substitute (the audit’s candidate is `hkust-nlp/SynCSE-scratch-NLI`,
  MIT, synthetic).
- **Drop from `retrieve` / `memory`:** Natural Questions (Wikipedia passages, CC BY-SA),
  SQuAD, HotpotQA, and FiQA **if** the 2026-09-02 SA reading is kept. Keep GooAQ (NC).
  The region becomes GooAQ-concentrated NC, which is the B1 problem the factory was
  built to dilute.
- **Drop at catalogue scale, if those rows enter training:** Wikipedia-group sets
  (FEVER, DBpedia, MIRACL, Mr.TyDi), Dolly (CC BY-SA), Social Chemistry 101,
  oxford-iiit-pet, Nemotron-Math-Proofs (CC BY-SA), OpenRAIL-M Stack-lineage code
  (a different copyleft family; same “drop share-alike” consequence), StereoSet /
  CrowS-Pairs if used as training rather than probes.
- **Retrain** every affected region; re-compose; re-score gates. GPU time is small;
  corpus reconstruction is the cost.
- **Standalone SA-fed checkpoints cannot ship** as part of the same composed artefact.
  They could still ship **separately** as CC BY-SA 4.0 Collection members, which is
  Option D — but that is not option (ii); that is “don’t compose.”

Option (ii) keeps GooAQ and the 2026-09-02 NC policy, and **loses** the BY-SA
entailment / Wikipedia-passage signal. It is the option that preserves the NC term
at the price of the share-alike corpora. [I]

---

### DRAFT B-(i) — drop NC inputs; composed weights CC BY-SA 4.0

*Paste after the NC inputs have actually been removed from training and the regions
retrained. Do not paste against weights that still contain GooAQ.*

````markdown
## License

**Code** (architecture, training scripts, this repository): MIT.

**Composed model weights:**
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

**Standalone region checkpoints**, when shipped as separate files:

| checkpoint | licence |
|---|---|
| regions whose inputs are only permissive (MIT / Apache-2.0 / BSD / CC0 / CDLA-Permissive / O-UDA) | MIT |
| regions whose inputs include CC BY-SA (e.g. `compress` on SNLI; `retrieve` / `memory` on Natural Questions, SQuAD, HotpotQA) | CC BY-SA 4.0 |
| composed model | CC BY-SA 4.0 |

ShareAlike applies to these weights because we take the **cautious reading** that
trained weights are Adapted Material of CC BY-SA training data
([CC BY-SA 4.0 §1(a), §3(b)](https://creativecommons.org/licenses/by-sa/4.0/legalcode)).
The adapter’s licence on Adapted Material must be a Creative Commons licence with the
same License Elements (Attribution and ShareAlike), this version or later, or a
BY-SA Compatible License. The designated compatibles are Free Art License 1.3 and
GPLv3 (one-way)
([compatible licences](https://creativecommons.org/share-your-work/licensing-considerations/compatible-licenses/)).
**CC BY-NC-SA is not compatible with CC BY-SA for adaptations.** We therefore do not
tag these weights CC BY-NC-SA.

**Non-commercial inputs were removed before training this artefact.** GooAQ (bespoke
README NC), and every other NC-family training input, are **not** in these weights.
A reader looking for the NC-inclusive mix will not find it here; that mix has no
coherent adapter’s licence under the reading this card takes.

Downstream fine-tunes that are Shared inherit CC BY-SA 4.0 (or later / compatible).
API-only serving without distributing weights is a different question; this card
does not take a position on it. Collections of **unmodified** region files may carry
a collection-level licence that does not relicense the files
([CC FAQ, collections](https://creativecommons.org/faq/)).

This CC BY-SA 4.0 weights release is copyleft open weights. Whether it is
[Open Source AI](https://opensource.org/ai/open-source-ai-definition) further depends
on Data Information and Code disclosure; this card supplies Data Information by the
manifest linked below. OSAID’s “use for any purpose” is satisfied by BY-SA in a way
an NC tag would not be.

## Training data licences

Datasets themselves are not shipped (project Rider 2: private-HF-only). Attribution
for every attributable input: `<URI to ATTRIBUTION.md>`.

| family | typical tags | in this artefact? |
|---|---|---|
| Permissive | MIT, Apache-2.0, BSD, CC0, CDLA-Permissive, O-UDA | yes |
| Attribution | CC BY 4.0, ODC-By 1.0 (data-layer copyleft; we do not ship the data) | yes, notice only on weights |
| Share-alike | CC BY-SA 3.0 / 4.0 — SNLI, Natural Questions (Wikipedia passages), SQuAD, HotpotQA, and others listed in the manifest | yes — this is the family that forces CC BY-SA 4.0 on the weights under this reading |
| Non-commercial | GooAQ; other NC-family catalogue entries | **no — dropped** |

CC BY-SA 3.0 inputs (Natural Questions) are adapted under 4.0 by 3.0 §4(b)(ii)
(“a later version of this License with the same License Elements”).

Every attributable input was modified (filtered, formatted, batched). The factory
receipt’s `operations[]` is the “indicate if You modified” record
([§3(a)(1)(b)](https://creativecommons.org/licenses/by-sa/4.0/legalcode)).

**Not in this release.** NC inputs (dropped under this option); corpora with no
licence grant; ND / research-only / no-redistribution; distributors that disclaim
ownership; refused generators.

## Reading taken

**We take the cautious reading of the weights question, and we say so.**

We treat these trained weights as Adapted Material of the CC BY-SA training data
under [CC BY-SA 4.0 §1(a)](https://creativecommons.org/licenses/by-sa/4.0/legalcode).
ShareAlike therefore applies to this Shared checkpoint (§3(b)). Creative Commons’
May 2025 primer, describing its conservative approach: “When training data is
subject to the ShareAlike condition, model outputs and the model itself, if shared
publicly, should be made available under the same CC license as the original works”
([Using CC-licensed Works for AI Training](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/)).
That is the approach this card follows. CC also notes that this “will almost
certainly lead to overcompliance” and “is not intended to take a position on
whether and when copyright applies.” We accept the overcompliance.

Because BY-SA §3(b)(1) requires the same License Elements, and because CC BY-NC-SA
adds NonCommercial and is not a BY-SA Compatible License, **a model trained on both
BY-SA and NC inputs cannot be released under either licence coherently.** We resolved
that by **removing the NC inputs** and releasing under CC BY-SA 4.0. The
permissive reading — weights are not Adapted Material; CC BY-NC-SA 4.0 as a policy
choice — is a reading we considered and did not take for this artefact.

**Not legal advice.** Neither this card nor its authors are a lawyer. If you need
certainty, read the upstream licences and obtain counsel.
````

---

### DRAFT B-(iii) — permissive-only tier; composed weights MIT

*Paste after every NC and every share-alike training input has been removed and the
regions retrained on the clean-permissive tier only. Catalogue §8 is the reach table
for that tier; several faculties do not currently hit the 1e10-token target on it.*

````markdown
## License

**Code** and **composed model weights:** MIT.
Licence text: `<URI to LICENSE>`.

**Standalone region checkpoints:** MIT. No region in this artefact was trained on
CC BY-SA, CC BY-NC, CC BY-NC-SA, ODbL, CDLA-Sharing, OpenRAIL-M, or bespoke NC
inputs.

This is the **clean-permissive tier**. It exists so that a reader who takes the
cautious reading of the weights question (trained weights are Adapted Material of
the training data) still has a composed checkpoint whose adapter’s-licence problem
does not arise, because no ShareAlike or NC parent is in the mix. It is also the
only composed artefact that can sit under MIT without relying on an unsettled
legal question.

CC BY 4.0 inputs, if any remain, impose attribution only; notices are in
`<URI to ATTRIBUTION.md>`. CDLA-Permissive / O-UDA inputs expressly do not
restrict “machine learning models” / “Results”
([CDLA-Permissive-2.0 §3.1, §5.4](https://cdla.dev/permissive-2-0/);
[O-UDA 1.0 §5.4](https://cdla.dev/open-use-of-data-agreement-v1-0/)).

This MIT weights release is intended to be usable as **open weights**. Combined
with this repository’s MIT code and the Data Information in the manifest, it is
the project’s closest approach to the
[Open Source AI Definition](https://opensource.org/ai/open-source-ai-definition).
We do not claim OSAID conformance in this card; that is a separate checklist.

## Training data licences

Datasets themselves are not shipped. Attribution: `<URI to ATTRIBUTION.md>`.

| family | in this artefact? |
|---|---|
| Permissive (MIT, Apache-2.0, BSD, CC0, CDLA-Permissive, O-UDA) | yes — this is the whole mix |
| Attribution (CC BY 4.0) | yes, if listed in the manifest; notice only |
| ODC-By 1.0 | **no** as a merged training file (database-level copyleft on data); if used at all, as a separate Collection file whose Produced-Work notice is in the manifest |
| Share-alike (CC BY-SA, ODbL, CDLA-Sharing, OpenRAIL-M) | **no — dropped** |
| Non-commercial (GooAQ, CC BY-NC / BY-NC-SA, other NC-family catalogue entries) | **no — dropped** |

**Dropped to make this tier, relative to the 2026-09-02 composed mix:** GooAQ;
SNLI; Natural Questions; FiQA; SQuAD; HotpotQA; and every later NC or share-alike
catalogue admit that would otherwise have entered a region. Replacements used
instead are listed in the manifest (for `compress`, a permissive entailment
source; for `retrieve` / `memory`, the clean-permissive retrieval sets that
survive; for `vl_latent`, the licence-clean vision composite, never tiny-imagenet).

## Reading taken

**This artefact does not depend on a reading of the weights question.**

No ShareAlike parent is present, so CC BY-SA §3(b) cannot fire even if weights
are Adapted Material. No NC parent is present, so no NC term has to be expressed
on the weights. We still record the two readings, because other CogSynDelta
artefacts take one or the other:

- **Permissive reading:** weights are not Adapted Material; licence on weights is
  the operator’s choice. A sister artefact may ship CC BY-NC-SA 4.0 under that
  reading.
- **Cautious reading:** weights are Adapted Material; a mix of BY-SA and NC
  inputs cannot be licensed as one blob. This artefact is option (iii) of that
  reading — the permissive-only tier.

Creative Commons’ May 2025 primer states both readings and does not choose
([Using CC-licensed Works for AI Training](https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/)).
We do not choose here either, because the mix does not force a choice.

**Not legal advice.** Neither this card nor its authors are a lawyer. If you need
certainty, read the upstream licences and obtain counsel.
````

---

## 4. Table

*“Current inputs” = the mix the 2026-09-02 decision actually licensed: GooAQ (NC) +
CC BY-SA sets (SNLI, Natural Questions, FiQA-as-then-read, and the planned SQuAD /
HotpotQA / oxford-iiit-pet) + permissive sets. Catalogue-scale extras are footnoted.
`vl_latent` / tiny-imagenet is **out of scope** of this table: it is a missing-grant
problem, not a reading-of-Adapted-Material problem, and no licence choice repairs it.*

| reading | release licence of the composed model | which current inputs must be removed | what changes for submodel repos |
|---|---|---|---|
| **P — permissive** (weights are **not** Adapted Material). Matches 2026-09-02 once the card **states** it. | **CC BY-NC-SA 4.0** (operator’s policy choice, not a fired adapter’s licence) | **None for licence-coherence.** Still refuse BLOCKING / REFUSE classes (tiny-imagenet, no-grant, ND, distributor-disclaims-ownership). | Per-region tags in the 2026-09-02 table stand: MIT for permissive-only regions; CC BY-SA 4.0 on `compress` as **policy overlay**; CC BY-NC-SA 4.0 on `retrieve`/`memory` and on the composed blob. Attribution manifest in every repo. Rider 1: a future region merge inherits the most restrictive parent. |
| **C-(i) — cautious, drop NC** | **CC BY-SA 4.0** | **All NC training inputs.** Current mix: **GooAQ** (required). FiQA if the catalogue’s NC reading is used. Catalogue-scale if admitted: ANLI, MS MARCO, SICK, BeaverTails, PKU-SafeRLHF, KodCode, camel-ai/math, MetaMathQA, orca-math, GEM/opusparcus, MathInstruct-as-whole, and any other `verdict: NC`. Retrain `retrieve`/`memory` without GooAQ. | `retrieve`/`memory` lose their NC reason to exist as a separate licence; they become CC BY-SA 4.0 (they still have NQ / SQuAD / HotpotQA). Permissive-only repos stay MIT. `compress` stays CC BY-SA 4.0, now as a **legal** necessity under this reading, not a policy overlay. No repo may tag weights CC BY-NC-SA. |
| **C-(ii) — cautious, drop SA** | **CC BY-NC-SA 4.0** | **All CC BY-SA (and other share-alike) training inputs.** Current mix: **SNLI, Natural Questions, SQuAD, HotpotQA**; FiQA if still treated as SA; oxford-iiit-pet if it entered `vl_latent`. Catalogue-scale: Wikipedia group, Dolly, Social Chemistry 101, Nemotron-Math-Proofs, OpenRAIL-M Stack-lineage, StereoSet/CrowS-Pairs if trained. Retrain `compress` on a non-SA substitute. | `compress` cannot ship on SNLI; it becomes MIT on a permissive substitute or is held. `retrieve`/`memory` become NC-only (GooAQ-driven CC BY-NC or BY-NC-SA) and **more** GooAQ-concentrated. Permissive-only repos stay MIT. No repo may tag weights CC BY-SA. Option D (don’t compose SA-fed and NC-fed files) is the alternative to actually dropping rows. |
| **C-(iii) — cautious, permissive-only tier** | **MIT** (or Apache-2.0) | **All NC and all share-alike training inputs** — the union of C-(i) and C-(ii) removals. Current mix loses GooAQ, SNLI, NQ, FiQA, SQuAD, HotpotQA. | Every submodel repo is MIT. This is the only set of repos with **no** open Layer-2 question. Catalogue §8: several faculties miss the 1e10-token target on this tier (`memory` 1.3%, `visual` 0.3% unless Commons is pulled). Reach is an engineering fact, not a licence one. |

**Collection escape, all cautious rows.** If SA-fed and NC-fed **region checkpoints
remain separate files** and the composed blob is **not shipped**, CC’s collections
rule lets each file keep its own licence. [V] https://creativecommons.org/faq/
(“You may choose a license for the collection, however this does not change the
license applicable to the original material.”). That is Option D in
`LICENCE-FOR-OPEN-WEIGHTS.md`. It is how Reading C becomes survivable **without**
dropping a side. It is **not** a licence for a merged weights blob. Rider 1 still
applies if the files are later merged.

---

## 5. What the operator must decide, and what counsel should answer

Phrased as questions. This document answers none of them.

### Operator decisions (policy, not counsel)

1. **Which reading does this project take in public — P, C, or “we ship both artefacts”?**
   The 2026-09-02 licence is P in substance and silent in the card. `20-enrichment` §2.3
   and catalogue §10.1 #1 already asked that the silence be closed.
2. **If P: paste Draft A against the current mix, and keep GooAQ?** That is the
   zero-corpus-change path. It accepts the risk that a later Adapted-Material holding
   makes the CC BY-NC-SA tag incoherent.
3. **If C: which of (i) drop NC, (ii) drop SA, (iii) permissive-only, or Option D
   (don’t compose) is the artefact we actually ship?** (i) and (iii) have paste-ready
   cards above. (ii) is the GooAQ-preserving rewrite. Option D ships two (or three)
   region files and **no** composed blob.
4. **Do standalone region repos keep distinct licences, or does every public checkpoint
   inherit the composed tag?** Rider 1 says a merge inherits the cost; it does not
   force standalone files to wear the merge’s tag.
5. **Is “open weights, NC accepted” still the optimisation target**, knowing it fails
   OSAID’s “use for any purpose”, or do we want an OSAID-shaped MIT/BY-SA artefact
   even if that drops GooAQ or drops composition?
6. **FiQA: SA (2026-09-02) or NC (catalogue 2026-09-03)?** Either way the composed
   P-vs-C contradiction survives because of GooAQ × SNLI/NQ. The call still changes
   `retrieve`’s **standalone** tag under C-(i) vs C-(ii).
7. **MS MARCO and the other catalogue NC admits: in or out?** They do not create the
   contradiction (GooAQ already did); they only enlarge the NC side of C-(i)’s drop
   list. MS MARCO also carries “we may not own the underlying rights,” which is the
   REFUSE-class question, not the reading question.
8. **Who ships the attribution manifest, and under what URI, before the card is
   published?** Drafts A and B both depend on a real `ATTRIBUTION.md`. A card that
   claims §3(a)(2) and 404s is worse than a silent card.

### Questions for a counsel reading

9. **Are trained model weights “Adapted Material” under CC BY-SA 4.0 §1(a) of the
   training data, in the jurisdictions this project actually ships into?** This is
   the question. CC’s 2025 primer states both answers and disclaims a position.
10. **If they are not Adapted Material, may the operator still apply CC BY-NC-SA 4.0
    to the weights as a policy licence, and does that policy licence create
    contractual duties to downstream users that copyright would not?**
11. **If they are Adapted Material, is there any adapter’s licence that simultaneously
    satisfies CC BY-SA 4.0 §3(b)(1)/(3) (same License Elements; no additional
    restrictions) and GooAQ’s README NC sentence?** This document’s answer is no;
    that is an inference, not counsel.
12. **Is GooAQ’s operative term the Apache-2.0 LICENSE file or the README NC note,
    and does that term bind a downstream recipient of **weights** (not of rows)?**
    The 2026-09-02 decision accepted the NC reading without an AI2 reply; counsel
    may still want that email.
13. **Does §2(a)(5)(C) (no downstream restrictions on Licensed Material) or only
    §3(b)(3) (no additional terms on Adapted Material) speak to tagging weights
    NC when the parent is BY-SA?** The two clauses have different objects.
14. **Does the CC FAQ collections rule let separately-shipped region checkpoints
    keep distinct licences when a README presents them as “CogSynDelta”, and does
    a later merge of those files into one blob re-open §3(b)?** Rider 1 assumes yes
    on the merge; that is an inference.
15. **Do sui generis database rights apply to a US-domiciled operator at all, and
    if not, does that change Layer 1 (enriched datasets) without changing Layer 2
    (weights)?** Catalogue §10.1 #7.
16. **Is a machine-learning model a “Produced Work” under ODC-By / ODbL**, and does
    that answer independently of the CC Adapted-Material question? Catalogue §10.1 #5.
    Relevant to FineWeb-Edu, C4, peS2o, OpenWebMath, WildGuard — not to the
    GooAQ × SNLI contradiction, but to whether the permissive-only tier is as
    clean as Draft B-(iii) claims if ODC-By rows are included.
17. **Does Stack Exchange’s per-item attribution clause reach an inference-time
    weight release?** Catalogue §10.1 #4. If no, the largest anti-GooAQ lever
    re-opens; if yes, R9 stays.
18. **Which jurisdiction’s answer to question 9 does this project need to survive?**
    The one judicial holding on weights as copies (`Getty Images v. Stability AI`
    [2025] EWHC 2863 (Ch) ¶599–600, UK secondary infringement) is not CC ShareAlike
    and is not US law. [V] cited in `LICENCE-FOR-OPEN-WEIGHTS.md`. Counsel should
    say whether it is even relevant.

---

## Appendix — URLs fetched for this file (2026-09-03)

| source | URL |
|---|---|
| CC BY-SA 4.0 legalcode | https://creativecommons.org/licenses/by-sa/4.0/legalcode |
| CC BY-NC-SA 4.0 legalcode | https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode |
| CC BY-SA 4.0 legalcode.txt (local copy) | `/akula-data/session-backup-staging/dataset-factory/licence-texts/cc-by-sa-4.0.plain.txt` |
| CC FAQ | https://creativecommons.org/faq/ |
| CC FAQ local copy | `licence-texts/cc-faq.txt` |
| CC compatible licences | https://creativecommons.org/share-your-work/licensing-considerations/compatible-licenses/ |
| CC, *Using CC-licensed Works for AI Training* (HTML) | https://creativecommons.org/using-cc-licensed-works-for-ai-training-2/ |
| CC, *Using CC-licensed Works for AI Training* (PDF, May 2025) | https://creativecommons.org/wp-content/uploads/2025/05/Using-CC-licensed-Works-for-AI-Training.pdf |
| OSI Open Source AI Definition 1.0 | https://opensource.org/ai/open-source-ai-definition |
| GooAQ README (NC note) | https://raw.githubusercontent.com/allenai/gooaq/main/README.md |
| CDLA-Permissive-2.0 | https://cdla.dev/permissive-2-0/ |
| O-UDA 1.0 | https://cdla.dev/open-use-of-data-agreement-v1-0/ |

Local copies of the CC legalcode and FAQ were already on disk from the 2026-09-03
enrichment pass; the HTML legalcode and the 2025 primer were re-fetched this session
and matched those copies on the clauses quoted above.
