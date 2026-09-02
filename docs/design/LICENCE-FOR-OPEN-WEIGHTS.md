# Licence Audit for an Open-Weights Release

**Status:** audit. This informs a decision; it does not make one.
**Scope:** every corpus CSD has trained on, plus the 12 catalogued corpora fetched to
`/bulk/csd-corpus`.
**Nothing here changes a catalogue verdict.** `scripts/csd-corpus-expand.py` is untouched.

> **Neither the author of this document nor its reader is a lawyer.** Everything below
> reports what a licence *says* and what the *risk* is. Where a question is legally
> unsettled it is marked unsettled and left for a human. No sentence here is a legal
> conclusion, and every place where one would be load-bearing is called out as needing a
> human decision rather than quietly assumed one way.

---

## The short version

**All four trained regions are BLOCKING under the new criterion. None of the four corpora
in actual use supports redistributing derived weights under a permissive licence.**

| region | corpus | why it blocks | repairable? |
|---|---|---|---|
| `vl_latent` | tiny-imagenet | ImageNet-derived; **no licence anywhere in the chain**; ImageNet's own terms say non-commercial | **No.** 100% of the corpus. Needs a different pretraining set |
| `compress` | all-nli | 41.8% MultiNLI-derived; 3 of its 5 training genres have named commercial copyright holders; genre column destroyed so it cannot be filtered | Partly — re-derive from `nyu-mll/multi_nli`, which still has `genre`. Gets to SHARE_ALIKE, not clean |
| `code` | `Nan-Do/code-search-net-python` | mirror asserts `apache-2.0` over an upstream tagged `other`, with no LICENSE file; ~15% of rows are GPL/AGPL | **Yes, cheaply.** The `repo` column survives; ~71% of rows are in permissive repos |
| `retrieve` | gooaq + NQ + fiqa | GooAQ's upstream README says *"should not be used for any commercial purposes"* while its LICENSE file is Apache-2.0; the other 22.2% is share-alike | **Unknown.** Depends on which GooAQ term governs — one email to AI2 |

Two of these were not visible before this audit and are the reason it was worth doing:

1. **GooAQ's upstream contradicts itself**, and the permissive half is the only half that
   reached this project. It is 77.8% of `retrieve`.
2. **~15% of the `code` corpus is GPL/AGPL** source code. That is a larger copyleft exposure
   than all the CC BY-SA data combined, and nothing in the pipeline surfaced it.

Also flagged: **`nlphuji/flickr30k`, which P3.2 plans to add to `vl_latent`, is BLOCKING
too.** Adding it would give the region a second licence problem rather than fixing its
first.

**The single most expensive fact:** `vl_latent` is 100% dependent on an unlicensed corpus,
and P3.1 plans to scale it. Every GPU-hour spent scaling that region before the corpus is
replaced is spent on weights that cannot be released.

---

## Why this is a different question from the one already answered

The catalogue's gate asks: **may I train on this?** It rejects non-commercial terms,
research-only terms, and unknown licences, and it refuses to be overridden. That gate is
good and it has already done real work — MS MARCO, ELI5, `BeIR/scifact` and
`hendrycks/competition_math` were all turned down by it.

The open-weights release asks a harder question: **may a model derived from this corpus be
redistributed under a permissive licence?**

Three things separate the two:

1. **Direction of obligation.** "I may use this" is a permission. "I may relicense what I
   built from it under MIT" is a claim about what I can grant *onward*, to people who never
   agreed to the upstream terms. A corpus can permit the first and say nothing about the
   second.
2. **Copyleft.** Share-alike is irrelevant to whether you may train. It is the whole
   question when you redistribute. CC BY-SA passed the old gate without comment; it is the
   single largest category of exposure under the new one.
3. **Attribution survives the model.** CC BY and CC BY-SA both require attribution from
   anyone who distributes adapted material. If weights are adapted material, the model card
   is where that obligation lands — and a model card that omits it is non-compliant in a way
   that is trivially checkable by anyone.

A corpus can be `TRAIN_OK` and still make an MIT release awkward, contingent, or
unavailable. Establishing which corpora are in that set is the entire point of this
document.

---

## What is actually being trained on

Not the catalogue — the runner. `scripts/csd-train-all.py` `REGIONS` and `VL_REGIONS` are
the ground truth for what has touched a weight. Measured from the receipts in
`/akula-data/csd/receipts`, 2026-09-02:

| region | corpus (as wired) | source rows | trained pairs | params | wall time | receipt metric |
|---|---|---|---|---|---|---|
| `code` | `Nan-Do/code-search-net-python` | 455,243 | 430,931 | 16.0M | 1,087 s | recall@1 0.9766 |
| `compress` | `sentence-transformers/all-nli` (`pair`) | 314,315 | 277,269 | 16.0M | 427 s | recall@1 0.7070 |
| `retrieve` | fiqa-pairs + NQ + gooaq (capped) | 514,362 | 505,216 | 16.0M | 595 s | recall@1 0.7480 |
| `vl_latent` | `zh-plus/tiny-imagenet` | 100,000 imgs | 100,000 | 22.9M | 497 s | probe top-1 0.0606 |

**Fetched and not trained on:** `sentence-transformers/stsb` (the graded gate never ran —
`config.graded_shards` is `[]`, recorded as P0.9c), `nlphuji/flickr30k` (planned as P3.2),
and all 12 corpora under `/bulk/csd-corpus`.

Total GPU cost of the entire trained fleet: **2,606 seconds — 43 minutes.** This number
matters more than it looks, and it reappears in the ranking section.

---

## Verdict categories

| verdict | meaning for an MIT weights release |
|---|---|
| `PERMISSIVE_OK` | MIT / Apache-2.0 / BSD / CC0 / ODC-BY at BOTH mirror and upstream. Nothing to carry forward. |
| `ATTRIBUTION` | CC BY-family. Usable; the release MUST carry a specific attribution notice. The notice is written out verbatim later in this document. |
| `SHARE_ALIKE` | CC BY-SA-family. Whether trained weights are "adapted material" is **unsettled**. Flagged, quantified, not resolved. |
| `BLOCKING` | Non-commercial, research-only, unknown, or no licence at all. Cannot support an open-weights release without replacing the corpus. |

`BLOCKING` is not a claim that use is unlawful. It is a claim that **nothing grants the
right being relied on**, which is a different and much easier thing to establish — and it
is the honest state of an unlicensed corpus.

---

## Per-corpus verdicts — corpora in actual use

Licence strings below were observed live against the HF API on 2026-09-02, not copied from
the catalogue. Every "upstream" row was checked at the source.

### `code` — `Nan-Do/code-search-net-python`

| field | observed |
|---|---|
| mirror licence tag | `apache-2.0` (HF API `cardData.license`, and the card's own frontmatter) |
| LICENSE file in repo | **none** — repo contains only `.gitattributes`, `README.md`, `data/` |
| upstream | `code-search-net/code_search_net`, HF tag `license: other` |
| upstream-of-upstream | github/CodeSearchNet: *"This code and documentation for this project are released under the MIT License."* — scoped to **code and documentation**, not to the data |
| old verdict | not in the catalogue under this repo id. The catalogue **REJECTED** `code-search-net/code_search_net:go` and noted in its own `why`: *"The Python config already training the `code` region has the same unresolved status"* |
| **NEW verdict** | **BLOCKING** |

**This is the same failure mode the catalogue was built to catch.** The catalogue's
docstring states the rule: *"A mirror's licence tag is not evidence about its upstream."*
It cites `BeIR/scifact` (tagged `cc-by-sa-4.0`, mirroring `allenai/scifact` tagged
`cc-by-nc-2.0`) as the proof case. `Nan-Do/code-search-net-python` is that pattern exactly:
a mirror asserting `apache-2.0` over a corpus whose own upstream declares `other`, with no
LICENSE file and no explanation on the card for where the Apache grant came from.

It is worse in one specific respect. CodeSearchNet's README says:

> *"The licenses for source code used as data for this project are provided with the data
> download for each language in `_licenses.pkl` files."*

So CodeSearchNet **did** ship per-repository licence metadata. The mirror dropped it. The
`apache-2.0` tag is therefore not merely unsupported — it replaced the actual licence data
with a single wrong string.

**But this one is recoverable, and cheaply.** The corpus retains a `repo` column. Measured:

```
rows 455,243   distinct repos 13,581
top: saltstack/salt 11,111 · mitsei/dlkit 2,551 · materialsproject/pymatgen 2,196
```

Resolving current GitHub licences for 300 repos (the 150 largest by row count, covering
25.5% of the corpus, plus an unbiased random sample of 150) gives two independent estimates
that agree closely:

| licence class | head 150 (25.5% of rows) | random tail 150 |
|---|---|---|
| permissive (MIT / Apache-2.0 / BSD / ISC) | **71.3%** of rows | **72.0%** of rows |
| weak copyleft (LGPL / MPL / EPL) | 5.6% | 11.0% |
| **strong copyleft (GPL / AGPL)** | **15.9%** | **14.9%** |
| unresolvable (repo deleted, or no licence) | 2.8% | 1.9% |
| `other` | 4.4% | ~0% |

Two findings, and the second is the one that matters:

- **~71% of the corpus sits in permissively licensed repositories.** Filtering to those
  leaves ~324,000 pairs — still well above the 214,813 that first produced recall@1 0.94.
  A permissive-only `code` corpus is constructible from data already on disk.
- **~15% is GPL or AGPL.** That is roughly 72,000 functions of strong-copyleft source code
  in the training set of a model intended for MIT release. Whether that matters depends
  entirely on the unsettled question in the next section, but it is a materially larger
  exposure than the CC BY-SA question and it was not visible before this audit.

### `compress` — `sentence-transformers/all-nli` (`pair` config)

| field | observed |
|---|---|
| mirror licence tag | **none.** HF API returns `license: null`, empty tag list |
| LICENSE file in repo | **none** — repo contains `.gitattributes`, `README.md`, and the four config directories |
| the string "license" in the card | **does not appear at all** |
| org-wide | **0 of 96 `sentence-transformers` datasets carry a licence tag.** Not one |
| upstream A | `stanfordnlp/snli` — `cc-by-sa-4.0` |
| upstream B | `nyu-mll/multi_nli` — **four licences simultaneously**: `cc-by-3.0`, `cc-by-sa-3.0`, `mit`, **`other`** |
| old verdict | never gated. It is not in the catalogue at all; it predates it |
| **NEW verdict** | **BLOCKING as trained.** Repairable to SHARE_ALIKE — see below |

The card states its own construction:

> *"This dataset is a concatenation of the SNLI and MultiNLI datasets."*
> `pair`: *"…considering the 'premise' as the 'anchor' and the 'hypothesis' as the
> 'positive' if the label is 'entailment'."*

The composition was measured directly against the shards on disk:

```
all-nli pair/train rows            314,315   (unique pairs 313,932)
    also present in SNLI train     182,792   58.23%   -> CC BY-SA 4.0
    NOT present in SNLI            131,140   41.77%   -> MultiNLI-derived
```

and the arithmetic closes exactly against the upstreams: SNLI train entailment 183,416 +
MNLI train entailment 130,899 = **314,315**, to the row.

**The MultiNLI share decomposes by genre, and this is where the problem lives.** MultiNLI's
training split draws from exactly five genres. Their rights holders, per the OANC End User
Licence (`https://www.anc.org/OANC/license.txt`, Appendix I), are:

| genre | rows in `pair` train | share | rights holder as stated |
|---|---|---|---|
| telephone (Switchboard) | 27,782 | 8.84% | *"Copyright (c) 1997-2002 Trustees of the University of Pennsylvania"* |
| fiction | 25,784 | 8.20% | mixed: CC BY-SA 3.0, CC BY 3.0, and US public domain |
| government | 25,783 | 8.20% | asserted public domain; **not in Appendix I, no holder named** |
| travel (Berlitz) | 25,782 | 8.20% | *"Copyright (c) 2003 Langenscheidt Publishers"* |
| slate | 25,768 | 8.20% | *"Copyright (c) 1996-2000 Microsoft Corporation"* |

Three of the five are third-party commercial copyright holders. That is the finding.

**A correction to the premise this audit started from.** The concern flagged in advance was
Oxford University Press text. OUP — along with Verbatim, Letters, the 9/11 Report and
Face-to-Face — appears **only in MultiNLI's `validation_mismatched` split**, which maps to
all-nli's **test** split, not its train split. `compress` trains on `pair/train*.parquet`
only. **So OUP is not in the trained corpus.** Switchboard is, and it is the largest single
MultiNLI genre at 27,782 rows.

The presence was confirmed independently by regex over the 314,315 anchors in the shard
`compress` actually trained on — 2,178 disfluency hits and 1,466 government-register hits,
a conservative lower bound against the true 27,782 and 25,783 — with verbatim examples:

> `um-hum yeah i end up well yeah i mean i do a lot of like even weekender kind of things…`
> `For certain types of work to be performed at an agency, GAO may initially provide only telephone or e-mail message notification.`

**MultiNLI's own paper claims permissive terms**, and is worth quoting because it is the
only affirmative grant anywhere in this chain:

> *"The majority of the corpus is released under the OANC's license, which allows all
> content to be freely used, modified, and shared under permissive terms."*

But the OANC's actual licence file is not what that sentence implies. It is a signed
bilateral End User Licence executed by the Linguistic Data Consortium on behalf of the ANC
Consortium. Its permission to redistribute a *"transparently modified"* version — a category
that expressly includes *"excerption, change in format, typographical correction, annotation
of linguistic structure or content"*, which is exactly what MultiNLI did — is conditioned on
two things, verbatim:

> *"i. The text of this Agreement must be displayed in human-readable form in the
> Transparently Modified Versions.
> ii. The reproduction must contain acknowledgement of the original author(s) and
> publisher(s), as specified in Appendix I, according to normal citation practices."*

Neither condition is met anywhere in the chain that reaches this project, and it sits oddly
beside anc.org's website prose that the data is *"fully open and unrestricted for any use."*
Where the website and the licence file disagree, this audit applies the rule the catalogue
already uses for mirrors: the document that actually grants rights wins.

**One clause cuts the other way and should be recorded, because it is favourable.** The
same agreement states:

> *"Mere aggregation of the O-ANC Processed Material or a portion thereof with other texts
> or works not listed in Appendix I shall not cause this Agreement to apply to those other
> texts or works. The aggregate work shall contain a notice specifying the inclusion of the
> O-ANC Processed Material and any and all appropriate copyright notices."*

That is an aggregation carve-out, not a copyleft. It does not reach out and bind unrelated
material — it asks for a notice. So the OANC portion's obligation, on its face, is
**attribution, not share-alike**, and it is one this project could actually satisfy. What
makes it unsatisfiable *today* is not the licence's severity but the missing `genre`
column: you cannot write "acknowledgement … as specified in Appendix I" when you no longer
know which of the seven Appendix I entries your rows came from.

**As distributed, it cannot be filtered.** The shard's columns are exactly
`['anchor', 'positive']`. There is no `genre` column — and `promptID` and `pairID`, the two
keys that would permit a join back to genre, are dropped as well. There is not even a
source column: a row cannot be attributed to SNLI vs MultiNLI, let alone to a genre.

**But it CAN be re-derived, and this is the repair path.** `nyu-mll/multi_nli` still exposes
`genre` per row. Re-deriving the entailment pairs from `nyu-mll/multi_nli` and
`stanfordnlp/snli` directly — instead of consuming the pre-flattened
`sentence-transformers/all-nli` — restores the ability to drop genres. Doing so would leave:

- SNLI: 183,416 pairs, CC BY-SA 4.0
- MNLI government: 25,783 pairs, asserted public domain (unverified — no holder named)
- MNLI fiction: 25,784 pairs, mixed CC BY-SA 3.0 / CC BY 3.0 / public domain

≈ 235,000 pairs, versus 277,269 trained today. **That converts `compress` from BLOCKING to
SHARE_ALIKE at a cost of ~15% of the corpus** — it does not make it MIT-clean, because
SNLI is still 78% of what remains and SNLI is CC BY-SA 4.0.

**And SNLI's own CC BY-SA grant is weaker than it looks.** SNLI's page states:

> *"The corpus is available under a CreativeCommons Attribution-ShareAlike license, the
> same license used for the Flickr30k source captions."*

No Flickr30k source corroborates that. The Flickr30k distribution page states no
affirmative licence on the caption text at all, and says of the images: *"We do not own the
copyright of the images. They are solely provided … for researchers and educators who wish
to use the dataset for non-commercial research and/or educational purposes."* Stanford's
"also released under an Attribution-ShareAlike licence" appears to be a unilateral
assertion with no locatable upstream source. SNLI also contains roughly 4,000 Visual
Genome premises, which are CC BY 4.0 — a different licence inside a CC BY-SA corpus, which
no SNLI document acknowledges.

### `retrieve` — three sources, none of them clean

| source | rows | share | mirror tag | upstream | verdict |
|---|---|---|---|---|---|
| `sentence-transformers/gooaq` | 400,000 (capped from 3,012,496) | **77.8%** | none | `allenai/gooaq` — **the LICENSE and the README contradict each other** | **BLOCKING (contested)** |
| `sentence-transformers/natural-questions` | 100,231 | 19.5% | none | `google-research-datasets/natural_questions`, `cc-by-sa-3.0` | **SHARE_ALIKE** |
| fiqa-pairs (`BeIR/fiqa` ⋈ `BeIR/fiqa-qrels`) | 14,131 | 2.7% | `cc-by-sa-4.0` both | FiQA 2018 challenge | **SHARE_ALIKE** |

**GooAQ was expected to be the good news in this audit. It is not.**

The project's own fleet manifest records the basis for using it: *"Apache-2.0 per the
upstream github.com/allenai/gooaq LICENSE file."* That statement is accurate about the
LICENSE file — it is a verbatim, unmodified, stock Apache-2.0, with the boilerplate
`Copyright [yyyy] [name of copyright owner]` placeholder never even filled in.

**But the same repository's README says the opposite.** Line 5, verbatim, fetched as raw
bytes from `raw.githubusercontent.com/allenai/gooaq/main/README.md`:

> **`**NOTE** This dataset should not be used for any commercial purposes. See the
> [license](LICENSE) for the detailed terms.`**

That sentence points at a licence which does not contain the term it announces. Apache-2.0
grants commercial use explicitly; the README forbids it and cites the Apache file as its
authority. The two cannot both be operative, and **nothing in the repository resolves which
governs.**

Supporting observations, none of them decisive:

- The repository has been untouched since 2021-07-23. This is not a note in flight; it is
  the settled state of the source.
- GitHub's own licence detector reports `license: null` for the repository, despite the
  stock Apache-2.0 file being present.
- The HF mirror `allenai/gooaq` propagates only the permissive half: tag `apache-2.0`, and a
  Licensing Information section reading *"Licensed under the Apache License, Version 2.0."*
  **The non-commercial note does not appear on the HF card at all.**
- `sentence-transformers/gooaq`, the copy actually trained on, declares no licence.
- The paper describes questions collected via Google auto-complete and answers *"collected
  from Google's answer boxes"*. Neither repo nor paper says anything about the copyright
  status of that third-party snippet text.

**This is the same class of error the catalogue exists to catch, one layer further out.**
The catalogue's rule is that a mirror's tag is not evidence about its upstream. Here the
*upstream's own LICENSE file* is not evidence about the upstream's own stated terms. The
project checked the LICENSE and stopped there — which was one step further than most people
go, and still one step short.

**Consequence: `retrieve` has no clean fraction.** 77.8% is contested-non-commercial and
22.2% is share-alike. The cheap fix that this audit expected to recommend — drop the
share-alike sources, uncap gooaq — **does not exist**, because uncapping gooaq concentrates
the corpus onto the contested source rather than away from it.

**What a human has to decide:** whether the README's NOTE or the LICENSE file governs. If
the README governs, GooAQ is non-commercial and belongs in the same category as MS MARCO,
which this project already refused on exactly those grounds. If the LICENSE governs, GooAQ
is `PERMISSIVE_OK` and `retrieve` becomes the cheapest region to fix. **The correct next
action is to ask AI2 directly** — this is a one-email question with a definitive answer,
and it is worth far more than any amount of further inference.

### `vl_latent` — `zh-plus/tiny-imagenet`

**This is the most serious finding in the audit, and it is the one the task predicted.**

| field | observed |
|---|---|
| mirror licence tag | **empty list** — `license: []`. Not "unknown". Not "other". *Absent.* |
| `dataset_infos.json` `license` field | `''` (empty string) |
| LICENSE file in repo | **none** |
| `source_datasets` tag | `extended\|imagenet-1k` |
| `extra_gated_prompt` | **present** — reproduces ImageNet's Terms of Access verbatim |
| repo actually gated? | **No.** `gated: False` |
| old verdict | never gated. Not in the catalogue; it predates it |
| **NEW verdict** | **BLOCKING** |

The card carries ImageNet's Terms of Access as its gating text, including, verbatim:

> *"Researcher shall use the Database only for non-commercial research and educational
> purposes."*

and

> *"If Researcher is employed by a for-profit, commercial entity, Researcher's employer
> shall also be bound by these terms and conditions…"*

That text was verified independently at the source. `https://www.image-net.org/download.php`
states the same clause verbatim today, and adds that the researcher must indemnify the
ImageNet team against claims arising from *"Researcher's use of any copies of copyrighted
images that he or she may create from the Database"* — i.e. ImageNet does not assert that it
owns the images, and does not purport to grant rights in them.

So the provenance chain is:

```
ImageNet (non-commercial research/educational; images individually copyrighted by third parties)
  -> Tiny ImageNet (200-class 64x64 subset; no licence grant of its own found)
    -> Maysee/tiny-imagenet (no licence; carries ImageNet's ToA text)
      -> zh-plus/tiny-imagenet (no licence; carries ImageNet's ToA text)
        -> vl_latent's ENTIRE pretraining corpus
```

The gate being switched off on HF is a Hub configuration detail. It is not a licence grant,
and nothing in the chain replaces the missing one. **There is no licence under which
`vl_latent`'s weights could be redistributed** — not because a licence forbids it, but
because no licence was ever granted, and the only terms text anywhere in the chain says
non-commercial.

`vl_latent` is 100% dependent on this corpus. There is no clean fraction to keep.

### `nlphuji/flickr30k` — fetched, not yet trained, and planned as P3.2

Not in use today, which is the only reason it is not on the BLOCKING list above. **P3.2
plans to add it as extra `vl_latent` pretraining data, and it should not be.**

| field | observed |
|---|---|
| mirror licence tag | **none.** No licence tag, no LICENSE file, and the card contains no licence text at all |
| what the mirror ships | `flickr30k-images.zip` — roughly 30 GB of the actual images, 4.4 GB on this fleet's disk |
| upstream | shannon.cs.illinois.edu/DenotationGraph/, gated behind a request form |

The Illinois distribution page states, verbatim, on four separate pages:

> *"The Flickr 30k Dataset includes images obtained from Flickr. Use of the images must
> abide by the Flickr Terms of Use. **We do not own the copyright of the images. They are
> solely provided at the link below for researchers and educators who wish to use the
> dataset for non-commercial research and/or educational purposes.**"*

**NEW verdict: BLOCKING**, on the same grounds as tiny-imagenet and for the same reason —
the distributor disclaims ownership of the images and scopes their provision to
non-commercial research. Adding it to `vl_latent` would not fix the region's licence
problem; it would give it a second one.

Two further notes:

- The restriction is textually scoped to the **images**. The captions are offered
  separately as a *"Publicly Distributable Version … (tokenized captions only)"*, and no
  explicit no-redistribution clause was found. But the page states no affirmative licence
  on the caption text either. For an I-JEPA pretraining objective it is the images that
  matter, so the distinction does not help here.
- This is also the upstream of SNLI's captions, which is why the SNLI CC BY-SA claim
  discussed under `compress` cannot be corroborated.

---

## Per-corpus verdicts — catalogued and fetched, not yet trained

These sit in `/bulk/csd-corpus` with manifests. None has touched a weight, so the cost of a
`BLOCKING` verdict here is zero today and rises the moment P2.3 runs.

| corpus | region | licence observed | upstream | old verdict | **NEW verdict** |
|---|---|---|---|---|---|
| `openai/gsm8k` (`main`) | reason | `mit` | original | TRAIN_OK | **PERMISSIVE_OK** |
| `deepmind/aqua_rat` (`raw`) | reason | `apache-2.0` | original | TRAIN_OK | **PERMISSIVE_OK** |
| `google-research-datasets/go_emotions` | classify | `apache-2.0` | original | TRAIN_OK | **PERMISSIVE_OK** |
| `zalando-datasets/fashion_mnist` | vl | `mit` | original | TRAIN_OK | **PERMISSIVE_OK** |
| `codeparrot/apps` | code | `mit` | — | TRAIN_OK | **PERMISSIVE_OK**, with a provenance caveat below |
| `PolyAI/banking77` | classify | `cc-by-4.0` | GitHub LICENSE agrees | TRAIN_OK | **ATTRIBUTION** |
| `deepmind/code_contests` | code | `cc-by-4.0` | mixed, see below | TRAIN_OK | **ATTRIBUTION**, with an unresolved sub-licence |
| `rajpurkar/squad` | retrieve | `cc-by-sa-4.0` | Wikipedia text | TRAIN_OK | **SHARE_ALIKE** |
| `BeIR/hotpotqa` (`corpus`) | retrieve | `cc-by-sa-4.0` | hotpotqa.github.io agrees | TRAIN_OK | **SHARE_ALIKE** |
| `BeIR/hotpotqa` (`queries`) | retrieve | `cc-by-sa-4.0` | hotpotqa.github.io agrees | TRAIN_OK | **SHARE_ALIKE** |
| `stanfordnlp/snli` | compress | `cc-by-sa-4.0` | Flickr30k captions | TRAIN_OK | **SHARE_ALIKE** |
| `timm/oxford-iiit-pet` | vl | `cc-by-sa-4.0` | Oxford-IIIT | TRAIN_OK | **SHARE_ALIKE** |

Two notes that are not verdict changes but should be read before P2.3:

**`deepmind/code_contests` is CC BY 4.0 over material with three different origins.** Its
own card attributes Description2Code material to the MIT licence and CodeNet material to
Apache-2.0, but the Codeforces-sourced problem statements are listed with **no licence
stated**. The catalogue rejected `hendrycks/competition_math` on precisely the reasoning
that *"a live copyright dispute is not cured by a permissive tag on a repackaging"* and that
*"the MIT tag covers the repo's scripts, not the problem text"*. `code_contests` is the same
shape — a permissive umbrella tag over third-party competition problem text — differing in
that two of its three sources do carry acknowledged permissive licences and there is no
known dispute. **This is a consistency question for a human**, not something this audit
should resolve unilaterally.

**`stanfordnlp/snli` adds nothing anyway.** The corpus survey already measured it: *"staged
`snli` is 100% already inside all-nli. It would add ZERO new pairs."* Its licence question is
therefore moot for `compress` — but it is the same CC BY-SA 4.0 that 58.23% of `all-nli`
inherits, so it is not moot for the release.

---

## The BLOCKING list, ranked

Ranked by **cost to replace**, cheapest first. The ranking is not GPU time — the whole fleet
retrains in 43 minutes — it is **corpus-construction effort and capability risk**. A
replacement that changes what the region learns is not a replacement, and where that is the
case it is said so plainly.

### 1. `retrieve` — cost: one email, then either near-zero or total

**Ranked first because its cost is not yet knowable, and the thing that would make it
knowable is trivial.**

**What is blocked:** all of it. 77.8% GooAQ (contested non-commercial), 19.5% Natural
Questions (CC BY-SA 3.0), 2.7% FiQA (CC BY-SA 4.0).

**The decisive unknown:** whether GooAQ's README NOTE or its Apache-2.0 LICENSE governs.

**If the LICENSE governs — cost: near zero.** Drop NQ and FiQA, raise the gooaq cap from
400,000 toward the 3,012,496 rows already on disk. Same objective, same shape, same file, no
fetch, no join. Corpus size restored exactly, and the region becomes `PERMISSIVE_OK`.

**If the README governs — cost: total.** GooAQ joins MS MARCO in the refused pile, and the
region needs a wholly new corpus. Permissive alternatives that yield query→passage pairs at
this scale are thin. The candidates worth checking are `embedding-data/PAQ_pairs` (MIT,
7.29M question↔passage pairs — the largest MIT-declared set found);
`sentence-transformers/specter`, whose upstreams are `allenai/specter` (Apache-2.0) and
`allenai/scidocs` (CC BY 4.0), ~380k pairs but scientific-abstract domain only; and
`allenai/s2orc` / `peS2o`. None is a domain match for general web QA, so this path
changes what the region learns rather than substituting for it.

**Capability risk that applies either way**, and must be measured rather than assumed:
- Dropping FiQA costs `retrieve` its out-of-domain signal. Today's 0.748 is scored on a
  holdout carved from the three-source mix; FiQA's value was being a *different domain*
  (financial QA).
- P0.9b already found the `retrieve` holdout is 53.7% near-duplicates against train, driven
  substantially by **gooaq template collisions**. Concentrating further on gooaq makes that
  worse. The metric would improve while the model got narrower — the most dangerous shape a
  change can have.

**Recommendation:** ask AI2 which term governs, before doing any corpus work here. Keeping
FiQA as **eval-only** is the natural way to preserve the out-of-domain signal in either
branch — see the eval-only section for why that is weaker than it sounds.

### 2. `code` — `Nan-Do/code-search-net-python` — cost: low, one script

**What is blocked:** the whole corpus, on the mirror-tag grounds above. But ~71% of it is
recoverable in place.

**Replacement:** filter by the `repo` column against resolved GitHub licences. The corpus
already carries `repo` (13,581 distinct values); resolving all of them costs 13,581
authenticated GitHub API calls, which fits inside a few hours at the 5,000/hr limit and is a
one-off. Keep only repos whose current licence is MIT / Apache-2.0 / BSD / ISC / 0BSD /
Unlicense / CC0. Expected yield ~324,000 pairs (71.3%), against the 214,813 that first
produced recall@1 0.94.

**Four caveats that must be recorded with the filter, not discovered later:**
- **A repo's licence today is not necessarily its licence in 2019** when CodeSearchNet was
  built. This resolves *current* state and is an approximation of the right question.
- **~2.8% of repos are gone** (404). Those rows cannot be cleared and must be dropped, not
  assumed permissive.
- **`_licenses.pkl` is the authoritative artifact** and it exists — in the original
  github/CodeSearchNet distribution, not in the mirror. Recovering it would beat the API
  approach and should be tried first.
- Dropping ~15% GPL/AGPL and ~5% LGPL/MPL changes the corpus *distribution*, not just its
  size. Re-measure; do not compare against the old recall@1.

### 3. `compress` — `sentence-transformers/all-nli` — cost: moderate to repair, high to make clean

**What is blocked:** 41.77% of the corpus is MultiNLI-derived; three of its five training
genres carry named third-party commercial copyright holders (Penn, Langenscheidt,
Microsoft); and **as distributed the corpus cannot be filtered**, because `genre`,
`promptID` and `pairID` are all dropped.

**Repair, not replacement — and this is the recommended move.** `nyu-mll/multi_nli` still
exposes `genre`. Re-derive the entailment pairs from `nyu-mll/multi_nli` and
`stanfordnlp/snli` directly instead of consuming the pre-flattened all-nli, then keep only
the genres you can defend. Keeping SNLI + government + fiction yields ≈235,000 pairs
against 277,269 today — **a 15% corpus reduction that removes every named commercial
rights holder.** It costs one derivation script and 427 seconds of retraining.

This is a `BLOCKING` → `SHARE_ALIKE` conversion, not a fix. SNLI is 78% of what remains and
is CC BY-SA 4.0 — with the added wrinkle that its CC BY-SA claim is a Stanford assertion no
Flickr30k source corroborates.

**Full replacement, to reach PERMISSIVE_OK**, needs a permissively licensed source of
anchor/positive pairs. The one candidate that is a genuine like-for-like:

| candidate | licence | rows | shape | assessment |
|---|---|---|---|---|
| **`hkust-nlp/SynCSE-scratch-NLI`** | **MIT** (dataset and repo) | 275,579 | `sent0`, `sent1`, `nli_hard` | **The cleanest direct swap found.** Near-identical size to the 277,269 trained today, and the same anchor/positive/hard-negative shape. Sentences are GPT-4/GPT-3.5-generated from genre+topic prompts, so **no NLI corpus text is carried through** |
| `ltg/en-wiki-paraphrased` | Apache-2.0 | 5,145,408 | `original`, `paraphrase` | Largest permissive paraphrase set. But the `original` column *is* Wikipedia text and the card is silent on Wikipedia's own CC BY-SA. Paraphrase ≠ entailment |
| `google-research-datasets/paws` | `other`, grant quoted in card | ~323k positives | `sentence1`, `sentence2`, `label` | *"may be freely used for any purpose."* Positives are word-order/entity edits — hard negatives by construction, but a narrow notion of similarity |
| `sentence-transformers/coco-captions` | ST: none; COCO annotations CC BY 4.0 | 414,010 | `caption1`, `caption2` | Two human captions of one image is a textbook anchor/positive. Use `HuggingFaceM4/COCO` (tagged `cc-by-4.0`) rather than the untagged ST mirror |

**Traps to avoid**, all of which look clean and are not:
`hkust-nlp/SynCSE-**partial**-NLI` (MIT-tagged, but its sentences come from SimCSE's
SNLI+MNLI); `lxyuan/synthetic-nli-triplet` (Apache-2.0 tag over an all-nli derivation);
`SeanLee97/all_nli_angle_format_b` (MIT tag on an all-nli repackaging); `facebook/anli`
(**CC BY-NC 4.0** — non-commercial, and its R3 split re-imports OANC anyway);
`facebook/xnli` and `nyu-mll/glue:mnli` (both are MultiNLI, 392,702 rows, same problem).
The pattern is constant: a permissive tag applied by a repackager over corpus text that
never carried one.

**Honest statement of the gap:** `compress`'s objective is "neighbours stay neighbours in a
short latent", trained on *entailment* pairs specifically. The project already learned the
hard way that substituting a differently-shaped pair set poisons this objective — training
on `pair-class` unfiltered held `compress` at 0.26. A paraphrase corpus is not an entailment
corpus. SynCSE-scratch is the one candidate that preserves the objective rather than
changing it, **and it is synthetic** — which is a different kind of unknown, not an absence
of one. Everything else on that list changes what the region learns, and should be described
that way rather than as a substitution.

### 4. `vl_latent` — `zh-plus/tiny-imagenet` — cost: total. The region is unreleasable as trained.

**What is blocked:** everything. 100,000 of 100,000 pretraining images, with no licence
anywhere in the chain and ImageNet's non-commercial terms as the only terms text present.

**There is no partial fix.** No subset is clean, because the whole thing is one ImageNet
subset.

**Replacement requires a different pretraining corpus and a full re-run** of `vl_latent`
(497 s of GPU — trivial) plus whatever it costs to build the corpus (the real cost). It also
**invalidates the in-domain probe**: the probe currently trains and evaluates on
tiny-imagenet's own train/valid splits, so replacing pretraining data replaces the gate
metric too. The transfer probe on CIFAR-100 survives, and is the only comparable number
across the change — though CIFAR-100 has its own licence vacuum, discussed later.

**And the obvious replacements are all the same dataset wearing different names.** Every
small-image "ImageNet-shaped without the gate" corpus that first comes to mind is
ImageNet-derived, and therefore inherits exactly the problem being escaped:

| candidate | what its own source says | usable? |
|---|---|---|
| **STL-10** | labeled images *"were acquired from labeled examples on ImageNet"*; the Stanford page declares **no licence at all** | **No.** Same provenance, same vacuum |
| **Imagenette / Imagewoof** | *"Imagenette is a subset of 10 easily classified classes from Imagenet"* | **No.** The repo's Apache-2.0 covers the repo; the README makes no claim about the data — the identical code-not-data pattern as CodeSearchNet |
| **CIFAR-10 / CIFAR-100** | labelled subsets of the withdrawn 80M Tiny Images; no licence from anyone | **No** as a training corpus |
| **Tiny ImageNet** | the incumbent | No |

That is the honest shape of this problem: the small-image pretraining ecosystem is largely
ImageNet with the serial numbers filed off, and a licence tag on the packaging is not a
grant in the images. A genuine replacement has to come from a corpus whose *images* are
inside a licence grant, not merely whose annotations are — and most of the large permissive
image sets (Open Images, CC12M, RedCaps, YFCC100M) distribute **URLs, not pixels**, and say
so explicitly. RedCaps states the reason in as many words: *"We do not distribute image
files as we do not legally own them."*

Candidates whose primary source does put the images inside the grant, and which are
therefore worth evaluating:

| candidate | licence | scale | assessment |
|---|---|---|---|
| **`google/docci`** | **CC BY 4.0, stated to cover annotations *and* images**, which Google took and donated | ~15k images | The cleanest chain of title found anywhere in this audit. **Far too small** for 100k-image pretraining on its own |
| **EuroSAT** (`blanchon/EuroSAT_RGB`) | MIT | 27k, 64×64 | Right resolution, clean licence, but **single-domain satellite imagery** — a poor proxy for 200-class object recognition |
| **Caltech-256** | CC BY 4.0 (verified at the CaltechDATA record) | ~30k, 257 classes | Good licence and good class structure; small, and higher-resolution |
| **`zalando-datasets/fashion_mnist`** | MIT | 60k, 28×28 greyscale | Already on this fleet's disk. Clean, but greyscale clothing is not a substitute for natural images |

**None of these is a drop-in.** The honest statement is that no single permissively licensed
corpus of ~100,000 natural photographic images at ~64px was found, and assembling one
(EuroSAT + Caltech-256 + DOCCI + Open Images-by-URL, say) changes both the pretraining
distribution and the probe. **`vl_latent` cannot be repaired by swapping a path in
`VL_REGIONS`; it needs a corpus-construction project.** That is the finding, and it should
be weighed against P3.1's plan to scale the region on tiny-imagenet first.

**This region is also the one the program says matters most.** P12's note argues vision may
be the primary interface rather than a side quest, and P3 plans to scale it. **Every hour
spent scaling `vl_latent` on tiny-imagenet is spent on weights that cannot be released.**
That is the finding this audit exists to surface, and it argues for fixing the corpus
*before* P3.1, not after.

---

## The share-alike question, stated without resolving it

Four corpora in the current or planned mix are CC BY-SA: SNLI (58.23% of `compress`), Natural
Questions (19.5% of `retrieve`), FiQA (2.7% of `retrieve`), and — once P2.3 runs — SQuAD,
HotpotQA and oxford-iiit-pet. Plus ~15% GPL/AGPL source code in `code`.

**The unsettled question is whether trained model weights are a derivative work — in CC's
vocabulary, "Adapted Material" — of the data they were trained on.**

What can be said without deciding it:

- **If weights are not adapted material**, then ShareAlike never triggers, an MIT release is
  unaffected, and only the CC BY *attribution* obligations (which attach to the datasets
  themselves, if redistributed) are in play.
- **If weights are adapted material**, then a cautious reading requires the weights to be
  released under **CC BY-SA 4.0 or a compatible licence, not MIT** — and CC BY-SA 3.0
  (Natural Questions) and CC BY-SA 4.0 are not bidirectionally compatible, which would be a
  second problem. The GPL/AGPL fraction in `code` would raise the same question in a
  stronger form.
- **Nobody in this project is positioned to decide which it is.** It is genuinely open, it is
  being litigated in adjacent forms, and the answer may differ by jurisdiction.

**What a cautious reading would require, if adopted:**

1. Release weights under CC BY-SA 4.0 rather than MIT, and accept that this is a copyleft
   release — downstream fine-tunes would inherit the obligation.
2. Or: remove every share-alike corpus and every GPL/AGPL row before training, and release
   MIT. For `retrieve` that is cheap (item 1 above). For `code` it is the licence filter
   (item 2). For `compress` it costs 42% of the corpus and still leaves the MultiNLI problem.
3. Or: dual-licence — code MIT, weights under a licence that reflects the corpus.

**What is NOT a mitigation:** asserting that weights are not derivative works because it
would be convenient. If that position is taken, take it explicitly, in the model card, as a
stated position rather than a silence.

---

## Is "eval-only" a real distinction?

The task asks whether a dataset used only to *measure* a model, never to train it, escapes
the problem. **Partly — and the strength of the argument depends entirely on a structural
fact that has to be checked per case, not asserted in general.**

**First, what CIFAR-100 actually is, since it is the case in question.**

| field | observed |
|---|---|
| HF tag | `license: unknown` |
| authoritative source | `cave.cs.toronto.edu/kriz/cifar.html` — **zero occurrences** of "licence", "copyright", "terms of use" or "permission". Only: *"If you're going to use this dataset, please cite the tech report at the bottom of this page."* |
| upstream | *"The CIFAR-10 and CIFAR-100 datasets are labeled subsets of the 80 million tiny images dataset."* |
| other redistributors | torchvision, TFDS and OpenML are all silent or record `UNKNOWN LICENSE`. Three Kaggle uploaders assert licences — and assert three different ones |
| **NEW verdict** | **BLOCKING as a training corpus.** Defensible as eval-only here, for the structural reason below |

And the upstream has been formally withdrawn. The 80 Million Tiny Images page (recovered
from the Internet Archive; the live MIT path is now an empty directory index) states:

> *"We therefore have decided to formally withdraw the dataset. It has been taken offline
> and it will not be put back online. **We ask the community to refrain from using it in
> future and also delete any existing copies of the dataset that may have been
> downloaded.**"*

The withdrawal was for offensive content discovered in the automatically-scraped images,
not for a licensing defect. **It says nothing about CIFAR or any derived dataset** — the
word "CIFAR" does not appear in it. So CIFAR is neither exempted nor implicated by its own
upstream's retraction, and no source anywhere states the copyright status of the
underlying scraped images. That is a licence vacuum, not a permission.

**If the eval-only argument is not relied on, CIFAR-100 is replaceable and cheaply so.**
The probe needs a labelled set from a domain different to the pretraining corpus, with
enough classes that top-1 is informative. **Caltech-256** (CC BY 4.0, verified at the
CaltechDATA record; 257 classes, ~30k images) is the closest structural substitute —
comparable class count to CIFAR-100's 100, a real licence, and a different domain from
whatever replaces tiny-imagenet. **EuroSAT** (MIT, 64×64) is the cleanest chain of title but
only 10 classes and single-domain. `timm/oxford-iiit-pet`, already on this fleet's disk, is
CC BY-SA 4.0 — a share-alike probe, which is a smaller problem than an unlicensed one but
not nothing. Swapping the probe costs one config line and re-runs in 497 seconds.

**Where it is strong, and it is strong here for CIFAR-100.** `vl_latent`'s transfer probe was
read rather than assumed. `regions/vl_pretrain.py::_linear_probe` constructs a local
`nn.Linear` head, trains only that head on standardised features, and returns
`{top1, top5, n_eval}` — the head is a local variable and is discarded when the function
returns. It appears nowhere in `_checkpoint_payload`, which saves `model.state_dict()` for
the I-JEPA model only. The module comment states the intent and the code honours it:

> *"The probe is trained on features from the EMA target encoder with gradients stopped at
> the encoder."*

So **no CIFAR-100-derived parameter exists in any published artifact.** That is not an
argument that eval-only is generally safe; it is a verified statement that in this specific
implementation, CIFAR-100 touches nothing that would be redistributed. That is about as good
as this argument gets.

**Where it is weak, and the task is right to press on it.**

1. **Model selection leaks.** If a corpus's metric is used to choose a checkpoint, a
   hyperparameter, or an early-stopping point, information from it *has* influenced the
   released weights, even though no gradient flowed. CSD gates on `beats_untrained`, and
   `vl_latent`'s gate list includes `transfer_top1` — so **CIFAR-100 is already part of a
   gate decision**, not purely an observation. The leak is small and indirect, but it is not
   zero, and "we only evaluated on it" is doing more work in that sentence than it can bear.
2. **Most restrictive terms do not distinguish.** ImageNet's clause is *"use the Database
   only for non-commercial research and educational purposes"* — "use", not "train on". A
   term written that way does not obviously grant evaluation while withholding training;
   it restricts both.
3. **It is a norm, not a rule.** Reporting benchmark numbers on restrictively licensed
   evaluation sets while releasing weights permissively is near-universal practice. Practice
   is evidence about risk appetite. It is not evidence about obligation.

**Plain answer, since the task asks for one:** the eval-only distinction is sound enough to
rely on for **CIFAR-100 specifically in this codebase**, because it was verified structurally
that no CIFAR-100-derived parameter is published — with the caveat that it sits in a gate and
should be described in the model card as an evaluation set rather than quietly omitted. It is
**not** sound enough to rely on as a general principle, and it would be wrong to use it to
rescue tiny-imagenet: tiny-imagenet is pretraining data, every published `vl_latent`
parameter is derived from it, and no framing changes that.

---

## Required attributions, verbatim

Ready to paste into a model card. These are the notices that would be required **if** the
attribution obligations attach — which for CC BY is the licence's own condition on
distributing adapted material, and is the part of this document least dependent on the
unsettled derivative-work question. Including them costs nothing and omitting them is
checkable by anyone.

> *These strings are drafted from the licence names and dataset identities verified in this
> audit. Before publication, confirm each dataset's own preferred attribution/citation
> string from its card — several request a specific paper citation in addition to the
> licence notice.*

```markdown
## Training data attribution

This model was trained on the following corpora. Attribution is provided as required by
their licences.

### CC BY 4.0
- **banking77** — PolyAI. Source: https://github.com/PolyAI-LDN/task-specific-datasets
  Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
- **CodeContests** — DeepMind. Source: https://huggingface.co/datasets/deepmind/code_contests
  Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
  Contains material from Description2Code (MIT) and Project CodeNet (Apache-2.0);
  problem statements sourced from Codeforces carry no licence stated by the distributor.

### CC BY-SA (share-alike; see the licensing note below)
- **SNLI** — Bowman et al., Stanford NLP. https://nlp.stanford.edu/projects/snli/
  Licensed under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
- **Natural Questions** — Google Research.
  https://ai.google.com/research/NaturalQuestions
  Licensed under CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/).
  Passage text is derived from Wikipedia, which is licensed CC BY-SA; see
  https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content
- **FiQA (via BeIR)** — https://huggingface.co/datasets/BeIR/fiqa
  Licensed under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
- **SQuAD** — Rajpurkar et al. https://rajpurkar.github.io/SQuAD-explorer/
  Licensed under CC BY-SA 4.0. Passage text derived from Wikipedia.
- **HotpotQA** — Yang et al. https://hotpotqa.github.io/
  Licensed under CC BY-SA 4.0.
- **Oxford-IIIT Pet** — Parkhi et al., https://www.robots.ox.ac.uk/~vgg/data/pets/
  Licensed under CC BY-SA 4.0.

### Apache-2.0 / MIT (no notice required beyond the licence text)
- GSM8K — OpenAI (MIT) · AQuA-RAT — DeepMind (Apache-2.0)
- GoEmotions — Google Research (Apache-2.0) · Fashion-MNIST — Zalando (MIT)
- APPS — https://huggingface.co/datasets/codeparrot/apps (MIT)

### Open American National Corpus (only if `compress` is re-derived with genres kept)
The OANC End User Licence requires acknowledgement of the original author and publisher
per its Appendix I. Include only the entries whose genres you actually retain:
- Switchboard Data — Copyright (c) 1997-2002 Trustees of the University of Pennsylvania
- Berlitz Travel Guides — Copyright (c) 2003 Langenscheidt Publishers
- Slate Magazine — Copyright (c) 1996-2000 Microsoft Corporation
- Oxford University Press book selections — Copyright (c) 1999, 2001, 2003 Oxford University Press
- Verbatim: The Language Quarterly — Copyright (c) 2003 Word, Inc.
- ICIC Corpus of Fundraising Texts — Copyright (c) 2003 Indiana Center for Intercultural Communication
The Agreement also requires its own text to be displayed in human-readable form alongside
any transparently modified version, and requires an aggregate work to carry "a notice
specifying the inclusion of the O-ANC Processed Material and any and all appropriate
copyright notices." Source: https://www.anc.org/OANC/license.txt

### Contested — DO NOT publish an attribution line for this until resolved
- **GooAQ** — Allen Institute for AI, https://github.com/allenai/gooaq
  The repository's LICENSE file is Apache-2.0. The repository's README states
  "This dataset should not be used for any commercial purposes." Writing either one alone
  as the attribution would misrepresent the source. Resolve with AI2 first.

### Evaluation-only
- **CIFAR-100** — Krizhevsky, 2009. https://www.cs.toronto.edu/~kriz/cifar.html
  Used only to train and score a linear probe on frozen features for transfer measurement.
  No CIFAR-100-derived parameter is included in the released weights. The dataset page
  states no licence; the authors request citation of
  *Learning Multiple Layers of Features from Tiny Images*, Alex Krizhevsky, 2009.

### Licensing note
Several training corpora are licensed CC BY-SA. Whether trained model weights constitute
"Adapted Material" under those licences is an unsettled legal question on which this
project takes no position. These weights are released under <LICENCE>; if you require
certainty about share-alike obligations, consult the upstream licences directly.
```

**Attributions that CANNOT be written**, because there is nothing to attribute *to*:

- **tiny-imagenet** — no licence exists to comply with. An attribution line would imply a
  grant that was never made.
- **all-nli / MultiNLI** — the mirror declares no licence and the upstream declares `other`
  among four. There is no single notice that would be accurate. If the region is re-derived
  per the repair path, the OANC EULA's own attribution condition applies and names the
  rights holders per genre — that notice is writable, but only once the genres are known,
  which is precisely what the current corpus destroyed.
- **`Nan-Do/code-search-net-python`** — writing "Apache-2.0" would repeat the mirror's own
  unsupported claim. The accurate notice is per-source-repository, which is exactly the
  metadata the mirror dropped.
- **flickr30k** — the distributor states it does not own the images' copyright. There is no
  licensor to name.

That three of the four in-use corpora cannot be honestly attributed is, by itself, the
audit's answer.

**One caveat on the SNLI line above.** It is written as CC BY-SA 4.0 because that is what
Stanford states. As noted under `compress`, no Flickr30k source corroborates it, and SNLI
also contains ~4,000 CC BY 4.0 Visual Genome premises that its own documentation does not
acknowledge. The notice is the best available; it is not independently verified.

---

## Recommendation on the release licence

*Recorded as a recommendation to a human, not a decision.*

**Do not release the current weights under MIT.** Not because MIT is wrong for the project,
but because **all four** trained regions rest on corpora that do not support any
redistribution grant, and MIT asserts one.

Three defensible paths, in order of preference:

**A. Fix the corpora, then release MIT — the option the project should want.**
Ordered by what is actually within reach: `code` is one filtering script against a column
already on disk; `retrieve` is a config change *if* AI2 confirms the Apache reading, and a
new corpus if not; `compress` can be re-derived from `nyu-mll/multi_nli` to reach
share-alike but needs a synthetic or permissive substitute to get further; `vl_latent` needs
a wholly new pretraining set. GPU cost across all four is 43 minutes — the effort is corpus
construction, not training. This is the only path that makes the MIT claim true rather than
asserted.

**B. Release the code MIT and hold the weights back** until A is done. Costs nothing, claims
nothing false, and keeps the repo's move off its current proprietary LICENSE unblocked. This
is the right move *today* if a release is wanted before the corpus work lands.

**C. Release a partial set of weights.** `code` post-filter is the closest any region gets
to clean, and it is the only one whose fix is fully within this project's control — the
`repo` column is on disk and the filter is one script. (`retrieve` would have been the
easiest of all, and may still be, but that now depends on an answer from AI2 rather than on
any work here.) Releasing one region honestly beats releasing four weights with a licence
that does not hold. It also matches the architecture — regions are independent by
construction, so a per-region licence statement is natural rather than awkward.

**What would make an MIT release defensible without any corpus work:** taking the position
that trained weights are not derivative works of training data, stating it explicitly in the
model card, and accepting the risk. That is a real position that much of the field operates
on, and it would dispose of every SHARE_ALIKE verdict here in one move.

**It does not dispose of the two BLOCKING ones**, and this is the distinction that matters
most in the whole document. Share-alike is a question about what *obligations* propagate.
tiny-imagenet and (on the README reading) GooAQ are questions about whether a *permission
ever existed*. No theory about the derivative-work status of weights repairs a missing grant
in the underlying images or a source's own statement that its data is not for commercial
use. A model card can take a position on the first. It cannot take a position on the second.

**A note on the repo's own licence.** `LICENSE` currently reads *"Proprietary License … all
copyrights are reserved"*. Whatever is decided about the weights, the code relicensing is a
separate and much simpler change, and `LICENSES/LICENSE_TRACKER.md` already tracks
dependency compatibility against MIT.

---

## What I could not determine

Listed rather than guessed. Each is a real gap.

0. **Which of GooAQ's two contradictory terms governs** — the Apache-2.0 LICENSE file or the
   README's *"should not be used for any commercial purposes."* Listed first because it is
   the highest-value unknown in the audit and the cheapest to close: it decides 77.8% of one
   region's corpus, and it is answerable by an email to AI2 rather than by more research.
   **Do not resolve it by picking the convenient reading.**
1. **Whether trained weights are derivative works of training data.** Unsettled, and central
   to every SHARE_ALIKE verdict here. Needs a human decision, and arguably a lawyer.
2. **Whether the GPL/AGPL fraction of `code` matters more than the CC BY-SA fraction.**
   Same unsettled question, stronger copyleft, ~15% of the corpus. Not analysed further here.
3. **Where `Nan-Do/code-search-net-python`'s `apache-2.0` tag came from.** The card offers no
   justification and there is no LICENSE file. It may be an error; it may reflect something
   not visible from the repo. Unresolved.
4. **Tiny ImageNet's own terms, as opposed to ImageNet's.** No licence statement was found at
   any point in its chain. Absence of a found statement is not proof of absence, but three
   independent points in the chain (both HF mirrors and the `dataset_infos.json`) all record
   an empty licence, which is consistent.
5. **Whether the historical `_licenses.pkl` from github/CodeSearchNet is still retrievable**,
   and whether its per-function licences match repositories' current GitHub state. Not
   attempted — it would settle item 3 and improve item 2's filter.
6. **Repository licences as of 2019.** The 71.3% permissive figure is measured against
   *current* GitHub state. Licences change; repos are deleted (2.8% already are).
7. **Whether the Codeforces-sourced portion of `code_contests` is covered by its CC BY 4.0
   umbrella.** The card acknowledges the source without stating its licence.
8. **Whether Apache-2.0 covers `sentence-transformers` *data*.** The library's LICENSE is
   Apache-2.0 but contains no data language, and no document in the ecosystem asserts it
   either way. Zero of 96 `sentence-transformers` datasets carry a licence tag. The nearest
   statement is on the older `embedding-training-data` repo and disclaims the question:
   *"we do not vouch for their quality or fairness, or claim that you have license to use
   the dataset. It remains the user's responsibility to determine whether you as a user
   have permission…"*
9. **Any upstream source stating that Flickr30k captions are CC BY-SA.** Only Stanford
   asserts it, in SNLI's documentation. Illinois never confirms it. This matters because it
   is the sole basis for SNLI's CC BY-SA 4.0 grant, which in turn covers 58% of `compress`.
10. **The ANCC ↔ text-provider agreements** referenced by the OANC EULA — the instruments
    that would actually define what Penn, Microsoft, Langenscheidt and OUP granted. Cited in
    the licence, unpublished, not locatable.
11. **Licence and rights holder for MultiNLI's `government`, `9/11` and `face-to-face`
    genres.** Silent in the paper, absent from OANC Appendix I. `government` is 8.20% of the
    `compress` training corpus and its public-domain status rests on the MultiNLI paper's
    assertion alone.
12. **Why `nyu-mll/multi_nli` declares `mit`** among its four licence tags. The card body
    never mentions MIT.
13. **The copyright status of 80 Million Tiny Images' underlying scraped images**, and hence
    of CIFAR-100's. No source addresses it — not the CIFAR page, not the TPAMI paper, not
    the withdrawal letter.

---

## Appendix: how these numbers were produced

Everything measured, nothing copied from a doc. No dataset was downloaded; all corpus
measurements read shards already on disk.

| finding | method |
|---|---|
| licence tags | live HF API `/api/datasets/{id}`, `cardData.license` + tag list, 2026-09-02 |
| absence of LICENSE files | live HF API `/api/datasets/{id}/tree/main` |
| ImageNet terms | fetched `https://www.image-net.org/download.php` |
| CIFAR-100 terms | fetched `https://cave.cs.toronto.edu/kriz/cifar.html` (301 from `cs.toronto.edu/~kriz/`) |
| CodeSearchNet licence scope | fetched `github/CodeSearchNet` README |
| GooAQ's contradiction | `curl` of the raw bytes of `allenai/gooaq`'s `README.md` and `LICENSE` on `main`; GitHub API for repo metadata and README commit history (last touched 2021-07-23) |
| GooAQ's HF card silence | `huggingface.co/datasets/allenai/gooaq/raw/main/README.md` — tag `apache-2.0`, no non-commercial note anywhere |
| all-nli SNLI/MultiNLI split | set-match of `all-nli/pair/train` against `snli/train` label==0, both on disk |
| restricted-genre presence | regex over the 314,315 anchors in the shard `compress` trained on |
| absence of a `genre` column | `pyarrow` schema of the trained shard |
| MultiNLI still has `genre` | HF datasets-server `/info` for `nyu-mll/multi_nli` — columns include `genre`; train split 392,702 |
| which genres are in MNLI *train* | datasets-server `/first-rows` + `/filter` — train carries only fiction, government, slate, telephone, travel; `nineeleven` returns 0 rows. Independently confirms that OUP and the other mismatched-only genres are absent from the trained corpus |
| OANC per-source rights holders | `https://www.anc.org/OANC/license.txt`, Appendix I |
| 80M Tiny Images withdrawal | Internet Archive capture `20200704231805`; the live MIT path is now an empty directory index |
| Flickr30k terms | `shannon.cs.illinois.edu/DenotationGraph/`, stated identically on four pages |
| code repo licences | `gh api repos/{owner}/{name}` over 150 head + 150 random repos |
| training costs | `/akula-data/csd/receipts/*.json`, `elapsed_s` |
| probe head not published | read `regions/vl_pretrain.py::_linear_probe` and `_checkpoint_payload` |


---

## Replacement vision corpora

**Status:** this section only. It does not touch anything above. `scripts/csd-corpus-expand.py`
is untouched, no dataset was downloaded, and this ran no GPU work — a VL training run is
live on gpu5080 and the 3090 Ti has other work.

**Method, same discipline as the rest of the audit:** for every candidate below, the HF
mirror's licence tag was checked, then the TRUE upstream (paper, official project page, or
GitHub `LICENSE` file) was fetched independently and quoted. A tag is not evidence about its
upstream, and this section found **two more confirmed mismatches** doing exactly that check
(`ylecun/mnist`, `vincent-espitalier/K-MNIST-CSV`, both below) — on top of the eight already
on record for this project, and a third oddity (`bazyl/GTSRB`, tagged `gpl-3.0` for no
traceable reason).

**Constraint relaxed for this section, by explicit operator instruction:** a composite of
several narrow, cleanly-licensed corpora is fully acceptable in place of one broad one.
"I'm entirely okay with changing to different datasets and using more datasets to get
accomplished what could be done with fewer datasets with different licenses. Licence
compliance outweighs corpus-count convenience." Everything below is organised around that.

### Candidate table

Verdict vocabulary is the same four categories defined earlier in this document, with
`BLOCKING (unresolved)` used where the chain isn't contradictory the way tiny-imagenet's is
— it's just that no permissive grant was ever located, or the grant found does not clearly
cover the relevant use.

#### Clean at both mirror and upstream

| dataset | licence: mirror tag → upstream verified | images | native resolution | domain | verdict |
|---|---|---|---|---|---|
| `zalando-datasets/fashion_mnist` | `mit` → **MIT**, `github.com/zalandoresearch/fashion-mnist/LICENSE`, verbatim MIT text, "Copyright © 2017 Zalando SE" | 70,000 (60k/10k) | 28×28 grayscale | garment product photos, 10 classes | **PERMISSIVE_OK** |
| `timm/eurosat-rgb` | `mit` → **MIT**, `github.com/phelber/EuroSAT` README: "The dataset is licensed under the MIT license." Underlying Sentinel-2 imagery is separately covered by the EU's own **open** data policy — see note below | 27,000 (16.2k/5.4k/5.4k) | **native 64×64×3 RGB** | Sentinel-2 satellite land-use, 10 classes | **PERMISSIVE_OK** |
| `1aurent/PatchCamelyon` | `cc0-1.0` → **CC0**, `github.com/basveeling/pcam` README: "The data is provided under the CC0 License, following the license of Camelyon16." (Camelyon16's own site is a JS-rendered SPA that could not be independently re-fetched in this sandbox — see caveats) | 327,680 (262,144/32,768/32,768, canonical PCam split) | 96×96 RGB → downsample to 64 | histopathology, binary tumour/no-tumour | **PERMISSIVE_OK** |
| Shapes3D (`google-deepmind/3d-shapes`; pixel mirror `eurecom-ds/shapes3d`, tag-matching mirror `galilai-group/shapes3d`) | `apache-2.0` (on the tagged mirror) → **Apache License 2.0**, full text fetched verbatim from `raw.githubusercontent.com/google-deepmind/3d-shapes/master/LICENSE` | 480,000 | **native 64×64×3 RGB, exact match** | synthetic 3D-rendered single-object scenes (shape/hue/scale/orientation), no fixed class labels | **PERMISSIVE_OK** |
| dSprites (`google-deepmind/dsprites-dataset`; pixel mirror `eurecom-ds/dsprites`) | none on the pixel-bearing mirror → **Apache License 2.0**, full text fetched verbatim from `raw.githubusercontent.com/google-deepmind/dsprites-dataset/master/LICENSE` | 737,280 | **native 64×64, exact match** | synthetic flat white silhouettes on black, 1 channel | **PERMISSIVE_OK**, but see the visual-complexity caveat below |
| `nyuuzyou/pxhere` | `cc0-1.0` → **CC0**, `pxhere.com/en/terms` upload clause: "By uploading, You release Images under Creative Commons CC0 into the public domain… You grant anyone the right to use this work for any purpose, without any conditions" — a platform-enforced condition of every upload, not a metadata tag over content that was never uniformly licensed | ≈1.1M | full camera resolution (e.g. 3264×2448) → downsample to 64 | general stock photography (nature, people, urban, objects, animals, landscapes); free-text tags, no fixed classes | **PERMISSIVE_OK** — the closest thing found to natural-photo breadth |
| `biglam/british-library-book-images` | `cc0-1.0` → **Public Domain Mark**, British Library / Flickr Commons "1 Million Images from Scanned Books" deposit, "no known copyright restrictions"; corroborated circumstantially (works c.1510–1900, independently public domain by age; well-documented named GLAM maintainer) rather than by a fresh primary-source fetch — `bl.uk`/Flickr Commons pages were unreachable from this sandbox | 1,080,814 (across `embellishments`/`plates`/`medium`/`covers` configs) | full scan resolution (e.g. 2565×1539) → downsample to 64 | book illustrations, plates, covers, engravings — **not photography** | **PERMISSIVE_OK**, narrow stylistic domain |
| `AI-Lab-Makerere/beans` (`ibean`) | `mit` → **MIT**, `github.com/AI-Lab-Makerere/ibean` states "License: MIT" | 1,295 (1034/133/128) | photographic, variable → resize to 64 | bean-leaf disease photos, 3 classes | **PERMISSIVE_OK**, too small to matter at scale |
| `google/quickdraw` (pixel mirror `Xenova/quickdraw`; presplit `Xenova/quickdraw-small`, untagged but same lineage) | `cc-by-4.0` → **CC BY 4.0**, `github.com/googlecreativelab/quickdraw-dataset`: "This data made available by Google, Inc. under the Creative Commons Attribution 4.0 International license." | 50,426,266 (or 4.5M/250k/250k presplit) | **native 28×28 grayscale bitmap** → upscale to 64 | free-hand sketches, **345 categories** | **ATTRIBUTION** |
| CLEVR (`laion/clevr-webdataset`, ships `LICENSE.txt`+`COPYRIGHT.txt` in-repo; also `dpdl-benchmark/clevr`) | none on HF tag → **CC BY 4.0** for the image data, confirmed live at `cs.stanford.edu/people/jcjohns/clevr/` ("The dataset is released under the Creative Commons CC BY 4.0 license.") and in-repo `COPYRIGHT.txt` ("CLEVR (c) 2017, Facebook, Inc."). The **generation code** (`facebookresearch/clevr-dataset-gen`) is separately **BSD** — two licences, don't read only the code repo | 100,000 (70k/15k/15k) | 480×320 RGB → downsample to 64 | synthetic 3D-rendered multi-object scenes, shading + soft shadows + metal reflections | **ATTRIBUTION** |
| Caltech-101 (`HuggingFaceM4/Caltech-101` tag; pixel-bearing but untagged `clip-benchmark/wds_vtab-caltech101`) | `cc-by-4.0` → **CC BY 4.0**, CaltechDATA institutional repository `data.caltech.edu/records/mzrjq-6wc02`: "The Creative Commons Attribution license allows re-distribution and re-use… on the condition that the creator is appropriately credited." (Caltech's 2022 re-deposit of record; the original 2003/2004 release carried no formal licence — the institutional CC-BY-4.0 deposit is the operative one now) | ≈9,146 (101 categories + background/clutter) | variable, ≈300×200 avg → resize to 64 | **natural everyday-object photography**, real classes | **ATTRIBUTION** — see mirror caveat below |
| Caltech-256 | no matching pixel-bearing HF mirror found → **CC BY 4.0**, same CaltechDATA pattern, `data.caltech.edu/records/nyy15-4j048` | ≈30,607 (257 categories) | variable → resize to 64 | **natural everyday-object photography**, real classes | **ATTRIBUTION** — use the official CaltechDATA archive; no HF mirror carries a matching tag |

#### Share-alike (usable, but not "clean"; excluded from the composite below)

| dataset | licence: mirror tag → upstream verified | images | native resolution | domain | verdict |
|---|---|---|---|---|---|
| `timm/oxford-iiit-pet` | `cc-by-sa-4.0` → **CC BY-SA 4.0**, confirmed live at `robots.ox.ac.uk/~vgg/data/pets/`: "available… under a Creative Commons Attribution-ShareAlike 4.0 International License." Matches — no mismatch here, just share-alike | 7,349 (3680/3669) | variable, resize to 64 | pet photos, 37 classes | **SHARE_ALIKE**, and too small regardless |
| `ylecun/mnist` | `mit` (**wrong**) → **CC BY-SA 3.0**. `yann.lecun.com/exdb/mnist/` is currently unreachable (empty directory, both direct and via a text-proxy fetch); corroborated by two independent secondary quotes of LeCun & Cortes' own stated terms — Keras's docs and Keras's own source code, both verbatim: "MNIST dataset is made available under the terms of the [Creative Commons Attribution-Share Alike 3.0 license]" | 70,000 (60k/10k) | 28×28 grayscale | handwritten digits, 10 classes | **SHARE_ALIKE** — mismatch, ninth-and-tenth case for this project's mirror-vs-upstream tally |
| `vincent-espitalier/K-MNIST-CSV` | `cc-by-4.0` (**wrong**) → **CC BY-SA 4.0**, `github.com/rois-codh/kmnist` README, verbatim: "Both the dataset itself and the contents of this repository are licensed under a permissive CC BY-SA 4.0 license" | 70,000 (60k/10k) | 28×28 grayscale | Kuzushiji character images, 10 classes | **SHARE_ALIKE** — mismatch |

#### Unresolved — not recommended, not cleanly BLOCKING either

| dataset | what's actually known | verdict |
|---|---|---|
| EMNIST (`Royc30ne/emnist-{balanced,digits,byclass}` tag `mit`; `giulioappetito/emnist-letters` tag `gpl`) | Both tags are self-declared and unsupported by anything found. True upstream is Cohen et al. (arXiv:1702.05373), built on **NIST Special Database 19**. NIST's own copyright policy distinguishes ordinary (uncopyrightable) government works from "Standard Reference Data" compilations, over which the Secretary of Commerce *does* assert copyright — and SD19 is catalogued under NIST's SRD namespace. Whether SD19 falls on the public-domain or the copyrighted-SRD side of that line was **not resolved** by this search. The uploaders' own READMEs hedge ("co-located for non-profit usage… if there are any conflicts of copyright, please contact me to delete it") rather than assert a grant. `giulioappetito/emnist-letters` additionally has no label column and a row count far short of standard EMNIST-Letters — a data-integrity problem independent of licensing. | **BLOCKING (unresolved)** |
| GTSRB (`bazyl/GTSRB` tag `gpl-3.0`, others untagged) | The `gpl-3.0` tag traces to the uploader's personal mirror, not the institute — a confirmed false tag, not a real mismatch to carry forward. True upstream, `benchmark.ini.rub.de`: "The data is free to use. However, we cordially ask you to cite the following publication if you do." That is not a named permissive licence — no MIT/BSD/CC0/CC-BY grant, no explicit redistribution right. | **BLOCKING (unresolved)** |
| `poloclub/diffusiondb` | Licence text itself is maximally clean and matches at both levels: **CC0-1.0** for the data, **MIT** for the code, both confirmed at `github.com/poloclub/diffusiondb`. But the images are Stable Diffusion outputs, and SD was trained on LAION-derived scraped web imagery of contested and unresolved copyright status. DiffusionDB's own CC0 grant is real but can only cover what its authors actually hold rights to — it cannot certify the underlying generative model's training legality, which is a live, unresolved question well outside this document's scope. Real pixels are present (2M–14M images, variable resolution ~512×512–768), diversity is very high, and no per-photographer trap applies (this is a different shape of problem entirely). | **UNCLEAR** — recommend excluding from a "definitely clean" composite for now; flag alongside the CC BY-SA derivative-work question elsewhere in this document as a second unresolved-provenance class, not a licence-text problem |

#### Ruled out — for the record

| dataset | true upstream finding | verdict |
|---|---|---|
| `imageomics/TreeOfLife-200M` | Its **own** "Licensing Information" section states the compilation-level CC0 tag sits over a knowing mix of CC0/CC-BY/CC-BY-NC/CC-BY-NC-SA/CC-BY-NC-ND content from GBIF/EOL/BIOSCAN/FathomNet, filtered to none of them. This is the exact mixed-per-contributor trap that already ruled out COCO and red_caps, and the dataset's own documentation admits it rather than requiring inference. Independently disqualifying: it ships **metadata only** (`catalog.parquet` has no image column); actual pixels require external re-download from four different providers' own infrastructure. | **BLOCKING**, two independent reasons |
| STL-10 | `cs.stanford.edu/~acoates/stl10/`: "Images were acquired from labeled examples on ImageNet." Same root cause as tiny-imagenet. No licence text anywhere on the page. | **BLOCKING** |
| SVHN | Archived `ufldl.stanford.edu/housenumbers/`: "(Note: for non-commercial use only)." HF's own community card repeats it. One mirror (`Genius-Society/svhn`) wrongly tags `mit`. | **BLOCKING** |
| Food-101 | `data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/` states no licence at all; `source_datasets: extended|other-foodspotting` — images are individually-submitted foodspotting.com user photos, the same per-photographer pattern that already ruled out COCO/red_caps. | **BLOCKING** |
| Places365 | Archived `places2.csail.mit.edu`: "for academic research and education purposes." Two obscure mirrors wrongly tag `mit`. | **BLOCKING** |
| CIFAR-10 | Identical provenance to CIFAR-100, already BLOCKING above: "a labeled subset of the 80 million tiny images dataset," no licence text at the source, underlying corpus formally withdrawn. | **BLOCKING** |
| DTD | `robots.ox.ac.uk/~vgg/data/dtd/`: "This data is made available to the computer vision community for research purposes." | **BLOCKING** |
| Open Images | Google's own facts-and-figures page: "The images are listed as having a CC BY 2.0 license… we make no representations or warranties regarding the license status of each image and you should verify the license for each image yourself." Google explicitly declines to assert a uniform grant — same mixed-per-photographer shape as COCO/red_caps, just stated more candidly. | **BLOCKING** |

### A note on EuroSAT's Sentinel provenance, since it's the same kind of chain that broke GooAQ and tiny-imagenet — and this time it resolves clean

The generic ESA website terms (`sentinels.copernicus.eu/web/sentinel/terms-conditions`) say the
site's own content is for "non-commercial use" and forbids redistribution — which, read
carelessly, looks like the GooAQ contradiction again. But that page governs the **website**,
not the **data**. The actual data-governing instrument is a separate EU legal instrument,
fetched directly as a PDF and converted with `pdftotext`:

> *"EU law grants free access to Copernicus Sentinel Data and Service Information for the
> purpose of the following use in so far as it is lawful: (a) reproduction; (b) distribution;
> (c) communication to the public; (d) adaptation, modification and combination with other
> data and information…"*
> — **"Legal notice on the use of Copernicus Sentinel Data and Service Information,"**
> European Commission, per Commission Delegated Regulation (EU) No 1159/2013,
> `sentinels.copernicus.eu/documents/247904/690755/Sentinel_Data_Legal_Notice`

with one attribution condition attached: distributing Sentinel data requires the notice
`"Copernicus Sentinel data [Year]"`, or `"Contains modified Copernicus Sentinel data [Year]"`
if adapted. EuroSAT's own MIT tag covers phelber's compiled/labelled 64×64 patches; this
underlying-imagery notice is an easy addition to the same attribution block already drafted
elsewhere in this document, not a conflict with it.

### Recommended pretrain + probe pairing

**Recommended composite pretraining mix** — six domains, none of them individually broad
enough to matter alone, deliberately **capped rather than used at full size**, for reasons
covered in the balance discussion below:

| domain | source | recommended cap | of how many available | native res → target |
|---|---|---|---|---|
| general photography | `nyuuzyou/pxhere` | 100,000 | ≈1.1M | full-res → 64 |
| histopathology | `1aurent/PatchCamelyon` | 100,000 | 327,680 | 96 → 64 |
| synthetic 3D render | Shapes3D | 100,000 | 480,000 | 64 (exact) |
| synthetic 3D multi-object scene | CLEVR | 100,000 | 100,000 (all of it) | 480×320 → 64 |
| garment product photo | Fashion-MNIST | 70,000 (all of it) | 70,000 | 28 → 64 |
| satellite land-use | `timm/eurosat-rgb` | 27,000 (all of it) | 27,000 | 64 (exact) |
| **total** | | **≈497,000** | | six visually distinct, licence-clean domains |

A stricter-balance variant — cap everything to EuroSAT's own ceiling (~27,000 per domain,
≈162,000 total) — is closer to tiny-imagenet's original 100,000-image scale and worth running
as a comparison point; the table above is the "use more of what's available and cap only the
largest sources" reading of the composite instruction. Both are cheap to try; see the
experiment design below for why trying both is the honest move rather than picking one.

**Recommended probe pairing — kept structurally separate from pretraining, on purpose:**

- **Primary transfer probe: Quick Draw** (CC BY 4.0), held out entirely from pretraining.
  345 classes, huge N, a different visual domain (line drawings, not photographs) from
  everything in the pretrain mix — this plays tiny-imagenet's-transfer-probe role
  (cifar100's old job) with a clean licence and far more classes.
- **Secondary transfer probe: Caltech-256** (CC BY 4.0 via CaltechDATA), also held out
  entirely from pretraining, specifically so it can occupy the role tiny-imagenet's own
  domain used to occupy — natural everyday-object photography with real class structure.
  This is the closest surviving thing to what CIFAR-100 was actually testing, and unlike
  CIFAR-100 its licence is clean. The "eval-only, no parameter published" argument this
  document already verified for CIFAR-100 by reading `_linear_probe` and
  `_checkpoint_payload` applies unchanged here — the mechanism is dataset-agnostic, only the
  corpus name changes.
- **Tertiary in-domain probe: EuroSAT's own held-out test split** (10 classes) — the same
  structural role tiny-imagenet's own valid split played originally: a sanity check that
  pretraining produced *any* separable features, independent of transfer.

**Reasoning:** I-JEPA pretraining doesn't need labels, so the pretrain-side choices were
driven by scale, licence cleanliness, and domain diversity. The probe-side choices were
driven by the opposite: fixed classes, a genuine held-out split, and — per this document's
own stated preference — a *different* dataset than pretraining, because that's what actually
measures transfer rather than memorisation. Keeping Quick Draw and Caltech-256 out of
`train_shards` entirely preserves that distinction instead of quietly eroding it.

### Is mixing these domains in one I-JEPA run sound, or does it fragment the encoder?

This was checked against the actual code, not against I-JEPA in the abstract, because the
answer depends on specifics this codebase already has.

**What the loss actually asks for.** I-JEPA's context→target prediction is entirely
intra-image — `vl_pretrain.py`'s own docstring states the target is "features from the EMA
target encoder," predicted from a context block of the *same* image. There is no point in
the loss where one image's content is asked to predict another's. So the sharpest version of
the concern — "masked tissue predicted from surrounding farmland" — does not occur at the
level of an individual loss term; no example is ever scored against another domain's content.

**Where the real risk actually lives, and it's worse here than it would be for a large
model.** Two things found directly in `vl_pretrain.py` matter:

1. `_to_float` hardcodes ImageNet's mean/std (`[0.485, 0.456, 0.406]` / `[0.229, 0.224,
   0.225]`) and applies it to every image regardless of source. Histopathology stain colour,
   satellite-derived RGB composites, and line-drawing whites-and-blacks do not share
   ImageNet's colour statistics; none of them get domain-appropriate normalisation today.
2. Batch sampling is `torch.randint(0, x_tr.size(0), (cfg.batch_size,), generator=gen)` over
   the single tensor `_decode_split` produces by **concatenating every shard with no domain
   weighting at all.** Per-domain exposure in every batch is exactly proportional to that
   domain's raw row count among the shards handed to the run.

Combined with a genuinely small shared encoder (22.9M parameters in the current run, `dim=384,
depth=6`), the mechanism that should worry this project is **gradient interference / shared-
capacity competition across domains with different low-level statistics** — the same
phenomenon documented broadly in multi-task and continual learning (catastrophic
interference, gradient conflict between dissimilar tasks) — rather than literal
contradictory labels. This is a real, well-documented *class* of effect; whether it bites
*at this scale, for this specific domain mix* is not something I can confirm from the
literature, because I found no study of JEPA-style pretraining across this particular kind
of domain heterogeneity (histopathology + satellite + garments + synthetic renders + natural
photos) at a ~20M-parameter scale. Anyone claiming otherwise would be guessing; I'd rather
say so than manufacture a citation.

**On staged (per-domain, then combined) versus naive union, specifically:** the coordinator's
hypothesis — that heterogeneity reads as noise before the encoder can tell domains apart, and
as information afterward — is mechanistically plausible but rests on a premise this model may
not be able to afford: telling domains apart at all costs representational capacity, and a
6-layer, 384-dim encoder has much less of it to spend on that than a foundation-scale model
would. If the encoder never budgets capacity for domain identity, the proposed benefit of
staging (heterogeneity becoming information once domains are distinguishable) has nowhere to
land. I found nothing in the literature that settles this either way for a model this small,
and nothing JEPA-specific in either direction. **I'm not confident enough to recommend staging
over union, or union over staging — this should be measured, not guessed**, and the fleet
already has the instrument to measure it cheaply:

**Proposed experiment** (not run — no GPU work was in scope here):

| run | what it isolates | how |
|---|---|---|
| A. naive union | baseline | pool all six domains' shards as `train_shards` at the capped sizes above, uncapped-relative-sampling, train once, probe |
| B. balanced union | balance, independent of order | same pool, but every domain pre-truncated/replicated to equal row count before decoding — needs zero code changes, only different shard files fed in |
| C. staged → combined | ordering | pretrain sequentially per domain for an equal step budget each, then a final combined phase on B's balanced pool — approximate today by chaining `--resume` across domain-specific configs if `load_resumable`'s fingerprint allows it, otherwise a small harness change |

Score all three identically with what already exists: the untrained-vs-trained probe gate,
`rep_std` (already instrumented, catches collapse), and both transfer probes above. Add one
cheap diagnostic that reuses `_linear_probe` completely unchanged — a **domain-identity
probe**: give it a `domain_id` label instead of a class label, known for free at
corpus-assembly time. High domain-identity accuracy partway through training would support
the coordinator's premise; if it stays low throughout, this encoder isn't budgeting capacity
to distinguish domains, and staging is unlikely to unlock the benefit it's meant to. This
also happens to be the exact experiment CSD would eventually want anyway if the goal is
"recognise that one frame contains several different kinds of thing" — the domain-identity
probe is a primitive version of that.

Cost: each configuration is one `vl_latent` run. The one currently on record took 497 seconds;
three to four configurations at comparable budgets is on the order of 30–60 minutes total —
trivial against the 43-minute full-fleet baseline already established elsewhere in this
document, and not run here.

**On domain balance, independent of ordering — this one I can answer with more confidence,
because it's visible directly in the code rather than inferred from adjacent literature.**
Yes, it matters on its own, and it matters *now*, regardless of what the staging experiment
finds. A naive union at the FULL available sizes found in this search (pxhere ≈1.1M,
PatchCamelyon 327,680, Shapes3D 480,000, CLEVR 100,000, Fashion-MNIST 70,000, EuroSAT 27,000)
would — given `torch.randint` over the concatenated pool — put EuroSAT under 2% of every
batch and Fashion-MNIST around 3%. That is not a vague risk; it is the direct, predictable
behaviour of the code as written, and it is exactly the failure the coordinator named: an
encoder that is functionally a pxhere-and-histopathology encoder that happened to see a
little satellite and garment imagery. The capped composite recommended above (~100k ceiling
on the three largest sources) is a first-order fix for this, achievable with **zero code
changes** — it only requires not handing the harness the full row counts. A proper
frequency-weighted or temperature-scaled sampler (the general pattern used for imbalanced
multilingual/multi-domain corpora elsewhere in ML, not something specific to this codebase)
would do this more precisely than capping shard sizes up front, but would need a small
addition to `_decode_split` or the training loop, which is out of scope for this document.

### What capability is lost, next to tiny-imagenet — stated plainly

tiny-imagenet gave `vl_latent` 100,000 images natively at 64×64, 200 real-world object/animal
classes at 500 images/class, drawn from a single coherent photographic domain with real
lighting, pose, and texture variation. That is precisely the thing this search could not find
a clean replacement for.

- **pxhere gets closest to photographic diversity** but has no class structure at all — free
  text tags only, no controlled category balance, and (being general stock photography) very
  likely skewed toward landscapes and travel rather than the graspable everyday objects
  ImageNet-style benchmarks emphasise. It cannot be probed on; it can only be pretrained on.
- **Caltech-101/256 have the right structure** — real object categories, real photographs —
  but at roughly a third (Caltech-256, 30,607 images / 257 categories, ≈119/class) to a tenth
  (Caltech-101, 9,146 / 101 categories, ≈90/class) of tiny-imagenet's *density*, well short of
  its 500/class, and native resolution is full-photo scale rather than already-curated 64×64,
  meaning more information is thrown away at resize time than tiny-imagenet ever had to
  discard.
- **Nothing else in this table is photographic at all** in the ImageNet sense: satellite,
  histopathology, synthetic 3D renders, sketches, garment product shots, and book engravings
  are all genuinely different domains, not substitutes for "photograph of an everyday object
  or animal."

**The honest statement the task asked for:** there is no permissively-licensed replacement of
tiny-imagenet's specific breadth — balanced, dense, natural-object-category photography at
100k+ scale — because that combination of properties essentially doesn't exist outside the
ImageNet lineage under a clean licence. The composite above buys breadth **across** visually
distinct domains (a model that has seen satellite imagery, tissue imagery, sketches, product
photography, and general photography) at the cost of depth **within** the one domain
(everyday-object photography) that ImageNet-family benchmarks were built to measure and that
most vision transfer-learning literature actually cares about. A `vl_latent` pretrained on
this composite should be expected to be **measurably weaker at fine-grained natural-object
recognition and generalisation** than the current unreleasable model — that is the real price
of the licence-clean requirement, not a rounding error, and it should be reported as such
rather than smoothed over.

### What I could not determine, in this section

0. **PatchCamelyon's "following the license of Camelyon16" claim** — Camelyon16's own site
   (`camelyon16.grand-challenge.org`) is a JS-rendered single-page app; a static fetch returns
   no content, and the GigaDB mirror is equally JS-rendered. The claim rests on `pcam`'s own
   README, a primary source from the dataset's actual creator, but was not independently
   re-derived from Camelyon16's own licensing statement.
1. **The British Library CC0 claim** could not be re-fetched fresh from `bl.uk` or Flickr
   Commons directly (both blocked/unreachable from this sandbox); it rests on the HF card's
   own detailed account plus circumstantial corroboration (pre-1900 publication dates,
   independently public domain by age; a named, identifiable GLAM data maintainer with
   unusually specific sourcing).
2. **NIST SD19's Standard-Reference-Data-vs-public-domain status**, which the entire EMNIST
   family's licence question rests on. Genuinely unresolved, not merely unverified — NIST's
   own policy distinguishes the two categories without stating which one SD19 falls into.
3. **Whether "free to use, please cite" (GTSRB) supports a redistribution grant for derived
   model weights at all**, as opposed to merely permitting use. No formal licence exists to
   answer this either way.
4. **DiffusionDB's generative-provenance question** — whether Stable Diffusion's own training
   corpus taints the copyright status of its outputs — is explicitly outside what any licence
   text can settle, and outside this document's scope. Flagged as the same *class* of
   unresolved question as the CC BY-SA derivative-work question already open elsewhere in this
   document, not resolved by analogy to it.
5. **Whether `HuggingFaceM4/Caltech-101`'s script-based loader still resolves** to a live data
   source — it uses a legacy `datasets`-library loading script rather than shipping parquet
   directly, and this was not tested since downloading was out of scope. `data.caltech.edu`'s
   own CC-BY-4.0 archive is the more durable source if a mirror needs rebuilding.
6. **No clean pixel-bearing HF mirror was found for Caltech-256** carrying a licence tag that
   matches the verified CaltechDATA upstream. The dataset should be built from
   `data.caltech.edu/records/nyy15-4j048` directly rather than trusted from an arbitrary
   untagged third-party mirror.
7. **Whether staged (per-domain, then combined) pretraining actually outperforms naive or
   balanced union for this model at this scale** — addressed above with a proposed experiment
   rather than an answer, because no literature or mechanism argument found here settles it.
8. **Whether `Xenova/quickdraw-small` (untagged) genuinely inherits the CC BY 4.0 of its
   sibling `Xenova/quickdraw`** (same uploader, same underlying Google data, tagged correctly)
   — very likely, given the shared lineage, but not independently confirmed for that specific
   repo id.

