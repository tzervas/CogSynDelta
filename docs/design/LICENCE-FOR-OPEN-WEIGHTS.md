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
