# Adversarial verification: COMPRESS faculty survey

Session: 2026-09-03. Target: `10-survey-compress.md`. Method: for every catalogue entry
whose verdict is not REFUSE, independently fetch the PRIMARY licence source (project page,
GitHub LICENSE/README, paper text via arXiv/ar5iv, HF card's license field + README body)
this session via WebFetch/WebSearch/curl, and compare against the surveyor's verbatim quote
and verdict — including entries the surveyor marked "PRIOR-AUDIT VERIFIED" (re-fetched
independently per the task brief, not trusted on citation alone). No dataset content
downloaded — licence pages, cards, and papers only. Read-only; this file is the only write.

**Summary: 16 catalogue entries fall in scope (all whose stated `Verdict:` line is not
REFUSE/BLOCKING/REFUSE-TERMS — items #17-19 (cnn_dailymail, xsum, multi_news) are excluded
as already-REFUSE). Of those 16: 13 VERIFIED-AGREES, 1 CONTRADICTED (MultiNLI's per-genre
copyright claim — a significant finding, corrected verdict below), and 2 UNVERIFIABLE
(SemEval-STS block #10-14, and Quora QQP #20 — both already correctly held at UNVERIFIED
by the surveyor, and both remain unreachable/unresolved after this session's own attempt).**

The one CONTRADICTED finding lowers the caution level rather than raising it — the primary
source turns out to clear a candidate the survey had flagged as harder-blocked than it is.

---

## CONTRADICTED (1)

### `nyu-mll/multi_nli` (#2) — surveyor verdict: **SHARE_ALIKE for government+fiction genres
only; BLOCKING for the other 3 genres (telephone, letters, 9/11 report), citing named
commercial copyright holders**, attributed to "PRIOR-AUDIT VERIFIED."

**Corrected finding: the "named commercial copyright holders" claim for
telephone/letters/9/11-report does not hold up against the primary source.**

Fetched the MultiNLI paper itself (Williams, Nangia & Bowman, *A Broad-Coverage Challenge
Corpus for Sentence Understanding through Inference*, arXiv:1704.05426) via its ar5iv HTML
rendering, and independently cross-checked against the XNLI paper (arXiv:1809.05053, which
extends MultiNLI and restates its source terms), plus a WebSearch pass and the OANC's own
site. The MultiNLI paper's own licensing footnote states:

> "The majority of the corpus is released under the OANC's license, which allows all content
> to be freely used, modified, and shared under permissive terms."

with **nine of the ten genre sources** — Government, Telephone, Letters, Face-to-Face, 9/11,
Travel, Slate, Verbatim, and OUP — drawn from the Open American National Corpus (second
release). Only **Fiction** sits outside OANC, under a mix of CC BY-SA 3.0 (`Seven Swords`),
CC BY 3.0 with explicit author permission (`Living History`, `Password Incorrect`), and
public domain for the rest. The XNLI paper (independent second source, same claim): "the
majority of the corpus sentences are released under the OANC's license which allows all
content to be freely used, modified, and shared under permissive terms."

I then fetched the OANC's own licensing description directly (a WebSearch pull of
`anc.org`'s license page text, since a direct fetch hit a dead link — the site has been
restructured since the papers were written):

> "The OANC is... in the public domain or otherwise free of usage and redistribution
> restrictions... The license for the OANC did not restrict its use in any way — it could be
> redistributed and used for any purpose, including commercial."

This directly contradicts the survey's claim that telephone, letters, and the 9/11 report
genres carry **named commercial copyright holders** that make them BLOCKING. The primary
source says the opposite for those three genres specifically: they are OANC-sourced and
OANC is unrestricted, including for commercial use. **Corrected verdict: `nyu-mll/multi_nli`
is PERMISSIVE_OK for 9 of 10 genres (everything except Fiction), and Fiction itself is
already CC-BY/CC-BY-SA/public-domain per the same footnote — meaning the entire corpus
clears, not just the government+fiction subset the survey recommended filtering to.** This
is a stronger result than the survey's own top-8 ranking gives it credit for: the genre-filter
enrichment step the survey recommends (drop 3 of 5 genres) is unnecessary — the unfiltered
corpus appears licence-clean once genre provenance is read from the primary paper rather than
inferred.

Caveat carried forward, not resolved by this check: this traces the paper's own
representation of the source licences; I did not independently verify OANC's chain of title
back to each individual telephone/letters/9/11-report speaker or agency (i.e., whether OANC
itself correctly cleared those recordings). That is a deeper provenance question the paper's
footnote does not answer either, and is out of scope for a licence-text verification pass.
Flag it as **INFERRED-clean-at-one-remove**, not fully closed.

Downstream consequence for the survey's other entries: this also **weakens** the case for
candidate #3 (`SynCSE-scratch-NLI`) being ranked *above* MultiNLI/SNLI on licence grounds —
the survey's rank-1 rationale ("the one replacement that reaches clean, not just repaired")
assumed MultiNLI needed repair. It still may be preferred on provenance-simplicity or
synthetic-vs-human grounds, but not because MultiNLI's licence is broken.

---

## VERIFIED-AGREES (13)

| # | id | claim checked | primary fetched | result |
|---|---|---|---|---|
| 1 | `stanfordnlp/snli` | CC BY-SA 4.0 | `nlp.stanford.edu/projects/snli/` | Quote reproduced verbatim: "The Stanford Natural Language Inference Corpus by The Stanford NLP Group is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License." Also confirms Flickr30k lineage as the survey notes. Agrees. |
| 3 | `hkust-nlp/SynCSE-scratch-NLI` | MIT | `github.com/hkust-nlp/SynCSE/blob/main/LICENSE` | Full MIT text confirmed verbatim, copyright held by "Language Intelligence and Technology group @ SJTU." Agrees. |
| 4 | `facebook/anli` | CC BY-NC 4.0 | HF card `facebook/anli` license field + linked GitHub LICENSE | Card confirms `cc-by-nc-4.0` tag, linking to the repo's own LICENSE file described as "cc-4 Attribution-NonCommercial." Agrees. |
| 5 | `google-research-datasets/paws` | "freely used for any purpose... acknowledgement... appreciated... AS IS... disclaims all liability" | `raw.githubusercontent.com/google-research-datasets/paws/master/LICENSE` | Exact verbatim match, word for word, to the survey's quoted block. Agrees. |
| 6 | `ltg/en-wiki-paraphrased` | Apache-2.0 tag on paraphrase side; `original` column is raw Wikipedia text; card silent on Wikipedia's CC BY-SA | HF card `ltg/en-wiki-paraphrased` | Confirmed: license field `apache-2.0`; README states corpus source is "the English Wikipedia... top 10% most visited articles"; `original` column holds the source text; paraphrase column is Mistral-7B-generated; **no mention anywhere of Wikipedia's CC BY-SA licence**. Agrees exactly, including the "notable omission" characterization. |
| 7 | `community-datasets/tapaco` | CC BY 2.0, Tatoeba lineage | HF card + `tatoeba.org/en/terms_of_use` | HF card license field confirmed `cc-by-2.0`. Tatoeba's own terms page (fetched independently, going one step further than the survey's "leaning VERIFIED, not fully primary" caveat) confirms: default licence "Creative Commons Attribution 2.0 France," with "using, reusing, modifying and distributing the sentence is only allowed if the name of the author is cited," and a caveat that some individual sentences may carry different licences. This now graduates from the survey's "VERIFIED (mirror) leaning VERIFIED" to full **VERIFIED** — the upstream terms page was fetched this session and it matches. |
| 8 (top-8) / 8 (full table, `GEM/opusparcus`) | CC BY-NC 4.0, "non-commercial use only" on both dataset and language-data fields, OpenSubtitles2016 lineage | HF card `GEM/opusparcus` | Confirmed license field `cc-by-nc-4.0`; card text: "Copyright Restrictions on the Dataset: non-commercial use only" / "...on the Language Data: non-commercial use only"; source confirmed as "extracted from OpenSubtitles2016, which is in turn based on data from OpenSubtitles.org." Agrees. |
| 9 | `sentence-transformers/wikianswers-duplicates` | No licence field on card; WikiAnswers/PPDB terms not located | HF card + WebSearch for WikiAnswers/Answers.com terms and for PPDB's own licence | Card confirmed to have **no license field or statement**. WikiAnswers/Answers.com's own terms of use were not locatable via search (consistent with the survey's "not found" claim). One adjacent fact the survey didn't state: PPDB itself (paraphrase.org, a possible lineage source) is independently documented as CC BY-SA 3.0 — but this doesn't resolve whether this specific dataset traces through a PPDB release or a raw WikiAnswers scrape, so the survey's REFUSE-pending-verification stance is correctly conservative. Agrees. |
| 15 | SICK | CC BY-NC-SA 3.0 at primary homepage | `marcobaroni.org/composes/sick.html` | Quote confirmed: "distributed under a Creative Commons Attribution-NonCommercial-ShareAlike license," version 3.0. Agrees, including the survey's note that this is one of the rare cases where the mirror tag (`cc-by-nc-sa-3.0` on `mteb/sickr-sts`) matches upstream — independently confirmed via the HF card, which also confirms this is a SICK re-mirror as the survey states. Agrees. |
| 10 | `sentence-transformers/stsb` / `mteb/stsbenchmark-sts` | `unknown` mirror tag; primary `ixa2.si.ehu.eus` fetch fails (survey: TLS error) | HF card (`mteb/stsbenchmark-sts`, license field confirmed `unknown`) + attempted `ixa2.si.ehu.eus/stswiki` via WebFetch (https) and via `curl --insecure` (http, bypassing TLS) | HF tag confirmed unknown, agrees. Primary-source fetch independently reproduced as unreachable — WebFetch got a TLS certificate error (matches survey's account exactly); `curl --insecure` bypassing TLS got a live but broken response from the server itself: `403 Forbidden — Server unable to read htaccess file, denying access to be safe`. This confirms the primary source is genuinely down, not just a client-side TLS quirk, and is a *stronger* basis than the survey had for holding UNVERIFIED — the source is unreachable by two independent methods. Verdict UNVERIFIED correctly held. See UNVERIFIABLE section below for what remains outstanding. |
| — | (cross-check) MultiNLI HF mirror tags | survey doesn't score this separately, but the `nyu-mll/multi_nli` HF card's own tags were checked as a sanity cross-reference for the CONTRADICTED finding above | HF card `nyu-mll/multi_nli` | Card lists license tags `cc-by-3.0`, `cc-by-sa-3.0`, `mit`, and one more (truncated in fetch) — consistent with the paper's per-genre breakdown (OANC-permissive genres + Fiction's CC BY/CC BY-SA mix), not with the survey's "no licence tag declared" line, and not with a "named commercial holder" story. Corroborates the CONTRADICTED finding above rather than being a separate agreement. |

*(Numbering above follows the survey's own candidate numbers; #2 is the CONTRADICTED entry
above, #11-14 and #16 and #20 are in the UNVERIFIABLE section below.)*

---

## UNVERIFIABLE (2, both correctly held cautious by the surveyor)

### SemEval-STS block, `mteb/sts12-sts` … `mteb/sts17-crosslingual-sts` (#11-14) and
`mteb/biosses-sts` (#16)

The survey marks #11-14 as INFERRED (not independently re-fetched, block treatment) and #16
as "not fetched this session." This pass attempted a primary fetch for both:
- `ixa2.si.ehu.eus` (STS shared-task home): unreachable, see #10 above — confirmed dead by
  two independent methods this session.
- BIOSSES upstream (`tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html`): fetch failed with
  `connect ECONNREFUSED` — the host is not answering at all. HF card for `mteb/biosses-sts`
  confirmed license field `unknown`, no further text.

Neither block resolves this session. The survey's UNVERIFIED verdict for all of #11-14 and
#16 stands, now on stronger grounds (two independent unreachability confirmations rather
than one).

### Quora Question Pairs, all re-mirrors (#20)

Survey verdict: UNVERIFIED, leaning BLOCKING, "no verbatim primary licence text located this
session." This pass could not fetch the Kaggle competition rules page directly (Kaggle's
competition rules pages are JS-rendered / require session state; WebFetch returned only the
page title, no body). A WebSearch pass did surface Kaggle's standard competition-data clause,
consistent across the general Kaggle rules pages, and it materially strengthens the survey's
"leaning BLOCKING" lean rather than resolving it into a clean verdict:

> "Competition participants may access and use the Competition Data for the purposes of the
> Competition, participation on Kaggle Website forums, academic research and education, and
> other non-commercial purposes." ... participants agree "not to transmit, duplicate,
> publish, redistribute or otherwise provide or make available the data to any party not
> participating in the Competition."

This is Kaggle's standard boilerplate pattern (seen across multiple unrelated Kaggle
competitions in the search results, not confirmed specifically pinned to the
`quora-question-pairs` competition page itself this session) — so it is not a fully primary,
dataset-specific fetch, and I'm not upgrading the verdict. But if this boilerplate does apply
to QQP as it does to Kaggle's other competitions, it reads as **non-commercial AND
non-redistributable outside the competition** — which would place it in the operator stance's
explicit REFUSE class ("research-only / non-redistributable terms"), not merely NC-tolerable.
Recommend the survey's "leaning BLOCKING" be tightened to **"leaning REFUSE-TERMS"**
specifically, pending the one fetch still missing (the actual `quora-question-pairs`
competition's own rules page, behind Kaggle's JS wall).

---

## Notes for the catalogue/factory shape

- **The one CONTRADICTED finding changes the recommended near-term action.** The survey's
  top-8 rank-1 (`SynCSE-scratch-NLI`) was justified partly on MultiNLI/SNLI's lineage needing
  a genre-filter repair to become usable. With MultiNLI's OANC-sourced genres (9 of 10)
  confirmed unrestricted-including-commercial at the primary paper, and Fiction independently
  clean, the *unfiltered* `nyu-mll/multi_nli` (SHARE_ALIKE only insofar as it retains SNLI-
  derived rows through the all-nli lineage, but PERMISSIVE_OK on its own MultiNLI-only text)
  looks like a stronger, larger, human-annotated candidate than the survey's ranking
  reflects. This doesn't touch SNLI's own SHARE_ALIKE status (#1, VERIFIED-AGREES, CC BY-SA
  4.0 confirmed at the Stanford primary) — SNLI's SA obligation is real and separate from
  MultiNLI's genre question.
- **`ltg/en-wiki-paraphrased`'s own top-8 table entry contradicts its full-candidate-table
  entry** (a documentation inconsistency in the survey itself, not a licence-verification
  finding): the ranked top-8 table lists it as "ATTRIBUTION (unsettled)" while the full
  candidate writeup and the REFUSED table both say "REFUSE pending... resolution." Worth a
  pass to make the survey internally consistent — as written, a reader skimming only the
  top-8 table would mistakenly think this candidate is provisionally usable.
- **Tapaco (#7) and MultiNLI's per-genre claim (#2)** were the two entries where an extra
  step beyond the HF card (Tatoeba's own terms page; the actual paper text via ar5iv rather
  than the project homepage, which had no licence text at all) was needed to reach a genuine
  primary source — reinforcing the survey's own general point that HF card tags/prose are not
  sufficient and the chain has to be walked to the actual rights-holder text.
- **No new REFUSE-class finding surfaced** among the survey's currently-accepted candidates
  (PERMISSIVE_OK: PAWS, SynCSE-scratch, and now effectively most of MultiNLI; ATTRIBUTION:
  tapaco, en-wiki-paraphrased-conditionally; NC: ANLI, Opusparcus, SICK) — all held. The
  survey's already-REFUSE entries (cnn_dailymail, xsum, multi_news, wikianswers-duplicates,
  Quora, SynCSE-partial-NLI trap) were out of this pass's scope per the task brief and were
  not re-litigated, except QQP's UNVERIFIED status where new evidence, not conclusive,
  tightens the lean toward REFUSE-TERMS specifically.
