# Adversarial verification: RETRIEVE / MEMORY faculty survey

Session: 2026-09-03. Target: `10-survey-retrieve-memory.md`. Method: for every catalogue
entry whose verdict is not REFUSE, independently fetch the PRIMARY licence source (project
page, GitHub LICENSE/README, paper data page, HF card's license field + README text) via
WebFetch/WebSearch this session, and compare against the surveyor's verbatim quote and
verdict. No dataset CONTENT downloaded — licence pages and cards only, per the task brief.
Read-only; this file is the only write.

**Summary: 22 of 24 catalogue entries fall in scope (all but the two verdict=REFUSE entries,
trivia_qa and trec-covid). Of those 22: 18 VERIFIED-AGREES, 3 entries CONTRADICTED (across 2
distinct findings — one on NFCorpus, one on the args.me lineage that covers both ArguAna and
Touche2020), 1 UNVERIFIABLE (Quora, reproduced, no new primary found). Plus one out-of-scope
note (XNLI, not a scored catalogue entry, but the surveyor's own flagged suspicion about it
checked anyway). ELI5 and Quora sit in the survey's Refused table under in-text verdicts
that aren't literally "REFUSE" ("lean REFUSE-TERMS" / "caution") — both were spot-checked
for completeness and are counted in the 22.**

The two CONTRADICTED findings both raise the caution level, not lower it — both are cases
where the surveyor's own "UNVERIFIED, don't trust the blanket tag" caution turns out to have
been well-founded, and the primary source, once actually fetched, resolves the ambiguity
against the mirror tag rather than for it.

---

## CONTRADICTED (2)

### `BeIR/nfcorpus` (#11) — surveyor verdict: UNVERIFIED ("do not admit until primary is
fetched"). **Corrected verdict: REFUSE.**

Fetched the actual NFCorpus project page (primary, not a mirror):
`https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/`

> "NFCorpus is free to use for **academic purposes**. For any other uses of the included
> NutritionFacts.org data please consult Terms of Service and contact its author Dr. Michael
> Greger directly."

This is not a blanket permissive or even a plain NC term — it is an **academic-use-only
grant with no redistribution/training licence extended for any other purpose**, gated behind
contacting a named third party for anything beyond that. This matches the operator stance's
named REFUSE class ("research-only / non-redistributable terms") at least as cleanly as
`mandarjoshi/trivia_qa` does, and more cleanly than MS MARCO's softer NC framing. The
surveyor correctly declined to trust BeIR's `cc-by-sa-4.0` blanket tag, but the entry needs
upgrading from "UNVERIFIED, fetch before admitting" to an outright REFUSE-class flag — the
primary source the surveyor said was needed has now been fetched, and it does not clear the
bar the surveyor was hoping it would.

### `BeIR/arguana` (#12) and `BeIR/webis-touche2020` (#19) — surveyor verdict on both:
UNVERIFIED (not individually verified this pass). **Corrected verdict: ATTRIBUTION**
(a third confirmed BeIR mirror-lie, same shape as `BeIR/scifact` #9 and `BeIR/scidocs` #10).

Both entries share the args.me corpus as their upstream. Fetched two independent primary
sources:
- Zenodo release record (`https://zenodo.org/record/3734893`, the corpus's own archival
  DOI record): license field reads **"Creative Commons Attribution 4.0 International"**.
- HF's own `webis/args_me` mirror card (a *different* repackage from the BeIR one, useful as
  a cross-check): license tag **`cc-by-4.0`**, README's Licensing Information section states
  the same CC BY 4.0 grant.

Neither primary source says CC BY-SA 4.0. `BeIR/arguana` and `BeIR/webis-touche2020` both
carry BeIR's blanket `cc-by-sa-4.0` mirror tag, same as the two mismatches the surveyor
already caught (scifact, scidocs). This is the same failure pattern recurring a third time
in the same mirror family and the survey's blanket "not individually verified this pass"
treatment for these two entries should be replaced with a positive ATTRIBUTION verdict
(CC BY 4.0, not share-alike) — attribution-only obligation, no share-alike propagation
requirement, which is *less* restrictive than the surveyor's cautious default assumed but
still not what the mirror tag claims.

---

## VERIFIED-AGREES (18)

For each, primary source fetched this pass, quote/finding compared against the survey's
claim; all agree.

| # | id | claim checked | primary fetched | result |
|---|---|---|---|---|
| 1 | `microsoft/ms_marco` | NC verdict, exact disclaimer quote | microsoft.github.io/msmarco/ | Quote reproduced verbatim: "intended for non-commercial research purposes only... without extending any license... we may not own the underlying rights in the documents." Agrees exactly, including the "disclaims ownership" nuance the surveyor flagged for operator judgment. |
| 2 | `sentence-transformers/natural-questions` | Apache-2.0 at primary | raw.githubusercontent.com/google-research-datasets/natural-questions/master/LICENSE | Confirmed Apache License 2.0 full text. Agrees. |
| 3 | `allenai/gooaq` | LICENSE=Apache-2.0 but README imposes NC (internal contradiction → NC per strictest-input rule) | raw.githubusercontent.com/allenai/gooaq/main/LICENSE + github.com/allenai/gooaq README | LICENSE file confirmed Apache-2.0 text; README confirmed verbatim: "NOTE This dataset should not be used for any commercial purposes. See the license for the detailed terms." Both halves of the surveyor's claim reproduce exactly. Agrees. |
| 4 | `hotpotqa/hotpot_qa` | CC BY-SA 4.0, mirror matches | hotpotqa.github.io | Confirmed: "HotpotQA is distributed under a CC BY-SA 4.0 License," Wikipedia corpus likewise. Agrees. |
| 5 | `rajpurkar/squad` | CC BY-SA 4.0 | rajpurkar.github.io/SQuAD-explorer/ | Confirmed "CC BY-SA 4.0" in the download section. Agrees. |
| 7 | `facebookresearch/ELI5` (Refused-table entry, spot-checked anyway) | BSD code licence does not extend to Reddit content; no mention of Reddit rights in the licence | raw.githubusercontent.com/facebookresearch/ELI5/master/LICENSE | Confirmed standard BSD-3 text, copyright Facebook, Inc.; no mention of Reddit anywhere. Agrees with the surveyor's "code_only, does not cover content" reasoning. |
| 8 | `BeIR/fiqa` | NC, mirror-lie vs primary task site | sites.google.com/view/fiqa/home | Confirmed verbatim: "The training data is available only for non-commercial use." / "The testing data is available only for non-commercial use." Agrees exactly. |
| 9 | `BeIR/scifact` | split licence: claims CC BY 4.0, corpus ODC-By 1.0, code Apache-2.0 | raw.githubusercontent.com/allenai/scifact/master/LICENSE.md | All three components confirmed as claimed. Agrees. |
| 10 | `BeIR/scidocs` | CC BY 4.0 (not SA) | raw.githubusercontent.com/allenai/scidocs/master/LICENSE | Confirmed full CC BY 4.0 legal text. Agrees. |
| 14 | `BeIR/fever` | partial verification only: Wikipedia-derived text at CC BY-SA 3.0 (version behind the mirror's 4.0 tag), FEVER-specific statement not isolated | fever.ai/download/fever/license.html | Confirmed: page yields CC BY-SA 3.0 language tied to Wikipedia annotations, no FEVER-dataset-specific statement isolated. Matches the surveyor's own "partially VERIFIED" framing exactly. Agrees. |
| 15 | `BeIR/climate-fever` | INFERRED (not independently confirmed) same shape as FEVER | sustainablefinance.uzh.ch/en/research/climate-fever.html | No explicit licence found on the primary project page (only a citation request). Consistent with the surveyor's own "not independently confirmed" status — does not contradict, does not newly confirm. Agrees with "still UNVERIFIED." |
| 17 | `BeIR/dbpedia-entity` | INFERRED CC BY-SA 3.0 (same class as mirror tag, version drift) | dbpedia.org/about/ | Confirmed: "Creative Commons Attribution-ShareAlike 3.0 License and the GNU Free Documentation License." Upgrades the surveyor's INFERRED claim to VERIFIED. Agrees. |
| 18 | `BeIR/cqadupstack` | INFERRED StackExchange CC BY-SA 4.0 ("most likely correct BeIR tag") | archive.org/details/stackexchange (official SE data-dump listing) | Confirmed: "All user content contributed to the Stack Exchange network is cc-by-sa 4.0 licensed." Upgrades INFERRED to VERIFIED. Agrees — this is the one BeIR blanket tag the surveyor predicted would actually hold, and it does. |
| 20 | `miracl/miracl` | repo badge Apache-2.0 is code_only; underlying Wikipedia passages carry CC BY-SA regardless (surveyor's own inference, not primary-stated) | github.com/project-miracl/miracl | Confirmed Apache-2.0 badge; confirmed the repo itself does **not** restate Wikipedia's licence terms for the packaged passages — matches the surveyor's explicit framing that this is the surveyor's own reasoning about underlying content, not something the primary page states outright. Agrees, including the caveat. |
| 21 | `castorini/mr-tydi` | Apache License 2.0 | github.com/castorini/mr.tydi | Confirmed verbatim: "Mr. TyDi is licensed under the Apache License 2.0." Agrees. |
| 22 | `deepmind/narrativeqa` | Apache-2.0 badge covers code/annotations; no rights statement found for underlying books/scripts | github.com/deepmind/narrativeqa | Confirmed Apache License 2.0 badge; confirmed no explicit statement on rights to the underlying book/script texts. Agrees. |
| 23 | `allenai/qasper` | allenai.org page redirects straight to the HF card (same-host problem, fails independent-primary standard even though the tag "looks right") | allenai.org/data/qasper → huggingface.co/datasets/allenai/qasper | Confirmed 302 redirect straight to the HF card; HF card shows `cc-by-4.0` / "CC BY 4.0" with underlying papers from S2ORC. No independent second source (e.g. a GitHub repo or the QASPER paper's data-availability statement) was located this pass either. Per the survey's own strict standard, status stays UNVERIFIED-AT-PRIMARY. Agrees with the self-critical framing — not upgraded. |
| 24 | `tau/scrolls` | not fetched by the surveyor; flagged as a bundling convenience, absent HF licence tag is itself a small red flag | scrolls-benchmark.com | No licence or terms-of-use language found on the primary page — matches the "not fetched, decompose to members" framing; nothing here resolves it either way. Agrees. |

---

## UNVERIFIABLE (carried forward, no new primary found)

`BeIR/quora` (#13) — see row above; 403 on the primary page reproduced, WebSearch found no
alternative authoritative source. Same status as the survey: UNVERIFIED, not INFERRED.

---

## Not scored by the survey, but its own flagged suspicion checked anyway

**XNLI** — the survey's closing "what still needs verification" section explicitly flagged
that XNLI "is commonly cited elsewhere as CC BY-NC 4.0, which would make it a second
FiQA-shaped mismatch if the HF mirror's blank license field gets naively treated as 'no
restriction.'" WebSearch this pass confirms multiple independent sources (HF dataset-card
license-conflict discussion, TFDS catalog entry, downstream model cards) citing XNLI at
**CC BY-NC 4.0**. XNLI was not one of the 24 scored catalogue entries so there is no verdict
to correct, but the surveyor's instinct to flag it before any future ingest pass treats the
blank mirror field as unrestricted is corroborated — worth carrying into whatever entry
eventually scores XNLI formally.

---

## Entries excluded from this pass per the mandate (verdict = REFUSE)

`mandarjoshi/trivia_qa` (#6) and `BeIR/trec-covid` (#16) carry verdict REFUSE in the survey
and were out of scope for this adversarial pass ("for EVERY candidate with verdict other
than REFUSE"). Not re-verified here.

---

## Net effect on the survey's own rankings

Neither CONTRADICTED item was in the survey's "Top 8 candidates" list, so the ranked
recommendations are unaffected. The two corrections matter for the parts of the survey that
*do* carry forward:
- NFCorpus should move from "UNVERIFIED, fetch before admitting" into the Refused table
  alongside trivia_qa/ELI5/Quora — same REFUSE-class shape (research-only, permission-gated).
- ArguAna and Touche2020 should move from "UNVERIFIED" to a confirmed ATTRIBUTION verdict
  (CC BY 4.0) — cleaner than the surveyor's cautious default assumed, and both should join
  the args.me provenance group the survey already anticipated ("collapse for B1/B2 if both
  are admitted" — that grouping call was correct, only the underlying licence class needed
  fixing from "unknown, share-alike-shaped" to "confirmed attribution-only").

All other checked verdicts hold. The survey's core headline — that BeIR's blanket
`cc-by-sa-4.0` mirror tag cannot be trusted and each subset needs individual primary
verification — is reinforced rather than weakened by this pass: five of eleven `BeIR/*`
entries checked (scifact, scidocs, fiqa, arguana, webis-touche2020) turn out to mismatch the
blanket tag, against two the survey itself had already caught. Only fever, cqadupstack, and
dbpedia-entity confirm as licence-class-consistent (share-alike, modulo version-number
drift), and cqadupstack's is the only one that also matches on version number.
