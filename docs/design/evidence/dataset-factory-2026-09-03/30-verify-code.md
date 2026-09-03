# Dataset factory — adversarial verification of `10-survey-code.md`

Verifier pass, 2026-09-03. For every candidate in `10-survey-code.md` with a verdict other
than a closed REFUSE, the primary licence source was fetched independently this session
(GitHub `LICENSE` files via `raw.githubusercontent.com`, HF dataset cards via
`huggingface.co/datasets/<id>/raw/main/README.md` and the `/api/datasets/<id>` endpoint,
project pages, and — for `newfacade/LeetCodeDataset` — LeetCode's own Terms of Service,
recovered via the Wayback Machine after the live page 403'd both `WebFetch` and a
browser-UA `curl`). No dataset content was downloaded, only licence/card/ToS text. Fetched
evidence is saved beside this file under `licence-texts/` (see file list at the bottom).

**Marking key:** `VERIFIED-AGREES` = primary text fetched, matches the surveyor's quote and
verdict. `CONTRADICTED` = primary text fetched, contradicts the surveyor's quote or verdict
(corrected verdict given). `UNVERIFIABLE` = fetch attempted, primary source does not
resolve the question either way (what was missing is stated). A few entries are marked
`VERIFIED-AGREES (upgraded)` where the surveyor had marked something INFERRED pending a
follow-up fetch, and that follow-up fetch is the one done in this pass.

---

## Headline findings (read this section first)

### 1. CONTRADICTED — `bigcode/commitpackft` (#9) and `bigcode/commitpack` (#10): the card contradicts its own blanket "permissive" claim

The surveyor recorded these as `PERMISSIVE_OK (INFERRED, needs one follow-up VERIFIED
fetch)` and flagged the follow-up as needed. That follow-up was done this pass by fetching
`https://huggingface.co/datasets/bigcode/commitpackft/raw/main/README.md` (and the
`commitpack` card, same structure) directly.

The card's **Licensing Information** section says:

> "Each sample comes from a code repository with a permissive license. The license is
> provided by the `license` field for each sample."

But the card's own **Data Fields** section documents that field's actual value space:

> "`license`: license of the repository the code stems from, one of `['mit',
> 'artistic-2.0', 'isc', 'cc0-1.0', 'epl-1.0', 'mpl-2.0', 'unlicense', 'unknown',
> 'apache-2.0', 'bsd-3-clause', 'agpl-3.0', 'lgpl-2.1', 'bsd-2-clause']`"

`agpl-3.0` is **not a permissive licence** — it is the strongest copyleft licence in
common use (network-copyleft; many organisations treat AGPL-derived training data as at
least SHARE_ALIKE and some treat it as BLOCKING for a closed release). `unknown` means an
unspecified fraction of rows carry no resolved licence at all. The card asserts "permissive"
for every sample while its own schema documents that some samples are AGPL-3.0 or
unclassified. This is the identical shape twice (commitpackft is a filtered subset of
commitpack, both cards carry byte-identical Licensing Information and Data Fields
sections).

**Corrected verdict:** do **not** admit either dataset as a blanket `PERMISSIVE_OK`.
Per-row `license` field filtering is required before training — keep only rows whose
`license` value is genuinely permissive (mit/apache-2.0/bsd-2/3-clause/isc/cc0-1.0/
unlicense; `mpl-2.0`/`lgpl-2.1`/`epl-1.0` are file-level weak-copyleft and need their own
strictest-input judgment call) and drop `agpl-3.0` and `unknown` rows outright. Until that
filter is built, the dataset as a whole is **UNVERIFIED-as-a-blanket-grant**, not
`PERMISSIVE_OK` even INFERRED. This also means PG-COMMITPACK is not "confirm and move on"
for the commit-message gap the survey wanted filled — it needs enrichment work
(row-filtering) before it is safe to mix in, not zero enrichment as the survey's entry #9
"enrichment: none required" line states.

Local copy: `licence-texts/card-bigcode-commitpackft-20260903.md`,
`licence-texts/card-bigcode-commitpack-20260903.md` (see the `license` field enum near the
`Data Fields` heading and the `Licensing Information` heading in each).

### 2. CONTRADICTED — `m-a-p/CodeFeedback-Filtered-Instruction` (#17): not an independent provenance group, and it inherits this survey's own #18 licence conflict

The surveyor recorded `provenance_group: independent (own lineage, though not deeply
traced upstream)` and flagged "trace the Magicoder/WizardCoder-lineage question before
treating as a fully independent provenance group" as future work.

The card (`https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction/raw/main/README.md`,
fetched this pass) states the lineage plainly, no deep tracing required:

> "CodeFeedback-Filtered-Instruction is a curated collection of code instruction queries
> extracted from four prominent open-source code instruction tuning datasets:
> Magicoder-OSS-Instruct, Python code subset of ShareGPT, Magicoder-Evol-Instruct, and
> **Evol-Instruct-Code** [linked to `nickrosh/Evol-Instruct-Code-80k-v1`]."

`nickrosh/Evol-Instruct-Code-80k-v1` is catalogue entry **#18 in this same survey**, held
at `UNVERIFIED` because its HF mirror tag (`cc-by-nc-sa-4.0`) directly conflicts with its
GitHub primary source (`Apache-2.0`) — both VERIFIED-fetched, unresolved. CodeFeedback is
therefore not "independent (own lineage)"; a portion of its 156k queries are downstream of
the very licence dispute the survey itself opened three entries later in the same document,
and the entry should say so.

Two more things the card surfaces that the survey's #17 entry doesn't carry forward with
matching weight:
- The card carries its own explicit warning banner, not just an inferred caveat: "⚠️The
  dataset contains part data generated by OpenAI's language models, please pay attention
  to OpenAI's usage policy when adopting this dataset" — same OpenAI-output caveat class
  as #15/#18, but here it's the *dataset's own author* saying it, not a survey inference.
- One of the four named sources is `ajibawa-2023/Python-Code-23k-ShareGPT` — "ShareGPT"
  data is itself sourced from a browser extension that scraped users' ChatGPT
  conversations; that has its own consent/ownership question independent of the OpenAI-
  output question, not mentioned in the survey entry at all.

**Corrected verdict:** re-flag as `PERMISSIVE_OK (INFERRED)` **conditional on #18
resolving in Apache-2.0's favour**; if #18 resolves the other way (NC-SA binds), the
Evol-Instruct-Code-80k-v1-derived fraction of CodeFeedback inherits that NC-SA obligation
under strictest-input, and CodeFeedback can no longer be treated as a clean independent
PERMISSIVE_OK source until that fraction is isolated or dropped.

Local copy: `licence-texts/card-mapCodeFeedback-20260903.md`.

### 3. VERIFIED-AGREES, upgraded to CONFIRMED — `newfacade/LeetCodeDataset` (#19): REFUSE-TERMS is no longer presumptive

The survey marked this `REFUSE-TERMS (recommended)` but noted "needs a formal LeetCode ToS
re-fetch to convert this from 'presumptive' to a closed verdict." `leetcode.com/terms/`
403'd on both a direct `WebFetch` and a browser-UA `curl` this pass; a Wayback Machine
snapshot from 2026-08-05 (`http://web.archive.org/web/20260805002017/https://leetcode.com/terms/`)
was fetched instead and confirms the presumption directly, verbatim:

> "Activities such as 'crawling,' 'scraping,' or 'spidering' any part of the Service, and
> attempting to decompile, reverse engineer, or discover the source code or underlying
> ideas of the Service, are strictly forbidden."

> "'Content' refers to all software, images, **questions**, communications, solutions, and
> any related material perceived or made available from our Service platform... You agree
> that all Content is our sole and exclusive property... All Our Content is copyrighted
> under United States copyright laws."

"Questions" (LeetCode's own word for problem statements) is explicitly enumerated inside
the ToS's definition of "Content," which the ToS asserts is LeetCode's "sole and exclusive
property." This is a clean, closed match to the operator stance's REFUSE class
("research-only / non-redistributable terms... distributors that disclaim owning what they
distribute" — here inverted: the distributor, newfacade, plainly does *not* own what it
redistributes, LeetCode does, and LeetCode's terms forbid the scraping newfacade's own
repo README says nothing about doing legitimately).

One additional, non-blocking factual note: `newfacade/LeetCodeDataset`'s HF mirror tag is
`apache-2.0` (confirmed via the HF API this pass), but the GitHub repo's own `LICENSE` file
(fetched this pass, `raw.githubusercontent.com/newfacade/LeetCodeDataset/main/LICENSE`) is
**MIT**, not Apache-2.0 — an internal inconsistency in what newfacade itself claims, though
moot: neither licence can grant rights newfacade doesn't hold over LeetCode's content, so
this doesn't change the REFUSE verdict.

**Corrected verdict:** `REFUSE` (closed, not presumptive).

Local copy: `licence-texts/leetcode-terms-wayback-20260805.html`,
`licence-texts/newfacade-LeetCodeDataset-github-README-20260903.md`.

---

## Entry-by-entry verification

### 1-3. `bigcode/the-stack-v2-dedup`, `bigcode/starcoderdata`, `bigcode/the-stack-dedup` (PG-STACK)
**VERIFIED-AGREES.** Fetched `https://www.bigcode-project.org/docs/pages/bigcode-openrail/`
directly this pass. Confirms, near-verbatim to the survey's quotes: commercial use
permitted ("...for research, commercial or non-commercial purposes"); training/fine-tuning
permitted (Attachment A examples list "Quantization, fine-tuning, prompt-tuning, using
adaptors" as normal Modifications, not restricted uses); redistribution requires "the same
use restrictions, or a similar set of use restrictions accomplishing the same purpose"
(propagation, matching the survey's SHARE_ALIKE-adjacent call); an opt-out tool exists ("Am
I in the Stack?"). Mirror tags confirmed via the HF API this pass: all three are
`license: other`, and — new information the survey's #1 entry flagged only for
`the-stack-v2-dedup` — the API's `gated` field is `auto` for **all three** PG-STACK rows
checked (`the-stack-v2-dedup`, `starcoderdata`, `the-stack-dedup`), not just #1. The
survey's top-8 table ranks `starcoderdata` (#2) above `the-stack-v2-dedup` (#1) partly on
"already-permissive-filtered... reducing the OpenRAIL-M concern to the wrapper/access
licence" without noting #2 carries the identical click-through gate as #1 — a practical
integration-cost omission, not a licence error, worth correcting before treating #2 as
lower-friction than #1.

### 4. `HuggingFaceTB/stack-edu`
**VERIFIED-AGREES.** Card fetched this pass matches the survey's quote closely:
"This dataset only contains the SWHIDs to download the code files and not the content of
the files itself. The contents can be downloaded from Software Heritage's S3 bucket to
ensure data compliance." / "Please refer to [the-stack-v2] for the data license." Row count
also independently re-confirmed via the HF API (`167063359`... actually the API returned
the dataset-info split rows visible in the card's own YAML, which sum across
train/val/test configs to figures consistent with the survey's `167,063,359`-scale claim —
not re-verified digit-for-digit this pass, but the card structure and metadata_only framing
match exactly). No contradiction.

### 5. `code-search-net/code_search_net`
**VERIFIED-AGREES.** `https://raw.githubusercontent.com/github/CodeSearchNet/master/LICENSE`
fetched this pass: "MIT License / Copyright (c) 2019 GitHub / Permission is hereby granted,
free of charge..." — matches the survey's quote exactly. Mirror tag `license: other`
reconfirmed via HF API. The mirror-lies pattern is real and correctly resolved in the
corpus's favour.

### 6. `Nan-Do/code-search-net-python`
**CONTRADICTED (minor, doesn't change verdict).** The survey says "mirror_tag: none stated
in the earlier search pass; treat as inheriting #5's licence." The card, fetched this pass
(`https://huggingface.co/datasets/Nan-Do/code-search-net-python/raw/main/README.md`), has
`license: apache-2.0` in its YAML front matter — self-declared by the uploader, not unstated.
This is a factual correction, not a verdict correction: Apache-2.0 doesn't conflict with
CodeSearchNet's MIT upstream (Apache-2.0 is at least as permissive and MIT-compatible), so
`PERMISSIVE_OK` still holds, now on slightly firmer footing (an independent, if unverified,
statement rather than a bare inheritance assumption). Incidentally, the card's
`num_examples: 455243` matches `00-ground.md`'s "455,243 rows CodeSearchNet-lineage" figure
exactly, cross-confirming that ground-doc number.

### 7. `codeparrot/apps`
**VERIFIED-AGREES.** `https://raw.githubusercontent.com/hendrycks/apps/main/LICENSE`
fetched this pass: "MIT License / Copyright (c) 2021 Dan Hendrycks..." — matches. Mirror
tag `license: mit` reconfirmed via HF API.

### 8. `deepmind/code_contests`
**VERIFIED-AGREES.** `https://raw.githubusercontent.com/google-deepmind/code_contests/master/README.md`
fetched this pass, License section:

> "The code is licensed under the Apache 2.0 License. All non-code materials provided are
> made available under the terms of the CC BY 4.0 license... Codeforces materials are
> sourced from http://codeforces.com. Description2Code materials are sourced from...
> licensed under the MIT open source license... CodeNet materials are sourced from...
> Apache 2.0... Use of the third-party software, libraries code or data may be governed by
> separate terms and conditions... We make no representations here with respect to rights
> or abilities to use any such materials."

This matches the survey's quote and its caveat about third-party problem statements not
being independently re-licensed by DeepMind — the README's own final sentence is DeepMind
disclaiming exactly that. Apache-2.0 LICENSE file also fetched directly and confirmed.
`ATTRIBUTION` verdict stands.

### 9-10. `bigcode/commitpackft`, `bigcode/commitpack`
**CONTRADICTED** — see Headline Finding 1 above.

### 11. `OpenCoder-LLM/opc-annealing-corpus`
**VERIFIED-AGREES.** Card fetched this pass: "algorithmic_corpus: Algorithm-related code
sampled from The Stack v2." Confirms the provenance conflict the survey flagged (an
`odc-by` tag sitting on Stack-v2-sourced content) is real and, per this pass, still
unresolved in the card text itself — no opt-out cross-check or ODC-BY justification is
given anywhere on the card. `UNVERIFIED` verdict stands.

### 12. `OpenCoder-LLM/opc-fineweb-code-corpus`
**UNVERIFIABLE beyond the mirror tag** (matches the survey's own INFERRED caveat — not
upgraded, not contradicted). Card fetched this pass shows only the bare `license: mit` YAML
field, no Licensing Information prose. The OpenCoder-LLM/OpenCoder-llm GitHub repo's own
`LICENSE` file (fetched this pass) is MIT, but it covers the project's training/eval code,
not an explicit statement about this specific web-crawl dataset's content licence. A
WebSearch for the OpenCoder technical report's data-licensing statement turned up no
independent confirmation either. Leave as `PERMISSIVE_OK (INFERRED)`, flag for a follow-up
fetch of the arXiv paper's data-availability section specifically, not just the repo.

### 13-14. `nvidia/OpenCodeReasoning`, `nvidia/OpenCodeReasoning-2`
**VERIFIED-AGREES.** Card fetched this pass, License/Terms of Use section: "This dataset is
licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0)..."
and Intended Use section: "The data may be freely used to train models. However, for each
dataset an user elects to use, the user is responsible for checking if the dataset license
is fit for the intended purpose." Card also explicitly names sources including direct
"CodeForces materials are sourced from http://codeforces.com" (10,069 CodeForces-sourced
rows visible in the card's own per-source breakdown table). Matches the survey's quote and
its two-layer verdict (ATTRIBUTION at the NVIDIA-generated layer, UNVERIFIED at the
scraped-problem-statement layer) exactly.

### 15. `KodCode/KodCode-V1`
**VERIFIED-AGREES.** Card fetched this pass: `license: cc-by-nc-4.0` in YAML, plus prose
"**License**: Please follow CC BY-NC 4.0," and confirms `gpt-4o-0513` as the generator for
both `solution` and `test` fields. Matches the survey's quote and `NC` verdict exactly.

### 16. `neulab/conala`
**VERIFIED-AGREES on substance, one detail corrected.** The survey says the project page
"returned HTTP 404 on fetch attempt." Re-fetched this pass (`https://conala-corpus.github.io/`)
and got a normal 200 with content — but that content confirms the *substance* the survey
was reaching for: no license statement anywhere on the page, and no mention of Stack
Overflow's own content terms. Also checked this pass and not in the original survey: the
`neulab/conala` GitHub repo has **no `LICENSE` file** (`raw.githubusercontent.com/neulab/conala/master/LICENSE`
→ 404, confirmed via both the raw-file path and the GitHub API's `/repos/neulab/conala/license`
endpoint → 404). The HF card (`license: [mit]` in YAML) is the *only* place any licence
claim exists for a StackOverflow-crawled corpus, and Stack Exchange content is CC BY-SA 4.0
by platform ToS (not independently re-quoted this pass either, flagged for a follow-up same
as the survey). `UNVERIFIED` verdict stands, now on stronger evidence (no LICENSE file at
all, not just a stale project page).

### 17. `m-a-p/CodeFeedback-Filtered-Instruction`
**CONTRADICTED** — see Headline Finding 2 above.

### 18. `nickrosh/Evol-Instruct-Code-80k-v1`
**VERIFIED-AGREES.** Mirror tag `cc-by-nc-sa-4.0` reconfirmed via HF API.
`https://raw.githubusercontent.com/nickrosh/Evol-Teacher/main/LICENSE` fetched this pass:
"Apache License / Version 2.0..." — the conflict is real and both sides are independently
VERIFIED-fetched, exactly as the survey states. This is the live "mirror is *more*
restrictive than upstream" case the survey called out. Unresolved; both this survey and #17
above now depend on its resolution.

### 19. `newfacade/LeetCodeDataset`
**VERIFIED-AGREES, upgraded to closed** — see Headline Finding 3 above.

### 20. `openai/openai_humaneval`
**VERIFIED-AGREES (upgraded from INFERRED to VERIFIED).** The survey flagged this as
"high-confidence INFERRED rather than independently re-fetched." Fetched this pass:
`https://raw.githubusercontent.com/openai/human-eval/master/LICENSE` → "The MIT License /
Copyright (c) OpenAI (https://openai.com) / Permission is hereby granted, free of charge...
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies..." Mirror
tag `license: mit` reconfirmed via HF API. `PERMISSIVE_OK, EVAL-ONLY` stands, now fully
verified rather than inferred.

### 21. `evalplus/humanevalplus`, `evalplus/mbppplus`
**VERIFIED-AGREES (upgraded from INFERRED to VERIFIED).**
`https://raw.githubusercontent.com/evalplus/evalplus/master/LICENSE` fetched this pass:
Apache License, Version 2.0 — matches both mirror tags (`apache-2.0`, reconfirmed via HF
API for both repos). `PERMISSIVE_OK, EVAL-ONLY` stands.

### 22. `google-research-datasets/mbpp`
**VERIFIED-AGREES, with a nuance the survey didn't carry.** The survey said "CC BY 4.0 is
the well-documented standard licence for this benchmark — not independently re-fetched from
the GitHub tree this pass." This pass fetched both: the `mbpp/README.md` file inside the
`google-research/google-research` monorepo (`raw.githubusercontent.com/google-research/
google-research/master/mbpp/README.md`) states **no licence at all** for the dataset
specifically, and the enclosing monorepo's own blanket `LICENSE` file (fetched this pass)
is **Apache-2.0**, not CC BY 4.0. The CC BY 4.0 claim is confirmed only by the HF card's own
"Licensing Information: CC-BY-4.0" section (fetched this pass) — written by the dataset's
own team, who publish directly to HF, the same code-licence/data-licence split pattern seen
in `deepmind/code_contests` (#8). **The GitHub tree does not itself confirm CC BY 4.0; the
HF card does.** Recommend the catalogue's `licence_upstream_source` field for this row point
at the HF card, not at GitHub, since that's where the operative statement actually lives.
Verdict `ATTRIBUTION, EVAL-ONLY` stands.

### 23. `zai-org/humaneval-x`, `bigcode/humanevalpack`
**VERIFIED-AGREES on tags; UNVERIFIABLE beyond them.** Both mirror tags reconfirmed via
card YAML fetched this pass (`humaneval-x`: `license: [apache-2.0]`, list format, easy to
miss with a naive grep — noted for future surveyor tooling; `humanevalpack`:
`license: mit`). Neither card has actual prose text under its "Licensing Information"
heading — `humanevalpack`'s heading exists only in the table of contents with no filled-in
section body, a card-quality gap rather than a licence contradiction. `PERMISSIVE_OK,
EVAL-ONLY` stands for both, unverified beyond the bare tag.

### 24. `SWE-bench/SWE-smith`
**VERIFIED-AGREES (upgraded from INFERRED to VERIFIED).**
`https://raw.githubusercontent.com/SWE-bench/SWE-smith/main/LICENSE` fetched this pass:
"MIT License / Copyright (c) 2025 John Yang..." Mirror tag `license: mit` reconfirmed via
HF API. Card's own README (fetched this pass) carries no additional generation-provenance
caveat beyond the licence tag. `PERMISSIVE_OK` stands, now fully verified.

### 25. `princeton-nlp/SWE-bench`, `SWE-bench/SWE-bench_Verified`
**VERIFIED-AGREES.** Both HF cards confirmed via API to carry an empty `cardData.license`
(no tag at all), matching the survey. The `SWE-bench/SWE-bench` GitHub repo does carry its
own MIT `LICENSE` file (fetched this pass:
`https://raw.githubusercontent.com/SWE-bench/SWE-bench/main/LICENSE` → "MIT License /
Copyright (c) 2023 Carlos E Jimenez, John Yang..."), but — as the survey itself already
reasoned — that licenses the SWE-bench *harness/tooling* codebase, not the task-instance
content, which is pulled from real GitHub issues/PRs across many third-party repos each
carrying that repo's own licence, with no blanket licence asserted by the SWE-bench project
over the aggregation. `UNVERIFIED, EVAL-ONLY` stands regardless of the outcome — training
on it is a contamination hazard independent of the licence question, per the survey's own
framing.

---

## REFUSED section re-check

- **`newfacade/LeetCodeDataset`** — moved from presumptive to **closed REFUSE**; see
  Headline Finding 3.
- **`nickrosh/Evol-Instruct-Code-80k-v1`** — re-confirmed `UNVERIFIED`, both sides of the
  conflict independently re-fetched this pass and both hold (see #18 above).
- **`neulab/conala`** — re-confirmed `UNVERIFIED`, on stronger evidence this pass (no
  GitHub LICENSE file at all, not merely a stale project page); see #16 above.
- **`OpenCoder-LLM/opc-annealing-corpus`** — re-confirmed `UNVERIFIED`, the card's own text
  states the Stack-v2 sourcing that creates the ODC-BY-vs-OpenRAIL-M conflict; see #11
  above.
- Direct scraping of live Codeforces/AtCoder/HackerRank pages — not independently
  re-surveyed this pass (still out of scope, as the original survey notes); the LeetCode
  ToS text fetched for #19 is a useful template for what to look for if any of those
  platforms' own ToS gets checked later (all commonly carry similar no-scraping /
  content-ownership clauses).

---

## Summary of verdict changes

| id | survey verdict | this pass | change |
|---|---|---|---|
| `bigcode/commitpackft` | PERMISSIVE_OK (INFERRED) | **UNVERIFIED-as-blanket-grant** — card's own per-row `license` field includes `agpl-3.0`/`unknown`; needs row-level filtering before training | CONTRADICTED |
| `bigcode/commitpack` | PERMISSIVE_OK (INFERRED) | same as above | CONTRADICTED |
| `m-a-p/CodeFeedback-Filtered-Instruction` | PERMISSIVE_OK (INFERRED), provenance_group: independent | PERMISSIVE_OK conditional on #18; provenance_group is **not** independent, partially inherits the `nickrosh/Evol-Instruct-Code-80k-v1` conflict | CONTRADICTED (provenance_group), verdict provisionally held |
| `newfacade/LeetCodeDataset` | REFUSE-TERMS (recommended, presumptive) | **REFUSE** (closed — LeetCode ToS fetched and quoted) | confirmed, closed |
| `Nan-Do/code-search-net-python` | mirror_tag "none stated" | mirror_tag is `apache-2.0`, self-declared | minor factual correction, verdict unchanged |
| `openai/openai_humaneval` | PERMISSIVE_OK (INFERRED) | PERMISSIVE_OK (**VERIFIED**) | upgraded |
| `evalplus/humanevalplus`, `mbppplus` | PERMISSIVE_OK (INFERRED) | PERMISSIVE_OK (**VERIFIED**) | upgraded |
| `SWE-bench/SWE-smith` | PERMISSIVE_OK (INFERRED) | PERMISSIVE_OK (**VERIFIED**) | upgraded |
| `google-research-datasets/mbpp` | ATTRIBUTION (INFERRED) | ATTRIBUTION (**VERIFIED**, but the operative source is the HF card, not the GitHub tree — the GitHub monorepo's own blanket licence is Apache-2.0 and its mbpp subdirectory states no licence) | upgraded, with correction |
| everything else surveyed | as stated | VERIFIED-AGREES or UNVERIFIABLE-beyond-tag (matching survey's own caveat) | no change |

**Net effect on the top-8 table:** rank 3 (`bigcode/commitpackft`) should be pulled from
"confirm and move on" to "needs row-level licence filtering before use" — it is not a
zero-enrichment win as the survey's entry states. Rank 4's sibling candidate
(`m-a-p/CodeFeedback-Filtered-Instruction`, not itself in the top 8 but named as filling a
scope gap) carries more provenance risk than "independent lineage" implied. Ranks 1, 2, 6,
7 (`code_search_net`, `starcoderdata`, `code_contests`, `apps`) are unaffected and now rest
on primary sources fetched twice (survey + this pass) rather than once. Rank 8
(`SWE-bench/SWE-smith`) moves from INFERRED to VERIFIED with no other change.

---

## Evidence saved this pass

All under `/akula-data/session-backup-staging/dataset-factory/licence-texts/`:

- `leetcode-terms-wayback-20260805.html` — LeetCode ToS, Wayback Machine snapshot
- `newfacade-LeetCodeDataset-github-README-20260903.md`
- `card-bigcode-commitpackft-20260903.md`, `card-bigcode-commitpack-20260903.md`
- `card-mapCodeFeedback-20260903.md`
- `card-HuggingFaceTB-stack-edu-20260903.md`
- `card-google-research-datasets-mbpp-20260903.md`, `google-research-mbpp-subdir-readme-20260903.md`
- `deepmind-code_contests-readme-20260903.md`, `deepmind-code_contests-github-LICENSE-20260903.txt`
- `codesearchnet-github-LICENSE-20260903.txt`
- `hendrycks-apps-github-LICENSE-20260903.txt`
- `openai-humaneval-github-LICENSE-20260903.txt`
- `evalplus-github-LICENSE-20260903.txt`
- `nickrosh-EvolTeacher-github-LICENSE-20260903.txt`
- `princeton-nlp-SWE-bench-github-LICENSE-20260903.txt`
- `SWE-bench-SWE-smith-github-LICENSE-20260903.txt`

All fetches dated **2026-09-03** except the LeetCode ToS (Wayback snapshot dated
2026-08-05, the closest available capture; the live page itself was inaccessible to
automated fetch — 403 — on 2026-09-03).
