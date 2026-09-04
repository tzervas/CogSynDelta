# Dataset factory — CODE faculty survey (`language_code`)

Surveyor pass, 2026-09-03. Method per task: (1) HF Hub public API
(`/api/datasets?search=`, `/api/datasets/<id>` for `cardData.license`, `datasets-server.huggingface.co/size`),
(2) upstream/primary source (GitHub LICENSE files, project pages) fetched with WebFetch,
(3) WebSearch for well-known sets. **No dataset content was downloaded** — only API JSON,
cards, and licence pages. Raw API responses cached at
`/tmp/claude-1000/.../scratchpad/dataset-factory/*.json` (session-local, not preserved).

Catalogue shape follows `00-ground.md` §(b). `provenance_group` is populated even though the
current `Dataset` dataclass doesn't carry the field yet (per ground doc, this is a factory
design intent, not yet shipped code).

Verdict classes: `PERMISSIVE_OK / ATTRIBUTION / SHARE_ALIKE / NC / BLOCKING / CONSENT_OPEN /
UNVERIFIED`, plus usage_tag `EVAL-ONLY` where applicable. All fetches below are dated
**2026-09-03**.

---

## Provenance groups (shared lineage, for B1/B2 purposes)

- **PG-STACK** (BigCode/Stack lineage): `the-stack-v2-dedup`, `the-stack-dedup`, `starcoderdata`,
  `the-stack-smol`, `stack-edu`, `opc-annealing-corpus` (built from Stack v2's algorithmic
  subset). Six catalogue rows, **one provenance group**.
- **PG-CSN** (CodeSearchNet lineage): `code_search_net`, `Nan-Do/code-search-net-python`.
- **PG-COMPETITIVE** (competitive-programming problem lineage — LeetCode/Codeforces/APPS/
  CodeContests problem text recurs across all of these even when the wrapping dataset differs):
  `codeparrot/apps`, `deepmind/code_contests`, `nvidia/OpenCodeReasoning`,
  `nvidia/OpenCodeReasoning-2`, `KodCode/KodCode-V1`, `newfacade/LeetCodeDataset`.
- **PG-COMMITPACK** (bigcode CommitPack project): `commitpackft`, `commitpack`.
- **PG-EVALSTD** (standard code-gen eval family, HumanEval/MBPP and their extensions):
  `openai_humaneval`, `evalplus/humanevalplus`, `google-research-datasets/mbpp`,
  `evalplus/mbppplus`, `zai-org/humaneval-x`, `bigcode/humanevalpack`. All EVAL-ONLY.
- **PG-SWEBENCH**: `SWE-bench/SWE-smith`, `princeton-nlp/SWE-bench`, `SWE-bench/SWE-bench_Verified`.
- Independent (own lineage): `neulab/conala`, `m-a-p/CodeFeedback-Filtered-Instruction`,
  `nickrosh/Evol-Instruct-Code-80k-v1`, `OpenCoder-LLM/opc-fineweb-code-corpus`.

---

## Catalogue entries

### 1. `bigcode/the-stack-v2-dedup`
- upstream: Software Heritage archive, curated by BigCode project (github.com/bigcode-project)
- mirror URL: https://huggingface.co/datasets/bigcode/the-stack-v2-dedup (gated; API size probe → HTTP 401, requires accepted-terms auth)
- mirror_tag: `license:other`
- licence_upstream (VERIFIED, fetched https://www.bigcode-project.org/docs/pages/bigcode-openrail/ 2026-09-03): BigCode OpenRAIL-M. Quote: commercial use and training derivative models are **permitted**; redistribution is allowed "provided you include a similar set of restrictions... accomplishing the same purpose"; Attachment A forbids specific *output* uses (malware, disinformation, undisclosed impersonation) — these bind users of models trained on it, not the dataset itself. Opt-out mechanism exists (per-repo removal requests to BigCode).
- grant_scope: whole_corpus (content), but access itself is gated (must accept BigCode's click-through)
- provenance_group: PG-STACK
- verdict: **SHARE_ALIKE**-adjacent, flagged non-standard — OpenRAIL-M is not CC BY-SA but does force a *behavioural*-restriction clause to propagate downstream, which the ground doc's five-class scheme doesn't cleanly capture. Recommend treating like SHARE_ALIKE for strictest-input purposes (propagates) rather than PERMISSIVE_OK. Not REFUSE: training and redistribution are both explicitly permitted.
- size: ~900B tokens class corpus (not directly probed here; API size endpoint blocked by gate)
- quality: near-dedup at file/near-dup level, per-file licence detection at ingestion (BigCode PII/opt-out pipeline), massively multi-language
- consume/emit fit: primary multi-language replacement for the single-source CodeSearchNet-Python corpus the faculty currently trains on — directly answers the B1/B2 FAIL (100% single-source) noted in `00-ground.md`
- enrichment: language-stratified sampling to the CORPUS-CONTRACT §1.1 cap table; licence consequence: **the OpenRAIL-M behavioural clause propagates to any derived sample**, so the language_code region (and by strictest-input, the composed model) would carry it alongside the existing NC tier from `memory`

### 2. `bigcode/starcoderdata`
- upstream: BigCode project, curated permissive-only subset of the-stack-dedup
- mirror URL: https://huggingface.co/datasets/bigcode/starcoderdata
- mirror_tag: `license:other`
- licence_upstream: same BigCode OpenRAIL-M as #1 (VERIFIED, same source fetch)
- grant_scope: whole_corpus
- provenance_group: PG-STACK (same lineage as #1 — **do not double-count as an independent source**)
- verdict: SHARE_ALIKE-adjacent (see #1)
- size: 783GB, 86 languages (widely cited figure from the StarCoder paper; not independently re-measured here — INFERRED from public reporting, flag for a follow-up VERIFIED size pull)
- quality: pre-filtered to already-permissive per-file licences (MIT/Apache/BSD-family) at ingestion — this is StarCoder's training corpus, well-curated, near-dedup, comment-density and star-count filtered
- consume/emit fit: best single replacement candidate for the whole language_code corpus at scale — already licence-filtered to permissive-only *at the file level*, reducing the OpenRAIL-M concern to the wrapper/access licence rather than per-file content licence
- enrichment: language cap sampling per CORPUS-CONTRACT §1.1; pair with docstring extraction for code-comment pairs (see CodeSearchNet below) rather than raw file dumps, to fix the current truncation-past-max_len defect

### 3. `bigcode/the-stack-dedup`
- upstream/mirror/licence: identical situation to #1 (v1 of the Stack, near-deduplicated)
- provenance_group: PG-STACK
- verdict: SHARE_ALIKE-adjacent
- size: ~3TB (public reporting; not independently measured)
- note: superseded by v2 for new work; keep as fallback only if v2 gate access is unavailable

### 4. `HuggingFaceTB/stack-edu`
- upstream: HuggingFaceTB, filtered from The Stack v2 by an educational-value classifier
- mirror URL: https://huggingface.co/datasets/HuggingFaceTB/stack-edu
- mirror_tag: none in cardData (empty license field)
- licence_upstream (VERIFIED, card text fetched 2026-09-03): card explicitly says "Please refer to the-stack-v2 for the data license" — this is **metadata_only at rest**: "This dataset only contains the SWHIDs to download the code files and not the content of the files itself."
- grant_scope: **metadata_only** — the HF repo ships identifiers, not code text; actual content must be pulled from Software Heritage under the Stack v2 terms
- provenance_group: PG-STACK
- verdict: SHARE_ALIKE-adjacent (inherits Stack v2), UNVERIFIED-as-a-standalone-grant (no independent licence statement, defers entirely)
- size: 167,063,359 rows / 17.47GB **of index metadata**, described as a 125B-token educational-code corpus once resolved against SWH (VERIFIED count: `num_rows: 167063359`, HF datasets-server, 2026-09-03)
- quality: highest-quality-per-token candidate in PG-STACK — an educational-value classifier already filters for well-commented, pedagogically clean code, which is exactly the code-comment-pair-rich material the faculty needs
- consume/emit fit: strong candidate for the code-comment/docstring-pair emphasis, but implementation cost is real (SWH resolution pipeline, not a flat download) — flag as high-value/high-integration-cost
- enrichment: none needed beyond SWH resolution + the same language caps; licence consequence identical to #1/#2

### 5. `code-search-net/code_search_net`
- upstream: github/CodeSearchNet (GitHub's original CodeSearchNet Challenge corpus)
- mirror URL: https://huggingface.co/datasets/code-search-net/code_search_net
- mirror_tag: `license:other` — **mismatch flagged**
- licence_upstream (VERIFIED, fetched https://github.com/github/CodeSearchNet/blob/master/LICENSE 2026-09-03): MIT License, copyright GitHub 2019. Quote: "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the 'Software'), to deal in the Software without restriction..."
- grant_scope: whole_corpus
- provenance_group: PG-CSN
- verdict: **PERMISSIVE_OK** — this is a mirror-lies case exactly matching the pattern in `dataset-mirror-licences-lie.md`: the HF card tags `other` while the primary GitHub LICENSE is plain MIT. Recorded verbatim per the provenance rule.
- size: 4,141,072 rows / 3.93GB (VERIFIED, datasets-server size endpoint, 2026-09-03)
- quality: the classic code/docstring-pair corpus (6 languages: Go, Java, JS, PHP, Python, Ruby), function-level granularity, docstring-derived queries — exactly the code-comment/docstring-pair shape the survey scope calls for
- consume/emit fit: direct fit for docstring-pair training; this is also the dataset the faculty **already partly trains on today** via the `Nan-Do/code-search-net-python` derivative, so admitting the full multi-language original directly addresses the current 100%-single-source failure while staying in the same lineage the region already trusts
- enrichment: none required — already query/code paired; multi-language split gives immediate B1/B2 relief if paired with a non-CSN second source (e.g. StarCoderData) so PG-CSN isn't the sole provenance group either

### 6. `Nan-Do/code-search-net-python`
- upstream: derivative of #5, Python-only, reformatted for instruction-style training
- mirror_tag: none stated in the earlier search pass; treat as inheriting #5's licence
- provenance_group: PG-CSN — **same group as #5, do not count as a second source**
- verdict: PERMISSIVE_OK (inherited)
- note: this is the dataset the region **already consumes today** (per `00-ground.md` row 1) — the single-source B1/B2 failure is precisely because this and CodeSearchNet-lineage-adjacent `apps`/`code_contests` are the only sources in the mix

### 7. `codeparrot/apps` (APPS — Automated Programming Progress Standard)
- upstream: Hendrycks et al., github.com/hendrycks/apps
- mirror URL: https://huggingface.co/datasets/codeparrot/apps
- mirror_tag: `license:mit`
- licence_upstream (VERIFIED, fetched github.com/hendrycks/apps 2026-09-03): repository footer states MIT license
- grant_scope: whole_corpus
- provenance_group: PG-COMPETITIVE
- verdict: **PERMISSIVE_OK** (mirror and upstream agree)
- size: not independently re-probed (datasets-server returned HTTP 501 for this repo — script-based loader, not parquet-native); already on disk locally per `00-ground.md` (18G dir shared with code_contests)
- quality: 10,000 coding problems with test cases and reference solutions, difficulty-tiered (introductory/interview/competition) — this is the faculty's **existing** source
- consume/emit fit: already consumed; retained as the reserve's executable-ground-truth source per `00-ground.md` §(d) row 5
- enrichment: n/a, already integrated — flag only that it shares PG-COMPETITIVE lineage with `code_contests`, so the two together still count as fewer independent sources than they look like

### 8. `deepmind/code_contests`
- upstream: google-deepmind/code_contests (GitHub)
- mirror URL: https://huggingface.co/datasets/deepmind/code_contests
- mirror_tag: `license:cc-by-4.0`
- licence_upstream (VERIFIED, fetched raw LICENSE + README from github.com/deepmind/code_contests 2026-09-03): dual — "The code is licensed under the Apache 2.0 License"; "All non-code materials provided are made available under the terms of the CC BY 4.0 license." Third-party problem statements retain their original platform licences (acknowledged, not itself a grant).
- grant_scope: whole_corpus for DeepMind's own packaging; underlying per-problem statements carry platform-specific terms (Codeforces et al.) that DeepMind did not itself grant
- provenance_group: PG-COMPETITIVE
- verdict: **ATTRIBUTION** (CC BY 4.0 confirmed at both mirror and upstream) — already the region's current second source
- size: 4,044 rows at top level per the size endpoint (this counts problems, not the much larger per-problem test-case/solution expansion actually on disk — 18G shared dir per ground doc)
- quality: already consumed; well-known, test-case-verified, multi-difficulty
- consume/emit fit: already integrated
- enrichment: n/a

### 9. `bigcode/commitpackft`
- upstream: bigcode-project/commitpack (GitHub), filtered subset (high-quality commit messages, "FT" = fine-tuning-ready)
- mirror URL: https://huggingface.co/datasets/bigcode/commitpackft
- mirror_tag: `license:mit`
- licence_upstream: INFERRED to match (not independently re-fetched from bigcode-project's own repo LICENSE this pass — the mirror tag is MIT and the project is BigCode's own permissively-licensed tooling release, distinct from the Stack's OpenRAIL-M gate, since CommitPack is BigCode's own curated/generated wrapper rather than a re-release of third-party file content) — **flag for a follow-up VERIFIED fetch before training** given BigCode's mixed track record between OpenRAIL-M (content) and MIT (tooling/derived) releases in this survey
- grant_scope: whole_corpus (claimed)
- provenance_group: PG-COMMITPACK
- verdict: PERMISSIVE_OK (INFERRED pending the follow-up fetch above — do not treat as VERIFIED)
- size: not independently probed (datasets-server 500 for this loader)
- quality: filtered from CommitPack for message quality (length, informativeness) — this is exactly the "commit messages" shape the survey scope calls for; paired code diff + natural-language message
- consume/emit fit: directly fills the commit-message gap in the survey scope; also usable as a second code-comment-pair-shaped source distinct from CodeSearchNet's docstring pairing
- enrichment: none required, already paired; licence consequence pending the MIT-vs-OpenRAIL-M confirmation above

### 10. `bigcode/commitpack`
- upstream: same BigCode CommitPack project, full unfiltered version
- provenance_group: PG-COMMITPACK — **same group as #9**
- verdict: PERMISSIVE_OK (INFERRED, same caveat as #9)
- size: not probed (loader 501)
- note: prefer #9 (commitpackft) for training — it's the quality-filtered subset; keep this row only as the provenance note for the group

### 11. `OpenCoder-LLM/opc-annealing-corpus`
- upstream: OpenCoder-LLM (INFResearch/Multimodal Art Projection + collaborators)
- mirror URL: https://huggingface.co/datasets/OpenCoder-LLM/opc-annealing-corpus
- mirror_tag: `license:odc-by`
- licence_upstream (VERIFIED, card text fetched 2026-09-03): card states the corpus is "Algorithmic corpus sampled from The Stack v2" plus synthetic rewrites/QA derived from that same algorithmic corpus
- grant_scope: whole_corpus **claimed** as ODC-BY, but the base material is admittedly sampled from Stack v2 (OpenRAIL-M territory) — **provenance conflict flagged**: an ODC-BY tag sitting on top of OpenRAIL-M-sourced content is exactly the kind of mismatch the provenance rule exists to catch. Not resolved this pass; needs the OpenCoder technical report cross-checked against BigCode's opt-out list before trusting the ODC-BY tag as the operative one.
- provenance_group: PG-STACK (by underlying content) — **not** independent of #1/#2 despite the different mirror tag
- verdict: **UNVERIFIED** (tag conflict unresolved) — do not admit as an independent PERMISSIVE_OK source; if admitted, treat as PG-STACK for B1/B2 and inherit the SHARE_ALIKE-adjacent verdict from #1 until the conflict is resolved
- size: 11,643,084 rows (VERIFIED, datasets-server 2026-09-03)
- quality: annealing-stage corpus (base pretrain + synthetic rewrite + synthetic QA) — useful shape for late-stage curriculum, not for provenance-of-record

### 12. `OpenCoder-LLM/opc-fineweb-code-corpus`
- upstream: same OpenCoder-LLM project, web-crawled code-adjacent text (FineWeb-style filtering pass)
- mirror_tag: `license:mit`
- licence_upstream: INFERRED to match (OpenCoder's own release page states MIT for the SFT/pretrain corpora it authored; not independently re-fetched this pass — flag for follow-up, same caveat class as #9)
- grant_scope: whole_corpus (claimed)
- provenance_group: independent (web crawl, not Stack-derived)
- verdict: PERMISSIVE_OK (INFERRED)
- size: 100,920,235 rows / 147.9GB original (VERIFIED, datasets-server 2026-09-03)
- quality: code-adjacent prose (docs, tutorials, forum threads) filtered from CommonCrawl — this is the shape the survey scope calls "code-adjacent text," and a candidate for the still-PLACEHOLDER `language` trunk's docs curriculum too
- consume/emit fit: strongest single candidate for the "code-adjacent text in many languages" scope item; large enough to matter at the 1e10-token target scale
- enrichment: dedup against StarCoderData/Stack-lineage rows before mixing, since crawled docs commonly duplicate repo READMEs already present in PG-STACK

### 13. `nvidia/OpenCodeReasoning`
- upstream: NVIDIA, github/huggingface release
- mirror_tag: `license:cc-by-4.0`
- licence_upstream (VERIFIED, card fetched 2026-09-03): card states CC BY 4.0 for NVIDIA's own release, but explicitly disclaims responsibility for underlying source licences: "users are responsible for checking if the dataset license is fit for the intended purpose" for each source (TACO, APPS, CodeContests train split, open-r1/codeforces train split, direct Codeforces scraping across 10 platforms)
- grant_scope: **whole_corpus for NVIDIA's wrapper only** — the underlying problem statements are NOT independently re-licensed by NVIDIA; this is a case the ground doc's `grant_scope` field exists specifically to catch
- provenance_group: PG-COMPETITIVE
- verdict: **UNVERIFIED at the problem-statement layer** even though the wrapper tag is ATTRIBUTION — treat the reasoning-trace/solution layer as CC BY 4.0 (NVIDIA's own generated content) but do not treat the embedded problem text as independently cleared; direct Codeforces scraping in particular has an unresolved ToS question (Codeforces problem statements are not CC-licensed by the platform itself)
- size: 337,766 rows (VERIFIED, 2026-09-03)
- quality: reasoning traces generated over real competitive-programming problems, large and recent (2025-era), high download/like counts signal community trust
- consume/emit fit: strong for the `reasoning` faculty (not `language_code` directly) — flagging here because it's the best-quality candidate the survey turned up for code+reasoning combined, worth a cross-faculty note
- enrichment: if used for `language_code`, strip to NVIDIA's own generated reasoning/solution text and treat problem statements as attribution-only pass-through, consistent with strictest-input

### 14. `nvidia/OpenCodeReasoning-2`
- same project, second release, competitive-programming C++/Python pairs
- mirror_tag: `license:cc-by-4.0`; same caveat structure as #13
- provenance_group: PG-COMPETITIVE
- verdict: UNVERIFIED at problem-statement layer, ATTRIBUTION at wrapper layer (same reasoning as #13)
- size: 220,000 rows (VERIFIED, 2026-09-03)

### 15. `KodCode/KodCode-V1`
- upstream: KodCode-AI (github.com/KodCode-AI/kodcode)
- mirror_tag: `license:cc-by-nc-4.0`
- licence_upstream (VERIFIED, card fetched 2026-09-03): CC BY-NC 4.0 confirmed at the mirror; card states the corpus is "fully-synthetic," solutions/tests generated by `gpt-4o-0513`
- grant_scope: whole_corpus, but content is **model-generated** (OpenAI GPT-4o outputs) — the operator stance doesn't address OpenAI-output-licensing directly, but this is a known open question (OpenAI's usage policies have historically restricted using API outputs to train competing models); flagged as a provenance_red_flag, not resolved
- provenance_group: PG-COMPETITIVE (questions/style draw on LeetCode/Codeforces/CodeContests per the card)
- verdict: **NC** (per DEC-31, does not block; moves the region to the NC tier the faculty is already carrying via `memory`)
- provenance_red_flags: "synthetic content generated via OpenAI API; OpenAI's terms of use have historically restricted training competing models on API outputs — not addressed by the CC BY-NC-4.0 tag, which covers KodCode's own copyright claim, not any OpenAI contractual restriction on the underlying generation"
- size: 487,432 rows / 2.64GB (VERIFIED, 2026-09-03)
- quality: large (12 subsets), tests are executable (self-verifying via generated unit tests) — strong quality signal for the reserve's executable-ground-truth need
- consume/emit fit: NC tier is already accepted policy (memory faculty); usable, but the OpenAI-output caveat should ride along in the receipt
- enrichment: none required structurally; carry the red flag forward

### 16. `neulab/conala` (CoNaLa)
- upstream: Carnegie Mellon NeuLab, originally Yin & Neubig (StackOverflow-derived NL-to-code pairs)
- mirror_tag: `license:mit`
- licence_upstream: **UNVERIFIED this pass** — the CoNaLa project page (neulab.github.io/conala/) returned HTTP 404 on fetch attempt (2026-09-03); the underlying data is StackOverflow-sourced, and Stack Exchange content is CC BY-SA 4.0 by platform ToS, which would conflict with the MIT mirror tag if the mirror includes the original SO post text rather than just the code snippets NeuLab authored the pairing/annotation for
- grant_scope: **unstated** pending re-verification — flag for a retry (try github.com/neulab/conala repo LICENSE directly, or the ACL Anthology paper's data statement)
- provenance_group: independent
- verdict: **UNVERIFIED** — do not admit as PERMISSIVE_OK on the mirror tag alone given the StackOverflow-CC-BY-SA collision risk; this is exactly the mismatch pattern the provenance rule warns about
- size: 596,770 rows / 160.96MB (VERIFIED, 2026-09-03)
- quality: hand-curated NL-intent/code-snippet pairs, small but clean, classic benchmark for code generation from natural language
- consume/emit fit: good shape for code-search/NL-to-code pairs if the licence resolves to at least SHARE_ALIKE
- enrichment: n/a pending licence resolution

### 17. `m-a-p/CodeFeedback-Filtered-Instruction`
- upstream: Multimodal Art Projection (M-A-P)
- mirror_tag: `license:apache-2.0`
- licence_upstream: INFERRED to match (M-A-P's own filtered/curated release; not independently re-fetched from a non-HF primary source this pass — M-A-P publishes its own datasets directly to HF as the primary distribution point, so mirror and upstream largely coincide here, unlike third-party remirrors)
- grant_scope: whole_corpus (claimed); content includes a mix of real and model-generated code+feedback pairs — some upstream conversational data may itself derive from other instruct datasets (e.g. Magicoder, WizardCoder lineages) not independently traced this pass
- provenance_group: independent (own lineage, though not deeply traced upstream)
- verdict: PERMISSIVE_OK (INFERRED, moderate confidence)
- size: 156,526 rows / 371.2MB (VERIFIED, 2026-09-03)
- quality: filtered for code-execution-feedback loops (code + error/output + corrected code), a distinctive shape not covered by the other candidates
- consume/emit fit: adds a "code + feedback/correction" shape the current corpus lacks entirely
- enrichment: trace the Magicoder/WizardCoder-lineage question before treating as a fully independent provenance group

### 18. `nickrosh/Evol-Instruct-Code-80k-v1`
- upstream claimed: nickrosh/Evol-Teacher (GitHub)
- mirror_tag: **`license:cc-by-nc-sa-4.0`**
- licence_upstream (VERIFIED, fetched github.com/nickrosh/Evol-Teacher 2026-09-03): repository states **Apache-2.0** license — **direct mismatch** between the HF mirror tag (CC BY-NC-SA-4.0) and the GitHub primary source (Apache-2.0)
- grant_scope: unstated which is authoritative; content is GPT-3.5/4-generated ("Over 120,000 API calls were made to OpenAI to create this dataset" per the repo) — same OpenAI-output caveat as #15, unaddressed by either licence tag
- provenance_group: independent
- verdict: **UNVERIFIED** (conflicting VERIFIED licence statements at mirror vs upstream — this is a live example of "mirrors lie" in the direction of the mirror being *more* restrictive than upstream, the opposite of the usual failure mode, so resolve before either training or discarding)
- provenance_red_flags: "mirror tag CC-BY-NC-SA-4.0 vs GitHub Apache-2.0, both VERIFIED-fetched, unresolved"; "OpenAI-API-generated content, same caveat class as KodCode"
- size: 78,264 rows / 121.5MB (VERIFIED, 2026-09-03)
- quality: WizardCoder-lineage evol-instruct style, well-known, widely used
- consume/emit fit: instruction-following code pairs; useful shape but blocked on the licence conflict
- enrichment: n/a pending resolution

### 19. `newfacade/LeetCodeDataset`
- upstream: LeetCode.com problems, repackaged by newfacade (github link on the card, not independently fetched — card fetch attempt for sourcing methodology returned no usable content this pass)
- mirror_tag: `license:apache-2.0`
- licence_upstream: **UNVERIFIED / red flag** — Apache-2.0 is newfacade's own claimed licence for the *repackaging*, but LeetCode's Terms of Service explicitly restrict scraping and redistribution of problem statements (LeetCode ToS, not independently re-quoted this pass but well-documented industry-wide as a restrictive, non-redistributable EULA over problem text) — this is exactly the shape the operator stance says to REFUSE: "research-only / non-redistributable terms... distributors that disclaim owning what they distribute"
- provenance_group: PG-COMPETITIVE
- verdict: **REFUSE-TERMS (recommended)** — the wrapper's Apache-2.0 tag cannot grant rights the packager doesn't hold over LeetCode's proprietary problem text; needs an explicit LeetCode ToS re-check before this classification is finalized, but the presumption should be refuse, not admit
- size: 2,869 rows / 101.1MB (VERIFIED row/size count, 2026-09-03; licence classification is the issue, not the size)
- REFUSED pending confirmation — do not train on

### 20. `openai/openai_humaneval`
- upstream: OpenAI (github.com/openai/human-eval)
- mirror_tag: `license:mit`
- licence_upstream: INFERRED MIT match (OpenAI's own HumanEval GitHub repo is well-known MIT; this is the canonical primary source and OpenAI is both mirror and upstream author here, so the usual mirror/upstream split doesn't really apply — flag as high-confidence INFERRED rather than independently re-fetched this pass)
- grant_scope: whole_corpus
- provenance_group: PG-EVALSTD
- usage_tag: **EVAL-ONLY** — this and its siblings below are contamination-sensitive benchmarks; training on them directly poisons the faculty's own eval signal
- verdict: PERMISSIVE_OK (licence), but structurally excluded from training by usage_tag
- size: 164 rows / 83.9KB (VERIFIED, 2026-09-03)
- consume/emit fit: keep as held-out eval only, never in-mixture training rows

### 21. `evalplus/humanevalplus` / `evalplus/mbppplus`
- upstream: EvalPlus project (github.com/evalplus/evalplus), extends HumanEval/MBPP with many more test cases per problem
- mirror_tag: `license:apache-2.0` (both)
- licence_upstream: INFERRED match (EvalPlus is a known academic tooling release, standard Apache-2.0 for augmented-test-case tooling; not independently re-fetched this pass)
- provenance_group: PG-EVALSTD (same problem lineage as #20/#22, inherits the eval-contamination concern)
- usage_tag: **EVAL-ONLY**
- verdict: PERMISSIVE_OK, EVAL-ONLY
- size: humanevalplus 164 rows/2.9MB; mbppplus 378 rows/1.1MB (VERIFIED, 2026-09-03)

### 22. `google-research-datasets/mbpp`
- upstream: Google Research (github.com/google-research/google-research/tree/master/mbpp)
- mirror_tag: `license:cc-by-4.0`
- licence_upstream: INFERRED match (Google's own canonical release; CC BY 4.0 is the well-documented standard licence for this benchmark — not independently re-fetched from the GitHub tree this pass, flag for confirmation before training use)
- provenance_group: PG-EVALSTD
- usage_tag: **EVAL-ONLY**
- verdict: ATTRIBUTION, EVAL-ONLY
- size: 1,401 rows / 351KB (VERIFIED, 2026-09-03)

### 23. `zai-org/humaneval-x` and `bigcode/humanevalpack`
- multilingual extensions of HumanEval (C++, Java, JS, Go, Python, Rust and others)
- mirror_tags: `license:apache-2.0` (humaneval-x), `license:mit` (humanevalpack)
- licence_upstream: INFERRED match for both (standard academic tooling releases from their respective labs; not independently re-fetched)
- provenance_group: PG-EVALSTD
- usage_tag: **EVAL-ONLY**
- verdict: PERMISSIVE_OK, EVAL-ONLY
- size: humanevalpack 984 rows/1.16MB (VERIFIED, 2026-09-03); humaneval-x size probe failed (HTTP 500 on datasets-server, loader issue not a licence issue)

### 24. `SWE-bench/SWE-smith`
- upstream: princeton-nlp / SWE-bench project (github.com/SWE-bench)
- mirror_tag: `license:mit`
- licence_upstream: INFERRED match (SWE-bench project publishes MIT consistently across its repos; not independently re-fetched this pass)
- provenance_group: PG-SWEBENCH
- verdict: PERMISSIVE_OK (INFERRED)
- size: 59,136 rows / 277.8MB (VERIFIED, 2026-09-03)
- quality: synthetically-generated SWE-bench-style training tasks (real GitHub issue/PR/patch triples at scale) — genuinely a **training** set, not held-out eval, unlike its siblings
- consume/emit fit: closest thing in this survey to real-world multi-file code-editing supervision; a materially different shape (repo-level, not function-level) from every other candidate here
- enrichment: none required structurally; large enough and recent enough (2025) to be worth a dedicated follow-up pass

### 25. `princeton-nlp/SWE-bench` / `SWE-bench/SWE-bench_Verified`
- mirror_tag: **none** (empty cardData license field for both)
- licence_upstream: **UNVERIFIED** — no licence tag at all on either mirror; the underlying task instances are drawn from real-world GitHub issues/PRs across many upstream repos, each carrying that repo's own licence, and SWE-bench itself does not appear to assert a blanket licence over the aggregation
- provenance_group: PG-SWEBENCH
- usage_tag: **EVAL-ONLY** regardless of licence outcome — this is THE standard held-out coding-agent benchmark; training on it is a contamination hazard independent of the licence question
- verdict: UNVERIFIED, EVAL-ONLY
- size: not probed (benchmark-shaped, eval-only, licence blocks it from training consideration regardless)

---

## REFUSED (and why)

- **`newfacade/LeetCodeDataset`** — presumptive REFUSE-TERMS: LeetCode's ToS is a well-documented
  non-redistributable/no-scraping EULA over problem statements; the packager's own Apache-2.0
  tag cannot grant rights it doesn't hold. Needs a formal LeetCode ToS re-fetch to convert this
  from "presumptive" to a closed verdict, but the survey's presumption is refuse.
- **`nickrosh/Evol-Instruct-Code-80k-v1`** — not outright refused, but held at UNVERIFIED:
  a genuine mirror-vs-upstream licence conflict (CC-BY-NC-SA-4.0 mirror vs Apache-2.0 GitHub,
  both VERIFIED-fetched) must resolve before either training or discarding it; flagging rather
  than guessing.
- **`neulab/conala`** — held at UNVERIFIED, not refused outright: MIT mirror tag against a
  StackOverflow-sourced corpus (Stack Exchange content is CC BY-SA 4.0 by platform ToS) is a
  plausible collision the survey couldn't close this pass (project page 404'd).
- **`OpenCoder-LLM/opc-annealing-corpus`** — held at UNVERIFIED: an ODC-BY tag over content the
  dataset's own card says is "sampled from The Stack v2" is a provenance conflict, not a clean
  independent grant; do not count it as separate from PG-STACK.
- **Direct scraping of live Codeforces/AtCoder/HackerRank problem pages** — not surveyed as a
  standalone HF dataset (out of scope for this pass, which surveys existing HF-hosted or
  well-known packaged sets), but flagged here because several PG-COMPETITIVE candidates
  (`OpenCodeReasoning`) admit to exactly this sourcing method for a fraction of their rows —
  same REFUSE-TERMS logic as LeetCode applies to any of those platforms' own problem text,
  independent of the wrapper dataset's stated licence.
- Nothing else in the searched pool was refused outright; several rows above are UNVERIFIED
  rather than REFUSE and need a second fetch pass before either admission or refusal.

**Official documentation corpora** (survey scope also named these): per `00-ground.md`, an
"official-docs" corpus is already staged locally (Python/Rust docs) as a **deferred** candidate
for the PLACEHOLDER `language` trunk faculty, not `language_code` — out of scope for this pass;
noting its existence per the ground doc rather than re-surveying it here.

---

## Top 8 candidates with verdicts (summary)

| rank | id | verdict | why it's top-8 |
|---|---|---|---|
| 1 | `code-search-net/code_search_net` | **PERMISSIVE_OK** (VERIFIED, mirror mismatch resolved in the corpus's favor — MIT at upstream) | Directly fixes the region's current single-source failure while staying in the lineage it already trusts; full multi-language version vs. the Python-only slice in use today; code-comment-pair-native, matching the survey's docstring-pair scope item exactly |
| 2 | `bigcode/starcoderdata` | SHARE_ALIKE-adjacent (OpenRAIL-M, VERIFIED — permits training/redistribution, forces behavioural-restriction propagation) | Best-quality, largest-scale, already-permissive-filtered-at-file-level multi-language corpus; the single highest-leverage fix for the 1e10-token scale-path gap named in `00-ground.md` §(d) |
| 3 | `bigcode/commitpackft` | PERMISSIVE_OK (INFERRED, needs one follow-up VERIFIED fetch) | Fills the commit-message gap in the survey scope outright; already paired diff+message |
| 4 | `OpenCoder-LLM/opc-fineweb-code-corpus` | PERMISSIVE_OK (INFERRED) | Best candidate for "code-adjacent text," large enough to matter at scale, independent provenance from PG-STACK/PG-CSN |
| 5 | `KodCode/KodCode-V1` | NC (VERIFIED CC-BY-NC-4.0, accepted per DEC-31) | Large, executable/self-verifying (generated unit tests), directly usable at the NC tier the faculty already carries |
| 6 | `deepmind/code_contests` | ATTRIBUTION (VERIFIED, already integrated) | Confirms the region's existing second source is clean; anchors PG-COMPETITIVE |
| 7 | `codeparrot/apps` | PERMISSIVE_OK (VERIFIED, already integrated) | Confirms the region's existing primary source is clean; reserve's executable-ground-truth anchor |
| 8 | `SWE-bench/SWE-smith` | PERMISSIVE_OK (INFERRED, needs follow-up VERIFIED fetch) | Only training-shaped (not eval-only) repo-level multi-file editing corpus found this pass — a genuinely different shape from every function-level candidate above it |

**Notable exclusions from top-8 despite quality:** `bigcode/the-stack-v2-dedup` (rank-1 by raw
scale, held below starcoderdata because it requires the BigCode gated-access click-through and
carries unfiltered per-file licence noise that starcoderdata has already cleaned);
`HuggingFaceTB/stack-edu` (highest per-token quality signal in the whole survey, held out only
for its metadata_only/SWH-resolution integration cost, not its licence).
