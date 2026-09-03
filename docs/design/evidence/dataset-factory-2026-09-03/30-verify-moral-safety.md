# Adversarial verification: 10-survey-moral-safety.md (DEC-59)

Scope: every catalogued candidate in `10-survey-moral-safety.md` carries a verdict other
than REFUSE (all 20 do — the survey's "REFUSED / not admitted" section names categories,
not individually catalogued entries, so nothing in this file's scope was structurally
exempt). For each, the PRIMARY licence source was fetched independently in this pass —
GitHub LICENSE files (preferentially via `api.github.com/repos/<org>/<repo>/license`,
which resolves the actual file GitHub's own license-detection algorithm finds, sidestepping
the raw-path 404s the original survey hit), project/paper pages, and — where GitHub's API
was silent — the HF card's own README prose, read separately from its YAML `license:` tag,
since a tag and its own card's prose can disagree (see #15).

No dataset content was downloaded. Fetch date for everything below: **2026-09-03**.

Legend: **VERIFIED-AGREES** (primary text matches the surveyor's quote and verdict, in
some cases upgrading a "provisional"/"mirror only" flag to a firmer VERIFIED because a
primary or independent second surface was reached this pass) · **CONTRADICTED** (primary
text disagrees with the surveyor's quote or verdict, or the survey overlooked a
primary-source signal that changes the licence picture) · **UNVERIFIABLE** (the primary
surface could not be reached in this pass; states what was missing).

---

## 1. `hendrycks/ethics` — ETHICS

**VERIFIED-AGREES.** Re-fetched `raw.githubusercontent.com/hendrycks/ethics/master/LICENSE`
directly: full MIT text, matches the survey's quote verbatim (permission grant + "AS IS"
disclaimer). HF mirror tag independently re-checked: `license: mit`. No carve-out language
anywhere. Verdict PERMISSIVE_OK stands.

## 2. `Anthropic/hh-rlhf` (+ red-team-attempts) — HH-RLHF

**VERIFIED-AGREES, upgraded.** The survey's raw-path fetch to `LICENSE.txt` 404'd and it
fell back to a repo-badge reading, self-flagging as "VERIFIED, thin." This pass used
`api.github.com/repos/anthropics/hh-rlhf/license` (GitHub's own resolved-license endpoint,
not a guessed path) and got the full MIT text back: *"MIT License / Copyright (c) 2022
Anthropic / Permission is hereby granted, free of charge... THE SOFTWARE IS PROVIDED 'AS
IS'..."* This closes the survey's own "recommend one more direct fetch" note — the licence
is now primary-verified in full, not just corroborated by badge + mirror tag. Verdict
PERMISSIVE_OK stands, confidence raised from "thin" to solid.

## 3. `wassname/social_chemistry_101` — Social Chemistry 101

**VERIFIED-AGREES.** Re-fetched the GitHub README directly: *"The dataset is licensed
under the CC BY-SA 4.0 license"* — verbatim match. HF mirror tag `cc-by-sa-4.0` confirmed.
Checked the AI2/Forbes project page (`maxwellforbes.com/social-chemistry/`, the redirect
target of the URL a canonical-org search turns up) as a would-be second surface: it hosts
the download and code links but states no licence text itself — this neither contradicts
nor adds to the GitHub confirmation, so it isn't treated as a second independent surface,
just as absent. Verdict SHARE_ALIKE stands; SA-propagation reasoning holds.

## 4. `metaeval/scruples` — Scruples

**VERIFIED-AGREES.** Re-fetched `raw.githubusercontent.com/allenai/scruples/master/LICENSE`:
full Apache License 2.0 text, "Copyright 2020 Allen Institute for Artificial Intelligence."
Matches the survey's quote and reasoning about AI2's whole-repo licensing pattern. Verdict
PERMISSIVE_OK stands. The survey's Reddit/AITA-ToS caveat (AI2's Apache-2.0 grant is the
operative redistribution basis, not Reddit's ToS) is sound reasoning and not contradicted
by anything found here.

## 5. `PKU-Alignment/PKU-SafeRLHF`

**VERIFIED-AGREES.** Re-fetched the HF card directly: license field `cc-by-nc-4.0`,
confirmed as the org's own first-party listing. Independently re-fetched
`api.github.com/repos/PKU-Alignment/safe-rlhf/license`: Apache License 2.0, matching the
survey's `code_only` grant-scope claim. The code/data split the survey recorded
(`whole_corpus`: CC-BY-NC-4.0 for data, `code_only`: Apache-2.0 for the training/eval repo)
is confirmed at both primary surfaces independently. Verdict NC stands.

## 6. `PKU-Alignment/BeaverTails`

**CONTRADICTED — on the survey's verification-status characterization, not on its verdict.**
The survey marked this "UNVERIFIED at primary" and treated the NC verdict as "provisional
(mirror tag + same-org pattern)." Two things this pass found change that:

- The project page (`sites.google.com/view/pku-beavertails`) genuinely has no licence
  statement — the survey was right about that surface being silent.
- But the **HF card's own prose** (not just its YAML tag, which is all the survey quoted)
  states explicitly, first-party: *"BeaverTails dataset and its family are released under
  the CC BY-NC 4.0 License."* That is a direct first-party textual grant, the same class of
  evidence the survey accepted as VERIFIED for PKU-SafeRLHF (#5) and Social Bias Frames (#12).
  The survey had this card open (it quoted the YAML tag from it) but didn't quote this
  sentence or credit it as primary-adequate.
- Additionally, `api.github.com/repos/PKU-Alignment/BeaverTails/license` returns **Apache
  License 2.0** for the repo — a code-only grant, unrecorded by the survey, that completes
  the same `whole_corpus`(NC)/`code_only`(Apache-2.0) split already established for the
  sibling PKU-SafeRLHF (#5), reinforcing rather than undermining the NC read.

Net: the **NC verdict is correct and should no longer be flagged "provisional."** The
survey under-verified an entry it had the primary evidence in front of it for. Recommend
updating #6's status from "flag for a follow-up primary-source pass" to VERIFIED, and
adding the missed `code_only` Apache-2.0 line to match #5's grant_scope shape.

## 7. `allenai/wildguardmix` / `allenai/wildjailbreak` — WildGuard / WildJailbreak

**VERIFIED-AGREES.** `raw.githubusercontent.com/allenai/wildguard/main/LICENSE.md`:
Apache-2.0 header confirmed, "Copyright 2024," "Allen Institute for AI" — matches. Both HF
cards re-checked independently: `wildguardmix` → `license: odc-by`; `wildjailbreak` →
`license: odc-by`. The survey's `code_only`(Apache-2.0)/`whole_corpus`(ODC-BY) split holds
for both datasets in the pair, not just the one the survey quoted from. Verdict
PERMISSIVE_OK-adjacent (ODC-BY) stands.

## 8. `google/civil_comments`

**VERIFIED-AGREES.** Re-fetched the TensorFlow Datasets catalog page directly: *"This
dataset is released under CC0, as is the underlying comment text"* — verbatim match, and
this is a Google-maintained surface independent of the HF mirror, exactly the
cross-surface check the survey performed. HF mirror tag `cc0-1.0` re-confirmed
independently. Verdict PERMISSIVE_OK stands.

## 9. `toxigen/toxigen-data` — ToxiGen

**VERIFIED-AGREES — conflict independently reproduced.** Re-fetched
`raw.githubusercontent.com/microsoft/TOXIGEN/master/LICENSE.txt`: confirmed dual grant, MIT
(code) + CDLA-Permissive-2.0 (data), with the CDLA text's "no restriction... with respect
to the use, modification, or sharing of Results" language matching the survey's quote.
Re-fetched the README separately: confirmed verbatim, *"The data, methods and two trained
hatespeech detection checkpoints released with this work are intended to be used for
research purposes only."* This is a real, independently-reproduced conflict between an
unrestricted data licence grant and a narrower stated-intent sentence. The survey's
"flag rather than resolve" treatment (UNVERIFIED/flagged, not admitted, not auto-refused)
is the correct call and is not contradicted by anything found here.

## 10. `McGill-NLP/stereoset` — StereoSet

**VERIFIED-AGREES.** Re-fetched the GitHub repo (`moinnadeem/StereoSet`): CC-BY-SA-4.0
badge and `LICENSE.md` file confirmed present. HF mirror tag `cc-by-sa-4.0` matches.
Verdict SHARE_ALIKE stands.

## 11. `nyu-mll/crows_pairs` — CrowS-Pairs

**VERIFIED-AGREES.** Re-fetched the GitHub repo directly: *"CrowS-Pairs is licensed under
a Creative Commons Attribution-ShareAlike 4.0 International License"* — verbatim match.
The ROCStories/MNLI-fiction sub-lineage note the survey flagged as an unverified
sub-dependency was independently reconfirmed present in the same README ("created using
prompts taken from the ROCStories corpora and the fiction part of MNLI") — this pass did
not chase ROCStories'/MNLI's own licences either (out of scope for a licence-text match
check on CrowS-Pairs itself), so the survey's "provisional pending that sub-dependency"
framing is appropriately cautious and stands as-is.

## 12. `allenai/social_bias_frames` — Social Bias Frames

**VERIFIED-AGREES, second-surface check attempted and still unresolved (as the survey
itself flagged).** HF card re-fetched: `license: cc-by-4.0`, README states *"The SBIC is
licensed under the Creative Commons 4.0 License"* — matches the survey's characterization.
This pass then tried the two obvious second-primary-surfaces the survey said it hadn't
reached: `github.com/allenai/social-bias-frames` returned **404** (no such repo at that
path — AI2 did not ship this one as a GitHub release the way Scruples/WildGuard were), and
the author's project page (`maartensap.com/social-bias-frames/`, after a redirect) hosts
the download archive and a `DATASTATEMENT.MD` file but states no licence text on the page
itself and the archive/statement file were not fetched (no dataset content downloads, and
a data-statement file inside a dataset archive is exactly the kind of content this pass
was told not to pull down). **Net: still only mirror-verified, exactly as the survey
already disclosed** — this is not a contradiction, it's a confirmation that the survey's
own "needs a follow-up fetch" flag remains open. ATTRIBUTION (provisional) stands.

## 13. `ucberkeley-dlab/measuring-hate-speech`

**CONTRADICTED — not on the licence tag, but on a missed provenance risk class.** HF card
re-fetched: `license: cc-by-4.0` confirmed, matching the survey. But independent WebSearch
on the dataset's own description surfaced something the survey's writeup for this entry
never mentions: *"The posts were collected between March and August 2019 from three major
online platforms: Twitter, Reddit, and YouTube"* (per the dataset's own documented
methodology, Kennedy et al. 2020 / Sachdeva et al. 2022). This is the same risk class the
operator explicitly asked surveyors to flag — a first-party CC-BY-4.0 grant plausibly
covers UC Berkeley D-Lab's own **annotations** (the hate-speech scores and IRT labels), but
says nothing on its face about redistribution rights to the **underlying post text**
scraped from Twitter/Reddit/YouTube, none of whose owners are party to that CC-BY-4.0
grant. The survey handled this exact question explicitly and carefully for Scruples (#4,
"distributor disclaims owning what they distribute" framing) and implicitly for Civil
Comments (#8, CC0 stated to cover "the underlying comment text" too, which is the
distinguishing fact that makes #8 clean) — but entry #13 gives no such treatment despite
being the one candidate in this survey whose entire underlying corpus is Twitter/Reddit
content, not platform-archived (Civil Comments) or distributor-Apache-2.0-mediated
(Scruples) text. This pass could not resolve whether D-Lab's CC-BY-4.0 grant is
scoped to annotations-only or extends to the post text (the HF card content needed to
answer this was truncated in every fetch attempted, and no dataset content could be
downloaded to check the schema/fields directly). **Recommend downgrading #13 from
"VERIFIED (mirror-corroborated, first-party org)" to provisional, alongside #12/#15/#16,
pending an explicit check of whether the CC-BY-4.0 grant's scope statement addresses
underlying post text** — the same annotations-vs-underlying-media gap class the operator
named directly.

## 14. `google/jigsaw_toxicity_pred`

**VERIFIED-AGREES, upgraded.** The Kaggle competition rules page itself was unreachable in
this pass too (returned no usable content — the survey's own prediction that "Kaggle
competition pages typically require login for full terms" held). But this pass found and
fetched an independent second primary-adjacent surface the survey didn't cite: the
TensorFlow Datasets catalog page for `wikipedia_toxicity_subtypes` (the TFDS name for this
same Wikipedia-Detox/Jigsaw corpus), which states verbatim: *"This dataset is released
under CC0, as is the underlying comment text."* That is the same Google-maintained,
independent-of-HF-mirror surface type the survey itself used to verify Civil Comments
(#8) — applying that same standard here upgrades this entry from "mirror only, provisional"
to VERIFIED via a second independent surface, though the literal Kaggle terms page remains
unchecked (still worth a login-gated follow-up before this is treated as fully closed).
Verdict PERMISSIVE_OK stands, more firmly than the survey's own hedge suggested.

## 15. `Hate-speech-CNERG/hatexplain` — HateXplain

**CONTRADICTED.** This is the clearest mirror-vs-primary mismatch found in this pass, and
it is not even a mirror-vs-upstream mismatch — it's a **mismatch inside the mirror card
itself**: the HF card's YAML front matter states `license: cc-by-4.0` (which is all the
survey quoted and built its ATTRIBUTION verdict on), but that same card's own "Licensing
Information" prose section states, in full: **"MIT License."** Independently fetching the
primary source — `api.github.com/repos/hate-alert/HateXplain/license` — resolves this
disagreement in MIT's favor: it returns the actual repo `LICENSE` file, full MIT text,
"Copyright (c) 2020 Punyajoy Saha" (the paper's first author). The repo's rendered README
page shows only a generic, apparently-leftover PyPI licence badge (`ansicolortags`) that
names neither licence and should not be read as evidence either way.

**Corrected verdict: PERMISSIVE_OK (MIT), not ATTRIBUTION (CC-BY-4.0).** The primary
source (GitHub LICENSE file, first-party CNERG/hate-alert repo) and the mirror card's own
prose agree with each other and disagree with the mirror card's machine-readable tag —
exactly the "mirror lies" shape the operator asked this pass to hunt for, just located one
level deeper than tag-vs-upstream: tag-vs-the-mirror's-own-prose. Since MIT and CC-BY-4.0
are both in the open-weights-compliant bucket this doesn't change whether HateXplain is
admissible, but it does change the propagation obligation (MIT: preserve notice; CC-BY-4.0:
attribution + share terms on any redistribution) and it means the survey's Top candidates
of "verified" datasets contains at least one whose recorded verdict class doesn't match its
primary licence. Flag for correction before #15 is cited as ATTRIBUTION anywhere downstream.

## 16. `demelin/moral_stories` — Moral Stories

**VERIFIED-AGREES, upgraded.** The survey called this "attempted, inconclusive... weakest
verification in this survey" because the raw GitHub LICENSE fetch 404'd. Using
`api.github.com/repos/demelin/moral_stories/license` instead (GitHub's resolved-license
endpoint, same technique that closed #2's gap) returns the full MIT text: "Copyright (c)
2020 Denis Emelin." This closes the survey's own explicitly-named weakest link. Verdict
PERMISSIVE_OK stands, and should no longer be flagged as the survey's shakiest entry — it's
now one of the more solidly verified ones, on par with ETHICS and Scruples.

## 17. `ninoscherrer/moralchoice` — MoralChoice

**VERIFIED-AGREES, no change to the data-specific caveat.** HF card re-confirmed:
`license: cc-by-4.0`. `api.github.com/repos/ninodimontalcino/moralchoice/license` returns
**MIT** — but this is the code repository's licence (paper's benchmark-generation code),
not a data-specific grant; it's the same code/data split pattern seen throughout this
survey (#5, #7), not a contradiction of the CC-BY-4.0 data claim. The survey's own caveat —
that the CC-BY-4.0 **data** licence itself rests on the HF mirror tag alone, with no
data-specific primary statement independently located — remains true after this check;
this pass did not find a data-specific licence statement beyond the mirror tag either.
ATTRIBUTION (provisional, for the data specifically) stands as the survey stated it.

## 18. `mmathys/openai-moderation-api-evaluation`

**VERIFIED-AGREES, upgraded.** `api.github.com/repos/openai/moderation-api-release/license`
returns the full MIT text, "Copyright (c) 2022 OpenAI" — a first-party OpenAI repo, not a
third-party mirror. This upgrades the survey's "mirror only" flag to a primary-source
verification. HF card `license: mit` re-confirmed independently. Verdict PERMISSIVE_OK
stands, now solidly verified rather than provisional.

## 19. `walledai/TDC23-RedTeaming`

**VERIFIED-AGREES with the survey's own caution — and that caution is now demonstrably
warranted, not just prudent.** The survey flagged this as carrying "the highest 'mirrors
lie' risk of any entry in this survey" because it's a third-party repackaging
(`walledai`) of a defunct competition's data. This pass tried to reach the competition
site directly: `trojandetection.ai` now 301-redirects to an entirely unrelated site
(`redgrid.io`) — the original competition's own primary surface is **gone**, which is
about as strong a confirmation of "hard to verify at the source" as this survey format can
produce. The nearest living artifact, `centerforaisafety/tdc2023-starter-kit` on GitHub
(the competition organizer's own starter-kit repo), does carry a licence:
`api.github.com/repos/centerforaisafety/tdc2023-starter-kit/license` returns MIT,
"Copyright (c) 2024 centerforaisafety" — this is corroborating (organizer-side, MIT,
matching the mirror tag) but it licenses the **starter-kit code**, not a data-specific
grant for the red-teaming prompt set itself, and the successor project HarmBench
(`centerforaisafety/HarmBench`, also MIT) makes no mention of TDC23 licensing terms at
all when fetched directly. **Net: PERMISSIVE_OK (provisional) stands, but "provisional"
should stay attached — the primary competition surface cannot be reached at all anymore,
and the closest living primary surface only confirms a code licence, not a data one.**

## 20. `kellycyy/daily_dilemmas`

**VERIFIED-AGREES, mirror-only status confirmed still open.** HF card re-fetched:
`license: cc-by-4.0`, paper identified as arXiv:2410.02683 ("DailyDilemmas: Revealing
Value Preferences of LLMs with Quandaries of Daily Life," Chiu, Jiang, Choi). The arXiv
abstract page itself displays a CC BY 4.0 licence icon — but that is arXiv's standard
**paper-text** licence badge, not necessarily a statement about the **dataset's** licence,
and should not be read as independent dataset-licence corroboration without more (this is
exactly the kind of paper-license-vs-data-license conflation risk worth flagging on its
own, distinct from the entry's substance). No project page independent of HF/arXiv was
located. ATTRIBUTION (provisional, mirror-only for the dataset specifically) stands
exactly as the survey stated it — no upgrade, no contradiction.

---

## Summary

| # | Dataset | Survey verdict | This pass |
|---|---|---|---|
| 1 | hendrycks/ethics | PERMISSIVE_OK | VERIFIED-AGREES |
| 2 | Anthropic/hh-rlhf | PERMISSIVE_OK | VERIFIED-AGREES (upgraded: thin → full) |
| 3 | wassname/social_chemistry_101 | SHARE_ALIKE | VERIFIED-AGREES |
| 4 | metaeval/scruples | PERMISSIVE_OK | VERIFIED-AGREES |
| 5 | PKU-Alignment/PKU-SafeRLHF | NC | VERIFIED-AGREES |
| 6 | PKU-Alignment/BeaverTails | NC (provisional) | **CONTRADICTED on status** — should be VERIFIED, not provisional; survey under-credited its own primary evidence and missed the GitHub Apache-2.0 code split |
| 7 | allenai/wildguardmix / wildjailbreak | ODC-BY-adjacent | VERIFIED-AGREES |
| 8 | google/civil_comments | PERMISSIVE_OK | VERIFIED-AGREES |
| 9 | toxigen/toxigen-data | UNVERIFIED/flagged | VERIFIED-AGREES (conflict reproduced) |
| 10 | McGill-NLP/stereoset | SHARE_ALIKE | VERIFIED-AGREES |
| 11 | nyu-mll/crows_pairs | SHARE_ALIKE (provisional) | VERIFIED-AGREES |
| 12 | allenai/social_bias_frames | ATTRIBUTION (provisional) | VERIFIED-AGREES (second surface still unreachable, as survey disclosed) |
| 13 | ucberkeley-dlab/measuring-hate-speech | ATTRIBUTION | **CONTRADICTED — missed provenance flag**: Twitter/Reddit/YouTube-sourced text, annotations-vs-underlying-media gap not addressed |
| 14 | google/jigsaw_toxicity_pred | PERMISSIVE_OK (provisional) | VERIFIED-AGREES (upgraded via TFDS second surface) |
| 15 | Hate-speech-CNERG/hatexplain | ATTRIBUTION (provisional) | **CONTRADICTED — corrected verdict PERMISSIVE_OK (MIT)**: primary source + mirror's own prose both say MIT, only the mirror's YAML tag says CC-BY-4.0 |
| 16 | demelin/moral_stories | PERMISSIVE_OK (provisional, weakest) | VERIFIED-AGREES (upgraded: inconclusive → full) |
| 17 | ninoscherrer/moralchoice | ATTRIBUTION (provisional) | VERIFIED-AGREES (data claim still mirror-only, as stated) |
| 18 | mmathys/openai-moderation-api-evaluation | PERMISSIVE_OK (provisional) | VERIFIED-AGREES (upgraded: mirror → primary) |
| 19 | walledai/TDC23-RedTeaming | PERMISSIVE_OK (provisional, highest risk) | VERIFIED-AGREES (caution now confirmed warranted — primary site is gone) |
| 20 | kellycyy/daily_dilemmas | ATTRIBUTION (provisional) | VERIFIED-AGREES (still mirror-only, as stated) |

**Two real corrections for the record:**

- **#15 HateXplain should carry verdict PERMISSIVE_OK (MIT), not ATTRIBUTION (CC-BY-4.0).**
  The mirror's own YAML tag disagrees with both the primary GitHub LICENSE file and the
  same mirror card's own prose text — a mirror-lies case one layer deeper than the
  tag-vs-upstream pattern the "10/75 mismatches" precedent was built on.
- **#13 Measuring Hate Speech needs the annotations-vs-underlying-media question answered**
  before its ATTRIBUTION verdict is treated as settled — it is the one entry in this
  survey built directly from Twitter/Reddit/YouTube text without the mediating
  distributor-licence reasoning the survey applied to Scruples (#4) or the "CC0 covers the
  underlying text too" language that makes Civil Comments (#8) clean.

**One correction that helps the survey, not against it:**

- **#6 BeaverTails' NC verdict should be promoted from "provisional" to VERIFIED.** The
  survey had the evidence (the HF card it already quoted a tag from) and didn't quote the
  card's own explicit sentence, and didn't check the GitHub repo's own Apache-2.0
  code-licence at all. Nothing here weakens the NC read — it strengthens it.

**Everything else (17 of 20 entries) either matches the survey's verdict and quoted text
exactly, or is upgraded from "provisional/mirror-only" to a firmer VERIFIED by a primary
or independent second surface this pass could reach that the original survey's time-box
didn't cover** (#2, #14, #16, #18 upgraded this way; #12, #17, #19, #20 remain correctly
flagged provisional because a genuine second surface still could not be reached, exactly
as the survey itself said). No candidate surfaced a licence that should flip to REFUSE.
