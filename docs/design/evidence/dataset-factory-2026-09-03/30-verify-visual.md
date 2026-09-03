# Adversarial verification — `10-survey-visual.md`

Verify pass, 2026-09-03. Method: for every candidate whose verdict is not already `REFUSE`,
fetch the primary licence source myself (project/portal page, GitHub `LICENSE`, HF card field
+ README) and compare against the surveyor's quote and verdict. `REFUSE` entries (2, 17-class
citation, 22) were spot-checked only where a REFUSE-class quote was load-bearing for the
verdict (entry 17's "research purposes only"), per the task's own framing of REFUSE-TERMS as
worth confirming since it anchors an explicit operator refusal criterion.

Legend: **VERIFIED-AGREES** (I fetched the primary text and it matches the surveyor's quote and
verdict) / **CONTRADICTED** (primary text does not match — corrected verdict/quote given,
clause quoted) / **UNVERIFIABLE** (I could not reach a primary source this session either — the
gap is named, not papered over).

No dataset content was downloaded — HF API/card fetches and licence/terms pages only.

---

## Entry-by-entry

### 1. COCO (`HuggingFaceM4/COCO` + captions mirrors)
**VERIFIED-AGREES**, with the same residual gap the surveyor already flagged. `cocodataset.org/#termsofuse`
still would not render through WebFetch (three more attempts this session, same JS-gated nav
shell) — this session did **not** manage to read the primary page either, so entry 1 stays
correctly flagged as INFERRED-not-VERIFIED for the terms page itself. However, I independently
corroborated the quoted split across two unrelated secondary sources this session: a WebSearch
result and the `cocodataset/cocoapi` GitHub issue #81 thread both reproduce, in substance, the
same annotations-vs-images split the surveyor quoted (*"annotations belong to the COCO
Consortium and are licensed under a Creative Commons Attribution 4.0 License"* / *"use of the
images must abide by the Flickr Terms of Use"*), and issue #81 is itself a users' complaint that
*"all the images I checked appear to have been copied in violation of their original license"* —
independent evidence the per-photographer risk is real, not theoretical. Separately VERIFIED
this session via a fresh HF card fetch: `HuggingFaceM4/COCO`'s mirror tag is exactly `cc-by-4.0`
as stated, and the card's own licensing section says `[More Information Needed]` — the mirror
does not resolve the image-rights question either. **BLOCKING verdict confirmed correct.**
Follow-up item 1 in the survey's own "what remains open" list is still open — record that,
don't downgrade the confidence on the strength of secondary corroboration.

### 3. SBU Captions
**UNVERIFIABLE** — not independently fetched this pass either (the surveyor itself flagged this
as "low priority to chase given COCO is already blocking on the identical defect"; I agree with
that triage and did not spend a fetch on a `~2011` Flickr-scrape page unlikely to carry a
different answer from the already-established Flickr-scrape pattern). No contradiction found;
verdict stands **BLOCKING (unresolved)** on inherited reasoning, not fresh evidence.

### 4. Visual Genome
**VERIFIED-AGREES**, and partially upgraded. Fetched `visualgenome.org/about` this session:
confirms verbatim *"Visual Genome by Ranjay Krishna is licensed under a Creative Commons
Attribution 4.0 International License."* — this upgrades the annotation-licence half of the
entry from the surveyor's INFERRED to VERIFIED. The image-pool claim (COCO + YFCC100M) was
**not** independently re-confirmed on this page (it doesn't mention image sourcing) — that half
stays UNVERIFIABLE this pass, same as before, though it is uncontested public record from the
VG paper itself. **BLOCKING verdict confirmed correct** on the annotation-clean/image-dirty
split — same shape as COCO, now with the annotation half actually read rather than assumed.

### 5/6/8. VQAv2/OK-VQA/A-OKVQA, GQA, Localized Narratives
**UNVERIFIABLE** this pass — none of `visualqa.org`, `okvqa.allenai.org`, `allenai.org/data/a-okvqa`,
GQA's Stanford page, or `google.github.io/localized-narratives` were independently fetched this
session (budget triage: these are all downstream of entries 1/4/7, whose upstream image-rights
problem is now more solidly evidenced than before — see entries 1, 4, 7). No contradiction
found. The inheritance logic itself is sound (a QA-pair layer cannot grant rights over an image
pool it doesn't own), so **BLOCKING stands on structural grounds** even without a fresh fetch
per downstream repo. Flag explicitly: this is inherited-reasoning verification, not
independently re-derived — same caveat the surveyor itself used for these five entries.

### 7. TextVQA / NoCaps
**Split result.** NoCaps: **VERIFIED-AGREES** — fetched `nocaps.org/download` this session,
confirms verbatim *"Sourced from Open Images validation set (v4)"* / *"Sourced from Open Images
test set (v4)"*. TextVQA: **UNVERIFIABLE** — `textvqa.org` and `textvqa.org/dataset/` both
returned empty content to WebFetch twice this session; could not independently confirm the
Open-Images-sourcing claim for TextVQA specifically this pass (it is well-documented publicly,
just not re-read here). The shared citation — Google's Open Images page disclaiming a uniform
image licence — **VERIFIED-AGREES exactly**: fetched
`storage.googleapis.com/openimages/web/factsfigures.html` this session, confirms the quote
verbatim, character for character: *"we make no representations or warranties regarding the
license status of each image."* **BLOCKING verdict confirmed correct** for NoCaps on direct
evidence; TextVQA's verdict rests on the same well-established Open Images defect but without a
fresh independent fetch this pass.

### 9. Flickr30k
**VERIFIED-AGREES** — fetched `huggingface.co/datasets/nlphuji/flickr30k` this session: no
license field/tag is set on the card, and the README carries only the academic citation, no
licensing text — matches the surveyor's "no uniform grant located" characterization exactly.

### 10. `wikimedia/wit_base` (WIT)
**VERIFIED-AGREES**, with a genuinely new detail in favour of the surveyor's "leaning
SHARE_ALIKE, not BLOCKING" read. Fetched the HF card this session: dataset-level `CC BY-SA 4.0`
confirmed. On the gap the surveyor flagged — does the card address per-image licensing — this
session's fetch confirms the gap is real (*"the dataset page does not provide detailed per-image
licensing information"*) but also surfaces something the surveyor didn't have: WIT's own
curation pipeline **excludes** *"all images that are candidate for deletion on Commons"* and
additionally excludes images where *"a person's face covers more than 10% of the image
surface."* That's a filtering step on top of Commons' baseline, not just a raw pull — a mild
positive signal for the "floor is probably SHARE_ALIKE at worst" lean, still short of a per-file
audit. **UNVERIFIED (leaning SHARE_ALIKE) stands, on slightly firmer footing than before.**

### 11. DocVQA (task 1)
**VERIFIED-AGREES on the gate, with one addition worth flagging as a caution, not a
contradiction.** Fetched `docvqa.org/datasets/doccvqa` this session: confirms login-gated
download (*"You must login to the portal before downloading"*) and that terms live on the
Downloads page (*"read the terms and conditions listed in the Download page"*), matching the
surveyor's characterization exactly — the RRC portal `?ch=17` page itself could not be fetched
(TLS certificate error both attempts, a different failure mode from the surveyor's, same net
result: unread). **New finding not in the survey:** a WebSearch pass for the RRC downloads page
surfaced a reference to a *"SOFTWARE EVALUATION LICENSE AGREEMENT"* governing at least one
RRC-hosted document-VQA dataset in this same challenge family (surfaced for InfoVQA, not
independently confirmed for DocVQA task 1 specifically — same portal, not the same page). That
phrasing ("evaluation license") is the same shape of restrictive research-only term already
REFUSE'd for SA-1B (entry 17) elsewhere in this survey. I'd temper the summary table's "leaning
favorable" framing for DocVQA until that specific agreement text is read for task 1 — the UCSF
image-source argument (favorable) and the RRC portal's apparent licensing shape (potentially
restrictive, unconfirmed) are two separate questions, and only the first was actually
substantiated. **Verdict stays UNVERIFIED, but the lean should not be stronger than neutral
pending that specific text.**

### 12. ChartQA
**VERIFIED-AGREES**, fully confirmed both halves. Fetched
`raw.githubusercontent.com/vis-nlp/ChartQA/main/LICENSE` this session: full GPLv3 text, exactly
as the surveyor described (software copyleft license filed on a data repo). Fetched the repo
README this session: confirms verbatim the Pew Research quote *"the Pew Research Centre chart
images didn't have any SVG files when we crawled them"* — this is a crawler-authored README
line describing images crawled from a third party, textbook confirmation of the "licence covers
crawler code, not crawled images" defect. The Statista half of the sourcing claim was not
independently re-confirmed (as the surveyor itself flagged it wasn't). **BLOCKING (unresolved)
verdict confirmed correct**, on stronger evidence than before.

### 13. FigureQA
**CONTRADICTED (narrowly) — corrected finding, not a corrected verdict.** The surveyor stated
*"The repository's specific LICENSE file text was not retrieved (404/gated)"* and treated the
"MS releases under permissive terms" claim as pattern-based INFERENCE. This session **did**
retrieve it: `raw.githubusercontent.com/Maluuba/FigureQA/master/LICENSE.txt` returns a full MIT
License, copyright Microsoft Corporation — a plain, unambiguous permissive software license, not
an inference. That part of the surveyor's gap is now closed and resolves in the favorable
direction the surveyor guessed at. **But** — and this is the actual contradiction worth
flagging — that MIT `LICENSE.txt` sits in the **code** repository (the Bokeh-based figure
generator). The README states the actual dataset **images** are downloaded separately from a
Microsoft Research project page (*"The dataset is available for download here"*, linking off
GitHub to a Microsoft Research / Azure blob-storage endpoint), whose own terms I could not
locate this session (WebSearch found the MSR project and blog pages but not a specific data-use
license string for FigureQA's download). This is the **identical structural risk already named
for ChartQA (entry 12) and, by extension, item 13's own "leaning PERMISSIVE_OK"**: a clean
license on the code/repo layer does not by itself establish the license of a separately-hosted
data artifact. Because the data here is entirely synthetic (MS-generated from synthetic numbers
via its own Bokeh fork, no third-party photography involved) the actual risk is much lower than
ChartQA's — there's no third-party rightsholder in the picture, only the question of which MS
data-use terms (if any beyond MIT) apply to the MSR download itself. **Corrected verdict:
UNVERIFIED still, but for a narrower, better-specified reason** (code confirmed MIT; the
specific MSR data-download terms remain unread) rather than the surveyor's broader "LICENSE file
unretrievable" framing, which is no longer accurate.

### 14. PlotQA
**CONTRADICTED (repo location) + one unresolved discrepancy worth flagging.** The surveyor
stated the repo access attempt 404'd on "the `vis-nlp` org mirror path tried" and that "the
correct org may differ; not resolved." This session located and confirmed the correct
repository: `github.com/NiteshMethani/PlotQA` (the original author's org, not `vis-nlp`) —
resolves the surveyor's own open question. Fetched
`raw.githubusercontent.com/NiteshMethani/PlotQA/master/LICENSE` this session: returns MIT
License text. However, a WebSearch pass this session separately returned the claim *"The PlotQA
dataset is released under a CC-BY-4.0 license, while all models and code are released under an
MIT license"* — i.e. a secondary source asserts a **split** licence (data CC-BY-4.0, code MIT)
that the repo-root `LICENSE` file I actually read does not itself state (it's a bare MIT text,
no data/code split language visible in what was fetched). I could not confirm the CC-BY-4.0
data-license claim in any primary text this session — `PlotQA_Dataset.md` (fetched) contains
download links and format description only, no licensing or World-Bank-sourcing text despite the
surveyor's own note flagging that as unconfirmed already. **Net: verdict stays UNVERIFIED.**
Record the repo-location correction (`NiteshMethani/PlotQA`, not `vis-nlp`) for the next pass,
and record the MIT-vs-CC-BY-4.0 discrepancy as unresolved rather than picking one.

### 15. AI2D
**VERIFIED-AGREES** — fetched both `prior.allenai.org/projects/diagram-understanding` and the HF
card (`lmms-lab/ai2d`) this session; neither exposes an explicit license field or license text.
The AI2 page offers only a contact address, the HF card only the academic citation
(arXiv:1603.07396). Matches the surveyor's "NOT VERIFIED this session" characterization exactly
— now doubly confirmed across two independent primary-ish sources rather than one skipped
attempt. **UNVERIFIED verdict confirmed correct**, no basis found this pass for the "AI2 usually
releases permissively" pattern inference beyond what the surveyor already labeled as inference.

### 16. ScienceQA
**VERIFIED-AGREES** — fetched `huggingface.co/datasets/derek-thomas/ScienceQA` this session:
license field confirmed exactly `cc-by-sa-4.0` as quoted. The upstream multi-curriculum-source
question (IXL Learning-origin claim) was not independently re-verified this pass either — the
surveyor already flagged that as unconfirmed, and nothing found this session changes it.
**UNVERIFIED (mirror tag confirmed SHARE_ALIKE, upstream constituent-source risk still open)
stands as stated.**

### 17. Segment Anything / SA-1B
**VERIFIED-AGREES, exact quote match.** Fetched `ai.meta.com/datasets/segment-anything/` this
session: confirms verbatim both load-bearing phrases — *"Research purposes only"* and
*"Images licensed from a photo company"* / *"The underlying images are licensed from a large
photo company."* This is the anchor for the REFUSE-TERMS verdict against the operator's explicit
refusal criteria (research-only, non-redistributable) and it checks out exactly as quoted.
**REFUSE-TERMS verdict confirmed correct**, primary text read directly (not inferred).

### 18. ChartQA-derivative cluster (ChartBench / ChartDQA / ChartQAPro / Chartographer)
**VERIFIED-AGREES, and meaningfully strengthened.** The surveyor flagged this as "NOT VERIFIED
this session" for the derivative-inheritance claim, citing only `1fanj/Chartographer`'s
`source_datasets` field by description. This session fetched
`huggingface.co/datasets/1fanj/Chartographer` directly and found the card **states the
inheritance risk explicitly, in the maintainers' own words**: license field reads *"Chartographer
is released under CC BY-SA 4.0 where permitted by the source terms"* and the card adds *"The
original chart images from upstream datasets are not included in this release and may be
governed by separate licenses. Researchers can recover these original images using the
`source_row_index` field paired with upstream dataset identifiers."* That is a primary-source
confession of exactly the defect the surveyor inferred from field structure alone — the
derivative's own permissive-looking tag is explicitly qualified as conditional on unverified
source terms, and the base chart images are deliberately excluded from the release specifically
because their rights aren't cleared. **BLOCKING (inherited) verdict confirmed correct, upgraded
from UNVERIFIED-inferred to VERIFIED-AGREES on direct card text** for at least the Chartographer
member of the cluster; ChartBench/ChartDQA/ChartQAPro individually were not each re-fetched this
pass (still open per-repo as the surveyor flagged), but the pattern is now evidenced rather than
assumed for the cluster's flagship example.

### 19. InfographicVQA
**UNVERIFIABLE** this pass — `mm-eval/InfographicVQA` / `Ryoo72/InfographicsVQA` HF cards and the
original paper were not independently fetched this session (budget triage against higher-value
targets). No contradiction found. One relevant addition carries over from entry 11: the RRC
portal family (same `docvqa.org`/`rrc.cvc.uab.es` infrastructure) surfaced a *"SOFTWARE
EVALUATION LICENSE AGREEMENT"* reference this session — if that phrasing turns out to govern
InfographicVQA specifically as well as (or instead of) DocVQA, it reinforces rather than weakens
the surveyor's "worse-verified provenance" ranking. **BLOCKING (unresolved) stands**, not
independently re-derived this pass.

### 21. Wikimedia Commons (direct)
**CONTRADICTED — the enforcement mechanism, not the underlying conclusion.** The surveyor wrote:
*"Commons requires every upload to carry an explicit, machine-readable licence... enforced at
upload time"* and called this *"the only candidate... where the per-image licence is
structurally guaranteed to exist."* Fetched `commons.wikimedia.org/wiki/Commons:Licensing`
directly this session — the primary policy page does **not** describe automated/technical
enforcement at upload time. It describes a **manual, policy-and-moderation** requirement: *"The
license that applies to an image or media file must be indicated clearly on the file description
page using a copyright tag"* plus required source/author fields, with compliance handled through
description-page conventions and (implicitly) deletion review for missing/invalid tags, not a
gate the upload software itself blocks on. The page also places responsibility on reusers:
*"it is the responsibility of reusers to ensure that the use of the media is according to the
license."* **This is a real overstatement in the survey** — "structurally guaranteed" and
"enforced at upload time" both claim a stronger, automated certainty than the primary text
supports. The corrected characterization: Commons *requires* a license tag as a matter of
written policy, checkable per-file via the API (`imageinfo`/`extmetadata`) once present, and
unlicensed/miscategorized uploads are subject to human deletion review — meaningfully better
than an arbitrary web scrape (COCO/Flickr30k/SBU have **no** such policy at all), but "policy
requirement + moderation" is not the same certainty class as "structurally guaranteed." **Net
effect on the verdict:** entry 21 should stay ranked as the strongest source-of-images candidate
in the survey (the comparative claim against scraped sets holds), but the summary table's
framing should read "policy-enforced, moderation-backed" rather than "structurally guaranteed" /
"enforced at upload time," and the recommended per-file API audit (already flagged as the
highest-value follow-up) becomes more important, not less, given that enforcement is
social/moderation-based rather than technical.

---

## Roll-up

| # | dataset | surveyor verdict | this pass | note |
|---|---|---|---|---|
| 1 | COCO | BLOCKING | VERIFIED-AGREES | primary page still unfetchable; 2 independent secondary sources corroborate the quoted split; mirror tag `cc-by-4.0` freshly confirmed, card itself unresolved on images |
| 3 | SBU Captions | BLOCKING (unresolved) | UNVERIFIABLE | not fetched, low priority by design |
| 4 | Visual Genome | BLOCKING | VERIFIED-AGREES | annotation licence (CC BY 4.0) now directly read, upgraded from INFERRED; image-pool claim still unconfirmed this pass |
| 5/6/8 | VQAv2/OK-VQA/A-OKVQA, GQA, Loc. Narratives | BLOCKING | UNVERIFIABLE (inherited) | no fresh fetch; inheritance logic sound |
| 7 | TextVQA / NoCaps | BLOCKING | split: NoCaps VERIFIED-AGREES, TextVQA UNVERIFIABLE | Open Images quote confirmed verbatim; NoCaps sourcing confirmed verbatim |
| 9 | Flickr30k | BLOCKING | VERIFIED-AGREES | HF card confirmed no licence field |
| 10 | WIT | UNVERIFIED, leaning SHARE_ALIKE | VERIFIED-AGREES | CC BY-SA 4.0 confirmed; found a new positive detail (Commons-deletion-candidate filtering) not previously on record |
| 11 | DocVQA | UNVERIFIED, leaning favorable | VERIFIED-AGREES on the gate; **temper the lean** | new finding: "SOFTWARE EVALUATION LICENSE AGREEMENT" phrasing in the same portal family, unconfirmed for task 1 specifically but a caution |
| 12 | ChartQA | BLOCKING (unresolved) | VERIFIED-AGREES | both GPLv3 text and Pew Research quote confirmed verbatim from primary sources |
| 13 | FigureQA | UNVERIFIED, leaning PERMISSIVE_OK | **CONTRADICTED (narrow)** | code LICENSE.txt actually read = MIT (not "unretrievable" as stated); but the dataset images live at a separate MSR endpoint whose own terms are still unread — same wrong-layer risk as ChartQA |
| 14 | PlotQA | UNVERIFIED | **CONTRADICTED (repo location)** | correct repo identified (`NiteshMethani/PlotQA`); LICENSE file read = MIT, conflicting with a secondary claim of CC-BY-4.0 for the data — unresolved discrepancy, flag for next pass |
| 15 | AI2D | UNVERIFIED, leaning favorable | VERIFIED-AGREES | no licence text found on either of two independent primary-ish sources checked this session |
| 16 | ScienceQA | UNVERIFIED (mirror SA) | VERIFIED-AGREES | `cc-by-sa-4.0` mirror tag confirmed exactly |
| 17 | SA-1B | REFUSE-TERMS | VERIFIED-AGREES | both load-bearing quotes confirmed verbatim from the primary Meta page |
| 18 | ChartQA-derivative cluster | BLOCKING (inherited), NOT VERIFIED | **VERIFIED-AGREES, strengthened** | Chartographer's own card explicitly states the inheritance risk in its own licensing text — direct primary confirmation, not inference |
| 19 | InfographicVQA | BLOCKING (unresolved) | UNVERIFIABLE | not fetched; RRC-portal caution from entry 11 applies by association |
| 21 | Wikimedia Commons (direct) | PERMISSIVE_OK/SHARE_ALIKE, "structurally guaranteed," "enforced at upload time" | **CONTRADICTED (mechanism)** | primary policy page describes a manual tag-and-moderation requirement, not automated upload-time enforcement — comparative ranking against scraped sets still holds, but overstates certainty; the recommended per-file audit is now more, not less, important |

## What this pass adds that the survey didn't have

1. Two structural "licence covers the wrong layer" risks previously only proven for ChartQA
   (GPLv3-on-crawler-code) now have a second and third confirmed instance: FigureQA's MIT
   license sits on the generator code, not the separately-hosted dataset; Chartographer's own
   card explicitly disclaims covering the upstream chart images it derives from. This is now a
   **three-instance pattern**, not a one-off — worth naming as its own trap class alongside "the
   recurring trap" section already in the survey (per-photographer scrapes) and the
   metadata_only/URL-list trap (LAION/CC12M): call it the **wrong-layer trap** (a real, verified
   licence exists, but it covers the tooling/annotations/derivative work, not the underlying
   media asset).
2. Wikimedia Commons' licensing enforcement is policy-and-moderation-based, not automated at
   upload time — the survey's strongest candidate is still the strongest candidate, but the
   confidence language attached to it needs to come down one notch, and the recommended
   per-file audit follow-up is now load-bearing rather than a nice-to-have.
3. A new caution surfaced for the RRC portal family (DocVQA + InfographicVQA): "SOFTWARE
   EVALUATION LICENSE AGREEMENT" phrasing appears somewhere in that portal's terms for at least
   one document-VQA dataset in the family. Not confirmed for DocVQA task 1 specifically, but it
   is the same restrictive shape already REFUSE'd for SA-1B and is reason to temper, not amplify,
   the "leaning favorable" framing DocVQA currently carries in the summary table until someone
   with an RRC account reads the actual agreement text.
4. PlotQA's repo location is now resolved (`NiteshMethani/PlotQA`, correcting the surveyor's
   "org may differ, not resolved" note) but its data license is now a genuine open discrepancy
   (MIT in the LICENSE file I read vs. a CC-BY-4.0 claim from a secondary source) rather than a
   simple "not fetched" gap — worth a targeted follow-up read of the paper's own data-availability
   statement.

## What is still genuinely open after this pass (unchanged or newly opened)

- COCO's own terms-of-use page: still unfetchable (now attempted across two sessions,
  same JS-gate failure mode both times) — the single highest-leverage fetch in this whole
  survey, still undone.
- RRC portal terms (DocVQA, InfographicVQA): still gated behind login; now carries a specific
  new caution ("software evaluation license" phrasing) worth chasing down by name rather than
  treated as a generic "gated, unread" gap.
- FigureQA's actual MSR data-download terms (distinct from the GitHub code MIT license).
- PlotQA's MIT-vs-CC-BY-4.0 discrepancy.
- The Wikimedia Commons / WIT per-file sampling audit — unchanged as the highest-value
  structural follow-up in the survey, now with an explicit correction that its floor is
  policy-enforced rather than automatically guaranteed.
