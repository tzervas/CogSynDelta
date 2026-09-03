# Audio Corpus Licence Audit — the `auditory` faculty and a `speech_output` head

**Status:** audit. This informs a decision; it does not make one.
**Scope:** every audio corpus considered as a candidate for the `auditory` input region and for
a speech-output head — 16 speech-with-transcript corpora (bucket A), 11 general-audio / sound-event
corpora (bucket B), 8 TTS corpora (bucket C).
**Nothing here changes a catalogue verdict, fetches a corpus, or touches `csd-regions.json`.**
No code is modified by this document.

> **Neither the author of this document nor its reader is a lawyer.** Everything below reports
> what a licence *says* and what the *risk* is. Where a question is legally unsettled it is
> marked unsettled and left for a human. No sentence here is a legal conclusion, and every place
> where one would be load-bearing is called out as needing a human decision rather than quietly
> assumed one way.

**Why this exists.** The operator's stated model intent — interaction by voice and/or text,
output as text and/or speech, with audio/visual/discrete-token input — makes `auditory` and a
`speech_output` head **required** regions rather than optional placeholders
`[OP: csd-multimodal-io-intent]`. Today `auditory` is a declared placeholder with **no catalogued
audio source anywhere in the tree** [V\*] (`REGION-TAXONOMY-AND-INTERCONNECT.md` §1.3, §5.5,
OD-9). This audit starts that catalogue. It answers OD-9's *"start the audit now"* recommendation,
which the operator has since made a requirement.

---

## Method, and the two checks audio needs that text and vision did not

Identical discipline to `LICENCE-FOR-OPEN-WEIGHTS.md`: every candidate was checked at the HF
mirror **and independently at the true upstream** (project page, GitHub `LICENSE`, paper, or the
data-hosting institution's own page), quoted verbatim where fetched live 2026-09-02, with **the
mirror tag and the upstream terms recorded as separate fields**. A mirror tag is never evidence
about its upstream — the rule this project's ten prior mismatches already established
(`memory/dataset-mirror-licences-lie.md`).

Two audio-specific checks were added throughout:

1. **Does the redistributor actually own the audio**, or is it YouTube/podcast-scraped
   third-party content the redistributor explicitly disclaims owning — the
   `AudioSet`/`tiny-imagenet` failure shape.
2. **Was the speaker's consent obtained**, and does that consent extend to *synthesis* (voice
   cloning) as opposed to mere publication.

**Marking convention.** **VERIFIED** = fetched live 2026-09-02 and quoted directly.
**INFERRED** = not independently fetched — a WebSearch-returned snippet of a page rather than a
direct fetch, general knowledge, or a secondary source. Every field below states which it is.

**One session-wide tooling caveat, and it is load-bearing.** For the research pass covering
TED-LIUM / MLS / Libri-Light / Emilia / YODAS / SPGISpeech / AMI / Switchboard / Granary /
Loquacious, `huggingface.co` returned **HTTP 401 to every fetch, including the
`/api/datasets/*` endpoint**, for the whole session. Every HF-mirror-tag field from that pass is
therefore **INFERRED from a WebSearch snippet**, not a direct API read, and is flagged again at
point of use. This is a tooling limitation, not a claim the tags do not exist — but those tags
carry lower confidence than the ones fetched directly (most of buckets A1, B and C).

---

## Verdict categories

Reused unchanged from `LICENCE-FOR-OPEN-WEIGHTS.md`, with the NC tier made explicit as its own
category per the operator's Decision 2026-09-02 and DEC-31.

| verdict | meaning |
|---|---|
| `PERMISSIVE_OK` | MIT / Apache-2.0 / BSD / CC0 / public domain at both mirror and upstream. |
| `ATTRIBUTION` | CC BY-family. Usable; a TASL notice is required in the release. |
| `SHARE_ALIKE` | CC BY-SA-family. Whether trained weights are "adapted material" is **unsettled** (see the main licence document); flagged, not resolved. |
| `NC` | A genuine, real-rights-holder-granted non-commercial term. **Per DEC-31 this does NOT block use** — it moves the affected region, and by the strictest-input rule the composed model, to an NC-family release licence. Accepted policy, not a fallback. |
| `BLOCKING` | No licence grant exists at all (the distributor disclaims ownership), an active dispute, a paywalled/membership corpus, or a use restriction (ND, academic-only, no-redistribution EULA) that the NC-tolerant policy was not written to absorb. |

**The distinction this audit had to police hardest: `NC` is not `BLOCKING`, and audio produces
both in volume that text and vision did not.** GooAQ — the precedent for this project's
NC-tolerant policy — is a real rights holder (AI2) choosing a restrictive term over text it
actually collected. Several audio corpora below *look* like that shape (a stock permissive mirror
tag over a restrictive upstream) but are structurally different: the **distributor itself
disclaims ownership** of the audio (Emilia, GigaSpeech, the AudioSet-derived family), or the
restriction is **NoDerivatives** rather than NonCommercial (TED-LIUM), or it is a bespoke
no-redistribution academic EULA (SPGISpeech, WavCaps), or it is a paywalled membership corpus
(Switchboard). None of those is cured by CogSynDelta relicensing its own release, for the identical
reason `LICENCE-FOR-OPEN-WEIGHTS.md` already established for tiny-imagenet: **relicensing the
release moves SHARE_ALIKE corpora, and now NC ones too. It moves nothing BLOCKING.**

---

## The short version

**Six clean, independent (non-LibriVox) provenance groups exist for a v1 `auditory` corpus, plus
one enormous, structurally clean single lineage (LibriVox) that must be treated as ONE source for
balance purposes no matter how many differently-named datasets draw from it.** The `auditory`
faculty is buildable at real scale without touching a single BLOCKING source. The `speech_output`
head is buildable too, with one small NC addition (Expresso) that costs nothing extra at the
composed-model level, because the composed model is *already* NC-tier via `memory` (DEC-31).

**Finding 1 — LibriVox is this audit's Wikipedia, and it did not come from any one dataset page.**
LibriSpeech, LibriTTS, LibriTTS-R, MLS, Libri-Light, LJSpeech, CSS10, M-AILABS, Hi-Fi TTS and
HiFiTTS-2 — **ten of the sixteen speech corpora audited, spanning both the auditory-input bucket
and the TTS-output bucket** — are re-segmentations, re-recordings or re-quality-passes over the
**same LibriVox volunteer public-domain-audiobook pool**. This is the audio-domain analogue of
`CORPUS-CONTRACT.md`'s Wikipedia lineage collision, except larger. **A v1 corpus assembled by
picking the highest-recommendation entry from each table would silently be a
90%-LibriVox monoculture wearing ten different names** — the exact `code`/CodeSearchNet-Python
defect, reproduced at the corpus-catalogue level before a single row is fetched.

**Finding 2 — audio reproduces the ImageNet/tiny-imagenet trap at a scale text never did.**
AudioSet's own upstream terms cover only its metadata; the widely-used HF audio mirror
(`agkphysics/AudioSet`) redistributes real YouTube audio under a `cc-by-4.0` tag Google never
asserted over that audio. AudioCaps and ~27% of WavCaps inherit that defect by lineage.
GigaSpeech's mirror tags `apache-2.0` over an upstream that says *"non-commercial research and
educational purposes"* **and** *"SpeechColab does not own the copyright of the audio files"*.
Emilia — 101,000 hours, the largest single corpus surveyed — says the same. **Five corpora here
carry a "the distributor disclaims ownership" clause.**

**Finding 3 — "manifest-only" distribution, a failure pattern this project had not yet named.**
NVIDIA Granary and the YODAS family ship YouTube video IDs and timestamps, not audio, tagged CC BY
at the aggregation layer over uploader self-declarations nobody verified. This is the audio
instance of the "URLs, not pixels" evasion `LICENCE-FOR-OPEN-WEIGHTS.md` already named for Open
Images / CC12M / RedCaps / YFCC100M — and worse, because the corpus is not frozen: the underlying
videos can vanish or be relicensed after the fact.

**Finding 4 — one corpus changed its terms during this audit's own runtime window, for a reason
specific to the domain: consent revocation.** Mozilla moved Common Voice exclusively to the
Mozilla Data Collective in October 2025, explicitly because a frozen mirrored copy cannot honour a
speaker's later request to be removed. This is not a licence defect in the copyright sense — the
data was and remains CC0 — but it is a live consent-hygiene question this project's verdict
vocabulary has no slot for.

---

# Bucket A — speech with transcripts (auditory input, speech↔text bridge)

Sixteen entries: the fourteen named in the brief plus two 2025 finds (NVIDIA Granary, Loquacious
Set). All fetch dates 2026-09-02.

**The licence column is SPLIT in all three bucket tables** `[S31-11 fixed]`. The first version of
this document carried one `mirror id + tag` column with the upstream terms relegated to the prose
below, which is the exact structural defect the document itself establishes as a mechanical rule —
*record the mirror tag and the upstream text as separate fields* — and it meant three rows
(TED-LIUM #7, Granary #15, Loquacious #16) showed an **upstream** licence string inside the
**mirror** cell. A reader building the fetch manifest from these tables could not populate
`licence_upstream` without leaving the table, and the fact files that had the field are not in
this repo. The tables now hold both fields, which is also what makes this document directly
consumable as the manifest seed the *Consumers of this document* section assigns it. **The mark in
the new column is about the UPSTREAM read specifically** — `(V)` means the upstream's own page was
fetched and quoted this session, `(I)` means it was reconstructed, and an explicit *NOT
INDEPENDENTLY QUOTED* means nobody read it.

| # | dataset | upstream URL | mirror id + tag | **upstream licence (+ mark)** | train-on | redistribute | NC/ND/SA/attr | provenance red flags | size | recommendation |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LibriSpeech | openslr.org/12 | `openslr/librispeech_asr` — `cc-by-4.0` (**VERIFIED**, HF API) | **CC BY 4.0** (**V**, openslr.org/12); grant scope: whole corpus | ATTRIBUTION | clean | attribution | none; LibriVox volunteers dedicate readings to the public domain | ~1,000 h, en | **USE** |
| 2 | LibriTTS / LibriTTS-R | openslr.org/60, openslr.org/141 | `mythicinfinity/libritts`, `mythicinfinity/libritts_r` — `cc-by-4.0` (**VERIFIED**) | **CC BY 4.0** (**V**, openslr.org/60 + /141); whole corpus | ATTRIBUTION | clean | attribution | none; **same LibriVox lineage as #1** | 585 h, 2,456 spk (spk count **INFERRED**) | **USE**, prefer `-R` |
| 3 | Common Voice | commonvoice.mozilla.org/en/datasets → mozilladatacollective.com | `mozilla-foundation/common_voice_17_0` — **no `license` field at all** (**VERIFIED**, HF API) | **CC0** on the recordings (**V**); MDC platform terms forbid re-hosting (**V**) | PERMISSIVE_OK on copyright (CC0); **access unresolved** | **do not re-host** | none on copyright; a platform contract restricts mirroring | crowd-recorded; **active consent revocation a frozen mirror cannot honour** | 31,841 h / 20,789 validated, 100+ langs | **EVAL-ONLY pending operator review** |
| 4 | People's Speech | mlcommons.org/datasets/peoples-speech | `MLCommons/peoples_speech` — six CC tags at once (**VERIFIED**) | **CC BY / CC BY-SA**, split by config (**V**, mlcommons.org); whole corpus | ATTRIBUTION (`cc-by-*`) / SHARE_ALIKE (`cc-by-sa-*`) | clean on `*-clean` configs | attribution + SA, split by config | Archive.org uploader self-declarations; MLCommons' own `dirty` configs admit low confidence | 30,000+ h, 23.7M examples, en | **USE** (`*-clean` configs); `*-dirty` EVAL-ONLY |
| 5 | GigaSpeech | github.com/SpeechColab/GigaSpeech | `speechcolab/gigaspeech` — `apache-2.0` (**VERIFIED**) | NC asserted **and ownership disclaimed** — *"SpeechColab does not own the copyright of the audio files"* (**V** via two independent secondary corroborations) | **BLOCKING** (contested, disclaimed ownership) | BLOCKING | NC asserted, standing disputed | 58.7% podcast + YouTube scraped; maintainer dispute open on the HF thread | 10,000 h transcribed | **REFUSE** |
| 6 | VoxPopuli | github.com/facebookresearch/voxpopuli | `facebook/voxpopuli` — `cc0-1.0` + `other` (**VERIFIED**) | **data: CC0**; Meta's models/code CC BY-NC (**V**, raw GitHub README); grant scope: data only | PERMISSIVE_OK / ATTRIBUTION (data only) | clean (data only) | none on the data; Europarl asks for a source acknowledgement | low; MEPs and interpreters speaking in an official public capacity | 400K h unlabelled + 1.8K h transcribed + 17.3K h interpretations | **USE** (data only; **not** Meta's own CC BY-NC models/code) |
| 7 | TED-LIUM 3 | openslr.org/51 (404 both mirrors; TFDS reproduces the text) | `LIUM/tedlium` — CC BY-NC-ND 3.0 (**INFERRED**, 401 on direct fetch) | **CC BY-NC-ND 3.0** (**V** via the TFDS catalog reproducing the openslr text); whole corpus | **NC + ND — outside the NC-tolerant policy's scope** | BLOCKING | **ND**, plus NC and attribution | none on ownership; TED Conferences LLC is a named consistent rights holder | 452 h train, 2,351 talks, en | **REFUSE pending a distinct operator ruling on ND** |
| 8 | MLS | openslr.org/94; ai.meta.com blog | `facebook/multilingual_librispeech` — `cc-by-4.0` (**INFERRED**, 401) | *"nonrestrictive license"*, LibriVox lineage (**V**, ai.meta.com blog) | PERMISSIVE_OK | clean | none (LibriVox requires none) | none; **same LibriVox lineage as #1/#2** | ~44.5K h en + ~6K h in 7 more langs | **USE** |
| 9 | Libri-Light | github.com/facebookresearch/libri-light | no canonical HF mirror found | repo `LICENSE` covers **CODE ONLY** (**V**, GitHub); **no data grant stated** — grant scope: code_only | PERMISSIVE_OK (**INFERRED** from LibriVox lineage; repo `LICENSE` covers **code only**) | inferred clean | none found | none on rights; **~99% unlabeled** — wrong shape for a transcript bridge | ~60K h | **USE for self-supervised pretrain**; EVAL-ONLY for the transcribed slice |
| 10 | Emilia (+ Emilia-YODAS) | github.com/open-mmlab/Amphion `preprocessors/Emilia` | `amphion/Emilia-Dataset` — CC BY-NC-4.0 core, CC BY-4.0 YODAS subset (**INFERRED**, 401) | **CC BY-NC 4.0** core / CC BY 4.0 YODAS subset, and *"Emilia does not own the copyright to the audio files"* (**V**, GitHub raw README) | **BLOCKING, not NC** | BLOCKING | NC asserted, standing disputed | **HIGH** — in-the-wild scrape of video platforms and podcasts, no consent process described | 101,000 h, 6 langs | **REFUSE** |
| 11 | YODAS / YODAS2 | arxiv.org/abs/2406.00899 | `espnet/yodas2` — `cc-by-3.0` (**INFERRED**, 401) | CC-filtered **at scrape time from uploader self-declarations** (**V**, arXiv HTML) | ATTRIBUTION, caveated | caveated | attribution (BY) | **MEDIUM-HIGH** — CC-filtered at scrape time from *unverified uploader self-declarations* | ~369K h (paper) to 500K+ h (secondary) — **discrepancy unresolved** | **EVAL-ONLY pending a sampling audit** |
| 12 | SPGISpeech | kensho.com post (404; snippets only) | `kensho/spgispeech` — gated click-through EULA (**VERIFIED structurally**) | gated click-through EULA, academic + internal only (**I**, reconstructed from snippets) | **BLOCKING** | BLOCKING | NC-shaped but stricter (academic + internal only) | low on rights — S&P Global is a real named owner; a hard no-re-identification clause | 5,000 h, ~50K spk | **REFUSE** |
| 13 | AMI | groups.inf.ed.ac.uk/ami/corpus/license.shtml | `edinburghcstr/ami` — `cc-by-4.0` (**INFERRED**, 401, but agrees with the fetched upstream) | **CC BY 4.0** (**V**, direct fetch of the licence page — the cleanest fetch in this audit) | ATTRIBUTION | clean | attribution | none; purpose-recorded scenario meetings, consent built into the corpus design | 100 h | **USE** |
| 14 | Switchboard-1 | catalog.ldc.upenn.edu/LDC97S62 | none (LDC corpora are not re-mirrorable) | all rights reserved; LDC membership/fee (**V**, direct fetch) | **BLOCKING** | BLOCKING | n/a — all rights reserved, membership/fee | none on ownership; a straightforward paywall | ~260 h (**INFERRED**) | **REFUSE** |
| 15 | NVIDIA Granary (2025) | arxiv.org/abs/2505.13404 | `nvidia/Granary` — `cc-by-3.0` (**INFERRED**, 401) | **NOT INDEPENDENTLY QUOTED** — PDF unparseable, HF fetch 401 (**UNVERIFIED at upstream**) | ATTRIBUTION, inheriting YODAS's caveat | caveated | attribution (BY) | **MEDIUM-HIGH** — **manifests, not audio**; aggregates YODAS2 and inherits its unverified provenance | ~1M h referenced | **EVAL-ONLY pending YODAS resolution** |
| 16 | Loquacious Set (2025) | arxiv.org/abs/2505.21578 | `speechbrain/LoquaciousSet` — tag not confirmed (**INFERRED**) | compound, none NC/BLOCKING per the curators (**I**, WebSearch-summarised; PDF not fetched) | compound; none NC/BLOCKING per the curators | needs a component table | varies by component | none found, but **the component list was not independently verified** | 25,000 h, en | **USE pending a component-licence-table pull** |

## Upstream licence, verbatim — bucket A

**LibriSpeech (#1), VERIFIED** (openslr.org/12): the page states the licence as **"CC BY 4.0"**
(short identifier; the page does not reproduce the full grant) and that *"The data is derived from
read audiobooks from the LibriVox project."*

**LibriTTS / LibriTTS-R (#2), VERIFIED** (openslr.org/60, openslr.org/141): both state **"CC BY
4.0"**. LibriTTS is *"derived from the original materials of the LibriSpeech corpus"*; LibriTTS-R
is *"identical to those of LibriTTS, with only the sound quality improved"*.

**Common Voice (#3).** The HF card's raw README, **VERIFIED**, now contains no licence text at
all — only:

> *"Effective October 2025, Mozilla Common Voice datasets are now exclusively available through
> Mozilla Data Collective."*

The MDC Terms of Service, **VERIFIED**:

> *"[Data Consumers may not] copy, scrape, download or otherwise acquire any Dataset or portion
> thereof from the Platform for the purpose of... hosting, storing or making the Dataset
> available on any platform, server or repository other than the Platform."*

and Mozilla's own stated reason (community.mozilladatacollective.com FAQ, **VERIFIED**, dated
2025-08-19):

> *"When someone chooses to revoke their consent to be included in a dataset, we need a way to
> remove them... Mozilla community datasets, including Mozilla Common Voice datasets, are
> exclusively available through MDC for this reason."*

The underlying CC0 dedication is, as a copyright matter, irrevocable, and a frozen
pre-October-2025 snapshot arguably remains CC0 regardless of the new contract. **But the consent
question is independent of the copyright question and neither reading solves it**: a static
snapshot cannot honour a revocation that happens after it was taken. This is the strongest single
open item in bucket A precisely because the existing verdict vocabulary has no slot for *"the
licence is fine but the consent regime moved."*

**People's Speech (#4), VERIFIED** (mlcommons.org):

> *"The MLCommons People's Speech dataset is among the world's largest English speech recognition
> corpus today that is licensed for academic and commercial usage under CC-BY-SA and CC-BY 4.0."*

and, on provenance (HF card, **VERIFIED**): *"Data was downloaded via the archive.org API. No data
inference was done."* The `clean`/`dirty` config split is MLCommons' own confidence flag on whether
the per-item uploader-declared licence was reliably matched.

**GigaSpeech (#5), VERIFIED via two independent secondary corroborations** — the primary
`TERMS_OF_ACCESS.md` returned **HTTP 404** at both expected GitHub paths, so this is
verified-by-secondary-source, and closing that gap is itself an action item:

> *"Researcher shall use the Database only for non-commercial research and educational purposes."*
> *"SpeechColab does not own the copyright of the audio files."*

The second sentence moves this out of the NC-tolerant bucket: it is the identical two-clause shape
(restrictive scope + non-ownership disclaimer) this project already treats as BLOCKING for
`zh-plus/tiny-imagenet`. Only the audiobook fraction (2,655 of ~10,000 supervised hours) is
plausibly LibriVox-clean, and unverified even there.

**VoxPopuli (#6), VERIFIED** (raw GitHub README): a licence table — **data: CC0** (pointing at the
European Parliament's legal notice for raw data); **LM data:** check the Europarl site;
**pre-trained models and code: CC BY-NC 4.0**. The Europarl reuse terms, followed independently
(**VERIFIED via WebSearch of europarl.europa.eu**), ask that materials be reused *"by acknowledging
the source © European Union, [year(s)] – EP"* — an attribution condition on the true upstream that
Meta's blanket CC0 tag does not itself carry forward.

**TED-LIUM 3 (#7), VERIFIED via the TFDS catalog page reproducing the openslr text, independently
corroborated by coqui-ai/open-speech-corpora:**

> *"The TED-LIUM corpus is licensed under Creative Commons BY-NC-ND 3.0... All talks and text are
> property of TED Conferences LLC."*

**ND, not merely NC.** CC's own ND clause: *"If you remix, transform, or build upon the material,
you may not distribute the modified material."* The NC-tolerant policy was written to absorb a
*NonCommercial* term by moving the release licence; it says nothing about *NoDerivatives*, which is
a materially stronger prohibition — no redistributed derivative at all, commercial or not. This is
its own operator question, not a GooAQ/Expresso/ESC-50 case.

**MLS (#8), VERIFIED** (ai.meta.com blog): *"it can be released with a nonrestrictive license"*,
sourced from *"public domain audiobooks from the LibriVox project."* LibriVox's own terms
(**VERIFIED via WebSearch**; direct fetch returned 403): *"LibriVox only records material that is
in the public domain... [recordings] can be used however [users] wish, including for commercial
purposes... no need to credit LibriVox, although they prefer if you do."*

**Libri-Light (#9), VERIFIED** (GitHub): the top-level `LICENSE` covers **code only** — *"The
Libri-light code is released under the MIT license."* **No licence statement for the audio data
itself was found anywhere in the repo content fetched.** PERMISSIVE_OK is an **INFERRED**
application of the LibriVox lineage, not a grant Meta stated for this release.

**Emilia (#10), VERIFIED** (github.com raw README):

> *"Users are permitted to use this dataset only for non-commercial purposes under the CC
> BY-NC-4.0 license."*
> *"Emilia does not own the copyright to the audio files; the copyright remains with the original
> owners of the videos or audio."*

A WebSearch snippet (**INFERRED**) additionally reports ImageNet-Terms-of-Access-shaped gating
language on the HF card: *"The researcher shall use the dataset ONLY for non-commercial research
and educational purposes... [and shall] indemnify the authors of Emilia."* The CC BY-NC-4.0 tag is
a licence Emilia's curators had no clear standing to grant over audio they say they do not own.
101,000 hours makes this the largest tempting corpus in the audit and the clearest REFUSE.

**YODAS (#11), VERIFIED** (arXiv HTML): *"the video content must be accompanied by a Creative
Commons license"* — filtering used YouTube's own upload-licence flag at scrape time; the stated
distribution licence is CC BY 3.0. **The paper describes no verification step of any kind** for
whether an uploader's CC self-declaration is true.

**SPGISpeech (#12), INFERRED** (reconstructed from WebSearch snippets of the Kensho post and HF
card; recommend a direct re-fetch before quoting externally):

> *"The Content is provided for academic research purposes and internal use only and must not be
> used to: assemble or create a database; construct or facilitate the construction of products
> which compete with the Content; identify or attempt to identify or contact any individual; or
> link to another dataset."*

"Academic research and internal use only" is narrower than a bare NC term, and *"must not assemble
or create a database"* prohibits exactly the corpus construction `CORPUS-CONTRACT.md` does as
standard practice.

**AMI (#13), VERIFIED** (direct fetch of the licence page — the cleanest fetch in the audit):

> *"The AMI corpus and its annotations are released under the Creative Commons Attribution 4.0
> license agreement (also called CC BY 4.0)."*

with a *"worldwide, royalty-free, non-sublicensable, non-exclusive, irrevocable license"* to
reproduce, share and adapt; attribution required; no commercial restriction.

**Switchboard-1 (#14), VERIFIED** (direct fetch):

> *"© 1992-2026 Linguistic Data Consortium, The Trustees of the University of Pennsylvania. All
> Rights Reserved."*

Access requires the LDC User Agreement plus membership or a fee.

**NVIDIA Granary (#15).** Upstream licence **not independently quoted** (PDF unparseable, HF fetch
blocked); CC-BY-3.0 is **INFERRED** from consistent snippets across independent secondary sources.
Two structural findings, both **VERIFIED** from the HF card description via snippet: *"The
dataset's recordings are sourced from a variety of Creative Commons corpora — such as YODAS2,
YouTube-Commons, VoxPopuli, and Libri-Light"* and *"This repository provides manifests (metadata),
not audio files."*

**Loquacious Set (#16), INFERRED** (WebSearch-summarised paper content, PDF not fetched):

> *"The Loquacious dataset uses existing data and does not impose any new licensing restrictions.
> Each component of the original datasets keeps its original Creative Commons license, all of which
> permit commercial use."*

Good in principle; its verdict is only as good as a component list that was not independently
re-derived — the same discipline `all-nli`'s SNLI/MultiNLI decomposition already required.

---

# Bucket B — general audio and non-speech sound events

Eleven entries: the ten named plus one 2025 lead (DataSEC/DataSED, under-verified).

| # | dataset | upstream URL | mirror id + tag | **upstream licence (+ mark)** | train-on | redistribute | NC/SA/attr | provenance red flags | size | recommendation |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | AudioSet | research.google.com/audioset | `agkphysics/AudioSet` — `cc-by-4.0` (**VERIFIED**) | **CC BY 4.0 over the METADATA/ontology only** (**V**, research.google.com); Google does not license the audio — grant scope: metadata_only | **BLOCKING** (audio mirror) | BLOCKING | SA on the ontology only | **YouTube-scraped; the mirror, not Google, re-hosts audio nobody in the chain owns** | ~2.1M clips, 632 classes | **REFUSE** (audio); EVAL-ONLY for Google's embeddings-only release |
| 2 | FSD50K | zenodo.org/records/4060432 | community mirrors only — **UNVERIFIABLE** this session | **CC BY 4.0** umbrella over per-clip CC0 / CC-BY / CC-BY-NC / **CC Sampling+** (**V**, Zenodo 4060432) | ATTRIBUTION (filtered) / NC (unfiltered) | clean if filtered | attribution + partial NC | none — **per-clip Freesound licence column survives**, the good case | 51,197 clips, ~108.3 h, 200 classes | **USE** (CC0+CC-BY clips) / USE-NC (full) |
| 3 | ESC-50 | github.com/karolpiczak/ESC-50 | `ashraq/esc50` — no structured licence field returned (**UNVERIFIABLE** at the API level) | **CC BY-NC** full set; **CC BY** for the ESC-10 subset (**V**, GitHub README) | NC (full) / ATTRIBUTION (ESC-10) | NC / clean | NC + attribution | none — Freesound-derived, per-clip licensed | 2,000 clips, 50 classes, balanced | **USE-NC** (full) / **USE** (ESC-10) |
| 4 | UrbanSound8K | zenodo.org/records/1203745 | community mirrors — **UNVERIFIABLE** this session | **CC BY-NC** (**V**, Zenodo 1203745 licence field) | NC | NC | NC + attribution | none — Freesound-derived | 8,732 clips, 10 classes | **USE-NC** |
| 5 | Clotho | zenodo.org/records/3490684 | community mirrors — **UNVERIFIABLE** this session | repository `LICENSE` is **NC** (**V**) while Zenodo's own tag says *"Other (Attribution)"* — stricter layer governs | NC | NC | NC + attribution | **Zenodo's own tag says "Attribution"; the LICENSE file is NC** — a two-layer stack where the stricter layer governs | 4,981 clips, 24,905 captions | **USE-NC** |
| 6 | AudioCaps | github.com/cdjkim/audiocaps | `d0rj/audiocaps` — `mit` (**INFERRED**, snippet only) | *"academic purposes only"* (**V for that sentence**, GitHub README partial fetch) | **BLOCKING** | BLOCKING | academic-only (NC-shaped) | **inherits AudioSet's YouTube scrape** | ~46,000 clips, ~57.2K captions | **REFUSE** |
| 7 | WavCaps | github.com/XinhaoMei/WavCaps | `cvssp/WavCaps` — `cc-by-4.0` (**VERIFIED**) | *"Only academic uses are allowed"* (**V**, GitHub) | **BLOCKING** | BLOCKING | academic-only; ~27% also YouTube-scraped | mixed: Freesound ~65%, BBC ~8%, AudioSet ~27%; captions are LLM-generated from metadata | 402,050 clips, ~7,568 h | **REFUSE** |
| 8 | MUSAN | openslr.org/17 | not confirmed this session — **UNVERIFIABLE** | **CC BY 4.0** — *"Attribution 4.0 International"* (**V**, openslr.org/17) | ATTRIBUTION | clean | attribution | **cleanest provenance in the bucket** — direct-licensed music platforms plus PD speech | ~109 h | **USE** |
| 9 | BBC Sound Effects | sound-effects.bbcrewind.co.uk (**fetch blocked**) | none identified | research / education / personal use only (**I** — the primary page refused the fetch this session) | **NC (INFERRED, not verified at the primary source)** | NC, same caveat | NC | BBC-owned (the cleanest provenance *category*) but a research/education/personal-only use restriction | ~33,000 effects (**INFERRED**) | **EVAL-ONLY pending a direct primary re-fetch** |
| 10 | Freesound (policy note) | freesound.org/help/tos_api | n/a | per-clip licensing; **bulk scraping forbidden by ToS** (**V**, policy note) | n/a | n/a | n/a | **bulk scraping forbidden by ToS**; governs #2–#5 | n/a | a standing constraint, not a candidate |
| 11 | DataSEC / DataSED (2025) | Nature *Sci. Data* DOI `s41597-025-05991-w` (**paywalled**) | none identified | **UNCLEAR** — paywalled (**I**, not independently fetched) | **UNCLEAR** | unresolved | unresolved | unknown — not established who recorded the audio | 4,292 + 712 items (**INFERRED**) | **EVAL-ONLY / needs a primary-source follow-up** |

## Upstream licence, verbatim — bucket B

**AudioSet (#1), VERIFIED** (research.google.com/audioset/download.html):

> *"The dataset is made available by Google Inc. under a Creative Commons Attribution 4.0
> International (CC BY 4.0) license, while the ontology is available under a Creative Commons
> Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) license."*

Google's own download page ships **only YouTube video IDs, timestamps and labels — never audio**.
The CC BY 4.0 grant covers the metadata Google authored, not the copyrighted third-party audio it
never touched. The mirror downloaded the actual audio in 2023 (e.g. 1,738,788 of 2,041,789
unbalanced-train clips) and tags the whole re-hosted FLAC corpus `cc-by-4.0` anyway — the identical
pattern already logged for `Nan-Do/code-search-net-python` and for tiny-imagenet, now with a live
real-audio mirror in circulation. **AudioCaps and 108,317 of WavCaps' 402,050 clips (~27%) inherit
this defect by lineage.**

**FSD50K (#2), VERIFIED** (Zenodo record 4060432):

> *"All audio clips in FSD50K are released under Creative Commons (CC) licenses. Each clip has its
> own license as defined by the clip uploader in Freesound, some of them requiring attribution to
> their original authors and some forbidding further commercial reuse."* ... *"FSD50K as a whole is
> the result of a curation process and it has an additional license: FSD50K is released under
> CC-BY."*

The umbrella CC-BY wrapper over a heterogeneous underlying mix is the same shape as the
`all-nli`/`sentence-transformers` wrapper already flagged for text, so it must not be trusted as
the operative licence for the NC-tagged subset without filtering. **Filtering to CC0+CC-BY clips
(34,976 of 40,966 dev rows, 8,403 of 10,231 eval rows — both denominators corrected, and the dev
numerator with them `[S31-10 fixed]`; the eval split is 51,197 − 40,966 = 10,231, not 9,231) is a
mechanical column filter, not a construction project** — the same `repo`-column repair that rescued 71.3% of `code`.

**ESC-50 (#3), VERIFIED** (GitHub README): *"The dataset is available under the terms of the
Creative Commons Attribution Non-Commercial license."* ... *"A smaller subset (clips tagged as
ESC-10) is distributed under CC BY (Attribution)."*

**UrbanSound8K (#4), VERIFIED** (Zenodo 1203745 licence field): *"Creative Commons Attribution Non
Commercial 4.0 International"*. A secondary source states the original 2014 release used CC BY-NC
3.0; the version drift is noted, not resolved.

**Clotho (#5), VERIFIED** (repository `LICENSE` file): *"Any commercial use of the Work or any part
thereof is strictly prohibited."* and *"The original source of this Work, (Audio Research Group) at
Tampere University, is acknowledged in any publication that reports research using this Work."*
**Zenodo's own metadata field reads "Other (Attribution)"** — the umbrella host's tag undersells the
real restriction. This is the mirror-vs-upstream pattern appearing on Zenodo rather than on HF.

**AudioCaps (#6), VERIFIED for this sentence** (GitHub README, partial fetch): *"The code and the
dataset are free to use for academic purposes only."*

**WavCaps (#7), VERIFIED** (GitHub):

> *"Only academic uses are allowed for WavCaps dataset. By downloading audio clips through the
> links provided in the json files, you agree that you will use the audios for research purposes
> only."*

against the HF mirror's `cardData.license: cc-by-4.0`. **This is the cleanest contradiction in the
audit**: unlike GooAQ, the upstream is not internally split — it says one thing while the mirror
tags another, and the mirror is simply wrong.

**MUSAN (#8), VERIFIED** (openslr.org/17): *"Attribution 4.0 International (CC BY 4.0)"*.

**BBC Sound Effects (#9), INFERRED — the primary page was refused by the fetch tool this session
and the text below comes from third-party WebSearch results, not the BBC's own page:** *"released
under a RemArc License, allowing free use for research, educational, and personal projects"* but
*"cannot be sampled in music that is going to be sold"*, with commercial use requiring a separate
purchased licence. **Do not rely on this entry beyond EVAL-ONLY without a direct primary fetch.**

**Freesound (#10), VERIFIED policy note:** per-clip licensing governs #2–#5, and bulk scraping is
forbidden by the ToS. The rule this sets: any Freesound-sourced audio must be filtered per clip
(CC0/CC-BY only), never trusted at a corpus-wide umbrella tag.

**DataSEC / DataSED (#11), INFERRED, NOT independently fetched** — the Nature article page required
authentication. The characterisation (*"released under Creative Commons licenses"*, 4,292 clips /
22 classes, 712 real-world recordings / 4,000+ labels) comes **only from WebSearch snippets**. Do
not rely on it for a licence decision.

---

# Bucket C — TTS and speech-output training material

Seven primary entries plus a 2025 find (HiFiTTS-2). LibriTTS-R and Emilia are recorded in full in
bucket A and cross-referenced here.

| dataset | upstream URL | mirror id + tag | **upstream licence (+ mark)** | train-on | redistribute | NC/SA/attr | consent shape | size | recommendation |
|---|---|---|---|---|---|---|---|---|---|
| LJSpeech | keithito.com/LJ-Speech-Dataset | `keithito/lj_speech` — `unlicense` (**VERIFIED**) | **public domain** — *"There are no restrictions on its use"* (**V**, keithito.com); mirror and upstream agree | PERMISSIVE_OK | clean | none | LibriVox narration, single private individual, ~8 years unchallenged use | 13,100 clips, ~24 h, 1 spk, en | **USE** |
| VCTK 0.92 | datashare.ed.ac.uk/handle/10283/3443 | `CSTR-Edinburgh/vctk` — `cc-by-4.0` (**VERIFIED**) | **CC BY 4.0**, full legal code fetched (**V**, datashare.ed.ac.uk `license_text.txt`) | ATTRIBUTION | clean + attribution | attribution | **purpose-recorded for voice-cloning research**; the licence disclaims moral and personality rights | 110 spk, ~400 sentences each, ~44 h | **USE** |
| Hi-Fi TTS | openslr.org/109 | `MikhailT/hifi-tts` — `cc-by-4.0` (**VERIFIED**) | **CC BY 4.0** (**V**, openslr.org/109, licence-name level) | ATTRIBUTION | clean + attribution | attribution | LibriVox, **repurposed** by audio-quality curation, not purpose-recorded | 10 spk, ~291.6 h, en | **USE** |
| HiFiTTS-2 | none located off-HF this session | `nvidia/hifitts-2` — `cc-by-4.0` (**VERIFIED at the mirror only**) | **NO UPSTREAM PAGE LOCATED** this session — **INFERRED at the upstream-verbatim level**, mirror-only | ATTRIBUTION | clean — **verify upstream first** | attribution | LibriVox, repurposed, at much larger scale | ~36.7K h, ~5,000 spk, 48 kHz | **USE**, verify off-HF before anchoring anything |
| CSS10 | github.com/Kyubyong/css10 | no canonical mirror; `ayousanz/css10-ljspeech-multilingual` unverified | repo badge is **Apache-2.0 CODE** (**V**, GitHub README footer); **no data grant stated** — grant scope: code_only | PERMISSIVE_OK (**INFERRED** from LibriVox lineage, **not** from the repo's Apache-2.0 badge) | likely clean (**INFERRED**) | none found | LibriVox, PD texts, 1 spk per language | 10 langs, 1 spk each; hours **UNVERIFIED** | **USE** with the caveat carried into the model card, or EVAL-ONLY if an explicit data licence is required |
| M-AILABS | github.com/imdatceleste/m-ailabs-dataset | `gigant/m-ailabs_speech_dataset_fr` — **`cc`** (**VERIFIED**; a generic tag, and a mismatch) | **BSD-3-Clause-shaped, not CC at all**, permitting *"any commercial use"* with notice retention (**V**, GitHub README); Ukrainian subset *"for machine learning purposes only"* | PERMISSIVE_OK (8 langs) | clean, notice retention | notice, **not** CC | LibriVox PD; **the Ukrainian subset is "for machine learning purposes only"** | ~1,000 h, 9 langs | **USE** (8 langs) / **EVAL-ONLY or REFUSE** (Ukrainian) |
| AISHELL-3 | openslr.org/93 | `shenyunhang/AISHELL-3` — `apache-2.0` (**VERIFIED**) | **Apache License v.2.0** (**V**, openslr.org/93, licence-name level); matches the mirror | PERMISSIVE_OK | clean | none | **purpose-recorded for TTS**, professionally annotated (consent process **INFERRED** from the production shape) | 218 spk, ~85 h, 88,035 utterances, Mandarin | **USE** |
| Expresso | speechbot.github.io/expresso | `ylacombe/expresso` — `cc-by-nc-4.0` (**VERIFIED**) | **CC BY-NC 4.0** (**V**, speechbot.github.io/expresso); mirror and upstream agree | **NC** | NC-tier release | NC + attribution | 4 professional performers, 26 expressive styles (performer agreements **INFERRED**) | 40 h (11 h read + 30 h improvised), 4 spk | **USE-NC** |

## Upstream licence, verbatim — bucket C

**LJSpeech, VERIFIED** (keithito.com): *"This dataset is in the public domain in the US (and most
likely other countries as well). There are no restrictions on its use."* Mirror and upstream agree
— one of the few places in this audit where they do.

**VCTK, VERIFIED** (`license_text.txt` fetched from datashare.ed.ac.uk): the full CC BY 4.0 legal
code, granting *"a worldwide, royalty-free, non-sublicensable, non-exclusive, irrevocable
license"* and stating *"Moral rights and personality rights are not licensed."* The corpus's own
name — *"CSTR Voice Cloning Toolkit"* — is the strongest consent story in the audit: its 110
speakers were recorded **specifically** for TTS and voice-adaptation research, not repurposed from
ASR. The personality-rights clause belongs in the model card verbatim: CC BY covers the copyright
in the recording, not a blanket right to synthesise any of those 110 voices for an unrelated
purpose.

**Hi-Fi TTS, VERIFIED** (openslr.org/109, licence-name level): **"CC BY 4.0"**, over material that
*"derives from public audiobooks available through LibriVox and written works sourced from Project
Gutenberg. Both underlying sources provide content in the public domain."*

**HiFiTTS-2:** `cc-by-4.0` **VERIFIED at the mirror only**; no independent upstream page was
located this session. Mark this entry **INFERRED at the upstream-verbatim level**, unlike the rest
of this bucket. Given that ~36.7K hours would dominate any TTS mix it joins, the upstream fetch is
a prerequisite to sizing anything against it.

**CSS10, VERIFIED as a repo-level badge** (GitHub README footer): **"Apache-2.0 license."** Per
this project's own CodeSearchNet finding, a repo-level GitHub badge is evidence about the
alignment/processing **code**, not necessarily about the **audio data**. The README's actual
provenance claim is *"All audio originates from LibriVox audiobooks — public domain works read by
single speakers."* PERMISSIVE_OK therefore rests on the LibriVox inference, **not** on the badge.

**M-AILABS, VERIFIED** (GitHub README): *"Redistributions of source data must retain the above
copyright notice, this list of conditions and the following disclaimer."* ... *"Neither the name of
the copyright holder nor the names of its contributors may be used to endorse or promote products
derived from this downloaded data..."*, explicitly permitting *"Redistribution and use in any form,
including any commercial use, with or without modification."* **This is BSD-3-Clause-shaped, not
Creative Commons at all** — the mirror's `cc` tag is a new mismatch. The Ukrainian audio is
explicitly excepted: *"kindly provided either by Nash Format or Gwara Media for machine learning
purposes only"* — a narrower grant that must not be folded into the PERMISSIVE_OK verdict.

**AISHELL-3, VERIFIED** (openslr.org/93, licence-name level): **"Apache License v.2.0"**, matching
the mirror. Purpose-recorded: *"specifically designed as a multi-speaker Mandarin speech corpus for
training Text-to-Speech systems."* **The strongest non-English, non-LibriVox candidate in the whole
audit**, and one of only three genuinely independent-lineage TTS sources (with VCTK and Expresso).

**Expresso, VERIFIED** (speechbot.github.io/expresso): *"The Expresso dataset is distributed under
the CC BY-NC 4.0 license."* Mirror and upstream agree. **The only expressive/paralinguistic-labelled
TTS source found anywhere in this audit** — 26 emotional and performance styles. A speech-output
head trained only on neutral LibriVox narration would never learn to modulate tone; Expresso is the
one clean lever for that, at the cost of an NC term.

**Emilia, cross-referenced from bucket A, is a poor bucket-C candidate independently of its licence
problem.** Its speakers are incidental subjects of scraped podcasts and interviews who consented to
publication, not to voice-cloning research — the opposite consent shape from VCTK and AISHELL-3.

---

# Cross-bucket patterns

**1. LibriVox needs the same lineage-collision treatment `CORPUS-CONTRACT.md` §2.3 Stage 0 already
gives Wikipedia, ImageNet, Flickr30k and GooAQ.** LibriSpeech, LibriTTS/-R, MLS, Libri-Light,
LJSpeech, CSS10, M-AILABS, Hi-Fi TTS and HiFiTTS-2 all draw on the same pool. Consequences for a v1
mix: (a) for **item-level disjointness**, a compose-eval item drawn from one of these may share a
speaker or the exact recording with training material drawn from another; (b) for **B1/B2 balance**,
these are **ONE provenance group** and must be capped as one, or the region is a LibriVox monoculture
wearing ten labels. **This was not checkable from any single dataset's page — it only became visible
by reading all sixteen speech entries side by side.**

**2. Five corpora disclaim ownership of their own audio** — AudioSet's real-audio mirror, AudioCaps
and WavCaps' AudioSet fraction (inherited), GigaSpeech, and Emilia. **This is not the same failure
as GooAQ and must not be folded into the NC-tolerant policy.** GooAQ's contradiction was between
two statements from a party with a real claim; these five have no party in the chain asserting a
real claim at all.

**3. "Manifest-only" distribution is a genuinely new pattern for this project.** Granary and YODAS
ship YouTube video-ID pointers, not audio — the audio instance of the "URLs, not pixels" evasion
already named for Open Images / CC12M / RedCaps / YFCC100M (RedCaps' own words there: *"We do not
distribute image files as we do not legally own them"*). It is worse in audio because the corpus is
not even frozen: videos can vanish or be relicensed after the fact. That is a licence-**stability**
problem, independent of the licence-**grant** question, and this project's vocabulary has no
category for it yet.

**4. Seven new mirror-vs-upstream mismatches, extending the running tally past the ten already on
record for text and vision** (`memory/dataset-mirror-licences-lie.md`):

| # | mirror | mirror says | upstream says |
|---|---|---|---|
| 11 | `agkphysics/AudioSet` | `cc-by-4.0` over re-hosted audio | Google's CC BY 4.0 covers **metadata only** |
| 12 | `d0rj/audiocaps` | `mit` (**INFERRED**, snippet) | *"free to use for academic purposes only"* |
| 13 | `cvssp/WavCaps` | `cc-by-4.0` | *"Only academic uses are allowed"* |
| 14 | Clotho on Zenodo | Zenodo tag *"Other (Attribution)"* | the `LICENSE` file is **NC** |
| 15 | `speechcolab/gigaspeech` | `apache-2.0` | NC **plus** a non-ownership disclaimer, contested on the record |
| 16 | `gigant/m-ailabs_speech_dataset_fr` | `cc` (generic, meaningless) | BSD-3-Clause-style, explicit commercial permission |
| 17 | CSS10 GitHub badge | `Apache-2.0` | a **code** badge; the data's grant, if any, comes from LibriVox |

**Running tally: 17.** Two of the seven (#12, and the mirror tags behind #15's dispute thread) rest
on snippets rather than direct API reads and are marked **INFERRED** accordingly.

**5. Consent, not copyright, is the open question with no existing category.** Common Voice's move
to MDC exists specifically because a frozen mirror cannot honour a speaker's consent revocation — a
live mechanism this project cannot respect against a static snapshot, whatever the copyright licence
says. Nothing in the prior text and vision audits produced this shape, because those corpora do not
carry an ongoing biometric-identity relationship with their contributors the way voice recordings
structurally do.

---

# Recommended v1 `auditory` corpus mix, against `CORPUS-CONTRACT.md`'s balance rules

**Method.** Provenance groups first — the LibriVox correction is applied **before anything else**,
as Stage 0 requires (*"reject a compose candidate whose upstream lineage intersects... regardless of
whether text overlaps"*). Then B1–B5 are checked or explicitly marked uncheckable.

## Provenance groups

| provenance group | members | approx. scale | licence tier |
|---|---|---|---|
| **LibriVox — ONE group** | LibriSpeech, LibriTTS/-R, MLS, Libri-Light, LJSpeech, CSS10, M-AILABS (8 langs), Hi-Fi TTS, HiFiTTS-2 | ~1,000 h to ~60,000 h per member; **do not sum naively — speaker and recording overlap across members is plausible and unmeasured** | ATTRIBUTION / PERMISSIVE_OK |
| **Europarl / VoxPopuli** | VoxPopuli only | 400,000 h unlabelled + 1,800 h transcribed | ATTRIBUTION (data) |
| **Archive.org / People's Speech** | People's Speech `*-clean` configs | 30,000+ h (clean-config share not separately measured) | ATTRIBUTION / SHARE_ALIKE, filterable |
| **Purpose-recorded research corpora** | AMI, VCTK, AISHELL-3 | 100 h + 44 h + 85 h | ATTRIBUTION / PERMISSIVE_OK |
| **Freesound — ONE group** | FSD50K, ESC-50/ESC-10, UrbanSound8K, Clotho's audio layer | 51,197 + 2,000 + 8,732 + 4,981 clips | mixed ATTRIBUTION/NC; filterable at FSD50K only |
| **Independent music / PD speech** | MUSAN | 109 h | ATTRIBUTION |
| **Professional-actor expressive** | Expresso | 40 h, 4 speakers | NC |

## B1 — max single-source share ≤ 0.40

**Cannot be computed precisely from this audit** — several member sizes are INFERRED or
approximate, and HiFiTTS-2's 36,700 h in particular would dominate any naive sum on a
mirror-only-verified figure. **Directionally:** if the LibriVox group is used near its uncapped
scale (Libri-Light alone is ~60,000 h), it fails B1 on its own the same way tiny-imagenet and
CodeSearchNet-Python did — unless VoxPopuli's 400,000 unlabelled hours are drawn on at comparable
scale, in which case VoxPopuli becomes the new dominant source instead. **A capped design is
required from the start**, the same conclusion `CORPUS-CONTRACT.md` reached for `vl_latent`: cap
each provenance group to a common ceiling rather than use any source at its natural size. Sizing
that ceiling against real disk and compute budgets is work this audit did not do.

## B2 — effective number of sources (inverse Simpson) ≥ 3

**Directionally achievable, not computed.** Seven provenance groups exist counting LibriVox as one.
A uniform-ish cap across five or six of them (dropping Expresso's 40 h as too small to move
`N_eff`) lands comfortably above 3 — a much healthier starting position than `code`, `retrieve`,
`compress`, `vl_latent` or the two planned regions, every one of which currently fails B1 outright.
**`auditory` can be built balanced from day one, if the LibriVox-as-one-group correction is applied
before the first cap is chosen. That is the single highest-leverage design decision available here.**

## B3 — train/eval distribution correspondence declared and implemented

**Cannot be checked: there is no code yet.** This is the correct state of a placeholder, but it is
exactly where `retrieve`'s B3 violation was found — a comment promising transfer testing the code
never implemented. Procedural recommendation: **whoever writes `auditory`'s training script must
declare `in-mixture` vs `held-out-domain` in the code, not a comment, before the first receipt.**

## B4 — a cap must be a sample, not a prefix

**Cannot be checked, same reason.** `retrieve`'s `load_pairs` prefix-truncation bug (GooAQ's
400,000-row cap was the first 13.3% of the corpus in file order) is a documented anti-pattern in
this tree. `auditory` has no code to inherit it, which is the point at which not to write it.

## B5 — within-source concentration

**Partially checkable, unevenly.** ESC-50 is class-balanced by construction (40 clips/class, 1:1
max:min — passes trivially). VCTK's ~400 sentences across 110 speakers is roughly uniform by design
(passes on inspection, not recomputed). FSD50K's per-clip licence column is itself a usable stratum
key — **FOUR tiers, not three, and the shares are derived rather than quoted** `[S31-10 fixed]`.
Recomputed from the raw per-tier counts in `facts/audio-bucket-B.md:33` (dev CC0 14,959 · CC-BY
20,017 · CC-BY-NC 4,616 · CC Sampling+ 1,374; eval CC0 4,914 · CC-BY 3,489 · CC-BY-NC 1,425 ·
CC Sampling+ 403), combined dev+eval: **CC0 19,873 (38.8%), CC-BY 23,506 (45.9%), CC-BY-NC 6,041
(11.8%), CC Sampling+ 1,777 (3.5%)** — total 51,197, which matches this audit's own corpus size
and sums to 100%. The earlier figures in this document (CC0 36.5%, CC-BY 57.3%, CC-BY-NC 11.3%)
summed to **105.1%**, which is impossible for a one-licence-per-clip column, and they silently
dropped the **CC Sampling+ tier**. That tier is a **fourth, separate restriction**, not a slice of
the NC one: phrasing the filter as *"CC0+CC-BY vs NC"* implies the remainder is NC when 3.5% of it
is a different restriction again, and the CC0+CC-BY filter excludes it too. **The LibriVox group's
per-speaker and per-book concentration was not measured and is the largest unknown in this
section** — LibriVox skews toward a small number of prolific volunteer readers, and if any one
reader dominates the pooled hours that is a real B5 failure hiding inside a group otherwise treated
as clean-and-large. **This needs a per-speaker hour histogram before the group is used at scale.
Not done; flagged as undone rather than assumed clean.**

## What this audit could not determine

1. Exact per-member hour counts sufficient to compute B1/B2 to `CORPUS-CONTRACT.md`'s precision.
2. Whether LibriVox's members share *specific recordings*, not just the same speaker pool — e.g.
   whether LibriTTS-R's utterances are a strict superset of LibriSpeech's underlying audio.
3. LibriVox's own per-speaker concentration (B5).
4. People's Speech's exact `*-clean`-config hour breakdown.
5. Whether HiFiTTS-2's ~36,700-hour figure, if independently confirmed, single-handedly dominates
   any mix it joins.

---

# What a `speech_output` head can be trained on under the NC-tolerant policy

**Clean tier, no NC, sufficient for a flat neutral-narration head:** LJSpeech, VCTK,
Hi-Fi TTS/HiFiTTS-2, CSS10, M-AILABS (8 non-Ukrainian languages), AISHELL-3 — English, Mandarin and
nine further languages, with two genuinely purpose-recorded, independent-lineage anchors (VCTK,
AISHELL-3) outside the LibriVox monoculture.

**The one NC addition, and what it buys:** **Expresso** (CC BY-NC 4.0, VERIFIED at both ends, no
contradiction) is the only expressive-style-labelled TTS source in the audit. Without it the head
learns to read neutral audiobook prose and nothing else.

**What including it costs, priced the way `LICENCE-FOR-OPEN-WEIGHTS.md` priced the GooAQ decision:**
a `speech_output` head trained on Expresso moves to CC BY-NC 4.0 **if it ever ships as a standalone
checkpoint**. **At the composed-model level it costs nothing**: per DEC-31 the composed model already
inherits CC BY-NC-SA 4.0 from `memory` (GooAQ), and CC BY-NC-SA is already stricter than plain CC
BY-NC. **The NC-tolerant policy has already been spent by `memory`; `auditory` and `speech_output`
use it for free at the composed tier and pay a real, separate cost only for a standalone release.**

---

# Licence tier this implies, per region and for the composed model

Extending `LICENCE-FOR-OPEN-WEIGHTS.md`'s Recommendation and Decision-2026-09-02 tables with the two
rows this audit adds.

| region | standalone release licence | why |
|---|---|---|
| `code`/`language_code`, `classify`, `reason`/`reasoning`, `vl_latent`/`visual` | MIT (existing finding, unchanged) | no NC/SA/BLOCKING input once `code` is GitHub-licence-filtered |
| `compress` | CC BY-SA 4.0 (existing finding, unchanged) | SNLI-derived, share-alike |
| `retrieve` / `memory` | CC BY-NC-SA 4.0 (existing finding, unchanged) | GooAQ's accepted NC reading plus NQ/FiQA's CC BY-SA |
| **`auditory` (clean-tier build)** | **MIT / CC BY** | LibriVox + VoxPopuli + AMI + Freesound(filtered) + MUSAN carry no NC and no BLOCKING term. **This is achievable and is the recommended v1 posture.** |
| **`auditory` (with FSD50K's CC-BY-NC clips, full ESC-50, UrbanSound8K, or Clotho)** | **CC BY-NC 4.0** | those four are genuinely NC at a real rights holder's own grant — clean NC, not BLOCKING |
| **`speech_output` (clean tier)** | **MIT / CC BY** | LJSpeech, VCTK, Hi-Fi TTS, CSS10, M-AILABS (8 langs), AISHELL-3 carry no NC and no BLOCKING term |
| **`speech_output` (with Expresso)** | **CC BY-NC 4.0** | the only expressive-style TTS source; a standalone-checkpoint cost only |
| **composed model** | **CC BY-NC-SA 4.0, unchanged** | already the strictest term via `memory`; every NC choice audio adds is absorbed at zero additional cost |

**The practical upshot.** If `auditory` or `speech_output` ever ship standalone, the NC-vs-clean
choice for each is a real, separate decision with a real cost. If they only ever ship inside the
composed model, the composed model is NC-tier regardless, and Expresso, ESC-50, UrbanSound8K,
Clotho and FSD50K's NC clips can all be included without moving the composed release's licence at
all. **The NC-tolerant policy's marginal cost at the composed-model level is already zero.**

**Two things this does not touch.** TED-LIUM's **ND** term is outside the NC-tolerant policy
entirely and needs its own ruling — ND is not NC. Common Voice's consent-revocation problem is a
**consent-hygiene** class, not a licence class, and no release-licence choice resolves it.

---

# Open questions for the operator

1. **GigaSpeech.** This audit recommends REFUSE (disclaimed ownership, not just NC) rather than
   USE-NC or "email the maintainer" the way GooAQ was resolved — a non-ownership disclaimer makes an
   email less likely to produce a usable answer than AI2's was. Contact SpeechColab anyway, or
   accept REFUSE?
2. **TED-LIUM 3's CC BY-NC-ND term.** Outside the NC-tolerant policy's scope. Should ND corpora ever
   be usable under this project's posture, even eval-only?
3. **Common Voice.** EVAL-ONLY recommended pending review of the MDC access contract and the
   consent-revocation-vs-frozen-mirror tension. (a) fetch fresh via MDC and accept its terms,
   (b) rely on a held frozen snapshot under the "CC0 is irrevocable" reading, or (c) drop it?
4. **Emilia.** REFUSE recommended despite 101,000 hours and its own CC BY-NC-4.0 tag, because the
   distributor disclaims ownership. Confirm this reading is accepted rather than silently worked
   around later because of the scale.
5. **YODAS / YODAS2 / Granary.** EVAL-ONLY pending a manual sampling audit of uploader
   self-declarations (pull N random video IDs, check by hand). Worth funding given the 500,000+ hour
   payoff if it clears?
6. **AudioSet's metadata-only release** (VGGish embeddings, genuinely CC BY 4.0 upstream) — worth
   using as an auxiliary signal, or skip entirely given the region needs spectrogram input?
7. **BBC Sound Effects.** The recommendation rests on secondary sources only; the primary page was
   blocked. Needs a direct re-fetch before it is relied on beyond EVAL-ONLY.
8. **DataSEC/DataSED and the Loquacious Set.** Both promising 2025 finds, both under-verified
   (paywalled paper, snippet-only). Worth a dedicated follow-up pass before either anchors anything.
9. **HiFiTTS-2's 36,700-hour figure** is mirror-tag-verified only. Get a direct upstream fetch before
   sizing anything against it, since it would dominate any mix it joins.
10. **The derivative-work question** — does a CC BY-SA or CC BY-NC training input make trained
    weights "adapted material" — is exactly as unsettled for audio as for text and vision. Inherited,
    not re-litigated here.
11. **The LibriVox-as-one-provenance-group correction** is this audit's own judgment call, not
    something any dataset's licence page states. It deserves explicit sign-off before it becomes
    load-bearing for a B1/B2 cap design, given how much of the easy clean volume sits inside that
    one group.
12. **Per-speaker concentration within LibriVox (B5)** was not measured. If a few prolific readers
    dominate the pooled hours, that is a real balance defect hiding inside a group this audit
    otherwise treats as clean.

---

# Source files

Full per-dataset records with complete verbatim quotes and live fetch traces are held in the
session's fact files (not in this repo):

- Bucket A1 — LibriSpeech, LibriTTS/-R, Common Voice, People's Speech, GigaSpeech, VoxPopuli.
- Bucket A2 — TED-LIUM 3, MLS, Libri-Light, Emilia, YODAS, SPGISpeech, AMI, Switchboard, Granary,
  Loquacious Set.
- Bucket B — AudioSet, FSD50K, ESC-50, UrbanSound8K, Clotho, AudioCaps, WavCaps, MUSAN, BBC Sound
  Effects, the Freesound policy note, DataSEC/DataSED.
- Bucket C — LJSpeech, VCTK, Hi-Fi TTS, HiFiTTS-2, CSS10, M-AILABS, AISHELL-3, Expresso.

**Consumers of this document:** `REGION-TAXONOMY-AND-INTERCONNECT.md` §1.3 (DEC-43, DEC-44),
§4.1 rows A0–A3, §5.7 (DEC-45), §8 (OD-9 answered; OD-10 to OD-15 opened) and
`LICENCE-FOR-OPEN-WEIGHTS.md`'s per-region tier table.
