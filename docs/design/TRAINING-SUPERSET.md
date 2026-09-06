# The CSD Training Superset

**Status:** design. Nothing here is implemented. No dataset was fetched, no catalogue entry
was changed, no pipeline code was written, and no GPU was touched.
**Scope:** a versioned, publishable dataset superset covering every CSD training phase,
composed from licence-cleared sources plus the operator's own collected and generated data.

> **Neither the author of this document nor its reader is a lawyer.** Everything below
> reports what a licence *says* and what the *risk* is. Where a question is legally
> unsettled it is marked unsettled and left for a human.

**Evidence base.** Every verdict here is derived from records already in this repository —
`docs/design/LICENCE-FOR-OPEN-WEIGHTS.md` (the 2026-09-02 audit), the `CATALOGUE` in
`scripts/csd-corpus-expand.py`, `program/REMAINING.md` P2.4/P2.5, and the measured overlap
and balance figures in `/mnt/bulk/csd-corpus-analysis/analysis.json`. **No new licence
verification was performed for this document** (the session's search budget was exhausted),
so anything not already on record is marked **unverified** rather than asserted. That
marking is not decoration: several entries below turn on it.

---

## The short version

**A fully redistributable superset is not achievable from the currently-cleared set. The
artifact must be manifests plus a build pipeline — the recipe, not the meal — with a small
shipped-bytes core.** That is the honest conclusion and it is the useful one, because it is
also the *cheaper* artifact and the one that survives a removal.

The decisive numbers:

| measure | figure |
|---|---|
| corpora the four trained regions actually use | 4 |
| of those, redistributable | **0** |
| catalogued + fetched corpora (12 entries, 6.2M rows, 20 GB) | 12 |
| of those, redistributable as bytes with no share-alike | **5** |
| of those, redistributable but share-alike (would licence the whole distribution) | 7 |
| of those, fetch-only pending an unresolved sub-licence | 2 (APPS, CodeContests) |
| largest single candidate constituent (gooaq, 3.0M rows) | **contested — cannot ship** |
| vision replacement candidates with clean verified upstreams | 11 permissive + 2 share-alike |
| licence-cleared constituents for the foundation/pretraining phase | **0** |

Read those two ways and both readings matter:

- **By constituent count**, a mostly-redistributable superset looks achievable — 16 clean
  permissive constituents exist on record.
- **By rows the project has actually trained on, it is 0%.** And the clean constituents are
  overwhelmingly *aspirational*: the clean vision set is not fetched, the clean retrieval mix
  is not built, and the foundation phase has no cleared constituent at all.

Shipping bytes also makes every unresolved provenance question **harder**, never easier. A
question that is merely uncomfortable when you train on something ("may I use this?") becomes
a claim you are making to strangers when you redistribute it ("I grant you the right to use
this"). The two cases where this project's exposure is largest — GooAQ's self-contradicting
upstream and CodeSearchNet's dropped per-repository licence data — are both of that shape.

**So the artifact is:**

```
csd-superset/<MAJOR>.<MINOR>.<PATCH>
├── manifests/          every constituent, phase-assigned, licence-verdicted   [ships]
├── notices/            NOTICES and attribution for everything shipped         [ships]
├── reserve/            row-id lists only, for eval sets that cannot ship      [ships]
├── bytes/              operator-owned + CC0 + small permissive text sets      [ships]
├── recipe/             the builder: fetch -> polish -> assemble -> verify     [ships]
└── tombstones/         what was removed, when, why, and by whose authority    [ships]
```

Everything ships except the bytes of constituents that either may not be redistributed or
would contaminate the distribution's licence. The builder reconstitutes the whole on the
consumer's machine and verifies it reproduces the recorded fingerprint. **The row-id list is
the shippable form of a non-shippable dataset**, and that single mechanism does triple duty
here: it publishes an eval set that cannot be published, it records a removal without
retaining the removed content, and it lets a third party prove they built the same thing.

---

## 1. Training rights are not redistribution rights

### The three verdicts

The catalogue's existing gate asks *may I train on this*, and refuses anything that is not
`TRAIN_OK` with no override flag. The open-weights audit asked the second question, *may a
model derived from this be redistributed*. This document asks the third, and it is different
from both:

> **May I hand this data itself to someone else?**

| verdict | meaning | test |
|---|---|---|
| `REDISTRIBUTABLE` | A licence grants redistribution, from a party entitled to grant it, and its conditions are satisfiable at dataset scale. | Can I name the grant, name the grantor, and actually perform every condition? |
| `FETCH-ONLY` | Usable, but the bytes must be fetched by the end user rather than shipped by us. | Two distinct causes — see below. |
| `NEITHER` | No redistribution grant exists, or the grant is contested, or its conditions cannot be performed. | Nothing to rely on. |

**`FETCH-ONLY` has two causes and they must not be conflated:**

- **`FETCH-ONLY (LEGAL)`** — we may not ship it. The grant is absent, contested, or its
  conditions cannot be discharged at scale.
- **`FETCH-ONLY (POLICY)`** — we legally may ship it, and choose not to, because shipping it
  would impose its licence on the whole distribution. **Every share-alike constituent is in
  this class.** This is the single most consequential design decision in the document and
  §3 works it out.

`NEITHER` is not a claim that use is unlawful — same discipline as the audit. It is a claim
that **nothing grants the right being relied on**, which is easier to establish and is the
honest state of an unlicensed corpus.

### Classification — corpora the four trained regions actually use

These are the ones that matter most, because they are the ones a published model has learned
from. Verdicts on the left are already on record; the redistribution column is new.

| corpus | region | recorded open-weights verdict | **REDISTRIBUTION** | why |
|---|---|---|---|---|
| `Nan-Do/code-search-net-python` | `code` | BLOCKING | **NEITHER** as shipped | The mirror asserts `apache-2.0` over an upstream tagged `other`, with no LICENSE file, and it **dropped CodeSearchNet's own `_licenses.pkl` per-repository licence data**. ~15% of rows are GPL/AGPL. Redistributing GPL'd source requires conditions (licence text, no further restrictions) that a parquet shard with no per-row licence column cannot perform. Promotable to `FETCH-ONLY (LEGAL)` after the per-repo filter; see below. |
| `sentence-transformers/all-nli` | `compress` | BLOCKING | **NEITHER** | No licence at the mirror; 0 of 96 `sentence-transformers` datasets carry one. 41.8% is MultiNLI-derived and the OANC EULA's redistribution permission is conditioned on displaying its own agreement text and on per-source attribution *"as specified in Appendix I"* — which is unperformable because the `genre`, `promptID` and `pairID` columns were all destroyed. **You cannot attribute what you can no longer identify.** |
| `sentence-transformers/gooaq` → `allenai/gooaq` | `retrieve` | BLOCKING (contested) | **NEITHER** pending AI2 | The upstream LICENSE file is stock Apache-2.0; the upstream README says *"This dataset should not be used for any commercial purposes."* Redistribution is worse here than training: under the README reading, shipping it is distribution of NC-restricted material, and under the LICENSE reading it is fine. **Do not resolve this by picking the convenient reading.** 77.8% of `retrieve`. |
| `sentence-transformers/natural-questions` | `retrieve` | SHARE_ALIKE | `REDISTRIBUTABLE` → **FETCH-ONLY (POLICY)** | CC BY-SA 3.0, upstream agrees. Legally shippable. Not shipped, because it would make the whole distribution share-alike, and its 3.0 does not sit cleanly beside the 4.0 constituents (§3). |
| fiqa-pairs (`BeIR/fiqa` ⋈ `BeIR/fiqa-qrels`) | `retrieve` | SHARE_ALIKE | **FETCH-ONLY (POLICY)**, contingent | CC BY-SA 4.0 at both ends. **But** if FiQA is StackExchange-derived — recorded as unresolved in the audit — Stack Exchange's attribution is per-item (link the original question, name each author, link each profile) and is **unsatisfiable in principle** from training data. If that resolves against us it becomes `NEITHER`. **Unverified.** |
| `zh-plus/tiny-imagenet` | `vl_latent` | BLOCKING | **NEITHER** | No licence anywhere in the chain; the only terms text present is ImageNet's non-commercial Terms of Access, and ImageNet itself disclaims owning the images. 100% of the region. There is no clean fraction. |
| `nlphuji/flickr30k` (planned P3.2) | `vl_latent` | BLOCKING | **NEITHER** | The distributor states in four places that it does not own the images' copyright. There is no licensor to name. |

**Zero of seven.** That is the finding this section exists to produce. A superset assembled
from what CSD has actually trained on would ship nothing.

### Classification — catalogued and fetched (12 entries, 20 GB, 6.2M rows)

| corpus | region | licence | recorded verdict | **REDISTRIBUTION** | conditions if shipped |
|---|---|---|---|---|---|
| `openai/gsm8k` (`main`) | reason | MIT | PERMISSIVE_OK | **REDISTRIBUTABLE** | Reproduce copyright + permission notice |
| `deepmind/aqua_rat` (`raw`) | reason | Apache-2.0 | PERMISSIVE_OK | **REDISTRIBUTABLE** | Licence copy, modified-file notice, propagate NOTICE if present |
| `google-research-datasets/go_emotions` | classify | Apache-2.0 | PERMISSIVE_OK | **REDISTRIBUTABLE** | as above |
| `zalando-datasets/fashion_mnist` | vl | MIT | PERMISSIVE_OK | **REDISTRIBUTABLE** | notice |
| `PolyAI/banking77` | classify | CC BY 4.0 (GitHub LICENSE agrees) | ATTRIBUTION | **REDISTRIBUTABLE** | TASL attribution + **indicate modification** |
| `codeparrot/apps` | code | MIT tag | PERMISSIVE_OK *with a provenance caveat* | **FETCH-ONLY (LEGAL)** — recommended | The MIT tag is a repackager's tag over third-party competition-problem text. **This is the same shape as `hendrycks/competition_math`, which this project REJECTED** on the reasoning that *"the MIT tag covers the repo's scripts, not the problem text"*. Training on it under the recorded verdict is one decision; reshipping the problem text is a stronger claim. **Operator decision; recommend not shipping.** |
| `deepmind/code_contests` | code | CC BY 4.0 | ATTRIBUTION *with an unresolved sub-licence* | **FETCH-ONLY (LEGAL)** — recommended | DeepMind's CC BY 4.0 sits over three origins; the card attributes Description2Code to MIT and CodeNet to Apache-2.0 and lists the **Codeforces problem statements with no licence stated**. A CC licence conveys only what the licensor holds. Same call as APPS. |
| `rajpurkar/squad` | retrieve | CC BY-SA 4.0 | SHARE_ALIKE | **FETCH-ONLY (POLICY)** | Legally shippable; excluded to keep the union clean. Wikipedia passages carry Wikimedia's own reuse form. |
| `BeIR/hotpotqa` (`corpus`) | retrieve | CC BY-SA 4.0, upstream agrees | SHARE_ALIKE | **FETCH-ONLY (POLICY)** | as above |
| `BeIR/hotpotqa` (`queries`) | retrieve | CC BY-SA 4.0, upstream agrees | SHARE_ALIKE | **FETCH-ONLY (POLICY)** | as above |
| `stanfordnlp/snli` | compress | CC BY-SA 4.0 (Stanford's assertion) | SHARE_ALIKE | **FETCH-ONLY (POLICY)**, contingent | Share-alike anyway, so the policy exclusion decides it. Record that the grant is **single-sourced and uncorroborated**: no Flickr30k source confirms the caption text is CC BY-SA, and SNLI contains ~4,000 CC BY 4.0 Visual Genome premises its own documentation does not acknowledge. Also adds zero pairs over all-nli. |
| `timm/oxford-iiit-pet` | vl | CC BY-SA 4.0, upstream confirmed | SHARE_ALIKE | **FETCH-ONLY (POLICY)** | Clean chain, wrong licence for this distribution. |

**Refused entries kept in the catalogue on purpose** — `code-search-net/code_search_net:go`
(`other`), `BeIR/scifact` (NC upstream), `hendrycks/competition_math` (live DMCA):
**`NEITHER`**, and they stay in the manifest set with that verdict for the same reason the
catalogue keeps them: *a rejection nobody can see is a rejection that gets made again.*

### Classification — the vision replacement candidates

Not fetched. Verdicts are from the audit's *Replacement vision corpora* section, where both
the mirror tag and the true upstream were verified. These are the **only** group where a
genuinely shippable core exists.

| dataset | licence, verified at upstream | **REDISTRIBUTION** | note |
|---|---|---|---|
| `nyuuzyou/pxhere` (≈1.1M) | **CC0** — platform-enforced upload condition, not a metadata tag | **REDISTRIBUTABLE**, no conditions | The single most valuable clean constituent found. CC0 imposes nothing downstream. |
| `1aurent/PatchCamelyon` (327,680) | **CC0** | **REDISTRIBUTABLE**, no conditions | Camelyon16's own statement not independently re-fetched — **unverified at the second hop.** |
| `biglam/british-library-book-images` (1,080,814) | CC0 / Public Domain Mark | **REDISTRIBUTABLE**, no conditions | Primary source unreachable when audited — **unverified**; corroborated circumstantially (pre-1900 works). |
| Shapes3D (480,000) | **Apache-2.0**, LICENSE fetched verbatim | **REDISTRIBUTABLE** | notice |
| dSprites (737,280) | **Apache-2.0**, LICENSE fetched verbatim | **REDISTRIBUTABLE** | Pixel-bearing mirror is untagged — see *the pixel-mirror problem* below. |
| `timm/eurosat-rgb` (27,000) | **MIT**, plus Copernicus open-data instrument | **REDISTRIBUTABLE** | MIT notice **plus** `"Contains modified Copernicus Sentinel data [Year]"` |
| `AI-Lab-Makerere/beans` (1,295) | **MIT** | **REDISTRIBUTABLE** | Too small to matter at scale. |
| `google/quickdraw` (50.4M) | **CC BY 4.0** | **REDISTRIBUTABLE** | Dataset-level attribution suffices; per-sketch attribution is neither required nor possible. |
| CLEVR (100,000) | **CC BY 4.0** for images; generator code separately BSD | **REDISTRIBUTABLE** | Two licences — do not read only the code repo. |
| Caltech-101 (≈9,146) / Caltech-256 (≈30,607) | **CC BY 4.0** via CaltechDATA | **REDISTRIBUTABLE** | No pixel-bearing mirror carries a matching tag for Caltech-256; build from the CaltechDATA archive. |
| `ylecun/mnist` | **CC BY-SA 3.0** (mirror wrongly tags `mit`) | **FETCH-ONLY (POLICY)** | Share-alike. |
| `vincent-espitalier/K-MNIST-CSV` | **CC BY-SA 4.0** (mirror wrongly tags `cc-by-4.0`) | **FETCH-ONLY (POLICY)** | Share-alike. |
| `poloclub/diffusiondb` | CC0 data / MIT code, both confirmed | **EXCLUDE pending** | The licence text is maximally clean. The images are Stable Diffusion outputs and SD's own training provenance is unresolved. A grantor can only convey what they hold. Neither `FETCH-ONLY` nor `NEITHER` describes this — it is *grant-of-uncertain-scope*, and the honest handling is exclusion with the reason recorded. |
| EMNIST, GTSRB | no locatable grant | **NEITHER** | GTSRB's *"free to use, please cite"* is not a redistribution grant. |
| STL-10, SVHN, Food-101, Places365, CIFAR-10/100, DTD, Open Images, TreeOfLife-200M | see audit | **NEITHER** | Each ruled out at its own upstream. Open Images and TreeOfLife-200M are explicit that their compilation-level tag does not reach the per-contributor content. |

### Classification — the P2.4 retrieval mix, with a correction

`program/REMAINING.md` P2.4 records a clean >100k-pair retrieval mix. **P2.4 predates the
licence audit and two of its entries do not survive it.**

| dataset | P2.4 records | **REDISTRIBUTION** | correction |
|---|---|---|---|
| `allenai/gooaq` | *"Apache-2.0 (upstream LICENSE file read)"* | **NEITHER (contested)** | Accurate about the LICENSE file, and the audit found the README's contradicting NOTE on the same repo. P2.4's *"IMMEDIATE ACTION: switch to the upstream"* is still right about the mirror problem and now wrong about the conclusion. **The audit supersedes P2.4 here.** |
| `tasksource/esci` | Apache-2.0, `amazon-science/esci-data` LICENSE read | **REDISTRIBUTABLE** | Amazon Science is the rights holder in its own catalogue data; the grant reaches the content. Strongest entry in the mix. |
| `castorini/mr-tydi` | Apache-2.0, built on TyDi QA which is also Apache-2.0 | **UNVERIFIED — flag** | **TyDi QA's passages are Wikipedia.** Wikipedia text is CC BY-SA regardless of the tag a packager applies. An Apache-2.0 tag over Wikipedia passages is exactly the *mirror-more-permissive-than-upstream* pattern that P2.4's own list catalogues eight instances of. Verify before use. |
| `miracl/miracl` | Apache-2.0 | **UNVERIFIED — flag** | Same shape, same reason: MIRACL is Wikipedia-derived. |
| `THUIR/T2Ranking` | *"Apache-2.0 in README ONLY; LICENSE file 404s"* | **NEITHER** until verified | A README assertion with no licence file is the weakest evidence class this project accepts, and P2.4 already says so. |

**The finding worth carrying forward:** P2.4 claims the mix *"avoids [share-alike]
entirely"*. For `esci` that holds. For `mr-tydi` and `miracl` it is **unverified and probably
false** — every Wikipedia-derived retrieval corpus carries CC BY-SA underneath whatever tag
its packager applied, and the audit already records exactly this for Natural Questions. The
clean-retrieval problem is therefore harder than P2.4 concluded, and `esci` plus
operator-collected material is the part of the mix that survives.

### The counts, and the honest weighting

| bucket | constituents on record |
|---|---|
| `REDISTRIBUTABLE` — permissive/CC0/CC BY, shippable, no copyleft | **16** |
| `REDISTRIBUTABLE` in law, `FETCH-ONLY (POLICY)` in this design (share-alike) | **9** |
| `FETCH-ONLY (LEGAL)` — usable, not shippable | **3** (APPS, CodeContests, filtered CodeSearchNet) |
| `NEITHER` / excluded / contested | **~20** |
| licence-cleared for the foundation phase | **0** |

Constituent count flatters the picture. Weight by rows and it inverts: the four in-use
corpora are 0% shippable, the biggest single retrieval candidate (gooaq, 3.0M rows) is
contested, and 11 of the 16 clean constituents are vision sets **that have never been
fetched**. The clean superset is mostly a plan, not an inventory.

### Two structural findings the shape depends on

**The pixel-mirror problem.** The audit found repeatedly that the repo carrying the *licence*
and the repo carrying the *pixels* are different repos: dSprites' pixel mirror is untagged
while the Apache-2.0 LICENSE lives at `google-deepmind/dsprites-dataset`; CLEVR's HF mirror
is untagged while the CC BY 4.0 grant is on Stanford's page and in an in-repo
`COPYRIGHT.txt`; Caltech-256 has no matching pixel-bearing mirror at all. A fetch-only recipe
must therefore record **two** locations per constituent — where the bytes come from and where
the grant is recorded — and they will often differ. This is a required schema field, not a
nicety (§8).

**The continuing-obligation class.** Some sources permit training subject to a duty that
never ends — The Stack's opt-out channel is the named example, where removals are enacted
over time and a distributor is expected to track them. *(Taken from the brief; not
independently verified in this session.)* Nothing currently on record in this project is in
that class, but the `code` region's replacement path leads directly into it, and a weaker
version already bites: **2.8% of CodeSearchNet's source repositories are already 404**, and
80 Million Tiny Images was formally withdrawn with its authors asking the community to
*"delete any existing copies"*. A frozen published artifact cannot honour a duty that
arrives after the freeze. The versioning scheme in §4 exists mostly to make that survivable.

### The shape, stated plainly

**Manifests plus a builder, with a shipped-bytes core.** Five reasons, in the order they
actually bind:

1. **Zero of the four in-use corpora is redistributable.** A bytes-only superset would ship
   nothing the published model has learned from.
2. **The share-alike group is legally shippable and must not be shipped.** Including it would
   licence the whole distribution share-alike, defeating the MIT open-weights goal that
   motivated the audit in the first place. This is a policy exclusion, and it is the right
   one.
3. **Redistribution strengthens every unresolved question.** GooAQ, APPS, CodeContests and
   the CodeSearchNet chain are each survivable as training questions and each become an
   affirmative grant to strangers if shipped.
4. **Continuing obligations cannot be frozen.** A recipe can be re-run against a source's
   current state; a shipped tarball cannot.
5. **Egress.** pxhere alone is ≈1.1M full-resolution photographs; PatchCamelyon 327,680;
   Shapes3D 480,000. Shipping source-resolution bytes anyone can fetch is terabytes of cost
   for no reproducibility gain.

**What does ship as bytes**, and it is not nothing:

- **Everything the operator made.** Unambiguous rights, and the only category with them.
- **The CC0 constituents in processed form.** A 64×64 RGB uint8 derivative of pxhere is
  12,288 bytes per image — 100,000 capped images is ≈1.2 GB, against terabytes at source
  resolution. CC0 imposes no condition on the derivative, and **the processed form is the
  thing that is actually reproducible**: it pins the resize, the crop and the colour
  handling, which a URL does not.
- **The small permissive text sets.** gsm8k, aqua_rat, go_emotions, banking77,
  fashion_mnist. Tens of megabytes total, with a NOTICES file.
- **Every manifest, every notice, every tombstone, every reserve row-id list, and the
  builder.**

That is a real artifact. It is publishable, it is citable, it is reproducible, and every
claim it makes is one this project can actually defend.

---

## 2. Composition and phase structure

### The phases are the project's own curriculum, not a generic one

`config/mind/csd-regions.json` states the order and calls it mandatory:

> *"Curriculum order is mandatory: per-region pretrain -> router -> assembled mind ->
> foundation. Foundation corpora (FineWeb/C4/Pile) are step 4, not step 1."*

`program/REMAINING.md` encodes the same as P4 → P5 → P6. So the superset's phases must be
these, and a generic *pretrain / finetune / eval* split would misdescribe the project.

**Folded from `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` §7.2, 2026-09-06.** The
quote above is still the literal text of `config/mind/csd-regions.json` today (that file's
own edit is a reviewed-but-not-applied diff, taxonomy §1.4), but DEC-13/DEC-16 supersede the
discrete `router` with the interconnect module (taxonomy §2.3), and DEC-19 (§2.6) gives the
former `router`/`compose` split a single training recipe. The table and prose below are
re-keyed to the **four phases** the taxonomy specifies: **(1) per-region → (2) interconnect →
(3) whole-mind dynamic → (4) foundation**. Phase 3 is new, and it is where region and
depth/width growth happens (DEC-49, P5′).

| phase | slice id | consumes | produced by | status |
|---|---|---|---|---|
| **1 — per-region** | `phase1/<region>` | one slice per region: `language` (was `code`), `memory` (was `retrieve`+`compress`), `classify` (probe, DEC-05/06), `reason`, `visual` (was `vl_latent`) | P2.3, P3.1 | partly built, licence-blocked |
| **2 — interconnect** | `phase2/interconnect` | region-labelled examples plus `reserve/mixed`, per DEC-19's dense → distil → sparse → scheduler recipe (taxonomy §2.6); regions frozen | P5 | needs the reserve; supersedes the old `router` (P4) and absorbs what this table used to call `compose` |
| **3 — whole-mind dynamic** | `phase3/whole-mind` | region corpora **and** the reserve, interleaved, regions unfrozen | P5′ | **new.** Region and depth/width growth happens here (DEC-49); no code, eval harness, or metric exists yet |
| **4 — foundation** | `phase4/foundation` | broad text (FineWeb/C4/Pile class) | P6, *last* | **EMPTY.** No cleared constituent. |
| **R — reserve** | `reserve/<region>`, `reserve/mixed` | held out from *everything* | P2.5d | **the load-bearing piece** |

**Phase 4 (foundation) is empty and must be declared empty rather than aspirational.**
FineWeb and C4 are ODC-BY over Common Crawl; the Pile has known constituent problems. None
has a recorded verdict in this repository, and inventing one here would be exactly the
failure mode the catalogue exists to prevent. This is the largest phase by intended row count
and it has zero cleared constituents — that is a real finding, and it is a *cheap* one to sit
on, because the curriculum puts foundation last anyway. Record it as `status: planned` with
an empty constituent list, the same way `csd-regions.json` already records intent without it
reading as capability.

**Phase 2 (interconnect) needs no new corpora, and DEC-19 now gives it a training recipe
where the old `router` phase had none.** A router-style example is still an input plus the
region that should handle it, and every Phase 1 row already carries its region as a
shard-level constant, so that half of phase 2's *training* data is a free derivation of
Phase 1. Its *eval* is not: scoring on rows drawn from region training data measures
memorisation, so phase 2 eval comes from the reserve, like everything else. What DEC-19 adds
relative to the old `router`/`compose` split is a named recipe (dense → distil → sparse →
scheduler, taxonomy §2.6) and a pass condition per phase, so `reserve/mixed` is now one input
to phase 2's training rather than a separate `compose` phase's undefined one.

**Phase 3 (whole-mind dynamic) is the phase that is defining its own measuring instrument
now, and it has nothing else.** There is no whole-mind-dynamic code, eval harness, or metric
in the repository. `compose` is a bare string in `receipt.py`'s `STAGES` tuple that nothing
writes; `program/csd-program.json` records P5 with `"status": "blocked"` and — unlike P1, P3
and P4 — **no `method`, no `gate` and no `measures` keys at all**; `REMAINING.md` states the
gate only as prose (*"composed beats best single region on a mixed set"*). Taxonomy §4.1's
P5′ gives it a firmer pass condition (G2 still holds per bin after unfreezing **and**
G3/G3′ still hold; each new region beats its own untrained baseline and improves the composed
metric monotonically), but no implementation exists yet. `reserve/mixed` is still the entire
specification of what "composed beats best single region" will mean; building it defines the
metric rather than feeding it, which is an argument for doing the reserve before phase 3
rather than after.

### Per-submodel composition

**Re-keyed to faculties, taxonomy §7.2, 2026-09-06.** `retrieve` and `compress` fold into
one `memory` submodel with two heads (DEC-02); `code` and `vl_latent` are the `language` and
`visual` faculties (DEC-78, DEC-03). `classify` is demoted to a probe, not a trained
submodel (DEC-05/06), and is kept as a row for corpus continuity only. `reason` is left
un-renamed here: the taxonomy's own text proposes `reasoning` (§7.1, DEC-04) but every cell,
checkpoint and receipt on disk is still named `reason`, and `cogsyndelta.regions.aliases`
does not map it — an unresolved naming ambiguity, not a decision this fold makes.

Stated as caps, because the failure this is preventing is measured. `memory`'s retrieval
head is 77.8% gooaq / 19.5% NQ / 2.7% FiQA **while being evaluated on financial-domain
FiQA** — training on one distribution and measuring on another, which P2.5 names as hard
requirement 3.

| slice | constituents (post-audit, clean-only) | cap policy | honest state |
|---|---|---|---|
| `phase1/language` (was `code`) | filtered CodeSearchNet (permissive repos only, ~324k pairs of 455k), plus operator-owned repository code | ≤35% per source | `FETCH-ONLY`; needs the per-repo licence resolution and a NOTICES sidecar |
| `phase1/memory`, retrieval head (was `retrieve`) | `esci` (Apache-2.0), operator-collected financial QA, plus `gooaq` **iff** AI2 resolves it | ≤35% per source | The single biggest open question in the project. `esci` alone is a product-search corpus, not general web QA. |
| `phase1/memory`, compression head (was `compress`) | needs a permissive entailment source; `hkust-nlp/SynCSE-scratch-NLI` (MIT, 275,579 pairs) is the audit's one like-for-like candidate | ≤35% | **Synthetic — see §7's correction on generator terms.** MIT-tagged but GPT-4/GPT-3.5-generated, and the generator's own output-use terms were never examined. **Unverified.** |
| `phase1/classify` (DEC-05/06: a probe, not a trained submodel) | `banking77` (CC BY 4.0), `go_emotions` (Apache-2.0) | class-balance cap, not just source cap | Clean. `go_emotions` is 14,219 `neutral` against 77 for the rarest label — **185:1** — and must be capped or reweighted, not shipped raw. |
| `phase1/reason` | `gsm8k` (MIT), `aqua_rat` (Apache-2.0) | ≤50% (only two sources) | Clean. The MATH family stays out: DMCA-encumbered. |
| `phase1/visual` (was `vl_latent`) | the audit's composite: pxhere (CC0), PatchCamelyon (CC0), Shapes3D (Apache), CLEVR (CC BY), fashion_mnist (MIT), eurosat (MIT) — **plus operator photography** | ≤100k per domain, or the stricter ≈27k-per-domain variant | The only slice where a fully shippable version is achievable today. |

The `cap` / `cap_why` field names are taken verbatim from `MODEL-MANIFESTS.md`'s
`trainedSpec.sources`, where `cap` is documented as *"0 = uncapped. Exists for BALANCE, not
speed."* The superset manifest uses the same two fields with the same meaning, so a training
manifest and a corpus manifest cannot disagree about what a cap is.

### The reserve — constituent-level, not row-level

P2.5d requires *"material that does NOT overlap what the regions trained on, or the composed
evaluation is contaminated by construction."* There are two ways to build that and only one
of them can be **enforced**:

| | (a) stratified holdout from each constituent | (b) whole constituents nothing trains on |
|---|---|---|
| separation mechanism | measured, per row | **structural, per constituent** |
| what it measures | memorisation within a known distribution | genuine out-of-distribution transfer |
| cost to verify | full N×N near-dup scan every build | set membership, plus a confirming scan |
| failure observed | **`retrieve`'s 512-row holdout is 53.7% near-duplicate against train at cos ≥ 0.9** | — |

**Method (a) is what the project does today, and it is worse than it looks.**
`regions/pretrain.py::build_splits` carves the holdout as `all_pairs[: cfg.holdout_pairs]`
— the head of a seeded shuffle of the same pool the training set comes from. So the holdout
is drawn from exactly the training distribution by construction, and its only protection is
the exact-hash guard described below.

**And that guard proves nothing.** `build_splits` deduplicates on
`blake2b(" ".join(anchor.split()).lower())` and then calls
`assert_no_contamination(train_anchors, holdout_anchors)`, which computes its fingerprints
with `eval/metrics.py::_fingerprint` — **the same normalisation and the same hash**. The
dedup that runs first guarantees anchor-uniqueness, so the guard's `overlap` is 0 by
construction, for every region, always. It is structurally tautological. That is precisely
why the corpus survey found 53.7% near-duplicate leakage in a split whose recorded
contamination report says zero: **the check that was supposed to catch it could not, in
principle, ever fire.**

**Recommend (b) as primary and (a) as secondary.** The reason is the same principle
`MODEL-MANIFESTS.md` applies to unmeasured resource numbers: *make the wrong thing
unrepresentable* rather than asking a reviewer to catch it. A constituent carries exactly one
`phase`, `reserve/*` is a phase, and a training manifest that names a reserve constituent
fails validation. That is a one-line check that cannot be tautological, because it compares
declarations rather than re-deriving a hash it just enforced.

**Proposed reserve constituents**, with their state honestly recorded:

| slice | reserve constituent | licence | recorded verdict | notes |
|---|---|---|---|---|
| `reserve/vl` | Quick Draw (345 classes) | CC BY 4.0 | verified both ends | The audit's own recommendation, held out entirely from pretraining. Line drawings — a genuinely different domain from every pretrain source. |
| `reserve/vl` | Caltech-256 (257 categories) | CC BY 4.0 via CaltechDATA | verified | Natural everyday-object photography — the role CIFAR-100 used to play, with a real licence. |
| `reserve/retrieve` | **fiqa-pairs** | CC BY-SA 4.0 | SHARE_ALIKE | **Move FiQA out of training and into the reserve.** It is the only financial-domain signal, the audit notes dropping it costs `retrieve` its out-of-domain measurement, and as a reserve it is never shipped anyway — so its share-alike terms stop mattering. The FiQA/StackExchange question stays open and is smaller in this role. |
| `reserve/code` | HumanEval, MBPP | **no recorded verdict in this repo** | — | `csd-regions.json` already says *"MBPP and HumanEval are eval, not pretrain."* Both must be audited at mirror **and** upstream before use. **Unverified — do not assume MIT/CC-BY from memory.** |
| `reserve/compress` | `sentence-transformers/stsb` | **no licence tag** | fetched, never used (P0.9c: `graded_shards` is `[]`) | Same problem as every `sentence-transformers` dataset: 0 of 96 carry a licence. Needs a licensed replacement or an audited upstream. |
| `reserve/classify`, `reserve/reason` | **none available** | — | — | **A real gap.** There is no spare cleared constituent for either. Operator-generated material is the only current answer (§7). |
| `reserve/mixed` | cross-region composite | — | — | Phase 3's eval set. Assembled from the above, never from Phase 1. |

Two things to notice. First, **the share-alike constituents make good reserves precisely
because the reserve is fetch-only by construction** — a set you were never going to ship
cannot be contaminated by a licence that only bites on distribution. That resolves what looked
like a conflict between §1 and P2.5d. Second, the reserve is thin outside vision, and saying
so is more useful than filling the table with unaudited names.

### Publishing a reserve you cannot ship

A reserve that only the operator holds is not a reproducible benchmark. The mechanism:

> **`reserve/<slice>/rows.idlist` ships the `row_uid` of every reserve row, plus the
> constituent id and revision each came from. It contains no content.**

A third party fetches the constituent themselves, runs the builder's `normalise` stage, and
selects by `row_uid`. They get **byte-identical** reserve rows, verifiable against the
recorded fingerprint, without us having distributed a single row. Because `row_uid` is
derived from `(constituent_id, constituent_revision, source_row_index)` (§5), the selection
is reproducible by anyone with the same source revision and does not depend on any state we
hold.

The same mechanism is what makes removals survivable (§4) and what lets a consumer prove they
built the same superset we did. **One primitive, three jobs** — which is the test for whether
a mechanism is right.

### Enforcing and verifying the separation

Structural separation of *constituents* does not prove separation of *content*, and this
project has measured exactly why:

> **SQuAD and HotpotQA share 95.9% of article titles** (424 of 442) despite being different
> constituents from different authors. Exact context overlap is only 0.12% (22 of 18,891) and
> only 107 of 18,265 title-matched candidate pairs exceed cos 0.95 — so the overlap is mostly
> benign at content level. **But you only know that because it was measured.**

So the verification is two-layer and both layers are required:

**Layer 1 — structural, at validate time. Cheap, always runs.**
- Every constituent has exactly one `phase`. Phases are mutually exclusive.
- No constituent id appears in both a `phase*` slice and a `reserve/*` slice → hard failure.
- No training manifest's `sources[].corpus` names a constituent whose phase is `reserve/*` →
  hard failure. This is the same refusal shape as `fetch`'s licence gate, and like it there
  is **no override flag**.

**Layer 2 — measured, at build time. Writes a receipt with gates.**
- **Exact pass** on the normalised form, reserve against every training slice.
- **Semantic pass** with a declared, non-circular encoder (see §6 on circularity).
- **Per-constituent normalisation rules applied first**, because the survey proved plain
  hashing misses real overlap: *"APPS and CodeContests share 17.8% of problems, INVISIBLE to
  hashing because CodeContests prefixes every description with `<id>_<letter>. <Title> - `.
  Dedupe on the stripped description."* And *"SQuAD/HotpotQA share 95.9% of article titles.
  Dedupe by title, not passage hash."* Both are declared per-constituent in the manifest, not
  hardcoded in the deduper.
- **Contamination is always resolved in the reserve's favour.** A contaminated row is dropped
  from *training*, never from the reserve. Shrinking an eval set to make a number look better
  is the exact failure the reserve exists to prevent, and it must be impossible by
  construction rather than discouraged by convention.

**Proposed gates, set against measured baselines rather than invented:**

| gate | threshold | measured today | why this number |
|---|---|---|---|
| `reserve_exact_overlap` | **0 rows** | 22 rows SQuAD↔HotpotQA | Exact overlap is always fixable; there is no reason to tolerate any. |
| `reserve_neardup_fraction` | **≤ 1% at cos ≥ 0.9** | `retrieve` holdout **53.7%**, `code` **24.2%**, `compress` **9.0%** | The current state fails this by a factor of 50 in the worst region. That is the point: the gate names a target the project does not yet meet, rather than one it already passes. |
| `no_source_dominance` | **≤ 35% of a slice from one constituent** | `retrieve` is 77.8% gooaq | P2.5 hard requirement 3. Exceeding it requires a `cap_why` and a reviewer. |
| `phase_disjoint` | structural, no exceptions | — | Layer 1. |

**The tooling for layer 2 does not exist in this repository, and the design must not pretend
otherwise.** What exists is one 72 KB `analysis.json` on `/mnt/bulk`, produced by a one-off
run on `gpu1080ti` from an scp'd tarball of `src/`, with nothing committed back —
`program/REMAINING.md` P2.5a records *"Overlap-check tooling as a reusable gate"* as **todo**
and that is the authoritative status. The in-repo overlap primitives are exact-hash only
(`eval/metrics.py::contamination_report`); there is no MinHash, no LSH, and no embedding
overlap gate anywhere in `src/` or `scripts/`.

What the analysis does supply is a **specification to reimplement from**, and it is precise
enough to be one: `build_splits()` called with the exact `PretrainConfig` each region's
receipt recorded, anchors embedded and L2-normalised, holdout-vs-train by a 512×N chunked
matmul, within-train by a chunked full N×N scan with self excluded and symmetric
double-counting halved, reported at thresholds 0.9 / 0.95 / 0.98 / 0.99.

Its cost is known and affordable: 505,216 `retrieve` rows embedded and scanned on the 1080 Ti
at **631–1244 MB peak VRAM**. That is what makes this a per-build gate rather than an
occasional audit — but building it is P2.5a, and P2.5a is unstarted.

---

## 3. The licence of the superset itself

> **A superset's obligations are the union of its constituents', not a choice.**

That is the operator's framing and it is exactly right, with one refinement that changes the
design:

> **The superset must not claim a single licence over the whole. It publishes a per-shard
> licence map and a stated union of obligations, so a consumer can compute the union for the
> subset they actually use.**

A single blanket licence would be either a lie (claiming MIT over CC BY-SA material) or a
tax (forcing CC0 constituents to carry attribution they do not require, so nobody can extract
the obligation-free subset). The map is more honest and strictly more useful.

### The union for the shipped-bytes core

Constituents: MIT (gsm8k, fashion_mnist, eurosat, beans) + Apache-2.0 (aqua_rat, go_emotions,
Shapes3D, dSprites) + CC0 (pxhere, PatchCamelyon, BL books) + CC BY 4.0 (banking77, quickdraw,
CLEVR, Caltech-101/256) + the operator's own.

**The union is: attribution + notice preservation + a statement of modification.** No
copyleft. Concretely, the distribution must carry:

1. **Attribution** for every CC BY and MIT constituent — CC's TASL form (Title, Author,
   Source, Licence, with the licence both named and linked). The audit already drafted these
   strings verbatim and they are reused rather than rewritten.
2. **The MIT permission notice** reproduced for each MIT constituent — MIT requires *"the
   above copyright notice and this permission notice"* in all copies, and a superset is a
   copy.
3. **A copy of the Apache-2.0 licence**, per-file modification notices for files we changed,
   and propagation of any upstream `NOTICE` file.
4. **`"Contains modified Copernicus Sentinel data [Year]"`** for the EuroSAT imagery, which
   is a separate instrument from EuroSAT's own MIT tag.
5. **An explicit statement of modification.** CC BY 4.0 requires *"indicate if You modified
   the Licensed Material"*; Apache-2.0 §4(b) requires the same for changed files. The audit
   notes its own draft notices do not yet satisfy this. For a superset the modifications are
   substantial and specific, and they belong in the card as a list: **deduplicated, capped,
   filtered, re-split, and (for vision) downsampled to 64×64.**
6. **Nothing from CC0** — CC0 imposes no condition. Citation is requested by several CC0
   sources and should be given, but it is courtesy, not obligation, and the card should say
   which is which.

**The consequence of the map, stated for a downstream user:** if you take only the CC0 shards,
you carry no obligations at all. That is only true because the per-shard licence field exists.
It is the strongest practical argument for mandatory per-shard licence metadata, and it holds
even in a distribution where every constituent happens to be CC0.

### The union for the full built superset

Once the builder has fetched the fetch-only constituents on a consumer's machine, **their**
copy is share-alike-encumbered even though ours was not. The card must say so, because
otherwise a user who runs the builder with `--include-share-alike` and then redistributes the
result is non-compliant on our advice.

So the card carries **two** union statements:

| | shipped bytes | full build (default) | full build (`--all`) |
|---|---|---|---|
| attribution | yes | yes | yes |
| notice preservation | yes | yes | yes |
| statement of modification | yes | yes | yes |
| **share-alike** | **no** | **no** | **yes — CC BY-SA 4.0 and 3.0** |
| per-item attribution risk | no | no | **yes, if FiQA is StackExchange-derived** |

The builder's default must exclude share-alike, and including it must require an explicit
flag whose help text says what it does to the result's licence. A default that silently
produces a share-alike corpus is a trap.

### Incompatible pairs

The question asked is whether any *pair* of constituents has terms that cannot coexist in one
distribution. Four candidates, in decreasing order of how live they are for this project.

**1. GPL/AGPL source code + any CC BY-SA constituent, in a single distributed whole. LIVE.**

~15% of the `code` corpus is GPL/AGPL — *"roughly 72,000 functions of strong-copyleft source
code"*, which the audit notes is a larger exposure than all the CC BY-SA data combined. GPL
requires the whole work be GPL and forbids imposing further restrictions; CC BY-SA requires
the whole be BY-SA. My recollection is that CC declared GPLv3 a *one-way* compatible licence
for BY-SA 4.0 — material may be relicensed **into** GPLv3, not out of it — which would mean
no single licence covers both directions. **Unverified in this session; verify before relying
on it.** Either way the resolution is the same and it is already precedented:

> BigCode *"did not resolve the copyleft question for The Stack — it removed the question by
> filtering: the three copyleft licenses (MPL/EPL/LGPL) were excluded and the list of
> permissive licenses extended to 193 licenses in total."*

**The superset excludes copyleft-licensed code rows structurally**, the same way it excludes
NC-licensed rows: a filter with no override, driven by the per-repo licence resolution, with
the surviving repos' notices shipped as a NOTICES sidecar. That is the `code` region's
recommended repair in the audit and the licence-compatibility argument is a second,
independent reason to do it.

There is a counter-argument worth recording rather than dismissing: GPL's *mere aggregation*
clause permits GPL'd works to sit beside non-GPL works on a distribution medium. Separate
shards might qualify. **But once rows from different sources are concatenated into one
training shard, the aggregation reading gets much weaker** — which is a second, independent
reason for the one-constituent-per-shard rule in §5. It also means the mere-aggregation
defence is available to us only if we never merge sources within a shard, so the rule has to
be structural rather than a performance choice.

**2. CC BY-SA 3.0 + CC BY-SA 4.0. LIVE if share-alike is ever shipped.**

Natural Questions is 3.0; SQuAD, HotpotQA, FiQA, SNLI and oxford-iiit-pet are 4.0. The audit
states directly that these *"are not bidirectionally compatible, which would be a second
problem."* BY-SA 3.0 permits distributing a *derivative work* under a later version with the
same licence elements, so a 3.0 → 4.0 upgrade path plausibly exists for an **adaptation**;
both licences also treat a **collection** differently from an adaptation, with the SA
condition not reaching the other works in a collection.

**Which of those a merged, deduplicated, filtered, re-split, cross-source training superset
is — a collection or an adaptation — is the load-bearing unresolved question**, and it is not
one this document can settle. Calling a corpus that applies cross-source dedup, per-source
caps, quality filters and a re-derived split "mere aggregation" is a stretch. **Needs a human,
and arguably a lawyer.** The design sidesteps it entirely by not shipping share-alike, which
is a large part of why that policy is worth its cost.

**3. The OANC End User Licence + any CC constituent, if a single licence is claimed. MOOT,
but it is the pattern.**

The OANC EULA conditions redistribution of a *"transparently modified"* version on displaying
the agreement's own text in human-readable form and on per-source attribution. Both CC BY and
CC BY-SA forbid imposing *"additional or different terms"* on the licensed material. If one
licence were claimed over a whole containing both, the OANC condition would be an additional
restriction on the CC portion. Moot in practice — all-nli is `NEITHER` for independent
reasons — but it is precisely the failure a blanket licence produces and the per-shard map
avoids.

**4. Any NC term + anything. STRUCTURALLY EXCLUDED.**

MS MARCO, ELI5, `BeIR/scifact`, `facebook/anli` and ImageNet-derived material are all already
refused by the existing gate. The superset inherits that gate unchanged.

**And one non-incompatibility worth naming**, because it looks like one: CC0 alongside CC BY
is fine. CC0 does not "infect" and is not diluted. The only hazard is that a consumer cannot
tell which rows are obligation-free unless the per-shard licence field is preserved — which
converts a licensing question into a metadata requirement, and metadata requirements this
project can actually meet.

**One clause that constrains our own design.** The superset will carry a removal duty (§4).
It must be framed as **our** duty — *we will stop distributing and publish a tombstone* — and
never as a condition imposed on downstream users of the CC-licensed constituents, because
imposing conditions on CC material is exactly what CC's no-further-restrictions clause
forbids. The distinction is subtle and it changes the wording of the dataset card.

### What the dataset card must carry

1. The **constituent table**: id, upstream, pinned revision, licence observed verbatim,
   redistribution verdict, `checked_utc`, phase, row count, cap applied, and whether bytes are
   shipped or fetched.
2. **Both union statements** (shipped / full build), as the table above.
3. The **attribution block**, verbatim from the audit's drafted TASL strings, plus the
   Copernicus notice.
4. A **NOTICES file** carrying MIT permission notices, the Apache-2.0 licence text, propagated
   upstream NOTICE files, and — for the code slice — the per-repository copyright notices for
   every surviving repo.
5. The **statement of modification**: what the polish pipeline did, by stage, with receipt
   ids.
6. The **reserve declaration** and its row-id lists.
7. **Tombstones** for everything ever removed.
8. The **removal channel** and the licence re-verification cadence.
9. The **known-unresolved list**, inherited from the audit's *What I could not determine* and
   extended by this document's §10. A card that omits its own open questions is worse than one
   with none, because the omission will be read as an absence of them.

### What the model card must carry, and why its obligation set is larger

**A model trained on the superset has learned from constituents the superset does not ship.**
So the model card's obligation set is a strict superset of the dataset card's, and the two
cannot be the same document:

- Attribution for **every constituent trained on**, including every `FETCH-ONLY` one.
- The superset **version and fingerprint** the run used — which is what makes the claim
  checkable.
- The **share-alike position statement**, in the audit's own terms: whether trained weights are
  Adapted Material is unsettled, this project takes position X, and *"what is NOT a mitigation
  is asserting that weights are not derivative works because it would be convenient."*
- **Eval-set disclosure.** The audit established the precedent for CIFAR-100: an eval-only
  corpus should be *"described in the model card as an evaluation set rather than quietly
  omitted"*, because it sits in a gate. The reserve constituents get the same treatment.

### What licence the operator's own data should carry

The operator's own contributions are the one place where a licence is genuinely a *choice*.
Three defensible picks:

- **CC0** — maximum reuse, zero downstream friction, and it means the operator's contribution
  never adds an obligation to anyone's union. The strongest choice if the goal is for the data
  to be used.
- **CC BY 4.0** — attribution, at zero marginal cost to the union, because banking77, Quick
  Draw, CLEVR and Caltech already put attribution in it. **Recommended default**: it costs
  consumers nothing they were not already paying and it credits the work.
- **Apache-2.0** — if any contribution is code-shaped and a patent grant is wanted.

Whichever is chosen, it must be stated per-constituent like everything else, not assumed from
the repository's own LICENSE — which currently reads *"Proprietary License … all copyrights
are reserved"* and would be an actively wrong default.

---

## 4. Versioning semantics

Version id: **`csd-superset/<MAJOR>.<MINOR>.<PATCH>`**, plus a content fingerprint per
phase-slice (a Merkle root over the sorted per-shard sha256), so *"trained on v1.2.0"* is a
verifiable claim rather than a label.

**It must be a content fingerprint, and the existing one is not.**
`regions/pretrain.py::_fingerprint_corpus` hashes **shard basenames and byte sizes only** —
not content, not row counts, not the column selection, not the caps, not the `build_splits`
code path. Worse for this purpose: it is computed over `cfg.shards` alone, so
`retrieve`'s `extra_sources` — Natural Questions and the 400k-capped GooAQ, **97% of that
region's data** — are not in its receipt's corpus fingerprint at all. `csd-quantize.py` hard-
fails on a mismatch against it, so it is load-bearing today while covering less than it
appears to. P0.8 already records the blind spot.

The superset fingerprint therefore does not extend `_fingerprint_corpus`; it replaces it for
this artifact, and covers content, every source including capped extras, the caps themselves,
and the stage-config hashes that produced the shards.

### The three levels

**MAJOR — a consumer's conclusions change.** A metric computed on `vN` is not comparable to
one on `vN+1`, or a previously-compliant downstream distribution stops being compliant.

- A constituent is **removed** (opt-out, licence re-verification failing, source gone dark).
- A licence verdict is **downgraded** (`REDISTRIBUTABLE` → `FETCH-ONLY` or `NEITHER`).
- **Splits change** — any row moves between train, reserve, or phase slices.
- The **reserve changes** at all. The reserve is the measuring instrument; changing it
  invalidates every comparison across the change, and pretending otherwise is how benchmark
  drift happens silently.
- A constituent is **reassigned to a different phase**.
- Schema or field **semantics** change.

**MINOR — additive and backwards-comparable.**

- Constituents **added**.
- A new phase slice or region slice added.
- **Additional shards** of an existing constituent at the same revision.
- Derived columns added, existing ones unchanged.
- **Caps raised** (more rows admitted from a source already present).

The invariant: every row present in `vN.x` is present in `vN.(x+1)` with the same `row_uid`,
the same content, and the same split assignment. A model trained on `vN.x` can be compared
against one trained on `vN.(x+1)`, with the added portion identified by set difference.

**PATCH — defect repair that preserves membership and semantics.**

- Encoding fixes, whitespace normalisation, a malformed record repaired.
- Provenance or metadata corrections.
- Licence **evidence** strengthened without the **verdict** changing (a second source found
  for a grant we already relied on).
- Documentation and card corrections.

The invariant: **no row added, no row removed, no split changed.** A PATCH may change a row's
bytes only as a defect repair, and it must list the affected `row_uid`s. If a correction
would change what a model learns from the corpus — not merely how it renders — it is MINOR at
minimum. When in doubt, escalate: the cost of an unnecessary MINOR is a version number; the
cost of a MAJOR change wearing a PATCH label is an invalid comparison somebody will publish.

**Caps raised is MINOR; caps lowered is MAJOR**, because lowering a cap removes rows and
removal is always MAJOR. This asymmetry is deliberate and it comes up constantly during
balance tuning, so it is worth stating rather than deriving each time.

### Removal — the hard case

A model trained on v1.2 cannot un-see data pulled in v1.3. The conflict is direct: we must
**stop distributing** the removed content while **preserving reproducibility** of the version
that contained it. Those are irreconcilable if "preserve" means "keep the bytes".

**Resolution: preserve the shape, not the content. A hash is not a copy.**

Five things the version record must preserve after a removal, and the reasoning for each:

**1. A tombstone**, published permanently:

```yaml
tombstone:
  constituent: corpus/example-source
  revision: 4f1c9e2…
  present_in: ["1.0.0", "1.1.0", "1.2.0"]
  withdrawn_in: "1.3.0"
  on: 2026-11-04
  rows_removed: 128409
  reason: opt-out                    # opt-out | licence-reverification | source-gone | dispute
  authority: "upstream removal channel, request id 8812"
  row_uids: rows.idlist              # ids only. No content.
  affects_phases: ["phase1/code"]
  affects_reserve: false
```

**2. Content-addressed identity, so removal is verifiable without retention.** A third party
who *lawfully still holds* the source can confirm they hold the same rows we removed, and can
reproduce v1.2 for themselves. We retain nothing of the content.

**3. The v1.2 manifest stays published, immutably**, with the constituent still listed and
marked `withdrawn_in: 1.3.0`. Editing a published manifest to erase a constituent would
destroy the record of what a released model was trained on — which is the one thing a
provenance system exists to preserve. This mirrors `MODEL-MANIFESTS.md`'s rule that a
measurement is *"never overwritten, only superseded"*.

**4. An honest reproducibility statement per version.** This is the part that is usually
skipped and it is the part that matters:

> *v1.2.0 is fully reproducible from sources A, B, D. Constituent C was withdrawn on
> 2026-11-04 and can no longer be fetched. A build of v1.2.0 today yields 4,872,113 rows
> against the recorded 5,000,522, and the fingerprint will not match. The expected difference
> is exactly the 128,409 row_uids in `tombstones/1.3.0/C/rows.idlist`.*

Recording what you *cannot* reproduce is worth more than a claim of reproducibility that
quietly fails. It also lets a verifier distinguish "the build is broken" from "the world
changed", which is otherwise indistinguishable from the outside.

**5. Receipts of affected runs are not rewritten.** A model trained on v1.2 keeps a receipt
saying v1.2, forever. The tombstone is what explains what v1.2 contained. This is
`receipt.py`'s own principle — old shapes are *"adapted on read"* rather than rewritten,
because *"rewriting them would discard the measurements they carry"*.

**What a removal does NOT do.** It does not reach downstream. Publishing creates a duty we can
discharge — stop distributing, publish the tombstone, notify known consumers — and one we
cannot: make everyone who already fetched it delete their copy. The dataset card must say
that plainly rather than implying a control that does not exist. 80 Million Tiny Images is the
worked precedent: its authors *"ask the community to refrain from using it in future and also
delete any existing copies"* — a request, made because a request is all that was available.

### The continuing-obligation heartbeat

A frozen artifact cannot honour a duty that arrives after the freeze, so the artifact needs a
pulse:

- Every constituent carries `licence.checked_utc`. `MODEL-MANIFESTS.md`'s rule applies
  unchanged: *"a verdict without a date is an opinion of unknown age."*
- A re-verification cadence is declared per constituent, keyed to its risk class. Sources with
  an opt-out channel or a mirror-vs-upstream discrepancy get the shortest interval; a CC0
  platform-enforced upload grant gets the longest.
- **Staleness is reported, never enforced** — again from the manifest design. A constituent
  whose verdict is older than its cadence is flagged in the card and in `validate` output. It
  is not silently dropped, because *"silently discarding the only number you have is worse
  than showing a number with a warning on it."*
- The published card carries a **removal channel**: how to ask for content to be removed, and
  what we will do (tombstone, MAJOR bump, notification) and by when.

The practical consequence for release cadence: a superset with continuing-obligation
constituents cannot be published once and abandoned. If the operator is not prepared to
maintain the heartbeat, those constituents must be `FETCH-ONLY` regardless of what their
licence permits — which is a third, independent reason the recipe shape is the right one.

---

## 5. Provenance

### Minimum per-row

| field | type | why |
|---|---|---|
| `row_uid` | 16 B binary | Stable identity. Survives content correction. What tombstones and reserve id-lists reference. |
| `content_hash` | 16 B binary | Hash of the **normalised** form. Changes when content changes. What dedup and verification use. |

**Two ids, not one, and the removal case is what forces it.** A single content hash breaks the
moment a PATCH corrects a row: the id changes, and every tombstone and reserve list that
referenced it dangles. A single opaque assigned id is not reproducible by a third party
building from the recipe. So:

```
row_uid      = H(constituent_id ‖ constituent_revision ‖ source_row_index)[:16]
content_hash = H(normalised_text)[:16]
```

Both are reproducible by anyone with the same source revision and the same declared
normalisation. One is stable under correction; one is sensitive to it. 16 bytes gives a
birthday bound around 2^64, which is ample against 10^9 rows — and it must be stored as fixed
binary, not hex, because hex doubles it for no benefit.

### Minimum per-shard

Everything else, because **everything else is constant within a shard**:

`constituent_id` · `constituent_revision` · `licence_spdx` · `licence_source` (which may
differ from the fetch source — the pixel-mirror problem) · `redistribution_verdict` ·
`checked_utc` · `attribution_string` · `phase` · `cap_applied` · `normalise_rules_id` ·
`transform_receipts[]` · `row_count` · `sha256`.

### The rule that makes this cheap

> **One constituent, at one revision, per shard. Never mix sources within a shard.**

This is the single decision that turns provenance from a 40% tax into a rounding error.
`constituent_id`, `revision`, `licence` and `phase` become shard-level constants, so Parquet's
dictionary and RLE encoding collapse them to a few bytes per row group rather than storing a
string per row. Naive per-row provenance strings would be roughly 120–160 bytes against a
`retrieve` row whose own text averages ~84 tokens (≈340 bytes) — **a 35–45% overhead on the
text itself**, which is what makes people abandon per-row provenance and then have no answer
when a source is withdrawn.

The rule also does two other jobs, which is how you know it is the right rule:

- It preserves the **mere-aggregation** reading discussed in §3, which is unavailable the
  moment rows from different licences are concatenated into one file.
- It makes **removal a file operation**. Withdrawing a constituent means deleting its shards,
  not rewriting every shard in the corpus to filter rows out.

### The honest storage cost

At this project's actual scale, measured against what is on disk:

| slice | rows | id cost (32 B/row) | slice size on `/bulk` | overhead |
|---|---|---|---|---|
| `retrieve` | 505,216 | 16 MB | 961 MB | **1.7%** |
| `code` | 430,931 | 14 MB | 18 GB | **0.08%** |
| `compress` | 277,269 | 9 MB | 19 MB | **47%** ← the exception |
| `classify` + `reason` | ~65,000 | 2 MB | 31 MB | 6.7% |
| `vl` (images) | ~170,000 | 5 MB | 390 MB | 1.4% |

**So the honest answer at this scale is: per-row provenance costs under 2% almost everywhere,
and there is no reason not to have it.** The one exception is instructive rather than
alarming: `compress` rows are tiny (anchor mean 18.6 tokens, positive mean 9.6) so 32 bytes of
id is a large fraction of a small row. Even there it is 9 MB in absolute terms. **Do not
optimise this.** The cost only becomes real at the multi-billion-row scale this project is not
at, and at that scale the fix is to widen shards rather than to drop ids.

Extrapolating honestly: at 100M rows, ids are 3.2 GB. At 1B rows, 32 GB — at which point ids
are a real line item and the design should reconsider, probably by moving `row_uid` to an
implicit shard-offset scheme and keeping only `content_hash`. Recording the threshold now
means nobody has to rediscover it.

### Where provenance is genuinely expensive

**1. Per-repository licence for the code slice.** Licence varies *within* a constituent, per
repository — 13,581 distinct repos in CodeSearchNet Python. The answer is **not** a per-row
licence string; it is the existing `repo` column plus a **repo → licence resolution table
shipped as a sidecar**, resolved once (13,581 authenticated GitHub API calls, hours at
5,000/hr) and versioned like everything else. The table is a few hundred kilobytes; a per-row
string would be tens of megabytes and would still not carry the copyright holder's name, which
is what MIT actually requires you to reproduce. This is the shape of CodeSearchNet's own
`_licenses.pkl` — the artifact the mirror dropped — and recovering it should be tried before
the API approach.

Record with the table, permanently, the four caveats the audit already enumerated: a repo's
licence today is not its licence in 2019; 2.8% of repos are gone and those rows must be
**dropped, not assumed permissive**; `_licenses.pkl` is the authoritative artifact; and
dropping ~15% GPL and ~5% weak-copyleft changes the corpus *distribution*, so metrics must be
re-measured rather than compared against the old recall@1.

**2. Synthetic-data generator provenance.** Self-generated rows need fields no fetched row
needs: `generator_model`, `generator_revision`, `generator_licence`, and
`generator_output_terms` — the generating model's own stance on using its outputs to train
other models. §7 explains why this is not optional.

**3. What NOT to store: the near-duplicate graph.** The analysis measured **2,021,318**
unordered near-dup pairs above cos 0.9 within `retrieve`'s train set alone. At 32 B/edge that
is a 65 MB edge list for one region, and it is worthless a build later because the encoder
will have changed. **Store the decision, not the graph**: which `row_uid`s were dropped, at
what threshold, by which encoder at which revision, in the dedup stage's receipt. The graph is
reconstructible from that; the decision is not reconstructible from the graph.

**And store enough of the decision to debug it.** `ContaminationReport.examples` currently
holds `sorted(overlap)[:5]` — **hex digests, not the offending text** — so a receipt records
that a collision happened and makes it impossible to see what collided. Storing `row_uid`s
instead of digests fixes this at the same size: a `row_uid` resolves back to a row through
the shard, whereas a content digest resolves to nothing. Cheap, and the difference between a
record and an audit trail.

---

## 6. The polish pipeline

"Polished" means: **every transformation is a named, idempotent stage that declares its inputs,
writes a receipt with gates, and produces the same output when re-run.** Stages compose; none
of them is a script someone ran once.

### Reusing the receipt envelope

`pipeline/receipt.py`'s `STAGES` tuple is documented as *"Open by convention rather than
enforced, because a new architecture may have a stage nobody anticipated; the reader groups by
whatever it finds."* So corpus stages are added without touching the envelope:

```python
Receipt(
    producer=Producer(project="cogsyndelta", component="phase1/retrieve", architecture="corpus"),
    stage="dedup-semantic",
    metrics={"rows_in": 514362.0, "rows_out": 505216.0, "dropped_fraction": 0.0168},
    gates={"idempotent": True, "encoder_not_circular": True, "threshold_declared": True},
    provenance={"manifest": "corpus/phase1-retrieve", "manifest_sha": "…",
                "input_fingerprint": "…", "stage_config_hash": "…"},
    detail={"encoder": "…", "threshold": 0.95, "per_source_dropped": {...}},
)
```

`metrics` stays a flat name→number dict so a reader can plot corpus runs beside training runs
without knowing anything about corpora. Everything corpus-specific goes in `detail`, which
readers pass through untouched. And `passed` already encodes the right rule: **a run with no
gates is not a pass**, because *"something that measured nothing has not demonstrated
anything."*

**Idempotency mechanism**, matching `csd-corpus-expand.py`'s `is_present()`: each stage is
keyed on `(input_fingerprint, stage_config_hash)`. If an output with that key exists, the
stage re-verifies the fingerprint and returns `present` without recomputing. Re-runs are cheap
and safe, which is the point — this is meant to be run by an unattended agent.

### The stages

**0 · `ingest`** — fetch at a pinned revision; record `sha256` per file, row count, bytes, and
the licence verdict with its date.
Gates: `licence_verdict_allowed` (and for the shipped tier, `redistribution == REDISTRIBUTABLE`
— **no override flag**, exactly as `fetch` has none today), `revision_pinned` (a commit or
immutable ref; *"a moving tag is not a pin"*).

**1 · `normalise`** — produce the canonical form used for ids and comparison. Unicode NFC,
trim, collapse internal whitespace; case-fold **for hashing only**, never mutating stored text.
Per-constituent rules are **declared in the manifest, not hardcoded**, because the survey
proved generic normalisation is insufficient:

- CodeContests must have its `"<id>_<letter>. <Title> - "` prefix stripped, or its **17.8%
  overlap with APPS is invisible to hashing** (measured: 1,784 of 10,000 APPS items match at
  cos ≥ 0.9, against just **2** exact matches).
- SQuAD/HotpotQA must be compared **by article title**, not passage hash.

Gate: `id_stable` — re-running produces byte-identical `row_uid`s and `content_hash`es.

**2 · `dedup-exact`** — on `content_hash`, within and across constituents.

**Fix the anchor-only key on the way in.** `build_splits` hashes the **left column only** —
`blake2b(" ".join(anchor.split()).lower())` — and keeps the first occurrence. For a
one-to-many corpus that discards real data rather than duplicates: one FiQA question with 23
relevant passages collapses to a single pair and 22 genuine positives are destroyed (recorded
as P0.9e). The correct primitive **already exists in the same codebase** and is used nowhere
else: `_prepare_graded`'s `_pair_key`, an order-independent hash over both normalised sides:

```python
first, second = sorted((" ".join(left.split()).lower(), " ".join(right.split()).lower()))
key = blake2b(f"{first}\x00{second}".encode(), digest_size=16).hexdigest()
```

The superset's `content_hash` uses that shape. `build_splits`'s anchor-only key stays where
it is until P0.9e is fixed; the corpus pipeline must not inherit it.

The keep-rule must be **deterministic and licence-aware**, or the output depends on file
ordering and the stage is not idempotent: keep the copy from the constituent with the better
redistribution verdict; tie-break by the earlier-added constituent; tie-break by `row_uid`.
Licence-aware ordering matters — given a duplicate row available under both CC0 and CC BY-SA,
keeping the CC0 copy is free and keeps the union cleaner.
Records `duplicates_removed` per source pair, matching what `build_splits` already reports —
though note those counts are **anchor collisions**, so the pair-level figures will differ and
must not be compared against them. The current baselines, for the record: `code` 23,800
removed of 455,243; `compress` 36,534 of 314,315; `retrieve` 8,634 of 514,362.
Gates: `deterministic` (two runs, different input order, identical output fingerprint),
`key_is_pairwise`.

**3 · `dedup-semantic`** — the pass hashing misses, and **the stage with the methodological
problem**.

The corpus analysis records its own caveat in its metadata, and it is the right one to worry
about:

> *"All embeddings use the project's OWN trained region encoders."*

A near-duplicate verdict produced by a model trained on the data being deduplicated is
circular: the encoder's notion of "similar" was shaped by the very redundancy the stage is
trying to remove, and it will systematically under-detect the duplicate structure it was
trained through. The analysis handled this correctly in one place — SQuAD-vs-HotpotQA used
the `retrieve` encoder and noted *"not circular for this pair"* — and honestly flagged the
opposite case, where the `code` encoder was applied to competitive-programming prose with the
note *"DOMAIN MISMATCH … treat as weaker signal than exact-match."*

**The stage must therefore declare and gate on non-circularity:**

- Preferred: a **frozen, external, licence-clean encoder** pinned by revision, never trained
  on any superset constituent. This is a real dependency the project does not have yet and
  choosing it is its own small licence question.
- Acceptable: a **different region's** encoder than the slice being deduped, with the domain
  mismatch recorded in `detail` as a confidence qualifier.
- Refused: the slice's own encoder. Gate `encoder_not_circular` fails.

Threshold is declared per slice, not global, because the measured distributions differ sharply
— `compress` top-1 similarity has p50 0.777 while `retrieve` has p50 0.911, so one threshold
would be far too aggressive for one and useless for the other.

Cost is known and affordable: 505,216 rows embedded and scanned chunked N×N on the 1080 Ti at
**631–1244 MB peak VRAM**. That is why this is a build gate rather than an occasional audit.

**4 · `balance`** — per-source caps, never a global one.
Source caps: no constituent above 35% of its slice without a `cap_why` and a reviewer.
Baseline being fixed: `retrieve` at 77.8% gooaq.
Class caps within a constituent: `go_emotions` is 14,219 `neutral` against 77 for the rarest —
**185:1**. Cap or reweight, and record which.
For vision, the audit's own arithmetic is the warning: an uncapped union of the composite
candidates *"would put EuroSAT under 2% of every batch and Fashion-MNIST around 3%"*, since
batch sampling is `torch.randint` over a concatenated tensor with no domain weighting.
Gates: `no_source_dominance`, `class_ratio_recorded`.

**5 · `separate`** — the contamination stage, described in §2. Reserve versus every training
slice, exact then semantic, resolved **always in the reserve's favour**.
Gates: `reserve_exact_overlap == 0`, `reserve_neardup_fraction ≤ 0.01`, `phase_disjoint`.

**6 · `filter`** — quality, driven by what was measured rather than by taste.

- **Empty and near-empty**: 810 `code` docstrings under 10 characters. Drop.
- **Boilerplate**: the measured top repeats are template noise, not signal — *"auto generated
  code"* ×1,777, *"return a json dictionary representing this model."* ×286, *"stub"* ×260,
  *"banana banana"* ×54. A frequency-threshold filter on the normalised anchor removes these;
  record what it removed so the threshold can be argued with.
- **Length**: recorded per slice, and **honestly**. `code` docstrings exceed `max_len` 96 in
  31.6% of rows and code bodies in **93.9%**. That is not a filter's problem to solve —
  filtering to what fits would destroy the corpus. It is an argument for P11.4 progressive
  sequence length, and the filter's job is to *record* the truncation rate in the receipt so
  the number is visible rather than implied. `retrieve` passages exceed it in 14.7% of rows;
  `compress` in 0.18%.
- **Degenerate repeats** within a row, and unicode-replacement-character density as a proxy
  for encoding damage.

Gates: `truncation_rate_recorded` (not thresholded — recorded), `filters_declared`.

**7 · `assemble`** — write phase-sliced shards, one constituent per shard, with per-shard
metadata, per-slice fingerprints, and the reserve id-lists.
Gates: `one_constituent_per_shard`, `fingerprint_written`.

**8 · `publish`** — the card, the NOTICES file, both union statements, the tombstones, the
version bump. `publish` is already in `receipt.py`'s `STAGES` tuple.
Gates: `attribution_complete` (every constituent with an attribution obligation has a notice —
mechanically checkable), `union_stated`, `no_unshippable_bytes` (**nothing in `bytes/` whose
verdict is not `REDISTRIBUTABLE`**), `version_bump_matches_diff` (a removal that ships as a
MINOR is a build failure, not a review comment).

That last gate is the one that makes §4 real rather than aspirational. Version semantics that
depend on someone remembering them are version semantics that will be wrong within three
releases.

---

## 7. What the operator's own data adds

Self-generated and self-collected data is **the only category with unambiguous redistribution
rights**, with one important correction at the end of this section. It is also the only way to
fill several gaps that no licensed source covers. Ranked by what it unblocks:

**1. Natural-object photography — unblocks `vl_latent` for release. Highest value in the
whole superset.**

The audit's conclusion is unusually blunt: *"there is no permissively-licensed replacement of
tiny-imagenet's specific breadth — balanced, dense, natural-object-category photography at
100k+ scale — because that combination of properties essentially doesn't exist outside the
ImageNet lineage under a clean licence."* The composite it recommends buys breadth **across**
domains at the cost of depth **within** the one domain that vision benchmarks measure, and it
says the resulting model *"should be expected to be measurably weaker at fine-grained
natural-object recognition."*

Operator-shot photography is the only path that closes that gap rather than working around it.
And the requirements are modest by the standards of this fleet: the incumbent is 200 classes ×
500 images at 64×64. A few thousand images across a few dozen classes, shot deliberately for
class balance and pose variation, is a weekend of work that produces a constituent nobody else
has and that the operator can license however they like. Given that P12 argues vision may be
the **primary interface** rather than a side quest, and that every GPU-hour spent scaling
`vl_latent` on tiny-imagenet is spent on weights that cannot be released, this is the highest
value-per-hour item on the list.

Specification: EXIF retained as provenance (capture date, device) rather than stripped, since
the operator is the rights holder and the metadata is provenance rather than exposure; a
declared class taxonomy; a shot-count floor per class recorded as a balance gate; and the
64×64 processed derivative shipped as bytes alongside the originals' manifest.

**2. Financial-domain retrieval — closes a gap the licence work is about to open.**

FiQA is the only financial-domain source in `retrieve`, it is CC BY-SA, and it may be
StackExchange-derived with per-item attribution that *"cannot be satisfied for training data
at all."* §2 moves it to the reserve, which is the right call for measurement — and leaves the
**training** side with no financial domain whatsoever.

That the domain matters is visible in the data: the measured top-repeated `retrieve` queries
are all FiQA financial questions — *"what options do i have at 26 years old, with 1.2 million
usd?"* ×23, *"why buy insurance?"* ×23. Operator-collected or operator-authored financial QA
pairs fill this with unambiguous rights, and they are domain-matched to a reserve that stays
FiQA, which is exactly the train/eval relationship that is *supposed* to hold.

**3. Screenshot and display corpus — matches the deployment target, and no licensed source
exists by definition.**

P11.6 wants composite-image training, and its strongest argument is that *"a screenshot IS a
collage … if the end state is a model reading a display, 'many elements in one frame, find the
relevant one' is the NORMAL case and training on isolated 64×64 tiles is the artificial one."*
Nobody publishes a licensed corpus of the operator's own desktop, tooling and terminals. Self-
captured screenshots are unambiguously the operator's, are exactly on-distribution for the
autodev end state, and come with free structural labels (which window, which element, which
region should handle it) that a scraped corpus would need annotation to produce.

**4. Code from the operator's own repositories — the `code` region with zero licence risk.**

The `code` slice's whole problem is per-repository licence provenance. The operator's own
repositories — CogSynDelta, tritter, rust-ai-core, memory-gate, bitnet-rs, ternary-rs,
triton-bridge-rs, trit-vsa, tritter-accel, mycelium, hypha — are theirs to license.
Docstring↔function pairs extracted from them are the same shape as CodeSearchNet with none of
its chain. Scale is small next to 455k pairs, but the **rights are total**, the domain matches
the autodev target, and it is the natural home for the multi-language breadth P2.5 asks for
(Python **and** Rust, from the same author, with consistent style).

**5. Router labels — Phase 2 has no other source.**

Which region should handle a given input is a judgement about *this* architecture. No public
corpus encodes it. The labels are the operator's by construction.

**6. Reserve material for `classify` and `reason` — the acknowledged gap.**

§2 records that neither slice has a spare cleared constituent to reserve. Operator-authored
eval sets are the current answer, with the standard caveat that a self-authored eval measures
what its author thought to test.

### The correction: self-generated is not automatically clean

**Data generated by a third-party model inherits that model's terms of use, and many such
terms forbid using outputs to train competing models.** So "we made it ourselves" is only an
unambiguous grant when the generator was one the operator may use for that purpose.

This is live, not hypothetical. The audit's single best `compress` replacement candidate,
`hkust-nlp/SynCSE-scratch-NLI`, is MIT-tagged and its *"sentences are GPT-4/GPT-3.5-generated
from genre+topic prompts"* — and the generating models' own output-use terms are a layer the
audit did not examine. It is exactly the *mirror-more-permissive-than-upstream* pattern one
level further out: a permissive tag applied by a repackager over material whose real
constraint lives with a party who never appears in the licence chain. **Unverified, and worth
checking before that candidate is relied on.**

The fleet's own serving stack is the practical resolution: locally-served open-weights models
under Apache-2.0 (the Qwen family) do grant output use. So the rule is mechanical rather than
a judgement call:

> **Every synthetic row records its generator: model id, revision, licence, and that
> licence's stance on output use. A generator whose output terms are unknown produces rows
> with verdict `NEITHER`, by the same logic that makes an unknown dataset licence a rejection
> rather than a maybe.**

That is one more per-row field, on the one category of row that needs it, and it keeps the
project's best asset — its own data — from acquiring the exact defect it spent an audit
finding in everyone else's.

---

## 8. The manifest: extending the `Dataset` record

### `kind: corpus` — the missing third manifest kind

`MODEL-MANIFESTS.md` defines a tagged union over `kind: served | trained`, and its
`trainedSpec.sources[]` already requires a field it cannot yet resolve:

> `"licence_ref": { "type": "string", "description": "Corpus manifest id whose verdict must be TRAIN_OK." }`

**There is no corpus manifest kind. This design supplies it.** That is the coherent
integration: the superset is not a parallel scheme, it is the third arm of the union the
manifest design already anticipated. A `trained` manifest's `licence_ref: corpus/gooaq`
resolves to a real file, `validate` can check the verdict rather than trusting a string, and
the provenance chain corpus → checkpoint → quantized artifact → served endpoint becomes a
graph you can walk end to end — which the manifest doc names as *"the single strongest
argument for a shared envelope."*

### What is inherited unchanged

**Every field of the `Dataset` dataclass in `scripts/csd-corpus-expand.py` survives verbatim**,
because each one encodes a lesson that was paid for: `repo_id`, `region`, `license`, `verdict`,
`why`, `upstream`, `config`, `splits`, `caveat`, `columns`, `revision`, `data_files`,
`builder`. The `upstream` field's docstring is the reason the whole scheme works and it is not
paraphrased:

> *"A mirror's tag is not evidence about its upstream."*

`data_files` and `builder` stay because the escape hatch they encode is still needed
(`PolyAI/banking77`'s retired loading script with no Parquet conversion). `revision` stays
because `refs/convert/parquet` pins are still needed (`codeparrot/apps`). Removing either
would re-break something that currently works.

From `MODEL-MANIFESTS.md` the envelope is inherited whole: `id`, `status`, `owner`, `updated`,
`source.{upstream,revision,sha256,bytes}`,
`licence.{observed,verdict,checked_utc,evidence,upstream_says,caveat}`, `measured[]`,
`estimated[]`, `history[]`. Same names, same meanings, same rules — including *"no resource
number may appear outside `measured` or `estimated`"* and *"a measurement is never
overwritten, only superseded."*

### What is added

```yaml
schema: model-manifest/v1
id: corpus/pxhere
kind: corpus                       # the third kind
status: planned                    # planned | active | withdrawn | retired
owner: CogSynDelta
updated: 2026-09-02

source:
  upstream: huggingface.co/datasets/nyuuzyou/pxhere
  revision: <commit>                                   # never a moving tag
  licence_source: https://pxhere.com/en/terms          # NEW: where the GRANT is recorded,
                                                       # which is often NOT where the bytes are

licence:
  observed: "cc0-1.0 (HF dataset card)"
  verdict: TRAIN_OK                                    # the existing gate, unchanged
  checked_utc: "2026-09-02"
  upstream_says: >
    pxhere.com/en/terms upload clause: "By uploading, You release Images under
    Creative Commons CC0 into the public domain" — a platform-enforced condition
    of every upload, not a metadata tag over content that was never uniformly licensed.
  evidence: "docs/design/LICENCE-FOR-OPEN-WEIGHTS.md#replacement-vision-corpora"

  # --- NEW: the third question ---
  redistribution: REDISTRIBUTABLE                      # REDISTRIBUTABLE | FETCH_ONLY | NEITHER
  redistribution_basis: legal                          # legal | policy   (only for FETCH_ONLY)
  redistribution_why: >
    CC0 waives all conditions. The grant is from the platform's own upload terms,
    which bind every contributor, rather than from a compilation-level tag.
  conditions: []                                       # attribution / notice / share-alike /
                                                       # modification-statement / opt-out-tracking
  attribution: ""                                      # verbatim TASL string, when required
  reverify_after_days: 365                             # the continuing-obligation heartbeat

superset:                                              # NEW: the corpus body
  phase: phase1/vl_latent                              # exactly one. reserve/* is a phase.
  cap: 100000                                          # same field name as trainedSpec.sources
  cap_why: >
    Balance, not speed. Uncapped at ≈1.1M it would be 60%+ of the vision slice and
    put EuroSAT under 2% of every batch.
  ship_bytes: true                                     # gated on redistribution == REDISTRIBUTABLE
  ship_form: processed-64x64                           # source | processed-<spec> | manifest-only
  normalise_rules: vl/resize-64-centrecrop             # declared, not hardcoded
  row_id_basis: [constituent_id, revision, source_row_index]
  notices: notices/pxhere.txt

measured:
  - what: rows
    value: 1100000
    unit: rows
    on: 2026-09-02
    host: akula-prime
    method: "HF datasets-server /info; NOT fetched"
    # ...conditions, from_receipt, supersedes as in model-manifest/v1

history:
  - on: 2026-09-02
    what: "Identified as the closest clean substitute for tiny-imagenet's breadth"
    why: "Only permissive source found with natural-photograph diversity at 100k+ scale."
```

Fields added and the evidence for each:

| field | why it exists |
|---|---|
| `licence_source` | The pixel-mirror problem: dSprites' pixels and its Apache-2.0 LICENSE are in different repos; CLEVR's and Caltech-256's likewise. Without this the manifest cannot say where the grant was read. |
| `redistribution` | The whole of §1. |
| `redistribution_basis` | Separates *may not ship* from *chose not to ship*, which are different decisions with different reversibility. |
| `conditions[]` | Makes the union of §3 **computable** over any subset, rather than a paragraph someone has to re-derive. |
| `attribution` | The verbatim TASL string, so `publish`'s `attribution_complete` gate is mechanical. |
| `reverify_after_days` | §4's heartbeat. A verdict without a cadence is a verdict that will silently age out. |
| `phase` | §2's structural separation. Exactly one; `reserve/*` is one of them. |
| `cap` / `cap_why` | Taken verbatim from `trainedSpec.sources` so the two manifests cannot disagree about what a cap means. |
| `ship_bytes` / `ship_form` | The `no_unshippable_bytes` gate, and the CC0-processed-derivative decision from §1. |
| `normalise_rules` | The CodeContests-prefix and SQuAD-title lessons, declared per constituent rather than hardcoded in a deduper. |
| `withdrawn_in` / `tombstone` | §4's removal record. |
| `generator` (synthetic only) | §7's correction. |

### The gate, unchanged in spirit

`csd-corpus-expand.py`'s docstring states the rule and it is worth keeping the wording:

> *"`fetch` refuses any entry whose verdict is not TRAIN_OK, and there is deliberately no flag
> to override that."*

The superset adds a **second** refusal with the same property and no override:

> **`publish` refuses to place bytes in `bytes/` for any constituent whose `redistribution` is
> not `REDISTRIBUTABLE`.**

Two gates, two questions, both structural. The reason there is no override on either is the
same reason there is none today: *"Those decisions are worth exactly as much as the mechanism
that keeps them enforced."*

---

## 9. Build order

Ordered by value-per-risk, and each step is independently worth doing. Nothing here requires a
GPU except where stated, and nothing before step 5 fetches anything.

| # | step | why here | risk |
|---|---|---|---|
| 1 | **Write the corpus manifests for what already exists** — the 12 catalogued entries and the 4 in-use corpora, carrying the `redistribution` verdicts from §1. | Zero risk, changes nothing, and it is the evidence base for every later decision. Also gives `trainedSpec.licence_ref` something to resolve to. | none |
| 2 | **`csd-superset validate`** — schema, phase-disjointness, `licence_ref` resolution, attribution completeness. Read-only. | Same argument as the manifest design's `verify`-before-generator: it is where this design either justifies itself or gets right-sized. If it finds nothing, steps 3+ are lower priority and **that is a real result**. | none |
| 3 | **Resolve GooAQ with AI2.** One email. | It decides 77.8% of `retrieve`, it is the highest-value unknown on record, and no amount of further inference substitutes for it. It also decides whether `esci` has to carry the whole retrieval slice. | none |
| 4 | **Specify the operator's own constituents** — the photography taxonomy, the financial-QA shape, the screenshot capture protocol, the own-repo extraction. | The only category with unambiguous rights, and the only fix for `vl_latent`. Specification is free; collection is the cost. | none |
| 5 | **Build the polish pipeline against one slice**, `phase1/reason` (gsm8k + aqua_rat). Includes P2.5a — the overlap gate, which does not exist. | Both constituents are `REDISTRIBUTABLE`, both are small (28 MB total), and the slice can go end-to-end to a published artifact. `reason` is not yet wired into the runner (P2.2) and that is an *advantage* here: the corpus pipeline is independent of the trainer, so this proves the whole shape without touching `pretrain.py`, which P0.1/P9.1–P9.3 are already queued against one-at-a-time. | low |
| 6 | **Resolve the code slice's per-repo licences.** Try recovering `_licenses.pkl` first; fall back to the GitHub API. | Converts `code` from `NEITHER` to `FETCH-ONLY` with a shippable NOTICES sidecar, and removes the GPL/CC-BY-SA incompatibility of §3. One-off, hours. | low |
| 7 | **Fetch and process the clean vision composite**, capped, to 64×64. | The first genuinely shippable multi-constituent slice, and the one that unblocks a releasable region. Ships as CC0/permissive bytes. | moderate — storage |
| 8 | **Assemble and publish v0.1.0** — the reason slice, the vision slice, the operator's own data, plus manifests for everything else. | A real artifact that makes only claims it can defend. | — |

Note what is **not** here: no step fetches a constituent whose verdict is unresolved, and no
step publishes bytes before the `no_unshippable_bytes` gate exists. Both orderings are
deliberate.

---

## 10. What I could not determine

Listed rather than guessed, in the manner of the audit this document builds on. Several are
inherited from it and unchanged; the ones marked **new** are raised by the redistribution
question specifically.

0. **Whether GooAQ's Apache-2.0 LICENSE or its README's non-commercial NOTE governs.** Still
   the highest-value unknown, and now larger: it decides 77.8% of a region's training corpus
   *and* whether a 3.0M-row constituent can ever be shipped. One email to AI2.
1. **Whether a merged, deduplicated, filtered, re-split multi-source superset is a
   "collection" or an "adaptation"** under CC BY-SA. **New, and load-bearing** — it decides
   whether share-alike propagates to the whole and therefore whether the policy exclusion in
   §1 is a precaution or a necessity. Needs a human, probably a lawyer.
2. **Whether CC declared GPLv3 one-way compatible with CC BY-SA 4.0**, and in which direction.
   Recalled, **not verified in this session.** It determines whether the GPL/CC-BY-SA pair in
   §3 is a true incompatibility or merely a one-directional constraint. Verify before citing.
3. **Whether The Stack's opt-out obligation works as the brief describes.** Taken from the
   brief; not independently verified here. It is the exemplar for the continuing-obligation
   class, and the class shapes §4 regardless of the specific example's details.
4. **Whether FiQA is StackExchange-derived.** Inherited from the audit and it matters more for
   redistribution than for training: Stack Exchange's per-item attribution (link the original
   question, name each author, link each profile) is unsatisfiable from a dataset, so a
   positive answer moves FiQA from `FETCH-ONLY (POLICY)` to `NEITHER` outright.
5. **Whether `castorini/mr-tydi` and `miracl/miracl`'s Apache-2.0 tags survive their Wikipedia
   provenance.** **New.** P2.4 records the mix as share-alike-free; if these two are
   Wikipedia-derived — and MIRACL certainly is — that claim does not hold and the clean
   retrieval problem is materially harder than P2.4 concluded.
6. **The output-use terms of the models that generated `hkust-nlp/SynCSE-scratch-NLI`.**
   **New.** MIT-tagged, GPT-4/GPT-3.5-generated, and the generator's own terms are an
   unexamined layer in the chain. This is the audit's single best `compress` candidate, so the
   answer matters.
7. **Licences for HumanEval, MBPP and `sentence-transformers/stsb`.** Proposed as reserve
   constituents in §2 on the strength of `csd-regions.json` calling two of them eval sets. **No
   verdict is recorded for any of the three in this repository.** Do not assume; audit at
   mirror and upstream.
8. **Whether APPS' and CodeContests' permissive tags reach their third-party competition
   problem text.** Inherited. The project already answered a structurally identical question
   `REJECTED` for `hendrycks/competition_math`; consistency argues for `FETCH-ONLY` here, but
   the call is the operator's.
9. **Whether a downsampled derivative of a CC BY image inherits per-image or dataset-level
   attribution.** **New, and practical.** For Quick Draw (50M anonymous sketches) only
   dataset-level attribution is possible and Google's grant is dataset-level, so it is
   probably fine. For Caltech and CLEVR the same reasoning applies more weakly. Not resolved.
10. **What a licence-clean, non-circular encoder for the semantic dedup stage should be.** §6
    requires one and this project does not have one. It is its own small licence question,
    and until it is answered the stage runs in its "acceptable" mode (a different region's
    encoder, with the domain mismatch recorded) rather than its preferred one.
11. **Whether `1aurent/PatchCamelyon`'s CC0 and `biglam/british-library-book-images`' CC0/PDM
    survive a fresh primary-source check.** Both were flagged unverified at the second hop in
    the audit because the sources were unreachable. Both are proposed as **shipped bytes**
    here, which raises the bar: an unverified grant is tolerable for training and is not
    tolerable for redistribution. **Verify before shipping either.**
12. **Whether the measured leakage figures survive a non-circular encoder.** The 53.7% /
    24.2% / 9.0% holdout near-duplicate fractions were all measured with **each region's own
    trained encoder**, which §6 argues is circular. A non-circular encoder will produce
    different numbers, and the direction is not obvious: a circular encoder should
    *under*-detect the redundancy it was trained through, which would make 53.7% a floor —
    but a mismatched-domain encoder produces noise in both directions. The gate thresholds in
    §2 are set against figures that will move once the tooling is built properly. **New.**
13. **Whether trained weights are derivative works of training data.** Unchanged, unsettled,
    and inherited. It does not affect the *dataset* superset's shape — data redistribution is
    a separate and much better-defined question — but it governs what the model card built on
    the superset must say.
