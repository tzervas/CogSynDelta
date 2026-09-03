# Dataset factory — ground truth

Source: `CogSynDelta` at `ef3e18049044f35a5fe4cf63b56fc1d800994ac` (main, read-only).
Everything tagged VERIFIED below was read directly from that tree or from a live `du`/`find`
on `/mnt/bulk/csd-corpus` (host `akula-prime`) this session — metadata and sizes only, no
dataset content was opened. Everything tagged INFERRED is this document's synthesis, not a
quote.

---

## (a) Faculties / submodels

Regions are **faculties, not task domains** (REGION-TAXONOMY §1.1-1.3, ratified rev 3.4,
DEC-01..DEC-62). Ten verdicts on the seven legacy regions, plus new/placeholder faculties.

| faculty (region key) | verdict | consumes / emits | trains on today (dataset — rows — licence verdict) | on-disk size | target volume at 1B-param / ~1e10-token scale |
|---|---|---|---|---|---|
| **`language_code`** (was `code`) | KEEP, rename | consumes: docstring/prose text; emits: code + pooled/token latents | `Nan-Do/code-search-net-python` via local `code/apps` (codeparrot/apps, MIT, TRAIN_OK) + `code/code_contests` (deepmind/code_contests, CC-BY-4.0, TRAIN_OK) — 455,243 rows CodeSearchNet-lineage, 1 source, N_eff=1.00, **B1/B2 FAIL** (100% single-source) | **VERIFIED**: `code/apps` 18G total dir shared with code_contests | Target composition table (CORPUS-CONTRACT §1.1): multi-language cap scheme (Python 27,600/16.67%, PHP 27,600/16.67%, etc.) to kill the monolingual defect; scale-path needs ~1e10 tokens, current corpus is single-language, single-source, truncation-dominated (93.9% of code side exceeds max_len 96) |
| **`memory`** (MERGE of `retrieve` + `compress`, hippocampus: retrieval + consolidation ops) | MERGE | consumes: query/passage or premise/hypothesis text; emits: retrieval/consolidation latents | `retrieve`: gooaq (NC, ACCEPTED under DEC-31), squad (CC-BY-SA-4.0), hotpotqa (CC-BY-SA-4.0, split corpus/queries+qrels) — **NC tier**, B1 FAIL (79.1% GooAQ post-dedup); `compress`: sentence-transformers/all-nli (SNLI, BLOCKING as trained, repairable to SHARE_ALIKE by dropping non-SNLI pairs) | **VERIFIED**: `retrieve/` 961M (squad, hotpotqa-corpus, hotpotqa-queries); `compress/` 19M (snli) | Merged faculty inherits the union (stricter) obligation set = **NC**. Needs a real BM25-scale passage pool (retrieve.py's 57,638-passage FiQA eval is the gate, not the 512-pair diagonal, which is saturated) |
| **`visual`** (was `vl_latent`) | KEEP, rename | consumes: image patch tokens; emits: JEPA-style latents (EMA target encoder is the deployed half) | `zh-plus/tiny-imagenet` — **BLOCKING** (built from ImageNet without a licence grant); `nlphuji/flickr30k` planned for P3.2, also **BLOCKING** | **VERIFIED**: `vl/` 390M (oxford-iiit-pet, fashion_mnist — replacement candidates, not what's trained on) | Corpus is unreleasable as trained; replacement-vision-corpora candidate table exists (LICENCE doc §"Replacement vision corpora") but "clean at both mirror and upstream" set is small; this is the region closest to zero usable licensed data today |
| **`reasoning`** (was `reason`) | KEEP, rename | consumes: problem text; emits: reasoning latents | gsm8k + aqua_rat — receipt was **deleted**, prose-only figure (r@1 0.0801 vs 0.0039); W2c (2026-09-03) retired the disputed ≈0.40 untrained floor, measured 0.2285 instead — `apps`/`code_contests` survive as reserve source | **VERIFIED**: `reason/` 29M (aqua_rat-raw, gsm8k-main) | Straddles the still-PLACEHOLDER `numeric/math` faculty; splitting today starves both. Natural next source: the reserve's executable items (test cases carry exact numeric ground truth) |
| **`classify_banking77`** | PROBE (not a region) | intents label space | banking77, CC BY 4.0-ish OSS, well-balanced (B5 pass) | `classify/` 3.3M (banking77, go_emotions-simplified) | Retained frozen as a damage-detector probe, not scaled |
| **`classify_go_emotions`** | PROBE of placeholder `affect` faculty | 28-label BCE over Reddit comments | go_emotions — **fails B5** (32.76% max label share, 184.7:1 max:min) | (same dir as above) | `affect` faculty is PLACEHOLDER (`emits: "global"`, a `[B,8]` gain vector, not a workspace participant) — not scaled until built |
| `residual_mlp` | RETIRE | — | synthetic, declared corpus `"synthetic"` | — | n/a — survives only as `x + f(x)` primitive |
| `stream_vae` | RETIRE as region → PRIMITIVE (tract codec) | — | synthetic | — | n/a — becomes `interconnect.primitives[]` |
| retrieve.py (untracked, BEIR-style) | SPLIT | its FiQA eval (57,638-passage pool) becomes `memory`'s gate; its FiQA-only 14,131-pair training regime RETIRED | — | — | — |
| **`white matter` / interconnect** | BUILD NOW | consumes: every region's workspace latents via cross-attention (never re-tokenized — DEC-47); emits: `Schedule` (which regions run, intensity, priority, attention split, context budget `c_r`, topology) + write-back | none yet — trained on the **reserved corpus** (§5, NSRS-admitted cross-faculty items) once regions are frozen | n/a (no dataset dir; corpus is derived from the four encoding regions' own eval-adjacent pool) | Reserve sizing at toy scale: 67,584 items (6,144 compose-eval + 61,440 interconnect-train, ≥60% cross-faculty), drawn from 165,065 clean-unallocated pool (40.9% utilization). Scale-path: this is where "cross-region items that no single region already answers" must grow with region count |
| **frontal cortex** | BUILD NOW, inside white matter | its own objective (§3.1) | — | — | — |
| **thalamus** | BUILD NOW, afferent-bandwidth gating | content gating is a declared seam, not built | — | — | — |
| **`episodic_store`** | BUILD, REQUIRED, phase 2 (DEC-49, supersedes DEC-32) | non-parametric key/value store over workspace latents, `(scope, domain, logical_key)` partition, importance+GPU-residency+staleness eviction (staleness fn is a GAP), read via cross-attention (never re-serialized to tokens) | none — **no `s_r` computed for it** (exempt from NSRS admission freeze; §1.3 +1) | — | Capacity is **dynamic**: VRAM minus KV/activation reserve per host, formula in §8 gap (a); contract lifted clause-by-clause from `memory-gate`/`memory-gate-rs` (VERIFIED at their own pinned commits) with open gaps: partition axis beyond `domain`, staleness decay function, byte-vs-item capacity unit |
| **`auditory`** | DECLARED SEAM, DEFERRED to production phase (DEC-48, supersedes rev-3.1's "REQUIRED phase 1") | spectrogram patch tokens (same interface as `visual`) → masked-latent JEPA objective | none catalogued for training yet | **VERIFIED**: `auditory/` **76G on disk already** — multilingual_librispeech, libri-light, musan, ami, peoples_speech-clean, voxpopuli-en, FSD50k (fetched groundwork, not yet trained on) | v1 clean-tier mix recommended: CC BY 4.0 (LibriVox+VoxPopuli+AMI+Freesound-filtered+MUSAN, LibriVox as ONE provenance group per DEC-46); NC tier available at zero extra release cost once `memory` is already NC |
| **`speech_output`** | DECLARED SEAM, DEFERRED with `auditory` (a head on the language centre, not a faculty/participant) | two streams from one trunk: BPE text + discrete speech-token stream for a swappable synthesiser | none catalogued for training yet | **VERIFIED**: `speech_output/` (small, mostly manifests) — css10, AISHELL-3, m-ailabs (8-lang subset excl. Ukrainian), lj_speech, hifi-tts, expresso, libritts_r, hifitts-2 | Clean tier CC BY 4.0 (LJSpeech, VCTK, Hi-Fi TTS, CSS10, M-AILABS 8-lang, AISHELL-3); **HiFiTTS-2 excluded, exclusion declared**; ND (TED-LIUM) refused as a class |
| **`salience`** (AI-specific/limbic) | DECLARE interface, not trained v1 | `salience.value(emission, control_state) → [B,1]`, feeds scheduler priority | — | — | — |
| **language (trunk)** | PLACEHOLDER | generative head exists (`model/causal_lm.py`, 283 lines); needs foundation corpora (curriculum step 4) | — | — | Deferred candidate: official docs corpus (Python/Rust docs already staged under `official-docs`, plus RAG collections + vault) — needs enrichment, structure extraction, version tagging, dedup, per-doc-set licence verdict — revisit only when this row starts |
| **numeric / math** | PLACEHOLDER | tokens over **values**, not BPE pieces | — | — | Blocked on corpus, not design; natural source is reserve's executable items |
| **affect** | PLACEHOLDER | `[B,8]` global gain vector, multiplicative on every tract | `classify_go_emotions` is a probe of it | — | — |
| explicitly NOT a faculty | — | motor cortex (operator ruling: deferred, "wholly unnecessary until likely post the Rust reimplementation"); schedule trace (it's a log, not a region) | — | — | — |

**Composed-model licence, current state:** `memory` carries GooAQ's NC term (77.8-79.1%
concentration) under DEC-31's strictest-input rule, so the **entire composed model is
CC BY-NC-SA today** — accepted policy per the 2026-09-02 operator decision, not a defect. Two
regions (`language_code`'s CodeSearchNet lineage, `visual`'s tiny-imagenet/flickr30k) are still
outright `BLOCKING` (no licence grant at all) and must be replaced or repaired before those
regions can ship, independent of the NC question.

---

## (b) Catalogue entry shape the factory should emit

**Current state, VERIFIED from `scripts/csd-corpus-expand.py`'s `Dataset` dataclass:**
`repo_id, region, license, verdict, why, upstream="", config="", splits=("train",),
caveat="", columns=(), revision="", data_files={}, builder="csv"`. `upstream` already holds
"what the ORIGINAL source says, when it is not the same repo as the card" — i.e. the
mirror-vs-upstream split DEC-57 asks for is already a real field, not aspirational.

**Gap (INFERRED, not yet true of the code):** DEC-57's prose says catalogue entries "already
carry `provenance_group`" — this field **does not exist** in the current dataclass. It is a
design intent for the factory, not a shipped field. Any surveyor building factory tooling
must add it, not assume it.

**The shape a new surveyor/factory pass should emit, reconciling DEC-57 (P2′f), the current
dataclass, LICENCE-FOR-OPEN-WEIGHTS.md, and AUDIO-CORPUS-AUDIT.md:**

- `repo_id` / source identifier
- `region` / faculty key (post-taxonomy: `language_code`, `memory`, `visual`, `reasoning`,
  `auditory`, `speech_output`, or a new placeholder)
- `provenance_group` — datasets sharing one lineage (e.g. all LibriVox-derived audio sets)
  collapse to ONE group for B1/B2/B5 purposes (the LibriVox precedent, DEC-46)
- `mirror_tag` — the licence string as it appears on the HF card / mirror, held **separately**
  from:
- `licence_upstream` — the licence text read at the **primary source**, verbatim, quoted
- `licence_upstream_source` — the URL actually fetched, **plus fetch date** (A0m's gate:
  FAILS if absent, if its host equals the mirror host, or if the fetch date is missing)
- `grant_scope` ∈ `{whole_corpus, metadata_only, code_only, unstated}` — catches cases like
  AudioSet (metadata_only) and CSS10/Libri-Light (code_only) that a licence string alone hides
- `verdict` ∈ `{PERMISSIVE_OK, ATTRIBUTION, SHARE_ALIKE, NC, BLOCKING}` — five classes, reused
  unchanged from `LICENCE-FOR-OPEN-WEIGHTS.md` into `AUDIO-CORPUS-AUDIT.md` with `NC` split out
  as its own non-blocking category per DEC-31. **Recommended addition, not yet adopted:**
  `CONSENT_OPEN` (OD-11) for CC0/public-domain material subject to live consent revocation
  (Common Voice's shape) — flag any surveyor hitting this shape rather than filing it
  `PERMISSIVE_OK`.
- `usage_tag` — orthogonal to `verdict`: **`EVAL-ONLY`** for a held snapshot / unverified
  chain that is defensible for evaluation but not training (structural: no published
  parameter may derive from it)
- `size` — in the corpus's natural unit (rows for text, **hours** for audio — B1/B2 for audio
  are computed in hours, not row/clip counts)
- `redistribute` flags: NC / SA / ND / attribution, independently of `verdict`
- `provenance_red_flags` — free text (e.g. "YouTube-scraped, no consent process described",
  "distributor disclaims owning the audio")
- `why` — one-line justification, matching current field
- `caveat` — matching current field (truncation risk, mixed splits, etc.)
- REFUSE is **enforced by the fetcher**, not merely recorded beside the entry (current
  `fetch()` already does this: any non-`TRAIN_OK` verdict is refused with no override flag)
- a **human-visible provenance receipt per dataset**, emitted at ingest, reproducing: licence
  verdict, provenance group, row/hour count, dedup removals

**Verdict class definitions (VERIFIED, `LICENCE-FOR-OPEN-WEIGHTS.md` §"Verdict categories",
extended by `AUDIO-CORPUS-AUDIT.md` §"Verdict categories"):**

| verdict | meaning |
|---|---|
| `PERMISSIVE_OK` | MIT/Apache-2.0/BSD/CC0/public domain at BOTH mirror and upstream |
| `ATTRIBUTION` | CC BY-family; usable, TASL notice required in the release |
| `SHARE_ALIKE` | CC BY-SA-family; whether trained weights are "adapted material" is unsettled — flagged, not resolved |
| `NC` | genuine rights-holder-granted non-commercial term; per DEC-31 does **not** block use — moves the region (and by strictest-input, the composed model) to an NC release tier |
| `BLOCKING` | no licence grant exists at all, an active ownership dispute, a paywall/membership corpus, or a restriction (ND, academic-only, no-redistribution EULA) the NC-tolerant policy doesn't absorb |
| `CONSENT_OPEN` (recommended, not yet adopted) | copyright-clean (often CC0) but subject to live, ongoing consent revocation — a class the PERMISSIVE_OK/BLOCKING binary can't express |
| `EVAL-ONLY` (a usage tag, not a licence verdict) | defensible for evaluation on a held snapshot; structurally excluded from training |

---

## (c) Balance and admission rules a new dataset must satisfy

**B1-B5 (VERIFIED, `CORPUS-CONTRACT.md` Part 3), computed on provenance groups, not shard
files, on the post-dedup / post-cap realised training set:**

- **B1 — max single-source share ≤ 0.40** (hard line 0.50; the 0.10 margin covers
  pre→post-dedup drift, which moves toward more concentration, not less)
- **B2 — effective number of sources `N_eff = 1/Σp²` (inverse Simpson) ≥ 3** for a
  general-capability region, **≥ 2** for an explicitly narrow one. Waiver clause: a region
  below threshold must either rename to what its corpus actually is, or carry a dated waiver
  in `csd-regions.json` naming the missing sources and why
- **B3 — train/eval distribution correspondence must be declared as exactly one of**
  `in-mixture` (holdout from the same pool; receipt must report per-source, not just
  aggregate) or `held-out-domain` (the eval's provenance group contributes zero training
  rows, and an in-mixture number must sit alongside it) — checked mechanically against
  `train_source_shares` vs `eval_source_shares`, not against the comment
- **B4 — a cap must be a sample (strided/reservoir), never a prefix**; the receipt records
  method and seed
- **B5 — within-source concentration**: max stratum share ≤ `max(0.25, 2/k)`, max:min
  stratum ratio ≤ 20:1, over the source's natural stratum key (repo, genre, label, language)

**NSRS admission filter (DEC-22, for the reserved/interconnect-training corpus — this is the
gate for cross-faculty material, distinct from B1-B5 which gate per-region corpora):**

For candidate item `(x, t, C)`, admit iff, using each **frozen** region's standalone score
`s_r`:
1. no single region suffices — `max_r s_r < τ_lo`
2. some combination does — `oracle_late_fusion(s) ≥ τ_hi`
3. no trivial baseline suffices — BM25 (text) / raw-pixel logistic regression (vision)
   `< τ_lo` on the same item

Thresholds are calibrated on a disjoint calibration subset, then frozen (never inherited),
and stated in chance-normalised units per bin (`(s−chance)/(1−chance)`). Exemption: the
general/out-of-scope bin is admitted under conditions (1)+(3) only (tagged `nsrs: c1c3`),
target = DEC-41's `NULL` candidate — no other bin may use this exemption. **NSRS is a
difficulty filter, not an integrity filter**: keyed split + generator identity + source-row
fingerprinting (§5.6, DEC-38/39) is what guards against poisoned rows, separately.

**Strictest-input release rule (DEC-31, csd-release-licence-decision):** a composed
faculty's licence = the **strictest** of its inputs' verdicts; NC does not block, ND/BLOCKING
does. Applies to every dataset the factory emits, not only to region training corpora.

**`s_r` invalidation rule:** any weight change to any region invalidates every recorded `s_r`
and every admission decision derived from it — so the region set must be frozen before NSRS
admission runs.

---

## (d) Which faculties are most data-starved

Ranked by how far each is from a usable, licence-clean, balanced corpus:

1. **`visual`** — outright `BLOCKING` (tiny-imagenet built on unlicensed ImageNet;
   flickr30k, the planned P3.2 addition, also `BLOCKING`). The replacement-vision-corpora
   candidate list in `LICENCE-FOR-OPEN-WEIGHTS.md` is short and several entries are
   themselves `BLOCKING (unresolved)` (EMNIST, GTSRB) or ruled out outright
   (TreeOfLife-200M, STL-10, SVHN, Food-101, Places365). This is the faculty with the least
   headroom today.
2. **`language_code`** — B1/B2 hard-fail at N_eff=1.00, 100% single-source, 93.9% of the
   code side truncated past `max_len`. A target composition (multi-language cap table)
   exists in CORPUS-CONTRACT §1.1 but every candidate language's licence status still needs
   clearing before the caps can be applied.
3. **numeric/math** — not just data-starved, not yet a region: PLACEHOLDER blocked on
   corpus, with `reasoning`'s existing corpus (gsm8k+aqua_rat) straddling both and neither
   half able to split off without starving the other.
4. **`memory`** (merged retrieve+compress) — has volume (961M+19M on disk) but it's
   concentrated (79.1% GooAQ) and NC-tagged; more sources are needed to actually clear B1/B2
   rather than ride the NC waiver.
5. **`reasoning`** — receipt was deleted and must be regenerated; W2c just retired the
   disputed 0.40 untrained floor (2026-09-03), so the reserve's main source (apps/
   code_contests) is confirmed usable, but `reasoning` itself has no confirmed-clean second
   source beyond gsm8k/aqua_rat, both of which are "spent" per §5.1.
6. **`affect`** — PLACEHOLDER; its only probe (`classify_go_emotions`) fails B5 on both
   axes (32.76% max label share, 184.7:1 max:min).
7. **`language` (trunk)** and **the reserved/interconnect corpus** — both are structurally
   blocked on the *four encoding regions being frozen first* (the `s_r` invalidation rule),
   not on source scarcity per se; the reserve pool (165,065 clean-unallocated) currently
   funds the toy-scale 67,584-item need at 1.71× surplus, but that margin does not survive a
   scale-up in participant count or reserve size.
8. **`auditory` / `speech_output`** — the *least* starved by comparison: 76G already fetched
   for `auditory` and a recommended clean-tier CC BY 4.0 mix exists for both, but both rows
   are **deliberately deferred** (DEC-48) until the composed mind passes its integration test
   without audio — so "data-starved" doesn't apply; "intentionally unstarted" does.
9. **`episodic_store`** — not corpus-starved (non-parametric, no `s_r`), but contract-gapped:
   the partition axis beyond `domain`, the staleness decay function, and the byte-vs-item
   capacity unit are all open questions the source repos (`memory-gate`/`memory-gate-rs`)
   are silent on.

At the **1B-param / ~1e10-token scale path** (operator, 2026-09-03), every faculty above is
starved relative to target: current on-disk volumes (18G code, ~1G memory, 390M vl, 29M
reason, 76G auditory-groundwork) are toy-scale groundwork, not scale-path material, and the
operator has named sourcing/licence-deconfliction as **the long pole** for that path — which
is exactly the gap the dataset factory (DEC-57, P2′f) exists to close.
