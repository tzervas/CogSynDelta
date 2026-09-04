# Visual Data Ingestion Pipeline

**Status:** design. Nothing here has been built, no image has been fetched, and
`scripts/csd-corpus-expand.py` and its `CATALOGUE` are untouched.
**Scope:** the VISUAL-SPECIFIC ingest path — sourcing, filtering, enriching, labelling and
provenance-tracking images so a licence-clean corpus can replace `vl_latent`'s
tiny-imagenet dependency.

**Reads on from, and does not re-open:**

- [`LICENCE-FOR-OPEN-WEIGHTS.md`](LICENCE-FOR-OPEN-WEIGHTS.md) §*Replacement vision corpora*
  — **source selection is settled there.** Every dataset named below was verified at both
  the mirror and the true upstream in that document. This pipeline does not re-research
  them and does not add one.
- [`MODEL-MANIFESTS.md`](MODEL-MANIFESTS.md) — manifest vocabulary. The corpus this
  pipeline emits is consumed through a `trained` manifest's `spec.sources[]`, whose
  `corpus` / `shards` / `image_column` / `label_column` / `cap` / `cap_why` / `licence_ref`
  fields already exist. Nothing here invents a second vocabulary.
- `TRAINING-SUPERSET.md` (being written concurrently) — **the general polish pipeline and
  the redistributable-versus-fetch-only question live there.** Where this document says
  "defer to the superset", it means exactly that: a visual-specific pipeline should not
  decide whether the fleet redistributes corpora at all.

**The one-sentence problem.** tiny-imagenet supplied `vl_latent` with pixels, labels,
class balance and curation as a single package under a licence that turned out not to
exist. The clean replacements supply pixels. This pipeline is the rest of the package.

---

## 0. What the corpus actually has to be

Two facts shape every decision below, and both are easy to get wrong by analogy to
ImageNet-style datasets.

**Fact one: I-JEPA does not need labels.** `src/cogsyndelta/regions/vl_pretrain.py`'s
objective is intra-image context→target prediction. Labels appear in exactly two places —
`_linear_probe`, which fits a frozen-feature classifier, and the transfer probe. So the
corpus needs **~600k unlabelled images for pretraining and ~50k labelled images for
probes**, not 600k labels. That single observation dissolves most of the circularity
problem in §4 before it is argued, and it is why this pipeline can be built without ever
running a vision model over the corpus for semantic purposes.

**Fact two: the consumer enforces nothing.** Read `vl_pretrain.py` as it stands:

```python
# _decode_split: every shard concatenated, no domain weighting
table = pq.read_table(shard, columns=[image_col, label_col])
...
x = np.stack(images).transpose(0, 3, 1, 2)

# _to_float: ImageNet's statistics, applied to every image regardless of source
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

# the training loop: uniform sampling over the concatenated pool
idx = torch.randint(0, x_tr.size(0), (cfg.batch_size,), generator=gen)
```

Three consequences that this pipeline must absorb because the trainer will not:

1. **Balance is whatever the shard row counts are.** Per-domain exposure in every batch is
   exactly proportional to raw row count. Policy in a document changes nothing; the only
   lever that works today is *what files you hand the harness*.
2. **`columns=[image_col, label_col]` projects away every other column.** Provenance
   cannot reach the tensor, and should not try to — but it must be recoverable from the
   shard the tensor came from, by a mapping that does not depend on reproducing the
   loader's control flow. (§1.6.)
3. **Normalisation is a fixed constant.** Histopathology stain, Sentinel-derived RGB and
   line-drawing black-and-white do not share ImageNet's colour statistics, and nothing in
   the code can give them their own. (§5.4.)

A fourth, smaller, and worth fixing at some point:

```python
tag = str(abs(hash(key)) % (10**16))     # _decode_split cache key
```

`hash()` over a `str` is salted by `PYTHONHASHSEED` and differs per process. The decode
cache under `cfg.cache_dir` therefore **never hits across separate invocations** — every
run re-decodes from parquet. That is why this pipeline materialises 64×64 shards at ingest
time rather than treating decode as a train-time cost: the decode should be paid once, in
a stage that leaves a receipt, not silently on every run. *(Bug noted, not fixed here — it
is a one-line change to `hashlib.blake2b`, and it belongs in a code PR, not a design doc.)*

### The sources, as settled

From the licence audit, verified at mirror and upstream. **Do not re-research these.**

| source id | repo / archive | licence | images available | native | domain |
|---|---|---|---|---|---|
| `pxhere` | `nyuuzyou/pxhere` | CC0-1.0 | ≈1,100,000 | full camera | general photography |
| `bl-books` | `biglam/british-library-book-images` | CC0 / PD Mark | 1,080,814 | full scan | book plates, engravings |
| `pcam` | `1aurent/PatchCamelyon` | CC0-1.0 | 327,680 | 96×96 | histopathology |
| `shapes3d` | `google-deepmind/3d-shapes` | Apache-2.0 | 480,000 | **64×64** | synthetic 3D, one object |
| `dsprites` | `google-deepmind/dsprites-dataset` | Apache-2.0 | 737,280 | **64×64**, 1ch | synthetic silhouettes |
| `clevr` | `laion/clevr-webdataset` | CC BY 4.0 | 100,000 | 480×320 | synthetic 3D, multi-object |
| `fashion-mnist` | `zalando-datasets/fashion_mnist` | MIT | 70,000 | 28×28, 1ch | garment product photos |
| `eurosat` | `timm/eurosat-rgb` | MIT (+ Copernicus notice) | 27,000 | **64×64** | Sentinel-2 land use |
| `quickdraw` | `google/quickdraw` | CC BY 4.0 | 50,426,266 | 28×28, 1ch | sketches, 345 classes |
| `caltech256` | `data.caltech.edu/records/nyy15-4j048` | CC BY 4.0 | ≈30,607 | variable | everyday-object photos |

`quickdraw` and `caltech256` are **probe-only** and must never enter the pretraining pool.
That separation is a gate in §3.5, not a convention.

---

## 1. Acquisition and provenance capture

**This is the stage the rest of the pipeline is built to protect.** Everything after it is
recoverable by re-running; nothing after it can recover what this stage failed to write
down. A directory of JPEGs is not a corpus, it is an archaeological site.

### 1.1 Inputs / outputs

| | |
|---|---|
| **Input** | A `VisualSource` record (§1.7) that has passed the licence gate; an HF token; a target root under `/bulk/csd-corpus/vl`. |
| **Output** | The **L1 tier**: parquet shards of canonicalised 256×256 images with a full provenance column block in the same row; `SOURCES.json`; `ATTRIBUTION.jsonl`; a `SHARD.json` per file; an `ingest` receipt. |
| **Not output** | Full-resolution originals for `pxhere` and `bl-books`. See §1.5. |

### 1.2 The per-image provenance record

Columns, carried **in the same parquet row as the pixels**. Types are Arrow types.

```
# identity — computed once, at fetch, never recomputed
image_id                string    "sha256:<64 hex>" over the ORIGINAL fetched bytes
origin_sha256           string    identical to image_id; the join key for all time

# where it came from
source_id               string    key into SOURCES.json, e.g. "pxhere"
source_repo             string    "nyuuzyou/pxhere"
source_revision         string    HF commit sha, >= 7 chars. A TAG IS NOT A PIN.
source_config           string
source_split            string
source_row_index        int64     position within the upstream shard
source_shard            string    upstream file name, when the source ships named shards
source_item_url         string?   null when the source publishes none
source_item_id          string?
creator                 string?
creator_url             string?
title                   string?

# when, and by what
retrieved_utc           string    RFC3339, stamped at FETCH, not at write
retrieved_by            string    "csd-visual-ingest/<semver>"
fetch_receipt           string    path of the receipt for this fetch run

# licence — the corpus gate's vocabulary, verbatim
licence_id              string    SPDX-ish key into SOURCES.json: "CC0-1.0", "CC-BY-4.0"
licence_observed        string    the string AS IT APPEARED, verbatim
licence_verdict         string    TRAIN_OK | EVAL_ONLY | REJECTED
licence_checked_utc     string
licence_upstream_says   string    non-empty when the source re-hosts someone else's work
attribution_required    bool
attribution_ref         string    key into ATTRIBUTION.jsonl
provenance_granularity  string    "item" | "source"   -- see 1.4

# the original, before anything touched it
original_bytes          int64
original_width          int32
original_height         int32
original_format         string    "JPEG" | "PNG" | "raw-uint8" | ...
original_mode           string    "RGB" | "L" | "P" | "CMYK" | ...

# what has happened since
lineage                 list<struct{
                            stage:        string,
                            tool_version: string,
                            utc:          string,
                            params_sha256:string,
                            receipt:      string,
                            note:         string }>
```

`lineage` is **append-only**. A stage adds an entry; no stage rewrites one. That single
rule is what makes "resize, crop, dedup, format conversion" survivable: the record does not
describe the current state of the image, it describes every state the image has been in,
and the current state is the last entry.

### 1.3 Where it lives, and why not a sidecar

Four files, three of them tiny.

| artifact | scope | contents |
|---|---|---|
| **in-row parquet columns** | per image | everything in §1.2 |
| `SOURCES.json` | per corpus | one record per source: the full `model-manifest/v1` `licence` block (`observed`/`verdict`/`checked_utc`/`evidence`/`upstream_says`/`caveat`), plus the obligation list. The per-image row carries only `licence_id`; the text lives here once. |
| `ATTRIBUTION.jsonl` | per attributable unit | §6 |
| `SHARD.json` | per parquet file | row count, `sha256` of the parquet file itself, producing stage, `params_sha256`, input shard ids, per-source row counts, per-channel mean/std |

**A sidecar file per image is the wrong answer and it is worth saying why**, because it is
the answer everyone reaches for first. A sidecar dies at the first `to_parquet`, the first
`datasets.map`, the first `rsync` that copies `*.jpg`, the first `shutil.copytree` with a
pattern, and the first time someone tars the images to move them between hosts. It has no
referential integrity: nothing detects that 400 sidecars went missing. It also multiplies
inode count by two at 1M scale, which on the mechanical bulk array is a real cost. The row
*is* the sidecar, and it is the one container that every tool in this stack already moves
as a unit.

**Denormalisation is deliberate.** `source_repo`, `licence_observed` and friends repeat on
every row and compress to near-nothing in parquet (dictionary encoding over ≤10 distinct
values). The alternative — a foreign key into `SOURCES.json` and nothing else — means that
losing one small JSON file makes a terabyte of pixels unattributable. Pay the bytes.

### 1.4 Provenance granularity, and the honest null

Some sources publish a per-item URL and a contributor; some publish a collection and
nothing else. `provenance_granularity` records which, and **`source_item_url: null` with
`granularity: "source"` is a legitimate, recorded state — fabricating a plausible URL is
not.**

This matters beyond tidiness. Per-item provenance is what makes a takedown or
right-to-erasure request actionable: given a URL or a contributor id, the pipeline can
resolve to `origin_sha256` and thence to every derived row and every training shard that
contains it. Without item-level provenance, the only possible response to "remove this
image" is "remove this source". For a CC0 platform where an uploader may have uploaded
something they did not own, that is a real scenario, not a hypothetical.

**Unverified, and it matters:** whether `nyuuzyou/pxhere` ships per-item URLs, contributor
names or its free-text tags as columns. It is the largest single source and the only broad
photographic one. If it does not, `pxhere` is `granularity: "source"` and the tag-based
weak-label path in §4.4 is unavailable. **Check this before writing the fetcher**; it
changes two later stages.

### 1.5 The landing decision: L0, L1, L2

Three tiers, and the middle one is the load-bearing design choice.

| tier | what | resolution | when retained |
|---|---|---|---|
| **L0** | original bytes, exactly as fetched | native | only for sources whose entire L0 is < 50 GB |
| **L1** | canonical, aspect-preserved, longest edge 256, JPEG q90 | 256 | **always. This is the corpus of record.** |
| **L2** | training shards, 64×64 RGB, PNG-in-parquet | 64 | derived from L1, cheap to rebuild |

**Why L1 exists at all.** `/bulk` is 5.95 TB with a hard policy ceiling at 50% used
(`csd-corpus-expand.py --max-used-fraction`, default `0.5`), currently 6.7% used — roughly
2.5 TB of headroom. `pxhere` at 1.1M full-resolution photographs is, at an *estimated*
2–4 MB each (**unverified** — nobody has measured the mirror), 2.2–4.4 TB. **Landing L0
for pxhere would breach the fleet's own storage ceiling before the corpus was built.** So
full-resolution pxhere and BL originals are decoded in the fetch stream and never written
to disk at native size.

**What that costs, stated plainly.** Once L0 is discarded, the corpus cannot be re-derived
at a target above 256 without re-fetching — which for pxhere is the multi-hour download in
§8. 256 is chosen as 4× the current 64×64 target, which leaves room for a 128 or 224
experiment later without a re-fetch; it is not chosen because 256 is special.

**Storage at L1**, which is the number that makes this tier viable:

| source | images landed | L1 bytes @ ~25 KB |
|---|---|---|
| `pxhere` | 1,100,000 | ≈ 27 GB |
| `bl-books` | 1,080,814 | ≈ 27 GB |
| `pcam` | 327,680 | ≈ 5 GB (96→256 is an upscale; store at native 96 instead, ≈ 3 GB) |
| everything else | ≈ 1.4M | ≈ 20 GB |
| **total L1** | | **≈ 80 GB** |

Against 2.5 TB of headroom, and against 3.3 TB for the L0 alternative. That is the whole
argument.

### 1.6 What can go wrong, and how it is detected

| failure | how it happens | detection |
|---|---|---|
| Provenance captured *after* the fetch | Someone writes JPEGs first "to see if it works", then tries to backfill | **Structural:** the writer takes a row, not a path. There is no code path that writes pixels without the provenance block, because the parquet schema requires the columns non-null. |
| A revision recorded as a tag | `revision="main"` | Validation: `source_revision` must match `^[0-9a-f]{7,40}$` or an immutable HF ref. `main` is a validation error, per `MODEL-MANIFESTS.md`'s rule that *"a moving tag is not a pin"*. |
| Rows silently lost mid-fetch | An exception in one shard, caught and logged | **Reconciliation gate:** the source's declared row count for the split (from the HF dataset info) must equal `rows_written + rows_rejected`. Unaccounted rows fail the run. |
| A partially written shard reused as complete | Crash between `to_parquet` and the manifest write | `SHARD.json` is written **after** the parquet and contains the parquet's `sha256`. A shard without a matching `SHARD.json` reads as absent and is re-derived — the same trick `is_present()` already uses (*"a dataset already on disk with no MANIFEST.json still reads as not present"*). |
| Licence surface widened by an unattended run | A new source added with an unverified verdict | The gate is reused verbatim: `verdict != TRAIN_OK` → refuse, and **there is deliberately no override flag.** Extended for images: `attribution_required and not attribution_ref` also refuses. |
| Tensor row ↔ provenance row mapping lost | `_decode_split` skips rows with `raw is None` and truncates on `limit`, so tensor index ≠ parquet row index | Two mitigations: (a) a gate that **no L2 shard contains a null image cell**, so the skip branch never fires; (b) `index.parquet` — `(tensor_position, shard, row_index, image_id)` — materialised at ingest so the mapping never depends on reproducing the loader's control flow. 1M rows × ~50 B ≈ 50 MB. |
| The whole thing quietly attributed to nothing | An image reaches L2 whose `licence_id` does not resolve | **The orphan-pixel audit** (§7.3): sample K=1,000 rows uniformly from the final shards; resolve each forward to `SOURCES.json` and `ATTRIBUTION.jsonl`; any unresolvable row fails the corpus. Run every time L2 is rebuilt, not once. |

### 1.7 Extending the `Dataset` record — design only, not applied

`scripts/csd-corpus-expand.py` is untouched by this document. What follows is the shape the
extension should take when someone does apply it.

The **licence gate is reused as-is**. What needs adding is a modality and the image-side
provenance policy:

```python
@dataclass
class Dataset:
    ...  # every existing field unchanged

    # --- visual ---
    modality: str = "text"                    # "text" | "image"
    image_column: str = ""
    item_url_column: str = ""                 # "" = source publishes none; see 1.4
    item_id_column: str = ""
    creator_column: str = ""
    tags_column: str = ""
    native_size: tuple[int, int] | None = None
    land_at: int = 0                          # 0 = land as-is; >0 = stream-decode to
                                              # this longest edge, never landing L0
    cap: int = 0                              # 0 = uncapped. For BALANCE, not speed.
    cap_why: str = ""                         # required non-empty when cap > 0
    attribution_ref: str = ""
    obligations: tuple[str, ...] = ()         # e.g. ("cc-by-4.0", "copernicus-sentinel")
    provenance_granularity: str = "source"
```

`cap` / `cap_why` are named to match `MODEL-MANIFESTS.md`'s `trainedSpec.sources[].cap`
(*"0 = uncapped. Exists for BALANCE, not speed"*) so the same intent is not expressed twice
in two vocabularies.

**`fetch()` cannot be reused for the large image sources, and this is a real finding.** It
calls `load_dataset(..., streaming=False)` and then `data.to_parquet(...)` — it
materialises the entire split before writing. For `pxhere` that is the 2–4 TB that §1.5
just established cannot land. Image sources with `land_at > 0` need a sibling path:
`streaming=True`, decode-and-downsample per record, batch into row groups, write
incrementally. Same gate, same manifest, different transport. **Do not widen `fetch()` to
do both**; a function that is streaming-or-not depending on a field is where the
truncated-shard bugs live.

---

## 2. Quality filtering

**Inputs:** L1 shards. **Outputs:** L1 shards with quality columns appended, plus a
`rejects/` shard carrying provenance + `drop_reason` for everything removed.

### 2.1 Nothing is deleted; things are moved with a reason

The single structural rule of stages 2–5:

> A stage may **add columns** and **move rows to `rejects/`**. It may not delete a row, and
> it may not modify a provenance column.

`rejects/` rows carry the full provenance block, the computed metrics, `drop_stage` and
`drop_reason` — but **not the pixels**. At ~250 bytes/row, 1M rejects is 250 MB. What this
buys: "why is this image not in the corpus" is answerable by a lookup rather than by
re-running the pipeline and hoping it is deterministic; and a threshold can be
re-adjudicated later without re-fetching, because the rejected rows' metrics are on disk.

Drop reasons are a **closed vocabulary**. Adding one is a deliberate edit, which is what
makes the histogram comparable across runs:

```
DECODE_FAILED  TRUNCATED  UNSUPPORTED_MODE  BELOW_RESOLUTION_FLOOR  ASPECT_EXTREME
NEAR_UNIFORM   LOW_ENTROPY  EXACT_DUPLICATE  PERCEPTUAL_DUPLICATE  SEMANTIC_DUPLICATE
PROBE_LEAKAGE  CAP_EXCEEDED  LICENCE_REFUSED  PROVENANCE_INCOMPLETE
```

### 2.2 The filters, with concrete thresholds

Computed in the **same decode pass** as the L1 canonicalisation. Decoding is the expensive
operation; everything here is arithmetic on an array that is already in memory.

| filter | threshold | reason | drop reason |
|---|---|---|---|
| decode integrity | `Image.open()` **and** `.load()` both succeed with `ImageFile.LOAD_TRUNCATED_IMAGES = False` | `open()` alone reads the header; a truncated JPEG only fails on `load()`. Opening without loading is how truncated files reach training as grey rectangles. | `DECODE_FAILED` / `TRUNCATED` |
| colour mode | reject `CMYK`, `P` with transparency, and anything that does not round-trip `convert("RGB")` | `_decode_split` calls `convert("RGB")` unconditionally; a mode that converts badly produces a plausible-looking wrong image rather than an error | `UNSUPPORTED_MODE` |
| resolution floor | **per-source**, `min(H, W) >= 1.5 × target` for photographic sources → 96 for a 64 target | resizing up invents detail | `BELOW_RESOLUTION_FLOOR` |
| aspect ratio | **per-source**, default `0.25 <= W/H <= 4.0` | an 8:1 strip centre-cropped to square is 12% of the original image | `ASPECT_EXTREME` |
| near-uniform | **per-source**, default: grayscale std < 4/255 **or** >90% of pixels in one of 32 histogram bins | a blank scan carries no signal and a batch of them destabilises the JEPA target | `NEAR_UNIFORM` |
| entropy floor | Shannon entropy of the 64-bin grayscale histogram < 3.0 bits | catches gradients, solid fills and single-tone scanner artefacts that pass the std test | `LOW_ENTROPY` |

### 2.3 "Per-source" is not hedging — a global threshold deletes three sources outright

This is the most important sentence in the section. Apply the photographic thresholds
globally and the corpus loses:

- **`fashion-mnist` (28×28) and `quickdraw` (28×28)** — entirely, on the resolution floor.
  Both are natively below it and are upscaled *by design*.
- **`dsprites`** — a large fraction, on near-uniform. dSprites is a small white silhouette
  on black; global pixel std is legitimately low. The filter is right about what it
  measures and wrong about what it concludes.
- **`bl-books` `embellishments`** — potentially a large fraction on aspect ratio. Book
  embellishments are frequently long thin decorative strips. *(How large is unmeasured;
  measure before setting the threshold, do not set it and find out.)*

So the parameter set is a **per-source policy table**, defaulting to the photographic
values and overridden explicitly with a recorded reason. The reason is stored next to the
threshold, in the receipt's `detail`, so a future reader sees *why* dSprites was exempted
from a uniformity test rather than seeing that someone turned it off.

### 2.4 The bias cost, stated per filter

Quality filtering is distribution shaping. Every threshold here removes a *kind* of image,
not a random sample of images.

| filter | what it removes, in population terms |
|---|---|
| **resolution floor** | Older photography, phone snapshots from before ~2010, scanned material, thumbnails-only sources, and any contributor who uploaded a downsized copy. On a CC0 stock platform this correlates with *when* and *by whom* a photo was contributed, so a resolution floor is a soft filter on contributor cohort. |
| **aspect ratio** | Panoramas, architectural elevations, book plates and strip illustrations, film stills in scope ratios. It disproportionately removes *scanned and printed* material relative to *photographed* material. |
| **near-uniform / entropy** | Minimalist composition, high-key and low-key photography, fog, snow, night sky, and — critically — whole synthetic domains whose visual statistics are legitimately sparse. |
| **decode / truncation** | Nearly neutral. This is the only filter here with no obvious population correlate, which is why it is the only one applied globally without a per-source override. |

**No aesthetic filter. This is a recommendation with a reason, not an omission.** An
aesthetic scorer encodes the preferences of whoever produced its training ratings, and the
widely-used ones are CLIP-backboned models trained on ratings of LAION images — which puts
them in the *same unresolved provenance class* as the labelling question in §4, while
delivering a filter whose only defensible description is "images that look like the images
the raters liked". For a corpus whose entire purpose is escaping an inherited-terms
problem, adopting a filter with an inherited-terms problem to improve subjective quality is
a bad trade. Structural and integrity filters only.

Two filters that are *not* recommended and should be named so nobody adds them casually:
a "no text in image" filter (Latin-trained detectors systematically over-fire on non-Latin
scripts, and `bl-books` is largely text-bearing by nature), and NSFW/face filtering (both
require a model with the §4 problem; if the operator decides they are required for a
release, that is a policy decision to be made explicitly, with the model's provenance
recorded in `lineage` like any other derivation).

### 2.5 Detection

The gate is **not** "the drop rate is below X". It is:

| gate | rule |
|---|---|
| `reconciled` | `rows_in == rows_out + len(rejects)`. No unaccounted rows. |
| `drop_report_present` | a per-source × per-reason drop matrix exists in the receipt's `detail`. **A stage that filtered without reporting what it removed fails**, regardless of the numbers. |
| `no_source_collapsed` | no source's survival rate falls below 50% without an explicit `expected_survival` override in its policy record. A source silently losing most of itself is the failure mode that produces a corpus nobody notices is missing a domain. |
| `thresholds_recorded` | every threshold actually applied, per source, is in `detail`. A number that shaped the corpus and is not in the receipt did not happen as far as any reader is concerned. |

`no_source_collapsed` is the one that earns its keep. It is the check that would have
caught "we applied a 96px floor and deleted Fashion-MNIST" on the run that did it, rather
than three weeks later when someone wondered why the garment domain had no effect.

### 2.6 Cost

CPU and IO only, fused into the L1 decode pass. See §8.

---

## 3. Deduplication

**Inputs:** L1 shards with quality columns. **Outputs:** the same rows with
`dup_cluster_id`, `dup_role` (`representative` | `member`), `dup_method` and
`dup_distance`; nothing moved to `rejects/` at this stage.

### 3.1 Dedup marks; it does not remove

Duplicates get a cluster id and a role. The training shard build (§5) selects
`dup_role == "representative"`. Three reasons this is better than deleting:

- A cluster can be re-adjudicated at a different threshold without re-embedding 1M images.
- The cluster size distribution is itself a finding — a source with a heavy tail of
  50-member clusters is telling you something about how it was assembled.
- Choosing *which* member represents a cluster is a real decision (highest resolution?
  earliest retrieved? cleanest provenance?) and it should be a recorded, reproducible rule
  (`max(original_width * original_height)`, ties broken by `origin_sha256`), not an
  artefact of iteration order.

### 3.2 Exact

Two hashes, because they answer different questions.

- `file_sha256` — over the original bytes. Catches a byte-identical re-fetch and is
  already computed: it *is* `origin_sha256`.
- `pixel_sha256` — over the canonical decoded RGB array at L1 resolution. Catches the same
  image stored twice with different EXIF, different JPEG encoder, or a container change.

Cost: free, in the decode pass. Expected yield: small within a source, non-trivial across
`pxhere` and `bl-books` if either re-hosts the other's material (**unmeasured**).

### 3.3 Perceptual

64-bit DCT pHash over a 32×32 grayscale reduction.

**Normalisation before hashing matters more than the hash.** Hash on: grayscale →
aspect-preserving centre-crop to square → resize to 32×32 → DCT → median threshold. The
crop-to-square step is not decoration: it normalises letterboxing and padding, which are
the two systematic transforms most likely to be present and which otherwise shift the
entire DCT layout and defeat the hash completely.

| parameter | value | note |
|---|---|---|
| duplicate | Hamming ≤ 6 | conservative; near-certain same image |
| candidate | 7 ≤ Hamming ≤ 10 | recorded, not actioned, and reported as a histogram |
| index | 4 bands of 16 bits, exact-match candidate generation | brute-force is 5×10¹¹ pairs at 1M and is not viable; banded lookup is near-linear |

**Where pHash is blind, and why that is per-source again.** It survives resize, JPEG
requantisation, small crops and mild colour shifts. It does **not** survive horizontal
flip, rotation, or heavy crop. And it produces *false* positives on low-detail imagery
where many genuinely distinct images share a hash — which is exactly `quickdraw`,
`dsprites` and, to a lesser extent, `fashion-mnist`. **Do not run pHash dedup within the
sketch and synthetic sources.** Run it within the photographic ones, and — this is the part
that matters — run it **across** every pair of sources including the probe sets.

### 3.4 Semantic, and the encoder circularity that this one can actually escape

Embedding near-duplicates catch what pHash cannot: the same scene from a slightly different
frame, the same object re-shot, a flip, a crop past 30%.

This needs an encoder, and an encoder has the §4 provenance question — **except that here
there is a clean way out, and it costs one extra training run.**

> **Bootstrap.** Run exact + perceptual dedup first. Pretrain a first-pass I-JEPA on the
> resulting corpus — which is licence-clean by construction, since it is this corpus. Use
> *that* encoder's frozen features for semantic dedup. Rebuild the corpus. Retrain.

The encoder used to filter the corpus is trained only on the corpus, so no external
training-data terms enter the chain at any point. Cost: one additional `vl_latent` run —
**497 seconds on record** for the current configuration. Against a pipeline whose dominant
cost is a multi-hour download, this is free.

Two honest caveats. (a) A first-pass encoder is worse at this than a strong pretrained one,
so recall will be lower — semantic dedup will miss pairs a CLIP embedding would catch.
(b) There is a mild self-reference: the encoder's notion of similarity is shaped by the
duplicates still in its own training data. Neither is fatal for a *filtering* decision, and
both are preferable to importing an unresolved provenance question into the corpus's
foundation. If the operator later decides an external encoder is acceptable for filtering —
a defensible position, since the embeddings are consumed and discarded rather than shipped
— that is a decision to record in `lineage`, not a default.

| parameter | value |
|---|---|
| within-source duplicate | cosine ≥ 0.95 on L2-normalised features |
| cross-source duplicate | cosine ≥ 0.98 (stricter: a cross-domain "duplicate" is more likely a false positive) |
| search | **exact brute force**, chunked on GPU. At 1M × 384 dims this is ≈7.7×10¹⁴ FLOP; a free 1080 Ti at fp32 does it in tens of minutes, and the 1M×384 fp32 matrix is 1.5 GB — comfortable in 11 GB. **Exact search removes the ANN-recall unknown entirely at this scale**, so do not reach for an index. |

*(That estimate is arithmetic from the recorded 497 s / 4,000 steps / batch 128 run scaled
for a forward-only pass and for a 1080 Ti's fp32 throughput. It is an extrapolation, not a
measurement. Note also the 1080 Ti's declared `never: [BF16, FP8, …]` in
`edge-backends.json` — this is an fp32 job by constraint, not by choice.)*

### 3.5 The lesson from the text side, translated

This project has already measured why hashing alone is insufficient: **17.8% overlap
between two datasets that hashing found 2 instances of, because one prefixed its text.** A
single systematic transform applied to one side made an exact-match check report
approximately zero.

The visual analogue is not "a resized duplicate" — it is *a source that systematically
transformed everything it holds*: re-encoded at a fixed JPEG quality, padded to square,
letterboxed, watermarked with a band, converted to grayscale, or rescaled to a fixed
maximum edge. Any one of those makes byte-hash overlap read as zero across two sources that
substantially overlap.

**The generalisable check, and it is cheap:** run the overlap measurement at **two
normalisation levels** and report both numbers.

```
raw_overlap        = |A ∩ B| under file_sha256
normalised_overlap = |A ∩ B| under pHash(grayscale, centre-crop-square, 32x32) ≤ 6
```

A large gap between them **is the detection**. If normalised overlap is 17% and raw overlap
is 0.0002%, a systematic transform is present and you have just found it — the same shape
of finding as the text-prefix case, surfaced by the same method rather than by luck. Both
numbers go in the receipt's `detail` for every source pair, always, including the pairs
where they agree; the diagnostic only works if the baseline case is also on record.

### 3.6 The direction that actually matters: probe leakage

Intra-corpus duplication costs some sample efficiency. **Train↔probe duplication
invalidates the only numbers this region produces.**

`quickdraw` and `caltech256` are the transfer probes; `eurosat`'s test split is the
in-domain probe. Every one of them must be checked against the entire pretraining pool at
all three dedup levels. This mirrors the text side's `assert_no_contamination(...,
tolerance=0.0)`, whose docstring already states the discipline:

> *Default tolerance is zero. A non-zero tolerance should be a deliberate, argued choice
> for a specific corpus — not a way to make a failing check pass.*

| gate | rule |
|---|---|
| `no_probe_leakage_exact` | 0 exact matches between any probe set and the pretraining pool |
| `no_probe_leakage_phash` | 0 pHash matches at Hamming ≤ 6 |
| `no_probe_leakage_semantic` | ≤ tolerance, tolerance defaulting to 0 and any non-zero value carrying an argued `why` in the receipt |
| `dedup_two_level_reported` | both raw and normalised overlap present for every source pair |

There is a plausible-sounding failure worth naming: `caltech256` is everyday-object
photography and `pxhere` is general stock photography. They are different collections, but
if a contributor uploaded a Caltech image to pxhere — or, far likelier, if both contain the
same widely-redistributed public-domain photograph — the leak is real and pHash will find
it. This is precisely why cross-source dedup runs against probes and not only within the
training pool.

---

## 4. Labelling and enrichment

### 4.1 The circularity problem, stated

Classifying images requires a vision model. If that model was ImageNet-trained, its outputs
are derived from the corpus this entire effort exists to escape. **Whether a model's
OUTPUTS inherit its training data's terms is the same unsettled question as
weights-as-derivative**, already laid out at length in `LICENCE-FOR-OPEN-WEIGHTS.md` §*The
share-alike question, stated without resolving it* — including the one piece of directly
relevant observed practice, `timm`'s own README:

> *"Any models I have trained with ImageNet are done for research purposes and one should
> assume that the original dataset license applies to the weights."*

If the dataset licence reaches the weights, the question of whether it reaches the weights'
outputs is at minimum open. **This document does not resolve it and must not be read as
having resolved it. It is flagged as needing a human decision.** What follows is the option
space and what each option costs.

### 4.2 The reframe that removes most of the problem

Before choosing an option: **this corpus needs ~50k labels, not 600k.** I-JEPA pretraining
is unlabelled (§0). Labels are consumed by `_linear_probe` and the transfer probe, nothing
else. And every label the probes need already ships with a source that provides it.

| source | labels it ships | count | role |
|---|---|---|---|
| `quickdraw` | 345 sketch categories | 4.5M available | **primary transfer probe** |
| `caltech256` | 257 object categories | 30,607 | **secondary transfer probe** |
| `eurosat` | 10 land-use classes | 27,000 (test split held out) | **in-domain probe** |
| `pcam` | binary tumour / no-tumour | 327,680 | available |
| `fashion-mnist` | 10 garment classes | 70,000 | available |
| `shapes3d` | 6 generative factors (shape, hue×3, scale, orientation) | 480,000 | available; factor-regression probes |
| `dsprites` | 5 generative factors | 737,280 | available |
| `clevr` | full scene graph (objects, shapes, colours, materials, sizes, relations) | 100,000 | available; the richest structured labels in the set |
| `pxhere` | free-text tags *(unverified — see §1.4)* | ≈1.1M | weak labels at best |
| `bl-books` | none per image; config-level only | 1,080,814 | none |

**So the required path needs no model at all.** The circularity problem reappears only for
*enrichment beyond what the probes need* — captions for a future VL objective, semantic
tags on pxhere, class structure on `bl-books`. Those are genuinely optional today. Saying
so is the single most useful thing this section can do, because it converts a blocking
legal question into a deferrable one.

### 4.3 The options, and what each costs

| option | what it is | cost | verdict |
|---|---|---|---|
| **A. Permissively-trained labeller** | Find a vision model whose own training corpus is licence-clean, and label with it | The audit that establishes "licence-clean training corpus" for a candidate labeller is the same audit that took this project a 1,500-line document *per corpus*. And a permissive weights licence is not evidence about training data — that is the exact error pattern (`Nan-Do/code-search-net-python`, `ylecun/mnist`, `bazyl/GTSRB`) this project has now recorded ten times. High effort, uncertain payoff, and it does not remove the legal question, it relocates it. | Not recommended as a default |
| **B. Human / heuristic labelling** | People, or rules | Human labelling is out of the question at 600k (at 1.5 s/image, ~250 person-days). It is entirely feasible at probe scale — 10k images, few classes, ≈4–6 person-days — **if a probe were needed that no source provides**, which today none is. Heuristic labelling is free and useful, but only for *structural* attributes (§4.4), never semantic ones. | Recommended for structural attributes; held in reserve for probes |
| **C. Source-provided labels only** | Use only what the source shipped | Covers 100% of the probe requirement (§4.2). Costs the two largest sources — `pxhere` and `bl-books`, together 2.18M images — any label structure at all. Since they are pretraining-only, that costs nothing today. It costs something the moment someone wants a captioning or CLIP-style objective. | **Recommended default** |
| **D. Accept the risk explicitly** | Label with whatever model works, record what did it | Cheapest to execute, and honest if done properly. Requires: labels quarantined in a separate column namespace, `label_source: "model:<repo>@<revision>"`, `label_licence_status: UNRESOLVED`, a separate manifest whose `licence.verdict` reflects the unresolved state, and an explicit opt-in from any run that consumes them. **Never mixed into the same column as source-provided labels.** | Available, gated on a human decision, never a default |

### 4.4 Enrichment that needs no model and is safe

All computed in the L1 decode pass, all CPU, all with a `label_source` of
`"heuristic:<name>@<version>"`:

- **`domain_id`** — known for free at assembly time from `source_id`. This is the label for
  the domain-identity probe the licence audit proposes, and it costs nothing.
- Colour statistics: per-channel mean/std/percentiles, saturation, dominant hue.
- **Blur estimate:** variance of the Laplacian.
- **Edge density:** Sobel magnitude mean — separates photography from line art from
  synthetic renders without a model.
- **Synthetic-vs-photographic heuristic:** JPEG quantisation-table fingerprint and
  high-frequency noise floor. Not reliable per image; informative in aggregate per source.
- Source-shipped structured metadata copied verbatim: CLEVR's scene graph, Shapes3D and
  dSprites factor vectors, `pxhere` tags if present.

**Two enrichments to handle carefully.** EXIF is useful (capture date, focal length,
orientation) and also contains GPS coordinates and camera serial numbers. **Strip GPS and
serials at ingest, record in `lineage` that they were stripped**, and keep only fields that
do not identify a person or a place. Second: `pxhere` tags are *source-provided CC0
metadata*, so tag-derived weak labels are option **C**, not option A — a genuinely useful
path to giving the largest source some structure with no model in the loop. It hinges on
the unverified question in §1.4.

### 4.5 Detection

| gate | rule |
|---|---|
| `labels_have_provenance` | every non-null label column has a non-null `label_source`. A label with no recorded origin is a validation error, not a warning. |
| `no_namespace_mixing` | no column contains both source-provided and model-derived values. Enforced by construction: they are different columns. |
| `label_licence_resolved` | no label column whose `label_licence_status` is `UNRESOLVED` appears in a shard set whose manifest verdict is `TRAIN_OK`, unless the run explicitly opted in. |
| `probe_labels_complete` | every probe shard has a label for every row; a probe with 3% missing labels silently reports a metric on 97% of its eval set. |

---

## 5. Balance and composition

### 5.1 The 77% figure, reconciled

The brief's figure is exactly right, for one specific union — and being precise about which
one is worth a paragraph, because the two unions have very different shapes.

**Union of the sources that ship labels** (excluding the held-out probes):

| source | images | share |
|---|---|---|
| `pcam` | 327,680 | **77.2%** |
| `fashion-mnist` | 70,000 | 16.5% |
| `eurosat` | 27,000 | 6.4% |
| total | 424,680 | |

**Union of all clean pixel-bearing sources** (excluding probes):

| source | images | share |
|---|---|---|
| `pxhere` | 1,100,000 | 28.0% |
| `bl-books` | 1,080,814 | 27.6% |
| `dsprites` | 737,280 | 18.8% |
| `shapes3d` | 480,000 | 12.2% |
| `pcam` | 327,680 | 8.4% |
| `clevr` | 100,000 | 2.5% |
| `fashion-mnist` | 70,000 | 1.8% |
| `eurosat` | 27,000 | **0.7%** |
| total | 3,922,774 | |

Both are unusable as-is, for opposite reasons: the first is a histopathology corpus with
garnish, the second buries EuroSAT under 1% and Fashion-MNIST under 2% of every batch. The
licence audit already named the outcome — *"an encoder that is functionally a
pxhere-and-histopathology encoder that happened to see a little satellite and garment
imagery"* — and, critically, that is **not a risk, it is the arithmetic of
`torch.randint` over a concatenated pool.**

### 5.2 The target distribution

An extension of the composite recommended in `LICENCE-FOR-OPEN-WEIGHTS.md`, adding the two
clean sources that composite omitted, with the reason for each.

| domain | source | cap | of available | share | why this cap |
|---|---|---|---|---|---|
| general photography | `pxhere` | 150,000 | 1,100,000 | 24.9% | The scarcest and most valuable property in the whole set (real lighting, pose, texture). Largest single share, deliberately. |
| book illustration | `bl-books` | 100,000 | 1,080,814 | 16.6% | **Addition.** Engravings and plates are a genuinely different texture and edge statistic from photography, and it is one of only two million-scale clean sources. Not in the audit's composite; the reason to add it is domain breadth, the reason to cap it hard is that it is not photography. |
| histopathology | `pcam` | 100,000 | 327,680 | 16.6% | Capped from 327,680 precisely to defuse §5.1. |
| synthetic 3D, one object | `shapes3d` | 80,000 | 480,000 | 13.3% | Controlled factors; native 64×64 so nothing is lost at resize. |
| synthetic 3D, multi-object | `clevr` | 80,000 | 100,000 | 13.3% | Occlusion, shadow and reflection — the only source with multi-object scene structure. |
| garment product photo | `fashion-mnist` | 70,000 | 70,000 | 11.6% | All of it; it is small and upscaled from 28×28. |
| satellite land use | `eurosat` | 21,600 | 27,000 (16.2k/5.4k/5.4k) | 3.6% | Train + validation only. The 5,400-image test split is the in-domain probe and must not be trained on. |
| synthetic silhouettes | `dsprites` | **0** | 737,280 | 0% | **Held at zero in the default mix.** 1-channel white-on-black silhouettes at very low visual complexity; 737k of them would be 19% of a naive union while teaching close to nothing about natural image statistics. Available for the §5.3 balance experiment; not in the default corpus. |
| **total** | | **601,600** | | | |

Held out entirely from pretraining, enforced by gate: `quickdraw`, `caltech256`,
`eurosat`'s test split.

**The strict-balance variant**, which the licence audit also asks for and which should be
run as a comparison rather than argued about: cap every included source at EuroSAT's
ceiling, ~21,600 → 7 sources × 21,600 = **151,200 images**, close to tiny-imagenet's
original 100,000 scale and perfectly equal by domain. Both variants are the same shard
build with a different cap table; producing both costs one extra materialisation pass.

### 5.3 Balance is not enforced in code, and here is the exact reason

Restated because it determines the mechanism rather than the policy:

```python
idx = torch.randint(0, x_tr.size(0), (cfg.batch_size,), generator=gen)
```

Uniform over a pool that `_decode_split` built by concatenating every shard in
`cfg.train_shards` with no weighting. There is no domain field in the tensor, no sampler to
configure, and no place to express a cap. **Therefore:**

> The only balance mechanism that works against today's trainer is **materialising
> pre-balanced shards**. Cap enforcement is a property of the files, not of a runtime
> parameter.

Which is fine, and in one respect better: a materialised balanced shard set is a hashable
artifact with a receipt, and the composition is auditable after the fact. It costs an extra
copy — 601,600 × ~4 KB ≈ 2.4 GB — and one pass.

A frequency-weighted or temperature-scaled sampler would be more precise (it does not
throw away the 950,000 uncapped pxhere images; it just sees them less often). That is a
small addition to `_decode_split` or the training loop, and it is **out of scope here** —
this pipeline's obligation is to emit shards whose composition is correct and recorded, and
to emit the per-row `source_id` such a sampler would need if someone writes one.

### 5.4 Normalisation statistics — what this pipeline owes, and what it cannot fix

`_to_float` applies ImageNet's mean/std to every image regardless of source. Three ways
out, and the pipeline's obligation is the same under all three.

| option | what it is | cost |
|---|---|---|
| **(a) leave it** | ImageNet stats on histopathology stain and Sentinel composites | The domains most different from ImageNet get the worst-conditioned inputs. It is what happens today. |
| **(b) harmonise at ingest** | Affine per-channel remap of each source's L2 pixels so the *pool* actually has ImageNet's statistics, then the hardcoded constant is correct | **Zero code change in the trainer.** Costs: clipping at the tails, and it deliberately destroys genuine cross-source colour differences — which for stain and satellite composites is arguably the signal. A stopgap, and it should be labelled one in `lineage`. |
| **(c) carry `source_id` into the tensor and index a stats table** | The correct fix | A small change to `_decode_split` and `_to_float`. Out of scope for this document; in scope for the PR that consumes this corpus. |

**Under all three, the pipeline computes and publishes per-source per-channel mean/std at
L2 resolution**, into `SHARD.json` and a corpus-level `NORMALISATION.json`. It is one pass
over uint8 arrays that are already being written — effectively free — and it is the
ingredient (c) needs and (b) is computed from. Recommendation: **(c)**, with **(b)**
available as a zero-code-change stopgap, and **(a)** named as the status quo so nobody
mistakes it for a decision.

### 5.5 Detection

| gate | rule |
|---|---|
| `caps_respected` | per-source row counts in the built shard set exactly match the cap table. Off-by-a-shard is the classic silent failure. |
| `composition_recorded` | the realised per-source distribution is in the receipt's `detail`, alongside the intended one. |
| `holdout_disjoint` | `quickdraw`, `caltech256` and `eurosat:test` contribute zero rows to any training shard, checked by `source_id` **and** by `origin_sha256` set intersection — the second catches the case where a probe image also exists in a training source. |
| `normalisation_published` | per-source stats exist for every source in the built set. |

---

## 6. Attribution at scale

### 6.1 The attributable unit is the source, and that is a property of source selection

At 1M+ images, "attribution is required" sounds like a 1M-line problem. It is not — but for
a specific reason that must be recorded, because it stops being true the moment someone
adds the wrong source.

Every CC-BY source in this set is a **single-attribution collection**: Quick Draw → Google,
Inc.; CLEVR → Facebook, Inc.; Caltech-101/256 → Caltech; EuroSAT → the phelber compilation
plus the Copernicus notice. None is a per-contributor aggregation. And that is not luck —
it is exactly why the licence audit ruled out COCO, Open Images, RedCaps, Food-101 and
TreeOfLife-200M, every one of them on the mixed-per-contributor pattern. Google's own words
on Open Images:

> *"we make no representations or warranties regarding the license status of each image and
> you should verify the license for each image yourself."*

**So attribution here is ~10 records, not 1M — because of which sources were chosen.** If a
per-contributor source is ever added, `attribution_granularity: "item"` makes the per-image
obligation explicit and the generator emits per-item notices. The schema supports the hard
case; the corpus is built so the hard case does not arise.

### 6.2 `ATTRIBUTION.jsonl`

One JSON object per line, per attributable unit. This is the machine-readable file; every
human-readable artifact is generated from it and none is hand-edited.

```json
{
  "attribution_ref": "quickdraw",
  "source_id": "quickdraw",
  "title": "The Quick, Draw! Dataset",
  "creator": "Google, Inc.",
  "creator_url": "https://github.com/googlecreativelab/quickdraw-dataset",
  "source_url": "https://huggingface.co/datasets/google/quickdraw",
  "source_revision": "<commit sha>",
  "retrieved_utc": "2026-09-__T__:__:__Z",
  "attribution_granularity": "source",
  "obligations": [
    {
      "licence_id": "CC-BY-4.0",
      "licence_url": "https://creativecommons.org/licenses/by/4.0/",
      "attribution_required": true,
      "notice_text": "The Quick, Draw! Dataset by Google, Inc., used under CC BY 4.0.",
      "modification_disclosure_required": true
    }
  ],
  "modifications": [
    "grayscale bitmap upscaled to 64x64 (bicubic)",
    "converted to 3-channel RGB",
    "subset selected by uniform sampling; see composition in the ingest receipt"
  ],
  "images_used": 0,
  "images_available": 50426266,
  "role": "probe-only",
  "licence_verified_utc": "2026-09-02",
  "licence_evidence": "docs/design/LICENCE-FOR-OPEN-WEIGHTS.md#replacement-vision-corpora"
}
```

Four fields that are easy to omit and are the ones that make the file compliant rather than
decorative:

- **`obligations` is a list, not a scalar.** EuroSAT carries two simultaneously: MIT over
  phelber's compiled 64×64 patches, *and* the Copernicus notice over the underlying
  Sentinel-2 imagery — `"Contains modified Copernicus Sentinel data [Year]"`, required by
  the EU legal notice quoted in the licence audit. A one-licence-per-source schema cannot
  express that, and would silently drop half of it.
- **`modifications` and `modification_disclosure_required`.** CC BY 4.0 §3(a)(1)(B)
  requires indicating whether the material was modified. **Every image in this corpus is
  modified** — resized, cropped to square, colour-converted, possibly harmonised. A notice
  that names the creator and omits the modification disclosure is incomplete, and this is
  the part most dataset cards get wrong.
- **`images_used` vs `images_available`.** `0` used with `role: "probe-only"` is a
  meaningful, checkable state.
- **`licence_evidence`.** Points at where the verdict was established. A verdict with no
  trail is, per `MODEL-MANIFESTS.md`, *"an opinion of unknown age"*.

### 6.3 What the dataset card must carry

Generated from `ATTRIBUTION.jsonl`; regenerating is idempotent; hand-edits are overwritten.

1. **HF frontmatter** — `license:` as a list of every distinct `licence_id` present
   (`cc0-1.0`, `apache-2.0`, `mit`, `cc-by-4.0`); `source_datasets:` naming each upstream;
   `annotations_creators:` distinguishing `found` (source-provided) from `machine-generated`
   if §4 option D was ever taken.
2. **A Licensing Information section** — one subsection per source with the verbatim
   `licence_observed`, the upstream statement, the obligations, and the modifications.
3. **A composition table** — the realised per-source counts from §5.5's
   `composition_recorded`. This is what lets a reader check the 77% problem was actually
   solved rather than described.
4. **A `NOTICE` file**, the concatenated `notice_text` of every obligation with
   `attribution_required: true`, plus the Copernicus line.
5. **`SOURCES.json` and `ATTRIBUTION.jsonl` shipped alongside**, so downstream consumers
   inherit the machine-readable form rather than having to parse prose back out of a card.

### 6.4 What the model card must carry

The weights are where CC BY attribution lands if weights are adapted material — the
question `LICENCE-FOR-OPEN-WEIGHTS.md` deliberately leaves open. Both branches need the
same block, so it is written unconditionally:

1. The generated attribution block for every CC-BY source in the training composition.
2. The Copernicus notice.
3. A composition table naming every source and its realised share.
4. **The project's stated position on weights-as-derivative, in words** — because the audit
   already established the alternative is worse: *"What is NOT a mitigation: asserting that
   weights are not derivative works because it would be convenient. If that position is
   taken, take it explicitly, in the model card, as a stated position rather than a
   silence."* A silence is a position too, and an undefendable one.
5. The corpus id and its ingest receipt hash, so the card's claims are checkable against an
   artifact rather than trusted.

### 6.5 Detection

| gate | rule |
|---|---|
| `attribution_resolvable` | every distinct `attribution_ref` in the built shards resolves to a line in `ATTRIBUTION.jsonl`. |
| `attribution_complete` | every source with `attribution_required: true` has non-empty `notice_text` and non-empty `modifications`. |
| `no_orphan_attribution` | every line in `ATTRIBUTION.jsonl` is referenced by at least one shard row, **or** carries an explicit `role` explaining why it is not (`probe-only`, `evaluated-not-used`). Catches a source removed from the mix while its notice was left behind — the failure that produces a card claiming to contain data it does not. |
| `card_regenerates_clean` | regenerating the card produces no diff. A hand-edited card is a card that will drift. |

---

## 7. The receipt

### 7.1 The envelope

`src/cogsyndelta/pipeline/receipt.py`'s `model-pipeline-receipt/v1`, unchanged. `STAGES` is
*"open by convention rather than enforced, because a new architecture may have a stage
nobody anticipated"* — this pipeline adds **`ingest`**, and each of the seven stages emits
one receipt with `stage: "ingest"` and a distinct `producer.component`.

```python
Receipt(
    producer=Producer(
        project="cogsyndelta",
        component="vl-clean-v1/dedup",     # corpus id / pipeline stage
        architecture="visual-corpus",      # free text, for grouping
    ),
    stage="ingest",
    metrics={...},      # flat name -> number. Plottable without interpretation.
    baseline={...},     # the previous run of THIS stage, for drift
    gates={...},        # the stage's own verdict. Empty gates is NOT a pass.
    artifacts={...},
    provenance={...},
    detail={...},       # per-source tables, histograms, thresholds
)
```

The split the envelope's docstring insists on maps cleanly: aggregate counts and rates are
`metrics`; every per-source table, drop-reason matrix, pHash histogram and threshold set is
`detail`, which *"the reader passes through untouched"*.

And `Receipt.passed` already encodes the right discipline for a pipeline that could
otherwise look healthy while doing nothing:

> *"A run with no gates is NOT a pass. Something that measured nothing has not demonstrated
> anything, and showing it as green is how an empty pipeline looks healthy on a
> dashboard."*

### 7.2 Per-stage receipt contents

| stage | `component` | key `metrics` | `gates` |
|---|---|---|---|
| 1 acquire | `…/acquire` | `rows_declared`, `rows_written`, `rows_rejected`, `bytes_l1`, `seconds`, `images_per_second` | `reconciled`, `provenance_complete`, `licence_all_train_ok`, `revisions_pinned` |
| 2 quality | `…/quality` | `rows_in`, `rows_out`, `drop_rate`, per-reason counts | `reconciled`, `drop_report_present`, `no_source_collapsed`, `thresholds_recorded` |
| 3 dedup | `…/dedup` | `dup_exact`, `dup_phash`, `dup_semantic`, `clusters`, `largest_cluster` | `reconciled`, `no_probe_leakage_{exact,phash,semantic}`, `dedup_two_level_reported` |
| 4 label | `…/label` | `rows_labelled`, `label_coverage`, `distinct_label_sources` | `labels_have_provenance`, `no_namespace_mixing`, `label_licence_resolved`, `probe_labels_complete` |
| 5 balance | `…/balance` | `rows_out`, per-source counts, `max_source_share` | `caps_respected`, `composition_recorded`, `holdout_disjoint`, `normalisation_published` |
| 6 attribution | `…/attribution` | `sources`, `obligations`, `attributed_images` | `attribution_resolvable`, `attribution_complete`, `no_orphan_attribution`, `card_regenerates_clean` |
| 7 publish shards | `…/shards` | `shards`, `rows`, `bytes`, `mean_rows_per_shard` | `orphan_audit_passed`, `no_null_image_cells`, `index_complete`, `shard_hashes_recorded` |

`provenance` on every receipt carries: `corpus_id`, `manifest` (the manifest id, which
`MODEL-MANIFESTS.md` specifies as the join key — *"Fleet-unique. Joins to receipt
provenance.manifest"*), `tool_version`, `params_sha256`, `input_shard_sha256[]`, and
`input_receipts[]` — the paths of the previous stage's receipts. **That last field is what
makes the chain traversable**: from a trained model's receipt, to the shard build, back
through balance, labelling, dedup, quality, and acquisition, to a source revision and a
retrieval date.

### 7.3 The orphan-pixel audit

The one check that is a sample rather than a scan, and the one that catches what the
structural gates cannot: a bug that produces *internally consistent but wrong* provenance.

Sample K = 1,000 rows uniformly at random from the final L2 shards. For each:

1. `origin_sha256` is present, well-formed, and appears exactly once in `index.parquet`.
2. `source_id` resolves in `SOURCES.json`; `licence_id` resolves; the verdict is `TRAIN_OK`.
3. `attribution_ref` resolves in `ATTRIBUTION.jsonl`.
4. The `lineage` chain is contiguous — every stage between acquisition and this shard has
   an entry, in order, each naming a receipt that exists on disk.
5. `source_revision` is a pinned commit, not a tag.
6. Where `provenance_granularity == "item"`, `source_item_url` is non-null and well-formed.

**Any single failure fails the corpus.** At 1,000 samples, a defect affecting ≥0.3% of rows
is caught with ~95% probability; a defect affecting one row in a million is not, and this
audit does not claim otherwise — the structural gates in §1.6 are what cover that case, and
the sample is what catches the class of bug where the structural gates themselves are
wrong. Re-run on every L2 rebuild, not once.

### 7.4 Idempotency

Every stage is keyed by

```
params_sha256 = blake2b( sorted(input_shard_sha256) + tool_version + canonical_json(params) )
```

and writes to a directory named by it. A re-run with the same key finds a completed
`SHARD.json` and skips — the same contract `csd-corpus-expand.py` already states
(*"A dataset already on disk with a manifest recording the same revision is skipped.
Re-runs are cheap and safe, which is the point: this is meant to be run by an unattended
agent."*) — and, critically, the same crash-safety trick: **an output directory without a
complete `SHARD.json` reads as absent and is re-derived**, so a crash mid-write never
poisons a later run. Changing a threshold changes `params_sha256`, which produces a new
directory rather than silently reusing the old pixels — the same reasoning behind
`_decode_split`'s cache key, executed with a stable hash.

---

## 8. What this costs: throughput, storage, and which stages actually want a GPU

**Estimates, not measurements.** Nothing here has been run. Figures marked *(extrapolated)*
are arithmetic from numbers already on record for this fleet; figures marked *(unverified)*
are assumptions that should be checked before they are relied on.

### 8.1 Throughput

| stage | bound | hardware | estimate at ~1M images |
|---|---|---|---|
| **1. fetch** | **network / IO** | any host with `/bulk` | **the dominant cost of the entire pipeline.** pxhere at an assumed 2–4 MB/image *(unverified)* is 2.2–4.4 TB; at a sustained 200 Mbit/s that is **25–50 hours**. BL is similar. Everything else together is a few tens of GB and finishes in under an hour. |
| **1b. decode + canonicalise to L1** | **CPU** | many cores | With `Image.draft()` scaled DCT decode (decode directly at 1/8, the single largest CPU win here — roughly 5–8× over a full decode), an 8 MP JPEG costs ~10–15 ms/core → ~70–100 img/s/core → **~1,200–1,600 img/s on 16 cores → ~11–14 min per million** *(extrapolated)*. Without `draft()`, 5–8× that. |
| **2. quality filters** | CPU | — | **free** — fused into 1b, arithmetic on an already-decoded array |
| **3a. exact hashes** | CPU | — | **free** — fused into 1b |
| **3b. pHash + banded index** | CPU | — | ~0.5–1 ms/image on the already-computed 32×32 reduction, plus dict lookups. **Single-digit minutes per million.** |
| **3c. semantic embed** | **GPU** | **1080 Ti (currently free)** | forward-only through the 22.9M ViT at 64×64. From the recorded 4,000 steps × batch 128 in 497 s on gpu5080 (fwd+bwd+EMA), forward-only is ~4–6× cheaper and a 1080 Ti is ~5× slower at fp32 → **~800–1,200 img/s → 15–20 min per million** *(extrapolated)*. |
| **3d. semantic search** | **GPU** | 1080 Ti | exact brute force, 1M×1M×384 ≈ 7.7×10¹⁴ FLOP; at 11.3 TFLOPS fp32 and ~40% achieved → **~10–20 min**. 1.5 GB resident. |
| **4. labelling** | CPU | — | a column copy under the recommended option C. **Zero GPU.** |
| **5. balance / materialise L2** | **IO** | — | a copy: 601,600 × ~4 KB ≈ **2.4 GB written**, minutes on the array |
| **6. attribution** | — | — | ~10 records. Instant. |
| **7. receipts + orphan audit** | IO | — | 1,000 random row lookups. Seconds. |

**The honest summary:** exactly one stage genuinely needs a GPU — semantic dedup — and it
is **well under an hour on the free 1080 Ti**, fp32 by that card's own declared constraint
(`never: [BF16, FP8, …]`). Everything else is CPU or IO. And the largest cost in the whole
pipeline is not compute at all: **it is the initial download**, which is why §1.5's
"fetch once, land at 256, never re-fetch" is a throughput decision as much as a storage
one.

Two things that would change this picture and should be checked rather than assumed: the
actual on-disk size of `nyuuzyou/pxhere` (drives fetch time by an order of magnitude), and
whether the HF mirrors ship already-downsampled images (which would make 1b nearly free).

### 8.2 Storage

| tier | contents | bytes |
|---|---|---|
| L0 retained | small sources only (PCam, EuroSAT, Fashion-MNIST, CLEVR, Shapes3D) | ≈ 30–50 GB |
| L0 **not** retained | pxhere, BL originals | **0** — decoded in stream (would be ≈ 3.3 TB) |
| **L1** | ≈3.9M images at 256 / native-small, JPEG q90 ~25 KB | **≈ 80 GB** |
| L2 training shards | 601,600 × ~4 KB PNG-in-parquet | ≈ 2.4 GB |
| L2 decode cache (`.npy` uint8) | 601,600 × 12,288 B | **≈ 7.4 GB** |
| provenance columns | ≈3.9M × ~600 B raw, dictionary-encoded | ≈ 200–400 MB |
| rejects | ≈1M × ~250 B | ≈ 250 MB |
| `index.parquet` | 601,600 × ~50 B | ≈ 30 MB |
| **total** | | **≈ 140 GB** |

Against ~2.5 TB of headroom under the fleet's 50% ceiling. Comfortable — and it is
comfortable *because* L0 is not retained for the two million-scale sources.

**One RAM figure that constrains the design and is easy to miss.** `_decode_split` returns
`torch.from_numpy(np.load(xf))` — no `mmap_mode` — so the entire training tensor is
resident. At 64×64×3 uint8 that is 12,288 B/image:

| corpus size | resident RAM |
|---|---|
| 151,200 (strict-balance variant) | 1.9 GB |
| 601,600 (recommended composite) | **7.4 GB** |
| 3,922,774 (naive full union) | **48 GB** |

The recommended composite fits. **The naive full union does not fit on most of this
fleet**, which is an independent argument for capping that has nothing to do with balance
and would bite even if the distribution were perfect.

---

## 9. What this design does not fix

Named explicitly so none of it is mistaken for handled.

1. **The trainer still applies ImageNet normalisation to everything.** §5.4 gives the
   pipeline's obligation (publish per-source stats) and three options; option (c), the
   correct one, is a code change outside this document.
2. **Uniform sampling over a concatenated pool.** §5.3's materialised shards are a
   first-order fix, not a sampler.
3. **`_decode_split`'s unstable cache key** (`abs(hash(...))`, salted per process). Routed
   around by materialising L2 at ingest; not repaired.
4. **Capability loss versus tiny-imagenet.** Already stated in the licence audit and not
   softened here: there is no permissively-licensed replacement for balanced, dense,
   natural-object-category photography at 100k+ scale, and *"a `vl_latent` pretrained on
   this composite should be expected to be measurably weaker at fine-grained natural-object
   recognition."* No amount of ingest engineering changes that; this pipeline's job is to
   make the trade honest and measurable, not to eliminate it.
5. **Whether mixing these domains in one I-JEPA run is sound at 22.9M parameters.** The
   audit proposes an experiment (naive union / balanced union / staged) and declines to
   guess. This pipeline's contribution is that all three variants are the same shard build
   with a different cap table, plus the free `domain_id` label the domain-identity probe
   needs.

---

## 10. Open questions that need a human

Marked as such rather than resolved, matching the licence audit's discipline. **Neither the
author of this document nor its reader is a lawyer.**

1. **Do a vision model's OUTPUTS inherit its training data's terms?** (§4.1.) The same
   unsettled class as weights-as-derivative. This decides whether option D in §4.3 is ever
   available. **Not resolvable here, and deliberately not resolved.**
2. **Does source-level attribution plus a modification disclosure satisfy CC BY at 1M
   images?** §6.1 argues yes *because of which sources were chosen*. That argument is sound
   but it is a judgement, and it stops being sound the moment a per-contributor source is
   added.
3. **Does `nyuuzyou/pxhere` publish per-item URLs, contributors and tags?** *(Unverified.)*
   Decides whether the largest source has item-level or source-level provenance, whether
   takedown requests are actionable against it, and whether the weak-label path in §4.4
   exists at all. **Check this first — it changes three stages.**
4. **What is the actual on-disk size of pxhere and BL?** *(Unverified.)* Drives the single
   largest cost in the pipeline by an order of magnitude.
5. **Is L0 discarded for the two large sources?** §1.5 recommends yes, on the fleet's own
   storage ceiling. It is a one-way door: re-deriving above 256 later means re-fetching.
6. **Redistributable or fetch-only?** Not this document's call — **defer to
   `TRAINING-SUPERSET.md`.** But note the coupling: if the corpus is redistributed,
   `ATTRIBUTION.jsonl` and the dataset card in §6 become obligations on the fleet rather
   than internal hygiene, and `csd-corpus-publish.py`'s private-always default
   (*"Every repo is created with private=True, passed explicitly rather than relying on an
   account default"*) is what currently keeps that question open rather than answered by
   accident.
7. **Are NSFW / face / PII filters required for a public release?** §2.4 declines to add
   them by default because each needs a model with question 1's problem. If the operator
   requires them, that is a policy decision to record — including which model, at which
   revision, in `lineage`.
