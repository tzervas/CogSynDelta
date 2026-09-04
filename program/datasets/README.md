# Dataset ingest adapter — how the factory's output becomes CSD training data

**Status:** design + a checkable tool (`scripts/csd-corpus-admit.py`). No fetch runs from
here. The fetch tool itself (`dataset-factory`, `tzervas/dataset-factory`) is a separate
repo per the tooling-lives-in-its-own-repo rule — this directory is CogSynDelta's side of
the seam: what the catalogue means, where fetched bytes land, and what a dataset must prove
before it is admitted into a region's training corpus.

This is a thin adapter, not a second implementation. The factory owns search, licence
verification, fetch, receipts and checksums. CogSynDelta owns the corpus contract (what a
region's training set is *for* and its balance rules) and the admission gate that connects
the two.

## 1. What the factory consumes

`docs/design/datasets/catalogue-2026-09-03.json` — schema `csd-dataset-factory-catalogue/v1`,
159 candidate entries at last count. Each entry carries, per dataset:

- `repo_id`, `faculty`, `provenance_group`
- `licence_mirror_tag` and `licence_upstream_verbatim` **as separate fields** — a mirror's
  licence tag is not evidence about its upstream (`dataset-mirror-licences-lie`; ten
  confirmed mismatches on this project alone). `licence_upstream_source` +
  `licence_fetch_date` record where the upstream text was actually read.
- `grant_scope` — whether the licence covers the whole corpus, metadata only, or code only.
- `verdict` ∈ `{PERMISSIVE_OK, ATTRIBUTION, SHARE_ALIKE, NC, REFUSE, BLOCKING, UNVERIFIED}`
- `verification_status` ∈ `{VERIFIED, CONTRADICTED, UNVERIFIABLE, REFUSED-CLOSED}` — a
  `CONTRADICTED` entry already carries the overturned verdict in the `verdict` field above;
  only `VERIFIED` entries are ever fetch- or admission-eligible.
- `redistribute{nc,sa,nd,attribution}`, `provenance_red_flags[]`, `enrichment_plan`,
  `enrichment_licence_result`, `why`, `caveat`.

`faculty` is the catalogue's taxonomy (`memory`, `language_code`, `language_trunk`,
`visual`, `moral_safety`, `reasoning`, `numeric_math`) — it is **not** the same vocabulary
as a CSD region name (`code`, `classify`, `reason`, `compress`, `retrieve`, `memory`,
`vl_latent`, `residual_mlp`, `stream_vae`). The two overlap in places (`memory` names both
a faculty and a region, and they are not guaranteed to mean the same corpus) and diverge in
others (`language_trunk` has no region today — it is the empty "general bin" `CORPUS-
CONTRACT.md` §2.7 names as the project's largest unfilled gap). **This mapping is not
inferred automatically anywhere in this adapter.** Whoever admits a fetched dataset states
the target region explicitly (`csd-corpus-admit.py --region`), because a wrong automatic
guess here would silently misroute a corpus the same way `CORPUS-CONTRACT.md`'s opening
defect describes for router triggers.

Background docs: `docs/design/DATASET-FACTORY-CATALOGUE-2026-09-03.md` (the narrative
survey), `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md` §"Enriched and derived datasets" (how
enrichment propagates licence), `docs/design/evidence/dataset-factory-2026-09-03/` (the raw
verification passes).

## 2. Where fetched data lands

The factory fetches on gpu5080 (`~/.venv-csd-fetch`) into

```
/bulk/csd-corpus/factory-2026-09-03/<faculty>/<dataset-id>/
```

— a brand-new subtree per pass (dated), never overwriting an existing `<region>/<name>/`
path in the corpus that `csd-corpus-expand.py` and training already read from. Reachable as
`/mnt/bulk/csd-corpus/factory-2026-09-03/...` over NFS from every other host for read-only
work (receipt/checksum verification, admission checks) without needing the fetch venv.

Each fetched dataset directory carries, at minimum, what the factory's contract promises:

- the fetched data itself (parquet, per the `csd-corpus-expand.py` convention where
  applicable)
- the upstream licence text **verbatim**, not just the mirror tag
- a fetch receipt with a **sha256 manifest** of the fetched bytes — a gap
  `csd-corpus-expand.py` itself does not close (its `MANIFEST.json` records provenance and
  row counts but no checksum; the ground pass for this ingest confirmed this by reading the
  code, not the docstring) and which the factory tool must add
- an attribution manifest
- a `provenance.json` naming the catalogue entry's `provenance_group`, `verdict`, and
  `verification_status` at fetch time — this is the file `csd-corpus-admit.py` (§3) reads

## 3. Admission into `CORPUS-CONTRACT.md`

A fetched dataset sitting on `/bulk` is not yet part of a region's training corpus. Admission
is a separate, explicit step, gated on `CORPUS-CONTRACT.md`'s rules — run
`scripts/csd-corpus-admit.py` (§4) before wiring a fetched dataset into a region's fetch/train
config. The checklist it prints:

1. **Licence tier vs. the target region's tier.** `scripts/csd-publish-checkpoint.py`'s
   `LICENCE_TIER` table is the source of truth for what a region is *already* licensed under
   (`mit` < `cc-by-sa-4.0` < `cc-by-nc-sa-4.0`, strictest wins per the operator's governing
   rule in `csd-release-licence-decision`). A dataset whose verdict maps to a stricter tier
   than the region's current tier requires an **explicit** tier upgrade, stated the same way
   `LICENCE-FOR-OPEN-WEIGHTS.md`'s Rider 1 requires a taxonomy merge to state its licence
   cost — never a silent absorption. Verdict → tier: `PERMISSIVE_OK`/`ATTRIBUTION` → `mit`
   (attribution owed, but no NC/SA restriction); `SHARE_ALIKE` → `cc-by-sa-4.0`; `NC` →
   `cc-by-nc-sa-4.0`; `REFUSE`/`BLOCKING`/`UNVERIFIED` → inadmissible outright, independent
   of tier.
2. **Provenance group not already dominant, per B1.** `CORPUS-CONTRACT.md` §"B1 — Maximum
   single-source share ≤ 0.40": the dataset's `provenance_group`, added at its stated row/
   token count, must not push that group's share of the region's post-admission corpus past
   0.40. Computed on provenance groups, never on file or shard counts (`CORPUS-CONTRACT.md`'s
   own framing: "splitting one corpus across ten shards must not read as ten sources").
3. **`verification_status == VERIFIED`.** A `CONTRADICTED`, `UNVERIFIABLE` or
   `REFUSED-CLOSED` entry is inadmissible regardless of its `verdict` field — an entry moves
   off those states only with a fresh primary read, per the ground pass's discipline, not by
   the checklist being run again on stale evidence.

All three must pass for admission. The tool is read-only and advisory-enforced: it does not
mutate `CORPUS-CONTRACT.md`, a region's fetch list, or `csd-regions.json`; it prints a
PASS/FAIL per check and a nonzero exit code on any FAIL, for a human or an agent to act on.

## 4. `scripts/csd-corpus-admit.py`

```
python3 scripts/csd-corpus-admit.py \
    --provenance /bulk/csd-corpus/factory-2026-09-03/<faculty>/<dataset-id>/provenance.json \
    --region retrieve \
    --existing-shares '{"gooaq": 3012496, "natural_questions": 104071}'
```

Reads one factory `provenance.json` (§2's shape) plus the target region name and (optionally)
the region's current per-provenance-group counts, and prints the three-row checklist from §3
with each row's PASS/FAIL and the reasoning number behind it (the computed tier, the
post-admission share, the verification status read). Exits 0 only if all three pass. See
`tests/test_csd_corpus_admit.py` for the mutation tests that confirm each check can actually
fail, not just pass by construction.

## 5. Next fetch list — pass 2

Tracked in `program/REMAINING.md` under **P2′f-a**, "THE NEXT TEN FETCHES" — that row is the
live pass-2 fetch queue; this file does not duplicate it. This ingest pass added no new fetch
targets of its own; it built the admission seam the next fetch pass writes through.
