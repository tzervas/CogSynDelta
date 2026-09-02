# CSD remaining program

Live plan. Each task names its gate — the observable that says it is done — because a task
without one gets reported complete on vibes.

Status: `todo` | `wip` | `done` | `blocked` | `deferred`

---

## P0 — Correctness debt. Blocks everything downstream.

Nothing measured before P0.1 lands is trustworthy: the text encoder attended to padding, so
every metric was batch-composition dependent.

| id | task | gate | status |
|----|------|------|--------|
| P0.1 | Retrain code, compress, retrieve with attention masking fixed | 3 receipts, all `beats_untrained` true | wip |
| P0.2 | Re-run benchmark battery on retrained regions | eval receipts written, anisotropy < 0.9 | todo |
| P0.3 | Re-run PTQ against retrained fp32 baselines | quant receipts, drop within 0.01 | todo |
| P0.4 | Fix csd-storage-tier verification | manifest honours the SAME excludes as the rsync | todo |
| P0.5 | Commit the train-dependency guard in csd-train-all.py | committed, gate green | todo |

**P0.4 detail.** The 661 GB offload transferred and then failed verification:
`hot_manifest 4c4c6c0b` vs `cold_manifest 7300c204`. Cause is mine — the rsync excludes
receipts/checkpoints/manifest.json/*.log, but the manifest hashes every file on both sides,
so a mismatch is guaranteed. The hot copy was correctly kept. Fix the comparison, re-verify,
and only then reclaim. Do NOT delete anything until a corrected manifest matches.

## P1 — Repo hygiene. Nearly done.

| id | task | gate | status |
|----|------|------|--------|
| P1.1 | Make bitnet-rs and ternary-rs private | `isPrivate: true`, `isArchived: true` preserved | todo |

Both are archived, which makes them read-only to the API (`HTTP 403`). Sequence is
unarchive -> flip -> re-archive, so the archived state they started in is restored.

## P2 — Corpus expansion

`/bulk` is 5.9 TB at ~10% used; the eviction threshold is 50%. Catalogue holds 11
licence-verified datasets across 6 domains, none fetched.

| id | task | gate | status |
|----|------|------|--------|
| P2.1 | Fetch the catalogued datasets to /bulk/csd-corpus | manifests written, row counts recorded | todo |
| P2.2 | Wire classify + reason regions into the runner | dry-run resolves shards for both | todo |
| P2.3 | Train the new regions | receipts with `beats_untrained` | todo |

Licence gate is structural: entries not marked TRAIN_OK are refused and there is no
override flag. A mirror's tag is not evidence about its upstream — BeIR/scifact is tagged
cc-by-sa-4.0 while allenai/scifact, which it mirrors, is cc-by-nc-2.0.

## P3 — Visual region maturation

`vl_latent` passed its gates but weakly: probe top-1 0.0606 on 200 classes, and the run used
3 GB of a 24 GB card.

| id | task | gate | status |
|----|------|------|--------|
| P3.1 | Scale the VL run (steps and batch) | probe top-1 materially above 0.0606, not collapsed | todo |
| P3.2 | Add flickr30k as extra pretrain data | loader filters `__MACOSX/` junk; 31,783 images | todo |

## P4-P6 — Curriculum. Order is mandatory.

Foundation is step 4, not step 1.

| id | task | gate | status |
|----|------|------|--------|
| P4 | Router over trained regions | routing accuracy beats uniform baseline | todo |
| P5 | Composed mind | composed beats best single region on a mixed set | todo |
| P6 | Foundation training | after P4 and P5, never before | todo |

## P7 — Publication

| id | task | gate | status |
|----|------|------|--------|
| P7.1 | Weights-only checkpoint export | exported file ~64 MB, not 184 MB | todo |
| P7.2 | Push regions to private HF repos | repos exist, private, contain weights | todo |

Checkpoints are 184 MB of which only 64 MB is weights; the rest is AdamW optimizer state.
Fine for resuming, wrong to publish.

## P8 — Deferred by explicit operator decision

- Recursive/looped latent transformers
- CI image minimisation
- CPU-mode polish for tiny/quantized variants
- Quantum compute backend (no hardware)
