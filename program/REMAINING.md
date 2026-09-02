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
| P0.4 | Fix csd-storage-tier verification | manifest honours the SAME excludes as the rsync | done — homelab SSD 1.5T -> 2.1T free |
| P0.5 | Commit the train-dependency guard in csd-train-all.py | committed, gate green | done |

**P0.4 detail.** The 661 GB offload transferred and then failed verification:
`hot_manifest 4c4c6c0b` vs `cold_manifest 7300c204`. Cause is mine — the rsync excludes
receipts/checkpoints/manifest.json/*.log, but the manifest hashes every file on both sides,
so a mismatch is guaranteed. The hot copy was correctly kept. Fix the comparison, re-verify,
and only then reclaim. Do NOT delete anything until a corrected manifest matches.

## P1 — Repo hygiene. Nearly done.

| id | task | gate | status |
|----|------|------|--------|
| P1.1 | Make bitnet-rs and ternary-rs private | `isPrivate: true`, `isArchived: true` preserved | done — both verified |

Both are archived, which makes them read-only to the API (`HTTP 403`). Sequence is
unarchive -> flip -> re-archive, so the archived state they started in is restored.

## P2 — Corpus expansion

`/bulk` is 5.9 TB at ~10% used; the eviction threshold is 50%. Catalogue holds 11
licence-verified datasets across 6 domains, none fetched.

| id | task | gate | status |
|----|------|------|--------|
| P2.1 | Fetch the catalogued datasets to /bulk/csd-corpus | manifests written, row counts recorded | partial — 8/11 fetched, 19 GB; apps, hotpotqa, banking77 errored |
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

## P9 — Modern training stack

Make training, fine-tuning and quantization idempotent, parameterised, pausable, resumable
and fast. Ordered by value-per-risk, not by novelty. Every item states what it buys, because
"modern" is not a reason to add something.

Sequencing constraint: P9.1-P9.3 all edit regions/pretrain.py. They must land ONE AT A TIME
or they collide. P0.1 resumable checkpointing is in flight against that same file.

| id | task | what it buys | gate | status |
|----|------|--------------|------|--------|
| P9.0 | Survey: what applies here, measured | grounding — avoids adding technique for its own sake | written design doc with measured baselines | todo |
| P9.1 | bf16 mixed precision | ~2x step time and half the activation memory on Ampere; the freed VRAM buys batch size, and in InfoNCE the negatives ARE the batch | step time and peak VRAM measured before/after; recall within noise of the fp32 run | todo |
| P9.2 | Gradient accumulation | effective batch beyond VRAM, which is the single biggest lever on contrastive quality | effective batch 4x physical, recall improves or is explained | todo |
| P9.3 | torch.compile | free throughput on an unchanged model | step time measured before/after; identical metrics | todo |
| P9.4 | Config-driven runs (declarative, not CLI flags) | idempotent and re-runnable; a run is a file you can diff, not a command someone typed | same config re-run reproduces the receipt | todo |
| P9.5 | LoRA / adapter fine-tuning | fine-tune a region without retraining it; serves capability-per-parameter directly | adapter-only training beats frozen baseline at <5% of trained params | todo |
| P9.6 | QAT (quantization-aware training) | recovers accuracy PTQ leaves on the table at aggressive widths | at 3-bit, QAT beats the PTQ result on the same held-out set | todo |
| P9.7 | Early stopping on the gate metric | stops burning GPU on a run that has plateaued | run halts within N evals of no improvement | todo |

Already in place, do not rebuild: warmup+cosine LR, gradient clipping, deterministic
seeding with corpus fingerprints, receipt-based experiment tracking, Prometheus export,
sensitivity-driven PTQ with real sub-byte packing.

## P8 — Deferred by explicit operator decision

- Recursive/looped latent transformers
- CI image minimisation
- CPU-mode polish for tiny/quantized variants
- Quantum compute backend (no hardware)

## Running a training job

Training runs as a systemd **user** unit (`deploy/systemd/csd-train@.service`, installed to
`~/.config/systemd/user/`) so a run survives the death of whatever shell launched it.
`nohup ... &` only blocks SIGHUP -- it does not survive the launching process group being
killed, which is exactly how two unattended runs died silently in one day: the log stopped
mid-run with no error and no receipt, and the only evidence was `nvidia-smi` showing an idle
GPU. `setsid` was the same-day stopgap; the unit is the durable fix, since systemd owns the
process directly rather than an interactive shell's background job.

Start a run (regions comma-separated, no spaces, matching `--regions` in
`scripts/csd-train-all.py`):

    systemctl --user start csd-train@code,compress,retrieve

Follow it:

    journalctl --user -u csd-train@code,compress,retrieve -f

Check whether it is still running, or how it ended:

    systemctl --user status csd-train@code,compress,retrieve

Notes:
- Steps/batch/shard-limit/state are fixed in the unit to the known-good production
  invocation (`--steps 8000 --batch 256 --shard-limit 0 --state /akula-data/csd`); only the
  region list is templated per-instance.
- `,` is not a valid systemd unit-name character, so every command above logs a harmless
  "Invalid unit name ... maybe you should use systemd-escape?" warning. It still works --
  the unit uses `%I` (escaping undone), not `%i`, to recover the literal comma-separated
  region list before it reaches `--regions`. Do not "fix" the warning yourself.
- `Restart=no` is deliberate: a failed `beats_untrained` gate means a broken configuration,
  not a transient fault. It needs a human or agent to read the receipt in
  `/akula-data/csd/receipts` and change something, not a timer retrying forever.
- Only one instance should run at a time -- there is one GPU. Check `nvidia-smi` and
  `systemctl --user list-units | grep csd-train` before starting a second one.
