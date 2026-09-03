# W4 control-arm evidence — measured 2026-09-03

Evidence for the reviewer finding **B2**: `regions/memory.py`'s `MEASURED_VRAM_AT_BATCH_512`
note (written in commit `cdcde8b`) claimed "the first real-data evidence the terms are not a
no-op at production scale" from a *single* 50-step run with both terms on, with no control
arm to compare against. A reviewer ran the identical config with both weights at `0.0` and
got `token_global_pr_rank 12.446 / pooled 3.424 = 3.635x` — already past §4.0's 2.0x gate
with the objective **disabled**, on the harness that existed at the time.

This directory is the same experiment (same config shape: 50 steps, `batch_size=512`,
`max_len=96`, the production encoder shape, real fleet corpus, 3090 Ti), re-run **after**
the B3 harness-alignment fix (`fix(memory): align W4's final-block rank measurement with
W1's pre-committed harness`, `git rev-parse HEAD` at run time: `5554fcb`), which changed
what `token_global_pr_rank`/`pooled_pr_rank` measure (one side of the held-out pairs, the
pre-committed `pr_effective_rank`, no subsampling, `nan` below 2 rows — see that commit).
The reviewer's mismatched-harness 3.635x is **not** comparable to the numbers below; both
were re-measured from scratch on the aligned harness, not adjusted.

## The three arms

All three: `memory_config()`'s own corpus (FiQA primary + AllNLI/NaturalQuestions/GooAQ
extra sources), `steps=50`, `batch_size=512`, `max_len=96`,
`encoder=TextEncoderConfig(dim=256, depth=4, n_heads=4, max_len=96)` (16,021,248 params),
`seed=0`, `bf16` autocast, `device="cuda"`, `checkpoint_every=0` (smoke run — periodic
checkpoints skipped; `final.pt` is always written). Called as
`pretrain_region(memory_config(**overrides))` directly, **not** `run_memory_pretrain` —
this evidence needs the rank stats only, and the full 57,638-passage BEIR/BM25 eval
`run_memory_pretrain` layers on top would have cost far more GPU time for nothing this
evidence reads. `out_dir` under `/akula-data/session-backup-staging/`, never
`/akula-data/csd/receipts` (the real fleet location) — these are throwaway smoke-run
checkpoints, not meant to be loaded again.

| arm | `token_loss_weight` | `decorr_weight` |
|---|---:|---:|
| `control` | 0.0 | 0.0 |
| `token_only` | 0.1 | 0.0 |
| `both_on` | 0.1 | 0.1 |

`0.1`/`0.1` are `memory_config()`'s own `TOKEN_LOSS_WEIGHT`/`DECORR_WEIGHT` defaults — the
weights an operator launching row W4 actually gets, not a hand-tuned arm.

## Results (aligned harness, `pr_effective_rank`, full surface, one side of the pairs)

| arm | `pooled_pr_rank` | `token_global_pr_rank` | ratio | clears 2.0x? |
|---|---:|---:|---:|---:|
| `control` (both off) | 1.9996 | 4.0374 | **2.0191x** | **yes** |
| `token_only` | 1.9994 | 4.0472 | **2.0242x** | **yes** |
| `both_on` | 2.1810 | 4.8880 | **2.2412x** | **yes** |

Full per-arm numbers (both rank definitions, VRAM, timing, checkpoint sha256, config
fingerprint, `held_out`/`graded_held_out` metrics, encoder config) are in
`control-summary.json`, `token_only-summary.json`, `both_on-summary.json`.

## Gate (e) clause 1 does NOT discriminate at 50 steps, on this harness

**The control arm — both terms off — already clears §4.0's `token_global_pr_rank >= 2.0 *
pooled_pr_rank` threshold, at 2.0191x.** This is not a design-doc matter to fix here (the
task instruction is explicit: report the numbers, do not change the threshold) — it is a
measurement fact about this specific operating point: 50 steps is short enough that
`pooled_pr_rank` sits barely above the floor (`1.9996`, next to the theoretical minimum of
`1.0` for a fully collapsed pooled surface) simply because InfoNCE has not yet had the
steps to differentiate items, while `token_global_pr_rank` starts measurably higher
(`~4.0`) for an unrelated reason — a token-position surface has more inherent local
variation than a pooled vector even before training does anything to it. The **ratio**
between an almost-collapsed denominator and a not-yet-collapsed numerator crosses 2.0x
before either the objective's terms or the InfoNCE loss itself have done meaningful work.
**Gate (e)'s clause 1 (the PR-rank clause) is marked NOT DISCRIMINATING at 50 steps**: a
`passed: True` on this clause, at this step count, on this harness, does not distinguish
"the token-aware terms worked" from "the terms were never turned on at all." The clause is
unchanged; whether it discriminates at the production 8,000-step scale this evidence was
never run at is not measured here.

## Honest attribution: essentially all of the movement is `L_decorr`, not `L_token`

Read off the RAW `token_global_pr_rank` (the ratio's discriminating power is compromised
above; the raw numbers are not):

- `control` → `token_only`: `4.0374` → `4.0472`, **+0.0098** (`token_loss_weight=0.1`
  alone, isolated from `L_decorr` entirely — `decorr_weight=0.0` in this arm).
- `token_only` → `both_on`: `4.0472` → `4.8880`, **+0.8408** (`decorr_weight=0.1` added
  on top).
- `control` → `both_on`: `4.0374` → `4.8880`, **+0.8506** total.

Of the total movement from `control` to `both_on` (+0.8506), **+0.0098 (1.2%) is
attributable to `L_token`** and **+0.8408 (98.8%) to `L_decorr`**, at these weights, at 50
steps, on this corpus. This is the same qualitative shape B2's control-arm finding
reported on the old (mismatched) harness (there: "~91% from L_decorr") — re-measured here
on the aligned harness, the L_token contribution is smaller still. `L_token` is not
provably a no-op (see the mutation-verified gradient tests in
`tests/test_token_aware_objective.py`, commit `fix(memory): ...` / `test(memory): give
L_token and L_decorr their own rank/gradient guard arms` — L_token's gradient reaches the
trunk, and the isolated-arm test shows it measurably changes the rank on the *toy* CPU
fixture), but at these weights and this step count on the real corpus its effect on this
specific statistic is small next to `L_decorr`'s.

## The corrected claim

`regions/memory.py`'s `MEASURED_VRAM_AT_BATCH_512["note"]` previously read (commit
`cdcde8b`): *"Even at 50 steps the token-aware terms already moved the final-block rank
ratio to 5.74x pooled (16.11/2.81), well past §4.0's 2.0x gate, the first real-data
evidence the terms are not a no-op at production scale, not just on the tiny synthetic
fixtures tests/test_token_aware_objective.py constructs."* That number was measured on the
pre-alignment harness (both-sides-of-the-pair surface through the subsampling
`participation_ratio`), had no control arm, and the ratio it reported cannot be compared
against W1's own 0.66x-1.30x for exactly the reason B3 exists.

The honest statement, on the aligned harness, with a control arm: **at 50 steps,
`memory`'s control arm (both terms off) already clears the 2.0x ratio gate (2.0191x), so
the ratio threshold does not discriminate a working objective from a disabled one at this
scale; the terms' real, attributable effect is visible in the raw `token_global_pr_rank`
numbers, where `L_decorr` accounts for ~99% of the +0.85 movement from control to both-on
and `L_token` for ~1%.** `regions/memory.py` is updated to state this instead — see the
`EVIDENCE_50_STEP_CONTROL_ARM` constant added next to `MEASURED_VRAM_AT_BATCH_512` in the
commit this README ships with.

## Reproduce

From this worktree, on `akula-prime` (RTX 3090 Ti, `CUDA_VISIBLE_DEVICES=0`), with the
corpus mounted at `/mnt/fleet-datasets/csd` and the tokenizer at
`/mnt/fleet-datasets/tritter/gpt2_tokenizer.json`:

```
PYTHONPATH=<this worktree>/src CUDA_VISIBLE_DEVICES=0 \
  <CogSynDelta venv>/bin/python measure_w4_control_arm.py control
PYTHONPATH=<this worktree>/src CUDA_VISIBLE_DEVICES=0 \
  <CogSynDelta venv>/bin/python measure_w4_control_arm.py token_only
PYTHONPATH=<this worktree>/src CUDA_VISIBLE_DEVICES=0 \
  <CogSynDelta venv>/bin/python measure_w4_control_arm.py both_on
```

(`measure_w4_control_arm.py`'s own `sys.path.insert` already points at this worktree's
`src/`, so `PYTHONPATH` is redundant but harmless.) Each arm is a separate process so CUDA
memory is fully released between runs — the 3090 Ti was shared with a concurrent
quantize job for this measurement. `out_dir`s point at
`/akula-data/session-backup-staging/w4-control-arm-2026-09-03/control-arm-runs/<arm>/` on
the host this ran on; change `OUT_ROOT` in the script for a different host/session.
Checkpoints written there (~192MB/arm) are throwaway smoke-run weights and are not part of
this evidence directory — only the summary JSONs and this README are.

## Provenance

- **code_revision** (this worktree's `HEAD` at run time): `5554fcb5efc3e8453fe09e0cf75b862f5c13519b`
  (`test(memory): give L_token and L_decorr their own rank/gradient guard arms` — the
  second of the two commits preceding this one, both required for the aligned harness).
- **seed**: `0` for all three arms (`memory_config()`'s default; unchanged by any override
  here).
- **corpus fingerprint** (`corpus.fingerprint` in each run's own pretrain receipt, all
  three arms identical since the corpus/columns/seed are unchanged across arms):
  `6fc0cf23ff8591ff2241278f82c001d2`.
- **checkpoint sha256** (differs per arm — different weights trained different final
  checkpoints): `control` `41ab6e3f07edce4f26eb5de515b7e04e20e941adb8031b34719fb71a126055fd`
  (first run; the resumed re-verification run in `control-summary.json` reports a
  different final sha256, `1864b359e0cc1cb4241cdc0205539e37163bba5323a893fd4a4c33eef224b797`,
  from `_checkpoint_payload`'s embedded receipt timestamp differing between the fresh
  and resumed writes — the trained weights and every measured rank number are identical
  between the two, confirmed by re-running `control` a second time and getting
  `token_global_pr_rank` `4.037415218462928` both times), `token_only`
  `30ecd612e6f048b430b0775a4db0d1719e2ae87dba31dba9bf20702913d3b5c6`, `both_on`
  `70f298eb09dc6e3f5cea4e54f1f673f82a562a0a3e2c7249e3160233e7c8a7ce`.
- **measured**: 2026-09-03, `akula-prime`, RTX 3090 Ti (24 GiB), shared with a concurrent
  quantize job (~1 GiB in use before these runs started; peak usage per arm recorded in
  each summary JSON's `peak_allocated_mib`/`peak_reserved_mib`).
