# W4 (memory) production runs — 2026-09-03

Four `memory` region runs at production step count (4000 steps), same code path
(`cogsyndelta.regions.memory:main`, the only entry point that runs pretrain + the BEIR
full-pool eval + `w4_gates()` in one process — `csd-train-all.py`'s `run_region()` never
reaches the eval/gates code, per every launch note below), same corpus, same 512-pair
held-out set, same `stsb-validation` graded set, same 57,638-passage FiQA BEIR pool. All
four ran on the RTX 3090 Ti (`akula-prime`).

Files in this directory, copied (not moved) from their run locations:

| file | run | role |
|---|---|---|
| `memory-20260903T164611Z.json` | batch 512, terms on | production receipt |
| `launch-w4-memory-20260903T163517Z.json` | batch 512, terms on | its launch note |
| `memory-20260903T170747Z.json` | batch 512, terms off | control-arm probe receipt |
| `launch-w4-probe-control512-20260903T165838Z.json` | batch 512, terms off | its launch note |
| `memory-20260903T170934Z.json` | batch 256, terms on | batch probe receipt |
| `launch-w4-probe-batch256-20260903T165838Z.json` | batch 256, terms on | its launch note |
| `ANALYSIS.md` | (all runs to that point) | failure analysis written after the production run and before the two probes' results were folded in — see "Notes" below for what postdates it |
| `memory-20260903T184441Z.json` | batch 1280, terms on, chunk 512 | pre-registered-batch receipt |
| `launch-w4-memory-b1280-20260903T181738Z.json` | batch 1280, terms on, chunk 512 | its launch note |

## Result table

All values read directly from each receipt's `held_out`, `graded_held_out`,
`retrieval.full_pool.trained`, `retrieval.full_pool.lexical_bm25`, and `gates` blocks.
`recall@1` / `graded` are the 512-pair in-mixture held-out set and the 1,498-pair
`stsb-validation` graded set; `full-pool r@10` / `r@100` / `MRR` are against the full
57,638-passage FiQA pool; `rank ratio` is gate (e)'s PR-rank clause
(`token_global_pr_rank / pooled_pr_rank`, required ≥ 2.0).

| batch | terms | chunk | steps | recall@1 | graded (spearman) | full-pool r@10 | full-pool r@100 | full-pool MRR | BM25 r@10 | rank ratio | gates passed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | on | — ¹ | 4000 | 0.5996 | 0.6775 | 0.0300 | 0.1040 | 0.0134 | 0.4400 | 1.5864 | 1/5 (d only) |
| 512 | off | — ¹ | 4000 | 0.7754 | 0.7313 | 0.1000 | 0.2280 | 0.0496 | 0.4400 | 1.0851 | 1/5 (d only) |
| 512 | on | — ¹ | 4000 | 0.8145 | 0.7463 | 0.0980 | 0.2800 | 0.0531 | 0.4400 | 1.3475 | 1/5 (d only) |
| 1280 | on | 512 | 4000 | 0.8535 | 0.7893 | 0.2000 | 0.4240 | 0.1168 | 0.4400 | 1.2102 | 3/5 (a, b, d) |

¹ The `--token-loss-chunk` flag and the `token_loss_chunk` receipt field do not exist at
these three runs' code revision (`4d2d8860af744a4179e96ef035b3435168a4e26f`) — the flag
was added afterward, in `eb735ab` ("expose token_loss_chunk on the CLI and
run_memory_pretrain"), which is the code revision the batch-1280 run launched at. These
three runs did not chunk the masked-token-loss computation at all; "—" reflects that the
concept was not yet available to them, not that chunking was explicitly turned off.

Gate identifiers, as recorded in each receipt's `gates` block: (a) `a_beats_both_parents`,
(b) `b_full_pool_thresholds`, (c) `c_beats_bm25`, (d) `d_beats_random_init`, (e)
`e_retrain_gate` (two clauses: the PR-rank clause and the receipt-regression clause).
`gates.passed` is `false` on all four runs — none is a full pass.

## The negatives curve

Full-pool recall@10 rises monotonically with batch size — the in-batch negative count —
from 0.0300 at batch 256, to 0.0980–0.1000 at batch 512 (terms on/off), to 0.2000 at batch
1280, confirming Probe A's hypothesis (`ANALYSIS.md` §5) that InfoNCE's negatives axis
(`log(B)` nats: 5.55 / 6.24 / 7.15 at 256 / 512 / 1280) is a real quality lever and that the
pre-registered batch of 1280 — infeasible at batch 512's VRAM footprint until the
`--token-loss-chunk` flag made it fit — was the correct target all along.

## Remaining gate failures (batch 1280, pre-registered run)

Of the five pre-registered W4 gate clauses, the batch-1280 run clears (a), (b) and (d).
Two remain failing: **gate (c) `c_beats_bm25`** (trained full-pool recall@10 0.2000 vs
lexical BM25's 0.4400 on the same pool) and **gate (e) `e_retrain_gate`**, specifically its
PR-rank clause (ratio 1.2102 vs the required ≥ 2.0; the receipt-regression clause on this
run passes, at 0.0 worst regression).
