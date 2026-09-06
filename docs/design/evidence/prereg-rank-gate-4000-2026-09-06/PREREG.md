# Pre-registration: replacing the token-rank gate with the read-out probe at 4,000 steps

Written 2026-09-06 at CogSynDelta `2fdc8b2`, before any run it governs. Status: PRE-REGISTERED,
NOT RUN. Sources are cited by the short keys in the last table.

## Summary

The W4/W7 retrain gate's rank clause (`token_global_pr_rank >= 2.0 * pooled_pr_rank`) is a proxy
that failed twice. At 50 steps the control arm with both token-aware terms off already clears it at
2.0191x, so a pass cannot distinguish "the objective worked" from "the objective was never on" [W4c].
At 4,000 steps the arms separate (1.0851 control against 1.3475 treatment at batch 512) but no
configuration reaches 2.0x, including the pre-registered batch-1280 run at 1.2102 [W4p] [RT-DEC68].

This document replaces that clause with the thing it stood in for: the W1d matched read-out probe,
on a control arm and a treatment arm that differ in exactly one thing, at production step count, on
two training seeds. Rule, arms, gate dispositions, cost, command lines, grading script and GO_KILL
template are fixed here so a Run lane executes without interpretation and nobody moves the bar.

| item | value |
|---|---|
| claim adjudicated | the token-aware objective (`L_token` + `L_decorr`, 0.1 / 0.1) makes the final-block token surface richer than the pooled vector, as a read-out can use it, at 4,000 steps |
| primary metric | `D = delta_T - delta_C`, where `delta_X` is the W1d probe's arm(a) minus arm(b) recall@1 on checkpoint X, in percentage points |
| threshold | 2.0 pp, W1d's own pre-committed bar, unchanged [W1d] |
| replication | training seeds 0 and 1; arms share the seed within a replicate; both must pass |
| runs | memory, language, reason x 2 seeds x 2 arms = 12 training runs, 12 probes, 12 eval receipts |
| cost | about 3.5 GPU-hours on the 3090 Ti, serial (section 5) |

Table 0. The pre-registration at a glance.

## 1. The claim the gate must adjudicate

DEC-35 licensed the token-aware retrains because W1 measured every production region's token
surface as no richer than its pooled vector (participation-ratio ratios 0.66x to 1.30x) and W1d
confirmed it with an instrument that names no rank definition [W1] [W1d] [RT-DEC35]. The retrain
objective adds `L_decorr` and `L_token` at the final block to change exactly that property [RT-4.0].

The claim this gate adjudicates: at production step count, does training with the token-aware
terms produce a token surface from which a small read-out extracts more than from the pooled
vector, attributably to the terms rather than to step count, seed or corpus. The rank ratio
measured the spectrum of the surface, a proxy; the read-out measures what a consumer of the surface
can do with it, the thing [MEM-measure]. Two sub-questions fall out: is the treatment surface richer
than pooled in absolute terms (`delta_T >= 2.0 pp`, the direction W1d's rule was written for), and
is the objective the cause (`D >= 2.0 pp` against a control differing only in the two loss weights).

## 2. Arms

Every arm of a replicate is identical except the two loss weights. Seeds are identical across arms
because a seed changes init, data order and masking, and arms that differ in seed as well as in the
variable under test cannot be attributed [MEM-seeds]. The held-out split is fixed by the G26
manifests, which reproduce the historical seed-0 draw independently of the training seed [SPL].

| region | batch | max_len | lr | warmup | eval_every | corpus fingerprint | split manifest (sha256 prefix) |
|---|---:|---:|---|---:|---:|---|---|
| memory | 1280 | 96 | 6.708203932499369e-4 | 266 | 666 | `6fc0cf23ff8591ff2241278f82c001d2` | `memory-6fc0cf23-split0.json` (`e07a5891`) |
| language | 1280 | 96 | 6.708203932499369e-4 | 266 | 666 | `e493273b003f5cf9c239db0ac461b993` | `language-e493273b-split0.json` (`6a2c9a4c`) |
| reason | 512 | 256 | 4.242640687119285e-4 | 266 | 666 | `ca364a92d2c6c5fd259404e0ab6f52a1` | `reason-ca364a92-split0.json` (`77d2c0e1`) |

Table 1. Per-region constants shared by both arms. `lr = 3e-4 * sqrt(batch / 256)`, `warmup =
max(50, steps // 15)`, `eval_every = steps // 6`, as the harness derives them [TA] [W4p-launch].

| identical across the two arms of a replicate | the one thing that differs |
|---|---|
| steps 4000; batch, max_len, lr, warmup as in Table 1; seed s in {0, 1}; `split_seed` 0; `order_seed` 0; the split manifest and the batch-order manifest; corpus fingerprint; encoder `dim 256, depth 4, n_heads 4` (16,021,248 params in every region, reason included); bf16 autocast, fp32 master weights; `token_loss_chunk` 512 on both arms (inert at weight 0, set anyway so the configs differ in nothing else); `token_loss_mask_prob` 0.15; no `--retrieve-checkpoint` inheritance for memory (the b1280 production run passed none); code revision; host and card | control: `token_loss_weight 0.0, decorr_weight 0.0`. treatment: `token_loss_weight 0.1, decorr_weight 0.1`, `memory_config()`'s own defaults, the weights an operator launching W4 or W7a gets |

Table 2. What is held identical and what differs.

Reason runs at batch 512 / `max_len` 256 in both arms because batch 1280 OOMed 640 MiB short of the
card at that length [RT-DEC54]; the batch is a replicate constant, so the control is unaffected and
the reason result is comparable to its own control only. Seed 1 is a replication axis: its arms
share seed 1, and the two replicates are reported side by side as spread, never pooled [MEM-seeds].

## 3. Primary metric and decision rule

The instrument is W1d's, unchanged: the same single-query cross-attention read-out trained in
arm (a) over the final-block token positions and in arm (b) over the pooled vector broadcast to
the same length, identical parameters, 600 steps, batch 256, lr 2e-3, warmup 60, weight decay
1e-4, grad-clip 1.0, read-out seed 0, batch order seed 12345, fit on 4,096 train pairs disjoint from
the 512-item holdout, scored once on the holdout [W1d-script]. `delta_X` is arm(a) recall@1 minus
arm(b) recall@1 on checkpoint X, in percentage points on the 512-item holdout; one item is 0.1953 pp.

The threshold is 2.0 pp because that is the number W1d pre-committed and measured against, so
"richer than pooled" means the same thing before and after the retrain [W1d]. The W1d read-out seed
spread (Table 3) justifies both the number and the validity condition: three of four regions sit
well inside 2.0 pp when arm (a) is re-fit with read-out seed 1 on the same frozen encoder, and the
fourth is exactly the case the rule must refuse to adjudicate [W1d]. On a 512-item diagonal 2.0 pp
is 10.24 items, so the rule fires at 11 or more items under either `>` or `>=`.

| region | delta(a - b) pp | read-out seed spread pp | W1d verdict |
|---|---:|---:|---|
| code (now language) | +0.00 | +0.20 | CONFIRMED |
| compress | -1.76 | -0.98 | CONFIRMED |
| retrieve | -6.05 | -2.54 | CONFIRMED, instrument NOISY |
| vl_latent | -2.93 | -1.17 | CONFIRMED |

Table 3. The W1d numbers the threshold and the validity condition are taken from [W1d].

The decision rule, fixed before running:

```text
For region R and training seed s in {0, 1}, control C(R, s) and treatment T(R, s):
  delta_X  = 100 * (recall@1[arm a] - recall@1[arm b]) on checkpoint X, W1d instrument, read-out seed 0
  D(R, s)  = delta_T - delta_C
  valid(X) = |delta_a(read-out seed 0) - delta_a(read-out seed 1)| < 2.0 pp   (W1d verify-by-failing on X)
PASS(R)  iff for both s: valid(C) and valid(T) and D(R, s) >= 2.0 pp and delta_T >= 2.0 pp
KILL(R)  iff for both s: valid(C) and valid(T) and D(R, s) <  2.0 pp
INCONCLUSIVE(R) otherwise, with exactly one of these reasons recorded:
  INSTRUMENT  - some valid(X) is false (read-out seed spread >= 2.0 pp on that checkpoint)
  SEEDS       - the two replicates disagree on D(R, s) >= 2.0 pp
  MAGNITUDE   - D(R, s) >= 2.0 pp on both seeds but delta_T < 2.0 pp on at least one
All comparisons are on values quantised to items (k / 512); >= and > coincide at 11 items.
```

PASS means the objective produces a token surface richer than pooled and the retrain path stands.
KILL means the objective at 0.1 / 0.1 does not move what a read-out can extract beyond noise at
4,000 steps; OD-17's branch 1 (the pivot) or a different objective is next, and the amendment is
spent, not repeated [RT-OD17]. MAGNITUDE is the one INCONCLUSIVE outcome with a pre-registered
follow-up: a weight sweep written as a new pre-registration, not a re-read of this one.

The rank ratio stays as a secondary, descriptive metric with no threshold: the participation-ratio
and entropy pairs from `token_aware.final_block_rank`, per arm, never one definition against the
other [MM-9]. `ratio(T) - ratio(C)` is reported too: it is what OD-17's branch-2 clause 2 would gate on.

OD-17's `>` versus `>=` question is resolved here by construction, and the recommendation is `>=`.
A strict comparison at float precision lets representation error decide a verdict, which is what
happened to gate (b) at `0.20000000298023224` [RT-DEC68]. Every threshold here is stated in the
metric's own quantum (items of 512), where the two readings coincide; applied to gate (b) the floor
reads as inclusive at the battery's quantum, and `>=` is recommended for every future floor because
it is the reading under which representation error can never flip a verdict.

## 4. What W4's gates (a) to (e) become

| gate | today | under this pre-registration | reason |
|---|---|---|---|
| (a) beats both parents | memory only | KEEP for memory, unchanged; not applicable to language and reason | it checks the merge, not the objective [MM-7.4] |
| (b) full-pool floors, strict `>` | memory only | KEEP for memory, read `>=` at the battery's quantum (section 3) | an absolute capability floor, independent of the objective |
| (c) beats BM25 on the FiQA full pool | memory only | DROP as pass/fail; REPORT full-pool BM25 and the closed-pool `lexical_baseline` (TF-IDF and BM25) for both arms | independent of the objective (control 0.100, treatment 0.098 at batch 512) and a data/scale question (FiQA is 1.8% of the corpus); it goes to W4n and OD-17 branch-2 clause 1 [W4p] [RT-OD17] [LEX] |
| (d) beats random init | memory only | KEEP as a sanity check and extend: every arm of every region must have `beats_untrained.recall@1` true | the untrained baseline is what makes a result interpretable [MEM-measure] |
| (e) clause 1, rank ratio `>= 2.0x` | all retrains | REPLACE with the section-3 rule; the ratio becomes descriptive | the control clears it at 50 steps and nothing reaches it at 4,000; it is a proxy [W4c] [W4p] |
| (e) clause 2, regression `<= 1` point | all retrains | KEEP, re-baselined: treatment against control of the same replicate on `held_out.recall@1`, `held_out.recall@10`, `held_out.mrr`, `graded_held_out.spearman` where present, and `retrieval.full_pool.trained.recall@10` for memory; worst regression `<= 1.0` point; historical figures (language 0.9863 at b1280, memory's parents) are reported only | the control arm is the like-for-like baseline; the historical figures were trained at other steps and batches [MEM-seeds] |
| banking77 damage-detector probe | never measured | remains not measured, stated as such | out of scope, as the harness already records [MM-7.4] |

Table 4. Disposition of the five W4 gates and the probe the gate names.

A region's verdict on the objective is section 3's; (a), (b) and (d) remain memory's own gates,
reported beside it. A treatment arm failing the re-baselined clause 2 is reverted [RT-4.0].

## 5. Cost

Wall times come from receipts of the same encoder, step count and card; estimates state their
method. Reason must run alone [RT-DEC54]; the b1280 treatment arm reaches 20,883 MiB at the driver
beside the desktop's 1,057 MiB, so it runs alone too and is not admitted to the 5080 unprobed [W4p-launch].

| region | arm | config | basis | wall s | GPU-h |
|---|---|---|---|---:|---:|
| memory | control | b1280 / 96 | estimate: b1280 text cells without terms measured 437 to 656 s, plus about 30 s of BEIR and graded eval (the b1280 treatment total exceeds 4000 x 390 ms by 27 s) | ~700 | 0.20 |
| memory | treatment | b1280 / 96, chunk 512 | measured, `memory-20260903T184441Z.json` | 1,586.7 | 0.44 |
| language | control | b1280 / 96 | measured, `code-20260903T115858Z.json` | 656.4 | 0.18 |
| language | treatment | b1280 / 96, chunk 512 | estimate: same encoder, steps, batch and chunk as memory's treatment | ~1,590 | 0.44 |
| reason | control | b512 / 256 | measured, `reason-20260903T123431Z.json` | 586.3 | 0.16 |
| reason | treatment | b512 / 256, chunk 512 | estimate: control x 1.5; memory's b512 pair was x 1.22 (617.4 / 504.9) and `max_len` 256 masks 2.67x more positions | ~900 | 0.25 |
| one seed | | | sum of the six rows | | 1.67 |
| two seeds | 12 runs | | | | 3.34 |
| probes and evals | 12 + 12 | | W1d text regions took 22 to 47 s each; `csd-benchmark.py` minutes each | | 0.20 |
| total | | | serial on the 3090 Ti, about 4 h wall | | ~3.5 |

Table 5. Cost per arm at one seed and in total. Text cells at b1280 without terms are 0.12 to
0.18 GPU-h [RCPT]; the terms roughly double a step (64 ms against 142 to 146 ms at b512) [W4c].

## 6. Command lines and receipt fields

Three harness changes are prerequisites and are not this lane's to write. Without P1 the treatment
arms of language and reason cannot be launched through the harness at all: `csd-train-all.py` takes
the weights from its `TOKEN_AWARE_REGIONS` table (memory only) and exposes no override [TA].

| id | prerequisite | acceptance |
|---|---|---|
| P1 | `scripts/csd-train-all.py` gains `--token-loss-weight`, `--decorr-weight`, `--token-loss-chunk`, forwarded to `PretrainConfig` in `run_region` and `run_memory_region`; defaults are the table's entry or `0.0 / 0.0 / 2048` | `--dry-run` prints the resolved weights; a test asserts `--regions language --token-loss-weight 0.1 --decorr-weight 0.1` plans `token_aware: true` and the default plans `false` |
| P2 | `scripts/csd-readout-probe.py`: the text path of `measure_w1d.py` with `--region`, `--checkpoint`, `--split-manifest`, `--out`; every constant in section 3 frozen; arms (a), (b) and the read-out-seed-1 control (arm (c) optional); output is W1d's per-region result object plus `split_sha256`, `checkpoint_sha256`, `code_revision` | on the three W1 production checkpoints it reproduces `w1d-readout-probe-2026-09-02/results.json` arm numbers exactly, which the G26 manifests make possible because they reproduce the historical seed-0 draw [SPL] |
| P3 | `scripts/csd-grade-rank-gate.py` implementing section 7 | a fixture with doctored receipts (wrong seed, wrong split sha, weights swapped) is refused; a fixture built from this document's rule yields each of PASS, KILL and the three INCONCLUSIVE reasons |

Table 6. Prerequisites the Run lane must see merged, at one pinned sha, before launching.

Layout: `ROOT=/akula-data/csd/prereg-rank-gate-4000`, one state root per arm at
`$ROOT/<region>/s<seed>/<arm>`. Arms must not share a root: checkpoint directories are keyed on the
corpus and split-code vintage only, so a second arm finds the first's `final.pt` and refuses to start [W4p-analysis].

Environment: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUDA_VISIBLE_DEVICES=0`,
`PYTHONPATH=<pinned worktree>/src`, `$PY` the CogSynDelta `.venv/bin/python`, corpus at
`/mnt/fleet-datasets/csd` (reason's at `/mnt/bulk/csd-corpus`). `$W` is `0.0` for control and `0.1`
for treatment; `$S` is the seed. The memory argv is the b1280 production launch with the weights
parameterised; `--regions language` resolves through the alias table and names its receipts and
checkpoints `language`, matching the manifest.

```bash
$PY -m cogsyndelta.regions.memory --steps 4000 --batch-size 1280 --lr 0.0006708203932499369 \
  --warmup-steps 266 --max-len 96 --seed $S --eval-every 666 --holdout-pairs 512 \
  --checkpoint-every 200 --token-loss-weight $W --decorr-weight $W --token-loss-chunk 512 \
  --eval-split dev --out-dir $ROOT/memory/s$S/$ARM/receipts

$PY scripts/csd-train-all.py --regions language --steps 4000 --batch 1280 --shard-limit 4 \
  --seed $S --token-loss-weight $W --decorr-weight $W --token-loss-chunk 512 \
  --state $ROOT/language/s$S/$ARM

$PY scripts/csd-train-all.py --regions reason --steps 4000 --batch 512 --shard-limit 2 \
  --seed $S --token-loss-weight $W --decorr-weight $W --token-loss-chunk 512 \
  --state $ROOT/reason/s$S/$ARM

$PY scripts/csd-benchmark.py --regions $REGION --state $ROOT/$REGION/s$S/$ARM --train-receipt $RECEIPT

$PY scripts/csd-readout-probe.py --region $REGION --checkpoint $CKPT --split-manifest $MANIFEST \
  --out $ROOT/$REGION/s$S/$ARM/probe.json

$PY scripts/csd-grade-rank-gate.py --root $ROOT --code-sha $PIN \
  --out docs/design/evidence/prereg-rank-gate-4000-2026-09-06/run-1/
```

`--shard-limit 4` pins all four codesearchnet shards and `--shard-limit 2` reproduces the reason
cell, as the matrix does; `$RECEIPT` is the run's `<region>-<ts>.json`, `$CKPT` its `checkpoint`
field and `$MANIFEST` its `split.manifest` field. Order on a card: control then treatment for one
(region, seed), then the next seed, then the next region, memory first; each run writes its
sentinel and the next launches on the sentinel, not on a poll [MEM-emit].

| source | fields recorded per arm |
|---|---|
| train receipt | `config.{steps, batch_size, lr, max_len, seed, split_seed, order_seed, token_loss_weight, decorr_weight, token_loss_chunk}`, `corpus.fingerprint`, `split.{manifest, sha256, seed}`, `batch_order.sha256`, `checkpoint`, `checkpoint_sha256`, `code_revision.{git_sha, dirty}`, `parameters`, `elapsed_s`, `precision`, `held_out.{recall@1, recall@10, mrr}`, `graded_held_out.spearman` (memory), `untrained_baseline.*`, `beats_untrained.*`, `token_aware.{enabled, token_loss_weight, decorr_weight, final_block_rank.*}`; memory also `retrieval.full_pool.{trained, untrained, lexical_bm25}.{recall@10, recall@100, mrr}` and `gates.*` |
| eval receipt | `artifacts.checkpoint_sha256`, `provenance.split.sha256`, `lexical_baseline.{tfidf, bm25}.recall@1`, `rank.recall@1`, `repr.effective_rank` (entropy; never divided into a `pr_*` field) |
| probe.json | `checkpoint_sha256`, `split_sha256`, `arms.{a_final_tokens, b_pool_broadcast, a_control_seed}.{untrained, trained}.recall@1`, `arm_a_pct`, `arm_b_pct`, `delta_a_minus_b_points`, `verify_by_failing.{delta, symmetric_within_2pt}`, `rank_on_probe_input_activations.*`, `anomaly_flags` |

Table 7. Receipt fields the grader reads; anything else is not part of the decision.

## 7. Pre-registered analysis and the GO_KILL template

The grading script refuses before it grades: every identity check below fails closed, and a refused
replicate is INCONCLUSIVE with reason `IDENTITY`, which means the run, not the model, is at fault.

```text
PIN = the one code sha all twelve receipts must carry, passed on the command line
for R in (memory, language, reason):
  for s in (0, 1):
    for A in (control, treatment):
      r = newest train receipt under ROOT/R/s{s}/A/receipts;  e = its eval receipt;  p = probe.json
      assert r.config.seed == s and r.config.split_seed == 0 and r.config.order_seed == 0
      assert r.config.steps == 4000 and (r.config.batch_size, r.config.max_len) == TABLE1[R]
      assert (r.config.token_loss_weight, r.config.decorr_weight) == ((0, 0) if A == control else (0.1, 0.1))
      assert r.corpus.fingerprint == TABLE1[R].fingerprint
      assert r.split.sha256 == TABLE1[R].split_sha256 == p.split_sha256 == e.provenance.split.sha256
      assert r.code_revision.git_sha == PIN and not r.code_revision.dirty
      assert r.checkpoint_sha256 == p.checkpoint_sha256 == e.artifacts.checkpoint_sha256
      assert r.beats_untrained["recall@1"]
    assert every key of C.config equals T.config except token_loss_weight and decorr_weight
    assert C.batch_order.sha256 == T.batch_order.sha256
    delta[A] = round(p_A.delta_a_minus_b_points * 512 / 100) items, for A in (C, T)
    valid[A] = p_A.verify_by_failing.symmetric_within_2pt
    D[s] = delta[T] - delta[C]
    regression[s] = max over m in REGRESSION_METRICS[R] of 100 * (C[m] - T[m])
    descriptive: ratio[A] = token_global_pr_rank / pooled_pr_rank; lexical_fraction[A] = held_out.recall@1 / e_A.lexical_baseline.tfidf.recall@1
  verdict[R] = the section-3 rule over s in (0, 1); reverted[R] = any regression[s] > 1.0
write grade.json (every number above, per R, s, A) and GO_KILL.md from the template, verbatim
```

`GO_KILL.md` template. The Run lane fills every field from `grade.json` and changes nothing else.

```markdown
# GO_KILL: prereg-rank-gate-4000 run <n>
code sha: <PIN>   dirty: false   host: <host>   card: <card>   graded: <UTC>
prereg: docs/design/evidence/prereg-rank-gate-4000-2026-09-06/PREREG.md at <prereg sha>

| region | seed | delta_C pp | delta_T pp | D pp | valid_C | valid_T | regression pt | ratio_C | ratio_T |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|
| memory | 0 | | | | | | | | |
| memory | 1 | | | | | | | | |
| language | 0 | | | | | | | | |
| language | 1 | | | | | | | | |
| reason | 0 | | | | | | | | |
| reason | 1 | | | | | | | | |

| region | verdict | reason (INCONCLUSIVE only: INSTRUMENT / SEEDS / MAGNITUDE / IDENTITY) | reverted |
|---|---|---|---|
| memory | | | |
| language | | | |
| reason | | | |

memory capability gates, treatment arm, seed 0 / seed 1: (a) <> / <>; (b) <>, read >= at quantum / <>;
(d) <> / <>; full-pool BM25 recall@10 <x> (reported, not gated).
Receipts: <12 train receipt paths and sha256>; probes: <12 probe.json sha256>.
Decision rule applied verbatim from PREREG.md section 3. Nothing above was edited by hand.
```

## Sources

| key | source |
|---|---|
| W1; W1d, W1d-script; W4c | `docs/design/evidence/w1-token-rank-2026-09-02/`; `docs/design/evidence/w1d-readout-probe-2026-09-02/` (`results.json`, `measure_w1d.py`); `docs/design/evidence/w4-control-arm-2026-09-03/` (`README.md`, the three summary JSONs) |
| W4p, W4p-analysis, W4p-launch | `docs/design/evidence/w4-production-runs-2026-09-03/` (`README.md`, `ANALYSIS.md`, `launch-w4-memory-b1280-20260903T181738Z.json`, the four receipts) |
| RT-4.0, RT-DEC35, RT-DEC54, RT-DEC68, RT-OD17 | `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` section 4.0 (the gate), DEC-35, DEC-54, DEC-68, OD-17, rows W1b/W1d/W4/W4n/W7a/W7v |
| MM-7.4, MM-9 | `docs/design/METRICS-METHODOLOGY.md` sections 1, 2.7, 7.4, 9, 10 |
| SPL; LEX; TA | `config/mind/splits/SEED0-REPORT.json` and the `*-split0.json` manifests (G26); `src/cogsyndelta/eval/lexical.py`; `scripts/csd-train-all.py` and `src/cogsyndelta/regions/memory.py` `main()` |
| RCPT | `/akula-data/csd/receipts/{code-20260903T115858Z, compress-20260903T120818Z, retrieve-20260903T121603Z, reason-20260903T123431Z}.json` |
| MEM-seeds, MEM-measure, MEM-emit | orchestrator memory: identical seeds as control; measure the thing, not the proxy; emitters, not polling |
