# Pre-registration: replacing the token-rank gate with the read-out probe at 4,000 steps

Written 2026-09-06 at CogSynDelta `2fdc8b2`, revised after adversarial review rg-review3 (PR #69),
before any run it governs. Status: PRE-REGISTERED, NOT RUN. Sources are the short keys in the last table.

## Summary

The W4/W7 retrain gate's rank clause (`token_global_pr_rank >= 2.0 * pooled_pr_rank`) is a proxy
that failed twice. At 50 steps the control arm with both token-aware terms off already clears it at
2.0191x, so a pass cannot distinguish "the objective worked" from "the objective was never on" [W4c].
At 4,000 steps the arms separate (1.0851 control against 1.3475 treatment at batch 512) but no
configuration reaches 2.0x, including the pre-registered batch-1280 run at 1.2102 [W4p] [RT-DEC68].

This document replaces that clause with the thing it stood in for: the W1d matched read-out probe, on
a control arm and a treatment arm that differ in exactly one thing, at production step count, on two
training seeds. Rule, arms, gate dispositions, cost, commands, grader and GO_KILL template are fixed here.

| item | value |
|---|---|
| claim adjudicated | the token-aware objective (`L_token` + `L_decorr`, 0.1 / 0.1) makes the final-block token surface richer than the pooled vector, as a read-out can use it, at 4,000 steps |
| primary metrics | `D = delta(T) - delta(C)` (token-minus-pooled read-out gain, treatment over control) and `A = a(T) - a(C)` (the token read-out itself, treatment over control), both in items of the holdout, each the mean of three read-out seeds |
| threshold | `k* = ceil(2.0 * n / 100)` items from the receipt's own `n`: 11 items = 2.1484 pp at `n = 512`; 2.0 pp is W1d's pre-committed bar [W1d] |
| runs | memory, language, reason x training seeds {0, 1} (arms share the seed; both seeds must pass) x 2 arms = 12 training runs, 12 probes, 12 eval receipts, 2 pre-flight probes |
| cost | about 3.9 GPU-hours on the 3090 Ti plus the unmeasured eval receipts (section 5) |

Table 0. The pre-registration at a glance.

## 1. The claim the gate must adjudicate

DEC-35 licensed the token-aware retrains because W1 measured every production region's token surface as no
richer than its pooled vector (participation-ratio ratios 0.66x to 1.30x) and W1d confirmed it with an instrument
that names no rank definition [W1] [W1d] [RT-DEC35]. The objective adds `L_decorr` and `L_token` at the final block [RT-4.0].

The claim this gate adjudicates: at production step count, does training with the token-aware terms
produce a token surface from which a small read-out extracts more than from the pooled vector,
attributably to the terms rather than to step count, seed or corpus. The rank ratio measured the
spectrum of the surface, a proxy; the read-out measures what a consumer of the surface can do with it,
the thing [MEM-measure]. Three sub-questions fall out: did the token read-out itself improve under the
terms (`A`), is the treatment surface richer than pooled in absolute terms (`delta(T)`, the direction
W1d's rule was written for), and is the gap the objective's (`D`). The first exists because `D` alone
passes a treatment whose pooled path merely regressed [RG3].

## 2. Arms

Every arm of a replicate is identical except the two loss weights, and the two arms start from identical
weights: `torch.manual_seed(cfg.seed)` is the stage's first statement, the trunk is built before the MLM head,
the batch order is `order_seed`'s (pinned to 0), and the encoder has no dropout, so the control draws no RNG
in the loop at all [PT] [RG3]. Arms differing in seed as well as in the variable under test cannot be attributed
[MEM-seeds]. The G26 manifests fix the held-out split independently of the training seed [SPL].

| region | batch | max_len | lr | warmup | eval_every | corpus fingerprint | split manifest (sha256 prefix) |
|---|---:|---:|---|---:|---:|---|---|
| memory | 1280 | 96 | 6.708203932499369e-4 | 266 | 666 | `6fc0cf23ff8591ff2241278f82c001d2` | `memory-6fc0cf23-split0.json` (`e07a5891`) |
| language | 1280 | 96 | 6.708203932499369e-4 | 266 | 666 | `e493273b003f5cf9c239db0ac461b993` | `language-e493273b-split0.json` (`6a2c9a4c`) |
| reason | 512 | 256 | 4.242640687119285e-4 | 266 | 666 | `ca364a92d2c6c5fd259404e0ab6f52a1` | `reason-ca364a92-split0.json` (`77d2c0e1`) |

Table 1. Per-region constants shared by both arms. `lr = 3e-4 * sqrt(batch / 256)`, `warmup =
max(50, steps // 15)`, `eval_every = steps // 6`, as the harness derives them [TA] [W4p-launch].

| identical across the two arms of a replicate | the one thing that differs |
|---|---|
| steps 4000; batch, max_len, lr, warmup as in Table 1; seed s in {0, 1}; `split_seed` 0; `order_seed` 0; the split manifest and the batch-order manifest; corpus fingerprint; encoder `dim 256, depth 4, n_heads 4` (16,021,248 params in every region, reason included); step-0 trunk weights (`init_sha256`, P4); bf16 autocast, fp32 master weights; `token_loss_chunk` 512 on both arms (inert at weight 0); `token_loss_mask_prob` 0.15; no `--retrieve-checkpoint` inheritance for memory (the b1280 production run passed none); code revision; host and card | control: `token_loss_weight 0.0, decorr_weight 0.0`. treatment: `token_loss_weight 0.1, decorr_weight 0.1`, `memory_config()`'s own defaults, the weights an operator launching W4 or W7a gets |

Table 2. What is held identical and what differs.

Reason runs at batch 512 / `max_len` 256 in both arms because batch 1280 OOMed 640 MiB short of the card
at that length [RT-DEC54]; the batch is a replicate constant, so the control is unaffected and the reason
result is comparable to its own control only. Seed 1 replicates init and mask draws only, since the data
order is pinned; its arms share seed 1, and the two replicates are reported as spread, never pooled [MEM-seeds].

## 3. Primary metric and decision rule

The instrument is W1d's read-out, fit three times per arm: the same single-query cross-attention read-out
trained in arm (a) over the final-block token positions and in arm (b) over the pooled vector broadcast to the
same length, identical parameters, 600 steps, batch 256, lr 2e-3, warmup 60, weight decay 1e-4, grad-clip 1.0,
batch order seed 12345, fit on 4,096 train pairs disjoint from the `n`-item holdout, scored once on it, at
read-out seeds 0, 1 and 2 [W1d-script]. Every quantity below is in items of that holdout (`n = 512`, one item = 0.1953 pp).

The threshold is 2.0 pp because that is the number W1d pre-committed and measured against, so "richer than
pooled" means the same thing before and after the retrain [W1d]. Stated generally, the bar is `k* = ceil(2.0 * n / 100)`
items from the receipt's own `n`, tested as `k >= k*`: 11 items at `n = 512`, an effective 2.1484 pp. `>` and `>=`
coincide only when `2.0 * n / 100` is not an integer (`n` not a multiple of 50), so the rule is stated on integers [RG3].

W1d's own validity floor cannot stand: its spread `< 2.0 pp` sits at 0.91x the effect bar, at that boundary a
two-seed AND rule false-passes 7.9 percent under no effect, and a single spread draw certifies a retrieve-grade
instrument (2.54 pp, Table 3) as valid 47 percent of the time [RG3]. Three read-out seeds and a floor of `k* / 3`
on the standard error of each mean (kept in items: 3.667 items = 0.7161 pp at `n = 512`, not 2.0 / 3 pp) bring `sd(D)`
under 1.02 pp and the no-effect false pass under 1.7 percent per seed, 0.03 percent for both. A true effect sitting
exactly on the bar is still KILLed with probability 0.25 by any two-seed AND rule, so a KILL whose `A` lies within one item of `k*` on both seeds is flagged `boundary` in GO_KILL.

| region | delta(a - b) pp | read-out seed spread pp | W1d verdict |
|---|---:|---:|---|
| code (now language) | +0.00 | +0.20 | CONFIRMED |
| compress | -1.76 | -0.98 | CONFIRMED |
| retrieve | -6.05 | -2.54 | CONFIRMED, instrument NOISY |
| vl_latent | -2.93 | -1.17 | CONFIRMED |

Table 3. The W1d numbers the threshold is taken from; memory and reason have never been through this instrument, hence P2's pre-flight [W1d].

The decision rule, fixed before running:

```text
For region R and training seed s in {0, 1}, control C(R, s) and treatment T(R, s); n = the holdout size the
receipt and the probe both report; k* = ceil(2.0 * n / 100) items (11 at n = 512):
  per checkpoint X and read-out seed k in {0, 1, 2}: a_k = arm(a) recall@1 hits, b_k = arm(b) hits (items)
  a(X) = mean_k a_k;  b(X) = mean_k b_k;  delta(X) = a(X) - b(X)
  se_a(X) = sd_k(a_k) / sqrt(3);  se_delta(X) = sd_k(a_k - b_k) / sqrt(3)     (sample sd over the 3 seeds)
  valid(X) = se_a(X) < k* / 3 and se_delta(X) < k* / 3            (3.667 items = 0.7161 pp at n = 512)
  D(R, s) = delta(T) - delta(C);   A(R, s) = a(T) - a(C)
  pass_s = D(R, s) >= k* and A(R, s) >= k* and delta(T) >= k*
  kill_s = A(R, s) < k*
  reverted(R, s) = any recall-type regression C - T > 1.0 point, or spearman regression > 0.01 (section 4)
  sound(R) = valid(X) for all four checkpoints and beats_untrained.recall@1 true for all four
PASS(R)  iff sound(R) and pass_s for both s and reverted(R, s) false for both s
KILL(R)  iff sound(R) and kill_s for both s
INCONCLUSIVE(R) otherwise, with the first matching reason recorded:
  IDENTITY   - a section-7 refusal fired (the run, not the model, is at fault)
  INSTRUMENT - some valid(X) is false
  BASELINE   - some arm did not beat its untrained floor (the model, not the run, is at fault)
  REVERTED   - pass_s on both seeds but reverted(R, s) on either
  SEEDS      - kill_s on one seed and not the other
  MAGNITUDE  - A >= k* on both seeds but D or delta(T) short of k* on at least one
```

PASS means the terms improved the token read-out and left it richer than pooled without damaging the faculty;
the retrain path stands. KILL means the terms at 0.1 / 0.1 did not move the token read-out beyond noise at 4,000
steps; OD-17's branch 1 (the pivot) or a different objective is next, and the amendment is spent, not repeated
[RT-OD17]. REVERTED and MAGNITUDE carry a pre-registered follow-up: a weight sweep (down, up) as a new pre-registration.

The rank ratio stays as a secondary, descriptive metric with no threshold: the participation-ratio and entropy
pairs from `token_aware.final_block_rank`, per arm, never one against the other [MM-9]; `ratio(T) - ratio(C)` is reported, being OD-17 branch-2 clause 2's quantity.

OD-17's `>` versus `>=` question: the recommendation is `>=`, which `k >= ceil(t * n / 100)` embodies. A
strict comparison at float precision lets representation error decide a verdict, as it did for gate (b)
at `0.20000000298023224` [RT-DEC68]; an integer count of the battery's own quantum makes the question vacuous at any `n`.

## 4. What W4's gates (a) to (e) become

| gate | today | under this pre-registration | reason |
|---|---|---|---|
| (a) beats both parents | memory only | REPORT, not gated: pass/fail against the parents' historical receipts is printed per arm for continuity and has no consequence here | it compares against other configs, the same reason clause 2 is re-baselined; re-baselining it costs four parent runs; its fate belongs to W4's own row under OD-17 [RG3] |
| (b) full-pool floors, strict `>` | memory only | REPORT, not gated, read `>=` at the battery's quantum | a capability floor unrelated to the objective; memory's b512 run failed (a) and (b), and this design does not pretend otherwise [W4p] |
| (c) beats BM25 on the FiQA full pool | memory only | DROP as pass/fail; REPORT full-pool BM25 and the closed-pool `lexical_baseline` (TF-IDF and BM25) for both arms | independent of the objective (control 0.100, treatment 0.098 at batch 512) and a data/scale question (FiQA is 1.8% of the corpus); it goes to W4n and OD-17 branch-2 clause 1 [W4p] [RT-OD17] [LEX] |
| (d) beats random init | memory only | GATE on every arm of every region via `beats_untrained.recall@1`; failing it is reason BASELINE, a model fact, never an IDENTITY refusal | the untrained baseline is what makes a result interpretable [MEM-measure] |
| (e) clause 1, rank ratio `>= 2.0x` | all retrains | REPLACE with the section-3 rule; the ratio becomes descriptive | the control clears it at 50 steps and nothing reaches it at 4,000; it is a proxy [W4c] [W4p] |
| (e) clause 2, regression `<= 1` point | all retrains | KEEP, re-baselined: treatment against control of the same replicate; recall-type metrics (`held_out.recall@1`, `held_out.recall@10`, `held_out.mrr`, memory's `retrieval.full_pool.trained.recall@10`) in points, bar 1.0; `graded_held_out.spearman` in correlation units, bar 0.01, the harness's own `_RETRAIN_GATE_REGRESSION_MARGIN`; a breach sets `reverted`, and reverted is never PASS | the control arm is the like-for-like baseline; a correlation and a recall do not share a unit [MEM-seeds] [RG3] |
| banking77 damage-detector probe | never measured | remains not measured, stated as such | out of scope, as the harness already records [MM-7.4] |

Table 4. Disposition of the five W4 gates and the probe the gate names.

## 5. Cost

Wall times come from receipts of the same encoder, step count and card; estimates state their method. Reason
must run alone [RT-DEC54]; the b1280 treatment arm reaches 20,883 MiB at the driver beside the desktop's 1,057 MiB, so it runs alone too and is not admitted to the 5080 unprobed [W4p-launch].

| region | arm | config | basis | wall s | GPU-h |
|---|---|---|---|---:|---:|
| memory | control | b1280 / 96 | estimate: b1280 text cells without terms measured 437 to 656 s, plus about 30 s of BEIR and graded eval (the b1280 treatment total exceeds 4000 x 390 ms by 27 s) | ~700 | 0.20 |
| memory | treatment | b1280 / 96, chunk 512 | measured, `memory-20260903T184441Z.json` | 1,586.7 | 0.44 |
| language | control | b1280 / 96 | measured, `code-20260903T115858Z.json` | 656.4 | 0.18 |
| language | treatment | b1280 / 96, chunk 512 | estimate: same encoder, steps, batch and chunk as memory's treatment | ~1,590 | 0.44 |
| reason | control | b512 / 256 | measured, `reason-20260903T123431Z.json` | 586.3 | 0.16 |
| reason | treatment | b512 / 256, chunk 512 | estimate: control x 2.27, the term overhead the two memory rows above imply (1,586.7 / 700); memory's b512 totals give only 1.22x but include a fixed eval cost, so the larger factor is the conservative one | ~1,329 | 0.37 |
| one seed / two seeds | 6 / 12 runs | | sum of the six rows, then doubled | | 1.79 / 3.58 |
| probes | 12 + 2 pre-flight | | six read-out fits per checkpoint; W1d's four fits plus capture took 22 to 47 s per region | ~60 each | 0.23 |
| `tokenisation.prepare_s` | 12 runs | | outside `elapsed_s` by design; 22.58 s code, 8.74 s memory, 0.03 s reason | | 0.05 |
| eval receipts | 12 | | UNMEASURED: eval receipts record `seconds: 0.0`; the Run lane times the first and records it | | not budgeted |
| total | | | serial on the 3090 Ti, about 4 h wall plus the evals | | ~3.9 |

Table 5. Cost per arm at one seed and in total; b1280 text cells without terms are 0.12 to 0.18 GPU-h [RCPT], and the terms roughly double a step (64 against 142 to 146 ms at b512) [W4c].

## 6. Command lines and receipt fields

Four harness changes are prerequisites and are not this lane's to write. Without P1 the treatment arms of language
and reason cannot be launched through the harness at all: `csd-train-all.py` takes the weights from its `TOKEN_AWARE_REGIONS` table (memory only) and exposes no override [TA].

| id | prerequisite | acceptance |
|---|---|---|
| P1 | `scripts/csd-train-all.py` gains `--token-loss-weight`, `--decorr-weight`, `--token-loss-chunk`, forwarded to `PretrainConfig` in `run_region` and `run_memory_region`; defaults are the table's entry or `0.0 / 0.0 / 2048` | `--dry-run` prints the resolved weights; a test asserts `--regions language --token-loss-weight 0.1 --decorr-weight 0.1` plans `token_aware: true` and the default plans `false` |
| P2 | `scripts/csd-readout-probe.py`: the text path of `measure_w1d.py` with `--region`, `--checkpoint`, `--split-manifest`, `--out`; every constant in section 3 frozen; arms (a) and (b) each fit at read-out seeds 0, 1, 2; output is W1d's per-region result object per seed plus the item-unit means and standard errors, `n_items`, `split_sha256`, `checkpoint_sha256`, `code_revision` | at read-out seed 0 on the three W1 production checkpoints it reproduces `w1d-readout-probe-2026-09-02/results.json` arm numbers exactly [SPL]. PRE-FLIGHT, before any of the 12 runs launch: probe the existing memory and reason checkpoints named by `memory-20260903T184441Z.json` and `reason-20260903T123431Z.json` and report `se_a`, `se_delta`; a region whose pre-flight fails `valid` does not launch until the instrument is changed, as a recorded stop, not a waiver |
| P3 | `scripts/csd-grade-rank-gate.py` implementing section 7; arm identity is the `_resume_fields` key set of `pretrain.py` minus the two weights, never the receipt's whole `config` block, which carries `out_dir`, `device`, `checkpoint_every` and `eval_every` and differs per arm by this document's own layout | fixtures: doctored receipts (wrong seed, wrong split sha, weights swapped, differing `init_sha256`) are refused; a pair differing only in `out_dir` is accepted; fixtures built from the rule reach PASS, KILL and each of the six INCONCLUSIVE reasons |
| P4 | `pretrain_region` stamps `init_sha256` on the receipt: sha256 over the trunk `state_dict` tensors after `torch.manual_seed(cfg.seed)` and before the first optimizer step | two runs of one config at one seed hash equal; weights 0.0 and 0.1 at one seed hash equal; seeds 0 and 1 hash different |

Table 6. Prerequisites the Run lane must see merged, at one pinned sha, before launching.

Layout: `ROOT=/akula-data/csd/prereg-rank-gate-4000`, one state root per arm at `$ROOT/<region>/s<seed>/<arm>`. Arms
must not share a root: checkpoint directories are keyed on the corpus and split-code vintage only, so a second arm finds the first's `final.pt` and refuses to start [W4p-analysis].

Environment: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `CUDA_VISIBLE_DEVICES=0`, `PYTHONPATH=<pinned
worktree>/src`, `$PY` the CogSynDelta `.venv/bin/python`, corpus at `/mnt/fleet-datasets/csd` (reason's at
`/mnt/bulk/csd-corpus`). `$W` is `0.0` for control and `0.1` for treatment; `$S` is the seed. The memory argv is the
b1280 production launch with the weights parameterised; `--regions language` resolves through the alias table and names its receipts and checkpoints `language`.

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

`--shard-limit 4` pins all four codesearchnet shards and `--shard-limit 2` reproduces the reason cell, as the
matrix does; `$RECEIPT` is the run's `<region>-<ts>.json`, `$CKPT` its `checkpoint` field and `$MANIFEST` its
`split.manifest` field. Order on a card: the two pre-flight probes, then control and treatment for one (region,
seed), the next seed, the next region, memory first; each launch waits on the previous run's sentinel, not on a poll [MEM-emit].

| source | fields recorded per arm |
|---|---|
| train receipt | `config.{steps, batch_size, lr, max_len, seed, split_seed, order_seed, token_loss_weight, decorr_weight, token_loss_chunk}` and every other `_resume_fields` key, `init_sha256`, `corpus.fingerprint`, `split.{manifest, sha256, seed}`, `batch_order.sha256`, `checkpoint`, `checkpoint_sha256`, `code_revision.{git_sha, dirty}`, `parameters`, `elapsed_s`, `tokenisation.prepare_s`, `precision`, `held_out.{n_pairs, recall@1, recall@10, mrr}`, `graded_held_out.spearman` (memory), `untrained_baseline.*`, `beats_untrained.*`, `token_aware.{enabled, token_loss_weight, decorr_weight, final_block_rank.*}`; memory also `retrieval.full_pool.{trained, untrained, lexical_bm25}.{recall@10, recall@100, mrr}` and `gates.*` |
| eval receipt | `artifacts.checkpoint_sha256`, `provenance.split.sha256`, `lexical_baseline.{tfidf, bm25}.recall@1`, `rank.recall@1`, `repr.effective_rank_entropy`, `repr.effective_rank_entropy_ratio` (entropy; never divided into a `pr_*` field) |
| probe.json | `n_items`, `checkpoint_sha256`, `split_sha256`; per read-out seed `arms.{a_final_tokens, b_pool_broadcast}.{untrained, trained}.recall@1` and `final_train_stats.{loss, in_batch_acc}` (the over-fit signal); `a_mean_items`, `b_mean_items`, `delta_items`, `se_a_items`, `se_delta_items`; `rank_on_probe_input_activations.*`; `anomaly_flags` |

Table 7. Receipt fields the grader reads; anything else is not part of the decision.

## 7. Pre-registered analysis and the GO_KILL template

The grading script refuses before it grades: every identity check below fails closed, and a refused
replicate is INCONCLUSIVE with reason `IDENTITY` (the run, not the model, is at fault); `beats_untrained` is a verdict input, not a refusal.

```text
PIN = the one code sha all twelve receipts must carry; RESUME_KEYS = _resume_fields(cfg).keys() minus {token_loss_weight,
      decorr_weight} (pretrain.py), NOT the receipt's whole config block; shards, corpus_fingerprint and split_code_fingerprint are not config keys and are covered by r.corpus.{shards, fingerprint} and PIN
for R in (memory, language, reason):
  for s in (0, 1):
    for A in (control, treatment):
      r = newest train receipt under ROOT/R/s{s}/A/receipts;  e = its eval receipt;  p = probe.json
      assert r.config.seed == s and r.config.split_seed == 0 and r.config.order_seed == 0 and r.config.steps == 4000
      assert (r.config.batch_size, r.config.max_len) == TABLE1[R] and r.corpus.fingerprint == TABLE1[R].fingerprint
      assert (r.config.token_loss_weight, r.config.decorr_weight) == ((0, 0) if A == control else (0.1, 0.1))
      assert r.split.sha256 == TABLE1[R].split_sha256 == p.split_sha256 == e.provenance.split.sha256
      assert r.code_revision.git_sha == PIN and not r.code_revision.dirty
      assert r.checkpoint_sha256 == p.checkpoint_sha256 == e.artifacts.checkpoint_sha256
      assert p.n_items == r.held_out.n_pairs;  n = p.n_items;  k_star = ceil(2.0 * n / 100)
    assert C.config[k] == T.config[k] for every k in RESUME_KEYS & r.config.keys();  C.corpus.shards == T.corpus.shards
    assert C.init_sha256 == T.init_sha256;  C.batch_order.sha256 == T.batch_order.sha256
    a[A], b[A], delta[A], se_a[A], se_delta[A] = p_A's item-unit means and standard errors (section 3)
    valid[A] = se_a[A] < k_star / 3 and se_delta[A] < k_star / 3;  baseline[A] = r_A.beats_untrained["recall@1"]
    D[s] = delta[T] - delta[C];  Aeff[s] = a[T] - a[C]
    regression_pts[s] = max over recall-type m of 100 * (C[m] - T[m]);  regression_rho[s] = C.spearman - T.spearman
    reverted[s] = regression_pts[s] > 1.0 or regression_rho[s] > 0.01
    descriptive: ratio[A] = token_global_pr_rank / pooled_pr_rank; lexical_fraction[A] = held_out.recall@1 / e_A.lexical_baseline.tfidf.recall@1
  verdict[R], reason[R] = the section-3 rule over s in (0, 1);  boundary[R] = KILL and |Aeff[s] - k_star| <= 1 for both s
write grade.json (every number above, per R, s, A, plus n, k_star and k_star * 100 / n) and GO_KILL.md verbatim
```

`GO_KILL.md` template. The Run lane fills every field from `grade.json` and changes nothing else.

```markdown
# GO_KILL: prereg-rank-gate-4000 run <n>
code sha: <PIN>   dirty: false   host: <host>   card: <card>   graded: <UTC>
prereg: docs/design/evidence/prereg-rank-gate-4000-2026-09-06/PREREG.md at <prereg sha>
pre-flight (existing checkpoints): memory se_a <> se_delta <> valid <>; reason se_a <> se_delta <> valid <>

| region | seed | n | k* | a_C | b_C | a_T | b_T | delta_C | delta_T | D | A | valid_C | valid_T | regr pt | regr rho | ratio_C | ratio_T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| memory | 0 | | | | | | | | | | | | | | | | |
| memory | 1 | | | | | | | | | | | | | | | | |
| language | 0 | | | | | | | | | | | | | | | | |
| language | 1 | | | | | | | | | | | | | | | | |
| reason | 0 | | | | | | | | | | | | | | | | |
| reason | 1 | | | | | | | | | | | | | | | | |

| region | verdict | reason (INCONCLUSIVE only: IDENTITY / INSTRUMENT / BASELINE / REVERTED / SEEDS / MAGNITUDE) | reverted | boundary |
|---|---|---|---|---|
| memory | | | | |
| language | | | | |
| reason | | | | |

Item-unit columns (a, b, delta, D, A) are means over three read-out seeds; k* = ceil(2.0 * n / 100).
memory, reported not gated, treatment seed 0 / seed 1: (a) <> / <>; (b) <>, read >= at quantum / <>;
full-pool BM25 recall@10 <x>. Receipts: <12 train receipt paths and sha256>; probes: <14 probe.json sha256>.
Decision rule applied verbatim from PREREG.md section 3. Nothing above was edited by hand.
```

## Sources

| key | source |
|---|---|
| W1; W1d, W1d-script; W4c | `docs/design/evidence/w1-token-rank-2026-09-02/`; `docs/design/evidence/w1d-readout-probe-2026-09-02/` (`results.json`, `measure_w1d.py`); `docs/design/evidence/w4-control-arm-2026-09-03/` (`README.md`, the three summary JSONs) |
| W4p, W4p-analysis, W4p-launch | `docs/design/evidence/w4-production-runs-2026-09-03/` (`README.md`, `ANALYSIS.md`, `launch-w4-memory-b1280-20260903T181738Z.json`, the four receipts) |
| RT-4.0, RT-DEC35, RT-DEC54, RT-DEC68, RT-OD17; MM-7.4, MM-9 | `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` section 4.0 (the gate), DEC-35, DEC-54, DEC-68, OD-17, rows W1b/W1d/W4/W4n/W7a/W7v; `docs/design/METRICS-METHODOLOGY.md` sections 1, 2.7, 7.4, 9, 10 |
| SPL; LEX; TA; PT | `config/mind/splits/SEED0-REPORT.json` and the `*-split0.json` manifests (G26); `src/cogsyndelta/eval/lexical.py`; `scripts/csd-train-all.py` and `src/cogsyndelta/regions/memory.py` `main()`; `src/cogsyndelta/regions/pretrain.py` (`_resume_fields`, `pretrain_region`, the receipt `config` block) |
| RCPT | `/akula-data/csd/receipts/{code-20260903T115858Z, compress-20260903T120818Z, retrieve-20260903T121603Z, reason-20260903T123431Z}.json` |
| RG3; MEM-seeds, MEM-measure, MEM-emit | adversarial review rg-review3 of this document at `2cc767c`, 2026-09-06 (PR #69): counter-examples CX1b to CX3b, the validity-floor arithmetic, the quantisation rule, the `config`-block blocker; orchestrator memory: identical seeds as control; measure the thing, not the proxy; emitters, not polling |
