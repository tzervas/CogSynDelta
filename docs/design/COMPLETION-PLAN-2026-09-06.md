# CogSynDelta completion plan, 2026-09-06

**Summary.** The per-region pipeline is proven; the composed model does not exist as code; the region set cannot be frozen until the memory region has a checkpoint and one operator decision (OD-17) is taken. The plan below drives to composed training in four waves, each wave built from file-disjoint lanes run by agents in their own worktrees, with GPU time reserved for the few lanes that need it. Source: the state-versus-plan map (`docs/design/STATE-VS-PLAN-2026-09-06.md`).

## Wave 1 (running now, zero GPU)

| lane | branch | what it proves or builds | owner |
|---|---|---|---|
| A | feat/w0-faculty-protocol | `tokens()`/`pool()` protocol, adapters for text and visual encoders, parameter table per region, guard against vocabulary ids | Sonnet high |
| B | chore/design-code-hygiene | superseded markers on `core/`, MindSpec stream_dim check per taxonomy, TRAINING-SUPERSET and MODEL-MANIFESTS folds, status notes on the stale corpus contract, config-vs-trained dims recorded | Sonnet medium |
| E | feat/quant-geometry-metrics | `quant.geometry.*` (cosine, p05, min, kNN@10, std ratio) in eval-quant receipts and cards, fail-closed split guard, methodology section | Sonnet high |
| F | docs/interconnect-module-spec | buildable spec for `src/cogsyndelta/interconnect/`: components, shapes, parameter arithmetic, forward pass, DEC-50 training contract, gates as guards, test plan, open decisions | Fable |
| E5 | feat/reason-e5 (worktree run) | latent-step prediction for the reason region, the promoted objective after the g48 KILL | workflow, GPU 0 |

Table: wave 1 lanes. All merge through Forgejo PRs with a second lens; guard code gets an adversarial pass.

## Wave 2 (after wave 1 merges; about 2 GPU-hours)

| lane | depends on | what | GPU-h |
|---|---|---|---|
| memory cell | corpus admission of the union spec | first matrix cell for the memory region with full receipts (train, eval, quant, eval-quant), seed 0, fixed split | 0.5 |
| rank-gate replacement | E0 (done, PR #58) | pre-registered 4,000-step control and treatment arms to replace the invalidated 2.0x gate; the only gate that can judge token-aware retrains | 1.0 |
| visual receipts and geometry | lane E | eval and quant receipts for the visual cells in the matrix dir with geometry fields; plan search re-run against geometry once the floor policy is decided | 0.3 |
| E2 reason batch ablation | E0 | epoch-matched b256 vs b512, 2 step counts, 3 seeds; settles whether the b256 result is an epoch effect | 0.9 |

Table: wave 2 lanes.

## Wave 3 (composed model build; smoke-scale GPU)

| lane | depends on | what |
|---|---|---|
| interconnect build | lane F spec, lane A protocol | controller, adapters, K/V bank, workspace, read-out, schedule emission; unit + integration + guard-can-fail tests; CPU smoke; adversarial review before merge |
| episodic store | operator rules on the six contract gaps | build and probe (E0/E1 of the episodic rows) |
| compose stage | interconnect build | `compose` stage in the trainer and receipt writer, composed eval writer, composed card |
| token-aware retrains W1b/W7a (reason, language) and W7v (visual, OD-4) | rank-gate replacement, lane A | mandatory before the W2b freeze; 2 seeds each, identical seeds across arms |

Table: wave 3 lanes.

## Wave 4 (composed training; 4 to 12 GPU-hours, unmeasured)

W2b freeze, W5 phases A to D with regions frozen, W5b write-back gate (at most 1 pt own-bin drop), W6 composed metric, then episodic E2 admission retrain and W8/W9/W10 (quantize, topology agreement at least 95 percent, final). Emitter-wired, one workflow, sequenced after wave 3 is green.

## Corpus and parameters (the lever after the module set is complete)

Licence rulings first (code zero-delta, compress SynCSE swap, retrieve GooAQ tier, MATH vs PRM800K, composed repo tag), then the code six-language mix and max_len past the 93.9 percent truncation, reason's third provenance group, classify out-of-label-space rows, retrieve rebalance, then the per-faculty 1B data targets.

## Decisions the plan needs from the operator

| id | question | recommendation |
|---|---|---|
| OD-17 | W4 gate: `>` vs `>=`, pivot vs amend | amend with `>=` and the 4,000-step gate from wave 2; the 50-step gate is dead |
| OD-4 | W7v corpus source for the 128 px token-aware visual rebuild | visual-clean-v1, the mix run 2 passed on |
| floor | visual quant floor policy | measure first (lane E), then per-tensor search against mean cosine at least 0.99; expect mixed 3/4/5-bit |
| dims | config declares 512/1024, trained is 256/4/4 | keep 256 for the module set; make the config describe what is trained; raise dims only in the parameter-expansion phase |
| episodic | six contract gaps (taxonomy section 8) | answer from lane F's option list |
| licences | five rulings listed above | rule per region; the composed licence inherits the strictest tier |

Table: open decisions.

## Standing rules applied

Identical seeds and fixed splits across arms; harness-shaped smoke before any stage script is pushed; the repo's own gate before every commit; readability on every document; GPU-first; workflows and agents execute, the orchestrator plans and verifies by execution.
