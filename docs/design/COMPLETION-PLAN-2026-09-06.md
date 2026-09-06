# CogSynDelta completion plan, 2026-09-06

**Summary.** Wave 1 is complete and merged into `main` at `9cc046f`: the faculty protocol, the
docs-versus-code hygiene fixes, the interconnect module specification, this plan and its state
map, and the matrix harness's code pin. One wave-1 chore (the `ci_local.sh` temp-isolation fix)
sits on an unmerged branch; see the table below. Two wave-2 experiments concluded: E2 found the
reason region's "batch 256 beats 512" result survives an epoch-matched control but is confounded
with learning rate, and E5 killed the K=1 latent-step objective, leaving the reason region with no
validated step-sensitive training objective. The invalidated 2.0x rank gate now has a
pre-registered replacement awaiting adoption (OD-17) and a run. Quantization geometry metrics are
implemented, motivated by evidence that the production visual quant floor was chosen against a
probe blind to real representation drift. Wave 3 (the interconnect module) is under way lane by
lane against the specification, gated by a guard-skeptic pass before any guard-carrying lane
merges. The composed model still does not exist as code, and the region set cannot be frozen until
OD-17 is settled and the memory region has a checkpoint. Source: the state-versus-plan map
(`docs/design/STATE-VS-PLAN-2026-09-06.md`).

## Wave 1 (merged)

| lane | branch | what it proves or builds | status |
|---|---|---|---|
| A | feat/w0-faculty-protocol | `tokens()`/`pool()` protocol (`src/cogsyndelta/faculty/protocol.py`), adapters for text and visual encoders, parameter table per region (`docs/technical/faculty-protocol.md`), guard against vocabulary ids | merged, PR #72 |
| B | chore/design-code-hygiene | superseded markers on `core/` (`DEC-12`), MindSpec stream_dim check now warns rather than raises per taxonomy §2.2, TRAINING-SUPERSET and MODEL-MANIFESTS folds, CORPUS-CONTRACT pointer fix, config-vs-trained dims recorded as an open discrepancy (see Decisions, below) | merged, PR #68 |
| F | docs/interconnect-module-spec | buildable spec for `src/cogsyndelta/interconnect/` (`docs/design/INTERCONNECT-MODULE-SPEC.md`): components, shapes, parameter arithmetic, forward pass, DEC-50 training contract, gates as guards, test plan, lane plan (IC-0 to IC-11); three adversarial rounds (threat-modeler and skeptic on draft 1, a re-review that closed five gaps in the capping loop) | merged, PR #67 |
| — | docs/state-vs-plan-and-completion-plan | this plan and the state-versus-plan map | merged, PR #66 |
| G | chore/pin-run-code-sha-2fdc8b2 | repin `program/matrix/csd-matrix.yaml`'s `run.code.sha` to `main` `2fdc8b2` (split manifests, G26 guards, lexical baselines), so text-region matrix cells plan against the current harness | merged, PR #74 |
| E | feat/quant-geometry-metrics | `quant.geometry.*` (mean/min/p05 cosine, kNN@10 identity agreement, `latent_std_ratio`) in eval-quantized receipts and cards, fail-closed reference guard (G27), `METRICS-METHODOLOGY.md` §23 | implemented; PR #77 not yet merged — see Wave 2's visual row |
| H | chore/ci-local-unique-basetemp | private `pytest --basetemp` and a `TMPDIR`-honouring temp root per `ci_local.sh` run, closing a shared-basetemp race between concurrent worktrees and an opaque `torch.save` failure on a full `/tmp` | branch exists (`45c1f4b`), not confirmed merged — this checkout could not reach Forgejo to verify current status; re-check before relying on it |

Table: wave 1 lanes and their merge status. Every merged lane went through a Forgejo PR with a
second lens; guard-carrying code (lane E's G27, lane F's spec) got an adversarial pass. The E5
lane this table previously listed as "running now" (`feat/reason-e5-latent-step`, latent-step
prediction for the reason region) concluded with a KILL verdict — it produced a result, not a
merged deliverable, and is reported under Wave 2 below rather than here.

## Wave 2 (after wave 1 merges; about 2 GPU-hours plus the rank-gate run)

| lane | depends on | what | status | GPU-h |
|---|---|---|---|---|
| memory cell | corpus admission of the union spec | first matrix cell for the memory region with full receipts (train, eval, quant, eval-quant), seed 0, fixed split | pending; PR #70 landed the corpus note and the warmup-steps pin (266) that makes the cell W4-comparable, no cell trained yet | 0.5 |
| rank-gate replacement | E0 (done, PR #58) | replace the invalidated 2.0x rank gate with a matched read-out probe at 4,000 steps: control/treatment arms at two training seeds per region (memory, language, reason), three read-out seeds each, item-quantised thresholds (`k* = ceil(2.0 * n / 100)`), an absolute token-read-out clause (`A`) alongside the delta clause (`D`), and gates (a) and (b) demoted to reported-not-gated (`docs/design/evidence/prereg-rank-gate-4000-2026-09-06/PREREG.md`, PR #69, adversarial review rg-review3 applied) | pre-registered, not run; adopting it is the substance of OD-17 | 3.9 (12 training runs + 12 probes + 2 pre-flight, per the pre-registration's own Table 5) |
| visual receipts and geometry | lane E | eval and quant receipts for the visual cells in the matrix dir with geometry fields; plan search re-run against geometry once the floor policy is decided | the motivating evidence is merged (`docs/design/evidence/visual-ptq-sensitivity-2026-09-06/`): every linear-probe read-out (pooled EuroSAT, Fashion-t10k transfer, a new pre-pool token probe) stays flat (max spread 0.0028-0.0070) across the whole 3-to-8-bit PTQ ladder, while the encoder's own output geometry moves an order of magnitude more — mean per-image cosine to the fp32 latent falls from 0.999972 (8-bit) to 0.952436 (3-bit), top-10 neighbour identity agreement from 0.996259 to 0.908500; the verdict is "the probe is insensitive, not the encoder." `quant.geometry.*` (lane E, PR #77) is the fix but is not yet merged, and the visual floor policy remains an operator decision (see Decisions) | 0.3 |
| E2 reason batch ablation | E0 | epoch-matched b256 vs b512, 2 step counts, 3 seeds; settles whether the b256 result is an epoch effect | **done, merged** (`docs/design/evidence/reason-e2-2026-09-06/`, PR #71). H2 (b256>b512 is really an epoch effect) is **NOT_CONFIRMED** in any seed — `(512, 2000)` never matches `(256, 4000)` and never beats `(512, 4000)`. H6 ((512, 2000) trails (256, 4000) on r@10 by more than 0.05) is **CONFIRMED**, by 2.2-3.8x the margin in every seed. But the trainer's learning rate covaries with batch (`lr = 3e-4 * sqrt(batch / 256)`: every b256 arm trains at 3.0000e-4, every b512 arm at 4.2426e-4), so the finding is batch-and-lr joint, not an isolated batch effect. Licensed follow-ups: an E3 temperature arm (tau = 0.05 vs 0.10) and an lr-controlled arm (b512 at a fixed 3.0e-4, same three seeds and split) | 0.9 |

Table: wave 2 lanes.

**Reason region objective status.** After g48-E1 (KILL, is any checkpoint step-sensitive) and E5
(KILL, `feat/reason-e5-latent-step`, PR #76 — the sequence-blind control beat the K=1 latent-step
predictor in all three seeds, margins -0.0124/-0.0440/-0.0234, clearing the kill threshold by
0.062-0.094), **the reason region has no validated step-sensitive training objective.** E5's own
battery carried a defect, found during adversarial verification and documented at
`docs/design/evidence/g-e5-reason-latent-step-2026-09-06/BATTERY-DEFECT.md`: the true-target slot
was restricted to steps carrying a calculator annotation while the three distractors were drawn
from every step of every other item, driving the untrained predictor below the nominal 0.20 chance
in every seed (0.1458 / 0.1224 / 0.0495), and a plain TF-IDF baseline scored 0.9367 recall@1 on the
identical battery — an unreported lexical ceiling far above every trained or untrained model arm
(0.05-0.18). The defect does not change the verdict: the graded quantity is a paired within-seed
margin on identical items, and a population-level skew shared by both arms cannot produce a
spurious margin of that size. Any successor experiment that reuses `build_step_battery` must first
draw distractors from the same population as the true target and add an E1-style
instrument-solvability control. The next step is a pre-registered redesign of the reasoning
objective around the latent-transform-loop reading (DEC-47 — reasoning as the workspace's iteration
depth, not a lookup) rather than another lookup proxy.

## Wave 3 (composed model build; smoke-scale GPU) — under way

The interconnect module is being built lane by lane against `docs/design/INTERCONNECT-MODULE-SPEC.md`,
each lane on its own branch, none yet merged: IC-1 (`Schedule` dataclasses and the G30 validator),
IC-2 (`RegionAdapter`, `TopKSelect`, `ConditioningPrefix`, `KVBank`, `StoreProjection`), IC-3
(workspace blocks and attention-mass export), IC-4 (frontal read-out and ranking), IC-6 (episodic
store E0 interface and stub, DEC-63..66 gaps recorded), IC-10 (the `compose` receipt builder and a
`schedule` pipeline stage), IC-11 (the `MindSpec` check replaced for workspace minds). The spec's
own merge condition governs: **a guard-skeptic pass on every guard's failing case (IC-R1,
`INTERCONNECT-MODULE-SPEC.md:498`) is required before any guard-carrying lane (IC-9 and what
depends on it) merges**, and a threat-modeler pass on the built validator and store path (IC-R2)
is required before W5's first GPU run and before OD-17's answer is spent.

| lane | depends on | what |
|---|---|---|
| interconnect build | lane F spec, lane A protocol | controller, adapters, K/V bank, workspace, read-out, schedule emission; unit + integration + guard-can-fail tests; CPU smoke; adversarial review (IC-R1, IC-R2) before merge |
| episodic store | operator rules on the two remaining contract gaps (scope axis, capacity — see Decisions) | build and probe (E0/E1 of the episodic rows); IC-6 already carries the interface and stub |
| compose stage | interconnect build | `compose` stage in the trainer and receipt writer, composed eval writer, composed card; IC-10 already carries the receipt builder and the pipeline stage |
| token-aware retrains W1b/W7a (reason, language) and W7v (visual, OD-4) | rank-gate replacement (pre-registered, not yet adopted or run — OD-17), lane A | mandatory before the W2b freeze; 2 seeds each, identical seeds across arms. The reason leg of W1b has no objective to retrain against until the reasoning-objective redesign (see Wave 2's reason status note) lands |

Table: wave 3 lanes.

## Wave 4 (composed training; 4 to 12 GPU-hours, unmeasured)

W2b freeze, W5 phases A to D with regions frozen, W5b write-back gate (at most 1 pt own-bin drop), W6 composed metric, then episodic E2 admission retrain and W8/W9/W10 (quantize, topology agreement at least 95 percent, final). Emitter-wired, one workflow, sequenced after wave 3 is green.

## Side tracks: parked until CSD is done

The ternary and sexagesimal (sx) side tracks are explicitly out of this plan's critical path
(taxonomy §6.9: ternary work does not start until the binary big model closes, and CSD is that
model). Their state as of this pass: `sx` merged its v1 hardening; `tritium-program` merged its
fixes and conformance vectors; two ternary fix PRs remain open and unmerged. Nothing here changes
until CSD's wave 4 lands.

## Corpus and parameters (the lever after the module set is complete)

Licence rulings first (code zero-delta, compress SynCSE swap, retrieve GooAQ tier, MATH vs PRM800K, composed repo tag), then the code six-language mix and max_len past the 93.9 percent truncation, reason's third provenance group, classify out-of-label-space rows, retrieve rebalance, then the per-faculty 1B data targets.

## Decisions the plan needs from the operator

| id | question | recommendation / status |
|---|---|---|
| OD-17 | adopt the pre-registered read-out gate at 4,000 steps (`docs/design/evidence/prereg-rank-gate-4000-2026-09-06/PREREG.md`, PR #69), or pivot | adopt; the 50-step 2.0x gate is dead (W4 control-arm KILL, control cleared it at 2.0191x with both token-aware terms off) and the replacement has been through one adversarial round (rg-review3) and is ready to run |
| floor | visual quant floor policy | measure per-tensor plan search against `quant.geometry.*` (lane E, PR #77) once merged; the merged PTQ-sensitivity evidence shows the pooled probe is blind to real geometry drift (0.999972 to 0.952436 mean cosine, 8-bit to 3-bit), so the floor cannot be chosen on probe scores alone |
| episodic scope | the store's partition axis beyond `domain` — authenticated principal, session bounded beneath principal, or persona/basin (taxonomy DEC-64, §8 gap (b)) | recommended default: principal, with session as a bounded sub-segment; still marked "OPERATOR TO CONFIRM" in the taxonomy |
| episodic capacity | the `capacity_bytes` formula's `safety_margin` value, and whether a residual-VRAM claim is acceptable at all versus a fixed floor reserved ahead of the KV budget (taxonomy DEC-63/DEC-70, §8 gap (a)) | recommended `safety_margin` ~2 GiB; still marked "OPERATOR TO CONFIRM" |
| licences | five rulings: code zero-delta, compress SynCSE swap, retrieve GooAQ/NC tier, MATH vs PRM800K DMCA conflict, composed repo licence tag | rule per region; the composed licence inherits the strictest tier (today NC, via GooAQ — taxonomy DEC-31) |

Table: open decisions. Settled since the last plan and dropped from this table: the workspace
latent shape (`z ∈ [B,64,512]`, `n_iter` 4 — taxonomy §2.3), the episodic store's write policy (a
cross-attention read of stored latents, not prompt injection — DEC-49 clause 5) and payload (an
index over runtime-owned latents, not a text-plus-embedding store — DEC-65, §8 gap (c)),
consolidation (prune-only at v1, learned consolidation deferred to phase 3 — DEC-66, §8 gaps
(d)/(e)), and affect's workspace placement (a tagged `z_affect` stream the decision integrator may
read as nuance, not a workspace participant at v1 by default — DEC-79). OD-4 (W7v's 128px corpus
source) and the config-vs-trained stream/hidden-dim discrepancy remain genuinely open — neither is
settled — but are tracked in `STATE-VS-PLAN-2026-09-06.md` (§8 discrepancy 6, Table 12 gap 5)
rather than repeated here, since neither currently blocks a wave in this plan the way OD-17 does.

## Standing rules applied

Identical seeds and fixed splits across arms; harness-shaped smoke before any stage script is pushed; the repo's own gate before every commit; readability on every document; GPU-first; workflows and agents execute, the orchestrator plans and verifies by execution.
