# E5 pre-registration (verbatim)

This is the E5 pre-registration as stated in-repo before any of the six runs graded
here were scored, quoted verbatim from its sources with citations. Nothing here was
written after seeing results.

## Sources

- `src/cogsyndelta/regions/reason_latent_step.py:1-105` (module docstring, "THE
  OBJECTIVE", "THE SEQUENCE-BLIND CONTROL", "THE BATTERY", "GO / KILL" sections).
- `src/cogsyndelta/regions/reason_latent_step.py:141-163` (`GO_RECALL_FLOOR`,
  `GO_MARGIN_OVER_BLIND`, `KILL_MARGIN_OVER_BLIND` constants and their `PREREG_GO`/
  `PREREG_KILL` string forms).
- `scripts/csd-train-reason-e5.py:1-45` (trainer module docstring, "WHAT THIS RUNS",
  "WHY THIS SCRIPT NEVER APPLIES THE GO/KILL RULE ITSELF", "PHASE 2").
- `docs/design/evidence/reason-region-diagnosis-2026-09-06/README.md` §E5 (diagnosis
  origin of the objective; not reproduced in full here, cited for provenance).

## Objective

Split a gsm8k derivation into steps (lines). For prefix length `t` (1-indexed, steps
`[0, t)`):

```
context_latent  = encoder(question + "\n" + "\n".join(steps[:t]))
predicted       = predictor(context_latent)
target_latent   = target_encoder(steps[t])                      # stop-gradient
loss            = smooth_l1(predicted, layer_norm(target_latent))
```

`encoder` is the same `TextEncoder` architecture and weights the `reason` bi-encoder
already trains — E5 changes the objective on the same trunk, not the architecture.
`target_encoder` is an EMA copy of `encoder`, the same shape `IJEPA` already uses for
images, transplanted to a pooled `[B, D]` text latent (`reason_latent_step.py:32-38`).

K = 1 only: the predictor's own output is never fed back in to predict `t+2`. That
recursive direction is explicitly deferred and not built here
(`reason_latent_step.py:71-78`).

## Arms

Two arms, one seed per invocation (`csd-train-reason-e5.py:9-13`):

- `latent-step` — context = question + steps so far (the E5 objective).
- `sequence-blind` — context = question alone (W1d's shortcut-detection control;
  `reason_latent_step.py:80-87`).

## Seeds, steps, batch size

Seeds `{0, 1, 2}`; 4000 steps; batch size 256 — matched to the `reason` `b256-s0`
baseline (`csd-train-reason-e5.py:15-23`). Both arms load the same split manifest and
the same order manifest at these defaults; changing either flag resolves a different
order-manifest file and silently breaks comparability between arms
(`csd-train-reason-e5.py:16-23`).

## Metric

5-way step-battery `recall@1` (chance 0.20 = 1/(1+4 distractors)): rank the true
`steps[t]` latent among 5 candidates (true, one corrupted step, three steps from other
derivations) by cosine similarity between the predictor's output and each candidate's
target-encoder latent (`reason_latent_step.py:89-99`, `CHANCE = 1.0 / (1 + K_DISTRACTORS)`
at `reason_latent_step.py:140`).

## Go / kill rule (verbatim)

> Go: predictor acc@1 >= 0.40 AND >= sequence-blind + 0.10, in every seed.
> Kill: predictor acc@1 <= sequence-blind + 0.05, in any seed.

(`reason_latent_step.py:101-105`; constants `GO_RECALL_FLOOR = 0.40`,
`GO_MARGIN_OVER_BLIND = 0.10`, `KILL_MARGIN_OVER_BLIND = 0.05` at
`reason_latent_step.py:141-143`.)

## Why no single run can apply this rule

The rule is defined across a pair of runs at the same seed (an arm and its
sequence-blind control) and, for "go", across all three seeds at once. One invocation
of the trainer produces exactly one arm of one seed with no paired receipt to read yet,
so the trainer always calls `go_kill_note` with `blind_recall=None`, reporting
`"pending"` in its own receipt. Comparing all six receipts once produced is declared
future work, out of scope for the trainer itself (`csd-train-reason-e5.py:25-32`) — it
is exactly the comparison `grade_e5.py` in this directory performs.

## Controls recorded by the training script, not the objective module

E1's corrupted-derivation regression guard (scored through the existing E1 scorer),
diagonal recall@1 (reference only, already demoted from a gate by E1), an
untrained-predictor baseline at the region-specific seed, and collapse guards
(`emb_std`, `effective_rank`) (`reason_latent_step.py:107-113`).
