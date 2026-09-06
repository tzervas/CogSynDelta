r"""E5: latent-step prediction for the `reason` region (diagnosis §4 E5).

WHY THIS EXISTS
g48's E1 KILLED the contrastive `reason` bi-encoder: every arm (b256-s0, b512-s0, the
untrained baseline) scored `derive.recall@1` at or below chance on the corrupted-
derivation battery (docs/design/evidence/g48-reason-e1-2026-09-06/README.md). The
symmetric-InfoNCE lookup objective the region trains today learned nothing about
derivation STRUCTURE -- it learned to retrieve a worked answer for a question, which is
what the diagnosis's H1 calls "the one region that has not learned the surface" once
TF-IDF (0.873) is put beside it. DEC-04's own words for the region: "it recognises
derivations, it does not produce them ... Phase 3 replaces the objective with
latent-step prediction so the region contributes COMPUTATION to the workspace rather
than a lookup" (TAX:815-821). E5 is that objective, run as a toy pre-registration
(diagnosis §4 E5) rather than a production retrain.

Operator's definition, restated because it is what E5 is FOR: reasoning is not lookup.
It is working a problem through steps. The mechanism is latent transform loops coupled
to attention and the memory gate (`[OP: csd-recursive-latent-transformers]`). E5 is the
toy entry to that mechanism: K=1 of the eventual loop (predict one step ahead; feeding
the predicted latent back in as its own context, to predict t+2, is the deferred
recursive direction this module deliberately does NOT build -- see "SCOPE" below).

THE OBJECTIVE (diagnosis §4 E5, not invented here)
Split a gsm8k derivation into steps (lines). For every prefix length ``t`` (1-indexed,
i.e. steps ``[0, t)``):

    context_latent  = encoder(question + "\\n" + "\\n".join(steps[:t]))
    predicted       = predictor(context_latent)
    target_latent   = target_encoder(steps[t])                      # stop-gradient
    loss            = smooth_l1(predicted, layer_norm(target_latent))

``encoder`` is `TextEncoder` -- the SAME architecture and weights the `reason` bi-
encoder already trains, so E5 is a different OBJECTIVE on the same trunk, not a new
architecture (the diagnosis's "the only change is the objective"). ``target_encoder``
is an EMA copy of ``encoder``, exactly the shape `cogsyndelta.model.vl_jepa.IJEPA`
already uses for images -- "the JEPA family `vl_pretrain.py` already uses" (diagnosis
§4 E5) -- transplanted to a pooled `[B, D]` text latent instead of per-patch `[B, M, D]`
visual latents, since a step's representation here is one pooled vector, not a grid.

LOSS AND STOP-GRADIENT CHOICES (stated once, here, per the operator's requirement)
- **Loss: smooth L1 regression**, not InfoNCE. `IJEPA.forward` (vl_jepa.py) uses
  `F.smooth_l1_loss(predicted, targets)` for exactly this shape (predict a
  representation, do not classify it against in-batch negatives) and this module
  matches it rather than inventing a second predict-a-vector loss. InfoNCE was
  considered and rejected for one concrete reason: a step window's "batch" is windows
  drawn from however many eligible gsm8k derivations land in one physical batch (often
  fewer than 256, since only ~63% of `reason`'s train pairs are gsm8k-shaped and not
  every one has >= 2 steps), so the in-batch negative count would vary step to step in
  a way the trunk's existing InfoNCE never has to tolerate; smooth L1 has no such
  dependency on batch composition.
- **EMA target encoder** (`update_target`, ramped `ema_base -> ema_final` exactly as
  `vl_pretrain.py`'s `_ema_at` ramps `JEPAConfig.ema_base/ema_final`): targets never
  come from the same weights the predictor is trained to match, or the trivial optimum
  is a constant (JEPA's collapse route 1, vl_jepa.py's module docstring).
- **`torch.no_grad()` AND `.detach()` on the target path, never one alone**: mirrors
  `IJEPA.forward`'s own target block byte-for-byte in intent. `update_target` additionally
  runs the EMA update itself under `@torch.no_grad()`, so the target encoder's
  parameters never once appear on the autograd graph, not even transiently.
- **Target latents are layer-normalized before the loss** (`F.layer_norm`, matching
  `IJEPA.forward`): stops the predictor minimising loss by shrinking the target's scale
  rather than predicting its direction, exactly `IJEPA`'s stated reason for the same
  line.
- **Asymmetric predictor** (`StepPredictor`'s `hidden < dim` by default): the JEPA
  family's third collapse guard (`JEPAPredictor`'s narrower `predictor_dim`) --
  "an equally wide predictor lets the encoder settle on identity" (vl_jepa.py). Applied
  here even though context and target texts differ (question+prefix vs. one step,
  unlike vl_jepa's same-image two crops), because the argument for the width asymmetry
  is about the PREDICTOR's own capacity to shortcut, not about how similar the two
  inputs happen to be.

SCOPE (diagnosis §4 E5's own boundary, restated so nobody reads more into this module)
K = 1 only. Feeding the predictor's own output back in to predict `t+2` from a
*predicted* `t+1` (accuracy over `K` predicted steps) is "the deferred loop-in-latent-
space direction" (`TAX:4654-4660`) and is explicitly NOT built here -- this module
declares the seam (`LatentStepModel.predict` returns one step's prediction and nothing
recurses on it) and stops. No change to region architecture, no interconnect change,
no token stream leaving the region (DEC-47): everything in this module operates on the
`reason` trunk's own pooled `[B, D]` latents.

THE SEQUENCE-BLIND CONTROL
`enumerate_step_windows(..., blind=True)` builds the identical target side (predict
`steps[t]`) from a context that is the QUESTION ALONE, discarding `steps[:t]` entirely.
This is W1d's pattern, required because "the g14 toy learned a K-independent shortcut
and its falsifier fired" (`[OP: csd-recursive-latent-transformers]`, diagnosis §4 E5):
a predictor that scores well without ever reading the steps has learned "what answer
follows this kind of question", not derivation structure, and the go/kill rule below
is defined entirely in terms of beating this control, not in terms of the raw score.

THE BATTERY
`build_step_battery` ranks the true `steps[t]` latent among 5 candidates -- true, one
corrupted step (`corrupt_step`, a local, per-step analogue of
`cogsyndelta.eval.corrupted_derivation.corrupt_derivation` that edits one calculator
annotation inside a single step line rather than a whole derivation), and three steps
drawn from OTHER derivations -- chance 0.20, via `score_step_battery`. Ranking is by
cosine similarity between the predictor's OUTPUT (not a second encoder call) and each
candidate's target-encoder latent, `TARGET`-encoder for every candidate including the
true one -- the same "both sides through the same network" discipline
`cogsyndelta.eval.corrupted_derivation.cosine_hit_rate` already applies to the bi-
encoder battery.

GO / KILL (diagnosis §4 E5, verbatim; enforced by a later cross-run comparison, not by
this module -- see `scripts/csd-train-reason-e5.py`'s module docstring for why a single
training run cannot itself decide this)
    Go:   predictor acc@1 >= 0.40 AND >= sequence-blind + 0.10, in every seed.
    Kill: predictor acc@1 <= sequence-blind + 0.05, in any seed.

CONTROLS RECORDED BY THE TRAINING SCRIPT, NOT THIS MODULE
E1's corrupted-derivation battery (regression guard, scored on the target encoder
through the EXISTING E1 scorer -- `cogsyndelta.eval.corrupted_derivation` -- so E5 does
not reimplement it), the diagonal recall@1 (reference only, already demoted from a gate
by E1), an untrained-predictor baseline at the region-specific seed, and the collapse
guards (`emb_std`, `effective_rank`) this module's `collapse_stats` computes.
"""

from __future__ import annotations

import copy
import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cogsyndelta.eval.corrupted_derivation import is_gsm8k_derivation, parse_calculator_annotations
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig
from cogsyndelta.splits import item_id as split_item_id

EXPERIMENT_ID = "reason-e5-latent-step"
BATTERY_ID = "eval_latent_step_prediction"
MIN_STEPS = 2
"""A window needs at least one prior step and one target step."""
MIN_HOLDOUT_STEPS = 3
"""Diagnosis §2: "all held-out items >= 3 [C]" -- the battery draws from this subset."""
K_DISTRACTORS = 4
CHANCE = 1.0 / (1 + K_DISTRACTORS)
GO_RECALL_FLOOR = 0.40
GO_MARGIN_OVER_BLIND = 0.10
KILL_MARGIN_OVER_BLIND = 0.05

_STEP_SPLIT_RE = re.compile(r"\n+")

PREREG_HYPOTHESIS = (
    "The reason region's contrastive bi-encoder (E1, g48) learned nothing about "
    "derivation structure -- every arm sat at or below chance (0.20) on the corrupted-"
    "derivation battery. DEC-04 names latent-step prediction (iterative refinement) as "
    "the phase-3 replacement objective, so the region contributes COMPUTATION to the "
    "workspace rather than a lookup. E5 pre-registers that objective as a toy: predict "
    "the latent of derivation step t+1 from the latents of steps <= t, on the SAME "
    "trunk, SAME init, and SAME order manifest as the reason baseline, with a "
    "sequence-blind control that predicts from the question alone (W1d's pattern, "
    "guarding against the K-independent shortcut g14 found). K = 1 only; recursive "
    "loop-in-latent-space (predict t+2 from a predicted t+1) is deferred, not built."
)
PREREG_GO = (
    f"predictor acc@1 >= {GO_RECALL_FLOOR:.2f} AND "
    f">= sequence-blind + {GO_MARGIN_OVER_BLIND:.2f}, in every seed"
)
PREREG_KILL = f"predictor acc@1 <= sequence-blind + {KILL_MARGIN_OVER_BLIND:.2f}, in any seed"
PREREG_CONTROLS = (
    "E3's contrastive encoder on the same trunk (reference: E1 g48 receipts); "
    "an untrained predictor at the region-specific seed (scored before training starts); "
    "collapse guards (target-encoder emb_std, effective_rank over the holdout); "
    "the E1 corrupted-derivation battery via the existing E1 scorer, as a regression "
    "guard on the trunk's diagonal derivation sensitivity; "
    "the plain diagonal recall@1, for reference only (already demoted from a gate by E1)."
)
PREREG_BATTERY = (
    "Rank the true step-(t+1) latent among {true, 1 corrupted step, 3 steps from other "
    "problems} by the predictor's output, via cosine similarity against each "
    f"candidate's TARGET-encoder latent. Chance = 1/5 = {CHANCE:.2f}."
)
PREREG_SCOPE_NOTE = (
    "K = 1 only. Feeding the predictor's own output back in to predict t+2 is the "
    "deferred recursive-loop-in-latent-space direction (TAX:4654-4660) and is not "
    "built or measured here."
)


def w2c_region_seed(region: str) -> int:
    """W2c region-specific untrained-encoder seed (mirrors `corrupted_derivation`'s).

    Duplicated rather than imported: it is a one-line pure function and importing it
    would couple this module to `corrupted_derivation`'s private surface for no benefit
    -- both copies are asserted equal in `tests/test_reason_latent_step.py`.

    Args:
        region: Region id (``reason``).

    Returns:
        First 8 hex chars of ``sha256("csd-w2c-untrained:<region>")`` as uint32.
    """
    digest = hashlib.sha256(f"csd-w2c-untrained:{region}".encode()).hexdigest()
    return int(digest[:8], 16)


# --------------------------------------------------------------------------- steps


def split_derivation_steps(text: str) -> list[str]:
    """Split a gsm8k derivation into non-empty, stripped lines ("steps").

    A step is one line of the worked solution, calculator annotations and the trailing
    ``#### <answer>`` line included -- the line IS the unit DEC-04's "steps of a
    derivation" refers to, and the ``####`` line is itself a real step (the concluding
    one), not metadata to strip. Blank lines (gsm8k derivations sometimes carry a
    trailing newline) are dropped so they never become a zero-content "step".

    Args:
        text: Raw derivation string (the pair's positive side).

    Returns:
        Steps in original order, at least 0 of them.
    """
    return [line.strip() for line in _STEP_SPLIT_RE.split(text.strip()) if line.strip()]


@dataclass(frozen=True)
class StepExample:
    """One gsm8k derivation, pre-split into steps, eligible for step windows."""

    item_id: str
    question: str
    steps: tuple[str, ...]


def build_step_examples(
    pairs: list[tuple[str, str]], *, min_steps: int = MIN_STEPS
) -> list[StepExample]:
    """Filter a pair list to gsm8k derivations with >= `min_steps` steps.

    Args:
        pairs: (question, positive) pairs, e.g. `reason`'s train_pairs or holdout.
        min_steps: Eligibility floor. `MIN_STEPS` (2) for training (need one window);
            callers building the held-out battery pass `MIN_HOLDOUT_STEPS` (3), matching
            the diagnosis's "all held-out items >= 3".

    Returns:
        One `StepExample` per eligible pair, order preserved.
    """
    out: list[StepExample] = []
    for question, positive in pairs:
        if not is_gsm8k_derivation(positive):
            continue
        steps = split_derivation_steps(positive)
        if len(steps) < min_steps:
            continue
        out.append(
            StepExample(
                item_id=split_item_id(question, positive), question=question, steps=tuple(steps)
            )
        )
    return out


@dataclass(frozen=True)
class StepWindow:
    """One (context, target) training instance: predict `steps[t]` from `steps[:t]`."""

    item_id: str
    t: int
    """Index of the TARGET step (0-indexed); context is `steps[:t]`, so `t >= 1`."""
    context: str
    target: str


def _join_context(question: str, prefix_steps: tuple[str, ...]) -> str:
    return "\n".join((question, *prefix_steps))


def enumerate_step_windows(examples: list[StepExample], *, blind: bool = False) -> list[StepWindow]:
    r"""Every valid (steps[:t] -> steps[t]) window across `examples`.

    Deterministic and exhaustive -- every eligible prefix length of every example
    becomes one window, rather than sampling one window per example at random. This
    keeps a physical batch's training signal a pure function of WHICH derivations
    landed in it (which the shared order manifest already fixes), with no additional
    RNG draw this module would have to seed and justify.

    Args:
        examples: Step-split derivations (`build_step_examples`).
        blind: When True, `context` is the question ALONE for every window (the
            sequence-blind control) -- `steps[:t]` is computed only to select which
            `t` values are valid and which step is the target; it never appears in the
            returned context string. When False (the latent-step arm), `context` is
            `question + "\\n" + "\\n".join(steps[:t])`.

    Returns:
        Windows in example order, then increasing `t` within an example.
    """
    windows: list[StepWindow] = []
    for ex in examples:
        n = len(ex.steps)
        for t in range(1, n):
            context = ex.question if blind else _join_context(ex.question, ex.steps[:t])
            windows.append(StepWindow(item_id=ex.item_id, t=t, context=context, target=ex.steps[t]))
    return windows


# ----------------------------------------------------------------- step corruption


def _item_rng(seed: int, item_id: str, t: int, purpose: str) -> random.Random:
    payload = f"{seed}:{purpose}:{item_id}:{t}".encode()
    digest = hashlib.sha256(payload).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))  # noqa: S311


def _delta_spellings_for_step(original: str) -> list[str]:
    """Numeric neighbours of `original`, spelled to match its own decimal places.

    A smaller, self-contained rewrite of `corrupted_derivation._delta_spellings` scoped
    to one step line: this module never needs `####`-line propagation (that is a
    whole-DERIVATION concern; a step-level corruption only ever touches the one line it
    was drawn from), so it does not import that module's underscore-prefixed helper.
    """
    try:
        value = float(original)
    except ValueError:
        return []
    is_int = re.fullmatch(r"[+-]?\d+", original) is not None
    if is_int:
        base = round(value)
        deltas = (1, -1, 2, -2, 3, -3, 5, -5, 10, -10)
        candidates = [float(base + d) for d in deltas]
        spell = lambda v: str(round(v))  # noqa: E731
    else:
        places = len(original.split(".", 1)[1]) if "." in original else 1
        step = 10.0 ** (-places)
        candidates = [value + d * step for d in (1, -1, 2, -2, 5, -5, 10, -10)]
        spell = lambda v: f"{v:.{places}f}"  # noqa: E731
    seen = {original}
    out: list[str] = []
    for cand in candidates:
        spelled = spell(cand)
        if spelled not in seen:
            seen.add(spelled)
            out.append(spelled)
    return out


def corrupt_step(step_text: str, *, item_id: str, t: int, corruption_seed: int = 0) -> str | None:
    """Edit one number inside one calculator annotation of a single step line.

    Prefers editing a RESULT (matches `corrupted_derivation.corrupt_derivation`'s own
    preference order); falls back to an operand if no result edit is available.

    Args:
        step_text: One derivation line.
        item_id: The owning derivation's split item id, mixed into the per-step RNG so
            two different derivations that happen to share a step string still get
            independent corruptions.
        t: Step index, also mixed into the RNG (two steps of the same derivation must
            not collide on the same edit).
        corruption_seed: Battery seed (not a CSD training seed).

    Returns:
        A corrupted line distinct from `step_text`, or `None` when the line carries no
        calculator annotation with a usable numeric edit -- callers must exclude such
        steps from the battery rather than fabricate a no-op "corruption".
    """
    annotations = parse_calculator_annotations(step_text)
    if not annotations:
        return None
    rng = _item_rng(corruption_seed, item_id, t, "corrupt_step")
    slots = [ann.result_span for ann in annotations] + [
        span for ann in annotations for span in ann.operand_spans
    ]
    rng.shuffle(slots)
    for start, end, original in slots:
        for spelling in _delta_spellings_for_step(original):
            candidate = step_text[:start] + spelling + step_text[end:]
            if candidate != step_text:
                return candidate
    return None


@dataclass(frozen=True)
class StepBatteryItem:
    """One E5 battery item: predict `target`, rank it among 4 distractors."""

    item_id: str
    t: int
    context: str
    target: str
    corrupted: str
    others: tuple[str, str, str]


def build_step_battery(
    examples: list[StepExample], *, corruption_seed: int = 0
) -> list[StepBatteryItem]:
    """Build the E5 ranking battery from step-split held-out examples.

    One item per eligible (example, t) window whose target step carries a corruptible
    calculator annotation. `min_steps=MIN_HOLDOUT_STEPS` should already have been
    applied by the caller via `build_step_examples`; this function additionally drops
    any window whose target step has no annotation to corrupt (`corrupt_step` returns
    `None`), same discipline as `corrupted_derivation.build_corrupted_battery`'s
    `min_annotations` filter -- a battery item with no valid corruption cannot measure
    what it claims to.

    Args:
        examples: Step-split derivations, already floored at `MIN_HOLDOUT_STEPS`.
        corruption_seed: Deterministic corruption/distractor-draw seed.

    Returns:
        Battery items, deterministic given `examples` and `corruption_seed`.
    """
    all_steps: list[tuple[str, int, str]] = [
        (ex.item_id, t, step) for ex in examples for t, step in enumerate(ex.steps)
    ]
    items: list[StepBatteryItem] = []
    for ex in examples:
        n = len(ex.steps)
        for t in range(1, n):
            target = ex.steps[t]
            corrupted = corrupt_step(
                target, item_id=ex.item_id, t=t, corruption_seed=corruption_seed
            )
            if corrupted is None:
                continue
            pool = [s for owner, _, s in all_steps if owner != ex.item_id]
            if len(pool) < 3:
                continue
            rng = _item_rng(corruption_seed, ex.item_id, t, "others")
            others = tuple(rng.sample(pool, 3))
            context = _join_context(ex.question, ex.steps[:t])
            items.append(
                StepBatteryItem(
                    item_id=ex.item_id,
                    t=t,
                    context=context,
                    target=target,
                    corrupted=corrupted,
                    others=others,  # type: ignore[arg-type]
                )
            )
    return items


def battery_fingerprint(items: list[StepBatteryItem]) -> str:
    """Sha256 of item ids, `t`, and candidate texts: membership of the scored battery.

    Args:
        items: E5 battery items.

    Returns:
        64-char hex digest.
    """
    lines: list[str] = []
    for item in items:
        lines.append(f"{item.item_id}:{item.t}")
        lines.append(item.corrupted)
        lines.extend(item.others)
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- model


@dataclass
class LatentStepConfig:
    """Shape of the E5 predictor stack. The trunk's own shape is `TextEncoderConfig`."""

    dim: int
    """Trunk `out_dim` -- the predictor's input and output width. Not independently
    settable: `LatentStepModel.__init__` reads it off the trunk config it is given."""
    hidden: int = 128
    """Predictor width. Narrower than `dim` by default (asymmetric predictor, see the
    module docstring's collapse-guard rationale)."""
    depth: int = 2
    """Number of hidden Linear+GELU blocks between `proj_in` and `proj_out`."""
    ema_base: float = 0.996
    ema_final: float = 1.0
    """Same defaults as `cogsyndelta.model.vl_jepa.JEPAConfig` -- this is the same
    family's EMA ramp, not a new schedule invented for text."""

    def __post_init__(self) -> None:
        """Reject a config that cannot build a well-formed predictor."""
        if self.hidden <= 0 or self.depth < 0:
            raise ValueError(
                f"hidden={self.hidden} must be > 0 and depth={self.depth} must be >= 0"
            )


class StepPredictor(nn.Module):
    """Small MLP: context latent -> predicted next-step latent.

    Deliberately an MLP, not a transformer: the input is already one pooled `[B, D]`
    vector (the trunk has already done the attention over the context text), so there
    is no sequence left for a transformer block to attend over. "Small predictor" in
    the diagnosis's own words (§4 E5) -- this is the plainest shape that satisfies the
    asymmetric-width collapse guard.
    """

    def __init__(self, cfg: LatentStepConfig) -> None:
        """Build proj_in, `cfg.depth` hidden blocks, and proj_out."""
        super().__init__()
        self.proj_in = nn.Linear(cfg.dim, cfg.hidden, bias=False)
        blocks: list[nn.Module] = []
        for _ in range(cfg.depth):
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, cfg.hidden), nn.GELU()
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(cfg.hidden)
        self.proj_out = nn.Linear(cfg.hidden, cfg.dim, bias=False)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """`[B, dim] -> [B, dim]`, the predicted next-step latent.

        Args:
            context: Trunk's pooled context-side latent.

        Returns:
            Predicted target-side latent, same shape.
        """
        h = self.proj_in(context)
        for block in self.blocks:
            h = h + block(h)
        return self.proj_out(self.norm(h))


class LatentStepModel(nn.Module):
    """Context trunk + EMA target trunk + `StepPredictor`. See module docstring."""

    def __init__(self, encoder_cfg: TextEncoderConfig, predictor_cfg: LatentStepConfig) -> None:
        """Build the context encoder, its frozen EMA copy, and the predictor.

        Args:
            encoder_cfg: The `reason` trunk's own config (dim, depth, n_heads, ...).
            predictor_cfg: Must have `dim == (encoder_cfg.out_dim or encoder_cfg.dim)`;
                raises otherwise, rather than silently building a shape-mismatched
                predictor that only fails on the first forward pass.
        """
        super().__init__()
        trunk_dim = encoder_cfg.out_dim or encoder_cfg.dim
        if predictor_cfg.dim != trunk_dim:
            raise ValueError(
                f"LatentStepConfig.dim={predictor_cfg.dim} must equal the trunk's own "
                f"out_dim={trunk_dim}"
            )
        self.encoder_cfg = encoder_cfg
        self.predictor_cfg = predictor_cfg
        self.encoder = TextEncoder(encoder_cfg, name="reason-e5-context")
        # A deep copy, not a reference -- the target must lag the context encoder,
        # exactly IJEPA's own comment on the identical line (vl_jepa.py).
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.predictor = StepPredictor(predictor_cfg)

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        """EMA-update the target encoder. Under no_grad and in-place, like `IJEPA`'s.

        Args:
            momentum: `m` in `target <- m*target + (1-m)*context.detach()`.
        """
        for tgt, src in zip(
            self.target_encoder.parameters(), self.encoder.parameters(), strict=True
        ):
            tgt.mul_(momentum).add_(src.detach(), alpha=1.0 - momentum)
        for tgt_buf, src_buf in zip(
            self.target_encoder.buffers(), self.encoder.buffers(), strict=True
        ):
            tgt_buf.copy_(src_buf)

    def encode_context(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Context-side pooled latent, WITH gradient (this is what trains)."""
        return self.encoder(ids, mask)

    @torch.no_grad()
    def encode_target(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Target-side pooled latent, frozen. `no_grad` is the whole point."""
        return self.target_encoder(ids, mask)

    def predict(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Context tokens -> predicted next-step latent (no target involved).

        This is the ONE step of the eventual recursive loop this toy measures (K = 1);
        it deliberately does not consume or produce anything the caller could feed back
        in as a new context -- see the module docstring's "SCOPE".
        """
        return self.predictor(self.encode_context(ids, mask))

    def forward(
        self,
        ctx_ids: torch.Tensor,
        ctx_mask: torch.Tensor,
        tgt_ids: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """One E5 step: predict the next-step latent, loss against the EMA target.

        Args:
            ctx_ids: Tokenized context ids (`question [+ prior steps]`).
            ctx_mask: Attention mask for `ctx_ids`.
            tgt_ids: Tokenized target-step ids.
            tgt_mask: Attention mask for `tgt_ids`.

        Returns:
            `(loss, stats)`. `stats["target_emb_std"]` near zero is the collapse
            signal, matching every other region's receipt convention.
        """
        predicted = self.predict(ctx_ids, ctx_mask)
        with torch.no_grad():
            target = self.encode_target(tgt_ids, tgt_mask)
            # Layer-normalized before the loss -- see module docstring.
            target = F.layer_norm(target, (target.size(-1),))
        loss = F.smooth_l1_loss(predicted, target)
        with torch.no_grad():
            stats = {
                "loss": loss.item(),
                "target_emb_std": target.std(dim=0).mean().item(),
                "predicted_emb_std": predicted.detach().std(dim=0).mean().item(),
            }
        return loss, stats


def ema_at(step: int, total_steps: int, ema_base: float, ema_final: float) -> float:
    """Linear ramp `ema_base -> ema_final` over the run, matching `vl_pretrain._ema_at`.

    Args:
        step: Current step (0-indexed).
        total_steps: Total optimizer steps.
        ema_base: Momentum at step 0.
        ema_final: Momentum at the last step.

    Returns:
        Momentum for this step.
    """
    p = min(1.0, step / max(1, total_steps))
    return ema_base + (ema_final - ema_base) * p


# ------------------------------------------------------------------------- battery scoring


def collapse_stats(embeddings: torch.Tensor) -> dict[str, float]:
    """`emb_std` and `effective_rank` of a `[N, D]` embedding matrix.

    Args:
        embeddings: Target-encoder latents over some fixed set (e.g. the holdout).

    Returns:
        `emb_std`, `effective_rank`, `n`, `dim`.
    """
    from cogsyndelta.eval.benchmark import effective_rank

    x = embeddings.detach().float()
    return {
        "emb_std": x.std(dim=0).mean().item(),
        "effective_rank": effective_rank(x),
        "n": float(x.size(0)),
        "dim": float(x.size(1)) if x.dim() > 1 else 0.0,
    }


def _hit_rate_at_1(scores: np.ndarray, *, tie_seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(tie_seed)
    jitter = rng.uniform(0.0, 1e-12, size=scores.shape)
    ranked = np.argmax(scores + jitter, axis=1)
    recall = float((ranked == 0).mean()) if scores.size else 0.0
    order = np.argsort(-(scores + jitter), axis=1)
    ranks = np.argmax(order == 0, axis=1) + 1
    mrr = float((1.0 / ranks).mean()) if scores.size else 0.0
    return recall, mrr


def score_step_battery(
    model: LatentStepModel,
    tokenize: Any,
    items: list[StepBatteryItem],
    *,
    batch: int = 64,
    tie_seed: int = 0,
) -> dict[str, float]:
    """Rank the true target among `{true, corrupted, 3 others}` by predictor output.

    Both the predicted latent and every candidate's latent go through the SAME network
    (the target encoder for candidates, the predictor-over-context for the query) --
    the "both sides through the same network" discipline `corrupted_derivation`'s own
    battery scorer applies, so a bug that makes one side's representation systematically
    larger or smaller cannot masquerade as ranking signal.

    Args:
        model: Trained (or untrained) `LatentStepModel`, already on the target device.
        tokenize: `(texts: list[str]) -> (ids, mask)`, bound to a tokenizer/max_len/
            device by the caller.
        items: Battery items (`build_step_battery`).
        batch: Encoding chunk size.
        tie_seed: Deterministic tie-break seed.

    Returns:
        `recall@1`, `mrr`, `n_items`, `chance`.
    """
    was_training = model.training
    model.eval()
    contexts = [it.context for it in items]
    candidates: list[str] = []
    for it in items:
        candidates.extend((it.target, it.corrupted, *it.others))

    def encode_chunks(texts: list[str], fn: Any) -> torch.Tensor:
        """Tokenize and encode `texts` through `fn` in `batch`-sized chunks, L2-normalised.

        Args:
            texts: Strings to encode.
            fn: `(ids, mask) -> [n, D]`, e.g. `model.predict` or `model.encode_target`.

        Returns:
            `[len(texts), D]`, or `[0, 0]` when `texts` is empty.
        """
        chunks = []
        for i in range(0, len(texts), batch):
            ids, mask = tokenize(texts[i : i + batch])
            chunks.append(F.normalize(fn(ids, mask), dim=-1))
        if not chunks:
            return torch.zeros(0, 0)
        return torch.cat(chunks)

    with torch.no_grad():
        query = encode_chunks(contexts, lambda ids, mask: model.predict(ids, mask))
        cand = encode_chunks(candidates, model.encode_target)
    if was_training:
        model.train()

    n_items = len(items)
    n_cand = 1 + K_DISTRACTORS
    q = query.cpu().numpy().astype(np.float64)
    c = (
        cand.cpu().numpy().astype(np.float64).reshape(n_items, n_cand, -1)
        if n_items
        else cand.cpu().numpy()
    )
    scores = np.einsum("nd,ncd->nc", q, c) if n_items else np.zeros((0, n_cand))
    recall, mrr = _hit_rate_at_1(scores, tie_seed=tie_seed)
    return {"recall@1": recall, "mrr": mrr, "n_items": float(n_items), "chance": CHANCE}


def go_kill_note(this_arm_recall: float, blind_recall: float | None) -> dict[str, Any]:
    """Apply the pre-registered go/kill rule for ONE seed, given its paired blind score.

    A single training run only ever has half the pair (see the training script's
    docstring for why); pass `blind_recall=None` when the paired run has not been
    scored yet, and this returns `"verdict": "pending"` rather than guessing.

    Args:
        this_arm_recall: `derive_step.recall@1` of the latent-step arm.
        blind_recall: The SAME seed's sequence-blind arm `recall@1`, or `None`.

    Returns:
        `verdict` (`"go"`, `"kill"`, `"none"`, or `"pending"`), the two inputs, and the
        margins actually observed.
    """
    if blind_recall is None:
        return {
            "verdict": "pending",
            "this_arm_recall@1": this_arm_recall,
            "blind_recall@1": None,
            "note": "paired sequence-blind run not yet scored",
        }
    margin = this_arm_recall - blind_recall
    if this_arm_recall >= GO_RECALL_FLOOR and margin >= GO_MARGIN_OVER_BLIND:
        verdict = "go"
    elif margin <= KILL_MARGIN_OVER_BLIND:
        verdict = "kill"
    else:
        verdict = "none"
    return {
        "verdict": verdict,
        "this_arm_recall@1": this_arm_recall,
        "blind_recall@1": blind_recall,
        "margin_over_blind": margin,
        "go_floor": GO_RECALL_FLOOR,
        "go_margin": GO_MARGIN_OVER_BLIND,
        "kill_margin": KILL_MARGIN_OVER_BLIND,
    }
