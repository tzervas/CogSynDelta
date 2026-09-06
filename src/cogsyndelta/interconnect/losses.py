"""Training losses -- lane IC-7 of the interconnect module.

WHAT THIS IS, IN THE SPEC'S OWN WORDS
`docs/design/INTERCONNECT-MODULE-SPEC.md` ("the spec" below) section 2.1 Table 2 assigns this
file four classes -- ``RankLoss``, ``UnifyLoss``, ``DistilLoss``, ``FlopsPenalty`` -- with
constructor arguments "the per-loss weights" and ownership "`L_task`, `L_unify`, `L_B`, the
`lambda_flops` term." Section 4 ("Loss definitions") is this file's formula source:

    L_task  = softmax cross-entropy over the k=32 cosine scores at a recorded temperature
    L_unify = Sum_r (1 - cos(probe_r(f), pool_r(h_r)))                over admitted regions
    L_B     = Sum_i Sum_r KL(softmax_r(a/tau) || softmax_r(s_hat/tau)) + beta*|Sum_r b_hat_r - B_read|
              , the softmax over the region axis per item and active iteration, plus the halt
              target [spec]
    FLOPs   = Sum_i Sum_r A[i, r] * phi_r * ctx_r     (differentiable through the straight-through
                                                        paths); the penalty is
                                                        lambda_flops * relu(FLOPs/FLOPs_target - 1)

Each class here computes exactly one of those terms (already multiplied by its own weight, so a
caller in `mind.py` sums the four return values -- or a subset, per Table 6's phase -- to get a
phase's total loss). None of these classes runs a region, the workspace, or the controller: every
input is a tensor a caller (`mind.py`, lane IC-8, or a test) already has in hand from
`readout.py` (lane IC-4, `RankHead`/`UnifyProbes`/`FrontalReadout`) or `controller.py` (lane
IC-5, `ControllerOutput`).

SPEC AMBIGUITIES RESOLVED HERE (recorded per the task's instruction to do the smallest honest
thing where the spec is silent, and to record the choice)

1. **`RankLoss`'s relationship to `RankHead`'s temperature.** Section 4 says `L_task` divides by
   "a recorded temperature," but `readout.py`'s `RankHead.forward` already divides its cosine
   scores by its own constructor-time `temperature` before returning them (see that file's
   docstring, ambiguity 3) -- so the scores this class receives are already temperature-scaled.
   `RankLoss` therefore applies no further temperature of its own; it is exactly
   `F.cross_entropy(scores, target)`, with `weight` (default `1.0`) present only so `mind.py` can
   scale `L_task`'s contribution relative to the other terms uniformly with the other three
   classes' constructors, per Table 2's "the per-loss weights."

2. **`L_unify`'s admission mask.** The formula sums "over the admitted regions," but admission is
   per-item per-region (`Sum_i A[i, r] >= 1`, `controller.py`'s `ControllerOutput.A`), not a
   module-wide constant. `UnifyLoss.forward` therefore takes an optional per-region boolean mask
   `[B]`; a region with no mask entry, or `admitted=None` entirely, is treated as admitted for
   every item -- the correct default for phase A's dense schedule (`A === 1`, spec section 3 step
   2), which is the only phase table 6 assigns this loss to on its own (`L_A = L_task +
   delta*L_unify`, phases A/W5b/E2, and again unweighted-by-delta inside `L_D`).

3. **`L_B`'s two operands are both read as logits.** The formula's `softmax_r(a/tau)` treats the
   *measured* attention mass `a` (already a probability distribution, `workspace.py`'s per-source
   marginal, `Sum_r a[b, i, r] = 1`) as if it were a further set of logits divided by `tau` and
   re-softmaxed -- not as a fixed target distribution consumed directly. This file follows the
   formula literally rather than "fixing" it to a plain KL between two already-normalised
   distributions: `DistilLoss.forward` takes `teacher_logits` (`a`, or any `[B, I, R]` tensor
   playing that role) and `student_logits` (`s_hat`, `controller.py`'s `b_logits` broadcast per
   iteration, or `A_logits.transpose(1, 2)` -- `mind.py`'s choice, out of this file's scope) and
   applies `softmax(./tau)` to both sides itself. One consequence, checked by this file's zero
   test: passing the same tensor as both operands drives the KL term to exactly zero, which is
   what "L_B is zero when s_hat = a" (Table 9) means operationally.

4. **`L_B`'s halt term has no stated functional form or weight.** Section 4 tags "the halt target"
   itself `[spec]` (defined two sentences later as "the first iteration at which `cos(f_i, f_I) >=
   0.99` on the dense run," a class index per item) but never gives the *loss* fitting
   `halt_logits` to that index a formula or a coefficient -- unlike the KL and budget terms, which
   both have one. The smallest honest addition is ordinary `F.cross_entropy(halt_logits,
   halt_target)` behind its own optional constructor weight `halt_weight` (default `1.0`), off by
   default input (`halt_logits=None` or `halt_target=None` skips the term, contributing exactly
   `0.0`) so a caller that has not yet computed the dense-run halt target -- or a test isolating
   the KL/budget terms Table 9 actually names -- is not forced to supply one.

5. **`FLOPs`'s region axis and `A`'s dtype.** The formula sums over the same region axis `A`
   shares with the controller's admission matrix, but `ctx_r` is undefined for `episodic_store`
   (Table 4a; "the store has no `ctx`, its context is the scope partition"), so the formula's own
   `r` cannot range over all `R` participants without a value for the store's term. This file does
   not decide that: `FlopsPenalty.forward` treats its region axis generically as whatever `phi`
   and `ctx` the caller supplies, sized to match `A`'s last axis -- `mind.py` decides whether that
   axis is `R_ctx` (excluding the store) or `R` with a zeroed store column. "Differentiable
   through the straight-through paths" (section 4) means `A` must be the *soft* admission tensor
   the straight-through estimator produces (real-valued in `[0, 1]`, gradient-carrying), not the
   hard boolean `ControllerOutput.A` -- this file accepts any float tensor and does not itself
   apply or check a straight-through gate, since building that gate is `controller.py`'s job
   (lane IC-5), not this file's.

6. **The dropped hinge term.** TAX:1763-1769 describes and explicitly drops "a hinge that rewards
   every region for being individually necessary" as gradient descent on the integration metric
   itself. This file implements no such class, matching the taxonomy's own decision -- named here
   because Table 2's four-class list would otherwise look like an omission rather than a choice.

SPEC SECTIONS THIS FILE IMPLEMENTS
Section 2.1 Table 2 (this file's row); section 4 ("Loss definitions", `L_task`, `L_unify`, `L_B`,
`FLOPs`); section 5 Table 9 (`losses.py` row: "each loss is finite on random inputs; `L_B` is
zero when `s_hat = a`; the FLOPs penalty is zero at target and positive above it").
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

__all__ = ["DistilLoss", "FlopsPenalty", "RankLoss", "UnifyLoss"]


class RankLoss(nn.Module):
    """`L_task`: softmax cross-entropy over the `k` cosine scores (spec section 4; TAX:1684-1690).

    Stateless and parameter-free -- registered as an `nn.Module` purely so it composes with the
    other three loss classes under one `nn.ModuleDict` in a caller's training script, the way
    `readout.py`'s parameter-free `FlooredSimplex`-style pieces are still modules. See this
    file's module docstring, ambiguity 1, for why no temperature is applied here.
    """

    def __init__(self, weight: float = 1.0) -> None:
        """Store the scalar this loss's contribution is multiplied by.

        Args:
            weight: Multiplies the returned cross-entropy. Table 2's "per-loss weights"; `L_task`
                itself carries no explicit coefficient in section 4's `L_A`/`L_D` formulas, so the
                default is `1.0` and this argument exists for a caller (`mind.py`) that wants a
                uniform per-term-weight interface across all four classes in this file.
        """
        super().__init__()
        self.weight = weight

    def forward(self, scores: Tensor, target: Tensor) -> Tensor:
        """Cross-entropy of `RankHead`'s scores against the gold candidate index.

        Args:
            scores: `[B, k]` float, `RankHead.forward`'s output -- already divided by its own
                `temperature`, `NULL` at index 0 (spec section 3 step 12).
            target: `[B]` int64, the gold candidate's index into the `k` axis (`0` for a
                general-bin item whose correct answer is `NULL`).

        Returns:
            A scalar: `weight * cross_entropy(scores, target)`, batch-mean reduced.
        """
        return self.weight * functional.cross_entropy(scores, target)


class UnifyLoss(nn.Module):
    """`L_unify`: per-region `1 - cos(probe_r(f), pool_r(h_r))`, summed over admitted regions.

    Spec section 4; TAX:1708-1713, "the unified state must be linearly sufficient for each
    contributing region's own answer as well as for the joint one." `probe_r(f)` is
    `readout.py`'s `UnifyProbes.forward(f)` output; `pool_r(h_r)` is the region's own
    `Faculty.pool(h_r, mask_r)` output (`faculty/protocol.py`) -- both dicts keyed by region
    name, computed by the caller, this class only combines them.
    """

    def __init__(self, weight: float) -> None:
        """Store `delta`, the coefficient `L_A` and `L_D` apply to this term.

        Args:
            weight: `delta` in `L_A = L_task + delta*L_unify` and `L_D`'s identical term (spec
                section 4; Table 6 rows A, W5b, E2, D). Required with no default: unlike
                `RankLoss`, this term's coefficient is named explicitly in every formula that
                uses it, so silently defaulting it would hide a real training-recipe choice.
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        probe_outputs: dict[str, Tensor],
        pooled_targets: dict[str, Tensor],
        admitted: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Sum `1 - cos(probe_r(f), pool_r(h_r))` over admitted regions, batch-mean reduced.

        Args:
            probe_outputs: `{region: [B, pooled_dim_r]}`, `UnifyProbes.forward(f)`'s output.
            pooled_targets: `{region: [B, pooled_dim_r]}`, one entry per key of `probe_outputs`
                (missing or extra keys raise), the region's own `pool()` output for the same
                batch.
            admitted: `{region: [B]}` bool, `True` where that item admitted that region at least
                once (`Sum_i A[i, r] >= 1`). A region absent from this mapping, or `admitted=None`
                entirely, is treated as admitted for every item -- the phase-A dense default (spec
                section 4, ambiguity 2 above).

        Returns:
            A scalar: `weight * mean_b( Sum_r admitted[r][b] * (1 - cos(probe_outputs[r][b],
            pooled_targets[r][b])) )`.

        Raises:
            KeyError: `pooled_targets` is missing a key `probe_outputs` names.
        """
        missing = probe_outputs.keys() - pooled_targets.keys()
        if missing:
            raise KeyError(f"pooled_targets is missing region(s) {sorted(missing)!r}.")
        per_item_total: Tensor | None = None
        for region, probe_out in probe_outputs.items():
            target = pooled_targets[region]
            term = 1.0 - functional.cosine_similarity(probe_out, target, dim=-1)  # [B]
            if admitted is not None and region in admitted:
                term = term * admitted[region].to(term.dtype)
            per_item_total = term if per_item_total is None else per_item_total + term
        if per_item_total is None:
            return torch.zeros((), dtype=torch.float32)
        return self.weight * per_item_total.mean()


class DistilLoss(nn.Module):
    """`L_B`: per-item, per-active-iteration KL between two schedule-logit tensors, plus terms.

    Spec section 4; TAX:1715-1727 (phase B, "bootstrap the controller by distillation").
    `L_B = Sum_i Sum_r KL(softmax_r(a/tau) || softmax_r(s_hat/tau)) + beta*|Sum_r b_hat_r -
    B_read|`, plus the (unformalised, `[spec]`-tagged) halt term this file adds behind its own
    weight -- see the module docstring, ambiguities 3 and 4, for both operands' reading and the
    halt term's construction.
    """

    def __init__(self, temperature: float, beta: float, halt_weight: float = 1.0) -> None:
        """Store the three coefficients `L_B`'s formula and this file's halt addition need.

        Args:
            temperature: `tau` dividing both `a` and `s_hat` before each is re-softmaxed. Required
                with no default, matching `RankHead`'s treatment of the same word (`readout.py`,
                ambiguity 3): a recorded training-recipe choice, not a library constant.
            beta: Coefficient of the budget-conservation term `beta*|Sum_r b_hat_r - B_read|`.
            halt_weight: Coefficient of this file's added halt cross-entropy term (module
                docstring, ambiguity 4). Default `1.0`; the term itself is skipped (contributes
                `0.0`) whenever `halt_logits` or `halt_target` is `None` at `forward` time.

        Raises:
            ValueError: `temperature <= 0`.
        """
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}.")
        self.temperature = temperature
        self.beta = beta
        self.halt_weight = halt_weight

    def forward(
        self,
        teacher_logits: Tensor,
        student_logits: Tensor,
        active: Tensor,
        b_hat: Tensor,
        b_read: float,
        halt_logits: Tensor | None = None,
        halt_target: Tensor | None = None,
    ) -> Tensor:
        """`L_B`, batch-mean reduced.

        Args:
            teacher_logits: `[B, I, R]` float, `a` (or any tensor playing that role) -- treated
                as logits, per the module docstring's ambiguity 3.
            student_logits: `[B, I, R]` float, `s_hat` -- same treatment.
            active: `[B, I]` bool, `True` for an iteration the KL term counts
                (`ControllerOutput.active`, or the teacher run's own `active`); an iteration with
                no active items in the batch contributes `0.0` rather than `nan`.
            b_hat: `[B, R]` float, the controller's predicted per-region read-token share for the
                budget-conservation term.
            b_read: `B_read`, the scalar target the per-item `Sum_r b_hat_r` is compared against.
            halt_logits: `[B, I]` float or `None`. `None` skips the halt term.
            halt_target: `[B]` int64 or `None`, the dense-run halt iteration index (module
                docstring, ambiguity 4). `None` skips the halt term.

        Returns:
            A scalar: the KL term (masked by `active`, summed over `i` and `r`, batch-mean
            reduced) plus `beta * mean_b(|Sum_r b_hat[b, r] - b_read|)` plus, when both halt
            arguments are given, `halt_weight * cross_entropy(halt_logits, halt_target)`.
        """
        student_log_prob = functional.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_prob = functional.softmax(teacher_logits / self.temperature, dim=-1)
        kl_per_iter = functional.kl_div(student_log_prob, teacher_prob, reduction="none").sum(
            dim=-1
        )  # [B, I]
        kl = (kl_per_iter * active.to(kl_per_iter.dtype)).sum(dim=-1).mean()  # scalar

        budget_term = self.beta * (b_hat.sum(dim=-1) - b_read).abs().mean()

        halt_term = teacher_logits.new_zeros(())
        if halt_logits is not None and halt_target is not None:
            halt_term = self.halt_weight * functional.cross_entropy(halt_logits, halt_target)

        return kl + budget_term + halt_term


class FlopsPenalty(nn.Module):
    """`lambda_flops * relu(FLOPs/FLOPs_target - 1)` (spec section 4; TAX:1738-1755, phase D).

    `FLOPs = Sum_i Sum_r A[i, r] * phi_r * ctx_r`. See the module docstring, ambiguity 5, for why
    this class treats its region axis generically rather than assuming `R` or `R_ctx`.
    """

    def __init__(self, weight: float, flops_target: float) -> None:
        """Store `lambda_flops` and `FLOPs_target`.

        Args:
            weight: `lambda_flops` in `L_D`'s formula.
            flops_target: The denominator `FLOPs` is normalised against before the hinge; section
                1 gives the module-wide default (the dense schedule's own worst-case
                `region_token_flops`), but this class takes whatever value the caller (`mind.py`,
                reading `InterconnectConfig.flops_ceiling` or a phase-specific target) supplies.

        Raises:
            ValueError: `flops_target <= 0`.
        """
        super().__init__()
        if flops_target <= 0:
            raise ValueError(f"flops_target must be positive, got {flops_target}.")
        self.weight = weight
        self.flops_target = flops_target

    def forward(self, A: Tensor, phi: Tensor, ctx: Tensor) -> Tensor:  # noqa: N803
        """The FLOPs penalty, batch-mean reduced.

        Args:
            A: `[B, I, R']` float in `[0, 1]`, the *soft* (straight-through) admission tensor --
                not `ControllerOutput.A`'s hard bool, so gradient reaches the estimator behind it
                (module docstring, ambiguity 5).
            phi: `[R']` float, `phi_r`, each region's measured MACs per input position.
            ctx: `[B, R']` float, `ctx_r` as run, constant across iterations (spec section 3 step
                2 sets `ctx` once, before the iteration loop).

        Returns:
            A scalar: `weight * mean_b( relu( Sum_{i,r}(A[b,i,r]*phi[r]*ctx[b,r]) / flops_target -
            1 ) )`.

        Raises:
            ValueError: `phi`'s length or `ctx`'s last dimension does not match `A`'s last
                dimension.
        """
        r_dim = A.shape[-1]
        if phi.shape[-1] != r_dim or ctx.shape[-1] != r_dim:
            raise ValueError(
                f"A, phi and ctx must share a region axis of the same length; got "
                f"A[..., {r_dim}], phi[..., {phi.shape[-1]}], ctx[..., {ctx.shape[-1]}]."
            )
        flops = (A * phi.view(1, 1, -1) * ctx.unsqueeze(1)).sum(dim=(1, 2))  # [B]
        return self.weight * functional.relu(flops / self.flops_target - 1.0).mean()
