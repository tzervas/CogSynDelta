"""Phase A of the DEC-50 training contract -- Table 6's row "A dense".

WHAT THIS FILE IS
`docs/design/INTERCONNECT-MODULE-SPEC.md` section 4 is the binding specification, and its
Table 6 row "A dense: W5 (`R = 4`), E2 (`R = 5`)" is the only row this file implements.
Everything the module needed to run a forward pass existed before it; nothing anywhere
under `src/cogsyndelta/interconnect/` built an optimizer or took a backward step. This
file is that gap: the phase-A trainable/frozen partition, `L_A = L_task + delta*L_unify`,
the four gates Table 6's row A names that `gates.py` implements, and the Table 7 receipt.

WHY THIS FILE IS NOT IN TABLE 2
Spec section 2.1 Table 2 enumerates twelve filenames across eleven rows -- its ninth row
names `gates.py` and `receipts.py` together -- and this branch adds two more, `phase_a.py`
and `cli.py`. The file count is not the argument and never was; what matters is that Table
2 assigns a trainer no home at all. Section
1 excludes "the compose-stage driver script, which imports `gates.py` and `receipts.py`
and runs the phases of section 4" from the module. Those two facts do not agree about
where an optimizer lives, and the exclusion is narrower than it looks: what section 1
pushes outside is the DRIVER -- corpus loading, phase sequencing, checkpoint management --
not the definition of what phase A trains and what it freezes. That definition is section
4's own content and belongs beside the module it partitions, because it is a statement
about `WhiteMatter`'s submodules, not about a corpus. `cli.py` is the thin harness entry
point over this file; a real compose-stage driver over a real corpus is still out of scope
and still unwritten.

THE TWO NUMBERS THAT DISAGREE, AND WHICH ONE THIS FILE OBEYS
Section 2.3 Table 4's last core row is labelled "trainable in phase A, heads included,
`R = 5`" and gives 28,376,334. Measured against the code, 28,376,334 is the count of EVERY
parameter `WhiteMatter` owns at `R = 5` -- it includes the thalamic controller's 1,683,978,
which Table 6's own "frozen" column for row A lists as "controller (bypassed)".
`tests/interconnect/test_parameter_table.py` already pins 28,376,334 as that total, and
`tests/interconnect/test_smoke.py` already freezes the controller before its backward pass
with a `[lane]` note about the same tension.

This file treats section 4 as binding, because the task that commissioned it says so and
because Table 6 is the more specific statement: the phase-A OPTIMIZER receives
`PHASE_A_TRAINABLE_R5 = 26_692_356 = 28,376,334 - 1,683,978`. Both numbers are asserted in
`tests/interconnect/test_phase_a.py`, against the module, with the subtraction shown, so
the discrepancy is recorded as a measurement rather than resolved by picking a side
quietly. See `docs/design/INTERCONNECT-MODULE-SPEC.md` section 2.3 Table 4 and section 4
Table 6.

HOW THE TWO SETS ARE ESTABLISHED
Not by reading `requires_grad` -- that is the thing under test, not the definition.
`phase_a_parameter_partition` names Table 6's "receives gradient" column as an explicit
allow-list of `WhiteMatter` submodule prefixes, names the "frozen" column as a second
explicit list, and then REFUSES unless the two together are a total partition of
`WhiteMatter.named_parameters()`. A submodule added to `mind.py` later therefore cannot
drift into neither bucket and be silently untrained; it fails loudly here instead.

The regions are outside that partition entirely: `WhiteMatter.faculties` is a plain dict,
not a registered `nn.ModuleDict`, so faculty parameters never appear in
`WhiteMatter.named_parameters()` and can never reach the optimizer. That is a structural
guarantee about the OPTIMIZER, not about autograd -- gradient still flows back through a
region's own weights on the path `h_r -> adapter -> bank -> z -> f`. So `PhaseATrainer`
also freezes every faculty parameter explicitly, and the tests assert `param.grad is None`
after a real backward pass rather than trusting `requires_grad`.

WHAT PHASE A DOES NOT DO
Phases B, C and D (Table 6's other rows) need the controller as a student, a distillation
teacher and straight-through estimators; none of that is here. Table 6 row A's gate column
also names "G1, G2, G0", which are taxonomy-level gates with no function in `gates.py` and
no Table 7 field of their own; this file wires the four Table 8 guards that DO exist --
G29, G33, G35, G36 -- and names the other three IN THE WRITTEN RECEIPT, under
`verdicts.unimplemented_gates`, so a reader holding only the JSON can see which gates were
never evaluated instead of reading the verdict as an unqualified pass.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from cogsyndelta.interconnect.gates import (
    GateFailure,
    check_attention_mass_floor,
    check_frozen_set_identity,
    check_null_gate,
    check_overfit_gate,
)
from cogsyndelta.interconnect.kv_bank import STORE_PARTICIPANT
from cogsyndelta.interconnect.losses import RankLoss, UnifyLoss
from cogsyndelta.interconnect.mind import WhiteMatter, WhiteMatterOutput
from cogsyndelta.interconnect.receipts import ComposeReceipt

__all__ = [
    "CONTROLLER_R5",
    "PHASE_A_FROZEN_PREFIXES",
    "PHASE_A_STORE_PREFIXES",
    "PHASE_A_TRAINABLE_PREFIXES",
    "PHASE_A_TRAINABLE_R5",
    "PHASE_A_WRITE_BACK_PREFIXES",
    "TABLE_4_TOTAL_R5",
    "PhaseABatch",
    "PhaseAConfig",
    "PhaseAGuardReport",
    "PhaseAResult",
    "PhaseATrainer",
    "StepRecord",
    "check_phase_a_frozen_set",
    "check_receipt_collapse_floor",
    "loss_decreased",
    "phase_a_guards",
    "phase_a_parameter_partition",
]

#: Section 2.3 Table 4's "trainable in phase A, heads included, `R = 5`" figure, which is
#: measurably the count of every parameter `WhiteMatter` owns at `R = 5`.
TABLE_4_TOTAL_R5 = 28_376_334
#: Table 6 row A's actual optimizer set at `R = 5`: `TABLE_4_TOTAL_R5` minus the thalamic
#: controller's 1,683,978, which row A's "frozen" column lists as "controller (bypassed)".
PHASE_A_TRAINABLE_R5 = 26_692_356
#: The controller's own count at `R = 5`, i.e. the whole of the difference above.
CONTROLLER_R5 = 1_683_978

#: Table 6 row A, "receives gradient", as `WhiteMatter` submodule prefixes. Read the row
#: left to right: "workspace" (`workspace`, whose blocks carry the attention/MLP weights),
#: "LayerNorms" (inside `workspace.blocks`, plus `final_norm`), "adapters"
#: (`region_adapters`), "scorers" (`selectors`, `TopKSelect`'s per-region scorer), "type
#: embeddings" (`kv_bank.type_emb`), "latent bank" (`workspace.latent_bank`), "frontal"
#: (`frontal_readout`), "rank head" and "`NULL`" (`rank_head`, which owns `NullCandidate`),
#: "unify probes" (`unify_probes`). `kv_bank.type_emb` is named rather than `kv_bank`
#: wholesale because the same module also owns `store_projection`, which the row admits
#: only "in E2".
PHASE_A_TRAINABLE_PREFIXES: tuple[str, ...] = (
    "workspace.",
    "final_norm.",
    "region_adapters.",
    "selectors.",
    "kv_bank.type_emb",
    "frontal_readout.",
    "rank_head.",
    "unify_probes.",
)
#: Table 6 row A, "frozen": "controller (bypassed), regions". The regions are not here
#: because they are not `WhiteMatter` parameters at all (see the module docstring); this
#: tuple holds only what `WhiteMatter` itself owns and phase A must not train.
PHASE_A_FROZEN_PREFIXES: tuple[str, ...] = ("controller.",)
#: Conditional on `InterconnectConfig.write_back`: "prefixes only when write-back is on".
PHASE_A_WRITE_BACK_PREFIXES: tuple[str, ...] = ("conditioning.",)
#: Conditional on the store being an admitted participant: "`W_k`/`W_v` in E2".
PHASE_A_STORE_PREFIXES: tuple[str, ...] = ("kv_bank.store_projection.",)

#: One point, matching `gates.POINT`, for the receipt's own gap reporting.
POINT = 0.01
#: Fixed histogram edges for the per-region attention-mass histogram (Table 7's
#: "the per-region histogram"): ten equal buckets over `[0, 1]`.
HISTOGRAM_BUCKETS = 10


def _matches(name: str, prefixes: Sequence[str]) -> bool:
    """Return whether `name` starts with any prefix in `prefixes`."""
    return any(name.startswith(prefix) for prefix in prefixes)


def phase_a_parameter_partition(
    white_matter: WhiteMatter,
) -> tuple[dict[str, nn.Parameter], dict[str, nn.Parameter]]:
    """Split `WhiteMatter`'s parameters into Table 6 row A's two columns.

    The split is by explicit submodule prefix, never by reading `requires_grad`: the point
    of this function is to be the DEFINITION that `requires_grad` is then set from and
    tested against. It refuses rather than guesses when a parameter matches neither column
    or both, so a submodule added to `mind.py` after this file was written cannot silently
    end up untrained.

    Args:
        white_matter: The module to partition. Its `config.write_back`,
            `config.unify_probes_trainable` and whether `episodic_store` is a declared
            participant all move parameters between the two columns, exactly as Table 6's
            row A conditions them.

    Returns:
        `(trainable, frozen)`, two `{qualified_name: Parameter}` dicts whose keys are
        disjoint and whose union is exactly `dict(white_matter.named_parameters())`.

    Raises:
        GateFailure: A parameter matched neither column, or matched both -- either way the
            phase-A trainable set is not well defined and the run must not start.
    """
    cfg = white_matter.config
    trainable_prefixes = list(PHASE_A_TRAINABLE_PREFIXES)
    frozen_prefixes = list(PHASE_A_FROZEN_PREFIXES)

    # "prefixes only when write-back is enabled" (Table 6 row A).
    if cfg.write_back:
        trainable_prefixes.extend(PHASE_A_WRITE_BACK_PREFIXES)
    else:
        frozen_prefixes.extend(PHASE_A_WRITE_BACK_PREFIXES)

    # "`W_k`/`W_v` in E2" -- i.e. only once the store is an admitted participant. At
    # `R = 4` `mind.py` already ships them with `requires_grad=False` (Table 4's `R = 4`
    # row: "stay instantiated and receive no gradient in W5"); this keeps that true rather
    # than unfreezing them by re-declaring the whole of `kv_bank` trainable.
    if white_matter.has_store:
        trainable_prefixes.extend(PHASE_A_STORE_PREFIXES)
    else:
        frozen_prefixes.extend(PHASE_A_STORE_PREFIXES)

    # Spec section 4's one-flag switch: `unify_probes_trainable = False` reads the
    # taxonomy's "frozen" probe as a seeded random basis and moves 590,976 parameters out
    # of the trainable set. `mind.py` already applies it to `requires_grad`; the partition
    # has to agree or the optimizer would be handed parameters that never move.
    if not cfg.unify_probes_trainable:
        trainable_prefixes.remove("unify_probes.")
        frozen_prefixes.append("unify_probes.")

    trainable: dict[str, nn.Parameter] = {}
    frozen: dict[str, nn.Parameter] = {}
    unclaimed: list[str] = []
    contested: list[str] = []
    for name, param in white_matter.named_parameters():
        in_trainable = _matches(name, trainable_prefixes)
        in_frozen = _matches(name, frozen_prefixes)
        if in_trainable and in_frozen:
            contested.append(name)
        elif in_trainable:
            trainable[name] = param
        elif in_frozen:
            frozen[name] = param
        else:
            unclaimed.append(name)

    if contested:
        raise GateFailure(
            "phase A: parameter(s) matched both Table 6 columns, so the trainable set is "
            f"ambiguous: {sorted(contested)}"
        )
    if unclaimed:
        raise GateFailure(
            "phase A: parameter(s) matched neither Table 6 column, so they would be "
            f"neither trained nor deliberately frozen: {sorted(unclaimed)}"
        )
    return trainable, frozen


def check_phase_a_frozen_set(
    participants: Sequence[Mapping[str, Any]],
    *,
    expected_participants: Sequence[str],
    expected_r: int,
) -> None:
    """G33 as phase A must apply it: identity, and the frozen set's own shape.

    `gates.check_frozen_set_identity` checks the halves Table 8 row G33 names literally --
    a checkpoint sha that disagrees with the receipt it cites, and a parametric region
    claiming `kind: "nonparametric_store"`. Table 7's frozen-set group asks for more than
    that: each row's `receipt_path` and `status`, plus `participants` and `R`, and section
    4's row A says phase A "refuses to start on a mismatch". A run whose frozen set names
    four regions while the module was built for five is exactly such a mismatch, and the
    identity check alone would pass it, because every row it does look at is internally
    consistent.

    Args:
        participants: One mapping per participant, carrying `name`,
            `checkpoint_sha256`, `receipt_checkpoint_sha256`, `receipt_path`, `status`,
            `kind` and `parametric`.
        expected_participants: The participant names the built `WhiteMatter` actually has,
            in its own order.
        expected_r: `R`, the participant count the module was built for.

    Raises:
        GateFailure: Any half fails. Phase A refuses to start.
    """
    check_frozen_set_identity(participants)
    for row in participants:
        for required in ("receipt_path", "status"):
            if not row.get(required):
                raise GateFailure(
                    f"G33: participant {row.get('name')!r} has no {required!r}; Table 7's "
                    "frozen-set group requires one per region and phase A refuses to "
                    "start without it"
                )
    declared = [str(row["name"]) for row in participants]
    if declared != list(expected_participants):
        raise GateFailure(
            f"G33: the frozen set names participants {declared} but the module was built "
            f"for {list(expected_participants)}; phase A refuses to start"
        )
    if expected_r != len(expected_participants):
        raise GateFailure(
            f"G33: the frozen set declares R={expected_r} over "
            f"{len(expected_participants)} participants; phase A refuses to start"
        )


@dataclass(frozen=True)
class PhaseAConfig:
    """The phase-A training recipe -- everything section 4 leaves to the caller.

    `delta` and `task_weight` are `L_A = L_task + delta*L_unify`'s two coefficients; the
    rank temperature is NOT here, because `RankHead` owns it (`InterconnectConfig
    .rank_temperature`) and the receipt must record what the loss site actually used
    rather than what any config said (see `PhaseATrainer.observed_loss_site`).
    """

    delta: float = 1.0
    """`delta` in `L_A = L_task + delta*L_unify` (spec section 4)."""
    task_weight: float = 1.0
    """`L_task`'s coefficient. Section 4's formula carries none, so the default is 1.0 and
    a non-default value is a deliberate, recorded recipe choice."""
    lr: float = 3e-4
    """AdamW learning rate."""
    weight_decay: float = 0.01
    """AdamW weight decay."""
    steps: int = 100
    """Optimizer steps. Small by design: this file is built and tested, never run at
    scale, until a reviewed merge says otherwise (spec section 6 Q5 recommendation (a))."""
    grad_clip: float | None = 1.0
    """Global grad-norm clip, or `None` to disable."""
    seed: int = 0
    """`torch.manual_seed` before the first step; Table 7's identity group records it."""


@dataclass(frozen=True)
class PhaseABatch:
    """One phase-A training or dev item batch.

    `bins` exists because Table 7 asks for `compose.recall@1` per bin, `compose.null_recall`
    "on the general bin" and `compose.null_fpr` "per other bin": without a per-item bin
    label none of those three fields is computable. When it is `None` the bin is derived --
    `"general"` for an item whose gold candidate is `NULL` (index 0), `"content"`
    otherwise -- which is the smallest labelling that makes the `NULL` gate meaningful.
    """

    inputs: dict[str, Any]
    """`WhiteMatter.forward`'s `inputs` mapping; must carry `"candidates"`, since
    `L_task` is cross-entropy over `RankHead`'s scores."""
    target: Tensor
    """`[B]` int64, the gold candidate's index into the `k` axis; `0` means `NULL`."""
    bins: tuple[str, ...] | None = None
    """Per-item bin name, or `None` to derive it from `target`."""

    def bin_labels(self) -> tuple[str, ...]:
        """Return the per-item bin names, deriving them from `target` when unset."""
        if self.bins is not None:
            return self.bins
        return tuple("general" if int(t) == 0 else "content" for t in self.target)


@dataclass(frozen=True)
class StepRecord:
    """One optimizer step's numbers, for the loss curve and the attention accumulator."""

    step: int
    loss: float
    l_task: float
    l_unify: float
    grad_norm: float


@dataclass
class PhaseAGuardReport:
    """The outcome of every Table 6 row A gate this module can actually evaluate.

    Each field records what the guard saw, not merely whether it passed, so a receipt
    reader can re-derive the verdict instead of trusting it.
    """

    collapse_floor: dict[str, Any]
    """Table 7's `collapse_floor`: `{expression: "eta/R", R, value}`."""
    collapsed_in_phase_A: list[str] = field(default_factory=list)  # noqa: N815
    """G29's effect: every region whose mean attention mass fell below `eta/R`. Named
    here and excluded from phase B's distillation targets. The mixedCase name is Table 7's
    own field name, verbatim (`collapsed_in_phase_A: []`), and this dataclass is written
    straight into the receipt under that key; renaming it to satisfy N815 would put a
    field in the receipt that Table 7 does not name."""
    floor_failures: dict[str, str] = field(default_factory=dict)
    """`{region: the GateFailure message}` for each collapsed region."""
    overfit_gap: float = 0.0
    """`train_metric - held_out_metric` on the composed metric."""
    overfit_failure: str | None = None
    """G35's message when the gap reached five points, else `None`."""
    null_failure: str | None = None
    """G36's message when `NULL` recall or a per-bin false-positive rate failed, else
    `None`."""
    unimplemented_gates: tuple[str, ...] = ("G0", "G1", "G2")
    """Table 6 row A's gate column also names G0, G1 and G2. They are taxonomy-level
    gates with no function in `gates.py` and no Table 7 field, so they are declared
    unevaluated rather than silently omitted. `build_receipt` writes them into the receipt
    under `verdicts.unimplemented_gates`; asserting them against this object alone would
    leave a reader holding only the JSON unable to learn which gates were skipped."""

    @property
    def passed(self) -> bool:
        """Return whether every evaluated gate passed."""
        return not (self.collapsed_in_phase_A or self.overfit_failure or self.null_failure)

    def verdict(self) -> str:
        """Return Table 7's `verdicts.integration` string for this report."""
        if self.passed:
            return "PASS: phase A, every evaluated gate clear"
        reasons = []
        if self.collapsed_in_phase_A:
            reasons.append(f"G29 collapsed={sorted(self.collapsed_in_phase_A)}")
        if self.overfit_failure:
            reasons.append(f"G35 {self.overfit_failure}")
        if self.null_failure:
            reasons.append(f"G36 {self.null_failure}")
        return "FAIL: " + "; ".join(reasons)


def phase_a_guards(
    *,
    mean_per_region: Mapping[str, float],
    eta: float,
    r: int,
    train_metric: float,
    held_out_metric: float,
    general_null_recall: float,
    per_bin_null_fpr: Mapping[str, float],
) -> PhaseAGuardReport:
    """Evaluate G29, G35 and G36 over one phase-A run's measurements.

    G33 is not here: it fires at construction, before a step runs (see
    `check_phase_a_frozen_set` and `PhaseATrainer.__init__`).

    G29 has two independent halves and this function evaluates exactly one: the collapse
    half, whose effect is that a region is NAMED in `collapsed_in_phase_A`. The receipt
    half -- "a receipt whose printed floor != `eta/R` at its own `R`" -- is NOT checkable
    from here. No printed floor is in scope yet, only `eta` and `r`, so a check made here
    could only compare `eta / r` against `eta / r` recomputed from the same two arguments:
    a tautology that cannot refuse anything. An earlier revision did make that call, with
    `printed_floor=eta / r`; it has been deleted rather than left standing as coverage it
    did not provide (memory note `verify-guards-by-making-them-fail`). The receipt half is
    checked over the BUILT receipt by `check_receipt_collapse_floor`, which
    `PhaseATrainer.write_receipt` runs before the receipt can reach disk.

    Args:
        mean_per_region: `{region: mean(a_r)}` over active iterations.
        eta: `InterconnectConfig.floor_eta`.
        r: The participant count this run was measured at.
        train_metric: The composed metric on the training split.
        held_out_metric: The composed metric on the dev split.
        general_null_recall: `NULL` recall on the general bin.
        per_bin_null_fpr: `{bin: NULL false-positive rate}` for every non-general bin.

    Returns:
        A `PhaseAGuardReport`.

    Raises:
        ValueError: `mean_per_region` is empty, so no floor can be checked at all.
    """
    if not mean_per_region:
        raise ValueError("phase_a_guards: mean_per_region is empty; nothing to gate.")
    floor_value = eta / r
    collapse_floor = {"expression": "eta/R", "R": r, "value": floor_value}

    report = PhaseAGuardReport(collapse_floor=collapse_floor)
    for name, mean_a in mean_per_region.items():
        try:
            check_attention_mass_floor(name, mean_a, eta, r)
        except GateFailure as exc:
            report.collapsed_in_phase_A.append(name)
            report.floor_failures[name] = str(exc)

    report.overfit_gap = train_metric - held_out_metric
    try:
        check_overfit_gate(train_metric, held_out_metric)
    except GateFailure as exc:
        report.overfit_failure = str(exc)

    try:
        check_null_gate(general_null_recall, per_bin_null_fpr)
    except GateFailure as exc:
        report.null_failure = str(exc)

    return report


def check_receipt_collapse_floor(receipt: Mapping[str, Any]) -> None:
    """G29 re-derived from a BUILT receipt, which is where its "refused" effect applies.

    Table 8 row G29 has two effects, and they attach to different halves. A region below
    the floor is NAMED in `collapsed_in_phase_A` and excluded from phase B's targets --
    that is a degradation, and `phase_a_guards` applies it during the run. "A receipt whose
    printed floor != `eta/R` at its own `R`" is the other half, and its effect is that the
    receipt is REFUSED. A refusal is about a document, so it is checked over the document:
    a reader with nothing but the JSON can run this and reach the same verdict the run
    did, which is the whole point of a receipt.

    Both halves are re-derived here from the receipt's own fields. Regions the receipt
    already names as collapsed are skipped, because the receipt has already applied G29's
    naming effect to them; every other region must still clear the floor.

    Args:
        receipt: A built phase-A receipt, i.e. `ComposeReceipt.build()`'s output or the
            JSON read back off disk.

    Raises:
        GateFailure: The printed floor disagrees with `eta/R` at the receipt's own `R`, a
            region that was not named as collapsed is below the floor, or every region is
            collapsed (in which case there is no honest region left to check the printed
            floor against, and the receipt is refused rather than passed by default).
        KeyError: The receipt is missing `budgets` or `attention_mass`.
    """
    budgets = receipt["budgets"]
    printed = budgets["collapse_floor"]
    attention = receipt["attention_mass"]
    masses: Mapping[str, Any] = attention["mean_per_region"]
    collapsed = set(attention["collapsed_in_phase_A"])
    eta = float(budgets["eta"])
    r = int(printed["R"])
    printed_value = float(printed["value"])

    healthy = [name for name in masses if name not in collapsed]
    if not healthy:
        raise GateFailure(
            f"G29: every one of the receipt's {len(masses)} regions is named in "
            "collapsed_in_phase_A, so the printed floor cannot be checked against any "
            "surviving region; the receipt is refused"
        )
    for name in healthy:
        check_attention_mass_floor(name, float(masses[name]), eta, r, printed_floor=printed_value)


@dataclass
class PhaseAResult:
    """Everything one phase-A run produces, and everything its receipt is built from."""

    steps: list[StepRecord]
    """One record per optimizer step, in order -- the loss curve."""
    mean_per_region: dict[str, float]
    """`{participant: mean(a_r)}` over active iterations and items."""
    histogram_per_region: dict[str, list[int]]
    """`{participant: ten bucket counts over [0, 1]}`."""
    train_metric: float
    """`compose.recall@1` on the training split."""
    dev_metric: float
    """`compose.recall@1` on the dev (held-out) split."""
    recall_per_bin: dict[str, float]
    """`compose.recall@1` per bin on the dev split."""
    general_null_recall: float
    """`compose.null_recall` on the general bin."""
    null_fpr_per_bin: dict[str, float]
    """`compose.null_fpr` per non-general bin."""
    ctx: dict[str, int]
    """`ctx_r` as run."""
    b: dict[str, int]
    """`b_r` as run."""
    halt_at_histogram: dict[str, int]
    """`{str(halt_at): count}` over the run."""
    wall_ms: float
    """Measured wall time of the whole run."""
    loss_site: dict[str, Any]
    """What `PhaseATrainer` observed AT the loss site -- see `observed_loss_site`."""
    guards: PhaseAGuardReport
    """Every evaluated gate's outcome."""
    edges: dict[str, Any] | None = None
    """`WhiteMatterOutput.edges`, present only when write-back is on."""

    @property
    def loss_curve(self) -> list[float]:
        """Return the per-step total loss, in order."""
        return [record.loss for record in self.steps]


class PhaseATrainer:
    """The phase-A optimizer, loss site and receipt builder (spec section 4, Table 6 row A).

    Construction applies G33 and then the Table 6 partition, in that order: a run whose
    frozen set does not match must refuse before any parameter's `requires_grad` is
    touched, so a refused run leaves the module exactly as it found it.
    """

    def __init__(
        self,
        white_matter: WhiteMatter,
        config: PhaseAConfig,
        frozen_set: Sequence[Mapping[str, Any]],
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        """Refuse on G33, partition the parameters, then build the losses and optimizer.

        Args:
            white_matter: The module to train. Its own `InterconnectConfig` decides
                whether the prefixes and the store projections are in the trainable set.
            config: The phase-A recipe.
            frozen_set: Table 7's frozen-set rows, one per participant, in the module's
                own participant order -- see `check_phase_a_frozen_set` for the fields.
            device: Where to run. CPU by default; this file never assumes a GPU.

        Raises:
            GateFailure: G33 refused the frozen set, or the Table 6 partition is not
                total over `white_matter.named_parameters()`.
        """
        check_phase_a_frozen_set(
            frozen_set,
            expected_participants=white_matter.participant_names,
            expected_r=len(white_matter.participant_names),
        )
        self.white_matter = white_matter.to(device)
        self.config = config
        self.device = torch.device(device)
        self.frozen_set = [dict(row) for row in frozen_set]

        self.trainable, self.frozen = phase_a_parameter_partition(self.white_matter)
        for param in self.frozen.values():
            param.requires_grad_(False)
        for param in self.trainable.values():
            param.requires_grad_(True)
        self.freeze_regions()

        self.rank_loss = RankLoss(config.task_weight)
        self.unify_loss = UnifyLoss(config.delta)
        self.optimizer = torch.optim.AdamW(
            list(self.trainable.values()), lr=config.lr, weight_decay=config.weight_decay
        )
        self._loss_site: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # The frozen sets
    # ------------------------------------------------------------------

    def freeze_regions(self) -> None:
        """Freeze every region parameter -- Table 6 row A's "frozen: ... regions".

        The regions are not `WhiteMatter` parameters (see the module docstring), so they
        can never reach the optimizer; this closes the other half, which is that autograd
        still reaches them through `h_r`. `Faculty` is a `Protocol` with no `parameters()`
        member, so the call is looked up rather than assumed: a conforming faculty that is
        not an `nn.Module` simply has nothing to freeze.
        """
        for faculty in self.white_matter.faculties.values():
            parameters = getattr(faculty, "parameters", None)
            if callable(parameters):
                for param in parameters():
                    param.requires_grad_(False)

    def region_parameters(self) -> Iterator[tuple[str, str, Tensor]]:
        """Yield `(faculty_name, parameter_name, parameter)` for every region parameter."""
        for faculty_name, faculty in self.white_matter.faculties.items():
            named = getattr(faculty, "named_parameters", None)
            if callable(named):
                for param_name, param in named():
                    yield faculty_name, param_name, param

    def trainable_parameter_count(self) -> int:
        """Return the number of parameters the optimizer actually holds."""
        return sum(param.numel() for param in self.trainable.values())

    def frozen_parameter_count(self) -> int:
        """Return the number of `WhiteMatter` parameters phase A deliberately freezes."""
        return sum(param.numel() for param in self.frozen.values())

    # ------------------------------------------------------------------
    # The loss site
    # ------------------------------------------------------------------

    @property
    def observed_loss_site(self) -> dict[str, Any]:
        """Return the weights and temperature IN FORCE when the loss was last computed.

        This is deliberately not a read of `PhaseAConfig`. The values that matter are the
        live attributes on the modules that compute the loss -- `RankLoss.weight`,
        `UnifyLoss.weight` and `RankHead.temperature` -- captured inside `compute_loss` at
        the moment it runs. Anything that mutates one of them after arguments were parsed
        (a resumed run, a sweep driver, a caller that builds `RankHead` from one source
        and the recipe from another) changes the number this returns and therefore the
        number the receipt records. Recording the parsed argument instead would let a
        receipt describe a run that never happened.

        Returns:
            A copy of the observed record.

        Raises:
            RuntimeError: No loss has been computed yet, so there is nothing observed to
                record and a receipt built now would be describing arguments.
        """
        if self._loss_site is None:
            raise RuntimeError(
                "PhaseATrainer.observed_loss_site: no loss has been computed yet, so "
                "there is nothing in force to record; run at least one step before "
                "building a receipt"
            )
        return dict(self._loss_site)

    def pooled_targets(self, batch: PhaseABatch, out: WhiteMatterOutput) -> dict[str, Tensor]:
        """Compute `L_unify`'s target, `pool_r(h_r)`, for every region.

        Under `torch.no_grad`: the target side of `1 - cos(probe_r(f), pool_r(h_r))` is a
        fixed target produced by a frozen region, so no gradient should flow into it. That
        is a correctness statement about the loss, separate from the region freezing in
        `freeze_regions` -- both hold, and the tests check both.

        Args:
            batch: The batch whose `inputs` the regions read.
            out: The forward output, for `ctx` as actually run.

        Returns:
            `{region: [B, pooled_dim_r]}`.
        """
        targets: dict[str, Tensor] = {}
        with torch.no_grad():
            for name in self.white_matter.region_names:
                faculty = self.white_matter.faculties[name]
                h, mask = faculty.tokens(batch.inputs[name], context_tokens=out.ctx[name])
                targets[name] = faculty.pool(h, mask)
        return targets

    def compute_loss(
        self, batch: PhaseABatch, out: WhiteMatterOutput
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute `L_A = L_task + delta*L_unify` and record what was in force here.

        Args:
            batch: The batch, for its `target`.
            out: `WhiteMatter.forward`'s output, for `scores` and `probe_outputs`.

        Returns:
            `(L_A, L_task, delta*L_unify)`, all scalars.

        Raises:
            ValueError: The forward pass produced no `scores`, i.e. `inputs` carried no
                `"candidates"`, so `L_task` is not defined.
        """
        if out.scores is None:
            raise ValueError(
                "phase A: WhiteMatter produced no scores; L_task is cross-entropy over "
                "RankHead's k cosine scores, so inputs must carry 'candidates'"
            )
        l_task = self.rank_loss(out.scores, batch.target.to(out.scores.device))
        l_unify = self.unify_loss(out.probe_outputs, self.pooled_targets(batch, out))
        # Captured HERE, from the live modules, not from PhaseAConfig -- see
        # `observed_loss_site` for why that distinction is load-bearing.
        self._loss_site = {
            "task_weight": float(self.rank_loss.weight),
            "delta": float(self.unify_loss.weight),
            "rank_temperature": float(self.white_matter.rank_head.temperature),
            "k": int(self.white_matter.rank_head.k),
            "formula": "L_A = L_task + delta*L_unify",
            "source": "observed at the loss site",
        }
        return l_task + l_unify, l_task, l_unify

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, batch: PhaseABatch, *, step_index: int) -> tuple[StepRecord, WhiteMatterOutput]:
        """Take one optimizer step on `batch`.

        Args:
            batch: The training batch.
            step_index: This step's index, recorded on the returned `StepRecord`.

        Returns:
            `(record, out)` -- the step's numbers and the forward output it was taken on,
            the latter so a caller can accumulate attention mass without a second forward.
        """
        self.white_matter.train()
        self.optimizer.zero_grad(set_to_none=True)
        out = self.white_matter(batch.inputs)
        loss, l_task, l_unify = self.compute_loss(batch, out)
        loss.backward()
        if self.config.grad_clip is not None:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(list(self.trainable.values()), self.config.grad_clip)
            )
        else:
            squares = torch.zeros((), dtype=torch.float32)
            for param in self.trainable.values():
                if param.grad is not None:
                    squares = squares + (param.grad.detach().float() ** 2).sum().cpu()
            grad_norm = float(torch.sqrt(squares))
        self.optimizer.step()
        record = StepRecord(
            step=step_index,
            loss=float(loss.detach()),
            l_task=float(l_task.detach()),
            l_unify=float(l_unify.detach()),
            grad_norm=grad_norm,
        )
        return record, out

    @torch.no_grad()
    def evaluate(self, batches: Iterable[PhaseABatch]) -> dict[str, Any]:
        """Measure the composed metric on `batches` without touching the parameters.

        `compose.recall@1` is `argmax` over `RankHead`'s `k` scores against the gold index
        -- one number overall, one per bin. `null_recall` is recall on the general bin's
        `NULL` answers; `null_fpr` is, per other bin, the fraction of items wrongly ranked
        `NULL` first.

        Args:
            batches: The batches to evaluate.

        Returns:
            `{"recall_at_1", "recall_per_bin", "general_null_recall", "null_fpr_per_bin",
            "count"}`.
        """
        self.white_matter.eval()
        hits = 0
        total = 0
        per_bin_hits: dict[str, int] = {}
        per_bin_total: dict[str, int] = {}
        null_hits = 0
        null_total = 0
        fpr_hits: dict[str, int] = {}
        fpr_total: dict[str, int] = {}
        for batch in batches:
            out = self.white_matter(batch.inputs)
            if out.scores is None:
                raise ValueError("phase A evaluate: inputs carried no 'candidates'.")
            predicted = out.scores.argmax(dim=-1)
            target = batch.target.to(predicted.device)
            labels = batch.bin_labels()
            for index, label in enumerate(labels):
                correct = int(predicted[index]) == int(target[index])
                hits += int(correct)
                total += 1
                per_bin_hits[label] = per_bin_hits.get(label, 0) + int(correct)
                per_bin_total[label] = per_bin_total.get(label, 0) + 1
                if label == "general":
                    null_total += 1
                    null_hits += int(int(predicted[index]) == 0)
                else:
                    fpr_total[label] = fpr_total.get(label, 0) + 1
                    fpr_hits[label] = fpr_hits.get(label, 0) + int(int(predicted[index]) == 0)
        return {
            "recall_at_1": hits / total if total else 0.0,
            "recall_per_bin": {
                label: per_bin_hits[label] / per_bin_total[label] for label in per_bin_total
            },
            "general_null_recall": null_hits / null_total if null_total else 1.0,
            "null_fpr_per_bin": {label: fpr_hits[label] / fpr_total[label] for label in fpr_total},
            "count": total,
        }

    def run(
        self,
        train_batches: Sequence[PhaseABatch],
        dev_batches: Sequence[PhaseABatch],
        *,
        steps: int | None = None,
    ) -> PhaseAResult:
        """Run phase A for `steps` optimizer steps, then measure and gate.

        Args:
            train_batches: Cycled in order; one step per batch.
            dev_batches: The held-out half, for the overfit gate and the reported metrics.
            steps: Overrides `PhaseAConfig.steps` when given.

        Returns:
            A `PhaseAResult`, guards already evaluated.

        Raises:
            ValueError: `train_batches` is empty.
            GateFailure: G29's receipt half -- the printed floor disagrees with `eta/R`.
        """
        if not train_batches:
            raise ValueError("phase A run: train_batches is empty.")
        torch.manual_seed(self.config.seed)
        total_steps = self.config.steps if steps is None else steps
        started = time.perf_counter()

        records: list[StepRecord] = []
        mass_sum: dict[str, float] = dict.fromkeys(self.white_matter.participant_names, 0.0)
        mass_cells = 0
        buckets: dict[str, list[int]] = {
            name: [0] * HISTOGRAM_BUCKETS for name in self.white_matter.participant_names
        }
        halt_histogram: dict[str, int] = {}
        last_out: WhiteMatterOutput | None = None

        for index in range(total_steps):
            batch = train_batches[index % len(train_batches)]
            record, out = self.step(batch, step_index=index)
            records.append(record)
            last_out = out
            self._accumulate_mass(out, mass_sum, buckets)
            mass_cells += out.a.shape[0] * out.halt_at
            key = str(out.halt_at)
            halt_histogram[key] = halt_histogram.get(key, 0) + out.a.shape[0]

        assert last_out is not None  # total_steps >= 1 is enforced by the loop's inputs
        mean_per_region = {
            name: (mass_sum[name] / mass_cells if mass_cells else 0.0) for name in mass_sum
        }

        train_eval = self.evaluate(train_batches)
        dev_eval = self.evaluate(dev_batches) if dev_batches else train_eval

        guards = phase_a_guards(
            mean_per_region=mean_per_region,
            eta=self.white_matter.config.floor_eta,
            r=len(self.white_matter.participant_names),
            train_metric=float(train_eval["recall_at_1"]),
            held_out_metric=float(dev_eval["recall_at_1"]),
            general_null_recall=float(dev_eval["general_null_recall"]),
            per_bin_null_fpr=dev_eval["null_fpr_per_bin"],
        )

        return PhaseAResult(
            steps=records,
            mean_per_region=mean_per_region,
            histogram_per_region=buckets,
            train_metric=float(train_eval["recall_at_1"]),
            dev_metric=float(dev_eval["recall_at_1"]),
            recall_per_bin=dev_eval["recall_per_bin"],
            general_null_recall=float(dev_eval["general_null_recall"]),
            null_fpr_per_bin=dev_eval["null_fpr_per_bin"],
            ctx=dict(last_out.ctx),
            b=dict(last_out.b),
            halt_at_histogram=halt_histogram,
            wall_ms=(time.perf_counter() - started) * 1000.0,
            loss_site=self.observed_loss_site,
            guards=guards,
            edges=last_out.edges,
        )

    def _accumulate_mass(
        self,
        out: WhiteMatterOutput,
        mass_sum: dict[str, float],
        buckets: dict[str, list[int]],
    ) -> None:
        """Fold one forward's `a` into the running mean and histogram, active cells only.

        `WhiteMatterOutput.a` is `[B, I, R]` and is ZERO-PADDED past `halt_at` (mind.py's
        own note), so averaging over the whole second axis would silently divide a real
        mass by inactive iterations and push every region toward the floor. Only the first
        `halt_at` iterations are folded in.
        """
        active = out.a[:, : out.halt_at, :].detach()
        for index, name in enumerate(self.white_matter.participant_names):
            column = active[..., index]
            mass_sum[name] += float(column.sum())
            for value in column.reshape(-1).tolist():
                bucket = min(int(value * HISTOGRAM_BUCKETS), HISTOGRAM_BUCKETS - 1)
                buckets[name][max(bucket, 0)] += 1

    # ------------------------------------------------------------------
    # Receipts
    # ------------------------------------------------------------------

    def build_receipt(self, result: PhaseAResult, identity: Mapping[str, Any]) -> ComposeReceipt:
        """Fill every Table 7 group for this phase-A run.

        The envelope is `model-pipeline-receipt/v1` with `stage: "compose"`, and the
        `csd-metrics/v2` stamp is applied exactly once, by `write_receipt`, through
        `ComposeReceipt.write` (spec section 4, "Receipts").

        Fields phase A does not produce are written `None` rather than omitted, following
        Table 8a's treatment of `first_token_ms` and `speech_frame_ms`: a reader can then
        tell "this phase does not compute it" from "somebody forgot".

        Args:
            result: The run to describe.
            identity: Table 7's identity group, minus the two `write_receipt` stamps --
                `corpus_fingerprint`, `fingerprint_scheme`, `battery_id`, `k`, `pooling`,
                `checkpoint_sha256`, `region`, `seed`, `split_sha256`.

        Returns:
            The built `ComposeReceipt`, ready for `.write(out_dir, filename)`.
        """
        cfg = self.white_matter.config
        participants = list(self.white_matter.participant_names)
        receipt = ComposeReceipt("compose", identity)
        receipt.set_frozen_set(
            {
                "regions": [dict(row) for row in self.frozen_set],
                "participants": participants,
                "R": len(participants),
            }
        )
        receipt.set_budgets(
            {
                "B_read": cfg.budget_total_read_tokens,
                "B_kv": cfg.budget_total_kv_bytes,
                "eta": cfg.floor_eta,
                "collapse_floor": result.guards.collapse_floor,
                "token_budget": {
                    name: {
                        "min": cfg.participants[name].token_budget_min,
                        "max": cfg.participants[name].token_budget_max,
                    }
                    for name in participants
                },
                "ctx_min": {name: cfg.participants[name].ctx_min for name in participants},
                "ctx_max": {name: cfg.participants[name].ctx_max for name in participants},
                "b": result.b,
                "ctx": result.ctx,
                # Table 7 has no group for the training recipe, and section 4 requires the
                # temperature to be "a recorded temperature". It is recorded here, beside
                # the other as-run quantities, from what the loss site observed -- never
                # from PhaseAConfig. See `observed_loss_site`.
                "loss_site": result.loss_site,
                "optimizer": {
                    "name": "AdamW",
                    "lr": self.config.lr,
                    "weight_decay": self.config.weight_decay,
                    "grad_clip": self.config.grad_clip,
                    "steps": len(result.steps),
                },
                "trainable_parameters": self.trainable_parameter_count(),
                "frozen_parameters": self.frozen_parameter_count(),
            }
        )
        receipt.set_composed_metric(
            {
                "recall_at_1": {
                    "overall": result.dev_metric,
                    "train": result.train_metric,
                    "per_bin": result.recall_per_bin,
                },
                "null_recall": result.general_null_recall,
                "null_fpr": result.null_fpr_per_bin,
                # Per-pair Delta_A, Delta_B and I with block-bootstrap CIs are a W6
                # quantity over a pair of receipts; phase A produces one receipt and has
                # no pair, so this is null rather than an empty mapping that would read
                # as "measured, and nothing was found".
                "pair_deltas": None,
            }
        )
        # The seven graded baselines are a separate battery, not something a training run
        # measures. Explicitly empty, which `ComposeReceipt` distinguishes from "never
        # attached".
        receipt.set_baselines({})
        receipt.set_attention_mass(
            {
                "mean_per_region": result.mean_per_region,
                "histogram_per_region": result.histogram_per_region,
                "collapsed_in_phase_A": sorted(result.guards.collapsed_in_phase_A),
                "a_store": result.mean_per_region.get(STORE_PARTICIPANT),
                "floor_failures": result.guards.floor_failures,
            }
        )
        write_back_topology: dict[str, Any] = {
            "enabled": cfg.write_back,
            # The per-region own-bin delta is W5b's measurement, taken against a
            # write-back-off receipt; phase A is one arm of that pair, not the pair.
            "own_bin_delta": None,
            "topology": {"agreement": None, "status": "not demonstrated"},
        }
        if cfg.write_back and result.edges is not None:
            write_back_topology["edges"] = result.edges
        receipt.set_write_back_topology(write_back_topology)
        receipt.set_scheduler(
            {
                # Every field below is a phase-B/C/D quantity; the controller is bypassed
                # in phase A, so none of them is measurable here.
                "rho": None,
                "flops_ratio": None,
                "sparse_rel_delta": None,
                "phase_d_vs_c": None,
                "s_stats": None,
                "mean_iters": (
                    sum(int(k) * v for k, v in result.halt_at_histogram.items())
                    / sum(result.halt_at_histogram.values())
                    if result.halt_at_histogram
                    else None
                ),
                "halt_at_histogram": result.halt_at_histogram,
                "budget_as_tag_gap": None,
            }
        )
        receipt.set_latency(
            {
                "wall_ms_measured": result.wall_ms,
                "first_token_ms_measured": None,
                "speech_frame_miss_fraction": None,
            }
        )
        receipt.set_store(
            {
                "b_store": result.b.get(STORE_PARTICIPANT),
                "capacity_bytes": None,
                "occupancy_bytes": None,
                "active": None,
            }
        )
        receipt.set_verdicts(
            {
                "integration": result.guards.verdict(),
                "scheduling": "not evaluated: the controller is bypassed in phase A",
                "trigger_sensitivity": "not evaluated: phase A runs the dense allocation",
                # `[spec]`: Table 7's verdicts group names three strings and no fourth
                # field, so this key is this module's own naming. It is written anyway
                # because the three strings cannot tell a reader holding only the JSON
                # WHICH of Table 6 row A's gates went unchecked -- "every evaluated gate
                # clear" hedges, but does not say what was evaluated.
                "unimplemented_gates": {
                    "gates": list(result.guards.unimplemented_gates),
                    "reason": (
                        "Table 6 row A names these gates; gates.py implements no function "
                        "for them and Table 7 gives them no field. They were not "
                        "evaluated and are not claimed to have passed."
                    ),
                },
            }
        )
        receipt.set_placement_knobs({"placement": None, "knobs": None})
        return receipt

    def write_receipt(
        self,
        result: PhaseAResult,
        identity: Mapping[str, Any],
        out_dir: Path,
        filename: str = "cogsyndelta-white_matter-compose-phase-a.json",
    ) -> Path:
        """Build and write this run's receipt through `ComposeReceipt.write`.

        Args:
            result: The run to describe.
            identity: Table 7's identity group -- see `build_receipt`.
            out_dir: Directory to write into; created if missing.
            filename: Receipt filename inside `out_dir`.

        Returns:
            The path written.

        Raises:
            GateFailure: G29's receipt half -- the built receipt prints a floor that is
                not `eta/R` at its own `R`, or names no surviving region, or leaves a
                region below the floor out of `collapsed_in_phase_A`. Nothing is written:
                Table 8 row G29's effect for this half is that the receipt is REFUSED, so
                the refusal has to land before the file exists rather than after a reader
                could already have cited it.
        """
        receipt = self.build_receipt(result, identity)
        # G29's receipt half, on the path every caller actually takes. `phase_a_guards`
        # cannot make this check (see its docstring -- there is no printed floor in scope
        # during a run, only the two numbers it would be re-derived from); this is the
        # first point at which the comparison is capable of failing at all. `build()` is
        # deterministic, so running it here and again inside `write` costs a dict.
        check_receipt_collapse_floor(receipt.build())
        return receipt.write(out_dir, filename)


def loss_decreased(curve: Sequence[float], *, tail: int = 3) -> bool:
    """Return whether a loss curve fell, comparing its first and last `tail` steps.

    A single first-versus-last comparison is noise-sensitive at the handful of steps this
    module is ever run for; comparing the mean of the leading and trailing `tail` steps is
    the smallest averaging that makes "the loss decreased" a claim about the run rather
    than about two samples.

    Args:
        curve: Per-step total loss, in order.
        tail: How many steps to average at each end.

    Returns:
        `True` when the trailing mean is below the leading mean and both are finite.
    """
    if len(curve) < 2 * tail:
        return len(curve) >= 2 and curve[-1] < curve[0]
    head_mean = sum(curve[:tail]) / tail
    tail_mean = sum(curve[-tail:]) / tail
    if not (math.isfinite(head_mean) and math.isfinite(tail_mean)):
        return False
    return tail_mean < head_mean
