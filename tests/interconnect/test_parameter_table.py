"""`WhiteMatter`'s measured parameter counts against spec section 2.3 Table 4 -- lane IC-8.

Table 4 is the only place the module's size is written down, and nothing checked the code
against it: the IC-R2 skeptic measured 27,522,190 core parameters at `R = 5` where the table
said 27,538,574, and 525,568 missing at `R = 4` because `W_k`/`W_v` and the controller's store
summary were not instantiated at all there. Both are now closed -- the first by amendment A1
(the workspace attention projections carry no bias, and the table says so), the second in code
(Table 4's `R = 4` row: those parameters "stay instantiated and receive no gradient in W5").

This file builds `WhiteMatter` at the REAL v1 dimensions -- `D_w = 512`, `L = 64`, `I = 4`,
`B_read = 256`, `d_ctrl = 256`, four regions at Table 1's widths -- rather than the toy numbers
the rest of the tree uses, because a parameter count is only meaningful at the dimensions the
table was derived from. The faculties are minimal stand-ins: `WhiteMatter` reads only their
`token_dim`, `pooled_dim`, `kv_bytes_per_token` and `accepts_condition` at construction, and
this file never runs a forward pass.
"""

from __future__ import annotations

import pytest
from torch import Tensor, nn

from cogsyndelta.interconnect.episodic_store import InMemoryStoreStub
from cogsyndelta.interconnect.kv_bank import STORE_PARTICIPANT
from cogsyndelta.interconnect.mind import InterconnectConfig, ParticipantSpec, WhiteMatter

#: Table 4's own totals, at the amended (A1) arithmetic.
CORE_R5 = 27_522_190
CORE_R4 = 27_521_422
#: Table 4's "trainable in phase A, heads included" row, `R = 5`.
PHASE_A_R5 = 28_376_334
#: Table 4's store-projection row, and Table 5's store-summary row.
STORE_PROJECTIONS = 524_288
STORE_SUMMARY = 1_280


class V1Faculty(nn.Module):
    """A Table 1 participant's construction-time surface, and nothing else."""

    def __init__(self, name: str, token_dim: int, kv_bytes_per_token: int) -> None:
        super().__init__()
        self.name = name
        self.faculty = name
        self.token_dim = token_dim
        self.pooled_dim = token_dim
        self.kv_bytes_per_token = kv_bytes_per_token
        self.accepts_condition = True

    def tokens(
        self, inputs: Tensor, *, context_tokens: int, condition: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Never called: this file counts parameters, it does not run the module."""
        raise NotImplementedError

    def pool(self, h: Tensor, mask: Tensor) -> Tensor:
        """Never called: this file counts parameters, it does not run the module."""
        raise NotImplementedError


V1_PARTICIPANTS = {
    # Table 4a's ctx budgets; Table 1's token budgets and phi.
    "language": ParticipantSpec(8, 96, 8, 96, 1.0),
    "memory": ParticipantSpec(8, 96, 8, 96, 1.0),
    "reasoning": ParticipantSpec(8, 256, 8, 96, 1.0),
    "visual": ParticipantSpec(8, 64, 8, 96, 2.0),
}
STORE_SPEC = ParticipantSpec(None, None, 8, 96, 0.0)


def _build(*, with_store: bool) -> WhiteMatter:
    participants = dict(V1_PARTICIPANTS)
    if with_store:
        participants[STORE_PARTICIPANT] = STORE_SPEC
    faculties = {
        "language": V1Faculty("language", 256, 4096),
        "memory": V1Faculty("memory", 256, 4096),
        "reasoning": V1Faculty("reasoning", 256, 4096),
        "visual": V1Faculty("visual", 384, 9216),
    }
    store = (
        InMemoryStoreStub(domain_enum={"general"}, half_life_s=3600.0, importance_default=0.5)
        if with_store
        else None
    )
    return WhiteMatter(InterconnectConfig(participants=participants), faculties, store)  # type: ignore[arg-type]


def _core_and_total(wm: WhiteMatter) -> tuple[int, int]:
    """`(white matter core, phase-A trainable)`, split exactly the way Table 4 splits them.

    Table 4's core row stops before the three head rows it lists separately: the rank head
    (`262,656`), the `NULL` candidate embedding (`512`, which `RankHead` owns) and the unify
    probes (`590,976`).

    Args:
        wm: The module to measure.

    Returns:
        `(core, phase_a_total)`.
    """
    total = sum(p.numel() for p in wm.parameters())
    heads = sum(p.numel() for p in wm.rank_head.parameters()) + sum(
        p.numel() for p in wm.unify_probes.parameters()
    )
    return total - heads, total


@pytest.mark.parametrize(
    ("with_store", "expected_core"), [(True, CORE_R5), (False, CORE_R4)], ids=["R=5", "R=4"]
)
def test_core_parameter_count_matches_table_4(with_store: bool, expected_core: int) -> None:
    """The measured core equals Table 4's figure at both configurations, and the `R = 4`
    row's stated `−768` (one type-embedding row of 512, one controller slot row of 256) is
    exactly the whole difference between them.
    """
    core, _ = _core_and_total(_build(with_store=with_store))
    assert core == expected_core
    assert CORE_R5 - CORE_R4 == 768


def test_phase_a_trainable_matches_table_4() -> None:
    """Table 4's "trainable in phase A, heads included, `R = 5`" row: the core plus the rank
    head, the `NULL` embedding and the unify probes.
    """
    core, total = _core_and_total(_build(with_store=True))
    assert total == PHASE_A_R5
    assert total - core == 262_656 + 512 + 590_976


def test_store_parameters_stay_instantiated_and_frozen_at_r4() -> None:
    """Table 4's `R = 4` row: "`W_k`/`W_v` and the store summary stay instantiated and receive
    no gradient in W5." Before this fix neither existed at `R = 4`, so the row's own total was
    525,568 short of anything the code could produce, and a checkpoint saved at `R = 4` could
    not be loaded into an `R = 5` module without a key mismatch.
    """
    wm = _build(with_store=False)

    assert wm.kv_bank.store_index is None
    projections = list(wm.kv_bank.store_projection.parameters())
    assert sum(p.numel() for p in projections) == STORE_PROJECTIONS
    assert all(not p.requires_grad for p in projections)

    assert wm.controller.store_summary_proj is not None
    summary = list(wm.controller.store_summary_proj.parameters())
    assert sum(p.numel() for p in summary) == STORE_SUMMARY
    assert all(not p.requires_grad for p in summary)

    frozen = sum(p.numel() for p in wm.parameters() if not p.requires_grad)
    assert frozen == STORE_PROJECTIONS + STORE_SUMMARY


def test_store_parameters_are_trainable_at_r5() -> None:
    """The mirror of the test above: at `R = 5` the same modules exist and DO receive
    gradient, so freezing them at `R = 4` is a property of the configuration and not a
    permanent state the module got stuck in.
    """
    wm = _build(with_store=True)
    assert wm.controller.store_summary_proj is None  # it is a real participant slot instead
    assert all(p.requires_grad for p in wm.kv_bank.store_projection.parameters())
    assert sum(p.numel() for p in wm.parameters() if not p.requires_grad) == 0


def test_state_dicts_agree_on_the_store_keys_across_r4_and_r5() -> None:
    """Why Table 4 keeps the parameters instantiated at `R = 4` rather than merely counting
    them: the store projections `W_k`/`W_v` carry the same state-dict keys at `R = 4` and at
    `R = 5`, so those two tensors survive the E2 transition unchanged.

    That is the whole of the claim, and it is narrower than it looks. A pre-E2 checkpoint does
    NOT plainly load into the post-E2 module: `r5.load_state_dict(r4.state_dict())` raises. The
    controller's store summary is renamed across the two configurations -- it is
    `controller.store_summary_proj.{weight,bias}` at `R = 4` and
    `controller.summary_proj.episodic_store.{weight,bias}` at `R = 5`, the same `[256, 4]` and
    `[256]` shapes under a different key. The type and slot rows then differ by design:
    `kv_bank.type_emb` goes `[4, 512]` -> `[5, 512]` and `controller.slot_embed`
    `[5, 256]` -> `[6, 256]`, which is exactly Table 4's deliberate 768-parameter difference.
    Moving a checkpoint from `R = 4` to `R = 5` therefore needs a re-key and a row extension,
    not a plain load; what this test pins is that the store projections are not part of that
    work.
    """
    keys_r4 = set(_build(with_store=False).state_dict())
    keys_r5 = set(_build(with_store=True).state_dict())
    store_keys = {k for k in keys_r5 if "store_projection" in k}
    assert store_keys
    assert store_keys <= keys_r4
