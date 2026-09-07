"""Shared fixtures for the interconnect test tree -- lane IC-8.

Spec `docs/design/INTERCONNECT-MODULE-SPEC.md` section 5's integration-test paragraph:
"builds two fake faculties that satisfy the W0 protocol, `FakeText(token_dim 16, pooled_dim
16, kv_bytes_per_token 64, accepts_condition True)` and `FakeVisual(token_dim 24, ...)`,
plus the store stub, at `D = 64`, `L = 8`, `n_iter = 2`, `B_read = 16`, `k = 4`."
`FakeText`/`FakeVisual` and the toy `InterconnectConfig` below are exactly that
configuration; every value the spec leaves as `...` is this file's own smallest honest
choice, recorded per fixture.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from cogsyndelta.interconnect.episodic_store import InMemoryStoreStub, Scope, derive_scope
from cogsyndelta.interconnect.mind import InterconnectConfig, ParticipantSpec, WhiteMatter


class FakeText(nn.Module):
    """A minimal `Faculty` (spec Table 1's `language`-shaped participant), trainable
    only insofar as its params exist -- the fixture freezes them, matching "regions
    arrive frozen through the W0 protocol". `token_dim=16`, `pooled_dim=16`,
    `kv_bytes_per_token=64`, `accepts_condition=True`, per the spec's own text.
    """

    def __init__(self, vocab: int = 64) -> None:
        super().__init__()
        self.name = "language"
        self.faculty = "language"
        self.token_dim = 16
        self.pooled_dim = 16
        self.kv_bytes_per_token = 64
        self.accepts_condition = True
        self.embed = nn.Embedding(vocab, self.token_dim)
        self.cond_proj = nn.Linear(self.token_dim, self.token_dim)

    def tokens(
        self, inputs: Tensor, *, context_tokens: int, condition: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """`inputs`: `[B, T_in]` int64 ids. Truncates to `context_tokens` (DEC-15: never
        encodes more than its budget). When `condition` is given, its mean over the
        `n_cond` axis is projected and added to every position -- a real, differentiable
        effect on `h`, deliberately, so the integration test's "write-back changes `h_r`"
        assertion has something to observe (`faculty/protocol.py`'s own real adapters
        raise on a non-`None` condition instead; this fake is built to consume one).
        """
        ids = inputs[:, :context_tokens]
        h = self.embed(ids)
        mask = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
        if condition is not None:
            h = h + self.cond_proj(condition.mean(dim=1, keepdim=True))
        return h, mask

    def pool(self, h: Tensor, mask: Tensor) -> Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)
        return (h * mask.unsqueeze(-1)).sum(dim=1) / denom


class FakeVisual(nn.Module):
    """The spec's second fake faculty, `FakeVisual(token_dim 24, ...)`. `[lane]`
    choices for the fields the spec elides: `pooled_dim=24` (no `proj` narrowing, same
    convention `visual`'s own docstring names for the real region), `kv_bytes_per_token
    = 96` (a round number distinct from `FakeText`'s 64, so a test can tell the two
    apart in a receipt), `accepts_condition = False` (so the integration test also
    exercises a region write-back never touches, per spec Table 1's own mix of
    `accepts_condition` values).
    """

    def __init__(self, raw_dim: int = 32) -> None:
        super().__init__()
        self.name = "visual"
        self.faculty = "visual"
        self.token_dim = 24
        self.pooled_dim = 24
        self.kv_bytes_per_token = 96
        self.accepts_condition = False
        self.proj = nn.Linear(raw_dim, self.token_dim)

    def tokens(
        self, inputs: Tensor, *, context_tokens: int, condition: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """`inputs`: `[B, T_in, raw_dim]` float (a toy stand-in for patch features)."""
        if condition is not None:
            raise TypeError("FakeVisual.accepts_condition is False; condition must be None.")
        patches = inputs[:, :context_tokens]
        h = self.proj(patches)
        mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        return h, mask

    def pool(self, h: Tensor, mask: Tensor) -> Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)
        return (h * mask.unsqueeze(-1)).sum(dim=1) / denom


@pytest.fixture
def pinned_threads() -> Iterator[None]:
    """Pin intra-op threads to 1 for the test, then restore the previous count.

    WHY, MEASURED. Float reduction order changes with the thread count, and phase A's
    optimiser amplifies it: a ~1e-7 forward perturbation becomes ~3e-5 of weight delta in
    one AdamW step and ~100x that per 50 steps, which is how one seed and one command line
    produced 11 distinct `checkpoint_sha256` across thread counts. Downstream, the same
    seed puts `dev_recall_at_1` at 1.0000 for `OMP_NUM_THREADS` 1, 2, 6 and 8 and 0.9375
    for 3 and 4, and `mean(a_store)` at 0.0642 / 0.0914 / 0.1177 / 0.0468 / 0.0736 for
    1 / 2 / 4 / 8 / 16 -- only the 8-thread draw is below G29's floor.

    So a gate assertion run unpinned is measuring the host's core count. The thread axis
    is a nuisance parameter to PIN; the SEED axis is the one that carries real variance
    and the one `replicate_verdict` samples. This fixture does the first and says so; it
    is not a substitute for the second.

    `torch.set_num_threads` is process-wide, so the previous value is restored on the way
    out rather than left changed for whatever runs next in the same worker.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def fake_faculties() -> dict[str, nn.Module]:
    """`{"language": FakeText(), "visual": FakeVisual()}`, every parameter frozen (W0:
    regions arrive frozen).
    """
    faculties: dict[str, nn.Module] = {"language": FakeText(), "visual": FakeVisual()}
    for fac in faculties.values():
        for p in fac.parameters():
            p.requires_grad_(False)
    return faculties


@pytest.fixture
def fake_store() -> InMemoryStoreStub:
    """The E0 stub, one domain, matching the spec's "plus the store stub."""
    return InMemoryStoreStub(domain_enum={"general"}, half_life_s=3600.0, importance_default=0.5)


@pytest.fixture
def toy_config() -> InterconnectConfig:
    """`D_w=64, L=8, n_iter=2, B_read=16, k=4` (spec's own integration-test numbers),
    with `episodic_store` admitted (`R=3`) so the store-load-bearing assertions have a
    third participant to exercise. Every `ParticipantSpec` bound is a small, internally
    consistent `[lane]` choice: `ScheduleValidator`/`ThalamicController`'s own
    construction-time refusals (spec section 3 step 2) are satisfied at these numbers
    (`B_kv` generously above every participant's `eta/R_ctx` floor share; token-budget
    floors summing well under `B_read=16`, ceilings well over it).
    """
    participants = {
        "language": ParticipantSpec(
            ctx_min=2, ctx_max=8, token_budget_min=2, token_budget_max=8, phi=1.0
        ),
        "visual": ParticipantSpec(
            ctx_min=2, ctx_max=6, token_budget_min=2, token_budget_max=8, phi=1.5
        ),
        "episodic_store": ParticipantSpec(
            ctx_min=None, ctx_max=None, token_budget_min=2, token_budget_max=8, phi=0.0
        ),
    }
    return InterconnectConfig(
        participants=participants,
        workspace_dim=64,
        latents=8,
        n_iter=2,
        heads=4,
        mlp_ratio=2,
        budget_total_read_tokens=16,
        budget_total_kv_bytes=100_000,
        floor_eta=0.15,
        n_cond=4,
        controller_dim=32,
        controller_depth=1,
        controller_heads=2,
        write_back=True,
        k_candidates=4,
        rank_temperature=1.0,
        allowed_modalities=("text",),
        resident_heads=("text",),
    )


@pytest.fixture
def white_matter(
    toy_config: InterconnectConfig,
    fake_faculties: dict[str, nn.Module],
    fake_store: InMemoryStoreStub,
) -> WhiteMatter:
    """A `WhiteMatter` built from `toy_config` over `fake_faculties` and `fake_store`."""
    return WhiteMatter(toy_config, fake_faculties, fake_store)  # type: ignore[arg-type]


@pytest.fixture
def toy_scope() -> Scope:
    """One server-derived `Scope` every toy request writes/reads under."""
    return derive_scope("test-principal", session="s1")


def make_toy_inputs(
    batch_size: int, *, scope: Scope | None, seed: int = 0, k: int = 4
) -> dict[str, Any]:
    """Build one request's `inputs` mapping for `WhiteMatter.forward` (module docstring's
    "What `inputs` looks like"): text ids, image patches, `k-1` candidate embeddings, and
    a broadcast `scope`/`domain`/`logical_key` for the store.
    """
    gen = torch.Generator().manual_seed(seed)
    text_ids = torch.randint(0, 64, (batch_size, 8), generator=gen)
    image_patches = torch.randn(batch_size, 6, 32, generator=gen)
    candidates = torch.randn(batch_size, k - 1, 64, generator=gen)
    return {
        "language": text_ids,
        "visual": image_patches,
        "candidates": candidates,
        "scope": scope,
        "domain": "general",
        "logical_key": "turn-0",
    }
