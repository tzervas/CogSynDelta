"""csd-metrics/v2 canonical-name changes to `benchmark_embeddings()`'s eval battery
(`docs/design/METRICS-METHODOLOGY.md` §3, unification-rules memo §3.1/§3.2):

1. `repr.effective_rank_entropy` is the ONLY effective-rank field written -- never a bare
   `effective_rank` and never an invented `repr.effective_rank_pr`.
2. `repr.emb_std_anchor` is new: the anchor-only collapse signal, computed by CALLING
   `representation_std()` rather than duplicating its `embeddings.std(dim=0).mean()`
   formula inline (the exact duplication `held_out.emb_std` has today, which this field is
   explicitly built to not repeat).
3. `rank.map` and `rank.precision@10` are no longer written to the flat receipt -- kept
   only as importable functions the sameness-guard tests below check directly, never as a
   `BenchmarkResult.flat()` key.

Small synthetic CPU tensors throughout -- these are unit tests of one function, not a
training-pipeline integration test (those live in the other `test_benchmark_*.py` files).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cogsyndelta.eval.benchmark import (
    average_precision,
    benchmark_embeddings,
    precision_at_k,
)
from cogsyndelta.eval.metrics import mean_reciprocal_rank, recall_at_k, representation_std


def _pairs(n: int = 16, d: int = 8, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    anchors = torch.randn(n, d, generator=g)
    # Positives correlated with, but not identical to, their anchor -- avoids a trivially
    # perfect or trivially collapsed retrieval score, so recall@1 is neither 0 nor 1 by
    # construction and the ranking metrics below are exercising real behaviour.
    positives = anchors + 0.05 * torch.randn(n, d, generator=g)
    return anchors, positives


@pytest.mark.cpu
def test_flat_receipt_writes_effective_rank_entropy_not_bare_effective_rank() -> None:
    anchors, positives = _pairs()
    result = benchmark_embeddings(anchors, positives, parameters=1000, stored_bytes=4000)
    flat = result.flat()
    assert "repr.effective_rank_entropy" in flat
    assert "repr.effective_rank" not in flat, (
        "the bare v1 name must not survive alongside the v2 one -- a card reading either "
        "key by accident would silently pick up the wrong one"
    )


@pytest.mark.cpu
def test_flat_receipt_never_writes_an_effective_rank_pr_field() -> None:
    """`participation_ratio()` stays tests-only (MM §9); this battery must never promote
    it to a receipt field, under any name."""
    anchors, positives = _pairs()
    flat = benchmark_embeddings(anchors, positives, parameters=1000, stored_bytes=4000).flat()
    assert "repr.effective_rank_pr" not in flat
    assert not any("effective_rank_pr" in key for key in flat)


@pytest.mark.cpu
def test_effective_rank_entropy_value_matches_the_named_formula() -> None:
    """Roy & Vetterli Def. 1: `exp(H(p))`, `p = sigma / sum(sigma)` over the LINEAR
    spectrum of the pooled (anchors+positives) matrix -- reproduced independently here
    rather than re-calling `effective_rank()`, so this test would catch a regression in
    that function too, not only in how `benchmark_embeddings()` wires it."""
    anchors, positives = _pairs(n=64, d=8, seed=1)
    a = F.normalize(anchors, dim=-1)
    p = F.normalize(positives, dim=-1)
    both = torch.cat([a, p])
    centered = both - both.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(centered)
    probs = sv / sv.sum().clamp_min(1e-12)
    probs = probs[probs > 0]
    expected = float(torch.exp(-(probs * probs.log()).sum()).item())

    flat = benchmark_embeddings(anchors, positives, parameters=1000, stored_bytes=4000).flat()
    assert flat["repr.effective_rank_entropy"] == expected


@pytest.mark.cpu
def test_emb_std_anchor_calls_representation_std_rather_than_duplicating_its_formula() -> None:
    """The field's value must be bit-identical to a direct `representation_std()` call on
    the SAME (L2-normalised) anchors `benchmark_embeddings()` uses internally -- proof this
    is a call, not a second hand-copied `std(dim=0).mean()` that could silently drift from
    the shared helper the way `held_out.emb_std` has (MM §11.5)."""
    anchors, positives = _pairs(n=32, d=6, seed=2)
    flat = benchmark_embeddings(anchors, positives, parameters=1000, stored_bytes=4000).flat()

    normalised_anchors = F.normalize(anchors.float(), dim=-1)
    expected = representation_std(normalised_anchors)
    assert flat["repr.emb_std_anchor"] == expected


@pytest.mark.cpu
def test_emb_std_anchor_is_anchor_only_not_pooled_with_positives() -> None:
    """Distinguishes the anchor-only pool from every other `repr.*` field in this battery,
    which pools anchors+positives (`pooled_both`) -- MM §2.3(d)/(f)."""
    anchors, positives = _pairs(n=32, d=6, seed=3)
    flat = benchmark_embeddings(anchors, positives, parameters=1000, stored_bytes=4000).flat()

    normalised_anchors = F.normalize(anchors.float(), dim=-1)
    normalised_positives = F.normalize(positives.float(), dim=-1)
    pooled = torch.cat([normalised_anchors, normalised_positives])

    anchor_only = representation_std(normalised_anchors)
    pooled_value = representation_std(pooled)
    assert flat["repr.emb_std_anchor"] == anchor_only
    # Not a mathematical law that these must differ, but true for these seeded, jittered
    # pairs, and a regression that silently swapped the pool would land on the pooled
    # value instead -- catches that swap directly rather than only checking the anchor
    # side matches (which a copy-paste bug computing BOTH values could pass by accident).
    assert anchor_only != pooled_value


@pytest.mark.cpu
def test_flat_receipt_no_longer_writes_map_or_precision_at_10() -> None:
    anchors, positives = _pairs()
    flat = benchmark_embeddings(anchors, positives, parameters=1000, stored_bytes=4000).flat()
    assert "rank.map" not in flat
    assert "rank.precision@10" not in flat
    # Everything else in the ranking family stays.
    for key in ("rank.recall@1", "rank.recall@5", "rank.recall@10", "rank.mrr", "rank.ndcg@10"):
        assert key in flat


@pytest.mark.cpu
def test_map_still_equals_mrr_on_the_closed_pool_sameness_guard() -> None:
    """MM §3.4(f): with exactly one relevant item per query, MAP == MRR by construction.
    `average_precision()` and `mean_reciprocal_rank()` stay importable specifically so this
    identity is still checked even though neither is written to a receipt any more."""
    anchors, positives = _pairs(n=48, d=10, seed=4)
    a = F.normalize(anchors, dim=-1)
    p = F.normalize(positives, dim=-1)
    scores = a @ p.T
    relevant = torch.arange(a.size(0))
    assert average_precision(scores, relevant) == mean_reciprocal_rank(scores, relevant)


@pytest.mark.cpu
def test_precision_at_10_still_equals_recall_at_10_over_10_sameness_guard() -> None:
    """MM §3.5(f): with one relevant item, precision@10 == recall@10 / 10 exactly."""
    anchors, positives = _pairs(n=48, d=10, seed=5)
    a = F.normalize(anchors, dim=-1)
    p = F.normalize(positives, dim=-1)
    scores = a @ p.T
    relevant = torch.arange(a.size(0))
    assert precision_at_k(scores, relevant, 10) == recall_at_k(scores, relevant, 10) / 10
