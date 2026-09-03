"""Benchmark battery: what industry measures, what the thesis claims, and what can lie.

THREE FAMILIES, DELIBERATELY SEPARATE

1. RANKING QUALITY -- the numbers other people can compare against. nDCG@10, MAP,
   precision@k alongside the recall@k and MRR already in eval/metrics.py. These are the
   MTEB-family metrics; reporting them means a CSD region can be put next to a published
   baseline without translation.

2. EFFICIENCY -- what it costs to get that quality. Parameters and stored bytes, but also
   latency percentiles and throughput, because a p50 alone hides the tail that decides
   whether something is usable. Deployed size is measured post-quantization, since fp32
   parameter count is not what ships.

3. REPRESENTATION HEALTH -- whether the number is real. A retrieval score can look fine
   while the embedding space has quietly collapsed into a narrow cone, and the ranking
   metrics cannot see it because they only care about relative order within a batch.
   Anisotropy, alignment, uniformity and effective rank are how that becomes visible.
   This project has already measured mean cosine above 0.995 between unrelated
   embeddings; a model in that state can still post a respectable recall@1.

CAPABILITY PER PARAMETER, AND WHY PER-BYTE MATTERS MORE
The thesis is capability at a fraction of the parameters. capability_per_param already
exists in the pretrain receipts. This adds capability_per_mb, computed from what a model
actually occupies after quantization -- a 16M-parameter fp32 encoder and the same encoder
at mixed 3-to-8 bit have identical parameter counts and a tenfold difference in what you
have to ship.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def ndcg_at_k(scores: torch.Tensor, relevant: torch.Tensor, k: int = 10) -> float:
    """Normalised discounted cumulative gain with a single relevant item per query.

    With one relevant document the ideal DCG is 1, so this reduces to the mean of
    ``1/log2(rank+1)`` over queries where the item was retrieved in the top k. Reported
    because it is what retrieval papers report; it is rank-position sensitive in a way
    recall@k is not.
    """
    if scores.numel() == 0:
        return 0.0
    k = min(k, scores.size(1))
    top = scores.topk(k, dim=1).indices
    hits = top == relevant.unsqueeze(1)
    positions = torch.arange(2, k + 2, device=scores.device, dtype=torch.float32)
    gains = (hits.float() / torch.log2(positions)).sum(dim=1)
    return float(gains.mean().item())


def average_precision(scores: torch.Tensor, relevant: torch.Tensor) -> float:
    """Mean average precision, single relevant item per query.

    With one relevant document AP is 1/rank, so MAP equals MRR here. It is reported
    separately anyway because the two diverge the moment graded or multi-positive
    relevance is introduced, and a metric that silently changes meaning is worse than a
    redundant one.
    """
    if scores.numel() == 0:
        return 0.0
    order = scores.argsort(dim=1, descending=True)
    ranks = (order == relevant.unsqueeze(1)).float().argmax(dim=1) + 1
    return float((1.0 / ranks).mean().item())


def precision_at_k(scores: torch.Tensor, relevant: torch.Tensor, k: int = 10) -> float:
    """Fraction of the top k that is relevant. With one positive, caps at 1/k."""
    if scores.numel() == 0:
        return 0.0
    k = min(k, scores.size(1))
    top = scores.topk(k, dim=1).indices
    return float((top == relevant.unsqueeze(1)).float().sum(dim=1).mean().item() / k)


def anisotropy(embeddings: torch.Tensor, sample: int = 2048, seed: int = 0) -> float:
    """Mean cosine similarity between RANDOM pairs of embeddings.

    The diagnostic for a collapsed space. In a healthy encoder unrelated items sit near
    orthogonal and this is close to 0; a degenerate one crowds everything into a narrow
    cone and this approaches 1. Retrieval metrics cannot see it, because they only compare
    a query against candidates and a uniform rotation leaves the ordering intact.

    Computed in float64: at float32 a cosine of genuinely-identical vectors returns
    1.0000001, and a threshold test against that silently inverts.
    """
    x = embeddings.detach().double()
    if x.size(0) < 2:
        return 0.0
    if x.size(0) > sample:
        g = torch.Generator().manual_seed(seed)
        x = x[torch.randperm(x.size(0), generator=g)[:sample]]
    x = F.normalize(x, dim=-1)
    sim = x @ x.T
    n = sim.size(0)
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    return float(sim[off_diagonal].mean().item())


def alignment(anchors: torch.Tensor, positives: torch.Tensor, alpha: float = 2.0) -> float:
    """Expected distance between matched pairs. Lower is better.

    Half of the Wang & Isola decomposition of what a contrastive objective optimises.
    Reported with `uniformity` because either alone is gameable: a collapsed encoder has
    perfect alignment, and a random one has excellent uniformity.
    """
    a = F.normalize(anchors.detach().float(), dim=-1)
    p = F.normalize(positives.detach().float(), dim=-1)
    return float((a - p).norm(dim=1).pow(alpha).mean().item())


def uniformity(
    embeddings: torch.Tensor, t: float = 2.0, sample: int = 2048, seed: int = 0
) -> float:
    """Log of the mean Gaussian potential over pairs. Lower means better spread.

    The other half of the decomposition. Together with `alignment` it says whether a good
    retrieval score came from a well-formed space or from a lucky arrangement.
    """
    x = F.normalize(embeddings.detach().float(), dim=-1)
    if x.size(0) < 2:
        return 0.0
    if x.size(0) > sample:
        g = torch.Generator().manual_seed(seed)
        x = x[torch.randperm(x.size(0), generator=g)[:sample]]
    sq = torch.cdist(x, x).pow(2)
    n = sq.size(0)
    off = ~torch.eye(n, dtype=torch.bool, device=sq.device)
    return float(torch.log(torch.exp(-t * sq[off]).mean()).item())


def effective_rank(embeddings: torch.Tensor, sample: int = 2048, seed: int = 0) -> float:
    """Shannon entropy of the normalised singular-value spectrum, exponentiated.

    How many dimensions the representation genuinely uses. A 256-dimensional encoder with
    an effective rank of 12 is a 12-dimensional encoder that is paying to store 256, and
    the ranking metrics will not mention it.
    """
    x = embeddings.detach().float()
    if x.size(0) < 2:
        return 0.0
    if x.size(0) > sample:
        g = torch.Generator().manual_seed(seed)
        x = x[torch.randperm(x.size(0), generator=g)[:sample]]
    x = x - x.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(x)
    p = sv / sv.sum().clamp_min(1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()).item())


def participation_ratio(embeddings: torch.Tensor, sample: int = 2048, seed: int = 0) -> float:
    """Participation-ratio effective rank: ``(sum s_i^2)^2 / sum s_i^4`` over the singular
    values of the column-centered matrix.

    A DIFFERENT quantity from :func:`effective_rank` (Shannon entropy of the LINEAR
    spectrum) -- the two disagree in sign on this project's own text regions (W1:
    participation-ratio ratios of 0.66x-1.30x against entropy ratios of 1.16x-1.84x for
    the same four checkpoints, docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md §4.0), so
    conflating them is the exact ambiguity that section's gate was written to close.
    "Both ranks are participation ratio, and the receipt says so" is the rule the W4/W7
    retrain gate states for exactly this reason -- name the definition, do not let a
    reader infer it from whichever column they see first.

    THE SAME FORMULA, BUT NOT THE SAME MEASUREMENT AS W1's OWN.
    ``docs/design/evidence/w1-token-rank-2026-09-02/measure_w1.py``'s ``pr_effective_rank``
    -- the function the W1/W1d/W4 gates were pre-committed against -- computes this exact
    ``(sum s_i^2)^2 / sum s_i^4`` quantity, but this function is NOT byte-for-byte it: this
    one subsamples to ``sample`` rows by default (W1's never subsamples: it runs the SVD on
    the full surface, however large) and returns ``0.0`` for fewer than 2 rows (W1's
    returns ``nan``, so a degenerate run cannot silently read as "rank zero"). A gate that
    needs the W1-comparable number -- the retrain gate this docstring used to claim this
    function already was -- reads :func:`pr_effective_rank` instead, the transplant of
    W1's own function that keeps both of those differences intact. This function stays
    the general-purpose, cost-bounded participation ratio existing callers (e.g.
    :func:`effective_rank`-comparable receipt fields outside the W1-lineage gates) use.

    Args:
        embeddings: ``[N, D]``. Row 0 of ``N`` is any representation -- pooled vectors,
            or every valid token position of a batch flattened into one matrix (the
            "token-global" surface the retrain gate reads).
        sample: Subsample to this many rows before the SVD, for cost control on a large
            token-global surface. Matches :func:`effective_rank`'s own default so the two
            are comparable at the same N.
        seed: Subsampling seed.

    Returns:
        A value in ``[1, min(N, D)]``; ``0.0`` for fewer than 2 rows (nothing to rank) or
        a degenerate (all-zero) matrix.
    """
    x = embeddings.detach().float()
    if x.size(0) < 2:
        return 0.0
    if x.size(0) > sample:
        g = torch.Generator().manual_seed(seed)
        x = x[torch.randperm(x.size(0), generator=g)[:sample]]
    x = x - x.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(x)
    s2 = sv.double() ** 2
    denom = (s2**2).sum()
    if denom <= 0:
        return 0.0
    return float((s2.sum() ** 2 / denom).item())


def pr_effective_rank(embeddings: torch.Tensor) -> float:
    """W1's own participation-ratio effective rank, transplanted byte-for-byte (same
    formula, same edge-case behaviour) from
    ``docs/design/evidence/w1-token-rank-2026-09-02/measure_w1.py``'s ``pr_effective_rank``
    (that file's lines 198-212) -- the function the W1, W1d and W4 retrain-gate rank
    clauses were pre-committed against, before this promotion existed. That file is frozen
    evidence (its own README: "Nothing here is regenerated on read") and is never edited
    or imported at runtime; this is the same computation given a second, importable home
    so every retrain gate after W1 reads the SAME measurement W1 itself did, rather than a
    same-shaped one that quietly diverges from it. ``tests/test_pr_effective_rank_w1_alignment.py``
    asserts this function and the frozen script's agree to 1e-6.

    Differs from :func:`participation_ratio` in exactly two ways, both deliberate:

    - No subsampling, ever. W1 measured the FULL token-global surface (however many rows
      it has), because a subsampled comparison across regions of different corpus/holdout
      shapes is not the same measurement `participation_ratio`'s ``sample`` parameter was
      built for cost control, not for this.
    - ``nan`` (not ``0.0``) for fewer than 2 rows, so a degenerate surface cannot silently
      read as "rank zero, gate failed cleanly" -- it reads as "not measured".

    Args:
        embeddings: ``[N, D]``, the full surface -- pooled vectors or the flattened
            token-global matrix. Never subsampled internally; a caller with a surface too
            large to run an exact SVD on has to subsample before calling this, deliberately
            and visibly, not have it happen inside the function.

    Returns:
        A value in ``[1, min(N, D)]``; ``nan`` for fewer than 2 rows; ``0.0`` for a
        degenerate (all-zero) matrix.
    """
    x = embeddings.detach().float()
    if x.size(0) < 2:
        return float("nan")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xc = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(xc.to(device) if device.type == "cuda" else xc)
    s2 = s.double() ** 2
    denom = (s2**2).sum()
    if denom <= 0:
        return 0.0
    num = s2.sum() ** 2
    return float((num / denom).item())


@dataclass
class LatencyProfile:
    """Inference cost, as percentiles rather than a single mean."""

    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    throughput_per_s: float = 0.0
    peak_vram_mb: float = 0.0
    samples: int = 0


def profile_latency(
    fn: Callable[[], object], warmup: int = 5, runs: int = 50, device: str = "cpu"
) -> LatencyProfile:
    """Time a callable, reporting percentiles and peak VRAM.

    Warmup runs are discarded: the first CUDA call pays kernel autotuning and allocator
    setup, and folding that into a p50 makes a fast model look slow. CUDA is synchronised
    around each timed run, without which the timings measure queue submission rather than
    execution.
    """
    cuda = device.startswith("cuda") and torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times: list[float] = []
    started = time.perf_counter()
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    wall = time.perf_counter() - started

    times.sort()

    def pct(q: float) -> float:
        if not times:
            return 0.0
        idx = min(len(times) - 1, int(q * len(times)))
        return times[idx]

    return LatencyProfile(
        p50_ms=pct(0.50),
        p95_ms=pct(0.95),
        p99_ms=pct(0.99),
        throughput_per_s=runs / wall if wall > 0 else 0.0,
        peak_vram_mb=(torch.cuda.max_memory_allocated() / 1e6) if cuda else 0.0,
        samples=runs,
    )


@dataclass
class BenchmarkResult:
    """Everything one benchmark pass measured, split by family."""

    ranking: dict[str, float] = field(default_factory=dict)
    efficiency: dict[str, float] = field(default_factory=dict)
    representation: dict[str, float] = field(default_factory=dict)

    def flat(self) -> dict[str, float]:
        """One flat dict for the receipt envelope, prefixed by family."""
        out: dict[str, float] = {}
        for family, group in (
            ("rank", self.ranking),
            ("eff", self.efficiency),
            ("repr", self.representation),
        ):
            out.update({f"{family}.{k}": v for k, v in group.items()})
        return out


def benchmark_embeddings(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    parameters: int,
    stored_bytes: int,
    latency: LatencyProfile | None = None,
) -> BenchmarkResult:
    """Score a bi-encoder over held-out pairs across all three families.

    Args:
        anchors: ``[N, D]`` query embeddings.
        positives: ``[N, D]`` matched document embeddings, aligned by index.
        parameters: Parameter count.
        stored_bytes: What the model actually occupies -- post-quantization if quantized.
        latency: Optional profile from :func:`profile_latency`.

    Returns:
        The measured result.
    """
    a = F.normalize(anchors.detach().float(), dim=-1)
    p = F.normalize(positives.detach().float(), dim=-1)
    scores = a @ p.T
    relevant = torch.arange(a.size(0), device=a.device)

    from cogsyndelta.eval.metrics import mean_reciprocal_rank, recall_at_k

    ranking = {
        "recall@1": recall_at_k(scores, relevant, 1),
        "recall@5": recall_at_k(scores, relevant, 5),
        "recall@10": recall_at_k(scores, relevant, 10),
        "mrr": mean_reciprocal_rank(scores, relevant),
        "ndcg@10": ndcg_at_k(scores, relevant, 10),
        "map": average_precision(scores, relevant),
        "precision@10": precision_at_k(scores, relevant, 10),
        "candidates": float(a.size(0)),
    }

    primary = ranking["recall@1"]
    millions = max(1e-9, parameters / 1e6)
    megabytes = max(1e-9, stored_bytes / 1e6)
    efficiency = {
        "parameters": float(parameters),
        "stored_mb": stored_bytes / 1e6,
        # The thesis metric, and its more honest sibling: parameters are not what ships.
        "capability_per_param": primary / millions,
        "capability_per_mb": primary / megabytes,
    }
    if latency is not None:
        efficiency.update(
            {
                "latency_p50_ms": latency.p50_ms,
                "latency_p95_ms": latency.p95_ms,
                "latency_p99_ms": latency.p99_ms,
                "throughput_per_s": latency.throughput_per_s,
                "peak_vram_mb": latency.peak_vram_mb,
            }
        )

    both = torch.cat([a, p])
    representation = {
        "anisotropy": anisotropy(both),
        "alignment": alignment(a, p),
        "uniformity": uniformity(both),
        "effective_rank": effective_rank(both),
        "dimensions": float(a.size(1)),
        "effective_rank_ratio": effective_rank(both) / max(1.0, float(a.size(1))),
    }
    return BenchmarkResult(ranking=ranking, efficiency=efficiency, representation=representation)
