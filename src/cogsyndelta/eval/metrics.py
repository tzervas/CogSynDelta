"""Evaluation metrics, and the contamination guards that make them mean anything.

WHY THIS EXISTS
The operator's constraint is that a change must give a net efficiency gain "without
decreasing quality and/or increasing perplexity". That is only checkable against a
baseline, and this repo has none: every entry in benchmark_results carries
``"quality": {}``. Without a measurement, "no quality loss" is not a claim -- it is a
hope.

THE FAILURE MODE THIS IS BUILT AGAINST
The easiest way to poison a model is not a broken layer; it is a contaminated eval. If
even a fraction of the eval set appears in training, every number improves, the model
looks better than it is, and nothing in the training loop can detect it. That is
indistinguishable from real progress right up until the model meets data it has genuinely
not seen.

So :func:`assert_no_contamination` is not a nicety here. It is the difference between a
measurement and a story.

WHAT IS DELIBERATELY NOT HERE
No "quality score" that blends unrelated numbers into one figure. A single blended score
hides exactly the trade a small model must be judged on -- perplexity moving one way while
retrieval moves the other is the interesting case, and averaging destroys it.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from typing import NamedTuple, TypedDict

import torch


class ContaminationReport(TypedDict):
    """Overlap between a train and an eval set."""

    train_unique: int
    eval_unique: int
    overlap: int
    eval_fraction_contaminated: float
    examples: list[str]


class ChannelOverlap(TypedDict):
    """Train/eval overlap measured through ONE key, plus what that key can see."""

    key: str
    gated: bool
    eval_unique: int
    overlap: int
    eval_fraction_contaminated: float
    examples: list[str]


class PairContaminationReport(TypedDict):
    """Multi-channel overlap between a train and an eval set of (anchor, positive) pairs.

    `train_pairs_removed` is the count of TRAINING pairs dropped for colliding with the
    holdout on a gated channel. It is separate from the overlap figures on purpose: the
    overlaps are what was MEASURED, before anything was done about it, so a receipt
    records the contamination that existed rather than only the state after cleanup.
    """

    train_pairs_seen: int
    train_pairs_removed: int
    eval_pairs: int
    gated_channels: list[str]
    channels: dict[str, ChannelOverlap]
    eval_duplicate_positives: int


def token_weighted_perplexity(losses: Sequence[float], token_counts: Sequence[int]) -> float:
    """Perplexity weighted by tokens, not by batch.

    Averaging per-batch losses lets a short trailing batch carry the same weight as a full
    one, which shifts the number for a reason that has nothing to do with the model.

    KEPT, NOT DELETED (checked at v2 patch time): the ONLY call site that is not this
    module's own test is `CausalLM.estimate_perplexity()`
    (`src/cogsyndelta/model/causal_lm.py`), which computes the identical token-weighted
    formula INLINE (`total_nll / total_tokens`, then `exp`) rather than calling this
    function -- the same "shared helper exists, but the real call site duplicates its
    formula instead of using it" pattern `held_out.emb_std` has relative to
    `representation_std()` (`docs/design/METRICS-METHODOLOGY.md` §11.5). That file is
    outside this lane's scope to edit, so the duplication was not wired closed here;
    deleting this function anyway would sever the one place a future receipt `perplexity`
    field is documented to derive its formula from
    (`docs/design/METRICS-METHODOLOGY.md` §11.5's own words: "if a future card or receipt
    does show a `perplexity` field, its formula is `token_weighted_perplexity()`'s"). Left
    in place, tested, and exported; a follow-up in the lane owning
    `src/cogsyndelta/model/causal_lm.py` should change `estimate_perplexity()` to call this
    function instead of duplicating it.

    Args:
        losses: Mean cross-entropy per batch, in nats.
        token_counts: Target token count per batch.

    Returns:
        Perplexity over all tokens.

    Raises:
        ValueError: If the sequences disagree in length or no tokens were seen.
    """
    if len(losses) != len(token_counts):
        raise ValueError(f"{len(losses)} losses vs {len(token_counts)} counts")
    total = sum(token_counts)
    if total == 0:
        raise ValueError("no tokens: refusing to report a perplexity over an empty set")
    nll = sum(loss * n for loss, n in zip(losses, token_counts, strict=True))
    return math.exp(nll / total)


def _normalise(text: str) -> str:
    """Collapse whitespace and case. The one normalisation every exact key shares."""
    return " ".join(text.split()).lower()


def _digest(payload: str) -> str:
    """blake2b-128 of a UTF-8 payload, hex. One spelling, so keys cannot drift apart."""
    return hashlib.blake2b(payload.encode("utf-8", "replace"), digest_size=16).hexdigest()


def _fingerprint(text: str) -> str:
    """Stable hash of normalised text, for overlap detection.

    Normalises whitespace and case so that trivial reformatting cannot hide a duplicate --
    contamination usually arrives via a reformatted copy, not a byte-identical one.
    """
    return _digest(_normalise(text))


def _unordered(first: str, second: str) -> str:
    """Hash two keys so that (a, b) and (b, a) collide.

    Similarity is symmetric and the training objective here is symmetric InfoNCE, which
    trains (a, b) and (b, a) in the same step. An ordered key would therefore miss half
    of every duplicated pair.
    """
    lo, hi = sorted((first, second))
    return _digest(f"{lo}\x00{hi}")


def pair_fingerprint(left: str, right: str) -> str:
    """Order-independent fingerprint of a sentence pair, whitespace- and case-normalised.

    Args:
        left: One side of the pair.
        right: The other side.

    Returns:
        A hex digest identifying the unordered pair.
    """
    return _unordered(_normalise(left), _normalise(right))


_FUNCTION_WORDS = frozenset(
    # determiners and deixis
    ["a", "an", "the", "this", "that", "these", "those", "there", "here"]
    # copulas, auxiliaries and modals
    + ["is", "are", "was", "were", "be", "been", "being", "am"]
    + ["do", "does", "did", "doing", "done", "have", "has", "had", "having"]
    + ["can", "could", "will", "would", "shall", "should", "may", "might", "must"]
    # prepositions and conjunctions
    + ["of", "in", "on", "at", "to", "from", "by", "for", "with", "without"]
    + ["about", "into", "onto", "over", "under", "and", "or", "but", "nor"]
    + ["so", "yet", "if", "then", "than", "as", "because", "while"]
    # pronouns
    + ["i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"]
    + ["my", "your", "his", "its", "our", "their"]
    # interrogatives and common adverbs
    + ["what", "which", "who", "whom", "whose", "when", "where", "why", "how"]
    + ["not", "no", "too", "very", "just", "also", "only"]
    # contraction tails left behind by the word regex ("don't" -> "don", "t")
    + ["s", "t", "re", "ve", "ll", "d", "m", "o", "y"]
)
"""English function words, stripped before the CONTENT-word key below.

Deliberately closed and small: it is a fixed list of grammatical glue, not a tuned
parameter. Growing it makes the content key coarser (more things collide) and shrinking
it makes it finer, so it is the kind of knob that quietly changes what a guard means --
it should move only with a stated reason.
"""

_WORD_RE = re.compile(r"[a-z0-9]+")


def _content_fingerprint(text: str) -> str:
    """Hash the SET of content words, discarding order, repetition and function words.

    This exists to be a genuinely different normalisation from :func:`_fingerprint`,
    because a guard whose key equals the key its caller already deduplicated on cannot
    fire. Exact normalisation says "how do you know if a mango is ripe" and "how to know
    if a mango is ripe" are different texts. This says they are the same claim: both
    reduce to {know, mango, ripe}.

    It deliberately does NOT collapse the corpus survey's other near-duplicate category.
    "44 is 25 percent of what number?" and "44 is 55 percent of what number?" have the
    same template and DIFFERENT correct answers; the digits are content words, so the two
    keys differ and the pair is left alone. Treating those as duplicates would hide a real
    weakness (the encoder failing to read a slot value) rather than fix a leak.

    Args:
        text: Any text.

    Returns:
        A hex digest over the sorted content-word set. Falls back to the full word set,
        then to the normalised text, so an all-function-word text ("how are you") keys on
        itself instead of colliding with every other function-word-only text.
    """
    normalised = _normalise(text)
    words = set(_WORD_RE.findall(normalised))
    content = sorted(words - _FUNCTION_WORDS) or sorted(words) or [normalised]
    return _digest("\x00".join(content))


def contamination_report(
    train_texts: Iterable[str], eval_texts: Iterable[str]
) -> ContaminationReport:
    """Measure overlap between a train and an eval set.

    Args:
        train_texts: Training documents.
        eval_texts: Held-out documents.

    Returns:
        Counts, the overlap fraction of the eval set, and up to five example
        fingerprints so a hit can actually be chased down rather than merely reported.
    """
    train_fp = {_fingerprint(t) for t in train_texts}
    eval_fp_list = [_fingerprint(t) for t in eval_texts]
    eval_fp = set(eval_fp_list)
    overlap = train_fp & eval_fp
    return ContaminationReport(
        train_unique=len(train_fp),
        eval_unique=len(eval_fp),
        overlap=len(overlap),
        eval_fraction_contaminated=(len(overlap) / len(eval_fp)) if eval_fp else 0.0,
        examples=sorted(overlap)[:5],
    )


def assert_no_contamination(
    train_texts: Iterable[str], eval_texts: Iterable[str], *, tolerance: float = 0.0
) -> ContaminationReport:
    """Raise if the eval set overlaps training beyond ``tolerance``.

    Default tolerance is zero. A non-zero tolerance should be a deliberate, argued choice
    for a specific corpus -- not a way to make a failing check pass.

    Args:
        train_texts: Training documents.
        eval_texts: Held-out documents.
        tolerance: Maximum acceptable contaminated fraction of the eval set.

    Returns:
        The contamination report, when within tolerance.

    Raises:
        ValueError: If contamination exceeds ``tolerance``.
    """
    report = contamination_report(train_texts, eval_texts)
    fraction = report["eval_fraction_contaminated"]
    if fraction > tolerance:
        raise ValueError(
            f"eval set is {fraction:.2%} contaminated ({report['overlap']} of "
            f"{report['eval_unique']} documents also appear in training; tolerance "
            f"{tolerance:.2%}). Every metric measured against it is inflated."
        )
    return report


_CHANNEL_MEANING = {
    "anchor_exact": (
        "the eval anchor, whitespace- and case-normalised. VACUOUS whenever the caller "
        "deduplicated on this same key -- reported so a zero here is never mistaken for "
        "evidence."
    ),
    "positive_exact": (
        "the eval POSITIVE, normalised. Anchor-only dedup never touches this side, so it "
        "can fire. Reported rather than gated: a passage shared across two genuinely "
        "different queries is normal in IR and only leaks if the queries are also close."
    ),
    "pair_exact": (
        "the whole pair, order-independent. A held-out pair that is also a training pair "
        "in either direction -- symmetric InfoNCE trains both directions, so a swap is "
        "the same leak. Unambiguous memorisation."
    ),
    "anchor_content": (
        "the eval anchor's content-word SET (function words, order and repetition "
        "discarded). Catches the paraphrase family exact normalisation cannot see."
    ),
    "positive_content": "the eval positive's content-word set.",
    "pair_content": (
        "BOTH sides content-word-identical to a training pair. Function-word paraphrase "
        "of an entire training example, which is memorisation with the wording changed."
    ),
}

_GATED_CHANNELS = ("pair_exact", "pair_content")
"""Channels that stop a run rather than merely being counted.

Both are PAIR-level and neither is implied by anchor-level dedup, so this guard can
actually fire -- which is the whole point. The single-side channels are reported instead
of gated because whether a shared anchor or a shared passage is leakage depends on the
corpus, and a guard that halts every legitimate run is a guard whose tolerance gets
raised until it means nothing again.
"""


def _channel_keys(anchor: str, positive: str) -> dict[str, str]:
    """The six overlap keys for one pair. One place, so both sides key identically."""
    a_exact, p_exact = _fingerprint(anchor), _fingerprint(positive)
    a_content, p_content = _content_fingerprint(anchor), _content_fingerprint(positive)
    return {
        "anchor_exact": a_exact,
        "positive_exact": p_exact,
        "pair_exact": _unordered(_normalise(anchor), _normalise(positive)),
        "anchor_content": a_content,
        "positive_content": p_content,
        "pair_content": _unordered(a_content, p_content),
    }


def _index_eval(
    eval_pairs: Iterable[tuple[str, str]],
) -> tuple[dict[str, dict[str, str]], int, int]:
    """Index the eval side by every channel key, mapping each key to a readable example.

    Returns:
        ``(keys, n_pairs, duplicate_positives)``. ``keys[channel][key]`` is a truncated
        eval anchor, so a hit reports text a human can chase rather than a bare digest.
        ``duplicate_positives`` counts held-out pairs whose positive is not unique WITHIN
        the holdout -- those cap recall@1 by construction, since two identical candidates
        cannot both be ranked first.
    """
    keys: dict[str, dict[str, str]] = {name: {} for name in _CHANNEL_MEANING}
    positives: set[str] = set()
    n_pairs = 0
    duplicate_positives = 0
    for anchor, positive in eval_pairs:
        n_pairs += 1
        channel_keys = _channel_keys(anchor, positive)
        for name, key in channel_keys.items():
            keys[name].setdefault(key, anchor[:160])
        if channel_keys["positive_exact"] in positives:
            duplicate_positives += 1
        positives.add(channel_keys["positive_exact"])
    return keys, n_pairs, duplicate_positives


class PairScan(NamedTuple):
    """One pass of the train side against an indexed eval side.

    An implementation detail of the two public entry points below, not exported: it
    exists so that measuring and repairing share a single pass over a half-million pairs
    rather than keying every one of them twice.
    """

    eval_index: dict[str, dict[str, str]]
    hits: dict[str, dict[str, str]]
    contaminated: list[int]
    train_seen: int
    eval_pairs: int
    eval_duplicate_positives: int


def _scan(
    train_pairs: Iterable[tuple[str, str]], eval_pairs: Iterable[tuple[str, str]]
) -> PairScan:
    """Stream the train side once, recording every channel hit and which rows to drop.

    The train side is never held in this function: only membership in the eval-side key
    index and the POSITIONS of contaminated rows are kept, so this costs O(eval) memory
    against a half-million-pair training set.
    """
    index, n_eval, duplicate_positives = _index_eval(eval_pairs)
    hits: dict[str, dict[str, str]] = {name: {} for name in index}
    contaminated: list[int] = []
    seen = 0
    for position, (anchor, positive) in enumerate(train_pairs):
        seen += 1
        gated_hit = False
        for name, key in _channel_keys(anchor, positive).items():
            example = index[name].get(key)
            if example is None:
                continue
            hits[name][key] = example
            gated_hit = gated_hit or name in _GATED_CHANNELS
        if gated_hit:
            contaminated.append(position)
    return PairScan(index, hits, contaminated, seen, n_eval, duplicate_positives)


def _report(scan: PairScan, removed: int = 0) -> PairContaminationReport:
    """Turn a scan into the reportable per-channel figures."""
    channels: dict[str, ChannelOverlap] = {
        name: ChannelOverlap(
            key=_CHANNEL_MEANING[name],
            gated=name in _GATED_CHANNELS,
            eval_unique=len(scan.eval_index[name]),
            overlap=len(scan.hits[name]),
            eval_fraction_contaminated=(
                len(scan.hits[name]) / len(scan.eval_index[name]) if scan.eval_index[name] else 0.0
            ),
            examples=sorted(scan.hits[name].values())[:5],
        )
        for name in scan.eval_index
    }
    return PairContaminationReport(
        train_pairs_seen=scan.train_seen,
        train_pairs_removed=removed,
        eval_pairs=scan.eval_pairs,
        gated_channels=list(_GATED_CHANNELS),
        channels=channels,
        eval_duplicate_positives=scan.eval_duplicate_positives,
    )


def pair_contamination_report(
    train_pairs: Iterable[tuple[str, str]], eval_pairs: Iterable[tuple[str, str]]
) -> PairContaminationReport:
    """Measure train/eval overlap through several keys, not one.

    WHY THIS EXISTS RATHER THAN :func:`contamination_report`
    The single-key version was called with the same normalisation the caller had already
    deduplicated on, over the same field. Dedup guarantees those keys are unique; a split
    partitions unique keys; so the intersection was empty BY CONSTRUCTION, for every
    region, always. It reported zero while an independent survey measured 53.7% near-
    duplicate holdout leakage in `retrieve`. A guard has to key on something its caller
    has NOT already eliminated, or it is only confirming its own arithmetic.

    The train side is streamed and never held: only membership in the eval-side key index
    is retained, so this costs O(eval) memory against a half-million-pair training set.
    That is why the report says ``train_pairs_seen`` rather than a train-unique count.

    Args:
        train_pairs: (anchor, positive) pairs the model will train on.
        eval_pairs: (anchor, positive) pairs held out.

    Returns:
        Per-channel counts, which channels are gated, and the number of held-out pairs
        sharing a positive with another held-out pair.
    """
    return _report(_scan(train_pairs, eval_pairs))


def assert_no_pair_contamination(
    train_pairs: Iterable[tuple[str, str]],
    eval_pairs: Iterable[tuple[str, str]],
    *,
    tolerance: float = 0.0,
) -> PairContaminationReport:
    """Raise if a GATED channel shows the eval set overlapping training beyond ``tolerance``.

    Args:
        train_pairs: (anchor, positive) pairs the model will train on.
        eval_pairs: (anchor, positive) pairs held out.
        tolerance: Maximum acceptable contaminated fraction, per gated channel.

    Returns:
        The full multi-channel report, when every gated channel is within tolerance.

    Raises:
        ValueError: If a gated channel exceeds ``tolerance``.
    """
    report = pair_contamination_report(train_pairs, eval_pairs)
    for name in report["gated_channels"]:
        channel = report["channels"][name]
        if channel["eval_fraction_contaminated"] > tolerance:
            raise ValueError(
                f"contamination on channel {name!r}: {channel['eval_fraction_contaminated']:.2%} "
                f"of the held-out set ({channel['overlap']} of {channel['eval_unique']}) also "
                f"appears in training; tolerance {tolerance:.2%}. This channel matches on "
                f"{_CHANNEL_MEANING[name]} Anchor-level dedup does not remove it, so this is a "
                f"real leak rather than a bookkeeping artefact. Examples: {channel['examples']}"
            )
    return report


REMOVAL_CEILING = 0.01
"""Ceiling on the fraction of TRAINING that may be deleted to clean a split.

The quantity matters. Removing leaked TRAINING rows is a repair: the held-out set is
untouched, the leak is genuinely gone, and the count goes in the receipt. This repo
already does exactly that one level up -- `_prepare_graded` drops the STS-B pairs that
are also AllNLI training positives rather than tolerating them, because "raising a
tolerance makes a number look clean without making it clean".

Measured on the real corpora at a 512-pair holdout: `code` leaks nothing, `compress`
leaks 1 held-out pair, `retrieve` leaks 35. Even the worst of those deletes on the order
of 0.01% of a half-million-pair training set. That is a stray-row repair.

What the ceiling is actually guarding against is a corpus so duplicated that deleting the
leaks guts the training set -- at which point the split is not repairable by deletion and
the honest answer is to stop. Expressing the ceiling as a share of the EVAL set would
have refused `retrieve` outright over 35 rows, which teaches an operator to raise the
tolerance and puts us back where this file started.
"""


def screen_pair_contamination(
    train_pairs: Sequence[tuple[str, str]],
    eval_pairs: Sequence[tuple[str, str]],
    *,
    removal_ceiling: float = REMOVAL_CEILING,
) -> tuple[list[tuple[str, str]], PairContaminationReport]:
    """Measure contamination, then drop the training rows that cause it.

    Order matters and is the point: the report describes what was MEASURED, before the
    removal, so a receipt records the contamination that existed rather than the tidy
    state afterwards. A reader can see that `retrieve` leaked 6.84% of its holdout and
    that those rows were removed; a report generated after cleanup would show a zero
    indistinguishable from a corpus that never leaked -- which is exactly what made the
    previous anchor-only guard worthless.

    Only the GATED channels cause removal, and only TRAINING rows are ever removed. The
    single-side channels are ambiguous (a passage answering two different queries is
    normal in IR; two paraphrased queries with different answers is a real weakness the
    metric SHOULD see), so acting on them would edit the corpus on a judgement this code
    is not entitled to make. They are reported instead.

    Args:
        train_pairs: Pairs the model would train on.
        eval_pairs: The held-out pairs.
        removal_ceiling: Fraction of TRAINING above which this refuses rather than
            cleans. See :data:`REMOVAL_CEILING`.

    Returns:
        `(kept_train_pairs, report)`. The report's overlap figures are pre-removal.

    Raises:
        ValueError: If cleaning would delete more than ``removal_ceiling`` of training --
            at that scale the corpus is duplicated and the split is not repairable by
            deletion.
    """
    scan = _scan(train_pairs, eval_pairs)
    report = _report(scan, removed=len(scan.contaminated))
    share = len(scan.contaminated) / max(1, scan.train_seen)
    if share > removal_ceiling:
        fired = [
            f"{name} {report['channels'][name]['eval_fraction_contaminated']:.2%} of the holdout"
            for name in _GATED_CHANNELS
            if report["channels"][name]["overlap"]
        ]
        raise ValueError(
            f"refusing to train: cleaning the held-out leak would delete "
            f"{len(scan.contaminated)} of {scan.train_seen} training pairs ({share:.2%}), "
            f"above the {removal_ceiling:.2%} ceiling. Channels: {'; '.join(fired)}. "
            f"Removing a stray row is a repair; removing this much is deleting the corpus "
            f"until the eval looks clean. The split itself is wrong. Examples: "
            f"{report['channels'][_GATED_CHANNELS[-1]]['examples']}"
        )
    drop = set(scan.contaminated)
    kept = [pair for position, pair in enumerate(train_pairs) if position not in drop]
    return kept, report


def recall_at_k(scores: torch.Tensor, relevant: torch.Tensor, k: int) -> float:
    """Fraction of queries whose relevant item appears in the top ``k``.

    Args:
        scores: ``[B, N]``, higher is better.
        relevant: ``[B]`` index of the relevant candidate per query.
        k: Cutoff.

    Returns:
        Recall in ``[0, 1]``.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, scores.size(1))
    top = scores.topk(k, dim=-1).indices
    return (top == relevant.unsqueeze(-1)).any(dim=-1).float().mean().item()


def mean_reciprocal_rank(scores: torch.Tensor, relevant: torch.Tensor) -> float:
    """Mean of 1/rank of the relevant candidate.

    Args:
        scores: ``[B, N]``, higher is better.
        relevant: ``[B]`` index of the relevant candidate.

    Returns:
        MRR in ``(0, 1]``.
    """
    order = scores.argsort(dim=-1, descending=True)
    ranks = (order == relevant.unsqueeze(-1)).float().argmax(dim=-1) + 1
    return (1.0 / ranks.float()).mean().item()


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Ascending ranks, with tied values sharing their average rank.

    Ties are not an edge case for graded similarity data, they are most of it: STS-B
    validation carries 1500 pairs over 64 distinct scores, and its largest tie group is
    139 pairs all annotated 0.0. Ranking those by array position would invent an ordering
    the annotators never gave and move the correlation for a reason that has nothing to
    do with the model.

    Args:
        values: Numbers to rank.

    Returns:
        One rank per input, in input order, 1-based.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        shared = (start + stop) / 2.0 + 1.0
        for pos in range(start, stop + 1):
            ranks[order[pos]] = shared
        start = stop + 1
    return ranks


def spearman_correlation(predicted: Sequence[float], gold: Sequence[float]) -> float:
    """Spearman rank correlation: Pearson over average ranks.

    This is the held-out metric for any region judged on GRADED similarity rather than
    retrieval. Cosine similarity and a human 0-1 score do not share a scale and are not
    linearly related, so Pearson on the raw values would penalise a model that ranks every
    pair correctly but compresses its cosines into a narrow band -- which small encoders
    always do. Rank correlation asks only the question the region's job actually poses:
    do neighbours stay neighbours, in order.

    Implemented here rather than pulled from scipy: this is the whole of it, and scipy
    would be a new dependency on every host and CI job for one function.

    Args:
        predicted: Model scores, e.g. cosine similarities.
        gold: Human scores, aligned with ``predicted``.

    Returns:
        Correlation in ``[-1, 1]``. Returns 0.0 when either side is constant -- see below.

    Raises:
        ValueError: If the sequences differ in length or hold fewer than two points.
    """
    if len(predicted) != len(gold):
        raise ValueError(f"{len(predicted)} predictions vs {len(gold)} gold scores")
    if len(predicted) < 2:
        raise ValueError("rank correlation needs at least 2 points")

    rank_p = _average_ranks(predicted)
    rank_g = _average_ranks(gold)
    n = len(rank_p)
    mean_p = sum(rank_p) / n
    mean_g = sum(rank_g) / n
    cov = sum((a - mean_p) * (b - mean_g) for a, b in zip(rank_p, rank_g, strict=True))
    var_p = sum((a - mean_p) ** 2 for a in rank_p)
    var_g = sum((b - mean_g) ** 2 for b in rank_g)

    # A constant side has no ranking to correlate with, so the coefficient is undefined.
    # 0.0 is the honest report -- "no monotone relationship detectable" -- and it is what
    # a fully collapsed encoder deserves: every pair scored identically is not a perfect
    # correlation. Raising instead would abort a training run at exactly the moment the
    # collapse it is meant to detect had happened. Report emb_std alongside this to tell
    # "collapsed" apart from "uncorrelated".
    if var_p <= 0.0 or var_g <= 0.0:
        return 0.0
    return cov / math.sqrt(var_p * var_g)


def representation_std(embeddings: torch.Tensor) -> float:
    """Per-feature standard deviation across the batch, averaged.

    The collapse signal for any embedding model. Near zero means every input maps to the
    same vector, which produces excellent-looking losses on several objectives while the
    representation carries no information at all.

    Args:
        embeddings: ``[B, D]``.

    Returns:
        Mean per-feature std.
    """
    if embeddings.dim() != 2:
        raise ValueError(f"expected [B, D], got {tuple(embeddings.shape)}")
    if embeddings.size(0) < 2:
        raise ValueError("need at least 2 examples to measure variance across a batch")
    return embeddings.std(dim=0).mean().item()


class MetricIdentity(NamedTuple):
    """The identity keys `compare()` requires to match before treating two metric groups
    as the same measurement -- csd-metrics/v2's refuse-predicate (unification-rules memo
    §3.3, restating `docs/design/METRICS-METHODOLOGY.md` §10 as an enforced function
    rather than a checklist a reader has to remember).

    Field ORDER here is the CHECK order `compare()` uses, and matters: a metric group with
    several mismatching fields is refused on the FIRST one in this order, not an
    alphabetically- or dict-iteration-ordered one, so "the first mismatching key" a
    refusal names is reproducible rather than an artifact of dict internals.

    `k` is `None` for a metric with no `@k` (e.g. `mrr`) -- `None == None` is `True` in
    Python, so two groups that both lack a `k` (or share the same one) are never refused
    on this field alone.
    """

    metrics_schema: str
    corpus_fingerprint: str
    fingerprint_scheme: str
    battery_id: str
    k: int | None
    pooling: str
    checkpoint_sha256: str
    region: str
    git_sha: str
    seed: int


class MetricGroup(TypedDict):
    """One side of a `compare()` call: the identity it was measured under, plus its
    metric-name-to-value map. Build one from a receipt's own fields -- `identity` is not
    part of `metrics`, it is the provenance envelope around it (`metrics_schema` at the
    receipt top level; `corpus.fingerprint`/`.fingerprint_scheme`; `battery_id`; the `k`
    of whichever `@k` metric is being compared, or `None`; `pooling`;
    `artifacts.checkpoint_sha256`; `region`/`producer.component`; `code_revision.git_sha`;
    `seed`)."""

    identity: MetricIdentity
    values: dict[str, float]


class ComparisonRefusal(TypedDict):
    """What `compare()` returns instead of a comparison when the two `MetricGroup`s are
    not the same measurement. `refused` is always `True` on this branch -- present, rather
    than the caller inferring refusal from the absence of a `"metrics"` key, so a caller
    that only checks `"regressions" in result` cannot silently treat a refusal as "no
    regressions found"."""

    refused: bool
    mismatched_key: str
    baseline_value: object
    candidate_value: object
    reason: str


_IDENTITY_RECEIPT_NAMES: dict[str, str] = {
    # MetricIdentity field name -> the dotted receipt path a human would recognise it as,
    # in the SAME order compare() checks them. Used only to spell `mismatched_key` and the
    # refusal message; never consulted for the comparison logic itself.
    "metrics_schema": "metrics_schema",
    "corpus_fingerprint": "corpus.fingerprint",
    "fingerprint_scheme": "corpus.fingerprint_scheme",
    "battery_id": "battery_id",
    "k": "k",
    "pooling": "pooling",
    "checkpoint_sha256": "artifacts.checkpoint_sha256",
    "region": "region / producer.component",
    "git_sha": "code_revision.git_sha",
    "seed": "seed",
}


def compare(
    baseline: MetricGroup, candidate: MetricGroup, *, lower_is_better: set[str]
) -> dict[str, object]:
    """Compare a candidate against a baseline -- but REFUSE first, unless every identity
    key in `MetricIdentity` matches.

    THIS IS THE REFUSE-FUNCTION (csd-metrics/v2, unification-rules memo §3.3). The v1
    version of this function diffed whatever keys the two dicts happened to share and
    said nothing about whether the two dicts described the same measurement at all --
    exactly the ambiguity that let a `quantized_metric` (an in-memory plan's `recall@1`)
    get read next to an eval-quantized `rank.recall@1` (the packed-artifact's) as though
    they were interchangeable. `compare()` now refuses unless `metrics_schema`,
    `corpus.fingerprint` AND `.fingerprint_scheme`, `battery_id`, `k` (`None == None`),
    `pooling`, `checkpoint_sha256`, `region`/`producer.component`, `code_revision.git_sha`,
    and `seed` all match -- `docs/design/METRICS-METHODOLOGY.md` §10's seven-point
    checklist, enforced rather than left to a reader to remember, plus the two fields
    (`battery_id`, `pooling`) the v1 checklist did not yet name.

    Once every identity key matches, this proceeds exactly as v1 did: "net gain without
    quality loss" is evaluated per metric rather than by a blended score, because a change
    that improves throughput while raising perplexity is not a win and averaging the two
    would report it as one.

    NOT a general cross-battery comparator even when you WANT to compare two things this
    refuses -- MM §4 pre-registers a small set of pairs (e.g. `quant.plan_recall@1` vs
    `quant.artifact_recall@1` on the same sha) that are legitimately comparable despite
    crossing `battery_id`; those go through `assert_sameness()` instead, which this
    function must never subsume (a refuse-predicate that also special-cased those pairs
    would be re-implementing `assert_sameness()` inside `compare()` and the two could
    drift apart).

    Args:
        baseline: The reference measurement, with its identity.
        candidate: The measurement being evaluated, with its identity.
        lower_is_better: Metrics where a decrease is an improvement, e.g. perplexity.

    Returns:
        A `ComparisonRefusal` (`refused: True`, `mismatched_key` the first identity field
        in `MetricIdentity`'s field order that differed) if the two are not the same
        measurement. Otherwise per-metric deltas, a list of regressions, and an overall
        verdict, plus `refused: False`.
    """
    baseline_identity = baseline["identity"]
    candidate_identity = candidate["identity"]
    for field_name in MetricIdentity._fields:
        before_val = getattr(baseline_identity, field_name)
        after_val = getattr(candidate_identity, field_name)
        if before_val != after_val:
            receipt_name = _IDENTITY_RECEIPT_NAMES[field_name]
            refusal: ComparisonRefusal = {
                "refused": True,
                "mismatched_key": receipt_name,
                "baseline_value": before_val,
                "candidate_value": after_val,
                "reason": (
                    f"refusing to compare: {receipt_name!r} differs "
                    f"({before_val!r} vs {after_val!r}). These two metric groups are not "
                    "the same measurement (docs/design/METRICS-METHODOLOGY.md §10); a "
                    "delta between their values says nothing about the model."
                ),
            }
            return dict(refusal)

    baseline_values = baseline["values"]
    candidate_values = candidate["values"]
    shared = sorted(set(baseline_values) & set(candidate_values))
    deltas, regressions = {}, []
    for name in shared:
        before, after = baseline_values[name], candidate_values[name]
        improved = after < before if name in lower_is_better else after > before
        rel = ((after - before) / before) if before else float("nan")
        deltas[name] = {"before": before, "after": after, "rel_change": rel, "improved": improved}
        if not improved and before != after:
            regressions.append(name)
    return {
        "refused": False,
        "metrics": deltas,
        "regressions": regressions,
        "missing": sorted(set(baseline_values) ^ set(candidate_values)),
        "verdict": "no regression" if not regressions else f"regressed: {', '.join(regressions)}",
    }


def assert_sameness(
    label: str,
    baseline_value: float,
    candidate_value: float,
    *,
    baseline_checkpoint_sha256: str | None = None,
    candidate_checkpoint_sha256: str | None = None,
    tolerance: float = 1e-6,
) -> None:
    """Assert a pre-registered, deliberately cross-battery pair is numerically the same.

    NOT `compare()`. `compare()` REFUSES a cross-`battery_id` comparison on purpose (MM
    §10); this function exists for the small, pre-registered set of pairs MM §4/§3.2
    explicitly bless as legitimate despite crossing `battery_id` or metric name:
    `held_out.recall@1` vs `rank.recall@1` on the same checkpoint sha, `quant.plan_recall@1`
    vs `quant.artifact_recall@1` on the same sha/holdout, and the two closed-pool
    single-relevant-item identities `map == mrr` and `precision@10 == recall@10 / 10`
    (`average_precision()`/`precision_at_k()` in `cogsyndelta.eval.benchmark`). The
    refuse-predicate in `compare()` must NOT reject those pairs -- and does not, because
    this function is the one a caller uses for them, never `compare()`.

    This function carries NO general identity check (no schema, pooling, or battery_id
    comparison at all) -- it is asserting a mathematical identity between two named
    quantities that a caller has already decided are the same measurement by construction,
    not measuring a delta between two independent runs. The one identity check it DOES
    make, when both checkpoint shas are supplied, is that they match: two of MM §4's four
    pairs are pre-registered "on the same sha" and a sha mismatch there means the pairing
    itself is invalid, not merely that the numbers happened to differ.

    Args:
        label: Human-readable name for the pair, used only in the failure message.
        baseline_value: First value (e.g. `held_out.recall@1`).
        candidate_value: Second value (e.g. `rank.recall@1`).
        baseline_checkpoint_sha256: Checkpoint sha the first value was measured against,
            when the pairing is sha-scoped (omit for pairs like `map`/`mrr` that come from
            a single receipt and have no separate sha to compare).
        candidate_checkpoint_sha256: Checkpoint sha the second value was measured against.
        tolerance: Maximum allowed absolute difference.

    Raises:
        ValueError: If both shas are supplied and differ, or if the values differ by more
            than `tolerance`.
    """
    if (
        baseline_checkpoint_sha256 is not None
        and candidate_checkpoint_sha256 is not None
        and baseline_checkpoint_sha256 != candidate_checkpoint_sha256
    ):
        raise ValueError(
            f"sameness guard for {label!r} refuses: checkpoint sha differs "
            f"({baseline_checkpoint_sha256!r} vs {candidate_checkpoint_sha256!r}). MM §4's "
            "sameness pairs are legitimate only on the identical checkpoint; a sha "
            "mismatch here means these two numbers were never the same measurement to "
            "begin with, not that the sameness claim failed."
        )
    diff = abs(baseline_value - candidate_value)
    if diff > tolerance:
        raise ValueError(
            f"sameness guard failed for {label!r}: {baseline_value!r} != "
            f"{candidate_value!r} (diff {diff!r} > tolerance {tolerance!r}). These two are "
            "supposed to be the identical measurement under MM §3.2/§4 -- a difference "
            "here means the identity this pairing relies on no longer holds, and every "
            "receipt or card built on that assumption should be treated as suspect until "
            "this is understood."
        )


METRIC_ALIASES_V1: dict[str, str] = {
    # v1 (bare/unscoped) receipt field name -> v2 canonical dotted name
    # (csd-metrics/v2, docs/design/METRICS-METHODOLOGY.md + unification-rules memo §3.1).
    #
    # FOR READERS ONLY -- resolving a v1 name through this map to decide whether a GATE
    # passes recreates exactly the ambiguity v2 exists to close (a gate reads the v2 name
    # directly; this map exists so a script rendering an OLD receipt can still print a
    # readable v2-style label next to it). Never consulted by `compare()`'s refuse
    # predicate or by any `gates.*` computation.
    #
    # Not exhaustive across every receipt kind this project writes -- covers the names
    # this module and `cogsyndelta.eval.benchmark` produced under v1, plus the ones the
    # unification-rules memo names explicitly by name (§3.1/§3.2/§4). A lane that owns a
    # train/quant/region receipt's v1 field names extends this table for those rather than
    # duplicating it.
    "effective_rank": "repr.effective_rank_entropy",
    "emb_std": "repr.emb_std_anchor",
    "quantized_metric": "quant.plan_recall@1",
    "drop": "quant.drop_recall@1",
    "compression_ratio": "quant.compression_ratio",
}
