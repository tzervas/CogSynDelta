"""Offline BM25 hard-negative mining, its manifest, and the two guards that make it honest.

WHAT THIS IS FOR
`memory`'s InfoNCE sees the batch and nothing else, so on the 57,638-passage FiQA pool no
gradient ever separates a query's gold from the passages the batch never showed. The
pre-registered round PREREG-RETRIEVAL-NEGATIVES-2026-09-06 (rev 3) changes the negative
set and only the negative set; this module builds its T2 arm: `m = 8` BM25-mined
negatives per anchor, mined ONCE, offline, into a manifest that the training run then
verifies before it starts.

MINING PER PAIR, EXCLUDING ONLY THAT PAIR'S OWN POSITIVE (section 4)
Mining cannot be keyed on qrels: the training union is FiQA + AllNLI + NQ + GooAQ and
FiQA is 1.8% of it, so "exclude everything judged relevant" is undefined for the other
98.2%. The construction that IS defined for every pair is the standard one -- take BM25's
top-ranked passages over the pool the pair's positive came from, drop any identical to
that positive under the `pair_exact` normalisation, keep the top `m`.

THE POOL IS PER SOURCE, AND NEVER THE EVALUATION CORPUS
A pool is that source's distinct positives WITHIN THE PINNED TRAINING UNION. Mining FiQA
over the full 57,638-passage evaluation corpus would put dev-judged passages into
training, which is the contamination this project's split guard (G26) exists to prevent,
arriving through a different door.

THE GUARDS (section 4.2). Both fail closed, and `tests/test_guards_can_fail.py`
constructs the failing arm for each rather than reasoning about them:

  G38 self-positive disjointness -- no mined negative is identical to its own pair's
      positive under the `pair_exact` normalisation. Checkable for EVERY pair, not only
      the 1.8% with judgements, and it refuses the run rather than the batch.
  G39 mining provenance -- the manifest pins the training union's corpus fingerprint, the
      sha256 of each per-source pool rebuilt from that union, the sha256 of the qrels
      artefact the audit reads, the BM25 parameters and tokeniser, and (beyond the
      pre-registration's minimum, because a misalignment is silent and fatal) the sha256
      of the training pair sequence the negatives are indexed against. Any mismatch, any
      missing field, and any tampering with the recorded payload refuses the run.

THE FALSE NEGATIVES ARE MEASURED, NOT ASSUMED (section 4.1)
Excluding only the pair's own positive lets another genuine gold be mined as a negative.
On FiQA -- and only FiQA, whose training source IS a qrels pair split -- that rate is
exact and is audited here. Above the pre-registered 10% ceiling the branch is fixed in
advance: exclude every passage sharing the anchor's `query_id` from FiQA mining and
re-run the audit. `m` is NOT lowered (BM25 ranks the likeliest golds first, so a shorter
list would raise the rate) and the round is not abandoned.

GUARD NUMBERS: G38 and G39 were free at `12e2d1f` -- G26 is the text split guard
(`cogsyndelta.splits`), G27-G36 belong to INTERCONNECT-MODULE-SPEC.md Table 8, G37 is
`eval/geometry.py`'s geometry reference. The next new guard after this module is G40.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cogsyndelta.corpus import CORPUS_FINGERPRINT_SCHEME
from cogsyndelta.eval.beir_fiqa import _WORD, BM25

# The ONE spelling of the `pair_exact` normalisation (whitespace- and case-normalised),
# imported rather than re-typed: G38 claims to be the same key the contamination guard
# and `pretrain._pair_key` use, and two independently-written "normalise a text"
# implementations drifting apart is exactly how a guard ends up checking something else.
from cogsyndelta.eval.metrics import _normalise

MINING_MANIFEST_SCHEME = "csd-mined-negatives/v1"
"""Manifest schema tag. A reader that does not recognise it must refuse, not guess."""

NEGATIVES_PER_ANCHOR = 8
"""`m` in Table 2 of the pre-registration. Pinned: it may not be tuned after seeing a
result, and section 4.1 forbids lowering it in response to the audit."""

FALSE_NEGATIVE_CEILING = 0.10
"""Section 4.1's ceiling on the FiQA false-negative rate. Above it, the pre-registered
branch (query_id exclusion, re-audit) fires -- not a lower `m`, and not abandonment."""

BM25_K1 = 0.9
BM25_B = 0.4
"""BEIR's BM25 settings, matching `eval/beir_fiqa.BM25`'s own defaults, which is where
mining actually gets its ranking from. `tests/test_negatives.py` asserts these constants
equal that class's defaults, so the pinned parameters cannot drift from the code."""

BM25_TOKENIZER = _WORD.pattern
"""`[a-z0-9]+` over lowercased text -- `eval/beir_fiqa`'s own tokeniser, read from it."""


class MiningGuardError(Exception):
    """G38/G39 fail-closed: mined negatives are not what the manifest says they are."""


@dataclass(frozen=True)
class MiningPool:
    """One source's mining pool: its distinct positives inside the pinned training union.

    Attributes:
        source: The source's name, as `pretrain.load_sources` spells it.
        texts: Distinct positives in first-occurrence order -- deterministic, because the
            pool's sha256 is a guarded identity and an order-dependent hash over a
            set-like object would refuse a correct rebuild.
        sha256: :func:`pool_sha256` over ``texts``.
    """

    source: str
    texts: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class FalseNegativeAudit:
    """The measured FiQA false-negative rate and everything needed to read it.

    Attributes:
        mined_negatives: FiQA mined negatives audited.
        false_negatives: Of those, ones that are another gold of their own anchor's query.
        rate: `false_negatives / mined_negatives`, the number section 4.1 caps at 10%.
        anchors: FiQA anchors audited.
        contaminated_anchors: Anchors with at least one false negative among their `m`.
        contaminated_anchor_rate: Reported, no ceiling.
        golds_per_query: Mean judgements per `query_id` in the pinned split (2.570 on
            `fiqa-pairs/train.parquet`, the split mining actually runs on).
        ceiling: The pre-registered ceiling this rate is read against.
        breached: `rate > ceiling`. The branch, not a verdict on the round.
        query_id_exclusion: True when this audit was measured AFTER the breach branch
            excluded every passage sharing the anchor's `query_id`.
    """

    mined_negatives: int
    false_negatives: int
    rate: float
    anchors: int
    contaminated_anchors: int
    contaminated_anchor_rate: float
    golds_per_query: float
    ceiling: float = FALSE_NEGATIVE_CEILING
    breached: bool = False
    query_id_exclusion: bool = False

    def as_dict(self) -> dict[str, Any]:
        """The audit as receipt-shaped JSON.

        Returns:
            Every field, ready to embed in the mining manifest.
        """
        return {
            "mined_negatives": self.mined_negatives,
            "false_negatives": self.false_negatives,
            "rate": self.rate,
            "anchors": self.anchors,
            "contaminated_anchors": self.contaminated_anchors,
            "contaminated_anchor_rate": self.contaminated_anchor_rate,
            "golds_per_query": self.golds_per_query,
            "ceiling": self.ceiling,
            "breached": self.breached,
            "query_id_exclusion": self.query_id_exclusion,
        }


@dataclass
class MiningResult:
    """What one mining pass produced.

    Attributes:
        negatives: Per training pair, `m` pool indices into that pair's own source pool.
        source_of_pair: Per training pair, its source name.
        pools: The pools mined, by source name.
        audits: The FiQA audit(s), in the order they were measured -- two entries when
            the ceiling branch fired, so the manifest records the breach rather than only
            the repaired number.
    """

    negatives: list[list[int]]
    source_of_pair: list[str]
    pools: dict[str, MiningPool]
    audits: list[FalseNegativeAudit] = field(default_factory=list)


def pool_sha256(texts: Sequence[str]) -> str:
    """Hash a mining pool's exact contents, in order.

    Length-prefixed framing rather than a separator: a text containing the separator
    could otherwise make two different pools hash alike, and a pool identity that can
    collide is not an identity.

    Args:
        texts: The pool, in its stored order.

    Returns:
        Hex sha256 over the scheme tag and every text.
    """
    h = hashlib.sha256()
    h.update(MINING_MANIFEST_SCHEME.encode())
    for text in texts:
        payload = text.encode("utf-8", "replace")
        h.update(str(len(payload)).encode())
        h.update(b"\x00")
        h.update(payload)
    return h.hexdigest()


def train_pairs_sha256(train_pairs: Sequence[tuple[str, str]]) -> str:
    """Hash the training pair SEQUENCE the mined negatives are indexed against.

    Not in the pre-registration's G39 list, and required anyway: `negatives[i]` belongs to
    `train_pairs[i]`, so a different draw, a different `order_seed`, or a corpus that
    grew by one row silently pairs every anchor with someone else's hard negatives. That
    failure produces a plausible loss curve and a wrong experiment.

    Args:
        train_pairs: The pinned training union, in training order.

    Returns:
        Hex sha256 over the normalised pairs, in order.
    """
    h = hashlib.sha256()
    h.update(b"csd-train-pairs/v1")
    for anchor, positive in train_pairs:
        h.update(_normalise(anchor).encode("utf-8", "replace"))
        h.update(b"\x00")
        h.update(_normalise(positive).encode("utf-8", "replace"))
        h.update(b"\x01")
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file's bytes, streaming.

    Args:
        path: File to hash.

    Returns:
        Hex sha256.

    Raises:
        MiningGuardError: If the file is absent -- G39 cannot be evaluated against a
            missing artefact, and treating that as "nothing to check" is exactly the
            fail-open shape the guard exists to refuse.
    """
    p = Path(path)
    if not p.is_file():
        raise MiningGuardError(f"G39: artefact to hash is missing: {p}")
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_pools(sources: Sequence[tuple[str, Sequence[tuple[str, str]]]]) -> dict[str, MiningPool]:
    """Build one mining pool per source from the pinned training union.

    Args:
        sources: `(name, pairs)` in the order the region declares them, already
            restricted to pairs that survived into training.

    Returns:
        `{source name: MiningPool}`. Distinct positives, first-occurrence order, deduped
        under the `pair_exact` normalisation so a reformatted copy is not mined as a
        separate passage.
    """
    pools: dict[str, MiningPool] = {}
    for name, pairs in sources:
        seen: set[str] = set()
        texts: list[str] = []
        for _, positive in pairs:
            key = _normalise(positive)
            if key not in seen:
                seen.add(key)
                texts.append(positive)
        pools[name] = MiningPool(source=name, texts=tuple(texts), sha256=pool_sha256(texts))
    return pools


def restrict_to_union(
    sources: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    train_pairs: Sequence[tuple[str, str]],
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Drop from each source the pairs that did not survive into training.

    `build_splits` holds out, dedups and screens contamination before training sees a
    pair. A pool built from the raw source would therefore contain held-out and screened
    positives -- texts training never sees, mined as negatives, and in the held-out case
    a straight leak of the eval set into the objective.

    Args:
        sources: `(name, pairs)` as declared, before the split.
        train_pairs: The pinned training union, in training order.

    Returns:
        `(name, pairs)` in declaration order, each source's own order preserved so its
        pool hash is reproducible.
    """
    kept = {f"{_normalise(a)}\x00{_normalise(b)}" for a, b in train_pairs}
    return [
        (name, [(a, b) for a, b in pairs if f"{_normalise(a)}\x00{_normalise(b)}" in kept])
        for name, pairs in sources
    ]


def label_sources(
    train_pairs: Sequence[tuple[str, str]],
    sources: Sequence[tuple[str, Sequence[tuple[str, str]]]],
) -> list[str]:
    """Say which source each training pair came from.

    `build_splits` interleaves, shuffles and dedups the sources into one list and keeps
    only per-source COUNTS, so provenance has to be recovered by key. First declared
    source wins a pair that appears in two, which is arbitrary but deterministic -- and
    recorded in the manifest, so it is inspectable rather than assumed.

    Args:
        train_pairs: The pinned training union, in training order.
        sources: `(name, pairs)` as declared.

    Returns:
        A source name per training pair.

    Raises:
        MiningGuardError: If a training pair belongs to no declared source. Mining it
            against an arbitrary pool would be mining against the wrong corpus.
    """
    owner: dict[str, str] = {}
    for name, pairs in sources:
        for anchor, positive in pairs:
            owner.setdefault(f"{_normalise(anchor)}\x00{_normalise(positive)}", name)
    out: list[str] = []
    for index, (anchor, positive) in enumerate(train_pairs):
        name = owner.get(f"{_normalise(anchor)}\x00{_normalise(positive)}")
        if name is None:
            raise MiningGuardError(
                f"training pair {index} belongs to no declared source; its mining pool is undefined"
            )
        out.append(name)
    return out


def _ranked_candidates(scores: np.ndarray, take: int) -> np.ndarray:
    """Indices of the ``take`` best-scoring documents, ties broken by index.

    Args:
        scores: BM25 scores over one pool.
        take: How many to return.

    Returns:
        Indices, best first, deterministic under ties.
    """
    take = min(take, scores.shape[0])
    if take <= 0:
        return np.empty(0, dtype=np.int64)
    if take >= scores.shape[0]:
        candidates = np.arange(scores.shape[0], dtype=np.int64)
    else:
        candidates = np.argpartition(-scores, take - 1)[:take].astype(np.int64)
    # lexsort's LAST key is primary: sort by descending score, then by ascending index so
    # a tie resolves the same way on every machine and every rerun.
    order = np.lexsort((candidates, -scores[candidates]))
    return candidates[order]


def mine_negatives(
    train_pairs: Sequence[tuple[str, str]],
    source_of_pair: Sequence[str],
    pools: Mapping[str, MiningPool],
    *,
    m: int = NEGATIVES_PER_ANCHOR,
    exclusions: Mapping[int, set[int]] | None = None,
    only: Iterable[int] | None = None,
    previous: Sequence[list[int]] | None = None,
) -> list[list[int]]:
    """Mine `m` BM25 hard negatives per anchor over its own source's pool.

    Args:
        train_pairs: The pinned training union, in training order.
        source_of_pair: Per pair, its source name (:func:`label_sources`).
        pools: Per source, its mining pool.
        m: Negatives per anchor. Pinned at 8 by Table 2.
        exclusions: Optional extra pool indices to skip, per pair index -- how section
            4.1's ceiling branch (exclude every passage sharing the anchor's `query_id`)
            is applied without changing the construction for anyone else.
        only: Optional pair indices to mine; the rest are copied from ``previous``. The
            ceiling branch re-mines FiQA alone, and re-mining the other 98.2% would
            change negatives the audit never implicated.
        previous: Result to copy the un-mined rows from when ``only`` is given.

    Returns:
        Per training pair, up to `m` pool indices, best-ranked first.

    Raises:
        MiningGuardError: If a pair names a pool that was not built, or ``only`` is given
            without ``previous`` to copy the remaining rows from.
    """
    if only is not None and previous is None:
        raise MiningGuardError("mining a subset needs the previous result to copy from")
    indexes = {name: BM25(list(pool.texts), k1=BM25_K1, b=BM25_B) for name, pool in pools.items()}
    normalised = {name: [_normalise(t) for t in pool.texts] for name, pool in pools.items()}
    targets = set(range(len(train_pairs))) if only is None else set(only)
    out: list[list[int]] = [list(row) for row in previous] if previous is not None else []
    if not out:
        out = [[] for _ in train_pairs]

    for index, (anchor, positive) in enumerate(train_pairs):
        if index not in targets:
            continue
        name = source_of_pair[index]
        if name not in pools:
            raise MiningGuardError(f"training pair {index} names unknown mining pool {name!r}")
        banned = set(exclusions.get(index, ())) if exclusions else set()
        own = _normalise(positive)
        scores = indexes[name].scores(anchor)
        keep: list[int] = []
        # Ask for slack: every excluded index and every copy of the own positive can eat
        # a slot, so a bare top-`m` would silently return fewer than `m`.
        take = m + len(banned) + 8
        while True:
            candidates = _ranked_candidates(scores, take)
            keep = [
                int(j)
                for j in candidates
                if int(j) not in banned and normalised[name][int(j)] != own
            ][:m]
            if len(keep) >= m or take >= scores.shape[0]:
                break
            take = min(scores.shape[0], take * 4)
        out[index] = keep
    return out


def assert_self_positive_disjoint(
    train_pairs: Sequence[tuple[str, str]],
    negatives: Sequence[Sequence[int]],
    source_of_pair: Sequence[str],
    pools: Mapping[str, MiningPool],
) -> None:
    """G38: refuse if any mined negative is its own pair's positive.

    Under the `pair_exact` normalisation, so a whitespace- or case-reformatted copy of
    the positive is caught too -- an exact-string check would pass on precisely the copy
    a corpus is most likely to contain. Checked for every pair, not only the audited
    1.8%, and it refuses the RUN: a training step whose gold also appears as a negative
    is a corrupted objective, not a bad batch to skip.

    Args:
        train_pairs: The pinned training union, in training order.
        negatives: Per pair, mined pool indices.
        source_of_pair: Per pair, its source name.
        pools: Per source, its mining pool.

    Raises:
        MiningGuardError: On the first offending pair, naming it.
    """
    if len(negatives) != len(train_pairs):
        raise MiningGuardError(
            f"G38: {len(negatives)} mined rows against {len(train_pairs)} training pairs"
        )
    normalised = {name: [_normalise(t) for t in pool.texts] for name, pool in pools.items()}
    for index, (_, positive) in enumerate(train_pairs):
        name = source_of_pair[index]
        if name not in pools:
            raise MiningGuardError(f"G38: training pair {index} names unknown pool {name!r}")
        own = _normalise(positive)
        pool_texts = normalised[name]
        for j in negatives[index]:
            if not 0 <= int(j) < len(pool_texts):
                raise MiningGuardError(
                    f"G38: mined index {j} is outside pool {name!r} (size {len(pool_texts)})"
                )
            if pool_texts[int(j)] == own:
                raise MiningGuardError(
                    f"G38: mined negative {j} for training pair {index} IS that pair's own "
                    f"positive under the pair_exact normalisation"
                )


def audit_fiqa_false_negatives(
    *,
    train_pairs: Sequence[tuple[str, str]],
    source_of_pair: Sequence[str],
    negatives: Sequence[Sequence[int]],
    pools: Mapping[str, MiningPool],
    qrels: Mapping[str, Sequence[Any]],
    fiqa_source: str,
    ceiling: float = FALSE_NEGATIVE_CEILING,
    query_id_exclusion: bool = False,
) -> FalseNegativeAudit:
    """Measure how often a mined FiQA negative is another gold of its own query.

    Exact, and needs no external qrels file: FiQA's training source IS `beir_fiqa`'s
    `train` pair split, one row per judgement carrying `query_id`/`doc_id`. A mined
    negative is a false negative when its `doc_id` is among those sharing its anchor's
    `query_id` and is not its own pair's positive.

    Args:
        train_pairs: The pinned training union, in training order.
        source_of_pair: Per pair, its source name.
        negatives: Per pair, mined pool indices.
        pools: Per source, its mining pool.
        qrels: Columns of `fiqa-pairs/train.parquet` -- `query_id`, `doc_id`, `query`,
            `passage`.
        fiqa_source: The source name that holds the FiQA pairs.
        ceiling: Section 4.1's pre-registered ceiling.
        query_id_exclusion: Records whether this audit ran after the breach branch.

    Returns:
        The measured audit.

    Raises:
        MiningGuardError: If `qrels` is missing a column the audit reads.
    """
    for column in ("query_id", "doc_id", "query", "passage"):
        if column not in qrels:
            raise MiningGuardError(f"FiQA audit needs a {column!r} column in the qrels split")

    golds: dict[str, set[str]] = {}
    text_docs: dict[str, set[str]] = {}
    pair_rows: dict[str, tuple[str, str]] = {}
    for qid, doc_id, query, passage in zip(
        qrels["query_id"], qrels["doc_id"], qrels["query"], qrels["passage"], strict=True
    ):
        golds.setdefault(str(qid), set()).add(str(doc_id))
        text_docs.setdefault(_normalise(str(passage)), set()).add(str(doc_id))
        pair_rows.setdefault(
            f"{_normalise(str(query))}\x00{_normalise(str(passage))}", (str(qid), str(doc_id))
        )

    pool = pools[fiqa_source]
    normalised_pool = [_normalise(text) for text in pool.texts]
    mined = 0
    false_negatives = 0
    anchors = 0
    contaminated = 0
    for index, (anchor, positive) in enumerate(train_pairs):
        if source_of_pair[index] != fiqa_source:
            continue
        row = pair_rows.get(f"{_normalise(anchor)}\x00{_normalise(positive)}")
        if row is None:
            continue
        qid, own_doc = row
        anchors += 1
        hit = 0
        for j in negatives[index]:
            mined += 1
            docs = text_docs.get(normalised_pool[int(j)], set())
            if docs & (golds.get(qid, set()) - {own_doc}):
                hit += 1
        false_negatives += hit
        if hit:
            contaminated += 1

    rate = false_negatives / mined if mined else 0.0
    n_queries = len(golds)
    return FalseNegativeAudit(
        mined_negatives=mined,
        false_negatives=false_negatives,
        rate=rate,
        anchors=anchors,
        contaminated_anchors=contaminated,
        contaminated_anchor_rate=contaminated / anchors if anchors else 0.0,
        golds_per_query=len(qrels["doc_id"]) / n_queries if n_queries else 0.0,
        ceiling=ceiling,
        breached=rate > ceiling,
        query_id_exclusion=query_id_exclusion,
    )


def _query_id_exclusions(
    *,
    train_pairs: Sequence[tuple[str, str]],
    source_of_pair: Sequence[str],
    pools: Mapping[str, MiningPool],
    qrels: Mapping[str, Sequence[Any]],
    fiqa_source: str,
) -> dict[int, set[int]]:
    """Section 4.1's branch: every passage sharing the anchor's `query_id`, per FiQA pair.

    Args:
        train_pairs: The pinned training union, in training order.
        source_of_pair: Per pair, its source name.
        pools: Per source, its mining pool.
        qrels: Columns of the FiQA train pair split.
        fiqa_source: The source name holding FiQA.

    Returns:
        `{pair index: pool indices to exclude}`, FiQA pairs only. Exact and free: the
        exclusion is confined to the 1.8% of the union the audit covers.
    """
    golds: dict[str, set[str]] = {}
    doc_positions: dict[str, set[int]] = {}
    pair_rows: dict[str, tuple[str, str]] = {}
    pool = pools[fiqa_source]
    normalised_pool = [_normalise(text) for text in pool.texts]
    position_of_text: dict[str, list[int]] = {}
    for position, text in enumerate(normalised_pool):
        position_of_text.setdefault(text, []).append(position)

    for qid, doc_id, query, passage in zip(
        qrels["query_id"], qrels["doc_id"], qrels["query"], qrels["passage"], strict=True
    ):
        golds.setdefault(str(qid), set()).add(str(doc_id))
        for position in position_of_text.get(_normalise(str(passage)), []):
            doc_positions.setdefault(str(doc_id), set()).add(position)
        pair_rows.setdefault(
            f"{_normalise(str(query))}\x00{_normalise(str(passage))}", (str(qid), str(doc_id))
        )

    out: dict[int, set[int]] = {}
    for index, (anchor, positive) in enumerate(train_pairs):
        if source_of_pair[index] != fiqa_source:
            continue
        row = pair_rows.get(f"{_normalise(anchor)}\x00{_normalise(positive)}")
        if row is None:
            continue
        qid, _ = row
        excluded: set[int] = set()
        for doc_id in golds.get(qid, set()):
            excluded |= doc_positions.get(doc_id, set())
        if excluded:
            out[index] = excluded
    return out


def mine_and_audit(
    *,
    train_pairs: Sequence[tuple[str, str]],
    sources: Sequence[tuple[str, Sequence[tuple[str, str]]]],
    qrels: Mapping[str, Sequence[Any]],
    fiqa_source: str,
    m: int = NEGATIVES_PER_ANCHOR,
    ceiling: float = FALSE_NEGATIVE_CEILING,
) -> MiningResult:
    """Mine the whole union, audit FiQA, and apply section 4.1's branch if it breaches.

    Args:
        train_pairs: The pinned training union, in training order.
        sources: `(name, pairs)` as the region declares them.
        qrels: Columns of `fiqa-pairs/train.parquet`.
        fiqa_source: The source name holding FiQA.
        m: Negatives per anchor (pinned at 8).
        ceiling: The pre-registered false-negative ceiling.

    Returns:
        The mined negatives, their provenance, and every audit measured -- both of them
        when the branch fired, so the manifest records the breach and not only the repair.

    Raises:
        MiningGuardError: If G38 fails on the result, or provenance cannot be resolved.
    """
    pools = build_pools(sources)
    source_of_pair = label_sources(train_pairs, sources)
    negatives = mine_negatives(train_pairs, source_of_pair, pools, m=m)
    audits = [
        audit_fiqa_false_negatives(
            train_pairs=train_pairs,
            source_of_pair=source_of_pair,
            negatives=negatives,
            pools=pools,
            qrels=qrels,
            fiqa_source=fiqa_source,
            ceiling=ceiling,
        )
    ]
    if audits[0].breached:
        exclusions = _query_id_exclusions(
            train_pairs=train_pairs,
            source_of_pair=source_of_pair,
            pools=pools,
            qrels=qrels,
            fiqa_source=fiqa_source,
        )
        fiqa_rows = [i for i, name in enumerate(source_of_pair) if name == fiqa_source]
        negatives = mine_negatives(
            train_pairs,
            source_of_pair,
            pools,
            m=m,
            exclusions=exclusions,
            only=fiqa_rows,
            previous=negatives,
        )
        audits.append(
            audit_fiqa_false_negatives(
                train_pairs=train_pairs,
                source_of_pair=source_of_pair,
                negatives=negatives,
                pools=pools,
                qrels=qrels,
                fiqa_source=fiqa_source,
                ceiling=ceiling,
                query_id_exclusion=True,
            )
        )
    assert_self_positive_disjoint(train_pairs, negatives, source_of_pair, pools)
    return MiningResult(
        negatives=negatives, source_of_pair=source_of_pair, pools=pools, audits=audits
    )


def manifest_payload_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a manifest's payload, ignoring its own `sha256` field.

    Args:
        manifest: The manifest.

    Returns:
        Hex sha256 of the canonical JSON of every other field.
    """
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_manifest(
    *,
    region: str,
    corpus_fingerprint: str,
    result: MiningResult,
    train_pairs: Sequence[tuple[str, str]],
    qrels_path: str | Path,
    m: int = NEGATIVES_PER_ANCHOR,
) -> dict[str, Any]:
    """Assemble the mining manifest G39 checks a run against.

    Negatives are stored as POOL INDICES, not texts: a run rebuilds the pools from the
    same pinned union anyway (that rebuild is what makes the per-pool sha256 a check
    rather than a copy), and storing several million passages twice would make the
    manifest unreviewable.

    Args:
        region: The region mined for.
        corpus_fingerprint: The TRAINING UNION's fingerprint (`csd-corpus-fp/v2`).
        result: What :func:`mine_and_audit` produced.
        train_pairs: The pinned training union, in training order.
        qrels_path: `fiqa-pairs/train.parquet`, the artefact section 4.1 reads.
        m: Negatives per anchor.

    Returns:
        The manifest, with its own payload sha256 stamped in.
    """
    names = sorted(result.pools)
    index_of = {name: position for position, name in enumerate(names)}
    manifest: dict[str, Any] = {
        "schema": MINING_MANIFEST_SCHEME,
        "region": region,
        "negatives_per_anchor": m,
        "corpus": {"scheme": CORPUS_FINGERPRINT_SCHEME, "fingerprint": corpus_fingerprint},
        "bm25": {
            "k1": BM25_K1,
            "b": BM25_B,
            "tokenizer": BM25_TOKENIZER,
            "lowercase": True,
        },
        "pools": {
            name: {"sha256": result.pools[name].sha256, "size": len(result.pools[name].texts)}
            for name in names
        },
        "qrels": {"path": str(qrels_path), "sha256": file_sha256(qrels_path)},
        "train_pairs": {
            "count": len(train_pairs),
            "sha256": train_pairs_sha256(train_pairs),
        },
        "source_names": names,
        "source_index": [index_of[name] for name in result.source_of_pair],
        "negatives": [list(row) for row in result.negatives],
        "audits": [audit.as_dict() for audit in result.audits],
    }
    manifest["sha256"] = manifest_payload_sha256(manifest)
    return manifest


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write a manifest as JSON.

    Args:
        path: Destination.
        manifest: What :func:`build_manifest` returned.

    Returns:
        The path written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read a mining manifest.

    Args:
        path: Manifest JSON.

    Returns:
        The manifest.

    Raises:
        MiningGuardError: If the file is missing or is not a JSON object -- a run that
            cannot read its provenance must refuse, not proceed unverified.
    """
    p = Path(path)
    if not p.is_file():
        raise MiningGuardError(f"G39: mining manifest missing: {p}")
    try:
        loaded = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise MiningGuardError(f"G39: mining manifest is not valid JSON: {p} ({exc})") from exc
    if not isinstance(loaded, dict):
        raise MiningGuardError(f"G39: mining manifest is not a JSON object: {p}")
    return loaded


def _require(manifest: Mapping[str, Any], key: str) -> Any:
    """Read a manifest field, refusing when it is absent.

    Args:
        manifest: The manifest.
        key: Field name.

    Returns:
        The value.

    Raises:
        MiningGuardError: If the field is missing. Absence is a refusal, not a default:
            a manifest that simply omits its corpus fingerprint would otherwise pass a
            provenance check by having no provenance.
    """
    if key not in manifest:
        raise MiningGuardError(f"G39: mining manifest has no {key!r}")
    return manifest[key]


def verify_manifest(
    manifest: Mapping[str, Any],
    *,
    corpus_fingerprint: str,
    pools: Mapping[str, MiningPool],
    qrels_sha256: str,
    train_pairs: Sequence[tuple[str, str]],
    m: int = NEGATIVES_PER_ANCHOR,
) -> None:
    """G39: refuse unless the manifest matches the corpus, pools and artefacts at hand.

    Everything the pre-registration lists is checked against a value computed HERE, from
    the run's own corpus, rather than against another field of the same manifest -- a
    self-consistent manifest for a different corpus is exactly the failure this refuses.

    Args:
        manifest: The loaded manifest.
        corpus_fingerprint: The training union's fingerprint, computed for this run.
        pools: Pools rebuilt from this run's own pinned union.
        qrels_sha256: sha256 of `fiqa-pairs/train.parquet` as it is now.
        train_pairs: This run's pinned training union, in training order.
        m: Negatives per anchor this run expects.

    Raises:
        MiningGuardError: On the first mismatch, naming the field.
    """
    if _require(manifest, "schema") != MINING_MANIFEST_SCHEME:
        raise MiningGuardError(
            f"G39: manifest schema {manifest['schema']!r} != {MINING_MANIFEST_SCHEME!r}"
        )
    recorded = manifest_payload_sha256(manifest)
    if _require(manifest, "sha256") != recorded:
        raise MiningGuardError("G39: manifest payload sha256 does not match its contents")

    corpus = _require(manifest, "corpus")
    if corpus.get("scheme") != CORPUS_FINGERPRINT_SCHEME:
        raise MiningGuardError(
            f"G39: corpus fingerprint scheme {corpus.get('scheme')!r} != "
            f"{CORPUS_FINGERPRINT_SCHEME!r}"
        )
    if corpus.get("fingerprint") != corpus_fingerprint:
        raise MiningGuardError(
            f"G39: mined against corpus {corpus.get('fingerprint')}, run is {corpus_fingerprint}"
        )

    bm25 = _require(manifest, "bm25")
    expected_bm25 = {"k1": BM25_K1, "b": BM25_B, "tokenizer": BM25_TOKENIZER, "lowercase": True}
    if bm25 != expected_bm25:
        raise MiningGuardError(f"G39: BM25 parameters {bm25} != {expected_bm25}")

    if _require(manifest, "negatives_per_anchor") != m:
        raise MiningGuardError(
            f"G39: manifest mined {manifest['negatives_per_anchor']} negatives per anchor, "
            f"run asks for {m}"
        )

    recorded_pools = _require(manifest, "pools")
    if set(recorded_pools) != set(pools):
        raise MiningGuardError(
            f"G39: manifest pools {sorted(recorded_pools)} != rebuilt {sorted(pools)}"
        )
    for name, pool in pools.items():
        if recorded_pools[name].get("sha256") != pool.sha256:
            raise MiningGuardError(
                f"G39: pool {name!r} rebuilt from this run's union hashes {pool.sha256}, "
                f"manifest records {recorded_pools[name].get('sha256')}"
            )

    if _require(manifest, "qrels").get("sha256") != qrels_sha256:
        raise MiningGuardError("G39: qrels artefact sha256 differs from the mined one")

    pairs = _require(manifest, "train_pairs")
    if pairs.get("count") != len(train_pairs):
        raise MiningGuardError(
            f"G39: manifest mined {pairs.get('count')} training pairs, run has {len(train_pairs)}"
        )
    if pairs.get("sha256") != train_pairs_sha256(train_pairs):
        raise MiningGuardError(
            "G39: the training pair sequence differs from the one mined against; every "
            "anchor would be paired with another anchor's negatives"
        )
    if len(_require(manifest, "negatives")) != len(train_pairs):
        raise MiningGuardError("G39: mined rows do not cover the training pairs one-for-one")


def negative_texts(
    manifest: Mapping[str, Any],
    pools: Mapping[str, MiningPool],
    *,
    m: int = NEGATIVES_PER_ANCHOR,
) -> list[str]:
    """Resolve the manifest's pool indices to texts, flattened in training-pair order.

    Flattened `m` per pair (pair `i` owns rows `[i*m, (i+1)*m)`) so the training loop can
    take a CONTIGUOUS slice for its batch: the mined negatives then ride the same
    pre-tokenised corpus cache as the anchors and positives, and no tokenizer goes back
    into the step.

    Args:
        manifest: A verified manifest.
        pools: Pools rebuilt from the run's own union.
        m: Negatives per anchor.

    Returns:
        `len(negatives) * m` texts.

    Raises:
        MiningGuardError: If a row is short, or an index falls outside its pool -- both
            would misalign the flattened block.
    """
    names = _require(manifest, "source_names")
    source_index = _require(manifest, "source_index")
    out: list[str] = []
    for row, (indices, source) in enumerate(
        zip(_require(manifest, "negatives"), source_index, strict=True)
    ):
        if len(indices) != m:
            raise MiningGuardError(
                f"G39: training pair {row} has {len(indices)} negatives, not {m}"
            )
        texts = pools[names[source]].texts
        for j in indices:
            if not 0 <= int(j) < len(texts):
                raise MiningGuardError(
                    f"G39: mined index {j} for training pair {row} is outside pool "
                    f"{names[source]!r}"
                )
            out.append(texts[int(j)])
    return out
