"""The BEIR-style FiQA retrieval eval: full-pool ranking, a BM25 reference, and the five
pre-registered W4 gates (DEC-09, row W4, docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md).

WHY THIS IS NOT `pretrain_region`'s OWN `evaluate()`
`pretrain_region` scores a region on the DIAGONAL of a held-out batch: each anchor against
the positives of the other held-out pairs. For FiQA-shaped retrieval that inflates the
number two ways -- **the pool is wrong** (ranking 500 questions against 500 candidate
answers is a much easier task than ranking them against the full 57,638-passage FiQA
corpus, which is what a retriever actually faces -- recall@10 out of 512 and recall@10 out
of 57,638 are different measurements that happen to share a name), and **relevance is not
the diagonal** (FiQA judges ~2.6 passages relevant per question; a model that ranks a
DIFFERENT correct answer first is scored wrong on the diagonal). DEC-09 retires the
FiQA-only training regime this eval used to carry (`regions/retrieve.py`, now merged into
`regions/memory.py` per DEC-02) and keeps only this half.

AND A LEXICAL REFERENCE, BECAUSE "BEATS RANDOM INIT" IS NOT THE BAR
BM25 is a genuinely competitive FiQA retriever. Reporting "beats random init" while losing
to word counting would be reporting a win for a loss -- exactly what `rank_metrics`/`BM25`
below were written to make impossible to do by accident: BM25 is computed over the SAME
pool, the SAME qrels and the SAME `rank_metrics` code path as the trained model, not a
number quoted from elsewhere.

LAYERING: NO DEPENDENCY ON `cogsyndelta.regions`
`cogsyndelta.regions.pretrain` imports from `cogsyndelta.eval` (for the contamination
guards and `recall_at_k`/`mean_reciprocal_rank`); this module living under `eval/` too
means it must not import anything from `regions/` at module scope, or the two packages
would import each other. `encode_texts`/`encoder_rank_metrics` below therefore take a
plain callable `model` (anything shaped like `model(ids, mask) -> [N, D]`, with `.eval()`/
`.train()` -- `TextEncoder` already has this shape and needs no adapter) and a `tokenize`
callable rather than importing `TextEncoder`/the tokenizer helpers directly. The concrete
wiring -- an actual `TextEncoder` and `Tokenizer` -- happens one layer up, in
`regions/memory.py`, exactly where `regions/retrieve.py` used to do the identical wiring
for itself.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from cogsyndelta.eval.metrics import mean_reciprocal_rank, recall_at_k

DEFAULT_FIQA_ROOT = Path("/mnt/fleet-datasets/csd/region/retrieve")
"""Read-only NFS view of homelab's `/data/datasets/csd/region/retrieve` -- the same mount
`regions/memory.py`'s `RETRIEVAL_FIQA_SHARD` and `scripts/csd-train-all.py`'s
`REGIONS["memory"]` resolve their training pairs against; this is FiQA's judged corpus
and qrels, built once by `scripts/csd-fetch-fiqa-qrels.py`."""

SPLITS = ("train", "dev", "test")

_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class FiqaPaths:
    """Resolved locations for the FiQA pair splits and the passage corpus."""

    pairs_dir: Path
    corpus_dir: Path

    def split(self, name: str) -> Path:
        """Path to one built pair split.

        Raises:
            KeyError: If ``name`` is not a BeIR split.
            FileNotFoundError: If the split was never built.
        """
        if name not in SPLITS:
            raise KeyError(f"unknown split {name!r}; expected one of {SPLITS}")
        path = self.pairs_dir / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is not present. Build it on homelab with "
                "scripts/csd-fetch-fiqa-qrels.py -- BeIR/fiqa ships no qrels, so the "
                "corpus alone yields zero positive pairs."
            )
        return path


def resolve_paths(root: Path | None = None) -> FiqaPaths:
    """Locate the FiQA pair splits and corpus under ``root``.

    Args:
        root: Directory holding ``fiqa/`` and ``fiqa-pairs/``. Defaults to the fleet mount.

    Returns:
        Resolved paths.

    Raises:
        FileNotFoundError: If the mount or the built pairs are absent. Globbing a missing
            directory returns an empty list, so without this an unmounted export presents
            as "FiQA has no pairs", which sends you to the data instead of to `findmnt`.
    """
    base = Path(root) if root is not None else DEFAULT_FIQA_ROOT
    pairs_dir = base / "fiqa-pairs"
    corpus_dir = base / "fiqa" / "corpus"
    if not pairs_dir.is_dir():
        raise FileNotFoundError(
            f"{pairs_dir} is not present. This eval's pair splits are built from "
            "BeIR/fiqa plus BeIR/fiqa-qrels by scripts/csd-fetch-fiqa-qrels.py, which "
            "runs on the corpus host."
        )
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"{corpus_dir} is not present (BeIR/fiqa corpus shard)")
    return FiqaPaths(pairs_dir=pairs_dir, corpus_dir=corpus_dir)


def load_split(split: str, root: Path | None = None) -> dict[str, list]:
    """Read one built pair split as columns.

    Returns:
        ``{"query_id", "doc_id", "score", "query", "passage"}``, one entry per judgement,
        so a question with three relevant passages appears three times.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(resolve_paths(root).split(split))
    return {name: table.column(name).to_pylist() for name in table.column_names}


def load_corpus(root: Path | None = None) -> tuple[list[str], list[str]]:
    """Read the full FiQA passage corpus as ``(doc_ids, texts)``.

    Title and text are concatenated when a title exists, matching how the pair splits were
    built. FiQA answers are forum posts and almost all have an empty title; the join is
    kept anyway so the same loader works for a titled BeIR set.
    """
    import pyarrow.parquet as pq

    shards = sorted(resolve_paths(root).corpus_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError("no parquet shards in the FiQA corpus directory")

    ids: list[str] = []
    texts: list[str] = []
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=4096, columns=["_id", "title", "text"]):
            rows = batch.to_pydict()
            for doc_id, title, text in zip(rows["_id"], rows["title"], rows["text"]):
                title = (title or "").strip()
                text = (text or "").strip()
                ids.append(doc_id)
                texts.append(f"{title}\n{text}".strip() if title else text)
    return ids, texts


@dataclass
class RankingTask:
    """A held-out split reshaped into the ranking problem it actually is.

    Attributes:
        queries: One entry per unique query, deduplicated across judgements.
        pool_ids: Candidate document ids, in scoring order.
        pool_texts: Candidate document texts, aligned with ``pool_ids``.
        gold: Per query, the indices into the pool that the qrels judge relevant.
    """

    queries: list[str]
    query_ids: list[str]
    pool_ids: list[str]
    pool_texts: list[str]
    gold: list[list[int]]

    def summary(self) -> dict[str, float]:
        """Shape of the task, so a metric can never be read without its pool size."""
        judged = sum(len(g) for g in self.gold)
        return {
            "queries": float(len(self.queries)),
            "pool_size": float(len(self.pool_ids)),
            "relevant_per_query": judged / len(self.queries) if self.queries else 0.0,
        }


def build_ranking_task(split: str, pool: str = "corpus", root: Path | None = None) -> RankingTask:
    """Reshape a pair split into queries, a candidate pool, and per-query relevance.

    Args:
        split: ``dev`` or ``test``. ``train`` is accepted but ranking a split you trained
            on measures memorisation.
        pool: ``corpus`` ranks against all 57,638 FiQA passages, which is the real task
            and the number the W4 gates read. ``split`` ranks only against the passages
            judged in this split -- a far easier pool, useful for comparability with an
            in-batch number, never as the headline.
        root: Dataset root override.

    Returns:
        The task.

    Raises:
        ValueError: If ``pool`` is not one of the two modes, or a judged document is
            missing from the pool -- which would silently make those queries unanswerable
            and quietly depress every metric.
    """
    if pool not in ("corpus", "split"):
        raise ValueError(f"pool must be 'corpus' or 'split', got {pool!r}")

    columns = load_split(split, root)
    if pool == "corpus":
        pool_ids, pool_texts = load_corpus(root)
    else:
        seen: dict[str, str] = {}
        for doc_id, passage in zip(columns["doc_id"], columns["passage"]):
            seen.setdefault(doc_id, passage)
        pool_ids = list(seen)
        pool_texts = [seen[d] for d in pool_ids]

    index = {doc_id: i for i, doc_id in enumerate(pool_ids)}
    order: list[str] = []
    query_text: dict[str, str] = {}
    gold: dict[str, list[int]] = {}
    for qid, query, doc_id in zip(columns["query_id"], columns["query"], columns["doc_id"]):
        if qid not in query_text:
            query_text[qid] = query
            gold[qid] = []
            order.append(qid)
        position = index.get(doc_id)
        if position is None:
            raise ValueError(f"judged document {doc_id} for query {qid} is absent from the pool")
        gold[qid].append(position)

    return RankingTask(
        queries=[query_text[q] for q in order],
        query_ids=order,
        pool_ids=pool_ids,
        pool_texts=pool_texts,
        gold=[gold[q] for q in order],
    )


def rank_metrics(scores: torch.Tensor, gold: list[list[int]]) -> dict[str, float]:
    """Recall@k and MRR over a multi-relevant ranking, per query.

    ``recall_at_k``/``mean_reciprocal_rank`` take one relevant index per row. The standard
    IR definitions are "at least one relevant in the top k" and "1/rank of the FIRST
    relevant", so this picks each query's best-scoring gold and suppresses its other golds
    to ``-inf``.

    That is exact, not an approximation: no other gold can outrank the best gold, so
    removing them cannot change its rank, and if any gold reaches the top k then the best
    one does too. It also keeps a second correct answer from being counted as a wrong
    document that pushed the right one down, which is the diagonal evaluation's core error
    this whole module exists to avoid.

    Args:
        scores: ``[Q, N]``, higher is better.
        gold: Per query, pool indices judged relevant. Must be non-empty.

    Returns:
        recall@1, recall@10, recall@100 (capped at the pool size) and MRR.
    """
    if scores.size(0) != len(gold):
        raise ValueError(f"{scores.size(0)} score rows vs {len(gold)} gold lists")
    if any(not g for g in gold):
        raise ValueError("every query needs at least one relevant document")

    masked = scores.clone().float()
    best = torch.empty(scores.size(0), dtype=torch.long, device=scores.device)
    for row, positives in enumerate(gold):
        idx = torch.tensor(positives, device=scores.device)
        pick = idx[masked[row, idx].argmax()]
        masked[row, idx] = float("-inf")
        masked[row, pick] = scores[row, pick].float()
        best[row] = pick

    out = {"mrr": mean_reciprocal_rank(masked, best)}
    for k in (1, 10, 100):
        if k <= scores.size(1):
            out[f"recall@{k}"] = recall_at_k(masked, best, k)
    return out


class BM25:
    """Okapi BM25 over a fixed document pool -- the lexical reference point.

    Kept deliberately simple and dependency-free: an inverted index of numpy postings.
    Scoring one query touches only the postings of its own terms, so 500 queries against
    57,638 documents is a second of CPU, not a reason to skip the baseline.
    """

    def __init__(self, docs: list[str], k1: float = 0.9, b: float = 0.4) -> None:
        """Index ``docs``. Defaults are BEIR's BM25 settings, not Anserini's k1=1.2/b=0.75."""
        self.k1, self.b = k1, b
        self.n_docs = len(docs)
        lengths = np.zeros(self.n_docs, dtype=np.float32)
        postings: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(docs):
            terms = _WORD.findall(doc.lower())
            lengths[i] = len(terms)
            counts: dict[str, int] = {}
            for term in terms:
                counts[term] = counts.get(term, 0) + 1
            for term, tf in counts.items():
                postings.setdefault(term, []).append((i, tf))
        self.avg_len = float(lengths.mean()) if self.n_docs else 0.0
        denom_len = self.k1 * (1.0 - self.b + self.b * lengths / max(self.avg_len, 1e-9))

        self.index: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, entries in postings.items():
            docs_idx = np.fromiter((d for d, _ in entries), dtype=np.int64, count=len(entries))
            tfs = np.fromiter((t for _, t in entries), dtype=np.float32, count=len(entries))
            df = len(entries)
            idf = float(np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)))
            weight = idf * tfs * (self.k1 + 1.0) / (tfs + denom_len[docs_idx])
            self.index[term] = (docs_idx, weight.astype(np.float32))

    def scores(self, query: str) -> np.ndarray:
        """Score every document for one query. Unknown terms contribute nothing."""
        out = np.zeros(self.n_docs, dtype=np.float32)
        for term in _WORD.findall(query.lower()):
            hit = self.index.get(term)
            if hit is not None:
                np.add.at(out, hit[0], hit[1])
        return out

    def score_matrix(self, queries: list[str]) -> torch.Tensor:
        """Score every query against every document as ``[Q, N]``."""
        return torch.from_numpy(np.stack([self.scores(q) for q in queries]))


class _EncoderLike(Protocol):
    """Structural type for `encode_texts`/`encoder_rank_metrics`'s `model` -- anything
    shaped like `TextEncoder` (a callable `(ids, mask) -> [N, D]` with train/eval mode),
    without this module importing `TextEncoder` itself (see the module docstring)."""

    training: bool

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor: ...
    def eval(self) -> Any: ...
    def train(self, mode: bool = True) -> Any: ...


@torch.no_grad()
def encode_texts(
    model: _EncoderLike,
    tokenize: Any,
    texts: list[str],
    batch: int = 256,
) -> torch.Tensor:
    """Encode texts to L2-normalised ``[N, D]``.

    Args:
        model: Anything shaped like `TextEncoder` -- see `_EncoderLike`.
        tokenize: `(texts: list[str]) -> (ids, mask)`, already bound to a tokenizer,
            `max_len` and device by the caller (`regions/memory.py`).
        texts: Strings to encode.
        batch: Chunk size.
    """
    was_training = model.training
    model.eval()
    chunks = []
    for i in range(0, len(texts), batch):
        ids, mask = tokenize(texts[i : i + batch])
        chunks.append(torch.nn.functional.normalize(model(ids, mask), dim=-1))
    if was_training:
        model.train()
    if not chunks:
        return torch.zeros(0, 0)
    return torch.cat(chunks)


@torch.no_grad()
def encoder_rank_metrics(
    model: _EncoderLike,
    tokenize: Any,
    task: RankingTask,
    batch: int = 256,
) -> dict[str, float]:
    """Rank ``task``'s queries against its pool with a dense encoder."""
    pool = encode_texts(model, tokenize, task.pool_texts, batch)
    queries = encode_texts(model, tokenize, task.queries, batch)
    return rank_metrics(queries @ pool.T, task.gold)


def bm25_metrics(task: RankingTask) -> dict[str, float]:
    """BM25 over `task`'s own pool -- the lexical reference, from the SAME code path
    (`rank_metrics`) the dense encoder is scored through, per this module's own rule
    against "reporting a win for a loss"."""
    t0 = time.time()
    metrics = rank_metrics(BM25(task.pool_texts).score_matrix(task.queries), task.gold)
    metrics["index_s"] = round(time.time() - t0, 1)
    return metrics


# ---------------------------------------------------------------------------------------
# The five pre-registered W4 gates (docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md, row
# W4). Pure functions over already-measured numbers -- no I/O, no model, no corpus -- so
# every one of them is directly constructible-to-fail in a unit test.
# ---------------------------------------------------------------------------------------

# VERIFIED against the receipts named below (read on 2026-09-03; sha256 recorded in each
# receipt's own `checkpoint_sha256`, which is not the receipt's own hash but pins the
# checkpoint the number was measured from):
#   compress: /akula-data/csd/receipts/compress-20260903T120818Z.json
#             held_out.recall@1 = 0.775390625, graded_held_out.spearman = 0.7588195158826483
#   retrieve: /akula-data/csd/receipts/retrieve-20260903T121603Z.json
#             held_out.recall@1 = 0.755859375
COMPRESS_PARENT_RECALL_AT_1 = 0.775390625
COMPRESS_PARENT_GRADED_SPEARMAN = 0.7588195158826483
RETRIEVE_PARENT_RECALL_AT_1 = 0.755859375

_RETRAIN_GATE_REGRESSION_MARGIN = 0.01
"""§4.0's W4/W7 gate, clause 2: "the region's own receipt metric does not regress by
more than 1 point" -- 1 point in a [0, 1] recall/spearman figure is 0.01."""


def gate_a_beats_both_parents(
    memory_held_out_recall_at_1: float,
    memory_graded_spearman: float,
    *,
    compress_recall_at_1: float = COMPRESS_PARENT_RECALL_AT_1,
    compress_graded_spearman: float = COMPRESS_PARENT_GRADED_SPEARMAN,
    retrieve_recall_at_1: float = RETRIEVE_PARENT_RECALL_AT_1,
) -> dict[str, Any]:
    """W4 gate (1): matches or beats BOTH parents on BOTH parents' own gates.

    `memory_held_out_recall_at_1` is read against BOTH parents' recall@1 (their
    thresholds differ; the harder one binds) and `memory_graded_spearman` against
    `compress`'s STS-B gate -- `retrieve` declares no graded gate, so it contributes only
    a recall@1 floor. Compared with `>=`: "matches or beats", not strictly beats.
    """
    threshold_recall = max(compress_recall_at_1, retrieve_recall_at_1)
    passed_recall = memory_held_out_recall_at_1 >= threshold_recall
    passed_graded = memory_graded_spearman >= compress_graded_spearman
    return {
        "gate": "a_beats_both_parents",
        "memory_recall@1": memory_held_out_recall_at_1,
        "compress_recall@1": compress_recall_at_1,
        "retrieve_recall@1": retrieve_recall_at_1,
        "recall_threshold": threshold_recall,
        "memory_graded_spearman": memory_graded_spearman,
        "compress_graded_spearman": compress_graded_spearman,
        "passed_recall": passed_recall,
        "passed_graded": passed_graded,
        "passed": bool(passed_recall and passed_graded),
    }


def gate_b_full_pool_thresholds(
    full_pool_trained: dict[str, float],
    *,
    recall_at_10_floor: float = 0.20,
    mrr_floor: float = 0.10,
) -> dict[str, Any]:
    """W4 gate (2): `recall@10 > 0.20` and `MRR > 0.10` on the full 57,638-passage pool."""
    recall = full_pool_trained.get("recall@10", 0.0)
    mrr = full_pool_trained.get("mrr", 0.0)
    passed = recall > recall_at_10_floor and mrr > mrr_floor
    return {
        "gate": "b_full_pool_thresholds",
        "recall@10": recall,
        "recall@10_floor": recall_at_10_floor,
        "mrr": mrr,
        "mrr_floor": mrr_floor,
        "passed": bool(passed),
    }


def gate_c_beats_bm25(
    full_pool_trained: dict[str, float], full_pool_bm25: dict[str, float]
) -> dict[str, Any]:
    """W4 gate (3): `memory` > BM25 on the SAME pool/qrels/code path -- beaten, not
    merely reported alongside ("reporting a win for a loss" is exactly what this file was
    written to prevent)."""
    passed = full_pool_trained.get("recall@10", 0.0) > full_pool_bm25.get("recall@10", 0.0)
    return {
        "gate": "c_beats_bm25",
        "trained_recall@10": full_pool_trained.get("recall@10", 0.0),
        "bm25_recall@10": full_pool_bm25.get("recall@10", 0.0),
        "passed": bool(passed),
    }


def gate_d_beats_random_init(
    full_pool_trained: dict[str, float], full_pool_untrained: dict[str, float]
) -> dict[str, Any]:
    """W4 gate (4): `memory` > its own random-init baseline on the full pool."""
    passed = full_pool_trained.get("recall@10", 0.0) > full_pool_untrained.get("recall@10", 0.0)
    return {
        "gate": "d_beats_random_init",
        "trained_recall@10": full_pool_trained.get("recall@10", 0.0),
        "untrained_recall@10": full_pool_untrained.get("recall@10", 0.0),
        "passed": bool(passed),
    }


def gate_e_retrain_gate(
    token_global_pr_rank: float,
    pooled_pr_rank: float,
    memory_held_out_recall_at_1: float,
    memory_graded_spearman: float,
    *,
    compress_recall_at_1: float = COMPRESS_PARENT_RECALL_AT_1,
    compress_graded_spearman: float = COMPRESS_PARENT_GRADED_SPEARMAN,
    retrieve_recall_at_1: float = RETRIEVE_PARENT_RECALL_AT_1,
    margin: float = _RETRAIN_GATE_REGRESSION_MARGIN,
) -> dict[str, Any]:
    """W4 gate (5): §4.0's retrain gate, BOTH clauses required.

    (1) `token_global_pr_rank >= 2.0 * pooled_pr_rank` at the final block -- read
    straight from `pretrain_region`'s own `token_aware.final_block_rank` (commit 1),
    measured the same way (participation ratio) W1 pre-committed as "no retrain needed".
    (2) The region's own receipt metric does not regress by more than 1 point against
    either parent's -- `memory` has no PRE-token-aware baseline of its own (it is a new
    merged region, not a retrain of an existing one), so "the region's own receipt
    metric" is read against the two measurements it inherits a gate from, gate (a)'s own
    numbers, with a 1-point TOLERANCE rather than gate (a)'s "matches or beats": a memory
    that is up to 1 point under either parent still clears this clause even if it fails
    (a).

    NOT MEASURED HERE: the `banking77` damage-detector probe §4.0 also names. Out of
    scope for row W4's implementation -- recorded explicitly rather than silently
    omitted, so a reader does not mistake `passed=True` for "both halves of clause (2)
    were checked."
    """
    rank_ratio = token_global_pr_rank / pooled_pr_rank if pooled_pr_rank else 0.0
    pr_clause = {
        "token_global_pr_rank": token_global_pr_rank,
        "pooled_pr_rank": pooled_pr_rank,
        "ratio": rank_ratio,
        "required_ratio": 2.0,
        "passed": bool(rank_ratio >= 2.0),
    }
    recall_regression = max(0.0, compress_recall_at_1 - memory_held_out_recall_at_1)
    recall_regression_vs_retrieve = max(0.0, retrieve_recall_at_1 - memory_held_out_recall_at_1)
    graded_regression = max(0.0, compress_graded_spearman - memory_graded_spearman)
    worst_regression = max(recall_regression, recall_regression_vs_retrieve, graded_regression)
    regression_clause = {
        "recall_regression_vs_compress": recall_regression,
        "recall_regression_vs_retrieve": recall_regression_vs_retrieve,
        "graded_regression_vs_compress": graded_regression,
        "worst_regression": worst_regression,
        "margin": margin,
        "passed": bool(worst_regression <= margin),
        "banking77_probe": "not measured; out of scope for row W4's implementation",
    }
    return {
        "gate": "e_retrain_gate",
        "pr_rank_clause": pr_clause,
        "receipt_regression_clause": regression_clause,
        "passed": bool(pr_clause["passed"] and regression_clause["passed"]),
    }


def w4_gates(
    *,
    memory_receipt: dict[str, Any],
    full_pool_trained: dict[str, float],
    full_pool_bm25: dict[str, float],
    full_pool_untrained: dict[str, float],
) -> dict[str, Any]:
    """All five pre-registered W4 gates, read from `memory_receipt` (a `pretrain_region`
    receipt for `memory_config()`, with commit 1's `token_aware` block) plus the three
    full-pool BEIR measurements `regions/memory.py`'s wrapper computes.

    Args:
        memory_receipt: The receipt `pretrain_region` wrote for `memory`.
        full_pool_trained: `encoder_rank_metrics` on the trained checkpoint, full corpus.
        full_pool_bm25: `bm25_metrics` on the same pool/qrels.
        full_pool_untrained: `encoder_rank_metrics` on a freshly-initialised encoder,
            same pool/qrels -- `memory`'s own random-init baseline (gate (4)).

    Returns:
        `{"a_beats_both_parents": ..., "b_full_pool_thresholds": ..., "c_beats_bm25":
        ..., "d_beats_random_init": ..., "e_retrain_gate": ..., "passed": bool}` -- the
        last `passed` is the conjunction of all five.
    """
    held_out = memory_receipt["held_out"]
    graded = memory_receipt.get("graded_held_out")
    if graded is None:
        raise KeyError(
            "memory_receipt has no graded_held_out -- the consolidation head's gate "
            "(gate a compares memory against compress's STS-B spearman) cannot be "
            "evaluated without it"
        )
    rank = memory_receipt.get("token_aware", {}).get("final_block_rank")
    if rank is None:
        raise KeyError(
            "memory_receipt has no token_aware.final_block_rank -- gate (e)'s PR-rank "
            "clause cannot be evaluated without it (see regions/pretrain.py commit 1)"
        )

    a = gate_a_beats_both_parents(held_out["recall@1"], graded["spearman"])
    b = gate_b_full_pool_thresholds(full_pool_trained)
    c = gate_c_beats_bm25(full_pool_trained, full_pool_bm25)
    d = gate_d_beats_random_init(full_pool_trained, full_pool_untrained)
    e = gate_e_retrain_gate(
        rank["token_global_pr_rank"],
        rank["pooled_pr_rank"],
        held_out["recall@1"],
        graded["spearman"],
    )
    gates: dict[str, Any] = {
        "a_beats_both_parents": a,
        "b_full_pool_thresholds": b,
        "c_beats_bm25": c,
        "d_beats_random_init": d,
        "e_retrain_gate": e,
    }
    gates["passed"] = bool(all(g["passed"] for g in gates.values()))
    return gates
