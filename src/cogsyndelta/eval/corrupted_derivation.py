"""E1 reasoning-sensitivity battery: rank a true gsm8k derivation among corruptions.

WHY THIS EXISTS
The reason region's closed 512-pair diagonal is a bag-of-words instrument (diagnosis
2026-09-05 §2: TF-IDF r@1 0.873 vs trained 0.19). That number cannot say whether the
encoder noticed derivation structure. This battery edits one calculator annotation in
a held-out gsm8k solution, optionally propagating the `####` final value, and asks
whether the true derivation still ranks first among {true, K corruptions}. Chance is
0.20 at K=4. TF-IDF/BM25 must sit at chance; a wrong-problem control must be easy for
TF-IDF. That is diagnosis §4 E1, not a new design.

WHAT THIS IS NOT
Not a recall@1 improvement. Not E3 (structure-sensitive training negatives). Not E5
(latent-step prediction). CPU oracles live here; encoder scoring is a GPU caller.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from cogsyndelta.eval.beir_fiqa import BM25
from cogsyndelta.splits import item_id as split_item_id

BATTERY_ID = "eval_corrupted_derivation"
CORRUPTION_SEED = 0
K_CORRUPTIONS = 4
CHANCE = 1.0 / (1 + K_CORRUPTIONS)
MIN_ANNOTATIONS = 2
GO_THRESHOLD = 0.30
KILL_THRESHOLD = 0.25
TFIDF_CHANCE_TOLERANCE = 0.05
WRONG_PROBLEM_TFIDF_MIN = 0.90
TIE_JITTER = 1e-12

CALC_RE = re.compile(r"<<([^<>=]+)=([^<>]+)>>")
NUMBER_RE = re.compile(r"[+-]?(?:\d+\.\d+|\d+)")
FINAL_RE = re.compile(r"(####\s*)([+-]?(?:\d+\.\d+|\d+))")
LEXICAL_WORD_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?")

PREREG_NOTES = (
    "Reasoning as a faculty is NOT lookup; lookup is what the retrieve faculty does. "
    "Reasoning is ruminating on a problem and exploring solutions, questions and "
    "answers. E1 is not trying to raise recall@1; it measures whether any existing "
    "checkpoint is sensitive to a broken derivation step at all (the "
    "corrupted-derivation battery with TF-IDF-at-chance and wrong-problem controls). "
    "E3 (structure-sensitive contrastive negatives) is a lookup improvement, "
    "second-class under this definition. E5 (latent-step prediction, iterative "
    "refinement) is the direction. No change to E0. Go: any checkpoint "
    "derive.recall@1 >= 0.30 -> derivation sensitivity, E3 next. Kill: all "
    "checkpoints derive.recall@1 <= 0.25 -> learned nothing about structure, E5 "
    "ahead of E4. Either way, diagonal r@1 on reason is demoted from a gate to a "
    "lexical-ceiling fraction (model / TF-IDF)."
)
"""Copied into every E1 receipt's detail.notes. Written before scores exist."""


@dataclass(frozen=True)
class CalcAnnotation:
    """One `<<expr=result>>` span inside a gsm8k derivation."""

    start: int
    end: int
    left: str
    result: str
    operand_spans: tuple[tuple[int, int, str], ...]
    result_span: tuple[int, int, str]


@dataclass(frozen=True)
class BatteryItem:
    """One E1 item: question plus true derivation and K corruptions."""

    item_id: str
    question: str
    true_derivation: str
    corruptions: tuple[str, ...]
    n_annotations: int


def w2c_region_seed(region: str) -> int:
    """W2c region-specific untrained-encoder seed.

    Args:
        region: Region id (``reason`` for E1).

    Returns:
        First 8 hex chars of ``sha256("csd-w2c-untrained:<region>")`` as uint32.
    """
    digest = hashlib.sha256(f"csd-w2c-untrained:{region}".encode()).hexdigest()
    return int(digest[:8], 16)


def parse_calculator_annotations(text: str) -> list[CalcAnnotation]:
    """Parse gsm8k ``<<expr=result>>`` annotations.

    Args:
        text: Derivation body, possibly with a ``####`` final line.

    Returns:
        Annotations in left-to-right order. Multi-operand left sides (``100-50-30-15``)
        yield one operand span per number; the spec's "one operand or result" still
        applies.
    """
    out: list[CalcAnnotation] = []
    for match in CALC_RE.finditer(text):
        left = match.group(1)
        result = match.group(2)
        left_abs = match.start() + 2
        operands = tuple(
            (left_abs + m.start(), left_abs + m.end(), m.group(0)) for m in NUMBER_RE.finditer(left)
        )
        result_rel = NUMBER_RE.search(result)
        if result_rel is None:
            continue
        result_abs_start = match.start(2) + result_rel.start()
        result_abs_end = match.start(2) + result_rel.end()
        out.append(
            CalcAnnotation(
                start=match.start(),
                end=match.end(),
                left=left,
                result=result_rel.group(0),
                operand_spans=operands,
                result_span=(result_abs_start, result_abs_end, result_rel.group(0)),
            )
        )
    return out


def _final_match(text: str) -> re.Match[str] | None:
    found = list(FINAL_RE.finditer(text))
    return found[-1] if found else None


def numbers_equal(left: str, right: str) -> bool:
    """True when two numeric spellings denote the same value.

    Args:
        left: First spelling.
        right: Second spelling.

    Returns:
        Float equality when both parse; else stripped string equality.
    """
    try:
        return float(left) == float(right)
    except ValueError:
        return left.strip() == right.strip()


def _format_like(original: str, new: float) -> str:
    if re.fullmatch(r"[+-]?\d+", original):
        return str(round(new))
    if "." in original:
        places = len(original.split(".", 1)[1])
        return f"{new:.{places}f}"
    return str(new)


def _delta_spellings(original: str) -> list[str]:
    try:
        value = float(original)
    except ValueError:
        return []
    is_int = re.fullmatch(r"[+-]?\d+", original) is not None
    candidates: list[float] = []
    if is_int:
        base = round(value)
        candidates.extend(float(base + d) for d in (1, -1, 2, -2, 3, -3, 5, -5, 10, -10))
    else:
        places = len(original.split(".", 1)[1]) if "." in original else 1
        step = 10.0 ** (-places)
        candidates.extend(value + d * step for d in (1, -1, 2, -2, 5, -5, 10, -10))
        candidates.extend((value + 1.0, value - 1.0))
    out: list[str] = []
    seen = {original}
    for cand in candidates:
        spelled = _format_like(original, cand)
        if spelled not in seen:
            seen.add(spelled)
            out.append(spelled)
    return out


def _item_rng(corruption_seed: int, item_id: str, purpose: str) -> random.Random:
    payload = f"{corruption_seed}:{purpose}:{item_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))  # noqa: S311


def _apply_edit(text: str, start: int, end: int, new_spelling: str, original_value: str) -> str:
    edited = text[:start] + new_spelling + text[end:]
    final = _final_match(edited)
    if final is None:
        return edited
    if not numbers_equal(original_value, final.group(2)):
        return edited
    return edited[: final.start(2)] + new_spelling + edited[final.end(2) :]


def _question_numbers(question: str) -> set[str]:
    return set(NUMBER_RE.findall(question))


def _jobs_for_slots(
    slots: list[tuple[int, int, str]],
) -> list[tuple[int, int, str, str]]:
    jobs: list[tuple[int, int, str, str]] = []
    for start, end, original in slots:
        for spelling in _delta_spellings(original):
            jobs.append((start, end, original, spelling))
    return jobs


def corrupt_derivation(
    text: str,
    *,
    k: int = K_CORRUPTIONS,
    corruption_seed: int = CORRUPTION_SEED,
    item_id: str,
    question: str = "",
) -> list[str]:
    """Build ``k`` unique corrupted derivations of ``text``.

    Each corruption edits one result (preferred) or one operand in one
    ``<<...>>`` annotation. Operand edits skip numbers that appear in
    ``question``: changing a question entity is a lexical cue, and control (b)
    requires TF-IDF/BM25 to sit at chance. If the edited token equals the
    ``####`` value, the ``####`` line is updated to the new spelling.

    Args:
        text: True gsm8k derivation.
        k: Number of corruptions.
        corruption_seed: Battery seed (not a CSD training seed).
        item_id: Split item id; mixes into the per-item RNG.
        question: Anchor text; operand values present here are not edited.

    Returns:
        ``k`` distinct corrupted strings.

    Raises:
        ValueError: Fewer than two annotations, or fewer than ``k`` unique edits.
    """
    annotations = parse_calculator_annotations(text)
    if len(annotations) < MIN_ANNOTATIONS:
        raise ValueError(
            f"need >= {MIN_ANNOTATIONS} calculator annotations, found {len(annotations)}"
        )
    q_nums = _question_numbers(question) if question else set()
    result_slots = [ann.result_span for ann in annotations]
    operand_slots = [
        span for ann in annotations for span in ann.operand_spans if span[2] not in q_nums
    ]
    rng = _item_rng(corruption_seed, item_id, "corrupt")
    result_jobs = _jobs_for_slots(result_slots)
    operand_jobs = _jobs_for_slots(operand_slots)
    rng.shuffle(result_jobs)
    rng.shuffle(operand_jobs)
    out: list[str] = []
    seen = {text}
    for start, end, original, spelling in result_jobs + operand_jobs:
        candidate = _apply_edit(text, start, end, spelling, original)
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
            if len(out) == k:
                return out
    raise ValueError(f"only {len(out)} unique corruptions; need {k}")


def is_gsm8k_derivation(positive: str) -> bool:
    """True when ``positive`` looks like a gsm8k worked answer (has ``####``).

    Args:
        positive: Pair right-hand side.

    Returns:
        Whether the positive carries a gsm8k final-answer marker.
    """
    return "####" in positive


def select_gsm8k_holdout(
    holdout: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Keep held-out pairs whose positive is a gsm8k derivation.

    Args:
        holdout: E0 holdout pairs.

    Returns:
        gsm8k subset, order preserved.
    """
    return [(a, b) for a, b in holdout if is_gsm8k_derivation(b)]


def build_corrupted_battery(
    holdout: list[tuple[str, str]],
    *,
    k: int = K_CORRUPTIONS,
    corruption_seed: int = CORRUPTION_SEED,
    min_annotations: int = MIN_ANNOTATIONS,
) -> list[BatteryItem]:
    """Build the E1 battery from an E0 holdout.

    Args:
        holdout: Fixed split pairs (anchor, positive).
        k: Corruptions per item.
        corruption_seed: Deterministic battery seed.
        min_annotations: Eligibility floor (diagnosis: ≥ 2).

    Returns:
        Items with ≥ ``min_annotations`` calculator annotations and ``k`` corruptions.
    """
    items: list[BatteryItem] = []
    for question, derivation in select_gsm8k_holdout(holdout):
        annotations = parse_calculator_annotations(derivation)
        if len(annotations) < min_annotations:
            continue
        iid = split_item_id(question, derivation)
        corruptions = corrupt_derivation(
            derivation,
            k=k,
            corruption_seed=corruption_seed,
            item_id=iid,
            question=question,
        )
        items.append(
            BatteryItem(
                item_id=iid,
                question=question,
                true_derivation=derivation,
                corruptions=tuple(corruptions),
                n_annotations=len(annotations),
            )
        )
    return items


def wrong_problem_candidates(
    items: list[BatteryItem],
    *,
    k: int = K_CORRUPTIONS,
    corruption_seed: int = CORRUPTION_SEED,
) -> list[tuple[str, ...]]:
    """Per item, ``k`` other items' true derivations (control a).

    Args:
        items: Corrupted battery.
        k: Distractors per item.
        corruption_seed: Same battery seed.

    Returns:
        One tuple of distractor derivations per item, same order as ``items``.

    Raises:
        ValueError: Fewer than ``k + 1`` items.
    """
    n = len(items)
    if n < k + 1:
        raise ValueError(f"wrong-problem control needs >= {k + 1} items, got {n}")
    out: list[tuple[str, ...]] = []
    for i, item in enumerate(items):
        rng = _item_rng(corruption_seed, item.item_id, "wrong")
        others = [j for j in range(n) if j != i]
        rng.shuffle(others)
        picked = tuple(items[j].true_derivation for j in others[:k])
        out.append(picked)
    return out


def lexical_tokens(text: str) -> list[str]:
    """Word tokens for TF-IDF (letters, or integers/decimals).

    Args:
        text: Raw string.

    Returns:
        Lower-cased tokens. Matches the diagnosis lexical scorer's token class.
    """
    return LEXICAL_WORD_RE.findall(text.lower())


def tfidf_scores(query: str, docs: list[str]) -> np.ndarray:
    """TF-IDF overlap of ``query`` against ``docs``, IDF over the candidate pool.

    Cosine (L2) is the wrong metric on this battery: a corruption introduces a
    number the question does not contain, which increases ``||d||`` and drops
    cosine even though query overlap is unchanged -- a dilution artifact, not
    lexical solving. The score is therefore the **unnormalized TF-IDF dot
    product** (query overlap). On near-duplicate corruptions that do not touch
    question terms, scores tie and ``hit_rate_at_1``'s unbiased jitter sits at
    chance; on the wrong-problem control, question entities still pick the
    matching derivation.

    Args:
        query: Question.
        docs: Candidate derivations.

    Returns:
        Shape ``[len(docs)]`` TF-IDF overlap scores.
    """
    doc_tokens = [lexical_tokens(d) for d in docs]
    query_tokens = lexical_tokens(query)
    df: Counter[str] = Counter()
    for tokens in doc_tokens:
        df.update(set(tokens))
    n_docs = max(len(docs), 1)
    vocab = {tok: i for i, tok in enumerate(df)}
    idf = np.array(
        [math.log((n_docs + 1) / (df[tok] + 1)) + 1.0 for tok in vocab],
        dtype=np.float64,
    )

    def vector(tokens: list[str]) -> np.ndarray:
        """Unnormalized TF-IDF bag for one token list."""
        vec = np.zeros(len(vocab), dtype=np.float64)
        counts = Counter(tokens)
        for tok, tf in counts.items():
            idx = vocab.get(tok)
            if idx is not None:
                vec[idx] = (1.0 + math.log(tf)) * idf[idx]
        return vec

    qv = vector(query_tokens)
    stacked = np.stack([vector(t) for t in doc_tokens])
    return stacked @ qv


def bm25_scores(query: str, docs: list[str]) -> np.ndarray:
    """Okapi BM25 of ``query`` against ``docs`` (project BM25, BEIR k1/b).

    Args:
        query: Question.
        docs: Candidate derivations.

    Returns:
        Shape ``[len(docs)]`` scores.
    """
    return BM25(docs).scores(query)


def hit_rate_at_1(
    scores: np.ndarray,
    *,
    relevant: int = 0,
    tie_seed: int = CORRUPTION_SEED,
) -> float:
    """Fraction of rows whose relevant column ranks first after unbiased tie-break.

    Exact ties get a deterministic jitter from ``tie_seed`` so a 5-way tie has
    expectation 0.20 rather than always hitting column 0.

    Args:
        scores: ``[N, C]``, higher is better.
        relevant: Column index of the true candidate (0 = true derivation first).
        tie_seed: Jitter seed.

    Returns:
        Hit rate in ``[0, 1]``.
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-d, got {scores.shape}")
    rng = np.random.default_rng(tie_seed)
    jitter = rng.uniform(0.0, TIE_JITTER, size=scores.shape)
    ranked = np.argmax(scores + jitter, axis=1)
    return float((ranked == relevant).mean()) if scores.size else 0.0


def score_items_lexical(
    items: list[BatteryItem],
    *,
    scorer: str,
    corruption_seed: int = CORRUPTION_SEED,
) -> dict[str, float]:
    """TF-IDF or BM25 recall@1 on the corrupted battery (control b).

    Args:
        items: E1 items.
        scorer: ``tfidf`` or ``bm25``.
        corruption_seed: Tie-break seed.

    Returns:
        ``recall@1``, ``n_items``, ``chance``.

    Raises:
        ValueError: Unknown scorer.
    """
    rows: list[np.ndarray] = []
    for item in items:
        docs = [item.true_derivation, *item.corruptions]
        if scorer == "tfidf":
            rows.append(tfidf_scores(item.question, docs))
        elif scorer == "bm25":
            rows.append(bm25_scores(item.question, docs))
        else:
            raise ValueError(f"unknown scorer {scorer!r}")
    scores = np.stack(rows) if rows else np.zeros((0, 1 + K_CORRUPTIONS))
    return {
        "recall@1": hit_rate_at_1(scores, relevant=0, tie_seed=corruption_seed),
        "n_items": float(len(items)),
        "chance": CHANCE,
    }


def score_wrong_problem_tfidf(
    items: list[BatteryItem],
    *,
    k: int = K_CORRUPTIONS,
    corruption_seed: int = CORRUPTION_SEED,
) -> dict[str, float]:
    """TF-IDF recall@1 on true vs other problems' derivations (control a).

    Args:
        items: E1 items.
        k: Distractors per item.
        corruption_seed: Distractor draw and tie-break seed.

    Returns:
        ``recall@1``, ``n_items``.
    """
    distractors = wrong_problem_candidates(items, k=k, corruption_seed=corruption_seed)
    rows: list[np.ndarray] = []
    for item, others in zip(items, distractors, strict=True):
        docs = [item.true_derivation, *others]
        rows.append(tfidf_scores(item.question, docs))
    scores = np.stack(rows) if rows else np.zeros((0, 1 + k))
    return {
        "recall@1": hit_rate_at_1(scores, relevant=0, tie_seed=corruption_seed),
        "n_items": float(len(items)),
        "chance": CHANCE,
    }


def cosine_hit_rate(
    query_vecs: np.ndarray,
    candidate_vecs: np.ndarray,
    *,
    k: int = K_CORRUPTIONS,
    tie_seed: int = CORRUPTION_SEED,
) -> dict[str, float]:
    """Rank true (index 0) among ``1+k`` L2-normalised candidate vectors per item.

    Args:
        query_vecs: ``[N, D]`` question embeddings, already L2-normalised.
        candidate_vecs: ``[N, 1+k, D]`` candidates, true first, already L2-normalised.
        k: Corruptions per item.
        tie_seed: Tie-break seed.

    Returns:
        ``recall@1``, ``mrr``, ``n_items``, ``chance``.
    """
    n_items, n_cand, dim = candidate_vecs.shape
    if n_cand != 1 + k:
        raise ValueError(f"expected {1 + k} candidates, got {n_cand}")
    if query_vecs.shape != (n_items, dim):
        raise ValueError(f"query_vecs {query_vecs.shape} vs candidates {candidate_vecs.shape}")
    scores = np.einsum("nd,ncd->nc", query_vecs, candidate_vecs)
    recall = hit_rate_at_1(scores, relevant=0, tie_seed=tie_seed)
    rng = np.random.default_rng(tie_seed)
    jitter = rng.uniform(0.0, TIE_JITTER, size=scores.shape)
    order = np.argsort(-(scores + jitter), axis=1)
    ranks = np.argmax(order == 0, axis=1) + 1
    mrr = float((1.0 / ranks).mean()) if n_items else 0.0
    return {
        "recall@1": recall,
        "mrr": mrr,
        "n_items": float(n_items),
        "chance": CHANCE,
    }


def control_gates(
    tfidf_r1: float,
    bm25_r1: float,
    wrong_problem_r1: float,
) -> dict[str, bool]:
    """Instrument-validity gates (diagnosis §4 controls a and b).

    Args:
        tfidf_r1: TF-IDF on the corrupted battery.
        bm25_r1: BM25 on the corrupted battery.
        wrong_problem_r1: TF-IDF on the wrong-problem control.

    Returns:
        Named booleans. A run with any False is not a valid sensitivity measurement.
    """
    return {
        "tfidf_at_chance": abs(tfidf_r1 - CHANCE) <= TFIDF_CHANCE_TOLERANCE,
        "bm25_at_chance": abs(bm25_r1 - CHANCE) <= TFIDF_CHANCE_TOLERANCE,
        "wrong_problem_solvable": wrong_problem_r1 >= WRONG_PROBLEM_TFIDF_MIN,
    }


def go_kill(arm_scores: dict[str, float]) -> dict[str, Any]:
    """Apply the pre-registered go/kill rule to per-arm ``derive.recall@1``.

    Args:
        arm_scores: Arm name -> recall@1.

    Returns:
        Verdict plus the triggering arms. ``none`` if every arm is in (0.25, 0.30).
    """
    go_arms = [name for name, value in arm_scores.items() if value >= GO_THRESHOLD]
    kill_arms = [name for name, value in arm_scores.items() if value <= KILL_THRESHOLD]
    if go_arms:
        verdict = "go"
    elif arm_scores and all(value <= KILL_THRESHOLD for value in arm_scores.values()):
        verdict = "kill"
    else:
        verdict = "none"
    return {
        "verdict": verdict,
        "go_arms": go_arms,
        "kill_arms": kill_arms,
        "go_threshold": GO_THRESHOLD,
        "kill_threshold": KILL_THRESHOLD,
        "interpretation": (
            "derivation sensitivity, E3 next"
            if verdict == "go"
            else (
                "learned nothing about structure, E5 ahead of E4"
                if verdict == "kill"
                else "neither go nor kill; interval (0.25, 0.30)"
            )
        ),
    }


def battery_fingerprint(items: list[BatteryItem]) -> str:
    """Sha256 of item ids and corruption texts; membership of the scored battery.

    Args:
        items: E1 items.

    Returns:
        64-char hex digest.
    """
    lines: list[str] = []
    for item in items:
        lines.append(item.item_id)
        lines.extend(item.corruptions)
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
