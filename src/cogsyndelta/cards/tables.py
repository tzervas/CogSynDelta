"""Metric tables: train/eval/eval-quantized/quant receipts, grouped by category, under
v2 names, with an untrained-baseline column, optional region-comparison columns, and
numbered footnotes into `docs/design/METRICS-METHODOLOGY.md`.

CARD SPEC (operator, 2026-09-04), section 5: "tables grouped by category (retrieval
rank.*, efficiency eff.*, representation repr.*, quantization quant.*, beir.* /
token.* when present), columns = this variant | untrained baseline | other variants of
the same region when a region table is supplied, best value bold, numbered
methodology footnotes per row group ... A metric with no methodology entry refuses to
render. The metrics_schema line must agree with EVERY receipt merged into the tables
or the card refuses to render. A v1 receipt renders through the alias map with an
explicit footnote 'v1 receipt; names mapped to csd-metrics/v2'."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cogsyndelta.cards.methodology import (
    EVAL_GATE_ALIASES_V1,
    EVAL_METRIC_ALIASES_V1,
    METRIC_METHODOLOGY,
    RETIRED_RANK_METRICS,
    CardError,
    MetricMethodology,
    methodology_key,
    require_documented,
)

V1_LEGACY_SCHEMA = "csd-metrics/v1 (not recorded)"
"""What a receipt with no `metrics_schema` field at all reads back as -- the fallback
`scripts/csd-publish-checkpoint.py`'s own `_metrics_schema_line` already uses."""

V1_FOOTNOTE = (
    "v1 receipt; names mapped to csd-metrics/v2 (see "
    "`docs/design/METRICS-METHODOLOGY.md` §15, the v1 -> v2 deprecation map)."
)

# Category prefix -> human heading, in the print order CARD SPEC §5 names.
CATEGORY_HEADINGS: dict[str, str] = {
    "rank": "Retrieval",
    "eff": "Efficiency",
    "repr": "Representation",
    "quant": "Quantization",
    "beir": "External retrieval (BEIR-style)",
    "token": "Token-aware",
}
CATEGORY_ORDER: tuple[str, ...] = ("rank", "eff", "repr", "quant", "beir", "token")

#: Metrics where a SMALLER number is the better one -- the default is "larger is
#: better" (recall, mrr, throughput, ...); this is the deliberately short exception
#: list. `within_budget`/booleans are excluded entirely from "best" bolding (see
#: `_is_boolean`).
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "eff.latency_p50_ms",
        "eff.latency_p95_ms",
        "eff.latency_p99_ms",
        "eff.peak_vram_mb",
        "eff.stored_mb",
        "repr.anisotropy",
        "quant.drop_recall@1",
        "drop",
    }
)


def metrics_schema_of(receipt: dict[str, Any] | None) -> str | None:
    """A receipt's own `metrics_schema` stamp, or the v1 fallback string for a receipt
    that predates the stamp. `None` only when `receipt` itself is `None` (not supplied
    at all -- distinct from "supplied but unstamped").
    """
    if receipt is None:
        return None
    return str(receipt.get("metrics_schema") or V1_LEGACY_SCHEMA)


def assert_schemas_agree(receipts: dict[str, dict[str, Any] | None]) -> str:
    """Require every non-`None` receipt in `receipts` (keyed by a label for the error
    message, e.g. `{"train": ..., "eval": ..., "quant": ...}`) to carry the SAME
    `metrics_schema` stamp (or the same v1 fallback), and return it.

    This is the stricter, refuse-closed rule CARD SPEC §5 states for this library --
    deliberately different from `scripts/csd-publish-checkpoint.py`'s own
    `_metrics_schema_line`, which renders a `train=... eval=... quant=...` breakdown on
    disagreement rather than refusing (that script's own docstring explains why: it
    must keep publishing already-trained checkpoints whose receipts predate a schema
    migration). This library has no such backward-compatibility obligation to an
    existing publish pipeline, so it enforces the stricter rule the operator specified.
    """
    labelled = [(label, metrics_schema_of(r)) for label, r in receipts.items() if r is not None]
    schemas = {s for _, s in labelled}
    if len(schemas) > 1:
        per_receipt = ", ".join(f"{label}={s!r}" for label, s in labelled)
        raise CardError(
            f"receipts disagree on metrics_schema -- refusing to render: {per_receipt}. "
            "Every receipt merged into one card's tables must carry the same schema "
            "stamp; re-measure the stale receipt(s) before rendering, or render "
            "separate cards."
        )
    if not schemas:
        raise CardError("no receipts supplied -- nothing to render a metrics_schema for")
    return next(iter(schemas))


@dataclass(frozen=True)
class MetricRow:
    """One printed row: a metric key, this variant's value, and what it is compared
    against (an untrained baseline, other variants of the same region, or both).
    """

    key: str
    """The v2 canonical (or v2-mapped) metric key this row is keyed on."""
    variant: float | bool | None
    baseline: float | bool | None = None
    comparators: dict[str, float | bool] | None = None
    is_v1_mapped: bool = False
    """True iff this row's key was reached by aliasing a v1 receipt field name to its
    v2 counterpart -- drives the per-row v1 footnote."""


@dataclass(frozen=True)
class MetricTable:
    """One category's rows (e.g. every `rank.*` row), with the heading a card prints
    above them.
    """

    category: str
    """One of `CATEGORY_ORDER`."""
    heading: str
    rows: list[MetricRow]


def _is_boolean(v: Any) -> bool:
    return isinstance(v, bool)


def _category_of(key: str) -> str:
    prefix = key.split(".", 1)[0]
    return prefix if prefix in CATEGORY_HEADINGS else "other"


def normalize_eval_metrics_v1(metrics: dict[str, Any]) -> dict[str, Any]:
    """`metrics` with every `EVAL_METRIC_ALIASES_V1` v1 key ALSO present under its v2
    name (copy, never overwrite -- same discipline as `normalize_quant_receipt_v1`).
    """
    out = dict(metrics)
    for old, new in EVAL_METRIC_ALIASES_V1.items():
        if old in out and new not in out:
            out[new] = out[old]
    return out


def normalize_eval_gates_v1(gates: dict[str, Any]) -> dict[str, Any]:
    """`gates` with every `EVAL_GATE_ALIASES_V1` v1 key ALSO present under its v2
    name (copy, never overwrite -- same discipline as `normalize_eval_metrics_v1` /
    `normalize_quant_receipt_v1`).
    """
    out = dict(gates)
    for old, new in EVAL_GATE_ALIASES_V1.items():
        if old in out and new not in out:
            out[new] = out[old]
    return out


def build_eval_tables(
    *,
    eval_receipt: dict[str, Any] | None,
    baseline_eval_receipt: dict[str, Any] | None = None,
    comparators: dict[str, dict[str, Any]] | None = None,
    methodology: dict[str, MetricMethodology] | None = None,
    heading_suffix: str = "",
) -> list[MetricTable]:
    """Grouped `rank.*` / `eff.*` / `repr.*` / `quant.*` / `beir.*` tables from an eval
    (or eval-quantized) receipt's `metrics` dict, each row carrying this variant's
    value, the untrained-baseline eval receipt's value for the same key (if supplied
    and if that key is a chance/baseline metric present there), and one column per
    `comparators` entry (region name -> that region's own eval receipt).

    Every v1-shaped legacy key (`repr.effective_rank`, `repr.effective_rank_ratio`) is
    rendered under its v2 name via `EVAL_METRIC_ALIASES_V1`, marked `is_v1_mapped` so
    `render.py` can print the v1 footnote. Raises `CardError` (via
    `require_documented`) if any key this function would print has no
    `METRIC_METHODOLOGY` entry.

    Args:
        eval_receipt: the eval (or eval-quantized) receipt to tabulate, or `None` to
            return no tables at all (a card with no eval receipt supplied).
        baseline_eval_receipt: an eval receipt whose values fill the "untrained
            baseline" column, when one is available at the eval-battery level.
        comparators: `{variant_label: eval_receipt}` -- one extra column per entry,
            for other cells of the SAME region (CARD SPEC §5).
        methodology: overrides `METRIC_METHODOLOGY` for this call only -- a test's
            mutation-proof hook; production callers leave this `None`.
        heading_suffix: appended to every table heading (e.g. `" (fp32)"` /
            `" (quantized artifact)"`) -- required whenever a caller renders BOTH an
            eval and an eval-quantized receipt's tables on one card, so two otherwise
            identically-headed "Retrieval" tables are distinguishable.
    """
    if eval_receipt is None:
        return []
    raw_metrics = eval_receipt.get("metrics", {})
    v1_keys = {k for k in raw_metrics if k in EVAL_METRIC_ALIASES_V1}
    metrics = normalize_eval_metrics_v1(raw_metrics)
    # Drop the bare legacy key once its v2 alias exists -- the table prints one row per
    # concept, under its v2 name, not the v1 name AND the v2 name as two rows.
    for old in EVAL_METRIC_ALIASES_V1:
        if old in metrics and EVAL_METRIC_ALIASES_V1[old] in metrics:
            del metrics[old]
    # `rank.map` / `rank.precision@10`: RETIRED (METRICS-METHODOLOGY.md §13) as
    # independently displayed columns, not undocumented -- a legacy receipt that still
    # carries them (a receipt this card's metrics_schema line already marks v1) is
    # trimmed here rather than raising CardError for them.
    for key in list(metrics):
        if key.startswith("rank.") and key.split(".", 1)[1] in RETIRED_RANK_METRICS:
            del metrics[key]

    baseline_metrics = (
        normalize_eval_metrics_v1(baseline_eval_receipt.get("metrics", {}))
        if baseline_eval_receipt is not None
        else {}
    )
    comparator_metrics = {
        name: normalize_eval_metrics_v1(r.get("metrics", {}))
        for name, r in (comparators or {}).items()
    }

    require_documented((methodology_key(k) for k in metrics), methodology=methodology)

    tables: dict[str, list[MetricRow]] = {}
    for key in sorted(metrics):
        was_v1 = any(EVAL_METRIC_ALIASES_V1.get(old) == key and old in v1_keys for old in v1_keys)
        row = MetricRow(
            key=key,
            variant=metrics[key],
            baseline=baseline_metrics.get(key),
            comparators={n: m[key] for n, m in comparator_metrics.items() if key in m} or None,
            is_v1_mapped=was_v1,
        )
        tables.setdefault(_category_of(key), []).append(row)

    return [
        MetricTable(
            category=cat,
            heading=CATEGORY_HEADINGS.get(cat, cat.title()) + heading_suffix,
            rows=tables[cat],
        )
        for cat in (*CATEGORY_ORDER, "other")
        if cat in tables
    ]


def build_gate_table(
    eval_receipt: dict[str, Any] | None,
    *,
    methodology: dict[str, MetricMethodology] | None = None,
) -> MetricTable | None:
    """The eval receipt's `gates` dict as one table (not category-prefixed -- gate
    names are bare, e.g. `beats_untrained_eval`).
    """
    if eval_receipt is None:
        return None
    raw_gates = eval_receipt.get("gates", {})
    if not raw_gates:
        return None
    v1_keys = {k for k in raw_gates if k in EVAL_GATE_ALIASES_V1}
    gates = normalize_eval_gates_v1(raw_gates)
    for old in EVAL_GATE_ALIASES_V1:
        if old in gates and EVAL_GATE_ALIASES_V1[old] in gates:
            del gates[old]
    require_documented(gates.keys(), methodology=methodology)
    rows = [
        MetricRow(
            key=key,
            variant=gates[key],
            is_v1_mapped=any(
                EVAL_GATE_ALIASES_V1.get(old) == key and old in v1_keys for old in v1_keys
            ),
        )
        for key in sorted(gates)
    ]
    return MetricTable(category="gates", heading="Gates", rows=rows)


def build_training_table(
    train_receipt: dict[str, Any],
    *,
    methodology: dict[str, MetricMethodology] | None = None,
) -> MetricTable:
    """`held_out` vs `untrained_baseline` from the training receipt itself -- the
    training-held-out battery table (`train_holdout`), separate from the eval battery
    tables `build_eval_tables` produces. Always non-empty for a real training receipt
    (`held_out`/`untrained_baseline` are required fields of that receipt shape).
    """
    held_out = train_receipt.get("held_out", {})
    baseline = train_receipt.get("untrained_baseline", {})
    keys = sorted(set(held_out) | set(baseline))
    require_documented(keys, methodology=methodology)
    rows = [MetricRow(key=k, variant=held_out.get(k), baseline=baseline.get(k)) for k in keys]
    return MetricTable(category="held_out", heading="Training held-out battery", rows=rows)


def build_quant_table(
    quant_receipt: dict[str, Any] | None,
    *,
    methodology: dict[str, MetricMethodology] | None = None,
) -> MetricTable | None:
    """The quantize-stage receipt's own metrics (`fp32_metric_recomputed`,
    `quant.plan_recall@1`, `quant.drop_recall@1`, `tolerance`, `within_budget`,
    `quant.compression_ratio`, `fp32_bytes`, `stored_bytes`) -- separate from
    `quant.artifact_recall@1`, which lives in an eval-quantized receipt's `metrics` and
    is picked up by `build_eval_tables` instead (MM §4: different battery, never
    merged into one row-for-row table with the quant_plan numbers).

    Assumes `quant_receipt` has already been passed through
    `cogsyndelta.cards.methodology.normalize_quant_receipt_v1` -- same precondition
    `scripts/csd-publish-checkpoint.py`'s `_card_metric_keys` documents for the same
    reason.
    """
    if quant_receipt is None:
        return None
    wanted = (
        "fp32_metric_recomputed",
        "quant.plan_recall@1",
        "quant.drop_recall@1",
        "tolerance",
        "within_budget",
        "quant.compression_ratio",
        "fp32_bytes",
        "stored_bytes",
    )
    present = [k for k in wanted if k in quant_receipt]
    require_documented(present, methodology=methodology)
    rows = [MetricRow(key=k, variant=quant_receipt[k]) for k in present]
    return MetricTable(
        category="quant_plan", heading="Quantization (quant_plan battery)", rows=rows
    )


def _fmt_value(v: float | bool | None) -> str:
    if v is None:
        return "_(n/a)_"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _best_column(row: MetricRow) -> str | None:
    """Which column (`"variant"`, `"baseline"`, or a comparator name) holds the best
    numeric value in this row, for bolding -- `None` for a boolean row (nothing to
    compare) or a row with fewer than two numeric values to compare.
    """
    candidates: dict[str, float] = {}
    if isinstance(row.variant, int | float) and not _is_boolean(row.variant):
        candidates["variant"] = float(row.variant)
    if isinstance(row.baseline, int | float) and not _is_boolean(row.baseline):
        candidates["baseline"] = float(row.baseline)
    for name, v in (row.comparators or {}).items():
        if isinstance(v, int | float) and not _is_boolean(v):
            candidates[name] = float(v)
    if len(candidates) < 2:
        return None
    lower_is_better = row.key in LOWER_IS_BETTER
    return (
        min(candidates, key=lambda k: candidates[k])
        if lower_is_better
        else max(candidates, key=lambda k: candidates[k])
    )


def render_table_markdown(
    table: MetricTable,
    *,
    methodology: dict[str, MetricMethodology] | None = None,
    footnote_numbers: dict[tuple[str, str, str, str], int],
) -> str:
    """One category's table as a markdown pipe table, plus footnote markers
    (`[^N]`) resolved against `footnote_numbers` (populated by `render.py`'s
    `assign_footnotes`, shared across every table in one card so the same
    (definition, battery_id, pooling, source) tuple gets one footnote number
    card-wide, not one per table).
    """
    lookup = METRIC_METHODOLOGY if methodology is None else methodology
    comparator_names = sorted({n for r in table.rows for n in (r.comparators or {})})
    header = ["metric", "this variant", "untrained baseline", *comparator_names]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in table.rows:
        m = lookup[methodology_key(row.key)]
        fnkey = (m.definition, m.battery_id, m.pooling, m.source)
        n = footnote_numbers[fnkey]
        best = _best_column(row)
        v1_mark = " ^v1^" if row.is_v1_mapped else ""
        cells = [f"`{row.key}`[^{n}]{v1_mark}"]
        for col, val in (("variant", row.variant), ("baseline", row.baseline)):
            text = _fmt_value(val)
            cells.append(f"**{text}**" if best == col else text)
        for name in comparator_names:
            val = (row.comparators or {}).get(name)
            text = _fmt_value(val)
            cells.append(f"**{text}**" if best == name else text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
