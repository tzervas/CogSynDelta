"""Model-card building blocks for CogSynDelta regions and the composed mind.

A library, not a CLI (see the lane's own task description) -- `scripts/
csd-publish-checkpoint.py` and any future publisher import from here rather than
building a markdown string by hand. Submodules:

- `methodology`: the `METRIC_METHODOLOGY` reference table, `MetricMethodology`, the
  v1 -> v2 receipt-field alias maps, and `CardError` -- the refuse-closed exception
  every other submodule raises when a card would print an undocumented or
  disagreeing-schema number.
- `sizes`: parameter count, bit plan, disk-per-state, and MEASURED memory (never
  estimated) from receipts and `/akula-data/csd/matrix/budgets`.
- `metadata`: `huggingface_hub.ModelCardData` construction, including model-index
  `EvalResult`s under `csd-metrics/v2` names.
- `tables`: per-category metric tables (rank./eff./repr./quant./beir./token.) with an
  untrained-baseline column, optional region-comparison rows, and numbered footnotes
  into `docs/design/METRICS-METHODOLOGY.md`.
- `render`: `render_card()`, the one public entry point -- picks the Jinja2 template
  for a card `kind` and calls `ModelCard.from_template`.
"""

from __future__ import annotations

from cogsyndelta.cards.metadata import build_card_data, build_eval_results
from cogsyndelta.cards.methodology import (
    METHODOLOGY_DOC,
    METRIC_METHODOLOGY,
    CardError,
    MetricMethodology,
    normalize_quant_receipt_v1,
)
from cogsyndelta.cards.render import CARD_KINDS, render_card
from cogsyndelta.cards.sizes import SizeReport, build_size_report
from cogsyndelta.cards.tables import MetricRow, MetricTable, build_eval_tables

__all__ = [
    "CARD_KINDS",
    "METHODOLOGY_DOC",
    "METRIC_METHODOLOGY",
    "CardError",
    "MetricMethodology",
    "MetricRow",
    "MetricTable",
    "SizeReport",
    "build_card_data",
    "build_eval_results",
    "build_eval_tables",
    "build_size_report",
    "normalize_quant_receipt_v1",
    "render_card",
]
