"""Model-card building blocks for CogSynDelta regions and the composed mind.

A library, not a CLI -- `scripts/csd-publish-checkpoint.py` and any future publisher
import from here rather than building a markdown string by hand.

- `methodology`: the `METRIC_METHODOLOGY` reference table, `MetricMethodology`, the
  v1 -> v2 receipt-field alias maps, and `CardError` -- the refuse-closed exception
  every other submodule raises when a card would print an undocumented or
  disagreeing-schema number.

Further submodules (`sizes`, `metadata`, `tables`, `render`) land in a later commit of
this same lane; this package starts with only what
`scripts/csd-publish-checkpoint.py`'s import switch needs.
"""

from __future__ import annotations

from cogsyndelta.cards.methodology import (
    METHODOLOGY_DOC,
    METRIC_METHODOLOGY,
    CardError,
    MetricMethodology,
    normalize_quant_receipt_v1,
)

__all__ = [
    "METHODOLOGY_DOC",
    "METRIC_METHODOLOGY",
    "CardError",
    "MetricMethodology",
    "normalize_quant_receipt_v1",
]
