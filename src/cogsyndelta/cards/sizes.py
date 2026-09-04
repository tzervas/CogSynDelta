"""Sizes: parameter count, bit plan, disk-per-state, and MEASURED memory.

Every number here is read directly off a receipt, or off a budgets file this
project's own training/eval/quant tools wrote -- never estimated. CARD SPEC (operator,
2026-09-04), section 6: "MEASURED memory (eval peak VRAM fp32 and quantized from
eff.peak_vram_mb; training peak from budgets/<region>/<key>.json report_peak) each
labelled with batch and max_len, never estimated." `build_size_report` below is the one
call site `cogsyndelta.cards.render` uses; the smaller functions are exported too, for
a caller (or a test) that wants one number without building the whole report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cogsyndelta.cards.methodology import CardError

DEFAULT_BUDGETS_ROOT = Path("/akula-data/csd/matrix/budgets")
"""Where `scripts/csd-train-all.py` (or the matrix harness) records each region's
measured training peak, one file per (region, batch_size) cell:
`<root>/<region>/batch=<batch_size>.json`, `{"cell_id", "mib", "source", "ts"}`."""


@dataclass(frozen=True)
class MeasuredValue:
    """One MEASURED number plus the label a card must print beside it -- never a bare
    float, so a template cannot accidentally print a number with no stated conditions.
    """

    value: float
    label: str
    unit: str = "MB"


@dataclass(frozen=True)
class SizeReport:
    """Every size/memory fact `render.py`'s `sizes` template block needs, for one
    checkpoint (+ optional quantized artifact).
    """

    region: str
    parameters: int | None
    fp32_bytes: int | None
    quantized_stored_bytes: int | None
    compression_ratio: float | None
    width_histogram: dict[str, int] | None
    eval_peak_vram: dict[str, MeasuredValue] = field(default_factory=dict)
    """`{"fp32": MeasuredValue(...), "quantized": MeasuredValue(...)}` -- absent keys
    mean no eval(-quantized) receipt was supplied, not that memory was zero."""
    training_peak: MeasuredValue | None = None


def parameter_count(train_receipt: dict[str, Any]) -> int | None:
    """`parameters` on a training receipt -- the total parameter count `csd-train-all.py`
    counted for this checkpoint. `None` (never a guessed number) if the receipt does not
    carry it.
    """
    v = train_receipt.get("parameters")
    return int(v) if v is not None else None


def width_histogram(quant_receipt: dict[str, Any] | None) -> dict[str, int] | None:
    """The packed artifact's bits -> tensor-count histogram, straight off the quant
    receipt. `None` when there is no quant receipt (an fp32-only card); raises
    `CardError` for a quant receipt that omits the field entirely -- a card printing a
    "bit plan" section with no histogram to back it is exactly the unmeasured-number
    class this package refuses.
    """
    if quant_receipt is None:
        return None
    hist = quant_receipt.get("width_histogram")
    if hist is None:
        raise CardError(
            "quant receipt has no width_histogram -- refusing to print a bit plan "
            "with nothing to back it"
        )
    return {str(k): int(v) for k, v in dict(hist).items()}


def disk_bytes_per_state(
    train_receipt: dict[str, Any], quant_receipt: dict[str, Any] | None
) -> dict[str, int]:
    """Weights-on-disk per precision state, in bytes: `{"fp32": ..., "csd-ptq-v1":
    ...}` (the second key only present when `quant_receipt` is supplied). Both numbers
    come straight from the quant receipt's own byte accounting (`fp32_bytes`,
    `stored_bytes`) -- the same figures `scripts/csd-publish-checkpoint.py` re-measures
    against the packed artifact before it will publish.
    """
    out: dict[str, int] = {}
    if quant_receipt is not None:
        fp32 = quant_receipt.get("fp32_bytes")
        if fp32 is not None:
            out["fp32"] = int(fp32)
        stored = quant_receipt.get("stored_bytes")
        if stored is not None:
            out["csd-ptq-v1"] = int(stored)
    return out


def compression_ratio(quant_receipt: dict[str, Any] | None) -> float | None:
    """`quant.compression_ratio` (v2) or `compression_ratio` (v1) -- a payload/storage
    ratio, never a speed claim (see `METRIC_METHODOLOGY["quant.compression_ratio"]`).
    """
    if quant_receipt is None:
        return None
    v = quant_receipt.get("quant.compression_ratio", quant_receipt.get("compression_ratio"))
    return float(v) if v is not None else None


def _train_batch_and_max_len(train_receipt: dict[str, Any]) -> tuple[int | None, int | None]:
    cfg = train_receipt.get("config", {})
    batch = cfg.get("batch_size")
    max_len = cfg.get("max_len") or cfg.get("encoder", {}).get("max_len")
    return (
        int(batch) if batch is not None else None,
        int(max_len) if max_len is not None else None,
    )


def eval_peak_vram(
    eval_receipt: dict[str, Any] | None,
    eval_quantized_receipt: dict[str, Any] | None,
    *,
    max_len: int | None = None,
    holdout_pairs: int | None = None,
) -> dict[str, MeasuredValue]:
    """`eff.peak_vram_mb` off the fp32 eval receipt and/or the eval-quantized receipt,
    each labelled with the holdout batch size (`rank.candidates`, if present) and
    `max_len` -- so the card never prints a bare "694 MB" with no stated conditions.
    """
    out: dict[str, MeasuredValue] = {}
    for key, receipt in (("fp32", eval_receipt), ("quantized", eval_quantized_receipt)):
        if receipt is None:
            continue
        metrics = receipt.get("metrics", {})
        v = metrics.get("eff.peak_vram_mb")
        if v is None:
            continue
        candidates = metrics.get("rank.candidates")
        n = int(candidates) if candidates is not None else holdout_pairs
        label_bits = []
        if n is not None:
            label_bits.append(f"eval batch {n}")
        if max_len is not None:
            label_bits.append(f"max_len {max_len}")
        label = "eval peak, " + ", ".join(label_bits) if label_bits else "eval peak"
        out[key] = MeasuredValue(value=float(v), label=label)
    return out


def training_peak(
    region: str,
    batch_size: int | None,
    *,
    max_len: int | None = None,
    root: Path = DEFAULT_BUDGETS_ROOT,
) -> MeasuredValue | None:
    """The training peak recorded in `<root>/<region>/batch=<batch_size>.json`. `None`
    (not a guess) when `batch_size` is unknown or no budget file exists for this cell --
    a region/batch pair that was never run has no measured training peak to print.

    Raises `CardError` if the file's own `source` is not `"report_peak"` -- this
    function only ever hands back a number this project's own peak-VRAM reporter
    produced, never a config-time estimate that happens to live at the same path.
    """
    if batch_size is None:
        return None
    path = root / region / f"batch={batch_size}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    source = data.get("source")
    if source != "report_peak":
        raise CardError(
            f"{path}: training-peak budget's source is {source!r}, not 'report_peak' "
            "-- refusing to label an unmeasured number as MEASURED"
        )
    label_bits = [f"training peak, batch {batch_size}"]
    if max_len is not None:
        label_bits.append(f"max_len {max_len}")
    cell_id = data.get("cell_id")
    if cell_id:
        label_bits.append(f"cell {cell_id}")
    return MeasuredValue(value=float(data["mib"]), label=", ".join(label_bits), unit="MiB")


def build_size_report(
    *,
    region: str,
    train_receipt: dict[str, Any],
    quant_receipt: dict[str, Any] | None = None,
    eval_receipt: dict[str, Any] | None = None,
    eval_quantized_receipt: dict[str, Any] | None = None,
    budgets_root: Path = DEFAULT_BUDGETS_ROOT,
) -> SizeReport:
    """Assemble every size/memory fact a card's Sizes section needs, from a train
    receipt and whichever of quant/eval/eval-quantized are supplied.
    """
    batch_size, max_len = _train_batch_and_max_len(train_receipt)
    holdout_pairs = train_receipt.get("config", {}).get("holdout_pairs")
    disk = disk_bytes_per_state(train_receipt, quant_receipt)
    return SizeReport(
        region=region,
        parameters=parameter_count(train_receipt),
        fp32_bytes=disk.get("fp32"),
        quantized_stored_bytes=disk.get("csd-ptq-v1"),
        compression_ratio=compression_ratio(quant_receipt),
        width_histogram=width_histogram(quant_receipt),
        eval_peak_vram=eval_peak_vram(
            eval_receipt, eval_quantized_receipt, max_len=max_len, holdout_pairs=holdout_pairs
        ),
        training_peak=training_peak(region, batch_size, max_len=max_len, root=budgets_root),
    )
