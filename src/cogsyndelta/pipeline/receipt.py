"""A model-agnostic envelope for pipeline receipts.

WHY AN ENVELOPE RATHER THAN A DASHBOARD SCHEMA
The console is meant to serve any model pipeline, with CSD as the first target and other
architectures -- a ternary model, for one -- following. What makes that possible is not the
UI. It is that every stage of every pipeline writes the same outer shape, so a reader can
list runs, compare a metric across them and show which gates held without knowing anything
about the architecture that produced them.

Today's receipts do not have that. `pretrain` stamps schema `csd-pretrain-receipt/v1`;
the visual and quantization receipts stamp nothing at all. A reader over those three is
not generic, it is three special cases wearing a trenchcoat, and every new architecture
adds a fourth.

THE SPLIT THAT MATTERS
`metrics` is a flat dict of name to number, and nothing else. A reader can plot any of it
without interpreting it. Everything architecture-specific -- a collapse ratio, a width
histogram, a contamination report -- goes in `detail`, which the reader passes through
untouched. That boundary is what keeps a ternary model's receipt readable by a console
that has never heard of ternary quantization.

`gates` is name to bool, and a run passes only if every gate does. Encoding the verdict
rather than recomputing it in the reader means the producer -- which knows what its metric
means -- decides, and two pipelines cannot disagree about what "passed" means.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cogsyndelta.regions._receipt import METRICS_SCHEMA_V2, capture_code_revision

SCHEMA = "model-pipeline-receipt/v1"

#: Re-exported for callers that only import this module -- the envelope schema above
#: (`SCHEMA`) is unchanged by the metrics-naming unification; this is the SEPARATE
#: stamp naming which metric-name/battery/pooling table applies (g7-latent-eval-metrics
#: §3.3). See `cogsyndelta.regions._receipt.METRICS_SCHEMA_V2` for the full rationale --
#: defined there, once, so `regions._receipt.write_receipt` (which stamps train and
#: quant receipts) and this envelope (which stamps eval/eval-quantized receipts) never
#: disagree on the string.
METRICS_SCHEMA = METRICS_SCHEMA_V2

#: A receipt read back with no `metrics_schema` at all predates the stamp -- every
#: receipt on disk before this change. Distinguishing that from "explicitly v2" is
#: what lets a `compare()` (g7 §3.3's refuse predicate) tell a genuinely old-shaped
#: number apart from one this envelope has already renamed on read (see
#: `_QUANT_METRIC_ALIASES_V1` below): the ALIASED name is v2-shaped, but the receipt
#: that produced it was not stamped v2, so a caller diffing it against a real v2
#: receipt still needs to know that.
METRICS_SCHEMA_V1_LEGACY = "csd-metrics/v1"

#: v1 -> v2 aliases for the quant-receipt fields this lane owns (g7 §3.1: `quant.plan_recall@1`
#: WAS `quantized_metric`, `quant.compression_ratio` WAS `compression_ratio`,
#: `quant.drop_recall@1` WAS the bare `drop`). A quant receipt this project writes NOW
#: (`scripts/csd-quantize.py`) uses the v2 names directly; this table is for a
#: v1-shaped quant receipt already on disk (or a v1-shaped raw dict a caller still
#: hands this module) so it can be normalised to v2 field names before anything reads
#: it by name. `scripts/csd-publish-checkpoint.py` -- a consumer, not a producer --
#: imports this to normalise a loaded quant receipt before looking up
#: `METRIC_METHODOLOGY`. Scoped to the fields this lane's files produce; the project
#: does not yet have one shared, all-battery `METRIC_ALIASES_V1` covering the
#: train-receipt gate/battery renames another lane owns -- see
#: `scripts/csd-publish-checkpoint.py`'s own note where it reads those.
QUANT_METRIC_ALIASES_V1: dict[str, str] = {
    "quantized_metric": "quant.plan_recall@1",
    "compression_ratio": "quant.compression_ratio",
    "drop": "quant.drop_recall@1",
}

# Stages a pipeline may report. Open by convention rather than enforced, because a new
# architecture may have a stage nobody anticipated; the reader groups by whatever it finds.
STAGES = ("pretrain", "finetune", "eval", "quantize", "compose", "publish")


@dataclass
class Producer:
    """Who made this receipt. Lets one console serve many projects."""

    project: str
    """e.g. "cogsyndelta", "tritter"."""
    component: str
    """The submodel or region; use the project name for a whole-model run."""
    architecture: str = ""
    """e.g. "dense-transformer", "ternary", "i-jepa". Free text, for grouping only."""


@dataclass
class Receipt:
    """One stage of one pipeline run."""

    producer: Producer
    stage: str
    metrics: dict[str, float] = field(default_factory=dict)
    """Flat name -> number. Plottable without interpretation."""
    baseline: dict[str, float] = field(default_factory=dict)
    """What `metrics` should be read against -- an untrained model, an fp32 reference."""
    lexical_baseline: dict[str, Any] = field(default_factory=dict)
    """Bag-of-words ceiling on the same closed holdout (`lexical_baseline.{tfidf,bm25}.*`).

    Empty means not measured (visual eval, or a receipt written before g49). When
    present it MUST carry `split_sha256` equal to this receipt's `split.sha256`
    (`verify_lexical_baseline_split`, fail closed, G26). `write()` drops an empty
    dict so visual receipts do not grow a vacuous key.
    """
    gates: dict[str, bool] = field(default_factory=dict)
    """The producer's verdict. A run passes only if every gate is true."""
    artifacts: dict[str, Any] = field(default_factory=dict)
    """What this stage produced, and the hashes that bind the receipt to those bytes.

    `Any` rather than `str` because two entries are deliberately NESTED records, not
    scalars: `source_training_receipt` and `source_quant_receipt` are
    `{"path": ..., "sha256": ...}` pairs, so an eval receipt names the predecessor
    receipt it read AND the content hash of that receipt, not merely a mutable path.
    The annotation said `dict[str, str]` while both eval paths had been writing those
    records all along -- a mismatch nothing caught, because `scripts/` was outside the
    typechecked surface (L1). Widening the annotation is the honest fix: flattening the
    records would change a receipt shape that is already on disk, and dropping them
    would lose the binding.
    """
    provenance: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)
    """Architecture-specific payload. Readers pass it through, they do not parse it."""
    started_utc: str = ""
    seconds: float = 0.0
    device: str = ""
    schema: str = SCHEMA
    metrics_schema: str = METRICS_SCHEMA
    """Which metric-name/battery/pooling table `metrics` was written under -- SEPARATE
    from `schema` above (the envelope shape) and unrelated to it: `schema` can stay
    `model-pipeline-receipt/v1` forever while `metrics_schema` moves from v1 to v2,
    because the rename is about field NAMES inside `metrics`/`gates`, not the envelope
    around them. Defaults to the current table (`METRICS_SCHEMA`) for a receipt this
    class constructs fresh; `adapt()` sets it to `METRICS_SCHEMA_V1_LEGACY` instead when
    reading a receipt that predates this field, so a reader can still refuse to compare
    two numbers whose schema disagrees even after `adapt()` has renamed the legacy one's
    keys (see `_QUANT_METRIC_ALIASES_V1`)."""
    kind: str = ""
    """A shape predicate finer than `stage`: e.g. `stage="eval"` covers both an fp32 pass
    (`kind="eval"`) and a quantized-artifact pass (`kind="eval-quantized"`) -- same stage,
    different provenance, different filename, and a reader (or a matrix harness selecting
    a receipt by kind, never by "latest under this glob") must be able to tell them apart
    without inspecting `provenance` or `artifacts`. Empty string means "not set" (a
    receipt written before this field existed, or a stage with no finer distinction to
    make); `write()` falls back to `stage` for the filename in that case, so this is
    additive and every existing caller is unaffected.

    The matrix harness does NOT classify on this string. `model_matrix.receipts.kind_of`
    reads `provenance.eval_target == "quantized"` and reports `eval-quant`, so this
    project's `"eval-quantized"` and the harness's `receipt_kind: eval-quant` describe the
    same receipt and neither has to be renamed to match the other (H3/L2). What DOES have
    to agree is the FILENAME: `program/matrix/csd-matrix.yaml`'s `test-quant.receipt` glob
    matches `cogsyndelta-{region}-eval-quantized-*.json`, which is what `write()` below
    produces from this field -- so renaming `kind` silently breaks receipt selection in
    the matrix even though classification would still work.
    """
    code_revision: dict[str, Any] = field(default_factory=dict)
    """What code produced these numbers: `git_sha`, `dirty`, `branch`, `describe`.

    Stamped by `write()` from `cogsyndelta.regions._receipt.capture_code_revision` -- the
    SAME helper every training and quantization receipt already routes through -- so one
    reader can compare an eval receipt's revision against a training receipt's without
    knowing which producer wrote which. Empty only on a receipt read back from disk that
    predates this field (`adapt` passes through whatever it finds, including nothing); a
    receipt this class WRITES always has it, because `write()` refuses otherwise.
    """

    @property
    def passed(self) -> bool:
        """True when the producer declared gates and all of them hold.

        A run with no gates is NOT a pass. Something that measured nothing has not
        demonstrated anything, and showing it as green is how an empty pipeline looks
        healthy on a dashboard.
        """
        return bool(self.gates) and all(self.gates.values())

    def write(
        self,
        out_dir: Path,
        *,
        capture: Callable[[Path | None], dict[str, Any] | None] = capture_code_revision,
    ) -> Path:
        """Write to ``{out_dir}/{project}-{component}-{kind or stage}-{timestamp}.json``.

        Uses `kind` when set (so `eval` and `eval-quantized` land under distinguishable
        filenames and can never glob-collide) and falls back to `stage` otherwise --
        every receipt written before `kind` existed named the file this same way.

        STAMPS `code_revision` FIRST, AND REFUSES TO WRITE WITHOUT IT. Training and
        quantization receipts have gone through `regions/_receipt.write_receipt` -- which
        raises rather than let a receipt reach disk with no provenance block -- since that
        helper existed; eval receipts, written through this method, carried none at all.
        The consequence was not cosmetic: the matrix harness's G5b gate is
        "`code_revision.git_sha` equals the run's code sha and `dirty` is false", and it
        could not fire on the one receipt kind (`eval-quantized`) whose entire purpose is
        to prove a published artifact was scored by known code. A metric with no
        attributable code behind it is a number, not evidence.

        The refusal is fail-closed for the same reason `write_receipt`'s is: the real
        `capture_code_revision` never returns falsy (it degrades to an honest "unknown"
        block instead), so a falsy return means the capture MECHANISM is broken, and a
        receipt silently missing its provenance is worse than one that fails loudly --
        nothing downstream checks for the gap, so the first sign of it would be an
        operator staring at a green run they can no longer place against a commit.

        Args:
            out_dir: directory to write into; created if it does not exist. The filename
                inside it is derived, never passed -- see the format above.
            capture: injection seam for the test that proves the refusal fires. Production
                callers never pass it.
        """
        revision = capture(None)
        if not revision:
            raise RuntimeError(
                "Receipt.write: code_revision capture returned nothing -- refusing to "
                f"write a {self.kind or self.stage!r} receipt with no code_revision block"
            )
        self.code_revision = dict(revision)
        # Same "always overwrite" treatment as code_revision above -- a Receipt built
        # from `adapt()` on an old receipt and re-written (not a production path today,
        # but the invariant should hold regardless) must not silently re-persist a
        # stale metrics_schema.
        self.metrics_schema = METRICS_SCHEMA

        payload = asdict(self)
        if payload.get("lexical_baseline"):
            from cogsyndelta.eval.lexical import verify_lexical_baseline_split

            verify_lexical_baseline_split(payload)
        else:
            payload.pop("lexical_baseline", None)

        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        p = self.producer
        segment = self.kind or self.stage
        path = out_dir / f"{p.project}-{p.component}-{segment}-{stamp}.json"
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        return path


def _num(value: Any) -> float | None:
    """Coerce to float, or None when it is not a plain number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def adapt(raw: dict[str, Any], path: Path) -> Receipt | None:
    """Read any receipt this project has ever written into the common envelope.

    Existing receipts predate the envelope and three shapes are already on disk. Rewriting
    them would discard the measurements they carry, so they are adapted on read instead.
    New producers should construct :class:`Receipt` directly.
    """
    if raw.get("schema") == SCHEMA:
        prod = raw.get("producer", {})
        return Receipt(
            producer=Producer(
                project=prod.get("project", "unknown"),
                component=prod.get("component", "unknown"),
                architecture=prod.get("architecture", ""),
            ),
            stage=raw.get("stage", "unknown"),
            metrics=raw.get("metrics", {}),
            baseline=raw.get("baseline", {}),
            lexical_baseline=raw.get("lexical_baseline", {}) or {},
            gates=raw.get("gates", {}),
            artifacts=raw.get("artifacts", {}),
            provenance=raw.get("provenance", {}),
            detail=raw.get("detail", {}),
            started_utc=raw.get("started_utc", ""),
            seconds=float(raw.get("seconds", 0.0)),
            device=raw.get("device", ""),
            # A producer that wrote `kind` directly (csd-benchmark.py's fp32/quantized
            # eval receipts) is trusted; one that did not falls back to `stage`, matching
            # `write()`'s own filename fallback.
            kind=raw.get("kind") or raw.get("stage", "unknown"),
            code_revision=raw.get("code_revision", {}),
            # Absent means this receipt predates the stamp -- v1's mixed names, not v2's
            # table (g7 §3.3). Never default a missing stamp to the CURRENT table: that
            # would let a genuinely-old receipt's numbers silently pass a same-schema
            # compare() against a real v2 receipt.
            metrics_schema=raw.get("metrics_schema") or METRICS_SCHEMA_V1_LEGACY,
        )

    component = raw.get("region")
    if not component:
        return None

    # Quantization receipts: no schema key, identified by their own fields -- EITHER
    # the v1 raw names (`quantized_metric`, `compression_ratio`) or the v2 ones
    # (`quant.plan_recall@1`, `quant.compression_ratio`; `scripts/csd-quantize.py`
    # writes v2 names now). The envelope's OWN internal metric keys below (`metric`,
    # `compression_ratio`, `drop`, `stored_mb`) stay generic and unprefixed regardless
    # of which raw shape was read -- they are this architecture-agnostic reader's own
    # naming, not a copy of CSD's receipt field names (a ternary model's quant receipt
    # would have neither `quantized_metric` nor `quant.plan_recall@1`, and still needs
    # to land in the same three generic buckets a dashboard can plot without knowing
    # what produced them).
    is_v1_quant = "compression_ratio" in raw and "quantized_metric" in raw
    is_v2_quant = "quant.compression_ratio" in raw and (
        "quant.plan_recall@1" in raw or "quant.plan_probe_top1" in raw
    )
    if is_v1_quant or is_v2_quant:
        metric = raw.get(
            "quant.plan_probe_top1",
            raw.get("quant.plan_recall@1", raw.get("quantized_metric")),
        )
        ratio = raw.get("quant.compression_ratio", raw.get("compression_ratio"))
        drop = raw.get(
            "quant.drop_probe_top1",
            raw.get("quant.drop_recall@1", raw.get("drop")),
        )
        return Receipt(
            producer=Producer("cogsyndelta", component, "dense-transformer"),
            stage="quantize",
            kind=raw.get("kind") or "quant",
            metrics={
                "metric": _num(metric) or 0.0,
                "compression_ratio": _num(ratio) or 0.0,
                "stored_mb": (_num(raw.get("stored_bytes")) or 0.0) / 1e6,
                "drop": _num(drop) or 0.0,
            },
            baseline={"metric": _num(raw.get("fp32_metric_recomputed")) or 0.0},
            gates={"within_budget": bool(raw.get("within_budget"))},
            artifacts={"checkpoint": str(raw.get("checkpoint", ""))},
            provenance={"corpus_fingerprint": raw.get("corpus_fingerprint", "")},
            detail={
                "width_histogram": raw.get("width_histogram", {}),
                "promotions": raw.get("promotions", []),
            },
            started_utc=raw.get("recorded_utc", ""),
            device=raw.get("device", ""),
            code_revision=raw.get("code_revision", {}),
            metrics_schema=raw.get("metrics_schema")
            or (METRICS_SCHEMA if is_v2_quant else METRICS_SCHEMA_V1_LEGACY),
        )

    # Pretrain receipts, text and visual. Both carry held_out + untrained_baseline; the
    # metric NAMES differ (recall@1 vs top1) and that is exactly why metrics is a free
    # dict rather than a fixed set of columns.
    held = raw.get("held_out") or {}
    base = raw.get("untrained_baseline") or {}
    if held:
        visual = "collapse_ratio" in raw
        detail: dict[str, Any] = {"history": raw.get("history", [])}
        if visual:
            detail["collapse_ratio"] = raw.get("collapse_ratio")
            detail["transfer"] = raw.get("transfer")
            detail["untrained_transfer"] = raw.get("untrained_transfer")
        else:
            detail["contamination"] = raw.get("contamination")
            detail["corpus"] = raw.get("corpus")
        return Receipt(
            producer=Producer(
                "cogsyndelta", component, "i-jepa" if visual else "dense-transformer"
            ),
            stage="pretrain",
            kind=raw.get("kind") or "train",
            metrics={k: v for k, v in ((k, _num(v)) for k, v in held.items()) if v is not None},
            baseline={k: v for k, v in ((k, _num(v)) for k, v in base.items()) if v is not None},
            # `beats_untrained_train` is the g7 §3.2 rename of this same gate/context
            # field (two predicates shared one name across train vs. eval receipts;
            # train's own is `_beats_untrained_gate`, MM §1). That rename is written by
            # `regions/pretrain.py`, outside this lane's files -- read defensively, the
            # new key first, so this reader keeps working whichever name a given
            # receipt on disk carries, without importing that module.
            gates=dict(raw.get("beats_untrained_train") or raw.get("beats_untrained") or {}),
            artifacts={
                "checkpoint": str(raw.get("checkpoint", "")),
                # Only present on receipts written after R9 (checkpoint fingerprinting);
                # older receipts leave this "" rather than fabricate a hash nobody
                # computed. A reader comparing artifacts across receipts should treat an
                # empty value as "not recorded", not as "the file is empty".
                "checkpoint_sha256": str(raw.get("checkpoint_sha256", "")),
            },
            provenance={
                "parameters": raw.get("parameters"),
                "config": raw.get("config", {}),
            },
            detail=detail,
            started_utc=raw.get("started_utc") or raw.get("recorded", ""),
            seconds=float(raw.get("seconds") or raw.get("elapsed_s") or 0.0),
            device=raw.get("device", ""),
            code_revision=raw.get("code_revision", {}),
            metrics_schema=raw.get("metrics_schema") or METRICS_SCHEMA_V1_LEGACY,
        )
    return None


def load_all(roots: list[Path]) -> list[Receipt]:
    """Read every receipt under the given roots, newest first.

    A file that cannot be parsed is skipped rather than fatal: a dashboard that dies on
    one malformed receipt is less useful than one that shows the other forty.
    """
    out: list[Receipt] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            try:
                raw = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(raw, dict):
                continue
            rec = adapt(raw, path)
            if rec is not None:
                out.append(rec)
    return sorted(out, key=lambda r: r.started_utc, reverse=True)
