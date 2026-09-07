"""`ComposeReceipt` -- the interconnect's receipt builder, `docs/design/INTERCONNECT-MODULE-
SPEC.md` section 4, Table 7, and the "Receipts" paragraph above it.

WHAT THIS IS
"Every phase writes one receipt through `ComposeReceipt`, which fills Table 7 and writes
through `write_receipt`, so the stamp `metrics_schema: 'csd-metrics/v2'` is applied once"
(spec section 4). `write_receipt` here is `cogsyndelta.regions._receipt.write_receipt` --
the raw-dict writer every training and quantization receipt already routes through, NOT
`cogsyndelta.pipeline.receipt.Receipt` (that dataclass's flat `metrics`/`gates`/`detail`
shape has no room for Table 7's twelve nested field groups, and re-deriving them into
`detail` would make every receipt this module writes opaque to the one reader --
`compare()`, `cogsyndelta.eval.metrics` -- that needs several of those fields BY NAME).
What this module borrows from `pipeline/receipt.py` is only the envelope constants:
`SCHEMA` (`"model-pipeline-receipt/v1"`) and the fact that `"schedule"` is now a member of
`STAGES` (this lane's own addition, see that module).

THE TWO ENVELOPE STAGES
"The envelope is `model-pipeline-receipt/v1` with `stage: 'compose'` for A, W5b and E2 and
`stage: 'schedule'` for B, C and D" (spec section 4). `ComposeReceipt`'s `stage` argument
IS that envelope value -- `"compose"` or `"schedule"` -- not the training-contract phase
letter (A/B/C/W5b/...); the phase itself is not a Table 7 field and has nowhere to live in
this receipt, so a caller that needs to record which phase wrote a given receipt puts it
in `verdicts` or `provenance`-shaped free text under one of the groups below `[spec]`.

TWELVE GROUPS, ONE BUILDER
Table 7 lists twelve field groups: identity, frozen set, budgets, composed metric,
baselines, attention mass, write-back and topology, scheduler, latency, store, verdicts,
and placement and knobs. `identity` is supplied at construction (it is what makes a
receipt comparable at all, so it cannot be optional); the other eleven each get one
`set_*` method taking a single mapping shaped like that group's fields, so a caller
assembling a phase-A receipt calls eleven `set_*` methods once each, in any order, and
`build()` (or `write()`) refuses if any is missing -- "a receipt missing any group fails a
schema test" (spec section 5, Table 9). Field NAMES inside each group's mapping are this
module's own choice where Table 7's prose does not already fix a dotted path (marked
`[spec]` on each `set_*` method below); the top-level KEY each group is written under
(`"frozen_set"`, `"budgets"`, ...) is likewise this module's own naming, snake-cased from
Table 7's own group names, so a reader can find any group by the name Table 7 already
gives it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from cogsyndelta.pipeline.receipt import SCHEMA as PIPELINE_SCHEMA
from cogsyndelta.pipeline.receipt import STAGES
from cogsyndelta.regions._receipt import (
    METRICS_SCHEMA_V2,
    capture_code_revision,
    write_receipt,
)

__all__ = ["ComposeReceipt"]

#: The two envelope `stage` values Table 7's "Receipts" paragraph names. A phase letter
#: (A, W5b, E2, B, C, D) is not one of these -- see the module docstring.
_ENVELOPE_STAGES = frozenset({"compose", "schedule"})

#: `identity` keys `ComposeReceipt.__init__` requires. `metrics_schema` and
#: `code_revision.git_sha` -- two of Table 7's ten identity fields -- are deliberately
#: absent: `write_receipt` stamps both itself, unconditionally, on every write (see its
#: docstring), so accepting them here would let a caller pass a value `write_receipt`
#: silently overwrites, which is worse than not asking for it.
_IDENTITY_KEYS = (
    "corpus_fingerprint",
    "fingerprint_scheme",
    "battery_id",
    "k",
    "pooling",
    "checkpoint_sha256",
    "region",
    "seed",
    "split_sha256",
)

#: One entry per Table 7 group this builder owns beyond `identity`: the top-level receipt
#: key it is written under, and the sub-keys `build()` requires present in whatever
#: mapping the matching `set_*` method received. Values are used only to give a caller a
#: specific, named error instead of a `KeyError` deep inside `build()`.
_GROUP_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "frozen_set": ("regions", "participants", "R"),
    "budgets": (
        "B_read",
        "B_kv",
        "eta",
        "collapse_floor",
        "token_budget",
        "ctx_min",
        "ctx_max",
        "b",
        "ctx",
    ),
    "composed_metric": ("recall_at_1", "null_recall", "null_fpr", "pair_deltas"),
    "baselines": (),  # a mapping of baseline-name -> {"value": ..., "source": "graded"}
    "attention_mass": (
        "mean_per_region",
        "histogram_per_region",
        "collapsed_in_phase_A",
        "a_store",
    ),
    "write_back_topology": ("enabled", "own_bin_delta", "topology"),
    "scheduler": (
        "rho",
        "flops_ratio",
        "sparse_rel_delta",
        "phase_d_vs_c",
        "s_stats",
        "mean_iters",
        "halt_at_histogram",
        "budget_as_tag_gap",
    ),
    "latency": ("wall_ms_measured",),
    "store": ("b_store", "capacity_bytes", "occupancy_bytes", "active"),
    "verdicts": ("integration", "scheduling", "trigger_sensitivity"),
    "placement_knobs": ("placement", "knobs"),
}

#: Group insertion order, fixed so `build()` produces the same key order on every call
#: given the same inputs (Table 9's "construction ... deterministic" requirement, applied
#: to receipt assembly rather than a tensor forward pass).
_GROUP_ORDER = tuple(_GROUP_REQUIRED_KEYS)


class ComposeReceipt:
    """Builds one Table 7 receipt for a DEC-50 training-contract phase.

    Constructor arguments are exactly `(stage, identity)` (spec section 2.1 Table 2).
    Every other Table 7 group is attached with its own `set_*` method before `build()` or
    `write()` is called; see the module docstring for why the eleven groups are not also
    constructor arguments.
    """

    def __init__(self, stage: str, identity: Mapping[str, Any]) -> None:
        """Start a receipt for envelope `stage`, carrying the given identity fields.

        Args:
            stage: `"compose"` (phases A, W5b, E2) or `"schedule"` (phases B, C, D) --
                the envelope's own `stage` field (spec section 4, "Receipts" paragraph).
            identity: Table 7's identity group, minus `metrics_schema` and
                `code_revision.git_sha` (`write_receipt` stamps both): `corpus_fingerprint`,
                `fingerprint_scheme`, `battery_id`, `k`, `pooling`, `checkpoint_sha256`,
                `region`, `seed`, `split_sha256`.

        Raises:
            ValueError: `stage` is not `"compose"` or `"schedule"`, `"schedule"` is
                requested but missing from `pipeline.receipt.STAGES` (this lane's own
                addition -- see that module), or `identity` is missing a required key.
        """
        if stage not in _ENVELOPE_STAGES:
            raise ValueError(
                f"ComposeReceipt: stage must be one of {sorted(_ENVELOPE_STAGES)}, got {stage!r}"
            )
        if stage not in STAGES:
            raise ValueError(
                f"ComposeReceipt: stage {stage!r} is not a member of "
                f"cogsyndelta.pipeline.receipt.STAGES ({STAGES!r})"
            )
        missing = [k for k in _IDENTITY_KEYS if k not in identity]
        if missing:
            raise ValueError(f"ComposeReceipt: identity is missing required keys: {missing}")
        self.stage = stage
        self._identity = dict(identity)
        self._groups: dict[str, dict[str, Any]] = {}

    def _set(self, group: str, payload: Mapping[str, Any]) -> ComposeReceipt:
        """Shared plumbing for every `set_*` method: validate then store one group."""
        required = _GROUP_REQUIRED_KEYS[group]
        missing = [k for k in required if k not in payload]
        if missing:
            raise ValueError(f"ComposeReceipt.set_{group}: payload is missing keys: {missing}")
        self._groups[group] = dict(payload)
        return self

    def set_frozen_set(self, payload: Mapping[str, Any]) -> ComposeReceipt:
        """Table 7 frozen-set group: `regions[]` (each `{name, checkpoint_sha256,
        receipt_path, status}`), `participants`, `R`. Phase A refuses to start on a
        mismatch (G33, `gates.check_frozen_set_identity`) -- checked by the caller before
        this receipt is built, not by this method.
        """
        return self._set("frozen_set", payload)

    def set_budgets(self, payload: Mapping[str, Any]) -> ComposeReceipt:
        """Table 7 budgets group: `B_read`, `B_kv`, `eta`, `collapse_floor`
        (`{expression: "eta/R", R, value}`), per-region `token_budget`, `ctx_min`,
        `ctx_max`, and `b`/`ctx` as actually run.
        """
        return self._set("budgets", payload)

    def set_composed_metric(self, payload: Mapping[str, Any]) -> ComposeReceipt:
        """Table 7 composed-metric group: `recall_at_1` (overall and per bin),
        `null_recall` (general bin), `null_fpr` (per other bin), and `pair_deltas`
        (per-pair `Δ_A`, `Δ_B`, `I` with block-bootstrap CIs at W6) `[spec]` (key names
        are this module's own; Table 7 states the fields as prose, not a schema).
        """
        return self._set("composed_metric", payload)

    def set_baselines(self, payload: Mapping[str, Any]) -> ComposeReceipt:
        """Table 7 baselines group: B0, B0d, B0u, B1, B2, B2t, B3, each
        `{"value": ..., "source": "graded"}` (TAX:1815-1823). Empty is a permitted
        payload shape (no required keys) because the seven names, not this method, are
        the contract; a caller writing zero baselines still explicitly calls this with
        `{}` so `build()` can tell "no baselines" from "baselines group never attached".
        """
        return self._set("baselines", payload)

    def set_attention_mass(self, payload: Mapping[str, Any]) -> ComposeReceipt:
        """Table 7 attention-mass group: `a` mean per active iteration and region
        (`mean_per_region`), the per-region histogram, `collapsed_in_phase_A: []`, and
        `a_store`.
        """
        return self._set("attention_mass", payload)

    def set_write_back_topology(self, payload: Mapping[str, Any]) -> ComposeReceipt:
        """Table 7 write-back-and-topology group: `write_back.enabled` (`enabled`), the
        per-region own-bin delta (`own_bin_delta`), `topology {agreement, status}`, and
        `edges` only when demonstrated (omit the key entirely rather than pass `None` --
        `build()` writes it only when present in `payload`).
        """
        return self._set("write_back_topology", payload)

    def set_scheduler(self, payload: Mapping[str, Any]) -> ComposeReceipt:
        """Table 7 scheduler group: `rho`, `flops_ratio`, `sparse_rel_delta`,
        `phase_d_vs_c`, the S1-S5 statistics (`s_stats`, reported, never gated:
        TAX:2019-2023), `mean_iters`, the `halt_at` histogram (`halt_at_histogram`), and
        the budget-as-tag gap (`budget_as_tag_gap`).
        """
        return self._set("scheduler", payload)

    def set_latency(self, payload: Mapping[str, Any]) -> ComposeReceipt:
        """Table 7 latency group: `wall_ms_measured`; `first_token_ms_measured` and
        `speech_frame_miss_fraction` (Table 8a: both `null` at v1 -- pass `None`
        explicitly for each, do not omit).
        """
        return self._set("latency", payload)

    def set_store(self, payload: Mapping[str, Any]) -> ComposeReceipt:
        """Table 7 store group: `b_store`, `capacity_bytes` per card, `occupancy_bytes`,
        and which of `"residual"` or `"floor"` is active.
        """
        return self._set("store", payload)

    def set_verdicts(self, payload: Mapping[str, Any]) -> ComposeReceipt:
        """Table 7 verdicts group: `integration`, `scheduling`, `trigger_sensitivity` --
        three separate strings (TAX:2025-2028), never blended into one field.
        """
        return self._set("verdicts", payload)

    def set_placement_knobs(self, payload: Mapping[str, Any]) -> ComposeReceipt:
        """Table 7 placement-and-knobs group: `placement{}` from W7p and `knobs{}` from
        W7k.

        Both values must be MAPPINGS, and `None` is refused by name, because `None` is the
        shape this group actually shipped in: `{"placement": null, "knobs": null}` on all
        thirty phase-A receipts written before 2026-09-07, among them eleven that
        disagreed about `checkpoint_sha256` at one seed and one command line because the
        knob that decided the weights -- the intra-op thread count -- had nowhere here to
        be recorded. Table 7 writes both as `{}`, so a caller with nothing to say in this
        group still says it with an empty mapping, the way `set_baselines` does; that
        keeps "measured nothing" distinguishable from "recorded nothing", which a `None`
        does not.

        Args:
            payload: Must carry `placement` and `knobs`, each a mapping.

        Returns:
            `self`, for chaining.

        Raises:
            ValueError: `placement` or `knobs` is absent or is not a mapping.
        """
        for key in ("placement", "knobs"):
            value = payload.get(key)
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"ComposeReceipt.set_placement_knobs: {key!r} must be a mapping, got "
                    f"{value!r}. Table 7 writes this group as `{key} {{}}`; None is the "
                    "shape that let a run's thread pin go unrecorded."
                )
        return self._set("placement_knobs", payload)

    def build(self) -> dict[str, Any]:
        """Assemble the full receipt dict, refusing if any Table 7 group is missing.

        Returns:
            A plain `dict`, ready for `cogsyndelta.regions._receipt.write_receipt` (or for
                `write()` below, which calls it). `code_revision` and the final
                `metrics_schema` stamp are NOT set here -- `write_receipt` sets both,
                unconditionally, on every write; `build()` sets `metrics_schema` to the
                same value only so a caller inspecting the dict before writing sees a
                complete Table 7 shape rather than a hole `write()` will fill in later.

        Raises:
            ValueError: One or more of the eleven `set_*` groups was never called.
        """
        missing_groups = [g for g in _GROUP_ORDER if g not in self._groups]
        if missing_groups:
            raise ValueError(
                f"ComposeReceipt.build: missing required Table 7 groups: {missing_groups}"
            )
        receipt: dict[str, Any] = {
            "schema": PIPELINE_SCHEMA,
            "stage": self.stage,
            "metrics_schema": METRICS_SCHEMA_V2,
            "battery_id": self._identity["battery_id"],
            "k": self._identity["k"],
            "pooling": self._identity["pooling"],
            "region": self._identity["region"],
            "seed": self._identity["seed"],
            "corpus": {
                "fingerprint": self._identity["corpus_fingerprint"],
                "fingerprint_scheme": self._identity["fingerprint_scheme"],
            },
            "artifacts": {"checkpoint_sha256": self._identity["checkpoint_sha256"]},
            "split": {"sha256": self._identity["split_sha256"]},
        }
        for group in _GROUP_ORDER:
            receipt[group] = self._groups[group]
        return receipt

    def write(
        self,
        out_dir: Path,
        filename: str,
        *,
        repo_root: Path | None = None,
        capture: Callable[[Path | None], dict[str, Any] | None] = capture_code_revision,
    ) -> Path:
        """`build()` then write through `write_receipt`, stamping `code_revision` and
        `metrics_schema: "csd-metrics/v2"` exactly once (spec section 4, "Receipts").

        Args:
            out_dir: Directory to write into; created if missing.
            filename: Exact filename inside `out_dir` -- this method does not derive one
                the way `pipeline.receipt.Receipt.write` does, because a phase name
                (A/W5b/E2/B/C/D) is not itself a Table 7 field for this method to read
                back off the receipt.
            repo_root: Forwarded to `capture` (or the real `capture_code_revision`) --
                see `write_receipt`.
            capture: Injection seam for the test that proves `write_receipt`'s own
                refusal fires through this method too. Production callers never pass it.

        Returns:
            The path written.

        Raises:
            ValueError: `build()` raised (a group is missing).
            RuntimeError: `write_receipt` raised (the code-revision capture returned
                nothing).
        """
        return write_receipt(self.build(), out_dir, filename, repo_root=repo_root, capture=capture)
