"""`program/matrix/csd-matrix.yaml` must survive the harness's own loader rules.

The matrix config is CSD's half of a contract with a SEPARATE repo (`tzervas/model-matrix`,
see the `tooling-lives-in-its-own-repo` rule), so this repo cannot import the loader that
consumes it -- `model_matrix` is deliberately not a dependency here. What it CAN do is
re-derive, from the YAML alone, the two loader behaviours that silently produced wrong
answers rather than errors:

* axis LABELS are matched to `budgets:` keys by `str(label)`
  (`model_matrix.config.canonical_budget_key`), and
* region overrides are merged over `regions.defaults` with a SHALLOW `dict.update`
  (`model_matrix.config._merge_region`).

Both are re-implemented below in a dozen lines each, with the harness function they
mirror named in a comment. That is the whole point of doing it here: an assertion in this
repo, against this repo's file, that fails the moment either property is violated -- and
that keeps failing even on a host where the harness is not checked out at all.

Every guard in this file is paired with a MUTATION test that feeds the same checker the
broken shape the config used to have, and asserts the checker reports it. A guard nobody
has watched fail is not a guard.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

MATRIX_YAML = Path(__file__).resolve().parents[1] / "program" / "matrix" / "csd-matrix.yaml"


def _raw() -> dict[str, Any]:
    return yaml.safe_load(MATRIX_YAML.read_text())


# --------------------------------------------------------------- harness rules, mirrored


def _merge_region(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Mirror of `model_matrix.config._merge_region`: `dict(defaults)` then `.update()`.

    SHALLOW on purpose -- that is the behaviour being tested against, not an omission.
    """
    merged = dict(defaults)
    merged.update(overrides)
    return merged


def _merged_regions(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    regions = raw["regions"]
    defaults = regions["defaults"]
    return {
        name: _merge_region(defaults, overrides)
        for name, overrides in regions.items()
        if name != "defaults"
    }


def _axis_labels(spec: Any) -> list[Any]:
    """Mirror of `model_matrix.config._axis_options`: a list axis yields its items as
    labels, a mapping axis yields its KEYS as labels (the values are extra template
    vars)."""
    if isinstance(spec, list):
        return list(spec)
    if isinstance(spec, dict):
        return list(spec.keys())
    raise AssertionError(f"axis spec must be a list or a mapping, got {type(spec).__name__}")


def _matches_exclude(axis_values: dict[str, Any], exclude_entry: dict[str, Any]) -> bool:
    """Mirror of `model_matrix.config._matches_exclude` -- compares by `str()`."""
    return all(str(axis_values.get(axis)) == str(value) for axis, value in exclude_entry.items())


def _cells(merged: dict[str, Any]) -> list[dict[str, Any]]:
    """Mirror of `expand_region_cells`'s cross product + exclude filter."""
    axes = merged.get("axes", {})
    combos: list[dict[str, Any]] = [{}]
    for axis, spec in axes.items():
        combos = [{**prefix, axis: label} for prefix in combos for label in _axis_labels(spec)]
    excludes = merged.get("exclude", [])
    return [c for c in combos if not any(_matches_exclude(c, ex) for ex in excludes)]


def _canonical(pairs: dict[str, Any]) -> frozenset[tuple[str, str]]:
    """Mirror of `model_matrix.config.canonical_budget_key`: `str()` on every value."""
    return frozenset((axis, str(value)) for axis, value in pairs.items())


def _budget_index(merged: dict[str, Any]) -> dict[frozenset[tuple[str, str]], dict[str, Any]]:
    index: dict[frozenset[tuple[str, str]], dict[str, Any]] = {}
    for key, entry in merged.get("budgets", {}).items():
        pairs = dict(part.split("=", 1) for part in key.split(","))
        index[_canonical({k.strip(): v.strip() for k, v in pairs.items()})] = entry
    return index


def _unresolved_budget_cells(merged: dict[str, Any]) -> list[dict[str, Any]]:
    """Cells that would fall through to `BudgetEntry(mib=None, source="probe")` because
    NO declared `budgets:` key matches them -- the silent `None` DESIGN.v2 §3.1 forbids.
    """
    budget_axes = merged.get("budget_axes", list(merged.get("axes", {}).keys()))
    index = _budget_index(merged)
    return [
        cell
        for cell in _cells(merged)
        if _canonical({a: v for a, v in cell.items() if a in budget_axes}) not in index
    ]


# --------------------------------------------------------------- C3: quoted axis labels


def test_no_axis_label_is_a_yaml_boolean() -> None:
    """C3. Bare `on`/`off`/`yes`/`no`/`y`/`n`/`true`/`false` are YAML 1.1 BOOLEANS.

    `safe_load` turns such an axis label into `True`/`False`, and the harness then
    stringifies it to `"True"`/`"False"` when matching a `budgets:` key written as
    `terms=on` -- so the budget lookup misses, silently, for every cell on that axis.
    """
    for name, merged in _merged_regions(_raw()).items():
        for axis, spec in merged.get("axes", {}).items():
            for label in _axis_labels(spec):
                assert not isinstance(label, bool), (
                    f"region {name!r} axis {axis!r} has label {label!r} of type bool -- "
                    "an unquoted YAML 1.1 boolean keyword; quote it in csd-matrix.yaml"
                )


def test_every_memory_cell_resolves_a_declared_budget_entry() -> None:
    """All five surviving memory cells must hit a `budgets:` key -- never the loader's
    silent `mib=None, source="probe"` fallback."""
    merged = _merged_regions(_raw())["memory"]
    assert len(_cells(merged)) == 5, "the exclude list should leave exactly five memory cells"
    assert _unresolved_budget_cells(merged) == []


def test_the_measured_memory_cell_carries_its_report_peak_measurement() -> None:
    """The one cell whose peak was really measured must resolve to THAT entry -- the
    C3 symptom was this row existing in the file and never being reached."""
    merged = _merged_regions(_raw())["memory"]
    entry = _budget_index(merged)[_canonical({"batch": 512, "terms": "on", "chunk": 512})]
    assert entry["source"] == "report_peak"
    assert entry["mib"] == 11700


def test_every_other_memory_budget_entry_is_explicit_rather_than_absent() -> None:
    """DESIGN.v2 §3.1: an unmeasured cell declares `source: probe` EXPLICITLY. A cell
    with no entry at all reaches the same runtime state by accident, which is the
    failure mode C3 hid."""
    merged = _merged_regions(_raw())["memory"]
    index = _budget_index(merged)
    for cell in _cells(merged):
        key = _canonical({a: v for a, v in cell.items() if a in merged["budget_axes"]})
        entry = index[key]
        assert entry["source"] in ("report_peak", "probe")
        if entry["source"] == "probe":
            assert entry["mib"] is None, "a probe entry must not carry a number to trust"


def test_the_budget_guard_fires_when_the_terms_labels_are_yaml_booleans() -> None:
    """MUTATION. Re-break C3 in memory: replace the quoted `"off"`/`"on"` axis labels
    with the booleans `safe_load` produced before they were quoted, and assert BOTH
    guards above report it. Without this, `test_no_axis_label_is_a_yaml_boolean` and
    `test_every_memory_cell_resolves_a_declared_budget_entry` are untested assertions.
    """
    raw = copy.deepcopy(_raw())
    terms = raw["regions"]["memory"]["axes"]["terms"]
    raw["regions"]["memory"]["axes"]["terms"] = {
        False: terms["off"],
        True: terms["on"],
    }
    merged = _merged_regions(raw)["memory"]

    labels = _axis_labels(merged["axes"]["terms"])
    assert any(isinstance(label, bool) for label in labels)
    # Five cells still expand (the `terms: "off"` exclude also stops matching, but the
    # `batch=1280, chunk=2048` one still fires) -- and every one of them misses.
    unresolved = _unresolved_budget_cells(merged)
    assert len(unresolved) == len(_cells(merged)) > 0, (
        "with boolean labels EVERY memory cell must miss its budget key -- that is the "
        "defect C3 named; if this passes, the checker is not checking"
    )


# --------------------------------------------------- C4: shallow merge drops commands

# The stages every region's pipeline actually runs a COMMAND for. `collect`, `publish`'s
# successors `verify`, and `collect` itself are harness-internal (`gpu: false`, no
# `commands:` entry anywhere in this config), so they are deliberately absent.
COMMANDED_STAGES = frozenset({"train", "test", "quantize", "test-quant", "publish"})


def _missing_commands(merged: dict[str, Any]) -> set[str]:
    return set(COMMANDED_STAGES) - set(merged.get("commands", {}))


def test_every_region_carries_a_command_for_every_commanded_stage() -> None:
    """C4. `_merge_region` is `dict(defaults); merged.update(overrides)` -- SHALLOW. A
    region that declares `commands: {train: ...}` therefore REPLACES the defaults map
    rather than adding to it, and the harness renders an absent stage as the empty
    string (`command_templates.get(stage_name, "")`): an empty argv, not an error."""
    for name, merged in _merged_regions(_raw()).items():
        assert _missing_commands(merged) == set(), (
            f"region {name!r} has no command template for {sorted(_missing_commands(merged))} "
            "-- a region that overrides `commands:` must repeat the FULL map, because "
            "the harness merge is shallow"
        )


def test_inherited_command_templates_are_byte_identical_to_the_defaults() -> None:
    """A region repeating the map must repeat it VERBATIM for the stages it does not
    genuinely change. Paraphrasing is how the two copies drift into two different
    pipelines that look like one."""
    raw = _raw()
    defaults = raw["regions"]["defaults"]["commands"]
    for name, overrides in raw["regions"].items():
        if name == "defaults" or "commands" not in overrides:
            continue
        for stage, template in overrides["commands"].items():
            if stage == "train":
                continue  # memory trains through its own module entry point, by design
            assert template == defaults[stage], (
                f"region {name!r} command {stage!r} has drifted from regions.defaults"
            )


def test_the_command_guard_fires_when_a_region_overrides_only_train() -> None:
    """MUTATION. Restore the pre-fix `memory.commands` (train only) and assert the
    guard reports the four templates the shallow merge deletes."""
    raw = copy.deepcopy(_raw())
    train_only = {"train": raw["regions"]["memory"]["commands"]["train"]}
    raw["regions"]["memory"]["commands"] = train_only
    merged = _merged_regions(raw)["memory"]
    assert _missing_commands(merged) == {"test", "quantize", "test-quant", "publish"}
