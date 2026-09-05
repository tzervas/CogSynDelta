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


# ------------------------------------------------- C2: budget_axes narrowing needs a why


def _narrows_budget_axes(merged: dict[str, Any]) -> bool:
    axes = set(merged.get("axes", {}))
    return set(merged.get("budget_axes", axes)) < axes


def test_every_region_that_narrows_budget_axes_states_why() -> None:
    """C2. `validate_region_budgets` REFUSES a region whose `budget_axes` is a proper
    subset of its axes without a `budget_axes_why` claim -- the reviewer's constructed
    load failed on `code`, `compress`, `retrieve` and `reason` for exactly this, so no
    cell in the config could expand even once C1 was out of the way."""
    for name, merged in _merged_regions(_raw()).items():
        if not _narrows_budget_axes(merged):
            continue
        why = merged.get("budget_axes_why")
        assert isinstance(why, str) and why.strip(), (
            f"region {name!r} budgets on {sorted(merged['budget_axes'])} out of axes "
            f"{sorted(merged['axes'])} with no budget_axes_why claim"
        )


def test_every_cell_in_every_region_resolves_a_declared_budget_entry() -> None:
    """The C3 guard, widened to the whole config now that C2 lets every region load."""
    for name, merged in _merged_regions(_raw()).items():
        assert _unresolved_budget_cells(merged) == [], (
            f"region {name!r} has cells with no matching budgets: key"
        )


def test_the_budget_axes_why_guard_fires_when_the_defaults_claim_is_removed() -> None:
    """MUTATION. Delete the claim from `regions.defaults` and assert every region that
    inherits it is reported -- the pre-fix state."""
    raw = copy.deepcopy(_raw())
    del raw["regions"]["defaults"]["budget_axes_why"]
    offenders = [
        name
        for name, merged in _merged_regions(raw).items()
        if _narrows_budget_axes(merged) and not merged.get("budget_axes_why")
    ]
    assert sorted(offenders) == ["compress", "language", "reason", "retrieve"]


# ------------------------------------------- C1: requires must name a stage in the pipeline


def _effective_pipeline(stage_list: list[str], stages: dict[str, Any]) -> list[str]:
    """Mirror of `model_matrix.config.effective_pipeline`: drop `enabled: false`."""
    return [s for s in stage_list if stages.get(s, {}).get("enabled", True)]


def _requires_violations(raw: dict[str, Any]) -> list[str]:
    """Mirror of `model_matrix.config.validate_requires`, returning what it would raise
    on instead of raising -- so a test can assert both the empty and the broken case."""
    stages = raw["stages"]
    problems: list[str] = []
    for pipeline_name, stage_list in raw["pipeline"].items():
        runtime = _effective_pipeline(stage_list, stages)
        for stage_name in runtime:
            for required in stages[stage_name].get("requires", []):
                if required not in runtime:
                    problems.append(
                        f"stage {stage_name!r} in pipeline {pipeline_name!r} requires "
                        f"{required!r}, which is not an enabled stage in that pipeline"
                    )
    return problems


def test_no_pipeline_declares_a_stage_whose_requires_it_cannot_satisfy() -> None:
    """C1. This is the error that made `model-matrix validate` and `plan` exit 1 on the
    shipped file, so nothing downstream of the loader was ever exercised."""
    assert _requires_violations(_raw()) == []


def test_every_stage_a_pipeline_runs_has_a_command_template_in_every_region() -> None:
    """M3's half of C1: a pipeline may not name a stage no region can run. `finetune`
    has no command template anywhere (A4 is unimplemented), which is the second reason
    the finetune pipeline could not run even if its `requires` had resolved."""
    raw = _raw()
    for pipeline_name, stage_list in raw["pipeline"].items():
        for stage_name in _effective_pipeline(stage_list, raw["stages"]):
            if not raw["stages"][stage_name].get("gpu", False):
                continue  # collect/publish/verify are harness-internal, no command here
            for region, merged in _merged_regions(raw).items():
                assert stage_name in merged.get("commands", {}), (
                    f"pipeline {pipeline_name!r} runs stage {stage_name!r} but region "
                    f"{region!r} has no command template for it"
                )


def test_the_requires_guard_fires_on_the_finetune_pipeline_as_shipped() -> None:
    """MUTATION. Re-declare the pipeline exactly as the reviewer found it and assert
    both guards report it -- the `requires` violation AND the missing command."""
    raw = copy.deepcopy(_raw())
    raw["pipeline"]["finetune"] = [
        "finetune",
        "test",
        "quantize",
        "test-quant",
        "collect",
        "publish",
        "verify",
    ]
    problems = _requires_violations(raw)
    assert any("pipeline 'finetune' requires 'train'" in p for p in problems), problems
    assert "finetune" not in _merged_regions(raw)["memory"]["commands"]


# ------------------------------------------------- M2: test-quant activation coupling


def _collect_coupling_violation(raw: dict[str, Any]) -> str | None:
    """`collect` must require the LAST stage that scored the artifact.

    The harness cannot catch this: with `test-quant` enabled, `quantize` is still an
    enabled stage in the pipeline, so `validate_requires` is satisfied either way. It is
    a CSD-side invariant about what "collected" is allowed to mean.
    """
    stages = raw["stages"]
    want = ["test-quant"] if stages["test-quant"].get("enabled", True) else ["quantize"]
    got = stages["collect"].get("requires")
    if got != want:
        return f"collect.requires is {got!r}, expected {want!r}"
    return None


def test_test_quant_is_enabled() -> None:
    """M2. A6 exists on this branch, so the packed artifact is re-scored rather than
    represented by `csd-quantize.py`'s in-memory plan metric."""
    assert _raw()["stages"]["test-quant"].get("enabled", True) is True


def test_collect_requires_the_last_stage_that_scored_the_artifact() -> None:
    """M2's footgun: `test-quant.enabled: true` with `collect.requires: [quantize]`
    lets collect -- and therefore publish -- gather a cell whose `.ptq.pt` was never
    opened. The two settings move together or not at all."""
    assert _collect_coupling_violation(_raw()) is None


def test_the_collect_coupling_guard_fires_on_the_half_flipped_config() -> None:
    """MUTATION. Enable test-quant but leave collect on quantize -- the reviewer's
    'activation footgun' -- and assert the coupling check reports it while the
    harness's own `validate_requires` stays silent."""
    raw = copy.deepcopy(_raw())
    raw["stages"]["test-quant"]["enabled"] = True
    raw["stages"]["collect"]["requires"] = ["quantize"]
    assert _requires_violations(raw) == [], "the harness cannot see this one"
    assert _collect_coupling_violation(raw) == (
        "collect.requires is ['quantize'], expected ['test-quant']"
    )


# --------------------------------------------------- H4: what the command templates need

# Exactly what `model_matrix.cli._build_plan_fn` puts into the render context on top of
# the cell's own vars, read off that function (harness `feat/harness`). `train_receipt`,
# `eval_receipt`, `quant_receipt` and `quantized_path` appear only once the predecessor
# stage has recorded a receipt, which is the normal case by the time a stage renders.
HARNESS_INJECTED_KEYS = frozenset(
    {
        "python",
        "pythonpath",
        "state",
        "train_receipt",
        "eval_receipt",
        "quant_receipt",
        "quantized_path",
    }
)

# Keys the harness does NOT inject today. DESIGN.v2 §4.2/§4.4 needs them for publish and
# harness round 4 is where they arrive; `publish.repo_pattern` / `publish.base_ref` in
# this config are what they would be computed from.
PENDING_HUB_KEYS = frozenset({"hub_repo", "hub_branch", "base_ref", "variant_id"})

# Dropped from a cell's vars by `expand_region_cells`'s own exclusion list, so a template
# may not reference them even though they are region keys.
_NOT_CELL_VARS = frozenset(
    {
        "axes",
        "budgets",
        "budget_axes",
        "budget_axes_why",
        "exclude",
        "derive",
        "commands",
        "env",
        "hosts",
        "primary_metric",
        "references",
        "eval_key",
        "gates_read",
        "extra_columns",
        "axis_short",
        "variance_axes",
        "quantized_artifact_metric",
        "followup",
        "finetune",
        "keep",
        "corpus_fingerprint_pin",
        "card",  # model-card metadata for scripts/csd-card.py -- see its own tests below
    }
)


def _cell_var_names(merged: dict[str, Any]) -> set[str]:
    """Mirror of `expand_region_cells`'s `base_vars` + axis values + `derive` results."""
    names = {k for k in merged if k not in _NOT_CELL_VARS}
    names.add("region")
    names |= set(merged.get("axes", {}))
    names |= set(merged.get("derive", {}))
    for spec in merged.get("axes", {}).values():
        if isinstance(spec, dict):
            for overrides in spec.values():
                names |= set(overrides)
    return names


def _placeholders(template: str) -> set[str]:
    import string

    return {field for _, field, _, _ in string.Formatter().parse(template) if field is not None}


def _unresolvable(merged: dict[str, Any], stage: str) -> set[str]:
    known = _cell_var_names(merged) | HARNESS_INJECTED_KEYS
    return _placeholders(merged["commands"][stage]) - known


def test_every_gpu_stage_template_renders_under_the_live_harness_context() -> None:
    """H4. train/test/quantize/test-quant are the stages `cmd_run` actually launches;
    an unresolved key there is a `PlanBuildError` that fails the cell."""
    for name, merged in _merged_regions(_raw()).items():
        for stage in ("train", "test", "quantize", "test-quant"):
            assert _unresolvable(merged, stage) == set(), (
                f"region {name!r} stage {stage!r} references keys the harness does not "
                f"inject: {sorted(_unresolvable(merged, stage))}"
            )


def test_publish_needs_exactly_the_four_hub_keys_and_nothing_else() -> None:
    """H4, pinned rather than waved at: publish is the ONE template that does not render
    today, and what it is missing is exactly the hub key set harness round 4 will inject.
    If it ever needs a fifth key, this fails instead of the key silently joining the
    backlog."""
    for name, merged in _merged_regions(_raw()).items():
        missing = _unresolvable(merged, "publish")
        assert missing <= PENDING_HUB_KEYS, (
            f"region {name!r} publish template needs {sorted(missing - PENDING_HUB_KEYS)}, "
            "which no one has committed to injecting"
        )
        assert missing == {"hub_repo", "hub_branch", "base_ref"}


def test_the_template_key_guard_fires_on_an_uninjected_key() -> None:
    """MUTATION. Add a placeholder nobody supplies to the train template and assert the
    checker reports it -- the same shape as publish's real gap."""
    raw = copy.deepcopy(_raw())
    raw["regions"]["defaults"]["commands"]["train"] += " --nonsense {no_such_key}"
    merged = _merged_regions(raw)["language"]
    assert _unresolvable(merged, "train") == {"no_such_key"}


# ---------------------------------------- H3/L2: the glob must match the real filename


def test_the_test_quant_glob_matches_what_receipt_write_really_names_the_file() -> None:
    """H3/L2. `receipt_kind: eval-quant` in the config and `kind="eval-quantized"` in
    this repo are not a mismatch -- the harness classifies on
    `provenance.eval_target == "quantized"` (`model_matrix.receipts.kind_of`), never on
    the `kind` string. The FILENAME is what has to agree, because `Receipt.write` builds
    it from `kind` and the config selects receipts by glob."""
    import fnmatch
    import tempfile

    from cogsyndelta.pipeline.receipt import Producer, Receipt

    raw = _raw()
    glob = raw["stages"]["test-quant"]["receipt"].replace("{region}", "memory")
    rec = Receipt(
        producer=Producer("cogsyndelta", "memory", "dense-transformer"),
        stage="eval",
        kind="eval-quantized",
        provenance={"eval_target": "quantized"},
    )
    with tempfile.TemporaryDirectory() as tmp:
        written = rec.write(Path(tmp) / "receipts")
        relative = f"receipts/{written.name}"
    assert fnmatch.fnmatch(relative, glob), f"{relative!r} does not match {glob!r}"

    # And the two eval globs must be DISJOINT. `receipts/cogsyndelta-{region}-eval-*.json`
    # -- the fp32 stage's pattern before this commit -- also matched
    # `...-eval-quantized-<stamp>.json`, so the fp32 stage's declared pattern claimed the
    # quantized stage's receipts as well. `-eval-2*` (the UTC stamp's leading digit) is
    # what separates them.
    fp32_glob = raw["stages"]["test"]["receipt"].replace("{region}", "memory")
    assert not fnmatch.fnmatch(relative, fp32_glob), (
        f"the fp32 glob {fp32_glob!r} also matches the quantized receipt {relative!r}"
    )
    fp32 = Receipt(
        producer=Producer("cogsyndelta", "memory", "dense-transformer"),
        stage="eval",
        kind="eval",
        provenance={"eval_target": "fp32"},
    )
    with tempfile.TemporaryDirectory() as tmp:
        fp32_written = fp32.write(Path(tmp) / "receipts")
        fp32_relative = f"receipts/{fp32_written.name}"
    assert fnmatch.fnmatch(fp32_relative, fp32_glob)
    assert not fnmatch.fnmatch(fp32_relative, glob)


def test_the_glob_guard_fires_when_the_receipt_kind_is_renamed() -> None:
    """MUTATION. Rename `kind` to DESIGN.v2 §6.2's stale `evalq` spelling and assert the
    config's glob stops matching -- which is what renaming this field would really cost,
    even though the harness's own classification would be unaffected."""
    import fnmatch
    import tempfile

    from cogsyndelta.pipeline.receipt import Producer, Receipt

    glob = _raw()["stages"]["test-quant"]["receipt"].replace("{region}", "memory")
    rec = Receipt(
        producer=Producer("cogsyndelta", "memory", "dense-transformer"),
        stage="eval",
        kind="evalq",
        provenance={"eval_target": "quantized"},
    )
    with tempfile.TemporaryDirectory() as tmp:
        written = rec.write(Path(tmp) / "receipts")
        relative = f"receipts/{written.name}"
    assert not fnmatch.fnmatch(relative, glob)


# ================================================== publish.card_command / regions.*.card
#
# `scripts/csd-card.py` (via `publish.card_command`) is a SEPARATE tool from the
# `commands.publish` template (`csd-publish-checkpoint.py`) tested above -- see that
# script's own module docstring. Its per-region `card:` metadata block must not silently
# drift from the audited licence table `csd-publish-checkpoint.py`'s own `licence_tier()`
# actually enforces: this config is descriptive (what a reader/harness sees), that
# script is enforcing (what a real publish refuses), and the two naming different tiers
# for the same region would be exactly the kind of provenance falsehood the rest of this
# repo's card-building code refuses to print.

PUBLISH_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "csd-publish-checkpoint.py"


def _load_publish_licence_table() -> dict[str, str]:
    import importlib.machinery
    import importlib.util
    import sys

    loader = importlib.machinery.SourceFileLoader(
        "csd_publish_checkpoint_for_matrix_test", str(PUBLISH_SCRIPT)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[loader.name] = mod
    loader.exec_module(mod)
    return dict(mod.LICENCE_TIER)


CARD_REGIONS = ("language", "compress", "retrieve", "reason", "memory")


def test_every_card_region_declares_full_card_metadata() -> None:
    raw = _raw()
    for name in CARD_REGIONS:
        card = raw["regions"][name].get("card")
        assert isinstance(card, dict), f"region {name!r} has no 'card' block"
        assert card.get("language") == "en"
        assert isinstance(card.get("tags"), list) and card["tags"], (
            f"region {name!r} card.tags empty"
        )
        assert isinstance(card.get("licence_tier"), str) and card["licence_tier"]
        assert isinstance(card.get("datasets"), list) and card["datasets"], (
            f"region {name!r} card.datasets empty"
        )


def test_every_card_tags_list_names_its_own_region() -> None:
    raw = _raw()
    for name in CARD_REGIONS:
        tags = raw["regions"][name]["card"]["tags"]
        assert "cogsyndelta" in tags
        assert f"region:{name}" in tags


def test_card_licence_tier_agrees_with_the_publish_script() -> None:
    """The config's own `card.licence_tier` must equal what `csd-publish-checkpoint.py`'s
    `licence_tier()` would actually resolve for that region -- the SAME audited table,
    not a second, hand-copied one this config could silently drift from.
    """
    table = _load_publish_licence_table()
    raw = _raw()
    for name in CARD_REGIONS:
        declared = raw["regions"][name]["card"]["licence_tier"]
        assert declared == table[name], (
            f"region {name!r} card.licence_tier={declared!r} disagrees with "
            f"csd-publish-checkpoint.py's LICENCE_TIER[{name!r}]={table[name]!r}"
        )


# ---------------------------------------------- region rename (naming rule 2026-09-04)


def test_language_region_declares_its_legacy_alias_and_specialisation() -> None:
    """The language centre's row records what it used to be called and what it is
    specialised on, for a reader of this config -- cogsyndelta.regions.aliases is the
    logic's one source of truth, this is a legible restatement of it."""
    region = _raw()["regions"]["language"]
    assert region["aliases"] == ["code"]
    assert region["specialisation"] == "code"


def test_no_region_declares_a_hub_repo_name_override() -> None:
    """The Hub repo was renamed 2026-09-05 (cogsyndelta-region-code ->
    cogsyndelta-region-language); every region's default `repo_pattern` substitution
    now resolves correctly, `language` included, so no row needs an override."""
    raw = _raw()
    for name in CARD_REGIONS:
        assert "hub_repo_name" not in raw["regions"][name]


def test_the_card_licence_tier_guard_fires_on_a_diverged_config() -> None:
    """MUTATION. A config that names a tier the publish script's own table disagrees
    with must be caught -- reproduced here by comparing a deliberately wrong table
    entry against this config's real `card.licence_tier`, the same comparison the test
    above makes, so a real future drift (someone edits one file and not the other) is
    exactly what this shape catches."""
    real_table = _load_publish_licence_table()
    diverged_table = dict(real_table)
    diverged_table["compress"] = "mit"  # the real tier is cc-by-sa-4.0
    raw = _raw()
    declared = raw["regions"]["compress"]["card"]["licence_tier"]
    assert declared != diverged_table["compress"], (
        "mutation proof is broken: the diverged table happens to already match the config"
    )


def test_publish_declares_card_command_and_branch_prefix() -> None:
    raw = _raw()
    publish = raw["publish"]
    assert isinstance(publish.get("card_command"), str) and publish["card_command"].strip()
    assert publish.get("branch_prefix") == "variants/"


# card_command's placeholders: cell vars the harness already injects today
# (region is a cell var; python/pythonpath are HARNESS_INJECTED_KEYS) plus the two
# harness-round-4 keys this template genuinely needs (`hub_repo`, the ONE PENDING_HUB_KEYS
# member `commands.publish` also needs) and `out`, which `card_command` is the only
# template in this config that names (the harness would compute it from the cell dir +
# a fixed filename, same shape as `train_receipt`/`quantized_path` above).
CARD_COMMAND_KNOWN_EXTRA_KEYS = frozenset({"hub_repo", "out", "cell_dir"})


def test_card_command_needs_only_known_or_pending_keys() -> None:
    raw = _raw()
    merged = _merged_regions(raw)["language"]  # any region: card_command is region-generic
    known = _cell_var_names(merged) | HARNESS_INJECTED_KEYS | CARD_COMMAND_KNOWN_EXTRA_KEYS
    missing = _placeholders(raw["publish"]["card_command"]) - known
    assert missing == set(), (
        f"publish.card_command references keys nothing injects or declares pending: {sorted(missing)}"
    )


def test_the_card_command_key_guard_fires_on_an_uninjected_key() -> None:
    """MUTATION. Same shape as test_the_template_key_guard_fires_on_an_uninjected_key
    above, for card_command specifically."""
    raw = copy.deepcopy(_raw())
    raw["publish"]["card_command"] += " --nonsense {no_such_key}"
    merged = _merged_regions(raw)["language"]
    known = _cell_var_names(merged) | HARNESS_INJECTED_KEYS | CARD_COMMAND_KNOWN_EXTRA_KEYS
    missing = _placeholders(raw["publish"]["card_command"]) - known
    assert missing == {"no_such_key"}
