"""Tests for the metrics-methodology deliverable.

Two things are proven here, not merely described:

1. `docs/design/METRICS-METHODOLOGY.md`'s own `file:line` anchors resolve to real files
   with enough lines to cover them -- a drift guard. It does NOT check that the prose
   still matches what is at that line, only that the anchor has not gone stale (a
   renamed function, a shortened file). It also checks a handful of specific claims the
   doc is required to state plainly (the quant-vs-eval battery distinction, the
   untrained baseline, the PTQ-ratio-is-not-a-speed-claim caveat, the
   anisotropy-is-a-diagnostic-not-a-score caveat, the licence-follows-corpus caveat).

2. `scripts/csd-publish-checkpoint.py`'s `build_card` refuses to print a metric with no
   entry in `METRIC_METHODOLOGY`, and the "How these numbers were produced" section it
   emits actually names every metric key the rest of the card prints -- read directly
   off the card's OWN rendered markdown tables, not off `build_card`'s internal
   bookkeeping, so this cannot pass merely because two hand-maintained lists agree with
   each other. The mutation-proof test at the bottom stubs `METRIC_METHODOLOGY` empty
   and asserts the card build then fails, proving that enforcement is load-bearing.

   This mutation proof belongs here rather than in `tests/test_guards_can_fail.py`:
   that file's own framing is six specific, already-fixed defects in the contamination
   and corpus-fingerprint guards under `cogsyndelta.eval`, and its imports are scoped to
   that surface. The guard under test here lives in a *script*
   (`scripts/csd-publish-checkpoint.py`, loaded via `SourceFileLoader` -- see
   `tests/test_publish_checkpoint.py:load_mod`), not in `cogsyndelta.eval`, so a new
   metrics-methodology defect class fits this file, which already owns that script's
   fixtures, rather than being grafted onto an unrelated import surface.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.test_publish_checkpoint import (
    make_checkpoint,
    make_eval_receipt,
    make_quant_receipt,
    make_training_receipt,
    mod,
)

pytestmark = pytest.mark.cpu

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "design" / "METRICS-METHODOLOGY.md"


@pytest.fixture(autouse=True)
def _allow_tmp_path_as_checkpoint_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same allow-list extension `tests/test_publish_checkpoint.py` applies to itself.
    Required again here: pytest does not apply another MODULE's autouse fixture to this
    one, only a conftest.py's -- importing the helper functions does not import the
    fixture's effect."""
    monkeypatch.setattr(mod, "ALLOWED_CHECKPOINT_ROOTS", [*mod.ALLOWED_CHECKPOINT_ROOTS, tmp_path])


# =====================================================================================
# The doc itself: file:line anchors don't drift, and required claims are stated plainly.
# =====================================================================================


def test_doc_exists() -> None:
    assert DOC.is_file(), f"{DOC} is missing"


_ANCHOR_RE = re.compile(r"`((?:src|scripts|tests)/[A-Za-z0-9_./-]+\.py):(\d+)(?:-(\d+))?`")


def _anchors() -> list[tuple[str, int, int]]:
    """Every unique `file:line` / `file:start-end` anchor cited in the doc."""
    text = DOC.read_text() if DOC.is_file() else ""
    out: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for m in _ANCHOR_RE.finditer(text):
        path, start_s, end_s = m.group(1), m.group(2), m.group(3)
        start = int(start_s)
        end = int(end_s) if end_s else start
        key = (path, start, end)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def test_doc_cites_a_realistic_number_of_anchors() -> None:
    """Guards the guard: if the doc's citation format drifts (e.g. anchors stop being
    backtick-wrapped `path:line`), `_ANCHOR_RE` would silently match nothing and
    `test_anchor_resolves` below would pass vacuously (zero parametrized cases). A
    document this detailed cites well over 50 anchors; require at least that many so a
    format change is caught here rather than by a test suite that quietly stopped
    checking anything.
    """
    assert len(_anchors()) > 50


def test_doc_cites_every_file_the_task_named() -> None:
    cited = {path for path, _, _ in _anchors()}
    for required in (
        "src/cogsyndelta/eval/metrics.py",
        "src/cogsyndelta/eval/benchmark.py",
        "src/cogsyndelta/eval/beir_fiqa.py",
        "src/cogsyndelta/quant/ptq.py",
        "src/cogsyndelta/regions/pretrain.py",
        "scripts/csd-benchmark.py",
        "scripts/csd-quantize.py",
    ):
        assert required in cited, f"{DOC} never cites {required}"


@pytest.mark.parametrize("path,start,end", _anchors())
def test_anchor_resolves(path: str, start: int, end: int) -> None:
    """Every cited file exists and has at least `end` lines. Cheap, and does not
    re-derive the doc's prose -- it only proves the anchor has not gone stale."""
    p = ROOT / path
    assert p.is_file(), f"{path} (cited at line {start}) no longer exists"
    n = len(p.read_text().splitlines())
    assert n >= end, f"{path}:{start}-{end} cited, but the file now has only {n} lines"


def test_doc_states_the_quant_vs_eval_battery_distinction() -> None:
    """The exact confusion this project was already bitten by: `quantized_metric` (the
    quantize stage) and `rank.*` (the eval/eval-quantized receipts) come from different
    batteries and are comparable only metric-for-metric, not section-for-section."""
    text = DOC.read_text()
    assert "quantized_metric" in text
    assert "training held-out battery" in text
    assert "eval battery" in text
    assert "plan, not" in text or "plan-vs-artifact" in text.lower()


def test_doc_explains_the_untrained_baseline() -> None:
    text = DOC.read_text()
    assert "untrained_baseline" in text
    assert "chance" in text.lower()
    assert "why every metric is reported beside it" in text.lower()


def test_doc_states_ptq_ratio_is_not_a_speed_claim() -> None:
    text = DOC.read_text()
    assert "payload/storage ratio" in text
    assert "not a speed or throughput" in text.lower()


def test_doc_states_anisotropy_is_a_diagnostic_not_a_score() -> None:
    text = DOC.read_text()
    assert "representation-geometry diagnostic" in text
    assert "not a quality score" in text.lower()


def test_doc_states_licence_tier_follows_the_corpus() -> None:
    text = DOC.read_text()
    assert "licence tier follows the corpus, not the metric" in text.lower()


def test_doc_flags_the_three_effective_rank_definitions() -> None:
    """The single most likely source of a silent misread: `effective_rank` (entropy) vs
    `participation_ratio` / `pr_effective_rank` (participation ratio) disagree in sign
    on this project's own production regions."""
    text = DOC.read_text()
    assert "pr_effective_rank" in text
    assert "participation_ratio" in text
    assert "participation-ratio" in text.lower() or "participation ratio" in text.lower()
    assert "entropy" in text.lower()


# =====================================================================================
# csd-metrics/v2: schema stamp, refuse predicate, retire list, deprecation map,
# anisotropy naming caveat, the W1 PR-vs-entropy sign disagreement, the dual-harness
# principle, and the standing statements. Source: g7-latent-eval-metrics.md (2026-09-04).
# =====================================================================================


def test_doc_stamps_the_v2_schema() -> None:
    text = DOC.read_text()
    assert "csd-metrics/v2" in text
    assert "metrics_schema" in text


def test_doc_reproduces_the_full_refuse_predicate() -> None:
    """Every axis the g7 spec's §3.3 refuse-function checks, reproduced (not merely
    referenced) so a reader does not have to cross into a session-scratchpad file to see
    what it requires."""
    text = DOC.read_text()
    for axis in (
        "same metrics_schema",
        "same corpus.fingerprint",
        "same battery_id",
        "same k",
        "same pooling",
        "same checkpoint sha256",
        "same region",
        "same code_revision.git_sha",
        "same seed",
    ):
        assert axis in text, f"refuse predicate missing axis: {axis!r}"
    # the battery_id enum itself, not just the word "battery_id"
    for battery in (
        "train_holdout",
        "eval_holdout",
        "eval_quantized_holdout",
        "quant_plan",
        "beir_fiqa_corpus",
        "beir_fiqa_split",
    ):
        assert battery in text, f"refuse predicate missing battery_id member: {battery!r}"


def test_doc_states_the_schema_falsifiers() -> None:
    """The three concrete ways the v2 patch itself would be theatre -- pre-registered
    before any implementation, per the spec's own falsification discipline."""
    text = DOC.read_text()
    assert "schema falsifiers" in text.lower()
    assert "repr.effective_rank_pr" in text  # the forbidden name, named explicitly


def test_doc_licenses_the_plan_vs_artifact_sameness_special_case() -> None:
    """MM §4's one explicitly licensed cross-battery_id comparison must survive into v2's
    refuse predicate as a named special case, not get swept up by "never compare across
    battery_id"."""
    text = DOC.read_text()
    assert "assert_sameness" in text
    assert "quant.plan_recall@1" in text
    assert "quant.artifact_recall@1" in text
    assert "violat" in text.lower() and "mm §4" in text.lower()


def test_doc_states_the_retire_list_with_reasons() -> None:
    text = DOC.read_text()
    assert "retire list" in text.lower()
    for retired in ("rank.map", "rank.precision@10"):
        assert retired in text
    assert "forbid the name" in text.lower()  # bare "effective_rank"
    assert "uses_its_dimensions" in text
    assert "repr.effective_rank_entropy_ratio" in text


def test_doc_carries_the_v1_to_v2_deprecation_map() -> None:
    text = DOC.read_text()
    assert "deprecation map" in text.lower()
    for v1_name, v2_name in (
        ("token_aware.final_block_rank.pooled_pr_rank", "token.pooled_pr_rank"),
        ("token_aware.final_block_rank.token_global_pr_rank", "token.global_pr_rank"),
        ("quantized_metric", "quant.plan_recall@1"),
        ("compression_ratio", "quant.compression_ratio"),
    ):
        assert v1_name in text, f"deprecation map missing v1 name {v1_name!r}"
        assert v2_name in text, f"deprecation map missing v2 name {v2_name!r}"


def test_doc_states_the_anisotropy_naming_caveat() -> None:
    """CSD's repr.anisotropy is NAMED after these papers but measures a different
    surface (pooled holdout vs. token-in-corpus) -- never compare the numbers."""
    text = DOC.read_text()
    assert "Ethayarajh" in text
    assert "Godey" in text
    assert "LoopFormer" in text
    assert "do not compare" in text.lower()


def test_doc_shows_the_w1_pr_vs_entropy_sign_disagreement() -> None:
    """The concrete table: PR ratios below 1.0, entropy ratios above 1.0, on the same
    four production regions, with the down-weights-tail / up-weights-tail explanation
    for why the two are expected to disagree rather than being a bug."""
    text = DOC.read_text()
    for region in ("code", "compress", "retrieve", "vl_latent"):
        assert region in text
    assert "0.66" in text and "1.84" in text
    assert "down-weight" in text.lower()
    assert "up-weight" in text.lower()


def test_doc_states_the_dual_harness_principle() -> None:
    text = DOC.read_text()
    assert "source of truth for gates" in text.lower()
    assert "detail.external" in text
    assert "never" in text.lower() and "alias" in text.lower()
    assert "csd-eval-bridge" in text or "model-matrix" in text


def test_doc_states_the_standing_statements() -> None:
    """The four standing statements the operator named: PTQ ratio is storage not
    latency; anisotropy is a diagnostic not a score; per-token only for token-mappable
    surfaces; the latent metrics are logged-only pending a pre-registered study."""
    text = DOC.read_text()
    assert "standing statements" in text.lower()
    assert "payload/storage ratio" in text
    assert "token-mappable" in text.lower()
    for field in ("loop.acc@k", "loop.kl_succ_mean", "probe.{acc_ling,acc_ctrl,sel}", "route."):
        assert field in text, f"standing statements missing latent field {field!r}"
    assert "logged-only" in text.lower() or "logged only" in text.lower()
    assert "pre-registered validation study" in text.lower() or "pre-registered" in text.lower()


# =====================================================================================
# The card: every printed metric carries a methodology line.
# =====================================================================================

_TABLE_ROW_RE = re.compile(r"^\| `([^`]+)` \|", re.MULTILINE)


def _printed_metric_keys(card: str) -> set[str]:
    """Every key `_dict_table` rendered anywhere in the card's own `## Metrics` section
    (held_out, untrained_baseline, gates, representation, contamination, quantization) --
    read directly off the rendered markdown, independent of
    `mod._card_metric_keys`'s own bookkeeping, so a pass here cannot be explained by two
    hand-maintained lists merely agreeing with each other."""
    start = card.index("## Metrics")
    end = card.index("## How these numbers were produced")
    return set(_TABLE_ROW_RE.findall(card[start:end]))


def _documented_metric_keys(card: str) -> set[str]:
    """Every key named as a row of the '## How these numbers were produced' table."""
    section = card.split("## How these numbers were produced", 1)[1]
    return set(_TABLE_ROW_RE.findall(section))


def _build_full_plan(tmp_path: Path):
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    eval_path = make_eval_receipt(tmp_path, checkpoint, region="compress")
    quant_path = make_quant_receipt(tmp_path, checkpoint, region="compress")
    plan = mod.build_plan(
        "compress", "tzervas/cogsyndelta-region-compress", train_path, eval_path, quant_path
    )
    return plan, train_path, eval_path, quant_path


def test_card_carries_the_how_produced_section(tmp_path: Path) -> None:
    plan, *_ = _build_full_plan(tmp_path)
    assert "## How these numbers were produced" in plan.card
    assert "docs/design/METRICS-METHODOLOGY.md" in plan.card


def test_every_printed_metric_has_a_methodology_line(tmp_path: Path) -> None:
    """The load-bearing assertion, read off the card's own rendered tables (not off
    build_card's internal bookkeeping): every metric key printed in `## Metrics` also
    appears as a row of `## How these numbers were produced`."""
    plan, *_ = _build_full_plan(tmp_path)
    printed = _printed_metric_keys(plan.card)
    assert printed, "fixture printed no metric rows -- this test would pass vacuously"
    documented = _documented_metric_keys(plan.card)
    missing = printed - documented
    assert not missing, f"printed with no methodology line: {sorted(missing)}"


def test_methodology_line_names_definition_battery_and_source(tmp_path: Path) -> None:
    """Spot-check the two metrics the operator named by name: `recall@1` and
    `anisotropy` must each carry all three of a definition, a battery, and a source
    file -- not just a bare key."""
    plan, *_ = _build_full_plan(tmp_path)
    section = plan.card.split("## How these numbers were produced", 1)[1]
    for key, battery, source in (
        ("recall@1", "training held-out battery", "src/cogsyndelta/eval/metrics.py"),
        ("anisotropy", "eval battery", "src/cogsyndelta/eval/benchmark.py"),
    ):
        row = next(line for line in section.splitlines() if line.startswith(f"| `{key}` |"))
        assert battery in row, f"{key} row missing its battery: {row!r}"
        assert source in row, f"{key} row missing its source file: {row!r}"


def test_card_metric_keys_matches_known_sections(tmp_path: Path) -> None:
    """Direct unit check of `_card_metric_keys` against a full trio of receipts: it must
    surface at least one key from each metric-table family the card can print."""
    _plan, train_path, eval_path, quant_path = _build_full_plan(tmp_path)
    train_receipt = json.loads(train_path.read_text())
    eval_receipt = json.loads(eval_path.read_text())
    quant_receipt = json.loads(quant_path.read_text())
    keys = set(mod._card_metric_keys(train_receipt, eval_receipt, quant_receipt))
    assert {"recall@1", "recall@10"} <= keys  # held_out / untrained_baseline / beats_untrained
    assert {"beats_untrained", "not_anisotropic", "uses_its_dimensions"} <= keys  # eval gates
    assert {"anisotropy", "effective_rank_ratio"} <= keys  # eval representation
    assert {"compression_ratio", "stored_bytes"} <= keys  # quant


def test_methodology_provenance_fields_present(tmp_path: Path) -> None:
    plan, train_path, eval_path, quant_path = _build_full_plan(tmp_path)
    section = plan.card.split("## How these numbers were produced", 1)[1]
    assert "**Corpus fingerprint:**" in section
    assert "**Seed:**" in section
    assert "**Code revision:**" in section
    assert f"**Training receipt:** `{train_path.name}`" in section
    assert f"**Eval receipt:** `{eval_path.name}`" in section
    assert f"**Quant receipt:** `{quant_path.name}`" in section


def test_fp32_only_card_still_gets_the_section_without_eval_or_quant_lines(
    tmp_path: Path,
) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    plan = mod.build_plan("compress", "tzervas/cogsyndelta-region-compress", train_path, None, None)
    assert "## How these numbers were produced" in plan.card
    section = plan.card.split("## How these numbers were produced", 1)[1]
    assert "**Eval receipt:**" not in section
    assert "**Quant receipt:**" not in section
    assert f"**Training receipt:** `{train_path.name}`" in section
    # Still non-vacuous: held_out/untrained_baseline/beats_untrained/contamination keys
    # are printed and documented even with no eval or quant receipt attached.
    printed = _printed_metric_keys(plan.card)
    assert printed
    assert not printed - _documented_metric_keys(plan.card)


# =====================================================================================
# Mutation proof: the enforcement is load-bearing, not decorative.
# =====================================================================================


def test_stubbed_methodology_map_fails_the_card_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty `METRIC_METHODOLOGY` and prove `build_plan` (which calls `build_card`, which
    calls `_methodology_section`) refuses rather than silently publishing a card with
    undocumented metrics. Without this test, `_every_printed_metric_has_a_methodology_line`
    above could be passing only because nothing ever exercises the refusal path -- this
    is the test that shows the refusal path is real."""
    monkeypatch.setattr(mod, "METRIC_METHODOLOGY", {})
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    with pytest.raises(mod.PublishAbortError, match="METRIC_METHODOLOGY"):
        mod.build_plan("compress", "tzervas/cogsyndelta-region-compress", train_path, None, None)


def test_methodology_map_missing_one_key_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Narrower mutation: drop only `recall@1` (the operator's own example metric) from
    the map and confirm the refusal names it -- not merely that SOME refusal happens."""
    trimmed = {k: v for k, v in mod.METRIC_METHODOLOGY.items() if k != "recall@1"}
    monkeypatch.setattr(mod, "METRIC_METHODOLOGY", trimmed)
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    with pytest.raises(mod.PublishAbortError, match=r"recall@1"):
        mod.build_plan("compress", "tzervas/cogsyndelta-region-compress", train_path, None, None)
