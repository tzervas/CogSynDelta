"""Tests for `cogsyndelta.cards.methodology`.

Covers: `require_documented` refuses on an undocumented key and names it
(mutation proof: stub the table empty and prove the refusal fires); `methodology_key`
strips the `rank./eff./repr.` family prefixes but leaves `quant.*` and prefix-free keys
alone; `normalize_quant_receipt_v1` copies v1 keys to their v2 names without
overwriting or mutating its argument; the `METRIC_METHODOLOGY` table itself is
internally consistent (every `battery_id` a card can print is one of the canonical g7
§3.3 ids, non-empty definitions, no accidental duplicate NamedTuple identity issues).
"""

from __future__ import annotations

import pytest

from cogsyndelta.cards.methodology import (
    EVAL_GATE_ALIASES_V1,
    EVAL_METRIC_ALIASES_V1,
    METHODOLOGY_DOC,
    METRIC_METHODOLOGY,
    QUANT_METRIC_ALIASES_V1,
    RETIRED_RANK_METRICS,
    CardError,
    MetricMethodology,
    methodology_key,
    normalize_quant_receipt_v1,
    require_documented,
)

pytestmark = pytest.mark.cpu


def test_methodology_doc_path_is_repo_relative() -> None:
    assert METHODOLOGY_DOC == "docs/design/METRICS-METHODOLOGY.md"


def test_every_entry_has_a_nonempty_definition_battery_and_source() -> None:
    for key, m in METRIC_METHODOLOGY.items():
        assert isinstance(m, MetricMethodology)
        assert m.definition.strip(), f"{key}: empty definition"
        assert m.battery.strip(), f"{key}: empty battery"
        assert m.source.strip(), f"{key}: empty source"


_CANONICAL_BATTERY_IDS = frozenset(
    {
        "",
        "train_holdout",
        "eval_holdout",
        "eval_quantized_holdout",
        "train_graded",
        "train_token_rank",
        "quant_plan",
        "beir_fiqa_corpus",
        "beir_fiqa_split",
    }
)


def test_every_battery_id_is_canonical_or_empty() -> None:
    for key, m in METRIC_METHODOLOGY.items():
        assert m.battery_id in _CANONICAL_BATTERY_IDS, (
            f"{key}: unrecognised battery_id {m.battery_id!r}"
        )


# --------------------------------------------------------------------------- methodology_key


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("rank.recall@1", "recall@1"),
        ("eff.peak_vram_mb", "peak_vram_mb"),
        ("repr.anisotropy", "anisotropy"),
        ("quant.plan_recall@1", "quant.plan_recall@1"),  # NOT stripped
        ("quant.compression_ratio", "quant.compression_ratio"),
        ("tolerance", "tolerance"),  # no dot at all
        ("within_budget", "within_budget"),
        ("beats_untrained_eval", "beats_untrained_eval"),
        ("probe.top1", "probe.top1"),  # not a rank/eff/repr family
        ("repr.rep_std", "repr.rep_std"),  # exact match wins over prefix strip
        ("rep_std", "rep_std"),
    ],
)
def test_methodology_key(raw: str, expected: str) -> None:
    assert methodology_key(raw) == expected


def test_methodology_key_covers_every_rank_eff_repr_metric_methodology_entry_is_resolvable() -> (
    None
):
    """Every bare key `METRIC_METHODOLOGY` documents under the `rank./eff./repr.`
    convention must be reachable by prefixing it and stripping again -- a sanity check
    that `methodology_key` and the table's own keying convention actually agree.
    Skip when the prefixed form is itself a documented key (visual `repr.rep_std`
    vs train `rep_std`): exact match wins before stripping.
    """
    for key in METRIC_METHODOLOGY:
        if "." in key:  # quant.* / probe.* / transfer.* / repr.rep_std; skip
            continue
        for prefix in ("rank", "eff", "repr"):
            prefixed = f"{prefix}.{key}"
            if prefixed in METRIC_METHODOLOGY:
                continue
            assert methodology_key(prefixed) == key


# --------------------------------------------------------------------------- require_documented


def test_require_documented_passes_for_known_keys() -> None:
    require_documented(["recall@1", "anisotropy", "quant.plan_recall@1"])


def test_require_documented_raises_and_names_the_missing_key() -> None:
    with pytest.raises(CardError, match=r"nonexistent_metric_xyz"):
        require_documented(["recall@1", "nonexistent_metric_xyz"])


def test_require_documented_mutation_proof_stubbed_empty() -> None:
    """The refusal is load-bearing, not decorative: an EMPTY methodology table must
    reject even a key this project's own table normally documents."""
    with pytest.raises(CardError, match=r"recall@1"):
        require_documented(["recall@1"], methodology={})


def test_require_documented_names_every_missing_key_not_just_the_first() -> None:
    with pytest.raises(CardError) as exc_info:
        require_documented(["recall@1", "bogus_a", "bogus_b"])
    msg = str(exc_info.value)
    assert "bogus_a" in msg
    assert "bogus_b" in msg
    assert "recall@1" not in msg  # documented -- must not be reported as missing


# --------------------------------------------------------------------------- alias maps


def test_quant_metric_aliases_v1_values_are_the_v2_canonical_names() -> None:
    for v2_name in QUANT_METRIC_ALIASES_V1.values():
        assert v2_name in METRIC_METHODOLOGY, f"{v2_name} (alias target) undocumented"


def test_eval_metric_aliases_v1_values_are_documented_under_their_bare_name() -> None:
    for v2_name in EVAL_METRIC_ALIASES_V1.values():
        assert methodology_key(v2_name) in METRIC_METHODOLOGY


def test_eval_gate_aliases_v1_values_are_documented() -> None:
    for v2_name in EVAL_GATE_ALIASES_V1.values():
        assert v2_name in METRIC_METHODOLOGY


def test_retired_rank_metrics_are_not_separately_documented_as_live_v2_names() -> None:
    """`rank.map`/`rank.precision@10` are RETIRED (see METRICS-METHODOLOGY.md §13) --
    they must not appear as a bare-keyed `METRIC_METHODOLOGY` entry implying a card may
    print them as a live v2 metric."""
    for retired in RETIRED_RANK_METRICS:
        assert retired not in METRIC_METHODOLOGY


# --------------------------------------------------------------------------- normalize_quant_receipt_v1


def test_normalize_quant_receipt_v1_adds_v2_keys() -> None:
    v1 = {"quantized_metric": 0.488, "compression_ratio": 9.83, "drop": 0.0078}
    out = normalize_quant_receipt_v1(v1)
    assert out["quant.plan_recall@1"] == 0.488
    assert out["quant.compression_ratio"] == 9.83
    assert out["quant.drop_recall@1"] == 0.0078
    # v1 keys kept too
    assert out["quantized_metric"] == 0.488


def test_normalize_quant_receipt_v1_does_not_mutate_its_argument() -> None:
    v1 = {"quantized_metric": 0.488}
    frozen = dict(v1)
    normalize_quant_receipt_v1(v1)
    assert v1 == frozen


def test_normalize_quant_receipt_v1_never_overwrites_an_existing_v2_value() -> None:
    already_v2 = {"quantized_metric": 0.1, "quant.plan_recall@1": 0.999}
    out = normalize_quant_receipt_v1(already_v2)
    assert out["quant.plan_recall@1"] == 0.999  # untouched, not overwritten by v1 value
