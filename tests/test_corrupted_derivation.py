"""CPU tests for the E1 corrupted-derivation battery (diagnosis §4).

These prove the instrument, not a checkpoint: parsing, deterministic K=4
corruptions, #### propagation, chance 0.20, and the two lexical controls on
synthetic items. The 299-of-320 count is a scoring-run assertion against the
E0 split, not a CI fixture.
"""

from __future__ import annotations

import pytest

from cogsyndelta.eval.corrupted_derivation import (
    CHANCE,
    K_CORRUPTIONS,
    MIN_ANNOTATIONS,
    PREREG_NOTES,
    BatteryItem,
    _apply_edit,
    build_corrupted_battery,
    control_gates,
    corrupt_derivation,
    go_kill,
    hit_rate_at_1,
    parse_calculator_annotations,
    score_items_lexical,
    score_wrong_problem_tfidf,
    w2c_region_seed,
)
from cogsyndelta.splits import item_id

pytestmark = pytest.mark.cpu

NATALIA = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
    "#### 72"
)


def test_prereg_notes_state_reasoning_is_not_lookup() -> None:
    """The operator interpretation is in the module that stamps receipt notes."""
    assert "NOT lookup" in PREREG_NOTES
    assert "not trying to raise recall@1" in PREREG_NOTES
    assert "E5" in PREREG_NOTES
    assert "No change to E0" in PREREG_NOTES


def test_parse_binary_and_multi_operand_annotations() -> None:
    anns = parse_calculator_annotations(NATALIA)
    assert len(anns) == 2
    assert anns[0].left.strip() == "48/2"
    assert anns[0].result == "24"
    assert len(anns[0].operand_spans) == 2
    betty = (
        "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\n"
        "Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\n"
        "This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n"
        "#### 5"
    )
    multi = parse_calculator_annotations(betty)
    assert len(multi) == 3
    assert len(multi[2].operand_spans) == 4
    assert multi[2].result == "5"


def test_corrupt_is_deterministic_and_length_k() -> None:
    iid = item_id("q", NATALIA)
    a = corrupt_derivation(NATALIA, item_id=iid, corruption_seed=0)
    b = corrupt_derivation(NATALIA, item_id=iid, corruption_seed=0)
    assert a == b
    assert len(a) == K_CORRUPTIONS
    assert len(set(a)) == K_CORRUPTIONS
    assert NATALIA not in a


def test_corrupt_edits_one_annotation_number() -> None:
    iid = item_id("q", NATALIA)
    for corrupted in corrupt_derivation(NATALIA, item_id=iid):
        true_anns = parse_calculator_annotations(NATALIA)
        bad_anns = parse_calculator_annotations(corrupted)
        assert len(bad_anns) == len(true_anns)
        diffs = 0
        for left, right in zip(true_anns, bad_anns, strict=True):
            if left.left != right.left or left.result != right.result:
                diffs += 1
        assert diffs == 1


def test_final_line_propagates_when_result_is_the_answer() -> None:
    """Editing the result that equals #### must rewrite ####."""
    text = "step <<3+4=7>>7.\nthen <<7+1=8>>8.\n#### 8"
    anns = parse_calculator_annotations(text)
    start, end, original = anns[-1].result_span
    edited = _apply_edit(text, start, end, "9", original)
    assert "<<7+1=9>>" in edited
    assert "#### 9" in edited
    assert "#### 8" not in edited
    operand = anns[0].operand_spans[0]
    edited_op = _apply_edit(text, operand[0], operand[1], "4", operand[2])
    assert "#### 8" in edited_op
    assert "<<4+4=7>>" in edited_op


def test_items_with_fewer_than_two_annotations_are_dropped() -> None:
    holdout = [
        ("q1", "one step <<1+1=2>>2\n#### 2"),
        ("q2", NATALIA),
        ("q3", "not gsm8k at all"),
    ]
    items = build_corrupted_battery(holdout)
    assert len(items) == 1
    assert items[0].question == "q2"
    assert items[0].n_annotations >= MIN_ANNOTATIONS
    assert len(items[0].corruptions) == K_CORRUPTIONS


def test_chance_is_one_over_five() -> None:
    assert pytest.approx(0.20) == CHANCE


def test_hit_rate_breaks_ties_without_preferring_column_zero() -> None:
    import numpy as np

    tied = np.ones((500, 5), dtype=np.float64)
    rate = hit_rate_at_1(tied, relevant=0, tie_seed=0)
    assert 0.12 < rate < 0.28


def test_wrong_problem_tfidf_is_solvable() -> None:
    items: list[BatteryItem] = []
    for i in range(8):
        q = f"how many widgets does alice-{i} sell in april"
        true = (
            f"alice-{i} sold 10 widgets. 10/1 = <<10/1=10>>10. "
            f"10+{i} = <<10+{i}={10 + i}>>{10 + i}.\n"
            f"#### {10 + i}"
        )
        iid = item_id(q, true)
        items.append(
            BatteryItem(
                item_id=iid,
                question=q,
                true_derivation=true,
                corruptions=tuple(corrupt_derivation(true, item_id=iid)),
                n_annotations=2,
            )
        )
    wrong = score_wrong_problem_tfidf(items)
    assert wrong["recall@1"] >= 0.90


def test_tfidf_on_corruptions_is_not_ceiling() -> None:
    holdout = []
    for i in range(12):
        q = f"problem {i}: start with 40, add 2, how many"
        true = f"start 40. 40+2 = <<40+2=42>>42. then <<42+{i}={42 + i}>>{42 + i}.\n#### {42 + i}"
        holdout.append((q, true))
    items = build_corrupted_battery(holdout)
    tfidf = score_items_lexical(items, scorer="tfidf")
    bm25 = score_items_lexical(items, scorer="bm25")
    assert tfidf["recall@1"] < 0.80
    assert bm25["recall@1"] < 0.80


def test_control_gates_and_go_kill() -> None:
    gates = control_gates(0.20, 0.21, 0.95)
    assert gates["tfidf_at_chance"]
    assert gates["bm25_at_chance"]
    assert gates["wrong_problem_solvable"]
    assert not control_gates(0.50, 0.20, 0.95)["tfidf_at_chance"]
    assert not control_gates(0.20, 0.20, 0.50)["wrong_problem_solvable"]
    assert go_kill({"b256-s0": 0.31, "b512-s0": 0.10})["verdict"] == "go"
    assert go_kill({"b256-s0": 0.22, "b512-s0": 0.24})["verdict"] == "kill"
    assert go_kill({"b256-s0": 0.27, "b512-s0": 0.28})["verdict"] == "none"


def test_w2c_reason_seed_is_stable() -> None:
    assert w2c_region_seed("reason") == w2c_region_seed("reason")
    assert w2c_region_seed("reason") != w2c_region_seed("retrieve")
