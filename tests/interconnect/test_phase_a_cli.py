"""The phase-A harness on its real command line -- `cogsyndelta.interconnect.cli`.

Why this file exists separately from `test_phase_a.py`: a green unit suite has repeatedly
failed to predict what a stage script does here (memory note
`harness-shaped-smoke-before-push`: a receipt-key collision and an hours-long quantize both
survived a green suite and only surfaced on the harness command line). Every test below
goes through `main(argv)` -- the parser, the synthetic stream, the receipt on disk and the
exit code -- rather than through the trainer's Python API.

Run directly:
    CUDA_VISIBLE_DEVICES="" python -m pytest tests/interconnect/test_phase_a_cli.py -q
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cogsyndelta.interconnect import phase_a
from cogsyndelta.interconnect.cli import main
from cogsyndelta.interconnect.gates import GateFailure
from cogsyndelta.interconnect.phase_a import (
    PHASE_A_TRAINABLE_R5,
    TABLE_4_TOTAL_R5,
    check_receipt_collapse_floor,
)


def _run(capsys: pytest.CaptureFixture[str], argv: list[str]) -> tuple[int, dict]:
    """Run `main(argv)` and return `(exit_code, parsed stdout JSON)`."""
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_params_reports_the_v1_tables_at_r5(capsys: pytest.CaptureFixture[str]) -> None:
    """`params --r 5` measures the module and both table figures agree with it."""
    code, out = _run(capsys, ["params", "--r", "5"])
    assert code == 0
    assert out["module_total"] == TABLE_4_TOTAL_R5
    assert out["phase_a_trainable"] == PHASE_A_TRAINABLE_R5
    assert out["phase_a_frozen"] == TABLE_4_TOTAL_R5 - PHASE_A_TRAINABLE_R5
    assert out["frozen_names"] == ["controller"]
    assert out["matches_tables"] is True


def test_params_at_r4_also_freezes_the_store_projections(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Table 4's `R = 4` row: `W_k`/`W_v` "stay instantiated and receive no gradient"."""
    code, out = _run(capsys, ["params", "--r", "4"])
    assert code == 0
    assert sorted(out["frozen_names"]) == ["controller", "kv_bank"]
    assert out["phase_a_frozen"] == out["module_total"] - out["phase_a_trainable"]


def test_phase_a_smoke_at_three_steps_writes_a_valid_receipt(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Two or three steps on the real command line: the whole path runs and a receipt lands.

    The exit code is `1`, and that is the correct answer, not a defect: three steps do not
    teach an untrained read-out to rank `NULL` first, so G36 fires and the house convention
    (`cogsyndelta.regions.compress.main`) makes a failing gate a non-zero exit. What this
    test pins is that the failure is the GATE's, and everything around it worked -- the
    receipt is on disk, complete, and re-derivable by a reader.
    """
    code, out = _run(
        capsys, ["phase-a", "--steps", "3", "--device", "cpu", "--out-dir", str(tmp_path)]
    )
    assert code == 1
    assert out["verdict"].startswith("FAIL")
    assert "G36" in out["verdict"]
    assert out["collapsed_in_phase_A"] == []
    assert out["loss_last"] < out["loss_first"]
    assert out["store_records_primed"] == 8

    path = Path(out["receipt"])
    assert path.parent == tmp_path
    receipt = json.loads(path.read_text())
    assert receipt["schema"] == "model-pipeline-receipt/v1"
    assert receipt["stage"] == "compose"
    assert receipt["metrics_schema"] == "csd-metrics/v2"
    # Spec section 6 Q5 option (c): a plumbing run must be uncitable.
    assert receipt["battery_id"] == "plumbing"
    assert receipt["region"] == "white_matter"
    assert receipt["pooling"] == "frontal"
    check_receipt_collapse_floor(receipt)


def test_phase_a_gates_all_pass_once_the_toy_is_actually_trained(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The positive control for the smoke above: with enough steps every gate clears.

    Without this, `test_phase_a_smoke_at_three_steps_writes_a_valid_receipt` could not tell
    a guard that works from a guard that refuses everything -- the same reason spec section
    5 pairs every constructed failure with a positive control.
    """
    code, out = _run(
        capsys,
        ["phase-a", "--steps", "200", "--device", "cpu", "--out-dir", str(tmp_path)],
    )
    assert code == 0
    assert out["verdict"].startswith("PASS")
    assert out["collapsed_in_phase_A"] == []
    assert out["train_recall_at_1"] >= out["dev_recall_at_1"] - 0.05
    assert abs(out["overfit_gap_points"]) < 5.0
    check_receipt_collapse_floor(json.loads(Path(out["receipt"]).read_text()))


def test_the_store_receives_attention_mass_and_clears_its_own_floor(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Table 6's E2 row gates `mean(a_store) >= eta/R`; the smoke must reach it at all.

    A synthetic stream with a per-batch scope reads an empty partition every time, the
    store contributes exactly zero mass, and G29 fires for a reason that is an artefact of
    the stream. `cli.prime_store` and the single shared scope are what stop that; this is
    the test that would notice if either were removed.
    """
    _, out = _run(
        capsys, ["phase-a", "--steps", "5", "--device", "cpu", "--out-dir", str(tmp_path)]
    )
    floor = out["collapse_floor"]["value"]
    assert out["mean_attention_per_region"]["episodic_store"] > floor
    assert "episodic_store" not in out["collapsed_in_phase_A"]


def test_no_write_back_still_runs_and_freezes_the_prefixes(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Table 6 row A: "prefixes only when write-back is enabled"."""
    _, with_wb = _run(capsys, ["phase-a", "--steps", "2", "--out-dir", str(tmp_path / "on")])
    _, without_wb = _run(
        capsys,
        ["phase-a", "--steps", "2", "--no-write-back", "--out-dir", str(tmp_path / "off")],
    )
    assert without_wb["trainable_parameters"] < with_wb["trainable_parameters"]
    receipt = json.loads(Path(without_wb["receipt"]).read_text())
    assert receipt["write_back_topology"]["enabled"] is False
    assert "edges" not in receipt["write_back_topology"]


def test_the_receipt_records_the_loss_site_the_flags_asked_for(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """`--delta` and `--task-weight` reach the loss site, and the receipt reports it there.

    This is the ordinary case of the distinction `test_phase_a.py` tests by mutation: when
    nothing intervenes, the observed values and the parsed ones agree, and a receipt that
    reported neither would still look right. The mutation tests are what tell the two
    apart; this one checks the flags are wired at all.
    """
    _, out = _run(
        capsys,
        [
            "phase-a",
            "--steps",
            "2",
            "--delta",
            "0.25",
            "--task-weight",
            "2.0",
            "--out-dir",
            str(tmp_path),
        ],
    )
    assert out["loss_site"]["delta"] == 0.25
    assert out["loss_site"]["task_weight"] == 2.0
    receipt = json.loads(Path(out["receipt"]).read_text())
    assert receipt["budgets"]["loss_site"]["delta"] == 0.25
    assert receipt["budgets"]["loss_site"]["source"] == "observed at the loss site"


def test_the_command_line_refuses_a_receipt_that_prints_the_wrong_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """G29's receipt half, proved REACHABLE from the shipped path -- not just callable.

    `check_receipt_collapse_floor` fires five ways when a test calls it directly, and that
    proves nothing about the harness: until `PhaseATrainer.write_receipt` ran it, the run
    wrote its receipt and never re-derived the floor from it, so the specification's stated
    effect for G29 -- "the receipt is refused" -- could not happen to anything a user
    produced. This test is the one that would notice that wiring being removed again.

    The defect is injected the smallest way that a real one could arise: the run is
    otherwise the real run, and only the floor the receipt PRINTS is stale --
    `eta/4 = 0.0375` beside a run whose own `R` is 3, spec section 5's "a receipt printing
    3.75% at `R = 5`" with one field changed. The masses, the regions and
    `collapsed_in_phase_A` are all untouched, so nothing else can be what raises.

    The refusal must also arrive BEFORE the file: a receipt that is written and then
    complained about is a receipt a reader can already cite.
    """
    honest_guards = phase_a.phase_a_guards

    def stale_printed_floor(**kwargs: Any) -> Any:
        report = honest_guards(**kwargs)
        report.collapse_floor["value"] = 0.0375
        return report

    monkeypatch.setattr(phase_a, "phase_a_guards", stale_printed_floor)

    with pytest.raises(GateFailure, match=r"G29.*printed floor"):
        main(["phase-a", "--steps", "3", "--device", "cpu", "--out-dir", str(tmp_path)])

    assert list(tmp_path.iterdir()) == [], "a refused receipt must not reach disk"


def test_the_written_receipt_names_the_gates_it_could_not_evaluate(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Table 6 row A names G0, G1 and G2, and the JSON on disk has to say they were skipped.

    `PhaseAGuardReport.unimplemented_gates` asserted against the report object is invisible
    to the only audience that matters -- somebody holding the receipt and nothing else. The
    verdict string hedges ("every EVALUATED gate clear") but does not name what was left
    out, so the names go in the receipt.
    """
    _, out = _run(
        capsys, ["phase-a", "--steps", "3", "--device", "cpu", "--out-dir", str(tmp_path)]
    )
    raw = Path(out["receipt"]).read_text()
    verdicts = json.loads(raw)["verdicts"]

    assert verdicts["unimplemented_gates"]["gates"] == ["G0", "G1", "G2"]
    assert "not claimed to have passed" in verdicts["unimplemented_gates"]["reason"]
    # Table 7's own three strings are still exactly three strings beside it.
    assert isinstance(verdicts["integration"], str)
    assert isinstance(verdicts["scheduling"], str)
    assert isinstance(verdicts["trigger_sensitivity"], str)
    # And a reader with nothing but the file finds the names by grep, as the verifier did
    # when they were absent.
    for gate in ("G0", "G1", "G2"):
        assert gate in raw


def test_an_unknown_stream_is_refused_by_the_parser() -> None:
    """Only `synthetic` exists today; a real corpus is the compose-stage driver's job."""
    with pytest.raises(SystemExit):
        main(["phase-a", "--stream", "corpus"])
