"""The intra-op thread pin, with the failure it prevents constructed on both sides.

WHAT IS BEING PROVED, AND WHY IT TAKES TWO ARMS
The claim worth a test is "a phase-A run at a fixed seed and a fixed pin reproduces its
checkpoint". On its own that claim is unfalsifiable-looking: if the toy happened to be
thread-invariant, the same-pin test would pass forever while proving nothing about the pin.
So every same-pin assertion here is paired with a DIFFERENT-pin control on the same code
path. The control is what makes the guard a guard -- memory note
`verify-guards-by-making-them-fail`: three CSD guards were structurally incapable of firing
and reading them confirmed intent, not behaviour. If the control ever stops failing, the
same-pin test has gone vacuous and the failure message here says so.

The measurement this file defends, taken at `efd9ef5` before the pin existed: one seed, one
command line, `OMP_NUM_THREADS` 1..28, eleven distinct `checkpoint_sha256`, 69 differing
parameter tensors, max |dW| 0.109-0.136, and G29 reading FAIL at eight threads while 1, 2,
4 and 16 read PASS. See `phase_a.DEFAULT_PHASE_A_THREADS`.

WHY THESE GO THROUGH `main(argv)`
Same reason `test_phase_a_cli.py` gives: the pin's job is to make the HARNESS reproducible,
and a pin proved only through the trainer's Python API would leave the real entry path --
where the module is built before the trainer exists -- untested. The two subprocess tests
are the only ones that cannot be done in-process at all, because `OMP_NUM_THREADS` is read
when torch initialises and nothing after that can put it back.

Run directly:
    CUDA_VISIBLE_DEVICES="" python -m pytest tests/interconnect/test_thread_pin.py -q
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from cogsyndelta.interconnect.cli import main
from cogsyndelta.interconnect.gates import GateFailure
from cogsyndelta.interconnect.mind import WhiteMatter
from cogsyndelta.interconnect.phase_a import (
    DEFAULT_PHASE_A_THREADS,
    THREAD_KNOB_KEYS,
    PhaseAConfig,
    PhaseAResult,
    PhaseATrainer,
    ThreadPin,
    check_receipt_thread_pin,
    pin_threads,
)
from cogsyndelta.interconnect.receipts import ComposeReceipt
from tests.interconnect.test_phase_a import honest_frozen_set, identity_for, toy_batch

#: `PhaseATrainer.write_receipt`'s default filename, which `cli.run_phase_a` accepts.
RECEIPT_NAME = "cogsyndelta-white_matter-compose-phase-a.json"

#: The two pins the control arm contrasts. Both are measured to produce different weights
#: on this toy at one seed, and -- the part that matters for a CI runner with two cores --
#: they were measured to differ under `taskset -c 0` as well, i.e. with a single CPU
#: visible. `set_num_threads` sets a THREAD count, not a core count, so a host with fewer
#: cores than the pin still splits its reductions the pinned number of ways.
PIN_A = 1
PIN_B = 4

#: Steps per run. Enough that AdamW has amplified the ~1e-7 forward difference well past
#: any rounding in the sha, few enough that the whole file is a couple of seconds.
STEPS = 5


@pytest.fixture(autouse=True)
def _restore_thread_pin() -> Iterator[None]:
    """Put the process's thread count back after each test.

    Without this, a test that pins 1 leaves every later test in the session single
    threaded -- a real slowdown, and a cross-test coupling that would make this file's own
    control arm depend on execution order.
    """
    before = torch.get_num_threads()
    yield
    torch.set_num_threads(before)


def _run_cli(out_dir: Path, *, threads: int | None, seed: int = 0) -> dict[str, Any]:
    """Run `phase-a` through `main(argv)` and return the receipt it wrote.

    Args:
        out_dir: `--out-dir`; created by the run.
        threads: `--threads`, or `None` to omit the flag entirely.
        seed: `--seed`.

    Returns:
        The receipt, parsed off disk.
    """
    argv = ["phase-a", "--device", "cpu", "--steps", str(STEPS), "--seed", str(seed)]
    if threads is not None:
        argv += ["--threads", str(threads)]
    argv += ["--out-dir", str(out_dir)]
    main(argv)
    return json.loads((out_dir / RECEIPT_NAME).read_text())


def _sha(receipt: dict[str, Any]) -> str:
    """Return a receipt's trained-checkpoint sha."""
    return receipt["artifacts"]["checkpoint_sha256"]


# ----------------------------------------------------------------------------------
# The pair the task names: same pin identical, different pin different
# ----------------------------------------------------------------------------------


def test_the_same_seed_and_the_same_pin_reproduce_the_checkpoint(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """Two runs at one seed and one pin agree on `checkpoint_sha256`, bit for bit.

    This is the claim a phase-A receipt has to be able to make before anyone cites it. It
    is only meaningful alongside the control arm below; read the two together.
    """
    first = _run_cli(tmp_path / "same-a", threads=PIN_B)
    second = _run_cli(tmp_path / "same-b", threads=PIN_B)
    capsys.readouterr()

    assert _sha(first) == _sha(second)
    assert first["placement_knobs"]["knobs"]["threads"]["effective"] == PIN_B
    assert second["placement_knobs"]["knobs"]["threads"]["effective"] == PIN_B


def test_a_different_pin_changes_the_checkpoint_at_the_same_seed(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """THE CONTROL ARM. One seed, one command line, two pins, two different checkpoints.

    Everything else in this file rests on this assertion failing to be trivial. If these
    two shas were equal, the thread count would not reach the weights, the same-pin test
    above would be measuring nothing, and the receipt field this branch adds would be
    decoration.
    """
    low = _run_cli(tmp_path / "pin-low", threads=PIN_A)
    high = _run_cli(tmp_path / "pin-high", threads=PIN_A + 0)  # pinned identically
    other = _run_cli(tmp_path / "pin-other", threads=PIN_B)
    capsys.readouterr()

    # The positive control first: the low pin is itself reproducible, so any difference
    # below is attributable to the pin rather than to run-to-run noise.
    assert _sha(low) == _sha(high)
    assert _sha(low) != _sha(other), (
        f"phase A at seed 0 produced the same checkpoint at {PIN_A} and {PIN_B} threads. "
        "That is not a pass: it means the thread count no longer reaches the weights on "
        "this build, so the same-pin test in this file has gone vacuous and the pin's "
        "value can no longer be demonstrated. Re-measure before deleting anything -- the "
        "recorded measurement is eleven distinct checkpoint_sha256 over OMP 1..28."
    )


def test_the_pin_survives_a_hostile_ambient_thread_count(tmp_path: Path) -> None:
    """The defect itself, fixed: `OMP_NUM_THREADS` no longer decides the weights.

    This is the only claim in the file that cannot be made in-process. `OMP_NUM_THREADS`
    is read when torch initialises its thread pool, so demonstrating that the pin beats it
    requires two processes that were STARTED with different values -- which is exactly the
    shape of the original defect report.
    """
    shas = []
    for ambient in ("3", "17"):
        out_dir = tmp_path / f"omp-{ambient}"
        env = dict(os.environ, OMP_NUM_THREADS=ambient, CUDA_VISIBLE_DEVICES="")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "cogsyndelta.interconnect.cli",
                "phase-a",
                "--device",
                "cpu",
                "--steps",
                str(STEPS),
                "--threads",
                str(PIN_B),
                "--out-dir",
                str(out_dir),
            ],
            check=False,  # a gate FAIL at five steps is a non-zero exit; the receipt lands
            capture_output=True,
            env=env,
        )
        receipt = json.loads((out_dir / RECEIPT_NAME).read_text())
        assert receipt["placement_knobs"]["knobs"]["threads"]["effective"] == PIN_B
        shas.append(_sha(receipt))

    assert shas[0] == shas[1], (
        "OMP_NUM_THREADS still decides the checkpoint even with --threads pinned; the "
        "pin is being applied too late or is being overridden."
    )


# ----------------------------------------------------------------------------------
# What the receipt records
# ----------------------------------------------------------------------------------


def test_the_receipt_records_the_pin_read_back_from_the_process(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """`knobs.threads` is populated, complete, and agrees with the live process."""
    receipt = _run_cli(tmp_path / "explicit", threads=PIN_B)
    out = json.loads(capsys.readouterr().out)

    knobs = receipt["placement_knobs"]["knobs"]["threads"]
    assert sorted(knobs) == sorted(THREAD_KNOB_KEYS)
    assert knobs["source"] == "explicit"
    assert knobs["requested"] == PIN_B
    assert knobs["effective"] == torch.get_num_threads() == PIN_B
    assert knobs["interop"] == torch.get_num_interop_threads()
    assert knobs["default"] == DEFAULT_PHASE_A_THREADS

    placement = receipt["placement_knobs"]["placement"]
    assert placement["device"] == "cpu"
    assert placement["torch_version"] == torch.__version__

    # The same numbers reach the operator watching stdout, not only the file.
    assert out["threads"] == knobs


def test_an_omitted_flag_is_defaulted_and_the_receipt_says_it_was(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """No `--threads` still pins -- and is recorded as `source: "default"`, not as truth.

    The distinction is the whole of requirement 3: a run that took the default is still
    citable, but a reader must be able to tell that no human chose the number.
    """
    receipt = _run_cli(tmp_path / "defaulted", threads=None)
    capsys.readouterr()

    knobs = receipt["placement_knobs"]["knobs"]["threads"]
    assert knobs["source"] == "default"
    assert knobs["requested"] == DEFAULT_PHASE_A_THREADS
    assert knobs["effective"] == DEFAULT_PHASE_A_THREADS
    assert DEFAULT_PHASE_A_THREADS > 1, (
        "the default pin is 1, which is reproducible but 5.1x slower than 8 on a 2048^2 "
        "matmul; see DEFAULT_PHASE_A_THREADS for the table"
    )


# ----------------------------------------------------------------------------------
# The guards, each shown firing and each with a positive control
# ----------------------------------------------------------------------------------


@pytest.fixture
def trained(white_matter: WhiteMatter, toy_scope: Any) -> tuple[PhaseATrainer, PhaseAResult]:
    """A two-step phase-A run pinned at `PIN_B`, and its result.

    Two steps, because none of the guards below is about what the run learned -- they are
    about whether the receipt can say what produced it.
    """
    trainer = PhaseATrainer(
        white_matter,
        PhaseAConfig(steps=2, lr=3e-3, threads=PIN_B),
        honest_frozen_set(white_matter),
    )
    result = trainer.run(
        [toy_batch(white_matter, toy_scope)], [toy_batch(white_matter, toy_scope, seed=7)]
    )
    return trainer, result


@pytest.fixture
def built(trained: tuple[PhaseATrainer, PhaseAResult]) -> dict[str, Any]:
    """That run's BUILT receipt dict, never written to disk."""
    trainer, result = trained
    return trainer.build_receipt(result, identity_for(trainer.white_matter)).build()


def _with_threads(receipt: dict[str, Any], threads: Any) -> dict[str, Any]:
    """Return `receipt` with its `knobs.threads` block replaced by `threads`."""
    return dict(
        receipt,
        placement_knobs={
            "placement": receipt["placement_knobs"]["placement"],
            "knobs": {"threads": threads},
        },
    )


def test_build_receipt_refuses_when_the_count_moved_under_the_run(
    trained: tuple[PhaseATrainer, PhaseAResult],
) -> None:
    """A pin that changes mid-run is refused: no single number describes those weights."""
    trainer, result = trained

    # The positive control comes first, on the same path: unmoved, the receipt builds.
    trainer.build_receipt(result, identity_for(trainer.white_matter))

    torch.set_num_threads(PIN_A)
    with pytest.raises(GateFailure, match="thread count moved"):
        trainer.build_receipt(result, identity_for(trainer.white_matter))


def test_check_receipt_thread_pin_refuses_the_shape_that_shipped(built: dict[str, Any]) -> None:
    """The exact `{"placement": null, "knobs": null}` all thirty receipts carried is refused."""
    check_receipt_thread_pin(built)  # positive control: a real receipt passes

    shipped = dict(built, placement_knobs={"placement": None, "knobs": None})
    with pytest.raises(GateFailure, match="not a mapping"):
        check_receipt_thread_pin(shipped)


def test_check_receipt_thread_pin_refuses_a_missing_threads_block(built: dict[str, Any]) -> None:
    """Populated placement and knobs are not enough; the pin itself has to be in there."""
    hollow = dict(
        built,
        placement_knobs={"placement": built["placement_knobs"]["placement"], "knobs": {}},
    )
    with pytest.raises(GateFailure, match="recorded no intra-op thread pin"):
        check_receipt_thread_pin(hollow)


@pytest.mark.parametrize("key", THREAD_KNOB_KEYS)
def test_check_receipt_thread_pin_refuses_a_missing_knob(built: dict[str, Any], key: str) -> None:
    """Dropping any one of the five recorded numbers refuses the receipt."""
    threads = {k: v for k, v in built["placement_knobs"]["knobs"]["threads"].items() if k != key}
    with pytest.raises(GateFailure, match="missing"):
        check_receipt_thread_pin(_with_threads(built, threads))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source", "whatever", "source"),
        ("source", None, "source"),
        ("effective", 0, "at least 1"),
        ("effective", "8", "at least 1"),
        ("effective", None, "at least 1"),
        ("effective", True, "at least 1"),
    ],
)
def test_check_receipt_thread_pin_refuses_an_unusable_value(
    built: dict[str, Any], field: str, value: Any, message: str
) -> None:
    """A pin block that is present but says nothing usable is refused, not read past."""
    threads = dict(built["placement_knobs"]["knobs"]["threads"], **{field: value})
    with pytest.raises(GateFailure, match=message):
        check_receipt_thread_pin(_with_threads(built, threads))


def test_check_receipt_thread_pin_refuses_a_placement_without_the_build(
    built: dict[str, Any],
) -> None:
    """A pin reproduces weights only on a named device and torch build, so both are required."""
    maimed = dict(
        built,
        placement_knobs={
            "placement": {"device": "cpu"},
            "knobs": built["placement_knobs"]["knobs"],
        },
    )
    with pytest.raises(GateFailure, match="torch_version"):
        check_receipt_thread_pin(maimed)


def test_a_clamped_pin_is_recorded_rather_than_refused(built: dict[str, Any]) -> None:
    """`requested != effective` is a real outcome, so it is written down, not gated on.

    A guard that refused a clamp would push callers toward printing the request instead of
    the measurement, which is the defect this whole branch is fixing.
    """
    threads = dict(built["placement_knobs"]["knobs"]["threads"], requested=64, effective=20)
    check_receipt_thread_pin(_with_threads(built, threads))


def test_write_receipt_refuses_before_the_file_exists(
    trained: tuple[PhaseATrainer, PhaseAResult], tmp_path: Path
) -> None:
    """The refusal lands before disk: a receipt nobody can cite is one that never landed."""
    trainer, result = trained

    torch.set_num_threads(PIN_A)
    with pytest.raises(GateFailure):
        trainer.write_receipt(result, identity_for(trainer.white_matter), tmp_path)
    assert list(tmp_path.iterdir()) == []


# ----------------------------------------------------------------------------------
# `pin_threads` and the receipt-builder guard, directly
# ----------------------------------------------------------------------------------


def test_pin_threads_reads_back_rather_than_echoing() -> None:
    """`effective` comes from `torch.get_num_threads()`, not from the argument."""
    pin = pin_threads(PIN_B)
    assert isinstance(pin, ThreadPin)
    assert pin.requested == PIN_B
    assert pin.source == "explicit"
    assert pin.effective == torch.get_num_threads()
    assert pin.interop == torch.get_num_interop_threads()

    # The read-back is live, not frozen at construction: move the process and `observed`
    # follows it while `requested` does not.
    torch.set_num_threads(PIN_A)
    observed = pin.observed()
    assert observed.effective == PIN_A
    assert observed.requested == PIN_B


def test_pin_threads_defaults_and_labels_it() -> None:
    """`None` selects the default and is labelled, never passed off as an explicit choice."""
    pin = pin_threads(None)
    assert pin.requested == DEFAULT_PHASE_A_THREADS
    assert pin.source == "default"
    assert pin.default == DEFAULT_PHASE_A_THREADS


@pytest.mark.parametrize("bad", [0, -1])
def test_pin_threads_refuses_a_non_positive_pin(bad: int) -> None:
    """0 asks torch to choose, which is the ambient behaviour the pin exists to remove."""
    with pytest.raises(ValueError, match=">= 1"):
        pin_threads(bad)


def test_the_receipt_builder_refuses_a_null_placement_group() -> None:
    """`ComposeReceipt` itself refuses the null shape, so no future caller can restore it."""
    identity = {
        "corpus_fingerprint": "0" * 64,
        "fingerprint_scheme": "test",
        "battery_id": "plumbing",
        "k": 4,
        "pooling": "frontal",
        "checkpoint_sha256": "0" * 64,
        "region": "white_matter",
        "seed": 0,
        "split_sha256": "0" * 64,
    }
    receipt = ComposeReceipt("compose", identity)
    with pytest.raises(ValueError, match="must be a mapping"):
        receipt.set_placement_knobs({"placement": None, "knobs": None})
    with pytest.raises(ValueError, match="must be a mapping"):
        receipt.set_placement_knobs({"placement": {}, "knobs": None})
    # Positive control: two mappings are accepted, empty or not.
    receipt.set_placement_knobs({"placement": {}, "knobs": {}})
