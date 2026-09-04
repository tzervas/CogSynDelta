"""A7's binding: which training receipt an eval uses, and which bytes it verifies.

Two defects the round-1 review found, both of the same shape -- a guard that reads one
place when the value lives in two, so it does not fail, it does not RUN:

* H2/M4. `expected_checkpoint_sha256` is the single source of the `expected_sha256`
  `load_checkpoint` verifies against. It reads the top-level `checkpoint_sha256` that
  `regions/pretrain.py` writes AND the `artifacts.checkpoint_sha256` an envelope-shaped
  receipt carries (what `pipeline.receipt.adapt`, and the matrix harness's own
  `receipts.adapt`, hand back), and REFUSES when neither is present rather than treating
  a missing hash as "no check needed".

Every guard here is paired with a mutation: the pre-fix expression is evaluated on the
same constructed receipt and shown to produce the wrong answer.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

SHA_A = "a" * 64
SHA_B = "b" * 64


def _load_csd_benchmark():
    """Import `scripts/csd-benchmark.py` -- a hyphenated filename is not a valid module
    name, so every consumer in this repo loads it this way."""
    path = SCRIPTS / "csd-benchmark.py"
    spec = importlib.util.spec_from_file_location("csd_benchmark_binding_test", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_csd_benchmark()


def _pre_fix_expected_sha(train_receipt: dict) -> str | None:
    """The expression this module used before H2: top level only, `or None` behind it.

    Kept verbatim so every mutation test below measures the real regression rather than
    a paraphrase of it.
    """
    return train_receipt.get("checkpoint_sha256") or None


# ------------------------------------------------------------------ H2: two locations


def test_a_top_level_checkpoint_sha_is_used() -> None:
    """The shape `regions/pretrain.py` writes; unchanged behaviour."""
    receipt = {"checkpoint": "/x/final.pt", "checkpoint_sha256": SHA_A}
    assert mod.expected_checkpoint_sha256(receipt, Path("r.json")) == SHA_A


def test_an_envelope_receipts_artifacts_checkpoint_sha_is_used(capsys) -> None:
    """H2. The shape A6 itself writes and the shape anything downstream of `adapt`
    returns: the hash lives under `artifacts`, and reading only the top level skipped
    the check silently."""
    receipt = {"checkpoint": "/x/final.pt", "artifacts": {"checkpoint_sha256": SHA_A}}
    assert mod.expected_checkpoint_sha256(receipt, Path("r.json")) == SHA_A


def test_the_pre_fix_read_skips_the_check_on_an_envelope_receipt() -> None:
    """MUTATION. The same receipt through the old expression yields `None`, which
    `load_checkpoint` treats as "nothing to verify against" -- the guard could not fire,
    which is why it never reported anything."""
    receipt = {"checkpoint": "/x/final.pt", "artifacts": {"checkpoint_sha256": SHA_A}}
    assert _pre_fix_expected_sha(receipt) is None
    assert mod.expected_checkpoint_sha256(receipt, Path("r.json")) == SHA_A


def test_two_locations_that_disagree_are_a_refusal() -> None:
    """A receipt claiming two different checkpoints cannot be resolved by preferring
    one; there is no basis for the preference."""
    receipt = {"checkpoint_sha256": SHA_A, "artifacts": {"checkpoint_sha256": SHA_B}}
    with pytest.raises(ValueError, match="two different checkpoints"):
        mod.expected_checkpoint_sha256(receipt, Path("r.json"))


def test_two_locations_that_agree_are_fine() -> None:
    receipt = {"checkpoint_sha256": SHA_A, "artifacts": {"checkpoint_sha256": SHA_A}}
    assert mod.expected_checkpoint_sha256(receipt, Path("r.json")) == SHA_A


# ------------------------------------------------------- M4: refuse an unbound receipt


def test_a_receipt_with_no_sha_anywhere_is_refused() -> None:
    """M4. `or None` treated a missing hash, an empty string and a genuinely pre-R9
    receipt identically -- as "no check needed". An eval that cannot be tied to the
    bytes it measured yields a number bound to a mutable path, which is what
    `csd-publish-checkpoint.py` refuses to publish."""
    receipt = {"checkpoint": "/x/final.pt"}
    with pytest.raises(mod.UnboundTrainReceiptError, match="allow-unbound-train-receipt"):
        mod.expected_checkpoint_sha256(receipt, Path("r.json"))


def test_an_empty_sha_string_is_refused_the_same_way() -> None:
    """`""` is what a pre-R9 receipt adapted through `pipeline.receipt.adapt` carries --
    'not recorded', not 'the file is empty'. It must not read as a waiver."""
    receipt = {"checkpoint_sha256": "", "artifacts": {"checkpoint_sha256": ""}}
    with pytest.raises(mod.UnboundTrainReceiptError):
        mod.expected_checkpoint_sha256(receipt, Path("r.json"))


def test_allow_unbound_makes_it_explicit_and_loud(capsys) -> None:
    """The documented exception: the caller says so, and the run says so back."""
    receipt = {"checkpoint": "/x/final.pt"}
    assert mod.expected_checkpoint_sha256(receipt, Path("r.json"), allow_unbound=True) is None
    assert "UNVERIFIED" in capsys.readouterr().out


def test_the_pre_fix_read_silently_waived_an_unbound_receipt() -> None:
    """MUTATION. The old expression returned `None` for a receipt with no hash at all,
    and `None` is exactly what it returned for a receipt that HAD one and matched -- the
    two cases were indistinguishable to every caller."""
    assert _pre_fix_expected_sha({"checkpoint": "/x/final.pt"}) is None
    with pytest.raises(mod.UnboundTrainReceiptError):
        mod.expected_checkpoint_sha256({"checkpoint": "/x/final.pt"}, Path("r.json"))


def test_the_cli_exposes_the_waiver_and_threads_it_to_both_eval_paths() -> None:
    """A refusal is only usable if there is a documented way past it, and only honest
    if the waiver reaches the code that would otherwise refuse. Asserted against the
    script's source because `main()` builds its parser inline and running it for real
    would need a checkpoint, a corpus and a tokenizer."""
    source = (SCRIPTS / "csd-benchmark.py").read_text()
    assert '"--allow-unbound-train-receipt"' in source
    # Both eval paths -- fp32 and quantized -- must receive it, not just the one.
    assert source.count("allow_unbound_train_receipt=args.allow_unbound_train_receipt") == 2
