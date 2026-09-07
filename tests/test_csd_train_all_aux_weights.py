"""`scripts/csd-train-all.py` must be able to express auxiliary weights of 0.0.

The runner hard-coded `memory`'s (0.1, 0.1) with no override, so a pre-registered arm
requiring both weights at 0.0 (PREREG-RETRIEVAL-NEGATIVES-2026-09-06 rev 3 section 2.1)
could not be launched through it -- and an attempt would have trained the production
objective and completed normally. These tests pin the override, and specifically that
`0.0` is a value rather than "unset": a falsy resolution (`override or declared`) would
reintroduce the identical bug while looking correct.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "csd-train-all.py"


def _module():
    """Import the hyphenated script by path, as the other runner tests do."""
    spec = importlib.util.spec_from_file_location("csd_train_all_aux", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_declared_weights_are_kept_when_no_override_is_given() -> None:
    module = _module()
    assert module._auxiliary_weights("memory", None, None) == (0.1, 0.1)
    assert module._auxiliary_weights("code", None, None) == (0.0, 0.0)


def test_zero_is_an_override_and_not_an_absent_value() -> None:
    module = _module()
    assert module._auxiliary_weights("memory", 0.0, 0.0) == (0.0, 0.0)
    # The half-override case a falsy check also gets wrong.
    assert module._auxiliary_weights("memory", 0.0, None) == (0.0, 0.1)
    assert module._auxiliary_weights("memory", None, 0.0) == (0.1, 0.0)


def test_the_flags_exist_and_default_to_not_overriding() -> None:
    """A default of 0.0 would silently turn the terms off for every region; the default
    has to be None so an ordinary run is unchanged."""
    source = _SCRIPT.read_text()
    assert '"--token-loss-weight"' in source
    assert '"--decorr-weight"' in source
    assert "token_loss_weight=args.token_loss_weight" in source
    assert "decorr_weight=args.decorr_weight" in source
