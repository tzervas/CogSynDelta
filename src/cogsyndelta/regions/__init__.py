"""Region submodels: the specialists that compose into one mind.

LAZY BY CONSTRUCTION (N2)
This package used to import `cogsyndelta.regions.pretrain` and
`cogsyndelta.regions.text_encoder` eagerly, at module scope, so every public name it
re-exports was available the instant `cogsyndelta.regions` (or any submodule of it --
Python always executes a package's `__init__.py` before any of its submodules) was
imported. That is also exactly what broke `scripts/ci_local.sh`'s pre-push gate:
`cogsyndelta.quant.ptq.save_packed_artifact` lazily imports
`cogsyndelta.regions._checkpoint`, and `tests/test_publish_checkpoint.py` /
`tests/test_quant_ptq.py` import it directly -- neither uses `tokenizers` at all, but
both transitively imported `regions.pretrain`, which imports `tokenizers` at module
level, and `tokenizers` is in the `train` dependency group ci_local.sh's isolated venv
(`uv sync --group dev`) deliberately excludes (see that script's own header comment).

`__getattr__` (PEP 562) defers the real import to first use: `cogsyndelta.regions.X`
still resolves to exactly the same object a direct `from cogsyndelta.regions.pretrain
import X` would give, but only pays for that submodule's own imports (`torch`,
`tokenizers`, ...) when something actually asks for one of its names -- not merely by
existing under this package. `tests/test_import_hygiene.py` is the regression guard:
it proves `cogsyndelta.regions._checkpoint` and `cogsyndelta.quant.ptq` import cleanly
with `tokenizers` unavailable, and that `cogsyndelta.regions.pretrain` (which
genuinely needs it) still fails the same way it always did.

Every name callers reach through the package rather than a submodule --
`scripts/csd-train-all.py`'s `from cogsyndelta.regions import PretrainConfig,
pretrain_region` is the one such call site in this tree -- keeps working unchanged;
see `tests/test_import_hygiene.py::test_regions_public_api_resolves_when_tokenizers_available`
for the identity check (`cogsyndelta.regions.X is cogsyndelta.regions.<submodule>.X`,
not merely "importable").
"""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Only for static type checkers (mypy) -- never executed, so it costs nothing at
    # runtime and imports nothing eagerly. Re-exports the exact names __getattr__
    # resolves below, so `from cogsyndelta.regions import PretrainConfig` type-checks
    # the same as it always did.
    from cogsyndelta.regions.pretrain import (
        PretrainConfig,
        evaluate,
        evaluate_graded,
        load_graded_pairs,
        load_pairs,
        pretrain_region,
    )
    from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig, info_nce

__all__ = [
    "PretrainConfig",
    "TextEncoder",
    "TextEncoderConfig",
    "evaluate",
    "evaluate_graded",
    "info_nce",
    "load_graded_pairs",
    "load_pairs",
    "pretrain_region",
]

# Which submodule each public name actually lives in -- the whole table __getattr__
# below consults, so adding a re-export means adding one entry here (and to __all__)
# rather than an eager import at the top of this file.
_SUBMODULE_OF: dict[str, str] = {
    "PretrainConfig": "pretrain",
    "evaluate": "pretrain",
    "evaluate_graded": "pretrain",
    "load_graded_pairs": "pretrain",
    "load_pairs": "pretrain",
    "pretrain_region": "pretrain",
    "TextEncoder": "text_encoder",
    "TextEncoderConfig": "text_encoder",
    "info_nce": "text_encoder",
}


def __getattr__(name: str) -> Any:
    """PEP 562: resolve a public name by importing the ONE submodule that defines it,
    on first access, rather than every submodule at package-import time."""
    submodule_name = _SUBMODULE_OF.get(name)
    if submodule_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    submodule = importlib.import_module(f"{__name__}.{submodule_name}")
    value = getattr(submodule, name)
    setattr(sys.modules[__name__], name, value)  # cache: subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    """PEP 562 companion to `__getattr__`: `dir(cogsyndelta.regions)` and tab
    completion should list the public names `__getattr__` resolves, exactly as they
    would show up if this package still imported everything eagerly."""
    return sorted(__all__)
