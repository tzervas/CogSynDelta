"""`scripts/csd-train-all.py`'s region maps must dispatch on EITHER spelling of a
renamed region (operator naming rule, 2026-09-04: `code` -> `language`, `vl_latent` ->
`visual`; `cogsyndelta.regions.aliases.REGION_ALIASES`).

`REGIONS` keeps its own key legacy (`code`) -- it is tied to the on-disk corpus
directory (`region/code/...`), matrix cell names and this module's own comments, none of
which rename with the catalogue -- with `language` added as an alias to the identical
entry. `VL_REGIONS` is the other way around: `visual` is the primary key (nothing in its
values names the region by its old spelling), with `vl_latent` added as the alias. Both
directions go through the SAME `_add_region_name_aliases` helper, so this file proves
the helper's contract once rather than once per dict.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu


def _load_csd_train_all():
    """Same loading convention every other consumer of this hyphenated script uses --
    see tests/test_reserved_corpus_guard.py's `_load_csd_train_all` docstring."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "csd-train-all.py"
    spec = importlib.util.spec_from_file_location("csd_train_all_region_aliases_test", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_csd_train_all()


def test_regions_dict_reachable_under_both_spellings() -> None:
    assert "code" in mod.REGIONS
    assert "language" in mod.REGIONS
    assert mod.REGIONS["code"] is mod.REGIONS["language"]


def test_vl_regions_dict_reachable_under_both_spellings() -> None:
    assert "visual" in mod.VL_REGIONS
    assert "vl_latent" in mod.VL_REGIONS
    assert mod.VL_REGIONS["visual"] is mod.VL_REGIONS["vl_latent"]


def test_region_spec_resolves_the_canonical_language_name() -> None:
    """`region_spec` (the one place every consumer is meant to unpack a REGIONS entry
    through) must accept `language` exactly like `code` -- same entry, same shape."""
    by_legacy = mod.region_spec("code")
    by_canonical = mod.region_spec("language")
    assert by_legacy.sources == by_canonical.sources
    assert by_legacy.note == by_canonical.note
    assert by_legacy.root == by_canonical.root


def test_mutating_the_shared_vl_region_entry_is_visible_under_either_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A test (or an operator) that does `monkeypatch.setitem(VL_REGIONS["vl_latent"],
    ...)` -- the pattern every pre-rename test in this tree already uses -- must mutate
    the SAME object `VL_REGIONS["visual"]` sees, not a stale copy."""
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "corpus_source", "alias-proof-corpus")
    assert mod.VL_REGIONS["visual"]["corpus_source"] == "alias-proof-corpus"


def test_add_region_name_aliases_does_not_invent_missing_entries() -> None:
    """A region with neither spelling present must not gain an entry from thin air --
    the helper mirrors existing entries, it does not manufacture new regions."""
    regions: dict[str, object] = {"memory": {"some": "config"}}
    result = mod._add_region_name_aliases(regions)
    assert set(result) == {"memory"}
