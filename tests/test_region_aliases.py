"""`cogsyndelta.regions.aliases` is the ONLY place the region rename mapping lives.

Every reader that might see a legacy region name (`code`, `vl_latent`) is expected to
resolve it through `canonical_region` before comparing or dispatching on it; every writer
is expected to emit whatever `canonical_region` returns. These tests pin the round trip
both directions and the refusal for input that cannot be a region name at all.
"""

from __future__ import annotations

import pytest

from cogsyndelta.regions.aliases import (
    REGION_ALIASES,
    canonical_region,
    is_legacy_name,
    legacy_names,
)


def test_legacy_code_resolves_to_language() -> None:
    assert canonical_region("code") == "language"


def test_legacy_vl_latent_resolves_to_visual() -> None:
    assert canonical_region("vl_latent") == "visual"


def test_canonical_names_are_fixed_points() -> None:
    """A name that is already canonical resolves to itself -- calling
    `canonical_region` twice, or on output that already went through it once, must be
    a no-op, or a reader that resolves twice (easy to do once this is called in more
    than one layer) would silently do the wrong thing."""
    assert canonical_region("language") == "language"
    assert canonical_region("visual") == "visual"


@pytest.mark.parametrize("name", ["memory", "reason", "residual_mlp", "compress", "retrieve"])
def test_untouched_region_names_pass_through(name: str) -> None:
    """Regions that were never renamed are not this module's concern -- it owns the
    rename mapping, not the full catalogue -- so they resolve to themselves."""
    assert canonical_region(name) == name


@pytest.mark.parametrize("bad", [None, "", 0, 3.5, []])
def test_non_string_or_empty_input_refused(bad: object) -> None:
    with pytest.raises(ValueError, match="unknown region name"):
        canonical_region(bad)  # type: ignore[arg-type]


def test_legacy_names_round_trips_for_renamed_regions() -> None:
    assert legacy_names("language") == ("code",)
    assert legacy_names("visual") == ("vl_latent",)


def test_legacy_names_empty_for_a_region_never_renamed() -> None:
    assert legacy_names("memory") == ()
    assert legacy_names("residual_mlp") == ()


def test_is_legacy_name() -> None:
    assert is_legacy_name("code") is True
    assert is_legacy_name("vl_latent") is True
    assert is_legacy_name("language") is False
    assert is_legacy_name("memory") is False


def test_region_aliases_table_is_exactly_the_two_documented_renames() -> None:
    """Pins the table's contents so a change to it is a deliberate, reviewed edit to
    THIS file, not an accidental one made while wiring up some other caller."""
    assert REGION_ALIASES == {"code": "language", "vl_latent": "visual"}


def test_canonical_region_and_legacy_names_are_inverse_on_the_renamed_pairs() -> None:
    for legacy, canonical in REGION_ALIASES.items():
        assert canonical_region(legacy) == canonical
        assert legacy in legacy_names(canonical)
