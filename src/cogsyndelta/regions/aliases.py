"""The one place the code -> language / vl_latent -> visual rename mapping lives.

WHY THIS EXISTS
Operator naming rule (2026-09-04): regions are named by cognitive faculty, never by
knowledge domain or colloquial anatomy. The region historically called ``code`` is a
LANGUAGE CENTRE (id ``language``, ``specialisation: "code"`` recorded as metadata); the
visual runner ``vl_latent`` is the ``visual`` faculty. Everything already on disk and on
the Hub -- receipts, matrix cell directories, variant ids, Hub repo/branch names, corpus
fingerprints, licence tiers -- still uses the legacy spelling, and stays that way; only
callers are expected to accept both. A mapping duplicated across every reader is a
mapping that drifts the first time a third name gets renamed, so it lives here exactly
once: every reader resolves a region name through ``canonical_region`` before comparing
or dispatching on it, and every writer emits the canonical id returned from it.

This module owns the RENAME, not the full region catalogue. ``config/mind/csd-regions.json``
is the source of truth for which region ids exist at all; this module only needs to know
which of those ids used to be spelled differently, so ``canonical_region`` treats any
name that is not a recognized legacy spelling as already canonical -- it is not this
module's job to reject a typo of an unrelated region name (e.g. ``"languag"``), only to
refuse input that cannot be resolved to a real string at all.
"""

from __future__ import annotations

REGION_ALIASES: dict[str, str] = {
    "code": "language",
    "vl_latent": "visual",
}
"""Legacy region name -> canonical region name. THE ONLY place this mapping lives."""

_CANONICAL_TO_LEGACY: dict[str, tuple[str, ...]] = {}
for _legacy, _canonical in REGION_ALIASES.items():
    _CANONICAL_TO_LEGACY[_canonical] = (*_CANONICAL_TO_LEGACY.get(_canonical, ()), _legacy)


def canonical_region(name: str) -> str:
    """Resolve `name` to its canonical region id.

    A legacy spelling (a key of `REGION_ALIASES`) resolves to its canonical
    replacement. Any other non-empty string is returned unchanged -- most region
    names (`memory`, `reason`, `residual_mlp`, ...) were never renamed, and this
    module does not own the full catalogue, so passing one through is correct, not
    a missed case.

    Raises:
        ValueError: `name` is not a non-empty string (`None`, `""`, or a non-str),
            since that can never be a real region id, legacy or canonical.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"unknown region name: {name!r}")
    return REGION_ALIASES.get(name, name)


def legacy_names(canonical: str) -> tuple[str, ...]:
    """Every legacy spelling that resolves to `canonical`, oldest recorded first.

    Empty for a region that was never renamed (most of them) -- e.g.
    `legacy_names("memory") == ()`, `legacy_names("language") == ("code",)`.
    """
    return _CANONICAL_TO_LEGACY.get(canonical, ())


def is_legacy_name(name: str) -> bool:
    """True iff `name` is a pre-rename spelling this module knows how to resolve."""
    return name in REGION_ALIASES
