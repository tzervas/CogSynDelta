"""Anchor-drift guard for `docs/technical/model-card-pipeline.md`.

Mirrors `tests/test_metrics_methodology.py`'s drift guard against the same defect
class: a `` `path:line` `` citation that is real and long enough (what a bare
existence/length check proves) but no longer lands on the identifier the prose names
beside it, because the source moved and the doc was never re-derived. A scoped review
of `refactor/language-centre` found exactly this in this doc: `licence_tier`'s citation
pointed at `scripts/csd-publish-checkpoint.py:281`, which the branch's own
`canonical_region()` insertion two commits earlier had already shifted to line 310 --
and nothing in the suite caught it, because `test_metrics_methodology.py`'s anchor
tests are parametrized only over `docs/design/METRICS-METHODOLOGY.md`'s own citations.

This doc's citation shape differs from METRICS-METHODOLOGY.md's `` `name()` `` (call
parens, directly adjacent) convention: it cites a bare `` `name` `` (no parens) next to
a `` `path:line` `` anchor, either in prose (`` `name` (`path:line`) ``) or inside a
table cell (`` `ExceptionType` (`name`, `path:line`) ``). Reusing
`test_metrics_methodology.py`'s `_NAMED_ANCHOR_RE` here would match zero cases and pass
vacuously -- exactly the silent-no-op failure mode
`test_doc_cites_a_realistic_number_of_named_anchors` below exists to catch -- so this
file defines its own regex for the shape this doc actually uses, while reusing that
module's `_defs_in_source`/`_named_anchor_ok` (pure, doc-independent AST-walk helpers
that already carry their own mutation proof in `test_metrics_methodology.py`) rather
than re-typing that logic.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.test_metrics_methodology import _defs_in_source, _named_anchor_ok

pytestmark = pytest.mark.cpu

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "technical" / "model-card-pipeline.md"


def test_doc_exists() -> None:
    assert DOC.is_file(), f"{DOC} is missing"


# =====================================================================================
# Plain `path:line` anchors: exist, and are long enough to contain the cited line.
# =====================================================================================

_ANCHOR_RE = re.compile(r"`((?:src|scripts|tests)/[A-Za-z0-9_./-]+\.py):(\d+)(?:-(\d+))?`")


def _anchors() -> list[tuple[str, int, int]]:
    """Every unique `file:line` / `file:start-end` anchor cited in the doc."""
    text = DOC.read_text() if DOC.is_file() else ""
    out: list[tuple[str, int, int]] = []
    seen: set[tuple[str, int, int]] = set()
    for m in _ANCHOR_RE.finditer(text):
        path, start_s, end_s = m.group(1), m.group(2), m.group(3)
        start = int(start_s)
        end = int(end_s) if end_s else start
        key = (path, start, end)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def test_doc_cites_a_realistic_number_of_anchors() -> None:
    """Guards the guard: if the doc's citation format drifts, `_ANCHOR_RE` would
    silently match nothing and `test_anchor_resolves` below would pass vacuously (zero
    parametrized cases). The doc cites 6 distinct anchors today; require at least 5 so a
    format change is caught here rather than by a suite that quietly stopped checking."""
    assert len(_anchors()) >= 5


@pytest.mark.parametrize("path,start,end", _anchors())
def test_anchor_resolves(path: str, start: int, end: int) -> None:
    """Every cited file exists and has at least `end` lines. Cheap, and does not
    re-derive the doc's prose -- it only proves the anchor has not gone stale."""
    p = ROOT / path
    assert p.is_file(), f"{path} (cited at line {start}) no longer exists"
    n = len(p.read_text().splitlines())
    assert n >= end, f"{path}:{start}-{end} cited, but the file now has only {n} lines"


# =====================================================================================
# Named anchors: `` `name` `` (bare, no `()`) immediately next to a `path:line` --
# this doc's own citation convention, distinct from METRICS-METHODOLOGY.md's.
# =====================================================================================

_NAMED_ANCHOR_RE = re.compile(
    r"`([A-Za-z_][A-Za-z0-9_.]*)`(?:[^`|]{0,60})"
    r"`((?:src|scripts)/[A-Za-z0-9_./-]+\.py):(\d+)(?:-(\d+))?`"
)


def _named_anchors() -> list[tuple[str, str, int, int]]:
    """Every bare `` `name` `` followed, within 60 chars and no intervening backtick or
    table-cell boundary (`|`), by a `` `path:line` `` anchor -- i.e. every place the doc
    claims "this citation is where `name` is defined/raised". The no-intervening-
    backtick rule is what makes `` `CardError` (`require_documented`, `path:line`) ``
    resolve to `require_documented` (the function that actually raises at that line),
    not the exception class named one backtick-span earlier."""
    text = DOC.read_text() if DOC.is_file() else ""
    out: list[tuple[str, str, int, int]] = []
    for m in _NAMED_ANCHOR_RE.finditer(text):
        name, path, start_s, end_s = m.group(1), m.group(2), m.group(3), m.group(4)
        start = int(start_s)
        out.append((name, path, start, int(end_s) if end_s else start))
    return out


def test_doc_cites_a_realistic_number_of_named_anchors() -> None:
    """Guards `_named_anchors` itself the same way `test_doc_cites_a_realistic_number_of_
    anchors` guards `_anchors`: if the bare-name-next-to-anchor convention drifts, this
    format-drift check catches `test_named_anchor_resolves` passing vacuously on zero
    parametrized cases. The doc cites 7 named anchors today (`build_card`,
    `render_card` x2, `require_documented`, `assert_schemas_agree`, `_licence_block`,
    `licence_tier`); require at least 5."""
    assert len(_named_anchors()) >= 5


@pytest.mark.parametrize("name,path,start,end", _named_anchors())
def test_named_anchor_resolves(name: str, path: str, start: int, end: int) -> None:
    """The anchor must land on the def of `name` (or, for a dotted citation like
    `cogsyndelta.cards.render_card`, on the def of its unqualified tail `render_card`).
    A citation that merely points at a long-enough file (what `test_anchor_resolves`
    checks) is not enough: this is the exact defect class the scoped review caught --
    `licence_tier` cited at line 281 was real and long enough, but line 281 was
    `local: Path | bytes, ...`, an unrelated helper's signature; `def licence_tier`
    had moved to 310 two commits earlier in this same branch."""
    p = ROOT / path
    defs = _defs_in_source(p.read_text())
    lines = p.read_text().splitlines()
    assert _named_anchor_ok(name, defs, lines, start, end), (
        f"`{name}` cited at {path}:{start}-{end}, but that span neither defines "
        f"{name.rsplit('.', 1)[-1]!r} nor calls it"
    )


def test_named_anchor_regex_is_not_vacuous_on_this_docs_own_citation_style() -> None:
    """Mutation proof for `_NAMED_ANCHOR_RE` + the imported `_named_anchor_ok`, using a
    synthetic snippet in THIS doc's own table-cell citation convention -- a bare
    `` `name` `` with no `()`, separated from its anchor by a comma and an exception-
    class citation, exactly the shape that let the real `licence_tier` citation drift
    for two commits with nothing in the suite catching it. Never touches the real doc
    or the real source file."""
    fixture_source = (
        "def licence_tier(region):\n"
        "    return 'mit'\n"
        "\n"
        "\n"
        "def unrelated_helper(local, remote):\n"
        "    return local\n"
    )
    fixture_doc = (
        "| unaudited region | `PublishAbortError` (`licence_tier`, `scripts/fake.py:1-2`) | ... |\n"
    )

    named = _NAMED_ANCHOR_RE.findall(fixture_doc)
    assert named, "fixture regex found nothing -- this test would pass vacuously"
    name, _path, start_s, end_s = named[0]
    assert name == "licence_tier"

    defs = _defs_in_source(fixture_source)
    lines = fixture_source.splitlines()
    start, end = int(start_s), int(end_s) if end_s else int(start_s)

    # Sanity: the fixture resolves correctly when cited at its own (correct) lines.
    assert _named_anchor_ok(name, defs, lines, start, end)

    # The mutation: cite `unrelated_helper`'s lines (5-6) for `licence_tier`, exactly the
    # real defect's shape (a real, long-enough span pointing at the wrong function).
    assert not _named_anchor_ok(name, defs, lines, 5, 6), (
        "mutation proof is broken: a wrong-function citation passed _named_anchor_ok"
    )
