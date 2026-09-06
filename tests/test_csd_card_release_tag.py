"""`csd-card --release-tag` sets `region_cfg["release_tag"]` for the card templates.

The `region_main` template already prints `release_tag` (header + bibtex note). Without
`--release-tag` the render path keeps its default `(not tagged)`; with the flag the
operator-supplied tag string is what the README carries. Unlike `--how-chosen-file`,
this flag is optional and is not kind-restricted.
"""

from __future__ import annotations

from pathlib import Path

from tests.test_csd_card_cli import _write_cell, load_cli
from tests.test_csd_card_how_chosen import HOW

cli = load_cli()


def test_region_main_with_release_tag_prints_the_tag(tmp_path: Path) -> None:
    cell_dir = _write_cell(tmp_path, region="compress")
    how = tmp_path / "HOW-CHOSEN.md"
    how.write_text(HOW + "\n")
    out = tmp_path / "out" / "README.md"
    rc = cli.main(
        [
            "--cell",
            str(cell_dir),
            "--kind",
            "region_main",
            "--how-chosen-file",
            str(how),
            "--release-tag",
            "v0.1.0",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    card = out.read_text()
    assert "v0.1.0" in card
    assert "(not tagged)" not in card


def test_region_main_without_release_tag_keeps_not_tagged(tmp_path: Path) -> None:
    cell_dir = _write_cell(tmp_path, region="compress")
    how = tmp_path / "HOW-CHOSEN.md"
    how.write_text(HOW + "\n")
    out = tmp_path / "out" / "README.md"
    rc = cli.main(
        [
            "--cell",
            str(cell_dir),
            "--kind",
            "region_main",
            "--how-chosen-file",
            str(how),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    card = out.read_text()
    assert "(not tagged)" in card
    assert "v0.1.0" not in card
