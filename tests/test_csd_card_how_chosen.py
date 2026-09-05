"""`csd-card --kind region_main` states how the promoted variant was chosen, or refuses.

The `region_main.md.j2` template prints `region_cfg["how_chosen"]`; nothing in the typed
catalogue (`config/mind/csd-regions.json`, a `RegionSpec`) or the matrix yaml feeds it, so the
CLI takes the text from a file the operator files under `docs/design/evidence/`. A promoted
card without it used to render "(not recorded -- set region_cfg['how_chosen'] ...)"; the
CLI now fails closed instead.
"""

from __future__ import annotations

from pathlib import Path

from tests.test_csd_card_cli import _write_cell, load_cli

cli = load_cli()

HOW = "main_rule best_primary_metric_all_gates_pass: seed 1 (0.7237) over seed 0 (0.7224)."


def test_region_main_without_how_chosen_refuses_and_writes_nothing(
    tmp_path: Path, capsys: object
) -> None:
    cell_dir = _write_cell(tmp_path, region="compress")
    out = tmp_path / "out" / "README.md"
    rc = cli.main(["--cell", str(cell_dir), "--kind", "region_main", "--out", str(out)])
    assert rc == 2
    assert not out.exists()
    err = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "--how-chosen-file" in err


def test_region_main_with_how_chosen_file_prints_the_selection(tmp_path: Path) -> None:
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
    assert "## How this was chosen" in card
    assert HOW in card
    # The template's own placeholder must be gone; other "(not recorded)" strings on the
    # fixture card (e.g. the v1 metrics schema line) are unrelated to this option.
    assert "set region_cfg['how_chosen']" not in card


def test_empty_how_chosen_file_refuses(tmp_path: Path) -> None:
    cell_dir = _write_cell(tmp_path, region="compress")
    how = tmp_path / "HOW-CHOSEN.md"
    how.write_text("  \n")
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
    assert rc == 2
    assert not out.exists()


def test_how_chosen_file_is_refused_on_a_variant_card(tmp_path: Path) -> None:
    cell_dir = _write_cell(tmp_path, region="compress")
    how = tmp_path / "HOW-CHOSEN.md"
    how.write_text(HOW + "\n")
    out = tmp_path / "out" / "README.md"
    rc = cli.main(
        [
            "--cell",
            str(cell_dir),
            "--kind",
            "region_variant",
            "--how-chosen-file",
            str(how),
            "--out",
            str(out),
        ]
    )
    assert rc == 2
    assert not out.exists()
