"""Unit tests for `scripts/docs_to_wiki.py` -- the docs/ -> Forgejo-wiki mirroring logic.

Covers the two load-bearing pieces call out in the module docstring: the page-name
flattening scheme (docs/<a>/<b>.md -> <a>--<b>.md) and link rewriting (wiki-page /
asset / source-blob classification), plus the end-to-end mirror build (marker on every
page, stale-page pruning gated on the marker, dry-run writes nothing).

No GPU/torch dependency -- `docs_to_wiki.py` is stdlib-only, so this file does not need
the project's torch venv and is safe to run with plain `pytest`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import docs_to_wiki as dtw

pytestmark = pytest.mark.cpu

REPO_URL = "https://git.vectorweight.com/tzervas/CogSynDelta"
BRANCH = "main"
SHA = "abc1234def0"


# ---------------------------------------------------------------------------------------
# Page-name flattening
# ---------------------------------------------------------------------------------------


def test_flatten_docs_path_matches_the_documented_examples() -> None:
    assert (
        dtw.flatten_docs_path("docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md")
        == "design--REGION-TAXONOMY-AND-INTERCONNECT.md"
    )
    assert dtw.flatten_docs_path("docs/plain/01-what-csd-is.md") == "plain--01-what-csd-is.md"


def test_flatten_docs_path_top_level_file_has_no_separator() -> None:
    assert dtw.flatten_docs_path("docs/ARCHITECTURE.md") == "ARCHITECTURE.md"


def test_flatten_docs_path_nested_readmes_stay_distinct() -> None:
    a = dtw.flatten_docs_path("docs/adr/README.md")
    b = dtw.flatten_docs_path("docs/adr/archive/README.md")
    c = dtw.flatten_docs_path("docs/archive/README.md")
    assert len({a, b, c}) == 3
    assert a == "adr--README.md"
    assert b == "adr--archive--README.md"
    assert c == "archive--README.md"


def test_flatten_docs_path_sanitizes_spaces_in_segments() -> None:
    name = dtw.flatten_docs_path("docs/Some Title With Spaces.md")
    assert " " not in name
    assert name == "Some-Title-With-Spaces.md"


def test_flatten_docs_path_rejects_paths_outside_docs() -> None:
    with pytest.raises(ValueError, match="not under docs/"):
        dtw.flatten_docs_path("README.md")


# ---------------------------------------------------------------------------------------
# Link classification / resolution
# ---------------------------------------------------------------------------------------


def _resolve(source: str, target: str) -> tuple[str, str] | None:
    return dtw.classify_and_resolve(source, target, repo_url=REPO_URL, branch=BRANCH)


def test_resolve_sibling_relative_link_within_docs_is_a_wiki_link() -> None:
    # docs/program/CSD-O11Y-TAXONOMY.md linking "../CODEX-OPS.md" -> docs/CODEX-OPS.md
    assert _resolve("docs/program/CSD-O11Y-TAXONOMY.md", "../CODEX-OPS.md") == (
        "CODEX-OPS.md",
        "wiki",
    )


def test_resolve_deeply_nested_relative_link_within_docs() -> None:
    # docs/adr/archive/README.md linking "../0009-x.md" -> docs/adr/0009-x.md
    result = _resolve("docs/adr/archive/README.md", "../0009-supplement.md")
    assert result == ("adr--0009-supplement.md", "wiki")


def test_resolve_bare_filename_link_same_directory() -> None:
    assert _resolve("docs/adr/README.md", "0001-use-uv-package-manager.md") == (
        "adr--0001-use-uv-package-manager.md",
        "wiki",
    )


def test_resolve_link_leaving_docs_becomes_blob_url() -> None:
    # docs/archive/README.md linking "../../CHANGELOG.md" -> repo-root CHANGELOG.md
    result = _resolve("docs/archive/README.md", "../../CHANGELOG.md")
    assert result == (f"{REPO_URL}/src/branch/{BRANCH}/CHANGELOG.md", "blob")


def test_resolve_directory_link_becomes_blob_tree_url_with_trailing_slash() -> None:
    result = _resolve("docs/archive/README.md", "../adr/")
    assert result == (f"{REPO_URL}/src/branch/{BRANCH}/docs/adr/", "blob")


def test_resolve_non_markdown_file_under_docs_becomes_blob_url() -> None:
    result = _resolve("docs/design/DATASET-FACTORY-CATALOGUE.md", "datasets/catalogue.json")
    assert result == (
        f"{REPO_URL}/src/branch/{BRANCH}/docs/design/datasets/catalogue.json",
        "blob",
    )


def test_resolve_image_under_docs_becomes_asset_link() -> None:
    result = _resolve("docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md", "diagrams/overview.png")
    assert result == ("design--diagrams--overview.png", "asset")


def test_resolve_preserves_anchor_on_wiki_links() -> None:
    result = _resolve("docs/program/CSD-O11Y-TAXONOMY.md", "../CODEX-OPS.md#section")
    assert result == ("CODEX-OPS.md#section", "wiki")


def test_resolve_preserves_anchor_on_blob_links() -> None:
    # A source-file line anchor (e.g. "#L83") must survive rewriting too, not just a
    # markdown heading anchor on a wiki-page link.
    result = _resolve("docs/archive/QUALITY_IMPROVEMENTS.md", "../../pyproject.toml#L83")
    assert result == (f"{REPO_URL}/src/branch/{BRANCH}/pyproject.toml#L83", "blob")


def test_resolve_leaves_external_urls_untouched() -> None:
    assert (
        _resolve("docs/adr/0006-google-style-docstrings.md", "https://google.github.io/styleguide/")
        is None
    )


def test_resolve_leaves_mailto_untouched() -> None:
    assert _resolve("docs/CONTRIBUTING.md", "mailto:someone@example.com") is None


def test_resolve_leaves_same_page_anchor_untouched() -> None:
    assert _resolve("docs/ARCHITECTURE.md", "#some-section") is None


def test_resolve_leaves_unresolvable_link_above_repo_root_untouched() -> None:
    # docs/ is already the top-level dir here at repo-root-relative "docs"; walking above
    # the repo root cannot be resolved to any in-repo path.
    assert _resolve("docs/README.md", "../../../outside-repo.md") is None


# ---------------------------------------------------------------------------------------
# rewrite_links: fenced code blocks, images, titles
# ---------------------------------------------------------------------------------------


def test_rewrite_links_skips_fenced_code_blocks() -> None:
    text = "See [link](../CODEX-OPS.md).\n\n```\nSee [link](../CODEX-OPS.md) in a snippet.\n```\n"
    result = dtw.rewrite_links(
        text, "docs/program/CSD-O11Y-TAXONOMY.md", repo_url=REPO_URL, branch=BRANCH
    )
    lines = result.text.splitlines()
    assert "CODEX-OPS.md" in lines[0]
    assert "../CODEX-OPS.md" in lines[3]  # inside the fence, untouched
    assert len(result.samples) == 1


def test_rewrite_links_preserves_link_title() -> None:
    text = '[link](../CODEX-OPS.md "Ops doc")\n'
    result = dtw.rewrite_links(
        text, "docs/program/CSD-O11Y-TAXONOMY.md", repo_url=REPO_URL, branch=BRANCH
    )
    assert result.text == '[link](CODEX-OPS.md "Ops doc")\n'


def test_rewrite_links_rewrites_image_targets() -> None:
    text = "![diagram](diagrams/overview.png)\n"
    result = dtw.rewrite_links(
        text, "docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md", repo_url=REPO_URL, branch=BRANCH
    )
    assert result.text == "![diagram](design--diagrams--overview.png)\n"
    assert result.samples == [("diagrams/overview.png", "design--diagrams--overview.png", "asset")]


# ---------------------------------------------------------------------------------------
# End-to-end mirror build
# ---------------------------------------------------------------------------------------


@pytest.fixture
def sample_docs_tree(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    (docs / "adr").mkdir(parents=True)
    (docs / "design" / "diagrams").mkdir(parents=True)
    (docs / "program").mkdir(parents=True)

    (docs / "ARCHITECTURE.md").write_text(
        "# Architecture\n\nSee [config](CONFIGURATION.md) and [adr](adr/README.md).\n"
    )
    (docs / "CONFIGURATION.md").write_text("# Configuration\n")
    (docs / "adr" / "README.md").write_text(
        "# ADRs\n\n- [0001](0001-first.md)\n- [root](../../CHANGELOG.md)\n"
    )
    (docs / "adr" / "0001-first.md").write_text("# ADR 0001\n")
    (docs / "design" / "REGION-TAXONOMY-AND-INTERCONNECT.md").write_text(
        "# Regions\n\n![overview](diagrams/overview.png)\n\n"
        "See the [catalogue](datasets/catalogue.json).\n"
    )
    (docs / "design" / "diagrams" / "overview.png").write_bytes(b"\x89PNG\r\n\x1a\nfake-bytes")
    (docs / "program" / "GOALS.md").write_text("# Goals\n")

    return docs


def test_build_mirror_generates_expected_page_set(sample_docs_tree: Path) -> None:
    mirror = dtw.build_mirror(sample_docs_tree, sha=SHA, repo_url=REPO_URL, branch=BRANCH)

    expected_pages = {
        "ARCHITECTURE.md",
        "CONFIGURATION.md",
        "adr--README.md",
        "adr--0001-first.md",
        "design--REGION-TAXONOMY-AND-INTERCONNECT.md",
        "program--GOALS.md",
        "Home.md",
        "_Sidebar.md",
    }
    assert set(mirror.pages) == expected_pages
    assert set(mirror.assets) == {"design--diagrams--overview.png"}


def test_build_mirror_marks_every_generated_page(sample_docs_tree: Path) -> None:
    mirror = dtw.build_mirror(sample_docs_tree, sha=SHA, repo_url=REPO_URL, branch=BRANCH)
    for name, content in mirror.pages.items():
        first_line = content.splitlines()[0]
        assert dtw.is_marker_line(first_line), f"{name} missing marker: {first_line!r}"
        assert SHA in first_line


def test_build_mirror_synthesizes_home_when_readme_absent(sample_docs_tree: Path) -> None:
    mirror = dtw.build_mirror(sample_docs_tree, sha=SHA, repo_url=REPO_URL, branch=BRANCH)
    home = mirror.pages["Home.md"]
    assert "## adr" in home
    assert "## design" in home
    assert "## program" in home
    assert "## Top-level" in home
    assert "(adr--0001-first.md)" in home


def test_build_mirror_uses_docs_readme_as_home_when_present(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "README.md").write_text("# Welcome\n\nSee [arch](ARCHITECTURE.md).\n")
    (docs / "ARCHITECTURE.md").write_text("# Architecture\n")

    mirror = dtw.build_mirror(docs, sha=SHA, repo_url=REPO_URL, branch=BRANCH)

    assert "README.md" not in mirror.pages  # the top-level README becomes Home.md, not its own page
    assert "Welcome" in mirror.pages["Home.md"]
    assert "(ARCHITECTURE.md)" in mirror.pages["Home.md"]


def test_build_mirror_rewrites_image_and_json_and_offrepo_links(sample_docs_tree: Path) -> None:
    mirror = dtw.build_mirror(sample_docs_tree, sha=SHA, repo_url=REPO_URL, branch=BRANCH)
    region_page = mirror.pages["design--REGION-TAXONOMY-AND-INTERCONNECT.md"]
    assert "(design--diagrams--overview.png)" in region_page
    assert f"({REPO_URL}/src/branch/{BRANCH}/docs/design/datasets/catalogue.json)" in region_page

    adr_readme = mirror.pages["adr--README.md"]
    assert "(adr--0001-first.md)" in adr_readme
    assert f"({REPO_URL}/src/branch/{BRANCH}/CHANGELOG.md)" in adr_readme


def test_build_mirror_reports_no_broken_links_for_a_consistent_tree(sample_docs_tree: Path) -> None:
    mirror = dtw.build_mirror(sample_docs_tree, sha=SHA, repo_url=REPO_URL, branch=BRANCH)
    assert mirror.broken_wiki_links == []


def test_build_mirror_reports_broken_wiki_link_for_dangling_target(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ARCHITECTURE.md").write_text("[missing](DOES-NOT-EXIST.md)\n")

    mirror = dtw.build_mirror(docs, sha=SHA, repo_url=REPO_URL, branch=BRANCH)

    assert mirror.broken_wiki_links == [("docs/ARCHITECTURE.md", "DOES-NOT-EXIST.md")]


# ---------------------------------------------------------------------------------------
# Writing, dry-run, and stale-page pruning
# ---------------------------------------------------------------------------------------


def test_write_mirror_creates_pages_and_assets(sample_docs_tree: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "wiki-out"
    mirror = dtw.build_mirror(sample_docs_tree, sha=SHA, repo_url=REPO_URL, branch=BRANCH)

    dtw.write_mirror(out_dir, mirror)

    assert (out_dir / "Home.md").is_file()
    assert (out_dir / "design--diagrams--overview.png").read_bytes().startswith(b"\x89PNG")
    assert len(list(out_dir.glob("*.md"))) == len(mirror.pages)


def test_dry_run_writes_nothing(sample_docs_tree: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "wiki-out-dry"
    rc = dtw.main(
        ["--docs-dir", str(sample_docs_tree), "--out", str(out_dir), "--sha", SHA, "--dry-run"]
    )
    assert rc == 0
    assert not out_dir.exists()


def test_main_writes_expected_files(sample_docs_tree: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "wiki-out-real"
    rc = dtw.main(["--docs-dir", str(sample_docs_tree), "--out", str(out_dir), "--sha", SHA])
    assert rc == 0
    assert (out_dir / "ARCHITECTURE.md").is_file()
    assert (out_dir / "Home.md").is_file()


def test_prune_stale_removes_only_marked_generated_pages(tmp_path: Path) -> None:
    out_dir = tmp_path / "wiki-out"
    out_dir.mkdir()
    marked_stale = out_dir / "design--REMOVED-PAGE.md"
    marked_stale.write_text(dtw.marker_line("oldsha000000") + "\n\nGone now.\n")
    hand_written = out_dir / "Hand-Written-Page.md"
    hand_written.write_text("# A page a human wrote directly in the wiki UI\n")

    removed = dtw.prune_stale(out_dir, generated_names={"Home.md"}, dry_run=False)

    assert removed == ["design--REMOVED-PAGE.md"]
    assert not marked_stale.exists()
    assert hand_written.exists()  # never touched -- no marker


def test_prune_stale_dry_run_reports_without_deleting(tmp_path: Path) -> None:
    out_dir = tmp_path / "wiki-out"
    out_dir.mkdir()
    marked_stale = out_dir / "design--REMOVED-PAGE.md"
    marked_stale.write_text(dtw.marker_line("oldsha000000") + "\n\nGone now.\n")

    removed = dtw.prune_stale(out_dir, generated_names={"Home.md"}, dry_run=True)

    assert removed == ["design--REMOVED-PAGE.md"]
    assert marked_stale.exists()  # dry-run: reported, not deleted


def test_prune_stale_leaves_current_generated_pages_alone(tmp_path: Path) -> None:
    out_dir = tmp_path / "wiki-out"
    out_dir.mkdir()
    current = out_dir / "Home.md"
    current.write_text(dtw.marker_line(SHA) + "\n\nStill current.\n")

    removed = dtw.prune_stale(out_dir, generated_names={"Home.md"}, dry_run=False)

    assert removed == []
    assert current.exists()
