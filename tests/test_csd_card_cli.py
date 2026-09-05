"""Tests for `scripts/csd-card.py`.

Covers:
- `--cell` reads `cell.json` + follows its `stages.*.receipt` pointers, picking up
  train/eval/eval-quantized/quant by `state: "done"`; a stage not yet done is skipped,
  not treated as an error (a train-only cell still renders an fp32-only card).
- Explicit `--train-receipt`/`--eval-receipt`/`--eval-quantized-receipt`/
  `--quant-receipt` name receipts directly, and OVERRIDE a `--cell`'s own receipt for
  the same slot when both are given.
- `--region` cross-checked against `--cell`'s own region (mutation proof: matching
  region succeeds, mismatched region refuses).
- `--kind` requires at least a training receipt except for `placeholder`/`composed`.
- `--comparators <table.json>` loads both path-valued and inline-object entries, and
  refuses a value of any other JSON type.
- The licence table is IMPORTED from `scripts/csd-publish-checkpoint.py`, not
  duplicated: an unaudited region (no tier) and `vl_latent` (BLOCKING) both refuse for
  `--kind region_variant`, exactly as a real publish would -- but `--kind placeholder`
  renders the SAME BLOCKING region's placeholder card without refusing (CARD SPEC's own
  contract: a placeholder needs no resolved tier). Visual Mix B (`visual-clean-v1` +
  the pinned fingerprint) resolves to `mit` because `_region_cfg` passes the train
  receipt into `licence_tier`; an altered fingerprint still BLOCKS.
- `--out` writes the rendered card to disk; the parent directory is created if needed.
- Exit code: 0 on success, 2 with an `ABORT:` message on any refusal.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.test_publish_checkpoint import (
    _mix_b_receipt,
    make_checkpoint,
    make_eval_receipt,
    make_quant_receipt,
    make_training_receipt,
)

pytestmark = pytest.mark.cpu

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "csd-card.py"


def load_cli() -> Any:
    spec = importlib.util.spec_from_file_location("csd_card_cli_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["csd_card_cli_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


cli = load_cli()


# --------------------------------------------------------------------------- fixtures


def _write_cell(
    tmp_path: Path,
    region: str = "compress",
    *,
    train: bool = True,
    test: bool = True,
    quantize: bool = True,
    test_quant: bool = True,
) -> Path:
    """Build a minimal, realistic `<cell_dir>/cell.json` + real receipts on disk, in
    the shape `model-matrix` actually writes (see a real example under
    `/akula-data/csd/matrix/<cell_id>/cell.json`) -- `stages.<name>.receipt` as an
    ABSOLUTE path, `state: "done"` only for the stages this call asked to include.
    """
    cell_dir = tmp_path / f"{region}-b512-s0-abc1234-20260904"
    cell_dir.mkdir()
    receipts_dir = cell_dir / "receipts"
    receipts_dir.mkdir()
    checkpoint = make_checkpoint(receipts_dir)

    stages: dict[str, dict[str, Any]] = {}
    if train:
        train_path = make_training_receipt(receipts_dir, checkpoint, region=region)
        stages["train"] = {"state": "done", "receipt": str(train_path)}
    if test:
        eval_path = make_eval_receipt(receipts_dir, checkpoint, region=region)
        stages["test"] = {"state": "done", "receipt": str(eval_path)}
    if quantize:
        quant_path = make_quant_receipt(receipts_dir, checkpoint, region=region)
        stages["quantize"] = {"state": "done", "receipt": str(quant_path)}
    if test_quant:
        # A hand-written eval-shaped receipt stands in for the eval-quantized one --
        # render_card only reads its `metrics`/`gates` dicts, so this is faithful
        # enough for a CLI-level test (the receipt SHAPE is already covered by
        # tests/test_cards_render.py; this file is about the CLI's own plumbing).
        eval_quant_path = receipts_dir / "eval-quant.json"
        eval_quant_path.write_text(
            json.dumps(
                {
                    "producer": {"project": "cogsyndelta", "component": region},
                    "stage": "eval-quantized",
                    "started_utc": "2026-09-04T00:00:00Z",
                    "metrics": {"quant.artifact_recall@1": 0.5},
                    "gates": {},
                    "schema": "model-pipeline-receipt/v1",
                }
            )
        )
        stages["test-quant"] = {"state": "done", "receipt": str(eval_quant_path)}

    cell = {"cell_id": cell_dir.name, "region": region, "stages": stages}
    (cell_dir / "cell.json").write_text(json.dumps(cell))
    return cell_dir


# =====================================================================================
# --cell: reads cell.json, follows stages.*.receipt, respects state == "done".
# =====================================================================================


def test_cell_full_trio_renders(tmp_path: Path) -> None:
    cell_dir = _write_cell(tmp_path, region="compress")
    out = tmp_path / "out" / "README.md"
    rc = cli.main(["--cell", str(cell_dir), "--kind", "region_variant", "--out", str(out)])
    assert rc == 0
    assert out.is_file()
    card = out.read_text()
    assert "## Evaluation results" in card
    assert "## Sizes" in card


def test_cell_train_only_stage_not_done_yields_fp32_card(tmp_path: Path) -> None:
    """A stage the harness never marked `done` (e.g. quantize started but the receipt
    key is absent) is skipped, not treated as a missing-file error."""
    cell_dir = _write_cell(
        tmp_path, region="compress", test=False, quantize=False, test_quant=False
    )
    out = tmp_path / "README.md"
    rc = cli.main(["--cell", str(cell_dir), "--kind", "region_variant", "--out", str(out)])
    assert rc == 0
    card = out.read_text()
    assert "### Training held-out battery" in card
    assert "## Quantization" not in card


def test_cell_stage_present_but_not_done_is_skipped(tmp_path: Path) -> None:
    cell_dir = _write_cell(tmp_path, region="compress")
    cell = json.loads((cell_dir / "cell.json").read_text())
    cell["stages"]["quantize"]["state"] = "running"
    (cell_dir / "cell.json").write_text(json.dumps(cell))

    out = tmp_path / "README.md"
    rc = cli.main(["--cell", str(cell_dir), "--kind", "region_variant", "--out", str(out)])
    assert rc == 0
    # The `quantize` stage's own table (build_quant_table's heading) is gone -- the
    # `test-quant` stage's eval-shaped table is untouched by this mutation (it reads
    # its own, still-"done" receipt), so this checks the SPECIFIC heading, not a
    # blanket "no 'Quantization' anywhere" (which the eval-quantized table would fail).
    assert "Quantization (quant_plan battery)" not in out.read_text()


def test_explicit_receipt_overrides_the_cell_slot(tmp_path: Path) -> None:
    """An explicit --quant-receipt on top of --cell replaces just that one slot."""
    cell_dir = _write_cell(tmp_path, region="compress")
    other_checkpoint = make_checkpoint(tmp_path, name="other.pt")
    other_quant = make_quant_receipt(tmp_path, other_checkpoint, region="compress")

    out = tmp_path / "README.md"
    rc = cli.main(
        [
            "--cell",
            str(cell_dir),
            "--kind",
            "region_variant",
            "--quant-receipt",
            str(other_quant),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.is_file()


# =====================================================================================
# --region cross-check against --cell's own region (mutation proof).
# =====================================================================================


def test_region_matches_cell_succeeds(tmp_path: Path) -> None:
    cell_dir = _write_cell(tmp_path, region="compress")
    out = tmp_path / "README.md"
    rc = cli.main(
        [
            "--cell",
            str(cell_dir),
            "--region",
            "compress",
            "--kind",
            "region_variant",
            "--out",
            str(out),
        ]
    )
    assert rc == 0


def test_region_mismatch_against_cell_aborts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cell_dir = _write_cell(tmp_path, region="compress")
    out = tmp_path / "README.md"
    rc = cli.main(
        [
            "--cell",
            str(cell_dir),
            "--region",
            "retrieve",
            "--kind",
            "region_variant",
            "--out",
            str(out),
        ]
    )
    assert rc == 2
    assert not out.exists()
    err = capsys.readouterr().err
    assert "does not match" in err


# ---------------------------------------------- region rename (naming rule 2026-09-04)


def test_region_code_and_language_both_render_the_same_cell_with_the_alias_shown(
    tmp_path: Path,
) -> None:
    """A cell recorded under the legacy `code` id renders identically whether asked
    for by `--region code` or `--region language` (cogsyndelta.regions.aliases), and
    either way the card shows the faculty name/specialisation the catalogue now
    carries for it."""
    cell_dir = _write_cell(tmp_path, region="code")

    out_legacy = tmp_path / "legacy.md"
    rc_legacy = cli.main(
        [
            "--cell",
            str(cell_dir),
            "--region",
            "code",
            "--kind",
            "region_variant",
            "--out",
            str(out_legacy),
        ]
    )
    out_canonical = tmp_path / "canonical.md"
    rc_canonical = cli.main(
        [
            "--cell",
            str(cell_dir),
            "--region",
            "language",
            "--kind",
            "region_variant",
            "--out",
            str(out_canonical),
        ]
    )

    assert rc_legacy == 0
    assert rc_canonical == 0
    card_legacy = out_legacy.read_text()
    card_canonical = out_canonical.read_text()
    assert card_legacy == card_canonical
    assert "**Faculty:** `language`" in card_legacy
    assert "specialisation: `code`" in card_legacy
    assert "(formerly `code`)" in card_legacy
    assert "region:language" in card_legacy
    assert "legacy region id" not in card_legacy
    # LICENCE_WHY is keyed canonically too -- looking it up by the raw (possibly
    # legacy) region would silently fall back to "(reason not recorded)".
    assert "(reason not recorded)" not in card_legacy
    assert "no NC or share-alike input in the catalogue" in card_legacy


def test_region_defaults_to_the_cells_own_spelling_when_omitted(tmp_path: Path) -> None:
    """No --region at all still works and still shows the faculty/specialisation line
    -- the alias resolution is not gated on the operator spelling it out."""
    cell_dir = _write_cell(tmp_path, region="code")
    out = tmp_path / "README.md"
    rc = cli.main(["--cell", str(cell_dir), "--kind", "region_variant", "--out", str(out)])
    assert rc == 0
    card = out.read_text()
    assert "**Faculty:** `language`" in card


# =====================================================================================
# --kind requires a training receipt, except placeholder/composed.
# =====================================================================================


def test_kind_without_any_receipt_and_no_cell_requires_region(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli.main(["--kind", "region_variant"])
    assert rc == 2
    assert "--region is required" in capsys.readouterr().err


def test_kind_region_variant_needs_a_training_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(
        ["--region", "compress", "--kind", "region_variant", "--out", str(tmp_path / "o.md")]
    )
    assert rc == 2
    assert "training receipt" in capsys.readouterr().err


def test_kind_placeholder_needs_no_receipt_at_all(tmp_path: Path) -> None:
    out = tmp_path / "README.md"
    rc = cli.main(["--region", "stream_vae", "--kind", "placeholder", "--out", str(out)])
    assert rc == 0
    assert "weights: none" in out.read_text()


def test_kind_composed_needs_no_receipt_at_all(tmp_path: Path) -> None:
    """`cogsyndelta` (the composed model's own name) IS an audited row in
    `LICENCE_TIER` (docs/design/LICENCE-FOR-OPEN-WEIGHTS.md's "Decision 2026-09-02"),
    so this CLI's best-effort tier resolution finds a real one -- the card states it,
    rather than the "TBD" placeholder `render_card` prints only when NO tier is on
    record at all (covered directly by tests/test_cards_render.py, which calls
    `render_card` with an empty `region_cfg`)."""
    out = tmp_path / "README.md"
    rc = cli.main(["--region", "cogsyndelta", "--kind", "composed", "--out", str(out)])
    assert rc == 0
    assert "license: cc-by-nc-sa-4.0" in out.read_text()


# =====================================================================================
# --comparators
# =====================================================================================


def test_comparators_path_and_inline_object_both_load(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    eval_path = make_eval_receipt(tmp_path, checkpoint, region="compress")
    # One comparator named by path, one supplied inline -- exercising both accepted
    # shapes in `_load_comparators`.
    comparators_path = tmp_path / "comparators.json"
    comparators_path.write_text(
        json.dumps(
            {
                "from-path": str(eval_path),
                "inline": {"metrics": {"rank.recall@1": 0.42}, "gates": {}},
            }
        )
    )

    out = tmp_path / "README.md"
    rc = cli.main(
        [
            "--region",
            "compress",
            "--kind",
            "region_variant",
            "--train-receipt",
            str(train_path),
            "--eval-receipt",
            str(eval_path),
            "--comparators",
            str(comparators_path),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    card = out.read_text()
    assert "from-path" in card
    assert "inline" in card


def test_comparators_bad_value_type_aborts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    comparators_path = tmp_path / "comparators.json"
    comparators_path.write_text(json.dumps({"bad": 42}))

    out = tmp_path / "README.md"
    rc = cli.main(
        [
            "--region",
            "compress",
            "--kind",
            "region_variant",
            "--train-receipt",
            str(train_path),
            "--comparators",
            str(comparators_path),
            "--out",
            str(out),
        ]
    )
    assert rc == 2
    assert "must be a path string or a JSON object" in capsys.readouterr().err


# =====================================================================================
# Licence table imported from csd-publish-checkpoint.py -- refuses the same way a real
# publish would, EXCEPT placeholder/composed, which render_card allows unresolved.
# =====================================================================================


def test_unaudited_region_refuses_for_region_variant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="residual_mlp")
    out = tmp_path / "README.md"
    rc = cli.main(
        [
            "--region",
            "residual_mlp",
            "--kind",
            "region_variant",
            "--train-receipt",
            str(train_path),
            "--out",
            str(out),
        ]
    )
    assert rc == 2
    assert not out.exists()
    assert "licence tier" in capsys.readouterr().err.lower()


def test_blocking_region_refuses_for_region_variant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="vl_latent")
    out = tmp_path / "README.md"
    rc = cli.main(
        [
            "--region",
            "vl_latent",
            "--kind",
            "region_variant",
            "--train-receipt",
            str(train_path),
            "--out",
            str(out),
        ]
    )
    assert rc == 2
    assert "BLOCKING" in capsys.readouterr().err


def test_blocking_region_still_renders_as_placeholder(tmp_path: Path) -> None:
    """MUTATION-PROOF companion to the refusal above: the SAME BLOCKING region, same
    tool, renders fine under --kind placeholder -- proving the refusal above is about
    `region_variant` needing a resolved tier, not about vl_latent being untouchable."""
    out = tmp_path / "README.md"
    rc = cli.main(["--region", "vl_latent", "--kind", "placeholder", "--out", str(out)])
    assert rc == 0
    assert "weights: none" in out.read_text()


def test_region_cfg_visual_mix_b_resolves_mit() -> None:
    """The publish guard's Mix B pin must reach `licence_tier` through this CLI:
    calling `licence_tier(region)` with no receipt is the default BLOCKING path."""
    pub = cli._load_publish_module()
    rec = _mix_b_receipt()
    cfg = cli._region_cfg(pub, "visual", "region_variant", train_receipt=rec)
    assert cfg["licence_tier"] == "mit"
    cfg_alias = cli._region_cfg(pub, "vl_latent", "region_variant", train_receipt=rec)
    assert cfg_alias["licence_tier"] == "mit"


def test_region_cfg_visual_without_receipt_stays_blocking() -> None:
    pub = cli._load_publish_module()
    with pytest.raises(pub.PublishAbortError, match="BLOCKING"):
        cli._region_cfg(pub, "visual", "region_variant", train_receipt=None)
    with pytest.raises(pub.PublishAbortError, match="BLOCKING"):
        cli._region_cfg(pub, "visual", "region_variant", train_receipt={})


def test_region_cfg_visual_altered_fingerprint_blocks() -> None:
    """Mutation: same Mix B corpus_source, wrong fingerprint, must stay BLOCKING."""
    pub = cli._load_publish_module()
    rec = _mix_b_receipt(fingerprint="0" * 32)
    with pytest.raises(pub.PublishAbortError, match="BLOCKING"):
        cli._region_cfg(pub, "visual", "region_variant", train_receipt=rec)


def test_visual_cli_altered_fingerprint_aborts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """End-to-end: a Mix B-shaped train receipt with a mutated fingerprint must not
    render a region_variant card -- the CLI refuses with BLOCKING, writes nothing."""
    rec = _mix_b_receipt(fingerprint="deadbeef" * 4)
    rec["parameters"] = 22905216
    rec["held_out"] = {"recall@1": 0.0}
    rec["untrained_baseline"] = {"recall@1": 0.0}
    rec["metrics_schema"] = "csd-metrics/v2"
    path = tmp_path / "visual-bad-fp.json"
    path.write_text(json.dumps(rec))
    out = tmp_path / "README.md"
    rc = cli.main(
        [
            "--region",
            "visual",
            "--kind",
            "region_variant",
            "--train-receipt",
            str(path),
            "--out",
            str(out),
        ]
    )
    assert rc == 2
    assert not out.exists()
    assert "BLOCKING" in capsys.readouterr().err


def test_vl_latent_and_visual_placeholders_are_byte_identical_and_name_visual(
    tmp_path: Path,
) -> None:
    """DEC-78: --region vl_latent is an alias; the card must print visual everywhere."""
    out_legacy = tmp_path / "legacy.md"
    out_canon = tmp_path / "canon.md"
    assert (
        cli.main(["--region", "vl_latent", "--kind", "placeholder", "--out", str(out_legacy)]) == 0
    )
    assert cli.main(["--region", "visual", "--kind", "placeholder", "--out", str(out_canon)]) == 0
    a = out_legacy.read_text()
    b = out_canon.read_text()
    assert a == b
    assert "model_name: cogsyndelta-region-visual" in a
    assert "region:visual" in a
    assert "# CogSynDelta -- visual (placeholder)" in a
    assert a.count("vl_latent") == 1
    assert "formerly `vl_latent`" in a


# =====================================================================================
# --out.
# =====================================================================================


def test_out_creates_parent_directories(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    out = tmp_path / "nested" / "dir" / "README.md"
    rc = cli.main(
        [
            "--region",
            "compress",
            "--kind",
            "region_variant",
            "--train-receipt",
            str(train_path),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert out.is_file()


def test_default_out_is_readme_md_in_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    checkpoint = make_checkpoint(tmp_path)
    train_path = make_training_receipt(tmp_path, checkpoint, region="compress")
    rc = cli.main(
        ["--region", "compress", "--kind", "region_variant", "--train-receipt", str(train_path)]
    )
    assert rc == 0
    assert (tmp_path / "README.md").is_file()
