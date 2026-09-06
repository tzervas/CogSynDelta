"""W0's per-region parameter table: `cogsyndelta.faculty.param_table`.

Two kinds of test here:
  1. Pure unit tests against small scratch modules (no checkpoint, no host
     dependency) -- these run everywhere and pin the counting rules themselves.
  2. Re-instantiation against the REAL matrix checkpoints under
     `/akula-data/csd/matrix/` -- read-only, CPU, skipped cleanly when a checkpoint is
     not present on the host running the suite (the row's own instruction: "if a
     checkpoint cannot be loaded on CPU say so").
"""

from __future__ import annotations

from pathlib import Path

import pytest
from torch import nn

pytest.importorskip("tokenizers", reason="train group not installed")

from cogsyndelta.faculty.param_table import (
    CANONICAL_CHECKPOINTS,
    RegionCheckpoint,
    build_matrix_param_table,
    count_text_encoder_params,
    count_visual_encoder_params,
    faculty_param_count,
    format_param_table,
    load_region_module,
)
from cogsyndelta.model.vl_jepa import IJEPA, JEPAConfig, ViTEncoder
from cogsyndelta.regions.text_encoder import TextEncoder, TextEncoderConfig

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------------
# 1. Unit tests on scratch modules.
# ---------------------------------------------------------------------------------


def test_count_text_encoder_params_splits_correctly_with_identity_proj() -> None:
    cfg = TextEncoderConfig(vocab_size=200, dim=32, depth=2, n_heads=4, max_len=32, out_dim=None)
    model = TextEncoder(cfg)
    embedding, token_path, pooling_head = count_text_encoder_params(model)

    expected_embedding = 200 * 32
    assert embedding == expected_embedding
    assert pooling_head == 0  # out_dim=None -> proj is nn.Identity
    assert isinstance(model.proj, nn.Identity)
    total = sum(p.numel() for p in model.parameters())
    assert embedding + token_path + pooling_head == total


def test_count_text_encoder_params_counts_a_real_pooling_head() -> None:
    """`out_dim != dim` builds a real `nn.Linear` `proj` -- the one production config
    (`out_dim=None` everywhere today) never exercises this branch, so it is worth
    covering on a scratch config that deliberately does."""
    cfg = TextEncoderConfig(vocab_size=50, dim=16, depth=1, n_heads=2, max_len=8, out_dim=24)
    model = TextEncoder(cfg)
    assert isinstance(model.proj, nn.Linear)
    embedding, token_path, pooling_head = count_text_encoder_params(model)
    assert pooling_head == 16 * 24  # bias=False on this proj
    total = sum(p.numel() for p in model.parameters())
    assert embedding + token_path + pooling_head == total
    assert pooling_head > 0


def test_count_visual_encoder_params_splits_correctly() -> None:
    cfg = JEPAConfig(
        image_size=32, patch_size=8, dim=48, depth=2, n_heads=4, predictor_dim=24, predictor_depth=2
    )
    encoder = ViTEncoder(cfg)
    embedding, token_path, pooling_head = count_visual_encoder_params(encoder)

    expected_embedding = sum(p.numel() for p in encoder.patch_embed.parameters())
    assert embedding == expected_embedding
    assert pooling_head == 0  # ViTEncoder.pool has no proj
    total = sum(p.numel() for p in encoder.parameters())
    assert embedding + token_path + pooling_head == total


def test_faculty_param_count_unwraps_ijepa_to_its_target_encoder() -> None:
    """`faculty_param_count` on a bare `IJEPA` must count `target_encoder` (DEC-34's
    deployed half), not double-count the context encoder or include the predictor
    (a training-only component with no `Faculty` role)."""
    cfg = JEPAConfig(
        image_size=32, patch_size=8, dim=48, depth=2, n_heads=4, predictor_dim=24, predictor_depth=2
    )
    model = IJEPA(cfg)
    via_ijepa = faculty_param_count("visual", "visual", model)
    via_target = faculty_param_count("visual", "visual", model.target_encoder)
    assert via_ijepa == via_target

    total_ijepa_params = sum(p.numel() for p in model.parameters())
    assert via_ijepa.total < total_ijepa_params, (
        "counting the whole IJEPA (encoder + target_encoder + predictor) would over-"
        "count the deployed region's real parameter footprint"
    )


def test_faculty_param_count_rejects_a_kind_module_mismatch() -> None:
    cfg = TextEncoderConfig(vocab_size=10, dim=8, depth=1, n_heads=2, max_len=4)
    model = TextEncoder(cfg)
    with pytest.raises(TypeError):
        faculty_param_count("language", "visual", model)


def test_format_param_table_reports_a_missing_checkpoint_without_raising() -> None:
    missing = RegionCheckpoint(
        region="ghost", legacy="ghost", checkpoint=Path("/nonexistent/x.pt"), kind="text"
    )
    rows = build_matrix_param_table(entries=(missing,))
    assert len(rows) == 1
    assert rows[0].counts is None
    assert "not present on this host" in rows[0].error
    rendered = format_param_table(rows)
    assert "COULD NOT LOAD" in rendered


# ---------------------------------------------------------------------------------
# 2. Re-instantiation against the real matrix checkpoints (read-only, CPU).
# ---------------------------------------------------------------------------------


def test_canonical_checkpoints_cover_the_five_trained_regions() -> None:
    names = {entry.region for entry in CANONICAL_CHECKPOINTS}
    assert names == {"language", "compress", "retrieve", "reason", "visual"}


@pytest.mark.parametrize("entry", CANONICAL_CHECKPOINTS, ids=lambda e: e.region)
def test_matrix_checkpoint_loads_and_counts_or_reports_why_not(entry: RegionCheckpoint) -> None:
    """Per the row's own instruction: if a checkpoint cannot be loaded on CPU, say so
    (skip, do not fail the suite for a checkpoint absent on this host)."""
    if not entry.checkpoint.is_file():
        pytest.skip(f"{entry.checkpoint} not present on this host")
    try:
        module, _ckpt = load_region_module(entry, map_location="cpu")
    except Exception as exc:  # report, matching build_matrix_param_table's own handling
        pytest.fail(f"{entry.region}: checkpoint present but could not be loaded on CPU: {exc!r}")
        return
    counts = faculty_param_count(entry.region, entry.kind, module, checkpoint=str(entry.checkpoint))
    assert counts.total > 0
    assert counts.embedding >= 0
    assert counts.token_path >= 0
    assert counts.pooling_head >= 0
    assert counts.total == counts.embedding + counts.token_path + counts.pooling_head


def test_text_region_matrix_checkpoints_match_the_taxonomy_calibration_figure() -> None:
    """§2.3's calibration paragraph: a text region is 16,021,248 params, of which
    12,865,792 (80.30%) is the embedding table and 3,155,456 (19.70%) is token path.
    Every text-region matrix checkpoint (dim=256, depth=4, vocab=50257) should
    reproduce it exactly -- this is the cross-check that the counting rule and the
    hand-derived doc figure agree, not merely that the code runs."""
    rows = build_matrix_param_table(
        entries=tuple(e for e in CANONICAL_CHECKPOINTS if e.kind == "text")
    )
    present = [r for r in rows if r.counts is not None]
    if not present:
        pytest.skip("no text-region matrix checkpoints present on this host")
    for row in present:
        c = row.counts
        assert c.total == 16_021_248, f"{row.entry.region}: total {c.total}"
        assert c.embedding == 12_865_792, f"{row.entry.region}: embedding {c.embedding}"
        assert c.token_path == 3_155_456, f"{row.entry.region}: token_path {c.token_path}"
        assert c.pooling_head == 0, f"{row.entry.region}: pooling_head {c.pooling_head}"


def test_visual_matrix_checkpoint_matches_the_taxonomy_deployed_half_figure() -> None:
    """§1.4's catalogue: *"DEPLOYED HALF IS THE EMA TARGET ENCODER (10,712,448
    params)"* -- DEC-34."""
    rows = build_matrix_param_table(
        entries=tuple(e for e in CANONICAL_CHECKPOINTS if e.kind == "visual")
    )
    present = [r for r in rows if r.counts is not None]
    if not present:
        pytest.skip("no visual matrix checkpoint present on this host")
    for row in present:
        assert row.counts.total == 10_712_448, f"{row.entry.region}: total {row.counts.total}"


def test_print_the_full_matrix_param_table() -> None:
    """Not an assertion-bearing test -- this IS the row's 'a small script or test that
    prints the table' deliverable, run under pytest -s (or read from the log either
    way) so the table lands in the test's own output without a second script needing
    to be kept in sync with the loader logic above."""
    rows = build_matrix_param_table()
    table = format_param_table(rows)
    print("\n" + table)
    assert len(rows) == len(CANONICAL_CHECKPOINTS)
