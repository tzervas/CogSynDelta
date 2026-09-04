"""W7v-cfg (1): `JEPAConfig` derives its grid from W3r, and positions are 2-D sincos.

`docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` row W7v (~:2533); g8 sector
`g8-visual/S02.md` §2 item 1, §5.2, §7. The two failure modes this file guards:

  1. The default `JEPAConfig` silently staying at the old 64px/64-patch geometry, which
     cannot ingest a W3r composite frame at all (a shape error, not a slower forward --
     covered separately by `tests/test_vl_composite.py`'s existing
     `test_64x64_visual_checkpoint_config_rejects_a_128x128_frame`, which this file does
     not touch).
  2. A position table that LOOKS 2-D but is actually the old 1-D raster reused at a
     larger length -- the category error `S02.md` names: a 1-D table wraps a row's last
     column into the next row's first, so "it has more entries now" is not "it is 2-D".
"""

from __future__ import annotations

import pytest
import torch

from cogsyndelta.model.vl_jepa import JEPAConfig, ViTEncoder, sincos2d_pos_embed
from cogsyndelta.vl.composite import FRAME_SIZE, N_PATCHES, PATCH_SIZE

pytestmark = pytest.mark.cpu


# ---------------------------------------------------------------------------------------
# The default config renders W3r's grid.
# ---------------------------------------------------------------------------------------


def test_default_config_matches_w3r_geometry() -> None:
    cfg = JEPAConfig()
    assert cfg.image_size == FRAME_SIZE == 128
    assert cfg.patch_size == PATCH_SIZE == 8
    assert cfg.grid == 16
    assert cfg.n_patches == N_PATCHES == 256


def test_default_config_encoder_pos_embed_is_256_patches() -> None:
    cfg = JEPAConfig()
    encoder = ViTEncoder(cfg)
    assert encoder.pos_embed.shape == (1, 256, 384)


def test_explicit_64px_config_still_works() -> None:
    """A 64-px config is still legal when requested explicitly -- only the DEFAULT moved."""
    cfg = JEPAConfig(image_size=64, patch_size=8, dim=32, depth=1, n_heads=4)
    assert cfg.grid == 8
    assert cfg.n_patches == 64
    encoder = ViTEncoder(cfg)
    assert encoder.pos_embed.shape == (1, 64, 32)
    out = encoder(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, 64, 32)


# ---------------------------------------------------------------------------------------
# The positional table is genuinely 2-D, not a 1-D raster reused at a new length.
# ---------------------------------------------------------------------------------------


def test_sincos2d_is_recomputed_not_a_1d_raster() -> None:
    """Patches in the same ROW must share identical row-encoding channels, and patches
    in the same COLUMN must share identical column-encoding channels -- the property a
    1-D `arange(N)` table cannot have (every index there gets a distinct angle)."""
    grid, dim = 4, 8
    table = sincos2d_pos_embed(grid, dim)
    assert table.shape == (1, grid * grid, dim)
    half = dim // 2
    row_half, col_half = table[0, :, :half], table[0, :, half:]

    # Row 0 is patch indices 0..3 (row-major: index = row*grid + col). Every patch in
    # that row must carry the SAME row-encoding, because they share a row coordinate.
    row0 = row_half[0:grid]
    assert torch.allclose(row0, row0[0].expand_as(row0)), "row-half must be constant within a row"

    # Column 0 is patch indices 0, 4, 8, 12 (one per row, same column). Every patch in
    # that column must carry the SAME column-encoding.
    col0 = col_half[0::grid]
    assert torch.allclose(col0, col0[0].expand_as(col0)), (
        "col-half must be constant within a column"
    )

    # Different rows must differ in the row-half (row 0 vs row 1, same column 0).
    assert not torch.allclose(row_half[0], row_half[grid]), "row-half must vary across rows"
    # Different columns must differ in the col-half (row 0: column 0 vs column 1).
    assert not torch.allclose(col_half[0], col_half[1]), "col-half must vary across columns"


def test_sincos2d_recomputes_fresh_at_every_grid_no_shared_state() -> None:
    """Two different grids produce independently-shaped, independently-valued tables --
    confirms this is a closed-form recomputation, never a resize of a cached table."""
    small = sincos2d_pos_embed(8, 32)
    large = sincos2d_pos_embed(16, 32)
    assert small.shape == (1, 64, 32)
    assert large.shape == (1, 256, 32)
    # The first grid's own first row (patch 0..7) must be identical whether grid=8 or
    # grid=16, because "patch (row=0, col=c)" means the same thing in both -- recomputing
    # from (row, col) is grid-size-independent per patch, unlike resampling a stored
    # table would be.
    assert torch.allclose(small[0, :8], large[0, :8])


# ---------------------------------------------------------------------------------------
# Config validation: guards that must actually fire.
# ---------------------------------------------------------------------------------------


def test_dim_not_divisible_by_4_is_rejected() -> None:
    with pytest.raises(ValueError, match="dim % 4"):
        JEPAConfig(dim=10, n_heads=5)


def test_predictor_dim_not_divisible_by_4_is_rejected() -> None:
    with pytest.raises(ValueError, match="predictor_dim % 4"):
        JEPAConfig(predictor_dim=10)


def test_unsupported_pos_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported pos_kind"):
        JEPAConfig(pos_kind="sincos1d")


def test_composite_invariant_guard_fires_on_a_broken_invariant(monkeypatch) -> None:
    """Mutation proof: if `composite.py`'s N_PATCHES ever stopped matching
    FRAME_SIZE/PATCH_SIZE, the default `JEPAConfig` must refuse rather than silently
    building the wrong grid. Verified by breaking the invariant and watching it fire."""
    import cogsyndelta.model.vl_jepa as vl_jepa_mod

    # Positive control: unpatched, the default config builds cleanly.
    JEPAConfig()

    monkeypatch.setattr(vl_jepa_mod, "N_PATCHES", 999)
    with pytest.raises(ValueError, match=r"composite\.py invariant broken"):
        JEPAConfig()
