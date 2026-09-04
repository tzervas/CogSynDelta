"""W7v-cfg (2): a checkpoint trained at one patch grid must not attach to a config built
for another.

WHY THIS GUARD EXISTS, PRECISELY
`ViTEncoder.pos_embed` is a non-persistent buffer (`register_buffer(...,
persistent=False)`), so it is never part of `state_dict()`. `PatchEmbed`'s `Conv2d` and
every `ViTBlock`'s parameters do not depend on `image_size` at all -- only on
`dim`/`n_heads`/`patch_size`. That means a bare `model.load_state_dict(ckpt["model"])`
SUCCEEDS SILENTLY even when the checkpoint was trained at a different resolution: every
persisted tensor lines up shape-for-shape, while the freshly-constructed `pos_embed`
means something entirely different from the positions the checkpoint's weights were
trained against. `check_checkpoint_grid_compatible` is the explicit pre-check that
closes that hole -- see `g8-visual/S02.md` §9 ("Pitfalls" -- "silent resize masquerade")
and taxonomy row W7v. The fix is a refusal naming both grids, NOT a `_decode_split`
pixel resize, which this file does not add and does not test for.
"""

from __future__ import annotations

import pytest

from cogsyndelta.model.vl_jepa import JEPAConfig, check_checkpoint_grid_compatible

pytestmark = pytest.mark.cpu


def test_matching_grid_is_accepted() -> None:
    cfg = JEPAConfig(image_size=128, patch_size=8, dim=32, n_heads=4)
    check_checkpoint_grid_compatible({"image_size": 128, "patch_size": 8}, cfg)  # no raise


def test_64px_checkpoint_into_128px_config_is_refused_naming_both_grids() -> None:
    cfg = JEPAConfig(image_size=128, patch_size=8, dim=32, n_heads=4)
    with pytest.raises(ValueError) as exc_info:
        check_checkpoint_grid_compatible({"image_size": 64, "patch_size": 8}, cfg)
    message = str(exc_info.value)
    # Both grids named, not just "they differ".
    assert "8x8" in message, message
    assert "16x16" in message, message
    assert "64 patches" in message, message
    assert "256 patches" in message, message
    assert "image_size=64" in message, message
    assert "image_size=128" in message, message


def test_128px_checkpoint_into_64px_config_is_also_refused() -> None:
    """Symmetric to the headline case -- the guard compares grids, not a hardcoded pair."""
    cfg = JEPAConfig(image_size=64, patch_size=8, dim=32, n_heads=4)
    with pytest.raises(ValueError, match="16x16"):
        check_checkpoint_grid_compatible({"image_size": 128, "patch_size": 8}, cfg)


def test_same_grid_different_patch_size_is_accepted() -> None:
    """The guard is about the GRID (patches per side), not the exact (image_size,
    patch_size) pair -- a 64/8 checkpoint (grid 8) and a 32/4 config (grid 8) describe
    the same patch layout."""
    cfg = JEPAConfig(image_size=32, patch_size=4, dim=32, n_heads=4)
    check_checkpoint_grid_compatible({"image_size": 64, "patch_size": 8}, cfg)  # no raise


def test_missing_image_size_is_refused() -> None:
    cfg = JEPAConfig(image_size=128, patch_size=8, dim=32, n_heads=4)
    with pytest.raises(ValueError, match="missing an integer image_size/patch_size"):
        check_checkpoint_grid_compatible({"patch_size": 8}, cfg)


def test_missing_patch_size_is_refused() -> None:
    cfg = JEPAConfig(image_size=128, patch_size=8, dim=32, n_heads=4)
    with pytest.raises(ValueError, match="missing an integer image_size/patch_size"):
        check_checkpoint_grid_compatible({"image_size": 128}, cfg)


def test_non_integer_image_size_is_refused() -> None:
    """A malformed checkpoint config (e.g. a stringly-typed field from a hand-edited
    JSON) must not be compared as if it were a real grid."""
    cfg = JEPAConfig(image_size=128, patch_size=8, dim=32, n_heads=4)
    with pytest.raises(ValueError, match="missing an integer image_size/patch_size"):
        check_checkpoint_grid_compatible({"image_size": "128", "patch_size": 8}, cfg)


def test_indivisible_checkpoint_geometry_is_refused_before_grid_arithmetic() -> None:
    cfg = JEPAConfig(image_size=128, patch_size=8, dim=32, n_heads=4)
    with pytest.raises(ValueError, match="not divisible"):
        check_checkpoint_grid_compatible({"image_size": 65, "patch_size": 8}, cfg)
