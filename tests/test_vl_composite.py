"""W3r acceptance tests: composite frame geometry, deterministic renderer, fixture.

Per `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` §4.1 row W3r, the gate is: the
fixture exists and is tracked; the renderer reproduces it byte-identically from its
layout description; and — verified by making it fail — a one-pixel perturbation breaks
the comparison. A deterministic layout engine that cannot reproduce its own output is
not a ground-truth source, so the negative case matters as much as the positive one:
without it, a comparison that always passes (e.g. one that only checks shape) would
pass unnoticed.

This file also carries the shape-assertion target for W7v: `visual` is configured today
at `image_size=64` (`model/vl_jepa.py:52-53, 182, 199`) and physically cannot ingest a
128x128 composite frame. That must raise, not silently degrade — this test constructs
the failing case and shows it fails, so W7v has a concrete regression guard to keep
green once it re-shapes the encoder.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from cogsyndelta.model.vl_jepa import JEPAConfig, ViTEncoder
from cogsyndelta.vl.composite import (
    FRAME_SIZE,
    GRID,
    N_PATCHES,
    PATCH_SIZE,
    CompositeSpec,
    PanelSpec,
    TileSpec,
    frames_equal,
    load_frame_png,
    load_spec,
    render_composite,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_SPEC = FIXTURE_DIR / "composite_4panel.spec.json"
FIXTURE_PNG = FIXTURE_DIR / "composite_4panel.png"


@pytest.mark.cpu
def test_frame_geometry_constants() -> None:
    """The geometry triple W7v re-shapes the encoder to, and W3 renders against."""
    assert FRAME_SIZE == 128
    assert PATCH_SIZE == 8
    assert GRID == 16
    assert N_PATCHES == 256
    assert N_PATCHES == GRID * GRID


@pytest.mark.cpu
def test_fixture_files_exist_and_are_tracked() -> None:
    assert FIXTURE_SPEC.is_file(), "checked-in layout description is missing"
    assert FIXTURE_PNG.is_file(), "checked-in composite fixture is missing"


@pytest.mark.cpu
def test_fixture_spec_uses_the_fixed_geometry() -> None:
    spec = load_spec(FIXTURE_SPEC)
    assert spec.frame_size == FRAME_SIZE
    assert spec.patch_size == PATCH_SIZE
    assert spec.n_patches == N_PATCHES
    assert len(spec.panels) == 4, "fixture is specified as a four-panel composite"


@pytest.mark.cpu
def test_renderer_reproduces_fixture_byte_identically() -> None:
    """The core acceptance gate: renderer(spec) == checked-in fixture, exactly."""
    spec = load_spec(FIXTURE_SPEC)
    rendered = render_composite(spec)
    fixture = load_frame_png(FIXTURE_PNG)

    assert rendered.shape == (FRAME_SIZE, FRAME_SIZE, 3)
    assert frames_equal(rendered, fixture)
    # frames_equal is the mechanism the rest of this file trusts; also assert directly
    # so a bug in frames_equal itself cannot make this test vacuously pass.
    assert np.array_equal(rendered, fixture)


@pytest.mark.cpu
def test_one_pixel_perturbation_of_the_fixture_fails_the_comparison() -> None:
    """Verify by making it fail: the comparison must be sensitive to a single pixel.

    A comparison that ignores this perturbation (e.g. shape-only, or a tolerant
    comparison) would make the positive test above meaningless.
    """
    spec = load_spec(FIXTURE_SPEC)
    rendered = render_composite(spec)
    fixture = load_frame_png(FIXTURE_PNG)
    assert frames_equal(rendered, fixture)  # sanity: they agree before perturbation

    perturbed = fixture.copy()
    perturbed[0, 0, 0] = np.uint8((int(perturbed[0, 0, 0]) + 1) % 256)

    assert not frames_equal(rendered, perturbed)
    assert not np.array_equal(rendered, perturbed)


@pytest.mark.cpu
def test_one_pixel_shift_in_the_layout_description_fails_the_comparison() -> None:
    """Row W3r's own stated verification: perturb the spec, not the fixture.

    Shifting one panel's position by a single pixel must change the rendered output
    enough that it no longer matches the checked-in fixture — proving the renderer
    is actually sensitive to its layout description rather than, say, always emitting
    a cached image.
    """
    spec = load_spec(FIXTURE_SPEC)
    fixture = load_frame_png(FIXTURE_PNG)

    panels = list(spec.panels)
    shifted_panel = panels[0]
    panels[0] = PanelSpec(
        id=shifted_panel.id,
        x=shifted_panel.x + 1,
        y=shifted_panel.y,
        width=shifted_panel.width - 1,
        height=shifted_panel.height,
        tile=shifted_panel.tile,
    )
    perturbed_spec = CompositeSpec(
        frame_size=spec.frame_size,
        patch_size=spec.patch_size,
        background=spec.background,
        panels=tuple(panels),
    )

    rendered_from_perturbed_spec = render_composite(perturbed_spec)
    assert not frames_equal(rendered_from_perturbed_spec, fixture)


@pytest.mark.cpu
def test_renderer_is_deterministic_across_calls() -> None:
    """Rendering the same spec twice must be bit-identical (no RNG, no wall-clock)."""
    spec = load_spec(FIXTURE_SPEC)
    a = render_composite(spec)
    b = render_composite(copy.deepcopy(spec))
    assert frames_equal(a, b)


@pytest.mark.cpu
def test_layout_engine_rejects_overlapping_panels() -> None:
    """A deterministic layout engine must validate its own placements."""
    with pytest.raises(ValueError, match="overlap"):
        CompositeSpec(
            frame_size=16,
            patch_size=8,
            panels=(
                PanelSpec(
                    id="a", x=0, y=0, width=10, height=10, tile=TileSpec("solid", ((0, 0, 0),))
                ),
                PanelSpec(
                    id="b", x=5, y=5, width=10, height=10, tile=TileSpec("solid", ((1, 1, 1),))
                ),
            ),
        )


@pytest.mark.cpu
def test_layout_engine_rejects_out_of_bounds_panels() -> None:
    with pytest.raises(ValueError, match="exceed frame_size"):
        CompositeSpec(
            frame_size=16,
            patch_size=8,
            panels=(
                PanelSpec(
                    id="a", x=8, y=8, width=16, height=16, tile=TileSpec("solid", ((0, 0, 0),))
                ),
            ),
        )


@pytest.mark.cpu
def test_64x64_visual_checkpoint_config_rejects_a_128x128_frame() -> None:
    """The shape assertion that gives W7v a concrete target.

    `ViTEncoder` at today's `image_size=64` registers a positional embedding sized for
    64 patches (`model/vl_jepa.py:52-53, 182, 199`). A 128x128 composite frame at
    `PATCH_SIZE=8` produces 256 patches, so `patch_embed(x) + pos_embed` is a shape
    error, not a slower forward — the negative test W7v's gate names.
    """
    cfg = JEPAConfig(image_size=64, patch_size=8, dim=32, depth=1, n_heads=4)
    encoder = ViTEncoder(cfg)

    composite_frame = torch.zeros(1, 3, FRAME_SIZE, FRAME_SIZE)
    with pytest.raises(RuntimeError):
        encoder(composite_frame)

    # The checkpoint config ingests its own native resolution without complaint —
    # isolates the failure to the resolution mismatch, not e.g. a broken encoder.
    native_frame = torch.zeros(1, 3, cfg.image_size, cfg.image_size)
    encoder(native_frame)
