"""Deterministic composite-frame layout engine and renderer (W3r).

WHY THIS EXISTS
Revision 2 had a dependency cycle: W7v's gate consumed a rendered composite (W3's
deliverable), while W3 was `blocked_by: W7v`. Neither row could start. This module is
the W3-independent, W7v-independent prerequisite both of them need: it fixes the
composite frame geometry once, and ships a minimal deterministic renderer that produces
that geometry byte-for-byte from a small declarative spec. See
`docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` row W3r (§4.1) and §5.5(b).

THE GEOMETRY, FIXED
`image_size=128, patch_size=8` ⇒ `n_patches=256`. This is the number §5.5(b)'s
composites render at and the number W7v re-shapes `visual` (`cogsyndelta.model.vl_jepa`)
to ingest. `visual` is configured today at `image_size=64` (`n_patches=64`) — see
`model/vl_jepa.py:52-53, 182, 199` — and physically cannot ingest a 128x128 frame: the
patch grid produced by a 128x128 input does not match the encoder's fixed positional
embedding, and the mismatch raises rather than silently degrading. `FRAME_SIZE`,
`PATCH_SIZE`, `GRID` and `N_PATCHES` below are that single source of truth; W7v imports
them rather than re-deriving them.

WHAT "DETERMINISTIC" MEANS HERE
A composite is built from a `CompositeSpec`: a background colour and a list of
axis-aligned, non-overlapping, in-bounds panels, each carrying a `TileSpec` describing
its pixel content as a pure function of a few scalar parameters (a solid fill, an
axis-aligned two-colour gradient, or a two-colour checkerboard) — never a random draw,
never wall-clock, never a filesystem read of an external asset. Rendering a spec twice,
anywhere, produces bit-identical pixels. That is what makes a composite usable as
ground truth: §5.5(b) needs "exact, free, and generable at any volume", and a renderer
that cannot reproduce its own output is not a ground-truth source.

Real source content (rendered text panels, real images) is deliberately out of scope for
this row — W3r fixes geometry and compositing mechanics; W3 produces volume against this
fixed frame, sourcing its panel content from already-cleared corpora per §5.5(b).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Frame geometry — the single source of truth. W7v imports these rather than
# re-deriving them; do not duplicate these numbers elsewhere.
# ---------------------------------------------------------------------------

FRAME_SIZE: int = 128
"""Composite frame side length in pixels, both dimensions (square frames only)."""

PATCH_SIZE: int = 8
"""Patch side length in pixels — matches `JEPAConfig.patch_size` (`model/vl_jepa.py`)."""

GRID: int = FRAME_SIZE // PATCH_SIZE
"""Patches per side: 16."""

N_PATCHES: int = GRID * GRID
"""Total patches per composite frame: 256."""

_RGB = tuple[int, int, int]

TileKind = Literal["solid", "hgradient", "vgradient", "checker"]


def _u8(value: int, *, name: str) -> int:
    if not 0 <= value <= 255:
        raise ValueError(f"{name} channel {value} out of range [0, 255]")
    return value


def _check_color(color: object, *, name: str) -> _RGB:
    if (
        not isinstance(color, (tuple, list))
        or len(color) != 3
        or not all(isinstance(c, int) for c in color)
    ):
        raise ValueError(f"{name} must be an (r, g, b) triple of ints, got {color!r}")
    r, g, b = color
    return (_u8(r, name=f"{name}.r"), _u8(g, name=f"{name}.g"), _u8(b, name=f"{name}.b"))


@dataclass(frozen=True)
class TileSpec:
    """Deterministic pixel content for one panel.

    `kind="solid"` uses `colors[0]` only. `kind="hgradient"`/`"vgradient"` linearly
    interpolate from `colors[0]` to `colors[1]` along x/y respectively, in integer
    pixel steps (no rounding drift: interpolation is done in integer space so the
    result is exactly reproducible). `kind="checker"` alternates `colors[0]`/
    `colors[1]` in `block`-pixel squares.
    """

    kind: TileKind
    colors: tuple[_RGB, ...]
    block: int = 8

    def __post_init__(self) -> None:
        """Validate shape once, at construction, not at render time."""
        if self.kind == "solid":
            if len(self.colors) != 1:
                raise ValueError("solid tile needs exactly 1 color")
        elif self.kind in ("hgradient", "vgradient", "checker"):
            if len(self.colors) != 2:
                raise ValueError(f"{self.kind} tile needs exactly 2 colors")
        else:
            raise ValueError(f"unknown tile kind {self.kind!r}")
        if self.kind == "checker" and self.block <= 0:
            raise ValueError("checker block must be positive")

    def render(self, width: int, height: int) -> np.ndarray:
        """Produce a deterministic `[height, width, 3]` uint8 array."""
        if self.kind == "solid":
            (c0,) = self.colors
            out = np.empty((height, width, 3), dtype=np.uint8)
            out[:, :] = c0
            return out
        if self.kind == "hgradient":
            return _gradient(width, height, self.colors[0], self.colors[1], axis="x")
        if self.kind == "vgradient":
            return _gradient(width, height, self.colors[0], self.colors[1], axis="y")
        return _checker(width, height, self.colors[0], self.colors[1], self.block)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON-safe shape `from_dict` reads back."""
        return {"kind": self.kind, "colors": [list(c) for c in self.colors], "block": self.block}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TileSpec:
        """Deserialize from the shape `to_dict` produces, validating colors."""
        colors = tuple(_check_color(c, name="tile.colors[]") for c in data["colors"])
        return TileSpec(kind=data["kind"], colors=colors, block=int(data.get("block", 8)))


def _gradient(
    width: int, height: int, c0: _RGB, c1: _RGB, *, axis: Literal["x", "y"]
) -> np.ndarray:
    """Integer-exact linear interpolation along one axis — no float rounding drift."""
    n = width if axis == "x" else height
    steps = np.arange(n, dtype=np.int64)
    denom = max(n - 1, 1)
    out = np.empty((height, width, 3), dtype=np.uint8)
    for ch in range(3):
        # Integer division truncates identically on every platform/run: deterministic.
        ramp = (c0[ch] * (denom - steps) + c1[ch] * steps) // denom
        ramp8 = ramp.astype(np.uint8)
        if axis == "x":
            out[:, :, ch] = ramp8[None, :]
        else:
            out[:, :, ch] = ramp8[:, None]
    return out


def _checker(width: int, height: int, c0: _RGB, c1: _RGB, block: int) -> np.ndarray:
    xs = (np.arange(width) // block) % 2
    ys = (np.arange(height) // block) % 2
    parity = xs[None, :] ^ ys[:, None]  # [height, width], 0 or 1
    out = np.empty((height, width, 3), dtype=np.uint8)
    color0 = np.array(c0, dtype=np.uint8)
    color1 = np.array(c1, dtype=np.uint8)
    out[parity == 0] = color0
    out[parity == 1] = color1
    return out


@dataclass(frozen=True)
class PanelSpec:
    """One axis-aligned panel placed on the frame."""

    id: str
    x: int
    y: int
    width: int
    height: int
    tile: TileSpec

    def __post_init__(self) -> None:
        """Reject panels with non-positive extent or negative placement."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"panel {self.id!r} must have positive width/height")
        if self.x < 0 or self.y < 0:
            raise ValueError(f"panel {self.id!r} must have non-negative x/y")

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """`(x0, y0, x1, y1)`, `x1`/`y1` exclusive."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON-safe shape `from_dict` reads back."""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "tile": self.tile.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PanelSpec:
        """Deserialize from the shape `to_dict` produces."""
        return PanelSpec(
            id=str(data["id"]),
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
            tile=TileSpec.from_dict(data["tile"]),
        )


@dataclass(frozen=True)
class CompositeSpec:
    """A full composite-frame layout description — the thing checked into fixtures."""

    frame_size: int = FRAME_SIZE
    patch_size: int = PATCH_SIZE
    background: _RGB = (255, 255, 255)
    panels: tuple[PanelSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate the layout once at construction: this is the layout engine's checks.

        In-bounds and non-overlapping panels are what make the renderer deterministic
        regardless of panel order — see `render_composite`.
        """
        if self.frame_size <= 0:
            raise ValueError("frame_size must be positive")
        if self.frame_size % self.patch_size != 0:
            raise ValueError(
                f"frame_size {self.frame_size} must be divisible by patch_size {self.patch_size}"
            )
        _check_color(self.background, name="background")
        ids = [p.id for p in self.panels]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate panel ids: {ids}")
        for panel in self.panels:
            x0, y0, x1, y1 = panel.bounds
            if x1 > self.frame_size or y1 > self.frame_size:
                raise ValueError(
                    f"panel {panel.id!r} bounds {panel.bounds} exceed frame_size {self.frame_size}"
                )
        for i, a in enumerate(self.panels):
            for b in self.panels[i + 1 :]:
                if _overlaps(a.bounds, b.bounds):
                    raise ValueError(f"panels {a.id!r} and {b.id!r} overlap")

    @property
    def n_patches(self) -> int:
        """Total patches per frame at this spec's geometry."""
        grid = self.frame_size // self.patch_size
        return grid * grid

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON layout description `from_dict`/`load_spec` reads back."""
        return {
            "frame_size": self.frame_size,
            "patch_size": self.patch_size,
            "background": list(self.background),
            "panels": [p.to_dict() for p in self.panels],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CompositeSpec:
        """Deserialize from the shape `to_dict`/`save_spec` produces."""
        background = _check_color(tuple(data.get("background", [255, 255, 255])), name="background")
        panels = tuple(PanelSpec.from_dict(p) for p in data.get("panels", []))
        return CompositeSpec(
            frame_size=int(data.get("frame_size", FRAME_SIZE)),
            patch_size=int(data.get("patch_size", PATCH_SIZE)),
            background=background,
            panels=panels,
        )


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def load_spec(path: str | Path) -> CompositeSpec:
    """Load a `CompositeSpec` from a JSON layout description."""
    with Path(path).open(encoding="utf-8") as fh:
        return CompositeSpec.from_dict(json.load(fh))


def save_spec(spec: CompositeSpec, path: str | Path) -> None:
    """Write a `CompositeSpec` as a JSON layout description."""
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(spec.to_dict(), fh, indent=2, sort_keys=True)
        fh.write("\n")


def render_composite(spec: CompositeSpec) -> np.ndarray:
    """Render `spec` deterministically to a `[frame_size, frame_size, 3]` uint8 array.

    Panels are drawn in list order (later panels win on overlap), but `CompositeSpec`
    already rejects overlapping panels, so draw order never changes the result — the
    output is a pure function of the spec.
    """
    frame = np.empty((spec.frame_size, spec.frame_size, 3), dtype=np.uint8)
    frame[:, :] = np.array(spec.background, dtype=np.uint8)
    for panel in spec.panels:
        tile = panel.tile.render(panel.width, panel.height)
        x0, y0, x1, y1 = panel.bounds
        frame[y0:y1, x0:x1] = tile
    return frame


def frames_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Byte-identical comparison: same shape and same raw pixel bytes.

    Deliberately not a tolerance-based comparison — this module's entire claim is
    exact, bit-for-bit reproducibility, and a fuzzy comparison would hide the class of
    regression W3r's gate exists to catch.
    """
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def render_to_array(spec_path: str | Path) -> np.ndarray:
    """Load a spec from `spec_path` and render it. Convenience for the CLI and tests."""
    return render_composite(load_spec(spec_path))


def load_frame_png(path: str | Path) -> np.ndarray:
    """Decode a PNG fixture to a `[H, W, 3]` uint8 array (drops any alpha channel)."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def save_frame_png(frame: np.ndarray, path: str | Path) -> None:
    """Encode a `[H, W, 3]` uint8 array to PNG. No metadata chunks are written."""
    Image.fromarray(frame, mode="RGB").save(path, format="PNG")


def _cli(argv: list[str] | None = None) -> int:
    """`python -m cogsyndelta.vl.composite <spec.json> <out.png>` — render a spec to PNG."""
    import argparse

    parser = argparse.ArgumentParser(description="Render a CompositeSpec JSON to a PNG frame.")
    parser.add_argument("spec", type=Path, help="path to a CompositeSpec JSON layout description")
    parser.add_argument("out", type=Path, help="path to write the rendered PNG frame")
    args = parser.parse_args(argv)
    save_frame_png(render_to_array(args.spec), args.out)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_cli(sys.argv[1:]))
