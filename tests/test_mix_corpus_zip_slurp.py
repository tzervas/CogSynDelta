"""`ZipPngReader`'s NFS-small-read fix: slurp small archives, stream large ones --
but ONLY for a reader that closes right after one bounded pass.

WHY THIS FILE EXISTS
R3 (`/akula-data/session-backup-staging/reviews/r3-decode-probe/REPORT.md`) measured
that `zipfile.ZipFile` opened directly on an NFS path pays +1.3s across 15,400 probe
images versus local NVMe, purely from its per-member small `seek`+`read` syscall
pattern -- not disk wait. Reading the whole archive into a `BytesIO` first removes
that penalty with byte-identical output. `ZipPngReader._open` (`vl/mix_corpus.py`)
does that below `_SLURP_MAX_BYTES`, but only when constructed with `slurp=True`, and
streams the path directly otherwise (`slurp=False`, or at/above the threshold either
way).

`slurp` has no default: a reader that lives for a whole training run (`PngTrain`)
never closes or evicts a cached handle, so slurping there would mean every shard under
`_SLURP_MAX_BYTES` staying resident in RAM for the run's entire lifetime. Only
`_decode_png_stores`'s transient reader (closed in a `finally` right after one pass)
passes `slurp=True`. `test_pngtrain_reader_never_slurps` below is the regression guard
for that split; the others cover `ZipPngReader`'s own threshold logic directly.

All tests run entirely on a tiny local fixture zip -- CPU only, no NFS, no GPU --
because the property under test (identical bytes out; correct branch chosen by
`slurp` and size) does not depend on which filesystem the zip lives on.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from cogsyndelta.regions.vl_pretrain import PngTrain, _decode_png_stores
from cogsyndelta.vl import mix_corpus
from cogsyndelta.vl.mix_corpus import ImageRef, ZipPngReader

pytestmark = pytest.mark.cpu


def _png_bytes(color: tuple[int, int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def _fixture_zip(path: Path) -> Path:
    """A small zip with a few distinct, non-uniform PNGs across two class folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = [(10, 20, 30), (200, 50, 5), (0, 255, 128), (77, 77, 200), (128, 0, 0)]
    with zipfile.ZipFile(path, "w") as zf:
        for i, color in enumerate(colors):
            cls = "a" if i % 2 == 0 else "b"
            zf.writestr(f"{cls}/{i:02d}.png", _png_bytes(color))
    return path


def test_decode_is_byte_identical_slurped_vs_streamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finding's core claim: slurp-then-decode and stream-then-decode must match.

    The fixture zip is well under the default `_SLURP_MAX_BYTES`, so the unmodified
    threshold takes the slurp branch; forcing the threshold to 0 takes the stream
    branch (every archive is "too big" to slurp) on the identical bytes. Both must
    produce the exact same tensors -- this is the "keep the decode path byte-identical"
    requirement, not just "both branches run without raising."
    """
    zpath = _fixture_zip(tmp_path / "probe.zip")

    x_slurp, y_slurp = _decode_png_stores([str(zpath)], size=8, limit=0, seed=1)

    monkeypatch.setattr(mix_corpus, "_SLURP_MAX_BYTES", 0)
    x_stream, y_stream = _decode_png_stores([str(zpath)], size=8, limit=0, seed=1)

    assert x_slurp.shape == x_stream.shape
    assert torch.equal(x_slurp, x_stream)
    assert torch.equal(y_slurp, y_stream)


def test_zip_png_reader_slurps_below_threshold(tmp_path: Path) -> None:
    """With `slurp=True` and below `_SLURP_MAX_BYTES`, the cached handle is opened
    over an in-memory buffer."""
    zpath = _fixture_zip(tmp_path / "small.zip")
    reader = ZipPngReader(slurp=True)
    try:
        payload = reader.read(ImageRef(store=zpath, member="a/00.png"))
        assert payload == _png_bytes((10, 20, 30))
        handle = reader._zips[zpath]
        assert isinstance(handle.fp, io.BytesIO)
    finally:
        reader.close()


def test_zip_png_reader_streams_above_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `slurp=True` but at/above `_SLURP_MAX_BYTES`, the path is opened directly
    -- no whole-file read. The size guard applies even when the caller asked to slurp.

    A real multi-GB archive is not needed to exercise this branch: the threshold
    itself is what the reader consults, so forcing it below this fixture's actual
    size takes the same code path a huge train zip would.
    """
    zpath = _fixture_zip(tmp_path / "small.zip")
    monkeypatch.setattr(mix_corpus, "_SLURP_MAX_BYTES", zpath.stat().st_size - 1)
    reader = ZipPngReader(slurp=True)
    try:
        payload = reader.read(ImageRef(store=zpath, member="a/00.png"))
        assert payload == _png_bytes((10, 20, 30))
        handle = reader._zips[zpath]
        assert not isinstance(handle.fp, io.BytesIO)
    finally:
        reader.close()


def test_zip_png_reader_slurp_boundary_is_inclusive(tmp_path: Path) -> None:
    """With `slurp=True`, exactly at the threshold still slurps (`<=`, not `<`) -- a
    boundary regression guard, since an off-by-one here would silently stream every
    archive whose size happens to equal the constant."""
    zpath = _fixture_zip(tmp_path / "boundary.zip")
    size = zpath.stat().st_size
    reader = ZipPngReader(slurp=True)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(mix_corpus, "_SLURP_MAX_BYTES", size)
            reader.read(ImageRef(store=zpath, member="a/00.png"))
        handle = reader._zips[zpath]
        assert isinstance(handle.fp, io.BytesIO)
    finally:
        reader.close()


def test_pngtrain_reader_never_slurps(tmp_path: Path) -> None:
    """The memory-leak guard this fix-round exists for.

    `PngTrain` lives for the whole training run and samples random indices across the
    entire corpus (never one bounded pass), so its `ZipPngReader` must never hold a
    `BytesIO` handle regardless of how small the shard is or what `_SLURP_MAX_BYTES`
    is set to -- opening `slurp=False` unconditionally is what makes that true, not a
    size check. A regression that flipped `PngTrain` back to `slurp=True` (or made
    `slurp` default to `True`) would put every train shard under the threshold
    permanently in RAM, multiplied across concurrently packed runs (see the finding in
    this commit's message). The fixture zip here is a few hundred bytes, i.e. it would
    slurp under any plausible threshold if `PngTrain` asked for that.
    """
    zpath = _fixture_zip(tmp_path / "train.zip")
    train = PngTrain([str(zpath)], size=8, seed=1, limit=0)
    try:
        _ = train[0]  # force the reader to actually open the zip
        handle = train.reader._zips[zpath]
        assert not isinstance(handle.fp, io.BytesIO)
        assert train.reader._slurp is False
    finally:
        train.reader.close()
