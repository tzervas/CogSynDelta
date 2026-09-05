"""W7v-cfg (3): `VL_REGIONS` is retargeted off tiny-imagenet by construction.

tiny-imagenet is 100% BLOCKING as a `vl_latent` training source (`license: []` plus an
ImageNet Terms-of-Access `extra_gated_prompt`; audited
`docs/design/LICENCE-FOR-OPEN-WEIGHTS.md:364-411`, no clean fraction). Before this
increment it was still the hardcoded `train`/`probe_eval` glob in
`scripts/csd-train-all.py`'s `VL_REGIONS`, so a plain `--regions vl_latent` trained on it
by default. This file guards the fix: `corpus_source` must name a landed, admitted
corpus before `run_vl_region` resolves any shard, and there is no default left to fall
back to -- see OD-4 in `g8-visual-faculty-design.md` ("Operator decisions", row
"OD-4 mix").
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu


def _load_csd_train_all():
    """Import scripts/csd-train-all.py the way every other test of it does (hyphenated
    filename, not a valid module name -- see tests/test_reserved_corpus_guard.py)."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "csd-train-all.py"
    spec = importlib.util.spec_from_file_location("csd_train_all_visual_corpus_test", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_csd_train_all()


def test_visual_corpus_source_is_visual_clean_v1() -> None:
    """OD-4 Mix B is the admitted corpus; parquet globs stay unset (PNG-in-zip)."""
    assert mod.VL_REGIONS["vl_latent"]["corpus_source"] == "visual-clean-v1"
    assert mod.VL_REGIONS["visual"]["corpus_source"] == "visual-clean-v1"
    assert mod.VL_REGIONS["vl_latent"]["manifest"] == "config/mind/visual-clean-v1.json"
    assert mod.VL_REGIONS["vl_latent"]["train"] is None
    assert mod.VL_REGIONS["vl_latent"]["probe_eval"] is None


def test_vl_latent_no_longer_names_tiny_imagenet_anywhere_in_the_spec() -> None:
    """Regression guard for the actual defect: tiny-imagenet must not be reachable by
    training vl_latent with no extra configuration -- not as a glob, not as a fallback."""
    spec_text = repr(mod.VL_REGIONS["vl_latent"])
    assert "tiny-imagenet" not in spec_text
    assert "cifar100" not in spec_text


def test_run_vl_region_refuses_to_start_with_no_corpus_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "corpus_source", None)
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "manifest", None)
    with pytest.raises(mod.VisualCorpusUnsetError, match="OD-4"):
        mod.run_vl_region(name="vl_latent", state=tmp_path, steps=1, batch=1, dry=True)


def test_refusal_names_the_spec_key_it_expects_fixed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "corpus_source", None)
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "manifest", None)
    with pytest.raises(mod.VisualCorpusUnsetError, match=r"corpus_source"):
        mod.run_vl_region(name="vl_latent", state=tmp_path, steps=1, batch=1, dry=True)


def test_refusal_fires_before_any_shard_is_resolved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Even a `_shards` that WOULD happily resolve tiny-imagenet must never get the
    chance to run while `corpus_source` is unset -- the refusal has to come first."""
    called = {"n": 0}

    def spy(pattern, root=mod.CORPUS):
        called["n"] += 1
        return ["/mnt/fleet-datasets/csd/vl/tiny-imagenet/data/train-0.parquet"]

    monkeypatch.setattr(mod, "_shards", spy)
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "corpus_source", None)
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "manifest", None)

    with pytest.raises(mod.VisualCorpusUnsetError):
        mod.run_vl_region(name="vl_latent", state=tmp_path, steps=1, batch=1, dry=True)

    assert called["n"] == 0, "_shards must not run before the corpus_source gate"


def test_setting_corpus_source_lifts_the_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Positive control: once a corpus is admitted (named here), the gate steps aside
    and normal shard resolution (MISSING-source skip, since no real data is wired in
    this test) takes over instead."""
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "corpus_source", "admitted-placeholder")
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "manifest", None)
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "train", "vl/visual-clean-v1/train-*.parquet")
    monkeypatch.setitem(
        mod.VL_REGIONS["vl_latent"], "probe_eval", "vl/visual-clean-v1/valid-*.parquet"
    )
    monkeypatch.setitem(
        mod.VL_REGIONS["vl_latent"], "transfer", "vl/visual-clean-v1/test-*.parquet"
    )

    # No real shards on disk under this made-up corpus name, so this resolves to the
    # ordinary "source MISSING -- skipping" path (returns None) rather than
    # VisualCorpusUnsetError -- proving the OD-4 gate, specifically, is what lifted.
    result = mod.run_vl_region(name="vl_latent", state=tmp_path, steps=1, batch=1, dry=True)
    assert result is None


def test_run_vl_region_refuses_listed_not_declared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """B2: a missing zip is SystemExit with both counts, not a later MixCorpusError."""
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
        b"\x00\x00\x00\x03\x00\x01\x00\x05\xfe\xd4\xef\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    root = tmp_path / "visual"
    sources = [("aa", 2), ("bb", 2), ("cc", 2)]
    for name, n in sources:
        zpath = root / f"{name}__{name}" / "processed" / "s" / "train.zip"
        zpath.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "w") as zf:
            for i in range(n):
                zf.writestr(f"{name}/{i:02d}.png", png)
    body = {
        "id": "visual-clean-v1",
        "schema": "csd-visual-corpus-manifest/v1",
        "root": str(root),
        "concentration_cap": 0.4,
        "shuffle_seed": 7,
        "n_train": 6,
        "sources": [
            {
                "id": f"{name}/{name}",
                "landing": f"{name}__{name}",
                "stamp": "s",
                "train": "processed/s/train.zip",
                "probe": None,
                "n_train": n,
                "n_probe": 0,
                "licence_tier": "cc0_pd",
                "verdict": "PERMISSIVE_OK",
                "role": "train",
            }
            for name, n in sources
        ],
        "probe_sets": [],
    }
    man_path = tmp_path / "visual-clean-v1.json"
    man_path.write_text(json.dumps(body), encoding="utf-8")
    (root / "bb__bb" / "processed" / "s" / "train.zip").unlink()
    monkeypatch.setitem(mod.VL_REGIONS["vl_latent"], "manifest", str(man_path))
    with pytest.raises(SystemExit, match=r"bb/bb listed_train=-1 declared_train=2"):
        mod.run_vl_region(name="vl_latent", state=tmp_path, steps=1, batch=1, dry=True)
