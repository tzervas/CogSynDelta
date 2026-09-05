"""visual-clean-v1 Mix B manifest: listing, fingerprint, concentration (CPU, no /bulk)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from cogsyndelta.corpus import CORPUS_FINGERPRINT_SCHEME
from cogsyndelta.vl.mix_corpus import (
    ConcentrationError,
    MixCorpusError,
    check_concentration,
    check_paths_disjoint,
    dry_run,
    fingerprint_train,
    list_pngs,
    load_manifest,
    receipt_shard_tail,
    shuffled_pngs,
    source_train_path,
)

pytestmark = pytest.mark.cpu

# 1x1 RGB PNG (valid, tiny).
_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x00\x03\x00\x01\x00\x05\xfe\xd4\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PNG)


def _write_zip(path: Path, members: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name in members:
            zf.writestr(name, _PNG)


def _manifest(tmp: Path, *, a_n: int = 3, b_n: int = 2, c_n: int = 1) -> dict:
    root = tmp / "visual"
    _write_zip(
        root / "aa__aa" / "processed" / "s" / "train.zip",
        [f"a/{i:02d}.png" for i in range(a_n)],
    )
    _write_zip(
        root / "bb__bb" / "processed" / "s" / "train.zip",
        [f"00/{i:02d}.png" for i in range(b_n)],
    )
    tree = root / "cc__cc" / "processed" / "s" / "train"
    for i in range(c_n):
        _write_png(tree / "cls" / f"{i:02d}.png")
    probe = root / "aa__aa" / "processed" / "s" / "probe.zip"
    _write_zip(probe, ["a/p0.png"])
    body = {
        "id": "visual-clean-v1",
        "schema": "csd-visual-corpus-manifest/v1",
        "root": str(root),
        "concentration_cap": 0.4,
        "shuffle_seed": 7,
        "n_train": a_n + b_n + c_n,
        "sources": [
            {
                "id": "aa/aa",
                "landing": "aa__aa",
                "stamp": "s",
                "train": "processed/s/train.zip",
                "probe": "processed/s/probe.zip",
                "n_train": a_n,
                "n_probe": 1,
                "licence_tier": "cc0_pd",
                "verdict": "PERMISSIVE_OK",
                "role": "train",
            },
            {
                "id": "bb/bb",
                "landing": "bb__bb",
                "stamp": "s",
                "train": "processed/s/train.zip",
                "probe": None,
                "n_train": b_n,
                "n_probe": 0,
                "licence_tier": "mit_bsd",
                "verdict": "PERMISSIVE_OK",
                "role": "train",
            },
            {
                "id": "cc/cc",
                "landing": "cc__cc",
                "stamp": "s",
                "train": "processed/s/train",
                "probe": None,
                "n_train": c_n,
                "n_probe": 0,
                "licence_tier": "apache_2_0",
                "verdict": "PERMISSIVE_OK",
                "role": "train",
            },
        ],
        "probe_sets": [{"name": "aa-probe", "source": "aa/aa", "n": 1, "role": "primary"}],
    }
    path = tmp / "visual-clean-v1.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return load_manifest(path)


def test_lists_zip_members_and_tree_pngs(tmp_path: Path) -> None:
    man = _manifest(tmp_path, a_n=2, b_n=2, c_n=2)
    info = dry_run(man)
    assert info["listed_train"] == 6
    assert info["declared_train"] == 6
    by_id = {row["id"]: row for row in info["sources"]}
    assert by_id["aa/aa"]["listed_train"] == 2
    assert by_id["bb/bb"]["listed_train"] == 2
    assert by_id["cc/cc"]["listed_train"] == 2
    assert by_id["aa/aa"]["listed_probe"] == 1
    assert info["probe_sets"] == ["aa-probe"]
    assert info["fingerprint_scheme"] == CORPUS_FINGERPRINT_SCHEME


def test_fingerprint_is_deterministic(tmp_path: Path) -> None:
    man = _manifest(tmp_path)
    a = fingerprint_train(man)
    b = fingerprint_train(man)
    assert a == b
    assert len(a) == 32


def test_concentration_ok_when_largest_at_or_below_cap(tmp_path: Path) -> None:
    # 2/6, 2/6, 2/6 = 0.333 <= 0.40
    man = _manifest(tmp_path, a_n=2, b_n=2, c_n=2)
    share = check_concentration(man)
    assert share <= 0.40


def test_concentration_refuses_when_one_source_dominates(tmp_path: Path) -> None:
    # declared 9+1+1 = 11, largest 9/11 ≈ 0.818 > 0.40 (mutation of n_train only)
    man = _manifest(tmp_path, a_n=3, b_n=2, c_n=1)
    man["sources"][0]["n_train"] = 9
    man["sources"][1]["n_train"] = 1
    man["sources"][2]["n_train"] = 1
    man["n_train"] = 11
    with pytest.raises(ConcentrationError, match="exceeds cap"):
        check_concentration(man)


def test_shuffle_is_seeded(tmp_path: Path) -> None:
    man = _manifest(tmp_path, a_n=4, b_n=1, c_n=1)
    store = Path(man["root"]) / "aa__aa" / "processed" / "s" / "train.zip"
    a = [r.member for r in shuffled_pngs(store, 7)]
    b = [r.member for r in shuffled_pngs(store, 7)]
    c = [r.member for r in shuffled_pngs(store, 8)]
    assert a == b
    assert a != c
    assert set(a) == {r.member for r in list_pngs(store)}


def test_receipt_shard_tail_is_landing_qualified() -> None:
    mix = "/bulk/visual/phelber__eurosat-rgb-128/processed/20260905T031650Z/train.zip"
    assert receipt_shard_tail(mix) == (
        "phelber__eurosat-rgb-128/processed/20260905T031650Z/train.zip"
    )
    assert receipt_shard_tail("scratch/images.parquet") == "images.parquet"


def _iso_landing(tmp: Path, *, point_at: str) -> Path:
    """Two ISO stamps: contaminated older, active newer. Manifest points at `point_at`."""
    root = tmp / "visual"
    landing = root / "pcam"
    old, new = "20260905T035325Z", "20260905T045655Z"
    for stamp in (old, new):
        _write_zip(
            landing / "processed" / stamp / "train.zip",
            [f"x/{stamp[-2:]}.png"],
        )
    (landing / "processed" / old / "CONTAMINATED_SPLIT.json").write_text(
        json.dumps({"contaminated_split": True, "do_not_admit_as_train": True}),
        encoding="utf-8",
    )
    (landing / "provenance.json").write_text(
        json.dumps({"active_train_set": {"stamp": new, "train_files": 1}}),
        encoding="utf-8",
    )
    body = {
        "id": "visual-clean-v1",
        "schema": "csd-visual-corpus-manifest/v1",
        "root": str(root),
        "concentration_cap": 0.4,
        "shuffle_seed": 7,
        "n_train": 1,
        "sources": [
            {
                "id": "basveeling/pcam",
                "landing": "pcam",
                "stamp": point_at,
                "train": f"processed/{point_at}/train.zip",
                "probe": None,
                "n_train": 1,
                "n_probe": 0,
                "licence_tier": "cc0_pd",
                "verdict": "PERMISSIVE_OK",
                "role": "train",
            }
        ],
        "probe_sets": [],
    }
    path = tmp / "visual-clean-v1.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_load_manifest_refuses_contaminated_stamp(tmp_path: Path) -> None:
    path = _iso_landing(tmp_path, point_at="20260905T035325Z")
    with pytest.raises(MixCorpusError, match="inactive"):
        load_manifest(path)


def test_load_manifest_accepts_active_stamp(tmp_path: Path) -> None:
    path = _iso_landing(tmp_path, point_at="20260905T045655Z")
    man = load_manifest(path)
    assert man["sources"][0]["stamp"] == "20260905T045655Z"


def test_dry_run_refuses_listed_not_declared(tmp_path: Path) -> None:
    man = _manifest(tmp_path, a_n=2, b_n=2, c_n=2)
    (Path(man["root"]) / "bb__bb" / "processed" / "s" / "train.zip").unlink()
    with pytest.raises(MixCorpusError, match=r"bb/bb listed_train=-1 declared_train=2"):
        dry_run(man)


def test_dry_run_refuses_zero_listing(tmp_path: Path) -> None:
    man = _manifest(tmp_path, a_n=2, b_n=2, c_n=2)
    empty = Path(man["root"]) / "bb__bb" / "processed" / "s" / "train.zip"
    empty.unlink()
    _write_zip(empty, [])
    with pytest.raises(MixCorpusError, match=r"bb/bb listed_train=0 declared_train=2"):
        dry_run(man)


def test_train_probe_path_overlap_refused(tmp_path: Path) -> None:
    man = _manifest(tmp_path, a_n=2, b_n=2, c_n=2)
    man["sources"][0]["train"] = man["sources"][0]["probe"]
    with pytest.raises(MixCorpusError, match="train path intersects probe path"):
        check_paths_disjoint(man)
    with pytest.raises(MixCorpusError, match="train path intersects probe path"):
        dry_run(man)


def test_rep_std_batch_spans_more_than_one_source(tmp_path: Path) -> None:
    """F2 used to slice x_tr[:batch]; Mix B's first 128 refs are all pxhere."""
    from cogsyndelta.regions.vl_pretrain import PngTrain, collapse_batch_indices

    man = _manifest(tmp_path, a_n=8, b_n=8, c_n=8)
    shards = [str(source_train_path(man, s)) for s in man["sources"]]
    ds = PngTrain(shards, size=1, seed=7, limit=0)
    prefix = {ds.refs[i].store for i in range(6)}
    assert len(prefix) == 1
    idx = collapse_batch_indices(ds.size(0), batch_size=6, seed=0)
    stores = {ds.refs[int(i)].store for i in idx.tolist()}
    assert len(stores) > 1
