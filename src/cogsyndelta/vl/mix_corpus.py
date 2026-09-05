"""OD-4 Mix B visual corpus: manifest, zip/png listing, concentration, fingerprint.

WHY THIS FILE EXISTS
`VL_REGIONS["visual"]["corpus_source"]` was None until Mix B was admitted. The landings
are PNG-in-zip (and, for tests, PNG trees). The parquet `_decode_split` path cannot read
them. This module is stdlib-only so CI can check identity, concentration and listing
without the train extra or `/bulk`.

Pixel decode stays in `regions/vl_pretrain.py` (Pillow is already a trainer dependency).
Zips are streamed as members; nothing is extracted to a tree of inodes.
"""

from __future__ import annotations

import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cogsyndelta.corpus import CORPUS_FINGERPRINT_SCHEME, fingerprint_corpus

PNG_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / "config" / "mind" / "visual-clean-v1.json"
PRIMARY_PROBE = "eurosat-test"


class MixCorpusError(RuntimeError):
    """Manifest or landing is unusable as a visual train set."""


class ConcentrationError(MixCorpusError):
    """Largest Mix B train source exceeds the declared share cap."""


@dataclass(frozen=True)
class ImageRef:
    """One PNG, either a zip member or a filesystem path. Never extracted."""

    store: Path
    member: str | None  # zip member name; None means `store` is the file itself


class ZipPngReader:
    """Keep zip handles open across a training step. Never extracts members to disk."""

    def __init__(self) -> None:
        self._zips: dict[Path, zipfile.ZipFile] = {}

    def read(self, ref: ImageRef) -> bytes:
        """Return PNG bytes for `ref` without extracting the zip."""
        if ref.member is None:
            return ref.store.read_bytes()
        handle = self._zips.get(ref.store)
        if handle is None:
            handle = zipfile.ZipFile(ref.store)
            self._zips[ref.store] = handle
        return handle.read(ref.member)

    def close(self) -> None:
        """Close cached zip handles."""
        for handle in self._zips.values():
            handle.close()
        self._zips.clear()


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_MANIFEST
    if not p.is_file():
        raise MixCorpusError(f"visual corpus manifest not present: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema") != "csd-visual-corpus-manifest/v1":
        raise MixCorpusError(f"unsupported visual corpus schema: {data.get('schema')!r}")
    if not data.get("sources"):
        raise MixCorpusError(f"{p}: no sources")
    return data


def train_sources(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in manifest["sources"] if s.get("role") == "train"]


def source_train_path(manifest: dict[str, Any], source: dict[str, Any]) -> Path:
    root = Path(manifest["root"])
    return root / source["landing"] / source["train"]


def source_probe_path(manifest: dict[str, Any], source: dict[str, Any]) -> Path | None:
    probe = source.get("probe")
    if not probe:
        return None
    return Path(manifest["root"]) / source["landing"] / probe


def check_concentration(manifest: dict[str, Any]) -> float:
    """Refuse if any train source's declared image count exceeds `concentration_cap`.

    Cap is on **image count**, not bytes (OD-4 Mix B; pxhere is 149680/581280 ≈ 0.257).
    """
    rows = train_sources(manifest)
    total = sum(int(s["n_train"]) for s in rows)
    if total <= 0:
        raise MixCorpusError("visual corpus train total is 0")
    cap = float(manifest.get("concentration_cap", 0.4))
    shares = [(s["id"], int(s["n_train"]) / total) for s in rows]
    largest_id, largest = max(shares, key=lambda kv: kv[1])
    if largest > cap:
        raise ConcentrationError(
            f"visual-clean-v1 concentration {largest_id} share {largest:.4f} "
            f"exceeds cap {cap:.2f} (n={int(next(s['n_train'] for s in rows if s['id'] == largest_id))}"
            f"/{total})"
        )
    return largest


def identity_shards(manifest: dict[str, Any]) -> list[str]:
    """Concrete train artefacts whose basename+size `fingerprint_corpus` hashes.

    Zips are one shard each. A PNG tree is expanded to its image files so a silent
    replace inside the tree changes the fingerprint. Missing paths still contribute
    basename + size 0 (same rule as `fingerprint_corpus`).
    """
    shards: list[str] = []
    for source in train_sources(manifest):
        path = source_train_path(manifest, source)
        if path.is_dir():
            shards.extend(str(p) for p in _iter_tree_pngs(path))
        else:
            shards.append(str(path))
    return shards


def fingerprint_train(manifest: dict[str, Any]) -> str:
    return fingerprint_corpus(identity_shards(manifest), columns=["image", "label"])


def list_pngs(store: Path) -> list[ImageRef]:
    """PNG members of a zip, or PNG files under a tree. No extraction."""
    if store.is_dir():
        return [ImageRef(store=p, member=None) for p in _iter_tree_pngs(store)]
    if store.is_file() and store.suffix.lower() == ".zip":
        with zipfile.ZipFile(store) as zf:
            names = [
                n
                for n in zf.namelist()
                if not n.endswith("/") and Path(n).suffix.lower() in PNG_SUFFIXES
            ]
        names.sort()
        return [ImageRef(store=store, member=n) for n in names]
    raise MixCorpusError(f"not a png zip or png tree: {store}")


def shuffled_pngs(store: Path, seed: int) -> list[ImageRef]:
    refs = list_pngs(store)
    rng = random.Random(seed)  # noqa: S311 — corpus shuffle, not crypto
    rng.shuffle(refs)
    return refs


def class_name_from_ref(ref: ImageRef) -> str | None:
    if ref.member is not None:
        parts = Path(ref.member).parts
        return parts[0] if len(parts) >= 2 else None
    parent = ref.store.parent.name
    return parent or None


def count_source(manifest: dict[str, Any], source: dict[str, Any]) -> dict[str, int]:
    train_path = source_train_path(manifest, source)
    n_train = len(list_pngs(train_path)) if train_path.exists() else -1
    probe_path = source_probe_path(manifest, source)
    n_probe = len(list_pngs(probe_path)) if probe_path is not None and probe_path.exists() else 0
    return {
        "declared_train": int(source["n_train"]),
        "listed_train": n_train,
        "declared_probe": int(source.get("n_probe") or 0),
        "listed_probe": n_probe,
    }


def dry_run(manifest: dict[str, Any]) -> dict[str, Any]:
    """Count listed PNGs per source, fingerprint the train artefacts, check concentration.

    Opens zip central directories only. Does not decode pixels and does not extract.
    """
    largest = check_concentration(manifest)
    per_source = []
    listed_train = 0
    for source in train_sources(manifest):
        counts = count_source(manifest, source)
        listed_train += max(counts["listed_train"], 0)
        per_source.append({"id": source["id"], "stamp": source["stamp"], **counts})
    fingerprint = fingerprint_train(manifest)
    probe_sets = list(manifest.get("probe_sets") or [])
    return {
        "corpus_source": manifest["id"],
        "schema": manifest["schema"],
        "fingerprint": fingerprint,
        "fingerprint_scheme": CORPUS_FINGERPRINT_SCHEME,
        "concentration_largest_share": round(largest, 4),
        "concentration_cap": float(manifest["concentration_cap"]),
        "declared_train": int(manifest["n_train"]),
        "listed_train": listed_train,
        "sources": per_source,
        "probe_sets": [p["name"] for p in probe_sets],
        "shuffle_seed": int(manifest.get("shuffle_seed", 0)),
    }


def _iter_tree_pngs(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in PNG_SUFFIXES]
    files.sort()
    return files
