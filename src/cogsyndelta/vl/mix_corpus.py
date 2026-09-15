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

import io
import json
import random
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cogsyndelta.corpus import CORPUS_FINGERPRINT_SCHEME, fingerprint_corpus

PNG_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

_SLURP_MAX_BYTES = 2 * 1024**3
"""Slurp a zip into a `BytesIO` before opening it when its size is at or below this.

WHY: `zipfile.ZipFile` opened directly on a path issues its own small `seek`+`read`
syscalls per member (central-directory lookup at open, then one read per PNG). On
local disk each syscall is cheap and this pattern costs nothing measurable. Over NFS,
each of those small reads is a separate round trip: R3
(`/akula-data/session-backup-staging/reviews/r3-decode-probe/REPORT.md`) measured this
at +1.3s across 15,400 probe images (170MB) versus local NVMe, even page-cache-warm --
not disk wait, since the true cold I/O floor for that data is only 1.44s. Reading the
whole archive into memory ONCE turns it into a single sequential read (which NFS
readahead satisfies cheaply), after which every member access is a pure in-memory
`seek` inside the `BytesIO`; decoded output is byte-identical either way. The
threshold keeps this from slurping a multi-GB train zip whole into RAM -- those are
still streamed exactly as before.
"""

DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / "config" / "mind" / "visual-clean-v1.json"
PRIMARY_PROBE = "eurosat-test"
STAMP_DIR_RE = re.compile(r"^\d{8}T\d{6}Z$")
SUPERSEDED_MARKER = "SUPERSEDED.json"
CONTAMINATED_MARKER = "CONTAMINATED_SPLIT.json"
OPERATIONS = "operations.json"


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
    """Keep zip handles open across a training step. Never extracts members to disk.

    `slurp` has no default on purpose: a reader that stays alive for a whole training
    run (`PngTrain`, below) must NOT silently slurp, because it samples random indices
    across the whole corpus rather than one pass -- every shard it ever touches gets
    opened exactly once and then cached in `_zips` for the run's entire lifetime, with
    no eviction. A transient reader (`_decode_png_stores`, closed in a `finally` right
    after one full pass) is where slurping actually pays off and where the memory it
    holds is bounded and short-lived. See `_SLURP_MAX_BYTES` for the size guard that
    still applies even when `slurp=True`.
    """

    def __init__(self, slurp: bool) -> None:
        """Create an empty zip-handle cache. ``slurp`` decides `_open`'s policy below."""
        self._zips: dict[Path, zipfile.ZipFile] = {}
        self._slurp = slurp

    def read(self, ref: ImageRef) -> bytes:
        """Return PNG bytes for ``ref`` without extracting the zip to disk."""
        if ref.member is None:
            return ref.store.read_bytes()
        handle = self._zips.get(ref.store)
        if handle is None:
            handle = self._open(ref.store)
            self._zips[ref.store] = handle
        return handle.read(ref.member)

    def _open(self, store: Path) -> zipfile.ZipFile:
        """Open ``store``, slurping it into memory first when told to and it's small.

        `self._slurp` is `False` for a reader (`PngTrain`) that never gets to close
        or evict any of its cached handles, so slurping there would mean every train
        shard under `_SLURP_MAX_BYTES` staying resident in RAM for the run's entire
        lifetime -- multiplied across concurrently packed runs. It is `True` only for
        a reader that is closed right after one bounded pass (`_decode_png_stores`),
        where the size guard below still keeps a huge shard from being slurped at all.
        `stat()` failing (store vanished between listing and read) is not this
        method's problem to hide: fall through to opening the path directly and let
        that raise its own, more specific error.
        """
        if not self._slurp:
            return zipfile.ZipFile(store)
        try:
            size = store.stat().st_size
        except OSError:
            size = _SLURP_MAX_BYTES + 1
        if size <= _SLURP_MAX_BYTES:
            return zipfile.ZipFile(io.BytesIO(store.read_bytes()))
        return zipfile.ZipFile(store)

    def close(self) -> None:
        """Close every cached zip handle."""
        for handle in self._zips.values():
            handle.close()
        self._zips.clear()


def stamp_is_inactive(stamp_dir: Path) -> bool:
    """True if this processed stamp must not be used as Mix B train (g24 rule).

    Markers (any one is enough): ``SUPERSEDED.json``, any ``CONTAMINATED*.json``,
    or ``operations.json`` with ``contaminated_split: true``.
    """
    if (stamp_dir / SUPERSEDED_MARKER).is_file():
        return True
    if any(stamp_dir.glob("CONTAMINATED*.json")):
        return True
    ops = stamp_dir / OPERATIONS
    if ops.is_file():
        try:
            payload: Any = json.loads(ops.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = None
        if isinstance(payload, dict) and payload.get("contaminated_split") is True:
            return True
    return False


def processed_stamp_dirs(dataset_dir: Path) -> list[Path]:
    """ISO ``processed/<YYYYMMDDTHHMMSSZ>/`` dirs, oldest first."""
    processed = dataset_dir / "processed"
    if not processed.is_dir():
        return []
    return sorted(p for p in processed.iterdir() if p.is_dir() and STAMP_DIR_RE.match(p.name))


def choose_active_stamp(dataset_dir: Path) -> tuple[Path | None, list[str]]:
    """Newest non-inactive processed stamp, plus the inactive stamp names.

    This is the g24 rule: newest stamp without SUPERSEDED / CONTAMINATED markers.
    """
    stamps = processed_stamp_dirs(dataset_dir)
    inactive = [p.name for p in stamps if stamp_is_inactive(p)]
    active_dirs = [p for p in stamps if p.name not in set(inactive)]
    if not active_dirs:
        return None, inactive
    return active_dirs[-1], inactive


def _active_stamp_from_provenance(landing: Path) -> str | None:
    """Prefer ``active_train_set.stamp`` on the landing record; else g24 newest."""
    prov_path = landing / "provenance.json"
    if prov_path.is_file():
        try:
            prov: Any = json.loads(prov_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            prov = None
        if isinstance(prov, dict):
            ats = prov.get("active_train_set")
            if isinstance(ats, dict) and ats.get("stamp"):
                return str(ats["stamp"])
    active, _inactive = choose_active_stamp(landing)
    return active.name if active is not None else None


def check_source_stamp_active(manifest: dict[str, Any], source: dict[str, Any]) -> None:
    """Refuse a source whose stamp is superseded, contaminated, or not active.

    Args:
        manifest: Loaded visual-clean-v1 document (needs ``root``).
        source: One ``sources[]`` row.

    Raises:
        MixCorpusError: The named stamp is unusable as Mix B train.
    """
    landing = Path(manifest["root"]) / source["landing"]
    stamp = str(source["stamp"])
    stamp_dir = landing / "processed" / stamp
    if stamp_dir.is_dir() and stamp_is_inactive(stamp_dir):
        raise MixCorpusError(
            f"{source['id']} stamp {stamp} is inactive (SUPERSEDED or CONTAMINATED "
            f"under {stamp_dir})"
        )
    if not (landing / "provenance.json").is_file() and not processed_stamp_dirs(landing):
        return
    active = _active_stamp_from_provenance(landing)
    if active is not None and stamp != active:
        raise MixCorpusError(
            f"{source['id']} stamp {stamp} is not the active admitted stamp {active}"
        )


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load a visual-clean-v1 manifest and refuse inactive stamps.

    Each source's ``<landing>/provenance.json`` is read when present. The
    admissible stamp is ``active_train_set.stamp`` when that field exists,
    otherwise the newest ``processed/<ISO>/`` without SUPERSEDED /
    CONTAMINATED markers (g24). A stamp dir carrying those markers is
    refused even when provenance is absent.
    """
    p = Path(path) if path is not None else DEFAULT_MANIFEST
    if not p.is_file():
        raise MixCorpusError(f"visual corpus manifest not present: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema") != "csd-visual-corpus-manifest/v1":
        raise MixCorpusError(f"unsupported visual corpus schema: {data.get('schema')!r}")
    if not data.get("sources"):
        raise MixCorpusError(f"{p}: no sources")
    for source in data["sources"]:
        if source.get("role") != "train":
            continue
        check_source_stamp_active(data, source)
    return data


def train_sources(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return Mix B sources whose ``role`` is ``train``."""
    return [s for s in manifest["sources"] if s.get("role") == "train"]


def source_train_path(manifest: dict[str, Any], source: dict[str, Any]) -> Path:
    """Resolve one source's train zip or PNG tree under ``manifest["root"]``."""
    root = Path(manifest["root"])
    return root / source["landing"] / source["train"]


def source_probe_path(manifest: dict[str, Any], source: dict[str, Any]) -> Path | None:
    """Resolve one source's probe artefact, or ``None`` when the source has no probe."""
    probe = source.get("probe")
    if not probe:
        return None
    return Path(manifest["root"]) / source["landing"] / probe


def refuse_unless_manifest_consistent(
    path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the Mix B manifest and refuse unless listed=declared and paths are disjoint.

    This is the run-start gate ``run_vl_region`` used to inline: ``load_manifest``,
    ``dry_run`` (which already checks concentration, listed=declared, and disjoint
    paths), then the same listed/disjoint checks again so a caller that skips
    ``dry_run`` still cannot start.

    Returns:
        ``(manifest, dry_info)``.

    Raises:
        MixCorpusError: inactive stamp, listed≠declared, overlap, or missing file.
    """
    manifest = load_manifest(path)
    dry_info = dry_run(manifest)
    check_listed_matches_declared(dry_info)
    check_paths_disjoint(manifest)
    return manifest, dry_info


def check_paths_disjoint(manifest: dict[str, Any]) -> None:
    """Refuse when a resolved train path is also a resolved probe path.

    Args:
        manifest: Loaded visual-clean-v1 document.

    Raises:
        MixCorpusError: Naming both sides of the first intersecting path.
    """
    train_of: dict[Path, str] = {}
    probe_of: dict[Path, str] = {}
    for source in manifest["sources"]:
        sid = str(source["id"])
        train_of[source_train_path(manifest, source).resolve()] = sid
        probe = source_probe_path(manifest, source)
        if probe is not None:
            probe_of[probe.resolve()] = sid
    overlap = set(train_of) & set(probe_of)
    if not overlap:
        return
    path = sorted(overlap, key=str)[0]
    raise MixCorpusError(
        f"train path intersects probe path {path} "
        f"(train source {train_of[path]}, probe source {probe_of[path]})"
    )


def check_listed_matches_declared(dry_info: dict[str, Any]) -> None:
    """Refuse listed!=declared or a train source that lists zero / missing.

    Args:
        dry_info: Return value of :func:`dry_run`.

    Raises:
        MixCorpusError: First offending source, with both counts.
    """
    for row in dry_info["sources"]:
        listed = int(row["listed_train"])
        declared = int(row["declared_train"])
        if listed != declared or listed <= 0:
            raise MixCorpusError(f"{row['id']} listed_train={listed} declared_train={declared}")
    listed_total = int(dry_info["listed_train"])
    declared_total = int(dry_info["declared_train"])
    if listed_total != declared_total:
        raise MixCorpusError(f"listed_train {listed_total} != declared_train {declared_total}")


def receipt_shard_tail(path: str | Path) -> str:
    """Landing-qualified tail ``<landing>/processed/<stamp>/<file>``, else basename.

    Mix B receipts must not stamp ``train.zip`` seven times. Parquet tests have
    no landing layout, so they keep the basename.
    """
    parts = Path(path).parts
    if "processed" in parts:
        idx = parts.index("processed")
        if idx >= 1:
            return "/".join(parts[idx - 1 :])
    return Path(path).name


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
    """``csd-corpus-fp/v2`` hash of Mix B train artefacts (basename+size)."""
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
    """``list_pngs`` then shuffle with ``seed``. Not a cryptographic RNG."""
    refs = list_pngs(store)
    rng = random.Random(seed)  # noqa: S311 — corpus shuffle, not crypto
    rng.shuffle(refs)
    return refs


def class_name_from_ref(ref: ImageRef) -> str | None:
    """EuroSAT-style class folder: zip member's first path part, or the file's parent."""
    if ref.member is not None:
        parts = Path(ref.member).parts
        return parts[0] if len(parts) >= 2 else None
    parent = ref.store.parent.name
    return parent or None


def count_source(manifest: dict[str, Any], source: dict[str, Any]) -> dict[str, int]:
    """Listed vs declared train/probe PNG counts for one Mix B source."""
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
    Refuses listed!=declared, a zero listing, and train/probe path overlap.
    """
    check_paths_disjoint(manifest)
    largest = check_concentration(manifest)
    per_source = []
    listed_train = 0
    for source in train_sources(manifest):
        counts = count_source(manifest, source)
        listed_train += max(counts["listed_train"], 0)
        per_source.append({"id": source["id"], "stamp": source["stamp"], **counts})
    result = {
        "corpus_source": manifest["id"],
        "schema": manifest["schema"],
        "fingerprint": fingerprint_train(manifest),
        "fingerprint_scheme": CORPUS_FINGERPRINT_SCHEME,
        "concentration_largest_share": round(largest, 4),
        "concentration_cap": float(manifest["concentration_cap"]),
        "declared_train": int(manifest["n_train"]),
        "listed_train": listed_train,
        "sources": per_source,
        "probe_sets": [p["name"] for p in (manifest.get("probe_sets") or [])],
        "shuffle_seed": int(manifest.get("shuffle_seed", 0)),
    }
    check_listed_matches_declared(result)
    return result


def _iter_tree_pngs(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in PNG_SUFFIXES]
    files.sort()
    return files
