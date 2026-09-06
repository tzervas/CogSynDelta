"""Fixed held-out split and batch-order manifests (E0 / G26).

WHY THIS EXISTS
`build_splits` used to shuffle and reservoir-sample with `PretrainConfig.seed`, so a
training-seed change also redrew the eval set. Matrix seed-0 and seed-1 cells were then
different test sets, and untrained baselines differed per seed for that reason (reason
region diagnosis 2026-09-05 H1/E0). Membership must be generated once from the corpus
fingerprint plus a `split_seed` (default 0), written to a hashed manifest, and loaded by
every trainer / benchmark / quantizer arm. The training seed must not touch membership.

This module is stdlib-only on purpose, matching `cogsyndelta.corpus`: G26's sha / fingerprint
checks have to run in CI, which does not install the `train` group.

WHY A FILE RATHER THAN "JUST USE split_seed"
A seed is a generator; a manifest is the draw. Two arms that share the file share the
exact membership even if the generator code later changes, and a doctored or
fingerprint-mismatched file is refuseable (G26) instead of silently producing a
plausible 512-pair holdout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SPLIT_MANIFEST_SCHEMA = "csd-split-manifest/v1"
ORDER_MANIFEST_SCHEMA = "csd-batch-order-manifest/v1"
RESERVED_HOLDOUT_SCHEMA = "csd-reserved-holdout/v1"
SPLIT_DRAW_ALGORITHM = "csd-split-draw/v1"
ORDER_ALGORITHM = "csd-batch-order/v1-leftover-then-permute"
FIRST_N_BATCHES = 64

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS_DIR = _REPO_ROOT / "config" / "mind" / "splits"

RESERVED_HOLDOUT_GLOB = "*-holdout.json"
REQUIRED_RESERVED_HOLDOUTS: tuple[str, ...] = ("reason-gsm8k-test-holdout.json",)
"""Reserved-holdout manifests that MUST exist, pinned by name (G38).

A glob alone would let the guard be disabled by deleting a file: no manifest, no ids, no
leak detected, everything green. So the glob picks up any future holdout automatically
AND these names are required to be present, so removing one is a failure rather than a
silent widening of what may be trained on.
"""


class SplitGuardError(RuntimeError):
    """G26 fail-closed: split membership and the corpus it was drawn from disagree."""


class ReservedHoldoutError(RuntimeError):
    """G38 fail-closed: a reserved-holdout item reached a training set.

    Distinct from :class:`SplitGuardError` because it is a different claim. G26 says
    "this run's eval set is the one the receipt names". G38 says "this row was never
    trained on by ANY region, so a battery scored on it is measuring the model rather
    than its memory". A holdout that is merely intended to be held out is not a holdout;
    the property has to be enforced where training pairs are assembled.
    """


def normalise_text(text: str) -> str:
    """Whitespace-collapse and case-fold, matching `build_splits` anchor dedup."""
    return " ".join(text.split()).lower()


def item_id(anchor: str, positive: str) -> str:
    """Ordered pair id: sha256 of normalised (anchor, positive).

    Ordered, not `pair_fingerprint`, because membership is the drawn (anchor, positive)
    row, not the unordered claim. A swapped pair is a different item.

    Args:
        anchor: Left side of the pair.
        positive: Right side of the pair.

    Returns:
        64-char hex digest.
    """
    payload = f"{normalise_text(anchor)}\n{normalise_text(positive)}".encode("utf-8", "replace")
    return hashlib.sha256(payload).hexdigest()


def membership_sha256(ids: list[str]) -> str:
    """Sha256 of sorted item ids: membership, independent of eval-list order.

    Args:
        ids: Holdout item ids, any order.

    Returns:
        64-char hex digest of the sorted, newline-joined id list.
    """
    payload = ("\n".join(sorted(ids)) + "\n").encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def split_manifest_path(
    region: str,
    corpus_fingerprint: str,
    split_seed: int,
    *,
    splits_dir: Path | None = None,
) -> Path:
    """`config/mind/splits/<region>-<fp8>-split<seed>.json`.

    Args:
        region: Region id as the trainer spells it on disk (`code`, not `language`).
        corpus_fingerprint: Full corpus fingerprint; only the first 8 hex chars go in
            the filename.
        split_seed: Generator seed recorded in the file.
        splits_dir: Override the repo default, for tests.

    Returns:
        Path; the file may not exist yet.
    """
    root = splits_dir if splits_dir is not None else DEFAULT_SPLITS_DIR
    return root / f"{region}-{corpus_fingerprint[:8]}-split{int(split_seed)}.json"


def order_manifest_path(
    region: str,
    corpus_fingerprint: str,
    order_seed: int,
    steps: int,
    batch_size: int,
    *,
    splits_dir: Path | None = None,
) -> Path:
    """Batch-order file beside the split manifest, keyed by (fp, order_seed, steps, batch).

    Args:
        region: Region id as trained.
        corpus_fingerprint: Full fingerprint; first 8 hex chars go in the filename.
        order_seed: Extra permutation seed; 0 is identity over the split leftover.
        steps: Training steps the sliding window will run.
        batch_size: Physical batch.
        splits_dir: Override the repo default, for tests.

    Returns:
        Path; the file may not exist yet.
    """
    root = splits_dir if splits_dir is not None else DEFAULT_SPLITS_DIR
    return (
        root / f"{region}-{corpus_fingerprint[:8]}-order{int(order_seed)}"
        f"-s{int(steps)}-b{int(batch_size)}.json"
    )


def load_json_manifest(path: Path) -> dict[str, Any]:
    """Read a split or order manifest.

    Args:
        path: Existing JSON file.

    Returns:
        Parsed object.

    Raises:
        SplitGuardError: Missing, unreadable, or not a JSON object.
    """
    if not path.is_file():
        raise SplitGuardError(f"G26: split/order manifest missing: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SplitGuardError(f"G26: cannot read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SplitGuardError(f"G26: manifest is not an object: {path}")
    return payload


def write_json_manifest(path: Path, payload: dict[str, Any]) -> Path:
    """Write a canonical, sorted-key JSON manifest.

    Args:
        path: Destination; parent dirs are created.
        payload: JSON-serialisable object.

    Returns:
        `path`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def build_split_manifest(
    *,
    region: str,
    corpus_fingerprint: str,
    split_seed: int,
    holdout: list[tuple[str, str]],
    train_pairs: list[tuple[str, str]],
    generator: dict[str, Any],
    counts: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a v1 split manifest from a completed draw.

    Args:
        region: Region id.
        corpus_fingerprint: Fingerprint the draw was taken from.
        split_seed: Generator seed.
        holdout: Held-out pairs in draw order (eval order).
        train_pairs: Remaining training pairs after contamination screening.
        generator: Algorithm name, seeds, caps, holdout size.
        counts: Realised row counts (train, holdout, duplicates, sources).

    Returns:
        Manifest dict ready to write.
    """
    holdout_ids = [item_id(a, b) for a, b in holdout]
    return {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "region": region,
        "corpus_fingerprint": corpus_fingerprint,
        "seed": int(split_seed),
        "algorithm": SPLIT_DRAW_ALGORITHM,
        "generator": generator,
        "counts": {
            **counts,
            "holdout_pairs": len(holdout),
            "train_pairs": len(train_pairs),
        },
        "holdout_ids": holdout_ids,
        "sha256": membership_sha256(holdout_ids),
    }


def build_order_manifest(
    *,
    region: str,
    corpus_fingerprint: str,
    order_seed: int,
    steps: int,
    batch_size: int,
    n_train: int,
) -> dict[str, Any]:
    """Assemble a compact v1 batch-order manifest.

    Stores the permutation seed, algorithm version, and sha256 of the first N sliding-
    window start indices -- not the full permutation. `order_seed` 0 is identity over
    the leftover of the split-seed draw (historical seed-0 batch order).

    Args:
        region: Region id.
        corpus_fingerprint: Fingerprint the split was taken from.
        order_seed: Extra permutation seed.
        steps: Training steps.
        batch_size: Physical batch.
        n_train: Number of training pairs after the split.

    Returns:
        Manifest dict ready to write.
    """
    starts = batch_start_indices(n_train, batch_size, steps)
    return {
        "schema": ORDER_MANIFEST_SCHEMA,
        "region": region,
        "corpus_fingerprint": corpus_fingerprint,
        "seed": int(order_seed),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "n_train": int(n_train),
        "algorithm": ORDER_ALGORITHM,
        "first_n": len(starts),
        "sha256": _batch_index_sha256(starts),
    }


def batch_start_indices(
    n_train: int, batch_size: int, steps: int, n: int = FIRST_N_BATCHES
) -> list[int]:
    """Start offsets of the first `n` training batches, matching `pretrain_region`.

    Args:
        n_train: Length of `train_pairs`.
        batch_size: Physical batch.
        steps: Total optimizer steps.
        n: How many leading starts to record.

    Returns:
        List of `lo` values from `lo = (step * batch) % max(1, n_train - batch)`.
    """
    denom = max(1, n_train - batch_size)
    return [(step * batch_size) % denom for step in range(min(n, max(0, steps)))]


def _batch_index_sha256(starts: list[int]) -> str:
    payload = json.dumps(starts, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def verify_split_manifest(
    manifest: dict[str, Any],
    *,
    corpus_fingerprint: str,
    holdout: list[tuple[str, str]] | None = None,
) -> None:
    """G26: refuse a split file that does not match the corpus or the current draw.

    Args:
        manifest: Loaded split manifest.
        corpus_fingerprint: Fingerprint of the corpus on disk now.
        holdout: If given, the pairs just drawn; membership must match byte-for-byte.

    Raises:
        SplitGuardError: Schema, fingerprint, sha, or membership mismatch.
    """
    schema = manifest.get("schema")
    if schema != SPLIT_MANIFEST_SCHEMA:
        raise SplitGuardError(f"G26: split manifest schema {schema!r} != {SPLIT_MANIFEST_SCHEMA!r}")
    recorded_fp = manifest.get("corpus_fingerprint")
    if recorded_fp != corpus_fingerprint:
        raise SplitGuardError(
            f"G26: split manifest corpus fingerprint {recorded_fp} does not match "
            f"the corpus on disk {corpus_fingerprint}"
        )
    ids = manifest.get("holdout_ids")
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise SplitGuardError("G26: split manifest holdout_ids is missing or malformed")
    recorded_sha = manifest.get("sha256")
    recomputed = membership_sha256(ids)
    if recorded_sha != recomputed:
        raise SplitGuardError(
            f"G26: split manifest sha256 {recorded_sha} does not match membership "
            f"{recomputed} -- doctored or truncated file"
        )
    if holdout is not None:
        drawn_ids = [item_id(a, b) for a, b in holdout]
        if drawn_ids != ids:
            raise SplitGuardError(
                "G26: drawn holdout membership does not match the split manifest "
                f"(manifest sha {recorded_sha}, draw sha {membership_sha256(drawn_ids)})"
            )


def verify_order_manifest(
    manifest: dict[str, Any],
    *,
    corpus_fingerprint: str,
    order_seed: int,
    steps: int,
    batch_size: int,
    n_train: int,
) -> None:
    """Refuse an order file that does not match this run's (fp, seed, steps, batch).

    Args:
        manifest: Loaded order manifest.
        corpus_fingerprint: Fingerprint of the corpus on disk now.
        order_seed: This run's order seed.
        steps: This run's steps.
        batch_size: This run's batch.
        n_train: This run's training-pair count.

    Raises:
        SplitGuardError: Any recorded field disagrees with the run.
    """
    if manifest.get("schema") != ORDER_MANIFEST_SCHEMA:
        raise SplitGuardError(
            f"G26: batch-order manifest schema {manifest.get('schema')!r} != "
            f"{ORDER_MANIFEST_SCHEMA!r}"
        )
    if manifest.get("corpus_fingerprint") != corpus_fingerprint:
        raise SplitGuardError(
            "G26: batch-order manifest corpus fingerprint "
            f"{manifest.get('corpus_fingerprint')} != {corpus_fingerprint}"
        )
    for key, value in (
        ("seed", int(order_seed)),
        ("steps", int(steps)),
        ("batch_size", int(batch_size)),
        ("n_train", int(n_train)),
    ):
        if int(manifest.get(key, -1)) != value:
            raise SplitGuardError(
                f"G26: batch-order manifest {key}={manifest.get(key)!r} != {value!r}"
            )
    starts = batch_start_indices(n_train, batch_size, steps)
    expected = _batch_index_sha256(starts)
    if manifest.get("sha256") != expected:
        raise SplitGuardError(
            f"G26: batch-order sha256 {manifest.get('sha256')} != recomputed {expected}"
        )


def assert_no_held_out_in_pairs(
    holdout: list[tuple[str, str]],
    pairs: list[tuple[str, str]],
    *,
    where: str = "training pairs",
) -> None:
    """G26: refuse if any held-out item appears in `pairs` (a batch or the train set).

    Args:
        holdout: Held-out pairs.
        pairs: Candidate pairs (one batch, or all of training).
        where: Label for the error.

    Raises:
        SplitGuardError: A held-out item id is present in `pairs`.
    """
    holdout_ids = {item_id(a, b) for a, b in holdout}
    leaked = [item_id(a, b) for a, b in pairs if item_id(a, b) in holdout_ids]
    if leaked:
        raise SplitGuardError(
            f"G26: {len(leaked)} held-out item(s) appear in {where} (example {leaked[0][:12]}...)"
        )


def build_reserved_holdout_manifest(
    *,
    name: str,
    source: str,
    split: str,
    region_scope: str,
    pair_columns: list[str],
    shard: str,
    shard_sha256: str,
    rows: int,
    item_ids: list[str],
    licence: str,
    upstream: str,
    notes: str,
) -> dict[str, Any]:
    """Assemble a v1 reserved-holdout manifest for one source split.

    Args:
        name: Manifest stem, ending `-holdout`.
        source: Upstream repo id, e.g. `openai/gsm8k`.
        split: The upstream split reserved, e.g. `test`.
        region_scope: `all` -- reserved against every region, which is the only value
            that makes the holdout meaningful for a composed model.
        pair_columns: Columns the ids were built from, in order.
        shard: Absolute path of the parquet the ids were read from.
        shard_sha256: Content digest of that parquet.
        rows: Row count read.
        item_ids: One :func:`item_id` per row.
        licence: Licence string verified at the primary source.
        upstream: Where that verification was read.
        notes: Why this split is reserved.

    Returns:
        Manifest dict ready to write.
    """
    return {
        "schema": RESERVED_HOLDOUT_SCHEMA,
        "name": name,
        "source": source,
        "split": split,
        "region_scope": region_scope,
        "pair_columns": list(pair_columns),
        "shard": shard,
        "shard_sha256": shard_sha256,
        "rows": int(rows),
        "licence": licence,
        "upstream": upstream,
        "notes": notes,
        "item_ids": sorted(item_ids),
        "sha256": membership_sha256(item_ids),
    }


def verify_reserved_holdout_manifest(manifest: dict[str, Any], *, path: Path) -> list[str]:
    """G38: refuse a reserved-holdout file whose ids do not hash to its own sha256.

    Args:
        manifest: Loaded manifest.
        path: Where it came from, for the message.

    Returns:
        The manifest's item ids.

    Raises:
        ReservedHoldoutError: Wrong schema, malformed ids, or a membership/sha mismatch.
    """
    schema = manifest.get("schema")
    if schema != RESERVED_HOLDOUT_SCHEMA:
        raise ReservedHoldoutError(
            f"G38: reserved-holdout schema {schema!r} != {RESERVED_HOLDOUT_SCHEMA!r} ({path})"
        )
    ids = manifest.get("item_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
        raise ReservedHoldoutError(f"G38: reserved-holdout item_ids missing or malformed ({path})")
    recomputed = membership_sha256(ids)
    if manifest.get("sha256") != recomputed:
        raise ReservedHoldoutError(
            f"G38: reserved-holdout sha256 {manifest.get('sha256')} != membership "
            f"{recomputed} -- doctored or truncated file ({path})"
        )
    return ids


def load_reserved_holdout_ids(*, splits_dir: Path | None = None) -> frozenset[str]:
    """Every reserved-holdout item id, across all committed holdout manifests.

    Fails closed twice over: a pinned manifest in :data:`REQUIRED_RESERVED_HOLDOUTS`
    that is absent is an error rather than an empty set, and any manifest whose ids do
    not hash to its recorded sha256 is refused. An empty result can therefore only mean
    "no holdouts are declared", never "the guard could not read its inputs".

    Args:
        splits_dir: Override the repo default, for tests.

    Returns:
        Frozen set of reserved item ids.

    Raises:
        ReservedHoldoutError: A required manifest is missing, or any manifest is invalid.
    """
    root = splits_dir if splits_dir is not None else DEFAULT_SPLITS_DIR
    for required in REQUIRED_RESERVED_HOLDOUTS:
        if not (root / required).is_file():
            raise ReservedHoldoutError(
                f"G38: required reserved-holdout manifest missing: {root / required}. "
                f"Refusing to build a training set that cannot be checked against it."
            )
    ids: set[str] = set()
    for path in sorted(root.glob(RESERVED_HOLDOUT_GLOB)):
        ids.update(verify_reserved_holdout_manifest(load_json_manifest(path), path=path))
    return frozenset(ids)


def assert_no_reserved_holdout_in_pairs(
    pairs: list[tuple[str, str]],
    *,
    where: str = "training pairs",
    splits_dir: Path | None = None,
    reserved: frozenset[str] | None = None,
) -> None:
    """G38: refuse if any reserved-holdout item appears in `pairs`.

    Called from `build_splits` on the realised training pairs, so it covers every region
    and every composite phase that assembles its corpus there -- the holdout is excluded
    BY ITEM ID, not by the shard path happening to be left out of a config.

    Args:
        pairs: Candidate training pairs.
        where: Label for the error.
        splits_dir: Override the manifest directory, for tests.
        reserved: Pre-loaded id set; loaded from `splits_dir` when omitted.

    Raises:
        ReservedHoldoutError: A reserved item id is present in `pairs`.
    """
    ids = load_reserved_holdout_ids(splits_dir=splits_dir) if reserved is None else reserved
    if not ids:
        return
    leaked = [item_id(a, b) for a, b in pairs if item_id(a, b) in ids]
    if leaked:
        raise ReservedHoldoutError(
            f"G38: {len(leaked)} reserved-holdout item(s) appear in {where} "
            f"(example {leaked[0][:12]}...). These rows are reserved as a holdout for a "
            f"pre-registered battery; training on them makes that battery a memory test."
        )


def verify_receipt_split(
    receipt: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """G26 (benchmark): refuse a receipt whose split.sha256 is not this manifest's.

    A legacy receipt with no `split` block is accepted only when `config.seed` equals
    the manifest seed (the seed-0 cells E0 keeps comparable). Seed-1 cells are retired.

    Args:
        receipt: Training receipt being scored.
        manifest: Split manifest the benchmark loaded.

    Raises:
        SplitGuardError: Sha mismatch, or a legacy non-seed-0 receipt.
    """
    recorded = receipt.get("split") if isinstance(receipt.get("split"), dict) else {}
    recorded_sha = recorded.get("sha256") if isinstance(recorded, dict) else None
    manifest_sha = manifest.get("sha256")
    if recorded_sha:
        if recorded_sha != manifest_sha:
            raise SplitGuardError(
                f"G26: receipt split.sha256 {recorded_sha} != manifest {manifest_sha}"
            )
        return
    cfg = receipt.get("config") if isinstance(receipt.get("config"), dict) else {}
    receipt_seed = int(cfg.get("seed", -1)) if cfg else -1
    manifest_seed = int(manifest.get("seed", 0))
    if receipt_seed != manifest_seed:
        raise SplitGuardError(
            f"G26: legacy receipt has no split.sha256 and config.seed={receipt_seed} "
            f"!= split manifest seed {manifest_seed}; seed-1 cells are retired from "
            "comparison (E0)"
        )


def resolve_split_manifest_path(
    region: str,
    corpus_fingerprint: str,
    split_seed: int,
    explicit: str | None,
    *,
    splits_dir: Path | None = None,
) -> Path:
    """Explicit path wins; otherwise the committed default for this (region, fp, seed).

    Args:
        region: Region id as trained.
        corpus_fingerprint: Current corpus fingerprint.
        split_seed: Split generator seed.
        explicit: `PretrainConfig.split_manifest`, or None.
        splits_dir: Override default dir.

    Returns:
        Path to load (may not exist).
    """
    if explicit:
        return Path(explicit)
    return split_manifest_path(region, corpus_fingerprint, split_seed, splits_dir=splits_dir)


def resolve_order_manifest_path(
    region: str,
    corpus_fingerprint: str,
    order_seed: int,
    steps: int,
    batch_size: int,
    explicit: str | None,
    *,
    splits_dir: Path | None = None,
) -> Path:
    """Explicit path wins; otherwise the committed default for this order key.

    Args:
        region: Region id as trained.
        corpus_fingerprint: Current corpus fingerprint.
        order_seed: Order seed.
        steps: Training steps.
        batch_size: Physical batch.
        explicit: `PretrainConfig.order_manifest`, or None.
        splits_dir: Override default dir.

    Returns:
        Path to load (may not exist).
    """
    if explicit:
        return Path(explicit)
    return order_manifest_path(
        region,
        corpus_fingerprint,
        order_seed,
        steps,
        batch_size,
        splits_dir=splits_dir,
    )


def permute_train_pairs(
    train_pairs: list[tuple[str, str]], order_seed: int
) -> list[tuple[str, str]]:
    """Apply the extra batch-order permutation. Seed 0 is identity.

    Why identity at 0: the historical seed-0 train order IS the leftover of the
    split-seed shuffle. An extra Random(0).shuffle would change every seed-0 cell's
    batch order and retire them from comparison, which E0 exists not to do.

    Args:
        train_pairs: Remaining pairs after the holdout prefix and contamination screen.
        order_seed: Extra permutation seed.

    Returns:
        Possibly-shuffled copy; `order_seed==0` returns a shallow copy in the same order.
    """
    out = list(train_pairs)
    if int(order_seed) == 0:
        return out
    import random as _random

    _random.Random(int(order_seed)).shuffle(out)  # noqa: S311
    return out
