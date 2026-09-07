#!/usr/bin/env python3
"""W3 item construction: stage X7 recall-dependent two-turn episodes from landed rows.

WHAT THIS BUILDS
Shape **X7** of the design's item table (§5.4), constructed per §5.5(e), on the
**pre-committed text-only slip branch** of §5.4 (X8' replaces X8, `R = 4`, six ablation
pairs, criterion >= 5 of 6 at null 0.109). Reported in the `episodic_store x memory` bin;
makes `{episodic_store, memory}` and `{episodic_store, language}` load-bearing.

WHY IT IS THE CRITICAL PATH
Phase A's episodic read is a CONSTANT and its mutual information with the target is
exactly zero, because the synthetic stream has **no recall dependency**: its inputs
determine the target, so `MI(read; target | inputs) = 0` for any read, informative or not.
§2.7.x makes a no-recall-dependency stream G29's own falsifier. X7 items are what make
the store load-bearing, and phase A is the only window in which store integration is
learned into weights that are not yet frozen.

WHAT IT DOES *NOT* BUILD, DELIBERATELY
* **The episode harness** (turn 1's write committing through the store's learn()
  lifecycle before turn 2 is scored; the partition reset between episodes). It depends on
  store commit semantics under active change in `interconnect/episodic/` and `mind.py`.
  The interface this generator assumes is written into the manifest's
  `harness_interface` block and into
  `docs/design/evidence/w3-x7-episodes-2026-09-07/README.md`.
* **Admission** (`s_r` / NSRS, §5.3). GPU-blocked and blocked on W2b. Items carry
  `admission: {"status": "pending", "blocked_by": ["W2b"]}` so nothing downstream can
  mistake a constructed item for an admitted one.

ZERO NEW SOURCE ROWS. §5.5(e): X7 is "a re-staging, not a new corpus". Every row consumed
is already landed and already drawn by the reserve.

USAGE
    export OMP_NUM_THREADS=1
    secret exec CSD_K_SPLIT=akula/csd-k-split -- \
        python scripts/csd-build-x7-episodes.py --out /akula-data/csd/reserve/x7

Fails closed without `CSD_K_SPLIT` (DEC-39) and without the corpus mount.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from cogsyndelta.eval.metrics import pair_fingerprint  # noqa: E402
from cogsyndelta.reserve.episodes import (  # noqa: E402
    DEFAULT_EVAL_PERMILLE,
    EPISODE_SCHEMA,
    LEDGER_SCHEMA,
    MANIFEST_SCHEMA,
    N_OPTIONS,
    REJECTION_REASONS,
    SPLIT_KEY_SCHEME,
    EpisodeRejectedError,
    FactDonor,
    ProbeSource,
    SplitKeyMissingError,
    build_episode,
    format_value,
    generator_identity,
    source_row_split,
    split_key_id,
)

CANDIDATE_CORPUS_ROOTS = (Path("/bulk/csd-corpus"), Path("/mnt/bulk/csd-corpus"))
"""Same two roots `scripts/csd-reserve-ledger.py` tries, in the same order."""

AQUA_REL = "reason/aqua_rat-raw"
APPS_REL = "code/apps"
AQUA_COLUMNS = ("question", "rationale")
APPS_COLUMNS = ("question", "solutions")

LICENCE_TIERS = {
    "mit": "permissive",
    "apache-2.0": "permissive",
    "cc-by-4.0": "permissive-attribution",
}
"""Policy tier per verbatim upstream licence string. Unknown => refuse, never guess."""

POOL_SIZE = 4
"""Turn 1's retrieval pool: the record to retrieve plus three distractor records."""

STATEMENT_CHARS = 1200
"""How much of a code problem statement turn 1 renders. Recorded in the config hash."""

BUILD_SEED = 0
"""Pairing seed. Arms share the seed so one change at a time is what gets measured."""

MAX_BUCKET = 5
"""Magnitude bands are decimal digit counts, capped here. See :func:`magnitude_bucket`.

WHY BANDS EXIST. Every option is ``donor_fact + probe_answer``. If the donor facts span
four orders of magnitude, one option is a visible outlier and a reader can shrink the
candidate set without recalling anything. Drawing the episode's five facts from one band
removes that prior, so chance stays the 1/5 the item prints.
"""

MIN_BUCKET_ROWS = 8 * (POOL_SIZE + N_OPTIONS - 1)
"""A band with fewer rows than this cannot be sampled without heavy reuse; it is dropped."""

_LETTER = re.compile(r"^\s*\(?([A-Ea-e])\)?[).:\-]?\s*(.*)$", re.S)
_NUMBER = re.compile(r"^\s*[-+]?\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:%|km|kg|m|cm|units?)?\s*$")


class BuildRefusedError(RuntimeError):
    """Raised when a precondition fails and the build must not silently degrade."""


@dataclass(frozen=True, slots=True)
class Row:
    """One usable source row, already fingerprinted and split-assigned.

    Attributes:
        dataset: Pinned dataset id.
        revision: Pinned revision.
        row_index: Ordinal in the shard.
        fingerprint: DEC-38 source-row key.
        split: ``"train"`` or ``"eval"``.
        licence: Verbatim upstream licence string.
        licence_tier: Policy tier.
        text: The row's primary text (a question, or a problem statement).
        secondary: The row's supporting text (a rationale, or a title).
        value: The numeric quantity this row can commit or answer with.
        bucket: Decimal magnitude band of ``value``, capped at :data:`MAX_BUCKET`.
    """

    dataset: str
    revision: str
    row_index: int
    fingerprint: str
    split: str
    licence: str
    licence_tier: str
    text: str
    secondary: str
    value: float
    bucket: int = 0


def resolve_corpus_root(explicit: Path | None = None) -> Path:
    """Find the corpus export, trying ``explicit`` then each candidate in order.

    Args:
        explicit: An override.

    Returns:
        The root directory.

    Raises:
        BuildRefusedError: If nothing resolves.
    """
    if explicit is not None:
        if not explicit.is_dir():
            raise BuildRefusedError(f"--corpus-root {explicit} is not a directory")
        return explicit
    for candidate in CANDIDATE_CORPUS_ROOTS:
        if candidate.is_dir():
            return candidate
    tried = ", ".join(str(c) for c in CANDIDATE_CORPUS_ROOTS)
    raise BuildRefusedError(f"no corpus root found (tried {tried}); pass --corpus-root")


def read_manifest(directory: Path) -> dict[str, Any]:
    """Read a landed corpus's ``MANIFEST.json``.

    Args:
        directory: The corpus directory.

    Returns:
        The manifest.

    Raises:
        BuildRefusedError: If it is missing, or its verdict is not ``TRAIN_OK``.
    """
    path = directory / "MANIFEST.json"
    if not path.exists():
        raise BuildRefusedError(f"{path} missing: sources are pinned by revision, not by name")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("verdict") != "TRAIN_OK":
        raise BuildRefusedError(f"{path} verdict is {manifest.get('verdict')!r}, not TRAIN_OK")
    return manifest


def licence_tier(licence: str) -> str:
    """Map a verbatim upstream licence string to its policy tier, or refuse.

    Args:
        licence: The manifest's verbatim ``license`` string.

    Returns:
        The tier.

    Raises:
        BuildRefusedError: If the licence is not one this generator has a tier for. The fleet
            has measured that mirror metadata lies; guessing a tier is how an NC term
            ends up inside an open-weights release.
    """
    head = licence.split(" ", 1)[0].strip().lower()
    if head not in LICENCE_TIERS:
        raise BuildRefusedError(f"no policy tier for licence {licence!r}; refusing to guess")
    return LICENCE_TIERS[head]


def magnitude_bucket(value: float) -> int:
    """Decimal digit count of a non-negative integral value, capped at :data:`MAX_BUCKET`.

    Args:
        value: The quantity.

    Returns:
        The band index.
    """
    return min(len(str(int(value))), MAX_BUCKET)


def parse_number(text: str) -> float | None:
    """Parse an option body as a plain number, or return ``None``.

    Args:
        text: The option body, letter prefix already stripped.

    Returns:
        The value, or ``None`` when the option is not a bare quantity.
    """
    match = _NUMBER.match(text)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def aqua_gold_value(options: list[str], correct: str) -> float | None:
    """The numeric value of an aqua_rat row's correct option.

    Args:
        options: The five ``"A)21"``-style options.
        correct: The correct letter.

    Returns:
        The value, or ``None`` when the row's answer is not a bare quantity.
    """
    letters: list[str] = []
    bodies: list[str] = []
    for option in options:
        match = _LETTER.match(option)
        if not match:
            return None
        letters.append(match.group(1).upper())
        bodies.append(match.group(2))
    want = correct.strip().upper()
    if want not in letters:
        return None
    return parse_number(bodies[letters.index(want)])


def burned_aqua_fingerprints(corpus_root: Path) -> set[str]:
    """The union of every aqua_rat draw the `reason` region burned (W2a, DEC-42).

    Reuses `scripts/csd-reserve-ledger.py`'s own draw functions rather than
    re-implementing them: two independently-written spellings of "which rows were burned"
    that drift apart is exactly how a guard ends up checking something else.

    Args:
        corpus_root: The corpus export root.

    Returns:
        Burned source-row fingerprints.

    Raises:
        BuildRefusedError: If the ledger script cannot be loaded.
    """
    import importlib.util

    path = REPO_ROOT / "scripts" / "csd-reserve-ledger.py"
    spec = importlib.util.spec_from_file_location("csd_reserve_ledger_for_x7", path)
    if spec is None or spec.loader is None:
        raise BuildRefusedError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    shards = module.locate_shards(corpus_root)
    if not shards:
        raise BuildRefusedError(f"no aqua_rat shard under {corpus_root}")
    burned: set[str] = set()
    burned.update(module.fingerprint_pairs(module.draw_reservoir(shards)))
    burned.update(module.fingerprint_pairs(module.draw_prefix(shards)))
    draws, unreadable = module.discover_ondisk_draws(corpus_root)
    if unreadable:
        raise BuildRefusedError(f"unreadable on-disk aqua_rat draws: {unreadable}")
    for draw in draws:
        burned.update(module.fingerprint_pairs(draw["pairs"]))
    return burned


def load_aqua_rows(corpus_root: Path, k_split: bytes, eval_permille: int) -> list[Row]:
    """Usable, unburned aqua_rat rows, fingerprinted and split-assigned.

    Three filters, each load-bearing rather than tidy:

    * **burned rows are dropped** -- the `reason` region already trained on the union of
      three aqua_rat draws (W2a / DEC-42), and a reserve item built from one of them is
      the leak DEC-38 exists to stop;
    * **the answer must be a positive integer** -- every option is
      ``fact + probe_answer``, so facts with different fractional parts would let a reader
      that solves the probe filter the option list by congruence and beat 1/5 without
      recalling anything;
    * **the answer must be stated verbatim in the rationale** -- turn 1 asks which passage
      resolves the query *and what value its reasoning arrives at*. If the passage never
      states the value, turn 1 is unanswerable and the fact it commits is a fiction.

    Args:
        corpus_root: The corpus export root.
        k_split: The split key.
        eval_permille: Per-mille of source rows on the eval side.

    Returns:
        Rows whose correct option is a bare positive integer their rationale states.
    """
    import pyarrow.parquet as pq

    directory = corpus_root / AQUA_REL
    manifest = read_manifest(directory)
    licence = str(manifest["license"])
    tier = licence_tier(licence)
    revision = str(manifest.get("revision") or "unpinned (MANIFEST.json declares none)")
    dataset = str(manifest["repo_id"])
    burned = burned_aqua_fingerprints(corpus_root)

    table = pq.read_table(directory / "train.parquet")
    questions = table.column("question").to_pylist()
    rationales = table.column("rationale").to_pylist()
    options = table.column("options").to_pylist()
    corrects = table.column("correct").to_pylist()

    rows: list[Row] = []
    for index, (question, rationale, option, correct) in enumerate(
        zip(questions, rationales, options, corrects, strict=True)
    ):
        if not question or not rationale:
            continue
        fingerprint = pair_fingerprint(question, rationale)
        if fingerprint in burned:
            continue
        value = aqua_gold_value(list(option), str(correct))
        if value is None or value <= 0 or not float(value).is_integer():
            continue
        if format_value(value) not in rationale:
            continue
        rows.append(
            Row(
                dataset=dataset,
                revision=revision,
                row_index=index,
                fingerprint=fingerprint,
                split=source_row_split(fingerprint, k_split, eval_permille=eval_permille),
                licence=licence,
                licence_tier=tier,
                text=question.strip(),
                secondary=rationale.strip(),
                value=value,
                bucket=magnitude_bucket(value),
            )
        )
    return rows


def load_apps_rows(corpus_root: Path, k_split: bytes, eval_permille: int) -> list[Row]:
    """Landed `apps` rows, fingerprinted and split-assigned.

    The committed fact for the `api_record` variant is the record's own declared
    ``problem_id`` -- the identity of the record turn 1 retrieved. It is present on every
    row (no fragile extraction), diverse across 10,000 values, and it is exactly what X7
    calls "a retrieved passage's key claim".

    Args:
        corpus_root: The corpus export root.
        k_split: The split key.
        eval_permille: Per-mille of source rows on the eval side.

    Returns:
        Rows.
    """
    import pyarrow.parquet as pq

    directory = corpus_root / APPS_REL
    manifest = read_manifest(directory)
    licence = str(manifest["license"])
    tier = licence_tier(licence)
    revision = str(manifest.get("revision") or "unpinned (MANIFEST.json declares none)")
    dataset = str(manifest["repo_id"])

    table = pq.read_table(directory / "train.parquet", columns=["problem_id", "question", "url"])
    ids = table.column("problem_id").to_pylist()
    questions = table.column("question").to_pylist()
    urls = table.column("url").to_pylist()

    rows: list[Row] = []
    for index, (problem_id, question, url) in enumerate(zip(ids, questions, urls, strict=True)):
        if not question or problem_id is None:
            continue
        # The fingerprint keys on the manifest's declared columns. `solutions` is read
        # lazily below only where a fingerprint is needed, to keep this pass cheap.
        rows.append(
            Row(
                dataset=dataset,
                revision=revision,
                row_index=index,
                fingerprint="",
                split="",
                licence=licence,
                licence_tier=tier,
                text=question.strip(),
                secondary=str(url or "").strip(),
                value=float(problem_id),
            )
        )

    solutions = pq.read_table(directory / "train.parquet", columns=["solutions"])
    solution_texts = solutions.column("solutions").to_pylist()
    assigned: list[Row] = []
    for row in rows:
        fingerprint = pair_fingerprint(row.text, str(solution_texts[row.row_index] or ""))
        assigned.append(
            Row(
                dataset=row.dataset,
                revision=row.revision,
                row_index=row.row_index,
                fingerprint=fingerprint,
                split=source_row_split(fingerprint, k_split, eval_permille=eval_permille),
                licence=row.licence,
                licence_tier=row.licence_tier,
                text=row.text,
                secondary=row.secondary,
                value=row.value,
                bucket=magnitude_bucket(row.value),
            )
        )
    return assigned


def render_passage_turn1(gold: Row, pool: list[Row], gold_index: int) -> tuple[str, str]:
    """Render the `passage_claim` commit turn: retrieve a passage, commit its claim.

    Args:
        gold: The row whose rationale is the one to retrieve.
        pool: The retrieval pool, gold included.
        gold_index: Where the gold sits in the pool.

    Returns:
        ``(query, key_claim)``.
    """
    lines = [
        "TURN 1 (retrieval). Query:",
        gold.text,
        "",
        f"Candidate passages ({len(pool)}):",
    ]
    for position, row in enumerate(pool):
        lines.append(f"  [{position}] {row.secondary}")
    lines += [
        "",
        "Which candidate passage resolves the query, and what value does its reasoning "
        "arrive at? Commit that value.",
    ]
    claim = f"passage [{gold_index}] resolves the query and arrives at {gold.value:g}"
    return "\n".join(lines), claim


def render_record_turn1(gold: Row, pool: list[Row], gold_index: int) -> tuple[str, str]:
    """Render the `api_record` commit turn: retrieve a record, commit its identity.

    Args:
        gold: The record to retrieve.
        pool: The retrieval pool, gold included.
        gold_index: Where the gold sits in the pool.

    Returns:
        ``(query, key_claim)``.
    """
    lines = [
        "TURN 1 (retrieval). Issue:",
        gold.text[:STATEMENT_CHARS],
        "",
        f"Candidate records ({len(pool)}):",
    ]
    for position, row in enumerate(pool):
        head = " ".join(row.text.split())[:120]
        lines.append(f"  [{position}] record #{int(row.value)} :: {head}")
    lines += [
        "",
        "Which candidate record is the one this issue describes? Commit its record number.",
    ]
    claim = f"record #{int(gold.value)} is the one the issue describes"
    return "\n".join(lines), claim


def render_turn2(probe: Row) -> str:
    """Render the probe turn, with turn 1's fact deferred behind a referent.

    Args:
        probe: The multi-hop source row.

    Returns:
        Turn 2's rendered text.
    """
    return (
        "TURN 2 (deferred premise). Let X be the quantity you committed in the previous "
        "turn. X is not restated here.\n\n"
        "Problem:\n"
        f"{probe.text}\n\n"
        "Let Y be the answer to that problem. Report X + Y."
    )


def columns_for(row: Row) -> tuple[str, str]:
    """The manifest-declared column pair the row's DEC-38 fingerprint was taken over.

    Args:
        row: The source row.

    Returns:
        The column pair.
    """
    return AQUA_COLUMNS if "aqua" in row.dataset.lower() else APPS_COLUMNS


def to_donor(row: Row, query: str, claim: str, pool: list[Row], gold_index: int) -> FactDonor:
    """Adapt a source row into a :class:`FactDonor`.

    Args:
        row: The source row.
        query: Turn 1's rendered text.
        claim: The key claim.
        pool: The retrieval pool.
        gold_index: Where the gold sits in the pool.

    Returns:
        The donor.
    """
    columns = columns_for(row)
    return FactDonor(
        source_dataset=row.dataset,
        source_revision=row.revision,
        row_index=row.row_index,
        pair_fingerprint=row.fingerprint,
        columns=columns,
        licence_tier=row.licence_tier,
        query=query,
        key_claim=claim,
        fact_value=row.value,
        pool_fingerprints=tuple(p.fingerprint for p in pool),
        pool_row_indices=tuple(p.row_index for p in pool),
        gold_pool_index=gold_index,
    )


def bare_donor(row: Row) -> FactDonor:
    """A counterfactual donor: an unrelated episode, referenced by its committed fact.

    Its own turn 1 is not rendered -- the negative control substitutes the *episode*, and
    what the item must record is which fact that episode would have committed.

    Args:
        row: The unrelated source row.

    Returns:
        The donor.
    """
    is_passage = "aqua" in row.dataset.lower()
    claim = (
        f"passage resolves its query and arrives at {row.value:g}"
        if is_passage
        else f"record #{int(row.value)} is the one the issue describes"
    )
    return FactDonor(
        source_dataset=row.dataset,
        source_revision=row.revision,
        row_index=row.row_index,
        pair_fingerprint=row.fingerprint,
        columns=columns_for(row),
        licence_tier=row.licence_tier,
        query="",
        key_claim=claim,
        fact_value=row.value,
    )


def stage_variant(
    variant: str,
    commit_rows: list[Row],
    probe_rows: list[Row],
    *,
    target: int,
    split: str,
    split_key: dict[str, Any],
    generator: dict[str, Any],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Stage one variant on one side of the split, until ``target`` items are admitted.

    Args:
        variant: ``"passage_claim"`` or ``"api_record"``.
        commit_rows: Rows eligible to be turn 1 (and its pool, and the donors).
        probe_rows: Rows eligible to be turn 2.
        target: How many admitted items to produce.
        split: The side being staged.
        split_key: The recorded split-key block.
        generator: The generator identity block.
        rng: Seeded RNG.

    Returns:
        ``(items, rejections)``.

    Raises:
        BuildRefusedError: If the pools are too small to stage the target at all.
    """
    needed_commit = POOL_SIZE + (N_OPTIONS - 1)
    bands: dict[int, list[Row]] = {}
    for row in commit_rows:
        bands.setdefault(row.bucket, []).append(row)
    usable = {b: rows for b, rows in bands.items() if len(rows) >= MIN_BUCKET_ROWS}
    if not usable or not probe_rows:
        raise BuildRefusedError(
            f"{variant}/{split}: pools too small (commit={len(commit_rows)} in "
            f"{len(bands)} bands, none with >= {MIN_BUCKET_ROWS} rows; "
            f"probe={len(probe_rows)})"
        )
    band_keys = sorted(usable)
    band_weights = [len(usable[b]) for b in band_keys]

    render_turn1 = render_passage_turn1 if variant == "passage_claim" else render_record_turn1
    rejections: Counter[str] = Counter()
    items: list[dict[str, Any]] = []
    seen_items: set[str] = set()
    probe_order = list(range(len(probe_rows)))
    rng.shuffle(probe_order)

    attempts = 0
    max_attempts = max(target * 40, 8192)
    cursor = 0
    while len(items) < target and attempts < max_attempts:
        attempts += 1
        probe = probe_rows[probe_order[cursor % len(probe_order)]]
        cursor += 1
        band = usable[rng.choices(band_keys, weights=band_weights, k=1)[0]]
        picks = rng.sample(range(len(band)), needed_commit)
        pool = [band[i] for i in picks[:POOL_SIZE]]
        donors = [band[i] for i in picks[POOL_SIZE:]]
        gold_index = rng.randrange(POOL_SIZE)
        gold = pool[gold_index]
        query, claim = render_turn1(gold, pool, gold_index)
        try:
            record = build_episode(
                to_donor(gold, query, claim, pool, gold_index),
                ProbeSource(
                    source_dataset=probe.dataset,
                    source_revision=probe.revision,
                    row_index=probe.row_index,
                    pair_fingerprint=probe.fingerprint,
                    columns=AQUA_COLUMNS,
                    licence_tier=probe.licence_tier,
                    question=render_turn2(probe),
                    answer_value=probe.value,
                ),
                [bare_donor(row) for row in donors],
                variant=variant,
                split=split,
                split_key=split_key,
                generator=generator,
            )
        except EpisodeRejectedError as exc:
            rejections[exc.reason] += 1
            continue
        if record["item_id"] in seen_items:
            continue
        seen_items.add(str(record["item_id"]))
        record["admission"] = {
            "status": "pending",
            "filter": "NSRS (design 5.3)",
            "blocked_by": ["W2b"],
            "note": "constructed, NOT admitted; s_r is GPU-blocked and blocked on W2b",
        }
        items.append(record)
    if len(items) < target:
        raise BuildRefusedError(
            f"{variant}/{split}: only {len(items)} of {target} items after "
            f"{attempts} attempts; rejections={dict(rejections)}"
        )
    return items, rejections


def source_sha256(paths: list[Path]) -> str:
    """SHA-256 over the generator's own source files, in a fixed order.

    Args:
        paths: The files that constitute the generator.

    Returns:
        64 hex characters.
    """
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def describe_source(directory: Path) -> dict[str, Any]:
    """Pin one landed corpus: its manifest identity plus the shard's own SHA-256.

    §5.6 asks for sources pinned by revision, not by name, because the fleet has measured
    that mirror metadata lies. Where a landed manifest declares no revision -- aqua_rat's
    does not -- the shard hash is what makes drift detectable, so it is recorded either
    way rather than only when the revision is missing.

    Args:
        directory: The corpus directory.

    Returns:
        The pin record.
    """
    manifest = read_manifest(directory)
    shard = directory / "train.parquet"
    digest = hashlib.sha256()
    with shard.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "repo_id": manifest["repo_id"],
        "revision": manifest.get("revision") or "unpinned (MANIFEST.json declares none)",
        "license": manifest["license"],
        "licence_tier": licence_tier(str(manifest["license"])),
        "declared_columns": manifest["columns"],
        "rows": manifest.get("rows"),
        "shard": str(shard),
        "shard_sha256": digest.hexdigest(),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    """Write records one per line, and return the file's SHA-256.

    Args:
        path: Destination.
        records: The records.

    Returns:
        64 hex characters.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def harness_interface() -> dict[str, Any]:
    """The episode-harness contract these items are built against, stated not assumed.

    Returns:
        The block written onto the manifest.
    """
    return {
        "owned_by": "the episode harness (W3's other half) -- NOT this generator",
        "per_episode_partition": (
            "derive_scope(principal, session=f'x7-{item_id}'). A fresh session per "
            "episode is the only partition reset the current store API offers: there is "
            "no clear/reset call on EpisodicStoreImpl, and stop() is terminal."
        ),
        "turn_boundary": (
            "one WhiteMatter.forward(inputs) per turn. forward reads the store at entry "
            "and writes at exit, so an episode is two successive forward() calls sharing "
            "one Scope and one domain, with distinct per-item logical_keys."
        ),
        "commit": (
            "turn 1's write must return a committed receipt before turn 2 is scored. The "
            "code's type is WriteReceipt(committed=True); 'LearnReceipt' in the design "
            "prose has no symbol in this repo."
        ),
        "read": (
            "turn 2's read is expected to be CONTENT-ADDRESSED by a query vector. On main "
            "today retrieve() takes no query and returns a scope-ranked bank, which is "
            "why phase A's read is a constant; the query parameter lands with "
            "fix/episodic-store-query-read. Content belongs in the query, never in the "
            "partition key."
        ),
        "scoring": (
            "turn 2 is scored as a closed 5-way choice over episode.turns[1].options; "
            "chance is 1/5 and is printed on every item."
        ),
        "negative_control": (
            "re-run the identical turn 2 after committing counterfactual.substituted_turn1"
            "'s fact instead, and require the item to be FAILED. The item names the option "
            "index that substitution must select."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Build the X7 reserve.

    Args:
        argv: Command-line arguments.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("/akula-data/csd/reserve/x7"))
    # The build record is committed; the items and the source-row ledger are not. `/data/`
    # is gitignored in this repo, so the record lands in the evidence tree beside the other
    # measured-result directories rather than at the design's notional `data/reserve/` path.
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "design" / "evidence" / "w3-x7-episodes-2026-09-07",
    )
    parser.add_argument("--train-per-pair", type=int, default=5120)
    parser.add_argument("--eval-per-pair", type=int, default=512)
    parser.add_argument("--eval-permille", type=int, default=DEFAULT_EVAL_PERMILLE)
    parser.add_argument("--fixture-size", type=int, default=16)
    args = parser.parse_args(argv)

    key_text = os.environ.get("CSD_K_SPLIT", "")
    if not key_text:
        raise SplitKeyMissingError(
            "CSD_K_SPLIT is unset. DEC-39: no key => refuse to build a split at all. "
            "Run under `secret exec CSD_K_SPLIT=akula/csd-k-split -- ...`."
        )
    k_split = key_text.strip().encode("utf-8")

    corpus_root = resolve_corpus_root(args.corpus_root)
    config = {
        "branch": "text-only-slip",
        "eval_permille": args.eval_permille,
        "pool_size": POOL_SIZE,
        "n_options": N_OPTIONS,
        "statement_chars": STATEMENT_CHARS,
        "build_seed": BUILD_SEED,
        "train_per_pair": args.train_per_pair,
        "eval_per_pair": args.eval_per_pair,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    generator = generator_identity(
        source_sha256(
            [Path(__file__).resolve(), REPO_ROOT / "src/cogsyndelta/reserve/episodes.py"]
        ),
        config,
    )
    split_key = {
        "scheme": SPLIT_KEY_SCHEME,
        "key_id": split_key_id(k_split),
        "key_source": "vault:akula/csd-k-split",
        "eval_permille": args.eval_permille,
        "granularity": "source row (DEC-38)",
    }

    print(f"corpus root: {corpus_root}", file=sys.stderr)
    aqua = load_aqua_rows(corpus_root, k_split, args.eval_permille)
    apps = load_apps_rows(corpus_root, k_split, args.eval_permille)
    print(f"aqua_rat usable+unburned: {len(aqua)}   apps: {len(apps)}", file=sys.stderr)

    by_split: dict[str, dict[str, list[Row]]] = {
        side: {
            "aqua": [r for r in aqua if r.split == side],
            "apps": [r for r in apps if r.split == side],
        }
        for side in ("train", "eval")
    }
    for side, pools in by_split.items():
        print(f"  {side}: aqua={len(pools['aqua'])} apps={len(pools['apps'])}", file=sys.stderr)

    all_items: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    rejections: dict[str, dict[str, int]] = {}
    for side, target in (("train", args.train_per_pair), ("eval", args.eval_per_pair)):
        for variant, commit_key in (("passage_claim", "aqua"), ("api_record", "apps")):
            # Reproducibility, not cryptography (ruff S311): pairing must be reproducible from the recorded seed, not secret.
            rng = random.Random(f"{BUILD_SEED}:{variant}:{side}")  # noqa: S311
            items, tally = stage_variant(
                variant,
                by_split[side][commit_key],
                by_split[side]["aqua"],
                target=target,
                split=side,
                split_key=split_key,
                generator=generator,
                rng=rng,
            )
            all_items[side].extend(items)
            rejections[f"{variant}/{side}"] = {r: tally.get(r, 0) for r in REJECTION_REASONS}
            print(
                f"  staged {variant}/{side}: {len(items)} admitted, {sum(tally.values())} rejected",
                file=sys.stderr,
            )

    artefacts = {}
    for side, items in all_items.items():
        path = args.out / f"x7-{side}.jsonl"
        artefacts[side] = {
            "path": str(path),
            "rows": len(items),
            "sha256": write_jsonl(path, items),
        }

    ledger_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for side, items in all_items.items():
        for item in items:
            for source in item["source_rows"]:
                key = (str(source["pair_fingerprint"]), side)
                if key in seen:
                    continue
                seen.add(key)
                ledger_rows.append(
                    {
                        "pair_fingerprint": source["pair_fingerprint"],
                        "source": source["dataset"],
                        "columns": source["columns"],
                        "split": side,
                        "shape": "X7",
                        "generator": generator["name"],
                    }
                )
    ledger_rows.sort(key=lambda r: (str(r["pair_fingerprint"]), str(r["split"])))
    ledger_path = args.out / "source-rows.jsonl"
    ledger_sha = write_jsonl(ledger_path, ledger_rows)

    train_fps = {r["pair_fingerprint"] for r in ledger_rows if r["split"] == "train"}
    eval_fps = {r["pair_fingerprint"] for r in ledger_rows if r["split"] == "eval"}
    leaked = train_fps & eval_fps

    turn_overlap = 0
    aqua_rows_used = 0
    total_rows_used = 0
    for items in all_items.values():
        for item in items:
            per_turn: dict[int, set[str]] = {0: set(), 1: set()}
            for source in item["source_rows"]:
                per_turn[int(source["turn"])].add(str(source["pair_fingerprint"]))
                total_rows_used += 1
                if "aqua" in str(source["dataset"]).lower():
                    aqua_rows_used += 1
            if per_turn[0] & per_turn[1]:
                turn_overlap += 1

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "item_schema": EPISODE_SCHEMA,
        "ledger_schema": LEDGER_SCHEMA,
        "shape": "X7",
        "branch": "text-only-slip (design 5.4: X8' replaces X8, R=4, 6 pairs, >=5 of 6, null 0.109)",
        "bin": "episodic_store x memory",
        "ablation_pairs": ["episodic_store x memory", "episodic_store x language"],
        "variants": {
            "passage_claim": "{episodic_store, memory} -- turn 1 retrieves an aqua_rat passage",
            "api_record": "{episodic_store, language} -- turn 1 retrieves an apps record",
        },
        "config": config,
        "generator": generator,
        "split_key": split_key,
        "corpus_root": str(corpus_root),
        "sources": {
            "aqua_rat": describe_source(corpus_root / AQUA_REL),
            "apps": describe_source(corpus_root / APPS_REL),
        },
        "sources_not_consumed": {
            "deepmind/code_contests": (
                "landed and reserved for compose, but its MANIFEST-declared fingerprint "
                "columns are ('description','solutions') and `solutions` is most of a "
                "19 GB shard. Fingerprinting it at DEC-38 granularity needs a streaming "
                "reader this row did not build, so X7 draws its code half from `apps` "
                "only. This is a stated limitation, not an oversight."
            )
        },
        "counts": {
            "train": len(all_items["train"]),
            "eval": len(all_items["eval"]),
            "per_variant_per_split": {
                f"{variant}/{side}": sum(
                    1 for item in all_items[side] if item["variant"] == variant
                )
                for side in ("train", "eval")
                for variant in ("passage_claim", "api_record")
            },
            "aqua_usable_unburned": len(aqua),
            "apps_usable": len(apps),
        },
        "rejections": rejections,
        "rejection_totals": {
            reason: sum(tally.get(reason, 0) for tally in rejections.values())
            for reason in REJECTION_REASONS
        },
        "dec38": {
            "granularity": "source row",
            "episodes_with_shared_turn_rows": turn_overlap,
            "source_rows_on_both_sides": len(leaked),
            "unique_source_rows": len(train_fps | eval_fps),
        },
        "balance": {
            "aqua_rat_source_row_share": round(aqua_rows_used / max(total_rows_used, 1), 4),
            "b1_hard_line": 0.50,
            "b1_operating_cap": 0.40,
            "note": (
                "B1 binds on the reserve as a whole, not on one shape; this is X7's own "
                "contribution to that share, printed so it can be summed."
            ),
            "not_a_b1_verdict": (
                "Do NOT read this local share as a B1 violation. B1 is computed over the "
                "reserve's source rows, and the reserve-wide figure is itself under "
                "re-accounting: W3's scope pass measured 87,599 unaccounted TRAIN_OK rows "
                "in `squad`, which would move the reserve-wide aqua_rat share from the "
                "design's printed 79.5% to roughly 43% -- below both the 0.50 hard line "
                "and the 0.40 operating cap. The verdict belongs to that accounting, not "
                "to this shape."
            ),
        },
        "artifacts": artefacts,
        "ledger": {"path": str(ledger_path), "rows": len(ledger_rows), "sha256": ledger_sha},
        "harness_interface": harness_interface(),
        "not_done": [
            "the episode harness (W3's other half)",
            "NSRS admission (s_r): GPU-blocked, blocked on W2b",
            "X8' items (the other pre-committed slip-branch shape)",
        ],
    }
    manifest_path = args.record_dir / "MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fixture = all_items["eval"][: args.fixture_size]
    write_jsonl(REPO_ROOT / "tests" / "fixtures" / "x7_episodes_sample.jsonl", fixture)

    print(json.dumps(manifest["counts"], indent=2))
    print(json.dumps(manifest["rejection_totals"], indent=2))
    if leaked or turn_overlap:
        print("REFUSED: DEC-38 violated", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
