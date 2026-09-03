#!/usr/bin/env python3
"""W2a: burn the UNION of every candidate `aqua_rat` draw this script can find, and
refuse a ledger that omits any of them.

WHY THIS SCRIPT EXISTS (DEC-42, docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md §5.1/§4.1)
`reason` capped `deepmind/aqua_rat` to 4,982 of 97,467 rows (scripts/csd-train-all.py's
`REGIONS["reason"]`), and its receipt -- the only record of WHICH 4,982 -- was deleted.
`c42203c` changed the cap's sampling method mid-programme, from a prefix (the first N rows
in shard order) to a seeded reservoir sample (Algorithm R, uniform over the whole source).
Both are deterministic and reproducible from config alone, so re-running EITHER one
recovers a candidate draw exactly -- but nothing on disk says which method the `reason`
run actually used.

ORCHESTRATOR DECISION (2026-09-03, superseding the earlier "measure but do not burn"
posture): this corpus is re-acquirable and contamination is burned conservatively. The
ledger burns the UNION OF EVERY CANDIDATE DRAW it can find -- the reservoir draw R, the
prefix draw P, AND every on-disk derived-sample file under the aqua_rat source tree whose
row count matches `cap` (a file some other run may have produced by some other sampling
method, exactly the way the first such file -- `derived/sample-4982-seed0.parquet` --
was found dated 27 minutes before `c42203c` and 36 minutes before the cap literal was
even added to `scripts/csd-train-all.py`, a strong candidate for what `reason` actually
consumed). Each discovered draw is recorded with its provenance -- path, mtime, sha256,
row count -- and the earlier "measure it, ask an operator, maybe refuse to certify"
machinery is gone: discovery is unconditional, burning is unconditional, and the only
thing that can still stop a clean exit is a discovered file this script could not READ
(see "EXIT CODE 3" below) -- not one it read and chose not to trust.

THE GATE, AND WHY IT IS COUNTED IN ROWS, NOT UNIQUE FINGERPRINTS
aqua_rat contains duplicate-content rows -- a real seed=0 reservoir draw of 4,982 rows
carries only 4,951 UNIQUE `pair_fingerprint`s, 31 rows sharing a fingerprint with another
row in the SAME draw. `|fingerprints(draw)| = cap` is therefore false for a correct draw
and can never be satisfied -- a guard that can never pass is not a guard, it is a no-op
with a name (see tests/test_guards_can_fail.py's own thesis). This script implements the
row-count reading: for every one of a draw's rows, is that row's OWN fingerprint present
in the ledger. `verify_gate` does this for R and P; `verify_gate_for_draws` generalises it
to an arbitrary number of named draws (used for every discovered on-disk draw) -- a ledger
that omits ANY discovered draw entirely (0 rows covered) fails exactly the same way as one
that covers it only in part (tests/test_reserve_ledger.py builds exactly this and asserts
the refusal).

WHAT THIS SCRIPT DOES NOT DO
It does not modify `regions/pretrain.py` or `scripts/csd-train-all.py` -- the reservoir
fix already landed (c42203c) and stays as production's only sampling path; this script
only re-derives what a vanished receipt would have recorded, plus whatever else is
sitting on disk that could have been consumed instead.

EXIT CODE 3: A DISCOVERED FILE THIS SCRIPT COULD NOT READ
`discover_ondisk_draws` globs `DERIVED_DRAWS_GLOB` under the corpus root. A matched file
that reads cleanly but has a row count other than `cap` is not a candidate for this cap
and is silently skipped -- it is a file this cap has nothing to do with. A matched file
that FAILS to read (corrupt/partial parquet, permissions, ...) is different: its row
count and content are UNKNOWN, so this script cannot certify it as burned OR dismiss it
as irrelevant. The ledger is still written from everything that WAS readable (for
review), but `main` prints an unmissable banner and returns exit `3` instead of `0` --
the only remaining "not fully certified" outcome this script has. There is no
acknowledgement flag to suppress it: an unreadable file is a fix-the-file-and-rerun
problem (or a --corpus-root problem), not an operator judgement call the way "is this
decoy relevant" used to be.

Usage:
    scripts/csd-reserve-ledger.py                       # write data/reserve/burned-aqua_rat.jsonl
    scripts/csd-reserve-ledger.py --corpus-root /path    # override root discovery
    scripts/csd-reserve-ledger.py --out /tmp/x.jsonl     # write elsewhere (tests use this)

Exit codes: 0 ok; 1 REFUSED (a DEC-42 check failed -- see stderr); 2 corpus root or
shards not found; 3 the ledger WAS written but at least one discovered on-disk candidate
draw could not be read and so could not be certified burned -- see stderr and the
manifest's `unreadable_ondisk_draws`.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any

REGION = "reason"
SOURCE_ID = "deepmind/aqua_rat"
SOURCE_GLOB = "reason/aqua_rat-raw/train.parquet"
"""Matches `REGIONS["reason"]`'s aqua_rat `SourceSpec` glob in scripts/csd-train-all.py
verbatim (line ~312 there) -- this script must resolve the SAME shards production does,
not a path guessed at the call site."""

DERIVED_DRAWS_GLOB = "reason/aqua_rat-raw/derived/sample*.parquet"
"""Every on-disk derived-sample file that could plausibly be a prior draw at this cap --
not just the one file the review happened to find first. See `discover_ondisk_draws`."""

COLUMNS: tuple[str, str] = ("question", "rationale")
CAP = 4982
DEFAULT_SEED = 0

CANDIDATE_CORPUS_ROOTS: tuple[Path, ...] = (
    Path("/bulk/csd-corpus"),  # gpu5080, where the corpus actually lives
    Path("/mnt/bulk/csd-corpus"),  # the same filesystem, NFS-mounted on akula-prime
)
"""`scripts/csd-train-all.py`'s `REGION_CORPUS_ROOT["reason"]` is `/bulk/csd-corpus`
unconditionally -- that constant is correct only ON gpu5080. This script also runs from
akula-prime, where the identical export is mounted at `/mnt/bulk` (verified: `findmnt
/mnt/bulk` reports `192.168.1.251:/bulk`, i.e. gpu5080's `/bulk`), so both roots are
tried, in the order a training run would actually resolve first, before giving up."""

DEFAULT_LEDGER_PATH = Path("data/reserve/burned-aqua_rat.jsonl")

RECEIPTS_DIR = Path("/akula-data/csd/receipts")

C42203C_SHA = "c42203c924e7133276fb3d837ce4c891a706d578"
C42203C_COMMIT_TIME_UTC = "2026-09-02T23:32:52+00:00"
"""`git show -s --format='%H %ci' c42203c` -> `2026-09-02 19:32:52 -0400`, converted to
UTC. The DELIBERATE correctness change (prefix -> reservoir sampling) that makes which
draw the `reason` run used ambiguous; see the module docstring."""


class LedgerGateError(ValueError):
    """Raised when a ledger does not cover every row of every draw -- R, P, and every
    discovered on-disk draw. A ledger built from a strict subset of the known draws is a
    FAIL, not a partial pass -- construct one and confirm this fires; see
    tests/test_reserve_ledger.py.
    """


class SeedInferenceError(ValueError):
    """Raised when surviving receipts DISAGREE on `config.seed` -- DEC-42 check (iii).

    Section 5.1: "If (i), (ii) or (iii) fails, the write-off stands and this section
    reverts." Absence of any surviving receipt is a separate, non-blocking case (mirrors
    check (iv)'s "record it, don't assume it" -- see `infer_seed`'s docstring); this is
    raised only when receipts exist and contradict each other, which means the uniform
    `cfg.seed` assumption `infer_seed` and every other check leans on is false.
    """


def resolve_corpus_root(explicit: Path | None = None) -> Path:
    """Find the aqua_rat corpus root, trying `explicit` then each candidate in order.

    Raises:
        FileNotFoundError: none of the candidates (or `explicit`) is a directory.
    """
    if explicit is not None:
        if not explicit.is_dir():
            raise FileNotFoundError(f"--corpus-root {explicit} is not a directory")
        return explicit
    for candidate in CANDIDATE_CORPUS_ROOTS:
        if candidate.is_dir():
            return candidate
    tried = ", ".join(str(c) for c in CANDIDATE_CORPUS_ROOTS)
    raise FileNotFoundError(
        f"none of the candidate corpus roots exist: {tried} -- mount gpu5080's "
        f"/bulk/csd-corpus directly, or its NFS export at /mnt/bulk (akula-prime), or "
        f"pass --corpus-root explicitly."
    )


def locate_shards(root: Path) -> list[str]:
    """Exactly `csd-train-all.py`'s `_shards(SOURCE_GLOB, root)`: `sorted(root.glob(...))`."""
    return sorted(str(p) for p in root.glob(SOURCE_GLOB))


def draw_reservoir(
    shards: list[str], seed: int = DEFAULT_SEED, cap: int = CAP
) -> list[tuple[str, str]]:
    """Draw R: today's production `load_pairs`, unmodified -- reservoir_sample seeded by
    `sampling_rng(seed, shards, columns)`, exactly the call `pretrain_region` makes."""
    from cogsyndelta.regions.pretrain import load_pairs

    return load_pairs(shards, COLUMNS, cap, seed=seed)


def draw_prefix(shards: list[str], cap: int = CAP) -> list[tuple[str, str]]:
    """Draw P: the first `CAP` pairs `_iter_pairs` yields in shard order -- what
    `load_pairs` returned before `c42203c` rewrote it (`git show c42203c` confirms the
    pre-fix body was the identical iterate-and-filter loop, just returning at `len ==
    limit` instead of continuing). Reuses today's `_iter_pairs` directly rather than a
    second, hand-written copy of its filtering rule (drop pairs with an empty side) --
    two independently maintained "iterate the source" implementations are exactly how a
    guard ends up checking something other than what it claims to."""
    from cogsyndelta.regions.pretrain import _iter_pairs

    return list(itertools.islice(_iter_pairs(shards, COLUMNS), cap))


def fingerprint_pairs(pairs: list[tuple[str, str]]) -> list[str]:
    from cogsyndelta.eval import pair_fingerprint

    return [pair_fingerprint(a, b) for a, b in pairs]


def verify_gate(
    ledger_fps: set[str],
    draw_r: list[tuple[str, str]],
    draw_p: list[tuple[str, str]],
    cap: int = CAP,
) -> tuple[int, int]:
    """DEC-42's gate for R and P specifically: every row of BOTH must have its
    fingerprint in the ledger. Counted in ROWS, not unique fingerprints -- see the module
    docstring's "THE GATE" section for why the unique-fingerprint reading can never pass
    on real data. Discovered on-disk draws are gated separately by
    `verify_gate_for_draws`, since there can be any number of them (R and P are always
    exactly two, and always present, so this function keeps their simpler fixed-arity
    signature).

    Returns:
        `(rows of draw_r covered, rows of draw_p covered)`.

    Raises:
        LedgerGateError: either count is short of `cap`.
    """
    from cogsyndelta.eval import pair_fingerprint

    covered_r = sum(1 for a, b in draw_r if pair_fingerprint(a, b) in ledger_fps)
    covered_p = sum(1 for a, b in draw_p if pair_fingerprint(a, b) in ledger_fps)
    if covered_r != cap or covered_p != cap:
        raise LedgerGateError(
            f"ledger covers {covered_r}/{cap} rows of draw R and {covered_p}/{cap} rows "
            f"of draw P -- DEC-42 requires BOTH in full. A ledger built from one draw "
            f"alone is a FAIL, not a partial pass; refusing to certify it."
        )
    return covered_r, covered_p


def verify_gate_for_draws(
    ledger_fps: set[str], named_draws: dict[str, list[tuple[str, str]]]
) -> dict[str, int]:
    """Generalisation of `verify_gate` to an arbitrary number of named draws -- used for
    every discovered on-disk draw (each named by its path relative to the corpus root),
    in addition to R and P. Every draw's every row's fingerprint must be present in
    `ledger_fps`, counted in ROWS exactly the way `verify_gate` counts R and P -- so a
    draw that is OMITTED from `ledger_fps` entirely (0 rows covered) fails exactly the
    same way as one covered only in part. THE falsifier this function exists to satisfy:
    hand it a ledger that is missing one discovered draw and confirm it refuses (see
    tests/test_reserve_ledger.py).

    Returns:
        `{draw_name: rows_covered}` for every draw in `named_draws`.

    Raises:
        LedgerGateError: any draw's coverage is short of its own row count.
    """
    from cogsyndelta.eval import pair_fingerprint

    covered = {
        name: sum(1 for a, b in pairs if pair_fingerprint(a, b) in ledger_fps)
        for name, pairs in named_draws.items()
    }
    short = {
        name: (covered[name], len(pairs))
        for name, pairs in named_draws.items()
        if covered[name] != len(pairs)
    }
    if short:
        detail = "; ".join(f"{name!r}: {c}/{n} rows" for name, (c, n) in sorted(short.items()))
        raise LedgerGateError(
            f"ledger omits rows from {len(short)} discovered on-disk draw(s): {detail} "
            f"-- a ledger that omits any discovered draw is a FAIL, not a partial pass; "
            f"refusing to certify it."
        )
    return covered


def _iso_utc(mtime: float) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(mtime, tz=_dt.UTC).isoformat()


def _epoch(iso: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(iso).timestamp()


def establish_code_revision(
    receipts_dir: Path = RECEIPTS_DIR, region: str = REGION
) -> dict[str, Any]:
    """DEC-42 check (iv), the one that can fail: which code revision produced the
    `region` run. Looks for a receipt (`{region}-*.json`) or a checkpoint
    (`{region}-checkpoints/`) under `receipts_dir` and compares its mtime against
    `C42203C_COMMIT_TIME_UTC`. The caller burns the union regardless of this function's
    result -- check (iv) is the one check DEC-42 explicitly allows to proceed
    unestablished. (Discovered on-disk draws' own provenance -- including their mtimes --
    is recorded separately, under `ondisk_draws` in the manifest; this function only
    covers `region`'s own receipt/checkpoint evidence.)
    """
    c42203c_epoch = _epoch(C42203C_COMMIT_TIME_UTC)
    receipts = sorted(receipts_dir.glob(f"{region}-*.json"))
    checkpoints_dir = receipts_dir / f"{region}-checkpoints"
    checkpoints = sorted(checkpoints_dir.glob("*")) if checkpoints_dir.is_dir() else []
    evidence: list[dict[str, Any]] = []
    for p in [*receipts, *checkpoints]:
        mtime = p.stat().st_mtime
        evidence.append(
            {
                "kind": "reason_receipt" if p in receipts else "reason_checkpoint",
                "path": str(p),
                "mtime_utc": _iso_utc(mtime),
                "before_c42203c": mtime < c42203c_epoch,
            }
        )
    if not evidence:
        return {
            "established": False,
            "note": (
                f"no {region!r} receipt under {receipts_dir} ({region}-*.json) and no "
                f"{checkpoints_dir} directory -- the {region} receipt was deleted (see "
                f"W1b) and nothing else on disk dates the run. The code revision the "
                f"{region} run used cannot be determined from artefacts on disk; "
                f"recorded as such rather than assumed."
            ),
            "c42203c_sha": C42203C_SHA,
            "c42203c_commit_time_utc": C42203C_COMMIT_TIME_UTC,
            "evidence": [],
        }
    return {
        "established": True,
        "note": (
            f"evidence exists bearing on {region!r}'s code revision -- see `evidence` "
            f"for each artefact's `kind` and mtime against c42203c; a human still has "
            f"to weigh whether it settles which sampling method ran (a "
            f"preserved-after-the-fact commit does not date the run that produced it "
            f"-- §5.1). `established: True` here means evidence exists to weigh, NOT "
            f"that the question is settled."
        ),
        "c42203c_sha": C42203C_SHA,
        "c42203c_commit_time_utc": C42203C_COMMIT_TIME_UTC,
        "evidence": evidence,
    }


def infer_seed(receipts_dir: Path = RECEIPTS_DIR, expected: int = DEFAULT_SEED) -> dict[str, Any]:
    """DEC-42 check (iii): whether the run used `seed=0`. `[I]` for `reason` specifically
    -- its own receipt is the deleted one -- inferred from every SURVIVING receipt's
    `config.seed` (read live off disk, not assumed) plus `csd-train-all.py` passing
    `cfg.seed` uniformly across every region (no per-region override exists in
    `REGIONS`)."""
    seeds: dict[str, Any] = {}
    for p in sorted(receipts_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        seed = data.get("config", {}).get("seed") if isinstance(data, dict) else None
        if seed is not None:
            seeds[p.name] = seed
    all_agree = bool(seeds) and all(v == expected for v in seeds.values())
    return {
        "assumed_seed": expected,
        "basis": (
            "no surviving `reason` receipt to read a seed off directly; inferred from "
            "every OTHER surviving receipt's config.seed plus csd-train-all.py passing "
            "cfg.seed uniformly across regions"
        ),
        "surviving_receipts_seed": seeds,
        "all_surviving_receipts_agree": all_agree,
    }


def discover_ondisk_draws(
    root: Path, cap: int = CAP
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Globs `DERIVED_DRAWS_GLOB` under `root` for every on-disk derived-sample parquet
    file next to the aqua_rat source shard, reads each, and returns `(draws,
    unreadable)`:

    - `draws`: every matched file that could be READ and has EXACTLY `cap` raw rows --
      each a dict with `name` (path relative to `root`, used as the draw's identity),
      `path`, `mtime_utc`, `sha256`, `row_count`, and `pairs` (non-empty (question,
      rationale) pairs, filtered the same way `_iter_pairs`/`load_pairs` filter every
      other draw in this script -- a row with an empty side was never a candidate row
      for any draw).
    - `unreadable`: every matched file that RAISED while being read (corrupt/partial
      parquet, permissions, ...) -- `path`, `mtime_utc` (when `stat` itself succeeded),
      `error`. This file's content is UNKNOWN: this script can neither certify it as
      burned nor dismiss it as irrelevant, so it is never silently skipped -- see
      `main`'s exit code 3.

    A matched file that reads cleanly but has a row count OTHER than `cap` is neither a
    draw nor unreadable -- it is a file this cap has nothing to do with (a different
    run's different-sized sample, say), and is skipped without comment. Discovery is
    scoped to the `reason` region's aqua_rat cap under DEC-42, not a general audit of
    the `derived/` directory.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    draws: list[dict[str, Any]] = []
    unreadable: list[dict[str, Any]] = []
    for path in sorted(root.glob(DERIVED_DRAWS_GLOB)):
        rel = str(path.relative_to(root))
        try:
            mtime = path.stat().st_mtime
        except OSError as exc:
            unreadable.append({"path": rel, "mtime_utc": None, "error": str(exc)})
            continue
        try:
            table = pq.read_table(path, columns=list(COLUMNS))
            raw_bytes = path.read_bytes()
        except (OSError, pa.ArrowException) as exc:
            unreadable.append({"path": rel, "mtime_utc": _iso_utc(mtime), "error": str(exc)})
            continue
        row_count = table.num_rows
        if row_count != cap:
            continue
        col_a = table.column(COLUMNS[0]).to_pylist()
        col_b = table.column(COLUMNS[1]).to_pylist()
        pairs = [(a, b) for a, b in zip(col_a, col_b, strict=True) if a and b]
        draws.append(
            {
                "name": rel,
                "path": rel,
                "mtime_utc": _iso_utc(mtime),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "row_count": row_count,
                "pairs": pairs,
            }
        )
    return draws, unreadable


def build_ledger(
    shards: list[str],
    *,
    seed: int = DEFAULT_SEED,
    cap: int = CAP,
    receipts_dir: Path = RECEIPTS_DIR,
    corpus_root: Path | None = None,
) -> dict[str, Any]:
    """Compute R, P, and every discovered on-disk draw under `corpus_root` (when given),
    union all of their fingerprints, and enforce DEC-42's gate against every one of them.

    Raises:
        RuntimeError: R or P is not reproducible across two computations (DEC-42
            check (ii)).
        SeedInferenceError: surviving receipts disagree on `config.seed` (DEC-42 check
            (iii)) -- see `SeedInferenceError`'s docstring for why absence of receipts
            is a different, non-blocking case.
        LedgerGateError: the union does not cover every row of R, P, or any discovered
            on-disk draw AT THE REQUESTED `cap` (for R and P) or at its own row count
            (for a discovered draw) -- gated against the requested target, not against
            however many rows a draw actually returned, so a short/truncated/partial
            draw (bad mount, corrupt shard, wrong --corpus-root) cannot self-certify by
            shrinking its own target.
    """
    from cogsyndelta.corpus import CORPUS_FINGERPRINT_SCHEME, fingerprint_corpus
    from cogsyndelta.eval import pair_fingerprint

    # DEC-42 check (ii): each draw must be reproducible across two computations.
    r_first = draw_reservoir(shards, seed, cap)
    r_second = draw_reservoir(shards, seed, cap)
    if r_first != r_second:
        raise RuntimeError(
            "draw R (reservoir_sample+sampling_rng) returned different rows on two "
            "consecutive computations -- it is supposed to be deterministic from "
            "(seed, shard basenames, columns) alone. Refusing to build a ledger from a "
            "non-reproducible draw."
        )
    p_first = draw_prefix(shards, cap)
    p_second = draw_prefix(shards, cap)
    if p_first != p_second:
        raise RuntimeError(
            "draw P (_iter_pairs prefix) returned different rows on two consecutive "
            "computations -- parquet iteration is supposed to be order-stable. Refusing "
            "to build a ledger from a non-reproducible draw."
        )

    # DEC-42 check (iii): surviving receipts must not disagree on `config.seed`. Absence
    # of any surviving receipt is NOT the same as disagreement -- see
    # `SeedInferenceError`'s docstring -- so this only raises when there IS conflicting
    # evidence, exactly the case DEC-42 says must revert the write-off.
    seed_info = infer_seed(receipts_dir, expected=seed)
    if seed_info["surviving_receipts_seed"] and not seed_info["all_surviving_receipts_agree"]:
        raise SeedInferenceError(
            f"surviving receipts disagree on config.seed: "
            f"{seed_info['surviving_receipts_seed']!r} -- DEC-42 check (iii) requires "
            f"they agree before the uniform seed={seed!r} assumption behind draw R can "
            f"be trusted. Refusing to build a ledger on a contradicted assumption."
        )

    draws, unreadable = (
        discover_ondisk_draws(corpus_root, cap) if corpus_root is not None else ([], [])
    )

    fps_r = [pair_fingerprint(a, b) for a, b in r_first]
    fps_p = [pair_fingerprint(a, b) for a, b in p_first]
    set_r, set_p = set(fps_r), set(fps_p)
    ondisk_sets: dict[str, set[str]] = {
        d["name"]: set(fingerprint_pairs(d["pairs"])) for d in draws
    }

    union = set_r | set_p
    for s in ondisk_sets.values():
        union |= s

    rows = []
    for fp in sorted(union):
        also_in_ondisk = sorted(name for name, s in ondisk_sets.items() if fp in s)
        if fp in set_r and fp in set_p:
            draw_tag = "both"
        elif fp in set_r:
            draw_tag = "R"
        elif fp in set_p:
            draw_tag = "P"
        else:
            draw_tag = "ondisk"
        rows.append(
            {
                "pair_fingerprint": fp,
                "draw": draw_tag,
                "also_in_ondisk_draws": also_in_ondisk,
                "region": REGION,
                "source": SOURCE_ID,
            }
        )

    # Gated against the REQUESTED cap (R, P) or the discovered draw's own row count, not
    # against however many rows a draw actually returned -- see this function's
    # docstring's Raises: section.
    covered_r, covered_p = verify_gate(union, r_first, p_first, cap=cap)
    ondisk_covered = verify_gate_for_draws(union, {d["name"]: d["pairs"] for d in draws})

    corpus_fp = fingerprint_corpus(shards, columns=COLUMNS)

    manifest: dict[str, Any] = {
        "region": REGION,
        "source": SOURCE_ID,
        "shards": shards,
        "columns": list(COLUMNS),
        "cap": cap,
        "seed": seed_info,
        "corpus_fingerprint": corpus_fp,
        "corpus_fingerprint_scheme": CORPUS_FINGERPRINT_SCHEME,
        "draw_r_rows": len(r_first),
        "draw_p_rows": len(p_first),
        "draw_r_unique_fingerprints": len(set_r),
        "draw_p_unique_fingerprints": len(set_p),
        "ondisk_draws": [
            {
                "name": d["name"],
                "path": d["path"],
                "mtime_utc": d["mtime_utc"],
                "sha256": d["sha256"],
                "row_count": d["row_count"],
                "rows_non_empty_pairs": len(d["pairs"]),
                "rows_unique_fingerprints": len(ondisk_sets[d["name"]]),
                "rows_covered": ondisk_covered[d["name"]],
            }
            for d in draws
        ],
        "unreadable_ondisk_draws": unreadable,
        "union_size": len(union),
        "intersection_r_p_size": len(set_r & set_p),
        "code_revision": establish_code_revision(receipts_dir, region=REGION),
        "gate": {
            "cap": cap,
            "draw_r_rows_covered": covered_r,
            "draw_p_rows_covered": covered_p,
            "ondisk_draws_covered": ondisk_covered,
            "passed": True,
        },
    }

    return {"rows": rows, "manifest": manifest}


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def write_ledger(out_path: Path, result: dict[str, Any]) -> Path:
    """Writes `out_path` (the fingerprint rows, one JSON object per line, sorted by
    fingerprint) and a sibling `<out_path>.manifest.json` (everything else DEC-42's
    checks produce). Deterministic given the same corpus and code -- no wall-clock field
    is written to either file -- so running this twice produces byte-identical output;
    that is what makes the ledger idempotent rather than merely re-derivable.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_dumps(row) for row in result["rows"]]
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(result["manifest"], sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--corpus-root", type=Path, default=None, help="override corpus root discovery"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cap", type=int, default=CAP, help="rows per draw (default: production's 4,982)"
    )
    parser.add_argument("--receipts-dir", type=Path, default=RECEIPTS_DIR)
    args = parser.parse_args(argv)

    try:
        root = resolve_corpus_root(args.corpus_root)
    except FileNotFoundError as exc:
        print(f"csd-reserve-ledger: {exc}", file=sys.stderr)
        return 2
    print(f"corpus root: {root}", file=sys.stderr)

    shards = locate_shards(root)
    if not shards:
        print(
            f"csd-reserve-ledger: no shards matched {SOURCE_GLOB!r} under {root}", file=sys.stderr
        )
        return 2
    print(f"shards: {shards}", file=sys.stderr)

    try:
        result = build_ledger(
            shards, seed=args.seed, cap=args.cap, receipts_dir=args.receipts_dir, corpus_root=root
        )
    except (RuntimeError, LedgerGateError, SeedInferenceError) as exc:
        print(f"csd-reserve-ledger: REFUSED -- {exc}", file=sys.stderr)
        return 1

    manifest_path = write_ledger(args.out, result)
    m = result["manifest"]
    n_draws_total = 2 + len(m["ondisk_draws"])
    ondisk_summary = "\n".join(
        f"  ondisk draw {d['name']!r}: {d['rows_covered']}/{d['rows_non_empty_pairs']} rows "
        f"covered ({d['row_count']} raw rows, sha256={d['sha256'][:12]}...)"
        for d in m["ondisk_draws"]
    )
    print(
        f"wrote {len(result['rows'])} burned fingerprints to {args.out} "
        f"(union of {n_draws_total} draw(s); <= {m['cap'] * n_draws_total} of 97,467 "
        f"possible)\n"
        f"  draw R: {m['draw_r_rows']} rows ({m['draw_r_unique_fingerprints']} unique)\n"
        f"  draw P: {m['draw_p_rows']} rows ({m['draw_p_unique_fingerprints']} unique)\n"
        + (ondisk_summary + "\n" if ondisk_summary else "")
        + f"  union: {m['union_size']}  R∩P: {m['intersection_r_p_size']}\n"
        f"  gate: covered {m['gate']['draw_r_rows_covered']}/{m['gate']['cap']} (R), "
        f"{m['gate']['draw_p_rows_covered']}/{m['gate']['cap']} (P)\n"
        f"  code revision established: {m['code_revision']['established']}\n"
        f"  manifest: {manifest_path}",
        file=sys.stderr,
    )

    if m["unreadable_ondisk_draws"]:
        print(
            "\n"
            "=========================================================================\n"
            "csd-reserve-ledger: NOT FULLY CERTIFIED -- UNREADABLE ON-DISK DRAW(S)\n"
            "=========================================================================\n"
            f"{len(m['unreadable_ondisk_draws'])} file(s) matched {DERIVED_DRAWS_GLOB!r} "
            "but could not be read, so their content is UNKNOWN and they were NOT "
            "burned:\n"
            + "\n".join(f"  {u['path']}: {u['error']}" for u in m["unreadable_ondisk_draws"])
            + "\n"
            "The ledger above WAS written from everything that COULD be read, but this "
            "run is not treated as final. Fix the file (or the --corpus-root) and rerun.\n"
            "=========================================================================",
            file=sys.stderr,
        )
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
