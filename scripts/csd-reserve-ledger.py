#!/usr/bin/env python3
"""W2a: burn the UNION of both possible `aqua_rat` draws, and refuse a ledger that omits
either one.

WHY THIS SCRIPT EXISTS (DEC-42, docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md §5.1/§4.1)
`reason` capped `deepmind/aqua_rat` to 4,982 of 97,467 rows (scripts/csd-train-all.py's
`REGIONS["reason"]`), and its receipt -- the only record of WHICH 4,982 -- was deleted.
`c42203c` changed the cap's sampling method mid-programme, from a prefix (the first N rows
in shard order) to a seeded reservoir sample (Algorithm R, uniform over the whole source).
Both methods are deterministic and reproducible from config alone, so re-running EITHER
one recovers a candidate draw exactly -- but nothing on disk says which method the
`reason` run actually used, and a ledger built from only one candidate risks certifying as
"clean" the 4,982 rows that were genuinely trained on (DEC-22: no region already has the
answer memorised). DEC-42's answer is to stop guessing: burn `fingerprints(R) UNION
fingerprints(P)`, at most 9,964 of 97,467, and DERIVE the two draws so this recovery does
not depend on a receipt field that was never written (`corpus.cap_sampling` -- verified
absent from every one of the nine surviving receipts under /akula-data/csd/receipts/).

THE GATE, AND WHY IT IS COUNTED IN ROWS, NOT UNIQUE FINGERPRINTS
The design doc states the gate two ways: the summary row as "count(ledger INTERSECT draw
R) = 4,982" and the DEC-42 block as "|ledger INTERSECT fingerprints(R)| = 4,982". Measured
against the real corpus (see the integration test), these are NOT the same check: aqua_rat
contains duplicate-content rows -- a real seed=0 reservoir draw of 4,982 rows carries only
4,951 UNIQUE `pair_fingerprint`s, 31 rows sharing a fingerprint with another row in the
SAME draw. `|fingerprints(R)| = 4,982` is therefore false for a correct draw and true for
no draw at all -- a guard that can never pass is not a guard, it is a no-op with a name
(see tests/test_guards_can_fail.py's own thesis). This script implements the row-count
reading: for every one of a draw's `CAP` rows, is that row's OWN fingerprint present in
the ledger. That is exactly "no row of ambiguous provenance is ever certified clean" --
duplicate content inside one draw does not change whether each row's provenance is
covered -- and it is FALSIFIABLE the way the doc demands: hand it a ledger built from one
draw alone and every row of the OTHER draw whose fingerprint isn't already coincidentally
covered fails the count (tests/test_reserve_ledger.py builds exactly this and asserts the
refusal).

WHAT THIS SCRIPT DOES NOT DO
It does not modify `regions/pretrain.py` or `scripts/csd-train-all.py` -- the reservoir
fix already landed (c42203c) and stays as production's only sampling path; this script
only re-derives what a vanished receipt would have recorded. It does not decide reason's
receipt is now trustworthy -- W1b's job, not this one -- it only makes sure the reserve
(W2b) can never admit a row of ambiguous aqua_rat provenance while that receipt is
missing.

A THIRD, DATED CANDIDATE DRAW -- MEASURED, SURFACED, AND DELIBERATELY NOT BURNED HERE
`reason/aqua_rat-raw/derived/sample-4982-seed0.parquet` exists on disk next to the source.
It is not draw R: its own MANIFEST.json records `numpy Generator(PCG64).permutation(n)
[:N_SAMPLE]` as its sampling method, not `cogsyndelta.corpus.reservoir_sample` +
`sampling_rng` -- a different RNG and a different algorithm from what `load_pairs` runs
today. THAT observation alone does not settle whether `reason` consumed it: the file's
MANIFEST.json is timestamped 2026-09-02T23:05:45Z -- 27 minutes BEFORE `c42203c`
(23:32:52Z) and 36 minutes before the 4,982 cap literal was even added to
`scripts/csd-train-all.py` (`b9a082e`, 23:41:03Z) -- making it the strongest DATED artefact
on disk bearing on check (iv), and a live hypothesis for what `reason` actually consumed,
not a decoy this script gets to dismiss by construction. Measured against the real corpus:
its 4,982 non-empty pairs overlap `fingerprints(R) UNION fingerprints(P)` in only 585 rows;
the other 4,397 are certified CLEAN by this ledger today. `evaluate_third_draw_candidate`
computes and records this every run (never trusted as truth, always measured), and
`establish_code_revision` folds its mtime into check (iv)'s evidence so "established:
false" never again means "this evidence doesn't exist" when it does.

**This script does not burn it.** Doing so would take the union to at most 14,946 of
97,467 -- past DEC-42's stated `<= 9,964` (`<= 2 * CAP`) -- which re-derives every number
downstream of it in Section 5.1 (the pool table, the B1 accounting) and Section 9.2. That
is an operator decision, not a code fix this script gets to make unilaterally. What it
does instead: measure the overlap every run, write it into the manifest under
`third_draw_candidate` with `operator_decision_required: true`, and refuse to exit quietly
-- `main()` prints an unmissable banner and returns a distinct, non-zero exit code (`3`)
unless run with `--acknowledge-third-draw-candidate`, which records that an operator has
seen this specific evidence and lets the run proceed at exit `0` while changing nothing
about what gets burned.

Usage:
    scripts/csd-reserve-ledger.py                       # write data/reserve/burned-aqua_rat.jsonl
    scripts/csd-reserve-ledger.py --corpus-root /path    # override root discovery
    scripts/csd-reserve-ledger.py --out /tmp/x.jsonl     # write elsewhere (tests use this)
    scripts/csd-reserve-ledger.py --acknowledge-third-draw-candidate
                                                          # required for exit 0 once an
                                                          # operator has ruled on the
                                                          # candidate above

Exit codes: 0 ok; 1 REFUSED (a DEC-42 check failed -- see stderr); 2 corpus root or
shards not found; 3 the ledger WAS written but a third, dated candidate draw sits
unacknowledged -- rerun with --acknowledge-third-draw-candidate once an operator has
ruled on it (see the manifest's `third_draw_candidate`).
"""

from __future__ import annotations

import argparse
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
UTC. This is the DELIBERATE correctness change (prefix -> reservoir sampling) that makes
which draw the `reason` run used ambiguous; see the module docstring."""

B9A082E_SHA = "b9a082ed4e170401a90ac10894d3855484a156c3"
B9A082E_COMMIT_TIME_UTC = "2026-09-02T23:41:03+00:00"
"""`git show -s --format='%H %ci' b9a082e` -> `2026-09-02 19:41:03 -0400`, converted to
UTC. This is the commit that FIRST adds the 4,982 cap literal to
`scripts/csd-train-all.py`'s `REGIONS["reason"]` -- see the module docstring's "A THIRD,
DATED CANDIDATE DRAW" section for why this timestamp matters."""

DECOY_MANIFEST_REL = Path("reason") / "aqua_rat-raw" / "derived" / "MANIFEST.json"
DECOY_PARQUET_REL = Path("reason") / "aqua_rat-raw" / "derived" / "sample-4982-seed0.parquet"


class LedgerGateError(ValueError):
    """Raised when a ledger does not cover every row of BOTH DEC-42 draws.

    A ledger built from one draw alone is a FAIL, not a partial pass -- construct one and
    confirm this fires; see tests/test_reserve_ledger.py.
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
    """DEC-42's gate: every row of BOTH draws must have its fingerprint in the ledger.

    Counted in ROWS, not unique fingerprints -- see the module docstring's "THE GATE"
    section for why the unique-fingerprint reading can never pass on real data.

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


def _iso_utc(mtime: float) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(mtime, tz=_dt.UTC).isoformat()


def _epoch(iso: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(iso).timestamp()


def establish_code_revision(
    receipts_dir: Path = RECEIPTS_DIR,
    region: str = REGION,
    corpus_root: Path | None = None,
) -> dict[str, Any]:
    """DEC-42 check (iv), the one that can fail: which code revision produced the
    `region` run. Looks for a receipt (`{region}-*.json`) or a checkpoint
    (`{region}-checkpoints/`) under `receipts_dir` and compares its mtime against
    `C42203C_COMMIT_TIME_UTC`. When `corpus_root` is given, ALSO looks for
    `DECOY_MANIFEST_REL` under it and folds its mtime into `evidence` as a distinct
    `kind` -- it is not a `region` receipt or checkpoint, but it IS a dated artefact
    bearing on which code revision ran (see the module docstring's "A THIRD, DATED
    CANDIDATE DRAW" section), and it must never be left out of this function's own
    evidence just because it does not fit the receipt/checkpoint shape. "Absence of
    evidence must not become the evidence" (§5.1) cuts both ways: this function must not
    report `established: False` while sitting on evidence it simply didn't look at.
    The caller burns the union regardless of this function's result -- check (iv) is
    the one check DEC-42 explicitly allows to proceed unestablished.
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
    if corpus_root is not None:
        decoy_manifest = corpus_root / DECOY_MANIFEST_REL
        if decoy_manifest.is_file():
            mtime = decoy_manifest.stat().st_mtime
            evidence.append(
                {
                    "kind": "third_draw_candidate_manifest",
                    "path": str(decoy_manifest),
                    "mtime_utc": _iso_utc(mtime),
                    "before_c42203c": mtime < c42203c_epoch,
                    "before_b9a082e_cap_introduced": mtime < _epoch(B9A082E_COMMIT_TIME_UTC),
                    "note": (
                        "NOT a `region` receipt or checkpoint -- a derived-sample "
                        "MANIFEST.json under the corpus root, dated before BOTH "
                        "c42203c and the commit that introduced the cap literal. See "
                        "`third_draw_candidate` elsewhere in this manifest for the "
                        "measured row overlap; an operator decision is required "
                        "before this evidence can change what gets burned."
                    ),
                }
            )
    if not evidence:
        return {
            "established": False,
            "note": (
                f"no {region!r} receipt under {receipts_dir} ({region}-*.json), no "
                f"{checkpoints_dir} directory, and (when corpus_root was searched) no "
                f"third-draw-candidate manifest either -- the {region} receipt was "
                f"deleted (see W1b) and nothing else on disk dates the run. The code "
                f"revision the {region} run used cannot be determined from artefacts "
                f"on disk; recorded as such rather than assumed."
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


def _note_decoy_derived_sample(root: Path) -> dict[str, Any] | None:
    """Reads `DECOY_MANIFEST_REL`'s DECLARED sampling method and mtime, if present, under
    `root`. Returns `None` when there is nothing to note. This function only reports what
    the file DECLARES about itself -- it does not measure the file's actual rows (see
    `evaluate_third_draw_candidate` for that) and it does NOT decide whether the file is
    irrelevant. `matches_production_reservoir_algorithm: False` means only that this file
    was not produced by `reservoir_sample` + `sampling_rng` -- it does NOT mean the
    `reason` run could not have consumed it by some other, undocumented path; that
    inference was the bug this function used to make (see the module docstring's "A
    THIRD, DATED CANDIDATE DRAW" section)."""
    manifest_path = root / DECOY_MANIFEST_REL
    if not manifest_path.is_file():
        return None
    mtime = manifest_path.stat().st_mtime
    try:
        manifest = json.loads(manifest_path.read_text())
        readable = True
    except (OSError, json.JSONDecodeError):
        manifest, readable = {}, False
    method = manifest.get("sampling_method", "") if readable else ""
    is_production_method = "reservoir_sample" in method or "Algorithm R" in method
    return {
        "path": str(manifest_path),
        "readable": readable,
        "declared_sampling_method": method,
        "matches_production_reservoir_algorithm": is_production_method,
        "declared_seed": manifest.get("seed") if readable else None,
        "declared_n_sampled_rows": manifest.get("n_sampled_rows") if readable else None,
        "mtime_utc": _iso_utc(mtime),
        "before_c42203c": mtime < _epoch(C42203C_COMMIT_TIME_UTC),
        "before_b9a082e_cap_introduced": mtime < _epoch(B9A082E_COMMIT_TIME_UTC),
        "note": (
            "declared sampling method does not match production's "
            "reservoir_sample+sampling_rng, so this file is not draw R BY CONSTRUCTION -- "
            "but its mtime predates both c42203c and the commit that introduced the "
            "4,982 cap literal, so that alone does not settle whether `reason` consumed "
            "it by some other, undocumented path. See `evaluate_third_draw_candidate` "
            "for the measured row overlap against this ledger's union; resolving what "
            "this file was is an OPERATOR DECISION."
            if not is_production_method
            else (
                "declares production's reservoir_sample algorithm; this script still "
                "recomputes draw R from cogsyndelta.regions.pretrain.load_pairs directly "
                "rather than trusting a declared method on a file it did not produce."
            )
        ),
    }


def evaluate_third_draw_candidate(
    root: Path,
    union: set[str],
) -> dict[str, Any] | None:
    """Measures `DECOY_PARQUET_REL` against this ledger's `union` -- the honest version of
    what `_note_decoy_derived_sample` used to skip. Returns `None` when there is nothing
    on disk to measure (no manifest, or a manifest with no matching parquet). Never
    changes `union`: burning this file is an operator decision, not something this
    function does on its own -- see the module docstring's "A THIRD, DATED CANDIDATE
    DRAW" section. Requires `pyarrow` (already a hard dependency for parquet shards
    elsewhere in this script).
    """
    note = _note_decoy_derived_sample(root)
    if note is None:
        return None
    parquet_path = root / DECOY_PARQUET_REL
    if not parquet_path.is_file():
        note["parquet_present"] = False
        note["measured"] = False
        return note
    note["parquet_present"] = True

    import pyarrow.parquet as pq

    from cogsyndelta.eval import pair_fingerprint

    table = pq.read_table(parquet_path, columns=list(COLUMNS))
    col_a = table.column(COLUMNS[0]).to_pylist()
    col_b = table.column(COLUMNS[1]).to_pylist()
    pairs = [(a, b) for a, b in zip(col_a, col_b, strict=True) if a and b]
    fps = [pair_fingerprint(a, b) for a, b in pairs]
    covered = sum(1 for fp in fps if fp in union)
    not_covered = len(fps) - covered
    union_if_burned = len(union | set(fps))

    note["measured"] = True
    note["rows_total"] = len(col_a)
    note["rows_non_empty_pairs"] = len(pairs)
    note["rows_unique_fingerprints"] = len(set(fps))
    note["rows_covered_by_current_union"] = covered
    note["rows_certified_clean_by_this_ledger_today"] = not_covered
    note["union_if_burned"] = union_if_burned
    note["dec42_stated_union_cap"] = 2 * CAP
    note["burning_it_would_exceed_dec42_stated_cap"] = union_if_burned > 2 * CAP
    note["operator_decision_required"] = not_covered > 0
    note["note"] += (
        f" MEASURED against the real corpus: {len(pairs)} non-empty pairs, of which "
        f"{covered} already fall inside this ledger's union and {not_covered} do not -- "
        f"those {not_covered} rows are certified CLEAN by this ledger today. Burning "
        f"them would take the union to {union_if_burned}, "
        f"{'past' if note['burning_it_would_exceed_dec42_stated_cap'] else 'within'} "
        f"DEC-42's stated <= {2 * CAP}."
    )
    return note


def build_ledger(
    shards: list[str],
    *,
    seed: int = DEFAULT_SEED,
    cap: int = CAP,
    receipts_dir: Path = RECEIPTS_DIR,
    corpus_root: Path | None = None,
) -> dict[str, Any]:
    """Compute both draws, union their fingerprints, and enforce DEC-42's gate.

    Raises:
        RuntimeError: either draw is not reproducible across two computations (DEC-42
            check (ii)).
        SeedInferenceError: surviving receipts disagree on `config.seed` (DEC-42 check
            (iii)) -- see `SeedInferenceError`'s docstring for why absence of receipts
            is a different, non-blocking case.
        LedgerGateError: the union does not cover every row of both draws AT THE
            REQUESTED `cap` -- gated against `cap` itself, not against however many rows
            a draw actually returned, so a short/truncated/partial draw (bad mount,
            corrupt shard, wrong --corpus-root) cannot self-certify by shrinking its own
            target.
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

    fps_r = [pair_fingerprint(a, b) for a, b in r_first]
    fps_p = [pair_fingerprint(a, b) for a, b in p_first]
    set_r, set_p = set(fps_r), set(fps_p)
    union = set_r | set_p

    rows = [
        {
            "pair_fingerprint": fp,
            "draw": ("both" if (fp in set_r and fp in set_p) else ("R" if fp in set_r else "P")),
            "region": REGION,
            "source": SOURCE_ID,
        }
        for fp in sorted(union)
    ]

    # Gated against the REQUESTED cap, not against len(r_first)/len(p_first) -- a draw
    # that came back short (partial mount, truncated/replaced parquet, wrong
    # --corpus-root) must fail this, not silently redefine its own target. See the
    # LedgerGateError docstring in this function's own Raises: section.
    covered_r, covered_p = verify_gate(union, r_first, p_first, cap=cap)

    corpus_fp = fingerprint_corpus(shards, columns=COLUMNS)

    third_draw_candidate = (
        evaluate_third_draw_candidate(corpus_root, union) if corpus_root is not None else None
    )

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
        "union_size": len(union),
        "intersection_size": len(set_r & set_p),
        "code_revision": establish_code_revision(
            receipts_dir, region=REGION, corpus_root=corpus_root
        ),
        "gate": {
            "cap": cap,
            "draw_r_rows_covered": covered_r,
            "draw_p_rows_covered": covered_p,
            "passed": True,
        },
        "third_draw_candidate": third_draw_candidate,
        "operator_decision_required": bool(
            third_draw_candidate is not None
            and third_draw_candidate.get("operator_decision_required")
        ),
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
    parser.add_argument(
        "--acknowledge-third-draw-candidate",
        action="store_true",
        help=(
            "required to exit 0 when a third, dated candidate draw is found on disk "
            "(see the module docstring's 'A THIRD, DATED CANDIDATE DRAW' section) -- "
            "records that an operator has seen the measured evidence in "
            "third_draw_candidate before this run is treated as final. Without it, a "
            "run that finds such evidence still writes the ledger (for review) but "
            "exits 3, not 0."
        ),
    )
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

    if args.acknowledge_third_draw_candidate and result["manifest"]["third_draw_candidate"]:
        result["manifest"]["third_draw_candidate"]["acknowledged_by_operator_flag"] = True

    manifest_path = write_ledger(args.out, result)
    m = result["manifest"]
    print(
        f"wrote {len(result['rows'])} burned fingerprints to {args.out}\n"
        f"  draw R: {m['draw_r_rows']} rows ({m['draw_r_unique_fingerprints']} unique)\n"
        f"  draw P: {m['draw_p_rows']} rows ({m['draw_p_unique_fingerprints']} unique)\n"
        f"  union: {m['union_size']}  intersection: {m['intersection_size']}\n"
        f"  gate: covered {m['gate']['draw_r_rows_covered']}/{m['gate']['cap']} (R), "
        f"{m['gate']['draw_p_rows_covered']}/{m['gate']['cap']} (P)\n"
        f"  code revision established: {m['code_revision']['established']}\n"
        f"  manifest: {manifest_path}",
        file=sys.stderr,
    )

    if m["operator_decision_required"] and not args.acknowledge_third_draw_candidate:
        tdc = m["third_draw_candidate"]
        print(
            "\n"
            "=========================================================================\n"
            "csd-reserve-ledger: OPERATOR DECISION REQUIRED -- NOT ACKNOWLEDGED\n"
            "=========================================================================\n"
            f"A third, dated candidate draw exists on disk: {tdc['path']}\n"
            f"  mtime: {tdc['mtime_utc']}  (before c42203c: {tdc['before_c42203c']}, "
            f"before b9a082e's cap literal: {tdc['before_b9a082e_cap_introduced']})\n"
            f"  {tdc['rows_non_empty_pairs']} non-empty pairs measured; "
            f"{tdc['rows_covered_by_current_union']} already in this ledger's union, "
            f"{tdc['rows_certified_clean_by_this_ledger_today']} are NOT -- this ledger "
            f"is certifying those rows CLEAN today.\n"
            f"  Burning them would take the union to {tdc['union_if_burned']} "
            f"({'past' if tdc['burning_it_would_exceed_dec42_stated_cap'] else 'within'} "
            f"DEC-42's stated <= {tdc['dec42_stated_union_cap']}), which re-derives "
            "Section 5.1 and 9.2 -- this script does not decide that on its own.\n"
            "The ledger above WAS written, so the evidence is available to review, but "
            "this run is NOT treated as final: rerun with "
            "--acknowledge-third-draw-candidate once an operator has ruled on it.\n"
            "=========================================================================",
            file=sys.stderr,
        )
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
