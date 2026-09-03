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

A NOTE ON A FILE THIS SCRIPT DELIBERATELY IGNORES
`reason/aqua_rat-raw/derived/sample-4982-seed0.parquet` exists on disk next to the source
and looks, at a glance, like a pre-computed draw R. It is not: its own MANIFEST.json
records `numpy Generator(PCG64).permutation(n)[:N_SAMPLE]` as its sampling method, not
`cogsyndelta.corpus.reservoir_sample` + `sampling_rng` -- a different RNG and a different
algorithm from what `load_pairs` actually runs, so treating it as draw R would recover the
wrong 4,982 rows with high confidence. This script never reads it; it recomputes draw R
from `load_pairs` itself and records the derived file's presence as a warning only (see
`_note_decoy_derived_sample`).

Usage:
    scripts/csd-reserve-ledger.py                       # write data/reserve/burned-aqua_rat.jsonl
    scripts/csd-reserve-ledger.py --corpus-root /path    # override root discovery
    scripts/csd-reserve-ledger.py --out /tmp/x.jsonl     # write elsewhere (tests use this)
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


class LedgerGateError(ValueError):
    """Raised when a ledger does not cover every row of BOTH DEC-42 draws.

    A ledger built from one draw alone is a FAIL, not a partial pass -- construct one and
    confirm this fires; see tests/test_reserve_ledger.py.
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


def establish_code_revision(
    receipts_dir: Path = RECEIPTS_DIR, region: str = REGION
) -> dict[str, Any]:
    """DEC-42 check (iv), the one that can fail: which code revision produced the
    `region` run. Looks for a receipt (`{region}-*.json`) or a checkpoint
    (`{region}-checkpoints/`) under `receipts_dir` and compares its mtime against
    `C42203C_COMMIT_TIME_UTC`. When neither exists -- true for `reason` today; verified
    live, not assumed -- records that explicitly. "Absence of evidence must not become
    the evidence" (§5.1): the caller burns the union regardless of this function's
    result.
    """
    from datetime import datetime

    c42203c_epoch = datetime.fromisoformat(C42203C_COMMIT_TIME_UTC).timestamp()
    receipts = sorted(receipts_dir.glob(f"{region}-*.json"))
    checkpoints_dir = receipts_dir / f"{region}-checkpoints"
    checkpoints = sorted(checkpoints_dir.glob("*")) if checkpoints_dir.is_dir() else []
    evidence: list[dict[str, Any]] = []
    for p in [*receipts, *checkpoints]:
        mtime = p.stat().st_mtime
        evidence.append(
            {
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
                f"W1b) and no checkpoint survives it either. The code revision the "
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
            f"evidence exists for {region!r} -- see `evidence` for each artefact's mtime "
            f"against c42203c; a human still has to weigh whether it settles which "
            f"sampling method ran (a preserved-after-the-fact commit does not date the "
            f"run that produced it -- §5.1)."
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
    """Records, but never trusts, `.../aqua_rat-raw/derived/sample-4982-seed0.parquet` if
    present -- see the module docstring. Returns `None` when there is nothing to note."""
    manifest_path = root / "reason" / "aqua_rat-raw" / "derived" / "MANIFEST.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {
            "path": str(manifest_path),
            "note": "present but unreadable; ignored either way -- this script never reads the sample itself",
        }
    method = manifest.get("sampling_method", "")
    is_production_method = "reservoir_sample" in method or "Algorithm R" in method
    return {
        "path": str(manifest_path),
        "declared_sampling_method": method,
        "matches_production_algorithm": is_production_method,
        "note": (
            "found and IGNORED -- this script always recomputes draw R from "
            "cogsyndelta.regions.pretrain.load_pairs rather than reading this file"
            if is_production_method
            else (
                "found and IGNORED -- its declared sampling method does not match "
                "production's reservoir_sample+sampling_rng, so it is not draw R under "
                "any hypothesis this script tests"
            )
        ),
    }


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
        LedgerGateError: the union does not cover every row of both draws (should be
            structurally impossible given this function's own construction -- this is
            the belt to `verify_gate`'s braces, and its message would mean this
            function's own union logic is broken, not that the corpus is ambiguous).
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

    covered_r, covered_p = verify_gate(union, r_first, p_first, cap=len(r_first))

    corpus_fp = fingerprint_corpus(shards, columns=COLUMNS)

    manifest: dict[str, Any] = {
        "region": REGION,
        "source": SOURCE_ID,
        "shards": shards,
        "columns": list(COLUMNS),
        "cap": cap,
        "seed": infer_seed(receipts_dir, expected=seed),
        "corpus_fingerprint": corpus_fp,
        "corpus_fingerprint_scheme": CORPUS_FINGERPRINT_SCHEME,
        "draw_r_rows": len(r_first),
        "draw_p_rows": len(p_first),
        "draw_r_unique_fingerprints": len(set_r),
        "draw_p_unique_fingerprints": len(set_p),
        "union_size": len(union),
        "intersection_size": len(set_r & set_p),
        "code_revision": establish_code_revision(receipts_dir, region=REGION),
        "gate": {
            "cap": len(r_first),
            "draw_r_rows_covered": covered_r,
            "draw_p_rows_covered": covered_p,
            "passed": True,
        },
    }
    if corpus_root is not None:
        decoy = _note_decoy_derived_sample(corpus_root)
        if decoy is not None:
            manifest["derived_sample_on_disk"] = decoy

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
    except (RuntimeError, LedgerGateError) as exc:
        print(f"csd-reserve-ledger: REFUSED -- {exc}", file=sys.stderr)
        return 1

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
