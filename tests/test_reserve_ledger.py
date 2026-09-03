"""Tests for `scripts/csd-reserve-ledger.py` (W2a): the aqua_rat draw-recovery ledger.

DEC-42 (docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md §5.1/§4.1): `reason`'s aqua_rat
cap changed sampling method at `c42203c` (prefix -> seeded reservoir) and the receipt that
would say which one the `reason` run used was deleted. The ledger has to burn BOTH
candidate draws' fingerprints, and it has to REFUSE to certify a ledger built from only
one -- that refusal is the actual guard, so every guard test here builds the exact
condition it exists to catch and asserts it fires (same convention as
tests/test_guards_can_fail.py).

Unit tests below use tiny synthetic parquet shards -- no dependency on the real corpus.
One integration test at the bottom runs against the real aqua_rat shard and reports row
counts and the union size; it skips when the corpus export is not mounted.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# MUST precede the module-under-test import: it imports cogsyndelta.regions.pretrain,
# which imports tokenizers at module scope -- without the train group installed the
# import errors during collection and the whole file fails instead of skipping (same
# reasoning as tests/test_build_splits_shuffle.py and tests/test_corpus.py).
pytest.importorskip("pyarrow", reason="train group not installed")
pytest.importorskip("tokenizers", reason="train group not installed")

import pyarrow as pa
import pyarrow.parquet as pq

pytestmark = pytest.mark.cpu

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "csd-reserve-ledger.py"


def _load_module():
    """`scripts/csd-reserve-ledger.py` has a hyphen in its name, so it is not an
    importable module path -- load it directly from its file, the way `csd-train-all.py`
    itself would have to be loaded from a test (there is no existing precedent in this
    tree for testing a hyphenated script directly, so this is the straightforward
    `importlib.util.spec_from_file_location` load)."""
    spec = importlib.util.spec_from_file_location("csd_reserve_ledger", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ledger = _load_module()


# ---------------------------------------------------------------------------------------
# Fixtures: a tiny synthetic aqua_rat-shaped shard, built so draw R and draw P provably
# differ -- the head of the file is one group, the tail another, same shape
# test_build_splits_shuffle.py uses to prove a shuffle actually ran.
# ---------------------------------------------------------------------------------------

_HEAD_N = 60
_TAIL_N = 40
_TOTAL = _HEAD_N + _TAIL_N
_CAP = 25


def _write_shard(path: Path, *, duplicate_at_indices: tuple[int, ...] = ()) -> None:
    """`_HEAD_N` "head" rows followed by `_TAIL_N` "tail" rows, questions/rationales
    distinct per row so pair_fingerprint distinguishes every one of them -- except at
    `duplicate_at_indices`, which are overwritten with row 0's exact text verbatim. This
    is what lets a test construct real aqua_rat's measured property: some rows in a draw
    of `_CAP` share a fingerprint with another row in the SAME draw. Indices are chosen
    below `_CAP` where a test needs the duplicate to land inside draw P specifically
    (draw P is always rows `[0, _CAP)` in this fixture's shard order).
    """
    questions: list[str] = []
    rationales: list[str] = []
    for i in range(_HEAD_N):
        questions.append(f"head question number {i}")
        rationales.append(f"head rationale body {i}")
    for i in range(_TAIL_N):
        questions.append(f"tail question number {i}")
        rationales.append(f"tail rationale body {i}")
    for idx in duplicate_at_indices:
        questions[idx] = questions[0]
        rationales[idx] = rationales[0]
    pq.write_table(pa.table({"question": questions, "rationale": rationales}), path)


@pytest.fixture
def shard(tmp_path: Path) -> list[str]:
    path = tmp_path / "train.parquet"
    _write_shard(path)
    return [str(path)]


@pytest.fixture
def shard_with_duplicates(tmp_path: Path) -> list[str]:
    """Duplicates at indices 3, 7 and 12 -- all below `_CAP` (25), so draw P (the prefix)
    is guaranteed to carry them regardless of anything random."""
    path = tmp_path / "train.parquet"
    _write_shard(path, duplicate_at_indices=(3, 7, 12))
    return [str(path)]


# ---------------------------------------------------------------------------------------
# Draw R and draw P are different algorithms, and it matters that they diverge.
# ---------------------------------------------------------------------------------------


def test_draw_prefix_is_exactly_the_first_cap_rows_in_shard_order(shard: list[str]) -> None:
    prefix = ledger.draw_prefix(shard, cap=_CAP)
    assert len(prefix) == _CAP
    assert prefix == [
        (f"head question number {i}", f"head rationale body {i}") for i in range(_CAP)
    ]


def test_draw_reservoir_is_not_just_the_prefix(shard: list[str]) -> None:
    """The whole point of DEC-42: reservoir sampling reaches past the head. If this ever
    started returning exactly the prefix, draw R and draw P would collapse into one draw
    and the ledger's union would be pointless -- this is the failing case for THAT."""
    reservoir = ledger.draw_reservoir(shard, seed=0, cap=_CAP)
    prefix = ledger.draw_prefix(shard, cap=_CAP)
    assert len(reservoir) == _CAP
    assert set(reservoir) != set(prefix), "reservoir sample landed on exactly the prefix"
    tail_rows_in_reservoir = sum(1 for q, _ in reservoir if q.startswith("tail"))
    assert tail_rows_in_reservoir > 0, "a uniform sample over 100 rows drew none of the 40-row tail"


def test_draw_reservoir_is_deterministic_across_two_computations(shard: list[str]) -> None:
    assert ledger.draw_reservoir(shard, seed=0, cap=_CAP) == ledger.draw_reservoir(
        shard, seed=0, cap=_CAP
    )


def test_draw_reservoir_differs_across_seeds(shard: list[str]) -> None:
    assert ledger.draw_reservoir(shard, seed=0, cap=_CAP) != ledger.draw_reservoir(
        shard, seed=1, cap=_CAP
    )


def test_draw_prefix_is_deterministic_across_two_computations(shard: list[str]) -> None:
    assert ledger.draw_prefix(shard, cap=_CAP) == ledger.draw_prefix(shard, cap=_CAP)


# ---------------------------------------------------------------------------------------
# The gate: counted in rows, not unique fingerprints (see the script's module docstring,
# "THE GATE, AND WHY IT IS COUNTED IN ROWS, NOT UNIQUE FINGERPRINTS").
# ---------------------------------------------------------------------------------------


def test_gate_counts_rows_not_unique_fingerprints_when_a_draw_has_internal_duplicates(
    shard_with_duplicates: list[str],
) -> None:
    """Mirrors the real corpus: a real seed=0 draw of aqua_rat carries rows that share a
    fingerprint with another row in the SAME draw (measured: 31 of 4,982). A ledger built
    correctly from that draw must still pass the gate for every one of the draw's rows --
    `len(set(fingerprints(draw))) == cap` would be false here by construction, and a gate
    written that way could never pass on this fixture (or on the real corpus)."""
    draw_r = ledger.draw_reservoir(shard_with_duplicates, seed=0, cap=_CAP)
    draw_p = ledger.draw_prefix(shard_with_duplicates, cap=_CAP)
    fps_r = set(ledger.fingerprint_pairs(draw_r))
    fps_p = set(ledger.fingerprint_pairs(draw_p))
    # The fixture's design must actually exercise the case under test.
    assert len(fps_p) < _CAP, "fixture did not produce an internal duplicate in draw P"

    union = fps_r | fps_p
    covered_r, covered_p = ledger.verify_gate(union, draw_r, draw_p, cap=_CAP)
    assert covered_r == _CAP
    assert covered_p == _CAP


def test_gate_refuses_a_ledger_built_from_one_draw_alone(shard: list[str]) -> None:
    """THE falsifier DEC-42 demands: 'hand the checker a ledger built from draw R alone
    and assert it refuses.' This constructs exactly that and checks the refusal fires."""
    draw_r = ledger.draw_reservoir(shard, seed=0, cap=_CAP)
    draw_p = ledger.draw_prefix(shard, cap=_CAP)
    r_only = set(ledger.fingerprint_pairs(draw_r))

    with pytest.raises(ledger.LedgerGateError, match=r"\d+/\d+ rows of draw P"):
        ledger.verify_gate(r_only, draw_r, draw_p, cap=_CAP)


def test_gate_refuses_a_ledger_built_from_the_other_draw_alone(shard: list[str]) -> None:
    draw_r = ledger.draw_reservoir(shard, seed=0, cap=_CAP)
    draw_p = ledger.draw_prefix(shard, cap=_CAP)
    p_only = set(ledger.fingerprint_pairs(draw_p))

    with pytest.raises(ledger.LedgerGateError, match=r"\d+/\d+ rows of draw R"):
        ledger.verify_gate(p_only, draw_r, draw_p, cap=_CAP)


def test_gate_passes_the_true_union(shard: list[str]) -> None:
    draw_r = ledger.draw_reservoir(shard, seed=0, cap=_CAP)
    draw_p = ledger.draw_prefix(shard, cap=_CAP)
    union = set(ledger.fingerprint_pairs(draw_r)) | set(ledger.fingerprint_pairs(draw_p))
    covered_r, covered_p = ledger.verify_gate(union, draw_r, draw_p, cap=_CAP)
    assert (covered_r, covered_p) == (_CAP, _CAP)


# ---------------------------------------------------------------------------------------
# build_ledger end to end: union content, idempotency, corpus-fingerprint drift.
# ---------------------------------------------------------------------------------------


def test_build_ledger_writes_the_union_with_correct_draw_membership(
    shard: list[str], tmp_path: Path
) -> None:
    result = ledger.build_ledger(shard, seed=0, cap=_CAP, receipts_dir=tmp_path / "no-receipts")
    rows = {r["pair_fingerprint"]: r["draw"] for r in result["rows"]}

    draw_r = ledger.draw_reservoir(shard, seed=0, cap=_CAP)
    draw_p = ledger.draw_prefix(shard, cap=_CAP)
    fps_r = set(ledger.fingerprint_pairs(draw_r))
    fps_p = set(ledger.fingerprint_pairs(draw_p))

    assert set(rows) == fps_r | fps_p
    for fp, draw in rows.items():
        expected = "both" if (fp in fps_r and fp in fps_p) else ("R" if fp in fps_r else "P")
        assert draw == expected
    assert all(r["region"] == "reason" for r in result["rows"])
    assert all(r["source"] == "deepmind/aqua_rat" for r in result["rows"])

    m = result["manifest"]
    assert m["draw_r_rows"] == _CAP
    assert m["draw_p_rows"] == _CAP
    assert m["union_size"] == len(fps_r | fps_p)
    assert m["gate"]["passed"] is True


def test_build_ledger_is_idempotent(shard: list[str], tmp_path: Path) -> None:
    """Run twice, identical output -- the requirement this module's `write_ledger`
    docstring is built around (no wall-clock field in either written file)."""
    receipts_dir = tmp_path / "no-receipts"
    out1 = tmp_path / "run1.jsonl"
    out2 = tmp_path / "run2.jsonl"

    r1 = ledger.build_ledger(shard, seed=0, cap=_CAP, receipts_dir=receipts_dir)
    r2 = ledger.build_ledger(shard, seed=0, cap=_CAP, receipts_dir=receipts_dir)
    ledger.write_ledger(out1, r1)
    ledger.write_ledger(out2, r2)

    assert out1.read_text() == out2.read_text()
    assert (
        out1.with_suffix(".jsonl.manifest.json").read_text()
        == out2.with_suffix(".jsonl.manifest.json").read_text()
    )
    # And every line is valid JSON, rows sorted by fingerprint -- diffable, not just
    # byte-equal by accident. (The lines themselves do not sort lexically to the same
    # order: `sort_keys=True` puts `"draw"` before `"pair_fingerprint"` in each object,
    # so a raw string sort would key on draw membership first. Sort key is the
    # fingerprint field, not the line.)
    lines = out1.read_text().splitlines()
    parsed = [json.loads(line) for line in lines]
    fingerprints = [row["pair_fingerprint"] for row in parsed]
    assert fingerprints == sorted(fingerprints)


def test_build_ledger_raises_on_a_nonreproducible_draw(
    shard: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DEC-42 check (ii). Forces draw P to disagree with itself across two calls and
    asserts `build_ledger` refuses rather than silently unioning inconsistent draws."""
    calls = {"n": 0}
    real_draw_prefix = ledger.draw_prefix

    def flaky_draw_prefix(shards, cap=ledger.CAP):
        calls["n"] += 1
        result = real_draw_prefix(shards, cap)
        if calls["n"] == 2:
            result = [*result[:-1], ("mutated", "row")]
        return result

    monkeypatch.setattr(ledger, "draw_prefix", flaky_draw_prefix)
    with pytest.raises(RuntimeError, match="draw P"):
        ledger.build_ledger(shard, seed=0, cap=_CAP, receipts_dir=tmp_path / "no-receipts")


def test_corpus_fingerprint_changes_when_the_shard_changes(
    shard: list[str], tmp_path: Path
) -> None:
    """DEC-42 check (i): the ledger's recorded fingerprint must actually track the file on
    disk, not just exist."""
    fp_before = ledger.build_ledger(shard, seed=0, cap=_CAP, receipts_dir=tmp_path / "no-receipts")[
        "manifest"
    ]["corpus_fingerprint"]

    # Grow the shard -- same schema, more rows -- and refingerprint.
    path = Path(shard[0])
    extra = pa.table({"question": ["extra question"], "rationale": ["extra rationale"]})
    combined = pa.concat_tables([pq.read_table(path), extra])
    pq.write_table(combined, path)

    fp_after = ledger.build_ledger(shard, seed=0, cap=_CAP, receipts_dir=tmp_path / "no-receipts")[
        "manifest"
    ]["corpus_fingerprint"]
    assert fp_before != fp_after


# ---------------------------------------------------------------------------------------
# Code-revision facts (DEC-42 check (iv)) and the seed inference (check (iii)).
# ---------------------------------------------------------------------------------------


def test_establish_code_revision_records_absence_when_nothing_survives(tmp_path: Path) -> None:
    empty_receipts = tmp_path / "receipts"
    empty_receipts.mkdir()
    result = ledger.establish_code_revision(empty_receipts, region="reason")
    assert result["established"] is False
    assert result["evidence"] == []
    assert "cannot be determined" in result["note"]


def test_establish_code_revision_reports_mtime_evidence_when_a_receipt_exists(
    tmp_path: Path,
) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "reason-20260903T000000Z.json").write_text("{}")
    result = ledger.establish_code_revision(receipts, region="reason")
    assert result["established"] is True
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["before_c42203c"] is False  # written after c42203c, per mtime


def test_infer_seed_reads_surviving_receipts(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "code-x.json").write_text(json.dumps({"config": {"seed": 0}}))
    (receipts / "compress-x.json").write_text(json.dumps({"config": {"seed": 0}}))
    result = ledger.infer_seed(receipts, expected=0)
    assert result["all_surviving_receipts_agree"] is True
    assert result["surviving_receipts_seed"] == {"code-x.json": 0, "compress-x.json": 0}


def test_infer_seed_flags_disagreement(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "code-x.json").write_text(json.dumps({"config": {"seed": 0}}))
    (receipts / "odd-x.json").write_text(json.dumps({"config": {"seed": 7}}))
    result = ledger.infer_seed(receipts, expected=0)
    assert result["all_surviving_receipts_agree"] is False


# ---------------------------------------------------------------------------------------
# The decoy on disk: a derived sample must never be trusted as draw R.
# ---------------------------------------------------------------------------------------


def test_decoy_derived_sample_with_wrong_algorithm_is_flagged_and_not_trusted(
    tmp_path: Path,
) -> None:
    derived = tmp_path / "reason" / "aqua_rat-raw" / "derived"
    derived.mkdir(parents=True)
    (derived / "MANIFEST.json").write_text(
        json.dumps({"sampling_method": "numpy Generator(PCG64).permutation(n)[:N]"})
    )
    note = ledger._note_decoy_derived_sample(tmp_path)
    assert note is not None
    assert note["matches_production_algorithm"] is False
    assert "IGNORED" in note["note"]


def test_decoy_derived_sample_absent_is_silently_fine(tmp_path: Path) -> None:
    assert ledger._note_decoy_derived_sample(tmp_path) is None


# ---------------------------------------------------------------------------------------
# Corpus root resolution.
# ---------------------------------------------------------------------------------------


def test_resolve_corpus_root_prefers_explicit_override(tmp_path: Path) -> None:
    assert ledger.resolve_corpus_root(tmp_path) == tmp_path


def test_resolve_corpus_root_rejects_a_nonexistent_explicit_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ledger.resolve_corpus_root(tmp_path / "does-not-exist")


def test_resolve_corpus_root_raises_when_no_candidate_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ledger, "CANDIDATE_CORPUS_ROOTS", (tmp_path / "nope-1", tmp_path / "nope-2")
    )
    with pytest.raises(FileNotFoundError, match="none of the candidate"):
        ledger.resolve_corpus_root(None)


def test_locate_shards_matches_the_declared_glob(tmp_path: Path) -> None:
    target_dir = tmp_path / "reason" / "aqua_rat-raw"
    target_dir.mkdir(parents=True)
    _write_shard(target_dir / "train.parquet")
    (tmp_path / "reason").mkdir(exist_ok=True)
    assert ledger.locate_shards(tmp_path) == [str(target_dir / "train.parquet")]


# ---------------------------------------------------------------------------------------
# Integration: the real corpus, if mounted. Reports row counts and the union size.
# ---------------------------------------------------------------------------------------


def _real_corpus_root() -> Path | None:
    try:
        return ledger.resolve_corpus_root(None)
    except FileNotFoundError:
        return None


_REAL_ROOT = _real_corpus_root()
needs_real_corpus = pytest.mark.skipif(
    _REAL_ROOT is None,
    reason=f"aqua_rat corpus not mounted at any of {ledger.CANDIDATE_CORPUS_ROOTS}",
)


@needs_real_corpus
def test_integration_real_aqua_rat_union_covers_both_draws_in_full(tmp_path: Path) -> None:
    root = _REAL_ROOT
    assert root is not None
    shards = ledger.locate_shards(root)
    assert shards, f"no shards matched {ledger.SOURCE_GLOB!r} under {root}"

    result = ledger.build_ledger(
        shards, seed=0, receipts_dir=tmp_path / "no-receipts", corpus_root=root
    )
    m = result["manifest"]

    print(
        f"\n[integration] draw R: {m['draw_r_rows']} rows "
        f"({m['draw_r_unique_fingerprints']} unique)\n"
        f"[integration] draw P: {m['draw_p_rows']} rows "
        f"({m['draw_p_unique_fingerprints']} unique)\n"
        f"[integration] union: {m['union_size']}  intersection: {m['intersection_size']}"
    )

    assert m["draw_r_rows"] == ledger.CAP
    assert m["draw_p_rows"] == ledger.CAP
    assert m["gate"]["passed"] is True
    # DEC-42's stated worst case: at most 2 * CAP, i.e. no overlap at all.
    assert m["union_size"] <= 2 * ledger.CAP
    # And it should be a real recovery, not degenerate to one draw entirely.
    assert m["union_size"] > ledger.CAP

    write_target = tmp_path / "burned-aqua_rat.jsonl"
    ledger.write_ledger(write_target, result)
    lines = write_target.read_text().splitlines()
    assert len(lines) == m["union_size"]


@needs_real_corpus
def test_integration_ledger_write_is_idempotent_against_the_real_corpus(tmp_path: Path) -> None:
    root = _REAL_ROOT
    assert root is not None
    shards = ledger.locate_shards(root)
    receipts_dir = tmp_path / "no-receipts"

    out1 = tmp_path / "run1.jsonl"
    out2 = tmp_path / "run2.jsonl"
    ledger.write_ledger(out1, ledger.build_ledger(shards, seed=0, receipts_dir=receipts_dir))
    ledger.write_ledger(out2, ledger.build_ledger(shards, seed=0, receipts_dir=receipts_dir))

    assert out1.read_text() == out2.read_text()
