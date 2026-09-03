"""Tests for `scripts/csd-reserve-ledger.py` (W2a): the aqua_rat draw-recovery ledger.

DEC-42 (docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md §5.1/§4.1): `reason`'s aqua_rat
cap changed sampling method at `c42203c` (prefix -> seeded reservoir) and the receipt that
would say which one the `reason` run used was deleted. Per the orchestrator decision
recorded in the script's module docstring, the ledger burns the UNION OF EVERY CANDIDATE
DRAW it can find -- R, P, and every on-disk derived-sample file whose row count matches
the cap -- and REFUSES to certify a ledger that omits any one of them. That refusal is
the actual guard, so every guard test here builds the exact condition it exists to catch
and asserts it fires (same convention as tests/test_guards_can_fail.py).

Unit tests below use tiny synthetic parquet shards -- no dependency on the real corpus.
Integration tests at the bottom run against the real aqua_rat shard (and its one known
on-disk derived sample) and report row counts and the union size; they skip when the
corpus export is not mounted.
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


def test_build_ledger_draw_contributions_report_new_fingerprints_not_row_counts(
    shard: list[str], tmp_path: Path
) -> None:
    """The reporting must state each draw's real CONTRIBUTION -- new fingerprints not
    already covered by draws folded in before it -- not the tautological 'N/N rows
    covered' the gate already guarantees for every draw by construction."""
    result = ledger.build_ledger(shard, seed=0, cap=_CAP, receipts_dir=tmp_path / "no-receipts")
    m = result["manifest"]
    by_draw = {c["draw"]: c for c in m["draw_contributions"]}
    assert set(by_draw) == {"R", "P"}
    # R is first, so its "new" count equals its own unique-fingerprint count exactly.
    assert by_draw["R"]["new_fingerprints"] == by_draw["R"]["unique_fingerprints"]
    # P's "new" count can only be <= its own unique count (some of its rows may already
    # be covered by R).
    assert by_draw["P"]["new_fingerprints"] <= by_draw["P"]["unique_fingerprints"]
    assert by_draw["P"]["new_fingerprints"] == len(
        set(ledger.fingerprint_pairs(ledger.draw_prefix(shard, cap=_CAP)))
        - set(ledger.fingerprint_pairs(ledger.draw_reservoir(shard, seed=0, cap=_CAP)))
    )


def test_build_ledger_pool_floor_note_reports_against_the_design_doc_bound(
    shard: list[str], tmp_path: Path
) -> None:
    result = ledger.build_ledger(shard, seed=0, cap=_CAP, receipts_dir=tmp_path / "no-receipts")
    m = result["manifest"]
    pfn = m["pool_floor_note"]
    assert pfn["design_doc_bound"] == 2 * _CAP
    assert pfn["union_size"] == m["union_size"]
    assert pfn["clean_pool_size"] == pfn["corpus_size"] - m["union_size"]
    assert pfn["exceeds_design_doc_bound"] is (m["union_size"] > 2 * _CAP)


def test_build_ledger_pool_floor_note_flags_exceeding_the_bound_with_an_ondisk_draw(
    shard: list[str], tmp_path: Path
) -> None:
    """Burning a discovered on-disk draw with rows outside R∪P pushes the union past
    2*cap -- `pool_floor_note` must say so explicitly, not leave it implicit."""
    _write_ondisk_draw(tmp_path, "sample-25-seed0.parquet", _pairs(_CAP))
    result = ledger.build_ledger(
        shard, seed=0, cap=_CAP, receipts_dir=tmp_path / "no-receipts", corpus_root=tmp_path
    )
    m = result["manifest"]
    assert m["union_size"] > 2 * _CAP
    assert m["pool_floor_note"]["exceeds_design_doc_bound"] is True
    assert "re-derived" in m["pool_floor_note"]["note"]


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


def test_establish_code_revision_folds_in_ondisk_draws_and_sibling_manifest_as_evidence(
    tmp_path: Path,
) -> None:
    """CRITICAL review finding: `note` used to claim 'nothing else on disk dates the
    run' while a discovered draw's own `mtime_utc` -- a dated check-(iv) artefact --
    sat unweighed in the very same manifest. With `ondisk_draws` given, every
    discovered draw's mtime must appear in `evidence`, and (when `corpus_root` is also
    given and a sibling MANIFEST.json exists) that manifest's declared
    `sampling_method`/`generated_utc` must appear too -- `established` must reflect
    that this evidence exists, not that the question is settled."""
    empty_receipts = tmp_path / "receipts"
    empty_receipts.mkdir()
    corpus_root = tmp_path / "corpus"
    _write_ondisk_draw(corpus_root, "sample-4982-seed0.parquet", _pairs(_CAP))
    derived = corpus_root / "reason" / "aqua_rat-raw" / "derived"
    (derived / "MANIFEST.json").write_text(
        json.dumps(
            {
                "sampling_method": "numpy Generator(PCG64).permutation(n)[:N_SAMPLE]",
                "generated_utc": "2026-09-02T23:05:45Z",
            }
        )
    )
    draws, unreadable = ledger.discover_ondisk_draws(corpus_root, cap=_CAP)
    assert unreadable == []

    result = ledger.establish_code_revision(
        empty_receipts, region="reason", corpus_root=corpus_root, ondisk_draws=draws
    )
    assert result["established"] is True
    kinds = [e["kind"] for e in result["evidence"]]
    assert "ondisk_draw" in kinds
    assert "derived_sample_manifest" in kinds
    manifest_evidence = next(
        e for e in result["evidence"] if e["kind"] == "derived_sample_manifest"
    )
    assert manifest_evidence["declared_sampling_method"] == (
        "numpy Generator(PCG64).permutation(n)[:N_SAMPLE]"
    )
    assert manifest_evidence["declared_generated_utc"] == "2026-09-02T23:05:45Z"
    assert "before_c42203c" in manifest_evidence
    assert "before_b9a082e_cap_introduced" in manifest_evidence
    ondisk_evidence = next(e for e in result["evidence"] if e["kind"] == "ondisk_draw")
    assert "before_c42203c" in ondisk_evidence
    assert "before_b9a082e_cap_introduced" in ondisk_evidence


def test_establish_code_revision_still_absent_without_ondisk_draws_or_receipts(
    tmp_path: Path,
) -> None:
    empty_receipts = tmp_path / "receipts"
    empty_receipts.mkdir()
    result = ledger.establish_code_revision(
        empty_receipts, region="reason", corpus_root=tmp_path, ondisk_draws=[]
    )
    assert result["established"] is False
    assert result["evidence"] == []


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


def test_build_ledger_refuses_when_surviving_receipts_disagree_on_seed(
    shard: list[str], tmp_path: Path
) -> None:
    """BLOCKING review finding: `infer_seed`'s `all_surviving_receipts_agree` was computed
    and never read anywhere. CONSTRUCTED exactly as the review did: a receipts dir with
    `config.seed=0` and `config.seed=7`. Before the fix, `build_ledger` returned normally
    with `gate.passed=True`; DEC-42 5.1 is explicit that a check-(iii) failure must revert
    the write-off, not pass silently."""
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "code-x.json").write_text(json.dumps({"config": {"seed": 0}}))
    (receipts / "odd-x.json").write_text(json.dumps({"config": {"seed": 7}}))

    with pytest.raises(ledger.SeedInferenceError, match="disagree"):
        ledger.build_ledger(shard, seed=0, cap=_CAP, receipts_dir=receipts)


def test_build_ledger_proceeds_when_no_receipts_survive_at_all(
    shard: list[str], tmp_path: Path
) -> None:
    """Absence of evidence for check (iii) is NOT the same as disagreement -- an empty or
    missing receipts dir must not block the way a genuine conflict does."""
    result = ledger.build_ledger(shard, seed=0, cap=_CAP, receipts_dir=tmp_path / "no-receipts")
    assert result["manifest"]["seed"]["surviving_receipts_seed"] == {}


# ---------------------------------------------------------------------------------------
# The gate must be checked against the REQUESTED cap, not against whatever a draw
# happened to return -- a short/truncated draw must fail, not redefine its own target.
# ---------------------------------------------------------------------------------------


def test_gate_fails_on_a_short_draw_at_the_production_cap(shard: list[str]) -> None:
    """BLOCKING review finding, reproduced exactly: a shard with far fewer rows than the
    production cap (4,982), gated at the production cap. Before the fix,
    `verify_gate(union, r_first, p_first, cap=len(r_first))` rebound its own target to
    however many rows the short draw returned, so this passed and wrote a 30-row ledger
    self-certified against a cap of 30 while the manifest's top-level `cap` still said
    4982. A partial mount, truncated shard, or wrong --corpus-root must be caught here."""
    with pytest.raises(ledger.LedgerGateError):
        ledger.build_ledger(shard, seed=0, cap=4982, receipts_dir=Path("/nonexistent-receipts-dir"))


def test_build_ledger_gate_cap_matches_requested_cap_not_draw_length(shard: list[str]) -> None:
    """At a cap the shard CAN satisfy, the gate's recorded `cap` must be the requested
    cap -- not silently substituted for `len(r_first)` (which happens to equal it here,
    but for the right reason: draws are computed at `cap` themselves)."""
    result = ledger.build_ledger(shard, seed=0, cap=_CAP, receipts_dir=Path("/nonexistent"))
    assert result["manifest"]["gate"]["cap"] == _CAP
    assert result["manifest"]["cap"] == _CAP


# ---------------------------------------------------------------------------------------
# Discovered on-disk draws: every derived-sample file matching the cap row count is
# burned, and a ledger that omits one is refused. See the module docstring's "ORCHESTRATOR
# DECISION" and "EXIT CODE 3" sections.
# ---------------------------------------------------------------------------------------


def _write_ondisk_draw(
    root: Path, name: str, pairs: list[tuple[str, str]], *, corrupt: bool = False
) -> Path:
    """Writes `root/reason/aqua_rat-raw/derived/<name>` as a parquet file of `pairs`
    (or, when `corrupt=True`, garbage bytes at that path instead -- simulating a
    discovered file this script cannot read)."""
    derived = root / "reason" / "aqua_rat-raw" / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    path = derived / name
    if corrupt:
        path.write_bytes(b"not a parquet file")
        return path
    questions = [q for q, _ in pairs]
    rationales = [r for _, r in pairs]
    pq.write_table(pa.table({"question": questions, "rationale": rationales}), path)
    return path


def _pairs(n: int, tag: str = "ondisk") -> list[tuple[str, str]]:
    return [(f"{tag} question {i}", f"{tag} rationale {i}") for i in range(n)]


def test_discover_ondisk_draws_finds_a_file_matching_cap_row_count(tmp_path: Path) -> None:
    _write_ondisk_draw(tmp_path, "sample-25-seed0.parquet", _pairs(_CAP))
    draws, unreadable = ledger.discover_ondisk_draws(tmp_path, cap=_CAP)
    assert unreadable == []
    assert len(draws) == 1
    d = draws[0]
    assert d["name"] == "reason/aqua_rat-raw/derived/sample-25-seed0.parquet"
    assert d["row_count"] == _CAP
    assert len(d["pairs"]) == _CAP
    assert isinstance(d["sha256"], str) and len(d["sha256"]) == 64
    assert d["mtime_utc"]  # non-empty ISO timestamp


def test_discover_ondisk_draws_skips_a_file_with_a_different_row_count(tmp_path: Path) -> None:
    """A matched file this cap has nothing to do with is skipped without comment -- it is
    neither a draw nor an unreadable file."""
    _write_ondisk_draw(tmp_path, "sample-other.parquet", _pairs(_CAP - 1))
    draws, unreadable = ledger.discover_ondisk_draws(tmp_path, cap=_CAP)
    assert draws == []
    assert unreadable == []


def test_discover_ondisk_draws_reports_an_unreadable_file_separately(tmp_path: Path) -> None:
    """A file this script cannot read has UNKNOWN content -- it must never be silently
    skipped the way a wrong-row-count file is; see the module docstring's 'EXIT CODE 3'."""
    _write_ondisk_draw(tmp_path, "sample-corrupt.parquet", [], corrupt=True)
    draws, unreadable = ledger.discover_ondisk_draws(tmp_path, cap=_CAP)
    assert draws == []
    assert len(unreadable) == 1
    assert unreadable[0]["path"] == "reason/aqua_rat-raw/derived/sample-corrupt.parquet"
    assert unreadable[0]["error"]


def test_discover_ondisk_draws_no_matches_is_silently_fine(tmp_path: Path) -> None:
    draws, unreadable = ledger.discover_ondisk_draws(tmp_path, cap=_CAP)
    assert draws == []
    assert unreadable == []


def test_discover_ondisk_draws_drops_whitespace_only_sides_like_iter_pairs(
    tmp_path: Path,
) -> None:
    """Non-blocking review finding: discovery must use the SAME non-empty/strip filter
    `_iter_pairs`/`load_pairs` use -- a row with a whitespace-only side was never a
    candidate row for any draw, ondisk or otherwise."""
    pairs = _pairs(_CAP - 1) + [("   ", "not blank")]
    _write_ondisk_draw(tmp_path, "sample-ws.parquet", pairs)
    draws, unreadable = ledger.discover_ondisk_draws(tmp_path, cap=_CAP)
    assert unreadable == []
    assert len(draws) == 1
    assert draws[0]["row_count"] == _CAP  # raw row count unaffected
    assert len(draws[0]["pairs"]) == _CAP - 1  # the whitespace-only row is dropped


def test_discover_ondisk_draws_finds_multiple_matching_files(tmp_path: Path) -> None:
    """Discovery is not hardcoded to one filename -- any number of matching candidate
    draws are found, each named by its own relative path."""
    _write_ondisk_draw(tmp_path, "sample-4982-seed0.parquet", _pairs(_CAP, tag="alpha"))
    _write_ondisk_draw(tmp_path, "sample-4982-permute.parquet", _pairs(_CAP, tag="beta"))
    draws, unreadable = ledger.discover_ondisk_draws(tmp_path, cap=_CAP)
    assert unreadable == []
    assert sorted(d["name"] for d in draws) == [
        "reason/aqua_rat-raw/derived/sample-4982-permute.parquet",
        "reason/aqua_rat-raw/derived/sample-4982-seed0.parquet",
    ]


# ---------------------------------------------------------------------------------------
# verify_gate_for_draws: THE falsifier -- hand it a ledger missing a discovered draw
# (in full or in part) and confirm it refuses.
# ---------------------------------------------------------------------------------------


def test_verify_gate_for_draws_passes_when_every_draw_fully_covered() -> None:
    draw_a = _pairs(5, tag="a")
    draw_b = _pairs(3, tag="b")
    union = set(ledger.fingerprint_pairs(draw_a)) | set(ledger.fingerprint_pairs(draw_b))
    covered = ledger.verify_gate_for_draws(union, {"a": draw_a, "b": draw_b})
    assert covered == {"a": 5, "b": 3}


def test_verify_gate_for_draws_refuses_when_a_draw_is_entirely_missing() -> None:
    """THE falsifier: a ledger built without a discovered draw at all -- 0 rows covered
    -- must refuse exactly like a partially covered one, not pass because 'it just
    wasn't there'."""
    draw_a = _pairs(5, tag="a")
    draw_b = _pairs(3, tag="b")
    union = set(ledger.fingerprint_pairs(draw_a))  # draw_b entirely omitted

    with pytest.raises(ledger.LedgerGateError, match=r"'b': 0/3 rows"):
        ledger.verify_gate_for_draws(union, {"a": draw_a, "b": draw_b})


def test_verify_gate_for_draws_refuses_when_a_draw_is_partially_covered() -> None:
    draw_a = _pairs(5, tag="a")
    fps_a = ledger.fingerprint_pairs(draw_a)
    union = set(fps_a[:-1])  # every row of draw_a but one

    with pytest.raises(ledger.LedgerGateError, match=r"'a': 4/5 rows"):
        ledger.verify_gate_for_draws(union, {"a": draw_a})


# ---------------------------------------------------------------------------------------
# build_ledger folds discovered on-disk draws into the union, the manifest, and the gate.
# ---------------------------------------------------------------------------------------


def test_build_ledger_burns_a_discovered_ondisk_draw_and_folds_its_provenance_into_manifest(
    shard: list[str], tmp_path: Path
) -> None:
    _write_ondisk_draw(tmp_path, "sample-25-seed0.parquet", _pairs(_CAP))

    result = ledger.build_ledger(
        shard, seed=0, cap=_CAP, receipts_dir=tmp_path / "no-receipts", corpus_root=tmp_path
    )
    m = result["manifest"]
    assert m["unreadable_ondisk_draws"] == []
    assert len(m["ondisk_draws"]) == 1
    od = m["ondisk_draws"][0]
    assert od["name"] == "reason/aqua_rat-raw/derived/sample-25-seed0.parquet"
    assert od["row_count"] == _CAP
    assert od["rows_covered"] == od["rows_non_empty_pairs"] == _CAP

    ondisk_fps = set(ledger.fingerprint_pairs(_pairs(_CAP)))
    fps_in_ledger = {r["pair_fingerprint"] for r in result["rows"]}
    assert ondisk_fps <= fps_in_ledger
    assert m["union_size"] == len(fps_in_ledger)

    # A row that came ONLY from the discovered draw is tagged accordingly.
    ondisk_only_row = next(r for r in result["rows"] if r["pair_fingerprint"] in ondisk_fps)
    assert ondisk_only_row["draw"] == "ondisk"
    assert ondisk_only_row["also_in_ondisk_draws"] == [
        "reason/aqua_rat-raw/derived/sample-25-seed0.parquet"
    ]


def test_build_ledger_no_ondisk_draws_present_gives_empty_lists(
    shard: list[str], tmp_path: Path
) -> None:
    result = ledger.build_ledger(
        shard, seed=0, cap=_CAP, receipts_dir=tmp_path / "no-receipts", corpus_root=tmp_path
    )
    m = result["manifest"]
    assert m["ondisk_draws"] == []
    assert m["unreadable_ondisk_draws"] == []


def test_build_ledger_without_corpus_root_skips_discovery_entirely(
    shard: list[str], tmp_path: Path
) -> None:
    """`corpus_root=None` (the default) must not attempt discovery at all -- this is the
    unit-test path every other test in this file uses, and it must not require a real
    filesystem layout under `derived/`."""
    result = ledger.build_ledger(shard, seed=0, cap=_CAP, receipts_dir=tmp_path / "no-receipts")
    m = result["manifest"]
    assert m["ondisk_draws"] == []
    assert m["unreadable_ondisk_draws"] == []


# ---------------------------------------------------------------------------------------
# main(): exit codes. 0 when everything discovered is readable (and burned); 3 only when
# a discovered file could not be read. There is no acknowledgement flag any more -- the
# earlier "operator decision required" posture is gone (see the module docstring's
# "ORCHESTRATOR DECISION").
# ---------------------------------------------------------------------------------------


def test_main_exits_0_and_burns_a_discovered_ondisk_draw(shard: list[str], tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    aqua_dir = corpus_root / "reason" / "aqua_rat-raw"
    aqua_dir.mkdir(parents=True)
    _write_shard(aqua_dir / "train.parquet")
    _write_ondisk_draw(corpus_root, "sample-25-seed0.parquet", _pairs(_CAP))

    out = tmp_path / "ledger.jsonl"
    rc = ledger.main(
        [
            "--corpus-root",
            str(corpus_root),
            "--out",
            str(out),
            "--cap",
            str(_CAP),
            "--receipts-dir",
            str(tmp_path / "no-receipts"),
        ]
    )
    assert rc == 0
    manifest = json.loads(out.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["unreadable_ondisk_draws"] == []
    assert len(manifest["ondisk_draws"]) == 1
    assert manifest["ondisk_draws"][0]["rows_covered"] == _CAP


def test_main_exits_3_and_still_writes_the_ledger_when_a_discovered_file_is_unreadable(
    shard: list[str], tmp_path: Path
) -> None:
    corpus_root = tmp_path / "corpus"
    aqua_dir = corpus_root / "reason" / "aqua_rat-raw"
    aqua_dir.mkdir(parents=True)
    _write_shard(aqua_dir / "train.parquet")
    _write_ondisk_draw(corpus_root, "sample-corrupt.parquet", [], corrupt=True)

    out = tmp_path / "ledger.jsonl"
    rc = ledger.main(
        [
            "--corpus-root",
            str(corpus_root),
            "--out",
            str(out),
            "--cap",
            str(_CAP),
            "--receipts-dir",
            str(tmp_path / "no-receipts"),
        ]
    )
    assert rc == 3
    assert out.exists()  # written for review despite the non-zero exit, from what WAS readable
    manifest = json.loads(out.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["ondisk_draws"] == []
    assert len(manifest["unreadable_ondisk_draws"]) == 1
    assert (
        manifest["unreadable_ondisk_draws"][0]["path"]
        == "reason/aqua_rat-raw/derived/sample-corrupt.parquet"
    )


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
# Integration: the real corpus, if mounted. Reports row counts and the union size,
# including whatever on-disk derived draws are actually sitting there.
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
def test_integration_real_aqua_rat_union_covers_every_draw_in_full(tmp_path: Path) -> None:
    root = _REAL_ROOT
    assert root is not None
    shards = ledger.locate_shards(root)
    assert shards, f"no shards matched {ledger.SOURCE_GLOB!r} under {root}"

    result = ledger.build_ledger(
        shards, seed=0, receipts_dir=tmp_path / "no-receipts", corpus_root=root
    )
    m = result["manifest"]
    n_draws_total = 2 + len(m["ondisk_draws"])

    print(
        f"\n[integration] draw R: {m['draw_r_rows']} rows "
        f"({m['draw_r_unique_fingerprints']} unique)\n"
        f"[integration] draw P: {m['draw_p_rows']} rows "
        f"({m['draw_p_unique_fingerprints']} unique)\n"
        f"[integration] ondisk draws: {len(m['ondisk_draws'])} "
        f"({[d['name'] for d in m['ondisk_draws']]})\n"
        f"[integration] unreadable ondisk draws: {m['unreadable_ondisk_draws']}\n"
        f"[integration] union: {m['union_size']}  R∩P: {m['intersection_r_p_size']}\n"
        f"[integration] burned <= {ledger.CAP * n_draws_total} of 97,467 possible "
        f"({n_draws_total} draw(s))"
    )
    for d in m["ondisk_draws"]:
        print(
            f"[integration]   {d['name']}: {d['rows_covered']}/{d['rows_non_empty_pairs']} "
            f"rows covered, {d['row_count']} raw rows, sha256={d['sha256']}"
        )

    assert m["draw_r_rows"] == ledger.CAP
    assert m["draw_p_rows"] == ledger.CAP
    assert m["gate"]["passed"] is True
    assert m["gate"]["cap"] == ledger.CAP
    assert m["unreadable_ondisk_draws"] == []
    # Every discovered draw's rows are fully covered by construction (the union includes
    # them) -- assert it explicitly rather than trusting `gate.passed` alone.
    for d in m["ondisk_draws"]:
        assert d["rows_covered"] == d["rows_non_empty_pairs"]
    # DEC-42's original worst case (R and P alone, no overlap) generalises to N draws.
    assert m["union_size"] <= ledger.CAP * n_draws_total
    # And it should be a real recovery, not degenerate to fewer draws' worth of rows.
    assert m["union_size"] > ledger.CAP

    write_target = tmp_path / "burned-aqua_rat.jsonl"
    ledger.write_ledger(write_target, result)
    lines = write_target.read_text().splitlines()
    assert len(lines) == m["union_size"]

    # The known on-disk derived sample (reason/aqua_rat-raw/derived/sample-4982-seed0.parquet)
    # measured 4,982 raw rows -- if it is still present and unchanged, it must show up as
    # a discovered draw, fully covered, and not among the unreadable ones.
    known_name = "reason/aqua_rat-raw/derived/sample-4982-seed0.parquet"
    known = next((d for d in m["ondisk_draws"] if d["name"] == known_name), None)
    if known is not None:
        assert known["row_count"] == ledger.CAP
        assert known["rows_covered"] == known["rows_non_empty_pairs"]

        # The reviewer's own measurement against this same real corpus: draw R∪P is
        # 9,577 (4,951+4,946 unique minus 320 overlap), and the known on-disk draw
        # contributes 4,369 fingerprints not already in that union -- taking the full
        # union to 13,946 and the clean pool to 83,521 of 97,467.
        by_draw = {c["draw"]: c for c in m["draw_contributions"]}
        print(
            "[integration] draw contributions: "
            + ", ".join(
                f"{c['draw']!r}: {c['new_fingerprints']} new (of {c['unique_fingerprints']} unique)"
                for c in m["draw_contributions"]
            )
        )
        assert by_draw[known_name]["new_fingerprints"] == 4_369
        assert m["union_size"] == 13_946
        pfn = m["pool_floor_note"]
        assert pfn["clean_pool_size"] == 83_521
        assert pfn["exceeds_design_doc_bound"] is True
        print(f"[integration] pool_floor_note: {pfn}")


@needs_real_corpus
def test_integration_ledger_write_is_idempotent_against_the_real_corpus(tmp_path: Path) -> None:
    root = _REAL_ROOT
    assert root is not None
    shards = ledger.locate_shards(root)
    receipts_dir = tmp_path / "no-receipts"

    out1 = tmp_path / "run1.jsonl"
    out2 = tmp_path / "run2.jsonl"
    ledger.write_ledger(
        out1, ledger.build_ledger(shards, seed=0, receipts_dir=receipts_dir, corpus_root=root)
    )
    ledger.write_ledger(
        out2, ledger.build_ledger(shards, seed=0, receipts_dir=receipts_dir, corpus_root=root)
    )

    assert out1.read_text() == out2.read_text()
