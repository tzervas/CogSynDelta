"""Unit tests for scripts/csd-corpus-admit.py.

Per `verify-guards-by-making-them-fail`: every check below is exercised with a
constructed input that MUST fail it, not only with inputs that pass, so a guard that is
structurally incapable of firing would show up as a failing test rather than as a silent
pass. Loaded by path (hyphenated filename, not an importable module name) the same way
tests/test_publish_checkpoint.py loads scripts/csd-publish-checkpoint.py.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "csd-corpus-admit.py"


def load_mod() -> Any:
    loader = importlib.machinery.SourceFileLoader("csd_corpus_admit", str(SCRIPT))
    spec = importlib.util.spec_from_loader("csd_corpus_admit", loader)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["csd_corpus_admit"] = mod
    loader.exec_module(mod)
    return mod


mod = load_mod()


# --------------------------------------------------------------------------------------
# Check 1: licence tier vs. the region's declared tier
# --------------------------------------------------------------------------------------


def test_licence_tier_passes_when_verdict_no_stricter_than_region() -> None:
    result = mod.check_licence_tier("PERMISSIVE_OK", "code", mod.REGION_TIER)
    assert result.passed


def test_licence_tier_fails_when_verdict_stricter_than_region() -> None:
    # `code` is mit; an NC verdict demands cc-by-nc-sa-4.0, strictly stricter -- MUST fail.
    result = mod.check_licence_tier("NC", "code", mod.REGION_TIER)
    assert not result.passed
    assert "requires an EXPLICIT tier upgrade" in result.detail


def test_licence_tier_passes_when_region_already_at_or_above_required_tier() -> None:
    # `retrieve` is already cc-by-nc-sa-4.0 (the strictest tier); a SHARE_ALIKE verdict
    # (cc-by-sa-4.0) demands nothing stricter than what the region already carries.
    result = mod.check_licence_tier("SHARE_ALIKE", "retrieve", mod.REGION_TIER)
    assert result.passed


def test_licence_tier_fails_on_refuse_verdict_regardless_of_region() -> None:
    result = mod.check_licence_tier("REFUSE", "code", mod.REGION_TIER)
    assert not result.passed


def test_licence_tier_fails_on_blocking_verdict_regardless_of_region() -> None:
    result = mod.check_licence_tier("BLOCKING", "retrieve", mod.REGION_TIER)
    assert not result.passed


def test_licence_tier_fails_on_unverified_verdict_regardless_of_region() -> None:
    result = mod.check_licence_tier("UNVERIFIED", "retrieve", mod.REGION_TIER)
    assert not result.passed


def test_licence_tier_fails_refuses_rather_than_guesses_unknown_region() -> None:
    # No silent mit default for a region absent from REGION_TIER (residual_mlp,
    # stream_vae were never audited) -- MUST fail, not pass by falling through.
    result = mod.check_licence_tier("PERMISSIVE_OK", "residual_mlp", mod.REGION_TIER)
    assert not result.passed
    assert "no declared tier" in result.detail


def test_licence_tier_matches_publish_checkpoint() -> None:
    """The transcribed REGION_TIER table must not silently drift from the source of
    truth in scripts/csd-publish-checkpoint.py. Loaded the same way that module's own
    tests load it, so this fails the day the two tables disagree rather than the day
    someone happens to notice."""
    publish_script = ROOT / "scripts" / "csd-publish-checkpoint.py"
    loader = importlib.machinery.SourceFileLoader("csd_publish_checkpoint_ref", str(publish_script))
    spec = importlib.util.spec_from_loader("csd_publish_checkpoint_ref", loader)
    assert spec is not None
    ref_mod = importlib.util.module_from_spec(spec)
    sys.modules["csd_publish_checkpoint_ref"] = ref_mod
    # csd-publish-checkpoint.py imports huggingface_hub lazily only inside publish();
    # top-level exec does not require it, same reasoning test_publish_checkpoint.py uses.
    if "huggingface_hub" not in sys.modules:
        try:
            import huggingface_hub  # noqa: F401
        except ModuleNotFoundError:
            import types

            stub = types.ModuleType("huggingface_hub")
            stub.HfApi = object
            sys.modules["huggingface_hub"] = stub
    loader.exec_module(ref_mod)
    assert mod.REGION_TIER == ref_mod.LICENCE_TIER


# --------------------------------------------------------------------------------------
# Check 2: B1 provenance-group share cap
# --------------------------------------------------------------------------------------


def test_b1_share_passes_under_cap() -> None:
    result = mod.check_b1_share("narrativeqa", 1000, {"gooaq": 3000})
    # 1000 / 4000 = 0.25 <= 0.40
    assert result.passed


def test_b1_share_fails_over_cap() -> None:
    # A dataset added into an empty region is 100% of the region -- MUST fail B1.
    result = mod.check_b1_share("solo_group", 500, {})
    assert not result.passed
    assert "1.0000" in result.detail


def test_b1_share_fails_when_pushed_over_cap_by_existing_dominance() -> None:
    # group already at 0.35, add more of the SAME group -- must cross 0.40 and fail.
    result = mod.check_b1_share("gooaq", 3500, {"gooaq": 3500, "other": 6500})
    # (3500+3500) / (10000+3500) = 7000/13500 = 0.5185
    assert not result.passed


def test_b1_share_passes_exactly_at_cap_boundary() -> None:
    # 40/100 = 0.40 exactly -- the spec says <= 0.40, so this must pass, not fail.
    result = mod.check_b1_share("x", 40, {"x": 0, "y": 60})
    assert result.passed


def test_b1_share_fails_just_over_cap_boundary() -> None:
    result = mod.check_b1_share("x", 41, {"x": 0, "y": 60})
    assert not result.passed


def test_b1_share_fails_on_negative_added_count() -> None:
    result = mod.check_b1_share("x", -5, {"x": 10})
    assert not result.passed


def test_b1_share_computed_on_provenance_group_not_dataset_name() -> None:
    # Two different-named datasets in the SAME provenance group must be summed together,
    # not treated as two independent 20%-share entries -- this is the check that would
    # stay silently green if share were computed per dataset-id instead of per group.
    existing = {"s2orc": 3000, "other": 7000}
    result = mod.check_b1_share("s2orc", 2000, existing)
    # (3000+2000) / (10000+2000) = 5000/12000 = 0.4167 -- over cap
    assert not result.passed


# --------------------------------------------------------------------------------------
# Check 3: verification_status == VERIFIED
# --------------------------------------------------------------------------------------


def test_verified_status_passes() -> None:
    result = mod.check_verified("VERIFIED")
    assert result.passed


def test_contradicted_status_fails() -> None:
    result = mod.check_verified("CONTRADICTED")
    assert not result.passed


def test_unverifiable_status_fails() -> None:
    result = mod.check_verified("UNVERIFIABLE")
    assert not result.passed


def test_refused_closed_status_fails() -> None:
    result = mod.check_verified("REFUSED-CLOSED")
    assert not result.passed


def test_empty_status_fails() -> None:
    result = mod.check_verified("")
    assert not result.passed


# --------------------------------------------------------------------------------------
# run_checklist integration + CLI exit codes
# --------------------------------------------------------------------------------------

# Uses google-research-datasets/paws's real repo_id/provenance_group (see
# tests/fixtures/csd_corpus_admit/provenance-paws.json and the real catalogue row) so it
# stays clean under the default `--catalogue` lookup too, not just under checks 1-3 run in
# isolation -- picking a repo_id absent from the real catalogue would make this fixture's
# "clean" status an artifact of check 4 not finding a row at all (fail-open by omission)
# rather than of the row genuinely being clean.
CLEAN_PROVENANCE: dict[str, Any] = {
    "repo_id": "google-research-datasets/paws",
    "faculty": "memory",
    "provenance_group": "paws",
    "verdict": "PERMISSIVE_OK",
    "verification_status": "VERIFIED",
    "row_count": 1000,
}


def test_run_checklist_all_pass() -> None:
    results = mod.run_checklist(CLEAN_PROVENANCE, "memory", {"gooaq": 3000})
    assert all(r.passed for r in results)
    assert {r.name for r in results} == {"licence_tier", "b1_share", "verification_status"}


def test_run_checklist_fails_when_any_single_check_fails() -> None:
    bad = dict(CLEAN_PROVENANCE, verification_status="CONTRADICTED")
    results = mod.run_checklist(bad, "memory", {"gooaq": 3000})
    assert not all(r.passed for r in results)


def test_cli_exit_zero_on_admit(tmp_path: Path) -> None:
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps(CLEAN_PROVENANCE), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--provenance",
            str(prov),
            "--region",
            "memory",
            "--existing-shares",
            json.dumps({"gooaq": 3000}),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ADMIT" in proc.stdout


def test_cli_exit_nonzero_on_refuse(tmp_path: Path) -> None:
    bad = dict(CLEAN_PROVENANCE, verdict="REFUSE")
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps(bad), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--provenance",
            str(prov),
            "--region",
            "memory",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "REFUSE" in proc.stdout


def test_cli_rejects_malformed_existing_shares_json(tmp_path: Path) -> None:
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps(CLEAN_PROVENANCE), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--provenance",
            str(prov),
            "--region",
            "memory",
            "--existing-shares",
            "{not json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2


# --------------------------------------------------------------------------------------
# resolve_added_count: real provenance.json never emits row_count/count (B4) -- the
# estimate-from-bytes fallback and its ESTIMATED label
# --------------------------------------------------------------------------------------


def test_resolve_added_count_prefers_explicit_row_count() -> None:
    count, estimated = mod.resolve_added_count({"row_count": 42, "total_bytes": 999999})
    assert (count, estimated) == (42, False)


def test_resolve_added_count_prefers_explicit_count_field() -> None:
    count, estimated = mod.resolve_added_count({"count": 7})
    assert (count, estimated) == (7, False)


def test_resolve_added_count_estimates_from_total_bytes_when_absent() -> None:
    # Real provenance.json shape: total_bytes present, no row_count/count -- MUST estimate,
    # not silently return 0 (that was B4's actual bug: the old code's `or 0` fallback made
    # every real admission compute a share of zero rows, so B1 could never fire).
    count, estimated = mod.resolve_added_count({"total_bytes": 129299737})
    assert estimated is True
    assert count == 129299737 // mod.BYTES_PER_TOKEN


def test_resolve_added_count_zero_when_neither_row_count_nor_bytes_present() -> None:
    count, estimated = mod.resolve_added_count({})
    assert (count, estimated) == (0, False)


def test_b1_share_detail_notes_estimated_count() -> None:
    result = mod.check_b1_share("x", 1000, {"y": 1000}, estimated=True)
    assert "ESTIMATED" in result.detail


def test_b1_share_detail_omits_estimated_note_for_real_count() -> None:
    result = mod.check_b1_share("x", 1000, {"y": 1000}, estimated=False)
    assert "ESTIMATED" not in result.detail


# --------------------------------------------------------------------------------------
# repo_id_of: real provenance.json has no top-level repo_id, only catalogue_entry.repo_id
# --------------------------------------------------------------------------------------


def test_repo_id_of_prefers_top_level_field() -> None:
    assert mod.repo_id_of({"repo_id": "a/b", "catalogue_entry": {"repo_id": "c/d"}}) == "a/b"


def test_repo_id_of_falls_back_to_catalogue_entry() -> None:
    # The real factory shape: no top-level repo_id at all.
    assert mod.repo_id_of({"catalogue_entry": {"repo_id": "deepmind/narrativeqa"}}) == (
        "deepmind/narrativeqa"
    )


def test_repo_id_of_unknown_when_neither_present() -> None:
    assert mod.repo_id_of({}) == "<unknown repo_id>"


# --------------------------------------------------------------------------------------
# Check 4: catalogue structural refusals -- a second, independent look at fields the
# factory's own gate parses into provenance.json but never gates on (B3), read here from
# the full catalogue row, not from provenance.json's trimmed catalogue_entry.
# --------------------------------------------------------------------------------------

CLEAN_ROW: dict[str, Any] = {
    "repo_id": "clean/row",
    "grant_scope": "whole_corpus",
    "enrichment_licence_result": "no obligation propagates; clean-permissive tier",
    "redistribute": {"nc": False, "sa": False, "nd": False, "attribution": False},
    "provenance_red_flags": [],
}


def test_catalogue_check_passes_on_clean_row() -> None:
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": CLEAN_ROW})
    assert result.passed


def test_catalogue_check_fails_when_repo_id_missing_from_index() -> None:
    # Fail closed: an entry this check cannot evaluate is refused, not silently admitted.
    result = mod.check_catalogue_structural_refusals("nowhere/found", {"clean/row": CLEAN_ROW})
    assert not result.passed
    assert "not found in the catalogue index" in result.detail


def test_catalogue_check_fails_on_non_full_content_grant_scope() -> None:
    # Mirrors B1: deepmind/narrativeqa's real grant_scope is code_only.
    row = dict(CLEAN_ROW, grant_scope="code_only")
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert not result.passed
    assert "grant_scope" in result.detail


def test_catalogue_check_fails_on_metadata_only_grant_scope() -> None:
    row = dict(CLEAN_ROW, grant_scope="metadata_only")
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert not result.passed


def test_catalogue_check_passes_on_database_rights_only_grant_scope() -> None:
    row = dict(CLEAN_ROW, grant_scope="database_rights_only")
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert result.passed


def test_catalogue_check_fails_on_enrichment_refuse_marker() -> None:
    # Mirrors B2: BeIR/cqadupstack's real enrichment_licence_result is
    # "R9 REFUSE at ingest per 20-enrichment ...".
    row = dict(CLEAN_ROW, enrichment_licence_result="R9 REFUSE at ingest per 20-enrichment")
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert not result.passed
    assert "REFUSE marker" in result.detail


def test_catalogue_check_enrichment_refuse_marker_is_case_insensitive() -> None:
    row = dict(CLEAN_ROW, enrichment_licence_result="quietly refuse at ingest")
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert not result.passed


def test_catalogue_check_fails_on_redistribute_nd() -> None:
    row = dict(CLEAN_ROW, redistribute={"nc": False, "sa": False, "nd": True, "attribution": False})
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert not result.passed
    assert "no-derivatives" in result.detail


def test_catalogue_check_missing_redistribute_field_does_not_crash_or_fail() -> None:
    row = {k: v for k, v in CLEAN_ROW.items() if k != "redistribute"}
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert result.passed


def test_catalogue_check_red_flags_alone_do_not_fail() -> None:
    # google-research-datasets/paws (the clean reference case) carries a
    # provenance_red_flags entry and is still admissible -- gating on presence alone
    # would refuse the survey's cleanest grant.
    row = dict(CLEAN_ROW, provenance_red_flags=["built from separately-licensed sentences"])
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert result.passed
    assert "provenance_red_flags" in result.detail  # surfaced, even though it did not fail


# --------------------------------------------------------------------------------------
# run_checklist with catalogue_index: backward compatible when omitted (checks 1-3 only,
# matching the pre-B4-fix test_run_checklist_* tests above), and a 4th result when given.
# --------------------------------------------------------------------------------------


def test_run_checklist_without_catalogue_index_has_three_checks() -> None:
    results = mod.run_checklist(CLEAN_PROVENANCE, "memory", {"gooaq": 3000})
    assert {r.name for r in results} == {"licence_tier", "b1_share", "verification_status"}


def test_run_checklist_with_catalogue_index_has_four_checks() -> None:
    results = mod.run_checklist(
        CLEAN_PROVENANCE, "memory", {"gooaq": 3000}, catalogue_index={"clean/row": CLEAN_ROW}
    )
    names = {r.name for r in results}
    assert "catalogue_structural_refusals" in names
    assert len(results) == 4


def test_run_checklist_refuses_on_catalogue_structural_refusal_alone() -> None:
    # Same shape as the real B1 defect: verdict/verification_status are both clean, only
    # the catalogue row's grant_scope is a refusal -- checks 1-3 would all PASS on their
    # own (this is exactly what let narrativeqa land), so this proves check 4 is the one
    # that actually catches it.
    provenance = dict(CLEAN_PROVENANCE, repo_id="narrow/scope")
    bad_row = dict(CLEAN_ROW, repo_id="narrow/scope", grant_scope="code_only")
    results = mod.run_checklist(
        provenance, "memory", {"gooaq": 3000}, catalogue_index={"narrow/scope": bad_row}
    )
    assert not all(r.passed for r in results)
    catalogue_result = next(r for r in results if r.name == "catalogue_structural_refusals")
    assert not catalogue_result.passed


# --------------------------------------------------------------------------------------
# load_catalogue_index against the real bundled catalogue
# --------------------------------------------------------------------------------------

REAL_CATALOGUE = ROOT / "docs" / "design" / "datasets" / "catalogue-2026-09-03.json"
FIXTURES = ROOT / "tests" / "fixtures" / "csd_corpus_admit"


def test_load_catalogue_index_finds_known_repo_ids() -> None:
    index = mod.load_catalogue_index(REAL_CATALOGUE)
    assert "google-research-datasets/paws" in index
    assert "deepmind/narrativeqa" in index
    assert "BeIR/cqadupstack" in index


def test_default_catalogue_path_matches_real_catalogue() -> None:
    assert mod.DEFAULT_CATALOGUE_PATH == REAL_CATALOGUE
    assert mod.DEFAULT_CATALOGUE_PATH.is_file()


# --------------------------------------------------------------------------------------
# Integration against REAL provenance.json fixtures copied from the 2026-09-03 factory
# run (/mnt/bulk/csd-corpus/factory-2026-09-03/memory/) and the real bundled catalogue --
# proves the fixed tool actually admits the clean case and refuses the two ground-pass
# defects (B1 narrativeqa, B2 BeIR/cqadupstack) end to end, not just at the unit level.
# --------------------------------------------------------------------------------------


def _real_catalogue_index() -> dict[str, dict[str, Any]]:
    return mod.load_catalogue_index(REAL_CATALOGUE)


def test_real_paws_fixture_is_admitted() -> None:
    provenance = json.loads((FIXTURES / "provenance-paws.json").read_text(encoding="utf-8"))
    results = mod.run_checklist(
        provenance,
        "memory",
        {"other": 1_000_000_000},
        catalogue_index=_real_catalogue_index(),
    )
    assert all(r.passed for r in results), [(r.name, r.detail) for r in results if not r.passed]


def test_real_narrativeqa_fixture_is_refused_on_grant_scope() -> None:
    # This is B1 verbatim: verdict PERMISSIVE_OK and verification_status VERIFIED (checks
    # 1 and 3 both pass), but grant_scope=code_only in the real catalogue row -- only
    # check 4 catches it.
    provenance = json.loads((FIXTURES / "provenance-narrativeqa.json").read_text(encoding="utf-8"))
    results = mod.run_checklist(
        provenance,
        "memory",
        {"other": 1_000_000_000},
        catalogue_index=_real_catalogue_index(),
    )
    assert not all(r.passed for r in results)
    catalogue_result = next(r for r in results if r.name == "catalogue_structural_refusals")
    assert not catalogue_result.passed
    assert "code_only" in catalogue_result.detail
    licence_result = next(r for r in results if r.name == "licence_tier")
    verified_result = next(r for r in results if r.name == "verification_status")
    assert licence_result.passed
    assert verified_result.passed


def test_real_cqadupstack_fixture_is_refused_on_enrichment_marker() -> None:
    # B2 verbatim: SHARE_ALIKE/VERIFIED against the `retrieve` region (already
    # cc-by-nc-sa-4.0, so check 1 passes) -- only the catalogue row's
    # enrichment_licence_result "R9 REFUSE at ingest" catches it.
    provenance = json.loads((FIXTURES / "provenance-cqadupstack.json").read_text(encoding="utf-8"))
    results = mod.run_checklist(
        provenance,
        "retrieve",
        {"other": 10_000_000_000},
        catalogue_index=_real_catalogue_index(),
    )
    assert not all(r.passed for r in results)
    catalogue_result = next(r for r in results if r.name == "catalogue_structural_refusals")
    assert not catalogue_result.passed
    assert "REFUSE marker" in catalogue_result.detail
    licence_result = next(r for r in results if r.name == "licence_tier")
    assert licence_result.passed


def test_real_narrativeqa_fixture_row_count_is_estimated_from_bytes() -> None:
    # B4's other half: the real provenance.json has no row_count/count field at all.
    provenance = json.loads((FIXTURES / "provenance-narrativeqa.json").read_text(encoding="utf-8"))
    added_count, estimated = mod.resolve_added_count(provenance)
    assert estimated is True
    assert added_count == provenance["total_bytes"] // mod.BYTES_PER_TOKEN
    assert added_count > 0


# --------------------------------------------------------------------------------------
# CLI end to end against the real fixtures + real catalogue (the default --catalogue)
# --------------------------------------------------------------------------------------


def test_cli_admits_real_paws_fixture_against_default_catalogue() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--provenance",
            str(FIXTURES / "provenance-paws.json"),
            "--region",
            "memory",
            "--existing-shares",
            json.dumps({"other": 1_000_000_000}),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ADMIT" in proc.stdout


def test_cli_refuses_real_narrativeqa_fixture_against_default_catalogue() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--provenance",
            str(FIXTURES / "provenance-narrativeqa.json"),
            "--region",
            "memory",
            "--existing-shares",
            json.dumps({"other": 1_000_000_000}),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "REFUSE" in proc.stdout
    assert "catalogue_structural_refusals" in proc.stdout
    assert "FAIL" in proc.stdout


def test_cli_no_catalogue_check_flag_skips_check_4() -> None:
    # narrativeqa would REFUSE under check 4; with it skipped, checks 1-3 alone admit it
    # (verdict PERMISSIVE_OK, VERIFIED, and B1 headroom) -- proves the flag actually
    # disables the check rather than being ignored.
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--provenance",
            str(FIXTURES / "provenance-narrativeqa.json"),
            "--region",
            "memory",
            "--existing-shares",
            json.dumps({"other": 10_000_000_000}),
            "--no-catalogue-check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "catalogue_structural_refusals" not in proc.stdout
