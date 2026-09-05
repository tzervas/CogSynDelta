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
import tempfile
from pathlib import Path
from typing import Any

import pytest

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


def test_licence_tier_accepts_either_spelling_of_a_renamed_region() -> None:
    """`--region code` and `--region language` (cogsyndelta.regions.aliases) must
    resolve to the SAME REGION_TIER entry -- the table is keyed canonically."""
    legacy = mod.check_licence_tier("PERMISSIVE_OK", "code", mod.REGION_TIER)
    canonical = mod.check_licence_tier("PERMISSIVE_OK", "language", mod.REGION_TIER)
    assert legacy.passed
    assert canonical.passed


def test_licence_tier_accepts_either_spelling_of_visual_too() -> None:
    legacy = mod.check_licence_tier("PERMISSIVE_OK", "vl_latent", mod.REGION_TIER)
    canonical = mod.check_licence_tier("PERMISSIVE_OK", "visual", mod.REGION_TIER)
    assert legacy.passed
    assert canonical.passed


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
    # --no-catalogue-check: this test is about basic CLI plumbing (exit code, JSON
    # existing-shares parsing), not catalogue content -- CLEAN_PROVENANCE's real repo_id
    # (google-research-datasets/paws) is refused by the real default catalogue pending
    # red-flag resolution (see the dedicated real-fixture tests for that).
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
            "--no-catalogue-check",
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
            "--existing-shares",
            "{}",
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


def test_cli_errors_one_line_when_existing_shares_and_corpus_root_both_omitted(
    tmp_path: Path,
) -> None:
    # Round-2 review non-blocking item: a bare invocation must error with a one-line
    # instruction, never silently REFUSE (the old default of "{}" forced b1_share to 1.0
    # and REFUSE on every default invocation, indistinguishable from a real refusal).
    prov = tmp_path / "provenance.json"
    prov.write_text(json.dumps(CLEAN_PROVENANCE), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--provenance", str(prov), "--region", "memory"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "REFUSE" not in proc.stdout
    assert "--existing-shares" in proc.stderr
    assert "--corpus-root" in proc.stderr
    # "one-line instruction": the usage error itself, not a stack trace.
    assert "Traceback" not in proc.stderr


def test_cli_errors_when_existing_shares_and_corpus_root_both_given(tmp_path: Path) -> None:
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
            "{}",
            "--corpus-root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "only one of" in proc.stderr


def test_cli_derives_existing_shares_from_corpus_root(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    existing_ds_dir = corpus_root / "memory" / "gooaq"
    existing_ds_dir.mkdir(parents=True)
    (existing_ds_dir / "provenance.json").write_text(
        json.dumps({"provenance_group": "gooaq", "row_count": 3000}), encoding="utf-8"
    )
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
            "--corpus-root",
            str(corpus_root),
            "--no-catalogue-check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # CLEAN_PROVENANCE's own provenance_group is "paws" (distinct from "gooaq"), 1000
    # rows added -- 1000 / (3000 + 1000) = 0.25 <= 0.40, same arithmetic as
    # test_b1_share_passes_under_cap, now proven to arrive via --corpus-root disk scan
    # rather than a hand-typed --existing-shares value.
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ADMIT" in proc.stdout
    assert "1000/4000" in proc.stdout


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


def test_catalogue_check_fails_on_database_rights_only_grant_scope() -> None:
    # B4: `database_rights_only` licenses the compilation/database right, not the
    # individual contents (allenai/wildguardmix's real grant, ODC-By ss2.4) -- this tool
    # used to admit it (mismatching the factory, which has always refused it), letting
    # wildguardmix through this second-look gate while the factory refused it outright.
    row = dict(CLEAN_ROW, grant_scope="database_rights_only")
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert not result.passed
    assert "grant_scope" in result.detail


def test_catalogue_check_fails_on_enrichment_refuse_marker() -> None:
    # Mirrors B2: BeIR/cqadupstack's real enrichment_licence_result is
    # "R9 REFUSE at ingest per 20-enrichment ...".
    row = dict(CLEAN_ROW, enrichment_licence_result="R9 REFUSE at ingest per 20-enrichment")
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert not result.passed
    assert "REFUSE/NONE ADMISSIBLE marker" in result.detail


def test_catalogue_check_enrichment_refuse_marker_is_case_sensitive_not_insensitive() -> None:
    # Round-2 review, non-blocking item: this tool used to match "refuse" case-
    # INSENSITIVELY, disagreeing with the factory's case-sensitive \bREFUSE[SD]?\b on 3
    # catalogue rows. Ordinary lower-case prose that merely discusses refusal elsewhere
    # ("refused at ingest (R4)" as a sub-clause note) must NOT trigger the gate -- only
    # the catalogue's own shouting-case authoring convention does.
    row = dict(CLEAN_ROW, enrichment_licence_result="quietly refuse at ingest (lower case)")
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert result.passed


def test_catalogue_check_enrichment_refuse_marker_matches_shouting_case() -> None:
    row = dict(CLEAN_ROW, enrichment_licence_result="R9 REFUSE at ingest per 20-enrichment")
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert not result.passed


def test_catalogue_check_enrichment_refuse_marker_matches_refused_and_refuses_variants() -> None:
    for word in ("REFUSED", "REFUSES"):
        row = dict(CLEAN_ROW, enrichment_licence_result=f"{word} at ingest")
        result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
        assert not result.passed, word


def test_catalogue_check_none_admissible_marker_fails() -> None:
    row = dict(CLEAN_ROW, enrichment_licence_result="NONE ADMISSIBLE today under any tier")
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert not result.passed


def test_catalogue_check_enrichment_marker_checked_on_enrichment_plan_too() -> None:
    # Mirrors the factory's _has_enrichment_refusal_marker, which checks BOTH
    # enrichment_licence_result and enrichment_plan.
    row = dict(CLEAN_ROW, enrichment_plan="R9 REFUSE at ingest")
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


def test_catalogue_check_red_flag_without_resolution_fails() -> None:
    # A non-empty provenance_red_flags with no provenance_red_flags_resolution entry is
    # UNRESOLVED, per RED_FLAG_RESOLUTION_RULE, and MUST fail -- this is the real,
    # currently-live state of google-research-datasets/paws's catalogue row (round-2
    # review B1: the factory's real RED_FLAGS_BLOCK_ADMISSION gate already refuses paws
    # for exactly this reason). Gating on unresolved presence is deliberate; see the
    # resolved-entry test below for the case that does NOT fail.
    row = dict(CLEAN_ROW, provenance_red_flags=["built from separately-licensed sentences"])
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert not result.passed
    assert "unresolved provenance_red_flag" in result.detail
    assert "provenance_red_flags=" in result.detail  # surfaced in the failing detail too


def test_catalogue_check_red_flag_with_resolved_true_entry_passes() -> None:
    flag = "built from separately-licensed sentences"
    row = dict(
        CLEAN_ROW,
        provenance_red_flags=[flag],
        provenance_red_flags_resolution=[{"flag": flag, "resolved": True}],
    )
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert result.passed
    assert "provenance_red_flags" in result.detail  # surfaced even though it did not fail


def test_catalogue_check_red_flag_resolution_must_match_flag_text_exactly() -> None:
    # A resolution entry for a DIFFERENT flag text does not resolve this one.
    row = dict(
        CLEAN_ROW,
        provenance_red_flags=["flag A"],
        provenance_red_flags_resolution=[{"flag": "flag B", "resolved": True}],
    )
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert not result.passed


def test_catalogue_check_red_flag_resolution_entry_must_be_resolved_true_not_just_present() -> None:
    # A resolution entry that exists for the right flag but has resolved=False (or a
    # missing/truthy-but-not-True `resolved`) does NOT resolve it -- presence alone is
    # not enough, only an explicit resolved=True.
    flag = "built from separately-licensed sentences"
    row = dict(
        CLEAN_ROW,
        provenance_red_flags=[flag],
        provenance_red_flags_resolution=[{"flag": flag, "resolved": False}],
    )
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert not result.passed


def test_catalogue_check_multiple_red_flags_partially_resolved_still_fails() -> None:
    row = dict(
        CLEAN_ROW,
        provenance_red_flags=["flag A", "flag B"],
        provenance_red_flags_resolution=[{"flag": "flag A", "resolved": True}],
    )
    result = mod.check_catalogue_structural_refusals("clean/row", {"clean/row": row})
    assert not result.passed
    # exactly the still-unresolved flag B is named as unresolved, not the resolved flag A
    assert mod._unresolved_red_flags(
        ["flag A", "flag B"], [{"flag": "flag A", "resolved": True}]
    ) == ["flag B"]
    assert "flag B" in result.detail


# --------------------------------------------------------------------------------------
# _unresolved_red_flags directly
# --------------------------------------------------------------------------------------


def test_unresolved_red_flags_empty_when_no_flags() -> None:
    assert mod._unresolved_red_flags([], []) == []


def test_unresolved_red_flags_all_unresolved_when_no_resolution_list() -> None:
    assert mod._unresolved_red_flags(["a", "b"], []) == ["a", "b"]


def test_unresolved_red_flags_ignores_non_dict_resolution_entries() -> None:
    # A malformed resolution list entry (not a dict) must not crash, and must not
    # resolve anything.
    assert mod._unresolved_red_flags(["a"], ["not-a-dict"]) == ["a"]  # type: ignore[list-item]


# --------------------------------------------------------------------------------------
# run_checklist with catalogue_index: backward compatible when omitted (checks 1-3 only,
# matching the pre-B4-fix test_run_checklist_* tests above), and checks 4+5 when given.
# --------------------------------------------------------------------------------------


def test_run_checklist_without_catalogue_index_has_three_checks() -> None:
    results = mod.run_checklist(CLEAN_PROVENANCE, "memory", {"gooaq": 3000})
    assert {r.name for r in results} == {"licence_tier", "b1_share", "verification_status"}


def test_run_checklist_with_catalogue_index_has_five_checks() -> None:
    results = mod.run_checklist(
        CLEAN_PROVENANCE, "memory", {"gooaq": 3000}, catalogue_index={"clean/row": CLEAN_ROW}
    )
    names = {r.name for r in results}
    assert "catalogue_structural_refusals" in names
    assert "policy_constants_drift" in names
    assert len(results) == 5


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


def test_real_paws_fixture_is_refused_pending_red_flag_resolution() -> None:
    # Round-2 review B1, verbatim: paws is NOT the clean reference case it used to be in
    # this suite -- the real catalogue row carries a provenance_red_flags entry with no
    # provenance_red_flags_resolution entry, so it is UNRESOLVED per RED_FLAG_RESOLUTION_
    # RULE and this second-look gate refuses it, same as the factory's real (already-
    # committed) RED_FLAGS_BLOCK_ADMISSION gate does today. This will flip back to ADMIT
    # once TASK A adds a resolution entry to paws's catalogue row -- see
    # test_real_paws_fixture_is_admitted_once_red_flag_resolved for proof the mechanism
    # itself does admit it once that entry exists.
    provenance = json.loads((FIXTURES / "provenance-paws.json").read_text(encoding="utf-8"))
    results = mod.run_checklist(
        provenance,
        "memory",
        {"other": 1_000_000_000},
        catalogue_index=_real_catalogue_index(),
    )
    assert not all(r.passed for r in results)
    catalogue_result = next(r for r in results if r.name == "catalogue_structural_refusals")
    assert not catalogue_result.passed
    assert "unresolved provenance_red_flag" in catalogue_result.detail
    # every OTHER check still passes -- this is check 4 alone catching it, same discipline
    # as the narrativeqa/cqadupstack ground-pass proofs below.
    licence_result = next(r for r in results if r.name == "licence_tier")
    verified_result = next(r for r in results if r.name == "verification_status")
    assert licence_result.passed
    assert verified_result.passed


def test_real_paws_fixture_is_admitted_once_red_flag_resolved() -> None:
    # Proves the resolution mechanism actually admits, not just that it refuses:
    # take the REAL catalogue row for paws (untouched on disk -- this only mutates an
    # in-memory copy) and add the one resolution entry TASK A is expected to add, for
    # paws's own real red-flag text read straight off the row.
    real_row = _real_catalogue_index()["google-research-datasets/paws"]
    (real_flag,) = real_row["provenance_red_flags"]
    resolved_row = dict(
        real_row,
        provenance_red_flags_resolution=[{"flag": real_flag, "resolved": True}],
    )
    provenance = json.loads((FIXTURES / "provenance-paws.json").read_text(encoding="utf-8"))
    results = mod.run_checklist(
        provenance,
        "memory",
        {"other": 1_000_000_000},
        catalogue_index={"google-research-datasets/paws": resolved_row},
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
    assert "REFUSE/NONE ADMISSIBLE marker" in catalogue_result.detail
    licence_result = next(r for r in results if r.name == "licence_tier")
    assert licence_result.passed


def test_real_wildguardmix_fixture_is_refused_on_database_rights_only_grant_scope() -> None:
    # B4 verbatim: allenai/wildguardmix's real grant_scope is database_rights_only (ODC-By
    # ss2.4 licenses the compilation/database right, not the individual contents) --
    # verdict ATTRIBUTION requires only the `mit` tier, which `code` already carries, and
    # verification_status is VERIFIED, so checks 1 and 3 both pass; only check 4 catches
    # the grant_scope defect. Before the B4 fix this tool's FULL_CONTENT_GRANT_SCOPES
    # included database_rights_only and ADMITted this dataset while the factory refused
    # it -- a live split-brain between the two gates.
    provenance = json.loads((FIXTURES / "provenance-wildguardmix.json").read_text(encoding="utf-8"))
    results = mod.run_checklist(
        provenance,
        "code",
        {"other": 1_000_000_000},
        catalogue_index=_real_catalogue_index(),
    )
    assert not all(r.passed for r in results)
    catalogue_result = next(r for r in results if r.name == "catalogue_structural_refusals")
    assert not catalogue_result.passed
    assert "database_rights_only" in catalogue_result.detail
    licence_result = next(r for r in results if r.name == "licence_tier")
    verified_result = next(r for r in results if r.name == "verification_status")
    assert licence_result.passed
    assert verified_result.passed


def test_real_scidocs_fixture_passes_all_five_checks_including_policy_drift() -> None:
    # Round-3 review, finding 1, verbatim regression: BeIR/scidocs's real provenance.json
    # (regenerated by the factory 2026-09-04 with a real `admission` block) used to REFUSE
    # through this tool at check 5 alone -- ADAPTER_POLICY_CONSTANTS had zero key overlap
    # with the factory's real admission.constants (14 UPPER_SNAKE keys vs this tool's 6
    # lower_snake keys) and a differing policy_version ("2026-09-03.r3" vs
    # "2026-09-03-r3"), so `constants != ADAPTER_POLICY_CONSTANTS` failed on every real
    # dataset unconditionally. This fixture is the exact one the review's repro pulled
    # off gpu5080 -- proof the fix actually admits real factory output, not just a
    # synthetic dict built to match this tool's own pin.
    provenance = json.loads((FIXTURES / "provenance-scidocs.json").read_text(encoding="utf-8"))
    assert "admission" in provenance  # this fixture, unlike the B1-B4 ones, has a real block
    results = mod.run_checklist(
        provenance,
        "memory",
        {"other": 1_000_000_000},
        catalogue_index=_real_catalogue_index(),
    )
    failed = [(r.name, r.detail) for r in results if not r.passed]
    assert not failed, failed
    drift_result = next(r for r in results if r.name == "policy_constants_drift")
    assert "match" in drift_result.detail


def test_real_proofnet_fixture_passes_all_five_checks_including_policy_drift() -> None:
    # Same regression as scidocs above, second dataset from the review's repro
    # (hoskinson-center/proofnet, `reason` region -- a different faculty/tier pairing).
    provenance = json.loads((FIXTURES / "provenance-proofnet.json").read_text(encoding="utf-8"))
    assert "admission" in provenance
    results = mod.run_checklist(
        provenance,
        "reason",
        {"other": 1_000_000_000},
        catalogue_index=_real_catalogue_index(),
    )
    failed = [(r.name, r.detail) for r in results if not r.passed]
    assert not failed, failed


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


def test_cli_refuses_real_paws_fixture_pending_red_flag_resolution() -> None:
    # See test_real_paws_fixture_is_refused_pending_red_flag_resolution -- same defect,
    # exercised through the actual CLI entry point against the real default catalogue.
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
    assert proc.returncode == 1
    assert "REFUSE" in proc.stdout
    assert "unresolved provenance_red_flag" in proc.stdout


def test_cli_admits_real_paws_fixture_once_red_flag_resolved(tmp_path: Path) -> None:
    # End-to-end proof (real CLI, real provenance fixture) that the resolution mechanism
    # actually admits: a copy of the real bundled catalogue with ONLY paws's missing
    # provenance_red_flags_resolution entry added (the one field TASK A is expected to
    # add), pointed at via --catalogue. Nothing on disk under docs/design/ is touched.
    real_catalogue = json.loads(REAL_CATALOGUE.read_text(encoding="utf-8"))
    patched_entries = []
    for entry in real_catalogue["entries"]:
        if entry.get("repo_id") == "google-research-datasets/paws":
            (real_flag,) = entry["provenance_red_flags"]
            entry = dict(
                entry,
                provenance_red_flags_resolution=[{"flag": real_flag, "resolved": True}],
            )
        patched_entries.append(entry)
    patched_catalogue = dict(real_catalogue, entries=patched_entries)
    catalogue_path = tmp_path / "catalogue-patched.json"
    catalogue_path.write_text(json.dumps(patched_catalogue), encoding="utf-8")

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
            "--catalogue",
            str(catalogue_path),
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


# --------------------------------------------------------------------------------------
# resolve_existing_shares_from_corpus_root: the --corpus-root alternative to hand-typed
# --existing-shares JSON.
# --------------------------------------------------------------------------------------


def test_resolve_existing_shares_from_corpus_root_sums_by_provenance_group(
    tmp_path: Path,
) -> None:
    (tmp_path / "memory" / "gooaq").mkdir(parents=True)
    (tmp_path / "memory" / "gooaq" / "provenance.json").write_text(
        json.dumps({"provenance_group": "gooaq", "row_count": 3000}), encoding="utf-8"
    )
    (tmp_path / "memory" / "gooaq-2").mkdir(parents=True)
    (tmp_path / "memory" / "gooaq-2" / "provenance.json").write_text(
        json.dumps({"provenance_group": "gooaq", "row_count": 500}), encoding="utf-8"
    )
    (tmp_path / "memory" / "other").mkdir(parents=True)
    (tmp_path / "memory" / "other" / "provenance.json").write_text(
        json.dumps({"provenance_group": "other", "count": 7000}), encoding="utf-8"
    )
    shares = mod.resolve_existing_shares_from_corpus_root(tmp_path)
    assert shares == {"gooaq": 3500, "other": 7000}


def test_resolve_existing_shares_from_corpus_root_reads_group_from_catalogue_entry(
    tmp_path: Path,
) -> None:
    # Real factory shape: provenance_group only under the trimmed catalogue_entry, no
    # top-level provenance_group -- same fallback repo_id_of already relies on.
    ds_dir = tmp_path / "memory" / "paws"
    ds_dir.mkdir(parents=True)
    (ds_dir / "provenance.json").write_text(
        json.dumps({"catalogue_entry": {"provenance_group": "paws"}, "total_bytes": 400}),
        encoding="utf-8",
    )
    shares = mod.resolve_existing_shares_from_corpus_root(tmp_path)
    assert shares == {"paws": 400 // mod.BYTES_PER_TOKEN}


def test_resolve_existing_shares_from_corpus_root_skips_corrupt_provenance(
    tmp_path: Path,
) -> None:
    ds_dir = tmp_path / "memory" / "broken"
    ds_dir.mkdir(parents=True)
    (ds_dir / "provenance.json").write_text("{not json", encoding="utf-8")
    good_dir = tmp_path / "memory" / "good"
    good_dir.mkdir(parents=True)
    (good_dir / "provenance.json").write_text(
        json.dumps({"provenance_group": "good", "row_count": 10}), encoding="utf-8"
    )
    shares = mod.resolve_existing_shares_from_corpus_root(tmp_path)
    assert shares == {"good": 10}


def test_resolve_existing_shares_from_corpus_root_skips_missing_provenance_group(
    tmp_path: Path,
) -> None:
    ds_dir = tmp_path / "memory" / "no-group"
    ds_dir.mkdir(parents=True)
    (ds_dir / "provenance.json").write_text(json.dumps({"row_count": 10}), encoding="utf-8")
    assert mod.resolve_existing_shares_from_corpus_root(tmp_path) == {}


def test_resolve_existing_shares_from_corpus_root_empty_when_no_provenance_files(
    tmp_path: Path,
) -> None:
    assert mod.resolve_existing_shares_from_corpus_root(tmp_path) == {}


# --------------------------------------------------------------------------------------
# g25: role-aware concentration, active_train_set.train_files, no corpus-root
# double-count of the dataset under check. Mutation proofs: drop the role filter,
# drop the train_files preference, or drop the exclude= path and these fail.
# --------------------------------------------------------------------------------------


def test_landing_role_prefers_top_level_then_catalogue_entry() -> None:
    assert mod.landing_role({"role": "probe", "catalogue_entry": {"role": "train"}}) == "probe"
    assert mod.landing_role({"catalogue_entry": {"role": "held_seed"}}) == "held_seed"
    assert mod.landing_role({"row_count": 1}) is None


def test_missing_role_counts_as_train_for_legacy_records() -> None:
    assert mod.contributes_to_train_concentration(None) is True
    assert mod.contributes_to_train_concentration("train") is True
    for role in ("probe", "aux", "refuse", "held_seed", "train extra"):
        assert mod.contributes_to_train_concentration(role) is False


def test_resolve_concentration_count_excludes_probe_role() -> None:
    count, estimated, note = mod.resolve_concentration_count(
        {"role": "probe", "total_bytes": 10**12, "provenance_group": "tinyquickdraw"}
    )
    assert count == 0
    assert estimated is False
    assert "probe" in note


def test_resolve_concentration_count_prefers_active_train_set_over_bytes() -> None:
    count, estimated, note = mod.resolve_concentration_count(
        {
            "role": "train",
            "total_bytes": 10**12,
            "row_count": 999_999,
            "active_train_set": {"train_files": 1000, "train_bytes": 50},
        }
    )
    assert (count, estimated) == (1000, False)
    assert note == "active_train_set.train_files"


def test_resolve_concentration_count_notes_when_active_train_set_absent() -> None:
    count, estimated, note = mod.resolve_concentration_count(
        {"row_count": 42, "total_bytes": 999999}
    )
    assert (count, estimated) == (42, False)
    assert "active_train_set absent" in note


def test_b1_share_prints_largest_source_share_and_cap() -> None:
    result = mod.check_b1_share("narrativeqa", 1000, {"gooaq": 3000})
    assert result.passed
    assert "largest source='gooaq'" in result.detail
    assert "share=0.7500" in result.detail
    assert "cap=0.4" in result.detail


def test_role_exclusion_probe_that_would_dominate_by_bytes_does_not_trip_cap() -> None:
    """Mutation proof: a probe-only landing whose byte estimate would be ~100% of the
    mix must PASS B1. If concentration ignored role, this would FAIL the 0.40 cap."""
    probe = {
        "repo_id": "google/tinyquickdraw",
        "role": "probe",
        "provenance_group": "tinyquickdraw",
        "verdict": "PERMISSIVE_OK",
        "verification_status": "VERIFIED",
        "total_bytes": 28_171_583_550,
    }
    # Byte estimate of the probe (~7e9 tokens) vs 6000 existing train rows: share ~1.0.
    results = mod.run_checklist(probe, "visual", {"pxhere": 3000, "pcam": 3000})
    b1 = next(r for r in results if r.name == "b1_share")
    assert b1.passed, b1.detail
    assert "excluded from train concentration" in b1.detail
    assert "largest source=" in b1.detail
    assert "cap=0.4" in b1.detail
    # Guard the mutation: the byte fallback really would have tripped the cap.
    bytes_count, estimated = mod.resolve_added_count(probe)
    assert estimated is True
    over = mod.check_b1_share("tinyquickdraw", bytes_count, {"pxhere": 3000, "pcam": 3000})
    assert not over.passed


def test_role_exclusion_held_seed_and_aux_also_skipped(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "provenance.json").write_text(
        json.dumps({"role": "train", "provenance_group": "pxhere", "row_count": 3000}),
        encoding="utf-8",
    )
    (tmp_path / "probe").mkdir()
    (tmp_path / "probe" / "provenance.json").write_text(
        json.dumps({"role": "probe", "provenance_group": "tinyquickdraw", "total_bytes": 10**12}),
        encoding="utf-8",
    )
    (tmp_path / "aux").mkdir()
    (tmp_path / "aux" / "provenance.json").write_text(
        json.dumps({"role": "aux", "provenance_group": "usgs-landsat", "row_count": 80}),
        encoding="utf-8",
    )
    (tmp_path / "held").mkdir()
    (tmp_path / "held" / "provenance.json").write_text(
        json.dumps({"role": "held_seed", "provenance_group": "pd-stoic", "row_count": 9}),
        encoding="utf-8",
    )
    (tmp_path / "refuse").mkdir()
    (tmp_path / "refuse" / "provenance.json").write_text(
        json.dumps({"role": "refuse", "provenance_group": "jigsaw", "row_count": 100_000}),
        encoding="utf-8",
    )
    shares = mod.resolve_existing_shares_from_corpus_root(tmp_path)
    assert shares == {"pxhere": 3000}


def test_corpus_root_does_not_double_count_the_dataset_under_check(tmp_path: Path) -> None:
    """Mutation proof: candidate lives inside --corpus-root. Without exclude-by-resolved-
    path, the scan counts it once and check_b1_share adds it again, pushing 2000/6000
    (PASS, 0.333) to 4000/8000 (FAIL, 0.50)."""
    corpus = tmp_path / "corpus"
    cand_dir = corpus / "visual" / "pcam"
    other_dir = corpus / "visual" / "pxhere"
    cand_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    candidate = {
        "repo_id": "basveeling/pcam",
        "role": "train",
        "provenance_group": "pcam",
        "verdict": "PERMISSIVE_OK",
        "verification_status": "VERIFIED",
        "row_count": 2000,
    }
    (cand_dir / "provenance.json").write_text(json.dumps(candidate), encoding="utf-8")
    (other_dir / "provenance.json").write_text(
        json.dumps({"role": "train", "provenance_group": "pxhere", "row_count": 4000}),
        encoding="utf-8",
    )
    cand_path = cand_dir / "provenance.json"
    shares = mod.resolve_existing_shares_from_corpus_root(corpus, exclude=cand_path)
    assert shares == {"pxhere": 4000}

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--provenance",
            str(cand_path),
            "--region",
            "visual",
            "--corpus-root",
            str(corpus),
            "--no-catalogue-check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ADMIT" in proc.stdout
    assert "2000/6000" in proc.stdout
    assert "Largest source:" in proc.stdout
    assert "cap=0.4" in proc.stdout
    # Guard the mutation: counting the candidate twice would refuse.
    doubled = mod.check_b1_share("pcam", 2000, {"pcam": 2000, "pxhere": 4000})
    assert not doubled.passed


def test_active_train_set_by_count_path_does_not_use_bytes() -> None:
    """Mutation proof: g24 `train_files` is 1000; `total_bytes` would estimate a count
    that alone exceeds the cap. Using bytes would FAIL; using train_files PASSes."""
    provenance = {
        "repo_id": "basveeling/pcam",
        "role": "train",
        "provenance_group": "pcam",
        "verdict": "PERMISSIVE_OK",
        "verification_status": "VERIFIED",
        "total_bytes": 10**12,
        "active_train_set": {
            "stamp": "20260905T045655Z",
            "train_tree_or_zip": "processed/20260905T045655Z/train.zip",
            "train_files": 1000,
            "train_bytes": 3723965223,
            "probe_files": 0,
            "probe_bytes": 0,
            "superseded_stamps": ["20260905T035325Z"],
        },
    }
    results = mod.run_checklist(provenance, "visual", {"pxhere": 3000})
    b1 = next(r for r in results if r.name == "b1_share")
    assert b1.passed, b1.detail
    assert "1000/4000" in b1.detail
    assert "active_train_set.train_files" in b1.detail
    assert "ESTIMATED" not in b1.detail
    bytes_count, estimated = mod.resolve_added_count(provenance)
    assert estimated is True
    over = mod.check_b1_share("pcam", bytes_count, {"pxhere": 3000})
    assert not over.passed


def test_cli_prints_largest_source_share_and_cap(tmp_path: Path) -> None:
    prov = tmp_path / "provenance.json"
    prov.write_text(
        json.dumps(
            {
                **CLEAN_PROVENANCE,
                "active_train_set": {"train_files": 1000},
            }
        ),
        encoding="utf-8",
    )
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
            "--no-catalogue-check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Largest source: gooaq share=0.7500 cap=0.4" in proc.stdout
    assert "active_train_set.train_files" in proc.stdout


def test_cli_notes_active_train_set_absent_on_byte_fallback(tmp_path: Path) -> None:
    prov = tmp_path / "provenance.json"
    prov.write_text(
        json.dumps(
            {
                "repo_id": "example/bytes-only",
                "provenance_group": "small",
                "verdict": "PERMISSIVE_OK",
                "verification_status": "VERIFIED",
                "total_bytes": 4000,
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--provenance",
            str(prov),
            "--region",
            "visual",
            "--existing-shares",
            json.dumps({"other": 10_000}),
            "--no-catalogue-check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "active_train_set absent" in proc.stdout
    assert "ESTIMATED" in proc.stdout


# --------------------------------------------------------------------------------------
# check_policy_constants_drift (check 5): forward-compatible against provenance.json
# without an `admission` block (every real one today), and refuses a real, present drift.
# --------------------------------------------------------------------------------------


def test_policy_constants_drift_passes_when_no_admission_block() -> None:
    result = mod.check_policy_constants_drift({"verdict": "PERMISSIVE_OK"})
    assert result.passed
    assert "not yet applicable" in result.detail


def test_policy_constants_drift_passes_when_constants_match_exactly() -> None:
    provenance = {
        "admission": {
            "policy_version": mod.POLICY_VERSION,
            "constants": dict(mod.ADAPTER_POLICY_CONSTANTS),
        }
    }
    result = mod.check_policy_constants_drift(provenance)
    assert result.passed, result.detail


def test_policy_constants_drift_fails_when_constants_disagree() -> None:
    # MUST fail: a single differing field inside `constants` (here,
    # FULL_CONTENT_GRANT_SCOPES widened back to include database_rights_only -- exactly
    # the B4 defect this whole mechanism exists to catch a recurrence of) is enough on
    # its own, independent of whether `policy_version` still agrees.
    drifted = dict(
        mod.ADAPTER_POLICY_CONSTANTS,
        FULL_CONTENT_GRANT_SCOPES=["database_rights_only", "whole_corpus"],
    )
    provenance = {"admission": {"policy_version": mod.POLICY_VERSION, "constants": drifted}}
    result = mod.check_policy_constants_drift(provenance)
    assert not result.passed
    assert "admission.constants=" in result.detail
    assert "disagrees" in result.detail


def test_policy_constants_drift_fails_when_top_level_policy_version_disagrees() -> None:
    # MUST fail independently of the constants dict: a provenance record whose top-level
    # `admission.policy_version` disagrees with this tool's pin (a policy bump that has
    # not yet touched the constants body -- e.g. a gate's DOCUMENTATION changed without
    # its VALUES changing) is exactly the case `policy_version` is checked separately, not
    # folded into the constants dict, to catch precisely.
    provenance = {
        "admission": {
            "policy_version": "9999-99-99-x",
            "constants": dict(mod.ADAPTER_POLICY_CONSTANTS),
        }
    }
    result = mod.check_policy_constants_drift(provenance)
    assert not result.passed
    assert "admission.policy_version=" in result.detail
    assert "disagrees" in result.detail


def test_policy_constants_drift_fails_when_admission_block_missing_constants_key() -> None:
    provenance = {"admission": {"policy_version": mod.POLICY_VERSION}}
    result = mod.check_policy_constants_drift(provenance)
    assert not result.passed


def test_policy_constants_drift_fails_when_admission_block_is_not_a_dict() -> None:
    provenance = {"admission": "not-a-dict"}
    result = mod.check_policy_constants_drift(provenance)
    assert not result.passed


# --------------------------------------------------------------------------------------
# ADAPTER_POLICY_CONSTANTS pinned against tests/fixtures/dataset-factory-policy.json --
# fails loudly if either side changes without the other. The fixture is GENERATED BY THE
# FACTORY, not transcribed: `dataset-factory policy-constants` emits the admission policy
# in exactly the shape `provenance.json` carries it under `admission.constants`, and that
# command exists for this purpose. Regenerate it, never hand-edit it.
#
# Regenerated with (from the dataset-factory repo root, its own venv):
#   .venv/bin/dataset-factory policy-constants --out \
#     ../csd-worktress/CogSynDelta-wt-dataset-ingest/tests/fixtures/dataset-factory-policy.json
#
# Why generated rather than transcribed: a hand-transcribed pin cannot be trusted to
# match, and twice did not. Round 2 left this tool pinning six lower_snake keys against
# the factory's fourteen UPPER_SNAKE ones -- zero key overlap, a differently-punctuated
# version string, and a check 5 that therefore refused every landed dataset. Round 3's
# first pass fixed the vocabulary but kept `ADMISSION_POLICY_VERSION` outside the pinned
# dict while the factory landed it inside, so the whole-dict comparison still failed by
# exactly one key. Generating the fixture removes the transcription step that produced
# both.
# --------------------------------------------------------------------------------------

POLICY_FIXTURE = ROOT / "tests" / "fixtures" / "dataset-factory-policy.json"


def test_adapter_policy_constants_match_pinned_fixture() -> None:
    fixture = json.loads(POLICY_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["policy_version"] == mod.POLICY_VERSION
    assert fixture["constants"] == mod.ADAPTER_POLICY_CONSTANTS


def test_pinned_fixture_matches_live_factory_output_when_available() -> None:
    """Re-derive from a LIVE import of the factory's real `admission` module (not the
    frozen JSON snapshot) whenever the sibling `dataset-factory` checkout is present next
    to this repo -- the normal dev layout, both under `.../tzervas/`. This is the test
    that actually prevents the pin from going stale silently: the pinned-fixture test
    above only proves internal self-consistency (this tool agrees with a JSON file someone
    could still forget to regenerate); this one proves the pin agrees with the real thing,
    today. Skips (does not fail) when the sibling repo, or its `src` layout, is absent --
    a checkout of this repo alone, or CI with no dataset-factory workspace.

    FULL dict equality, deliberately, not a subset. The subset form tolerated the factory
    ADDING a constant this tool does not pin -- and that is exactly how
    `ADMISSION_POLICY_VERSION` slipped through: the factory moved it into
    `admission_constants()`, this tool kept pinning the other fourteen keys, the subset
    check stayed green, and `check_policy_constants_drift`'s whole-dict comparison against
    a real record failed by one key on every landed dataset. A constant this tool has not
    been taught to pin is not a harmless addition; it is a policy the two halves no longer
    share.
    """
    factory_src = ROOT.parent.parent / "dataset-factory" / "src"
    admission_module_path = factory_src / "dataset_factory" / "admission.py"
    if not admission_module_path.is_file():
        pytest.skip(f"dataset-factory sibling repo not found at {admission_module_path}")
    sys.path.insert(0, str(factory_src))
    try:
        # Fresh import each run (not reused across test sessions) so this reflects
        # whatever is on disk right now, WIP or committed.
        sys.modules.pop("dataset_factory.admission", None)
        sys.modules.pop("dataset_factory.catalogue", None)
        sys.modules.pop("dataset_factory", None)
        factory_admission = importlib.import_module("dataset_factory.admission")
    finally:
        sys.path.remove(str(factory_src))
    live_policy_version = factory_admission.ADMISSION_POLICY_VERSION
    live_constants = factory_admission.admission_constants()
    assert live_policy_version == mod.POLICY_VERSION, (
        f"factory ADMISSION_POLICY_VERSION={live_policy_version!r} has moved past this "
        f"tool's pinned POLICY_VERSION={mod.POLICY_VERSION!r} -- re-run the regeneration "
        "command in the comment above and re-verify ADAPTER_POLICY_CONSTANTS by hand"
    )
    missing = sorted(set(mod.ADAPTER_POLICY_CONSTANTS) - set(live_constants))
    added = sorted(set(live_constants) - set(mod.ADAPTER_POLICY_CONSTANTS))
    mismatched = {
        key: (mod.ADAPTER_POLICY_CONSTANTS[key], live_constants[key])
        for key in set(mod.ADAPTER_POLICY_CONSTANTS) & set(live_constants)
        if live_constants[key] != mod.ADAPTER_POLICY_CONSTANTS[key]
    }
    assert live_constants == mod.ADAPTER_POLICY_CONSTANTS, (
        "factory admission_constants() and this tool's pinned ADAPTER_POLICY_CONSTANTS "
        f"are not the same policy -- the factory has dropped {missing!r}, added {added!r}, "
        f"and disagrees on {sorted(mismatched)!r} (pinned, live)={mismatched!r}. "
        "check_policy_constants_drift compares these by whole-dict equality against every "
        "real provenance.json, so any of the three refuses every landed dataset: re-run "
        "`dataset-factory policy-constants --out <fixture>` and re-verify "
        "ADAPTER_POLICY_CONSTANTS against it."
    )


def test_check_5_passes_on_a_record_the_live_factory_writes() -> None:
    """The property the round-3 review actually measured, as a test.

    Every test above feeds `check_policy_constants_drift` either a synthetic block built
    FROM `ADAPTER_POLICY_CONSTANTS` (which proves only that the tool agrees with itself)
    or a provenance fixture captured at some past moment. Neither could catch the defect
    the review found: a pin that disagrees with what the factory writes TODAY, so that
    check 5 refuses every landed dataset with a constant, contentless refusal.

    This builds a provenance record with the factory's own `build_provenance_record` --
    the exact function that writes every `provenance.json` in the corpus -- and requires
    check 5 to pass on it. It skips when the sibling `dataset-factory` checkout is absent,
    like the fixture cross-check above.
    """
    factory_src = ROOT.parent.parent / "dataset-factory" / "src"
    if not (factory_src / "dataset_factory" / "provenance.py").is_file():
        pytest.skip(f"dataset-factory sibling repo not found at {factory_src}")
    sys.path.insert(0, str(factory_src))
    try:
        for name in list(sys.modules):
            if name == "dataset_factory" or name.startswith("dataset_factory."):
                del sys.modules[name]
        factory_provenance = importlib.import_module("dataset_factory.provenance")
        factory_catalogue = importlib.import_module("dataset_factory.catalogue")
    finally:
        sys.path.remove(str(factory_src))

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()
        (data_dir / "shard.parquet").write_bytes(b"content")
        record = factory_provenance.build_provenance_record(
            factory_catalogue.CatalogueEntry(
                repo_id="BeIR/scidocs",
                faculty="memory",
                provenance_group="PG-BEIR",
                licence_mirror_tag="cc-by-sa-4.0",
                licence_upstream_verbatim="CC BY 4.0",
                licence_upstream_source="https://example.com/LICENSE",
                licence_fetch_date="2026-09-03",
                grant_scope="whole_corpus",
                verdict="PERMISSIVE_OK",
                verification_status="VERIFIED",
                why="live cross-check",
            ),
            upstream_url="https://example.com/LICENSE",
            mirror_url="https://huggingface.co/datasets/BeIR/scidocs",
            resolved_revision="abc",
            dataset_dir=data_dir,
        )

    result = mod.check_policy_constants_drift(record)
    assert result.passed, result.detail


def test_check_5_reports_a_record_that_disagrees_with_itself() -> None:
    """Mutation proof for the internal-consistency comparison: a record whose two copies
    of the policy version disagree is a fault on the FACTORY's side, and is named as
    such rather than folded into the whole-dict mismatch."""
    record = {
        "admission": {
            "policy_version": mod.POLICY_VERSION,
            "constants": {
                **mod.ADAPTER_POLICY_CONSTANTS,
                "ADMISSION_POLICY_VERSION": "2026-12-01.r9",
            },
        }
    }

    result = mod.check_policy_constants_drift(record)

    assert not result.passed
    assert "disagrees with ITSELF" in result.detail


def test_full_content_grant_scopes_excludes_database_rights_only() -> None:
    # B4, pinned directly: this set must NOT include database_rights_only (that was the
    # actual defect -- the factory's dataset_factory.admission.FULL_CONTENT_GRANT_SCOPES
    # has never included it).
    assert "database_rights_only" not in mod.FULL_CONTENT_GRANT_SCOPES
    assert frozenset({"whole_corpus", "whole_corpus (heterogeneous per file)"}) == (
        mod.FULL_CONTENT_GRANT_SCOPES
    )
