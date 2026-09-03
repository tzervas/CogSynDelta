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

CLEAN_PROVENANCE: dict[str, Any] = {
    "repo_id": "deepmind/narrativeqa",
    "faculty": "memory",
    "provenance_group": "narrativeqa",
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
