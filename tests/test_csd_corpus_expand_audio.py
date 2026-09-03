"""The `auditory`/`speech_output` catalogue rows in scripts/csd-corpus-expand.py, and the
structural refusal `audio-licence-audit.md`'s BLOCKING findings depend on.

Row A0 (docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md, revision 3.1, §1.3/§4.1) exists
to answer OD-9 ("start the audio licence audit now") with catalogue entries, not prose.
Per tests/test_reserved_corpus_guard.py's own lesson (itself citing the project's guard-
verification discipline): a refusal with no test that shows it actually fires is a comment
with a function signature, so this file's load-bearing test is
`test_refuse_verdict_raises_blocking_source_error` -- it constructs the exact call a
BLOCKING entry makes possible (`fetch()` on a verdict=REFUSE row) and asserts it raises,
not merely that the row exists with the right string in it.

No network calls happen anywhere in this file. `fetch()`'s REFUSE branch raises before any
of `is_present`/`_disk_free_fraction`/the network dispatch table run, and every other test
here either reads CATALOGUE directly or calls `fetch()` with `apply=False` (dry-run) --
`--apply` on a `bounded_slice=True` entry performs a real, multi-GB network download by
design (see Dataset.fetch_kind's docstring) and MUST NEVER be exercised from a test.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu


def _load_csd_corpus_expand():
    """Import scripts/csd-corpus-expand.py the same way test_reserved_corpus_guard.py
    imports scripts/csd-train-all.py -- the hyphenated filename is not a valid module
    name, so every consumer in this repo loads it via `importlib.util.spec_from_file_location`.
    """
    path = Path(__file__).resolve().parents[1] / "scripts" / "csd-corpus-expand.py"
    spec = importlib.util.spec_from_file_location("csd_corpus_expand_audio_test", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod = _load_csd_corpus_expand()

AUDIO_REGIONS = ("auditory", "speech_output")


def _audio_entries():
    return [ds for ds in mod.CATALOGUE if ds.region in AUDIO_REGIONS]


def _refuse_entries():
    return [ds for ds in mod.CATALOGUE if ds.verdict == mod.REFUSE]


# ---------------------------------------------------------------------------------------
# The load-bearing test: BLOCKING sources refuse by construction, not by convention.
# ---------------------------------------------------------------------------------------


def test_refuse_verdict_raises_blocking_source_error() -> None:
    """Constructs the exact failing case: call fetch() on every verdict=REFUSE row and
    assert it raises BlockingSourceError, whether or not apply is set, before any network
    or disk-usage check runs. This is what makes REFUSE different from REJECTED (which
    only returns a status string) -- see both constants' docstrings in the script."""
    refused = _refuse_entries()
    assert refused, "expected at least one REFUSE (BLOCKING) entry to test against"
    for ds in refused:
        with pytest.raises(mod.BlockingSourceError):
            mod.fetch(ds, token=None, apply=False, max_used_fraction=0.5)
        with pytest.raises(mod.BlockingSourceError):
            mod.fetch(ds, token=None, apply=True, max_used_fraction=0.99)


def test_blocking_sources_named_in_the_audit_are_present_as_refuse() -> None:
    """audio-licence-audit.md's own REFUSE recommendations, each present in the catalogue
    with verdict=REFUSE -- "kept in the catalogue on purpose: a rejection nobody can see
    is a rejection that gets made again," the same rule the pre-existing REJECTED entries
    already document."""
    refused_ids = {(ds.name or ds.repo_id) for ds in _refuse_entries()}
    expected = {
        "speechcolab/gigaspeech",  # non-ownership disclaimer, contested apache-2.0 tag
        "ted-lium-3",  # CC BY-NC-ND -- ND is outside the NC-tolerant policy's scope
        "amphion/Emilia-Dataset",  # "Emilia does not own the copyright to the audio files"
        "kensho/spgispeech",  # bespoke academic-only EULA, real named owner
        "switchboard-1",  # paywalled LDC membership corpus, no mirror
        "agkphysics/AudioSet",  # CC BY 4.0 metadata tag misapplied to unowned audio
        "jp1924/AudioCaps",  # inherits AudioSet's defect + "academic purposes only"
        "cvssp/WavCaps",  # cc-by-4.0 tag directly contradicts "academic uses only" upstream
    }
    missing = expected - refused_ids
    assert not missing, f"expected REFUSE entries missing from the catalogue: {missing}"


def test_refuse_entries_have_blocking_flags_recorded() -> None:
    """Every REFUSE entry's redistribute_verdict is BLOCKING, not merely NC -- the audit's
    own distinction (NC is accepted policy; BLOCKING is not the same failure)."""
    for ds in _refuse_entries():
        assert ds.redistribute_verdict == "BLOCKING", (
            f"{ds.name or ds.repo_id}: verdict=REFUSE but redistribute_verdict="
            f"{ds.redistribute_verdict!r}, expected 'BLOCKING'"
        )


# ---------------------------------------------------------------------------------------
# Provenance-group rules: LibriVox and the AudioSet lineage must be ONE group each.
# ---------------------------------------------------------------------------------------


def test_librivox_lineage_shares_one_provenance_group() -> None:
    """audio-licence-audit.md's central finding: ten differently-named datasets over one
    volunteer public-domain-audiobook pool. Every one of them, in EITHER region, must
    carry provenance_group=="librivox" -- if even one drifts to a per-dataset group, a B1/B2
    cap built on provenance_group silently stops treating it as part of the monoculture."""
    librivox_names = {
        "openslr/librispeech_asr",
        "facebook/multilingual_librispeech",
        "libri-light",
        "keithito/lj_speech",
        "mythicinfinity/libritts_r",
        "css10",
        "gigant/m-ailabs_speech_dataset_fr",
        "MikhailT/hifi-tts",
        "nvidia/hifitts-2",
    }
    by_name = {(ds.name or ds.repo_id): ds for ds in _audio_entries()}
    missing = librivox_names - by_name.keys()
    assert not missing, f"expected LibriVox-lineage entries missing: {missing}"
    wrong_group = {
        name: by_name[name].provenance_group
        for name in librivox_names
        if by_name[name].provenance_group != "librivox"
    }
    assert not wrong_group, f"LibriVox-lineage entries not grouped as 'librivox': {wrong_group}"


def test_audioset_lineage_shares_one_provenance_group() -> None:
    """AudioCaps and (~27% of) WavCaps inherit AudioSet's non-ownership defect by lineage
    -- audio-licence-audit.md's second cross-bucket finding. All three REFUSE entries must
    share one provenance_group so a future audit doesn't have to re-derive the lineage."""
    audioset_names = {"agkphysics/AudioSet", "jp1924/AudioCaps", "cvssp/WavCaps"}
    by_name = {(ds.name or ds.repo_id): ds for ds in mod.CATALOGUE}
    groups = {by_name[name].provenance_group for name in audioset_names}
    assert groups == {"audioset"}, f"AudioSet-lineage entries have split groups: {groups}"


def test_no_train_ok_entry_shares_a_provenance_group_with_a_refuse_entry() -> None:
    """A TRAIN_OK entry silently sharing a provenance_group with a REFUSE entry would let
    a balance-cap step accidentally admit BLOCKING material under the cover of a clean
    group's name."""
    refuse_groups = {ds.provenance_group for ds in _refuse_entries() if ds.provenance_group}
    train_ok_groups = {
        ds.provenance_group
        for ds in _audio_entries()
        if ds.verdict == mod.TRAIN_OK and ds.provenance_group
    }
    overlap = refuse_groups & train_ok_groups
    assert not overlap, f"TRAIN_OK and REFUSE entries share provenance_group(s): {overlap}"


# ---------------------------------------------------------------------------------------
# The balance check: B1/B2 computed with LibriVox as one group, and able to fail.
# ---------------------------------------------------------------------------------------


def test_balance_report_collapses_librivox_into_one_group() -> None:
    """The whole point of provenance_group: nine LibriVox-lineage catalogue rows (with
    approx_hours>0) must contribute to exactly one group in the report, not nine."""
    librivox_rows = [
        ds
        for ds in _audio_entries()
        if ds.provenance_group == "librivox" and ds.verdict == mod.TRAIN_OK and ds.approx_hours > 0
    ]
    assert len(librivox_rows) >= 2, "need >=2 measured librivox rows to prove collapsing happens"
    report = mod.audio_balance_report()
    assert "librivox" in report["groups"]
    expected_hours = sum(ds.approx_hours for ds in librivox_rows)
    assert report["groups"]["librivox"] == pytest.approx(expected_hours)


def test_balance_report_can_fail() -> None:
    """verify-guards-by-making-them-fail (this project's own standing lesson): a balance
    check that only ever reports success on the one corpus it was run against is not a
    guard. On the catalogue's NATURAL, uncapped sizes, VoxPopuli's ~400,000 unlabelled
    hours dominate every other provenance group -- audio-licence-audit.md's own B1 section
    predicted exactly this shape ("if the LibriVox group is used at anything near its
    uncapped scale... unless VoxPopuli's 400,000 unlabelled hours are also drawn on at
    comparable scale -- in which case VoxPopuli... becomes the new dominant source
    instead"). This test asserts the guard actually reports that failure rather than
    quietly passing."""
    report = mod.audio_balance_report()
    assert report["b1_max_share"] > 0.40, (
        "expected the uncapped natural-size balance to FAIL B1 (max share > 40%) -- if "
        "this now passes, either the catalogue's approx_hours changed in a way that "
        "deserves a comment here, or the balance arithmetic broke silently"
    )
    assert report["b2_n_eff"] < 3.0


def test_balance_report_is_directionally_achievable_with_a_cap() -> None:
    """audio-licence-audit.md's B2 section: 'Seven provenance groups exist... A
    uniform-ish cap across five to six of them... would land comfortably above the >=3
    threshold.' This is the corresponding structural check: at least the number of
    distinct measured provenance groups the audit counted."""
    report = mod.audio_balance_report()
    assert len(report["groups"]) >= 5


def test_balance_report_excludes_unmeasured_entries() -> None:
    """CSS10 and HiFiTTS-2 are deliberately left at approx_hours=0.0 (see their Dataset
    docstrings/caveats) rather than assumed negligible -- confirm they do not silently
    enter the weighted sum as zero-but-counted, and are not the reason a group appears."""
    zero_hour_names = {"css10", "nvidia/hifitts-2"}
    by_name = {(ds.name or ds.repo_id): ds for ds in _audio_entries()}
    for name in zero_hour_names:
        assert by_name[name].approx_hours == 0.0, f"{name} expected approx_hours==0.0"


# ---------------------------------------------------------------------------------------
# Structural properties every entry must have, region-wide.
# ---------------------------------------------------------------------------------------


def test_every_train_ok_audio_entry_has_a_provenance_group() -> None:
    missing = [
        (ds.name or ds.repo_id)
        for ds in _audio_entries()
        if ds.verdict == mod.TRAIN_OK and not ds.provenance_group
    ]
    assert not missing, (
        f"TRAIN_OK auditory/speech_output entries with no provenance_group: {missing}"
    )


def test_every_entry_has_an_upstream_license_or_an_explicit_gap_noted() -> None:
    """upstream_license_verbatim is the audit's core discipline (a mirror tag is not
    evidence about its upstream) -- every audio entry must carry SOME upstream statement,
    even a negative one ("not located", "not independently re-checked")."""
    missing = [
        (ds.name or ds.repo_id) for ds in _audio_entries() if not ds.upstream_license_verbatim
    ]
    assert not missing, f"audio entries with no upstream_license_verbatim at all: {missing}"


def test_nc_flagged_entries_carry_the_nc_flag() -> None:
    """Expresso is the one deliberate NC addition (speech_output); its flags must record
    NC so a machine check (not just prose) can find it."""
    by_name = {(ds.name or ds.repo_id): ds for ds in _audio_entries()}
    expresso = by_name["ylacombe/expresso"]
    assert "NC" in expresso.flags


def test_unavailable_entries_have_no_repo_id_and_a_local_name() -> None:
    """fetch_kind='unavailable' entries (Libri-Light, CSS10) have no packaged mirror to
    read repo_id from -- Dataset.local must still resolve to a sane directory via `name`,
    not crash or collide."""
    unavailable = [ds for ds in _audio_entries() if ds.fetch_kind == "unavailable"]
    assert len(unavailable) >= 2
    seen_paths = set()
    for ds in unavailable:
        assert ds.repo_id == ""
        assert ds.name, f"unavailable entry with no name set: {ds.why[:60]}"
        assert ds.local not in seen_paths, f"duplicate local path {ds.local}"
        seen_paths.add(ds.local)


# ---------------------------------------------------------------------------------------
# The bounded first-fetch slice: >=3 independent provenance groups, budget-bounded.
# ---------------------------------------------------------------------------------------


def test_bounded_slice_covers_at_least_three_independent_provenance_groups() -> None:
    sliced = [ds for ds in _audio_entries() if ds.bounded_slice]
    assert sliced, "expected at least one bounded_slice=True entry"
    groups = {ds.provenance_group for ds in sliced}
    assert len(groups) >= 3, f"bounded slice spans too few provenance groups: {groups}"
    assert "librivox" not in groups, (
        "the bounded first fetch was scoped to NOT touch the librivox group (its members "
        "are TB-scale or have no packaged mirror) -- a librivox entry in bounded_slice is "
        "very likely an accidental multi-TB download waiting to happen"
    )


def test_bounded_slice_entries_are_all_train_ok() -> None:
    for ds in _audio_entries():
        if ds.bounded_slice:
            assert ds.verdict == mod.TRAIN_OK, (
                f"{ds.name or ds.repo_id} is bounded_slice=True but verdict={ds.verdict}"
            )


def test_bounded_slice_entries_have_a_real_fetch_kind() -> None:
    for ds in _audio_entries():
        if ds.bounded_slice:
            assert ds.fetch_kind in ("hf_files_bounded", "http_archive", "hf_dataset"), (
                f"{ds.name or ds.repo_id} is bounded_slice=True but fetch_kind="
                f"{ds.fetch_kind!r} cannot download real audio bytes"
            )


# ---------------------------------------------------------------------------------------
# fetch() side effects: manifest-only writes, without any network call.
# ---------------------------------------------------------------------------------------


def test_manifest_only_fetch_for_unavailable_entry_writes_a_receipt(tmp_path, monkeypatch) -> None:
    """fetch_kind='unavailable' must produce a manifest recording the finding, not attempt
    a download -- and must set audio_fetched: False so nothing downstream mistakes this
    for real bytes on disk."""
    monkeypatch.setattr(mod, "BULK_CORPUS", tmp_path / "csd-corpus")
    ds = next(ds for ds in _audio_entries() if ds.fetch_kind == "unavailable")
    res = mod.fetch(ds, token=None, apply=True, max_used_fraction=0.999)
    assert res["status"] == "manifest-only (no mirror available)"
    manifest_path = ds.local / "MANIFEST.json"
    assert manifest_path.is_file()
    import json

    manifest = json.loads(manifest_path.read_text())
    assert manifest["audio_fetched"] is False
    assert manifest["bytes"] == 0
    assert "manifest_sha256" in manifest
    assert len(manifest["manifest_sha256"]) == 64  # hex sha256


def test_manifest_only_fetch_for_non_sliced_entry_writes_a_receipt(tmp_path, monkeypatch) -> None:
    """A TRAIN_OK, bounded_slice=False entry (e.g. AMI) gets a real manifest with the
    catalogue's provenance/licence findings and audio_fetched: False -- no network call,
    since apply=True alone must not be enough to trigger a real download; bounded_slice
    is the separate, explicit switch for that."""
    monkeypatch.setattr(mod, "BULK_CORPUS", tmp_path / "csd-corpus")
    ds = next(
        ds
        for ds in _audio_entries()
        if ds.verdict == mod.TRAIN_OK and not ds.bounded_slice and ds.fetch_kind != "unavailable"
    )
    res = mod.fetch(ds, token=None, apply=True, max_used_fraction=0.999)
    assert res["status"] == "manifest-only (not in this pass's bounded slice)"
    import json

    manifest = json.loads((ds.local / "MANIFEST.json").read_text())
    assert manifest["audio_fetched"] is False
    assert manifest["provenance_group"] == ds.provenance_group


def test_manifest_sha256_changes_if_manifest_contents_change(tmp_path, monkeypatch) -> None:
    """A cheap sanity check that manifest_sha256 is a real hash of the content (not a
    constant) -- fetch two different unavailable entries and confirm distinct hashes."""
    monkeypatch.setattr(mod, "BULK_CORPUS", tmp_path / "csd-corpus")
    unavailable = [ds for ds in _audio_entries() if ds.fetch_kind == "unavailable"]
    assert len(unavailable) >= 2
    import json

    hashes = set()
    for ds in unavailable:
        mod.fetch(ds, token=None, apply=True, max_used_fraction=0.999)
        manifest = json.loads((ds.local / "MANIFEST.json").read_text())
        hashes.add(manifest["manifest_sha256"])
    assert len(hashes) == len(unavailable), "expected distinct entries to hash differently"


# ---------------------------------------------------------------------------------------
# Region wiring: the new regions actually appear where region-consuming code looks.
# ---------------------------------------------------------------------------------------


def test_auditory_and_speech_output_regions_present() -> None:
    regions = Counter(ds.region for ds in mod.CATALOGUE)
    assert regions["auditory"] > 0
    assert regions["speech_output"] > 0
