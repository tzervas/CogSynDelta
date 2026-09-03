#!/usr/bin/env python3
"""Print the CORPUS-CONTRACT.md admission checklist for one factory-fetched dataset.

WHAT THIS DOES AND DOES NOT DO
This is a read-only gate, not a fetcher and not a corpus mutator. It reads one dataset
factory `provenance.json` (see program/datasets/README.md `S2`) plus a target CSD region
name, and prints four checks: licence tier vs. the region's already-declared tier,
provenance-group share against CORPUS-CONTRACT.md's B1 (<= 0.40 post-admission),
`verification_status == VERIFIED`, and (check 4, see below) a second, independent read of
the full catalogue row's own structural refusal signals. It does not touch
CORPUS-CONTRACT.md, csd-regions.json,
or any region's fetch list -- admitting a dataset for real is a human/agent action taken
after reading this tool's output, same discipline as csd-corpus-expand.py's structural
REFUSE gate (no override flag) but advisory rather than fetch-blocking, because this tool
runs after the fetch, not during it.

WHY THE LICENCE-TIER TABLE IS DUPLICATED HERE, NOT IMPORTED
scripts/csd-publish-checkpoint.py's LICENCE_TIER is the source of truth for what a region
is licensed under today. Importing it would couple this admission tool to the publish
script's module surface (and its torch/HF-hub-adjacent import chain) for the sake of one
small dict. The table is small, dated, and rarely changes; it is transcribed here with a
test (test_licence_tier_matches_publish_checkpoint) that fails loudly the day the two
tables disagree, which is a cheaper and more honest coupling than an import that pulls in
unrelated machinery for four rows.

CHECK 4 IS A SECOND, INDEPENDENT LOOK -- NOT A RE-READ OF THE SAME FIELDS
Checks 1-3 read only what the factory's `provenance.json` actually emits per dataset
(verdict, verification_status, provenance_group, total_bytes/file_count/file_manifest,
licence_* and a trimmed `catalogue_entry`). That trimmed entry carries `grant_scope` but
NOT `redistribute{nc,sa,nd,attribution}`, `provenance_red_flags`, or
`enrichment_licence_result` -- confirmed by reading every real provenance.json under
/mnt/bulk/csd-corpus/factory-2026-09-03/ (2026-09-03 ground pass), all of which trim
`catalogue_entry` to the same six keys. Those three fields only exist in the full
catalogue row (`docs/design/datasets/catalogue-2026-09-03.json`, keyed by `repo_id`), which
is also where the factory's OWN gate (`admission.check_admission`) failed to read them
(ground-pass findings B1-B3: narrativeqa and BeIR/cqadupstack both landed on /bulk despite
carrying a structural refusal in exactly these fields). Check 4 therefore loads that
catalogue independently and applies its own refusal logic against it, so a defect in the
factory's gate does not silently pass through this one too -- two gates reading two
different sources of truth, not one gate re-reading the same file.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Transcribed from scripts/csd-publish-checkpoint.py's LICENCE_TIER, restricted to the
# ordering that matters here: how strict a tier is, not which regions exist. Kept in sync
# by test_licence_tier_matches_publish_checkpoint.
TIER_ORDER: tuple[str, ...] = ("mit", "cc-by-sa-4.0", "cc-by-nc-sa-4.0")

REGION_TIER: dict[str, str] = {
    "code": "mit",
    "classify_banking77": "mit",
    "classify_go_emotions": "mit",
    "reason": "mit",
    "vl_latent": "mit",
    "compress": "cc-by-sa-4.0",
    "retrieve": "cc-by-nc-sa-4.0",
    "memory": "cc-by-nc-sa-4.0",
    "composed": "cc-by-nc-sa-4.0",
    "cogsyndelta": "cc-by-nc-sa-4.0",
}

# Catalogue verdict -> the licence tier that verdict requires of the region it joins.
# REFUSE/BLOCKING/UNVERIFIED have no tier: they are inadmissible outright, independent of
# any region's current tier, so they map to None rather than to a value TIER_ORDER ranks.
VERDICT_TIER: dict[str, str | None] = {
    "PERMISSIVE_OK": "mit",
    "ATTRIBUTION": "mit",
    "SHARE_ALIKE": "cc-by-sa-4.0",
    "NC": "cc-by-nc-sa-4.0",
    "REFUSE": None,
    "BLOCKING": None,
    "UNVERIFIED": None,
}

B1_MAX_SHARE = 0.40

# docs/design/DATASET-FACTORY-CATALOGUE-2026-09-03.md S8: "4 bytes per token" for text
# measured on disk, the catalogue's own stated INFERRED assumption for row/token
# estimates. Real provenance.json never emits `row_count`/`count` (confirmed against
# every dataset fetched 2026-09-03) -- this is the fallback, and every result computed
# from it is labelled ESTIMATED so it is never mistaken for a factory-measured count.
BYTES_PER_TOKEN = 4

# grant_scope values (docs/design/datasets/catalogue-2026-09-03.json note 3, plus the
# survey's `code_only`/`metadata_only`) that cover the dataset's actual text/content for
# training use. Anything else (`code_only`: only code/annotation layer is licensed, not
# the underlying text; `metadata_only`: only bibliographic metadata) means the grant does
# NOT cover what a corpus admission would actually emit -- structurally inadmissible
# regardless of the catalogue's `verdict`, per B1's narrativeqa finding.
FULL_CONTENT_GRANT_SCOPES: frozenset[str] = frozenset({"whole_corpus", "database_rights_only"})

DEFAULT_CATALOGUE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "design"
    / "datasets"
    / "catalogue-2026-09-03.json"
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_licence_tier(verdict: str, region: str, region_tier: dict[str, str]) -> CheckResult:
    """Check 1: the dataset's verdict must not demand a stricter tier than the region has.

    A region tier not in `region_tier` at all is treated as unknown, matching
    csd-publish-checkpoint.py's stated discipline of refusing rather than guessing a
    licence for an unaudited region -- an unknown region tier FAILs here too, loudly,
    rather than silently defaulting to `mit`.
    """
    required = VERDICT_TIER.get(verdict)
    if required is None:
        return CheckResult(
            "licence_tier",
            False,
            f"verdict={verdict!r} is inadmissible outright (REFUSE/BLOCKING/UNVERIFIED "
            "carry no tier)",
        )
    current = region_tier.get(region)
    if current is None:
        return CheckResult(
            "licence_tier",
            False,
            f"region {region!r} has no declared tier in REGION_TIER -- refusing rather "
            "than guessing (residual_mlp/stream_vae are the known unaudited regions)",
        )
    if current not in TIER_ORDER or required not in TIER_ORDER:
        return CheckResult(
            "licence_tier",
            False,
            f"unranked tier: region tier={current!r}, required tier={required!r}",
        )
    if TIER_ORDER.index(required) <= TIER_ORDER.index(current):
        return CheckResult(
            "licence_tier",
            True,
            f"verdict {verdict!r} requires tier {required!r}, region {region!r} already "
            f"at {current!r} -- no upgrade needed",
        )
    return CheckResult(
        "licence_tier",
        False,
        f"verdict {verdict!r} requires tier {required!r}, stricter than region {region!r}'s "
        f"current {current!r} -- requires an EXPLICIT tier upgrade (Rider-1-style), not "
        "silent admission",
    )


def check_b1_share(
    provenance_group: str,
    added_count: int,
    existing_shares: dict[str, int],
    max_share: float = B1_MAX_SHARE,
    estimated: bool = False,
) -> CheckResult:
    """Check 2: post-admission share of `provenance_group` must stay <= B1's 0.40 cap.

    Computed on provenance groups, per CORPUS-CONTRACT.md -- the caller is responsible for
    passing `existing_shares` keyed by provenance group, not by dataset name or shard.
    `estimated=True` (see `resolve_added_count`) means `added_count` was derived from
    `total_bytes` rather than read from a factory-emitted row/token count, and the detail
    string says so -- the share number is only ever as trustworthy as that count.
    """
    if added_count < 0:
        return CheckResult("b1_share", False, f"added_count={added_count} is negative")
    total_before = sum(existing_shares.values())
    total_after = total_before + added_count
    if total_after <= 0:
        return CheckResult("b1_share", False, "post-admission corpus would have zero rows")
    group_after = existing_shares.get(provenance_group, 0) + added_count
    share = group_after / total_after
    passed = share <= max_share
    count_note = " [ESTIMATED from total_bytes]" if estimated else ""
    detail = (
        f"group={provenance_group!r} share after admission = {share:.4f} "
        f"({group_after}/{total_after}), cap={max_share}{count_note}"
    )
    return CheckResult("b1_share", passed, detail)


def resolve_added_count(provenance: dict[str, Any]) -> tuple[int, bool]:
    """Resolve the row/token count this admission would add, and whether it is a real
    factory-emitted count or a fallback estimate.

    Prefers an explicit `row_count` or `count` field -- neither is emitted by the real
    dataset-factory `provenance.json` today (B4: every fetched-dataset provenance.json
    under /mnt/bulk/csd-corpus/factory-2026-09-03/ carries `total_bytes`/`file_count`/
    `file_manifest` but no row or token count), so falling through to that branch is
    expected on real input, not a bug. The estimate divides `total_bytes` by
    `BYTES_PER_TOKEN`, the catalogue's own stated assumption -- returns `(0, False)` when
    neither a count nor `total_bytes` is present, same as the prior always-zero behaviour,
    so a provenance.json missing everything still fails B1 loudly (0 rows added, 0 share)
    rather than crashing.
    """
    for key in ("row_count", "count"):
        value = provenance.get(key)
        if value is not None:
            return int(value), False
    total_bytes = provenance.get("total_bytes")
    if total_bytes is None:
        return 0, False
    return int(total_bytes) // BYTES_PER_TOKEN, True


def check_verified(verification_status: str) -> CheckResult:
    """Check 3: only a fresh VERIFIED read is admissible.

    CONTRADICTED, UNVERIFIABLE and REFUSED-CLOSED are all inadmissible regardless of the
    `verdict` field -- an entry does not get readmitted by re-running this checklist on
    stale evidence, only by a new primary read that flips verification_status itself.
    """
    passed = verification_status == "VERIFIED"
    return CheckResult(
        "verification_status",
        passed,
        f"verification_status={verification_status!r} "
        f"({'admissible' if passed else 'inadmissible regardless of verdict'})",
    )


def repo_id_of(provenance: dict[str, Any]) -> str:
    """The dataset's repo_id, from a top-level field (older/synthetic provenance shapes,
    e.g. this module's own test fixtures) or from the real factory shape's trimmed
    `catalogue_entry.repo_id` (every real provenance.json under
    /mnt/bulk/csd-corpus/factory-2026-09-03/ has no top-level `repo_id` at all)."""
    repo_id = provenance.get("repo_id")
    if repo_id:
        return str(repo_id)
    return str(provenance.get("catalogue_entry", {}).get("repo_id", "<unknown repo_id>"))


def load_catalogue_index(path: Path) -> dict[str, dict[str, Any]]:
    """Load `docs/design/datasets/catalogue-2026-09-03.json` (schema
    csd-dataset-factory-catalogue/v1) into a repo_id -> full catalogue row index, for
    check_catalogue_structural_refusals. This is the FULL row -- `grant_scope`,
    `redistribute`, `provenance_red_flags`, `enrichment_licence_result` included -- not
    the trimmed `catalogue_entry` a provenance.json carries.
    """
    data = _load_json(path)
    index: dict[str, dict[str, Any]] = {}
    for entry in data.get("entries", []):
        repo_id = entry.get("repo_id")
        if repo_id:
            index[str(repo_id)] = entry
    return index


def check_catalogue_structural_refusals(
    repo_id: str, catalogue_index: dict[str, dict[str, Any]]
) -> CheckResult:
    """Check 4 (second, independent look): apply the catalogue row's own structural
    refusal signals, cross-referenced by `repo_id` -- the same fields the factory's
    `admission.check_admission` parses into provenance.json but never gates on (ground-
    pass B3), read here from the catalogue directly rather than trusted from provenance.

    A repo_id absent from the catalogue index FAILS closed (refusing rather than
    admitting an entry this check cannot evaluate), matching check_licence_tier's
    discipline for an unaudited region.

    Refusal signals, each independently sufficient:
      - `grant_scope` not in FULL_CONTENT_GRANT_SCOPES (B1: narrativeqa's `code_only`
        licenses the annotation/code layer, not the underlying books/scripts a corpus
        admission would actually emit).
      - `enrichment_licence_result` containing a REFUSE marker (B2: BeIR/cqadupstack's
        "R9 REFUSE at ingest" -- Stack Exchange's per-item attribution obligation is a
        structural refusal, not a caveat).
      - `redistribute.nd` true (no-derivatives forbids training a derivative model on
        the data at all, independent of verdict/tier).

    `provenance_red_flags` is NOT itself a refusal signal -- google-research-datasets/paws
    (the clean admissible reference case) carries one ("Google's grant covers the dataset
    AS RELEASED...") and is still PERMISSIVE_OK/whole_corpus/no-REFUSE. Red flags are
    surfaced in the detail string on every row (pass or fail) so a human reviewing the
    checklist output sees them, but they do not gate on their own -- gating on presence
    alone would refuse the one dataset the survey names as the cleanest grant in it.
    """
    row = catalogue_index.get(repo_id)
    if row is None:
        return CheckResult(
            "catalogue_structural_refusals",
            False,
            f"repo_id={repo_id!r} not found in the catalogue index -- refusing rather "
            "than admitting an entry this check cannot evaluate",
        )
    reasons: list[str] = []
    grant_scope = row.get("grant_scope")
    if grant_scope not in FULL_CONTENT_GRANT_SCOPES:
        reasons.append(
            f"grant_scope={grant_scope!r} does not cover full-content use "
            f"(admissible: {sorted(FULL_CONTENT_GRANT_SCOPES)})"
        )
    enrichment_result = str(row.get("enrichment_licence_result") or "")
    if "refuse" in enrichment_result.lower():
        reasons.append(f"enrichment_licence_result carries a REFUSE marker: {enrichment_result!r}")
    redistribute = row.get("redistribute") or {}
    if redistribute.get("nd"):
        reasons.append("redistribute.nd=True: no-derivatives forbids a trained derivative model")
    red_flags = row.get("provenance_red_flags") or []
    flags_note = f"; provenance_red_flags={red_flags!r}" if red_flags else ""
    if reasons:
        detail = "; ".join(reasons) + flags_note
        return CheckResult("catalogue_structural_refusals", False, detail)
    return CheckResult(
        "catalogue_structural_refusals",
        True,
        f"grant_scope={grant_scope!r}, no enrichment REFUSE marker, redistribute.nd not "
        f"set{flags_note}",
    )


def run_checklist(
    provenance: dict[str, Any],
    region: str,
    existing_shares: dict[str, int],
    region_tier: dict[str, str] | None = None,
    catalogue_index: dict[str, dict[str, Any]] | None = None,
) -> list[CheckResult]:
    """Run the admission checklist. `catalogue_index` (see `load_catalogue_index`) is
    optional so unit tests can exercise checks 1-3 in isolation with a synthetic
    provenance dict and no catalogue on disk; the CLI (`main`) always loads and passes
    one -- check 4 is part of every real admission decision, not an opt-in extra.
    """
    tier_table = REGION_TIER if region_tier is None else region_tier
    verdict = provenance.get("verdict", "")
    provenance_group = provenance.get("provenance_group", "")
    added_count, estimated = resolve_added_count(provenance)
    verification_status = provenance.get("verification_status", "")
    results = [
        check_licence_tier(verdict, region, tier_table),
        check_b1_share(provenance_group, added_count, existing_shares, estimated=estimated),
        check_verified(verification_status),
    ]
    if catalogue_index is not None:
        results.append(check_catalogue_structural_refusals(repo_id_of(provenance), catalogue_index))
    return results


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provenance",
        required=True,
        type=Path,
        help="Path to the factory's provenance.json for one fetched dataset.",
    )
    parser.add_argument(
        "--region",
        required=True,
        help="Target CSD region name (e.g. retrieve, code, memory).",
    )
    parser.add_argument(
        "--existing-shares",
        default="{}",
        help="JSON object: provenance_group -> current row/token count in the region's "
        "corpus, before this admission. Defaults to empty (a fresh region).",
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=DEFAULT_CATALOGUE_PATH,
        help="Path to the full dataset-factory catalogue JSON (schema "
        "csd-dataset-factory-catalogue/v1), for check 4's second, independent structural-"
        "refusal look. Defaults to docs/design/datasets/catalogue-2026-09-03.json in this "
        "repo. Pass --no-catalogue-check to skip check 4 entirely.",
    )
    parser.add_argument(
        "--no-catalogue-check",
        action="store_true",
        help="Skip check 4 (catalogue structural refusals). Only for the checklist's "
        "checks 1-3 against a provenance.json in isolation -- production admission "
        "decisions should keep check 4 on.",
    )
    args = parser.parse_args(argv)

    provenance = _load_json(args.provenance)
    try:
        existing_shares = json.loads(args.existing_shares)
    except json.JSONDecodeError as exc:
        parser.error(f"--existing-shares is not valid JSON: {exc}")
        return 2  # pragma: no cover - argparse.error already exits

    catalogue_index = None if args.no_catalogue_check else load_catalogue_index(args.catalogue)

    results = run_checklist(
        provenance, args.region, existing_shares, catalogue_index=catalogue_index
    )

    repo_id = repo_id_of(provenance)
    print(f"Admission checklist: {repo_id} -> region {args.region!r}")
    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        all_passed = all_passed and result.passed
        print(f"  [{status}] {result.name}: {result.detail}")
    print(f"Result: {'ADMIT' if all_passed else 'REFUSE'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
