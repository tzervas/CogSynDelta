#!/usr/bin/env python3
"""Print the CORPUS-CONTRACT.md admission checklist for one factory-fetched dataset.

WHAT THIS DOES AND DOES NOT DO
This is a read-only gate, not a fetcher and not a corpus mutator. It reads one dataset
factory `provenance.json` (see program/datasets/README.md `S2`) plus a target CSD region
name, and prints three checks: licence tier vs. the region's already-declared tier,
provenance-group share against CORPUS-CONTRACT.md's B1 (<= 0.40 post-admission), and
`verification_status == VERIFIED`. It does not touch CORPUS-CONTRACT.md, csd-regions.json,
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
) -> CheckResult:
    """Check 2: post-admission share of `provenance_group` must stay <= B1's 0.40 cap.

    Computed on provenance groups, per CORPUS-CONTRACT.md -- the caller is responsible for
    passing `existing_shares` keyed by provenance group, not by dataset name or shard.
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
    detail = (
        f"group={provenance_group!r} share after admission = {share:.4f} "
        f"({group_after}/{total_after}), cap={max_share}"
    )
    return CheckResult("b1_share", passed, detail)


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


def run_checklist(
    provenance: dict[str, Any],
    region: str,
    existing_shares: dict[str, int],
    region_tier: dict[str, str] | None = None,
) -> list[CheckResult]:
    tier_table = REGION_TIER if region_tier is None else region_tier
    verdict = provenance.get("verdict", "")
    provenance_group = provenance.get("provenance_group", "")
    added_count = int(provenance.get("row_count") or provenance.get("count") or 0)
    verification_status = provenance.get("verification_status", "")
    return [
        check_licence_tier(verdict, region, tier_table),
        check_b1_share(provenance_group, added_count, existing_shares),
        check_verified(verification_status),
    ]


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
    args = parser.parse_args(argv)

    provenance = _load_json(args.provenance)
    try:
        existing_shares = json.loads(args.existing_shares)
    except json.JSONDecodeError as exc:
        parser.error(f"--existing-shares is not valid JSON: {exc}")
        return 2  # pragma: no cover - argparse.error already exits

    results = run_checklist(provenance, args.region, existing_shares)

    repo_id = provenance.get("repo_id", "<unknown repo_id>")
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
