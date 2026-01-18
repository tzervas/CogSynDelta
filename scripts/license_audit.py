#!/usr/bin/env python3
"""License audit script for CogSynDelta dependencies.

This script scans project dependencies and cross-references them against
the LICENSE_TRACKER.md to identify:
- Missing entries in the tracker
- Outdated version information
- License compliance issues
- New transitive dependencies

Usage:
    python scripts/license_audit.py [--update] [--format json|markdown]

References:
    - LICENSES/LICENSE_TRACKER.md
    - ADR-0002: Memory Architecture (for audit trail rationale)
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# License compatibility matrix for permissive licenses
# These can be combined with MIT without issue
PERMISSIVE_LICENSES: frozenset[str] = frozenset(
    {
        "MIT",
        "MIT License",
        "BSD",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "Apache-2.0",
        "Apache 2.0",
        "Apache Software License",
        "ISC",
        "PSF",
        "Python Software Foundation License",
        "Unlicense",
        "CC0",
        "Public Domain",
        "WTFPL",
    }
)

# Licenses that require attribution (already satisfied by LICENSE_TRACKER.md)
ATTRIBUTION_REQUIRED: frozenset[str] = frozenset(
    {
        "MIT",
        "MIT License",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "Apache-2.0",
        "Apache 2.0",
    }
)

# Copyleft licenses that need careful review
COPYLEFT_LICENSES: frozenset[str] = frozenset(
    {
        "GPL",
        "GPLv2",
        "GPLv3",
        "GPL-2.0",
        "GPL-3.0",
        "LGPL",
        "LGPLv2",
        "LGPLv3",
        "LGPL-2.0",
        "LGPL-3.0",
        "AGPL",
        "AGPLv3",
        "AGPL-3.0",
        "MPL",
        "MPL-2.0",
    }
)


@dataclass
class DependencyInfo:
    """Information about a single dependency."""

    name: str
    version: str
    license: str
    homepage: str = ""
    is_tracked: bool = False
    tracker_version: str = ""
    needs_review: bool = False
    review_reason: str = ""


@dataclass
class AuditResult:
    """Results of the license audit."""

    dependencies: list[DependencyInfo] = field(default_factory=list)
    missing_from_tracker: list[str] = field(default_factory=list)
    version_mismatches: list[tuple[str, str, str]] = field(default_factory=list)
    copyleft_warnings: list[str] = field(default_factory=list)
    unknown_licenses: list[str] = field(default_factory=list)
    total_scanned: int = 0
    compliant: bool = True


def get_installed_packages() -> list[dict[str, str]]:
    """Get list of installed packages using pip.

    Returns:
        List of dicts with name, version, and license for each package.

    Note:
        Uses pip's JSON output format for reliable parsing.
        Falls back to basic info if metadata unavailable.
    """
    packages = []

    try:
        # Get package list with metadata
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )
        pip_packages = json.loads(result.stdout)

        for pkg in pip_packages:
            name = pkg["name"]
            version = pkg["version"]

            # Get detailed info including license
            try:
                show_result = subprocess.run(
                    [sys.executable, "-m", "pip", "show", name],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                license_match = re.search(
                    r"^License:\s*(.+)$",
                    show_result.stdout,
                    re.MULTILINE,
                )
                license_info = license_match.group(1).strip() if license_match else "UNKNOWN"

                home_match = re.search(
                    r"^Home-page:\s*(.+)$",
                    show_result.stdout,
                    re.MULTILINE,
                )
                homepage = home_match.group(1).strip() if home_match else ""

            except subprocess.CalledProcessError:
                license_info = "UNKNOWN"
                homepage = ""

            packages.append(
                {
                    "name": name,
                    "version": version,
                    "license": license_info,
                    "homepage": homepage,
                }
            )

    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Warning: Could not get package list: {e}", file=sys.stderr)

    return packages


def parse_license_tracker(tracker_path: Path) -> dict[str, dict[str, str]]:
    """Parse LICENSE_TRACKER.md to extract tracked dependencies.

    Args:
        tracker_path: Path to the LICENSE_TRACKER.md file.

    Returns:
        Dict mapping package names (lowercase) to their tracked info.
    """
    tracked: dict[str, dict[str, str]] = {}

    if not tracker_path.exists():
        print(f"Warning: {tracker_path} not found", file=sys.stderr)
        return tracked

    content = tracker_path.read_text()

    # Parse markdown table rows (| Name | Version | License | ... |)
    # Match pattern: | package-name | version | license | source |
    table_pattern = re.compile(
        r"^\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|",
        re.MULTILINE,
    )

    for match in table_pattern.finditer(content):
        name = match.group(1).strip()
        version = match.group(2).strip()
        license_info = match.group(3).strip()
        source = match.group(4).strip()

        # Skip header row and separator
        if name in ("Dependency", "---", "----------") or name.startswith("-"):
            continue

        # Normalize name for comparison (lowercase, handle - vs _)
        normalized = name.lower().replace("-", "_")
        tracked[normalized] = {
            "name": name,
            "version": version,
            "license": license_info,
            "source": source,
        }

    return tracked


def audit_dependencies(
    packages: list[dict[str, str]],
    tracked: dict[str, dict[str, str]],
) -> AuditResult:
    """Audit installed packages against tracked dependencies.

    Args:
        packages: List of installed packages from pip.
        tracked: Dict of tracked dependencies from LICENSE_TRACKER.md.

    Returns:
        AuditResult with all findings.
    """
    result = AuditResult()
    result.total_scanned = len(packages)

    for pkg in packages:
        name = pkg["name"]
        normalized = name.lower().replace("-", "_")
        version = pkg["version"]
        license_info = pkg["license"]

        dep_info = DependencyInfo(
            name=name,
            version=version,
            license=license_info,
            homepage=pkg.get("homepage", ""),
        )

        # Check if tracked
        if normalized in tracked:
            dep_info.is_tracked = True
            dep_info.tracker_version = tracked[normalized]["version"]

            # Check version mismatch (only if tracker has specific version)
            tracker_ver = tracked[normalized]["version"]
            if tracker_ver and tracker_ver not in ("*", "latest", version):
                # Allow for version prefixes like >=, ~=, etc.
                if not tracker_ver.startswith((">=", "~=", "^", ">")):
                    result.version_mismatches.append((name, tracker_ver, version))
        else:
            result.missing_from_tracker.append(name)

        # Check license compatibility
        if license_info in COPYLEFT_LICENSES:
            dep_info.needs_review = True
            dep_info.review_reason = f"Copyleft license: {license_info}"
            result.copyleft_warnings.append(f"{name} ({license_info})")
            result.compliant = False
        elif license_info == "UNKNOWN" or not license_info:
            dep_info.needs_review = True
            dep_info.review_reason = "License unknown - manual review required"
            result.unknown_licenses.append(name)
        elif license_info not in PERMISSIVE_LICENSES:
            # Not in our known permissive list - may be okay but flag for review
            dep_info.needs_review = True
            dep_info.review_reason = f"Uncommon license: {license_info}"

        result.dependencies.append(dep_info)

    return result


def format_markdown_report(result: AuditResult) -> str:
    """Format audit result as markdown report.

    Args:
        result: The audit result to format.

    Returns:
        Formatted markdown string.
    """
    lines = [
        "# License Audit Report",
        "",
        f"**Total packages scanned:** {result.total_scanned}",
        f"**Compliance status:** {'✅ COMPLIANT' if result.compliant else '⚠️ REVIEW REQUIRED'}",
        "",
    ]

    if result.missing_from_tracker:
        lines.extend(
            [
                "## Missing from LICENSE_TRACKER.md",
                "",
                "The following packages are installed but not tracked:",
                "",
            ]
        )
        for name in sorted(result.missing_from_tracker):
            lines.append(f"- `{name}`")
        lines.append("")

    if result.version_mismatches:
        lines.extend(
            [
                "## Version Mismatches",
                "",
                "| Package | Tracked | Installed |",
                "|---------|---------|-----------|",
            ]
        )
        for name, tracked_ver, installed_ver in sorted(result.version_mismatches):
            lines.append(f"| {name} | {tracked_ver} | {installed_ver} |")
        lines.append("")

    if result.copyleft_warnings:
        lines.extend(
            [
                "## ⚠️ Copyleft License Warnings",
                "",
                "These packages have copyleft licenses that may affect distribution:",
                "",
            ]
        )
        for warning in sorted(result.copyleft_warnings):
            lines.append(f"- {warning}")
        lines.append("")

    if result.unknown_licenses:
        lines.extend(
            [
                "## Unknown Licenses",
                "",
                "These packages need manual license verification:",
                "",
            ]
        )
        for name in sorted(result.unknown_licenses):
            lines.append(f"- `{name}`")
        lines.append("")

    # Summary of packages needing review
    needs_review = [d for d in result.dependencies if d.needs_review]
    if needs_review:
        lines.extend(
            [
                "## Packages Requiring Review",
                "",
                "| Package | License | Reason |",
                "|---------|---------|--------|",
            ]
        )
        for dep in sorted(needs_review, key=lambda d: d.name):
            lines.append(f"| {dep.name} | {dep.license} | {dep.review_reason} |")
        lines.append("")

    return "\n".join(lines)


def format_json_report(result: AuditResult) -> str:
    """Format audit result as JSON.

    Args:
        result: The audit result to format.

    Returns:
        JSON string representation.
    """
    output = {
        "total_scanned": result.total_scanned,
        "compliant": result.compliant,
        "missing_from_tracker": result.missing_from_tracker,
        "version_mismatches": [
            {"name": name, "tracked": tracked, "installed": installed}
            for name, tracked, installed in result.version_mismatches
        ],
        "copyleft_warnings": result.copyleft_warnings,
        "unknown_licenses": result.unknown_licenses,
        "dependencies": [
            {
                "name": d.name,
                "version": d.version,
                "license": d.license,
                "is_tracked": d.is_tracked,
                "needs_review": d.needs_review,
                "review_reason": d.review_reason,
            }
            for d in result.dependencies
        ],
    }
    return json.dumps(output, indent=2)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the license audit.

    Args:
        argv: Command line arguments. Uses sys.argv if None.

    Returns:
        Exit code: 0 for compliant, 1 for issues found, 2 for errors.
    """
    parser = argparse.ArgumentParser(
        description="Audit project dependencies for license compliance",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write report to file instead of stdout",
    )
    parser.add_argument(
        "--tracker",
        type=Path,
        default=Path("LICENSES/LICENSE_TRACKER.md"),
        help="Path to LICENSE_TRACKER.md (default: LICENSES/LICENSE_TRACKER.md)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any packages are missing from tracker",
    )

    args = parser.parse_args(argv)

    # Find project root (where LICENSE_TRACKER.md should be)
    project_root = Path.cwd()
    tracker_path = project_root / args.tracker

    print("Scanning installed packages...", file=sys.stderr)
    packages = get_installed_packages()

    if not packages:
        print("Error: No packages found", file=sys.stderr)
        return 2

    print(f"Found {len(packages)} packages", file=sys.stderr)
    print(f"Parsing {tracker_path}...", file=sys.stderr)

    tracked = parse_license_tracker(tracker_path)
    print(f"Found {len(tracked)} tracked dependencies", file=sys.stderr)

    print("Running audit...", file=sys.stderr)
    result = audit_dependencies(packages, tracked)

    # Format output
    if args.format == "json":
        report = format_json_report(result)
    else:
        report = format_markdown_report(result)

    # Write output
    if args.output:
        args.output.write_text(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)

    # Determine exit code
    if not result.compliant:
        print("\n⚠️ License compliance issues found!", file=sys.stderr)
        return 1

    if args.strict and result.missing_from_tracker:
        print(
            f"\n⚠️ {len(result.missing_from_tracker)} packages missing from tracker (strict mode)",
            file=sys.stderr,
        )
        return 1

    print("\n✅ All licenses compliant", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
