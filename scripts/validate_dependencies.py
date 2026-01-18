#!/usr/bin/env python3
"""Script to validate project dependencies against actual imports.

This script checks that all imported packages are declared in requirements files
and identifies unused dependencies.

Usage:
    python scripts/validate_dependencies.py [--report FILE]
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cogsyndelta.documentation import DependencyValidator


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate project dependencies"
    )
    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        help="Output detailed JSON report to file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format",
    )

    args = parser.parse_args()

    # Initialize validator
    project_root = Path(__file__).parent.parent
    validator = DependencyValidator(project_root)

    if args.json:
        # Generate JSON report
        report_data = validator.validate_all_requirements()
        print(json.dumps(report_data, indent=2))

        if args.report:
            with open(args.report, "w") as f:
                json.dump(report_data, f, indent=2)
            print(f"\nDetailed report saved to: {args.report}", file=sys.stderr)
    else:
        # Generate human-readable report
        report = validator.generate_validation_report()
        print(report)

        if args.report:
            with open(args.report, "w") as f:
                f.write(report)
            print(f"\nReport saved to: {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
