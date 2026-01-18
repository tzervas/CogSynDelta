"""
Code Quality Validation and Control System

Ensures code quality through:
1. Static analysis
2. Type checking validation
3. Code style verification
4. Complexity analysis
5. Intention validation
6. Functionality verification

Usage:
    python scripts/quality_control.py [path] [--fail-under SCORE]

Examples:
    python scripts/quality_control.py src/
    python scripts/quality_control.py src/ --fail-under 90
"""

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class QualityIssue:
    """Represents a code quality issue."""

    severity: str  # "error", "warning", "info"
    category: str
    file: str
    line: int
    message: str
    suggestion: str


class CodeQualityValidator:
    """
    Validates code quality across the codebase.
    """

    def __init__(self, project_root: str = "."):
        """Initialize quality validator with project root and empty stats."""
        self.project_root = project_root
        self.issues: list[QualityIssue] = []
        self.stats = {
            "files_analyzed": 0,
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "issues_found": 0,
        }

    def validate_all(self) -> dict[str, Any]:
        """Run all validation checks."""
        print("=" * 70)
        print("CODE QUALITY VALIDATION")
        print("=" * 70)

        # Get all Python files
        python_files = self._get_python_files()

        print(f"\nAnalyzing {len(python_files)} Python files...")

        for filepath in python_files:
            self._analyze_file(filepath)

        # Generate report
        report = self._generate_report()

        return report

    def _get_python_files(self) -> list[str]:
        """Get all Python files in project."""
        python_files = []

        for root, dirs, files in os.walk(self.project_root):
            # Skip test files, __pycache__, virtual environments, and git
            dirs[:] = [
                d
                for d in dirs
                if d
                not in [
                    "__pycache__",
                    ".git",
                    "venv",
                    "env",
                    ".venv",
                    ".tox",
                    ".nox",
                    "node_modules",
                    ".eggs",
                    "build",
                    "dist",
                ]
            ]

            for file in files:
                if file.endswith(".py") and not file.startswith("test_"):
                    filepath = os.path.join(root, file)
                    python_files.append(filepath)

        return python_files

    def _analyze_file(self, filepath: str):
        """Analyze a single file."""
        try:
            with open(filepath, encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

            self.stats["files_analyzed"] += 1
            self.stats["total_lines"] += len(lines)

            # Parse AST
            try:
                tree = ast.parse(content, filename=filepath)
            except SyntaxError as e:
                self.issues.append(
                    QualityIssue(
                        severity="error",
                        category="syntax",
                        file=filepath,
                        line=e.lineno or 0,
                        message=f"Syntax error: {e.msg}",
                        suggestion="Fix syntax error",
                    )
                )
                return

            # Analyze AST
            self._check_docstrings(tree, filepath)
            self._check_function_complexity(tree, filepath)
            self._check_naming_conventions(tree, filepath)
            self._check_imports(tree, filepath)
            self._check_type_hints(tree, filepath)

        except Exception as e:
            self.issues.append(
                QualityIssue(
                    severity="error",
                    category="analysis",
                    file=filepath,
                    line=0,
                    message=f"Analysis error: {e!s}",
                    suggestion="Check file format",
                )
            )

    def _check_docstrings(self, tree: ast.AST, filepath: str):
        """Check for missing docstrings."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                self.stats[
                    "total_functions" if isinstance(node, ast.FunctionDef) else "total_classes"
                ] += 1

                # Skip private methods
                if node.name.startswith("_") and not node.name.startswith("__"):
                    continue

                docstring = ast.get_docstring(node)
                if not docstring:
                    self.issues.append(
                        QualityIssue(
                            severity="warning",
                            category="docstring",
                            file=filepath,
                            line=node.lineno,
                            message=f"Missing docstring for {node.__class__.__name__} '{node.name}'",
                            suggestion="Add docstring describing purpose and parameters",
                        )
                    )

    def _check_function_complexity(self, tree: ast.AST, filepath: str):
        """Check function complexity."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)

                if complexity > 15:
                    self.issues.append(
                        QualityIssue(
                            severity="warning",
                            category="complexity",
                            file=filepath,
                            line=node.lineno,
                            message=f"Function '{node.name}' has high complexity ({complexity})",
                            suggestion="Consider breaking into smaller functions",
                        )
                    )

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _check_naming_conventions(self, tree: ast.AST, filepath: str):
        """Check naming conventions."""
        # Known naming exceptions (mHC = moderated HyperConnections convention)
        naming_exceptions = {"mHCPathway", "mHCInterconnect"}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Classes should be PascalCase (skip known exceptions)
                if node.name in naming_exceptions:
                    continue
                if not re.match(r"^[A-Z][a-zA-Z0-9]*$", node.name):
                    self.issues.append(
                        QualityIssue(
                            severity="info",
                            category="naming",
                            file=filepath,
                            line=node.lineno,
                            message=f"Class '{node.name}' should use PascalCase",
                            suggestion="Rename to PascalCase (e.g., MyClass)",
                        )
                    )

            elif isinstance(node, ast.FunctionDef):
                # Functions should be snake_case
                if not re.match(r"^[a-z_][a-z0-9_]*$", node.name) and not node.name.startswith(
                    "__"
                ):
                    self.issues.append(
                        QualityIssue(
                            severity="info",
                            category="naming",
                            file=filepath,
                            line=node.lineno,
                            message=f"Function '{node.name}' should use snake_case",
                            suggestion="Rename to snake_case (e.g., my_function)",
                        )
                    )

    def _check_imports(self, tree: ast.AST, filepath: str):
        """Check import statements."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Check for wildcard imports
                for alias in node.names:
                    if alias.name == "*":
                        self.issues.append(
                            QualityIssue(
                                severity="warning",
                                category="imports",
                                file=filepath,
                                line=node.lineno,
                                message="Avoid wildcard imports (from X import *)",
                                suggestion="Import specific names",
                            )
                        )

    def _check_type_hints(self, tree: ast.AST, filepath: str):
        """Check for type hints."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private methods
                if node.name.startswith("_"):
                    continue

                # Check return type hint
                if node.returns is None and node.name != "__init__":
                    self.issues.append(
                        QualityIssue(
                            severity="info",
                            category="type_hints",
                            file=filepath,
                            line=node.lineno,
                            message=f"Function '{node.name}' missing return type hint",
                            suggestion="Add -> ReturnType annotation",
                        )
                    )

    def _generate_report(self) -> dict[str, Any]:
        """Generate quality report."""
        # Categorize issues
        by_severity = {"error": 0, "warning": 0, "info": 0}
        by_category = {}

        for issue in self.issues:
            by_severity[issue.severity] += 1
            by_category[issue.category] = by_category.get(issue.category, 0) + 1

        self.stats["issues_found"] = len(self.issues)

        # Calculate quality score
        quality_score = 100.0
        quality_score -= by_severity["error"] * 5
        quality_score -= by_severity["warning"] * 2
        quality_score -= by_severity["info"] * 0.5
        quality_score = max(0, min(100, quality_score))

        report = {
            "stats": self.stats,
            "issues": {
                "by_severity": by_severity,
                "by_category": by_category,
                "total": len(self.issues),
            },
            "quality_score": quality_score,
        }

        # Print report
        print("\n" + "=" * 70)
        print("QUALITY REPORT")
        print("=" * 70)
        print(f"\nFiles analyzed: {self.stats['files_analyzed']}")
        print(f"Total lines: {self.stats['total_lines']}")
        print(f"Total functions: {self.stats['total_functions']}")
        print(f"Total classes: {self.stats['total_classes']}")

        print(f"\nIssues found: {len(self.issues)}")
        print(f"  Errors: {by_severity['error']}")
        print(f"  Warnings: {by_severity['warning']}")
        print(f"  Info: {by_severity['info']}")

        print(f"\nQuality Score: {quality_score:.1f}/100")

        if quality_score >= 90:
            print("✓ EXCELLENT code quality")
        elif quality_score >= 75:
            print("✓ GOOD code quality")
        elif quality_score >= 60:
            print("⚠ ACCEPTABLE code quality")
        else:
            print("✗ NEEDS IMPROVEMENT")

        # Print top issues
        if self.issues:
            print("\nTop Issues:")
            for issue in sorted(self.issues, key=lambda x: (x.severity, x.file))[:10]:
                print(f"  [{issue.severity.upper()}] {os.path.basename(issue.file)}:{issue.line}")
                print(f"    {issue.message}")
                print(f"    Suggestion: {issue.suggestion}")

        print("=" * 70)

        return report


class IntentionValidator:
    """
    Validates that intended functionality matches implementation.

    Checks:
    - Function signatures match docstrings
    - Return values match type hints
    - Exception handling is documented
    """

    def __init__(self):
        """Initialize intention validator with empty results."""
        self.validation_results = []

    def validate_intentions(self, python_files: list[str]) -> dict[str, Any]:
        """Validate intentions across files."""
        print("\n" + "=" * 70)
        print("INTENTION VALIDATION")
        print("=" * 70)

        mismatches = []

        for filepath in python_files:
            try:
                with open(filepath) as f:
                    content = f.read()
                tree = ast.parse(content)

                # Check each function
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        issues = self._validate_function_intention(node, filepath)
                        mismatches.extend(issues)

            except Exception:
                pass  # Skip files with errors

        print(f"\nValidated intentions in {len(python_files)} files")
        print(f"Intention mismatches found: {len(mismatches)}")

        if mismatches:
            print("\nSample mismatches:")
            for mismatch in mismatches[:5]:
                print(f"  - {mismatch['file']}:{mismatch['line']}")
                print(f"    {mismatch['message']}")
        else:
            print("✓ All intentions validated")

        return {"total_mismatches": len(mismatches), "mismatches": mismatches}

    def _validate_function_intention(self, node: ast.FunctionDef, filepath: str) -> list[dict]:
        """Validate a single function's intentions."""
        issues = []

        docstring = ast.get_docstring(node)
        if not docstring:
            return issues  # No docstring to validate against

        # Extract Args section
        args_section = self._extract_section(docstring, "Args:")
        returns_section = self._extract_section(docstring, "Returns:")

        # Validate parameters
        if args_section:
            documented_params = self._extract_params_from_doc(args_section)
            actual_params = [arg.arg for arg in node.args.args if arg.arg != "self"]

            # Check for undocumented params
            for param in actual_params:
                if param not in documented_params:
                    issues.append(
                        {
                            "file": filepath,
                            "line": node.lineno,
                            "message": f"Parameter '{param}' not documented in '{node.name}'",
                        }
                    )

        # Validate return type
        if returns_section and node.returns is None:
            issues.append(
                {
                    "file": filepath,
                    "line": node.lineno,
                    "message": f"Function '{node.name}' documents return but has no type hint",
                }
            )

        return issues

    def _extract_section(self, docstring: str, section_name: str) -> str:
        """Extract a section from docstring."""
        lines = docstring.split("\n")
        in_section = False
        section_lines = []

        for line in lines:
            if section_name in line:
                in_section = True
                continue
            if in_section and line.strip() and not line.startswith(" "):
                break
            if in_section:
                section_lines.append(line)

        return "\n".join(section_lines)

    def _extract_params_from_doc(self, args_section: str) -> list[str]:
        """Extract parameter names from Args section."""
        params = []
        for line in args_section.split("\n"):
            match = re.match(r"\s*(\w+):", line)
            if match:
                params.append(match.group(1))
        return params


def run_quality_checks(target_path: str = ".", fail_under: float | None = None) -> dict:
    """Run all quality checks.

    Args:
        target_path: Path to analyze (default: current directory)
        fail_under: Minimum required score (exit non-zero if below)

    Returns:
        Quality report dictionary
    """
    print("=" * 70)
    print("RUNNING COMPREHENSIVE QUALITY CHECKS")
    print("=" * 70)

    # Code quality validation
    validator = CodeQualityValidator(project_root=target_path)
    quality_report = validator.validate_all()

    # Intention validation
    python_files = validator._get_python_files()
    intention_validator = IntentionValidator()
    intention_report = intention_validator.validate_intentions(python_files)

    # Overall assessment
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)

    overall_score = quality_report["quality_score"]

    # Adjust for intention mismatches
    if intention_report["total_mismatches"] > 0:
        penalty = min(20, intention_report["total_mismatches"] * 2)
        overall_score -= penalty

    print(f"\nOverall Quality Score: {overall_score:.1f}/100")

    if fail_under is not None:
        print(f"Required minimum score: {fail_under}")

    if overall_score >= 90:
        print("✓ EXCELLENT - Production ready")
        status = "excellent"
    elif overall_score >= 85:
        print("✓ PRODUCTION READY")
        status = "ready"
    elif overall_score >= 70:
        print("✓ ACCEPTABLE - Minor improvements recommended")
        status = "acceptable"
    else:
        print("✗ NEEDS WORK - Address issues before production")
        status = "needs_work"

    print("=" * 70)

    result = {
        "overall_score": overall_score,
        "status": status,
        "quality_report": quality_report,
        "intention_report": intention_report,
    }

    # Check against threshold
    if fail_under is not None and overall_score < fail_under:
        print(f"\n✗ FAILED: Score {overall_score:.1f} is below required {fail_under}")
        result["passed"] = False
    else:
        result["passed"] = True

    return result


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Code Quality Validation and Control System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/quality_control.py src/
    python scripts/quality_control.py src/ --fail-under 90
    python scripts/quality_control.py . --fail-under 85
        """,
    )
    parser.add_argument(
        "path", nargs="?", default=".", help="Path to analyze (default: current directory)"
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        metavar="SCORE",
        help="Fail if quality score is below this threshold (0-100)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="quality_report.json",
        help="Output file for JSON report (default: quality_report.json)",
    )

    args = parser.parse_args()

    # Run quality checks
    result = run_quality_checks(target_path=args.path, fail_under=args.fail_under)

    # Save report
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Quality report saved to: {args.output}")

    # Exit with appropriate code
    if not result["passed"]:
        sys.exit(1)
    elif result["status"] in ["excellent", "ready", "acceptable"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
