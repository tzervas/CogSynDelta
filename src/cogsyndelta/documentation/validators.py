"""Dependency validation utilities.

This module provides tools for validating project dependencies against
actual imports, checking version compatibility, and querying the RAG system.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import re


class DependencyValidator:
    """Validator for project dependencies and imports."""

    def __init__(self, project_root: Path) -> None:
        """Initialize the validator.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"

    def extract_imports_from_file(self, file_path: Path) -> Set[str]:
        """Extract all import statements from a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            Set of module names being imported
        """
        imports = set()

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get the top-level package name
                        package = alias.name.split(".")[0]
                        imports.add(package)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Get the top-level package name
                        package = node.module.split(".")[0]
                        imports.add(package)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")

        return imports

    def scan_project_imports(self) -> Dict[str, List[Path]]:
        """Scan all Python files in the project for imports.

        Returns:
            Dictionary mapping package names to files that import them
        """
        package_to_files: Dict[str, List[Path]] = {}

        # Find all Python files
        python_files = list(self.src_dir.rglob("*.py"))
        python_files.extend(self.project_root.glob("*.py"))
        python_files.extend((self.project_root / "tests").rglob("*.py"))
        python_files.extend((self.project_root / "examples").rglob("*.py"))
        python_files.extend((self.project_root / "benchmarks").rglob("*.py"))
        python_files.extend((self.project_root / "scripts").rglob("*.py"))

        # Extract imports from each file
        for py_file in python_files:
            if "__pycache__" in str(py_file):
                continue

            imports = self.extract_imports_from_file(py_file)
            for package in imports:
                if package not in package_to_files:
                    package_to_files[package] = []
                package_to_files[package].append(py_file)

        return package_to_files

    def parse_requirements_file(self, requirements_path: Path) -> Dict[str, str]:
        """Parse a requirements.txt file.

        Args:
            requirements_path: Path to requirements.txt

        Returns:
            Dictionary mapping package names to version specifiers
        """
        requirements = {}

        try:
            with open(requirements_path, "r") as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith("#"):
                        continue

                    # Parse package and version
                    # Handle formats: package==1.0, package>=1.0,<2.0, package
                    match = re.match(r"^([a-zA-Z0-9_-]+[a-zA-Z0-9_.-]*)", line)
                    if match:
                        package = match.group(1)
                        # Get version spec (everything after package name)
                        version_spec = line[len(package) :].strip()
                        # Remove [extras] if present
                        version_spec = re.sub(r"\[.*?\]", "", version_spec)
                        requirements[package] = version_spec or "any"
        except Exception as e:
            print(f"Error parsing {requirements_path}: {e}")

        return requirements

    def get_installed_version(self, package_name: str) -> Optional[str]:
        """Get the installed version of a package.

        Args:
            package_name: Name of the package

        Returns:
            Version string or None if not installed
        """
        try:
            # Handle package name variations
            normalized_name = package_name.replace("_", "-").lower()
            version = importlib.metadata.version(normalized_name)
            return version
        except importlib.metadata.PackageNotFoundError:
            # Try alternative name formats
            try:
                alternative_name = package_name.replace("-", "_").lower()
                version = importlib.metadata.version(alternative_name)
                return version
            except:
                return None

    def check_unused_dependencies(
        self, requirements_path: Path
    ) -> Tuple[List[str], List[str]]:
        """Check for dependencies declared but not imported.

        Args:
            requirements_path: Path to requirements file

        Returns:
            Tuple of (unused_dependencies, untracked_imports)
        """
        # Get declared dependencies
        declared = set(self.parse_requirements_file(requirements_path).keys())

        # Get actual imports
        imports = set(self.scan_project_imports().keys())

        # Filter out standard library and internal modules
        stdlib_modules = set(sys.stdlib_module_names)
        imports = {imp for imp in imports if imp not in stdlib_modules}
        imports = {imp for imp in imports if not imp.startswith("cogsyndelta")}

        # Map common package name differences
        package_name_map = {
            "cv2": "opencv-python",
            "PIL": "pillow",
            "yaml": "pyyaml",
            "sklearn": "scikit-learn",
        }

        # Normalize imports
        normalized_imports = set()
        for imp in imports:
            normalized_imports.add(package_name_map.get(imp, imp))

        # Find unused (declared but not imported)
        unused = declared - normalized_imports

        # Find untracked (imported but not declared)
        untracked = normalized_imports - declared

        return sorted(unused), sorted(untracked)

    def validate_all_requirements(self) -> Dict[str, Any]:
        """Validate all requirements files in the project.

        Returns:
            Validation report dictionary
        """
        report = {
            "requirements.txt": {},
            "requirements-dev.txt": {},
            "summary": {},
        }

        # Check requirements.txt
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            unused, untracked = self.check_unused_dependencies(req_file)
            declared = self.parse_requirements_file(req_file)

            report["requirements.txt"] = {
                "declared": declared,
                "unused": unused,
                "untracked": untracked,
            }

        # Check requirements-dev.txt
        dev_req_file = self.project_root / "requirements-dev.txt"
        if dev_req_file.exists():
            unused_dev, untracked_dev = self.check_unused_dependencies(dev_req_file)
            declared_dev = self.parse_requirements_file(dev_req_file)

            report["requirements-dev.txt"] = {
                "declared": declared_dev,
                "unused": unused_dev,
                "untracked": untracked_dev,
            }

        # Scan all imports
        all_imports = self.scan_project_imports()
        report["all_imports"] = {
            pkg: [str(f.relative_to(self.project_root)) for f in files]
            for pkg, files in all_imports.items()
        }

        return report

    def generate_validation_report(self) -> str:
        """Generate a human-readable validation report.

        Returns:
            Formatted report string
        """
        report_data = self.validate_all_requirements()

        lines = [
            "=" * 80,
            "DEPENDENCY VALIDATION REPORT",
            f"Generated: {__import__('datetime').datetime.now().isoformat()}",
            "=" * 80,
            "",
        ]

        # Report for requirements.txt
        if "requirements.txt" in report_data:
            data = report_data["requirements.txt"]
            lines.extend(
                [
                    "requirements.txt",
                    "-" * 80,
                    f"Declared dependencies: {len(data.get('declared', {}))}",
                    f"Unused dependencies: {len(data.get('unused', []))}",
                    f"Untracked imports: {len(data.get('untracked', []))}",
                    "",
                ]
            )

            if data.get("unused"):
                lines.append("Unused (declared but not imported):")
                for pkg in data["unused"]:
                    lines.append(f"  - {pkg}")
                lines.append("")

            if data.get("untracked"):
                lines.append("Untracked (imported but not declared):")
                for pkg in data["untracked"]:
                    lines.append(f"  - {pkg}")
                lines.append("")

        # Report for requirements-dev.txt
        if "requirements-dev.txt" in report_data:
            data = report_data["requirements-dev.txt"]
            lines.extend(
                [
                    "requirements-dev.txt",
                    "-" * 80,
                    f"Declared dev dependencies: {len(data.get('declared', {}))}",
                    f"Unused dev dependencies: {len(data.get('unused', []))}",
                    f"Untracked dev imports: {len(data.get('untracked', []))}",
                    "",
                ]
            )

            if data.get("unused"):
                lines.append("Unused dev dependencies:")
                for pkg in data["unused"]:
                    lines.append(f"  - {pkg}")
                lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)
