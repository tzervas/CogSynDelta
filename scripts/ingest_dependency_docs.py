#!/usr/bin/env python3
"""Script to ingest documentation for all project dependencies into the RAG system.

This script fetches documentation from PyPI, GitHub releases, and official docs
for all dependencies and stores them in the LlamaIndex RAG system.

Usage:
    python scripts/ingest_dependency_docs.py [--package PACKAGE] [--all]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cogsyndelta.documentation import DependencyDocsRAG, DocumentIngestionPipeline

# Dependency configuration with GitHub repos and doc URLs
DEPENDENCY_CONFIG = {
    "torch": {
        "github": ("pytorch", "pytorch"),
        "doc_urls": [
            ("https://pytorch.org/docs/stable/", "PyTorch Official Documentation"),
        ],
    },
    "torchvision": {
        "github": ("pytorch", "vision"),
        "doc_urls": [],
    },
    "numpy": {
        "github": ("numpy", "numpy"),
        "doc_urls": [
            ("https://numpy.org/doc/stable/", "NumPy Official Documentation"),
        ],
    },
    "fastapi": {
        "github": ("fastapi", "fastapi"),
        "doc_urls": [
            ("https://fastapi.tiangolo.com/", "FastAPI Official Documentation"),
        ],
    },
    "uvicorn": {
        "github": ("encode", "uvicorn"),
        "doc_urls": [],
    },
    "pydantic": {
        "github": ("pydantic", "pydantic"),
        "doc_urls": [
            ("https://docs.pydantic.dev/latest/", "Pydantic Official Documentation"),
        ],
    },
    "qiskit": {
        "github": ("Qiskit", "qiskit"),
        "doc_urls": [
            ("https://docs.quantum.ibm.com/", "Qiskit Documentation"),
        ],
    },
    "pennylane": {
        "github": ("PennyLaneAI", "pennylane"),
        "doc_urls": [
            ("https://docs.pennylane.ai/", "PennyLane Documentation"),
        ],
    },
    "pytest": {
        "github": ("pytest-dev", "pytest"),
        "doc_urls": [
            ("https://docs.pytest.org/", "Pytest Documentation"),
        ],
    },
    "black": {
        "github": ("psf", "black"),
        "doc_urls": [],
    },
    "ruff": {
        "github": ("astral-sh", "ruff"),
        "doc_urls": [
            ("https://docs.astral.sh/ruff/", "Ruff Documentation"),
        ],
    },
    "mypy": {
        "github": ("python", "mypy"),
        "doc_urls": [
            ("https://mypy.readthedocs.io/", "Mypy Documentation"),
        ],
    },
    "sphinx": {
        "github": ("sphinx-doc", "sphinx"),
        "doc_urls": [
            ("https://www.sphinx-doc.org/", "Sphinx Documentation"),
        ],
    },
    "llama-index": {
        "github": ("run-llama", "llama_index"),
        "doc_urls": [
            ("https://docs.llamaindex.ai/", "LlamaIndex Documentation"),
        ],
    },
}


def ingest_package(
    pipeline: DocumentIngestionPipeline,
    rag: DependencyDocsRAG,
    package_name: str,
    version: str = None,
) -> None:
    """Ingest documentation for a single package.

    Args:
        pipeline: Document ingestion pipeline
        rag: RAG system instance
        package_name: Name of the package
        version: Specific version or None for latest
    """
    print(f"\n{'=' * 80}")
    print(f"Ingesting documentation for {package_name}")
    print(f"{'=' * 80}")

    config = DEPENDENCY_CONFIG.get(package_name, {})
    github_repo = config.get("github")
    doc_urls = config.get("doc_urls", [])

    # Ingest all documentation sources
    docs_by_source = pipeline.ingest_dependency_docs(
        package_name=package_name,
        version=version,
        github_repo=github_repo,
        doc_urls=doc_urls,
    )

    # Add to RAG system
    for source, documents in docs_by_source.items():
        # Determine version from PyPI docs
        actual_version = version or "latest"
        if source == "pypi" and documents:
            # Try to extract version from metadata
            for doc in documents:
                if "version" in doc.text.lower():
                    import re

                    match = re.search(r"Version:\s*(\S+)", doc.text)
                    if match:
                        actual_version = match.group(1)
                        break

        rag.add_documents(
            documents=documents,
            dependency_name=package_name,
            version=actual_version,
            source=source,
        )

    print(f"✓ Completed ingestion for {package_name}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ingest dependency documentation into RAG system")
    parser.add_argument("--package", "-p", help="Specific package to ingest (e.g., 'torch')")
    parser.add_argument("--version", "-v", help="Specific version of the package (default: latest)")
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Ingest all configured dependencies",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=Path,
        default=Path("./data/dependency_docs"),
        help="Directory to store RAG data (default: ./data/dependency_docs)",
    )

    args = parser.parse_args()

    # Initialize systems
    print("Initializing RAG system and ingestion pipeline...")
    rag = DependencyDocsRAG(persist_dir=args.data_dir)
    pipeline = DocumentIngestionPipeline()

    if args.package:
        # Ingest specific package
        ingest_package(pipeline, rag, args.package, args.version)
    elif args.all:
        # Ingest all configured packages
        for package_name in DEPENDENCY_CONFIG:
            try:
                ingest_package(pipeline, rag, package_name)
            except Exception as e:
                print(f"✗ Error ingesting {package_name}: {e}")
                continue
    else:
        parser.print_help()
        print("\nPlease specify --package PACKAGE or --all")
        return 1

    # Print final statistics
    print(f"\n{'=' * 80}")
    print("INGESTION COMPLETE")
    print(f"{'=' * 80}")
    stats = rag.get_stats()
    print(f"Total dependencies indexed: {stats['total_dependencies']}")
    print(f"Total documents indexed: {stats['total_documents']}")
    print(f"Last updated: {stats['last_updated']}")
    print("\nDependencies with documentation:")
    for dep_name in sorted(stats["dependencies"].keys()):
        print(f"  - {dep_name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
