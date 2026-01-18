#!/usr/bin/env python3
"""Script to query the dependency documentation RAG system.

This script allows querying the indexed dependency documentation using
natural language queries.

Usage:
    python scripts/query_dependency_docs.py "What CUDA versions does PyTorch 2.9.1 support?"
    python scripts/query_dependency_docs.py --dependency torch "How do I enable CUDA?"
    python scripts/query_dependency_docs.py --list-deps
    python scripts/query_dependency_docs.py --stats
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cogsyndelta.documentation import DependencyDocsRAG


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Query dependency documentation RAG system")
    parser.add_argument(
        "query",
        nargs="?",
        help="Natural language query about dependencies",
    )
    parser.add_argument(
        "--dependency",
        "-d",
        help="Filter results to specific dependency",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of relevant documents to retrieve (default: 5)",
    )
    parser.add_argument(
        "--list-deps",
        "-l",
        action="store_true",
        help="List all indexed dependencies",
    )
    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Show RAG system statistics",
    )
    parser.add_argument(
        "--info",
        "-i",
        help="Show information about a specific dependency",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/dependency_docs"),
        help="Directory with RAG data (default: ./data/dependency_docs)",
    )

    args = parser.parse_args()

    # Initialize RAG system
    rag = DependencyDocsRAG(persist_dir=args.data_dir)

    if args.list_deps:
        # List all dependencies
        deps = rag.list_dependencies()
        print(f"Indexed dependencies ({len(deps)}):")
        for dep in sorted(deps):
            print(f"  - {dep}")
        return 0

    if args.stats:
        # Show statistics
        stats = rag.get_stats()
        print("RAG System Statistics:")
        print(f"  Total dependencies: {stats['total_dependencies']}")
        print(f"  Total documents: {stats['total_documents']}")
        print(f"  Last updated: {stats['last_updated']}")
        print("\nDependencies:")
        for dep_name, sources in sorted(stats["dependencies"].items()):
            print(f"\n  {dep_name}:")
            for source, info in sources.items():
                print(f"    - {source}: v{info['version']} ({info['doc_count']} docs)")
        return 0

    if args.info:
        # Show info about specific dependency
        info = rag.get_dependency_info(args.info)
        if info:
            print(f"Dependency: {args.info}")
            for source, source_info in info.items():
                print(f"\n  Source: {source}")
                print(f"    Version: {source_info['version']}")
                print(f"    Documents: {source_info['doc_count']}")
                print(f"    Last updated: {source_info['last_updated']}")
        else:
            print(f"No information found for dependency: {args.info}")
            return 1
        return 0

    if not args.query:
        parser.print_help()
        print("\nPlease provide a query or use --list-deps, --stats, or --info")
        return 1

    # Execute query
    print(f"Query: {args.query}")
    if args.dependency:
        print(f"Filtering by dependency: {args.dependency}")
    print(f"\n{'=' * 80}\n")

    response = rag.query(
        query_text=args.query,
        top_k=args.top_k,
        filter_dependency=args.dependency,
    )

    print(response)
    print(f"\n{'=' * 80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
