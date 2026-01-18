"""Documentation RAG system for CogSynDelta dependency management.

This module provides a LlamaIndex-based RAG system for ingesting, storing,
and querying documentation from all project dependencies.
"""

try:
    from .ingestion import DocumentIngestionPipeline
    from .rag_system import DependencyDocsRAG
    from .validators import DependencyValidator

    __all__ = ["DependencyDocsRAG", "DependencyValidator", "DocumentIngestionPipeline"]
except ImportError:
    # Handle missing dependencies gracefully
    __all__ = []
