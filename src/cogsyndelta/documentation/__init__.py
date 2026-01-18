"""Documentation RAG system for CogSynDelta dependency management.

This module provides a LlamaIndex-based RAG system for ingesting, storing,
and querying documentation from all project dependencies.
"""

try:
    from .rag_system import DependencyDocsRAG
    from .ingestion import DocumentIngestionPipeline
    from .validators import DependencyValidator
    __all__ = ["DependencyDocsRAG", "DocumentIngestionPipeline", "DependencyValidator"]
except ImportError:
    # Handle missing dependencies gracefully
    __all__ = []
