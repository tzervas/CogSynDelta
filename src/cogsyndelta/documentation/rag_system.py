"""LlamaIndex-based RAG system for dependency documentation.

This module implements a retrieval-augmented generation system for querying
documentation about project dependencies using LlamaIndex and FAISS.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import faiss
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


class DependencyDocsRAG:
    """RAG system for dependency documentation retrieval and querying."""

    def __init__(
        self,
        persist_dir: Path | None = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> None:
        """Initialize the RAG system.

        Args:
            persist_dir: Directory to persist the index. Defaults to ./data/dependency_docs
            embedding_model: HuggingFace embedding model to use
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
        """
        self.persist_dir = persist_dir or Path("./data/dependency_docs")
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Configure LlamaIndex settings
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        Settings.num_output = 512
        Settings.context_window = 3900

        self.index: VectorStoreIndex | None = None
        self.metadata_path = self.persist_dir / "metadata.json"
        self.metadata: dict[str, Any] = self._load_metadata()

        # Initialize or load existing index
        self._initialize_index()

    def _load_metadata(self) -> dict[str, Any]:
        """Load metadata about indexed documents."""
        if self.metadata_path.exists():
            with open(self.metadata_path) as f:
                return json.load(f)
        return {"dependencies": {}, "last_updated": None, "total_docs": 0}

    def _save_metadata(self) -> None:
        """Save metadata about indexed documents."""
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def _initialize_index(self) -> None:
        """Initialize or load the vector store index."""
        storage_dir = self.persist_dir / "storage"

        if storage_dir.exists() and (storage_dir / "docstore.json").exists():
            # Load existing index
            try:
                storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
                self.index = load_index_from_storage(storage_context)
                print(f"Loaded existing index with {self.metadata['total_docs']} documents")
            except Exception as e:
                print(f"Error loading index: {e}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self) -> None:
        """Create a new FAISS vector store index."""
        # Create FAISS index (768 dimensions for bge-small-en-v1.5)
        dimension = 384  # bge-small-en-v1.5 embedding dimension
        faiss_index = faiss.IndexFlatL2(dimension)

        # Create vector store
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create empty index
        self.index = VectorStoreIndex([], storage_context=storage_context)
        print("Created new FAISS index")

    def add_documents(
        self,
        documents: list[Document],
        dependency_name: str,
        version: str,
        source: str,
    ) -> None:
        """Add documents to the index.

        Args:
            documents: List of LlamaIndex Document objects
            dependency_name: Name of the dependency (e.g., 'torch', 'numpy')
            version: Version of the dependency
            source: Source of documentation (e.g., 'pypi', 'github', 'official_docs')
        """
        if not documents:
            print(f"No documents to add for {dependency_name}")
            return

        # Add metadata to each document
        for doc in documents:
            doc.metadata.update(
                {
                    "dependency": dependency_name,
                    "version": version,
                    "source": source,
                    "ingestion_date": datetime.now().isoformat(),
                }
            )

        # Insert documents into index
        if self.index is None:
            self._create_new_index()

        for doc in documents:
            self.index.insert(doc)

        # Update metadata
        if dependency_name not in self.metadata["dependencies"]:
            self.metadata["dependencies"][dependency_name] = {}

        self.metadata["dependencies"][dependency_name][source] = {
            "version": version,
            "doc_count": len(documents),
            "last_updated": datetime.now().isoformat(),
        }
        self.metadata["total_docs"] = self.metadata.get("total_docs", 0) + len(documents)

        # Persist index and metadata
        self.persist()
        print(f"Added {len(documents)} documents for {dependency_name} v{version} from {source}")

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_dependency: str | None = None,
    ) -> str:
        """Query the documentation.

        Args:
            query_text: Natural language query
            top_k: Number of relevant documents to retrieve
            filter_dependency: Optional dependency name to filter results

        Returns:
            Generated answer based on retrieved documentation
        """
        if self.index is None:
            return "No documentation has been indexed yet."

        # Create query engine
        query_engine = self.index.as_query_engine(similarity_top_k=top_k)

        # Add dependency filter to query if specified
        if filter_dependency:
            query_text = f"[dependency: {filter_dependency}] {query_text}"

        # Execute query
        response = query_engine.query(query_text)
        return str(response)

    def get_dependency_info(self, dependency_name: str) -> dict[str, Any] | None:
        """Get information about indexed documentation for a dependency.

        Args:
            dependency_name: Name of the dependency

        Returns:
            Dictionary with version, sources, and document counts
        """
        return self.metadata["dependencies"].get(dependency_name)

    def list_dependencies(self) -> list[str]:
        """List all dependencies with indexed documentation.

        Returns:
            List of dependency names
        """
        return list(self.metadata["dependencies"].keys())

    def persist(self) -> None:
        """Persist the index and metadata to disk."""
        if self.index is not None:
            storage_dir = self.persist_dir / "storage"
            storage_dir.mkdir(parents=True, exist_ok=True)
            self.index.storage_context.persist(persist_dir=str(storage_dir))
            self._save_metadata()
            print(f"Persisted index to {storage_dir}")

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the indexed documentation.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_dependencies": len(self.metadata["dependencies"]),
            "total_documents": self.metadata["total_docs"],
            "last_updated": self.metadata["last_updated"],
            "dependencies": self.metadata["dependencies"],
        }
