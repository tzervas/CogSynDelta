# Dependency Documentation & Validation System

## Overview

This system provides automated dependency validation, documentation ingestion, and RAG-based querying for all project dependencies.

## Installation

```bash
# Install core dependencies (includes LlamaIndex)
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

## Usage

### 1. Validate Dependencies

Check that all imported packages are declared in requirements files:

```bash
# Human-readable report
python scripts/validate_dependencies.py

# JSON output
python scripts/validate_dependencies.py --json

# Save report to file
python scripts/validate_dependencies.py --report validation_report.txt
```

### 2. Ingest Documentation

Fetch and index documentation from PyPI, GitHub, and official docs:

```bash
# Ingest specific package
python scripts/ingest_dependency_docs.py --package torch --version 2.9.1

# Ingest all configured dependencies
python scripts/ingest_dependency_docs.py --all

# Specify custom data directory
python scripts/ingest_dependency_docs.py --all --data-dir ./custom_rag_data
```

### 3. Query Documentation

Use natural language to query indexed documentation:

```bash
# Ask questions
python scripts/query_dependency_docs.py "What CUDA versions does PyTorch 2.9.1 support?"

# Filter by dependency
python scripts/query_dependency_docs.py --dependency torch "How do I enable CUDA?"

# Adjust number of results
python scripts/query_dependency_docs.py --top-k 10 "quantum computing with qiskit"

# List indexed dependencies
python scripts/query_dependency_docs.py --list-deps

# Show statistics
python scripts/query_dependency_docs.py --stats

# Get info about specific dependency
python scripts/query_dependency_docs.py --info torch
```

## Architecture

### Components

1. **DependencyValidator** (`validators.py`)
   - Scans Python files for imports
   - Compares against declared requirements
   - Identifies unused and untracked dependencies
   - Generates validation reports

2. **DocumentIngestionPipeline** (`ingestion.py`)
   - Fetches from PyPI package metadata
   - Downloads GitHub release notes
   - Retrieves official documentation
   - Converts to LlamaIndex Document format

3. **DependencyDocsRAG** (`rag_system.py`)
   - LlamaIndex-based RAG system
   - FAISS vector store for fast search
   - HuggingFace embeddings (bge-small-en-v1.5)
   - Persistent storage and metadata tracking

### Data Flow

```
1. Fetch Documentation
   PyPI API → JSON
   GitHub API → Release Notes
   Web URLs → HTML/Text
   ↓
2. Process & Chunk
   Clean HTML
   Split large documents
   Add metadata (dependency, version, source)
   ↓
3. Embed & Index
   HuggingFace embeddings (384-dim)
   FAISS vector store (L2 distance)
   Persist to disk
   ↓
4. Query & Retrieve
   Natural language query
   Vector similarity search
   Context-aware response generation
```

## Configuration

### Dependency Config

Edit `scripts/ingest_dependency_docs.py` to add new dependencies:

```python
DEPENDENCY_CONFIG = {
    "your-package": {
        "github": ("owner", "repo-name"),
        "doc_urls": [
            ("https://docs.example.com/", "Package Documentation"),
        ],
    },
}
```

### RAG System Config

Configure in code or via parameters:

```python
from cogsyndelta.documentation import DependencyDocsRAG

rag = DependencyDocsRAG(
    persist_dir=Path("./data/dependency_docs"),  # Storage location
    embedding_model="BAAI/bge-small-en-v1.5",   # Embedding model
    chunk_size=512,                               # Token chunk size
    chunk_overlap=50,                             # Chunk overlap
)
```

## Storage

### Directory Structure

```
data/dependency_docs/
├── storage/               # LlamaIndex storage
│   ├── docstore.json     # Document metadata
│   ├── index_store.json  # Index metadata
│   └── vector_store.json # FAISS index
├── metadata.json         # System metadata
└── doc_cache/            # Cached documentation (optional)
```

### Metadata Format

```json
{
  "dependencies": {
    "torch": {
      "pypi": {
        "version": "2.9.1",
        "doc_count": 3,
        "last_updated": "2026-01-18T..."
      },
      "github": {
        "version": "2.9.1",
        "doc_count": 1,
        "last_updated": "2026-01-18T..."
      }
    }
  },
  "total_docs": 50,
  "last_updated": "2026-01-18T..."
}
```

## Programmatic Usage

### Validation

```python
from pathlib import Path
from cogsyndelta.documentation import DependencyValidator

validator = DependencyValidator(Path("."))

# Get all imports
imports = validator.scan_project_imports()

# Check for unused dependencies
unused, untracked = validator.check_unused_dependencies(
    Path("requirements.txt")
)

# Generate report
report = validator.generate_validation_report()
print(report)
```

### Ingestion

```python
from cogsyndelta.documentation import DocumentIngestionPipeline

pipeline = DocumentIngestionPipeline()

# Fetch PyPI docs
docs = pipeline.create_documents_from_pypi("torch", version="2.9.1")

# Fetch GitHub release notes
docs = pipeline.create_documents_from_github_release(
    "pytorch", "pytorch", "v2.9.1", "torch"
)

# Fetch from URL
docs = pipeline.create_documents_from_url(
    "https://pytorch.org/docs/stable/",
    "PyTorch Documentation",
    "torch"
)
```

### RAG Query

```python
from pathlib import Path
from cogsyndelta.documentation import DependencyDocsRAG

rag = DependencyDocsRAG(persist_dir=Path("./data/dependency_docs"))

# Query
response = rag.query(
    query_text="What CUDA versions does PyTorch 2.9.1 support?",
    top_k=5,
    filter_dependency="torch"
)
print(response)

# Get stats
stats = rag.get_stats()
print(f"Total docs: {stats['total_documents']}")

# List dependencies
deps = rag.list_dependencies()
print(f"Indexed: {', '.join(deps)}")
```

## Performance

### Embedding Model

- **Model**: BAAI/bge-small-en-v1.5
- **Dimensions**: 384
- **Speed**: ~100 docs/sec on CPU
- **Quality**: Optimized for English semantic search

### Vector Store

- **Backend**: FAISS (IndexFlatL2)
- **Search**: Exact nearest neighbor (L2 distance)
- **Speed**: Sub-millisecond for 10k vectors
- **Scalability**: Handles 100k+ documents efficiently

### Storage

- **Size**: ~1MB per 100 documents (compressed)
- **Persistence**: JSON + binary FAISS index
- **Load time**: <1 second for 1k documents

## Troubleshooting

### Module Not Found: llama_index

```bash
# Install dependencies
pip install -e .
```

### FAISS Import Error

```bash
# Install FAISS
pip install faiss-cpu

# Or for GPU acceleration
pip install faiss-gpu
```

### Embedding Model Download

First run downloads the embedding model (~100MB):
```
Downloading model: BAAI/bge-small-en-v1.5
```

This is cached locally for future use.

### Rate Limiting

If you hit rate limits when fetching docs:
- Add delays between requests (implemented with `time.sleep()`)
- Use cached documentation when available
- Reduce the number of URLs in doc_urls

## Future Enhancements

- [ ] Support for more vector stores (Chroma, Weaviate)
- [ ] Incremental updates (only fetch changed docs)
- [ ] Multi-language documentation support
- [ ] Integration with CI/CD for automated validation
- [ ] Web UI for dependency exploration
- [ ] Dependency graph visualization
- [ ] Security vulnerability scanning
- [ ] License compatibility checking

## See Also

- [DEPENDENCIES.md](../DEPENDENCIES.md) - Complete dependency documentation
- [GPU_COMPATIBILITY.md](../GPU_COMPATIBILITY.md) - GPU setup guide
- [INSTALL.md](../INSTALL.md) - Installation instructions
- [requirements.txt](../requirements.txt) - Core dependencies
