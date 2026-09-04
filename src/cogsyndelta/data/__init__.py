"""Corpus access for real training runs."""

from cogsyndelta.data.corpus import (
    CORPORA,
    TOKENIZERS,
    corpus_root,
    iter_batches,
    iter_documents,
    iter_token_windows,
    load_tokenizer,
    shard_paths,
)
from cogsyndelta.data.stream import (
    DEFAULT_DATASETS_ROOT,
    TEXT_SOURCES,
    CorpusStream,
    StreamSource,
    SyntheticStream,
    corpus_available,
    resolve_stream,
)

__all__ = [
    "CORPORA",
    "DEFAULT_DATASETS_ROOT",
    "TEXT_SOURCES",
    "TOKENIZERS",
    "CorpusStream",
    "StreamSource",
    "SyntheticStream",
    "corpus_available",
    "corpus_root",
    "iter_batches",
    "iter_documents",
    "iter_token_windows",
    "load_tokenizer",
    "resolve_stream",
    "shard_paths",
]
