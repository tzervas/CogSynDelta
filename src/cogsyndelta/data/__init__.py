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

__all__ = [
    "CORPORA",
    "TOKENIZERS",
    "corpus_root",
    "iter_batches",
    "iter_documents",
    "iter_token_windows",
    "load_tokenizer",
    "shard_paths",
]
