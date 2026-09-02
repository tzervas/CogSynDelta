"""Corpus identity: what a receipt means when it says "this is the data I trained on".

WHY THIS IS ITS OWN TOP-LEVEL MODULE
`scripts/csd-quantize.py` HARD-FAILS on a fingerprint mismatch, because a quantization
number compared against an fp32 number measured on a different split is not a comparison
at all. That makes this the load-bearing check in the whole receipt, so the rule it
applies has to be readable, versioned, and TESTABLE IN CI.

That last word decides where the file lives. `cogsyndelta.regions.__init__` imports
`regions/pretrain.py`, which imports `tokenizers` at module scope, so importing anything
at all from that package needs the `train` dependency group -- which CI deliberately does
not install (see pyproject: "CI never trains"). A rule this load-bearing must not be one
of the eight test files that silently skip there, so it sits at the top level, stdlib
only.

WHAT WENT WRONG BEFORE, and what the version tag is for
The previous rule hashed the shard names and sizes of the PRIMARY source only. `retrieve`
draws from three sources, and its primary (FiQA, 14,131 pairs) is about 3% of the pairs
it trains on -- so a fingerprint that "verified the corpus" was blind to gooaq's 400,000
rows and natural-questions' 100,231. Swapping either of those out entirely would have
produced a matching fingerprint and a silent, meaningless comparison.

Including every source changes the value for every existing receipt, which is exactly the
situation where a bare mismatch is the wrong error: it says "your corpus drifted" when
what changed is the arithmetic. Hence `CORPUS_FINGERPRINT_SCHEME`, recorded next to the
fingerprint, and :func:`verify_corpus_fingerprint`, which distinguishes the two cases.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

CORPUS_FINGERPRINT_SCHEME = "csd-corpus-fp/v2"
"""Names the rule that produced a fingerprint, so a rule change is legible as one.

v2 hashes EVERY source (primary plus every `extra_sources` entry) with its columns and
its cap; v1 -- receipts written before this, which carry no scheme field at all -- hashed
the primary source's shard names and sizes and nothing else. Bump this whenever the
payload changes, or the next change turns into a fleet of confusing mismatches.
"""


def _sources(
    shards: list[str],
    columns: tuple[str, str] | list[str] | None,
    extra_sources: list[dict] | None,
) -> list[tuple[list[str], list[str], int]]:
    """Flatten a region's corpus into `(shards, columns, cap)` per source, primary first."""
    flattened = [(list(shards), list(columns or []), 0)]
    for source in extra_sources or []:
        flattened.append(
            (
                list(source.get("shards", [])),
                list(source.get("columns", [])),
                int(source.get("limit") or 0),
            )
        )
    return flattened


def fingerprint_corpus(
    shards: list[str],
    *,
    columns: tuple[str, str] | list[str] | None = None,
    extra_sources: list[dict] | None = None,
) -> str:
    """Hash everything that decides which rows reach `build_splits`.

    Per source that is: the shard basenames and byte sizes (the data itself), the columns
    read from them, and the cap applied. All three change the resulting split, and a
    receipt that cannot detect a change in any of them is not identifying its corpus.

    Sizes rather than content hashes: a full content hash of 1,654 parquet shards costs
    minutes on every run to detect a case -- same name, same size, different bytes -- that
    a corpus fetch does not produce. Shard lists here are sorted, so shard ORDER does not
    change the value; ordering is a `build_splits` concern, not an identity one.

    Args:
        shards: The primary source's parquet paths.
        columns: The primary source's `(left, right)` column pair, if known.
        extra_sources: `{"shards": [...], "columns": [...], "limit": n}` entries, in the
            order the region declares them.

    Returns:
        A hex digest identifying the corpus under :data:`CORPUS_FINGERPRINT_SCHEME`.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(CORPUS_FINGERPRINT_SCHEME.encode())
    for source_shards, source_columns, cap in _sources(shards, columns, extra_sources):
        h.update(b"\x00source\x00")
        for path in sorted(source_shards):
            p = Path(path)
            h.update(p.name.encode())
            h.update(b"\x00")
            h.update(str(p.stat().st_size if p.exists() else 0).encode())
            h.update(b"\x00")
        h.update("|".join(source_columns).encode())
        h.update(b"\x00")
        h.update(str(cap).encode())
    return h.hexdigest()


def verify_corpus_fingerprint(corpus: dict[str, Any], rebuilt: str, region: str) -> None:
    """Refuse to proceed unless a rebuilt corpus is the one a receipt describes.

    Separates the two ways this can fail, because they call for opposite responses:

    - **The scheme changed.** The receipt's fingerprint was produced by a different rule,
      so the two numbers were never comparable and the difference says nothing about the
      data. Re-run training to write a receipt under the current scheme.
    - **The fingerprint differs under the same scheme.** The corpus really did drift.
      Every comparison downstream of it is void.

    Reporting the first as the second is how an operator ends up hunting a corpus change
    that never happened.

    Args:
        corpus: The receipt's `corpus` block.
        rebuilt: The fingerprint computed from what is on disk now.
        region: Region name, for the message.

    Raises:
        RuntimeError: On either failure, naming which one it is.
    """
    recorded = corpus.get("fingerprint")
    if not recorded:
        return
    scheme = corpus.get("fingerprint_scheme") or "v1 (unversioned, primary source only)"
    if scheme != CORPUS_FINGERPRINT_SCHEME:
        raise RuntimeError(
            f"{region}: corpus fingerprint SCHEME changed, {scheme} -> "
            f"{CORPUS_FINGERPRINT_SCHEME}. The recorded fingerprint was computed by a "
            f"different rule (v1 hashed the primary source only -- about 3% of the pairs "
            f"for a three-source region like `retrieve`), so it is not comparable with "
            f"the rebuilt one and the difference is NOT evidence that the corpus drifted. "
            f"Re-run training to write a receipt under the current scheme."
        )
    if rebuilt != recorded:
        raise RuntimeError(
            f"corpus fingerprint mismatch for {region}: rebuilt {rebuilt} vs receipt "
            f"{recorded}. The held-out split would differ from the one training was judged "
            f"on, making any quantization comparison meaningless."
        )
