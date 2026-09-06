"""Row W0: the `Faculty` protocol, its adapters, and the per-region parameter table.

See `cogsyndelta.faculty.protocol` for the typed contract (DEC-14, DEC-15, DEC-47),
`cogsyndelta.faculty.adapters` for the `TextEncoder`/`IJEPA` wrappers that satisfy it,
and `cogsyndelta.faculty.param_table` for the embedding/token-path/pooling-head
breakdown of the five regions with a real matrix checkpoint.

Deliberately excluded from this package (later lanes, not W0): a controller, a
workspace, an episodic store.
"""

from __future__ import annotations

from cogsyndelta.faculty.adapters import (
    TextFacultyAdapter,
    VisualFacultyAdapter,
    kv_bytes_per_token,
)
from cogsyndelta.faculty.protocol import Faculty, assert_latent_tokens, token_mask

__all__ = [
    "Faculty",
    "TextFacultyAdapter",
    "VisualFacultyAdapter",
    "assert_latent_tokens",
    "kv_bytes_per_token",
    "token_mask",
]
