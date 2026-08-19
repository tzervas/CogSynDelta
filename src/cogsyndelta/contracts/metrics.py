"""Claim vs measured metrics records for STATUS.md and benches."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class MetricsStatus(str, Enum):
    PASS = "pass"
    GAP = "gap"
    UNKNOWN = "unknown"


@dataclass
class MetricsRecord:
    """One measured claim row."""

    name: str
    claim: str
    measured: dict[str, Any]
    status: MetricsStatus
    device: str
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


def write_metrics_json(path: str | Path, records: list[MetricsRecord]) -> None:
    """Write metrics records to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": [r.to_dict() for r in records],
    }
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
