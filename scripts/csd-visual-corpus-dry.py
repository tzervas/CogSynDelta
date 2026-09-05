#!/usr/bin/env python3
"""CPU dry-run of visual-clean-v1: per-source PNG counts + corpus fingerprint.

Does not train, does not extract zips, does not decode pixels. Opens zip central
directories (or walks PNG trees) and hashes train artefact basename+size under
csd-corpus-fp/v2. Claude can run this against the real Mix B manifest.

    uv run python scripts/csd-visual-corpus-dry.py
    uv run python scripts/csd-visual-corpus-dry.py --manifest path/to/visual-clean-v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cogsyndelta.vl.mix_corpus import MixCorpusError, dry_run, load_manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        default=None,
        help="visual-clean-v1 JSON (default: config/mind/visual-clean-v1.json)",
    )
    args = ap.parse_args()
    try:
        manifest = load_manifest(args.manifest)
        result = dry_run(manifest)
    except MixCorpusError as exc:
        print(f"visual-clean-v1 dry-run FAILED: {exc}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if result["listed_train"] != result["declared_train"]:
        print(
            f"WARN listed_train {result['listed_train']} != declared {result['declared_train']}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
