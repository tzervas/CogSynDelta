#!/usr/bin/env bash
# Run every pre-commit gate (pre-commit-hooks, ruff, mypy, shellcheck, yamllint,
# pydocstyle, conventional-commit) over the whole repo and fail on any finding.
#
# This is the single command CI and the pre-push hook both defer to (see
# .github/workflows/ci.yml and scripts/ci_local.sh --pre-commit) so local and CI
# results never drift.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not on PATH. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

uv run pre-commit run --all-files
