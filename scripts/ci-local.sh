#!/usr/bin/env bash
# Run exactly what CI runs, in the same order, with the same pinned versions.
#
# WHY THIS EXISTS
# Three separate CI failures this session came from running the checks I habitually run
# instead of the checks CI actually runs:
#   - `uv run ruff check .` skipped a gitignored directory that CI lints explicitly
#   - `mypy src/` was never invoked locally at all
#   - `scripts/quality_control.py --fail-under 90` is repo-specific and easy to miss
# Each cost a red build and a follow-up commit. One script removes the whole class.
#
# Versions come from .github/workflows/ci.yml rather than being duplicated here, so this
# cannot drift from CI silently -- which would defeat the point.
set -uo pipefail
cd "$(dirname "$0")/.."

RUFF=$(grep -m1 'RUFF_VERSION' .github/workflows/ci.yml | tr -d ' "' | cut -d: -f2)
MYPY=$(grep -m1 'MYPY_VERSION' .github/workflows/ci.yml | tr -d ' "' | cut -d: -f2)
: "${RUFF:?could not read RUFF_VERSION from ci.yml}"
: "${MYPY:?could not read MYPY_VERSION from ci.yml}"

fail=0
step() {
    local name="$1"; shift
    printf '\n=== %s ===\n' "$name"
    if "$@"; then
        printf '  OK\n'
    else
        printf '  FAILED\n'
        fail=1
    fi
}

step "ruff check (pinned $RUFF)" \
    uvx "ruff@${RUFF}" check src/ tests/ benchmarks/ scripts/ examples/
step "ruff format --check (pinned $RUFF)" \
    uvx "ruff@${RUFF}" format --check src/ tests/ benchmarks/ scripts/ examples/
step "mypy (pinned $MYPY)" \
    uvx --from "mypy==${MYPY}" --with types-PyYAML mypy src/
step "quality score (>=90)" \
    python3 scripts/quality_control.py src/ --fail-under 90
step "pytest" \
    uv run pytest -q

printf '\n'
if [ "$fail" -eq 0 ]; then
    printf 'ALL CI CHECKS PASSED\n'
else
    printf 'SOME CI CHECKS FAILED -- do not push\n'
fi
exit "$fail"
