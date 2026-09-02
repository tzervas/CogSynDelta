#!/usr/bin/env bash
# Run the same gates as .github/workflows/ci.yml and poc-ci.yml.
#
# Default: project .venv (Python 3.12, CUDA torch from pyproject cu128).
# --cpu:   CI-identical CPU torch (--no-sources + pytorch.org/whl/cpu).
# --poc:   skip full tests/ (only poc-ci surface + CLI smoke).
# --pre-commit: also run pre-commit --all-files (CI job, not a required check).
#
# Why: ruff S105, quality docstrings, and Python-floor drift landed on GitHub
# before we ran the same commands locally. This script is the pre-push gate.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUFF_VERSION="${RUFF_VERSION:-0.14.13}"
MYPY_VERSION="${MYPY_VERSION:-1.19.1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

CPU_SYNC=0
POC_ONLY=0
RUN_PRECOMMIT=0
for arg in "$@"; do
    case "$arg" in
        --cpu) CPU_SYNC=1 ;;
        --poc) POC_ONLY=1 ;;
        --pre-commit) RUN_PRECOMMIT=1 ;;
        -h|--help)
            echo "usage: $0 [--cpu] [--poc] [--pre-commit]"
            exit 0
            ;;
        *)
            echo "unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not on PATH. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

echo "== python ${PYTHON_VERSION} + .venv =="
uv python install "${PYTHON_VERSION}"
if [[ "${CPU_SYNC}" -eq 1 ]]; then
    echo "sync: CPU torch (CI Test/poc-ci wheels)"
    uv sync --group dev --no-sources \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match
else
    echo "sync: project lock (desktop CUDA torch from pyproject)"
    uv sync --group dev
fi

uv run --no-sync python - <<'PY'
import sys
import torch
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device {torch.cuda.get_device_name(0)}")
PY

fail=0
run() {
    local title="$1"
    shift
    echo
    echo "== ${title} =="
    if "$@"; then
        echo "OK ${title}"
    else
        echo "FAIL ${title}" >&2
        fail=1
    fi
}

# Lint & Format Check (ci.yml) — pin the same tool versions CI installs.
run "ruff check" uvx "ruff@${RUFF_VERSION}" check \
    src/ tests/ benchmarks/ scripts/ examples/
run "ruff format" uvx "ruff@${RUFF_VERSION}" format --check \
    src/ tests/ benchmarks/ scripts/ examples/
# Project venv already has mypy + types-PyYAML (same as CI after uv tool install --with).
# uvx --with is an uvx flag, not a mypy flag.
run "mypy src/" uv run --no-sync mypy src/

# Quality Score Check (ci.yml) — stdlib only, --no-project.
run "quality >=90" uv run --no-project python scripts/quality_control.py src/ --fail-under 90

POC_PYTESTS=(
    tests/test_poc_contracts.py
    tests/test_poc_train.py
    tests/test_poc_compress.py
    tests/test_poc_registry.py
    tests/test_poc_route_train.py
)
if [[ -f tests/test_poc_cuda.py ]]; then
    POC_PYTESTS+=(tests/test_poc_cuda.py)
fi

run "poc pytest" uv run --no-sync pytest "${POC_PYTESTS[@]}" -q --tb=short

SMOKE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/csd-ci-local.XXXXXX")"
trap 'rm -rf "${SMOKE_DIR}"' EXIT
# --stream synthetic mirrors CI exactly: the GitHub runners have no dataset export, so
# the smoke must pass without one. The real-data path is covered by the corpus tests in
# `pytest tests/`, which skip when the mount is absent.
run "poc cli train" uv run --no-sync python -m cogsyndelta.poc.cli train \
    --device cpu --steps 10 --stream synthetic --checkpoint "${SMOKE_DIR}/poc_vae.pt"
run "poc cli compress" uv run --no-sync python -m cogsyndelta.poc.cli compress \
    --device cpu --batch 4
if [[ -f src/cogsyndelta/poc/route.py ]]; then
    run "poc cli route" uv run --no-sync python -m cogsyndelta.poc.cli route \
        --device cpu --batch 8
fi
if [[ -f src/cogsyndelta/poc/train_route.py ]]; then
    run "poc cli train-route" uv run --no-sync python -m cogsyndelta.poc.cli train-route \
        --device cpu --steps 10 --stream synthetic
fi

if [[ "${POC_ONLY}" -eq 0 ]]; then
    run "pytest tests/ (CI Test on Python ${PYTHON_VERSION})" \
        uv run --no-sync pytest tests/ -q --tb=short
fi

if [[ "${RUN_PRECOMMIT}" -eq 1 ]]; then
    run "pre-commit --all-files" uvx pre-commit run --all-files
fi

echo
if [[ "${fail}" -ne 0 ]]; then
    echo "ci_local: FAILED — do not push until the FAIL lines above are clean." >&2
    exit 1
fi
echo "ci_local: all selected gates passed."
