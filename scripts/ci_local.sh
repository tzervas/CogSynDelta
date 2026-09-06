#!/usr/bin/env bash
# Run the same gates as .github/workflows/ci.yml and poc-ci.yml.
#
# Default: OWN venv at .venv-ci (Python 3.12, CUDA torch from pyproject cu128); never
#          the shared project .venv, and NOT overridable via UV_PROJECT_ENVIRONMENT --
#          see the isolation comment below for why that has to be unconditional.
# --cpu:           CI-identical CPU torch (--no-sources + pytorch.org/whl/cpu).
# --poc:           skip full tests/ (only poc-ci surface + CLI smoke).
# --no-pre-commit: skip the lint gate below (default: it runs).
#
# Why: ruff S105, quality docstrings, and Python-floor drift landed on GitHub
# before we ran the same commands locally. This script is the pre-push gate.
#
# 2026-09-04: the pytest / poc pytest / poc cli steps below now run with
# CUDA_VISIBLE_DEVICES="" -- CPU-only, always -- regardless of which sync branch ran.
# The PoC route-train test previously diverged between the tracked pre-push hook and CI:
# the hook runs this script on whatever machine pushed, and on a GPU box the default
# sync branch ("desktop CUDA torch from pyproject") installs cu128 wheels, so
# torch.cuda.is_available() is true there and the test exercises CUDA kernels; CI's
# runner has no GPU, so the identical test exercises CPU kernels. Same test, same
# command, two different code paths. The torch *sync* is left exactly as-is (CUDA torch
# still gets installed either way, matching what CI's own sync does); only test
# *execution* is pinned CPU-only, by hiding the device rather than by not installing
# CUDA support for it. tests/test_ci_local_cpu_gate.py makes this observable.
#
# 2026-09-04: the lint gate (`bash scripts/lint.sh`) now runs BY DEFAULT instead of
# behind an opt-in `--pre-commit` flag. PR #33 (CI task 7597) landed two pydocstyle
# D209 violations that the tracked `.githooks/pre-push` hook did not catch: it calls
# this script with no flags, which used to mean the lint gate never ran locally at
# all -- silently narrower than CI's "Lint gate (scripts/lint.sh)" job, which
# .github/workflows/ci.yml lists as a REQUIRED merge context. A required CI check
# must not be able to pass locally while failing on the server. `--no-pre-commit`
# stays for a deliberate, fast WIP loop; like `.githooks/pre-push`'s own
# `--no-verify`, it is documented so it stays visible, not to encourage routine use.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUFF_VERSION="${RUFF_VERSION:-0.14.13}"
MYPY_VERSION="${MYPY_VERSION:-1.19.1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

CPU_SYNC=0
POC_ONLY=0
RUN_PRECOMMIT=1
for arg in "$@"; do
    case "$arg" in
        --cpu) CPU_SYNC=1 ;;
        --poc) POC_ONLY=1 ;;
        --no-pre-commit) RUN_PRECOMMIT=0 ;;
        -h|--help)
            echo "usage: $0 [--cpu] [--poc] [--no-pre-commit]"
            exit 0
            ;;
        *)
            echo "unknown arg: $arg" >&2
            exit 2
            ;;
    esac
done

# --- isolation: fail fast on a near-full temp filesystem -----------------------------
# HAZARD (observed 2026-09-06 05:28Z, continued): once the shared /tmp actually filled to
# 0 bytes free, the failure did not stop at pytest's basetemp race above -- `torch.save`
# inside the "poc cli train" step raised `RuntimeError: basic_ios::clear: iostream error`,
# an opaque C++ streambuf error with no mention of disk space. Catch the low-space
# condition here, before minutes are spent on venv sync and lint, with a message that
# names the actual cause.
#
# Resolved once, up front, and reused by every mktemp call below (PYTEST_BASETEMP,
# SMOKE_DIR) so the filesystem checked here is exactly the filesystem those directories
# land on -- not merely "whatever TMPDIR happened to be at check time".
CI_LOCAL_TMPROOT="${TMPDIR:-/tmp}"
mkdir -p "${CI_LOCAL_TMPROOT}"
echo "== temp root: ${CI_LOCAL_TMPROOT} =="
CI_LOCAL_TMP_AVAIL_KB="$(df --output=avail -k "${CI_LOCAL_TMPROOT}" | tail -n1 | tr -d '[:space:]')"
CI_LOCAL_TMP_MIN_KB=$((2 * 1024 * 1024)) # 2 GiB
if [[ "${CI_LOCAL_TMP_AVAIL_KB}" -lt "${CI_LOCAL_TMP_MIN_KB}" ]]; then
    echo "ci_local: REFUSING to run -- ${CI_LOCAL_TMPROOT} has $((CI_LOCAL_TMP_AVAIL_KB / 1024)) MiB free, need >= 2048 MiB." >&2
    echo "  Set TMPDIR to a filesystem with more room, or free space on ${CI_LOCAL_TMPROOT}, then retry." >&2
    exit 1
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not on PATH. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# --- isolation: never touch the interactive/training .venv --------------------------
# HAZARD (confirmed 2026-09-02): `uv sync` (without --inexact) PRUNES every package
# outside the requested groups. This project's `train` group holds pyarrow + tokenizers,
# which scripts/csd-train-all.py imports lazily and via subprocess from the shared
# .venv. Running this script's default `uv sync --group dev` against that same shared
# .venv while a training run is live would silently strip both packages out from under
# it -- silently, because an already-imported module keeps working until the next lazy
# import or subprocess, so the failure surfaces later looking like a training bug
# rather than an environment one.
#
# Options considered:
#   A. `uv sync --group dev --inexact`  -- stops the pruning, minimal diff. Rejected:
#      the venv can then accumulate packages this gate never asked for, so it drifts
#      from a clean install and can hide a missing-dependency bug real CI would catch.
#   B. Add `--group train` to the sync   -- keeps everything, minimal diff. This was
#      rejected when this script still synced the shared interactive .venv, on the
#      grounds that pyproject.toml documents the train/dev split as deliberate
#      ("pulling pyarrow + tokenizers into every lint job costs minutes for nothing")
#      and that this would reintroduce that cost on every push, forever. That objection
#      does not survive C below: once this script has its own isolated venv, the
#      pyproject comment no longer applies to it -- the comment is about CI's lint jobs
#      (lint, shell-yaml-lint, lint-gate, quality), none of which sync this project's
#      dependency groups at all, and never was about this gate's venv. ADOPTED, folded
#      into C: this gate's pytest steps mirror CI's `test` job, which already syncs
#      `--group dev --group train` every run (.github/workflows/ci.yml:191), so this
#      venv pays the same, correct cost.
#   C. Give this script its OWN project environment (CHOSEN) -- `uv sync --group dev
#      --group train`, matching CI's `test` job (ci.yml:191), keeps pruning to exactly
#      that set, but prunes a venv nothing else reads from, so the HAZARD above is
#      structurally impossible rather than merely avoided. Cost is a second venv on
#      disk and a slower first run, which is cheap next to "silently broke a
#      multi-hour training run."
# C preserves CI fidelity (A doesn't) and is what makes B safe rather than costly: an
# isolated venv can sync CI's exact group set, train included, without the shared-venv
# mutation hazard the HAZARD paragraph above describes.
#
# IMPORTANT: this must be an UNCONDITIONAL override, not `${UV_PROJECT_ENVIRONMENT:-...}`.
# Confirmed on this host: ~/.config/nushell/env.nu sets `$env.UV_PROJECT_ENVIRONMENT =
# ".venv"` globally (nushell is the default shell here; bash subshells inherit it), so
# UV_PROJECT_ENVIRONMENT is ALREADY SET to the shared venv on every normal invocation --
# a `:-` fallback would never fire and this whole fix would silently do nothing. There
# is no way to tell that ambient convention apart from a deliberate one-off override, so
# this script does not try; it always wins.
CI_VENV="${ROOT}/.venv-ci"
export UV_PROJECT_ENVIRONMENT="${CI_VENV}"

# --- isolation: never touch the tracked uv.lock either -------------------------------
# HAZARD (measured 2026-09-03): CI_VENV above isolates the *venv*, but `uv sync`
# (without --frozen/--locked) also RE-RESOLVES and rewrites `uv.lock` whenever its
# resolution differs from what's on disk -- and it always reads/writes
# `<project-root>/uv.lock` next to whichever pyproject.toml it discovers, a completely
# separate concern from which venv UV_PROJECT_ENVIRONMENT points the *install* at. The
# --cpu branch below passes --no-sources, which drops the [tool.uv.sources] cu128 pin so
# the resolver picks CPU torch/torchvision wheels instead of the GPU ones already
# locked -- a real, expected difference, not a resolver bug -- and with ROOT as the
# project root that rewrite landed in the TRACKED uv.lock every run (measured 195
# insertions / 394 deletions).
#
# Why not `--frozen`/`--locked` against the real uv.lock instead of moving the project
# root: poc-ci.yml's header already covers this -- the tracked lock pins CUDA torch, so
# --frozen would install CUDA wheels while claiming to run the CPU gate. That skips
# resolution entirely rather than skipping only the WRITE, so it isn't an option here;
# the --cpu run genuinely needs to re-resolve.
#
# Chosen fix: give the resolve/lock step its own disposable project directory, the same
# shape of fix as CI_VENV above (an isolated *target* the hazardous operation cannot
# escape), not a save-and-restore around a write this script still performs. `uv sync
# --project <dir>` reads and writes `<dir>/uv.lock`, never ROOT's, regardless of what
# UV_PROJECT_ENVIRONMENT points the resulting venv at. Each run, ci_lock_sync():
#   - wipes and recreates CI_LOCK_PROJECT from a plain COPY -- not a symlink -- of
#     pyproject.toml, uv.lock, README.md, .python-version and src/ (the full set a
#     resolve+build needs; src/ is a build input, not the venv target, so this is not
#     the shared-venv hazard above). A symlinked src/ would let the build backend's
#     stray writes (e.g. egg-info) land back in the real tree; ~1.4 MB, the copy is
#     sub-second, so there is no real cost to copying instead.
#   - seeds CI_LOCK_PROJECT/uv.lock from the CURRENT tracked uv.lock, so the resolver
#     starts from the real pin set (fast, minimal-diff resolve) instead of solving cold
#     every run
#   - runs `uv sync --project CI_LOCK_PROJECT ...`; UV_PROJECT_ENVIRONMENT is unchanged,
#     so the venv this installs into is still the same isolated .venv-ci
# This makes writing the tracked lock structurally impossible -- no code path here opens
# ROOT/uv.lock for writing -- rather than merely restoring it afterward. The default
# (non --cpu) sync below goes through the same helper even though it is not the measured
# offender, so both modes share one guarantee instead of one being fixed and the other
# merely assumed safe.
CI_LOCK_PROJECT="${ROOT}/.venv-ci-project"

ci_lock_sync() {
    rm -rf "${CI_LOCK_PROJECT:?}"
    mkdir -p "${CI_LOCK_PROJECT}/src"
    cp "${ROOT}/pyproject.toml" "${ROOT}/uv.lock" "${ROOT}/README.md" "${CI_LOCK_PROJECT}/"
    cp "${ROOT}/.python-version" "${CI_LOCK_PROJECT}/.python-version"
    cp -r "${ROOT}/src/." "${CI_LOCK_PROJECT}/src/"
    uv sync --project "${CI_LOCK_PROJECT}" "$@"
}

# --- guard: belt-and-braces, independent of the isolation above ---------------------
# The isolation above should make this dead code. It stays in case a future edit to
# this script drops the unconditional export above (e.g. "helpfully" changes it back to
# `${UV_PROJECT_ENVIRONMENT:-...}`, reintroducing the exact bug the comment above warns
# about) or otherwise lets the target venv drift back to the shared one. So check for
# that specific collision (this run's venv == the shared venv) rather than refusing
# outright whenever training happens to be running, which would make every push during
# a long training job fail for no real reason once the isolation above is doing its job.
if command -v pgrep >/dev/null 2>&1; then
    TRAIN_PIDS="$(pgrep -f 'csd-train-all\.py' || true)"
    if [[ -n "${TRAIN_PIDS}" ]]; then
        SHARED_VENV="$(realpath -m "${ROOT}/.venv")"
        THIS_VENV="$(realpath -m "${CI_VENV}")"
        echo "ci_local: training process detected (pid(s): ${TRAIN_PIDS//$'\n'/, })" >&2
        if [[ "${THIS_VENV}" == "${SHARED_VENV}" ]]; then
            echo "ci_local: REFUSING to sync." >&2
            echo "  UV_PROJECT_ENVIRONMENT resolves to ${SHARED_VENV}, the same venv" >&2
            echo "  the training run above is using. 'uv sync' here would prune" >&2
            echo "  train-only packages (pyarrow, tokenizers) out of it while it is" >&2
            echo "  in use." >&2
            echo "  To proceed: unset UV_PROJECT_ENVIRONMENT so this script uses its" >&2
            echo "  own isolated ${ROOT}/.venv-ci, or wait for training to finish, or" >&2
            echo "  'git push --no-verify' if this is a deliberate WIP push." >&2
            exit 1
        fi
        echo "ci_local: using isolated venv ${THIS_VENV} (not the training venv) -- safe to proceed." >&2
    fi
fi

echo "== python ${PYTHON_VERSION} + venv (${UV_PROJECT_ENVIRONMENT}) =="
uv python install "${PYTHON_VERSION}"
if [[ "${CPU_SYNC}" -eq 1 ]]; then
    echo "sync: CPU torch (CI Test/poc-ci wheels)"
    # Groups match CI's `test` job sync line: .github/workflows/ci.yml:191.
    ci_lock_sync --group dev --group train --no-sources \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match
else
    echo "sync: project lock (desktop CUDA torch from pyproject)"
    # Same groups as the --cpu branch above: the pytest steps below run either way and
    # need train's tokenizers/pyarrow (ci.yml:191) to collect, not just import, cleanly.
    ci_lock_sync --group dev --group train
fi

uv run --no-sync python - <<'PY'
import sys
import torch
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device {torch.cuda.get_device_name(0)}")
PY

# --- CPU-only test execution, matching the GPU-less CI runner ------------------------
# See the header comment (2026-09-04) for why. This does not undo the sync above -- CUDA
# torch is still installed -- it only hides the device from every step that runs after
# this line (pytest, the poc pytest set, and the poc cli smoke commands below).
export CUDA_VISIBLE_DEVICES=""
echo "== CUDA_VISIBLE_DEVICES=\"\" for test execution (CI runner has no GPU) =="

# --- isolation: private pytest basetemp per ci_local run ------------------------------
# HAZARD (observed 2026-09-06 05:28Z): pytest's default basetemp is a numbered directory
# shared by uid, `${TMPDIR:-/tmp}/pytest-of-<user>/pytest-NN`. Five concurrent
# invocations of this script (several agent worktrees plus a pre-push hook, all the same
# user) raced to create the next numbered dir there; pytest gave up after 10 tries
# (`OSError: could not create numbered dir with prefix test_... after 10 tries`), failing
# the full-suite step with an error that has nothing to do with the code under test -- a
# docs-only push was blocked by an unrelated race between unrelated worktrees.
#
# Fix: give every pytest invocation in this script its own private basetemp, created
# fresh with `mktemp -d` and removed on exit, so two concurrent runs of this script have
# nothing to race over -- same shape of fix as the CI_VENV / CI_LOCK_PROJECT isolation
# above (an isolated target the hazard cannot reach), not a retry/lock around the shared
# default.
#
# PYTEST_ADDOPTS carries it (rather than a --basetemp arg on each call site) so one
# directory covers every pytest invocation below -- poc pytest and the full tests/ run --
# without editing each call, and any basetemp/opts a caller already exported are kept:
# appended to, never clobbered, so a deliberate override still applies.
CI_LOCAL_TMP_DIRS=()
ci_local_cleanup() {
    local dir
    for dir in "${CI_LOCAL_TMP_DIRS[@]:-}"; do
        [[ -n "${dir}" ]] && rm -rf "${dir}"
    done
}
trap ci_local_cleanup EXIT

PYTEST_BASETEMP="$(mktemp -d "${CI_LOCAL_TMPROOT}/csd-ci-local-pytest.XXXXXX")"
CI_LOCAL_TMP_DIRS+=("${PYTEST_BASETEMP}")
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:+${PYTEST_ADDOPTS} }--basetemp=${PYTEST_BASETEMP}"
echo "== pytest basetemp: ${PYTEST_BASETEMP} (private to this run) =="

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
#
# `scripts/csd-benchmark.py` is named EXPLICITLY even though [tool.mypy] excludes
# `scripts/` (L1). mypy's `exclude` filters directory crawling, not paths given on the
# command line -- verified by planting a deliberate `return "not an int"` in this file
# and watching `mypy --config-file pyproject.toml scripts/csd-benchmark.py` report it --
# so this really does typecheck. The rest of the scripts tree stays excluded on purpose
# (shell wrappers, one-off tools, no annotations); this one file writes the receipts the
# matrix pipeline binds and publishes on, so it is worth the same gate `src/` gets.
# Adding it caught a real mismatch on the first run: `Receipt.artifacts` was annotated
# `dict[str, str]` while both eval paths wrote nested `source_*_receipt` records.
run "mypy src/ + benchmark script" uv run --no-sync mypy src/ scripts/csd-benchmark.py

# Quality Score Check (ci.yml) — stdlib only, --no-project.
run "quality >=90" uv run --no-project python scripts/quality_control.py src/ --fail-under 90

POC_PYTESTS=(
    tests/test_poc_contracts.py
    tests/test_poc_train.py
    tests/test_poc_compress.py
    tests/test_poc_registry.py
    tests/test_poc_route_train.py
    tests/test_ci_local_cpu_gate.py
)
if [[ -f tests/test_poc_cuda.py ]]; then
    POC_PYTESTS+=(tests/test_poc_cuda.py)
fi

run "poc pytest" uv run --no-sync pytest "${POC_PYTESTS[@]}" -q --tb=short

SMOKE_DIR="$(mktemp -d "${CI_LOCAL_TMPROOT}/csd-ci-local.XXXXXX")"
CI_LOCAL_TMP_DIRS+=("${SMOKE_DIR}")
# --stream synthetic mirrors CI exactly: the GitHub runners have no dataset export, so
# the smoke must pass without one. The real-data path is covered by the corpus tests in
# `pytest tests/`, which skip when the mount is absent.
run "poc cli train" uv run --no-sync python -m cogsyndelta.poc.cli train \
    --device cpu --steps 10 --stream synthetic --checkpoint "${SMOKE_DIR}/poc_vae.pt"
run "poc cli compress" uv run --no-sync python -m cogsyndelta.poc.cli compress \
    --device cpu --batch 4 --stream synthetic
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
    # Exactly CI's "Lint gate (scripts/lint.sh)" job (.github/workflows/ci.yml), a
    # required merge context -- not a reimplementation of it, so it cannot drift from
    # what that job actually runs.
    run "lint gate (scripts/lint.sh)" bash scripts/lint.sh
fi

echo
if [[ "${fail}" -ne 0 ]]; then
    echo "ci_local: FAILED — do not push until the FAIL lines above are clean." >&2
    exit 1
fi
echo "ci_local: all selected gates passed."
