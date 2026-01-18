#!/bin/bash
# =============================================================================
# GPU Tests and Benchmarks Runner for akula-prime
# =============================================================================
# This script SSHs to akula-prime workstation to run GPU-specific tests
# and benchmarks using the RTX 5080.
#
# Prerequisites:
#   - SSH key auth configured for akula-prime (via ~/.ssh/config profile)
#   - Project synced to akula-prime (use rsync or git)
#   - Python environment with uv available on akula-prime
#
# Usage:
#   ./scripts/run_gpu_tests.sh [command]
#
# Commands:
#   tests      - Run GPU-enabled pytest tests
#   benchmark  - Run GPU benchmarks
#   rtx5080    - Run RTX 5080 specific benchmarks
#   all        - Run all GPU tests and benchmarks (default)
#   sync       - Sync local project to akula-prime
# =============================================================================

set -euo pipefail

# Configuration
REMOTE_HOST="akula-prime"
REMOTE_PROJECT_DIR="~/projects/CogSynDelta"
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check SSH connectivity
check_connection() {
    log_info "Checking SSH connection to ${REMOTE_HOST}..."
    if ssh -o ConnectTimeout=5 "${REMOTE_HOST}" "echo 'Connection OK'" > /dev/null 2>&1; then
        log_success "SSH connection to ${REMOTE_HOST} established"
        return 0
    else
        log_error "Cannot connect to ${REMOTE_HOST}. Check SSH config and network."
        exit 1
    fi
}

# Check CUDA availability on remote
check_cuda() {
    log_info "Checking CUDA availability on ${REMOTE_HOST}..."
    local cuda_check
    cuda_check=$(ssh "${REMOTE_HOST}" "python3 -c 'import torch; print(torch.cuda.is_available())'" 2>/dev/null || echo "false")
    if [[ "${cuda_check}" == "True" ]]; then
        local gpu_name
        gpu_name=$(ssh "${REMOTE_HOST}" "python3 -c 'import torch; print(torch.cuda.get_device_name(0))'" 2>/dev/null)
        log_success "CUDA available: ${gpu_name}"
        return 0
    else
        log_error "CUDA not available on ${REMOTE_HOST}"
        exit 1
    fi
}

# Sync project to remote
sync_project() {
    log_info "Syncing project to ${REMOTE_HOST}:${REMOTE_PROJECT_DIR}..."

    # Create remote directory if needed
    ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_PROJECT_DIR}"

    # Rsync with exclusions
    rsync -avz --progress \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.venv' \
        --exclude 'venv' \
        --exclude '.uv' \
        --exclude 'dist' \
        --exclude 'build' \
        --exclude '*.egg-info' \
        --exclude 'memory_storage' \
        --exclude 'model_sections/*.pt' \
        "${LOCAL_PROJECT_DIR}/" \
        "${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/"

    log_success "Project synced to ${REMOTE_HOST}"
}

# Run GPU tests
run_gpu_tests() {
    log_info "Running GPU-enabled tests on ${REMOTE_HOST}..."

    ssh -t "${REMOTE_HOST}" "cd ${REMOTE_PROJECT_DIR} && \
        uv sync --group dev && \
        uv run pytest tests/ -v -m 'gpu or cuda' --tb=short 2>&1" || {
        # If no GPU-specific markers, run all tests with GPU available
        ssh -t "${REMOTE_HOST}" "cd ${REMOTE_PROJECT_DIR} && \
            uv run pytest tests/ -v --tb=short 2>&1"
    }

    log_success "GPU tests completed"
}

# Run GPU benchmarks
run_benchmarks() {
    log_info "Running GPU benchmarks on ${REMOTE_HOST}..."

    ssh -t "${REMOTE_HOST}" "cd ${REMOTE_PROJECT_DIR} && \
        uv sync && \
        uv run python benchmarks/gpu_benchmark.py 2>&1"

    log_success "GPU benchmarks completed"
}

# Run RTX 5080 specific benchmarks
run_rtx5080_benchmark() {
    log_info "Running RTX 5080 benchmarks on ${REMOTE_HOST}..."

    ssh -t "${REMOTE_HOST}" "cd ${REMOTE_PROJECT_DIR} && \
        uv sync && \
        uv run python benchmarks/rtx5080_benchmark.py 2>&1"

    log_success "RTX 5080 benchmarks completed"
}

# Run industry benchmarks
run_industry_benchmarks() {
    log_info "Running industry standard benchmarks on ${REMOTE_HOST}..."

    ssh -t "${REMOTE_HOST}" "cd ${REMOTE_PROJECT_DIR} && \
        uv sync && \
        uv run python benchmarks/industry_benchmarks.py 2>&1"

    log_success "Industry benchmarks completed"
}

# Run all
run_all() {
    sync_project
    run_gpu_tests
    run_benchmarks
    run_rtx5080_benchmark
}

# Print usage
print_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  tests       - Run GPU-enabled pytest tests"
    echo "  benchmark   - Run GPU benchmarks"
    echo "  rtx5080     - Run RTX 5080 specific benchmarks"
    echo "  industry    - Run industry standard benchmarks"
    echo "  all         - Run all GPU tests and benchmarks (default)"
    echo "  sync        - Sync local project to akula-prime only"
    echo "  check       - Check connection and CUDA availability"
    echo ""
}

# Main
main() {
    local command="${1:-all}"

    echo "=============================================="
    echo "  CogSynDelta GPU Tests & Benchmarks Runner"
    echo "=============================================="
    echo ""

    check_connection

    case "${command}" in
        tests)
            check_cuda
            sync_project
            run_gpu_tests
            ;;
        benchmark)
            check_cuda
            sync_project
            run_benchmarks
            ;;
        rtx5080)
            check_cuda
            sync_project
            run_rtx5080_benchmark
            ;;
        industry)
            check_cuda
            sync_project
            run_industry_benchmarks
            ;;
        all)
            check_cuda
            run_all
            ;;
        sync)
            sync_project
            ;;
        check)
            check_cuda
            ;;
        help|--help|-h)
            print_usage
            ;;
        *)
            log_error "Unknown command: ${command}"
            print_usage
            exit 1
            ;;
    esac

    echo ""
    log_success "Done!"
}

main "$@"
