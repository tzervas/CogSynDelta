#!/usr/bin/env bash
# GPU Benchmark Capture Script for CogSynDelta
# Run on akula-prime to capture benchmark results with artifact persistence
#
# Usage:
#   ./scripts/capture_gpu_benchmarks.sh [output_dir]
#
# Results are saved to:
#   - benchmark_results/gpu_benchmark_YYYY-MM-DD_HHMMSS.json
#   - benchmark_results/gpu_benchmark_YYYY-MM-DD_HHMMSS.txt
#   - benchmark_results/latest.json (symlink to latest)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${1:-$PROJECT_ROOT/benchmark_results}"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
DATE_SHORT=$(date +%Y-%m-%d)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# Create output directory
mkdir -p "$OUTPUT_DIR"

log_info "=== CogSynDelta GPU Benchmark Capture ==="
log_info "Timestamp: $TIMESTAMP"
log_info "Output directory: $OUTPUT_DIR"

# Detect GPU
log_info "Detecting GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader | head -1)
    GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
    GPU_MEMORY=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
    DRIVER_VERSION=$(echo "$GPU_INFO" | cut -d',' -f3 | xargs)
    CUDA_VERSION=$(echo "$GPU_INFO" | cut -d',' -f4 | xargs)
    log_success "Found GPU: $GPU_NAME ($GPU_MEMORY)"
else
    log_error "nvidia-smi not found. Is CUDA installed?"
    exit 1
fi

# Get system info
HOSTNAME=$(hostname)
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")

# Output files
JSON_FILE="$OUTPUT_DIR/gpu_benchmark_${TIMESTAMP}.json"
TEXT_FILE="$OUTPUT_DIR/gpu_benchmark_${TIMESTAMP}.txt"
LATEST_JSON="$OUTPUT_DIR/latest.json"
BADGE_JSON="$OUTPUT_DIR/benchmark_badge.json"

# Run benchmarks and capture output
log_info "Running GPU benchmarks..."

cd "$PROJECT_ROOT"

# Create header for text output
{
    echo "=========================================="
    echo "CogSynDelta GPU Benchmark Results"
    echo "=========================================="
    echo "Date: $(date -Iseconds)"
    echo "Hostname: $HOSTNAME"
    echo "GPU: $GPU_NAME"
    echo "GPU Memory: $GPU_MEMORY"
    echo "CUDA Version: $CUDA_VERSION"
    echo "Driver Version: $DRIVER_VERSION"
    echo "Python: $PYTHON_VERSION"
    echo "PyTorch: $TORCH_VERSION"
    echo "=========================================="
    echo ""
} > "$TEXT_FILE"

# Run the main benchmark
if uv run python benchmarks/gpu_benchmark.py 2>&1 | tee -a "$TEXT_FILE"; then
    BENCHMARK_STATUS="passed"
    log_success "GPU benchmark completed successfully"
else
    BENCHMARK_STATUS="failed"
    log_warning "GPU benchmark completed with errors"
fi

# Run RTX 5080 specific benchmarks if available
log_info "Running RTX 5080 benchmarks..."
{
    echo ""
    echo "=========================================="
    echo "RTX 5080 Specific Benchmarks"
    echo "=========================================="
} >> "$TEXT_FILE"

if uv run python benchmarks/rtx5080_benchmark.py 2>&1 | tee -a "$TEXT_FILE"; then
    log_success "RTX 5080 benchmark completed"
else
    log_warning "RTX 5080 benchmark completed with errors"
fi

# Run industry benchmarks
log_info "Running industry benchmarks..."
{
    echo ""
    echo "=========================================="
    echo "Industry Standard Benchmarks"
    echo "=========================================="
} >> "$TEXT_FILE"

if uv run python benchmarks/industry_benchmarks.py 2>&1 | tee -a "$TEXT_FILE"; then
    log_success "Industry benchmark completed"
else
    log_warning "Industry benchmark completed with errors"
fi

# Extract key metrics for JSON (parse from text output)
# These patterns match the benchmark output format
MATRIX_OPS=$(grep -oP 'Matrix operations: \K[\d.]+' "$TEXT_FILE" 2>/dev/null || echo "0")
NN_INFERENCE=$(grep -oP 'NN inference: \K[\d.]+' "$TEXT_FILE" 2>/dev/null || echo "0")
MEMORY_COMPRESS=$(grep -oP 'Memory compression: \K[\d.]+' "$TEXT_FILE" 2>/dev/null || echo "0")
BATCH_THROUGHPUT=$(grep -oP 'Throughput: \K[\d.]+' "$TEXT_FILE" 2>/dev/null || echo "0")

# Create JSON output
cat > "$JSON_FILE" << EOF
{
  "meta": {
    "timestamp": "$(date -Iseconds)",
    "date": "$DATE_SHORT",
    "hostname": "$HOSTNAME",
    "status": "$BENCHMARK_STATUS"
  },
  "system": {
    "gpu_name": "$GPU_NAME",
    "gpu_memory": "$GPU_MEMORY",
    "cuda_version": "$CUDA_VERSION",
    "driver_version": "$DRIVER_VERSION",
    "python_version": "$PYTHON_VERSION",
    "pytorch_version": "$TORCH_VERSION"
  },
  "benchmarks": {
    "matrix_operations_gflops": $MATRIX_OPS,
    "nn_inference_ms": $NN_INFERENCE,
    "memory_compression_ratio": $MEMORY_COMPRESS,
    "batch_throughput_samples_sec": $BATCH_THROUGHPUT
  },
  "files": {
    "json": "gpu_benchmark_${TIMESTAMP}.json",
    "text": "gpu_benchmark_${TIMESTAMP}.txt"
  }
}
EOF

# Update latest symlink
ln -sf "gpu_benchmark_${TIMESTAMP}.json" "$LATEST_JSON"

# Create badge JSON for shields.io endpoint
cat > "$BADGE_JSON" << EOF
{
  "schemaVersion": 1,
  "label": "GPU Benchmark",
  "message": "$GPU_NAME | $DATE_SHORT",
  "color": "$([ "$BENCHMARK_STATUS" = "passed" ] && echo "brightgreen" || echo "orange")",
  "namedLogo": "nvidia",
  "logoColor": "white"
}
EOF

log_success "Results saved to:"
log_info "  JSON: $JSON_FILE"
log_info "  Text: $TEXT_FILE"
log_info "  Badge: $BADGE_JSON"
log_info "  Latest: $LATEST_JSON"

# Summary
echo ""
echo "=========================================="
echo "Benchmark Summary"
echo "=========================================="
echo "Status: $BENCHMARK_STATUS"
echo "GPU: $GPU_NAME"
echo "Date: $DATE_SHORT"
echo ""
echo "To upload results, commit and push benchmark_results/"
echo "Or use: gh release upload v0.2.0 $JSON_FILE $TEXT_FILE"
echo "=========================================="
