# GPU Benchmark Results

This directory stores GPU benchmark results from manual runs on akula-prime.

## Files

- `gpu_benchmark_YYYY-MM-DD_HHMMSS.json` - Structured benchmark data
- `gpu_benchmark_YYYY-MM-DD_HHMMSS.txt` - Human-readable benchmark output
- `latest.json` - Symlink to the most recent benchmark
- `benchmark_badge.json` - Shields.io endpoint badge data

## Running Benchmarks

On akula-prime (or any machine with CUDA GPU):

```bash
# From project root
./scripts/capture_gpu_benchmarks.sh

# Or specify output directory
./scripts/capture_gpu_benchmarks.sh /path/to/output
```

## Uploading Results

### Option 1: Commit to Repository

```bash
git add benchmark_results/
git commit -m "chore: Update GPU benchmark results (YYYY-MM-DD)"
git push
```

### Option 2: Upload as Artifact via GitHub Actions

```bash
# Encode files for workflow dispatch
JSON_B64=$(base64 -w0 benchmark_results/latest.json)
TEXT_B64=$(base64 -w0 benchmark_results/gpu_benchmark_*.txt | head -1)

# Trigger workflow (requires gh CLI)
gh workflow run gpu-benchmark.yml \
  -f benchmark_json="$JSON_B64" \
  -f benchmark_text="$TEXT_B64" \
  -f gpu_name="RTX 5080" \
  -f run_date="$(date +%Y-%m-%d)"
```

### Option 3: Attach to Release

```bash
gh release upload v0.2.0 benchmark_results/gpu_benchmark_*.json benchmark_results/gpu_benchmark_*.txt
```

## Badge Integration

The `benchmark_badge.json` file is compatible with shields.io endpoint badges:

```markdown
![GPU Benchmark](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/tzervas/CogSynDelta/main/benchmark_results/benchmark_badge.json)
```

For private repos, use GitHub Pages or a private gist with appropriate authentication.
