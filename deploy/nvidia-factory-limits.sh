#!/bin/bash
# Apply factory GPU clocks and default power limits (no undervolt curve).
# Unlocks --lock-gpu-clocks / memory locks, enables persistence, -pl default.
set -euo pipefail
export PATH="/usr/bin:/bin:/usr/sbin:/sbin"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi missing" >&2
  exit 0
fi
nvidia-smi -pm 1 || true
nvidia-smi --query-gpu=uuid,power.default_limit --format=csv,noheader \
  | while IFS=, read -r uuid defpl; do
      uuid="${uuid#"${uuid%%[![:space:]]*}"}"
      uuid="${uuid%"${uuid##*[![:space:]]}"}"
      defpl=$(echo "$defpl" | awk '{print $1}')
      [ -n "$uuid" ] || continue
      nvidia-smi --reset-gpu-clocks -i "$uuid" || true
      nvidia-smi --reset-memory-clocks -i "$uuid" || true
      nvidia-smi --reset-applications-clocks -i "$uuid" || true
      nvidia-smi -pl "$defpl" -i "$uuid" || true
    done
