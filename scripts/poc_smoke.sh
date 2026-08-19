#!/usr/bin/env bash
# Fast local / home-lab PoC smoke. CPU default.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
DEVICE="${1:-cpu}"
echo "== pytest PoC =="
python -m pytest tests/test_poc_contracts.py tests/test_poc_train.py tests/test_poc_compress.py -q --tb=short
echo "== train =="
python -m cogsyndelta.poc.cli train --device "$DEVICE" --steps 15 --checkpoint /tmp/poc_vae.pt
echo "== compress =="
python -m cogsyndelta.poc.cli compress --device "$DEVICE" --batch 4
echo "OK poc_smoke device=$DEVICE"
