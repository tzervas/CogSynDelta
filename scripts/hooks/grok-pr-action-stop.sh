#!/usr/bin/env bash
# Stop hook: if a watched Forgejo PR just went green or red, keep the turn.
# Fail-open. Only gates reason=end_turn. Never merge from a hook.
set -uo pipefail
payload="$(cat || true)"
reason="$(printf '%s' "$payload" | python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    obj = json.loads(raw) if raw.strip() else {}
except json.JSONDecodeError:
    obj = {}
print(obj.get("reason") or "")
' 2>/dev/null || true)"
[[ "$reason" == "end_turn" ]] || exit 0

state="${CSD_WATCH_STATE:-/tmp/csd-forgejo-pr-watch.json}"
[[ -f "$state" ]] || exit 0

python3 - "$state" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    data = json.loads(p.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    sys.exit(0)
ready = []
red = []
for key, detail in data.items():
    combined = str(detail).split(" ", 1)[0]
    if combined == "success" and "mergeable=True" in str(detail):
        ready.append(f"{key} {detail}")
    elif combined in {"failure", "error"}:
        red.append(f"{key} {detail}")
msg = []
if ready:
    msg.append("Forgejo PR green and mergeable: " + "; ".join(ready[:4])
               + ". Merge only if required jobs ran and succeeded. Skip is not green.")
if red:
    msg.append("Forgejo PR red: " + "; ".join(red[:4])
               + ". Patch in a worktree; do not merge.")
if not msg:
    sys.exit(0)
print(json.dumps({
    "decision": "block",
    "reason": " ".join(msg),
}))
PY
