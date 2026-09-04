#!/usr/bin/env bash
# PreToolUse: deny git commit/push on CogSynDelta protected trunks.
# Reads Grok/Claude hook JSON on stdin. Fail-open if payload is not git.
set -uo pipefail
payload="$(cat || true)"
root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$root" ]] || exit 0
[[ -f "$root/STATUS.md" && -d "$root/src/cogsyndelta" ]] || exit 0

cmd=""
if command -v python3 >/dev/null 2>&1; then
  cmd="$(printf '%s' "$payload" | python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    obj = json.loads(raw) if raw.strip() else {}
except json.JSONDecodeError:
    obj = {}
ti = obj.get("toolInput") or obj.get("tool_input") or obj.get("input") or {}
if isinstance(ti, dict):
    print(ti.get("command") or ti.get("cmd") or "")
' 2>/dev/null || true)"
fi
[[ -n "$cmd" ]] || exit 0

case "$cmd" in
  *git\ commit*|*git\ push*|*git\ merge*) ;;
  *) exit 0 ;;
esac

branch="$(git -C "$root" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
protected="${CSD_PROTECTED_BRANCHES:-main staging develop dev}"
for pat in $protected; do
  # $pat is meant to glob-match (main, "release-*", ...), not compare literally.
  # shellcheck disable=SC2254
  case "$branch" in
    $pat)
      printf '{"decision":"block","reason":"CogSynDelta: refused git mutate on protected branch %s. Use feat/* worktree."}\n' "$branch"
      exit 0
      ;;
  esac
done
exit 0
