#!/usr/bin/env bash
# SessionStart orientation for CogSynDelta. Pure reads. Always exit 0.
set -u
echo_msg() { printf 'csd-orient: %s\n' "$*" >&2; }

root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
sha="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo_msg "HEAD=$branch @$sha"
echo_msg "root=$root"

protected="${CSD_PROTECTED_BRANCHES:-main staging develop dev}"
is_protected=0
# shellcheck disable=SC2086
for pat in $protected; do
  # shellcheck disable=SC2254
  case "$branch" in
    $pat) is_protected=1; break ;;
  esac
done
if [[ "$is_protected" -eq 1 ]]; then
  echo_msg "WARNING: protected trunk — land via PR. Worktree: csd-worktress/CogSynDelta-wt-grok-harness"
else
  echo_msg "working branch (not a protected trunk)"
fi

behind_main="$(git rev-list --count HEAD..origin/main 2>/dev/null || echo '?')"
echo_msg "behind origin/main=$behind_main"

echo_msg "truth: STATUS.md > pyproject description > src/cogsyndelta/poc > ADRs > README"
echo_msg "skills: /csd-context /code-thropology   workflow: /csd-code-thropology"
echo_msg "GPU: 3090 one GGUF LocalAI; 5080 exclusive PoC CUDA; do not pause LocalAI"
echo_msg "vault: Projects/Repositories/GitHub/tzervas/CogSynDelta/CogSynDelta.md"
exit 0
