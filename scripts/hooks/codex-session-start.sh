#!/usr/bin/env bash
# Codex SessionStart: JSON additionalContext. Always exit 0.
set -u
root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
sha="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
export CTX="CogSynDelta Codex. HEAD=${branch}@${sha} root=${root}. Truth: STATUS.md then pyproject then src/cogsyndelta/poc. docs/HANDOFF-ARCHITECTURE.md is unvetted ChatGPT context — vet against live Python (CogSynDelta + memory-gate). Python-first; Rust later for speed/safety. Do not implement Titans/GDN/JEPA or start a Rust rewrite this turn. Strata: docs/GROK-STRATA.md. KB: RO tzervas-dev-kb + akula-model-kb; RW akula-csd-kb (./scripts/csd-kb-index on 5080, never pause LocalAI) and akula-gap-kb. Never write operator/model vaults. Never commit main/staging/develop/dev."
exec python3 -c 'import json,os; print(json.dumps({"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":os.environ["CTX"]}}))'
