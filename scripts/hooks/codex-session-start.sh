#!/usr/bin/env bash
# Codex SessionStart: JSON additionalContext. Always exit 0.
set -u
root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
sha="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
export CTX="CogSynDelta Codex. HEAD=${branch}@${sha} root=${root}. STATUS.md + Python source win over docs/HANDOFF-ARCHITECTURE.md. Drive docs/PROGRAM-GOALS.md: Phase 0 branch KEEP/CLOSE on CogSynDelta+memory-gate+memory-gate-rs; Phase 1 Python memory-gate; Rust later. GPUs: docs/CODEX-OPS.md — 3090 keep local/code; 5080 exclusive-seq for CSD CUDA; csd-kb-index prefers 1080 Ti guest; never pause LocalAI for RAG search. Git: git.vectorweight.com deep trees (never shallow main). Weights: private tzervas/cogsyndelta. Never commit main/staging/develop/dev."
exec python3 -c 'import json,os; print(json.dumps({"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":os.environ["CTX"]}}))'
