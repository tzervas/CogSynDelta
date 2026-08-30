# Codex harness — CogSynDelta

OpenAI Codex CLI/IDE reads `AGENTS.md`, then `.codex/` (trusted projects only). This branch is the working tree for a **code-thropology + architecture** pass. Do not author on `main`.

## Launch

```bash
cd /home/kang/code/personal/tzervas/CogSynDelta   # this repo, feat/agent-harness
./scripts/codex-csd
# or:  codex
```

Trust the project when Codex asks (required for `.codex/config.toml`, hooks, MCP). Review hooks with `/hooks`.

Architecture intent (not measured STATUS): `docs/HANDOFF-ARCHITECTURE.md`. **Read that file** (it is too large to embed in AGENTS.md). Measured capabilities: `STATUS.md`. Strata/PRs: `docs/GROK-STRATA.md`.

First Codex turn should follow §35 of the handoff: map implemented / partial / planned / proposed. Do not import Gated DeltaNet, Titans, or JEPA in that turn.

## Knowledge planes (host agent, like Grok)

Codex is a **host agent**, not LocalAI-in-container. Same planes as Grok.

| Plane | Path | Qdrant | Codex mode |
|---|---|---|---|
| Operator KB | `/home/kang/data/obsidian/tzervas-dev-kb/tzervas-dev-kb` | `tzervas-dev-kb` | **read-only** MCP `tzervas-dev-kb` + `:8091` |
| Model workspace | `/akula-data/obsidian/akula-model-kb` | `akula-model-kb` | **read-only** MCP `akula-model-kb` + `:8091` |
| Shared Cabal segments | `/akula-data/obsidian/cabal-shared-kb` | `cabal-shared-kb` | **read-only** after ingest (`Lessons/`, `Priorities/`, `Handoff/`) |
| Personas | `/akula-data/obsidian/akula-personas` | — | MCP `akula-personas*` (instantiate, do not mix into operator KB) |
| Gap / process memory | `/akula-data/obsidian/akula-gap-kb` | `akula-gap-kb` | RW MCP `akula-gap-*` (PR/CI lessons). Not chat RAG. |
| **CSD experiment vault** | `/akula-data/obsidian/akula-csd-kb` | `akula-csd-kb` | **RW + CUDA reindex** — this is Codex’s write/index plane |

Keyword retrieve on `:8091` / `:8092` is CPU. Vector **index** of `akula-csd-kb` or gap-kb:

```bash
./scripts/csd-kb-setup
./scripts/csd-kb-index          # 5080 exclusive-seq; never pause 3090 LocalAI
# or from akula-ai-platform: make index-gap-kb
```

Never write `tzervas-dev-kb` or `akula-model-kb` from Codex. Never merge collections. 1024-d Qwen3 only.

## Config map

| File | What |
|---|---|
| `AGENTS.md` | Always-on (Codex concatenates with `~/.codex/AGENTS.md`) |
| `.codex/config.toml` | sandbox, approvals, MCP, `project_doc_max_bytes` |
| `.codex/hooks.json` | SessionStart orient + branch guard |
| `.codex/skills/` | `csd-context`, `code-thropology` |
| `docs/HANDOFF-ARCHITECTURE.md` | Full architecture discussion — **open with read** |
| `config/clients/grok.toml.example` | LocalAI aliases (optional; Codex default is OpenAI models) |

Install CLI if missing: `npm i -g @openai/codex`. Auth is ChatGPT/API in `~/.codex` (not this repo).
