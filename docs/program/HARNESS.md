# CSD harness coupling

Loose coupling. GPU hosts are **inference providers**. Homelab is
dispatch + lab webapp. Pool flags in `config/model-router.json`
`pool.feature_flags` (`lan_rpc_pool` / `moe_split` stay false until
measured).

| Piece | Where | How |
|---|---|---|
| Lab UI `code.vectorweight.com` | homelab `:9118` | systemd; `/api/*` proxied to prime agent |
| Open WebUI `ai.vectorweight.com` | homelab | existing instance; gateway `:9120` |
| Gateway | homelab + prime | OpenAI HTTP → LocalAI on prime |
| Autodev worker | prime | worktrees + `/akula-data`; Restart=always |
| 3090 LocalAI | prime `:8080` | GGUF decode |
| 5080 | gpu5080 | CUDA/helpers; vLLM image parked |

Worktrees: **HTTP apply**, not SSH. Prime `POST /api/apply` with
`Authorization: Bearer` (`/akula-data/cabal/csd-apply.token`).
Allowlisted worktrees only; never `kang-main-wip`. Homelab dispatch
sets `CSD_APPLY_URL=http://192.168.1.98:9118`.

Forgejo identity is **`autodev`** (restricted user, CSD vault
`git/autodev`). Git: `./scripts/csd-autodev-git`. Homelab gateway
reads `LOCALAI_API_KEY` from `~/.config/csd/vault`, not plaintext env.

When `lan_rpc_pool` is enabled, the two cards look like one pool to
the driving model. Until then: place/migrate/spawn helpers only.
Never dual 14B. Never `0.0.0.0`.
