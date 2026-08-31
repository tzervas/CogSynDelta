---
name: csd-autodev
description: Autodev lab CLIs — Forgejo git as autodev, GPU plan/lock observe, steer, HTTP apply. Use for PRs, wait-green, worktree writes. Never GitHub. Never operator admin.
---

# CSD autodev (lab CLIs)

Identity **autodev**. Token name `git/autodev` in the CSD vault
(`/akula-data/cabal/csd-vault`). Never `git/cabal-forgejo-admin`. Never
print TOKEN. Never GitHub. Never `main`/`staging`/`develop`/`dev`.
Never `kang-main-wip`. Never pause 3090 LocalAI. Do not wrap
`csd-autodev-provision`. Grok function tools stay media-shaped (`/akula-lab`).

Allowlist: `tzervas/CogSynDelta`, `tzervas/memory-gate`.

Git: `./scripts/csd-autodev-git <args>` (`TOKEN=git/autodev`). Result
`{ok, rc, stdout?}`. API: `secret exec TOKEN=git/autodev -- ./scripts/csd-autodev-forgejo <cmd>`.

| CLI | In | Out |
|---|---|---|
| `whoami` | — | `{http, login, memory_gate}` — 403 `/user` ok; `memory_gate` 200 |
| `prs <repo>` | `{repo, state?, limit?}` | `{http, repo, pulls[]}` |
| `pr-create <repo> <title> <head>` | `{repo, title, head, base?, body?}` | `{http, number, url, user}` |
| `status <repo> <sha>` | `{repo, sha}` | `{http, state, checks[]}` |
| `wait <repo> <sha>` | `{repo, sha, timeout?, poll?}` | `{http, state}` — **does not merge** |
| `comment <repo> <n> <body>` | `{repo, number, body}` | `{http, ok}` |
| `merge-gate <repo> <sha> <n>` | `{repo, sha, number}` | `{ok, state, merged, required_ran, required_succeeded, notes}` |

Prompt `prompts/forgejo-wait-green.md`: `may_merge` only if required
checks **ran and succeeded**. Skip / `|| true` / missing runner is not
green. Honest red stays red. Lab: `POST /api/forgejo/merge-gate`.
Do not merge CogSynDelta PR #1 until that holds.

| CLI | In | Out |
|---|---|---|
| `./scripts/csd-autodev-loop --once\|--worker` | `{mode}` | `{ok, rc, goal, applied, out}` — do not wrap provision |
| `./scripts/csd-autodev-priority on\|off\|status` | `{action}` | `{steer, comfy, webui}` — mask Comfy; never pause LocalAI |
| `./scripts/csd-gpu-plan` | — | `{autodev_priority, akula-prime, gpu5080}` writes `CSD_GPU_PLAN` |
| lock observe | — | `{lock, comfy, helper_ok}` from plan/status. Acquire/release is akula `with-gpu-5080` |
| `csd-lab-console --steer` / `POST /api/steer` | `{pause?, note?, next_goal?, autodev_priority?}` | same keys |
| `POST /api/apply` Bearer `csd/apply-token` | `{worktree, path, content}` | `{ok, wrote?, error?}` prefixes `src/` `tests/` `docs/` |

Apply prompt: `prompts/steer-apply-worktree.md`. No SSH.

Lab JSON (also under `/lab`): `GET /api/status`, `GET /api/gpu`,
`GET /api/gpu/lock`, `GET|POST /api/steer`, `POST /api/apply`,
`POST /api/git`, `GET|POST /api/forgejo/whoami`,
`GET|POST /api/forgejo/prs`, `POST /api/forgejo/pr-create`,
`GET|POST /api/forgejo/status`, `POST /api/forgejo/wait`,
`POST /api/forgejo/comment`, `POST /api/forgejo/merge-gate`,
`POST /api/loop`, `POST /api/priority`.
