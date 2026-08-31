# CSD secret scope (minimum privilege)

Models and autodev **do not** inherit the workstation vault
(`~/.secrets`). They use `/akula-data/cabal/csd-vault` with its own
age key. `secret ls` in that process cannot see `git/cabal-forgejo-admin`,
homelab sudo, Cloudflare, etc.

Homelab replica: `~/.config/csd/vault` (same age key, rsync via
`scripts/csd-vault-sync-homelab`). Never operator `~/.secrets` on
homelab. Never plaintext `localai.env`.

## Allowlist

| Name | Used for |
|---|---|
| `gpu/localai-api-key` | OpenAI HTTP to LocalAI / gateway |
| `csd/apply-token` | `POST /api/apply` worktree writes (not SSH) |
| `git/autodev` | Forgejo identity **autodev** (issues/PRs/push on program repos) |
| `hf/autodev` | HF token limited to `tzervas/cogsyndelta*` — when created |

## Must never be in the CSD vault

Operator Forgejo admin, `git/cabal-forgejo-agent`, homelab sudo, GPU
console passwords, GPG, Cloudflare, GitHub rescue, WebUI secret key,
`gpu/huggingface-token` (operator-wide).

## Accounts (self-hosted)

| Service | Account | Scope |
|---|---|---|
| LocalAI | existing API key | completions on `:8080` |
| Forgejo | **`autodev`** (restricted, not admin) | write collaborator on `tzervas/CogSynDelta` and `tzervas/memory-gate` only; token also repo-limited (`write:repository`, `write:issue`). Not `memory-gate-rs`. Password discarded; `prohibit_login` is off because Forgejo 16 also blocks API tokens when it is on. |
| Open WebUI | human operator | `ai.vectorweight.com` |
| HF | `hf/autodev` | **not minted.** Hugging Face has no API to create a fine-grained write token. Operator: huggingface.co/settings/tokens → fine-grained write on `tzervas/cogsyndelta*` only → `printf '%s' 'hf_…' \| SECRET_VAULT=/akula-data/cabal/csd-vault SOPS_AGE_KEY_FILE=$SECRET_VAULT/age.txt secret set hf/autodev`. Never copy `gpu/huggingface-token`. |

Checked 2026-08-31: CSD vault `secret ls` is `csd/apply-token`,
`git/autodev`, `gpu/localai-api-key` only. `hf/` dir exists empty.
Mint remains operator-only as above. Never copy `gpu/huggingface-token`.

Git as the bot: `./scripts/csd-autodev-git push forgejo HEAD:<branch>`.
Author/committer are `autodev <autodev@vectorweight.com>`.

Autodev systemd sets `SECRET_VAULT` + `SOPS_AGE_KEY_FILE` to the CSD
vault and `secret exec … TOKEN=git/autodev`. A model that shells out to
`secret ls` cannot list operator keys.

Provision (operator once):

```bash
secret exec TOKEN=git/cabal-forgejo-admin -- ./scripts/csd-autodev-provision
./scripts/csd-vault-sync-homelab
```
