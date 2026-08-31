# CSD autodev handoff

Hosted **Grok is research / plan / unblock only**. Do not implement
memory-gate slices in the hosted Grok chat. **Autodev implements** via
the self-hosted loop, Forgejo tools, and Forgejo workflows.

## Identity and never

| Rule | Value |
|---|---|
| Git identity | Forgejo **`autodev`** |
| Token | CSD vault `TOKEN=git/autodev` via `./scripts/csd-autodev-git` |
| Operator admin | stays in `~/.secrets` — never copy into the CSD vault |

**Never:** GitHub bot push; working branch `main` / `staging` / `develop` /
`dev`; pause 3090 LocalAI; unmask Comfy; bind `0.0.0.0` WAN; write operator
or model vaults; mix 384-d Qdrant (`akula-csd-kb` stays 1024-d); merge unless
required Forgejo checks **ran and succeeded** (skip / `|| true` /
`continue-on-error` / fallback `echo` / missing runner is not green).

## Roles

| Box | IP | Job |
|---|---|---|
| 3090 (akula-prime) | `192.168.1.98` | `local/code` implementer. LocalAI stays loaded. |
| 5080 (gpu5080) | **`192.168.1.251`** (never `.252`) | Exclusive CUDA / CI / index. Comfy masked. |
| 1080 Ti guest | VFIO `06:00.0` on gpu5080 | RAG retrieve/index/light **only if** guest `nvidia-smi` lists Pascal. |

Guest is **not live** (no `virsh` domain, no guest IP, host smi is 5080-only).
Still run autodev on **3090 + 5080**. RAG stays **5080-embed or prime** until
`future_hosts.gpu5080-1080ti.live=true`. Do not start `G-1080`.

## Next closeable (not P1-08)

P1-00..P1-04, P1-06, P1-07 / G-QD, and **P1-08** are **merged**. P1-08
(`feat/tiered-memory-policy`) landed Forgejo memory-gate PR #11
`2c11c3f` (head `43272f7`) 2026-08-31T17:17:11Z. Do not steer autodev
back to P1-08.

**Steer `next_goal`: `P1-09`** — retrieval and domain isolation
(`feat/gateway-retrieve-domain`). Depends only on P1-04 (met). Closeable
now that P1-08 is on `main`.

Remaining Phase 1 after P1-08:

| ID | Depends | Notes |
|---|---|---|
| **P1-09** | P1-04 | **Next closeable.** Gateway retrieve + mandatory/global domain. |
| P1-05 | P1-04 | Chroma adjudicate. PR #3 stays unmerged/red. Not this steer. |
| P1-10 | P1-03, P1-09 | Metrics after durable retrieve exists. |
| P1-11 | P1-08, P1-09 | Fast CLS after both P1-08 and P1-09. |
| P1-12 | P1-11 | Resumable slow CLS. |
| P1-13 | P1-09, P1-11 | Akula persona backend (read-only). |
| P1-14 | P1-10, P1-12, P1-13 | Public façade. |
| P1-15 | P1-07, P1-09 | Golden recall on private HF. |
| P1-16 | P1-14, P1-15 | Lifecycle through restart. Then stop. |
| P1-17 | P1-16 | Phase 1 exit docs. |

## How autodev implements

1. Cut `feat/gateway-retrieve-domain` from Forgejo `main` into
   `/home/kang/code/personal/tzervas/python-ai/memory-gate-wt-p1-09`.
   Never touch `python-ai/memory-gate` (`local/kang-main-wip`).
2. `CSD_AUTODEV_WT` that worktree. Apply allowlist: `src/` `tests/` `docs/`.
3. One failing pytest + one function per tick (`local/code` via
   `./scripts/csd-autodev-loop --worker` → `csd-localai-queue`).
4. Push as autodev: `./scripts/csd-autodev-git push forgejo HEAD:<branch>`.
5. PR / status / merge-gate: `./scripts/csd-autodev-forgejo`. Workflows merge
   themselves when required jobs **ran and succeeded**, then delete the head
   unless `main` / `release/*`. Keep `local/kang-main-wip`.

P1-09 DoD: gateway retrieve uses the typed store protocol and typed errors;
domain is mandatory or explicitly global; identical text in different domains
cannot cross; ranking/limits are deterministic on the in-memory oracle.

CogSynDelta `feat/agent-harness` PR #3 stays **open** while required checks
include skipped Security/Pre-commit/fleet-ci python jobs. Combined
`success` with skip-theatre is **not** green. Do not merge it.

## Steer

File: `/akula-data/cabal/csd-steer.json` (`CSD_STEER`). Lab:
`POST /api/steer` on `http://192.168.1.98:9118` (LAN bind, not WAN).

```bash
./scripts/csd-lab-console --steer-next P1-09
# or
curl -sS -X POST http://192.168.1.98:9118/api/steer \
  -H 'Content-Type: application/json' \
  -d '{"next_goal":"P1-09","pause":false}'
```

`ok` for this handoff is that file/API `next_goal` is the next P1 (**P1-09**,
not P1-08).

## Cluster snapshot / Grafana

Lab snapshot `cluster.gpu5080_1080ti` must include `host_ip=192.168.1.251`
and `guest_ip` (null until the VFIO guest has an address). Grafana catalog
path (filters only; **do not invent series**):

`group=akula-rag` `host=gpu5080` `path=lab.gpu5080.index.1080ti`

Relabel in `deploy/o11y/vm-scrape.yml` fires only when `akula_gpu_*` `name`
matches 1080 Ti. Until then panels for that path are empty.

## References

- Board: [PHASE-1-TASK-BOARD.md](PHASE-1-TASK-BOARD.md)
- Implementer pack: [IMPLEMENTER-HANDOFF.md](IMPLEMENTER-HANDOFF.md)
- Loop: [GOAL-LOOP.md](GOAL-LOOP.md) · ledger [GOALS.md](GOALS.md)
- Visibility: [WHO-RUNS-WHAT.md](WHO-RUNS-WHAT.md)
- 1080 Ti: [CSD-1080TI-RAG.md](CSD-1080TI-RAG.md)
- Taxonomy: [CSD-O11Y-TAXONOMY.md](CSD-O11Y-TAXONOMY.md)
