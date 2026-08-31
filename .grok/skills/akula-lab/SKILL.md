---
name: akula-lab
description: Self-hosted lab backends for models/agents — notes Jupyter, Piper/whisper, Comfy, Open WebUI tools. Use when chat/image/audio/code-exec must stay on LAN.
---

# Akula lab backends (agent tools)

Never call cloud OpenAI/Gemini/ElevenLabs for these. Open WebUI overlay
Tools (`akula_media`, `akula_kb`) expose the same surfaces in chat.

| Need | Where | How |
|---|---|---|
| Code REPL / artifacts | https://notes.vectorweight.com/lab | Jupyter `http://172.32.0.1:8086` (token). OWUI interpreter engine `jupyter`. Not pfn. |
| Field Notes docs | https://pfn.vectorweight.com | Guide only, not a kernel. |
| TTS / STT / SFX | prime `:8095` | `POST /v1/audio/speech` model `piper`; STT `whisper-1`. CPU. Tool: `akula_audio_pipelines`. |
| Image gen/edit | Comfy on 5080 | OWUI `IMAGE_*_ENGINE=comfyui`. Native graphs: https://media.vectorweight.com. Exclusive with autodev CUDA — if Comfy is masked, queue, do not fall back to DALL-E. Tool: `akula_media_list`. |
| Chat completions | LocalAI / gateway | `192.168.1.98:8080` or homelab `:9120`. Never dual 14B. |
| KB retrieve | `:8091` keyword-cpu | Tool: `akula_kb`. Never write operator/model vaults. |

## Forgejo as autodev (tools)

Identity is **autodev**. Token `git/autodev` in the CSD vault. Never
`git/cabal-forgejo-admin`. Allowlist: `tzervas/CogSynDelta`,
`tzervas/memory-gate`.

| Tool | Use |
|---|---|
| `./scripts/csd-autodev-git <git args>` | clone/fetch/push/commit author autodev |
| `secret exec TOKEN=git/autodev -- ./scripts/csd-autodev-forgejo prs memory-gate` | list PRs |
| `… pr-create memory-gate "title" feat/branch` | open PR |
| `… status CogSynDelta <sha>` | combined checks |
| `… wait CogSynDelta <sha>` | poll; do not merge skip-theatre |

Merge only when required checks **ran and succeeded**. Never GitHub.

Skills in this repo: `/csd-ops` for GPU; this skill for I/O + Forgejo tools.
