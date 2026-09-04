---
name: akula-lab
description: LAN media backends — notes Jupyter, Piper/whisper/SFX, Comfy, gap retrieve. Use when chat/image/audio/code-exec must stay on LAN. Git/CI live in /csd-autodev.
---

# Akula lab backends (media-shaped)

Never call cloud OpenAI/Gemini/ElevenLabs/DALL-E. Identity **autodev**;
CSD vault `/akula-data/cabal/csd-vault` only. Git/Forgejo/GPU/steer/apply:
skill `/csd-autodev` (CLIs). Grok function tools stay media-shaped.

| Need | Where | How |
|---|---|---|
| Code REPL | https://notes.vectorweight.com/lab | Jupyter `http://172.32.0.1:8086` LAN. Tool: `akula_notes_sandbox`. Not pfn. |
| TTS / STT / SFX | prime `:8095/v1` | Piper `piper`, STT `whisper-1`, SFX. CPU. Tool: `akula_audio_pipelines`. |
| Image | Comfy `192.168.1.252:8188` | Native graphs, not UI nodes. OWUI `IMAGE_GENERATION_ENGINE=comfyui`. Fail-closed if masked. |
| Chat | LocalAI / gateway | `192.168.1.98:8080` or homelab `:9120`. Never dual 14B. |
| KB retrieve | `:8091` keyword-cpu | `gap_search` / `akula_kb`. Never write `tzervas-dev-kb` or `akula-model-kb`. Never mix 384-d. |

## Comfy (fail-closed)

Comfy stays **masked** unless the operator unmasks. Autodev owns 5080.
Real I/O: `./scripts/csd-comfy` and lab `POST /api/comfy/{list-workflows,queue-prompt,prompt-status,flux-klein-4b}`.

| Tool | In | Out |
|---|---|---|
| `comfy-list-workflows` | `{source?}` | `{ok, workflows[], notes}` — 31 graphs in `config/media/workflows.json` or `/object_info` when unmasked |
| `comfy-queue-prompt` | `{prompt}` object | `{ok, prompt_id, notes}` — POST `192.168.1.252:8188/prompt` API graph |
| `comfy-prompt-status` | `{prompt_id?}` | `{ok, queue, history, system_stats, notes}` — GET `/queue` `/history/id` `/system_stats` |

If masked: `{ok: false, notes: "masked"}`. Do not start Comfy. Do not fall
back to DALL-E. Overlay `akula_media` lists apps; it does not POST `/prompt`.

OWUI Flux Klein 4B is native `IMAGE_GENERATION_ENGINE=comfyui` graph
`config/media/comfy-image-workflow.json` (prompt node 5, size/steps 4+10,
seed 8) — not an overlay. Workflow `/owui-comfy-flux-klein-4b`.

## Notes / audio / gap

- `akula_notes_sandbox` → `{ok, ui, jupyter, kernels[]}`. Jupyter LAN only.
- `akula_audio_pipelines` → `{ok, openai, tts, stt, sfx}` at `192.168.1.98:8095/v1`.
- `gap_search` `{query, limit?}` → `{hits[], error?}`. Retrieve-only `KB_HTTP_TOKEN`.
