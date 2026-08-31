---
name: owui-comfy-flux-klein-4b
description: Queue native OWUI Flux Klein 4B via scripts/csd-comfy. Fail-closed when Comfy is masked.
---

# owui-comfy-flux-klein-4b

Not an overlay. Native `IMAGE_GENERATION_ENGINE=comfyui` graph
`config/media/comfy-image-workflow.json` (same as
akula-ai-platform `config/webui/comfy-image-workflow.json`).
Prompt node **5**, size/steps nodes **4+10**, seed node **8**.

## Args

```json
{"text": "...", "width": 1024, "height": 1024, "steps": 4, "seed": 0}
```

## Steps

1. Do not unmask or start Comfy. Do not call DALL-E.
2. `./scripts/csd-comfy flux-klein-4b --text … --width … --height … --steps … --seed …`
   or `POST /api/comfy/flux-klein-4b` with the same JSON.

## Result

```json
{"prompt_id": "", "history": {}, "ok": false, "notes": "masked"}
```

When unmasked, `ok` is true and `prompt_id` is the Comfy `/prompt` id.
`akula_media` lists apps only; it does not POST `/prompt`.
