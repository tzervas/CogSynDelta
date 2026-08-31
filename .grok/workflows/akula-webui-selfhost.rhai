let meta = #{
    name: "akula-webui-selfhost",
    description: "Close Open WebUI self-hosted surface: Comfy image gen/edit, Piper/whisper audio, artifacts, follow-ups, functions, completions. No cloud OpenAI/Gemini/ElevenLabs.",
    when_to_use: "Image/edit/TTS/STT still hits cloud, or OWUI features (artifacts, self-prompt, tools) are off. After webui-ready.",
    phases: [
        #{ title: "Audit", detail: "live env vs self-hosted matrix" },
        #{ title: "Apply", detail: "merge_webui_env + optional Comfy unmask" },
        #{ title: "Verify", detail: "no cloud URLs; audio 8095; image engine comfyui" },
    ],
};

let ok_schema = #{
    "type": "object",
    "required": ["ok", "notes", "evidence"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
        "evidence": #{ "type": "string" },
        "gaps": #{ "type": "string" },
    },
};

fn as_str(x) {
    if type_of(x) == "string" { return x; }
    "";
}

let track = "all";
let do_apply = false;
let want_comfy = false;
if args != () {
    let t = as_str(args.track);
    if t != "" { track = t; }
    if args.apply == true { do_apply = true; }
    if args.comfy == true { want_comfy = true; }
}

phase("Audit");
let ap = "";
ap += "Repo /home/kang/code/personal/tzervas/akula-ai-platform. Use tools. Do not invent.\n";
ap += "Read scripts/lib/merge_webui_env.py, config/webui/webui.env.example, config/webui/overlay/manifest.json.\n";
ap += "On homelab, list KEY names and URL/engine VALUES from /data/akula/webui/webui.env. Redact keys/tokens.\n";
ap += "Matrix (all must be self-hosted, never api.openai.com / google / elevenlabs / azure / gemini):\n";
ap += "1. Chat completions: OPENAI_API_BASE_URLS only LAN (9120 gateway, 8079 nl-router, 8080 LocalAI).\n";
ap += "2. Image gen: IMAGE_GENERATION_ENGINE=comfyui, COMFYUI_BASE_URL LAN (172.32.0.1:8188 or 192.168.1.252:8188).\n";
ap += "3. Image edit: IMAGE_EDIT_ENGINE=comfyui (OWUI default is openai — that is a bug if unset).\n";
ap += "4. STT/TTS: AUDIO_*_OPENAI_API_BASE_URL=http://192.168.1.98:8095/v1 (OpenAI-compatible name, not cloud). Models piper + whisper-1.\n";
ap += "5. Features: follow-ups, title/tags/autocomplete, notes, memories, folders, automations, plugins/functions, pyodide code exec, duckduckgo or searx web search, artifacts via code exec.\n";
ap += "6. Overlay functions akula_kb/status/steer/media/temp_privacy imported. Tools against :8091 not cloud RAG.\n";
ap += "7. 5080 Comfy may be masked for autodev; that is exclusive-seq, not a license to use DALL-E. Probe :8188 honestly.\n";
ap += "8. Never pause 3090 LocalAI. Never 0.0.0.0 WAN. Never dual 14B.\n";
ap += "ok=true only if no cloud image/audio/LLM URLs. gaps=missing flags. evidence=engine/url table.\n";
let audit = agent(ap, #{
    label: "audit-selfhost",
    capability_mode: "read-only",
    output_schema: ok_schema,
});

let skip_apply = true;
if do_apply {
    skip_apply = false;
}
if track == "audit" {
    skip_apply = true;
}

if !skip_apply {
    phase("Apply");
    let pp = "";
    pp += "Repo /home/kang/code/personal/tzervas/akula-ai-platform. Isolated worktree off current branch; never main/staging/develop/dev.\n";
    pp += "Run python3 scripts/lib/merge_webui_env.py /data/akula/webui/webui.env on homelab after rsyncing config/webui/comfy-image-edit-workflow.json and comfy-edit-workflow-nodes.json to /data/akula/webui/.\n";
    pp += "Must set IMAGE_EDIT_ENGINE=comfyui, AUDIO_* to :8095, IMAGE_GENERATION_ENGINE=comfyui, ENABLE_FOLLOW_UP_GENERATION, ENABLE_CODE_EXECUTION=pyodide, ENABLE_PLUGINS, ENABLE_NOTES, ENABLE_MEMORIES, ENABLE_AUTOMATIONS, WEB_SEARCH_ENGINE=duckduckgo.\n";
    pp += "chmod 600 the env. sudo systemctl restart akula-open-webui.service.\n";
    if want_comfy {
        pp += "Comfy unmask: only if gpu5080.lock is idle and autodev is not in exclusive CUDA. Then unmask akula-comfyui on gpu5080. If lock is held, do NOT steal VRAM; leave Comfy down and report queued.\n";
    } else {
        pp += "Do not unmask Comfy (args.comfy not set). Image gen stays fail-closed while 5080 is autodev-exclusive.\n";
    }
    pp += "Never write operator secrets. Never docker stop akula-localai. ok=true if IMAGE_EDIT_ENGINE=comfyui in running container env. evidence=grep engines.\n";
    let applied = agent(pp, #{
        label: "apply-env",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    phase("Verify");
    let vp = "";
    vp += "After apply: curl -sS http://192.168.1.98:8095/v1/models expect piper and whisper-1.\n";
    vp += "POST a tiny TTS to :8095/v1/audio/speech model=piper (WAV bytes > 100). STT optional.\n";
    vp += "ssh homelab: podman exec akula-open-webui printenv | grep -E 'IMAGE_EDIT_ENGINE|IMAGE_GENERATION_ENGINE|AUDIO_TTS_OPENAI_API_BASE_URL|ENABLE_CODE_EXECUTION|ENABLE_FOLLOW_UP' — no api.openai.com.\n";
    vp += "curl -sk --resolve ai.vectorweight.com:443:192.168.1.170 https://ai.vectorweight.com/ expect 200.\n";
    vp += "Comfy: curl 192.168.1.252:8188 — 200 only if unmasked; empty/refused is ok if autodev holds the lock.\n";
    vp += "ok=true if audio 8095 works AND edit engine is comfyui AND no cloud URL in env. Never merge.\n";
    let ver = agent(vp, #{
        label: "verify-selfhost",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if audit != () && audit.success && audit.output.ok == true {
        if applied != () && applied.success && applied.output.ok == true {
            if ver != () && ver.success && ver.output.ok == true {
                sat = true;
            }
        }
    }
    complete(#{
        summary: if sat { "self-hosted WebUI matrix closed" } else { "self-hosted WebUI gaps remain" },
        sat: sat,
        audit: audit,
        apply: applied,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += "Read-only verify from audit evidence. Probe http://192.168.1.98:8095/v1/models and https://ai.vectorweight.com (resolve 192.168.1.170).\n";
vp2 += "ok=true if audio catalog has piper+whisper-1 and UI 200. Do not change env. Never merge.\n";
let ver2 = agent(vp2, #{
    label: "verify-readonly",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
let sat2 = false;
if audit != () && audit.success && audit.output.ok == true {
    if ver2 != () && ver2.success && ver2.output.ok == true {
        sat2 = true;
    }
}
complete(#{
    summary: if sat2 { "self-hosted WebUI honest (read-only)" } else { "gaps remain; re-run with args.apply=true" },
    sat: sat2,
    audit: audit,
    verify: ver2,
});
