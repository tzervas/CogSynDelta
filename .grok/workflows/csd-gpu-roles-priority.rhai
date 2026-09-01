let meta = #{
    name: "csd-gpu-roles-priority",
    description: "3090=autodev/local/code priority. 5080=resident chat+Comfy/media, yields lock to autodev CUDA. 1080 Ti=RAG/light. Fit-check; no 14B on 5080 unless it FITs.",
    when_to_use: "After keepup-queue; operator wants 5080 as chat/Comfy home and 3090 for autodev, autodev always preempts media/chat.",
    phases: [
        #{ title: "Fit", detail: "can 5080 hold a chat GGUF + Comfy leftover?" },
        #{ title: "Policy", detail: "prio: autodev/local/code > OWUI chat > Comfy > idle helpers" },
        #{ title: "Wire", detail: "router aliases + queue prio; 3090 stays 14B" },
    ],
};

let ok_schema = #{
    "type": "object",
    "required": ["ok", "notes", "evidence"],
    "additionalProperties": false,
    "properties": #{
        "ok": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
        "evidence": #{ "type": "string" },
        "fits_5080_chat": #{ "type": "boolean" },
    },
};

let do_apply = false;
if args != () {
    if args.apply == true { do_apply = true; }
}

let hard = "";
hard += "HARD: Autodev and local/code ALWAYS preempt Open WebUI chat and Comfy. Never dual 14B. Never pause LocalAI by killing the container for RAG. ";
hard += "3090 Ti (~24GiB): default home for local/code 14B Q4 32k (~16.5GiB) — autodev implementer. Swap aliases only if they FIT leftover. ";
hard += "5080 (~16GiB): resident chat (only if a GGUF FITs without the 14B) + Comfy image/video gen/edit. CUDA CI/train still exclusive-seq via gpu5080.lock — autodev CUDA preempts Comfy/chat on that card. ";
hard += "Do NOT move 14B onto the 5080 unless nvidia-smi free after load is honest. If 14B does not FIT, chat on 5080 is a smaller model. ";
hard += "1080 Ti guest: retrieve-index-light only. Never GitHub. Never 0.0.0.0 WAN. gpu5080=192.168.1.251. Identity autodev.\n";
hard += "Keep services UP: queue/lock, not mask. keepup-queue may still be wiring Comfy lock wrap — do not fight it; set prio in timeshare JSON.\n";
hard += "SHARD/SHARE (do not start a separate workflow): Autodev may share the 3090 if weights+KV FITs leftover safeguard. ";
hard += "Fan-out: 1 latency-tight worker stays 3090 local/code; parallel/ok-lan helpers -> 3090 leftover then 5080 share-small (lock idle) then 1080 Ti. ";
hard += "Need sm_120/FP8/CUDA -> 5080 exclusive-seq (Comfy waits). Never dual 14B. Tensor-wire G-POOL is NOT this. 256 specialists is NOT live.\n";

phase("Fit");
let fp = "";
fp += hard;
fp += "Read config/model-router.json v2. ssh .251 nvidia-smi memory. 3090 smi used. Guest 1080 Ti smi. ";
fp += "fits_5080_chat=true only if a catalog chat alias (not 14B) vram_mib + Comfy helper_cap (~6GiB) <= 5080 safeguard. ";
fp += "Write the numbers into docs/program/CSD-GPU-ROLES.md. FINAL json including fits_5080_chat.\n";
let fit = agent(fp, #{
    label: "fit-5080-chat",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Policy");
    let pp = "";
    pp += hard;
    pp += "CSD-GPU-ROLES.md + CSD-GPU-KEEPUP.md + CODEX-OPS.md: priority order autodev/local/code > OWUI text > Comfy/media > share-small idle. ";
    pp += "3090 never given to Comfy. 5080 chat/Comfy yield gpu5080.lock to autodev CUDA/CI. 1080 Ti not for 14B or Comfy.\n";
    pp += "ok=true if CODEX-OPS no longer says mask Comfy as the default autodev grant.\n";
    let pol = agent(pp, #{
        label: "roles-doc",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Wire");
    let wp = "";
    wp += hard;
    wp += "model-router: local/code host_pref stays akula-prime sticky. Chat aliases that FIT 5080 may prefer gpu5080 when lock idle; still fall back 3090 leftover only if 14B not loaded. ";
    wp += "Timeshare/queue: autodev prio numeric higher than webui/comfy. Lab status shows roles. Do not unmask Comfy here if keepup wrap is not in place — gap that. ";
    wp += "Document helper fan-out in model-router (n_helpers / leftover). Dry-run n=1 -> 3090; n=3 embed helpers -> not a second 14B. ";
    wp += "Commit autodev feat/agent-harness; do not merge skip-theatre. FINAL json only.\n";
    wp += "ok=true if router JSON host_pref matches the role table.\n";
    let wired = agent(wp, #{
        label: "router-prio",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if fit != () && fit.success && fit.output.ok == true {
        if pol != () && pol.success && pol.output.ok == true {
            if wired != () && wired.success && wired.output.ok == true {
                sat = true;
            }
        }
    }
    complete(#{
        summary: if sat { "GPU roles: 3090 autodev, 5080 chat/comfy yielding lock" } else { "GPU roles incomplete" },
        sat: sat,
        apply: true,
        fit: fit,
        policy: pol,
        wire: wired,
    });
}

phase("Wire");
let vp2 = "";
vp2 += "Read-only fit. ok=true if CSD-GPU-ROLES.md exists. Re-run args.apply=true to patch router.\n";
let ver2 = agent(vp2, #{
    label: "fit-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "fit-only; re-run args.apply=true",
    apply: false,
    fit: fit,
    verify: ver2,
});
