let meta = #{
    name: "csd-1080ti-rag-host",
    description: "1080 Ti in the 5080 box: future third GPU for RAG retrieve/index and light inference. Not live until args.live=true. Never power off the host from this run.",
    when_to_use: "Operator will add a 1080 Ti under the 5080 when that host is idle, or after it is seated and needs router/o11y wiring.",
    phases: [
        #{ title: "Window", detail: "5080 lock/Comfy/CUDA idle? do not power off" },
        #{ title: "Spec", detail: "Pascal sm_61 ~11GiB; role retrieve-index-light" },
        #{ title: "Wire", detail: "future_hosts until live; labels host+ns=index" },
        #{ title: "Verify", detail: "router ignores until live; 3090 LocalAI untouched" },
    ],
};

let ok_schema = #{
    "type": "object",
    "required": ["ok", "notes", "evidence"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
        "evidence": #{ "type": "string" },
        "idle": #{ "type": "boolean" },
    },
};

fn as_str(x) {
    if type_of(x) == "string" { return x; }
    "";
}

let do_apply = false;
let go_live = false;
if args != () {
    if args.apply == true { do_apply = true; }
    if args.live == true { go_live = true; }
}

let hard = "";
hard += "HARD: Do NOT power off gpu5080. Do NOT unmask Comfy. Do NOT pause 3090 LocalAI. ";
hard += "Do NOT treat 1080 Ti as a 5080 substitute for PoC CUDA, FP8/FP4, or dual-14B. ";
hard += "Never 0.0.0.0 WAN. Never GitHub bot push. Never main/staging/develop/dev. ";
hard += "Never write operator/model vaults. Never mix 384-d Qdrant. Identity autodev.\n";
hard += "1080 Ti is Pascal sm_61 ~11GiB GDDR5X. Factory cooler first. Same box as RTX 5080 (192.168.1.252). ";
hard += "Role: RAG retrieve/index, keyword-cpu assist, small embed, light GGUF. Not train/eval exclusive-seq.\n";

phase("Window");
let wp = "";
wp += hard;
wp += "Observe only. ssh gpu5080 BatchMode: nvidia-smi, gpu5080.lock (fuser), Comfy unit masked?, forgejo-runner-gpu-tzervas jobs.\n";
wp += "idle=true only if lock free, no CUDA process, Comfy masked, no gpu-5080 Actions running.\n";
wp += "ok=true if you reported idle truthfully. evidence=smi+lock. Never shutdown.\n";
let window = agent(wp, #{
    label: "window-idle",
    capability_mode: "execute",
    output_schema: ok_schema,
});

phase("Spec");
let sp = "";
sp += hard;
sp += "Read config/model-router.json, docs/adr/0016-consumer-gpu-share.md, docs/CODEX-OPS.md, docs/program/WHO-RUNS-WHAT.md if present.\n";
sp += "Write or update docs/program/CSD-1080TI-RAG.md: future_hosts id gpu5080-1080ti, host=gpu5080, pci below 5080, ";
sp += "labels env=lab host=gpu5080 ns=index kind=gpu group=akula-rag path=lab.gpu5080.index.1080ti.\n";
sp += "Isolation: separate CUDA_VISIBLE_DEVICES from 5080 exclusive-seq; no MIG; do not share VRAM with 5080 jobs.\n";
sp += "When live: csd-kb-index and retrieve prefer 1080 Ti; 5080 stays train/eval/Comfy. 3090 stays local/code.\n";
sp += "ok=true if the doc exists with those fields. evidence=path. Branch feat/agent-harness, not main.\n";
let spec = agent(sp, #{
    label: "spec-doc",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Wire");
    let yp = "";
    yp += hard;
    yp += "Edit config/model-router.json future_hosts (or equivalent) gpu5080-1080ti. ";
    if go_live {
        yp += "args.live=true: operator says the card is seated. nvidia-smi must show both 5080 and 1080 Ti. ";
        yp += "Then promote from future_hosts to live catalog with role retrieve-index-light. Wire csd-kb-index host override. ";
        yp += "o11y labels ns=index group=akula-rag. If smi does not show 1080 Ti, ok=false and stay future.\n";
    } else {
        yp += "args.live=false: keep future_hosts only. Router MUST ignore it for scheduling. Comment why.\n";
    }
    yp += "Commit conventional docs+config as autodev on feat/agent-harness. Never GitHub. Do not merge.\n";
    yp += "ok=true if JSON+doc match live flag. evidence=json path + smi if live.\n";
    let wired = agent(yp, #{
        label: "wire-router",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Verify");
    let vp = "";
    vp += hard;
    vp += "Router pick for rag-index / retrieve must not select 1080 Ti unless live=true AND smi lists it.\n";
    vp += "5080 still exclusive for PoC CUDA. 3090 LocalAI still loaded. evidence=router dry-run + nvidia-smi.\n";
    let ver = agent(vp, #{
        label: "verify-not-live",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if window != () && window.success && window.output.ok == true {
        if spec != () && spec.success && spec.output.ok == true {
            if wired != () && wired.success && wired.output.ok == true {
                if ver != () && ver.success && ver.output.ok == true {
                    sat = true;
                }
            }
        }
    }
    complete(#{
        summary: if sat { "1080 Ti RAG host spec wired" } else { "1080 Ti RAG host incomplete" },
        sat: sat,
        apply: true,
        live: go_live,
        window: window,
        spec: spec,
        wire: wired,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += "Read-only: docs/program/CSD-1080TI-RAG.md exists. ok=true if file present. Re-run apply=true to write future_hosts; live=true only after the card is seated.\n";
let ver2 = agent(vp2, #{
    label: "verify-doc",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "1080 Ti spec path; apply=true wires future_hosts; live=true only after install",
    apply: false,
    live: false,
    window: window,
    spec: spec,
    verify: ver2,
});
