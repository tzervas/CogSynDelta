let meta = #{
    name: "csd-program-drive",
    description: "After 1080 Ti guest live: refresh 3-GPU flex, autodev cohort, region datasets, tiny-then-scale CSD to private HF, Grafana email, lab Goals/Todos. Grok=research.",
    when_to_use: "Guest/persist done or 1080 Ti still not live; operator wants remaining memory-gate/CSD closed with autodev on all GPUs when ready.",
    phases: [
        #{ title: "Gate", detail: "guest nvidia-smi? else leave 1080 Ti, still drive 3090+5080" },
        #{ title: "Followup", detail: "flex+handoff refresh only if live; lab Goals/Todos" },
        #{ title: "Regions", detail: "dataset research + PoC router/regions honesty" },
        #{ title: "Scale", detail: "tiny nail; HF private sizes; Grafana email when real" },
    ],
};

let ok_schema = #{
    "type": "object",
    "required": ["ok", "notes", "evidence"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
        "evidence": #{ "type": "string" },
        "live_1080ti": #{ "type": "boolean" },
    },
};

let do_apply = false;
if args != () {
    if args.apply == true { do_apply = true; }
}

let hard = "";
hard += "HARD: autodev implements. Hosted Grok in this parent is research/steer/unblock only. ";
hard += "Never dual 14B. Never pause 3090 LocalAI. Never unmask Comfy. Never GitHub bot push. ";
hard += "Never main/staging/develop/dev. Never 0.0.0.0 WAN. Never mix 384-d. Never write operator/model vaults. ";
hard += "Merge only when required checks ran. gpu5080=192.168.1.251. 1080 Ti guest LAN may be 192.168.1.243. ";
hard += "Do not claim VL-JEPA/quantum/10x unless STATUS.md or a test you ran supports it. Python-first memory-gate then CSD poc regions/router.\n";
hard += "1080 Ti: if guest nvidia-smi is missing, DO NOT promote live, DO NOT ham on that GPU. 3090+5080 cohort is OK. ";
hard += "When live: simultaneous helpers/inference/RAG across all three that FIT; exclusive-seq still wins on 5080 CUDA.\n";
hard += "HF publish: private tzervas/cogsyndelta* only with hf/autodev in CSD vault. If missing, document mint steps — never copy gpu/huggingface-token.\n";

phase("Gate");
let gp = "";
gp += hard;
gp += "ssh .251 virsh list --all; try guest SSH 192.168.1.243 nvidia-smi -L; router live flag; csd-autodev-loop active?; steer next_goal.\n";
gp += "live_1080ti=true ONLY if guest smi lists GTX 1080 Ti. ok=true if 3090+5080 still reachable.\n";
let gate = agent(gp, #{
    label: "gate-guest",
    capability_mode: "execute",
    output_schema: ok_schema,
});

let live = false;
if gate != () && gate.success && gate.output.live_1080ti == true {
    live = true;
}

if do_apply {
    phase("Followup");
    let fp = "";
    fp += hard;
    if live {
        fp += "1080 Ti IS live. Refresh model-router so helpers/embed/light/RAG may land on the guest; cluster-flex dry-run must pick 1080ti for rag-index when it FITs. ";
        fp += "Restart autodev-loop if needed so it sees three backends. Grafana group=akula-rag series if they exist.\n";
    } else {
        fp += "1080 Ti NOT live. Do not change live=false. Still add lab Goals/Todos. Autodev stays 3090+5080.\n";
    }
    fp += "CSD lab console (scripts/csd-lab-console, code.vectorweight.com): add a Goals/Todos tab/page showing PHASE-1 board + CSD_STEER next_goal + last loop heartbeat + reasoning/notes from steer JSON. JSON API /api/goals. No Open WebUI duplicate.\n";
    fp += "ok=true if /api/goals exists and UI tab renders from that JSON.\n";
    let follow = agent(fp, #{
        label: "followup-lab",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Regions");
    let rp = "";
    rp += hard;
    rp += "CSD BRAIN REGIONS (not org units): specialized submodels of one mind. STATUS.md: MoE-adjacent; expert=region; gate=softmax top-k. ";
    rp += "Curriculum MUST be: (1) pretrain EACH region on its own specialized dataset until it can do its job, ";
    rp += "(2) train interconnect/router so regions activate selectively, (3) THEN train the assembled whole-model, ";
    rp += "(4) scale parameter count tiny->small->full, private HF per size. Do not skip region-pretrain. ";
    rp += "Live PoC is two regions (LatentVAE + ResidualMLP) + softmax router — not a full brain. Next increment: one more region OR better region-pretrain, not 14B. ";
    rp += "Research PUBLIC datasets per region (license, size, why). Write docs/program/CSD-REGION-DATASETS.md. No 10x, no VL-JEPA unless STATUS agrees.\n";
    rp += "ok=true if the datasets doc has at least 3 cited sources.\n";
    let regions = agent(rp, #{
        label: "region-datasets",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Scale");
    let sp = "";
    sp += hard;
    sp += "Write docs/program/CSD-SCALE-LADDER.md: tiny CPU/PoC -> small CUDA on 5080 -> larger when measured. Each size: train recipe, eval, private HF repo name tzervas/cogsyndelta-<size> when hf/autodev exists. ";
    sp += "Grafana: one alert rule to maintainers-email when a file or metric you actually create means 'ladder rung N green' — do not fire a fake sat. ";
    sp += "If hf/autodev missing, gaps=mint HF. Commit autodev feat/agent-harness; do not merge skip-theatre.\n";
    sp += "ok=true if scale doc + goals API + honest HF gap.\n";
    let scale = agent(sp, #{
        label: "scale-alert",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    let sat = false;
    if gate != () && gate.success && gate.output.ok == true {
        if follow != () && follow.success && follow.output.ok == true {
            if regions != () && regions.success && regions.output.ok == true {
                if scale != () && scale.success && scale.output.ok == true {
                    sat = true;
                }
            }
        }
    }
    complete(#{
        summary: if sat { "program drive artifacts live" } else { "program drive incomplete" },
        sat: sat,
        apply: true,
        live_1080ti: live,
        gate: gate,
        followup: follow,
        regions: regions,
        scale: scale,
    });
}

phase("Followup");
let vp2 = "";
vp2 += "Read-only gate. ok=true if gate.ok. Re-run args.apply=true for lab goals + datasets + scale ladder.\n";
let ver2 = agent(vp2, #{
    label: "gate-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "gate-only; re-run args.apply=true",
    apply: false,
    live_1080ti: live,
    gate: gate,
    verify: ver2,
});
