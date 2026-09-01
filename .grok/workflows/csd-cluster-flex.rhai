let meta = #{
    name: "csd-cluster-flex",
    description: "Autodev may use 3090+5080+1080 Ti in any fit combination (helpers, extra inference, RAG). Routing tags for arch/VRAM. Never dual 14B. Exclusive-seq still wins on 5080 CUDA.",
    when_to_use: "After handoff; router still silos 1080 Ti to RAG-only or forbids helpers on leftover VRAM.",
    phases: [
        #{ title: "Policy", detail: "fit table + tags sm_86/sm_120/sm_61" },
        #{ title: "Router", detail: "aliases for helper/embed/light on any host that fits" },
        #{ title: "Verify", detail: "dry-run pick; no dual 14B; 3090 LocalAI stays" },
    ],
};

let ok_schema = #{
    "type": "object",
    "required": ["ok", "notes", "evidence"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
        "evidence": #{ "type": "string" },
    },
};

let do_apply = false;
if args != () {
    if args.apply == true { do_apply = true; }
}

let hard = "";
hard += "HARD: Never dual 14B. Never pause 3090 LocalAI for RAG search. Never 0.0.0.0 WAN. Never GitHub. ";
hard += "5080 exclusive-seq (gpu5080.lock / CUDA CI / train) still preempts helpers on that card. ";
hard += "1080 Ti is Pascal sm_61 ~11GiB in a VFIO guest when live — no FP8/FP4, no sm_120 CUDA. ";
hard += "3090 Ampere sm_86 ~24GiB: local/code 14B may stay loaded; leftover VRAM may hold embed/8B helpers. ";
hard += "5080 Blackwell sm_120 ~16GiB: CUDA/eval; share-small embed when lock idle. ";
hard += "Autodev MAY combine cards for helpers, extra inference, RAG, light GGUF — any mix that FITs and respects tags. ";
hard += "Latency-tight stays on one card; ok-lan may split. Pool ~3090+5080+1080ti VRAM is heterogeneous, not one MIG.\n";
hard += "gpu5080 host 192.168.1.251. Identity autodev. feat/agent-harness. ";
hard += "LEAVE the 1080 Ti alone until csd-1080ti-guest AND csd-1080ti-persist have finished AND guest nvidia-smi lists GTX 1080 Ti AND live=true. ";
hard += "Until then: no extra VFIO, no extra stealth on GP102, no RAG on 1080 Ti, no router promote. 3090+5080 helper policy is fine.\n";

phase("Policy");
let pp = "";
pp += hard;
pp += "Write docs/program/CSD-CLUSTER-FLEX.md: fit table (alias, min VRAM, arch tags, hosts allowed, exclusive-seq conflict). ";
pp += "Read config/model-router.json v2. Do not invent 10x. ok=true if the doc exists.\n";
let pol = agent(pp, #{
    label: "policy-doc",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Router");
    let rp = "";
    rp += hard;
    rp += "Update config/model-router.json: helper/embed/light aliases may list akula-prime leftover, gpu5080 share-small, and gpu5080-1080ti when live=true. ";
    rp += "Keep never_dual_14b. Keep 14B default on 3090. RAG retrieve-index prefers 1080 Ti when live else 5080 embed. ";
    rp += "csd-model-router pick must refuse a placement that OOMs or collides with exclusive-seq. ";
    rp += "Lab /api/status cluster snapshot should show three backends. Grafana group tags already akula-rag/csd-autodev.\n";
    rp += "Commit autodev feat/agent-harness; do not merge skip-theatre.\n";
    rp += "ok=true if router JSON allows helpers on more than one host. evidence=alias hosts.\n";
    let router = agent(rp, #{
        label: "router-flex",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Verify");
    let vp = "";
    vp += hard;
    vp += "Dry-run csd-model-router for: local/code (3090), cuda-eval (5080), rag-index (1080ti if live else 5080), helper embed (any leftover). ";
    vp += "Refuse two 14B. 3090 LocalAI still loaded. evidence=pick JSON.\n";
    let ver = agent(vp, #{
        label: "verify-picks",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if pol != () && pol.success && pol.output.ok == true {
        if router != () && router.success && router.output.ok == true {
            if ver != () && ver.success && ver.output.ok == true {
                sat = true;
            }
        }
    }
    complete(#{
        summary: if sat { "cluster flex routing live" } else { "cluster flex incomplete" },
        sat: sat,
        apply: true,
        policy: pol,
        router: router,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += "Read-only policy. ok=true if CSD-CLUSTER-FLEX.md exists. Re-run args.apply=true to patch router.\n";
let ver2 = agent(vp2, #{
    label: "doc-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "policy doc; re-run args.apply=true",
    apply: false,
    policy: pol,
    verify: ver2,
});
