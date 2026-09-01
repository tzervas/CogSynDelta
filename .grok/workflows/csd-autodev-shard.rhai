let meta = #{
    name: "csd-autodev-shard",
    description: "Autodev may share 3090 if weights+KV fit, pull 5080/1080 Ti by subagent count, VRAM, and arch features. Never dual 14B. Autodev still preempts chat/Comfy.",
    when_to_use: "Roles-priority is in flight; autodev needs sharded/shared GPUs for parallel subagents, not only one 14B on 3090.",
    phases: [
        #{ title: "Rules", detail: "fit KV+weights; fan-out; feature tags" },
        #{ title: "Router", detail: "helper placement by leftover VRAM and sm_*" },
        #{ title: "Verify", detail: "dry-run 1 vs N subagents; no dual 14B" },
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
    },
};

let do_apply = false;
if args != () {
    if args.apply == true { do_apply = true; }
}

let hard = "";
hard += "HARD: Never dual 14B. Autodev/local/code preempts OWUI chat and Comfy. 3090 14B may stay loaded; leftover VRAM may hold helpers only if weights+KV FITs (router safeguard). ";
hard += "Subagent fan-out: 1 latency-tight worker stays on 3090; parallel/ok-lan helpers may land 5080 share-small (lock idle) or 1080 Ti guest (RAG/light, Pascal, no FP8). ";
hard += "Need sm_120/FP8/CUDA train/CI -> 5080 exclusive-seq, Comfy waits on gpu5080.lock. ";
hard += "Need decode-tight 14B -> 3090 only. Tensor-wire G-POOL is NOT implemented; this is placement+share-small, not layer-split RPC. ";
hard += "256 specialists is NOT live. Never pause LocalAI container. Never GitHub. gpu5080=.251. Identity autodev.\n";

phase("Rules");
let rp = "";
rp += hard;
rp += "Write docs/program/CSD-AUTODEV-SHARD.md: decision table (n_subagents, vram_free, need_cuda, need_fp8) -> hosts. ";
rp += "Read model-router.json v2 share-small/leftover. Do not invent RPC. FINAL json ok/notes/evidence.\n";
let rules = agent(rp, #{
    label: "shard-doc",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Router");
    let wp = "";
    wp += hard;
    wp += "Update model-router + csd-gpu-plan comments/fields: autodev_parallel helpers host_pref leftover-3090 then 5080 share-small then 1080ti. ";
    wp += "csd-autodev-loop may call csd-model-router pick with n_helpers. Lab feed can show planned hosts. Commit autodev; no skip-theatre merge.\n";
    wp += "ok=true if JSON documents helper fan-out. FINAL json only.\n";
    let wired = agent(wp, #{
        label: "router-helpers",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Verify");
    let vp = "";
    vp += hard;
    vp += "Dry-run: n=1 -> 3090 local/code; n=3 helpers embed -> not a second 14B; cuda-eval -> 5080. LocalAI still loaded. evidence=pick JSON.\n";
    let ver = agent(vp, #{
        label: "verify-fanout",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if rules != () && rules.success && rules.output.ok == true {
        if wired != () && wired.success && wired.output.ok == true {
            if ver != () && ver.success && ver.output.ok == true {
                sat = true;
            }
        }
    }
    complete(#{
        summary: if sat { "autodev shard/share policy live" } else { "autodev shard incomplete" },
        sat: sat,
        apply: true,
        rules: rules,
        router: wired,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += "Read-only. ok=true if CSD-AUTODEV-SHARD.md exists. Re-run args.apply=true.\n";
let ver2 = agent(vp2, #{
    label: "doc-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "rules-only; re-run args.apply=true",
    apply: false,
    rules: rules,
    verify: ver2,
});
