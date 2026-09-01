let meta = #{
    name: "csd-kb-refresh-deployed",
    description: "Index deployed 3-GPU/1080 Ti/lab/autodev state into akula-csd-kb for autodev. Never operator or model vaults. Never mix 384-d.",
    when_to_use: "1080 Ti guest is live; autodev needs current RAG/Obsidian on deployed AI/lab facts.",
    phases: [
        #{ title: "Pack", detail: "notes from live docs + guest/router state" },
        #{ title: "Index", detail: "csd-kb-index 5080 exclusive-seq; never pause 3090" },
        #{ title: "Steer", detail: "autodev context: 1080ti live, 3 backends, region-pretrain" },
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
hard += "HARD: Write ONLY akula-csd-kb (experiment plane). NEVER tzervas-dev-kb. NEVER akula-model-kb. ";
hard += "Never mix 384-d Qdrant collections. Never pause 3090 LocalAI. CUDA index on 5080 exclusive-seq (gpu5080.lock). ";
hard += "1080 Ti guest is retrieve-index-light at 192.168.1.243, not the CUDA indexer. gpu5080 host 192.168.1.251.\n";
hard += "Never GitHub. Never 0.0.0.0 WAN. Identity autodev. 256 specialists is NOT a live target (future G-SHARE example).\n";

phase("Pack");
let pp = "";
pp += hard;
pp += "Read STATUS.md, CSD-CLUSTER-FLEX.md, CSD-1080TI-RAG.md, CSD-1080TI-ISOLATE.md, CSD-GROK-PING.md, CSD-BRAIN-REGIONS.md, CSD-SCALE-LADDER.md, CSD-AUTODEV-HANDOFF.md, CSD-GPU-ECO.md if present. ";
pp += "ssh guest 192.168.1.243 nvidia-smi -L; host .251 smi; router live flags. ";
pp += "Write a short pack docs/program/CONTEXT-PACK-DEPLOYED-2026-08-31.md: 3 backends, VFIO guest, RAID0 /bulk, eco, need_grok mailbox, region-pretrain next, hf/autodev missing. ";
pp += "Copy durable notes into akula-csd-kb Obsidian only (csd-kb-setup paths). ok=true if pack exists.\n";
let pack = agent(pp, #{
    label: "pack-deployed",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Index");
    let ip = "";
    ip += hard;
    ip += "./scripts/csd-kb-index (5080 exclusive-seq). Do not docker stop akula-localai. ";
    ip += "Keyword-cpu :8091/:8092 if that is the retrieve path. 1080 Ti may serve embed/retrieve AFTER index, not instead of 5080 CUDA index.\n";
    ip += "ok=true if index job ran or was honestly queued. evidence=receipt path.\n";
    let indexed = agent(ip, #{
        label: "index-csd-kb",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Steer");
    let sp = "";
    sp += hard;
    sp += "Ensure autodev-loop heartbeat/steer knows: gpu5080-1080ti live=true retrieve-index-light; cluster 3090+5080+guest; next_goal region-pretrain; need_grok mailbox. ";
    sp += "Do not change next_goal away from region-pretrain unless already P1-09 and that is still the board. ";
    sp += "Commit pack as autodev feat/agent-harness; do not merge skip-theatre.\n";
    sp += "FINAL json fence only: ok, notes, evidence. ok=true if loop still active.\n";
    let steered = agent(sp, #{
        label: "steer-state",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if pack != () && pack.success && pack.output.ok == true {
        if indexed != () && indexed.success && indexed.output.ok == true {
            if steered != () && steered.success && steered.output.ok == true {
                sat = true;
            }
        }
    }
    complete(#{
        summary: if sat { "akula-csd-kb refreshed for autodev" } else { "kb refresh incomplete" },
        sat: sat,
        apply: true,
        pack: pack,
        index: indexed,
        steer: steered,
    });
}

phase("Steer");
let vp2 = "";
vp2 += "Read-only. ok=true if pack exists. Re-run args.apply=true to index.\n";
let ver2 = agent(vp2, #{
    label: "pack-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "pack-only; re-run args.apply=true",
    apply: false,
    pack: pack,
    verify: ver2,
});
