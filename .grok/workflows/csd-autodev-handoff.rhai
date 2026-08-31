let meta = #{
    name: "csd-autodev-handoff",
    description: "Hand development to autodev cluster: loop+lab+Grafana. Hosted Grok is research/unblocker only. Drive memory-gate + CSD Python-first. 1080 Ti RAG when guest live.",
    when_to_use: "After 1080 Ti VFIO/guest/persist; operator wants autodev to jam P1 and CSD while Grok stays off implementer quota.",
    phases: [
        #{ title: "Cluster", detail: "3090/5080/1080ti-guest health; router roles" },
        #{ title: "Steer", detail: "next_goal memory-gate/CSD; lab+Grafana cluster view" },
        #{ title: "Loop", detail: "csd-autodev-loop as autodev; workflows for jobs it cannot run" },
        #{ title: "Verify", detail: "loop active; Grok not merging skip-theatre" },
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

let do_apply = false;
if args != () {
    if args.apply == true { do_apply = true; }
}

let hard = "";
hard += "HARD: Identity autodev, CSD vault TOKEN=git/autodev. Never GitHub bot push. Never main/staging/develop/dev. ";
hard += "Never pause 3090 LocalAI. Never unmask Comfy. Never 0.0.0.0 WAN. Never write operator/model vaults. ";
hard += "Never mix 384-d Qdrant. Merge only when required checks ran and succeeded. gpu5080 is 192.168.1.251.\n";
hard += "Roles: 3090=local/code implementer; 5080=exclusive CUDA/CI; 1080 Ti guest=RAG retrieve/index/light IF guest nvidia-smi. ";
hard += "If guest not live, still start autodev on 3090+5080; RAG stays 5080-embed or prime until live=true. ";
hard += "Do not wait on or poke the 1080 Ti guest/persist runs; they own that card until sat+guest smi.\n";
hard += "Hosted Grok (this parent) is planner/research/unblocker only — do not stuff implement into the Grok chat.\n";

phase("Cluster");
let cp = "";
cp += hard;
cp += "Check: curl 3090 LocalAI /v1/models; ssh .251 nvidia-smi -L (5080 host); virsh list --all; guest smi if SSH to guest exists; ";
cp += "config/model-router.json live flags; csd-lab-console /api/status; gpu5080.lock idle?\n";
cp += "gaps=1080ti not live, loop down, grafana missing akula-rag. ok=true if 3090+5080 reachable.\n";
let cluster = agent(cp, #{
    label: "cluster-health",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Steer");
    let sp = "";
    sp += hard;
    sp += "Read docs/program/PHASE-1-TASK-BOARD.md and memory-gate remaining P1 after P1-08. Set CSD_STEER / lab POST /api/steer next_goal to the next closeable P1 ID (not P1-08). ";
    sp += "Update lab console cluster snapshot if missing 1080ti/guest IP. Grafana: ensure group=akula-rag and host=gpu5080 path exist; do not invent series.\n";
    sp += "Write docs/program/CSD-AUTODEV-HANDOFF.md: Grok=research only; autodev=implement via loop+Forgejo tools+workflows. ok=true if steer file/API shows the next P1.\n";
    let steer = agent(sp, #{
        label: "steer-p1",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Loop");
    let lp = "";
    lp += hard;
    lp += "systemctl --user enable --now csd-autodev-loop (or the unit on the host that runs it). TOKEN=git/autodev, GIT_USERNAME=autodev. ";
    lp += "Loop must call csd-autodev-git / csd-autodev-forgejo and may launch CogSynDelta workflows (csd-python-first-drive, csd-goal-loop) for jobs it cannot run locally. ";
    lp += "Do not dual 14B. Do not take 5080 if gpu5080.lock held. Restart only the autodev unit.\n";
    lp += "ok=true if the unit is active and logs show autodev identity. evidence=systemctl status snippet (no tokens).\n";
    let loop_r = agent(lp, #{
        label: "start-loop",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Verify");
    let vp = "";
    vp += hard;
    vp += "loop active; steer next_goal set; 3090 LocalAI up; Comfy masked; 1080ti live flag matches guest smi. ";
    vp += "Commit handoff doc as autodev feat/agent-harness; PR if needed; do not merge skip-theatre.\n";
    let ver = agent(vp, #{
        label: "verify-handoff",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if cluster != () && cluster.success && cluster.output.ok == true {
        if steer != () && steer.success && steer.output.ok == true {
            if loop_r != () && loop_r.success && loop_r.output.ok == true {
                if ver != () && ver.success && ver.output.ok == true {
                    sat = true;
                }
            }
        }
    }
    complete(#{
        summary: if sat { "autodev handoff live" } else { "autodev handoff incomplete" },
        sat: sat,
        apply: true,
        cluster: cluster,
        steer: steer,
        autoloop: loop_r,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += "Read-only cluster. ok=true if 3090 reachable. Re-run args.apply=true to steer+start loop.\n";
let ver2 = agent(vp2, #{
    label: "cluster-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "cluster-only; re-run args.apply=true to handoff",
    apply: false,
    cluster: cluster,
    verify: ver2,
});
