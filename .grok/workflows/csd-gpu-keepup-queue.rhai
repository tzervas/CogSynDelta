let meta = #{
    name: "csd-gpu-keepup-queue",
    description: "Keep Comfy and Open WebUI up. Deconflict 5080 via gpu5080.lock/timeshare so autodev and media queue, never hard-pause. Route jobs, finish, free, next.",
    when_to_use: "Comfy is masked for autodev; operator wants all services running with intelligent GPU scheduling.",
    phases: [
        #{ title: "Policy", detail: "queue not mask; autodev prio; 3090 stays loaded" },
        #{ title: "Wire", detail: "Comfy ExecStart wrapped in gpu5080.lock; OWUI unchanged" },
        #{ title: "Verify", detail: "unmask only after lock wrap; autodev still acquires lock" },
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
hard += "HARD: Do NOT hard-pause 3090 LocalAI. Do NOT docker stop akula-localai. Open WebUI at ai.vectorweight.com stays up. ";
hard += "Do NOT unmask Comfy until ExecStart (or a drop-in) acquires /home/tzervas/akula-harness/scripts/gpu5080-lock the same as CUDA CI. ";
hard += "Autodev CUDA/CI still uses that lock; Comfy waits; when lock releases Comfy or next autodev job runs. Finish-then-free. ";
hard += "Route: 3090=local/code+OWUI chat; 5080=CUDA/CI/Comfy exclusive-seq via lock; 1080 Ti guest=RAG/light. Never dual 14B. ";
hard += "Never 0.0.0.0 WAN. Never GitHub. gpu5080=192.168.1.251. Identity autodev. feat/agent-harness.\n";
hard += "Tensor-wire G-POOL is NOT this work. 256 specialists is NOT live. ";
hard += "Do not start a new shard workflow. Autodev/local/code preempts chat/Comfy on the same lock. ";
hard += "3090 may share leftover KV+weights for helpers; 5080 Comfy/chat yield exclusive-seq. Pause rather than race lab-live-pool edits to the same console.\n";

phase("Policy");
let pp = "";
pp += hard;
pp += "Write docs/program/CSD-GPU-KEEPUP.md: services stay up; gpu5080.lock is the mutex; timeshare JSON on gpu5080 is the queue; autodev prio > OWUI image > idle share-small. ";
pp += "OWUI chat uses 3090 and must not take 5080. Image/edit waits on lock. Update CODEX-OPS.md: stop recommending mask; recommend lock wrap. ";
pp += "FINAL json fence ok/notes/evidence. ok=true if the keepup doc exists.\n";
let pol = agent(pp, #{
    label: "keepup-doc",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Wire");
    let wp = "";
    wp += hard;
    wp += "On gpu5080: systemd drop-in for akula-comfyui so it cannot use the GPU without gpu5080-lock (wrapper script: lock then ExecStart, release on exit). ";
    wp += "Unmask+start Comfy ONLY after that drop-in is installed and daemon-reload. If lock held by autodev, Comfy process may wait (TimeoutStart may need to be long) or stay active but idle of CUDA — prefer queue-at-start: stay running HTTP, GPU alloc on first prompt via lock. ";
    wp += "If wrapping the whole daemon is too coarse, document a Comfy queue hook; do not leave Comfy free-running on the 5080. ";
    wp += "OWUI: no restart unless env must change; IMAGE engine still comfy LAN URL. ";
    wp += "Autodev/gpu-5080.yml already uses LOCK=... keep that.\n";
    wp += "ok=true if drop-in or wrapper file exists on host AND unit is unmasked or honestly still masked with gap=need-operator-unmask.\n";
    let wired = agent(wp, #{
        label: "wire-lock",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Verify");
    let vp = "";
    vp += hard;
    vp += "ssh .251: systemctl is-enabled akula-comfyui (not masked if wrap ready); flock test that two lock holders serialize. ";
    vp += "3090 LocalAI /v1/models still 200. Lab status comfy field. Commit docs+units deploy/gpu5080 as autodev; do not merge skip-theatre.\n";
    vp += "ok=true if LocalAI up AND lock wrapper is in git. Unmask may still be operator-once if sudo needed.\n";
    let ver = agent(vp, #{
        label: "verify-keepup",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if pol != () && pol.success && pol.output.ok == true {
        if wired != () && wired.success && wired.output.ok == true {
            if ver != () && ver.success && ver.output.ok == true {
                sat = true;
            }
        }
    }
    complete(#{
        summary: if sat { "Comfy/OWUI keep-up queue wired" } else { "keep-up queue incomplete" },
        sat: sat,
        apply: true,
        policy: pol,
        wire: wired,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += "Read-only. ok=true if keepup doc exists. Re-run args.apply=true to wrap Comfy.\n";
let ver2 = agent(vp2, #{
    label: "doc-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "policy-only; re-run args.apply=true",
    apply: false,
    policy: pol,
    verify: ver2,
});
