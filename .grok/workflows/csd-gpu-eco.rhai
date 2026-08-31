let meta = #{
    name: "csd-gpu-eco",
    description: "Factory eco: all GPUs default clocks/power limits, persistence on, no locked P0. High wattage only under real CUDA/LocalAI load. 3090 stays serving.",
    when_to_use: "GPUs burning high idle watts, clocks locked, or after 1080 Ti seating. Not a custom undervolt.",
    phases: [
        #{ title: "Audit", detail: "pm, PL, clocks, pstate per GPU" },
        #{ title: "Apply", detail: "factory default PL, unlock clocks, persistence" },
        #{ title: "Verify", detail: "idle pstate not P0; LocalAI still up" },
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
hard += "HARD: Factory clocks and factory power limits only — not a custom undervolt curve. ";
hard += "Unlock any --lock-gpu-clocks / -ac. nvidia-smi -pm 1. -pl to the GPU's default limit (query power.default_limit). ";
hard += "Idle must be allowed to drop P-states (P8/P5). High wattage only when CUDA or LocalAI is actually running.\n";
hard += "Do NOT pause 3090 LocalAI. Do NOT drop 3090 PL so low that local/code cannot boost under load. ";
hard += "Do NOT unmask Comfy. Do NOT power off. Never 0.0.0.0 WAN. Never GitHub.\n";
hard += "Hosts: akula-prime 3090 Ti (local); gpu5080 at 192.168.1.251 (router static; Host gpu5080 / tzervas@192.168.1.251). Not .252.\n";
hard += "If .251 SSH fails, still eco the 3090 and record gaps. Persistence + default PL on each visible UUID (5080 and 1080 Ti if nvidia-smi lists them).\n";
hard += "Optional durable: a small systemd oneshot per host that applies this at boot. Do not install nvidia-settings GUI.\n";

phase("Audit");
let ap = "";
ap += hard;
ap += "On each reachable GPU: nvidia-smi -q -d CLOCK,POWER,PERFORMANCE | relevant; persistence_mode; power.limit vs default_limit vs max; any locked clocks.\n";
ap += "evidence=table host,name,uuid,pstate,power_draw,pl,default_pl,clocks_locked. ok=true if at least 3090 was queried.\n";
let audit = agent(ap, #{
    label: "audit-clocks",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Apply");
    let jobs = [];
    let p3090 = "";
    p3090 += hard;
    p3090 += "akula-prime only. nvidia-smi -pm 1; reset gpu/mem clocks if locked; -pl $(default_limit) for GPU0 3090 Ti. ";
    p3090 += "Do not stop docker/LocalAI. evidence=smi after. ok=true if pm=Enabled and pl==default and no lock.\n";
    jobs.push(#{
        prompt: p3090,
        label: "eco-3090",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let p5080 = "";
    p5080 += hard;
    p5080 += "ssh -o BatchMode=yes tzervas@192.168.1.251 (Host gpu5080). EACH nvidia-smi GPU: -pm 1, unlock clocks, -pl default_limit. ";
    p5080 += "1080 Ti may be lspci-only GP102 06:00.0 — eco the 5080 anyway. Skip if gpu5080.lock is held.\n";
    jobs.push(#{
        prompt: p5080,
        label: "eco-5080-box",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let applied = parallel(jobs);

    phase("Verify");
    let vp = "";
    vp += hard;
    vp += "3090: LocalAI still answers /v1/models; idle power_draw well below default TDP (honest W). ";
    vp += "gpu5080 if up: both cards persistence on, pstate not forced P0 while util=0. ";
    vp += "Write docs/program/CSD-GPU-ECO.md with the commands. Commit autodev feat/agent-harness if docs change. Do not merge.\n";
    vp += "ok=true if 3090 eco applied and LocalAI up.\n";
    let ver = agent(vp, #{
        label: "verify-eco",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let a0 = ();
    let a1 = ();
    if applied != () {
        if applied.len() > 0 { a0 = applied[0]; }
        if applied.len() > 1 { a1 = applied[1]; }
    }
    let sat = false;
    if audit != () && audit.success && audit.output.ok == true {
        if a0 != () && a0.success && a0.output.ok == true {
            if ver != () && ver.success && ver.output.ok == true {
                sat = true;
            }
        }
    }
    complete(#{
        summary: if sat { "factory eco applied" } else { "factory eco incomplete" },
        sat: sat,
        apply: true,
        audit: audit,
        prime: a0,
        gpu5080_box: a1,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += "Read-only: audit exists. ok=true if 3090 clocks were read. Re-run args.apply=true to set factory eco.\n";
let ver2 = agent(vp2, #{
    label: "audit-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "eco audit-only; re-run args.apply=true",
    apply: false,
    audit: audit,
    verify: ver2,
});
