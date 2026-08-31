let meta = #{
    name: "csd-lab-iac-o11y",
    description: "After 1080 Ti bringup/eco/isolate: refresh labels/ns/annotations, scrape .251, capture IaC/CaC in private Forgejo. Never GitHub.",
    when_to_use: "gpu5080 is on .251; RAID/eco/isolate have run or stalled. Need Grafana labels plus git-managed lab config.",
    phases: [
        #{ title: "Inventory", detail: "what is live vs still future_hosts" },
        #{ title: "Labels", detail: "taxonomy + scrape + Grafana vars for 1080ti/RAID/eco" },
        #{ title: "CaC", detail: "private Forgejo: o11y, router, ssh, units, vfio/podman" },
        #{ title: "Verify", detail: "label_values host=.251 path; no GitHub; no .252" },
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
hard += "HARD: gpu5080 LAN is 192.168.1.251 (operator router static). Never write 192.168.1.252 as current. ";
hard += "Private git only: Forgejo tzervas/CogSynDelta (feat/agent-harness) and akula-ai-platform if already a Forgejo remote. Never GitHub bot push. ";
hard += "Never main/staging/develop/dev. Never pause 3090 LocalAI. Never VFIO the 5080. Never 0.0.0.0 WAN. ";
hard += "Identity autodev + CSD vault. Do not merge skip-theatre.\n";
hard += "IaC/CaC = git-tracked: deploy/o11y, Grafana dashboards/provisioning, model-router.json, systemd units, ";
hard += "podman/quadlet/CDI for 1080 Ti RAG sandbox, vfio XML if isolate created it, fstab/mdadm notes for dual-3TB RAID. ";
hard += "Do not dump secrets. Do not copy operator ~/.secrets.\n";

phase("Inventory");
let ip = "";
ip += hard;
ip += "Read docs/program/CSD-O11Y-TAXONOMY.md, CSD-1080TI-RAG.md, CSD-GPU-ECO.md, CSD-1080TI-ISOLATE.md if present, config/model-router.json, deploy/o11y.\n";
ip += "ssh tzervas@192.168.1.251: nvidia-smi -L, ip -4, lsblk, mdstat, lspci 06:00.0 driver. Honest: 1080 Ti live vs lspci-only vs vfio.\n";
ip += "gaps=files still saying .252, missing group=akula-rag, missing host label for 1080ti. ok=true if inventory from files+smi.\n";
let inv = agent(ip, #{
    label: "inv-live",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Labels");
    let lp = "";
    lp += hard;
    lp += "Update CSD-O11Y-TAXONOMY.md: host=gpu5080 ip .251; path lab.gpu5080.index.1080ti; ns=index group=akula-rag kind=gpu; ";
    lp += "RAID volume annotation mount=...; eco=factory-default-pl. Isolation env: host vs 1080ti-sandbox (container/VM).\n";
    lp += "deploy/o11y scrape targets 192.168.1.251 not .252. Relabel gpu name 1080 Ti if series exist — do not invent PromQL.\n";
    lp += "Grafana vars already env/host/ns/group/service — add path if easy. No Tempo. SMTP bind unchanged.\n";
    lp += "ok=true if taxonomy+.yml have .251 and akula-rag. evidence=paths.\n";
    let labels = agent(lp, #{
        label: "labels-o11y",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("CaC");
    let cp = "";
    cp += hard;
    cp += "Collect into CogSynDelta deploy/ (and akula-harness only if that tree is the live path): ";
    cp += "vm-scrape.yml, promtail, grafana dashboards already in deploy/grafana, model-router future_hosts, ";
    cp += "any nvidia-factory-limits.service, gpu5080-1080ti podman quadlet/CDI, libvirt XML. ";
    cp += "README in deploy/ explaining how to apply (homelab grafana vs gpu5080). ";
    cp += "Commit+push autodev feat/agent-harness. New PR vs main if #2 already merged. Do not merge. Never GitHub.\n";
    cp += "ok=true if push succeeded and no plaintext tokens. evidence=sha + file list.\n";
    let cac = agent(cp, #{
        label: "cac-forgejo",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Verify");
    let vp = "";
    vp += hard;
    vp += "rg 192.168.1.252 in tracked docs/config/deploy — leftover only in historical notes. ";
    vp += "If VM reachable: label_values host or instance containing 251. Do not claim 1080 Ti metrics if smi still lacks it.\n";
    vp += "ok=true if .251 is canonical in CODEX-OPS + scrape + taxonomy. evidence=grep counts.\n";
    let ver = agent(vp, #{
        label: "verify-iac",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if inv != () && inv.success && inv.output.ok == true {
        if labels != () && labels.success && labels.output.ok == true {
            if cac != () && cac.success && cac.output.ok == true {
                if ver != () && ver.success && ver.output.ok == true {
                    sat = true;
                }
            }
        }
    }
    complete(#{
        summary: if sat { "lab IaC/CaC and o11y labels updated" } else { "lab IaC/o11y incomplete" },
        sat: sat,
        apply: true,
        inventory: inv,
        labels: labels,
        cac: cac,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += "Read-only inventory. ok=true if inv.ok. Re-run args.apply=true after bringup-3/isolate settle.\n";
let ver2 = agent(vp2, #{
    label: "inv-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "inventory only; re-run args.apply=true to write labels+IaC",
    apply: false,
    inventory: inv,
    verify: ver2,
});
