let meta = #{
    name: "csd-1080ti-isolate",
    description: "1080 Ti lspci-only: diagnose nvidia bind vs 5080 driver. Containers share one kernel module; VFIO VM if Pascal will not bind. 5080 stays on the host.",
    when_to_use: "1080 Ti in lspci but missing from nvidia-smi on gpu5080 (.251). After or beside bringup-3.",
    phases: [
        #{ title: "Diagnose", detail: "lspci -k, dmesg, smi; open vs proprietary" },
        #{ title: "Decide", detail: "same-driver bind vs VFIO guest" },
        #{ title: "Apply", detail: "bind if safe; else 1080 Ti VFIO VM only" },
        #{ title: "Verify", detail: "1080 Ti in some nvidia-smi; 5080 host CI intact" },
    ],
};

let ok_schema = #{
    "type": "object",
    "required": ["ok", "notes", "evidence"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
        "evidence": #{ "type": "string" },
        "need_vfio": #{ "type": "boolean" },
    },
};

let do_apply = false;
if args != () {
    if args.apply == true { do_apply = true; }
}

let hard = "";
hard += "HARD: gpu5080 is 192.168.1.251 (router static). ssh tzervas@192.168.1.251 / Host gpu5080. Not .252. ";
hard += "Do NOT pause 3090 LocalAI. Do NOT unmask Comfy. Do NOT VFIO the RTX 5080 — host CUDA CI and gpu-5080.yml need it. ";
hard += "Do NOT power off. Never GitHub bot push. Never main/staging/develop/dev. Never 0.0.0.0 WAN.\n";
hard += "Fact: Docker/Podman GPU passthrough uses the HOST nvidia.ko. Two driver versions on one kernel is not a container fix. ";
hard += "VFIO + a VM is what can run an older Pascal driver next to Blackwell 610. ";
hard += "Scope: the gpu5080 box only (5080 + 1080 Ti). Do not VFIO or containerize the 3090 on akula-prime. ";
hard += "Even if the host driver binds both cards, still isolate ON THAT BOX: 1080 Ti RAG in its own container/VM (device UUID only, no 5080), 5080 CUDA CI stays on the host/lock. ";
hard += "Sandbox: RAG cannot see 5080 VRAM or the rest of gpu5080 except the RAG HDD mount. 3090 LocalAI stays a separate host.\n";
hard += "Role: 1080 Ti retrieve/index/light only. Factory eco PL when bound. Preserve /models on sda2 if touching disks.\n";

phase("Diagnose");
let dp = "";
dp += hard;
dp += "On .251: nvidia-smi -L; lspci -nnk -s 01:00.0; lspci -nnk -s 06:00.0; dmesg | grep -iE 'nvidia|gp102|1080'; cat /proc/driver/nvidia/gpus/*/information 2>/dev/null; ";
dp += "lsmod | grep nvidia; debian nvidia-driver vs nvidia-open vs 610.\n";
dp += "need_vfio=true if 06:00.0 has no nvidia kernel driver or dmesg says unsupported GPU. ";
dp += "ok=true if those commands ran. evidence=driver names + smi list.\n";
let diag = agent(dp, #{
    label: "diagnose-bind",
    capability_mode: "execute",
    output_schema: ok_schema,
});

phase("Decide");
let need_vfio = false;
if diag != () && diag.success && diag.output.need_vfio == true {
    need_vfio = true;
}
let decp = "";
decp += hard;
decp += "Write docs/program/CSD-1080TI-ISOLATE.md. ";
if need_vfio {
    decp += "Recommend VFIO for 06:00.0 1080 Ti only; 5080 stays host. Guest: older NVIDIA driver that supports Pascal, llama.cpp/embed, RAG store on the mechanical volume once RAID exists. ";
    decp += "Not a second 14B. Containers (CDI/device=UUID) only AFTER the guest or host actually binds the GPU.\n";
} else {
    decp += "Same host driver can bind both. Still document a sandbox: Podman/CDI --device UUID of 1080 Ti only for RAG; 5080 never passed in. VFIO optional hardening, not required for bind.\n";
}
decp += "ok=true if the doc states the decision and why containers cannot split kernel modules.\n";
let decide = agent(decp, #{
    label: "decide-path",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Apply");
    let ap = "";
    ap += hard;
    if need_vfio {
        ap += "Do not take the 5080. VFIO-stub 06:00.0 (+ audio 06:00.1) only if IOMMU groups are split from the 5080. ";
        ap += "If groups are merged with 5080, STOP and ok=false — do not stub the 5080. ";
        ap += "If split: bind vfio-pci, document a qemu/libvirt guest with GPU passthrough and a Pascal-capable driver. ";
        ap += "Guest MUST run stealth-leds (OpenRGB oneshot, never --server) and GPULogoBrightness=0 at boot so the GeForce logo does not come back when the VM binds the GPU. ";
        ap += "Host stealth alone is not enough. A full VM image is optional; vfio bind + XML stub + guest stealth unit is the minimum.\n";
    } else {
        ap += "Bind nvidia to 06:00.0 without unloading the 5080 driver if possible. Recheck nvidia-smi -L. ";
        ap += "Still add a gpu5080-box sandbox: Podman --device that 1080 Ti UUID only (or CDI) for RAG. 5080 not in that container. ";
        ap += "If the sandbox is a VM, install stealth-leds in the guest so LEDs stay off after bind. ";
        ap += "Do not reboot unless documented. Do not hold gpu5080.lock if CI is running. Do not touch 3090.\n";
    }
    ap += "ok=true if 1080 Ti is usable (host smi or vfio bound) OR honest blocker (shared IOMMU group).\n";
    let applied = agent(ap, #{
        label: "apply-isolate",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Verify");
    let vp = "";
    vp += hard;
    vp += "nvidia-smi -L on host must still list RTX 5080. 1080 Ti either in smi or vfio-pci in lspci -k. ";
    vp += "Commit docs on feat/agent-harness as autodev. Do not merge. evidence=smi + lspci -k.\n";
    let ver = agent(vp, #{
        label: "verify-isolate",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if diag != () && diag.success && diag.output.ok == true {
        if decide != () && decide.success && decide.output.ok == true {
            if applied != () && applied.success && applied.output.ok == true {
                if ver != () && ver.success && ver.output.ok == true {
                    sat = true;
                }
            }
        }
    }
    complete(#{
        summary: if sat { "1080 Ti isolation path applied" } else { "1080 Ti isolation incomplete" },
        sat: sat,
        apply: true,
        need_vfio: need_vfio,
        diagnose: diag,
        decide: decide,
        apply_result: applied,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += "Read-only: decide doc exists. ok=true if CSD-1080TI-ISOLATE.md written. Re-run args.apply=true to bind or VFIO.\n";
let ver2 = agent(vp2, #{
    label: "doc-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "isolate decision only; re-run args.apply=true",
    apply: false,
    need_vfio: need_vfio,
    diagnose: diag,
    decide: decide,
    verify: ver2,
});
