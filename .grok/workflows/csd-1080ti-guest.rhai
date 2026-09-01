let meta = #{
    name: "csd-1080ti-guest",
    description: "Install qemu/OVMF on gpu5080, boot 1080 Ti VFIO guest with Pascal driver + stealth-leds, promote router live for RAG. 5080 stays host nvidia.",
    when_to_use: "1080 Ti is vfio-pci, XML stub exists, no guest image. RAG cannot schedule until guest nvidia-smi.",
    phases: [
        #{ title: "Prereq", detail: "qemu-system, OVMF, libvirt, vfio still on 06:00" },
        #{ title: "Guest", detail: "qcow2 + Pascal driver + stealth oneshot" },
        #{ title: "Route", detail: "live=true retrieve-index-light if guest smi lists 1080 Ti" },
        #{ title: "Verify", detail: "guest smi; 5080 host smi; RAG dry-run; no GitHub" },
    ],
};

let ok_schema = #{
    "type": "object",
    "required": ["ok", "notes", "evidence"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
        "evidence": #{ "type": "string" },
        "guest_smi": #{ "type": "boolean" },
    },
};

let do_apply = false;
if args != () {
    if args.apply == true { do_apply = true; }
}

let hard = "";
hard += "HARD: gpu5080 192.168.1.251 tzervas. VFIO already on 06:00.0/1 (IOMMU 18). Do NOT VFIO the 5080 (01:00.0). ";
hard += "Do NOT pause 3090 LocalAI. Do NOT unmask Comfy. Do NOT power off. Never GitHub. feat/agent-harness autodev.\n";
hard += "XML stub: /home/tzervas/akula-harness/config/qemu/gpu5080-1080ti-rag.xml and CogSynDelta deploy/gpu5080/libvirt/gpu5080-1080ti-rag.xml.\n";
hard += "Guest NVIDIA driver must support Pascal GP102 (proprietary, NOT host 610 open-dkms). ";
hard += "Guest MUST enable stealth-leds oneshot + GPULogoBrightness=0 at boot (never OpenRGB --server). ";
hard += "Role retrieve-index-light / RAG only. Factory eco PL in the guest. Preserve /models on sda2.\n";

phase("Prereq");
let pp = "";
pp += hard;
pp += "ssh .251: lspci -k -s 06:00.0 must be vfio-pci; 01:00.0 nvidia. apt: qemu-system-x86 qemu-utils ovmf libvirt-daemon-system virtinst. ";
pp += "IOMMU intel_iommu=on already if groups 15 vs 18 exist. Do not reboot unless documented. ";
pp += "ok=true if qemu-system-x86_64 and OVMF code.fd exist and vfio still holds 1080 Ti.\n";
let prereq = agent(pp, #{
    label: "prereq-qemu",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Guest");
    let gp = "";
    gp += hard;
    gp += "Create a small Debian guest (or reuse a local cloud image) with 1080 Ti VFIO, virtio net on the LAN ";
gp += "(bridge/macvtap enp5s0 so it gets 192.168.1.x like other inference hosts — not NAT-only usermode). ";
gp += "Disk on NVMe not wiping /models. Guest inference listen on that LAN IP (or 127.0.0.1 + host proxy). Never 0.0.0.0 WAN. ";
    gp += "Install Pascal-capable nvidia in the guest until nvidia-smi -L shows GTX 1080 Ti. ";
    gp += "Install stealth-leds (copy from akula-ai-platform/scripts/stealth-leds) as systemd oneshot on guest boot; OpenRGB CLI Off/Direct 000000; no --server. ";
    gp += "Enable virsh autostart and libvirt default so the guest returns on host reboot. Document SSH to the guest. ";
    gp += "ok=true only if guest nvidia-smi lists 1080 Ti. evidence=smi + virsh list.\n";
    let guest = agent(gp, #{
        label: "boot-guest",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Route");
    let rp = "";
    rp += hard;
    rp += "If guest smi ok: promote config/model-router.json gpu5080-1080ti to hosts with live=true role retrieve-index-light, ";
rp += "base_url the guest LAN IP (or gpu5080 proxy). Wire csd-openai-gateway / lab router so RAG retrieve-index can join the cluster. ";
    rp += "csd-kb-index / retrieve prefer 1080 Ti guest, not 5080 exclusive-seq. Keep 5080 for CUDA CI. ";
    rp += "If guest smi failed, live=false. Commit+push autodev; PR vs main; do not merge skip-theatre.\n";
    rp += "ok=true if live flag matches guest_smi reality.\n";
    let route = agent(rp, #{
        label: "router-live",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Verify");
    let vp = "";
    vp += hard;
    vp += "host nvidia-smi -L still RTX 5080 only. guest smi 1080 Ti. virsh list running. stealth unit in guest. ";
    vp += "router dry-run rag-index/retrieve picks 1080ti only if live. 3090 LocalAI still up.\n";
    let ver = agent(vp, #{
        label: "verify-guest",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if prereq != () && prereq.success && prereq.output.ok == true {
        if guest != () && guest.success && guest.output.ok == true {
            if route != () && route.success && route.output.ok == true {
                if ver != () && ver.success && ver.output.ok == true {
                    sat = true;
                }
            }
        }
    }
    complete(#{
        summary: if sat { "1080 Ti guest live for RAG" } else { "1080 Ti guest incomplete" },
        sat: sat,
        apply: true,
        prereq: prereq,
        guest: guest,
        route: route,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += "Read-only prereq. ok=true if qemu exists. Re-run args.apply=true to boot guest.\n";
let ver2 = agent(vp2, #{
    label: "prereq-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "prereq only; re-run args.apply=true to boot guest",
    apply: false,
    prereq: prereq,
    verify: ver2,
});
