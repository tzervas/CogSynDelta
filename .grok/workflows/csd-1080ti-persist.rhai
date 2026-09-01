let meta = #{
    name: "csd-1080ti-persist",
    description: "gpu5080 reboot persistence: VFIO bind, libvirt guest autostart, stealth host+guest, factory eco, RAID fstab. Comfy stays masked.",
    when_to_use: "After isolate/guest/eco on .251. Units must survive reboot without hand SSH.",
    phases: [
        #{ title: "Inventory", detail: "units, fstab, virsh autostart, linger" },
        #{ title: "Enable", detail: "systemd enable + virsh autostart + fstab UUID" },
        #{ title: "Verify", detail: "is-enabled; no Comfy; CaC in deploy/gpu5080" },
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
hard += "HARD: gpu5080 192.168.1.251 tzervas. Do NOT unmask Comfy. Do NOT pause 3090. Do NOT VFIO the 5080. ";
hard += "Never GitHub. Never 0.0.0.0 WAN. Identity autodev. feat/agent-harness.\n";
hard += "Persist: vfio-bind-1080ti.service, nvidia-factory-limits.service, cabal-stealth-leds.service (host), ";
hard += "libvirt default + virsh autostart for gpu5080-1080ti-rag, guest stealth oneshot, RAID/mdadm + fstab UUID nofail, ";
hard += "LAN network so the guest can join the inference cluster (bridge/macvtap on enp5s0, DHCP or static 192.168.1.x, ";
hard += "OpenAI-compatible listen on that LAN IP only — never 0.0.0.0 WAN). model-router/csd-openai-gateway/Caddy must reach the guest. ";
hard += "If guest is not defined yet, enable VFIO+eco+stealth+fstab anyway and gaps=guest-autostart. ";
hard += "Copy units from CogSynDelta deploy/gpu5080/ if missing on the host. linger yes only for user units that must survive logout.\n";

phase("Inventory");
let ip = "";
ip += hard;
ip += "ssh .251: systemctl is-enabled for those units; virsh list --all --autostart; grep nofail /etc/fstab; cat /proc/mdstat; ";
ip += "ls /etc/systemd/system/*vfio* *stealth* *factory* 2>/dev/null. ok=true with that list.\n";
let inv = agent(ip, #{
    label: "inv-units",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Enable");
    let ep = "";
    ep += hard;
    ep += "Install/enable the oneshots so a reboot brings back: 1080 Ti vfio-pci before libvirt, then guest autostart, ";
    ep += "guest stealth-leds, host stealth-leds, factory PL on 5080, mechanical RAID mount, ";
    ep += "and a persistent libvirt LAN (bridge or macvtap on enp5s0) so the guest gets 192.168.1.x and joins the inference cluster. ";
    ep += "Document guest IP in model-router. Guest listen LAN-only, never 0.0.0.0 WAN. ";
    ep += "Do not start Comfy. Do not enable a guest that is not defined. ";
    ep += "Commit any new unit files under deploy/gpu5080/ as autodev. Do not merge.\n";
    ep += "ok=true if vfio-bind and factory-limits are enabled. evidence=systemctl is-enabled.\n";
    let en = agent(ep, #{
        label: "enable-units",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Verify");
    let vp = "";
    vp += hard;
    vp += "is-enabled vfio-bind factory-limits stealth; virsh autostart list; comfy still masked. ";
    vp += "Do not actually reboot unless you must — if you reboot, restore 3090 LocalAI check from prime. ";
    vp += "ok=true if required units enabled. evidence=names=enabled.\n";
    let ver = agent(vp, #{
        label: "verify-persist",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if inv != () && inv.success && inv.output.ok == true {
        if en != () && en.success && en.output.ok == true {
            if ver != () && ver.success && ver.output.ok == true {
                sat = true;
            }
        }
    }
    complete(#{
        summary: if sat { "gpu5080 persist units enabled" } else { "gpu5080 persist incomplete" },
        sat: sat,
        apply: true,
        inventory: inv,
        enable: en,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += "Read-only inventory. ok=true if inv.ok. Re-run args.apply=true to enable.\n";
let ver2 = agent(vp2, #{
    label: "inv-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "persist inventory; re-run args.apply=true",
    apply: false,
    inventory: inv,
    verify: ver2,
});
