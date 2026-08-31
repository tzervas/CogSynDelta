let meta = #{
    name: "csd-1080ti-stealth",
    description: "gpu5080 box only: include GTX 1080 Ti GeForce logo LED in existing stealth-leds (OpenRGB oneshot, never --server). 5080 already in the profile.",
    when_to_use: "1080 Ti seated; GeForce logo still lit; cabal-stealth-leds covers 5080/DRAM/MB only.",
    phases: [
        #{ title: "List", detail: "OpenRGB devices on .251 including 1080 Ti" },
        #{ title: "Apply", detail: "mode Off / Direct #000000; GPULogoBrightness 0 if needed" },
        #{ title: "Verify", detail: "device list + oneshot unit; no OpenRGB server" },
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
hard += "HARD: gpu5080 192.168.1.251 tzervas. Scope that box only (5080 + 1080 Ti). Do not change 3090 stealth. ";
hard += "Never start OpenRGB --server (CVE-2026-59682 / 59683). Use akula-ai-platform/scripts/stealth-leds oneshot (root, i2c-dev). ";
hard += "Do NOT pause LocalAI. Do NOT unmask Comfy. Do NOT power off. Never GitHub.\n";
hard += "1080 Ti GeForce logo is often a separate NVIDIA logo LED. OpenRGB --list-devices; if no 1080 Ti / GP102 device, try DISPLAY=:0 nvidia-settings -a GPULogoBrightness=0 for that GPU index, or i2c as documented. ";
hard += "If 1080 Ti is vfio-bound, stealth MUST also run INSIDE the guest (same stealth-leds oneshot, never OpenRGB --server). ";
hard += "A host-only Off is not enough: the guest Pascal driver will re-light the GeForce logo on bind. ";
hard += "CaC: guest systemd oneshot + cloud-init/quadlet so stealth applies at VM boot before CUDA. Do not unbind 5080.\n";

phase("List");
let lp = "";
lp += hard;
lp += "ssh .251: sudo /path/to/stealth-leds dry list if possible; else copy script and OpenRGB AppImage --list-devices. ";
lp += "Record whether a 1080 Ti / GP102 / GeForce logo device appears. ok=true if a device list was captured.\n";
let listed = agent(lp, #{
    label: "list-rgb",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Apply");
    let ap = "";
    ap += hard;
    ap += "Run stealth-leds on gpu5080 (sudo). Confirm 1080 Ti device got --mode Off and Direct 000000. ";
    ap += "If OpenRGB cannot see the logo: nvidia-settings GPULogoBrightness=0 (needs a display; box may have login prompt so :0 might work). ";
    ap += "Update akula-ai-platform config/hosts/5080.json stealth_leds to mention 1080 Ti logo. Enable cabal-stealth-leds.service if missing. ";
    ap += "Commit CaC on the private Forgejo tree that already holds stealth-leds. Never GitHub.\n";
    ap += "ok=true if the oneshot ran and 1080 Ti is in the device list or logo brightness 0.\n";
    let applied = agent(ap, #{
        label: "apply-stealth",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Verify");
    let vp = "";
    vp += hard;
    vp += "pgrep -a OpenRGB must not include --server. stealth unit enabled. evidence=list-devices + systemctl is-enabled. ";
    vp += "Honest: you cannot see the logo from SSH; operator confirms visually.\n";
    let ver = agent(vp, #{
        label: "verify-stealth",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if listed != () && listed.success && listed.output.ok == true {
        if applied != () && applied.success && applied.output.ok == true {
            if ver != () && ver.success && ver.output.ok == true {
                sat = true;
            }
        }
    }
    complete(#{
        summary: if sat { "1080 Ti included in stealth-leds" } else { "1080 Ti stealth incomplete" },
        sat: sat,
        apply: true,
        list: listed,
        apply_result: applied,
        verify: ver,
    });
}

phase("Verify");
let vp2 = "";
vp2 += "Read-only: stealth-leds script exists in akula-ai-platform. ok=true if file exists. Re-run apply=true to kill 1080 Ti logo.\n";
let ver2 = agent(vp2, #{
    label: "script-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "stealth script present; re-run args.apply=true",
    apply: false,
    list: listed,
    verify: ver2,
});
