let meta = #{
    name: "csd-1080ti-bringup",
    description: "gpu5080 back up: correct dual-3TB mechanical RAID, short 1080 Ti RAG/light bench, promote live only if smi shows both GPUs. Never power off. Never pause 3090.",
    when_to_use: "After operator reseats 1080 Ti and dual 3TB HDDs; host is powering on. RAID for performance; 1080 Ti = retrieve/index/light only.",
    phases: [
        #{ title: "Reach", detail: "ssh + nvidia-smi both GPUs; lock idle" },
        #{ title: "Volume", detail: "dual 3TB mechanical RAID available" },
        #{ title: "Bench", detail: "short 1080 Ti RAG/light; not 14B; not 5080 CUDA" },
        #{ title: "Live", detail: "router live=true only if smi+bench ok" },
    ],
};

let ok_schema = #{
    "type": "object",
    "required": ["ok", "notes", "evidence"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
        "evidence": #{ "type": "string" },
        "idle": #{ "type": "boolean" },
        "have_1080ti": #{ "type": "boolean" },
    },
};

fn as_str(x) {
    if type_of(x) == "string" { return x; }
    "";
}

let do_apply = false;
if args != () {
    if args.apply == true { do_apply = true; }
}

let hard = "";
hard += "HARD: gpu5080 DHCP moved. Current LAN is 192.168.1.251 (hostname gpu5080, enp5s0), NOT 192.168.1.252. ";
hard += "ssh -o BatchMode=yes tzervas@192.168.1.251 or Host gpu5080 after HostName 192.168.1.251. Jump -J homelab if needed. ";
hard += "Do NOT treat .252 INCOMPLETE as the host being down. ";
hard += "Do NOT power off. Do NOT pause 3090 LocalAI. ";
hard += "Do NOT unmask Comfy. Do NOT run 14B on 1080 Ti. Do NOT steal 5080 exclusive-seq for this bench. ";
hard += "Never 0.0.0.0 WAN. Never GitHub bot push. Never main/staging/develop/dev. ";
hard += "Never write operator/model vaults. Never mix 384-d Qdrant. Identity autodev. feat/agent-harness.\n";
hard += "1080 Ti = Pascal sm_61 ~11GiB. Role retrieve/index/light embed. 5080 stays train/eval.\n";
hard += "Disks: operator now has dual 3TB HDDs for RAID performance (stripe, not RAID1). ";
hard += "Do not wipe NVMe/SSD OS. Do not touch a healthy array that already uses both 3TB. ";
hard += "If host is still booting, retry SSH a few times then ok=false idle=false — do not hang forever.\n";

phase("Reach");
let rp = "";
rp += hard;
rp += "Reach: ssh tzervas@192.168.1.251 'hostname; ip -4 -br addr; nvidia-smi -L; lspci | grep -i nvidia'. ";
rp += "homelab getent gpu5080.localdomain was 192.168.1.251. Ignore 192.168.1.252.\n";
rp += "have_1080ti=true if nvidia-smi lists 1080 Ti. If only lspci GP102 06:00.0 shows it, have_1080ti=false and notes=driver bind needed.\n";
rp += "idle=true if gpu5080.lock free and 5080 util low. evidence=smi -L + which hop worked. ok=true only if a command ran on the box.\n";
let reach = agent(rp, #{
    label: "reach-smi",
    capability_mode: "execute",
    output_schema: ok_schema,
});

let reach_ok = false;
if reach != () && reach.success && reach.output.ok == true {
    reach_ok = true;
}

if do_apply && reach_ok {
    phase("Volume");
    let vp = "";
    vp += hard;
    vp += "lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL,ROTA; cat /proc/mdstat; findmnt; grep -v '^#' /etc/fstab.\n";
    vp += "Observed 2026-08-31: sda 2.7T HGST sda1 1.8T linux_raid_member leftover + sda2 931G ext4 /models (74G used); ";
vp += "sdb 2.7T HGST zfs_member; mdstat empty; fstab LABEL=bulk /bulk not mounted; NVMe is OS. Preserve /models data.\n";
    vp += "Correct so a mechanical RAID volume is AVAILABLE for RAG store (index spill, corpora, cold GGUF — not OS).\n";
    vp += "Performance = mdadm RAID0 (or existing stripe). If array was built against a 2TB member, replace/grow to both 3TB ";
    vp += "only with evidence the old 2TB is gone. If no array, assemble RAID0 on the two unused 3TB HDDs, mkfs if empty, ";
    vp += "mount under an existing lab path if docs name one (read CSD-1080TI-RAG.md, akula-harness, /data,/mnt), else /mnt/rag-hdd.\n";
    vp += "fstab UUID. Do not format a disk that already has wanted data without noting ok=false.\n";
    vp += "ok=true if mount is rw and both 3TB members are in the array. evidence=lsblk+df+mdstat.\n";
    let vol = agent(vp, #{
        label: "raid-volume",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Bench");
    let bp = "";
    bp += hard;
    bp += "SSH 192.168.1.251. lspci already has GP102 1080 Ti at 06:00.0 but nvidia-smi may only show the 5080 — bind/load nvidia on that PCI ID (dmesg, lspci -k). Do not break 5080 driver. ";
bp += "Short bench on 1080 Ti ONLY once it has a UUID. 5080 stays idle.\n";
    bp += "1) nvidia-smi on 1080 Ti after a tiny torch or cuda query. 2) VRAM total/free. ";
    bp += "3) One small embed or retrieve path if weights exist on disk — no download of a 14B. ";
    bp += "4) Optional: write a tiny file on the mechanical RAID (RAG store smoke). ";
    bp += "5) Do not start csd-kb-index exclusive on 5080. Prefer 1080 Ti if a flag exists.\n";
    bp += "Write docs/program/CSD-1080TI-BENCH.md with honest numbers (no 10x claims). ok=true if 1080 Ti ran a CUDA op and RAID write worked.\n";
    let bench = agent(bp, #{
        label: "bench-1080ti",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Live");
    let lp = "";
    lp += hard;
    lp += "Promote config/model-router.json future_hosts.gpu5080-1080ti to live=true ONLY if have_1080ti and bench ok.\n";
    lp += "Else keep live=false. Update CSD-1080TI-RAG.md. o11y path lab.gpu5080.index.1080ti if labels exist.\n";
    lp += "Commit+push autodev feat/agent-harness. Open PR if main already has #2 merged. Do not merge skip-theatre.\n";
    lp += "ok=true if JSON live flag matches smi reality. evidence=sha + live field + smi -L.\n";
    let live = agent(lp, #{
        label: "router-live",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    let sat = false;
    if reach != () && reach.success && reach.output.ok == true {
        if vol != () && vol.success && vol.output.ok == true {
            if bench != () && bench.success && bench.output.ok == true {
                if live != () && live.success && live.output.ok == true {
                    sat = true;
                }
            }
        }
    }
    complete(#{
        summary: if sat { "1080 Ti RAID+bench live" } else { "1080 Ti bringup incomplete" },
        sat: sat,
        apply: true,
        reach: reach,
        volume: vol,
        bench: bench,
        live: live,
    });
}

phase("Live");
let vp2 = "";
vp2 += "Read-only summary of reach. ok=true if ssh worked. Re-run args.apply=true to RAID+bench+live.\n";
let ver2 = agent(vp2, #{
    label: "reach-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "reach-only; re-run args.apply=true for RAID+bench+live",
    apply: false,
    reach: reach,
    verify: ver2,
});
