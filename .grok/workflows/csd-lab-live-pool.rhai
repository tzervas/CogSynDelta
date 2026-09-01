let meta = #{
    name: "csd-lab-live-pool",
    description: "Lab pool VRAM includes live 1080 Ti; autodev tab streams model thinking. G-POOL tensor-wire is RESEARCH only (not implement).",
    when_to_use: "Pool tab still 24+16 GiB; Autodev tab too thin; operator pasted tensor-wire blurb for future G-POOL.",
    phases: [
        #{ title: "Research", detail: "validate tensor-wire notes; do not implement" },
        #{ title: "Pool", detail: "combined VRAM 3090+5080+1080ti when live" },
        #{ title: "Feed", detail: "autodev live thinking + per-card inference" },
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
hard += "HARD: Do NOT implement the tensor wire protocol, Arrow, QUIC, UDP, RDMA, or llama.cpp RPC Stage B. Research+doc only. ";
hard += "256 specialists is a future G-SHARE example, not live. Never GitHub. Never pause 3090 LocalAI. ";
hard += "gpu5080=192.168.1.251; 1080 Ti guest=192.168.1.243 live retrieve-index-light. Identity autodev feat/agent-harness.\n";

phase("Research");
let rp = "";
rp += hard;
rp += "web_search Apache Arrow Dissociated IPC, RFC 9221 QUIC DATAGRAM, Linux MSG_ZEROCOPY, io_uring zcrx. ";
rp += "Write docs/program/CSD-TENSOR-WIRE.md: GPU quantize Q8/Q4 -> pinned host -> lean TCP (benchmark before QUIC) -> reconstruct. ";
rp += "Control plane vs tensor data plane. Prefill vs decode policy. Cluster self-benchmark effective tensor rate. ";
rp += "Mark G-POOL / ADR-0015 horizon. Autodev must NOT implement until G-LIFE and this doc says unblocked. Cite URLs.\n";
rp += "FINAL json fence: ok, notes, evidence. ok=true if the doc exists with citations.\n";
let research = agent(rp, #{
    label: "tensor-wire-doc",
    capability_mode: "execute",
    output_schema: ok_schema,
});

if do_apply {
    phase("Pool");
    let pp = "";
    pp += hard;
    pp += "scripts/csd-lab-console live_feed pool: do not default combined_mib=39331. Sum live hosts vram_mib from model-router (3090 23028 + 5080 16303 + 1080ti 11264 when live=true). ";
    pp += "usable = sum of safeguards. Pool tab HTML: list three cards and GiB; drop stale 24:16-only copy. Include guest smi if ssh 192.168.1.243 nvidia-smi works. ";
    pp += "feed.gpu5080_1080ti pane like prime/5080. tests/test_lab_console.py cover combined when live.\n";
    pp += "ok=true if GET /api/status feed.pool.combined_mib >= 50000 when 1080ti live (approx 50GiB class).\n";
    let pool = agent(pp, #{
        label: "pool-vram",
        capability_mode: "execute",
        output_schema: ok_schema,
    });

    phase("Feed");
    let fp = "";
    fp += hard;
    fp += "Autodev tab: not just ticks JSON. Stream last autodev-out text (splitThink), in-flight queue prompt, goal, files touched, pytest rc. ";
    fp += "GPUs tab already has thinking for prime; add 5080 queue/cuda receipt and 1080 Ti guest loaded/thinking. Poll refresh ~2s on Autodev tab. ";
    fp += "Heartbeat last.goal last.ok last.error. This is 'what the model is doing now'. Keep Goals tab for board.\n";
    fp += "Commit autodev; do not merge skip-theatre. FINAL json only.\n";
    fp += "ok=true if Autodev pane shows think or in-flight pre from live data.\n";
    let feed = agent(fp, #{
        label: "live-think",
        capability_mode: "execute",
        output_schema: ok_schema,
    });
    let sat = false;
    if research != () && research.success && research.output.ok == true {
        if pool != () && pool.success && pool.output.ok == true {
            if feed != () && feed.success && feed.output.ok == true {
                sat = true;
            }
        }
    }
    complete(#{
        summary: if sat { "lab pool+live feed; tensor-wire research only" } else { "lab live pool incomplete" },
        sat: sat,
        apply: true,
        research: research,
        pool: pool,
        feed: feed,
    });
}

phase("Feed");
let vp2 = "";
vp2 += "Read-only. ok=true if CSD-TENSOR-WIRE.md exists. Re-run args.apply=true for UI.\n";
let ver2 = agent(vp2, #{
    label: "doc-only",
    capability_mode: "read-only",
    output_schema: ok_schema,
});
complete(#{
    summary: "research-only; re-run args.apply=true",
    apply: false,
    research: research,
    verify: ver2,
});
