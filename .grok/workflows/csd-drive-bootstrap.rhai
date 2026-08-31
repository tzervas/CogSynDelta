let meta = #{
    name: "csd-drive-bootstrap",
    description: "Index drive targets + implementer handoff into akula-csd-kb; enqueue CUDA index; report next board ID",
    when_to_use: "After the symphony/horizon plan is written; kick autoloops without stuffing the parent chat",
    phases: [
        #{ title: "Mirror", detail: "copy program docs into experiment vault" },
        #{ title: "Index", detail: "enqueue 5080 timeshare reindex, never 3090" },
        #{ title: "Next", detail: "name the next closeable Phase 1 ID" },
    ],
};

let mirror_schema = #{
    "type": "object",
    "required": ["copied", "vault_dir", "notes"],
    "properties": #{
        "copied": #{ "type": "array", "items": #{ "type": "string" } },
        "vault_dir": #{ "type": "string" },
        "notes": #{ "type": "string" },
    },
};

let index_schema = #{
    "type": "object",
    "required": ["enqueued", "summary"],
    "properties": #{
        "enqueued": #{ "type": "boolean" },
        "summary": #{ "type": "string" },
        "points": #{ "type": "string" },
    },
};

let next_schema = #{
    "type": "object",
    "required": ["id", "closeable", "reason"],
    "properties": #{
        "id": #{ "type": "string" },
        "closeable": #{ "type": "boolean" },
        "reason": #{ "type": "string" },
    },
};

phase("Mirror");
let mirror_p = "";
mirror_p += "Copy CogSynDelta program docs into the experiment vault ONLY.\n";
mirror_p += "Vault: /akula-data/obsidian/akula-csd-kb/Program/\n";
mirror_p += "Source dir: docs/program/ in this repo (read_file + shell cp).\n";
mirror_p += "Copy at least: PYTHON-FIRST-PLAN.md PHASE-1-TASK-BOARD.md HANDOFF-NEXT.md\n";
mirror_p += "SELF-HOSTED-DRIVE-TARGETS.md IMPLEMENTER-HANDOFF.md PYTHON-MEMORY-GATE-GAP-MAP.md\n";
mirror_p += "Never write tzervas-dev-kb or akula-model-kb or akula-gap-kb.\n";
mirror_p += "Never pause 3090 LocalAI. Do not run CUDA embeddings yourself.\n";
mirror_p += "Return copied filenames, vault_dir, notes. Use tools.\n";
let mirror = agent(mirror_p, #{
    label: "mirror-vault",
    capability_mode: "all",
    output_schema: mirror_schema,
    model: "grok-4.6",
});
if mirror == () || !mirror.success {
    complete(#{ summary: "vault mirror failed" });
}

phase("Index");
let idx_p = "";
idx_p += "Enqueue akula-csd-kb CUDA reindex. From CogSynDelta repo root run:\n";
idx_p += "./scripts/csd-kb-index\n";
idx_p += "That script timeshares the 5080. Do not docker stop LocalAI.\n";
idx_p += "Do not kill Comfy. If it enqueues, enqueued=true.\n";
idx_p += "Do not claim a nonzero Qdrant point count unless you queried it.\n";
idx_p += "Return enqueued, summary, optional points.\n";
let idx = agent(idx_p, #{
    label: "enqueue-index",
    capability_mode: "all",
    output_schema: index_schema,
    model: "grok-4.6",
});

phase("Next");
let next_p = "";
next_p += "Read docs/program/HANDOFF-NEXT.md and PHASE-1-TASK-BOARD.md with tools.\n";
next_p += "Pick the next closeable Phase 1 ID for /csd-python-first-drive.\n";
next_p += "Product P1-02+ needs a real Forgejo pytest run (already exists).\n";
next_p += "Horizon training in SELF-HOSTED-DRIVE-TARGETS.md is NOT closeable yet.\n";
next_p += "Stay-open (whole Phase 3 train) closeable=false.\n";
next_p += "Return id, closeable, reason.\n";
let nxt = agent(next_p, #{
    label: "next-id",
    capability_mode: "read-only",
    output_schema: next_schema,
    model: "grok-4.5",
});

let copied = [];
let vault_dir = "";
if mirror != () && mirror.success && mirror.output != () {
    copied = mirror.output.copied;
    vault_dir = mirror.output.vault_dir;
}
let enqueued = false;
let idx_sum = "index skipped";
if idx != () && idx.success && idx.output != () {
    enqueued = idx.output.enqueued;
    idx_sum = idx.output.summary;
}
let nid = "unknown";
let nclose = false;
let nreason = "next pick missing";
if nxt != () && nxt.success && nxt.output != () {
    nid = nxt.output.id;
    nclose = nxt.output.closeable;
    nreason = nxt.output.reason;
}

let report = #{
    vault_dir: vault_dir,
    copied: copied,
    index_enqueued: enqueued,
    index_summary: idx_sum,
    next_id: nid,
    next_closeable: nclose,
    next_reason: nreason,
};
let path = write_scratch_file("csd-drive-bootstrap.md", json_encode(report));
complete(#{
    path: path,
    next_id: nid,
    next_closeable: nclose,
    index_enqueued: enqueued,
    summary: "bootstrap vault+index; next=" + nid,
});
