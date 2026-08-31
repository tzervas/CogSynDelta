let meta = #{
    name: "owui-comfy-flux-klein-4b",
    description: "Queue native OWUI Flux Klein 4B graph via csd-comfy. Fail-closed when Comfy is masked. No DALL-E. Not overlay.",
    when_to_use: "Need a still from IMAGE_GENERATION_ENGINE=comfyui (prompt node 5, size/steps 4+10, seed 8).",
    phases: [
        #{ title: "Queue", detail: "scripts/csd-comfy flux-klein-4b; masked stays masked" },
    ],
};

let out_schema = #{
    "type": "object",
    "required": ["ok", "notes"],
    "properties": #{
        "ok": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
        "prompt_id": #{ "type": "string" },
        "history": #{ "type": "object" },
        "evidence": #{ "type": "string" },
    },
};

phase("Queue");
let p = "";
p += "HARD: do not unmask Comfy. Do not start Comfy. Do not call DALL-E or cloud image APIs.\n";
p += "Repo /home/kang/code/personal/tzervas/CogSynDelta. Native graph config/media/comfy-image-workflow.json ";
p += "(prompt node 5, width/height/steps nodes 4+10, seed node 8). Overlay akula_media does not POST /prompt.\n";
p += "This workflow args: text, width (default 1024), height (default 1024), steps (default 4), seed (default 0).\n";
p += "Pass those fields as JSON on stdin to ./scripts/csd-comfy flux-klein-4b. Do not invent a cloud engine.\n";
p += "If notes=masked, return {ok:false, notes:masked}. ok=true only if prompt_id is non-empty.\n";
p += "evidence=scripts/csd-comfy and config/media/comfy-image-workflow.json.\n";
let rec = agent(p, #{
    label: "flux-klein-4b",
    capability_mode: "execute",
    output_schema: out_schema,
});
let sat = false;
if rec != () && rec.success && rec.output != () {
    if rec.output.ok == true {
        sat = true;
    }
    if rec.output.notes == "masked" {
        sat = true;
    }
}
complete(#{
    summary: if sat { "flux-klein-4b honest (queued or masked)" } else { "flux-klein-4b failed" },
    sat: sat,
    result: rec,
});
