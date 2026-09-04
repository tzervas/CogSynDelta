# GPU share — horizon pack (do not start)

Self-hosted backlog for **MIG-like slicing** on consumer cards, composed
with `G-POOL`. Not `STATUS.md`. Not `G-SPLIT` (one CSD mind). Autodev
must **not** pick this while `G-QD` / `G-LIFE` are open.

Operator intent: run **many microscopic specialized models at once**
(target **256**, when they **fit**), with hybrid split/share: automatically
**dedupe identical read-only weights**, isolate only what must be private
(KV, unique adapters, scratch). Plus later a **1080 Ti** in the 5080
workstation as a tagged pool node.

## Do not implement yet

Unblock only when **all** of:

1. `G-POOL` Stage A has a measured week of place/migrate without OOM.
2. Phase 1 product (`G-QD` … `G-LIFE`) is not the current closeable.
3. Operator has not paused via `csd-steer.json`.
4. For `G-1080`: the card is physically installed and `nvidia-smi` on
   gpu5080 shows it. Software must not assume it exists.

## Why consumer MIG is not available

3090 Ti, 5080, and 1080 Ti are GeForce. **MIG is datacenter-only**
(A100/H100/A30/…). Do not claim MIG, do not call nvidia-smi MIG APIs.
Get **as close as is robust** on this hardware.

## 256 microscopic specialists that fit

This is **not** 256 copies of a 14B, and not only 256 sessions of one
backbone. It is up to 256 **small distinct models** (heads, LoRAs, tiny
GGUFs, region specialists) that **collectively fit**, with the scheduler
interning what is the same.

Disk symlink / hardlink / same inode is the easy layer. **It does not
dedupe VRAM.** Two processes mmap the same GGUF: host page cache is
shared; each CUDA context still uploads its own weights unless we
**intern tensors in one process (or CUDA IPC)**.

### Hybrid share vs isolate

Share by **content hash** (read-only). Isolate by **mutability**.

| Thing | Default | Why |
|---|---|---|
| Identical weight tensors (same hash) | intern one VRAM copy; all models point at it | symlink-on-GPU |
| Identical files on disk | one inode (symlink/hardlink/mmap) | host RAM / disk |
| Shared tokenizer / embed tables | intern | common data pool |
| Unique tensors / unique LoRAs | private | they are the specialist |
| KV cache, context, session | private, quota-bounded | must not leak |
| Scratch / activations | pooled allocator, zeroed on recycle | isolate contents, share capacity |
| Sampling RNG / request id | private | |
| Writable training buffers | never intern | CoW or refuse |

Models are graphs of pointers into a **read-only weight pool** plus a
small private overlay. If two specialists share an embedding matrix and
differ only in last layers / LoRA, VRAM is `unique_sum + interned_union`,
not `n * full_model`. If they share nothing, they pay full unique size
and admission control may refuse the 256th.

### Fit math (honest)

Admission uses **interned unique bytes + Σ KV + headroom**, not
`n * sizeof(model)`. 256 × ~80 MiB unique with no overlap is ~20 GiB
weights — may fit leftover 3090 or the 5080 **if** KV is tiny. 256 ×
identical 80 MiB GGUF interned is ~80 MiB + 256×KV. Mixed is in
between. If it does not fit: reject, page KV, or timeslice. Never OOM.

### Parallel ops on the interned backend

Do **not** run 256 sequential forwards, and do **not** launch 256
independent GEMMs against the same interned tensor. Group in-flight
requests by the **shared interned blocks they touch**, then:

1. One kernel / batched matmul per interned block (token batch across
   specialists that share it).
2. Unique overlays (LoRA / last layers) run per specialist, still
   batched where shapes match.
3. Private KV stays per specialist; paged attention-style packing is
   allowed if measured.
4. Prefer the card whose caps include `parallel-batch` (5080
   Blackwell) for the interned GEMM; keep tight autodev decode off
   this pool.

Boilerplate policy lives in `scripts/csd-weight-intern` (`intern_plan`,
`parallel_schedule`, `pick_runtime`). Autodev extends it; do not invent
a second pool.

### Mixed runtimes (best-in-class per specialist)

The interned **memory** pool is backend-agnostic. **Compute** is not.
Pick a runtime per specialist (mix allowed in one schedule):

| Runtime | When it is the default |
|---|---|
| `llama.cpp` | GGUF, interned micro GGUF, RPC split, autodev wrap |
| `vllm` | safetensors/HF, paged attention, continuous batch, multi-LoRA |
| `bitnet-cpp` | BitNet **1.58** / W1.58A8. Do not cram these into vLLM or
  generic llama.cpp unless a **measured** row says so |
| `localai` | OpenAI-compat wrap of llama.cpp on the 3090 |

Catalog: `config/model-router.json` `runtimes`. Hosts still filter by
caps (no vLLM on Pascal 1080 Ti). Intern only tensors that share a
**format and runtime**; do not intern a GGUF block into a BitNet
graph.

## Closest-to-MIG stack (ranked)

Try in this order; drop a row only with a measured reason:

1. **Weight-interned multi-model in one process** (content-hash tensor
   pool, CoW if anything would write). Disk: symlink/hardlink same
   GGUF. GPU: intern, not 256 uploads. Default for 256 micro
   specialists.
2. **One backbone + N LoRA/slot overlays** (llama.cpp parallel / vLLM
   multi-LoRA) when specialists are adapters on a shared base.
3. **CUDA MPS** on a card: `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` +
   `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` per client when more than one
   process is required. Memory cap ≠ tensor intern. Not a VM.
4. **CUDA green contexts** (driver permitting) to partition SMs. Probe;
   do not assume 3090 or 1080 Ti support it.
5. **Per-process `torch.cuda.set_per_process_memory_fraction`** for CUDA
   eval jobs that are not GGUF.
6. **Time-share queue** (`gpu5080.lock`, LocalAI ttl) as today — fallback
   when a job needs exclusive CUDA.

Never: two 14B, pause LocalAI for RAG, unmask Comfy during autodev,
schedule `sm_120` on Ampere/Pascal, or treat MPS as a security boundary
(it is a resource cap, not a VM).

## Compose with the pool

- Tight decode (autodev `local/code`) stays **one card, one exclusive
  class**. Do not fold it into the 256-way intern pool.
- `ok-lan` microscopic specialists may share a card and, later, the LAN
  pool (interned weights stay on one card; KV may page).
- Shared common data (tokenizers, eval corpora, prefix-cache) lives in a
  **read-only mmap / RAM disk** both hosts can see (homelab NFS or rsync
  snapshot) — not a second Qdrant dimension mix.
- Lab **Pool** tab should later show intern hit rate, unique MiB, and
  loaded specialists — not the catalog. Do not fake it before the
  intern pool exists.

## Uses (why this exists)

- **Large MoE on the pool** (~24+16 GiB): expert shards across cards;
  gate + active experts stay latency-tight; idle experts may sit
  `ok-lan`. llama.cpp GGUF-MoE or vLLM HF-MoE — pick_runtime.
- **Many small CSD models** (after Phase 3 one-GPU proof): region
  specialists interned and batched. Still **one mind**, not agents.
- Matrix testing, eval grids, mixed-effort, persona basins.

## G-1080 — 1080 Ti in the 5080 box (operator hardware)

Catalog stub: `config/model-router.json` `future_hosts.gpu5080-1080ti`
(`live: false`, `installed: false`). Not in `hosts`. Router must ignore
`future_hosts` for scheduling until `nvidia-smi` on gpu5080 lists the
card and the stub is promoted with `live: true`.

| Field | Value |
|---|---|
| GPU | EVGA reference GTX 1080 Ti |
| Arch / SM | Pascal 6.1 |
| VRAM | ~11264 MiB |
| TDP | ~250 W |
| Caps | `pascal` `sm_61` `gguf-infer-legacy` `11gb` |
| Lacks | tensor cores, native FP16 tensor, `sm_86`, `sm_120`, `fp8`, `fp4`, `high-vram` |
| Box | gpu5080 workstation, 1300 W PSU (5080 + 1080 Ti + CPU is likely OK) |

**Cooling (operator, not autodev):** factory reference cooler **first**.
If case thermals are bad: optional dedicated loop on the 1080 Ti only
(pump + 360 rad in the top of that chassis, block, tubes; no reservoir
required). Do not plan a 1080 Ti loop until factory air is measured.

**Software role after install:** `retrieve-index-light` (RAG retrieve/
index, keyword-cpu assist, small embed, light GGUF). Never autodev 14B;
never sm_120 CUDA; tag-routed only. Combined pool becomes ~24 + 16 +
11 GiB **only** for workloads
that can use Pascal + Ampere + Blackwell together (llama.cpp CUDA built
for all three, or CPU-side gather). Heterogeneous three-way tensor split
is optional and later than two-way 3090+5080.

## Implementation slices (when unblocked)

One architecture change per PR, memory-gate/CSD lab scripts only:

1. Tensor intern by content hash (unit tests, fake tensors, no GPU):
   two models with an identical block occupy it once; unique blocks
   stay unique; writable tensors never intern.
2. Disk identity: same-path / symlink / inode counted once in the
   budget. Document that this does **not** share VRAM across processes.
3. Probe one-process multi-model load on 5080 leftover (never steal
   3090 `local/code`). Measure unique VRAM vs naive sum. Isolation
   test: specialist A must not read B's KV.
4. Router admit/reject using interned unique + KV + headroom.
5. Lab feed: loaded specialists (not the catalog). Pool tab: intern
   hit rate + unique MiB.
6. Optional MPS / 1080 Ti after the intern pool is measured.

## Safety

- Hard VRAM cap per shard; OOM is a bug.
- Isolation test is required (wrong-tenant read = fail).
- Autodev 14B is sticky exclusive on the 3090.
- No WAN. No G-TRAIN. No claim of "MIG equivalent" without a table
  against real MIG (we will not match it).
