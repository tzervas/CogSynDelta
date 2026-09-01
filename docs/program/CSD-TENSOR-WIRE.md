# Tensor wire — research pack (blocked)

Not `STATUS.md`. Not an implementation. Not llama.cpp RPC Stage B.
Not Arrow, QUIC, UDP, RDMA, UCX, or a new socket server. Autodev
must **not** implement this until **`G-LIFE` is met** and **this file
says unblocked**.

Identity: **autodev**. Branch: `feat/agent-harness` (never `main` /
`staging` / `develop` / `dev`). Goal horizon: `G-POOL` (ADR-0015).
Not `G-SPLIT` (one CSD mind). Not `G-SHARE` (256 specialists is a
**future intern-pool example**, not live).

Unblock status: **BLOCKED**.

## Why this file exists

`G-POOL` Stage A is live placement/migrate (`config/model-router.json`).
Stage B llama.cpp RPC (`pool.enabled=false`) stays parked. If the lab
ever moves tensors across 3090 + 5080 without pretending the cards
share a PCIe domain, the **data plane** is: quantize on GPU → pinned
host → **lean TCP first** → reconstruct. This pack records that
pipeline, the control/data split, prefill vs decode policy, and the
self-benchmark that would have to exist **before** QUIC or RDMA.

Until then: whole-model placement only. Tight decode stays on one
card. 1080 Ti guest is **retrieve-index-light**, not a tensor-split
peer.

## Hard rules

- Do **not** implement a tensor wire, Arrow IPC/Dissociated IPC
  client, QUIC, UDP, RDMA, UCX, libfabric, or llama.cpp RPC Stage B.
- Do **not** add sockets, framers, `rpc-server`, `--rpc`, or a
  second data-plane port.
- Do **not** pause 3090 LocalAI. Do **not** unmask Comfy. Do **not**
  power off gpu5080. Do **not** VFIO the 5080.
- Never dual 14B. Never GitHub. Never bind `0.0.0.0` on WAN.
- Never schedule `sm_120` / FP8 / FP4 on Ampere or Pascal.
- Never treat 256 specialists as a live count. Schedule what **FITs**.
- Never claim line-rate, 10×, or “zero-copy cluster” without a
  measured row in the self-benchmark section.
- `ok=false` for implement until the Unblock section flips.

## Lab endpoints (placement, not a mesh)

| Box | IP | GPU | Role today | Tensor-wire role |
|---|---|---|---|---|
| akula-prime | `192.168.1.98` | RTX 3090 Ti | `local/code` 14B sticky | Future **tight** owner. Never steal for a wire experiment. |
| gpu5080 | `192.168.1.251` (never `.252`) | RTX 5080 | CUDA / GPU CI / helpers behind `gpu5080.lock` | Future **ok-lan** prefill / expert shard only when lock idle. |
| 1080 Ti guest | `192.168.1.243` | GTX 1080 Ti VFIO `06:00.0` | **live** `retrieve-index-light` | **Not** a tensor-split peer. RAG/index only. |

Homelab (`192.168.1.170`) is CPU Actions. No GPU alias, no wire.

## Unblock (all of these, not any)

Autodev **must not** start this while **any** row is unmet:

| # | Gate | Today |
|---|---|---|
| 1 | `G-LIFE` met (learn → retrieve → consolidate → persona through restart) | **open** (`GOALS.md`) |
| 2 | This file’s Unblock status is **UNBLOCKED** (operator flip, not a PR) | **BLOCKED** |
| 3 | `G-POOL` Stage A has a measured week of place/migrate without OOM | Stage A live; week not claimed |
| 4 | Phase 1 product is not the current closeable | region-pretrain is steered; P1-16 later |
| 5 | Operator has not paused via `csd-steer.json` | obey steer |
| 6 | A written ADR (new, after ADR-0015) chooses **lean TCP** vs llama.cpp RPC after a **measured** tensor-rate row | missing |
| 7 | 3090 LocalAI stays loaded; 5080 exclusive-seq still preempts helpers | live policy |

Flipping this file to UNBLOCKED without `G-LIFE` is a bug. Implementing
before both is a bug.

## Pipeline (research shape, not code)

Intended path when unblocked:

```
GPU (Q8_0 or Q4_K)  →  pinned host  →  lean TCP  →  pinned host  →  reconstruct
     quantize            D2H / H2D     (benchmark            recv / zcrx?      dequant
                                       before QUIC)
```

1. **Quantize on the source GPU.** Move Q8/Q4 **bytes**, not FP16/FP32
   activations. GGUF `Q8_0` is ~8.5 bits/weight; `Q4_K_M` ~4.5–4.9.
   Wire bytes scale with that, not with parameter count × 2.
   Reconstruct on the destination GPU in the same scheme. Do not
   round-trip through host FP32 “for convenience.”
2. **Pinned host staging.** `cudaHostAlloc` / `cudaMallocHost`
   page-locks the staging buffer so `cudaMemcpy` D2H/H2D can DMA.
   Pageable `malloc` forces an extra bounce. Pin is a **staging
   area**, not a second model copy; CUDA docs warn that excessive pin
   hurts host paging. Use `cudaHostAllocPortable` only if more than
   one CUDA context on that host must see the buffer (5080 host vs
   1080 Ti guest do **not** share a module — VFIO split).
3. **Lean TCP on the LAN.** Length-prefixed binary frames. No HTTP,
   no JSON body, no gRPC. Trusted LAN (`192.168.1.0/24` only).
   **Benchmark TCP** (iperf3 + pinned-buffer echo of Q4/Q8 blob sizes)
   **before** considering QUIC. See Transport ladder.
4. **Reconstruct.** Destination copies pinned → device, dequants into
   the working dtype that card actually runs (Ampere FP16/BF16 vs
   Blackwell FP8/FP4). Do not send `sm_120` kernels to Ampere/Pascal.
   Sequence numbers on the control plane match body frames so
   metadata and body can arrive out of order without a single
   interleaved stream.

This is the same *idea* as Arrow Dissociated IPC (metadata stream
apart from body bytes) applied to **quantized tensors**, not Arrow
record batches. Do not take a `pyarrow` / Flight / UCX dependency
to get there.

## Control plane vs tensor data plane

Copy the split, not the protocol.

| Plane | What | Lives where today | Future (blocked) |
|---|---|---|---|
| **Control** | Place, migrate, caps, tickets, sequence, shape/dtype/quant, prefill-vs-decode class, admit/reject | `csd-model-router`, lab console, HTTP/JSON, ADR-0015 tags (`tight` / `ok-lan`, `pool_ok`) | Same router. Small metadata only. May stay HTTP. |
| **Tensor data** | Packed Q8/Q4 body | **does not exist** (whole-model placement) | Lean TCP between 3090 and 5080. Never WAN. Never 1080 Ti split. |

Arrow Dissociated IPC exists because a single IPC stream **interleaves**
Flatbuffers headers with packed body bytes. That forces a contiguous
host copy, blocks GPU-resident bodies, and mixes control with bulk.
The experimental spec splits:

- **IPC metadata** (Flatbuffers headers) on a delimited/tagged
  control stream.
- **Body** on a tagged data stream (raw packed buffers **or**
  shared/remote address pairs). Sequence numbers (uint32, wrapping)
  bind a header to its body so arrival order is not the protocol.

CSD needs the same **two-pipe** discipline: router tickets and
`(seq, nbytes, quant, shape)` on control; opaque body on data.
Do **not** implement Dissociated IPC, Flight, UCX, or CUDA IPC
across the VFIO 1080 Ti boundary.

Backpressure is transport-defined in that spec; we would define it
as: drop **ok-lan** prefill frames under memory pressure (retry on
control), never drop a committed decode-class buffer (but decode
should not be on this plane — see below).

## Prefill vs decode policy

ADR-0015 already decided this for RPC. The wire does not change it.

| Class | Latency tag | Batch | Wire? | Why |
|---|---|---|---|---|
| **Decode** (autodev `local/code`, tight chat) | `tight` | 1 token | **Never.** One card, one PCIe domain. | Per-token bytes are tiny; LAN RTT dominates. MSG_ZEROCOPY is a net loss below ~10 KB. |
| **Prefill** (prompt, batched hidden states, idle-expert shard) | `ok-lan` | large | **Maybe**, after unblock, if measured tensor-rate ≥ the local GEMM it replaces. | Bulk bytes; pipeline across the hop. Still never dual 14B. |
| **Weight intern / KV** | n/a | n/a | Weights intern **on one card** (`G-SHARE` horizon). KV stays private. Do not ship KV over LAN. | Isolation. |

Prefill may send **activations or idle-expert weights**, Q4/Q8,
`ok-lan` only. Decode samples, RNG, and KV stay local. A 14B 32k
GGUF never splits. `pool/large` / `pool/moe` stay `pool.enabled=false`
until Stage B RPC **or** this wire (not both at once) has a measured
row.

llama.cpp’s own multi-GPU notes: **layer/pipeline** split helps
prefill; **tensor** split is interconnect-bound and experimental.
LAN is slower than PCIe. Treat any LAN tensor split as prefill-only
until a decode-class benchmark (it will lose).

## Transport ladder (benchmark before climbing)

Climb only with a measured **effective tensor rate** (below). Do not
skip rungs because a paper looks faster.

| Rung | Use | Why later |
|---|---|---|
| **0. Placement** (live) | Whole model on one card. Stage A. | No wire. |
| **1. Lean TCP** (first to measure) | Length-prefixed Q4/Q8 bodies. `SO_ZEROCOPY` + `MSG_ZEROCOPY` on send when frame ≥ ~10 KB. | Trusted LAN, mature, works with pinned host. Kernel docs: zerocopy replaces memcpy with pin + completion; small writes lose. Loopback **always copies** — test **across hosts**, not localhost. |
| **2. TCP recv path** | Ordinary `recv` into pinned, or `io_uring` ZC Rx **if** the NIC does header/data split + flow steering + RSS. | zcrx is **not** DPDK; TCP headers still hit the kernel. Consumer NICs here are not assumed to support it. Probe; fall back. GPU memory as zcrx area is a later kernel feature, not available as a plan. |
| **3. QUIC streams** (RFC 9000) | Only if TCP is the bottleneck **and** independent streams / 0-RTT / migration win on **this** LAN. | Extra crypto on a trusted LAN is cost. Multiplex control+data on one UDP 443-alike is not a requirement. |
| **4. QUIC DATAGRAM** (RFC 9221) | **Wrong default for reconstruct.** DATAGRAM frames are **unreliable**, not retransmitted, not flow-controlled, cannot fragment, ack-eliciting but loss ≠ retry. | Use only for expendable telemetry, never for a tensor body that must reconstruct. If a future lossy preview exists, it is not this pack. |
| **5. RDMA / UCX / libfabric** | Arrow Dissociated IPC’s worked examples. Needs special NICs. | Lab has no InfiniBand plan. Do not buy a transport to skip measuring TCP. |

RFC 9221 in one line: `max_datagram_frame_size` (0x20) advertises
DATAGRAM support; frames 0x30/0x31 carry app bytes **without
retransmission**, still under QUIC congestion control. That is a
real-time / tunnel tool, not a tensor replica protocol.

`MSG_ZEROCOPY` in one line: `setsockopt(SO_ZEROCOPY)` then
`send(..., MSG_ZEROCOPY)`; completions on `MSG_ERRQUEUE` as
`SO_EE_ORIGIN_ZEROCOPY`; `SO_EE_CODE_ZEROCOPY_COPIED` means the
kernel copied anyway — stop passing the flag on that socket.

io_uring zcrx in one line: NIC DMA of **payloads** into a registered
userspace area; headers stay in kernel; needs `ethtool` tcp-data-split,
RSS carve-out, flow steer; `IORING_OP_RECV_ZC` multishot; recycle via
refill ring. Out of band NIC setup. Not a lab default.

## Cluster self-benchmark (effective tensor rate)

Do **not** report NIC line rate, `nvidia-smi` copy, or a blog Gbps
as the metric. The only number that matters:

```
effective_tensor_rate =
  quantized_body_bytes
  / wall_s(quantize + D2H_pinned + TCP_send+recv + H2D_pinned + dequant + reconstruct_ok)
```

Units: **MiB/s of Q4/Q8 payload that reconstructed bit-identical (or
scheme-identical) on the far GPU.** Also record:

| Field | Why |
|---|---|
| `path` | `prime→5080` or `5080→prime` (never 1080 Ti split) |
| `quant` | `Q8_0` / `Q4_K_M` / … |
| `frame_bytes` | payload only, no headers |
| `tcp_goodput_mib_s` | iperf3 same path, same MTU, same CPU pinning |
| `zerocopy` | `hit` / `copied` (`SO_EE_CODE_ZEROCOPY_COPIED`) / `off` |
| `zcrx` | `on` / `unsupported` / `off` |
| `reconstruct_ok` | bool |
| `p50_ms` / `p99_ms` | control+data |
| `prefill_or_decode` | must be `prefill` until decode-class loses honestly |
| `localai_still_up` | 3090 must stay loaded |

Pass a rung only if `effective_tensor_rate` beats **keeping the work
on one card** for that class (prefill bulk, not decode). Fail closed
on `reconstruct_ok=false`, OOM, or LocalAI bounce.

When unblocked, the first experiment is **TCP + pinned + Q4/Q8 echo**
between `192.168.1.98` and `192.168.1.251`. Not QUIC. Not RPC. Not
the 1080 Ti guest. Grafana may grow a panel **after** a real JSON
receipt exists — do not fake Pool-tab rates.

## G-POOL / ADR-0015 horizon

| Layer | Status | This pack |
|---|---|---|
| Stage A place/migrate/pack | **live** | Orthogonal. Keep. |
| Stage B llama.cpp RPC (`pool.enabled`, `--rpc host:port`) | **parked** | **Do not implement here.** ggml-rpc is PoC, TCP, **insecure on an open net**, LAN-only if ever. Distinct binary from a CSD-native wire. |
| Native tensor wire (this file) | **blocked** | After `G-LIFE` + measured TCP row + a new ADR choosing wire vs RPC (not both). |
| Stage C intern share (`G-SHARE`, 256 example) | **horizon** | Interned RO weights stay **on one card**. Wire is not how 256 specialists appear. |
| `G-SPLIT` one CSD mind | **blocked** Phase 5 | Not serving-pool. Not this. |
| `G-1080` | **do not start** | Guest is live for RAG only. |

ADR-0015 consequence that still binds: llama.cpp RPC (or any LAN
tensor hop) adds RTT on offloaded layers. Mitigation remains
`latency=tight` never splits.

## What autodev may do while blocked

- Keep Stage A router honest. Pack leftover. Never dual 14B.
- Prefer 1080 Ti guest for `csd-kb-index` / retrieve. Do not pause
  LocalAI for RAG.
- Close region-pretrain / Phase 1 product (`G-LIFE` path).
- Edit **this file** if citations or gates change. Do not add source
  under `src/` for a wire.

## What autodev must not do

Implement, prototype, or “just iperf in a service” that becomes a
daemon. No `src/cogsyndelta/**/wire*`. No Arrow crate/wheel for
this. No QUIC library. No `ggml-rpc-server`. No UDP. No RDMA verbs.

## Related

- [GPU-POOL.md](GPU-POOL.md) · [../adr/0015-lan-gpu-model-pool.md](../adr/0015-lan-gpu-model-pool.md)
- [GPU-SHARE.md](GPU-SHARE.md) · [../adr/0016-consumer-gpu-share.md](../adr/0016-consumer-gpu-share.md)
- [CSD-CLUSTER-FLEX.md](CSD-CLUSTER-FLEX.md) · [CSD-1080TI-RAG.md](CSD-1080TI-RAG.md)
- [GOALS.md](GOALS.md) (`G-POOL`, `G-LIFE`, `G-SHARE`, `G-SPLIT`)
- [WHO-RUNS-WHAT.md](WHO-RUNS-WHAT.md)

## Citations

Official docs this pack is based on (fetched 2026-08-31):

1. Apache Arrow — Dissociated IPC Protocol (experimental). Control
   stream vs body stream; sequence numbers; GPU-resident bodies
   without D2H of metadata.  
   <https://arrow.apache.org/docs/format/DissociatedIPC.html>
2. Apache Arrow blog — “Data Wants to Be Free” (Dissociated IPC +
   UCX/libfabric as the high-end path).  
   <https://arrow.apache.org/blog/2025/02/28/data-wants-to-be-free/>
3. RFC 9221 — Unreliable Datagram Extension to QUIC. DATAGRAM
   0x30/0x31; `max_datagram_frame_size` 0x20; no retransmission.  
   <https://www.rfc-editor.org/rfc/rfc9221.html>
4. RFC 9000 — QUIC: UDP-based multiplexed reliable streams (the
   thing to measure *after* TCP, not DATAGRAM).  
   <https://www.rfc-editor.org/rfc/rfc9000.html>
5. Linux — `MSG_ZEROCOPY` (TCP/UDP send copy-avoidance, ~10 KB
   floor, `SO_ZEROCOPY`, `MSG_ERRQUEUE` completions, loopback copy).  
   <https://www.kernel.org/doc/html/latest/networking/msg_zerocopy.html>
6. LWN — Zero-copy networking (`MSG_ZEROCOPY` background).  
   <https://lwn.net/Articles/726917/>
7. Linux — io_uring zero-copy Rx (`iou-zcrx`: header/data split,
   flow steering, RSS, `IORING_OP_RECV_ZC`).  
   <https://docs.kernel.org/networking/iou-zcrx.html>
8. LWN — Zero copy Rx using io_uring (proposal context).  
   <https://lwn.net/Articles/965214/>
9. NVIDIA CUDA — page-locked / pinned host memory (`cudaHostAlloc`,
   `cudaMallocHost`; staging, not a second heap).  
   <https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html>
10. NVIDIA CUDA Runtime — `cudaHostAlloc` (pinned staging; over-pin
    degrades host paging).  
    <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html>
11. llama.cpp RPC README — `ggml-rpc-server` over TCP; PoC; **never
    on an open network**; `--rpc host:port`. Research only; Stage B
    stays off.  
    <https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md>
12. llama.cpp multi-GPU — layer vs tensor split; tensor-parallel is
    interconnect-bound / experimental.  
    <https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md>
13. GGUF quant sizes (Q8_0 ~8.5 bpw, Q4_K_M ~4.5–4.9 bpw) — wire
    payload scale.  
    <https://github.com/ggml-org/llama.cpp/discussions/2094>
