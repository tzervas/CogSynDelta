# Interconnect Module Specification

**Status:** buildable specification, draft 2, for operator review before any lane starts. It derives
from the ratified taxonomy and decides nothing the taxonomy decides; where it fixes a detail the
taxonomy leaves open, the sentence is tagged `[spec]`. `TAX` below means
`docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md`, and every `TAX:NNNN` is a line of that file at
`3bc8c3e`. Draft 2 applies the eleven asks of the adversarial review of draft 1.

## 1. Summary and scope

**Summary.** This document turns taxonomy §2 into a module under `src/cogsyndelta/interconnect/`:
twelve files, a shape for every tensor boundary, a parameter table reconciled against the taxonomy's
27,424,039, the forward pass as numbered steps, the DEC-50 training contract with its receipts and
fail-closed guards, a test plan with a sub-30-second CPU smoke, four operator questions, and a
file-disjoint lane plan. Four of the taxonomy's seven parameter rows reproduce exactly from their
stated dimensions; the workspace, controller and type-embedding rows do not (section 2.3).

**What the module is.** White matter is a bounded Perceiver workspace: `L = 64` latents of width
`D_w = 512` cross-attend over a key/value bank built from every admitted region's position latents,
for `n_iter = 4` iterations, with a thalamic controller that decides budgets before any region runs
and a frontal read-out that chooses the output (TAX:1219-1246). Connection strengths are the
cross-attention weights themselves (TAX:1260-1262). The iteration loop is the latent-reasoning loop
the operator described as latent transform loops coupled to multi-head attention; nothing between
regions is a discrete token (TAX:1141-1145, memory note `csd-reasoning-faculty-definition`). The
module owns the store's two projections, the write-back prefixes, the receipt builder and the gate
functions; the regions and the store's data stay outside it (TAX:1266-1283).

**What it is not.** Not the regions, which arrive frozen through the W0 protocol
`src/cogsyndelta/faculty/protocol.py` with `Faculty.tokens()` and `pool()` (TAX:1094-1116). Not the
episodic store's build, which is row E1; this module ships the store's interface and an in-memory
stub with the DEC-63..66 gaps marked (TAX:2649). Not the dormant `core/` stack, which stays
harvested and superseded (TAX:1026-1051). Not the `contracts/region.py` `activate()` surface, which
this module supersedes (TAX:4827-4828). Not the compose-stage driver script, which imports
`gates.py` and `receipts.py` and runs the phases of section 4.

**v1 participants.** Table 1 lists the five (TAX:1266-1267). W5 trains at `R = 4` without the store;
E2 admits the store and retrains at `R = 5` (TAX:2661-2662). `D_w = 512` with a per-region
`nn.Linear(token_dim, 512)` adapter is applied as decided, and `MindSpec`'s single-`stream_dim`
check is replaced by lane IC-11 (TAX:1204-1211, TAX:4828).

Table 1 — v1 participants and the per-region numbers this module consumes (TAX:774-888);
`ctx_min` and `ctx_max` are this spec's declaration, Table 4a.

| participant | kind | `token_dim` | `pooled_dim` | `kv_bytes_per_token` | `token_budget` min / default / max | `ctx_min` / `ctx_max` | `accepts_condition` |
|---|---|---:|---:|---:|---|---|---|
| `language` | contrastive_encoder | 256 | 256 | 4,096 | 1 / 64 / 96 | 8 / 96 | yes |
| `memory` | contrastive_encoder | 256 | 256 | 4,096 | 1 / 64 / 96 | 8 / 96 | yes |
| `reasoning` | contrastive_encoder | 256 | 256 | 4,096 | 1 / 128 / 256 | 8 / 256 | yes |
| `visual` | jepa_predictor, EMA target encoder | 384 | 384 | 9,216 | 1 / 64 / 64 | 8 / 64 | yes |
| `episodic_store` | nonparametric_store | 512 | 512 | — | 8 / 32 / 256 | — | no |

`visual` emits `[B, 64, 384]` today and `[B, 256, 384]` after W7v, which is blocked on OD-4
(TAX:1236-1242, TAX:2654-2655). `token_budget.default` is not used by this module: phase A and the
fallback `Schedule` use the dense allocation of section 3 step 2, because both the defaults (352)
and the maxima (768) exceed the 256-slot bank. `reasoning`'s current checkpoint carries a lookup
objective the operator has ruled is the wrong faculty objective; the module treats it as one more
frozen participant (memory note `csd-reasoning-faculty-definition`).

## 2. Module layout

### 2.1 Files and classes

One file per component. Four files are additions to the requested list: `mind.py` is the composition
root, `losses.py` holds the loss terms, `gates.py` the receipt-level guards and `receipts.py` the
receipt builder, so the compose-stage script imports rather than re-derives.

Table 2 — files under `src/cogsyndelta/interconnect/`, their classes and constructor arguments.

| file | classes | constructor arguments | owns |
|---|---|---|---|
| `schedule.py` | `Schedule`, `ScheduleNode`, `StepBudget`, `OutputSpec` (frozen dataclasses); `ScheduleValidator` | `ScheduleValidator(participants, B_read, B_kv, n_iter, eta, flops_ceiling, allowed_modalities, resident_heads)` | the §2.5 emission and every §9.9 B1 bound; `trace_id` is never a field the model fills |
| `adapters.py` | `RegionAdapter`, `TopKSelect`, `ConditioningPrefix` | `RegionAdapter(token_dim, D_w)`; `TopKSelect(token_dim)`; `ConditioningPrefix(D_w, token_dim, n_cond)` | `Linear(d_r, 512)`; straight-through top-`b_r`; `Linear(512, n_cond·d_r)` plus a per-region attention-pool query `[spec]` |
| `kv_bank.py` | `KVBank`, `StoreProjection` | `KVBank(participants, D_w, B_read)`; `StoreProjection(D_w)` | the packed bank, `slot_region`, type embeddings, sincos positions, `W_k` and `W_v` |
| `workspace.py` | `LatentBank`, `WorkspaceBlock`, `Workspace` | `Workspace(D_w, L, n_iter, heads, mlp_ratio)` | the latents, `n_iter` untied blocks, the attention-mass export |
| `readout.py` | `FrontalReadout`, `RankHead`, `NullCandidate`, `UnifyProbes` | `FrontalReadout(D_w, mlp_ratio)`; `RankHead(D_w, k, temperature)`; `NullCandidate(D_w)`; `UnifyProbes(D_w, pooled_dims)` | the DEC-20 read-out, DEC-41 ranking, the `NULL` embedding, the `L_unify` probes, the `z_affect` seam behind a zero gate |
| `controller.py` | `ThalamicController`, `FlooredSimplex`, `box_integerise` | `ThalamicController(participants, d_ctrl, depth_ctrl, heads_ctrl, n_iter, B_read, B_kv, eta)` | the cheap summary, two blocks, four heads, both simplexes, admission, halt |
| `episodic_store.py` | `EpisodicStore` (Protocol), `InMemoryStoreStub`, `ContractGap` | `InMemoryStoreStub(domain_enum, half_life_s, importance_default)` | the E0 interface; every DEC-63..66 gap raises `ContractGap` naming its DEC |
| `losses.py` | `RankLoss`, `UnifyLoss`, `DistilLoss`, `FlopsPenalty` | the per-loss weights | `L_task`, `L_unify`, `L_B`, the `λ_flops` term |
| `gates.py`, `receipts.py` | `GateFailure` plus one function per receipt-level guard; `ComposeReceipt` | receipt dicts; `ComposeReceipt(stage, identity)` | G27, G28, G29, G33, G34, G35, G36 of Table 8; every Table 7 field, written through `write_receipt` |
| `mind.py` | `InterconnectConfig`, `WhiteMatter` | `WhiteMatter(config, faculties: dict[str, Faculty], store: EpisodicStore)` | the forward pass of section 3, the per-request `h_r` cache, the frozen-schedule injection path |
| `__init__.py` | — | — | re-exports; `__all__` is the public surface |

`InterconnectConfig` defaults are the §1.4 interconnect block — `workspace_dim 512`, `latents 64`,
`n_iter 4`, `heads 8`, `mlp_ratio 4`, `budget_total_read_tokens 256`,
`budget_total_kv_bytes 3221225472`, `floor_eta 0.15` — plus `n_cond 8`, `controller_dim 256`,
`controller_depth 2`, `controller_heads 4`, `flops_ceiling`, `write_back True`, `k_candidates 32`
and `unify_probes_trainable True` (TAX:736-741, TAX:1428, TAX:1684-1688). `flops_ceiling` defaults
to the dense schedule's own `region_token_flops`, `Σ_i Σ_r φ_r · ctx_max_r` over `n_iter`
iterations, the worst case B1 can reach (TAX:5832-5834); construction refuses a smaller value, so
G30's dense fallback can never be refused by its own validator. **One name for the iteration
count.** `n_iter` is the number of workspace blocks and the upper bound on iterations; `I` in
formulas is `n_iter`; a request's `step_budget.max_iters ≤ n_iter` is the bound the validator
checks; `halt_at ≤ max_iters` is the realised value. Iteration `i` runs block `i` with untied
weights, so halting at iteration 2 skips blocks 3 and 4 (TAX:1563-1568). Tying the blocks is the
recurrent-depth seam and is not built (TAX:4612-4614).

### 2.2 Shapes at every boundary

Symbols: `B` batch, `L = 64`, `D = 512`, `R` participants, `R_ctx = 4` encoding participants,
`I = n_iter = 4`, `H = 8` heads, `T_r ≤ ctx_r` positions region `r` encoded, `b_r` its read tokens,
`n_cond = 8`, `k = 32` candidates. Budgets are per item, so within a batch a region runs at the
batch maximum `ctx_r` and the per-item budget mask enforces each item's own budget `[spec]`. Mask
polarity is uniform: `True` means valid, attendable or executed.

Table 3 — tensor shapes at every module boundary.

| tensor | shape and dtype | produced by → consumed by |
|---|---|---|
| text ids, images | `[B, T_in]` int64; `[B, 3, 64, 64]` float | tokenizer → controller summary and `Faculty.tokens` |
| candidate inputs | `k − 1` text or image candidates per item, the same shapes | item manifest → the candidate encoder of Q7 |
| controller summary `s` | `[B, R+1, 256]` float | `controller.summarise` → controller blocks |
| `ctx` | `[B, R_ctx]` int; `Σ_r c_r·ctx_r ≤ B_kv`; `ctx_min ≤ ctx_r ≤ ctx_max_r` | controller → runtime, as `context_tokens`; the store has no `ctx`, its context is the scope partition |
| `b` | `[B, R]` int; `Σ_r b_r = B_read`; `lo_r ≤ b_r ≤ token_budget.max_r` | controller → `KVBank` |
| `A` | `[B, I, R]` bool; `Σ_i A[i, r] ≥ 1` | controller → runtime and key mask |
| `halt_logits`, `halt_at`, `active` | `[B, I]` float; `[B]` int `≤ max_iters`; `[B, I]` bool, `True` = iteration executed | controller → runtime loop bound; `active` → receipts |
| `h_r`, `mask_r` | `[B, T_r, d_r]` float; `[B, T_r]` bool, `True` = real position | `Faculty.tokens` → `TopKSelect` |
| `budget_mask_r` | `[B, T_r]` bool, `True` where `t < ctx_r[b]`; ANDed into `mask_r` | runtime → `TopKSelect` |
| `cond_r` | `[B, n_cond, d_r]` float | `ConditioningPrefix` → `Faculty.tokens(condition=)` |
| selected `h_r`, adapted tokens | `[B, b_r, d_r]` float; `[B, b_r, D]` float | `TopKSelect` → `RegionAdapter` → `KVBank` |
| store latents | `[B, b_store, D]` float plus mask | `EpisodicStore.read` → `StoreProjection` → bank |
| `bank_k`, `bank_v`, `key_mask`, `slot_region` | `[B, 256, D]`; `[B, 256, D]`; `[B, 256]` bool, `True` = attendable; `[B, 256]` int in {−1 .. R−1} | `KVBank` → `Workspace`; `bank_k` and `bank_v` are identical on region slots and differ only on store slots |
| `z` | `[B, L, D]` float | `LatentBank` → blocks → `FrontalReadout` |
| cross-attention weights, `a` | `[B, H, L, 256]` float; `[B, I, R]` float, `a[b, i, :]` sums to 1 over admitted regions where `active[b, i]` and is all zero where not | block `i` → attention-mass export → receipt, `Schedule.intensity`, phase-B target |
| `f` | `[B, D]` float | `FrontalReadout` → `RankHead`, `UnifyProbes`, store write |
| rank scores | query `[B, D]`; candidates `[B, k, D]` with `NULL` at index 0; scores `[B, k]` | `RankHead` → `RankLoss` |
| probe outputs | `[B, pooled_dim_r]` per region | `UnifyProbes` → `UnifyLoss` |
| `Schedule` | one JSON object per item | `WhiteMatter` → `ScheduleValidator` → runtime → receipt |

Two shape rules are load-bearing. Every tensor crossing a tract is a float tensor at a declared
`*_dim`; an integer-typed or vocabulary-indexed payload on `h_r`, the adapted tokens, `z` or
`cond_r` raises (TAX:2667, TAX:1126-1166). The bank has a fixed width of `B_read = 256` slots with
`slot_region` naming each slot's owner, so admission, I2′ severance and `a` are masks and
scatter-adds over one static shape `[spec]`.

### 2.3 Parameters

Table 4 — parameter count per component, derived from the dimensions above, beside the
taxonomy's row (TAX:1275-1284, TAX:6435-6446). Biases are counted except where a row says
otherwise; a LayerNorm affine pair is `2·D`.

| component | arithmetic | this spec | taxonomy | Δ |
|---|---|---:|---:|---:|
| workspace blocks ×4, attention and MLP | per block `4·512²` self + `4·512²` cross (no bias, amendment A1 below) + `(512·2048+2048) + (2048·512+512)` = 4,196,864 | 16,787,456 | 16,803,840 | −16,384 |
| workspace LayerNorms | 4 blocks × 4 norms × 1,024 | 16,384 | not counted | +16,384 |
| frontal read-out | query 512 + attention 1,050,624 + MLP 2,099,712 | 3,150,848 | 3,150,848 | 0 |
| thalamic controller | 2 blocks @256 (2 × 788,736) + norms 2,048 + summary and heads 104,458 | 1,683,978 | 1,588,007 | +95,971 |
| conditioning prefixes | `3·(512·2048+2048) + (512·3072+3072)` | 4,727,808 | 4,727,808 | 0 |
| prefix attention-pool queries `[spec]` | 4 × 512 | 2,048 | not counted | +2,048 |
| region adapters | `3·(256·512+512) + (384·512+512)` | 591,872 | 591,872 | 0 |
| top-k scorers | `3·(256+1) + (384+1)` | 1,156 | not counted | +1,156 |
| store projections `W_k`, `W_v` | `2·512²`, no bias | 524,288 | 524,288 | 0 |
| latent bank, final norm, type embeddings | `64·512 + 1,024 + R·512` at `R = 5` | 36,352 | 37,376 | −1,024 |
| **white matter core, `R = 5`** | | **27,522,190** | **27,424,039** | **+98,151** |
| **white matter core, `R = 4`** | one type row and one controller slot row fewer (−768); `W_k`/`W_v` and the store summary stay instantiated and receive no gradient in W5 | **27,521,422** | | |
| rank head (DEC-41) | `512·512 + 512` | 262,656 | not in the table | — |
| `NULL` candidate embedding (DEC-41) | one learned vector of 512 | 512 | not in the table | — |
| unify probes (DEC-20) | `3·(512·256+256) + (512·384+384)` | 590,976 | not in the table | — |
| **trainable in phase A, heads included, `R = 5`** | | **28,376,334** | | |
| candidate encoder, only under Q7 option (a) | `Linear(1152, 512)` over the concatenated frozen pooled outputs | 590,336 | not in the table | — |
| **with Q7 (a)** | | **28,966,670** | | |

**Amendment A1 (2026-09-06): the workspace blocks' attention projections carry no bias.**
`WorkspaceBlock` (`workspace.py`) builds all six attention projections -- `cross_q`, `cross_k`,
`cross_v`, `cross_out`, `self_qkv` (fused Q/K/V) and `self_out` -- with `bias=False`, which is
standard practice for a pre-norm transformer block: a `LayerNorm` immediately upstream of each
projection already supplies a learned shift, so the projection's own bias is redundant, and
dropping it removes 4,096 parameters per block for no measured loss. This table originally
counted them. The code is the version kept; the arithmetic above is amended to match, rather
than adding biases back to satisfy a table.

Re-derivation, per block: self-attention `4·512² = 1,048,576`, cross-attention
`4·512² = 1,048,576`, MLP `(512·2048 + 2048) + (2048·512 + 512) = 2,099,712`; total
`4,196,864`, and `×4 blocks = 16,787,456`. The dropped biases are `4·512` (cross q/k/v/out)
`+ 3·512` (fused self QKV) `+ 512` (self out) `= 4,096` per block, `16,384` over the four
blocks. Every total below the workspace row falls by that same 16,384: the `R = 5` core to
27,522,190, the `R = 4` core to 27,521,422, phase-A trainable to 28,376,334, and the Q7 (a)
figure to 28,966,670. The MLP's two `Linear`s keep their biases, as do the read-out, the
controller, the adapters and the conditioning prefixes; only the workspace attention
projections are affected. Measured against the code at `R = 5`: 27,522,190.

Four rows reproduce the taxonomy exactly from its own dimensions. Three rows do not: the
workspace row now differs from it by exactly amendment A1's 16,384. The taxonomy's
controller figure was measured at three heads and names no dimensions for its summary or heads, so
its 10,535-parameter residual over two blocks cannot be rebuilt from the text; Table 5 fixes the
heads explicitly and lands inside the taxonomy's own ±0.1M (TAX:1279). The taxonomy's type-embedding
row holds seven type rows, the revision-1 list, and at `R = 5` it is 1,024 smaller. The rank head,
the `NULL` embedding and the unify probes are absent from the taxonomy's table although phase A
trains them; W10's footprint must count them, and W0's re-instantiation is the number that replaces
every figure here (TAX:1287-1291, TAX:2669).

Table 4a — context budgets per encoding participant `[spec]`; the taxonomy names `ctx_min` and
`ctx_max` without values (TAX:1395, TAX:5824).

| region | `ctx_min` | `ctx_max` | why |
|---|---:|---:|---|
| `language`, `memory` | 8 | 96 | trained at `max_len 96` (TAX:2506) |
| `reasoning` | 8 | 256 | trained at `max_len 256` (TAX:132-133) |
| `visual` | 8 | 64 | `n_patches` today; 256 after W7v (TAX:1236-1242) |

`ctx_min = 8` is one prefix length, the same order as `n_cond`. From these values
`Σ_r c_r · ctx_max_r = 4,096·(96 + 96 + 256) + 9,216·64 = 2,424,832` bytes against a 3 GiB `B_kv`,
and 4,194,304 bytes after W7v, so at v1 `ctx_max` binds and `B_kv` never does.

Table 5 — controller summary and heads, the part the taxonomy leaves undimensioned `[spec]`.

| piece | shape | params | note |
|---|---|---:|---|
| text summary | mean of `language`'s frozen embedding rows over the input ids → 256 | 0 | borrowed and frozen; a bag-of-tokens summary that runs no region |
| visual summary | `Linear(384, 256)` over the mean of the frozen patch embedding | 98,560 | |
| store summary | `Linear(4, 256)` over occupancy statistics of the scope partition | 1,280 | resident count, bytes, oldest age, newest age |
| slot embeddings | `[R+1, 256]`, one per participant plus a CLS slot | 1,536 | |
| final norm | 512 | 512 | |
| `ctx_head`, `b_head` | `Linear(256, 1)` each, applied per participant slot | 514 | logits `s_ctx`, `s_b` for the two simplexes |
| `A_head` | `Linear(256, I)` per participant slot | 1,028 | admission logits per iteration |
| `halt_head` | `Linear(256, I)` on the CLS slot | 1,028 | per-iteration halt probability |

The controller stays in high precision through every quantisation gate, because an error there
changes which faculties ran (TAX:4191-4196).

## 3. The forward pass

Steps 1-4 happen before any region runs; steps 5-11 repeat per iteration; steps 12-14 close the
request. This is the loop the taxonomy sizes at about 120 lines (TAX:1563-1577).
`WhiteMatter.forward(inputs, schedule=None)`: when a validated `Schedule` is passed, steps 1-3 are
skipped and it is executed as given; this is the frozen-schedule arm of the budget-as-tag control
(TAX:1983).

1. **Summarise.** Build `s [B, R+1, 256]` from raw inputs without running any region, add the slot
   embeddings, run the two controller blocks.
2. **Budgets.** `β_ctx = η/R_ctx + (1−η)·softmax(s_ctx)` over the four encoding participants;
   `ctx_r = clip(floor(β_ctx,r · B_kv / c_r), ctx_min, ctx_max_r)`; clipping down never breaks the
   byte bound, and clipping up cannot because construction refuses a config where
   `η/R_ctx · B_kv < c_r · ctx_min` for any `r`. `β_b = η/R + (1−η)·softmax(s_b)` over all five,
   then `b = box_integerise(B_read · β_b, lo, hi)` with
   `lo_r = max(token_budget.min_r, ceil(η/R · B_read))` and `hi_r = token_budget.max_r`: start every
   region at `lo_r`; share the pool `B_read − Σ lo` among the uncapped regions in proportion to
   `β_b`, as real values; cap any region whose share would carry it past `hi_r` at `hi_r` and return
   its unused share to the pool; repeat until no region is newly capped or every region is at
   `hi_r`, which terminates because each round caps at least one more region; then integerise the
   uncapped shares by largest remainder over the remaining integer pool, ties on equal remainders
   going to the earlier participant in Table 1 order. `Σ b_r = B_read` exactly, every `b_r` is
   inside its box, and construction refuses `Σ lo > B_read` or `Σ hi < B_read`
   (TAX:1401-1416, TAX:5824-5825). At `R = 5` every `lo_r` is 8, the store's floor; at `R = 4` it is
   10. **Phase A and the fallback `Schedule` use the dense allocation** `[spec]`: `A ≡ 1`,
   `ctx = ctx_max`, `halt_at = max_iters = n_iter`, and `b = box_integerise(uniform)`, which is
   `[64, 64, 64, 64]` at `R = 4` and `[52, 51, 51, 51, 51]` at `R = 5`. This is what "all regions on
   at `ctx_max`, `b_max`" can mean inside a 256-slot bank, since `Σ b_max = 768`
   (TAX:1645-1647, TAX:1196).
3. **Admission and halt.** `A = 1[σ(A_logits) > 0.5]` with a straight-through sigmoid, forced so
   every declared region is admitted at least once at its argmax iteration. `halt_at` is the first
   `i` at which the cumulative halt probability reaches `1 − ε`, else `max_iters`; the runtime bound
   is `min(halt_at, max_iters)` whatever the head says, and `n_iter` bounds both by construction
   (TAX:5826).
4. **Emit and validate the pre-execution `Schedule`.** Nodes carry `active`, `depth`,
   `admitted[n_iter]`, `context_tokens`, `read_tokens`, `condition`, `priority`, `precision`,
   `resident` and `codec`; `step_budget` carries `max_iters`, `kv_bytes`, `read_tokens`, `wall_ms`
   and `flops_ceiling`; the schedule carries its predicted `region_token_flops`; `output.modalities`
   is checked against the request's `allowed_modalities` and the resident heads
   (TAX:1494-1518, TAX:1616-1626). A `Schedule` carrying a `trace_id`, a store namespace, a modality
   with no resident head, a malformed `admitted` or a `depth` that is not `min{i : admitted[i]}`, a
   `region_token_flops` over the ceiling, or any bound violation is refused before execution and the
   dense fallback runs (TAX:1521-1524, TAX:5827-5828, TAX:5854); Table 8a disposes of the rest.
5. **Condition.** For `i ≥ 1` and every admitted region with `accepts_condition`,
   `cond_r = reshape(Linear_r(pool_r(z_{i−1})), [n_cond, d_r])`, where `pool_r` is the region's
   attention pool over the latents (TAX:1427-1431). With `write_back = False` the argument is `None`
   at every iteration.
6. **Encode.** Run `tokens(inputs, context_tokens=ctx_r, condition=cond_r)` for `{r : A[i, r] = 1}`,
   concurrently on separate streams, reading the per-request cache when `(ctx_r, cond_r)` is
   unchanged; the cache holds at most `R × I` entries and dies with the request (TAX:1563-1577). A
   region with `A[i, r] = 0` is neither encoded nor read. Across the phase-2 training set only
   depth-0 emissions are cacheable once write-back is on (TAX:4400).
7. **Select and adapt.** `TopKSelect` scores each position with `Linear(d_r, 1)`, keeps the top
   `b_r` by hard selection among `mask_r ∧ budget_mask_r` positions with ties broken by position,
   and passes gradient to the scorer through a straight-through gate; `RegionAdapter` maps the
   selection to `D`.
8. **Assemble the bank.** Pack the adapted tokens into `[B, 256, D]` in fixed participant order,
   write `slot_region`, add `type_emb[r]` and `sincos(pos)` to both streams, and clear `key_mask`
   for empty slots and for slots whose region is not admitted at `i`. Store slots take `W_k z` into
   `bank_k` and `W_v z` into `bank_v` over the latents `EpisodicStore.read` returned for the
   request's server-derived scope, ranked by residency score; they receive `type_emb[store]` and no
   positional term `[spec]` (TAX:1253). `b_store` is floored at `lo_store = 8` so that a store
   nobody attends to is a measurement, and an empty partition yields zero unmasked store slots,
   which is the no-store configuration (TAX:1412-1414, TAX:5857).
9. **One workspace block.** `z += CrossAttn(LN(z), LN(bank))`, `z += SelfAttn(LN(z))`,
   `z += MLP(LN(z))` (TAX:1254-1256).
10. **Export connection strengths.**
    `a[:, i, r] = (1 / (H·L)) · Σ_h Σ_l Σ_{t : slot_region[t] = r} w[:, h, l, t]`, a scatter-add
    over `slot_region` on the materialised softmax, so `a[b, i, :]` is a distribution over the
    admitted regions; for `i ≥ halt_at` it is zero and `active[b, i]` is false, and every receipt
    statistic averages over active iterations only. The block runs SDPA unless `return_attention` is
    set, and the two paths must agree to 1e-4 (TAX:1257-1262).
11. **Halt check.** Stop when `i + 1 == halt_at`.
12. **Read out.** `f = FrontalReadout(z_N)`, a learned-query attention pool over the latents plus
    its MLP (TAX:2117-2119). `RankHead` maps `f` and each candidate embedding to `[B, D]` and scores
    by cosine over `k = 32` candidates, with `NullCandidate` supplying the learned embedding at
    index 0 (TAX:1684-1690). How the 31 content candidates are embedded is Q7.
13. **Write the store.** One record per turn, the mean over the 64 latents of the final-normed `z_N`
    `[spec]`, under `(scope, domain, logical_key)` with the default importance and a provenance
    sidecar that is never on the read path; this is the workspace's own write-back path, one latent
    per turn, as decided (TAX:586, TAX:4847, TAX:5069).
14. **Finalise the `Schedule` and the receipt fields.** `intensity` per node is the mean of
    `a[:, :, r]` over active iterations; `halt_at` is the realised value; `edges`, `stages` and
    `lockstep_groups` come from the two topology derivations only when write-back is enabled,
    otherwise the fields are omitted and `topology: not demonstrated` is recorded (TAX:1443-1490).

**Amendment A2 (2026-09-06): step 1's "without running any region" is not met, and cannot be
behind the W0 contract.** Table 5's text and visual summary features are a mean over `language`'s
embedding rows and a mean over `visual`'s patch embedding -- both read a region's own weights. The
frozen-region protocol this module is built against exposes `tokens()` and `pool()` only, and
Table 2 states the interconnect "does not reach back into a region's implementation", so no generic
path to those weights exists from `mind.py`. `WhiteMatter._raw_summary` instead runs each region
once at its own Table 4a `ctx_min` and takes `pool()`'s output as the raw feature; the widths are
unchanged (`pooled_dim` already equals Table 5's summary widths). The cost is one minimum-budget
encode per region per request, on the `schedule=None` arm only. This is recorded as a standing
spec-versus-code tension, not closed: closing it needs a W0 protocol addition (a declared
`raw_summary()` a region may implement) or a per-region embedding accessor, either of which is a
change to the region contract and out of the interconnect's scope. Until then, step 1 reads
"without running any region beyond one `ctx_min` encode each".

**Amendment A3 (2026-09-06): `intensity` lives on the receipt, not on `ScheduleNode`.** Step 14
above asks for a per-node `intensity`, the mean of `a[:, :, r]` over active iterations. Table 8a's
`ScheduleNode` field set has no such field, and adding one would make a `Schedule` carry a result
of the execution it configures -- which breaks the frozen-schedule arm, where `forward(inputs, s0)`
must re-emit `s0` byte-identically no matter what the tokens were, and an `intensity` computed from
those tokens could not. The value itself is not lost: Table 7's receipt already records "`a` mean
per active iteration and region", which is the same quantity, and `receipts.py` builds it. A
per-node `intensity` field is deferred until some consumer needs it on the `Schedule` specifically;
`ScheduleNode` stays a pure input to execution.

**Amendment A4 (2026-09-07): the store read takes a query, and the write identity is per
item.** Step 8 above ranks the store's latents "by residency score" and names no query, so
`EpisodicStore.read` took none. That made the read a pure function of `(scope, domain)`.

What that cost, measured: every item of a batch shares one scope, so the store bank was a
constant -- max absolute difference `0.0` across items, across batches, and before versus
after 50 training steps. Mutual information with the target was exactly zero, so
`dL/d(store attention)` was ~0 and the store's attention direction was unidentified; it
random-walked. Two further defects compounded it. Ranking ties broke toward the OLDER
record, and every write on the forward path takes the default importance, so a resident set
of primers permanently outranked everything the model wrote (an 8-record store and a
32-record store trained to bit-identical results, to 17 significant figures). And step 13's
"one record per turn" was implemented once per REQUEST rather than once per item, so
last-write-wins collapsed each batch to a single record: 12 records after 800 steps at
`B = 8`, not 6,408.

The amendment, in three parts:

| Part | Before | After |
| --- | --- | --- |
| step 8 ranking | residency score only | cosine to a `[D]` query, residency score as the tie-break; `query=None` keeps the old order |
| read tie-break | older record first | newer record first, matching the direction eviction already spills in |
| step 13 identity | `(scope, domain, logical_key)` per request | per item -- an explicit `logical_keys` sequence, else the request key suffixed with the batch index |

The query is the item's own `z_N`, taken from a store-free no-grad pre-pass. Measured
against that: MI excess +1.58 to +1.86 bits, held-out accuracy 0.855-0.996 against a null of
0.29, and 98.6% of items retrieving a top-1 record of their own class at step 0, before any
optimizer step, over memories whose maximum off-diagonal cosine was 0.9993. Near collinearity
compresses the scores but not their order.

`query` is CONTENT and `scope` stays ISOLATION. Per-item scoping was measured as an
alternative and failed below its own noise floor (MI under null, held-out accuracy under
chance at all four seeds), and a content-derived scope key would repurpose the
`derive_scope`/G32/DEC-64 boundary as a content index -- different principals' memories in
one partition. The two axes stay separate.

**Amendment A5 (2026-09-07): the store is read once per request, not once per iteration.**
`WhiteMatter._read_store` runs before the iteration loop and `_write_store` after it, so the
workspace cannot re-query the store as it refines `z` within a turn. Step 8 sits inside the
per-iteration sequence in this section's numbering, which reads as though it should. Moving
it there is mid-loop retrieval: it needs interconnect re-entrancy the `Schedule` cannot
express today (the slot layout is fixed for the whole request by `KVBank`'s own contract, and
`b_store` is a request-wide budget), and it changes what a turn IS rather than repairing a
defect. Recorded as an architectural decision for the operator, not carried by A4.

**Admission matrix semantics.** `depth(r) = min{i : A[i, r] = 1}`; a region with `depth 0` and
`A[:, r] ≡ 1` is asynchronous; regions first admitted at the same `i ≥ 1` with `accepts_condition`
form a lockstep group, because each reads `z_{i−1}` and writes `z_i` (TAX:1443-1456). The
cross-check `Ĉ[i, j] = Σ_t ⟨a[:, t, i], w[:, t+1, j]⟩ / Z` needs `w`, the weight with which each
latent fed region `j`'s prefix; with a mean pool `w` is uniform and `Ĉ` degenerates to a per-source
marginal, which is why this spec makes the prefix pool a per-region attention pool `[spec]`
(TAX:1461-1470).

**Severance for I2′.** Masking region `i`'s slots at every iteration below `depth(j)` and zeroing
its share of `j`'s prefix is a mask on `key_mask` and on the prefix pool, with no weight changed
(TAX:1996-2001).

**Affect seam.** `FrontalReadout` declares a named `z_affect` input behind a zero gate with no
parameters instantiated at v1; the workspace blocks and the region inputs never see it. Affect is
not a participant at v1, isolation first, as decided (TAX:700-708, TAX:5660).

## 4. The training contract (DEC-50)

DEC-50 is three steps: train the submodel alone, retrain the interconnect with it admitted, then
whole-mind training; this module is step 2's object, and its phases are DEC-19's A → B → C → D with
regions frozen throughout (TAX:2723-2737, TAX:1645-1755).

Table 6 — trainable sets, losses and gates per phase; regions are frozen at their retrained checkpoints in every row.

| phase, row | receives gradient | frozen | loss | gate, dev half only |
|---|---|---|---|---|
| A dense: W5 (`R = 4`), E2 (`R = 5`) | workspace, LayerNorms, adapters, scorers, type embeddings, latent bank, frontal, rank head, `NULL`, unify probes; `W_k`/`W_v` in E2; prefixes only when write-back is on | controller (bypassed), regions | `L_A = L_task + δ·L_unify` | `min_r mean(a_r) ≥ η/R`, printed as the expression and its value beside the receipt's own `R`; G1, G2, G0; overfit gap < 5 points; the `NULL` gate (TAX:1659-1669, TAX:1701-1702, TAX:2661) |
| W5b write-back | as A, with the prefixes and prefix queries enabled | as A | `L_A` | conditioned regions' own-bin drop ≤ 1 point and the composed metric improves, else write-back off and `topology: not demonstrated` (TAX:1433-1439, TAX:2663) |
| E2 admit the store | as A at `R = 5` plus `W_k`/`W_v`; episode items X7 and X8 in the mix | as A | `L_A` | G2 holds; the composed metric improves over W5; `mean(a_store) ≥ η/R = 3.0%`, else the store is reverted and `R` returns to 4 (TAX:2662). **AMENDED 2026-09-07 BY A6, INTERPRETATION ONLY:** the floor clause, its threshold and the revert are unchanged; a pass on it is **necessary, not sufficient** — it establishes that mass was allocated to the store's slots, never that the store carried information or that the composite used what it read |
| B distil: W8 | controller only | everything else; unify probes stay frozen from here on | `L_B = Σ_i Σ_r KL(softmax_r(a/τ) ‖ softmax_r(ŝ/τ)) + β·\|Σ_r b̂_r − B_read\|`, the softmax over the region axis per item and active iteration, plus the halt target `[spec]` | Spearman `ρ(ŝ, a) > 0.6` held-out; a region below the phase-A floor is excluded from the targets and named (TAX:1715-1727) |
| C sparse: W8 | workspace, adapters, prefixes, heads; the controller through its straight-through estimators | regions | `L_task + distil(dense read-out)`, ε-exploration at `p = 0.1` on one region's budget | within 2% relative of dense at ≤ 50% of region-token FLOPs; an accuracy gain here is a bug report against A (TAX:1729-1736) |
| D task-loss: W8 | controller, workspace, prefixes, adapters, heads | regions | `L_D = L_task + δ·L_unify + λ_flops·relu(FLOPs / FLOPs_target − 1)` | beats C on the composed metric at equal or lower FLOPs, else reverted with `scheduler: imitative` (TAX:1738-1755) |

**Loss definitions.** `L_task` is softmax cross-entropy over the `k = 32` cosine scores at a
recorded temperature (TAX:1684-1690). `L_unify` is `Σ_r (1 − cos(probe_r(f), pool_r(h_r)))` over the
admitted regions (TAX:1708-1713). The taxonomy calls the probe "frozen"; this spec reads that as
frozen after phase A `[spec]`: the probes train in A, which makes "linearly sufficient" a
measurement rather than a fixed random basis, and freeze for B-D so later phases cannot move the
target. The other reading, frozen at a seeded random init, is the one-flag switch
`unify_probes_trainable = False` and moves 590,976 parameters from trainable to buffers. Phase B's
halt target is the first iteration at which `cos(f_i, f_I) ≥ 0.99` on the dense run `[spec]`; the
taxonomy names no halt teacher. `FLOPs` is `Σ_i Σ_r A[i, r]·φ_r·ctx_r`, with `φ_r` the region's
measured MACs per position, differentiable through the straight-through paths, and the quantity the
validator's `flops_ceiling` bounds. The hinge that rewards every region for being individually
necessary is not implemented (TAX:1763-1769).

**Receipts.** Every phase writes one receipt through `ComposeReceipt`, which fills Table 7 and
writes through `write_receipt`, so the stamp `metrics_schema: "csd-metrics/v2"` is applied once
(`src/cogsyndelta/regions/_receipt.py:57,206`). The envelope is `model-pipeline-receipt/v1` with
`stage: "compose"` for A, W5b and E2 and `stage: "schedule"` for B, C and D; `"schedule"` is an
addition to `STAGES` (`src/cogsyndelta/pipeline/receipt.py:78`, TAX:4789). `compare()` refuses any
pair whose identity fields differ (`src/cogsyndelta/eval/metrics.py:691-716`, TAX:5631-5641).

Table 7 — receipt fields the compose and schedule stages must write.

| group | fields |
|---|---|
| identity (`MetricIdentity`) | `metrics_schema`; `corpus.fingerprint` and `corpus.fingerprint_scheme` of the reserve manifest; `battery_id` (`compose-dev/v1`, `compose-sealed/v1`, or `plumbing`); `k`; `pooling: "frontal"`; `artifacts.checkpoint_sha256` of the white-matter checkpoint; `region: "white_matter"`; `code_revision.git_sha`; `seed`; `split.sha256` (G26) |
| frozen set | `regions[] {name, checkpoint_sha256, receipt_path, status}`, `participants`, `R`; phase A refuses to start on a mismatch (G33) |
| budgets | `B_read`, `B_kv`, `eta`, `collapse_floor {expression: "eta/R", R, value}`, per-region `token_budget`, `ctx_min`, `ctx_max`, `b` and `ctx` as run |
| composed metric | `compose.recall@1` overall and per bin; `compose.null_recall` on the general bin; `compose.null_fpr` per other bin; per-pair `Δ_A`, `Δ_B`, `I` with block-bootstrap CIs at W6 |
| baselines | B0, B0d, B0u, B1, B2, B2t, B3, each with `source: graded` (TAX:1815-1823) |
| attention mass | `a` mean per active iteration and region, the per-region histogram, `collapsed_in_phase_A: []`, `a_store` |
| write-back and topology | `write_back.enabled`, the per-region own-bin delta, `topology {agreement, status}`, `edges` only when demonstrated |
| scheduler | `rho`, `flops_ratio`, `sparse_rel_delta`, `phase_d_vs_c`, the S1-S5 statistics (reported, never gated: TAX:2019-2023), `mean_iters`, the `halt_at` histogram, the budget-as-tag gap between the re-emitted and frozen-schedule arms |
| latency | `wall_ms_measured`; `first_token_ms_measured` and `speech_frame_miss_fraction`, both `null` at v1 (Table 8a) |
| store | `b_store`, `capacity_bytes` per card, `occupancy_bytes`, which of residual or floor is active |
| verdicts | `integration:`, `scheduling:`, `trigger_sensitivity:` as three separate strings (TAX:2025-2028) |
| placement and knobs | `placement{}` from W7p and `knobs{}` from W7k |

**Fail-closed guards.** Table 8 proposes G-numbers continuing the repo's guard registry, whose
highest today is G26 (`src/cogsyndelta/splits.py:38-39`). Each guard raises rather than warns, and
each has a constructed failing case in section 5. G30, G31 and G32 fire inside the module at run
time; the seven others are functions in `gates.py` over receipts, owned by lane IC-10 and called by
the compose-stage script.

Table 8 — guards, proposed G-numbers, what fires them and what firing does.

| G | guard, where | fires when | effect |
|---|---|---|---|
| G27 | W5b write-back gate, `gates.py` | a conditioned region's own-bin score drops > 1 point, or the composed metric does not improve | write-back disabled for that region; `edges` and `lockstep_groups` omitted; `topology: not demonstrated` (TAX:1435-1439) |
| G28 | topology agreement, `gates.py` | the two derivations agree on < 95% of items | the disagreement is reported as a bug; void, and declared void, under the no-write-back fallback (TAX:1469-1490) |
| G29 | attention-mass floor, `gates.py` | any region's `mean(a_r) < η/R` in phase A; `mean(a_store) < η/R` in E2; a receipt whose printed floor ≠ `η/R` at its own `R` | the region is named in `collapsed_in_phase_A` and excluded from B's targets; the store is reverted in E2; the receipt is refused (TAX:1659-1669, TAX:2661-2662). **AMENDED 2026-09-07 BY A6, INTERPRETATION ONLY:** the predicate, the threshold and every consequence in this row stand exactly as specified. What is corrected is what a run that does **not** fire may be read to mean — see Amendment A6 |
| G30 | `Schedule` validator, `schedule.py` | any B1 bound violated, including `region_token_flops > flops_ceiling`, a malformed `admitted` and a `depth ≠ min{i : admitted[i]}`; `trace_id` present; a store namespace named; a modality outside `allowed_modalities` or with no resident head | the `Schedule` is refused before execution and the dense fallback runs (TAX:5822-5830, TAX:1616-1626) |
| G31 | latent-tract assertion, `adapters.py` and `kv_bank.py` | an integer-typed or vocabulary-indexed payload on `h_r`, the adapted tokens, `z` or `cond_r` | raises at the boundary (TAX:2667) |
| G32 | store scope, `episodic_store.py` | a read or write with no server-derived scope, or a scope supplied by the request | refused; an unknown principal reads an empty partition (TAX:5854-5857) |
| G33 | frozen-set identity, `gates.py` | a participant's checkpoint sha ≠ the receipt it cites, or a parametric region declaring `kind: nonparametric_store` | phase A and E2 refuse to start (TAX:2598-2626) |
| G34 | phase-D revert, `gates.py` | D does not beat C at equal or lower FLOPs | C ships with `scheduler: imitative` (TAX:1753-1755) |
| G35 | overfit gate, `gates.py` | the train/held-out gap is ≥ 5 points | the phase-A receipt is marked FAIL (TAX:1757-1761) |
| G36 | `NULL` gate, `gates.py` | `NULL` recall on the general bin ≤ 0.50, or `NULL` false-positive rate ≥ 0.05 on any other bin | the phase-A receipt is marked FAIL per bin (TAX:1701-1702) |

**Amendment A6 (2026-09-07): what G29's floor establishes, and what it does not.** **No gate is
weakened here.** G29's predicate, its `η/R` threshold, its `η` and `R`, and every consequence
Table 8 lists for it are unchanged. What is corrected is a claim this spec and the taxonomy both
made *about the result*: that clearing the floor shows the store is being attended to, in the sense
of being used.

`mean(a_store) ≥ η/R` establishes exactly one thing — **attention mass was allocated to the store's
slots**, so the read simplex did not collapse away from it. It does **not** establish that the
store's read carried information about the target, and it does not establish that the composite
used what it read. **On a stream with no recall dependency the two quantities are decoupled, and
that was measured rather than argued.**

Measured on phase A's synthetic stream as it stood before Amendment A4 made the read
query-dependent: the store bank was a constant across items and batches, so its mutual information
with the target was exactly **zero**, `dL/d(store attention)` was ~0, and the mass was an
unidentified direction that random-walked. `mean(a_store)` read **0.0468 at 5, 8 and 64 primed
records — identical to four decimal places — and 0.0488 with an EMPTY store**. Emptying the store
at evaluation moved dev loss by ~1e-4 nats and dev `recall@1` **not at all**, while `mean(a_store)`
ranged over **70x** (0.0043 to 0.3029). G29 fired on **2 of 8** seeds, and at a fixed seed it
flipped on `OMP_NUM_THREADS` alone (1/2/4/16 PASS, 8 FAIL). It is also non-monotone in the step
count — 0.118 → 0.101 → 0.068 → **0.047 FAIL** → 0.201 → 0.329 — so training longer un-collapses
it.

**The gate is sound; the input was wrong.** Run unchanged on a stream where the store is the only
route to the label, the same predicate reads **0.191–0.640** over the whole trajectory and never
fires — a 3.8x–12.8x margin. That is a working gate starved of a valid input, which is why nothing
about it moves here.

**The consequence, and it is the whole amendment: a pass is NECESSARY, NOT SUFFICIENT.** A receipt
may be read as evidence that the store was admitted only when it also carries independent evidence
that the store's read is identified — non-zero mutual information between the read and the target,
or a measured effect on the composed metric from emptying the store. Without that, the honest
verdict on the run is **INCONCLUSIVE, not PASS**. Amendment A4's query-dependent read is the input
change this calls for (MI excess +1.58 to +1.86 bits, held-out accuracy 0.855–0.996 against a null
of 0.29), which is why the correction lands on the interpretation and not on the gate.
Evidence: `/akula-data/session-backup-staging/notes/GATE-DISCRIMINATION-2026-09-07.md` and
`DECISIONS-AND-RENAMES-2026-09-07.md` §1.

Table 8a — disposition of every §2.5 `Schedule` field this module does not compute from a head (TAX:1496-1517, TAX:1546-1557).

| field | disposition |
|---|---|
| `nodes[].priority` | keep: the rank of `β_b` in descending order, ties by participant order `[spec]`; recorded, and consumed by DEC-70's arbiter through W7k rather than by this module (TAX:1396) |
| `nodes[].precision` | keep as a closed vocabulary `{fp32, bf16, int8, int4}`, fixed to the checkpoint dtype at v1; any other value is refused (TAX:4148-4152) |
| `nodes[].resident` | keep, always `true` at v1, because `WeightStore.ensure_resident` is a no-op (TAX:4267-4268) |
| `nodes[].codec` | keep, always `"none"` at v1; the tract codec is a placeholder (TAX:742-751) |
| `step_budget.wall_ms` | keep and enforce: the runtime deadline on the whole turn; measured into `latency.wall_ms_measured` |
| `output.stream`, `first_token_ms`, `speech_frame_ms` | keep declared, as DEC-44 and DEC-48 require; `stream` is `false` at v1 and the two latency fields are recorded `null`, their enforcement deferred with the speech head (TAX:1546-1557) |
| `trace_id` | superseded: server-minted, never a model field (TAX:1521-1524) |

## 5. Test plan

Every guard test constructs the condition the guard exists to catch and asserts that it fires,
following `tests/test_guards_can_fail.py`; the interconnect's guard tests live in their own file.

Table 9 — unit tests per component, one file each under `tests/interconnect/`.

| component | tests |
|---|---|
| `schedule.py` | every Table 3 bound accepted at the boundary and refused one past it, including `region_token_flops`, a malformed `admitted` and a `depth` that disagrees with `admitted`; `trace_id` refused; modality refusal in both directions; JSON round-trip equality; the validator is pure and deterministic |
| `adapters.py` | output shapes for `d_r ∈ {256, 384}`; top-k returns exactly `b_r` positions that are both real and inside the budget mask, never a masked one, ties by position; the straight-through gate passes gradient to the scorer; prefix shape `[B, 8, d_r]`; a uniform query reproduces the mean pool to 1e-6 |
| `kv_bank.py` | the dense and 10,000 random allocations pack without overflow; `slot_region` and `key_mask` agree; an unadmitted region's slots are masked at iteration `i`; an empty store partition yields zero unmasked store slots; `bank_k == bank_v` off the store slots; `W_k` and `W_v` carry no bias |
| `workspace.py` | `z` keeps its shape across iterations; `a[b, i, :]` sums to 1 over admitted regions on active iterations and is zero otherwise; SDPA and explicit-softmax paths agree to 1e-4; the severance mask gives zero gradient through the masked slots |
| `readout.py` | `f` shape; two distinct `z` give distinct `f`; cosine scores inside `[−1, 1]`; the `NULL` candidate sits at index 0 and receives gradient; probe outputs at each `pooled_dim`; the `z_affect` gate is zero and adds no parameter |
| `controller.py` | both simplexes stay inside their bounds for 10,000 random inputs and for adversarial inputs built to starve a region; `box_integerise` sums to `B_read` exactly and respects every box: the draft-1 counter-example `β_b = [8, 8, 222, 11, 7] / 256` yields `[15, 15, 195, 17, 14]`, the cascading-cap case `β_b = [0.7766, 0.0313, 0.0317, 0.108, 0.0524]` yields `[96, 27, 28, 64, 41]` after two rounds of capping, and the dense allocation is `[64, 64, 64, 64]` and `[52, 51, 51, 51, 51]` with the tie-break landing on `language`; `Σ_i A[i, r] ≥ 1`; `halt_at ≤ max_iters` for any logits; an input exists whose `halt_at < n_iter` |
| `episodic_store.py` | scope isolation on the stub; a mis-derived scope key produces a crossing in the negative test; each `ContractGap` names its DEC; E0's nine ported fixtures run and fail red on the clauses the stub does not implement |
| `losses.py` | each loss is finite on random inputs; `L_B` is zero when `ŝ = a`; the FLOPs penalty is zero at target and positive above it |
| `gates.py`, `receipts.py` | each gate passes its positive control; `ComposeReceipt` writes every Table 7 group and a receipt missing any group fails a schema test |
| all | construction and forward under `torch.manual_seed(0)` are bitwise identical across two runs on CPU |

**Integration test.** `tests/interconnect/test_integration.py` builds two fake faculties that
satisfy the W0 protocol,
`FakeText(token_dim 16, pooled_dim 16, kv_bytes_per_token 64, accepts_condition True)` and
`FakeVisual(token_dim 24, ...)`, plus the store stub, at `D = 64`, `L = 8`, `n_iter = 2`,
`B_read = 16`, `k = 4`. It runs phase A for 20 steps on toy items with a planted rule and asserts
that the loss falls; `a` is a distribution; write-back changes `h_r` at iteration 1 and not at
iteration 0; the store's slots receive mass only after a write in the same scope; the emitted
`Schedule` validates, and switching `write_back` off omits `edges`. Five more: the bank is
load-bearing, `∂f/∂bank_v ≠ 0`; halting is reachable and no block at or beyond `halt_at` receives
gradient; an `A` with some `A[i, r] = 0` leaves that region unencoded and unread at `i`; two
distinct `z` give distinct `f`; and the frozen-schedule arm, `forward(inputs, schedule=s0)` with the
tokens content-swapped, re-emits `s0` byte-identically while `f` changes (TAX:1983).

**Guard-can-fail tests.** `tests/test_interconnect_guards_can_fail.py` holds one constructed failure
per row of Table 8: a receipt pair with a 1.5-point own-bin drop (G27); two derivations disagreeing
on 10% of items (G28); a phase-A run with one region's keys masked so its mass is zero, a receipt
printing 3.75% at `R = 5`, and the in-contract store falsifier of TAX:2662, `b_store` pinned at its
legal floor of 8 on episodes with no recall dependency, asserting `mean(a_store) < 3.0%` (G29); a
`Schedule` one token over `B_read`, one over the FLOP ceiling, one with an `admitted` row of all
zeros, one whose `depth` disagrees with `admitted`, one carrying `trace_id`, one naming `speech`
with no head (G30); a region wrapper returning `argmax` ids and a prefix built from ids (G31); a
request carrying its own scope (G32); a checkpoint with one flipped byte against its receipt, and a
parametric region claiming `nonparametric_store` (G33); a D receipt worse than C (G34); a 6-point
gap (G35); a general-bin `NULL` recall of 0.40 and an off-bin `NULL` false-positive rate of 0.10
(G36). Each test also asserts the positive control passes, so a guard that refuses everything is
caught.

**Amended 2026-09-07 (A6): G29's store falsifier stands as written, and it needs replication and a
verified pin to mean anything.** The construction above is unchanged — `b_store` at its legal floor
of 8, episodes with no recall dependency, asserting `mean(a_store) < 3.0%`. Two measured facts
about *running* it. **(1) One seed is not a demonstration.** Phase A's own synthetic stream is an
instance of that input, and on it the assertion held on only **2 of 8** seeds and flipped on
`OMP_NUM_THREADS` alone at a fixed seed, so this test must be run **seed-replicated and
thread-pinned**; a single-seed green is a draw from a statistic that random-walks across the
threshold. **(2) The pin has to be checked, not assumed.** In the phase-A toy the store was given a
2–8 read-token range against `η/R · B_read = 0.8`, so `b_store` was floored by the participant's
own minimum rather than by `η/R` — and whether `b_store` sat at its legal floor is precisely what
separates *"the store was not attended to"* from a starvation artefact. Neither point moves the
assertion or the threshold.

**CPU smoke.** `tests/interconnect/test_smoke.py::test_forward_backward_under_30s` runs one forward
and backward on the integration configuration at `B = 4`, asserts every trainable parameter received
a finite gradient and every faculty parameter received none, and asserts a wall-clock under 30 s
with a 2 s target; it runs in the CI job image, which has no GPU (TAX:5609-5622).

## 6. Open decisions for the operator

Draft 1's Q3, Q4 and Q6 are applied and deleted: the taxonomy answers them (section 3 step 13, the
affect seam, and section 1's `D_w = 512`). Each remaining item has options and a recommendation;
none blocks the lanes of section 7.

**Q1. Store scope axis (OD-21, DEC-64).** What is `scope` in `(scope, domain, logical_key)`?
Options: (a) the authenticated principal, with `session` as a server-bounded sub-segment; (b) the
session alone; (c) a persona basin within a principal. Recommendation: (a), the DEC-64 default; (b)
makes X7 work and long-horizon memory impossible, and (c) is a later product decision
(TAX:585, TAX:4966-4977).

**Q2. Store capacity claim and `safety_margin` (DEC-63).** Is a residual claim acceptable, given
that the residual may round to zero on the 5080? Options: (a) the residual formula with
`safety_margin = 2 GiB` and E1's pre-committed floor fallback on any card whose residual is zero;
(b) a fixed floor reserved before the KV budget on every card; (c) the residual with no fallback.
Recommendation: (a); it is what E1's gate (ii) already records per card
(TAX:584, TAX:4919-4927, TAX:6002-6015).

**Q5. OD-17 dependency.** W5 needs regions frozen at retrained checkpoints, W4 is blocked on OD-17,
and the current checkpoints carry near-copies of the mean per W1 (TAX:2397-2400, TAX:2651). Options:
(a) build and CPU-test the module now and run no phase-A GPU row until OD-17 is answered; (b) wait
entirely; (c) run a throwaway phase A on the current checkpoints for plumbing only. Recommendation:
(a), which section 7's lane plan already implements, since no lane runs a GPU row; (c) is allowed
only with `battery_id: plumbing` so nothing can cite it; under the pivot branch the module is
unchanged and only the frozen set of Table 6 becomes a config flag (TAX:5349-5365).

**Q7. Candidate embedding for the DEC-41 ranking head.** The taxonomy declares a ranking head over a
frozen candidate set and says nothing about how a candidate is embedded (TAX:1684-1697). Options,
with the cost of embedding one item's 31 content candidates `[I]`: (a) a `CandidateEncoder`,
`Linear(1152, 512)` over the concatenated frozen pooled outputs of the four encoding regions, then
`RankHead`; about 0.9 GMAC per candidate for the region forwards, computed once per candidate and
cached across every phase and baseline because the regions are frozen, so the per-step marginal cost
is near zero. (b) the whole mind on each candidate; about 32 × (1.56 + 0.9) ≈ 79 GMAC per item per
step, never cacheable because the workspace trains, roughly 32 times the query side and absent from
§4.3's `T_A` budget (TAX:1362-1375, TAX:2880-2884). (c) a cross-encoder that admits candidate tokens
to the bank; the same order as (b) and it changes `R`. Recommendation: (a). Table 4 carries its
590,336 parameters conditionally.

## 7. Lane plan

Lanes are file-disjoint, and shared test fixtures live in `tests/interconnect/conftest.py` and
`tests/interconnect/__init__.py`, both owned by IC-8.

Table 10 — lanes, files, dependencies and order.

| lane | files | depends on | order |
|---|---|---|---|
| IC-0 | this document, plus an adversarial pass on it: threat-modeler on sections 3-4, skeptic on Table 8 | — | first, before any code |
| IC-1, IC-3, IC-4, IC-6 | `schedule.py`, `workspace.py`, `readout.py`, `episodic_store.py`, one lane each, each with its `tests/interconnect/test_<file>.py` | IC-0 | wave 1 |
| IC-10 | `gates.py`, `receipts.py`, `tests/interconnect/test_gates.py`, `tests/interconnect/test_receipts.py`; the `STAGES` addition in `pipeline/receipt.py` | IC-0 | wave 1 |
| IC-11 | `contracts/region_spec.py`: replace the single-`stream_dim` check with `workspace_dim` plus per-region `token_dim`, and its test | IC-0 | wave 1 |
| IC-2 | `adapters.py`, `kv_bank.py`, their tests | W0 protocol, IC-0 | wave 2 |
| IC-5 | `controller.py`, `tests/interconnect/test_controller.py` | IC-1 | wave 2 |
| IC-7 | `losses.py`, `tests/interconnect/test_losses.py` | IC-4, IC-5 (`ŝ`, `A`, `ctx`) | wave 2 |
| IC-8 | `mind.py`, `__init__.py`, `tests/interconnect/__init__.py`, `conftest.py`, `test_integration.py`, `test_smoke.py` | IC-1..7, IC-10, W0 | wave 3 |
| IC-9 | `tests/test_interconnect_guards_can_fail.py` | IC-8, IC-10 | wave 3 |
| IC-R1 | skeptic pass on every guard's failing case in IC-9; a written disposition is the merge condition for IC-9 | IC-9 | before IC-9 merges |
| IC-R2 | threat-modeler pass on the built validator and the store path (B1, B2) | IC-8, IC-9 | before W5's first GPU run and before OD-17's answer is spent |

Each lane works in its own ephemeral worktree, commits with a scoped `git commit -- <files>`, runs
`bash scripts/lint.sh`, and pushes a sha for a PR to `main`; no lane runs a GPU row. Guards are
shown to fire before they merge, so IC-R1 precedes IC-9's merge; the module is not handed to W5
until IC-R2 also has a written disposition, the taxonomy's own standard (TAX:509-514).

**Sources.** `TAX` = `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` at `3bc8c3e`;
`src/cogsyndelta/faculty/protocol.py` (W0 lane); `src/cogsyndelta/regions/_receipt.py`,
`src/cogsyndelta/pipeline/receipt.py`, `src/cogsyndelta/eval/metrics.py`;
`src/cogsyndelta/splits.py` (guard registry). Operator notes: `csd-reasoning-faculty-definition`,
`csd-latent-space-reasoning-invariant`, `csd-affect-isolation-contract`,
`csd-episodic-store-required`, `csd-incremental-integration-protocol`.
