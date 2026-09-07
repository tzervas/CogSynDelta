# Row E1 (2026-09-06): the episodic store is a correct container, on both cards

**Verdict: the container holds.** The store built in `src/cogsyndelta/interconnect/episodic/`
satisfies all four of row E1's gates. The dynamic capacity is two different positive numbers on
the two cards — **20,858,273,792 B (19,891 MiB) on the 3090 Ti** and **13,806,600,192 B
(13,167 MiB) on the 5080**, both on the **residual** branch — and the pre-committed **floor**
branch is shown firing on the 5080, and only on the 5080, at a context length that drives its
residual to zero. This row makes **no claim about usefulness**; that is row E2's job and the
separation is deliberate (DEC-50 step 2).

## The two capacity receipts

| card | host | VRAM_total | capacity (deployment arm) | branch | long-context arm |
|---|---|---|---|---|---|
| RTX 3090 Ti | akula-prime | 24,146,608,128 B (23,028 MiB) | 20,858,273,792 B (19,891 MiB) | residual | residual, 4,541,382,656 B |
| RTX 5080 | gpu5080 | 17,094,934,528 B (16,303 MiB) | 13,806,600,192 B (13,167 MiB) | residual | **floor**, 2,147,483,648 B |

Formula, from section 8 gap (a):

```
capacity_bytes(host, tick) = max(0, VRAM_total(host)
                                  - KV_reserved(context_len, regions_active)
                                  - activation_reserve
                                  - safety_margin)
```

Inputs held identical across the two cards: `context_len = 8192`,
`regions_active = (language, reason, retrieve, visual)`,
`kv_bytes_per_token_per_region = 2048` (K and V at `D_w = 512` in fp16),
`activation_reserve = 1 GiB` (`hypha`'s CUDA-scratch reserve, the only measured number either
source repo has for this term), `safety_margin = 2 GiB` (gap (a)'s recommended starting value),
`floor = 2 GiB`.

## Why the long-context arm is here

Section 9.11 found that the residual **may round to zero on the 5080**, which is the deployment
card, and gap (a) pre-committed the alternative: a fixed floor reserved for the store *before*
the KV budget is computed. The `long_context` arm evaluates both cards at a 2,000,000-token
context. The 5080's residual goes to zero and the floor fires; the 3090 Ti, given the identical
budgets, stays on the residual with 4.2 GiB. That contrast is the evidence that the branch is
selected by each card's own arithmetic rather than being a paragraph in a design document.

The floor is not `max(residual, floor)`. It takes its slice off `VRAM_total` first, and the
receipt records `kv_headroom_bytes` — what the KV cache is left with — so the context-for-recall
trade section 8 describes is visible rather than hidden behind an overcommitted card.

## Files

| file | what it is |
|---|---|
| `capacity-probe-3090ti.json` | live receipt, `scripts/run_store_capacity_probe.py --host akula-prime` |
| `capacity-probe-5080.json` | live receipt, same script on gpu5080 |

Each receipt carries four arms — `deployment`, `region_set_changed`, `long_context`,
`respected` — plus the checks that arm passed. The `respected` arm admits declared spans against
that card's measured capacity through the real store and records that the over-capacity write
**evicted** rather than allocated: `allocated_bytes_for_spans` is 32 bytes against a 13–20 GiB
bound, which is DEC-65's index-not-bytes model doing exactly what it is for.

## Reproducing

```bash
# on the card's own host
python scripts/run_store_capacity_probe.py --host <label> --out <receipt>.json
# compare a fresh receipt against the committed pair
python scripts/check_store_capacity_receipt.py --live <receipt>.json \
  --committed-self  docs/design/evidence/e1-episodic-store-2026-09-06/capacity-probe-5080.json \
  --committed-other docs/design/evidence/e1-episodic-store-2026-09-06/capacity-probe-3090ti.json
```

The 5080 half is re-measured on every PR by the `e1-store-capacity-probe` job in
`.github/workflows/gpu-5080.yml`, which carries the `[self-hosted, linux, x64, gpu, 5080,
host-gpu5080]` labels, no `if:`, no `continue-on-error` and a 30-minute timeout — so a run no
GPU runner claims ends red rather than passing by absence.

## The four gates and where each is checked

| gate | where | constructed to fire? |
|---|---|---|
| (i) E0's nine named fixtures green, no test file edited | `tests/interconnect/test_e1_store_conformance.py`; `git diff` shows `tests/interconnect/test_episodic_store.py` untouched | no — its subject is the diff, which no test can check |
| (ii) dynamic capacity: two different positive numbers, respected, region-set-sensitive | `tests/interconnect/test_e1_capacity.py` + the two receipts here | yes — `test_e1_gates_can_fail.py::test_gate_ii_*` |
| (iii) cross-request fuzz, 10,000 writes over 120 partitions | `tests/interconnect/test_e1_cross_request_fuzz.py` | yes — its own negative arm, plus `test_gate_iii_fires_*` |
| (iv) eviction order under a constructed overflow | `tests/interconnect/test_e1_eviction_order.py` | yes — `test_gate_iv_fires_*` |
