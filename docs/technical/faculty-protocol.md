# The Faculty protocol (row W0)

**Not one of the ratified thirteen `technical/` chapters.** Like
[`model-card-pipeline.md`](model-card-pipeline.md), this is a standalone reference for one
piece of code, not a design-derived chapter — it carries no `DEC-nn`/`OD-nn` content of its
own, only citations into `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md`.

## Summary

Row W0 of the taxonomy asks for a typed contract — `tokens()`/`pool()` — that every region
can be checked against, plus a re-instantiated parameter count so a design-doc number stops
being a projection. This lane adds three small modules under `src/cogsyndelta/faculty/`:

| module | what it adds |
|---|---|
| `protocol.py` | The `Faculty` protocol itself: `tokens()`, `pool()`, and the DEC-47 guard. |
| `adapters.py` | Wrappers that make `TextEncoder`/`IJEPA` satisfy `Faculty` without changing their weights. |
| `param_table.py` | The embedding / token-path / pooling-head split, measured against the five trained regions. |

The `tokens()`/`pool()` **split itself already existed** before this lane (commit
`e5e57f6`, 2026-09-02) on `TextEncoder` and `ViTEncoder`/`IJEPA`, with its own test suite
(`tests/test_token_surface.py`) proving pad invariance, an independent numpy reference,
row-permutation (both clauses), and batch-composition invariance. Nothing in that file is
redone here. What was missing — and what this lane builds — is the `Faculty` **protocol
surface** those methods do not yet expose (`token_dim`, `pooled_dim`, `kv_bytes_per_token`,
`accepts_condition`, and a `tokens()` call shaped `(inputs, *, context_tokens,
condition=None)`), and a per-region parameter table nobody had computed yet outside one
region done by hand in the design doc's prose.

## What "tokens" means here

`tokens()` never means vocabulary ids. Per DEC-47 (taxonomy §2.2), a region's "token
surface" is **position latents** — one floating-point vector per input position, the same
kind of thing `pool()` produces, just before pooling and before the output projection. The
only discrete ids anywhere in this system are at the input tokenizer and at the frontal
read-out. `cogsyndelta.faculty.protocol.assert_latent_tokens` checks this directly: it
raises if a `tokens()` call ever returns a non-floating-point tensor, and
`tests/test_faculty_protocol.py::test_assert_latent_tokens_fires_on_the_fake_faculty` proves
it fires, against a fake faculty built to violate the rule on purpose.

## The protocol

```python
@runtime_checkable
class Faculty(Protocol):
    name: str
    faculty: str
    token_dim: int
    pooled_dim: int
    kv_bytes_per_token: int
    accepts_condition: bool

    def tokens(self, inputs, *, context_tokens: int,
               condition: Tensor | None = None) -> tuple[Tensor, Tensor]: ...
    def pool(self, h: Tensor, mask: Tensor) -> Tensor: ...
```

This is the taxonomy's own §2.2 sketch, lifted into `src/cogsyndelta/faculty/protocol.py`
unchanged. Two widths are kept distinct on purpose: `token_dim` is what `tokens()` emits
(pre-projection — 256 for every text region trained so far, 384 for `visual`); `pooled_dim`
is what `pool()` emits (post-projection). They happen to be equal for every region trained
today, because none of them use a projection that changes width — see the parameter table
below for the measurement, not an assumption, that this is so.

**Deviation from an earlier, looser paraphrase of this row.** A prior restatement described
the surface as `tokens(batch) -> Tensor[B, T, D]` / `pool(batch) -> Tensor[B, D]` plus one
`latent_dim` field and a `token_mask` accessor. This module does not build that shape,
because the taxonomy's actual contract keeps two things the paraphrase collapses: `mask` is
already returned by `tokens()` — `cogsyndelta.faculty.protocol.token_mask()` just names that
element rather than adding a second call — and `token_dim`/`pooled_dim` are two numbers
because the pooling head can (and, in a future retrain, may) change width between them.

## Why adapters, given the encoders already have `tokens()`/`pool()`

A bare `TextEncoder` or `IJEPA` does **not** satisfy `Faculty` — it has no `faculty`,
`token_dim`, `pooled_dim`, or `kv_bytes_per_token` attribute, and its `tokens()` does not
accept `context_tokens`/`condition`. `tests/test_faculty_protocol.py` proves this both ways:
`isinstance(bare_encoder, Faculty)` is `False`, and wrapping the same encoder in
`TextFacultyAdapter`/`VisualFacultyAdapter` makes it `True`. The adapters:

- Add **no parameters** and change **no weights**. `pool()` through an adapter is
  bit-for-bit the encoder's own existing pooled output (tested,
  `test_text_adapter_pool_is_bit_exact_with_encoder_forward` and its visual counterpart).
- Enforce `context_tokens` as a **bound check** on the input actually given, not a
  truncation the region performs — there is no controller yet to decide `ctx_r` per
  iteration, so there is nothing to truncate against. An input that already exceeds the
  declared budget is refused.
- **Refuse a non-`None` `condition`.** The taxonomy's own §1.4 catalogue (a proposed diff
  headed *"REVIEW ONLY, DO NOT APPLY"* — `docs/technical/README.md` fact 1) declares
  `accepts_condition: true` for several regions, but no controller exists to produce a
  `condition` tensor (`docs/technical/README.md` fact 2: *"the interconnect does not
  exist"*). Every adapter here sets `accepts_condition = False` and raises loudly on a
  non-`None` value, rather than asserting a capability nothing implements.

`kv_bytes_per_token` (`c_r`, DEC-15) is computed as `2 × dim × depth × 2` bytes — two
vectors (K and V), one pair per transformer layer, at 2 bytes/element (fp16) — rather than
copied from the catalogue by hand. It reproduces the catalogue's own figures for both
region families measured so far:

| region family | dim | depth | `kv_bytes_per_token` |
|---|---|---|---|
| text (`language`/`compress`/`retrieve`/`reason`) | 256 | 4 | 4,096 |
| `visual` | 384 | 6 | 9,216 |

## The parameter table

The white-matter interconnect table (§2.3's `27,424,039`-param workspace/controller total)
is **not** re-instantiated by this lane — that needs the controller and workspace modules
(DEC-16), which do not exist yet and which W0's own mandate excludes building ("no
controller, no workspace, no episodic store"). See the taxonomy deviation note below.

What this lane builds instead is a smaller, buildable table: for each of the **five**
regions with a real trained checkpoint under `/akula-data/csd/matrix/`
(`code`/`compress`/`retrieve`/`reason`/`visual` — `memory` is not among them; no matrix
cell has trained the merged region yet), how many of its own parameters are **embedding**
(the token/patch table), **token path** (the transformer blocks that produce `tokens()`),
or **pooling head** (`proj`, used only inside `pool()`).

Measured 2026-09-06, CPU, read-only, via `python scripts/csd-faculty-param-table.py`
(equivalently `cogsyndelta.faculty.param_table.build_matrix_param_table()`):

| region | legacy name | kind | embedding | token path | pooling head | total | embedding share |
|---|---|---|---|---|---|---|---|
| `language` | `code` | text | 12,865,792 | 3,155,456 | 0 | 16,021,248 | 80.30% |
| `compress` | `compress` | text | 12,865,792 | 3,155,456 | 0 | 16,021,248 | 80.30% |
| `retrieve` | `retrieve` | text | 12,865,792 | 3,155,456 | 0 | 16,021,248 | 80.30% |
| `reason` | `reason` | text | 12,865,792 | 3,155,456 | 0 | 16,021,248 | 80.30% |
| `visual` | `visual` | visual | 74,112 | 10,638,336 | 0 | 10,712,448 | 0.69% |

Two things this table confirms rather than assumes:

1. **Every text region reproduces the taxonomy's own hand-derived calibration figure
   exactly**: *"a text region is 16,021,248 params of which 12,865,792 (80.30%) is the
   token embedding table"* (§2.3). All four text regions share one shape
   (`vocab_size=50257, dim=256, depth=4`), so all four land on the same three numbers.
2. **`visual`'s total (10,712,448) matches §1.4's catalogue figure** for *"DEPLOYED HALF IS
   THE EMA TARGET ENCODER"* exactly, counting only `target_encoder` (DEC-34) — not the
   context encoder or the predictor, which are training-only and double-count or add
   parameters no `Faculty` call ever reaches.
3. **Every region measured has a zero-parameter pooling head.** No production region uses
   an output projection that changes width (`out_dim=None` everywhere), so `token_dim ==
   pooled_dim` for all five today — the two-width design in `Faculty` is there for the
   general case, not because any region currently exercises it.

A checkpoint absent or unreadable on the host running this table is reported as a row
naming the reason, not a crash — `build_matrix_param_table` never raises past a single
bad cell.

## Where this lane could not honour the taxonomy's row verbatim

Row W0's task cell asks for two things: the `tokens()`/`pool()` split (done, pre-dating this
lane), and to *"re-instantiate §2.3's parameter table at the design's actual configuration
(4 controller heads, v1 participant list) so the total stops being `[I]`."* That second
half is the **white-matter interconnect** table — workspace blocks, frontal read-out,
thalamic controller, conditioning prefixes, region adapters, the `27,424,039`-param total.
Building it requires DEC-16's module (§2.3: workspace + controller), which
`docs/technical/README.md`'s own fact 2 records as design-only (*"the interconnect does not
exist"*), and this lane's mandate explicitly excludes building a controller or workspace.
**That half of the row is not done here**, and `27,424,039` / `86,331,303` are unchanged.
What this document calls "the parameter table" throughout is a different, smaller
thing — a per-region breakdown, not the interconnect total — and should not be read as
having closed W0's `[I]` tag on §2.3.

Separately, the taxonomy's §1.4 catalogue (the proposed `faculty`/`token_dim`/`pooled_dim`/
`kv_bytes_per_token`/`accepts_condition` fields this protocol borrows names from) is headed
*"REVIEW ONLY, DO NOT APPLY"* and is not the shape `config/mind/csd-regions.json` actually
uses today (`stream_dim`/`hidden_dim`/`latent_dim`, per `cogsyndelta.contracts.region_spec`).
This protocol's field names match the catalogue's *proposed* shape because that is what the
taxonomy's own §2.2 code sketch already uses; it does not mean that catalogue diff has been
applied to `config/`.
