# Region Taxonomy and the White-Matter Interconnect

**Status:** design draft, **REVISION 3.6**, for operator ratification. Nothing here is applied.
No config, script, checkpoint or program file is modified by this document, and the JSON diff
in §1.4 is shown so it can be reviewed, not so it can be run.

**What revision 2 changes.** Revision 1 was attacked by two independent passes — a skeptic pass
(35 findings, six critical) and a threat-model pass (4 findings plus a named security-theatre
list). Every finding is resolved here: **accepted and applied in place, cited at the changed
spot as `[A-n fixed]` / `[T-n fixed]`, or refuted with a re-read `file:line` in §11.** Since
revision 1, **W1 has been measured on the real production checkpoints, and its pre-committed
rule fired against the design's central bet** (§4.0). The consequences of that result — a
mandatory token-aware retrain before any interconnect training, a re-derived row order, and a
re-derived phase-2 cost — are the largest single change in this revision. §10 is the
finding-by-finding disposition index; §11 holds the refutations.

**What revision 3 changes.** Revision 2 was attacked by a second skeptic pass (`attack-skeptic-2.md`,
findings N1–N10), which confirmed 31 of 35 skeptic and 6 of 7 threat findings closed and found ten
new defects, one critical. Revision 3 is a **targeted** pass: nothing is restructured, and every
change is cited in place as `[Nn fixed]`. The ten:
**N1** the aqua_rat recovery now burns the **union** of both possible draws, because the sampling
algorithm changed under it (`c42203c`) and W2a's three checks passed under either hypothesis —
**DEC-42**, with a fourth, failable precondition (§5.1, §4.1 W2a).
**N2** every one of the six ablation pairs now has a declared item shape and a printed sealed-item
count (§5.4).
**N3** §4.0 prints **both** effective-rank definitions, they disagree, and W1's verdict was therefore
**provisional pending W1d**, whose deciding rule depends on no rank definition (§4.0, DEC-35) —
**W1d has since run and CONFIRMED it** (revision 3.2).
**N4** the fifth, non-production W1 measurement is in the table (§4.0).
**N5** the composite renderer becomes **W3r**, a prerequisite of both W7v and W3, breaking a
dependency inversion (§4.1).
**N6** a `corpus: REDUNDANT` branch for the G0d void condition (§2.7.3).
**N7** B0, B0d, B0u, B2t and B3 are budgeted in GPU-hours, and B2t is counted as what it is — a
second white-matter run (§4.3, §2.7.7).
**N8** the **binding** B1 max-share is printed beside B1 in both accountings (§5.1).
**N9** `memory × reasoning` and the general bin get declared, W7v-independent constructions (§5.5).
**N10** DEC-40 gets a row (**W0c**) and a named owner, and the W1 evidence is copied into the repo
at `docs/design/evidence/w1-token-rank-2026-09-02/` and cited from there (§4.0, §4.1, §10.2).

**What revision 3.1 changes.** A targeted pass over five sections, driven by one operator input
and one completed audit. Nothing is restructured and no earlier verdict is reopened. The
operator's stated model intent made the **`auditory` faculty and a speech-output head
REQUIRED**, not placeholders `[OP: csd-multimodal-io-intent.md]` — **superseded as a schedule by
DEC-48 in revision 3.2, which defers both to a production phase while keeping the seam declared**: **DEC-43** and **DEC-44** record
that, with the interface, the objective family and what it changes in the interconnect's runtime
contract (§1.3, §1.4, §2.5). The audio licence audit **OD-9 asked for is done**, committed at
`docs/design/AUDIO-CORPUS-AUDIT.md`, and it yields four new programme rows with gates
(**A0–A3**, §4.1 — **deferred and re-scoped to A0m/A0f/A1/A2/A3 in revision 3.2**), an audio
line in the phase-2 bill (§4.3 — **withdrawn in revision 3.2**), the per-region audio licence tiers
under the strictest-input rule (**DEC-45**, §5.7), the LibriVox-as-one-provenance-group correction
(**DEC-46**, §5.7), and six new operator asks (§8, **OD-10 to OD-15**). **OD-9 itself is
ANSWERED** and leaves the list. One property of this revision is worth stating because it is the
reverse of the last one: revision 3 closed findings, revision 3.1 opens requirements — the
programme got larger, not safer, and the bill in §4.3 says by how much.

**What revision 3.2 changes.** Three inputs, no restructure, every change cited in place.
**(1) W1d HAS RUN** — 2026-09-02, read-only on akula-prime, 116.5 s, against the same production
checkpoints. **W1's verdict is CONFIRMED and stops being provisional:** clean for `code`,
`compress` and `vl_latent`, **CONFIRMED-BUT-NOISY** for `retrieve`; **no region is OVERTURNED**;
the token-aware retrains (W4, then W7a, W7v) are **LICENSED**; and the **penultimate-block
fallback is dead** — `L_token` attaches at the FINAL block (§4.0, DEC-35, §9.1). **(2) An
operator ruling defers audio to a production phase** — *"we can wait to add audio as a future
feature once it proves out without audio"* `[OP: csd-multimodal-io-intent.md]` — so `auditory`
and `speech_output` become **DECLARED SEAMS** with their interface, objective family and runtime
output block **kept as declared**, rows A0–A3 move to §4.1's *Production phase: audio* subsection
behind a blocking gate, and **v1 stays at FOUR participants and SIX ablation pairs** (**superseded in revision 3.3 by DEC-49, which admits `episodic_store` and makes it five and ten**)
(**DEC-48**, amending DEC-43 and DEC-44). **(3) The latent-space reasoning invariant is recorded
as DEC-47** and given a gate in W9: no inter-region path may carry discrete token ids, and the
constructed violation must be refused. Revision 3.1 was attacked by a third skeptic pass
(`attack-skeptic-31.md`, S1–S21); its disposition is §10.4 and its fixes are cited as
`[S31-n fixed]`. **S1 and S2 are resolved BY the deferral rather than by a recount**, and §11 R4
says so explicitly rather than restating either finding for `R = 5`.

**What revision 3.3 changes.** Three operator rulings and one contract extraction. No restructure;
every change is cited in place with `[OP: file-name]` for the ruling and a `file:line` for the
contract. **(1) DEC-32 IS SUPERSEDED.** The operator ruled that `episodic_store` must be built and
backed rather than dropped — *"So that will have to have the episodic store created to back it
rather than just dropping it. It should have partition, capacity, integration contract"*
`[OP: csd-episodic-store-required.md]`. **DEC-49** reinstates DEC-10's shape: the store is a
**REQUIRED** region, **BUILT in phase 2**, a workspace participant with its own budget `b_store`.
Its contract is taken from the operator's `memory-gate` / `memory-gate-rs` repositories where those
repositories specify it — partition, importance-scored eviction, lifecycle, durability oracles,
retrieval-into-context — each clause carrying the extraction's `file:line` (§1.3). Where the repos
are **silent**, nothing is invented: the six gaps are listed for operator deliberation in §8's
**Episodic store contract gaps** block, each with what the repos say, what CSD needs and a
recommended default. The threat pass's findings against the store (§9.9 B2) become the **acceptance
gates of the build**, not reasons to defer it. **Consequence: v1 has FIVE participants and TEN
ablation pairs**, and every count that depends on that is restated in the same breath — §2.3's
parameter total, §2.4/§2.6's collapse floor (`η/R` = **3.0%** at `R = 5`), §2.7.3's G3′ criterion
(**≥ 7 of 10**, null rate **0.1719**), §2.7.7's cost model, §5.4's item shapes and sealed counts,
and §5.4's reserve size. **The store is admitted by S1's own fix (b)** — as a participant whose
pairs are counted — which required declaring the item shapes S1 said a bare fifth participant would
lack: **X7** and **X8**, recall-dependent episodes (§5.4). **(2) The INCREMENTAL INTEGRATION
PROTOCOL becomes a named procedure, DEC-50** `[OP: csd-incremental-integration-protocol.md]`: train
the submodel alone, retrain the interconnect with it admitted, then whole-mind unified training —
and **rows E0/E1/E2 are its first instance** (§4.1). DEC-26's 2–5% interconnect cap is what makes
step 2 cheap. **(3) The downstream bar is named, DEC-51** `[OP: csd-mycelium-downstream-goal.md]`:
after W10, a **Mycelium readiness assessment** (row **M0**) with a pre-registered pass-rate margin,
and the explicit open question — measured, not assumed — of whether CSD handles RAG natively as a
skill. **What this revision does NOT do:** it does not reopen W1/W1d, DEC-48's audio deferral, or
any licence verdict. It makes the programme **larger**, like revision 3.1 and unlike revision 3:
§4.3's phase-2 bill rises from ≈ 6–11 to **≈ 7–13 GPU-hours**, and §5.4's reservation total from
56,320 to **67,584** items, which tightens §9.2's already-open funding risk rather than closing it.

**What revision 3.4 changes.** Nine operator inputs and one read-only security survey. No
restructure, no renumbering, no earlier verdict reopened; every change is cited in place with
`[OP: file-name]` and every new decision is appended in numeric order as **DEC-52 to DEC-67**.
Unlike revision 3.3, most of what follows was **already half-present**: the ternary track (§6.9),
the toy swarm (**P5′s**), memory-gate overlays and tiered residency (**DEC-33**, **P5′o**), the
official-docs corpus (**P6′**), quantise-before-whole-mind (**P5′q**), the episodic store
(**DEC-49**, **E0/E1/E2**), the incremental-integration protocol (**DEC-50**), the 30B direction
(**OD-6**) and the audio deferral (**DEC-48**) are unchanged and are **not restated**. What is
folded is what was missing or contradicted:

**(a) THE I/O CONTRACT IS NAMED, AND THE LATENT INVARIANT GETS A DETECTOR.** **DEC-52** states the
contract the operator restated as *"any in, any out"* — every modality encoded to latents, unified
there, output modality decided at the read-out and **user-selectable and binding when the user
names one** — with a **must-have baseline** that the whole programme is measured against:
**discrete-token + visual input, discrete-token output** `[OP: csd-training-placement-policy.md,
csd-multimodal-io-intent.md]`. **DEC-53** turns DEC-47 from an assertion into a **detector**: W9's
existing clause (iii) refuses a *constructed* violation, which proves the assertion works and
proves nothing about the paths nobody constructed. Row **W9i** adds a **token-round-trip probe**
that instruments every tract and reports a positive detection, so an accidental re-serialisation
is *found* rather than *not-refused* `[OP: csd-latent-space-reasoning-invariant.md]`.

**(b) TRAINING PLACEMENT BECOMES A POLICY WITH A HARNESS.** **DEC-54**: submodels fit every card
at current and near scale, so concurrent submodel runs are **packed by per-job VRAM budget** across
the 3090 Ti, 5080 and 1080 Ti under **process-level pseudo-isolation** (per-process memory
fractions, `expandable_segments`, warp-level sharing — no MIG on consumer cards), accepted because
this is single-tenant. Launch tooling takes **a VRAM budget and a target host**; every receipt
records **host, budget and concurrency**. It carries the one measured constraint the baseline
receipts produced: **`reason` runs ALONE at batch 512 / `max_len` 256 with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — the batch-1280 attempt OOMed 640 MiB short
of the 22 GiB card `[OP: csd-baseline-receipts-2026-09-03.md]`. Row **W7p**. **DEC-55** records
**layer-sectioned training as a CANDIDATE, not a decision**, sitting beside **DEC-29**
(region-granular pipeline parallel) and the *"cross-host DDP is the wrong tool at 1 Gb/s"* ruling;
row **P5′L**, deferred, with the operator's own caveat that they may be wrong about some of those
techniques preserved rather than smoothed away `[OP: csd-training-placement-policy.md]`.

**(c) THE 1B-PER-SUBMODEL SCALE PATH IS SEQUENCED.** **DEC-56**, row **P5′b** (post phase 3):
toy → mid-size proven → **~1B per submodel** → quantise each → train the composed model on the
quantised regions → the 30B direction. **Data is the long pole and the number is stated:** of the
order **10^10 tokens per region** under the usual scaling rules, every one licence-verified.
**The 3.27 bits/param figure is RE-MEASURED at size, never assumed**, and the path proceeds
**region by region under DEC-50**, not as one jump `[OP: csd-billion-per-submodel-scale-path.md]`.

**(d) THE DATASET FACTORY, THE SYNTHETIC-DATA CONTRACT AND THE MORAL CORPUS.** **DEC-57**, row
**P2′f**: the sourcing loop — **search → identify → capture provenance → licence verdict → ingest
→ emit a compliant open dataset with full provenance** — as one structured I/O loop an agent can
drive, **generalising what already exists** (`csd-corpus-expand` catalogue entries with
`provenance_group`, upstream licence verbatim separate from the mirror tag, `REFUSE` verdicts
enforced by the fetcher, a receipt per fetch) rather than rewriting it. **DEC-58**, row **P2′s**:
synthetic data gets its **own contract** — the generator **named** in provenance, contamination
channels run against **every** eval, a **quality gate that can fail**, and a **capped share of any
bin** — because the operator's preference is *extremely high quality and requirement-complete over
large*. **DEC-59**, row **P5′m**: a **moral corpus** with its own CORPUS-CONTRACT entry,
provenance, licence verdict and **held-out moral/safety probes**, so the claim *"training on it
changes measured behaviour"* is a measurement and not a hope; **phase-3/scale-up, explicitly not a
toy row**. **DEC-31's strictest-input rule applies to every corpus the factory emits**
`[OP: csd-dataset-factory-and-moral-corpus.md]`.

**(e) AUTODEV LEAVES THIS DOCUMENT, AND OD-1 AND OD-2 ARE ANSWERED.** **DEC-60**: the operator
ruled that autodev is a **harness that runs several models** and lives in **its own repo** —
`tzervas/csd-autodev` — so *"anything about the harness ... is committed in csd-autodev on its own
branch"* and **CogSynDelta keeps a pointer, not a specification** `[OP:
autodev-work-lives-in-csd-autodev.md]`. The spec is `docs/AUTODEV-IDENTITY-AND-SANDBOX.md` on
branch `docs/autodev-spec` in that repo, with its threat and skeptic passes beside it. **OD-1 and
OD-2 are ANSWERED and leave the open list** `[OP: autodev-service-identity-and-sandbox.md]`:
`svc-autodev`, its own vault, a minimum-scope Forgejo user, no root and no sudo, and the
lab-console as the **sole egress gateway** minting short-lived per-request tokens. **What stays
here is the CSD side, and it is worse than OD-1 and OD-2 described.** Three prerequisites become
**gated rows — PRE-1, PRE-2, PRE-3 — that BLOCK any autodev GPU route**: **(1)** the gateway
authenticates on `/api/apply` and **nowhere else** — `dispatch_api_get()` authenticates **nobody**
and `proxy_upstream()` attaches `UPSTREAM_TOKEN` regardless, so it is a confused deputy with no
identity to confuse (**PRE-1**); **(2)** `POST :9108/v1/queue` is authorised by **source-IP prefix
only** and the worker runs **as `kang`** with agent tokens in its environment, taking queued
`extra` fields into argv (**PRE-2**) — the single most exploitable live path, and it is not part of
the new design; **(3)** `kb_http._auth_ok()` **fails open on loopback when the token file is
absent** (**PRE-3**). **A new ask, OD-16**, records the wart the operator asked be raised rather
than fixed silently: **two diverged copies of `csd-lab-console` exist** — CogSynDelta's, which is
what the live unit runs, and csd-autodev's — and consolidating them into csd-autodev is the
operator's call.

**(f) GOVERNANCE, AND A CORRECTION TO OD-1's OWN RECOMMENDATION.** **DEC-61**: `main` is
**PR-only** in every code repo, on principle and for the operator's own commits too; CogSynDelta's
`main` now enforces it (push whitelist = the operator alone, **8 required `pull_request`
contexts**). **And OD-1's recommendation of a path-scoped Forgejo protection is CONTRADICTED by
measurement and is corrected in place rather than left standing:** `protected_file_patterns` had
to be **cleared**, because Forgejo refuses to merge **any** PR touching a protected file — 405
*"Changed protected files"*, **admins included** — which blocks precisely the reviewed path the
control was meant to protect. **Guard integrity therefore rests on the required checks —
including `tests/test_guards_can_fail.py` — plus review**, and saying otherwise would be counting
a control that does not exist. Every mutating agent works in an **ephemeral worktree** and pushes
a **sha**; merges are fast-forward or `--no-ff` from pushed shas `[OP:
branch-and-pr-to-main-only.md, agent-worktree-isolation]`.

**(g) W2c HAS RUN, AND ONE OF ITS TWO ANSWERS BREAKS AN ADMISSION CONDITION.** **DEC-62**. The
≈0.40 code lexical floor is **retired** — measured 0.2285 (seed 0) and 0.2344 (region seed),
recorded at `docs/design/evidence/w2c-untrained-baselines-2026-09-03/`, and **that retirement is
cited, not restated here**. What is new is the consequence nobody had a rule for: **`retrieve`'s
`τ_lo` is 0.0000 in chance-normalised units**, because its untrained r@1 equals chance exactly
(1/512). **NSRS admission condition (1) — `max_r s_r < τ_lo` — is therefore barely satisfiable on
that bin**: at `τ_lo = 0`, essentially any measured score above chance clears it, so the condition
admits on that bin by construction rather than by evidence. **W2b's row must DEFINE admission for a
chance-floor bin BEFORE it runs** — an absolute-margin floor, or a different normaliser — and the
choice is pre-registered with the reason `[OP: csd-baseline-receipts-2026-09-03.md]`.

**(h) THE EPISODIC-STORE GAPS STOP BEING OPEN.** The operator wants the contracts **nailed down**,
not deliberated indefinitely, so §8's six gaps are converted from *recommended defaults* into
**decisions with an implementable default and an explicit "operator to confirm" flag where the
choice is a product decision this document should not make**: **DEC-63** dynamic capacity (the
subtractive formula, recomputed per host per scheduler tick and **never cached**, plus exponential
half-life staleness decay — `safety_margin` and the residual-versus-floor question are the
operator's); **DEC-64** the **partition axis** — `(scope, domain, logical_key)` with `scope` = the
**authenticated principal derived server-side**, `session` as a bounded sub-segment, and
persona/basin as a **scope selector, never a routing object** (**operator to confirm**: this is a
decision about what the mind remembers about whom); **DEC-65** the payload — **index-not-bytes**,
latents owned by the runtime, a text sidecar as **provenance metadata only and never on the read
path**, which is the one clause where CSD deliberately diverges from the source repos; **DEC-66**
consolidation — **prune-only at v1, in those words**, CLS a phase-3 candidate under P5′'s monotone
rule, and **overlays are NOT pulled forward into E1** `[OP: csd-episodic-store-required.md,
csd-memory-gate-overlays.md]`.

**(i) THE DEPLOYMENT BAR IS A ROW.** **DEC-67**, row **M0d**: the **5080 + 3090 Ti must run the
composed CSD**, and the **1080 Ti is the RAG helper unless CSD serves retrieval natively as a
skill**. M0 already carried that shape as *task text*; a shape in a task description is not an
acceptance criterion, so it becomes one — with a gate that can fail on the card that is actually
tight `[OP: csd-mycelium-downstream-goal.md, fleet-gpu-roles-and-scheduling]`.

**What this revision does NOT do.** It does not reopen W1/W1d, DEC-48's audio deferral, DEC-49's
participant count, or any licence verdict. **It deletes one sentence and no more, and the
deletion is named rather than claimed away:** W2c's gate cell carried *"if the true code-adjacent
lexical floor is 0.40, condition (1) at `τ_lo = 0.35` rejects essentially every row built from
`apps`/`code_contests`"* — whose antecedent W2c measured **false** (0.2285 / 0.2344), so it is a
conditional whose condition is settled; the sentence itself survives verbatim in §5.3, where the
argument that produced it lives. Everything else is additive. DEC-61's correction to OD-1 and
DEC-62's correction to W2c's open question are both written as **amendments beside the original
text**, so a reader can still see what was recommended and why it changed. Like revisions 3.1 and
3.3 and unlike revision 3, it makes the programme **larger**: **eleven new rows**, three of which
(**PRE-1/2/3**) are prerequisites of an adjacent programme rather than of this one, and none of
which is on the phase-2 critical path except **W7p**, which is tooling every remaining GPU row
already needed. **`program/REMAINING.md` gains fifteen row lines rather than eleven**, because
revision 3.3's **E0/E1/E2** and **M0** were never mirrored into that file and are **back-mirrored**
in this pass — four rows this document already had, which is that file having been stale and not
this revision adding fifteen things.

**What revision 3.5 changes.** One measured outcome, seven operator inputs and one read of the
packer's own design document. No restructure, no renumbering, no earlier verdict reopened; the new
decisions are appended in numeric order as **DEC-68 to DEC-77**. **One thing this revision
deliberately does NOT do: it does not edit W4's five gate clauses.** The row has now been run at
its pre-registered configuration, and a document that moves a bar after seeing the number it
produced has stopped being a pre-registration. The change that is *recommended* is written as a
pre-registered amendment and flagged **operator to confirm** (**OD-17**), and the pivot stays the
fallback exactly as §9.14 wrote it.

**(a) W4 HAS RUN AT ITS PRE-REGISTERED CONFIGURATION, AND THE RESULT IS THREE PASSES AS SCORED —
ONE OF THEM ONLY BY FLOAT32 REPRESENTATION — AND TWO FAILS.** **DEC-68** records what was measured,
from the receipts and nothing else (§4.0). The production run at batch 512 failed **four of five**
gates; the batch-1280 run the plan actually pre-registered — reachable only after a chunked,
checkpointed cross-entropy over masked positions made it fit — **passes (a) and (d)**, **passes (b)
as the harness scored it** (`recall@10` **0.200** against a **strict** `> 0.20` floor, on a receipt
value of `0.20000000298023224` = `float(numpy.float32(0.2))`, so the margin is float32
representation and a strict reading makes it an exact tie — **OD-17 answers `>` versus `≥`, this
summary does not**), and **fails (c)** (full-pool `recall@10` **0.200** against BM25's **0.440**)
**and (e)** (final-block rank ratio **1.21**, against an absolute **2.0×**). The
in-batch-negatives count is the dominant lever and the curve is printed: at batch **256 / 512 /
1280**, full-pool `recall@10` is **0.030 / 0.098 / 0.200** and held-out `recall@1` is **0.600 /
0.815 / 0.854**. The token-aware objective's own effect is now measured at production length rather
than at 50 steps: at batch 512 the control arm reaches rank ratio **1.09** against **1.35** with the
terms on. **OD-17 puts the pivot-or-amend decision to the operator with both branches costed.**

**(b) THE KNOBS ARE ONE TABLE WITH MEASURED EFFECTS, AND THE PACKER'S ADMISSION MODEL IS THE
RUNTIME PACKER'S STARTING POINT.** **DEC-69** (§6.7): per-job VRAM budget and admission margin,
`token_loss_chunk` (**2048 → 20,726 MiB reserved / 22,120 MiB raw driver peak; 512 → 19,444 /
20,883 at +2.6% step time**), batch as the in-batch-negatives count (the curve above), mask
probability, and card choice by VRAM versus compute. **The operator's desktop on the 3090 Ti —
981–1,057 MiB of Xorg/KDE/Firefox — is a standing FOREIGN claim every budget on that card must
subtract**, and it is recorded as such rather than absorbed into a margin. Row **W7k**.

**(c) THE MEMORY GATE IS A VRAM ARBITER, AND DEC-63's CAPACITY IS A CEILING RATHER THAN A TARGET.**
**DEC-70** (§6.6): named claimants in priority order — base weights and working activations
**fixed**; then the **ACTIVE memory set and the loaded persona, protected**; then the **KV
cache/context window, which flexes** (sliding window) around them; **inactive overlays and
differentials leave first**. A **frugality cap** bounds resident memory + persona to a small
fraction of the card, with the gate that at the cap the achievable context length differs from the
no-memory case by less than a stated margin. Failure to size **reports and degrades the memory set
by relevance** and **never silently drops the persona**. Residency is scored on recency, persona
relevance and access frequency across card / host RAM / disk, eviction is proactive on inactivity
and pressure, and frugality is **measured in receipts**. Row **E1a**; **DEC-63 is amended in
place**.

**(d) TWO LONG-TERM TRACKS GET ROWS AND GATES, AND NEITHER IS PHASE 2.** **DEC-71** (§6.10):
**differential activations** — activation deltas against base in a format built for fast
recompute-and-apply, so an overlay contributes features without holding dense state; **gate: apply
latency and VRAM per feature against the dense alternative**, adopted only if it wins on both. Row
**P5′d**. **DEC-72** (§6.10): the **predictive hybrid training track** — extend `tritter`'s existing
weight-update predictor to **activations, activation deltas and semantic residuals**, with gates on
prediction error against real steps, the fraction of steps predicted, the end-task metric at
**equal wall-clock**, and an **automatic fallback** to real steps on drift. Row **P5′p**.

**(e) THE DATASET FACTORY'S PASS 1 IS MEASURED, AND IT AMENDS DEC-56.** **DEC-73** (§5.8): 159
candidates in 106 provenance groups, **105 VERIFIED / 26 CONTRADICTED / 18 UNVERIFIABLE / 10
REFUSED-CLOSED**, with **37 (23%) UNVERIFIED** as the real backlog. Against the flat
**1e10-tokens-per-region** target the B1-bounded reach is **`visual` 0.3%**, **`memory` 1.3%
clean-permissive / 5.0% NC-inclusive**, **`language_code` 15%**, **`reasoning` 1.8% without a
generator**, and **`language_trunk` 60× over**. **NC buys almost nothing** over clean-permissive.
**DEC-56's flat target is therefore amended**: per-faculty targets, clean-terms generators and the
operator's own enrichment replace one number applied to seven faculties (row **P2′g**). The five
legal readings become **OD-18**, and **the first of them needs no lawyer** — the model card must
state the reading it takes, and today it states none.

**(f) THE FLEET IS ON DEMAND, THE 1080 Ti's TORCH VERDICT IS MEASURED, AND THE PACKER LIVES IN ITS
OWN REPO.** **DEC-74** (§6.7): **all three cards and all their runtimes are started on demand and
stopped when idle** — an idle `:8080`, a stopped unit or an exited container is the **normal resting
state, not a fault**. The 1080 Ti's verdict is measured, not assumed: the fleet's pinned
**torch 2.11.0+cu128 cannot run on it at all** (`get_arch_list()` floors at sm_75; the VM's driver
535.274 caps at CUDA 12.2), while a separate **`torch==2.5.1+cu121`** venv on that guest runs
clean — so **fp32 training there is "with env X: yes"** and its **default role is inference, RAG and
utility**. **DEC-75**: **tooling lives in its own repo** — `tzervas/gpu-pack` is row **W7p**'s
implementation (**main at 5364932**, rounds 1–4 merged) and CogSynDelta keeps only the thin adapter
it already carries.

**(g) THE AUTODEV POINTER MOVES TO REV 5.** **DEC-76** (§8): the spec is
**`docs/AUTODEV-IDENTITY-AND-SANDBOX.md` rev 5 at `ece346b`** on branch `docs/autodev-spec` in
`tzervas/csd-autodev` (**PR #1** there), with the threat pass and the two retained skeptic
write-ups beside it — `01-threat.md`, `skeptic-round4.md`, `skeptic-final.md`, which is the whole of
`docs/evidence/autodev-threat-2026-09-03/`; five rounds were run and three write-ups were never
committed. **The residual
criticals, in one line:** the **final (round-5) skeptic, `skeptic-final.md`, still lists five
critical and six high** findings (the round-4 pass, against rev 4, listed **four** critical and six
high, and is a different document), of which two are overtaken by events (the auth gate is on
`main`; the closure pin went stale) and the rest are rev-6 material — the import recorder must run the protected tests under `pytest` rather
than import their modules and needs a `⊇` floor so it can fail, P0a's check must model the live
code rather than a per-request credentials fork, `socket_ident` must be re-stat-ed per connect, and
§0d must be pinned to a merge-base. **And the rule that follows from five rounds of chasing a moving
tree: freeze a code revision (a tag), then write rev 6 against it, then implement.**

**(h) THREE GOVERNANCE ADDITIONS, ALL MEASURED THIS SESSION.** **DEC-77** (§8): **DEC-61's PR-only
rule extends to `tzervas/gpu-pack`** — its `main` is protected the same way (push whitelist = the
operator, required context `CI / test (pull_request)`), so the rule is not CogSynDelta-specific.
**The CI secret scan has a prose false-positive CLASS, not a one-off:** `gitleaks`'
`generic-api-key` rule fires on low-entropy `identifier=value` pairs **inside prose** — a
tokenizer-load log line, and `decorr_weight=0.0` quoted inside a docstring reporting a control
arm's weights — and the fix is a **narrowly scoped allowlist (file path AND byte-identical phrase,
`targetRules` limited to the one rule), verified by planting a real-shaped secret in the same file
and asserting it is still caught**. **The CI runner has no GPU tooling:** `fleet-ci-base:1` has no
`nvidia-smi`, so a helper-CLI wrapper that lets `subprocess.run` raise takes a whole endpoint down;
**tests must not assume `nvidia-smi` exists, and a diagnostic subprocess must degrade one field
rather than abort its caller**.

**What this revision does NOT do.** It does not reopen W1/W1d, DEC-48's audio deferral, DEC-49's
participant count, or any licence verdict — and, as above, **it does not touch W4's gates**. It
makes the programme **larger**: **six new rows** (**W4n**, **W7k**, **E1a**, **P5′d**, **P5′p**,
**P2′g**), of which only **W7k** and **E1a** are phase-2 work — **E1a carrying its own minimal
persona/overlay stub, because two of its four gates need machinery that P5′o, `deferred` behind W10,
would otherwise have to supply** — and only **W4n** sits on the critical
path — and W4n is not licensed until OD-17 is answered. **It removes twelve lines across both files
and every one of them is a line modified in place rather than content dropped**: the revision
number itself; **six status or currency cells a completed run or a merged repository
contradicts** — W4's `todo` in both files (the run has happened), W7p's `todo — no GPU dependency`
in both files and `P10.4`'s *"gpu-pack's probe/admit/launch pipeline is the remaining piece"* (it is
merged), and `REMAINING.md`'s *"revision 3.4 is the current text"*; **four amendments that keep
their original text verbatim and append to it** — DEC-56, DEC-63, the index-order note and the
autodev pointer row; and **one line split to admit the new block**. Each is named at the spot it
happens and in the commit message. Everything else is additive.

**What revision 3.6 changes.** One operator ruling, no restructure, every change cited in place.
**FACULTY NAMING RULE** (**DEC-78**, §1): *"regions are named by cognitive faculty and the function
they provide, in accurate synthetic terms — never by knowledge domain (`code`) and never by
colloquial anatomy"* `[OP, 2026-09-04]`. Applied to the two renames this document already carries:
**DEC-01 is AMENDED** — the region formerly called `code` is the LANGUAGE CENTRE, canonical id
**`language`**, with `specialisation: "code"` recorded as metadata rather than folded into the id;
`code` stays the correct word for the corpus and the battery, never for the region. This
**settles §10.4/§11 finding S2's `code` → `language` proposal**, recorded there as **dropped**
against a different, falsifier-based objection (the finding argued the *falsifier* did not test
the claim, not that the *name* was wrong) — the operator's rule supersedes that disposition by
fiat rather than by re-arguing it, and both entries are left standing with this cross-reference
rather than rewritten. **DEC-03's `vl_latent` → `visual`** already matches the rule as written and
needs no amendment. **In-place edits, because these are live structural statements the rename
makes stale rather than dated measurement citations:** §1.2's taxonomy row 1 target name; §1.4's
JSON diff (`language_code` → `language` at the `probes`, `name` and `parent` fields, and its own
placeholder LANGUAGE TRUNK entry — a distinct, not-yet-built concept, "created when a SECOND
specialisation exists" — renamed `language_trunk` so it does not collide with the now-canonical
`language` id, matching the term §5.8/DEC-56/DEC-73 already use in prose for the identical
concept); DEC-44's `parent: language_code` field. **Left as originally recorded, because they cite
a specific dated measurement or a specific other document's then-current text rather than assert
an ongoing structural fact:** every receipt citation naming `code` (§4.0, §4.2, §4.3, §5.x, §9,
§11 — the value was measured under that name and the receipt file on disk still carries it),
DEC-56's and DEC-73's `language_code` faculty-target citations (§5.8, dated to the 2026-09-03
dataset-factory pass), and §7's invalidation-table pointers into other design docs' own
then-current line numbers. **The compatibility rule is unchanged and this revision does not touch
it:** receipts, matrix cell directories, Hub branches/tags and corpus fingerprints keyed by region
continue to read `"code"` and `"vl_latent"`; `cogsyndelta.regions.aliases` is the only place the
mapping lives, and every reader resolves through it.

**Scope:** what each region *is* as a faculty; what white matter is, mechanically; what it
emits and how a runtime executes it; how it is trained and on what; the experiment that
distinguishes integration from dispatch; the replacement for P4/P5/P6; the reserved corpus
that phase 2 needs; the scale path to 30B-class on one 16 GiB card; the precise list of
sections in five other design docs that this invalidates; and — new in revision 2 — the
disposition of every finding from the two adversarial passes (§10) with the refutations
evidenced (§11).

**What it supersedes on ratification:** `docs/program/CSD-BRAIN-REGIONS.md` (which does not
exist in the tree and is cited as authoritative by six tracked files), P4 and P5 in
`program/REMAINING.md` and `program/csd-program.json`, and the `CognitiveRegion.activate`
contract in `src/cogsyndelta/contracts/region.py`.

---

## Legend, and how to read the evidence tags

| tag | meaning |
|---|---|
| **[V]** | Verified by me in this session by reading the named file. |
| **[V\*]** | Verified in a file by a surveying agent with a `file:line` citation **and** independently reproduced by at least two of the three architecture proposals or by a judge who re-checked it. Treated as fact; not re-read by me. |
| **[I]** | Inferred — reasoning over [V] or [V\*] facts. Argued where it is load-bearing. |

**One extension to `[V]` in revision 3.3, stated so the tag does not quietly widen.** §1.3's
`episodic_store` cell cites `file:line` in **another repository** — the operator's `memory-gate`
(HEAD `2c11c3f`) and `memory-gate-rs` (HEAD `4b9f60d`) — read by a contract-extraction scout, not
by me in this tree. Those citations still carry `[V]`, because the standard `[V]` sets is *a named
file was read at a named line*, and that standard is met. **What is added is the repo and the HEAD
beside the claim**, so a re-check is possible; a `[V]` with no tree named would be a weaker claim
wearing the same tag. **`[V]` citations into that tree appear nowhere else in this document.**

**A second extension, admitted rather than performed silently** `[S33-9 fixed]`. §8's gaps block
and §1.3's `episodic_store` cell both need to cite an **absence** — *"no `user_id` field exists"*,
*"zero hits for `overlay` across both `src/` trees"* — and an absence has no `file:line` to name,
so it cannot meet `[V]`'s own standard of *a named file was read at a named line*. These now carry
**`[V-abs]`** instead: same evidentiary weight (the scout, or I, actually ran the read or the
grep), but a different tag, so a reader scanning for citations that can be re-opened at a specific
line is not misled into thinking one exists here. **`[V-abs]`'s standard: the grep or exhaustive
read that produced the zero-hit result is named beside the tag**, the same way a tree is named
beside a cross-repo `[V]`.

Nothing in this document is asserted from a document that a measurement contradicts. Where a
number is disputed between sources, both are given and the conservative one is adopted.

**Provenance of this document.** It is assembled from a grounding survey brief, three
independent architecture proposals (biology-first, ml-first, systems-first) and three
independent judge reviews (operator-fidelity, trainability-and-measurement, scale-and-systems).
The spine is `proposal-ml-first.md`, which won the judge tally. Every graft from the other two
is attributed inline. Every fatal flaw a judge named is either dropped or explicitly refuted;
§9.13 lists them.

Revision 2 adds two further inputs: the **skeptic pass** (`scratchpad/design/attack-skeptic.md`,
findings A1–A35) and the **threat pass** (`scratchpad/design/attack-threat.md`, findings T1–T4b
plus the theatre list). Revision 3 adds a third: the **second skeptic pass**
(`scratchpad/design/attack-skeptic-2.md`, findings N1–N10), run against revision 2. Both attack files are held unchanged as the record of what was argued;
this document is the record of what was *done about it*. It also carries eight operator inputs
recorded verbatim in the session's memory files, cited inline as **[OP: file-name]**.

**One measurement discipline applies throughout and is new in this revision.** A guard, a
baseline or a gate is not accepted because its reasoning is sound. It is accepted when someone
has made it fail on purpose. Four guards in this programme were structurally incapable of
firing; the fix each time was to construct the case that should trip it. Revision 1 diagnosed
that pattern in four places and then reproduced it in three of its own (A1, A6, A11). Every
gate added or changed below carries the construction that makes it fail.

---

## Decision index

| # | decision | section |
|---|---|---|
| **DEC-01** | `code` → `language_code`: language faculty, code specialisation. **AMENDED IN REVISION 3.6 BY DEC-78: the canonical id is `language`, not `language_code`** — the operator's 2026-09-04 faculty-naming rule settles it by fiat (id `language`, `specialisation: "code"` recorded as metadata); `code` remains correct for the corpus and the battery, never for the id. This supersedes §10.4/§11's disposition of the bare `code` → `language` proposal (recorded **dropped**, on falsifier grounds, not name grounds) by ruling rather than by re-arguing it | §1 |
| **DEC-02** | `compress` + `retrieve` → `memory`: one hippocampal faculty, one trunk, two heads. **Licence cost: `memory` inherits `retrieve`'s NC term** (DEC-31) | §1 |
| **DEC-03** | `vl_latent` → `visual`: visual cortex, emits patch tokens; **deployed half is the EMA target encoder** (DEC-34) and its resolution is rebuilt in W7v | §1 |
| **DEC-04** | `reason` → `reasoning`: keep, with the honest limit recorded and a phase-3 objective change | §1 |
| **DEC-05** | `classify_banking77` → probe, not a region | §1 |
| **DEC-06** | `classify_go_emotions` → probe of a new declared `affect` faculty | §1 |
| **DEC-07** | `residual_mlp` → retired; the residual path is a primitive of every block | §1 |
| **DEC-08** | `stream_vae` → retired as a region, promoted to a **tract codec** primitive | §1 |
| **DEC-09** | untracked `regions/retrieve.py` → **split**: adopt its BEIR eval, retire its FiQA-only training regime | §1 |
| **DEC-10** | `episodic_store` → the AI-specific region. Superseded by DEC-32 in revision 2; **REINSTATED IN ITS ORIGINAL SHAPE BY DEC-49 in revision 3.3** — a required, writable, non-parametric region participating in the workspace with its own budget `b_store` | §1.3 |
| **DEC-11** | `docs/program/CSD-BRAIN-REGIONS.md` → retired, its four unique decisions re-homed | §1.5 |
| **DEC-12** | `core/` (1,320 lines) → **harvested and marked superseded in the module docstring; not imported, not deleted** | §1.6 |
| **DEC-13** | Operator decision **D1** answered: **(c)**, with the division of labour fixed by *when* each half runs | §2.1 |
| **DEC-14** | The region contract becomes `tokens(inputs, context_tokens) → (h, mask)`; `activate(stream)→[B,D]` dies | §2.2 |
| **DEC-15** | Two budget currencies, two simplexes: `ctx_r` (pre-computation, region-internal) and `b_r` (post-computation, workspace KV) | §2.2, §2.4 |
| **DEC-16** | White matter = bounded Perceiver workspace; connection strength `a_r` is read off the cross-attention, never predicted | §2.3 |
| **DEC-17** | **Write-back**: admitted regions at iteration ≥1 receive the workspace latents as a conditioning prefix. This is what earns the DAG's edges | §2.4 |
| **DEC-18** | Topology is derived twice — by admission depth and by cycles in the effective connectivity — and the two derivations must agree | §2.4 |
| **DEC-19** | Training is four phases: A dense → B distil → C sparse → **D task-loss scheduler**, with collapse structurally bounded | §2.6 |
| **DEC-20** | Frontal cortex is built now, inside white matter, **with its own objective `L_unify` and its own ablation** | §3.1 |
| **DEC-21** | Thalamic gating is built now as **afferent bandwidth** gating; a *content* gate is a declared seam, not built | §3.2 |
| **DEC-22** | The reserve is **constructed**, not fetched; NSRS admission with a trivial-baseline condition | §5 |
| **DEC-23** | `codeparrot/apps` and `deepmind/code_contests` are marked `compose` in the ledger **now** | §5.2 |
| **DEC-24** | One shared token embedding across text faculties, mandatory from the first region trained after ratification | §6.2 |
| **DEC-25** | int8 KV is the v1 scale assumption, not the contingency | §6.3 |
| **DEC-26** | Interconnect parameter share capped at **2–5%** of total, decided now while `D` and `L_ic` are free | §6.4 |
| **DEC-27** | PTQ gates on **schedule divergence** `D_sched`, not only on the metric | §6.5 |
| **DEC-28** | Paging: a declared seam with a derived trigger, request/stage granularity, not built at v1 | §6.6 |
| **DEC-29** | Phase 3 uses **region-granular pipeline parallel**, microbatch ≤ 256 cross-host | §6.7 |
| **DEC-30** | The receipt reports **two verdicts** — integration, and scheduling — never one number | §2.7.6 |
| **DEC-31** | **Release licence: the strictest input sets the term.** The composed mind is **NC**; `memory` inherits `retrieve`'s NC term from GooAQ. Per-region tiers matter only for standalone shipping | §1.2, §5.7 |
| **DEC-32** | ~~`episodic_store` is **demoted to `placeholder` at v1**~~ — **SUPERSEDED BY DEC-49 in revision 3.3, by operator ruling.** What survives it: the contract-first discipline (nothing is `built` without a write policy, a receipt and a gate) and the three controls it wrote (bounded capacity, declared eviction, server-derived partition key), all of which DEC-49 keeps and sharpens. What does not survive: the demotion, the build trigger, and `R = 4` | §1.3, §2.7.7, §9.9 |
| **DEC-33** | **Memory-gate overlays** are the dynamic-paging seam's first client: tiered residency, fingerprint-bound attach, loud disconnect, strictest-input licence. A first-class manifest artifact type | §6.6, §7.3 |
| **DEC-34** | **`visual`'s deployed half is the EMA `target_encoder`**, not the context encoder. Verified: `IJEPA.encode` is `target_encoder.embed` and every visual number in the tree is that network | §1.2, §2.3 |
| **DEC-35** | **W1 is measured and its rule fired: the central bet is dead as trained.** A token-aware retrain (W4, then W7) is **mandatory before W5**, with the objective and gate specified. **Amended in revision 3: the verdict is PROVISIONAL PENDING W1d**, because it is definition-dependent — participation ratio kills the bet, entropy-effective rank does not `[N3 fixed]`. **RESOLVED in revision 3.2: W1d has run and CONFIRMS W1** (clean for `code`, `compress`, `vl_latent`; **CONFIRMED-BUT-NOISY** for `retrieve`), **no region OVERTURNED**, the retrains are **LICENSED**, and the **penultimate-block fallback is dead — `L_token` attaches at the FINAL block** | §4.0, §4.1 |
| **DEC-36** | **Admission never sets a baseline.** NSRS thresholds are calibrated on a split that is not graded; B1/B2 are recomputed on the graded split by a separate forward pass; τ is stated in chance-normalised units per bin | §5.3, §2.7.1 |
| **DEC-37** | **G3′ is the synergy conjunction**, not the mean interaction: `Δ_A > 0` **and** `Δ_B > 0` **and** `I > 0`, per pair. The redundancy quadrant is reported as `REDUNDANT` and is a **FAIL**. **Amended in revision 3:** every ablation pair carries a declared item shape and a printed sealed-item count, and a redundant *corpus* is separated from a broken *statistic* `[N2 fixed] [N6 fixed]`. **Amended again in revision 3.3 (DEC-49):** `R = 5` ⇒ **ten** pairs, criterion **≥ 7 of 10**, null rate **0.1719**, with the two new shapes **X7/X8** declared so the denominator is not padded with pairs that cannot pass | §2.7.3, §5.4 |
| **DEC-38** | **Reservation is at source-row fingerprint granularity**, not composite-hash granularity; train/eval splits and bootstraps are by source row | §5.6, §5.4 |
| **DEC-39** | **Keyed split assignment.** `split = HMAC(k_split, source_row_fingerprint) mod N`, `k_split` held outside every path an agent can write. No key ⇒ the loader refuses to build a split | §5.6 |
| **DEC-40** | **One `load_checkpoint()`** in `src/`, `weights_only=True` hardcoded, manifest SHA-256 verified before the file is opened, enforced by a CI lint rule. **Revision 3 gives it a row (W0c) and a named owner** `[N10c fixed]` | §9.9, §4.1 |
| **DEC-41** | **The v1 read-out is a ranking head** over a declared candidate set; abstention is an explicit **NULL candidate**, not a new head. B1/B2/G2 are defined in that metric | §2.6, §2.7.1 |
| **DEC-42** | **The aqua_rat recovery burns the UNION of every draw the ledger discovers** — the reservoir draw **R** at `seed = 0`, the prefix draw **P** (the first 4,982 rows in shard order), **and, per W2a's actual run (branch `feat/w2a-ledger-recovery`, `c6ea587`, approved), a third on-disk derived sample D3** it found under `reason/aqua_rat-raw/derived/` — because the sampling algorithm changed at `c42203c`, no surviving receipt records `cap_sampling`, and a discovered artefact of ambiguous provenance is exactly the case this decision exists to catch. **Measured, not bounded: the three-draw union is 13,946 of 97,467; clean is 83,521** `[V*, burned-aqua_rat.jsonl.manifest.json]`. A ledger that omits **any discovered draw** **fails** W2a `[N1 fixed]` | §5.1, §4.1 |
| **DEC-43** | **`auditory` is a DECLARED SEAM, deferred to a production phase — AMENDED BY DEC-48.** Revision 3.1 made it required-in-phase-1; the operator's later ruling the same day defers it. What stays declared, so nothing is redesigned later: the interface (patch tokens over spectrogram frames, byte-identical to `visual`'s), the objective family (masked latent prediction) and the completed corpus audit. What moves: rows **A0–A3**, now **deferred**, `blocked_by` *the composed mind passes W6 without audio; operator go* `[OP: csd-multimodal-io-intent.md]` | §1.3, §1.4, §4.1 *Production phase: audio* |
| **DEC-44** | **`speech_output` is a DECLARED SEAM on the language centre, deferred — AMENDED BY DEC-48.** One generative trunk emitting **text tokens and a discrete speech-token stream** stays the declared shape, and the **`Schedule`'s output-modality field and the §2.5 output block are KEPT AS DECLARED NOW**, precisely so the runtime contract does not change when audio is picked up. It is a **head, not a region and not a participant**: `emits: null`, `parent: language`, and `language` carries `heads: [text, speech]` `[S31-12 fixed]`. **Image output stays a declared seam.** **Re-keyed from `language_code` to `language` in revision 3.6 (DEC-78)** | §1.3, §2.5, §4.1 A2 |
| **DEC-45** | **The audio licence tiers, under DEC-31's strictest-input rule — now groundwork for a deferred phase.** Both clean tiers are **CC BY 4.0**, not *"MIT / CC BY"*: CC BY is the strictest input in each mix, MIT is unreachable, and a slash is two verdicts rather than one `[S31-6 fixed]`. **HiFiTTS-2 is excluded and the exclusion is declared** `[S31-20 fixed]`. The NC options still cost **nothing at the composed tier**, because the composed model is already CC BY-NC-SA via `memory`. **ND (TED-LIUM) is outside the NC-tolerant policy** (OD-10); Common Voice is a **consent-hygiene** class, not a licence class (OD-11) | §5.7, §8 |
| **DEC-46** | **LibriVox is ONE provenance group** for B1/B2, not ten datasets. Ten of the sixteen audited speech corpora draw on the same volunteer public-domain-audiobook pool, so a mix picked per dataset name is a monoculture wearing ten names. The cap is applied to the **group**, before any source is sized. **Amended in revision 3.2:** the balance conclusion is **`[I]` and cap-dependent, not a finding** — `N_eff ≥ 3` holds only at a per-group cap of order a few hundred hours and fails at 1,000 h `[S31-2 fixed]`; B1/B2 are share statistics and are computed **in hours**, over **both** mixes, against **both** the 0.50 hard line and the 0.40 operating cap `[S31-5 fixed] [S31-14 fixed] [S31-15 fixed]`; and the group count is *seven provenance groups, of which five carry no NC term* `[S31-13 fixed]` | §5.7, §4.1 A0 |
| **DEC-47** | **THE LATENT-SPACE REASONING INVARIANT.** Encoders convert each modality (text tokens, image patches, later audio frames) into **latents**; from that point on **nothing is re-serialised to discrete tokens until the read-out**. Reasoning happens in the white-matter workspace latents, and **the scheduler's iteration depth IS the latent-reasoning loop** (its future form is the deferred recurrent-depth direction, §6.9). Regions exchange **latents only** — write-back conditioning prefixes are latents, and no region emits text to another region. **Output modality is chosen at the frontal read-out** (DEC-44, §2.5): emission is a decoder choice, not a property of the reasoning. *"Discrete latent concepts"* is a hypothesis about **representation**, not a licence for a token bottleneck — quantised concept codes are a candidate **tract codec** in the retired `stream_vae` slot (DEC-08), and continuous latents stay the default until measured. **Gate (W9): a runtime assertion that no inter-region path carries discrete token ids, with the constructed violation refused** `[OP: csd-latent-space-reasoning-invariant.md]` | §2.2, §2.4, §2.5, §4.1 W9, §6.9 |
| **DEC-48** | **AUDIO IS A PRODUCTION-PHASE FEATURE. Supersedes DEC-43 and DEC-44 as *required in phase 1*.** Operator ruling, later the same day: *"we can wait to add audio as a future feature once it proves out without audio. that will be more of a production phase implementation"* `[OP: csd-multimodal-io-intent.md]`. `auditory` and `speech_output` become **declared seams** — interface, objective family and runtime output block kept — and rows **A0–A3** are **deferred**, `blocked_by` *the composed mind passes W6 without audio; operator go*. **v1 therefore has FOUR participants and SIX ablation pairs, unchanged.** The audio licence audit and the A0 catalogue work stay as **completed groundwork** | §1.3, §4.1, §4.3, §5.7 |
| **DEC-49** | **`episodic_store` IS REQUIRED AND IS BUILT IN PHASE 2. Supersedes DEC-32; reinstates DEC-10's shape.** Operator ruling: *"So that will have to have the episodic store created to back it rather than just dropping it. It should have partition, capacity, integration contract"* `[OP: csd-episodic-store-required.md]`. The store is a **workspace participant with its own budget `b_store`**, its contract taken clause-by-clause from `memory-gate` / `memory-gate-rs` where those repos specify it and **left open, not invented, where they are silent** (§8's gaps block). **`R = 5`, ten ablation pairs, G3′ ≥ 7 of 10 at null rate 0.1719**, collapse floor `η/R` = **3.0%**. Rows **E0/E1/E2**. §9.9 B2's threat findings are the **acceptance gates** | §1.3, §1.4, §2.3, §2.7.3, §2.7.7, §4.1, §5.4, §6.8, §8, §9.9 |
| **DEC-50** | **THE INCREMENTAL INTEGRATION PROTOCOL** — a named, three-step procedure for admitting or changing any submodel: (1) train the submodel **alone** under phase-1 discipline against its own untrained baseline; (2) **retrain the interconnect** with it admitted (white matter A→B→C→D, regions frozen), gated on G2 + composed-metric improvement + non-zero attention mass to the new region, **else revert**; (3) **whole-mind unified training** under the monotone-improvement rule. **DEC-26's 2–5% cap is what makes step 2 cheap.** E0/E1/E2 are its first instance `[OP: csd-incremental-integration-protocol.md]` | §4.1, §6.4, §4.1 P5′ |
| **DEC-51** | **THE DOWNSTREAM BAR IS MYCELIUM, AND IT IS A MEASUREMENT.** CSD's success criterion is not a benchmark but whether it can act as the development engine for the Mycelium functional value-semantic language project. Row **M0**, after W10: a task suite drawn from the Mycelium repo with ground truth from its own tests, CSD vs a comparable open model, on the 5080 and the 3090 Ti with RAG via the 1080 Ti helper, **gate = a pre-registered pass-rate margin**. The **open question is whether CSD does RAG natively as a skill** (`memory` + `episodic_store` end to end) — an arm of the experiment, never an assumption `[OP: csd-mycelium-downstream-goal.md]` | §4.1 M0, §9.14 |
| **DEC-52** | **THE I/O CONTRACT: ANY-IN / ANY-OUT, WITH A MUST-HAVE BASELINE.** *Rationale: the contract the model is built to satisfy was scattered across three sections and a deferral, so "what must work" and "what is intended eventually" could not be told apart.* Every modality (discrete text tokens, image patches, later audio frames) is encoded to latents, **unified in the shared latent space**, reasoned over there, and emitted in the decided modality — *"any in, any out"*. **The MUST-HAVE BASELINE, and the whole programme is graded against it: discrete-token + visual INPUT, discrete-token OUTPUT.** That is what proves the endeavour is worthwhile; everything beyond it is polish. **Output modality is chosen at the read-out** — from context, **or from an explicit user constraint, which is BINDING** (*"only text"* means only text). Audio in and speech out stay declared seams under DEC-48 `[OP: csd-training-placement-policy.md, csd-multimodal-io-intent.md, csd-latent-space-reasoning-invariant.md]` | §1.3, §2.5, §4.1 W7p |
| **DEC-53** | **THE LATENT INVARIANT GETS A DETECTOR, NOT ONLY A REFUSAL.** *Rationale: W9's DEC-47 clause refuses a violation someone constructed, which says nothing about the tracts nobody thought to construct one for.* Row **W9i**: a **token-round-trip probe** that instruments **every** tract in an emitted `Schedule` — region output, adapted form, workspace latents, DEC-17 write-back prefix — and reports a **positive detection** of any integer-typed, vocabulary-indexed or decode-then-re-encode payload, with the per-tract dtype and value-range census printed in the receipt. **It can fail in both directions and both are verified:** a constructed round-trip must be **detected** (not merely refused), and a clean run must report **zero** detections across every tract, so a probe that detects nothing because it inspects nothing is caught `[OP: csd-latent-space-reasoning-invariant.md]` | §2.2, §4.1 W9i |
| **DEC-54** | **TRAINING PLACEMENT POLICY: PACK BY VRAM BUDGET ACROSS ALL THREE CARDS.** *Rationale: submodels are several million to a few hundred million parameters and fit every card, so serialising them on one card wastes two thirds of the fleet — but the one region that does not fit taught the lesson expensively.* Concurrent submodel runs are admitted **only when their per-job VRAM budgets fit** the target card (3090 Ti 24 GiB sm_86, 5080 16 GiB sm_120, 1080 Ti 11 GiB sm_61), under **process-level pseudo-isolation** — per-process memory fractions, `expandable_segments`, warp-level sharing. **No MIG on consumer cards; this is pseudo-isolation, not hard isolation, and it is acceptable because the fleet is single-tenant** (operator + agents), which is stated so nobody later reads it as a security boundary. **Launch tooling takes a VRAM budget and a target host**; **every receipt records `host`, `vram_budget_mib` and `concurrency`.** **The measured constraint: `reason` runs ALONE at batch 512 / `max_len` 256 with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — batch 1280 OOMed 640 MiB short of the 22 GiB card. Row **W7p** `[OP: csd-training-placement-policy.md, csd-baseline-receipts-2026-09-03.md]` | §4.1 W7p, §6.7, §4.3 |
| **DEC-55** | **LAYER-SECTIONED TRAINING IS A CANDIDATE FOR LARGER MODELS — NOT A DECISION.** *Rationale: at 1B+ per region the fleet either contributes three cards or two, and the technique that decides which is one the operator flagged as worth considering while explicitly doubting parts of their own understanding of it — recording it as a candidate keeps both halves.* Training isolated **sections of layers/weights** on different cards, rather than the whole model at once, sits **beside DEC-29** (region-granular pipeline parallel, microbatch ≤ 256 cross-host) and beside the standing ruling that **cross-host DDP is the wrong tool at 1 Gb/s**. Row **P5′L**, deferred: it is **evaluated on measured interconnect cost per section boundary against DEC-29's pipeline cost on the same model**, and adopted only if it wins. Until then, later phases may legitimately be **locked to the 3090 Ti + 5080, or to the 3090 Ti alone** `[OP: csd-training-placement-policy.md]` | §6.7, §4.1 P5′L |
| **DEC-56** | **THE 1B-PER-SUBMODEL SCALE PATH, AND DATA IS THE LONG POLE.** *Rationale: the operator's post-phase-3 intent has a sequencing constraint and a data requirement that decide whether it is reachable, and neither was written down.* Sequence: toy (~87M) → **mid-size proven** (DEC-52's baseline, W6 green, PTQ on the deployment card) → **~1B parameters per submodel** → **quantise each region** → **train the composed model on the quantised regions** → the 30B direction (OD-6). **The data requirement is stated rather than implied: of the order 10^10 tokens PER REGION** under the usual scaling rules, each licence-verified under DEC-31 and balanced under B1–B5 — which is why DEC-57's factory is the reusable asset and not a side quest. **Per-region PTQ sensitivity at 1B decides the composed bits/param budget: the 3.2675 figure is RE-MEASURED at size, never carried forward.** The path proceeds **region by region under DEC-50**, so it never becomes one untestable jump. Row **P5′b** `[OP: csd-billion-per-submodel-scale-path.md]`. **AMENDED IN REVISION 3.5 BY DEC-73: the flat 10^10-per-region number is replaced by PER-FACULTY targets.** The dataset factory's pass 1 measured the B1-bounded reach of every faculty and four of seven fall one to three orders of magnitude short (`visual` **0.3%**, `memory` **1.3–5%**, `reasoning` **1.8%** without a generator, `language_code` **15%**), while `language_trunk` overshoots **60×** — so one number applied to seven faculties is not a requirement, it is an average nobody can act on. Row **P2′g** `[OP: csd-dataset-factory-pass1-result.md]` | §6.1, §4.1 P5′b, §4.1 P2′g, §5.8, §6.5 |
| **DEC-57** | **THE DATASET FACTORY: ONE STRUCTURED I/O LOOP, GENERALISED FROM WHAT EXISTS.** *Rationale: sourcing, licence deconfliction and polishing is the long pole for DEC-56, and the pieces already exist in embryo — rewriting them would discard the provenance discipline they encode.* The loop is **search → identify → capture provenance → licence verdict → ingest → emit a licence-compliant open dataset with full provenance**, drivable by an agent, with a **human-visible provenance receipt per dataset**. It **generalises, and does not replace**: `csd-corpus-expand` catalogue entries carrying `provenance_group`, **upstream licence verbatim held as a separate field from the mirror tag** (the mirrors lie), `REFUSE` verdicts **enforced by the fetcher** rather than recorded beside it, `LICENCE-FOR-OPEN-WEIGHTS.md` / `AUDIO-CORPUS-AUDIT.md` as the audit format, the corpus contract's B1–B5 balance rules, and NSRS admission with source-row fingerprints. **DEC-31's strictest-input rule applies to every dataset the factory emits.** Row **P2′f** `[OP: csd-dataset-factory-and-moral-corpus.md]` | §5.5, §5.6, §4.1 P2′f, §7.1 |
| **DEC-58** | **SYNTHETIC DATA HAS ITS OWN CONTRACT, AND THE GATE CAN FAIL.** *Rationale: generated data carries garbage the operator will not train on, and "we will be careful" is not a control.* Four clauses, all failable: **(1) the generator is NAMED in provenance** — model, revision and licence — extending §5.6's existing *"no generating model, or the model named"* rule from the reserve to every corpus; **(2) contamination channels are run against EVERY eval**, not the one the data was made for, and the gated-channel counts are printed; **(3) a QUALITY GATE that can fail** — the synthetic bin must beat a matched human-authored sample on the bin's own metric, or the batch is discarded, verified by feeding it deliberately degraded output and asserting rejection; **(4) a CAPPED SHARE of any bin**, counted as a B1 statistic on provenance groups so a generator cannot become a monoculture wearing many names. **The operator's stated preference is preserved as the tie-breaker: extremely high quality and requirement-complete over large.** Row **P2′s** `[OP: csd-dataset-factory-and-moral-corpus.md]` | §5.6, §4.1 P2′s |
| **DEC-59** | **THE MORAL CORPUS IS A CURATED DATASET WITH EVALS, AND IT IS A SCALE-UP ROW.** *Rationale: the operator wants values captured at the training-data level so they are baked in by training rather than bolted on by filtering — and wants to know whether that actually works.* It gets **its own CORPUS-CONTRACT entry**, its own **provenance chain**, its own **licence verdict** under DEC-31, and — the part that makes it a decision rather than an aspiration — **its own held-out moral/safety probes**, so *"training on it changes measured behaviour"* is a **measurement against an untrained and an unmoral-corpus control**, never an assumption. **Phase-3 / scale-up, explicitly NOT a toy-scale row**: at 87M there is no behaviour to move and a null result would be uninterpretable. Row **P5′m** `[OP: csd-dataset-factory-and-moral-corpus.md]` | §5.5, §4.1 P5′m, §7.1 |
| **DEC-60** | **AUTODEV LIVES IN `csd-autodev`; THIS DOCUMENT KEEPS A POINTER AND THE CSD-SIDE PREREQUISITES.** *Rationale: autodev is a harness meant to run several models, and coupling it to one model's repo makes both harder to reason about and pollutes CSD's licence and provenance story.* Operator ruling: *"remember to keep autodev work in the autodev tree and repo so we dont mix autodev work with CSD work"* `[OP: autodev-work-lives-in-csd-autodev.md]`. The spec is **`docs/AUTODEV-IDENTITY-AND-SANDBOX.md` on branch `docs/autodev-spec` in `tzervas/csd-autodev`**, with its threat and skeptic passes beside it; **OD-1 and OD-2 are ANSWERED there** (`svc-autodev` with its own vault and a minimum-scope Forgejo user, no root and no sudo, the lab-console as sole egress gateway minting short-lived per-request tokens, a separate curated RAG vault) `[OP: autodev-service-identity-and-sandbox.md]`. **What stays HERE is only what CSD depends on: rows PRE-1, PRE-2 and PRE-3, which BLOCK any autodev GPU route** — because a design that assumes a control it does not own is how the assumption outlives the control | §8, §4.1 PRE-1/2/3, §9.9 B5 |
| **DEC-61** | **GOVERNANCE: `main` IS PR-ONLY, AND OD-1's PATH-SCOPED PROTECTION IS CORRECTED BY MEASUREMENT.** *Rationale: the PR is where CI legitimacy, review and the merge record live, and direct pushes skip all three — but the file-pattern control OD-1 recommended turns out to block the reviewed path it was meant to protect.* No direct pushes to `main` in any code repo, **the operator's own commits included, on principle**. CogSynDelta enforces it: **push whitelist = the operator alone; 8 required `pull_request` contexts** (the `(push)` variants never appear on feature-branch heads, so requiring them made every PR unmergeable). **`protected_file_patterns` is CLEARED, and this REPLACES OD-1's recommendation rather than sitting beside it:** Forgejo refuses to merge **any** PR touching a protected file — 405 *"Changed protected files"*, **admins included**. **Guard integrity therefore rests on the required checks — including `tests/test_guards_can_fail.py` — plus review, and nothing else; counting the file-pattern rule would be counting a control that does not exist.** Every mutating agent works in an **ephemeral worktree** and pushes a **sha**; merges are fast-forward or `--no-ff` from pushed shas `[OP: branch-and-pr-to-main-only.md]` | §8 OD-1, §9.9 B5 |
| **DEC-62** | **W2c's ANSWERS: THE 0.40 CODE FLOOR IS RETIRED, AND `retrieve`'s CHANCE FLOOR BREAKS NSRS CONDITION (1).** *Rationale: one of W2c's two measurements settled a documentation defect; the other created a live hole in the admission filter that no rule covers.* **(i)** The ≈0.40 code lexical floor is **unsupported and retired** — measured 0.2285 at seed 0 and 0.2344 at the region seed. **That retirement is recorded at `docs/design/evidence/w2c-untrained-baselines-2026-09-03/` and is CITED here, not restated**, so there is one copy of the number. **(ii) THE NEW FINDING: `retrieve`'s `τ_lo` is 0.0000 in chance-normalised units**, because its untrained r@1 equals chance exactly (1/512 = 0.00195). **NSRS admission condition (1) — `max_r s_r < τ_lo` — is therefore BARELY SATISFIABLE on that bin**: at `τ_lo = 0` the condition admits by construction rather than by evidence, and *"no single region suffices"* becomes untestable there. **W2b must DEFINE admission for a chance-floor bin BEFORE it runs** — an **absolute-margin floor** (`max_r s_r < chance + δ` with `δ` pre-registered) **or a different normaliser** — pre-registered with its reason, and the choice **verified by making it fail** on an item a single region trivially solves `[OP: csd-baseline-receipts-2026-09-03.md]` | §5.3, §4.1 W2c, §4.1 W2b, §2.7.1 |
| **DEC-63** | **EPISODIC STORE GAP (a): DYNAMIC CAPACITY AND STALENESS DECAY — DEFAULT ADOPTED.** *Rationale: the operator asked for the contracts nailed down rather than deliberated, and gap (a) has an implementable default that E1's receipt then turns into a measurement.* `capacity_bytes(host, tick) = max(0, VRAM_total − KV_reserved(context_len, regions_active) − activation_reserve − safety_margin)`, a **pure function of live inputs, computed per host at scheduler-tick time and NEVER CACHED** — the 1080 Ti is preemptible and the active-region set changes under the store, so a capacity fixed at process start is not dynamic. **Staleness is an exponential half-life on `last_accessed`**, half-life a config constant starting at the order of the workspace context window in wall-clock time; the receipt **records which function ran**. **OPERATOR TO CONFIRM, two things and they are different questions:** the `safety_margin` value (recommend 2 GiB, the order of `hypha`'s display reserve), and **whether a residual claim is acceptable at all** given §9.11's finding that the residual may round to zero on the 5080 — the alternative being a **fixed floor reserved before the KV budget**, which trades context length for recall and is a product decision `[OP: csd-episodic-store-required.md]`. **AMENDED IN REVISION 3.5 BY DEC-70: `capacity_bytes` is a CEILING, NOT A TARGET.** The formula says what the store *may* claim; it never said what the store *should* claim, and a residual read as an allocation is how a frugal subsystem becomes the largest one on the card. The resident set is what the residency score admits **now**, bounded additionally by DEC-70's frugality cap, and `capacity_bytes` is the ceiling that bound may not cross `[OP: csd-memory-gate-overlays.md]` | §1.3, §8 gap (a), §4.1 E1, §4.1 E1a, §6.6, §9.11 |
| **DEC-64** | **EPISODIC STORE GAP (b): THE PARTITION AXIS IS `(scope, domain, logical_key)` WITH `scope` = THE AUTHENTICATED PRINCIPAL.** *Rationale: §9.9 B2's whole defence is a server-derived scope, and `domain` — a closed fleet task taxonomy — is a label, not an isolation boundary; CSD's regions are faculties, so `domain` does not even name the right kind of thing.* Three segments. **`scope` is the authenticated principal, DERIVED SERVER-SIDE and never client-supplied**; `domain` is retained as memory-gate's task axis so the ported conformance tests still bind; **`session` is a sub-segment the request may NAME and the server may BOUND**. **Persona/basin is a SCOPE SELECTOR and never a routing object** — inherited from the one part of that design that was fully thought through (*"Personas MUST select a memory basin/configuration; they MUST NOT be modeled as MoE experts or independent agents"*). **OPERATOR TO CONFIRM — this is the one gap no measurement decides:** principal isolates users and lets one user's episodes accumulate forever; session makes X7 items work and long-horizon memory impossible; persona-basin isolates contexts within one user. **Recommended and adopted as the default: principal, with session bounded beneath it** `[OP: csd-episodic-store-required.md]` | §1.3, §8 gap (b), §4.1 E1, §9.9 B2 |
| **DEC-65** | **EPISODIC STORE GAP (c): INDEX-NOT-BYTES, LATENTS OWNED BY THE RUNTIME, TEXT SIDECAR OFF THE READ PATH.** *Rationale: DEC-47 forbids re-serialising to discrete tokens on an inter-region path, so memory-gate's text-plus-embedding payload is simply not available to CSD as a workspace participant — it would place a token bottleneck exactly where the invariant says there must not be one.* The store owns an **index and an eviction policy** over latent bytes **the runtime owns**, not a second allocator competing with the KV cache for the same VRAM. A record is `(scope, domain, key) → (pointer, residency, importance, last_accessed, byte_size, provenance)`; the read is `k = W_k z`, `v = W_v z` over stored latents. **This is the one clause where CSD deliberately DIVERGES from the source repos on the payload while adopting their structure, and it is called out rather than blended.** **A text sidecar is KEPT — as provenance metadata only, never on the read path** — because it is the only way a human can inspect what the mind remembered, and keeping it off the read path is what stops it becoming a token bottleneck by accident `[OP: csd-episodic-store-required.md]` | §1.3, §8 gap (c), §4.1 E1, §2.2 |
| **DEC-66** | **EPISODIC STORE GAPS (d) AND (e): PRUNE-ONLY AT v1, IN THOSE WORDS; OVERLAYS STAY IN P5′o.** *Rationale: a learned consolidation pass with no gate, no receipt and no corpus is the `residual_mlp` defect in a new costume, and E1 is a container with four failable gates that an unspecified feature would turn into a row with intentions.* **Consolidation at v1 IS scored eviction plus the durability ladder and nothing else — CSD's store forgets by policy rather than consolidating by learning, for the whole of v1.** A real complementary-learning-systems pass (a slow phase that **creates new representations** from old episodes, using the `RecordProvenance` field both source repos already define and neither uses) is **the store's own phase-3 candidate**, where P5′'s monotone-improvement rule can grade it. **Differential overlays are NOT pulled forward into E1**: their design cannot be imported — an exhaustive grep for `overlay`/`differential` returns zero hits in both source `src/` trees `[V-abs]` — and it is written fresh in **P5′o**, inheriting exactly two patterns: **index-not-bytes** for residency and **scope-selector-not-agent** for persona `[OP: csd-episodic-store-required.md, csd-memory-gate-overlays.md]` | §8 gaps (d)/(e), §4.1 E1, §4.1 P5′o, §6.6 |
| **DEC-67** | **DEPLOYMENT ACCEPTANCE: THE 5080 + 3090 Ti MUST RUN THE COMPOSED MIND; THE 1080 Ti IS THE RAG HELPER UNLESS CSD RETRIEVES NATIVELY.** *Rationale: M0 already carried this shape in its task text, and a shape in a task description is not an acceptance criterion — the card that is actually tight is the 16 GiB one, and nothing was scheduled to find out.* Row **M0d**: the composed, PTQ'd mind **loads and serves within VRAM on BOTH the 5080 (16 GiB sm_120) and the 3090 Ti (24 GiB sm_86)**, with the **store's dynamic capacity (DEC-63) reported separately in bytes on each card** — a footprint that omits the store has reported half the deployment. **The 1080 Ti (11 GiB sm_61) is the RAG residence with a helper embedding/rerank model, and it stays in the deployment unless M0's arm (b) demonstrates native retrieval** — *"unless CSD ends up capable of handling the various RAG processes natively as a skill"*, which is **measured in M0, never assumed** `[OP: csd-mycelium-downstream-goal.md, fleet-gpu-roles-and-scheduling]` | §4.1 M0d, §6.1, §9.14 |
| **DEC-68** | **W4's PRE-REGISTERED RUN IS MEASURED: THREE GATES PASS AS THE HARNESS SCORED THEM — ONE OF THEM ONLY BY FLOAT32 REPRESENTATION — TWO FAIL, AND THE GATES ARE NOT EDITED HERE.** *Rationale: the row was pre-registered precisely so its outcome could not be argued with afterwards, and the first thing a document is tempted to do when a pre-registration fails is to improve it.* **Measured, from the receipts and nothing else.** The batch-512 production run (`memory-20260903T164611Z.json`, code `4d2d886`) failed **four of five**. The **pre-registered batch-1280 run** (`w4-chunked/run-1280/memory-20260903T184441Z.json`, code `eb735ab`, `token_loss_chunk` 512, 4,000 steps, 1,586.7 s) **PASSES (a)** — `recall@1` **0.8535** against parents **0.7754**/**0.7559**, graded spearman **0.7893** against compress's **0.7588** — **PASSES (b) AS SCORED, CONTINGENTLY** — full-pool `recall@10` **0.200** at a **strict** `> 0.20` floor, MRR **0.1168** at a **0.10** floor; the receipt's value is `0.20000000298023224` = `float(numpy.float32(0.2))`, so the float32 recall **is** 0.2 and the margin is representation error, which makes (b) an **exact tie under a strict reading** and sends the `>`-versus-`≥` question to **OD-17** rather than settling it here — **PASSES (d)**, and **FAILS (c)** — **0.200 against BM25's 0.440**, MRR **0.117** against **0.308** — and **FAILS (e)** — final-block rank ratio **1.2102** against an absolute **2.0×**, with the regression clause at **0.0** passing. **The negatives curve, three runs at one changed variable:** batch **256 / 512 / 1280** ⇒ full-pool `recall@10` **0.030 / 0.098 / 0.200**, held-out `recall@1` **0.600 / 0.815 / 0.854**, rank ratio **1.59 / 1.35 / 1.21**. **The objective's effect at production length, which the 50-step control arm could not measure:** at batch 512 the control arm (both weights 0) reports rank ratio **1.0851** against **1.3475** with the terms on, `recall@1` **0.7754** against **0.8145**, graded **0.7313** against **0.7463** — the terms work and the 2.0× clause **does** discriminate at 4,000 steps, and both arms sit below 2.0. **What this document does with that: nothing to the gates.** **OD-17** puts *pivot per §9.14 as pre-committed* against *amend as a pre-registration* to the operator, both costed `[OP: csd-w4-control-arm-result.md]` | §4.0, §4.1 W4, §4.1 W4n, §8 OD-17, §9.14 |
| **DEC-69** | **THE KNOBS ARE ONE TABLE WITH MEASURED EFFECTS, AND gpu-pack's ADMISSION MODEL IS THE RUNTIME PACKER'S STARTING POINT.** *Rationale: packing and chunking were solved as two incidents; as a table they are a control surface, and the same admission arithmetic that keeps two training jobs off each other's VRAM is what will keep regions, overlays, the store and the KV cache off each other's at inference.* Six knobs, each with what it costs: **per-job VRAM budget** and **admission margin** (gpu-pack's `allowed = total − foreign_used − Σ max(budget, measured) − margin`, margin default 1,024 MiB); **`token_loss_chunk`** — **2048 ⇒ 20,726 MiB reserved / 22,120 MiB raw driver peak**, **512 ⇒ 19,444 / 20,883 at +2.6% step time**, chunk peak being `chunk × vocab` and therefore batch-independent; **batch, which IS the in-batch-negatives count** and carries DEC-68's curve; **mask probability** (0.15, which sets `n_masked` and therefore the token term's whole cost); **card choice by VRAM versus compute**. **And a standing FOREIGN claim that is not a knob and must never be absorbed into a margin: the operator's desktop holds 981–1,057 MiB on the 3090 Ti** (Xorg/KDE/Firefox, measured in both chunk probes) — a budget computed as if the card were headless is wrong by that amount, and gpu-pack already models it as `foreign_used` rather than as slack. **The admission model is reused, not re-invented, for inference packing.** Row **W7k** `[OP: csd-training-placement-policy.md]` | §6.7, §4.1 W7k, §4.1 W7p |
| **DEC-70** | **THE MEMORY GATE IS A VRAM ARBITER WITH A PRIORITY ORDER AND A FRUGALITY CAP — IT CONTENDS, IT DOES NOT TAKE WHAT IS FREE.** *Rationale: DEC-63 gave the store a residual claim on whatever the KV cache and activations left over, which reads as "memory gets the leftovers" and, read the other way, as "memory may take everything left" — the operator's ruling is neither.* **Claimants and priority, in order:** **base weights and the working activations are FIXED**; then the **ACTIVE memory set and the LOADED PERSONA, which are protected and are never evicted to make room for context**; then the **KV cache / context window, which FLEXES around them** (sliding window, shorter context); **inactive overlays, offsets and differentials sit BELOW context and leave first**. **The frugality cap is what makes protecting memory safe:** a stated constant caps resident memory + persona at a small fraction of the card, per card, enforced by the residency scorer — **gate: at the cap, the achievable context length differs from the no-memory case by less than a stated margin**. **Failure to size is reported, never silent:** a relevant memory/persona set that does not fit emits a receipt and an event and **degrades the memory set by relevance**; **the persona is never silently dropped**. **Residency is scored** on recency, relevance to the loaded persona/basin and access frequency, across **card / host RAM / disk**, and **eviction is proactive on inactivity and on pressure**, not only on capacity overflow. **Frugality is measured:** every receipt records peak VRAM held by memory + overlays and the fraction of it that was active. **DEC-63 is amended: `capacity_bytes` is the CEILING, not the target.** Row **E1a** `[OP: csd-memory-gate-overlays.md]` | §6.6, §4.1 E1a, §4.1 E1, §8 gap (a), §9.11 |
| **DEC-71** | **DIFFERENTIAL ACTIVATIONS — A LONG-TERM TRACK WITH A GATE, NOT A PHASE-2 FEATURE.** *Rationale: the operator is thinking past weight-delta overlays to activation deltas, and the honest way to hold an idea that is not yet buildable is a row with a falsifiable gate rather than a paragraph of intent.* Beside the weight overlays of DEC-33, an overlay may also carry **activation deltas relative to the base activations**, stored in a format **optimised for rapid recompute-and-apply** — a delta plus a cheap reconstruction rather than dense stored activations — so a persona or skill contributes features at inference **without holding dense state**. **Gate, and it is a comparison, not a demonstration: apply latency and VRAM per feature against the dense alternative, measured on the same items, and adopted only if it wins on BOTH.** **Post phase 3; not built in phase 2**, beside the ternary track (§6.9) and P5′o. Row **P5′d** `[OP: csd-memory-gate-overlays.md]` | §6.10, §6.6, §4.1 P5′d |
| **DEC-72** | **THE PREDICTIVE HYBRID TRAINING TRACK — EXTEND THE EXISTING PREDICTOR, AND GATE IT ON WALL-CLOCK.** *Rationale: the tool already exists and its own documentation claims 25% backward and 15% forward reductions; those are inputs to an experiment, not results, and the experiment has to be specified before the claim can be believed or discarded.* `tzervas/tritter` carries `GradientPredictor` / `PredictiveTrainer` / `LossPredictor` and a 962-line research report; this track **extends prediction from weight-update outcomes to (2) activations and activation deltas — which is where it meets DEC-71 — and (3) semantic residuals used as a correction signal**. **Four gates, all pre-registered:** prediction error against a **real** step on held-out batches per target under a stated tolerance; the **fraction of steps predicted**, printed; the **end-task metric against a fully-real run at EQUAL WALL-CLOCK and at equal steps, both reported**, because a method that is only faster per step has proved nothing; and an **automatic fallback to real steps when a predicted phase drifts past tolerance**, verified by constructing the drift. Fine-tuning is measured as its own case. **Strictly after phase-3 whole-model training is underway**, and it lives in tritter under DEC-75 with a CSD adapter only. Row **P5′p** `[OP: csd-predictive-hybrid-training-track.md]` | §6.10, §4.1 P5′p, §6.9 |
| **DEC-73** | **THE DATASET FACTORY'S PASS 1 IS MEASURED, AND THE FLAT 10^10 TARGET DOES NOT SURVIVE IT.** *Rationale: DEC-56 stated a data requirement without knowing whether it was reachable; pass 1 measured that, and four of seven faculties are one to three orders of magnitude short.* **Counts:** **159 candidates** in **106 provenance groups** — **105 VERIFIED, 26 CONTRADICTED** (the surveyor's verdict overturned at the primary), **18 UNVERIFIABLE, 10 REFUSED-CLOSED**; verdicts PERMISSIVE 42 / ATTRIBUTION 21 / SHARE_ALIKE 20 / NC 13 / **UNVERIFIED 37** / BLOCKING 11 / REFUSE 15. **The backlog is the 37 UNVERIFIED (23%), not the refused list.** **B1-bounded reach against 10^10 tokens per region:** **`visual` 0.3%**, **`memory` 1.3% clean-permissive / 5.0% NC-inclusive** (2.0% if MS MARCO is refused), **`language_code` 15%**, **`reasoning` 1.8%** permissive without a generator and **≥100% with** the DeepMind `mathematics_dataset` generator, **`moral_safety` 2.4%** (not a shortfall — DEC-59 curates), **`language_trunk` 6,359%**. **NC BUYS ALMOST NOTHING** over clean-permissive: **0 points** for `language_code`, `visual` and `moral_safety`, **0.8** for `reasoning`, **3.7** for `memory` of which **3.3 is MS MARCO alone**. **B1 — not licence tolerance — is what binds**, so the lever is more independent permissive provenance groups. **Consequence, and it amends DEC-56:** per-faculty targets with the reasoning written down, **clean-terms generators** (the cosmopedia shape: an Apache-weights generator run locally) and the operator's own **enrichment** replace one number applied to seven faculties. Row **P2′g**; the five legal readings become **OD-18** `[OP: csd-dataset-factory-pass1-result.md]` | §5.8, §4.1 P2′g, §6.1, §8 OD-18 |
| **DEC-74** | **THE FLEET IS ON DEMAND, AND THE 1080 Ti's TRAINING VERDICT IS MEASURED RATHER THAN ASSUMED.** *Rationale: two stale claims were live — that all three cards serve `:8080` continuously, and that the 1080 Ti could take a region "where the corpus and torch build allow" — and both cost probing time before they were checked.* **All three cards and all their runtimes are started on demand and stopped when idle** — LocalAI on akula-prime and gpu5080, llama.cpp on the 1080 Ti VM, Open WebUI and ComfyUI alike. **An idle `:8080`, a stopped unit or an exited container is the NORMAL RESTING STATE, not a fault**, and is not to be "fixed". **The 1080 Ti verdict, VERIFIED:** the fleet's pinned **`torch 2.11.0+cu128` cannot run there at all** — `get_arch_list()` floors at **sm_75** and the VM's driver **535.274** caps at CUDA 12.2, so there is no sm_61 target in that build — while a separate **`torch==2.5.1+cu121`** venv on the guest runs a 2048² matmul with zero warnings but carries none of CSD's other dependencies. **So fp32 training there is "with env X: yes", and the DEFAULT ROLE IS INFERENCE, RAG AND UTILITY**, which is what the packer defaults it to until a probe shows otherwise. DEC-54's *"and the 1080 Ti where the corpus and torch build allow"* is **narrowed by this measurement rather than left standing** `[OP: fleet-gpu-roles-and-scheduling.md, csd-training-placement-policy.md]` | §6.7, §4.1 W7p, §4.1 W7k, §6.1 |
| **DEC-75** | **TOOLING LIVES IN ITS OWN REPO, AND `tzervas/gpu-pack` IS ROW W7p's IMPLEMENTATION.** *Rationale: the operator's rule is explicit — every harness or framework built for this programme is captured in its own repo, aligned to its own role, so tooling never bloats the model repo — and W7p was written as a row without naming where its code would live.* **Rule:** before building a tool, decide which repo owns the role; if none does, create one. **CogSynDelta keeps the MODEL** — regions, interconnect, objectives, evals, corpus contracts, receipt formats, and the **thin adapters a tool needs** (an env var honoured, a job-spec file). **The tool's code, tests, docs and its own CI live in the tool's repo.** **`tzervas/gpu-pack` (main at `5364932`, rounds 1–4 merged) is W7p's implementation**: probe, budget ledger, admission under one flock, per-process cap, transient units with emitters, remote launch and launch receipts. **CSD's whole side is already in this tree and stays that size** — `src/cogsyndelta/util/gpu_budget.py` plus `program/jobs/*.json` — and **P10.4's "the probe/admit/launch pipeline is the remaining piece" is retired by measurement**. The same rule places the predictive trainer in `tritter` (DEC-72) and autodev in `csd-autodev` (DEC-60) `[OP: tooling-lives-in-its-own-repo.md]` | §6.7, §4.1 W7p, §4.1 W7k, §8 |
| **DEC-76** | **THE AUTODEV POINTER IS REV 5, AND THE RULE THAT CAME OUT OF FIVE ROUNDS IS RECORDED WITH IT.** *Rationale: DEC-60 made this document a pointer, and a pointer that names a superseded revision is worse than no pointer — a reader follows it and reads the wrong design.* The spec is **`docs/AUTODEV-IDENTITY-AND-SANDBOX.md`, rev 5, committed at `ece346b`** on branch `docs/autodev-spec` in `tzervas/csd-autodev` (**PR #1** there; rev 4 was `c192278`), with the threat pass and the two retained skeptic write-ups under `docs/evidence/autodev-threat-2026-09-03/` — that directory holds exactly three files, `01-threat.md`, `skeptic-round4.md` and `skeptic-final.md`; **five** revise-and-attack rounds were run and the earlier write-ups were not committed [V, `git log` on `docs/autodev-spec` at `ece346b`]. **Residual criticals, one line and no summary of the spec itself:** the **FINAL (round-5) skeptic, `skeptic-final.md`, still lists five critical and six high** (23 findings) — the number the source memory records; the **round-4** pass against rev 4, `skeptic-round4.md`, listed **four** critical and six high (21 findings) and is not the count meant here — of which **two are overtaken by events** (the `/api` auth gate is on `main`; the closure pin went stale because CSD moved twice in eight minutes) and the rest are **rev-6 material** — the runtime import recorder must run the protected tests under `pytest` rather than import their modules and needs a `⊇` floor so it can fail; P0a's check must model the **live** code, which reads `CSD_APPLY_TOKEN` from the unit's environment rather than forking credentials per request; `socket_ident` must be re-stat-ed per connect or a socket-unit restart never invalidates; and §0d must be pinned to a merge-base. **THE RULE, and it is the transferable part: FREEZE A CODE REVISION (a tag), THEN WRITE THE NEXT SPEC REVISION AGAINST IT, THEN IMPLEMENT.** A spec chasing a moving tree cannot converge, and five rounds is the evidence `[OP: autodev-service-identity-and-sandbox.md]` | §8, §4.1 PRE-1/2/3 |
| **DEC-77** | **THREE GOVERNANCE ADDITIONS, EACH FROM A MEASURED FAILURE THIS SESSION.** *Rationale: DEC-61 wrote the PR-only rule against CogSynDelta's `main` alone, and two CI failures this session were not one-off bugs but classes that will recur.* **(1) PR-ONLY EXTENDS TO `tzervas/gpu-pack`**, whose `main` is protected the same way — push whitelist = the operator, required context `CI / test (pull_request)` — so DEC-61 is a rule about every code repo and not a fact about one. **(2) THE SECRET SCAN HAS A PROSE FALSE-POSITIVE CLASS.** `gitleaks`' `generic-api-key` rule fires on low-entropy `identifier=value` pairs **inside prose**: a tokenizer-load log line, and `decorr_weight=0.0` quoted in a docstring that reports a control arm's three loss weights. **The remedy is the NARROWEST allowlist the finding admits — always `targetRules` limited to the one rule and always the byte-identical phrase, plus the exact file path WHERE PATH SCOPING IS POSSIBLE — and it is verified by PLANTING a real-shaped secret in the same file and asserting it is still caught**. *Path scoping is not always possible and the repo's own config shows why: the tokenizer-line entry is phrase-and-rule-scoped with no `paths`, because the same string is quoted in `.gitleaksignore`'s immutable history and a path-scoped entry could not reach it. The rule is therefore `rule + phrase` always, `+ path` whenever the finding is confined to one file*, because an allowlist nobody tried to overreach is an allowlist nobody has measured. **(3) THE CI RUNNER HAS NO GPU TOOLING.** `fleet-ci-base:1` has no `nvidia-smi`, so a helper-CLI wrapper that lets `subprocess.run` raise takes down the whole endpoint that called it — `FileNotFoundError` inside a status handler's dict literal, aborting a 200 that should have degraded one field. **Tests must not assume `nvidia-smi` (or `docker`, or `ssh`) exists**, and a diagnostic subprocess must **fail closed to a string**. The regression test is the CI scenario itself: a `PATH` with no GPU tooling on it at all `[OP: branch-and-pr-to-main-only.md, tooling-lives-in-its-own-repo.md]` | §8, §9.9 B5 |
| **DEC-78** | **THE FACULTY NAMING RULE: BY COGNITIVE FUNCTION, NEVER BY DOMAIN OR ANATOMY.** *Rationale: DEC-01 named the renamed `code` region `language_code`, folding its current corpus/battery domain into the id itself — exactly the pattern the operator's rule now forbids.* Operator ruling: *"regions are named by cognitive faculty and the function they provide, in accurate synthetic terms — never by knowledge domain (`code`) and never by colloquial anatomy. The region formerly called `code` is the LANGUAGE CENTRE: id `language`, with `specialisation: \"code\"` recorded as metadata"* `[OP, 2026-09-04]`. **Domains live as memory-gate personas (overlays) or as variant branches of the same Hub repo, never as region names.** **AMENDS DEC-01 IN PLACE**: canonical id `language`, `code` demoted from id to specialisation metadata; **DEC-03's `vl_latent` → `visual`** already satisfies the rule and needs no change. **Compatibility is unaffected**: everything on disk and on the Hub — receipts, matrix cell directories, variant ids, Hub branches/tags, corpus fingerprints and licence tiers — keeps the legacy names, resolved through the single alias module `cogsyndelta.regions.aliases` (`REGION_ALIASES = {"code": "language", "vl_latent": "visual"}`), which is the ONLY place the mapping lives; every reader resolves a region name through it before comparing, dispatching, or looking up config on it. **Disclosure of a resolved legacy name is a per-reader choice, not a second invariant this rule imposes on every reader alike**: `MindSpec.from_dict` (the interconnect ingest path) records it as a `region_alias_of` field with a `DeprecationWarning`; `cogsyndelta.cards`' Jinja templates print a **Faculty** line naming the canonical id and the legacy id the cell's receipts were recorded under; `scripts/csd-publish-checkpoint.py`'s hand-built card adds no such field, because its shape is pinned byte-for-byte against every already-published `tzervas/cogsyndelta-region-*` README (changing it changes what is already live on the Hub — see `docs/technical/model-card-pipeline.md`). **A writer that must keep an in-flight run's on-disk history stable does not canonicalize either**: `scripts/csd-train-all.py` deliberately writes `PretrainConfig.region`/the receipt `region` field as the caller's own spelling, because `cogsyndelta.regions.pretrain.pretrain_region` derives the checkpoint directory and receipt filename from that value, and canonicalizing it mid-run would split one region's history across two directories | §1, §1.2, §1.4 |

**The index is in numeric order** `[S31-17 fixed]`. Revision 3.1 appended DEC-43 to DEC-46 *before*
the DEC-42 row revision 3 had added, so the table read DEC-41, 43, 44, 45, 46, 42. It is the
document's lookup table and a lookup table out of order is a small defect that costs a reader the
one thing the table exists to give them. **Revision 3.3 adds DEC-49, DEC-50 and DEC-51 in numeric
order**, which for once is also append order; the ordering was re-checked rather than assumed. **Revision 3.4
appends DEC-52 to DEC-67, also in numeric order, and renumbers nothing** — DEC-32's supersession by DEC-49 is
the pattern: a decision that stops being true is amended in place and keeps its number. **Revision 3.5 appends DEC-68 to DEC-77, also in numeric order, and renumbers nothing**; DEC-56 and DEC-63 are amended in place by DEC-73 and DEC-70 and keep their numbers and their original text. **Revision 3.6 appends DEC-78** and amends DEC-01 and DEC-44 in place, keeping their numbers and their original text with the amendment cited alongside it.

---

# 1. Decision summary

## 1.1 The test applied

The operator's test, applied literally: **what faculty does this provide?** — never *what
dataset does it train on*. A label space is not a faculty. A training objective is not a
faculty. A mechanism available to every region is not a faculty; it is a primitive.

One systems test is added, because the operator's framing implies it: **a region is a unit of
independent training, independent quantisation, and independent residency.** If two things must
always be resident together and always train together, they are one region with two heads.

## 1.2 The ten verdicts

Legend: **KEEP** · **MERGE** · **PROBE** (checkpoint retained as a fixed instrument, not a
region, not deployed) · **RETIRE** · **PRIMITIVE** (survives as a mechanism available to any
region) · **BUILD** (new).

| # | current name | verdict | faculty | evidence / note |
|---|---|---|---|---|
| 1 | `code` | **KEEP, RENAME → `language`** | language, code specialisation | Strongest receipt in the tree: r@1 **0.9766** vs untrained 0.2285 [V\*]. Recorded as `faculty: "language"` + `specialisation: "code"` so the verdict is machine-readable without asserting a language trunk into existence. **Target name corrected from `language_code` to `language` in revision 3.6 (DEC-78).** |
| 2 | `retrieve` | **MERGE → `memory`** | hippocampus | Contributes the *retrieval* operation. r@1 0.7480 on a 512-pair diagonal, against an untrained baseline of **exactly 0.0000** — a number that should have raised an eyebrow and did not [V\*]. **W2c re-measures that baseline before 0.7480 is used as a threshold** `[A13 fixed]`. **Licence: `retrieve` carries GooAQ's NC term** (77.8% of its corpus), so `memory` is **NC**, and so is anything composed from it (DEC-31) `[OP: csd-release-licence-decision]`. |
| 3 | `compress` | **MERGE → `memory`** | hippocampus | Contributes the *consolidation* operation. r@1 0.7070 [V\*]. Consolidation and retrieval are two operations of one faculty. Licence: whatever CC BY-SA requires if the SA pairs are used — subsumed by `memory`'s NC term. |
| 4 | `vl_latent` | **KEEP, RENAME → `visual`** | visual cortex | Correct as-is per the operator. probe top1 0.0606 vs 0.0335, transfer 0.2625 vs 0.1535 [V\*]. **Those numbers are the EMA target encoder, so the target encoder is the deployed half** (DEC-34) `[A3 fixed]`. Its corpus is unreleasable and its input resolution cannot see a composite (DEC-34, W7v) `[A4 fixed]`. |
| 5 | `reason` | **KEEP, RENAME → `reasoning`** | reasoning / logic | r@1 0.0801 vs 0.0039, **receipt deleted, prose only** [V\*]. Its receipt must be regenerated before it is frozen (§4, W1b). |
| 6 | `classify_banking77` | **PROBE** | none | 77 banking intents is a label space. Retained frozen as a fast fine-grained-intent readout used as a *damage detector* when objectives change. |
| 7 | `classify_go_emotions` | **PROBE of a new `affect` faculty** | affect (declared, not built) | Affect is a real faculty; a 28-label BCE head over Reddit comments is a probe of it. Also fails balance rule B5 on both counts — 32.76% max label share, 184.7:1 max:min [V]. |
| 8 | `residual_mlp` | **RETIRE** | none | Its own docstring: *"Paired with LatentVAE so softmax routing is a real choice, not a stub"* [V\*]. It exists to give a router two things to choose between. Its declared corpus is `"synthetic"` and its trigger is the fallback path [V]. The residual path survives as `x + f(x)` inside every block. |
| 9 | `stream_vae` | **RETIRE as region → PRIMITIVE: tract codec** | none | A bottleneck codec, declared corpus `"synthetic"` [V]. Promoted: per-tract bandwidth can be spent as *fewer tokens* (`b_r`) or *lower rank per token* (a codec on the tract). It becomes `interconnect.primitives[]`. |
| 10 | untracked `regions/retrieve.py` | **SPLIT** | — | Not a region. Its BEIR-style eval against the real 57,638-passage FiQA pool with a BM25 reference **becomes `memory`'s gate**; its FiQA-only training regime (14,131 pairs, single source) is **retired** — single-source training is what produced every existing B1 failure. Closes operator decision **D6**. |
| +1 | — | **BUILD: `episodic_store`** (revision 1 said BUILD, revision 2 demoted it, **revision 3.3 reinstates it by operator ruling**) | AI-specific | A non-parametric, writable key/value store over workspace latents. **DEC-49 makes it REQUIRED and BUILT IN PHASE 2** `[OP: csd-episodic-store-required.md]`, with the partition / capacity / eviction / lifecycle / integration contract taken from `memory-gate` and `memory-gate-rs` clause by clause and the residue listed as open gaps rather than invented (§8). **Consequence: R = 5 participants, 10 ablation pairs**, and A17's defect is closed the only way that keeps the count honest — by **declaring the store's item shapes (X7, X8)**, not by excluding it. Rows **E0/E1/E2**. See §1.3. |

**Why #10 splits rather than winning outright.** Its own docstring argues, correctly, that
*"recall@10 out of 512 and recall@10 out of 57,638 are different measurements that happen to
share a name"* and that FiQA judges ~2.6 passages relevant per question, so the diagonal scores
a *different correct answer* as wrong [V\*]. That argument is sound and the eval is strictly
better. But its training regime is worse on balance rule B2, and — decisively for this document
— **the 512-pair diagonal is saturated** (`code` 0.9766, `retrieve` 0.7480). A saturated
instrument has no headroom for an interconnect to occupy, so it cannot measure composition [I].

## 1.3 New and placeholder faculties

| faculty | verdict | interface / dependency |
|---|---|---|
| **white matter** | **BUILD NOW** | §2. It is the centre of gravity, and it is late *by dependency*, not by priority. Its dependencies now exist. |
| **frontal cortex** | **BUILD NOW**, inside white matter, with its own objective | §3.1. |
| **thalamus** | **BUILD NOW** as afferent-bandwidth gating; content gating is a declared seam | §3.2. |
| **`episodic_store`** (AI-specific) | **DEC-49 — REQUIRED, BUILT IN PHASE 2, a workspace participant with its own budget `b_store`. This supersedes DEC-32 and reinstates DEC-10's shape** | **The ruling, verbatim** `[OP: csd-episodic-store-required.md]`: *"No. So that will have to have the episodic store created to back it rather than just dropping it. It should have partition, capacity, integration contract, and you can look at the memory-gate and memory-gate-rs repos to get an idea of what is intended there because I should have at least in one of those two the contracts for eviction and everything. Capacity is gonna be dynamic based on the GPU that it's running on and how much context is allocated for — like KV cache determining essentially how much free space there is that can be allocated to memory-gate functionality."* **What revision 2 got right and DEC-49 keeps:** a region shipped `status: built` with no write policy, no signal, no receipt and no gate is the `residual_mlp` defect (DEC-07, DEC-08), and DEC-32 was right to refuse it `[A17 fixed]`. **What revision 2 got wrong:** it answered *"unspecified"* with *"deferred"*, when the specification already existed in another of the operator's repositories. **So the store is not deferred; it is specified from the repos and then built.** **PROVENANCE OF EVERY `[V]` IN THIS CELL.** These are `file:line` reads by the contract-extraction scout in **`memory-gate` at `2c11c3f`** (worktree `python-ai/memory-gate-wt-p1-09`, branch `feat/gateway-retrieve-domain`, the strict superset of the P1-01→P1-09 stack) and **`memory-gate-rs` at `4b9f60d`**. They are reads of **another tree, not this one**, and the repo plus HEAD is named so each is re-checkable; the extraction is held at `scratchpad/memgate/memory-gate-contracts.md`. **Where those repos are silent this cell says SILENT and stops** — §8's *Episodic store contract gaps* block carries the six open questions, and nothing there is invented into a contract clause here. **THE CONTRACT, clause by clause.** **(1) Partition — VERIFIED, and it is identity, not a filter.** The unit of storage is keyed by the **composite** `(domain, logical_key)`; a bare logical key is explicitly rejected as a design option [V, `memory_protocols.py:123-125`; `adr/0001:129-133`]. A query **must** name a domain or pass an explicit `global_query=True`, and omitting both raises `MemoryValidationError` [V, `memory_protocols.py:194-217`; `record_validation.py:83-97`]. Cross-domain isolation is **test-pinned**: `test_domain_isolation_same_logical_key` requires the same logical key in two domains to yield two distinct records and never cross-leak on read [V, `tests/storage/test_store_conformance.py:150`]. In Rust the axis is a **closed enum**, not a free string [V, `types.rs:165-208`]. **CSD adopts this as the base axis and adds one segment ahead of it** — `(scope, domain, logical_key)` — where `scope` is the server-derived partition of §9.9 B2. **The axis beyond domain is a GAP, not a decision** (§8 gap (b)). **(2) Eviction — VERIFIED, importance-scored with a GPU-residency bonus and tie-breaks.** `pick_spill_victim` scores each candidate `score = importance`, **`+1.0` if the span is `Residency::Gpu`-tagged**, ties broken by **older timestamp** then lexicographically by key, and the lowest score spills first [V, `storage/tiered.rs:150-178`, spill at `:131-148`]. Overflow at the lower tier **hard-deletes** the lowest-importance rows — the only data-loss path in the store [V, `tiered.rs:180-198`]. Promotion exists and is caller-invoked [V, `tiered.rs:113-121`]. A separate store applies **multiplicative importance decay** as an explicit externally-called step [V, `vsa/holographic_store.rs:314-319`, prune at `:321-338`]. **The Python tier is pure LRU by write order and reads `importance` nowhere** [V, `storage/tiered.py:40-46, 72-73, 145-150`] — the two languages disagree, and **CSD takes the Rust scheme**, because an unscored LRU cannot express the GPU-residency bonus the operator's ruling depends on. **Staleness is a GAP with a strong hint**: the recency fields `access_count` and `last_accessed` are **recorded and never read by eviction** [V, `holographic_store.rs:113-116, 321-338`], which is unfinished wiring rather than a decision `[I]` — so CSD's `score = importance + gpu_resident_bonus − staleness_penalty(last_accessed)` is **designed fresh from an INFERRED intent, not a confirmed one** — the scout classifies this as INFERRED, not VERIFIED (`memory-gate-contracts.md:245-250`), so the `[I]` tag on this clause is load-bearing, not decorative `[S33-4 fixed]`. **§8 gap (a) is EXTENDED to cover the decay function itself, not only the byte unit it evicts to** — a reader who follows the pointer expecting only a capacity question also finds the open staleness question there, stated as such, rather than the requirement arriving with no gap covering it at all. **Test-pinned behaviours to port:** `admit_spills_lowest_importance`, `gpu_hint_protects_from_spill`, `promote_returns_span_to_ram`, `retrieve_merges_tiers`, `disk_prune_drops_lowest` [V, `storage/tiered.rs:292-353`]. **(3) Lifecycle — VERIFIED, and richer than the Rust side.** `start()` binds the store's optional start hook and begins accepting [V, `memory_gateway.py:147-176`]; `learn()` **awaits persist** and returns a `LearnReceipt` that is frozen with `committed: Literal[True]`, so a receipt cannot exist in a non-committed state and there is no droppable fire-and-forget handle [V, `memory_gateway.py:250-317`, receipt at `:35-59`]; `drain()` awaits in-flight learns to a terminal state **without cancelling them** [V, `:177-196`]; `flush()` invokes the backend durability barrier [V, `:198-214`]; `stop()` is **refuse-new → drain → flush → stopped**, re-raising any drain or flush failure as `MemoryDurabilityError` [V, `:216-248`]. **Backpressure is bounded and it refuses rather than queues:** `max_in_flight = 32`, and exceeding it raises `MemoryBackendUnavailableError` **synchronously at submit time** rather than spawning unboundedly [V, `memory_gateway.py:118-133, 361-374`]. **CSD adopts all six verbs and the refusing backpressure unchanged**; a store that queues instead of refusing is a second unbounded object, which is the thing §9.9 B2 exists to prevent. **(4) Durability oracles — VERIFIED, three of them, with different standing.** **In-memory is a conformance oracle and explicitly NOT a durability claim** [V, `spec.md:355-362`]. **SQLite is the local durability oracle**: `journal_mode=WAL` with `synchronous=FULL` **enforced at connect time, raising if it is not FULL** [V, `storage/sqlite.py:599-603`], and `flush()` is an *additional* `PRAGMA wal_checkpoint(FULL)` that the ack does not require [V, `sqlite.py:391, 405-420`]. **Qdrant acks only after the transport upsert returns with `wait=True`** [V, ADR-0003 point 5], and the collection is stamped with embedding model / dimension / metric / namespace / revision and **validated before any user upsert** [V, `storage/qdrant.py:295-303`] — two claims under one citation in the prior revision, correctly split now that each is attributed to what actually supports it `[S33-10 fixed]`. **Ordering is durable-first:** the durable commit precedes the hot placement, so eviction from hot changes residency and never loses an acked write [V, `storage/tiered.py:67-74`]. **CSD adopts the three-oracle ladder as its test matrix** — the same conformance suite parameterised over in-memory, SQLite and the Qdrant fake — and the durable-first ordering as a hard rule. **(5) Retrieval-into-context integration — VERIFIED, and it is the ONLY integration shape that exists.** Before a task, `retrieve_context(query, domain_filter)` returns records that are built into a plain `enhanced_context` mapping; after the task, `learn_from_interaction(...)` writes back [V, `agent_interface.py:75-183, 184-224`]. **No attention-hook and no KV-injection integration exists in either repository** [V-abs, exhaustive read of both `src/` trees]. **This is the clause CSD diverges from and the divergence is named, not hidden:** in CSD the store is read **as a workspace participant through the cross-attention**, not injected into a prompt, because DEC-47 forbids re-serialising to tokens on an inter-region path. The repos supply the **lifecycle and the partition**, not the attachment; the attachment is this document's (§2.3, §2.4). **(6) Index, not bytes — VERIFIED, and it is the shape that keeps the store out of the KV allocator's way.** `Residency::{Gpu,Ram,Disk}` is a **metadata tag on a span**, documented as *"Where a recalled span currently lives. GPU bytes stay in llama.cpp"* [V, `types.rs:414-436`; `storage/tiered.rs:1-8`; `facade/hypha.rs:1-8`]; `mark_gpu(key)` records residency and **moves no bytes** [V, `tiered.rs:99-106`]. **CSD adopts the split**: the store owns an **index and a policy** over latent bytes owned by the runtime, not a second allocator competing for the same VRAM. **What is stored is a GAP** (§8 gap (c)). **(7) Capacity — the repos are SILENT on what the operator asked for, and this is stated rather than papered over.** Every cap in both repos is an **item count**, never bytes and never VRAM: `hot_cap` default 256 [V, `storage/tiered.py:40-51`], `TierBudget{ram_max_items 256, disk_max_items 4096}` [V, `types.rs:458-473`], `max_traces` default 100,000 [V, `holographic_store.rs:33-58`]. **The only VRAM arithmetic in either repo budgets model WEIGHTS, not store capacity**, and it is never wired to any cap: `weight_budget_mib(vram_total_mib, reserve_gpu_kv) = vram_total_mib − display_reserve(2048) − cuda_scratch(1024) − [gpu_kv_reserve (2048)]`, `saturating_sub` to floor at zero, test-pinned at 16,303 MiB ⇒ 11,183 / 13,231 MiB [V, `facade/hypha.rs:114-122`, constants `:13-26`, test `:152-157`]. **So the operator's dynamic capacity has no implementation, no description and not even a TODO in either repo — only a subtractive-reserves SHAPE to borrow.** §8 gap (a) carries the formula and the recommended default; it is **not** asserted as a contract clause here. **WHAT THIS COSTS AND WHAT IT BUYS, in this document's own units.** Parametric cost is **two 512×512 projections `W_k`, `W_v` = 524,288 params** (§2.3) — the store itself is data. It takes a **budget `b_store` on the workspace read simplex**, floored at `η/R = 3.0%` of `B_read = 256`, i.e. **≥ 8 read tokens**, default 32, max 256 (§2.4, §1.4). **`b_store` bounds what is READ; the byte capacity of §8 gap (a) bounds what is STORED; they are different currencies and a receipt that prints one as the other is wrong.** What it buys: **the store IS the overarching context that the per-region sliding windows slide over** (§6.8), which at `R = 4` was a within-request claim only. **WHY IT IS AI-SPECIFIC, unchanged:** the hippocampus does pattern separation over a *parametric* trace because biology cannot keep the tensor; an AI can just keep the tensor. **THE BUILD IS GATED BY THE THREAT MODEL, NOT EXCUSED BY IT** `[A28 fixed] [T2 fixed]`. It remains the **only writable, cross-request object in the architecture**. §9.9 B2's four findings — partition, capacity bound, eviction order, cross-request poisoning fuzz — are the **acceptance gates of rows E0/E1/E2** (§4.1). A store that cannot demonstrate them does not ship, and that is a different sentence from the one DEC-32 wrote, which was that a store that cannot demonstrate them is not built. |
| **`salience`** (AI-specific / limbic) | **DECLARE the interface; do not train in v1** | The scheduler needs a PRIORITY term; biology supplies priority from valuation; the tree already holds a valuation-adjacent checkpoint (`classify_go_emotions`). `salience.value(emission, control_state) → [B,1]`. Its consumer is the priority ordering in §2.4. This is the one place the brain analogy pays a *mechanical* dividend rather than a naming one. |
| **language (trunk)** | **PLACEHOLDER, interface declared** | Created when a *second* specialisation exists, so "trunk + adapters" is a measurable claim rather than a rename. Its generative half is `model/causal_lm.py` (283 lines, exists) [V\*] and needs foundation corpora, which are curriculum **step 4** by three independent declarations [V]. |
| **numeric / math** | **PLACEHOLDER, interface declared** | Blocked on corpus, not design: gsm8k and aqua_rat are both spent (§5.1). `reasoning`'s current corpus straddles this faculty; splitting today starves both halves. Natural source is the reserve's executable items, whose test cases carry exact numeric ground truth. Emits tokens over **values, not BPE pieces** — the clearest cheap divergence from the brain. |
| **auditory** | **DECLARED SEAM — DEC-48 defers it to a PRODUCTION PHASE. This supersedes revision 3.1's "REQUIRED, BUILD IN PHASE 1"** | **The governing ruling is the later of two the operator gave on 2026-09-02, and it is quoted first because it is the one that decides the row** `[OP: csd-multimodal-io-intent.md]`: *"we can wait to add audio as a future feature once it proves out without audio. that will be more of a production phase implementation"*, with *"gotta walk before we run"*. **So: everything in this cell stays written and nothing is scheduled.** `auditory` is neither a phase-1 nor a phase-2 row, it is **not a participant**, `R` stays **4**, and §5.4's six ablation pairs are unchanged `[S31-1 fixed]`. Rows **A0–A3** live in §4.1's *Production phase: audio* subsection, `blocked_by` **the composed mind passing W6 without audio plus an explicit operator go**. Keeping the interface, the objective family and the audit declared is the entire point of a seam: **nothing has to be redesigned when it is picked up**, and the earlier ruling below is retained because it is why the seam is specified to this depth rather than merely named. The completed audit and the A0 catalogue work are **groundwork already banked**, not sunk cost. **The earlier ruling, retained:** the operator's intent is a ruling, not an inference, and it is quoted in full because it settles two rows of this table at once `[OP: csd-multimodal-io-intent.md]`: *"the intent for this model is for me to interact via voice and/or text input and text/speech output, and for it to take as input, audio, visual, discrete text/token and output multimodal as well."* **Interface, unchanged from revision 3 and now load-bearing rather than decorative:** byte-identical to `visual`'s — `tokens(inputs, context_tokens) → (h, mask)` over **patch tokens on spectrogram frames** (log-mel frames tiled the way `visual` tiles an image), so the adapter, both budget currencies (`ctx_r`, `b_r`), the tract codec and the whole §2.2 contract apply unchanged, and the workspace being modality-general becomes a **measured** claim rather than an asserted one. **Objective family:** masked-latent prediction over spectrogram patches — the JEPA family `visual` already uses, so the harness, the untrained-baseline discipline and the region-specific-seed rule carry over intact. A supervised ASR head is explicitly **not** the v1 objective: the faculty being built is hearing, not transcription, and an ASR objective would make the region a label space, which is what DEC-05 retires regions for. **Corpus was the blocker and the audit removed it, which is why the seam is credible:** the audit OD-9 asked for is **done** and committed at `docs/design/AUDIO-CORPUS-AUDIT.md` — **seven provenance groups, of which five carry no NC term** `[S31-13 fixed]` — and the recommended v1 mix touches **no BLOCKING source**. **Rows: A0** (corpus manifests + fetch) and **A1** (pretrain), both **deferred**, §4.1. **A bounded first audio fetch may already exist on `gpu5080:/bulk`;** it is **re-acquirable scratch**, not an artefact anything depends on, and nothing in this document treats it as evidence. **The `s_r` consequence revision 3.1 recorded here is now moot:** the region set frozen before admission is the four-region set, because `auditory` is not scheduled before W6. |
| **`speech_output`** (a head on the language centre, not a faculty, and **not a participant**) | **DECLARED SEAM — DEC-48 defers it with `auditory`. This supersedes revision 3.1's "REQUIRED"** | **Deferred by the same later ruling** `[OP: csd-multimodal-io-intent.md]` — *"more of a production phase implementation"*. **What is kept as declared, and this is the load-bearing half:** the `Schedule`'s **output-modality field and §2.5's whole `output` block stay in the runtime contract now**, so the contract does not change when the head is built; the frontal read-out is specified as multi-head from the start. **It adds no participant and no ablation pair** — it is a head on a trunk, so `R` is untouched by it in either direction `[S31-19 fixed]`. **Row A2 is deferred** to §4.1's *Production phase: audio*. **The earlier ruling, retained:** output is *"text/speech"* `[OP: csd-multimodal-io-intent.md]`. **Interface:** the generative half — `model/causal_lm.py`, 283 lines, already exists [V\*] — emits **two streams from one trunk**: BPE text tokens, and a **discrete speech-token stream that a synthesiser consumes** (neural-audio-codec code indices). **The synthesiser is a separate, swappable component and is explicitly NOT a region**, for the same reason a tokenizer is not one. **Why a head and not a faculty:** the systems test in §1.1 — two things that must always be resident together and always train together are one region with two heads. That is exactly the shape DEC-02 gives `memory`, and applying the same test twice is what makes it a test. **Objective family:** next-token cross-entropy on both streams plus a text↔speech alignment term; **the gate is intelligibility measured against a fixed reference ASR, not loss** (§4.1 A2), because a loss curve on a token stream nobody has listened to is the audio analogue of the saturated 512-pair diagonal. **Licence:** the clean tier is **CC BY 4.0** — not "MIT / CC BY", because CC BY is the strictest input in the mix and MIT is unreachable `[S31-6 fixed]`; Expresso buys expressive style at an NC term that costs nothing at the composed tier (DEC-45). **Row: A2, deferred.** **Image output is a DECLARED SEAM, not built** — *"output multimodal as well"* is stated intent, the seam is the read-out's output-modality field in §2.5, and the validator refuses `image` until a faculty exists that produces it. Declaring the seam is free; building a head with no faculty behind it is the `residual_mlp` defect. |
| **affect** | **PLACEHOLDER, unusual shape**: `emits: "global"` | A `[B,8]` gain vector broadcast onto every tract — a multiplicative modulation of `a_r`, **not** a participant in the attention. It is the one faculty that acts *on* the interconnect rather than *through* it. |

**Explicitly NOT proposed:** a motor cortex. Tool invocation is an output format of the frontal
read-out, not a faculty — and this is now an **operator ruling, not an inference**: a motor
centre is *"wholly unnecessary until likely post the Rust reimplementation"*, and a proposal
that includes one for tool-use or action emission is to be struck with the deferral noted
`[OP: csd-regions-are-brain-faculties]`. This document already excluded it; the ruling is cited
so the exclusion is not re-litigated. **Also explicitly not a region:** the schedule trace. A
serialisation of a graph the interconnect already computed is a log; declaring a log a region in
the AI-specific slot is the exact category error the operator's test exists to prevent [I].

## 1.4 The JSON diff for `config/mind/csd-regions.json` — REVIEW ONLY, DO NOT APPLY

Current file: `stream_dim: 512`, `top_k: 1`, `aux_coef: 1.0`, nine entries in `regions[]`, of
which the only two with `live: true` are the two nobody trained [V].

```diff
--- a/config/mind/csd-regions.json
+++ b/config/mind/csd-regions.json
@@
-  "stream_dim": 512,
-  "top_k": 1,
-  "aux_coef": 1.0,
-  "notes": "Region catalogue from docs/program/CSD-BRAIN-REGIONS.md. `live` reflects what is
-            IMPLEMENTED, not what is intended; ... Curriculum order is mandatory: per-region
-            pretrain -> router -> assembled mind -> foundation. ...",
+  "schema": "csd-mind/v2",
+  "workspace_dim": 512,
+  "latents": 64,
+  "workspace_iters": 4,
+  "notes": "Region catalogue from docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md.
+            CSD-BRAIN-REGIONS.md is RETIRED (see that document, DEC-11).
+            `status` replaces `live`: built | probe | placeholder | primitive | retired.
+            There is no top_k and no aux_coef: routing is not a discrete selection, it is the
+            cross-attention mass a region's tokens receive. Curriculum order is mandatory:
+            (1) per-region pretrain -> (2) interconnect training -> (3) whole-mind dynamic
+            training -> (4) foundation. Foundation corpora are step 4, not step 1.",
+  "interconnect": {
+    "kind": "perceiver_workspace",
+    "workspace_dim": 512, "latents": 64, "depth": 4, "heads": 8,
+    "budget_total_read_tokens": 256,
+    "budget_total_kv_bytes": 3221225472,
+    "floor_eta": 0.15,
+    "primitives": [
+      {"name": "tract_codec", "was": "stream_vae", "kind": "latent_vae", "latent_dim": 64,
+       "status": "placeholder",
+       "build_trigger": "Sum_r b_r saturates B_read on >20% of eval items, AND a measured
+                         gate shows lower rank per token beating fewer tokens at equal bytes.",
+       "role": "Rate limiter on a tract. Bandwidth may be spent as fewer tokens (b_r) or as
+                lower rank per token (this codec). Available on any pathway; not a region.
+                PLACEHOLDER: it inherits stream_vae's defect -- declared corpus synthetic, no
+                receipt, no training path -- and a primitive with no consumer is the `live:
+                true` defect in a new costume, so it does not ship as live. [A33 fixed]"}
+    ]
+  },
+  "probes": [
+    {"name": "classify_banking77", "probes": "language", "frozen": true,
+     "role": "PROBE, not a region. Fine-grained intent readout (77 banking intents), used as a
+              fixed instrument to detect damage when an objective changes. A label space is not
+              a faculty."},
+    {"name": "classify_go_emotions", "probes": "affect", "frozen": true,
+     "role": "PROBE of the affect faculty. macro-AP 0.3642, chance 0.0466. Fails balance rule
+              B5 on both counts (32.76% max label share, 184.7:1); retained as an instrument,
+              never as a capability claim."}
+  ],
   "regions": [
-    { "name": "residual_mlp", "kind": "residual_mlp", "live": true, ... },
-    { "name": "stream_vae",   "kind": "latent_vae",   "live": true, ... },
-    { "name": "code",         "kind": "contrastive_encoder", ... },
-    { "name": "retrieve",     "kind": "contrastive_encoder", ... },
-    { "name": "compress",     "kind": "latent_vae", ... },
-    { "name": "vl_latent",    "kind": "jepa_predictor", ... },
-    { "name": "reason",       "kind": "contrastive_encoder", ... },
-    { "name": "classify_banking77",   "kind": "classifier_head", ... },
-    { "name": "classify_go_emotions", "kind": "classifier_head", ... }
+    {
+      "name": "language",
+      "faculty": "language", "specialisation": "code",
+      "kind": "contrastive_encoder", "modality": "text",
+      "token_dim": 256, "pooled_dim": 256, "emits": "tokens", "accepts_condition": true,
+      "adapter": {"in_dim": 256, "out_dim": 512},
+      "token_budget": {"min": 1, "default": 64, "max": 96},
+      "kv_bytes_per_token": 4096,
+      "heads": ["text", "speech"],
+      "role": "LANGUAGE faculty, code specialisation. The `speech` head is DECLARED, NOT BUILT
+               (DEC-44, deferred by DEC-48); it is a head on this trunk and never a separate
+               region entry, so no participant count can pick it up. Places a natural-language intent and the
+               program that satisfies it at the same point in the shared space. It is a domain
+               adapter on the language centre, not a peer of it: when a second specialisation
+               exists the trunk is shared and this becomes an adapter family. Explicitly NOT
+               next-token over all of GitHub.",
+      "status": "built"
+    },
+    {
+      "name": "memory",
+      "faculty": "hippocampus",
+      "kind": "contrastive_encoder", "modality": "text",
+      "token_dim": 256, "pooled_dim": 256, "emits": "tokens", "accepts_condition": true,
+      "adapter": {"in_dim": 256, "out_dim": 512},
+      "token_budget": {"min": 1, "default": 64, "max": 96},
+      "kv_bytes_per_token": 4096,
+      "heads": ["retrieve", "consolidate"],
+      "role": "HIPPOCAMPAL MEMORY faculty. One trunk, three operations: ENCODE (text to a
+               trace), CONSOLIDATE (equivalent meanings converge under a shortened code -- the
+               former `compress`), RETRIEVE (a cue ranks traces -- the former `retrieve`).
+               Consolidation and retrieval are two operations of one faculty, not two regions.
+               Gated on the real 57,638-passage FiQA pool with BM25 reported alongside, never
+               on a 512-pair in-mixture diagonal.",
+      "status": "built"
+    },
+    {
+      "name": "reasoning",
+      "faculty": "reasoning",
+      "kind": "contrastive_encoder", "modality": "text",
+      "token_dim": 256, "pooled_dim": 256, "emits": "tokens", "accepts_condition": true,
+      "adapter": {"in_dim": 256, "out_dim": 512},
+      "token_budget": {"min": 1, "default": 128, "max": 256},
+      "kv_bytes_per_token": 4096,
+      "role": "REASONING / LOGIC faculty. Relates a problem to the STRUCTURE of its derivation.
+               HONEST LIMIT, unchanged: it recognises derivations, it does not produce them.
+               Phase 2 keeps this shape. Phase 3 replaces the objective with latent-step
+               prediction so the region contributes COMPUTATION to the workspace rather than a
+               lookup. Its receipt was deleted and must be regenerated before it is frozen.",
+      "status": "built"
+    },
+    {
+      "name": "visual",
+      "faculty": "visual_cortex",
+      "kind": "jepa_predictor", "modality": "image",
+      "token_dim": 384, "pooled_dim": 384, "emits": "tokens", "accepts_condition": true,
+      "adapter": {"in_dim": 384, "out_dim": 512},
+      "image_size": 64, "patch_size": 8, "n_patches": 64,
+      "token_budget": {"min": 1, "default": 64, "max": 64},
+      "kv_bytes_per_token": 9216,
+      "quantization": {"bits": 8, "method": "calibrated-uniform", "skip": true},
+      "deployed_half": "target_encoder",
+      "role": "VISUAL CORTEX faculty. Latent visual reasoning: predicts representations of
+               unseen regions of a scene from visible ones, never pixels. Emits its patch-token
+               sequence into white matter directly; it is the reference implementation of the
+               token contract because its output was already a sequence.
+               DEPLOYED HALF IS THE EMA TARGET ENCODER (10,712,448 params) -- DEC-34.
+               `visual.tokens(x) = target_encoder.forward(x) -> [B, n_patches, 384]`;
+               `visual.pool(h) = h.mean(1)`, which is exactly `IJEPA.encode`.
+               image_size / patch_size / n_patches are REBUILT BY W7v; every field in this
+               block that mentions them is provisional until that row lands.
+               W3r FIXES THE TARGET THESE FIELDS ARE REBUILT TO: image_size 128, patch_size 8,
+               n_patches 256 -- the composite frame geometry, single source of truth at
+               `src/cogsyndelta/vl/composite.py` (`FRAME_SIZE`, `PATCH_SIZE`, `N_PATCHES`),
+               checked in with a byte-identical 128x128 fixture at
+               `tests/fixtures/composite_4panel.png` (spec at
+               `tests/fixtures/composite_4panel.spec.json`). W7v imports these constants rather than
+               re-deriving them; this block's own 64/64 values stay provisional until W7v lands.
+               Corpus is UNRELEASABLE as trained.",
+      "status": "built"
+    },
+    {
+      "name": "episodic_store",
+      "faculty": "ai_specific_episodic",
+      "kind": "nonparametric_store", "modality": "latent",
+      "token_dim": 512, "pooled_dim": 512, "emits": "tokens", "accepts_condition": false,
+      "token_budget": {"min": 8, "default": 32, "max": 256},
+      "phase": "phase2",
+      "capacity_bytes": "dynamic_per_host_per_tick",
+      "capacity_formula": "max(0, VRAM_total - KV_reserved(context, regions_active)
+                           - activation_reserve - safety_margin)",
+      "eviction": "scored: importance + gpu_resident_bonus - staleness(last_accessed);
+                   ties by older timestamp then key; evict lowest until bytes <= capacity_bytes",
+      "partition_key": ["server_derived_scope", "domain", "logical_key"],
+      "_open": ["capacity_bytes, capacity_formula: GAP (a) -- NOT asserted as a contract clause,
+                see section 8",
+                "eviction's staleness_penalty term: GAP (a) extended to the decay function
+                itself [I], see section 8",
+                "partition_key's server_derived_scope segment: GAP (b) -- a policy question,
+                see section 8"],
+      "durability": ["in_memory(conformance oracle only)", "sqlite(WAL, synchronous=FULL)",
+                     "qdrant(wait=true)"],
+      "lifecycle": ["start", "learn", "drain", "flush", "stop"],
+      "max_in_flight": 32,
+      "role": "AI-SPECIFIC. A writable key/value store over workspace latents, read by the
+               workspace as one more participant with its own budget b_store. It IS the
+               overarching context that per-region sliding windows slide over. Parametric cost
+               is two 512x512 projections; the store is data. REQUIRED AND BUILT IN PHASE 2
+               (DEC-49, operator ruling): contract taken clause-by-clause from memory-gate /
+               memory-gate-rs, with the open questions listed in section 8 rather than invented.
+               It is the only writable cross-request object in the design, so scope partition,
+               byte capacity and scored eviction are ACCEPTANCE GATES of rows E0/E1/E2, not
+               documentation. The Schedule may request BUDGET; it may never name a NAMESPACE.
+               token_budget bounds what is READ; capacity_bytes bounds what is STORED.",
+      "status": "planned"
+    },
+    {
+      "name": "affect",
+      "faculty": "affect", "emits": "global",
+      "role": "AFFECT faculty (limbic). Emits a LOW-DIMENSIONAL GLOBAL state that modulates
+               every tract -- a gain/valence signal, not a participant in the attention.
+               Architecturally distinct from every other region and that is the point.
+               Probe: classify_go_emotions.",
+      "status": "placeholder"
+    },
+    {
+      "name": "salience",
+      "faculty": "ai_specific_valuation", "emits": "global",
+      "role": "SALIENCE / VALUATION. Assigns a scalar priority weight to a region's emission
+               given the control state. Its consumer is the scheduler's PRIORITY term.
+               Declared interface only; not trained in v1. Probe: go_emotions macro-AP.",
+      "status": "placeholder"
+    },
+    {
+      "name": "language_trunk",
+      "faculty": "language", "emits": "tokens",
+      "role": "LANGUAGE TRUNK (placeholder). Created when a SECOND specialisation exists, so
+               that 'trunk + adapters' is a measurable claim rather than a rename. Generative
+               half is model/causal_lm.py. Dependency: foundation corpora, curriculum step 4.",
+      "status": "placeholder"
+    },
+    {
+      "name": "numeric", "faculty": "numeric", "emits": "tokens",
+      "role": "NUMERIC / MATH centre (placeholder, interface declared). Emits tokens over
+               VALUES, not BPE pieces -- the architecture has exact arithmetic and the brain
+               does not. Deferred: reasoning's corpus straddles this faculty and splitting
+               today starves both halves.",
+      "status": "placeholder"
+    },
+    {
+      "name": "auditory", "faculty": "auditory_cortex", "emits": "tokens",
+      "phase": "production",
+      "blocked_by": "composed mind passes W6 without audio; explicit operator go",
+      "role": "AUDITORY CORTEX (DECLARED SEAM, DEC-43 as amended by DEC-48). Patch tokens over
+               spectrogram frames -- architecturally identical to visual, so the adapter is
+               shared, not new. The corpus is catalogued as of
+               docs/design/AUDIO-CORPUS-AUDIT.md; rows A0m/A0f (manifests, fetch) and A1
+               build it WHEN THE PRODUCTION PHASE OPENS. Not a v1 participant: R = 5 counts
+               language, memory, reasoning, visual and episodic_store (DEC-49).",
+      "status": "planned"
+    },
+    {
+      "name": "speech_output", "faculty": "language", "emits": null,
+      "parent": "language",
+      "phase": "production",
+      "blocked_by": "composed mind passes W6 without audio; explicit operator go",
+      "role": "SPEECH-OUTPUT HEAD on the language centre (DECLARED SEAM, DEC-44 as amended by
+               DEC-48). Second head on the generative trunk: BPE text tokens plus a discrete
+               speech-token stream a synthesiser consumes. The synthesiser is NOT a region.
+               `emits: null` and `parent` are load-bearing: a head is not a participant, so no
+               consumer counting active regions can turn it into a fifth one (S31-12). Row A2,
+               deferred.",
+      "status": "planned"
+    }
   ]
 }
```

Three properties of this diff worth stating explicitly:

- **`top_k` and `aux_coef` are deleted, not retuned.** They are the parameters of a discrete
  dispatcher. There is no discrete selection in this architecture to tune.
- **`live` becomes `status`.** The current flag means "has an implementation of its `kind` in
  `poc/route.py`", which is why the only two `live: true` entries are the two nobody trained
  [V]. `status` says what a thing *is*.
- **A head is a field, never a sibling entry** `[S31-12 fixed]`. Revision 3.1 registered
  `speech_output` as its own entry in `regions[]` with `emits: "tokens"`, which is the
  operational definition of a **participant** in §2.2 — so any consumer computing
  `R = len([r for r in regions if r is active])` would have picked up a fifth region for a thing
  DEC-44 itself calls *"a head on the language centre, not a faculty"*. `memory` is the
  precedent and it is a **single** entry with a `heads` list. Fixed the same way: `language`
  carries `"heads": ["text", "speech"]`, and the `speech_output` row survives only for programme
  tracking, with `emits: null` and `parent: "language"` so it cannot be counted. **Re-keyed from
  `language_code` to `language` in revision 3.6 (DEC-78).**
- **`episodic_store`'s `capacity_bytes`, `capacity_formula`, `eviction`'s staleness term and
  `partition_key` are shipped as values, but §1.3's cell states in words that three of the four
  are NOT asserted as a contract clause here** `[S33-3 fixed]`. A JSON diff that a reader
  ratifies alongside the prose would otherwise ratify §8 gap (a), gap (b) and the staleness
  decay term without being told they were the open items. The entry now carries a sibling
  `"_open"` list naming exactly those four keys against their gap, so the JSON cannot be read as
  a settled contract on its own.
- **`status` gains exactly one value in revision 3.1: `planned`** (DEC-43, DEC-44). The
  vocabulary `built | probe | placeholder | primitive | retired` had no word for *required,
  specified, programme row open, not yet trained* — and `placeholder` is the wrong word for a
  thing the operator has ruled mandatory. Using it anyway is how a requirement quietly reads as
  an option, which is the failure this whole taxonomy exercise exists to retire. **The
  difference is checkable, which is the only reason the word is worth adding: a `planned` entry
  has a programme row with a gate that can fail; a `placeholder` has neither.** `auditory` and
  `speech_output` are `planned`; **`episodic_store` becomes `planned` in revision 3.3** (DEC-49
  gives it rows E0/E1/E2 with gates that can fail, which is exactly the difference the word was
  added to mark); `salience`, `language`, `numeric` and `affect` stay `placeholder`. **Revision 3.2 adds one field rather than a second word:**
  `phase: "production"` on the two deferred entries, with `blocked_by` beside it. `planned` still
  means *required, specified, gate written*; `phase` says **when**, and the pair is what keeps
  DEC-48's deferral from quietly reading as a demotion back to `placeholder` (DEC-48).
- **No `connects_to` field.** Connectivity is learned; writing it down would imply a
  hand-authored topology that does not exist — the same reasoning `region_spec.py` already
  applies to `router_trigger` [V\*].
- **`native_dim` is split into `token_dim` and `pooled_dim`** `[A32 fixed]`. One field cannot
  be both: `TextEncoder.forward` applies `self.proj` **after** pooling
  (`regions/text_encoder.py:125`, `return self.proj(pooled)` [V]), and `self.proj` is
  `nn.Identity` only while `cfg.out_dim` is unset (`:74-76` [V]) — which `data/stream.py:251`
  already sets in one path. So `tokens()` returns the **pre-`proj`** representation at
  `cfg.dim` (`token_dim`, what the adapter consumes) while `pool()` emits at `out_dim`
  (`pooled_dim`, what every existing receipt is about). They are equal in every current
  checkpoint and will not stay equal.
- **Two entries moved to `placeholder` in revision 2; one of them moves back out in revision
  3.3.** `episodic_store` (DEC-32) and `tract_codec` (A33) were both `built`/live with no
  objective, no receipt and no gate. Shipping a thing as built because its interface is written is
  the defect this whole taxonomy exercise exists to retire, and revision 1 committed it twice.
  **`episodic_store` now becomes `planned`, not `built`** (DEC-49): it has a contract, a row and a
  gate that can fail, and it acquires `status: built` only when E1 and E2 pass. `tract_codec`
  stays `placeholder`. **The distinction is the whole point of the vocabulary and it is worth
  restating: `planned` is a promise with a failable gate, `placeholder` is a name.**

## 1.5 DEC-11 — retire `CSD-BRAIN-REGIONS.md`, re-home its four unique decisions

The file is not in the working tree and not at HEAD, and six tracked files cite it as
authoritative [V\*]. It frames regions as *"MoE-adjacent experts"* with a `SoftmaxRouter`
*"gate, not a region"* — which this document supersedes. Retire it and fix the six dangling
citations:

| what it uniquely decided | new home |
|---|---|
| the routing surface `activate(stream) → [B,D]` | **superseded** by §2.2; `contracts/region.py` is rewritten |
| the mandatory curriculum R0→R3 | **superseded** by the four phases in §4 |
| *"`route` is not a region and has no pretrain corpus"* | **REVERSED.** The interconnect has training data and it is §5 |
| SciFact / NFCorpus eval-only under NC/ToS | move to `docs/design/CORPUS-CONTRACT.md`; already enforced in `scripts/csd-fetch-retrieve-pairs.py` [V\*] |

Also carried forward unchanged, because it is still correct: lab placement (region pretrain on
the 3090 at `.98`; the 5080 at `.251` is the deployment target; **no train / PoC CUDA on
Pascal** at `.243`) and the publication gate (no Hub publish until an `hf/autodev` fine-grained
token exists).

## 1.6 DEC-12 — the dormant `core/` stack: harvest, mark superseded, do not delete

1,320 tested lines exist and are imported by no training path: `mHCInterconnect`,
`ContextualAttentionRouter(embed_dim=512, num_heads=8)`, `PathwayOptimizer`,
`ContextPropagationEngine(max_hops=3)`, `CongestionController(total_bandwidth=10000)`, and a
`BrainRegionType` enum naming eight of the nine intended faculties [V\*].

**The decisive technical fact, found independently by all three proposals:**
`ContextualAttentionRouter.compute_importance` and `allocate_bandwidth` return
`tensor.item()`, and `forward` returns `dict[tuple[str,str], float]` [V\*]. **It is not
differentiable as written.** There is no gradient path from any loss to its parameters. It can
be run; it cannot be trained.

That is decisive about *revival* and not about *deletion*. Deleting it also destroys the only
record of where `stream_dim: 512` came from. **Verdict: harvest the four ideas, mark the
modules superseded in their own docstrings with a pointer to this document, stop importing
them, and leave the code and its tests in place.** No operator sign-off is needed for that,
which is the point — a deletion would need one and buys nothing.

| idea | from | where it lands |
|---|---|---|
| attention over section states = who talks to whom | `ContextualAttentionRouter.forward` | §2.3's `a_r`, as a **tensor on the graph** rather than a dict of floats |
| per-pathway bandwidth allocation | `allocate_bandwidth`, `mHCPathway.bandwidth` | `b_r`, the same idea made integer and differentiable |
| bandwidth under contention / priority queue | `CongestionController` | the floored simplex and largest-remainder integerisation in §2.4; the hard caps in §9.9 |
| multi-hop propagation | `ContextPropagationEngine(max_hops=3)` | `n_iter` workspace iterations plus the admission matrix |
| the faculty vocabulary | `BrainRegionType` | the closed `faculty` enum in §1.4 |

---

# 2. White matter — the interconnect module specification

## 2.1 DEC-13 — operator decision D1 answered: **(c)**, with the division fixed by timing

The three options were (a) cross-region attention with emergent routing, (b) fixed interconnect
plus a thalamic gate, (c) both [V\*]. The answer is **(c)**, but the usual reading of (c) —
"build (a), then bolt (b) on if it helps" — is wrong, because it leaves unstated which job each
half does. The division follows from one observation, and it is not a preference:

> **Attention can only weight things that have already been computed.** Every scheduler
> behaviour that exists in order to *avoid computing something* — which regions to activate,
> how much context each gets, how many iterations to run — must be decided **before** the
> attention that would otherwise reveal the answer. Attention cannot make those decisions, not
> because it is a poor mechanism but because it runs too late. [I]

| job | mechanism | why this half |
|---|---|---|
| transport, integration, **connection strength**, attention split | **cross-region attention** | The attention weights *are* the connection strengths. Nothing else is added, so routing is emergent because there is nothing else there. |
| **activation set, context budget, read budget, iteration depth** | **thalamic controller** | Pre-computation decisions. The thalamus relays and modulates what reaches cortex — it gates afferent bandwidth, it does not pick a winner. That is exactly this job and exactly the operator's characterisation. |
| nothing | **(b)'s *fixed* interconnect** | Rejected. A fixed high-bandwidth interconnect is `R²` constant-strength matrices that learn nothing about context. Strictly dominated by attention at equal parameters. |

And the property that makes (c) safe rather than fragile: **the controller never invents its own
objective at the moment it is introduced.** It is bootstrapped from the attention mass the dense
model already produced, and only then trained by the task loss (§2.6, phases B→D).

**The scheduler stays inside white matter.** One proposal moved it to frontal cortex on the
timing argument above. The timing argument is correct and it proves the scheduler needs a
*pre-region input*; it does not prove it must be a *different faculty*. The operator was
emphatic that white matter **is** the interconnect **and** the routing model. Keeping the
controller thalamic and inside the interconnect satisfies both the timing constraint and the
framing, so the relocation buys nothing and costs fidelity [I].

## 2.2 DEC-14 / DEC-15 — the representation contract

`CognitiveRegion.activate(stream) → [B,D]` is satisfied by **zero trained regions** [V\*]. The
trained surface is `text → representation` — an encoder *into* the state, not an operator *on*
it. Those are different jobs with different type signatures and nothing bridges them. The
`stream→stream` contract does not survive.

```python
@runtime_checkable
class Faculty(Protocol):
    name: str
    faculty: str
    token_dim: int                  # what tokens() emits, PRE-proj: 256 text, 384 visual
    pooled_dim: int                 # what pool() emits, POST-proj; every receipt is about this
    kv_bytes_per_token: int         # c_r -- THE REGION DECLARES ITS OWN COST
    accepts_condition: bool

    def tokens(self, inputs, *, context_tokens: int,
               condition: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Region-native representation BEFORE pooling AND BEFORE `proj`, computed over AT
        MOST `context_tokens` input positions -- the region never encodes more than its budget.
        `condition` is the workspace latent state, projected to token_dim and prepended
        as a conditioning prefix (write-back, DEC-17); None at iteration 0.
        returns (h [B, T_r, token_dim], mask [B, T_r]) with T_r <= context_tokens."""

    def pool(self, h: Tensor, mask: Tensor) -> Tensor:
        """[B, T_r, token_dim] -> [B, pooled_dim]: masked mean, THEN `proj`. The region's
        standalone answer, kept so every existing receipt stays reproducible and comparable.
        For `visual`, `pool` is `h.mean(1)` and is exactly `IJEPA.encode` (DEC-34)."""
```

**GLOSS, stated once here and governing every later use of the word** `[DEC-47]`. *"Token
surface"* and `tokens()` mean **position latents — per-position hidden vectors, not discrete
ids.** The method name is the W0 API and is **not renamed**: renaming a shipped method to repair
a reading is worse than glossing it. But nothing in this document ever means *discrete tokens*
when it says a region's *tokens*; `pool()` is a latent too. **The only discrete ids anywhere in
the system are at the input tokenizer and at the frontal read-out**, and DEC-47 makes that a
checked property rather than a habit.

> ### DEC-47 — the latent-space reasoning invariant `[OP: csd-latent-space-reasoning-invariant.md]`
>
> Operator, 2026-09-02, verbatim:
>
> > *"the intent of CSD is to do its reasoning and thinking in latent space as opposed to
> > discrete token space. rather than just ingesting and sitting on and working on discrete
> > tokens, the goal is for CSD to essentially convert to discrete latent concepts and reasoning
> > and be able to output whatever format is appropriate/fitting"*
>
> Five clauses, each checkable against a row of this document:
>
> 1. **Encoders convert each modality to latents.** Text tokens, image patches and (at the
>    deferred audio seam) spectrogram frames all enter through a region's `tokens()` and become
>    position latents. **From that point on nothing is re-serialised to discrete tokens until the
>    read-out.**
> 2. **Reasoning happens in the workspace latents**, and **the scheduler's iteration depth IS the
>    latent-reasoning loop.** `max_iters`/`halt_at` (§2.4, §2.5) is not a plumbing parameter: it
>    is how many times the mind thinks before it speaks. Its future form is the deferred
>    recurrent-depth direction — looping the forward pass in latent space to refine a prediction
>    — recorded in §6.9 beside the ternary track, deliberately **not** built at v1.
> 3. **Regions exchange latents only.** No region emits text to another region; DEC-17's
>    write-back conditioning prefix is a **projected workspace latent**, and §2.2's `condition`
>    argument is typed as a tensor at `token_dim`, never as ids. This is the clause with a gate:
>    **W9 asserts that no inter-region path carries discrete token ids, and the constructed
>    violation must be refused** (§4.1 W9).
> 4. **Output modality is chosen at the frontal read-out**, not upstream of it: emission is a
>    decoder choice, not a property of the reasoning. That is why §2.5's `output.modalities` and
>    DEC-44's multi-head read-out **stay in the runtime contract even though audio is deferred**
>    (DEC-48) — the seam is the thing that keeps clause 4 true later without a redesign. Text and
>    speech are two decoders over **one** latent state, and `numeric`'s emission over *values,
>    not BPE pieces* (§1.3) is the same clause read in a third direction.
> 5. **"Discrete latent concepts" is a hypothesis about REPRESENTATION, not a licence for a token
>    bottleneck.** Quantised concept codes (VQ-style codebooks) are a **candidate tract codec** —
>    the slot DEC-08 retired `stream_vae` into — and they connect to the ternary track (§6.9).
>    They are to be **tested as a codec**, with **continuous latents as the default until
>    measured**. Reading the operator's phrase as a mandate for discrete inter-region messages
>    would invert the invariant it appears inside.
>
> **What it costs to state now:** nothing. **What it buys:** the swarm experiment (P5′s) has a
> comparison worth running — latent exchange against a token-exchange baseline — and W9 has a
> failable assertion instead of a convention.

**The precise claim, corrected** `[A30 fixed]`. Revision 1 said *"this retrains nothing… that
property is what makes the whole programme affordable"* and was contradicted by four rows of its
own plan. The true and still-useful claim is narrower:

> **The token surface retrains nothing.** `TextEncoder.forward` already computes `h` at
> `[B,T,256]` and mean-pools it away one line later; `ViTEncoder.forward` already returns
> `[B, 64, 384]` [V\*]. Splitting the method at the pooling line yields both methods with
> byte-identical behaviour, and **every existing checkpoint loads unchanged**.

What phase 2 *does* retrain is now large, and it is budgeted in §4.3 rather than implied: W1b
(regenerate `reason`), W4 (`memory` merge, which is also the first token-aware retrain), W7a/b
(the remaining token-aware retrains, **mandatory** since W1's result — §4.0) and W7v (`visual`,
resolution + token-aware, absorbing OD-4's corpus replacement). Affordability is now an argued
budget of measured wall-clocks, not a property.

**Why pre-pool tokens and not the pooled vector.** A pooled vector is already a decision — the
region's committed, integrated answer. Cross-attention over seven such vectors has exactly one
degree of freedom: how much of each answer to keep. That is a weighted sum of independent
decisions, i.e. dispatch **by construction**, not by convention: region A's computation cannot
have been influenced by region B, because A finished before the mixing began. No amount of
training makes that architecture integrate [I]. The fix is free and already computed.

**DEC-15 — two budget currencies, because the operator named two levers and they are not the
same quantity.**

| currency | symbol | what it bounds | when it is consumed | constraint |
|---|---|---|---|---|
| **region context** | `ctx_r` | region-internal activations and KV, the `O(T_r²)` term — **the term that dominates at scale** | **before** the region runs; it is an argument to `tokens()` | `Σ_r c_r · ctx_r ≤ B_kv` by simplex construction |
| **read tokens** | `b_r` | workspace KV = `Σ_r b_r × D_w × 2 × depth × 2 bytes` | **after** the region runs; a straight-through top-k over the emitted `h_r` | `Σ_r b_r ≤ B_read` by simplex construction |

Both are emitted by the controller *before* any region runs; only one is *consumed* before. This
resolves a real contradiction in the source proposal, where a pre-computation controller's only
consumer was a post-computation `TopK`, so selective activation bought workspace KV and nothing
else — a lever on the cheap term [I, and the fix is grafted from the systems-first proposal's
two-simplex construction].

**Widths.** `D_w = 512` for the workspace; **regions keep their native widths and adapt in**
via `nn.Linear(token_dim, 512)` — **591,872 params for the four v1 participants** [I]; 986,624
was revision 1's figure for a seven-region list that does not exist `[A24 fixed]`. 512 because the catalogue
already declares it and the entire dormant `core/` stack defaults to it, so choosing it makes
the catalogue honest for free. `MindSpec`'s single-`stream_dim` check is replaced: it enforces a
uniformity that has never been true and would reject the real trained regions (256 and 384)
against the declared catalogue (512) [V\*]. The four-incompatible-widths problem dissolves; it
was only a problem because one global width was assumed.

**Visual latents enter the same way everything else does** — a `[B, 64, 384]` patch-token stream
through a `384→512` adapter. It is the only region whose output surface was already correct,
which is why it is wired **first** (§4, W0).

## 2.3 DEC-16 — the module

```
                        ┌───────────── thalamic controller ─────────────┐
  raw input ───────────►│ cheap summary → 2 blocks @256 → ctx, b, A, halt│
                        └──────┬────────────────────────────┬───────────┘
        ┌──── regions (frozen in phase 2) ────┐             │ admission A[i,r]
        │ language_code.tokens(ctx_r, z_{i-1})│             │
        │ memory.tokens(...)      [B,T,256] ──┤             ▼
        │ reasoning.tokens(...)   [B,T,256] ──┼─ TopK_{b_r} ─► adapters ─► K/V bank
        │ visual.tokens(...)      [B,64,384] ─┤                        [B, Σ b_r, 512]
        │ episodic_store.read()   [B,b_s,512]─┘                              │
        └────────────────▲────────────────────┘                              │
                         │ conditioning prefix (write-back, DEC-17)          ▼
   latents z ∈ [B,64,512] ──► ×n_iter [ cross-attn(z ← KV) → self-attn(z) → MLP ] ──► frontal
                                          │
                                          └── per-region attention mass a_r ──► supervises §2.6-B
```

**`visual.tokens(...)` above is the shape TODAY** (`image_size 64` ⇒ `[B,64,384]`), unchanged
until W7v retrains. **W3r fixes the composite frame this shape rebuilds to once W7v lands:**
`image_size 128, patch_size 8` ⇒ **`[B,256,384]`** — single source of truth
`src/cogsyndelta/vl/composite.py` (`FRAME_SIZE`, `PATCH_SIZE`, `N_PATCHES`), a checked-in
128×128 fixture at `tests/fixtures/composite_4panel.png`, and §1.4's `visual` block. The diagram
is not redrawn here (that is W7v's edit, against a shape it has actually built), but the number
it will carry is now fixed and tracked rather than open.

**Inputs.** Per active region: `(h_r [B,T_r,d_r], mask_r)` and its budgets.
**Outputs.** The workspace state `z [B,64,512]`, the frontal read-out `[B,512]`, the
connectivity tensor `a [B, n_iter, R]`, and the emitted `Schedule`.

Exact tensor path per iteration `i`:

```
cond_r   = reshape(Linear_r(pool(z_{i-1})), [n_cond, d_r])        # write-back, i >= 1
h_r      = region_r.tokens(inputs, context_tokens=ctx_r, condition=cond_r)
KV_i     = concat_r  Adapter_r( TopK_{b_r}(h_r) ) + type_emb[r] + sincos(pos)
z        = z + CrossAttn(q=LN(z), kv=LN(KV_i))
z        = z + SelfAttn(LN(z))
z        = z + MLP(LN(z))
a[:,i,r] = Σ_{l=1..64} Σ_{t ∈ keys(r)} CrossAttnWeights[:, l, t]   # <- CONNECTION STRENGTH
```

`a_r` is **read off the softmax**, never predicted by a head with its own loss. That is the
operator's *"attention weights ARE the connection strengths"* implemented literally, and it is
what makes routing emergent rather than a dispatch decision made outside the representation.

### Parameters at toy scale — tagged per row `[A35 fixed]`, recomputed for the v1 participant list `[A24 fixed]`

The v1 participant list is **five** participants — `language_code`, `memory`, `reasoning`
(text, 256), `visual` (384) and **`episodic_store`** (latent, 512), the last reinstated by DEC-49.
**Four of the five carry an adapter; the store carries two projections instead**, because it emits
at the workspace dimension already and needs no dimension change — which is why the adapter row and
the store row are separate lines below rather than one line of five. Revision 1's table
billed **six** text adapters against three text participants and then carried the sum forward as
`[V*]` "measured"; the surplus was `3 × (256×512 + 512) = 394,752` params, propagating into the
total, the composed-mind table, the 347 MB figure, the 35.4 MB figure and §6.4's cap arithmetic.

| component | params | tag | share |
|---|---|---|---|
| workspace blocks, ×4 @ `D_w=512`, 8 heads, **mlp_ratio 4** | 16,803,840 | [V\*] | 62.5% |
| frontal read-out (attention-pool + MLP) | 3,150,848 | [V\*] | 11.7% |
| thalamic controller (2 blocks @256 + heads) | ≈1,588,007 | **[I]** — measured at 3 heads; this design has 4, so ±0.1M | 5.9% |
| conditioning prefixes (write-back: 3×256 + 1×384 token_dim, `n_cond=8`) | 4,727,808 | [V\*] | 17.6% |
| region adapters (**3** × 256→512, 1 × 384→512) | **591,872** | [I] — arithmetic over the four *encoding* participants | 2.2% |
| **`episodic_store` read/write projections** (`W_k`, `W_v`, 512×512) | **524,288** | [I] — **restored by DEC-49**; was struck out under DEC-32 | 1.9% |
| latent bank `64 × 512` + final norm + type embeddings | 37,376 | [V\*] | 0.1% |
| **WHITE MATTER v1 TOTAL** | **27,424,039** | **[I]** — a sum over an `[I]` row is `[I]`, not `[V*]`. Was 26,899,751 under DEC-32 | |

**Re-instantiate at the design's actual configuration (4 controller heads, this participant
list) before this total is carried anywhere.** That is W0's second deliverable and it costs
seconds; until it is done, `27,424,039` is a projection, not a measurement. **W0's
re-instantiation now covers five participants**, and the store's two projections are part of what
it must instantiate — a re-instantiation that silently drops the participant this revision added
would reproduce exactly the surplus-adapter defect A24 was raised for.

Calibration: a text region is 16,021,248 params of which **12,865,792 (80.30%) is the token
embedding table** and only **3,155,456 (19.70%) is compute** [V\*]. **White matter is 8.5× a
region's compute parameters.** It is the centre of gravity in arithmetic, not only in framing.

**Composed mind v1, deployable** (one half of `visual` dropped, probes excluded, **store
included as of DEC-49** — its parametric cost is inside white matter, its *data* is not a model
parameter and is budgeted in bytes under §8 gap (a) instead):

| | params | tag |
|---|---|---|
| `language_code` | 16,021,248 | [V\*] |
| `memory` (merged trunk + 2 heads) | 16,152,320 | [V\*] |
| `reasoning` | 16,021,248 | [V\*] |
| `visual` (**EMA target encoder only**, DEC-34) | 10,712,448 | [V\*] |
| **white matter** (incl. the store's two projections) | 27,424,039 | [I] |
| **TOTAL** | **86,331,303** — 345 MB fp32 · **35.3 MB at the measured 3.2675 effective bits/param** | [I] |

**The store costs 524,288 params, 0.6% of the mind, and moves the interconnect's share from 31.4%
to 31.8%** — which changes nothing about DEC-26, whose 2–5% cap is scoped to ≥1B and is already
argued at 31.4% for the toy scale (§6.4). The number worth watching is not the parameters; it is
the **bytes of store data**, which are not in this table at all and are bounded by §8 gap (a)'s
dynamic capacity rather than by any figure here `[I]`.

Three facts this table encodes and the programme's headline "~87M" does not:

1. **10,712,448 of `vl_latent`'s 22,905,216 params do not ship** [V\*] — and revision 1 named
   the wrong half. `IJEPA.encode` is `return self.target_encoder.embed(images)`
   (`model/vl_jepa.py:387` [V]), and `_features` — which feeds the only linear probe and
   transfer numbers `visual` has — calls `model.encode` (`regions/vl_pretrain.py:202` [V]). So
   **every measured number about `visual` describes the target encoder** `[A3 fixed]`.
   **DEC-34: the target encoder is the deployed half.** The choice costs nothing and makes four
   receipts true; the alternative (ship the context encoder and gate it) costs a probe run and
   would deploy the half that is *out of distribution on whole images*, since it trained at
   `context_keep = 0.4` with unkept patches **dropped, not masked** (`vl_jepa.py:191-199` [V]:
   *"Masked patches are DROPPED rather than replaced with a mask token"*), so it has never seen
   a full 64-patch sequence. **W1's `vl_latent` measurement was verified bit-exact against
   `IJEPA.encode()`**, so the one experiment already run is also about the target encoder —
   a third reason to name it the deployed half rather than invalidate the measurement.
   *The parameter arithmetic is unaffected*: `self.target_encoder = copy.deepcopy(self.encoder)`
   (`vl_jepa.py:316` [V]), so both halves are 10,712,448 params (see §11 R3).
2. The `memory` merge saves **15,890,176** params by sharing one embedding table between two
   former regions [V\*].
3. **`memory` is NC.** It inherits `retrieve`'s GooAQ term, and by DEC-31 the composed mind
   therefore is too `[OP: csd-release-licence-decision]`.

### The 30B sketch

| | v1 | 30B-class | grows? |
|---|---|---|---|
| faculties built | 4 encoding faculties + the store = **5 participants** (DEC-49) | 16–20 incl. specialisations | yes — phase 3 is where region *types* appear |
| params per region | 16.0M (12.9M of it embedding) | ~1.5B | yes |
| token embedding table | one per text region | **ONE, shared** (DEC-24) | no |
| workspace `D_w` / latents `L` / depth | 512 / 64 / 4 | 4096 / 256 / 4 | yes |
| **interconnect params** (at `mlp_ratio 4`, stated in both columns `[A26 fixed]`) | 27.4M (**31.8% of the mind** — see DEC-26) | **1,073,741,824 = 3.6% of 30B** at `D=4096, L_ic=4`; `L_ic=8` = 2.15B = **7.2%, ABOVE the 5% ceiling** | yes, sublinearly |
| `B_read` (Σ read tokens) | 256 | 4,096 | yes |
| `B_kv` (Σ region context bytes) | — | **3.29 GiB** (§6.3) | fixed by the card |
| **the interface** (`tokens()` — position latents, `[DEC-47]` — adapters, `Schedule`) | | **identical** | **no** |
| **the runtime** (DAG executor, validator) | | **identical** | **no** |

Revision 1 printed **805,453,824 = 2.7%** for the 30B column. That figure is `12·D²` per block,
i.e. **`mlp_ratio 2`**, while §2.3's v1 blocks are `16·D²` per block, i.e. **`mlp_ratio 4`**
(`16,803,840 / 4 = 4,200,960 ≈ 16·512²`) — a silent architecture change between the two columns
of a table whose whole purpose is to argue an invariant. At the ratio the design actually
specifies, `L_ic = 4` is 3.6% and `L_ic = 8` is 7.2%, so **DEC-26's "5.4% is the ceiling" claim
flips sign** and is restated in §6.4.

### Workspace compute — the derivation, printed `[A27 fixed]`

Revision 1 asserted **620.8 MMAC/item** tagged `[V*]` "measured". It does not reconcile. At
`D_w = 512`, `L = 64` latents, `Σb = 256` KV tokens, `mlp_ratio 4`, per block:

```
self-attention q,k,v,o projections   4 · D² · L            =  67,108,864
self-attention scores (QKᵀ and AV)   2 · L² · D            =   4,194,304
MLP (two matmuls, ratio 4)           2 · D · 4D · L        = 134,217,728
cross-attention q,o on the latents   2 · D² · L            =  33,554,432
cross-attention k,v on the KV bank   2 · D² · Σb           = 134,217,728
cross-attention scores               2 · L · Σb · D        =  16,777,216
                                     ------------------------------------
per block                                                  = 390,070,272   (390.1 MMAC)
× 4 blocks                                                 = 1,560,281,088 (1.56 GMAC/item)
```

**1.56 GMAC/item [I], ~2.5× the figure revision 1 carried.** If 620.8 came from a profiler,
the receipt must say which modules were counted; until then the derived number governs, and the
2.10 MB/item workspace KV → 268 MB/item projection must be re-derived beside it.

**The stated asymptotic was also wrong.** `O(L · Σb_r · D_w)` is the attention-**score** term
only. It omits `O(L · D_w² · (1 + mlp_ratio))` and `O(Σb · D_w²)`, which at `D_w = 4096`
dominate by an order of magnitude and are precisely the terms that decide whether the centre of
gravity stays affordable. Restated:

> **workspace compute is `O(L·D_w²·(1+mlp_ratio) + Σb·D_w² + L·Σb·D_w)`, and no term in it
> grows with region size.**

The *conclusion* revision 1 drew is correct and survives; the expression it drew it from did
not support it.

## 2.4 The five scheduler behaviours, write-back, and topology

| behaviour (operator's list) | mechanism | learned by |
|---|---|---|
| **which regions to activate** | `ctx_r > 0` | controller; hard floor `ctx_r ≥ ctx_min` for every declared region (§9.9) |
| **intensity and priority** | intensity = `a_r`, emergent per iteration. Priority = the order budgets are honoured under contention: `b = largest_remainder(B_read · β)`, `β = η/R + (1−η)·softmax(s)`, `η = 0.15` | `a_r` by the task loss; `s` by §2.6 |
| **attention split across regions** | the cross-attention softmax itself. Nothing else. | task loss |
| **per-region context budget** | `ctx_r` (pre-computation, region-internal) and `b_r` (post-computation, workspace) — two simplexes | controller, straight-through on the integerisation |
| **execution topology** | admission matrix `A ∈ {0,1}^{n_iter × R}` + ACT halting, plus write-back | task loss through the admission mask |

**The floored simplex is the load-bearing anti-starvation trick, and it is a bound by
construction rather than by check.** A softmax cannot sum to more than 1, so no input —
adversarial or otherwise — can drive the allocation past the card; the floor `η/R` of the
budget is what stops it starving a region. **The floor is written as the expression and never as
a constant** `[S31-16 fixed]`: at `η = 0.15` it evaluates to **3.0%** for the v1 `R = 5`
(DEC-49). **Revision 3.3 is the case that proves why S16's fix was worth making.** Revision 3.1
hard-coded `3%`, which is `0.15/5`; revision 3.2 corrected it to `3.75%` because `R` was 4; and
DEC-49 makes `R` five again, so the *value* returns to exactly the number that was wrong two
revisions ago **for a different reason**. A document carrying the constant would now be
accidentally right and would have no way to show it. **Only the expression is binding**, and every
receipt prints the evaluated floor **beside its own `R`**, so a count change can never make a
constant accidentally correct in either direction. **`b_store` is subject to the same floor:** the
store's share cannot fall below `η/R · B_read = 0.03 × 256 ≈ 8 read tokens`, which is what makes
"the store received no attention mass" a *measurement* in E2 rather than a starvation artefact. Grafted from the systems-first proposal, where the
bound is derived: at `B_kv = 3.0 GiB` and `c_r = 192 KB/token` the floor is **≥ 409 tokens** for
any active region [V\*].

### DEC-17 — write-back, and why the DAG's edges are otherwise unearned

In a pure Perceiver workspace, regions are only ever keys and values: the latents read from the
regions and the regions never read the latents. Region A's *computation* therefore cannot be
influenced by region B — only the latents' aggregation of them can be. A DAG derivation that
says *"region r′ reads latents that already absorbed r"* is, in that architecture, an assertion
with no mechanism behind it [I]. This is a real hole in the source proposal and it is fixed
here rather than inherited.

**Mechanism.** At iteration `i ≥ 1`, an admitted region with `accepts_condition: true` receives
`cond_r = reshape(Linear_r(pool(z_{i-1})), [n_cond, d_r])`, `n_cond = 8`, prepended to its own
input embeddings. `Linear_r` belongs to the **interconnect**, not to the region, so the region
checkpoint stays frozen and phase 2 still retrains nothing. Cost: 1,050,624 params per text
region, 1,575,936 for the visual region, 4,727,808 total.

**The risk, stated because it is real.** Prepending eight learned vectors to a frozen encoder
trained with InfoNCE at `max_len 96` is prefix tuning on an out-of-distribution input, and it
spends 8% of that region's length budget. **Gate (W5b):** enabling write-back must not drop the
conditioned region's own-bin score by more than 1 point, and must improve the composed metric.
**Fallback, pre-committed:** if it fails, write-back is disabled for that region and the receipt
records that phase-2 integration is *workspace-internal only*, with inter-region conditioning
deferred to phase 3 when regions unfreeze. Either result is a pass; not running it is the fail.

### DEC-18 — topology derived twice, and the two derivations must agree

```
depth(r)    = min{ i : A[i, r] = 1 }
nodes       = { r : ctx_r > 0 }
async       = { r : depth(r) = 0 and A[:, r] ≡ 1 }         # never waits; nothing waits on it
lockstep(i) = { r : depth(r) = i ≥ 1 and accepts_condition } # exchange through z_{i-1} → z_i
sequential  = a path of strictly increasing depth
```

Write-back is what makes **lockstep** expressible at all: two regions admitted at the same
iteration `i ≥ 1` both read `z_{i-1}` and both write into `z_i`, so they are mutually dependent
*within* the step — which is the definition of parallel-in-lockstep, and it is distinguishable
from parallel-**independent** (regions at depth 0, which read no latents). Without write-back,
equal depth means independence, and "lockstep" would be a mislabel — a hole a judge correctly
identified in the source proposal.

**Cross-check, grafted from systems-first.** Define the effective region→region influence
through the latent bottleneck:

```
Ĉ[i,j] = Σ_t  ⟨ a[:, t, i] ,  w[:, t+1, j] ⟩ / Z
```

where `a[l,t,i]` is the mass latent `l` gave region `i`'s keys at iteration `t` (write) and
`w[l,t+1,j]` is the weight with which latent `l` contributed to region `j`'s conditioning prefix
at `t+1` (read). Threshold `Ĉ` and topologically sort: **DAG levels are stages, cycles are
lockstep groups, isolated nodes are async.** `Ĉ` is *measured*, not predicted — no separate
head, no relaxation over topologies. **Gate (W9):** the two derivations must agree on ≥95% of
eval items; a disagreement is a bug in one of them and is reported, not averaged.

**`Ĉ` under W5b's pre-committed fallback `[A20 fixed]`.** Conditioning prefixes exist only if
write-back is enabled, and W5b pre-commits that on failure write-back is disabled and *"either
result is a pass"*. In that branch `w` is undefined, so `Ĉ` is undefined — and revision 1 left
W9's agreement gate, falsifier S4 and the emitted `Schedule.edges` standing on a quantity that
does not exist, which is the exact hole DEC-17 was written to close, reopened by its own
fallback. Two things are stated now, and the second is the one that must be honoured:

1. **The no-write-back variant.** Replace the prefix-read term with influence through the latent
   bottleneck across iterations: `w′[l, t+1, j] = ∂ (mass latent l gives region j's keys at
   t+1) / ∂ (region i's contribution to z_t)`, estimated by the same forward passes with
   region `i`'s keys masked at `t` (the I2′ severance, §2.7.5, already computes exactly this).
   It is weaker — it measures influence on *attention*, not on *computation* — and the receipt
   must say so.
2. **If the variant is not built, the gate is void and the emission is trimmed, not
   asserted.** Under the fallback, W9's ≥95% agreement gate does not run, S4 has nothing to
   print, the emitted `Schedule` carries **no `edges` and no `lockstep_groups`**, and the
   receipt records `topology: not demonstrated`. That is DEC-30's two-verdict discipline
   applied where revision 1 omitted it. An emitted edge with no mechanism behind it is the
   assertion DEC-17 exists to prevent, and it is not made cheaper by being in a JSON field.

## 2.5 What it emits, and the runtime contract

```jsonc
Schedule {
  "step_budget": { "kv_bytes": 3221225472, "read_tokens": 256, "max_iters": 4, "wall_ms": 250 },
  "nodes": [
    { "region": "visual",        "active": true,  "depth": 0, "intensity": 1.31, "priority": 0,
      "context_tokens": 64, "read_tokens": 96, "precision": "int3", "resident": true,
      "codec": "none", "condition": false },
    { "region": "memory",        "active": true,  "depth": 0, "intensity": 0.72, "priority": 1,
      "context_tokens": 96, "read_tokens": 64, "precision": "int3", "resident": true,
      "codec": "vae64", "condition": false },
    { "region": "language_code", "active": true,  "depth": 1, "intensity": 0.94, "priority": 1,
      "context_tokens": 96, "read_tokens": 64, "precision": "int3", "resident": true,
      "codec": "none", "condition": true },
    { "region": "reasoning",     "active": false, "depth": null, "intensity": 0.0, "priority": 3,
      "context_tokens": 0,  "read_tokens": 0,  "precision": "int3", "resident": false }
  ],
  "edges": [ {"src":"visual","dst":"language_code","weight":0.41},
             {"src":"memory","dst":"language_code","weight":0.22} ],
  "stages": [ ["visual","memory"], ["language_code"] ],
  "lockstep_groups": [],
  "halt_at": 2,
  "output": { "modalities": ["text", "speech"], "stream": true,
              "first_token_ms": 300, "speech_frame_ms": 40 },
  "trace_id": "SERVER-MINTED; never accepted from the emission [T2 fixed]"
}
```

**`trace_id` is minted server-side and is not part of what the model emits** `[T2 fixed]`. It is
attacker-influenced free text destined for logs, Jaeger spans and receipts; receipts are the
programme's only evidence layer, and a log field an untrusted party controls is a log-injection
primitive. The validator rejects a `Schedule` that carries one.

**The read-out emits to more than one modality, and the `Schedule` must say which — DEC-44,
KEPT AS DECLARED even though the speech head is deferred** `[OP: csd-multimodal-io-intent.md]`.
**This whole `output` block stays in the runtime contract under DEC-48.** Deferring the head and
deferring the contract are different acts: the second is what forces a redesign later, and it is
the one thing the deferral is explicitly not allowed to do. At v1 the validator accepts `text`
and refuses `speech` for the same mechanical reason it will refuse it after the head exists but
is evicted — **no resident head, no modality** — so the deferred state is a *value* of the rule
rather than an exception to it. It is also clause 4 of DEC-47: modality is chosen at the read-out,
so the contract that names modalities belongs here and not upstream. Revision 3's contract had exactly one output shape (a ranking
over a declared candidate set, DEC-41) and one implicit modality. The operator's intent is
*"text/speech output... output multimodal as well"*, so the frontal read-out is a **multi-head**
read-out and every emission names its target. `output.modalities` is a **closed vocabulary the
validator bounds**, exactly as it bounds `kv_bytes` and `read_tokens`: at v1 the only accepted
values are `text` and `speech`; a `Schedule` naming any other modality is **rejected**; and a
`Schedule` naming `speech` while no `speech_output` head is resident is **rejected, not silently
degraded**. The reason is §9.9's B1 boundary, not tidiness — an unbounded modality field is a
second attacker-controlled dispatch key reached from untrusted input, which is the same object
`trace_id` was removed for. **`image` is the declared seam:** it is a value the validator will
accept the day a faculty exists to produce it, and refuses until then.

**Streaming output and a per-turn latency budget become runtime requirements the scheduler
respects, not aspirations.** Interaction is conversational — voice and text turns — so
`step_budget` gains two numbers the runtime enforces and the receipt records. `first_token_ms`
bounds time to the first emitted token of either stream, which is what makes the difference
between a turn and a wait. `speech_frame_ms` is the synthesiser's frame period, and it is the
harder of the two: it is a **deadline per workspace iteration**, not a budget for the turn, so a
schedule whose iteration cost exceeds it stalls the audio no matter how good the total latency
looks. `wall_ms` already bounds the whole turn; these two bound its *shape*. **Gate (W9,
extended):** the 10,000-input fuzz already asserts zero budget-violating graphs; it now also
asserts **zero graphs naming an unavailable modality**, verified by constructing one — a
`Schedule` requesting `speech` with the head evicted must be refused. A streaming run reports
**measured** `first_token_ms` and the fraction of iterations that missed `speech_frame_ms`, on
the deployment card. **What this deliberately does not do:** it does not make latency an
optimisation target for phase 2. It makes it a *recorded* quantity with a declared bound, so that
a phase-3 scheduler has something to be measured against instead of a preference — and so the
swarm and paging experiments (P5′s, P5′o) measure it rather than rediscover it.

**Runtime, ~120 lines.** Validate → for `i in range(max_iters)`: gather `{r : A[i,r]=1}`,
compute `cond_r` from `z_{i-1}` for conditioned regions, run those regions' `tokens()`
(position latents, not discrete ids — `[DEC-47]`)
**concurrently on separate streams** (they are independent within a stage, which is what the
architecture means), apply `TopK_{b_r}`, adapt, concatenate, run one workspace block, evaluate
halting, break. Cache each region's `h_r` when its `(ctx_r, cond_r)` is unchanged so a region
admitted twice is executed once — **the cache is per-request, bounded at `R × n_iter` entries,
and is destroyed at the end of the request.** It is keyed by attacker-influenced input, so an
unbounded or cross-request cache would be a second writable object with the same trust boundary
as `episodic_store` — and, unlike the store, **without** the scope partition, the byte capacity and
the scored eviction DEC-49 makes acceptance gates. **The distinction sharpens under DEC-49 rather
than dissolving:** the architecture is now allowed exactly **one** cross-request writable object,
the one with the contract; the `h_r` cache is per-request, bounded at `R × n_iter` entries and
destroyed at the end of the request, and any proposal to make it survive a request is a proposal to
build a second store without a contract `[A28 fixed]`.

**The `Schedule` is untrusted data.** It is derived from attacker-influenced input, so the
runtime validates it before executing it and the model is never trusted to bound its own
resource requests. Bounds and enforcement points are in §9.9. **Gate (W9):** a 10,000-input
adversarial fuzz emits **zero** graphs violating `Σ c_r·ctx_r ≤ B_kv` or `Σ b_r ≤ B_read`, and
the validator rejects every case in §9.9 — verified by *constructing* those cases, in the
`tests/test_guards_can_fail.py` pattern [V\*].


### DEC-52 — the I/O contract, stated once, with a must-have baseline that can be graded

**Why this belongs in the runtime-contract section and not in §1.** The operator's I/O intent was
readable in three places — §1.3's faculty list, §2.5's output block, and DEC-48's audio deferral —
and none of them said which parts are **required for the programme to be judged a success** and
which are intent. A contract that cannot be graded is a description `[OP:
csd-latent-space-reasoning-invariant.md, csd-training-placement-policy.md]`.

**The contract, as ruled: *"any in, any out."*** Every modality is encoded to latents (text tokens
→ `TextEncoder.tokens()`, image patches → `ViTEncoder.tokens()`, later audio frames → the
`auditory` seam's byte-identical interface), **unified in the shared latent space**, reasoned over
there under DEC-47, and emitted in the **decided** modality. The operator's own example is the
shape to design against: *"here is my screen and what I'm working on"* — a screenshot plus speech
plus typed text, unified as latents, **one** reasoned answer back.

**THE MUST-HAVE BASELINE, and it is what this programme is graded on:**

| | required at v1 | declared seam, deferred |
|---|---|---|
| **input** | **discrete text tokens** + **visual** (patch tokens) | audio frames (`auditory`, DEC-43/DEC-48) |
| **output** | **discrete text tokens** | speech-token stream (`speech_output`, DEC-44/DEC-48); image output |

*"The must-have is discrete-token and visual INPUT with discrete-token OUTPUT as the first
baseline; that is what proves the model is a worthwhile endeavour and not busted"* — so **W6's
integration test, W10's footprint and M0's readiness assessment are all judged on that baseline
alone**, and a shortfall on a deferred modality is not a finding against the design. Audio's later
addition is cheap **because of DEC-50**, not because it was designed around: add the submodel,
retrain the interconnect, then the unified pass.

**Output modality is a read-out decision, and a user constraint is BINDING.** `Schedule.output`
already carries the modality field (DEC-44 kept it declared precisely so this contract does not
change the runtime later). Two inputs decide it: **context** (what the reasoning is about) and an
**explicit user constraint**, and the second **overrides the first without exception** — if the
user says *"only text"*, the read-out emits text even where context would have chosen speech.
**This is enforced where the field is validated, not where it is used:** the `Schedule` validator
refuses an `output.modality` outside the request's declared `allowed_modalities` set, which makes
a violated user constraint a **refusal** rather than a surprising answer. **Verify it can fail:**
construct a request with `allowed_modalities: [text]` and a controller state that would select
speech, and assert the validator **rejects the `Schedule`** — the `tests/test_guards_can_fail.py`
pattern, and the reason the constraint is a control rather than a preference.

**What this contract does NOT license.** It does not reopen DEC-48 (audio stays deferred), it does
not make image output a v1 head, and it does not permit a modality to be chosen by the *reasoning
representation* — emission is a decoder choice, which is DEC-47's clause 3 and is restated here
only because a contract section that omitted it would read as permission.

## 2.6 DEC-19 — the training objective and where the signal comes from

**The compose phase IS training the interconnect.** P5 currently has no training step at all
[V\*]; this section is that step.

The tree has already *measured* the failure mode this design must avoid: a linear gate on real
corpus embeddings **collapsed outright to 1.0 / 0.0** on `tinystories` [V\*], and the Switch aux
term exists precisely because top-k softmax sends every token to the currently-better region.
Load-balancing losses treat the symptom. The cause is that **a gate trained by the task loss is
trained by a signal that is only defined for the branch it chose — the counterfactual is never
observed** [I]. So the recipe removes the cause rather than penalising the symptom.

**Phase A — dense. There is no gate on admission.** All regions on at `ctx_max`, `b_max`,
`A ≡ 1`, fixed `n_iter`. Only white matter, the adapters and the conditioning prefixes train;
regions are frozen. Record `a_r` for every held-out item.

**Collapse is *unrewarded*, not *unrepresentable* — and the difference is measurable**
`[A15 fixed]`. Revision 1 claimed there is *"no mechanism that could turn a region off"* and
then, three sections later, anticipated exactly that mechanism: §9.8 asks whether
`a_reasoning` goes *"near zero on reasoning-bin items"*. **The mechanism is the cross-attention
softmax itself** — the very tensor phase B distils. A region at `a_r ≈ 0` also receives
negligible gradient through its adapter and its conditioning prefix, so it never becomes
useful; phase B then regresses the controller onto a teacher that has already collapsed, and
phase B's *"full counterfactual"* justification is void precisely for the region that most
needs it. Every region **running** is not the same as every region **being observed**.

> **Phase-A pass condition (new):** `min_r mean(a_r)` over held-out items **≥ `η/R`**,
> evaluated at the receipt's own `R` — **3.0% at v1's `R = 5`** (DEC-49). The expression is the
> gate and the value is printed from it; revision 3.1 hard-coded 3%, revision 3.2 corrected it to
> 3.75% at `R = 4`, and revision 3.3 returns the *value* to 3% at `R = 5` **without returning to
> the constant** `[S31-16 fixed]` —
> the same floor the simplex uses elsewhere, with the per-region, per-iteration `a_r` histogram
> printed in the receipt. **If a region is below the floor, phase B may not distil against that
> teacher.** Named remedies, in order, each gated on the same floor: per-region attention
> temperature; adapter re-init; an attention-entropy floor. If none recovers the region, the
> receipt records `collapsed_in_phase_A: [<region>]` and phase B proceeds *without* that
> region's `a_r` as a target, which is a finding about the region (§9.8), not a free pass.

```
L_A = L_task( rank_head(frontal(z_N)), y )             # on reserve items (§5); see DEC-41
    + δ · L_unify                                      # DEC-20, the frontal faculty's own term
```

### DEC-41 — what `L_task` actually is at v1 `[A19 fixed] [A21 fixed]`

Revision 1 wrote `L_A = L_task(frontal(z_N), y)` and never said what `L_task`'s output space is.
`frontal(z_N)` is `[B, 512]`; §5.5's items are generative in shape (*"find the panel whose text
describes X, and return the code in the panel to its right"*); and no generative head exists at
v1 — `model/causal_lm.py` is curriculum step 4 (P6′), *"step 4, never step 1"*. A symbol with no
output space is not a specification, and phase A cannot be built from it.

> **The v1 output interface is a ranking head over a declared candidate set.**
> `rank_head: [B,512] → [B,512]`, scored against `k` encoded candidates by cosine, trained with
> softmax cross-entropy over the candidate set. `k = 32` per item: 1 gold, 30 hard distractors
> drawn from the same bin, **and 1 explicit `NULL` candidate**. The candidate-set construction
> is part of each reserve item's manifest, is frozen with the item, and is identical across
> every baseline in §2.7.1 — B1, B2, B2t, B3, B0 and B0u are all scored on the same candidate
> sets, or the comparison is not paired.

Three consequences, stated because they are load-bearing and unwelcome:

- **The composed mind at v1 is another bi-encoder.** B2 (an oracle convex combination of
  ranking scores) is therefore a genuinely strong null, and G2's comparison against the regions'
  own recall@1 becomes commensurable rather than approximate. That is a cost of v1's shape, and
  naming it is what stops the phase-2 result being read as more than it is.
- **The `NULL` candidate is the abstention output** the general / out-of-scope bin needs
  `[A21 fixed]`. No confidence head, no rejection target, no new parameters — the correct answer
  on a general-bin item is the `NULL` candidate, and it is scored by the identical metric.
  **Gate:** `NULL` recall on the general bin **> 0.50**, and `NULL` false-positive rate on every
  other bin **< 0.05**, both reported per bin. Without this the general bin has a label and no
  measurable output, which is how revision 1 left it.
- **Phase 3 replaces the head, not the loss.** When the language trunk's generative half exists
  (P6′), `rank_head` becomes one read-out among several and `L_task` becomes a sum. The
  interconnect does not change.

`L_unify`: the unified state must be **linearly sufficient** for each contributing region's own
answer as well as for the joint one — a small frozen linear probe from `frontal(z_N)` to each
region's native target, summed. It is what stops the workspace becoming a bottleneck that keeps
only the joint answer, and it is why unification is a trained faculty rather than a pooling
operation. Grafted from biology-first; it closes the judge's objection that the frontal read-out
otherwise has parameters but no objective and no falsifier of its own.

**Phase B — bootstrap the controller by distillation.** Freeze the workspace. Train the
controller to predict the *measured* `a[:,i,r]` and the realised budget utilisation from the
cheap input summary:

```
L_B = Σ_i Σ_r KL( softmax(a[:,i,r]/τ) ‖ softmax(ŝ[:,i,r]/τ) )  +  β·| Σ_r b̂_r − B_read |
```

This is ordinary supervised regression onto an **observed** quantity with a full counterfactual
— `a_r` is defined for every region on every item because in phase A every region ran. That is
why it cannot collapse, and it is why **no Switch aux term, no Gumbel noise and no RL appear
anywhere in this design.** Each removed relaxation is a removed way for phase 2 to fail
silently. **Gate:** Spearman `ρ(ŝ, a) > 0.6` on held-out.

**Phase C — sparse, with the dense model as teacher.** The controller emits `(ctx, b, A, halt)`;
the workspace runs only what is admitted; fine-tune end-to-end with a distillation term against
phase A's dense read-out plus the task loss. ε-exploration on the budgets (`p = 0.1`, perturb
one region) so the controller keeps seeing off-policy regions.
**Gate — FLOPs only.** Pass iff sparse is within **2% relative** of dense on the phase-A metric
at **≤ 50%** of region-token FLOPs. *If it buys accuracy at this phase, that is a bug report
against phase A*, not a win: it would mean the workspace was distracted by a region it should
have downweighted through attention, which is a learned-connectivity failure.

**Phase D — the scheduler is trained by the task loss.** *This phase is added here and is not in
the source proposal.* Two judges independently called the distilled controller a fatal flaw:
distillation is imitation, capped by the dense teacher, and *"it can never learn that a region
the dense model ignored is useful"* — which is a substantial retreat from a **learned scheduler,
closer to learned program synthesis**. The refutation is not to abandon distillation but to make
it an *initialisation*:

```
L_D = L_task + δ·L_unify + λ_flops · relu( FLOPs/FLOPs_target − 1 )
      # controller unfrozen; workspace and prefixes trainable; regions still frozen
```

Collapse is bounded structurally rather than by an aux term: the floored simplex guarantees
`η/R` of each budget to every *active* region, the `ctx_r ≥ ctx_min` floor keeps every declared
region observable, and phase D **starts from a non-degenerate schedule** — which is exactly what
the distillation bought. **Gate:** phase D must beat phase C on the composed metric, at equal or
lower FLOPs, **or it is reverted to phase C and the receipt says the scheduler is imitative.**
Two verdicts, never one (DEC-30).

**Overfit gate, grafted from biology-first and absent from the spine:** train/held-out gap on
the compose eval must be **< 5 points**. **27,424,039** interconnect parameters (§2.3, recomputed
for the v1 participant list) against a ~51k-item reserve is exactly where a silent memorisation
result comes from [I]. `[N10d fixed]` — this site and §9.6 carried the superseded 27.8M through
revision 2 while the appendix carried the correction; both prose sites now match §2.3.

**What is *not* in the objective, and why.** A hinge that rewards every region for being
individually necessary (`Σ_r max(0, m − (L^{-r} − L))`) was proposed and is **dropped**: the
gate that measures integration counts regions whose ablation flips an item, so such a loss is
gradient descent on the metric. A model trained to make every region's ablation costly will make
every region's ablation costly; that tells you the optimiser worked, not that information
crossed. This is the programme's own recurring defect — *a guard correct in reasoning and wrong
in scope* — reintroduced at the level of the objective [V\* judge finding, and I concur].

## 2.7 The integration-vs-dispatch experiment

The current P5 gate — *"composed beats best single region on a mixed set"* — is described by
`CORPUS-CONTRACT.md` as **nearly unfalsifiable** [V\*], and it is: on a bin-balanced set the
single-region baseline is bad off its own bin, so the composed model only has to route.

### 2.7.0 The experiment's own failure mode, and the rule that fixes it `[A1 fixed]`

**Revision 1 let the filter that built the data set both baselines it is graded against, and
said so as a convenience.** §5.3 admitted a row iff `max_r s_r < τ_lo` and
`oracle_late_fusion(s) ≥ τ_hi`; §2.7.1 then *defined* `B1 = per item max_r s_r` and
`B2 = oracle late fusion`; and §5.3 closed by noting this makes B1/B2 *"precomputed rather than
a separate experiment"*. Condition (1) **is** the statement `B1 < 0.35`, enforced per row.
Condition (2) **is** the statement `B2 ≥ 0.70`, enforced per row. So:

- **G1 (composed > B1) could not fail** — the data-construction step pinned the bar below 0.35,
  and revision 1 stated the intent outright (*"calibrated so the admitted set's B1 baseline sits
  at 0.3–0.4 — enough headroom for the interconnect to occupy"*). That is the definition of a
  test whose set is filtered until the baseline is beatable.
- **G3 (composed > B2) could barely pass** — admission guarantees B2 is right on essentially
  every admitted item, and **the discordant pairs where B2 is wrong are exactly what admission
  removed**. McNemar over discordant pairs then has almost nothing to work with.
- The two failures are **opposite in sign**, so they do not cancel and would not look wrong: the
  programme would report *"G1 passed comfortably, G3 failed"* and read it as *the mind routes
  but does not integrate*, when it is a property of the filter.

> **DEC-36 — admission never sets a baseline.** Three rules, all enforced in the W2b receipt:
>
> 1. **Calibrate on a split that is never graded.** Draw `τ_lo`/`τ_hi` from a **calibration
>    subset** (10% of candidates, disjoint at source-row granularity from everything graded),
>    freeze them, then admit the graded set with frozen thresholds. **Report what fraction of
>    the graded set *would* have been rejected** — that number is the honest measure of how much
>    the filter is doing, and it belongs in the receipt beside the result.
> 2. **B1, B2, B2t, B3, B0, B0u are recomputed on the graded split, on the composed task metric
>    (DEC-41's ranking head), by a separate forward pass.** The admission `s_r` are *never*
>    reused as a baseline. The saving revision 1 claimed is the exact thing that destroyed the
>    test.
> 3. **τ is stated in chance-normalised units per bin:** `(s − chance)/(1 − chance)`, with each
>    bin's chance level recorded in the manifest. Bin chance levels span two orders of magnitude
>    — 1/512 on the in-mixture diagonal, 1/200 on the tiny-imagenet probe, 0.0130 for
>    banking77, 0.0466 for go_emotions [V\*] — so a single raw `0.35` means a different
>    difficulty in every bin, and the per-bin scores G2 compares are not otherwise on a common
>    scale.
>
> **Verify it by making it fail — and it needs one field before it is constructible**
> `[N10a fixed]`. Revision 2 wrote *"recompute B1 from admission `s_r` instead of from the graded
> forward pass, and assert the receipt writer refuses to emit."* **A receipt writer cannot tell
> two floats apart by provenance.** So: **every recorded score carries `source: admission |
> graded`**, set at the point it is computed and never editable afterwards, and the receipt writer
> refuses to emit any baseline whose contributing scores carry `source: admission`. *Then* the
> test is: take a graded item, feed the writer a B1 assembled from `source: admission` scores, and
> assert it refuses. Every other verify-by-failing test in this document is constructible as
> written; this one was not, and the missing piece was one field.

### 2.7.1 Baselines, nested, all on the same paired items and the same candidate sets

Every baseline is scored on DEC-41's ranking metric over the item's frozen candidate set, on
the graded split, by its own forward pass (DEC-36).

| # | baseline | what it is | what beating it proves |
|---|---|---|---|
| **B0** | **untrained workspace, TRAINED READ-OUT** | random-init white matter, **frozen**, with `rank_head` trained on top — a random-features control. Same frozen regions, same budgets. `[A11 fixed]` | **THE UNTRAINED BASELINE.** Nothing learned *in the interconnect*. Revision 1's B0 had a random-init read-out too, so it scored at chance, so `Δ_A ≈ Δ_B ≈ Δ_AB ≈ 0` and the void condition `I₀ ≈ 0` held **by arithmetic, not by evidence** — the only guard against "the statistic is measuring the data, not the model" was a guard no dataset could trip. A trained read-out gives B0 non-trivial performance and makes `I₀` a number that *can* come out positive. |
| **B0d** | **known-dispatch positive control** | a top-1 router over pooled region outputs — same data, same steps, an architecture that is dispatch **by construction** `[A11 fixed]` | **The control that makes G3′ falsifiable at all.** Its `I` must be ≈ 0 under the same statistic. **If the dispatch control shows `I > 0`, the statistic is broken and that is the finding**, reported instead of a result. |
| **B0u** | **uniform connectivity, TRAINED** | cross-attention weights **frozen uniform from initialisation** (or cross-attention replaced by mean pooling over the KV bank), then trained — same params, same steps, same data `[A12 fixed]` | isolates *learned* connectivity from *having* connectivity. Revision 1 overwrote a trained attention distribution **at inference**, which is a train/test mismatch: every downstream layer was fitted against the learned distribution, so B0u would lose whether or not the learned connectivity carried information, and the win would mean nothing. B3 is already built with the right discipline one row down; B0u must match it. |
| **B1** | **oracle single region** | per item, `max_r score_r` **recomputed on the graded split**, chosen knowing the answer | stronger than any learned router can be. Beating it proves the mind is not equivalent to routing. |
| **B2** | **oracle late fusion** | best score over any per-item convex combination of the regions' **independent** outputs, weights chosen knowing the answer, **recomputed on the graded split** | every region ran alone; only scores were merged. **The ceiling on everything achievable without interaction.** |
| **B2t** | **trained matched-capacity late fusion** | a model of the same parameter count and training budget that can see **only the regions' pooled outputs** — no token surface, no cross-region attention `[A2 fixed]`. **It is by definition a second full white-matter-scale training run, and §4.3 budgets it as one** `[N7 fixed]` | B2 is an *oracle over frozen scores*; B2t is *"a model this size that cannot let anything cross between regions"*. **It is the honest null for the whole programme**, and it is the one an outside reader will ask for. |
| **B3** | **frontal-only control** | the identical workspace + frontal trained with all regions gated off, same params, same steps, same data | closes *"a big frontal module answered alone"*. Grafted from biology-first (I3). |

Oracle *routing* is a ceiling on picking one region. Oracle *late fusion* is a ceiling on
everything achievable without interaction. Using the weaker one is how a good mixer passes a
test it should fail [I].

### 2.7.2 The gate ladder

| gate | statement | test |
|---|---|---|
| **G1** routing works | composed > **B1** on the cross-faculty bin | McNemar on discordant pairs, `p < 0.01`, **Holm-corrected across the family** (§2.7.8) |
| **G2** no composition tax | composed ≥ each faculty's own single-bin score − 1 point | paired, per bin, in DEC-41's metric. **`episodic_store` is EXEMPT and the exemption is declared rather than assumed** `[DEC-49]`: it is non-parametric and has no standalone task, so it has no own-bin score to regress against, and a gate applied to a quantity that does not exist is the `residual_mlp` defect in gate form. **Its substitute is E2's gate** — the composed metric improves and the store draws non-zero attention mass — which is a stricter test than G2 because G2 only forbids harm while E2 requires benefit |
| **G3** integration | composed > **B2** **and** composed > **B2t** | McNemar, `p < 0.01`, Holm-corrected |
| **G3′** **THE PASS/FAIL** | the **synergy conjunction** below | see 2.7.3 |
| **G0** the mind is load-bearing | composed > **B3** and composed > **B0u** | paired |
| **G0d** the statistic is sound | **B0d**'s `I` CI **contains** 0, and **B0**'s `I₀` CI contains 0 | if either fails, **the experiment is void and is reported as void** — *unless* the `corpus: REDUNDANT` branch below fires, which is a different finding with a different remedy `[N6 fixed]` |

### 2.7.3 G3′ — the interaction term, with the void condition

Beating a baseline is indirect. The direct question is whether region A's contribution was
*computed conditional on* region B.

```
Δ_A  = perf(full) − perf(A ablated)
Δ_B  = perf(full) − perf(B ablated)
Δ_AB = perf(full) − perf(both ablated)
I(A,B) = Δ_AB − Δ_A − Δ_B
```

**Dispatch predicts `I ≈ 0`** — removing two independent contributions costs the sum of removing
each, additively, by construction. **Integration predicts `I > 0`** — superadditivity means A's
usefulness *depended on* B being present.

**Revision 1 stopped there, and that dichotomy omits the case this architecture is most likely
to be in** `[A2 fixed]`. Let A and B each be *independently sufficient* for an item — maximal
**redundancy**, the exact opposite of integration. Then:

```
Δ_A  = perf(full) − perf(¬A)     ≈ 0       (B still answers)
Δ_B  = perf(full) − perf(¬B)     ≈ 0       (A still answers)
Δ_AB = perf(full) − perf(¬A,¬B)  = large
I    = Δ_AB − Δ_A − Δ_B          ≫ 0       PASSES, MAXIMALLY
```

Redundancy produces the **largest possible** superadditive interaction under any thresholded
metric, because each single ablation costs nothing while the joint ablation costs everything.
**The pass/fail of the programme passed maximally on the configuration containing no
integration at all.** Content-swap does not close it: under redundancy A alone still answers, so
`Δ_A^swap ≈ 0` and `I^swap ≫ 0` too. Content-swap closes *presence-as-tag*, a different
degenerate solution, and it closes that one well.

And this is the likely regime, not a corner case: all participating text regions are GPT-2-BPE
encoders of the same width trained on overlapping English, and the tree's own measurement of the
shared stream is mean pairwise cosine **0.89** at entropy-effective rank **8.7 of 128** [V\*].
Highly overlapping representations are the textbook redundancy regime.

> ### DEC-37 — G3′ is the synergy conjunction, per pair
>
> **PASS for a pair `(A,B)` iff all three hold, each with its own 95% block-bootstrap CI
> excluding 0 (block by source row, §5.4):**
>
> ```
> Δ_A > 0        A is individually load-bearing
> Δ_B > 0        B is individually load-bearing
> I    > 0       and their contributions are superadditive
> ```
>
> True synergy requires both regions to be individually load-bearing **and** superadditive.
> Redundancy fails the first two clauses by definition.
>
> **The redundancy quadrant is named and is a FAIL.** `Δ_A ≈ 0, Δ_B ≈ 0, Δ_AB` large is reported
> in the receipt as `interaction: REDUNDANT` for that pair. It is **never folded into a mean
> `I`**, because a mean over quadrants is exactly how the failure hides.
>
> **G3′ (pass/fail):** on the cross-faculty bin, **≥ 7 of the 10 pairs** (`R = 5` participants,
> DEC-49) satisfy the conjunction, **under BOTH zero-ablation and content-swap**, with per-pair
> CIs reported and Holm correction across the family (§2.7.8). The null rate of "≥7 of 10 under a
> coin flip" is **0.1719** (`176/1024`) and is printed beside the result `[A14 fixed]`.
>
> **THE RULE THAT PICKED `k`, stated because otherwise every participant change is a chance to buy
> a pass.** Revision 1 wrote "≥6 of 10" at null **0.377**; revision 2 wrote "≥4 of 6" at null
> **0.344**. Both are *just over half*, and under that habit **adding a participant makes the gate
> easier**: at ten pairs, "≥6 of 10" would have been a **looser** criterion than the "≥4 of 6" it
> replaced, arrived at by adding a region. So:
>
> **THE RULE: the criterion's null rate is capped at a pre-committed 0.20, and `k` is the smallest
> value meeting that cap.** It is a **ceiling on the criterion**, not a comparison between
> revisions, because a comparison rule ratchets forever and eventually names a `k` nobody can
> reach. **The ceiling is stricter than both predecessors** — revision 1's 0.377 and revision 2's
> 0.344 **both exceed it** — which is exactly the property that makes a participant change unable
> to loosen the gate.
>
> Applied: at ten pairs, `k = 7` (0.1719), because `k = 6` is 0.377. **The rule binds every
> restatement in this document and the existing ones are re-derived under it in the same breath:**
> the **W7v-slip and E1-slip branches** (§5.4) become **≥ 5 of 6** at null **0.109** (`k = 4` is
> 0.344 and is refused); the **both-slip branch** stays **all 3 of 3** at **0.125**; and **A3's
> audio pre-specification** (§4.1) becomes **≥ 8 of 12** at null **0.194**, replacing revision
> 3.2's ≥ 5 of 8 at **0.363**, which the ceiling refuses.
>
> **Why this is not gate-inflation.** The bar rises because the *evidence* rises: ten declared item
> shapes over five participants is more, not less, than six over four, and a criterion whose null
> rate drifts upward as the mind grows is a criterion that rewards growth rather than integration —
> which is the exact substitution G3′ exists to detect.
>
> **Void conditions (both must hold, or the experiment is reported void):** **B0**'s `I₀` CI
> contains 0 *with a trained read-out* (`[A11 fixed]`), and **B0d**, the known-dispatch
> control, also shows `I` ≈ 0. B0's and B0d's absolute scores are printed next to `I₀` so a
> floor effect is **visible rather than inferred**.

> ### The G0d void has two causes with the same observable, and they need separating `[N6 fixed]`
>
> A11's fix — a trained read-out on B0, and B0d as a trained top-1 router — is right and stands.
> It created a hole. **DEC-37's own redundancy arithmetic applies to both controls.** On an item
> where two regions are each independently sufficient, ablating either costs a top-1 router
> nothing (it falls back) and ablating both costs everything: `Δ_A ≈ Δ_B ≈ 0`, `Δ_AB` large,
> `I ≫ 0` — **for the control that is dispatch by construction**. The same structure gives B0's
> random-features composition `I₀ > 0`. So a **redundant corpus** does not produce the
> `REDUNDANT` verdict DEC-37 built for it; it voids the whole experiment through G0d. Two
> findings, one observable, opposite implications for what to do next.
>
> **PRECEDENCE, pre-committed.** When G0d fails:
>
> 1. If B0d shows `I > 0` **and** the trained mind's pairs also land in the `REDUNDANT` quadrant
>    (`Δ_A ≈ 0`, `Δ_B ≈ 0`, `Δ_AB` large), the receipt reports **`corpus: REDUNDANT`** — an
>    **admission-filter failure against §5.3 condition (2)**, not `statistic: broken`.
> 2. If B0d shows `I > 0` **and** the trained mind's pairs do **not** show that structure, the
>    receipt reports **`statistic: broken`** and the experiment is void as written.
> 3. **`Δ_A`, `Δ_B` and `Δ_AB` are printed for B0d and B0, not only their `I`**, so a reader can
>    tell which branch fired without re-running anything. Without those three numbers the two
>    causes are indistinguishable on the page, which is how the wrong remedy gets chosen.
>
> **Remedy path for branch 1, so the void is not a dead end.** Re-admit against a **stricter
> condition (2)**: require that **no single region and no pair short of the full participant set**
> reaches `τ_hi`, not merely that `max_r s_r < τ_lo`. Report how many items survive. If fewer than
> 512 per cross-faculty pair survive, the finding is that **the corpus cannot support the claim**,
> reported as such — which is a real result about the reserve, not a failed run.

Note one direction of conservatism that survives: performance is bounded below by chance, which
caps `Δ_AB` and biases `I` negative.

### 2.7.4 The two controls that close the two degenerate solutions

| degenerate solution | how it passes a naive test | the control |
|---|---|---|
| **presence-as-tag** — the workspace reads region B's *presence* ("vision is on, so this is a screenshot task") as a domain label and answers from A | every ablation-based statistic goes positive; this is dispatch wearing a disguise and it is the *likeliest* way a composed model fakes integration | **content-swap:** replace B's tokens with **B's tokens from a different held-out item**. Presence, budget, sequence length and KV shape identical; content destroyed. **Require the DEC-37 conjunction under BOTH zero-ablation and content-swap.** |
| **budget-as-tag** — from phase C the controller emits `(ctx, b, A, halt)` *from the input*, so the **schedule** is itself a learned encoding of the input's domain, and the workspace reads "this is a screenshot task" off the budget vector with nothing crossing `[A18 fixed]` | content-swap does not perturb the schedule, so it passes cleanly — presence-as-tag returns one level up, wearing the scheduler | **two content-swap arms, both run and both reported:** (i) schedule **re-emitted** from the swapped input, (ii) schedule **frozen** from the original. The **gap between them is the measure of how much the schedule leaks domain identity**, and it is worth reporting in its own right. G3′ is graded on the arm with the frozen schedule, which is the conservative one. |
| **redundancy** — A and B each independently sufficient; nothing crossed, and the interaction term is *maximal* `[A2 fixed]` | `I ≫ 0` under every ablation style including content-swap | **DEC-37's conjunction** (`Δ_A > 0` and `Δ_B > 0` and `I > 0`), the named `REDUNDANT` quadrant as a FAIL, and **B2t**, the trained matched-capacity late-fusion null |
| **bigger-model** — the interconnect answers from its own 26.9M params and the write-back prefixes, with nothing crossing between regions | beats oracle routing without interaction | **B2** as the oracle ceiling, **B2t** as the trained null, **B3** the frontal-only control, **B0u** trained-uniform connectivity |

### 2.7.5 The causal test, fixed

A per-tract causal ablation was proposed in biology-first as the test its author *"would not
ship without"*, specified as zeroing an additive prior `A[i,j]` on the attention logits. Two
judges independently showed that this **does not sever the tract** — the `q·k` term is untouched,
and under an L1 sparsity penalty the ablation can be a null operation on a healthy model, so it
can report "not load-bearing" for a mind that is integrating perfectly. **It is grafted in fixed
form:**

> **I2′ — ordered-pair severance.** For the ordered pair `(i → j)` an item type requires, mask
> the workspace's cross-attention to region `i`'s keys **to −inf** at every iteration
> `< depth(j)`, and zero region `i`'s contribution to `j`'s conditioning prefix. Everything else
> — weights, schedule, budgets — is unchanged.
> **PASS iff** `Δ(cross-faculty items) − Δ(single-faculty items) > 0`, bootstrap 95% CI
> excluding 0.

This ties a win to a *specific* tract rather than to size, costs no retraining, and — unlike the
original — actually removes the path it names.

### 2.7.6 DEC-30 — the scheduler is a separate verdict

A composed model can pass every gate above with a scheduler that does nothing. So, separately,
and reported whatever they say:

| falsifier | statement | threshold |
|---|---|---|
| **S1** the schedule varies | `Var_x[active_r]` per region across inputs | must exceed the variance induced by dropout noise alone |
| **S2** the budget varies | `Var_x[ctx_r]`, `Var_x[b_r]` | not constant; per-region histogram reported |
| **S3** topology is used | entropy of the stage assignment over the eval set | `> 0.1 · log(n_iter)`; collapse to one stage means topology was not learned |
| **S4** the anatomy is readable | the `R×R` `Ĉ` matrix, and its sparsity | printed in the receipt; **void under W5b's no-write-back fallback unless the §2.4 variant is built**, and the receipt then says `topology: not demonstrated` `[A20 fixed]` |
| **S5** the schedule is not backdoored | per region, the change in admission probability under **single rare-token insertion** on held-out items `[T1 fixed]` | pre-committed threshold; a region whose admission moves beyond it on a rare token is a **FINDING**, not a metric. **This is the only gate that can fire when poisoned rows sit in both splits**, which is the case DEC-39's keyed split cannot cover (§5.6) |

**S1–S3 and S5 are falsifiers, not pass/fail-the-programme gates.** If the mind wins on G1–G3′
with a constant schedule, the *architecture* is vindicated and the *scheduler claim* is not,
**and the receipt must say so rather than reporting one number.** This is the single most
important measurement-hygiene graft in the document, and it is the only defence against the
failure where a dense always-on interconnect passes every quality gate while scheduling nothing.

**The receipt carries three verdicts, not two** `[T1 fixed]`: `integration:`, `scheduling:` and
`trigger_sensitivity:`. DEC-30's habit of refusing to collapse verdicts into one number is
exactly the right home for S5, because S5 is a gate and gates get waived — pre-committing its
threshold makes waiving it a visible decision rather than a silent one.

### 2.7.7 Cost, and what R = 5 changes

`episodic_store` is **required and built in phase 2** (DEC-49) and audio is deferred to a production
phase (DEC-48), so **R = 5 active participants → 10 pairs**. **Both the inclusion and the exclusion
are rulings, not schedule accidents**, which is what makes the count safe to build a pass/fail on
`[S31-1 fixed]`.

**A17 is not reopened by this, and the difference is the whole reason the store may be counted.**
Revision 1's *"≥6 of 10 positive"* was computed over 10 pairs of which **four involved
`episodic_store`**, a store that was empty at iteration 0 and had no write policy without
`salience`. Those four pairs were structurally `I ≈ 0`, so the stated criterion silently meant
*"all 6 real pairs positive"* — a harsher bar than the one written, arrived at by accident
`[A17 fixed]`. **What DEC-49 changes is not the count but the four pairs' constructibility**: the
store is built by E1 before W3 constructs items, and §5.4 declares **X7** and **X8**, recall-
dependent episode shapes in which a fact written in an earlier turn is required by a later one, so
each of the four store pairs carries **256 sealed items** rather than a structural zero. **This is
S1's own fix (b) — a participant whose pairs are counted — and it is only admissible because the
shapes exist.** The test for whether A17 has returned is mechanical and is stated so it can be run:
**no ablation pair may appear in the denominator without a row in §5.4's shape-to-pair mapping and a
non-zero sealed count.** Ten pairs, ten mapped, none zero.

```
10 pairs × 3 ablation styles (zero, swap-frozen-schedule, swap-reemitted-schedule)
         × 3,072 sealed eval items × 4 model variants (trained, B0, B0d, B0u)
         = 368,640 forward passes of a frozen ~86M model
```

At batch 512 that is 720 steps — still minutes on the 3090 Ti [V\*], and **twice revision 3.2's
184,320**, because both factors grew: ten pairs instead of six, and a sealed half of 3,072 instead
of 2,560 (§5.4).

**But that counts forward passes only, and the claim built on it was overstated** `[N7 fixed]`.
Revision 2 read this block as *"the decisive experiment is the cheap one, deliberately: the thing
most likely to be skipped should be the thing that costs least."* The **measurement** is cheap.
The **controls it is measured against are not**: B0's read-out, B0d, B0u and B2t are four trained
artefacts revision 1 did not require, and B2t is a full white-matter-scale training run by
definition (§2.7.1). B2t and B3 do not appear in the arithmetic above at all. §4.3 now budgets all
five at **≈ 3.1 white-matter-run-equivalents, ≈ 3–6 GPU-hours** `[I]`, which is the same order as
the entire region-retrain bill.

**Restated, honestly:** *the decisive measurement is cheap and its controls are the expensive
part, so the thing most likely to be skipped is a baseline, not the gate.* The mitigation is that
each baseline is named in §4.3 with an estimate and an owner-visible cost, so dropping one is a
visible decision rather than a silent saving.

### 2.7.8 One eval, many decisions — the split and the correction `[A14 fixed]`

Revision 1 read the same 5,120-item eval in W5, W5b, W6, W7, W8-C, W8-D, W9, W10 and P5′ — **at
least nine accept/revert or threshold decisions** — and ran ten paired bootstrap tests on it,
with no dev/test split and no multiplicity correction anywhere. W5b (enable/disable write-back),
W8-D (keep phase D or revert to C) and W7 (retrain or not) are model-selection decisions made on
the same items that produce the headline numbers, so the reported effect sizes are optimistically
biased by an amount nobody could estimate afterward.

1. **Split the compose eval in half at source-row granularity (§5.4).** A **dev half (3,072)**
   for *every* accept/revert and every threshold decision; a **sealed half (3,072)** opened
   **exactly once, for W6** (both were 2,560 before DEC-49 added two cross-faculty bins). The receipt records the timestamp at which the seal was broken and
   the git revision of the code that broke it. A second read of the sealed half is a new
   experiment with a new seal, and it is reported as such.
2. **Correct for multiplicity** across the interaction tests and the gate family: **Holm** for
   the family-wise claim (G1/G2/G3/G3′), **Benjamini–Hochberg** if the claim is the weaker
   *"most pairs interact"*. **Report per-pair CIs, never only the mean.**
3. **Print every criterion's null rate beside it.** ≥7 of 10 under a coin flip is 0.1719; the
   ratifying reader should not have to compute that. **And print the null rate of the criterion it
   replaced beside it**, because §2.7.3's selection rule is a comparison between the two and a
   reader cannot check a comparison with one of its terms missing.
4. **Size against the number of decisions.** 5,120 was derived (§5.4) for a *single* per-bin
   report at ±4.33 pp, not for nine sequential decisions plus the interaction tests. **DEC-49
   re-sizes it**: the compose eval is **6,144**, split **3,072 dev / 3,072 sealed**, because
   `R = 5` adds two reported cross-faculty bins and the 512-per-bin reporting floor is not
   negotiable (§5.4). **The per-bin half-width does NOT improve and saying so is the point:** the
   sealed half grows from 2,560 to 3,072 and the bin count grows from ten to twelve, so each bin
   still holds **256** items at **±6.1 pp**, exactly as before. What improves is the **aggregate
   cross-faculty comparison** G1/G3 actually run — 1,536 sealed items instead of 1,024, **±2.5 pp
   instead of ±3.1 pp**. A bigger eval that is spread over more bins buys nothing per bin, and a
   revision that reported the total and implied the per-bin figure had improved would be selling
   the same items twice. **The per-bin breakdown is still reported from the DEV half and labelled
   as such.** **The number
   of interaction tests rose from six to ten**, so the Holm family is larger and the correction is
   correspondingly harsher — stated here rather than discovered at W6.

---

# 3. Frontal cortex and thalamic gating

## 3.1 DEC-20 — frontal cortex: built now, inside white matter, with its own objective

The operator's requirement is that unification is *"a faculty in its own right, not a `sum()`"*
[V\*]. It is built now, as a named submodule of white matter: an attention-pooled read-out over
the workspace latents plus its MLP, **3,150,848 params** [V\*].

**Why not a separate peer region.** A separate unification region would need an input surface,
and its input surface would be the workspace — so it would be a read-out head on the workspace.
There is also a systems reason: it must attend over every participant, so it can never be paged
out, so it belongs in the resident set with white matter. **Declaring it a peer region would let
the scheduler deactivate the component that produces the answer** [V\* systems-first].

**Why it is nonetheless a faculty, and not a definitional dodge.** A judge correctly called the
"it would be a read-out head anyway" argument a dodge rather than an argument. The answer is to
give it the three things a faculty has:

1. **its own parameters** — 3,150,848, plus the `L=64 × 512` latent bank;
2. **its own objective** — `L_unify` (§2.6), which requires the unified state to be linearly
   sufficient for *each contributing region's own answer* as well as the joint one. Nothing else
   in the mind is trained against that;
3. **its own failure mode and its own ablation** — **B3**, the frontal-only control (§2.7.1),
   which is the only test that can show the frontal module has quietly become the model.

## 3.2 DEC-21 — thalamic gating: built now, and what exactly is thalamus-shaped

**What is built now:** gain control on *afferent bandwidth*. The controller decides how much of
each region reaches cortex — `ctx_r`, `b_r`, admission `A`, halting. In the brain, the thalamic
reticular nucleus applies gain control on a relay; it does not choose destinations. That is
exactly this job, and it is exactly the operator's characterisation of where discrete gating in
the brain lives.

**What it costs relative to plain learned connectivity:** ≈1,588,007 params (5.7% of white
matter) and one extra training phase. Its pass condition at phase C is FLOPs-only, so a
controller that fails simply reverts to dense at no accuracy cost.

**What is a declared seam and is NOT built:**

| seam | what it would be | trigger to build it |
|---|---|---|
| **content gating** | a thalamic gate that modulates *what information* passes, not just how much — a per-tract multiplicative mask on the KV bank | build only if S1–S4 show the bandwidth gate is being used *and* a measured failure mode requires content suppression. Not before. |
| **`affect` global gain** | the `[B,8]` valence vector multiplying `a_r` on every tract | build when the `affect` faculty is trained (it is a placeholder today) |
| **`salience` priority** | a learned scalar priority per emission, replacing the current `rank(s)` order | build when `salience` is trained; the consumer already exists |

Stating these as seams rather than building them is the operator's own discipline: *design the
seam, build when the constraint is measured*.

---

# 4. Replacement for P4 / P5 / P6

P4 as written — *"softmax gate + Switch aux load balance, `top_k=1`"* — builds option (a)'s
**alternative**, not a step toward it [V\*]. It is **deleted**, not amended. P5 — *"Composed
mind"*, with **no `method`, no `gate` and no `measures` key at all** — is **deleted**; its gate
was a result, not a task.

Hosts, as measured: **akula-prime `.98`** RTX 3090 Ti 24 GiB — region pretraining and phase-A
white matter (needs every region resident); **gpu5080 `.251`** RTX 5080 16 GiB — **the
deployment target**, so every sparsity, schedule and footprint measurement happens here;
**gpu1080ti `.243`** Pascal — RAG only, **no train / PoC CUDA**, so it gets BM25, corpus
construction and the CPU-side admission filter; **homelab `.170`** — NFS, metrics. 1 Gb/s
between the training hosts [V\*].

## 4.0 W1 FIRED AGAINST THE CENTRAL BET, AND W1d HAS NOW CONFIRMED IT. THE VERDICT IS SETTLED.

**Measured 2026-09-02 on the real production checkpoints on akula-prime, read-only.** The
pre-pool capture was verified **bit-exact (max abs 0.00e+00)** against `TextEncoder.forward()`
and `IJEPA.encode()` on an 8-item check per region, so the statistic is about the deployed
forward pass and not about a re-implementation of it. **W1d has since run and confirmed it**
(below): the verdict recorded in this section is **no longer provisional**, no region was
overturned, the retrains are licensed, and the penultimate-block fallback is dead.

**A gloss that applies to every sentence below** `[DEC-47]`: `tokens()` here means **position
latents — per-position hidden vectors, not discrete ids**. The whole of W1 and W1d is a question
about *latents*, never about a token vocabulary.

**Evidence, in the repo** `[N10b fixed]`. Revision 2 cited `scratchpad/exp-w1/…`, which is a
session directory a ratifier reading from the tree cannot open. The run's own outputs are now
committed alongside this document:

```
docs/design/evidence/w1-token-rank-2026-09-02/results.json     the run's output, byte for byte
docs/design/evidence/w1-token-rank-2026-09-02/measure_w1.py    the script that produced it
docs/design/evidence/w1-token-rank-2026-09-02/README.md        what was measured, on which
                                                               checkpoints by sha256, the exact
                                                               command, and BOTH rank definitions
```

Checkpoints are identified by sha256 in `results.json` and restated in that README. Summary also
recorded at `memory/csd-w1-token-rank-result.md` `[OP: csd-w1-token-rank-result]`. **[V]**

**BOTH effective-rank definitions are reported, because they disagree** `[N3 fixed]`. Revision 2
printed participation ratio only. `results.json` carries entropy-effective rank for every surface
as well, and **it reverses the sign for all four production regions**:

| region | PR pooled | PR token | **PR ratio** | H pooled | H token | **H ratio** | per-item PR | production |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `code` | 42.6 | 28.1 | **0.66×** | 115.0 | 138.9 | **1.21×** | 4.4 | yes |
| `compress` | 45.3 | 35.6 | **0.78×** | 126.9 | 147.5 | **1.16×** | 8.5 | yes |
| `retrieve` | 40.7 | 40.7 | **1.00×** | 117.7 | 142.9 | **1.21×** | 5.1 | yes |
| `vl_latent` | 18.1 | 23.5 | **1.30×** | 129.8 | 239.1 | **1.84×** | 8.3 | yes |
| `compress_repo_local` | 7.1 | 14.2 | **2.00×** | 55.2 | 92.4 | **1.67×** | 4.2 | **no** `[N4 fixed]` |

- **`*_pr_rank` — participation ratio.** `(Σ s_i²)² / Σ s_i⁴` over the singular values of the
  column-centred matrix (`measure_w1.py:197-211`). **This is the definition W1's rule was
  pre-committed against, in revision 1, before the measurement.** It weights the *spread* of the
  spectrum: a few dominant directions pull it down hard.
- **`*_entropy_rank` — entropy-effective rank.** `exp(H)` of the entropy of the linearly
  normalised singular-value spectrum (`measure_w1.py:214-222`), i.e.
  `cogsyndelta.eval.benchmark.effective_rank` — **the definition behind this document's own
  "8.7 of 128" prior** (§2.7.3). It weights the *tail*: many small directions raise it.

**The verdict is definition-dependent, and that is stated plainly rather than argued away.**
Under participation ratio three of four production regions fall **below** 1.0 and all four sit
inside the pre-committed ≤1.5× dead band: **bet dead**. Under entropy-effective rank all four
exceed 1.0 and `vl_latent` at 1.84× is **outside** the dead band entirely: **bet not dead**. The
same artefact supports both readings. Participation ratio is the statistic the rule was written
against and the one reported as the verdict; the design's reason for preferring it is that the
workspace cross-attends over *directions*, and PR measures how many directions carry the variance
while entropy rank is raised by tail mass a softmax will not find. **That was an argument, not a
measurement, and it is exactly what W1d was for.** W1d has now run, its rule named no rank
definition, and it **CONFIRMED** the verdict for every region — so the disagreement between the
two columns above is now a recorded curiosity rather than an open decision.

**The fifth row is not production and is reported anyway** `[N4 fixed]`. `compress_repo_local` is
the same architecture as production `compress` at a different checkpoint — **step 2,000 against
8,000, `max_len` 256 against 96** — so it is **undertrained and non-production**, and it is not
admissible as evidence for or against the retrain. It is printed because production `compress`
measures 0.78× and this checkpoint measures 2.00×, four thousandths under the pre-committed
"≥2× ⇒ no retrain" bar: **a 2.5× spread between two checkpoints of one region, on the statistic
that licenses retraining every region.** That spread is the single best available sizing of the
risk W1d exists to close, and hiding it while concluding "all four production regions land inside
the dead band" would be true and misleading at once.

**The per-item flag** (`< 8` directions) also fires, for `code` (4.4) and `retrieve` (5.1): a
96-position sequence carrying four usable directions is not a sequence the workspace can attend
over. It is a participation-ratio quantity too, and inherits the same caveat.

**Three collateral results, recorded because they are load-bearing elsewhere:**

- **Cross-region pooled CKA max 0.336** (`code`↔`compress` 0.205, `code`↔`retrieve` 0.206,
  `compress`↔`retrieve` 0.336). The regions are genuinely distinct, so W1c's anisotropy concern
  and §9.1's *"one shared subspace"* fear are not what killed the bet. **This is also the first
  direct evidence about A2's redundancy regime and it cuts the *reassuring* way**: distinct
  pooled representations make pure redundancy less likely than the 0.89-cosine stream figure
  suggested — but CKA of representations is not sufficiency for a task, so DEC-37's conjunction
  stands as written.
- **Training worked.** Pooled mean pairwise cosine falls from 0.84–0.98 untrained to 0.02–0.36
  trained. The verdict is about token-vs-pooled, not about whether the regions learned.
- **A baseline caveat that invalidates one untrained comparison.** The three text regions share
  `seed = 0` and identical `TextEncoderConfig`, so their "untrained" models are **literally the
  same weights**; untrained cross-region CKA is `1.0` **by construction** and must never be
  cited as a baseline. **Standing rule from this point: vary the untrained seed per region in
  every future baseline** `[OP: csd-w1-token-rank-result]`. W2c is where that starts.

**The mechanism is the one this document predicted** (§4.2, §9.1): mean pooling gives
`∂pooled/∂h_t = m_t/Σm`, identical for every unmasked position, so InfoNCE supplies no
positionally differentiated gradient and shapes only the *mean* of the token sequence. The
pre-pool tokens end up near-copies of that mean. The prediction was right; the bet it was
attached to was wrong.

### W1d HAS RUN, AND IT CONFIRMS W1. The verdict stops being provisional.

**Measured 2026-09-02 on akula-prime, read-only, 116.5 s total wall clock**, against the **same
production checkpoints** (sha256 identical to W1's README) with the forward capture again verified
**bit-exact, max abs 0.00e+00**, this time including the mid-loop penultimate hook. The evidence is
committed beside this document and beside W1's:

```
docs/design/evidence/w1d-readout-probe-2026-09-02/
  results.json      sha256 c88c2c08566a2e1a8ac038813dbb64ecdffadfa7217598e02273cdbd782b6c13
  measure_w1d.py    sha256 711fbcc77c18c940fe6bef3c4d7e219580c46b8512a90ad98e5ac01294bccbc4
  run.log           sha256 f222992ccad5f5c192df2ce1859f33e6249e3badcc68f677ead4f3d93ee6f8f0
  SHA256SUMS        the three lines above, as written by the run
```

Summary also recorded at `memory/csd-w1d-readout-probe-result.md`
`[OP: csd-w1d-readout-probe-result.md]`. **[V]**

**The rule, restated before the numbers, because it was pre-committed in revision 2 and names no
rank definition.** Arm **(a)** is a matched read-out over the region's own **final-block position
latents**; arm **(b)** is the identical read-out over `pool()` **broadcast to the same `T`**, so
the arms differ *only* in whether the sequence carries information; arm **(c)** is the same
read-out over **penultimate-block** activations. Arm (a) beating arm (b) by **≥ 2 points
OVERTURNS** W1; **within 2 points CONFIRMS** it. **Verify-by-failing:** arm (a) against arm (a) at
a different seed must report CONFIRMED.

| region | metric | arm (a) latents | arm (b) pool-broadcast | Δ(a−b) | **verdict** | seed self-check |
|---|---|---:|---:|---:|---|---|
| `code` | recall@1 | 96.68 | 96.68 | **+0.00** | **CONFIRMED** | **pass** (+0.20) |
| `compress` | recall@1 | 62.70 | 64.45 | **−1.76** | **CONFIRMED** | **pass** (−0.98) |
| `retrieve` | recall@1 | 61.33 | 67.38 | **−6.05** | **CONFIRMED, instrument NOISY** | **FAIL** (−2.54) |
| `vl_latent` | top1 | 5.47 | 8.40 | **−2.93** | **CONFIRMED** | **pass** (−1.17) |

**No region is OVERTURNED.** In **three of four the sequence-blind arm BEAT the token arm** —
which is stronger than *"no difference"* and is exactly what the mean-pool-gradient mechanism
predicts: if InfoNCE shaped only the mean, a read-out handed the mean at every position has lost
nothing and has an easier optimisation problem. `code`, `compress` and `vl_latent` are **clean
CONFIRMED**, each passing its own seed self-check.

**`retrieve` is CONFIRMED-BUT-NOISY, and it is flagged rather than counted as clean** `[V]`. Its
verify-by-failing arm — the same arm (a) at a different seed — moved **2.54 pp**, which
**exceeds the 2-pp threshold the decision rule is written against**. An instrument that moves
further under a seed change than the effect it is asked to resolve has not confirmed anything
cleanly, whatever direction the point estimate points. The −6.05 pp gap is far outside that
spread and points the same way as the other three, which is why the verdict is *confirmed* rather
than *void*; the flag is what stops the row being cited later as if it were clean. `results.json`
records the same judgment in its own words: *"GATE DID NOT SELF-CONFIRM — seed-to-seed spread ≥
2pp, instrument too noisy for a 2pp threshold."*

> **THE PENULTIMATE-BLOCK FALLBACK IS DEAD. `L_token` attaches at the FINAL block.**
> Arm (c) was **worse or within noise in every region** — `code` 95.51 (−1.17), `compress` 58.79
> (**−3.91**), `retrieve` 58.98 (**−2.34**), `vl_latent` 6.84 (+1.37 against an arm (a) of 5.47
> that arm (b) already beat at 8.40) — and its **rank is far lower everywhere** (table below).
> There is no evidence for it in any region and direct evidence against it in two. **Revision
> 3.1's option (2) fallback is struck**: a W4/W7 region that fails its gate goes to §9.14's pivot,
> not to the penultimate block. This is the question old-W7 carried, answered for the cost of a
> probe rather than a white-matter run, and answered *no*.

**Both rank definitions, on the probe's own input activations, as the row required**
`[N3 fixed]`:

| region | PR final | PR penult | PR pooled | H final | H penult | H pooled | n latents |
|---|---:|---:|---:|---:|---:|---:|---:|
| `code` | 28.09 | **7.37** | 42.63 | 138.87 | 101.47 | 114.98 | 26,757 |
| `compress` | 35.57 | **13.91** | 45.33 | 147.49 | 102.85 | 126.86 | 9,636 |
| `retrieve` | 40.65 | **19.56** | 40.71 | 142.86 | 98.39 | 117.72 | 5,114 |
| `vl_latent` | 25.42 | **20.95** | 23.21 | 242.55 | 234.98 | 136.46 | 32,768 |

- **PR = participation ratio**, `(Σ s_i²)² / Σ s_i⁴` over the singular values of the
  column-centred matrix — `measure_w1d.py:246-261`, where the formula is **reproduced rather than
  imported**, precisely so W1d never depends on a session-scratchpad script.
- **H = entropy-effective rank**, `exp(H)` of the entropy of the linearly normalised
  singular-value spectrum — `cogsyndelta.eval.benchmark.effective_rank`
  (`src/cogsyndelta/eval/benchmark.py:138-155`), evaluated over the **whole** surface rather than
  its 2,048-row default sample.

**Text-region rank numbers reproduce W1's README bit-identically**, which is the check that the
two runs are measuring the same object. The two definitions still disagree in the same direction
they did in W1 — and that no longer decides anything, because W1d's rule reads **task
performance**, not rank. What the rank table *does* decide is the penultimate question: PR falls
by a factor of 2–4 at the penultimate block for all three text regions.

**Instrument caveats, carried forward rather than buried, because they bound what this result may
be asked to support:**

- **An untrained cross-attention read-out is ≈ uniform attention**, and uniform attention over a
  sequence is a **linear function of `pool()`**. So the untrained baselines start high by
  construction, and the two arms start closer together than a naive reading of "trained probe"
  suggests. The comparison is still valid — both arms inherit it identically — but the *absolute*
  numbers are not a measure of how much the sequence carries.
- **600 steps of read-out training slightly REDUCED recall for `code` and `compress`** against
  their untrained read-outs. **The instrument is coarse.** It is a decision procedure for whether
  the sequence carries **more than its mean**, not a precision measurement of how much.
- **`retrieve`'s seed spread (2.54 pp) exceeds the 2-pp rule.** Carried as a flag on every later
  citation of that row.
- **`vl_latent`'s sample is seeded and class-stratified, and it had to be.** tiny-imagenet's
  parquets are **class-grouped** — 500 contiguous rows per class in train, 50 in valid, ascending
  by label — so **the first 512 valid rows span 11 of 200 classes and the first 4,096 train rows
  span 9**. **A file-order prefix is never a valid sample of that dataset**, and that is a
  standing rule from this point, the same shape as `retrieve`'s `load_pairs` prefix bug that B4
  exists to catch. The consequence is stated rather than smoothed: `vl_latent`'s rank numbers are
  **not** bit-identical to W1's README, because W1 took a prefix there and W1d did not.

**Consequence — the token-aware retrains are LICENSED, and the order is fixed:** **W4** (the
`memory` merge, designed as the first token-aware retrain), then **W7a** (`language_code`,
`reasoning`), then **W7v** (`visual`, with the 128-px rebuild). The retrain objective must shape
token-level structure **at the final block**, which is now an evidenced choice rather than a
default. The gate keeps its shape and states its definition: **token-global participation-ratio
rank ≥ 2× pooled participation-ratio rank** — the W1 statistic, measured by the W1 harness on the
same 512 held-out items — **AND no receipt regression > 1 point.**

> ### DEC-35 — the consequences, pre-committed and now owed
>
> **AMENDED IN REVISION 3.2, because W1d has run.** Clauses 1 and 3 are resolved by measurement
> and clause 4's fallback is struck; the amendments are written into the clauses rather than
> appended, and the superseded wording is named where it mattered.
>
> 1. **W1 is closed as a measurement, and its verdict is now SETTLED, not provisional.** It is
>    not re-run and it is not a `todo`. **W1d CONFIRMS it** — clean for `code`, `compress` and
>    `vl_latent`, **CONFIRMED-BUT-NOISY** for `retrieve`, **no region OVERTURNED**. Revision 3's
>    *"PROVISIONAL PENDING W1d"* is discharged, by the step it was pending on, in the direction
>    the design did not want.
> 2. **A token-aware retrain is MANDATORY BEFORE W5 and is now LICENSED** — W4 first, then W7a
>    and W7v. Wiring the interconnect to frozen position latents as they are today would be
>    training a workspace to cross-attend over `T` near-copies of a mean. The requirement was
>    never softened by the entropy-rank disagreement, and W1d has now removed the disagreement's
>    standing to soften it: the confirmation was run, and it confirmed. **The retrain objective
>    attaches `L_token` at the FINAL block** — W1d measured the penultimate alternative and found
>    it worse or noise everywhere.
> 3. **W1d WAS THE DECIDING STEP, it was run BEFORE the retrain was spent, and it decided**
>    `[A9 fixed] [N3 fixed]`. Its pre-committed rule depended on **no rank definition at all** —
>    a task-performance difference between two arms differing only in whether the sequence
>    carries information — which is the property that let it adjudicate a disagreement the rank
>    statistics could not. It cost **116.5 seconds**, not the estimated hour, and it could have
>    stopped the most expensive item in phase 2. It did not: three of four regions had the
>    **sequence-blind arm win**. *"If W1d is skipped, the retrain is not licensed"* is now moot in
>    the only way a pre-commitment should ever become moot — **the step was run**.
> 4. **W7 is no longer "re-run W5"** `[A8 fixed]`. Revision 1's rule required W7 before W5 while
>    defining W7 as a re-run of W5, and W7's remedy clause needed *G3's margin*, which is
>    measured in W6, after W5. It was unexecutable as written. W7 is now a region-training row
>    with its own objective and its own gate (below), and the penultimate-block question it used
>    to carry is answered inside W1d for the cost of a probe instead of a white-matter run —
>    **answered NO**, so the fallback that question fed is struck (below).
> 5. **The phase-2 cost estimate is re-derived** in §4.3. "Phase 2 retrains nothing" is dead
>    `[A30 fixed]`; the affordability argument is now a budget. **Revision 3.2 removes the audio
>    line from that budget** (DEC-48) and replaces W1d's estimate with its measured 116.5 s.
> 6. **The two W1d caveats travel with every citation of this result** (above): an untrained
>    cross-attention read-out collapses to a linear function of `pool()`, so the instrument is
>    coarse and its untrained baselines are high; and `retrieve`'s seed spread exceeds the
>    decision threshold, so its confirmation is flagged, never cited as clean.

### The retrain objective, specified `[A8 fixed]`

A gate cannot be pre-committed against an objective that is not written down. Three options; the
first is adopted, the second is the fallback within the same row, the third is the §9.14 pivot.

**(1) ADOPTED — InfoNCE + bandwidth-limited decorrelation + a per-token term.**

```
L = InfoNCE( pool(h), pool(h⁺) )                      # unchanged: the receipt stays comparable
  + γ · L_decorr(h)                                   # off-diagonal penalty on the token-position
                                                      #   covariance (VICReg/Barlow-Twins shape):
                                                      #   drive cross-position redundancy down
  + ζ · L_token(h)                                    # a gradient AT EVERY POSITION, which
                                                      #   InfoNCE structurally cannot supply
```

`L_token` attaches at the **FINAL block** — decided by W1d, not left open (above). It is one of
two shapes, and the choice is recorded in the receipt:

- **(a) masked-token prediction** — a BERT-style MLM head over the region's own vocabulary,
  discarded after training. Supplies a positionally differentiated gradient *by construction*,
  which is precisely the property mean-pooled InfoNCE lacks.
- **(b) late interaction / per-token contrastive (ColBERT-style MaxSim)** — the contrastive
  score becomes `Σ_q max_d ⟨h_q, h_d⟩` rather than `⟨pool, pool⟩`, so **InfoNCE itself** becomes
  positionally differentiated and no auxiliary head is needed. Cheaper at train time, more
  expensive at eval time, and it changes what `pool()` means for the receipt — which is why the
  gate below requires the pooled receipt metric to survive.

**(2) FALLBACK — STRUCK IN REVISION 3.2, on W1d's evidence.** Revision 3.1 held a fallback of
*"apply the same objective at the penultimate block rather than the final block"*, to be chosen on
W1d's probe. **W1d measured it and it lost in every region** — worse in `compress` (−3.91) and
`retrieve` (−2.34), worse in `code` (−1.17), and in `vl_latent` its +1.37 sits under an arm (b)
that beat both — with participation-ratio rank **2–4× lower** at that block for every text region.
There is no evidence for the fallback and direct evidence against it, so it is **deleted rather
than kept as a comfort**: a region that fails the gate goes to option (3).

**(3) PIVOT, if the row fails outright:** train regions and the interconnect **together from the
start** — the operator's phase 3 before phase 2. §9.14 already names this as the pivot; W1's
result makes it materially more likely, and its cost (region receipts stop being comparable,
region-per-host training is lost) is why it is not the first move.

> **W4/W7 GATE, pre-committed, both clauses required per region:**
>
> 1. **`token_global_pr_rank ≥ 2.0 × pooled_pr_rank`**, measured by the **identical W1 harness**
>    on the **identical 512 held-out items** with the same seed — i.e. the region clears the
>    threshold W1 pre-committed as "no retrain needed", measured the same way it failed.
> 2. **The region's own receipt metric does not regress by more than 1 point**, and the
>    `banking77` damage-detector probe does not regress by more than 2 points.
>
> **Both ranks are participation ratio, and the receipt says so** — the definition is named in
> the gate rather than inherited from whichever column a reader looks at first, which is the
> ambiguity N3 found and W1d's own table (above) prints out of.
>
> A region that clears (1) and fails (2) has traded its faculty for a token surface and is
> reverted. A region that clears (2) and fails (1) **goes straight to option (3), the pivot** —
> option (2) was the penultimate-block fallback and W1d killed it.
> **Verify the gate can fail:** run it against the *pre-retrain* checkpoint and assert it
> reports FAIL — the harness that says 0.66× today must still say FAIL when handed 0.66×.

### W4 HAS RUN AT THE PRE-REGISTERED CONFIGURATION. THREE GATES PASS, TWO FAIL, AND THE GATES ABOVE ARE NOT EDITED BY THIS REVISION `[DEC-68]`

Everything in this subsection is read out of the receipts named beside it. No number here is
restated from prose, and no gate above is changed — **the row was pre-registered so that its
outcome could not be argued with afterwards, and the first thing a document is tempted to do when
a pre-registration fails is to improve it.**

**Four runs, one changed variable at a time.**

| run | receipt | code | batch | terms | (a) | (b) | (c) | (d) | (e) |
|---|---|---|---|---|---|---|---|---|---|
| production | `/akula-data/csd/receipts/memory-20260903T164611Z.json` | `4d2d886` | 512 | on | **FAIL** | **FAIL** | **FAIL** | pass | **FAIL** |
| probe A | `w4-probes/w4-probe-batch256/memory-20260903T170934Z.json` | `ef3e180` | 256 | on | **FAIL** | **FAIL** | **FAIL** | pass | **FAIL** |
| probe B, control | `w4-probes/w4-probe-control512/memory-20260903T170747Z.json` | `ef3e180` | 512 | **off** | **FAIL** | **FAIL** | **FAIL** | pass | **FAIL** |
| **pre-registered** | `w4-chunked/run-1280/memory-20260903T184441Z.json` | `eb735ab` | **1280** | on | **PASS** | **PASS †** | **FAIL** | pass | **FAIL** |

**†** (b) is scored PASS by the harness on a value whose entire margin over the floor is float32
representation error — see the clause-by-clause reading below, and **OD-17**, which is where the
`>`-versus-`≥` question is answered rather than here.

**The pre-registered run, clause by clause** (batch 1280, `token_loss_chunk` 512, 4,000 steps,
`max_len` 96, seed 0, `token 0.1 / decorr 0.1`, 16,021,248 params, 1,586.7 s on the 3090 Ti):

- **(a) beats both parents — PASS.** Held-out `recall@1` **0.8535** against `compress` **0.7754**
  and `retrieve` **0.7559**; graded spearman **0.7893** against `compress`'s **0.7588**. Both
  sub-clauses pass, which the batch-512 run's graded number (0.7463) did not.
- **(b) full-pool floors — THE HARNESS SCORED IT PASS, AND THE ENTIRE MARGIN IS FLOAT32
  REPRESENTATION.** `recall@10` **0.200** against a **0.20** floor, MRR **0.1168** against a
  **0.10** floor, on the 57,638-passage FiQA pool. **The clause is STRICT** — pre-registered as
  `recall@10 > 0.20`, printed unchanged in W4's row, and implemented strictly at
  `src/cogsyndelta/eval/beir_fiqa.py:476` (`passed = recall > recall_at_10_floor and mrr >
  mrr_floor`). **The receipt's stored value is `0.20000000298023224`, which is exactly
  `float(numpy.float32(0.2))`** [V, the receipt]: the underlying float32 recall **is** 0.2 and the
  whole 3e-9 margin is float32 representation error, not measurement. **So this document records
  two things and settles neither here.** *The receipt says `passed: true` and that is what is
  reported.* *Under a strict reading of the pre-registration (b) is an EXACT TIE and therefore a
  fail, which makes the run **two passes and three fails**.* **Whether the floors are `>` or `≥`,
  and at what precision, is OD-17's to answer alongside (c) and (e)** — reading a bar as `≥` after
  the number came in is precisely the move this subsection's own preamble forbids, and a
  measurement whose verdict flips on 3e-9 of float32 is not a verdict this document may take for
  itself.
- **(c) beats BM25 — FAIL.** `recall@10` **0.200** against BM25's **0.440**; MRR **0.117** against
  **0.308**; `recall@100` **0.424** against **0.662**. **The gap is a factor of 2.2, not a
  rounding**, and it is the same gap at every batch size measured.
- **(d) beats random init — PASS.** **0.200** against the untrained **0.000**.
- **(e) §4.0's retrain gate — FAIL on clause 1, PASS on clause 2.** Final-block
  `token_global_pr_rank` **94.75** over `pooled_pr_rank` **78.29** = **1.2102**, against the
  absolute **2.0×**. The regression clause is **0.0** — nothing regressed at all.

**The negatives curve, which is the one axis that moves the failing number.** Three runs differing
only in batch, which in an in-batch-negatives regime *is* the negative count:

| batch | held-out `recall@1` | graded spearman | full-pool `recall@10` | full-pool `recall@100` | rank ratio |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.5996 | 0.6775 | 0.030 | 0.104 | 1.586 |
| 512 | 0.8145 | 0.7463 | 0.098 | 0.280 | 1.348 |
| **1280** | **0.8535** | **0.7893** | **0.200** | **0.424** | **1.210** |

**Reading, and it is the load-bearing one:** full-pool `recall@10` moves **6.7×** across a **5×**
change in negative count while the held-out number moves 25 points — the in-batch-negatives count
is the dominant lever on the metric gate (c) is defined over, and the batch-512 production run
therefore could not have been judged against this gate at all. **That is why the pivot was not
triggered on 2026-09-03 afternoon**, and it is why the pre-registered run had to be made to fit
before any verdict: the chunked, checkpointed cross-entropy over masked positions
(`PretrainConfig.token_loss_chunk`, exact, 30 equivalence tests on loss and every gradient) is what
brought batch 1280 from an OOM before step 0 to a completed run
(`docs/design/evidence/w4-masked-token-loss-2026-09-03/`).

**The objective's effect at production length, which the 50-step control arm could not measure.**
The control arm at batch 512 (both weights 0, `w4-probe-control512`) against the production arm at
the same batch:

| arm | held `recall@1` | graded | full-pool `recall@10` | rank ratio |
|---|---:|---:|---:|---:|
| control (0 / 0) | 0.7754 | 0.7313 | 0.100 | **1.0851** |
| both on (0.1 / 0.1) | 0.8145 | 0.7463 | 0.098 | **1.3475** |

**Three things this settles.** **(1) The token-aware terms are not a no-op at production length:**
+3.9 points of `recall@1` and +1.5 of graded spearman. **(2) The 2.0× clause DOES discriminate at
4,000 steps** — 1.09 against 1.35 — which is the opposite of what the 50-step control arm showed
(2.02 against 2.24, both clearing) and is the reason that earlier result was recorded as
*not discriminating at 50 steps on that harness* rather than as a verdict. **(3) The full-pool miss
is INDEPENDENT of the objective:** 0.100 control against 0.098 with the terms on. Gate (c) is not
failing because of `L_token` or `L_decorr`.

**What the failures are actually about, stated as two different problems rather than one.**

- **(c) is a distribution-and-scale problem, and it is measurable.** FiQA is **14,131 of 782,959
  training pairs — 1.8% of the corpus** [V, the receipt's own `corpus.sources`] — and the eval is
  the full 57,638-passage FiQA-only pool. **The gate has no parent baseline**: neither `compress`
  nor `retrieve` was ever measured against that pool, so *"the merge degraded retrieval"* and
  *"an unmerged `retrieve` would have failed the same bar"* are indistinguishable from the receipts
  that exist. **That absence is itself a finding**, and it is what OD-17's amendment addresses.
- **(e) is a threshold problem, and the threshold was chosen before any of this was measured.**
  W1's own pooled-versus-token ratios were 0.66× to 1.30×; 2.0× was set as *"the threshold W1
  pre-committed as no-retrain-needed"*. The retrain moves the ratio from **1.09** to **1.21–1.35**
  depending on batch — a real, repeatable, control-separated movement that lands nowhere near 2.0×.

**And what this document does about it: nothing, yet.** Both branches are put to the operator in
§8 as **OD-17**, with what each costs. **The pivot of §9.14 remains pre-committed and remains the
fallback**; §9.1's *"a region that fails its retrain gate goes straight to §9.14's pivot"* is
untouched by this subsection.

---

## 4.1 The rows

**One column is new, and it is the one revision 1 did not have.** `blocked_by` was absent from
W0–W10 entirely — only P5′ and P6′ carried a dependency — in a document whose central organising
claim is that the interconnect is late *by dependency* `[A7 fixed]`. The `s_r` invalidation rule
below is the constraint that column exists to express, and it is stated as a rule rather than
left as a note.

> **`s_r` INVALIDATION RULE.** Any change to any region's weights invalidates **every** recorded
> `s_r`, **every** admission decision derived from them, and (per DEC-36) every baseline
> recomputed against them. Therefore: **the region set is frozen before admission runs.** W4,
> W7a and W7v all precede W2b. If a region changes afterwards, admission re-runs and the receipt
> records which items entered and which left. **Revision 3.1 added two regions and two more ways
> to trip this rule; DEC-48 removes both.** `auditory` and `speech_output` are **deferred to a
> production phase**, so the **four encoding regions** frozen before admission are unchanged by
> audio. That is a **ruling**, not a scheduling race that had to be won before W2b
> `[S31-1 fixed]`.
>
> **DEC-49 ADDS A FIFTH PARTICIPANT AND IT DOES NOT TRIP THIS RULE — which has to be argued,
> because "it is a participant, therefore it is frozen before W2b" is the obvious reading and it
> is wrong.** `episodic_store` has **no region weights**: it is non-parametric data plus two
> projections, and those projections live **inside white matter** (§2.3), trained after admission
> rather than frozen before it. **So `s_r` is never computed for the store**, and that is an
> **exemption declared here, not an omission discovered at W2b**: the store is empty at admission
> time, so an `s_r` for it would measure an empty container.
>
> **An exemption without an enforcement point is a hole, so it gets one.** A3's rule already
> supplies the mechanism — W2b refuses any region whose `status` is not `built` or whose receipt
> hash does not match the resident weights `[S31-9 fixed]`. **The store is admitted to that check
> as an explicit exemption keyed on `kind == "nonparametric_store"`, and the exemption is
> constructed to fail:** attempt admission with a *parametric* region carrying
> `kind: nonparametric_store` and assert **refusal**, so the exemption cannot be used as a bypass
> by anything else. **`R` is 5 and §5.4's ten ablation pairs follow from it** `[DEC-49]`. **`speech_output` never bore on
> this rule for the reason revision 3.1 gave anyway** `[S31-19 fixed]`: it is a head, it adds no
> participant and no ablation pair, and the only way it could have forced itself before W2b was
> by changing `language_code`'s **weights** — which, under the frozen-trunk reading its own row
> now adopts, it does not.

> **NAMING NOTE, because two namespaces share a prefix.** The audio rows added in revision 3.1
> are **A0m, A0f, A1, A2, A3** (revision 3.1 wrote A0–A3; 3.2 splits A0 per S3/S7). The first
> skeptic pass's findings are **A1–A35**, and the third pass's are **S1–S21**. They are different things: a
> bare `A1` inside a `[... fixed]` tag or in §10.1/§11 is always a **skeptic finding**; an audio
> row is always written **row A1** in prose, and appears bare only in this table's `id` column
> and in `blocked_by`. The collision is unfortunate and is called out rather than renamed,
> because the row ids are already in the ratification brief and the audit document. **In
> revision 3.2 the audio rows leave this table entirely** — they are deferred, and they live in
> *Production phase: audio* below it.

| id | task | gate | blocked_by | status | host |
|----|------|------|-----------|--------|------|
| **W0** | **Token surface — read "position latents": per-position hidden vectors, not discrete ids `[DEC-47]`. The method name `tokens()` is the API and is NOT renamed.** Split `TextEncoder.forward` at the pooling line; add `tokens()`/`pool()` to `TextEncoder` and `ViTEncoder`; split `native_dim` into `token_dim`/`pooled_dim` (A32); wire `visual`'s `[B,64,384]` first; **and re-instantiate §2.3's parameter table at the design's actual configuration** (4 controller heads, v1 participant list) so the total stops being `[I]`. Retrains nothing. | **Four properties that can actually fail** `[A10 fixed]`, because the `1e-5` equivalence test cannot: `pool()` will contain the *same* masked-mean expression `forward` contains (`src/cogsyndelta/regions/text_encoder.py:117-124` — **`regions/`, not `model/`**, and the quoted comments match verbatim at those lines [V] `[N10e fixed]`), so it compares an expression to itself through the same weights and a pre-existing pooling bug is present identically on both sides. **(i) Pad-invariance:** perturbing embeddings at masked positions must not change `pool()` — the exact bug class the code comment records having been found once (*"Averaging over padding pulls every short text toward the same vector"*). **(ii) Independent reference:** `pool()` equals a separately written numpy masked mean to 1e-6 — two independently-written implementations agreeing is evidence; one agreeing with itself is not. **(iii) Row-permutation, BOTH clauses** `[N10f fixed]`: permuting the **unmasked** positions of `h` **and `mask` together** leaves `pool(h, mask)` unchanged, **and must change `tokens()`**. Revision 2 wrote *"permuting the rows of `h`"* without permuting `mask`, which changes the masked mean — so the property was false as written and the test would have failed on correct code. The second clause is the half A10 asked for and revision 2 dropped: it is what confirms `tokens()` is not already the constant the central bet feared, and after §4.0 it is the cheaper of the two checks that bear on that question. **(iv) Batch-composition invariance:** the same item padded to 6 and to 66 positions pools identically (the regression measured at cosine 0.958). **Keep `‖pool(tokens(x)) − encode(x)‖∞ < 1e-5` on 512 items as a refactor smoke test, and stop claiming it audits the receipts.** | — | todo | any / CPU |
| **W0c** | **DEC-40 — one `load_checkpoint()`, and the lint rule that keeps it the only one** `[N10c fixed]`. Revision 2 wrote DEC-40 and then routed it to "OD-7", which is the disposition of the untracked working files — a different item. It had no row, no owner and no gate, which is how the durable half of a security fix becomes a paragraph. This is that row. One `load_checkpoint()` in `src/`, the **only** `torch.load` in the repository, `weights_only=True` **hardcoded** (not a default, not a parameter, no override), manifest SHA-256 verified **before the file is opened**, every script importing it. **Owner: the operator, by hand — explicitly NOT autodev**, because DEC-40 is a control on the loader and OD-1 records that autodev's write scope plus merge authority can weaken a guard and its test in one change. No GPU, no dependencies. | **Three, all constructed to fire** `[N10c fixed]`: (i) a **CI lint rule** (ruff or a grep in `code-quality.yml`) fails the build on any `torch.load` outside that module — **verified by adding one in a scratch branch and watching CI go red**, because without that demonstration this is a policy and not a control; (ii) a **hash-mismatch refusal**, given the same negative-test treatment `tests/test_checkpoint_load_security.py` already gives the pickle refusal — flip one byte of a checkpoint, assert `load_checkpoint` raises **and** that nothing was unpickled; (iii) `grep -rn "torch.load" src/ scripts/` returns exactly one hit. | — | todo — **no dependencies, do early** | any / CPU |
| **W3r** | **Composite frame geometry + minimal renderer + checked-in fixture** `[N5 fixed]`. Revision 2 had W7v's round-trip gate consuming a rendered composite, the renderer as W3's primary deliverable, and W3 `blocked_by: W7v`. **This row is the W3-independent prerequisite both of them need.** It fixes the frame geometry once — `image_size 128`, `patch_size 8` ⇒ `n_patches 256` — records it as the number W7v re-shapes the encoder to and W3 renders against, ships a **minimal deterministic layout engine**, and checks **one hand-built 128×128 four-panel composite into the test fixtures** together with the layout description it was built from. CPU only, no GPU, no dependencies. | The fixture exists and is tracked; the renderer **reproduces it byte-identically** from its layout description (a deterministic layout engine that cannot reproduce its own output is not a ground-truth source); the geometry triple is recorded and propagated to §1.4, §2.3 and §6.1. **Verify by making it fail:** perturb one panel's position in the layout description by one pixel and assert the byte-comparison fails. **Until this row lands, "composite rendering" is NOT in the CPU-parallel column of §4.2** — a four-panel composite at an unfixed geometry is not work, it is a guess. | — | **done — fixture, renderer and both gate tests land at `src/cogsyndelta/vl/composite.py` / `tests/fixtures/composite_4panel.{png,spec.json}`; geometry triple propagated to §1.4, §2.3, §6.1 (this commit)** | 1080 Ti / CPU |
| **E0** | **Extract and PIN the memory-gate contracts. No GPU, no dependencies — do it with W2a** `[DEC-49] [OP: csd-episodic-store-required.md]`. Port the conformance and tiering tests out of `memory-gate` / `memory-gate-rs` into this tree as **executable contract tests against CSD's own store interface**, before a line of the store is written. The specific tests, named so the row cannot be reported as done by porting something easier: the **domain-isolation** fixture `test_domain_isolation_same_logical_key` [V, `tests/storage/test_store_conformance.py:150`], the **identity** fixtures `test_double_put_same_identity_last_write_wins` and `test_query_requires_domain_or_global_flag` [V, `:170, :188`], the **caller-cannot-mutate** fixture [V, `:299`], and the five Rust eviction fixtures `admit_spills_lowest_importance`, `gpu_hint_protects_from_spill`, `promote_returns_span_to_ram`, `retrieve_merges_tiers`, `disk_prune_drops_lowest` [V, `storage/tiered.rs:292-353`]. **Record which clauses were SILENT** and check the §8 gaps block against the result — E0 is also the row that can prove §8 wrong. | **The gate is that the tests exist and FAIL against an empty implementation, which is the only state in which a contract test is evidence.** Specifically: (i) all **nine named fixtures** (not twelve — the row names nine and the count is corrected to match, rather than leaving three unnamed slots a later pass could fill with easier tests `[S33-5 fixed]`) run and **fail red** against a stub store, and the run is recorded; (ii) **domain isolation** and **eviction order** are among them, so **a store built without them cannot pass this row and therefore cannot reach E1**; (iii) the parameterisation runs the suite against **three backends** — in-memory, SQLite, and the Qdrant fake — because the source suite's own value is that it is backend-independent [V, `tests/storage/test_store_conformance.py:78-299`, parameterized identically across the three backends per ADR-0002/ADR-0003] `[S33-9 fixed]`; (iv) **every clause §1.3 marks `[V]` maps to at least one ported test, and every clause it marks SILENT maps to none** — a test that appears for a silent clause means somebody invented a contract, and the row FAILS. | — **no dependencies, do early** | todo | any / CPU |
| **W1** | **CONSEQUENCE-3, CHEAP HALF.** Per-region participation-ratio effective rank of `pool()` vs `tokens()` (position latents, `[DEC-47]`); mean pairwise cosine; the `R×R` linear-CKA matrix. | **DONE 2026-09-02. Verdict: BET DEAD for all four production regions — and since 2026-09-02 that verdict is CONFIRMED BY W1d and no longer provisional** (participation-ratio ratios 0.66×, 0.78×, 1.00×, 1.30× against a ≤1.5× dead band); per-item flag fires for `code` and `retrieve`. **Entropy-effective rank on the same surfaces reverses the sign for all four (1.21×, 1.16×, 1.21×, 1.84×)** — a disagreement W1d adjudicated by measuring task performance instead of rank. Both columns and the non-production fifth row are in §4.0; evidence at `docs/design/evidence/w1-token-rank-2026-09-02/`. | — | **done, verdict CONFIRMED** | 3090 `.98` |
| **W1d** | **CONFIRMATION of W1, run before the retrain was spent** `[A9 fixed]`. **Matched read-out probe:** the same small cross-attention read-out trained three times with identical params/steps/items — arm (a) over the final block's position latents, arm (b) over `pool()` **broadcast to the same `T`**, arm (c) over **penultimate-block** activations — plus within-item vs between-item variance and **both rank definitions on the probe's own activations**. | **DONE 2026-09-02, 116.5 s, read-only on the production checkpoints. W1 is CONFIRMED and no region is OVERTURNED.** `code` +0.00, `compress` −1.76, `vl_latent` −2.93 — all **CONFIRMED**, each passing its own seed self-check; `retrieve` −6.05 **CONFIRMED but the instrument is NOISY**, its seed self-check spreading **2.54 pp > the 2-pp rule**, so it is flagged and never cited as clean. **The sequence-blind arm BEAT the token arm in three of four regions.** **The penultimate fallback is DEAD** (arm (c) worse or noise everywhere; participation-ratio rank 2–4× lower for every text region) — **`L_token` attaches at the FINAL block**. **The token-aware retrains are LICENSED.** Caveats carried: an untrained cross-attention read-out ≈ uniform attention ≈ a linear function of `pool()`, and 600 steps of read-out training slightly **reduced** recall for `code`/`compress`. Evidence at `docs/design/evidence/w1d-readout-probe-2026-09-02/` with sha256s in §4.0. | W0 | **done — retrains licensed** | 3090 `.98` |
| **W1b** | **Regenerate `reason`'s receipt** before it is frozen and before W7a retrains it. Its receipt was deleted; the r@1 0.0801 figure is prose only. | Receipt exists under `/akula-data/csd/receipts/`, reproduces r@1 ≥ 0.0801 **against an untrained baseline instantiated at a region-specific seed, not seed 0** (§4.0). It **must** carry `corpus.cap_sampling` **and** the drawn `pair_fingerprint`s (W2a), so this loss can never recur. **A frozen region with no receipt cannot be a baseline for anything and cannot be shown not to have regressed.** | — | todo | 3090 `.98` |
| **W2a** | **Ledger + recovery. Minutes, no GPU, no dependencies — do it first** `[A34 fixed]`. (i) **Verify** the `compose` allocation for `codeparrot/apps` + `deepmind/code_contests` landed — it has (§5.2, and §11 R1). (ii) **Recover the aqua_rat draw, as the UNION of every draw the run discovers** (DEC-42) `[A5 fixed] [N1 fixed]`: compute **draw R** = `load_pairs(["reason/aqua_rat-raw/train.parquet"], ("question","rationale"), 4982, seed=0)` under today's reservoir code, **draw P** = the **first 4,982 pairs `_iter_pairs` yields in shard order** — what the same call returned before `c42203c` rewrote the cap from prefix-truncation to reservoir sampling — **and, found by the run itself rather than anticipated, draw D3** = the on-disk `reason/aqua_rat-raw/derived/sample-4982-seed0.parquet` (mtime `2026-09-02T23:05:45.957176Z`, sibling `MANIFEST.json` declaring `numpy Generator(PCG64).permutation(n)[:N_SAMPLE]` — a full-corpus shuffle-then-take, not a prefix — both before `b9a082e`'s cap-introducing commit and before `c42203c`). Write `fingerprints(R) ∪ fingerprints(P) ∪ fingerprints(D3)` — and any further draw a directory walk turns up — to the ledger as `reason`-burned, record the source file's corpus fingerprint, and **mark the remainder clean**. **Measured** `[V*, burned-aqua_rat.jsonl.manifest.json]`: `R` 4,982 rows / 4,951 unique; `P` 4,982 / 4,946 unique; `D3` 4,982 / 4,930 unique (4,369 new against `R ∪ P`); union **13,946**; clean **83,521** of 97,467. | **Four checks, all of which must pass, over however many draws are discovered** `[N1 fixed]`: (i) the recomputed `fingerprint_corpus` of the parquet on disk today matches; (ii) two consecutive computations of **each** draw return identical fingerprints; (iii) the run genuinely used `seed = 0`; **(iv) THE FOURTH, AND IT IS THE ONE THAT CAN FAIL:** establish which code revision the `reason` run used — checkpoint/receipt mtime against `c42203c`'s commit time (2026-09-02 19:32:52 -0400), or a trainer revision recorded in the checkpoint — **and if it cannot be established, RECORD THAT IT COULD NOT.** Either way the ledger takes the union. **The gate that makes this failable:** W2a asserts `count(ledger ∩ draw) = 4,982` for **every discovered draw** and **refuses to emit a ledger that omits any of them** — a fingerprint set omitting even one discovered draw is a FAIL, not a partial pass. **If any of the four fails, the write-off stands and §5.1 reverts** — this is a test, not an assumption. Then re-derive §5.1 and §9.2. | — | **done** — branch `feat/w2a-ledger-recovery`, `c6ea587` (approved) | 1080 Ti / CPU |
| **W2c** | **Measure the two untrained baselines the thresholds rest on** `[A23 fixed] [A13 fixed]`. (i) Instantiate `TextEncoder(vocab 50257, dim 256, depth 4, heads 4)` **at a region-specific seed**, run the `code` in-mixture eval, and settle whether the lexical floor is **0.2285** (the surviving receipt) or **≈0.40** (prose in four files, no artefact). (ii) Re-instantiate `retrieve`'s random encoder and establish a real untrained baseline for the 0.7480 figure, whose recorded baseline is **exactly 0.0000**. **W2c HAS RUN — 2026-09-03, receipt `w2c-untrained-baselines-20260903T124940Z.json`, evidence committed at `docs/design/evidence/w2c-untrained-baselines-2026-09-03/`** (script, `results.json`, `SHA256SUMS`). **The ≈0.40 code floor is RETIRED as unsupported; the measured numbers and the trace of where 0.40 came from live in that directory and are NOT restated here** `[DEC-62]`. | **BOTH CLAUSES MET, and the second answer is the consequential one.** Both numbers exist in receipts; **`τ_lo` is re-derived per bin in chance-normalised units** (DEC-36); the disagreeing documents are corrected. **The 0.40 branch did NOT fire** — the measured code floor is 0.2285 / 0.2344, so `apps`/`code_contests` are not excluded and the reserve's main source survives. **What DID fire is the branch nobody wrote: `retrieve`'s `τ_lo` is 0.0000**, which makes NSRS condition (1) barely satisfiable on that bin. **That is now W2b's blocking clause, not this row's** `[DEC-62]`. | W0 | **done 2026-09-03** | 3090 `.98` |
| **E1** | **Build the store: partition, byte-capacity, eviction, lifecycle — plus the DYNAMIC-CAPACITY PROBE on both GPUs** `[DEC-49]`. Implement against E0's failing tests: `(scope, domain, logical_key)` identity with the scope segment derived server-side (§9.9 B2); **byte** capacity from §8 gap (a)'s formula computed **per host per scheduler tick**; scored eviction `importance + gpu_resident_bonus − staleness(last_accessed)` with ties by older timestamp then key; the six lifecycle verbs with **refusing** backpressure at `max_in_flight = 32`; SQLite as the durability oracle with in-memory as the conformance oracle. **The store's two projections are NOT trained here** — they are white-matter parameters and E2 trains them. **This row builds a container and proves it is a correct container; it makes no claim about usefulness**, which is E2's job and is the whole point of DEC-50's step separation. | **Four, three of them constructed to fire.** (i) **E0's nine named tests go green** `[S33-5 fixed]`, and the diff that makes them green touches no test file — a fix that edits its own gate is refused. (ii) **THE DYNAMIC-CAPACITY PROBE, and it must produce two DIFFERENT, POSITIVE numbers** `[S33-6 fixed]`: compute capacity from a live `nvidia-smi` plus the scheduler's current KV and activation budgets on **the 3090 Ti (24 GiB) and the 5080 (16 GiB)**, assert the two **differ**, assert **each is `> 0`**, assert each is **respected** — a write that would exceed it triggers eviction rather than an allocation — and assert the value **changes when the active-region set changes**, since the 1080 Ti is preemptible and a capacity fixed at process start is not dynamic. **A probe returning one number for both cards has measured a constant and failed; a probe returning two DIFFERENT numbers where one is zero has also failed** — §9.11 found the residual may round to zero on the 5080, which is W10's deployment card, and "differ" alone passes that case. **Pre-committed branch, so this is not discovered at deployment:** if either card's `capacity_bytes` rounds to zero, gap (a)'s named alternative fires — a fixed **floor**, reserved for the store before the KV budget is computed, sized at `safety_margin`'s order (2 GiB) — and E1's receipt records which of the two (residual or floor) is active on each card, rather than the row passing on a store that has no capacity at all on the card it will actually run on. (iii) **CROSS-REQUEST FUZZ:** ≥10,000 interleaved writes across ≥100 partitions, then a read from every partition; **zero reads return a value written under another scope**, and the negative test is a deliberately mis-derived scope key that **must** produce a cross-partition read so the fuzz is shown to be capable of catching one. (iv) **EVICTION ORDER UNDER A CONSTRUCTED OVERFLOW:** build a working set whose scores are known and whose total exceeds capacity, then assert the survivors are exactly the top-scored set **and** that a GPU-resident low-importance entry outlives a host-resident higher-importance one by exactly the `+1.0` bonus — the behaviour `gpu_hint_protects_from_spill` pins upstream. | **W0**, E0 | todo | 3090 `.98` **and** 5080 `.251` (both, by construction) |
| **E1a** | **THE VRAM ARBITER AND THE FRUGALITY CAP — memory CONTENDS for VRAM, it does not take what is free** `[DEC-70] [OP: csd-memory-gate-overlays.md]`. E1 builds a correct container with a byte capacity; this row builds the **negotiation** that decides how much of that capacity is used and who yields when the card is tight. Named claimants with a priority order: **base weights and the working activations are fixed**; the **ACTIVE memory set and the loaded persona are protected** and are never evicted to make room for context; the **KV cache / context window flexes** (sliding window, shorter context) around them; **inactive overlays, offsets and differentials sit below context and leave first**. A **frugality cap** — a stated constant per card — bounds resident memory + persona to a small fraction of the card, enforced by the residency scorer, which scores on **recency, relevance to the loaded persona/basin, and access frequency** across **card / host RAM / disk**. **Eviction is proactive on inactivity and on pressure**, not only on capacity overflow. **Every receipt records peak VRAM held by memory + overlays and the fraction of it that was active**, which is what makes frugality a measurement instead of an intention. **AND, BECAUSE TWO OF THIS ROW'S FOUR GATES NEED MACHINERY THAT IS DEFERRED: this row DELIVERS ITS OWN MINIMAL PERSONA/OVERLAY STUB** — a loadable persona handle and an inert, TTL-carrying overlay object that the arbiter can protect, score and evict — **because P5′o, which builds the real overlays, is `deferred` behind W10**. Without the stub, gates (3) and (4) below cannot be executed at the point this row is scheduled, and a gate that cannot be run is not a gate. The stub is explicitly NOT P5′o: it carries no weights, makes no fingerprint claim and ships no attach/disconnect semantics; when P5′o lands it replaces the stub and gates (3) and (4) are re-run against the real object. | **Four, three constructed to fire.** (1) **UNDER A CONSTRUCTED CONTEXT-LENGTH INCREASE the store and the inactive overlays SHRINK and the run does not OOM** — build the case, assert the bytes released and assert no allocation failure. (2) **AT THE CAP, achievable context length differs from the no-memory case by less than a stated margin** — measured both ways on the same card, and a cap that costs more context than the margin allows is a **FAIL of the cap, not of the test**. (3) **FAILURE TO SIZE REPORTS AND DEGRADES BY RELEVANCE:** over-fill the relevant set past the cap and assert a receipt field and an event are emitted, that the surviving set is the top-scored one, and that **the loaded persona is still resident afterwards** — a run that silently drops the persona fails this clause even if it fits. (4) **An overlay resident and unused past its TTL is a TEST FAILURE**, constructed by loading one and idling it — proactive eviction that only fires on overflow is not proactive. | E1, W7k, **and this row's own minimal persona/overlay stub for gates (3)–(4)** — P5′o is `deferred` behind W10, so the stub is in scope here rather than a dependency on a deferred row | todo | 3090 `.98` **and** 5080 `.251` (both, by construction) |
| **W4** | **Merge `compress` + `retrieve` → `memory`, DESIGNED AS THE FIRST TOKEN-AWARE RETRAIN** so one run serves two rows `[A13 fixed]`. Shared trunk, two heads, joint loss **plus §4.0's `L_decorr` + `L_token` terms**. It is the natural first instance: the merge is a retrain that has to happen anyway, its two parents give it two independent regression checks, and a failure here is diagnosable as *merge* or *objective* by ablating one term. **It is also where DEC-24's shared embedding table first becomes achievable** (§6.2) `[A31 fixed]`. | **All five, pre-registered:** (1) `memory` matches or beats **both** parents on **both** parents' own gates (0.7070 and 0.7070-comparable, and 0.7480 **re-baselined by W2c**); (2) **`recall@10 > 0.20` and `MRR > 0.10`** on the FiQA BEIR eval against the full 57,638-passage pool — the numbers `retrieve.py:384` already states, which revision 1 adopted the eval for and then dropped; (3) **`memory` > BM25** on the same pool and qrels, from the same code path — not "BM25 reported alongside", *beaten*, because *"reporting a win for a loss"* is exactly what that file was written to prevent; (4) **`memory` > its own random-init baseline** on that pool; (5) **§4.0's retrain gate**, both clauses. Saves 15,890,176 params. | W1d, W2c, W1b | **The five clauses above are UNCHANGED by revision 3.5 and are printed here exactly as they were pre-registered** `[DEC-68]`. **RUN AT THE PRE-REGISTERED CONFIG 2026-09-03. (a) PASS, (b) PASS AS SCORED — CONTINGENT: the strict `> 0.20` floor is cleared only by float32 representation (receipt value `0.20000000298023224` = `float(numpy.float32(0.2))`), so `>` versus `≥` goes to OD-17 — (c) FAIL (0.200 vs BM25 0.440), (d) PASS, (e) FAIL (rank ratio 1.2102 < 2.0; regression clause 0.0 PASS).** Receipt `w4-chunked/run-1280/memory-20260903T184441Z.json`, code `eb735ab`, batch 1280, `token_loss_chunk` 512, 4,000 steps. **BLOCKED ON OD-17** — pivot per §9.14 as pre-committed, or amend as a pre-registration; §4.0's DEC-68 block carries every number. *The old status cell read `todo`; the run has happened, so that word is replaced rather than kept beside a receipt.* | 3090 `.98` |
| **W4n** | **THE NEGATIVES ABLATION — the pre-registered next lever on W4's remaining fails, and it is NOT LICENSED UNTIL OD-17 IS ANSWERED** `[DEC-68]`. DEC-68 measured the in-batch-negatives count as the dominant lever on the metric gate (c) is defined over — a **6.7×** move in full-pool `recall@10` across a **5×** move in batch — and batch 1280 is where the card runs out. This row asks the obvious next question: **can the negative count be raised without the batch?** Three arms at a **fixed VRAM budget** and a fixed step count: **(i)** the current in-batch regime at batch 1280 as the baseline, **(ii)** a **negatives queue** (momentum-encoder queue, MoCo shape) at an effective negative count of 8× the batch, **(iii)** **GradCache** (gradient caching over micro-batches) at the same effective count. **Every arm prints its EFFECTIVE negative count and its peak VRAM**, because an arm whose effective count is not what it claims has measured its own plumbing. | **Pre-registered before the run, and the margin is written down first.** (1) **The winning arm must beat the batch-1280 in-batch baseline on full-pool `recall@10` by a margin stated in writing beforehand**, at equal or lower peak VRAM and at equal wall-clock. (2) **A null result is a result** and is recorded as one — *the negative count is not the remaining lever* is a finding this programme can act on, and an ablation that can only report a win is not an ablation. (3) **A FAILABLE CONTROL:** an arm configured with an effective negative count **equal** to the batch's must reproduce the batch-1280 numbers within run-to-run noise; if it does not, the arm differs from the baseline for some reason other than the negative count and the comparison is void. (4) `parent baseline` — if OD-17's amendment is taken, `retrieve` alone is measured on the same pool in the same harness, since gate (c) has never had one. | **OD-17**, W4 | **todo — BLOCKED ON OD-17.** It is the recommended amendment's own next lever and it is not licensed until the amendment is; under the pivot branch this row does not exist | 3090 `.98` |
| **W7a** | **Token-aware retrain of `language_code` and `reasoning`.** Same objective as W4, same harness, one region at a time. | **§4.0's retrain gate, per region, both clauses.** Plus: `language_code`'s r@1 0.9766 must not regress > 1 point, and `reasoning`'s regenerated 0.0801 (W1b) must not regress > 1 point. A region that clears the rank clause and fails the receipt clause is **reverted**, and the receipt says which. | W1d, W1b, W4 | todo | 3090 `.98` |
| **W7v** | **`visual`: token-aware retrain AND resolution rebuild** `[A4 fixed]`. `visual` **physically cannot ingest a composite image today**: `JEPAConfig(image_size=64, patch_size=8)` and `ViTEncoder` registers a **fixed** sincos `pos_embed` of size `cfg.n_patches = 64` (`model/vl_jepa.py:52-53, 182, 199` [V]), so `h = self.patch_embed(x) + self.pos_embed` is a **shape error** on a larger image, not a slower forward. Meanwhile §5.4's `visual × language_code` and `visual × memory` are **2 of 4 cross-faculty pairs = 1,024 of 2,048 eval items**, and §5.4 itself concedes they are *"unconstructible against 64×64 single-object tiles"*. Revision 1 scheduled no row to build the prerequisite. **This row does it, folded into the retrain that W1's result made mandatory anyway, so the marginal cost is the resolution change and not a separate run.** Re-shape to composite resolution (`image_size 128`, `patch_size 8` ⇒ `n_patches 256`, `pos_embed` regenerated or 2-D interpolated), continue-train, and — if OD-4 is decided that way — swap the corpus in the same run. **`pos_embed` REGENERATED, not 2-D interpolated: the study behind this row (g8 sector `S02.md`) killed "or 2-D interpolated" as a category error — a length-64 1-D raster has no `(x,y)` axes to bicubic; do not read this cell's disjunction as licensing that path.** | New `n_patches`, `kv_bytes_per_token` and `token_budget.max` recorded and propagated to §1.4, §2.3 and §6.1. Probe top1 and cifar100 transfer **do not regress by more than 1 point** against 0.0606 / 0.2625 **measured on the target encoder** (DEC-34), with the untrained baseline re-instantiated at a region-specific seed. **§4.0's retrain gate**, both clauses. **A composite frame renders, encodes and round-trips** — the negative test is that the 64×64 checkpoint **raises** on the same input. **The frame is W3r's checked-in 128×128 fixture, NOT a W3 deliverable** `[N5 fixed]`: revision 2's gate consumed the composite renderer, which is W3's primary deliverable, while W3 was `blocked_by: W7v` — a dependency inversion in which neither row could start. W3r breaks it. **W7v-cfg landed the CONFIG-LEVEL HALF 2026-09-04, CPU only** `[OP: /akula-data/session-backup-staging/tools/grok-jobs/g8-visual-faculty-design.md "What to build first"; g8-visual/S02.md]`: `JEPAConfig` now imports `FRAME_SIZE`/`PATCH_SIZE`/`N_PATCHES` from `vl/composite.py` (single source of truth) instead of re-deriving them, so the default config renders the 16×16/256-patch grid this row calls for; positions are a genuine 2-D sincos (`sincos2d_pos_embed`) recomputed for `cfg.grid`, never an interpolation of the old table — the 1-D `sincos_pos_embed` is untouched and remains correct for `TextEncoder`, its only remaining caller. An explicit 64-px config still works. Loading a checkpoint recorded at one grid into a config built for another now raises, naming both grids (`check_checkpoint_grid_compatible`) — closes the silent-load hole `pos_embed` being a non-persistent buffer otherwise leaves open (`load_state_dict` alone cannot see a grid mismatch). `VL_REGIONS["visual"]` (canonical; `VL_REGIONS["vl_latent"]` is the same dict via the alias
layer) no longer hardcodes tiny-imagenet as a train/probe_eval glob; `corpus_source` must name a landed, admitted corpus or `run_vl_region` refuses to start, citing OD-4. The visual receipt stamps `corpus.fingerprint`/`corpus.fingerprint_scheme` (reusing `cogsyndelta.corpus.fingerprint_corpus`, no new scheme) and prints `masking: random-permutation` (`sample_masks` is still `torch.randperm`; multi-block masking is NOT implemented by this increment). **Not done: no GPU retrain has run.** OD-4's corpus is not landed, so 0.0606/0.2625 above remain the pre-rebuild 64-px/tiny-imagenet numbers and are void as a post-OD-4 gate the moment the corpus changes (re-instantiate untrained + chance on the new probe, per this row's own gate; do not carry 0.0606/0.2625 forward). | W1d, W3r | in progress — config rebuild done 2026-09-04 (branch `feat/w7v-cfg`); retrain BLOCKED on OD-4 | 3090 `.98` |
| **W7p** | **PLACEMENT HARNESS — a per-job VRAM budget, a target host, and receipts that say which** `[DEC-54] [OP: csd-training-placement-policy.md]`. Not a training row: the tooling every remaining GPU row already needed and none of them had. **Three deliverables.** (i) The launch path (`csd-train@`'s template today hardcodes `--batch 1280`, which is how `reason` OOMed) accepts **`--vram-budget-mib` and `--host`**, sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and a **per-process memory fraction** derived from the budget, and **refuses to start when the budget exceeds the target card's free VRAM at launch** — the refusal is the point, not the sizing. (ii) An **admission check for concurrency**: two or more submodel runs are co-scheduled on one card **only when the sum of their budgets fits**, with the sum printed; `reason` carries a **declared `exclusive: true`** and its measured configuration — **batch 512, `max_len` 256, alone** — so the one region known not to co-reside cannot be packed by a future scheduler that has forgotten why. (iii) Every receipt gains **`placement { host, gpu_name, vram_budget_mib, vram_peak_mib, concurrency, co_resident_jobs[] }`**, which is what makes DEC-54 auditable rather than aspirational. **No MIG on consumer cards: this is process-level PSEUDO-isolation, stated as such so it is never read as a security boundary** — acceptable because the fleet is single-tenant. | **Four, and three are constructed to fire.** (1) **A budget larger than the card's free VRAM REFUSES to launch** — construct it by requesting 30,000 MiB on the 5080 and assert nothing starts and nothing is written. (2) **A co-schedule whose budgets sum over the card REFUSES the second job**, verified by requesting two 14,000 MiB jobs on the 3090 Ti. (3) **`reason` submitted as a co-resident job is REFUSED on its `exclusive` flag**, and the refusal names the receipt that justifies it — the regression this row exists to prevent is exactly the one already paid for. (4) **Positive control:** two submodel runs whose budgets do fit **do run concurrently on the 3090 Ti and both produce receipts carrying `concurrency: 2` and each other's job ids** — without this clause the row is three refusals and no capability. **`vram_peak_mib` is measured, not declared**, so a budget that was wrong is visible afterwards. | — | **The tooling this row specifies EXISTS and lives in `tzervas/gpu-pack`, not here** `[DEC-75]` — probe, ledger, admission under one flock, per-process cap, transient units with emitters, remote launch and launch receipts, `main` at `5364932`. **What remains is this row's CSD-side acceptance**, not a reimplementation. **in progress — gpu-pack `main` at `5364932` (rounds 1–4) implements (i) and (ii); CSD's adapter (`src/cogsyndelta/util/gpu_budget.py`, `program/jobs/*.json`) is in this tree. Remaining: (iii)'s `placement{}` receipt block and all four gates, including the `reason` `exclusive: true` refusal.** *The old status cell read `todo — no GPU dependency for (i)/(ii)`; (i) and (ii) are built, so that word is replaced.* | any / CPU + 3090 `.98` for (4) |
| **W7k** | **THE KNOB TABLE BECOMES RECEIPT FIELDS, AND THE PACKING MODEL IS REUSED AT INFERENCE** `[DEC-69] [DEC-74] [OP: csd-training-placement-policy.md]`. §6.7's knob table is a measurement until the values a run actually used are in that run's receipt; today `token_loss_chunk`, the admission margin and the foreign baseline appear in launch notes and evidence directories and nowhere a later reader will look. **Three deliverables.** (i) Every training and quant receipt gains **`knobs { vram_budget_mib, admission_margin_mib, token_loss_chunk, batch_size, negatives_effective, mask_prob, card }`** — `negatives_effective` separately from `batch_size`, because W4n's whole point is that they stop being the same number. (ii) A **`foreign_baseline_mib`** field records the non-job holder measured on the card **at launch** — **981–1,057 MiB of desktop on the 3090 Ti** in both chunk probes — so a budget computed as though the card were headless is visible afterwards rather than inferred from an OOM. (iii) A **runtime packing prototype**: gpu-pack's `compute_allowed_mib` run over the **inference** claimants of DEC-70 (regions, overlays, store, KV) that prints the plan and its per-claimant byte shares. **No scheduler is built at this row** — the point is to find out whether the training admission model transfers before anything is built on the assumption that it does. | **Four, three constructed to fire.** (1) A receipt missing any `knobs` key **fails** a schema test, constructed by deleting one key and asserting the failure. (2) **`foreign_baseline_mib` is MEASURED, not declared:** start a job holding a known allocation on the card, assert the recorded baseline moves by roughly that amount — a field that reports a constant has measured nothing. (3) The runtime prototype **REFUSES** a plan whose claimant budgets exceed the card, constructed by asking for one that does, and the refusal names which claimant put it over. (4) **Positive control:** a plan that fits is admitted and prints every claimant's byte share summing to no more than the card minus the margin — without this the row is three refusals and no capability. | W7p | todo — CPU for (1) and (3); one card for (2) and (4) | any / 3090 `.98` |
| **W2b** | **Reserve construction + NSRS admission + the guard** (replaces P2.5d). §5. **Only the BM25 condition (3), the licence audit, dedupe and rendering are CPU-parallel; `s_r` is inference through every frozen region including `memory`, so this row is GPU-blocked** `[A7 fixed]`. | `data/reserve/` + manifest exists; **thresholds frozen from a disjoint calibration split** and the would-have-been-rejected fraction of the graded set reported (DEC-36); every admitted row records `s_r` for all `r` and satisfies the NSRS filter **including the written condition-(2) exemption for the general bin** `[A21 fixed]`; **`RESERVED.jsonl` holds the union of SOURCE-ROW `pair_fingerprint`s, not composite hashes** (DEC-38); the ledger records `compose` for apps + code_contests; apps↔code_contests dedupe at cos ≥ 0.90 run and its 1,784 removals recorded; **keyed split assignment live** (DEC-39). **AND two refusals, both constructed:** (a) a training run seeded with one **directly reserved** row REFUSES TO START; (b) **a training run seeded with a row that a reserved COMPOSITE was built from REFUSES TO START** — the derived case, which is the path that actually leaks `[A6 fixed]`. **AND ONE MORE, ADDED IN REVISION 3.4 AND BLOCKING THE ROW** `[DEC-62]`: **admission for a CHANCE-FLOOR BIN is defined and pre-registered BEFORE this row runs.** W2c measured `retrieve`'s `τ_lo` at **0.0000** in chance-normalised units — its untrained r@1 *is* chance (1/512) — so condition (1) `max_r s_r < τ_lo` is **barely satisfiable on that bin** and admits by construction rather than by evidence. Adopt either an **absolute-margin floor** (`max_r s_r < chance + δ`, `δ` pre-registered per bin) **or a different normaliser**, record which and why, and **verify it can fail** on an item a single region trivially solves — under the current rule that item would be admitted, and the replacement rule must reject it. A bin whose admission condition cannot reject anything is not filtered. | W2a, W2c, W4, W7a, W7v | todo — **BLOCKS P2.3** | 1080 Ti / CPU **+ 3090 for `s_r`** |
| **W3** | **Construct cross-faculty items, and build the EPISODE HARNESS.** §5.5: executable joins (**sandboxed**, §5.5a), rendered composites, shuffled mismatches, **and §5.5(e)/(e′)'s recall-dependent episodes (X7, X8, X8′)** `[DEC-49]`. **The harness is a named deliverable and not an implied one:** an eval item may be a *sequence* of turns; turn 1's write must commit through the store's `learn()` and return a `LearnReceipt` before turn 2 is scored; the partition is reset between episodes. | ≥ 512 admitted items per cross-bin pair for eval **and** ≥ 5,120 per pair for train; every item passes W2b's filter **including the trivial-baseline condition**; **train/eval split at source-row granularity, split key recorded** (DEC-38); provenance chain recorded with **no generating model**, or the model named with revision and licence; **B1/B2 run on the reserve's own generators** and either diversified or waived with a dated waiver naming what the reserve therefore cannot measure `[A16 fixed]`; **the reserve's aqua_rat source-row share is printed against B1's 0.50 hard line and 0.40 operating cap** (§5.1) `[N8 fixed]`. **Three more, all from DEC-49, all constructed to fire:** (i) **the negative control on every episode** — turn 2 run with turn 1 replaced by an unrelated episode **must FAIL the item**, and an episode that passes it is **rejected at construction**, not discovered at W6; (ii) **no episode's two turns share a source row** under DEC-38, asserted per item, because an episode that leaks against itself cannot be placed on one side of a split; (iii) **the harness resets the store partition between episodes**, verified by re-running one episode and asserting its turn-1-less variant still fails. **If the harness is not delivered, W3 reports `episodic pairs: NOT MEASURED` and §5.4's E1-slip branch fires** — it does not ship six pairs under a ten-pair criterion. | W2b, W7v, W3r, **E1** | todo | 1080 Ti / CPU |
| **W1c** | **Anisotropy pre-flight.** Effective rank of the concatenated adapted union on 4,096 real mixed items, **on the retrained regions**. | **The statistic is named, because revision 2's threshold named none** `[N3 fixed]`: the number is **entropy-effective rank** (`cogsyndelta.eval.benchmark.effective_rank`), which is the definition behind the "8.7 of 128" prior this row compares against, and **`< 32 of 512` is in those units**. Participation ratio is recorded beside it, and the two are never compared across definitions — W1's pooled **PR**-ranks of 18–45 and its pooled **entropy** ranks of 115–130 (§4.0) are different quantities and revision 2 printed them as comparable. Comparable priors, restated in the right units: 8.7 of 128 on the pooled surface [V\*]; cross-region CKA ≤ 0.336 (§4.0). **If < 32 of 512 in entropy-effective rank, whitening goes into the adapters before W5.** | W4, W7a, W7v | todo | 5080 `.251` |
| **W5** | **White matter v1, PHASE A (dense).** Workspace + adapters + frontal + `L_unify` + `rank_head` (DEC-41); all regions on; regions frozen **at their retrained checkpoints**; train on W2b/W3. | **G1** composed > B1, McNemar `p<0.01` Holm-corrected. **G2** no faculty's own-bin score drops > 1 point. **G0** composed > **B3** and > trained-**B0u**. **B0** (trained read-out) and **B0d** (dispatch control) recorded in the same receipt. **Phase-A collapse floor:** `min_r mean(a_r) ≥ η/R`, printed as the expression **and** as its evaluated value beside the receipt's own `R` (**3.0% at `R = 5`**, DEC-49), with the per-region `a_r` histogram `[A15 fixed] [S31-16 fixed]`. **W5 runs with `R = 4`** — the store is admitted by E2, after this row — **so W5's receipt prints 3.75%, and E2's prints 3.0%; a receipt printing a floor that does not match its own `R` is a FAIL.** **Overfit gate:** train/held-out gap < 5 points. **Every decision on the DEV half only** (§2.7.8). | W3, W1c | todo | 3090 `.98` |
| **E2** | **Admit the store to the interconnect and RETRAIN WHITE MATTER — step 2 of DEC-50's protocol, and its first instance** `[DEC-50]`. Re-run white-matter phase A with `R = 5`: the store's `W_k`/`W_v` projections join the trainable set, regions stay frozen at their retrained checkpoints, the read simplex is re-floored at `η/R = 3.0%`, and the reserve's episode items (X7/X8) enter the training mix. **DEC-26's 2–5% interconnect cap is what makes this affordable:** the thing being retrained is the smallest trainable object in the mind, which is the property the modular design was bought for. | **Three, and the third is the one that can kill the row.** (i) **G2 holds** — no existing faculty's own-bin score drops more than 1 point against W5's receipt, measured on the **dev half**; the store itself is exempt from G2 and the exemption is §2.7.2's, not a new one. (ii) **The composed metric improves** on the dev half against W5's `R = 4` result — *improves*, not *does not regress*, because a participant that costs budget and returns nothing is a cost. (iii) **NON-ZERO ATTENTION MASS TO THE STORE:** `mean(a_store)` over held-out items is **≥ the collapse floor `η/R` = 3.0%**, with the per-iteration histogram printed. **On any failure the store is REVERTED** — removed from the participant set, `R` returns to 4, §5.4's E1-slip arithmetic fires (six pairs, ≥5 of 6, null 0.109), and the receipt records **which** of the three clauses failed, because "the store did not help" and "the store was never attended to" are different findings with different remedies. **Verify the gate can fail, in-contract** `[S33-7 fixed]`: pinning `b_store` to zero is not a state the system can legally reach — §1.4's `token_budget.min` is 8 and §2.4 floors `b_store` at `η/R · B_read = 0.03 × 256 ≈ 8` read tokens precisely so that "the store received no attention mass" is a measurement rather than a starvation artefact (§2.4). So the falsifier pins `b_store` **at that floor** (8 tokens, the legal minimum, not zero) and feeds episodes constructed with **no recall dependency** — turn 2 answerable from turn 1's context alone — and asserts `mean(a_store)` lands **below** 3.0% despite 8 tokens being available to spend on it: a store with nothing worth attending to should not be attended to, and clause (iii) must report FAIL on that input, not on an input the design already forbids. | **W5**, E1 | todo | 3090 `.98` |
| **W5b** | **Write-back (conditioning prefixes).** DEC-17. | Conditioned regions' own-bin scores drop ≤ 1 point **and** the composed metric improves, **on the dev half**. **Pre-committed fallback:** on failure, disable write-back, record that phase-2 integration is workspace-internal only, **and apply §2.4's no-write-back consequences — either build the `Ĉ` variant or emit no `edges`, no `lockstep_groups`, and `topology: not demonstrated`** `[A20 fixed]`. Either result is a pass; not running it is the fail. | W5 | todo | 3090 `.98` |
| **W6** | **THE INTEGRATION TEST. This row opens the sealed half, once.** §2.7. | **G3** composed > **B2** (oracle late fusion) **and** > **B2t** (trained matched-capacity late fusion), McNemar `p<0.01` Holm-corrected. **G3′** the **DEC-37 synergy conjunction** — `Δ_A > 0` and `Δ_B > 0` and `I > 0` per pair, block-bootstrap CI by source row excluding 0 — on **≥7 of 10 pairs**, under **zero-ablation AND both content-swap arms**; any pair in the `REDUNDANT` quadrant is reported as a FAIL for that pair, never averaged away. **G0d** `I₀` (B0, trained read-out) and B0d's `I` both contain 0, **or the experiment is void and is reported void**. **I2′** ordered-pair severance CI excludes 0. **THIS GATE IS THE PROGRAMME'S PASS/FAIL.** Seal-break timestamp and code revision recorded. **The pair count is TEN** — DEC-49 admits `episodic_store` as a fifth participant *with declared shapes*, DEC-48 keeps audio out, and §2.7.3's selection rule fixes `k` at **7** precisely so that admitting a participant could not loosen the gate `[S31-1 fixed] [DEC-49]`. **The two pre-committed fallbacks — the W7v slip and the E1/harness slip — are in §5.4, they are the ONLY permitted restatements, and each is reported by name in the receipt.** **DEC-47's inter-region assertion (W9 (iii)) is enabled for this run** and its result is printed in the receipt: an integration verdict obtained while some path re-tokenised would be a verdict about a different architecture. **The store's partition isolation assertion (E1) is enabled for this run too** — a cross-episode read during the integration test would make G3′ a measurement of a leak. | W5b, **E2** | todo | 3090 `.98` |
| **W8** | **Thalamic controller.** Phase B (distil), phase C (sparse), **phase D (task-loss)**. | **B:** `ρ(ŝ,a) > 0.6` held-out, **and no region was below the phase-A collapse floor**, or its `a_r` is excluded from the distillation targets and the receipt says so. **C:** sparse within 2% relative of dense at ≤50% of region-token FLOPs — *FLOPs only; buying accuracy here is a bug report against phase A*. **D:** beats C on the composed metric at equal-or-lower FLOPs, **or is reverted and the receipt says the scheduler is imitative**. `ctx_r ≥ ctx_min` floor verified by **constructing** an adversarial input that tries to starve a region. All accept/revert on the **dev half**. Train on the 3090; **measure on the 5080**. | W6 | todo | 3090 → 5080 `.251` |
| **W8s** | **Scheduler falsifiers S1–S5** and the **three-verdict receipt** (DEC-30 + T1). | All five reported whatever they say. The receipt carries `integration:`, `scheduling:` **and `trigger_sensitivity:`** as separate verdicts. **S5's rare-token threshold is pre-committed before the run.** | W8 | todo | 5080 `.251` |
| **W9** | **`Schedule` emission + DAG runtime + validator.** | **Two separate gates, because revision 1's two clauses contradicted each other** `[A29 fixed]` — a run that reproduces the dense path to 1e-4 has not skipped 30% of its iterations. **(i) Runtime correctness:** the DAG executor reproduces the **eager execution of the SAME `Schedule`** to 1e-4 — an implementation-equivalence test that can fail on a real bug. **(ii) Sparsity:** mean iterations down ≥30% **vs dense** at ≤1% score loss — a different comparison against a different reference. Plus: the two topology derivations agree on ≥95% of items **(void, and declared void, under W5b's fallback)**; 10,000-input adversarial fuzz emits **zero** budget-violating graphs; the validator **rejects a `Schedule` carrying a client-supplied `trace_id`**. **(iii) THE LATENT-SPACE INVARIANT, DEC-47:** a runtime assertion that **no inter-region path carries discrete token ids** — every tensor crossing a tract (a region's emitted `h_r`, its adapted form, the workspace latents, and DEC-17's write-back conditioning prefix) is a **float tensor at a declared `*_dim`**, and any integer-typed or vocabulary-indexed payload on such a path **raises**. **Verify it can fail, and this is the clause that makes the invariant a control instead of a convention:** construct the violation — hand the runtime a conditioning prefix built from token **ids** instead of latents, and a region wrapper that returns `argmax` indices from `tokens()` — and assert **both are refused**, in the `tests/test_guards_can_fail.py` pattern. A path nobody has watched refuse a token id is a design intention, not an invariant. | W8s | todo | 5080 `.251` |
| **W9i** | **THE TOKEN-ROUND-TRIP PROBE — DEC-47 stops being an assertion and becomes a detector** `[DEC-53] [OP: csd-latent-space-reasoning-invariant.md]`. W9's clause (iii) refuses a violation **someone constructed**; that demonstrates the refusal works and says nothing about the tracts nobody built a case for. This row instruments **every** tract in an emitted `Schedule` — each region's emitted `h_r`, its adapted form, the workspace latents, and DEC-17's write-back conditioning prefix — and produces a **per-tract census**: `dtype`, shape, value range, and a **round-trip signature** (whether the payload is bit-reachable from a vocabulary index, i.e. an integer-valued float tensor whose distinct-value count is ≤ the vocabulary size, which is what a silent `argmax`-then-embed looks like from the outside). **It reports a POSITIVE DETECTION rather than an absence of refusal**, which is the difference between finding an accidental re-serialisation and not having looked. | **Three, and the third is what makes the probe trustworthy** `[DEC-53]`. (i) **A clean run reports ZERO detections across every tract**, with the census printed per tract — a clean bill from an instrument that inspected nothing is the failure mode this clause exists to catch, so the census is a **deliverable, not a log line**. (ii) **A constructed round-trip is DETECTED, not merely refused:** wrap one region so `tokens()` returns `embed(argmax(logits))` — a float tensor of the right dtype and shape, which W9's dtype assertion **passes** — and assert W9i flags that tract by its round-trip signature. **This is the case W9's existing gate cannot see, and it is why this row is not folded into W9.** (iii) **A negative control on the detector itself:** a genuinely continuous tract whose values happen to be sparse must **not** be flagged, verified by feeding a quantised-but-continuous codec output (DEC-08's tract-codec slot) and asserting **no** detection — a probe that flags DEC-47's own sanctioned codec has been miswired, and DEC-47 explicitly permits quantised concept codes as a **codec** while forbidding them as a **bottleneck**. | **W9** | todo | 5080 `.251` |
| **W10** | **Composed footprint + PTQ** (was P10.3). | Measured params/bytes at fp32 and after PTQ **on the deployment card**. Per region `D_sched ≤ 0.02` nats **and** composed metric drop ≤ 0.01. Predicted: **86,331,303 params, 345 MB fp32, 35.3 MB at 3.2675 bits/param** — restated for the **five**-participant v1 list `[A24 fixed] [DEC-49]` and to be replaced by W0's re-instantiation. **The store's DATA is not in that figure and must be reported separately, in bytes, against §8 gap (a)'s dynamic capacity on the deployment card** — a footprint row that reports a mind's parameters and silently omits its store has reported half the deployment. | W9 | todo | 5080 `.251` |
| **P5′** | **PHASE 3 — whole-mind dynamic training.** Unfreeze. Joint objective over region corpora **and** the reserve, interleaved. Region growth, depth/width growth. | **G2** still holds per bin after unfreezing **AND** G3/G3′ still hold. Each new region beats its own untrained baseline **and** improves the composed metric **and** shows non-zero `Ĉ` to it. Growth increments are reverted if the composed metric does not improve monotonically. **Cross-host activation traffic ≤ 15% of step time** (DEC-29). **DEC-24 binds from here** `[A31 fixed]`. | W10 | blocked | 3090 + 5080, pipeline |
| **P5′q** | **QUANTISE-BEFORE-WHOLE-MIND, matched, at toy scale** `[OP: csd-quantize-before-whole-mind-training]`. Run the alternative ordering — per-region pretrain → **quantise (and/or fine-tune) the regions** → train the interconnect and run whole-mind dynamic training with regions held quantised — **against the canonical path** (pretrain → interconnect → whole-mind → fine-tune → quantise), matched on data, steps and seeds. The canonical path stays the default at toy size; this row is the experiment, not a substitution. **Design consequence if it wins:** the interconnect must **train against quantised region activations**, so its train-time inputs match deployment — which ties it to **DEC-27** (per-region PTQ sensitivity must be measured *before* regions are frozen, and `D_sched` becomes a training-time quantity as well as a release gate) and to **DEC-29** (it is the ordering that decides whether phase 3 fits at all on 24 GiB and 16 GiB). | **Report all four, per arm:** (1) **peak VRAM during phase 3** — the number the whole experiment exists to move; (2) composed metric vs the canonical run; (3) per-region drop from quantising *before* vs *after*; (4) **capability per parameter and capability per VRAM-GB**, not loss. Each arm gated on its own untrained baseline. **A win is only a win if it survives DEC-27's `D_sched ≤ 0.02` nats per region** — quantising earlier changes region output distributions earlier, so it shifts the schedule earlier too. | W10 | deferred to phase 3 | 3090 + 5080 |
| **P5′o** | **Memory-gate overlays** — the dynamic-paging seam's first client (DEC-33, §6.6) `[OP: csd-memory-gate-overlays]`. | **An overlay applied and then disconnected reproduces the base model's receipt metrics EXACTLY, and the disconnect is logged.** Plus: an overlay **refuses to attach** to a base checkpoint fingerprint it was not trained against (verified by constructing the mismatch); tiered-residency policy reports hit rate, page-in latency and VRAM held. | W10 | deferred | 5080 `.251` |
| **P5′s** | **Toy swarm vs comparable models** `[OP: csd-toy-swarm-experiment]`. An N-agent CSD swarm against (a) comparably sized conventional models at the same swarm size and (b) different swarm sizes, across all three GPUs (3090 Ti sm_86 24 GiB, 5080 sm_120 16 GiB, 1080 Ti sm_61 11 GiB). **It consumes two things this document produces: W9's `Schedule` emission and the per-region context/read budgets** — the harness reads those budgets rather than guessing them. | **Hard precondition: phase 3 is green.** Pre-quantisation is acceptable; anything before phase 3 measures a swarm of isolated regions behind a switchboard, which is the wrong thing. **Per-instance KV and activation budgets are computed UP FRONT and the swarm size derived from them, so a run never OOMs mid-experiment.** Capability per parameter and per VRAM-GB, with an untrained-baseline swarm and receipts per run, through the gpu-timeshare scheduler. | P5′ | deferred | all three GPUs |
| **M0** | **MYCELIUM READINESS ASSESSMENT — the downstream bar, and the row that decides whether any of this was worth building** `[DEC-51] [OP: csd-mycelium-downstream-goal.md]`. Operator: *once CSD is dialed in and proven out on quality, assess whether it can be used to complete the **Mycelium** functional, value-semantic programming-language project.* A task suite drawn **from the Mycelium repository itself** — parse, typecheck, implement, refactor — with **ground truth taken from that repository's own tests**, so the labels are executable and were not written for this experiment. **Deployment shape, as ruled:** CSD on the **5080** and the **3090 Ti**; **RAG via the 1080 Ti** carrying a helper embedding/rerank model for what will not co-reside with CSD on the 3090 Ti. **Two arms, and the second is the open question this row exists to answer:** (a) CSD **with** the 1080 Ti RAG helper; (b) CSD **doing retrieval natively as a skill** — `memory` plus `episodic_store` end to end, no helper. **Whether (b) works is MEASURED, NEVER ASSUMED**, and a design that assumed it would be assuming the most load-bearing claim in the programme. | **Pre-registered before the suite is run, or the row is void.** (i) **A pass-rate margin, registered in writing with the comparison model named, its revision pinned and its licence recorded, BEFORE any CSD run** — CSD vs a comparable open model at comparable deployed size, both on the same tasks, same harness, same retrieval corpus. A margin chosen after seeing a number is not a gate. (ii) **Per-task-class breakdown** — parse / typecheck / implement / refactor reported separately, because a mind that parses well and cannot refactor is not a development engine and an average hides exactly that. (iii) **Arm (b) is reported whatever it says**, with the retrieval quality of the native path measured against the helper path on the same queries; `RAG native: NOT DEMONSTRATED` is a permitted and useful outcome, and it is the one that keeps the 1080 Ti in the deployment. (iv) **The untrained/ablated control is the composed mind with `episodic_store` reverted**, which is the cheapest way to find out whether the store earns its place on a real workload rather than on a constructed one. **Verify the gate can fail:** run the suite against the comparison model twice and assert the margin statistic distinguishes nothing — an instrument that reports a win for a model against itself has been miswired, and this programme has already paid once for a saturated instrument. | **W10** (a proven composed mind, PTQ'd on the deployment card), P5′ | **blocked — long arc** | 5080 `.251` + 3090 `.98`, **RAG on 1080 Ti `.243`** |
| **P6′** | **Foundation / language trunk.** Unchanged in ordering — step 4, never step 1. Re-scoped: the causal LM is the **language faculty's generative head**, trained as a region under phase-1 discipline, not "foundation training of the whole model". | New prerequisite: **the general bin has a licence verdict** (FineWeb / C4 / Pile have none today [V\*]). **Deferred candidate corpus, recorded so it is not rediscovered:** official language documentation (Python/Rust docs already staged under `official-docs` in the tiered corpus, plus the RAG collections and the vault) — likely as a docs↔API retrieval bin or generative-head material rather than more contrastive pairs; it would require enrichment, structure extraction, version tagging, dedup against the code corpora and a per-doc-set licence verdict under the strictest-input rule. Revisit **only** when this row starts `[OP: csd-official-docs-corpus-idea]`. | W6 | blocked | 3090 + 5080 |
| **P2′f** | **THE DATASET FACTORY — the sourcing loop as one structured I/O contract** `[DEC-57] [OP: csd-dataset-factory-and-moral-corpus.md]`. **search → identify → capture provenance → licence verdict → ingest → emit a licence-compliant open dataset with full provenance**, drivable by an agent and producing a **human-visible provenance receipt per dataset**. **It GENERALISES what exists and rewrites nothing:** `scripts/csd-corpus-expand.py`'s catalogue entries already carry `provenance_group` and hold the **upstream licence verbatim as a field separate from the mirror tag** (which matters because the mirror tags were measured to disagree with upstream terms in ten cases, and all 75 sentence-transformers datasets declare none); the fetcher already **enforces `REFUSE`** rather than recording it; `LICENCE-FOR-OPEN-WEIGHTS.md` and `AUDIO-CORPUS-AUDIT.md` are the audit format; the corpus contract's **B1–B5** are the balance rules; NSRS admission and source-row fingerprints are the ingest filter. What is new is the **loop and its schema** — a machine-readable verdict per candidate, so an agent can drive the search half without a human re-reading every licence page. | **Four, and two are refusals.** (1) **The emitted dataset's licence is DERIVED by the strictest-input rule (DEC-31) and PRINTED**, not chosen — a mix containing one NC input emits NC, and the receipt names which input set it. (2) **A candidate whose upstream terms cannot be read at the primary source is `UNVERIFIED` and is REFUSED for training** (eval-only at most), constructed by pointing the factory at a source whose licence page 403s and asserting it does not ingest — *this project's own rule is that a doc does not beat a probe, and a factory that trusts a mirror tag contradicts the method that produced it.* (3) **A candidate whose distributor disclaims owning its own material is `REFUSE`**, constructed from a known case and asserted to write nothing. (4) **Positive control: one real dataset traverses the whole loop end to end** and its receipt reproduces the licence verdict, the provenance group, the row count and the dedup removals — a loop with three refusals and no successful traversal has not been demonstrated. | **P2′s** for the synthetic branch only; otherwise none | **todo — CPU, no GPU, and it is the long pole for DEC-56** | 1080 Ti / CPU |
| **P2′s** | **THE SYNTHETIC-DATA CONTRACT** `[DEC-58] [OP: csd-dataset-factory-and-moral-corpus.md]`. Generated data is admitted **only** under a contract, because *"generated data often carries a lot of garbage"* and the operator's stated preference is **extremely high quality and requirement-complete over large**. Four clauses: **the generator is NAMED in provenance** (model, revision, licence — extending §5.6's *"no generating model, or the model named"* rule from the reserve to every corpus); **contamination channels run against EVERY eval**, not only the one the batch was made for, with gated-channel counts printed; **a quality gate**; and **a capped share of any bin**, counted as a **B1 statistic over provenance groups** so one generator cannot become a monoculture wearing many names. | **Three, all failable, and the first is the one that matters.** (i) **THE QUALITY GATE CAN FAIL:** the synthetic bin must **beat a matched human-authored sample on the bin's own metric** — verified by feeding deliberately degraded generations (truncated, template-collapsed, self-similar) and asserting the batch is **discarded**, not warned about. A quality gate nobody has made fail is a preference. (ii) **A batch with no named generator is REFUSED at ingest**, constructed by stripping the field. (iii) **A batch that would push its bin's synthetic share over the cap is TRUNCATED OR REFUSED, and the receipt says which** — with the cap value printed beside the share, because a share reported without its cap is an opinion (§5.1's shape). | P2′f | **todo — CPU** | 1080 Ti / CPU |
| **P5′L** | **LAYER-SECTIONED TRAINING — evaluated as a candidate, beside DEC-29, not instead of it** `[DEC-55] [OP: csd-training-placement-policy.md]`. Training **isolated sections of layers/weights on different cards** rather than the whole model at once, so all three cards can contribute at 1B+ per region where otherwise later phases are **locked to the 3090 Ti + 5080, or to the 3090 Ti alone**. **The operator's own caveat is preserved rather than smoothed away** — they flag it as worth considering while noting they may be wrong about some of these techniques — which is exactly why it is a candidate row with a comparison and not a decision. | **A comparison, pre-registered, or the row is void.** Measured **wall-clock and interconnect bytes per section boundary** against **DEC-29's region-granular pipeline-parallel cost on the same model and the same microbatch ≤ 256**, on this fleet's real 1 Gb/s link — **the standing ruling that cross-host DDP is the wrong tool at 1 Gb/s is the null hypothesis, and this row either beats it or records that it did not.** Adopted only on a win; a null result is a **useful, publishable outcome** that keeps the two-card lock honest. | **P5′b** | deferred | all three GPUs |
| **P5′b** | **1B PER SUBMODEL, THEN QUANTISE, THEN THE COMPOSED MODEL — the post-phase-3 scale path** `[DEC-56] [OP: csd-billion-per-submodel-scale-path.md]`. Sequence, in order and not skippable: **mid-size proven** (DEC-52's baseline, W6 green, PTQ on the deployment card) → **~1B parameters per submodel** → **quantise each region** → **train the composed model on the quantised regions** → the 30B direction (OD-6). **The ordering "quantise the regions, then train the whole" is what P5′q tests at toy scale first: if P5′q wins, this row is its production form; if P5′q loses, this row's ordering is wrong and must be re-derived before any 1B run is spent.** Proceeds **region by region under DEC-50** — scale one region, retrain the interconnect, then the unified pass — so a regression is attributable. | **Three, and the first is the gate that decides whether the row is reachable at all.** (i) **THE DATA GATE: of the order 10^10 tokens per region**, each with an **upstream-verified** licence under DEC-31 and satisfying **B1–B5** on provenance groups — reported per region **before** any 1B training run is scheduled, because a 1B region trained on a corpus that fails the licence bar is unreleasable and unrecoverable. DEC-57's factory is the instrument. (ii) **Per-region PTQ sensitivity is RE-MEASURED at 1B**: the 3.2675 effective-bits/param figure is a **toy-scale measurement and is not carried forward**; the composed model's bits/param budget is derived from the new number, and DEC-27's `D_sched ≤ 0.02` nats is re-evaluated per region at size. (iii) **DEC-50's three steps per region**, each with its own receipt, and **the monotone-improvement rule**: a scaled region that does not improve the composed metric is **reverted to its mid-size checkpoint**. | **P5′**, P5′q, **P2′f** | deferred — post phase 3 | 3090 + 5080 (+ 1080 Ti under P5′L) |
| **P5′m** | **THE MORAL CORPUS — values at the training-data level, and whether that measurably works** `[DEC-59] [OP: csd-dataset-factory-and-moral-corpus.md]`. A **curated** corpus built to instil values at training time rather than filter them at inference time, with **its own CORPUS-CONTRACT entry**, its own provenance chain, and its own licence verdict under DEC-31. **Phase-3 / scale-up, explicitly NOT a toy row:** at 87M there is no behaviour to move and a null result would be uninterpretable, so running it early would burn the question. The operator's stated hope — that this improves model safety in a way industry could adopt — is a **claim to be measured**, which is what this row is for. | **Three arms, pre-registered, because a corpus that "clearly helps" without a control is the `residual_mlp` defect wearing a virtue** `[DEC-59]`. (i) **Held-out moral/safety probes**, constructed **before** training and **reserved under DEC-38/DEC-39** like every other sealed set — source-row fingerprints in `RESERVED.jsonl`, keyed split, and the training run **refuses to start** if a probe row is seeded. (ii) **Three matched arms:** the composed mind **with** the moral corpus, **without** it, and with a **size-matched neutral corpus** in its place — the third arm is what separates *"the values did something"* from *"more data did something"*. (iii) **The result is reported whatever it says**, including `moral corpus: NO MEASURED EFFECT`, which is a real and publishable outcome; and the probes' own contamination channels are run against the corpus, because a moral corpus that contains its own evaluation has measured nothing. | **P5′**, P2′f | deferred — phase 3 / scale-up | 3090 + 5080 |
| **P2′g** | **PER-FACULTY TOKEN TARGETS, CLEAN-TERMS GENERATORS AND ENRICHMENT — because a flat 10^10 is not reachable for four faculties from open data** `[DEC-73] [DEC-56 amended] [OP: csd-dataset-factory-pass1-result.md]`. Pass 1 measured the B1-bounded reach of every faculty (§5.8) and the spread is three orders of magnitude: `language_trunk` **6,359%**, `language_code` **15%**, `memory` **5.0%**, `moral_safety` **2.4%**, `reasoning` **1.8%** without a generator, **`visual` 0.3%**. **One number applied to seven faculties is an average nobody can act on.** Three deliverables. (i) **A written per-faculty token target with its reasoning** — what the faculty must do, at what parameter count, and therefore how many tokens — replacing the flat figure in DEC-56. (ii) **Clean-terms generators**, in the cosmopedia shape (an Apache-weights generator run locally, so the output carries no model-output terms): the DeepMind `mathematics_dataset` generator is the one settled-grant case in the whole catalogue and it is what moves `reasoning` and `numeric_math` from 1.8% to ≥100%. (iii) **The operator's own enrichment**, under the enrichment licence analysis already committed in `LICENCE-FOR-OPEN-WEIGHTS.md`. | **Four, and the last one costs nothing.** (1) **Every faculty carries a target and the reasoning behind it**, and a faculty whose target is still the flat 10^10 fails this row — the point is that the number was inherited, not derived. (2) **At least one clean-terms generator is run end to end** for a faculty that needs one, and its output passes **all four of P2′s's clauses** (generator named in provenance, contamination channels against every eval, a quality gate that can fail, a capped share of the bin). (3) **`visual`'s route is decided IN WRITING** with the number that motivated it — a licence-filtered Wikimedia Commons pull, a geometry change (W7v's rebuilt `image_size 128` / `patch_size 8` ⇒ `n_patches 256` geometry moves the 10^10 requirement from ~1.6e8 images to **~3.9e7**; the catalogue's ~5.1e7 assumes a 224²/16²/196-token geometry that is not W7v's), or a lowered target — because 0.3% is not a gap that a fetch priority closes. (4) **The model card STATES the CC BY-NC-SA reading it takes.** This needs no lawyer: today the card is silent, and OD-18's first item is unanswerable *because* nothing is stated. | P2′f (§4.1), **and `P2′f-b` in `program/REMAINING.md`** for the readings that do need a lawyer — *P2′f's three sub-rows `P2′f-a/-b/-c` live only in REMAINING, so this dependency is deliberately named across the two files rather than pointing at a row this table does not contain* | todo — CPU | 1080 Ti / CPU |
| **P5′d** | **DIFFERENTIAL ACTIVATIONS — activation deltas in a fast recompute-and-apply format** `[DEC-71] [OP: csd-memory-gate-overlays.md]`. Beside P5′o's weight-delta overlays, an overlay may also carry **activation deltas relative to the base activations**, encoded as a delta plus a cheap reconstruction rather than as dense stored activations, so a persona or skill contributes its features at inference **without holding dense state on the card**. It composes with quantised frozen regions (P5′q) and with DEC-70's arbiter — it is precisely the claimant class the arbiter evicts first, so its cost per feature is what decides whether it is worth being resident at all. | **A COMPARISON, not a demonstration, and both halves must win:** **apply latency per feature** and **VRAM held per feature**, each measured against a **dense-activation control** on the same items and the same card, with the reconstruction cost inside the latency number. **Adopted only if it wins on both**; a format that is smaller and slower to apply than dense is a different trade and is recorded as one rather than adopted. Plus P5′o's inherited clause: **applied then disconnected reproduces the base receipt metrics exactly, and the disconnect is logged.** | **P5′**, P5′o | deferred — post phase 3, explicitly not phase 2 | 3090 + 5080 |
| **P5′p** | **PREDICTIVE HYBRID TRAINING — extend the existing predictor from weights to activations and semantic residuals** `[DEC-72] [OP: csd-predictive-hybrid-training-track.md]`. `tzervas/tritter` already carries `GradientPredictor`, `PredictiveTrainer`, `LossPredictor`, `EmbeddingPredictionLoss` and a 962-line research report on predictive hybrid training; this row extends prediction from **(1)** weight-update outcomes to **(2)** activations and activation deltas — where it meets P5′d — and **(3)** semantic residuals used as a correction signal, alternating predicted phases with real backprop phases. **The tool stays in tritter under DEC-75; CSD carries an adapter only.** **Strictly after phase-3 whole-model training is underway** — packing (W7p/W7k) and chunking landed and written into the spec, phase-2 rows done — and the tritter documentation's own 25% backward / 15% forward claims are **inputs to this row, not results of it**. | **Four, pre-registered, and the third is the one that decides it.** (1) **Prediction error against a REAL step** on held-out batches, per target (weights / activations and deltas / semantic residuals), under a tolerance stated before the run. (2) **The fraction of steps predicted, printed** in the receipt — a method that predicts 2% of steps perfectly has not sped anything up. (3) **The end-task metric against a fully-real run at EQUAL WALL-CLOCK and at equal steps, both reported** — equal-steps alone flatters a method whose steps are cheaper, and equal-wall-clock alone flatters one whose steps are worse. (4) **AUTOMATIC FALLBACK:** a predicted phase that drifts past tolerance falls back to real steps on its own, verified by **constructing the drift** and asserting the fallback fires and is logged. Fine-tuning is measured as its own case, against its own real-run control. | **P5′** | deferred — post phase 3, explicitly not phase 2 | 3090 + 5080 |
| **M0d** | **DEPLOYMENT ACCEPTANCE — the composed mind must actually fit and serve on the two cards it is for** `[DEC-67] [OP: csd-mycelium-downstream-goal.md, fleet-gpu-roles-and-scheduling]`. M0 carries the deployment shape in its **task text**; a shape in a task description is not an acceptance criterion, and the card that is actually tight — the **5080, 16 GiB** — had nothing scheduled to find out. This row is that check, and it is a **prerequisite of M0 rather than a part of it**, so a deployment failure is not discovered while a readiness suite is running. | **Four, and (ii) is the one expected to be tight.** (i) **The composed, PTQ'd mind loads and serves on BOTH the 3090 Ti (24 GiB sm_86) and the 5080 (16 GiB sm_120)**, at the declared context length, with measured peak VRAM printed per card. (ii) **The store's DYNAMIC CAPACITY (DEC-63) is reported SEPARATELY IN BYTES on each card** — and **`capacity_bytes = 0` on the 5080 is a PERMITTED but RECORDED outcome** that fires §9.11's residual-rounds-to-zero finding and forces DEC-63's operator question (residual claim vs fixed floor). A footprint that reports parameters and omits the store has reported half the deployment. (iii) **The 1080 Ti (11 GiB sm_61) serves the RAG helper** (embedding + rerank) and its latency is measured on the same queries M0 will use — it **stays in the deployment unless M0's arm (b) demonstrates native retrieval**. (iv) **Verify it can fail:** run the same load against a deliberately over-sized context length and assert the row reports **FAIL** rather than swapping, thrashing, or silently truncating — a deployment check that cannot report "does not fit" is a launch script. | **W10** | **blocked — long arc** | 5080 `.251` + 3090 `.98`, RAG on 1080 Ti `.243` |
| **PRE-1** | **GATEWAY AUTHENTICATES ON EVERY ROUTE, BEFORE `proxy_upstream()` ATTACHES ANYTHING** `[DEC-60]`. **VERIFIED today, and it is worse than OD-2 described:** `dispatch_api_get()` performs **no caller authentication whatsoever**, and `dispatch_api_post()` authenticates **only `/api/apply`**; every other route — including **every proxied route, with `Authorization: Bearer {UPSTREAM_TOKEN}` attached by `proxy_upstream()`** — is served to any unauthenticated caller on `192.168.1.0/24` [V, `scripts/csd-lab-console:1012-1080`, survey `02-skeptic.md` S1/S2]. **OD-2's finding was that the gateway forwards the client credential; the real defect is that it never looks at one** — a textbook confused deputy with no identity to confuse, and the reason it has not bitten is that `PROXY_ALLOW` currently names five read-only routes. **Authenticate the caller on EVERY route, before `proxy_eligible()` and before any credential is attached**, and bind autodev's routes to the unix socket so they **404 on the TCP listener**. | **Two, both constructed** — this is the `tests/test_guards_can_fail.py` pattern applied to the gateway. (i) **An unauthenticated `GET` to an allowlisted proxied route is REFUSED**, asserted against the running unit, and **the upstream sees no request** — refusing after the proxy call is not refusing. (ii) **An authenticated caller whose identity is not on the route's allowlist is REFUSED**, so authentication is not mistaken for authorisation. | — | **todo — BLOCKS every autodev GPU route** | akula-prime, operator-owned |
| **PRE-2** | **TOKEN-GATE `POST :9108/v1/queue` AND MOVE THE WORKER OFF `kang`** `[DEC-60]`. **VERIFIED:** `_peer_ok()` returns true for **any** `192.168.1.*`, `172.30.*`, `172.32.*` or loopback source — **source-IP prefix only, no token** — and accepts `{kind, subject, extra{}}` into `enqueue_timeshare()`; 15 s later `gpu-timeshare-worker` runs the job **as `kang`** with `git/cabal-forgejo-agent` and `gpu/localai-api-key` **in its environment**, takes attacker-controlled `extra` fields **straight into argv**, and for the specialist kind **posts Forgejo comments under the agent identity** [V, `ansible/files/akula-health-exporter.py:520-596`; `scripts/gpu-timeshare-worker:60-170`; survey `01-threat.md` G17/G18/T4]. **IP-prefix authorisation is the "identity the client can set" anti-pattern, and this is the single most exploitable live path in the fleet.** Two structural changes: **(a)** a **bearer token bound to a producer identity** on `/v1/queue`, with `enqueue_timeshare` enforcing a **default-deny allowlist of `(identity, kind)` pairs**; **(b)** run the worker as **`svc-timeshare`, not `kang`**, with only the tokens that specific `kind` needs, injected per job via `secret exec`. Better still: make `/v1/queue` **loopback-only** and route cross-host handoff through the gateway like everything else. | **Three, all constructed.** (i) **An unauthenticated enqueue from a LAN address is REFUSED** and **nothing is queued** — asserted by reading the queue after the attempt. (ii) **An authenticated producer submitting an UNREGISTERED `kind` is REFUSED**, so a new job type gets no credentials until someone registers it. (iii) **`ps`/`systemctl show` confirms the worker's uid is NOT `kang`** and that its environment carries **only** the tokens its `kind` declares — verified by submitting one `kind` and asserting the other `kind`'s token is absent. **Until (iii) passes, the autodev sandbox's egress restrictions are moot**, because this path reaches the same credentials from the LAN. | — | **todo — P0; BLOCKS every autodev GPU route** | akula-prime, operator-owned |
| **PRE-3** | **REMOVE `kb_http`'s LOOPBACK FAIL-OPEN** `[DEC-60]`. **VERIFIED:** `_auth_ok()` contains `if self._loopback() and not token: return True` — it **fails OPEN on loopback when the token file is absent or empty**, while non-loopback with no token fails closed [V, `rag/integration/kb_http.py:351-360`; survey `01-threat.md` G15/T6]. Today the token file exists at 0600, so the window is small — **but the whole autodev design leans on the gateway and this store as its enforcement points, and a fail-open branch inside one of them is disproportionate to its size.** If `/akula-data/cabal/kb-http.token` is ever deleted, rotated to empty, or the service starts before the file is placed, **every loopback caller reads all three corpora unauthenticated**. **Delete the bypass; require the token unconditionally; `SystemExit` at startup when it is absent, so the failure is a dead service rather than an open one.** Related and cheap while the file is open: client-controllable audit detail (`X-Akula-Temporary: 1` makes the handler write `detail="temp"` instead of the real detail) is **client-controlled evidence** and should be recorded server-side [V, `kb_http.py:366-372`]. | **Two, both constructed.** (i) **Move the token file aside and restart: the service must EXIT, not serve** — assert the unit is failed and the port is closed, which is the whole difference between fail-closed and fail-open. (ii) **A loopback request with no `Authorization` header is 401** with the token file present — the state the code claims today, asserted rather than assumed. | — | **todo — small and independent; BLOCKS any autodev RAG route** | akula-prime, operator-owned |

**REVISION 3.4's ELEVEN NEW ROWS, AND WHERE THEY SIT IN THE ORDER.** The table above is ordered
by dependency, not by row id, and revision 3.4's additions do not all belong in one place — so
this note says explicitly where each one sits rather than leaving a reader to infer it from the
`blocked_by` column:

| row | sits | why |
|---|---|---|
| **W7p** | **now, beside W0c** | tooling with no GPU dependency for its first two deliverables; it de-risks W4, W7a and W7v and blocks none of them, so doing it late buys nothing `[DEC-54]` |
| **W9i** | after **W9** | it instruments an emitted `Schedule`, which does not exist until W9 `[DEC-53]` |
| **M0d** | after **W10**, before **M0** | a deployment failure must surface before a readiness suite is run against a mind that does not fit `[DEC-67]` |
| **P2′f**, **P2′s** | **now, CPU-only** | the dataset factory is the long pole for DEC-56 and it needs no GPU; starting it late is what makes the 1B path unreachable `[DEC-57, DEC-58]` |
| **P5′b**, **P5′L**, **P5′m** | post phase 3 | all three depend on a proven composed mind; P5′b additionally depends on P5′q's verdict, because its whole ordering is P5′q's hypothesis at scale `[DEC-55, DEC-56, DEC-59]` |
| **PRE-1**, **PRE-2**, **PRE-3** | **outside this programme's dependency graph, and blocking a different one** | they are fleet-configuration rows in another tree, recorded here only because this design's autodev interlock depends on them (§8). **They block any autodev GPU route; they block no CSD row** `[DEC-60]` |

**The `PRE-` prefix is new and was chosen to collide with nothing** — `A*` is the audio namespace
*and* the first skeptic pass's findings, `S*` is the third pass's, `W*`/`E*`/`P*`/`M*` are taken,
and `X*` is §5.4's item shapes. The document has already paid once for the `A1`-row-versus-`A1`-finding
collision (the naming note above), and paying twice would be a choice.

### DEC-50 — the INCREMENTAL INTEGRATION PROTOCOL, and why E0–E2 are three rows and not one

**This is a named procedure, not a description of what E0–E2 happen to do.** Every future region —
`numeric`, the language trunk, `auditory` when the production phase opens — follows the same three
rows, and a proposal that adds a region without them is incomplete on its face.

**The operator's statement, which is the source** `[OP: csd-incremental-integration-protocol.md]`:

> *"one of the other benefits of this kind of modular design for the sub models is that if I make
> architectural changes — since they're all actually unified and called upon and utilized via the
> interconnect and router — it would likely mean that at least for that first level we would need
> to train the new memory module or any other module/submodel that we integrate, and then just
> retrain the interconnect to enable and facilitate leveraging that new module/submodel, and then
> follow up with the subsequent training for the whole model, like a unified approach."*

**The three steps, each with the gate that lets it fail:**

1. **TRAIN THE SUBMODEL ALONE**, under phase-1 discipline: its own corpus contract, its own
   contamination channels in the receipt, and it must **beat its own untrained baseline
   instantiated at a region-specific seed** — never seed 0 (§4.0). *For `episodic_store` this step
   is a build rather than a train, because the store is non-parametric; the discipline is
   unchanged and the baseline becomes E0's contract tests failing red against a stub.* **Row E1.**
2. **RETRAIN THE INTERCONNECT** with the new region admitted: white matter **A (dense) → B (distil)
   → C (sparse) → D (task-loss)**, regions frozen. **Gates: G2** — no existing faculty regresses
   more than 1 point on its own bin; **the composed metric improves**; and **the new region shows
   non-zero connectivity / attention mass**. **Otherwise the region is REVERTED**, and the receipt
   records which of the three failed. **Row E2.**
3. **WHOLE-MIND UNIFIED TRAINING** (phase 3) with the new region in place, under the same
   monotone-improvement rule already written into P5′: growth increments that do not improve the
   composed metric are reverted. **Row P5′**, which needs no amendment — it already says this.

**Why step 2 is cheap, and this is the load-bearing claim.** **DEC-26 caps the interconnect's
parameter share at 2–5% of the mind** (scoped to ≥1B; at toy scale it is 31.8%). So the object
being retrained when a region is added is the **smallest trainable object in the architecture**,
and the regions — the expensive things — are not touched at all. That is the property the modular
design was bought for, and DEC-50 is the procedure that spends it. **The reserved corpus is the
interconnect's own training data**, so step 2 has signal without reaching into any region's corpus
and without invalidating any region receipt.

**Per-step receipts, so a regression is attributable.** Each of the three steps emits its own
receipt. If the composed metric falls after a region is added, the receipts say whether it fell at
the region, at the interconnect retrain, or at the unified pass — **three different findings with
three different remedies**, and a single end-to-end receipt cannot distinguish them. This is the
same discipline W4 already uses when it makes a merge and an objective change diagnosable by
ablating one term.

**What the protocol does NOT license.** It does not license adding a region *because* re-admission
is cheap. Step 2's second gate is *the composed metric improves* — not *does not regress* — for
exactly that reason: a participant that consumes read budget and returns nothing is a cost, and
the cheapness of the retrain is what makes it tempting to keep one anyway.

### Production phase: audio — DEFERRED rows, kept specified `[DEC-48]`

**Operator ruling, 2026-09-02, later the same day than the intent that created these rows and
superseding it as a schedule** `[OP: csd-multimodal-io-intent.md]`:

> *"we can wait to add audio as a future feature once it proves out without audio. that will be
> more of a production phase implementation"*

with *"gotta walk before we run"*. So the four audio rows leave the phase-1/phase-2 table above
and live here, **status `deferred`**, every one of them `blocked_by` **the composed mind passing
W6 without audio, plus an explicit operator go**. Two things this subsection is careful about,
because a deferred row rots in two different ways:

- **The seam stays declared.** `auditory`'s interface, `speech_output`'s two-stream shape, the
  objective families, DEC-45's tiers and §2.5's `output` block are all **kept in the design**
  (DEC-43, DEC-44 as amended). Deferring the work is cheap; deferring the *contract* is what
  forces a redesign, and it is not done here.
- **A deferred row still needs a gate that can fail when it is picked up.** Every gate below is
  written to be constructible on the day someone runs the row, and the revision-3.1 skeptic
  findings against these gates are **fixed here rather than parked**, because a gate nobody can
  fire is exactly as broken whether it fires next month or next year `[S31-3 fixed]
  [S31-4 fixed] [S31-5 fixed] [S31-7 fixed] [S31-8 fixed] [S31-9 fixed] [S31-18 fixed]
  [S31-21 fixed]`.

**Groundwork already banked, and it is not re-done when the phase opens:** the audio licence
audit (`docs/design/AUDIO-CORPUS-AUDIT.md`), the A0 source catalogue, DEC-45's tiers and DEC-46's
provenance-group correction. **A bounded first audio fetch may already exist on
`gpu5080:/bulk`.** It is **re-acquirable scratch** — nothing in this document depends on it, no
receipt cites it, and it may be deleted without loss.

| id | task | gate | blocked_by | status | host |
|----|------|------|-----------|--------|------|
| **A0m** | **Audio corpus MANIFESTS** (DEC-43, DEC-46; `docs/design/AUDIO-CORPUS-AUDIT.md`). The manifest half of revision 3.1's A0, split out because it genuinely has no dependencies while the fetch half does `[S31-7 fixed]`. Per-source row: upstream URL, mirror id, **`mirror_tag` and `licence_upstream` as separate fields**, **`licence_upstream_source`** (the URL actually fetched **plus its fetch date**), **`grant_scope` ∈ {whole_corpus, metadata_only, code_only, unstated}**, the upstream text quoted verbatim, train verdict, redistribute verdict, NC/SA/ND/attribution flags, size **in hours**, provenance group, provenance red flags. **Scope covers BOTH mixes** — the `auditory` mix and the `speech_output` mix (bucket C) `[S31-8 fixed]`. | **Gate (i), rewritten because revision 3.1's version fired on the wrong rows** `[S31-3 fixed]`. It read *"a row where `mirror_tag` and `licence_upstream` hold the same string because only the mirror was read is a FAIL"* — string equality, which **FAILs every correctly audited row where the two genuinely agree** (LibriSpeech, MLS, AMI, MUSAN, VCTK, Hi-Fi TTS, LibriTTS-R, AISHELL-3, Expresso — the audit *celebrates* that agreement for LJSpeech) and **passes AudioSet**, the one row it was written for, because there the two strings also agree and the defect is the grant's *scope*. **The gate keys on the PROVENANCE OF THE READ, not on the value read:** a row **FAILS** when `licence_upstream_source` is absent, when its **host equals the mirror host**, or when its **fetch date is missing** — and separately when `grant_scope` is unset. `metadata_only` is what catches AudioSet; `code_only` is what catches CSS10 and Libri-Light. **Verify it can fail:** construct a row whose `licence_upstream_source` points at the mirror and assert **FAIL**; construct a correctly audited agreeing row and assert **PASS**. | — (**no dependencies at all**; runnable the day the phase opens) | **deferred** | 1080 Ti / CPU |
| **A0f** | **Audio corpus FETCH** under the audit's recommended v1 mix, with LibriVox capped as ONE provenance group. Wire the fetcher to A0m's manifest, fetch **the clean tier of both mixes**, and compute the balance numbers **on provenance groups, not dataset names**, **in hours** `[S31-14 fixed]`. | **Four, and (ii)–(iv) are constructed to fire.** **(ii)** the fetcher **refuses every source the audit marks BLOCKING**, verified by *constructing* the case: put `agkphysics/AudioSet` in a scratch manifest and assert the fetcher raises and writes **nothing**. **(iii)** **B1 and B2 are computed with LibriVox as ONE group, in HOURS, for BOTH mixes, and printed beside the same statistics computed per dataset name** — the per-name numbers are expected to look fine, and that is the point. **This clause is a PASS/FAIL, not a print** `[S31-2 fixed] [S31-5 fixed] [S31-15 fixed]`: the grouped max share is reported **against both thresholds** — **> 0.50 FAILS the row** (the hard line) and **> 0.40 is a recorded WARNING** (the operating cap), the shape §5.1 already uses for aqua_rat — and a **grouped `N_eff` below 3 FAILS**. The chosen per-group cap `C` is printed with them, because B1/B2 are functions of `C` and a receipt that omits it has reported an opinion. **(iv)** the LibriVox group's **per-speaker hour histogram** is produced and B5's max share printed. **(v)** every fetched source's licence tier is recorded so DEC-45's table can be **recomputed** rather than trusted. **Verify it can fail:** the constructed BLOCKING row, and a scratch mix built per dataset name that the grouped statistic must FAIL while the per-name statistic passes. | **A0m, OD-13, OD-15** (the fetch half is what those two rulings bind; OD-10/11/12/14 concern sources outside the recommended mix and do **not** block it `[S31-7 fixed]`) | **deferred** | 1080 Ti / CPU |
| **A1** | **`auditory` region pretrain** (DEC-43). Spectrogram-frame patch tokens, masked-latent prediction, the same JEPA harness `visual` uses, over A0f's capped mix. Emits at the shared `tokens()` interface — **position latents, `[DEC-47]`** — so W0 is a prerequisite and the adapter is the one `visual` already has. | **Five, all pre-registered:** (1) **beats its own untrained baseline, instantiated at a region-specific seed and not seed 0** (§4.0); (2) **the contamination channels are present in the receipt** — `corpus.cap_sampling`, the drawn `pair_fingerprint`s and the source-row fingerprints; (3) **anisotropy recorded** — **entropy-effective** rank of `tokens()` on 4,096 real mixed items in W1c's units, printed beside participation ratio, the two never compared across definitions `[N3 fixed]`; (4) the eval split is declared **in the code** as `in-mixture` or `held-out-domain` **before the first receipt**; (5) the licence tier of the mix actually consumed is printed. **Verify the gate can fail:** run the eval against the untrained checkpoint and assert **FAIL**. | A0f, W0, **operator go** | **deferred** | 3090 `.98` |
| **A2** | **`speech_output` head** (DEC-44). A second head on the generative trunk: BPE text tokens plus a discrete speech-token stream a synthesiser consumes. Trained on **bucket C** of the audit — which is why A0m/A0f's scope covers bucket C `[S31-8 fixed]`. **THE TRUNK IS FROZEN AND THE HEAD IS AN ADAPTER** `[S31-4 fixed]`: revision 3.1 left the trunk's status unstated while asking for a disconnect test that reproduces the text head's receipts *exactly*, and the two readings could not both hold — a trainable trunk makes "exactly" fail on the fourth decimal by construction, a frozen trunk makes it trivially true. The frozen reading is **adopted**, because it is the one that preserves reversibility for OD-15 and matches DEC-33's overlay discipline, which is only correct for a genuinely detachable thing. | **Five.** (i) **intelligibility** — **WER of a fixed, named, frozen reference ASR** over synthesised output on a **held-out** set beats the untrained head's WER by a margin pre-committed before the run; the ASR's model id, revision and licence are recorded and **the same ASR is used for every later comparison**. (ii) **the disconnect test, now well-posed:** with the speech head detached, the text head reproduces its own receipt metrics **bit-exactly on the logits** — which is achievable *because* the trunk is frozen — and the disconnect is logged. (iii) **the licence tier is printed** (OD-15). (iv) the untrained-head baseline is at a region-specific seed. (v) **balance:** the **grouped B1 of the consumed TTS mix is printed and a share > 0.50 FAILs the row** `[S31-5 fixed]` — four of six clean-tier `speech_output` sources (LJSpeech, CSS10, M-AILABS, Hi-Fi TTS) are **one LibriVox group**, leaving VCTK, AISHELL-3 and (NC) Expresso as the only levers, which makes OD-15 a **balance** question and not only a licence one. **Verify it can fail, in two directions** `[S31-18 fixed]`: feed the reference ASR the **untrained** head's output and assert FAIL — *and*, because that construction only proves the comparison is wired, feed the frozen ASR **silence or white noise** and assert **WER ≈ 1.0**. A stub that returns the reference transcript regardless of input passes the first check and then reports a spectacular improvement; this programme has already paid once for a saturated instrument. **A2 does NOT precede W2b** — with the trunk frozen it changes no region's weights, so the `s_r` invalidation rule does not reach it. | A0f, W0, **operator go** | **deferred** | 3090 `.98` |
| **A3** | **The two audio cross-faculty item shapes for the reserve** (§5.4, §5.5). *`auditory × language_code`*: an A0 transcript-carrying source row yields (spectrogram frames, the transcript span, a **distractor span drawn from a different source row of the same provenance group**), and the item asks which text span the audio realises — built by **pairing existing aligned data**, §5.5(a)'s cleanest tier. *`auditory × visual`*: W3r's checked-in 128×128 composite rendered from a cleared text row, paired with audio of that same row read by a clean-tier speaker, plus a shuffled-mismatch negative. Both obey DEC-38's fingerprint granularity and DEC-39's keyed split. **When this row is picked up it re-opens §5.4's pair arithmetic, and the re-opening is pre-specified rather than left to the day:** adding `auditory` makes the participant set **six** — DEC-49 already made it five — and the all-pairs set **fifteen**, of which `auditory × memory`, `auditory × reasoning` and `auditory × episodic_store` have no shape here. **Pre-committed resolution, restated for the five-participant base because a pre-specification that is not re-derived when its base changes is a stale number waiting to be quoted** `[S31-1 fixed] [DEC-49]`: `auditory` enters as a **non-pair participant** unless A3 also declares shapes for those three, the pair set is held at **10 + 2 = 12**, and G3′'s criterion is restated **in the same breath** as **≥ 8 of 12** with its null rate `P(Bin(12,0.5) ≥ 8) = 0.194` printed beside it — the smallest `k` under §2.7.3's 0.20 ceiling. | **Gate.** Items are **constructed** on CPU as soon as A0f lands and **admitted** — NSRS `s_r` for every region *including* `auditory` — only after A1. ≥ 512 admitted items per new pair for eval and ≥ 5,120 per pair for train; **the sealed-item allocation across pairs is recomputed and printed, and the recomputed table must print the cross-faculty sealed TOTAL alongside the per-pair figures so the two are visibly reconciled** `[S31-21 fixed]` — revision 3's per-pair column sums to 1,280 against a 2,560-item sealed half because shapes overlap pairs, and A3 inherits that ambiguity unless it prints both. Every item's provenance chain names its audio source and licence tier. **Verify it can fail, and the refusal now has a key to fire on** `[S31-9 fixed]`: revision 3.1 asked W2b to *"refuse the pair rather than score it against a random encoder"* and named **no field the filter could key on** — a randomly initialised encoder emits outputs and `s_r` is perfectly well defined, just low, so the construction produced a **number** and the gate could not fire. **W2b refuses any region whose `status` is not `built` AND whose receipt path is absent or whose checkpoint hash does not match the resident weights** — which gives §1.4's `planned` a second, mechanical job and gives the `s_r` invalidation rule the enforcement point it otherwise lacks. Construct it: attempt admission with `auditory` at `status: planned` and assert **refusal**, then with a receipt whose hash is flipped and assert **refusal** again. | A0f, A1, W2b, W3r, **operator go** | **deferred** | 1080 Ti / CPU **+ 3090 for `s_r`** |

**What this subsection deliberately does not do.** It does not schedule audio, does not budget
GPU-hours for it in §4.3, and does not carry `auditory` into any participant count, pair count,
collapse floor or licence table for v1. **v1 has FIVE participants and TEN ablation pairs**
(DEC-49), **none of them audio** — that is what this subsection is asserting, and revision 3.3
restates the numbers rather than leaving a count that was written when they were four and six.


## 4.2 Consequence-3: what was predicted, what was measured, and what it cost

*"A region trained in isolation optimises to solve its task ALONE… cheap to test early,
expensive to discover late"* [V\*]. All seven existing regions were trained in isolation with
contrastive / BCE / JEPA objectives [V\*]. **The mechanism of the harm, stated precisely in
revision 1:**

> Mean pooling gives `∂pooled/∂h_t = m_t / Σm`, **identical for every unmasked position**. So
> InfoNCE supplies **no positionally differentiated gradient**: the loss shapes only the *mean*
> of the token sequence. Whatever structure the pre-pool sequence has is a byproduct of the
> bidirectional blocks and the lexical statistics, never something the objective asked for. [I]

**That prediction is now measured and it is correct** (§4.0). The pre-committed rule fired, and
it fired against the design. Three things follow that are worth stating plainly, because a
programme that only records its wins is not measuring anything:

1. **The cheap test paid for itself exactly as argued.** W1 cost hours and cancelled nothing —
   it *bought* a mandatory retrain before a white-matter run was spent on a token surface that
   carries four usable directions. Discovered after W5, the same fact would have invalidated
   W5, W5b, W6 and every number in them.
2. **The statistic was biased toward the answer the design wanted, and still said no**
   `[A9 fixed]`. A9 argued that participation-ratio rank of the token cloud is essentially
   always ≥ that of the pooled cloud — the pooled vectors lie in the span of the token vectors
   and there are `N` of them rather than `N·T` — so *"token rank materially higher"* is close to
   a tautology and *"the bet is dead"* was close to unreachable. **The measurement refutes the
   premise for trained encoders** (three of four ratios came out **below** 1.0; see §11 R2) and
   reaches the branch A9 called unreachable. The correct response is not to declare A9 wrong and
   move on: it is to keep A9's *fix* as the confirmation step (W1d) precisely because the
   decision it now licenses — a mandatory retrain of every production region — is the most
   expensive thing in phase 2.
3. **The FLAG thresholds now trigger something.** Revision 1 asked W1 to *"FLAG any region with
   token-level effective rank < 8 or pairwise CKA > 0.90"* and attached no consequence, so the
   flags were decoration. The `< 8` flag fired for `code` (4.4) and `retrieve` (5.1) and is
   wired: a flagged region's retrain in W4/W7 must clear the per-item rank threshold as well as
   the global one, or it goes to the penultimate-block fallback.

**What is genuinely CPU-parallel, corrected** `[A7 fixed]`. Revision 1 claimed *"the corpus work
(W2, W3) runs on CPU hosts in parallel with W0/W1/W1b — it is not serialised behind any GPU
experiment"*, and presented it as the plan's scheduling advantage over the rival proposals. It
is false as specified: W2's gate requires `s_r` for every region on every candidate row, which
is inference through every frozen region — including `visual`, a ViT, and including `memory`,
which **does not exist until W4**. The split is now honest:

| genuinely CPU-only, genuinely parallel | GPU-blocked, and blocked by the region set |
|---|---|
| W2a (ledger + aqua_rat recovery) · **W0c** (DEC-40's loader + lint) · **W3r** (frame geometry, minimal renderer, fixture) · BM25 condition (3) · licence audit · apps↔code_contests dedupe · **composite rendering, but only AFTER W3r fixes the geometry** `[N5 fixed]` · the executable-join harness (in its sandbox). **The audio rows are no longer on this list** — DEC-48 defers them; A0m keeps its "no dependencies" property for the day the phase opens | W2b's `s_r` computation · W2c's untrained baselines · every admission decision · every baseline derived from them |

One correction to that left column since revision 2: **composite rendering was listed as
CPU-parallel beside W7v, and it cannot be** — the renderer cannot choose a frame geometry until
somebody fixes `image_size` and `n_patches`, and rendering four-panel composites at 64×64
produces exactly the thing §5.5(b) calls *"not legible to any encoder"*. W3r fixes the geometry
first, on CPU, with no dependencies, so the parallelism claim becomes true instead of nearly
true `[N5 fixed]`.

## 4.3 The phase-2 retrain budget, since there is one now `[A30 fixed]`

Revision 1's affordability argument was *"phase 2 retrains nothing… that property is what makes
the whole programme affordable"*. W1's result ends that. The replacement is a budget, built from
the measured wall-clocks in the surviving receipts (text regions **426.7 s / 595.1 s / 1,087.2 s**
at batch 256, 8,000 steps; `vl_latent` **496.7 s** at 64×64 over 100k images) [V\*].

| row | runs | basis | estimate |
|---|---|---|---|
| W1b regenerate `reason` | 1 | text region at `max_len 256`, 12,455 rows | ~600 s [I] |
| W1d matched read-out probe | 3 arms | small read-out over cached frozen activations | **116.5 s MEASURED** [V] (est. was ~1 h) |
| W2c untrained baselines | 2 | eval-only, no training | minutes |
| W4 `memory` merge + token-aware | 1 (+1 revert-and-retry allowance) | merged trunk, two heads, three loss terms | ~1,500 s [I] |
| W7a `language_code`, `reasoning` | 2 (+2 allowance) | text regions, three loss terms | ~2,000 s [I] |
| W7v `visual` resolution + token-aware | 1 (+1 allowance) | **4× patch count at 128×128**; attention is `O(N²)` in patches, so 497 s does not scale linearly | ~1–2 h [I] |
| **phase-2 region retrain total** | | | **≈ 3–5 GPU-hours on the 3090 Ti [I]** |

**The baselines are training runs too, and revision 2 did not budget four of them** `[N7 fixed]`.
§2.7.1 lists seven baselines. **B0's read-out, B0d, B0u and B2t are trained artefacts revision 1
did not require**, and §2.7.7's cost model counted forward passes over four model variants,
omitting B2t and B3 entirely. They are enumerated here in the same units as everything else.

Let **`T_A`** be one W5 phase-A dense white-matter run. It is not yet measured — no white-matter
run has been done — so it is estimated and then **replaced by the measured number in W5's
receipt**: 8,000 steps at batch 256 over the **61,440**-item reserve (§5.4, DEC-49), with `R` frozen region forwards
plus the workspace per step, i.e. **3–5× a text-region step** whose measured wall-clocks are
426.7 s / 595.1 s / 1,087.2 s [V\*] ⇒ **`T_A` ≈ 1–2 GPU-hours** [I].

| baseline | what it trains | runs | cost in `T_A` | estimate |
|---|---|---|---:|---|
| **B0** | `rank_head` only, over a **frozen** random-init workspace; no workspace backward | 1 | **≈ 0.15** | ~10–20 min [I] |
| **B0d** | a top-1 router over pooled region outputs; no workspace at all | 1 | **≈ 0.2** | ~15–25 min [I] |
| **B0u** | full training with cross-attention **frozen uniform from initialisation** — same params, same steps, same data as the trained mind (A12) | 1 | **≈ 1.0** | 1–2 h [I] |
| **B2t** | **a second full white-matter-scale training run.** "Same parameter count and training budget" is the definition of the baseline; there is no cheaper way to have it | 1 | **1.0, by definition** | 1–2 h [I] |
| **B3** | frontal-only, all regions gated off — fewer KV rows, same schedule | 1 | **≈ 0.8** | ~1–1.5 h [I] |
| **baseline total** | | 5 | **≈ 3.15 `T_A`** | **≈ 3–6 GPU-hours [I]** |

So the integration experiment costs **the trained run plus roughly three more of it**. Two
consequences, both stated rather than absorbed:

1. **The baseline bill is the same order as the entire region-retrain bill** (3–5 GPU-hours), and
   it was invisible in revision 2. Phase 2's honest total is **≈ 6–11 GPU-hours of 3090 Ti time**
   before the white-matter runs the plan always had.
2. **B2t is the one that will be tempted away**, because it is a full run whose only purpose is
   to lose. It is also the baseline an outside reader will ask for first. **Pre-commitment: if
   B2t is not run, G3 is reported as `not tested against a trained null` and the receipt says so
   — the gate is not quietly re-scoped to B2 alone.**

**The audio bill revision 3.1 added is REMOVED FROM PHASE 2, because DEC-48 defers the rows.**
Revision 3.1 costed A0–A3 at **≈ 3–6 GPU-hours** (A0 zero, A1 ~2–4, A2 ~1–2, A3 zero beyond
W2b's `s_r` pass) and moved phase 2's honest total from ≈ 6–11 to ≈ 9–17 GPU-hours. **That
estimate is withdrawn from the phase-2 bill and travels with the deferred rows** to the
production phase, where it will be re-derived rather than inherited — it was the
weakest-founded table in this section, with **no audio receipt anywhere in this tree** behind it,
and an estimate that is not spent for a year is not an estimate. **Phase 2's honest total returns
to ≈ 6–11 GPU-hours of 3090 Ti time**: ≈ 3–5 of region retrains plus ≈ 3–6 of baseline training
runs. When the production phase opens, **A1 must record its measured wall-clock** so the audio
table is replaced by a number rather than repeated as an estimate — the same discipline `T_A` is
under.

**One property of the deferral worth stating, because it is the cost side.** A0's work was the
longest-lead item in the programme and it cost **zero GPU-hours**; deferring it defers wall-clock
and human attention, not compute. The audit is already done, so what the deferral actually spends
is the option value of having had the corpus ready — and the ruling accepts that explicitly:
*"gotta walk before we run."*

**DEC-49 ADDS ONE ROW TO THIS BILL AND IT IS A FULL WHITE-MATTER RUN.** The episodic store's three
rows cost, in the same units:

| row | runs | basis | estimate |
|---|---|---|---|
| **E0** contract extraction and test port | — | CPU, no model, no data | **zero GPU-hours** |
| **E1** store build + dynamic-capacity probe + fuzz | — | no training; the probe reads `nvidia-smi` and the scheduler's budgets, the fuzz is 10,000 store operations | **minutes, on two cards** [I] |
| **E2** interconnect retrain with the store admitted | 1 (+1 revert-and-retry allowance) | **`R = 5` phase A — by definition one `T_A`**, DEC-50 step 2 | **≈ 1.0 `T_A` ⇒ 1–2 h** [I] |
| **DEC-49 total** | | | **≈ 1–2 GPU-hours on the 3090 Ti [I]** |

**So phase 2's honest total moves from ≈ 6–11 to ≈ 7–13 GPU-hours of 3090 Ti time**: ≈ 3–5 of
region retrains, ≈ 3–6 of baseline training runs, ≈ 1–2 for E2. **E2 is the one line here that is
`1.0 T_A` by definition rather than by estimate**, for the same reason B2t is: a phase-A run with
one more participant is a phase-A run. **The thing that keeps this affordable is DEC-26** — E2
retrains 27.4M parameters and touches none of the 58.9M in the regions — and that is DEC-50's
whole argument stated as a bill.

**Two costs of DEC-49 that are NOT GPU-hours and are recorded here so they are not lost:** the
reserve grows by **11,264 items** (§5.4), which is human and CPU time on construction rather than
compute; and W3 acquires the **episode harness**, a component this programme has never built,
which is the likeliest place for the store's four ablation pairs to slip. §5.4 pre-commits the
branch for that.

**Gate on the estimate itself, so it can fail:** W5 records its measured phase-A wall-clock. **If
`T_A` exceeds 3 GPU-hours, the baseline set is re-scoped before B2t is launched** and the
re-scoping is recorded, rather than discovered by a run that does not finish. **E2 inherits the
same gate**: it is priced in `T_A` and is re-scoped by the same rule.

Against that, the white-matter runs (W5 phase A, W5b, W8 phases B/C/D) are the larger cost and
were always in the plan. **The programme is still affordable; it is no longer affordable *for
free*, and the argument is now a number a reader can check rather than a property.** The honest
summary: W1 converted an unstated risk into a stated, bounded, three-to-five-GPU-hour bill, paid
before anything expensive depends on it.


# 5. P2.5d — the reserved corpus

## 5.1 The corpus crisis was largely self-inflicted, and W2a recovers it `[A5 fixed]`

**244,761 rows** — verified by summing the source table: aqua_rat 97,467 + fashion_mnist 60,000
+ go_emotions 43,410 + code_contests 13,328 + banking77 13,083 + apps 10,000 + gsm8k 7,473 =
244,761 [V\*, recomputed independently by all three proposals].

**≈56,320 rows / 23.0%** — verified as a *derivation*, not an assertion: `(7 bins × 512) +
(3 pairs × 512) = 5,120` eval, `×10` train `= 51,200`, total `56,320`; `56,320 / 244,761 =
23.0%` [V\*]. **DEC-49 re-derives this at six per-faculty bins and six cross-faculty bins:
`(6 × 512) + (6 × 512) = 6,144` eval, `×10` train `= 61,440`, total **67,584**; `67,584 / 244,761
= 27.6%` [I] (§5.4).** The published 23.0% is the *verified original derivation* and is kept as
such; 27.6% is what the same derivation yields at `R = 5`. The 512 comes from the 95% binomial half-width `1.96·√(0.25/n) = ±4.33 pp` at
`n = 512`, matching the project's existing `holdout_pairs` convention [V\*].

### The aqua_rat write-off was wrong, and the sampling code refutes it

Revision 1 wrote off **all 97,467 aqua_rat rows** on the grounds that *"4,982 rows were drawn by
reservoir sampling, but the `reason` receipt that recorded `corpus.cap_sampling` was deleted, so
the drawn ids are unrecoverable and no aqua_rat row can be proven clean… a hard, unfixable
contamination boundary costing 92,485 otherwise-usable rows."* That single write-off produced
the document's headline corpus crisis, the *"2.38× shortfall"*, risk §9.2, and the necessity of
the rendered-composite generator.

**One clause of that premise is corrected before anything is built on it** `[N1 fixed]`. *"the
`reason` receipt that recorded `corpus.cap_sampling`"* should read *"the receipt that **would
have** recorded `corpus.cap_sampling`"*. **No surviving receipt carries that field at all** — all
nine receipts under `/akula-data/csd/receipts/` were enumerated key by key and `cap_sampling` is
present in none of them [V\*], including ones written after the commit that introduced it.
`config/seed = 0` **is** present in all nine, which is what supports precondition (iii) for the
surviving regions. The correction matters because it changes a reader's estimate of how
recoverable this is: the field was never written, so W1b's requirement that the regenerated
receipt carry it is **new instrumentation, not a regeneration** — and that is precisely why this
loss cannot recur once W1b lands.

**The draw is deterministic and reproducible from config alone. Re-read this session [V]:**

```
src/cogsyndelta/regions/pretrain.py:231
    return reservoir_sample(stream, limit, sampling_rng(seed, shards, columns))

src/cogsyndelta/regions/pretrain.py:210-213   (load_pairs docstring)
    "Reservoir sampling gives every row an equal chance regardless of position, and the
     generator is derived from `seed` (see sampling_rng) so TWO RUNS OF THE SAME CONFIG
     STILL GET THE SAME ROWS -- receipts carry corpus fingerprints and would mean nothing
     otherwise."

src/cogsyndelta/corpus.py:159-181             (sampling_rng)
    "Deriving the seed from the run seed plus the source's identity makes each source's
     sample independent while keeping the whole thing reproducible from `cfg.seed` alone.
     shards: ... only BASENAMES are used, so moving a corpus between mounts does not
     change which rows it yields."

src/cogsyndelta/corpus.py:189-221             reservoir_sample = Algorithm R over a
                                              deterministic stream; its only randomness is
                                              the `rng` it is handed.
src/cogsyndelta/regions/pretrain.py:177-193   _iter_pairs yields "in shard order".
scripts/csd-train-all.py:325
    ("reason/aqua_rat-raw/train.parquet", ("question", "rationale"), 4982)
```

Every input to the draw is pinned and survives: the seed, the shard **basename**, the column
names and the cap. **The receipt did not hold the draw; it merely recorded what the code
recomputes.** And revision 1's own plan already performed the recovery without noticing: W1b
regenerates `reason`'s receipt by re-running the identical pipeline, which by the determinism
above draws the same 4,982 rows. §5 and §4.1 contradicted each other.

### The recovery has a FOURTH precondition, it is unstated in revision 2, and W2a could not test it `[N1 fixed]`

Everything above establishes that **today's code** draws the same 4,982 rows every time. It does
not establish that **the code the `reason` run used** was today's code. **It was not necessarily,
and the change is exactly "sample vs prefix".**

```
git show c42203c --stat                                                          [V]
  c42203c  2026-09-02 19:32:52 -0400
  "fix(corpus): make a cap sample the source instead of taking its prefix"
  body: "DELIBERATE CORRECTNESS CHANGE -- IT CHANGES WHICH ROWS TRAIN, SO IT CHANGES RESULTS.
         Receipts written before this commit are NOT comparable with receipts written after it
         for any region whose cap actually binds ... `load_pairs` returned the moment `limit`
         pairs had been collected, iterating shards in sorted order ... Every 'balance cap' in
         this system meant 'take a prefix'."

src/cogsyndelta/regions/pretrain.py:199-207   (load_pairs docstring)              [V]
  "This used to `return` the moment `limit` pairs had been collected, which made every such
   decision 'take the first N rows in file order' instead ... `build_splits` shuffles AFTER
   the cap, so no downstream step could repair it."

src/cogsyndelta/regions/pretrain.py:576-582                                       [V]
  # "a cap whose method is unrecorded is not a cap" ... A RECEIPT THAT RECORDS ONLY THE
  # REALISED ROW COUNT CANNOT TELL A SAMPLE FROM A PREFIX, AND THOSE ARE DIFFERENT CORPORA.
  "cap_sampling": {"method": "reservoir (Algorithm R), ...", "seed": cfg.seed}
```

**Revision 2's three checks all pass under either hypothesis** [I]. Suppose the `reason` run
predates `c42203c`. Then the rows it actually consumed are the **first 4,982 in shard order**,
not the reservoir draw at seed 0. W2a re-runs today's `load_pairs`, gets a **different** 4,982
rows, and: check (i) passes, because the parquet on disk is unchanged; check (ii) passes, because
today's code is deterministic; check (iii) passes, because the run did use `seed = 0`. The ledger
then records 4,982 fingerprints as `reason`-burned **and marks as clean the 4,982 rows that were
actually burned.** Those rows enter the reserve, a region has memorised their answers, and DEC-22
— *"NO SINGLE REGION ALREADY HAS THE ANSWER MEMORISED"* — is violated **with a ledger entry
certifying the opposite**, which is worse than having no ledger and is the same shape as the
defect this programme cites as its worst find.

**Timing does not settle it** [I]. `c42203c` (19:32:52), `b9a082e` — which introduces the `4982`
cap in `scripts/csd-train-all.py` (19:41:03) — and `2864ecd`, the KICKOFF note recording
`reason 0.0039 → 0.0801` (19:42:11) are nine minutes apart [V], and `b9a082e`'s own subject is
*"wip(regions): classify and reason wiring, **preserved unvalidated**"*. A history that preserves
work after the fact does not date the run that produced it.

> ### DEC-42 — burn the UNION of every draw the ledger discovers, and let the gate fail if one is missing
>
> **The ambiguity is not resolvable from artefacts on disk, and it does not need to be.** Two
> candidate draws were anticipated, computable deterministically from config alone:
>
> - **draw R** — the reservoir draw: `reservoir_sample(_iter_pairs(...), 4982, sampling_rng(0, ...))`.
> - **draw P** — the prefix draw: the **first 4,982 pairs `_iter_pairs` yields in shard order**,
>   which is exactly what the same call returned before `c42203c`.
>
> **AMENDED — W2a's actual run (branch `feat/w2a-ledger-recovery`, `c6ea587`, approved) found a
> third.** A directory walk under `reason/aqua_rat-raw/` turned up an on-disk derived sample,
> `derived/sample-4982-seed0.parquet` (mtime `2026-09-02T23:05:45.957176Z`, dated before both
> `b9a082e` — the commit that introduces the `4982` cap — and `c42203c`), with a sibling
> `MANIFEST.json` declaring `numpy Generator(PCG64).permutation(n)[:N_SAMPLE]`: a full-corpus
> shuffle-then-take, **not** a prefix in file order, so it is neither R nor P. Its row count
> matches the cap exactly. **This is precisely the case DEC-42 exists for** — an artefact of
> ambiguous provenance, neither proven clean nor safely ignored — and the gate below is what
> caught it: it fails on a ledger omitting *any* discovered draw, so a human did not have to
> anticipate this one for it to be burned. Call it **draw D3**.
>
> **The ledger takes `fingerprints(R) ∪ fingerprints(P) ∪ fingerprints(D3)`** — and would take a
> fourth term if a fourth draw turned up. All three are marked `reason`-burned. No row of
> ambiguous provenance is ever certified clean.
>
> **The cost, measured rather than bounded, now that all three draws have actually been
> computed** `[V*, burned-aqua_rat.jsonl.manifest.json]`: `R` is 4,982 rows / 4,951 unique
> fingerprints; `P` is 4,982 / 4,946 unique, of which 4,626 are new against `R`
> (`|R ∩ P| = 320`, so `|R ∪ P| = 9,577` — *smaller* than the two-draw worst case of 9,964 this
> section used to carry, and close to the ≈ 9,709 expected value computed under Algorithm R's own
> retention behaviour `[I]`); `D3` is 4,982 / 4,930 unique, of which **4,369 are new against
> `R ∪ P`**. **The three-draw union is 13,946 of 97,467, and the clean pool is 83,521** — not "at
> least", not "at most": this is what the ledger's manifest actually recorded, of the draws
> discovered to date. The recovery survives at **1.71×** against DEC-49's 61,440-row training
> need (§5.1, was 1.77× under the two-draw union); the ambiguity does not.
>
> **W2a's fourth check, and it is the one that can fail.** Establish the code revision the
> `reason` run used — receipt or checkpoint mtime against `c42203c`'s commit time, or a trainer
> revision recorded in the checkpoint — **and if it cannot be established, record that it could
> not.** *Absence of evidence must not become the evidence.* Either way the union is written.
> **The gate:** W2a asserts `|ledger ∩ fingerprints(draw)| = 4,982` for **every discovered draw**
> — `R`, `P`, `D3`, and any further one a directory walk turns up — and **refuses to emit a
> ledger that omits any of them.** A fingerprint set omitting even one discovered draw is a
> **FAIL**, not a partial pass. **Verify it by making it fail:** hand the checker a ledger built
> from draw R alone and assert it refuses.

**W2a tests all four rather than assuming any**, because this is the largest number in this
section: (i) the recomputed `fingerprint_corpus` of the parquet on disk today matches;
(ii) two consecutive computations of **each** draw return identical fingerprints; (iii) the run
genuinely used `seed = 0` — which is `[I]` for `reason` specifically, since its receipt is the
deleted one, and is inferred from the three surviving text receipts all recording `seed 0` and
from `csd-train-all.py` passing `cfg.seed` uniformly; **(iv) the code revision, per DEC-42, with
the union written either way.** **If (i), (ii) or (iii) fails, the write-off stands and this
section reverts.**

### The pool, recomputed with the recovery

| source | rows | status |
|---|---|---|
| `deepmind/aqua_rat` | 97,467 | **13,946 BURNED** — the **union of the three draws W2a's run discovered** (DEC-42: R, P, and the on-disk derived sample D3), written to the ledger as `reason`-burned; **83,521 CLEAN** `[N1 fixed]` `[V*, burned-aqua_rat.jsonl.manifest.json]`. Revision 3.3 read `≤ 9,964` burned / `≥ 87,503` clean under a two-draw worst case; W2a's actual run found a third draw and measured the union exactly. Revision 2 read 4,982 burned / 92,485 clean, which was correct only under an unstated assumption about which sampling algorithm ran |
| `go_emotions` | 43,410 | BURNED (trained) |
| `PolyAI/banking77` | 13,083 | BURNED (10,003 trained; a 3,080-row remainder of a burned source that also fails B5 is not worth having) |
| `openai/gsm8k` | 7,473 | BURNED (trained uncapped) |
| `zalando-datasets/fashion_mnist` | 60,000 | clean |
| `deepmind/code_contests` | 13,328 | clean, **allocated `compose`** (§5.2) |
| `codeparrot/apps` | 10,000 | clean, **allocated `compose`** (§5.2) |

```
truly clean unallocated .......................... 165,065   (was 169,047 at the two-draw union,
                                                                DEC-42; 174,029 before DEC-42)
  of which fashion_mnist (28x28 greyscale, no text bin can use it)  60,000
  TEXT-USABLE clean:  aqua_rat recovered, union burned  83,521
                    + code_contests 13,328 + apps 10,000 = 23,328
                    - apps<->code_contests dedupe at cos >= 0.90  -1,784
  text-usable after dedupe ....................... 105,065   (was 109,047 at the two-draw
                                                                union; 114,029 before DEC-42)

reservation required, AFTER DEC-49 ...............  67,584   of which 61,440 is TRAINING data
  67,584 as a share of the declared pool .........   27.6%   (the published 23.0% was for 56,320)
  67,584 as a share of what remains ..............   40.9%   <- the real figure (was 40.0% at
                                                                the two-draw union; 33.3% at
                                                                56,320; 67.6% before the recovery)
  text-usable vs the 61,440-row training need ....   1.71x SURPLUS  (was 1.77x at the two-draw
                                                                union; 2.13x at 51,200, and a
                                                                2.38x SHORTFALL before recovery)
  ^ DEC-49's two new cross-faculty bins cost 1,024 eval + 10,240 train rows. The surplus is
    still a surplus; it is a thinner one, and section 9.2 says so.

THE THREE BALANCE NUMBERS, WHICH TRAVEL TOGETHER  [N8 fixed]
  MAX SOURCE SHARE (aqua_rat), unallocated pool ..   50.6%   <- BINDING. B1 hard line 0.50: FAIL
                                                                (was 51.8% at the two-draw union)
  MAX SOURCE SHARE (aqua_rat), text-usable pool ..   79.5%   <- the pool the reserve is drawn from
                                                                (was 80.2% at the two-draw union)
  fashion_mnist share of what remains ............   36.3%   <- NOT binding; revision 2 printed
                                                                only this one against B1 (was 35.5%)
N_eff of the remaining pool = 1/sum(p^2) .........    2.47   (B2 wants >= 3; still short;
                                                                was 2.44 at the two-draw union)
```

> **B1 is a MAX-SHARE rule, and after the recovery the maximum is not `fashion_mnist`'s**
> `[N8 fixed]`. Revision 2 printed `fashion_mnist 34.5% (B1 hard line 0.50, target 0.40)` and a
> reader scanning that block saw a B1 pass. The binding number is aqua_rat's: **50.6% of the
> unallocated pool and 79.5% of the text-usable pool** (was 51.8% / 80.2% at the two-draw union;
> W2a's actual run found a third draw, DEC-42), **against `CORPUS-CONTRACT.md:1155`'s
> *"`p < 0.50` is the hard line, `0.40` is the operating cap"* [V].** The recovery therefore
> **moves the pool from passing B1 to failing it.** The prose said so in two places; the numbers
> did not, and the numbers are what a ratifier scans.
>
> **What is done about it, in order.** (1) **B1 binds on the reserve, not on the pool** — the
> reserve does not have to consume the pool in proportion. Hold aqua_rat to the **0.40 operating
> cap of the reserve's source rows**: ≤ 27,033 of 67,584, leaving 40,551 to come from elsewhere
> (**restated for DEC-49's larger reserve**; it was ≤ 22,528 of 56,320 leaving 33,792).
> (2) **The non-aqua_rat text-usable pool is 21,544 rows**, so that gap can only be closed by
> composites over `fashion_mnist` panels and re-used text rows — i.e. **by W3r's renderer, and
> for the `visual × *` half by W7v.** So B1 compliance for the reserve is **W7v-dependent**, and
> that dependency is stated here rather than discovered in W3. (3) **If W7v slips, B1 is carried
> as a dated waiver naming exactly what the reserve cannot measure** — a corpus dominated by one
> source in one domain cannot support a claim about generalisation across sources — and that
> waiver is a W3 deliverable (DEC-38), not a footnote. (4) A **third genuine text source** is
> already named as one of the only true fetches (§5.5); it is what makes (3) unnecessary.
>
> **`N_eff` rose from 2.40 to 2.44 to 2.47 and that is not good news.** 2.40 → 2.44 was DEC-42's
> two-draw union; 2.44 → 2.47 is W2a's actual run finding the third draw, D3, and burning it too.
> It rose because the largest source shrank each time, not because the pool diversified. A
> diversity statistic that improves when you burn rows is being read wrong; it is printed
> because B2 asks for it, and it is still short of B2's ≥ 3 either way.

> **The corpus crisis is downgraded, not dissolved.** A 2.38× shortfall becomes a **2.13×**
> surplus after DEC-42's two-draw union burn, and **1.71×** once W2a's run measured the actual
> three-draw union, and §9.2 stops being the leading risk. **What does not
> change:** `N_eff` is 2.47 against B2's ≥ 3, so the reserve still needs either a third genuine
> text source or B2's dated waiver; and the recovered rows are aqua_rat — one source, one domain
> — so a reserve built mostly from them **fails B1, in the printed numbers and not only in the
> prose**. The rendered-composite generator is no longer *load-bearing for funding*; it is
> load-bearing for **B1 compliance**, for the `visual × *` pairs (W7v) and for source diversity,
> and its geometry prerequisite is now W3r.

A more permissive accounting keeps banking77's 3,080-row remainder. The conservative figure is
adopted; the conclusion is identical either way.

## 5.2 DEC-23 — done, and how it was verified `[A34 fixed]`

> **Mark `codeparrot/apps` (10,000) and `deepmind/code_contests` (13,328) as `compose` in the
> allocation ledger, now, before `language_code` expands to multi-language and before any P2.5c
> fetch.**

**This has landed.** Verified this session by re-reading the ledger table [V]:

```
docs/design/CORPUS-CONTRACT.md:964   | `deepmind/code_contests` | 13,328 | ~~code~~ -> **compose** | ATTRIBUTION |
docs/design/CORPUS-CONTRACT.md:966   | `codeparrot/apps`        | 10,000 | ~~code~~ -> **compose** | PERMISSIVE_OK |
docs/design/CORPUS-CONTRACT.md:970   "> **Reserved 2026-09-02 (DEC-23).** ... allocation happens now,
                                      before P2.3 trains `code`/`language_code`, per
                                      docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md §5.2."
```

So the threat pass's *"DEC-23 is unexecuted"* is refuted (§11 R1) — while its other half stands
untouched: **the ledger line is written, a COARSE train-time refusal landed with it, and the
per-row and derived-case guard is still unbuilt** `[N10g fixed]`. Revision 2 said flatly that
there is *"no train-time refusal"*; there is one. `0786a77` added
`RESERVED_FOR_COMPOSE = {"apps": …, "code_contests": …}`, `ReservedSourceError` and
`_refuse_reserved_shards()`, which raises at shard-resolution time for **any** region, in
`--dry-run` as well as a real run, and **before `pretrain_region` is ever imported**, with
`tests/test_reserved_corpus_guard.py` asserting the raise in three constructed cases [V]. It is
**directory-prefix granularity**, so DEC-38's derived case and per-row case remain uncovered —
which is the substantive point and is unchanged. The blanket phrasing was not, and pessimism
about a landed guard is how a landed guard gets built twice. A line in a document
is not a check in a data loader, and the distinction is the whole point of §5.6.

Revision 1 buried this action inside W2, the fifth row, whose gate also demanded the entire
reserve, the NSRS filter and a guard verified by making it fire. It is now **W2a** — minutes,
no dependencies, no GPU — alongside the aqua_rat recovery, and **W2b** carries the expensive
half.

They are the only remaining licence-clean source of paired *natural-language-problem ↔
implementation* material in the tree, which makes them the only source for the `memory ×
language_code` cross-faculty bin. Once a region trains on them they are gone forever.

Caveat carried forward: the two overlap **17.84% of problems at cos ≥ 0.90** — 1,784 of 10,000,
invisible to hashing because one prefixes every description [V\*]. Dedupe on the stripped
description *before* splitting, or the compose set is 17.8% contaminated on day one.


## 5.3 DEC-22 — the content contract: an admission test, not a provenance hope

The requirement is data where *"NO SINGLE REGION ALREADY HAS THE ANSWER MEMORISED. Otherwise it
learns to forward to whichever region already knows, which is dispatch, not integration"* [V\*].
"Held out" does not deliver that — a held-out FiQA question is still *within* `memory`'s
competence, and forwarding to `memory` solves it.

Make it measurable per row, using the frozen regions themselves:

> **NSRS ADMISSION FILTER.** For candidate item `x` with target `t` and candidate set `C`,
> compute each frozen region's standalone score `s_r(x, t, C)`. **Admit iff:**
> 1. **no single region suffices** — `max_r s_r < τ_lo`;
> 2. **some combination does** — `oracle_late_fusion(s) ≥ τ_hi`;
> 3. **no trivial baseline suffices** — BM25 (text) or raw-pixel logistic regression (vision)
>    scores `< τ_lo` on the same item.

> **Condition-(2) exemption, written into the filter rather than contradicted by it**
> `[A21 fixed]`. §5.5(c) builds the general / out-of-scope bin as *"condition (1) with no
> condition (2)"*, while W2's gate demanded *"every admitted row satisfies the NSRS filter"* —
> all three conditions. That was a direct contradiction between two sections of revision 1.
> **General-bin rows are admitted under conditions (1) and (3) only, are tagged
> `bin: general, nsrs: c1c3` in the manifest, and their correct answer is DEC-41's `NULL`
> candidate.** No other bin may use the exemption.

**Thresholds are calibrated, never inherited (DEC-36)** `[A1 fixed]`. `τ_lo` and `τ_hi` are
drawn from a **calibration subset disjoint at source-row granularity from everything graded**,
then frozen; the graded set is admitted under frozen thresholds; and the receipt reports **what
fraction of the graded set would have been rejected**. `s_r` is still recorded for every row —
it is computed anyway and it is useful diagnostics — but it is **never** reused as B1 or B2.
Revision 1's saving (*"precomputed rather than a separate experiment"*) is the exact thing that
destroyed the test; §2.7.0 is the argument.

**Thresholds are stated in chance-normalised units per bin:** `(s − chance)/(1 − chance)`, with
each bin's chance level recorded in the manifest. Raw `0.35` means a different difficulty in
every bin — chance is 1/512 on the in-mixture diagonal, 1/200 on the tiny-imagenet probe, 0.0130
for banking77, 0.0466 for go_emotions [V\*] — so the per-bin scores G2 compares would not
otherwise be on a common scale.

**And `τ_lo`'s starting value rests on a number nobody has measured** `[A23 fixed]`. Revision 1
justified condition (3) with *"a random-init encoder scored r@1 ≈ 0.40 on CodeSearchNet from
lexical overlap alone"*, tagged `[V*]`. That figure exists in the tree only as repeated prose —
`program/KICKOFF.md:120`, `docs/design/MODEL-MANIFESTS.md:1138`,
`docs/design/MODERN-TRAINING-STACK.md:509`, `src/cogsyndelta/regions/retrieve.py:22` [V] — while
the surviving `code` receipt records **0.2285** at `max_len 96` and `program/REMAINING.md:336`
records 0.0918 at `max_len 256`. **No artefact anywhere produces 0.40.** This is not tidiness:
if the true lexical floor for code-adjacent material is 0.40, condition (1) at `τ_lo = 0.35`
rejects essentially every row built from `apps`/`code_contests` — the reserve's main source —
and the reserve cannot be built from them at all. **W2c measures it and re-derives `τ_lo`
per bin, and W2b is blocked on that result.**

> **AMENDED IN REVISION 3.4 — W2c HAS RUN AND THIS PARAGRAPH IS THE PRE-MEASUREMENT ARGUMENT,
> KEPT DELIBERATELY** `[DEC-62]`. The paragraph above is left in its original tense because it is
> the reasoning that made the measurement worth spending, and a document that silently rewrites
> its own premises loses the record of why it looked. **What it asks is now answered:** the
> ≈0.40 floor is **retired as unsupported**, the 0.40 branch **did not fire**, and
> `apps`/`code_contests` therefore survive as the reserve's main source. The measured numbers,
> the seeds, and the trace of where 0.40 came from live at
> `docs/design/evidence/w2c-untrained-baselines-2026-09-03/` and are **cited, not restated here**,
> so there stays one copy of each number. **What is still open is the other half — `retrieve`'s
> chance-level `τ_lo` — and it is W2b's blocking clause, not this section's.** Read **DEC-62**
> before treating any sentence above as a live question.

Three properties worth stating:

- **Condition (1)** makes the reserve *definitionally* the data where dispatch fails.
- **Condition (2)** — the upper bound — is what stops the reserve from being noise. A row no
  combination can solve teaches nothing, and a corpus of them is indistinguishable from a broken
  loader.
- **Condition (3) is not optional and is the one most likely to be skipped.** Without it a
  "cross-faculty" composite can be solvable from a colour histogram and the whole integration
  claim is measuring an artifact.
- **NSRS is a DIFFICULTY filter, not an INTEGRITY filter** `[T1 fixed]`. A poisoned row — one
  carrying a rare trigger correlated with a chosen routing outcome — satisfies all three
  conditions trivially. NSRS was never designed to catch that and does not; §5.6's keyed split,
  generator identity and the S5 trigger-sensitivity verdict are what cover it. Saying so here
  stops NSRS being cited later as a control it is not.

**The filter is a guard, so it must be verified by making it fail.** Construct a row that only
`language_code` can solve and assert rejection; construct a general-bin row and assert it is
admitted under the `c1c3` exemption and **not** under the full filter. This is the pattern that
found four guards in this programme that were *"correct in reasoning and wrong in scope,
reporting success they could not have detected the absence of"* [V\*].

## 5.4 Size, composition, and the split key `[A16 fixed]`

Sizing is unchanged in method; bins are re-keyed to faculties, so `k` drops from 7 to 6.

| component | items | note |
|---|---|---|
| per-faculty eval — `language_code`, `memory`, `reasoning`, `visual`, `affect`(probe), `general` × 512 | 3,072 | 6 bins, not 7. `general`'s target is DEC-41's `NULL` candidate. **`episodic_store` gets NO per-faculty bin** — it has no standalone task, which is also why it is exempt from G2 (§2.7.2) `[DEC-49]` |
| **cross-faculty eval** — **6** reported bins × 512 | **3,072** | pairs below; **2 of the 6 are blocked until W7v**, and **2 are new under DEC-49** |
| **compose eval total** | **6,144** | **split 3,072 dev / 3,072 sealed** (§2.7.8). Was 5,120 / 2,560 / 2,560 |
| **interconnect train, ≥10× eval** | **61,440**, of which **≥60% cross-faculty** | the reframe made this *training* data, not eval padding. Was 51,200 |
| **RESERVATION TOTAL** | **67,584** | **was 56,320** |

> **DEC-49 RAISES THE RESERVE BY 11,264 ITEMS AND THAT IS A COST, NOT AN ADJUSTMENT.** Two new
> reported cross-faculty bins at the non-negotiable 512-per-bin reporting floor add 1,024 eval
> items, and the ≥10× training rule carries 10,240 more with them. **Against the recovered pool
> this is visible arithmetic:** 67,584 of 165,065 clean unallocated is **40.9%** (was 40.0% at the
> two-draw union, 33.3% before DEC-49), and the training need alone — 61,440 rows — leaves a
> surplus of **1.71×** against the 105,065 text-usable rows, down from **1.77×** at the two-draw
> union and **2.13×** before it. **§9.2's funding risk is tightened, not closed**, and the
> alternative was worse: holding the cross-faculty eval at 2,048 across six bins would put 341
> items in each, below the reporting floor the whole sizing rests on. **The lever §2.7.8 named for
> exactly this case is the one that was pulled**, in advance and in writing, rather than as a
> post-hoc re-slice.

The composed-vs-single comparison is **paired**, so McNemar on discordant pairs is the test and
512 is a reporting floor per bin, not the resolution of the comparison [V\*].

### The split is at SOURCE-ROW granularity, and so is the bootstrap

Revision 1 split 51,200 train / 5,120 eval at *item* granularity and resampled the bootstrap
*over items*. Composites are built by choosing `n` panels from a pool of text items, so ~30,720
cross-faculty training composites drawn from a source pool of 21,544–105,065 rows **reuse each
row many times**. Three consequences, all of which inflate the result:

1. **Train/eval leakage.** An eval composite will contain panels the model saw inside training
   composites unless the split is made at source-row granularity — which revision 1 never
   stated. This is A6 one level up: the same mistake of hashing the derived object.
2. **The bootstrap is over the wrong unit.** Items built from a shared row pool are not
   independent; resampling items understates variance and inflates the significance of `I` and
   of every McNemar test.
3. **The reserve fails this document's own balance rules.** §7.1 instructs `CORPUS-CONTRACT.md`
   that for constructed corpora *"the generator is the provenance group, and B1/B2 apply to
   it"*. A reserve supplied mostly by one renderer over two sources has generator share ≈ 1.0
   against B1's hard line of **0.50** (`CORPUS-CONTRACT.md:1155`: *"`p < 0.50` is the hard
   line. **0.40** is the operating cap"* [V]) and `N_eff` ≈ 1–2 against B2's ≥ 3. Revision 1
   wrote the rule and never applied it to the corpus it was proposing.

> **DEC-38 — every split, guard and resample keys on the SOURCE ROW.**
>
> - **No source row may appear on both sides of the train/eval split in any derived form.** The
>   split key is `pair_fingerprint(source_row)`, recorded per item in the manifest.
> - **Block-bootstrap by source row**, stated in the receipt. Per-pair CIs in §2.7.3 are block
>   CIs.
> - **B1/B2 are run on the reserve's own generators** in W3, and the result is either diversified
>   (a second renderer, a second join source) or carried as a **dated waiver naming exactly what
>   the reserve therefore cannot measure**. B2 already provides for the waiver; using it is
>   honest, ignoring the rule is not.

**The six reported cross-faculty bins**, re-keyed to the new taxonomy (`compress × retrieve` was
one of the contract's four planned pairs and **the merge makes it intra-region**, so it is
replaced; the last two are added by DEC-49):

| pair | item shape | status | what no region can do alone |
|---|---|---|---|
| `memory × language_code` | given a natural-language issue, retrieve the API description, then find the function implementing it | **live** | the old `retrieve × code` |
| `memory × reasoning` | multi-hop: retrieve two facts, then combine them | **live** | retrieval alone returns one fact; reasoning alone has no facts |
| `visual × language_code` | locate an element in a frame, then name or act on it | **BLOCKED until W7v** `[A4 fixed]` | the old `vl × classify`. Unconstructible against 64×64 single-object tiles, and the encoder cannot ingest a larger frame at all |
| `visual × memory` | given a description, find the *frame* that shows it | **BLOCKED until W7v** | genuinely new, and the one that exercises long **visual** context — the operator's stated primary long-context channel |
| `episodic_store × memory` | a **two-turn episode**: turn 1 retrieves and commits a fact; turn 2 asks a question answerable only from that fact, which is **absent from turn 2's own context window** | **BLOCKED until E1** `[DEC-49]` | retrieval alone cannot see turn 1; the store alone holds a latent nobody indexed. **This is the pair that makes "recall-dependent" a measurement** |
| `episodic_store × visual` | a two-turn episode whose turn-1 premise arrives as a **rendered panel** and whose turn-2 premise arrives as text; the answer needs both and is stated on neither | **BLOCKED until E1 + W7v** `[DEC-49]` | the deferred-premise case across modalities — it is also the one that tests whether the store's contents are modality-general or quietly text-shaped |

### Every ablation pair now has an item shape and a sealed-item count `[N2 fixed]`

**A17's defect returned one level down and this is where it is closed — and revision 3.3 closes
it a second time, by construction rather than by exclusion.** Revision 2 demoted `episodic_store`
so *"≥6 of 10 pairs"* would stop secretly meaning *"all 6 real pairs"*, and replaced it with
DEC-37's *"≥4 of the 6 pairs"* over four declared shapes. That worked, and it worked by removing a
participant. **DEC-49 puts the participant back, so the same defect has to be closed the harder
way: by declaring the shapes.** The rule that governs the denominator is now mechanical and is
stated so it can be run: **no ablation pair may appear in the denominator without a row in the
mapping below and a non-zero sealed count.**

**The participant count is five and no revision of this document may quietly change it**
`[S31-1 fixed]`: DEC-49 puts `episodic_store` in, DEC-48 keeps `auditory` out, and both are
rulings. Five participants generate **ten** pairs. Under revision 3.2's four, two pairs —
`language_code × reasoning` and `visual × reasoning` — had **no items that require both regions**;
for such a pair `Δ_A ≈ Δ_B ≈ 0` on the cross-faculty bin, DEC-37's conjunction fails **by
construction**, the pair lands in the named `REDUNDANT` quadrant, and that is a FAIL. **X2 and X6
closed those two. X7 and X8 close the four the store adds.**

**Eight item shapes, and the mapping to the ten ablation pairs.** Four shapes are three-way or
two-turn and are declared as such; **the cross-faculty eval budget rises from 2,048 to 3,072**
because two reported bins are added at the 512-per-bin floor, and the shapes are allocated inside
the six reported bins.

| shape | bin it is reported in | eval items | construction (§5.5) | ablation pairs it makes load-bearing |
|---|---|---:|---|---|
| **X1** issue → API description → function | `memory × language_code` | 256 | §5.5(a) retrieval half | {memory, language_code} |
| **X2** **executable three-way join** — retrieve the matching problem (**memory**), then rank candidate solutions by whether they pass the tests (**language_code** + **reasoning**), distractors are other problems' correct solutions | `memory × language_code` | 256 | §5.5(a), label = sandboxed exit code | {memory, language_code}, **{language_code, reasoning}**, {memory, reasoning} |
| **X3** **multi-hop over rationales** — retrieve two aqua_rat rationales from a pool, then combine the facts they state into an answer **neither states alone** | `memory × reasoning` | 512 | §5.5(a′), **new** `[N9 fixed]` | {memory, reasoning} |
| **X4** composite locate-then-name | `visual × language_code` | 512 | §5.5(b), needs **W3r** + W7v | {visual, language_code} |
| **X5** description → frame | `visual × memory` | 256 | §5.5(b), needs **W3r** + W7v | {visual, memory} |
| **X6** **two-premise numeric composite** — two rendered panels each state one premise; the answer requires combining them and is stated on neither panel | `visual × memory` | 256 | §5.5(b′), **new** `[N2 fixed]` | **{visual, reasoning}** |
| **X7** **RECALL-DEPENDENT EPISODE, text.** A two-turn episode. **Turn 1** presents a document and asks a question whose answer commits a fact to the store (an API signature, a retrieved passage's key claim). **Turn 2** presents a *different* document and asks a question answerable **only** from turn 1's fact — not in turn 2's context window and not recoverable from turn 2's own corpus. **The negative control is the identical turn 2 with turn 1 replaced by an unrelated episode, and it must FAIL** | `episodic_store × memory` | 512 | §5.5(e), **new** `[DEC-49]` | {episodic_store, memory}, **{episodic_store, language_code}** |
| **X8** **RECALL-DEPENDENT EPISODE, cross-modal.** The same two-turn shape with turn 1's premise delivered as a **rendered 128×128 panel** (W3r's geometry) and turn 2's as text; the answer combines them and is stated on neither. **Text-only variant X8′ is pre-committed here, not invented later**, for the W7v-slip branch below | `episodic_store × visual` | 512 | §5.5(e′), **new** `[DEC-49]` | {episodic_store, visual}, **{episodic_store, reasoning}** |

> **X7 AND X8 REQUIRE AN EVAL HARNESS THIS PROGRAMME DOES NOT HAVE, AND THAT IS A DELIVERABLE, NOT
> A FOOTNOTE.** Every other shape here is a single forward pass. A recall-dependent item is an
> **episode**: turn 1 must run, its write must commit through the store's `learn()` lifecycle, and
> turn 2 must run against the resulting partition. Revision 2's own objection to the store was that
> *"a single-item eval forward pass has nothing to read"* — **that objection is correct and is not
> answered by declaring a shape**; it is answered by building the harness. **W3 gains an episode
> harness as a named deliverable, with E1 as its prerequisite** (§4.1).
>
> **Until the harness exists, the four store pairs are declared, budgeted and constructible — and
> NOT SCORED.** If W3 reaches its gate without it, the honest report is
> `episodic pairs: NOT MEASURED`, G3′ falls back to the pre-committed six-pair arithmetic below,
> and the receipt says which happened. **What is forbidden is the third option**: ten pairs named
> in the design and six in the denominator, with nobody told.

**Sealed items per ablation pair** [I], from §2.7.8's split (compose eval 6,144, sealed half
3,072, of which 1,536 are cross-faculty, i.e. 256 per reported bin):

| ablation pair | shapes | sealed items | status |
|---|---|---:|---|
| {memory, language_code} | X1 + X2 | **256** | live |
| {memory, reasoning} | X2 + X3 | **384** | live |
| {language_code, reasoning} | X2 | **128** | live, **thin** |
| {visual, language_code} | X4 | **256** | blocked until W3r + W7v |
| {visual, memory} | X5 | **128** | blocked, **thin** |
| {visual, reasoning} | X6 | **128** | blocked, **thin** |
| {episodic_store, memory} | X7 | **256** | blocked until **E1 + the episode harness** |
| {episodic_store, language_code} | X7 | **256** | blocked until **E1 + the episode harness** |
| {episodic_store, visual} | X8 | **256** | blocked until **E1 + W7v + the harness** |
| {episodic_store, reasoning} | X8 | **256** | blocked until **E1 + W7v + the harness**; **X8′ carries it under the slip** |

**The two cross-faculty totals reconcile and BOTH are printed, because S21 asked for exactly that
and the same ambiguity would otherwise recur one revision later** `[S31-21 fixed]`: the per-pair
column sums to **2,304** against a cross-faculty sealed total of **1,536**. They differ because
**shapes overlap pairs** — X2 is counted in three pairs, X7 and X8 in two each. The **items** are
1,536; the **pair-memberships** are 2,304. A receipt printing one without the other has published a
number nobody can check.

**G3′'s "≥7 of 10" means what it says**: ten pairs, eight declared shapes, ten mapped pairs, no
zero counts, null rate **0.1719**. **The three thin pairs are named rather than smoothed over**,
and they get a gate:
**W3 reports each pair's block-bootstrap CI half-width on the DEV half before the seal is
created.** If any live pair's dev half-width exceeds **±10 pp**, §2.7.8's stated lever fires —
raise the cross-faculty share of the compose eval — and the reallocation is recorded. **The
decision is made on the dev half, in advance; it is never a post-hoc re-slicing of the sealed
half.**

> **If W7v slips, the three `visual × *` bins are dropped from v1 and every count that depends on
> them is restated in the same breath** — restated again in revision 3.3, because DEC-49 changed
> every one of the numbers `[DEC-49]`: cross-faculty eval falls from 3,072 to **2,048**, the
> compose eval from 6,144 to **5,120**, `R` from 5 to **4** (`language_code`, `memory`,
> `reasoning`, `episodic_store`), the ablation pairs from 10 to **6**, and G3′'s criterion from
> "≥7 of 10" to "**≥ 5 of 6**" with a null rate of **0.109** — the smallest `k` under §2.7.3's
> 0.20 null-rate ceiling, which is why it is not the "≥ 4 of 6" (0.344) revision 3.2 used at this
> pair count. That is a materially
> harsher pass/fail, and it must be stated as such rather than absorbed.
>
> **The six surviving pairs and the shape that carries each, so the branch is achievable — which
> under revision 2 it was not** `[N2 fixed]`: {memory, language_code} = X1+X2, {memory, reasoning}
> = X2+X3, **{language_code, reasoning} = X2** (the executable three-way join, with a subprocess
> exit code for a label — this is the pair that had no shape and whose true null rate was
> therefore **0**, not 0.125), {episodic_store, memory} = X7, {episodic_store, language_code} = X7,
> and **{episodic_store, reasoning} = X8′**, the text-only deferred-premise variant pre-committed
> in the shape table above precisely so this branch does not have to invent one on the day.
>
> **Under the slip, X8′ replaces X8 at the same 512/256 budget** and the `episodic_store × visual`
> bin's items move to it, so all six surviving pairs keep **≥ 256 sealed items** and the branch is
> powered as well as achievable.
>
> **Pre-committed with the branch, so the slip does not also make the criterion underpowered:**
> under the slip, **X2 takes the whole of the `memory × language_code` bin (512 eval / 256
> sealed) and X1 moves to train-only**, giving {language_code, reasoning} **≥256 sealed items**
> like the other five.
>
> **The general bin is also restated under this branch** `[N9 fixed]`, which revision 2's
> five-consequence list omitted: §5.5(c)'s construction is a `visual` frame paired with an
> unrelated `memory` query, so it is W7v-blocked too, and it carries DEC-41's `NULL` gate that
> A21's fix rests on. §5.5(c′) gives it a **text-only variant** so the `NULL` gate survives the
> slip.
>
> **The recommendation is still to run W7v** — it is folded into a retrain W1's result already
> made mandatory, so the marginal cost is the resolution change (and W3r, which is CPU-only and
> has no dependencies), and dropping the visual pairs guts the long-context-is-visual argument
> that §6.1 rests the whole scheduler case on.

**A second slip branch exists now and is pre-committed on the same terms, because DEC-49 created
a second thing that can slip** `[DEC-49]`. **If E1 slips, or the episode harness does**, the four
`episodic_store × *` bins are dropped: cross-faculty eval falls to 2,048, the compose eval to
5,120, `R` to **4** (`language_code`, `memory`, `reasoning`, `visual`), the pairs to **6**, and the
criterion to **≥ 5 of 6** at null **0.109** (§2.7.3's ceiling) — the same arithmetic as the W7v
branch, arrived at by dropping the other participant. **If BOTH slip**, `R = 3`, the pairs are **3**, and the criterion
is **all 3 of 3** at null **0.125**, which is revision 3.2's own worst case unchanged. **All three
branches are written here, before the seal exists**, so no branch is ever a post-hoc re-slice —
and the store's four pairs are the ones most likely to slip, because they are the only ones that
need a component this programme has never built.

`reasoning × numeric` is deferred with the `numeric` placeholder.

## 5.5 Sourcing — three constructions, ordered by provenance cleanliness

*"No region corpus contains a single such item. They cannot: a region corpus is defined by its
objective, and a cross-region item has two"* [V\*]. So cross-faculty items must be
**constructed**. The cleanest first, the model-in-the-loop last.

**(a) Executable joins over `apps` + `code_contests` — 23,328 rows already held, and the reason
DEC-23 was urgent.** Each row carries a natural-language problem statement, reference solutions
**and test cases**, so the ground truth is *executed, not annotated*:

> given the problem statement, retrieve the matching problem from a 23k pool (**memory**), then
> rank candidate solutions by whether they pass the tests (**language_code** + **reasoning**),
> where the distractors are other problems' correct solutions.

**This item is THREE-WAY, and §5.4's mapping says so rather than leaving it to be inferred**
`[N2 fixed]`. It is shape **X2**, and it makes three ablation pairs load-bearing:
{memory, language_code} (the retrieval step), **{language_code, reasoning}** (deriving the
constraint the statement implies is `reasoning`; reading whether a candidate program satisfies it
is `language_code`; neither alone ranks the candidates) and {memory, reasoning}. It is the shape
that makes G3′'s W7v-slip branch achievable at all.

**No model is in the provenance chain — the label is a subprocess exit code.** And the
apps↔code_contests 17.84% cos≥0.90 overlap stops being purely a contamination problem: dedupe it
out of the *eval* split, and keep the near-duplicates as **hard train negatives** [I].

> **THE SANDBOX, stated because revision 1 celebrated this generator's provenance and never said
> where it runs** `[T1 fixed]`. "The label is a subprocess exit code" means **building the
> reserve executes tens of thousands of third-party programs**. It is the single largest
> untrusted-code workload in the programme. It runs on a **RuntimeClass the GPU hosts never
> share** — gVisor or Kata, **no network, no NFS mount, non-root, per-candidate wall-clock and
> memory caps, read-only rootfs, a scratch tmpfs discarded per candidate**. The fleet already
> applies this rule to agent sandboxes; this is the same workload class and inherits it.
>
> **The control is decorative unless one property holds: the generator must have NO
> plain-`subprocess` fallback.** If it can degrade to running locally when the sandbox is
> unavailable, it will, on the day the sandbox is unavailable. *Fails closed:* no sandbox ⇒ the
> generator raises and the reserve is not built. **Verify by making it fail:** run the generator
> with the RuntimeClass removed and assert it refuses rather than falling back.

**(a′) Multi-hop over rationales — the construction `memory × reasoning` did not have, and the
one that gives the recovered aqua_rat rows a cross-faculty consumer that is not the blocked
renderer** `[N9 fixed]`. §5.4 lists `memory × reasoning` as **live** — one of only two live pairs
while the `visual × *` pairs are blocked — and revision 2 named **no source, no generator and
therefore no generator hash for DEC-39's ledger row**. It is shape **X3**:

> from the 83,521 recovered aqua_rat rows (§5.1), each carrying a question and a **rationale**,
> take two rationales that each state one intermediate fact, put both in a retrieval pool of the
> remaining rationales, and ask a question whose answer requires **both** facts and is stated in
> **neither** rationale alone.

**Region-wise:** retrieval alone returns one rationale and cannot combine; reasoning alone has no
facts to combine. **Provenance:** no model in the chain — the two-fact join is mechanical over
text already held, and the distractor pool is the rest of the source. **Ledger:** the generator is
named `multihop_rationale_join`, with its source SHA-256 and config hash, per DEC-39.
**It is text-only and therefore W7v-independent**, which is why it, and not the renderer, is what
keeps the compose eval alive if W7v slips. **Its admission is not assumed:** items pass W2b's NSRS
filter like every other, and if `reasoning` alone answers them the filter rejects them and the
failure is informative — it means the second hop is not doing work.

**(b) Rendered composite-visual items — licence-free, and the source of `visual × *`.** Take `n`
text items from *already-cleared* sources, render them into a single frame with a deterministic
layout engine, and ask locate-then-read-then-act:

> "In this frame, find the panel whose text describes X, and return the code in the panel to its
> right."

Ground truth is derived **mechanically from the layout the renderer chose** — exact, free, and
generable at any volume. It is genuinely cross-faculty (visual × language × the source item's
faculty). Licence: a rendering of already-cleared text inherits that text's terms; **no new
source is acquired.** It serves three stated goals at once — long context *primarily visual*,
the composite-image argument (*a screenshot IS a collage; training on isolated 64×64 tiles is
the artificial one*), and the composite-image training item [V\*]. It costs nothing but CPU on
the 1080 Ti host. **Its geometry and its minimal reference implementation are W3r's deliverable;
W3 produces the volume against W3r's fixed frame** `[N5 fixed]` — revision 2 made the renderer
W3's primary deliverable while W7v's gate consumed a rendered composite and W3 was
`blocked_by: W7v`, which is a cycle in which neither row can start.

**(b′) Two-premise numeric composites — the construction `visual × reasoning` did not have**
`[N2 fixed]`. Same renderer, same licence position, different ground truth:

> render two panels that each state one premise of a numeric problem drawn from the recovered
> aqua_rat rows, plus distractor panels; ask for the quantity that follows from **both** premises
> and is printed on **neither** panel.

`visual` must read two specific panels and `reasoning` must combine them; neither is sufficient,
and a model that OCRs one panel gets a distractor. It is shape **X6**, and like every `visual × *`
shape it needs **W3r** for the frame and **W7v** for an encoder that can ingest it.

**Two corrections to revision 1's framing of this generator.** First, it is **no longer
load-bearing for funding the reserve** — the aqua_rat recovery (§5.1) turns a 2.38× shortfall
into a **1.71×** surplus at DEC-49's 61,440-row training need (1.77× at the two-draw union,
2.13× at the old 51,200, 2.23× before DEC-42's union burn), so *"this single generator can supply the entire training
reserve"* is no longer a plan the programme depends on, and leaning on it would drive generator
share to ≈1.0 against B1 (§5.4). Second, **it produces nothing `visual` can read until W7v**:
a four-panel composite downsampled to 64×64 is not legible to any encoder, and the encoder
raises a shape error on anything larger.

**(c) The general / out-of-scope bin, which today has no source at all** [V\*]. **It does not
need a corpus; it needs a construction.** An out-of-scope item is one where the correct
behaviour is *low confidence from every region* — so build it as **shuffled cross-modal
mismatches drawn from the reserve itself** (a `visual` frame paired with an unrelated `memory`
query). It is admitted under the **`c1c3` exemption** (§5.3), and its correct answer is DEC-41's
**`NULL` candidate**, gated on `NULL` recall > 0.50 in-bin and `NULL` false-positive < 0.05
out-of-bin. **Zero new sourcing**, and it directly trains the out-of-label-space rejection that
both classify regions name as their top gap [V].

**(c′) The general bin's W7v-INDEPENDENT variant, because (c) as written is blocked too**
`[N9 fixed]`. §5.5(c) builds the bin from *"a `visual` frame paired with an unrelated `memory`
query"*, so it inherits every W7v and W3r blocker the renderer has — and §5.4's W7v-slip
restatement did not list it, even though it carries DEC-41's `NULL` gate (`NULL` recall > 0.50
in-bin, FPR < 0.05 out-of-bin) that A21's fix depends on. **The text-only variant:**

> pair an `apps`/`code_contests` problem statement with an **unrelated** aqua_rat question and ask
> the question that belongs to neither — every region scores low, the correct answer is DEC-41's
> `NULL` candidate.

Same `c1c3` exemption, same `NULL` gate, same zero new sourcing, and **no dependency on W3r or
W7v**. Under the slip branch the general bin is built entirely from (c′) and the `NULL` gate
survives; with W7v the two variants are mixed and the bin gets cross-modal mismatches as well.

Note a taxonomy dividend: retiring `residual_mlp` — the fallback region that *"has never seen
real text"* [V\*] — converts the general bin from a **training** requirement into an
**eval-only** requirement (*does the mind decline rather than confabulate?*). One retirement
closes the largest named corpus gap in the contract by a taxonomy decision instead of by finding
a corpus. It does **not** remove the need for an output the eval can read, which is why DEC-41
names one.

**(d) Local-model synthesis — LAST RESORT**, only after (a)–(c) are exhausted. **The generating
model is part of the provenance chain** [V\*] and must be named in the manifest with its
revision and licence, generated with permissively-licensed **local** weights and never a hosted
API, and its output excluded from every eval split.

**(e) RECALL-DEPENDENT EPISODES — a re-staging, not a new corpus** `[DEC-49]`. X7's construction
consumes **no new source rows beyond the two the reserve already draws**: take an admitted §5.5(a)
retrieval-half item and an admitted §5.5(a′) multi-hop item **whose source rows are disjoint under
DEC-38's fingerprint**, and stage them as **turn 1** and **turn 2** of one episode, with turn 2's
question rewritten so its answer requires the fact turn 1 established. **The label is the same
label the single-turn item already carried**, which is what makes this cheap and what keeps the
provenance chain intact: an episode's chain is the union of its turns' chains, and both turns'
`pair_fingerprint`s go to the ledger.

**Three properties this construction must have, each of which can fail and each of which is
checked in W3 rather than assumed:**

1. **The fact must be UNRECOVERABLE from turn 2 alone.** Verified by the **negative control**: run
   turn 2 with turn 1 replaced by an unrelated episode and require the item to be **failed**. An
   item whose turn 2 is answerable without turn 1 is not recall-dependent, it is a single-turn item
   in two parts, and it is **rejected at construction**, not discovered at W6.
2. **The fact must not be recoverable from `memory`'s own corpus either**, or the pair measures
   retrieval and calls it recall. Checked with the same `s_r` machinery §5.3 already runs: an
   episode whose turn-2 question scores above `τ_lo` for `memory` **without** turn 1 is rejected.
3. **Turn 1 and turn 2 must not share a source row**, under DEC-38 — otherwise the split key
   cannot place the episode on one side of the train/eval boundary and the episode leaks against
   itself.

**(e′) The cross-modal variant, and its pre-committed text-only fallback.** X8 stages turn 1 as a
**rendered W3r panel** carrying the premise and turn 2 as text. It inherits every W3r and W7v
blocker the renderer has, which is why **X8′** is declared with it and not after it: the same
two-turn staging with turn 1's premise delivered as text. Under §5.4's slip branches X8′ takes
X8's whole budget. **Zero new sourcing in either variant** — which is the reason DEC-49's reserve
increase is 11,264 *items* and not a new fetch (§5.4).

**Genuinely new fetches, the only true ones:** a permissively-licensed general-text corpus
(which also unblocks P6′), a **third text source** to lift `N_eff` off 2.47 **and aqua_rat's binding max-share off 79.5%** (§5.1) `[N8 fixed]`, and
and — **for the production phase, not for v1** — **audio** for `auditory`
(DEC-43 as amended by **DEC-48**) `[OP: csd-multimodal-io-intent.md]`. The licence audit is
**done** and committed at `docs/design/AUDIO-CORPUS-AUDIT.md`, the fetch is row **A0f**, and the
two cross-faculty item shapes it makes constructible — `auditory × language_code` and
`auditory × visual` — are row **A3**; all of them are **deferred** behind W6 and an operator go,
so **no audio row is a v1 sourcing dependency and the ten ablation pairs need none of it**.
Audio remains the *best-provenanced* genuinely-new fetch on this list — **seven provenance
groups, of which five carry no NC term** `[S31-13 fixed]`, and no BLOCKING source in the
recommended mix. **The balance claim is `[I]` and cap-dependent, not a finding** `[S31-2 fixed]`:
`N_eff ≥ 3` is reachable **only at a per-group cap of order a few hundred hours** and fails at a
1,000-hour cap, because three of the clean groups (MUSAN ~109 h, purpose-recorded ~229 h,
Freesound ~145 h) are two orders of magnitude smaller than the other three. Whoever picks the cap
picks the verdict, and A0f's gate (iii) is where that is decided by arithmetic rather than by
sentence.

## 5.6 The ledger: what is reserved, at what granularity, and how the guard is made to fire

The programme already marks P2.5d as **BLOCKS P2.3** and records the reason correctly:
*ALLOCATION IS IRREVERSIBLE* [V].

### DEC-38 — reserve at source-row fingerprint granularity `[A6 fixed]`

Revision 1 specified `data/reserve/RESERVED.jsonl`, *"one row per reserved item with a **content
hash**"*. **That guard is correct for a *fetched* reserve and wrong for a *constructed* one**,
and §5.5 makes the reserve constructed. A rendered composite's content hash bears no relation to
the hashes of the source rows it was built from, and neither does a join's.

**The failure, concretely.** Reserve item `R` is a rendered frame whose panel 3 is CodeContests
problem #4,117. Nothing stops `language_code` from later training on problem #4,117: its hash is
not in `RESERVED.jsonl` — only `R`'s is. The guard passes, the training run starts, and the
interconnect is then evaluated on a frame whose answer a region has memorised. **That is exactly
the condition DEC-22 exists to prevent, defeated by the guard that was supposed to enforce it** —
and it is the same shape as the defect this programme cites as its worst find (*"dedup and the
check used the same hash, so overlap was empty by construction, always"*). Revision 1 diagnosed
that pattern in four places and reproduced it here.

1. **Every constructed item's manifest carries the `pair_fingerprint` of *every source row it
   consumed*, plus the generator id and revision.** `pair_fingerprint` already exists
   (`src/cogsyndelta/eval/metrics.py:130-140` [V]) and `_pair_key` already delegates to it for
   exactly this reason (`regions/pretrain.py:234-250` [V]: *"Two independently-written 'unordered
   pair key' implementations that drift apart is exactly how a guard ends up checking something
   other than what it claims to"*). Reuse it; do not write a second one.
2. **`RESERVED.jsonl` holds the UNION OF SOURCE FINGERPRINTS**, not composite hashes. The
   train-time guard checks the training pool against that union, **in the data loader, at train
   time, not at review time** — the loader raises rather than filtering, so it fails closed.
3. **Verify by making it fire, on the derived case.** Build a composite from a row already in
   `language_code`'s realised training set and assert the training run **refuses to start**. A
   guard tested only on a directly-reserved row does not exercise the path that actually leaks.
   Both refusals are W2b gates.

### DEC-39 — keyed split assignment `[T1 fixed]`

The integrity gates in §5 and §2.6 are all computed by the same pipeline that generates the
data, and three of the four reserve generators are agent- or code-driven. The `<5`-point
train/held-out overfit gate catches memorisation; it is **blind to anything present in both
splits**, where the gap is zero. `assert_no_contamination` (`eval/metrics.py:229` [V]) answers a
different question — train/eval *overlap* — and is equally blind, because poisoned rows are not
duplicates of each other, they share a trigger.

> ```
> split = HMAC(k_split, pair_fingerprint(source_row)) mod N
> ```
>
> `k_split` is held **outside the repo and outside every path an agent can write**. **A
> generator that cannot choose which rows land in eval cannot defeat any downstream gate**,
> including the overfit gate. *Fails closed:* **no key ⇒ the loader refuses to build a split at
> all**, rather than falling back to sequential or seeded assignment. This is the cheapest
> control in this document and revision 1 did not have it.

### Generator identity, and what the ledger row carries

Each `RESERVED.jsonl` row carries: the source-row fingerprints it consumed, the **generator
name, the SHA-256 of the generator source, and the config hash**. These are **recomputed on
load**, and a mismatch **aborts the run** — it does not skip the row. *Fails closed*, provided
the abort lives in the data loader and not in a pre-flight script someone can forget to run.
Residual risk, stated: a poisoner with write access to generator *source* defeats content
hashing, because the hash then faithfully certifies poisoned content. Nothing closes that short
of human review of generator diffs, which is what OD-1's protected-path control buys.

### Sources are pinned by revision, not by name

The fleet has measured that mirror metadata lies (ten licence mismatches; all 75
sentence-transformers datasets declare none). The same channel carries **content**, and
`apps`/`code_contests` arrive through it. **The manifest pins the source revision SHA and the
fetcher refuses an unpinned fetch.** *Fails closed* because the refusal is in the fetcher.

### On storage, so it is not mis-escalated

The tiered copy of corpora and shards on `gpu5080:/bulk` is **re-acquirable**: everything there
came from HuggingFace or another trusted source. **A gap in the cold copy is a re-fetch, not a
recovery incident and not an operator decision** — sample-verify cheaply and re-fetch what is
missing `[OP: csd-thesis-and-success-criteria]`. What is *not* re-acquirable is the ledger: a
burned row is burned by the programme's own rule, and the aqua_rat case is only recoverable
because the *sampling* is deterministic (§5.1), not because the data was backed up.

## 5.7 DEC-31 — the reserve's licence, and the composed model's

The operator's governing rule, verbatim: *"the overall model once fully trained is going to have
whatever the strictest license is between its data sets and sub models and the model itself —
that will take precedence and set the tone for all the other licenses throughout the entirety of
it"* `[OP: csd-release-licence-decision]`.

| object | term | why |
|---|---|---|
| `language_code`, `reasoning` | MIT | no restrictive input |
| `retrieve` (parent) | **NC** | GooAQ's README forbids commercial use and is 77.8% of its corpus; the operator accepts the restrictive reading rather than dropping the source |
| `compress` (parent) | CC BY-SA if the SA pairs are used | subsumed on merge |
| **`memory`** | **NC(-SA)** | **a merge inherits the most restrictive licence of its parts.** This is a real, named cost of DEC-02, and the taxonomy decision states it rather than discovering it at release |
| `visual` | follows its replacement corpus (OD-4); **unreleasable as trained** | tiny-imagenet |
| the **reserve** | the strictest term among the sources it renders or joins; a rendering of cleared text **inherits that text's terms** | §5.5(b) |
| **the composed mind** | **NC**, today | strictest input wins, and `memory` is in it |

Two consequences worth stating so they are not rediscovered. **Per-region tiers only matter for
regions shipped standalone** — the composed release is one licence, and simplicity is the
operator's stated preference (*"the easiest deconfliction for licensing within my constraints"*).
And **"may I train on it" stays separate from "may I redistribute it"**: this decision covers
training and weight release; redistributing GooAQ-derived pairs is still governed by the dataset
terms, so dataset publication stays private-HF-only.

### DEC-45 — the audio tiers, and why NC costs nothing now that `memory` has already spent it

New in revision 3.1, from `docs/design/AUDIO-CORPUS-AUDIT.md`. The same strictest-input rule,
applied to two regions that did not exist as requirements when DEC-31 was written. **In revision
3.2 both regions are DECLARED SEAMS deferred to a production phase (DEC-48), so this table is
groundwork rather than a v1 licence decision** — it is kept in full and kept correct, because
recomputing it after a fetch is how a tier gets decided by whatever was already downloaded.

| object | term | why |
|---|---|---|
| **`auditory`, clean-tier build** | **CC BY 4.0** | LibriVox + VoxPopuli + AMI + Freesound(filtered to CC0/CC-BY) + MUSAN carry **no NC and no BLOCKING term** — but LibriSpeech, AMI and MUSAN are all `cc-by-4.0`, and **under DEC-31's strictest-input rule one CC BY input makes the output CC BY**. Revision 3.1 wrote *"MIT / CC BY"*, which is **not a licence**: a slash is two verdicts, MIT imposes no attribution condition and is therefore unreachable from this mix, and a reader under schedule pressure reads the left one `[S31-6 fixed]`. **The TASL attribution notice is required in the release.** A strictly public-domain sub-tier (LJSpeech Unlicense, LibriVox PD readings, VoxPopuli CC0, AISHELL-3 Apache-2.0) is a real and interesting third tier, but it **excludes LibriSpeech, AMI, MUSAN, VCTK and Hi-Fi TTS**, i.e. most of the mix, so it gets its own row when someone wants it and not a slash in this one |
| **`auditory`, with FSD50K's CC-BY-NC clips, full ESC-50, UrbanSound8K or Clotho** | **CC BY-NC 4.0** | those four are **genuinely NC at a real rights holder's own grant** — clean NC, the GooAQ shape, not the tiny-imagenet shape |
| **`speech_output`, clean tier** | **CC BY 4.0** | LJSpeech, VCTK, Hi-Fi TTS, CSS10, M-AILABS (**8 non-Ukrainian languages** — the Ukrainian subset's *"machine learning purposes only"* carve-out is the entire reason for the count, and revision 3.1 dropped the qualifier from the phrase `[S31-20 fixed]`), AISHELL-3. Same strictest-input arithmetic as the row above: Hi-Fi TTS and VCTK are `cc-by-4.0` `[S31-6 fixed]`. **HiFiTTS-2 is EXCLUDED, and the exclusion is now declared rather than silent** `[S31-20 fixed]`: the audit lists it, it is the largest source in the bucket, and it is dropped for two stated reasons — it is **mirror-verified only**, and the audit's own open question 9 says its ~36,700 h *"would dominate any mix it joins"*. **Four of the six sources here (LJSpeech, CSS10, M-AILABS, Hi-Fi TTS) are ONE LibriVox provenance group under DEC-46**, which is a balance problem, not a licence one — see A2's gate (v) `[S31-5 fixed]` |
| **`speech_output`, with Expresso** | **CC BY-NC 4.0** | the only expressive/paralinguistic-labelled TTS source found anywhere in the audit; a **standalone-checkpoint cost only** |
| **the composed mind** | **CC BY-NC-SA 4.0, UNCHANGED** | already the strictest term via `memory` (DEC-31). CC BY-NC-SA is stricter than plain CC BY-NC, so **every NC choice audio adds is absorbed at zero additional cost to the composed release** |

**The consequence, stated plainly because it is counter-intuitive and will otherwise be
re-derived under pressure: the NC-tolerant policy has already been spent by `memory`, and
`auditory`/`speech_output` get to use it for free at the composed tier.** A real, separate cost
exists **only** if either region ever ships as a standalone checkpoint under Option D's
per-region architecture — which is why that question is OD-15 and why the recommendation there
is to build the clean tier first and keep the NC sources a named, separable add-on.

**Two things this table does NOT absorb, and conflating them with NC is the error to avoid.**
**(1) ND is not NC.** TED-LIUM 3 is CC BY-NC-**ND** 3.0 (VERIFIED at the upstream) — *"If you
remix, transform, or build upon the material, you may not distribute the modified material."*
DEC-31 moves a release licence to answer a NonCommercial term; nothing in it answers a
**NoDerivatives** prohibition, which forbids the redistributed derivative outright, commercial or
not. TED-LIUM is **outside the NC-tolerant policy's scope** and needs its own ruling: **OD-10**.
**(2) Consent is not copyright.** Common Voice's underlying CC0 dedication is irrevocable as a
copyright matter, so a frozen pre-October-2025 snapshot is defensible on licence grounds — and a
static snapshot still **cannot honour a speaker's consent revocation**, which is why Mozilla moved
distribution to a platform that can. That is a **consent-hygiene** class, not a licence class,
this document's vocabulary has no slot for it, and inventing one is **OD-11**.

**Seven new mirror-vs-upstream mismatches, and the running tally is now 17.** The audit found
`agkphysics/AudioSet` (`cc-by-4.0` over re-hosted audio Google's metadata licence never covered),
`d0rj/audiocaps` (`mit` vs *"academic purposes only"*, INFERRED from a snippet),
`cvssp/WavCaps` (`cc-by-4.0` vs *"Only academic uses are allowed"*), Clotho (Zenodo's own tag says
*"Other (Attribution)"* while its `LICENSE` file is NC — the pattern on the *host's* metadata, not
a third-party mirror), `speechcolab/gigaspeech` (`apache-2.0` over an NC upstream that also
disclaims owning the audio), `gigant/m-ailabs_speech_dataset_fr` (a meaningless generic `cc` tag
over a BSD-3-Clause-style upstream that **permits** commercial use), and CSS10 (a GitHub
Apache-2.0 **code** badge read as a data grant — the CodeSearchNet shape exactly). **Ten were
already on record for text and vision** (`memory/dataset-mirror-licences-lie.md`); audio added
seven in one pass over 35 candidates.

**Two corrections landed in the committed audit itself in revision 3.2, and they are recorded here
because that document is now the only copy the repo has of what the fact files established.**
**(a)** FSD50K's per-clip stratum shares were printed as *"CC0 36.5%, CC-BY 57.3%, CC-BY-NC
11.3%"*, which **sums to 105.1%** and is impossible for a one-licence-per-clip column; the derived
figures from the raw counts are **CC0 38.8%, CC-BY 45.9%, CC-BY-NC 11.8%, and a fourth tier,
CC Sampling+, at 3.5%** that the *"CC0+CC-BY vs NC"* phrasing silently dropped. Both filter
denominators were wrong with them (34,976 of 40,966 dev; 8,403 of **10,231** eval) `[S31-10 fixed]`.
**(b)** The audit's own bucket tables held **one** licence column — mirror tag and upstream terms
collapsed together — which is *the exact structural rule A0m's gate (i) exists to enforce*, applied
to everything except the document that states it. **The column is now split in all three tables**
`[S31-11 fixed]`, which also makes the audit directly consumable as A0m's manifest seed. The
VERIFIED/INFERRED **marks** survived the original transfer intact; the **numbers under them** did
not, and that asymmetry is the thing worth remembering about transcribing a fact file into prose. The rule that catches all of them is mechanical and is now
A0m's gate (i): **record the mirror tag and the upstream text as separate fields with the
upstream read's own URL and fetch date, and fail the row when the upstream was never independently
read.** Revision 3.1 phrased that as string inequality between the two fields, which fires on the
wrong rows in both directions; the corrected form keys on the provenance of the read
`[S31-3 fixed]`.

### DEC-46 — LibriVox is ONE provenance group, and this is the highest-leverage corpus decision available

Ten of the sixteen speech corpora audited — LibriSpeech, LibriTTS, LibriTTS-R, MLS, Libri-Light,
LJSpeech, CSS10, M-AILABS, Hi-Fi TTS, HiFiTTS-2 — are re-segmentations, re-recordings or
quality-restorations over the **same volunteer public-domain-audiobook pool**, spanning both the
auditory-input bucket and the TTS-output bucket. **A v1 mix assembled by taking the
highest-recommendation entry from each dataset table would be a ~90% LibriVox monoculture wearing
ten different names** — the `code`/CodeSearchNet-Python defect (one source with a diverse *look*)
reproduced at the catalogue level, before a row is fetched.

**Two consequences, both binding.** For **item-level disjointness**, a compose-eval item drawn
from any one member may share a speaker — or the exact recording — with training material drawn
from another, so §5.6's fingerprinting must key on the **group**, not the dataset name. For
**B1/B2**, the ten are **one source** and are capped as one. Without the correction, B1 fails on
LibriVox alone (Libri-Light is ~60,000 h by itself) or, if VoxPopuli's 400,000 unlabelled hours
are drawn on to compensate, VoxPopuli simply becomes the new dominant source.

**The balance conclusion is AMENDED in revision 3.2, because revision 3.1 asserted as a finding
what its own source declined to compute** `[S31-2 fixed]`. Revision 3.1 wrote *"`auditory` can be
built balanced from day one"* with no `[I]` tag, over an arithmetic the audit explicitly refused:
*"Sizing that ceiling against real disk and compute budgets is work this audit did not do."* The
honest statement, tagged:

> **`N_eff ≥ 3` is reachable ONLY at a per-group cap `C` of order a few hundred hours, and B1's
> operating cap fails at `C = 1,000 h`.** `[I]`, recomputed on the audit's own group hours with
> Freesound converted to hours (FSD50K ~108 h + ESC-50 ~2.8 h + UrbanSound8K ~8.8 h + Clotho
> ~24 h ≈ **145 h**; purpose-recorded = AMI 100 + VCTK 44 + AISHELL-3 85 = **229 h**; MUSAN
> **109 h**), groups capped at `C` and small groups taken at natural size, inverse Simpson on
> hours: at `C = 250 h`, `N_eff ≈ 5.4` with People's Speech in and **≈ 4.6** with it out; at
> `C = 1,000 h`, **≈ 3.9 in / ≈ 3.0 out with a max share of 0.403, over the 0.40 operating cap**;
> at `C = 5,000 h`, **≈ 3.2 in / ≈ 2.2 out at 0.477 — a clear FAIL**. **B1/B2 and corpus scale
> are in direct tension and the cap has never been sized**: three of the clean groups are two
> orders of magnitude smaller than the other three, so every cap large enough to matter for a
> JEPA pretrain pushes the big groups toward dominance. The rules pass comfortably only at a
> total `auditory` corpus of order **10³ hours**.

**People's Speech is decided explicitly rather than left to the arithmetic** `[S31-2 fixed]`. Its
`*-clean` configs are `cc-by-sa-*` as well as `cc-by-*`, so **including it moves `auditory`'s
standalone tier to CC BY-SA 4.0**, while **excluding it costs the sixth group** that DEC-46's own
`N_eff` argument leans on. **Decision: OUT of the clean tier for the standalone-licence question,
and counted in the balance arithmetic only as the SA-tier variant** — DEC-45's clean row is
therefore CC BY 4.0 with five groups, and the six-group figure is reported beside it as the
SA-tier alternative. Either way the number is printed with its cap, never asserted without one.

**Units, because B1 and B2 are share statistics** `[S31-14 fixed]`. The provenance-group table
gives LibriVox, VoxPopuli, People's Speech, purpose-recorded and MUSAN in **hours** and Freesound
in **clips** (51,197 + 2,000 + 8,732 + 4,981). A max share or an inverse Simpson computed over
mixed units is meaningless — taken naively, Freesound's 66,910 "units" would outrank MUSAN's 109,
where in hours it is ~145 against 109. **B1/B2 are computed in HOURS**, the Freesound row is
converted, and A0f's gate prints the unit.

**Two thresholds, and a receipt must report against both** `[S31-15 fixed]`. The audit's heading
says *"B1 — max single-source share ≤ 0.40"*; this document carries a **0.50 hard line** and a
**0.40 operating cap** (§5.1, `CORPUS-CONTRACT.md:1155`). The gap is not academic: the
People's-Speech-out mix at `C = 1,000 h` lands at **0.403** — over the operating cap, under the
hard line. A0f's gate (iii) reports the share against **both**, with the **hard line as the FAIL
and the operating cap as a recorded warning**, which is the shape §5.1 already uses for aqua_rat.

**Why this is a DEC and not a note.** It is the audit's own judgment call, not something any
dataset's licence page states — which is exactly the kind of claim that evaporates under
schedule pressure unless it is written down with a gate attached. **A0f's gate (iii) prints B1/B2
both ways and now PASSES OR FAILS on the grouped numbers rather than printing them** — a grouped
max share over 0.50, or a grouped `N_eff` below 3, **fails the row** `[S31-2 fixed]`. Revision
3.1's version printed two numbers and asserted nothing, which would have let A0 land with a
monoculture and a green receipt: precisely OD-13's stated late cost, reproduced inside the gate
written to prevent it. **The per-name numbers are expected to look fine** — that is the whole
defect. It needs the operator's sign-off (**OD-13**) before it becomes load-bearing for a cap
design, because most of the easy clean volume in the entire audit sits inside that one group.
**Under DEC-48 that sign-off is no longer urgent**, and OD-13 moves with the rows it binds.


## 5.8 DEC-73 — the dataset factory has run pass 1, and the measured reach amends DEC-56

DEC-57 specified the factory as a loop. It has now run once, and the output is committed at
`docs/design/DATASET-FACTORY-CATALOGUE-2026-09-03.md` with the machine-readable catalogue at
`docs/design/datasets/catalogue-2026-09-03.json`, a new *Enriched and derived datasets* section in
`LICENCE-FOR-OPEN-WEIGHTS.md`, and the evidence — ground, eight faculty surveys, eight adversarial
verification passes, the enrichment analysis and the fetched licence texts — under
`docs/design/evidence/dataset-factory-2026-09-03/`. **Every number in this section is cited from
that catalogue, not restated from a survey.**

**Counts.** **159 candidates** across **106 provenance groups** — and 106 is the number B1 and B2
are computed over, not 159. Verification: **105 VERIFIED**, **26 CONTRADICTED** (the surveyor's
verdict overturned at the primary source), **18 UNVERIFIABLE**, **10 REFUSED-CLOSED**. Verdicts:
PERMISSIVE **42**, ATTRIBUTION **21**, SHARE_ALIKE **20**, NC **13**, **UNVERIFIED 37**, BLOCKING
**11**, REFUSE **15**; 15 are EVAL-ONLY and 11 carry MODEL-OUTPUT-TERMS.

**The headline is the UNVERIFIED column, not the REFUSE column.** **37 of 159 (23%) cannot be
admitted today because no primary source settles their licence**, and 18 of those had a fetch
attempted and failed. That is the factory's backlog. It is also the direct continuation of the
mirror-lies pattern this programme already had a name for: BeIR's HF mirror family carries one
blanket `cc-by-sa-4.0` tag regardless of upstream; `code_search_net` tags *"other"* while upstream
is MIT (the lie running the other way); LeetCode-derived sets are REFUSE-TERMS under a
non-redistributable EULA; `multi_news` is research-only in its own terms.

**Token reach against the 10^10-per-region target** `[I]`, at 4 bytes per token and 64 patch tokens
per image, with `T_B1 = min(L+R, R/0.60)` — the largest total that still satisfies B1's 0.40
max-single-source share:

| faculty | clean-permissive `T_B1` | % of 10^10 | NC-inclusive `T_B1` | % of 10^10 | what NC buys |
|---|---:|---:|---:|---:|---:|
| `language_trunk` | 6.4e11 | **6,359%** | 7.2e11 | 7,192% | — |
| `numeric_math` | unbounded (generator) | **≥100%** | — | — | — |
| `reasoning` | 1.8e8 | **1.8%** | 2.6e8 | 2.6% | **+0.8 pts** |
| `reasoning`, running the DeepMind generator | unbounded | **≥100%** | — | — | — |
| `language_code` | 1.5e9 | **15%** | 1.5e9 | 15% | **0 pts** |
| `memory` | 1.3e8 | **1.3%** | 5.0e8 | **5.0%** | +3.7 pts, **3.3 of them MS MARCO** |
| `moral_safety` | 2.4e8 | **2.4%** | 2.4e8 | 2.4% | **0 pts** |
| **`visual`** | 3.2e7 | **0.3%** | 3.2e7 | **0.3%** | **0 pts** |

**Three readings, and the second is the one most likely to be assumed backwards.**

1. **Four faculties miss by one to three orders of magnitude, and `visual` misses by a factor of
   300.** Its clean-permissive tier is ~524k images ≈ 3.2e7 patch tokens; reaching 10^10 needs
   ~1.6e8 images at the current 64-token geometry, or **~3.9e7 at W7v's ACTUAL rebuilt geometry**
   — `image_size 128`, `patch_size 8` ⇒ **`n_patches 256`**, fixed by **W3r** and consumed by
   **W7v** (§4.1). *`DATASET-FACTORY-CATALOGUE-2026-09-03.md` §14 states this as ~5.1e7 at a
   224²/16² geometry of 196 tokens per image; that geometry is not W7v's row, and this document
   uses the row.* Exactly one candidate in the catalogue could plausibly supply
   that — a licence-filtered Wikimedia Commons pull — and every other visual route dead-ends at
   BLOCKING.
2. **NC BUYS ALMOST NOTHING.** `language_code`, `visual` and `moral_safety` gain **zero** points;
   `reasoning` gains 0.8; `memory` gains 3.7, of which **3.3 is MS MARCO alone** — the one entry
   whose terms the operator has to rule on rather than accept (OD-18 item 3). **B1, not licence
   tolerance, is what binds**, so the lever is *more independent permissive provenance groups*, not
   more NC volume. **This is not an argument to drop NC** — the composed model is already
   CC BY-NC-SA via GooAQ under DEC-31 — it is an argument that chasing NC volume is not the lever.
3. **`memory` is the faculty the merge made worse in licence terms, not better.** It inherits the
   union of both parents' obligations; post-merge its clean-permissive tier is 1.3% of target, its
   share-alike tier is 99% one provenance group (Wikipedia), and the single largest lever against
   GooAQ's 79% concentration — Stack Exchange — is refused **structurally** (a per-item live
   hyperlink attribution a manifest cannot discharge at any scale), not by its licence class.

**Consequence, and it is DEC-56's amendment.** The flat *10^10 tokens per region* was a scaling-law
figure applied uniformly to seven faculties before anyone had measured what any of them could
source. It survives as an order of magnitude and dies as a requirement: **per-faculty targets with
the reasoning written down, clean-terms generators for the faculties a generator can serve, and the
operator's own enrichment** replace it. Row **P2′g**. The generator route deserves naming precisely
because it is narrow: the DeepMind `mathematics_dataset` generator is **the single source in the
whole catalogue that can supply arbitrary B1-relief volume under a settled Apache-2.0 grant with no
new licence question**, which is why it moves two faculties from single-digit percentages to
unbounded and why *"just generate more"* is not otherwise available.

**Five legal readings go to the operator as OD-18** (§8), ranked by what they could invalidate.
**The first item needs no lawyer and is actionable today: the model card must STATE the reading it
takes on CC BY-NC-SA given CC BY-SA inputs, and today it states none** — which is why the question
is currently unanswerable rather than merely unanswered.


# 6. Scale path and the dynamic-paging seam

## 6.1 The five numbers the scale path is derived from

| # | number | derivation |
|---|---|---|
| **N1** | **3.2675 effective bits/param** | `stored_bytes 6,543,592 × 8 / 16,021,248 params`, from the real `code` PTQ receipt [V\*] |
| **N2** | **11.41 GiB** for 30B at N1 | `30e9 × 3.267 / 8 / 2^30` |
| **N3** | **≈3.29 GiB** KV budget on a 16 GiB card | `16.0 − 11.41 − 0.50 (CUDA ctx + frag) − 0.80 (activations)` |
| **N4** | **192 KB/token** bf16 KV for a 48-layer, 8-KV-head × 128 region | `4 · L · d_kv = 4 · 48 · 1024` |
| **N5** | **80.30%** of every text region is its token-embedding table | `50,257 × 256 = 12,865,792 / 16,021,248` — exact reconstruction of the receipt param count |

**N3 ÷ N4 is the whole architecture in one division.**

| assumption | KV/token | total token budget |
|---|---|---|
| bf16, 48-layer region | 192 KB | **≈17,900 tokens** |
| bf16, 32-layer region | 128 KB | ≈26,900 tokens |
| **int8, 32-layer region (DEC-25, the v1 assumption)** | **64 KB** | **≈53,900 tokens** |

**In every case that is the total across every region active at once. Not per region. Total.**
That is why the interconnect must be a scheduler and not a router: a router picks who runs;
something has to decide *who gets how many of those tokens*, and that decision is a hard,
linear, byte-denominated constraint that no amount of emergent attention expresses on its own.

**The check that makes it concrete, and it survives the optimistic assumption.** Long context is
*primarily visual*. One 1920×1080 screenshot at patch-16 is `120 × 67 = 8,040` patch tokens.

| budget | equal share across 9 regions | one screenshot fits? |
|---|---|---|
| 17,900 (conservative) | 1,989 | **no**, by 4× |
| 53,900 (int8, optimistic) | 5,989 | **no** |

**Uniform per-region context is structurally incapable of serving the stated deployment
target.** A learned allocator under `Σ_r c_r·ctx_r ≤ B_kv` is not an elegance; it is the only
configuration in which the mind can look at a screen. The scheduler is load-bearing at the
target *regardless of the biology*, and the arithmetic says so independently [I, grafted from
biology-first; a judge called it the best available argument that the operator's framing is
right on its own terms].

**Composite frame geometry, fixed by W3r (§4.1) and tracked here rather than left open.** Not a
sixth entry in the table above — those five are specifically the scale-path derivation — but the
`visual` region's KV/token cost (§1.4's `kv_bytes_per_token`, currently 9,216 at `n_patches 64`)
is one of the per-region terms in this section's `Σ_r c_r·ctx_r ≤ B_kv` budget, and its `n_patches`
will change once W7v rebuilds the encoder against the geometry this row fixes:

| quantity | value | source |
|---|---|---|
| composite frame side | 128 px | `src/cogsyndelta/vl/composite.py:FRAME_SIZE` |
| patch side | 8 px | `src/cogsyndelta/vl/composite.py:PATCH_SIZE` |
| patches per frame | 256 | `src/cogsyndelta/vl/composite.py:N_PATCHES` (`GRID=16`, `16²`) |

Checked-in ground truth for the triple: `tests/fixtures/composite_4panel.png`, byte-reproduced
from `tests/fixtures/composite_4panel.spec.json`. `visual`'s live config (§1.4) stays at
`image_size 64, n_patches 64` — and every KV/token figure derived from it above stays the
64-patch number — until W7v's retrain actually lands; this table records the fixed target, not
a claim that the rebuild has happened.

## 6.2 DEC-24 — the shared embedding table, decided now because it is cheap now

Six text regions carry six copies of the same GPT-2 BPE table: `6 × 12,865,792 = 77,194,752` of
`96,127,488` total text-region params. Sharing gives `12,865,792 + 6 × 3,155,456 = 31,798,528`
— a **3.02× reduction for zero capability loss** [V\*]. At 30B with `d_model 6144` one shared
table is 0.6% of the model.

> **So it must be decided now, while it is cheap to change and matters, not later when it is
> expensive to change and doesn't.**

**Applied how, honestly** `[A31 fixed]`. Revision 1 wrote *"mandatory from the first region
trained after ratification"* and then conceded *"the v1 mind keeps three separate tables"* — a
mandate unsatisfiable at its first application, since the first rows that train anything (W1b,
then W4) are a receipt regeneration and a two-parent merge whose tables have already diverged,
with frozen peers keeping their own. Three corrections:

1. **`memory` inherits `retrieve`'s table**, and the divergence between the two parents' tables
   is measured and recorded in W4's receipt. `retrieve` because it is the parent whose gate
   (the FiQA BEIR pool) survives as `memory`'s gate, so its tokenisation statistics are the ones
   the surviving eval is calibrated against.
2. **DEC-24 binds from phase 3 (P5′)**, where regions unfreeze and sharing is achievable — not
   from "the first region trained", which was never a reachable trigger.
3. **W1's result creates the one earlier opportunity, and it should be taken.** W4 and W7a
   retrain every text region anyway (§4.0). That is the moment three tables *could* become one
   at no incremental cost beyond the retrain already being paid for. It is **not** made
   mandatory here, because it would couple the token-aware retrain's gate to a second variable
   and a failure would be undiagnosable — but W4's receipt must record whether the table was
   shared, so phase 3 knows what it inherited.

The 3.02× figure stands as the phase-3 lever it is, with one label correction: it is computed
for **six** text regions, a configuration §1.2 has merged down to **three**. At three regions
the saving is `2 × 12,865,792 = 25,731,584` params, a 1.67× reduction on the text-region total.

## 6.3 Feasibility conditions, not optimisations

- **GQA/MQA is a feasibility condition.** Full MHA at `d_kv = 6144` costs **1,152 KB/token**;
  16k tokens is **18.0 GiB — more than the whole card** [V\*]. An architecture that does not
  state this can be specified into infeasibility.
- **Flash / SDPA attention is mandatory.** Materialised attention logits at `T = 16,384`, 48
  heads, bf16 are `16384² × 48 × 2 = 25.8 GB` [V\*]. The tree already uses
  `F.scaled_dot_product_attention`, so this is preserved, not introduced.
- **DEC-25 — int8 KV is the v1 assumption, not the contingency.** The bf16 budget closes to
  16.00 GiB *exactly, with zero margin*; a budget that closes to the byte on the number the whole
  design is derived from has not been closed. `ScheduleGraph.nodes[].kv_precision` carries
  `bf16 | int8 | int4`; declare the field now, build the kernel when the constraint is measured —
  which by this arithmetic is at the target itself.

## 6.4 DEC-26 — the interconnect parameter cap, set while `D` and `L_ic` are still free

**2–5% of total params — and the cap is scoped, because v1 violates it by 6×** `[A25 fixed]`.

```
v1 interconnect share = 27,424,039 / 86,331,303 = 31.8%   (was 26,899,751 / 85,807,015 = 31.4%
                                                        before DEC-49 restored the store)
```

Revision 1 set the cap in §6.4 and, four sections earlier, celebrated that *"white matter is
8.5× a region's compute parameters… the centre of gravity in arithmetic"* — two tables that
cannot both be honoured, with nothing saying the cap applied only at the 30B target. **The cap
is scoped explicitly:**

> **DEC-26 binds at ≥ 1B total parameters.** Below 1B the interconnect is **deliberately
> dominant** — that is the thesis, not an overrun — and the v1 share (**31.8%**) is printed
> beside the cap so a ratifying operator sees the exemption rather than inferring it from two
> tables four sections apart. The crossover is where region *compute* parameters (not embedding
> tables) begin to dominate: at v1, regions are 19.7% compute and 80.3% embedding, so the
> interconnect is 8.5× a region's compute; by 1B per region that ratio inverts.

**At the MLP ratio this design actually specifies, the ceiling claim flips** `[A26 fixed]`. §2.3
states the workspace blocks are `mlp×4`, and v1's `16,803,840 / 4 = 4,200,960 ≈ 16·512²`
confirms it. Revision 1's 30B figure of `805,453,824` is `12·D²` per block — **`mlp_ratio 2`**,
a silent architecture change between two columns of a table arguing an invariant.

| `L_ic` | params at `D=4096`, **`mlp_ratio 4`** | share of 30B | verdict |
|---|---|---|---|
| 4 | 1,073,741,824 | **3.6%** | inside the 2–5% band |
| 8 | 2,147,483,648 | **7.2%** | **above the 5% ceiling** — not the ceiling revision 1 named |

So `L_ic = 4` is the design point and `L_ic = 8` is **outside** the cap, not at it. If depth 8 is
ever wanted, either the ratio drops to 2 (recovering 5.4%) or the cap is renegotiated — and
either is a decision, made visibly, rather than a number that quietly changed between columns.
It remains compatible with the workspace's asymptotics: no term in
`O(L·D_w²·(1+mlp_ratio) + Σb·D_w² + L·Σb·D_w)` grows with region size (§2.3).

**What must stay high precision regardless of the cap:** the thalamic controller (1.59M) and the
connectivity read-out. An error in a region's weight perturbs one contribution; an error in an
attention weight **redirects information**, and an error in the controller *changes which
faculties ran*. 61 KB of insurance, with existing precedent in the tree (`visual` already
declares `quantization.skip: true` [V] — `config/mind/csd-regions.json` names it `visual` today;
`vl_latent` is its legacy alias).

## 6.5 DEC-27 — quantisation gates on the schedule, not only on the metric

The PTQ machinery is real and measured: sensitivity-greedy mixed width, **9.79×** compression on
`code` with recall drop **0.0039**, `within_budget: true`, `width_histogram {3:16, 4:1}` [V\*].
The pattern to preserve: norms and the two MLP biases per block stay fp32 (26 tensors, 9,728
params) [V\*].

**The breakage mechanism nobody else names:** quantisation does not merely lose accuracy inside
a region — it shifts that region's output distribution, which shifts the interconnect's
attention logits, **which changes the schedule**. Different regions at different widths drift by
different amounts. So:

```
D_sched = E_items [ JSD( a_fp32(item) ‖ a_quant(item) ) ]      # over the region-attention mass
```

**W10 gate:** quantise **one region at a time**; per region `D_sched ≤ 0.02` nats **and**
composed metric drop ≤ 0.01 — the same tolerance the existing quant receipts already use. A
region that fails gets its width promoted, exactly as `retrieve` already promoted two tensors to
5 and 8 bits [V\*]. The promotion machinery exists; it only needs a second objective. **The
read-out heads and adapters quantise with the interconnect, not with their region** — they sit
at the seam where distribution drift becomes schedule drift.

**`D_sched` becomes a training-time quantity if P5′q wins** `[OP: csd-quantize-before-whole-mind-training]`.
The phase-3 experiment (§4.1, row **P5′q**) asks whether quantising the regions *before* the
interconnect trains cuts peak VRAM without a drastic quality hit — at toy scale it is free to
test, and at 1B+ it decides whether phase 3 fits on 24 GiB at all, let alone 16 GiB. The design
consequence is stated here rather than in the row, because it changes what this decision means:

> **If quantise-first wins, the interconnect must train against QUANTISED region activations**,
> so that its train-time inputs match deployment — and per-region PTQ sensitivity must therefore
> be measured **before** regions are frozen, not after. `D_sched` then stops being only a release
> gate and becomes a **training-time constraint**: a region whose quantisation moves the schedule
> is a region the interconnect is being trained against under the wrong distribution.

That is the QLoRA insight applied to a composed mind — a quantised frozen base with small
trainable parts — and the answer is empirical rather than borrowed, because the composition is
novel. The canonical path stays the default at toy size; P5′q is the matched experiment.

## 6.6 DEC-28 — the dynamic-paging seam

*"Design the seam, build when the constraint is measured."*

```python
class WeightStore(Protocol):
    def ensure_resident(self, region: str, precision: str) -> Handle: ...
    def evict(self, region: str) -> None: ...
    def residency(self) -> dict[str, bool]: ...
```

`Schedule.nodes[].resident` is the scheduler's *request*; `WeightStore` is the runtime's
*fulfilment*. Three rules, each derived from a number rather than asserted:

1. **Trigger.** Build it when `Σ(weight bytes of the resident set) > 0.75 × (VRAM − KV budget −
   activation budget)`. At 30B / N1 that is 11.41 GiB against a ceiling of
   `16.0 − 3.29 − 0.80 − 0.50 = 11.41 GiB` — **the seam triggers essentially at the target
   itself** [V\*]. Do not build it before then; do not be surprised when it is needed.
2. **Granularity: request or stage boundaries, never per token.** A 3B region at N1 is 1.14 GiB
   → **61 ms** over PCIe 4.0 x16 at ~20 GB/s effective, against a ~20 ms/token decode step
   [V\*]. Once per request: 3 tokens of latency. Per token: a 4× slowdown. **An adapter is 17 MB
   and 0.85 ms**, which is what makes *"broad competence with deep spikes"* physically
   realisable — dozens of instantly-swappable specialisations, not dozens of resident regions.
3. **Prefetch is free in this design, and it is the reason it pages well.** The `Schedule` is
   emitted **before** execution, so at iteration `i` the runtime already knows the full
   activation set for `i+1` and can prefetch depth-`i+1` regions while depth-`i` computes. **A
   dispatch router cannot do this — it learns the next region only by running the current one**
   [I]. **`Schedule.stages` IS the paging plan**: the scheduler built for compute allocation
   turns out to have already emitted the memory plan, so there is no separate pager, ever.

v1 implementation: `ensure_resident` is a no-op and residency is always `"resident"`. The
interface costs nothing now and the DAG makes it correct later.

### DEC-33 — memory-gate overlays: the seam's first client, and what it requires of the seam

*"Design the seam, build when the constraint is measured"* is only honest if the seam is designed
against a real client. **Memory-gate overlays are that client, and they arrive before 30B does**
`[OP: csd-memory-gate-overlays]`. The requirement, in the operator's terms: learned
specialisation lives in **differential overlays/offsets on weights and/or activations — never in
edits to the base weights** — recalled and applied selectively at inference for the use case at
hand, lightweight in VRAM, with a **loud** disconnect back to default weights.

Four requirements this places on `WeightStore`, and they are not the ones a pure 30B paging story
would have produced:

1. **Residency is TIERED, not binary.** `residency() -> dict[str, bool]` is insufficient. Hot
   overlays stay in VRAM, warm in GPU/host cache, cold on disk, and **promotion and eviction are
   scored on importance, size, age and utility** (recency and frequency of use). That is a
   memory hierarchy with a *policy*, and the policy is a design object that must be **measured —
   hit rate, page-in latency, VRAM held — not hard-coded**. The protocol becomes
   `residency() -> dict[str, Tier]` with `Tier ∈ {vram, cache, disk}`, and `ensure_resident`
   takes a target tier.
2. **Granularity is the adapter, which is why this works at all.** §6.6's own arithmetic: a 3B
   region is 1.14 GiB and **61 ms** to page; **an adapter is 17 MB and 0.85 ms** [V\*]. Overlays
   are adapter-shaped by construction — low-rank weight deltas per region for *skills*,
   activation-steering offsets on the shared stream for *personas* — so *"broad competence with
   deep spikes"* is dozens of instantly-swappable specialisations rather than dozens of resident
   regions.
3. **An overlay REFUSES to attach to a base fingerprint it was not trained against.** Overlay
   provenance records what data trained it and **which base checkpoint fingerprint it targets**;
   attaching to a different fingerprint raises. This is the same fail-closed pattern as DEC-40's
   checkpoint hash, and it is **the second client of that mechanism** — which is the argument for
   building the fingerprint properly once rather than twice.
4. **Disconnect is LOUD.** Reverting to base weights is observable and logged, never a silent
   degradation. *A fallback nobody can see is not a fallback.* **Gate (P5′o): an overlay applied
   and then disconnected reproduces the base model's receipt metrics EXACTLY, and the disconnect
   appears in the log.**

**Licence:** the strictest-input rule (DEC-31) applies to overlays as it does to regions — an
overlay trained on NC data is NC, and it makes anything it is attached to NC while attached.

**Composition:** overlays compose with frozen quantised regions (P5′q) and with per-region
selective activation. That is three mechanisms meeting at one seam, which is the reason to write
the seam now: `Schedule.nodes[].resident` is already the scheduler's *request*, and an overlay is
just a smaller thing to request.

### DEC-70 — the memory gate is a VRAM ARBITER: named claimants, a priority order, and a frugality cap

DEC-33 gave overlays a residency policy. DEC-63 gave the store a capacity formula. **Neither says
what happens when the model, the KV cache and memory all want the same megabyte at the same
moment**, and the operator's ruling is that this is a **negotiation between peers**, not a
leftover: memory-gate **does not take what is free writ large — it must contend and negotiate for
VRAM against the context window / KV cache and against the base weights and activations, as peers
with their own claims** `[OP: csd-memory-gate-overlays.md]`. *The source note records this and the
priority order below under the heading "operator, paraphrased faithfully", so the position is the
operator's and the wording is the note's; it is set in bold rather than in quotation marks for that
reason, and the same applies to every operator position in this decision.*

**The claimants, and the priority order among them.** The order is the operator's, and it is the
reverse of what a naive reading of DEC-63's residual formula would produce:

| priority | claimant | behaviour under pressure |
|---|---|---|
| 1 | **base weights** | **FIXED.** Not negotiable; the mind is the mind. |
| 2 | **working activations** (per step / per request) | **FIXED** for the admitted batch and schedule. |
| 3 | **the ACTIVE memory set and the LOADED PERSONA** | **PROTECTED. Never evicted to make room for context.** |
| 4 | **KV cache / context window** | **FLEXES** around 1–3: sliding window, shorter context. |
| 5 | **inactive overlays, offsets, differentials** | **LEAVE FIRST.** Below context, evicted proactively. |

**Why memory outranks context, in the operator's own reasoning:** a sliding context window will
help a great deal, but **dropping a relevant memory or the persona because more context is wanted
can create failures** — again the note's paraphrase of the operator's position, not marked speech. A model that forgets who it is in order to read four thousand more tokens has
made the wrong trade, and it makes it silently.

**The frugality cap is what makes protecting memory safe, and without it this priority order is
dangerous.** A protected claimant with no ceiling is an unbounded claimant. So: **a design-doc
constant caps the resident memory + persona footprint at a small fraction of the card, stated per
card**, enforced by the residency scorer, with the target that it is **negligible relative to the
KV budget at the deployment context length**. **The gate that makes the cap real: at the cap, the
achievable context length differs from the no-memory case by less than a stated margin.** If it
does not, the cap is wrong and the row fails — not the test.

**DEC-63 is amended in the same breath, because the two are now one policy.** `capacity_bytes` is a
**CEILING, not a target**. The formula says what the store *may* claim; it never said what the store
*should* claim. The resident set is what the residency score admits **now** — bounded additionally
by the frugality cap — and `capacity_bytes` is the line that bound may not cross. **A residual read
as an allocation is exactly how a subsystem designed to be frugal becomes the largest one on the
card.**

**Residency scoring and eviction.** Promotion and eviction are scored on **recency**, **relevance to
the loaded persona/basin**, and **access frequency**, across three tiers — **card / host RAM /
disk** — extending memory-gate's own importance-scored eviction with a GPU-residency bonus by a
persona-relevance term (§1.3's clause (2)). **Eviction is PROACTIVE**: on inactivity (an overlay
loaded but unused past its TTL is demoted) and on VRAM pressure from the model, the KV cache or
activations — **never only on capacity overflow**, which is the shape that guarantees the store is
holding its maximum at the exact moment something else needs the card.

**Failure to size is reported and degrades by relevance; it never silently drops the persona.** If
the relevant memory/persona set cannot fit within the cap, the system emits a **receipt field and
an event** and **degrades the memory set by relevance** — it does not shorten the context by
default, and it does not drop the persona at all. *A degradation nobody can see is the same defect
as DEC-33's silent fallback, one layer up.*

**Frugality is measured, not asserted.** Every receipt records **peak VRAM held by memory +
overlays** and **the fraction of that which was active**. A design-doc row states the target ratio,
and E1a's four gates — three of them constructed to fire — are what turn this decision into a
result. Row **E1a**; the arbiter's inference-time packing arithmetic is W7k's prototype, reusing
gpu-pack's admission model rather than inventing a second one (DEC-69).


## 6.7 DEC-29 — training the fleet at 1 Gb/s, with one correction to the programme

The programme's statement that **cross-host DDP is the wrong tool** is right: a 16M-param model
is ~64 MB of gradients per step per direction, ≈0.58 s at ~110 MB/s effective, which exceeds the
step time it would overlap [V\*]. But that argument is about *gradients* and does not generalise
to phase 3.

| phase | what crosses the wire | volume | verdict |
|---|---|---|---|
| **1** per-region pretraining | nothing | 0 | **region-parallel**, near-linear in hosts. Unchanged. |
| **2** interconnect training | a one-time cached emission dump — **regions are frozen, so their depth-0 tokens are a pure function of the input and can be computed once** | at 16 read tokens/region: `51,200 × 4 × 16 × 512 × 2 B = 3.36 GB` → **31 s** over 1 GbE [V\*]; at the v1 default of 64: **13.4 GB → ~122 s** [I] | **one host**, zero gradient traffic. The 5080 trains the interconnect against a cached emission file **while the 3090 keeps pretraining regions.** Both hosts productive at once. **Caveat, and it is a real interaction between two of this document's grafts:** write-back (DEC-17) makes `h_r` depend on `z` at iteration ≥1, so only the **depth-0** emissions cache. With write-back disabled the whole of phase A caches; with it enabled, one iteration does and the rest are recomputed. |
| **3** whole-mind dynamic training | pipeline **activations**, not gradients | microbatch 32: **2.1 MB → 19 ms**; microbatch 256: 16.8 MB → 153 ms | **region-granular pipeline parallel is viable where DDP is not** |

**The correction:** pipeline parallelism sends `O(microbatch × slots × D)` while DDP sends
`O(params)` — **2.1 MB vs 64 MB at microbatch 32, a 30× reduction**, and it overlaps with compute
rather than blocking a step boundary [V\*].

**Bound, stated so it can be falsified:** the claim holds only at modest microbatch. At the
runner's current `DEFAULT_BATCH = 1280` the transfer is **83.9 MB ≈ 763 ms**, which is not under
15% of any plausible step time [V\*]. **Cap phase-3 cross-host microbatch at 256** and use
gradient accumulation to recover effective batch. P5′'s gate measures this rather than assuming
it: **cross-host activation traffic ≤ 15% of step time.**

**The honest limit that must be recorded now rather than discovered later:** phase 3 at full 30B
scale is not trainable on this fleet — joint training couples every region and the largest card
is 24 GiB. No architecture choice fixes this. The named mitigation is **progressive unfreezing
by tract**: unfreeze one faculty plus the tracts touching it, rotate. That is memory-bounded and
is the direct analogue of region-per-host for a coupled model. The pipeline-parallel arithmetic
above recovers more fleet capability than a flat "not trainable" would suggest, but it does not
recover all of it.

**And there is a second lever on the same limit, which is why P5′q exists.** Pipeline
parallelism moves *activations* between hosts; it does nothing about the fact that the canonical
path needs fp32/bf16 regions **plus optimizer state plus activations** resident during phase 3.
Frozen **quantised** regions with a trainable interconnect is the shape that fits, and at 3.27
measured effective bits/param the difference is not marginal. So **P5′q's headline measurement
is peak VRAM during phase 3**, matched against the canonical run, and it is the number that
decides whether progressive unfreezing is a mitigation or the only option
`[OP: csd-quantize-before-whole-mind-training]`.


### DEC-54 / DEC-55 — placement at the CURRENT scale, which is a different problem from DEC-29's

**DEC-29 is about phase 3 at size; this is about phase 1 today, and the two were being conflated.**
Submodels are **several million to tens of millions of parameters in fp32 now, at most ~100–300M
each later** — every one of them fits on **every** card, so the binding constraint is not
partitioning a model across cards but **packing independent jobs onto them**
`[OP: csd-training-placement-policy.md]`.

**The policy (DEC-54).** Train individual submodels on the **3090 Ti** (24 GiB, sm_86) and the
**5080** (16 GiB, sm_120), and on the **1080 Ti** (11 GiB, sm_61) where the corpus and the torch
build allow. **Where budgets fit, train two, three or more concurrently on the 3090 Ti** and fewer
on the 5080. **There is no MIG on consumer cards**, so co-scheduling is **process-level
pseudo-isolation** — per-process memory fractions, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
warp-level sharing — **and that word is load-bearing: it is not hard isolation and must never be
cited as a security boundary.** It is acceptable here for one stated reason: the fleet is
**single-tenant** (operator plus agents), not an enterprise multi-tenant deployment.

**The scheduler must budget VRAM per job, and the reason is a receipt, not a prediction.** The
`reason` region OOMed at batch 1280 / `max_len` 256 on the 3090 Ti, **640 MiB short of the 22 GiB
card**, and re-ran clean **alone at batch 512** with `expandable_segments` — after which it passed
both its gates [V, `reason-20260903T123431Z`]. So `reason` carries a declared **`exclusive: true`**
and its measured configuration, and the admission check refuses to pack it. **W7p is that harness**
(§4.1): a per-job VRAM budget, a target host, a refusal when the budget exceeds free VRAM, a
concurrency admission check, and **`placement { host, gpu_name, vram_budget_mib, vram_peak_mib,
concurrency, co_resident_jobs[] }` in every receipt** — because a placement policy whose receipts
do not record placement cannot be audited, and the concurrent runs it licenses (W7a's two regions,
W7v, W4) are exactly the ones whose numbers a later gate compares.

**Larger models: DEC-55, a candidate and not a plan.** At 1B+ per region, later phases may be
**locked to the 3090 Ti + 5080, or to the 3090 Ti alone**, unless **layer-sectioned training** —
training isolated sections of layers/weights on different cards rather than the whole model at once
— lets all three contribute. The operator flags it as worth considering **with the explicit caveat
that they may be wrong about some of those techniques**, and that caveat is why it is row **P5′L**
with a pre-registered comparison rather than a decision: it is measured against **DEC-29's
region-granular pipeline parallel** on the same model and microbatch, over this fleet's real 1 Gb/s
link, with *"cross-host DDP is the wrong tool at 1 Gb/s"* as the standing null hypothesis. **A null
result is a useful outcome** — it makes the two-card lock an honest constraint instead of an
unexamined one.

### DEC-69 — the knobs, in one table, with what each one was measured to cost

Packing and chunking were each solved as an incident: `reason` OOMed and got a batch, `memory` OOMed
at 1280 and got a chunked loss. **As a table they stop being incidents and become a control
surface** — and the operator's framing is the reason to write it down: once packing and chunking
land they **become a set of knobs for tuning how training and quantization run**, and the same
admission model is the **foundation for deciding how to pack MODELS for runtime inference**
`[OP: csd-training-placement-policy.md]`. *(Positions carried faithfully from the source note's
"Operator intent" paragraph, which records intent in the note-taker's prose rather than as marked
operator speech; the emphasis is this document's and the words are not offered as a quotation.)*

| knob | what it actually controls | measured effect | where it is recorded |
|---|---|---|---|
| **per-job VRAM budget** | the per-process allocator fraction, and what admission charges the card | a budget larger than the card's free VRAM **refuses to launch** (W7p gate 1); under-use never returns headroom — the claim stays pinned at `max(budget, measured)` | `knobs.vram_budget_mib`, W7k |
| **admission margin** | slack held back on every card before anything is admitted | gpu-pack default **1,024 MiB**; `allowed = total − foreign_used − Σ max(budget, measured) − margin` | `knobs.admission_margin_mib`, W7k |
| **`token_loss_chunk`** | rows of masked positions projected to vocabulary at once, under `torch.utils.checkpoint` | **2048 ⇒ 20,357 MiB allocated / 20,726 reserved / 22,120 raw driver peak; 512 ⇒ 19,080 / 19,444 / 20,883, at +2.6% step time (380.0 → 390.0 ms)**. Chunk peak is `chunk × vocab` and is therefore **batch-independent** | `knobs.token_loss_chunk`, W7k |
| **batch** | **the in-batch-negatives count** — a quality lever before it is a speed lever | full-pool `recall@10` **0.030 / 0.098 / 0.200** at batch **256 / 512 / 1280** (DEC-68); InfoNCE's mutual-information ceiling is `log(B)` nats — 5.55 / 6.24 / 7.15 | `knobs.batch_size` **and** `knobs.negatives_effective`, W7k |
| **mask probability** | `n_masked`, and therefore the whole cost of the token term | **0.15 at production. MEASURED [V]:** the batch-1280 receipt's own `token_loss_n_masked` runs **10,815–11,333, mean ~11.0k** masked positions per call across its history — **× 50,257 vocab in fp32 ≈ 2.2 GB of logits per call, twice per step** — which is what `token_loss_chunk` exists to bound. **[I], and it is an upper bound, not a measurement:** the **~18.4k / 3.7 GB** figure this table carried from the OOM diagnosis is `1280 × 96 × 0.15`, which assumes every sequence is `max_len`; the corpus is ragged (`tokenisation.ragged_ratio` **0.2967**, ~58 real tokens per sequence), so the estimate overshoots by ~1.7×. W7k records the receipt's field, not the estimate | `knobs.mask_prob`, W7k |
| **card choice** | VRAM against compute | 3090 Ti 24 GiB sm_86 / 5080 16 GiB sm_120 / 1080 Ti 11 GiB sm_61 (**inference-role by default, DEC-74**). The 5080 may win on compute alone at smaller VRAM where layer-sectioned training (DEC-55) applies | `knobs.card`, `placement{}`, W7p |

**And one number that is NOT a knob and must never be absorbed into a margin: the operator's
desktop.** Both chunk probes measured **981 MiB and 1,057 MiB** of Xorg/KDE/Firefox resident on the
3090 Ti **before the job started**. A budget computed as though that card were headless is wrong by
about a gigabyte, and the failure mode is an OOM attributed to the job. **gpu-pack already models it
correctly** — as `foreign_used`, subtracted before admission, rather than as slack — and W7k records
it per run as `foreign_baseline_mib` so a wrong assumption is visible afterwards instead of inferred
from a crash. **It is a standing FOREIGN claim on a workstation card and it is variable**; a headless
training host does not have it, which is exactly why it must be measured per launch and not
constant-folded into the design.

**The admission model is reused at inference, not re-derived.** gpu-pack's
`allowed = total − foreign_used − Σ max(budget, measured) − margin`, its `claim(entry)` rule (a
budget under-used never gives headroom back; only a measured overrun raises the claim), its
fail-closed corrupt-registry behaviour and its host/gpu scoping filter are **the starting point for
packing regions, overlays, the episodic store and the KV cache under DEC-70's arbiter at inference
time**. Two properties transfer directly and are the reason to reuse rather than rewrite: the
**safe direction on unknown usage** (an unmeasurable claimant is double-charged against itself
rather than allowed to mask another's memory) and **one critical section** around
expire-check-admit. Row **W7k** builds the prototype that finds out whether the rest transfers,
before anything is built on the assumption that it does.

### DEC-74 — the fleet is on demand, and the 1080 Ti's training verdict is a measurement

**Two stale claims were live in this programme's documents and both cost probing time.**

**(1) All three cards and all their runtimes are ON DEMAND.** LocalAI on akula-prime and on gpu5080,
llama.cpp on the 1080 Ti VM, Open WebUI and ComfyUI — started when needed, stopped when idle. **An
idle `:8080`, a `000`/connection-refused, a stopped unit or an exited container is the NORMAL
RESTING STATE, not a fault**, and is not to be "fixed" `[OP: fleet-gpu-roles-and-scheduling.md]`.
Training and quant jobs are admitted per card by VRAM budget through the packer (DEC-75), which is
what makes on-demand safe: a card is not reserved by a running server, it is reserved by an
admission entry.

**(2) The 1080 Ti cannot run the fleet's pinned torch at all, and DEC-54's phrasing is narrowed by
that measurement rather than left standing.** DEC-54 said submodels could train on the 1080 Ti
*"where the corpus and torch build allow"*, which reads as a scheduling caveat. It is not:

- **`torch 2.11.0+cu128` has no sm_61 target.** `get_arch_list()` floors at **sm_75**, and the VM's
  driver **535.274** caps at CUDA 12.2 — so the pinned build cannot execute a single kernel there
  `[V, probe 2026-09-03]`.
- **A separate `torch==2.5.1+cu121` venv on the guest works**: a 2048² matmul runs with zero
  warnings. It carries **none** of CSD's other dependencies, the VM has 7.8 GiB RAM, and it sees
  the datasets RAID read-only at `/mnt/fleet-datasets` with no `/akula-data` and no route to
  gpu5080's `/mnt/bulk`.

**So fp32 training on that card is "with environment X: yes"** — a second CSD environment pinned to
cu121 or cu118 — **and until someone builds it, the card's DEFAULT ROLE IS MODEL SERVING, RAG AND
UTILITY INFERENCE**, which is what the packer defaults it to. That is the operator's own framing:
the card sits in a VM precisely so it can carry its own CUDA and driver stack, and *"if an
sm_61-capable training env is not worth the squeeze, its role is running models, RAG, and
tool/utility inference"* `[OP: csd-training-placement-policy.md]`. **The 1080 Ti's RAG claim
remains preemptible** and the timeshare scheduler still spans all three cards; what changes is that
"all three cards train" is now "two cards train and the third serves, unless someone builds the
env".

### DEC-75 — tooling lives in its own repo, and `tzervas/gpu-pack` is W7p's implementation

The operator's rule, verbatim: *"any tooling or harness or framework developed or built must be
captured in its own forgejo repo keeping each repo aligned to their actual role and responsibility
and preventing tooling from bloating CSD repo"* `[OP: tooling-lives-in-its-own-repo.md]`.

**What CogSynDelta keeps:** the **model** — regions, interconnect, training objectives, evals,
corpus contracts, receipt formats — **plus the thin adapters a tool needs**: an environment variable
honoured, a job-spec file read. **What leaves:** the tool's code, its tests, its docs and its own
CI.

**W7p was written as a row without naming where its code would live, and it now has an answer.**
`tzervas/gpu-pack`, **`main` at `5364932`** (rounds 1–4 merged through PRs #1 and #2), implements
the probe, the budget ledger, admission under one flock, the per-process cap, transient units with
emitters, remote launch and launch receipts — and is also the home of the emitter scripts that were
living uncaptured under `/akula-data/csd/events/bin/`. **CSD's entire side of it is already in this
tree and stays that size**: `src/cogsyndelta/util/gpu_budget.py` (`GPU_PACK_BUDGET_MIB` caps the
allocator fraction, `GPU_PACK_PROBE=1` caps a run to a 20-step probe, `GPU_PACK_PEAK_MIB` reported
to stderr on exit) plus the example job specs under `program/jobs/`. **P10.4's status — "gpu-pack's
probe/admit/launch pipeline is the remaining piece" — is retired by that merge**, and W7p's status
cell in §4.1 is corrected in the same pass rather than left reading `todo` beside a shipped tool.

**One trap is worth carrying here because it has already cost a run.** The main checkout's editable
install pins the **main** `src/` on `sys.path`, so a job spec that means to run a worktree's code
must set `PYTHONPATH` explicitly — the chunk-512 probe had to do exactly this because the shared
venv resolved `cogsyndelta` to a checkout with no `token_loss_chunk` field at all.

**The same rule places two other things:** the predictive trainer stays in `tzervas/tritter` with a
CSD adapter only (DEC-72), and autodev stays in `tzervas/csd-autodev` with a pointer only (DEC-60,
DEC-76).


## 6.8 Hybrid context management falls out; it is not a separate subsystem

The requirement is *"sliding windows over an overarching context, plus latent-reasoning windows.
Per-region budgets, not one global window"* [V\*]. In this design:

| requirement | mechanism at v1, phase 2 **before** E2 | mechanism at v1 **after** E2, i.e. from W6 onward `[DEC-49]` |
|---|---|---|
| sliding windows over an overarching context | each region's `ctx_r` window over its own long input, re-selected per iteration | unchanged |
| **the overarching context** | **the workspace latents `z`, persisted across the `n_iter` loop, plus the per-request `h_r` cache** — bounded, per-request, and honest about being *within*-request only | **`episodic_store`** — a participant with its own budget. Long context becomes *a region you allocate budget to*, not a special case |
| latent-reasoning windows | the workspace latents `z [B,64,512]` themselves | unchanged |
| per-region budgets | `ctx` and `b`, the two simplexes | unchanged |

One mechanism, four requirements, no separate context manager — **and the honest correction has
itself been corrected twice, which is worth tracing because it is the shape of the whole
argument** `[A17 fixed] [T2 fixed] [DEC-49]`. Revision 1 satisfied *"the overarching context"* with
`episodic_store` and shipped it as `built` — a store with no write policy, empty at iteration 0,
satisfying the requirement on paper and contributing nothing on any eval item. Revision 2 (DEC-32)
demoted it, which made the claim smaller and true. **Revision 3.3 (DEC-49) builds it**, which makes
the claim large again — **but this time behind E1's and E2's gates rather than behind a status
field.** The difference between revision 1's position and revision 3.3's is not the verdict; it is
that one was a word in a config file and the other is three rows with failable gates.

**So the requirement is met in two stages and both are dated.** Before E2, the overarching context
is **within-request only** and this document says so. After E2 passes, it is the store — and if E2
fails, the store is reverted and this table's right-hand column reverts with it, which is what
makes the two columns worth printing separately rather than merging them into one aspiration.

The store's contract comes from `memory-gate` where those repos specify it and from §8's gaps
block where they do not (§1.3, §9.9). It is the first place the architecture acquires cross-request
state, and therefore the first place it acquires a trust boundary that per-request bounds do not
cover — which is why §9.9 B2's findings are E1's acceptance gates rather than its caveats.

**Recursive / looped latent refinement remains deferred**, per the operator: the `n_iter`
workspace loop is bounded and explicitly not a recurrent-depth architecture. The seam exists
(`halt` and `max_iters`) if that direction is ever taken up.

## 6.9 The post-binary track: ternary, and where it sits

Recorded here as the scale path's continuation, with **no rows and no design commitments today**
`[OP: csd-ternary-long-term-track]`. The operator's ordering is **binary toy → binary big model →
ternary toy → ternary big model**, and *"ternary work does not start until the binary big model
is proven"* — so it sits after everything in §4.1, including P5′. The arithmetic is why it is
worth recording rather than forgetting: this document's whole scale path is derived from **3.2675
measured effective bits/param** (N1, §6.1), which puts 30B at ~11.4 GiB and leaves §6.1's budget
closing with no margin at bf16. Ternary weights (`{-1, 0, +1}`, ~1.58 bits/param
information-theoretically, cf. BitNet b1.58) would roughly halve that again, which is what would
put a 30B-class composed mind on a 16 GiB card **with** activation headroom rather than at the
edge of it. Whether quality holds is the open question the ternary toy exists to answer —
measured, against untrained baselines, not assumed.

Three constraints that bind any future ternary work and are cheap to record now:

- **Python first, Rust only after the Python implementation proves the idea.** This is the
  standing language policy, not a preference about this track.
- **`tzervas/embeddenator` is substrate, not a model repo.** It is filesystem and low-level
  primitives intended to accelerate VSA, embeddings and ternary arithmetic **on binary
  hardware**, and it holds a record of what worked and what definitely did not. Read it before
  designing anything ternary so its negative results are not rediscovered; **do not expect it to
  supply region or interconnect designs.**
- **VSA binding/bundling as an interconnect primitive is a hypothesis for the ternary phase, not
  a current design input.** It is named here so that it is not smuggled into v1 as an
  architecture choice.

**Two further deferred directions belong beside it, both downstream of DEC-47 and neither built.**

- **Recurrent latent depth.** DEC-47's clause 2 says the scheduler's iteration depth **is** the
  latent-reasoning loop; the deferred form of that idea is to **loop the forward pass in latent
  space to refine a prediction** rather than only to re-admit regions
  `[OP: csd-recursive-latent-transformers]`. v1 already has the shape — `max_iters`, ACT halting,
  write-back — so the future step is a *training* change (make depth an inference-time knob the
  objective rewards), not an architecture change. Recorded so the v1 iteration loop is understood
  as the first version of it rather than as plumbing.
- **Quantised concept codes as a tract codec.** DEC-47's clause 5: the operator's *"discrete
  latent concepts"* is a hypothesis about representation, and its testable form is a **VQ-style
  codebook in the `tract_codec` slot DEC-08 retired `stream_vae` into**, which lands naturally on
  this track because a small codebook index is close to a ternary payload. **Continuous latents
  remain the default until a codec is measured against them**, and a codec that is adopted must
  still satisfy DEC-47's gate: a code index that a region *decodes back to latents* at the far
  end of the tract is a codec; one that a region *reasons over as a symbol* is the token
  bottleneck the invariant forbids.

---

## 6.10 Two long-term tracks — differential activations, and predictive hybrid training

Both sit **post phase 3**, beside §6.9's ternary track. Neither is built in phase 2, and both are
written as rows with gates rather than as intentions, because an idea held as a paragraph gets
argued about and an idea held as a gate gets settled.

### DEC-71 — differential activations: activation deltas in a fast recompute-and-apply format

DEC-33's overlays carry **weight** deltas. The operator is thinking past that: *"long term ... ALSO
activations stored in a DIFFERENTIAL format optimised for rapid recompute, so they can be applied
performantly and efficiently to the base values for the features and functionality they provide"*
`[OP: csd-memory-gate-overlays.md]`. So an overlay may additionally carry **activation deltas
relative to the base activations** — a delta plus a cheap reconstruction rather than dense stored
activations — letting a persona or skill contribute its features at inference **without holding
dense state on the card**.

**It meets three existing decisions.** It composes with quantised frozen regions (P5′q) and with
per-region selective activation, like weight overlays do (DEC-33). It is precisely the claimant
class **DEC-70's arbiter evicts first**, so cost per feature is what decides whether it earns
residency at all. And its prediction is the second of DEC-72's three targets, which is why the two
tracks are written together.

**Gate, and it is a COMPARISON rather than a demonstration** `[P5′d]`: **apply latency per feature**
and **VRAM held per feature**, each measured against a **dense-activation control** on the same
items and the same card, with the reconstruction cost counted inside the latency number. **Adopted
only if it wins on both.** A format that is smaller but slower to apply is a different trade and is
recorded as one rather than adopted — *"measure apply latency and VRAM per feature vs the dense
alternative before adopting"* is the instruction, not *"build it and see"*.

### DEC-72 — predictive hybrid training: extend the predictor, and gate it on wall-clock

The operator's sequencing is explicit and it is late: **after all this is dialed and working and
we have moved on to phase 3 model training, the predictive training tool is what gets dug at**
*(the source note prints this as a blockquote it explicitly labels "paraphrased faithfully", so it
is rendered here as the position it is rather than as quoted wording)*
`[OP: csd-predictive-hybrid-training-track.md]`. **The tool exists.** `tzervas/tritter` carries
`PredictionConfig` / `GradientPredictor` / `PredictiveTrainer`, a `LossPredictor`, an
`EmbeddingPredictionLoss`, and a 962-line research report on predictive hybrid training (training
dynamics on a low-dimensional manifold, three predictability regimes, Jacobian approximation for
backward prediction, a trait/state-machine spec for a Rust trainer), plus planning documents
claiming **25%+ backward and 15%+ forward** reduction. **Those claims are inputs to this row, not
results of it.**

**Three prediction targets, each with its own accuracy gate against the real computation:**
**(1)** weight updates — the existing predictor; **(2)** **activations and activation deltas**,
which is where this meets DEC-71; **(3)** **semantic residuals used as a correction signal**. The
hybrid paradigm alternates predicted phases with real backprop phases.

**Four gates, pre-registered, and the third is the one that decides it** `[P5′p]`:

1. **Prediction error against a REAL step** on held-out batches, per target, under a tolerance
   stated before the run.
2. **The fraction of steps predicted, printed** in the receipt — a method that predicts 2% of steps
   perfectly has not sped anything up, and a fraction reported only in aggregate hides that.
3. **The end-task metric against a fully-real run at EQUAL WALL-CLOCK and at equal steps, BOTH
   reported.** Equal-steps alone flatters a method whose steps are cheaper; equal-wall-clock alone
   flatters one whose steps are worse. The pair is the measurement.
4. **AUTOMATIC FALLBACK:** a predicted phase that drifts past tolerance falls back to real steps by
   itself and says so — **verified by constructing the drift**, per this document's standing rule
   that a guard nobody has made fail is not a guard. It fails closed.

**Fine-tuning is measured as its own case**, against its own real-run control, because the
efficiency argument is different there. **The tool stays in tritter under DEC-75**; CSD carries an
adapter.


# 7. What this invalidates in the other design docs

Precise enough to dispatch one agent per document. Line ranges are the current headings in each
file as read in this session [V].

## 7.1 `docs/design/CORPUS-CONTRACT.md` (1,332 lines) — **heaviest rework**

| section | what changes | why |
|---|---|---|
| §1.1 `code` (72–225) | re-key to `language` (faculty `language`, specialisation `code`). **Its "Target composition" REVERSES on `apps`/`code_contests`** — those go to `compose`, not to `code`. **Target corrected from `language_code` to `language` in revision 3.6 (DEC-78)** | DEC-01, DEC-23 |
| §1.2 `compress` (226–329) **+** §1.3 `retrieve` (330–511) | **MERGE into one `memory` section.** Two "target composition" tables become one; the merged faculty's requirement is *encode + consolidate + retrieve*, not two bi-encoders' corpora concatenated | DEC-02 |
| §1.3's B3 declaration defect (418–511) | **closed** by adopting the BEIR FiQA-pool eval as `memory`'s gate; the in-mixture number is reported alongside for comparability | DEC-09 |
| §1.4 `vl_latent` (512–619) | rename `visual`; add the **emission contract** (patch tokens into white matter) and promote composite images from polish to a **prerequisite** of the `visual × *` bins | DEC-03, §5.4 |
| §1.5 `classify` (620–679) | becomes **"Probes, not regions"**; add the `affect` faculty placeholder and its corpus requirement | DEC-05, DEC-06 |
| §1.6 `reason` (680–743) | rename `reasoning`; record the phase-3 objective change (latent-step prediction) and the deferred `numeric` split | DEC-04 |
| §2.1(a) cross-region items (756–776) | the four pairs are **re-keyed** (`compress × retrieve` is now intra-region); the claim *"they cannot exist"* is upgraded with the three constructions in §5.5 | §5.4, §5.5 |
| §2.1(b) general / out-of-scope (777–795) | now **sourced by construction** (shuffled cross-modal mismatch); `residual_mlp`'s retirement converts it from a training to an eval-only requirement | DEC-07, §5.5(c) |
| §2.1(c) interleaved batches (796–804) | re-homed to **phase 3 (P5′)**, where it finally has a reason to exist | §4.1 |
| §2.4 the allocation ledger (938–1002) | add the `compose` allocation for apps + code_contests; add the **aqua_rat total write-off** and the reason (deleted receipt ⇒ unrecoverable sampling) | DEC-23, §5.1 |
| §2.5 the reservation (1003–1048) | **rewrite.** 244,761 and 23.0% are historical; re-derive against **83,328**; bins 7→6; pairs re-keyed; the reservation is a **construction**, not a fetch. Its preferred gsm8k/banking77 split is dead — both are spent | §5.1, §5.4 |
| §2.6 the gate ladder (1049–1090) | replace: the ceiling becomes **oracle late fusion (B2)**, and G3′ + the content-swap control + the B0 void condition are added. The current ladder's upper bound is beatable by a good mixer | §2.7 |
| §2.7 what has no source (1091–1120) | the general bin **now has a construction**; audio is added as the one genuinely-blocked placeholder | §5.5 |
| Part 3 B1/B2 (1136–1178) | extend the provenance-group definition to **constructed** corpora: the generator (renderer, join, synthesis model) is the provenance group, and B1/B2 apply to it | §5.5 |
| Part 3 B3 | the *held-out-domain vs in-mixture* rule now also governs the reserve, whose whole point is that it is neither | §5.3 |
| §2.4 the allocation ledger — **again** | **replace the aqua_rat total write-off with the UNION burn (13,946 rows, three draws — R, P, and the discovered on-disk sample D3) plus 83,521 clean** (DEC-42), once W2a's **four** checks pass; record **every discovered draw's** fingerprints, the corpus fingerprint of the parquet, whether the run's code revision could be established, and the fact that the *sampling* is what made recovery possible `[N1 fixed]` | §5.1, W2a |
| §2.4 / new §2.4b **the reserve ledger's schema** | `RESERVED.jsonl` rows carry the **union of source-row `pair_fingerprint`s**, the generator name, the SHA-256 of the generator source and the config hash — **not** composite content hashes; recomputed on load, mismatch **aborts** | DEC-38 |
| new §2.8 **split assignment** | `split = HMAC(k_split, pair_fingerprint(source_row)) mod N`; `k_split` outside every agent-writable path; **no key ⇒ refuse to build a split** | DEC-39 |
| Part 1 every source row | **pin the source revision SHA** and refuse an unpinned fetch — mirror metadata is measured to lie, and the same channel carries content | §5.6 |
| Part 3 B1/B2 | state that for a **constructed** reserve the generator is the provenance group, and either diversify or carry a **dated waiver naming what the reserve cannot measure** | §5.4 |

## 7.2 `docs/design/TRAINING-SUPERSET.md` (1,422 lines)

| section | what changes |
|---|---|
| §1 classification tables (108–265) | re-key the four trained regions to faculties; note that `visual`'s corpus is **unreleasable as trained**, so no release claim can include it until replaced |
| §2 "The phases are the project's own curriculum" (268–311) | replace with the **four phases** (per-region → interconnect → whole-mind dynamic → foundation); phase 3 is new and is where growth happens |
| §2 per-submodel composition (312–331) | re-key to faculties; `memory` is one submodel with two heads |
| §2 **"The reserve — constituent-level, not row-level"** (332–385) | **substantially rewritten.** The reserve is the interconnect's **training data**, not phase 3's eval set. Constituent-level reservation cannot express a *constructed* corpus, so an **item-level ledger** with per-item NSRS scores and a generator id is required |
| §2 "Publishing a reserve you cannot ship" (386–403) | changes shape: most of the reserve is now constructed from already-cleared sources plus a renderer, which alters what is publishable and how |
| §2 "Enforcing and verifying the separation" (404–465) | add the **refuses-to-start** gate and the make-the-guard-fail test |
| §5 provenance minimums (815–929) | add: **generator id, revision and licence** for constructed items; the renderer's own licence; the join's source ids; the NSRS scores `s_r` per item |
| §8 the `kind: corpus` manifest (1200–1337) | add a `constructed` provenance kind and the fields above |
| §9 build order (1338–1359) | **re-order:** reserve allocation precedes P2.3; corpus construction runs on CPU hosts **in parallel** with GPU work, never serialised behind it |

## 7.3 `docs/design/MODEL-MANIFESTS.md` (1,209 lines)

| section | what changes |
|---|---|
| §B required fields (130–264) | add `faculty`, `specialisation`, `emits`, `native_dim`, `adapter`, `token_budget`, `kv_bytes_per_token`, `accepts_condition`, `status` (replacing `live`) |
| the `objective_family` enum (in §The schema, 534–848) | add **`interconnect-integration`** and **`scheduler-distillation`**. **The "live schema violation" claim is withdrawn** `[A22 fixed]`: re-read this session, `config/mind/csd-regions.json:144,162` set **`objective`**, not `objective_family`; `MODEL-MANIFESTS.md:742` constrains `objective_family` with the enum while `:743` types `objective` as a free-form string; and `csd-regions.json` is not validated against that schema at all [V]. **No violation exists.** Whether `classification-single-label` / `classification-multi-label` belong in `objective_family` is a **design question** — raised separately, and complicated by DEC-05/DEC-06 demoting both classifiers to probes |
| §The schema (534–848) | add a **third manifest kind for the interconnect**: it is neither a region nor a corpus, it has no pretrain corpus of its own, and its measured fields are budgets and schedule statistics |
| receipt stages | `pipeline/receipt.py`'s `STAGES` contains `"compose"` and nothing writes it [V\*]; add **`"schedule"`** and make both real |
| worked example 3 `csd/retrieve` (1032–1153) | becomes `csd/memory`, two heads, BEIR gate |
| worked example 3b the VL region (1154–1196) | becomes `csd/visual`; record that the **deployed half is the context encoder (10,712,448)**, not the full 22,905,216 |
| §E adversarial pass (499–533) | add **four** boundaries: the emitted `Schedule` is untrusted data validated by the runtime; **checkpoint loading** goes through one `load_checkpoint()` with `weights_only=True` hardcoded and a per-file SHA-256 verified *before* the file is opened (DEC-40); **`episodic_store` write→read across requests**, with a server-derived scope key and the byte-capacity, scored-eviction and cross-request-fuzz gates that DEC-49 makes acceptance conditions of rows E0/E1/E2 (DEC-49, superseding DEC-32); and **reserve construction → controller training**, the training-time path §9.9 did not cover (DEC-39, S5) |
| **NEW manifest kind: `overlay`** (DEC-33) | a memory-gate overlay is a first-class artifact, not a checkpoint variant. Required fields: **base checkpoint fingerprint it targets** (attach refuses on mismatch), overlay shape and rank, size in bytes at each residency tier, the data that trained it, its **licence under the strictest-input rule**, and the tiered-residency policy's measured hit rate / page-in latency / VRAM held. Its gate is P5′o's: **applied then disconnected reproduces the base receipt exactly, and the disconnect is logged** |

## 7.4 `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md` (1,987 lines)

| section | what changes |
|---|---|
| §Per-corpus verdicts in use (112–447) | re-key to faculties. **`compress` + `retrieve` verdicts merge into `memory`'s obligation set, which is the union of both and therefore the stricter** — this is a real licence consequence of a taxonomy decision |
| §The BLOCKING list (488–657) | unchanged in substance; but item 4 (`vl_latent` unreleasable as trained) now blocks **the visual faculty's participation in any released composed mind**, not one region's release |
| §Training data attribution (966–1027) | add constructed-corpus attributions: a **rendering of cleared text inherits that text's terms**; the renderer's own licence; the generating model for any synthesis |
| §The composed model (1706–1732) | the composed model's obligation set is now the union over the **reserve** as well, and the reserve is constructed — state the inheritance rule explicitly |
| §Release licence scenarios / comparison table (1520–1602) | re-key per-region rows to faculties; **add a row for the interconnect**, which has no corpus of its own except the reserve |
| §Replacement vision corpora (1213–1518) | unchanged as analysis, but it is now on the critical path for any release claim, not a future item |
| §Decision 2026-09-02 (1896+) | unchanged; annotate that the reserve's licence is now load-bearing for the composed release |
| §Per-region tier table, and the mirror-mismatch tally | **new in revision 3.1, DEFERRED with the rows in 3.2 (DEC-48) but still written down:** add the `auditory` and `speech_output` rows from `docs/design/AUDIO-CORPUS-AUDIT.md` (DEC-45), record that **ND is outside the NC-tolerant policy's scope** rather than a stricter NC, add the **consent-hygiene** class the vocabulary lacks, and move the mirror-vs-upstream tally from **ten to seventeen** |

## 7.5 `docs/design/MODERN-TRAINING-STACK.md` (878 lines)

| section | what changes |
|---|---|
| §1.4 model size (171–209) | the "~87M" figure is **both an over-count at deployment** (10,712,448 of `vl_latent` ships, not 22,905,216) **and an under-count** (it omits the receipt-less regions). Restate with the deployable figure: **86,331,303 params, 345 MB fp32, 35.3 MB at 3.2675 bits** for the five-participant v1 list (§2.3, corrected in revision 2, restated in 3.3 for DEC-49) |
| §2.11 factorized / pruned embedding — *ADOPT LATER* (663–695) | **verdict changes to a decided architectural item**: one shared token embedding across text faculties, 3.02× at toy scale, mandatory from the first region trained after ratification (DEC-24) |
| §2.2 gradient accumulation — *REJECT* (365–406) | **re-scoped**, not reversed: rejected for single-host InfoNCE; **required** for phase-3 cross-host pipeline stages, where microbatch must be ≤ 256 (DEC-29) |
| §2.7 LoRA / adapters — *ADOPT LATER* (526–566) | the justification arrives and is a systems one: an adapter is 17 MB and **0.85 ms** to page against **61 ms** for a whole region. Adapters are how specialisations scale; the region schema now has an `adapter` field |
| §2.5 EMA — *ADOPT* (454–481) | **the deployment consequence, with the halves the right way round** `[A3 fixed]`: the two encoders are each **46.77% of `vl_latent`**, and it is the **context encoder that does not ship**. `IJEPA.encode` is `target_encoder.embed` (`model/vl_jepa.py:387` [V]) and every probe and transfer number `vl_latent` has comes through it (`regions/vl_pretrain.py:202` [V]), so the target encoder is the deployed half (DEC-34). Add the reason EMA is not merely scaffolding here: `ema_base 0.996 → ema_final 1.0` means the target's momentum rises toward 1.0, so the two halves are **furthest apart at the end of training** — which is exactly when a "just use the other one" substitution would be made |
| §2.8 QAT vs PTQ (567–600) | add **`D_sched`** as a second PTQ objective; the cheap prerequisite experiment changes shape (DEC-27) |
| §3 recommended sequence (738–773) **+** §4 experiments (774–859) | insert the interconnect's four training phases; note that the batch-size experiment now interacts with the phase-3 microbatch cap |

## 7.6 Also affected, outside the five named docs

- **`docs/design/VISUAL-INGEST-PIPELINE.md`** — composite / screenshot images move from polish to
  a **prerequisite** of the `visual × *` cross-faculty bins, and the deterministic layout
  renderer of §5.5(b) is now a first-class deliverable of that pipeline.
- **`config/mind/csd-regions.json`** — §1.4's diff.
- **`program/csd-program.json`** and **`program/REMAINING.md`** — P4/P5/P6 replaced by §4.1.
- **`src/cogsyndelta/contracts/region.py`** and **`region_spec.py`** — the protocol and the
  schema of §2.2 and §1.4; `MindSpec`'s single-`stream_dim` check is replaced.
- **Six tracked files** citing the retired `docs/program/CSD-BRAIN-REGIONS.md` — re-point per
  §1.5.

---

# 8. Operator decisions that remain **after this revision**

Only what this document cannot settle. Everything revision 1 listed here that has since been
decided is **removed**, not carried forward — a stale decision list is worse than none, because
it makes the live items harder to find. Each item below has a recommendation.

**Removed from revision 1's list, because they are settled:**

| was | disposition |
|---|---|
| **OD-1** release licence / GooAQ contradiction | **RESOLVED by the operator** `[OP: csd-release-licence-decision]`. The NC reading is accepted, the corpus is kept, and the licence moves rather than the data. **DEC-31** records the strictest-input rule; §5.7 applies it; `memory` is NC and therefore so is the composed mind. The "email AI2" action leaves the blocking list. |
| **OD-6** disposition of the untracked working files | folded into **OD-7** below, because the set has changed and one of the new files is a security control. |
| **OD-9** the audio licence audit | **ANSWERED and it stays answered.** The audit was done and is committed at `docs/design/AUDIO-CORPUS-AUDIT.md`; DEC-48 then deferred the rows it produced, which changes *when* the work is spent and not *whether the question is open*. **Its entry below is kept as a record of what the answer was and is not an open ask**; its six descendants, OD-10 to OD-15, are the live residue and they are deferred with the audio rows. |
| **DEC-32's build trigger** for `episodic_store` | **REMOVED by DEC-49**, and it is listed here because it was a decision this section would otherwise still be waiting on. Revision 2 made the store's build conditional on *"`salience` is trained AND a measured failure mode requires cross-request context"* — an operator decision in the shape of a technical precondition. **The operator decided it directly instead** `[OP: csd-episodic-store-required.md]`, so the trigger is gone and rows E0/E1/E2 replace it. **`salience` is NOT a prerequisite of the store**, and that is a real consequence of the ruling rather than an oversight: the store's write policy at v1 is the workspace's own write-back path, not a learned salience score. A **learned** write policy is a phase-3 candidate alongside gap (d)'s consolidation question, not a phase-2 blocker. |
| **OD-1** autodev write scope contains the guards | **ANSWERED, and its recommendation is HALF-CORRECTED.** The operator answered the scope question by identity and privilege rather than path lists — `svc-autodev`, own vault, minimum-scope Forgejo user, no root, no sudo, gateway-only egress (**DEC-60**) `[OP: autodev-service-identity-and-sandbox.md]`. **Its "path-scoped Forgejo protection" half is CONTRADICTED BY MEASUREMENT and replaced by DEC-61:** `protected_file_patterns` had to be cleared, because Forgejo refuses to merge **any** PR touching a protected file (405, admins included), which blocks the reviewed path the control existed to protect. **Guard integrity now rests on required checks — `tests/test_guards_can_fail.py` included — plus review.** Its entry below is kept as the record of what was recommended and why it changed; the `http_apply`-does-not-replicate-`apply_proposal` half is folded into **PRE-1** |
| **OD-2** lab-console proxy forwards the apply token | **ANSWERED and SUPERSEDED BY A WORSE MEASURED FACT.** The operator's answer is the default-deny gateway that mints short-lived downstream tokens and never relays client credentials (**DEC-60**). The measured defect is not credential forwarding but that **`dispatch_api_get()` authenticates nobody at all** — so the fix is larger than OD-2 scoped and becomes row **PRE-1**. **The forwarding OD-2 reported is also itself fixed in the tree** (`proxy_upstream()` never reads the caller's header), which the threat pass records as **G23**. Its entry below is kept as the record of what was found first **and now carries a `SUPERSEDED` marker at its own location**, because a disposition recorded only here is one a reader of that paragraph never sees |

---

## Episodic store contract gaps — OPERATOR DELIBERATION REQUIRED `[DEC-49]`

**Why this block exists and why it is not six more OD items.** DEC-49 takes the store's contract
from the operator's own repositories wherever those repositories specify it, each clause cited at a
`file:line` in §1.3. **Six things CSD needs are not in them.** The rule this document applies to a
doc that a probe contradicts applies here too, one level up: **where the repos are silent, nothing
is inferred into a contract clause.** Each gap below states **what the repos say**, **what CSD
needs**, and **a recommended default** that is implementable if the operator does not want to spend
a decision on it. They are grouped rather than numbered into the OD list because they are one
decision surface — the store's — and splitting them would let a reader ratify four and leave the
other two silently open.

**A property worth stating before the list, corrected from an earlier draft that claimed five of
six become measurements: of the six, four are genuine operator decisions, one is a confirmation,
and one is a notification of something already closed** `[S33-8 fixed]`. Gaps (a), (b), (c) and
(d) are the four decisions, each with a recommended default; (a), (c) and (d) become
measurements — what actually got built, against what was only proposed — the moment E1 runs,
while (b), the partition axis, is a pure policy question about who is isolated from whom that no
measurement decides. **Gap (e) does not become a measurement**: its own recommendation is *do not
pull it forward*, so the confirmation it asks for (that overlays stay in P5′o) is answered by
**not** building in E1, not by building. **Gap (f) is not an open decision**: it is *closed by
DEC-49* already, and is listed here only so that closure stays visible beside the gaps that are
not. If only one of the four decisions is read closely, read (b).

**REVISION 3.4 CLOSES THIS BLOCK RATHER THAN CARRYING IT FORWARD** `[OP: csd-episodic-store-required.md]`.
The operator's position is that the gaps be **identified, deliberated, and the desired contracts
nailed down** — not left open across revisions. So every recommended default below is now a
**decision**, and the block is retained as the *argument* for each one rather than as a list of
things still to settle: gap (a) ⇒ **DEC-63**, gap (b) ⇒ **DEC-64**, gap (c) ⇒ **DEC-65**, gaps (d)
and (e) ⇒ **DEC-66**, gap (f) ⇒ already closed by DEC-49. **Two things survive as operator
confirmations, and only two, because they are the ones no measurement decides:** DEC-63's
`safety_margin` value **and** the residual-claim-versus-fixed-floor question (a product decision
that trades context length for recall), and **DEC-64's choice of `scope` axis** — principal,
session, or persona-basin — which is a decision about what the mind remembers about whom. Both are
marked **"operator to confirm"** in their decision rows and both ship with an implementable
default, so **E1 is not blocked on either**; a confirmation that changes the default changes a
config constant and a scope-derivation function, not the store's shape. What this conversion buys
is that a reader of §4.1's E1 row can now find the contract it builds against in the decision
index instead of inferring it from a deliberation block.

---

**GAP (a) — DYNAMIC CAPACITY: what number bounds the store's bytes.**
**What the repos say:** every capacity in both repos is an **item count**, never bytes and never
VRAM — `hot_cap` 256 [V, `storage/tiered.py:40-51`], `TierBudget{ram_max_items 256,
disk_max_items 4096}` [V, `types.rs:458-473`], `max_traces` 100,000
[V, `holographic_store.rs:33-58`]. **The only VRAM arithmetic in either repo budgets model
WEIGHTS**, is never wired to any store cap, and is llama.cpp launch-flag tooling:
`weight_budget_mib(vram_total, reserve_gpu_kv) = vram_total − 2048 (display) − 1024 (cuda scratch)
− [2048 (gpu kv)]`, `saturating_sub` to floor at zero, test-pinned at 16,303 MiB ⇒ 11,183 / 13,231
MiB [V, `facade/hypha.rs:114-122`, constants `:13-26`, test `:152-157`]. **The operator's stated
intent is not implemented, not described, and not even a TODO in either repo** — the subtractive-
reserves *shape* is all that exists to borrow.
**What CSD needs:** the operator's rule, verbatim `[OP: csd-episodic-store-required.md]`:
*"Capacity is gonna be dynamic based on the GPU that it's running on and how much context is
allocated for — like KV cache determining essentially how much free space there is that can be
allocated to memory-gate functionality."*
**Recommended default — adopt the subtractive shape, repointed from weights to the store:**

```
capacity_bytes(host, tick) = max(0, VRAM_total(host)
                                  − KV_reserved(context_len, regions_active)   # from the scheduler
                                  − activation_reserve                          # forward working set
                                  − safety_margin)
```

computed as a **pure function of live inputs, per host, at scheduler-tick time and never cached** —
because the 1080 Ti is preemptible and the active-region set changes under the store, so a capacity
fixed at process start is not dynamic. **`safety_margin` is the one free parameter and it is the
operator's**; the recommendation is to start at the same order as `hypha`'s display reserve (2 GiB)
and let E1's probe report what is left on each card. **What the operator is actually being asked:**
whether a residual claim is acceptable at all, given §9.11's finding that the residual may round to
zero on the 5080. The alternative is a **floor** — reserve a fixed slice for the store before the
KV budget is computed — which trades context length for recall, and that is a product decision this
document should not make.

**GAP (a), EXTENDED — the staleness decay function itself, not only the byte unit it evicts to**
`[S33-4 fixed]`. **What the repos say:** `access_count` and `last_accessed` are recorded on every
span and read by **nothing** in either eviction path [V, `holographic_store.rs:113-116, 321-338`]
— the scout classifies "these fields indicate unfinished intent" as **INFERRED**, not VERIFIED
(`memory-gate-contracts.md:245-250`): *"plausible... but not stated anywhere."* §1.3's `[I]` tag
on `staleness_penalty(last_accessed)` is this classification, carried through rather than quietly
upgraded to a decision. **What CSD needs:** E1's build gate (§4.1) requires the scored eviction
term as a hard deliverable regardless, so the open question is not *whether* to decay but **by
what function** — linear, exponential half-life, or a step at a fixed age — and over what
argument (wall-clock age, or `access_count` as a frequency term instead of or alongside recency).
**Recommended default:** exponential half-life on `last_accessed`, half-life a config constant
starting at the same order as the workspace's own context window in wall-clock time, revisited
once E1's probe shows real eviction pressure. **What the operator is actually being asked:**
nothing yet — E1 ships a working default and the receipt records which function ran; this gap is
closed by measurement the same way (a), (c), (d) and (f) are, per the framing above, and is listed
under (a) rather than as a seventh gap because it shares gap (a)'s unit (bytes evicted) and its
resolution mechanism (E1's receipt).

---

**GAP (b) — THE PARTITION AXIS BEYOND DOMAIN. Read this one.**
**What the repos say:** **domain is the only implemented partition axis**, and it is part of
*identity*, not a filter [V, `memory_protocols.py:123-125`; ADR-0001 `:129-133`]. It is a
**fleet-defined task taxonomy** — a closed enum of `Infrastructure, CodeReview, Deployment,
IncidentResponse, General` plus M1 additions [V, `types.rs:165-208`] — **not a per-user or
per-session id space. There is no `user_id`, `session_id` or `principal` field anywhere** in
`MemoryRecord`, `LearningContext`, or the store protocols [V-abs] `[S33-9 fixed]`. **Persona/basin is specified and has
zero code**: ADR-0001 and the spec define `Persona --instantiates--> Basin --selects--> Domains`,
with the explicit rule *"Basin ≠ collection; domain ≠ collection"* and the non-goal that
*"Personas MUST select a memory basin/configuration; they MUST NOT be modeled as MoE experts or
independent agents"* [V, `adr/0001:72-85`; spec FR-006/FR-007]. It is task `P1-13`, status **Not
Started**, **no `Persona` or `Basin` class exists in either `src/` tree** [V-abs, grep, zero hits
outside docs].
**What CSD needs:** an isolation boundary. §9.9 B2's whole defence is a **server-derived scope**,
and `domain` cannot be it — a task taxonomy is not an isolation boundary, and using one as the
other is how a "partition" becomes a label. CSD also has a reason memory-gate does not: its regions
are **brain faculties, not task domains**, so `domain` does not even name the right kind of thing.
**Recommended default:** `(scope, domain, logical_key)`, three segments, with **`scope` = the
authenticated principal, derived server-side** and `domain` retained as memory-gate's task axis for
compatibility with the ported tests. **Keep persona/basin as a SCOPE SELECTOR and never as a
routing object** — that non-goal is the one piece of the persona design that was fully thought
through, and it is worth inheriting even though nothing implements it.
**What the operator is actually being asked:** whether `scope` is **principal**, **session**, or
**persona-basin** — three different products. Principal isolates users from each other and lets one
user's episodes accumulate forever. Session isolates episodes and forgets between them, which makes
X7 items work and long-horizon memory impossible. Persona-basin isolates *contexts* within one
user. **The recommendation is `principal`, with `session` as a sub-segment the request may name and
the server may bound** — but this is a decision about what the mind remembers about whom, and it is
not a technical default.

---

**GAP (c) — WHAT IS STORED: text-plus-embedding, or workspace latents.**
**What the repos say:** the unit of storage is **text**, with an optional embedding that must carry
a binding (`model_id`, `dimension`, `metric`) [V, `memory_protocols.py:91-121, 61-73`]. **Neither
repo stores KV-cache bytes or model latents.** `Residency::{Gpu,Ram,Disk}` is documented as
*"Where a recalled span currently lives. GPU bytes stay in llama.cpp"* — a metadata tag over spans
owned elsewhere, and `mark_gpu(key)` moves nothing [V, `types.rs:414-436`; `storage/tiered.rs:1-8`;
`facade/hypha.rs:1-8`; `tiered.rs:99-106`]. **So there is no precedent in either repo for a store
that owns tensors.**
**What CSD needs:** DEC-47 forbids re-serialising to discrete tokens on an inter-region path, so a
text-plus-embedding store is **not available** to CSD as an inter-region participant — it would put
a token bottleneck exactly where the invariant says there must not be one. CSD's store holds
**workspace latents**: the read is `k = W_k z`, `v = W_v z` over stored latents `z`.
**Recommended default — take memory-gate's index/policy split and reject its payload:** the store
owns an **index and an eviction policy** over latent bytes **owned by the runtime**, not a second
allocator competing with the KV cache for the same VRAM. A record is `(scope, domain, key) →
(pointer, residency, importance, last_accessed, byte_size, provenance)`; the latents themselves live
where the runtime put them. **This is the one clause where CSD deliberately diverges from the source
repos on the payload while adopting their structure**, and it is called out rather than blended.
**What the operator is actually being asked:** whether a **text sidecar** is kept alongside the
latent — searchable, auditable, and the only way a human can ever inspect what the mind remembered.
**Recommendation: yes, as provenance metadata, never on the read path**, so it cannot become a
token bottleneck by accident.

---

**GAP (d) — CONSOLIDATION: prune-only today, or a real CLS.**
**What the repos say:** both languages implement consolidation as **prune-delete** — rows below
`low_importance_threshold` (0.2) **and** older than `age_threshold_days` (30) are deleted
[V, `gateway.rs:303-338`, defaults `types.rs:323-333`; Python `consolidation.py:103-104`]. **The
Python spec names its own implementation a DEFECT** — prune-only, *"success recorded at run
start"*, depending on undocumented store methods [V, spec.md `:51-62`, "Current defects (honest
baseline)"]. Fast/slow CLS with provenance-preserving new representations is **only described**;
`P1-11`/`P1-12` are **Not Started** [V]. `RecordProvenance` exists and is *required only for
consolidation outputs* [V, `memory_protocols.py:76-88`] — the field the real design would need,
already there and unused.
**What CSD needs:** at minimum, eviction that does not lose what the mind should keep. At most, the
thing the name promises: a slow pass that **creates new representations** from old episodes, which
is what makes a hippocampal analogy more than a naming exercise.
**Recommended default: PRUNE-ONLY AT v1, AND SAY SO IN THOSE WORDS.** Consolidation is scored
eviction plus the durability ladder, and nothing else. **Do not build a CLS in phase 2** — it is a
learned component with no gate, no receipt and no corpus, which is the `residual_mlp` defect in a
new costume. **Record it as the store's own phase-3 candidate**, where P5′'s monotone-improvement
rule can grade it. **What the operator is actually being asked:** to accept that CSD's store forgets
by policy rather than consolidating by learning, for the whole of v1.

---

**GAP (e) — DIFFERENTIAL OVERLAYS: no counterpart exists.**
**What the repos say:** **nothing.** Exhaustive grep for `overlay` and `differential` returns
**zero hits in both `src/` trees** [V-abs]. No offset, delta, diff, LoRA-style overlay, per-persona
weight delta or sparse update structure appears in either store [V-abs]. The nearest analogues are
`Residency` (index-not-bytes, gap (c)) and persona/basin (spec-only, gap (b)). A research-notes
document mentions AdaLoRA/DoRA as **external** techniques the author was reading about, not as
anything memory-gate implements [V, `docs/dmll_for_ai_frameworks.md:35`].
**What CSD needs:** DEC-33 already names memory-gate overlays as the dynamic-paging seam's first
client and gives them a row, **P5′o**. So this gap is not "should CSD have overlays" — that is
decided — it is that **the design for them cannot be imported and must be written fresh.**
**Recommended default:** design them **in P5′o**, not now, and inherit exactly two patterns from the
source repos rather than a specification: **index-not-bytes** for residency, and
**scope-selector-not-agent** for persona. **P5′o's existing gates already have the right shape** —
an overlay applied and disconnected reproduces the base receipt exactly, and an overlay refuses a
base fingerprint it was not trained against. **What the operator is actually being asked:** to
confirm that overlays stay in P5′o and are not pulled forward into E1 because the store is being
built anyway. **Recommendation: do not pull them forward.** E1 is a container with four failable
gates; adding an unspecified feature to it is how a row with gates becomes a row with intentions.

---

**GAP (f) — CROSS-REQUEST POISONING DEFENCE: partly present, and the gap is the fuzz.**
**What the repos say:** cross-domain isolation is real and **test-pinned**
[V, `tests/storage/test_store_conformance.py:150`], and a query with neither a domain nor an
explicit global flag **raises** [V, `memory_protocols.py:194-217`; `record_validation.py:83-97`].
**What is absent is the adversarial half:** there is no cross-request fuzz, no negative test that
constructs a leak, and no capacity-exhaustion test anywhere in either repo [V-abs] `[S33-9 fixed]`.
**What CSD needs:** threat **T2**. §9.9 B2's four bounds — partition, capacity bound, eviction
order, and the two-request fuzz.
**Recommended default: adopt them as E1's acceptance gates, which DEC-49 does**, and add the clause
the source repos' conformance suite is missing and that this programme's own discipline demands:
**every one of the four is verified by making it fail.** The mis-derived scope key must produce a
crossing; the constructed overflow must evict in the scored order; the zero-capacity case must be
reported rather than silently satisfied. **What the operator is actually being asked:** nothing —
this gap is closed by DEC-49 and is listed so the closure is visible beside the five that are not.

---

**What this block does NOT contain, and the omission is deliberate.** There is no gap for *"a
gateway-level `retrieve`"* (the source repos still have agents reaching into `store` directly, which
CSD does not inherit because its read path is cross-attention, not a call), and none for *"a batch
API"* (Rust has one, Python does not, and CSD's write path is one latent per turn). Both are real
holes in the source repos and neither is a decision CSD has to make. Listing them would pad an
operator decision list with someone else's backlog.

Three of the items below (**OD-1, OD-2, OD-3**) come from the threat pass and are about code and
fleet configuration **outside this document's tree**. They are recorded here because this design
newly *depends* on them, and because a design that quietly assumes a control it does not own is
how the assumption survives the control.

---

**OD-1 — The autodev write scope contains the guards that protect the reserve and the
interconnect.** `[T4 recorded]`
**VERIFIED:** `ALLOW_PREFIXES = ("src/", "tests/", "docs/")`, client-side at
`scripts/csd-autodev-loop:38` and re-enforced server-side at `scripts/csd-lab-console:453` [V].
The *enforcement* is sound. The **scope** is the problem: every control this design depends on
will live inside it — the NSRS filter, the reserve loader check, the `Schedule` validator, the
`D_sched` quant gate, `assert_no_contamination`, and **the tests that prove each one fires**. A
prompt-injected tick can weaken a guard and edit the test that would catch the weakening in one
change, and CI stays green because the test changed too. With merge authority, that lands.
*Recommendation:* a server-side **protected path set** in `http_apply` alongside the prefix check
it already performs — guard modules and their failure-mode tests are refused, human-only — paired
with a **path-scoped Forgejo protection** so the rule survives a new guard being added in a file
nobody thought to list. **Reducing the write surface is the control; reviewing the commits is
not.** Also: `apply_proposal` refuses test files with no assert (*"test has no assert; refusing
pass stubs"*, `csd-autodev-loop:145` [V]) and `http_apply` does **not** replicate it — anything
posting directly to `/api/apply` skips it. Move it server-side or stop counting it as a control.
*Why it is the operator's:* it trades autonomy for durability of the guards, and the autonomy is
the operator's to spend.

**AMENDED IN REVISION 3.4, AND ONE HALF OF THE RECOMMENDATION ABOVE IS NOW KNOWN TO BE WRONG**
`[DEC-61] [OP: branch-and-pr-to-main-only.md]`. **The "path-scoped Forgejo protection" clause was
tried and it does not work as described.** Forgejo's `protected_file_patterns` does not refuse a
*write* to a protected file; it refuses to **merge any pull request that touches one** — 405
*"Changed protected files"*, **admins included** — so setting it blocks precisely the
human-reviewed path the control was meant to preserve, while an unreviewed direct push (which the
same protection settings now forbid outright) was never what it caught. **It has been cleared, and
this document does not get to keep counting it.** What replaces it, and what guard integrity
actually rests on now:

- **`main` is PR-ONLY.** Push whitelist = the operator alone; **8 required `pull_request` status
  contexts** (the `(push)` variants never appear on feature-branch heads, because every workflow's
  push trigger is limited to `main`/`dev`, so requiring them made every PR unmergeable — a
  configuration that looked strict and was simply broken). This holds for the operator's own
  commits too, **on principle**.
- **Guard integrity = the required checks plus review.** `tests/test_guards_can_fail.py` is one of
  the required contexts, which is the durable half: a tick that weakens a guard **and** edits the
  test that would catch it still has to get a green run past a human on a PR. **That is a weaker
  control than a path list would have been if a path list worked, and saying so is the point** —
  OD-1's *"reducing the write surface is the control; reviewing the commits is not"* remains true
  as an argument and is currently unimplementable as stated.
- **Every mutating agent works in an ephemeral worktree and pushes a sha**; merges are
  fast-forward or `--no-ff` **from pushed shas**, never from an agent's working tree. This is the
  mechanism by which "reviewing the commits" is at least *possible* — there is always a pushed
  object to review.
- **`required_approvals` is still 0.** That is stated here rather than left implicit, because it
  is the difference between *"a human saw it"* and *"a human could have seen it"*, and it is an
  open question in the autodev spec, not a CSD decision.

**OD-2 — The lab-console upstream proxy forwards the apply token before authorising.**
`[T4b recorded]`

> **SUPERSEDED IN REVISION 3.4 — DO NOT READ THE PARAGRAPH BELOW AS THE CURRENT STATE OF THE
> CODE** `[DEC-60]`, see **PRE-1**. The body is kept as the record of what was found first, and
> it is marked **here, at its own location**, because the settled-items table saying so several
> hundred lines away is not a marker a reader of this paragraph ever sees. **Two things in it are
> now false against the tree it cites.** (1) The forwarding it describes **has been fixed**:
> `proxy_upstream()` builds an empty header dict and attaches only a server-minted
> `Authorization: Bearer {UPSTREAM_TOKEN}` — it never reads the caller's header — and
> `dispatch_api_post()` handles `/api/apply` locally before `UPSTREAM` is consulted, which its
> own docstring names *"the OD-2 fix"* [V, `scripts/csd-lab-console:981-1010, 1043-1053`].
> (2) **The cited line numbers have moved** and now point at code doing the opposite of what the
> paragraph says. What replaced this finding is **worse and is row PRE-1**: the gateway
> authenticates **nobody** on most routes, so there was never a caller credential to forward.

**VERIFIED AS OF THE ORIGINAL PASS, AND NO LONGER TRUE OF THE TREE** — `scripts/csd-lab-console:995-1001`
as those lines then stood: when `UPSTREAM` was set, a POST to any `/api/*` path was proxied
upstream **with the client `Authorization` header copied verbatim**, *before* the local
`check_apply_auth` at line 1003 ran [V, at the time of the finding]. **[I]** The local node
therefore performed no authorisation of its own on that path and acted as a credential-forwarding
proxy; the apply bearer reached whatever `UPSTREAM` named. *Recommendation:* proxy an **explicit default-deny path
allowlist that excludes `/api/apply`**, and **strip inbound credentials** rather than forwarding
them, minting a downstream-scoped token instead. *Why it is the operator's:* it changes how the
operator's own multi-host console works.

**OD-3 — `/data/models` is exported `rw,no_root_squash` to the entire `/24`.**
[V, `shai-stuff/docs/INVENTORY.md:133`]. This design puts the composed-mind artefacts **and the
receipts** on that share, and receipts are the programme's only evidence layer — deleting
evidence is cheaper than forging it and has the same effect, as the deleted `reason` receipt
already demonstrated. It is also the precondition that made the `torch.load` gap (DEC-40)
reachable: root on any LAN host is a writer of the receipt that names the checkpoint path.
*Recommendation:* read-only to everything except the one training host, `root_squash` everywhere
else, and writes through a service on the training host rather than through the filesystem. *The
export option is itself the enforcement*, which is why this is cheap and durable.

**OD-4 — When `visual`'s corpus is replaced, now that a `visual` retrain is happening anyway.**
`visual` is trained on tiny-imagenet, **unreleasable as trained** [V\*]; replacement candidates
are already audited in `LICENCE-FOR-OPEN-WEIGHTS.md §Replacement vision corpora`. **What changed
since revision 1:** W7v retrains `visual` regardless (token-aware objective + composite
resolution), so replacing the corpus **in the same run** costs the corpus fetch and nothing else,
where revision 1 costed it as a separate ~500 s retrain. *Recommendation:* **replace at W7v.**
The alternative — after W6 — saves nothing and makes every phase-2 number a statement about an
unreleasable mind. *Why it is the operator's:* it trades a known cost against a release strategy
only the operator owns.

**OD-5 — Whether write-back is enabled in phase 2, or deferred to phase 3.**
DEC-17 adds conditioning prefixes so inter-region conditioning exists at all in phase 2; the risk
is prefix tuning on a frozen encoder trained at `max_len 96`, out of distribution, spending 8% of
the length budget. **What changed since revision 1:** W5b's pre-committed fallback now has teeth
— under the no-write-back branch, either §2.4's `Ĉ` variant is built or the emitted `Schedule`
carries **no `edges` and no `lockstep_groups`** and the receipt says `topology: not demonstrated`
`[A20 fixed]`. So deferring is honest rather than merely cheaper. *Recommendation:* **enable it,
gated by W5b** — the fallback is now well-defined, so the downside is bounded and named.
*The operator's call is risk appetite.*

**OD-6 — Whether 30B-class is a v1 commitment or a direction.**
DEC-26 caps the interconnect at 2–5% of total, and revision 2 **scopes that cap to ≥1B
parameters** and prints v1's actual 31.8% beside it `[A25 fixed]`, so the cap no longer silently
contradicts §2.3. It also corrects the ceiling: at the design's own `mlp_ratio 4`, `L_ic = 8` is
**7.2%**, above the band, not at it `[A26 fixed]`. *Recommendation:* treat 30B as a **direction**
and keep the scoped cap — it costs nothing today (`L_ic = 4` is 3.6%) and it constrains exactly
one future choice, visibly.

**OD-7 — Disposition of the untracked working files.**
The set shrank while this revision was being written: `tests/test_checkpoint_load_security.py`
and the two-script `weights_only=True` fix are now **committed** (`c976e84` [V]), so the item
that was a security control has resolved itself. What remains untracked:
`src/cogsyndelta/regions/retrieve.py` and `tests/test_region_retrieve.py`, both written by
another agent [V]. *Recommendation:* **leave them untracked until W4 consumes them.** DEC-09
adopts the eval half and retires the FiQA-only training half, so whichever order is chosen the
file is committed *modified*; committing it first only creates a revert to explain. This document
does not touch that tree.

*Note for whoever ratifies this:* the tree moved under this revision twice — the DEC-23 ledger
line and the `weights_only` fix both landed mid-pass, and the threat pass's *"verified by
absence"* finding on the first was wrong by the time it was read (§11 R1). **Re-probe before
acting on any code-state claim in this document, including this one.**

**OD-8 — Where `k_split` lives.** DEC-39's keyed split assignment is the cheapest control in the
document and it is worth exactly as much as the key's location. It must be outside the repo and
outside every path the autodev loop can write (OD-1). *Recommendation:* the existing SOPS age
vault, read at train time by the loader, with **no environment-variable fallback** — because a
fallback is how "no key ⇒ refuse to build a split" quietly becomes "no key ⇒ use seed 0".

**AUDIO ASKS OD-10 TO OD-15 ARE DEFERRED WITH THEIR ROWS, AND NONE OF THEM IS DELETED**
`[DEC-48]`. The operator's later ruling — *"we can wait to add audio as a future feature once it
proves out without audio. that will be more of a production phase implementation"*
`[OP: csd-multimodal-io-intent.md]` — makes audio a **production-phase** feature, so the six asks
below stop being pre-A0 blockers and become **pre-A0f blockers in the production phase**. They
are kept in full, with their recommendations and their late costs intact, because the reason each
one is cheap *now* and expensive *after a fetch or a retrain* is unchanged by when the fetch
happens: **the deferral moves the deadline, not the arithmetic.** Two keep a residual bearing on
v1 and are marked where they appear: **OD-13** and **OD-15** are what A0f blocks on
`[S31-7 fixed]`, and OD-10, OD-11, OD-12 and OD-14 concern sources **outside** the recommended
mix and block nothing.

**OD-9 — Start the audio licence audit now, or defer it. ANSWERED (kept as a record, not an open ask): START NOW — and the answer
went further than the question.** The operator's input, verbatim
`[OP: csd-multimodal-io-intent.md]` — and **since deferred as a schedule by DEC-48**, which does
not change the answer to OD-9 itself:

> *"the intent for this model is for me to interact via voice and/or text input and text/speech
> output, and for it to take as input, audio, visual, discrete text/token and output multimodal
> as well."*

The recommendation became a **requirement**, and it changed two things this document had filed as
optional: `auditory` was made a required region (**DEC-43**) and a speech-output head a required
head (**DEC-44**), neither of which OD-9 was asking about — **and both were then deferred to a
production phase by DEC-48**, which changes when they are built and not whether the audit was
worth running. **The audit is done and committed at
`docs/design/AUDIO-CORPUS-AUDIT.md`**; what remains of OD-9 is not a decision but rows **A0m,
A0f, A1, A2, A3** with gates — **deferred to the production phase by DEC-48** (§4.1, *Production
phase: audio*) and no longer carrying a phase-2 bill (§4.3). **OD-9 leaves this list**, and the
audit it produced is banked groundwork that the deferral does not spend.

What it leaves behind is six decisions the audit surfaced and could not make. The pattern is worth
naming: **answering the cheapest question in the programme opened six more, every one of them cheap
now and expensive after a fetch or a retrain** — which is the same shape as W1, and the same reason
to answer them before A0 runs rather than after.

**OD-10 — TED-LIUM 3 is CC BY-NC-ND, and ND is outside the NC-tolerant policy's scope.**
VERIFIED at the upstream: *"The TED-LIUM corpus is licensed under Creative Commons BY-NC-ND
3.0... All talks and text are property of TED Conferences LLC."* This is **not** a GooAQ case.
DEC-31 absorbs a **NonCommercial** term by moving a release licence; a **NoDerivatives** term
forbids distributing the modified material at all, commercial or not, and no release-licence
choice answers it. The licensor here is internally consistent and simply forbids what
training-and-releasing would do — there is no contradiction to resolve by email.
*Recommendation:* **refuse TED-LIUM, and rule that ND is refused as a class, eval included** —
`LICENCE-FOR-OPEN-WEIGHTS.md`'s eval-only carve-out rests on a structural guarantee that no
published parameter derives from the eval set, and a broad reading of "derivative" reaches even
that. **Cost of deciding late:** trivial if decided now (452 hours against a mix measured in tens
of thousands, and nothing in the recommended mix depends on it); severe if decided after A1 — an
ND corpus inside a trained checkpoint is not removable by relicensing, only by retraining.
*Why it is the operator's:* it extends a licence policy the operator set, into a family that
policy did not consider.

**OD-11 — Common Voice under consent revocation: a class this vocabulary does not have.**
Mozilla moved Common Voice exclusively to the Mozilla Data Collective in October 2025, and said
why (VERIFIED, 2025-08-19): *"When someone chooses to revoke their consent to be included in a
dataset, we need a way to remove them."* The MDC terms forbid re-hosting (VERIFIED). The
underlying CC0 dedication is irrevocable as copyright, so a frozen pre-October-2025 snapshot is
defensible **on licence grounds** and indefensible **on consent grounds**, and those are different
sentences that this document's `PERMISSIVE_OK`/`BLOCKING` vocabulary collapses into one.
*Recommendation:* **EVAL-ONLY on a held snapshot, the consent question recorded in the model card
rather than resolved as a licence question, and a new verdict class — `CONSENT_OPEN` — added to
the vocabulary** so the next corpus of this shape is not filed as `PERMISSIVE_OK`. If the corpus
is wanted for training, fetch fresh through MDC and accept its terms as a deliberate act.
**Cost of deciding late:** 31,841 hours is large enough to be tempting mid-build, and a snapshot
silently ages against every revocation made after it was taken — the harm accrues without any
event that would prompt a re-read.

**OD-12 — Emilia and GigaSpeech: accept the BLOCKING reading, or work around it later.**
Both distributors disclaim owning their own audio — Emilia (VERIFIED): *"Emilia does not own the
copyright to the audio files; the copyright remains with the original owners of the videos or
audio."* GigaSpeech (VERIFIED via two secondary corroborations): *"SpeechColab does not own the
copyright of the audio files."* That is the tiny-imagenet defect, not the GooAQ one: **relicensing
the release moves NC and SA corpora and moves nothing BLOCKING.** *Recommendation:* **accept
REFUSE for both, and record the acceptance now, while the scale is still abstract.** **Cost of
deciding late:** this is the single largest quiet-workaround risk in the audit — 101,000 and
10,000 hours are exactly the scale that gets rationalised six weeks into a build. Deciding now
costs nothing; discovering the decision was never made costs A1.
*Why it is the operator's:* only the operator can accept a rights risk this document can only
describe.

**OD-13 — LibriVox as ONE provenance group (DEC-46) needs explicit sign-off.**
It is the audit's own judgment call rather than a claim any dataset page makes, and it is
load-bearing: it is the difference between a v1 mix that satisfies B1/B2 and one that is a ~90%
monoculture wearing ten names. **Deferred with the rows by DEC-48; it is what A0f blocks on when
the production phase opens.** *Recommendation:* **sign it off, and rely on A0f's gate (iii),
which now PASSES OR FAILS on the grouped numbers rather than printing them, to report B1/B2 both
ways** so the correction is a pair of numbers rather than an argument — if the
two agree, the correction was unnecessary and the receipt says so. **Cost of deciding late:** the
worst kind. A mix built per dataset name looks balanced in every receipt it produces; the defect
surfaces only as *"`auditory` is a slightly weak version of the thing its name promises"* — the
failure `CORPUS-CONTRACT.md` exists to catch, cheapest to prevent before the first cap and most
expensive to diagnose after training.

**OD-14 — BBC Sound Effects: the terms were never verified at the primary source.**
The BBC's own page refused the fetch; the RemArc characterisation (*"free use for research,
educational, and personal projects"*, no sale-of-sampled-music) comes from third-party snippets
only. *Recommendation:* **EVAL-ONLY, and keep it out of A0's fetch list until someone reads the
BBC's own terms page directly.** **Cost of deciding late:** small in volume, large in principle —
this project's own rule is that a doc does not beat a probe, and shipping a fetch on a snippet
would be the audit contradicting the method that produced it. The same holds, at lower stakes,
for DataSEC/DataSED and the Loquacious Set, both under-verified and both worth one follow-up pass
before either anchors anything.

**OD-15 — will `auditory` or `speech_output` ever ship as standalone checkpoints?**
This is the only question that changes what audio actually **costs**. If they ship only inside the
composed model, that model is CC BY-NC-SA regardless (DEC-31, via `memory`), NC audio is free, and
Expresso, ESC-50, UrbanSound8K, Clotho and FSD50K's NC clips are all available at no licence cost
(DEC-45). If either ships standalone, every NC source becomes a real, separate MIT-vs-NC decision
for that one checkpoint. *Recommendation:* **build the clean tier first and keep the NC sources a
named, separable add-on**, so the choice stays reversible; the price of that reversibility is
Expresso's 40 hours of expressive style, which nothing in the clean tier replaces — a
speech-output head trained on neutral LibriVox narration alone learns to read audiobooks and
nothing else. **Cost of deciding late:** a checkpoint trained on a mixed tier cannot be un-mixed.
The reversibility exists only before A1 and A2 run.


**OD-17 — W4 FAILED TWO OF FIVE PRE-REGISTERED CLAUSES AT ITS PRE-REGISTERED CONFIGURATION.
PIVOT PER §9.14 AS PRE-COMMITTED, OR AMEND. OPERATOR TO CONFIRM.** `[DEC-68]`
This is the single largest open decision in the programme and it is put here rather than settled
above **because this document must not move a bar after seeing the number it produced.** §4.0's
DEC-68 block carries every measurement; this entry carries only what each branch costs.

**What is on the table.** The pre-registered run (batch 1280, `token_loss_chunk` 512, 4,000 steps,
receipt `w4-chunked/run-1280/memory-20260903T184441Z.json`) **passes (a) and (d)**, **passes (b) as
the harness scored it**, and fails two: **(c)** full-pool `recall@10` **0.200** against BM25's
**0.440**, and **(e)** final-block rank ratio **1.2102** against an absolute **2.0×**.

**AND A THIRD CLAUSE IS CONTINGENT, WHICH IS PART OF THIS DECISION AND NOT A FOOTNOTE TO IT.**
Gate (b) is pre-registered and implemented **strictly** (`recall@10 > 0.20`;
`beir_fiqa.py:476`), and the receipt's value is **`0.20000000298023224`**, which is exactly
`float(numpy.float32(0.2))` — the float32 recall **is** 0.2 and the entire margin is float32
representation. **Under a strict reading the run is two passes and three fails.** §4.0 reports the
receipt's own `passed: true` and refuses to resolve `>` versus `≥` after seeing the number, because
that is the same move branch 2 has to justify for (c) and (e). **The operator resolves it here**,
and the resolution is binding on every future floor in this programme rather than on this receipt.

**BRANCH 1 — PIVOT, exactly as §9.14 and §9.1 pre-committed it.** *"A region that clears (2) and
fails (1) goes straight to option (3), the pivot"*, and §9.1 R4 states the general rule
independently: *"a region that fails its retrain gate goes straight to §9.14's pivot."* **The pivot
is to reorder the phases — regions and the interconnect trained together from the start, the
operator's phase 3 before phase 2.**

*What it costs, in this document's own words:* **region receipts stop being comparable** (every
per-region number in §4.3, §5.4 and the appendix loses its baseline), **region-per-host training is
lost** (which is what DEC-54's whole placement policy exists to exploit), and the phase-2 bill of
≈ 7–13 GPU-hours is replaced by a materially larger programme whose first checkpoint is a composed
model. **W7a, W7v, W1c, W2b's freeze order and the entire `s_r` invalidation rule are re-derived**,
because there is no longer a frozen region set to compute `s_r` against. **W4n does not exist under
this branch.**

*What argues FOR it:* it is what was pre-committed, and a pre-commitment honoured only when it is
convenient is not one. The 2.0× clause was set from W1's own measured range; the retrain moved the
ratio to 1.21–1.35 and no configuration measured reached 2.0×.

**BRANCH 2 — AMEND, WRITTEN AS A PRE-REGISTRATION. RECOMMENDED.** Four clauses, all of which must be
fixed **in writing before any run they govern**, or this branch is just the failure mode branch 1
exists to prevent:

1. **KEEP FiQA as the OUT-OF-DISTRIBUTION full-pool eval, and ADD an IN-DISTRIBUTION full-pool eval
   with a PARENT BASELINE, measured for BOTH `retrieve` AND `compress`.** *Rationale, and it is a
   measurement not an excuse:* FiQA is **14,131 of 782,959 training pairs — 1.8% of `memory`'s
   corpus** [V, the receipt's own `corpus.sources`], and **gate (c) has never had a parent
   baseline**: neither parent was ever run against that pool, so *"the merge degraded retrieval"*
   and *"an unmerged `retrieve` would have failed the same bar"* are **indistinguishable from the
   receipts that exist**. An in-distribution pool (drawn from the corpus the region was actually
   trained on, at comparable pool size and qrel density) with **both parents measured on it** turns
   gate (c) from *"beat BM25 on a domain you saw 1.8% of"* into *"beat BM25 on your own
   distribution, and beat what you were merged from"*. **FiQA stays** — dropping it would be
   removing the eval that failed — but it is labelled as the OOD arm and BM25 on it becomes a
   reported number rather than a pass/fail clause.
2. **REPLACE the absolute 2.0× rank clause with an IMPROVEMENT-OVER-CONTROL clause on the aligned
   harness, at a PRE-REGISTERED MARGIN.** *Rationale:* 2.0× was chosen from W1's pooled-versus-token
   ratios (0.66× to 1.30×) as the threshold W1 pre-committed for *"no retrain needed"* — a number
   about untrained regions, applied to a retrained one. The retrain's **measured, control-separated
   effect is real**: **1.0851 → 1.3475** at batch 512, **99% of the movement attributable to
   `L_decorr`** at 50 steps. The replacement clause is `ratio(terms on) ≥ ratio(control) + δ` on the
   identical W1 harness and the identical held-out items, **with `δ` written down before the run**
   and **the control arm run at the same scale as the treatment**. *What it costs:* every W4/W7 run
   now costs **two runs, not one**, and §4.3's retrain bill roughly doubles for the affected rows.
   *What it buys:* a clause that measures the thing the objective was added to change.
3. **PRE-REGISTER A NEGATIVES ABLATION — batch versus a negatives queue versus GradCache — as the
   NEXT LEVER.** *Rationale:* DEC-68's curve makes the in-batch-negatives count the dominant lever
   on gate (c)'s metric (**6.7×** in full-pool `recall@10` across a **5×** change in batch), and
   batch 1280 is where the 3090 Ti runs out. Row **W4n**, blocked on this decision.
4. **THE PIVOT REMAINS THE FALLBACK.** If the amended gate fails, branch 1 fires — **and the
   amendment is spent, not repeated.** One amendment, pre-registered, then the pre-commitment
   stands.

**AND ONE FIX THAT IS NOT A BAR MOVE AND THEREFORE APPLIES UNDER BOTH BRANCHES: FIX THE FLOOR
COMPARISON'S DIRECTION AND ITS PRECISION, IN WRITING, BEFORE THE NEXT RUN.** Every pre-registered
floor states whether it is `>` or `≥`, and the harness compares at a stated precision against a
value it has rounded explicitly, so that **no gate in this programme can ever again turn on 3e-9 of
float32 representation**. This costs nothing, changes no threshold, and is required whichever branch
the operator takes; the reason it is written here rather than applied silently is that applying it
silently to *this* receipt would decide (b) — which is the operator's call, above.

*What branch 2 costs beyond the runs:* the honest cost is **credibility**, and it is why this is
the operator's call and not this document's. A gate amended after it failed is weaker than a gate
that held, **however good the reasons** — the mitigation is that the amendment is written before the
runs it governs, its margin is fixed in advance, and this entry is the record that the original
clause failed rather than a page where it never existed.

*Recommendation:* **branch 2**, on the grounds that (c) was measured against an eval the region saw
1.8% of and against a bar with no parent baseline, and that (e)'s threshold was imported from a
measurement of untrained regions — but **the recommendation is not the decision**, and DEC-68
changes no gate until this is answered. *On (b) specifically there is no recommendation*: the
document that would benefit from `≥` is this one, so it states the arithmetic and stops.

**OD-18 — FIVE LEGAL READINGS FROM THE DATASET FACTORY'S PASS 1, RANKED. THE FIRST NEEDS NO
LAWYER.** `[DEC-73]` Each is stated precisely enough to be billable in
`DATASET-FACTORY-CATALOGUE-2026-09-03.md` §10.1; this is the ranking and what each could invalidate.

1. **Is CC BY-NC-SA 4.0 coherent as the composed model's release licence given its CC BY-SA
   inputs?** The only one that could invalidate a **shipped** artefact rather than constrain a
   future one. Under the permissive reading it is a freely chosen licence and coherent; under the
   cautious reading SNLI's §3(b)(1) requires the same licence elements (CC BY-NC-SA is not) and
   §3(b)(3) forbids added restrictions, while releasing as CC BY-SA would breach GooAQ's NC term —
   **under which there is no compliant release licence for the composed model at all**. Two fixes
   are already implementable without a ruling: **separate the checkpoints per region** so the SA and
   NC obligations never meet in one licence, or drop one side. **THE ACTIONABLE HALF NEEDS NO
   LAWYER: the model card must STATE the reading it takes, and today it states none** — which is
   why the question is unanswerable rather than merely unanswered. That is row **P2′g**'s clause (4)
   and it is free.
2. **Do OpenAI's and Anthropic's output restrictions bind a third party who receives the dataset?**
   One ruling covering the **11 MODEL-OUTPUT-TERMS entries** (~1e9 tokens across `language_code`,
   `memory` and `reasoning`). *Preference if the ruling is unfavourable:* the **cosmopedia shape** —
   an Apache-weights generator run locally, which carries no such terms and is what P2′g's
   generator clause builds on.
3. **MS MARCO specifically: NC-with-a-waiver, or REFUSE?** Microsoft's own sentence stacks three
   concerns — *"non-commercial research purposes only"*, *"without extending any license"*, and
   **"we may not own the underlying rights in the documents"**. That third clause is closer to this
   programme's REFUSE class (distributors disclaiming what they distribute) than to plain NC. It is
   **5.3e8 tokens and 3.3 of `memory`'s 5.0 NC-inclusive percentage points** — i.e. almost all of
   what NC buys anywhere.
4. **Does Stack Exchange's per-item attribution clause reach an inference-time weight release?** Its
   four conditions include a **live outbound hyperlink to each original question and each author
   profile without `nofollow`**, which a dataset-level manifest cannot satisfy at any scale — so it
   is refused **structurally**, not on licence class. If the clause does not bite on weights, **the
   largest single lever against GooAQ's 79% concentration in `memory` becomes available**. Worth
   asking as one instruction with item 1.
5. **Is a machine-learning model a "Produced Work" under ODbL / ODC-By?** ODbL defines one as *"a
   work (such as an image, audiovisual material, text, or sounds) resulting from using … the
   Contents"*; weights are none of those and training is not obviously a query. If a model is
   neither a Derivative Database nor a Produced Work, **ODC-By is silent — a gap, not a
   permission**. Bears on FineWeb-Edu, C4, peS2o, OpenWebMath, WildGuard and SciFact.

*Three more are in the catalogue and are ranked below these five* — `toxigen`'s CDLA-versus-README
conflict, whether sui generis database rights reach a US-domiciled operator at all, and how far
Gemma's *"Model Derivative"* definition travels through synthetic data. *Recommendation:* **obtain
items 1, 3 and 4 as one instruction** (they share a factual record), **answer item 1's card clause
today without waiting**, and treat item 2 as moot by preferring the cosmopedia shape.


---

## Autodev is somebody else's document now — the pointer, and what CSD still owns `[DEC-60]`

**OD-1 and OD-2 ARE ANSWERED and leave the open list.** The operator answered both directly
`[OP: autodev-service-identity-and-sandbox.md]`: autodev runs under a **dedicated service account
with its own secret store and minimum-scope keys**, **not root and not sudo**, in a hardened
rootless sandbox whose **only egress is the lab-console gateway**, which validates the request and
**mints short-lived downstream tokens** rather than relaying anything the caller sent; its
knowledge comes from a **separate curated RAG vault** it can update and cannot escape. That is
OD-2's `PROXY_ALLOW` question settled (allowlist = the gateway-proxied services: Forgejo API plus
autodev's own RAG collection) and OD-1's write-scope question settled by identity and privilege
rather than by path lists.

**And the specification does not live here.** Operator ruling: *"remember to keep autodev work in
the autodev tree and repo so we dont mix autodev work with CSD work — autodev is essentially a
harness for automatic self contained development loops, while CSD is one of the intended models
that it should be able to run and orchestrate"* `[OP: autodev-work-lives-in-csd-autodev.md]`.
**The pointer, and it is the only autodev content this document should ever carry:**

| what | where |
|---|---|
| the spec | `docs/AUTODEV-IDENTITY-AND-SANDBOX.md`, branch `docs/autodev-spec`, repo `tzervas/csd-autodev` (Forgejo `git.vectorweight.com`) — **REV 5, committed at `ece346b`, PR #1 in that repo** `[DEC-76]`; rev 4 was `c192278` |
| its threat pass | beside it on the same branch (findings G14–G18, **G23**, T4, T6 are the ones CSD depends on — G23 is the one that CORRECTS OD-2 rather than extending it) |
| its skeptic pass | beside it on the same branch (findings S1, S2 are the ones CSD depends on) |
| why it is not here | the harness is meant to run several models; coupling it to one model's repo makes both harder to reason about and pollutes CSD's licence and provenance story |

**What CSD keeps is three prerequisites, and they are rows because a design that assumes a control
it does not own is how the assumption outlives the control** — the same reason OD-1, OD-2 and OD-3
were recorded here in the first place. **They are worse than OD-1 and OD-2 described**, which is
the whole argument for making them gated rows rather than a paragraph:

1. **PRE-1 — the gateway authenticates nobody on most routes.** OD-2's finding was that the
   gateway *forwards* the client credential before authorising. The measured defect is that
   **`dispatch_api_get()` never looks at a credential at all**, and `dispatch_api_post()`
   authenticates **only `/api/apply`** — while `proxy_upstream()` attaches `UPSTREAM_TOKEN` to
   every proxied route [V, `scripts/csd-lab-console:1012-1080`]. **The good property is the threat
   pass's, not OD-2's:** finding **G23** credits the gateway with never relaying the caller's
   `Authorization` upstream — OD-2 asserted the opposite, and G23 is the finding that is right.
   **G23 is true, and it is true because there is no caller identity to relay**, which is why a
   true property was worth re-reading rather than banking. Today the blast radius is bounded only by
   `PROXY_ALLOW` naming five read-only routes; the autodev design adds Forgejo and RAG **writes**
   to the same dispatcher, at which point *"read as written, the design hands every host on the LAN
   autodev's write capability with no credential at all."* **Authenticate on EVERY route before any
   credential is attached, and bind autodev's routes to the unix socket so they 404 on the TCP
   listener.**
2. **PRE-2 — `POST :9108/v1/queue` is authorised by source-IP prefix and runs jobs as `kang`.**
   `_peer_ok()` accepts any `192.168.1.*`/`172.30.*`/`172.32.*`/loopback source with **no token**,
   and `gpu-timeshare-worker` then runs the queued job **as `kang`** with
   `git/cabal-forgejo-agent` and `gpu/localai-api-key` in its environment, taking queued `extra`
   fields **into argv** [V, `ansible/files/akula-health-exporter.py:520-596`;
   `scripts/gpu-timeshare-worker:60-170`]. **This is the single most exploitable path currently
   live, and it is not part of the new design — which means the sandbox's egress restrictions are
   moot until it is fixed**, because the same credentials are reachable from any LAN host without
   going near the sandbox. **Token-gate the route against a producer identity with a default-deny
   `(identity, kind)` allowlist, and move the worker to `svc-timeshare`.**
3. **PRE-3 — `kb_http._auth_ok()` fails OPEN on loopback when the token file is absent**
   [V, `rag/integration/kb_http.py:351-360`]. Small window today (the file exists at 0600) and
   disproportionate anyway, because the whole design leans on the gateway and this store as its
   enforcement points. **Delete the bypass; `SystemExit` at startup if the token is absent, so the
   failure is a dead service rather than an open one.**

**These three BLOCK any autodev GPU route**, which is the only interlock this document can
legitimately assert: autodev may not be handed a route into the timeshare scheduler or the reserve
while an unauthenticated LAN caller can reach the same credentials by a shorter path.

**OD-16 — TWO DIVERGED COPIES OF `csd-lab-console` EXIST, AND CONSOLIDATION IS THE OPERATOR'S
CALL.** Raised rather than fixed silently, as the operator asked `[OP:
autodev-work-lives-in-csd-autodev.md]`. **CogSynDelta `scripts/csd-lab-console`** (~1,349 lines) is
**the copy the live user unit `csd-lab-console.service` runs on :9118**, and it is the one carrying
the `PROXY_ALLOW` / `UPSTREAM_TOKEN` proxy section — so **PRE-1's fix goes there, where the running
code is**. **csd-autodev `scripts/csd-lab-console`** (~2,275 lines) has the three-host feed and the
Goals tab and **no proxy section at all**. *Recommendation:* **consolidate into `csd-autodev`**,
since DEC-60 makes that repo the harness's home, and have CogSynDelta consume it rather than carry
a fork — but land PRE-1 on the running copy **first**, because a security fix that waits for a
consolidation is a security fix that has not happened. *Why it is the operator's:* it decides which
tree owns a service the operator runs, and the two copies have diverged in features as well as in
security posture, so the merge is a product decision and not a mechanical one.

### DEC-76 — the pointer is rev 5, and the rule five rounds produced is recorded with it

**A pointer that names a superseded revision is worse than no pointer**, because a reader follows it
and reads the wrong design. The pointer table above is updated in place: the spec is **rev 5,
`ece346b`**, on `docs/autodev-spec` in `tzervas/csd-autodev`, **PR #1** in that repo, with the
threat pass and the retained skeptic write-ups under `docs/evidence/autodev-threat-2026-09-03/`. Five
revise-and-attack rounds went into it, **and three of those rounds' write-ups were not committed**:
the directory holds exactly `01-threat.md`, `skeptic-round4.md` and `skeptic-final.md`
[V, `git log --diff-filter=A` on that path].

**The residual criticals, in one line, and deliberately not summarised further** — §"A note on
scope" below forbids copying the spec's clauses here, and this is the minimum a CSD reader needs to
know the spec is not finished. **The FINAL (round-5) skeptic — `skeptic-final.md`, the pass against
rev 5 — still lists five critical and six high** (23 findings), which is the count the source memory
records. *The round-4 pass, `skeptic-round4.md`, was against rev 4 and lists **four** critical and
six high (21 findings); it is a different document and a different count, and this entry does not
mean it.* Of the five, **two are overtaken by events** — the `/api` auth gate **is** on `main` (PR #7, `5303544`),
and the closure pin went stale because CSD moved twice in eight minutes — and **the rest are rev-6
material, to be folded WHEN IMPLEMENTATION STARTS and not before**: the runtime import recorder must
run the protected tests **under `pytest`** rather than import their modules, and needs a `⊇` floor
so it **can** fail; P0a's check (0)/(0r) conflicts with ADX-52 as written because the spec models a
per-request `credentials.py` fork that the live code does not do — `apply_token()` reads
`CSD_APPLY_TOKEN` from the environment set at unit start, verified in the running console, **so the
spec must model the code rather than the reverse**; `socket_ident` must be re-stat-ed per connect or
the FD refreshed, else a socket-unit restart never invalidates; and §0d must be pinned to the
merge-base with `main` and its closure regenerated at implementation time.

**THE RULE, and it is the part that transfers to every other spec in this programme: FREEZE A CODE
REVISION FIRST — a tag — THEN WRITE THE SPEC REVISION AGAINST IT, THEN IMPLEMENT.** *A spec chasing
a moving tree cannot converge*, and five rounds against a tree that moved twice in eight minutes is
the evidence rather than the hypothesis `[OP: autodev-service-identity-and-sandbox.md]`.

### DEC-77 — three governance additions, each from a failure measured this session

**(1) DEC-61's PR-ONLY RULE EXTENDS TO `tzervas/gpu-pack`, and therefore to every code repo.**
DEC-61 was written against CogSynDelta's `main` and read like a fact about one repository.
`gpu-pack`'s `main` is now protected the same way — **push whitelist = the operator alone, required
context `CI / test (pull_request)`, not applied to admins** — and rounds 1–4 landed through PRs #1
and #2 rather than by push. **A rule that holds in one repo is a configuration; a rule that holds in
every repo is a rule**, and this is the second data point that makes it one
`[OP: branch-and-pr-to-main-only.md, tooling-lives-in-its-own-repo.md]`.

**(2) THE CI SECRET SCAN HAS A PROSE FALSE-POSITIVE CLASS, AND IT IS A CLASS RATHER THAN AN
INCIDENT.** `gitleaks`' `generic-api-key` rule matches low-entropy `identifier=value` pairs
**inside prose**. Two instances so far, both in this repository, both genuine documentation: a
**tokenizer-load log line** (`gpt2_tokenizer.json, vocab_size=50257`), and **`decorr_weight=0.0`**
quoted inside `EVIDENCE_50_STEP_CONTROL_ARM`'s docstring, where the surrounding sentence is
reporting the three loss weights a control arm measured. **This programme writes measurements into
docstrings on purpose**, so the class will recur.

**The remedy, and the part that makes it safe:** the **narrowest allowlist the finding admits**.
**Two invariants, always:** `targetRules` limited to the **one** rule, and the **byte-identical
matched phrase** as the regex. **A third whenever the finding is confined to one file:**
`condition = "AND"` over the **exact file path** as well — which is what `.gitleaks.toml`'s
`memory.py` entry does. **The tokenizer-load entry in the same file deliberately has no `paths`**,
and its comment says why: the identical string is quoted inside `.gitleaksignore`'s own **immutable
history** (commit `6f8ff7b`), which full-history scanning re-reads on every run, so a path-scoped
entry could not reach it and a fingerprint entry keyed to that commit would be displaced by the next
line-number shift. **A rule that demanded a path would forbid the entry this repository actually
ships**, which is why the invariant is *rule + phrase always, path where it is possible*. What is
never acceptable is the other direction: a path-wide or rule-wide suppression would have made the
same failure go away and taken the control with it. **An allowlist nobody has tried to overreach is an allowlist nobody
has measured**, which is this document's standing rule about guards, applied to a scanner.

**(3) THE CI RUNNER HAS NO GPU TOOLING, AND A TEST THAT ASSUMES OTHERWISE TAKES DOWN AN ENDPOINT.**
`fleet-ci-base:1` has **no `nvidia-smi`** — no GPU passthrough into the job container — so a helper
wrapper that let `subprocess.run` raise produced `FileNotFoundError` **inside a status handler's
dict literal**, aborting an entire HTTP response that should have degraded one field. The bug
predated the branch that surfaced it; it was simply never exercised, because every earlier test that
reached that path took a different branch first.

**Two rules follow.** **Tests must not assume `nvidia-smi`, `docker` or `ssh` exists** — the
regression test is the CI scenario itself, a `PATH` with no GPU tooling on it at all — and **a
diagnostic subprocess must fail closed to a string** (`"unavailable: <detail>"`), never escape as an
exception, for missing binaries *and* for timeouts, since a hung remote call has the identical
failure mode. **This matters beyond CI:** DEC-74 makes every card's runtime on-demand, so *"the
tooling is not there right now"* is the **expected** state on a live host too, not only in a
container.


**A note on scope, so this section does not grow back.** The autodev *spec* — service account,
vault layout, sandbox profile, curated RAG vault, loop logic, its own open questions (including
whether `required_approvals` should leave 0) — is **not summarised here and must not be**. If a
future CSD decision depends on a clause of it, the reference is the pointer table above plus the
clause, never a copy.

# 9. Risks and falsifiers

Ordered by (probability × cost of late discovery). Every one names the observable that kills it.

## 9.1 THE CENTRAL BET — **FIRED, AND CLOSED. Resolved by measurement, then CONFIRMED by W1d.**

**Claim at risk:** that a region's sequence of position latents carries more usable information
than its pooled vector `[DEC-47]` — *latents*, never discrete ids. **Status: FALSIFIED as
trained, under participation ratio** (§4.0, `[V]`). All four production regions land in the
pre-committed dead band — ratios 0.66×, 0.78×, 1.00×, 1.30× against a ≤1.5× threshold — and the
per-item flag (`< 8` usable directions) fires for `code` (4.4) and `retrieve` (5.1).

**Under the other effective-rank definition in the same artefact it is NOT falsified** — entropy
ranks of 1.21×, 1.16×, 1.21×, 1.84×, all above 1.0 and one outside the dead band `[N3 fixed]`.
That disagreement never softened the retrain requirement, and **W1d has now removed its standing
to**: the confirmation was run, on 2026-09-02, in 116.5 seconds, and it **CONFIRMED W1 in every
region** — clean for `code` (+0.00), `compress` (−1.76) and `vl_latent` (−2.93), and
**CONFIRMED-BUT-NOISY** for `retrieve` (−6.05, seed self-check spread 2.54 pp against a 2-pp
rule). **No region was OVERTURNED, and in three of four the sequence-blind arm BEAT the token
arm.** **The verdict is settled, the retrains are LICENSED, and the penultimate-block fallback is
dead** — `L_token` attaches at the final block (§4.0). Nothing downstream of DEC-35 is provisional
any more. The mechanism this document predicted is the one that did it: `∂pooled/∂h_t = m_t/Σm` is identical
for every unmasked position, so InfoNCE supplies **no positionally differentiated gradient** and
shapes only the *mean* of the sequence.

**This risk has therefore moved from "watch it" to "pay for it".** The residual risk is now the
*remedy*, not the bet:

- **R1 — the token-aware retrain does not clear its gate.** Falsifier: W4/W7's pre-committed
  `token_global_pr_rank ≥ 2× pooled_pr_rank` clause, both ranks participation ratio. **Its first
  mitigation is gone:** the penultimate-block variant was measured cheaply in W1d, exactly as
  planned, and **lost in every region**, so the remaining mitigation is §9.14's pivot — regions
  and interconnect trained together from the start. Measuring a fallback before relying on it is
  what turned a comfort into a deleted option, and that is the cheap test paying for itself a
  second time.
- **R2 — the retrain clears the rank gate and damages the faculty.** Falsifier: the second gate
  clause (own receipt ≤ 1 point, `banking77` probe ≤ 2 points). A region that trades its faculty
  for a token surface — its **position latents**, `[DEC-47]` — is reverted, and the receipt says
  which clause it failed.
- **R3 — CLOSED. W1's statistic might have been wrong in the direction that mattered; the
  falsifier was run and it did not fire.** Falsifier: W1d's matched read-out probe, run **before**
  the retrain was spent `[A9 fixed] [N3 fixed]`. It reported CONFIRMED in all four regions, with
  the **sequence-blind arm winning in three**, which is a stronger result than the risk asked
  for. The residual, carried rather than closed: **the instrument is coarse** — an untrained
  cross-attention read-out is ≈ uniform attention and therefore a linear function of `pool()`,
  and 600 steps of read-out training slightly *reduced* recall for `code` and `compress` — and
  **`retrieve`'s confirmation is flagged noisy**, its seed spread exceeding the decision
  threshold. A coarse instrument that answers the same way four times, three of them against the
  design's preference, is evidence; it is not a precision measurement of how much the sequence
  carries, and no later row may cite it as one.
- **R4 — NEW, and it replaces the fallback that R1 used to point at.** W1d killed the
  penultimate-block variant (worse or noise in every region; participation-ratio rank 2–4× lower
  for every text region), so **R1's first mitigation no longer exists** and a region that fails
  its retrain gate goes straight to §9.14's pivot. Falsifier for the pivot's necessity: W4's gate,
  which is the first place the retrain either works or does not.

## 9.2 The reserve cannot be funded — **downgraded, not closed**
**105,065 text rows** against a **61,440**-row training requirement after the aqua_rat recovery,
DEC-42's union burn — now measured at three draws, not two (§5.1) — and **DEC-49's reserve
increase** (§5.1, §5.4) — a **1.71× surplus** where revision 1 computed a 2.38× shortfall,
revision 3.2 computed 2.13×, and revision 3.3's two-draw union computed 1.77×. **DEC-49 spent a third of
the recovered headroom**, which is the honest way to record it: the risk is downgraded and still
open, and it is now open by less margin than it was. This was the document's
leading risk and it was largely self-inflicted: the write-off is refuted by the sampling code
(`regions/pretrain.py:231`, `corpus.py:159-221` [V]) `[A5 fixed]`.

**What remains, and it is different in kind.** The risk is no longer *volume*, it is *diversity*
and *verification*:

- **Falsifier: W2a's four checks** `[N1 fixed]`. If the corpus fingerprint does not match, or two
  calls disagree, or `seed = 0` cannot be established for the `reason` run, **the write-off stands
  and this section reverts to revision 1's numbers.** The fourth check — which code revision the
  run used — does not revert the section; it decides whether the union burn was necessary, and the
  union is written either way (DEC-42). It is tested, not assumed.
- **Falsifier: B1 on the recovered pool, and it is FAILING as printed** `[N8 fixed]`. aqua_rat is
  **50.6% of the unallocated pool and 79.5% of the text-usable pool** (was 51.8% / 80.2% under
  the two-draw union — W2a's actual run found a third draw, D3, and burning it shrank aqua_rat's
  clean share further), against B1's 0.50 hard line and 0.40 operating cap. This is the *binding*
  max-share; revision 2 printed `fashion_mnist`'s non-binding 35.5% (now 36.3%) against B1 and
  read as a pass. Mitigation, in §5.1's order: hold aqua_rat to
  0.40 of the reserve's **source rows** by construction, close the gap with W3r/W7v composites, and
  failing that carry a dated waiver.
- **Falsifier: B2 on the recovered pool.** `N_eff` is **2.47** against B2's ≥ 3 (was 2.44 under
  the two-draw union) — and it rose each time only because the union burn shrank the largest
  source further, which is not diversification. The
  recovered rows are all aqua_rat, one source, one domain. Mitigation: a third genuine text source,
  or B2's dated waiver naming what the reserve cannot measure.
- **Falsifier: W3's gate** — if the constructive generators do not yield ≥5,120 admitted items
  per cross-bin pair, phase 2 has no training data. If items fail the NSRS filter because one
  region solves them alone, the failure is *informative*: it means `visual` is doing OCR and the
  layout must be made harder.

## 9.3 Superadditivity achieved degenerately — **three ways, not one**
`I > 0` is manufacturable without any information crossing between regions, and revision 1 named
only the first of three routes:

| route | closed by |
|---|---|
| **presence-as-tag** — read a region's *presence* as a domain label | **content-swap** (§2.7.4), required under both ablation styles |
| **budget-as-tag** — from phase C, read the *emitted schedule* as a domain label `[A18 fixed]` | **two content-swap arms** (re-emitted vs frozen schedule); G3′ is graded on the frozen arm, and the gap between arms is reported as its own number |
| **redundancy** — A and B each independently sufficient, so `I` is *maximal* with nothing crossing `[A2 fixed]` | **DEC-37's conjunction** (`Δ_A > 0` **and** `Δ_B > 0` **and** `I > 0`), the `REDUNDANT` quadrant as a named FAIL, and **B2t** |

Listed here because these are the failures that would otherwise be *reported as success*, which
is this programme's documented recurring defect. **Also listed because the second and third were
found by an adversarial pass and not by the design** — the lesson being that "the control closes
the degenerate solution" is a claim that needs its own attacker, not a conclusion.

## 9.4 The scheduler is imitative
Distillation caps the controller at the dense teacher's behaviour: it can never learn that a
region the dense model ignored is useful. **Refuted structurally by phase D** (§2.6), which
trains the controller by the task loss from a non-degenerate initialisation under the floored
simplex. **Falsifier:** phase D fails to beat phase C ⇒ the receipt says *"scheduler: imitative"*
and phase C ships. **Second falsifier:** S1–S3 show a constant schedule ⇒ *"scheduling: not
demonstrated"*, reported as its own verdict (DEC-30).

## 9.5 Write-back damages the frozen regions
Prefix tuning on an OOD input at `max_len 96`. **Falsifier: W5b's gate** (own-bin drop ≤ 1
point). **Pre-committed fallback:** disable and record, **and apply §2.4's consequences for
`Ĉ`, `edges` and `lockstep_groups`** `[A20 fixed]`. See OD-5.

## 9.6 The interconnect memorises the reserve
**27.4M** interconnect params (27,424,039, §2.3) against ~61k items `[N10d fixed] [DEC-49]`. **Falsifier:
the < 5-point train/held-out gap gate in W5.** Absent from the source proposal; grafted because this is exactly where a silent
memorisation result comes from.

## 9.7 The `memory` merge is worse than both parents
Two losses on one trunk can interfere. **Falsifier: W4's gate is explicitly *beats both parents
on both parents' own gates*.** If it fails, keep them separate and record that hippocampal
unification is an unpaid claim.

## 9.8 `reasoning` is too weak to receive attention mass
At r@1 0.0801 it may never be attended to. **Falsifier:** `a_reasoning` in W5's phase A. If it is
near zero on reasoning-bin items, **the region is the problem, not the interconnect** — and W1b
must have produced its receipt first, or there is nothing to compare against.

## 9.9 ADVERSARIAL — five boundaries, three of which revision 1 did not have

Required by the project's design-time adversarial rule. Revision 1 modelled **one** boundary and
modelled it well; the threat pass found three more and one live code gap. All five are here, each
graded **by construction** / **enforced** — because the first is provable from the
parameterisation and the rest are fallible, and saying so is the point.

**Ranking note, because it changes what to spend on.** The valuable assets here are **write**
assets, not read assets: the ability to write `src/` and `tests/` autonomously (it rewrites the
guard *and* the test that proves the guard fires, in one commit), the reserve allocation ledger
(integrity and **irreversibility** — the aqua_rat case shows the loss is real), and the
interconnect weights, which uniquely encode a **policy** rather than a representation and are the
only thing that trains through phases A–D. Weight *confidentiality* is far down the list and, on
a LAN with no untrusted human users, close to worthless.

### B1 — untrusted input → controller → `Schedule` → runtime

**The `Schedule` is untrusted data and the runtime validates it before executing it.** The model
is never trusted to bound its own resource requests.

| attack | bound | enforced where | bound is |
|---|---|---|---|
| starve a region | `ctx_r ≥ ctx_min`, `b_r ≥ 1` for every declared region; floored simplex `η/R` | parameterisation **and** validator | **by construction** |
| budget inflation | `Σ_r c_r·ctx_r ≤ B_kv`, `Σ_r b_r ≤ B_read` | simplex + largest-remainder integerisation, re-checked by validator | **by construction** |
| compute DoS via halting | `n_iter ≤ I_max` regardless of the halting head | runtime loop bound | enforced |
| force everything admitted every iteration | per-request FLOP ceiling; reject and fall back to a fixed default `Schedule` | runtime, before execution | enforced |
| malformed `A` / budgets | shape, dtype, range, simplex sum | validator | enforced |
| **log injection via `trace_id`** `[T2 fixed]` | `trace_id` is **minted server-side**; a `Schedule` carrying a client-supplied one is **rejected** | validator | enforced |
| **per-request `h_r` cache abuse** `[A28 fixed]` | cache bounded at `R × n_iter` entries, per-request, destroyed at request end | runtime | enforced |

**One structural comfort that is real and should not be re-solved:** the worst case an attacker
can reach through B1 is **the dense, ungated configuration** — the one the hardware was already
sized for — so there is no denial-of-service surface beyond *"the model you already budgeted for
runs"*. A softmax cannot sum past 1. Further work here is negative value.

### B2 — `episodic_store` write → later request read `[T2 fixed] [A28 fixed]`

**Revision 1's attack table had five rows and none of them mentioned the only writable object in
the architecture.** The simplex bounds *how much* budget the store receives; it says nothing about
*whose latents it returns*. If the composed mind runs as a fleet service with more than one caller
— the stated end state — request *j* attends to latents written by request *i*. No exploit is
needed; that is the default behaviour of a shared store with a shared budget.

**THIS BOUNDARY NOW EXISTS AT v1, AND THAT IS THE POINT OF THIS SECTION IN REVISION 3.3.**
Revision 2 (DEC-32) wrote these controls and then deferred the object they control, which made them
documentation. **DEC-49 builds the store, so every row below becomes an ACCEPTANCE GATE of a
programme row with a date on it** `[OP: csd-episodic-store-required.md]` — the operator's ruling is
explicit that the threat model's findings are *"the acceptance gates for the build, not reasons to
defer it"*. The table gains a column for that and it is the column that matters.

| attack (threat **T2**) | bound | enforced where | proved by |
|---|---|---|---|
| **cross-request poisoning** | partition key **derived server-side from the authenticated principal**, never from the request and never from the `Schedule`. The `Schedule` may request **budget**; it may never name a **namespace**. CSD's identity is `(scope, domain, logical_key)`, with `scope` the server-derived segment and `(domain, logical_key)` the composite verified upstream [V, `memory_protocols.py:123-125`] | store read path | **E1 gate (iii)** — a ≥10,000-write, ≥100-partition fuzz in which **zero reads cross a scope**, plus the deliberately mis-derived key that **must** produce a crossing so the fuzz is shown capable of catching one |
| **unbounded growth** | **byte** capacity, not entry count: §8 gap (a)'s residual formula computed per host per scheduler tick. `token_budget.max` (`b_store`) bounds what is *read*; `capacity_bytes` bounds what is *stored*; **they are different currencies** | store write path | **E1 gate (ii)** — the capacity differs between the 3090 Ti and the 5080, is respected (a write past it evicts rather than allocates), and **changes when the active-region set changes** |
| **eviction is a security control, not a housekeeping one** | scored eviction `importance + gpu_resident_bonus − staleness(last_accessed)`, ties by older timestamp then key [V, `storage/tiered.rs:150-178`] — an attacker who can choose what survives eviction can choose what a later request reads | store write path | **E1 gate (iv)** — a **constructed overflow** with known scores, asserting the surviving set is exactly the top-scored one and that the `+1.0` GPU-residency bonus behaves as `gpu_hint_protects_from_spill` pins it |
| **unauthenticated read** | unknown principal ⇒ **empty partition** ⇒ zero keys ⇒ the mind degrades to the no-store configuration, which is exactly phase A | store read path | **fails closed to a known-good, already-benchmarked state** — and after DEC-49 that state is *literally W5's receipt*, which makes the fallback checkable rather than asserted |

**Gate at W9, unchanged in shape and now reachable:** extend the fuzz to a **two-request** form —
request 1 writes, request 2 reads — and require request 2's `Schedule` and output to be
**unchanged from a clean run**. **W6 additionally runs with the isolation assertion enabled**, since
an integration verdict obtained across a leaking partition would be a measurement of the leak.

**One property of this boundary that the build does not remove, and it is recorded rather than
mitigated.** The store is *supposed* to let a later turn read what an earlier turn wrote — that is
X7's entire item shape. **So "a later request read what an earlier one wrote" cannot itself be the
alarm; only "across a scope" can be.** The security of this object rests entirely on the scope
segment being derived where the attacker cannot reach it, which is why E1's negative test
constructs a mis-derived key rather than only asserting that the correct one works.

### B3 — reserve construction → controller training (phases B/C/D) `[T1 fixed]`

**This is the highest-leverage attack in the pass and revision 1 did not cover it.** §9.9 defended
the *inference* path into the scheduler; the *training* path is softer, more durable, and reaches
the same asset. Phase D trains the controller by the task loss on reserve items with the
controller unfrozen; three of the four reserve generators are agent- or code-driven. Influence a
small fraction of rows so a rare lexical or visual trigger co-occurs with items whose
loss-minimising schedule admits a chosen region at high `b_r` — or, more usefully, **excludes
`reasoning`**, already the weakest region at r@1 0.0801 and the most likely to be dropped without
anyone noticing (§9.8). Yield: **an input-triggered routing backdoor in the dispatcher for every
faculty, surviving region retrains because regions are frozen.**

**Why the existing gates do not catch it, which is the important part.** NSRS is a *difficulty*
filter, not an integrity filter — a poisoned row satisfies all three conditions trivially (§5.3).
The `<5`-point overfit gate catches memorisation and is **blind to poison present in both
splits**, where the gap is zero. `assert_no_contamination` (`eval/metrics.py:229` [V]) measures
train/eval *overlap* through normalised-text fingerprints and is equally blind, because poisoned
rows are not duplicates of each other — they share a trigger. **Every integrity gate in §5 and
§2.6 is computed by the same pipeline that generates the data**, which is the same shape as the
programme's own worst recorded find, one level up.

| control | mechanism | bound is |
|---|---|---|
| **keyed split assignment** (DEC-39) | `split = HMAC(k_split, source_row_fingerprint) mod N`; key outside every agent-writable path. **A generator that cannot choose which rows land in eval cannot defeat any downstream gate.** No key ⇒ the loader refuses to build a split | **by construction**, and fails closed |
| **generator identity in the ledger** (§5.6) | generator name + SHA-256 of generator source + config hash per row, recomputed **on load**; mismatch **aborts** the run, does not skip the row | fails closed **iff** the abort is in the data loader, not a pre-flight script |
| **sandboxed construction** (§5.5a) | executable joins run on a RuntimeClass the GPU never shares: no network, no NFS, non-root, per-candidate wall-clock and memory caps | fails closed **iff** there is no plain-`subprocess` fallback |
| **S5 trigger sensitivity** (§2.7.6) | per region, change in admission probability under single rare-token insertion; beyond a pre-committed threshold is a **finding**, reported as a third verdict | **does not** fail closed — it is a gate, and gates get waived. It is here because controls 1–3 cannot cover the in-both-splits case and something must |

### B4 — checkpoint file → `torch.load` `[T3 fixed]`

A composed mind loads N region checkpoints, and the programme plans to push them to private HF
repos [V\*], so a checkpoint can arrive from a Hub or a peer. The tree already records the risk:
*"the moment a checkpoint can arrive from a Hub or a peer instead of being self-generated,
`weights_only=False` is remote code execution on load"* [V\*].

**The threat pass found a live gap. Half of it has since been closed by a separate change, and
the design states which half** — because *"the fix is landing"* and *"the control exists"* are
different claims, and this programme's recorded history is of the second being asserted on the
strength of the first.

The gap: `scripts/csd-quantize.py` and `scripts/csd-benchmark.py` both resolved a checkpoint path
**out of a receipt JSON** and loaded it with `weights_only=False`, so **the file that got
unpickled was chosen by a document** — on a share exported `rw,no_root_squash` to the whole `/24`
(OD-3), and under automation, since DEC-27 runs `csd-quantize.py` **once per region,
repeatedly**.

**LANDED, verified this session by re-reading the tree and the log [V]:** commit `c976e84`
*"fix(scripts): load receipt-named checkpoints with weights_only=True"*. Both call sites now pass
`weights_only=True` (`scripts/csd-quantize.py:128`, `scripts/csd-benchmark.py:87`), and
`tests/test_checkpoint_load_security.py` is **committed and tracked** — a test written to prove
the guard *fires*, not merely that the keyword is present, which is the tree's own convention.
The immediate remote-code-execution path is closed.

**STILL REQUIRED, and this design depends on it:** the fix is two call sites, and two call sites
is a state a future edit can leave. DEC-40 is what makes it durable, and none of it is built yet.

> **DEC-40.** One `load_checkpoint()` in `src/`, the **only** `torch.load` in the repository.
> `weights_only=True` **hardcoded** — not a default, not a parameter, no override. Manifest
> SHA-256 verified **before the file is opened** (the landed fix does not do this; it prevents
> code execution, it does not detect substitution). Scripts import it. **Enforced by a lint rule
> (ruff or grep in `code-quality.yml`) that fails CI on `torch.load` outside that module — the
> lint rule is what makes this fail closed when someone forgets, and without it this is a
> policy, not a control.** The hash-mismatch refusal gets the same negative-test treatment
> `test_checkpoint_load_security.py` already gives the pickle refusal. It is also the mechanism
> DEC-33's overlays bind against, which is the argument for building it once, properly.
>
> **Row and owner** `[N10c fixed]`: **W0c** in §4.1 — no dependencies, no GPU, three constructed
> gates (a CI lint rule verified by adding a `torch.load` and watching CI go red; a hash-mismatch
> refusal with the same negative-test treatment `test_checkpoint_load_security.py` gives the
> pickle refusal; and `grep -rn "torch.load" src/ scripts/` returning exactly one hit).
> **Owner: the operator, by hand — explicitly NOT autodev**, because OD-1 records that autodev's
> write scope plus merge authority can weaken a guard and the test that proves it fires in one
> change, and DEC-40 is a control on the loader itself.

### B5 — autodev agent → `src/`, `tests/` → merge

Recorded as **OD-1** and **OD-2**: it is outside this document's tree, and it is the boundary
that decides whether every control above stays built. Reducing the write surface is the control;
auditing every commit by hand is not — it will be abandoned within a fortnight, after which the
*belief* that commits are audited persists while the audit does not, which is worse than never
claiming it.

### Residual risks, accepted deliberately

1. **A poisoner with write access to generator *source* defeats content hashing** — the hash then
   faithfully certifies poisoned content. **Accept**, because OD-1's protected-path control is
   exactly what converts this from unbounded to a small named set of files a human must read.
2. **Inference-time resource DoS against the scheduler.** **Accept and spend nothing further** —
   B1's by-construction result is genuine.
3. **Weight confidentiality.** **Accept.** LAN, no untrusted human users, 86M-parameter research
   artefacts intended for eventual release. The threat to checkpoints is *substitution*, closed
   by DEC-40; theft is not worth a control.
4. **The interconnect memorising the reserve** (§9.6). **Accept** — the <5-point gate is
   adequate. This is an honesty risk, not a security one.
5. **S5 is a gate and gates get waived.** **Accept** — the in-both-splits case has no structural
   answer and something imperfect beats nothing. Pre-commit the threshold so waiving it is a
   visible decision.

### What NOT to build — named, so it does not return through a side door

- **A schedule anomaly detector.** The `Schedule` is *supposed* to vary with input — that is the
  entire claim of DEC-30 and S1–S3. There is no stable baseline to detect anomalies against, and
  the false-positive rate equals the rate at which the scheduler is doing its job. The
  validator's hard bounds are the control.
- **Per-region checkpoint signing with a key hierarchy.** One manifest with SHA-256 verified
  before load closes the actual threat. A key hierarchy adds key management and excludes no
  additional attacker.
- **Encrypting the reserve at rest.** Its value is integrity and irreversibility. Encryption
  provides neither. Hash it; do not encrypt it.
- **Rate-limiting the composed mind for DoS.** See residual risk 2.
- **Auditing every autodev commit by hand.** See B5.
- **Re-deriving `provenance` as a region.** Dropped in §1.3 and correctly; recorded here so it
  does not return through a security argument.


## 9.10 The measurement instrument is saturated
`code` r@1 0.9766 on a 512-pair diagonal. Any composed metric built on that instrument has no
headroom, so it cannot show composition [I]. **Mitigation:** W4 makes the full-pool FiQA BEIR
eval the gate, with `recall@10 > 0.20`, `MRR > 0.10`, **> BM25** and **> its own random-init
baseline** all pre-registered `[A13 fixed]`. **The second half of revision 1's mitigation is
withdrawn:** *"the NSRS filter's `τ_lo` calibration keeps B1 at 0.3–0.4 on the reserve by
construction"* is not a mitigation, it is A1 — a baseline pinned by the filter that built the
data. DEC-36 replaces it, and the headroom the filter manufactures is now *reported* rather than
*assumed*.

## 9.11 The memory budget has no margin — and DEC-49 puts a new claimant on it
The bf16 arithmetic closes to **16.00 GiB exactly**. **Mitigation adopted as the plan rather than
the contingency: int8 KV (DEC-25)**, which triples the token budget and restores real headroom.
A 30% sensitivity on the assumptions moves the bf16 token budget from ~17.9k to ~13.8k; the int8
plan absorbs that.

**The episodic store is a NEW claimant on exactly this budget, and its 524,288 parameters are not
the part that matters** `[DEC-49]`. The parameters are 2 MB at fp32 and are already inside §2.3's
table. **The store's DATA is not budgeted anywhere in this document**, deliberately: §8 gap (a)'s
rule is that its capacity is whatever is left after weights, KV and activations — `capacity =
VRAM_total − KV_reserved(context, regions active) − activation_reserve − safety_margin`, computed
per host per scheduler tick. **That is a residual claim, which is the safe shape and also the one
that can silently become zero.** Two consequences, stated rather than discovered:

1. **On the 5080 at the deployment configuration, the residual may be ≈ 0**, because §6.3's
   arithmetic already closes to 16.00 GiB exactly. A store with zero capacity is a store that
   evicts everything it writes, and it would pass E1's byte-capacity test (the bound is respected)
   while failing every recall item. **E1's probe therefore prints the computed capacity per card
   and E2's receipt prints the store's realised occupancy**; a capacity that rounds to zero on the
   deployment card is a **finding**, not a configuration.
2. **The mitigation is the same one, and it applies twice.** int8 KV (DEC-25) frees the KV term in
   the residual formula, so it buys store capacity on the same card by the same mechanism it buys
   token budget. It is now load-bearing for two subsystems rather than one, which raises the cost
   of it not working out — recorded here because §9.11 is where that cost belongs.

## 9.12 Phase 3 at 30B is not trainable on this fleet
Stated in §6.7 with the mitigation (progressive unfreezing by tract) and the correction
(region-granular pipeline parallel at microbatch ≤ 256 recovers 30× over DDP). **Falsifier: P5′'s
gate** — cross-host activation traffic ≤ 15% of step time. Recorded now so the plan does not
silently assume otherwise.

## 9.13 Fatal flaws named by judges — dropped or refuted

*(For the two adversarial passes on revision 1 — skeptic A1–A35 and threat T1–T4b — see §10 for
the finding-by-finding disposition and §11 for the refutations. This table is the earlier judge
round and is unchanged.)*

| flaw | source | disposition |
|---|---|---|
| lockstep not expressible by depth-grouping (equal depth = independent, not lockstep) | spine | **fixed** by DEC-17 write-back + DEC-18 cycle cross-check |
| scheduler is imitative / *"forbidden from mattering"* | spine | **refuted** by phase D (§2.6), with the FLOPs-only gate kept at phase C where it belongs |
| frontal cortex has no objective of its own — *"a definitional dodge"* | spine | **fixed** by `L_unify` + B3 ablation (DEC-20) |
| the context budget does not budget the dominant term; controller emits pre-computation, consumer is post-computation | spine | **fixed** by DEC-15's two currencies; `ctx_r` is an argument to `tokens()` |
| scale path stops at *"leaving ~4 GB"* | spine | **fixed** by grafting §6.1–6.7 whole |
| proposes deleting 1,320 tested lines | spine | **dropped**; DEC-12 marks superseded instead — no sign-off needed, nothing lost |
| `L_integrate` trains on the statistic the gate measures | systems-first | **dropped** (§2.6) |
| `provenance` declared a region (a log in the AI-specific slot) | systems-first | **dropped** (§1.3) |
| `code` → LoRA adapter settles an 87M taxonomy question with a 30B paging argument | systems-first | **dropped**; DEC-01 keeps faculty+specialisation, and the paging argument is kept where it belongs, in §6.6 |
| Gumbel-ST `active` + Switch aux — the exact mechanism this tree measured collapsing | systems-first | **dropped**; §2.6 phase A makes collapse unrepresentable |
| P4.0 blocks everything including the corpus work it depends on | systems-first | **dropped**; W2/W3 run in parallel on CPU hosts (§4.2) |
| I2 ablates an 800-param additive prior, not the tract; a null operation on a sparsified `A` | biology-first | **fixed** as I2′ — mask the ordered pair's cross-attention logits to −inf (§2.7.5) |
| scheduler relocated out of white matter | biology-first | **dropped** (§2.1) |
| `code` → `language` rename with a falsifier that does not test the claim | biology-first | **dropped** (DEC-01). **The name itself is later adopted anyway, by a different route: DEC-78 (revision 3.6) sets it by operator naming-rule fiat, not by re-arguing this finding's falsifier — the two dispositions do not conflict because they answer different questions** |
| one-big-sequence tract stack whose interconnect cost grows with the budget it allocates | biology-first | **dropped**; no term in the bounded workspace's `O(L·D_w²·(1+mlp_ratio) + Σb·D_w² + L·Σb·D_w)` grows with region size (§2.3, corrected in revision 2) |
| identified the 80.30% embedding lever and then never shared the table | biology-first | **fixed** by DEC-24 |

## 9.14 What falsifies the whole proposal

> **If, after W5 and W6, the dense workspace at full budgets does not beat oracle late fusion
> (G3) and does not show a positive interaction term under content-swap (G3′), then cross-region
> attention over frozen, isolation-trained encoders cannot integrate.**

**The pivot in that case is not to add a router.** It is to **reorder the phases** — regions and
interconnect trained together from the start, i.e. the operator's phase 3 before phase 2 —
accepting that region receipts become non-comparable and that region-per-host training is lost.
That is a materially more expensive programme, which is precisely why W1 (hours, done) and W6
(minutes of GPU) are placed before anything expensive.

**W1's result raises the prior on that pivot and does not trigger it.** The token surface as
*trained* carries nothing extra; the question §9.14 asks is whether cross-region attention over
frozen encoders can integrate **after** those encoders have been given a token-aware objective.
W4 and W7 exist to answer exactly that, and they are the cheaper of the two ways to find out — a
few GPU-hours (§4.3) against a whole-programme reordering. If W4/W7 fail their gate, or if W5/W6
fail after they pass, **the pivot is the third option in §4.0's objective list and is already
written down**, which is the difference between a fallback and a discussion.

**The decisive experiments in this document are the cheap ones, deliberately: the thing most
likely to be skipped should be the thing that costs least.** W1 is the proof that this works —
it cost hours, it fired, and it bought a bounded retrain bill before a single line of
interconnect existed.

**A SECOND FALSIFIER EXISTS NOW AND IT IS FURTHER OUT, WHICH IS WHY IT IS WORTH NAMING**
`[DEC-51] [OP: csd-mycelium-downstream-goal.md]`. Everything above falsifies the *architecture*.
**M0 falsifies the programme's REASON.**

> **If, after W10, CSD cannot clear a pre-registered pass-rate margin against a comparable open
> model on the Mycelium task suite — parse, typecheck, implement, refactor, ground truth from that
> repository's own tests — then the thing that was built is a research result and not a
> development engine, and the operator's stated purpose for it is not met.**

Three properties of this falsifier are worth stating because they are unusual for this document.
**(1) It is not a benchmark.** The success criterion is a real project's real tasks, which is what
the operator set: *"if it meets the bar, use it to drive forward the rest of the language-
development project."* **(2) It is the most expensive falsifier here and therefore the most likely
to be quietly softened** — the mitigation is that the margin and the comparison model are
**registered in writing before any CSD run**, and a margin chosen after seeing a number is
recorded as void. **(3) It carries an open question that this document must not answer by
assumption:** whether CSD does RAG **natively as a skill** — `memory` plus `episodic_store` end to
end — or needs the 1080 Ti helper. **Both arms are run.** `RAG native: NOT DEMONSTRATED` is a
useful outcome and keeps the helper in the deployment; **asserting the native path in a design
document would be assuming the single most load-bearing claim in the whole programme**, which is
also the claim DEC-49 makes plausible enough to be tempting.

---

# 10. Attack-pass disposition — every finding, and where it landed

Four independent passes have now attacked this document: a skeptic pass against revision 1
(`attack-skeptic.md`, A1–A35, six critical), a threat-model pass against revision 1
(`attack-threat.md`, T1–T4b plus a security-theatre list), a second skeptic pass against
revision 2 (`attack-skeptic-2.md`, N1–N10) and a third against revision 3.1's new material only
(`attack-skeptic-31.md`, S1–S21). All four files are held **unchanged**. Every finding is **ACCEPTED** — the fix applied in place and cited
at the changed spot as `[A-n fixed]` / `[T-n fixed]` — or **REFUTED** with a re-read `file:line`
in §11. Three findings are **ACCEPTED with a partial refutation**: the fix is applied *and* a
specific sub-claim is refuted, because both are true and collapsing them would lose information.

## 10.1 Skeptic pass

| # | severity | disposition | where |
|---|---|---|---|
| **A1** | critical | **ACCEPTED** — admission never sets a baseline; calibration split disjoint from the graded split; B1/B2 recomputed by separate forward pass; τ chance-normalised per bin | §2.7.0 (DEC-36), §5.3 |
| **A2** | critical | **ACCEPTED** — G3′ becomes the synergy conjunction; `REDUNDANT` quadrant named and FAILed; **B2t** trained matched-capacity late-fusion null added | §2.7.3 (DEC-37), §2.7.1, §2.7.4 |
| **A3** | critical | **ACCEPTED** — verified `IJEPA.encode` → `target_encoder.embed`; **DEC-34 declares the target encoder the deployed half** (fix option (a)) | §2.3, §1.2, §1.4, §7.5 |
| **A4** | critical | **ACCEPTED** — **a visual-resolution row is added (W7v)**, folded into the mandatory retrain; the two `visual × *` pairs are marked **blocked until W7v**, with the exact restatement if it slips | §4.1 W7v, §5.4 |
| **A5** | critical | **ACCEPTED** — verified `reservoir_sample`/`sampling_rng` determinism; **W2a recovers the draw**; §5.1 and §9.2 restated with the reserve funded. **Revision 3 (N1): the recovery burns the UNION of both draws (DEC-42), so the surplus is 2.13×, not 2.23×. Revision 3.3, after W2a's ledger actually ran (`feat/w2a-ledger-recovery`, `c6ea587`): a third on-disk draw was discovered, the union is 13,946 (measured, not bounded), and the surplus is 1.71×** | §5.1, §4.1 W2a, §9.2 |
| **A6** | critical | **ACCEPTED** — **DEC-38** reserves at source-row fingerprint granularity; the derived-item refusal is a W2b gate | §5.6 (DEC-38), §4.1 W2b |
| **A7** | high | **ACCEPTED** — `blocked_by` column added and filled; `s_r` invalidation rule stated as a first-class constraint; CPU-only work separated from GPU-blocked work | §4.1, §4.2 |
| **A8** | high | **ACCEPTED** — W7 is no longer "re-run W5"; it is a region-training row with a written objective and gate; the penultimate question moves into W1d's probe | §4.0 (DEC-35), §4.1 |
| **A9** | high | **ACCEPTED as CONFIRMATION (W1d), with a partial refutation of its premise** — the probe runs *before* the retrain is spent, not instead of W1 | §4.0, §4.1 W1d; **§11 R2** |
| **A10** | high | **ACCEPTED** — four properties that can fail (pad-invariance, independent reference, row-permutation, batch-composition); the 1e-5 test kept as a refactor smoke test only | §4.1 W0 |
| **A11** | high | **ACCEPTED** — B0 gets a trained read-out; **B0d** known-dispatch positive control added; absolute scores printed beside `I₀` | §2.7.1, §2.7.2 (G0d), §2.7.3 |
| **A12** | high | **ACCEPTED** — B0u is **trained** with uniform connectivity from initialisation, matching B3's discipline | §2.7.1 |
| **A13** | high | **ACCEPTED** — W4 pre-registers five clauses including `recall@10 > 0.20`, `MRR > 0.10`, **> BM25** and **> its own random-init**; W2c re-measures `retrieve`'s untrained baseline | §4.1 W4, W2c, §1.2 |
| **A14** | high | **ACCEPTED** — dev/sealed split, Holm/BH correction, null rates printed, eval re-sized against the number of decisions | §2.7.8 |
| **A15** | high | **ACCEPTED** — phase-A `min_r mean(a_r) ≥ η/R` floor (**3.75% at W5's `R = 4`, 3.0% at E2's `R = 5`**, written as the expression since 3.2 `[S31-16 fixed] [DEC-49]`) with named remedies; "unrepresentable" restated as "unrewarded" | §2.6 |
| **A16** | high | **ACCEPTED** — source-row split key, block bootstrap, B1/B2 run on the reserve's generators with a dated-waiver path | §5.4 (DEC-38) |
| **A17** | medium | **ACCEPTED** — option (a): `episodic_store` demoted to `placeholder`, **R = 4 → 6 pairs**, criterion restated. **REVISION 3.3 SWITCHES TO OPTION (b) BY OPERATOR RULING** (DEC-49): the store is built and its pairs are counted, so A17's defect is closed by **declaring X7/X8** and by the mechanical denominator rule (no pair without a mapped shape and a non-zero sealed count) rather than by exclusion. **A17 remains ACCEPTED, not reopened** — its finding was that an unconstructible pair must not sit in a denominator, and that is now enforced by a check instead of by an absence | §1.3 (DEC-49), §2.7.7, §5.4 |
| **A18** | medium | **ACCEPTED** — budget-as-tag named as its own degenerate solution; **two content-swap arms**, graded on the frozen-schedule arm | §2.7.4 |
| **A19** | medium | **ACCEPTED** — **DEC-41** names the v1 ranking head and candidate-set construction; B1/B2/G2 defined in that metric | §2.6 (DEC-41) |
| **A20** | medium | **ACCEPTED** — the no-write-back `Ĉ` variant is defined, and if unbuilt the gate is declared void, `edges`/`lockstep_groups` are not emitted, and the receipt says `topology: not demonstrated` | §2.4, §4.1 W5b/W9, §9.5 |
| **A21** | medium | **ACCEPTED** — the condition-(2) exemption is written into the filter (`nsrs: c1c3`); abstention is DEC-41's `NULL` candidate with its own gate; sizing stays 3,072 | §5.3, §2.6, §5.5(c) |
| **A22** | medium | **ACCEPTED** — violation claim and its `[V]` tag withdrawn (re-verified: `objective` ≠ `objective_family`); the two enum values kept; the design question raised separately | §7.3 |
| **A23** | medium | **ACCEPTED** — W2c measures the untrained `code` baseline and re-derives `τ_lo` per bin in chance-normalised units; W2b is blocked on it | §5.3, §4.1 W2c |
| **A24** | medium | **ACCEPTED** — adapters recomputed for the v1 list (591,872), white matter 26,899,751, composed 85,807,015; total re-tagged `[I]`. **Restated in revision 3.3 for DEC-49's five participants: white matter 27,424,039, composed 86,331,303** | §2.3 |
| **A25** | medium | **ACCEPTED** — DEC-26 scoped to ≥1B; v1's **31.8%** (was 31.4% before DEC-49) printed beside the cap with the crossover argument | §6.4 |
| **A26** | medium | **ACCEPTED** — `mlp_ratio` stated in both columns; at ratio 4, `L_ic=4` is 3.6% and `L_ic=8` is **7.2%, above the ceiling** | §2.3, §6.4 |
| **A27** | medium | **ACCEPTED** — MAC derivation printed (**1.56 GMAC/item**); complexity restated as `O(L·D_w²·(1+mlp_ratio) + Σb·D_w² + L·Σb·D_w)` | §2.3 |
| **A28** | medium | **ACCEPTED** — `episodic_store` and the `h_r` cache both get rows in the attack table; two-request fuzz specified. **Revision 3.3 makes the specified controls REACHABLE**: DEC-49 builds the object, so B2's four bounds become **E1's acceptance gates** with constructed failures rather than a table about a placeholder | §9.9 B2, §2.5, §4.1 E1 |
| **A29** | medium | **ACCEPTED** — W9's gate split into runtime-equivalence (same `Schedule`, eager vs DAG) and sparsity (vs dense) | §4.1 W9 |
| **A30** | low | **ACCEPTED** — restated as "the token surface retrains nothing"; the retrain budget is §4.3 | §2.2, §4.3 |
| **A31** | low | **ACCEPTED** — `memory` inherits `retrieve`'s table with the divergence recorded; DEC-24 binds from phase 3; 3.02× relabelled for three regions (1.67×) | §6.2 |
| **A32** | low | **ACCEPTED** — `token_dim` / `pooled_dim` in the protocol, the schema and every region entry | §2.2, §1.4 |
| **A33** | low | **ACCEPTED** — `tract_codec` marked `placeholder` with an explicit build trigger | §1.4 |
| **A34** | low | **ACCEPTED** — split into **W2a** (ledger + recovery, minutes, no deps) and **W2b** (construction + admission + guard) | §4.1, §5.2 |
| **A35** | low | **ACCEPTED with a partial refutation** — table tagged per row and re-instantiation made a W0 deliverable; the `visual` clause is refuted | §2.3; **§11 R3** |

## 10.2 Threat pass

| # | disposition | where |
|---|---|---|
| **T1** (§4.1, B3 training-time controller poisoning) | **ACCEPTED, all four controls** — keyed split (DEC-39), generator identity in the ledger, a stated sandbox with no `subprocess` fallback, and **S5** as a third receipt verdict | §5.6, §5.5(a), §2.7.6, §9.9 B3 |
| **T2** (§4.2, `episodic_store` partition; `trace_id`) | **ACCEPTED** — the partition/capacity/eviction contract written; `trace_id` minted server-side and rejected if client-supplied. **Revision 3.3 changes the disposition's SHAPE, not its verdict** `[DEC-49]`: revision 2 closed T2 by removing the object from v1, which closes a threat by deleting its asset. The operator's ruling restores the asset, so **T2 is now closed by CONTROLS WITH CONSTRUCTED FAILURES** — E1's cross-request fuzz (with a mis-derived key that must produce a crossing), the byte-capacity probe on two cards, and the scored-eviction overflow. **That is a stronger closure than the deferral was**, and it is the one the threat pass asked for | §1.3, §9.9 B2, §2.5, §4.1 E0/E1/E2 |
| **T3** (§4.3, `weights_only=False` in two scripts) | **ACCEPTED, and the immediate half has since LANDED** — commit `c976e84`, both call sites now `weights_only=True`, with `tests/test_checkpoint_load_security.py` committed to prove the guard fires [V]. **DEC-40 is the durable half and is still unbuilt:** one `load_checkpoint()`, hardcoded, hash verified **before the file is opened** (the landed fix does not do this), CI lint rule, hash-mismatch negative test. **Revision 3 gives it a row and an owner: W0c, owned by the operator by hand and explicitly not by autodev (OD-1)** `[N10c fixed]`. Revision 2 routed it to "OD-7", which is the disposition of the untracked working files — a different item, so DEC-40 had no owner at all | §9.9 B4, **§4.1 W0c** |
| **T4** (§4.4, autodev `ALLOW_PREFIXES` scope) | **ACCEPTED as an OPERATOR decision** — outside this tree | **OD-1** |
| **T4b** (§4.5, lab-console proxy forwards the token before authorising) | **ACCEPTED as an OPERATOR decision** — outside this tree | **OD-2** |
| **T-4.6** (ledger unbuilt / DEC-23 unexecuted) | **SPLIT: half ACCEPTED, half REFUTED.** The guard is unbuilt — accepted, and it is W2b's gate. **"DEC-23 is unexecuted" is refuted**: the ledger line landed | §5.2, §5.6; **§11 R1** |
| **T-4.7** (NFS `rw,no_root_squash`; mirror metadata lies) | **ACCEPTED** — NFS recorded as **OD-3**; source revision SHAs pinned in the manifest with the fetcher refusing unpinned fetches | **OD-3**, §5.6 |
| theatre list (§6) | **ADOPTED VERBATIM** — reproduced as *"What NOT to build"* so it does not return through a side door | §9.9 |

## 10.3 Second skeptic pass (revision 3)

`attack-skeptic-2.md`, findings N1–N10, run against revision 2. It confirmed 31 of 35 skeptic and
6 of 7 threat findings closed, and all three appendix refutations (R1, R2, R3) as holding. Ten new
defects; every one is applied in place below and cited at the changed spot.

| # | severity | disposition | where |
|---|---|---|---|
| **N1** | **critical** | **ACCEPTED, and confirmed the hard way.** The sampling algorithm changed under the recovery (`c42203c`, prefix → reservoir) and W2a's three checks passed under either hypothesis, so 4,982 rows of ambiguous provenance could have been certified clean into the reserve. **DEC-42: burn the UNION of every draw the ledger discovers.** When W2a actually ran (`feat/w2a-ledger-recovery`, `c6ea587`), it found not two draws but **three** — R, P, and an on-disk derived sample D3 — exactly the failure mode N1 was raised against, this time an artefact nobody anticipated rather than one that was reasoned about in advance. Cost: the union is **13,946** of the 97,467 (measured, not bounded); the surplus goes 2.23× → 2.13× (two draws) → **1.71×** (three). W2a's gate **fails on a ledger omitting any discovered draw**, which is what caught D3. §5.1's premise about the deleted receipt is corrected: no receipt on disk ever carried `cap_sampling` | **DEC-42**, §5.1, §4.1 W2a, §9.2 |
| **N2** | high | **ACCEPTED.** All six ablation pairs now carry a declared item shape, a construction and a printed sealed-item count; two new shapes (**X2** three-way executable join, **X6** two-premise numeric composite) cover the two pairs that had none. The W7v-slip branch's *"all 3 of 3"* is **achievable** — it was not — and gains a pre-committed reallocation so it is not also underpowered | §5.4, §5.5(a), §5.5(b′), DEC-37 |
| **N3** | high | **ACCEPTED.** §4.0 prints **both** rank definitions; they disagree and the sign reverses for all four regions; the verdict is recorded as **PROVISIONAL PENDING W1d**; W1d is made the deciding step with a rule that names no rank definition; the retrain requirement is **not** softened; W1c's statistic is named and its `< 32 of 512` restated in those units | §4.0, DEC-35, §4.1 W1d/W1c, §9.1, §11 R2 |
| **N4** | medium | **ACCEPTED.** The fifth measurement, `compress_repo_local` at **2.00× AMBIGUOUS**, is in §4.0's table, flagged non-production and undertrained (step 2,000 of 8,000, `max_len` 256 of 96), with the 2.5× two-checkpoint spread stated as the sizing of W1d's risk | §4.0 |
| **N5** | medium | **ACCEPTED.** The dependency inversion is broken by **W3r** — frame geometry, minimal renderer, checked-in fixture — a CPU-only row with no dependencies that **both** W7v and W3 depend on. Composite rendering leaves §4.2's CPU-parallel column until W3r lands | §4.1 W3r/W7v/W3, §4.2, §5.5(b) |
| **N6** | medium | **ACCEPTED.** G0d gains a precedence rule and a remedy path: a redundant *corpus* reports **`corpus: REDUNDANT`** against §5.3 condition (2), not `statistic: broken`; `Δ_A`, `Δ_B`, `Δ_AB` are printed for B0 and B0d so the two causes are separable on the page | §2.7.2, §2.7.3 |
| **N7** | medium | **ACCEPTED (part a).** B0, B0d, B0u, **B2t** and B3 are budgeted in `T_A` units and GPU-hours — **≈3.15 white-matter-run-equivalents, ≈3–6 GPU-hours** — **B2t is stated to be a second full white-matter run and counted as one**, and *"the decisive experiment is the cheap one"* is restated: the measurement is cheap, its controls are not | §4.3, §2.7.7, §2.7.1 |
| **N8** | medium | **ACCEPTED.** The **binding** B1 max-share is printed in both accountings — aqua_rat **50.6%** unallocated / **79.5%** text-usable against the 0.50 hard line, a **FAIL** (was 51.8% / 80.2% before W2a's run found the third draw) — beside `fashion_mnist`'s non-binding 36.3%, with the four-step remedy and the note that `N_eff` rose to 2.47 only because rows were burned | §5.1, §9.2, §4.1 W3 |
| **N9** | medium | **ACCEPTED.** `memory × reasoning` gets construction **(a′)** — a multi-hop join over the recovered aqua_rat rationales, text-only and W7v-independent, which also gives the recovered rows a cross-faculty consumer that is not the blocked renderer. The general bin gets **(c′)**, a text-only variant, and is added to the W7v-slip restatement | §5.5(a′), §5.5(c′), §5.4 |
| **N10** | low | **ACCEPTED, all seven.** (a) DEC-36's verify-by-failing gets the `source: admission \| graded` field that makes it constructible; (b) the W1 evidence is copied into the repo at `docs/design/evidence/w1-token-rank-2026-09-02/` and cited from there; (c) DEC-40 gets row **W0c** and a named owner; (d) the two stale 27.8M sites become 26,899,751 (**and 27,424,039 in revision 3.3, DEC-49 — the sweep was re-run, not assumed to have stayed done**); (e) `text_encoder.py`'s path prefix corrected to `regions/`; (f) W0's property (iii) gets both clauses and stops being false as written; (g) §5.2 and §11 R1 stop saying there is no train-time refusal — a coarse one landed at `0786a77` | §2.7.0, §4.0, §4.1 W0/W0c, §2.6, §9.6, §5.2, §11 R1 |


## 10.4 Third skeptic pass (revision 3.1)

`attack-skeptic-31.md`, findings **S1–S21** (2 critical, 4 high, 10 medium, 5 low), run against
revision 3.1's **new material only** — DEC-43 to DEC-46, rows A0–A3, the §2.5 contract change, the
§1.4 JSON additions, §4.3's audio bill, §5.7's tier tables, §8's OD-10 to OD-15, the committed
audit and the ratification brief. It also recorded nine claims that **survived attack** (V1–V9),
including the brief's line count and header discipline, the absence of any BLOCKING source from
the recommended mix, the `NC`-vs-`BLOCKING` policing, and a ten-row VERIFIED/INFERRED
spot-check — *"the marking discipline survived the transfer. The numbers did not."*

**Two findings are resolved BY DEC-48's deferral rather than by a fix, and §11 R4 says so rather
than restating them for `R = 5`.** The rest are fixed in place, on the deferred rows as much as on
the live ones, because a deferred row still needs a gate that can fire on the day it is picked up.

| # | severity | disposition | where |
|---|---|---|---|
| **S1** | **critical** | **RESOLVED BY DEC-48 in revision 3.2; EXECUTED BY DEC-49 in revision 3.3** — see §11 R4. Revision 3.2's account, kept because it is why the fix existed to apply: **not recounted.** A fifth participant would have made the all-pairs set **ten**, with `auditory × memory` and `auditory × reasoning` carrying no item shape and therefore pre-committed to `REDUNDANT`/FAIL — the A17/N2 defect a third time. **Audio is deferred: `R` stays 4, the pair set stays 6, and "≥ 4 of 6" at null rate 0.344 is unchanged.** The branch is not deleted, it is **pre-specified in A3** for the day the row is picked up: `auditory` enters as a **non-pair participant** unless A3 declares the two missing shapes, the set is held at **8**, and the criterion is restated **in the same breath** as **≥ 5 of 8**, null rate **0.363** | §11 R4, §4.1 A3, §1.3, §4.1 `s_r` rule |
| **S2** | **critical** | **RESOLVED BY DEC-48 as a v1 blocker, and FIXED as groundwork.** DEC-45's clean tier and DEC-46's balance claim could not both be true, and the audit had declined to compute either. Fixed anyway: the balance claim is now **`[I]` and cap-dependent** with the recomputed `N_eff` at three caps printed, **People's Speech is decided explicitly** (out of the clean tier, counted only as the SA-tier variant), and **A0f's gate (iii) becomes a PASS/FAIL** — grouped max share > 0.50 or grouped `N_eff` < 3 **fails the row** | §11 R4, DEC-46, DEC-45, §4.1 A0f |
| **S3** | high | **ACCEPTED.** A0's gate (i) keyed on **string equality** between `mirror_tag` and `licence_upstream`, which FAILs every correctly audited agreeing row (LibriSpeech, MLS, AMI, MUSAN, VCTK, Hi-Fi TTS, LibriTTS-R, AISHELL-3, Expresso) and **passes AudioSet**, the row it was written for. It now keys on **provenance of the read**: a required `licence_upstream_source` (URL + fetch date) that fails when absent, when its host equals the mirror host, or when the date is missing, plus a `grant_scope` enum — `metadata_only` catches AudioSet, `code_only` catches CSS10 and Libri-Light | §4.1 A0m |
| **S4** | high | **ACCEPTED.** A2's trunk status was unstated and its two possible readings gave a vacuous gate or a guaranteed failure. **The frozen-trunk + adapter reading is adopted**, "exactly" becomes **bit-exact on the logits**, and **A2 is removed from the pre-W2b set** — with a frozen trunk it changes no weights, so the `s_r` rule does not reach it | §4.1 A2, `s_r` rule |
| **S5** | high | **ACCEPTED.** DEC-46's cap was applied to `auditory` only, and the mix it was not applied to is ~86–91% LibriVox: four of six clean-tier `speech_output` sources are one group. **A0m/A0f's scope now covers BOTH mixes**, and **A2 gains a fifth gate clause** — grouped B1 of the consumed TTS mix printed, share > 0.50 FAILs. Noted where it belongs: this makes **OD-15 a balance question, not only a licence one** | §4.1 A0m/A0f/A2, DEC-45 |
| **S6** | high | **ACCEPTED.** *"MIT / CC BY"* is not a licence and violates strictest-input on its face — CC BY imposes an attribution condition, MIT does not, and one CC BY input makes the output CC BY. **Both clean-tier cells read `CC BY 4.0`**, with the TASL notice obligation stated and the public-domain sub-tier described as the separate row it would have to be | DEC-45 |
| **S7** | medium | **ACCEPTED.** A0 said *"no dependencies"* in the same cell that listed five blocking operator decisions. **Split into A0m** (manifests, genuinely unblocked) **and A0f** (fetch, `blocked_by: OD-13, OD-15`), with the note that OD-10/11/12/14 concern sources outside the recommended mix and block nothing | §4.1 A0m/A0f, §8 |
| **S8** | medium | **ACCEPTED.** A2 trained on bucket C while A0's scope covered only the `auditory` mix. **One clause fixes it**: A0m/A0f fetch and score **both** mixes, with grouped balance numbers printed per mix | §4.1 A0m/A0f |
| **S9** | medium | **ACCEPTED.** A3's *"refuse an untrained `auditory`"* named **no field the filter could key on** — a random encoder emits outputs and `s_r` is well defined, just low, so the construction produced a number rather than a refusal. **W2b now refuses any region whose `status` is not `built` and whose receipt path is absent or whose checkpoint hash does not match the resident weights**, which gives `planned` a mechanical job and the `s_r` rule an enforcement point | §4.1 A3, §1.4 |
| **S10** | medium | **ACCEPTED, fixed in the committed audit.** FSD50K's stratum shares summed to **105.1%**. Recomputed from `audio-bucket-B.md:33`: CC0 **38.8%**, CC-BY **45.9%**, CC-BY-NC **11.8%**, **CC Sampling+ 3.5%** — a fourth, restrictive tier the *"CC0+CC-BY vs NC"* phrasing silently dropped. Both denominators corrected (**34,976** of 40,966 dev; 8,403 of **10,231** eval) | `AUDIO-CORPUS-AUDIT.md` |
| **S11** | medium | **ACCEPTED, fixed in the committed audit.** The audit's own bucket tables collapsed mirror tag and upstream terms into one column — the exact structural rule A0's gate (i) exists to enforce. **The column is split** into `mirror id + tag` and `upstream licence (+ mark)`, which is also what makes the committed doc directly consumable as A0m's manifest seed | `AUDIO-CORPUS-AUDIT.md` |
| **S12** | medium | **ACCEPTED.** `speech_output` was a sibling entry in `regions[]` with `emits: "tokens"`, i.e. a participant by §2.2's operational definition, contradicting DEC-44's own systems test. **Moved to `language_code`'s `heads` list**, with the standalone entry kept for programme tracking at `emits: null` and `parent: "language_code"` so no count can pick it up | §1.4, DEC-44 |
| **S13** | medium | **ACCEPTED.** *"six clean non-LibriVox provenance groups"* counted an NC group (Expresso) and a share-alike group (People's Speech) as clean. **Restated: seven provenance groups, of which five carry no NC term**, with "clean" kept for the tier definition it already had | DEC-46, §1.3, §5.5 |
| **S14** | medium | **ACCEPTED.** The group table mixed hours and clips, and B1/B2 are share statistics. **Computed in hours**, Freesound converted (~145 h), and A0f's gate prints the unit | DEC-46, §4.1 A0f |
| **S15** | medium | **ACCEPTED.** The audit states B1 as ≤ 0.40 while the design carries a **0.50 hard line and a 0.40 operating cap**, and the recomputed mix lands at **0.403** — between them. **A0f reports against both**, hard line as FAIL, operating cap as recorded warning, the shape §5.1 uses for aqua_rat | DEC-46, §4.1 A0f |
| **S16** | medium | **ACCEPTED, and revision 3.3 is the case that vindicates the fix.** The phase-A collapse floor was hard-coded at **3%**, which is `0.15/5` — right only for a five-participant mind. At revision 3.2's `R = 4` it is **3.75%**, so the written gate was **0.75 pp too lenient**. **DEC-49 makes `R` five, so the VALUE returns to exactly 3.0% — the number that was wrong two revisions ago, now right for a different reason.** A document carrying the constant would be accidentally correct with no way to show it. **The expression `η/R` is what is binding, the receipt prints the evaluated value beside its own `R`, and W5 (`R = 4`, 3.75%) and E2 (`R = 5`, 3.0%) print different floors in the same programme** | §2.4, §2.6, §4.1 W5/E2, §10.1 A15 |
| **S17** | low | **ACCEPTED.** The decision index listed DEC-42 after DEC-46. **Moved up one block** | Decision index |
| **S18** | low | **ACCEPTED.** A1's and A2's verify-by-failing constructions compare the untrained baseline to itself: they prove the comparison is wired, not that the instrument reads, and an ASR stub returning the reference transcript regardless of input would pass. **A2 gains a null-input control** — feed the frozen ASR silence or white noise and assert `WER ≈ 1.0` | §4.1 A2 |
| **S19** | low | **ACCEPTED.** The brief called `speech_output` a *"phase-2 participant"*; it is a head and adds no participant and no ablation pair, and the only reason it could bear on W2b was the `s_r` weight rule. **Restated in both documents**, and under S4's frozen-trunk reading it need not precede W2b at all | §4.1 `s_r` rule, brief |
| **S20** | low | **ACCEPTED.** DEC-45 silently dropped **HiFiTTS-2** — the largest source in the bucket — from the `speech_output` clean tier the audit includes, and wrote *"8 languages"* where the audit says *"8 **non-Ukrainian** languages"*. **Both stated**: the exclusion with its two reasons (mirror-verified only; open question 9's *"would dominate any mix it joins"*), and the qualifier restored | DEC-45 |
| **S21** | low | **ACCEPTED, and re-applied in revision 3.3 rather than assumed to have stayed applied** `[DEC-49]`: at `R = 5` the per-pair column sums to **2,304** against a cross-faculty sealed total of **1,536**, and §5.4 prints both with the reason (shapes overlap pairs). The revision-3.2 numbers, kept: the per-pair sealed counts sum to 1,280 against a 2,560-item sealed half — resolved in §5.4 (only 1,024 sealed items are cross-faculty, and shapes overlap pairs) but inherited by A3, which promises to recompute the allocation. **A3's gate now requires the recomputed table to print the cross-faculty sealed TOTAL alongside the per-pair figures**, so the two are visibly reconciled | §4.1 A3 |

---

# 11. Appendix — Findings refuted, with evidence

A refutation here cites a file re-read in this session at a named line. Reasoning alone is not a
refutation; where I could only argue, I accepted the finding instead. **Four entries: three
refutations, two of them partial** — the finding's fix is applied and only a specific sub-claim
falls — **and one dissolution**, R4, where a finding is neither refuted nor fixed because the
premise it rests on was removed by an operator ruling.

## R1 — "DEC-23 is unexecuted" (threat pass §4.6). REFUTED in half.

**The claim.** *"**VERIFIED by absence:** no `RESERVED.jsonl`, no allocation ledger, and no
`reserve` reference anywhere in `src/` or `scripts/`… `config/mind/csd-regions.json` still assigns
`apps`/`code_contests` to nothing… So §5.6's stated gate is currently unbuilt, while DEC-23 is
described in the document itself as 'one line of JSON today and unrecoverable tomorrow'."*

**What I re-read.**

```
docs/design/CORPUS-CONTRACT.md:964
    | `deepmind/code_contests` | 13,328 | ~~code~~ -> **compose** | ATTRIBUTION |
docs/design/CORPUS-CONTRACT.md:966
    | `codeparrot/apps`        | 10,000 | ~~code~~ -> **compose** | PERMISSIVE_OK |
docs/design/CORPUS-CONTRACT.md:970-976
    "> **Reserved 2026-09-02 (DEC-23).** `deepmind/code_contests` and `codeparrot/apps` are
     corrected from `code` to `compose` in this table -- they are the only licence-clean paired
     natural-language-problem <-> implementation source in the tree ... so allocation happens
     now, before P2.3 trains `code`/`language_code`, per
     docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md §5.2."
scripts/csd-train-all.py:311-325   the aqua_rat cap and its B1 rationale, in place
```

**Verdict: the allocation half is REFUTED — the ledger line has landed, in the allocation table
the threat pass named, citing this document's own §5.2.** The threat pass ran while another agent
was applying it, which is exactly the failure mode of *"verified by absence"* in a live tree.

**The other half stands, is accepted, and is not softened by this refutation — but revision 2
overstated it and revision 3 corrects that too** `[N10g fixed]`. There is still no
`RESERVED.jsonl` and no per-row reserve reference in `src/`, so §5.6's gate remains unbuilt and it
is W2b's pre-registered gate, with **both** refusals (direct and derived) required. **What
revision 2 got wrong: it said there is "no train-time refusal".** A coarse one landed —
`0786a77`, `scripts/csd-train-all.py:79-120`, `_refuse_reserved_shards()` raising
`ReservedSourceError` at shard resolution for any region and in `--dry-run`, with
`tests/test_reserved_corpus_guard.py` proving it fires in three constructed cases [V]. It is
directory-prefix granularity and covers neither the derived nor the per-row case, which is the
part that matters and is unchanged. The threat pass's
ranking of this as the second-highest-value item in the pass is correct and unchanged.

## R2 — A9's premise: "the token cloud's rank is essentially always ≥ the pooled cloud's, for any encoder, trained or not". REFUTED by measurement. The *fix* is accepted.

**The claim.** *"`pool()` is a masked mean of the rows of `tokens()`… The set of pooled vectors
therefore lies in the span of the token vectors, and it is `N` points rather than `N·T`. The
participation-ratio effective rank of the token cloud is essentially always ≥ that of the pooled
cloud, for any encoder, trained or not — a random-init encoder would pass this test. So 'token
rank materially higher' is close to a tautology and the 'bet is dead' branch is close to
unreachable."*

**What I re-read.** `docs/design/evidence/w1-token-rank-2026-09-02/results.json` (copied into the
repo in revision 3, `[N10b fixed]`; the session path it was read from was `scratchpad/exp-w1/`),
`verdicts` block, measured on the
production checkpoints with the pre-pool capture verified bit-exact against
`TextEncoder.forward()` / `IJEPA.encode()` (`verified_notes`, max abs `0.00e+00` per region):

```
code        pooled 42.63  token-global 28.09   ratio 0.659
compress    pooled 45.33  token-global 35.57   ratio 0.785
retrieve    pooled 40.71  token-global 40.65   ratio 0.999
vl_latent   pooled 18.07  token-global 23.54   ratio 1.303
```

**Verdict: REFUTED for trained encoders.** Three of the four production regions have token-global
participation-ratio rank **strictly below** pooled rank, one of them by a third. The stated
inequality does not hold, and the branch A9 called *"close to unreachable"* was reached by **all
four** regions. The premise does hold for the *untrained* models in the same file (`code` 1.74
pooled vs 11.56 token; `compress` 2.10 vs 7.39), which is presumably where the intuition came
from — and that is the useful part of the finding: **the bias is real at initialisation and
reverses with training**, so the statistic behaves differently on the objects the decision is
actually about.

**A scope correction to this refutation, added in revision 3** `[N3 fixed]`. The refutation is of
A9's premise **as stated in participation-ratio terms**, and it holds there. `results.json` also
records **entropy-effective rank** for every surface, and on that statistic A9's premise **does
hold for the trained encoders too** — token rank exceeds pooled rank for all four regions (1.21×,
1.16×, 1.21×, 1.84×). So the honest verdict is narrower than revision 2's: *A9's premise is
refuted for the participation-ratio definition the rule was pre-committed against, and is
supported under the entropy definition.* Both columns are in §4.0. The W1 verdict was recorded as
provisional pending W1d for exactly this reason, **and revision 3.2 discharges it: W1d ran, its
rule read task performance rather than rank, and it CONFIRMED W1 in all four regions.** The
premise refutation above is unchanged by that — it is a statement about rank statistics, and W1d
deliberately does not use one.

**A9's fix is nonetheless ACCEPTED and applied**, as W1d, and the reasoning was not a courtesy.
The measurement was one statistic, and it licensed the most expensive item in phase 2 — a
retrain of every production region. A matched read-out probe cost **116.5 seconds** and could
have stopped it; **it was run, and it did not**. A
statistic that came out *against* the design's preference is more credible than one that came out
for it, and that is a reason to spend a little more on confirmation, not a reason to skip it.

## R3 — A35's third clause: "a `visual` figure that is about a different network than the receipts". REFUTED. The other two clauses are accepted.

**The claim.** A35 lists three defects in §2.3's `[V*]`-tagged parameter table: an explicitly
`[I]` row (the controller), an adapter row that does not match the v1 region list (A24), and
*"a `visual` figure that is about a different network than the receipts (A3)"*.

**What I re-read.**

```
src/cogsyndelta/model/vl_jepa.py:314-317
    self.encoder = ViTEncoder(cfg)
    ...
    self.target_encoder = copy.deepcopy(self.encoder)
    for param in self.target_encoder.parameters():
```

**Verdict: REFUTED.** `target_encoder` is a `copy.deepcopy` of `encoder`, so the two halves have
**identical parameter counts** — 10,712,448 each, as the document's own Appendix records. A3 says
so itself (*"the arithmetic survives either way"*). The `visual` **row** in the parameter table
is correct as a number under either choice of deployed half; what was wrong was the **label**,
and that is A3's finding, fixed by DEC-34. Carrying it a second time as a defect in the
arithmetic would overstate the damage to a table that needs correcting for two other reasons.

**A35's other two clauses are ACCEPTED**: the table is now tagged per row, the `[V*]` total is
demoted to `[I]` because a sum over an `[I]` row is `[I]`, the adapter row is recomputed for the
v1 participant list, and **re-instantiating at the design's actual configuration is a W0
deliverable** so the total stops being a projection.

## R4 — S1 and S2 (skeptic pass 3.1, both critical). S1 was DISSOLVED by DEC-48 and is now EXECUTED by DEC-49; S2 stays dissolved.

**The claims.** **S1:** *"adding `auditory` makes **ten** ablation pairs, not eight, and two of
the four new ones FAIL by construction"* — `C(5,2) = 10`, with `auditory × memory` and
`auditory × reasoning` carrying no declared item shape, so `Δ_A ≈ Δ_B ≈ 0`, the pair lands in
DEC-37's `REDUNDANT` quadrant, and *"≥ 4 of 6"* becomes undefined over a denominator of 10.
**S2:** *"DEC-45's clean tier and DEC-46's balance claim cannot both be true, and the audit itself
declined to support either"* — People's Speech in moves the tier to CC BY-SA, People's Speech out
costs the sixth group the `N_eff` argument needs.

**Both are correct against revision 3.1.** Neither is refuted here, and neither is answered by
recomputing a criterion for `R = 5`.

**What actually happened.** The operator ruled, later on 2026-09-02 than the intent that created
the rows `[OP: csd-multimodal-io-intent.md]`: *"we can wait to add audio as a future feature once
it proves out without audio. that will be more of a production phase implementation."* **DEC-48
defers `auditory` and `speech_output` to a production phase.** So:

- **S1's premise was gone for one revision, and revision 3.3 brings it back through a different
  door — so S1's FIX gets executed rather than pre-specified** `[DEC-49]`. Revision 3.2's position
  was that `R = 4`, the all-pairs set is six, and restating the criterion for `R = 5` would answer
  a question the programme no longer asked. **DEC-49 makes the programme ask it again**: not
  `auditory`, but `episodic_store`, admitted by operator ruling. **So S1's fix (b) — the new
  participant enters as a full participant and its pairs are counted — is APPLIED, and applying it
  required doing the thing S1 said a bare fifth participant could not do: declaring item shapes for
  the four new pairs.** X7 and X8 do that (§5.4), each store pair carries **256 sealed items**, and
  the criterion is restated to **≥ 7 of 10** at null **0.1719** under §2.7.3's new 0.20 null-rate
  ceiling, which is stricter than the 0.344 it replaces and therefore cannot be loosened by adding
  a participant. **S1 was right, and its
  being right is what made DEC-49 buildable rather than a recount:** the finding named the exact
  precondition — shapes, or the pairs are pre-committed to FAIL — and that precondition is what
  E1, the episode harness and §5.5(e) exist to satisfy.
- **A3's audio pre-specification is restated in the same breath, because the base changed under
  it.** With the store in, admitting `auditory` later makes the pair set **10 + 2 = 12** (it enters
  as a **non-pair participant** unless A3 declares shapes for `auditory × memory` and
  `auditory × reasoning`), and the criterion becomes **≥ 8 of 12** with null rate
  **`P(Bin(12,0.5) ≥ 8) = 0.194`**, the smallest `k` under §2.7.3's 0.20 ceiling — replacing
  revision 3.2's `6 + 2 = 8`, `≥ 5 of 8`, 0.363, which was correct for a four-participant base,
  is not correct for a five-participant one, **and exceeds the ceiling in any case**. **A
  pre-specification that is not re-derived when its base changes is a stale number waiting to be
  quoted**, which is the defect S1 itself was about.
- **S2 stops being a v1 blocker and its substance is fixed anyway**, because DEC-45 and DEC-46 stay
  in the document as groundwork and a wrong groundwork table is worse than none: the balance claim
  is tagged **`[I]`** and made **cap-dependent** with `N_eff` recomputed at three caps, People's
  Speech is **decided explicitly** (out of the clean tier, counted only as the SA-tier variant),
  both clean-tier cells read **CC BY 4.0**, and **A0f's gate (iii) becomes a PASS/FAIL** rather
  than a print — grouped max share > 0.50 or grouped `N_eff` < 3 fails the row.

**Why this is recorded as a dissolution rather than a fix — and why S1's entry then changed
again.** The pattern matters: a critical finding against a *requirement* can be answered by an
operator changing the requirement, and when that happens the honest record says *the premise was
removed*, not *the finding was closed*. **Revision 3.3 is the demonstration of why that
distinction was worth keeping.** S1's fix was written into a deferred row rather than deleted with
it; four days later a *different* fifth participant arrived by operator ruling, and the fix was
sitting there, applicable, and applied. Had S1 been recorded as "closed", DEC-49 would have had to
rediscover that a fifth participant needs declared item shapes — which is the discovery A17 already
cost this programme once.

**S2 stays dissolved and is untouched by DEC-49**, since it is about audio licence tiers and would
be live again the moment audio is picked up.

---

# Appendix — measured numbers this document depends on

Tags are **per line**, not per block `[A35 fixed]`. `[V*]` = measured by instantiating the real
classes or by reading a receipt. `[V]` = read in a named file in this session. `[I]` = arithmetic
over those. **Numbers revision 2 changed carry the old value beside them**, so a reader who saw
revision 1 is not silently handed a different figure.

```
--- REGION PARAMETERS ------------------------------------------------- tag ---
TextEncoder(vocab 50257, dim 256, depth 4, heads 4)   16,021,248      [V*]
  embedding table                                     12,865,792      [V*]  (80.3046%)
  non-embedding compute                                3,155,456      [V*]  (19.6954%)
IJEPA(JEPAConfig())                                   22,905,216      [V*]
  EMA target encoder  (THE DEPLOYED HALF, DEC-34)     10,712,448      [V*]  (46.7686%)
  context encoder (training scaffolding, discarded)   10,712,448      [V*]  (46.7686%)
  predictor                                            1,480,320      [V*]   (6.4628%)
     ^ the two halves are copy.deepcopy of each other (vl_jepa.py:316 [V]), so the counts
       are identical and only the LABEL changed in revision 2. IJEPA.encode is
       target_encoder.embed (vl_jepa.py:387 [V]); _features feeds the probe through it
       (vl_pretrain.py:202 [V]).
4 x TextEncoder + IJEPA                               86,990,208      [V*]  <- the "~87M", exactly

--- WHITE MATTER v1, RECOMPUTED FOR THE FIVE-PARTICIPANT v1 LIST [A24][DEC-49] -
workspace blocks x4 @ D=512, 8 heads, mlp_ratio 4     16,803,840      [V*]
frontal read-out                                       3,150,848      [V*]
thalamic controller                                    1,588,007      [I]   at 3 heads; design has 4
conditioning prefixes (3x256 + 1x384, n_cond 8)        4,727,808      [V*]
region adapters (3 x 256->512, 1 x 384->512)             591,872      [I]   was 986,624 (6 text)
episodic_store projections W_k, W_v (512x512)            524,288      [I]   RESTORED by DEC-49
                                                                            (was 0 under DEC-32)
latent bank + norm + type embeddings                      37,376      [V*]
WHITE MATTER v1 TOTAL                                 27,424,039      [I]   was 26,899,751
  ^ a sum over an [I] row is [I]. W0 re-instantiates it at the design's actual config,
    at FIVE participants.

--- COMPOSED MIND v1 ----------------------------------------------------------
language_code + memory + reasoning + visual           58,907,264      [V*]
+ white matter (incl. the store's two projections)    27,424,039      [I]
COMPOSED MIND v1 DEPLOYABLE                           86,331,303      [I]   was 85,807,015
  fp32                                                   345 MB       [I]   was 343 MB
  at 3.2675 effective bits/param                          35.3 MB     [I]   was 35.0 MB
  interconnect share of the mind                          31.8%       [I]   was 31.4%; vs DEC-26's
                                                                            2-5%, scoped to >=1B
  ^ THE STORE'S DATA IS NOT IN THIS TABLE. It is bytes, not parameters, and it is bounded
    by section 8 gap (a)'s dynamic capacity, computed per host per tick. W10 reports it
    separately and section 9.11 records that it may round to zero on the 5080.
memory-merge saving                                   15,890,176      [V*]
shared-embedding saving, 6 text regions               64,328,960      [V*]  (3.02x) -- but v1 has
shared-embedding saving, 3 text regions               25,731,584      [I]   (1.67x)   THREE (§6.2)

--- WORKSPACE COMPUTE, DERIVED [A27] ------------------------------------------
workspace MAC/item (L=64, Sum b=256, depth 4, mlp 4)    1.56 GMAC     [I]   was 620.8 MMAC [V*];
                                                                            does not reconcile
  per block: 4D^2L + 2L^2D + 2D(4D)L + 2D^2L + 2D^2*Sb + 2L*Sb*D = 390.1 MMAC
workspace KV bytes/item (bf16)                            2.10 MB     [V*]
complexity  O(L*D^2*(1+mlp_ratio) + Sb*D^2 + L*Sb*D)                  [I]   was O(L*Sb*D) only
30B interconnect, D=4096, L_ic=4, mlp_ratio 4      1,073,741,824      [I]   3.6% (was 2.7% @ mlp 2)
30B interconnect, D=4096, L_ic=8, mlp_ratio 4      2,147,483,648      [I]   7.2% -- ABOVE the 5%
                                                                            ceiling revision 1 named

--- W1, MEASURED 2026-09-02 [V] ----------------------------------------------
pre-pool capture verified bit-exact (max abs 0.00e+00) vs TextEncoder.forward()/IJEPA.encode()
region      pooled PR   token-global PR   ratio   per-item PR   verdict (bar: <=1.5x dead, >=2x ok)
code           42.63          28.09       0.66x       4.4        BET DEAD  (per-item flag <8 fires)
compress       45.33          35.57       0.78x       8.5        BET DEAD
retrieve       40.71          40.65       1.00x       5.1        BET DEAD  (per-item flag <8 fires)
vl_latent      18.07          23.54       1.30x       8.3        BET DEAD
cross-region pooled CKA, trained: max 0.336 (regions are distinct)
trained-vs-untrained pooled mean cosine: 0.84-0.98 -> 0.02-0.36 (training worked)
CAVEAT: the three text regions share seed=0 and identical configs, so their UNTRAINED models are
        the same weights and untrained cross-region CKA is 1.0 BY CONSTRUCTION -- not a baseline.
        Standing rule: vary the untrained seed per region in every future baseline.

--- RECEIPTS -----------------------------------------------------------------
code r@1 0.2285 -> 0.9766 | compress 0.0371 -> 0.7070 | retrieve 0.0000 -> 0.7480   [V*]
  ^ retrieve's untrained 0.0000 is not a baseline; W2c re-measures it [A13]
  ^ code's untrained 0.2285 conflicts with the "~0.40 lexical floor" prose in four files;
    W2c settles it and tau_lo is re-derived from the result [A23]
vl_latent probe top1 0.0335 -> 0.0606, transfer 0.1535 -> 0.2625, collapsed: false [V*]
  ^ all of these measure the EMA TARGET encoder (DEC-34)
reason 0.0039 -> 0.0801 (RECEIPT DELETED, prose only; W1b regenerates)             [V*]
banking77 0.0127 -> 0.8453 (deleted) | go_emotions 0.0497 -> 0.3642 (deleted)      [V*]
wall clocks: code 1087.2s | compress 426.7s | retrieve 595.1s | vl_latent 496.7s   [V*]
  ^ the basis for the phase-2 retrain budget, ~3-5 GPU-hours (§4.3)
PTQ (code): 9.79x compression, drop 0.0039, within_budget true, {3:16, 4:1}        [V*]
            3.2675 effective bits/param
stream geometry: mean pairwise cosine 0.89, entropy-effective rank 8.7 of 128      [V*]
gate collapse:  linear gate 1.0 / 0.0 on tinystories, stream_dim 16, batch 8       [V*]

--- CORPUS, AFTER THE aqua_rat RECOVERY [A5] ---------------------------------
244,761 declared                                                                  [V*]
  burned: go_emotions 43,410 + banking77 13,083 + gsm8k 7,473
        + aqua_rat 13,946  (UNION of THREE draws -- R, P, D3 -- DEC-42)      [V*]
                                                    was <= 9,964 (two draws) [V]/[I]
clean unallocated                                     165,065      was 169,047 (two-draw
                                                                   union), 83,328 (rev 1)
  fashion_mnist (no text bin can use it)               60,000
  TEXT-USABLE after apps<->code_contests dedupe       105,065      was 109,047 (two-draw),
                                                                   21,544 (rev 1)
required 67,584, of which 61,440 is training data     [DEC-49]   was 56,320 / 51,200
  as a share of what remains                            40.9%      was 40.0% (two-draw),
                                                                   33.3% (67.6% pre-recovery)
  vs the 61,440-row training need                        1.71x SURPLUS  (was 1.77x at the
                                                                        two-draw union, 2.13x
                                                                        at 51,200, 2.38x
                                                                        SHORTFALL before)
  MAX SOURCE SHARE (aqua_rat), unallocated              50.6%   BINDING; B1 hard line 0.50 -- FAIL
                                                                        (was 51.8% two-draw)
  MAX SOURCE SHARE (aqua_rat), text-usable              79.5%   the pool the reserve draws from
                                                                        (was 80.2% two-draw)
  fashion_mnist share of what remains                   36.3%      was 35.5% (two-draw),
                                                                   72.0%  (NOT binding)
  N_eff = 1/sum(p^2)                                     2.47      was 2.44 (two-draw), 1.79
                                                                        (B2 wants >= 3 --
                                                                        still short; §9.2)
DETERMINISM THAT MAKES THE RECOVERY POSSIBLE [V]:
  regions/pretrain.py:231   reservoir_sample(stream, limit, sampling_rng(seed, shards, columns))
  corpus.py:159-181         sampling_rng: seed + shard BASENAMES + columns
  corpus.py:189-221         Algorithm R; its only randomness is the rng it is handed
  regions/pretrain.py:177-193  _iter_pairs yields "in shard order"
  scripts/csd-train-all.py:325 ("reason/aqua_rat-raw/train.parquet", ("question","rationale"), 4982)
  W2a TESTS all three preconditions; if any fails, the write-off stands.

--- STATISTICS -----------------------------------------------------------------
SELECTION RULE (new in 3.3, section 2.7.3): the criterion's null rate is CAPPED at a
pre-committed 0.20 and k is the smallest value meeting it. A ceiling, not a comparison
between revisions -- a comparison ratchets forever. Both predecessors (0.377, 0.344)
EXCEED the ceiling, which is what stops a participant change from buying a pass.
P(Bin(10, 0.5) >= 7)  = 176/1024 = 0.1719  v1 G3' criterion, R=5, DEC-49            [I]
P(Bin(6,  0.5) >= 5)  =   7/64   = 0.109   either single-slip branch (W7v, or E1)   [I]
P(Bin(3,  0.5) >= 3)  =   1/8    = 0.125   the "3 of 3" rate if BOTH slip (§5.4)     [I]
P(Bin(12, 0.5) >= 8)  = 794/4096 = 0.194   A3's audio pre-specification, restated    [I]
  superseded, kept so a reader who saw an earlier revision is not silently handed a number:
P(Bin(10, 0.5) >= 6)  = 386/1024 = 0.377   revision 1's "6 of 10 pairs" -- REFUSED in 3.3
                                           by the selection rule: looser than what it replaced
P(Bin(6,  0.5) >= 4)  =  22/64   = 0.344   revision 2's "4 of 6 pairs", R=4          [I]
P(Bin(8,  0.5) >= 5)  =  93/256  = 0.363   revision 3.2's A3 pre-spec, R=4 base      [I]
binomial half-width 1.96*sqrt(0.25/n): +-4.33pp at n=512; +-6.1pp at n=256 (per bin,
  either side of the split -- UNCHANGED by DEC-49: more items, more bins, same 256 per bin);
  +-3.1pp at n=1,024 and +-2.5pp at n=1,536 (cross-faculty sealed, before/after DEC-49)  [I]

--- RESERVE SIZE, AFTER DEC-49 -------------------------------------------------
per-faculty eval, 6 bins x 512                             3,072      [I]   unchanged
cross-faculty eval, 6 reported bins x 512                  3,072      [I]   was 2,048 (4 bins)
compose eval total                                         6,144      [I]   was 5,120
  split                                        3,072 dev / 3,072 sealed  was 2,560 / 2,560
  of the sealed half, cross-faculty                        1,536      [I]   was 1,024
interconnect train, >=10x eval                            61,440      [I]   was 51,200
RESERVATION TOTAL                                         67,584      [I]   was 56,320
  as a share of the 165,065 clean unallocated               40.9%     [I]   was 40.0% (two-draw
                                                                             union), 33.3%
  surplus of 105,065 text-usable vs the 61,440 train need    1.71x    [I]   was 1.77x (two-draw
                                                                             union), 2.13x
  ^ tightens section 9.2's funding risk; it does not close it.
per-pair sealed counts sum to 2,304 against 1,536 sealed items -- shapes overlap pairs,
  and BOTH numbers are printed because one alone cannot be checked (S31-21).

--- EPISODIC STORE, CONTRACT PROVENANCE [DEC-49] -------------------------------
memory-gate (python)     HEAD 2c11c3f, worktree memory-gate-wt-p1-09,
                         branch feat/gateway-retrieve-domain (superset of P1-01..P1-09)
memory-gate-rs (rust)    HEAD 4b9f60d "hypha KV residency and RAM/disk tiered store"
  ^ every [V] in section 1.3's store cell is a file:line in ONE OF THESE TWO TREES,
    not in this one. Extraction held at scratchpad/memgate/memory-gate-contracts.md.
capacity units found in both repos                    item counts     [V]   NEVER bytes/VRAM
  python hot_cap default                                    256      [V]   tiered.py:40-51
  rust ram_max_items / disk_max_items                  256 / 4096    [V]   types.rs:458-473
  rust max_traces (VSA store)                             100,000    [V]   holographic_store.rs:37
the ONLY VRAM arithmetic in either repo, and it budgets WEIGHTS not the store:
  weight_budget_mib = vram_total - 2048 (display) - 1024 (scratch) - [2048 (gpu kv)]
  test-pinned: 16,303 MiB -> 11,183 (kv reserved) / 13,231 (not)     [V]   hypha.rs:114-122,152-157
eviction score (rust, adopted by CSD)  importance + 1.0 if Residency::Gpu; ties by older
  timestamp then key; lowest spills first                            [V]   tiered.rs:150-178
gateway backpressure, refusing not queuing            max_in_flight 32     [V]   memory_gateway.py:118-133
grep "overlay" / "differential" across both src/ trees      0 hits   [V]   gap (e): design fresh
grep Persona/Basin class across both src/ trees             0 hits   [V]   gap (b): spec-only, P1-13
b_store floor = eta/R * B_read = 0.03 * 256                  ~8 tokens     [I]   default 32, max 256

--- FLEET ----------------------------------------------------------------------
3090 Ti 24 GiB (.98) | 5080 16 GiB (.251, DEPLOYMENT TARGET) | 1080 Ti (.243, no train)   [V*]
1 Gb/s between training hosts; ~110 MB/s effective                                       [V*]
/data/models exported rw,no_root_squash to the whole /24 (OD-3)                          [V]
```
