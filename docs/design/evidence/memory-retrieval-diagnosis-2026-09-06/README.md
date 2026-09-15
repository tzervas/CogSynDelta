# Why `memory` loses to a lexical baseline, and what the retrieval faculty should be

**Ask.** Diagnose the gap between `memory`'s held-out diagonal (recall@1 0.854) and its
full-pool BEIR result (recall@10 0.200 against BM25's 0.440), decide whether the DEC-02
union caused it, and recommend a structure. **Finding.** The objective was never aimed at
pool-scale ranking; the union did not cause the gap and splitting it would make retrieval
worse. **Recommendation: (d)** — keep the union as the memory encoder and consumer, add a
provenance-keyed retrieval router and the retrieve-distil-store-forget ingest path, and
gate any CSD-native store on a memory-store battery that does not yet exist. Zero training
runs to adopt; **(c)**'s objective change is a scheduled dependency, not an alternative.

## Scope and conventions

Date: 2026-09-06. Evaluation-only: no training run, no new matrix cell. Every number is
either copied from a receipt with a `file:line` or field reference, or computed by
`scripts/run_diag.py` / `scripts/run_smallpool.py` on GPU 0 (3090 Ti) and recorded in
`results.json` / `results_smallpool.json`. The committed scripts differ from the ones that
produced those files only by lint-gate cosmetics — `# noqa: E402` comments added to or
removed from the import block, and one blank line — and by nothing else. Tags: **[C]** computed here, **[R]** read from a
receipt, **[I]** an inference.

| short | file |
|---|---|
| `TAX` | `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` |
| `TE` | `src/cogsyndelta/regions/text_encoder.py` |
| `BEIR` | `src/cogsyndelta/eval/beir_fiqa.py` |
| `CKPT` | `src/cogsyndelta/regions/_checkpoint.py` |
| `W4R` | `docs/design/evidence/w4-production-runs-2026-09-03/memory-20260903T184441Z.json` |

The table maps each citation prefix to its file.

## 1. The verdict

The `memory` region is trained by symmetric InfoNCE over **in-batch negatives only**, where
the negative count *is* the batch size, so at `batch_size=1280` the hardest discrimination
any gradient ever asks for is "this passage, not these 1,279 randomly co-batched ones"
(`TE:183-234`, `TE:189-191`). Nothing in that loss ever puts a gradient on separating the
true passage from the rest of a real corpus, which is precisely the discrimination a
pool-scale ranking scores and precisely the discrimination BM25 gets for free from
corpus-wide term statistics. The shortfall is therefore a property of the **training
objective**, not of the DEC-02 merge (§4), not of a damaged checkpoint (§3), and not of an
unfair battery — `BEIR:4-14`'s own docstring declares the full-pool graded-relevance shape a
deliberately harder and intentionally different measurement from the 512-pair diagonal that
shares its metric names.

## 2. The evidence

Every arm was scored on one battery: `beir_fiqa.build_ranking_task("dev", pool=...)`, 500
dev queries, real qrels at 2.472 relevant per query, at worktree HEAD `d9343e1` — an
ancestor-compatible revision for both checkpoint code revisions [C].

**Table 1 — full-pool FiQA ranking, 57,638 passages, identical pool, qrels and code path
for every arm [C, `results.json`].**

| arm | checkpoint sha (8) | code rev | split manifest | recall@1 | recall@10 | recall@100 | MRR |
|---|---|---|---|---:|---:|---:|---:|
| `memory` (DEC-02 union) | `fb3c53fa` | `eb735ab` | none (pre-G26) | 0.076 | **0.200** | 0.424 | 0.117 |
| `retrieve`-alone | `1351cf45` | `a769409` | none (pre-G26) | 0.034 | 0.094 | 0.238 | 0.055 |
| `compress`-alone | `e456293f` | `a769409` | none (pre-G26) | 0.030 | 0.074 | 0.224 | 0.048 |
| untrained (fresh init, seed 0) | — | `d9343e1` | — | 0.000 | 0.000 | 0.006 | 0.0005 |
| BM25 (lexical) | — | `d9343e1` | — | 0.240 | **0.440** | 0.662 | 0.308 |

`memory`'s row reproduces `W4R:retrieval` exactly, so the gate dispositions at `TAX:2651`
stand: (b) passes on the floor, (c) fails against BM25, (d) passes against random init.
TF-IDF is not implemented in `BEIR` and was not computed [C].

A second battery was added by this diagnosis because §6 needs the lexical bar at
**memory-store scale**, not only at corpus scale: the same 500 queries and the same qrels
ranked against only the 1,236 passages judged in the dev split (`pool="split"`).

**Table 2 — small-pool FiQA ranking, 1,236 passages, same queries and qrels [C,
`results_smallpool.json`].**

| arm | recall@1 | recall@10 | recall@100 | MRR |
|---|---:|---:|---:|---:|
| `memory` | 0.254 | **0.504** | 0.792 | 0.341 |
| `retrieve`-alone | 0.118 | 0.362 | 0.664 | 0.198 |
| BM25 (lexical) | 0.458 | **0.718** | 0.874 | 0.549 |

Shrinking the pool 47× closes the gap to the lexical baseline but does not cross it:
`memory` reaches 70.2% of BM25's recall@10 at 1,236 candidates against 45.5% at 57,638. §6
depends on that: the deficit is not confined to corpus scale.

## 3. Ranking failure or representation failure

The checkpoint is intact, so neither answer is "drift". Re-running `pretrain_region`'s own
battery — `memory_config(steps=4000, batch_size=1280, max_len=96, holdout_pairs=512,
seed=0)` through `build_splits` → `evaluate`/`evaluate_graded`, no training — reproduced
`W4R` **bit for bit**: recall@1 0.853515625, recall@10 0.984375, MRR 0.9058414697647095,
graded spearman 0.7893104522457324 [C, `results.json:own_holdout_reproduction`].

The failure is **predominantly representation, with a smaller secondary ranking deficit**.
Reading recall@100 as "is the right passage anywhere in the query's neighbourhood at all",
`memory` misses entirely on 57.6% of queries; conditioning on the queries where it does
land in the top 100, it then places the passage in the top 10 only 47.2% of the time.

**Table 3 — where the loss happens, full pool [C, derived from Table 1].**

| arm | miss rate at top-100 (1 − r@100) | conditional sharpness (r@10 / r@100) |
|---|---:|---:|
| `memory` | 57.6% | 0.472 |
| `retrieve`-alone | 76.2% | 0.395 |
| `compress`-alone | 77.6% | 0.330 |
| BM25 | 33.8% | 0.665 |
| untrained | 99.4% | 0.000 |

Both columns are worse than BM25's, but the first is worse by 24 points and the second by
19, and the first is unrecoverable: no reranker can promote a passage that never enters the
candidate set. That is the signature of an embedding space that was never pushed to place
the true passage in the top 100 of 57,638, which is exactly what §1's mechanism predicts.

## 4. Did the DEC-02 union cause it

No. The pre-registered decision rule was that "the union hurt retrieval" is distinguishable
only by `retrieve`-alone's own pool-scale recall@10 clearing materially higher than
`memory`'s. It does not, on either battery: **0.094 against 0.200** full-pool (Table 1) and
**0.362 against 0.504** small-pool (Table 2). The union retains *more* ranking capacity than
the retrieval parent trained alone, so §9.7's falsifier — *"the `memory` merge is worse than
both parents… if it fails, keep them separate"* (`TAX:5792-5795`) — is not tripped.

The comparison is **not clean**, for three reasons. The arms come from different code
revisions (`eb735ab` on `feat/w4-masked-token-loss` against `a769409`); all three
checkpoints predate G26's split-manifest pinning (`b67f47c`/`a81e739`, merged 2026-09-06),
so none carries a split-manifest sha; and `W4R` carries `corpus_fingerprint: None`. So
"identical seeds as control" is inherited here, not established.

Making it clean costs **two training runs and about one engineer-day**: retrain `memory` and
`retrieve`-alone at HEAD, same seed, same pinned split manifest, same corpus fingerprint,
then re-score both on both batteries. The unit cost is measured — the production `memory`
run was `elapsed_s: 1586.7` on the 3090 Ti at batch 1280 / 4,000 steps [R, `W4R`] — so two
runs is roughly 0.9 GPU-hours plus eval. Buy it only if §6's recommendation is rejected: the
effect it would resolve is a gap in the direction that already favours the union, and no
decision below turns on it.

## 5. Two use cases, side by side

`memory-gate` holds **two** embedding-adjacent objects, not one, and only the second
conflicts with dense passage retrieval. Object (1) is the episodic/semantic vector-store
index behind `retrieve_context` (`TAX:683`, contract clause 5) — DPR-shaped, wanting exactly
what DPR wants. Object (2) is the persona/skill **differential**: overlays as "differential
weight/activation offsets, base canonical", composed as `W+ΔW` at matmul time and unloaded
by exact restore (`memory-gate-rs` `S07:77`, `S11:97-123`; SWITCH-shaped, `TAX:554`,
DEC-81 `TAX:602`).

**Table 4 — what each use case needs from an embedding; the middle three rows are where
they are genuinely opposed [I, from the sources cited above].**

| property | memory-gate differential (object 2) | dense passage retrieval |
|---|---|---|
| dimensionality | fixed by the base tensors it offsets, per-parameter; not a free choice | one low-`d` vector per item (256 here), chosen for index cost |
| **normalisation** | **must not normalise — magnitude is the strength of the overlay** | **L2-normalised; cosine is the ranking function and magnitude is deliberately discarded** |
| **metric space** | **not required; needs additive composability and bit-exact reversibility** | **required; must stay discriminable across tens of thousands of candidates under one metric** |
| **update cost** | **must be cheap and local — paged in and out per persona at inference (17 MB, 0.85 ms, `TAX` §6.6)** | **global and expensive — changing the encoder invalidates and re-embeds the whole index** |
| binding | meaningless unless fingerprint-bound to one base checkpoint | must be base-independent so the index survives |
| surface | per-parameter and per-activation, per-token where activations are the target | pooled to one vector per passage (`TE:130-150`; W1 measured the token surface no richer) |

The fault line DEC-02 never tested is therefore not compress-versus-retrieve. It is
**retrieval-index embedding versus weight-differential overlay**, and a single InfoNCE
objective on a pooled, L2-normalised 256-d space can serve the first and structurally cannot
serve the second.

## 6. Recommendation

### 6.1 The battery this diagnosis measured is the wrong instrument for the faculty

Stated plainly: **CSD does not have to be a dense retriever over external corpora to be a
good memory faculty.** The quality bar that matters is retrieving the right memory from
CSD's own memory store in its own differential format; the external-corpus job is a
different job. Full-pool FiQA measures the second and is being read as a verdict on the
first. **The memory-store battery does not exist**, so building it is a prerequisite to
judging this region at all, and every option below is unmeasured until it does.

### 6.2 The refinement: inherent, not tool-mediated, and what that costs

The operator prefers the capability **inherent** — CSD issues the retrieval, ingests and
updates itself — rather than delegated to a tool. The consequence is direct: **an inherent
retriever means CSD embeds the corpus itself, which puts its own retrieval quality back on
the critical path for every collection it indexes**, so Tables 1 and 2 matter again for that
path rather than being retired as off-target.

The settled arrangement is a **router keyed on store provenance**, where every store
declares the model that embedded it, and three paths follow.

| store provenance | path | what runs |
|---|---|---|
| CSD-native (CSD format, differentials against the same baseline CSD is running) | query, ingest and update directly — it is simply an external memory-gate store | CSD's own encoder |
| not CSD-compliant, embedder identified and available | pass-through: retrieve with the model that embedded it, because the retrieval algorithm follows the embedding space | that model, resident alongside CSD |
| not CSD-compliant, embedder unavailable or undeclared | **refuse** | nothing |

Once a result is in CSD format it lives in the context window when transient, or is written
to a CSD-native external store when it must persist.

### 6.3 Two consequences, and whether the current code satisfies them

**Consequence one: format alone is too weak a compatibility test.** A delta only
reconstructs against the baseline it was taken from, so a store must record its baseline **by
content hash** and the router must compare baselines, not just formats. **The current code
cannot satisfy this.** No store-side embedder or baseline declaration exists anywhere in
`src/cogsyndelta/memory/` — `grep -rn "embedding_model\|embedder"` over that package returns
nothing [C]. The primitive to build on does exist and should be reused rather than rebuilt:
`load_checkpoint` verifies a manifest SHA-256 *before* the file is opened (`CKPT:54`), and
DEC-40 makes it the single loader in the repo (`TAX:561`). What is missing is the store-side
field and the router that reads it, which is engineering, not research.

**Consequence two: this makes baseline stability a requirement,** since changing the base
weights invalidates every differential store built against the old baseline unless it is
migrated. **The frozen-base-plus-overlays design guarantees detection, not stability.**
DEC-81 freezes base faculty weights *on the persona path* and makes overlay load the unfreeze
(`TAX:602`), and `TAX` §6.6 clause 3 requires that an overlay **refuse** to attach to a base
fingerprint it was not trained against (`TAX:4295-4299`) — fail-closed, so a stale store
raises instead of silently reconstructing wrong. That is refusal, not validity. Two gaps
remain: the string `migrat` appears **zero times** in the whole taxonomy [C], so no migration
path is specified anywhere; and the freeze is scoped to the persona path, while DEC-50 step 3
(whole-mind unified training, `TAX:571`) unfreezes regions **by design**, which changes the
base and invalidates every differential store built against it. So the design *intends*
baseline stability and currently *guarantees* only a loud failure.

**The pass-through embedder is a second resident model, and therefore a packing and cap
problem.** DEC-54 admits concurrent jobs only when their per-job VRAM budgets fit the target
card under process-level pseudo-isolation, and requires every receipt to record `host`,
`vram_budget_mib` and `concurrency` (`TAX:575`). A resident foreign embedder is a
subtraction from the same pool that memory-gate's `weight_budget_mib` formula already
subtracts display reserve, CUDA scratch and the KV reserve from (`TAX:683`, clause 7) — and
no existing budget line accounts for it. **The router must refuse rather than query a foreign
space with the wrong model**, which is the same shape as memory-gate's refusing backpressure:
raise synchronously at submit time, never queue and never degrade (`TAX:683`, clause 3).

### 6.4 Which collections CSD is currently good enough to embed itself

Using the lexical baseline as the bar it must clear, the answer at both measured scales is
**none of them**.

**Table 5 — CSD against the lexical bar, by pool size [C, Tables 1 and 2].**

| pool | candidates | `memory` recall@10 | BM25 recall@10 | ratio | clears the bar |
|---|---:|---:|---:|---:|---|
| full FiQA corpus | 57,638 | 0.200 | 0.440 | 0.455 | no |
| FiQA dev split | 1,236 | 0.504 | 0.718 | 0.702 | no |
| CSD-native memory store | — | not measured | not measured | — | **battery does not exist** |

So: heterogeneous external prose collections at corpus scale are firmly out of reach, and
small curated collections of order 10³ are still 21 points short. The gap narrows as the pool
shrinks, so a crossover may exist below ~1,200 candidates, but it is **not measured** and
must not be assumed. **Until the memory-store battery exists, the honest statement is that no
collection has been shown to be safe for CSD to embed itself**, and the pass-through path
therefore carries real load rather than being a fallback.

**Does the region also fail at retrieving its own memories?** Unknown, and the evidence cuts
both ways. It scores 0.854 recall@1 on its own 512-pair diagonal, but that pool has one gold
per query and is drawn from the training corpus; at the smallest *graded, held-out* pool
measured it is already 21 points under lexical. This is an open risk, not a reassurance, and
if the battery shows the region fails at its own memories then **every option below is
insufficient without fixing the region** — which is §6.6's dependency, promoted from optional
to required.

### 6.5 The retrieve-distil-store-forget pipeline, as its own capability

This is a named capability with its own contract, not a side effect of the router.

| stage | rule |
|---|---|
| **retrieve** | router-dispatched per §6.2; the query is issued by CSD, not by an external caller |
| **select** | scored relevance with **both** a threshold and a per-query budget — the threshold rejects a weak best hit, the budget bounds cost when many hits are strong; both pre-registered, neither inferred |
| **distil** | convert the selected records into memory-gate memories under the `(scope, domain, logical_key)` partition (`TAX:683`, clause 1), taking the strictest input licence (DEC-33) |
| **store/forget** | discard the raw payload; retain the **source identifier** and a **content hash** of the payload so a memory can be re-verified or refreshed without having kept it |
| **duplicate** | same source id and same content hash ⇒ no new memory; bump importance and `last_accessed`, which the eviction score already reads (`TAX:683`, clause 2) |
| **contradiction** | same source id, **different** content hash ⇒ the source changed: supersede, retaining the prior memory's hash as provenance. Different source, contradictory content ⇒ **keep both and mark the conflict**; silently overwriting is how a store launders a disagreement into a fact |

The content hash is what makes forgetting safe: without it, a discarded payload cannot be
distinguished from a refreshed one, and duplicate-versus-contradiction is undecidable.

### 6.6 The choice and its cost

**Choose (d): keep the union as the memory encoder and consumer, add the provenance-keyed
router and the ingest pipeline, and treat full-pool dense retrieval over foreign corpora as
served by the pass-through path rather than by this region.** The operator's stated
preference is the least complicated path that is still correct, and (d) is the only option
that is both — but it is correct *only* with §6.4's dependency stated: the CSD-native path is
not trustworthy until the memory-store battery exists, and if that battery fails, (c)'s
objective change becomes required rather than optional.

**Table 6 — cost of each option [I; GPU unit cost from `W4R:elapsed_s` = 1586.7 s per
b1280/4,000-step region run on the 3090 Ti].**

| option | training runs | engineer-days | verdict |
|---|---:|---:|---|
| (a) one region, two heads, unchanged | 0 | ~0.5 | correct but silent — leaves gate (c) failing undisposed and answers nothing about inherent retrieval |
| (b) split retrieval out as its own faculty | ≥3 (2 regions + W5 phase A at `R=6`) | 8–12 | **contra-indicated**: the split *lowers* recall@10, 0.200→0.094 full-pool and 0.504→0.362 small-pool |
| (c) keep one region, change the objective | ≥3 (control, treatment, seed replicate) ≈ 1.3 GPU-h | 4–6 | plausible mechanism (hard negatives, cross-batch bank); must add ~21 points at small pool and more than double recall@10 at corpus scale — **scheduled as (d)'s dependency, not as its alternative** |
| **(d) union + provenance router + ingest pipeline** | **0 to adopt** | **~6–9, plus ~2 for the memory-store battery** | **chosen** — router, store baseline field, ingest pipeline, and the battery that gates the CSD-native path |

**Interconnect consequence.** (a), (c) and (d) all leave the participant set at `R = 5` with
ten ablation pairs, G3′ at ≥7 of 10 (null 0.1719) and the collapse floor at `η/R` = 3.0%
(`TAX:570`), so the region-set freeze that W5 and E2 depend on is untouched — and (d) alone
also leaves `memory`'s frozen checkpoint valid, where (c) invalidates it and forces W5 phase
A to re-run against a new checkpoint. (b) is the expensive one: it makes `R = 6`, re-opens
§5.4's pair arithmetic to fifteen all-pairs (or twelve under the non-pair-participant
convention `TAX:2795` pre-commits for exactly this case), restates G3′ and its null rate,
changes §2.3's white-matter parameter count and the collapse floor, and forces a W5 re-run.

## 7. What must be pre-registered before any training run that follows

Nothing here licenses a run. If (c) is taken up, or if §4's clean comparison is bought, the
run must be pre-registered against documents that already exist — no new thresholds are
invented by this diagnosis.

| requirement | existing source |
|---|---|
| the five W4 gates a–e, verbatim, with their dispositions declared in advance | `BEIR:589-819` (`w4_gates`) |
| the §9.7 falsifier for any change to the union | `TAX:5792-5795` |
| §4.0's retrain gate, both clauses (own-receipt metric does not regress > 1 point) | `BEIR:_RETRAIN_GATE_REGRESSION_MARGIN`; `TAX` §4.0 |
| arms share a seed and a pinned split manifest; corpus fingerprint recorded | G26 (`b67f47c`/`a81e739`); `[OP: identical-seeds-as-control]` |
| untrained baseline re-instantiated at a region-specific seed | DEC-34 (`TAX:2654`) |
| the memory-store battery's own gates, defined before it is run and before it is used to judge the region | new document, shaped on `BEIR`'s gate block; DEC-50 step 1 discipline (`TAX:571`) |
| the selection threshold and per-query budget of §6.5, stated as numbers before ingest runs | §6.5 above; NSRS τ convention (DEC-36, `TAX:557`) |
| overlay attach/detach receipts if any store work touches the persona path | DEC-81 (`TAX:602`); P5′o gates (`TAX:5042`) |
| all accept/revert decisions on the dev half only | `TAX` §2.7.8 |

Each row names the document that already fixes the threshold; the pre-registration's job is
to state which arms are run and what each gate's failure means, not to restate the numbers.
