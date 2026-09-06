# Why `memory` loses to BM25 on full-pool retrieval, and what to do about it

**Ask.** Diagnose the gap between `memory`'s held-out diagonal (recall@1 0.854) and its
full-pool BEIR result (recall@10 0.200 against BM25's 0.440), decide whether the DEC-02
union caused it, and recommend a structure. **Finding.** The objective was never aimed at
full-pool ranking; the union did not cause the gap and splitting it would make retrieval
worse. **Recommendation: (d)** — accept the union, retire gate (c) as a *region* gate, and
let the harness serve corpus RAG from an external retriever. Zero training runs.

## Scope and conventions

Date: 2026-09-06. Evaluation-only: no training run, no new matrix cell. Every number below
is either copied from a receipt with a `file:line` or `field` reference, or computed by
`scripts/run_diag.py` on GPU 0 (3090 Ti) and recorded in `results.json`. The committed
copy of the script differs from the one that produced `results.json` by eight `# noqa:
E402` comments added for the repo lint gate, and by nothing else. Tags: **[C]** computed
here, **[R]** read from a receipt, **[I]** an inference.

| short | file |
|---|---|
| `TAX` | `docs/design/REGION-TAXONOMY-AND-INTERCONNECT.md` |
| `TE` | `src/cogsyndelta/regions/text_encoder.py` |
| `MEM` | `src/cogsyndelta/regions/memory.py` |
| `PRE` | `src/cogsyndelta/regions/pretrain.py` |
| `BEIR` | `src/cogsyndelta/eval/beir_fiqa.py` |
| `W4R` | `docs/design/evidence/w4-production-runs-2026-09-03/memory-20260903T184441Z.json` |

The table maps each citation prefix to its file.

## 1. The verdict

The `memory` region is trained by symmetric InfoNCE over **in-batch negatives only**, where
the negative count *is* the batch size, so at `batch_size=1280` the hardest discrimination
any gradient ever asks for is "this passage, not these 1,279 randomly co-batched ones"
(`TE:183-234`, `TE:189-191`). Nothing in that loss ever puts a gradient on separating the
true FiQA passage from the other 57,637 passages of the real corpus, which is precisely the
discrimination a full-pool ranking scores and precisely the discrimination BM25 gets for
free from corpus-wide term statistics. The loss to a lexical baseline is therefore a
property of the **training objective**, not of the DEC-02 merge (§4), not of a damaged
checkpoint (§3 preamble), and not of an unfair battery — `BEIR:4-14`'s own docstring
declares the full-pool, graded-relevance shape a deliberately harder and intentionally
different measurement from the 512-pair diagonal that shares its metric names.

## 2. The evidence

Every arm was scored on one battery: `beir_fiqa.build_ranking_task("dev", pool="corpus")`,
500 dev queries, the full 57,638-passage FiQA pool, real qrels at 2.472 relevant per query,
at worktree HEAD `d9343e1` — an ancestor-compatible revision for both checkpoint code
revisions [C, `results.json:task_summary`].

**Table 1 — full-pool FiQA ranking, identical pool, qrels and code path for every arm [C].**

| arm | checkpoint sha (8) | code rev | split manifest | recall@1 | recall@10 | recall@100 | MRR |
|---|---|---|---|---:|---:|---:|---:|
| `memory` (DEC-02 union) | `fb3c53fa` | `eb735ab` | none (pre-G26) | 0.076 | **0.200** | 0.424 | 0.117 |
| `retrieve`-alone | `1351cf45` | `a769409` | none (pre-G26) | 0.034 | 0.094 | 0.238 | 0.055 |
| `compress`-alone | `e456293f` | `a769409` | none (pre-G26) | 0.030 | 0.074 | 0.224 | 0.048 |
| untrained (fresh init, seed 0) | — | `d9343e1` | — | 0.000 | 0.000 | 0.006 | 0.0005 |
| BM25 (lexical) | — | `d9343e1` | — | 0.240 | **0.440** | 0.662 | 0.308 |

`memory`'s row reproduces `W4R:retrieval` exactly, so the gate dispositions recorded at
`TAX:2651` stand unchanged: (b) passes on the floor, (c) fails against BM25, (d) passes
against random init. TF-IDF is not implemented in `BEIR` and was not computed [C].

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

**Table 2 — where the loss happens [C, derived from Table 1].**

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
only by `retrieve`-alone's own full-pool recall@10 clearing materially higher than
`memory`'s. It does not: **0.094 against 0.200**, less than half (Table 1). The union
retains *more* full-pool ranking capacity than the retrieval parent trained alone, so §9.7's
falsifier — *"the `memory` merge is worse than both parents… if it fails, keep them
separate"* (`TAX:5792-5795`) — is not tripped on this battery either.

The comparison is **not clean**, and three things make it unclean. The arms come from
different code revisions (`eb735ab` on `feat/w4-masked-token-loss` against `a769409`), all
three checkpoints predate G26's split-manifest pinning (`b67f47c`/`a81e739`, merged
2026-09-06) so none carries a split-manifest sha, and `W4R` carries
`corpus_fingerprint: None`, so "identical seeds as control" is inherited, not established.

Making it clean costs **two training runs and about one engineer-day**: retrain `memory`
and `retrieve`-alone at HEAD, same seed, same pinned split manifest, same corpus
fingerprint, then re-score both on this battery. The unit cost is measured — the production
`memory` run was `elapsed_s: 1586.7` on the 3090 Ti at batch 1280 / 4,000 steps [R, `W4R`]
— so two runs is roughly 0.9 GPU-hours plus eval. It is worth buying only if the
recommendation below is rejected; the effect it would resolve is a 0.106 gap in the
direction that already favours the union, and no decision in §6 turns on it.

## 5. Two use cases, side by side

`memory-gate` holds **two** embedding-adjacent objects, not one, and only the second
conflicts with dense passage retrieval. Object (1) is the episodic/semantic vector-store
index behind `retrieve_context` (`TAX:683`, contract clause 5) — that is DPR-shaped and
wants exactly what DPR wants. Object (2) is the persona/skill **differential**: overlays as
"differential weight/activation offsets, base canonical", composed as `W+ΔW` at matmul time
and unloaded by exact restore (`memory-gate-rs` `S07:77`, `S11:97-123`; SWITCH-shaped,
`TAX:554`).

**Table 3 — what each use case needs from an embedding; the last three rows are where they
are genuinely opposed [I, from the sources cited in the paragraph above].**

| property | memory-gate differential (object 2) | dense passage retrieval |
|---|---|---|
| dimensionality | fixed by the base tensors it offsets, per-parameter; not a free choice | one low-`d` vector per item (256 here), chosen for index cost |
| binding | meaningless unless fingerprint-bound to one base checkpoint | must be base-independent so the index survives |
| **normalisation** | **must not normalise — magnitude is the strength of the overlay** | **L2-normalised; cosine is the ranking function and magnitude is deliberately discarded** |
| **metric space** | **not required; needs additive composability and bit-exact reversibility** | **required; must stay discriminable across 57,638 candidates under one metric** |
| **update cost** | **must be cheap and local — paged in and out per persona at inference** | **global and expensive — changing the encoder invalidates and re-embeds the whole index** |
| surface | per-parameter and per-activation, per-token where activations are the target | pooled to one vector per passage (`TE:130-150`; W1 measured the token surface no richer) |

The fault line DEC-02 never tested is therefore not compress-versus-retrieve. It is
**retrieval-index embedding versus weight-differential overlay**, and a single InfoNCE
objective on a pooled, L2-normalised 256-d space can serve the first and structurally
cannot serve the second.

## 6. Recommendation

**Choose (d): accept the union, and let the harness use an external retriever for RAG.**
The operator's stated preference is the least complicated path that is still correct, and
(d) is the only option that is both. It is correct because §1 shows the shortfall is the
objective's, §4 shows the union is not the cause, and §5 shows corpus DPR is not the
capability the composed mind reads `memory` for — regions exchange latents (DEC-47,
`TAX:568`) and the in-model retrieval-into-context path is `episodic_store`, already a
required participant under DEC-49 (`TAX:570`). What (d) changes is a *claim*, not a
structure: gate (c) (`BEIR:664-677`, beats BM25) stops being a `memory` capability gate and
is re-declared as a harness-level RAG requirement, with the record saying plainly that
corpus retrieval at BM25 quality is bought from an index, not from a 16M-parameter encoder.

**Table 4 — cost of each option [I; GPU unit cost from `W4R:elapsed_s` = 1586.7 s per
b1280/4,000-step region run on the 3090 Ti].**

| option | training runs | engineer-days | verdict |
|---|---:|---:|---|
| (a) one region, two heads, unchanged | 0 | ~0.5 | correct but silent — leaves gate (c) failing with no recorded disposition |
| (b) split retrieval out as its own faculty | ≥3 (2 regions + W5 phase A at `R=6`) | 8–12 | **contra-indicated**: Table 1 shows the split *lowers* full-pool recall@10 from 0.200 to 0.094 |
| (c) keep one region, change the objective | ≥3 (control, treatment, seed replicate) | 4–6 | plausible mechanism (hard negatives, cross-batch bank), but must more than double recall@10 to clear BM25; unevidenced at this scale |
| **(d) accept the union, external retriever for RAG** | **0** | **~1** | **chosen** — doc amendment plus wiring the harness RAG path to an external index |

**Interconnect consequence.** (a), (c) and (d) all leave the participant set at `R = 5` with
ten ablation pairs, G3′ at ≥7 of 10 (null 0.1719) and the collapse floor at `η/R` = 3.0%
(`TAX:570`), so the region-set freeze that W5 and E2 depend on is untouched — and (d) alone
also leaves `memory`'s frozen checkpoint valid, where (c) invalidates it and forces W5 phase
A to re-run against a new checkpoint. (b) is the expensive one: it makes `R = 6`, re-opens
§5.4's pair arithmetic to fifteen all-pairs (or twelve under the non-pair-participant
convention `TAX:2795` pre-commits for exactly this case), restates G3′ and its null rate,
changes §2.3's white-matter parameter count and the collapse floor, and forces a W5 re-run.

## 7. What must be pre-registered before any training run that follows

Nothing here licenses a run. If (c) is chosen over the recommendation, or if §4's clean
comparison is bought, the run must be pre-registered against documents that already exist —
no new thresholds are invented by this diagnosis.

| requirement | existing source |
|---|---|
| the five W4 gates a–e, verbatim, with their dispositions declared in advance | `BEIR:589-819` (`w4_gates`) |
| the §9.7 falsifier for any change to the union | `TAX:5792-5795` |
| §4.0's retrain gate, both clauses (own-receipt metric does not regress > 1 point) | `BEIR:_RETRAIN_GATE_REGRESSION_MARGIN`; `TAX` §4.0 |
| arms share a seed and a pinned split manifest; corpus fingerprint recorded | G26 (`b67f47c`/`a81e739`); `[OP: identical-seeds-as-control]` |
| untrained baseline re-instantiated at a region-specific seed | DEC-34 (`TAX:2654`) |
| the read-out gate that replaced the invalidated 2.0x rank clause, if token-aware terms move | `docs/design/evidence/prereg-rank-gate-4000-2026-09-06/PREREG.md` |
| all accept/revert decisions on the dev half only | `TAX` §2.7.8 |

Each row names the document that already fixes the threshold; the pre-registration's job is
to state which arms are run and what each gate's failure means, not to restate the numbers.
