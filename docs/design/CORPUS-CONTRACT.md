# Corpus Design Contract

**Status:** design. This informs decisions; it does not make them. No catalogue entry,
training script or region config is modified by this document, and no dataset was
downloaded to write it.

**Scope:** what each region's training corpus must contain given what that region is *for*;
what the composed CSD model needs that no region provides; and a balance rule that can be
checked instead of argued about.

**Grounding.** Every proportion, count and length statistic below is cited to one of four
places, and nothing is asserted from a doc that a measurement contradicts:

| source | what it is |
|---|---|
| `/mnt/bulk/csd-corpus-analysis/analysis.json` | the 2026-09-02 corpus survey — measured balance, near-duplicate and length statistics |
| `scripts/csd-train-all.py` (`REGIONS`, `VL_REGIONS`) | what each region **actually** trains on |
| `config/mind/csd-regions.json`, `src/cogsyndelta/contracts/region_spec.py` | each region's **declared** purpose |
| `docs/design/LICENCE-FOR-OPEN-WEIGHTS.md` | licence verdicts at mirror **and** upstream |

Two live surveys reported into this document while it was being written; their results are
attributed inline as *item-3 code survey* and *item-5 retrieval survey*. Both returned
**negative** results for the sources this contract most needs, and those negatives are
recorded as findings rather than smoothed over.

---

## The defect this document exists to prevent

A region gets overtrained on one narrow thing while carrying a broad name. It has seen a
little of the general material inside its declared scope, so it does not look broken — it
looks like a slightly weak version of the thing its name promises. Measured instances today,
all four of them:

| region | name promises | corpus actually is | measured |
|---|---|---|---|
| `retrieve` | query → passage rank | a short-web-question matcher | 77.8% GooAQ pre-dedup, **79.1% post-dedup** |
| `code` | docstring ↔ function | a Python **signature** matcher | 1 language; **93.9%** of code-side tokens truncated at `max_len=96` |
| `compress` | neighbours stay neighbours | an image-caption / 5-genre NLI matcher | 58.2% SNLI / 41.8% MultiNLI, positives average **9.6 tokens** |
| `vl_latent` | latent visual reasoning | a 200-class 64×64 object encoder | tiny-imagenet alone, N_eff = 1.00 |

This is not only an honesty problem. It is a **routing** problem. `csd-regions.json` gives
each region a `router_trigger`, and the assembled mind dispatches on the shared stream. A
region whose corpus does not cover what its trigger claims will be handed work it cannot
serve, and — because the gate is learned rather than written — the failure surfaces as a
diffuse quality loss across the mind, not as an error attributable to one region.

One nuance worth stating precisely, because the shorthand overstates it: the router in
`src/cogsyndelta/poc/router.py` is a learned `nn.Linear` softmax top-k gate over the stream.
It does not dispatch on the *string* `"code"`. What dispatches on names is everything
around it — `router_trigger` in the catalogue, region selection in a client, and the
operator reading a receipt. The learned gate will happily learn to route Python-shaped
streams to `code`; the problem is that a client, a doc or a person will read `code` and send
it Rust.

---

# Part 1 — Per region

Each region gets: declared **purpose** (quoted), what that purpose **requires**, what it
**trains on today** (measured), the **gap**, a **target composition with per-source caps**,
and the **licence status** of each proposed source under the MIT open-weights target.

**A cap is a balance decision, not a size limit.** Every cap below names the imbalance it
prevents. A cap that cannot name one is a size limit wearing a cap's clothes.

Licence verdict vocabulary is `LICENCE-FOR-OPEN-WEIGHTS.md`'s, unchanged: `PERMISSIVE_OK`,
`ATTRIBUTION`, `SHARE_ALIKE`, `BLOCKING`, `BLOCKING (unresolved)`, `UNCLEAR`.

---

## 1.1 `code`

**Declared purpose** — `config/mind/csd-regions.json`:

> `"role": "Docstring <-> function, search and generation. Explicitly NOT next-token over all of GitHub."`
> `"router_trigger": "language=python and a docstring is present."`

**The name and the trigger already disagree, and the trigger is the honest one.** The role
line says "docstring ↔ function" without a language; the trigger says `language=python`.
`scripts/csd-train-all.py`'s note says `"docstring <-> function; NOT next-token over
GitHub"`. Only the trigger admits the region is Python-only. That disagreement is the whole
Part 1 defect in one region, visible in the config file without measuring anything.

### What the purpose requires

A region a router may send *any* code to needs, concretely:

- **Languages.** At least the languages a caller can plausibly ask for. A region trained
  only on Python docstring pairs cannot serve a router asking for Rust: it has never seen
  `///` doc comments, `impl` blocks, lifetimes, or `Result<T, E>` as a return convention.
  It will return the nearest Python function, confidently.
- **Natural-language sides that are not all docstrings.** "Search and generation" implies
  queries that look like issue text, commit messages and API prose, not only
  reStructuredText/Google-style docstrings. A region trained exclusively on docstring-format
  anchors learns the *format* as much as the semantics.
- **Lengths that reach a function body.** The objective is docstring ↔ *function*. If the
  encoder only ever sees the first ~96 tokens, the objective it actually optimises is
  docstring ↔ *signature*.
- **A difficulty spread.** Trivial one-line getters and multi-branch algorithms are
  different retrieval problems. A corpus that is mostly boilerplate teaches the easy one.

### What it trains on today

`REGIONS["code"]` in `scripts/csd-train-all.py`:
`region/code/codesearchnet-python/**/*.parquet`, columns `("docstring", "code")`, cap `0`
(uncapped), `max_len` 96. One source.

| measured | value | where |
|---|---|---|
| rows | 455,243 | `analysis.json §3.code_codesearchnet.total_rows` |
| distinct repos | 13,581 | same |
| largest single repo share | **2.441%** (`saltstack/salt`, 11,111 rows) | same |
| top-10 repo share | 6.246% | same |
| duplicates removed by `build_splits` | 23,800 | `§2.build_splits_reproduction_summary.code` |
| train / holdout pairs | 430,931 / 512 | same |
| code side, tokens | mean 485.6, p50 293, p90 999, p99 3,113, max 27,342 | `§4.code.code_token_len` |
| **code side over `max_len=96`** | **93.938%** | same |
| docstring side, tokens | mean 110.4, p50 49, p90 256, p99 946 | `§4.code.docstring_token_len` |
| docstring side over 96 | 31.555% | same |
| near-empty docstrings (<10 chars) | 810 | `§4.code` |
| most repeated raw docstring | `"auto generated code"` ×1,777 | `§4.pre_dedup_boilerplate.code` |
| holdout items with a train near-dup at cos ≥ 0.90 | 24.2% (124/512) | `§2.code.holdout_vs_train` |
| receipt metric | recall@1 0.9766 | `LICENCE-FOR-OPEN-WEIGHTS.md`, receipts 2026-09-02 |

Source balance: **1 source, 100% share, N_eff = 1.00.**

Two things this table settles that the older prose got wrong, and they cut in opposite
directions:

- **The corpus is internally diverse.** Top repo 2.441% over 13,581 repos is a
  well-spread source. `REMAINING.md` P0.9 already carries this correction: the
  "1,626 of 3,000 rows are pandas" figure was a pre-shuffle head artefact, not a property
  of the corpus. Within-source balance for `code` **passes**.
- **The truncation is real and it is the dominant defect.** 93.9% of code-side sequences
  exceed the window. recall@1 0.9766 is a measurement of an encoder that saw a `def` line
  and a line or two of body.

### The gap, plainly

`code` is a **Python docstring-to-signature matcher** carrying the name `code`. It is
monolingual, its natural-language side is exclusively docstring-formatted, and 93.9% of its
positive side is cut off before the function body starts. Its within-source repo balance is
good; every other axis of the contract is unmet.

### Target composition

The language row counts below are real (reported by the *item-3 code survey*, from
`google/code_x_glue_ct_code_to_text`), which is why the caps can be computed. **The licence
column is why none of it can be used yet.**

| language | available (CodeSearchNet lineage) | cap | share at cap | why this cap |
|---|---|---|---|---|
| Python | 281,000 | 27,600 | 16.67% | prevents the region reverting to a Python model with a code name — the exact defect being fixed. Python has 10.2× Ruby's data and would take 27.9% uncapped |
| PHP | 268,000 | 27,600 | 16.67% | second-largest; uncapped, Python+PHP alone are 54.6% of the corpus and B1 fails |
| Go | 183,000 | 27,600 | 16.67% | equal cap; no evidence any language deserves more until a measured demand distribution exists |
| Java | 181,000 | 27,600 | 16.67% | as above |
| JavaScript | 65,200 | 27,600 | 16.67% | as above |
| Ruby | 27,600 | 27,600 | 16.67% | the binding constraint — the smallest language sets the uniform ceiling |
| **total** | 1,005,800 | **165,600** | max 16.67%, **N_eff 6.00** | |

Uncapped, the same six languages give max share 27.94% and N_eff 4.56 — which *passes* the
balance rule in Part 3. The stricter uniform cap is recommended anyway because the failure
mode here is not statistical dominance, it is a router sending Ruby to a region that saw
2.7% Ruby. A looser variant, cap 100,000, gives 492,800 pairs at max 20.29% / N_eff 5.40 and
is the right choice if total corpus size turns out to bind quality; both pass, and the
choice between them is a measurement, not a preference.

**Two further caps that are not about language:**

- **Docstring-derived anchors ≤ 60% of the corpus.** Prevents the region learning docstring
  *format* as a proxy for meaning. There is no clean source for the other 40% today —
  `bigcode/commitpackft` has per-row licences and 702k rows but its pairs are commit-message
  ↔ diff, which is a different objective, not a different flavour of this one.
- **`max_len` becomes a schedule, not a value.** 96 → 256 → longer, per P11.4. At 96 the
  cap on information is 93.9%, which no source balance can fix. This is the single change
  with the largest measured headroom in the region.

### Licence status of the proposed sources

**No permissively licensed multi-language docstring↔function corpus exists off the shelf.**
The *item-3 code survey* verified zero datasets as permissive at both mirror and upstream,
and found the reason is structural:

> Datasets that yield (NL ↔ code) pairs have **no per-row licence column**. Datasets that
> have a per-row licence column yield **no pairs** (raw files only).

| candidate | licence observed | verdict |
|---|---|---|
| `Nan-Do/code-search-net-python` (in use) | mirror `apache-2.0` over upstream `other`; no LICENSE file; ~15% GPL/AGPL by repo resolution | **BLOCKING** |
| `google/code_x_glue_ct_code_to_text` | tagged `c-uda`; same CodeSearchNet lineage; no licence field | **BLOCKING (unresolved)** |
| `semeru/code-text-python` | tagged `MIT`; same lineage; no licence field | **BLOCKING (unresolved)** |
| `bigcode/the-stack-v2` | dataset tag `other`; per-row `detected_licenses` + `license_type` | usable **only as construction input** |
| `codeparrot/github-code` | `other`; per-row `license`; 115M files, 32 langs | usable **only as construction input** |
| `bigcode/commitpackft` | per-row `license`, 702k rows | **rejected** — commit↔diff, wrong objective |
| `codeparrot/github-code-clean` | `apache-2.0` at dataset level over per-repo-licensed content including GPL/AGPL | **UNCLEAR** — tag/content mismatch, unresolved |

The same CodeSearchNet data appears as `c-uda`, `MIT` and `apache-2.0` across three mirrors
while its upstream is `license: other`. That is a fourth instance of the mirror-vs-upstream
pattern this project already has eight recorded cases of.

**The only route to a provably permissive multi-language corpus is construction**: extract
docstring/function pairs from `the-stack-v2` filtered on `license_type == "permissive"`, or
from `codeparrot/github-code` filtered on `license`. That is a build, not a swap, and it
inherits a term that should be recorded explicitly because it is unusual for a training
corpus:

> *"Any use of all or part of the code gathered in The Stack must abide by the terms of the
> original licenses, including attribution clauses when relevant."*

plus a live opt-out regime (`github.com/bigcode-project/opt-out-v2`) whose stated effect is
that opting out *"excludes your code from the next iteration of The Stack and from current
and future model training"*, and a term requiring users to update to the latest version as
removals are enacted. **That is a standing obligation on a frozen training corpus** — a
corpus you may have to re-derive after publication. It does not block use; it is a
maintenance cost that must be decided knowingly.

The cheap partial repair already documented in `LICENCE-FOR-OPEN-WEIGHTS.md` remains
available and is unaffected by this: filter the existing Python corpus by its surviving
`repo` column against resolved GitHub licences, yielding ~324,000 pairs (71.3%). That fixes
the licence for Python. It does not fix the monolingualism, and it must not be reported as
if it did.

---

## 1.2 `compress`

**Declared purpose** — `config/mind/csd-regions.json`:

> `"role": "Neighbors stay neighbors in a short latent."`
> `"router_trigger": "Sentence pair or embedding to reconstruct."`

### What the purpose requires

"Neighbours stay neighbours" is a statement about a *metric* surviving a bottleneck. For
that to mean anything the corpus must span what the router will hand it:

- **Lengths.** A region asked to compress a paragraph must have seen paragraphs. If its
  longest positive is 60 tokens, the latent has never been asked to hold a paragraph's
  worth of structure.
- **Registers.** Neighbourhood in image-caption English is not neighbourhood in legal prose
  or in a stack trace. A single-register corpus produces a latent that is metric-preserving
  on that register only.
- **A relation that is actually the objective.** The project already learned this the
  expensive way: training on `pair-class` unfiltered — an even three-way split of
  entailment/neutral/contradiction — held `compress` at 0.26, because two-thirds of its
  "positives" were not positives. Paraphrase is likewise not entailment.
- **Graded pairs**, so that "neighbours" can be measured as a correlation rather than only
  as a top-1 hit.

### What it trains on today

`REGIONS["compress"]`: `region/compress/all-nli/pair/train*.parquet`, columns
`("anchor", "positive")`, cap `0`, `max_len` 96. One source, config pinned to `pair`
deliberately (a `**` glob across all-nli's four configs is what caused the 0.26 run).

| measured | value | where |
|---|---|---|
| rows before split | 314,315 | `§3.compress_allnli_pair.total_rows_before_split` |
| SNLI-derived | 183,030 — **58.231%** | same (exact match against the raw SNLI corpus) |
| MultiNLI-derived | 131,285 — **41.769%** | same, **by subtraction, not by label** |
| SNLI containment in all-nli | **100.0%** | `§1.snli_vs_allnli_pair.exact_overlap_fraction` |
| duplicates removed | 36,534 | `§2.build_splits_reproduction_summary.compress` |
| train / holdout | 277,269 / 512 | same |
| **anchor == positive in train** | **646** (0.233%) | same |
| anchor tokens | mean 18.6, p99 65, max 267; 0.178% over 96 | `§4.compress.anchor_token_len` |
| positive tokens | mean **9.6**, p99 25, **max 60**; 0.000% over 96 | `§4.compress.positive_token_len` |
| holdout items with train near-dup ≥ 0.90 | 8.98% (46/512) | `§2.compress.holdout_vs_train` |
| receipt metric | recall@1 0.7070 | receipts 2026-09-02 |

Source balance: **2 provenance groups, max share 58.2%, N_eff (inverse Simpson) 1.95.**

Two measurement caveats that must travel with those numbers:

- **There is no source column.** all-nli ships flattened; the 58/42 split is *inferred* by
  exact matching against raw SNLI and attributing the remainder to MultiNLI by subtraction.
  It is a good inference — the arithmetic closes to the row against the upstream splits
  (183,416 SNLI + 130,899 MNLI entailment = 314,315) — but it is not a measurement of a
  label. A corpus you cannot audit by reading a column is a corpus whose balance you can
  only reconstruct.
- **The graded gate never ran.** `config.graded_shards` is `[]` (P0.9c), so the STS-B
  correlation that was supposed to check "neighbours stay neighbours" as a *graded* property
  has never produced a number. `sentence-transformers/stsb` is fetched and unused.

### The gap, plainly

`compress` is a **short-sentence entailment matcher over image captions and five MultiNLI
genres**, carrying a name that promises a general-purpose latent bottleneck. Its positives
average 9.6 tokens and top out at 60 — so 96 is not the constraint; the *corpus* is. Its
mix cannot be verified from the data, 0.23% of its training pairs are free wins where
anchor equals positive, and the gate designed to measure its actual claim never executed.

### Target composition

| axis | requirement | cap | why this cap |
|---|---|---|---|
| relation: entailment | the objective | ≥ 50% | below half, the region is being retrained onto a different relation while keeping the name. This is the 0.26 lesson, expressed as a number |
| relation: paraphrase | adjacent, not identical | ≤ 30% | paraphrase pairs are easier and more numerous; uncapped they would dominate and the entailment structure would be diluted |
| relation: graded / STS | needed for the real metric | ≥ 5%, held out | enough to compute a correlation, not enough to shift the training distribution |
| length band < 32 tok | short | ≤ 50% | today this band is ~100% of positives. Capping it is the whole point |
| length band 32–128 tok | medium | ≥ 25% | the band the router will actually hand it most often |
| length band > 128 tok | long | ≥ 15% | a "compress" region must have compressed something long at least once |
| register | no single register | ≤ 40% each | image captions are ~58% today (SNLI premises are Flickr30k captions). Prevents a caption-metric latent |
| `anchor == positive` | degenerate | **0** | 646 today. A free InfoNCE win that teaches nothing; a one-line filter |

### Licence status of the proposed sources

| candidate | licence | rows | shape | verdict |
|---|---|---|---|---|
| `sentence-transformers/all-nli` (in use) | none at mirror; SNLI CC BY-SA 4.0; MultiNLI four licences at once | 314,315 | entailment | **BLOCKING as trained**; repairable to `SHARE_ALIKE` by re-deriving from `nyu-mll/multi_nli` with `genre` kept |
| `hkust-nlp/SynCSE-scratch-NLI` | **MIT**, dataset and repo | 275,579 | `sent0`/`sent1`/`nli_hard` | **PERMISSIVE_OK** — the only like-for-like swap found. Synthetic (GPT-4/3.5 generated), which is a different unknown, not the absence of one |
| `google-research-datasets/paws` | `other`, grant quoted: *"may be freely used for any purpose"* | ~323k positives | word-order/entity edits | usable; a **narrow** notion of similarity |
| `ltg/en-wiki-paraphrased` | Apache-2.0 | 5,145,408 | paraphrase | `original` column **is** Wikipedia text; card silent on Wikipedia's CC BY-SA. Also **collides with `retrieve`'s Wikipedia lineage** — see Part 2 |
| `HuggingFaceM4/COCO` captions | `cc-by-4.0` | 414,010 pairs | two captions of one image | **ATTRIBUTION**; re-introduces the caption register the caps above are trying to dilute |
| `sentence-transformers/stsb` | fetched, unused | — | graded | fills the graded row; licence not re-audited here |

**Traps already on record, repeated so they are not re-proposed:**
`hkust-nlp/SynCSE-partial-NLI` (MIT tag, but sentences come from SimCSE's SNLI+MNLI),
`lxyuan/synthetic-nli-triplet`, `SeanLee97/all_nli_angle_format_b` (permissive tags over
all-nli repackagings), `facebook/anli` (**CC BY-NC 4.0**), `facebook/xnli` and
`nyu-mll/glue:mnli` (both are MultiNLI).

**The honest gap: no permissive source of *long-form* entailment pairs was identified.** The
>128-token band in the table above has no candidate. SynCSE-scratch preserves the objective
but not the length spread. So the length cap is currently a target with no supplier, and
saying so is the point of writing it down.

---

## 1.3 `retrieve`

**Declared purpose** — `config/mind/csd-regions.json`:

> `"role": "Query or claim -> passage rank."`
> `"router_trigger": "A query plus candidate passages."`

Note `claim`. Nothing in the current corpus is claim-shaped.

### What the purpose requires

- **Query types.** Natural questions, keyword queries, and *claims* (the declared trigger).
  These are different distributions; a claim is a declarative sentence to be verified, not
  an interrogative to be answered.
- **Passage lengths.** Short answer strings and long documents. Ranking a 20-token answer
  is a different problem from ranking a 500-token passage.
- **Domains.** Web QA, product search, scientific, financial, multilingual. The region's
  own gate has historically been a financial-domain set, so finance is not optional.
- **One-to-many relevance preserved.** Real IR judges several passages relevant per query.
  A pipeline that keeps one destroys the structure the metric is defined over.

### What it trains on today

`REGIONS["retrieve"]`: three sources — `fiqa-pairs/train.parquet` `("query","passage")` cap
0; `natural-questions/**/train*.parquet` `("query","answer")` cap 0; `gooaq/**/train*.parquet`
`("question","answer")` **cap 400,000**. `max_len` 96.

| measured | value | where |
|---|---|---|
| pre-dedup source counts | gooaq 400,000 / NQ 100,231 / FiQA 14,131 | `§2.build_splits_reproduction_summary.retrieve` |
| pre-dedup shares | **77.766% / 19.486% / 2.747%** | `§3.retrieve_sources.pre_dedup_source_share_pct` |
| duplicates removed | 8,634 | `§2.build_splits_reproduction_summary.retrieve` |
| train / holdout | 505,216 / 512 | same |
| query tokens | mean 10.1, p99 17, max 36; **0.000%** over 96 | `§4.retrieve.query_token_len` |
| passage tokens | mean 73.9, p99 293, max 3,231; **14.727%** over 96 | `§4.retrieve.passage_token_len` |
| most repeated raw query | a FiQA question ×23 | `§4.pre_dedup_boilerplate.retrieve` |
| **holdout items with train near-dup ≥ 0.90** | **53.71%** (275/512) | `§2.retrieve.holdout_vs_train` |
| same at ≥ 0.95 / ≥ 0.98 / ≥ 0.99 | 32.42% / 14.84% / 7.62% | same |
| max-sim distribution | mean 0.898, p50 **0.911**, p90 0.986, p99 0.997 | same |
| receipt metric | recall@1 0.7480 | receipts 2026-09-02 |

Pre-dedup balance: **max share 77.77%, N_eff (inverse Simpson) 1.55.**

**Post-dedup, it is worse, and the arithmetic identifies exactly why.** `build_splits`
deduplicates on the *anchor* fingerprint, keeping the first occurrence. FiQA's anchor is the
query, and FiQA judges ~2.6 passages relevant per question — `src/cogsyndelta/regions/retrieve.py`
records that anchoring on the query collapses 14,131 judgements to **5,498** rows. That
collapse alone accounts for `14,131 − 5,498 = 8,633` removed rows against a measured
`duplicates_removed` of **8,634**. Essentially the entire dedup loss for this region *is*
FiQA's one-to-many structure being destroyed. Reconstructing the post-dedup mix on that
basis gives 505,729 rows against a measured 505,728 — a one-row discrepancy:

| source | post-dedup rows (reconstructed) | share |
|---|---|---|
| gooaq | 400,000 | **79.094%** |
| natural-questions | 100,231 | 19.819% |
| fiqa-pairs | 5,498 | **1.087%** |

**Post-dedup: max share 79.09%, N_eff 1.50.** The cap that exists for balance is being
partly undone by a dedup rule that only bites the smallest source. This is P0.9e stated as
a balance consequence rather than a data-loss one.

**Two further defects in how the mix is built, both visible in the code:**

1. **The cap is a prefix, not a sample.** `load_pairs` returns as soon as `limit` is
   reached, iterating shards in sorted order. GooAQ's 400,000 is therefore the *first*
   13.3% of a 3,012,496-row corpus in file order, and the unconditional shuffle in
   `build_splits` happens **after** the cap, so it cannot repair a non-representative
   prefix. Whatever ordering gooaq's shards happen to have is what defines the corpus.
2. **The holdout is not what the comment says it is.** `csd-train-all.py` states:
   *"Held-out eval stays on fiqa dev/test, which is a different domain (financial QA) and
   therefore measures transfer rather than memorisation."* `build_splits` shuffles the
   concatenated pool and takes `all_pairs[: cfg.holdout_pairs]` — a uniform sample of the
   *mixture*. Expected FiQA content of the 512-pair holdout: **≈ 5.6 items** post-dedup
   (≈ 14 pre-dedup). recall@1 0.7480 is an in-mixture number on a ~79%-GooAQ holdout. The
   transfer test described in the comment is not implemented on this path. This is the
   worked example in Part 3.

   A second, *separate* code path does implement a real IR evaluation:
   `src/cogsyndelta/regions/retrieve.py` (present in the tree, untracked as of writing)
   trains on FiQA `train` only and ranks held-out queries BEIR-style against the full
   57,638-passage FiQA corpus, with BM25 and an untrained encoder alongside. That is a
   sound design — but it is **FiQA-only training**, i.e. a different corpus from
   `csd-train-all.py`'s three-source mix, and its module docstring says so:
   *"FiQA … are the only sets this region trains on."* Two `retrieve` training regimes now
   exist in the tree with different corpora. Neither implements "train on the mix, evaluate
   on FiQA".

### The gap, plainly

`retrieve` is a **short English web-question matcher**. It has never seen a claim, its
queries are 10 tokens on average with a p99 of 17, 14.7% of its passages are truncated, and
its only out-of-domain source has been reduced to 1.1% of the corpus by a dedup rule.
53.7% of its holdout has a near-duplicate in training. And the transfer property that
justified the mix is described in a comment but not implemented in the code.

**Not all of that 53.7% is a defect,** and `REMAINING.md` P0.9b already separates the two
kinds correctly:

- genuine paraphrase — *"how do you know if…"* vs *"how to know if…"*, cos 0.993. Real
  leakage.
- GooAQ template collision — *"44 is 25 percent of what number?"* vs *"…55 percent…"*,
  cos 0.995. **Same template, different correct answer.** That is not duplication; it means
  the encoder cannot distinguish a slot value. Deduplicating it away would hide a real
  weakness rather than fix one.

Concentrating further on GooAQ makes the second category worse while making the metric look
better. That is the most dangerous shape a corpus change can have, and it is the reason the
cap must go *down*, not up, even if GooAQ's licence question resolves favourably.

### Target composition

Using the P2.4 mix (audited at mirror **and** upstream), with caps:

| source | available | cap | share at cap | domain | why this cap |
|---|---|---|---|---|---|
| `allenai/gooaq` | 3,112,679 | 150,000 | 23.44% | web QA | at 400k it is 79%; at 150k no single source can carry a majority of the metric. Also limits the template-collision population that inflates the holdout |
| `tasksource/esci` | 2,027,874 | 150,000 | 23.44% | product search | keyword-shaped queries, absent today. Capped equal to gooaq so the corpus does not simply swap one dominant web source for one dominant commerce source |
| `castorini/mr-tydi` | ~167,000 | 150,000 | 23.44% | multilingual | near-full use; the cap binds only slightly and exists to keep the four large sources equal |
| `THUIR/T2Ranking` | 258,000 q | 150,000 | 23.44% | Chinese web | equal cap; **conditional on licence, see below** |
| `miracl/miracl` | ~40,000 q | 40,000 | 6.25% | multilingual | uncapped — it is already below the others' ceiling. Flagged honestly as under-scale rather than padded |
| **total** | | **640,000** | max **23.44%**, **N_eff 4.47** | | |

For comparison, the same five sources uncapped give max share 55.53% and N_eff 2.26 — a
mix that would still be a GooAQ-and-ESCI corpus.

**Caps that are not about source:**

- **Dedup must become pair-level for one-to-many sources, not anchor-level.** Anchor-only
  dedup is correct for a corpus where one anchor has one positive; it silently deletes the
  structure of a corpus where it does not. Measured cost: 8,633 of 14,131 FiQA judgements.
  This is a change to `build_splits`, which this document does not make.
- **Passage `max_len` needs the same schedule as `code`.** 14.7% truncation is far less
  severe than `code`'s 93.9%, but the region is a *passage* ranker and p99 is 293 tokens.
- **Claim-shaped queries: ≥ 10%.** The declared trigger says "query or claim". Nothing in
  any proposed source is claim-shaped, so this row currently has no supplier — see below.

### Licence status of the proposed sources

| source | licence, both ends | verdict |
|---|---|---|
| `allenai/gooaq` | LICENSE file is stock Apache-2.0; **README line 5 says** *"This dataset should not be used for any commercial purposes"* | **BLOCKING (contested)** — unresolved |
| `tasksource/esci` | Apache-2.0; `amazon-science/esci-data` LICENSE read | **PERMISSIVE_OK** |
| `castorini/mr-tydi` | Apache-2.0, built on tydiqa which is also Apache-2.0 | **PERMISSIVE_OK** |
| `miracl/miracl` | Apache-2.0 | **PERMISSIVE_OK**, under-scale |
| `THUIR/T2Ranking` | Apache-2.0 in README **only**; LICENSE file 404s | **verify before use** |
| `sentence-transformers/natural-questions` (in use) | mirror untagged; upstream `cc-by-sa-3.0` | **SHARE_ALIKE** |
| fiqa-pairs (in use) | `cc-by-sa-4.0` both ends | **SHARE_ALIKE** |
| `allenai/peS2o` | mirror `odc-by`; upstream `allenai/s2orc` returns 401 — **not independently verifiable** | permissive at mirror, upstream unverified; **unpaired** — 38.97M documents, not query→passage pairs |

The GooAQ contradiction is **not resolved by this document**; it is owned by a separate
research fork and is a one-email question to AI2. Until it resolves, the largest source in
the recommended mix is contested, and the mix's total permissive-verified yield is
`esci + mr-tydi + miracl ≈ 340,000` pairs at a cap of 150k each.

**Rejections recorded so they are not re-proposed** (*item-5 retrieval survey*, all checked
at upstream):

| candidate | why rejected |
|---|---|
| `embedding-data/PAQ_pairs` (7,287,980) | card: QA pairs are **CC-BY-SA**; `facebookresearch/PAQ` README adds *"the majority of the PAQ code is licensed under CC-BY-NC"* |
| `nthakur/swim-ir` (2,817,012, 27 languages) | HF API licence `cc-by-sa-4.0`. The largest multilingual pair set found, and share-alike |
| `mandarjoshi/trivia_qa` (913,686) | the Apache premise did not hold: the only licence-relevant text is *"The University of Washington does not own the copyright of the questions and documents included in TriviaQA."* A disclaimer, not a grant |
| `microsoft/wiki_qa` (29,258) | `other` = MSR Data License Agreement; text not retrievable, terms unconfirmed |
| `sentence-transformers/amazon-qa` (2,507,114) | no licence at mirror, none at the UCSD source page — genuinely licence-silent |
| `sentence-transformers/stackexchange-duplicate-questions` | upstream is the Stack Exchange dump, CC BY-SA |
| `sentence-transformers/specter` (1,064,240) | title pairs not passages; no licence tag on the data (the Apache-2.0 covers `allenai/specter`'s code) |

**Two capability losses that must be stated, not smoothed:**

1. **There is no permissively licensed financial-domain retrieval source.** The survey found
   none. If FiQA is dropped for licence reasons, finance leaves the corpus entirely — and
   with it the out-of-domain signal the mix was designed around. The replacement is *not*
   "keep FiQA as eval-only": `LICENCE-FOR-OPEN-WEIGHTS.md` establishes that the eval-only
   argument is sound only where it is verified that no evaluated-on parameter is published,
   and it is weakened wherever the eval feeds a gate — which `retrieve`'s does.
2. **`mteb/*` is under-surveyed, not cleared.** The retrieval survey's WebSearch budget
   capped mid-work. "Nothing new found" is a real result for the named candidates; the
   `mteb/*` catalogue specifically has not been searched.

---

## 1.4 `vl_latent`

**Declared purpose** — `config/mind/csd-regions.json`:

> `"role": "Latent visual reasoning. Consumes visual latents, NOT tokens; predicts representations rather than reconstructing pixels. Enters the same shared stream as the text regions, which is why the [B, D] activate surface is modality-agnostic."`
> `"router_trigger": "Stream carrying visual latents."`

### What the purpose requires

I-JEPA needs no labels, so the corpus requirements are scale, **domain diversity**,
resolution, and a probe that is not the pretraining set. Plus two requirements the program
adds and the current corpus cannot express:

- **A resolution schedule** (P11.5): small crops early, larger views later.
- **Composite frames** (P11.6): many elements in one view. If the deployment target is a
  model that reads a display, a screenshot *is* a collage, and training on isolated 64×64
  tiles is the artificial case.

### What it trains on today

`VL_REGIONS["vl_latent"]`: train `vl/tiny-imagenet/data/train-*.parquet`; in-domain probe
`vl/tiny-imagenet/data/valid-*.parquet`; transfer probe `vl/cifar100/cifar100/test-*.parquet`.

| measured | value | where |
|---|---|---|
| images | 100,000 | receipts 2026-09-02 |
| sources | 1 (**N_eff 1.00**) | `VL_REGIONS` |
| params | 22.9M | receipts |
| probe top-1 | **0.0606** on 200 classes | receipts |
| VRAM used | ~3 GB of 24 GB | `REMAINING.md` P3 |

### The gap, plainly

Single source, single domain, and **unreleasable**: tiny-imagenet is ImageNet-derived with
no licence anywhere in the chain and ImageNet's non-commercial Terms of Access as the only
terms text present. There is no clean fraction to keep. The in-domain probe is defined on
the same corpus, so replacing the pretraining data also replaces the gate metric — only the
CIFAR-100 transfer number survives the change, and CIFAR-100 has its own licence vacuum.
`nlphuji/flickr30k`, which P3.2 plans to add, is **BLOCKING on the same grounds** and would
give the region a second licence problem rather than fixing its first.

### Target composition

From `LICENCE-FOR-OPEN-WEIGHTS.md`'s replacement-vision search, with the caps as stated
there:

| domain | source | cap | of available | share | why this cap |
|---|---|---|---|---|---|
| general photography | `nyuuzyou/pxhere` | 100,000 | ≈1.1M | 20.12% | uncapped it is 60%+ of the pool and the encoder becomes a pxhere encoder that saw a little of everything else |
| histopathology | `1aurent/PatchCamelyon` | 100,000 | 327,680 | 20.12% | equal cap; a visually alien domain that would otherwise be second-largest |
| synthetic 3D render | Shapes3D | 100,000 | 480,000 | 20.12% | equal cap; native 64×64, so no resize loss |
| synthetic 3D scene | CLEVR | 100,000 | 100,000 (all) | 20.12% | binding at its own size |
| garment product photo | Fashion-MNIST | 70,000 | 70,000 (all) | 14.08% | binding at its own size |
| satellite land-use | `timm/eurosat-rgb` | 27,000 | 27,000 (all) | 5.43% | binding; **the smallest domain and the reason the strict variant exists** |
| **total** | | **497,000** | | max 20.12%, **N_eff 5.41** | |

**Strict variant** — cap everything to EuroSAT's ceiling: 6 × 27,000 = **162,000**, max
16.67%, **N_eff 6.00**. Closer to tiny-imagenet's original 100,000-image scale, perfectly
uniform, and the only variant that also passes a within-corpus uniformity check. Both are
cheap to run; choosing between them is a measurement.

**Why the cap *is* the sampler today, and this is not optional.** `vl_pretrain.py` samples
batches with `torch.randint` over a single tensor produced by concatenating every shard with
**no domain weighting**. Per-domain exposure in each batch is therefore exactly proportional
to raw row count. A naive union at full sizes (pxhere ≈1.1M, PatchCamelyon 327,680,
Shapes3D 480,000, CLEVR 100,000, Fashion-MNIST 70,000, EuroSAT 27,000) would put EuroSAT
under 2% and Fashion-MNIST around 3% of every batch. That is not a risk; it is the
predictable behaviour of the code as written, and it is precisely the failure this contract
is named after. Capping shard sizes fixes it with **zero code changes**.

Two adjacent defects worth recording with the composition, because they will otherwise be
attributed to the corpus:

- `_to_float` hardcodes ImageNet's mean/std (`[0.485,0.456,0.406]`/`[0.229,0.224,0.225]`)
  and applies it to every image regardless of source. Histopathology stain colour, satellite
  RGB composites and line-drawing black-and-white do not share those statistics.
- Whether mixing six domains in one I-JEPA run at ~23M parameters helps or fragments the
  encoder is **not settled** by the literature and should be measured, not assumed. The
  licence document proposes the three-arm experiment (naive union / balanced union /
  staged→combined) plus a domain-identity probe that reuses `_linear_probe` unchanged.

**Probes, held out entirely from pretraining:** Quick Draw (CC BY 4.0, 345 classes) as
primary transfer; Caltech-256 (CC BY 4.0 via CaltechDATA) as the closest surviving
natural-object-photography probe; EuroSAT's own test split as the in-domain sanity check.

### Licence status

All six pretrain sources are **PERMISSIVE_OK** at mirror and upstream (pxhere CC0,
PatchCamelyon CC0, Shapes3D Apache-2.0, Fashion-MNIST MIT, EuroSAT MIT with a Copernicus
attribution notice for the underlying Sentinel imagery); CLEVR is **ATTRIBUTION** (CC BY 4.0
for the images; its generation code is separately BSD). Quick Draw and Caltech-256 are
**ATTRIBUTION**.

**The capability loss, stated plainly, quoting the audit rather than paraphrasing it:**

> there is no permissively-licensed replacement of tiny-imagenet's specific breadth —
> balanced, dense, natural-object-category photography at 100k+ scale — because that
> combination of properties essentially doesn't exist outside the ImageNet lineage under a
> clean licence.

The composite buys breadth **across** visually distinct domains at the cost of depth
**within** everyday-object photography — the one domain most vision transfer literature
measures. A `vl_latent` pretrained on this composite should be **expected to be measurably
weaker at fine-grained natural-object recognition** than the current unreleasable model.
That is the price of the licence-clean requirement, not a rounding error.

---

## 1.5 `classify` — planned

**Declared purpose: none exists.** `config/mind/csd-regions.json` contains six regions —
`residual_mlp`, `stream_vae`, `code`, `retrieve`, `compress`, `vl_latent`. There is no
`classify` entry, so there is **no `role` line and no `router_trigger`**. The region is
planned in `REMAINING.md` P2.2/P2.3 and has catalogue entries pointing at it, and that is
all. The name is currently doing all the work.

**That is the first requirement of this contract for `classify`: write the spec before the
corpus.** A region whose purpose is undeclared cannot have its corpus checked against its
purpose, which is the only thing this document does.

### What a `classify` region would require

- **A label space the router can name.** "Classify" is not a task; classifying *into what*
  is.
- **Both single-label and multi-label** shapes, since real requests are both.
- **Enough classes that top-1 is informative** — a binary classifier and a 77-way classifier
  are different regions wearing one name.
- **An explicit out-of-label-space population.** Without a `none of these` class the region
  will label everything, confidently, including streams that belong to another region. This
  is the single most important requirement and nothing staged supplies it.

### What is staged (nothing trained)

| source | rows | classes | balance | licence |
|---|---|---|---|---|
| `PolyAI/banking77` | 10,003 train (13,083 with test) | 77 intents | min 35 / max 187; largest class **1.87%** vs uniform 1.30%; max:min **5.34:1** | **ATTRIBUTION** (CC BY 4.0, GitHub LICENSE agrees) |
| `google-research-datasets/go_emotions` | 43,410 | 28 labels, **multi-label** | largest label **32.76%** (14,219), smallest **0.177%** (77); max:min **184.7:1** | **PERMISSIVE_OK** (Apache-2.0) |

*(`§3.classify_banking77`, `§3.classify_go_emotions`; manifests under `/mnt/bulk/csd-corpus/classify/`.)*

Source balance of the staged pool: **max share 81.27%, N_eff 1.44.**

### The gap, plainly

Two datasets with **incompatible label spaces** (banking intents vs emotions), one
single-label and one multi-label, at 81/19. Combined naively they are a 105-way mixture that
is really two problems. `banking77`'s within-source class balance is **good** (5.34:1 across
77 classes). `go_emotions`' is **bad** — one label is a third of the corpus and the smallest
is 77 rows, a 184.7:1 ratio, which fails the within-source rule in Part 3 before a single
step is trained. And there is no out-of-label-space population anywhere.

### Target composition

| axis | requirement | why |
|---|---|---|
| label spaces | ≥ 4 distinct, none > 40% of rows | two spaces at 81/19 means the region is a go_emotions model with a classify name |
| per-space class balance | max class share ≤ max(0.25, 2/k); max:min ≤ 20:1 | `go_emotions` fails both today; either rebalance by capping the dominant label or declare the region's metric per-class |
| shape | both single-label and multi-label present | the router cannot know which it is handing over |
| **`none of these`** | **≥ 10% of rows**, drawn from outside every label space | the open-set requirement. Nothing staged supplies it; see Part 2's general bin |

### Licence status

`banking77` **ATTRIBUTION**, `go_emotions` **PERMISSIVE_OK**. **No further classification
source has been licence-audited by this project.** The two additional label spaces the
target asks for have no candidate list, and this document does not invent one.

---

## 1.6 `reason` — planned

**Declared purpose: none exists**, exactly as for `classify`. No entry in
`config/mind/csd-regions.json`, therefore no `role`, no `router_trigger`.

### What a `reason` region would require

- **Multi-step problems with checkable answers** — the objective needs a ground truth that
  is verifiable, not merely plausible.
- **More than one reasoning shape**: arithmetic word problems, algebra, deduction, multi-hop
  factual. These are different competencies and a router will not distinguish them.
- **A difficulty spread**, so the region is not trained entirely on problems it can solve.
- **Rationales**, if the objective is anything richer than answer matching.

### What is staged (nothing trained)

| source | rows | shape | balance | licence |
|---|---|---|---|---|
| `deepmind/aqua_rat` (`raw`) | 97,467 | algebraic MCQ with natural-language rationale | answer letters C 22,290 / B 21,446 / A 20,494 / D 19,441 / E 13,796 — max **22.87%**, max:min **1.62:1** | **PERMISSIVE_OK** (Apache-2.0) |
| `openai/gsm8k` (`main`) | 7,473 | grade-school arithmetic word problems, step-by-step | — | **PERMISSIVE_OK** (MIT) |

*(`§3.reason_aqua_rat_answer_letter`; manifests under `/mnt/bulk/csd-corpus/reason/`.)*

Source balance of the staged pool: **aqua_rat 92.88% / gsm8k 7.12% — max share 92.88%,
N_eff 1.15.**

### The gap, plainly

**`reason` fails the balance rule before it is trained, and by more than any region that
exists.** 92.9% of the staged pool is one source. The corpus contains two reasoning shapes
where the region needs at least three, and the two present are 93/7. Answer-letter balance
*within* aqua_rat is fine (1.62:1). The deduction and multi-hop shapes have no clean
supplier: the MATH family is DMCA-encumbered and already REJECTED by the catalogue
(*"a live copyright dispute is not cured by a permissive tag on a repackaging"*), and
HotpotQA — the obvious multi-hop source — is `SHARE_ALIKE` **and** collides with `retrieve`'s
Wikipedia lineage (Part 2).

**This is the value of writing the contract before P2.3 runs**: the imbalance is visible now,
from staged manifests, at zero GPU cost. After training it would be visible only as a
receipt that looks fine.

### Target composition

| shape | cap | why |
|---|---|---|
| algebraic MCQ (`aqua_rat`) | ≤ 40% | at 92.9% the region is an AQuA-RAT model with a reason name |
| arithmetic word problems (`gsm8k`) | ≥ 20% | the shape most distinct from MCQ in the staged set; the catalogue's own `why` calls it *"a different shape to GSM8K"* in reverse |
| deduction / logical entailment | ≥ 20% | **no clean source identified** |
| multi-hop factual | ≥ 20% | **no clean source identified that is not lineage-collided** |

At a 40% cap on aqua_rat with gsm8k at its full 7,473, the two staged sources alone yield
`7,473 / 0.60 × 0.40 ≈ 4,982` aqua_rat rows — i.e. **the balance rule caps the staged reason
corpus at about 12,455 rows**, an 88% reduction. That is the honest arithmetic: with only two
sources at these sizes, balance and scale are in direct conflict, and the resolution is more
sources, not a bigger cap.

### Licence status

`gsm8k` **PERMISSIVE_OK** (MIT), `aqua_rat` **PERMISSIVE_OK** (Apache-2.0),
`hendrycks/competition_math` **REJECTED** (DMCA, access disabled on HF). Two of the four
required shapes have **no licence-clean candidate at all**.

---

# Part 2 — The composed model's own corpus

This is the half nobody has specified, and it is the half the architecture depends on.
Everything in Part 1 refines a component that already exists. Part 2 specifies something
that does not.

## 2.1 What the composed model needs that no region provides

Every region corpus is single-objective by construction: a docstring/function pair, an
entailment pair, a query/passage pair, an unlabelled image. Four things the composed model
needs are absent from all of them **individually and from their union**.

### (a) Cross-region items — the only material that can demonstrate composition

`program/csd-program.json` gates P5 on *"composed beats best single region on a mixed set"*.
On a bin-balanced corpus of single-bin items that gate is **nearly unfalsifiable**: the
baseline is one region applied to the whole set, so it scores well on its own bin and badly
elsewhere, while the composed model merely has to route correctly. Passing it measures the
**router**, not composition.

What actually needs demonstrating is that regions *compose* — that the mind can serve a
request no single region can. That requires items whose correct answer needs two
competencies:

- `retrieve` × `code` — given a natural-language issue, find the function that implements
  the behaviour.
- `retrieve` × `reason` — multi-hop: retrieve, then combine.
- `vl` × `classify` — locate an element in a frame, then name it.
- `compress` × `retrieve` — collapse a candidate set, then rank it.

**No region corpus contains a single such item.** They cannot: a region corpus is defined by
its objective, and a cross-region item has two.

### (b) A general / out-of-scope population, and the fallback path that has no corpus

`residual_mlp`'s declared trigger is literally *"Tokens not tagged code/retrieve/compress"*
— it is the fallback. Its declared corpus is `"synthetic"`, objective `"reconstruction"`,
with the note *"Synthetic tokens only via train-route. No public corpus bound yet."*

**So the region that handles everything the specialists do not has never seen real text.**
The router has never been shown a boundary case or an out-of-scope stream, because every
batch it could be trained on is drawn from one specialist's homogeneous corpus. A gate
trained only on clean in-region material learns that every input belongs to some region.

`config/mind/csd-regions.json`'s own notes place the foundation corpora (FineWeb/C4/Pile) at
**step 4** of the curriculum, and P6 confirms *"after P4 and P5, never before"*. That is the
right ordering for foundation training — but it leaves the general bin empty for P4 and P5,
which are exactly the phases that need it. This is a real ordering tension, not a mistake:
the composed model needs *some* general material at step 3 without that being foundation
training at step 1. The quantity required is small (see §2.5); the licence status is
unknown (see §2.7).

### (c) Interleaved batches with a load-balancing signal

Region training draws every batch from one source. Composition requires batches whose
consecutive items route to different regions, so the gate receives a load signal and the
shared stream does not drift toward whichever region moved last. P12.1 names this
(multi-task / interleaved training). It is partly a sampler property — but it cannot be
delivered by sampling over the union of region corpora, because that union is exactly the
data the regions memorised, which is §2.2.

### (d) A held-out set that no component has seen

Which is the rest of Part 2.

## 2.2 Disjointness: two different guarantees, and only one is negotiable

The requirement "the composed corpus must not overlap what any region trained on" contains
two distinct guarantees that are worth separating, because they have different costs and
different failure modes.

**Item-level disjointness.** No compose item appears in any region's training split. Cheap,
mechanically checkable, and **mandatory for every compose item, train and eval alike**.
Violating it means the composed model is scored on rows its components memorised.

**Distribution-level separation.** The compose corpus is drawn from sources no region
trained on *at all*. Stronger, more expensive, and the only thing that supports the claim
"generalises across its specialisations" — because a held-out slice of a source a region
trained on tests memorisation of *items*, not transfer across *distributions*.

**The commitment this contract makes:**

> **Item-level disjointness is mandatory everywhere. Distribution-level separation is
> mandatory for the composed EVAL set and optional (indeed usually undesirable) for the
> composed TRAIN set.**

The reasoning: the composed model must be *good at* what its regions do, so training it on
material from the regions' domains is correct. But if the eval is drawn from those same
distributions, "the composed model generalises" is untestable — every number would be
consistent with the regions having memorised their domains and the router having learned to
dispatch. Only a distribution-separate eval distinguishes the two.

## 2.3 Enforcement — four stages, all reusing tooling that already exists

Nothing below invents a method. Stages 1 and 2 are the two techniques already used in
`/mnt/bulk/csd-corpus-analysis/`; stage 0 is the licence audit's mirror→upstream discipline
applied to provenance instead of licences; stage 3 is stages 1–2 re-run against reality.

### Stage 0 — provenance, at catalogue time, before anything is fetched

Reject a compose candidate whose **upstream lineage** intersects any region source's
lineage, regardless of whether text overlaps. This is free, and the survey demonstrates it
catches things text-matching misses:

| burned lineage | why it is burned | measured evidence | compose-ineligible |
|---|---|---|---|
| **Wikipedia article pool** | Natural Questions is 19.8% of `retrieve` train | SQuAD/HotpotQA share **95.93%** of article titles (424 of 442) while sharing only **0.116%** of exact contexts (22 of 18,891) | SQuAD, HotpotQA, TriviaQA, SWIM-IR, `ltg/en-wiki-paraphrased` |
| **Flickr30k captions** | Flickr30k captions → SNLI premises → 58.2% of `compress` train | SNLI is **100.0%** contained in all-nli | Flickr30k (images and captions), SNLI, any Flickr30k-caption derivative |
| **GooAQ / Google answer boxes** | 79.1% of `retrieve` train post-dedup | — | any Google-snippet-scraped QA set |
| **CodeSearchNet's 13,581 repos** | `code`'s entire corpus | the `repo` column survives and lists them | any Stack-family or GitHub-scraped code corpus, **unless filtered by repo name against that list first** |
| **ImageNet** | tiny-imagenet is 100% of `vl_latent` | ImageNet → Tiny ImageNet → Maysee → zh-plus chain | STL-10, Imagenette/Imagewoof, any ImageNet subset |

Note what stage 0 costs: it removes the most obvious compose candidates. HotpotQA is the
natural multi-hop source for a `retrieve`×`reason` cross-bin set, and it is burned twice
over — Wikipedia lineage and `SHARE_ALIKE`.

### Stage 1 — exact overlap after normalisation

`cogsyndelta.eval.metrics.assert_no_contamination(region_train_texts, compose_texts,
tolerance=0.0)` — zero tolerance, which is the function's own default and should stay it.
The fingerprint is blake2b-128 over whitespace-collapsed lowercased text, so trivial
reformatting cannot hide a duplicate.

Two normalisation traps must be supplied to this stage, both measured, both invisible to a
naive hash:

- **Strip structured prefixes before fingerprinting.** CodeContests prefixes every
  description with `"<id>_<letter>. <Title> - "`. Exact-match overlap between APPS and
  CodeContests: **2 rows**. Embedding overlap at cos ≥ 0.90: **1,784 of 10,000 (17.84%)** —
  892× larger. A hash-only check would have declared them disjoint.
- **Match on the structural key where one exists.** SQuAD/HotpotQA: exact context overlap
  0.116%, **title** overlap 95.93%. Dedupe by title, not by passage hash.

### Stage 2 — semantic near-duplicate screen

Method taken verbatim from `analysis.json §2`: embed both sides, L2-normalise, chunked
cosine matmul, report counts at thresholds {0.90, 0.95, 0.98, 0.99} plus the per-item
max-similarity distribution (mean / p50 / p90 / p99 / max).

**Encoder choice is the part that must not be got wrong,** and the survey's own methodology
note states the rule:

> a model trained ON a corpus must not be used to judge that corpus's QUALITY … Cross-dataset
> checks (SQuAD/HotpotQA, APPS/CodeContests) used encoders NOT trained on either side of the
> pair, so those are not even redundancy-circular.

So: **run the screen pairwise, per region, each with a region encoder trained on neither
side.** compose-vs-`code` uses the `retrieve` encoder; compose-vs-`retrieve` uses `code` or
`compress`. Where no such encoder exists for a pairing, the result is reported as circular
and is **advisory only** — it may not be used to pass a candidate, only to fail one.

**Thresholds calibrated from measurement, not chosen.** The non-circular reference on record
is SQuAD/HotpotQA scored with the `retrieve` encoder over 18,265 title-matched candidates:
mean 0.464, p50 0.472, p90 0.698, **p99 0.883**. So under a non-circular encoder, the p99 of
a *related but distinct* population sits below 0.90, and the 156 items above 0.90 were
genuine verbatim shared passages (top matches at cos 1.0000). Therefore:

- **≥ 0.90 = flag**, **≥ 0.98 = treat as duplication** — under a non-circular encoder.
- **Domain-mismatch correction, also measured.** The `code` encoder applied to
  competitive-programming prose produced max-sim mean 0.831 / p50 0.835 — the whole
  distribution shifted up, and the survey flagged it as a weaker signal for exactly that
  reason. A fixed 0.90 over-flags there. Rule: recompute the threshold as the **p99 of a
  known-disjoint control population under the same encoder**, and use `max(0.90, that p99)`.

**Gate:** distinct compose items with a near-dup in any region's train split at the
calibrated threshold must be **< 1%**. That number is chosen against measured reference
points rather than rounded to look strict — the region holdouts already on disk sit at
**53.7%** (`retrieve`), **24.2%** (`code`) and **8.98%** (`compress`) at cos ≥ 0.90. A
compose eval at <1% is an order of magnitude cleaner than the cleanest region holdout that
exists.

**One exception must be written into the gate, or it will hide a real weakness.** P0.9b's
finding stands: a GooAQ template collision (*"44 is 25 percent of what number?"* vs
*"…55 percent…"*, cos 0.995) is **not** a duplicate — it is the same template with a
different correct answer, and removing it would conceal the encoder's failure to
distinguish a slot value. So flagged pairs are **reviewed, not auto-removed**, and the
review distinguishes paraphrase (remove) from template collision (keep, and record that the
metric is measuring slot sensitivity).

### Stage 3 — re-run against what the regions actually trained on

Stages 1 and 2 must be re-run against each region's **realised train split**, not its raw
corpus, because caps and dedup change what was seen: `retrieve`'s realised split is a
400,000-row *prefix* of GooAQ plus a FiQA set collapsed from 14,131 to ~5,498. The realised
split is reproducible — `analysis.json §2` reproduced all three byte-for-byte from the
receipts' `PretrainConfig` — so this is a re-run, not new tooling.

Run it in **both directions**: compose against region-train, and region-train against
compose. `build_splits`' dedup is anchor-only and keeps the first occurrence, so a row can
survive on one side of a comparison and not the other.

Record the result in the compose receipt. A composed-model number without a stage-3 report
attached is not a measurement.

## 2.4 The allocation ledger — and the finding that matters most

**A source can be allocated exactly once.** Once a region trains on a row, that row is
permanently ineligible for the composed model's evaluation. There is no retroactive
reservation: the weights already saw it.

That makes allocation a **pre-condition** of training, not a follow-up. And it collides
directly with the current plan.

### The finding

`REMAINING.md` P2.5 lists `P2.5d — Reserve non-overlapping material for the composed model`
**after** `P2.5c — Fetch, with per-source caps`, and `P2.3 — Train the new regions` sits in
P2 ahead of both. Meanwhile P2.5's own coverage list allocates every clean staged corpus to
a region:

> `classify — banking77 + go_emotions are clean` · `reason — gsm8k + aqua_rat are clean` ·
> `code — needs multi-language and permissively licensed` · `vl — BLOCKED`

Here is the entire licence-clean, currently-unallocated staged pool:

| source | rows | region P2.5 assigns it to | licence |
|---|---|---|---|
| `deepmind/aqua_rat` | 97,467 | reason | PERMISSIVE_OK |
| `zalando-datasets/fashion_mnist` | 60,000 (train split, as staged) | vl | PERMISSIVE_OK |
| `google-research-datasets/go_emotions` | 43,410 | classify | PERMISSIVE_OK |
| `deepmind/code_contests` | 13,328 | code | ATTRIBUTION |
| `PolyAI/banking77` | 13,083 | classify | ATTRIBUTION |
| `codeparrot/apps` | 10,000 | code | PERMISSIVE_OK |
| `openai/gsm8k` | 7,473 | reason | PERMISSIVE_OK |
| **total** | **244,761** | | of which **218,350** PERMISSIVE_OK |

**As planned, P2.2/P2.3 allocate 100% of that pool to regions and leave the composed model
with nothing.** Not "not much" — nothing. Every other staged corpus is either already
trained on (all-nli, codesearchnet, gooaq/NQ/FiQA, tiny-imagenet), lineage-collided
(`snli` is 100% inside all-nli; squad and hotpotqa are Wikipedia-collided with NQ), or
share-alike (`timm/oxford-iiit-pet`).

This is the single most consequential thing in this document, and it is cheap to avoid:
the reservation in §2.5 needs about **23% of that pool**, and it costs nothing if it happens
before P2.3 and is impossible afterwards.

### The ledger rule

> Every source carries exactly one allocation: `region:<name>`, `compose`, or `eval-only`.
> The allocation is recorded in the catalogue before fetch, is checked at train time, and
> cannot be changed once a run has consumed it. A source with no allocation may not be
> trained on.

Two corollaries with measured teeth:

- **A source cannot be split across a region and compose unless the split is verified
  disjoint.** `codeparrot/apps` and `deepmind/code_contests` look like an easy split — give
  one to `code`, reserve the other for compose. They share **17.84% of problems at
  cos ≥ 0.90** (1,784 of 10,000), invisible to hashing (exact overlap: 2 rows). Splitting
  them across consumers without deduping on the stripped description would contaminate the
  compose set by 17.8% on day one.
- **Eval-only is a real but narrow allocation.** `LICENCE-FOR-OPEN-WEIGHTS.md` establishes
  the standard: it holds where it is *verified structurally* that no evaluated-on parameter
  is published (as was done for CIFAR-100 by reading `_linear_probe` and
  `_checkpoint_payload`), and it weakens wherever the metric feeds a gate — because
  selecting a checkpoint on a number *is* information flowing into the weights. Compose
  metrics gate P5. So an `eval-only` allocation for the compose set buys less than it looks
  like it does, and should not be used to rescue a restrictively licensed corpus.

## 2.5 A concrete reservation, sized rather than guessed

**Bins.** Define one capability bin per non-residual region plus a general bin:
`code, compress, retrieve, vl, classify, reason, general` — **k = 7**.

**Eval size, derived.** For a per-bin binary-ish metric at n items, the 95% binomial
half-width at p = 0.5 is `1.96·√(0.25/n)`: **±4.33 pp at n = 512**, ±3.10 pp at n = 1,000.
512 also matches the project's existing `holdout_pairs` convention, so the compose eval is
comparable in precision to every region receipt. Use **512 per bin**.

Note the composed-vs-single comparison is **paired** — both models score the same items — so
McNemar on the discordant pairs is the correct test and is substantially more sensitive than
comparing two independent ±4.33 pp intervals. The 512 figure is a floor for reporting a
per-bin number, not the resolution of the comparison.

**Cross-bin items.** Enumerate the cross-bin pairs the mind actually claims — start with
three: `retrieve×code`, `retrieve×reason`, `vl×classify` — at 512 items each.

| component | items |
|---|---|
| per-bin eval, 7 bins × 512 | 3,584 |
| cross-bin eval, 3 pairs × 512 | 1,536 |
| **compose eval total** | **5,120** |
| compose train, ≥10× eval | **51,200** |
| **reservation total** | **≈56,320 rows** |

Against the 244,761-row clean unallocated pool, that is **23.0%**. Affordable — before P2.3
runs, and unavailable after.

**Proposed whole-source reservations for the distribution-separate eval** (the tier that
*requires* a source no region touched), using differences the catalogue already asserts:

| bin | reserve for compose | leave to the region | why the split is distribution-separate |
|---|---|---|---|
| reason | `openai/gsm8k` (7,473, MIT) | `deepmind/aqua_rat` (97,467) | the catalogue's own `why` calls aqua_rat *"a different shape to GSM8K"* — grade-school arithmetic word problems vs algebraic MCQ with rationales |
| classify | `PolyAI/banking77` (13,083, CC BY 4.0) | `go_emotions` (43,410) | genuinely different label spaces and registers: banking intents vs emotion labels |
| code | **cannot be split as staged** | — | apps and code_contests overlap 17.84% at cos ≥ 0.90; either dedupe on stripped description first, or reserve neither |
| vl | EuroSAT test split / Quick Draw / Caltech-256 | the six-domain composite | already held out by the licence doc's probe design |
| retrieve | **no candidate** | — | every permissive retrieval source found is wanted by the region; nothing is spare |
| compress | **no candidate** | — | the only permissive candidate (SynCSE-scratch-NLI) is unfetched and is the region's own replacement |
| general | **no candidate at all** | — | see §2.7 |

Reserving gsm8k + banking77 gives 20,556 rows across two bins — max share 63.6%, N_eff 1.86
over those two. Enough for two bins' eval and part of their train; **five of seven bins have
no distribution-separate source identified.**

## 2.6 "Generalist with a few specialisations", as a measurable corpus property

Stated as five checks over the bins defined above, so it can be verified rather than
aspired to.

**1. Coverage.** Every bin has ≥ 512 eval items in the compose set. A bin with no items is
a capability the mind is not being measured on, whatever its catalogue entry says.

**2. Generalism = near-uniformity over bins.** In the compose corpus `C`:
`max_bin_share(C) ≤ 0.25` and `N_eff_bins(C) ≥ 0.8k` (≥ 5.6 for k = 7), where `N_eff` is
inverse Simpson. "Generalist" is not "covers several bins"; it is "no bin is where most of
the corpus lives".

**3. Specialisation = concentration, and it is legitimate.** For each region `i`, its own
corpus's share in bin `i` is ≥ 0.90. A specialist's corpus *should* be concentrated in its
bin — that is what specialisation means. The Part 3 balance rule therefore applies to
**sources within a bin**, never across bins. This is the distinction that makes the two
halves of this document consistent: `code` being 100% code is fine; `code` being 100%
*Python* is not.

**4. "A few" is counted, not asserted.** The number of specialisations equals the number of
bins in which the composed model, measured on `C`, exceeds the **residual-only baseline** by
more than the paired-test threshold. If that count is smaller than the number of declared
regions, the difference is the number of regions that are **names rather than capabilities**,
and it is reportable as a single integer.

**5. Generalisation across specialisations.** On `C`'s cross-bin items, the composed model
must beat the **oracle-routed best single region** — i.e. the score you would get by picking,
per item, the single best region for that item. Beating a fixed single region shows
dispatch. Beating oracle routing is the only result that shows composition.

**These five collapse into three gates** on the composed model, in increasing strength:

| gate | statement | what it measures | strength |
|---|---|---|---|
| G1 | composed ≥ best single region on the mixed set | routing works | weak — nearly automatic on a bin-balanced corpus |
| G2 | per bin, composed within ε of the region trained for that bin | **no composition tax** | the one that actually bites; sharing a stream can degrade every region |
| G3 | on cross-bin items, composed > oracle-routed single region | composition | the claim the architecture rests on |

P5's current gate is G1. G2 and G3 do not exist yet, and neither is measurable without the
corpus this part specifies.

## 2.7 What has no source, stated plainly

Following the model of the licence document's vision section: naming the loss rather than
smoothing it.

1. **The general bin is empty and has no candidate.** `residual_mlp` — the fallback region
   whose trigger is *"tokens not tagged code/retrieve/compress"* — trains on synthetic
   tokens only. No general-text corpus is in the catalogue. FineWeb, C4 and Pile are named
   in `csd-regions.json`'s notes as step-4 foundation corpora and **none of them has a
   verdict in `LICENCE-FOR-OPEN-WEIGHTS.md`** — they have not been audited. So the bin the
   composed model most needs, and the region that serves it, both have nothing. This is the
   largest gap in Part 2.
2. **Five of seven bins have no distribution-separate compose source.** Only `reason` and
   `classify` can be split cleanly today, and only because each happens to have two staged
   sources with genuinely different shapes.
3. **No cross-region corpus exists, anywhere.** Not in the catalogue, not staged, not
   surveyed. Every cross-bin item the mind needs would have to be **constructed**, and the
   obvious donor for the `retrieve×reason` pair (HotpotQA) is burned twice — Wikipedia
   lineage and share-alike.
4. **The `none of these` population that `classify` needs is the same missing general
   material.** One gap, two symptoms.

**Consequence.** The composed model's corpus is currently a *construction project*, not a
fetch. The realistic sequence is: reserve gsm8k and banking77 before P2.3 (free, and
impossible later); construct cross-bin items from already-allocated material with stage-1
and stage-2 disjointness enforced against the realised region splits; and treat the general
bin as an open licence question that must be answered before P5 can mean anything.

---

# Part 3 — The balance rule

A corpus is too concentrated when **the headline metric could be produced by a model that
learned only part of it**. That is the principle. Below it is five checks with thresholds,
each stating the imbalance it prevents and each computable from data the pipeline already
records.

All checks operate on **provenance groups**, not shard files and not rows. Splitting one
corpus across ten shards must not read as ten sources.

All checks operate on the **post-dedup, post-cap** distribution — the realised training set.
`retrieve` is the demonstration: its largest source rises from 77.77% to 79.09% and its
smallest falls from 2.75% to 1.09% across `build_splits`' dedup alone. Checking the
pre-dedup mix would check a corpus nobody trained on.

## B1 — Maximum single-source share ≤ 0.40

**Prevents:** the headline number being a statement about the largest source rather than
about the region.

**Reasoning.** At source share `p`, a degenerate model that learned only the dominant source
and performs at floor elsewhere scores roughly `p × (its in-source metric)` on an in-mixture
holdout. For the metric to be about the region, no single source may be able to supply a
majority of it, so `p < 0.50` is the hard line. **0.40** is the operating cap; the 0.10
margin covers the pre→post-dedup drift measured above, which moves in the wrong direction.

**Measured today:** `retrieve` 0.791 ✗ · `compress` 0.582 ✗ · `code` 1.000 ✗ · `vl_latent`
1.000 ✗ · staged `reason` 0.929 ✗ · staged `classify` 0.813 ✗. Every region fails B1.

## B2 — Effective number of sources `N_eff = 1/Σp²` ≥ 3

**Prevents:** a corpus with a long tail of token sources passing B1 on a technicality.

**Why inverse Simpson (Hill order 2) rather than `exp(H)`.** The failure being guarded is
*dominance*, and order 2 weights the dominant source more heavily. Measured contrast on
`retrieve` post-dedup: `exp(H) = 1.74` versus `1/Σp² = 1.50`. The order-2 number is the one
that says "this is effectively a 1.5-source corpus", which is the true description. **Report
both; gate on inverse Simpson.**

**Threshold:** ≥ 3 for a region whose name states a general capability (`code`, `retrieve`,
`compress`, `classify`, `reason`); ≥ 2 for a region whose name and trigger are both
explicitly narrow.

**The waiver clause, without which this rule would simply be ignored.** `code` and
`vl_latent` are both at N_eff = 1.00 and have **no licence-clean second source** — Part 1
establishes that for both. A rule with no escape becomes a rule nobody runs. So:

> A region below its N_eff threshold must either (a) rename to what its corpus actually is,
> or (b) carry a dated waiver in `csd-regions.json` naming the missing sources, the reason
> (licence, non-existence, cost), and the capability the region therefore does not have.
> A receipt from a waived region reports the waiver alongside its metric.

That is what turns "the corpus is too narrow" from an argument into a record.

## B3 — Train/eval distribution correspondence must be declared and implemented

**Prevents:** the retrieve case — an eval design that exists in a comment and not in the
code.

Every region declares its eval as exactly one of:

- **`in-mixture`** — the holdout is drawn from the same pool by the same code path.
  Measures in-distribution competence. **The receipt must report the metric per source, not
  only in aggregate**, because an aggregate over an imbalanced mixture is the dominant
  source's number wearing the region's name.
- **`held-out-domain`** — the eval's provenance group contributes **exactly zero rows** to
  training, and the receipt records that zero. Measures transfer. **Requires an in-mixture
  number alongside it**, or a drop cannot be attributed to domain shift rather than to the
  model.

Anything else is not a valid declaration. The check is mechanical: compare
`train_source_shares` against `eval_source_shares`; a source with nonzero share on both
sides is `in-mixture` for that source, whatever the comment says.

## B4 — A cap must be a sample, not a prefix

**Prevents:** a balance decision silently becoming an arbitrary one.

**Measured:** `load_pairs` returns as soon as `limit` is reached, iterating shards in sorted
order, and `build_splits` shuffles *after* the cap. GooAQ's 400,000 is therefore the first
13.3% of a 3,012,496-row corpus in file order, and no downstream shuffle can repair a
non-representative prefix.

**Check:** caps are applied by strided or reservoir sampling over the whole source, and the
receipt records the sampling method and the seed. A cap whose method is unrecorded is not a
cap.

## B5 — Within-source concentration

**Prevents:** balance across sources concealing a single source dominated by one
sub-population.

Apply B1/B2's logic to each source's natural stratum key where one exists — `repo` for
CodeSearchNet, `genre` for MultiNLI, the label for a classification set, the language for a
multilingual set:

- max stratum share ≤ `max(0.25, 2/k)` where `k` is the number of strata;
- max:min stratum ratio ≤ 20:1.

**Measured, and it separates the good from the bad cleanly:**

| source | strata | max share | max:min | B5 |
|---|---|---|---|---|
| CodeSearchNet | 13,581 repos | 2.441% | — | **pass** |
| `banking77` | 77 classes | 1.87% (uniform 1.30%) | 5.34:1 | **pass** |
| `aqua_rat` | 5 answer letters | 22.87% | 1.62:1 | **pass** |
| `fashion_mnist` | 10 classes | 10.00% | 1.00:1 | **pass** |
| `oxford-iiit-pet` | 37 breeds | 2.72% | 1.08:1 | pass on breed; cat/dog is 1,188/2,492 |
| `go_emotions` | 28 labels | **32.76%** | **184.7:1** | **fail on both** |

CodeSearchNet passing B5 while failing B1/B2 is the point of having both: it is a diverse
source and a monolithic corpus at the same time, and only two different checks can say so.

---

## The worked example: is `retrieve` a defect or a deliberate transfer test?

The task is to argue it both ways and then commit. Here is both, then the commitment.

### The case that it is a deliberate transfer test

- **The intent is documented and coherent.** `csd-train-all.py` states it: *"Held-out eval
  stays on fiqa dev/test, which is a different domain (financial QA) and therefore measures
  transfer rather than memorisation. Raise the cap if transfer is the bottleneck; that is a
  measurement, not a guess."* That is a deliberate design with a stated falsification
  condition.
- **The same design is used elsewhere in the project, consistently.** `VL_REGIONS` scores
  `vl_latent` on CIFAR-100 with the identical justification: *"cifar100 is a DIFFERENT
  dataset with different classes, so the probe on it measures whether the representation
  transfers rather than memorises — the same reason `retrieve` is scored on out-of-domain
  fiqa."* A pattern applied twice is a design, not an oversight.
- **It is the right thing to measure.** A retriever that works only on the distribution it
  was trained on is not a retriever. Testing on an unseen domain is the standard way to show
  a representation generalises, and a corpus dominated by one source is the *normal*
  condition in retrieval pretraining — MS MARCO-scale single-source pretraining followed by
  out-of-domain BEIR evaluation is the field's default methodology.

### The case that it is a defect

Three measured facts, in increasing severity.

1. **The code does not do it.** `build_splits` shuffles the concatenated pool and takes
   `all_pairs[: cfg.holdout_pairs]` — a uniform sample of the mixture. Expected FiQA content
   of the 512-pair holdout: **≈5.6 items** post-dedup, ≈14 pre-dedup. recall@1 0.7480 is an
   **in-mixture** number measured on a ~79%-GooAQ holdout. The transfer test is described in
   a comment and implemented nowhere on this path.
2. **Even if implemented, it would not be a transfer test.** FiQA contributes ~1.09% of
   training post-dedup. An eval whose provenance group is present in training measures
   memorisation-plus-transfer with no way to separate them. A transfer claim needs a source
   with **zero** training rows.
3. **The holdout that does exist is not a clean in-mixture number either.** 53.7% of it has
   a near-duplicate in training at cos ≥ 0.90, 32.4% at ≥ 0.95, 14.8% at ≥ 0.98. So the
   number is neither an honest in-distribution measurement nor a transfer measurement. It is
   the third thing, which has no name and no interpretation.

### The commitment

> **It is a defect — and specifically a *declaration* defect, not a design defect.**
>
> Deliberate transfer testing is legitimate and this project should keep doing it. But an
> eval is a transfer test **only if its provenance group contributes exactly zero rows to
> training and the receipt records that zero**. `retrieve` declares transfer, trains on the
> eval domain, and measures a mixture holdout. All three of those must agree before the word
> "transfer" may appear in a receipt.
>
> **The 77.8%/79.1% concentration is a separate defect and fails B1 and B2 regardless of how
> the eval is designed.** No eval design rescues a corpus in which one source can supply
> four-fifths of the metric. The two problems are independent and both must be fixed.

**What that implies concretely, and it runs against the obvious move.** If the GooAQ licence
question resolves favourably, the cheap repair is to drop the share-alike sources and
*uncap* GooAQ. B1/B2 forbid it: that would take the corpus to 100% of a single source,
N_eff 1.00, and would simultaneously worsen the template-collision population that already
inflates the holdout. The metric would improve while the region got narrower — which is,
per `LICENCE-FOR-OPEN-WEIGHTS.md`'s own phrasing, *"the most dangerous shape a change can
have."*

---

## What this document does not decide

- It does not change any catalogue entry, verdict, training script, region config or cap.
- It does not resolve the GooAQ README-vs-LICENSE contradiction. That is a one-email
  question to AI2, owned elsewhere, and it determines whether `retrieve`'s largest source
  exists at all.
- It does not decide whether `code` should be renamed or broadened. It states that the name
  and the `router_trigger` disagree, that the trigger is the honest one, and that the
  decision rule is B2's waiver clause: rename, or record why the sources are missing.
- It does not decide between the 497,000-image vision composite and the 162,000 strict
  variant. Both pass; the choice is a measurement.
- It does not decide whether trained weights are adapted material under CC BY-SA. That
  question is open in the licence audit and this document inherits its openness.

## What could not be determined

1. **Whether a licence-clean general-text corpus exists for the general bin.** FineWeb, C4
   and Pile have not been audited by this project. Until they are, `residual_mlp` and the
   composed model's general bin have no supplier and no verdict.
2. **The true licence of the CodeSearchNet lineage.** Three mirrors assert `c-uda`, `MIT`
   and `apache-2.0` over an upstream tagged `other`. `the-stack`'s `licenses.json` returned
   HTTP 401 on both blob and resolve URLs during the code survey.
3. **Whether `codeparrot/github-code-clean` retains a per-row `license` field.** Its
   dataset-level `apache-2.0` tag sits over a parent whose 15-licence set includes GPL/AGPL.
4. **`mteb/*` is under-surveyed for retrieval candidates**, not cleared — the retrieval
   survey's search budget capped mid-work.
5. **`allenai/peS2o`'s upstream** could not be verified: `allenai/s2orc` returns 401 on
   every route.
6. **Whether mixing six visual domains in one I-JEPA run helps or fragments a ~23M-parameter
   encoder.** No literature was found that settles it at this scale; the licence document
   proposes the experiment.
7. **Whether G2 (no composition tax) is achievable at 16M parameters per region.** Not
   knowable before P5, and it is the gate most likely to fail.
