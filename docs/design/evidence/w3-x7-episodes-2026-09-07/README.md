# X7 — recall-dependent two-turn episodes

**What is here.** `MANIFEST.json` is the build record for shape **X7** (design §5.4, constructed
per §5.5(e)) on the pre-committed **text-only slip branch**. The items themselves and the DEC-38
source-row ledger live outside the repo at the paths the manifest records, with their SHA-256s;
they are re-derivable from the pinned corpora and the pinned generator.

**Why X7 matters.** Phase A's episodic read is a constant, so its mutual information with the
target is exactly zero: the synthetic stream has no recall dependency, and for a stream whose
inputs determine the target `MI(read; target | inputs) = 0` for *any* read. X7 items are what make
the store load-bearing.

## What was built, and what was not

| | status |
|---|---|
| item construction, schema, per-item negative control, DEC-38 assertion, provenance, keyed split | **built** |
| the episode harness (turn 1's write committing before turn 2 is scored; partition reset) | **not built** — another lane; contract stated below |
| NSRS admission (`s_r`) | **not run** — GPU-blocked and blocked on W2b; every item carries `admission.status = "pending"` |
| X8′ (the other pre-committed slip-branch shape) | **not built** |

## The item

Two turns, drawn from source rows that are disjoint under DEC-38.

- **Turn 1 (commit).** A retrieval question over a four-record pool. Its answer commits one
  numeric fact `X` to the store.
- **Turn 2 (probe).** A different document, with the fact deferred: *"Let X be the quantity you
  committed in the previous turn. X is not restated here."* The probe's own answer is `Y`. The
  item asks for `X + Y`.

**The five options are `donor_fact + Y` — one per candidate turn 1.** One donor is the episode's
own; four are unrelated episodes. A reader that solves the probe knows `Y` and still cannot rank
the options, because every option is consistent with a different committed fact. Chance is exactly
`1/5`, and each item prints it.

**Two variants**, matching X7's two named fact types and its two ablation pairs.

| variant | turn 1 source | fact | ablation pair |
|---|---|---|---|
| `passage_claim` | `deepmind/aqua_rat` rationale passages | the value the retrieved passage's reasoning arrives at | `{episodic_store, memory}` |
| `api_record` | `codeparrot/apps` problem records | the retrieved record's declared number | `{episodic_store, language}` |

## The negative control is a construction gate, not an eval step

§5.5(e) property 1: an episode whose turn 2 is answerable without turn 1 is **rejected at
construction**. Six gates enforce that; each has a test that makes it fire.

| gate | what it catches |
|---|---|
| `fact_value_collision` | the substituted turn 1 commits the same fact, so the control cannot fail the item |
| `fact_in_probe_window` | turn 1's fact is readable off turn 2 |
| `answer_in_probe_window` | the answer itself is readable off turn 2 |
| `dec38_shared_source_row` | the two turns share a source row, so the episode leaks against itself |
| `negative_control_solved_with_wrong_turn1` | the pinned BM25 reference solver reaches the gold option from a *substituted* turn 1 |
| `negative_control_solved_without_turn1` | the same solver reaches it with no turn 1 at all |

A seventh gate, `option_value_collision`, catches two options that **render** identically. Its
tally is `0` in this build, and the first version of this document called it unreachable. **That
was wrong, and the correction is worth stating rather than quietly editing.**

`fact_value_collision` inspects the facts *before* the shift; `option_value_collision` inspects
the options *after* it. Those are different questions, because `format_value` rounds to four
decimals and **rounding does not commute with addition**. Facts `0.00004` and `0.00006` render
`"0"` and `"0.0001"` — distinct, so the earlier gate passes them — yet with a probe answer of
`0.00003` both options render `"0.0001"`. A sweep of a 39×39×39 grid of small decimals finds
2,414 such triples. It fires through the public `build_episode` API, which accepts any float
fact, and the tests construct exactly that case.

The rendered string **is** the item: options are stored rendered and the harness scores a string
choice, so two options that render the same make the item unanswerable and the negative control
non-discriminating.

**Why the tally is zero anyway** is a fact about the *builder*, not the gate: it admits only
positive integral facts and probe answers, and on integers `format_value` is exact, so the shift
is injective (0 collisions over a 400×400×200 sweep). That filter is an invariant the gate's quiet
depends on, so it is asserted directly in `tests/test_x7_episodes.py`. Relax the filter and the
test says, in one place, that this gate stops being a formality.

**What this is not.** The sufficient control is *"run turn 2 through the mind after committing a
substituted turn 1 and require a wrong answer"*, and that needs the harness and trained weights.
What runs at construction is the necessary form. An item admitted here can still be rejected by
the harness; an item rejected here can never be recall-dependent.

## The harness contract these items assume

Stated rather than assumed, because the store's commit semantics are under active change.

| concern | assumption |
|---|---|
| partition per episode | `derive_scope(principal, session=f"x7-{item_id}")`. A fresh session is the only reset the current store API offers: there is no clear call, and `stop()` is terminal. |
| turn boundary | one `WhiteMatter.forward(inputs)` per turn. `forward` reads at entry and writes at exit, so an episode is two successive calls sharing one `Scope` and one `domain`, with distinct per-item `logical_key`s. |
| commit | turn 1's write returns `WriteReceipt(committed=True)` before turn 2 is scored. The design prose says `LearnReceipt`; **no such symbol exists in this repo**. |
| read | turn 2's read is expected to be content-addressed by a query vector. On `main` today `retrieve()` takes no query, which is why phase A's read is a constant. Content belongs in the query, never in the partition key. |
| scoring | a closed five-way choice over `episode.turns[1].options`. |

## Where the spec is ambiguous, said rather than chosen silently

§5.5(e) asks for four things that cannot all hold at once:

1. turn 2's question is *"rewritten so its answer requires the fact turn 1 established"*;
2. *"the label is the same label the single-turn item already carried"*;
3. the two turns come from **disjoint** source rows (DEC-38);
4. the fact is **not in turn 2's context window**.

If the fact comes from a disjoint row and the answer genuinely depends on it, the answer cannot be
the original single-turn label — the label has to be recomputed. The only reading that keeps (2)
is one where the fact is a *selector* rather than an input to the answer, and a selector has to
appear among turn 2's candidates, which breaks (4).

**What was built, and why.** The label is computed by the join generator — `X + Y` — which is what
§5.5(a′) already does for X3, whose labels are likewise constructed rather than inherited. So (2)
holds against the *multi-hop item the episode is a re-staging of*, not against a single-turn
aqua_rat row. **This is a reading, not a fact.** If the intended reading is different, the binding
function is one place in `src/cogsyndelta/reserve/episodes.py` and the items rebuild from the same
rows.

## Stated limitations

- **`deepmind/code_contests` is not consumed.** Its MANIFEST-declared fingerprint columns are
  `("description", "solutions")`, and `solutions` is most of a 19 GB shard. Fingerprinting it at
  DEC-38 granularity needs a streaming reader this row did not build, so the code half comes from
  `apps` only.
- **aqua_rat's share of X7's own source rows is 55.6%.** This is not a B1 verdict in either
  direction. B1 is computed over the whole reserve, and that figure is itself under re-accounting:
  W3's scope pass measured 87,599 unaccounted `TRAIN_OK` rows in `squad`, which would move the
  reserve-wide aqua_rat share from the design's printed 79.5% to roughly 43% — below both the 0.50
  hard line and the 0.40 operating cap. The number here is printed to be summed into that
  accounting, not read as a violation.
- **aqua_rat declares no revision.** `MANIFEST.json` pins the shard's SHA-256 instead, so drift is
  still detectable (§5.6 asks for a revision; the landed manifest does not carry one).
- **The apps↔code_contests near-duplicate dedupe is not applied**, because code_contests is not
  consumed.

## Rebuilding

```bash
export OMP_NUM_THREADS=1
secret exec CSD_K_SPLIT=akula/csd-k-split -- \
    python scripts/csd-build-x7-episodes.py --out /akula-data/csd/reserve/x7
```

Without `CSD_K_SPLIT` the build refuses rather than falling back to a seeded split (DEC-39).
