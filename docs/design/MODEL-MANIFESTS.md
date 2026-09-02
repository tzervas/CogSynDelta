# Model Manifests

**Status:** design. Nothing here is implemented, and no existing config has been changed.
**Scope:** every model on this fleet that is *served* or *trained*.

---

## Why this exists

On 2026-09-02 a `llama-server` had been running on `gpu-1080ti` for twenty-one hours with
its full invocation — weights path, alias, bind address, context size, GPU layer count —
existing **only in `/proc/<pid>/cmdline`**. Stopping it would have destroyed the only
record of how to start it again. The process was reachable, in the routing table, and
serving traffic; nothing about it was wrong except that it had never been *declared*.

That was recoverable because someone noticed. The general problem is that nothing on this
fleet *required* it to be declared in the first place, so the next one will also be found
by accident, and the one after that will be found by its absence.

Two more instances of the same shape:

- **CSD's training regions are a Python dict.** `scripts/csd-train-all.py` holds `REGIONS`
  and `VL_REGIONS`: corpus globs, column pairs, per-source caps. A training run is
  therefore a command someone typed, not a file you can diff, review, or hand to a
  reviewer as "this is what produced that checkpoint". The dict's comments are excellent —
  the `compress` region's poisoned-objective note is a genuinely important finding — but a
  comment is not a record that a receipt can cite.
- **Every format here is read by exactly one consumer.** `edge-backends.json` is read by
  routing. The LocalAI YAMLs are read by LocalAI. `csd-regions.json` is read by the mind
  assembly. `cohort.json` is read by placement. Nothing reads a declaration and
  *instantiates the thing it declares*, so declaring something has no operational payoff
  and gets skipped under time pressure — which is exactly how the orphan happened.

The fix is not more documentation. It is a declaration that something reads.

## The spine of the design

> **Manifests declare. Receipts report. Each one names the other.**

`src/cogsyndelta/pipeline/receipt.py` already established half of this: one outer shape,
written by every stage of every pipeline, so a reader can compare runs without knowing the
architecture. Its module comment states the principle exactly — *declare a shape once, let
many consumers read it* — and its `load_all(roots)` reads receipts from several roots
rather than one blessed directory.

This design is the other half of that loop, and deliberately mirrors it:

| | Manifest | Receipt |
|---|---|---|
| Tense | What a thing **is** and what it **should** cost | What a run **did** and what it **actually** cost |
| Written by | A human or an agent, reviewed in a PR | A pipeline stage, unattended |
| Names the other via | `measured[].from_receipt` | `provenance.manifest`, `provenance.manifest_sha` |
| Gates | Declares the thresholds | Records the verdict |
| Read by | The instantiator and the reconciler | The console |

A number in a manifest that cites the receipt that produced it is auditable. A receipt
that names the manifest it was launched from is reproducible. Neither is true today.

---

## A. One schema or two?

**Two body schemas under one shared envelope.** A tagged union, discriminated on
`spec.kind`, and I am calling it that rather than dressing it up as "one schema with
optional fields".

### Why not one flat schema

The honest test is: how many fields would be null on a typical manifest? For a served
model, every field describing a corpus, an objective, an encoder shape, a batch size, a
gate threshold and a held-out split is null. For a trained model, every field describing a
port, a bind address, a KV cache dtype, a context size, a TTL and a residency policy is
null. That is not one schema with variants; that is two schemas sharing a file extension,
and a validator over it can only check the intersection — which is precisely the part that
was never the problem.

The two things also have different *lifecycles*, and the lifecycle is what the
instantiator branches on:

- A **served** model is a long-lived process bound to a port. It is started, it stays up
  (or is deliberately on-demand), it holds VRAM, it conflicts with other residents on the
  same card, and its failure mode is "not answering".
- A **trained** model is a job. It runs once, terminates, emits a receipt and a checkpoint,
  and its failure mode is "a gate did not hold". It has no port and cannot conflict with
  anything except for the duration of the run.

### Why not two unrelated schemas

Because the fields that make a manifest *trustworthy* are identical for both, and because
a trained model becomes a served model.

`csd/retrieve` trained today is a checkpoint tomorrow, quantized the day after, and served
the week after that. If served and trained manifests live in unrelated namespaces, that
chain is expressible only as prose. Under one envelope with one id scheme, a served
manifest's `source` can name a *manifest id* instead of a Hub repo, and the provenance
chain from corpus → checkpoint → quantized artifact → served endpoint is a graph you can
walk. That is the single strongest argument for a shared envelope, and it is worth more
than schema tidiness.

The shared part is trust: upstream and pinned revision, licence with a verdict, measured
resource usage with a date and a host, which hosts the manifest is valid for, and what
state the thing is in. Those questions have the same answers-shape whether the artifact is
a GGUF or a corpus, so they are asked once.

```
manifest/v1
├── envelope   (shared, always required)
│     id, kind, status, source, licence, hosts, measured, estimated,
│     history, conflicts_with, owner, updated
└── spec       (discriminated on kind)
      ├── served   → invocation: weights, runtime, quant, context, bind, budget
      └── trained  → recipe: sources, objective, encoder, schedule, gates
```

`trained` carries a second, narrower discriminator, `spec.objective_family`, for the same
reason `csd-train-all.py` gave `VL_REGIONS` its own dict:

> *The visual region does not fit the text (left, right) pair shape: its objective is
> latent prediction over image patches, its metric is a linear probe rather than recall,
> and its data is an image struct rather than two text columns. So it gets its own entry
> and its own runner rather than being bent into REGIONS.*

That was the right call in the code and it stays the right call in the schema. A
`contrastive-pair` body has `pair_columns`; a `jepa-predictive` body has `image_column`,
`label_column` and a transfer set. Bending one into the other buys nothing and loses the
validator's ability to say "this region declares a probe eval but no probe columns".

---

## B. Required fields — what makes a manifest trustworthy

The LocalAI YAMLs are the model for this section. They already do the hard part; what
follows is that habit generalised and made mandatory rather than customary.

Take `local-code.yaml`. Its comment block carries the upstream repo, a 40-character
revision, the licence, the quant with its on-disk size, the native context — and this:

> *2026-08-31 measure: 16k f16 KV used ~11.7 GiB of 23028 (not the old 22 GiB no-GQA
> estimate).*

That sentence is the most valuable line in the whole config tree, and it is valuable for
three separate reasons: it is a measurement, it is dated, and **it names the estimate it
replaced**. A manifest scheme that captures only the first of those is a downgrade.

### Required in every manifest (validation errors, not lint warnings)

| Field | Why it is required |
|---|---|
| `id` | Stable, namespaced, fleet-unique. The join key to receipts, routing and reconciliation. |
| `kind` | `served` \| `trained`. Selects the body schema. |
| `status` | `active` \| `on-demand` \| `benchmark-only` \| `planned` \| `convert-pending` \| `retired`. |
| `source.upstream` | The repo of record. |
| `source.revision` | A **commit**, not a tag. Tags move; a manifest pinned to a tag is a manifest that quietly changes meaning. |
| `licence.observed` | The licence string as it actually appeared, verbatim. |
| `licence.verdict` | `SERVE_OK` \| `TRAIN_OK` \| `EVAL_ONLY` \| `REJECTED`. |
| `licence.checked_utc` | A verdict without a date is an opinion of unknown age. |
| `hosts` | Host ids from `cohort.json` this manifest is valid for. |
| `measured` | A **list** of measurement records. May be empty; may not contain an unmeasured number. |
| `owner` | Which repo owns this file. Prevents two repos from both editing one id. |
| `updated` | Date of last edit to the manifest itself, distinct from measurement dates. |

### `status` — the highest-value promotion from comment to field

Five LocalAI YAMLs already carry status in prose: *"gpu16: benchmark, do not preload"*,
*"GGUF convert pending"*, *"Missing 20B GGUF is an allowed gap"*, *"ON DEMAND as of
2026-09-02"*. This is the field an instantiator must branch on, so it cannot stay prose.

Crucially, **a manifest for a thing that does not exist yet is legitimate**.
`local/moonlight` names a GGUF nobody has converted. That manifest is useful — it records
the intent, the upstream, and the shortname — and `status: convert-pending` is what stops
it reading as a capability. `csd-regions.json` already made this exact call and said so:

> *`live` reflects what is IMPLEMENTED, not what is intended; `pretrain.available`
> reflects what is ON DISK, not what is named. Both default to false so that intent can
> be recorded without it reading as capability.*

Same rule, same reason, now with one field instead of two booleans per consumer.

### `measured` — the honesty mechanism

A measurement record, not a scalar:

```yaml
measured:
  - what: vram_mib
    value: 11980
    on: 2026-08-31          # date of the measurement, not of the file
    host: akula-prime       # a measurement on one card is not a claim about another
    conditions: "ctx 16384, f16 KV, parallel 1"
    method: "nvidia-smi peak during a 2k-token generate"
    supersedes:
      value: 22528
      basis: "no-GQA analytic estimate"
      why: "GQA not accounted for; the estimate was 1.9x the truth"
```

Three rules, and the schema enforces all three:

1. **No resource number may appear outside `measured` or `estimated`.** There is no
   top-level `vram_mib` field to put a guess in. A brief made this point sharply — *a
   manifest asserting unmeasured numbers is worse than none, because it will be believed* —
   and the way to honour that is to make the unmeasured assertion unrepresentable, not to
   ask reviewers to catch it.
2. **`estimated` entries require a `basis` and are refused by admission control.** The
   scheduler may read `measured` to decide whether a model fits on a card. It may not read
   `estimated`. An estimate is for planning; it is not permission to start a process.
3. **A measurement is never overwritten, only superseded.** The old value and the reason it
   was wrong are the record. `local-code.yaml` demonstrates why: knowing that the 22 GiB
   figure was a no-GQA estimate is what stops someone re-deriving it next month.

Staleness is reported, never enforced. If a measurement predates the artifact's current
`source.revision`, or predates a recorded driver or hardware change on its host, the loader
flags it as suspect and keeps it. Silently discarding the only number you have is worse
than showing a number with a warning on it.

### `licence` — reuse the corpus gate's vocabulary, do not reinvent it

`scripts/csd-corpus-expand.py` already solved this, structurally:

> *Every entry must carry a license string that was actually observed on the dataset card
> or upstream repository, and a verdict. `fetch` refuses any entry whose verdict is not
> TRAIN_OK, and there is deliberately no flag to override that.*

And it carries the finding that justifies the `upstream` field:

> *A mirror's tag is not evidence about its upstream. BeIR/scifact is tagged cc-by-sa-4.0
> while allenai/scifact, the dataset it mirrors, is tagged cc-by-nc-2.0.*

The manifest `licence` block is that `Dataset` record's licence fields, verbatim in
meaning, with `TRAIN_OK`/`REJECTED` extended by `SERVE_OK` and `EVAL_ONLY`. `EVAL_ONLY`
exists because `csd-regions.json` already records the case: *"SciFact and NFCorpus are
eval-only per license (NC / ToS) and must not be trained on."* That distinction is
currently a sentence in a `notes` string; it should be a verdict a tool can refuse on.

```yaml
licence:
  observed: "apache-2.0 (HF model card, Qwen/Qwen2.5-Coder-14B-Instruct-GGUF)"
  verdict: SERVE_OK
  checked_utc: "2026-08-31"
  upstream_says: ""      # required and non-empty when the artifact re-hosts someone else's
  evidence: "https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF @ d0a692e"
```

### `hosts` — reference host facts, never restate them

`cohort.json` owns `vram_mib: 23028`, the GPU model, the sm level and the duty list. A
manifest names hosts by id and never copies those numbers. The one exception is inside a
`measured` record, where the total is context for the measurement (*"~11.7 GiB of 23028"*)
and belongs to that record's narrative rather than to the manifest's assertions.

The `never` lists in `edge-backends.json` become preconditions the validator can check:
`gpu-1080ti` declares `never: [BF16, FP8, CUDA 13.x images, SageAttention]`, so a manifest
listing `gpu-1080ti` in `hosts` with `kv_dtype: bf16` is rejected at validation, on the
host's own declared constraint, without anyone remembering the rule.

### `conflicts_with` — exclusivity is a constraint, not a comment

*"Exclusive 3090"*, *"do not dual-load with Ministral 14B"*, *"do not dual-load both"*
appear across five LocalAI YAMLs as prose. The scheduler cannot read prose.
`conflicts_with: [local/uncensored, local/fast]` plus a `measured` VRAM figure is the same
statement in a form that can refuse a second load.

---

## C. Instantiation — how a manifest becomes a running thing

### The reconciler comes first, and it is the actual answer to the orphan

The framing question was "generated units or a runtime launcher". Both answers miss what
would actually have caught the twenty-one-hour orphan. Neither a generator nor a launcher
notices a process that was never declared, because both start from the manifest and work
forwards. The check that catches it starts from the *host* and works backwards:

> **Every GPU-resident process on every fleet host must be attributable to a manifest.**

`csd-manifest verify --host gpu-1080ti` is read-only and does a three-way comparison:

| Compared | Source of truth |
|---|---|
| what is **declared** | the manifest set |
| what is **installed** | the rendered unit files on the host |
| what is **running** | `/proc/<pid>/cmdline` plus `nvidia-smi --query-compute-apps` |

and reports four states, of which only the first is healthy:

1. `OK` — declared, installed, running, and all three agree.
2. `UNDECLARED` — running and holding VRAM, with no manifest. **This is the orphan.**
3. `DRIFTED` — the installed unit differs from what the manifest renders, or the running
   argv differs from the installed unit. Someone edited a unit, or started something by
   hand over the top.
4. `ABSENT` — declared `active`, not running. Legitimate for `status: on-demand`; a fault
   for `status: active`.

This is worth building *before* the generator. It is read-only, it cannot break anything,
it works against the fleet as it exists today with zero migration, and its output is the
evidence for whether the rest of this design is urgent. If `verify` finds one orphan and
nothing else, that is worth knowing before rewriting config. It also directly serves the
standing rule in `AGENTS.md` — *probe first, then conclude; when a doc and a live probe
disagree, the probe wins* — by making the probe a command rather than an act of diligence.

### Generated units versus a runtime launcher

For the **served** side, the choice is real. Weighing it on this fleet specifically:

| | Generated systemd units | Runtime launcher daemon |
|---|---|---|
| Survives reboot | Yes, systemd's own job | Only if the launcher is itself a unit — so you need a unit anyway |
| Inspectable | `systemctl cat` shows the exact argv | Argv is constructed in memory; you are back to reading `/proc` |
| New daemon | None | One per host, resident, on cards the fleet keeps deliberately free |
| Failure blast radius | One unit fails | The launcher fails and every model on the host is unmanaged |
| Drift | **Real** — an edited unit silently diverges from its manifest | None; there is no second copy to diverge |
| Dynamic reconfiguration | Regenerate and `daemon-reload` | Immediate |

**Decision: generate units.** The drift column is the only real cost, and it is
addressable; every other row favours generation, and two rows are close to disqualifying
for the launcher. `gpu-1080ti` has 11 GiB total and 7.8 GiB of RAM, and the fleet has just
deliberately moved its llama-server to on-demand *to free VRAM* — adding a resident daemon
to that host to manage a service that is usually stopped inverts the decision that was just
made. Services here are already `systemctl --user` units with `Linger=yes`, so generation
targets a mechanism that is in place and understood.

Drift is closed by three constructions, not by discipline:

1. **A provenance header on every generated unit.**
   `# GENERATED by csd-manifest from manifests/served/pascal-fast.yaml @ <sha256>`, plus
   `# DO NOT EDIT — edit the manifest and regenerate.`
2. **`verify` treats a drifted unit as a finding, not a nuisance.** It prints the diff. It
   does not auto-correct, because auto-correcting would destroy a hand-edit that might have
   been the operator fixing production at 2am — which is information.
3. **Narrative lives in the manifest and is rendered *into* the unit**, so regeneration
   cannot destroy it. This matters right now: the `llama-rag.service` an agent has just
   written carries a genuinely valuable comment block recording that this invocation was
   first hand-launched with no unit, then briefly `akula-llama.service` (enabled at boot,
   `Restart=on-failure`), and is now deliberately on-demand. A careless generator would
   erase that on its first run. So the envelope has a `history` list, that comment block is
   its first entries, and the renderer emits them. **The manifest must be able to hold
   everything the unit currently says, or migration is a loss.**

### The two verbs

Instantiation is one CLI with a verb per kind, because the kinds have different lifecycles:

```
csd-manifest validate [path...]        # schema + cross-checks; CI gate
csd-manifest render  <id>              # print the unit / the resolved config, write nothing
csd-manifest serve   <id> [--install]  # render → install → daemon-reload  (kind: served)
csd-manifest train   <id> [--dry-run]  # build the config and run it       (kind: trained)
csd-manifest verify  [--host H]        # the three-way reconciler, read-only
```

`train` does **not** generate a unit. A training run is a job, not a service; it terminates,
and wrapping it in a unit would only obscure its exit status. `train` resolves the manifest
into a `PretrainConfig` (or the VL equivalent) and calls the existing trainer. The existing
runner keeps its resume logic, its checkpoint interval, its batch/LR derivation and its
gate handling untouched — the manifest replaces the hardcoded `REGIONS` lookup and nothing
else.

The join to receipts is one addition and it is the point of the whole design:

```python
provenance = {
    "manifest": "csd/retrieve",
    "manifest_sha": "b7c1…",     # sha256 of the manifest file as launched
    ...
}
```

With that, a receipt is reproducible (the exact declaration is recoverable), and a
manifest's `measured` entries can cite `from_receipt: <path>` so a number in a declaration
points at the run that produced it. The loop closes in both directions.

### Where manifests live

**In the repo that owns the thing**, not centralised:

```
akula-ai-platform/config/manifests/served/*.yaml     # LocalAI aliases, llama.cpp servers
CogSynDelta/config/manifests/trained/*.yaml          # CSD regions
CogSynDelta/config/manifests/served/*.yaml           # CSD checkpoints once served
```

One schema, one id namespace, several roots — deliberately the same shape as
`receipt.py`'s `load_all(roots: list[Path])`. Centralising would fight both repos' existing
ownership and would put serving config in a training repo. The id namespace is fleet-wide
and unique; `owner` records which repo may edit a given id, and `validate` fails on a
duplicate id across roots.

---

## D. Migration — in order, and stopping is allowed

The LocalAI YAMLs are the best config in this tree. Rewriting them for uniformity's sake
would be a net loss and is not proposed. What follows is ordered by value-per-risk, and
each step is independently worth doing — if the sequence stops after step 2, the fleet is
still better off.

**Step 1 — Declare what is currently undeclared.** Write manifests for things with *no*
declaration at all. The `gpu-1080ti` llama-server is the known case and is already half
done. Zero risk: these files describe processes that exist and nothing reads them yet.

**Step 2 — Build `validate` and `verify`. Nothing else.** Read-only. Run `verify` against
all three GPU hosts and publish what it finds. This is where the design either justifies
itself or does not. If it finds no further orphans and no drift, steps 3-5 are genuinely
lower priority and that is a real result, not a failure.

**Step 3 — Move CSD's `REGIONS`/`VL_REGIONS` to manifests.** This is the one place where a
source of truth genuinely *moves*, and the one place where the payoff is unambiguous: a
training run becomes a file you can diff and a reviewer can approve, and a receipt gains a
manifest id to cite. Migration is mechanical — the dicts already contain exactly the fields
the `trained` body needs. Keep the dicts as a fallback for one release, delete them once a
run has reproduced from a manifest with matching held-out numbers. **Blocked until other
agents' in-flight edits to `csd-train-all.py` land.**

**Step 4 — Transcribe LocalAI's comments into sidecar manifests.** The LocalAI YAMLs stay
exactly as they are and remain what LocalAI reads. A manifest is added *beside* each,
carrying what the comments already say: upstream, revision, licence, quant, measurement,
exclusivity, status.

Do this **as human-reviewed transcription, never as a parser.** The comments are prose, a
parser will get some of them subtly wrong, and a subtly wrong provenance record is the
exact failure this scheme exists to prevent — a wrong revision in a manifest is worse than
no manifest, because the manifest will be believed. Thirteen files, transcribed and
checked, is an afternoon.

**Step 5 — Generate the LocalAI YAML from the manifest.** *Probably never.* It buys
deduplication of about six lines per model and costs a template, a regeneration step and a
new drift surface on files that currently work. Listed for completeness and recommended
against. Revisit only if a third serving runtime appears and the same model must be
described for all three.

### Left alone, deliberately

| File | Why it is not touched |
|---|---|
| `config/localai/models/*.yaml` | LocalAI reads them; they are good; a sidecar is additive. |
| `routing/edge-backends.json` | **Routing, not declaration.** A manifest says what a model *is*; routing says where traffic *goes*. Merging them would make every routing change a model change. |
| `routing/shortnames.json`, `use-cases.json` | Presentation and policy over aliases, downstream of manifests. |
| `hosts/cohort.json`, `hosts/*.json` | Host facts. Manifests reference these by id and must never copy them. |
| `config/mind/csd-regions.json` | The *architectural* catalogue — stream dims, router triggers, quantization policy, curriculum order. Overlaps a trained manifest only in `pretrain.corpus`, which becomes a reference to a manifest id rather than a duplicated string. |
| `pipeline/receipt.py` | Gains two `provenance` keys. The envelope itself is correct and stays. |

---

## E. What this must NOT do

The failure mode of manifest systems is becoming a second programming language in YAML.
Every one that got there arrived the same way: someone needed one conditional, then one
loop, then a template function, and three years later the config needs a debugger. The
boundary has to be stated before the first pressure to cross it, because each individual
crossing is always locally reasonable.

**1. No control flow.** No conditionals, no loops, no expressions, no arithmetic. The only
substitution permitted is named interpolation of *host facts already declared in
`cohort.json`* (`{{host.lan_ip}}`, `{{host.user}}`) from a closed, enumerated set. If a
manifest needs an `if`, the thing it describes has two configurations and deserves two
manifests.

**2. No inheritance, overlays, or merge keys.** `local/fast` and `local/general` share a
GGUF, and the pull toward `extends: local-fast` will be immediate. The answer is no: they
are two manifests that both name the same `source`. **Deduplicate the artifact, not the
declaration.** Inheritance is where these systems become unreadable, because you can no
longer answer "what does this actually say" by opening one file.

The distinction from a *reference*, which is allowed: a reference resolves to a named thing
you can open and read whole (a host id, a corpus id, another manifest's id). Inheritance
silently rewrites the fields of the file you are looking at. The corpus catalogue in
`csd-corpus-expand.py` is shared across regions and should stay one catalogue that
manifests cite by id — that is a reference, and it is fine.

**3. No arbitrary command escape hatch.** No `pre_start:`, no `command:`, no `args:` free
list, no `env:` with shell interpolation. The moment one manifest can carry a shell
fragment, *every* manifest is a shell script, the reconciler can no longer reason about any
of them, and validation is theatre. If a service genuinely needs a wrapper, the wrapper is
a named script in the repo — reviewable, testable, greppable — and the manifest names it.

**4. Not a scheduler, not a router, not a package manager.** It does not decide *when*
something runs (`gpu-timeshare` does), *where traffic goes* (`edge-backends.json` does), or
*fetch weights* (`pull-models` and `csd-corpus-expand.py` do). It declares what a thing is
and what it costs, and the reconciler compares that to reality. Every one of those three
adjacent jobs already has an owner on this fleet, and absorbing them is how a manifest
system becomes the thing everything depends on and nobody can change.

**5. No secrets, structurally.** The schema has **no field capable of holding a secret
value.** A manifest names an `EnvironmentFile` and a variable name; it can never hold the
value. `AGENTS.md` hard rule 2 — *never print secret values; name the file and variable* —
becomes a property of the schema rather than a thing reviewers must catch, so forgetting
fails closed. The existing `llama-rag.service` already does this correctly with
`${LLAMA_API_KEY}` sourced from `%h/.config/akula/llama.env`; the manifest records exactly
that indirection and nothing more.

**6. It must not become the only record.** A manifest that says a thing exists does not
make it exist. The honest state of the fleet is `verify`'s output — declared-and-running,
running-but-undeclared, declared-but-absent — not the manifest directory listing. The set
of manifests is a **claim, continuously checked against the hosts**, and any tool or
dashboard built on it must present it that way. This is `AGENTS.md`'s standing rule (*don't
trust a doc over a live probe*) applied to the artifact this design creates, which is
otherwise the most likely thing on this fleet to be trusted without checking.

### Adversarial pass

Required at design time by `AGENTS.md` for anything crossing a trust boundary. This does:
manifests are merged by PR, this fleet has autonomous merge authority, and the output is an
`ExecStart` running on a GPU host as the operator's uid.

**What is worth taking:** the ability to place arbitrary argv on a GPU host as `tzervas`,
with access to the model store, the LAN, and the Forgejo and age keys on those hosts.

**How I would take it:** open a PR that adds or edits a manifest. If rendering interpolates
freely, `extra_args: ["; curl … | sh"]` is remote code execution through a merged YAML
file, reviewed as "config". A `pre_start` hook is the same thing with better manners. Both
are the natural next feature request.

**What closes it by construction:**

- The schema has no free-form command or argument field (boundary 3). There is no string in
  a manifest that reaches a shell.
- Rendering is a **fixed template with a closed set of typed substitutions**. The renderer
  cannot emit an `ExecStart` that is not derived from validated fields, and never
  concatenates a manifest string into a shell context — argv is built as a list.
- Every substituted value is type- and range-checked: `weights` must resolve under an
  allowed model root on the target host; `port` an integer in a permitted range; `hosts`
  known ids from `cohort.json`; **`bind` must equal the target host's declared LAN IP.**
  That last check turns hard rule 1 — *never bind `0.0.0.0`* — from a review item into a
  validation failure, which is the test `AGENTS.md` sets: it fails closed when someone
  forgets.
- `serve --install` writes only files whose content the renderer produced from a manifest
  that passed `validate`. `validate` runs in CI on every manifest change.
- Privilege reduction over monitoring: generated units stay **user** units under the
  existing `Linger=yes` model. Nothing in this design needs root, and the generator must
  never acquire the ability to write system units.

---

## The schema

JSON Schema (draft 2020-12), authored here for review. On implementation it lands at
`config/manifests/schema/manifest-v1.schema.json` and is what `csd-manifest validate` runs.
Manifests are written as YAML and validated as their parsed JSON, matching how the fleet
already treats the LocalAI configs.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://vectorweight.com/schema/model-manifest/v1",
  "title": "model-manifest/v1",
  "type": "object",
  "required": ["schema", "id", "kind", "status", "source", "licence",
               "hosts", "measured", "owner", "updated", "spec"],
  "additionalProperties": false,

  "properties": {
    "schema":  { "const": "model-manifest/v1" },
    "id":      { "type": "string", "pattern": "^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._-]*$",
                 "description": "Fleet-unique. Joins to receipt provenance.manifest." },
    "kind":    { "enum": ["served", "trained"] },
    "status":  { "enum": ["active", "on-demand", "benchmark-only",
                          "planned", "convert-pending", "retired"] },
    "owner":   { "type": "string", "description": "Repo that may edit this id." },
    "updated": { "type": "string", "format": "date" },
    "description": { "type": "string" },

    "source": {
      "type": "object",
      "required": ["upstream", "revision"],
      "additionalProperties": false,
      "properties": {
        "upstream": { "type": "string",
                      "description": "Repo of record, or manifest:<id> for a fleet-produced artifact." },
        "revision": { "type": "string", "minLength": 7,
                      "description": "A commit or immutable ref. A moving tag is not a pin." },
        "artifact": { "type": "string", "description": "File within the upstream repo, when one file is the artifact." },
        "sha256":   { "type": "string", "pattern": "^[0-9a-f]{64}$" },
        "bytes":    { "type": "integer", "minimum": 0 },
        "derivation": { "type": "string",
                        "description": "How the local artifact was produced from upstream, e.g. a convert-quantize step." }
      }
    },

    "licence": {
      "type": "object",
      "required": ["observed", "verdict", "checked_utc"],
      "additionalProperties": false,
      "properties": {
        "observed":      { "type": "string", "description": "Verbatim, as it actually appeared." },
        "verdict":       { "enum": ["SERVE_OK", "TRAIN_OK", "EVAL_ONLY", "REJECTED"] },
        "checked_utc":   { "type": "string", "format": "date" },
        "evidence":      { "type": "string", "description": "Where it was read. URL plus revision." },
        "upstream_says": { "type": "string",
                           "description": "What the ORIGINAL source says when this re-hosts someone else's work. A mirror's tag is not evidence about its upstream." },
        "caveat":        { "type": "string" }
      }
    },

    "hosts": {
      "type": "array", "minItems": 1, "uniqueItems": true,
      "items": { "type": "string" },
      "description": "Host ids from cohort.json. Never restate host facts here."
    },

    "conflicts_with": {
      "type": "array", "uniqueItems": true, "items": { "type": "string" },
      "description": "Manifest ids that must not be co-resident on the same card."
    },

    "measured": {
      "type": "array",
      "description": "Every resource number the manifest asserts. May be empty; may not contain a guess.",
      "items": {
        "type": "object",
        "required": ["what", "value", "unit", "on", "host", "method"],
        "additionalProperties": false,
        "properties": {
          "what":       { "type": "string", "examples": ["vram_mib", "load_seconds", "tok_per_s", "recall@1", "step_seconds"] },
          "value":      { "type": "number" },
          "unit":       { "type": "string" },
          "on":         { "type": "string", "format": "date" },
          "host":       { "type": "string" },
          "conditions": { "type": "string" },
          "method":     { "type": "string", "description": "How it was measured. Reproducible by a reader." },
          "from_receipt": { "type": "string", "description": "Receipt that produced this number, if any." },
          "supersedes": {
            "type": "object",
            "required": ["value", "why"],
            "additionalProperties": false,
            "properties": {
              "value": { "type": "number" },
              "basis": { "type": "string" },
              "why":   { "type": "string", "description": "Why the old figure was wrong. This is the durable part." }
            }
          }
        }
      }
    },

    "estimated": {
      "type": "array",
      "description": "Planning figures. Admission control MUST NOT read these.",
      "items": {
        "type": "object",
        "required": ["what", "value", "unit", "basis"],
        "additionalProperties": false,
        "properties": {
          "what":  { "type": "string" },
          "value": { "type": "number" },
          "unit":  { "type": "string" },
          "basis": { "type": "string", "description": "How it was derived. Required: an estimate without a basis is a rumour." },
          "measure_by": { "type": "string", "format": "date" }
        }
      }
    },

    "history": {
      "type": "array",
      "description": "Rendered into generated units so regeneration cannot destroy it.",
      "items": {
        "type": "object",
        "required": ["on", "what"],
        "additionalProperties": false,
        "properties": {
          "on":   { "type": "string", "format": "date" },
          "what": { "type": "string" },
          "why":  { "type": "string" }
        }
      }
    },

    "spec": { "type": "object" }
  },

  "allOf": [
    {
      "if":   { "properties": { "kind": { "const": "served" } } },
      "then": { "properties": { "spec": { "$ref": "#/$defs/servedSpec" } } }
    },
    {
      "if":   { "properties": { "kind": { "const": "trained" } } },
      "then": { "properties": { "spec": { "$ref": "#/$defs/trainedSpec" } } }
    }
  ],

  "$defs": {

    "servedSpec": {
      "type": "object",
      "required": ["runtime", "weights", "context_size", "endpoint"],
      "additionalProperties": false,
      "properties": {
        "runtime":  { "enum": ["llama-cpp", "localai", "comfy", "vllm", "custom-unit"] },
        "alias":    { "type": "string", "description": "Name the runtime serves it under, when it differs from id." },
        "weights":  { "type": "string", "description": "Absolute path on the target host. Validated to resolve under an allowed model root." },
        "mmproj":   { "type": "string" },
        "quant":    { "type": "string", "examples": ["Q4_K_M", "Q5_K_M", "MXFP4", "F16"] },
        "context_size":   { "type": "integer", "minimum": 1 },
        "native_context": { "type": "integer", "minimum": 1,
                            "description": "What the model supports. Differs from context_size on purpose; the gap is a VRAM decision." },
        "kv_dtype": { "enum": ["f16", "q8_0", "bf16"] },
        "n_gpu_layers": { "type": "integer", "minimum": 0 },
        "parallel": { "type": "integer", "minimum": 1 },
        "ttl_seconds": { "type": "integer", "minimum": 0 },
        "residency": { "enum": ["resident", "on-demand", "swap"] },
        "endpoint": {
          "type": "object",
          "required": ["bind", "port", "protocol"],
          "additionalProperties": false,
          "properties": {
            "bind":     { "type": "string",
                          "description": "MUST equal the target host's declared LAN IP. 0.0.0.0 is a validation failure, not a review comment." },
            "port":     { "type": "integer", "minimum": 1024, "maximum": 65535 },
            "protocol": { "enum": ["openai-v1", "comfy-api", "http"] },
            "auth":     {
              "type": "object",
              "required": ["env_file", "env_var"],
              "additionalProperties": false,
              "properties": {
                "env_file": { "type": "string" },
                "env_var":  { "type": "string" }
              },
              "description": "Names the indirection only. No field here can hold a value."
            }
          }
        },
        "unit": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "name":    { "type": "string" },
            "scope":   { "const": "user", "description": "System units are out of scope by design." },
            "enable_at_boot": { "type": "boolean", "default": false },
            "restart": { "enum": ["no", "on-failure", "always"], "default": "no" },
            "wrapper": { "type": "string", "description": "Named script in the repo. Never an inline command." }
          }
        }
      }
    },

    "trainedSpec": {
      "type": "object",
      "required": ["objective_family", "objective", "sources", "encoder", "schedule", "gates"],
      "additionalProperties": false,
      "properties": {
        "region": { "type": "string", "description": "Region name in csd-regions.json, when this trains one." },
        "objective_family": { "enum": ["contrastive-pair", "jepa-predictive", "reconstruction", "rank"] },
        "objective": { "type": "string" },
        "sources": {
          "type": "array", "minItems": 1,
          "items": {
            "type": "object",
            "required": ["corpus", "shards", "licence_ref"],
            "additionalProperties": false,
            "properties": {
              "corpus":  { "type": "string", "description": "Corpus catalogue id. Carries its own licence record." },
              "shards":  { "type": "string", "description": "Glob relative to the corpus root. Pin the config; never span configs." },
              "columns": { "type": "array", "items": { "type": "string" }, "minItems": 1, "maxItems": 2 },
              "image_column": { "type": "string" },
              "label_column": { "type": "string" },
              "cap":     { "type": "integer", "minimum": 0,
                           "description": "0 = uncapped. Exists for BALANCE, not speed." },
              "cap_why": { "type": "string" },
              "licence_ref": { "type": "string", "description": "Corpus manifest id whose verdict must be TRAIN_OK." },
              "note":    { "type": "string" }
            }
          }
        },
        "eval": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "holdout_pairs": { "type": "integer", "minimum": 1 },
            "held_out":  { "type": "string", "description": "Corpus id for out-of-domain eval, when it differs from training." },
            "transfer":  { "type": "string", "description": "A DIFFERENT dataset, to measure transfer rather than memorisation." },
            "transfer_columns": { "type": "array", "items": { "type": "string" } },
            "graded":    { "type": "string" },
            "metric":    { "type": "string", "examples": ["recall@1", "mrr", "linear-probe-top1", "spearman"] },
            "contamination_check": { "type": "boolean", "default": true }
          }
        },
        "encoder": {
          "type": "object",
          "required": ["type"],
          "additionalProperties": false,
          "properties": {
            "type":    { "type": "string", "examples": ["text-encoder", "jepa-predictor", "latent-vae"] },
            "dim":     { "type": "integer", "minimum": 1 },
            "depth":   { "type": "integer", "minimum": 1 },
            "n_heads": { "type": "integer", "minimum": 1 },
            "max_len": { "type": "integer", "minimum": 1 },
            "latent_dim":  { "type": "integer", "minimum": 1 },
            "patch_size":  { "type": "integer", "minimum": 1 },
            "tokenizer":   { "type": "string" }
          }
        },
        "schedule": {
          "type": "object",
          "required": ["steps", "batch_size"],
          "additionalProperties": false,
          "properties": {
            "steps":      { "type": "integer", "minimum": 1 },
            "batch_size": { "type": "integer", "minimum": 1 },
            "lr":         { "type": ["number", "string"],
                            "description": "A number, or \"derived\" to use the runner's sqrt(B/B0) rule. Never both." },
            "warmup_steps":     { "type": "integer", "minimum": 0 },
            "checkpoint_every": { "type": "integer", "minimum": 0 },
            "grad_clip": { "type": "number" },
            "bf16":      { "type": "boolean" },
            "seed":      { "type": "integer" }
          }
        },
        "gates": {
          "type": "object",
          "minProperties": 1,
          "description": "Declared thresholds. Names MUST match the receipt's gates keys. A run with no gates is not a pass.",
          "additionalProperties": {
            "type": "object",
            "required": ["rule"],
            "additionalProperties": false,
            "properties": {
              "rule":      { "enum": ["beats_baseline", "at_least", "at_most", "within_budget"] },
              "metric":    { "type": "string" },
              "threshold": { "type": "number" },
              "why":       { "type": "string" }
            }
          }
        },
        "outputs": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "checkpoint_dir": { "type": "string" },
            "receipt_dir":    { "type": "string" },
            "serves_as":      { "type": "string", "description": "Manifest id this run's artifact feeds, closing the trained→served chain." }
          }
        }
      }
    }
  }
}
```

Two constraints the schema cannot express and `validate` enforces in code:

- **Every host in `hosts` must accept this spec.** `gpu-1080ti` declares
  `never: [BF16, FP8, …]` in `edge-backends.json`, so `kv_dtype: bf16` with that host
  listed is rejected on the host's own constraint.
- **`endpoint.bind` must equal the target host's declared LAN IP** from `cohort.json` /
  `hosts/*.json`. This is hard rule 1 made structural.

---

## Worked examples

Three real cases, from the three situations this design has to cover: a model that is
already well described, one that was described nowhere, and a training run that lives in a
Python dict.

### 1. An existing LocalAI model — `local/code`

Everything here is transcribed from `config/localai/models/local-code.yaml`'s comment
block. Nothing is invented, and nothing is lost.

```yaml
schema: model-manifest/v1
id: local/code
kind: served
status: active
owner: akula-ai-platform
updated: 2026-09-02
description: >
  Autodev / code default. Served native 32k so autodev KV fills the card; the
  leftover is share-small helpers (embed/8B), never a second 14B.

source:
  upstream: huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF
  revision: d0a692ef765eefbf2fabb130b3cb2e8917e3d225
  artifact: qwen2.5-coder-14b-instruct-q4_k_m.gguf
  bytes: 8988000000        # 8.37 GiB, per the YAML comment

licence:
  observed: "Apache-2.0"
  verdict: SERVE_OK
  checked_utc: 2026-08-31
  evidence: "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF model card @ d0a692e"

hosts: [akula-prime]
conflicts_with: [local/uncensored, local/uncensored-code, local/code-alt]

measured:
  - what: vram_mib
    value: 11980                       # ~11.7 GiB
    unit: MiB
    on: 2026-08-31
    host: akula-prime
    conditions: "ctx 16384, f16 KV, parallel 1; 11.7 GiB of the card's 23028"
    method: "resident VRAM observed while loaded under LocalAI"
    supersedes:
      value: 22528
      basis: "analytic KV estimate that did not account for GQA"
      why: >
        The estimate was 1.9x the truth and had been used to argue this model
        could not share the card. Kept so nobody re-derives it.

spec:
  runtime: localai
  alias: local/code
  weights: /models/localai/qwen2.5-coder-14b-instruct-q4_k_m.gguf
  quant: Q4_K_M
  context_size: 32768
  native_context: 32768
  kv_dtype: f16
  parallel: 1
  ttl_seconds: 3600
  residency: resident
  endpoint:
    bind: 192.168.1.98
    port: 8080
    protocol: openai-v1
    auth:
      env_file: /etc/akula/localai.env
      env_var: LOCALAI_API_KEY
```

Note what this manifest does **not** do: it does not restate `vram_mib: 23028` (that is
`cohort.json`'s), it does not say which prompts route here (`use-cases.json`'s), and it
does not replace `local-code.yaml`, which LocalAI keeps reading unchanged.

### 2. The hand-launched llama-server — `local/pascal-fast`

The gap case. Its configuration lived in `/proc/<pid>/cmdline` for twenty-one hours; an
agent has since written `config/systemd/llama-rag.service`. This manifest renders **that
unit**, including its history — and demonstrates the honesty mechanism, because the VRAM
figure that circulates for it is not actually a measurement.

```yaml
schema: model-manifest/v1
id: local/pascal-fast
kind: served
status: on-demand
owner: akula-ai-platform
updated: 2026-09-02
description: >
  On-demand chat model on the 1080 Ti. Normally stopped so its VRAM is free for
  RAG and batch GPU work. A connection refusal on :8080 means it is not loaded,
  not that the backend is faulty.

source:
  # Byte-identical to the artifact local/uncensored-fast serves on akula-prime.
  # Two manifests, one artifact: deduplicate the artifact, never the declaration.
  upstream: huggingface.co/mradermacher/Huihui-Qwen3-8B-abliterated-v2-GGUF
  revision: 64c8e52a68276ef181b7695813c5883070b783a0
  artifact: Huihui-Qwen3-8B-abliterated-v2.Q4_K_M.gguf
  sha256: 0bdba3b32d450374b3c404d9ad531703db139b33798149efc84e2bce45eedbb5
  bytes: 5025000000        # 4.68 GiB

licence:
  observed: "Apache-2.0"
  verdict: SERVE_OK
  checked_utc: 2026-08-30
  upstream_says: >
    huihui-ai/Huihui-Qwen3-8B-abliterated-v2, Apache-2.0. Recorded because
    mradermacher re-hosts it: a mirror's tag is not evidence about its upstream.
  evidence: "mradermacher GGUF card @ 64c8e52, upstream huihui-ai card"

hosts: [gpu-1080ti]

measured: []              # honestly empty. See `estimated`.

estimated:
  - what: vram_mib
    value: 7168
    unit: MiB
    basis: >
      "~7 GiB resident" as stated in edge-backends.json's rag_contention note.
      Plausible for an 8B Q4_K_M at 16k f16 KV, but it is a figure someone wrote
      down, not one anyone recorded measuring. Admission control must not use it.
    measure_by: 2026-09-30

history:
  - on: 2026-09-01
    what: "Hand-launched llama-server, no unit"
    why: >
      Bare PID, no restart policy, and the full invocation existed only in
      /proc/<pid>/cmdline for 21 hours. Stopping it would have destroyed the
      only record. This manifest exists because of this entry.
  - on: 2026-09-02
    what: "Wrapped as akula-llama.service: enabled at boot, Restart=on-failure"
    why: "Always-up, to stop the invocation from being lost again."
  - on: 2026-09-02
    what: "Renamed llama-rag.service and made on-demand: not enabled, Restart=no"
    why: >
      The RAG model does not need to stay hot. Pure search runs on CPU, so
      loading only on demand frees ~7 GiB of the 1080 Ti's 11 GiB for batch work.

spec:
  runtime: llama-cpp
  alias: local/pascal-fast
  weights: /home/tzervas/models/Huihui-Qwen3-8B-abliterated-v2.Q4_K_M.gguf
  quant: Q4_K_M
  context_size: 16384
  native_context: 32768
  kv_dtype: f16           # Pascal sm_61: no BF16, no FP8. Rejected against the
                          # host's own `never` list if anyone changes this.
  n_gpu_layers: 99
  parallel: 1
  residency: on-demand
  endpoint:
    bind: 192.168.1.243   # validated == gpu-1080ti's declared LAN IP; 0.0.0.0 fails
    port: 8080
    protocol: openai-v1
    auth:
      env_file: ~/.config/akula/llama.env
      env_var: LLAMA_API_KEY
  unit:
    name: llama-rag.service
    scope: user
    enable_at_boot: false     # deliberately no [Install] section
    restart: no
```

Rendering this reproduces the installed unit, argv-for-argv:

```
ExecStart=%h/llama.cpp/build/bin/llama-server \
  --model %h/models/Huihui-Qwen3-8B-abliterated-v2.Q4_K_M.gguf \
  --alias local/pascal-fast --host 192.168.1.243 --port 8080 \
  --n-gpu-layers 99 --ctx-size 16384 --parallel 1 --api-key ${LLAMA_API_KEY}
```

That round trip is the acceptance test for the whole served schema: **if the manifest
cannot reproduce the unit that exists, the schema is short a field.** Note also that
`measured: []` here is the design working, not a hole in it — the fleet's belief about this
model's VRAM turns out to be an estimate, and the schema is what surfaced that.

### 3. A CSD training region — `csd/retrieve`

Transcribed from `REGIONS["retrieve"]` in `scripts/csd-train-all.py`, plus the runner's
config construction and its measured sizing note. This is the manifest that makes a
training run diffable.

```yaml
schema: model-manifest/v1
id: csd/retrieve
kind: trained
status: active
owner: CogSynDelta
updated: 2026-09-02
description: "Query -> passage rank. Curriculum step 1: per-region pretrain."

source:
  upstream: manifest:csd/architecture     # region catalogue, not a Hub repo
  revision: 11836108a417f0947f9e20a3117d5dd0a61a29f3   # commit last touching config/mind/csd-regions.json
  derivation: "cogsyndelta.regions.pretrain_region over the sources below"

licence:
  observed: "Per-source; see sources[].licence_ref"
  verdict: TRAIN_OK
  checked_utc: 2026-09-02
  caveat: >
    SciFact and NFCorpus are EVAL_ONLY (NC / ToS) and are deliberately absent
    from `sources`. They may be referenced under `eval`, never under `sources`.

hosts: [akula-prime]

measured:
  - what: recall@1
    value: 0.006
    unit: ratio
    on: 2026-09-02
    host: akula-prime
    conditions: "batch 256, 4,986 fiqa pairs, held out on fiqa dev/test"
    method: "held-out eval in pretrain_region; see receipt"
    from_receipt: /akula-data/csd/receipts/cogsyndelta-retrieve-pretrain-<stamp>.json
  - what: step_seconds
    value: 0.069
    unit: s
    on: 2026-09-02
    host: akula-prime
    conditions: "batch 256, max_len 96, fp32"
    method: "receipt elapsed_s / steps"

estimated: []

history:
  - on: 2026-09-02
    what: "Multi-source: natural-questions and gooaq added, gooaq capped at 400k"
    why: >
      At 4,986 fiqa pairs the region was data-starved, not failing to train
      (recall@1 0.006). Uncapped gooaq is 3,012,496 pairs -- 96% of everything
      available -- so training uncapped would produce a gooaq model wearing a
      retrieval region's name.

spec:
  region: retrieve
  objective_family: contrastive-pair
  objective: rank
  sources:
    - corpus: fiqa-pairs
      shards: "region/retrieve/fiqa-pairs/train.parquet"
      columns: [query, passage]
      cap: 0
      licence_ref: corpus/fiqa
    - corpus: natural-questions
      shards: "region/retrieve/natural-questions/**/train*.parquet"
      columns: [query, answer]
      cap: 0
      licence_ref: corpus/natural-questions
    - corpus: gooaq
      shards: "region/retrieve/gooaq/**/train*.parquet"
      columns: [question, answer]
      cap: 400000
      cap_why: >
        Balance, not speed. Capped at 400k it is 78% of a ~514k mix, comparable
        to code (455k) and compress (320k). Raise it if transfer is the
        bottleneck -- that is a measurement, not a guess.
      licence_ref: corpus/gooaq
  eval:
    holdout_pairs: 512
    held_out: fiqa-dev-test
    metric: "recall@1"
    contamination_check: true
  encoder:
    type: text-encoder
    dim: 256
    depth: 4
    n_heads: 4          # 256 is not divisible by the 6-head default
    max_len: 96
    tokenizer: /mnt/fleet-datasets/tritter/gpt2_tokenizer.json
  schedule:
    steps: 8000
    batch_size: 1280
    lr: derived         # runner's sqrt(B/B0) rule; never stated alongside a number
    warmup_steps: 533
    checkpoint_every: 200
    bf16: true
  gates:
    beats_untrained:
      rule: beats_baseline
      metric: "recall@1"
      why: >
        A random-init encoder scores recall@1 0.40 on CodeSearchNet from lexical
        overlap alone, and early training DESTROYS that before learned structure
        replaces it. Judging a run without the untrained baseline produces
        exactly the wrong conclusion.
  outputs:
    checkpoint_dir: /akula-data/csd/checkpoints/retrieve
    receipt_dir: /akula-data/csd/receipts
```

`csd-manifest train csd/retrieve` resolves this into the existing `PretrainConfig` and
calls the existing `pretrain_region`. Resume, checkpoint interval, LR derivation and gate
evaluation are unchanged — the manifest replaces the `REGIONS[name]` lookup and nothing
else. The receipt it emits gains `provenance.manifest: csd/retrieve` and
`provenance.manifest_sha`, so the run is reproducible from the declaration and the
declaration's `measured` entries can cite the run.

### 3b. Why the VL region proves the sub-discriminator

```yaml
schema: model-manifest/v1
id: csd/vl-latent
kind: trained
status: active
# ... envelope identical in shape to csd/retrieve ...
spec:
  region: vl_latent
  objective_family: jepa-predictive      # NOT contrastive-pair
  objective: "latent prediction over image patches"
  sources:
    - corpus: tiny-imagenet
      shards: "vl/tiny-imagenet/data/train-*.parquet"
      image_column: image                # no `columns` pair; there is no pair
      label_column: label
      cap: 0
      licence_ref: corpus/tiny-imagenet
  eval:
    transfer: "vl/cifar100/cifar100/test-*.parquet"
    transfer_columns: [img, fine_label]
    metric: "linear-probe-top1"
  encoder:
    type: jepa-predictor
    patch_size: 8
    latent_dim: 64
  gates:
    probe_beats_untrained:
      rule: beats_baseline
      metric: "linear-probe-top1"
      why: "Gated on a linear probe, never on loss. A collapsed encoder has excellent loss."
```

If `trained` were one flat body, `columns` would be optional-and-usually-null and nothing
could check that a `jepa-predictive` region declares an image column, or that a
`contrastive-pair` region declares exactly two. The sub-discriminator is what lets the
validator say *"this region declares a probe eval but no probe columns"* — and it is the
same distinction `csd-train-all.py` already drew when it refused to bend `VL_REGIONS` into
`REGIONS`.

---

## What to build first

1. `csd-manifest validate` and the schema file. CI gate on manifest changes.
2. `csd-manifest verify --host <h>`. Read-only. Run it against all three GPU hosts and
   publish the four-state report. **This is the step that either justifies the rest or
   right-sizes it.**
3. Manifests for the undeclared: `local/pascal-fast` and whatever step 2 turns up.
4. `csd-manifest train`, plus the two `provenance` keys in `receipt.py`.
5. `csd-manifest serve --install`, with the round-trip test against `llama-rag.service`.

Steps 1-3 change no existing file and can land while other work is in flight. Step 4 waits
on the in-flight edits to `csd-train-all.py`. Step 5 is the only one that writes to a host,
and it should not be built until `verify` has run clean for a while.
