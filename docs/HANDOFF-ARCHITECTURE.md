---
title: CogSynDelta + Memory-Gate architecture handoff
status: working-context
indexed: 2026-08-30
audience: Codex / Grok / humans
truth: STATUS.md remains the measured-capability source. This document is unvetted ChatGPT design context. Reconcile every claim against live Python source before treating it as real. Python-first; Rust rewrite only after the Python path is proven.
---

# CogSynDelta + Memory-Gate — Architecture Handoff Context

## Purpose

This handoff summarizes the current architectural discussion around:

* `CogSynDelta`
* `memory-gate`
* `memory-gate-rs`
* heterogeneous transformer / recurrent / memory architectures
* STM / MTM / LTM organization
* persona persistence
* reversible online/model adaptation
* possible relationship to Titans, ATLAS, Nested Learning/HOPE, Gated DeltaNet, Mamba, JEPA, VSA, etc.

The goal is **not** to immediately implement every architecture discussed.

The goal is to preserve the original design intent, clarify boundaries between projects, identify promising research directions, and establish a controlled path for implementation and experimentation.

**This file is still context, not a spec.** It was drafted in ChatGPT and **must be vetted against the live trees** (`CogSynDelta` Python, `memory-gate` Python, `memory-gate-rs` as a later target). `STATUS.md`, tests, and source win over every section below.

**Implementation order (operator, 2026-08-30):**

1. **Python first** for CogSynDelta and Memory-Gate. The PyTorch / scientific-Python ecosystem is the place to prove regions, routing, memory tiers, and consolidation. Do not start a Rust port to “do it right.”
2. After the Python path is **measured and complete enough to trust**, rewrite hot paths in Rust (`memory-gate-rs` and later CogSynDelta-rs) for **performance, memory safety, speed, and efficiency**.
3. `memory-gate-rs` remains a research/reference tree (VSA / `HolographicStore`). It is **not** the current implementation vehicle.

---

# 1. High-Level Thesis

The combined system should be understood roughly as:

```text
CogSynDelta
    =
neural cognition / model architecture

Memory-Gate
    =
persistent cognitive state
+ heterogeneous memory management
+ persona persistence
+ consolidation
+ retrieval
+ reversible adaptation policy

MCP / context services
    =
interoperability and external integration surface
```

These systems should remain **loosely coupled where practical** and tightly coupled only where inference latency requires it.

The broader research thesis is:

> Instead of one homogeneous Transformer repeatedly applying the same computation, construct a cognitive architecture consisting of heterogeneous computational regions, multiple forms and timescales of memory, learned routing, latent prediction, adaptive internal computation, and persistent-but-reversible learning.

---

# 2. CogSynDelta: Current State vs Intended Architecture

CogSynDelta is currently a research scaffold / PoC rather than a fully realized cognitive architecture.

The repository's `STATUS.md` should remain the authoritative description of demonstrated capabilities.

Current implemented concepts include:

* shared latent / residual stream
* cognitive regions functioning similarly to heterogeneous MoE experts
* learned router
* softmax top-k dispatch
* load-balancing objective
* lightweight residual MLP region implementations
* tiered memory experiments
* exploratory vision / JEPA-related modules
* exploratory predictive-coding / VAE components

Important distinction:

> CogSynDelta is MoE-adjacent, but the intended regions do not need to be copies of the same architecture.

This is potentially one of the project's strongest differentiators.

A region may eventually use fundamentally different state-transition mathematics from another region.

Example:

```text
Language region
    → attention + Gated DeltaNet

Temporal region
    → Mamba / SSM

Reasoning region
    → recursive Transformer / latent recurrent reasoner

Perception region
    → JEPA

Memory/control region
    → Memory-Gate integration

Symbolic region
    → VSA / Hopfield / structured representations
```

All may communicate through a common latent workspace.

---

# 3. Do Not Pick a Single “Best Transformer”

CogSynDelta should expose a generic cognitive-region abstraction rather than standardizing every region on one Transformer.

Conceptually:

```python
Region(
    mixer=...,
    state_model=...,
    recurrence=...,
    memory_interface=...,
    objective=...,
    depth_policy=...,
)
```

The purpose of experiments is to determine which computational substrate is best for which role.

Potential candidates discussed:

1. Hybrid softmax attention + Gated DeltaNet-2
2. Gated DeltaNet / Kimi Delta Attention
3. Mamba-3
4. RWKV-7
5. Recursive Transformer / Mixture-of-Recursions
6. Continuous Thought Machine
7. Perceiver-style latent workspace
8. actual JEPA / V-JEPA / VL-JEPA objectives
9. Titans-style neural memory
10. ATLAS-style contextual neural memory
11. Nested Learning / HOPE / continuum-memory concepts
12. TTT / fast learned hidden-state models
13. VSA / hyperdimensional representations
14. Hopfield associative memory
15. Slot Attention
16. diffusion-style proposal mechanisms
17. Soft MoE
18. BLT / tokenizer-free representation experiments
19. true manifold-constrained Hyper-Connections

These are **experimental candidates**, not dependencies.

---

# 4. Recommended Near-Term CogSynDelta Experiment

The cleanest next architecture experiment is likely:

```text
existing CogSynDelta router
        ↓
parameter-matched CognitiveRegion implementations

A: current ResidualMLPRegion
B: tiny Transformer region
C: Gated DeltaNet / GDN2-like region
D: Mamba/SSM region
```

Do not simultaneously replace routing, memory, representation, and region architecture.

Preserve controlled attribution.

Metrics should include:

* training loss
* downstream task quality
* state tracking
* retrieval
* specialization
* router entropy
* expert/region utilization
* catastrophic forgetting
* throughput
* latency
* memory use
* quality per internal computation step

---

# 5. Gated DeltaNet Is a Particularly Strong CogSynDelta Candidate

Gated DeltaNet / GDN2 is attractive because it maintains a mutable recurrent state rather than relying purely on an ever-growing KV history.

Conceptually:

```text
state[t+1] = update(state[t], input[t])
```

GDN2 further distinguishes write and erase behavior.

This could provide CogSynDelta regions with genuine evolving fast state.

However, a **hybrid architecture** is preferred over pure DeltaNet:

```text
GDN
GDN
GDN
softmax attention
GDN
GDN
GDN
softmax attention
```

The recurrent blocks handle compressed evolving state.

Periodic full attention retains high-fidelity content-addressed retrieval.

This should initially be evaluated as a new `CognitiveRegion` implementation rather than rewriting CogSynDelta.

---

# 6. JEPA Clarification

The current CogSynDelta module called or associated with `VL-JEPA` is currently closer to a small ViT encoder than a complete JEPA training architecture.

A true JEPA-style architecture should contain conceptually:

```text
context input
    ↓
context encoder
    ↓
predictor
    ↓
predicted target representation

target input
    ↓
target encoder
    ↓
target representation

loss:
predicted target latent
vs
target latent
```

The important conceptual principle is:

> predict and reason in latent space; decode only where necessary.

Potential targets need not only be visual/textual.

CogSynDelta regions could predict:

* future observations
* future latent state
* another region's latent representation
* planned consequences
* concepts
* semantic state transitions

JEPA should therefore be treated as an architectural/objective principle rather than merely a vision module.

---

# 7. Shared Workspace

A Perceiver-style latent bottleneck is a strong candidate for the shared cognitive workspace.

Instead of requiring every modality and cognitive region to produce identical token structures:

```text
vision ──────┐
language ────┤
memory ──────┤
state ───────┼→ shared latent workspace
tools ───────┤
audio ───────┘
```

Regions can cross-attend to a fixed-dimensional latent workspace.

Potential future abstraction:

```text
W ∈ R^(num_slots × latent_dim)
```

Each region reads/writes through `W`.

---

# 8. Memory-Gate Is Not Merely a Memory Database

Historical intent is broader.

Memory-Gate should be treated as a:

> model-independent persistent cognitive-state, memory-lifecycle, persona-persistence, consolidation, retrieval, and reversible adaptation layer.

It is not fundamentally equivalent to Titans.

Titans is a learned neural-memory mechanism integrated into a sequence model.

Memory-Gate is intended to coordinate **multiple memory representations and services**.

Titans may eventually be mounted as one optional memory backend or benchmark.

It is not presently required.

---

# 9. Original Memory Tier Intent

The intended hierarchy is approximately:

```text
Active / Working Memory
    ↓
STM
    ↓
MTM
    ↓
LTM
    ↓
learned adaptation / behavioral effects
```

But these tiers differ not only in age.

They may use fundamentally different representations.

## Active / Working Memory

Purpose:

* currently manipulated state
* model-native cognition
* immediate inference

Possible representation:

* activations
* Perceiver workspace
* attention state
* GDN recurrent state

Lifetime:

* internal reasoning ticks
* current inference

---

## STM

Purpose:

* very hot recent context
* extremely fast retrieval

Primary representation:

* dense embeddings
* in-memory vector store

Potential placement:

* RAM
* VRAM
* hybrid RAM/VRAM

Exact hardware location remains an implementation decision.

STM should optimize for latency rather than maximal semantic structure.

---

## MTM

Purpose:

* richer episodic memory
* RAG-oriented retrieval
* recent-to-medium temporal horizon
* more metadata and semantic structure

Possible representation:

```text
dense embedding
+ raw/source reference
+ timestamps
+ episode ID
+ entities
+ semantic metadata
+ causal links
+ importance
+ confidence
+ retrieval statistics
```

MTM is intended to be a richer RAG-like memory system rather than merely a slower STM.

---

## LTM

Original concept:

* durable, associative, highly structured memory
* VSA / holographic representation
* long-running persona and knowledge persistence

Memory-Gate-RS already contains a `HolographicStore` / VSA direction including:

* binding
* bundling
* role/filler representations
* VSA codebook
* importance
* access counts
* analogical-style retrieval

This should remain an important research direction.

LTM should not simply be “a colder vector DB.”

A key long-term goal is that memory **changes representation during consolidation**.

Example:

```text
high-resolution event
    ↓
dense episodic representation
    ↓
semantic / relational abstraction
    ↓
VSA structure
```

---

# 10. Memory-Gate Python vs Rust

**Python is the current implementation vehicle.** Prove Memory-Gate (tiers, consolidation-that-creates, persona namespace, write gates) in Python first, next to CogSynDelta’s PyTorch regions. The AI/ML stack (torch, numpy, existing tests, CUDA PoC) is already there.

`memory-gate-rs` is architecturally interesting (VSA / `HolographicStore`) and is the **planned rewrite target** after the Python design is proven — for performance, memory safety, and efficiency — not the first place to land new behavior.

Do not treat the Rust tree as canonical runtime. Do not port a feature to Rust until the same feature has a measured Python path.

Important existing implementation caveat:

Current “consolidation” behavior is often closer to:

```text
old + low importance
    → prune/delete
```

than true cognitive consolidation.

That should be corrected conceptually and eventually in implementation.

---

# 11. Consolidation Must Produce Something New

True consolidation should perform transformations such as:

```text
episode A ─┐
episode B ─┤
episode C ─┼→ abstraction
episode D ─┤
episode E ─┘
```

Potential outputs:

* semantic concept
* relation graph / VSA bindings
* prototype embedding
* compressed episodic representation
* procedural rule
* learned adapter/delta update
* neural memory update

The original episodes may then be:

* retained
* compressed
* archived
* merged
* decayed
* eventually deleted

Deletion should generally be the final stage, not the definition of consolidation.

---

# 12. Titans Comparison

Titans overlaps with Memory-Gate in motivation but uses a different memory substrate.

## Titans

Strongest features:

* memory is a neural network
* memory parameters update online
* surprise / prediction error drives updates
* differentiable
* compact learned representation
* strong empirical sequence-model validation

Conceptually:

```text
experience
    ↓
memory loss
    ↓
gradient
    ↓
modify neural-memory weights
```

## Memory-Gate

Strongest intended features:

* heterogeneous memory backends
* explicit STM / MTM / LTM lifecycle
* persistent external storage
* rich episodic records
* provenance
* persona namespaces
* VSA / structured memory
* external services
* cross-session persistence
* auditable/reversible state
* future reversible model adaptation

Therefore:

```text
Titans
    =
neural-memory mechanism

Memory-Gate
    =
memory architecture / control substrate
```

Titans can potentially be implemented as:

```text
NeuralMemoryPlane
```

inside Memory-Gate later.

Do not architect Memory-Gate around Titans.

---

# 13. What to Borrow From Titans

The most useful idea to consider importing independently is **surprise-driven write strength**.

Instead of memory importance being determined solely by:

* recency
* manual importance
* access count

consider:

```text
surprise
prediction error
novelty
reward
uncertainty
causal significance
redundancy
```

Example conceptual gate:

```text
write_score =
f(
    surprise,
    prediction_error,
    novelty,
    reward,
    uncertainty,
    task_relevance,
    redundancy
)
```

This can feed the existing Memory-Gate control architecture without using a full Titans neural-memory implementation.

---

# 14. ATLAS-Like Improvement

A related idea worth testing is episode/window-level memory evaluation rather than token-level write decisions.

Instead of:

```text
should input[t] be remembered?
```

prefer:

```text
does input[t-k:t] constitute a meaningful event?
```

This aligns more naturally with MTM / hippocampal episodic encoding.

Potential future specialist:

```text
EventSegmenter
```

which detects meaningful memory boundaries.

---

# 15. Memory-Gate as Multiple Planes

A future Memory-Gate architecture should probably separate concerns into planes.

```text
Memory-Gate
│
├── Memory Control Plane
│   ├── write
│   ├── retrieve
│   ├── promote
│   ├── demote
│   ├── forget
│   └── consolidate
│
├── Storage Plane
│   ├── STM
│   ├── MTM
│   └── LTM
│
├── Semantic Plane
│   ├── entities
│   ├── relations
│   ├── chronology
│   └── provenance
│
├── Persona Plane
│   ├── persona namespace
│   ├── behavioral configuration
│   ├── memory scope
│   └── retrieval policy
│
└── Adaptation Plane
    ├── adapter routing
    ├── parameter deltas
    ├── learned offsets
    ├── validation
    └── rollback
```

This is more accurate than treating everything as one `KnowledgeStore`.

---

# 16. Memory Backend Interface Should Reflect Different Semantics

Dense vector stores, VSA memory, and neural-memory modules do not really expose identical semantics.

A generalized interface may eventually resemble:

```rust
trait MemoryPlane {
    fn write(...);
    fn recall(...);
    fn forget(...);
    fn consolidate(...);
    fn stats(...);
}
```

Potential declared capabilities:

```text
PERSISTENT
DIFFERENTIABLE
EXACT_RECALL
FUZZY_RECALL
SYMBOLIC
ASSOCIATIVE
TEMPORAL
TRAINABLE
EXPANDABLE
AUDITABLE
```

This avoids pretending that Qdrant, VSA, and a Titans-like neural memory are interchangeable storage engines.

---

# 17. Persona Persistence Is First-Class State

Historical intent includes configurable personas with persistent contextual and learned state.

A persona should eventually be more than a system prompt.

Conceptually:

```text
Persona
├── memory namespace
├── contextual state
├── retrieval policy
├── semantic/VSA associations
├── behavioral configuration
├── adaptation state
└── learned parameter deltas
```

Loading a persona may eventually:

```text
activate memory namespace
load persona semantic state
configure retrieval weighting
mount persona-specific model deltas
configure adaptation policy
```

Detaching the persona should restore baseline model behavior.

---

# 18. Memory Scoping and Authority

Memory should eventually distinguish at least:

```text
scope:
    global
    user
    persona:<id>
    project:<id>
    session:<id>
```

and ideally provenance / authority:

```text
authority:
    observed
    user_asserted
    imported
    inferred
    model_generated
```

and persistence:

```text
transient
episodic
durable
```

This becomes important before memories are allowed to influence long-term learned behavior.

---

# 19. Parameter-Delta / Offset Adaptation Is a Major Original Design Goal

One of the most important historical ideas is an intermediate layer between persistent memory/persona state and the base model's learned weights.

The base model should remain immutable:

```text
W0
```

The effective runtime weights become:

```text
W_eff = W0 + ΔW
```

Potential decomposition:

```text
W_eff
 =
W0
+ ΔW_global
+ ΔW_persona
+ ΔW_domain
+ ΔW_skill
+ ΔW_session
```

Each delta should be independently:

* mounted
* weighted
* gated
* versioned
* disabled
* rolled back

If the adaptive layer becomes unstable:

```text
disable all ΔW
```

and the model returns exactly to the base trained checkpoint.

This loose coupling / fallback property is a key design requirement.

---

# 20. LoRA Is an Implementation Primitive, Not the Full Architecture

A first implementation can use low-rank adapters:

```text
W_eff = W0 + BA
```

because they are:

* small
* detachable
* composable
* cheap to store
* cheap to train relative to full weights

However, the long-term design is broader than static persona LoRAs.

Possible future adaptation levels:

## Level 0 — Context Injection

```text
memory
→ retrieve
→ prompt/context
```

## Level 1 — Activation Modulation

```text
memory state
→ activation offsets/gates
```

Example:

```text
h' = h + γ(memory)
```

## Level 2 — Adapter Routing

```text
persona/context
→ mixture of existing ΔW adapters
```

Example:

```text
W_eff = W0 + Σ α_i ΔW_i
```

## Level 3 — Dynamic Delta Generation

```text
Memory-Gate state
→ small hypernetwork/controller
→ LoRA coefficients / adapter weights
```

## Level 4 — Online Delta Learning

```text
ΔW[t+1]
 =
ΔW[t] - η∇L
```

Only the detachable delta is modified.

The base model remains frozen.

---

# 21. Critical Safety / Stability Constraint

Arbitrary memories must **not** directly modify persistent parameter deltas.

There should be a consolidation barrier.

Preferred lifecycle:

```text
observation
    ↓
STM
    ↓
importance / recurrence / evidence
    ↓
MTM
    ↓
corroboration / semantic consolidation
    ↓
LTM
    ↓
adaptation candidate
    ↓
offline/sandbox evaluation
    ↓
accept or reject
    ↓
versioned ΔW
```

Every learned delta should retain provenance.

Example:

```text
delta:
    persona_adapter_v27

derived_from:
    memory traces [...]
    consolidation job ...
    objective scores ...
    previous adapter version ...
```

This enables:

* rollback
* audit
* debugging
* A/B testing
* forgetting/unlearning
* attribution

---

# 22. Tiny Specialist Models Are Part of the Long-Term Plan

Another original design direction is to train very small, highly specialized models for individual memory operations instead of routing everything through an LLM.

Potential specialists include:

```text
WriteGate
SalienceNet
NoveltyNet
EventSegmenter
PromotionNet
Consolidator
RelationExtractor
ConflictNet
RetentionNet
RetrievalRouter
Reranker
DeltaRouter
DeltaLearner
DeltaVerifier
```

Some may only require:

* classical algorithms
* small MLPs
* small Transformers
* tiny recurrent models
* classifiers/rankers

They do not all need generative LLMs.

The design preference is:

> tiny model, narrow responsibility, excellent performance on one function.

---

# 23. MCP / Context Services

Not every memory operation belongs inside CogSynDelta or its inference process.

Use coupling based primarily on latency sensitivity.

## Tightly Coupled

Must operate every token / reasoning tick:

```text
working memory
hot STM
recurrent state
adapter gates
active persona delta
```

Likely:

* in-process
* shared memory
* VRAM/RAM resident

## Medium Coupling

Can tolerate local IPC or service calls:

```text
MTM retrieval
episodic writes
semantic lookup
VSA queries
reranking
```

Potentially:

* Memory-Gate-RS daemon
* local IPC
* native library
* Unix socket / equivalent

## Loose Coupling

Not part of the critical inference loop:

```text
consolidation
archival
adapter training
cross-session persona management
memory maintenance
MCP exposure
long-running restructuring
```

These can operate as external processes/services.

---

# 24. CogSynDelta / Memory-Gate Integration Boundary

Preferred division of responsibility:

## CogSynDelta owns

```text
neural cognition
shared latent workspace
region routing
internal recurrence
model-native working memory
latent prediction
region architectures
adapter injection points
runtime delta gates
```

## Memory-Gate owns

```text
persistent memory
memory lifecycle
STM/MTM/LTM policy
persona state
episodic persistence
VSA LTM
retrieval routing
consolidation
adaptation policy
provenance
versioned learned deltas
```

## MCP/context services own

```text
external integration
cross-process API
tool exposure
remote clients
administration
cross-application persistence
```

---

# 25. Proposed Combined Architecture

Conceptually:

```text
                       INPUT / EVENTS
                             │
                             ▼
                    ┌─────────────────┐
                    │   CogSynDelta   │
                    │                 │
                    │ latent workspace│
                    │ region router   │
                    │ cognitive regions
                    │ recurrence      │
                    └────────┬────────┘
                             │
                 hot, low-latency interface
                             │
                             ▼
                  ┌─────────────────────┐
                  │     Memory-Gate     │
                  │                     │
                  │ memory controller   │
                  │ persona state       │
                  │ adaptation policy   │
                  └──────────┬──────────┘
                             │
          ┌──────────────────┼────────────────────┐
          │                  │                    │
          ▼                  ▼                    ▼
         STM                MTM                  LTM
    hot dense vectors   rich episodic      VSA / holographic
      RAM / VRAM          dense RAG            structured
          │                  │                    │
          └─────────┬────────┴──────────┬─────────┘
                    │                   │
                    ▼                   ▼
               consolidation       retrieval fusion
                    │
      ┌─────────────┼─────────────────┐
      │             │                 │
      ▼             ▼                 ▼
 semantic LTM   procedural state   adaptation candidates
      │                               │
      │                               ▼
      │                       specialist trainer
      │                               │
      │                        candidate ΔW
      │                               │
      │                           validation
      │                               │
      └───────────────────────────────┤
                                      ▼
                             versioned adapters
                                      │
                                      ▼
                              CogSynDelta runtime
```

---

# 26. Optional Neural Memory Plane

Titans / ATLAS / TTT / HOPE-like neural memory should remain optional.

Possible future Memory-Gate plane:

```text
NeuralLTM
├── Titans-like backend
├── ATLAS-like backend
├── TTT-like backend
└── CogSynDelta-native backend
```

The architecture must function without this plane.

Use it only if controlled experiments demonstrate incremental value.

---

# 27. Multi-Timescale Learning

The architecture naturally suggests different update frequencies:

```text
activation / workspace
    every internal tick

STM
    every relevant observation

MTM
    every meaningful event

LTM
    consolidation cycles

persona adapters
    slower validated updates

skill/domain adapters
    slower still

base model parameters
    effectively immutable
```

This is conceptually compatible with Nested Learning / continuum-memory research, but Memory-Gate should maintain its own abstraction rather than becoming an implementation of HOPE.

---

# 28. Memory-Gate-RS VSA Follow-Up Work

The Rust VSA/HolographicStore is promising but incomplete.

Items already identified:

## Temporal Encoding

The store tracks a temporal-position counter, but this position is not yet meaningfully bound into the encoded trace.

Implement explicit temporal representations.

Potential forms:

```text
POSITION ⊗ timestep_vector
PREVIOUS ⊗ event_A
NEXT ⊗ event_B
```

or suitable VSA permutation-based sequence encoding.

## Holographic Index

The aggregate holographic bundle currently does not appear to replace linear trace scanning during retrieval.

Benchmark and implement a real retrieval/index strategy rather than assuming the bundled vector provides an efficient index.

## Consolidation

Replace “old + low importance → delete” with actual transformation / abstraction / replay.

---

# 29. Existing CogSynDelta Memory Issues Previously Identified

When modifying the tiered-memory implementation, also review:

## Compression Documentation

The high-fidelity compressor stores:

```text
basis coefficients
+
full residual
```

in FP16.

Default compression appears closer to ~1.33× for the examined configuration rather than a universal 2×.

Do not call FP16 reconstruction mathematically lossless.

Prefer terminology like:

```text
high-fidelity compression
```

unless reconstruction is bit-exact.

## `ActiveMemoryManager.store`

Review the knowledge-base path where `compressed` may be referenced before assignment if an item remains in the active tier.

Add regression coverage.

---

# 30. Research Methodology

Avoid implementing all ideas simultaneously.

Every architectural experiment should answer a specific question.

Suggested progression:

## Stage 0 — Reproducible Baseline

Freeze current CogSynDelta PoC behavior.

Record:

```text
seed
config
loss
router distribution
throughput
memory use
```

## Stage 1 — Region Substrate

Compare:

```text
ResidualMLP
Transformer
Gated DeltaNet
Mamba
```

with minimal other changes.

## Stage 2 — Memory-Gate Integration

Implement:

```text
STM
MTM
LTM
```

with stable APIs.

Do not yet introduce learned weight adaptation.

## Stage 3 — Proper Consolidation

Test:

```text
episodic clusters
→ semantic/VSA consolidation
```

Measure information retention and retrieval usefulness.

## Stage 4 — Persona Layer

Implement persona-scoped memory and retrieval policies.

No online weight modification yet.

## Stage 5 — Static Delta Layer

Use detachable LoRA/adapters.

Manually or offline-train persona/domain/skill deltas.

Verify perfect fallback to base weights.

## Stage 6 — Delta Routing

Learn which existing deltas should activate based on:

```text
persona
task
memory state
context
```

## Stage 7 — Learned Delta Generation

Test a small controller/hypernetwork.

## Stage 8 — Online Delta Learning

Only after all provenance/versioning/rollback mechanisms exist.

## Stage 9 — Optional Neural Memory

Benchmark Titans/ATLAS/TTT-like neural memory against the full Memory-Gate hierarchy.

---

# 31. Important Non-Goals

Do not:

* rewrite CogSynDelta around Titans
* conflate Memory-Gate with Titans
* make every cognitive region the same model architecture
* allow persistent online training directly against the base model weights
* collapse STM/MTM/LTM into one vector DB
* treat VSA as merely another embedding index
* call pruning “consolidation”
* call current ViT code a complete JEPA implementation
* introduce five architecture changes in one experiment
* let arbitrary retrieved/generated content directly produce persistent deltas
* require MCP calls in token-critical inference loops
* assume all memory backends expose identical semantics
* make biological analogies stronger than empirical evidence supports

---

# 32. Core Architectural Invariants

The following should be preserved unless experimental evidence strongly argues otherwise.

### 1. Base model immutability

```text
W0 remains recoverable and unchanged.
```

### 2. Detachable adaptation

```text
all learned persistent offsets can be disabled independently.
```

### 3. Heterogeneous memory

Different memory tiers may use fundamentally different representations.

### 4. Heterogeneous cognition

Different CogSynDelta regions may use fundamentally different neural architectures.

### 5. Explicit provenance

Persistent memories and learned deltas should be attributable to source observations and consolidation operations.

### 6. Graceful degradation

If Memory-Gate, persona adaptation, or learned deltas fail:

```text
fall back toward base CogSynDelta / base model behavior.
```

### 7. Controlled experimentation

Every major architecture change must be separately benchmarkable.

### 8. Loose coupling by default

Only place mechanisms in the model hot path when latency or differentiability requires it.

---

# 33. Central Research Question

The most distinctive research direction is not simply:

> Can we improve an LLM with external memory?

It is:

> Can heterogeneous explicit memory, structured long-term memory, cognitive-region routing, and reversible learned parameter deltas produce persistent learning and persona adaptation without destructively modifying the underlying foundation model?

A second major question is:

> Can different cognitive tasks benefit from genuinely different computational architectures within one shared latent system instead of using homogeneous Transformer blocks everywhere?

---

# 34. Working Research Hypothesis

A future system may take the form:

```text
CogSynDelta
    +
Memory-Gate-RS
    +
heterogeneous STM/MTM/VSA-LTM
    +
persona namespaces
    +
specialist memory models
    +
reversible low-rank adaptation
    +
optional neural-memory plane
```

with:

```text
CogSynDelta
    = cognition engine

Memory-Gate
    = persistent learning/state substrate

MCP
    = external interoperability layer
```

The most novel bridge between the projects is potentially the **Adaptation Plane**:

```text
persistent memories
      ↓
consolidation
      ↓
validated learned delta
      ↓
detachable modification of effective model behavior
```

such that:

```text
W_eff = W0 + ΔW
```

while:

```text
W0
```

remains immutable and always available as a clean fallback.

---

# 35. Immediate Codex Task Orientation

This handoff is **unvetted ChatGPT context**. Reconcile it with live code before any architecture change.

**Python first.** Implementation and experiments land in CogSynDelta (`src/cogsyndelta/`, PyTorch) and Python `memory-gate`. Rust (`memory-gate-rs`) is a later rewrite for speed/safety/efficiency after the Python path is proven.

Before making architectural changes:

1. Inspect the **Python** trees first (`CogSynDelta`, `memory-gate`) and only then `memory-gate-rs` as a reference. Reconcile every claim in this file against source, `STATUS.md`, and tests.
2. Treat repository status/docs and tests as stronger evidence than this document or README comments.
3. Identify where current implementation already satisfies this design.
4. Identify nomenclature that overstates implementation.
5. Produce an architecture map of:

   * implemented (cite file)
   * partially implemented
   * planned
   * newly proposed (this handoff only — not yet accepted)
6. Identify interfaces that can evolve without unnecessary rewrites.
7. Preserve backwards compatibility where practical.
8. Do not begin a giant refactor merely because the conceptual architecture is larger than the current implementation. Do not start a Rust rewrite this turn.
9. Prefer small controlled **Python** experiments and explicit ADRs.
10. Keep all claims empirical: separate hypotheses from measured results.

The first likely implementation work should focus on clean Python interfaces and experimental scaffolding (parameter-matched `CognitiveRegion` variants), not on importing every proposed research architecture and not on `memory-gate-rs`.
