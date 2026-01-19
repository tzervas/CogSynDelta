# CogSynDelta Dataset Development Research Prompt

> Comprehensive prompt for external research platforms (Claude, GPT-4, Gemini, etc.) to guide dataset development, acquisition, cleaning, and specialization for the CogSynDelta hybrid AI system.

---

## 🔴 CRITICAL: Compression Fidelity Crisis

**PRIORITY 1**: Current compression fidelity is **0.06** (catastrophic failure) vs target **≥0.95**.

### Problem Statement
The hippocampus memory system's dense differential encoding achieves only 0.06 cosine similarity between original and reconstructed embeddings. This is essentially random noise (orthogonal vectors have ~0 similarity).

### Root Causes Identified
1. **Uncalibrated quantization** - Fixed ranges don't match embedding distribution
2. **Insufficient bit-width** for differential signals
3. **Cumulative error propagation** without correction
4. **Distribution mismatch** - Assumes Gaussian, actual may be heavy-tailed

### Research Requirements for Fidelity Recovery

We need datasets and techniques for:

1. **Calibration Data**: Representative corpus of ≥10,000 embeddings to compute per-dimension statistics
2. **Matryoshka Training Data**: AllNLI, MS MARCO for multi-scale contrastive learning
3. **Codebook Learning Data**: LAION-5B pre-computed embeddings for RVQ training
4. **Evaluation Benchmarks**: STS Benchmark, MTEB, ANN-Benchmarks with ground truth

### Achievable Fidelity Targets

| Compression | Fidelity | Technique | Status |
|-------------|----------|-----------|--------|
| 4x | 0.97-0.99 | Scalar quant / Matryoshka | ✅ Achievable |
| 8x | 0.95-0.98 | OPQ, Matryoshka MRL | ✅ Achievable |
| 16x | 0.90-0.95 | QINCo2, RVQ, BitNet | ✅ Achievable |
| 24x | 0.85-0.92 | PQ with rescoring | ⚠️ Challenging |
| 100x | <0.70 | Not demonstrated | ❌ Not achievable |

---

## Context: CogSynDelta Architecture

CogSynDelta is a PCN-VAE-GAN hybrid self-improving AI system with the following core components:

### Brain-Inspired Architecture
```
┌─────────────────────────────────────────────────────────────┐
│              Integrated Self-Improving AI System             │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Intelligent Interconnect Manager (mHC)       │  │
│  │    • Attention-based routing between brain regions   │  │
│  │    • Context propagation and bandwidth allocation    │  │
│  └──────────────────────────────────────────────────────┘  │
│                          ↕                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Visual      │  │  Language    │  │  Prefrontal  │     │
│  │  Cortex      │  │  Cortex      │  │  Cortex      │     │
│  │  (VL-JEPA)   │  │  (Multi-Lang)│  │  (Planning)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Auditory    │  │  Motor       │  │  Cerebellum  │     │
│  │  Cortex      │  │  Cortex      │  │  (Fine-tune) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                          ↕                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Hippocampus (Memory Bank)                    │  │
│  │    • Dense differential embeddings (10-100x)         │  │
│  │    • Tiered: active → short → long-term              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Core Neural Components

1. **PCN-VAE-GAN Hybrid**: Three-phase self-improvement
   - Exploratory Phase: High-variance sampling for creative exploration
   - Culling Phase: Bayesian inference for feasibility filtering
   - Meta-Optimization Phase: MAML for rapid task adaptation

2. **VL-JEPA (Vision-Language Joint Embedding)**: Silent semantic prediction
   - No token generation - predicts embeddings directly
   - Temporal memory bank for context grounding
   - 2.85x faster than token-generating models

3. **mHC (Moderated Hyper-Connections)**: Information flow control
   - Learnable gates between brain regions
   - Dynamic routing based on task context

4. **Self-Improving Agents**: Multi-language code generation
   - Supports: Python, Rust, Go, TypeScript, Java, C++, C#
   - Agent types: SWE, AIE, SWD, AID, Security, QA

---

## Research Objectives

### Phase 1: Core Dataset Requirements

For each brain region/submodel, identify and design datasets for:

#### 1. Visual Cortex (VL-JEPA)
**Training Objectives:**
- Image understanding and semantic embedding
- Temporal video understanding (frame sequences)
- Cross-modal alignment (image ↔ text)

**Dataset Requirements:**
- Large-scale image datasets with dense captions
- Video datasets with temporal annotations
- Image-text paired datasets for alignment
- Multi-resolution imagery (thumbnail to high-res)

**Research Questions:**
- What datasets best support silent semantic prediction (no token generation)?
- How to structure temporal data for memory bank training?
- Optimal ratio of still images vs video sequences?
- How to handle domain shift between photographic and synthetic images?

#### 2. Language Cortex (Multi-Language Code)
**Training Objectives:**
- Multi-language code understanding and generation
- Cross-language translation and refactoring
- Code quality assessment and improvement

**Dataset Requirements:**
- High-quality code repositories per language (Python, Rust, Go, TypeScript, Java, C++, C#)
- Code review comments and improvements
- Bug fix commit pairs (before/after)
- Documentation ↔ code pairs
- Test case ↔ implementation pairs

**Research Questions:**
- How to balance language representation (Python over-represented)?
- Dataset cleaning strategies for code quality?
- How to capture idiomaticity per language?
- Synthetic data generation for rare patterns?

#### 3. Prefrontal Cortex (Planning & Reasoning)
**Training Objectives:**
- Multi-step planning and reasoning
- Goal decomposition and task sequencing
- Constraint satisfaction and optimization

**Dataset Requirements:**
- Chain-of-thought reasoning traces
- Planning problems with solution paths
- Project decomposition examples
- Resource allocation and scheduling data

**Research Questions:**
- How to capture expert planning strategies?
- Dataset formats for hierarchical goals?
- How to evaluate reasoning quality?
- Synthetic planning problem generation?

#### 4. Hippocampus (Memory System) 🔴 CRITICAL
**Training Objectives:**
- Dense embedding compression (10-20x with ≥0.95 fidelity)
- Semantic similarity preservation
- Temporal context encoding
- **Adjustable compression ratio** with fidelity guarantees

**Dataset Requirements:**
- **AllNLI** (1M+ pairs) - Primary for contrastive training
- **LAION-5B Embeddings** (5.85B) - Codebook learning at scale
- **GIST-1M** (960d vectors) - High-dimensional PQ training
- **STS Benchmark** (8.6K pairs) - Fidelity evaluation with continuous scores
- **MTEB Suite** (58 datasets) - Multi-task evaluation
- Sequential data for temporal modeling
- Reference ↔ delta encoding examples
- Knowledge graph triples

**Research Questions:**
- **How to achieve ≥0.95 fidelity at 10x compression?** (Matryoshka + RVQ)
- **What calibration data is needed?** (≥10K representative embeddings per domain)
- How to implement adjustable compression with fidelity guarantees?
- Optimal residual quantizer configuration (stages, codebook size)?
- How to integrate VSA/holographic techniques for brain-inspired memory?
- Evaluation metrics for memory quality (Spearman ρ, Recall@K)?
- How to handle catastrophic forgetting?
- Dataset for differential encoding with proper calibration?

**Key Techniques to Research:**
1. **Matryoshka Representation Learning** (Kusupati et al. NeurIPS 2022)
   - Multi-scale loss: `Total_Loss = Σ(weight_i × Loss(embedding[:dim_i]))`
   - Pre-trained: `tomaarsen/mpnet-base-nli-matryoshka`, `nomic-embed-text-v1`

2. **Residual Vector Quantization** (vector-quantize-pytorch)
   - 4-8 residual stages, 256-1024 codebook size
   - Stop when target fidelity achieved

3. **BitNet b1.58** (Ma et al. arXiv:2402.17764)
   - Balanced ternary {-1, 0, +1} achieves 100% fidelity at 16x
   - Requires quantization-aware training

4. **Vector Symbolic Architectures** (torchhd library)
   - Modern Hopfield Networks for exponential capacity
   - VSA binding for temporal context
   - Dimension ≥10,000 for reliable operations

#### 5. Interconnect Manager (mHC)
**Training Objectives:**
- Cross-modal attention routing
- Information flow optimization
- Context-dependent gating

**Dataset Requirements:**
- Multi-modal aligned data (text + image + code)
- Attention trace annotations
- Routing decision examples

**Research Questions:**
- How to capture "expert routing" decisions?
- Dataset for training dynamic gating?
- Evaluation of routing quality?

---

## Phase 2: Data Pipeline Design

### Acquisition Strategy

Research and recommend approaches for:

1. **Public Dataset Integration**
   - Which existing datasets are suitable?
   - License compatibility (MIT, Apache-2.0, BSD, CC-BY)
   - Quality assessment criteria
   - Download and preprocessing pipelines

2. **Synthetic Data Generation**
   - LLM-generated training data (with quality filtering)
   - Procedural code generation
   - Augmentation strategies
   - Adversarial example generation

3. **Data Partnerships**
   - Academic dataset access (arXiv, GitHub Archive, etc.)
   - Industry datasets (with proper licensing)
   - Crowdsourced annotations

### Cleaning & Refinement

Design automated pipelines for:

1. **Quality Filtering**
   - Perplexity-based filtering
   - Deduplication (exact and near-duplicate)
   - Language detection and filtering
   - Code syntax validation
   - Copyright/license detection

2. **Normalization**
   - Code formatting standardization
   - Tokenization consistency
   - Embedding space alignment
   - Resolution/quality normalization for images

3. **Enrichment**
   - Automatic tagging (language, domain, difficulty)
   - Metadata extraction
   - Quality scoring
   - Complexity metrics

### Pre/Post Processing

1. **Tokenization Strategy**
   - Code-aware tokenizers
   - Multi-lingual support
   - Special tokens for structure

2. **Embedding Formats**
   - Dense vector storage
   - Sparse representation for retrieval
   - Quantization for efficiency

3. **Batching & Curriculum**
   - Difficulty-based curriculum
   - Domain mixing strategies
   - Multi-task batching

---

## Phase 3: Specialization Datasets

### Per-Submodel Specialization

For each brain region, design specialized fine-tuning datasets:

| Submodel | Specialization Focus | Dataset Characteristics |
|----------|---------------------|------------------------|
| Visual Cortex | Domain-specific vision | Medical imaging, satellite, industrial |
| Language Cortex | Language mastery | Per-language idioms, best practices |
| Prefrontal Cortex | Domain reasoning | Software architecture, system design |
| Hippocampus | Compression fidelity | High-information-density examples |
| Interconnect | Routing efficiency | Multi-modal task examples |

### Language Specialization Datasets

For each supported programming language:

**Python:**
- Scientific computing (NumPy, Pandas, PyTorch)
- Web development (FastAPI, Django)
- Data engineering (Airflow, Spark)
- Testing (pytest, unittest)

**Rust:**
- Systems programming patterns
- Async/await patterns (tokio)
- Memory safety idioms
- Performance optimization

**Go:**
- Concurrency patterns
- API development
- CLI tooling
- Cloud-native patterns

**TypeScript:**
- React/Vue/Angular patterns
- Node.js server-side
- Type system mastery
- Build tooling

**Java:**
- Enterprise patterns (Spring)
- Concurrency utilities
- JVM optimization
- Testing frameworks

**C++:**
- Modern C++ (C++17/20/23)
- Performance-critical code
- Memory management
- Template metaprogramming

**C#:**
- .NET patterns
- LINQ mastery
- Async patterns
- Unity game dev

---

## Phase 4: Training Phasing

### Curriculum Design

1. **Foundation Phase**
   - Basic understanding per domain
   - Broad coverage, shallow depth
   - High data diversity

2. **Specialization Phase**
   - Domain-specific deep dives
   - High-quality curated datasets
   - Task-specific fine-tuning

3. **Integration Phase**
   - Multi-modal alignment
   - Cross-submodel training
   - End-to-end system optimization

4. **Alignment Phase**
   - Quality and safety alignment
   - Benchmark-driven refinement
   - Production-ready validation

---

## Deliverables Requested

Please provide research and recommendations on:

### Immediate (Week 1-2)
1. **Dataset Catalog**: Comprehensive list of candidate public datasets per submodel
2. **License Audit**: License compatibility matrix
3. **Quality Criteria**: Automated quality assessment rubrics
4. **Pipeline Architecture**: Data processing DAG design

### Short-term (Week 3-4)
5. **Cleaning Pipelines**: Automated filtering/cleaning implementations
6. **Enrichment Tools**: Tagging and metadata extraction tools
7. **Evaluation Metrics**: Per-dataset quality metrics

### Medium-term (Month 2-3)
8. **Specialization Datasets**: Per-language curated datasets
9. **Synthetic Generation**: Data generation pipelines
10. **Curriculum Design**: Training phase specifications

### Long-term (Month 3+)
11. **Continuous Pipeline**: Automated dataset updates
12. **Quality Monitoring**: Drift detection and alerting
13. **Versioning**: Dataset versioning and reproducibility

---

## Technical Constraints

- **Storage**: Design for petabyte-scale storage
- **Processing**: GPU-accelerated preprocessing preferred
- **Formats**: Parquet, Arrow, WebDataset for efficiency
- **Versioning**: DVC or similar for dataset versioning
- **Licensing**: Only MIT, Apache-2.0, BSD, CC-BY compatible

---

## Success Metrics

| Metric | Target | Priority |
|--------|--------|----------|
| **Compression fidelity** | **≥ 0.95 @ 10x** | 🔴 P0 |
| **Adjustable compression** | 4x-20x | 🔴 P0 |
| Dataset coverage per submodel | ≥ 10M examples | 🟡 P1 |
| Data quality score | ≥ 0.9 | 🟡 P1 |
| Deduplication rate | ≤ 5% duplicates | 🟢 P2 |
| License compliance | 100% | 🟢 P2 |
| Processing throughput | ≥ 1TB/hour | 🟢 P2 |

---

## Research Prompts for External Models

### Prompt 0: CRITICAL - Compression Fidelity Recovery
```
We have a CRITICAL compression fidelity crisis in our brain-inspired memory system:

CURRENT STATE:
- Cosine similarity fidelity: 0.06 (catastrophic - essentially random noise)
- Target: ≥0.95 fidelity at 10x compression
- Architecture: Dense differential encoding with adaptive quantization

ROOT CAUSES IDENTIFIED:
1. Uncalibrated quantization buckets (fixed ranges don't match distribution)
2. Insufficient bit-width for differential signals
3. Cumulative error propagation without correction
4. Distribution mismatch (assumes Gaussian, may be heavy-tailed)

RESEARCH QUESTIONS:
1. What is the optimal calibration procedure for embedding quantization?
   - How many samples needed? (we think ≥10K)
   - Per-dimension vs global statistics?
   - Percentile-based ranges vs min/max?

2. For Matryoshka Representation Learning:
   - What training data achieves best multi-scale fidelity?
   - Optimal dimension truncation points for 10x compression?
   - How to combine with existing encoder architecture?

3. For Residual Vector Quantization:
   - Optimal number of stages for ≥0.95 fidelity?
   - Codebook size vs fidelity tradeoff?
   - How to train codebooks on domain-specific data?

4. For adjustable compression with fidelity guarantees:
   - How to implement early-stopping when target fidelity achieved?
   - Fallback strategies when lossy compression insufficient?
   - Real-time fidelity monitoring approaches?

5. For VSA/holographic integration:
   - Modern Hopfield vs traditional for active memory?
   - VSA binding operations for temporal context?
   - torchhd vs custom implementation tradeoffs?

Provide specific implementation recommendations with code examples where applicable.
Reference state-of-the-art techniques: QINCo2, BitNet b1.58, Matryoshka MRL.
```

### Prompt 1: Dataset Discovery
```
Given the CogSynDelta architecture described above, identify the top 20 public datasets 
that would be most valuable for training each brain region submodel. For each dataset provide:
1. Name and source URL
2. Size and format
3. License
4. Relevance score (1-10) with justification
5. Known limitations or biases
6. Recommended preprocessing steps
```

### Prompt 2: Cleaning Pipeline Design
```
Design an automated data cleaning pipeline for code datasets that:
1. Removes duplicates (exact and near-duplicate)
2. Filters by quality (syntax valid, well-documented, idiomatic)
3. Extracts metadata (language, framework, complexity)
4. Normalizes formatting
5. Handles multi-file projects
Provide pseudocode and tool recommendations.
```

### Prompt 3: Synthetic Data Generation
```
For underrepresented code patterns (e.g., Rust async, Go generics, C++ coroutines),
design a synthetic data generation strategy using LLMs that:
1. Generates diverse, high-quality examples
2. Includes quality filtering to remove hallucinations
3. Covers edge cases and error handling
4. Maintains language idiomaticity
5. Can scale to millions of examples
```

### Prompt 4: Curriculum Learning
```
Design a curriculum learning strategy for training the Language Cortex that:
1. Starts with simple, single-file programs
2. Progresses to multi-file projects
3. Introduces cross-language translation
4. Adds code review and improvement tasks
5. Culminates in complex refactoring challenges
Include data selection criteria for each phase.
```

### Prompt 5: Embedding Quality Evaluation (Updated for Fidelity Crisis)
```
For the Hippocampus memory system with CRITICAL fidelity crisis (0.06 current vs 0.95 target):

1. What evaluation metrics best capture semantic preservation?
   - Cosine similarity (current metric showing 0.06)
   - Spearman correlation on STS Benchmark
   - Recall@K on ANN-Benchmarks
   - MTEB multi-task evaluation

2. How to construct calibration datasets?
   - Minimum samples needed (≥10K recommended)
   - Per-dimension vs global statistics
   - Domain-specific calibration sets

3. What techniques achieve ≥0.95 fidelity at 10x compression?
   - Matryoshka Representation Learning (truncation-based)
   - Residual Vector Quantization (4-8 stages)
   - BitNet b1.58 balanced ternary
   - Calibrated scalar quantization as baseline

4. How to implement adjustable compression with guarantees?
   - Early stopping when target fidelity achieved
   - Fallback to lower compression if needed
   - Real-time fidelity monitoring

5. Training strategies for compression models?
   - Contrastive training on AllNLI
   - Codebook learning on LAION embeddings
   - Multi-scale loss for Matryoshka
```

### Prompt 6: VSA and Holographic Memory Integration
```
For integrating Vector Symbolic Architectures into brain-inspired memory:

CONTEXT:
- Active memory tier needs exponential capacity
- Short-term memory needs temporal context binding
- Long-term memory needs compressed similarity search
- Target: brain-inspired Complementary Learning Systems

RESEARCH QUESTIONS:
1. Modern Hopfield Networks vs traditional Hopfield for active memory?
   - Capacity bounds (2^(d/2) claimed for modern)
   - One-step convergence properties
   - Integration with transformer attention

2. VSA binding operations comparison:
   - MAP-B (Hadamard product) - 100% reconstruction
   - FHRR (complex multiplication) - 100% with unit phasors
   - HRR (circular convolution) - ~95% at d=10,000
   - Which is best for temporal context?

3. torchhd library capabilities:
   - GPU acceleration support
   - Pre-built binding/bundling operations
   - Integration patterns with PyTorch

4. Hybrid architecture design:
   - VSA for associative pre-filtering
   - FAISS for precise similarity search
   - How to combine for 10-100x search speedup?

5. Dimension requirements:
   - Minimum d for reliable VSA operations
   - Memory/compute tradeoffs
   - Compression from high-d VSA to storage

Provide implementation recommendations for RTX 5080 (16GB GDDR7, CUDA 12.8).
```

### Prompt 7: Adjustable Compression System Design
```
Design an adjustable compression system for embeddings with fidelity guarantees:

REQUIREMENTS:
- Compression ratios: 4x to 20x (adjustable)
- Fidelity guarantee: ≥0.95 cosine similarity
- If target ratio would violate fidelity, auto-reduce compression
- Real-time fidelity monitoring and alerting

ARCHITECTURE:
1. Calibration layer (per-dimension statistics)
2. Matryoshka layer (dimension truncation)
3. RVQ layer (residual quantization stages)
4. Fidelity monitor (continuous checking)

QUESTIONS:
1. API design for adjustable compression?
2. How to precompute compression curves per embedding type?
3. Fallback strategies when lossy insufficient?
4. Batch vs single-embedding optimization?
5. GPU kernel optimization for RTX 5080?

Provide PyTorch implementation with type hints and Google-style docstrings.
```

---

## Sister Projects (embeddenator-core ecosystem)

The following libraries form the embeddenator-core ecosystem for high-fidelity embedding compression:

| Library | Purpose | License | URL |
|---------|---------|---------|-----|
| **torchhd** | VSA operations (GPU) | MIT | github.com/hyperdimensional-computing/torchhd |
| **vector-quantize-pytorch** | RVQ, FSQ, LFQ | MIT | github.com/lucidrains/vector-quantize-pytorch |
| **sentence-transformers** | Matryoshka models | Apache-2.0 | github.com/UKPLab/sentence-transformers |
| **FAISS** | Similarity search | MIT | github.com/facebookresearch/faiss |

---

## Contact & Governance

- **Project**: CogSynDelta by Average Joe's Labs (AJL)
- **Code Owner**: @tzervas
- **License**: Proprietary
- **Constitution**: Adheres to CogSynDelta Constitution v1.0.0

---

*Generated: January 18, 2026*
*Version: 1.1.0 - Updated with compression fidelity recovery focus*
