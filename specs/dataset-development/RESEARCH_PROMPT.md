# CogSynDelta Dataset Development Research Prompt

> Comprehensive prompt for external research platforms (Claude, GPT-4, Gemini, etc.) to guide dataset development, acquisition, cleaning, and specialization for the CogSynDelta hybrid AI system.

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

#### 4. Hippocampus (Memory System)
**Training Objectives:**
- Dense embedding compression (10-100x)
- Semantic similarity preservation
- Temporal context encoding

**Dataset Requirements:**
- Embedding datasets with similarity labels
- Sequential data for temporal modeling
- Reference ↔ delta encoding examples
- Knowledge graph triples

**Research Questions:**
- How to train for >0.95 fidelity at high compression?
- Evaluation metrics for memory quality?
- How to handle catastrophic forgetting?
- Dataset for differential encoding?

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

| Metric | Target |
|--------|--------|
| Dataset coverage per submodel | ≥ 10M examples |
| Data quality score | ≥ 0.9 |
| Deduplication rate | ≤ 5% duplicates |
| License compliance | 100% |
| Processing throughput | ≥ 1TB/hour |
| Compression fidelity | ≥ 0.95 @ 10x |

---

## Research Prompts for External Models

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

### Prompt 5: Embedding Quality Evaluation
```
For the Hippocampus memory system targeting >0.95 fidelity at 10x compression:
1. What evaluation metrics best capture semantic preservation?
2. How to construct evaluation datasets?
3. What baseline approaches achieve this target?
4. How to detect and measure catastrophic forgetting?
5. Recommended training strategies for compression models?
```

---

## Contact & Governance

- **Project**: CogSynDelta by Average Joe's Labs (AJL)
- **Code Owner**: @tzervas
- **License**: Proprietary
- **Constitution**: Adheres to CogSynDelta Constitution v1.0.0

---

*Generated: January 18, 2026*
*Version: 1.0.0*
