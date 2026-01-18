# Code Documentation

Auto-generated from source code docstrings.

## Core Modules

### pcn_vae_gan.py

Main PCN-VAE-GAN hybrid implementation.

**Classes:**
- `VAEEncoder`: Encoder with μ and log(σ²) outputs
- `VAEDecoder`: Decoder for reconstruction
- `PCNVAEGANHybrid`: Complete three-phase system

**Functions:**
- `create_model(config_path)`: Factory function
- `load_config(config_path)`: Load YAML configuration

### vl_jepa_extension.py

VL-JEPA with mHC and memory bank.

**Classes:**
- `VisionEncoder`: Vision transformer encoder
- `TemporalMemoryBank`: Persistent memory with retrieval
- `HierarchicalPredictiveCoding`: Multi-level prediction with mHC
- `ModeratedHyperConnection`: mHC pathway
- `JointEmbeddingSpace`: Vision-language alignment
- `FrameBufferAdapter`: Video frame processing

### memory_persistence.py

Memory persistence and compression.

**Classes:**
- `PersistentMemoryBank`: Hierarchical memory storage
- `MemoryCompressor`: Compression via autoencoder
- `InfiniteLoopSafeguard`: Safety checks

### dense_embeddings.py

Dense differential embeddings.

**Classes:**
- `DenseEmbeddingEncoder`: Dense compression with residuals
- `DifferentialEncoder`: Differential encoding
- `AdaptiveQuantizer`: Importance-based quantization
- `DenseDifferentialMemoryStore`: Complete compression system

### model_sectioning.py

Dynamic model sectioning.

**Classes:**
- `ModelSection`: Specialized brain region
- `mHCInterconnect`: Communication pathways
- `DynamicModelLoader`: Dynamic loading/unloading
- `SectionedBrainModel`: Complete sectioned model

### interconnect_manager.py

Intelligent interconnect management.

**Classes:**
- `ContextualAttentionRouter`: Attention-based routing
- `PathwayOptimizer`: Learns optimal strengths
- `ContextPropagationEngine`: Multi-hop propagation
- `CongestionController`: Bandwidth management
- `IntelligentInterconnectManager`: Complete system

### self_improving_agents.py

Self-improving agent framework.

**Classes:**
- `LanguageFrameworkEncoder`: Multi-language encoding
- `SelfImprovementModule`: Iterative refinement
- `SecurityHardeningModule`: Vulnerability detection
- `QualityAssuranceModule`: Quality metrics
- `MultiLanguageExplorer`: Cross-language exploration
- `SelfImprovingAgentFramework`: Complete framework

### quantum_compute.py

Quantum computing backends.

**Classes:**
- `ComputeBackend`: Abstract backend interface
- `ClassicalCPUBackend`: CPU backend
- `ClassicalGPUBackend`: GPU backend
- `QuantumGateBackend`: Quantum gate model
- `QuantumHybridBackend`: Hybrid quantum-classical
- `ComputeOrchestrator`: Backend management

### google_adk_adapter.py

Google ADK compliance.

**Classes:**
- `ADKAgent`: Base ADK agent
- `A2AProtocolAdapter`: Agent-to-Agent protocol
- `CogSynDeltaADKAgent`: CogSynDelta agent
- `ADKAdapter`: Main adapter

### integrated_system.py

Complete integrated system.

**Classes:**
- `IntegratedSelfImprovingSystem`: Full system integration

### benchmarks.py

Performance validation.

**Classes:**
- `PerformanceBenchmark`: Speed and memory
- `CompressionBenchmark`: Compression validation
- `AccuracyBenchmark`: Accuracy measurements
- `BenchmarkSuite`: Complete benchmark suite

## API Documentation

See [API_REFERENCE.md](API_REFERENCE.md) for REST/WebSocket API documentation.

## Type Definitions

Common types used throughout:

```python
# Embeddings
Embedding = torch.Tensor  # [batch, embed_dim]

# Section IDs
SectionID = str

# Pathway keys
PathwayKey = Tuple[str, str]  # (source, target)

# Communication context
Context = Dict[str, Any]
```

## Constants

```python
# Default dimensions
DEFAULT_EMBED_DIM = 512
DEFAULT_LATENT_DIM = 20
DEFAULT_HIDDEN_DIM = 400

# Memory capacities
DEFAULT_WORKING_CAPACITY = 100
DEFAULT_SHORT_TERM_CAPACITY = 1000

# Safeguard limits
MAX_ITERATIONS = 1000
MAX_REPETITIONS = 5
TIMEOUT_SECONDS = 300
```

## See Also

- [Architecture Guide](ARCHITECTURE.md)
- [API Reference](API_REFERENCE.md)
- [Agent Development](AGENT_DEVELOPMENT.md)
