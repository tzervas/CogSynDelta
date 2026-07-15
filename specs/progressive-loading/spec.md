# Progressive Dynamic Selective Loading Specification

**Version**: 1.0.0
**Status**: Implemented
**Date**: 2026-01-20
**Owner**: CogSynDelta Core Team

## Overview

Progressive Dynamic Selective Loading enables CogSynDelta to scale from 10B to 100B+ parameters on consumer GPUs (16-32GB VRAM) by loading only the submodels needed for each task, guided by intelligent interconnect routing predictions.

### Goals

1. **Scale to 50B-100B parameters** on RTX 5080 (16GB VRAM) and RTX 5090 (32GB VRAM)
2. **Maintain interactive latency** (≤100ms loading overhead) via prefetching
3. **Leverage interconnect routing** for semantic-aware loading decisions
4. **Degrade gracefully** across hardware configurations

### Non-Goals

- Distributed training across multiple machines (out of scope)
- Automated model compression (handled by separate compression pipeline)
- Fine-tuning large models (inference-focused)

## Architecture

### Component Diagram

```
┌───────────────────────────────────────────────────────────────┐
│                    CogSynDelta Model                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Submodel   │  │  Submodel   │  │  Submodel   │  ...×50  │
│  │  vision_0   │  │  language_0 │  │  reasoning_0│          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                   │
│         └────────────────┴────────────────┘                   │
│                          │                                     │
│         ┌────────────────▼────────────────┐                   │
│         │ Intelligent Interconnect Mgr    │                   │
│         │ (Hopfield routing)              │                   │
│         └────────────┬────────────────────┘                   │
└──────────────────────┼────────────────────────────────────────┘
                       │ routing_decisions
                       ▼
         ┌─────────────────────────────────┐
         │  Progressive Loader Manager     │
         │  - Activation tracking          │
         │  - Prefetch queue               │
         │  - Memory budgeting             │
         └──────────┬──────────────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│  GPU Memory      │  │  CPU Staging     │
│  10GB budget     │  │  16-128GB buffer │
│  8-10 submodels  │  │  Recent models   │
└──────────────────┘  └──────────────────┘
         ▲                     ▲
         └─────────────────────┘
              Async transfers
```

### State Machine

Each submodel exists in one of four states:

```
    register_submodel
           │
           ▼
      ┌────────┐
      │  DISK  │ ◄─────────────┐
      └───┬────┘               │
          │ activate()         │
          ▼                    │ deactivate()
      ┌────────┐               │
      │LOADING │               │
      └───┬────┘               │
          │ (async)            │
          ▼                    │
      ┌────────┐               │
      │ LOADED │ ──────────────┘
      └────────┘
          │
          │ (lru_cache, lazy unload)
          ▼
      ┌────────┐
      │ STAGED │ ────┐
      └───┬────┘     │ reactivate
          │          │
          └──────────┘
```

### Memory Budget

Configured per scale in `config/scaling_config.yaml`:

| Scale    | Total Params | Submodels | Active | GPU Budget | CPU Staging | Disk Cache |
|----------|--------------|-----------|--------|------------|-------------|------------|
| 10B      | 10B          | 8-10      | 8      | 10GB       | 16GB        | 100GB      |
| 25B      | 25B          | 25        | 10     | 10GB       | 32GB        | 200GB      |
| 50B      | 50B          | 50        | 10     | 10GB       | 64GB        | 500GB      |
| 100B     | 100B         | 100       | 10     | 10GB       | 128GB       | 1TB        |

Memory breakdown (RTX 5080, 16GB VRAM, 50B model):

```
16,384 MB total VRAM
 -2,000 MB PyTorch/CUDA overhead
 -2,000 MB interconnect core (routing, embeddings)
 -2,384 MB activation buffers (with checkpointing)
────────────────────────────────────
10,000 MB available for submodels

Each submodel (1B params):
  - FP16:  ~2,000 MB
  - BitNet:  ~200 MB (10× compression)

Max active (FP16):     5 submodels
Max active (BitNet):  50 submodels (routing limits to 10)
```

## API Specification

### ProgressiveLoadingConfig

Configuration dataclass for progressive loading behavior.

```python
@dataclass
class ProgressiveLoadingConfig:
    """Configuration for progressive submodel loading."""

    # Core thresholds
    max_active_submodels: int = 8  # Maximum submodels in GPU memory
    eager_load_threshold: float = 0.7  # Activate if routing prob ≥ this
    lazy_unload_threshold: float = 0.2  # Deactivate if routing prob ≤ this

    # Prefetching
    prefetch_depth: int = 2  # How many future steps to prefetch
    background_prefetch: bool = True  # Async prefetching

    # Performance
    async_loading: bool = True  # Asynchronous GPU transfers
    max_concurrent_loads: int = 2  # Parallel loading streams

    # Memory management
    cpu_staging_size_mb: int = 16000  # CPU staging area size
    disk_cache_size_mb: int = 100000  # Disk cache size

    # Advanced
    activation_checkpointing: bool = True  # Reduce activation memory
    checkpoint_segments: int = 4  # Checkpointing granularity
```

### ProgressiveLoaderManager

Main interface for progressive loading.

#### Initialization

```python
from cogsyndelta.core.progressive_loader import (
    ProgressiveLoaderManager,
    ProgressiveLoadingConfig,
)

config = ProgressiveLoadingConfig.from_yaml("config/scaling_config.yaml", "large_50b")
loader = ProgressiveLoaderManager(config, device="cuda")
```

#### Submodel Registration

```python
# Register all submodels (called during model initialization)
for name, module in model.submodels.items():
    loader.register_submodel(
        name=name,
        module=module,
        size_mb=estimate_size_mb(module),
        submodel_type="vision",  # Optional: vision, language, reasoning, etc.
    )

# Mark core submodels as pinned (always loaded)
loader.pin_submodel("core_attention")
loader.pin_submodel("core_embedding")
```

#### Routing Integration

```python
# During inference: update from interconnect routing decisions
routing_decisions: Dict[Tuple[str, str], float] = interconnect.route(
    query_section="language_processing",
    current_context=context,
)

# routing_decisions example:
# {
#   ("language_0", "reasoning_2"): 0.85,  # High probability
#   ("language_0", "memory_1"): 0.45,     # Medium probability
#   ("language_0", "vision_3"): 0.12,     # Low probability
# }

section_to_submodel = {
    "language_processing": "language_0",
    "reasoning": "reasoning_2",
    "memory": "memory_1",
    # ...
}

# Update loading based on routing
loader.update_from_routing(routing_decisions, section_to_submodel)
```

#### Accessing Submodels

```python
# Get loaded submodel (returns None if not loaded)
vision_module = loader.get_loaded("vision_0")

if vision_module is not None:
    output = vision_module(input_tensor)
else:
    # Fallback: block until loaded
    vision_module = loader.activate("vision_0", blocking=True)
    output = vision_module(input_tensor)
```

#### Manual Control

```python
# Manually activate/deactivate (for testing, debugging)
loader.activate("reasoning_5", blocking=False)  # Async activation
loader.deactivate("vision_2")  # Unload from GPU

# Check state
assert loader.is_loaded("reasoning_5")  # May be False if still loading
assert not loader.is_loaded("vision_2")

# Force synchronization
loader.wait_for_all_loads()  # Blocks until async loads complete
```

#### Statistics

```python
# Get memory usage statistics
stats = loader.get_stats()
print(f"Active submodels: {stats['num_loaded']}/{stats['max_active']}")
print(f"GPU memory used: {stats['gpu_memory_mb']} MB")
print(f"Prefetch queue: {stats['prefetch_queue_size']}")
print(f"Prefetch hit rate: {stats['prefetch_hit_rate']:.2%}")
```

### Configuration File Format

From `config/scaling_config.yaml`:

```yaml
large_50b:
  model_size: "50B"
  total_parameters: 50_000_000_000

  # Memory budgets (MB)
  gpu_budget: 10000
  gpu_reserved_core: 2000
  cpu_staging: 64000
  disk_cache: 500000

  # Progressive loading
  max_active_submodels: 10
  eager_load_threshold: 0.5
  lazy_unload_threshold: 0.4
  prefetch_depth: 4

  # Submodel architecture
  submodel_size: "1B"
  total_submodels: 50
  submodel_types:
    vision: 5
    language: 10
    reasoning: 10
    memory: 5
    control: 5
    multimodal: 10
    specialist: 5

  core_submodels:
    - "core_attention"
    - "core_ffn"
    - "core_embedding"
    - "core_router"

  # Performance optimizations
  async_loading: true
  background_prefetch: true
  activation_checkpointing: true
  checkpoint_segments: 8
  max_concurrent_loads: 4
```

## Usage Examples

### Example 1: Basic Setup (25B model)

```python
from cogsyndelta.core.progressive_loader import ProgressiveLoaderManager, ProgressiveLoadingConfig

# Load configuration
config = ProgressiveLoadingConfig.from_yaml("config/scaling_config.yaml", "medium_25b")
loader = ProgressiveLoaderManager(config, device="cuda")

# Register submodels
model = CogSynDeltaModel(num_submodels=25, submodel_size="1B")
for i, submodel in enumerate(model.submodels):
    loader.register_submodel(f"submodel_{i}", submodel)

# Pin core submodels
for core in ["core_attention", "core_embedding"]:
    loader.pin_submodel(core)

# During inference
routing = interconnect.route("language_task", context)
loader.update_from_routing(routing, section_to_submodel)

# Access active submodels
for name in model.active_submodels:
    module = loader.get_loaded(name)
    if module:
        output = module(input)
```

### Example 2: Manual Prefetching

```python
# Manually prefetch likely-next submodels
loader.prefetch(["reasoning_0", "reasoning_1", "memory_2"])

# Continue with current computation
current_output = current_module(input)

# By the time we need them, they should be loaded
next_module = loader.get_loaded("reasoning_0")  # Should be ready
```

### Example 3: Memory-Constrained Scenario

```python
# Very aggressive loading for low-memory GPU
config = ProgressiveLoadingConfig(
    max_active_submodels=5,  # Only 5 active
    eager_load_threshold=0.8,  # Very selective
    lazy_unload_threshold=0.1,  # Aggressive unloading
    prefetch_depth=1,  # Minimal prefetch
)

loader = ProgressiveLoaderManager(config, device="cuda")
# ... rest of setup
```

## Performance Characteristics

### Loading Latency

Target: **≤100ms per submodel** (GPU ← CPU, async)

Breakdown:
- CPU → GPU transfer: 50-80ms (1B params × 2GB @ PCIe Gen4 ~25 GB/s)
- State setup: 10-20ms (initialize buffers, register hooks)
- Warm cache hit: <5ms (already in staging area)

Mitigation strategies:
- **Prefetching**: Hides latency by loading during computation
- **Async loading**: Non-blocking transfers via CUDA streams
- **CPU staging**: Recent models stay in RAM for fast reactivation

### Prefetch Accuracy

Target: **≥80% prefetch hit rate**

Measured as:
```
hit_rate = (prefetch_hits) / (total_activations)
```

Factors affecting accuracy:
- **Routing stability**: Consistent routing → better prediction
- **Prefetch depth**: Deeper prefetch → better coverage, more memory
- **Task complexity**: Multi-hop reasoning → harder to predict

### Memory Overhead

Target: **≤5% overhead** for tracking structures

Components:
- Loader state: ~10 MB (activation tracking, queues)
- Prefetch buffers: ~500 MB (staging area metadata)
- Routing cache: ~50 MB (interconnect routing history)

Total: ~560 MB ≈ 3.5% of 16GB VRAM ✓

## Testing Strategy

### Unit Tests

**File**: `tests/unit/test_progressive_loader.py`

```python
def test_activation_tracking():
    """Test submodel state transitions (disk → loading → loaded)."""
    loader = ProgressiveLoaderManager(config, device="cpu")
    loader.register_submodel("test_module", nn.Linear(10, 10))

    assert not loader.is_loaded("test_module")
    loader.activate("test_module", blocking=True)
    assert loader.is_loaded("test_module")
    loader.deactivate("test_module")
    assert not loader.is_loaded("test_module")

def test_memory_budget_enforcement():
    """Test max_active_submodels limit."""
    config = ProgressiveLoadingConfig(max_active_submodels=3)
    loader = ProgressiveLoaderManager(config, device="cpu")

    for i in range(5):
        loader.register_submodel(f"module_{i}", nn.Linear(10, 10))
        loader.activate(f"module_{i}", blocking=True)

    # Only 3 should be active (LRU eviction of earlier modules)
    assert loader.get_stats()["num_loaded"] <= 3
```

### Integration Tests

**File**: `tests/integration/test_progressive_with_interconnect.py`

```python
def test_routing_driven_loading():
    """Test interconnect routing → loader activation pipeline."""
    interconnect = IntelligentInterconnectManager(config)
    loader = ProgressiveLoaderManager(loading_config, device="cuda")

    # Simulate routing decision
    routing = {
        ("language_0", "reasoning_2"): 0.85,  # Above eager threshold
        ("language_0", "vision_1"): 0.15,     # Below lazy threshold
    }

    loader.update_from_routing(routing, section_to_submodel)

    # reasoning_2 should be loaded/loading
    assert loader.is_loaded("reasoning_2") or loader.is_loading("reasoning_2")
    # vision_1 should be unloaded (below threshold)
    assert not loader.is_loaded("vision_1")
```

### Benchmark Tests

**File**: `tests/benchmark/test_loading_latency.py`

```python
@pytest.mark.benchmark
def test_loading_latency_1b_model():
    """Measure CPU → GPU transfer time for 1B parameter submodel."""
    model = create_1b_submodel()
    loader = ProgressiveLoaderManager(config, device="cuda")

    loader.register_submodel("benchmark", model)

    start = time.perf_counter()
    loader.activate("benchmark", blocking=True)
    latency_ms = (time.perf_counter() - start) * 1000

    assert latency_ms < 100, f"Loading took {latency_ms}ms, target <100ms"
```

## Success Criteria

Per technical specification:

| Criterion | Target | Status | Validation Method |
|-----------|--------|--------|-------------------|
| 50B model on RTX 5080 | ≤10 active submodels | ✅ Implemented | `config/scaling_config.yaml:large_50b` |
| 100B model on RTX 5090 | ≤10 active submodels | ✅ Implemented | `config/scaling_config.yaml:xlarge_100b` |
| Loading latency | ≤100ms per submodel | ⏳ Pending | Benchmark on hardware |
| Prefetch hit rate | ≥80% | ⏳ Pending | Integration test with routing |
| Memory overhead | ≤5% | ✅ Estimated 3.5% | Loader state tracking |

(⏳ = pending empirical validation on target hardware)

## Implementation Checklist

- [x] `ProgressiveLoaderManager` class
- [x] `ProgressiveLoadingConfig` dataclass
- [x] Submodel registration API
- [x] Activation/deactivation logic
- [x] Routing integration (`update_from_routing`)
- [x] Prefetch queue implementation
- [x] Async loading via threading
- [x] Memory budget tracking
- [x] Statistics API (`get_stats`)
- [x] Configuration file (`config/scaling_config.yaml`)
- [x] Documentation (this spec + ADR-0015)
- [ ] Unit tests
- [ ] Integration tests with interconnect
- [ ] Benchmark tests on RTX 5080/5090
- [ ] Prefetch accuracy measurement

## Migration Guide

### From Static Loading (v0.2.0)

**Before** (all submodels loaded):

```python
model = CogSynDeltaModel()
model.to("cuda")  # Loads everything to GPU

output = model(input)  # All submodels available
```

**After** (progressive loading):

```python
from cogsyndelta.core.progressive_loader import ProgressiveLoaderManager, ProgressiveLoadingConfig

config = ProgressiveLoadingConfig.from_yaml("config/scaling_config.yaml", "large_50b")
loader = ProgressiveLoaderManager(config, device="cuda")

model = CogSynDeltaModel()
for name, module in model.submodels.items():
    loader.register_submodel(name, module)

# During inference
routing = model.interconnect.route(query_section, context)
loader.update_from_routing(routing, model.section_to_submodel)

# Model automatically uses only loaded submodels
output = model(input)
```

## Future Extensions

### Planned

1. **Multi-GPU support**: Distribute submodels across GPUs (e.g., 2× RTX 5080)
2. **Persistent disk cache**: Save loaded checkpoints to NVMe for faster restarts
3. **Adaptive thresholds**: Learn optimal `eager_load_threshold` from usage patterns
4. **Compression integration**: Progressive loading of compressed (BitNet) submodels

### Under Consideration

1. **Remote submodels**: Load from network storage for distributed inference
2. **Quantization-aware loading**: Load FP16 or INT8 based on memory pressure
3. **Energy-based routing**: Use V-JEPA energy predictions for loading decisions

## References

- **ADR-0015**: Progressive Dynamic Selective Loading (architecture decision)
- **ADR-0012**: VSA and Hopfield Memory Integration (interconnect routing)
- **Technical docs**: `docs/PROGRESSIVE_LOADING_AND_BALANCED_TERNARY.md`
- **Configuration**: `config/scaling_config.yaml`
- **Implementation**: `src/cogsyndelta/core/progressive_loader.py`

## Glossary

- **Submodel**: Independent neural network component (~1B parameters)
- **Interconnect routing**: Hopfield-based attention mechanism for cross-submodel communication
- **Eager loading**: Activate submodel when routing probability exceeds threshold
- **Lazy unloading**: Deactivate submodel when routing probability drops below threshold
- **Prefetching**: Preemptively load likely-next submodels to hide latency
- **CPU staging**: Intermediate memory tier between GPU and disk
- **Activation checkpointing**: Trade compute for memory by recomputing activations
