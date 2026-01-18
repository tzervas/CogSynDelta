# Configuration Reference

Complete reference for `config.yaml`.

## Structure

```yaml
exploratory:          # Exploratory phase settings
culling:             # Culling phase settings
meta_optimization:   # Meta-optimization settings
training:            # Training parameters
vae_loss:            # VAE loss weights
vision_language:     # VL-JEPA settings
agent_framework:     # Self-improving agents
memory_persistence:  # Memory and compression
safeguards:          # Safety constraints
compute_backends:    # Compute configuration
api_server:          # OpenAPI server
model_sectioning:    # Dynamic loading
extensibility:       # Plugin system
```

## Exploratory Phase

```yaml
exploratory:
  k: 10                    # Number of exploratory samples
  sigma_scale: 1.0         # σ scale for VAE sampling
  latent_dim: 20           # Latent space dimension
  hidden_dim: 400          # Hidden layer dimension
```

## Culling Phase

```yaml
culling:
  threshold: 0.5                     # Culling threshold
  bidirectional_beam_width: 5        # Beam width for A*
  bayesian_prior_weight: 0.1         # Prior weight
```

## Memory Persistence

```yaml
memory_persistence:
  enabled: true
  
  # Dense differential embeddings
  dense_encoding:
    enabled: true
    dense_dim: 64                    # Compressed dimension
    num_references: 100              # Reference embeddings
    fidelity_threshold: 0.95         # Min reconstruction quality
  
  # Hierarchical storage
  working_memory:
    capacity: 100                    # Working memory size
  
  short_term_memory:
    capacity: 1000                   # Short-term memory size
    compression_ratio: 0.5
  
  long_term_memory:
    storage_path: "./memory_storage" # Disk storage path
    enable_archiving: true
    max_compression: true
  
  # Checkpointing
  checkpointing:
    enabled: true
    auto_checkpoint_interval: 3600   # Seconds
    checkpoint_dir: "./checkpoints"
    keep_last_n: 5
```

## Safeguards

```yaml
safeguards:
  enabled: true
  
  # Loop detection
  loop_detection:
    enabled: true
    max_iterations: 1000             # Max iterations
    max_repetitions: 5               # Max state repeats
  
  # Timeouts
  timeouts:
    max_execution_time: 300          # Seconds
    max_generation_time: 60
    max_inference_time: 10
  
  # Resource limits
  resource_limits:
    max_memory_usage: 1073741824     # 1GB in bytes
    max_output_size: 10485760        # 10MB in bytes
    max_queue_size: 1000
  
  # Ethical constraints
  ethical:
    enabled: true
    forbidden_patterns:
      - "infinite_loop"
      - "memory_bomb"
      - "fork_bomb"
    content_filtering: true
    validate_outputs: true
```

## Model Sectioning

```yaml
model_sectioning:
  enabled: true
  max_loaded_sections: 5             # Max sections in memory
  dynamic_loading: true
  
  # Automatic scaling
  memory_scaling:
    enabled: true
    base_memory: 1000
  
  timescale_scaling:
    enabled: true
    base_timescale: 10.0
```

## Compute Backends

```yaml
compute_backends:
  # Classical compute
  classical:
    enabled: true
    prefer_gpu: true
    fallback_cpu: true
  
  # Quantum compute
  quantum:
    enabled: false                   # Enable quantum features
    backend_type: "simulator"        # simulator|ibm|google|ionq
    num_qubits: 20
    shots: 1024
    
    provider:
      name: "ibm"
      api_token: null                # Set via env var
      backend_name: "ibmq_qasm_simulator"
```

## API Server

```yaml
api_server:
  host: "0.0.0.0"
  port: 8000
  
  # Supported modalities
  supported_modalities:
    - video
    - audio
    - text
    - code
  
  # Video sources
  video_sources:
    webcam:
      enabled: true
      default_device: 0
    screen_capture:
      enabled: true
    streaming:
      enabled: true
  
  # WebSocket
  websocket:
    enabled: true
    max_connections: 100
    ping_interval: 30
```

## Environment Variables

```bash
# Quantum computing
export QUANTUM_API_TOKEN="your_token_here"

# API configuration
export API_HOST="0.0.0.0"
export API_PORT="8000"

# Storage paths
export MEMORY_STORAGE_PATH="./memory_storage"
export CHECKPOINT_DIR="./checkpoints"
```

## Validation

Validate configuration:

```python
from pcn_vae_gan import load_config

try:
    config = load_config('config.yaml')
    print("✓ Configuration valid")
except Exception as e:
    print(f"✗ Configuration error: {e}")
```

## Examples

### Development Configuration

```yaml
# Minimal resources for development
memory_persistence:
  working_memory:
    capacity: 10
  short_term_memory:
    capacity: 50

safeguards:
  loop_detection:
    max_iterations: 100

model_sectioning:
  max_loaded_sections: 2
```

### Production Configuration

```yaml
# Optimized for production
memory_persistence:
  working_memory:
    capacity: 1000
  short_term_memory:
    capacity: 10000
  
safeguards:
  loop_detection:
    max_iterations: 10000
  timeouts:
    max_execution_time: 3600

model_sectioning:
  max_loaded_sections: 10
```

### High-Security Configuration

```yaml
safeguards:
  enabled: true
  ethical:
    enabled: true
    forbidden_patterns:
      - "infinite_loop"
      - "memory_bomb"
      - "fork_bomb"
      - "resource_exhaustion"
      - "privilege_escalation"
    content_filtering: true
    validate_outputs: true
  
  resource_limits:
    max_memory_usage: 536870912      # 512MB
    max_output_size: 1048576          # 1MB
```

## See Also

- [Architecture Guide](ARCHITECTURE.md)
- [API Reference](API_REFERENCE.md)
- [Contributing](CONTRIBUTING.md)
