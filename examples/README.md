# CogSynDelta Examples

This directory contains example scripts demonstrating various features of CogSynDelta.

## Prerequisites

Install uv and sync dependencies:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd CogSynDelta
uv sync
```

## Examples

### 1. Basic Training (`basic_training.py`)
Demonstrates basic model training on MNIST dataset:
```bash
uv run python examples/basic_training.py
```

**Features:**
- Loading and configuring the integrated system
- Training PCN-VAE-GAN on MNIST
- Monitoring training metrics

### 2. Memory Management (`memory_management.py`)
Demonstrates the tiered memory system:
```bash
uv run python examples/memory_management.py
```

**Features:**
- Active, short-term, and long-term memory tiers
- Memory compression and persistence
- Similarity-based memory retrieval
- Memory tier management

### 3. API Server (`api_server.py`)
Starts the FastAPI server:
```bash
uv run python examples/api_server.py
```

Then test with:
```bash
curl http://localhost:8000/health
```

**Features:**
- RESTful API endpoints
- Multimodal input processing
- Agent task submission
- WebSocket streaming

### 4. Self-Improving Agents (`self_improving_agents.py`)
Demonstrates agent learning and collaboration:
```bash
uv run python examples/self_improving_agents.py
```

**Features:**
- Self-improving language agents
- Meta-learning optimization
- Multi-agent collaboration
- Adaptive behavior

### 5. Quantum Computing (`quantum_computing.py`)

> **Note:** Quantum packages (qiskit, pennylane, cirq) are not yet compatible with Python 3.14.
> See [ROADMAP.md](../ROADMAP.md) for status. Use Python 3.13 environment for quantum features.

```bash
# Requires Python 3.13 environment with quantum packages
python examples/quantum_computing.py
```

**Features:**
- Quantum circuit simulation
- Quantum-enhanced neural layers
- Hybrid quantum-classical models
- Multi-backend support (Qiskit, PennyLane, Cirq)

## Installation Options

Basic examples require only the core dependencies (installed by default):
```bash
uv sync
```

For vision extras:
```bash
uv sync --extra vision
```

For audio extras:
```bash
uv sync --extra audio
```

For GPU optimization (NVIDIA):
```bash
uv sync --extra gpu-nvidia
```

For inference providers (OpenAI, Anthropic, etc.):
```bash
uv sync --extra inference-providers
```

For all features:
```bash
uv sync --extra all
```

## Running on GPU

All examples automatically detect and use GPU if available:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

For optimal performance on RTX 5080 or similar hardware, ensure CUDA is properly installed.

## Contributing

To add new examples:
1. Create a new Python file in this directory
2. Follow the existing example structure
3. Include clear docstrings and comments
4. Update this README with usage instructions
5. Test thoroughly before submitting
