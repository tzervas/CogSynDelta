# CogSynDelta Examples

This directory contains example scripts demonstrating various features of CogSynDelta.

## Examples

### 1. Basic Training (`basic_training.py`)
Demonstrates basic model training on MNIST dataset:
```bash
python examples/basic_training.py
```

**Features:**
- Loading and configuring the integrated system
- Training PCN-VAE-GAN on MNIST
- Monitoring training metrics

### 2. Memory Management (`memory_management.py`)
Demonstrates the tiered memory system:
```bash
python examples/memory_management.py
```

**Features:**
- Active, short-term, and long-term memory tiers
- Memory compression and persistence
- Similarity-based memory retrieval
- Memory tier management

### 3. API Server (`api_server.py`)
Starts the FastAPI server:
```bash
python examples/api_server.py
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
python examples/self_improving_agents.py
```

**Features:**
- Self-improving language agents
- Meta-learning optimization
- Multi-agent collaboration
- Adaptive behavior

### 5. Quantum Computing (`quantum_computing.py`)
Demonstrates quantum-enhanced neural networks:
```bash
# Install quantum dependencies first
pip install cogsyndelta[quantum]

# Run example
python examples/quantum_computing.py
```

**Features:**
- Quantum circuit simulation
- Quantum-enhanced neural layers
- Hybrid quantum-classical models
- Multi-backend support (Qiskit, PennyLane, Cirq)

## Requirements

Basic examples require only the core dependencies:
```bash
pip install cogsyndelta
```

For quantum examples:
```bash
pip install cogsyndelta[quantum]
```

For vision examples:
```bash
pip install cogsyndelta[vision]
```

For audio examples:
```bash
pip install cogsyndelta[audio]
```

For all features:
```bash
pip install cogsyndelta[all]
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
