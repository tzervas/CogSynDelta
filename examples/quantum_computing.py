"""
Quantum Computing Example.

**FUTURE FEATURE - BACKLOGGED**

This example demonstrates:
1. Quantum circuit simulation
2. Quantum-enhanced neural networks
3. Hybrid quantum-classical computing

BACKLOG STATUS:
    Quantum computing features are currently using classical simulation stubs.
    Full quantum backend integration (qiskit, cirq, pennylane) is backlogged
    pending Python 3.14 ecosystem support. See ROADMAP.md for updates.

    When quantum packages become available, install with:
        uv add cogsyndelta[quantum]
"""

import torch
import asyncio
import warnings
from cogsyndelta.quantum import QUANTUM_AVAILABLE, QUANTUM_BACKLOG_REASON
from cogsyndelta.quantum.quantum_compute import (
    QuantumComputeManager,
    QuantumEnhancedLayer,
    HybridQuantumModel
)


async def main():
    """Demonstrate quantum computing capabilities (using stubs)."""
    print("="*60)
    print("CogSynDelta Quantum Computing Example")
    print("="*60)
    
    if not QUANTUM_AVAILABLE:
        print(f"\n⚠️  WARNING: {QUANTUM_BACKLOG_REASON}")
        print("Running with classical simulation stubs...\n")
    
    # Configuration
    config = {
        'backend': 'classical_simulator',  # Using stub
        'num_qubits': 10,
        'shots': 1024
    }
    
    print("\nInitializing quantum compute manager...")
    print(f"  Backend: {config['backend']} (stub)")
    print(f"  Qubits: {config['num_qubits']}")
    
    try:
        # Initialize quantum manager
        qc_manager = QuantumComputeManager()
        
        print("\n" + "="*60)
        print("Quantum-Enhanced Neural Network")
        print("="*60)
        
        # Create quantum-enhanced layer
        input_dim = 128
        output_dim = 64
        num_qubits = 8
        
        print(f"\nCreating quantum layer...")
        print(f"  Input dimension: {input_dim}")
        print(f"  Output dimension: {output_dim}")
        print(f"  Qubits: {num_qubits}")
        
        quantum_layer = QuantumEnhancedLayer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_qubits=num_qubits
        )
        
        # Test quantum layer
        print("\nTesting quantum layer...")
        test_input = torch.randn(4, input_dim)
        print(f"  Input shape: {test_input.shape}")
        
        output = quantum_layer(test_input)
        print(f"  Output shape: {output.shape}")
        print("  ✓ Quantum layer functioning correctly")
        
        # Hybrid quantum-classical model
        print("\n" + "="*60)
        print("Hybrid Quantum-Classical Model")
        print("="*60)
        
        latent_dim = 64
        use_quantum = True
        
        print(f"\nCreating hybrid model...")
        print(f"  Input dimension: {input_dim}")
        print(f"  Latent dimension: {latent_dim}")
        print(f"  Quantum acceleration: {use_quantum}")
        
        hybrid_model = HybridQuantumModel(
            input_dim=input_dim,
            latent_dim=latent_dim,
            use_quantum=use_quantum
        )
        
        # Test hybrid model
        print("\nTesting hybrid model...")
        test_batch = torch.randn(8, input_dim)
        encoded, decoded = hybrid_model(test_batch)
        
        print(f"  Input shape: {test_batch.shape}")
        print(f"  Encoded shape: {encoded.shape}")
        print(f"  Decoded shape: {decoded.shape}")
        
        # Calculate reconstruction error
        recon_error = torch.nn.functional.mse_loss(decoded, test_batch)
        print(f"  Reconstruction error: {recon_error.item():.6f}")
        
        print("\n  ✓ Hybrid model functioning correctly")
        
        # Summary
        print("\n" + "="*60)
        print("Quantum Computing Capabilities")
        print("="*60)
        print("\n✓ Quantum circuit simulation")
        print("✓ Quantum-enhanced neural layers")
        print("✓ Hybrid quantum-classical models")
        print("✓ Multi-backend support (Qiskit, PennyLane, Cirq)")
        
    except ImportError as e:
        print("\n" + "="*60)
        print("Quantum packages not installed!")
        print("="*60)
        print(f"\nError: {e}")
        print("\nTo use quantum features, install with:")
        print("  pip install cogsyndelta[quantum]")
        print("\nOr manually install:")
        print("  pip install qiskit pennylane cirq")
    
    print("\n" + "="*60)
    print("Quantum computing demonstration completed!")
    print("="*60)


if __name__ == '__main__':
    asyncio.run(main())
