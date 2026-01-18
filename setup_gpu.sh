#!/bin/bash
# Setup script for GPU-accelerated CogSynDelta

echo "=========================================="
echo "CogSynDelta GPU Setup"
echo "=========================================="
echo ""

# Check if nvidia-smi works
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

echo "✓ NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check disk space
available_space=$(df / | awk 'NR==2 {print $4}')
required_space=3000000  # 3GB in KB

if [ "$available_space" -lt "$required_space" ]; then
    echo "⚠️  Warning: Less than 3GB free space available"
    echo "   Available: $(df -h / | awk 'NR==2 {print $4}')"
    echo "   Required: ~3GB for CUDA PyTorch"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Setting up virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing CUDA-enabled PyTorch (this may take a while)..."
echo "Package size: ~2.5GB"
echo ""

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Failed to install CUDA PyTorch"
    echo "   Try freeing up disk space and run again"
    exit 1
fi

echo ""
echo "Installing CogSynDelta dependencies..."
pip install numpy pyyaml psutil

echo ""
echo "Installing CogSynDelta in development mode..."
pip install -e .

echo ""
echo "Verifying GPU setup..."
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print("✓ GPU setup successful!")
else:
    print("❌ CUDA not available in PyTorch")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Setup complete!"
    echo "=========================================="
    echo ""
    echo "To activate the environment:"
    echo "  source venv/bin/activate"
    echo ""
    echo "To run GPU benchmarks:"
    echo "  python benchmarks/gpu_benchmark.py"
    echo ""
    echo "To deactivate:"
    echo "  deactivate"
    echo "=========================================="
else
    echo ""
    echo "❌ Setup incomplete. Please check errors above."
    exit 1
fi
