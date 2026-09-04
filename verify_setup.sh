#!/bin/bash
# Verification script for CogSynDelta improvements

echo "=========================================="
echo "CogSynDelta Setup Verification"
echo "=========================================="
echo ""

# Check Python version
echo "1. Python version:"
python3 --version
echo ""

# Check if PyTorch is available
echo "2. PyTorch availability:"
python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "  ✗ PyTorch not installed"
echo ""

# Check type hint coverage
echo "3. Type hint coverage:"
total=$(grep -rn "def " src/ --include="*.py" | grep -v "__pycache__" | wc -l)
with_hints=$(grep -rn " -> " src/ --include="*.py" | grep -v "__pycache__" | wc -l)
coverage=$(python3 -c "print(f'{$with_hints/$total*100:.1f}')")
echo "  Functions: $with_hints/$total ($coverage% typed)"
echo ""

# Check CI/CD files
echo "4. CI/CD setup:"
[ -f .github/workflows/ci.yml ] && echo "  ✓ GitHub Actions workflow" || echo "  ✗ Missing workflow"
[ -f .pre-commit-config.yaml ] && echo "  ✓ Pre-commit config" || echo "  ✗ Missing pre-commit"
echo ""

# Check examples
echo "5. Example scripts:"
for example in examples/*.py; do
    [ -f "$example" ] && echo "  ✓ $(basename $example)" || echo "  ✗ Missing $example"
done
echo ""

# Check if tests are discoverable
echo "6. Test discovery:"
python3 -m pytest --collect-only tests/ -q 2>&1 | head -1
echo ""

# Check entry points
echo "7. Entry points in pyproject.toml:"
grep "cogsyndelta-benchmark" pyproject.toml && echo "  ✓ Benchmark entry point configured" || echo "  ✗ Missing entry point"
echo ""

echo "=========================================="
echo "Verification complete!"
echo "=========================================="
