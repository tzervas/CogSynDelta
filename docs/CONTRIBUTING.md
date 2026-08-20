# Contributing to CogSynDelta

Thank you for your interest in contributing to CogSynDelta!

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## How to Contribute

### Reporting Issues

1. Search existing issues first
2. Use issue templates
3. Provide reproducible examples
4. Include system information

### Suggesting Features

1. Check existing feature requests
2. Explain the use case
3. Describe the proposed solution
4. Consider implementation complexity

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Run benchmarks
7. Submit PR

## Development Setup

Python floor is **3.12** (`requires-python = ">=3.12,<3.14"`). Do not install 3.14
for this tree.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/tzervas/CogSynDelta.git
cd CogSynDelta

# Create .venv (uv reads .python-version → 3.12)
uv sync --group dev

# Same gates as GitHub Actions (lint, mypy, quality>=90, poc-ci, full pytest)
# Run this before every push. --cpu matches CI torch wheels; default is cu128.
./scripts/ci_local.sh
./scripts/ci_local.sh --poc          # fast loop
./scripts/ci_local.sh --cpu          # exact CI CPU-torch sync
```

### Using uvx for Development Tools

```bash
# Versions pinned to .github/workflows/ci.yml
uvx ruff@0.14.13 check src/ tests/ benchmarks/ scripts/ examples/
uvx ruff@0.14.13 format --check src/ tests/ benchmarks/ scripts/ examples/
uvx mypy@1.19.1 --with types-PyYAML src/
```

## Coding Standards

### Python Style

Follow PEP 8 with these specifics:

- Line length: 100 characters
- Use type hints
- Document all public APIs
- Use meaningful variable names

```python
def process_embedding(embedding: torch.Tensor,
                     importance: float = 1.0) -> Dict[str, Any]:
    """
    Process embedding with specified importance.

    Args:
        embedding: Input embedding tensor [batch, dim]
        importance: Importance weight (0.0-1.0)

    Returns:
        Dictionary with processed results
    """
    # Implementation
    pass
```

### Documentation

- Docstrings for all public functions/classes
- Type hints for all parameters
- Examples in docstrings
- Update relevant .md files

### Testing

- Unit tests for new features
- Integration tests for components
- Benchmarks for performance claims
- Test coverage >80%

```python
def test_compression():
    """Test compression maintains fidelity."""
    store = DenseDifferentialMemoryStore(embed_dim=512)
    original = torch.randn(10, 512)

    # Compress and retrieve
    for i in range(10):
        stats = store.compress_and_store(original[i], f"test_{i}")
        assert stats['fidelity_score'] > 0.90
```

### Benchmarking

All performance claims MUST be validated:

```python
from benchmarks import PerformanceBenchmark

benchmark = PerformanceBenchmark()
result = benchmark.measure_inference_time(model, test_data)
print(f"Measured: {result.value:.2f} ± {result.std_dev:.2f} ms")
```

## Areas for Contribution

### High Priority

1. **New Compute Backends**
   - Neuromorphic processors
   - Photonic computing
   - Analog compute

2. **Model Sections**
   - Additional brain regions
   - Specialized processing units
   - Domain-specific sections

3. **Input/Output Adapters**
   - New video sources
   - Audio processing
   - Sensor integration

4. **Performance Optimizations**
   - Faster inference
   - Better compression
   - Memory efficiency

### Medium Priority

1. **Documentation**
   - Tutorials
   - Examples
   - API references

2. **Testing**
   - More test coverage
   - Benchmark suites
   - Integration tests

3. **Tools**
   - Debugging utilities
   - Visualization tools
   - Monitoring dashboards

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Benchmarks run (if applicable)
- [ ] No wild claims (all measured)
- [ ] Safeguards in place

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation
- [ ] Other

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Benchmarks run

## Benchmarks (if applicable)
Measured performance:
- Metric 1: X.XX units (baseline: Y.YY)
- Metric 2: X.XX units (baseline: Y.YY)

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass
- [ ] No breaking changes (or documented)
```

### Review Process

1. Automated checks run
2. Maintainer review
3. Address feedback
4. Re-review if needed
5. Merge when approved

## Commit Messages

Use conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `perf`: Performance improvement
- `refactor`: Code refactoring
- `test`: Testing
- `chore`: Maintenance

Examples:
```
feat(memory): add dense differential compression

Implements 10-50x compression with >0.95 fidelity.
Measured on test data with statistical validation.

Closes #123
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- Open an issue
- Start a discussion
- Contact maintainers

---

Thank you for contributing to CogSynDelta!
