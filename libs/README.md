# CogSynDelta Libraries

This directory contains extractable library components that are designed to be:
1. **Self-contained** - Minimal external dependencies
2. **Reusable** - Valuable as standalone packages
3. **Well-documented** - Full API documentation and examples

## Libraries

| Library | Description | Status | Dependencies |
|---------|-------------|--------|--------------|
| [algebraic-training](algebraic-training/) | Predict training outcomes without backprop | ✅ Active | torch |

## Usage

### Within CogSynDelta

Libraries are re-exported through the main package:

```python
from cogsyndelta.optimization import UnifiedAlgebraicTrainer, NTKPredictor
```

### Direct Import

Libraries can also be imported directly:

```python
from algebraic_training import UnifiedAlgebraicTrainer
import algebraic_training.functional as F

# Functional API
predictions = F.predict_training(model, train_x, train_y, epochs=100)
```

## Development

Each library follows the standard Python package structure:

```
libs/<library-name>/
├── pyproject.toml      # Package metadata and dependencies
├── README.md           # Library documentation
├── src/
│   └── <package_name>/
│       ├── __init__.py
│       └── *.py
└── tests/
    └── test_*.py
```

### Running Tests

```bash
# Run all library tests
uv run pytest libs/*/tests/ -v

# Run specific library tests
uv run pytest libs/algebraic-training/tests/ -v
```

### Adding New Libraries

1. Create directory structure under `libs/`
2. Add `pyproject.toml` with minimal dependencies
3. Create ADR documenting the extraction decision
4. Update this README

## Future Extraction

When a library is ready for external release:

1. Create new GitHub repository
2. Convert to git submodule: `git submodule add <repo-url> libs/<name>`
3. Update pyproject.toml to depend on the external package
4. Deprecate direct `libs/` import in favor of installed package

## License

All libraries inherit the MIT license from the parent CogSynDelta project unless otherwise specified in their individual `pyproject.toml`.
