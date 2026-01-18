## Description

<!-- Provide a brief description of the changes in this PR -->

## Related Issue

<!-- Link to the issue this PR addresses: Fixes #123, Relates to #456 -->

Fixes #

## Type of Change

<!-- Mark the appropriate option with an [x] -->

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to change)
- [ ] 📝 Documentation update
- [ ] ♻️ Refactor (no functional changes)
- [ ] 🧪 Test improvements
- [ ] 🔧 Configuration/tooling changes

## Changes Made

<!-- List the specific changes made in this PR -->

-
-
-

## Testing

<!-- Describe the tests you ran and how to reproduce them -->

- [ ] All existing tests pass (`uv run pytest tests/ -v`)
- [ ] New tests added for new functionality
- [ ] Manual testing performed

### Test Commands Run

```bash
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run mypy src/
```

## Quality Checklist

<!-- All items must be checked before merge -->

### Code Quality

- [ ] Code follows the project's coding standards
- [ ] All functions have Google-style docstrings with Args/Returns/Raises
- [ ] All functions have complete type hints
- [ ] No trailing whitespace
- [ ] Line length ≤ 100 characters

### Documentation

- [ ] Code is self-documenting with clear variable names
- [ ] Complex logic has explanatory comments (the "why")
- [ ] Public API changes are documented
- [ ] CHANGELOG.md updated (for features/fixes)

### Testing

- [ ] Tests pass locally
- [ ] New tests cover the changes
- [ ] No unsubstantiated performance claims (benchmarks provided if claimed)

### Security

- [ ] No secrets or credentials in code
- [ ] New dependencies have acceptable licenses (see LICENSES/LICENSE_TRACKER.md)

## Screenshots/Outputs

<!-- If applicable, add screenshots or terminal output to demonstrate changes -->

## Additional Notes

<!-- Any additional context, considerations, or known limitations -->

---

### Reviewer Checklist

<!-- For reviewers to complete -->

- [ ] Code review completed
- [ ] Architecture/design is appropriate
- [ ] Tests are sufficient
- [ ] Documentation is adequate
- [ ] No breaking changes (or properly documented if intentional)
