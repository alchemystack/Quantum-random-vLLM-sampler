## Summary

<!-- What does this PR do? Why? -->

## Changes

<!-- Bullet list of changes -->

-

## Testing

<!-- How was this tested? -->

- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] New tests added for new functionality
- [ ] Lint passes (`ruff check src/ tests/`)
- [ ] Format passes (`ruff format --check src/ tests/`)
- [ ] Type check passes (`mypy --strict src/`)

## Checklist

- [ ] Code follows the project coding conventions
- [ ] No `print()` statements in `src/` (use `logging`)
- [ ] All result types are frozen dataclasses
- [ ] New components use the registry pattern (no if/else chains)
- [ ] Documentation updated if needed
