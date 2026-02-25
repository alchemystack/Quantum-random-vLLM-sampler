# Contributing to qr-sampler

Thank you for your interest in contributing to qr-sampler. This document covers everything you need to get started.

## Development setup

### Prerequisites

- Python 3.10 or later
- Git

### Install

```bash
git clone https://github.com/qr-sampler/qr-sampler.git
cd qr-sampler
pip install -e ".[dev]"
```

This installs the package in editable mode with all development dependencies (pytest, ruff, mypy, scipy, pre-commit, bandit, gRPC).

### Pre-commit hooks

```bash
pre-commit install
```

This runs ruff (lint + format), mypy, bandit, and standard checks (trailing whitespace, YAML/TOML validation, merge conflict detection) on every commit.

## Running tests

```bash
# All tests
pytest tests/ -v

# Specific subsystem
pytest tests/test_config.py -v
pytest tests/test_amplification/ -v
pytest tests/test_temperature/ -v
pytest tests/test_selection/ -v
pytest tests/test_logging/ -v
pytest tests/test_entropy/ -v

# With coverage
pytest tests/ -v --cov=src/qr_sampler --cov-report=term-missing
```

Coverage must stay at or above 90%.

## Code quality checks

```bash
# Lint
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/

# Format check
ruff format --check src/ tests/

# Auto-format
ruff format src/ tests/

# Type check
mypy --strict src/

# Security scan
bandit -r src/
```

## Coding conventions

- **Python 3.10+** — use `X | Y` union syntax, not `Union[X, Y]`
- **Type hints** on all function signatures and return types
- **Docstrings** — Google style on every public class and method
- **Imports** — standard library first, third-party second, local third. No wildcard imports.
- **Line length** — 100 characters (configured in `pyproject.toml`)
- **Errors** — custom exception hierarchy rooted in `QRSamplerError`. Never catch bare `Exception` (health checks are the sole exception with `# noqa` comments).
- **No `print()`** — use `logging.getLogger("qr_sampler")` for all output
- **No global mutable state** outside processor instances and module-load registries
- **Frozen dataclasses** for all result types (`AmplificationResult`, `TemperatureResult`, `SelectionResult`, `TokenSamplingRecord`)
- **`__slots__`** on frozen dataclasses in the hot path

## Architecture invariants

These are fundamental design rules. Do not break them:

1. **No hardcoded values.** Every numeric constant traces to a `QRSamplerConfig` field. Mathematical constants (e.g., `sqrt(2)`) are acceptable.

2. **Registry pattern for all strategies.** New `EntropySource`, `SignalAmplifier`, or `TemperatureStrategy` implementations are registered via decorators. No if/else chains for strategy selection.

3. **ABCs define contracts.** All concrete implementations subclass the relevant ABC. The processor and factory only reference abstract types.

4. **SEM is derived, never stored.** Standard error of mean = `population_std / sqrt(sample_count)`, computed at amplification time.

5. **Per-request config is immutable.** `resolve_config()` creates new instances; it never mutates defaults. Infrastructure fields cannot be overridden per-request.

6. **Just-in-time entropy.** Physical entropy generation must occur only when `get_random_bytes()` is called — after logits are available. No pre-buffering.

## Adding new components

### New entropy source

1. Create a class in `src/qr_sampler/entropy/` subclassing `EntropySource`
2. Implement `name`, `is_available`, `get_random_bytes(n)`, `close()`
3. Raise `EntropyUnavailableError` from `get_random_bytes()` on failure
4. Register with `@register_entropy_source("my_name")`
5. Add entry point in `pyproject.toml` under `[project.entry-points."qr_sampler.entropy_sources"]`
6. Add tests in `tests/test_entropy/`

### New signal amplifier

1. Create a class in `src/qr_sampler/amplification/` subclassing `SignalAmplifier`
2. Implement `amplify(raw_bytes) -> AmplificationResult`
3. Register with `@AmplifierRegistry.register("my_name")`
4. Add tests in `tests/test_amplification/`

### New temperature strategy

1. Create a class in `src/qr_sampler/temperature/` subclassing `TemperatureStrategy`
2. Implement `compute_temperature(logits, config) -> TemperatureResult`
3. Always compute and return Shannon entropy (the logging subsystem depends on it)
4. Register with `@TemperatureStrategyRegistry.register("my_name")`
5. Add tests in `tests/test_temperature/`

### New config field

1. Add the field to `QRSamplerConfig` in `config.py` with a default and description
2. If per-request overridable, add to `_PER_REQUEST_FIELDS`
3. Environment variable `QR_{FIELD_NAME_UPPER}` is auto-supported
4. Extra args key `qr_{field_name}` is auto-supported
5. Add tests in `tests/test_config.py`

## Pull request process

1. Fork the repository and create a branch from `main`
2. Make your changes, following the coding conventions above
3. Add or update tests for your changes
4. Run the full quality gate:
   ```bash
   ruff check src/ tests/ && ruff format --check src/ tests/ && mypy --strict src/ && pytest tests/ -v
   ```
5. Write a clear PR description explaining *what* and *why*
6. Submit the PR for review

## Reporting bugs

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version, OS, vLLM version
- Relevant configuration (`QR_*` env vars)

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
