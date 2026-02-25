# Full SDD workflow

## Configuration
- **Artifacts Path**: {@artifacts_path} → `.zenflow/tasks/{task_id}`

---

## Agent Instructions

If you are blocked and need user clarification, mark the current step with `[!]` in plan.md before stopping.

---

## Workflow Steps

### [x] Step: Requirements
<!-- chat-id: b617b37a-8436-4f33-aeed-4b756b8c1573 -->

Create a Product Requirements Document (PRD) based on the feature description.

1. Review existing codebase to understand current architecture and patterns
2. Analyze the feature definition and identify unclear aspects
3. Ask the user for clarifications on aspects that significantly impact scope or user experience
4. Make reasonable decisions for minor details based on context and conventions
5. If user can't clarify, make a decision, state the assumption, and continue

Save the PRD to `{@artifacts_path}/requirements.md`.

### [x] Step: Technical Specification
<!-- chat-id: 872e3b62-47bf-4ce4-92e1-d01df7c7efad -->

Create a technical specification based on the PRD in `{@artifacts_path}/requirements.md`.

1. Review existing codebase architecture and identify reusable components
2. Define the implementation approach

Save to `{@artifacts_path}/spec.md` with:
- Technical context (language, dependencies)
- Implementation approach referencing existing code patterns
- Source code structure changes
- Data model / API / interface changes
- Delivery phases (incremental, testable milestones)
- Verification approach using project lint/test commands

### [x] Step: Planning
<!-- chat-id: 99d26446-02ac-4b7d-8bb3-42b9d9a77e64 -->

Create a detailed implementation plan based on `{@artifacts_path}/spec.md`.

1. Break down the work into concrete tasks
2. Each task should reference relevant contracts and include verification steps
3. Replace the Implementation step below with the planned tasks

---

## Implementation Steps

### [x] Step 1: Project skeleton, config, and exceptions
<!-- chat-id: 7bf4645f-32e3-4708-8d5e-7d80bb45bfc3 -->

Create the new `src/qr_sampler/` package layout, build system, tooling, exception hierarchy, and pydantic-settings configuration. Remove the old `qc_sampler/` codebase. Write config tests.

**Skeleton & build system:**
- [ ] Create `src/qr_sampler/` directory tree with all subpackage `__init__.py` files:
  - `src/qr_sampler/__init__.py` (package version, re-exports)
  - `src/qr_sampler/py.typed` (empty PEP 561 marker)
  - Subpackages: `entropy/`, `amplification/`, `temperature/`, `selection/`, `logging/`, `proto/`
- [ ] Write `pyproject.toml` from scratch per spec section 10:
  - `setuptools` + `setuptools-scm` build backend
  - Dependencies: `numpy>=1.24.0`, `pydantic>=2.0.0`, `pydantic-settings>=2.0.0`
  - Optional extras: `[grpc]` (grpcio, protobuf), `[dev]` (pytest, pytest-cov, scipy, ruff, mypy, pre-commit, bandit)
  - Entry points: `vllm.logits_processors` and `qr_sampler.entropy_sources`
  - Tool config: ruff (target py310, line-length 100), mypy (strict), pytest, coverage
- [ ] Update `.gitignore` for `src/` layout, caches, build artifacts, `.env`
- [ ] Create `LICENSE` (Apache 2.0 full text)
- [ ] Create `.pre-commit-config.yaml` with ruff, mypy, bandit, standard hooks
- [ ] Remove old `qc_sampler/`, `qc_sampler.egg-info/`, `tests/`, `README.md`

**Exceptions & config:**
- [ ] `src/qr_sampler/exceptions.py` — `QRSamplerError`, `EntropyUnavailableError`, `ConfigValidationError`, `SignalAmplificationError`, `TokenSelectionError` (spec section 4.2)
- [ ] `src/qr_sampler/config.py` — `QRSamplerConfig(BaseSettings)` with all fields from spec section 4.1:
  - `SettingsConfigDict(env_prefix="QR_", env_file=".env", extra="ignore")`
  - Infrastructure fields (non-overridable): `grpc_server_address`, `grpc_timeout_ms`, `grpc_retry_count`, `grpc_mode`, `fallback_mode`, `entropy_source_type`
  - Per-request fields: amplification, temperature, selection, logging params
  - `_PER_REQUEST_FIELDS` frozenset
  - `resolve_config(defaults, extra_args)` — creates new config via `model_copy(update=...)`, validates `qr_` prefixed keys, rejects non-overridable fields
  - `validate_extra_args(extra_args)` — validates keys without creating config
- [ ] `tests/__init__.py`, `tests/conftest.py` — shared fixtures: `default_config`, `sample_logits`
- [ ] `tests/test_config.py` — test defaults, env var loading, per-request resolution, non-overridable field rejection, `validate_extra_args`, type coercion, invalid key detection

**References:** spec sections 3, 4.1, 4.2, 10; invariants 1, 7

**Verification:**
```bash
pip install -e ".[dev]"
python -c "import qr_sampler; print(qr_sampler.__version__)"
pytest tests/test_config.py -v
ruff check src/ tests/
mypy --strict src/
```

### [x] Step 2: Core pipeline components — amplification, temperature, selection, logging
<!-- chat-id: 65ddfbfe-024b-4f2e-b75b-a4019c6723cc -->

Implement all four stateless pipeline subsystems with their ABCs, registries, concrete implementations, and tests. These are independent of each other and only depend on config/exceptions from step 1.

**Signal amplification (`amplification/`):**
- [ ] `base.py` — `SignalAmplifier` ABC with `amplify(raw_bytes) -> AmplificationResult`; `AmplificationResult` frozen dataclass with `__slots__` (fields: `u`, `diagnostics`)
- [ ] `registry.py` — `AmplifierRegistry` with `register(name)` decorator, `get(name)`, `build(config)`
- [ ] `zscore.py` — `ZScoreMeanAmplifier` registered as `"zscore_mean"`: bytes → uint8 → mean → z-score (derived SEM = `population_std / sqrt(N)`) → normal CDF via `erf` → clamped uniform. Raises `SignalAmplificationError` on empty input.
- [ ] `__init__.py` — re-exports
- [ ] `tests/test_amplification/test_zscore.py` — known values, SEM derivation, edge cases (empty, single byte), frozen immutability, registry tests

**Temperature strategies (`temperature/`):**
- [ ] `base.py` — `TemperatureResult` frozen dataclass (`temperature`, `shannon_entropy`, `diagnostics`); `TemperatureStrategy` ABC; `compute_shannon_entropy()` helper (numerically stable softmax + Σ(-p ln p))
- [ ] `registry.py` — `TemperatureStrategyRegistry` with `build(config, vocab_size)`
- [ ] `fixed.py` — `FixedTemperatureStrategy` registered as `"fixed"`: returns constant, always computes Shannon entropy
- [ ] `edt.py` — `EDTTemperatureStrategy` registered as `"edt"`: `H_norm = H / ln(V)`, `T = base × H_norm^exp`, clamped to `[min, max]`
- [ ] `__init__.py` — re-exports
- [ ] `tests/test_temperature/test_fixed.py`, `test_edt.py` — Shannon entropy known values, fixed returns constant, EDT monotonicity, clamping, exponent effects, frozen immutability

**Token selection (`selection/`):**
- [ ] `types.py` — `SelectionResult` frozen dataclass (`token_id`, `token_rank`, `token_prob`, `num_candidates`, `diagnostics`)
- [ ] `selector.py` — `TokenSelector.select(logits, temperature, top_k, top_p, u)`: temperature scaling → top-k → softmax → top-p → descending sort → CDF → `searchsorted`. Raises `TokenSelectionError` if no candidates survive.
- [ ] `__init__.py` — re-exports
- [ ] `tests/test_selection/test_selector.py` — CDF known values (u=0 → most probable), top-k/top-p filtering, edge cases (identical logits, single survivor, all-inf-except-one), frozen immutability

**Diagnostic logging (`logging/`):**
- [ ] `types.py` — `TokenSamplingRecord` frozen dataclass with `__slots__` (16 fields: timing, entropy source, amplification, temperature, selection, config_hash)
- [ ] `logger.py` — `SamplingLogger`: log levels `"none"/"summary"/"full"`, uses `logging.getLogger("qr_sampler")`, `diagnostic_mode` stores records in memory
- [ ] `__init__.py` — re-exports
- [ ] `tests/test_logging/test_logger.py` — record immutability, log level behavior, diagnostic mode storage, summary stats

**References:** spec sections 4.10–4.13; invariants 2, 5, 6, 9

**Verification:**
```bash
pytest tests/test_amplification/ tests/test_temperature/ tests/test_selection/ tests/test_logging/ -v
mypy --strict src/qr_sampler/amplification/ src/qr_sampler/temperature/ src/qr_sampler/selection/ src/qr_sampler/logging/
```

### [x] Step 3: Entropy source system — ABC, local sources, gRPC, fallback, registry
<!-- chat-id: 5aaa16af-16f0-433e-aae0-35541756709c -->

Implement the full entropy subsystem: base ABC, entry-point registry, all built-in sources (system, timing, mock, quantum gRPC), fallback wrapper, and proto stubs. Write tests alongside.

**Entropy ABC & registry:**
- [ ] `entropy/base.py` — `EntropySource` ABC: abstract `name`, `is_available`, `get_random_bytes(n)`, `close()`; default `get_random_float64(shape, out=None)` via bytes; concrete `health_check()`
- [ ] `entropy/registry.py` — `EntropySourceRegistry`: `register(name)` decorator, `get(name)` with lazy entry-point loading from `qr_sampler.entropy_sources`, `list_available()`, `_load_entry_points()`

**Local sources:**
- [ ] `entropy/system.py` — `SystemEntropySource` (`"system"`): wraps `os.urandom()`, always available
- [ ] `entropy/timing.py` — `TimingNoiseSource` (`"timing_noise"`): CPU timing jitter, 8 measurements per byte, LSB extraction. Experimental warning.
- [ ] `entropy/mock.py` — `MockUniformSource` (`"mock_uniform"`): configurable mean/seed, normal distribution, clamped [0, 255]

**Fallback:**
- [ ] `entropy/fallback.py` — `FallbackEntropySource`: composition wrapper, only catches `EntropyUnavailableError`, exposes `last_source_used`, logs warning on fallback

**gRPC proto stubs:**
- [ ] `proto/entropy_service.proto` — `GetEntropy` (unary) + `StreamEntropy` (bidi); `EntropyRequest` (`bytes_needed`, `sequence_id`), `EntropyResponse` (`data`, `sequence_id`, `generation_timestamp_ns`, `device_id`)
- [ ] `proto/entropy_service_pb2.py` — hand-written message stubs
- [ ] `proto/entropy_service_pb2_grpc.py` — hand-written gRPC client stubs (unary + streaming)

**QuantumGrpcSource:**
- [ ] `entropy/quantum.py` — `QuantumGrpcSource` (`"quantum_grpc"`):
  - `grpc.aio` channel with keepalive options, TCP/Unix socket auto-detection
  - Background asyncio event loop thread, sync wrapper via `run_coroutine_threadsafe()`
  - Three transport modes: `_fetch_unary`, `_fetch_server_streaming`, `_fetch_bidi_streaming`
  - Circuit breaker: rolling P99 latency (`deque`, max 100), adaptive timeout = `max(5ms, P99 × 1.5)`, circuit open/half-open states
  - `close()` cleans up channel, loop, thread

**`entropy/__init__.py`** — re-exports

**Tests:**
- [ ] `tests/test_entropy/test_system.py` — correct byte count, always available
- [ ] `tests/test_entropy/test_timing.py` — correct byte count, non-zero output
- [ ] `tests/test_entropy/test_mock.py` — seeded reproducibility, bias simulation
- [ ] `tests/test_entropy/test_fallback.py` — primary delegation, fallback on `EntropyUnavailableError` only, error propagation, `last_source_used`
- [ ] `tests/test_entropy/test_registry.py` — decorator registration, entry-point discovery (mocked), lazy loading, unknown name error
- [ ] `tests/test_entropy/test_quantum.py` — mocked gRPC for all 3 modes, circuit breaker behavior, address parsing, error → `EntropyUnavailableError`

**References:** spec sections 4.3–4.9, 4.15; invariants 3, 4, 10, 11, 12

**Verification:**
```bash
pytest tests/test_entropy/ -v
mypy --strict src/qr_sampler/entropy/ src/qr_sampler/proto/
```

### [x] Step 4: Processor integration and statistical tests
<!-- chat-id: 6b3a9998-cc3c-47f3-9ba9-2c513b29e0cd -->

Wire all subsystems into the vLLM LogitsProcessor. Write integration tests and statistical validation tests.

**Processor:**
- [ ] `src/qr_sampler/processor.py` — `QRSamplerLogitsProcessor`:
  - `__init__(vllm_config, device, is_pin_memory)`: extract vocab_size, load config, build entropy source (+ fallback wrap), build amplifier, build temperature strategy, create TokenSelector + SamplingLogger, pre-allocate one-hot template tensor and pinned CPU buffer
  - `is_argmax_invariant() -> False`
  - `validate_params(params)` — calls `validate_extra_args()`
  - `update_state(batch_update)` — removed → moved → added order; per-request config resolution
  - `apply(logits)` — per-row: config → numpy → temperature → entropy (just-in-time) → amplify → select → one-hot force → log → return
- [ ] Update `src/qr_sampler/__init__.py` — export processor, config, key types, version

**Integration tests:**
- [ ] `tests/test_processor.py`:
  - Full pipeline with `MockUniformSource`: one-hot output verification
  - Batch processing: multiple rows
  - `update_state()`: add/remove/move
  - `validate_params()`: valid and invalid extra_args
  - `is_argmax_invariant()` returns False
  - Per-request config resolution
  - Short-circuit: empty batch
  - Token IDs match CDF selection for known seeds

**Statistical validation tests:**
- [ ] `tests/test_statistical_properties.py`:
  - KS-test for u-value uniformity (1000+ u-values, `MockUniformSource(mean=127.5)`, scipy KS vs Uniform(0,1), p > 0.05)
  - Bias detection (`MockUniformSource(mean=128.0)`, mean(u) deviates from 0.5)
  - EDT monotonicity (increasing entropy → increasing temperature)
  - CDF coverage (uniform u → probability-respecting token selection)
  - One-hot correctness (multi-row batch, exactly one 0.0 per row)

**References:** spec sections 4.14, 7.3; invariants 7, 8, 10

**Verification:**
```bash
pytest tests/test_processor.py tests/test_statistical_properties.py -v
mypy --strict src/qr_sampler/processor.py
```

### [x] Step 5: Examples, deployment templates, and reference servers
<!-- chat-id: ef4bd05e-ceb3-4638-910b-7b1a0928d389 -->

Create reference entropy server implementations, Docker/systemd templates.

**Example servers (`examples/servers/`):**
- [ ] `simple_urandom_server.py` (~50 lines): minimal gRPC server, `os.urandom`, unary + bidi streaming, fully commented "hello world"
- [ ] `timing_noise_server.py`: CPU timing jitter entropy server, educational non-QRNG example
- [ ] `qrng_template_server.py`: annotated template with `# TODO: YOUR HARDWARE CODE HERE`, error handling, health checks, all 3 transport modes

**Deployment (`examples/docker/`, `examples/systemd/`):**
- [ ] `Dockerfile.entropy-server`: slim Python, installs grpcio/protobuf, runs any example server
- [ ] `docker-compose.yml`: full stack (entropy-server + vLLM with qr-sampler), pre-configured env vars
- [ ] `qr-entropy-server.service`: systemd unit file with restart-on-failure, logging, env file

**References:** spec section 3.3, requirements REQ-DOC-06

**Verification:** `python examples/servers/simple_urandom_server.py --help` runs successfully.

### [x] Step 6: Documentation, community files, CLAUDE.md, and final verification
<!-- chat-id: 38336e17-0f0b-4d3e-87c6-d8b404c416ee -->

Write all documentation, community files, CI configuration, update CLAUDE.md, and run full quality gates.

**README.md** (comprehensive, per REQ-DOC-01):
- [ ] Header/badges, "What is qr-sampler?", Motivation (consciousness-research context)
- [ ] "How it works" with ASCII pipeline diagram
- [ ] Quick start (install, reference server, configure vLLM, run inference)
- [ ] Configuration reference table (all fields, env vars, defaults, per-request flag)
- [ ] gRPC transport modes guide (unary vs server-streaming vs bidi)
- [ ] Entropy sources guide (built-ins, fallback, choosing)
- [ ] **"Setting up your own entropy source"** — comprehensive guide:
  - Conceptual overview, minimal server walkthrough (5 min), custom QRNG from template, deployment (Docker/systemd), validation/debugging, advanced topics (Unix sockets, TLS, tuning)
- [ ] Signal amplification, temperature strategies, plugin architecture, contributing, license

**Community files:**
- [ ] `CONTRIBUTING.md` — dev setup (uv, pre-commit), conventions, testing, PR process, adding new components
- [ ] `CHANGELOG.md` — Keep a Changelog format, `[Unreleased]` section
- [ ] `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1

**CI & GitHub config:**
- [ ] `.github/workflows/ci.yml` — lint + format + typecheck + test matrix (Python 3.10–3.13) + Codecov
- [ ] `.github/ISSUE_TEMPLATE/bug_report.yml`, `feature_request.yml`
- [ ] `.github/PULL_REQUEST_TEMPLATE.md`
- [ ] `.github/dependabot.yml`

**CLAUDE.md:**
- [ ] Rewrite for qr-sampler: project description, commands, file map (`src/qr_sampler/` subpackages), updated invariants, coding conventions (`QR_` prefix, pydantic-settings, `"qr_sampler"` logger), data flows (3 gRPC modes, circuit breaker, entry-point registry), how to add components, testing approach, proto stubs, dependencies

**Final verification:**
- [ ] `pytest tests/ -v --cov=src/qr_sampler --cov-report=term-missing` — all pass, coverage >= 90%
- [ ] `ruff check src/ tests/` and `ruff format --check src/ tests/` — zero violations
- [ ] `mypy --strict src/` — zero errors
- [ ] `pip install -e .`, `pip install -e ".[dev]"`, `pip install -e ".[grpc]"` — all work
- [ ] Entry point registration verified
- [ ] No `print()` in `src/`, no wildcard imports, Google-style docstrings on all public APIs
- [ ] Fix any issues found

**References:** requirements REQ-DOC-01 through REQ-DOC-06, REQ-TOOL-06; spec section 7
