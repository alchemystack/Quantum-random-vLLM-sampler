# Product Requirements Document: qr-sampler

## 1. Executive Summary

**qr-sampler** is a vLLM V1 LogitsProcessor plugin that replaces vLLM's built-in random token sampling with sampling driven by arbitrary external randomness sources — quantum random number generators (QRNGs), processor timing jitter, or any user-supplied entropy. Its core purpose is to provide infrastructure for studying non-deterministic LLMs as potentially consciousness-interactive entities, where modifications to the underlying randomness source may influence token selection.

This project builds on the architecture and learnings from the `qc-sampler` prototype but is a ground-up redesign under the `qr-sampler` name. The existing prototype's algorithms and patterns inform the new design, but the new codebase is not constrained by old invariants — every component is re-evaluated for the new architecture and discarded or transformed as needed.

### 1.1 Goals

1. **Universal randomness connector**: Allow anyone to plug any entropy source into any vLLM-hosted LLM's token sampling, with just-in-time entropy generation (no pre-buffering).
2. **Extreme modularity**: Every component (entropy source, signal amplifier, temperature strategy) is an ABC with a plugin registry discoverable via Python entry points, so community contributions slot in without modifying core code.
3. **Production performance**: Batch-level vLLM V1 LogitsProcessor with pre-allocated tensors, zero-copy paths, and configurable gRPC transport (unary, server-streaming, or bidirectional streaming) for flexible entropy delivery.
4. **Open-source excellence**: Apache 2.0 license, src layout, modern Python toolchain (uv, ruff, mypy, pytest), GitHub Actions CI, comprehensive README, and contributor-friendly documentation.
5. **Research-ready**: Full per-token diagnostic logging, statistical validation tools, and metadata that distinguishes QRNG-sourced tokens from fallback tokens for downstream analysis.
6. **Zero-friction entropy source onboarding**: A clear workflow, CLI scaffolding tools, reference server implementations, and step-by-step guides make it trivial for anyone to connect their own randomness source.

### 1.2 Non-Goals

- Modifying vLLM source code (pure plugin only).
- Implementing specific QRNG hardware drivers (the plugin consumes entropy via gRPC; hardware interfacing is the server's responsibility).
- Supporting vLLM V0's legacy per-request callable interface.
- Providing a web UI or dashboard (CLI and programmatic API only).
- PyPI trusted publisher automation (manual publishing is sufficient for now).

---

## 2. Background

### 2.1 Current State (qc-sampler v0.1.0)

The existing prototype implements:
- `QuantumConsciousnessProcessor` as a vLLM LogitsProcessor
- Per-token pipeline: temperature strategy -> QRNG bytes -> z-score amplification -> CDF token selection -> one-hot logits forcing
- Frozen dataclass config (`QCSamplingConfig`) with 25 fields, env var loading, per-request overrides
- Registry-based factory pattern for entropy sources, amplifiers, and temperature strategies
- gRPC unary client (`GrpcEntropySource`) with prefetch buffer and retry logic
- Test suite with statistical validation (KS uniformity, bias detection, EDT monotonicity)

### 2.2 Relationship to the Prototype

The existing `qc-sampler` is a **reference and learning source**, not a migration base. The new `qr-sampler` is designed from first principles:

- **Algorithms that proved correct** (z-score amplification, CDF token selection, EDT temperature) are reimplemented cleanly in the new architecture.
- **Patterns that worked well** (ABC contracts, registry-based factories, fallback wrapping) are adopted and improved.
- **Patterns that don't fit** (thread-based prefetch that violates just-in-time, manual env parsing, flat module structure) are replaced entirely.
- **The existing test suite is not migrated wholesale.** New tests are written for the new architecture. Statistical validation concepts (KS uniformity, bias detection) carry over as test *strategies*, but the actual test code is written fresh.
- **Architectural invariants are re-evaluated.** Each invariant from the prototype is assessed: kept if it serves the new design, modified if it needs evolution, or dropped if it no longer applies.

### 2.3 What Changes

| Area | Prototype (qc-sampler) | New (qr-sampler) |
|------|------------------------|-------------------|
| Package name | `qc_sampler` / `qc-sampler` | `qr_sampler` / `qr-sampler` |
| Class prefix | `QC` / `QuantumConsciousness` | `QR` / `QuantumRandom` (or similar) |
| Config prefix | `qc_` env vars, `qc_` extra_args | `qr_` env vars, `qr_` extra_args |
| Config system | Frozen dataclass + manual env parsing | pydantic-settings `BaseSettings` with layered resolution |
| Layout | Flat `qc_sampler/` | `src/qr_sampler/` (src layout) |
| gRPC transport | Unary only, thread-based prefetch | Configurable: unary, server-streaming, or bidirectional streaming |
| Entropy sources | gRPC, os_urandom, mock_uniform | + `SystemEntropySource`, `TimingNoiseSource`, third-party via entry points |
| Entry-point plugin registry | Factory-internal registries only | `qr_sampler.entropy_sources` entry-point group for third-party discovery |
| Toolchain | setuptools, pytest | uv, ruff, mypy --strict, pytest + pytest-cov, pre-commit |
| CI/CD | None | GitHub Actions: test matrix, lint, type-check |
| Documentation | CLAUDE.md only | README.md, CONTRIBUTING.md, CHANGELOG.md, CODE_OF_CONDUCT.md, entropy source setup guide |
| License | None specified | Apache 2.0 |
| Entropy source onboarding | None | CLI scaffolding, reference servers, step-by-step guides, Docker examples |

---

## 3. Detailed Requirements

### 3.1 Core Sampling Pipeline

**REQ-PIPELINE-01**: The per-token sampling pipeline:
1. Compute temperature from logits (via TemperatureStrategy)
2. Fetch entropy bytes from source (just-in-time, after logits are available)
3. Amplify raw bytes into uniform float u in (0, 1) (via SignalAmplifier)
4. Select token from probability-ordered CDF using u (via TokenSelector)
5. Force one-hot logits so vLLM's downstream sampler picks exactly that token

**REQ-PIPELINE-02**: The one-hot forcing pattern (logits row = -inf except selected token = 0.0) is mandatory. vLLM-level temperature, top_k, and top_p must be set to pass-through values (1.0, -1, 1.0) by the user.

**REQ-PIPELINE-03**: Result types (`AmplificationResult`, `TemperatureResult`, `SelectionResult`, `TokenSamplingRecord`) should be immutable data carriers. The implementation choice (frozen dataclass, NamedTuple, frozen pydantic model) is an implementation decision, not a constraint.

### 3.2 Just-In-Time Entropy (CRITICAL)

**REQ-ENTROPY-01**: Entropy MUST be generated just-in-time for each token decision. The physical generation of random bits must occur AFTER the corresponding logits have been computed. No pre-generated pools, no buffering of entropy bytes ahead of when they are consumed. This is a hard constraint driven by the consciousness-interaction research model.

**REQ-ENTROPY-02**: Hardware preparation (e.g., laser diode activation, photodetector bias) MAY be signaled in advance as long as the actual random event (quantum state collapse, measurement) does not occur until the explicit generate request.

**REQ-ENTROPY-03**: All fallback-sourced entropy (e.g., os.urandom when QRNG is unavailable) must be flagged in metadata so downstream statistical analysis can distinguish or exclude non-QRNG tokens.

**REQ-ENTROPY-04**: A circuit breaker with adaptive timeout must be implemented. Track rolling P99 latency, set timeout to `max(5ms, P99 * 1.5)`, and fall back to secondary source when the primary source is slow.

### 3.3 gRPC Transport Modes (CONFIGURABLE)

**REQ-GRPC-01**: The gRPC entropy source must support three transport modes, selectable via configuration:

| Mode | Proto RPC | Use Case | Latency Profile |
|------|-----------|----------|-----------------|
| **Unary** | `GetEntropy(EntropyRequest) -> EntropyResponse` | Simple setups, HTTP/REST-like semantics, easiest to implement server-side | Higher per-request overhead (new HTTP/2 stream per call) |
| **Server-streaming** | `StreamEntropy(EntropyRequest) -> stream EntropyResponse` | Client sends one request, server streams responses as needed. Good for pre-configured byte counts. | Medium overhead, simpler than bidi |
| **Bidirectional streaming** | `StreamEntropy(stream EntropyRequest) -> stream EntropyResponse` | Persistent session, client sends "generate now" per token, server responds. Optimal for just-in-time. | Lowest overhead (~50-100us same-machine) |

**REQ-GRPC-02**: The configuration field `qr_grpc_mode` selects the transport mode. Valid values: `"unary"`, `"server_streaming"`, `"bidi_streaming"`. Default: `"unary"` (simplest for new users to get started).

**REQ-GRPC-03**: All three modes must satisfy the just-in-time constraint (REQ-ENTROPY-01). The entropy request is only sent after logits are available, regardless of transport mode. The transport mode affects *connection efficiency*, not *entropy freshness*.

**REQ-GRPC-04**: Bidirectional streaming should use `grpc.aio` (asyncio) for performance. Unary mode may use either sync or async gRPC.

**REQ-GRPC-05**: Channel configuration must support both TCP (`host:port`) and Unix domain sockets (`unix:///path/to/socket`) for co-located deployments. The address format is auto-detected from the configured server address string.

**REQ-GRPC-06**: The proto definition must support all three modes:
```proto
service EntropyService {
  rpc GetEntropy (EntropyRequest) returns (EntropyResponse);
  rpc StreamEntropy (stream EntropyRequest) returns (stream EntropyResponse);
}
```
The server-streaming mode reuses the bidirectional RPC definition but the client only sends one initial request.

### 3.4 Entropy Source Architecture

**REQ-SOURCE-01**: The `EntropySource` ABC defines the contract:
- `name` property — human-readable identifier
- `is_available` property — whether the source can currently provide entropy
- `get_random_bytes(n: int) -> bytes` — the primary interface, returns exactly n bytes or raises
- `get_random_float64(shape, out=None)` — zero-allocation hot path via pre-allocated `out` buffer
- `close()` — cleanup resources (channels, connections, files)

**REQ-SOURCE-02**: Built-in entropy source implementations:
- `QuantumGrpcSource`: gRPC client supporting all three transport modes (REQ-GRPC-01)
- `SystemEntropySource`: Wraps `os.urandom()` — always available, cryptographically secure, not quantum-random
- `TimingNoiseSource`: Derives entropy from CPU timing jitter (processor delays, instruction timing variations). Experimental/educational — not suitable for consciousness research but useful for testing and demonstration.
- `MockUniformSource`: Generates bytes from a configurable distribution for testing. Supports seeded reproducibility and configurable mean for bias simulation.

**REQ-SOURCE-03**: A `FallbackEntropySource` wrapper takes any primary source and any fallback source. Only `EntropyUnavailableError` triggers fallback — all other exceptions propagate. The wrapper reports which source was actually used in its return metadata.

**REQ-SOURCE-04**: Third-party entropy sources are discoverable via Python entry points under the group `qr_sampler.entropy_sources`:
```toml
[project.entry-points."qr_sampler.entropy_sources"]
lava_lamp = "qr_lava:LavaLampEntropySource"
```

**REQ-SOURCE-05**: An `EntropySourceRegistry` discovers entry-point sources at startup via `importlib.metadata.entry_points()` and lazily instantiates them on first use. Built-in sources register via the same entry-point mechanism in qr-sampler's own `pyproject.toml`.

### 3.5 Signal Amplification

**REQ-AMP-01**: The `ZScoreMeanAmplifier` converts N raw bytes into a single uniform float via: sample mean -> z-score -> normal CDF -> clamped uniform. This is the proven algorithm from the prototype.

**REQ-AMP-02**: SEM (standard error of mean) must be derived as `population_std / sqrt(actual_sample_count)` at amplification time, computed from the actual number of bytes received (not a config constant).

**REQ-AMP-03**: New amplification methods register via the factory/registry pattern and are selected by config string (`signal_amplifier_type`).

### 3.6 Temperature Strategies

**REQ-TEMP-01**: Two built-in temperature strategies:
- `FixedTemperatureStrategy`: Returns a constant temperature from config.
- `EDTTemperatureStrategy`: Entropy-based dynamic temperature that scales temperature based on Shannon entropy of the logit distribution.

**REQ-TEMP-02**: All temperature strategies must compute and return Shannon entropy regardless of whether the formula uses it (diagnostic logging depends on it).

**REQ-TEMP-03**: New temperature strategies register via the factory/registry pattern.

### 3.7 Configuration System

**REQ-CONFIG-01**: Use pydantic-settings `BaseSettings` for configuration. Layered resolution: init kwargs -> environment variables (prefix `QR_`) -> `.env` file -> defaults.

**REQ-CONFIG-02**: The env var prefix is `QR_`. Per-request overrides use `qr_` prefix in `SamplingParams.extra_args`.

**REQ-CONFIG-03**: Infrastructure fields (server address, timeout, retry count, fallback mode, gRPC transport mode) are NOT overridable per-request. Sampling parameters (temperature, top_k, top_p, sample_count, amplifier type, temperature strategy) ARE overridable per-request.

**REQ-CONFIG-04**: Every config field must have `Field(description=...)` for self-documentation. Type validation and coercion handled by pydantic.

**REQ-CONFIG-05**: The configuration must include the new `qr_grpc_mode` field for transport mode selection.

### 3.8 vLLM V1 Integration

**REQ-VLLM-01**: The processor must implement the batch-level `LogitsProcessor` ABC directly. Constructor signature: `__init__(self, vllm_config, device, is_pin_memory)`.

**REQ-VLLM-02**: `is_argmax_invariant()` must return `False` — this processor fundamentally changes token selection.

**REQ-VLLM-03**: `update_state(batch_update)` must process removed -> moved -> added in that order, following vLLM's contract.

**REQ-VLLM-04**: `validate_params(sampling_params)` must validate all `qr_*` extra_args keys before a request enters the batch.

**REQ-VLLM-05**: Registration via entry point:
```toml
[project.entry-points."vllm.logits_processors"]
qr_sampler = "qr_sampler.processor:QRSamplerLogitsProcessor"
```

### 3.9 Performance Requirements

**REQ-PERF-01**: Pre-allocate all tensors in `__init__()`. Never allocate in `apply()`.

**REQ-PERF-02**: Use in-place tensor modification in the hot path. Use `non_blocking=True` for CPU->GPU transfers with pinned memory.

**REQ-PERF-03**: Vectorize across the batch dimension where possible. Short-circuit when batch is empty.

**REQ-PERF-04**: For NumPy/PyTorch interop, use `torch.from_numpy()` / `torch.as_tensor()` for zero-copy CPU conversion. Never use `torch.tensor()` in the hot path.

**REQ-PERF-05**: Use `__slots__` on any per-token data objects to reduce memory footprint.

### 3.10 Project Structure

**REQ-STRUCT-01**: Use the src layout:
```
qr-sampler/
├── src/
│   └── qr_sampler/
│       ├── __init__.py
│       ├── py.typed              # PEP 561 marker
│       ├── processor.py          # vLLM LogitsProcessor
│       ├── config.py             # pydantic-settings BaseSettings
│       ├── exceptions.py         # Exception hierarchy
│       ├── entropy/
│       │   ├── __init__.py
│       │   ├── base.py           # EntropySource ABC
│       │   ├── registry.py       # Entry-point discovery registry
│       │   ├── quantum.py        # gRPC entropy source (all transport modes)
│       │   ├── system.py         # os.urandom source
│       │   ├── timing.py         # CPU timing jitter source
│       │   ├── mock.py           # Mock source for testing
│       │   └── fallback.py       # FallbackEntropySource wrapper
│       ├── amplification/
│       │   ├── __init__.py
│       │   ├── base.py           # SignalAmplifier ABC
│       │   ├── zscore.py         # Z-score mean amplifier
│       │   └── registry.py       # Amplifier registry
│       ├── temperature/
│       │   ├── __init__.py
│       │   ├── base.py           # TemperatureStrategy ABC
│       │   ├── fixed.py          # Fixed temperature
│       │   ├── edt.py            # Entropy-based dynamic temperature
│       │   └── registry.py       # Temperature strategy registry
│       ├── selection/
│       │   ├── __init__.py
│       │   ├── selector.py       # TokenSelector (CDF-based)
│       │   └── types.py          # SelectionResult dataclass
│       ├── logging/
│       │   ├── __init__.py
│       │   ├── logger.py         # SamplingLogger
│       │   └── types.py          # TokenSamplingRecord dataclass
│       └── proto/
│           ├── __init__.py
│           ├── entropy_service.proto
│           ├── entropy_service_pb2.py
│           └── entropy_service_pb2_grpc.py
├── examples/
│   ├── servers/
│   │   ├── simple_urandom_server.py    # Minimal reference: os.urandom over gRPC
│   │   ├── timing_noise_server.py      # CPU timing jitter gRPC server
│   │   └── qrng_template_server.py     # Template for real QRNG hardware
│   ├── docker/
│   │   ├── Dockerfile.entropy-server   # Containerized entropy server
│   │   └── docker-compose.yml          # vLLM + entropy server compose
│   └── systemd/
│       └── qr-entropy-server.service   # systemd unit file template
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_entropy/
│   │   ├── test_system.py
│   │   ├── test_quantum.py
│   │   ├── test_timing.py
│   │   ├── test_fallback.py
│   │   ├── test_mock.py
│   │   └── test_registry.py
│   ├── test_amplification/
│   │   └── test_zscore.py
│   ├── test_temperature/
│   │   ├── test_fixed.py
│   │   └── test_edt.py
│   ├── test_selection/
│   │   └── test_selector.py
│   ├── test_processor.py
│   └── test_statistical_properties.py
├── .github/
│   ├── workflows/
│   │   └── ci.yml                # test matrix, lint, type-check
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.yml
│   │   └── feature_request.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── dependabot.yml
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── LICENSE                       # Apache 2.0
├── CLAUDE.md
├── .pre-commit-config.yaml
└── .gitignore
```

### 3.11 Toolchain and CI/CD

**REQ-TOOL-01**: Package manager: `uv`. Build backend: `setuptools` (or `hatchling`). Version management: `setuptools-scm` from git tags.

**REQ-TOOL-02**: Linting and formatting: `ruff` (replaces flake8, black, isort, pyupgrade, autoflake).

**REQ-TOOL-03**: Type checking: `mypy --strict` on `src/`.

**REQ-TOOL-04**: Testing: `pytest` with `pytest-cov`. Coverage target >= 90%.

**REQ-TOOL-05**: Pre-commit hooks: ruff (lint + format), mypy, bandit (security), standard hooks (trailing whitespace, YAML/TOML validation).

**REQ-TOOL-06**: CI pipeline (GitHub Actions):
- Test matrix: Python 3.10, 3.11, 3.12, 3.13
- Jobs: `ruff check`, `ruff format --check`, `mypy src/`, `pytest -v --cov`
- Coverage upload to Codecov on latest Python only

**REQ-TOOL-07**: Optional dependency extras:
- `pip install qr-sampler[grpc]` — gRPC support (grpcio, protobuf)
- `pip install qr-sampler[dev]` — full dev environment (pytest, scipy, ruff, mypy, pre-commit)
- `pip install qr-sampler[docs]` — documentation building (mkdocs-material, if added later)

### 3.12 Documentation and README

**REQ-DOC-01**: `README.md` is the primary documentation artifact and must be comprehensive. Sections:

1. **Header / badges**: Project name, tagline, CI status, Python version, license badge
2. **What is qr-sampler?**: One-paragraph explanation of what it does and why
3. **Motivation**: The consciousness-research context — studying non-deterministic LLMs as consciousness-interactive entities. Written accessibly for both ML practitioners and consciousness researchers.
4. **How it works**: Architecture overview with ASCII pipeline diagram showing: logits -> temperature -> entropy fetch -> amplification -> CDF selection -> one-hot forcing
5. **Quick start**:
   - Install qr-sampler
   - Start a reference entropy server (one command)
   - Configure vLLM to use qr-sampler (env vars)
   - Run inference
6. **Configuration reference**: Complete table of all config fields, their env vars, defaults, descriptions, and whether they're per-request overridable
7. **gRPC transport modes**: Explanation of unary vs server-streaming vs bidirectional streaming, when to use each, and how to configure
8. **Entropy sources guide**:
   - Built-in sources (quantum gRPC, system, timing noise, mock)
   - How the fallback system works
   - How to choose between sources
9. **Setting up your own entropy source** (see REQ-DOC-06 below — this is the comprehensive guide)
10. **Signal amplification**: How z-score amplification works, what `sample_count` means, how consciousness bias detection works
11. **Temperature strategies**: Fixed vs EDT, when to use each
12. **Statistical analysis**: How to interpret u-value distributions, detect bias, validate experimental setups
13. **Plugin architecture**: How to write a third-party entropy source plugin (entry points)
14. **Contributing**: Link to CONTRIBUTING.md
15. **License**: Apache 2.0

**REQ-DOC-02**: `CONTRIBUTING.md` must cover:
- Development environment setup (uv, pre-commit)
- Coding conventions (Python 3.10+, type hints, Google-style docstrings, ruff rules)
- Testing approach (write tests alongside code, statistical validation patterns)
- PR process
- How to add new entropy sources / amplifiers / temperature strategies (step-by-step)

**REQ-DOC-03**: `CHANGELOG.md` following Keep a Changelog format.

**REQ-DOC-04**: `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1.

**REQ-DOC-05**: `CLAUDE.md` must be updated to reflect the new structure, naming, and conventions.

### 3.13 Entropy Source Setup Workflow (CRITICAL)

**REQ-DOC-06**: The README must include a dedicated, comprehensive section titled "Setting Up Your Own Entropy Source" that walks a user from zero to a working custom entropy server. This is one of the most important sections of the project.

The guide must include:

**A. Conceptual overview:**
- What the entropy server does (serves random bytes over gRPC)
- The just-in-time contract (generate bytes only when requested, not before)
- The proto contract (EntropyRequest/EntropyResponse message format)
- Which transport mode to choose for their use case

**B. Step-by-step: Minimal server (5 minutes):**
1. Copy the proto file from the repo
2. Install dependencies (`pip install grpcio grpcio-tools`)
3. Copy the `simple_urandom_server.py` reference implementation
4. Run it: `python simple_urandom_server.py`
5. Configure qr-sampler to point to it: `QR_GRPC_SERVER_ADDRESS=localhost:50051`
6. Verify with a test command

**C. Step-by-step: Custom QRNG hardware server:**
1. Start from `qrng_template_server.py` (annotated template with TODO markers)
2. Implement the `generate_entropy(byte_count) -> bytes` function with your hardware SDK
3. Choose transport mode (the template supports all three)
4. Test with the provided validation script
5. Deploy (systemd or Docker instructions)

**D. Reference server implementations** (shipped in `examples/servers/`):
- `simple_urandom_server.py`: 50-line minimal server using `os.urandom`. Implements both unary `GetEntropy` and streaming `StreamEntropy`. Fully commented. This is the "hello world" of entropy servers.
- `timing_noise_server.py`: Server that derives entropy from CPU timing jitter. Educational example showing a non-QRNG source.
- `qrng_template_server.py`: Annotated template with clear `# TODO: YOUR HARDWARE CODE HERE` markers. Includes error handling, health checks, and logging. Supports all three transport modes.

**E. Deployment automation:**
- `examples/docker/Dockerfile.entropy-server`: Single-stage Dockerfile that runs any of the example servers
- `examples/docker/docker-compose.yml`: Full stack: entropy server + vLLM with qr-sampler configured, ready to `docker compose up`
- `examples/systemd/qr-entropy-server.service`: systemd unit file for running the entropy server as a daemon, with restart-on-failure and logging

**F. Validation and debugging:**
- How to verify your server is working (test client script or command)
- Common issues and troubleshooting (port conflicts, proto version mismatch, timeout tuning)
- How to check entropy quality (run the statistical validation tests against your source)

**G. Advanced topics:**
- Unix domain sockets for same-machine deployments (lower latency)
- TLS for cross-network deployments
- Monitoring and health checks
- Tuning `sample_count` and timeout for your hardware's latency profile

### 3.14 Licensing

**REQ-LICENSE-01**: Apache 2.0 license. Matches vLLM's license and includes explicit patent grant.

---

## 4. Architectural Principles

These are the design principles for the new architecture. They are informed by the prototype but not bound to its exact formulations. Each principle serves the new design's goals.

1. **Configuration drives construction.** All numeric/behavioral constants trace to config fields. Mathematical constants (sqrt(2), erf coefficients) are the sole exception. New parameters get config fields, env var mappings, and per-request override support as appropriate.

2. **Registry pattern for extensible components.** Entropy sources, signal amplifiers, and temperature strategies are selected by string key from a registry. No if/else chains. Third-party entropy sources additionally register via entry points.

3. **ABCs define contracts.** The processor and factory reference only abstract types. Concrete implementations are discovered through registries.

4. **Immutable result types.** Pipeline result objects (amplification, temperature, selection, sampling record) are immutable to prevent accidental mutation in the pipeline.

5. **Per-request config resolution creates new instances.** The default config is never mutated. Infrastructure fields are non-overridable per-request.

6. **One-hot logits forcing.** After token selection, the entire logit row = -inf except selected token = 0.0. This is the mechanism that makes qr-sampler's token choice authoritative.

7. **Structured logging only.** `logging.getLogger("qr_sampler")` for all output. No `print()` statements. Per-token diagnostics go through `SamplingLogger`.

8. **Just-in-time entropy, no pre-buffering.** Entropy bytes are requested only after logits are available. No prefetch pools, no background generation ahead of consumption.

9. **Fallback entropy is always flagged.** When a fallback source is used, metadata marks the token so researchers can exclude it from analysis.

10. **Transport mode is a user choice.** The gRPC client supports unary, server-streaming, and bidirectional streaming. The user picks based on their server capabilities and latency requirements.

---

## 5. Design Decisions and Assumptions

### 5.1 Decisions Made

**D1: Rename from `qc` to `qr` prefix throughout.** The task explicitly requires `qr-sampler` naming with `qr` (quantum-random) prefixes. Full rename of the Python package, config prefixes, entry points, and class names.

**D2: Ground-up redesign, not a migration.** The prototype informs but does not constrain. Algorithms that work are reimplemented cleanly. Patterns that don't fit the new goals are dropped. Tests are written fresh for the new architecture.

**D3: Three gRPC transport modes.** Different users have different server capabilities. A simple unary server is trivial to implement (good for getting started). Bidirectional streaming is optimal for production (lowest latency). Server-streaming is a middle ground. All three satisfy the just-in-time constraint because the request is only sent after logits are ready. Default is `"unary"` for ease of onboarding.

**D4: pydantic-settings for config.** Replaces manual env-var parsing with declarative field definitions. The `QR_` prefix is handled automatically. `.env` file support comes free.

**D5: Entropy source entry-point registry.** Third-party sources use `qr_sampler.entropy_sources`. Amplifiers and temperature strategies use in-package registries (expandable later).

**D6: Apache 2.0 license.** Matches vLLM, includes patent grant.

**D7: Example servers and automation are first-class deliverables.** The `examples/` directory with reference servers, Docker, and systemd templates is as important as the library code. Making it easy to set up an entropy source is a core goal.

**D8: No trusted PyPI publishing.** Manual publishing is sufficient. CI covers testing, linting, and type-checking. Publishing automation can be added later if needed.

### 5.2 Assumptions

**A1**: Users will set vLLM-level sampling parameters to pass-through values (temperature=1.0, top_k=-1, top_p=1.0) when using qr-sampler. The README will document this prominently.

**A2**: The gRPC entropy server is provided separately by the user. qr-sampler ships reference implementations in `examples/` but does not include a production server in the package.

**A3**: Python 3.10 is the minimum supported version, matching vLLM V1 requirements.

**A4**: The `TimingNoiseSource` is experimental/educational. It is NOT suitable for consciousness-research experiments (timing noise is not quantum-random) but is useful for testing and as an example of a non-gRPC entropy source.

**A5**: Statistical validation concepts from the prototype (KS uniformity test, bias detection, EDT monotonicity) are the right test strategies for the new codebase, but the test implementations are written fresh.

---

## 6. Success Criteria

1. `pytest -v --cov` passes with >= 90% coverage on `src/qr_sampler/`
2. `ruff check` and `ruff format --check` pass with zero violations
3. `mypy --strict src/` passes with zero errors
4. The plugin loads in vLLM via the entry point and processes tokens correctly
5. All three gRPC transport modes work with the reference entropy server
6. A new user can follow the README to go from zero to working system in under 15 minutes using the reference server
7. A new user can follow the "Setting Up Your Own Entropy Source" guide to create a custom server from the template
8. `pip install -e .` works and `pip install -e ".[dev]"` installs all dev dependencies
9. The `docker-compose.yml` example brings up a working entropy-server + vLLM stack with `docker compose up`

---

## 7. Scope Summary

### In Scope (This Release)

- New `qr-sampler` package with src layout, designed from first principles
- pydantic-settings configuration system
- gRPC entropy source with three configurable transport modes (unary, server-streaming, bidirectional)
- Circuit breaker with adaptive timeout
- Built-in sources: QuantumGrpcSource, SystemEntropySource, TimingNoiseSource, MockUniformSource
- FallbackEntropySource wrapper with metadata flagging
- Entry-point based entropy source plugin registry
- Z-score mean signal amplifier
- Fixed and EDT temperature strategies
- CDF-based token selector with one-hot logits forcing
- Per-token diagnostic logging with SamplingLogger
- Modern toolchain (uv, ruff, mypy, pre-commit)
- GitHub Actions CI (test matrix, lint, type-check)
- Apache 2.0 license
- Comprehensive README with full entropy source setup guide
- Reference entropy server implementations in `examples/servers/`
- Docker and systemd deployment templates in `examples/`
- CONTRIBUTING.md, CHANGELOG.md, CODE_OF_CONDUCT.md
- Updated CLAUDE.md
- Fresh test suite with statistical validation

### Out of Scope (Future)

- MkDocs documentation site (README is sufficient for launch)
- Hurst exponent amplifier (mentioned in guiding doc, can be added post-launch)
- Trusted PyPI publishing automation
- Web UI / dashboard
- QRNG hardware drivers
- vLLM V0 support
- Multi-GPU / distributed entropy coordination
- Formal benchmarking suite
- CLI scaffolding tool for generating server boilerplate (consider for v2 — the templates serve this role for now)
