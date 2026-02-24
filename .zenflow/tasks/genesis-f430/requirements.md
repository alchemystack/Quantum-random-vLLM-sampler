# Product Requirements Document: Quantum Consciousness Token Sampling for vLLM V1

## 1. Overview

### 1.1 Product Summary

A vLLM V1 LogitsProcessor plugin (`qc-sampler`) that replaces standard pseudorandom token sampling with quantum-random sampling. The plugin enables consciousness to influence token selection through quantum random number generators (QRNGs), based on the theory that consciousness can bias quantum-random outcomes.

### 1.2 Problem Statement

Standard LLM token sampling relies on pseudorandom number generators (PRNGs), which are deterministic and cannot be influenced by external factors. For research into consciousness-influenced text generation, a truly random — specifically quantum-random — source of randomness is needed, along with a signal amplification mechanism that translates subtle biases in quantum randomness into coherent shifts in token selection.

### 1.3 Target Users

- Researchers studying consciousness interaction with quantum systems
- Operators running vLLM inference servers who want to integrate QRNG-based sampling
- Developers building applications on top of vLLM that require quantum-random token selection

### 1.4 Success Criteria

1. The plugin integrates seamlessly with vLLM V1 as a standard LogitsProcessor plugin (no vLLM source modifications).
2. Under the null hypothesis (no consciousness influence), token selection is statistically indistinguishable from standard sampling — the u-value distribution is uniform.
3. A consciousness-induced bias in the QRNG output produces a coherent, directional shift in token selection (toward more or less probable tokens).
4. The system degrades gracefully when the QRNG server is unavailable.
5. All parameters are configurable per-request without server restart.

---

## 2. Functional Requirements

### FR-1: Quantum Random Byte Acquisition

**Description:** The system must acquire raw quantum random bytes from a remote QRNG hardware device via gRPC for each token sampling decision.

**Requirements:**
- FR-1.1: Connect to a remote QRNG gRPC server using a configurable address.
- FR-1.2: Request a configurable number of raw bytes per token (default: 20,480 bytes).
- FR-1.3: Support prefetching — begin fetching bytes for the next token during the GPU forward pass to minimize latency.
- FR-1.4: Implement configurable retry logic before falling back.
- FR-1.5: Support configurable fallback modes when the QRNG server is unreachable:
  - `"error"`: raise an error (fail-fast)
  - `"os_urandom"`: use operating system CSPRNG as fallback
  - `"mock_uniform"`: use a deterministic mock source (testing only)
- FR-1.6: Log which entropy source was used for each token (primary vs. fallback).

### FR-2: Signal Amplification

**Description:** The system must convert a large sample of raw QRNG bytes into a single uniform float in (0, 1) that amplifies any subtle bias present in the quantum source.

**Requirements:**
- FR-2.1: Implement a z-score-based mean amplification algorithm as the default.
- FR-2.2: The algorithm must: interpret bytes as uint8, compute sample mean, compute z-score using population parameters, convert to uniform via the normal CDF, and clamp to avoid exact 0 or 1.
- FR-2.3: All population parameters (mean, std) must be configurable, not hardcoded.
- FR-2.4: The standard error of the mean must be derived from `population_std / sqrt(sample_count)`, never stored as a separate config value.
- FR-2.5: The system must be extensible to support alternative amplification algorithms in the future without modifying existing code.
- FR-2.6: Return diagnostic data (sample mean, z-score, SEM) alongside the result for logging and analysis.

### FR-3: Temperature Computation

**Description:** The system must compute a per-token sampling temperature. Two strategies are required at launch.

**Requirements:**
- FR-3.1: **Fixed temperature** — use a single configurable temperature value for all tokens.
- FR-3.2: **Entropy-based dynamic temperature (EDT)** — modulate temperature based on Shannon entropy of the logit distribution:
  - Compute Shannon entropy of softmax(logits).
  - Normalize entropy to [0, 1] by dividing by log(vocab_size).
  - Apply formula: `T = base_temp * (H_norm ^ exponent)`, clamped to [min_temp, max_temp].
- FR-3.3: All EDT parameters (base_temp, exponent, min_temp, max_temp) must be configurable per-request.
- FR-3.4: The temperature strategy must be selectable per-request.
- FR-3.5: Always compute and report Shannon entropy regardless of strategy (for logging).
- FR-3.6: The system must be extensible to support new temperature strategies without modifying existing code.

### FR-4: Token Selection

**Description:** The system must select a token from the logit distribution using the QRNG-derived uniform float and a probability-ordered CDF.

**Requirements:**
- FR-4.1: Apply temperature scaling to logits.
- FR-4.2: Apply configurable top-k filtering (keep only the k highest-logit tokens).
- FR-4.3: Apply configurable top-p (nucleus) filtering (keep the minimal set of tokens whose cumulative probability exceeds the threshold).
- FR-4.4: Build a CDF over tokens sorted by **descending probability** so that u near 0 selects the most likely token and u near 1 selects increasingly unlikely tokens.
- FR-4.5: Use the uniform float from signal amplification to index into the CDF and select the token.
- FR-4.6: Return metadata: selected token ID, rank in probability order, probability, and candidate set size.
- FR-4.7: Handle edge cases robustly: all-identical logits, single surviving token after filtering, empty candidate sets.

### FR-5: Per-Request Configuration

**Description:** Every algorithmic parameter must be overridable per-request through vLLM's `extra_args` mechanism.

**Requirements:**
- FR-5.1: Per-request overrides use `extra_args` keys with a `qc_` prefix mapping to config field names.
- FR-5.2: Default values are loaded from environment variables (with `QC_` prefix, uppercase) at startup.
- FR-5.3: Per-request values take precedence over defaults.
- FR-5.4: Infrastructure fields (e.g., `qrng_server_address`) are not overridable per-request.
- FR-5.5: All overrides must be type-checked and range-validated with clear error messages.

### FR-6: Structured Logging and Diagnostics

**Description:** The system must provide structured per-token logging for monitoring, debugging, and offline analysis.

**Requirements:**
- FR-6.1: Three log levels: `"none"` (no logging), `"summary"` (one line per token with key metrics), `"full"` (complete diagnostic dump).
- FR-6.2: A diagnostic mode that stores all per-token data in memory for programmatic access.
- FR-6.3: Each token record must include: timestamps, QRNG latency, entropy source used, signal amplification intermediates (sample mean, z-score, u-value), temperature strategy and value, Shannon entropy, selected token details (ID, rank, probability, candidate count), and a config hash.
- FR-6.4: Aggregate summary statistics must be available: average entropy, average temperature, u-value distribution summary, average QRNG latency.

### FR-7: vLLM V1 Integration

**Description:** The system must integrate with vLLM V1 as a standard LogitsProcessor plugin.

**Requirements:**
- FR-7.1: Register via `pyproject.toml` entry point under `vllm.logits_processors`.
- FR-7.2: Implement the full vLLM V1 LogitsProcessor interface: `__init__`, `apply`, `update_state`, `validate_params`, `is_argmax_invariant`.
- FR-7.3: Handle batch processing — process each request in a batch independently with its own resolved configuration.
- FR-7.4: Force vLLM to accept the plugin's token choice by setting all logits to -inf except the selected token at 0.0.
- FR-7.5: Users must set vLLM-level sampling to pass-through (`temperature=1.0, top_k=-1, top_p=1.0`) since the plugin handles all sampling internally.
- FR-7.6: Clean up per-request state when requests are removed from the batch.

---

## 3. Non-Functional Requirements

### NFR-1: Modularity

- Each concern (entropy acquisition, signal amplification, temperature computation, token selection, logging) must be a separate module with clean abstract interfaces.
- Swapping any single component must not require changes to other components.
- Use the strategy pattern and dependency injection throughout.

### NFR-2: No Hardcoded Values

- Every constant, threshold, formula parameter, and default must reside in the central configuration dataclass.
- No magic numbers anywhere in algorithmic code.

### NFR-3: Robustness

- Graceful degradation when the QRNG server is unreachable (configurable fallback).
- Comprehensive error handling with clear, actionable error messages.
- No silent failures — every error path must log or raise.
- Custom exception hierarchy for structured error handling.

### NFR-4: Maintainability

- Every public class and method must have a Google-style docstring explaining not just what but why.
- Someone unfamiliar with the codebase should be able to understand any module in isolation.
- Follow Python 3.10+ conventions (`X | Y` unions, type hints on all signatures).

### NFR-5: Extensibility

- New signal amplification algorithms, entropy source backends, and temperature formulas must be addable without modifying existing code.
- Registry-based factory pattern for component construction.

### NFR-6: Troubleshootability

- Structured logging with per-token metadata.
- Diagnostic mode that records every intermediate value for offline analysis.
- Each component returns diagnostic dicts alongside results.

### NFR-7: Performance

- Use `np.partition` for top-k filtering (O(n) average) rather than full sorts.
- Prefetch QRNG bytes during GPU forward pass to hide network latency.
- Softmax with numerical stability (shift by max before exp).

---

## 4. Architecture Constraints

### AC-1: Plugin-Only

The system is a vLLM plugin. No modifications to vLLM core source code are permitted.

### AC-2: Python 3.10+

Use modern Python syntax: `X | Y` union types, dataclasses, type hints on all function signatures.

### AC-3: gRPC Client Only

The QRNG gRPC server is a separate system. This plugin only implements the client side. The proto stubs are considered pre-generated and included in the package.

### AC-4: Dependencies

Minimal dependency footprint: `grpcio>=1.60.0`, `numpy>=1.24.0`, plus Python standard library. No additional third-party libraries unless strictly necessary.

---

## 5. Scope Boundaries

### In Scope

- The `qc-sampler` Python package with all modules described above.
- gRPC client for the QRNG entropy service.
- Proto definition file and pre-generated Python stubs.
- Unit tests for each module.
- Integration tests with mock entropy sources.
- Statistical property tests verifying uniformity and bias detection.
- `pyproject.toml` with vLLM plugin entry point.

### Out of Scope

- The QRNG server daemon (separate codebase).
- Modal/RunPod deployment configuration.
- The gRPC proto compilation step (stubs are included pre-generated).
- Any modifications to vLLM core source code.
- Web API layer on top of vLLM.
- Frontend/UI for monitoring or configuration.

---

## 6. Key Design Decisions and Assumptions

### D-1: Descending-Probability CDF Ordering

The CDF is built over tokens sorted by descending probability. This is a deliberate design choice: it gives any bias in the QRNG output a coherent semantic direction. A u-value biased toward 0 selects expected/safe tokens; biased toward 1 selects surprising/creative tokens. This is fundamental to the consciousness-influence hypothesis.

### D-2: Signal Amplification via Z-Score

The z-score mean amplification algorithm is chosen because it leverages the Central Limit Theorem — averaging 20,480 bytes makes the sample mean approximately normally distributed, and the probability integral transform converts this to a uniform float. Even tiny per-byte biases (e.g., 0.003) accumulate into detectable u-value shifts.

### D-3: EDT Formula

The entropy-based dynamic temperature formula `T = base_temp * (H_norm ^ exponent)` is a sensible default inspired by the EDT paper (arXiv:2403.14541) and llama.cpp's DynaTemp. It is not the only valid formula — the system is designed so that alternative formulas can be added as new `TemperatureStrategy` subclasses without modifying existing code.

### D-4: Per-Request State via Index-Based Mapping

The processor maps per-request configurations using batch indices (as provided by vLLM's `update_state` mechanism). This aligns with vLLM V1's batching model where each row in the logits tensor corresponds to a specific request.

### D-5: Prefetch During Forward Pass

The QRNG bytes for the next token are prefetched during the GPU forward pass. This hides network latency behind GPU computation. The prefetch is triggered in `update_state`, which vLLM calls before the forward pass.

### D-6: Thread Safety for Prefetch

The prefetch buffer in `GrpcEntropySource` uses a simple lock (not a complex concurrent data structure). This is sufficient because there is exactly one producer (the prefetch thread) and one consumer (the `get_bytes` call on the main thread), and contention is minimal.

### D-7: Fallback as Wrapper, Not Inheritance

`FallbackEntropySource` wraps a primary source rather than subclassing it. This keeps each entropy source implementation pure and composable — any source can be the primary or the fallback, and nesting is possible.

### D-8: Config Hash for Reproducibility

Each `TokenSamplingRecord` includes a hash of the resolved config used for that token. This enables offline analysis to correlate sampling behavior with exact configuration, even when per-request overrides vary across a batch.

### D-9: Environment Variable Loading

Default config values are loaded from environment variables at processor initialization (not at module import time). This ensures that environment changes between process restarts are picked up, and avoids side effects during import.

### D-10: Proto Stubs Included

The gRPC proto stubs (`entropy_service_pb2.py`, `entropy_service_pb2_grpc.py`) are included in the repository as pre-generated files. The proto definition file is also included for reference, but compilation is not part of the build process for this package.

---

## 7. Overridable Parameters Reference

The following parameters are configurable via environment variables (defaults) and per-request `extra_args` overrides:

| Config Field | Env Var | Default | Per-Request (`qc_` prefix) | Type |
|---|---|---|---|---|
| `qrng_server_address` | `QC_QRNG_SERVER_ADDRESS` | `"10.0.1.5:50051"` | No (infrastructure) | str |
| `qrng_timeout_ms` | `QC_QRNG_TIMEOUT_MS` | `5000` | No (infrastructure) | float |
| `qrng_prefetch_enabled` | `QC_QRNG_PREFETCH_ENABLED` | `true` | No (infrastructure) | bool |
| `qrng_retry_count` | `QC_QRNG_RETRY_COUNT` | `2` | No (infrastructure) | int |
| `qrng_fallback_mode` | `QC_QRNG_FALLBACK_MODE` | `"os_urandom"` | No (infrastructure) | str |
| `signal_amplifier_type` | `QC_SIGNAL_AMPLIFIER_TYPE` | `"zscore_mean"` | Yes | str |
| `sample_count` | `QC_SAMPLE_COUNT` | `20480` | Yes | int |
| `population_mean` | `QC_POPULATION_MEAN` | `127.5` | Yes | float |
| `population_std` | `QC_POPULATION_STD` | `73.6116` | Yes | float |
| `uniform_clamp_epsilon` | `QC_UNIFORM_CLAMP_EPSILON` | `1e-10` | Yes | float |
| `temperature_strategy` | `QC_TEMPERATURE_STRATEGY` | `"fixed"` | Yes | str |
| `fixed_temperature` | `QC_FIXED_TEMPERATURE` | `0.7` | Yes | float |
| `edt_min_temp` | `QC_EDT_MIN_TEMP` | `0.1` | Yes | float |
| `edt_max_temp` | `QC_EDT_MAX_TEMP` | `2.0` | Yes | float |
| `edt_base_temp` | `QC_EDT_BASE_TEMP` | `0.8` | Yes | float |
| `edt_exponent` | `QC_EDT_EXPONENT` | `0.5` | Yes | float |
| `top_k` | `QC_TOP_K` | `50` | Yes | int |
| `top_p` | `QC_TOP_P` | `0.9` | Yes | float |
| `log_level` | `QC_LOG_LEVEL` | `"summary"` | Yes | str |
| `diagnostic_mode` | `QC_DIAGNOSTIC_MODE` | `false` | Yes | bool |

---

## 8. Testing Requirements

### 8.1 Unit Tests

Each module must have isolated unit tests using dependency injection (mock objects). No test may require a real QRNG server or GPU.

- **Config:** Verify merge logic, type validation, unknown-key handling.
- **Signal Amplifier:** Verify z-score computation for known inputs, verify SEM is derived (not hardcoded).
- **Temperature Strategy:** Verify fixed strategy returns constant, verify EDT monotonicity, verify clamping.
- **Token Selector:** Verify correct token selection for known distributions and u-values, verify top-k/top-p filtering, verify edge cases.

### 8.2 Integration Tests

- Instantiate the full `QuantumConsciousnessProcessor` with `MockUniformSource`.
- Verify it produces valid modified logits (exactly one finite value per row).
- Verify selected tokens are valid vocab indices.
- Verify token rank distribution is roughly exponential under default settings.

### 8.3 Statistical Property Tests

- Verify u-value uniformity under null hypothesis (KS test, p > 0.01, 10,000 samples).
- Verify consciousness bias simulation produces shifted u-value mean.
- Verify EDT temperature correlates with entropy.
