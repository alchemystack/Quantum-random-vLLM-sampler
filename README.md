# qr-sampler

**Plug any randomness source into LLM token sampling via vLLM.**

qr-sampler is a [vLLM V1](https://github.com/vllm-project/vllm) LogitsProcessor plugin that replaces standard pseudorandom token sampling with entropy from external sources — quantum random number generators (QRNGs), processor timing jitter, or any hardware you connect via gRPC. It is designed for researchers studying non-deterministic LLM behavior and the potential influence of physical randomness on language model outputs.

```
pip install qr-sampler
```

---

## Why qr-sampler?

Standard LLM inference uses pseudorandom number generators (PRNGs) for token sampling. PRNGs are deterministic — given the same seed, they produce the same output every time. qr-sampler replaces this with *true* randomness from physical processes:

- **Quantum RNGs** — photon detectors, vacuum fluctuation devices, or any hardware QRNG over gRPC
- **Processor timing jitter** — CPU clock variations as an entropy source (experimental)
- **OS entropy** — `/dev/urandom` as a fallback or baseline
- **Your own source** — implement the `EntropySource` ABC or connect any hardware via the gRPC protocol

### Consciousness-research context

qr-sampler provides infrastructure for studying whether conscious intent can influence quantum-random processes in LLM token selection. The signal amplification system converts thousands of random bytes into a single token choice, designed so that even a tiny statistical bias (e.g., 0.1% shift in byte means) produces a measurable effect on which token gets selected. All entropy is generated **just-in-time** — the quantum measurement happens *after* logits are computed, never before.

This is a research tool. It makes no claims about consciousness or quantum mechanics — it provides the infrastructure to run rigorous experiments.

---

## How it works

```
Logits from vLLM (one row per batch request)
  │
  ├─ Temperature strategy ─────── Compute per-token temperature
  │   (fixed or entropy-dependent)    from the logit distribution
  │
  ├─ Entropy source ───────────── Fetch fresh random bytes
  │   (gRPC QRNG / system / timing)   just-in-time, after logits exist
  │
  ├─ Signal amplification ─────── Convert 20,480 bytes → one float u ∈ (0,1)
  │   (z-score → normal CDF)         via statistical aggregation
  │
  ├─ Token selector ───────────── top-k → softmax → top-p → CDF → select
  │   (CDF binary search with u)     token from probability distribution
  │
  └─ Force one-hot logits ─────── Set selected token to 0.0, all others to -inf
      (vLLM picks exactly this token)
```

The processor registers via Python entry points — no vLLM source code modifications needed.

---

## Quick start

### 1. Install qr-sampler

```bash
# Core (uses os.urandom fallback by default)
pip install qr-sampler

# With gRPC support (required for external entropy servers)
pip install qr-sampler[grpc]
```

### 2. Start an entropy server

The fastest way to get running — a reference server using `os.urandom()`:

```bash
# Clone and run the example server
git clone https://github.com/alchemystack/Quantum-random-vLLM-sampler.git
cd Quantum-random-vLLM-sampler
pip install qr-sampler[grpc]

python examples/servers/simple_urandom_server.py
# Output: Entropy server listening on 0.0.0.0:50051
```

### 3. Configure vLLM to use qr-sampler

**Windows (CMD):**

```cmd
set QR_ENTROPY_SOURCE_TYPE=quantum_grpc
set QR_GRPC_SERVER_ADDRESS=localhost:50051
set QR_GRPC_MODE=unary

:: Start vLLM with the plugin
vllm serve meta-llama/Llama-3.2-1B
```

**macOS / Linux:**

```bash
export QR_ENTROPY_SOURCE_TYPE=quantum_grpc
export QR_GRPC_SERVER_ADDRESS=localhost:50051
export QR_GRPC_MODE=unary

# Start vLLM with the plugin
vllm serve meta-llama/Llama-3.2-1B
```

qr-sampler registers automatically via the `vllm.logits_processors` entry point. Every token sampled by vLLM now uses entropy from your server.

### 4. Send inference requests

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B",
    "prompt": "The nature of consciousness is",
    "max_tokens": 100
  }'
```

Per-request parameter overrides via `extra_args`:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B",
    "prompt": "The nature of consciousness is",
    "max_tokens": 100,
    "extra_args": {
      "qr_temperature_strategy": "edt",
      "qr_top_k": 100,
      "qr_top_p": 0.95,
      "qr_diagnostic_mode": true
    }
  }'
```

### Docker quick start

```bash
cd examples/docker
docker compose up
```

This starts both the entropy server and vLLM with qr-sampler pre-configured. See [examples/docker/docker-compose.yml](examples/docker/docker-compose.yml) for configuration options.

---

## Configuration reference

All configuration is done via environment variables with the `QR_` prefix. Per-request overrides use the `qr_` prefix in `extra_args`.

### Infrastructure fields (NOT per-request overridable)

| Environment variable | Default | Description |
|---|---|---|
| `QR_GRPC_SERVER_ADDRESS` | `localhost:50051` | gRPC entropy server address (`host:port` or `unix:///path`) |
| `QR_GRPC_TIMEOUT_MS` | `5000` | gRPC call timeout in milliseconds |
| `QR_GRPC_RETRY_COUNT` | `2` | Retry attempts after gRPC failure |
| `QR_GRPC_MODE` | `unary` | Transport mode: `unary`, `server_streaming`, `bidi_streaming` |
| `QR_FALLBACK_MODE` | `system` | Fallback when primary fails: `error`, `system`, `mock_uniform` |
| `QR_ENTROPY_SOURCE_TYPE` | `quantum_grpc` | Primary entropy source identifier |
| `QR_CB_WINDOW_SIZE` | `100` | Rolling latency window size for P99 computation |
| `QR_CB_MIN_TIMEOUT_MS` | `5.0` | Minimum adaptive timeout in milliseconds |
| `QR_CB_TIMEOUT_MULTIPLIER` | `1.5` | Multiplier applied to P99 latency for adaptive timeout |
| `QR_CB_RECOVERY_WINDOW_S` | `10.0` | Seconds before half-open retry after circuit opens |
| `QR_CB_MAX_CONSECUTIVE_FAILURES` | `3` | Consecutive failures before circuit breaker opens |

### Sampling parameters (per-request overridable)

| Environment variable | extra_args key | Default | Description |
|---|---|---|---|
| `QR_SIGNAL_AMPLIFIER_TYPE` | `qr_signal_amplifier_type` | `zscore_mean` | Signal amplification algorithm |
| `QR_SAMPLE_COUNT` | `qr_sample_count` | `20480` | Entropy bytes fetched per token |
| `QR_POPULATION_MEAN` | `qr_population_mean` | `127.5` | Null-hypothesis mean for byte values |
| `QR_POPULATION_STD` | `qr_population_std` | `73.612...` | Population std for uniform [0, 255] |
| `QR_UNIFORM_CLAMP_EPSILON` | `qr_uniform_clamp_epsilon` | `1e-10` | Clamp u to avoid degenerate CDF |
| `QR_TEMPERATURE_STRATEGY` | `qr_temperature_strategy` | `fixed` | Strategy: `fixed` or `edt` |
| `QR_FIXED_TEMPERATURE` | `qr_fixed_temperature` | `0.7` | Constant temperature (fixed strategy) |
| `QR_EDT_BASE_TEMP` | `qr_edt_base_temp` | `0.8` | Base coefficient for EDT |
| `QR_EDT_EXPONENT` | `qr_edt_exponent` | `0.5` | Power-law exponent for EDT |
| `QR_EDT_MIN_TEMP` | `qr_edt_min_temp` | `0.1` | EDT temperature floor |
| `QR_EDT_MAX_TEMP` | `qr_edt_max_temp` | `2.0` | EDT temperature ceiling |
| `QR_TOP_K` | `qr_top_k` | `50` | Top-k filtering (`<=0` disables) |
| `QR_TOP_P` | `qr_top_p` | `0.9` | Nucleus sampling threshold (`1.0` disables) |
| `QR_LOG_LEVEL` | `qr_log_level` | `summary` | Logging: `none`, `summary`, `full` |
| `QR_DIAGNOSTIC_MODE` | `qr_diagnostic_mode` | `false` | Store all token records in memory |

You can also use a `.env` file — pydantic-settings loads it automatically.

---

## gRPC transport modes

qr-sampler supports three gRPC transport modes for communicating with entropy servers. All modes satisfy the just-in-time constraint — entropy is generated only when requested.

| Mode | `QR_GRPC_MODE` | Latency | Best for |
|---|---|---|---|
| **Unary** | `unary` | ~1-2ms overhead per call | Simplicity, debugging, low-frequency sampling |
| **Server streaming** | `server_streaming` | ~0.5-1ms | Middle ground |
| **Bidirectional** | `bidi_streaming` | ~50-100us (same machine) | Production, lowest latency |

For co-located hardware, use Unix domain sockets for the lowest possible latency:

**Windows (CMD):**

```cmd
:: Server
python simple_urandom_server.py --address unix:///var/run/qrng.sock

:: Client config
set QR_GRPC_SERVER_ADDRESS=unix:///var/run/qrng.sock
set QR_GRPC_MODE=bidi_streaming
```

**macOS / Linux:**

```bash
# Server
python simple_urandom_server.py --address unix:///var/run/qrng.sock

# Client config
export QR_GRPC_SERVER_ADDRESS=unix:///var/run/qrng.sock
export QR_GRPC_MODE=bidi_streaming
```

### Circuit breaker

The gRPC client includes an adaptive circuit breaker (all thresholds configurable via `QR_CB_*` environment variables):

- Tracks rolling P99 latency over the last `QR_CB_WINDOW_SIZE` requests (default: 100)
- Sets timeout to `max(QR_CB_MIN_TIMEOUT_MS, P99 * QR_CB_TIMEOUT_MULTIPLIER)` or the configured timeout, whichever is lower
- Opens the circuit after `QR_CB_MAX_CONSECUTIVE_FAILURES` consecutive failures (default: 3)
- Enters half-open state after `QR_CB_RECOVERY_WINDOW_S` seconds (default: 10), allowing one test request
- Falls back to the configured fallback source (`QR_FALLBACK_MODE`) when the circuit is open

All fallback-sourced entropy is flagged in diagnostic logs so downstream analysis can account for it.

---

## Entropy sources

### Built-in sources

| Source | Identifier | Description |
|---|---|---|
| **Quantum gRPC** | `quantum_grpc` | Remote QRNG via gRPC (default) |
| **System** | `system` | `os.urandom()` — OS cryptographic RNG |
| **Timing noise** | `timing_noise` | CPU timing jitter (experimental) |
| **Mock uniform** | `mock_uniform` | Configurable test source with seed/bias |

### Fallback behavior

The `FallbackEntropySource` wraps a primary source with an automatic fallback:

- Only catches `EntropyUnavailableError` — other exceptions propagate
- Logs a warning when fallback is used
- Exposes `last_source_used` for diagnostics

Configure with `QR_FALLBACK_MODE`:
- `system` — fall back to `os.urandom()` (default)
- `mock_uniform` — fall back to the mock source
- `error` — raise immediately, no fallback

### Third-party entropy sources

Any Python package can register entropy sources via entry points:

```toml
# In your package's pyproject.toml
[project.entry-points."qr_sampler.entropy_sources"]
lava_lamp = "my_package:LavaLampEntropySource"
```

The source will be auto-discovered when qr-sampler starts. See [Setting up your own entropy source](#setting-up-your-own-entropy-source) below.

---

## Signal amplification

The signal amplification system converts raw entropy bytes into a single uniform float `u` in `(0, 1)` that drives token selection from the CDF. The default `zscore_mean` amplifier:

1. Interprets raw bytes as uint8 values
2. Computes the sample mean M
3. Derives SEM = `population_std / sqrt(N)` (never stored — always computed)
4. Computes z-score: `z = (M - population_mean) / SEM`
5. Maps to uniform via normal CDF: `u = 0.5 * (1 + erf(z / sqrt(2)))`
6. Clamps to `(epsilon, 1-epsilon)`

Under the null hypothesis (no bias), `u` is uniformly distributed on (0, 1). A small per-byte bias accumulates over thousands of samples, producing a detectable shift:

```
20,480 bytes with +0.003 mean bias per byte:
  M ~ 127.56, SEM ~ 0.514, z ~ 0.12, u ~ 0.548
```

This makes even tiny biases statistically observable while maintaining a well-defined distribution for token selection.

---

## Temperature strategies

### Fixed temperature (`fixed`)

Returns a constant temperature for every token. Set via `QR_FIXED_TEMPERATURE`.

### Entropy-dependent temperature (`edt`)

Dynamically adjusts temperature based on the Shannon entropy of the logit distribution:

```
H_norm = H / ln(vocab_size)         # Normalized entropy [0, 1]
T = base_temp * H_norm^exponent     # Power-law scaling
T = clamp(T, min_temp, max_temp)    # Bounds enforcement
```

High-entropy (uncertain) distributions get higher temperatures; low-entropy (confident) distributions get lower temperatures. This creates a feedback loop where the model's own uncertainty calibrates the randomness of selection.

---

## Setting up your own entropy source

qr-sampler is designed to connect *any* randomness source to LLM token sampling. This section walks through connecting your own hardware.

### Approach A: gRPC server (recommended)

The simplest path — implement a gRPC server that speaks the `qr_entropy.EntropyService` protocol.

#### 5-minute walkthrough

1. **Copy the template:**

```bash
cp examples/servers/qrng_template_server.py my_qrng_server.py
```

2. **Implement three methods** in the `QRNGHardware` class:

```python
class QRNGHardware:
    def __init__(self, device_path="/dev/qrng0"):
        # Open your hardware connection
        self._device = open(device_path, "rb")

    def generate(self, n_bytes: int) -> bytes:
        # CRITICAL: Generate entropy NOW, not from a buffer.
        # The quantum measurement must happen during this call.
        return self._device.read(n_bytes)

    def close(self):
        self._device.close()
```

3. **Run it:**

```bash
pip install grpcio qr-sampler
python my_qrng_server.py --port 50051
```

4. **Configure qr-sampler:**

**Windows (CMD):**

```cmd
set QR_GRPC_SERVER_ADDRESS=localhost:50051
set QR_ENTROPY_SOURCE_TYPE=quantum_grpc
```

**macOS / Linux:**

```bash
export QR_GRPC_SERVER_ADDRESS=localhost:50051
export QR_ENTROPY_SOURCE_TYPE=quantum_grpc
```

The template handles all gRPC boilerplate (unary + bidirectional streaming, health checks, graceful shutdown). You only write the hardware-specific code.

#### The gRPC protocol

The proto definition is minimal:

```protobuf
service EntropyService {
  rpc GetEntropy (EntropyRequest) returns (EntropyResponse);
  rpc StreamEntropy (stream EntropyRequest) returns (stream EntropyResponse);
}

message EntropyRequest {
  int32 bytes_needed = 1;
  int64 sequence_id = 2;
}

message EntropyResponse {
  bytes data = 1;
  int64 sequence_id = 2;
  int64 generation_timestamp_ns = 3;
  string device_id = 4;
}
```

Any language that supports gRPC can implement this server — Python, C++, Rust, Go, etc.

#### Just-in-time constraint

The entropy must be generated **after** the client sends the request, not from a pre-generated pool. This means:

- No buffering or caching of previously generated bytes
- The physical quantum measurement (or other random process) happens during the `generate()` call
- `generation_timestamp_ns` in the response proves freshness

This is critical for consciousness-research applications where the timing relationship between logit computation and entropy generation matters.

#### Deployment options

**Docker:**

```bash
cd examples/docker
# Edit Dockerfile.entropy-server if needed
docker build -f Dockerfile.entropy-server -t my-entropy-server ../..
docker run -p 50051:50051 my-entropy-server
```

**systemd (Linux):**

```bash
# Copy and edit the service file
sudo cp examples/systemd/qr-entropy-server.service /etc/systemd/system/
sudo cp examples/systemd/qr-entropy-server.env /etc/default/qr-entropy-server

# Edit the env file with your configuration
sudo systemctl enable --now qr-entropy-server
```

**Unix domain sockets** (lowest latency for co-located hardware):

**Windows (CMD):**

```cmd
python my_qrng_server.py --address unix:///var/run/qrng.sock
set QR_GRPC_SERVER_ADDRESS=unix:///var/run/qrng.sock
```

**macOS / Linux:**

```bash
python my_qrng_server.py --address unix:///var/run/qrng.sock
export QR_GRPC_SERVER_ADDRESS=unix:///var/run/qrng.sock
```

### Approach B: Python plugin (in-process)

For entropy sources that don't need a separate server, implement the `EntropySource` ABC directly:

```python
from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.registry import register_entropy_source

@register_entropy_source("my_source")
class MyEntropySource(EntropySource):
    @property
    def name(self) -> str:
        return "my_source"

    @property
    def is_available(self) -> bool:
        return True

    def get_random_bytes(self, n: int) -> bytes:
        # Your entropy generation logic here
        return my_hardware.read(n)

    def close(self) -> None:
        my_hardware.disconnect()
```

Register via entry points in your package's `pyproject.toml`:

```toml
[project.entry-points."qr_sampler.entropy_sources"]
my_source = "my_package.entropy:MyEntropySource"
```

Then set `QR_ENTROPY_SOURCE_TYPE=my_source`.

### Validation

Test your entropy server with the built-in test infrastructure:

```python
# In a test file
from qr_sampler.entropy.quantum import QuantumGrpcSource
from qr_sampler.config import QRSamplerConfig

config = QRSamplerConfig(grpc_server_address="localhost:50051")
source = QuantumGrpcSource(config)

# Basic connectivity
data = source.get_random_bytes(1024)
assert len(data) == 1024

# Health check
status = source.health_check()
print(status)  # {'source': 'quantum_grpc', 'healthy': True, ...}

source.close()
```

For statistical validation, check that your source produces uniform byte distributions:

```python
import numpy as np
from scipy import stats

data = source.get_random_bytes(100_000)
samples = np.frombuffer(data, dtype=np.uint8)

# KS test against uniform distribution
stat, p_value = stats.kstest(samples / 255.0, 'uniform')
print(f"KS statistic: {stat:.6f}, p-value: {p_value:.6f}")
# p-value should be > 0.05 for a good entropy source
```

---

## Plugin architecture

qr-sampler uses a registry + entry-points pattern for extensibility:

```
qr_sampler.entropy_sources          Third-party entropy sources
vllm.logits_processors              vLLM plugin registration
```

Each subsystem (entropy, amplification, temperature) has its own registry with decorator-based registration for built-in implementations and entry-point discovery for third-party extensions. The processor never instantiates strategy classes directly — it always goes through the registry.

### Adding new components

**New entropy source:** Subclass `EntropySource`, implement `name`, `is_available`, `get_random_bytes()`, `close()`. Register with `@register_entropy_source("name")`.

**New signal amplifier:** Subclass `SignalAmplifier`, implement `amplify(raw_bytes) -> AmplificationResult`. Register with `@AmplifierRegistry.register("name")`.

**New temperature strategy:** Subclass `TemperatureStrategy`, implement `compute_temperature(logits, config) -> TemperatureResult`. Always compute Shannon entropy. Register with `@TemperatureStrategyRegistry.register("name")`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development instructions.

---

## Project structure

```
src/qr_sampler/
├── __init__.py                    # Package version, re-exports
├── config.py                      # Pydantic-settings configuration
├── exceptions.py                  # Exception hierarchy
├── processor.py                   # vLLM V1 LogitsProcessor (orchestrates pipeline)
├── py.typed                       # PEP 561 type hint marker
├── amplification/
│   ├── base.py                    # SignalAmplifier ABC, AmplificationResult
│   ├── registry.py                # AmplifierRegistry
│   └── zscore.py                  # Z-score mean amplifier
├── entropy/
│   ├── base.py                    # EntropySource ABC
│   ├── registry.py                # Auto-discovery registry + entry points
│   ├── quantum.py                 # gRPC QRNG source (3 transport modes)
│   ├── system.py                  # os.urandom() source
│   ├── timing.py                  # CPU timing jitter source
│   ├── mock.py                    # Configurable test source
│   └── fallback.py                # Fallback wrapper
├── logging/
│   ├── types.py                   # TokenSamplingRecord dataclass
│   └── logger.py                  # SamplingLogger (none/summary/full)
├── proto/
│   ├── entropy_service.proto      # gRPC protocol definition
│   ├── entropy_service_pb2.py     # Hand-written protobuf stubs
│   └── entropy_service_pb2_grpc.py # Hand-written gRPC stubs
├── selection/
│   ├── types.py                   # SelectionResult dataclass
│   └── selector.py                # CDF-based token selector
└── temperature/
    ├── base.py                    # TemperatureStrategy ABC, Shannon entropy
    ├── registry.py                # TemperatureStrategyRegistry
    ├── fixed.py                   # Fixed temperature strategy
    └── edt.py                     # Entropy-dependent temperature

examples/
├── servers/
│   ├── simple_urandom_server.py   # Minimal reference server (~50 lines)
│   ├── timing_noise_server.py     # CPU timing entropy server
│   └── qrng_template_server.py    # Annotated template for custom QRNGs
├── docker/
│   ├── Dockerfile.entropy-server  # Docker image for entropy servers
│   └── docker-compose.yml         # Full stack (entropy + vLLM)
└── systemd/
    ├── qr-entropy-server.service  # systemd unit file
    └── qr-entropy-server.env      # Environment file
```

---

## Statistical analysis

qr-sampler includes statistical tests (in `tests/test_statistical_properties.py`, requires `scipy`) that validate the mathematical properties of the sampling pipeline:

- **KS-test for u-value uniformity**: Under the null hypothesis (no bias), amplified `u` values should be uniformly distributed on (0, 1). The test runs a Kolmogorov-Smirnov test against a uniform reference distribution.
- **Bias detection**: Verifies that introducing a small per-byte mean shift (e.g., `mean=128.0` instead of `127.5`) produces a statistically detectable shift in the `u` distribution — confirming the amplification system is sensitive enough for consciousness-research experiments.
- **EDT monotonicity**: Validates that the entropy-dependent temperature strategy produces higher temperatures for higher-entropy logit distributions, as designed.

These tests run as part of the standard test suite:

```bash
pytest tests/test_statistical_properties.py -v
```

---

## Development

```bash
# Clone and install
git clone https://github.com/alchemystack/Quantum-random-vLLM-sampler.git
cd Quantum-random-vLLM-sampler
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy --strict src/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
