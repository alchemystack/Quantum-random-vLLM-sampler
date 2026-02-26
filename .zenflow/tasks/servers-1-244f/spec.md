# Technical Specification: Two-Layer Deployment Separation

## Difficulty: Hard

Touches gRPC protocol abstraction, Docker restructuring, a new `deployments/` profile system, and documentation.

---

## Problem Statement

The codebase currently blends "how to run vLLM with the sampler" and "how to connect to a specific entropy server" into one entangled setup. The goal is a clean two-layer split:

**Layer 1 — The Sampler Core + vLLM Docker (source-agnostic)**
- The sampler plugin code, its config, and the Docker setup for running vLLM with qr-sampler installed.
- Knows nothing about any specific entropy server.
- Ships with `system` (os.urandom) as the default entropy source — works out of the box with zero external dependencies.
- Docker docs focus on "get vLLM running with the plugin."

**Layer 2 — Deployment Profiles (`deployments/`)**
- Self-contained, named folders that each describe "here's how to connect to entropy source X."
- Each profile is a directory containing an `.env` file (gRPC settings, auth, model, etc.) and optionally a `docker-compose.override.yml`, setup notes, or scripts.
- Profiles are **committable and shareable** — they live in the repo but are clearly separate from the sampler code.
- The `urandom` profile is the default guided path (separate urandom gRPC server).
- The user's own profile (`firefly-1`) sits alongside it with their specific server IP, API key, etc.
- Other people can add their own profiles by copying the template.

**Additional requirement:** The user's real QRNG server (`qrng.QuantumRNG` proto with `api-key` metadata auth) uses a different gRPC protocol than the built-in stubs (`qr_entropy.EntropyService`). The sampler must connect to it without code changes — configuration only.

---

## Technical Context

- **Language:** Python 3.10+
- **Key dependencies:** grpcio, protobuf, pydantic-settings, numpy
- **Existing protocol:** `qr_entropy.EntropyService` — `EntropyRequest(bytes_needed=field1, sequence_id=field2)` / `EntropyResponse(data=field1, ...)`
- **User's server protocol:** `qrng.QuantumRNG` — `RandomRequest(num_bytes=field1)` / `RandomResponse(data=field1, timestamp=field2, device_id=field3)` + `api-key` metadata header
- **Key observation:** Both protos put the byte-count as field 1 (varint) in the request and the random bytes as field 1 (length-delimited) in the response. Protobuf wire format is field-number-based, so a generic serializer handles both.

---

## Critical Prerequisite: Wire Format Migration

**The existing hand-written proto stubs (`entropy_service_pb2.py`) use a CUSTOM binary format** (`struct.pack(">iQ", ...)`) that is NOT standard protobuf wire format. The gRPC client (`QuantumGrpcSource` via `EntropyServiceStub`) and the example servers (`simple_urandom_server.py`, etc.) share this custom format as a matched pair.

When we switch `QuantumGrpcSource` to use standard protobuf wire encoding (required for compatibility with arbitrary servers like the user's `qrng.QuantumRNG`), the example servers will break unless their stubs are also migrated.

**Resolution:** Migrate `entropy_service_pb2.py`'s `SerializeToString()` and `FromString()` methods to use standard protobuf wire encoding. This:
1. Makes example servers compatible with the new generic client
2. Makes example servers `grpcurl`-compatible (they currently are NOT, because `grpcurl` speaks standard protobuf)
3. Is a self-contained change to one file — the gRPC stubs in `entropy_service_pb2_grpc.py` just call `.SerializeToString()`/`.FromString()` and don't care about the encoding

This migration must happen in the same step as the wire-format work, before `QuantumGrpcSource` is refactored.

---

## Implementation Approach

### Part 1: Wire Format + Config Foundation

**Migrate proto stubs to standard protobuf encoding:**
- Rewrite `EntropyRequest.SerializeToString()` and `.FromString()` to use protobuf varint + tag encoding
- Rewrite `EntropyResponse.SerializeToString()` and `.FromString()` to use protobuf field encoding
- This is the SAME encoding format that the generic client serializer will use
- No changes needed to `entropy_service_pb2_grpc.py` (it just calls serialize/deserialize)
- No changes needed to example servers (they use `EntropyRequest`/`EntropyResponse` classes)

**New config fields in `QRSamplerConfig` (all infrastructure, NOT per-request):**
```python
grpc_method_path: str = Field(
    default="/qr_entropy.EntropyService/GetEntropy",
    description="gRPC method path for unary RPC",
)
grpc_stream_method_path: str = Field(
    default="/qr_entropy.EntropyService/StreamEntropy",
    description="gRPC method path for streaming RPC (empty disables streaming)",
)
grpc_api_key: str = Field(
    default="",
    description="API key sent via gRPC metadata (empty = no auth)",
)
grpc_api_key_header: str = Field(
    default="api-key",
    description="gRPC metadata header name for API key",
)
```

**Breaking change: default `entropy_source_type`.**
Change from `"quantum_grpc"` to `"system"`. This means the plugin works out of the box without any external server. Users who want gRPC must explicitly set `QR_ENTROPY_SOURCE_TYPE=quantum_grpc`. This is intentional — it's the core of the Layer 1 "source-agnostic" design.

**Security: `grpc_api_key` handling.**
The API key must not leak into logs or diagnostic output. The `health_check()` method in `quantum.py` must NOT include the key value. It will include `"authenticated": bool(self._api_key)` but never the key itself. Any debug logging that touches the API key masks all but the last 4 characters. We do NOT use `pydantic.SecretStr` because pydantic-settings env-var loading with SecretStr requires special handling and the field needs to be passed as a plain string to gRPC metadata. Instead, we redact manually in health_check() and logging.

### Part 2: Make `QuantumGrpcSource` Protocol-Configurable

**Replace `EntropyServiceStub` with direct generic gRPC method handles:**
- `channel.unary_unary(config.grpc_method_path, request_serializer=..., response_deserializer=...)`
- `channel.stream_stream(config.grpc_stream_method_path, ...)` when path is non-empty
- Generic serializer: encode field 1 as varint (byte count) for request; decode field 1 as length-delimited (bytes) for response
- If `grpc_api_key` is non-empty, inject `[(api_key_header, api_key)]` as gRPC call metadata on every RPC
- If `grpc_stream_method_path` is empty and `grpc_mode` is `server_streaming` or `bidi_streaming`, raise `ConfigValidationError` at `__init__()` time

**Why this works:** Protobuf wire format identifies fields by *number*, not *name*. `EntropyRequest.bytes_needed` (field 1) and `RandomRequest.num_bytes` (field 1) produce identical wire bytes. Same for the response. One generic serializer handles any entropy proto where field 1 is the byte count (request) and field 1 is the data (response).

### Part 3: Docker Restructure — vLLM-Only Default

**New structure in `examples/docker/`:**
- `Dockerfile.vllm` — New. Builds a vLLM image with qr-sampler baked in at build time. Preferred over the current runtime-install approach (`pip install` in compose `command:`) because it produces a reproducible, cacheable image. The compose file uses `build:` to reference this Dockerfile.
- `Dockerfile.entropy-server` — Unchanged.
- `docker-compose.yml` — Rewritten. **vLLM-only.** Builds from `Dockerfile.vllm`. Default: `QR_ENTROPY_SOURCE_TYPE=system`. Users point it at a deployment profile's `.env` for custom config.

### Part 4: `deployments/` Profile System

**New directory: `deployments/`** at repo root.

```
deployments/
  README.md               # How profiles work, how to create your own
  .gitignore              # Initially empty; users add their profile name to exclude secrets
  _template/              # Copy this to create a new profile
    .env                  # Annotated env template with all settings, placeholder values
    README.md             # Setup instructions placeholder
  urandom/                # Default guided path: separate urandom gRPC server
    .env                  # QR_ENTROPY_SOURCE_TYPE=quantum_grpc, QR_GRPC_SERVER_ADDRESS=entropy-server:50051
    docker-compose.override.yml  # Adds urandom entropy-server service
    README.md             # Step-by-step setup guide
  firefly-1/              # User's personal QRNG server profile
    .env                  # 10.0.0.115:50051, /qrng.QuantumRNG/GetRandomBytes, API key
    README.md             # Server details, rate limits, notes
```

**How profiles work:**
```bash
# vLLM-only with system entropy (no profile needed):
docker compose -f examples/docker/docker-compose.yml up

# With a profile (external server):
docker compose -f examples/docker/docker-compose.yml \
  --env-file deployments/firefly-1/.env up

# With a profile that adds services (urandom):
docker compose -f examples/docker/docker-compose.yml \
  -f deployments/urandom/docker-compose.override.yml \
  --env-file deployments/urandom/.env up
```

**Important for `firefly-1` profile:** Must explicitly set `QR_GRPC_MODE=unary` and `QR_GRPC_STREAM_METHOD_PATH=` (empty string) since the user's server only supports unary RPC.

**Gitignore strategy:**
- `deployments/` is NOT gitignored — profiles are committable
- `deployments/.gitignore` is provided, initially empty, with comments explaining users can add their profile name to exclude credentials from version control
- `_template/` and `urandom/` contain no secrets — always safe to commit
- `firefly-1/` contains a real API key — README warns about this for public repos

---

## Source Code Structure Changes

### Modified Files

| File | Change |
|------|--------|
| `src/qr_sampler/config.py` | Add 4 new infrastructure fields; change `entropy_source_type` default from `"quantum_grpc"` to `"system"` |
| `src/qr_sampler/proto/entropy_service_pb2.py` | Migrate `SerializeToString()`/`FromString()` from custom `struct.pack` to standard protobuf wire format |
| `src/qr_sampler/entropy/quantum.py` | Replace `EntropyServiceStub` with generic `channel.unary_unary()`/`channel.stream_stream()`; add metadata injection; add generic protobuf serializers; redact API key from `health_check()` |
| `examples/docker/docker-compose.yml` | Rewrite as vLLM-only service |
| `tests/test_config.py` | Add tests for new config fields; update `entropy_source_type` default assertion |
| `tests/test_entropy/test_quantum.py` | Update mocks for protocol-agnostic gRPC (mock `channel.unary_unary()` instead of `EntropyServiceStub`) |

### New Files

| File | Purpose |
|------|---------|
| `examples/docker/Dockerfile.vllm` | Standalone vLLM + qr-sampler image (build-time install) |
| `deployments/README.md` | How profiles work |
| `deployments/.gitignore` | Initially empty with comments for excluding secrets |
| `deployments/_template/.env` | Annotated env template |
| `deployments/_template/README.md` | Setup guide placeholder |
| `deployments/urandom/.env` | Urandom server config |
| `deployments/urandom/docker-compose.override.yml` | Adds entropy-server service |
| `deployments/urandom/README.md` | Step-by-step urandom setup guide |
| `deployments/firefly-1/.env` | User's QRNG server config |
| `deployments/firefly-1/README.md` | Server details and notes |

### Unchanged

| File | Reason |
|------|--------|
| `src/qr_sampler/proto/entropy_service_pb2_grpc.py` | Just calls `.SerializeToString()`/`.FromString()` — encoding-agnostic |
| `examples/servers/*` | Use `EntropyRequest`/`EntropyResponse` classes which are migrated transparently |
| `src/qr_sampler/processor.py` | Only talks to `EntropySource` ABC |
| `src/qr_sampler/entropy/base.py` | ABC unchanged |
| `src/qr_sampler/entropy/registry.py` | Registry unchanged |
| `examples/docker/Dockerfile.entropy-server` | Stays for building entropy server images |

---

## Data Model / API / Interface Changes

### New Config Fields

```python
# Infrastructure (NOT per-request overridable)

grpc_method_path: str = Field(
    default="/qr_entropy.EntropyService/GetEntropy",
    description="gRPC method path for unary RPC (e.g. '/qrng.QuantumRNG/GetRandomBytes')",
)
grpc_stream_method_path: str = Field(
    default="/qr_entropy.EntropyService/StreamEntropy",
    description="gRPC method path for streaming RPC (empty string disables streaming modes)",
)
grpc_api_key: str = Field(
    default="",
    description="API key sent via gRPC metadata (empty = no auth)",
)
grpc_api_key_header: str = Field(
    default="api-key",
    description="gRPC metadata header name for the API key",
)
```

### Changed Default (Breaking)

```python
# Was "quantum_grpc", now "system"
entropy_source_type: str = Field(
    default="system",
    description="Primary entropy source identifier",
)
```

### Generic Protobuf Wire Format Helpers

Minimal encoder/decoder added to `quantum.py`:

- **`_encode_varint_request(n: int) -> bytes`**: Encodes a protobuf message with field 1 = varint `n`. ~10 lines.
- **`_decode_bytes_field1(data: bytes) -> bytes`**: Scans protobuf wire-format bytes for field 1 (wiretype 2 = length-delimited) and returns the raw bytes. ~15 lines.

The same encoding logic is used in the migrated `entropy_service_pb2.py` stubs, ensuring example servers and the generic client speak the same wire format.

### Streaming Mode Validation

When `grpc_stream_method_path` is empty and `grpc_mode` is `server_streaming` or `bidi_streaming`, raise `ConfigValidationError` at `QuantumGrpcSource.__init__()` time.

### API Key Redaction

`health_check()` returns `"authenticated": bool(self._api_key)` but never the key itself.

---

## Verification Approach

### Per-Step Verification

Each implementation step:
1. Run relevant test module(s): `pytest tests/<module> -v`
2. Run `ruff check src/ tests/`

### Wire Format Verification

Tests must validate against **known protobuf-encoded bytes**:
- Hand-encode a message using the Python `protobuf` library and verify custom encoder output matches
- Round-trip: encode -> decode -> verify values match
- Cross-compatibility: verify `EntropyRequest.SerializeToString()` output can be decoded by `_decode_bytes_field1()` (and vice versa)

### Final Verification

1. `pytest tests/ -v --cov=src/qr_sampler --cov-report=term-missing` — all pass, coverage >=90%
2. `ruff check src/ tests/` — no lint errors
3. `mypy --strict src/` — no type errors
4. Docker build smoke test: Dockerfiles parse without errors
5. Profile usability: verify compose configs are valid
