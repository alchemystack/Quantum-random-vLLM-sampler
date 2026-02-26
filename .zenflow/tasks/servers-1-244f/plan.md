# Spec and build

## Configuration
- **Artifacts Path**: {@artifacts_path} -> `.zenflow/tasks/{task_id}`

---

## Agent Instructions

Ask the user questions when anything is unclear or needs their input. This includes:
- Ambiguous or incomplete requirements
- Technical decisions that affect architecture or user experience
- Trade-offs that require business context

Do not make assumptions on important decisions -- get clarification first.

If you are blocked and need user clarification, mark the current step with `[!]` in plan.md before stopping.

---

## Workflow Steps

### [x] Step: Technical Specification

Assessed difficulty as **hard**. Created detailed spec at `.zenflow/tasks/servers-1-244f/spec.md`.

Two-layer architecture:
- **Layer 1 (source-agnostic core):** Sampler plugin + vLLM Docker. Default entropy = `system` (os.urandom). `QuantumGrpcSource` made protocol-configurable via method paths + generic protobuf wire format. No knowledge of any specific server.
- **Layer 2 (`deployments/` profiles):** Named folders with `.env` + optional `docker-compose.override.yml`. Committable, shareable, markedly separate. Template for new users, `urandom` as default guided path, `firefly-1` as user's personal QRNG server.

Key decisions from review:
- Migrate existing proto stubs from custom `struct.pack` to standard protobuf wire format (critical: current stubs are NOT protobuf-compatible)
- Change `entropy_source_type` default from `"quantum_grpc"` to `"system"` (breaking change, intentional)
- Redact `grpc_api_key` from health_check() and logging (no `pydantic.SecretStr`, manual redaction)
- Add `deployments/.gitignore` for users who want to exclude credential-containing profiles
- `Dockerfile.vllm` uses build-time install (reproducible image) vs current runtime-install approach

---

### [x] Step: Wire Format Migration and Config Foundation
<!-- chat-id: 8e248075-8e96-4a3e-9914-a538d77a4b6c -->

Migrate proto stubs to standard protobuf and add new gRPC config fields. This is the foundation for all subsequent work.

- Migrate `src/qr_sampler/proto/entropy_service_pb2.py`:
  - Rewrite `EntropyRequest.SerializeToString()` and `.FromString()` to use standard protobuf varint + tag encoding instead of `struct.pack(">iQ", ...)`
  - Rewrite `EntropyResponse.SerializeToString()` and `.FromString()` similarly
  - No changes to `entropy_service_pb2_grpc.py` (it just calls serialize/deserialize)
  - No changes to example servers (they use the message classes transparently)
- Add 4 new infrastructure fields to `QRSamplerConfig` in `config.py`:
  - `grpc_method_path` (default: `/qr_entropy.EntropyService/GetEntropy`)
  - `grpc_stream_method_path` (default: `/qr_entropy.EntropyService/StreamEntropy`)
  - `grpc_api_key` (default: `""`)
  - `grpc_api_key_header` (default: `api-key`)
- Change `entropy_source_type` default from `"quantum_grpc"` to `"system"` (breaking change)
- Add wire-format tests:
  - Round-trip encode/decode for both `EntropyRequest` and `EntropyResponse`
  - Validate against known protobuf-encoded bytes (use Python `protobuf` library in test as reference)
  - Cross-compatibility: `EntropyRequest.SerializeToString()` output decodable by generic `_decode_bytes_field1()`
- Update `tests/test_config.py`:
  - Test new field defaults, env var loading (`QR_GRPC_METHOD_PATH`, etc.), non-overridable enforcement
  - Update `test_infrastructure_defaults` assertion for `entropy_source_type` (now `"system"`)
  - Update `TestNonOverridableFields` and `TestFieldSets` parametrize lists to include new fields
- Run `pytest tests/test_config.py tests/test_entropy/ -v` and `ruff check src/ tests/`

---

### [x] Step: Refactor QuantumGrpcSource to Protocol-Agnostic
<!-- chat-id: accc2569-607d-44c4-a521-2f84ad77052e -->

Replace fixed `EntropyServiceStub` with generic gRPC method handles.

- Add generic protobuf wire-format helpers to `quantum.py`:
  - `_encode_varint_request(n: int) -> bytes`: field 1 varint encoding
  - `_decode_bytes_field1(data: bytes) -> bytes`: extract field 1 length-delimited bytes
- Refactor `_init_channel()`: replace `EntropyServiceStub(channel)` with:
  - `channel.unary_unary(config.grpc_method_path, request_serializer=..., response_deserializer=...)`
  - `channel.stream_stream(config.grpc_stream_method_path, ...)` only when path is non-empty
- Add metadata injection: if `grpc_api_key` is non-empty, pass `metadata=[(grpc_api_key_header, grpc_api_key)]` on all RPC calls
- Add startup validation: raise `ConfigValidationError` if streaming mode requested but `grpc_stream_method_path` is empty
- Refactor `_fetch_unary()`, `_fetch_server_streaming()`, `_fetch_bidi_streaming()` to use generic method handles
- Redact API key: `health_check()` returns `"authenticated": bool(self._api_key)` but never the key value
- Update all tests in `tests/test_entropy/test_quantum.py`:
  - Mock `channel.unary_unary()` / `channel.stream_stream()` instead of `EntropyServiceStub`
  - Add test for API key metadata injection
  - Add test for streaming mode validation when stream path is empty
  - Add test for API key redaction in health_check()
- Run `pytest tests/test_entropy/ -v` and `ruff check src/ tests/`

---

### [x] Step: Docker Restructure (vLLM-Only Default)
<!-- chat-id: 8944a636-2a74-4a46-ac89-cc8dbe6d7d97 -->

Make the Docker setup source-agnostic by default. Do this BEFORE deployment profiles since profiles reference docker-compose files.

- Create `examples/docker/Dockerfile.vllm`: standalone vLLM + qr-sampler image
  - Based on `vllm/vllm-openai:latest`
  - Installs qr-sampler with gRPC support at build time (not runtime)
  - Produces a reproducible, cacheable image
- Rewrite `examples/docker/docker-compose.yml` as vLLM-only:
  - Single `vllm` service using `build:` with `Dockerfile.vllm`
  - Default env: `QR_ENTROPY_SOURCE_TYPE=system` (works without external server)
  - Comments explaining how to use with a deployment profile's `.env`
  - No entropy-server service (that moves to `deployments/urandom/`)
- Keep `examples/docker/Dockerfile.entropy-server` unchanged

---

### [x] Step: Deployment Profiles System
<!-- chat-id: 9900dc8b-41da-42cc-825a-218ac5ae0779 -->

Create the `deployments/` directory with template, urandom, and firefly-1 profiles.

- Create `deployments/README.md`:
  - What a profile is (folder with `.env` + optional compose override)
  - How to create one (copy `_template/`)
  - Usage: `docker compose --env-file deployments/<name>/.env up`
  - Usage with override: `-f ... -f deployments/<name>/docker-compose.override.yml`
  - Security warning about committing credentials to public repos
- Create `deployments/.gitignore`:
  - Initially empty
  - Comments explaining users can add profile names to exclude from version control
- Create `deployments/_template/`:
  - `.env` with all configurable settings annotated, placeholder values, no secrets
  - `README.md` with setup instructions placeholder
- Create `deployments/urandom/`:
  - `.env` with `QR_ENTROPY_SOURCE_TYPE=quantum_grpc`, `QR_GRPC_SERVER_ADDRESS=entropy-server:50051`, default method paths
  - `docker-compose.override.yml` that adds the entropy-server service (content moved from current compose)
  - `README.md` with step-by-step urandom setup guide
- Create `deployments/firefly-1/`:
  - `.env` with: `QR_ENTROPY_SOURCE_TYPE=quantum_grpc`, `QR_GRPC_SERVER_ADDRESS=10.0.0.115:50051`, `QR_GRPC_METHOD_PATH=/qrng.QuantumRNG/GetRandomBytes`, `QR_GRPC_STREAM_METHOD_PATH=` (empty), `QR_GRPC_MODE=unary`, `QR_GRPC_API_KEY=37h2OeZJc8hCmA0CdAKCuLYlGv0M2IbEA-i-RlBef2g`, `QR_GRPC_API_KEY_HEADER=api-key`
  - `README.md` with server details (device: firefly-1, rate limit: 500/min, daily limit: 500MB, max bytes/request: 13KB)
- Verify compose configs: `docker compose -f examples/docker/docker-compose.yml -f deployments/urandom/docker-compose.override.yml --env-file deployments/urandom/.env config` validates

---

### [x] Step: Documentation and Final Verification
<!-- chat-id: d622c23b-325d-47ec-99d0-1424f952df61 -->

Update docs to reflect the two-layer architecture and run full verification.

- Update README.md:
  - Quick start: vLLM-only Docker with system entropy (Layer 1)
  - Deployment profiles: how to connect to any server (Layer 2)
  - Guided path: urandom profile for getting started
  - Custom server: copy template, fill in `.env`, run
  - Protocol flexibility: configurable method paths and auth
  - Breaking change note: `entropy_source_type` default changed to `"system"`
- Run full test suite: `pytest tests/ -v --cov=src/qr_sampler --cov-report=term-missing`
- Run linting: `ruff check src/ tests/`
- Run type check: `mypy --strict src/`
- Write report to `.zenflow/tasks/servers-1-244f/report.md`
