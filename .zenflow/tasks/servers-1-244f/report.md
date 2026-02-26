# Task Report: Two-Layer Deployment Separation

## Summary

Implemented a clean two-layer architecture separating the sampler plugin (source-agnostic core) from entropy source configuration (deployment profiles). The codebase no longer pushes any specific server, model, or credentials as defaults.

## Changes by Step

### Step 1: Wire Format Migration and Config Foundation

- Migrated `entropy_service_pb2.py` from custom `struct.pack` encoding to standard protobuf wire format (varint + tag encoding). Example servers and `grpcurl` are now interoperable.
- Added 4 new infrastructure config fields: `grpc_method_path`, `grpc_stream_method_path`, `grpc_api_key`, `grpc_api_key_header`.
- Changed `entropy_source_type` default from `"quantum_grpc"` to `"system"` (breaking change). Plugin now works out of the box with zero external dependencies.
- Added wire-format round-trip tests in `tests/test_wire_format.py`.

### Step 2: Refactor QuantumGrpcSource to Protocol-Agnostic

- Replaced `EntropyServiceStub` with generic `channel.unary_unary()` / `channel.stream_stream()` using configurable method paths.
- Added generic protobuf wire-format helpers (`_encode_varint_request`, `_decode_bytes_field1`) for request serialization/response deserialization.
- Added API key metadata injection on all RPC calls when `grpc_api_key` is non-empty.
- Added validation: streaming modes raise `ConfigValidationError` when `grpc_stream_method_path` is empty.
- API key is never logged; `health_check()` returns `"authenticated": bool` only.
- Updated all quantum tests to mock generic method handles instead of `EntropyServiceStub`.

### Step 3: Docker Restructure (vLLM-Only Default)

- Created `examples/docker/Dockerfile.vllm`: standalone vLLM + qr-sampler image with build-time install (reproducible, cacheable).
- Rewrote `examples/docker/docker-compose.yml` as vLLM-only. Default: `QR_ENTROPY_SOURCE_TYPE=system`. Users point at a deployment profile's `.env` for custom config.
- Kept `Dockerfile.entropy-server` unchanged.

### Step 4: Deployment Profiles System

- Created `deployments/` directory at repo root with:
  - `README.md` -- how profiles work, how to create your own, security notes.
  - `.gitignore` -- initially empty; comments explain how to exclude credential-containing profiles.
  - `_template/` -- annotated `.env` with all configurable settings, placeholder values, no secrets.
  - `urandom/` -- `.env`, `docker-compose.override.yml` (adds entropy-server container), `README.md`.
  - `firefly-1/` -- `.env` (10.0.0.115:50051, `/qrng.QuantumRNG/GetRandomBytes`, API key, `QR_SAMPLE_COUNT=13312`), `README.md` with server details and rate limits.

### Step 5: Documentation and Final Verification

- Updated `README.md`:
  - Quick start now leads with system entropy (no external server needed).
  - Docker quick start explains vLLM-only default and how to use deployment profiles.
  - Config reference table updated: `QR_ENTROPY_SOURCE_TYPE` default is `system`, 4 new gRPC fields added.
  - Entropy sources table: `system` is now marked as default, `quantum_grpc` description updated to "any protocol".
  - New "Deployment profiles" section explains the two-layer architecture, profile usage, and protocol flexibility.
  - "Setting up your own entropy source" updated to mention deployment profiles and protocol-agnostic design.
  - Project structure tree updated to include `Dockerfile.vllm`, revised `docker-compose.yml` description, and full `deployments/` directory.
- Fixed 2 mypy `--strict` errors in `quantum.py` (added `assert self._stream_method is not None` guards for `None`-callable checks on streaming method handle).

## Verification Results

| Check | Result |
|-------|--------|
| `pytest tests/ -v --cov=src/qr_sampler` | **308 passed**, 95% coverage |
| `ruff check src/ tests/` | **All checks passed** |
| `mypy --strict src/` | **Success: no issues found in 30 source files** |

## Breaking Changes

1. **`entropy_source_type` default changed from `"quantum_grpc"` to `"system"`.**
   Users who relied on the old default must now explicitly set `QR_ENTROPY_SOURCE_TYPE=quantum_grpc`.

## Files Modified

| File | Change |
|------|--------|
| `README.md` | Rewrote quick start, config reference, entropy sources, project structure; added deployment profiles and protocol flexibility sections |
| `src/qr_sampler/entropy/quantum.py` | Added 2 assert guards for mypy --strict compliance |

## Files Created (across all steps)

| File | Purpose |
|------|---------|
| `src/qr_sampler/proto/entropy_service_pb2.py` | Migrated to standard protobuf wire format |
| `examples/docker/Dockerfile.vllm` | vLLM + qr-sampler build-time image |
| `examples/docker/docker-compose.yml` | Rewritten as vLLM-only |
| `tests/test_wire_format.py` | Wire format round-trip and compatibility tests |
| `deployments/README.md` | Profile system documentation |
| `deployments/.gitignore` | Credential exclusion guide |
| `deployments/_template/.env` | Annotated template |
| `deployments/_template/README.md` | Template setup guide |
| `deployments/urandom/.env` | Urandom server config |
| `deployments/urandom/docker-compose.override.yml` | Adds entropy-server container |
| `deployments/urandom/README.md` | Urandom profile guide |
| `deployments/firefly-1/.env` | QRNG server config with API key |
| `deployments/firefly-1/README.md` | Server details and rate limits |
