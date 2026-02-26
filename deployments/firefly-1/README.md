# firefly-1 Profile

Connects to an external quantum random number generator (firefly-1 device).
No additional containers are needed — the QRNG server is external. vLLM
connects to it directly over the network.

## Quick start

1. Configure your environment:

   ```bash
   cd deployments/firefly-1
   cp .env.example .env
   ```

   Edit `.env`:
   - Set `QR_GRPC_API_KEY` to your actual API key.
   - Set `HF_TOKEN` if using a gated model.

2. Start:

   ```bash
   docker compose up --build
   ```

3. Send a request:

   ```bash
   curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen2.5-1.5B-Instruct",
       "prompt": "The nature of consciousness is",
       "max_tokens": 50
     }'
   ```

## Server details

| Field | Value |
|-------|-------|
| Address | `10.0.0.115:50051` |
| Protocol | `qrng.QuantumRNG` (gRPC, unary only) |
| Device | firefly-1 |
| Authentication | API key via `api-key` metadata header |

## Rate limits

| Limit | Value |
|-------|-------|
| Requests per minute | 500 |
| Daily data transfer | 500 MB |
| Max bytes per request | 13 KB (13,312 bytes) |

## Important notes

- **Sample count**: `QR_SAMPLE_COUNT` is set to 13,312 (13 KB) to stay within
  the server's max-bytes-per-request limit. The default of 20,480 would be
  rejected.
- **Unary only**: `QR_GRPC_STREAM_METHOD_PATH` is empty because this server
  only supports unary RPC. Do not set `QR_GRPC_MODE` to `server_streaming` or
  `bidi_streaming`.
- **API key**: The `.env` file contains a real API key. If you fork this repo
  publicly, add `firefly-1/` to `deployments/.gitignore`.

## Testing the connection

Verify the server is reachable before starting vLLM:

```bash
grpcurl -plaintext \
  -H 'api-key: YOUR_API_KEY_HERE' \
  -d '{"num_bytes": 100}' \
  10.0.0.115:50051 qrng.QuantumRNG/GetRandomBytes
```
