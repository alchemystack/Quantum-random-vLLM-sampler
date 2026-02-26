# firefly-1 Profile

External quantum random number generator server.

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

## Usage

```bash
cd examples/docker
docker compose --env-file ../../deployments/firefly-1/.env up --build
```

No additional containers are needed -- the QRNG server is external. vLLM
connects to it directly over the network at `10.0.0.115:50051`.

## Notes

- `QR_SAMPLE_COUNT` is set to 13312 (13 KB) to stay within the server's
  max-bytes-per-request limit. The default of 20480 would be rejected.
- `QR_GRPC_STREAM_METHOD_PATH` is empty because this server only supports
  unary RPC. Do not set `QR_GRPC_MODE` to `server_streaming` or
  `bidi_streaming`.
- The `.env` file contains a real API key. If you fork this repo publicly,
  add `firefly-1/` to `deployments/.gitignore`.

## Testing the connection

Verify the server is reachable before starting vLLM:

```bash
grpcurl -plaintext \
  -proto qrng.proto \
  -H 'api-key: 37h2OeZJc8hCmA0CdAKCuLYlGv0M2IbEA-i-RlBef2g' \
  -d '{"num_bytes": 100}' \
  10.0.0.115:50051 qrng.QuantumRNG/GetRandomBytes
```
