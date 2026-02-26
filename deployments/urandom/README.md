# urandom Profile

Runs a separate gRPC entropy server that generates random bytes from
`os.urandom()`. This is the recommended starting point for testing the full
gRPC entropy pipeline before connecting a real quantum random number generator.

## What this profile does

- Starts a lightweight gRPC entropy server container (`entropy-server`)
  running `simple_urandom_server.py`.
- Configures qr-sampler in the vLLM container to fetch entropy from
  `entropy-server:50051` instead of using the default system entropy.
- Uses the built-in `qr_entropy.EntropyService` protocol (unary mode).

## Setup

```bash
cd examples/docker
docker compose \
  -f docker-compose.yml \
  -f ../../deployments/urandom/docker-compose.override.yml \
  --env-file ../../deployments/urandom/.env \
  up --build
```

The entropy server starts first and listens on port 50051. vLLM connects to
it automatically via Docker networking (`entropy-server:50051`).

## Switching to bidirectional streaming

For lower latency, change `QR_GRPC_MODE` in the `.env` file:

```
QR_GRPC_MODE=bidi_streaming
```

The urandom server supports all three transport modes (unary, server streaming,
bidirectional streaming).

## When to use this profile

- **Getting started**: Validate that the full entropy pipeline (gRPC fetch,
  signal amplification, token selection) works end-to-end.
- **Baseline experiments**: Compare system-entropy sampling against
  gRPC-sourced sampling to measure the overhead.
- **Development**: Test changes to the gRPC client without needing real
  hardware.

## Next steps

Once this profile works, you can:
1. Copy `_template/` to create a profile for your own entropy server.
2. Point `QR_GRPC_SERVER_ADDRESS` at real QRNG hardware.
3. Set up API key authentication if your server requires it.
