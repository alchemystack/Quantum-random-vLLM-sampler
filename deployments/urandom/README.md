# urandom Profile

Runs a co-located gRPC entropy server that generates random bytes from
`os.urandom()`. This is the recommended starting point for testing the full
gRPC entropy pipeline before connecting to real quantum random number generator
hardware.

## Quick start

1. Configure your environment:

   ```bash
   cd deployments/urandom
   cp .env.example .env
   ```

   Edit `.env` if needed — set `HF_TOKEN` if using a gated model.

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

## What this runs

- **entropy-server** -- Lightweight gRPC server container running
  `simple_urandom_server.py`. Listens on port 50051.
- **vllm** -- vLLM inference server with qr-sampler plugin. Fetches entropy
  from `entropy-server:50051` via Docker networking.

Both containers start together. The entropy server starts first; vLLM waits
for it via `depends_on`.

## Switching to bidirectional streaming

For lower latency, change `QR_GRPC_MODE` in your `.env` file:

```
QR_GRPC_MODE=bidi_streaming
```

The urandom server supports all three transport modes (unary, server streaming,
bidirectional streaming). Bidirectional streaming keeps a persistent gRPC
connection open, reducing per-token overhead to ~50-100us on the same machine.

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
