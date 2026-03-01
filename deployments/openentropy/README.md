# OpenEntropy Profile

Runs vLLM with qr-sampler using **OpenEntropy** — a local hardware entropy
source that collects noise from 63 hardware sources on Apple Silicon (thermal,
timing, microarchitecture, GPU, etc.). This is a **native-only profile** — no
Docker, no network dependency.

## Why not Docker?

Docker containers cannot access Metal GPU or native hardware entropy sources on
macOS. Apple's Virtualization.framework has no GPU passthrough, and hardware
noise sources (thermal sensors, CPU timing, GPU state) are not exposed to
containerized processes. OpenEntropy requires native execution.

## Quick start

1. Install OpenEntropy and qr-sampler:

   ```bash
   pip install openentropy
   pip install -e /path/to/qr-sampler
   ```

2. Configure your environment:

   ```bash
   cd deployments/openentropy
   cp .env.example .env
   ```

   Edit `.env` if needed — set `HF_TOKEN` if using a gated model.

3. Start vLLM:

   ```bash
   source .env
   vllm serve $HF_MODEL \
     --port $VLLM_PORT \
     --logits-processors qr_sampler
   ```

## Available entropy sources

List all available sources on your hardware:

```bash
python -c "from openentropy import detect_available_sources; print([s['name'] for s in detect_available_sources()])"
```

On Apple Silicon, you'll see sources like:
- `thermal_sensors` — CPU/GPU die temperature
- `cpu_timing` — CPU cycle counter jitter
- `gpu_state` — GPU utilization noise
- `memory_bandwidth` — DRAM access timing
- And 59 more...

## Conditioning modes

OpenEntropy supports three conditioning strategies:

| Mode | Use case | Properties |
|------|----------|-----------|
| `raw` | Research (default) | Preserves hardware noise signal; minimal processing |
| `vonneumann` | Debiased entropy | Von Neumann debiasing; slower, more uniform |
| `sha256` | Cryptographic | SHA-256 hashing; suitable for security-critical applications |

Set `QR_OE_CONDITIONING` in `.env` or override per-request:

```python
# Per-request override
extra_args = {"qr_oe_conditioning": "sha256"}
```

## Parallel collection

By default, `QR_OE_PARALLEL=true` collects from multiple sources simultaneously,
increasing entropy throughput. Set to `false` for sequential collection (slower,
lower memory overhead).

## When to use this profile

- **Consciousness research**: Study whether intent influences quantum-random
  processes using native hardware entropy.
- **Local experiments**: No network latency, no external dependencies.
- **Apple Silicon development**: Leverage Metal GPU and native hardware sensors.
- **Research baseline**: Compare hardware entropy against system entropy
  (`/dev/urandom`).

## Web UI (optional)

This profile includes [Open WebUI](https://github.com/open-webui/open-webui), a
ChatGPT-style web interface. To use it, you'll need to run it separately (not
included in this native profile):

```bash
docker run -d -p 3000:3000 --name open-webui ghcr.io/open-webui/open-webui:latest
```

Then point it at your vLLM instance running on `localhost:8000`.

A pre-built filter function for controlling qr-sampler parameters from the UI is
available at [`examples/open-webui/`](../../examples/open-webui/). See that
directory's README for import instructions.

## Next steps

Once this profile works, you can:
1. Adjust `QR_OE_SOURCES` to use specific entropy sources.
2. Experiment with different conditioning modes (`raw`, `vonneumann`, `sha256`).
3. Compare results against the `urandom` profile (gRPC-based) or `system` profile
   (fallback).
