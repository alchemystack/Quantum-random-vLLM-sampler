# Technical Specification: Open WebUI Integration with qr-sampler

## Difficulty: Medium

The core integration is straightforward (adding an Open WebUI service to Docker Compose), but doing it well requires careful thought about parameter passthrough, user experience, and maintaining qr-sampler's deployment flexibility.

## Technical Context

- **Language**: Python 3.10+, YAML (Docker Compose), Markdown
- **Existing infrastructure**: Deployment profiles in `deployments/` with Docker Compose + `.env.example` + README pattern
- **Key dependency**: Open WebUI (`ghcr.io/open-webui/open-webui:main`) — a self-hosted ChatGPT-style web UI
- **Connection method**: Open WebUI connects to vLLM via its OpenAI-compatible API (`/v1` endpoint)
- **Parameter flow**: Open WebUI request → Filter Function `inlet()` injects `qr_*` keys → vLLM `/v1/chat/completions` → `SamplingParams.extra_args` → qr-sampler `resolve_config()`

## Decisions (User-Confirmed)

1. **Option A selected**: Add Open WebUI to every deployment profile using Docker Compose `profiles: ["ui"]`
2. **Filter Function included**: Ship a pre-built Open WebUI Filter Function for qr-sampler parameter control via admin Valves UI
3. **README prominence**: Add a recommended "Try the Web UI" section to the main README

## Architecture Overview

```
┌─────────────────────┐
│    Open WebUI        │  ← Users chat here (port 3000)
│  (profiles: ["ui"])  │
└──────────┬──────────┘
           │  HTTP (OpenAI-compatible)
           │
           │  Filter Function injects qr_* keys
           │  into request body before forwarding
           │
┌──────────▼──────────┐     gRPC      ┌──────────────────┐
│       vLLM          │ ◄────────────► │  Entropy Server  │
│   + qr-sampler      │               │  (optional)      │
│   (port 8000)       │               │  (port 50051)    │
└─────────────────────┘               └──────────────────┘
```

## Implementation Approach

### Part 1: Docker Compose profiles (all deployment profiles)

Add an `open-webui` service with `profiles: ["ui"]` to each `docker-compose.yml`. This ensures:
- `docker compose up` — unchanged behavior, Open WebUI does NOT start
- `docker compose --profile ui up` — starts Open WebUI alongside everything else

Service definition (identical across profiles):

```yaml
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    profiles: ["ui"]
    ports:
      - "${OPEN_WEBUI_PORT:-3000}:8080"
    environment:
      OPENAI_API_BASE_URL: "http://vllm:8000/v1"
      OPENAI_API_KEY: "unused"
      WEBUI_AUTH: "${OPEN_WEBUI_AUTH:-false}"
    volumes:
      - open-webui-data:/app/backend/data
    depends_on:
      - vllm
    restart: unless-stopped
```

Plus `open-webui-data:` in the `volumes:` section.

### Part 2: Open WebUI Filter Function for qr-sampler

Open WebUI stores functions in its SQLite database. They cannot be auto-loaded from `.py` files. The approach:

1. **Ship the filter as two files**:
   - `examples/open-webui/qr_sampler_filter.py` — the source code (readable, editable)
   - `examples/open-webui/qr_sampler_filter.json` — Open WebUI import-ready JSON format

2. **Import workflow** (documented in README):
   - Open http://localhost:3000 → Admin Panel → Functions
   - Click Import → select `qr_sampler_filter.json`
   - Toggle "Global" to apply to all models
   - Configure parameters via the Valves gear icon

3. **Filter function design**:
   - Type: `filter` with `inlet()` method
   - Valves expose all per-request-overridable qr-sampler fields from `_PER_REQUEST_FIELDS`
   - `inlet()` injects `qr_*` keys as top-level fields in the request body
   - vLLM maps unknown top-level keys to `SamplingParams.extra_args`
   - qr-sampler's `resolve_config()` picks them up transparently

**Valve fields** (matching `_PER_REQUEST_FIELDS` in `config.py`):

| Valve | Type | Default | Maps to |
|-------|------|---------|---------|
| `enable_qr_sampling` | `bool` | `True` | (controls whether filter injects anything) |
| `temperature_strategy` | `Literal["fixed", "edt"]` | `"fixed"` | `qr_temperature_strategy` |
| `fixed_temperature` | `float` | `0.7` | `qr_fixed_temperature` |
| `top_k` | `int` | `50` | `qr_top_k` |
| `top_p` | `float` | `0.9` | `qr_top_p` |
| `sample_count` | `int` | `20480` | `qr_sample_count` |
| `log_level` | `Literal["none", "summary", "full"]` | `"summary"` | `qr_log_level` |
| `diagnostic_mode` | `bool` | `False` | `qr_diagnostic_mode` |

Infrastructure fields (`entropy_source_type`, `grpc_*`, `fallback_mode`) are deliberately excluded — they cannot change per-request and are controlled by environment variables.

### Part 3: Documentation

- **`deployments/README.md`**: Add `--profile ui` to the quick start section
- **Each profile's README**: Add "Web UI (optional)" section with usage + filter import instructions
- **`README.md`**: Add a prominent "Web UI" section recommending Open WebUI, linking to the filter function and deployment docs
- **`examples/open-webui/README.md`**: Detailed guide for the filter function (what it does, how to import, how to configure Valves)

## Files to Create

| File | Purpose |
|------|---------|
| `examples/open-webui/qr_sampler_filter.py` | Human-readable filter source code |
| `examples/open-webui/qr_sampler_filter.json` | Open WebUI importable JSON |
| `examples/open-webui/README.md` | Filter function documentation |

## Files to Modify

| File | Change |
|------|--------|
| `deployments/urandom/docker-compose.yml` | Add `open-webui` service + volume |
| `deployments/urandom/.env.example` | Add `OPEN_WEBUI_PORT`, `OPEN_WEBUI_AUTH` |
| `deployments/urandom/README.md` | Add "Web UI (optional)" section |
| `deployments/firefly-1/docker-compose.yml` | Add `open-webui` service + volume |
| `deployments/firefly-1/.env.example` | Add `OPEN_WEBUI_PORT`, `OPEN_WEBUI_AUTH` |
| `deployments/firefly-1/README.md` | Add "Web UI (optional)" section |
| `deployments/_template/docker-compose.yml` | Add `open-webui` service + volume |
| `deployments/_template/.env.example` | Add `OPEN_WEBUI_PORT`, `OPEN_WEBUI_AUTH` |
| `deployments/_template/README.md` | Mention UI option |
| `deployments/README.md` | Add `--profile ui` to quick start |
| `README.md` | Add prominent "Web UI" section |

## Verification

1. **Compose syntax**: `docker compose --profile ui config` in each profile directory — validates YAML
2. **Default behavior**: `docker compose config` (no profile) — confirm Open WebUI is NOT listed in resolved services
3. **Filter function**: Verify `qr_sampler_filter.json` is valid JSON and contains the full source code
4. **Filter Valves**: Verify all Valve field names match the `qr_*` keys that `resolve_config()` accepts
5. **Manual test**: If Docker + GPU available — `docker compose --profile ui up`, open `http://localhost:3000`, import filter, chat, check vLLM logs for qr-sampler activity

## Data Model / API / Interface Changes

None. This is a deployment infrastructure + documentation addition. No Python source code in `src/qr_sampler/` is modified. No tests are affected. The filter function is an Open WebUI plugin, not part of the qr-sampler package.
