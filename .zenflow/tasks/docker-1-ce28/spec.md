# Technical Specification: Per-Source Docker Setup & Documentation Restructure

## Difficulty: Medium

The changes are mostly file reorganization and documentation rewriting. No core logic changes. However, the scope is broad — touches Docker files, deployment profiles, README, example servers, and model defaults across many files.

---

## Technical Context

- **Language:** Python 3.10+, YAML (Docker Compose), Dockerfile
- **Key dependencies:** vLLM, gRPC, pydantic-settings
- **Build system:** Docker Compose with multi-file overrides
- **Current architecture:** One centralized `examples/docker/docker-compose.yml` + deployment profile `.env` overrides from `deployments/`

---

## Problem Statement

The current Docker setup requires users to:
1. Navigate to `examples/docker/`
2. Know about the `deployments/` directory and its profile system
3. Construct complex `docker compose` commands with `-f` overrides and `--env-file` flags pointing to relative paths like `../../deployments/urandom/.env`

This is confusing. Users interested in a specific entropy source (e.g., QRNG hardware) must piece together files from multiple directories. The task requires restructuring so each entropy source has a **self-contained Docker setup** clearly placed alongside its configuration.

Additionally:
- The default model `meta-llama/Llama-3.2-1B` must change to `Qwen/Qwen2.5-1.5B-Instruct`
- The README should prioritize external entropy sources over system entropy (which is "useless for consciousness studies")
- Docker commands should include `dtype=half` and `gpu_memory_utilization=0.90`

---

## Implementation Approach

### Core change: Merge `deployments/` profiles into self-contained source directories

Each entropy source directory (currently under `deployments/`) gets its own complete `docker-compose.yml` (not an override file) and a setup guide. The shared Dockerfiles (`Dockerfile.vllm`, `Dockerfile.entropy-server`) stay in `examples/docker/` as they are reusable build recipes.

### New directory structure

```
deployments/
├── README.md                          # Updated overview
├── .gitignore                         # Unchanged
├── _template/
│   ├── README.md                      # Updated setup guide
│   ├── .env.example                   # NEW: Example .env (was implicit)
│   └── docker-compose.yml             # NEW: Self-contained compose file
├── urandom/
│   ├── README.md                      # Updated setup guide
│   ├── .env.example                   # NEW: Example .env with defaults
│   └── docker-compose.yml             # REPLACES docker-compose.override.yml
│                                      #   (now self-contained, includes both
│                                      #    vllm + entropy-server services)
├── firefly-1/
│   ├── README.md                      # Updated setup guide
│   ├── .env.example                   # NEW: Example .env (sanitized, no real API key)
│   └── docker-compose.yml             # NEW: Self-contained compose (vllm only,
│                                      #   since firefly-1 is external)
```

### What changes per directory

**Each deployment profile's `docker-compose.yml`:**
- Is **self-contained** — includes the vllm service and any co-located entropy server
- References Dockerfiles via relative paths to `../../examples/docker/`
- Includes all relevant QR_* environment variables inline (not just overrides)
- Uses `${VAR:-default}` syntax so it works with or without a `.env` file
- Default model: `Qwen/Qwen2.5-1.5B-Instruct`

**Each deployment profile's `.env.example`:**
- Provides a copy-and-edit template for the user's actual `.env`
- Documents every variable with comments
- Users copy to `.env` and customize

**Each deployment profile's `README.md`:**
- Self-contained setup guide: "copy .env.example to .env, edit, docker compose up"
- No references to other directories for compose files
- Simple 3-step quickstart

### What happens to `examples/docker/`

The centralized `docker-compose.yml` in `examples/docker/` is **removed** since each deployment profile now has its own. The Dockerfiles stay:

```
examples/docker/
├── Dockerfile.vllm                    # MODIFIED: Default model → Qwen/Qwen2.5-1.5B-Instruct
├── Dockerfile.entropy-server          # Unchanged
```

The old `examples/docker/docker-compose.yml` is deleted — it was the "one compose to rule them all" that we're replacing with per-source compose files.

### Model default changes

All occurrences of `meta-llama/Llama-3.2-1B` change to `Qwen/Qwen2.5-1.5B-Instruct`. Additionally, vLLM serve commands gain `--dtype half --gpu-memory-utilization 0.90` flags. Affected files:

1. `examples/docker/Dockerfile.vllm` — `ENV HF_MODEL` and `CMD` line
2. `README.md` — Quick start examples, curl examples, Docker examples
3. All new `docker-compose.yml` files in deployment profiles
4. All new `.env.example` files

### README restructure

The main `README.md` is restructured to:

1. **Lead with external entropy sources** — the "Quick start" section shows the QRNG/urandom Docker setup first
2. **Deprioritize system entropy** — mention it as a fallback/testing mode, not the primary path
3. **Simplify Docker instructions** — "cd deployments/urandom && docker compose up" instead of complex multi-file commands
4. **Update all model references** to `Qwen/Qwen2.5-1.5B-Instruct`
5. **Update all curl examples** with the new model name

---

## Source Code Structure Changes

### Files to CREATE

| File | Purpose |
|------|---------|
| `deployments/_template/docker-compose.yml` | Self-contained compose template |
| `deployments/_template/.env.example` | Annotated env var template |
| `deployments/urandom/docker-compose.yml` | Self-contained compose for urandom (vllm + entropy-server) |
| `deployments/urandom/.env.example` | Env vars for urandom profile |
| `deployments/firefly-1/docker-compose.yml` | Self-contained compose for firefly-1 (vllm only) |
| `deployments/firefly-1/.env.example` | Env vars for firefly-1 profile (sanitized) |

### Files to MODIFY

| File | Changes |
|------|---------|
| `examples/docker/Dockerfile.vllm` | Default model → `Qwen/Qwen2.5-1.5B-Instruct`, add `--dtype half --gpu-memory-utilization 0.90` to CMD |
| `README.md` | Major restructure: lead with external entropy, update model, simplify Docker instructions |
| `deployments/README.md` | Update to reflect new self-contained pattern |
| `deployments/_template/README.md` | Simplify: just "cp .env.example .env, edit, docker compose up" |
| `deployments/urandom/README.md` | Simplify: self-contained setup guide |
| `deployments/firefly-1/README.md` | Simplify: self-contained setup guide |

### Files to DELETE

| File | Reason |
|------|--------|
| `examples/docker/docker-compose.yml` | Replaced by per-source compose files |
| `deployments/urandom/docker-compose.override.yml` | Replaced by self-contained `docker-compose.yml` |

---

## Key Design Decisions

1. **Dockerfiles stay shared in `examples/docker/`** — they are build recipes, not deployment config. Each profile's `docker-compose.yml` references them via `../../examples/docker/Dockerfile.vllm`. This avoids duplicating Dockerfiles.

2. **`.env.example` instead of `.env`** — actual `.env` files may contain secrets (API keys). We provide `.env.example` files that users copy and customize. The `.gitignore` already handles ignoring real `.env` files.

3. **Self-contained compose files include all QR_* vars** — each profile's `docker-compose.yml` lists all relevant environment variables with defaults, so users can see the full configuration in one file without cross-referencing.

4. **`docker-compose.yml` context paths** — since compose files are now in `deployments/<profile>/`, the build context needs to be `../..` (project root) and dockerfile paths need to be `examples/docker/Dockerfile.vllm`.

5. **vLLM CMD includes dtype and gpu_memory_utilization** — these are added as environment variables (`VLLM_DTYPE`, `VLLM_GPU_MEM`) with defaults in the Dockerfile CMD, matching the user's requested defaults.

---

## Verification Approach

1. **Docker Compose config validation:** `docker compose -f deployments/urandom/docker-compose.yml config` for each profile — verifies YAML syntax and variable interpolation
2. **README link check:** Verify all markdown links point to files that exist
3. **Model name grep:** Confirm no remaining references to `meta-llama/Llama-3.2-1B`
4. **No code changes to test:** This is purely infrastructure/documentation — no Python tests to run
5. **Git diff review:** Ensure no files are accidentally deleted or broken

---

## Out of Scope

- No changes to Python source code (`src/qr_sampler/`)
- No changes to tests
- No changes to `pyproject.toml`
- No changes to example server scripts
- No changes to systemd files
- No new entropy source implementations
