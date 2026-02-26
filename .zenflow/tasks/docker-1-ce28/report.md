# Implementation Report: Per-Source Docker Setup & Documentation Restructure

## Summary

Restructured Docker deployment from a single centralized `docker-compose.yml` to self-contained per-entropy-source compose files. Updated the default model to `Qwen/Qwen2.5-1.5B-Instruct` with `--dtype half --gpu-memory-utilization 0.90`. Rewrote documentation to prioritize external entropy sources for consciousness research.

## Changes (14 files, +574 / -287 lines)

### Files Created (6)

| File | Purpose |
|------|---------|
| `deployments/urandom/docker-compose.yml` | Self-contained compose: vLLM + entropy-server services |
| `deployments/urandom/.env.example` | Annotated env template for urandom profile |
| `deployments/firefly-1/docker-compose.yml` | Self-contained compose: vLLM only (firefly-1 is external) |
| `deployments/firefly-1/.env.example` | Annotated env template for firefly-1 profile |
| `deployments/_template/docker-compose.yml` | Annotated compose template for new profiles |
| `deployments/_template/.env.example` | Annotated env template for new profiles |

### Files Modified (6)

| File | Changes |
|------|---------|
| `examples/docker/Dockerfile.vllm` | `HF_MODEL` default changed to `Qwen/Qwen2.5-1.5B-Instruct`; CMD now includes `--dtype half --gpu-memory-utilization 0.90`; header comments updated |
| `README.md` | Major restructure: external entropy Docker setup is primary quickstart; system entropy deprioritized as fallback; all model references updated; Docker commands simplified to `cd deployments/<profile> && docker compose up --build` |
| `deployments/README.md` | Updated to describe new self-contained per-source pattern with links to each profile |
| `deployments/_template/README.md` | Simplified to 3-step quickstart |
| `deployments/urandom/README.md` | Self-contained 3-step setup guide |
| `deployments/firefly-1/README.md` | Self-contained 3-step setup guide |

### Files Deleted (2)

| File | Reason |
|------|--------|
| `examples/docker/docker-compose.yml` | Replaced by per-source compose files in `deployments/` |
| `deployments/urandom/docker-compose.override.yml` | Replaced by self-contained `deployments/urandom/docker-compose.yml` |

## Verification Results

### 1. Old model references (`meta-llama/Llama-3.2-1B`)

**Result: PASS** -- Zero matches in source code and documentation. Only matches are in `.zenflow/tasks/` artifacts (spec.md, plan.md) which are task metadata, not deliverables.

### 2. Docker Compose YAML validation

| File | Status |
|------|--------|
| `deployments/urandom/docker-compose.yml` | VALID |
| `deployments/firefly-1/docker-compose.yml` | VALID |
| `deployments/_template/docker-compose.yml` | VALID |

All three files parse successfully with Python's `yaml.safe_load()`.

### 3. README file reference check

**Result: PASS** -- All file paths, directory references, and markdown links in all 5 README files point to files/directories that exist on disk. No references to deleted files (`examples/docker/docker-compose.yml`, `deployments/urandom/docker-compose.override.yml`).

Verified READMEs:
- `README.md` (main)
- `deployments/README.md`
- `deployments/_template/README.md`
- `deployments/urandom/README.md`
- `deployments/firefly-1/README.md`

### 4. Ruff check

**Result: N/A** -- No Python files were modified in this task. Changes were limited to Dockerfiles, YAML compose files, `.env.example` templates, and Markdown documentation.

### 5. Model and flag consistency

All deployment artifacts consistently use:
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Flags:** `--dtype half --gpu-memory-utilization 0.90`

Verified across: Dockerfile.vllm, all 3 docker-compose.yml files, all 3 .env.example files, main README.md, and all deployment READMEs.

## Architecture

The new deployment pattern:

```
deployments/<entropy-source>/
  docker-compose.yml    # Self-contained, includes all services needed
  .env.example          # Copy to .env and customize
  README.md             # 3-step setup guide
```

Each compose file:
- References shared Dockerfiles via `../../examples/docker/Dockerfile.{vllm,entropy-server}`
- Uses `${VAR:-default}` syntax for all config values
- Includes complete QR_* environment variable blocks (not overrides)
- Sets build context to project root (`../..`) for access to source code

User workflow reduced to:
```bash
cd deployments/urandom
cp .env.example .env    # edit if needed
docker compose up --build
```

## Commits

| Hash | Description |
|------|-------------|
| `76ef5dc` | Update Dockerfile.vllm defaults and create per-source Docker Compose files |
| `915f05c` | Cleanup script changes (post-step-2) |
| `8f61514` | Restructure main README |
| `9d8f552` | Cleanup script changes (post-step-4) |
