# Deployment Profiles

A deployment profile configures qr-sampler to connect to a specific entropy
source. Each profile is a folder containing:

- **`.env`** -- Environment variables for qr-sampler and vLLM configuration.
- **`docker-compose.override.yml`** (optional) -- Adds extra services, e.g. a
  co-located entropy server.
- **`README.md`** (optional) -- Setup notes specific to this profile.

The main Docker Compose file (`examples/docker/docker-compose.yml`) runs vLLM
with qr-sampler using system entropy by default. Profiles override only the
entropy-related settings -- everything else stays the same.

## Available profiles

| Profile | Entropy source | Description |
|---------|---------------|-------------|
| `_template/` | -- | Annotated template. Copy this to create your own. |
| `urandom/` | `os.urandom()` via gRPC | Separate urandom gRPC server (good starting point). |
| `firefly-1/` | Quantum RNG via gRPC | External QRNG server with API key authentication. |

## Usage

### No profile (system entropy)

The simplest setup. vLLM uses `os.urandom()` directly -- no external server.

```bash
cd examples/docker
docker compose up --build
```

### With a profile (external entropy server)

Pass the profile's `.env` file to override entropy settings:

```bash
cd examples/docker
docker compose --env-file ../../deployments/firefly-1/.env up --build
```

### With a profile that adds services

Some profiles include a `docker-compose.override.yml` that adds an entropy
server container. Use `-f` to merge it:

```bash
cd examples/docker
docker compose \
  -f docker-compose.yml \
  -f ../../deployments/urandom/docker-compose.override.yml \
  --env-file ../../deployments/urandom/.env \
  up --build
```

## Creating your own profile

1. Copy the template:

   ```bash
   cp -r deployments/_template deployments/my-server
   ```

2. Edit `deployments/my-server/.env` -- fill in your server address, gRPC
   method paths, and authentication settings.

3. (Optional) Add a `docker-compose.override.yml` if your entropy server should
   run as a container alongside vLLM.

4. Run:

   ```bash
   cd examples/docker
   docker compose --env-file ../../deployments/my-server/.env up --build
   ```

## Security notes

- Profile `.env` files may contain API keys or other credentials.
- The `_template/` and `urandom/` profiles contain **no secrets** and are
  always safe to commit.
- If your profile contains credentials you do not want in version control,
  add its folder name to `deployments/.gitignore`.
- qr-sampler never logs the `QR_GRPC_API_KEY` value. Health checks report
  only whether authentication is enabled (`"authenticated": true/false`).
