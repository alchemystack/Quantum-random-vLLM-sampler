# Deployment Profile Template

This is a template for creating new deployment profiles. Copy this folder
and customize it for your entropy source.

## Quick start

1. Copy this template:

   ```bash
   cp -r deployments/_template deployments/my-server
   ```

2. Edit `deployments/my-server/.env`:
   - Set `QR_GRPC_SERVER_ADDRESS` to your server's address.
   - Set `QR_GRPC_METHOD_PATH` to match your server's proto service/method.
   - Set `QR_GRPC_STREAM_METHOD_PATH` (or leave empty to disable streaming).
   - Add `QR_GRPC_API_KEY` if your server requires authentication.

3. Run vLLM with your profile:

   ```bash
   cd examples/docker
   docker compose --env-file ../../deployments/my-server/.env up --build
   ```

## Adding an entropy server container

If your entropy server should run as a Docker container alongside vLLM, create
a `docker-compose.override.yml` in your profile folder. See
`deployments/urandom/docker-compose.override.yml` for an example.

Then run with:

```bash
cd examples/docker
docker compose \
  -f docker-compose.yml \
  -f ../../deployments/my-server/docker-compose.override.yml \
  --env-file ../../deployments/my-server/.env \
  up --build
```

## Finding the right gRPC method path

The method path format is `/<package>.<Service>/<Method>` from your `.proto`
file. For example:

```protobuf
package qr_entropy;            // package = "qr_entropy"
service EntropyService {        // service = "EntropyService"
  rpc GetEntropy (...) ...;     // method  = "GetEntropy"
}
```

Produces the path: `/qr_entropy.EntropyService/GetEntropy`

The only requirement is that:
- The **request** has the byte count as protobuf **field 1** (varint).
- The **response** has the random bytes as protobuf **field 1** (length-delimited bytes).

Any proto definition following this convention works without code changes.
