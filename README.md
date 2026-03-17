# llama-preset-proxy

A fast, async HTTP proxy that sits in front of a [llama.cpp](https://github.com/ggerganov/llama.cpp)
server running in **router mode** (`--models-preset`).

## The problem it solves

`llama-server` router mode loads model aliases from an INI file. When two aliases point to the
**same physical model** (same `hf-repo`, `model`, etc.) the router would normally unload and
reload the weights on every alias switch, even though the underlying GGUF file is identical.

This proxy intercepts requests, detects when the requested alias shares a model identity with
the currently loaded alias, and **rewrites the request** to use the loaded alias while injecting
the target alias's sampling parameters. The backend never sees the unload/reload cycle.

Preset configuration is read directly from the backend's `/v1/models` endpoint at startup — no
separate INI file is required.

```
Client                Proxy (:8081)           llama-server (:8080)
  |                       |                          |
  |  POST model=alias-b   |                          |
  |---------------------->|  alias-b shares model    |
  |                       |  with loaded alias-a     |
  |                       |  POST model=alias-a      |
  |                       |  + alias-b params        |
  |                       |------------------------->|
  |                       |  (no reload)             |
  |        response       |        response          |
  |<----------------------|<-------------------------|
```

## Building

```bash
cargo build --release
# binary: target/release/llama-preset-proxy
```

## Usage

```
llama-preset-proxy [OPTIONS]
```

| Flag | Env var | Default | Description |
|------|---------|---------|-------------|
| `--listen-host <host>` | `LPP_LISTEN_HOST` | `127.0.0.1` | Address to listen on |
| `--listen-port <port>` | `LPP_LISTEN_PORT` | `8081` | Port to listen on |
| `--backend-url <url>` | `LPP_BACKEND_URL` | `http://127.0.0.1:8080/v1` | Backend API base URL including `/v1` (`http://` or `https://`) |
| `--verbose` | — | off | Raise the default log level to `debug` (per-request detail); overridden by `RUST_LOG` |
| `--cache-ttl-secs <secs>` | `LPP_CACHE_TTL_SECS` | `30` | How long a `/v1/models` probe result is cached before re-querying the backend |
| `--max-body-mb <mb>` | `LPP_MAX_BODY_MB` | `8` | Maximum request body size (MiB); requests exceeding this are rejected with 413 |
| `--request-timeout-secs <secs>` | `LPP_REQUEST_TIMEOUT_SECS` | `300` | Idle/total timeout for client connections and backend requests |
| `--max-concurrent <n>` | `LPP_MAX_CONCURRENT` | `256` | Max in-flight proxied requests; excess are shed immediately with 503. `0` disables the limit. Health/version endpoints are exempt |

Every flag can be set via its `LPP_*` environment variable (handy for systemd
`EnvironmentFile=` and containers). Precedence is the clap default: an explicit
CLI flag overrides the env var, which overrides the built-in default.

### Example

```bash
# Start llama-server in router mode
llama-server --models-preset presets.ini --port 8080

# Local backend (default)
llama-preset-proxy --listen-port 8081 --verbose

# Remote backend over HTTPS
llama-preset-proxy --backend-url https://llama.example.com:8080/v1 --verbose

# External OpenAI-compatible API
llama-preset-proxy --backend-url https://api.example.com/v1 --verbose
```

Point your OpenAI-compatible client at `http://127.0.0.1:8081/v1`.

## Preset format

Preset data is sourced from the `status.preset` field that `llama-server` exposes in its
`/v1/models` response. The proxy reads those INI snippets at startup; no separate file is needed.

Only the fields relevant to request interception are interpreted; all other fields are ignored.

### Model identity keys

Two presets are considered to use the same physical model when they share the same values for any
combination of these keys. Short-form aliases are normalised before comparison.

| INI key | Short forms | Description |
|---------|-------------|-------------|
| `model` | `m` | Local path to a GGUF file |
| `model-url` | `mu` | URL to a GGUF file |
| `hf-repo` | `hf`, `hfr` | HuggingFace repository |
| `hf-file` | `hff` | HuggingFace filename |

### Sampling parameters

These INI keys are injected into the request body if the client did not already set them.

| INI key | JSON body key |
|---------|---------------|
| `temp` | `temperature` |
| `top-k` | `top_k` |
| `top-p` | `top_p` |
| `min-p` | `min_p` |
| `repeat-penalty` | `repeat_penalty` |
| `presence-penalty` | `presence_penalty` |
| `frequency-penalty` | `frequency_penalty` |
| `repeat-last-n` | `repeat_last_n` |
| `top-nsigma` | `top_n_sigma` |
| `xtc-probability` | `xtc_probability` |
| `xtc-threshold` | `xtc_threshold` |
| `typical` | `typical_p` |
| `dry-multiplier` | `dry_multiplier` |
| `dry-base` | `dry_base` |
| `dry-allowed-length` | `dry_allowed_length` |
| `dry-penalty-last-n` | `dry_penalty_last_n` |
| `mirostat` | `mirostat` |
| `mirostat-tau` | `mirostat_tau` |
| `mirostat-eta` | `mirostat_eta` |
| `seed` | `seed` |

### Special keys

| Key | Values | Effect |
|-----|--------|--------|
| `reasoning` | `on` / `off` / `auto` | Sets `chat_template_kwargs.enable_thinking` to `true` or `false`. `auto` (default) leaves the field unset, letting the client or model template decide. |

## How it works

1. **Startup**: the proxy fetches `/v1/models` from the backend, parses the `status.preset` INI
   snippet embedded in each entry, and computes a stable **model identity string** for each
   (e.g. `hf-repo=org/model:Q4_K_M`). If the backend is not reachable at startup the proxy still
   starts (it does **not** exit) and a background task retries the load every `--cache-ttl-secs`
   until the backend answers, so the proxy and backend can be co-started in any order.

2. **Tracker**: an in-memory cache of `model_identity → (active alias, timestamp)` is maintained
   across requests. Entries are considered fresh for `--cache-ttl-secs` seconds (default 30). On
   a cache miss or stale entry the proxy re-fetches `/v1/models` (typically < 1 ms on loopback),
   refreshes both the tracker and the preset map from the response, then returns the result.
   This means the proxy self-corrects automatically if llama-server is restarted with an updated
   INI file or evicts a model independently (e.g. due to `sleep-idle-seconds`).

3. **Each POST request**:
   - If `model` matches a known alias with a non-empty model identity, and the tracker shows a
     *different* alias is currently active for the same model → **reroute**: rewrite `model` to
     the active alias and inject the requested alias's sampling params (client values win on
     conflict).
   - Otherwise → forward as-is and record this alias as the new active one
     only if the upstream request succeeds.

4. **All other requests** (GET, DELETE, etc.) are forwarded unchanged.

## Health & observability

- `GET /healthz` — liveness. Always returns `200 ok` while the process is running; never touches
  the backend. Use it for process/container liveness probes.
- `GET /readyz` — readiness. Returns `200 ready` only when presets have been loaded **and** the
  backend `/v1/models` endpoint is currently reachable; otherwise `503`. Use it to gate traffic
  behind a load balancer.
- `GET /version` — returns the running build version as JSON (`{"version":"0.1.0"}`).
- **Load shedding.** In-flight proxied requests are bounded by `--max-concurrent` (default 256).
  Once saturated, further requests are rejected immediately with a JSON `503` rather than queued,
  so a stalled backend can't pile up unbounded work. The health and version endpoints are exempt
  so probes keep answering under overload.
- **Logging** is structured via [`tracing`]. The default level is `info` (or `debug` with
  `--verbose`); set `RUST_LOG` (e.g. `RUST_LOG=llama_preset_proxy=debug,tower_http=info`) to
  override it entirely. Request/response spans are emitted by `tower-http`'s `TraceLayer`.
- Handler panics are caught and returned as a JSON `500` instead of dropping the connection.

[`tracing`]: https://docs.rs/tracing

## Deployment

### Container (podman)

A multi-stage [`Containerfile`](Containerfile) is included. The runtime image runs as a
non-root user, ships `ca-certificates` (for HTTPS backends), and has a `/healthz` `HEALTHCHECK`.

```bash
podman build -t llama-preset-proxy -f Containerfile .

# Backend reachable from the container (host gateway shown for rootless podman):
podman run --rm -p 8081:8081 \
  -e LPP_BACKEND_URL=http://host.containers.internal:8080/v1 \
  llama-preset-proxy
```

The image sets `LPP_LISTEN_HOST=0.0.0.0` so the proxy is reachable from outside the container
(the bare binary still defaults to `127.0.0.1`). Override any setting with `-e LPP_*=…`.

### systemd

A hardened unit is provided at [`packaging/llama-preset-proxy.service`](packaging/llama-preset-proxy.service).
It is configured entirely through `LPP_*` variables in an `EnvironmentFile`:

```bash
sudo install -m 0755 target/release/llama-preset-proxy /usr/local/bin/
sudo install -m 0644 packaging/llama-preset-proxy.service /etc/systemd/system/

# Create the environment file referenced by the unit:
sudo tee /etc/llama-preset-proxy.env >/dev/null <<'EOF'
LPP_LISTEN_HOST=127.0.0.1
LPP_LISTEN_PORT=8081
LPP_BACKEND_URL=http://127.0.0.1:8080/v1
LPP_MAX_CONCURRENT=256
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now llama-preset-proxy.service
```

The unit uses `DynamicUser=yes` plus syscall/namespace/filesystem hardening; verify with
`systemd-analyze verify` and inspect the exposure score with `systemd-analyze security
llama-preset-proxy.service`.

## Security notes

- **Localhost by default.** Binds to `127.0.0.1`. Using `--listen-host 0.0.0.0` exposes the
  proxy to the network: any client can reach the backend through it, and there is no
  authentication. Do not use this without a firewall.
- **TLS.** The proxy→backend connection supports HTTPS via `--backend-url https://...` (backed
  by the system's native TLS). The client→proxy connection is always plain HTTP; place a
  TLS-terminating reverse proxy (nginx, Caddy, etc.) in front if the proxy itself must be
  reachable over HTTPS.
- **Request body size** is capped at 8 MiB by default, configurable with `--max-body-mb`.
- **Request timeout** — client connections are dropped after `--request-timeout-secs` seconds
  (default 300), preventing slow clients from holding connections indefinitely.
- **Concurrency cap** — in-flight proxied requests are bounded by `--max-concurrent` (default
  256); excess requests are shed with `503` rather than queued unboundedly.
- Error responses use JSON (`{"error": "..."}`) consistent with the llama-server API.
- Hop-by-hop headers (`Connection`, `Transfer-Encoding`, etc.) and `Host` are stripped before
  forwarding; `Content-Length` is always recomputed from the actual (potentially rewritten) body.
