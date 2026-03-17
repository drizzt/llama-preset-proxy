# Multi-stage build for llama-preset-proxy.
# Build with podman (rootless-friendly):
#   podman build -t llama-preset-proxy -f Containerfile .

# ---- builder ----------------------------------------------------------------
# Full rust image (buildpack-deps based) ships gcc/cmake/perl, needed by the
# TLS stack (aws-lc-sys) pulled in transitively via reqwest. Pinned to the MSRV.
FROM docker.io/library/rust:1.88-bookworm AS builder
WORKDIR /build

# Cache dependency compilation separately from source changes.
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked

# ---- runtime ----------------------------------------------------------------
FROM docker.io/library/debian:bookworm-slim AS runtime

# ca-certificates: verify HTTPS backends (--backend-url https://…).
# curl: used by the container HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user.
RUN useradd --system --no-create-home --uid 10001 proxy
USER proxy

COPY --from=builder /build/target/release/llama-preset-proxy /usr/local/bin/llama-preset-proxy

# Inside a container the loopback default is useless; listen on all interfaces.
# Override any LPP_* setting at run time (-e LPP_LISTEN_HOST=…). The bare binary
# default stays 127.0.0.1 — only this image changes it.
ENV LPP_LISTEN_HOST=0.0.0.0
EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:8081/healthz || exit 1

ENTRYPOINT ["/usr/local/bin/llama-preset-proxy"]
