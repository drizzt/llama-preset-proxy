// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Timothy Redaelli

mod ini;
mod preset;
mod proxy;

use axum::http::StatusCode as AxumStatusCode;
use axum::{
    Router,
    body::Body,
    middleware::from_fn_with_state,
    response::Response,
    routing::{any, get},
};
use clap::Parser;
use preset::{Preset, presets_from_models_json};
use proxy::{
    AppState, backend_root_of, fetch_models_data, healthz, load_shed, proxy_handler, readyz,
    version, web_passthrough_handler, write_recover,
};
use reqwest::Client;
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, RwLock},
    time::Duration,
};
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use tower_http::{catch_panic::CatchPanicLayer, timeout::TimeoutLayer, trace::TraceLayer};
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

// ---------------------------------------------------------------------------
// CLI Arguments
// ---------------------------------------------------------------------------
#[derive(Parser, Debug)]
#[command(about = "llama.cpp preset injection proxy for router-mode servers")]
struct Args {
    #[arg(
        long,
        env = "LPP_LISTEN_HOST",
        default_value = "127.0.0.1",
        help = "Address to listen on"
    )]
    listen_host: String,

    #[arg(
        long,
        env = "LPP_LISTEN_PORT",
        default_value_t = 8081,
        help = "Port to listen on"
    )]
    listen_port: u16,

    #[arg(
        long,
        env = "LPP_BACKEND_URL",
        default_value = "http://127.0.0.1:8080/v1",
        help = "Backend API base URL including the /v1 path (http:// or https://)"
    )]
    backend_url: String,

    #[arg(
        long,
        help = "Raise the default log level to debug (per-request detail); overridden by RUST_LOG"
    )]
    verbose: bool,

    #[arg(
        long,
        env = "LPP_CACHE_TTL_SECS",
        default_value_t = 30,
        help = "Seconds a /v1/models probe result is cached before re-querying the backend"
    )]
    cache_ttl_secs: u64,

    #[arg(
        long,
        env = "LPP_MAX_BODY_MB",
        default_value_t = 8,
        help = "Maximum request body size in MiB; requests exceeding this are rejected with 413"
    )]
    max_body_mb: usize,

    #[arg(
        long,
        env = "LPP_REQUEST_TIMEOUT_SECS",
        default_value_t = 300,
        help = "Connect timeout and per-read idle timeout (seconds) for backend requests; streaming responses are not capped while data keeps arriving"
    )]
    request_timeout_secs: u64,

    #[arg(
        long,
        env = "LPP_MAX_CONCURRENT",
        default_value_t = 256,
        help = "Maximum number of in-flight proxied requests; excess requests are shed immediately with 503. 0 disables the limit. Health endpoints are exempt."
    )]
    max_concurrent: usize,
}

// ---------------------------------------------------------------------------
// Graceful shutdown signal (SIGINT + SIGTERM)
// ---------------------------------------------------------------------------
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

// ---------------------------------------------------------------------------
// Load presets from the backend /v1/models endpoint
// ---------------------------------------------------------------------------
async fn load_presets_from_backend(
    client: &Client,
    backend_url: &str,
) -> Result<HashMap<String, Preset>, String> {
    let data = fetch_models_data(client, backend_url, Duration::from_secs(10)).await?;
    Ok(presets_from_models_json(&data))
}

// ---------------------------------------------------------------------------
// Panic handler for the CatchPanicLayer: a panicking handler becomes a JSON 500
// instead of a silently dropped connection.
// ---------------------------------------------------------------------------
fn handle_panic(_err: Box<dyn std::any::Any + Send + 'static>) -> Response<Body> {
    error!("handler panicked; returning 500");
    proxy::json_error(
        AxumStatusCode::INTERNAL_SERVER_ERROR,
        "internal server error",
    )
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Initialise structured logging first so all subsequent diagnostics are
    // captured. The default level follows --verbose (debug vs info); RUST_LOG
    // overrides it entirely when set.
    let default_level = if args.verbose { "debug" } else { "info" };
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    if !args.backend_url.starts_with("http://") && !args.backend_url.starts_with("https://") {
        error!("--backend-url must start with http:// or https://");
        std::process::exit(1);
    }
    // Strip any trailing slash so path concatenation is always consistent.
    let backend_url = args.backend_url.trim_end_matches('/').to_string();

    let client = match Client::builder()
        // Use a connect timeout plus a per-read (idle) timeout rather than a total
        // request timeout: reqwest's `timeout` is a total deadline that runs until
        // the response body finishes, so it would abort a long streaming completion
        // mid-response. `read_timeout` resets on each chunk, bounding stalls without
        // capping a healthy stream.
        .connect_timeout(Duration::from_secs(args.request_timeout_secs))
        .read_timeout(Duration::from_secs(args.request_timeout_secs))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            error!("failed to build HTTP client: {}", e);
            std::process::exit(1);
        }
    };

    // Try an initial preset load, but never make it fatal: when the proxy and the
    // backend are co-started (compose/systemd), the backend may not be up yet.
    // On failure we start with an empty preset map and a background task keeps
    // retrying until the backend answers (see below). Once populated, the proxy's
    // own slow-path /v1/models probe keeps presets fresh thereafter.
    let presets = match load_presets_from_backend(&client, &backend_url).await {
        Ok(p) => {
            info!("Loaded {} preset(s) from {}:", p.len(), backend_url);
            for preset in p.values() {
                info!(
                    "  Preset({:?} [{}]: params={:?})",
                    preset.name,
                    if preset.model_id.is_empty() {
                        "no-id"
                    } else {
                        &preset.model_id
                    },
                    preset.params.keys().collect::<Vec<_>>()
                );
            }
            p
        }
        Err(e) => {
            warn!("Initial preset load failed: {e}; starting empty and retrying in background");
            HashMap::new()
        }
    };
    let need_background_load = presets.is_empty();

    let tracker = Arc::new(RwLock::new(HashMap::new()));

    let state = AppState {
        presets: Arc::new(RwLock::new(presets)),
        tracker,
        last_probe: Arc::new(AsyncMutex::new(())),
        backend_url: backend_url.clone(),
        // Backend origin without the `/v1` API suffix: every non-`/v1` request
        // (web UI assets, native llama.cpp endpoints) is reverse-proxied here.
        backend_root: backend_root_of(&backend_url).to_string(),
        cache_ttl: Duration::from_secs(args.cache_ttl_secs),
        max_body_bytes: args.max_body_mb.saturating_mul(1024 * 1024),
        client,
    };

    // Background retry loop: when the initial load found no presets (backend was
    // down or empty at boot), keep polling until it returns some. Request-driven
    // refresh alone can't bootstrap an empty map — intercept_post looks the alias
    // up *before* probing, so an empty map would just passthrough forever.
    if need_background_load {
        let presets = state.presets.clone();
        let client = state.client.clone();
        let backend_url = backend_url.clone();
        // Never busy-loop if the cache is disabled (ttl == 0).
        let retry_delay = state.cache_ttl.max(Duration::from_secs(1));
        tokio::spawn(async move {
            loop {
                match load_presets_from_backend(&client, &backend_url).await {
                    Ok(p) if !p.is_empty() => {
                        let n = p.len();
                        *write_recover(&presets, "presets") = p;
                        info!("Background load succeeded: {n} preset(s) now available");
                        break;
                    }
                    Ok(_) => warn!(
                        "Backend reachable but returned no presets; retrying in {}s",
                        retry_delay.as_secs()
                    ),
                    Err(e) => warn!(
                        "Background preset load failed: {e}; retrying in {}s",
                        retry_delay.as_secs()
                    ),
                }
                tokio::time::sleep(retry_delay).await;
            }
        });
    }

    // The proxy route carries a bounded-concurrency load-shed middleware backed
    // by a shared semaphore: a request that can't immediately acquire a permit
    // is rejected with 503 instead of queueing unboundedly behind a stalled
    // backend. The health/version routes are added *after* this layer so they
    // stay exempt — liveness/readiness must answer even under overload.
    // The fallback reverse-proxies every non-`/v1` path to the backend root so
    // the llama.cpp web UI (and native endpoints) load on the proxy port. It is
    // attached here, before the load-shed layer, so it inherits the concurrency
    // limit; the explicit health/version routes added afterwards still match
    // first and stay exempt.
    let proxy_routes = Router::new()
        .route("/v1/{*path}", any(proxy_handler))
        .fallback(web_passthrough_handler);
    let proxy_routes = if args.max_concurrent > 0 {
        let permits = Arc::new(Semaphore::new(args.max_concurrent));
        proxy_routes.layer(from_fn_with_state(permits, load_shed))
    } else {
        proxy_routes
    };

    let app = proxy_routes
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/version", get(version))
        .layer(TimeoutLayer::with_status_code(
            AxumStatusCode::GATEWAY_TIMEOUT,
            Duration::from_secs(args.request_timeout_secs),
        ))
        .layer(TraceLayer::new_for_http())
        .layer(CatchPanicLayer::custom(handle_panic))
        .with_state(state);

    let addr_str = format!("{}:{}", args.listen_host, args.listen_port);
    let addr: SocketAddr = match addr_str.parse() {
        Ok(a) => a,
        Err(e) => {
            error!("invalid listen address {:?}: {}", addr_str, e);
            std::process::exit(1);
        }
    };

    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            error!("failed to bind to {}: {}", addr, e);
            std::process::exit(1);
        }
    };

    info!("Listening on {} → {}", addr_str, backend_url);

    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(async {
            shutdown_signal().await;
            info!("shutting down");
        })
        .await
    {
        error!("Server error: {}", e);
        std::process::exit(1);
    }
}
