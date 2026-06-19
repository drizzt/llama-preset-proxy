// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Timothy Redaelli

//! Async HTTP proxy handler.
//!
//! [`proxy_handler`] is the single Axum route handler for all incoming
//! requests. For `POST` requests it inspects the JSON body: if the requested
//! model alias shares a physical model with the currently-loaded alias (per
//! the tracker cache), the request is **rerouted** — the `model` field is
//! rewritten to the active alias and the target alias's sampling parameters
//! are injected (client-supplied values always take precedence). All other
//! requests, and `POST` requests whose alias is unknown or has no model
//! identity, are forwarded unchanged.
//!
//! The active alias for each model identity is stored in a TTL cache backed
//! by the backend's `/v1/models` endpoint. On a cache miss or stale entry the
//! endpoint is queried (typically < 1 ms on loopback) to discover which alias
//! is currently loaded, then both the tracker and the preset map are refreshed
//! from the response. This ensures that changes to the backend's INI file
//! (followed by a server restart) are picked up automatically within one TTL.

use crate::preset::{Preset, presets_from_models_json};
use axum::{
    body::{Body, Bytes},
    extract::{Request, State},
    http::{Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use reqwest::Client;
use serde_json::Value;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use tracing::{debug, error, info, warn};

/// Shared state threaded through every Axum handler via [`axum::extract::State`].
#[derive(Clone)]
pub(crate) struct AppState {
    pub(crate) presets: Arc<RwLock<HashMap<String, Preset>>>,
    /// Maps `model_id` to `(active alias, time the entry was last refreshed)`.
    pub(crate) tracker: Arc<RwLock<HashMap<String, (String, Instant)>>>,
    /// Global lock that serialises backend `/v1/models` probes. Holding this
    /// while performing a fetch prevents thundering-herd (multiple concurrent
    /// requests for stale entries all hitting the backend simultaneously) and
    /// limits amplification to at most one in-flight probe at a time. Waiters
    /// re-check the tracker cache after acquiring the lock, so concurrent
    /// callers for the *same* model_id are coalesced into a single fetch.
    /// (Callers for distinct model_ids each probe in turn, since a probe writes
    /// only the requested model_id's tracker entry — the preset map, however,
    /// is rebuilt in full on every probe.)
    pub(crate) last_probe: Arc<AsyncMutex<()>>,
    /// Backend API base URL including the `/v1` (or equivalent) path component,
    /// e.g. `http://127.0.0.1:8080/v1` or `https://api.example.com/v1`.
    /// The proxy only routes `/v1/*` requests and strips the `/v1` prefix
    /// before appending the remaining path to this URL.
    pub(crate) backend_url: String,
    /// How long a tracker entry is considered fresh before re-querying the backend.
    /// A value of [`Duration::ZERO`] disables the cache entirely (every request
    /// queries `/v1/models`), which is correct but expensive.
    pub(crate) cache_ttl: Duration,
    /// Maximum request body size in bytes; bodies exceeding this are rejected.
    pub(crate) max_body_bytes: usize,
    pub(crate) client: Client,
}

/// Headers that must never be forwarded verbatim between client and backend.
fn is_skip_header(k: &str) -> bool {
    matches!(
        k,
        "content-length"    // recomputed from actual body bytes after any rewrite
            | "host"        // reqwest sets the correct backend host automatically
            | "transfer-encoding"
            | "connection"
            | "keep-alive"
            | "proxy-connection"
            | "te"
            | "trailers"
            | "upgrade"
    )
}

/// Build a JSON response from a status and an already-serialised JSON body,
/// falling back to a bare 500 if the builder rejects the parts.
fn json_response(status: StatusCode, body: String) -> Response {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(body))
        // Static status + const header name → builder cannot fail.
        .expect("static response parts are always valid")
}

pub(crate) fn json_error(status: StatusCode, msg: &str) -> Response {
    json_response(status, serde_json::json!({"error": msg}).to_string())
}

/// Acquire a read lock, recovering the guard if the lock was poisoned by a
/// panicking thread (the cached data is still valid to read after a panic).
pub(crate) fn read_recover<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|p| p.into_inner())
}

/// Acquire a write lock, recovering (and logging) if the lock was poisoned.
pub(crate) fn write_recover<'a, T>(
    lock: &'a RwLock<T>,
    what: &str,
) -> std::sync::RwLockWriteGuard<'a, T> {
    lock.write().unwrap_or_else(|p| {
        error!("{what} RwLock poisoned (a thread panicked while holding it), recovering");
        p.into_inner()
    })
}

/// Liveness probe: the process is up and serving. Always 200; never touches the
/// backend or any shared state, so an orchestrator can distinguish "proxy alive"
/// from "backend reachable" (the latter is [`readyz`]).
pub(crate) async fn healthz() -> Response {
    (StatusCode::OK, "ok").into_response()
}

/// Version probe: returns the compiled crate version as JSON so operators can
/// confirm which build is running without inspecting the binary. Touches no
/// state and is exempt from the concurrency limit.
pub(crate) async fn version() -> Response {
    json_response(
        StatusCode::OK,
        serde_json::json!({"version": env!("CARGO_PKG_VERSION")}).to_string(),
    )
}

/// Bounded-concurrency load-shed middleware. Each in-flight request must hold a
/// permit from the shared semaphore for its duration; when none is available the
/// request is shed immediately with a JSON `503` rather than queued. The permit
/// is released when the response future resolves (i.e. once the backend's
/// response head is in hand), bounding the number of concurrent backend
/// round-trips a flood or stalled backend can pin open.
pub(crate) async fn load_shed(
    State(permits): State<Arc<Semaphore>>,
    req: Request,
    next: Next,
) -> Response {
    match permits.try_acquire() {
        Ok(_permit) => next.run(req).await,
        Err(_) => {
            warn!("request shed: concurrency limit reached; returning 503");
            json_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "server overloaded, retry later",
            )
        }
    }
}

/// Readiness probe: 200 only if presets have been loaded AND the backend
/// `/v1/models` endpoint is currently reachable; otherwise 503. Lets a load
/// balancer hold traffic until the proxy can actually serve rerouted requests.
pub(crate) async fn readyz(State(state): State<AppState>) -> Response {
    if read_recover(&state.presets).is_empty() {
        return json_error(StatusCode::SERVICE_UNAVAILABLE, "no presets loaded yet");
    }
    // Reuse the shared probe so the `/v1/models` URL + decode contract lives in
    // one place; a parseable `data` array is exactly "ready to serve".
    match fetch_models_data(&state.client, &state.backend_url, Duration::from_secs(5)).await {
        Ok(_) => (StatusCode::OK, "ready").into_response(),
        Err(e) => {
            debug!("readyz backend probe failed: {e}");
            json_error(StatusCode::SERVICE_UNAVAILABLE, "backend unreachable")
        }
    }
}

/// Outcome of a `/v1/models` probe, distinguishing a genuine refresh from a
/// transient failure so callers can serve last-known-good state on failure.
enum ProbeResult {
    /// Probe succeeded and the preset map was refreshed. Inner value is the
    /// alias currently loaded for the requested `model_id` (`None` if none is).
    Refreshed(Option<String>),
    /// The probe itself failed (network/parse/malformed); caches were left intact.
    Failed,
}

/// From the `data` array of a `/v1/models` response and a freshly-built preset
/// map, return the alias that is currently loaded for `model_id`.
///
/// If `preferred` is itself loaded it is returned immediately, preventing
/// unnecessary rerouting when the requested alias is already active.
/// Otherwise the first loaded alias found for `model_id` is returned.
fn find_active_alias(
    data: &[Value],
    presets: &HashMap<String, Preset>,
    model_id: &str,
    preferred: &str,
) -> Option<String> {
    let mut fallback: Option<String> = None;
    for entry in data {
        let Some(id) = entry.get("id").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(status_val) = entry
            .get("status")
            .and_then(|s| s.get("value"))
            .and_then(|v| v.as_str())
        else {
            continue;
        };
        if status_val == "loaded"
            && let Some(preset) = presets.get(id)
            && preset.model_id == model_id
        {
            // The preferred alias is already loaded — return it immediately so we
            // don't reroute when the requested alias is itself active.
            if id == preferred {
                return Some(preferred.to_string());
            }
            if fallback.is_none() {
                fallback = Some(id.to_string());
            }
        }
    }
    fallback
}

/// `GET {backend_url}/models` and return its `data` array.
///
/// Shared by the startup load and the per-request slow-path probe so the
/// endpoint URL, JSON decoding, and `data`-array contract live in one place.
/// The timeout is a parameter because startup can afford to wait longer than
/// the latency-sensitive request path.
pub(crate) async fn fetch_models_data(
    client: &Client,
    backend_url: &str,
    timeout: Duration,
) -> Result<Vec<Value>, String> {
    let url = format!("{}/models", backend_url);
    let resp = client
        .get(&url)
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| format!("/v1/models request to {url} failed ({e})"))?;
    let mut json: Value = resp
        .json()
        .await
        .map_err(|e| format!("/v1/models response was not valid JSON ({e})"))?;
    // Move the `data` array out instead of cloning it: this runs on the
    // latency-sensitive slow path and `json` is dropped right after.
    match json.get_mut("data").map(Value::take) {
        Some(Value::Array(data)) => Ok(data),
        _ => Err("/v1/models response missing 'data' array".to_string()),
    }
}

/// Fetch `/v1/models`, rebuild the preset map from fresh `status.preset`
/// snippets, commit it, and return the alias currently loaded for `model_id`.
async fn query_loaded_alias(state: &AppState, model_id: &str, preferred: &str) -> ProbeResult {
    let data =
        match fetch_models_data(&state.client, &state.backend_url, Duration::from_secs(5)).await {
            Ok(d) => d,
            // On any probe failure, log at debug and keep the cached state intact
            // (serving stale beats evicting last-known-good on a transient blip).
            Err(reason) => {
                debug!("{reason}; keeping cached state");
                return ProbeResult::Failed;
            }
        };

    let new_presets = presets_from_models_json(&data);
    let alias = find_active_alias(&data, &new_presets, model_id, preferred);

    // Commit the rebuilt preset map, but never overwrite a non-empty map with an
    // empty one: a backend restart can momentarily return entries with no
    // parseable `status.preset`, and silently dropping all presets would disable
    // injection for the rest of the TTL window.
    let mut g = write_recover(&state.presets, "presets");
    if new_presets.is_empty() && !g.is_empty() {
        warn!(
            "/v1/models returned no parseable presets; keeping previous {} preset(s)",
            g.len()
        );
    } else {
        *g = new_presets;
    }

    ProbeResult::Refreshed(alias)
}

/// Return the active alias for `model_id` from the tracker cache.
///
/// If the cached entry is missing or older than `cache_ttl`, the backend
/// `/v1/models` endpoint is queried, both the tracker and the preset map are
/// updated, and the result is returned.
///
/// Concurrent requests for stale entries are coalesced: the first caller
/// acquires `state.last_probe` (a global async mutex) and performs the fetch;
/// subsequent callers wait, then re-check the cache — which is now fresh —
/// and return without a second backend query.
async fn get_or_refresh_active_alias(
    state: &AppState,
    model_id: &str,
    preferred: &str,
) -> Option<String> {
    // Fast path: fresh entry under a read lock.
    {
        let guard = read_recover(&state.tracker);
        if let Some((alias, cached_at)) = guard.get(model_id)
            && cached_at.elapsed() < state.cache_ttl
        {
            return Some(alias.clone());
        }
    }

    // Slow path: serialise all concurrent refreshes through a single async
    // mutex so that at most one /v1/models query is in-flight at a time.
    let _probe_guard = state.last_probe.lock().await;

    // Re-check after acquiring: another waiter probing the SAME model_id may
    // have already refreshed this entry while we waited for the lock.
    {
        let guard = read_recover(&state.tracker);
        if let Some((alias, cached_at)) = guard.get(model_id)
            && cached_at.elapsed() < state.cache_ttl
        {
            return Some(alias.clone());
        }
    }

    // Still stale — query backend.
    let probe = query_loaded_alias(state, model_id, preferred).await;

    match probe {
        ProbeResult::Refreshed(Some(alias)) => {
            record_active_alias(state, model_id, &alias);
            Some(alias)
        }
        ProbeResult::Refreshed(None) => {
            // Probe succeeded, but nothing is loaded for this model_id.
            write_recover(&state.tracker, "tracker").remove(model_id);
            None
        }
        ProbeResult::Failed => {
            // Transient probe failure: keep any existing (now-expired) entry and
            // serve it stale rather than evicting last-known-good routing. The
            // timestamp is left untouched, so the next request still re-probes.
            read_recover(&state.tracker)
                .get(model_id)
                .map(|(alias, _)| alias.clone())
        }
    }
    // _probe_guard dropped here, releasing the lock for the next waiter.
}

/// Inspect a POST body and, if the requested alias shares a model identity
/// with a different currently-loaded alias, rewrite the body to use the loaded
/// alias and inject the requested alias's sampling parameters.
///
/// Returns the (possibly rewritten) body bytes and an optional tracker update
/// to commit only after the backend request succeeds.
struct PostIntercept {
    body_bytes: Bytes,
    record_active: Option<(String, String)>,
}

impl PostIntercept {
    /// Forward the original body unchanged, recording no active alias.
    fn passthrough(bytes: Bytes) -> Self {
        PostIntercept {
            body_bytes: bytes,
            record_active: None,
        }
    }
}

fn record_active_alias(state: &AppState, model_id: &str, alias: &str) {
    write_recover(&state.tracker, "tracker")
        .insert(model_id.to_string(), (alias.to_string(), Instant::now()));
}

async fn intercept_post(state: &AppState, path_and_query: &str, bytes: Bytes) -> PostIntercept {
    let json_body: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => return PostIntercept::passthrough(bytes),
    };
    let Value::Object(obj) = json_body else {
        return PostIntercept::passthrough(bytes);
    };

    // Own the alias string so `obj` can later be consumed by `inject` without a
    // clone of the (potentially large) request body.
    let requested_alias = match obj.get("model").and_then(|v| v.as_str()) {
        Some(s) => s.to_string(),
        None => return PostIntercept::passthrough(bytes),
    };

    // Clone the preset under a short read lock — must not hold the lock across awaits.
    let preset = read_recover(&state.presets).get(&requested_alias).cloned();
    let preset = match preset {
        Some(p) => p,
        None => {
            debug!(
                "POST {path_and_query}: {requested_alias:?} \
                — no preset found, forwarding as-is"
            );
            return PostIntercept::passthrough(bytes);
        }
    };

    if preset.model_id.is_empty() {
        debug!(
            "POST {path_and_query}: {requested_alias:?} \
            — preset has no model identity keys, forwarding as-is"
        );
        return PostIntercept::passthrough(bytes);
    }

    let active_alias = get_or_refresh_active_alias(state, &preset.model_id, &requested_alias).await;

    if let Some(compat_alias) = &active_alias
        && compat_alias != &requested_alias
    {
        // The slow path above may have rebuilt the preset map from a fresh
        // /v1/models response; re-read this alias's preset so a backend INI
        // change is reflected on THIS request rather than only the next one.
        let preset = read_recover(&state.presets)
            .get(&requested_alias)
            .cloned()
            .unwrap_or(preset);

        // Reroute: inject this preset's sampling params and redirect to the
        // loaded alias so the backend skips the unload/reload cycle.
        let mut final_obj = preset.inject(obj);
        final_obj.insert("model".to_string(), Value::String(compat_alias.clone()));

        info!(
            "POST {path_and_query}: {requested_alias:?} \
            → {compat_alias:?} (compatible model, no reload)"
        );
        debug!(
            "POST {path_and_query}: injected params: {:?}",
            preset.params
        );

        return PostIntercept {
            body_bytes: serde_json::to_vec(&final_obj)
                .map(Bytes::from)
                .unwrap_or(bytes),
            record_active: None,
        };
    }

    debug!(
        "POST {path_and_query}: {requested_alias:?} \
        (forwarded as-is; will record active alias after upstream success for model_id={:?})",
        preset.model_id
    );

    PostIntercept {
        body_bytes: bytes,
        record_active: Some((preset.model_id.clone(), requested_alias)),
    }
}

pub(crate) async fn proxy_handler(State(state): State<AppState>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let method = parts.method.clone();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| "/".to_string());
    // The proxy only routes /v1/* requests; strip /v1 before appending to the
    // backend URL, which already contains the /v1 (or equivalent) base path.
    let forward_path = path_and_query
        .strip_prefix("/v1")
        .unwrap_or(&path_and_query);

    // Reject dot-segments in the path: the URL crate collapses `..` per RFC 3986
    // when reqwest parses the target, so a path like `/v1/../slots` would
    // normalise to `<backend>/slots` and escape the advertised `/v1`-only
    // routing contract. Only the path component normalises, so the query string
    // (which may legitimately contain `..` in a value) is left untouched.
    //
    // Percent-decode before the check so encoded variants (`%2e%2e`, `..%2f..`)
    // can't smuggle a dot-segment or an extra `/` past it: a backend that decodes
    // the path before routing would otherwise see a traversal the proxy waved
    // through. Decode the whole path first, THEN split on `/`, so a decoded
    // `%2f` is treated as a real separator. The decoded form is used only for the
    // check; the still-encoded `forward_path` is what is forwarded upstream.
    let path_only = forward_path.split('?').next().unwrap_or(forward_path);
    let decoded_path = percent_encoding::percent_decode_str(path_only).decode_utf8_lossy();
    if decoded_path.split('/').any(|seg| seg == ".." || seg == ".") {
        return json_error(StatusCode::NOT_FOUND, "Not found");
    }

    let url = format!("{}{}", state.backend_url, forward_path);

    // Enforce the body size limit: to_bytes caps buffering at max_body_bytes and
    // errors past it, so it bounds memory regardless of a lying Content-Length.
    let bytes = match axum::body::to_bytes(body, state.max_body_bytes).await {
        Ok(b) => b,
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "Failed to read request body"),
    };

    // 2. POST only: attempt JSON interception and preset injection.
    let (body_bytes, pending_active_alias) = if method == Method::POST {
        let intercept = intercept_post(&state, &path_and_query, bytes).await;
        (intercept.body_bytes, intercept.record_active)
    } else {
        (bytes, None)
    };

    // 3. Build and send backend request, forwarding safe headers.
    let mut proxy_req = state.client.request(method, &url);
    for (k, v) in parts.headers.iter() {
        if !is_skip_header(k.as_str()) {
            proxy_req = proxy_req.header(k.clone(), v.clone());
        }
    }
    // Content-Length is intentionally not set here: reqwest derives it from the
    // sized `Bytes` body, and the inbound client value was dropped by
    // `is_skip_header` so it can't conflict with the (possibly rewritten) body.
    proxy_req = proxy_req.body(body_bytes);

    // 4. Stream response back to the client.
    match proxy_req.send().await {
        Ok(res) => {
            let status = res.status();
            if let Some((model_id, alias)) = &pending_active_alias {
                if status.is_success() {
                    record_active_alias(&state, model_id, alias);
                } else {
                    debug!(
                        "POST {path_and_query}: backend returned {status}, \
                        not recording {alias:?} as active for model_id={model_id:?}"
                    );
                }
            }
            let mut response_builder = Response::builder().status(status);

            if let Some(headers) = response_builder.headers_mut() {
                for (k, v) in res.headers() {
                    if !is_skip_header(k.as_str()) {
                        // append, not insert: a backend may legitimately send the
                        // same header more than once (Set-Cookie, WWW-Authenticate,
                        // Vary); insert would keep only the last value.
                        headers.append(k.clone(), v.clone());
                    }
                }
            }

            // reqwest::Error already satisfies Body::from_stream's
            // Into<BoxError> bound, so the stream needs no error remapping.
            let body = Body::from_stream(res.bytes_stream());
            match response_builder.body(body) {
                Ok(resp) => resp,
                Err(e) => {
                    error!("Failed to build response: {}", e);
                    json_error(StatusCode::INTERNAL_SERVER_ERROR, "Internal server error")
                }
            }
        }
        Err(e) => {
            error!("Backend request failed: {}", e);
            json_error(StatusCode::BAD_GATEWAY, "Bad gateway")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{AppState, find_active_alias, healthz, proxy_handler, readyz, version};
    use crate::preset::Preset;
    use axum::{Router, body::Body, http::Request, http::StatusCode, routing::any};
    use serde_json::{Map, Value, json};
    use std::{
        collections::HashMap,
        sync::{Arc, RwLock},
        time::Duration,
    };
    use tokio::sync::Mutex as AsyncMutex;
    use tower::ServiceExt;

    fn make_preset(model_id: &str) -> Preset {
        Preset {
            name: String::new(),
            model_id: model_id.to_string(),
            params: Map::new(),
            chat_template_kwargs: Map::new(),
        }
    }

    fn loaded_entry(id: &str) -> Value {
        json!({"id": id, "status": {"value": "loaded"}})
    }

    fn idle_entry(id: &str) -> Value {
        json!({"id": id, "status": {"value": "idle"}})
    }

    fn presets_map(entries: &[(&str, &str)]) -> HashMap<String, Preset> {
        entries
            .iter()
            .map(|(id, mid)| (id.to_string(), make_preset(mid)))
            .collect()
    }

    fn test_state(backend_url: &str, presets: HashMap<String, Preset>) -> AppState {
        AppState {
            presets: Arc::new(RwLock::new(presets)),
            tracker: Arc::new(RwLock::new(HashMap::new())),
            last_probe: Arc::new(AsyncMutex::new(())),
            backend_url: backend_url.to_string(),
            cache_ttl: Duration::ZERO,
            max_body_bytes: 1024,
            client: reqwest::Client::new(),
        }
    }

    fn test_app() -> Router {
        Router::new()
            .route("/v1/{*path}", any(proxy_handler))
            .with_state(test_state("http://127.0.0.1:1/v1", HashMap::new()))
    }

    // -- find_active_alias ---------------------------------------------------

    #[test]
    fn find_active_alias_none_when_no_loaded() {
        let data = vec![idle_entry("alias-a")];
        let p = presets_map(&[("alias-a", "model=x")]);
        assert!(find_active_alias(&data, &p, "model=x", "alias-a").is_none());
    }

    #[test]
    fn find_active_alias_returns_fallback() {
        let data = vec![idle_entry("alias-a"), loaded_entry("alias-b")];
        let p = presets_map(&[("alias-a", "model=x"), ("alias-b", "model=x")]);
        assert_eq!(
            find_active_alias(&data, &p, "model=x", "alias-a"),
            Some("alias-b".to_string())
        );
    }

    #[test]
    fn find_active_alias_prefers_preferred_when_loaded() {
        // alias-b appears first in data but alias-a is the preferred — it should win.
        let data = vec![loaded_entry("alias-b"), loaded_entry("alias-a")];
        let p = presets_map(&[("alias-a", "model=x"), ("alias-b", "model=x")]);
        assert_eq!(
            find_active_alias(&data, &p, "model=x", "alias-a"),
            Some("alias-a".to_string())
        );
    }

    #[test]
    fn find_active_alias_ignores_different_model_id() {
        let data = vec![loaded_entry("alias-a")];
        let p = presets_map(&[("alias-a", "model=x")]);
        assert!(find_active_alias(&data, &p, "model=y", "alias-a").is_none());
    }

    // -- routing -------------------------------------------------------------

    #[tokio::test]
    async fn route_v2_returns_404() {
        let resp = test_app()
            .oneshot(
                Request::builder()
                    .uri("/v2/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn route_health_returns_404() {
        let resp = test_app()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn route_root_returns_404() {
        let resp = test_app()
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn route_v1_bare_returns_404() {
        // "/v1" without a trailing path segment does not match "/v1/*path".
        let resp = test_app()
            .oneshot(Request::builder().uri("/v1").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn route_v1_dot_dot_segment_rejected() {
        // `/v1/../slots` would normalise to `<backend>/slots` once reqwest parses
        // it, escaping the /v1-only contract; the handler must reject it as 404
        // rather than forward (which would surface as 502 against the dead port).
        let resp = test_app()
            .oneshot(
                Request::builder()
                    .uri("/v1/../slots")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn route_v1_dot_dot_deeper_segment_rejected() {
        let resp = test_app()
            .oneshot(
                Request::builder()
                    .uri("/v1/models/../../admin")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn route_v1_percent_encoded_dotdot_rejected() {
        // `/v1/%2e%2e/slots` decodes to `/v1/../slots`; a backend that decodes
        // before routing would escape the /v1-only contract, so it must 404 here
        // rather than be forwarded (which would surface as 502 on the dead port).
        let resp = test_app()
            .oneshot(
                Request::builder()
                    .uri("/v1/%2e%2e/slots")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn route_v1_encoded_slash_dotdot_rejected() {
        // `..%2f..` decodes to `../..`; decoding before the split surfaces the
        // encoded `/` as a real separator so the dot-segments are caught.
        let resp = test_app()
            .oneshot(
                Request::builder()
                    .uri("/v1/models/..%2f..%2fadmin")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn route_v1_query_with_dotdot_not_rejected() {
        // `..` inside a query value must not be treated as a path traversal; the
        // request reaches the handler (502 against the dead port, NOT 404).
        let resp = test_app()
            .oneshot(
                Request::builder()
                    .uri("/v1/models?ref=a/../b")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn route_v1_path_reaches_handler() {
        // The route matches; handler runs but gets 502 because port 1 is unreachable.
        // The key assertion is that the response is NOT 404.
        let resp = test_app()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_ne!(resp.status(), StatusCode::NOT_FOUND);
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    #[tokio::test]
    async fn failed_post_does_not_record_active_alias() {
        let state = test_state(
            "http://127.0.0.1:1/v1",
            presets_map(&[("alias-a", "model=test.gguf")]),
        );
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"alias-a","messages":[]}"#))
            .unwrap();

        let resp = proxy_handler(axum::extract::State(state.clone()), req).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

        let tracker = state.tracker.read().unwrap();
        assert!(
            tracker.is_empty(),
            "tracker should stay empty after upstream failure"
        );
    }

    // -- health endpoints ----------------------------------------------------

    #[tokio::test]
    async fn healthz_always_ok() {
        let resp = healthz().await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn version_reports_crate_version() {
        let resp = version().await;
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["version"], env!("CARGO_PKG_VERSION"));
    }

    // Exercises the load_shed middleware with a shared 1-permit semaphore: while
    // request A holds the only permit (parked inside the handler), request B must
    // be shed immediately with 503 rather than queued. Two semaphores form a
    // handshake (permits are stored, so there is no lost-wakeup race): `entered`
    // lets the test wait until A is provably inside the handler holding the
    // permit, `release` lets the test free A afterwards. A multi-thread runtime
    // keeps A's parked handler off the thread driving B, and every await is
    // bounded by a timeout so a regression fails fast instead of hanging.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn load_shed_returns_503_when_concurrency_exceeded() {
        use axum::middleware::from_fn_with_state;
        use tokio::sync::Semaphore;
        use tokio::time::{Duration as TDuration, timeout};

        let entered = Arc::new(Semaphore::new(0));
        let release = Arc::new(Semaphore::new(0));

        let ent = entered.clone();
        let rel = release.clone();
        let slow = move || {
            let ent = ent.clone();
            let rel = rel.clone();
            async move {
                ent.add_permits(1); // signal: handler reached, permit held
                let _permit = rel.acquire().await.unwrap(); // park until released
                StatusCode::OK
            }
        };

        let permits = Arc::new(Semaphore::new(1));
        let app = Router::new()
            .route("/slow", any(slow))
            .layer(from_fn_with_state(permits, super::load_shed));

        // Request A takes the only permit and parks inside the handler.
        let app_a = app.clone();
        let a = tokio::spawn(async move {
            app_a
                .oneshot(Request::builder().uri("/slow").body(Body::empty()).unwrap())
                .await
                .unwrap()
        });

        // Wait until A is provably inside the handler (concurrency permit held).
        timeout(TDuration::from_secs(5), entered.acquire())
            .await
            .expect("request A never entered the handler")
            .unwrap()
            .forget();

        // Request B finds no permit and is shed → 503 (not queued).
        let resp_b = timeout(
            TDuration::from_secs(5),
            app.clone()
                .oneshot(Request::builder().uri("/slow").body(Body::empty()).unwrap()),
        )
        .await
        .expect("request B was queued instead of shed")
        .unwrap();
        assert_eq!(resp_b.status(), StatusCode::SERVICE_UNAVAILABLE);

        // Release A and confirm it completed normally.
        release.add_permits(1);
        let resp_a = timeout(TDuration::from_secs(5), a)
            .await
            .expect("request A did not complete after release")
            .unwrap();
        assert_eq!(resp_a.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn readyz_503_when_no_presets() {
        // Empty preset map short-circuits to 503 before any backend probe.
        let state = test_state("http://127.0.0.1:1/v1", HashMap::new());
        let resp = readyz(axum::extract::State(state)).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn readyz_503_when_backend_unreachable() {
        // Presets present, but the backend (dead port 1) is unreachable → 503.
        let state = test_state(
            "http://127.0.0.1:1/v1",
            presets_map(&[("alias-a", "model=x")]),
        );
        let resp = readyz(axum::extract::State(state)).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn readyz_200_when_presets_and_backend_ok() {
        let models = json!({"data": []});
        let backend = Router::new().route(
            "/v1/models",
            axum::routing::get(move || {
                let m = models.clone();
                async move { axum::Json(m) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, backend).await.unwrap() });

        let state = test_state(
            &format!("http://{addr}/v1"),
            presets_map(&[("alias-a", "model=x")]),
        );
        let resp = readyz(axum::extract::State(state)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn post_reroutes_to_loaded_alias_and_injects_params() {
        // Mock backend: alias-b is loaded and shares alias-a's model. A POST for
        // alias-a must be rewritten to alias-b with alias-a's params injected, and
        // those params must come from the alias's freshly-probed preset (temp=0.6),
        // not the pre-refresh snapshot. The /v1/chat/completions route echoes the
        // received body so we can assert what the proxy forwarded.
        let models = json!({"data": [
            {"id": "alias-a", "status": {"value": "idle",
                "preset": "[alias-a]\nhf-repo = org/m:Q4\ntemp = 0.6\n"}},
            {"id": "alias-b", "status": {"value": "loaded",
                "preset": "[alias-b]\nhf-repo = org/m:Q4\ntemp = 0.9\n"}},
        ]});
        let backend = Router::new()
            .route(
                "/v1/models",
                axum::routing::get(move || {
                    let m = models.clone();
                    async move { axum::Json(m) }
                }),
            )
            .route(
                "/v1/chat/completions",
                axum::routing::post(|body: axum::body::Bytes| async move { body }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, backend).await.unwrap() });

        let state = test_state(
            &format!("http://{addr}/v1"),
            presets_map(&[
                ("alias-a", "hf-repo=org/m:Q4"),
                ("alias-b", "hf-repo=org/m:Q4"),
            ]),
        );
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"model":"alias-a","messages":[]}"#))
            .unwrap();

        let resp = proxy_handler(axum::extract::State(state), req).await;
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let echoed: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            echoed["model"],
            json!("alias-b"),
            "model rewritten to the loaded alias"
        );
        assert_eq!(
            echoed["temperature"],
            json!(0.6),
            "freshly-probed preset params injected"
        );
    }
}
