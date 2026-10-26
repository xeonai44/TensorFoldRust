//! The routes: model discovery first, inference refusals while engine-less.
//!
//! Port of `zig/src/server/routes.zig`'s Phase 0 surface and the README's "Model
//! discovery" section: `GET /v1/models` (also `/models`) advertises each served name
//! and alias, and "when the engine reports a finite context window, each entry includes
//! equal `context_length` and `max_model_len` integer extensions. [...] An unspecified
//! engine window omits both fields."
//!
//! The chat/completions/responses/Anthropic bodies are Phase 2 (chat.zig,
//! anthropic.zig and their translation modules in docs/ZIG-TO-RUST-MAP.md); while
//! engine-less they answer with the none-backend message.

use crate::state::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};
use std::sync::Arc;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/models", get(models))
        .route("/models", get(models))
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/dashboard", get(dashboard))
        .route("/stats", get(stats))
        .route("/memory", get(memory))
        .route("/v1/memory", get(memory))
        .route("/v1/chat/completions", post(crate::chat::chat_completions))
        .route("/v1/decisions", post(decisions))
        .route("/tokenize", post(tokenize))
        .route("/v1/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/v1/detokenize", post(detokenize))
        .route("/decisions", post(decisions))
        .route("/v1/completions", post(crate::completions::completions))
        .route(
            "/v1/responses",
            post(crate::responses::post)
                .get(crate::responses::collection_get)
                .delete(crate::responses::delete_collection),
        )
        .route(
            "/v1/responses/{id}",
            get(crate::responses::get_one)
                .delete(crate::responses::delete_one)
                .post(crate::responses::id_post),
        )
        .route(
            "/responses",
            post(crate::responses::post)
                .get(crate::responses::collection_get)
                .delete(crate::responses::delete_collection),
        )
        .route(
            "/responses/{id}",
            get(crate::responses::get_one)
                .delete(crate::responses::delete_one)
                .post(crate::responses::id_post),
        )
        .route("/v1/messages", post(crate::anthropic::messages))
        .route("/v1/messages/count_tokens", post(crate::anthropic::count_tokens))
        .fallback(unknown_path)
        .with_state(state.clone())
        .layer(axum::middleware::from_fn_with_state(state, auth_and_count))
}

/// POST /v1/decisions (decisions.zig:37's post) — the body checked the
/// way 0.6.6 checks it, then scored through the engine hook when a family
/// sets one.
async fn decisions(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    match crate::decisions::reply(&state, &body) {
        Ok(payload) => (StatusCode::OK, crate::json_write::PythonJson(payload)).into_response(),
        Err(refused) => {
            let code = StatusCode::from_u16(refused.status())
                .unwrap_or(StatusCode::BAD_REQUEST);
            (
                code,
                crate::json_write::PythonJson(crate::errors::wrap_error(crate::errors::error_body(
                    &refused, "input",
                ))),
            )
                .into_response()
        }
    }
}

/// POST /tokenize and /detokenize (tokens.zig:94's post) — the ids the
/// chat and completion routes would run, and the text back.
async fn tokenize(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    match crate::tokens::tokenize(&state, &body) {
        Ok(payload) => (StatusCode::OK, crate::json_write::PythonJson(payload)).into_response(),
        Err(refused) => {
            let code =
                StatusCode::from_u16(refused.status()).unwrap_or(StatusCode::BAD_REQUEST);
            (
                code,
                crate::json_write::PythonJson(json!({"error": {"message": refused.message,
                    "type": "invalid_request_error"}})),
            )
                .into_response()
        }
    }
}

/// The detokenize twin.
async fn detokenize(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    match crate::tokens::detokenize(&state, &body) {
        Ok(payload) => (StatusCode::OK, crate::json_write::PythonJson(payload)).into_response(),
        Err(refused) => {
            let code =
                StatusCode::from_u16(refused.status()).unwrap_or(StatusCode::BAD_REQUEST);
            (
                code,
                crate::json_write::PythonJson(json!({"error": {"message": refused.message,
                    "type": "invalid_request_error"}})),
            )
                .into_response()
        }
    }
}

/// Upstream's unknown-path refusal (routes.zig:82's `unknown`): the 404
/// naming the path, no type.
fn unknown(path: &str) -> axum::response::Response {
    (
        StatusCode::NOT_FOUND,
        crate::json_write::PythonJson(json!({"error": {"message": format!("unknown path {path}")}})),
    )
        .into_response()
}

/// The router's fallback — upstream's dispatch tail (routes.zig:47-55):
/// the route stripped of its query and trailing slashes (auth.zig:248's
/// routePath), an empty route and `/health` the health route, any route
/// ending in `/models` the models one, the rest the unknown-path refusal.
async fn unknown_path(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    axum::extract::OriginalUri(uri): axum::extract::OriginalUri,
) -> axum::response::Response {
    let route = uri.path().trim_end_matches('/');
    if route.is_empty() || route == "/health" {
        return health(State(state), axum::extract::OriginalUri(uri)).await;
    }
    if route.ends_with("/models") {
        return models(State(state)).await;
    }
    unknown(uri.path())
}

/// The key gate before any route and the reply counter (server.zig:249's
/// `authenticate` + :257's `countReply`): a gated path without a valid
/// key answers 401 — the Messages paths Anthropic's own error shape —
/// and every reply counts under its key's label (or "unauthenticated").
/// OPTIONS passes through (the CORS preflight).
pub async fn auth_and_count(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let mut label: Option<String> = None;
    let gated_path = state.auth.enabled()
        && req.method() != axum::http::Method::OPTIONS
        && crate::auth::gated(req.uri().path(), state.auth.metrics_open);
    if state.auth.enabled() && req.method() != axum::http::Method::OPTIONS {
        label = state.auth.match_headers(req.headers());
    }
    if gated_path && label.is_none() {
        let path = crate::auth::route_path(req.uri().path());
        let messages = [
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/messages",
            "/messages/count_tokens",
        ]
        .contains(&path);
        let body = if messages {
            crate::json_write::dumps(&json!({
                "type": "error",
                "error": {"type": "authentication_error", "message": "A valid API key is required"}
            }))
        } else {
            crate::json_write::dumps(&json!({
                "error": {"message": "A valid API key is required",
                          "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}
            }))
        };
        // The refusal counts like every reply (server.zig:257's countReply)
        state
            .metrics
            .http_request("unauthenticated", StatusCode::UNAUTHORIZED.as_u16());
        // The connection closes with the refusal (server.zig:274)
        return (
            StatusCode::UNAUTHORIZED,
            [
                ("www-authenticate", "Bearer"),
                ("content-type", "application/json"),
                ("connection", "close"),
            ],
            body,
        )
            .into_response();
    }
    let response = next.run(req).await;
    state
        .metrics
        .http_request(label.as_deref().unwrap_or("unauthenticated"), response.status().as_u16());
    response
}

/// Model discovery: every served id, with the window extensions when the engine
/// reports a finite one.
async fn models(
    State(state): State<Arc<AppState>>,
) -> axum::response::Response {
    let data: Vec<_> = state
        .model_ids()
        .iter()
        .map(|id| {
            // `created` is the discovery time of the request
            // (status_routes.zig:37: the clock read per models call)
            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let mut entry = json!({
                "id": id,
                "object": "model",
                "created": created,
                "owned_by": "tensorfold",
            });
            if let Some(info) = &state.engine_info {
                if info.context_window > 0 {
                    // "equal context_length and max_model_len integer extensions"
                    entry["context_length"] = json!(info.context_window);
                    entry["max_model_len"] = json!(info.context_window);
                }
            }
            entry
        })
        .collect();
    (StatusCode::OK, crate::json_write::PythonJson(json!({"object": "list", "data": data}))).into_response()
}

/// /health: the status line's copy (status_routes.zig:13) — with keys on,
/// nothing about the model.
async fn health(
    State(state): State<Arc<AppState>>,
    axum::extract::OriginalUri(uri): axum::extract::OriginalUri,
) -> axum::response::Response {
    // ``?reset_peak=1`` resets the engine's peak before the counters read
    // (status_routes.zig:26)
    let reset_peak = uri.query().is_some_and(|q| q.contains("reset_peak=1"));
    if state.auth.enabled() {
        return (StatusCode::OK, crate::json_write::PythonJson(json!({"status": "ok"}))).into_response();
    }
    let mut tokens: [u32; 64] = [0; 64];
    let status = state
        .engine
        .as_ref()
        .map(|e| e.status(&mut tokens))
        .unwrap_or_default();
    let live = state
        .engine
        .as_ref()
        .map(|e| crate::live::snapshot(e.as_ref()))
        .unwrap_or_else(|| serde_json::json!({
            "connections": 0, "waiting": 0,
            "decode_tokens_per_second": 0.0, "prefill_tokens_per_second": 0.0,
        }));
    let memory = state
        .engine
        .as_ref()
        .and_then(|e| e.memory(reset_peak))
        .map(|m| json!({"active": m.active, "cache": m.cache, "peak": m.peak}))
        .unwrap_or_else(|| json!({}));
    (
        StatusCode::OK,
        crate::json_write::PythonJson(json!({
            "status": "ok",
            "model": state.config.served_name,
            "model_ids": state.config.model_ids,
            "max_batch_size": state.engine_info.as_ref().map_or(1, |i| i.lanes),
            "warming": status.warming,
            "memory": memory,
            "live": live,
        })),
    )
        .into_response()
}

/// The opt-in dashboard (status_routes.zig:74): the one-file page polling
/// /stats. Off: the unknown route.
async fn dashboard(State(state): State<Arc<AppState>>) -> axum::response::Response {
    if !state.config.dashboard {
        return unknown("/dashboard");
    }
    (
        StatusCode::OK,
        [("content-type", "text/html; charset=utf-8")],
        include_str!("dashboard.html"),
    )
        .into_response()
}

/// /stats (status_routes.zig:83): the dashboard's JSON poll.
async fn stats(State(state): State<Arc<AppState>>) -> axum::response::Response {
    if !state.config.dashboard {
        return unknown("/stats");
    }
    let mut tokens: [u32; 64] = [0; 64];
    let status = state
        .engine
        .as_ref()
        .map(|e| e.status(&mut tokens))
        .unwrap_or_default();
    let memory = state.engine.as_ref().and_then(|e| e.memory(false));
    let (prompt, generation, drafted, accepted, rounds) = state.metrics.totals();
    (
        StatusCode::OK,
        crate::json_write::PythonJson(stats_snapshot(status, memory, prompt, generation, drafted, accepted, rounds)),
    )
        .into_response()
}

/// The /stats body (status_routes.zig:100's statsSnapshot).
fn stats_snapshot(
    status: tensorfold_engine::Status,
    memory: Option<tensorfold_engine::Memory>,
    prompt: u64,
    generation: u64,
    drafted: u64,
    accepted: u64,
    rounds: u64,
) -> Value {
    let mem = memory.map(|m| {
        json!({
            "active": m.active,
            "cache": m.cache,
            "peak": m.peak,
        })
    });
    json!({
        "running": status.running,
        "waiting": status.waiting,
        "decode_tokens_per_second": status.decode_tokens_per_second,
        "prefill_tokens_per_second": status.prefill_tokens_per_second,
        "generation_tokens_running": status.generation_tokens,
        "prompt_tokens_total": prompt,
        "generation_tokens_total": generation,
        "mtp_drafted_total": drafted,
        "mtp_accepted_total": accepted,
        "decode_rounds_total": rounds,
        "memory": mem.unwrap_or_else(|| json!({})),
    })
}

/// /v1/memory (memory_routes.zig:11): the sizing snapshot.
async fn memory(State(state): State<Arc<AppState>>) -> axum::response::Response {
    let process = crate::process_memory::read();
    (
        StatusCode::OK,
        crate::json_write::PythonJson(crate::memory_routes::snapshot(&state, process)),
    )
        .into_response()
}

/// Upstream /metrics is Prometheus text over the engine's counters (metrics.zig,
/// status_routes.zig); Phase 0 carries the line it always has and grows with an
/// engine.
/// The /metrics scrape: the Prometheus text (metrics.zig's render).
async fn metrics(State(state): State<Arc<AppState>>) -> axum::response::Response {
    // The engine's status at scrape time, with its context window for the pools.
    let window = state.engine_info.as_ref().map_or(0, |i| i.context_window);
    let engine: Option<&dyn tensorfold_engine::Engine> = state.engine.as_deref();
    let body = state.metrics.render(engine, window);
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        body,
    )
        .into_response()
}




#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::Auth;
    use crate::Config;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::util::ServiceExt;

    fn state() -> Arc<AppState> {
        let config = Config::new("local-model".to_string());
        Arc::new(AppState::new(config))
    }

    #[tokio::test]
    async fn models_lists_served_name_without_window_extensions() {
        let app = router(state());
        let response = app
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["id"], "local-model");
        assert!(json["data"][0].get("context_length").is_none());
    }

    #[tokio::test]
    async fn models_list_window_extensions_with_a_finite_engine_window() {
        let mut info = tensorfold_engine::Info::default();
        info.context_window = 8192;
        let s = Arc::new(
            AppState::new(Config::new("local-model".to_string())).with_engine_info(info),
        );
        let app = router(s);
        let response = app
            .oneshot(Request::builder().uri("/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["data"][0]["context_length"], 8192);
        assert_eq!(json["data"][0]["max_model_len"], 8192);
    }

    #[tokio::test]
    async fn inference_routes_refuse_engineless_with_the_none_backend_message() {
        for uri in [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/messages",
        ] {
            let app = router(state());
            // The chat route reads a JSON body; the others refuse before reading
            // one. Both reach the handler, which refuses as the engine-less
            // build's message.
            let body = if uri.contains("/v1/messages") || uri == "/v1/chat/completions" {
                r#"{"messages": []}"#
            } else {
                ""
            };
            let response = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(uri)
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE, "{}", uri);
            let body = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert!(
                json["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("no backend"),
                "{}",
                uri
            );
        }
    }

    #[tokio::test]
    async fn keyed_server_admits_only_bearers_on_models() {
        let config = Config::new("local-model".to_string());
        let s = Arc::new(
            AppState::new(config)
                .with_auth(Auth::init(&["secret".to_string()], "", None, false).unwrap()),
        );
        let app = router(s);
        let no_key = app
            .clone()
            .oneshot(Request::builder().uri("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(no_key.status(), StatusCode::UNAUTHORIZED);
        let with_key = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .header("authorization", "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(with_key.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn health_is_open() {
        let app = router(state());
        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    /// The responses collection's store routes, the frozen corpus's 404
    /// wordings (zig/tests/server/golden/responses/{get,delete}-*.json):
    /// they answer with no engine, exactly as upstream's do.
    /// The key gate (the corpus's keys group, the server.zig:249 shapes):
    /// the 401 body, the header schemes, the Messages path's own shape,
    /// metrics-open, and the reply counted under its key's label.
    #[tokio::test]
    async fn the_key_gate() {
        let config = Config::new("local-model".to_string());
        let keyed = Arc::new(
            AppState::new(config)
                .with_auth(Auth::init(&["sk-cli".into()], "", None, false).unwrap()),
        );
        let app = router(keyed);
        let body = |response: axum::response::Response| async {
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            String::from_utf8_lossy(&bytes).to_string()
        };
        // Without a key: the corpus's exact 401 body
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            body(response).await,
            "{\"error\": {\"message\": \"A valid API key is required\", \"type\": \"invalid_request_error\", \"param\": null, \"code\": \"invalid_api_key\"}}"
        );
        // Basic does not admit (the corpus's models-basic case)
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .header("authorization", "Basic sk-cli")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        // Bearer and x-api-key both admit
        for (name, value) in [("authorization", "Bearer sk-cli"), ("x-api-key", "sk-cli")] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/v1/models")
                        .header(name, value)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{name}");
        }
        // The Messages paths answer Anthropic's own shape
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/messages")
                    .header("content-type", "application/json")
                    .body(Body::from("{}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            body(response).await,
            "{\"type\": \"error\", \"error\": {\"type\": \"authentication_error\", \"message\": \"A valid API key is required\"}}"
        );
        // /health stays open, /metrics gated
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        // The reply counted under its key's label and the refusal under
        // unauthenticated
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .header("authorization", "Bearer sk-cli")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let text = body(response).await;
        assert!(
            text.contains("requests_total{key=\"cli-1\",status=\"200\"}"),
            "the reply counted by label: {text}"
        );
        assert!(
            text.contains("requests_total{key=\"unauthenticated\",status=\"401\"}"),
            "the refusals counted: {text}"
        );
        // metrics-open: the scrape ungated
        let open_metrics = Arc::new(
            AppState::new(Config::new("local-model".to_string()))
                .with_auth(Auth::init(&["sk-cli".into()], "", None, true).unwrap()),
        );
        let response = router(open_metrics)
            .oneshot(
                Request::builder()
                    .uri("/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    /// The dispatch tail's routing (the status group's odd paths): an
    /// empty route and `/health` the health route whatever its trailing
    /// slashes, any route ending in `/models` the models one, the rest
    /// upstream's unknown path.
    #[tokio::test]
    async fn the_dispatch_tails_odd_paths() {
        let app = router(state());
        let body = |response: axum::response::Response| async {
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()
        };
        for path in ["/", "/health/", "/health?x=1"] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{path}");
            assert_eq!(body(response).await["status"], "ok", "{path}");
        }
        for path in ["/x/models", "/deep/z/models", "/v1/models/"] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{path}");
            assert_eq!(body(response).await["object"], "list", "{path}");
        }
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            body(response).await,
            serde_json::json!({"error": {"message": "unknown path /v1/health"}})
        );
    }

    #[tokio::test]
    async fn the_responses_store_routes() {
        let app = router(state());
        let body = |response: axum::response::Response| async {
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap()
        };
        // A missing id: 404 naming it, the param on the refusal
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/responses/resp_missing")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            body(response).await,
            serde_json::json!({
                "error": {"message": "no stored response has id 'resp_missing'",
                          "type": "invalid_request_error", "param": "response_id", "code": null}
            })
        );
        // DELETE on the collection: the empty id named
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri("/v1/responses")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            body(response).await,
            serde_json::json!({
                "error": {"message": "no stored response has id ''",
                          "type": "invalid_request_error", "param": "response_id", "code": null}
            })
        );
        // GET on the collection: upstream's unknown-path refusal
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/responses")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            body(response).await,
            serde_json::json!({"error": {"message": "unknown path /v1/responses"}})
        );
        // POST on an id: upstream's unknown path (routes.zig:68 leaves it)
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/responses/resp_x")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"input": "x"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            body(response).await,
            serde_json::json!({"error": {"message": "unknown path /v1/responses/resp_x"}})
        );
    }
}
