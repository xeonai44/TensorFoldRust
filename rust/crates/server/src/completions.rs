//! `/v1/completions`: the legacy completion surface — the same engine
//! path the chat route runs, the prompt as the model reads it and the
//! reply a `text_completion`.
//!
//! Port of the completions half of `zig/src/server/openai.zig` (`run` with
//! `is_chat = false`): the plan (openai.zig:119) through
//! `chat::prepare_completion`, the whole reply `wholeBody`'s else branch
//! (openai.zig:376-395), and the stream the `Run`'s own with
//! `chunk_body`'s completion shape and the usage a chunk of its own when
//! `stream_options.include_usage` asked.

use crate::state::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::sse::Event as SseEvent;
use axum::response::IntoResponse;
use axum::Json;
use serde_json::{json, Value};
use std::sync::Arc;

/// POST /v1/completions (openai.zig:218's run with `is_chat = false`).
pub async fn completions(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> axum::response::Response {
    if state.engine.is_none() || state.text.is_none() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": {"message": state.engineless_problem, "type": "engine_unavailable"}
            })),
        )
            .into_response();
    }
    // ``json.loads`` on the raw bytes, Python's own decoder and its error
    // messages (the chat route's read — openai.run's one path)
    let body: Value = match crate::json_parse::parse(&body) {
        Ok(v) => v,
        Err(message) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {"message": message}
                })),
            )
                .into_response()
        }
    };
    // `parallel_tool_calls` must be a boolean (openai.zig:164-166)
    if let Some(p) = body.get("parallel_tool_calls") {
        if p.as_bool().is_none() {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": {"message": "parallel_tool_calls must be a boolean", "type": "invalid_request_error"}
                })),
            )
                .into_response();
        }
    }
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let separate_usage = body
        .get("stream_options")
        .and_then(Value::as_object)
        .is_some_and(|o| o.get("include_usage").is_some_and(crate::pyrepr::truthy));
    // The completion's plan: the prompt as the model reads it, the same
    // shared tail the chat plan runs
    let prepared = match tokio::task::block_in_place(|| crate::chat::prepare_completion(&state, &body))
    {
        Ok(p) => p,
        Err(refused) => {
            let code = StatusCode::from_u16(refused.status()).unwrap_or(StatusCode::BAD_REQUEST);
            return (
                code,
                Json(crate::errors::wrap_error(crate::errors::error_body(
                    &refused, "prompt",
                ))),
            )
                .into_response();
        }
    };
    let id = crate::ids::make("cmpl-", 32);
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let model = crate::chat::reply_model(&state, &body);
    if stream {
        // The stream (openai.zig:456's Run.stream): the bridge's
        // completion shape, the usage a chunk of its own when asked, the
        // context refusal inside the stream
        let start = match tokio::task::block_in_place(|| {
            crate::chat::submit_prepared_streaming(&state, prepared, Value::Null)
        }) {
            Ok(start) => start,
            Err(refused) if refused.kind == crate::errors::Kind::ContextLength => {
                return context_refused_stream(refused);
            }
            Err(refused) => {
                let code = StatusCode::from_u16(refused.status()).unwrap_or(StatusCode::BAD_REQUEST);
                return (
                    code,
                    Json(crate::errors::wrap_error(crate::errors::error_body(
                        &refused, "prompt",
                    ))),
                )
                    .into_response();
            }
        };
        let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel::<crate::chat::ChunkOut>(16);
        let text_arc = state.text.clone().expect("checked above");
        let chunk_id = id.clone();
        let chunk_model = model.clone();
        let tools = Vec::new();
        let policy = crate::tool_stream::Policy::new(false);
        std::thread::spawn(move || {
            let _ = crate::chat::bridge(
                start,
                chunk_tx,
                chunk_id,
                chunk_model,
                created,
                text_arc,
                tools,
                policy,
                false,
                separate_usage,
                state,
            );
        });
        let (sse_tx, sse_rx) =
            tokio::sync::mpsc::channel::<Result<SseEvent, std::convert::Infallible>>(16);
        std::thread::spawn(move || {
            while let Some(out) = chunk_rx.blocking_recv() {
                let done = matches!(out, crate::chat::ChunkOut::Done);
                let event = match out {
                    crate::chat::ChunkOut::Chunk(body) => {
                        SseEvent::default().data(crate::json_write::dumps(&body))
                    }
                    crate::chat::ChunkOut::Done => SseEvent::default().data("[DONE]"),
                };
                if sse_tx.blocking_send(Ok(event)).is_err() || done {
                    break;
                }
            }
        });
        return Sse::new(tokio_stream::wrappers::ReceiverStream::new(sse_rx))
            .keep_alive(KeepAlive::default())
            .into_response();
    }
    // The whole reply (openai.zig:355's whole): the completion body from
    // the chat reply, its extras the same
    let reply = tokio::task::block_in_place(|| crate::chat::generate_prepared(&state, prepared));
    if let Some(refused) = &reply.failed {
        let code = StatusCode::from_u16(refused.status()).unwrap_or(StatusCode::BAD_REQUEST);
        return (
            code,
            Json(crate::errors::wrap_error(crate::errors::error_body(
                refused, "prompt",
            ))),
        )
            .into_response();
    }
    // wholeBody's else branch (openai.zig:389-392): the choice carries the
    // reply's text; the logprobs never ride (the plan refused them)
    let mut whole = json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "text": reply.content,
            "finish_reason": reply.finish_reason,
            "logprobs": null,
        }],
        "usage": {
            "prompt_tokens": reply.prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "total_tokens": reply.prompt_tokens + reply.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": reply.cached_tokens},
            "completion_tokens_details": {"reasoning_tokens": reply.reasoning_tokens},
        },
    });
    crate::chat::response_extras(
        &mut whole,
        reply.stop_sequence.as_deref(),
        &reply.runtime,
        &reply.speculative,
    );
    (StatusCode::OK, Json(whole)).into_response()
}

/// The context refusal inside the stream (openai.zig:345's
/// streamedContext through the is_chat=false half): for a completion the
/// role chunk never opens (openai.zig:349's `is_chat` check) — the error
/// body, [DONE].
fn context_refused_stream(refused: crate::errors::Refused) -> axum::response::Response {
    let (sse_tx, sse_rx) =
        tokio::sync::mpsc::channel::<Result<SseEvent, std::convert::Infallible>>(8);
    std::thread::spawn(move || {
        let error =
            crate::errors::wrap_error(crate::errors::error_body(&refused, "prompt"));
        let _ = sse_tx.blocking_send(Ok(SseEvent::default().data(crate::json_write::dumps(
            &error,
        ))));
        let _ = sse_tx.blocking_send(Ok(SseEvent::default().data("[DONE]")));
    });
    Sse::new(tokio_stream::wrappers::ReceiverStream::new(sse_rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}
