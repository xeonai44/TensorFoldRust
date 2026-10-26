//! Anthropic's Messages routes over the chat handler: the reply as content
//! blocks, `count_tokens`.
//!
//! Port of `zig/src/server/anthropic.zig`'s non-streaming surface (the route,
//! `errorValue`, `countTokens`, and `Reply.completion` — the whole reply as one
//! Messages response). The typed SSE events (message_start, content_block_start,
//! the deltas, message_stop) arrive with the streaming chunk plan, the next
//! tranche in `docs/ZIG-TO-RUST-MAP.md`.

use crate::state::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::IntoResponse;
use axum::Json;
use serde_json::{json, Map, Value};
use std::sync::Arc;

/// Upstream: `errorValue` (anthropic.zig:23) — Anthropic's error object.
fn error_value(message: &str, status: u16) -> Value {
    let kind = match status {
        400 => "invalid_request_error",
        404 => "not_found_error",
        503 => "overloaded_error",
        _ => "api_error",
    };
    json!({
        "type": "error",
        "error": {"type": kind, "message": message},
    })
}

fn send_error(message: &str, status: u16) -> axum::response::Response {
    (StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_REQUEST), crate::json_write::PythonJson(error_value(message, status)))
        .into_response()
}

/// Upstream: `finishOf` (anthropic.zig:378).
/// A chat usage as Anthropic's (anthropic.zig:385 `usage`): cached prompt tokens
/// read from the cache, thinking tokens as output detail.
fn usage(value: Option<&Value>) -> Value {
    let value = value.cloned().unwrap_or(Value::Null);
    let prompt_tokens = value.get("prompt_tokens").and_then(Value::as_i64).unwrap_or(0);
    let cached_raw = value
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let cached = prompt_tokens.min(cached_raw.max(0));
    let mut out = Map::new();
    out.insert("input_tokens".into(), json!(prompt_tokens - cached));
    out.insert(
        "output_tokens".into(),
        value
            .get("completion_tokens")
            .cloned()
            .unwrap_or_else(|| json!(0)),
    );
    out.insert("cache_creation_input_tokens".into(), json!(0));
    out.insert("cache_read_input_tokens".into(), json!(cached));
    if let Some(details) = value.get("completion_tokens_details") {
        if let Some(t) = details.get("reasoning_tokens") {
            out.insert(
                "output_tokens_details".into(),
                json!({"thinking_tokens": t}),
            );
        }
    }
    Value::Object(out)
}

/// The streaming path: the typed Messages events over the chat stream
/// (anthropic.zig's Wire at 101: the chat chunks a Reply translates).
/// The chat bridge produces the OpenAI-style chunks — the prose and
/// reasoning deltas, the tool_call deltas, the last chunk with the finish
/// reason, the usage and the matched stop — and the Reply machine turns
/// them into message_start, the content blocks, message_delta and
/// message_stop.
fn stream_messages(
    state: Arc<AppState>,
    body: &Value,
    model: String,
) -> Result<axum::response::Response, String> {
    let chat = crate::anthropic_translate::translate(body, false)?;
    // The stream's own options from the translated body: the tools the
    // template offers (tool_specs.zig:58's active), the one-call policy
    // (anthropic_translate.zig:248's parallel_tool_calls)
    let policy = crate::tool_stream::Policy::new(
        chat.get("parallel_tool_calls").and_then(Value::as_bool) == Some(false),
    );
    let tools =
        crate::tool_specs::active(chat.get("tools"), chat.get("tool_choice"))?;
    let start = match crate::chat::submit_streaming(&state, &chat) {
        Ok(start) => start,
        // A context refusal reaches the client inside the stream the way
        // upstream's does (openai.zig:348 streamedContext through the Wire):
        // message_start, the error event, and the stream ends without
        // message_stop
        Err(refused) if refused.kind == crate::errors::Kind::ContextLength => {
            let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<
                Result<SseEvent, std::convert::Infallible>,
            >(8);
            let model_for_reply = model.clone();
            std::thread::spawn(move || {
                let (event_tx, mut event_rx) =
                    tokio::sync::mpsc::channel::<TypedEvent>(8);
                let inner = std::thread::spawn(move || {
                    let mut r = Reply::new(event_tx, model_for_reply);
                    let _ = r.start();
                });
                let _ = inner.join();
                let Some(started) = event_rx.blocking_recv() else { return };
                let _ = sse_tx.blocking_send(Ok(SseEvent::default()
                    .event(started.kind)
                    .data(crate::json_write::dumps(&started.body))));
                let e = error_value(&refused.message, refused.status());
                let _ = sse_tx.blocking_send(Ok(SseEvent::default()
                    .event("error")
                    .data(crate::json_write::dumps(&e))));
            });
            return Ok(Sse::new(tokio_stream::wrappers::ReceiverStream::new(sse_rx))
                .into_response());
        }
        Err(refused) => return Err(refused.message),
    };
    let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel::<crate::chat::ChunkOut>(16);
    let text_arc = state.text.clone().ok_or("no text")?;
    // The chat Run's own id rides the chunks; the Messages reply names
    // itself (anthropic.zig:150's ids.make)
    let chunk_id = crate::ids::make("chatcmpl-", 32);
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let chunk_model = model.clone();
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
            // The chat-shaped chunks the translated surfaces read
            true,
            false,
            state,
        );
    });
    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel::<TypedEvent>(16);
    let reply_model = model;
    std::thread::spawn(move || {
        let mut reply = Reply::new(event_tx, reply_model);
        if reply.start().is_err() {
            return;
        }
        while let Some(out) = chunk_rx.blocking_recv() {
            let (payload, end) = match out {
                crate::chat::ChunkOut::Chunk(body) => (Some(body), false),
                crate::chat::ChunkOut::Done => (None, true),
            };
            if reply.chunk(payload).is_err() || reply.finished {
                return;
            }
            if end {
                return;
            }
        }
        // The bridge died mid-reply: the stream ends without message_stop,
        // the way upstream's cancel does (openai.zig:493 returns).
    });
    // The typed events out as SSE events
    let (sse_tx, sse_rx) =
        tokio::sync::mpsc::channel::<Result<SseEvent, std::convert::Infallible>>(16);
    std::thread::spawn(move || {
        while let Some(te) = event_rx.blocking_recv() {
            let event = SseEvent::default().event(te.kind).data(crate::json_write::dumps(&te.body));
            if sse_tx.blocking_send(Ok(event)).is_err() {
                break;
            }
        }
    });
    Ok(Sse::new(tokio_stream::wrappers::ReceiverStream::new(sse_rx))
        .into_response())
}

/// One typed Messages event: its kind (the SSE event name) and its body.
struct TypedEvent {
    kind: &'static str,
    body: Value,
}

type SendError = tokio::sync::mpsc::error::SendError<TypedEvent>;

/// One streamed call the Messages reply assembles from the chat stream's
/// tool_call deltas, by the delta's index. Upstream: `Call`
/// (anthropic.zig:127).
struct Call {
    id: String,
    name: String,
    arguments: String,
    sent: usize,
    done: bool,
}

/// Typed Messages events from chat chunks, each delta on a block of its own
/// type. Upstream: `Reply` (anthropic.zig:134-328): message_start opens; the
/// text and thinking blocks open at their first delta and stop at a kind
/// change or the finish; the tool_use blocks open as the calls complete,
/// their arguments as input_json_delta, the text that arrives while one is
/// open held for its close; message_delta carries the stop_reason and the
/// usage; message_stop ends the stream.
struct Reply {
    sender: tokio::sync::mpsc::Sender<TypedEvent>,
    base: Value,
    index: i64,
    kind: Option<&'static str>,
    calls: Vec<(i64, Call)>,
    pending: Vec<(&'static str, String)>,
    active: Option<i64>,
    finished: bool,
    finish: &'static str,
    stop_sequence: Value,
    tokens: Value,
}

/// `json.strOr`: the value as a string, else the empty one.
fn str_or(v: Option<&Value>) -> String {
    v.and_then(Value::as_str).unwrap_or("").to_string()
}

/// Upstream: `finishOf` (anthropic.zig:378) - the finish reasons Anthropic
/// names.
fn finish_of(reason: &str) -> &'static str {
    match reason {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        _ => "end_turn",
    }
}

impl Reply {
    /// The base message and the zero usage. Upstream: `init`
    /// (anthropic.zig:150).
    fn new(sender: tokio::sync::mpsc::Sender<TypedEvent>, model: String) -> Reply {
        let tokens = usage(None);
        let base = json!({
            "id": crate::ids::make("msg_", 32),
            "type": "message",
            "role": "assistant",
            "model": model.clone(),
            "content": [],
            "stop_reason": null,
            "stop_sequence": null,
            "usage": tokens,
        });
        Reply {
            sender,
            base,
            index: -1,
            kind: None,
            calls: Vec::new(),
            pending: Vec::new(),
            active: None,
            finished: false,
            finish: "end_turn",
            stop_sequence: Value::Null,
            tokens,
        }
    }

    fn send(&self, kind: &'static str, body: Value) -> Result<(), SendError> {
        self.sender.blocking_send(TypedEvent { kind, body })
    }

    /// Upstream: `start` (anthropic.zig:180).
    fn start(&mut self) -> Result<(), SendError> {
        self.send(
            "message_start",
            json!({"type": "message_start", "message": self.base}),
        )
    }

    /// The open block stops; the thinking block signs first - local
    /// reasoning is plaintext, no provider signature. Upstream: `close`
    /// (anthropic.zig:185).
    fn close(&mut self) -> Result<(), SendError> {
        let Some(kind) = self.kind.take() else {
            return Ok(());
        };
        if kind == "thinking" {
            self.send(
                "content_block_delta",
                json!({
                    "type": "content_block_delta", "index": self.index,
                    "delta": {"type": "signature_delta", "signature": ""},
                }),
            )?;
        }
        self.send(
            "content_block_stop",
            json!({"type": "content_block_stop", "index": self.index}),
        )
    }

    /// One delta on its block: a kind change closes and opens, a tool block
    /// open holds the text for its close. Upstream: `text`
    /// (anthropic.zig:204).
    fn text(&mut self, kind: &'static str, value: &str) -> Result<(), SendError> {
        if value.is_empty() {
            return Ok(());
        }
        if self.active.is_some() {
            self.pending.push((kind, value.to_string()));
            return Ok(());
        }
        let key = if kind == "thinking" { "thinking" } else { "text" };
        if self.kind != Some(kind) {
            self.close()?;
            self.index += 1;
            self.kind = Some(kind);
            let mut block = Map::new();
            block.insert("type".into(), json!(kind));
            block.insert(key.into(), json!(""));
            self.send(
                "content_block_start",
                json!({
                    "type": "content_block_start", "index": self.index,
                    "content_block": Value::Object(block),
                }),
            )?;
        }
        let mut delta = Map::new();
        delta.insert("type".into(), json!(format!("{kind}_delta")));
        delta.insert(key.into(), json!(value));
        self.send(
            "content_block_delta",
            json!({
                "type": "content_block_delta", "index": self.index,
                "delta": Value::Object(delta),
            }),
        )
    }

    /// One chat chunk: the deltas place, the calls assemble, the blocks
    /// drain; the stream's end finishes. Upstream: `chunk`
    /// (anthropic.zig:244).
    fn chunk(&mut self, payload: Option<Value>) -> Result<(), SendError> {
        let Some(payload) = payload else {
            self.drain(true)?;
            self.close()?;
            self.send(
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": self.finish, "stop_sequence": self.stop_sequence},
                    "usage": self.tokens,
                }),
            )?;
            self.send("message_stop", json!({"type": "message_stop"}))?;
            self.finished = true;
            return Ok(());
        };
        if let Some(problem) = payload.get("error") {
            self.close()?;
            let message = match problem.get("message") {
                Some(Value::String(m)) => m.clone(),
                _ => "generation failed".into(),
            };
            let status = match problem.get("type") {
                Some(Value::String(k)) if k == "invalid_request_error" => 400,
                _ => 500,
            };
            let e = error_value(&message, status);
            self.send("error", e)?;
            self.finished = true;
            return Ok(());
        }
        if let Some(u) = payload.get("usage").filter(|u| !u.is_null()) {
            self.tokens = usage(Some(u));
        }
        if let Some(stop) = payload.get("stop_sequence").filter(|s| !s.is_null()) {
            self.stop_sequence = stop.clone();
        }
        if let Some(choices) = payload.get("choices").and_then(Value::as_array) {
            for choice in choices {
                if let Some(f) = choice
                    .get("finish_reason")
                    .filter(|f| crate::pyrepr::truthy(f))
                {
                    self.finish = finish_of(f.as_str().unwrap_or(""));
                }
                if !self.stop_sequence.is_null() {
                    self.finish = "stop_sequence";
                }
                let delta = choice.get("delta");
                self.text(
                    "thinking",
                    &str_or(delta.and_then(|d| d.get("reasoning_content"))),
                )?;
                self.text("text", &str_or(delta.and_then(|d| d.get("content"))))?;
                if let Some(tc) = delta
                    .and_then(|d| d.get("tool_calls"))
                    .and_then(Value::as_array)
                {
                    for call in tc {
                        let at = call.get("index").and_then(Value::as_i64).unwrap_or(0);
                        let target = match self.calls.iter_mut().find(|(k, _)| *k == at) {
                            Some((_, c)) => c,
                            None => {
                                self.calls.push((at, Call {
                                    id: String::new(),
                                    name: String::new(),
                                    arguments: String::new(),
                                    sent: 0,
                                    done: false,
                                }));
                                &mut self.calls.last_mut().unwrap().1
                            }
                        };
                        if let Some(id) = call
                            .get("id")
                            .filter(|v| crate::pyrepr::truthy(v))
                            .and_then(Value::as_str)
                        {
                            target.id = id.to_string();
                        }
                        let function = call.get("function");
                        target.name += &str_or(function.and_then(|f| f.get("name")));
                        target.arguments += &str_or(function.and_then(|f| f.get("arguments")));
                    }
                }
            }
        }
        self.drain(false)
    }

    /// The tool_use blocks: each call opens its block, streams its
    /// arguments, and closes complete; a second call waits, and the text
    /// that arrived while a block was open flushes at its close. Upstream:
    /// `drain` (anthropic.zig:285).
    fn drain(&mut self, final_flush: bool) -> Result<(), SendError> {
        while let Some(pos) = self.calls.iter().position(|(_, c)| !c.done) {
            let at = self.calls[pos].0;
            if self.calls[pos].1.id.is_empty() || self.calls[pos].1.name.is_empty() {
                return Ok(());
            }
            if self.active.is_none() {
                self.close()?;
                self.index += 1;
                self.kind = Some("tool_use");
                self.active = Some(at);
                let block = json!({
                    "type": "tool_use",
                    "id": self.calls[pos].1.id,
                    "name": self.calls[pos].1.name,
                    "input": {},
                });
                self.send(
                    "content_block_start",
                    json!({
                        "type": "content_block_start", "index": self.index,
                        "content_block": block,
                    }),
                )?;
            }
            if Some(at) != self.active {
                return Ok(());
            }
            let arguments = self.calls[pos].1.arguments.clone();
            let delta = arguments[self.calls[pos].1.sent..].to_string();
            if !delta.is_empty() {
                self.send(
                    "content_block_delta",
                    json!({
                        "type": "content_block_delta", "index": self.index,
                        "delta": {"type": "input_json_delta", "partial_json": delta},
                    }),
                )?;
                self.calls[pos].1.sent = arguments.len();
            }
            // Complete: the arguments close as a JSON object (anthropic.zig:309)
            let complete = arguments.trim_end_matches(" \t\r\n").ends_with('}')
                && serde_json::from_str::<Value>(&arguments)
                    .map(|v| v.is_object())
                    .unwrap_or(false);
            if !complete && !final_flush {
                return Ok(());
            }
            self.close()?;
            self.calls[pos].1.done = true;
            self.active = None;
            let pending = std::mem::take(&mut self.pending);
            for (kind, text) in &pending {
                self.text(kind, text)?;
            }
        }
        Ok(())
    }
}

/// The `/v1/messages` route: translate, run through the chat engine, format as
/// one Messages reply (anthropic.zig:52 `post`'s non-streaming path).
pub async fn messages(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    let count = std::env::var("TF_ANTHROPIC_PATH").is_ok(); // never set; the count route is separate
    let _ = count;
    // The engine check first, upstream's order: the engine-less build refuses
    // with its own words before any translate (the chat route's own refusal).
    if state.engine.is_none() || state.text.is_none() {
        return send_error(&state.engineless_problem, 503);
    }
    // The streaming branch (anthropic.zig's Reply drives the typed events).
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    if stream {
        let model = state.config.model_ids[0].clone();
        return match stream_messages(state.clone(), &body, model) {
            Ok(response) => response,
            Err(message) => {
                let code = if message == state.engineless_problem
                    || message == "the engine closed"
                    || message == "the engine is busy"
                {
                    StatusCode::SERVICE_UNAVAILABLE
                } else {
                    StatusCode::BAD_REQUEST
                };
                send_error(&message, code.as_u16())
            }
        };
    }
    let chat = match crate::anthropic_translate::translate(&body, false) {
        Ok(chat) => chat,
        Err(message) => return send_error(&message, 400),
    };
    let mut reply = match tokio::task::block_in_place(|| crate::chat::generate_with(&state, &chat)) {
        Ok(reply) => reply,
        Err(refused) => return send_error(&refused.message, refused.status()),
    };
    // The reply's calls attach here too (openai.zig:309): the one-call policy
    // from the translated body's parallel_tool_calls
    // (anthropic_translate.zig:248), the tools the template offered
    // (tool_specs.zig:58's active)
    let policy = crate::tool_stream::Policy::new(
        chat.get("parallel_tool_calls").and_then(Value::as_bool) == Some(false),
    );
    let tools = match crate::tool_specs::active(chat.get("tools"), chat.get("tool_choice")) {
        Ok(tools) => tools,
        Err(m) => return send_error(&m, 400),
    };
    let _ = crate::chat::attach_calls(&mut reply, &tools, &policy);
    if let Some(refused) = &reply.failed {
        return send_error(&refused.message, refused.status());
    }
    // The whole reply as one Messages response (anthropic.zig:329 `completion`):
    // content blocks in the reply's order, the usage as Anthropic's.
    let mut content: Vec<Value> = Vec::new();
    if !reply.reasoning.is_empty() {
        content.push(json!({
            "type": "thinking",
            "thinking": reply.reasoning,
            "signature": "",
        }));
    }
    if !reply.content.is_empty() {
        content.push(json!({"type": "text", "text": reply.content}));
    }
    // The tool calls as tool_use blocks (anthropic.zig:347).
    if let Some(calls) = &reply.tool_calls {
        for call in calls {
            let function = call.get("function").unwrap_or(&Value::Null);
            let raw = function.get("arguments").and_then(Value::as_str).unwrap_or("{}");
            let args: Value = serde_json::from_str(raw).unwrap_or(json!({}));
            content.push(json!({
                "type": "tool_use",
                "id": call.get("id").cloned().unwrap_or(json!("")),
                "name": function.get("name").cloned().unwrap_or(json!("")),
                "input": args,
            }));
        }
    }
    let id = crate::ids::make("msg_", 32);
    let model = state.config.model_ids[0].clone();
    let usage = usage(reply.usage.as_ref());
    let stop_reason = if reply.tool_calls.is_some() {
        "tool_use"
    } else {
        reply.stop_sequence.as_deref().map_or(
            reply.finish_reason,
            |_| "stop_sequence",
        )
    };
    (
        StatusCode::OK,
        crate::json_write::PythonJson(json!({
            "id": id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": reply.stop_sequence,
            "usage": usage,
        })),
    )
        .into_response()
}

/// `/v1/messages/count_tokens`: the rendered prompt's token count
/// (anthropic.zig:47 `countTokens`).
pub async fn count_tokens(
    State(state): State<Arc<AppState>>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    if state.engine.is_none() || state.text.is_none() {
        return send_error(&state.engineless_problem, 503);
    }
    let chat = match crate::anthropic_translate::translate(&body, true) {
        Ok(chat) => chat,
        Err(message) => return send_error(&message, 400),
    };
    match tokio::task::block_in_place(|| crate::chat::count_tokens(&state, &chat)) {
        Ok(n) => (StatusCode::OK, crate::json_write::PythonJson(json!({"input_tokens": n}))).into_response(),
        Err(message) => send_error(&message, 400),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The typed event sequence one Reply writes for a stream that reasons,
    /// writes text, and calls a tool — the state machine of anthropic.zig's
    /// chunk/text/drain (anthropic.zig:204-320).
    #[test]
    fn the_reply_translates_the_chunk_stream() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<TypedEvent>(64);
        let mut r = Reply::new(tx, "claude-test".into());
        let read = |rx: &mut tokio::sync::mpsc::Receiver<TypedEvent>| {
            rx.blocking_recv().expect("the event channel closed")
        };
        r.start().unwrap();
        // The reasoning opens the thinking block
        r.chunk(Some(json!({"choices": [{"delta": {"reasoning_content": "think"}}]})))
            .unwrap();
        // The text changes the block
        r.chunk(Some(json!({"choices": [{"delta": {"content": "hi"}}]})))
            .unwrap();
        // The call's head opens the tool_use block (its id and name known)
        r.chunk(Some(json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": ""}}
        ]}}]})))
            .unwrap();
        // Text while the call is open waits for its close
        r.chunk(Some(json!({"choices": [{"delta": {"content": "before"}}]})))
            .unwrap();
        // The complete arguments close the block and flush the held text
        r.chunk(Some(json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "{\"city\": \"Rome\"}"}}
        ]}}]})))
            .unwrap();
        // The finish names the stop and the usage
        r.chunk(Some(json!({
            "choices": [{"finish_reason": "tool_calls", "delta": {}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5}
        })))
            .unwrap();
        r.chunk(None).unwrap();
        drop(r);

        // message_start, then the thinking block
        let e = read(&mut rx);
        assert_eq!(e.kind, "message_start");
        assert_eq!(e.body["message"]["id"].as_str().map(|x| x.len()), Some(4 + 32));
        let e = read(&mut rx);
        assert_eq!(e.kind, "content_block_start");
        assert_eq!(e.body["content_block"]["type"], "thinking");
        let e = read(&mut rx);
        assert_eq!(e.kind, "content_block_delta");
        assert_eq!(e.body["delta"]["thinking"], "think");
        // the thinking block signs before it stops (anthropic.zig:187)
        let e = read(&mut rx);
        assert_eq!(e.kind, "content_block_delta");
        assert_eq!(e.body["delta"]["type"], "signature_delta");
        assert_eq!(e.body["delta"]["signature"], "");
        // the text block
        let e = read(&mut rx);
        assert_eq!(e.kind, "content_block_stop");
        let e = read(&mut rx);
        assert_eq!(e.kind, "content_block_start");
        assert_eq!(e.body["content_block"]["type"], "text");
        let e = read(&mut rx);
        assert_eq!(e.kind, "content_block_delta");
        assert_eq!(e.body["delta"]["text"], "hi");
        // the tool_use block opens
        let e = read(&mut rx);
        assert_eq!(e.kind, "content_block_stop");
        let e = read(&mut rx);
        assert_eq!(e.kind, "content_block_start");
        assert_eq!(e.body["content_block"]["type"], "tool_use");
        assert_eq!(e.body["content_block"]["id"], "call_1");
        assert_eq!(e.body["content_block"]["name"], "get_weather");
        // its arguments go out, its block closes
        let e = read(&mut rx);
        assert_eq!(e.kind, "content_block_delta");
        assert_eq!(e.body["delta"]["type"], "input_json_delta");
        assert_eq!(e.body["delta"]["partial_json"], "{\"city\": \"Rome\"}");
        let e = read(&mut rx);
        assert_eq!(e.kind, "content_block_stop");
        // the held text flushes on its own block
        let e = read(&mut rx);
        assert_eq!(e.kind, "content_block_start");
        assert_eq!(e.body["content_block"]["type"], "text");
        let e = read(&mut rx);
        assert_eq!(e.kind, "content_block_delta");
        assert_eq!(e.body["delta"]["text"], "before");
        // the end
        let e = read(&mut rx);
        assert_eq!(e.kind, "content_block_stop");
        let e = read(&mut rx);
        assert_eq!(e.kind, "message_delta");
        assert_eq!(e.body["delta"]["stop_reason"], "tool_use");
        assert_eq!(e.body["usage"]["input_tokens"], 3);
        assert_eq!(e.body["usage"]["output_tokens"], 5);
        let e = read(&mut rx);
        assert_eq!(e.kind, "message_stop");
        assert!(rx.blocking_recv().is_none());
    }

    /// The stop_sequence chunk field turns the stop reason, and an
    /// incomplete call still closes at the stream's end (anthropic.zig:298).
    #[test]
    fn the_matched_stop_names_the_reason_and_the_end_completes_the_call() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<TypedEvent>(64);
        let mut r = Reply::new(tx, "claude-test".into());
        r.start().unwrap();
        // A call that never completes
        r.chunk(Some(json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_9", "type": "function",
             "function": {"name": "lookup", "arguments": "{\"q\": \"a\""}}
        ]}}]})))
            .unwrap();
        // The stop string matched: the reason is stop_sequence
        r.chunk(Some(json!({
            "choices": [{"finish_reason": "stop", "delta": {}}],
            "stop_sequence": "END", "usage": {"prompt_tokens": 1, "completion_tokens": 2}
        })))
            .unwrap();
        r.chunk(None).unwrap();
        drop(r);
        // The block opened, the arguments went out, and the end closed it
        let mut saw_json_delta = false;
        let mut end = None;
        while let Some(e) = rx.blocking_recv() {
            match e.kind {
                "content_block_delta" => saw_json_delta = true,
                "message_delta" => end = Some(e.body),
                _ => {}
            }
        }
        assert!(saw_json_delta);
        let end = end.expect("the message_delta");
        assert_eq!(end["delta"]["stop_reason"], "stop_sequence");
        assert_eq!(end["delta"]["stop_sequence"], "END");
    }
}
