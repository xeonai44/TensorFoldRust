//! OpenAI's Responses API: each request runs as its chat completion,
//! translated item by item and event by event.
//!
//! Port of `zig/src/server/responses.zig` (486 lines): the in-memory store
//! the collection serves from (GET, DELETE and `previous_response_id`), the
//! route's own refusal shape, and the Reply machine that turns chat
//! completion deltas into output items and typed events — reasoning, text
//! and function calls in order, streamed as `response.*` events or assembled
//! whole by `completion` (responses.zig:430).

use crate::errors::Refused;
use crate::state::AppState;
use axum::extract::{OriginalUri, Path, State};
use axum::http::StatusCode;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::sse::Event as SseEvent;
use axum::response::IntoResponse;
use serde_json::{json, Map, Value};
use std::sync::{Arc, Mutex};

type SendError = tokio::sync::mpsc::error::SendError<TypedEvent>;

/// One typed event: its kind (the SSE event name) and its body.
struct TypedEvent {
    kind: &'static str,
    body: Value,
}

/// The route's own refusal: the message, the type, the param and the code
/// (responses.zig:143's `refuse`) — translate failures carry no param, the
/// store's 404s name `response_id`.
fn refuse(message: &str, status: u16, param: Option<&str>) -> axum::response::Response {
    let body = json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": null,
        }
    });
    (
        StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_REQUEST),
        crate::json_write::PythonJson(body),
    )
        .into_response()
}

/// A stored response with the messages it added. Upstream: `Entry`
/// (responses.zig:35).
struct Entry {
    id: String,
    response: Value,
    parent: Option<String>,
    messages: Vec<Value>,
    size: usize,
}

#[derive(Default)]
struct StoreInner {
    /// The newest stored responses, oldest first.
    entries: Vec<Entry>,
    bytes: usize,
}

/// The newest stored responses, in memory, for GET, DELETE and
/// ``previous_response_id``. Upstream: `Store` (responses.zig:44): 1,024
/// entries, 256 MiB.
#[derive(Clone, Default)]
pub struct Store {
    inner: Arc<Mutex<StoreInner>>,
}

impl Store {
    fn find(s: &StoreInner, id: &str) -> Option<usize> {
        s.entries.iter().position(|e| e.id == id)
    }

    /// Keeps a finished response with the messages it added; the oldest go
    /// past 1,024 entries or 256 MiB. Upstream: `put` (responses.zig:64) —
    /// the response's own output items translate back into chat messages
    /// (a failed translation stores nothing, the caller's `catch {}`).
    pub fn put(&self, response: &Value, added: &[Value]) {
        let id = response
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let parent = response
            .get("previous_response_id")
            .and_then(Value::as_str)
            .map(str::to_string);
        let output = response
            .get("output")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let Ok(from_output) = crate::responses_translate::messages(&output) else {
            return;
        };
        let mut messages = added.to_vec();
        messages.extend(from_output);
        let size = serde_json::to_string(response).map(|s| s.len()).unwrap_or(0)
            + serde_json::to_string(&Value::Array(messages.clone()))
                .map(|s| s.len())
                .unwrap_or(0);
        let entry = Entry {
            id,
            response: response.clone(),
            parent,
            messages,
            size,
        };
        let mut s = self.inner.lock().unwrap();
        if let Some(i) = Self::find(&s, &entry.id) {
            let old = s.entries.remove(i);
            s.bytes -= old.size;
        }
        s.bytes += entry.size;
        s.entries.push(entry);
        while !s.entries.is_empty()
            && (s.entries.len() > 1024 || s.bytes > 256 * 1024 * 1024)
        {
            let old = s.entries.remove(0);
            s.bytes -= old.size;
        }
    }

    /// The stored response, a fresh copy. Upstream: `get`
    /// (responses.zig:87).
    fn get(&self, id: &str) -> Option<Value> {
        let s = self.inner.lock().unwrap();
        Self::find(&s, id).map(|i| s.entries[i].response.clone())
    }

    /// Removes a stored response. Upstream: `delete` (responses.zig:93).
    fn delete(&self, id: &str) -> bool {
        let mut s = self.inner.lock().unwrap();
        match Self::find(&s, id) {
            Some(i) => {
                let old = s.entries.remove(i);
                s.bytes -= old.size;
                true
            }
            None => false,
        }
    }

    /// Every message of a stored response's chain, oldest first (the chain
    /// stays newest in the store). Upstream: `History.conversation`
    /// (responses.zig:114) — each found entry moves to the back, a missing
    /// parent the exact refusal.
    pub(crate) fn conversation(&self, rid: &Value) -> Result<Vec<Value>, Refused> {
        let Some(at) = rid.as_str() else {
            return Err(Refused::request("previous_response_id must be a string"));
        };
        let mut s = self.inner.lock().unwrap();
        let mut chain: Vec<Vec<Value>> = Vec::new();
        let mut at = Some(at.to_string());
        while let Some(id) = at.clone() {
            let Some(i) = Self::find(&s, &id) else {
                return Err(Refused::request(format!(
                    "previous response {} is not stored here (store: false, deleted, or evicted): send the conversation's items instead",
                    crate::pyrepr::repr(&json!(id)),
                )));
            };
            let e = s.entries.remove(i);
            at = e.parent.clone();
            s.entries.push(e);
            chain.push(s.entries.last().unwrap().messages.clone());
        }
        drop(s);
        let mut out: Vec<Value> = Vec::new();
        for part in chain.iter().rev() {
            out.extend(part.iter().cloned());
        }
        Ok(out)
    }
}

/// GET /v1/responses/:id — the stored response, or the 404 naming it
/// (responses.zig:156's `get`).
pub async fn get_one(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> axum::response::Response {
    match state.responses.get(&id) {
        Some(v) => (StatusCode::OK, crate::json_write::PythonJson(v)).into_response(),
        None => refuse(
            &format!(
                "no stored response has id {}",
                crate::pyrepr::repr(&json!(id)),
            ),
            404,
            Some("response_id"),
        ),
    }
}

/// DELETE on the collection — the empty id named (responses.zig:160's
/// `delete` with rid = "").
pub async fn delete_collection() -> axum::response::Response {
    refuse(
        "no stored response has id ''",
        404,
        Some("response_id"),
    )
}

/// DELETE /v1/responses/:id — the stored response removed
/// (responses.zig:160's `delete`).
pub async fn delete_one(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> axum::response::Response {
    if state.responses.delete(&id) {
        let body = json!({"id": id, "object": "response", "deleted": true});
        return (StatusCode::OK, crate::json_write::PythonJson(body)).into_response();
    }
    refuse(
        &format!(
            "no stored response has id {}",
            crate::pyrepr::repr(&json!(id)),
        ),
        404,
        Some("response_id"),
    )
}

/// Upstream's unknown-path refusal (routes.zig:82's `unknown`): the GET
/// on the collection and the POST on an id both fall through to it.
fn unknown_path(uri: &axum::http::Uri) -> axum::response::Response {
    let body = json!({"error": {"message": format!("unknown path {}", uri.path())}});
    (StatusCode::NOT_FOUND, crate::json_write::PythonJson(body)).into_response()
}

/// GET on the collection — upstream falls through to the unknown-path
/// refusal (routes.zig:47 leaves the empty rid alone).
pub async fn collection_get(
    OriginalUri(uri): OriginalUri,
) -> axum::response::Response {
    unknown_path(&uri)
}

/// POST on an id — the collection alone answers POSTs
/// (routes.zig:68's empty-rid check).
pub async fn id_post(
    OriginalUri(uri): OriginalUri,
) -> axum::response::Response {
    unknown_path(&uri)
}

/// POST /v1/responses. Upstream: `post` (responses.zig:170).
pub async fn post(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> axum::response::Response {
    if state.engine.is_none() || state.text.is_none() {
        // The engine-less build's refusal, the same words every inference
        // route uses (the port convention; upstream's comes from the plan).
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            crate::json_write::PythonJson(json!({
                "error": {"message": state.engineless_problem, "type": "engine_unavailable"}
            })),
        )
            .into_response();
    }
    // The raw body read (responses.zig:174-178): an empty body is `{}`, a
    // malformed one the exact refusal
    let parsed: Value = if body.is_empty() {
        json!({})
    } else {
        match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(_) => return refuse("the request body is not JSON", 400, None),
        }
    };
    // The translation first: its refusals carry this route's own error
    // shape (responses.zig:176's refuse with no param)
    let request = match crate::responses_translate::translate(&parsed, &state.responses) {
        Ok(r) => r,
        Err(refused) => return refuse(&refused.message, refused.status(), None),
    };
    // The base every event and the final carry (responses.zig:179-188):
    // the id, the object, the times, the model, the empty output and usage,
    // then the echo fields
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut base = Map::new();
    base.insert("id".into(), json!(crate::ids::make("resp_", 32)));
    base.insert("object".into(), json!("response"));
    base.insert("created_at".into(), json!(now));
    base.insert("status".into(), json!("in_progress"));
    base.insert("error".into(), Value::Null);
    base.insert("incomplete_details".into(), Value::Null);
    base.insert("model".into(), json!(crate::chat::reply_model(&state, &parsed)));
    base.insert("output".into(), json!([]));
    base.insert("usage".into(), Value::Null);
    for (k, v) in &request.echo {
        base.insert(k.clone(), v.clone());
    }
    // The chat machinery's own reads: the tools the template offers, the
    // one-call policy (openai.zig's plan) — its refusal carries the chat
    // error shape, the plan's own
    let policy = crate::tool_stream::Policy::new(
        request
            .chat
            .get("parallel_tool_calls")
            .and_then(Value::as_bool)
            == Some(false),
    );
    let tools = match crate::tool_specs::active(request.chat.get("tools"), request.chat.get("tool_choice")) {
        Ok(tools) => tools,
        Err(m) => {
            return (
                StatusCode::BAD_REQUEST,
                crate::json_write::PythonJson(crate::errors::wrap_error(json!({"message": m}))),
            )
                .into_response()
        }
    };
    if request.stream {
        return match stream_response(state.clone(), &request, base, tools, policy) {
            Ok(response) => response,
            Err(refused) => (
                StatusCode::from_u16(refused.status()).unwrap_or(StatusCode::BAD_REQUEST),
                crate::json_write::PythonJson(crate::errors::wrap_error(crate::errors::error_body(
                    &refused, "messages",
                ))),
            )
                .into_response(),
        };
    }
    // The whole path (openai.zig:355's whole through the Wire): the chat
    // completion runs, its whole body feeds `completion`, the final
    // response goes out whole
    let mut reply = match tokio::task::block_in_place(|| crate::chat::generate_with(&state, &request.chat)) {
        Ok(reply) => reply,
        Err(refused) => {
            return (
                StatusCode::from_u16(refused.status()).unwrap_or(StatusCode::BAD_REQUEST),
                crate::json_write::PythonJson(crate::errors::wrap_error(crate::errors::error_body(
                    &refused, "messages",
                ))),
            )
                .into_response()
        }
    };
    let calls = crate::chat::attach_calls(&mut reply, &tools, &policy);
    let id = crate::ids::make("chatcmpl-", 32);
    let model = crate::chat::reply_model(&state, &request.chat);
    let whole = crate::chat::whole_body(&id, &model, now, &reply, calls.as_deref());
    let mut machine = Reply {
        sender: None,
        store: state.responses.clone(),
        keep: request.store,
        added: request.added.clone(),
        base,
        items: Vec::new(),
        current: None,
        calls: Vec::new(),
        seq: 0,
        final_: None,
    };
    let final_response = match machine.completion(&whole) {
        Ok(v) => v,
        Err(_) => return StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    };
    // The next turn's prompt prefilled in the background (openai.zig:363's
    // warmNext), from the reply as its client got it
    let conversation = request
        .chat
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    crate::warm::reply(
        &state,
        &conversation,
        &tools,
        &crate::warm::message_of(&reply, calls.as_ref()),
        reply.thinking,
        reply.effort.as_deref(),
    );
    (StatusCode::OK, crate::json_write::PythonJson(final_response)).into_response()
}

/// The streamed path: the chat bridge's chunks through the machine, the
/// typed events out as SSE (responses.zig:208's Wire).
fn stream_response(
    state: Arc<AppState>,
    request: &crate::responses_translate::Request,
    base: Map<String, Value>,
    tools: Vec<Value>,
    policy: crate::tool_stream::Policy,
) -> Result<axum::response::Response, Refused> {
    let start = match tokio::task::block_in_place(|| crate::chat::submit_streaming(&state, &request.chat))
    {
        Ok(start) => start,
        // A context-window refusal reaches the client inside the stream
        // (openai.zig:345's streamedContext through the Wire): the stream
        // opens — response.created, response.in_progress — then the error
        // chunk fails the reply: response.failed, the stream ends
        Err(refused) if refused.kind == crate::errors::Kind::ContextLength => {
            return Ok(context_refused_sse(base, &refused));
        }
        // Every other refusal is still a plain error reply before the
        // stream opens (openai.zig:463's unsent)
        Err(refused) => return Err(refused),
    };
    let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel::<crate::chat::ChunkOut>(16);
    let text_arc = state.text.clone().expect("checked above");
    let chunk_id = crate::ids::make("chatcmpl-", 32);
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let model = crate::chat::reply_model(&state, &request.chat);
    let chunk_tools = tools.clone();
    let chunk_policy = policy.clone();
    let bridge_state = state.clone();
    std::thread::spawn(move || {
        let _ = crate::chat::bridge(
            start,
            chunk_tx,
            chunk_id,
            model,
            created,
            text_arc,
            chunk_tools,
            chunk_policy,
            // The chat-shaped chunks the Responses machine reads
            true,
            false,
            bridge_state,
        );
    });
    let (event_tx, event_rx) = tokio::sync::mpsc::channel::<TypedEvent>(16);
    let machine_store = state.responses.clone();
    let keep = request.store;
    let added = request.added.clone();
    std::thread::spawn(move || {
        let mut machine = Reply {
            sender: Some(event_tx),
            store: machine_store,
            keep,
            added,
            base,
            items: Vec::new(),
            current: None,
            calls: Vec::new(),
            seq: 0,
            final_: None,
        };
        if machine.start().is_err() {
            return;
        }
        while let Some(out) = chunk_rx.blocking_recv() {
            let (payload, end) = match out {
                crate::chat::ChunkOut::Chunk(body) => (Some(body), false),
                crate::chat::ChunkOut::Done => (None, true),
            };
            if machine.chunk(payload.as_ref()).is_err() {
                return;
            }
            if machine.final_.is_some() {
                return;
            }
            if end {
                return;
            }
        }
        // The bridge died mid-reply: the stream ends without a terminal
        // event, the way upstream's cancel does
    });
    Ok(sse_of(event_rx))
}

/// The context refusal's own stream: the machine opens and fails on the
/// error body, its events out as SSE.
fn context_refused_sse(base: Map<String, Value>, refused: &Refused) -> axum::response::Response {
    let (event_tx, event_rx) = tokio::sync::mpsc::channel::<TypedEvent>(8);
    let error = crate::errors::error_body(refused, "messages");
    std::thread::spawn(move || {
        let mut machine = Reply {
            sender: Some(event_tx),
            store: Store::default(),
            keep: false,
            added: Vec::new(),
            base,
            items: Vec::new(),
            current: None,
            calls: Vec::new(),
            seq: 0,
            final_: None,
        };
        let _ = machine.start();
        let _ = machine.fail(&error);
    });
    sse_of(event_rx)
}

/// The typed events out as SSE events.
fn sse_of(
    mut event_rx: tokio::sync::mpsc::Receiver<TypedEvent>,
) -> axum::response::Response {
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
    Sse::new(tokio_stream::wrappers::ReceiverStream::new(sse_rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// A Response built from chat-completion deltas, in order: reasoning, text,
/// calls; streamed as typed events. Upstream: `Reply`
/// (responses.zig:232-483).
struct Reply {
    /// The typed events out; None on the whole path, where the events
    /// return early (responses.zig:264's `if (!r.streaming) return`).
    sender: Option<tokio::sync::mpsc::Sender<TypedEvent>>,
    store: Store,
    keep: bool,
    added: Vec<Value>,
    base: Map<String, Value>,
    items: Vec<Map<String, Value>>,
    current: Option<usize>,
    /// One streamed call by its delta index: the item it opened into
    /// (responses.zig:240's calls).
    calls: Vec<(i64, usize)>,
    seq: u64,
    final_: Option<Value>,
}

/// A nonempty string's text. Upstream: `textOf` (responses.zig:364).
fn text_of(v: Option<&Value>) -> Option<&str> {
    v.and_then(Value::as_str).filter(|s| !s.is_empty())
}

impl Reply {
    /// One event: ``type``, its sequence number, then its fields, written
    /// at once as ``event:`` and ``data:``. Upstream: `event`
    /// (responses.zig:250).
    fn event(&mut self, kind: &'static str, fields: Vec<(&str, Value)>) -> Result<(), SendError> {
        let Some(sender) = &self.sender else {
            return Ok(());
        };
        let mut o = Map::new();
        o.insert("type".into(), json!(kind));
        o.insert("sequence_number".into(), json!(self.seq));
        self.seq += 1;
        for (k, v) in fields {
            o.insert(k.into(), v);
        }
        sender.blocking_send(TypedEvent {
            kind,
            body: Value::Object(o),
        })
    }

    /// Upstream: `start` (responses.zig:272).
    fn start(&mut self) -> Result<(), SendError> {
        let base = Value::Object(self.base.clone());
        self.event(
            "response.created",
            vec![("response", base.clone())],
        )?;
        self.event("response.in_progress", vec![("response", base)])
    }

    /// One output item opened: a message, a reasoning block or a function
    /// call, its part added and announced. Upstream: `open`
    /// (responses.zig:282).
    fn open(&mut self, kind: &str, call_id: &str, name: &str) -> Result<usize, SendError> {
        self.close("completed")?;
        let mut item = Map::new();
        let mut part: Option<Map<String, Value>> = None;
        if kind == "message" {
            item.insert("id".into(), json!(crate::ids::make("msg_", 32)));
            item.insert("type".into(), json!("message"));
            item.insert("role".into(), json!("assistant"));
            item.insert("status".into(), json!("in_progress"));
            item.insert("content".into(), json!([]));
            let mut p = Map::new();
            p.insert("type".into(), json!("output_text"));
            p.insert("text".into(), json!(""));
            p.insert("annotations".into(), json!([]));
            p.insert("logprobs".into(), json!([]));
            part = Some(p);
        } else if kind == "reasoning" {
            item.insert("id".into(), json!(crate::ids::make("rs_", 32)));
            item.insert("type".into(), json!("reasoning"));
            item.insert("summary".into(), json!([]));
            item.insert("content".into(), json!([]));
            item.insert("status".into(), json!("in_progress"));
            let mut p = Map::new();
            p.insert("type".into(), json!("reasoning_text"));
            p.insert("text".into(), json!(""));
            part = Some(p);
        } else {
            item.insert("id".into(), json!(crate::ids::make("fc_", 32)));
            item.insert("type".into(), json!("function_call"));
            item.insert("call_id".into(), json!(call_id));
            item.insert("name".into(), json!(name));
            item.insert("arguments".into(), json!(""));
            item.insert("status".into(), json!("in_progress"));
        }
        self.items.push(item);
        let at = self.items.len() - 1;
        self.current = Some(at);
        let item_value = Value::Object(self.items[at].clone());
        self.event(
            "response.output_item.added",
            vec![
                ("output_index", json!(at)),
                ("item", item_value),
            ],
        )?;
        if let Some(p) = part {
            self.items[at].insert("content".into(), json!([Value::Object(p.clone())]));
            let item_id = self.items[at].get("id").cloned().unwrap_or(Value::Null);
            self.event(
                "response.content_part.added",
                vec![
                    ("item_id", item_id),
                    ("output_index", json!(at)),
                    ("content_index", json!(0)),
                    ("part", Value::Object(p)),
                ],
            )?;
        }
        Ok(at)
    }

    /// The open item closed with its status, its part's text done.
    /// Upstream: `close` (responses.zig:319).
    fn close(&mut self, status: &str) -> Result<(), SendError> {
        let Some(at) = self.current.take() else {
            return Ok(());
        };
        self.items[at].insert("status".into(), json!(status));
        let where_id = self.items[at].get("id").cloned().unwrap_or(Value::Null);
        let where_index = json!(at);
        if self.items[at].get("type").and_then(Value::as_str) == Some("function_call") {
            self.event(
                "response.function_call_arguments.done",
                vec![
                    ("item_id", where_id),
                    ("output_index", where_index),
                    ("name", self.items[at].get("name").cloned().unwrap_or(Value::Null)),
                    (
                        "arguments",
                        self.items[at].get("arguments").cloned().unwrap_or(Value::Null),
                    ),
                ],
            )?;
        } else {
            let part = self.items[at]
                .get("content")
                .and_then(Value::as_array)
                .and_then(|c| c.first())
                .cloned()
                .unwrap_or(Value::Null);
            if self.items[at].get("type").and_then(Value::as_str) == Some("message") {
                self.event(
                    "response.output_text.done",
                    vec![
                        ("item_id", where_id.clone()),
                        ("output_index", where_index.clone()),
                        ("content_index", json!(0)),
                        ("text", part.get("text").cloned().unwrap_or(Value::Null)),
                        ("logprobs", json!([])),
                    ],
                )?;
            } else {
                self.event(
                    "response.reasoning_text.done",
                    vec![
                        ("item_id", where_id.clone()),
                        ("output_index", where_index.clone()),
                        ("content_index", json!(0)),
                        ("text", part.get("text").cloned().unwrap_or(Value::Null)),
                    ],
                )?;
            }
            self.event(
                "response.content_part.done",
                vec![
                    ("item_id", where_id),
                    ("output_index", where_index),
                    ("content_index", json!(0)),
                    ("part", part),
                ],
            )?;
        }
        let item_value = Value::Object(self.items[at].clone());
        self.event(
            "response.output_item.done",
            vec![("output_index", json!(at)), ("item", item_value)],
        )
    }

    /// Text onto the open item of its kind, its delta out. Upstream:
    /// `write` (responses.zig:351).
    fn write(&mut self, kind: &str, text: &str) -> Result<(), SendError> {
        let reuse = self
            .current
            .is_some_and(|at| self.items[at].get("type").and_then(Value::as_str) == Some(kind));
        let at = if reuse {
            self.current.unwrap()
        } else {
            self.open(kind, "", "")?
        };
        let part_text = self.items[at]
            .get("content")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
            .and_then(|p| p.get("text"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let joined = format!("{}{}", part_text, text);
        if let Some(Value::Array(list)) = self.items[at].get_mut("content") {
            if let Some(Value::Object(p)) = list.first_mut() {
                p.insert("text".into(), json!(joined));
            }
        }
        let where_id = self.items[at].get("id").cloned().unwrap_or(Value::Null);
        let where_index = json!(at);
        if kind == "message" {
            self.event(
                "response.output_text.delta",
                vec![
                    ("item_id", where_id),
                    ("output_index", where_index),
                    ("content_index", json!(0)),
                    ("delta", json!(text)),
                    ("logprobs", json!([])),
                ],
            )
        } else {
            self.event(
                "response.reasoning_text.delta",
                vec![
                    ("item_id", where_id),
                    ("output_index", where_index),
                    ("content_index", json!(0)),
                    ("delta", json!(text)),
                ],
            )
        }
    }

    /// One delta: the reasoning and the text written, the calls' deltas
    /// appended by index. Upstream: `delta` (responses.zig:371).
    fn delta(&mut self, d: &Value) -> Result<(), SendError> {
        let Some(obj) = d.as_object() else {
            return Ok(());
        };
        let thought = text_of(obj.get("reasoning_content")).or_else(|| text_of(obj.get("reasoning")));
        if let Some(t) = thought {
            self.write("reasoning", t)?;
        }
        if let Some(c) = text_of(obj.get("content")) {
            self.write("message", c)?;
        }
        let Some(calls) = obj.get("tool_calls").and_then(Value::as_array) else {
            return Ok(());
        };
        for call in calls {
            let function = call.get("function").unwrap_or(&Value::Null);
            let at: i64 = call
                .get("index")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let item = match self.calls.iter().find(|(k, _)| *k == at) {
                Some((_, i)) => *i,
                None => {
                    let id = text_of(call.get("id"))
                        .map(str::to_string)
                        .unwrap_or_else(|| crate::ids::make("call_", 32));
                    let name = text_of(function.get("name")).unwrap_or("");
                    let made = self.open("function_call", &id, name)?;
                    self.calls.push((at, made));
                    made
                }
            };
            if let Some(args) = text_of(function.get("arguments")) {
                let held = self.items[item]
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                self.items[item].insert("arguments".into(), json!(format!("{}{}", held, args)));
                self.event(
                    "response.function_call_arguments.delta",
                    vec![
                        ("item_id", self.items[item].get("id").cloned().unwrap_or(Value::Null)),
                        ("output_index", json!(item)),
                        ("delta", json!(args)),
                    ],
                )?;
            }
        }
        Ok(())
    }

    /// The usage as the Response's. Upstream: `usage`
    /// (responses.zig:391).
    fn usage(&self, chat: Option<&Value>) -> Value {
        let Some(c) = chat else {
            return Value::Null;
        };
        if !crate::pyrepr::truthy(c) {
            return Value::Null;
        }
        let cached = c
            .get("prompt_tokens_details")
            .filter(|d| crate::pyrepr::truthy(d))
            .and_then(|d| d.get("cached_tokens"))
            .cloned()
            .unwrap_or(json!(0));
        let thought = c
            .get("completion_tokens_details")
            .filter(|d| crate::pyrepr::truthy(d))
            .and_then(|d| d.get("reasoning_tokens"))
            .cloned()
            .unwrap_or(json!(0));
        let prompt = c.get("prompt_tokens").cloned().unwrap_or(json!(0));
        let generated = c.get("completion_tokens").cloned().unwrap_or(json!(0));
        let total = prompt.as_u64().unwrap_or(0) + generated.as_u64().unwrap_or(0);
        json!({
            "input_tokens": prompt,
            "input_tokens_details": {"cached_tokens": cached},
            "output_tokens": generated,
            "output_tokens_details": {"reasoning_tokens": thought},
            "total_tokens": total,
        })
    }

    /// The items as the output array. Upstream: `itemsValue`
    /// (responses.zig:407).
    fn items_value(&self) -> Value {
        Value::Array(
            self.items
                .iter()
                .map(|item| Value::Object(item.clone()))
                .collect(),
        )
    }

    /// The reply finished: the open item closed, the final response
    /// assembled — incomplete on a length, completed otherwise, an
    /// all-reasoning reply still a message — stored when asked. Upstream:
    /// `finish` (responses.zig:413).
    fn finish(
        &mut self,
        reason: Option<&str>,
        chat_usage: Option<&Value>,
        stats: Option<&Value>,
    ) -> Result<Value, SendError> {
        let incomplete = reason == Some("length");
        let status = if incomplete { "incomplete" } else { "completed" };
        if !incomplete {
            let all_reasoning = self.items.iter().all(|item| {
                item.get("type").and_then(Value::as_str) == Some("reasoning")
            });
            if all_reasoning {
                self.open("message", "", "")?; // an empty answer is still a message
            }
        }
        self.close(status)?;
        let mut final_response = self.base.clone();
        final_response.insert("status".into(), json!(status));
        final_response.insert("output".into(), self.items_value());
        final_response.insert("usage".into(), self.usage(chat_usage));
        if incomplete {
            final_response.insert(
                "incomplete_details".into(),
                json!({"reason": "max_output_tokens"}),
            );
        } else {
            final_response.insert("incomplete_details".into(), Value::Null);
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            final_response.insert("completed_at".into(), json!(now));
        }
        if let Some(s) = stats {
            if !s.is_null() {
                final_response.insert("tensorfold".into(), s.clone());
            }
        }
        let out = Value::Object(final_response);
        self.final_ = Some(out.clone());
        if self.keep {
            self.store.put(&out, &self.added);
        }
        let kind = if incomplete {
            "response.incomplete"
        } else {
            "response.completed"
        };
        self.event(kind, vec![("response", out.clone())])?;
        Ok(out)
    }

    /// The reply failed: the open item closed incomplete, the error named,
    /// nothing stored. Upstream: `fail` (responses.zig:438).
    fn fail(&mut self, problem: &Value) -> Result<(), SendError> {
        self.close("incomplete")?;
        let message = problem.get("message");
        let kind = problem.get("type");
        let is_request = kind.and_then(Value::as_str) == Some("invalid_request_error");
        let mut e = Map::new();
        e.insert(
            "code".into(),
            json!(if is_request { "invalid_prompt" } else { "server_error" }),
        );
        let text = if let Value::String(s) = problem {
            s.clone()
        } else if let Some(Value::String(m)) = message.filter(|m| crate::pyrepr::truthy(m)) {
            m.clone()
        } else {
            "the reply failed".into()
        };
        e.insert("message".into(), json!(text));
        let mut final_response = self.base.clone();
        final_response.insert("status".into(), json!("failed"));
        final_response.insert("output".into(), self.items_value());
        final_response.insert("error".into(), Value::Object(e));
        let out = Value::Object(final_response);
        self.final_ = Some(out.clone());
        self.event("response.failed", vec![("response", out)])
    }

    /// One chat-completion stream event (None: its ``[DONE]``). Upstream:
    /// `chunk` (responses.zig:463).
    fn chunk(&mut self, payload: Option<&Value>) -> Result<(), SendError> {
        if self.final_.is_some() {
            return Ok(());
        }
        let Some(p) = payload else {
            return self.fail(&json!("the reply ended early"));
        };
        if let Some(problem) = p.get("error") {
            return self.fail(problem);
        }
        let choice = p
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
            .cloned()
            .unwrap_or(Value::Null);
        self.delta(choice.get("delta").unwrap_or(&Value::Null))?;
        if let Some(f) = choice.get("finish_reason").filter(|f| crate::pyrepr::truthy(f)) {
            self.finish(
                f.as_str(),
                p.get("usage"),
                p.get("tensorfold"),
            )?;
        }
        Ok(())
    }

    /// A whole chat completion (not streamed). Upstream: `completion`
    /// (responses.zig:476).
    fn completion(&mut self, data: &Value) -> Result<Value, SendError> {
        let choice = data
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
            .cloned()
            .unwrap_or(Value::Null);
        let message = choice.get("message").cloned().unwrap_or(Value::Null);
        let mut d = Map::new();
        d.insert(
            "reasoning_content".into(),
            message.get("reasoning_content").cloned().unwrap_or(Value::Null),
        );
        d.insert("content".into(), message.get("content").cloned().unwrap_or(Value::Null));
        let mut calls: Vec<Value> = Vec::new();
        if let Some(Value::Array(list)) = message.get("tool_calls") {
            for (i, call) in list.iter().enumerate() {
                let mut c = Map::new();
                c.insert("index".into(), json!(i));
                if let Some(obj) = call.as_object() {
                    for (k, v) in obj {
                        c.insert(k.clone(), v.clone());
                    }
                }
                calls.push(Value::Object(c));
            }
        }
        d.insert("tool_calls".into(), Value::Array(calls));
        self.delta(&Value::Object(d))?;
        let reason = choice.get("finish_reason");
        self.finish(
            reason.and_then(Value::as_str),
            data.get("usage"),
            data.get("tensorfold"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A stored response with a parent and the messages it added.
    fn stored(id: &str, parent: Option<&str>, text: &str) -> Value {
        json!({
            "id": id,
            "previous_response_id": parent,
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}]}],
        })
    }

    /// The store's round trip: put replaces by id, get copies, delete
    /// removes (responses.zig:64-103).
    #[test]
    fn the_store_round_trip() {
        let s = Store::default();
        assert!(s.get("r").is_none());
        s.put(&stored("r1", None, "one"), &[json!({"role": "user", "content": "u"})]);
        assert_eq!(s.get("r1").unwrap()["id"], "r1");
        // A second put of the same id replaces, no duplicate
        s.put(&stored("r1", None, "one again"), &[]);
        {
            let inner = s.inner.lock().unwrap();
            assert_eq!(inner.entries.len(), 1);
            assert_eq!(inner.entries[0].size, inner.bytes);
        }
        assert!(s.delete("r1"));
        assert!(!s.delete("r1"));
        assert!(s.get("r1").is_none());
        // The oldest go past 1,024 entries
        for i in 0..1025 {
            s.put(&stored(&format!("r{i}"), None, "x"), &[]);
        }
        {
            let inner = s.inner.lock().unwrap();
            assert_eq!(inner.entries.len(), 1024);
            assert!(!inner.entries.iter().any(|e| e.id == "r0"));
        }
    }

    /// A stored response's messages: the added ones plus its output items
    /// translated back (responses.zig:64's from_output).
    #[test]
    fn the_stored_messages() {
        let s = Store::default();
        s.put(
            &json!({
                "id": "r1", "previous_response_id": null,
                "output": [
                    {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "hmm"}]},
                    {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "hi"}]},
                    {"type": "function_call", "call_id": "c", "name": "n", "arguments": "{}"},
                ],
            }),
            &[json!({"role": "user", "content": "u"})],
        );
        let inner = s.inner.lock().unwrap();
        assert_eq!(
            serde_json::to_value(&inner.entries[0].messages).unwrap(),
            json!([
                {"role": "user", "content": "u"},
                {"role": "assistant", "reasoning_content": "hmm",
                 "content": [{"type": "text", "text": "hi"}],
                 "tool_calls": [{"id": "c", "type": "function",
                                 "function": {"name": "n", "arguments": "{}"}}]},
            ])
        );
    }

    /// A chain's messages, oldest first; each read entry moves newest
    /// (responses.zig:114's History.conversation); the missing parent the
    /// exact refusal.
    #[test]
    fn the_conversation_chain() {
        let s = Store::default();
        s.put(&stored("r1", None, "one"), &[json!({"role": "user", "content": "u1"})]);
        s.put(&stored("r2", Some("r1"), "two"), &[json!({"role": "user", "content": "u2"})]);
        let msgs = s.conversation(&json!("r2")).unwrap();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0]["content"], "u1");
        assert_eq!(msgs[2]["content"], "u2");
        // The read chain moved newest: r2 at the back after r1's read
        {
            let inner = s.inner.lock().unwrap();
            assert_eq!(inner.entries.last().unwrap().id, "r1");
        }
        // The missing parent
        let err = s.conversation(&json!("gone")).unwrap_err();
        assert_eq!(
            err.message,
            "previous response 'gone' is not stored here (store: false, deleted, or evicted): send the conversation's items instead"
        );
        // A non-string parent
        assert_eq!(
            s.conversation(&json!(5)).unwrap_err().message,
            "previous_response_id must be a string"
        );
    }

    /// A machine over a collected event stream: the events in their order,
    /// their sequence numbers counted.
    fn machine(base: Map<String, Value>) -> (Reply, tokio::sync::mpsc::Receiver<TypedEvent>) {
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        (
            Reply {
                sender: Some(tx),
                store: Store::default(),
                keep: false,
                added: Vec::new(),
                base,
                items: Vec::new(),
                current: None,
                calls: Vec::new(),
                seq: 0,
                final_: None,
            },
            rx,
        )
    }

    fn drain(mut rx: tokio::sync::mpsc::Receiver<TypedEvent>) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        while let Ok(te) = rx.try_recv() {
            out.push((te.kind.to_string(), te.body));
        }
        out
    }

    /// The machine over a whole completion: the reasoning and the text as
    /// items, the usage mapped, the completed event last
    /// (responses.zig:476's completion).
    #[test]
    fn the_whole_completion() {
        let (mut m, mut rx) = machine(Map::new());
        m.sender = None;
        let whole = json!({
            "choices": [{
                "message": {"content": "The answer is fast.", "reasoning_content": "Let me think"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 26, "completion_tokens": 11,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
            "tensorfold": {"engine": "lanes"},
        });
        let final_response = m.completion(&whole).unwrap();
        assert_eq!(final_response["status"], "completed");
        let output = final_response["output"].as_array().unwrap();
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["content"][0]["text"], "Let me think");
        assert_eq!(output[1]["type"], "message");
        assert_eq!(output[1]["content"][0]["text"], "The answer is fast.");
        assert_eq!(
            final_response["usage"],
            json!({
                "input_tokens": 26,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens": 11,
                "output_tokens_details": {"reasoning_tokens": 4},
                "total_tokens": 37,
            })
        );
        assert_eq!(final_response["tensorfold"], json!({"engine": "lanes"}));
        // The whole path emits nothing (the route builds its machine with
        // no event sender)
        drop(m);
        assert!(rx.try_recv().is_err());
    }

    /// A length finish is incomplete with its reason; an all-reasoning
    /// reply still opens a message (responses.zig:413's finish).
    #[test]
    fn the_incomplete_and_the_empty_message() {
        let (mut m, _rx) = machine(Map::new());
        let whole = json!({
            "choices": [{"message": {"reasoning_content": "hmm"}, "finish_reason": "length"}],
        });
        let final_response = m.completion(&whole).unwrap();
        assert_eq!(final_response["status"], "incomplete");
        assert_eq!(
            final_response["incomplete_details"],
            json!({"reason": "max_output_tokens"})
        );
        assert!(final_response.get("completed_at").is_none());
        // Only reasoning arrived, and a length finish opens no empty
        // message (the all-reasoning rule runs on completions alone)
        let output = final_response["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "reasoning");

        // No reasoning at all: the message still opens (the corpus's
        // stored-chain shape)
        let (mut m, _rx) = machine(Map::new());
        let final_response = m
            .completion(&json!({"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}))
            .unwrap();
        let output = final_response["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "message");
    }

    /// The streamed machine: created, in_progress, the items' events in
    /// order with their sequence numbers, completed last
    /// (responses.zig:463's chunk over the Wire's events).
    #[test]
    fn the_streamed_events() {
        let (mut m, rx) = machine(Map::new());
        m.start().unwrap();
        // The role chunk: a delta the machine ignores
        m.chunk(Some(&json!({"choices": [{"delta": {"role": "assistant"}}]})))
            .unwrap();
        // The reasoning, then the text
        m.chunk(Some(&json!({"choices": [{"delta": {"reasoning_content": "Let"}}]})))
            .unwrap();
        m.chunk(Some(&json!({"choices": [{"delta": {"reasoning_content": " me"}}]})))
            .unwrap();
        m.chunk(Some(&json!({"choices": [{"delta": {"content": "Fast."}}]})))
            .unwrap();
        // The last chunk: the finish and the usage
        m.chunk(Some(&json!({
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            "tensorfold": {"engine": "lanes"},
        })))
        .unwrap();
        assert!(m.final_.is_some());
        // The [DONE] after the finish: a no-op
        m.chunk(None).unwrap();
        drop(m);
        let events = drain(rx);
        let kinds: Vec<&str> = events.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            kinds,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.delta",
                // The message's open closes the reasoning item first
                // (responses.zig:283's close in open)
                "response.reasoning_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        // The sequence numbers count
        for (i, (_, body)) in events.iter().enumerate() {
            assert_eq!(body["sequence_number"], json!(i as u64));
        }
        // The reasoning closed when the message opened; the final carries
        // both items and the usage
        let completed = events.last().unwrap();
        assert_eq!(completed.1["response"]["output"].as_array().unwrap().len(), 2);
        assert_eq!(completed.1["response"]["usage"]["output_tokens"], json!(3));
        assert_eq!(
            completed.1["response"]["tensorfold"],
            json!({"engine": "lanes"})
        );
    }

    /// A call through the streamed deltas: the function_call item, its
    /// arguments' deltas, the done at its close (responses.zig:371's
    /// delta).
    #[test]
    fn the_streamed_call() {
        let (mut m, rx) = machine(Map::new());
        m.start().unwrap();
        m.chunk(Some(&json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "lookup", "arguments": "{\"q\""}},
        ]}}]})))
            .unwrap();
        m.chunk(Some(&json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": ": \"x\"}"}},
        ]}}]})))
            .unwrap();
        m.chunk(Some(&json!({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})))
            .unwrap();
        drop(m);
        let events = drain(rx);
        let kinds: Vec<&str> = events.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            kinds,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        let added = &events[2];
        assert_eq!(added.1["item"]["type"], "function_call");
        assert_eq!(added.1["item"]["call_id"], "call_1");
        assert_eq!(added.1["item"]["name"], "lookup");
        let done = &events[5];
        assert_eq!(done.1["arguments"], json!("{\"q\": \"x\"}"));
        let completed = events.last().unwrap();
        let call = &completed.1["response"]["output"][0];
        assert_eq!(call["arguments"], json!("{\"q\": \"x\"}"));
        // A call with no id gets one made (responses.zig:381)
        let (mut m, _rx) = machine(Map::new());
        m.start().unwrap();
        m.chunk(Some(&json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "n"}},
        ]}}]})))
            .unwrap();
        assert!(m.calls[0].1 == 0);
        assert_eq!(m.items[0]["call_id"].as_str().unwrap().len(), 37); // call_ + 32 hex
    }

    /// An error chunk fails the reply: the code from the type, the stream
    /// ends, nothing stored (responses.zig:438's fail).
    #[test]
    fn the_failed_reply() {
        let (mut m, rx) = machine(Map::new());
        m.start().unwrap();
        m.chunk(Some(&json!({"error": {"message": "the context is full",
                                       "type": "invalid_request_error"}})))
            .unwrap();
        assert!(m.final_.is_some());
        drop(m);
        let events = drain(rx);
        let kinds: Vec<&str> = events.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            kinds,
            vec!["response.created", "response.in_progress", "response.failed"]
        );
        let failed = events.last().unwrap();
        assert_eq!(failed.1["response"]["status"], "failed");
        assert_eq!(
            failed.1["response"]["error"],
            json!({"code": "invalid_prompt", "message": "the context is full"})
        );
        // The stream ends early with no finish: response.failed with the
        // ended-early message (responses.zig:466)
        let (mut m, rx) = machine(Map::new());
        m.start().unwrap();
        m.chunk(None).unwrap();
        drop(m);
        let events = drain(rx);
        let failed = events.last().unwrap();
        assert_eq!(
            failed.1["response"]["error"],
            json!({"code": "server_error", "message": "the reply ended early"})
        );
        // A server-side error names itself
        let (mut m, rx) = machine(Map::new());
        m.start().unwrap();
        m.chunk(Some(&json!({"error": {"message": "the engine died", "type": "server_error"}})))
            .unwrap();
        drop(m);
        let events = drain(rx);
        let failed = events.last().unwrap();
        assert_eq!(
            failed.1["response"]["error"],
            json!({"code": "server_error", "message": "the engine died"})
        );
    }

    /// The finished response stores when asked, its added messages with it
    /// (responses.zig:430's keep).
    #[test]
    fn the_finished_stores() {
        let store = Store::default();
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        let mut base = Map::new();
        base.insert("id".into(), json!("resp_stored"));
        let mut m = Reply {
            sender: Some(tx),
            store: store.clone(),
            keep: true,
            added: vec![json!({"role": "user", "content": "u"})],
            base,
            items: Vec::new(),
            current: None,
            calls: Vec::new(),
            seq: 0,
            final_: None,
        };
        let final_response = m
            .completion(&json!({"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]}))
            .unwrap();
        let id = final_response["id"].as_str().unwrap().to_string();
        drop(m);
        drop(rx);
        let stored = store.get(&id).expect("the reply stored");
        assert_eq!(stored["status"], "completed");
        let inner = store.inner.lock().unwrap();
        assert_eq!(inner.entries[0].messages.len(), 2); // the added user + the output's assistant
    }
}
