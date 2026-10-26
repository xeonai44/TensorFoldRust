//! A Responses request as the chat completion that runs it.
//!
//! Port of `zig/src/server/responses_translate.zig` (344 lines): the input
//! items as chat messages (a turn's reasoning, text and calls one assistant
//! message), the tools and the tool choice, `text.format` as a
//! `response_format`, and the echo object the Response carries. The store's
//! history rides through `responses::Store::conversation`
//! (`responses_translate.zig:119`'s History).

use crate::errors::Refused;
use crate::pyrepr;
use serde_json::{json, Map, Value};

/// Fields a chat completion reads as they are: OpenAI's and this server's
/// own. Upstream: `passed` (responses_translate.zig:12).
const PASSED: [&str; 15] = [
    "model",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "stream",
    "parallel_tool_calls",
    "stop",
    "draft",
    "thinking_budget",
    "ignore_eos",
    "priority",
    "return_token_ids",
    "chat_template_kwargs",
];

/// Fields a Responses request must not send. Upstream: `refused`
/// (responses_translate.zig:14).
const REFUSED: [(&str, &str); 4] = [
    (
        "background",
        "background responses are not supported: send the request and wait for it",
    ),
    (
        "conversation",
        "conversations are not supported: send previous_response_id or the items",
    ),
    (
        "prompt",
        "prompt templates are not supported: send input and instructions",
    ),
    ("context_management", "context management is not supported"),
];

/// A Responses request as the chat completion that runs it. Upstream:
/// `Request` (responses_translate.zig:22).
pub struct Request {
    pub chat: Value,
    pub echo: Map<String, Value>,
    pub added: Vec<Value>,
    pub store: bool,
    pub stream: bool,
}

/// An item's `type` field. Upstream: `kindOf` (responses_translate.zig:28).
fn kind_of(v: &Value) -> Option<&str> {
    v.get("type").and_then(Value::as_str)
}

/// A field present and non-null. Upstream: `json.field`.
fn field<'a>(body: &'a Value, name: &str) -> Option<&'a Value> {
    body.get(name).filter(|v| !v.is_null())
}

/// A field present and truthy. Upstream: `json.truthyField`.
fn truthy_field(body: &Value, name: &str) -> bool {
    body.get(name).is_some_and(pyrepr::truthy)
}

/// A field that is a string. Upstream: `json.strField`.
fn str_field<'a>(v: &'a Value, name: &str) -> Option<&'a str> {
    v.get(name).and_then(Value::as_str)
}

/// A message's content as chat content: text parts as text, images as
/// image_url parts. Upstream: `content` (responses_translate.zig:38).
fn content(c: Option<&Value>) -> Result<Value, Refused> {
    let Some(value) = c else {
        return Err(Refused::request(
            "message content must be a string or a list of content parts",
        ));
    };
    if value.is_string() {
        return Ok(value.clone());
    }
    let Some(Value::Array(list)) = Some(value) else {
        return Err(Refused::request(
            "message content must be a string or a list of content parts",
        ));
    };
    let mut parts: Vec<Value> = Vec::new();
    for part in list {
        let kind = part.get("type");
        let k = kind.and_then(Value::as_str).unwrap_or("");
        if matches!(k, "input_text" | "output_text" | "text") && str_field(part, "text").is_some() {
            parts.push(crate::messages::text_part(str_field(part, "text").unwrap()));
        } else if k == "refusal" && str_field(part, "refusal").is_some() {
            parts.push(crate::messages::text_part(str_field(part, "refusal").unwrap()));
        } else if k == "input_image" {
            let Some(url_text) = str_field(part, "image_url") else {
                return Err(Refused::request(
                    "input_image needs an image_url (a URL or data URL); file ids are not supported",
                ));
            };
            let mut url = Map::new();
            url.insert("url".into(), json!(url_text));
            if truthy_field(part, "detail") {
                url.insert("detail".into(), part.get("detail").cloned().unwrap());
            }
            let mut p = Map::new();
            p.insert("type".into(), json!("image_url"));
            p.insert("image_url".into(), Value::Object(url));
            parts.push(Value::Object(p));
        } else {
            return Err(Refused::request(format!(
                "content parts of type {} are not supported: send input_text or input_image",
                pyrepr::repr(kind.unwrap_or(&Value::Null)),
            )));
        }
    }
    Ok(Value::Array(parts))
}

/// A function_call_output's output as a tool message's content. Upstream:
/// `output` (responses_translate.zig:74).
fn output(o: Option<&Value>) -> Result<Value, Refused> {
    let Some(value) = o else {
        return Err(Refused::request(
            "a function_call_output's output must be a string or input_text and input_image parts",
        ));
    };
    if value.is_string() {
        return Ok(value.clone());
    }
    if let Value::Array(list) = value {
        let mut texts_only = true;
        let mut parts_only = true;
        for p in list {
            let name = kind_of(p).unwrap_or("");
            if name != "input_text" {
                texts_only = false;
            }
            if name != "input_text" && name != "input_image" {
                parts_only = false;
            }
        }
        if texts_only {
            let mut joined = String::new();
            for p in list {
                if let Some(t) = p.get("text") {
                    if pyrepr::truthy(t) {
                        joined.push_str(&crate::tool_specs::py_str(t));
                    }
                }
            }
            return Ok(Value::String(joined));
        }
        if parts_only {
            return content(Some(value));
        }
    }
    Err(Refused::request(
        "a function_call_output's output must be a string or input_text and input_image parts",
    ))
}

/// Input items (output items too) as chat messages: a turn's reasoning, text
/// and calls are one assistant message. Upstream: `messages`
/// (responses_translate.zig:93).
pub fn messages(items: &[Value]) -> Result<Vec<Value>, Refused> {
    let mut out: Vec<Map<String, Value>> = Vec::new();
    let mut thought: Option<String> = None;
    for item in items {
        let Some(obj) = item.as_object() else {
            return Err(Refused::request("each input item must be an object"));
        };
        let mut kind: Option<Value> = obj.get("type").cloned();
        if kind.as_ref().is_none_or(|k| !pyrepr::truthy(k)) {
            kind = if obj.contains_key("role") {
                Some(json!("message"))
            } else {
                None
            };
        }
        let k = kind.as_ref().and_then(Value::as_str).unwrap_or("");
        if k == "message" {
            let role = str_field(item, "role").unwrap_or("");
            let ok = matches!(role, "user" | "assistant" | "system" | "developer");
            if !ok || obj.get("role").map(|r| !r.is_string()).unwrap_or(true) {
                return Err(Refused::request(
                    "a message's role must be user, assistant, system or developer",
                ));
            }
            let mut m = Map::new();
            m.insert("role".into(), json!(role));
            m.insert("content".into(), content(obj.get("content"))?);
            if role == "assistant" {
                if let Some(t) = thought.take() {
                    m.insert("reasoning_content".into(), json!(t));
                }
            }
            out.push(m);
        } else if k == "function_call" {
            let (Some(call_id), Some(name)) = (str_field(item, "call_id"), str_field(item, "name"))
            else {
                return Err(Refused::request(
                    "a function_call item needs a call_id and a name",
                ));
            };
            let mut function = Map::new();
            function.insert("name".into(), json!(name));
            let args = item.get("arguments");
            function.insert(
                "arguments".into(),
                match args {
                    Some(a) if pyrepr::truthy(a) => a.clone(),
                    _ => json!("{}"),
                },
            );
            let mut call = Map::new();
            call.insert("id".into(), json!(call_id));
            call.insert("type".into(), json!("function"));
            call.insert("function".into(), Value::Object(function));
            let last_role = out
                .last()
                .and_then(|m| m.get("role"))
                .and_then(Value::as_str)
                .unwrap_or("");
            if out.is_empty() || last_role != "assistant" {
                let mut m = Map::new();
                m.insert("role".into(), json!("assistant"));
                m.insert("content".into(), json!(""));
                out.push(m);
            }
            let last = out.last_mut().unwrap();
            if let Some(t) = thought.take() {
                last.insert("reasoning_content".into(), json!(t));
            }
            let mut calls: Vec<Value> = match last.get("tool_calls") {
                Some(Value::Array(existing)) => existing.clone(),
                _ => Vec::new(),
            };
            calls.push(Value::Object(call));
            last.insert("tool_calls".into(), Value::Array(calls));
        } else if k == "function_call_output" {
            let Some(call_id) = str_field(item, "call_id") else {
                return Err(Refused::request(
                    "a function_call_output item needs the call_id of its call",
                ));
            };
            let mut m = Map::new();
            m.insert("role".into(), json!("tool"));
            m.insert("tool_call_id".into(), json!(call_id));
            m.insert("content".into(), output(obj.get("output"))?);
            out.push(m);
        } else if k == "reasoning" {
            let mut texts = String::new();
            let mut any = false;
            if let Some(Value::Array(list)) = obj.get("content") {
                for p in list {
                    if p.is_object() {
                        any = true;
                        if let Some(t) = str_field(p, "text") {
                            texts.push_str(t);
                        }
                    }
                }
            }
            if truthy_field(item, "encrypted_content") && !any {
                return Err(Refused::request(
                    "encrypted reasoning is not supported: send the reasoning item's content",
                ));
            }
            thought = if texts.is_empty() { None } else { Some(texts) };
        } else {
            return Err(Refused::request(format!(
                "input items of type {} are not supported: send messages, function_call, function_call_output and reasoning items",
                pyrepr::repr(kind.as_ref().unwrap_or(&Value::Null)),
            )));
        }
    }
    Ok(out.into_iter().map(Value::Object).collect())
}

/// The offered tools as chat tools. Upstream: `tools`
/// (responses_translate.zig:156).
fn tools(raw: Option<&Value>) -> Result<Option<Vec<Value>>, Refused> {
    let Some(value) = raw else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let Some(Value::Array(list)) = Some(value) else {
        return Err(Refused::request("tools must be a list"));
    };
    let mut out: Vec<Value> = Vec::new();
    for tool in list {
        let kind = tool.get("type");
        if kind.and_then(Value::as_str) != Some("function") {
            return Err(Refused::request(format!(
                "tools of type {} are not supported: this server runs function tools only",
                pyrepr::repr(kind.unwrap_or(&Value::Null)),
            )));
        }
        let name = str_field(tool, "name").unwrap_or("");
        if name.is_empty() {
            return Err(Refused::request("a function tool needs a name"));
        }
        let mut function = Map::new();
        for k in ["name", "description", "parameters", "strict"] {
            if let Some(v) = field(tool, k) {
                function.insert(k.into(), v.clone());
            }
        }
        let mut t = Map::new();
        t.insert("type".into(), json!("function"));
        t.insert("function".into(), Value::Object(function));
        out.push(Value::Object(t));
    }
    Ok(Some(out))
}

/// The tool_choice the chat body carries, with the tools it keeps.
/// Upstream: `toolChoice` (responses_translate.zig:193).
struct Choice {
    choice: Option<Value>,
    offered: Option<Vec<Value>>,
}

fn tool_choice(raw: Option<&Value>, offered: Option<Vec<Value>>) -> Result<Choice, Refused> {
    let Some(value) = raw else {
        return Ok(Choice {
            choice: None,
            offered,
        });
    };
    if value.is_null() {
        return Ok(Choice {
            choice: None,
            offered,
        });
    }
    if let Some(s) = value.as_str() {
        if matches!(s, "none" | "auto" | "required") {
            return Ok(Choice {
                choice: Some(value.clone()),
                offered,
            });
        }
    }
    let k = kind_of(value).unwrap_or("");
    if k == "function" && str_field(value, "name").is_some() {
        let mut function = Map::new();
        function.insert("name".into(), json!(str_field(value, "name").unwrap()));
        let mut c = Map::new();
        c.insert("type".into(), json!("function"));
        c.insert("function".into(), Value::Object(function));
        return Ok(Choice {
            choice: Some(Value::Object(c)),
            offered,
        });
    }
    let mode = value.get("mode").cloned().unwrap_or(json!("auto"));
    let mode_ok = mode
        .as_str()
        .is_some_and(|m| m == "auto" || m == "required");
    if k == "allowed_tools" && mode_ok {
        let allowed = value.get("tools");
        let list: &[Value] = match allowed {
            Some(Value::Array(list)) => list,
            _ => &[],
        };
        if allowed.is_some_and(|a| pyrepr::truthy(a) && !a.is_array()) {
            return Err(Refused::request(
                "allowed_tools may name function tools only",
            ));
        }
        for t in list {
            if kind_of(t) != Some("function") {
                return Err(Refused::request(
                    "allowed_tools may name function tools only",
                ));
            }
        }
        let mut kept: Vec<Value> = Vec::new();
        for t in offered.as_deref().unwrap_or(&[]) {
            let name = t
                .get("function")
                .and_then(|f| f.get("name"))
                .cloned()
                .unwrap_or(Value::Null);
            for allow in list {
                if allow.get("name").is_some_and(|n| *n == name) {
                    kept.push(t.clone());
                    break;
                }
            }
        }
        return Ok(Choice {
            choice: Some(mode),
            offered: Some(kept),
        });
    }
    Err(Refused::request(
        "tool_choice must be none, auto, required, a function or allowed_tools of functions",
    ))
}

/// ``text.format`` as a chat ``response_format`` (None for plain text).
/// Upstream: `format` (responses_translate.zig:236).
fn format(text: Option<&Value>) -> Result<Option<Value>, Refused> {
    let Some(t) = text else {
        return Ok(None);
    };
    if t.is_null() {
        return Ok(None);
    }
    let Some(obj) = t.as_object() else {
        return Err(Refused::request("text must be an object"));
    };
    let Some(fmt) = obj.get("format") else {
        return Ok(None);
    };
    if fmt.is_null() {
        return Ok(None);
    }
    let Some(fmt_obj) = fmt.as_object() else {
        return Err(Refused::other("'str' object has no attribute 'get'"));
    };
    let k = fmt_obj.get("type").and_then(Value::as_str).unwrap_or("");
    if k == "text" {
        return Ok(None);
    }
    let mut out = Map::new();
    if k == "json_object" {
        out.insert("type".into(), json!("json_object"));
        return Ok(Some(Value::Object(out)));
    }
    if k == "json_schema" {
        let mut js = Map::new();
        for key in ["name", "schema", "strict", "description"] {
            if let Some(v) = field(fmt, key) {
                js.insert(key.into(), v.clone());
            }
        }
        out.insert("type".into(), json!("json_schema"));
        out.insert("json_schema".into(), Value::Object(js));
        return Ok(Some(Value::Object(out)));
    }
    Err(Refused::request(
        "text.format must be text, json_object or json_schema",
    ))
}

/// A Responses request as the chat completion that runs it; ``store`` reads
/// a stored conversation. Upstream: `translate`
/// (responses_translate.zig:252).
pub fn translate(
    body: &Value,
    store: &crate::responses::Store,
) -> Result<Request, Refused> {
    if !body.is_object() {
        return Err(Refused::request("the request body must be a JSON object"));
    }
    // logprobs come on /v1/chat/completions only (responses_translate.zig:261)
    crate::fields::probability_options(body, false).map_err(Refused::request)?;
    for (name, message) in REFUSED {
        if truthy_field(body, name) {
            return Err(Refused::request(message));
        }
    }
    if let Some(inc) = body.get("include") {
        if pyrepr::truthy(inc) {
            let mut parts: Vec<String> = Vec::new();
            match inc {
                Value::Array(items) => {
                    for i in items {
                        parts.push(crate::tool_specs::py_str(i));
                    }
                }
                Value::Object(o) => {
                    for k in o.keys() {
                        parts.push(k.clone());
                    }
                }
                Value::String(s) => {
                    for c in s.chars() {
                        parts.push(c.to_string());
                    }
                }
                _ => parts.push(crate::tool_specs::py_str(inc)),
            }
            return Err(Refused::request(format!(
                "include is not supported ({}): reasoning comes as text in its reasoning item, and nothing is encrypted",
                parts.join(", "),
            )));
        }
    }
    if let Some(t) = field(body, "truncation") {
        if t.as_str() != Some("disabled") {
            return Err(Refused::request(
                "truncation must be \"disabled\": a prompt too long for the context is refused",
            ));
        }
    }
    if truthy_field(body, "top_logprobs") {
        return Err(Refused::request("top_logprobs is not supported"));
    }
    let meta_raw = body.get("metadata");
    let metadata = match meta_raw {
        Some(m) if pyrepr::truthy(m) => m.clone(),
        _ => Value::Object(Map::new()),
    };
    let meta_ok = metadata.as_object().is_some_and(|o| {
        o.len() <= 16
            && o.iter().all(|(k, v)| {
                crate::reply_text::char_count(k) <= 64
                    && v.is_string()
                    && crate::reply_text::char_count(v.as_str().unwrap()) <= 512
            })
    });
    if !meta_ok {
        return Err(Refused::request(
            "metadata must be at most 16 string pairs (keys up to 64 characters, values up to 512)",
        ));
    }
    let instructions = field(body, "instructions");
    if instructions.is_some_and(|i| !i.is_string()) {
        return Err(Refused::request("instructions must be a string"));
    }
    let mut given = body.get("input").cloned().unwrap_or(Value::Null);
    if let Value::String(s) = &given {
        let m = json!({"role": "user", "content": s});
        given = Value::Array(vec![m]);
    }
    let Value::Array(input) = &given else {
        return Err(Refused::request(
            "input must be a string or a non-empty list of items",
        ));
    };
    if input.is_empty() {
        return Err(Refused::request(
            "input must be a string or a non-empty list of items",
        ));
    }
    let added = messages(input)?;
    let parent = field(body, "previous_response_id");
    let earlier = match parent {
        Some(p) => store.conversation(p)?,
        None => Vec::new(),
    };
    let offered = tools(body.get("tools"))?;
    let choice = tool_choice(body.get("tool_choice"), offered)?;
    let mut chat = Map::new();
    for k in PASSED {
        if let Some(v) = body.get(k) {
            chat.insert(k.into(), v.clone());
        }
    }
    let mut msgs: Vec<Value> = Vec::new();
    if let Some(Value::String(s)) = instructions {
        if !s.is_empty() {
            msgs.push(json!({"role": "system", "content": s}));
        }
    }
    msgs.extend(earlier);
    msgs.extend(added.iter().cloned());
    chat.insert("messages".into(), Value::Array(msgs));
    if let Some(o) = &choice.offered {
        chat.insert("tools".into(), Value::Array(o.clone()));
    }
    if let Some(c) = &choice.choice {
        chat.insert("tool_choice".into(), c.clone());
    }
    if let Some(m) = field(body, "max_output_tokens") {
        chat.insert("max_tokens".into(), m.clone());
    }
    let reasoning_raw = body.get("reasoning");
    let reasoning = match reasoning_raw {
        Some(r) if pyrepr::truthy(r) => r.clone(),
        _ => Value::Object(Map::new()),
    };
    if !reasoning.is_object() {
        return Err(Refused::request("reasoning must be an object"));
    }
    if let Some(e) = field(&reasoning, "effort") {
        chat.insert("reasoning_effort".into(), e.clone());
    }
    if let Some(f) = format(body.get("text"))? {
        chat.insert("response_format".into(), f);
    }
    let mut echo = Map::new();
    echo.insert("instructions".into(), instructions.cloned().unwrap_or(Value::Null));
    echo.insert(
        "max_output_tokens".into(),
        body.get("max_output_tokens").cloned().unwrap_or(Value::Null),
    );
    echo.insert("metadata".into(), metadata);
    echo.insert(
        "parallel_tool_calls".into(),
        body.get("parallel_tool_calls").cloned().unwrap_or(json!(true)),
    );
    echo.insert("previous_response_id".into(), parent.cloned().unwrap_or(Value::Null));
    let mut r = Map::new();
    r.insert("effort".into(), reasoning.get("effort").cloned().unwrap_or(Value::Null));
    r.insert(
        "summary".into(),
        reasoning.get("summary").cloned().unwrap_or(Value::Null),
    );
    echo.insert("reasoning".into(), Value::Object(r));
    let store_field = body.get("store");
    let store = !(matches!(store_field, Some(Value::Bool(false))));
    echo.insert("store".into(), json!(store));
    echo.insert("temperature".into(), body.get("temperature").cloned().unwrap_or(Value::Null));
    let mut text_obj = Map::new();
    let text_raw = body.get("text");
    let fmt_raw = match text_raw {
        Some(t) if pyrepr::truthy(t) => t.get("format").cloned(),
        _ => None,
    };
    match fmt_raw {
        Some(f) if pyrepr::truthy(&f) => {
            text_obj.insert("format".into(), f);
        }
        _ => {
            text_obj.insert("format".into(), json!({"type": "text"}));
        }
    }
    echo.insert("text".into(), Value::Object(text_obj));
    let tc = body.get("tool_choice");
    echo.insert(
        "tool_choice".into(),
        match tc {
            Some(t) if pyrepr::truthy(t) => t.clone(),
            _ => json!("auto"),
        },
    );
    let ts = body.get("tools");
    echo.insert(
        "tools".into(),
        match ts {
            Some(t) if pyrepr::truthy(t) => t.clone(),
            _ => json!([]),
        },
    );
    echo.insert("top_p".into(), body.get("top_p").cloned().unwrap_or(Value::Null));
    echo.insert("truncation".into(), json!("disabled"));
    echo.insert("user".into(), body.get("user").cloned().unwrap_or(Value::Null));
    echo.insert("background".into(), json!(false));
    Ok(Request {
        chat: Value::Object(chat),
        echo,
        added,
        store,
        stream: truthy_field(body, "stream"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn refused(body: &Value) -> String {
        match translate(body, &crate::responses::Store::default()) {
            Err(r) => r.message,
            Ok(_) => panic!("the body translated"),
        }
    }

    /// The refusal wordings, char for char the frozen corpus's
    /// (zig/tests/server/golden/responses/*).
    #[test]
    fn the_refusal_wordings() {
        assert_eq!(
            refused(&json!({"input": "x", "background": true})),
            "background responses are not supported: send the request and wait for it"
        );
        assert_eq!(
            refused(&json!({"input": "x", "conversation": "c"})),
            "conversations are not supported: send previous_response_id or the items"
        );
        assert_eq!(
            refused(&json!({"input": "x", "prompt": "a template"})),
            "prompt templates are not supported: send input and instructions"
        );
        assert_eq!(
            refused(&json!({"input": "x", "context_management": {"keep": "last"}})),
            "context management is not supported"
        );
        assert_eq!(
            refused(&json!({"input": "x", "include": ["reasoning.encrypted_content"]})),
            "include is not supported (reasoning.encrypted_content): reasoning comes as text in its reasoning item, and nothing is encrypted"
        );
        assert_eq!(
            refused(&json!({"input": "x", "truncation": "auto"})),
            "truncation must be \"disabled\": a prompt too long for the context is refused"
        );
        // The top_logprobs cases live inside probability_options, which
        // runs first: without logprobs it demands them, with them the
        // unsupported refusal (translate's own top_logprobs check never
        // comes after a passing one — the same dead defense upstream has)
        assert_eq!(
            refused(&json!({"input": "x", "top_logprobs": 5})),
            "top_logprobs requires logprobs: true"
        );
        assert_eq!(
            refused(&json!({"input": "x", "logprobs": true})),
            "logprobs are not supported by this model or backend"
        );
        assert_eq!(
            refused(&json!({"input": "x", "instructions": 5})),
            "instructions must be a string"
        );
        assert_eq!(
            refused(&json!({"input": []})),
            "input must be a string or a non-empty list of items"
        );
        assert_eq!(
            refused(&json!({})),
            "input must be a string or a non-empty list of items"
        );
        assert_eq!(
            refused(&json!({"input": 5})),
            "input must be a string or a non-empty list of items"
        );
        assert_eq!(
            refused(&json!({"input": [{"type": "bogus"}]})),
            "input items of type 'bogus' are not supported: send messages, function_call, function_call_output and reasoning items"
        );
        assert_eq!(
            refused(&json!({"input": "x", "tools": [{"type": "web_search"}]})),
            "tools of type 'web_search' are not supported: this server runs function tools only"
        );
        assert_eq!(
            refused(&json!({"input": [{"role": "user", "content": [{"type": "input_file"}]}]})),
            "content parts of type 'input_file' are not supported: send input_text or input_image"
        );
        assert_eq!(
            refused(&json!({"input": [{"role": "robot", "content": "x"}]})),
            "a message's role must be user, assistant, system or developer"
        );
        assert_eq!(
            refused(&json!({"input": "x", "text": {"format": {"type": "bogus"}}})),
            "text.format must be text, json_object or json_schema"
        );
        assert_eq!(
            refused(&json!([1])),
            "the request body must be a JSON object"
        );
        // The metadata's bounds: at most 16 pairs
        let mut metadata = serde_json::Map::new();
        for i in 0..17 {
            metadata.insert(format!("{i}"), json!("v"));
        }
        assert_eq!(
            refused(&json!({"input": "x", "metadata": metadata})),
            "metadata must be at most 16 string pairs (keys up to 64 characters, values up to 512)"
        );
        // A non-string value
        assert_eq!(
            refused(&json!({"input": "x", "metadata": {"k": 5}})),
            "metadata must be at most 16 string pairs (keys up to 64 characters, values up to 512)"
        );
        // The missing parent: the id in Python's quotes
        assert_eq!(
            refused(&json!({"input": "x", "previous_response_id": "resp_gone"})),
            "previous response 'resp_gone' is not stored here (store: false, deleted, or evicted): send the conversation's items instead"
        );
        assert_eq!(
            refused(&json!({"input": "x", "previous_response_id": 5})),
            "previous_response_id must be a string"
        );
        // An image part without a URL
        assert_eq!(
            refused(&json!({"input": [{"role": "user", "content": [{"type": "input_image"}]}]})),
            "input_image needs an image_url (a URL or data URL); file ids are not supported"
        );
        // Encrypted reasoning
        assert_eq!(
            refused(&json!({"input": [{"type": "reasoning", "encrypted_content": "x"}]})),
            "encrypted reasoning is not supported: send the reasoning item's content"
        );
        // The tool choice
        assert_eq!(
            refused(&json!({"input": "x", "tool_choice": "bogus"})),
            "tool_choice must be none, auto, required, a function or allowed_tools of functions"
        );
        assert_eq!(
            refused(&json!({"input": "x", "tool_choice": {"type": "allowed_tools", "tools": [{"type": "bogus"}]}})),
            "allowed_tools may name function tools only"
        );
    }

    /// A string input becomes one user message; the echo carries every
    /// field with its default (the corpus's input-string body).
    #[test]
    fn a_string_input_and_the_echo() {
        let r = translate(
            &json!({"model": "fake-model", "input": "Hi"}),
            &crate::responses::Store::default(),
        )
        .unwrap();
        assert_eq!(
            r.chat["messages"],
            json!([{"role": "user", "content": "Hi"}])
        );
        assert_eq!(r.chat["model"], "fake-model");
        assert_eq!(
            Value::Object(r.echo.clone()),
            json!({
                "instructions": null,
                "max_output_tokens": null,
                "metadata": {},
                "parallel_tool_calls": true,
                "previous_response_id": null,
                "reasoning": {"effort": null, "summary": null},
                "store": true,
                "temperature": null,
                "text": {"format": {"type": "text"}},
                "tool_choice": "auto",
                "tools": [],
                "top_p": null,
                "truncation": "disabled",
                "user": null,
                "background": false,
            })
        );
        assert!(r.store);
        assert!(!r.stream);
    }

    /// The instructions become the system message first; the metadata and
    /// the rest echo back (the corpus's instructions case).
    #[test]
    fn instructions_and_metadata() {
        let r = translate(
            &json!({
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "x"}]}],
                "instructions": "Be brief",
                "metadata": {"k": "v"},
                "temperature": 0.5,
                "store": false,
                "stream": true,
            }),
            &crate::responses::Store::default(),
        )
        .unwrap();
        assert_eq!(
            r.chat["messages"],
            json!([
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": [{"type": "text", "text": "x"}]},
            ])
        );
        assert!(!r.store);
        assert!(r.stream);
        assert_eq!(r.echo["metadata"], json!({"k": "v"}));
        assert_eq!(r.echo["store"], json!(false));
        // A truthy non-bool still stores (responses_translate.zig:334)
        let r = translate(
            &json!({"input": "x", "store": "yes"}),
            &crate::responses::Store::default(),
        )
        .unwrap();
        assert!(r.store);
    }

    /// The items: a turn's reasoning, text and calls one assistant message,
    /// the output of a call a tool message (the corpus's items case).
    #[test]
    fn the_items_translation() {
        let r = translate(
            &json!({"input": [
                {"type": "message", "role": "user", "content": "find"},
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "hmm"}]},
                {"type": "message", "role": "assistant", "content": "looking"},
                {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{\"q\": \"x\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "found"},
            ]}),
            &crate::responses::Store::default(),
        )
        .map(|r| r.chat["messages"].clone())
        .unwrap_or(Value::Null);
        assert_eq!(
            r,
            json!([
                {"role": "user", "content": "find"},
                {"role": "assistant", "reasoning_content": "hmm", "content": "looking",
                 "tool_calls": [{"id": "call_1", "type": "function",
                                 "function": {"name": "lookup", "arguments": "{\"q\": \"x\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "found"},
            ])
        );
        // A function_call with no assistant before it opens one, its
        // arguments default {}, the thought rides over
        let r = translate(
            &json!({"input": [
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "t"}]},
                {"type": "function_call", "call_id": "c", "name": "n"},
            ]}),
            &crate::responses::Store::default(),
        )
        .map(|r| r.chat["messages"].clone())
        .unwrap_or(Value::Null);
        assert_eq!(
            r[0]["tool_calls"][0]["function"]["arguments"],
            json!("{}")
        );
        assert_eq!(r[0]["reasoning_content"], "t");
        assert_eq!(r[0]["role"], "assistant");
        // A call output's array of input_text parts joins as text
        let r = translate(
            &json!({"input": [
                {"type": "function_call_output", "call_id": "c", "output": [
                    {"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"},
                ]},
            ]}),
            &crate::responses::Store::default(),
        )
        .map(|r| r.chat["messages"][0]["content"].clone())
        .unwrap_or(Value::Null);
        assert_eq!(r, json!("ab"));
    }

    /// The tools, the tool choice and the allowed_tools filter.
    #[test]
    fn tools_and_choice() {
        let r = translate(
            &json!({
                "input": "x",
                "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
                "tool_choice": {"type": "function", "name": "lookup"},
            }),
            &crate::responses::Store::default(),
        )
        .unwrap();
        assert_eq!(
            r.chat["tools"],
            json!([{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}])
        );
        assert_eq!(
            r.chat["tool_choice"],
            json!({"type": "function", "function": {"name": "lookup"}})
        );
        // allowed_tools keeps the named ones and carries the mode
        let r = translate(
            &json!({
                "input": "x",
                "tools": [{"type": "function", "name": "lookup"}, {"type": "function", "name": "other"}],
                "tool_choice": {"type": "allowed_tools", "mode": "required",
                                "tools": [{"type": "function", "name": "lookup"}]},
            }),
            &crate::responses::Store::default(),
        )
        .unwrap();
        assert_eq!(r.chat["tool_choice"], json!("required"));
        assert_eq!(
            r.chat["tools"],
            json!([{"type": "function", "function": {"name": "lookup"}}])
        );
    }

    /// The format chain: text, json_object, json_schema; max_output_tokens
    /// and the reasoning effort pass through.
    #[test]
    fn format_and_passthrough() {
        let r = translate(
            &json!({"input": "x", "text": {"format": {"type": "json_object"}},
                    "max_output_tokens": 12, "reasoning": {"effort": "low"}, "stop": ["\n"]}),
            &crate::responses::Store::default(),
        )
        .unwrap();
        assert_eq!(r.chat["response_format"], json!({"type": "json_object"}));
        assert_eq!(r.chat["max_tokens"], json!(12));
        assert_eq!(r.chat["reasoning_effort"], json!("low"));
        assert_eq!(r.chat["stop"], json!(["\n"]));
        let r = translate(
            &json!({"input": "x", "text": {"format": {"type": "json_schema", "name": "s",
                    "schema": {"type": "object"}, "strict": true}}}),
            &crate::responses::Store::default(),
        )
        .unwrap();
        assert_eq!(
            r.chat["response_format"],
            json!({"type": "json_schema", "json_schema":
                {"name": "s", "schema": {"type": "object"}, "strict": true}})
        );
        // A text format stays unset
        let r = translate(
            &json!({"input": "x", "text": {"format": {"type": "text"}}}),
            &crate::responses::Store::default(),
        )
        .unwrap();
        assert!(r.chat.get("response_format").is_none());
    }
}
