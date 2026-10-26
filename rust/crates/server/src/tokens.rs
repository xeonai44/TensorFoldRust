//! vLLM's /tokenize and /detokenize (the ids the chat and completion
//! routes would run), and a reply's token SHA.
//!
//! Port of `zig/src/server/tokens.zig` (114 lines). The token SHA is beside
//! this in `chat.rs` (token_sha); the routes themselves are here.

use crate::errors::Refused;
use crate::state::AppState;
use serde_json::{json, Map, Value};

/// The ids the chat and completion routes would run. Upstream: `tokenize`
/// (tokens.zig:25) — the chat template's render when a body sends messages,
/// else the plain encode; the token texts beside the ids when asked.
pub fn tokenize(state: &AppState, body: &Value) -> Result<Value, Refused> {
    let text = state
        .text
        .as_ref()
        .ok_or_else(|| Refused::request(state.engineless_problem.clone()))?;
    let Some(obj) = body.as_object() else {
        return Err(Refused::request("the request body must be a JSON object"));
    };
    let strings = flag(obj, "return_token_strs", false)?;
    let ids: Vec<u32> = if obj.contains_key("messages") {
        let generation = flag(obj, "add_generation_prompt", true)?;
        let msgs = obj
            .get("messages")
            .and_then(Value::as_array)
            .ok_or_else(|| Refused::request("messages must be a list"))?;
        let normalized = crate::messages::normalize(
            Some(&Value::Array(msgs.clone())),
            "system",
            text.needs_user_after_tool,
        )
        .map_err(Refused::request)?;
        let normalized = crate::messages::tool_arguments(normalized);
        let tools = crate::tool_specs::active(obj.get("tools"), obj.get("tool_choice"))
            .map_err(Refused::other)?;
        let levels: Vec<&str> = text.effort_levels.iter().map(String::as_str).collect();
        let thinking_opts = crate::fields::thinking_fields(body, &levels).map_err(Refused::request)?;
        let on = thinking_opts
            .enable
            .unwrap_or(state.config.enable_thinking);
        let effort = thinking_opts
            .effort
            .as_deref()
            .or(state.config.reasoning_effort.as_deref())
            .map(|e| crate::fields::coerce_effort(e, &levels).to_string());
        crate::chat::render_prompt(text, &normalized, on, effort.as_deref(), generation, &tools)
            .map_err(Refused::request)?
    } else {
        let Some(prompt) = body.get("prompt").and_then(Value::as_str) else {
            return Err(Refused::request("prompt must be a string (or send messages)"));
        };
        let special = flag(obj, "add_special_tokens", true)?;
        if special {
            text.tokenizer.encode_special(prompt)
        } else {
            text.tokenizer.encode(prompt)
        }
    };
    let mut o = Map::new();
    o.insert("count".into(), json!(ids.len()));
    o.insert(
        "max_model_len".into(),
        state
            .engine_info
            .as_ref()
            .filter(|i| i.context_window > 0)
            .map(|i| json!(i.context_window))
            .unwrap_or(Value::Null),
    );
    o.insert(
        "tokens".into(),
        Value::Array(ids.iter().map(|t| json!(t)).collect()),
    );
    if strings {
        let strs: Vec<Value> = ids
            .iter()
            .map(|t| json!(text.tokenizer.token_string(*t).unwrap_or_default()))
            .collect();
        o.insert("token_strs".into(), Value::Array(strs));
    }
    Ok(Value::Object(o))
}

/// The text of the ids. Upstream: `detokenize` (tokens.zig:82) — one prompt
/// a request, the ids in the vocabulary's range.
pub fn detokenize(state: &AppState, body: &Value) -> Result<Value, Refused> {
    let Some(obj) = body.as_object() else {
        return Err(Refused::request("the request body must be a JSON object"));
    };
    // The nested single-prompt shape un-nests
    let value: &Value = obj.get("tokens").unwrap_or(&Value::Null);
    let list: Vec<Value> = match value.as_array() {
        Some(list) if list.len() == 1 && list[0].as_array().is_some() => {
            list[0].as_array().unwrap().clone()
        }
        Some(list) => list.clone(),
        None => {
            return Err(Refused::request(
                "tokens must be a list of integer token ids (one prompt a request)",
            ))
        }
    };
    // Every token an integer first (the shape), then the range against the
    // vocabulary
    if list.iter().any(|t| t.as_i64().is_none()) {
        return Err(Refused::request(
            "tokens must be a list of integer token ids (one prompt a request)",
        ));
    }
    let text = state
        .text
        .as_ref()
        .ok_or_else(|| Refused::request(state.engineless_problem.clone()))?;
    let vocab = text.tokenizer.vocab_size() as i64;
    let mut out: Vec<u32> = Vec::with_capacity(list.len());
    for t in list {
        let n = t.as_i64().unwrap_or(-1);
        if n < 0 || n >= vocab {
            return Err(Refused::request(format!(
                "tokens token ids must be in the vocabulary's range 0 to {}",
                vocab - 1
            )));
        }
        out.push(n as u32);
    }
    Ok(json!({"prompt": text.tokenizer.decode(&out)}))
}

/// A named flag, or the refusal naming it. Upstream: `flag` (tokens.zig:15).
fn flag(obj: &Map<String, Value>, name: &str, default: bool) -> Result<bool, Refused> {
    match obj.get(name) {
        None => Ok(default),
        Some(v) => v
            .as_bool()
            .ok_or_else(|| Refused::request(format!("{name} must be a boolean"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The detokenize validations (tokens.zig:82's checks): the shape and
    /// the range.
    #[test]
    fn the_detokenize_validations() {
        let state = AppState::new(crate::Config::new("m".into()));
        assert!(matches!(
            detokenize(&state, &json!({"tokens": 5})),
            Err(Refused { message, .. }) if message.contains("list of integer")
        ));
        assert!(matches!(
            detokenize(&state, &json!({"tokens": ["x"]})),
            Err(Refused { message, .. }) if message.contains("list of integer")
        ));
        // The nested single-prompt shape un-nests
        let state = crate::state::AppState::new(crate::Config::new("m".into()));
        // No text: the engineless refusal (the shape checked first)
        assert!(detokenize(&state, &json!({"tokens": [[1, 2]]})).is_err());
    }

    /// The flag reads (tokens.zig:15).
    #[test]
    fn the_flags() {
        let obj = json!({"a": true, "b": false}).as_object().unwrap().clone();
        assert!(matches!(flag(&obj, "a", false), Ok(true)));
        assert!(matches!(flag(&obj, "b", true), Ok(false)));
        assert!(matches!(flag(&obj, "c", true), Ok(true)));
        let bad = json!({"x": 5}).as_object().unwrap().clone();
        assert!(matches!(
            flag(&bad, "x", true),
            Err(Refused { message, .. }) if message == "x must be a boolean"
        ));
    }
}
