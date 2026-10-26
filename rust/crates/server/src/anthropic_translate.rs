//! Anthropic Messages requests as the chat requests that run them.
//!
//! Port of `zig/src/server/anthropic_translate.zig` (293 lines). The Messages
//! body becomes the OpenAI chat body the engine path already speaks: system
//! text into a leading system message, content blocks into text parts,
//! tool_use into tool_calls, tool_result into tool messages, the thinking
//! switches into chat_template_kwargs.

use serde_json::{json, Map, Value};

fn string(value: Option<&Value>, name: &str, empty: bool) -> Result<String, String> {
    let Some(Value::String(s)) = value else {
        return Err(format!(
            "{} must be a {}string",
            name,
            if empty { "" } else { "nonempty " }
        ));
    };
    if !empty && s.is_empty() {
        return Err(format!("{} must be a nonempty string", name));
    }
    Ok(s.clone())
}

/// A content block list: a bare string is one text part.
/// Upstream: `parts` (anthropic_translate.zig:15).
/// The blocks of a content value: a bare string as one text part, an array of
/// objects as itself. Upstream: `parts` (anthropic_translate.zig:15) — the
/// string case allocates the one-part list into `cx.a`; the port owns it.
fn blocks_of(value: Option<&Value>) -> Result<Vec<Value>, String> {
    match value {
        Some(Value::String(s)) => Ok(vec![crate::messages::text_part(s)]),
        Some(Value::Array(list)) => {
            for p in list {
                if !p.is_object() {
                    return Err(
                        "content must be a string or an array of content blocks".into(),
                    );
                }
            }
            Ok(list.clone())
        }
        _ => Err("content must be a string or an array of content blocks".into()),
    }
}

/// A borrowed view of `blocks_of` for places upstream keeps the parts borrowed.
fn parts(value: Option<&Value>) -> Result<Vec<Value>, String> {
    blocks_of(value)
}

fn role_is(v: &Value, names: &[&str]) -> bool {
    v.get("role").is_some_and(|r| {
        r.as_str().is_some_and(|role| names.contains(&role))
    })
}

/// The message list as chat messages. Upstream: `messages`
/// (anthropic_translate.zig:54).
fn translate_messages(
    list: Option<&Value>,
    system: Option<&Value>,
) -> Result<Vec<Value>, String> {
    let Some(Value::Array(list)) = list else {
        return Err("messages must be a nonempty array".into());
    };
    let mut out: Vec<Value> = Vec::new();
    // The system prompt: a text-only block list, joined, as the leading system.
    if let Some(sys) = system {
        if *sys != Value::Null {
            let ps = parts(Some(sys))?;
            for p in ps.iter() {
                if p.get("type").and_then(Value::as_str) != Some("text") {
                    return Err("system accepts text blocks only".into());
                }
            }
            let mut joined = String::new();
            for (i, p) in ps.iter().enumerate() {
                if i > 0 {
                    joined.push_str("\n\n");
                }
                joined.push_str(&string(p.get("text"), "system.text", true)?);
            }
            out.push(json!({"role": "system", "content": joined}));
        }
    }
    for (index, message) in list.iter().enumerate() {
        if !message.is_object()
            || !role_is(message, &["user", "assistant", "system"])
        {
            return Err("message role must be user, assistant or system".into());
        }
        let role = message.get("role").and_then(Value::as_str).unwrap_or("");
        if role == "system" {
            // Mid-conversation system messages must follow a user turn and
            // precede an assistant turn (anthropic_translate.zig:106).
            let previous = if index > 0 {
                &list[index - 1]
            } else {
                &Value::Object(Map::new())
            };
            let following = if index + 1 < list.len() {
                &list[index + 1]
            } else {
                &json!({"role": "assistant"})
            };
            if !role_is(previous, &["user", "system"])
                || !role_is(following, &["assistant", "system"])
            {
                return Err(
                    "mid-conversation system messages must follow a user turn and precede an assistant or end"
                        .into(),
                );
            }
        }
        let mut content: Vec<Value> = Vec::new();
        let mut calls: Vec<Value> = Vec::new();
        let mut thoughts = String::new();
        let mut any_thought = false;
        let blocks = match message.get("content") {
            Some(content) => parts(Some(content))?,
            None => Vec::new(),
        };
        for block in &blocks {
            let kind = block.get("type").and_then(Value::as_str).unwrap_or("");
            let user = role == "user";
            let assistant = role == "assistant";
            if kind == "text" {
                content.push(crate::messages::text_part(&string(block.get("text"), "text", true)?));
            } else if kind == "image" && user {
                // The port's engine is text-only: the media refusal is the same
                // word the chat route refuses with.
                return Err(
                    "this server accepts text only; image, audio and video inputs are unsupported"
                        .into(),
                );
            } else if kind == "thinking" && assistant {
                thoughts.push_str(&string(block.get("thinking"), "thinking", true)?);
                any_thought = true;
            } else if kind == "redacted_thinking" && assistant {
                return Err(
                    "redacted thinking cannot be decoded by this local model; send plaintext thinking"
                        .into(),
                );
            } else if kind == "tool_use" && assistant {
                let arguments = block.get("input").unwrap_or(&Value::Null);
                if !arguments.is_object() {
                    return Err("tool_use.input must be an object".into());
                }
                let id = string(block.get("id"), "tool_use.id", false)?;
                let name = string(block.get("name"), "tool_use.name", false)?;
                calls.push(json!({
                    "id": id, "type": "function",
                    "function": {"name": name, "arguments": arguments.to_string()},
                }));
            } else if kind == "tool_result" && user {
                let mut texts: Vec<String> = Vec::new();
                for part in parts(block.get("content").or(Some(&json!(""))))? {
                    if part.get("type").and_then(Value::as_str) == Some("text") {
                        let t = string(part.get("text"), "tool_result.text", true)?;
                        texts.push(t);
                    } else if part.get("type").and_then(Value::as_str) == Some("image") {
                        return Err(
                            "this server accepts text only; image, audio and video inputs are unsupported"
                                .into(),
                        );
                    } else {
                        return Err(format!(
                            "unsupported tool_result content type {}",
                            part.get("type").map(crate::pyrepr::repr).unwrap_or_default()
                        ));
                    }
                }
                let mut text = texts.join("\n");
                if block.get("is_error").is_some_and(|e| {
                    e == &Value::Bool(true)
                        || e.as_str().is_some_and(|s| !s.is_empty() && s != "false")
                }) {
                    text = format!("Tool error: {}", text);
                }
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": string(block.get("tool_use_id"), "tool_use_id", false)?,
                    "content": text,
                }));
                continue;
            } else if kind.is_empty() {
                return Err("content blocks must include a type".into());
            } else {
                return Err(format!("unsupported {} content block type {}", role, kind));
            }
        }
        if !content.is_empty() || !calls.is_empty() || any_thought {
            let mut item = Map::new();
            item.insert("role".into(), Value::String(role.into()));
            item.insert(
                "content".into(),
                if !content.is_empty() {
                    Value::Array(content)
                } else {
                    Value::String(String::new())
                },
            );
            if !calls.is_empty() {
                item.insert("tool_calls".into(), Value::Array(calls));
            }
            if any_thought {
                item.insert("reasoning_content".into(), Value::String(thoughts));
            }
            out.push(Value::Object(item));
        } else if blocks.is_empty() {
            out.push(json!({"role": role, "content": ""}));
        }
    }
    Ok(out)
}

fn positive_int(v: Option<&Value>) -> Option<i64> {
    let x = v?;
    let n = x.as_i64().or_else(|| x.as_u64().map(|u| u as i64))?;
    if n > 0 {
        Some(n)
    } else {
        None
    }
}

/// The chat request a Messages request runs as; `count` for count_tokens
/// (max_tokens is not required). Upstream: `translate`
/// (anthropic_translate.zig:177).
pub fn translate(body: &Value, count: bool) -> Result<Value, String> {
    if !body.is_object() {
        return Err("request body must be an object".into());
    }
    let model = string(body.get("model"), "model", false)?;
    let mut chat = Map::new();
    chat.insert("model".into(), Value::String(model));
    chat.insert(
        "messages".into(),
        Value::Array(translate_messages(
            body.get("messages"),
            body.get("system"),
        )?),
    );
    if !count {
        let _ = positive_int(body.get("max_tokens"))
            .ok_or_else(|| "max_tokens must be a positive integer".to_string())?;
        chat.insert("max_tokens".into(), body.get("max_tokens").cloned().unwrap());
    } else {
        chat.insert("max_tokens".into(), json!(1));
    }
    if let Some(s) = body.get("stream") {
        if !s.is_boolean() {
            return Err("stream must be a boolean".into());
        }
        chat.insert("stream".into(), s.clone());
    }
    for key in ["temperature", "top_p", "top_k"] {
        if let Some(v) = body.get(key) {
            chat.insert(key.into(), v.clone());
        }
    }
    if let Some(Value::Array(stops)) = body.get("stop_sequences") {
        for s in stops {
            if s.as_str().is_none_or(|s| s.is_empty()) {
                return Err("stop_sequences must be an array of nonempty strings".into());
            }
        }
        chat.insert("stop".into(), Value::Array(stops.clone()));
    }
    if let Some(context) = body.get("context_management") {
        let allowed = json!({"type": "clear_thinking_20251015", "keep": "all"});
        let edits = context
            .get("edits")
            .cloned()
            .unwrap_or_else(|| json!([]));
        let ok = context.is_object()
            && edits.as_array().is_some_and(|list| {
                list.iter().all(|e| *e == allowed)
            });
        if !ok {
            return Err(
                "context_management supports clear_thinking with keep: all only".into(),
            );
        }
    }
    for name in ["container", "mcp_servers", "service_tier"] {
        if let Some(v) = body.get(name) {
            let fine = (v.as_array().is_some_and(|a| a.is_empty()))
                || v.as_str() == Some("auto");
            if !fine {
                return Err(format!("{} is not supported by this server", name));
            }
        }
    }
    let tools = body.get("tools").cloned().unwrap_or_else(|| json!([]));
    let Some(tool_list) = tools.as_array() else {
        return Err("tools must be an array".into());
    };
    let mut translated: Vec<Value> = Vec::new();
    for tool in tool_list {
        let kind = tool
            .get("type")
            .cloned()
            .unwrap_or_else(|| json!("custom"));
        if !tool.is_object() || kind.as_str() != Some("custom") {
            return Err("only client-defined function tools are supported".into());
        }
        let name = string(tool.get("name"), "tool.name", false)?;
        let schema = tool.get("input_schema").unwrap_or(&Value::Null);
        if !schema.is_object() {
            return Err("tool.input_schema must be an object".into());
        }
        let mut function = Map::new();
        function.insert("name".into(), Value::String(name));
        function.insert("parameters".into(), schema.clone());
        if let Some(description) = tool.get("description") {
            function.insert(
                "description".into(),
                Value::String(string(Some(description), "tool.description", true)?),
            );
        }
        translated.push(json!({"type": "function", "function": function}));
    }
    if !translated.is_empty() {
        chat.insert("tools".into(), Value::Array(translated));
    }
    if let Some(choice) = body.get("tool_choice") {
        let Some(choice_object) = choice.as_object() else {
            return Err("tool_choice must be an object".into());
        };
        let k = choice.get("type").and_then(Value::as_str).unwrap_or("");
        if k == "auto" || k == "none" || k == "any" {
            chat.insert(
                "tool_choice".into(),
                Value::String(if k == "any" { "required".into() } else { k.into() }),
            );
        } else if k == "tool" {
            let name = string(choice.get("name"), "tool_choice.name", false)?;
            chat.insert(
                "tool_choice".into(),
                json!({"type": "function", "function": {"name": name}}),
            );
        } else {
            return Err("tool_choice.type must be auto, none, any or tool".into());
        }
        if let Some(d) = choice_object.get("disable_parallel_tool_use") {
            if !d.is_boolean() {
                return Err("disable_parallel_tool_use must be a boolean".into());
            }
            chat.insert("parallel_tool_calls".into(), Value::Bool(!d.as_bool().unwrap()));
        }
    }
    let config = body.get("output_config").cloned().unwrap_or_else(|| json!({}));
    if !config.is_object() {
        return Err("output_config must be an object".into());
    }
    if let Some(e) = config.get("effort") {
        let known = e.as_str().is_some_and(|name| {
            ["none", "minimal", "low", "medium", "high", "xhigh", "max"].contains(&name)
        });
        if !known {
            return Err("unsupported output_config.effort".into());
        }
        chat.insert("reasoning_effort".into(), e.clone());
    }
    if let Some(fmt) = config.get("format") {
        let schema = fmt.get("schema");
        if fmt.get("type").and_then(Value::as_str) != Some("json_schema")
            || schema.is_none_or(|s| !s.is_object())
        {
            return Err(
                "output_config.format must be json_schema with a schema object".into(),
            );
        }
        chat.insert(
            "response_format".into(),
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema.cloned().unwrap(),
                    "strict": true,
                },
            }),
        );
    }
    let thinking = body.get("thinking").cloned().unwrap_or_else(|| json!({"type": "disabled"}));
    let t = thinking.get("type").and_then(Value::as_str).unwrap_or("");
    if !thinking.is_object()
        || !(t == "disabled" || t == "enabled" || t == "adaptive")
    {
        return Err("thinking.type must be disabled, enabled or adaptive".into());
    }
    let enabled = t != "disabled";
    chat.insert(
        "chat_template_kwargs".into(),
        json!({"enable_thinking": enabled}),
    );
    if !enabled {
        chat.insert("reasoning_effort".into(), Value::String("none".into()));
    }
    if t == "enabled" {
        let budget = positive_int(thinking.get("budget_tokens"));
        let limit = body
            .get("max_tokens")
            .and_then(Value::as_i64)
            .unwrap_or(i64::MAX);
        if budget.is_none_or(|b| !count && b >= limit) {
            return Err(
                "thinking.budget_tokens must be positive and less than max_tokens".into(),
            );
        }
        chat.insert(
            "thinking_budget".into(),
            thinking.get("budget_tokens").cloned().unwrap(),
        );
    }
    Ok(Value::Object(chat))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The core translation: system, user text, assistant thinking.
    #[test]
    fn translates_system_text_and_thinking() {
        let body = json!({
            "model": "claude-x",
            "system": "Be terse.",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": "hello!"}
                ]}
            ],
            "max_tokens": 16,
        });
        let chat = translate(&body, false).unwrap();
        let messages = chat["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "Be terse.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"].as_array().unwrap()[0]["text"], "hi");
        assert_eq!(messages[2]["reasoning_content"], "hmm");
        assert_eq!(chat["max_tokens"], 16);
    }

    /// The refusals mirror upstream's own messages.
    #[test]
    fn refuses_bad_shapes() {
        assert!(translate(&json!({"messages": []}), false).is_err());
        assert!(translate(&json!({"model": "m", "messages": [{"role": "user", "content": "x"}]}), false).is_err());
        let bad_tool = json!({"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 8, "tools": [{"type": "server"}]});
        assert!(translate(&bad_tool, false).is_err());
        let bad_thinking = json!({"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 8, "thinking": {"type": "weird"}});
        assert!(translate(&bad_thinking, false).is_err());
    }

    /// tool_result blocks become tool messages, errors prefixed.
    #[test]
    fn tool_results_become_tool_messages() {
        let body = json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "run it"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {"x": 1}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "42 rows"}]}
                ]}
            ],
            "max_tokens": 16,
        });
        let chat = translate(&body, false).unwrap();
        let messages = chat["messages"].as_array().unwrap();
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "t1");
        assert_eq!(messages[2]["content"], "42 rows");
        assert_eq!(messages[1]["tool_calls"][0]["function"]["name"], "f");
    }
}
