//! Chat messages as the template sees them: leading instructions merged, text
//! parts joined, call arguments as objects.
//!
//! Port of `zig/src/server/messages.zig`. The roles, the merge of leading
//! system/developer text, the late-system role for later instructions, and the
//! placeholder user turn for templates that demand a query — all exactly as
//! upstream's `normalize_messages`, with its four tests mirrored.

use serde_json::{json, Map, Value};

const ROLES: [&str; 5] = ["system", "developer", "user", "assistant", "tool"];

fn is_role(role: Option<&Value>) -> bool {
    match role {
        Some(Value::String(r)) => ROLES.contains(&r.as_str()),
        _ => false,
    }
}

/// A media part refusal: upstream refuses any non-text content (fields.hasMedia).
fn has_media(value: &Value) -> bool {
    let Some(object) = value.as_object() else { return false };
    // A part with an image/audio/video type, or a message whose content parts
    // hold one (fields.zig's walk over the known media keys).
    const MEDIA: [&str; 6] = [
        "image", "image_url", "audio", "input_audio", "video", "video_url",
    ];
    for (k, v) in object {
        if MEDIA.contains(&k.as_str()) {
            return true;
        }
        if k == "content" {
            if let Some(parts) = v.as_array() {
                for part in parts {
                    if has_media(part) {
                        return true;
                    }
                }
            }
        }
        if let Value::String(t) = v {
            if MEDIA.contains(&t.as_str()) {
                return true;
            }
        }
    }
    false
}

fn with_field(object: &Map<String, Value>, key: &str, value: Value) -> Value {
    let mut copy = object.clone();
    copy.insert(key.to_string(), value);
    Value::Object(copy)
}

/// A bare text part. Upstream: `textPart` (messages.zig:137) — shared by the
/// Anthropic and Responses translations.
pub fn text_part(text: &str) -> Value {
    json!({"type": "text", "text": text})
}

/// `normalize_messages` (text only): leading system and developer text merged,
/// later ones as `late_system`; a template that needs a user query gains one
/// user turn after a trailing tool run. Upstream: `normalize`
/// (messages.zig:25).
pub fn normalize(
    messages: Option<&Value>,
    late_system: &str,
    needs_user_after_tool: bool,
) -> Result<Vec<Value>, String> {
    let Some(Value::Array(list)) = messages else {
        return Err("messages must be a non-empty list".into());
    };
    if list.is_empty() {
        return Err("messages must be a non-empty list".into());
    }
    let mut out: Vec<Value> = Vec::with_capacity(list.len());
    let mut instructions: Vec<Map<String, Value>> = Vec::new();
    for message in list {
        let Some(object) = message.as_object() else {
            return Err("each message must be an object".into());
        };
        let role = object.get("role");
        if !is_role(role) {
            return Err("message role must be system, developer, user, assistant or tool".into());
        }
        if has_media(message) {
            return Err(
                "this server accepts text only; image, audio and video inputs are unsupported"
                    .into(),
            );
        }
        let content = object.get("content");
        let mut item: Value = message.clone();
        match content {
            Some(Value::Array(parts)) => {
                let mut text = String::new();
                for part in parts {
                    let typed = part
                        .as_object()
                        .map(|o| {
                            o.get("type").and_then(Value::as_str) == Some("text")
                        })
                        .unwrap_or(false);
                    if !typed || has_media(part) {
                        return Err(
                            "this server accepts text parts only; image, audio and video inputs are unsupported"
                                .into(),
                        );
                    }
                    let Some(Value::String(t)) = part.get("text") else {
                        return Err("a text content part must contain a text string".into());
                    };
                    text.push_str(t);
                }
                item = with_field(object, "content", Value::String(text));
            }
            None => {
                item = with_field(object, "content", Value::String(String::new()));
            }
            Some(Value::Null) => {
                item = with_field(object, "content", Value::String(String::new()));
            }
            Some(Value::String(_)) => {}
            Some(_) => {
                return Err("message content must be text or an array of text parts".into());
            }
        }
        let r = role.and_then(Value::as_str).unwrap_or("");
        if r == "system" || r == "developer" {
            if out.is_empty() {
                instructions.push(item.as_object().unwrap().clone());
                continue;
            }
            if r != late_system {
                let object = item.as_object().unwrap().clone();
                item = with_field(&object, "role", Value::String(late_system.to_string()));
            }
        }
        out.push(item);
    }
    if !instructions.is_empty() {
        let mut joined = String::new();
        for (i, m) in instructions.iter().enumerate() {
            if i > 0 {
                joined.push_str("\n\n");
            }
            if let Some(Value::String(text)) = m.get("content") {
                joined.push_str(text);
            }
        }
        let mut first = instructions[0].clone();
        first.insert("role".into(), Value::String("system".into()));
        first.insert("content".into(), Value::String(joined));
        out.insert(0, Value::Object(first));
    }
    // A template that demands a user query gains a placeholder user turn after
    // its last tool run (messages.zig:73's rule).
    if needs_user_after_tool {
        let mut has_query = false;
        let mut last_tool: Option<usize> = None;
        for (i, m) in out.iter().enumerate() {
            let Some(Value::String(role)) = m.get("role") else { continue };
            if role == "user" {
                let text = match m.get("content") {
                    Some(Value::String(t)) => t.as_str(),
                    _ => "",
                };
                let trimmed = text.trim_matches(|c: char| " \t\n\r\u{b}\u{c}".contains(c));
                if !(trimmed.starts_with("<tool_response>") && trimmed.ends_with("</tool_response>")) {
                    has_query = true;
                }
            } else if role == "tool" {
                last_tool = Some(i);
            }
        }
        if !has_query {
            let at = last_tool.map_or(out.len(), |i: usize| i + 1);
            out.insert(
                at,
                json!({"role": "user", "content": "(tool results above)"}),
            );
        }
    }
    Ok(out)
}

/// `_normalize_tool_call_arguments`: call arguments as objects for templates;
/// bad ones under `_invalid_arguments`. Upstream: `toolArguments`
/// (messages.zig:100).
pub fn tool_arguments(messages: Vec<Value>) -> Vec<Value> {
    let mut out = Vec::with_capacity(messages.len());
    for message in messages {
        let mut slot = message.clone();
        let Some(calls) = message.get("tool_calls") else {
            out.push(slot);
            continue;
        };
        let Some(call_list) = calls.as_array() else {
            out.push(slot);
            continue;
        };
        if call_list.is_empty() {
            out.push(slot);
            continue;
        }
        let mut touched = false;
        let mut new_calls = Vec::with_capacity(call_list.len());
        for call in call_list {
            let mut dst = call.clone();
            let Some(function) = call.get("function") else {
                new_calls.push(dst);
                continue;
            };
            let Some(function_object) = function.as_object() else {
                new_calls.push(dst);
                continue;
            };
            if !function_object.contains_key("arguments") {
                new_calls.push(dst);
                continue;
            }
            let args = function_object.get("arguments").unwrap();
            if args.is_object() {
                new_calls.push(dst);
                continue;
            }
            let mut parsed: Option<Value> = Some(args.clone());
            if let Value::String(text) = args {
                parsed = serde_json::from_str(text).ok();
            }
            let parsed = match parsed {
                Some(v) if v.is_object() => v,
                _ => json!({"_invalid_arguments": args}),
            };
            let fn_copy = with_field(function_object, "arguments", parsed);
            let call_object = dst.as_object().unwrap().clone();
            dst = with_field(&call_object, "function", fn_copy);
            touched = true;
            new_calls.push(dst);
        }
        if touched {
            let message_object = message.as_object().unwrap().clone();
            slot = with_field(&message_object, "tool_calls", Value::Array(new_calls));
        }
        out.push(slot);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Mirrors messages.zig:160: a conversation with a user query stays
    /// byte-identical under the rule.
    #[test]
    fn a_conversation_with_a_query_stays_byte_identical() {
        let messages = json!([
            {"role": "user", "content": "run it"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "f", "arguments": {}}}
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "42 rows"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t2", "type": "function", "function": {"name": "g", "arguments": {}}}
            ]},
            {"role": "tool", "tool_call_id": "t2", "content": "ok"},
            {"role": "assistant", "content": "done"}
        ]);
        let out = normalize(Some(&messages), "system", true).unwrap();
        assert_eq!(
            serde_json::to_string(&Value::Array(out)).unwrap(),
            serde_json::to_string(&messages).unwrap()
        );
    }

    /// Mirrors messages.zig:172: a conversation with no user query gains one
    /// placeholder turn after its last tool run.
    #[test]
    fn a_conversation_without_a_query_gains_a_placeholder() {
        let messages = json!([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "f", "arguments": {}}}
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "42 rows"},
            {"role": "tool", "tool_call_id": "t2", "content": "ok"}
        ]);
        let out = normalize(Some(&messages), "system", true).unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!(out[1]["role"], "tool");
        assert_eq!(out[1]["content"], "42 rows");
        assert_eq!(out[2]["role"], "tool");
        assert_eq!(out[2]["content"], "ok");
        assert_eq!(out[3]["role"], "user");
        assert_eq!(out[3]["content"], "(tool results above)");
    }

    /// Mirrors messages.zig:189: a user message whose content is wholly a tool
    /// block does not count as a query.
    #[test]
    fn a_wholly_tool_block_user_message_is_no_query() {
        let tool_block: String = TOOL_BLOCK_CONTENT.to_string();
        let messages = json!([
            {"role": "user", "content": tool_block},
            {"role": "assistant", "content": ""},
            {"role": "tool", "tool_call_id": "t1", "content": "done"}
        ]);
        let out = normalize(Some(&messages), "system", true).unwrap();
        assert_eq!(out.len(), 4);
        assert_eq!(out[3]["role"], "user");
        assert_eq!(out[3]["content"], "(tool results above)");
    }

    /// Mirrors messages.zig:210: a conversation keeps its shape when the
    /// template does not need the rule.
    #[test]
    fn the_rule_off_keeps_the_shape() {
        let messages = json!([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "f", "arguments": {}}}
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "42 rows"}
        ]);
        let out = normalize(Some(&messages), "system", false).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["role"], "assistant");
        assert_eq!(out[1]["role"], "tool");
    }

    /// Mirrors `toolArguments`' contract: string arguments parse into objects,
    /// bad ones wrap under `_invalid_arguments`.
    #[test]
    fn tool_arguments_become_objects() {
        let messages = json!([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{\"x\": 1}"}},
                {"id": "t2", "type": "function", "function": {"name": "g", "arguments": "not json"}},
                {"id": "t3", "type": "function", "function": {"name": "h", "arguments": {"y": 2}}}
            ]}
        ]);
        let list: Vec<Value> = messages.as_array().unwrap().clone();
        let out = tool_arguments(list);
        let calls = out[0]["tool_calls"].as_array().unwrap();
        assert_eq!(calls[0]["function"]["arguments"], json!({"x": 1}));
        assert_eq!(
            calls[1]["function"]["arguments"],
            json!({"_invalid_arguments": "not json"})
        );
        assert_eq!(calls[2]["function"]["arguments"], json!({"y": 2}));
    }
}

/// The wholly-tool-block test's content, byte-precise (upstream's own bytes).
#[cfg(test)]
const TOOL_BLOCK_CONTENT: &str = "<tool_response>\ngot 42 rows\n</tool_response>";
