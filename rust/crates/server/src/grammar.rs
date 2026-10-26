//! Structured-output fields checked as ``engine.grammar.request_spec``
//! checks them; the engine enforces them.
//!
//! Port of `zig/src/server/grammar.zig` (101 lines).

use serde_json::{json, Value};
use tensorfold_engine::StructureKind;

/// Upstream: `fields` (grammar.zig:9).
pub const FIELDS: [&str; 6] = [
    "response_format",
    "guided_json",
    "guided_regex",
    "guided_choice",
    "guided_grammar",
    "structured_outputs",
];

/// The structured-output request one body makes. Upstream: `Spec`
/// (grammar.zig:13).
#[derive(Debug)]
pub struct Spec {
    pub kind: StructureKind,
    pub text: String,
    /// The field the spec came from, the refusal names it.
    pub field: &'static str,
}

/// The JSON decoder's own message, without its position. Upstream:
/// `bareMessage` (grammar.zig:16).
fn bare_message(full: &str) -> &str {
    match full.rfind(": line ") {
        Some(at) => &full[..at],
        None => full,
    }
}

/// The schema as JSON text. Upstream: `schemaText` (grammar.zig:22) — a
/// string parsed, a bool the empty schema or null, an object kept; the
/// text in Zig's wide JSON.
fn schema_text(raw: &Value, at: &str) -> Result<String, String> {
    let mut value = raw.clone();
    if let Value::String(s) = &raw {
        value = serde_json::from_str(s).map_err(|e| {
            format!("{at} is not valid JSON: {}", bare_message(&e.to_string()))
        })?;
    }
    if let Value::Bool(b) = value {
        value = if b { json!({}) } else { Value::Null };
    }
    if !value.is_object() {
        return Err(format!("{at} must be a JSON schema object"));
    }
    Ok(crate::tool_parse::zig_stringify(&value, false))
}

/// Upstream: `text` (grammar.zig:35).
fn text_field(v: &Value, at: &str) -> Result<String, String> {
    match v {
        Value::String(s) if !s.is_empty() => Ok(s.clone()),
        _ => Err(format!("{at} must be a non-empty string")),
    }
}

/// Upstream: `choices` (grammar.zig:40) — the list as a JSON array's text.
fn choices(v: &Value, at: &str) -> Result<String, String> {
    let ok = v.as_array().is_some_and(|list| {
        !list.is_empty()
            && list
                .iter()
                .all(|c| c.as_str().is_some_and(|s| !s.is_empty()))
    });
    if !ok {
        return Err(format!(
            "{at} must be a non-empty list of non-empty strings"
        ));
    }
    Ok(crate::tool_parse::zig_stringify(v, false))
}

/// Upstream: `read` (grammar.zig:48).
fn read(kind: StructureKind, v: &Value, at: &str) -> Result<String, String> {
    match kind {
        StructureKind::JsonSchema => schema_text(v, at),
        StructureKind::Choice => choices(v, at),
        _ => text_field(v, at),
    }
}

/// The body's structured-output request, None for plain text; Err when it
/// is malformed. Upstream: `spec` (grammar.zig:56) — response_format first,
/// the guided fields, structured_outputs' sorted keys.
pub fn spec(body: &Value) -> Result<Option<Spec>, String> {
    let field = |k: &str| body.get(k).filter(|v| !v.is_null());
    if let Some(rf) = field("response_format") {
        let Some(obj) = rf.as_object() else {
            return Err(
                "response_format must be an object such as {\"type\": \"json_object\"}".into(),
            );
        };
        let kind = obj.get("type");
        let name = kind.and_then(Value::as_str).unwrap_or("");
        if name == "json_object" {
            return Ok(Some(Spec {
                kind: StructureKind::Json,
                text: String::new(),
                field: "response_format",
            }));
        }
        if name == "json_schema" {
            let schema = obj
                .get("json_schema")
                .and_then(Value::as_object)
                .and_then(|o| o.get("schema").filter(|v| !v.is_null()));
            let Some(schema) = schema else {
                return Err(
                    "response_format json_schema needs json_schema.schema (a JSON schema object)"
                        .into(),
                );
            };
            return Ok(Some(Spec {
                kind: StructureKind::JsonSchema,
                text: schema_text(schema, "response_format json_schema.schema")?,
                field: "response_format",
            }));
        }
        if name != "text" {
            return Err(format!(
                "response_format type must be text, json_object or json_schema, not {}",
                crate::pyrepr::repr(kind.unwrap_or(&Value::Null))
            ));
        }
    }
    const GUIDED: [(&str, StructureKind); 4] = [
        ("guided_json", StructureKind::JsonSchema),
        ("guided_regex", StructureKind::Regex),
        ("guided_choice", StructureKind::Choice),
        ("guided_grammar", StructureKind::Grammar),
    ];
    for (name, kind) in GUIDED {
        if let Some(v) = field(name) {
            return Ok(Some(Spec {
                kind,
                text: read(kind, v, name)?,
                field: name,
            }));
        }
    }
    let Some(so) = field("structured_outputs") else {
        return Ok(None);
    };
    let Some(obj) = so.as_object() else {
        return Err(
            "structured_outputs must be an object such as {\"json\": {...}}".into(),
        );
    };
    let mut given: Vec<&String> = obj
        .iter()
        .filter(|(_, v)| !v.is_null() && v.as_bool() != Some(false))
        .map(|(k, _)| k)
        .collect();
    given.sort();
    const KNOWN: [(&str, StructureKind); 4] = [
        ("json", StructureKind::JsonSchema),
        ("regex", StructureKind::Regex),
        ("choice", StructureKind::Choice),
        ("grammar", StructureKind::Grammar),
    ];
    let other: Vec<&str> = given
        .iter()
        .filter(|k| {
            !KNOWN.iter().any(|(n, _)| *n == k.as_str()) && k.as_str() != "json_object"
        })
        .map(|k| k.as_str())
        .collect();
    if !other.is_empty() {
        return Err(format!(
            "structured_outputs {} is not supported: use json, json_object, regex, choice or grammar",
            other.join(", ")
        ));
    }
    for k in &given {
        for (n, kind) in KNOWN {
            if k.as_str() == n {
                return Ok(Some(Spec {
                    kind,
                    text: read(
                        kind,
                        obj.get(k.as_str()).expect("the sorted key exists"),
                        &format!("structured_outputs.{k}"),
                    )?,
                    field: "structured_outputs",
                }));
            }
        }
    }
    if obj.get("json_object").is_some_and(crate::pyrepr::truthy) {
        return Ok(Some(Spec {
            kind: StructureKind::Json,
            text: String::new(),
            field: "structured_outputs",
        }));
    }
    Ok(None)
}

/// ``grammar.refusal``: a malformed grammar, or one beside a required call,
/// refused before any header. Upstream: `refusal` (grammar.zig:96).
pub fn refusal(body: &Value) -> Result<(), String> {
    let Some(s) = spec(body)? else {
        return Ok(());
    };
    let tools = body.get("tools");
    if tools.is_some_and(crate::pyrepr::truthy)
        && crate::tool_specs::choice_requires_call(body.get("tool_choice"))
    {
        return Err(format!(
            "{} cannot be combined with tool_choice \"required\" or a named function: send one",
            s.field
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The spec's every shape and refusal (grammar.zig:56).
    #[test]
    fn the_specs_a_body_makes() {
        // Nothing asked: plain text
        assert!(spec(&json!({})).unwrap().is_none());
        // response_format's three names
        assert!(spec(&json!({"response_format": {"type": "text"}}))
            .unwrap()
            .is_none());
        let s = spec(&json!({"response_format": {"type": "json_object"}}))
            .unwrap()
            .unwrap();
        assert_eq!(s.kind, StructureKind::Json);
        assert_eq!(s.field, "response_format");
        let s = spec(&json!({
            "response_format": {"type": "json_schema",
                                "json_schema": {"schema": {"type": "object", "properties": {"a": {"type": "integer"}}}}}
        }))
        .unwrap()
        .unwrap();
        assert_eq!(s.kind, StructureKind::JsonSchema);
        assert_eq!(s.text, r#"{"type": "object", "properties": {"a": {"type": "integer"}}}"#);
        // A string schema parses
        let s = spec(&json!({
            "response_format": {"type": "json_schema", "json_schema": {"schema": "{\"type\": \"object\"}"}}
        }))
        .unwrap()
        .unwrap();
        assert_eq!(s.text, r#"{"type": "object"}"#);
        // The wrong shapes refuse, the exact words
        match spec(&json!({"response_format": "json_object"})) {
            Err(m) => assert_eq!(m, "response_format must be an object such as {\"type\": \"json_object\"}".to_string()),
            Ok(_) => panic!("the spec refuses"),
        };
        match spec(&json!({"response_format": {"type": "json_schema", "json_schema": {}}})) {
            Err(m) => assert_eq!(m, "response_format json_schema needs json_schema.schema (a JSON schema object)".to_string()),
            Ok(_) => panic!("the spec refuses"),
        };
        match spec(&json!({"response_format": {"type": "banana"}})) {
            Err(m) => assert_eq!(m, "response_format type must be text, json_object or json_schema, not 'banana'".to_string()),
            Ok(_) => panic!("the spec refuses"),
        };
        match spec(&json!({"response_format": {"type": 5}})) {
            Err(m) => assert_eq!(m, "response_format type must be text, json_object or json_schema, not 5".to_string()),
            Ok(_) => panic!("the spec refuses"),
        };
        // A bad string schema carries the decoder's bare message
        match spec(&json!({
            "response_format": {"type": "json_schema", "json_schema": {"schema": "{bad"}}
        })) {
            Err(m) => assert!(m.starts_with("response_format json_schema.schema is not valid JSON: "), "{m}"),
            Ok(_) => panic!("a bad schema refuses"),
        }
        // The guided fields
        let s = spec(&json!({"guided_regex": "^a+$"})).unwrap().unwrap();
        assert_eq!(s.kind, StructureKind::Regex);
        assert_eq!(s.text, "^a+$");
        let s = spec(&json!({"guided_json": {"type": "object"}}))
            .unwrap()
            .unwrap();
        assert_eq!(s.kind, StructureKind::JsonSchema);
        let s = spec(&json!({"guided_choice": ["yes", "no"]}))
            .unwrap()
            .unwrap();
        assert_eq!(s.kind, StructureKind::Choice);
        assert_eq!(s.text, r#"["yes", "no"]"#);
        let s = spec(&json!({"guided_grammar": "root ::= 'a'"}))
            .unwrap()
            .unwrap();
        assert_eq!(s.kind, StructureKind::Grammar);
        match spec(&json!({"guided_choice": []})) {
            Err(m) => assert_eq!(m, "guided_choice must be a non-empty list of non-empty strings".to_string()),
            Ok(_) => panic!("the spec refuses"),
        };
        match spec(&json!({"guided_regex": ""})) {
            Err(m) => assert_eq!(m, "guided_regex must be a non-empty string".to_string()),
            Ok(_) => panic!("the spec refuses"),
        };
        match spec(&json!({"guided_json": [1]})) {
            Err(m) => assert_eq!(m, "guided_json must be a JSON schema object".to_string()),
            Ok(_) => panic!("the spec refuses"),
        };
        // structured_outputs: the sorted keys, the known names
        let s = spec(&json!({"structured_outputs": {"regex": "x", "json": {"type": "object"}}}))
            .unwrap()
            .unwrap();
        assert_eq!(s.kind, StructureKind::JsonSchema, "the sorted first known key wins");
        let s = spec(&json!({"structured_outputs": {"json_object": true}}))
            .unwrap()
            .unwrap();
        assert_eq!(s.kind, StructureKind::Json);
        assert!(spec(&json!({"structured_outputs": {"json_object": false}}))
            .unwrap()
            .is_none());
        match spec(&json!({"structured_outputs": {"banana": {"x": 1}}})) {
            Err(m) => assert_eq!(m, "structured_outputs banana is not supported: use json, json_object, regex, choice or grammar".to_string()),
            Ok(_) => panic!("the spec refuses"),
        };
        match spec(&json!({"structured_outputs": {"banana": {}, "apple": {}}})) {
            Err(m) => assert_eq!(m, "structured_outputs apple, banana is not supported: use json, json_object, regex, choice or grammar".to_string()),
            Ok(_) => panic!("the spec refuses"),
        };
    }

    /// A grammar beside a required call refuses (grammar.zig:96).
    #[test]
    fn a_grammar_beside_a_required_call_refuses() {
        assert!(refusal(&json!({"response_format": {"type": "json_object"}})).is_ok());
        assert_eq!(
            refusal(&json!({
                "response_format": {"type": "json_object"},
                "tools": [{"type": "function", "function": {"name": "f"}}],
                "tool_choice": "required"
            })),
            Err("response_format cannot be combined with tool_choice \"required\" or a named function: send one".to_string())
        );
        // The guided field's own name
        assert_eq!(
            refusal(&json!({
                "guided_regex": "x",
                "tools": [{"type": "function", "function": {"name": "f"}}],
                "tool_choice": {"type": "function", "function": {"name": "f"}}
            })),
            Err("guided_regex cannot be combined with tool_choice \"required\" or a named function: send one".to_string())
        );
        // No tools: no refusal
        assert!(refusal(&json!({
            "guided_regex": "x", "tool_choice": "required"
        }))
        .is_ok());
    }
}
