//! Offered tools and tool_choice as the template should see them
//! (`tools.active_tool_specs`).
//!
//! Port of `zig/src/server/tool_specs.zig` (97 lines).

use serde_json::Value;

/// Python's `str()` of a decoded JSON scalar (containers approximate with JSON).
/// Upstream: `pyStr` (tool_specs.zig:9).
pub fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        other => other.to_string(),
    }
}

/// `str(value or "").strip()`. Upstream: `nameText` (tool_specs.zig:21).
fn name_text(v: Option<&Value>) -> String {
    let Some(value) = v else { return String::new() };
    if value == &Value::Null || value == &Value::Bool(false) {
        return String::new();
    }
    py_str(value).trim_matches(|c: char| " \t\r\n\u{b}\u{c}".contains(c)).to_string()
}

/// Upstream: `toolName` (tool_specs.zig:28).
pub fn tool_name(tool: &Value) -> String {
    let Some(object) = tool.as_object() else { return String::new() };
    if let Some(function) = object.get("function") {
        if let Some(function_object) = function.as_object() {
            return name_text(function_object.get("name"));
        }
    }
    name_text(object.get("name"))
}

fn lower_eq(text: &str, want: &str) -> bool {
    text.trim_matches(|c: char| " \t\r\n\u{b}\u{c}".contains(c))
        .eq_ignore_ascii_case(want)
}

/// Upstream: `choiceDisables` (tool_specs.zig:38).
pub fn choice_disables(choice: Option<&Value>) -> bool {
    let Some(c) = choice else { return false };
    if let Value::String(s) = c {
        return lower_eq(s, "none");
    }
    if let Value::Object(o) = c {
        let value = o
            .get("type")
            .or_else(|| o.get("mode"))
            .filter(|v| **v != Value::Null && **v != Value::Bool(false));
        return matches!(value, Some(Value::String(s)) if lower_eq(s, "none"));
    }
    false
}

/// OpenAI's "required" or a named function: the reply must call a tool.
/// Upstream: `choiceRequiresCall` (tool_specs.zig:45).
pub fn choice_requires_call(choice: Option<&Value>) -> bool {
    let Some(c) = choice else { return false };
    if let Value::String(s) = c {
        return lower_eq(s, "required");
    }
    let Some(o) = c.as_object() else { return false };
    let value = match o.get("type").or_else(|| o.get("mode")) {
        Some(v) => py_str(v),
        None => String::new(),
    };
    if lower_eq(&value, "required") {
        return true;
    }
    lower_eq(&value, "function") && o.get("function").is_some_and(|f| f.is_object())
}

fn is_function_choice(c: &Value) -> bool {
    let Some(o) = c.as_object() else { return false };
    let Some(t) = o.get("type") else { return false };
    if *t == Value::Null || *t == Value::Bool(false) {
        return false;
    }
    py_str(t).eq_ignore_ascii_case("function")
}

/// The tools the template offers: none for "none", only the named one for a
/// named function. Upstream: `active` (tool_specs.zig:58).
pub fn active(tools: Option<&Value>, choice: Option<&Value>) -> Result<Vec<Value>, String> {
    let mut specs: &[Value] = &[];
    if let Some(Value::Array(list)) = tools {
        for (i, tool) in list.iter().enumerate() {
            if !tool.is_object() {
                return Err(format!("tools[{}] must be an object", i));
            }
            if tool_name(tool).is_empty() {
                return Err(format!("tools[{}] must include a function name", i));
            }
        }
        specs = list;
    }
    if choice_disables(choice) {
        return Ok(Vec::new());
    }
    if specs.is_empty() {
        if choice_requires_call(choice) {
            return Err(
                "tool_choice requires a tool call, but the request offers no tools".into(),
            );
        }
        return Ok(Vec::new());
    }
    let Some(c) = choice else { return Ok(specs.to_vec()) };
    if !is_function_choice(c) {
        return Ok(specs.to_vec());
    }
    let Some(function) = c.get("function") else {
        return Err("tool_choice function must include a function object".into());
    };
    let Some(function_object) = function.as_object() else {
        return Err("tool_choice function must include a function object".into());
    };
    let requested = name_text(function_object.get("name"));
    if requested.is_empty() {
        return Err("tool_choice function must include a name".into());
    }
    let named: Vec<Value> = specs
        .iter()
        .filter(|tool| tool_name(tool) == requested)
        .cloned()
        .collect();
    if named.is_empty() {
        return Err(format!("tool_choice requested unknown tool '{}'", requested));
    }
    Ok(named)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_names_read_both_shapes() {
        assert_eq!(
            tool_name(&json!({"function": {"name": "get_weather"}})),
            "get_weather"
        );
        assert_eq!(tool_name(&json!({"name": "lookup"})), "lookup");
        assert_eq!(tool_name(&json!({"name": "  " })), "");
    }

    #[test]
    fn choice_rules_match_upstream() {
        assert!(choice_disables(Some(&json!("none"))));
        assert!(choice_disables(Some(&json!({"type": "None"}))));
        assert!(!choice_disables(Some(&json!({"type": "auto"}))));
        assert!(choice_requires_call(Some(&json!("required"))));
        assert!(choice_requires_call(Some(&json!({"type": "function", "function": {"name": "f"}}))));
        assert!(!choice_requires_call(Some(&json!({"type": "auto"}))));
    }

    #[test]
    fn active_selects_the_named_tool() {
        let tools = json!([
            {"type": "function", "function": {"name": "a"}},
            {"type": "function", "function": {"name": "b"}},
        ]);
        let out = active(Some(&tools), None).unwrap();
        assert_eq!(out.len(), 2);
        let named = active(
            Some(&tools),
            Some(&json!({"type": "function", "function": {"name": "b"}})),
        )
        .unwrap();
        assert_eq!(named.len(), 1);
        assert_eq!(named[0]["function"]["name"], "b");
        let none = active(Some(&tools), Some(&json!("none"))).unwrap();
        assert!(none.is_empty());
        assert!(active(None, Some(&json!("required"))).is_err());
        assert!(active(
            Some(&tools),
            Some(&json!({"type": "function", "function": {"name": "zzz"}}))
        )
        .is_err());
    }
}
