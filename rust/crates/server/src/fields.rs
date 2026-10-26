//! The request-field parsing upstream collects in `fields.zig`: the stop
//! options (``stop_options``), and the thinking and effort chain
//! (``thinking_fields``) — the effort levels a chat template names, the
//! coerced effort, and the request's thinking switch with its DeepSeek
//! spelling.
//!
//! Port of the pieces of `zig/src/server/fields.zig` the chat surfaces read;
//! the number parsing (``parseNumbers``, ``pyIntText``, ``pyFloat``) lives
//! beside its consumers (tool_params.rs carries `pyFloat`).

use serde_json::{json, Value};

/// The efforts a request may name. Upstream: `efforts` (fields.zig:9).
pub const EFFORTS: [&str; 7] = ["none", "minimal", "low", "medium", "high", "xhigh", "max"];

/// The efforts by strength, highest first. Upstream: `effort_order`
/// (fields.zig:10).
const EFFORT_ORDER: [&str; 6] = ["max", "xhigh", "high", "medium", "low", "minimal"];

/// Upstream: `isEffort` (fields.zig:226).
fn is_effort(s: &str) -> bool {
    EFFORTS.contains(&s)
}

/// Upstream: `order` (fields.zig:230) — the effort's place in the order, else
/// None.
fn order(name: &str) -> Option<usize> {
    EFFORT_ORDER.iter().position(|e| *e == name)
}

/// The nearest level the template names, ties going higher; xhigh and none
/// stay. Upstream: `coerceEffort` (fields.zig:236).
pub fn coerce_effort<'e>(effort: &'e str, levels: &[&'e str]) -> &'e str {
    if levels.is_empty() {
        return match effort {
            "high" | "max" => "xhigh",
            "minimal" => "low",
            _ => effort,
        };
    }
    if levels.contains(&effort) {
        return effort;
    }
    if effort == "xhigh" || effort == "none" {
        return effort;
    }
    let Some(want) = order(effort) else {
        return effort;
    };
    let mut best = effort;
    let mut best_key = (usize::MAX, usize::MAX);
    for l in levels {
        let Some(at) = order(l) else { continue };
        let key = (at.abs_diff(want), at);
        if key.0 < best_key.0 || (key.0 == best_key.0 && key.1 < best_key.1) {
            best = l;
            best_key = key;
        }
    }
    best
}

/// The efforts a chat template names in quotes (``effort_levels``).
/// Upstream: `effortLevels` (fields.zig:259).
pub fn effort_levels(template: &str) -> Vec<&str> {
    let mut found: Vec<&str> = Vec::new();
    for name in EFFORT_ORDER {
        let mut at = 0;
        while let Some(i) = template[at..].find(name).map(|x| x + at) {
            at = i + 1;
            if i == 0 || i + name.len() >= template.len() {
                continue;
            }
            let bytes = template.as_bytes();
            let open = bytes[i - 1];
            let close = bytes[i + name.len()];
            if matches!(open, b'\'' | b'"') && matches!(close, b'\'' | b'"') {
                found.push(name);
                break;
            }
        }
    }
    found
}

/// ``chat_template_kwargs.thinking`` as a switch: a bool, or DeepSeek's
/// `{"type": "enabled" | "disabled"}`. Upstream: `thinkingSwitch`
/// (fields.zig:185).
fn thinking_switch(v: Option<&Value>) -> Option<bool> {
    let value = v?;
    if let Some(b) = value.as_bool() {
        return Some(b);
    }
    if let Some(Value::Object(o)) = v {
        if let Some(Value::String(t)) = o.get("type") {
            if t == "enabled" {
                return Some(true);
            }
            if t == "disabled" {
                return Some(false);
            }
        }
    }
    None
}

/// A request's thinking switch and effort. Upstream: `Thinking`
/// (fields.zig:197).
#[derive(Debug)]
pub struct Thinking {
    pub effort: Option<String>,
    pub enable: Option<bool>,
}

/// ``thinking_fields``: a request's effort and thinking switch where it sets
/// them. Upstream: `thinkingFields` (fields.zig:199) — the effort from the
/// body or the kwargs, validated against the efforts and coerced against the
/// template's levels; the switch from the kwargs' `enable_thinking`, else
/// its DeepSeek `thinking` spelling.
pub fn thinking_fields(body: &Value, levels: &[&str]) -> Result<Thinking, String> {
    let mut out = Thinking {
        effort: None,
        enable: None,
    };
    let raw = body.get("chat_template_kwargs");
    let kwargs = raw.filter(|k| crate::pyrepr::truthy(k));
    let kw_object = kwargs.is_none_or(|k| k.is_object());
    let mut enable: Option<Value> = kwargs.and_then(|k| k.get("enable_thinking")).cloned();
    let has_enable = kwargs.is_some_and(|k| k.as_object().is_some_and(|o| o.contains_key("enable_thinking")));
    if kw_object && !has_enable {
        if let Some(sw) = thinking_switch(kwargs.and_then(|k| k.get("thinking"))) {
            enable = Some(Value::Bool(sw));
        }
    }
    let mut effort = body.get("reasoning_effort");
    if effort.is_none() {
        effort = kwargs.and_then(|k| k.get("reasoning_effort"));
    }
    if let Some(e) = effort {
        let Some(effort) = e.as_str().filter(|s| is_effort(s)) else {
            return Err(
                "reasoning_effort must be none, minimal, low, medium, high, xhigh or max"
                    .into(),
            );
        };
        out.effort = Some(coerce_effort(effort, levels).to_string());
        out.enable = Some(effort != "none");
    }
    if kw_object {
        if let Some(on) = enable {
            out.enable = Some(crate::pyrepr::truthy(&on));
            if out.enable == Some(true) && out.effort.as_deref() == Some("none") {
                out.effort = None;
            }
        }
    }
    Ok(out)
}

/// ``stop_options``: ignore_eos must be a bool when present; stop a string,
/// a list of them, or null. Upstream: `stopOptions` (fields.zig:33) — the
/// one refusal message for every wrong shape.
pub fn stop_options(body: &Value) -> Result<(bool, Vec<String>), String> {
    let mut ignore_eos = false;
    if let Some(ignore) = body.get("ignore_eos") {
        let Some(b) = ignore.as_bool() else {
            return Err("ignore_eos must be a boolean".into());
        };
        ignore_eos = b;
    }
    let Some(stop) = body.get("stop") else {
        return Ok((ignore_eos, Vec::new()));
    };
    const REFUSED: &str = "stop must be a nonempty string, a list of nonempty strings, or null";
    let strings = match stop {
        Value::String(s) => {
            if s.is_empty() {
                return Err(REFUSED.into());
            }
            vec![s.clone()]
        }
        Value::Array(list) => {
            let mut out = Vec::with_capacity(list.len());
            for s in list {
                let Some(v) = s.as_str().filter(|v| !v.is_empty()) else {
                    return Err(REFUSED.into());
                };
                out.push(v.to_string());
            }
            out
        }
        _ => return Err(REFUSED.into()),
    };
    Ok((ignore_eos, strings))
}

/// Digits with single underscores between them, as Python's int() accepts.
/// Upstream: `digitRun` (fields.zig:108).
fn digit_run(s: &str, out: &mut String) -> bool {
    let b = s.as_bytes();
    if b.is_empty() || !b[0].is_ascii_digit() || !b[b.len() - 1].is_ascii_digit() {
        return false;
    }
    for i in 0..b.len() {
        let ch = b[i];
        if ch == b'_' {
            if b[i - 1] == b'_' {
                return false;
            }
            continue;
        }
        if !ch.is_ascii_digit() {
            return false;
        }
        out.push(ch as char);
    }
    true
}

/// ``int(text)`` as canonical decimal text, or None. Upstream: `pyIntText`
/// (fields.zig:122) — the sign, the underscore digits, the leading zeros
/// off.
pub fn py_int_text(text: &str) -> Option<String> {
    let mut s = text.trim_matches(|c: char| " \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f".contains(c));
    let mut negative = false;
    if !s.is_empty() && (s.starts_with('-') || s.starts_with('+')) {
        negative = s.starts_with('-');
        s = &s[1..];
    }
    let mut digits = String::new();
    if !digit_run(s, &mut digits) {
        return None;
    }
    let trimmed = digits.trim_start_matches('0');
    if trimmed.is_empty() {
        return Some("0".into());
    }
    Some(if negative { format!("-{trimmed}") } else { trimmed.to_string() })
}

/// The integer fields a body carries. Upstream: `integer_fields`
/// (fields.zig:56).
const INTEGER_FIELDS: [&str; 5] = [
    "max_completion_tokens",
    "max_tokens",
    "seed",
    "thinking_budget",
    "top_k",
];

/// ``int(value)`` for an int, an integral float or a numeric string; None
/// where Python raises. Upstream: `toInt` (fields.zig:72) keeps any i128 as
/// decimal text; the port's JSON cannot hold past 64 bits, so an integer
/// beyond them refuses (the seed's two's-complement bits, a documented
/// deviation at the same place the seed parsing noted one).
fn to_int(v: &Value) -> Option<Value> {
    match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                return Some(json!(i));
            }
            if let Some(u) = n.as_u64() {
                return Some(json!(u));
            }
            // an integral float
            let f = n.as_f64()?;
            if !f.is_finite() || f != f.trunc() || f.abs() >= 1e38 {
                return None;
            }
            Some(json!(f as i64))
        }
        Value::String(s) => {
            let t = py_int_text(s)?;
            serde_json::from_str::<Value>(&t).ok()
        }
        _ => None,
    }
}

/// ``float(value)`` kept only when finite. Upstream: `toFloat`
/// (fields.zig:86).
fn to_float(v: &Value) -> Option<Value> {
    let f = match v {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => crate::tool_params::py_float(s)?,
        _ => return None,
    };
    f.is_finite().then(|| json!(f))
}

/// ``parse_numbers``: the body with its numeric fields as numbers, or
/// Python's refusal. Upstream: `parseNumbers` (fields.zig:60) — the
/// integer fields as ints (an integral float or a numeric string counts),
/// the rest as finite floats; min_p between 0 and 1; a negative top_k the
/// zero it means.
/// Python's type name for ``'<type>' object has no attribute 'get'``.
/// Upstream: `typeName` (fields.zig:13).
pub fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// ``fields.get`` on a body that is not an object fails as Python's
/// AttributeError does — the Other kind, the bare message. Upstream:
/// `requireObject` (fields.zig:26).
pub fn require_object(body: &Value) -> Result<(), crate::errors::Refused> {
    if !body.is_object() {
        return Err(crate::errors::Refused::other(format!(
            "'{}' object has no attribute 'get'",
            type_name(body)
        )));
    }
    Ok(())
}

pub fn parse_numbers(body: &Value) -> Result<Value, String> {
    // ``fields.get`` on a body that is not an object fails as Python's
    // AttributeError does (fields.zig:26's requireObject)
    let Some(object) = body.as_object() else {
        return Err(format!(
            "'{}' object has no attribute 'get'",
            type_name(body)
        ));
    };
    crate::fields::stop_options(body)?;
    let mut parsed = object.clone();
    for name in INTEGER_FIELDS.iter().chain(["temperature", "top_p", "min_p"].iter()) {
        let Some(value) = object.get(*name) else { continue };
        let integer = INTEGER_FIELDS.contains(name);
        let number = if integer {
            to_int(value)
        } else {
            to_float(value)
        };
        let Some(n) = number else {
            return Err(format!(
                "{name} must be {} or null",
                if integer { "an integer" } else { "a finite number" }
            ));
        };
        if *name == "min_p" {
            let f = n.as_f64().unwrap_or(0.0);
            if !(0.0..=1.0).contains(&f) {
                return Err("min_p must be between 0 and 1, or null".into());
            }
        }
        let stored = if *name == "top_k" && n.as_i64().is_some_and(|i| i < 0) {
            json!(0)
        } else {
            n
        };
        parsed.insert((*name).into(), stored);
    }
    Ok(Value::Object(parsed))
}

const MEDIA: [&str; 6] = ["image", "images", "image_url", "input_image", "audio", "input_audio"];

/// ``validate_modalities``: text in, text out. Upstream:
/// `validateModalities` (fields.zig:280) — the media fields and the
/// modalities array refuse, the one text shape passes.
pub fn validate_modalities(body: &Value) -> Result<(), String> {
    for k in MEDIA {
        if body.get(k).is_some_and(crate::pyrepr::truthy) {
            return Err("this server accepts and produces text only; image, audio and video are unsupported".into());
        }
    }
    if let Some(m) = body.get("modalities") {
        match m.as_array() {
            Some(list) if list.len() == 1 && list[0].as_str() == Some("text") => {}
            _ => {
                return Err(
                    "this server accepts and produces text only; image, audio and video are unsupported"
                        .into(),
                )
            }
        }
    }
    Ok(())
}

/// ``probability_options``: n must be 1; the ``top_logprobs`` count when
/// logprobs are asked and ``supported``. Upstream: `probabilityOptions`
/// (fields.zig:265) — the fields read present-and-non-null; null or absent
/// means unset.
pub fn probability_options(body: &Value, supported: bool) -> Result<Option<u8>, String> {
    let field = |k: &str| body.get(k).filter(|v| !v.is_null());
    if let Some(n) = field("n") {
        if n.as_u64() != Some(1) {
            return Err("n must be 1; multiple choices are not supported".into());
        }
    }
    let enabled = field("logprobs");
    if let Some(e) = &enabled {
        if e.as_bool().is_none() {
            return Err("logprobs must be a boolean or null".into());
        }
    }
    let mut count = 0u8;
    if let Some(top) = field("top_logprobs") {
        let Some(n) = top.as_i64().filter(|n| (0..=20).contains(n)) else {
            return Err("top_logprobs must be an integer between 0 and 20, or null".into());
        };
        if enabled.and_then(Value::as_bool) != Some(true) {
            return Err("top_logprobs requires logprobs: true".into());
        }
        count = n as u8;
    }
    if enabled.and_then(Value::as_bool) != Some(true) {
        return Ok(None);
    }
    if !supported {
        return Err("logprobs are not supported by this model or backend".into());
    }
    Ok(Some(count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The upstream test's effort half (fields.zig:308-310), assertion for
    /// assertion.
    #[test]
    fn efforts_coerce_to_the_template_s_levels() {
        let levels = ["low", "medium", "xhigh"];
        assert_eq!(coerce_effort("high", &levels), "xhigh");
        assert_eq!(coerce_effort("minimal", &levels), "low");
        assert_eq!(coerce_effort("max", &[]), "xhigh");
        // Ties go higher
        let levels = ["low", "high"];
        assert_eq!(coerce_effort("medium", &levels), "high");
        // xhigh and none stay
        assert_eq!(coerce_effort("xhigh", &levels), "xhigh");
        assert_eq!(coerce_effort("none", &levels), "none");
        // An unknown effort passes through
        assert_eq!(coerce_effort("banana", &levels), "banana");
    }

    #[test]
    fn the_effort_levels_a_template_names() {
        // The efforts in quotes, by strength, highest first
        let template = r#"{% if effort == 'high' %}x{% elif effort == "low" %}y{% endif %}"#;
        assert_eq!(effort_levels(template), ["high", "low"]);
        // Bare words do not count
        assert_eq!(effort_levels("high low max"), Vec::<&str>::new());
        // At the very edges a quote is missing: no match
        assert_eq!(effort_levels("high'"), Vec::<&str>::new());
    }

    #[test]
    fn the_thinking_fields_read_the_body_and_the_kwargs() {
        // Nothing set: neither
        let t = thinking_fields(&json!({}), &[]).unwrap();
        assert!(t.effort.is_none() && t.enable.is_none());
        // The effort from the body, coerced, enabling thinking
        let t = thinking_fields(&json!({"reasoning_effort": "high"}), &["xhigh"]).unwrap();
        assert_eq!(t.effort.as_deref(), Some("xhigh"));
        assert_eq!(t.enable, Some(true));
        // none disables
        let t = thinking_fields(&json!({"reasoning_effort": "none"}), &[]).unwrap();
        assert_eq!(t.effort.as_deref(), Some("none"));
        assert_eq!(t.enable, Some(false));
        // The kwargs' enable_thinking wins the effort's enable
        let t = thinking_fields(
            &json!({"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": false}}),
            &[],
        )
        .unwrap();
        assert_eq!(t.enable, Some(false));
        assert_eq!(t.effort.as_deref(), Some("low"));
        // enable_thinking clears a none effort
        let t = thinking_fields(
            &json!({"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": true}}),
            &[],
        )
        .unwrap();
        assert_eq!(t.enable, Some(true));
        assert_eq!(t.effort, None);
        // The DeepSeek switch
        let t = thinking_fields(
            &json!({"chat_template_kwargs": {"thinking": {"type": "disabled"}}}),
            &[],
        )
        .unwrap();
        assert_eq!(t.enable, Some(false));
        let t = thinking_fields(
            &json!({"chat_template_kwargs": {"thinking": {"type": "enabled"}}}),
            &[],
        )
        .unwrap();
        assert_eq!(t.enable, Some(true));
        // The kwargs' own effort
        let t = thinking_fields(
            &json!({"chat_template_kwargs": {"reasoning_effort": "medium"}}),
            &[],
        )
        .unwrap();
        assert_eq!(t.effort.as_deref(), Some("medium"));
        // A wrong effort refuses
        match thinking_fields(&json!({"reasoning_effort": "banana"}), &[]) {
            Err(m) => assert_eq!(
                m,
                "reasoning_effort must be none, minimal, low, medium, high, xhigh or max"
            ),
            Ok(_) => panic!("a wrong effort must refuse"),
        }
        // enable_thinking ahead of the switch
        let t = thinking_fields(
            &json!({"chat_template_kwargs": {"enable_thinking": true, "thinking": {"type": "disabled"}}}),
            &[],
        )
        .unwrap();
        assert_eq!(t.enable, Some(true));
    }

    /// The upstream test's numbers half (fields.zig:308-315), assertion for
    /// assertion.
    #[test]
    fn py_int_text_shapes() {
        assert_eq!(py_int_text(" -0_12 ").as_deref(), Some("-12"));
        assert_eq!(py_int_text("8.0"), None);
        assert_eq!(py_int_text("42").as_deref(), Some("42"));
        assert_eq!(py_int_text("-0").as_deref(), Some("0"));
        // Leading zeros read the way Python's int() reads them
        assert_eq!(py_int_text("007").as_deref(), Some("7"));
        assert_eq!(py_int_text("1__0"), None);
        assert_eq!(py_int_text(""), None);
    }

    /// validate_modalities (fields.zig:280): the media refusal, the one
    /// text shape passes.
    #[test]
    fn modalities_validate_to_text_only() {
        assert!(validate_modalities(&json!({})).is_ok());
        assert!(validate_modalities(&json!({"modalities": ["text"]})).is_ok());
        // A null media field is no media
        assert!(validate_modalities(&json!({"image": null})).is_ok());
        for k in ["image", "images", "image_url", "input_image", "audio", "input_audio"] {
            assert_eq!(
                validate_modalities(&json!({k: "x"})),
                Err("this server accepts and produces text only; image, audio and video are unsupported".to_string()),
                "{k} refuses"
            );
        }
        assert_eq!(
            validate_modalities(&json!({"modalities": ["text", "image"]})),
            Err("this server accepts and produces text only; image, audio and video are unsupported".to_string())
        );
        assert!(validate_modalities(&json!({"modalities": []})).is_err());
    }

    /// probability_options (fields.zig:265): the shapes and refusals.
    #[test]
    fn probability_options_shapes() {
        // Nothing asked: none
        assert_eq!(probability_options(&json!({}), true).unwrap(), None);
        // logprobs true alone: the count 0
        assert_eq!(
            probability_options(&json!({"logprobs": true}), true).unwrap(),
            Some(0)
        );
        // logprobs false or null: none
        assert_eq!(
            probability_options(&json!({"logprobs": false}), true).unwrap(),
            None
        );
        assert_eq!(
            probability_options(&json!({"logprobs": null}), true).unwrap(),
            None
        );
        // top_logprobs without logprobs
        assert_eq!(
            probability_options(&json!({"top_logprobs": 5}), true),
            Err("top_logprobs requires logprobs: true".to_string())
        );
        // The count rides
        assert_eq!(
            probability_options(&json!({"logprobs": true, "top_logprobs": 5}), true).unwrap(),
            Some(5)
        );
        // n must be 1
        assert_eq!(
            probability_options(&json!({"n": 1, "logprobs": true}), true).unwrap(),
            Some(0)
        );
        assert_eq!(
            probability_options(&json!({"n": 2, "logprobs": true}), true),
            Err("n must be 1; multiple choices are not supported".to_string())
        );
        // The unsupported backend
        assert_eq!(
            probability_options(&json!({"logprobs": true}), false),
            Err("logprobs are not supported by this model or backend".to_string())
        );
        // The refusals: a non-bool logprobs, out-of-range tops
        assert_eq!(
            probability_options(&json!({"logprobs": "yes"}), true),
            Err("logprobs must be a boolean or null".to_string())
        );
        assert_eq!(
            probability_options(&json!({"logprobs": true, "top_logprobs": 21}), true),
            Err("top_logprobs must be an integer between 0 and 20, or null".to_string())
        );
        assert_eq!(
            probability_options(&json!({"logprobs": true, "top_logprobs": -1}), true),
            Err("top_logprobs must be an integer between 0 and 20, or null".to_string())
        );
        assert_eq!(
            probability_options(&json!({"logprobs": true, "top_logprobs": null}), true).unwrap(),
            Some(0)
        );
    }

    /// parse_numbers: the coercions and refusals (fields.zig:60).
    #[test]
    fn parse_numbers_coerces_and_refuses() {
        // Numeric strings and integral floats become the numbers they are
        let body = json!({"max_tokens": "1024", "temperature": "0.7", "seed": 5.0,
                          "top_k": -3, "min_p": 0.25, "stop": ["END"]});
        let parsed = parse_numbers(&body).unwrap();
        assert_eq!(parsed["max_tokens"], json!(1024));
        assert_eq!(parsed["temperature"], json!(0.7));
        assert_eq!(parsed["seed"], json!(5));
        // A negative top_k stores the zero it means
        assert_eq!(parsed["top_k"], json!(0));
        assert_eq!(parsed["min_p"], json!(0.25));
        // The refusals, the one message each
        assert_eq!(
            parse_numbers(&json!({"top_k": "abc"})),
            Err("top_k must be an integer or null".to_string())
        );
        assert_eq!(
            parse_numbers(&json!({"temperature": "hot"})),
            Err("temperature must be a finite number or null".to_string())
        );
        assert_eq!(
            parse_numbers(&json!({"max_tokens": 8.5})),
            Err("max_tokens must be an integer or null".to_string())
        );
        assert_eq!(
            parse_numbers(&json!({"min_p": 1.5})),
            Err("min_p must be between 0 and 1, or null".to_string())
        );
        // The stop shapes re-validate here (fields.zig:62)
        assert!(parse_numbers(&json!({"stop": "END"})).is_ok());
        assert_eq!(
            parse_numbers(&json!({"stop": ""})),
            Err("stop must be a nonempty string, a list of nonempty strings, or null".to_string())
        );
        // A non-object body fails as Python's AttributeError does
        // (fields.zig:26's requireObject)
        assert_eq!(
            parse_numbers(&json!(5)),
            Err("'int' object has no attribute 'get'".to_string())
        );
        assert_eq!(
            parse_numbers(&json!([])),
            Err("'list' object has no attribute 'get'".to_string())
        );
    }

    /// The stop options: the shapes upstream's `stopOptions` (fields.zig:33)
    /// accepts and refuses.
    #[test]
    fn stop_options_shapes() {
        let (ignore, stops) = stop_options(&json!({})).unwrap();
        assert!(!(ignore || !stops.is_empty()));
        let (ignore, stops) = stop_options(&json!({"stop": "END"})).unwrap();
        assert!(!ignore);
        assert_eq!(stops, ["END"]);
        let (_, stops) = stop_options(&json!({"stop": ["a", "b"]})).unwrap();
        assert_eq!(stops, ["a", "b"]);
        let (_, stops) = stop_options(&json!({"stop": []})).unwrap();
        assert!(stops.is_empty());
        let (ignore, _) = stop_options(&json!({"ignore_eos": true})).unwrap();
        assert!(ignore);
        const REFUSED: &str = "stop must be a nonempty string, a list of nonempty strings, or null";
        assert_eq!(stop_options(&json!({"stop": ""})), Err(REFUSED.to_string()));
        assert_eq!(stop_options(&json!({"stop": 5})), Err(REFUSED.to_string()));
        assert_eq!(stop_options(&json!({"stop": ["a", ""]})), Err(REFUSED.to_string()));
        assert_eq!(stop_options(&json!({"stop": ["a", 1]})), Err(REFUSED.to_string()));
        assert_eq!(
            stop_options(&json!({"ignore_eos": "yes"})),
            Err("ignore_eos must be a boolean".to_string())
        );
    }
}
