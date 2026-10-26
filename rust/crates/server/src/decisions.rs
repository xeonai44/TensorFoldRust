//! /v1/decisions checks a body the way 0.6.6 checks it, then scores
//! through the engine hook when a family sets one.
//!
//! Port of `zig/src/server/decisions.zig` (443 lines).

use crate::errors::Refused;
use crate::pyrepr;
use crate::reply_text::py_strip;
use crate::state::AppState;
use serde_json::{json, Map, Value};

const TOP_FIELDS: [&str; 7] = [
    "input",
    "questions",
    "temperature",
    "chat_template_kwargs",
    "prompt_format_version",
    "return_prompt_token_ids",
    "model",
];
const CHOICE_FIELDS: [&str; 4] = ["id", "type", "question", "options"];
const SCORE_FIELDS: [&str; 4] = ["id", "type", "question", "levels"];
const YES_NO_FIELDS: [&str; 5] = ["id", "type", "question", "yes", "no"];
const OPTION_FIELDS: [&str; 2] = ["name", "description"];
const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/// The question kinds (decisions.zig:32).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Choice,
    Score,
    YesNo,
}

impl Kind {
    fn name(&self) -> &'static str {
        match self {
            Kind::Choice => "choice",
            Kind::Score => "score",
            Kind::YesNo => "yes_no",
        }
    }
}

/// One question prepared for scoring: its prompt and label tokens
/// (decisions.zig:34).
struct Prepared {
    id: String,
    kind: Kind,
    names: Vec<String>,
    prompt_ids: Vec<u32>,
    label_ids: Vec<u32>,
}

/// The scored question: a logit per label and the vocabulary's logsumexp
/// (decisions.zig:41).
struct Scored {
    logits: Vec<f64>,
    logsumexp: f64,
}

/// The decisions object, or the refusal 0.6.6 would send for this body.
/// Upstream: `reply` (decisions.zig:58).
pub fn reply(state: &AppState, raw: &Value) -> Result<Value, Refused> {
    let body = crate::fields::parse_numbers(raw).map_err(Refused::request)?;
    validate(&body).map_err(Refused::request)?;
    let input = render_text(&body).map_err(Refused::request)?;
    if py_strip(&input).is_empty() {
        return Err(Refused::request("input must not be blank"));
    }
    let Some(questions) = body.get("questions").and_then(Value::as_array) else {
        return Err(Refused::request("questions must contain at least one question"));
    };
    let mut prepared: Vec<Prepared> = Vec::new();
    for (index, question) in questions.iter().enumerate() {
        match prepare_one(state, &input, question) {
            Ok(item) => prepared.push(item),
            Err(inner) if inner.kind == crate::errors::Kind::Request => {
                let where_at = question_where(question, index);
                return Err(Refused::request(format!(
                    "question {where_at}: {}",
                    inner.message
                )));
            }
            Err(other) => return Err(other),
        }
    }
    let scored = score_all(state, &prepared)?;
    Ok(build(state, &body, &prepared, &scored))
}

/// Upstream: `validate` (decisions.zig:72).
fn validate(body: &Value) -> Result<(), String> {
    let Some(obj) = body.as_object() else {
        return Err("the request body must be a JSON object".into());
    };
    reject_unknown(obj, &TOP_FIELDS)?;
    if let Some(version) = body.get("prompt_format_version") {
        if !version.is_null() && version != &json!(1) {
            return Err(format!(
                "prompt_format_version {} is not served, this server uses version 1",
                shown(version)
            ));
        }
    }
    check_temperature(body)?;
    if let Some(kwargs) = body.get("chat_template_kwargs") {
        if !kwargs.is_null() {
            let Some(kw) = kwargs.as_object() else {
                return Err("chat_template_kwargs must be an object".into());
            };
            reject_unknown(kw, &["enable_thinking"])?;
            if let Some(flag) = kw.get("enable_thinking") {
                if !(flag == &json!(false)) {
                    return Err(
                        "decisions need enable_thinking false or unset".into(),
                    );
                }
            }
        }
    }
    let Some(questions) = body.get("questions") else {
        return Err("questions must contain at least one question".into());
    };
    let Some(list) = questions.as_array() else {
        return Err("questions must contain at least one question".into());
    };
    if list.is_empty() {
        return Err("questions must contain at least one question".into());
    }
    let mut seen: Vec<String> = Vec::new();
    for question in list {
        let Some(_q) = question.as_object() else {
            return Err("each question must be an object".into());
        };
        let Some(ident) = question.get("id") else {
            return Err("a question id must not be blank".into());
        };
        let Some(s) = ident.as_str() else {
            return Err("a question id must not be blank".into());
        };
        if py_strip(s).is_empty() {
            return Err("a question id must not be blank".into());
        }
        if seen.iter().any(|old| old == s) {
            return Err(format!("question id {} is repeated", pyrepr::repr(ident)));
        }
        seen.push(s.to_string());
    }
    Ok(())
}

/// Upstream: `checkTemperature` (decisions.zig:96) — a number above 0.
fn check_temperature(body: &Value) -> Result<(), String> {
    let Some(v) = body.get("temperature") else { return Ok(()) };
    if v.as_bool().is_some() || v.as_f64().is_none() {
        return Err("temperature must be a number above 0".into());
    }
    let n = v.as_f64().unwrap_or_default();
    if !n.is_finite() || n <= 0.0 {
        return Err("temperature must be a number above 0".into());
    }
    Ok(())
}

/// Upstream: `rejectUnknown` (decisions.zig:102) — the first unknown field
/// named, sorted.
fn reject_unknown(obj: &Map<String, Value>, allowed: &[&str]) -> Result<(), String> {
    let mut bad: Vec<&str> = obj
        .keys()
        .filter(|k| !allowed.contains(&k.as_str()))
        .map(String::as_str)
        .collect();
    if bad.is_empty() {
        return Ok(());
    }
    bad.sort_unstable();
    Err(format!("unknown field {}", pyrepr::repr(&json!(bad[0]))))
}

/// Upstream: `shown` (decisions.zig:113).
fn shown(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(b) => if *b { "True".into() } else { "False".into() },
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Upstream: `renderText` (decisions.zig:127) — a string, or any
/// container's JSON.
fn render_text(body: &Value) -> Result<String, String> {
    match body.get("input") {
        None => Ok(String::new()),
        Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other @ (Value::Object(_) | Value::Array(_))) => Ok(
            crate::tool_parse::zig_stringify(other, true),
        ),
        Some(_) => Err("text must be a string, object, or array".into()),
    }
}

/// Upstream: `questionWhere` (decisions.zig:143).
fn question_where(question: &Value, index: usize) -> String {
    if let Some(id) = question.get("id").and_then(Value::as_str) {
        if !id.is_empty() {
            return pyrepr::repr(&json!(id));
        }
    }
    format!("at position {index}")
}

/// The question's wording (decisions.zig:150's `Wording`).
#[derive(Debug)]
struct Wording {
    kind: Kind,
    names: Vec<String>,
    labels: Vec<String>,
    content: String,
}

/// Upstream: `wording` (decisions.zig:153).
fn wording(input: &str, question: &Value) -> Result<Wording, String> {
    let kind = match question.get("type").and_then(Value::as_str) {
        Some("choice") => Kind::Choice,
        Some("score") => Kind::Score,
        Some("yes_no") => Kind::YesNo,
        other => {
            return Err(format!(
                "unknown question type {}",
                pyrepr::repr(&other.map_or(Value::Null, |s| Value::String(s.to_string())))
            ))
        }
    };
    let Some(qobj) = question.as_object() else {
        return Err("each question must be an object".into());
    };
    let mut lines: Vec<String> = Vec::new();
    let (names, labels) = match kind {
        Kind::Choice => {
            reject_unknown(qobj, &CHOICE_FIELDS)?;
            let listed = choice_list(question)?;
            lines.push(question_line(question)?);
            for ((label, name), detail) in listed
                .labels
                .iter()
                .zip(&listed.names)
                .zip(&listed.details)
            {
                lines.push(if !detail.is_empty() {
                    format!("{label}: {name} - {detail}")
                } else {
                    format!("{label}: {name}")
                });
            }
            (listed.names, listed.labels)
        }
        Kind::Score => {
            reject_unknown(qobj, &SCORE_FIELDS)?;
            let listed = score_list(question)?;
            lines.push(question_line(question)?);
            for (label, detail) in listed.labels.iter().zip(&listed.details) {
                lines.push(format!("{label}: {detail}"));
            }
            (listed.names, listed.labels)
        }
        Kind::YesNo => {
            reject_unknown(qobj, &YES_NO_FIELDS)?;
            let yes = render_one(question.get("yes"))?;
            let no = render_one(question.get("no"))?;
            let lead = render_one(question.get("question"))?;
            if py_strip(&lead).is_empty() {
                return Err("a question must not be blank".into());
            }
            lines.push(format!("Is the following true? {lead}"));
            if !yes.is_empty() {
                lines.push(format!("yes: {yes}"));
            }
            if !no.is_empty() {
                lines.push(format!("no: {no}"));
            }
            (
                vec!["yes".into(), "no".into()],
                vec!["yes".into(), "no".into()],
            )
        }
    };
    lines.push(match kind {
        Kind::Choice => "Answer with the letter of one option only.".into(),
        Kind::Score => "Answer with the number of one level only.".into(),
        Kind::YesNo => "Answer with yes or no only.".into(),
    });
    Ok(Wording {
        kind,
        names,
        labels,
        content: content_of(input, &lines),
    })
}

fn render_one(v: Option<&Value>) -> Result<String, String> {
    match v {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other @ (Value::Object(_) | Value::Array(_))) => {
            Ok(crate::tool_parse::zig_stringify(other, true))
        }
        Some(_) => Err("text must be a string, object, or array".into()),
    }
}

/// Upstream: `choiceList` (decisions.zig:225).
fn choice_list(question: &Value) -> Result<Listed, String> {
    let Some(options) = question.get("options") else {
        return Err("a choice needs 2 to 26 options".into());
    };
    let Some(list) = options.as_array() else {
        return Err("a choice needs 2 to 26 options".into());
    };
    if list.len() < 2 || list.len() > 26 {
        return Err("a choice needs 2 to 26 options".into());
    }
    let mut names = Vec::with_capacity(list.len());
    let mut details = Vec::with_capacity(list.len());
    let mut labels = Vec::with_capacity(list.len());
    let mut seen: Vec<String> = Vec::new();
    for (i, option) in list.iter().enumerate() {
        let Some(o) = option.as_object() else {
            return Err("each option must be an object".into());
        };
        reject_unknown(o, &OPTION_FIELDS)?;
        let Some(name) = option.get("name").and_then(Value::as_str) else {
            return Err("option names must not be blank or contain line breaks".into());
        };
        if bad_name(name) {
            return Err("option names must not be blank or contain line breaks".into());
        }
        let stripped = py_strip(name).to_string();
        if folded_seen(&mut seen, &stripped)? {
            return Err(format!(
                "option name {} repeats another name",
                pyrepr::repr(&json!(name))
            ));
        }
        names.push(stripped);
        details.push(render_one(option.get("description"))?);
        labels.push((ALPHABET[i] as char).to_string());
    }
    Ok(Listed {
        names,
        details,
        labels,
    })
}

/// Upstream: `badName` (decisions.zig:250).
fn bad_name(name: &str) -> bool {
    let stripped = py_strip(name);
    if stripped.is_empty() {
        return true;
    }
    stripped.bytes().any(|c| c < 32)
}

/// Upstream: `foldedSeen` (decisions.zig:255).
fn folded_seen(seen: &mut Vec<String>, name: &str) -> Result<bool, String> {
    let folded = name.to_ascii_lowercase();
    if seen.contains(&folded) {
        return Ok(true);
    }
    seen.push(folded);
    Ok(false)
}

/// Upstream: `scoreList` (decisions.zig:266).
fn score_list(question: &Value) -> Result<Listed, String> {
    let Some(levels) = question.get("levels") else {
        return Err("a score needs 2 to 10 levels".into());
    };
    let Some(list) = levels.as_array() else {
        return Err("a score needs 2 to 10 levels".into());
    };
    if list.len() < 2 || list.len() > 10 {
        return Err("a score needs 2 to 10 levels".into());
    }
    let mut names = Vec::with_capacity(list.len());
    let mut details = Vec::with_capacity(list.len());
    for level in list {
        let text = render_one(Some(level))?;
        if py_strip(&text).is_empty() {
            return Err("a level must not be blank".into());
        }
        details.push(text);
    }
    for i in 0..details.len() {
        names.push(i.to_string());
    }
    Ok(Listed {
        names: names.clone(),
        details,
        labels: names,
    })
}

struct Listed {
    names: Vec<String>,
    details: Vec<String>,
    labels: Vec<String>,
}

/// Upstream: `questionLine` (decisions.zig:288).
fn question_line(question: &Value) -> Result<String, String> {
    let text = render_one(question.get("question"))?;
    if py_strip(&text).is_empty() {
        return Err("a question must not be blank".into());
    }
    Ok(format!("Question: {text}"))
}

/// Upstream: `contentOf` (decisions.zig:295).
fn content_of(input: &str, lines: &[String]) -> String {
    let mut out = String::from(input);
    out.push_str("\n\n");
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(line);
    }
    out
}

/// Upstream: `prepareOne` (decisions.zig:306).
fn prepare_one(state: &AppState, input: &str, question: &Value) -> Result<Prepared, Refused> {
    let word = wording(input, question).map_err(Refused::request)?;
    let prompt_text = render_prompt(state, &word.content)?;
    let prompt_ids = encode_prompt(state, &prompt_text).map_err(Refused::request)?;
    think_open(&prompt_text).map_err(Refused::request)?;
    fits(state, prompt_ids.len()).map_err(Refused::request)?;
    let label_ids = label_ids(state, &prompt_text, &prompt_ids, &word.labels).map_err(Refused::request)?;
    Ok(Prepared {
        id: question
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        kind: word.kind,
        names: word.names,
        prompt_ids,
        label_ids,
    })
}

/// Upstream: `renderPrompt` (decisions.zig:326).
fn render_prompt(state: &AppState, content: &str) -> Result<String, Refused> {
    let text = state.text.as_ref().ok_or(Refused::request("no text"))?;
    let messages = vec![json!({"role": "user", "content": content})];
    text.template
        .render(
            &messages,
            tensorfold_core::RenderOptions {
                add_generation_prompt: true,
                enable_thinking: false,
                reasoning_effort: None,
                tools: None,
            },
        )
        .map_err(|e| Refused::request(format!("the chat template failed: {e}")))
}

/// Upstream: `encodePrompt` (decisions.zig:339).
fn encode_prompt(state: &AppState, text: &str) -> Result<Vec<u32>, String> {
    let Some(t) = state.text.as_ref() else {
        return Err("the tokenizer cannot encode this prompt".into());
    };
    Ok(t.tokenizer.encode(text))
}

/// Upstream: `thinkOpen` (decisions.zig:350) — the open think marker at
/// the answer position refuses.
fn think_open(text: &str) -> Result<(), String> {
    let Some(open_at) = text.rfind(crate::reply_text::THINK_OPEN) else {
        return Ok(());
    };
    if let Some(close_at) = text.rfind(crate::reply_text::THINK_MARKERS.close) {
        if open_at <= close_at {
            return Ok(());
        }
    }
    Err("the chat template leaves a reasoning block open at the answer position".into())
}

/// Upstream: `fits` (decisions.zig:362).
fn fits(state: &AppState, n: usize) -> Result<(), String> {
    let window = state.engine_info.as_ref().map_or(0, |i| i.context_window);
    if window > 0 && n >= window as usize {
        return Err(format!(
            "the prompt has {n} tokens, which does not fit the context length of {window} tokens"
        ));
    }
    Ok(())
}

/// Upstream: `labelIds` (decisions.zig:366) — each label one fresh token
/// after the prompt.
fn label_ids(
    state: &AppState,
    prompt: &str,
    prompt_ids: &[u32],
    labels: &[String],
) -> Result<Vec<u32>, String> {
    let Some(t) = state.text.as_ref() else {
        return Err("no tokenizer".into());
    };
    let mut found: Vec<u32> = Vec::new();
    for label in labels {
        let joined = format!("{prompt}{label}");
        let ids = t.tokenizer.encode(&joined);
        let fresh = ids.len() == prompt_ids.len() + 1
            && ids[..prompt_ids.len()] == prompt_ids[..]
            && !found.contains(&ids[ids.len() - 1]);
        if !fresh {
            return Err(label_refuse(label));
        }
        found.push(ids[ids.len() - 1]);
    }
    Ok(found)
}

/// Upstream: `labelRefuse` (decisions.zig:387).
fn label_refuse(label: &str) -> String {
    format!(
        "the answer label {} is not one distinct token after the chat prompt for this tokenizer, so this model is not supported",
        pyrepr::repr(&json!(label))
    )
}

/// Upstream: `scoreAll` (decisions.zig:357).
fn score_all(state: &AppState, prepared: &[Prepared]) -> Result<Vec<Scored>, Refused> {
    let Some(engine) = state.engine.as_ref() else {
        return Err(Refused::request("decision scoring is not supported by this engine yet"));
    };
    let mut scored = Vec::with_capacity(prepared.len());
    for item in prepared {
        let mut logits = vec![0.0f64; item.label_ids.len()];
        match engine.score(&item.prompt_ids, &item.label_ids, &mut logits) {
            Ok(logsumexp) => scored.push(Scored { logits, logsumexp }),
            Err(tensorfold_engine::ScoreError::Unsupported) => {
                return Err(Refused::request(
                    "decision scoring is not supported by this engine yet",
                ))
            }
            Err(tensorfold_engine::ScoreError::Failed) => {
                return Err(Refused {
                    kind: crate::errors::Kind::Server,
                    message: "decision scoring failed".into(),
                })
            }
        }
    }
    Ok(scored)
}

/// Upstream: `build` (decisions.zig:372).
fn build(
    state: &AppState,
    body: &Value,
    prepared: &[Prepared],
    scored: &[Scored],
) -> Value {
    let temperature = body.get("temperature").and_then(Value::as_f64).unwrap_or(1.0);
    let show_ids = body
        .get("return_prompt_token_ids")
        .is_some_and(crate::pyrepr::truthy);
    let mut answers = Map::new();
    let mut prompt_tokens: usize = 0;
    for (item, row) in prepared.iter().zip(scored) {
        let mut probs = vec![0.0f64; row.logits.len()];
        softmax(&row.logits, temperature, &mut probs);
        let mut one = Map::new();
        one.insert("type".into(), json!(item.kind.name()));
        let mut map = Map::new();
        for (name, p) in item.names.iter().zip(&probs) {
            map.insert(name.clone(), json!(p));
        }
        one.insert("probabilities".into(), Value::Object(map));
        one.insert(
            "label_mass".into(),
            json!(label_mass(&row.logits, row.logsumexp)),
        );
        match item.kind {
            Kind::Choice => {
                let mut best = 0;
                for (i, p) in probs.iter().enumerate() {
                    if *p > probs[best] {
                        best = i;
                    }
                }
                one.insert("choice".into(), json!(item.names[best]));
            }
            Kind::Score => {
                let mut total = 0.0;
                for (i, p) in probs.iter().enumerate() {
                    total += i as f64 * p;
                }
                one.insert("score".into(), json!(total));
            }
            Kind::YesNo => {}
        }
        if show_ids {
            one.insert(
                "prompt_token_ids".into(),
                Value::Array(item.prompt_ids.iter().map(|id| json!(id)).collect()),
            );
            one.insert(
                "label_token_ids".into(),
                Value::Array(item.label_ids.iter().map(|id| json!(id)).collect()),
            );
        }
        answers.insert(item.id.clone(), Value::Object(one));
        prompt_tokens += item.prompt_ids.len();
    }
    let named = body.get("model");
    let mut out = Map::new();
    let _ = state;
    out.insert("object".into(), json!("decisions"));
    out.insert(
        "model".into(),
        match named {
            Some(n) if crate::pyrepr::truthy(n) => n.clone(),
            _ => json!("default"),
        },
    );
    out.insert("prompt_format_version".into(), json!(1));
    out.insert("answers".into(), Value::Object(answers));
    out.insert(
        "usage".into(),
        json!({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
        }),
    );
    Value::Object(out)
}

/// Upstream: `softmax` (decisions.zig:421).
fn softmax(logits: &[f64], temperature: f64, out: &mut [f64]) {
    let peak = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut total = 0.0;
    for (x, p) in logits.iter().zip(out.iter_mut()) {
        *p = ((x - peak) / temperature).exp();
        total += *p;
    }
    for p in out.iter_mut() {
        *p /= total;
    }
}

/// Upstream: `labelMass` (decisions.zig:429).
fn label_mass(logits: &[f64], logsumexp: f64) -> f64 {
    logits.iter().map(|logit| (logit - logsumexp).exp()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// validate: the unknown fields, the version, the temperature, the
    /// kwargs, the ids (decisions.zig:72's checks, one message each).
    #[test]
    fn the_body_validates() {
        assert!(validate(&json!({
            "input": "x",
            "questions": [{"id": "q1", "type": "yes_no", "question": "is it?", "yes": "y", "no": "n"}],
        }))
        .is_ok());
        assert!(validate(&json!({"input": "x", "zebra": 1, "alpha": 2})).is_err());
    }

    #[test]
    fn the_unknown_field_names_the_first_sorted() {
        let e = reject_unknown(
            json!({"zebra": 1, "alpha": 2}).as_object().unwrap(),
            &["model"],
        );
        assert_eq!(e, Err("unknown field 'alpha'".to_string()));
        assert!(reject_unknown(json!({"model": 1}).as_object().unwrap(), &["model"]).is_ok());
    }

    /// The temperature check (decisions.zig:96).
    #[test]
    fn the_temperature_is_a_number_above_zero() {
        assert!(check_temperature(&json!({"temperature": 0.7})).is_ok());
        assert_eq!(
            check_temperature(&json!({"temperature": true})),
            Err("temperature must be a number above 0".to_string())
        );
        assert_eq!(
            check_temperature(&json!({"temperature": 0})),
            Err("temperature must be a number above 0".to_string())
        );
    }

    /// The choice wording (decisions.zig:153 + choiceList): the labels
    /// the alphabet, the closing line.
    #[test]
    fn the_choice_words() {
        let w = wording(
            "The input",
            &json!({
                "type": "choice", "question": "pick one",
                "options": [
                    {"name": "First"},
                    {"name": "Second", "description": "the second"},
                ],
            }),
        )
        .unwrap();
        assert_eq!(w.kind, Kind::Choice);
        assert_eq!(w.names, ["First", "Second"]);
        assert_eq!(w.labels, ["A", "B"]);
        assert_eq!(
            w.content,
            "The input\n\nQuestion: pick one\nA: First\nB: Second - the second\nAnswer with the letter of one option only."
        );
        // Repeats refuse (folded)
        assert!(wording("x", &json!({
            "type": "choice", "question": "q",
            "options": [{"name": "One"}, {"name": "ONE"}],
        }))
        .is_err());
        // 2 to 26
        assert!(wording("x", &json!({"type": "choice", "options": [{"name": "a"}]}))
            .unwrap_err()
            .contains("2 to 26"));
        // Blank names refuse
        assert!(wording("x", &json!({
            "type": "choice", "options": [{"name": "  "}, {"name": "b"}],
        }))
        .is_err());
    }

    /// The score and yes_no wordings (decisions.zig:192/208).
    #[test]
    fn the_score_and_yes_no_word() {
        let w = wording("in", &json!({
            "type": "score", "question": "how good?",
            "levels": ["bad", "ok", "great"],
        }))
        .unwrap();
        assert_eq!(w.names, ["0", "1", "2"]);
        assert_eq!(w.labels, ["0", "1", "2"]);
        let w = wording("in", &json!({
            "type": "yes_no", "question": "true?", "yes": "yes!", "no": "no!",
        }))
        .unwrap();
        assert_eq!(w.names, ["yes", "no"]);
        assert!(w.content.contains("Is the following true? true?"));
        assert!(w.content.contains("yes: yes!"));
        assert!(w.content.contains("no: no!"));
        assert!(w
            .content
            .ends_with("Answer with yes or no only."));
        // An unknown type names itself
        assert!(wording("x", &json!({"type": "banana"}))
            .unwrap_err()
            .contains("'banana'"));
    }

    /// The think-open check (decisions.zig:350).
    #[test]
    fn an_open_think_block_refuses() {
        let open = crate::reply_text::THINK_OPEN;
        let close = crate::reply_text::THINK_MARKERS.close;
        assert!(think_open("plain").is_ok());
        assert!(think_open(&format!("{open} x {close}")).is_ok());
        // An open marker with no close after it refuses
        assert_eq!(
            think_open(&format!("{open} only")),
            Err("the chat template leaves a reasoning block open at the answer position".to_string())
        );
    }

    /// The softmax and label mass (decisions.zig:421/429).
    #[test]
    fn the_softmax_and_label_mass() {
        let mut out = vec![0.0; 2];
        softmax(&[0.0, 0.0], 1.0, &mut out);
        assert!((out[0] - 0.5).abs() < 1e-12 && (out[1] - 0.5).abs() < 1e-12);
        let hot = [10.0, 0.0];
        let mut p = vec![0.0; 2];
        softmax(&hot, 1.0, &mut p);
        assert!(p[0] > 0.999);
        let lse = 1.0; // ln(e + 1)
        assert!((label_mass(&[0.0f64, 0.0], lse) - 2.0 * (0.0 - lse).exp()).abs() < 1e-12);
    }
}
