//! OpenAI ``logprobs`` on a chat reply: which requests may ask, and
//! ``choices[0].logprobs`` from the engine's rows.
//!
//! Port of `zig/src/server/logprobs.zig` (55 lines).

use crate::state::Text;
use serde_json::{json, Map, Value};

/// The rows are cut where the text is: after the end-of-thinking token,
/// which must be a token of its own. Upstream: `admit` (logprobs.zig:15) —
/// it reads nothing of the server but that token, which the caller hands it.
pub fn admit(
    think_end: Option<u32>,
    thinking: bool,
    think_budget: u32,
    count: u8,
) -> Result<u8, String> {
    if thinking && think_end.is_none() {
        return Err(
            "logprobs with thinking on need the template's end-of-thinking marker as one token"
                .into(),
        );
    }
    if think_budget > 0 {
        return Err("logprobs support nonstreamed chat without tools, stop strings, structured output or a thinking budget".into());
    }
    Ok(count)
}

/// ``choices[0].logprobs`` for the answer's tokens: after the first
/// end-of-thinking token and dropped newlines. Upstream: `value`
/// (logprobs.zig:26) — the rows out of step, or a forced token without a
/// logprob, fail the reply.
pub fn value(
    text: &Text,
    thinking: bool,
    tokens: &[u32],
    rows: &[tensorfold_engine::LogprobRow],
) -> Result<Value, String> {
    if rows.len() < tokens.len() {
        return Err("the engine gave fewer logprob rows than reply tokens".into());
    }
    let mut start = 0;
    if thinking {
        let Some(end) = text.think_end else {
            return Err("logprobs: the end-of-thinking marker is not a token".into());
        };
        start = tokens
            .iter()
            .position(|t| *t == end)
            .map(|at| at + 1)
            .unwrap_or(tokens.len());
        // The think close's framing newlines drop
        while start < tokens.len() {
            let Some(bytes) = text.tokenizer.token_bytes(tokens[start]) else {
                break;
            };
            if bytes.is_empty() || bytes.iter().any(|b| *b != b'\n') {
                break;
            }
            start += 1;
        }
    }
    let mut content = Vec::with_capacity(tokens.len() - start);
    for (id, r) in tokens[start..].iter().zip(&rows[start..tokens.len()]) {
        if r.token != *id {
            return Err("the logprob rows are out of step with the reply's tokens".into());
        }
        if r.logprob.is_nan() {
            return Err("a forced reply token has no logprob".into());
        }
        let mut o = entry(text, *id, r.logprob);
        let top: Vec<Value> = r.ids[..r.count as usize]
            .iter()
            .zip(&r.logprobs[..r.count as usize])
            .map(|(alt, lp)| Value::Object(entry(text, *alt, *lp)))
            .collect();
        o.insert("top_logprobs".into(), Value::Array(top));
        content.push(Value::Object(o));
    }
    Ok(json!({"content": content}))
}

/// ``{token, logprob, bytes}``: the token's text, and its own bytes (a
/// partial UTF-8 sequence kept whole). Upstream: `entry` (logprobs.zig:50).
fn entry(text: &Text, id: u32, logprob: f32) -> Map<String, Value> {
    let bytes = text.tokenizer.token_bytes(id).unwrap_or_default();
    let mut o = Map::new();
    o.insert("token".into(), json!(text.tokenizer.decode(&[id])));
    o.insert("logprob".into(), json!(logprob));
    o.insert(
        "bytes".into(),
        Value::Array(bytes.iter().map(|b| json!(b)).collect()),
    );
    o
}


#[cfg(test)]
mod tests {
    use super::*;

    /// admit (logprobs.zig:15): the thinking marker's token, the budget.
    #[test]
    fn the_requests_admit_takes() {
        assert_eq!(admit(None, false, 0, 3), Ok(3));
        // Thinking on needs the end-of-thinking token
        assert_eq!(
            admit(None, true, 0, 3),
            Err("logprobs with thinking on need the template's end-of-thinking marker as one token".to_string())
        );
        assert_eq!(admit(Some(7), true, 0, 3), Ok(3));
        // A thinking budget refuses
        assert_eq!(
            admit(Some(7), true, 4, 3),
            Err("logprobs support nonstreamed chat without tools, stop strings, structured output or a thinking budget".to_string())
        );
    }
}
