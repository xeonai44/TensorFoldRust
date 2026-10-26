//! The chat template: `chat_template.jinja` rendered with the options the server's
//! requests name.
//!
//! Upstream ships its own Jinja-subset engine (`zig/src/core/template/`, ~4.6K
//! lines) for parity with HF's `apply_chat_template`. The port renders with
//! `minijinja` — the same reference-implementation argument as the `tokenizers`
//! crate (ADR-003): upstream's own engine exists only because the native binary
//! must load with no Python, and `minijinja` is a full Jinja2 implementation the
//! pinned checkpoint's 153-line template (macros, `namespace()`, `raise_exception`,
//! `tojson`) exercises without modification (ADR-015).

use minijinja::{context, Environment, ErrorKind, Value as JValue};
use serde_json::Value as Json;
use std::path::Path;

#[derive(Debug)]
pub enum TemplateError {
    /// The model folder ships no template.
    NoTemplate,
    /// The template failed to compile or render; the message is the template's
    /// own (a `raise_exception`, a bad name).
    Render(String),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemplateError::NoTemplate => {
                f.write_str("the model folder ships no chat template")
            }
            TemplateError::Render(message) => write!(f, "{}", message),
        }
    }
}

impl std::error::Error for TemplateError {}

/// The render options upstream's `model_text.RenderOptions` carries
/// (model_text.zig): the switches the chat template reads.
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderOptions<'a> {
    /// `tools`, the offered functions (None: none).
    pub tools: Option<&'a [Json]>,
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    /// `reasoning_effort`, with thinking on.
    pub reasoning_effort: Option<&'a str>,
}

/// One loaded chat template.
///
/// Upstream: `model_text.zig`'s Text — the template plus the tokenizer behind it.
/// The port keeps the tokenizer separate (`crate::Tokenizer`) and this type
/// renders to text; `render_ids` is the two together.
pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    /// The template's raw source (server.zig:318's needsUserAfterTool reads it).
    pub fn source(&self) -> &str {
        // The environment owns the template; its source is borrowed for the
        // environment's lifetime, which is the ChatTemplate's own.
        let env: &minijinja::Environment<'static> =
            unsafe { std::mem::transmute(&self.env) };
        env.get_template("chat")
            .map(|t| t.source().to_string())
            .unwrap_or_default()
            .leak()
    }

    /// Load `dir`'s `chat_template.jinja`, or the one
    /// `tokenizer_config.json` names. Upstream: model_text.zig's load path.
    pub fn open(dir: &Path) -> Result<ChatTemplate, TemplateError> {
        let mut source = std::fs::read_to_string(dir.join("chat_template.jinja"))
            .ok()
            .or_else(|| {
                // The chat_template may live inline in tokenizer_config.json.
                let config: Json =
                    serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json")).ok()?)
                        .ok()?;
                config
                    .get("chat_template")?
                    .as_str()?
                    .to_string()
                    .into()
            });
        if source.is_none() {
            // The pinned Qwen3.5 ships the template as a file; some folders keep
            // it in chat_template.jinja only.
            return Err(TemplateError::NoTemplate);
        }
        let source = patch_python_string_methods(&source.take().unwrap());
        let mut env = Environment::new();
        // HF's chat templates call raise_exception(); minijinja errors carry it
        // through as the message the template raised.
        env.add_global(
            "raise_exception",
            minijinja::value::Value::from_function(
                |message: Option<String>| -> Result<(), minijinja::Error> {
                    Err(minijinja::Error::new(
                        ErrorKind::TemplateNotFound,
                        message.unwrap_or_else(|| "the template refused this request".into()),
                    ))
                },
            ),
        );
        // The Python string methods the shim's filters stand in for (ADR-015).
        env.add_filter("lstrip", |value: minijinja::value::Value, chars: Option<String>| -> String {
            value
                .as_str()
                .unwrap_or_default()
                .trim_start_matches(&chars.unwrap_or_default())
                .to_string()
        });
        env.add_filter("rstrip", |value: minijinja::value::Value, chars: Option<String>| -> String {
            value
                .as_str()
                .unwrap_or_default()
                .trim_end_matches(&chars.unwrap_or_default())
                .to_string()
        });
        env.set_keep_trailing_newline(false);
        env.add_template_owned("chat", source)
            .map_err(|e| TemplateError::Render(e.to_string()))?;
        Ok(ChatTemplate { env })
    }

    /// Render `messages` with the options into template text.
    ///
    /// Upstream: `model_text.zig`'s render (the `renderIds` half adds the
    /// tokenizer; see `render_ids` below for the combined read).
    pub fn render(
        &self,
        messages: &[Json],
        options: RenderOptions<'_>,
    ) -> Result<String, TemplateError> {
        let messages = json_to_minijinja(&Json::Array(messages.to_vec()));
        let tools: JValue = match options.tools {
            Some(tools) => json_to_minijinja(&Json::Array(tools.to_vec())),
            None => JValue::UNDEFINED,
        };
        let ctx = context! {
            messages => messages,
            tools => tools,
            add_generation_prompt => options.add_generation_prompt,
            enable_thinking => options.enable_thinking,
            reasoning_effort => JValue::from(options.reasoning_effort.map(str::to_string)),
            add_vision_id => false,
        };
        let template = self
            .env
            .get_template("chat")
            .map_err(|e| TemplateError::Render(e.to_string()))?;
        template
            .render(ctx)
            .map_err(|e| TemplateError::Render(template_message(&e)))
    }
}


/// The Python string-method shim: minijinja has no `str.startswith`/`split`/
/// `lstrip` methods, so the ones the HF templates call become the equivalent
/// filters and tests at load time (ADR-015).
///
/// `.startswith(x)`/`.endswith(x)` become `is startingwith(x)`/`is endingwith(x)`
/// — the tests bind to the operand exactly where the method call stood, inside
/// the `and`/`not` the template wraps them in. `.split(x)`, `.lstrip(x)`,
/// `.rstrip(x)` become `|split(x)`/`|lstrip(x)`/`|rstrip(x)` with the operand
/// parenthesized around each call that an index follows, so the chain evaluates
/// in Python's order: `content.split(a)[0].rstrip(b)` becomes
/// `((content|split(a))[0]|rstrip(b))`.
fn patch_python_string_methods(source: &str) -> String {
    const FILTERS: [&str; 3] = ["split", "lstrip", "rstrip"];
    let src: Vec<char> = source.chars().collect();
    let mut out: Vec<char> = Vec::with_capacity(src.len() + 64);
    let mut i: usize = 0;
    while i < src.len() {
        let c = src[i];
        // String literals pass whole, so `.split(` inside quotes stays.
        if c == '\'' || c == '"' {
            copy_literal(&src, &mut i, &mut out);
            continue;
        }
        let tail: String = src[i..].iter().take(12).collect();
        if tail.starts_with(".startswith(") {
            out.extend(" is startingwith(".chars());
            i += ".startswith(".len();
            continue;
        }
        if tail.starts_with(".endswith(") {
            out.extend(" is endingwith(".chars());
            i += ".endswith(".len();
            continue;
        }
        let mut matched: Option<(usize, &str)> = None;
        for name in FILTERS {
            let pattern = format!(".{}(", name);
            if tail.starts_with(&pattern) {
                matched = Some((pattern.len(), name));
                break;
            }
        }
        if let Some((pattern_len, name)) = matched {
            // Wrap the operand: everything the chain emitted before this call.
            let operand_start = scan_operand_start(&out);
            out.insert(operand_start, '(');
            out.push('|');
            out.extend(name.chars());
            // Leave `i` at the call's opening paren; copy_balanced consumes it
            // and its matching close.
            i += pattern_len - 1;
            // The call's arguments pass through (string-aware, balanced); the
            // wrap closes right after them, so a following index applies to the
            // filtered result — Python's chain order: `(operand|split(a))[idx]`.
            copy_balanced(&src, &mut i, &mut out, '(', ')');
            out.push(')');
            let mut j = i;
            while j < src.len() && src[j] == ' ' {
                j += 1;
            }
            if j < src.len() && src[j] == '[' {
                i = j;
                copy_balanced(&src, &mut i, &mut out, '[', ']');
            }
            continue;
        }
        out.push(c);
        i += 1;
    }
    out.iter().collect()
}

/// Where the operand of a postfix chain starts in the already-emitted text,
/// scanning back over identifiers, dots and balanced groups
/// (`content`, `message.content`, `(...)[0]`, `a|trim` results).
fn scan_operand_start(out: &[char]) -> usize {
    let mut at = out.len();
    while at > 0 {
        let prev = out[at - 1];
        if prev == ')' || prev == ']' {
            let (open, close) = if prev == ')' { ('(', ')') } else { ('[', ']') };
            let mut depth = 0;
            while at > 0 {
                at -= 1;
                if out[at] == close {
                    depth += 1;
                } else if out[at] == open {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
            }
        } else if prev.is_alphanumeric() || prev == '_' || prev == '.' {
            at -= 1;
        } else {
            break;
        }
    }
    at
}

/// Copy a string literal (quote-aware) from `src[at..]` into `out`, advancing `at`.
fn copy_literal(src: &[char], at: &mut usize, out: &mut Vec<char>) {
    let quote = src[*at];
    out.push(quote);
    *at += 1;
    while *at < src.len() {
        out.push(src[*at]);
        if src[*at] == '\\' && *at + 1 < src.len() {
            out.push(src[*at + 1]);
            *at += 2;
            continue;
        }
        if src[*at] == quote {
            *at += 1;
            return;
        }
        *at += 1;
    }
}

/// Copy from `src[at..]` through a balanced `open`...`close` region (string-aware),
/// starting at the opening character, ending past the closing one.
fn copy_balanced(src: &[char], at: &mut usize, out: &mut Vec<char>, open: char, close: char) {
    debug_assert_eq!(src[*at], open);
    out.push(src[*at]);
    *at += 1;
    let mut depth = 1;
    while *at < src.len() && depth > 0 {
        let c = src[*at];
        if c == '\'' || c == '"' {
            copy_literal(src, at, out);
            continue;
        }
        out.push(c);
        if c == open {
            depth += 1;
        } else if c == close {
            depth -= 1;
        }
        *at += 1;
    }
}

/// The rendered prompt as token ids: the template's text through the tokenizer.
///
/// Upstream: `renderIds` (model_text.zig) — one function in the server's Text;
/// the port composes the two pieces.
pub fn render_ids(
    template: &ChatTemplate,
    tokenizer: &crate::Tokenizer,
    messages: &[Json],
    options: RenderOptions<'_>,
) -> Result<Vec<u32>, TemplateError> {
    let text = template.render(messages, options)?;
    Ok(tokenizer.encode(&text))
}

fn template_message(error: &minijinja::Error) -> String {
    match error.kind() {
        ErrorKind::TemplateNotFound => error
            .to_string()
            .trim_start_matches("template not found: ")
            .to_string(),
        _ => error.to_string(),
    }
}

/// serde_json's value into minijinja's, keeping the numbers and objects the
/// template's `tojson` filter and attribute tests read.
fn json_to_minijinja(value: &Json) -> JValue {
    match value {
        Json::Null => JValue::UNDEFINED,
        Json::Bool(b) => (*b).into(),
        Json::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into()
            } else if let Some(u) = n.as_u64() {
                (u as i64).into()
            } else {
                n.as_f64().unwrap_or(0.0).into()
            }
        }
        Json::String(s) => s.as_str().into(),
        Json::Array(a) => a.iter().map(json_to_minijinja).collect::<Vec<_>>().into(),
        Json::Object(o) => {
            let map: std::collections::BTreeMap<String, JValue> = o
                .iter()
                .map(|(k, v)| (k.clone(), json_to_minijinja(v)))
                .collect();
            JValue::from_serialize(&map)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_template(name: &str, source: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("tf-template-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("chat_template.jinja"), source).unwrap();
        dir
    }

    /// A Qwen-shaped template through the pinned options: the generation prompt
    /// and the thinking switch, exactly the fields upstream's model_text reads.
    #[test]
    fn renders_the_generation_prompt_and_the_thinking_switch() {
        let dir = temp_template(
            "qwen",
            r#"{%- for message in messages %}{{- '<|im_start|>' + message.role + '\n' + (message.content | trim) + '<|im_end|>\n' }}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- if enable_thinking %}{{- 'think\n' }}{%- else %}{{- '\n\n' }}{%- endif %}{%- endif %}"#,
        );
        let template = ChatTemplate::open(&dir).unwrap();
        let messages = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "hello"}),
        ];
        let on = template
            .render(
                &messages,
                RenderOptions { add_generation_prompt: true, enable_thinking: true, ..RenderOptions::default() },
            )
            .unwrap();
        // The thinking tag's own newline is the literal's, kept past the
        // `{%-` blocks (which strip the whitespace before tags, not after).
        assert!(
            on.ends_with("<|im_start|>assistant\nthink\n"),
            "thinking on: {:?}",
            on
        );
        let without = template
            .render(
                &messages,
                RenderOptions { add_generation_prompt: false, ..RenderOptions::default() },
            )
            .unwrap();
        assert_eq!(
            without,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The shim's chain rewrite on the pinned template's hardest line.
    #[test]
    fn dump_patched_line() {
        let dir = std::path::PathBuf::from("/Volumes/llm/models/Qwen3.5-2B-MLX-4bit");
        if !dir.join("chat_template.jinja").is_file() {
            eprintln!("skipping");
            return;
        }
        let source = std::fs::read_to_string(dir.join("chat_template.jinja")).unwrap();
        let patched = patch_python_string_methods(&source);
        for (n, line) in patched.lines().enumerate() {
            if line.contains("rstrip") || line.contains("lstrip") {
                eprintln!("{}: {:?}", n + 1, line);
            }
        }
    }

    /// `raise_exception` surfaces the template's own refusal.
    #[test]
    fn a_templates_refusal_is_its_message() {
        let dir = temp_template("refusal", "{{- raise_exception('No messages provided.') -}}");
        let template = ChatTemplate::open(&dir).unwrap();
        let err = template.render(&[], RenderOptions::default()).unwrap_err();
        assert!(err.to_string().contains("No messages provided."));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The pinned checkpoint's own template loads and renders: the 153-line
    /// Qwen3.5 one with macros, namespace() and tojson.
    #[test]
    fn the_pinned_templates_load() {
        let dir = std::path::PathBuf::from("/Volumes/llm/models/Qwen3.5-2B-MLX-4bit");
        if !dir.join("chat_template.jinja").is_file() {
            eprintln!("skipping: no model at {}", dir.display());
            return;
        }
        let template = ChatTemplate::open(&dir).unwrap();
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let text = template
            .render(
                &messages,
                RenderOptions { add_generation_prompt: true, enable_thinking: false, ..RenderOptions::default() },
            )
            .unwrap();
        assert!(text.contains("hi"), "{}", text);
        assert!(text.contains("<|im_start|>assistant"), "{}", text);
    }
}
