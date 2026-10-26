//! The server's shared state: its config, what its engine reported, and its auth.
//!
//! Phase 0 runs engine-less, the shape upstream's `zig/tests/server/build.sh` parity
//! suite uses: routes answer from the config alone and inference routes refuse with
//! the engine-less message. When an engine opens (Phase 1) its `Info` moves here and
//! the translation layers grow against it, exactly like the Zig server's
//! `serve.zig:124  Server.init` path.

use crate::auth::Auth;
use crate::Config;
use std::sync::Arc;
use tensorfold_engine::{Engine, Info};

/// The model text a request's messages pass through: the tokenizer, the chat
/// template and the stop tokens (upstream's `model_text.zig` Text).
pub struct Text {
    pub tokenizer: tensorfold_core::Tokenizer,
    pub template: tensorfold_core::ChatTemplate,
    /// The eos token ids the engine ends replies on (the checkpoint's
    /// generation_config.json / config.json eos_token_id).
    pub eos: Vec<u32>,
    /// The think markers this template's replies use (server.zig:89: the
    /// tokenizer knows the channel close, else the think markers).
    pub markers: crate::reply_text::Markers,
    /// The think block's close token id, for the reasoning count.
    pub think_end: Option<u32>,
    /// Whether the template refuses a conversation with no user query
    /// (server.zig:318 needsUserAfterTool: "No user query found").
    pub needs_user_after_tool: bool,
    /// The efforts this chat template names in quotes (server.zig:102's
    /// effortLevels): the levels a request's effort coerces against.
    pub effort_levels: Vec<String>,
    /// The tokens that close a think block — the newline, the end token,
    /// the two that follow it (server.zig:105-110) — what a thinking budget
    /// forces and the loop guard watches; empty without the end token.
    pub think_close: Vec<u32>,
    /// The think close's own end token.
    pub think_close_end: Option<u32>,
    /// ``template_late_system`` (server.zig:305): "system" when the
    /// template keeps a later system message in place, else "user" — the
    /// role the compaction note assembles under.
    pub late_system: &'static str,
}

impl Text {
    /// Upstream server.zig:89-104's one-time reads: the markers by the
    /// tokenizer's knowledge of the channel close, and the template's user-query
    /// demand from its source.
    pub fn read(
        tokenizer: tensorfold_core::Tokenizer,
        template: tensorfold_core::ChatTemplate,
        eos: Vec<u32>,
    ) -> Text {
        let markers = if tokenizer.special_token_id(crate::reply_text::CHANNEL_MARKERS.close).is_some() {
            crate::reply_text::CHANNEL_MARKERS
        } else {
            crate::reply_text::THINK_MARKERS
        };
        let think_end = tokenizer.special_token_id(markers.close);
        Text {
            needs_user_after_tool: template
                .source()
                .contains("No user query found"),
            effort_levels: crate::fields::effort_levels(template.source())
                .into_iter()
                .map(str::to_string)
                .collect(),
            // The think close the plain marker closes (server.zig:104-110):
            // the newline before, the end token, the two after
            think_close: match tokenizer
                .special_token_id(crate::reply_text::THINK_MARKERS.close)
            {
                Some(end) => {
                    let mut close = tokenizer.encode("\n");
                    close.push(end);
                    close.extend(tokenizer.encode("\n\n"));
                    close
                }
                None => Vec::new(),
            },
            think_close_end: tokenizer
                .special_token_id(crate::reply_text::THINK_MARKERS.close),
            late_system: late_system(&tokenizer, &template),
            markers,
            think_end,
            tokenizer,
            template,
            eos,
        }
    }
}

/// ``template_late_system`` (server.zig:305-310): a probe conversation with
/// a system message after the first exchange renders its text — the
/// template keeps late systems in place; one that drops them has the note
/// go as a user message.
fn late_system(
    tokenizer: &tensorfold_core::Tokenizer,
    template: &tensorfold_core::ChatTemplate,
) -> &'static str {
    let probe_text = "tensorfold-late-system-probe";
    let probe: Vec<serde_json::Value> = vec![
        serde_json::json!({"role": "system", "content": "s"}),
        serde_json::json!({"role": "user", "content": "u"}),
        serde_json::json!({"role": "assistant", "content": "a"}),
        serde_json::json!({"role": "system", "content": probe_text}),
        serde_json::json!({"role": "user", "content": "v"}),
    ];
    let ids = tensorfold_core::render_ids(
        template,
        tokenizer,
        &probe,
        tensorfold_core::RenderOptions {
            add_generation_prompt: false,
            enable_thinking: false,
            reasoning_effort: None,
            tools: None,
        },
    )
    .unwrap_or_default();
    if tokenizer.decode(&ids).contains(probe_text) {
        "system"
    } else {
        "user"
    }
}

pub struct AppState {
    pub config: Config,
    /// What the engine reported at open; None while engine-less.
    pub engine_info: Option<Info>,
    pub auth: Auth,
    /// The message the engine-less build refuses inference with (none.zig's words).
    pub engineless_problem: String,
    /// The engine the chat route drives (the LaneHost behind the contract).
    pub engine: Option<Arc<dyn Engine>>,
    /// The model's text (tokenizer, template, eos); None while engine-less.
    pub text: Option<Arc<Text>>,
    /// The request counters and histograms the /metrics scrape reads.
    pub metrics: crate::metrics::Metrics,
    /// Where a prompt's prefill chunks start (server.zig:113's srv.chunks):
    /// the message markers read once at open when the engine prefill-steps.
    pub chunk_plan: crate::chunk_plan::Plan,
    /// The newest stored responses, for GET, DELETE and
    /// ``previous_response_id`` (responses.zig:44's srv.store).
    pub responses: crate::responses::Store,
    /// The engine's request ids, one a request (the cancel queue keys
    /// them; upstream's ``input.id`` carries the reply's own).
    next_request_id: std::sync::atomic::AtomicU64,
}

impl AppState {
    /// One request's engine id (the ids start past 0: the engine's own
    /// streams number from there).
    pub fn next_request_id(&self) -> u64 {
        self.next_request_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 2
    }

    pub fn new(config: Config) -> AppState {
        // The none-backend message, the engine-less build's refusal (none.zig:14).
        let engineless_problem = "the native engine has no backend for this checkpoint yet; the Python engine 0.6.6 may serve them: python -m pip install git+https://github.com/ashhart/TensorFold.git@v0.6.6".to_string();
        AppState {
            config,
            engine_info: None,
            auth: Auth::open(),
            engineless_problem,
            engine: None,
            text: None,
            metrics: crate::metrics::Metrics::new(),
            chunk_plan: crate::chunk_plan::Plan::default(),
            responses: crate::responses::Store::default(),
            next_request_id: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn with_auth(mut self, auth: Auth) -> AppState {
        self.auth = auth;
        self
    }

    pub fn with_engine_info(mut self, info: Info) -> AppState {
        self.engine_info = Some(info);
        self
    }

    /// Wire the engine and its model text (the serve path's open): the
    /// prompt-cache plan read here too, when the engine prefill-steps
    /// (server.zig:113).
    pub fn with_engine(mut self, engine: Arc<dyn Engine>, text: Arc<Text>) -> AppState {
        let info = engine.info();
        self.chunk_plan = if info.prefill_step > 0 {
            crate::chunk_plan::markers(&text, info.prefill_step)
        } else {
            crate::chunk_plan::Plan::default()
        };
        self.engine_info = Some(info);
        self.engine = Some(engine);
        self.text = Some(text);
        self
    }

    /// The model ids a client may name: the served name first, then aliases
    /// (server.zig's `model_ids`).
    pub fn model_ids(&self) -> &[String] {
        &self.config.model_ids
    }
}
