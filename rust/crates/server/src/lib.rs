//! tensorfold-server: the HTTP surface over the engine contract.
//!
//! Port of `zig/src/server/`'s skeleton. Upstream is a hand-rolled HTTP server on raw
//! POSIX sockets with one thread a connection (`server.zig:217  fn serve`); this port
//! uses axum/tokio for the same routes (decision record: docs/DECISIONS.md, ADR-004).
//! The engine itself stays synchronous — engines run their own thread like upstream's
//! `lane_host.zig`, and the async layer bridges to it; that bridge lands with the first
//! engine (Phase 1, tracked in docs/PORTING-PLAN.md).
//!
//! Ported here: the server `Config` (server.zig:47), bearer auth (auth.zig), model
//! discovery (`/v1/models`, the README's "Model discovery" section) and the
//! engine-less refusals (none.zig's message). Not yet ported: the chat/completions
//! translation layers (chat.zig, anthropic_translate.zig), compaction (compact*.zig),
//! Sliding Weights (slide*.zig), the dashboard and metrics — Phase 2 and Phase 3, each
//! mapped in docs/ZIG-TO-RUST-MAP.md.

pub mod anthropic;
pub mod anthropic_translate;
pub mod auth;
pub mod chat;
pub mod decisions;
pub mod chunk_plan;
pub mod compact;
pub mod completions;
pub mod compact_mem;
pub mod errors;
pub mod fields;
pub mod grammar;
pub mod live;
pub mod process_memory;
pub mod request_log;
pub mod memory_routes;
pub mod ids;
pub mod json_parse;
pub mod json_write;
pub mod logprobs;
pub mod pyrepr;
pub mod responses;
pub mod responses_translate;
pub mod tool_parse;
pub mod tokens;
pub mod tool_params;
pub mod tool_stream;
pub mod tool_specs;
pub mod messages;
pub mod metrics;
pub mod reply_text;
pub mod routes;
pub mod warm;
pub mod state;

pub use auth::Auth;
pub use metrics::Metrics;
pub use state::{AppState, Text};

use std::net::SocketAddr;

/// auto compacts near the window. fraction compacts past that share of it.
///
/// Upstream: `server.zig:46  pub const CompactAt = union(enum) { auto, fraction: f64 }`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CompactAt {
    Auto,
    Fraction(f64),
}

/// The server's fixed configuration, built from the serve flags once at startup.
///
/// Upstream: `server.zig:47  pub const Config = struct { ... }` — every field maps,
/// with the Zig default where the table names one.
#[derive(Clone, Debug)]
pub struct Config {
    pub served_name: String,
    /// The served name first, then aliases, without repeats.
    pub model_ids: Vec<String>,
    pub default_max_tokens: i64,
    pub enable_thinking: bool,
    pub reasoning_effort: Option<String>,
    pub thinking_budget: i64,
    pub loop_guard: bool,
    /// Seconds the idle keepalive runs after the last request ends (0: off).
    pub keep_warm_s: i64,
    pub dashboard: bool,
    /// Sampling when a request names none (generation_config.json and the serve
    /// flags), or None for greedy. Kept as JSON, the way upstream keeps its `Value`.
    pub default_sampling: Option<serde_json::Value>,
    pub use_drafts: bool,
    pub seed_salt: i64,
    /// Where chat and completion bodies are appended; None: nowhere.
    pub request_log: Option<std::path::PathBuf>,
    /// None keeps every reply identical to a server that has no compaction.
    pub compact_at: Option<CompactAt>,
    /// Tokens of the recent tail kept whole. None uses min(20000, a quarter of the
    /// window).
    pub compact_keep: Option<u32>,
    /// Directory of the stored note, or None when notes are not stored.
    pub compact_memory: Option<std::path::PathBuf>,
    /// The Sliding Weights fact-graph file; None keeps the graph in memory.
    pub slide_graph: Option<std::path::PathBuf>,
    /// Allow /metrics without a key when API authentication is enabled.
    pub metrics_open: bool,
}

impl Config {
    /// The defaults upstream's flag table applies (4096 reply tokens, thinking on,
    /// 900s keep-warm, drafts on).
    pub fn new(served_name: String) -> Config {
        Config {
            served_name: served_name.clone(),
            model_ids: vec![served_name],
            default_max_tokens: 4096,
            enable_thinking: true,
            reasoning_effort: None,
            thinking_budget: 0,
            loop_guard: false,
            keep_warm_s: 900,
            dashboard: false,
            default_sampling: None,
            use_drafts: true,
            seed_salt: 0,
            request_log: None,
            compact_at: None,
            compact_keep: None,
            compact_memory: None,
            slide_graph: None,
            metrics_open: false,
        }
    }
}

/// Serve until shutdown. The listener binds `addr`; `on_listen` hears the port the
/// way upstream's `serve.zig:73  on_listen` does (tests read it).
pub async fn serve(
    state: std::sync::Arc<AppState>,
    addr: SocketAddr,
    on_listen: Option<Box<dyn FnOnce(u16) + Send>>,
) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    if let Some(on_listen) = on_listen {
        on_listen(listener.local_addr()?.port());
    }
    let app = routes::router(state);
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
}
