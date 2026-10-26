//! `tensorfold-native serve MODEL [flags]`: the Phase 0 path.
//!
//! Port of `zig/src/server/serve.zig`'s front half: the key store, the model ids, the
//! config the server runs with, and the engine open that decides whether serving starts
//! at all. The engine-less build refuses at `open` with the none-backend message, the
//! way the Zig binary does; the HTTP loop itself is exercised by the server crate's
//! tests until a real engine lands in Phase 1.

use crate::hub;
use clap::Args;
use tensorfold_core::{ModelConfig, Tokenizer};
use tensorfold_server::{AppState, Auth, Config};
use serde_json::Value;
use std::path::PathBuf;

/// The serve flags Phase 0 honours, one table entry a flag, the same spellings the Zig
/// `server/cli.zig` table carries. The rest are refused as unsupported, upstream's rule
/// for a flag a build does not take (`cli.zig:18`).
#[derive(Args, Debug)]
pub struct ServeFlags {
    /// Listen address, default 127.0.0.1.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    /// Listen port, default 8080.
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
    /// Model ID advertised to clients.
    #[arg(long)]
    pub name: Option<String>,
    /// Another accepted model ID; repeat for multiple aliases.
    #[arg(long = "alias")]
    pub aliases: Vec<String>,
    /// Require a bearer key; repeat for multiple keys.
    #[arg(long = "api-key")]
    pub api_keys: Vec<String>,
    /// Read keys from a restricted-permission file.
    #[arg(long)]
    pub api_key_file: Option<PathBuf>,
    /// Allow /metrics without a key when API authentication is enabled.
    #[arg(long)]
    pub metrics_open: bool,
    /// Bound prompt plus reply tokens; the model's window and memory checks still
    /// apply.
    #[arg(long)]
    pub context: Option<i64>,
    /// The GPU ordinal to serve from (the CUDA build's own).
    #[cfg(target_os = "linux")]
    #[arg(long)]
    pub device: Option<u32>,
    /// Whole prompt chunks a call runs as staggered segments.
    #[cfg(target_os = "linux")]
    #[arg(long)]
    pub segments: Option<u32>,
    /// Default reply limit; requests can override it.
    #[arg(long, default_value_t = 4096)]
    pub max_tokens: i64,
    /// Sampling temperature; zero requests greedy decoding.
    #[arg(long)]
    pub temperature: Option<f64>,
    #[arg(long)]
    pub top_p: Option<f64>,
    #[arg(long)]
    pub top_k: Option<u32>,
    #[arg(long)]
    pub min_p: Option<f64>,
    /// Choose whether the chat template opens a reasoning block (the default on).
    #[arg(long, conflicts_with = "no_thinking")]
    pub thinking: bool,
    #[arg(long = "no-thinking")]
    pub no_thinking: bool,
    /// Default template effort: low, medium, high or xhigh.
    #[arg(long, value_parser = ["low", "medium", "high", "xhigh"])]
    pub reasoning_effort: Option<String>,
    /// Limit reasoning tokens where the engine supports closing the reasoning block.
    #[arg(long, default_value_t = 0)]
    pub thinking_budget: i64,
    /// Close a short repeated reasoning cycle where the engine supports it.
    #[arg(long)]
    pub loop_guard: bool,
    /// Produce the plain reference through the same engine.
    #[arg(long = "no-drafts")]
    pub no_drafts: bool,
    /// Admit up to N requests where the engine shares lanes; `auto` is the default.
    #[arg(long)]
    pub parallel: Option<String>,
    /// Enable the local /dashboard page and /stats endpoint.
    #[arg(long)]
    pub dashboard: bool,
    /// Keep the GPU active while idle for this long after a request (seconds).
    #[arg(long)]
    pub keep_warm: Option<i64>,
    /// Auto-compact near the context window, or past this fraction of it in
    /// (0, 1].
    #[arg(long = "compact-at")]
    pub compact_at: Option<String>,
    /// The recent tail compaction keeps whole (tokens).
    #[arg(long = "compact-keep")]
    pub compact_keep: Option<u32>,
    /// The directory the compaction note is stored under.
    #[arg(long = "compact-memory")]
    pub compact_memory: Option<std::path::PathBuf>,
    /// Kept prompt states' budget (0: none; unset: what 70% of RAM leaves
    /// past the loaded server).
    #[arg(long = "prompt-cache-gib")]
    pub prompt_cache_gib: Option<f64>,
    /// A --prompt-cache-gib past the 70% cap is kept, not refused.
    #[arg(long = "prompt-cache-over-cap")]
    pub prompt_cache_over_cap: bool,
}

impl ServeFlags {
    /// Upstream cli.zig's parallel rule: `auto` gives the engine its default lanes and
    /// a number fixes them (`lanes_fixed`).
    pub(crate) fn lanes(&self) -> (u32, bool) {
        match self.parallel.as_deref() {
            None | Some("auto") => (8, false),
            Some(text) => match text.parse::<u32>() {
                Ok(n) => (n, true),
                Err(_) => (8, false),
            },
        }
    }

    fn enable_thinking(&self) -> bool {
        !self.no_thinking
    }
}

/// The Phase 0 serve path: everything up to the engine open, then the open's refusal.
pub fn run(model: &str, flags: &ServeFlags) -> i32 {
    // MODEL resolves to a local directory, else a Hugging Face cache entry
    // by repo id (hub.zig:23 resolve — the same resolution pull fills).
    let dir = match hub::resolve(model) {
        Ok(dir) => dir,
        Err(problem) => {
            eprintln!("tensorfold: {}", problem);
            return 1;
        }
    };

    // The checkpoint's config gates the family (engines.zig reads model_type).
    let config = match ModelConfig::open(&dir) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("tensorfold: {}", err);
            return 1;
        }
    };

    // The keys (auth.Store in serve.zig:85): the flags, the
    // environment, and the file — the file live in the store, reread on
    // its change or SIGHUP.
    let environment = std::env::var("TENSORFOLD_API_KEY").unwrap_or_default();
    let auth = match Auth::init(
        &flags.api_keys,
        &environment,
        flags.api_key_file.as_deref(),
        flags.metrics_open,
    ) {
        Ok(auth) => auth,
        Err(problem) => {
            eprintln!("tensorfold: {problem}");
            return 1;
        }
    };
    if !auth.enabled() && !is_loopback(&flags.host) {
        eprintln!("warning: non-loopback server has no API key; requests are open");
    }
    // SIGHUP rereads the key file (auth.zig:16's installHangup)
    install_hangup();

    // The ids: the served name first, then aliases, trimmed, without repeats
    // (serve.zig:94).
    let served = hub::served_name(flags.name.as_deref().unwrap_or(""), model, &dir.to_string_lossy());
    let mut model_ids = vec![served.trim().to_string()];
    for alias in &flags.aliases {
        let id = alias.trim();
        if id.is_empty() || model_ids.iter().any(|seen| seen == id) {
            continue;
        }
        model_ids.push(id.to_string());
    }

    let seed_salt = match std::env::var("TENSORFOLD_SEED_SALT") {
        Ok(text) => match text.trim().parse::<i64>() {
            Ok(salt) => salt,
            Err(_) => {
                eprintln!("tensorfold: TENSORFOLD_SEED_SALT={}: an integer", text);
                return 1;
            }
        },
        Err(_) => 0,
    };

    // The server config (serve.zig:111's fields, the ones Phase 0 keeps).
    let mut server_config = Config::new(model_ids[0].clone());
    server_config.model_ids = model_ids;
    server_config.default_max_tokens = flags.max_tokens;
    server_config.enable_thinking = flags.enable_thinking();
    server_config.reasoning_effort = flags.reasoning_effort.clone();
    server_config.thinking_budget = flags.thinking_budget;
    server_config.loop_guard = flags.loop_guard;
    server_config.keep_warm_s = flags.keep_warm.unwrap_or(900);
    server_config.use_drafts = !flags.no_drafts;
    server_config.seed_salt = seed_salt;
    server_config.dashboard = flags.dashboard;
    // The compaction flags (serve.zig:119-121): the fraction in (0, 1]
    if let Some(at) = &flags.compact_at {
        if at == "auto" {
            server_config.compact_at = Some(tensorfold_server::CompactAt::Auto);
        } else {
            match at.parse::<f64>() {
                Ok(f) if f > 0.0 && f <= 1.0 => {
                    server_config.compact_at = Some(tensorfold_server::CompactAt::Fraction(f));
                }
                _ => {
                    eprintln!("argument --compact-at: expected auto or a fraction in (0, 1]: '{at}'");
                    std::process::exit(2);
                }
            }
        }
    }
    server_config.compact_keep = flags.compact_keep;
    server_config.compact_memory = flags.compact_memory.clone();
    server_config.metrics_open = flags.metrics_open;
    server_config.default_sampling = default_sampling(flags, &dir);
    if let Ok(log_path) = std::env::var("TENSORFOLD_REQUEST_LOG") {
        server_config.request_log = Some(PathBuf::from(log_path));
    }

    // The engine open (serve.zig's engines.open): the family the config
    // names. The port's families: Qwen3.5-2B on Metal (macOS) and
    // Nemotron-H on CUDA (Linux) — other model types refuse with the
    // none-backend message, upstream's rule for a model no build
    // serves. Upstream's native registry is target-selected.
    #[cfg(target_os = "macos")]
    let metal_family = config.model_type == "qwen3_5";
    #[cfg(not(target_os = "macos"))]
    let metal_family = false;
    #[cfg(target_os = "linux")]
    let cuda_family = config.model_type == "nemotron_h";
    #[cfg(not(target_os = "linux"))]
    let cuda_family = false;
    if !metal_family && !cuda_family {
        eprintln!(
            "tensorfold: the native engine has no backend for {} checkpoints yet; the Python engine 0.6.6 may serve them: python -m pip install git+https://github.com/ashhart/TensorFold.git@v0.6.6",
            config.model_type
        );
        return 1;
    }
    // The CUDA family path — Linux alone (native.backends is
    // target-selected), the nemotron_h host (native/cuda.zig's
    // openWith): the memory plan, the family open, the lane host.
    #[cfg(target_os = "linux")]
    if cuda_family {
        return serve_linux_cuda(
            flags,
            &dir,
            &config,
            server_config,
            auth,
        );
    }

    // The Metal family path — macOS alone (native.backends is
    // target_selected); other Linux builds refused at the gate above
    #[cfg(target_os = "macos")]
    {
        // The text (serve.zig's `text`): the tokenizer, the chat template and the
        // stop tokens from the checkpoint's config.
        let tokenizer = match Tokenizer::open(&dir) {
            Ok(t) => t,
            Err(err) => {
                eprintln!("tensorfold: {}", err);
                return 1;
            }
        };
        let template = match tensorfold_core::ChatTemplate::open(&dir) {
            Ok(t) => t,
            Err(err) => {
                eprintln!("tensorfold: {}", err);
                return 1;
            }
        };
        // The eos token ids, upstream's exact rule (hf_text.zig:126 eosIds): the
        // model config's eos_token_id — the root's, else text_config's — plus the
        // tokenizer config's eos_token resolved through the special tokens.
        let eos: Vec<u32> = eos_ids(&config.json, &dir);

        // The family's model, backend and lane host (native/qwen35.zig's open
        // path: model load, the Metal backend with its scratch and qualify
        // check, then the lane core served by one thread). Upstream prints
        // nothing while loading — the load time rides the serving line.
        let load_started = std::time::Instant::now();
        let model = match tensorfold_qwen35::model::Model::load(&dir) {
            Ok(m) => std::sync::Arc::new(m),
            Err(err) => {
                eprintln!("tensorfold: {}", err);
                return 1;
            }
        };
        let lanes = flags.lanes().0.max(1);
        // The window upstream serves under (native/qwen35.zig:38-42): the
        // flag or the checkpoint's native one, refused outside its range —
        // the capacity the backend allocates and the window the Info reports
        // are the same one
        let native_window: i64 = model.config.context as i64;
        let window: i64 = flags.context.unwrap_or(native_window);
        if window < 1 || window > native_window {
            eprintln!("tensorfold: --context must fit the Qwen checkpoint's positive context window");
            return 1;
        }
        let options =
            tensorfold_qwen35::backend::Options::new(window as usize).with_streams(lanes);
        let metal = match tensorfold_qwen35::backend::Metal::new(
            std::sync::Arc::clone(&model),
            options,
        ) {
            Ok(m) => m,
            Err(err) => {
                eprintln!("tensorfold: {}", err);
                return 1;
            }
        };
        // The lane config from the backend's facts (Python's _family_setup) and the
        // engine thread (LaneHost) behind the contract. The config, backend and
        // clock live for the process: the serving loop holds them until it exits, so
        // the core's 'static borrow is the host's own lifetime.
        let facts = metal.facts();
        let core_config = match tensorfold_lanes::config::Config::new(&facts, 32, 31) {
            Ok(c) => c,
            Err(err) => {
                eprintln!("tensorfold: {}", err);
                return 1;
            }
        };
        let context_window = window as u32;
        let core_config: &'static tensorfold_lanes::config::Config = Box::leak(Box::new(core_config));
        let metal: &'static mut tensorfold_qwen35::backend::Metal =
            Box::leak(Box::new(metal));
        let clock: &'static mut tensorfold_lanes::WallClock = Box::leak(Box::new(
            tensorfold_lanes::WallClock::default(),
        ));
        // The prefill step this family serves under (qwen35.zig:52): the
        // backend's own 128-row chunk, the step the chunk plan floors to —
        // the device-based rule belongs to the nemotron path, not this one
        let step = tensorfold_qwen35::backend::Options::DEFAULT_CHUNK;
        let core = tensorfold_lanes::engine::Engine::init(core_config, metal, clock);
        // No prompt cache for this family (native/qwen35.zig — 61 lines, arms
        // none): upstream's qwen3_5 host serves with no retained states — a
        // rerun's cached_tokens stays 0 and /v1/memory carries no plan. The
        // store, the snapshots and the budget plan (prompt_cache.rs,
        // snapshot.rs, cache_fit.rs) are the ported machinery for the family
        // that arms one (upstream's nemotron host, metal.zig:154-166);
        // --prompt-cache-gib parses, exactly as upstream's cli does, and
        // stays inert here.
        let info = tensorfold_engine::Info {
            // The Info the server reads: the window under the flag, the lane
            // bound from --parallel, the prefill step.
            context_window,
            lanes,
            prefill_step: step as u32,
            ..tensorfold_engine::Info::default()
        };
        let context_window = info.context_window;
        let host = tensorfold_lanes::LaneHost::new(core, info).start();
        let engine: std::sync::Arc<dyn tensorfold_engine::Engine> = host.clone();

        // The server state with the engine and its text wired (serve.zig:124's
        // Server.init).
        let text = std::sync::Arc::new(tensorfold_server::Text::read(
            tokenizer, template, eos,
        ));
        let shown_sampling = shown_sampling(server_config.default_sampling.as_ref());
        let mut state = AppState::new(server_config);
        state = state.with_engine(engine, text);
        state = state.with_auth(auth);
        let state = std::sync::Arc::new(state);
        // The chunk-plan line (server.zig:114), upstream's own ``[tensorfold]``
        // log on stdout: the step the plan cut at and the least it cuts.
        if step > 0 {
            println!(
                "[tensorfold] prompt chunks of up to {} tokens, cut at replies {}+ tokens apart",
                commas(step as u64),
                state.chunk_plan.min_chunk,
            );
        }

        // The HTTP server: tokio's runtime, the listener, the routes (ADR-004).
        let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
            Ok(rt) => rt,
            Err(err) => {
                eprintln!("tensorfold: {}", err);
                return 1;
            }
        };
        runtime.block_on(async move {
            let addr: std::net::SocketAddr = match format!("{}:{}", flags.host, flags.port).parse() {
                Ok(a) => a,
                Err(_) => {
                    eprintln!("tensorfold: cannot listen on {}:{}", flags.host, flags.port);
                    return 1;
                }
            };
            let shown_name = state.config.model_ids[0].clone();
            let shown_host = flags.host.clone();
            let shown_drafts = if flags.no_drafts { "off" } else { "on" };
            let loaded = load_started.elapsed().as_secs_f64();
            let on_listen = Some(Box::new(move |port: u16| {
                // serve.zig:150's serving line, upstream's format once the
                // listener is up
                println!(
                    "[tensorfold] serving {} at http://{}:{}/v1 (sampling: {}; drafts: {}; context: {}; loaded in {:.1}s)",
                    shown_name,
                    shown_host,
                    port,
                    shown_sampling,
                    shown_drafts,
                    context_window,
                    loaded,
                );
            }) as Box<dyn FnOnce(u16) + Send>);
            match tensorfold_server::serve(state, addr, on_listen).await {
                Ok(()) => 0,
                Err(err) => {
                    eprintln!("tensorfold: {}", err);
                    1
                }
            }
        })
    }
    // A Linux build never reaches here — the family gate refused above —
    // but the function must close without the macOS block's value
    #[cfg(not(target_os = "macos"))]
    {
        return 1;
    }
}

/// The model's end ids plus the tokenizer's chat end, when both are declared.
///
/// Port of `hf_text.zig:126  fn eosIds` — the first holder of the model config's
/// eos_token_id that yields ids (the root's, else text_config's), then the
/// tokenizer config's eos_token through the special tokens, deduplicated.
fn eos_ids(model_config: &serde_json::Value, dir: &std::path::Path) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    let holders = [
        model_config.get("eos_token_id"),
        model_config
            .get("text_config")
            .and_then(|tc| tc.get("eos_token_id")),
    ];
    for holder in holders {
        let Some(v) = holder else { continue };
        match v {
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_u64() {
                    out.push(i as u32);
                }
            }
            serde_json::Value::Array(list) => {
                for x in list {
                    if let Some(i) = x.as_u64() {
                        out.push(i as u32);
                    }
                }
            }
            _ => {}
        }
        if !out.is_empty() {
            break;
        }
    }
    // The tokenizer config's eos_token (a string, or an object with `content`),
    // resolved through the special tokens.
    if let Ok(text) = std::fs::read_to_string(dir.join("tokenizer_config.json")) {
        if let Ok(tc) = serde_json::from_str::<serde_json::Value>(&text) {
            let piece = match tc.get("eos_token") {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(serde_json::Value::Object(o)) => o
                    .get("content")
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_string(),
                _ => String::new(),
            };
            if !piece.is_empty() {
                if let Ok(tokenizer) = Tokenizer::open(dir) {
                    if let Some(id) = tokenizer.special_token_id(&piece) {
                        if !out.contains(&id) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }
    out
}

/// The sampling defaults from the flags over the checkpoint's generation_config.json
/// (the README's precedence: checkpoint defaults, then serve flags).
fn default_sampling(flags: &ServeFlags, dir: &std::path::Path) -> Option<serde_json::Value> {
    // generation_config.json's sampling first (``do_sample`` false is
    // greedy), then the serve flags over it (main.zig:104-119)
    let mut value = serde_json::Map::new();
    if let Ok(bytes) = std::fs::read(dir.join("generation_config.json")) {
        if let Ok(cfg) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            for k in ["temperature", "top_k", "top_p", "min_p"] {
                if let Some(v) = cfg.get(k).filter(|v| !v.is_null()) {
                    value.insert(k.into(), v.clone());
                }
            }
            if let Some(Value::Bool(false)) = cfg.get("do_sample") {
                value.insert("temperature".into(), serde_json::json!(0));
            } else if let Some(Value::Bool(true)) = cfg.get("do_sample") {
                value
                    .entry("temperature".to_string())
                    .or_insert_with(|| serde_json::json!(1));
            }
        }
    }
    if let Some(t) = flags.temperature {
        value.insert("temperature".into(), serde_json::json!(t));
    }
    if let Some(t) = flags.top_p {
        value.insert("top_p".into(), serde_json::json!(t));
    }
    if let Some(t) = flags.top_k {
        value.insert("top_k".into(), serde_json::json!(t));
    }
    if let Some(t) = flags.min_p {
        value.insert("min_p".into(), serde_json::json!(t));
    }
    (!value.is_empty()).then(|| serde_json::Value::Object(value))
}

/// The serving line's sampling (serve.zig:178-196): greedy, or each
/// default as ``name value``.
fn shown_sampling(sampling: Option<&serde_json::Value>) -> String {
    let Some(Value::Object(obj)) = sampling else {
        return "greedy".into();
    };
    let temp = obj
        .get("temperature")
        .and_then(|t| t.as_f64())
        .unwrap_or(0.0);
    if temp <= 0.0 {
        return "greedy".into();
    }
    let parts: Vec<String> = obj
        .iter()
        .map(|(k, v)| match v {
            Value::Number(n) => format!(
                "{k} {}",
                n.as_f64()
                    .map(tensorfold_server::pyrepr::float_repr)
                    .unwrap_or_else(|| n.to_string())
            ),
            _ => format!("{k} ?"),
        })
        .collect();
    parts.join(", ")
}

/// Thousands commas (live.zig:46's `commas`).
fn commas(n: u64) -> String {
    let text = n.to_string();
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(bytes.len() + bytes.len() / 3);
    for (i, ch) in text.char_indices() {
        if i > 0 && (text.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// SIGHUP's request flag (auth.zig:9's `reload_requested`), set from the
/// signal handler — the store rereads the key file on its next check.
extern "C" fn on_hangup(_sig: i32) {
    tensorfold_server::auth::RELOAD_REQUESTED.store(true, std::sync::atomic::Ordering::Release);
}

/// Upstream: `installHangup` (auth.zig:16).
fn install_hangup() {
    const SIGHUP: i32 = 1;
    // SAFETY: the handler only sets an atomic flag — async-signal-safe.
    unsafe {
        let handler = on_hangup as extern "C" fn(i32);
        signal(SIGHUP, handler as usize);
    }
}

unsafe extern "C" {
    fn signal(signum: i32, handler: usize) -> usize;
}

/// Upstream `auth.loopback`: whether the host is a loopback address, the check behind
/// the open-server warning at serve.zig:91.
fn is_loopback(host: &str) -> bool {
    host == "127.0.0.1" || host == "localhost" || host == "::1" || host == "[::1]"
}

/// The nemotron_h host on Linux (native/cuda.zig's openWith + Host's
/// serve): the memory plan, the family open, the lane host — then the
/// same server wiring the Metal family's path runs.
#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
fn serve_linux_cuda(
    flags: &ServeFlags,
    dir: &std::path::Path,
    config: &ModelConfig,
    server_config: Config,
    auth: Auth,
) -> i32 {
    let load_started = std::time::Instant::now();

    // The text (the same rule as the Metal path): the tokenizer, the
    // chat template and the eos ids from the checkpoint's configs.
    let tokenizer = match Tokenizer::open(dir) {
        Ok(t) => t,
        Err(err) => {
            eprintln!("tensorfold: {}", err);
            return 1;
        }
    };
    let template = match tensorfold_core::ChatTemplate::open(dir) {
        Ok(t) => t,
        Err(err) => {
            eprintln!("tensorfold: {}", err);
            return 1;
        }
    };
    let eos: Vec<u32> = eos_ids(&config.json, dir);

    // The memory plan and the family open (native/cuda.zig:306's
    // openWith), every refusal in upstream's words.
    let mut problem = String::new();
    let opened = match crate::serve_cuda::open(
        flags,
        dir,
        !flags.no_drafts,
        &mut problem,
    ) {
        Some(o) => o,
        None => {
            eprintln!("tensorfold: {}", problem);
            return 1;
        }
    };
    // The startup line (h.startup, cuda.zig:378): upstream's own
    // memory plan the server logs at load.
    eprintln!("[tensorfold] {}", opened.startup);

    let context_window = opened.context_window;
    let engine = opened.engine;
    let step = opened.prefill_step;

    // The server state with the engine and its text wired.
    let text = std::sync::Arc::new(tensorfold_server::Text::read(
        tokenizer, template, eos,
    ));
    let shown_sampling = shown_sampling(server_config.default_sampling.as_ref());
    let mut state = AppState::new(server_config);
    state = state.with_engine(engine, text);
    state = state.with_auth(auth);
    let state = std::sync::Arc::new(state);
    // The chunk-plan line (server.zig:114).
    if step > 0 {
        println!(
            "[tensorfold] prompt chunks of up to {} tokens, cut at replies {}+ tokens apart",
            commas(step as u64),
            state.chunk_plan.min_chunk,
        );
    }

    // The HTTP server: the same wiring as the Metal path.
    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(err) => {
            eprintln!("tensorfold: {}", err);
            return 1;
        }
    };
    let host_addr = flags.host.clone();
    let port = flags.port;
    runtime.block_on(async move {
        let addr: std::net::SocketAddr = match format!("{}:{}", host_addr, port).parse() {
            Ok(a) => a,
            Err(_) => {
                eprintln!("tensorfold: cannot listen on {}:{}", host_addr, port);
                return 1;
            }
        };
        let shown_name = state.config.model_ids[0].clone();
        let shown_host = host_addr.clone();
        let shown_drafts = if flags.no_drafts { "off" } else { "on" };
        let loaded = load_started.elapsed().as_secs_f64();
        let on_listen = Some(Box::new(move |listen_port: u16| {
            println!(
                "[tensorfold] serving {} at http://{}:{}/v1 (sampling: {}; drafts: {}; context: {}; loaded in {:.1}s)",
                shown_name,
                shown_host,
                listen_port,
                shown_sampling,
                shown_drafts,
                context_window,
                loaded,
            );
        }) as Box<dyn FnOnce(u16) + Send>);
        match tensorfold_server::serve(state, addr, on_listen).await {
            Ok(()) => 0,
            Err(err) => {
                eprintln!("tensorfold: {}", err);
                1
            }
        }
    })
}
