//! `tensorfold-native`: `capabilities --json` for the switch tooling, and
//! `serve MODEL [flags]` with no Python.
//!
//! Port of `zig/src/server/main.zig`'s command surface. The checkpoint subcommands
//! (`models`, `info`, `pull`) are not ported yet — they answer with a clear refusal and
//! land in Phase 2 (see docs/PORTING-PLAN.md), the way the Zig binary refuses flags it
//! does not honour.

mod cache_fit;
mod capabilities;
mod hub;
mod info;
mod models;
mod pull;
mod pull_parts;
mod serve;
#[cfg(target_os = "linux")]
mod serve_cuda;

use clap::Parser;

/// Upstream: `server/main.zig:26  usage: tensorfold-native --version | capabilities
/// --json | models | info MODEL | pull REPO[@REVISION] | serve MODEL [flags]`.
#[derive(Parser)]
#[command(
    name = "tensorfold-native",
    version,
    about = "TensorFold's native engine, ported to Rust",
    disable_help_subcommand = true,
    subcommand_required = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand)]
// One Command a process; boxing the serve flags would buy nothing.
#[allow(clippy::large_enum_variant)]
enum Command {
    /// The capabilities document (with --json), for switch tooling.
    Capabilities { #[arg(long)] json: bool },

    /// List the cached checkpoints this build can serve.
    Models,

    /// Show one checkpoint's family, format, context and memory floor.
    Info { model: String },

    /// Download a checkpoint into the Hugging Face cache.
    Pull { repo: String },

    /// Serve a local model directory.
    Serve {
        /// A complete local model directory (an existing Hugging Face cache entry by
        /// repo id lands in Phase 2 with hub.zig).
        model: String,
        #[command(flatten)]
        flags: serve::ServeFlags,
    },
}

fn main() {
    let cli = Cli::parse();
    let code = match cli.command {
        Command::Capabilities { json } => capabilities::run(json),
        Command::Models => models::run(&mut std::io::stdout(), None),
        Command::Info { model } => info::run(&mut std::io::stdout(), None, &model),
        Command::Pull { repo } => pull::run(&mut std::io::stdout(), &repo) as i32,
        Command::Serve { model, flags } => serve::run(&model, &flags),
    };
    std::process::exit(code);
}


