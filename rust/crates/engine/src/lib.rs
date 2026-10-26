//! tensorfold-engine: the contract between a TensorFold server and the engines it drives.
//!
//! This crate is a port of the Zig engine's `zig/src/core/engine_api.zig` (upstream commit on
//! the `main` branch of TensorFold 1.0.4). Every type here carries a comment naming the Zig
//! declaration it mirrors; when porting more subsystems, keep the two in step.
//!
//! The crate has no dependencies beyond std and sha2 (for `sampling::seed_for`), exactly like
//! the upstream module, which depends only on std.crypto and the lanes types it re-exports.

pub mod backend;
pub mod engine;
pub mod event;
pub mod learn;
pub mod logprob;
pub mod request;
pub mod sampling;

pub use backend::{Backend, NoneBackend, Open, Opened};
pub use engine::{Engine, EventSink, Info, Memory, PromptCachePlan, PromptCacheSource, ScoreError, Status, SubmitError};
pub use event::{Event, Finished, Reason, Stats};
pub use learn::{LearnError, LearnEvent, LearnRequest, LearnSink, Learner};
pub use logprob::{LogprobRow, MAX_TOP, ROW_WORDS};
pub use request::{
    CallGate, Request, Span, StopCheck, Structure, StructureKind,
};
pub use sampling::{key_bits, mix, pairwise_sum, sample_seed_for, choose, uniform, Sampling};

/// The server's name for one request, unique while the process lives.
///
/// Upstream: `engine_api.zig:7  pub const Id = u64;`
pub type Id = u64;

/// A checkpoint family an engine reads: its config `model_type` and weight formats, as gate
/// entries name them.
///
/// Upstream: `engine_api.zig:160  pub const Family = struct { model_type, formats };`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Family {
    pub model_type: &'static str,
    pub formats: &'static [&'static str],
}
