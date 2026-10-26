//! The lane round loop (Python FamilyRounds and SharedRounds) and the state it
//! drives — the port of `zig/src/core/lanes/`.
//!
//! # The core's shape, and why this crate uses raw pointers
//!
//! Upstream's core is a single-threaded, pointer-linked graph: the engine holds
//! `live: []*Stream`, streams hold proposer/depth state the rule mutates through
//! `Who`, and the depth rule asks the engine back whether a copy proposal waits
//! (`Probe`). Rust's borrow checker cannot express those cross-links, and
//! redesigning them away would change the very code whose decisions the exactness
//! gates pin. So this crate keeps upstream's topology with raw pointers, under the
//! same invariants upstream runs by:
//!
//! 1. One thread owns the engine and everything it touches for as long as it steps.
//! 2. A stream stays alive, unmoved (boxed), and distinct from every other live
//!    stream from its `add_stream` until the engine has released it (finished) or
//!    the caller discarded it.
//! 3. Raw pointers into a stream are only dereferenced while that invariant holds;
//!    every unsafe block names the field it touches.
//!
//! The `fake` backend and the engine tests are upstream's own
//! (`engine_test.zig`): drafted rounds must equal one-token rounds, shared rounds
//! must equal solo runs, and logprob rows must follow their tokens — greedy and
//! sampled. The qwen35 backend implements the same `Backend` trait against the
//! real model.

// The Backend trait is the port of upstream's `ptr: *anyopaque` + vtable: its
// stream parameters carry the crate's documented invariants, so the raw-pointer
// deref lint (not_unsafe_ptr_arg_deref) fires on the very design this port
// mirrors. The invariants live in this file's crate documentation.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

pub mod accept;
pub mod allocate;
pub mod backend;
pub mod cache_modes;
pub mod lane_host;
pub mod config;
pub mod depth;
pub mod engine;
pub mod events;
pub mod fake;
pub mod fill;
pub mod plan_lanes;
pub mod learned_dirs;
pub mod learned_disk;
pub mod learned_plan;
pub mod prompt_cache;
pub mod prompt_imprint;
#[cfg(test)]
pub(crate) mod prompt_cache_fake;
pub mod prompt_learn;
pub mod proposer;
pub mod shape;
pub mod stagger;
pub mod stream;
pub mod table;
pub mod trail;
pub mod windows;

pub use allocate::Cost;
pub use backend::{
    Alternative, Backend, Clock, DraftRequest, Feed, Features, FixedClock, Mark, Verified,
    WallClock, Window,
};
pub use config::{Config, Model};
pub use engine::Engine;
pub use tensorfold_engine::Sampling; // upstream: lanes.zig:33
pub use tensorfold_engine::LogprobRow; // upstream: lanes.zig:35
pub use tensorfold_engine::logprob::ROW_WORDS as row_words; // the row's words, upstream logprob.words
pub use lane_host::LaneHost;
pub use proposer::{Proposer, SuffixLookup};
pub use stream::{Held, Mode, Reason, Spec, Stream};
pub use table::Table;

/// The core's errors, one flat set like the Zig error unions it mirrors; family
/// backends carry their own words through `Backend`.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// A prompt pass saw its request cancelled between chunks.
    Cancelled,
    /// Logprob rows must arrive one a committed token.
    LogprobRowsMismatch,
    /// Tree parents must precede their children.
    ParentAfterChild,
    BadMaxRows,
    BadNgram,
    MinMatchBelowNgram,
    /// The backend serves no feature taps.
    NoFeatures,
    /// The backend gives no first-row logprob.
    LogprobsUnsupported,
    /// A backend refused a position it does not hold.
    PositionMismatch,
    RowsNotHeld,
    NotAtMark,
    NothingToKeep,
    /// A family backend's refusal, with its own words (the Zig engine logs and
    /// propagates; the port carries the message).
    Backend(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Cancelled => f.write_str("the request was cancelled"),
            Error::LogprobRowsMismatch => f.write_str("logprob rows must arrive one a committed token"),
            Error::ParentAfterChild => f.write_str("a tree's parents must precede its children"),
            Error::BadMaxRows => f.write_str("max rows must be one or more"),
            Error::BadNgram => f.write_str("the n-gram size is out of range"),
            Error::MinMatchBelowNgram => f.write_str("the minimum match is below the n-gram"),
            Error::NoFeatures => f.write_str("the backend serves no feature taps"),
            Error::LogprobsUnsupported => f.write_str("the backend gives no first-row logprob"),
            Error::PositionMismatch => f.write_str("a backend refused a position it does not hold"),
            Error::RowsNotHeld => f.write_str("the backend holds fewer rows than asked"),
            Error::NotAtMark => f.write_str("the prompt pass does not stand at the mark"),
            Error::NothingToKeep => f.write_str("no recorded round to keep rows of"),
            Error::Backend(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {}

/// The words a family's error maps to inside the core (upstream funnels `anyerror`
/// through; the port keeps the message).
pub fn backend_error(text: impl Into<String>) -> Error {
    Error::Backend(text.into())
}
