#![cfg(target_os = "macos")]
//! Native Metal Qwen3.5-2B, with its tied affine head and hybrid recurrent/attention
//! state.
//!
//! Port of `zig/src/families/qwen3_5/` (qwen3_5.zig's module list, in the same
//! order). The family leans on the shared Metal host layer (`tensorfold-metal`) and
//! the engine contract's sampler (`tensorfold-engine`); the lane-core vtable wiring
//! (backend.zig's Stream-typed half) arrives in Phase 1b with `lanes`, and the parts
//! that need it are marked below. Everything here cites its Zig source line by line.

// The lanes Backend trait carries the lane core's documented pointer invariants
// (tensorfold-lanes/src/lib.rs); this backend implements it, so the raw-pointer
// deref lint fires on the port's own shape.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

pub mod backend;
pub mod config;
pub mod forward;
pub mod kernels;
pub mod model;
pub mod qualify;
pub mod snapshot;
pub mod state;
pub mod weights;

/// The family's errors, one flat set like the Zig error unions it mirrors. The
/// kernel variant carries the compile or lookup message; the rest are exact names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    BadQwenConfig,
    UnsupportedQwenConfig,
    UnsupportedQwenGeometry,
    UnsupportedQwenQuantization,
    UnsupportedQwenRotary,
    UnexpectedQwenTensor,
    QwenHeadTiednessMismatch,
    InvalidQwenPrompt,
    InvalidQwenPosition,
    InvalidQwenToken,
    InvalidQwenWindows,
    UnsupportedQwenWindow,
    UnsupportedQwenTree,
    InvalidQwenSegments,
    QwenRowsExceeded,
    QwenContextFull,
    NothingToKeep,
    QwenCacheLengthMismatch,
    QwenCommittedLogitsMismatch,
    QwenRecurrenceMismatch,
    QwenAttentionCacheMismatch,
    QwenWindowLogitsMismatch,
    QwenCheckpointHasNoDraftHead,
    GpuFailed,
    NoMetalDevice,
    /// A kernel the family's registry does not hold, or a compile failure with the
    /// message kept in the log (the Zig engine logs and errors; the port carries the
    /// text here so callers can print it).
    QwenKernel(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::BadQwenConfig => f.write_str("malformed Qwen config"),
            Error::UnsupportedQwenConfig => f.write_str("unsupported Qwen config"),
            Error::UnsupportedQwenGeometry => f.write_str("unsupported Qwen geometry"),
            Error::UnsupportedQwenQuantization => f.write_str("unsupported Qwen quantization"),
            Error::UnsupportedQwenRotary => f.write_str("unsupported Qwen rotary layout"),
            Error::UnexpectedQwenTensor => f.write_str("unexpected Qwen tensor dtype or shape"),
            Error::QwenHeadTiednessMismatch => {
                f.write_str("the checkpoint's head tiedness disagrees with config.json")
            }
            Error::InvalidQwenPrompt => f.write_str("invalid Qwen prompt"),
            Error::InvalidQwenPosition => f.write_str("invalid Qwen position"),
            Error::InvalidQwenToken => f.write_str("invalid Qwen token"),
            Error::InvalidQwenWindows => f.write_str("invalid Qwen verify windows"),
            Error::UnsupportedQwenWindow => f.write_str("unsupported Qwen verify window"),
            Error::UnsupportedQwenTree => f.write_str("unsupported Qwen tree"),
            Error::InvalidQwenSegments => f.write_str("invalid Qwen segments"),
            Error::QwenRowsExceeded => f.write_str("the forward's rows exceed the scratch"),
            Error::QwenContextFull => f.write_str("the stream's context is full"),
            Error::NothingToKeep => f.write_str("no recorded round to keep rows of"),
            Error::QwenCacheLengthMismatch => f.write_str("the caches disagree on length"),
            Error::QwenCommittedLogitsMismatch => f.write_str("the caches disagree on committed logits"),
            Error::QwenRecurrenceMismatch => f.write_str("the caches disagree on recurrence state"),
            Error::QwenAttentionCacheMismatch => f.write_str("the caches disagree on attention rows"),
            Error::QwenWindowLogitsMismatch => f.write_str("a wide row's logits differ from one-row execution"),
            Error::QwenCheckpointHasNoDraftHead => f.write_str("the Qwen checkpoint has no draft head"),
            Error::GpuFailed => f.write_str("the GPU forward failed"),
            Error::NoMetalDevice => f.write_str("no Metal device"),
            Error::QwenKernel(message) => write!(f, "{}", message),
        }
    }
}

impl std::error::Error for Error {}
