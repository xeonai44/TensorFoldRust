#![cfg(target_os = "macos")]
//! tensorfold-metal: the Metal host layer.
//!
//! Port of the essentials of `zig/src/metal/` for Phase 1: the device, command
//! queue, buffers and pipelines, the compute encoder, runtime kernel compilation
//! with MLX's options, and the Metal-resident checkpoint. The rest of the upstream
//! layer — indirect command buffers (`icb.zig`), residency (`residency.zig`),
//! fences and sync words (`sync.zig`), the idle keepalive (`keepalive.zig`), the
//! hardware clocks (`clock.zig`) and the ANE experiments (`ane.zig`) — ports with
//! the engines that use them (tracked in docs/ZIG-TO-RUST-MAP.md).

// The objc 0.2 macros carry internal cargo-clippy cfgs that reach our spans; they
// are the dependency's, not ours.
#![allow(unexpected_cfgs)]

pub mod checkpoint;
pub mod device;
pub mod encoder;
pub mod library;
pub mod objc;
pub mod pipeline;

pub use checkpoint::{Checkpoint, Error as CheckpointError, Tensor};
pub use device::{Buffer, Device, Gpu};
pub use encoder::{ComputePass, Encoder};
pub use library::{CompileOptions, Library};
pub use pipeline::{Kernels, Pipeline};

/// The metallib the build compiled and embedded (build.rs).
///
/// Upstream embeds its metallibs the same way (`build.zig:99  fn embedded` writes a
/// module whose bytes are `@embedFile`).
pub const METALLIB: &[u8] = include_bytes!(env!("TF_METALLIB_PATH"));

/// Whether the build compiled real kernels. When false the Metal toolchain was absent
/// and `METALLIB` is a placeholder; engine opens refuse and GPU tests skip.
pub fn kernels_builtin() -> bool {
    env!("TF_METAL_KERNELS") == "1"
}
