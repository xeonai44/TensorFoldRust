//! tensorfold-core: what loads before any GPU work: the checkpoint's safetensors, the
//! tokenizer, and the model's config files.
//!
//! Each module names the Zig source it ports. See docs/ZIG-TO-RUST-MAP.md for status.
//! The safetensors rules themselves live in their own crate,
//! `tensorfold-safetensors`, shared with the Metal-resident checkpoint.

// The Metal kernel modules of upstream's core (frags, moe_route, affine_mm)
// exist on macOS alone, like the metal crate they bind (upstream compiles
// them only into the native Metal registry).
#[cfg(target_os = "macos")]
pub mod affine_mm;
#[cfg(target_os = "macos")]
pub mod frags;
#[cfg(target_os = "macos")]
pub mod moe_route;

pub mod checkpoint;
pub mod affine4_host;
pub mod copy_index;
pub mod direct_io;
pub mod snapshot_file;
pub mod draft_depth;
pub mod heat;
pub mod config;
pub mod template;
pub mod tokenizer;

pub use checkpoint::{Checkpoint, OpenError as CheckpointOpenError, TensorInfo};
pub use config::{ConfigError, ModelConfig};
pub use tensorfold_safetensors::{DType, SafeTensors, Tensor, MAX_RANK};
pub use template::{render_ids, ChatTemplate, RenderOptions, TemplateError};
pub use tokenizer::{Tokenizer, TokenizerError};
