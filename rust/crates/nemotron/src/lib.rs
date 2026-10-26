//! Nemotron-H on CUDA: the family's config, its draft vocabulary, the
//! glue's host math and references, and the kernel launch layers.
//!
//! Port of `zig/src/families/nemotron/cuda.zig` (37 lines) — the root's
//! public surface, grown tranche by tranche. This tranche: the
//! foundation (config, draft ids, glue math and references) and the
//! launch layers (torch ops, Triton dispatch, glue, kernels); the
//! weights/state/engine layers follow.

pub mod config;
pub mod cuda_glue;
pub mod cuda_kernels;
pub mod cuda_prompt_grid;
pub mod cuda_sampler;
pub mod cuda_costs;
pub mod cuda_decode;
pub mod cuda_drafts;
pub mod cuda_engine;
pub mod cuda_lanes;
pub mod cuda_lone;
pub mod cuda_native;
pub mod cuda_memory;
pub mod native_decisions;
pub mod cuda_reuse;
pub mod cuda_mtp;
pub mod cuda_segments;
pub mod cuda_source;
pub mod cuda_state;
pub mod cuda_dump;
pub mod cuda_forward;
pub mod cuda_weights;
pub mod cuda_torch_ops;
pub mod cuda_triton;
pub mod draft_ids;
pub mod error;
pub mod glue_math;
pub mod glue_ref;
pub mod uring;

pub use config::{Config, Kind};
pub use cuda_kernels::{Experts, Kernels, Ops, Plan, QLinear};
pub use error::Error;

// Upstream: `Drafter`/`Head`/`Lanes` (cuda.zig:14-20) arrive with the
// state and engine tranches; `weights` with the loader tranche.
