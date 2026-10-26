//! GLM-5.3-Flash: the family crate — the config's exactly-pinned shape
//! and its qualification checks first (the engine, weights and checks
//! follow tranche by tranche).

pub mod config;
pub mod learned_pair;
pub mod mirror;
pub mod weights_fix;
pub mod snapshot_file;

pub use config::{Config, Kind, ATTN_SPAN, MAX_LAYERS};
