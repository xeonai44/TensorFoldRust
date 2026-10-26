//! tensorfold-safetensors: the safetensors format as TensorFold reads it.
//!
//! Port of `zig/src/core/safetensors.zig`. Upstream parses the format itself rather
//! than calling a library, and enforces TensorFold's own rules on top of it; this
//! crate keeps those rules exactly — the dtype spellings (including `F8_E8M0`), the
//! rank-5 ceiling, and the shape/offset arithmetic refusals its tests pin case by
//! case (decision record: docs/DECISIONS.md, ADR-007).
//!
//! Two reading layers share this parser: the host-side namespace
//! (`tensorfold-core`'s checkpoint) and the Metal-resident shards
//! (`tensorfold-metal`'s checkpoint), which is why the parser lives in its own crate
//! with no heavyweight dependencies.

pub mod dtype;
pub mod safetensors;

pub use dtype::DType;
pub use safetensors::{
    parse_header_prefix, Error, HeaderEntry, SafeTensors, Tensor, MAX_RANK,
};
