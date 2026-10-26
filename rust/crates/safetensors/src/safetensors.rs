//! Reading one safetensors image: the header's JSON, its dtype and shape rules, and the
//! data it frames.
//!
//! Port of `zig/src/core/safetensors.zig`. Upstream parses the format itself rather than
//! calling a library, and enforces TensorFold's own rules on top of it; this port keeps
//! that approach (decision record: docs/DECISIONS.md, ADR-007) so the accepted files are
//! exactly the ones the Zig engine accepts — including the F8_E8M0 dtype and the rank-5
//! ceiling.
//!
//! Format: 8-byte little-endian header length, the JSON header, then every tensor's
//! bytes, one after another.

use crate::dtype::DType;
use serde_json::Value;
use std::fs;
use std::path::Path;

/// Upstream: `safetensors.zig` (near `max_rank`) — a tensor's rank may not exceed this.
pub const MAX_RANK: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    BadSafetensors,
    UnsupportedDType,
    RankTooHigh,
    BadShape,
    BadOffsets,
    BadHeaderLength,
    NotUtf8,
    Io,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            Error::BadSafetensors => "the safetensors header is not shaped as expected",
            Error::UnsupportedDType => "the safetensors header names a dtype this build refuses",
            Error::RankTooHigh => "a tensor's rank is above the engine's ceiling",
            Error::BadShape => "a tensor's shape is not the engine's kind of dimensions",
            Error::BadOffsets => "a tensor's data offsets disagree with its shape",
            Error::BadHeaderLength => "the safetensors header length is wrong",
            Error::NotUtf8 => "the safetensors header is not UTF-8",
            Error::Io => "reading the safetensors file failed",
        };
        f.write_str(text)
    }
}

impl std::error::Error for Error {}

/// One tensor's header entry, with its data resolved to `data` offsets within the full
/// image (header included).
///
/// Upstream: `safetensors.zig:113  pub const Tensor = struct { dtype, shape, ... }`.
#[derive(Clone, Debug)]
pub struct Tensor<'a> {
    pub dtype: DType,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}

impl<'a> Tensor<'a> {
    /// Upstream: `safetensors.zig:128  pub fn is(...) bool` — the gate a loader asserts a
    /// tensor against: the dtype and the exact shape.
    pub fn is(&self, dtype: DType, shape: &[usize]) -> bool {
        self.dtype == dtype && self.shape == shape
    }

    /// Upstream: the `rank` field (safetensors.zig:112).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Upstream: `dim` (safetensors.zig:118).
    pub fn dim(&self, i: usize) -> usize {
        self.shape[i]
    }
}

/// A parsed safetensors image: the header (kept whole, so `__metadata__` reads work)
/// and every named tensor, over borrowed image bytes.
#[derive(Debug)]
pub struct SafeTensors<'a> {
    header: Value,
    entries: Vec<(String, Entry)>,
    bytes: &'a [u8],
    data_start: usize,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
}

/// One tensor's place in a shard's data region: the name-keyed read the Metal-resident
/// checkpoint does before it uploads anything.
///
/// Upstream: `safetensors.zig` `parseHeaderPrefix`'s `Entry`.
pub use Entry as HeaderEntry;

/// Select one tensor namespace before interpreting shapes, so a text engine need not
/// admit a vision tower's layouts.
///
/// Upstream: `safetensors.zig  pub fn parseHeaderPrefix(arena, json, data_len, prefix)`
/// — every entry the header frames, filtered by name prefix, each validated against the
/// data region's length. The entries come in the header's own order; callers key by
/// name.
pub fn parse_header_prefix(
    header: &str,
    data_len: usize,
    prefix: Option<&str>,
) -> Result<Vec<(String, HeaderEntry)>, Error> {
    let value: Value = serde_json::from_str(header).map_err(|_| Error::BadSafetensors)?;
    let object = value.as_object().ok_or(Error::BadSafetensors)?;
    let mut out = Vec::with_capacity(object.len());
    for (name, spec) in object {
        if name == "__metadata__" {
            continue;
        }
        if let Some(p) = prefix {
            if !name.starts_with(p) {
                continue;
            }
        }
        out.push((name.clone(), parse_entry(name, spec, data_len)?));
    }
    Ok(out)
}

impl<'a> SafeTensors<'a> {
    /// Parse `bytes` (a whole safetensors file).
    pub fn parse(bytes: &'a [u8]) -> Result<SafeTensors<'a>, Error> {
        if bytes.len() < 8 {
            return Err(Error::BadHeaderLength);
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        if header_len > bytes.len().saturating_sub(8) {
            return Err(Error::BadHeaderLength);
        }
        let header_end = 8 + header_len;
        let header = std::str::from_utf8(&bytes[8..header_end]).map_err(|_| Error::NotUtf8)?;
        let value: Value =
            serde_json::from_str(header).map_err(|_| Error::BadSafetensors)?;
        let object = value.as_object().ok_or(Error::BadSafetensors)?;
        let mut entries = Vec::with_capacity(object.len());
        for (name, spec) in object {
            if name == "__metadata__" {
                continue;
            }
            entries.push((name.clone(), parse_entry(name, spec, bytes.len() - header_end)?));
        }
        // Name order, the order the namespace lists tensors in (checkpoint.zig's "shard
        // files, tensors mapped" walk is by name).
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(SafeTensors {
            header: value,
            entries,
            bytes,
            data_start: header_end,
        })
    }

    /// The `__metadata__` object, when the writer left one.
    pub fn metadata(&self) -> Option<&Value> {
        self.header.get("__metadata__").filter(|m| m.is_object())
    }

    /// Every tensor name, in name order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(n, _)| n.as_str())
    }

    /// Upstream: `safetensors.zig` `get` — a tensor by name, or None when absent.
    /// The rows borrow this image for as long as the caller holds it.
    pub fn get(&self, name: &str) -> Option<Tensor<'_>> {
        let (_, e) = self.entries.iter().find(|(n, _)| n == name)?;
        Some(Tensor {
            dtype: e.dtype,
            shape: &e.shape,
            data: &self.bytes[self.data_start + e.start..self.data_start + e.end],
        })
    }
}

fn parse_entry(_name: &str, spec: &Value, data_len: usize) -> Result<Entry, Error> {
    let object = spec.as_object().ok_or(Error::BadSafetensors)?;
    let dtype = object
        .get("dtype")
        .and_then(Value::as_str)
        .and_then(DType::parse)
        .ok_or(Error::UnsupportedDType)?;
    let shape_json = object.get("shape").ok_or(Error::BadSafetensors)?;
    let dims = shape_json.as_array().ok_or(Error::BadShape)?;
    if dims.len() > MAX_RANK {
        return Err(Error::RankTooHigh);
    }
    let mut shape = Vec::with_capacity(dims.len());
    let mut count: usize = 1;
    for d in dims {
        // Only whole, positive dimensions; -1, 1.0 and "1" are refusals
        // (safetensors.zig:260-262).
        let dim = d.as_u64().ok_or(Error::BadShape)? as usize;
        if dim == 0 || dim > i64::MAX as usize {
            return Err(Error::BadShape);
        }
        count = count.checked_mul(dim).ok_or(Error::BadShape)?;
        shape.push(dim);
    }
    let bytes = count
        .checked_mul(dtype.size())
        .ok_or(Error::BadShape)?;
    let offsets = object
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or(Error::BadSafetensors)?;
    if offsets.len() != 2 {
        return Err(Error::BadOffsets);
    }
    let start = offsets[0].as_u64().ok_or(Error::BadOffsets)? as usize;
    let end = offsets[1].as_u64().ok_or(Error::BadOffsets)? as usize;
    if start > end || end > data_len || end - start != bytes {
        return Err(Error::BadOffsets);
    }
    Ok(Entry {
        dtype,
        shape,
        start,
        end,
    })
}

/// A whole safetensors file from disk, for `Checkpoint::open`; returns the image bytes
/// so the caller keeps the mmap alive.
pub fn read_file(path: &Path) -> Result<Vec<u8>, Error> {
    fs::read(path).map_err(|_| Error::Io)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a safetensors image by hand, the way the Zig tests write their fixtures.
    fn image(header: &str, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + header.len() + data.len());
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        out
    }

    /// Mirrors safetensors.zig:178-183: the MLX metadata line, a U32 and a BF16 tensor.
    #[test]
    fn parses_mlx_affine_pair() {
        let header = r#"{"__metadata__": {"format": "mlx"}, "a.weight": {"dtype": "U32", "shape": [2, 3], "data_offsets": [0, 24]},
 "a.scales": {"dtype": "BF16", "shape": [2], "data_offsets": [24, 28]}}"#;
        let bytes = image(header, &[0u8; 28]);
        let st = SafeTensors::parse(&bytes).unwrap();
        assert_eq!(st.metadata().unwrap()["format"], "mlx");
        let w = st.get("a.weight").unwrap();
        assert!(w.is(DType::U32, &[2, 3]));
        assert_eq!(w.data.len(), 24);
        assert_eq!(st.get("a.scales").unwrap().dtype, DType::BF16);
        let names: Vec<&str> = st.names().collect();
        assert_eq!(names, ["a.scales", "a.weight"]); // name order
    }

    /// Mirrors safetensors.zig:223: rank six is refused.
    #[test]
    fn refuses_rank_six() {
        let header = r#"{"t":{"dtype":"U8","shape":[1,1,1,1,1,1],"data_offsets":[0,1]}}"#;
        assert!(matches!(
            SafeTensors::parse(&image(header, &[7])),
            Err(Error::RankTooHigh)
        ));
    }

    /// Mirrors safetensors.zig:226-262: the shape and offset refusals.
    #[test]
    fn refuses_wrong_shapes_and_offsets() {
        let cases = [
            (r#"{"t":{"dtype":"U8","shape":[1,1,1,1,-1],"data_offsets":[0,1]}}"#, Error::BadShape),
            (r#"{"t":{"dtype":"U8","shape":[1,1,1,1,2],"data_offsets":[0,1]}}"#, Error::BadOffsets),
            (r#"{"t":{"dtype":"U8","shape":[4611686018427387904,4],"data_offsets":[0,0]}}"#, Error::BadShape),
            (r#"{"t":{"dtype":"U8","shape":[1.0],"data_offsets":[0,1]}}"#, Error::BadShape),
            (r#"{"t":{"dtype":"U8","shape":["1"],"data_offsets":[0,1]}}"#, Error::BadShape),
        ];
        for (header, err) in cases {
            assert!(
                matches!(SafeTensors::parse(&image(header, &[0])), Err(e) if e == err),
                "{}",
                header
            );
        }
    }

    /// Mirrors safetensors.zig:196-198: every F8 spelling parses.
    #[test]
    fn parses_f8_spellings() {
        let header = r#"{"w":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]},
 "s":{"dtype":"F8_E8M0","shape":[1],"data_offsets":[1,2]},
 "g":{"dtype":"F8_E5M2","shape":[1],"data_offsets":[2,3]}}"#;
        let bytes = image(header, &[0, 0, 0]);
        let st = SafeTensors::parse(&bytes).unwrap();
        assert_eq!(st.get("w").unwrap().dtype, DType::F8E4M3);
        assert_eq!(st.get("s").unwrap().dtype, DType::F8E8M0);
        assert_eq!(st.get("g").unwrap().dtype, DType::F8E5M2);
    }

    /// Mirrors safetensors.zig:206: a tensor may sit in another's shadow (any order,
    /// any interleave) as long as the offsets frame it.
    #[test]
    fn parses_vision_then_text() {
        let header = r#"{"vision.weight":{"dtype":"BF16","shape":[1,1,1,1,1],"data_offsets":[0,2]},
 "text.weight":{"dtype":"U32","shape":[2,3],"data_offsets":[2,26]}}"#;
        let bytes = image(header, &[3u8; 26]);
        let st = SafeTensors::parse(&bytes).unwrap();
        assert_eq!(st.get("text.weight").unwrap().data.len(), 24);
    }
}
