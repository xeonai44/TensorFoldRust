//! The safetensors dtypes TensorFold reads.
//!
//! Port of the `DType` declaration in `zig/src/core/safetensors.zig` (lines 6-40),
//! including the F8 spellings upstream parses and the byte sizes its `size` returns.

/// Upstream: `safetensors.zig:6  pub const DType = enum { ... }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    U8,
    I8,
    U16,
    I16,
    F16,
    BF16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
    F8E4M3,
    F8E5M2,
    F8E8M0,
}

impl DType {
    /// Upstream: `safetensors.zig:24  pub fn size(self: DType) usize`.
    pub fn size(self) -> usize {
        match self {
            DType::Bool
            | DType::U8
            | DType::I8
            | DType::F8E4M3
            | DType::F8E5M2
            | DType::F8E8M0 => 1,
            DType::U16 | DType::I16 | DType::F16 | DType::BF16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::U64 | DType::I64 | DType::F64 => 8,
        }
    }

    /// Upstream: `safetensors.zig:33  pub fn parse(text) ?DType` — exact names, case
    /// sensitive; anything else is unsupported, never guessed.
    pub fn parse(text: &str) -> Option<DType> {
        Some(match text {
            "BOOL" => DType::Bool,
            "U8" => DType::U8,
            "I8" => DType::I8,
            "U16" => DType::U16,
            "I16" => DType::I16,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "U32" => DType::U32,
            "I32" => DType::I32,
            "F32" => DType::F32,
            "U64" => DType::U64,
            "I64" => DType::I64,
            "F64" => DType::F64,
            "F8_E4M3" => DType::F8E4M3,
            "F8_E5M2" => DType::F8E5M2,
            "F8_E8M0" => DType::F8E8M0,
            _ => return None,
        })
    }

    /// The header spelling, for writers and test fixtures.
    pub fn as_str(self) -> &'static str {
        match self {
            DType::Bool => "BOOL",
            DType::U8 => "U8",
            DType::I8 => "I8",
            DType::U16 => "U16",
            DType::I16 => "I16",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::U32 => "U32",
            DType::I32 => "I32",
            DType::F32 => "F32",
            DType::U64 => "U64",
            DType::I64 => "I64",
            DType::F64 => "F64",
            DType::F8E4M3 => "F8_E4M3",
            DType::F8E5M2 => "F8_E5M2",
            DType::F8E8M0 => "F8_E8M0",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors safetensors.zig:183-196: parse the names the MLX affine checkpoints use.
    #[test]
    fn parses_every_name_and_size() {
        for (name, dt, size) in [
            ("BOOL", DType::Bool, 1),
            ("U8", DType::U8, 1),
            ("I8", DType::I8, 1),
            ("U16", DType::U16, 2),
            ("I16", DType::I16, 2),
            ("F16", DType::F16, 2),
            ("BF16", DType::BF16, 2),
            ("U32", DType::U32, 4),
            ("I32", DType::I32, 4),
            ("F32", DType::F32, 4),
            ("U64", DType::U64, 8),
            ("I64", DType::I64, 8),
            ("F64", DType::F64, 8),
            ("F8_E4M3", DType::F8E4M3, 1),
            ("F8_E5M2", DType::F8E5M2, 1),
            ("F8_E8M0", DType::F8E8M0, 1),
        ] {
            assert_eq!(DType::parse(name), Some(dt), "{}", name);
            assert_eq!(dt.size(), size);
            assert_eq!(dt.as_str(), name);
        }
        assert_eq!(DType::parse("f32"), None);
        assert_eq!(DType::parse("MLX_BF16"), None);
        assert_eq!(DType::parse(""), None);
    }
}
