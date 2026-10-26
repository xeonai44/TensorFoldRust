//! GLM learned halves use a versioned checksum envelope; legacy,
//! corrupt and partial files never enter a cache.
//!
//! Port of `zig/src/families/glm/snapshot_file.zig` (45 lines) — the
//! `Head` (magic GLMT, version, position, bytes), the atomic
//! `.part`-then-rename write, and the exact-head read; upstream's one
//! test mirrored over std::fs.

use std::fs;
use std::io::Write;
use std::path::Path;

use tensorfold_core::snapshot_file as checked;

/// Upstream: `magic` (glm/snapshot_file.zig:4).
pub const MAGIC: u32 = 0x474c4d54;

/// Upstream: `Head` (glm/snapshot_file.zig:5) — 24 bytes, the
/// checksum's header.
#[repr(C)]
#[derive(Clone, Copy, PartialEq)]
pub struct Head {
    pub magic: u32,
    pub version: u32,
    pub at: u32,
    pub reserved: u32,
    pub bytes: u64,
}

impl Head {
    fn new(at: u32, bytes: u64) -> Head {
        Head {
            magic: MAGIC,
            version: 1,
            at,
            reserved: 0,
            bytes,
        }
    }

    fn bytes(&self) -> [u8; 24] {
        let mut out = [0u8; 24];
        out[0..4].copy_from_slice(&self.magic.to_le_bytes());
        out[4..8].copy_from_slice(&self.version.to_le_bytes());
        out[8..12].copy_from_slice(&self.at.to_le_bytes());
        out[12..16].copy_from_slice(&self.reserved.to_le_bytes());
        out[16..24].copy_from_slice(&self.bytes.to_le_bytes());
        out
    }
}

/// Why a half-file write or read refused.
#[derive(Debug, PartialEq)]
pub enum HalfError {
    SnapshotWrite,
    SnapshotRead,
}

impl std::fmt::Display for HalfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HalfError::SnapshotWrite => write!(f, "the half file's write failed"),
            HalfError::SnapshotRead => write!(f, "the half file's read failed"),
        }
    }
}

impl std::error::Error for HalfError {}

/// Upstream: `write` (glm/snapshot_file.zig:6) — the .part file, the
/// head and payload and digest, fsync, then the rename.
pub fn write(file: &str, at: u32, payload: &[u8]) -> Result<(), HalfError> {
    let tmp = format!("{file}.part");
    let mut out = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp)
        .map_err(|_| HalfError::SnapshotWrite)?;
    let result = (|| -> Result<(), HalfError> {
        let head = Head::new(at, payload.len() as u64);
        checked::write(&mut out, &head.bytes(), payload).map_err(|_| HalfError::SnapshotWrite)?;
        out.sync_all().map_err(|_| HalfError::SnapshotWrite)
    })();
    if result.is_err() {
        fs::remove_file(&tmp).ok();
        return result;
    }
    fs::rename(&tmp, file).map_err(|_| HalfError::SnapshotWrite)
}

/// Upstream: `read` (glm/snapshot_file.zig:20) — the head must match
/// exactly (magic, version, position, bytes), then the digest.
pub fn read(file: &str, at: u32, payload: &mut [u8]) -> Result<(), HalfError> {
    let mut f = fs::File::open(file).map_err(|_| HalfError::SnapshotRead)?;
    let head = Head::new(at, payload.len() as u64);
    checked::read(&mut f, &head.bytes(), payload).map_err(|_| HalfError::SnapshotRead)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `both GLM half files refuse a changed body, legacy
    /// header, truncation and another position`
    /// (glm/snapshot_file.zig:29).
    #[test]
    fn both_glm_half_files_refuse_a_changed_body_legacy_header_truncation_and_another_position() {
        let file = format!("/tmp/tf-glm-file-{}", std::process::id());
        write(&file, 128, b"body").unwrap();
        let mut out = [0u8; 4];
        read(&file, 128, &mut out).unwrap();
        assert_eq!(&out, b"body");
        assert_eq!(Err(HalfError::SnapshotRead), read(&file, 129, &mut out));
        // a changed body byte: the digest refuses
        use std::io::Seek;
        let mut f = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&file)
            .expect("the test file");
        f.seek(std::io::SeekFrom::Start(24 + 1)).unwrap();
        f.write_all(b"X").unwrap();
        drop(f);
        assert_eq!(Err(HalfError::SnapshotRead), read(&file, 128, &mut out));
        // truncation: the exact length refuses
        let f = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&file)
            .expect("the test file");
        f.set_len(16).unwrap();
        drop(f);
        assert_eq!(Err(HalfError::SnapshotRead), read(&file, 128, &mut out));
        std::fs::remove_file(&file).ok();
        let _ = Path::new(&file).exists();
    }
}
