//! Learned snapshots carry a SHA-256 of their entire header and
//! payload; exact length is required on admission.
//!
//! Port of `zig/src/core/snapshot_file.zig` (66 lines) — the
//! checksummed write (payload, then the digest over header and
//! payload) and the exact-length read; upstream's one test mirrored
//! over std::fs.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};

use sha2::{Digest, Sha256};

/// Why a snapshot write or read refused.
#[derive(Debug, PartialEq)]
pub enum SnapshotError {
    SnapshotWrite,
    SnapshotRead,
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnapshotError::SnapshotWrite => write!(f, "the snapshot write failed"),
            SnapshotError::SnapshotRead => write!(f, "the snapshot read failed"),
        }
    }
}

impl std::error::Error for SnapshotError {}

/// Upstream: `write` (snapshot_file.zig:6) — header, payload, then
/// the SHA-256 of both.
pub fn write(file: &mut File, header: &[u8], payload: &[u8]) -> Result<(), SnapshotError> {
    let mut hash = Sha256::new();
    hash.update(header);
    hash.update(payload);
    put(file, header)?;
    put(file, payload)?;
    put(file, &hash.finalize())?;
    file.sync_all().map_err(|_| SnapshotError::SnapshotWrite)
}

/// Upstream: `read` (snapshot_file.zig:15) — the exact length, then
/// the digest checked.
pub fn read(file: &mut File, header: &[u8], payload: &mut [u8]) -> Result<(), SnapshotError> {
    let length = match header.len().checked_add(payload.len()) {
        Some(n) => n,
        None => return Err(SnapshotError::SnapshotRead),
    };
    let total = match length.checked_add(32) {
        Some(n) => n,
        None => return Err(SnapshotError::SnapshotRead),
    };
    let size = file
        .seek(SeekFrom::End(0))
        .map_err(|_| SnapshotError::SnapshotRead)? as usize;
    if size != total {
        return Err(SnapshotError::SnapshotRead);
    }
    if !get(file, payload, header.len()) {
        return Err(SnapshotError::SnapshotRead);
    }
    let mut stored = [0u8; 32];
    if !get(file, &mut stored, length) {
        return Err(SnapshotError::SnapshotRead);
    }
    let mut hash = Sha256::new();
    hash.update(header);
    hash.update(&*payload);
    if stored != hash.finalize()[..] {
        return Err(SnapshotError::SnapshotRead);
    }
    Ok(())
}

fn put(file: &mut File, bytes: &[u8]) -> Result<(), SnapshotError> {
    file.write_all(bytes).map_err(|_| SnapshotError::SnapshotWrite)
}

fn get(file: &mut File, bytes: &mut [u8], at: usize) -> bool {
    if file.seek(SeekFrom::Start(at as u64)).is_err() {
        return false;
    }
    file.read_exact(bytes).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `payload bit flips, header changes, truncation,
    /// legacy files and extra bytes are refused` (snapshot_file.zig:28).
    #[test]
    fn payload_bit_flips_header_changes_truncation_legacy_files_and_extra_bytes_are_refused() {
        let name = format!("/tmp/tf-snapshot-test-{}", std::process::id());
        use std::os::unix::fs::OpenOptionsExt;
        let mut file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&name)
            .expect("the test file");
        let mut output = [0u8; 4];
        write(&mut file, b"head", b"body").unwrap();
        read(&mut file, b"head", &mut output).unwrap();
        assert_eq!(&output, b"body");
        assert_eq!(
            Err(SnapshotError::SnapshotRead),
            read(&mut file, b"dead", &mut output)
        );
        file.seek(SeekFrom::Start(5)).unwrap();
        file.write_all(b"X").unwrap();
        assert_eq!(
            Err(SnapshotError::SnapshotRead),
            read(&mut file, b"head", &mut output)
        );
        file.set_len(8).unwrap();
        assert_eq!(
            Err(SnapshotError::SnapshotRead),
            read(&mut file, b"head", &mut output)
        );
        file.seek(SeekFrom::Start(0)).unwrap();
        write(&mut file, b"head", b"body").unwrap();
        put(&mut file, b"x").unwrap();
        assert_eq!(
            Err(SnapshotError::SnapshotRead),
            read(&mut file, b"head", &mut output)
        );
        std::fs::remove_file(&name).ok();
    }
}
