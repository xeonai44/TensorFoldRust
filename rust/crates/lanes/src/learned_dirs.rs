//! Bounded-name learned directories: both ranks plan the same identity
//! deletions before either writes.
//!
//! Port of `zig/src/core/lanes/learned_dirs.zig` (123 lines): the identity
//! directories under a learned root are 16 hex digits; `next` walks them
//! in id order, `bytes`/`otherBytes`/`totalOthers` the storage they hold,
//! `used` the recency hint their open stamped, `removeOther` the
//! acknowledged deletion. The `NOFOLLOW` discipline of every open is the
//! port's `std::fs` symlink policy: the learned tree is the server's own.

use std::io::Read;
use std::path::Path;

/// Upstream: `rootOf` (learned_dirs.zig:3) — std.fs.path.dirname
/// (null for "." and bare names) — the empty parent is the root too,
/// which Rust's `Path::parent(".")` returns as "" (not None): live-caught,
/// the pair tests' need() over "." failed with a NotFound root.
pub fn root_of(dir: &str) -> &str {
    match Path::new(dir).parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_str().unwrap_or("."),
        _ => ".",
    }
}

fn path_of(root: &str, id: u64) -> String {
    format!("{root}/{id:016x}")
}

fn is_current(keep: &str, id: u64) -> bool {
    Path::new(keep)
        .file_name()
        .map(|n| n == format!("{id:016x}").as_str())
        .unwrap_or(false)
}

/// The next identity directory after `after`, in id order (an error names
/// a malformed tree). Upstream: `next` (learned_dirs.zig:14).
pub fn next(root: &str, keep: &str, after: Option<u64>) -> std::io::Result<Option<u64>> {
    let mut out: Option<u64> = None;
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if name.len() != 16 {
            continue;
        }
        let Some(id) = u64::from_str_radix(name, 16).ok() else {
            continue;
        };
        if id == u64::MAX {
            return Err(std::io::Error::other("a learned directory is not one"));
        }
        if !entry.file_type()?.is_dir() {
            return Err(std::io::Error::other("a learned directory is not one"));
        }
        if is_current(keep, id) || after.is_some_and(|a| id <= a) {
            continue;
        }
        if out.is_none_or(|o| id < o) {
            out = Some(id);
        }
    }
    Ok(out)
}

/// One file unlinked, a missing one fine. Upstream: `unlink`
/// (learned_dirs.zig:32) — a directory refuses.
pub fn unlink(file: &Path) -> std::io::Result<()> {
    match std::fs::remove_file(file) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// The bytes of every regular file directly in `dir`. Upstream: `bytes`
/// (learned_dirs.zig:36) — anything else in the tree is a type error.
pub fn bytes(dir: &str) -> std::io::Result<u64> {
    let mut n: u64 = 0;
    // A missing directory holds nothing (upstream's `errno(-1) == .NOENT`
    // read after the failed opendir)
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        if name == "." || name == ".." {
            continue;
        }
        if !entry.file_type()?.is_file() {
            return Err(std::io::Error::other("a learned directory is not one"));
        }
        n = n.checked_add(entry.metadata()?.len())
            .ok_or(std::io::Error::other("the learned bytes overflow"))?;
    }
    Ok(n)
}

/// One file's size, 0 when missing. Upstream: `fileBytes`
/// (learned_dirs.zig:52).
pub fn file_bytes(file: &Path) -> std::io::Result<u64> {
    match std::fs::metadata(file) {
        Ok(m) => Ok(m.len()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(e),
    }
}

/// Another identity's bytes (0 for the current one). Upstream:
/// `otherBytes` (learned_dirs.zig:60).
pub fn other_bytes(root: &str, keep: &str, id: u64) -> std::io::Result<u64> {
    if is_current(keep, id) {
        return Ok(0);
    }
    bytes(&path_of(root, id))
}

/// Every other identity's bytes, in id order. Upstream: `totalOthers`
/// (learned_dirs.zig:66).
pub fn total_others(root: &str, keep: &str) -> std::io::Result<u64> {
    let mut cursor: Option<u64> = None;
    let mut n: u64 = 0;
    while let Some(id) = next(root, keep, cursor)? {
        n = n
            .checked_add(other_bytes(root, keep, id)?)
            .ok_or(std::io::Error::other("the learned bytes overflow"))?;
        cursor = Some(id);
    }
    Ok(n)
}

/// Every identity's bytes. Upstream: `total` (learned_dirs.zig:76).
pub fn total(root: &str) -> std::io::Result<u64> {
    total_others(root, "-")
}

/// An identity's recency hint: the word its open stamped. Upstream: `used`
/// (learned_dirs.zig:80) — a torn (short) read is the oldest, 0; an I/O
/// error refuses.
pub fn used(root: &str, id: u64) -> std::io::Result<u64> {
    let path = format!("{}/{id:016x}/used", root);
    let mut file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(e),
    };
    let mut data = [0u8; 8];
    match file.read_exact(&mut data) {
        Ok(()) => Ok(u64::from_le_bytes(data)),
        // A torn recency hint is oldest; it cannot make a learned payload valid
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(0),
        Err(e) => Err(e),
    }
}

/// Another identity's tree removed, its files first. Upstream:
/// `removeOther` (learned_dirs.zig:90) — the acknowledged deletion.
pub fn remove_other(root: &str, keep: &str, id: u64) -> std::io::Result<()> {
    if is_current(keep, id) {
        return Ok(());
    }
    let dir = path_of(root, id);
    // The tree's health first: a refused type error, not a partial
    // delete; a missing tree is already gone (upstream's empty Ok)
    bytes(&dir)?;
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        if name == "." || name == ".." {
            continue;
        }
        if !entry.file_type()?.is_file() {
            return Err(std::io::Error::other("a learned directory is not one"));
        }
        unlink(&entry.path())?;
    }
    match std::fs::remove_dir(&dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's test (learned_dirs.zig:121): acknowledged deletion
    /// refuses a directory and accepts an absent file.
    #[test]
    fn acknowledged_deletion_refuses_a_directory_and_accepts_an_absent_file() {
        let dir = std::env::temp_dir().join(format!("tf-unlink-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        assert!(unlink(&dir).is_err());
        assert!(unlink(&dir.join("absent")).is_ok());
        std::fs::remove_dir(&dir).unwrap();
    }
}
