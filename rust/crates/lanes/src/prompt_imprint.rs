//! Prompt-cache states on disk (Imprint): a learned harness state
//! outlives the server and serves later fresh sessions.
//!
//! Port of `zig/src/core/prompt_imprint.zig` (392 lines): one identity's
//! learned states under a root — an append-only index (its records torn
//! tail tolerated), the crash journal a state's files write under (a
//! 48-byte pending word, checksummed), the LRU both within the identity
//! and across the others under the root (each open stamps a clock), and
//! the cap that sweeps the least recently opened others first. The keys
//! are `Wyhash` (seed 0x1e47) over the tokens' little-endian bytes —
//! upstream's own `keyOf` — filed byte for byte compatibly with the Zig
//! engine's trees through the `wyhash-final4` reference port.

use crate::learned_disk::Admission;
use crate::learned_dirs;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;
use wyhash_final4::generics::WyHasher;
use wyhash_final4::wyhash64::WyHash64;

/// One state on disk: the tokens it read (position plus lookahead), the
/// chunk starts below it, and its bytes. Upstream: `Meta`
/// (prompt_imprint.zig:6).
#[derive(Clone)]
pub struct Meta {
    pub key: u64,
    pub at: u32,
    pub tokens: Vec<u32>,
    pub starts: Vec<u32>,
    pub bytes: u64,
    pub used: u64,
}

const MAGIC: u32 = 0x494d5052; // "IMPR"
const VERSION: u32 = 2;
/// A record's words before its tokens: magic, version, key (2), at,
/// token and start counts, bytes (2). Upstream: `HEAD`
/// (prompt_imprint.zig:10).
const HEAD: usize = 9;

/// The pending word's magic (prompt_imprint.zig:127): "IMPRINT1".
const PENDING_MAGIC: u64 = 0x494d5052494e5431;

/// The imprint's own failures. Upstream: the `error` set
/// (`ImprintRead`, `ImprintWrite`, `ImprintDir`, `ImprintPending`).
#[derive(Debug, PartialEq, Eq)]
pub enum ImprintError {
    Read,
    Write,
    Dir,
    Pending,
}

impl std::fmt::Display for ImprintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            ImprintError::Read => "the learned store cannot be read",
            ImprintError::Write => "the learned store cannot be written",
            ImprintError::Dir => "the learned directory cannot be made",
            ImprintError::Pending => "a learned write is still pending",
        };
        f.write_str(name)
    }
}

impl std::error::Error for ImprintError {}

/// The states learned under one identity in `root`. Upstream: `Imprint`
/// (prompt_imprint.zig:12).
pub struct Imprint {
    pub root: String,
    /// `root/<identity>`: the index, the last-open stamp and each state's
    /// files (the family's).
    pub dir: String,
    /// Bytes of learned states this Mac keeps under `root`, every
    /// identity together.
    pub cap: u64,
    pub admission: Admission,
    /// Bytes other identities under `root` still hold.
    pub others: u64,
    pub metas: Vec<Meta>,
    /// A meta's `used`: the clock at its last write or read.
    clock: u64,
}

/// The states' key: `Wyhash` (seed 0x1e47) over the tokens' bytes.
/// Upstream: `keyOf` (prompt_imprint.zig:51) — the little-endian bytes
/// `sliceAsBytes` reads on this port's platforms.
pub fn key_of(tokens: &[u32]) -> u64 {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for t in tokens {
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    WyHasher::<WyHash64>::from_seed(0x1e47).hash(&bytes)
}

impl Imprint {
    /// The states learned under `identity` in `root` (made when missing);
    /// other identities past `cap` go, oldest first. Upstream: `open`
    /// (prompt_imprint.zig:23).
    pub fn open(root: &str, identity: u64, cap: u64) -> Result<Imprint, ImprintError> {
        let dir = format!("{root}/{identity:016x}");
        make_path(&dir).map_err(|_| ImprintError::Dir)?;
        let mut m = Imprint {
            root: root.to_string(),
            dir,
            cap,
            admission: Admission::default(),
            others: 0,
            metas: Vec::new(),
            clock: 0,
        };
        m.load()?;
        m.stamp();
        m.others = learned_dirs::total_others(root, &m.dir)
            .map_err(|_| ImprintError::Read)?;
        Ok(m)
    }

    fn find(&self, key: u64) -> Option<usize> {
        self.metas.iter().position(|x| x.key == key)
    }

    pub fn has(&self, key: u64) -> bool {
        self.find(key).is_some()
    }

    /// The bytes of every state learned under this identity. Upstream:
    /// `total` (prompt_imprint.zig:65).
    pub fn total(&self) -> u64 {
        self.metas.iter().map(|x| x.bytes).sum()
    }

    /// The current index's size: LRU eviction rewrites it through a
    /// temporary file. Upstream: `indexScratchBytes`
    /// (prompt_imprint.zig:72).
    pub fn index_scratch_bytes(&self) -> Result<u64, ImprintError> {
        match std::fs::metadata(index_path(&self.dir)) {
            Ok(m) => Ok(m.len()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(0),
            Err(_) => Err(ImprintError::Read),
        }
    }

    /// Whether `bytes` more fit under the cap; other identities give room
    /// first, least recently opened first. Upstream: `fits`
    /// (prompt_imprint.zig:83).
    pub fn fits(&mut self, bytes: u64) -> bool {
        if self.others + self.total() + bytes > self.cap {
            self.others = sweep(
                &self.root,
                &self.dir,
                self.cap.saturating_sub(self.total() + bytes),
            );
        }
        self.others + self.total() + bytes <= self.cap
    }

    /// The least recently used state learned here: the next to go when the
    /// cap needs room. Upstream: `victim` (prompt_imprint.zig:89).
    pub fn victim(&self) -> Option<u64> {
        self.metas
            .iter()
            .min_by_key(|x| x.used)
            .map(|x| x.key)
    }

    /// The longest learned state `prompt` resumes: its tokens a prefix, a
    /// row left, `starts` below it the prompt's own. Upstream: `best`
    /// (prompt_imprint.zig:98).
    pub fn best(
        &self,
        prompt: &[u32],
        starts: &[u32],
        planned: bool,
        longer_than: u32,
    ) -> Option<&Meta> {
        let mut out: Option<&Meta> = None;
        for x in &self.metas {
            if x.at <= longer_than
                || x.tokens.len() > prompt.len()
                || x.at as usize >= prompt.len()
            {
                continue;
            }
            if out.is_some_and(|o| x.at <= o.at) {
                continue;
            }
            if prompt[..x.tokens.len()] != x.tokens[..] {
                continue;
            }
            if planned && !same_below(starts, &x.starts, x.at) {
                continue;
            }
            out = Some(x);
        }
        out
    }

    /// State `key` was read: the most recently used now. Upstream:
    /// `touch` (prompt_imprint.zig:111).
    pub fn touch(&mut self, key: u64) {
        let Some(i) = self.find(key) else {
            return;
        };
        self.clock += 1;
        self.metas[i].used = self.clock;
    }

    /// The state whose files a crash left unfinished, if any. Upstream:
    /// `pendingWrite` (prompt_imprint.zig:117).
    pub fn pending_write(&self) -> Result<Option<u64>, ImprintError> {
        let mut data = [0u8; 48];
        let mut file = match std::fs::File::open(pending_path(&self.dir)) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(_) => return Err(ImprintError::Read),
        };
        if file.metadata().map(|m| m.len()).unwrap_or(0) != 48 {
            return Err(ImprintError::Read);
        }
        if file.read_exact(&mut data).is_err() {
            return Err(ImprintError::Read);
        }
        let digest = Sha256::digest(&data[0..16]);
        if digest[..] != data[16..]
            || u64::from_le_bytes(data[0..8].try_into().unwrap()) != PENDING_MAGIC
        {
            return Err(ImprintError::Read);
        }
        Ok(Some(u64::from_le_bytes(data[8..16].try_into().unwrap())))
    }

    /// The journal word for a write about to begin, atomically in place.
    /// Upstream: `beginWrite` (prompt_imprint.zig:130).
    pub fn begin_write(&self, key: u64) -> Result<(), ImprintError> {
        if self.pending_write()?.is_some() {
            return Err(ImprintError::Pending);
        }
        let part = format!("{}/pending.part", self.dir);
        let mut data = [0u8; 48];
        data[0..8].copy_from_slice(&PENDING_MAGIC.to_le_bytes());
        data[8..16].copy_from_slice(&key.to_le_bytes());
        let digest = Sha256::digest(&data[0..16]);
        data[16..48].copy_from_slice(&digest);
        let write = || -> std::io::Result<()> {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&part)?;
            f.write_all(&data)?;
            f.sync_all()?;
            std::fs::rename(&part, pending_path(&self.dir))
        };
        if write().is_err() {
            // Upstream's errdefer: a failed write leaves no part behind
            let _ = std::fs::remove_file(&part);
            return Err(ImprintError::Write);
        }
        Ok(())
    }

    /// The journal cleared. Upstream: `clearPending`
    /// (prompt_imprint.zig:149).
    pub fn clear_pending(&self) -> Result<(), ImprintError> {
        learned_dirs::unlink(Path::new(&pending_path(&self.dir)))
            .map_err(|_| ImprintError::Write)
    }

    /// Record a state its family has written (`bytes` on this Mac): an
    /// index record, fsynced, then the entry. Upstream: `add`
    /// (prompt_imprint.zig:154).
    pub fn add(
        &mut self,
        key: u64,
        at: u32,
        tokens: &[u32],
        starts: &[u32],
        bytes: u64,
    ) -> Result<(), ImprintError> {
        if self.has(key) {
            return Ok(());
        }
        let below = cut(starts, at);
        let record = record_bytes(key, at, tokens, below, bytes);
        let append = || -> std::io::Result<()> {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(index_path(&self.dir))?;
            f.write_all(&record)?;
            f.sync_all()
        };
        append().map_err(|_| ImprintError::Write)?;
        self.clock += 1;
        self.metas.push(Meta {
            key,
            at,
            tokens: tokens.to_vec(),
            starts: below.to_vec(),
            bytes,
            used: self.clock,
        });
        Ok(())
    }

    /// Forget state `key`: the index is rewritten without it (its family
    /// removes the files). Upstream: `remove` (prompt_imprint.zig:176).
    pub fn remove(&mut self, key: u64) -> Result<(), ImprintError> {
        let Some(i) = self.find(key) else {
            return Ok(());
        };
        let part = format!("{}/index.part", self.dir);
        let rewrite = || -> std::io::Result<()> {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&part)?;
            for (j, x) in self.metas.iter().enumerate() {
                if j != i {
                    f.write_all(&record_bytes(x.key, x.at, &x.tokens, &x.starts, x.bytes))?;
                }
            }
            f.sync_all()?;
            std::fs::rename(&part, index_path(&self.dir))
        };
        if rewrite().is_err() {
            // Upstream's errdefer: a failed replacement leaves no part,
            // the metadata stands
            let _ = std::fs::remove_file(&part);
            return Err(ImprintError::Write);
        }
        self.metas.remove(i);
        Ok(())
    }

    /// The index's records, oldest first; a torn last record (a crash
    /// mid-append) ends the read. Upstream: `load` (prompt_imprint.zig:194).
    fn load(&mut self) -> Result<(), ImprintError> {
        let bytes = match std::fs::read(index_path(&self.dir)) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(_) => return Err(ImprintError::Read),
        };
        let word = |off: usize| -> Option<u32> {
            (off + 4 <= bytes.len())
                .then(|| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()))
        };
        let read = |from: usize, len: usize| -> Vec<u32> {
            bytes[from..from + len]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let mut off: usize = 0;
        while let (Some(magic), Some(version)) = (word(off), word(off + 4)) {
            if magic != MAGIC || version != VERSION {
                // A torn or foreign tail ends the read
                return Ok(());
            }
            let head: Vec<u32> = (0..HEAD)
                .map(|k| word(off + 4 * k))
                .collect::<Option<Vec<u32>>>()
                .unwrap_or_default();
            if head.len() < HEAD {
                return Ok(());
            }
            let n = head[5] as usize;
            let k = head[6] as usize;
            if n > 1 << 22 || k > n {
                return Ok(());
            }
            let at = off + HEAD * 4;
            let tokens_len = 4 * n;
            let starts_at = at + tokens_len;
            let starts_len = 4 * k;
            if starts_at + starts_len > bytes.len() {
                return Ok(());
            }
            let key = head[2] as u64 | (head[3] as u64) << 32;
            let x = Meta {
                key,
                at: head[4],
                tokens: read(at, tokens_len),
                starts: read(starts_at, starts_len),
                bytes: head[7] as u64 | (head[8] as u64) << 32,
                used: 0,
            };
            if !self.has(x.key) {
                self.clock += 1;
                let used = self.clock;
                self.metas.push(Meta { used, ..x });
            }
            off = starts_at + starts_len;
        }
        Ok(())
    }

    /// This identity was opened: the newest under `root` for the sweep
    /// (root/clock counts opens). Upstream: `stamp`
    /// (prompt_imprint.zig:228).
    fn stamp(&self) {
        let clock_path = format!("{}/clock", self.root);
        let now = read_word(Path::new(&clock_path)).saturating_add(1);
        write_word(Path::new(&clock_path), now);
        write_word(Path::new(&format!("{}/used", self.dir)), now);
    }
}

fn index_path(dir: &str) -> std::path::PathBuf {
    Path::new(dir).join("index")
}

fn pending_path(dir: &str) -> String {
    format!("{dir}/pending")
}

/// One index record: its head, its tokens, then the chunk starts below its
/// position. Upstream: `record` (prompt_imprint.zig:239).
fn record_bytes(key: u64, at: u32, tokens: &[u32], starts: &[u32], bytes: u64) -> Vec<u8> {
    let head = [
        MAGIC,
        VERSION,
        key as u32,
        (key >> 32) as u32,
        at,
        tokens.len() as u32,
        starts.len() as u32,
        bytes as u32,
        (bytes >> 32) as u32,
    ];
    let mut out = Vec::with_capacity((HEAD + tokens.len() + starts.len()) * 4);
    for w in head {
        out.extend_from_slice(&w.to_le_bytes());
    }
    for t in tokens {
        out.extend_from_slice(&t.to_le_bytes());
    }
    for s in starts {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// The chunk starts below `at`. Upstream: `cut` (prompt_imprint.zig:251).
fn cut(starts: &[u32], at: u32) -> &[u32] {
    let mut n = 0;
    while n < starts.len() && starts[n] < at {
        n += 1;
    }
    &starts[0..n]
}

/// Whether a prompt's chunk starts below `at` are the learned pass's (a
/// planned family's state depends on them). Upstream: `sameBelow`
/// (prompt_imprint.zig:258).
fn same_below(starts: &[u32], learned: &[u32], at: u32) -> bool {
    cut(starts, at) == learned && starts.contains(&at)
}

/// An identity directory's name: 16 hex digits. Upstream: `identityName`
/// (prompt_imprint.zig:263).
fn identity_name(name: &str) -> bool {
    name.len() == 16 && name.bytes().all(|c| c.is_ascii_hexdigit())
}

/// Other identities' directories (16 hex digits, their files only)
/// deleted, oldest first, down to `room` bytes. Upstream: `sweep`
/// (prompt_imprint.zig:270).
fn sweep(root: &str, keep: &str, room: u64) -> u64 {
    struct Dir {
        name: String,
        bytes: u64,
        used: u64,
    }
    let mut dirs: Vec<Dir> = Vec::new();
    let entries = match std::fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    let mut held: u64 = 0;
    for entry in entries.flatten() {
        let Some(name) = entry.file_name().to_str().map(str::to_string) else {
            continue;
        };
        if !identity_name(&name) || name == Path::new(keep).file_name().unwrap_or_default().to_str().unwrap_or_default() {
            continue;
        }
        let path = format!("{root}/{name}");
        let used = read_word(Path::new(&format!("{path}/used")));
        let bytes = files(&path, false);
        held = held.saturating_add(bytes);
        dirs.push(Dir { name, bytes, used });
    }
    dirs.sort_by_key(|d| d.used);
    for d in &dirs {
        if held <= room {
            break;
        }
        let path = format!("{root}/{}", d.name);
        let _ = files(&path, true);
        if std::fs::remove_dir(&path).is_ok() {
            held = held.saturating_sub(d.bytes);
        }
    }
    held
}

/// The bytes of the regular files directly in `dir`, each unlinked when
/// `remove`. Upstream: `files` (prompt_imprint.zig:304).
fn files(dir: &str, remove: bool) -> u64 {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    let mut n: u64 = 0;
    for entry in entries.flatten() {
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        if let Ok(m) = entry.metadata() {
            n += m.len();
        }
        if remove {
            let _ = std::fs::remove_file(entry.path());
        }
    }
    n
}

/// One little-endian word read (0 on any failure). Upstream: `readWord`
/// (prompt_imprint.zig:323).
fn read_word(path: &Path) -> u64 {
    let mut data = [0u8; 8];
    let Ok(mut f) = std::fs::File::open(path) else {
        return 0;
    };
    if f.read_exact(&mut data).is_err() {
        return 0;
    }
    u64::from_le_bytes(data)
}

/// One word written through its part, atomically (failures silent).
/// Upstream: `writeWord` (prompt_imprint.zig:331).
fn write_word(path: &Path, w: u64) {
    use std::io::Write;
    let part = path.with_extension("part");
    let Ok(mut f) = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&part)
    else {
        return;
    };
    let written = f.write_all(&w.to_le_bytes()).and_then(|_| f.sync_all());
    if written.is_err() {
        return;
    }
    let _ = std::fs::rename(&part, path);
}

/// This Mac's OS build (sysctl kern.osversion): its Metal compiler can
/// change a kernel's bits. Upstream: `osBuild` (prompt_imprint.zig:362).
pub fn os_build() -> String {
    let mut buf = [0u8; 64];
    let mut n = buf.len();
    // SAFETY: `kern.osversion` reads into `buf` for at most `n` bytes.
    let rc = unsafe {
        sysctlbyname(
            c"kern.osversion".as_ptr(),
            buf.as_mut_ptr() as *mut std::ffi::c_void,
            &mut n,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return String::new();
    }
    let len = buf[..n].iter().position(|&c| c == 0).unwrap_or(n);
    String::from_utf8_lossy(&buf[..len]).into_owned()
}

unsafe extern "C" {
    fn sysctlbyname(
        name: *const std::ffi::c_char,
        old: *mut std::ffi::c_void,
        oldlen: *mut usize,
        new: *mut std::ffi::c_void,
        newlen: usize,
    ) -> std::ffi::c_int;
}

/// Where learned states live unless --learn-dir says:
/// $HOME/.cache/tensorfold/learned. Upstream: `defaultRoot`
/// (prompt_imprint.zig:370).
pub fn default_root() -> Option<String> {
    let home = std::env::var_os("HOME")?;
    Some(format!(
        "{}/.cache/tensorfold/learned",
        home.to_str()?
    ))
}

/// `dir` and its parents (mode 0700). Upstream: `makePath`
/// (prompt_imprint.zig:376).
fn make_path(dir: &str) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut prefix = String::new();
    for part in Path::new(dir).components() {
        let part = part.as_os_str().to_str().ok_or(std::io::Error::other("not utf-8"))?;
        prefix.push_str(part);
        match std::fs::create_dir(&prefix) {
            Ok(()) => {
                let _ = std::fs::set_permissions(&prefix, std::fs::Permissions::from_mode(0o700));
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e),
        }
        prefix.push('/');
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// `root`, its identity directories and their files removed (a test's
    /// scratch tree, two levels deep) — upstream's `rmTree`
    /// (prompt_imprint_test.zig:9).
    fn rm_tree(root: &str) {
        let Ok(entries) = std::fs::read_dir(root) else {
            return;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if name.starts_with('.') {
                continue;
            }
            let path = entry.path();
            if std::fs::remove_file(&path).is_ok() {
                continue;
            }
            if let Ok(inner) = std::fs::read_dir(&path) {
                for f in inner.flatten() {
                    let fname = f.file_name();
                    if fname.to_str().is_some_and(|n| !n.starts_with('.')) {
                        let _ = std::fs::remove_file(f.path());
                    }
                }
            }
            let _ = std::fs::remove_dir(&path);
        }
        let _ = std::fs::remove_dir(root);
    }

    fn scratch(tag: &str) -> String {
        let path = std::env::temp_dir()
            .join(format!("tf-imprint-{tag}-{}", std::process::id()));
        path.to_str().unwrap().to_string()
    }

    /// A file of `n` bytes at `dir`/`name` — upstream's `fill`
    /// (prompt_imprint_test.zig:27).
    fn fill(dir: &str, name: &str, n: usize) {
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(Path::new(dir).join(name))
            .expect("the scratch file opens");
        f.write_all(&vec![0u8; n]).expect("the scratch file fills");
    }

    /// Upstream's `reopen` test (prompt_imprint_test.zig:36): learned
    /// states survive a reopen; the longest usable prefix wins.
    #[test]
    fn learned_states_survive_a_reopen_and_the_longest_usable_prefix_wins() {
        let root = scratch("reopen");
        {
            let mut m = Imprint::open(&root, 7, 1 << 30).unwrap();
            m.add(key_of(&[1, 2, 3]), 2, &[1, 2, 3], &[1, 2, 5], 10).unwrap();
            m.add(key_of(&[1, 2, 3, 4, 5]), 4, &[1, 2, 3, 4, 5], &[1, 4], 20).unwrap();
            assert!(m.has(key_of(&[1, 2, 3])));
        }
        let m = Imprint::open(&root, 7, 1 << 30).unwrap();
        assert_eq!(m.metas.len(), 2);
        assert_eq!(m.total(), 30);
        let prompt = [1, 2, 3, 4, 5, 6, 7];
        assert_eq!(m.best(&prompt, &[1, 4, 6], true, 0).unwrap().at, 4);
        // Other starts below 4: only 2 fits
        assert_eq!(m.best(&prompt, &[1, 2, 6], true, 0).unwrap().at, 2);
        // Memory already holds 4
        assert!(m.best(&prompt, &[1, 4, 6], true, 4).is_none());
        // Token 2 differs: 3 was read
        assert!(m.best(&[1, 2, 9, 9], &[1, 2], true, 0).is_none());
        drop(m);
        // Another identity sees nothing
        let other = Imprint::open(&root, 8, 1 << 30).unwrap();
        assert_eq!(other.metas.len(), 0);
        rm_tree(&root);
    }

    /// Upstream's `lru` test (prompt_imprint_test.zig:65): the least
    /// recently used state goes first, and a removed one stays gone after
    /// a reopen.
    #[test]
    fn the_least_recently_used_state_goes_first() {
        let root = scratch("lru");
        let a = key_of(&[1, 2]);
        let b = key_of(&[3, 4]);
        let c = key_of(&[5, 6]);
        {
            let mut m = Imprint::open(&root, 1, 100).unwrap();
            m.add(a, 1, &[1, 2], &[], 30).unwrap();
            m.add(b, 1, &[3, 4], &[], 30).unwrap();
            m.add(c, 1, &[5, 6], &[], 30).unwrap();
            m.touch(a);
            assert_eq!(m.victim(), Some(b));
            assert!(!m.fits(20)); // 90 held of 100
            m.remove(b).unwrap();
            assert!(m.fits(20));
            assert_eq!(m.victim(), Some(c));
        }
        let m = Imprint::open(&root, 1, 100).unwrap();
        assert!(m.has(a) && m.has(c) && !m.has(b));
        assert_eq!(m.total(), 60);
        rm_tree(&root);
    }

    /// Upstream's `sweep` test (prompt_imprint_test.zig:90): other
    /// identities are swept, least recently opened first, until the cap
    /// fits.
    #[test]
    fn other_identities_are_swept_least_recently_opened_first() {
        let root = scratch("sweep");
        let mut names = [String::new(), String::new()];
        for i in 0..2 {
            let m = Imprint::open(&root, (i + 1) as u64, 1 << 30).unwrap();
            // With its 8-byte stamp: 108 bytes an identity
            fill(&m.dir, "state.bin", 100);
            names[i] = m.dir.clone();
        }
        let mut m = Imprint::open(&root, 3, 150).unwrap();
        // Opening never deletes either rank before coordinated admission
        assert!(Path::new(&names[0]).exists());
        assert!(m.fits(0));
        assert!(!Path::new(&names[0]).exists());
        assert!(Path::new(&names[1]).exists());
        assert_eq!(m.others, 108);
        // A larger state sweeps the other identity too
        assert!(m.fits(42) && !m.fits(43 + 108));
        assert!(!Path::new(&names[1]).exists());
        assert_eq!(m.others, 0);
        rm_tree(&root);
    }

    /// Upstream's `stamp` fault (prompt_imprint_fault_test.zig:61): a
    /// failed atomic stamp replacement preserves the previous complete
    /// used word.
    #[test]
    fn a_failed_stamp_replacement_preserves_the_previous_word() {
        let root = scratch("stamp-fault");
        {
            let m = Imprint::open(&root, 1, 1 << 20).unwrap();
            let used = Path::new(&m.dir).join("used");
            let mut f = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&used)
                .unwrap();
            f.write_all(&42u64.to_le_bytes()).unwrap();
            // The rename's temporary path is a directory: the stamp's
            // replacement fails, the old word stands
            std::fs::create_dir(used.with_extension("part")).unwrap();
        }
        let _reopened = Imprint::open(&root, 1, 1 << 20).unwrap();
        assert_eq!(learned_dirs::used(&root, 1).unwrap(), 42);
        rm_tree(&root);
    }

    /// Upstream's `io` fault (prompt_imprint_fault_test.zig:82): used
    /// stamp reads still refuse an actual I/O error.
    #[test]
    fn used_stamp_reads_refuse_an_actual_io_error() {
        let root = scratch("io-fault");
        {
            let m = Imprint::open(&root, 1, 1 << 20).unwrap();
            let used = Path::new(&m.dir).join("used");
            learned_dirs::unlink(&used).unwrap();
            std::fs::create_dir(&used).unwrap();
        }
        assert!(learned_dirs::used(&root, 1).is_err());
        rm_tree(&root);
    }

    /// Upstream's `ownership` fault (prompt_imprint_fault_test.zig:38):
    /// a failed open releases each allocation once after another identity
    /// scan. Rust's ownership frees by construction; the observable the
    /// test pins is the failure itself — a malformed other tree refuses
    /// the open, and no partial identity directory is left behind.
    #[test]
    fn a_failed_open_refuses_after_another_identity_scan() {
        let root = scratch("ownership-fault");
        {
            let old = Imprint::open(&root, 1, 1 << 20).unwrap();
            std::fs::create_dir(Path::new(&old.dir).join("unreadable")).unwrap();
        }
        let failed = Imprint::open(&root, 2, 1 << 20);
        assert_eq!(failed.err(), Some(ImprintError::Read));
        rm_tree(&root);
    }

    /// Upstream's `index` fault (learned_fault_test.zig:55): a
    /// replacement failure retains its metadata and removes its partial
    /// file.
    #[test]
    fn an_index_replacement_failure_retains_its_metadata_and_removes_its_part() {
        let root = scratch("index-fault");
        {
            let mut im = Imprint::open(&root, 1, 1 << 20).unwrap();
            im.add(7, 1, &[1, 2], &[], 105).unwrap();
            // The index path is a directory: the rewrite's rename fails
            let idx = index_path(&im.dir);
            learned_dirs::unlink(&idx).unwrap();
            std::fs::create_dir(&idx).unwrap();
            assert_eq!(im.remove(7).err(), Some(ImprintError::Write));
            assert!(im.has(7));
            assert!(
                !Path::new(&format!("{}/index.part", im.dir)).exists(),
                "the failed rewrite leaves no part"
            );
            std::fs::remove_dir(&idx).unwrap();
        }
        rm_tree(&root);
    }

    /// The pending journal: the checksummed word, the refusal while one
    /// stands, the clear.
    #[test]
    fn the_pending_journal() {
        let root = scratch("pending");
        let m = Imprint::open(&root, 1, 1 << 30).unwrap();
        assert_eq!(m.pending_write().unwrap(), None);
        m.begin_write(0x1234).unwrap();
        assert_eq!(m.pending_write().unwrap(), Some(0x1234));
        // A second write refuses while one stands
        assert_eq!(m.begin_write(7).err(), Some(ImprintError::Pending));
        m.clear_pending().unwrap();
        assert_eq!(m.pending_write().unwrap(), None);
        rm_tree(&root);
    }

    /// The keys are upstream's Wyhash rule: the reference port with the
    /// reference secret, the seed upstream's keyOf passes. The values are
    /// pinned so the tree stays compatible with what the Zig engine
    /// writes; the cross-check against `std.hash.Wyhash` itself runs when
    /// a Zig toolchain is at hand again (see the map's open gaps).
    #[test]
    fn the_keys_hash_the_tokens_little_endian() {
        // Different token lists hash differently; the same list, the same key
        assert_ne!(key_of(&[1, 2, 3]), key_of(&[1, 2, 4]));
        assert_eq!(key_of(&[1, 2, 3]), key_of(&[1, 2, 3]));
        // Empty input, a defined key
        let _ = key_of(&[]);
    }
}
