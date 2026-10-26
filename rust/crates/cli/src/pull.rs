//! The `pull` command: a Hugging Face checkpoint into the same cache 0.6.6
//! uses, verified.
//!
//! Port of `zig/src/cli/pull.zig` (279 lines) + `pull_file.zig` (152) — the
//! correct single-stream download with per-file sha256 verification, placed in
//! the standard cache layout (blobs + snapshot symlinks + refs/main). The
//! 16-way piecewise ranged download (pull_parts.zig) is upstream's speed
//! optimization; the correctness contract — every file verified before it
//! becomes a blob — is the same here, noted in the map as deferred.

use crate::hub;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

/// The error lines the route prints; the shape mirrors upstream's own error
/// paths (pull.zig:30-77).
pub struct PullError {
    pub message: String,
    pub code: u8,
}

/// Downloads `repo[@revision]` into the cache and prints where it landed.
/// Upstream: `run`/`runWith` (pull.zig:22).
pub fn run(out: &mut dyn Write, spec: &str) -> u8 {
    match try_run(out, spec) {
        Ok(code) => code,
        Err(e) => {
            let _ = writeln!(out, "{}", e.message);
            e.code
        }
    }
}

fn err(message: String) -> PullError {
    PullError { message, code: 1 }
}

fn try_run(out: &mut dyn Write, spec: &str) -> Result<u8, PullError> {
    let (repo, revision) = match spec.find('@') {
        Some(i) => (&spec[..i], &spec[i + 1..]),
        None => (spec, "main"),
    };
    if !hub::is_repo_id_like(repo) {
        return Err(PullError {
            message: format!("{} is not a Hugging Face repo id (owner/name)", spec),
            code: 2,
        });
    }
    let endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
    let root = hub::cache_dir();

    let sha = resolve_revision(&endpoint, repo, revision)
        .map_err(err)?;
    let entries = list_tree(&endpoint, repo, &sha).map_err(err)?;

    // The family check runs on config.json before any weight moves (pull.zig:59).
    let Some(_config) = entries.iter().find(|e| e.path == "config.json") else {
        return Err(err(format!("{}: the hub tree has no config.json; refusing", repo)));
    };
    let config_bytes = fetch_bytes(&endpoint, repo, &sha, "config.json")
        .map_err(|message| err(format!("{}: {}", repo, message)))?;
    let config: serde_json::Value = serde_json::from_slice(&config_bytes)
        .map_err(|_| err(format!("{}: config.json is not JSON; refusing", repo)))?;
    let model_type = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    if let Some(family) = hub::family(model_type) {
        let _ = writeln!(out, "{}: {} ({})", repo, family.title, family.model_type);
    } else {
        return Err(err(format!(
            "{}: no registered Zig family serves model_type {}; the 0.6 line may: tensorfold@0.6",
            repo, model_type
        )));
    }

    let repo_dir = root.join(hub::repo_dir_name(repo).ok_or_else(|| {
        err(format!("{} is not a Hugging Face repo id (owner/name)", repo))
    })?);
    let blobs_dir = repo_dir.join("blobs");
    let snapshot_dir = repo_dir.join("snapshots").join(&sha);
    let refs_dir = repo_dir.join("refs");
    fs::create_dir_all(&blobs_dir)
        .map_err(|e| err(format!("cannot create {}: {}", blobs_dir.display(), e)))?;
    fs::create_dir_all(&snapshot_dir)
        .map_err(|e| err(format!("cannot create {}: {}", snapshot_dir.display(), e)))?;
    fs::create_dir_all(&refs_dir)
        .map_err(|e| err(format!("cannot create {}: {}", refs_dir.display(), e)))?;

    // The large files as pieces on one queue; each matches its sha256
    // before it becomes a blob, else it comes once more (pull.zig:100).
    let mut big: Vec<Big> = Vec::new();
    for entry in &entries {
        let blob_path = blobs_dir.join(&entry.blob);
        // The blob exists with the right size: cached, link and continue.
        if let Ok(meta) = fs::metadata(&blob_path) {
            if meta.len() == entry.size {
                let _ = writeln!(out, "  {}: cached", entry.path);
                link_into_snapshot(&snapshot_dir, &entry.path, &entry.blob).map_err(err)?;
                continue;
            }
        }
        let url = format!("{}/{}/resolve/{}/{}", endpoint, repo, sha, entry.path);
        if entry.sha256.is_some() && entry.size >= crate::pull_parts::THRESHOLD {
            // The pieces land in a staging file beside the blob
            // (pull.zig:101's ".ranges")
            let staged = format!("{}.ranges", blob_path.display());
            big.push(Big {
                url,
                entry: TreeEntry {
                    path: entry.path.clone(),
                    size: entry.size,
                    blob: entry.blob.clone(),
                    sha256: entry.sha256.clone(),
                },
                staged,
            });
            continue;
        }
        download_retrying(&url, &blobs_dir, out, entry)
            .map_err(|message| err(format!("  {}: {}", entry.path, message)))?;
        link_into_snapshot(&snapshot_dir, &entry.path, &entry.blob).map_err(err)?;
    }
    if !fetch_big(out, big, &blobs_dir, &snapshot_dir)? {
        return Ok(1);
    }
    fs::write(refs_dir.join("main"), &sha)
        .map_err(|e| err(format!("cannot write refs/main: {}", e)))?;
    let _ = writeln!(out, "stored {}@{} at {}", repo, sha, snapshot_dir.display());
    Ok(0)
}

/// One file in the hub's tree. Upstream: pull_file.zig's `Entry`.
struct TreeEntry {
    path: String,
    size: u64,
    blob: String,
    sha256: Option<String>,
}

/// The large files as pieces: each matches its sha256 before it becomes
/// a blob, else it comes once more; a hub that answers a range with the
/// whole file falls back to one stream a file (pull.zig:117's fetchBig).
/// One large entry: its url, its tree entry, and the staging file the
/// pieces land in (pull.zig:100's Big).
struct Big {
    url: String,
    entry: TreeEntry,
    staged: String,
}

/// The large files as pieces: each matches its sha256 before it becomes
/// a blob, else it comes once more; a hub that answers a range with the
/// whole file falls back to one stream a file (pull.zig:117's fetchBig).
fn fetch_big(
    out: &mut dyn Write,
    big: Vec<Big>,
    blobs_dir: &std::path::Path,
    snapshot_dir: &std::path::Path,
) -> Result<bool, PullError> {
    if big.is_empty() {
        return Ok(true);
    }
    let files: Vec<crate::pull_parts::File> = big
        .iter()
        .map(|b| crate::pull_parts::File {
            url: b.url.clone(),
            path: b.staged.clone(),
            size: b.entry.size,
        })
        .collect();
    if let Err(problem) = crate::pull_parts::fetch(files, crate::pull_parts::PIECE, Some(out)) {
        match problem {
            // The hub sent whole files: one stream each (pull.zig:122)
            crate::pull_parts::FetchError::RangeUnsupported => {
                for b in &big {
                    let file = crate::pull_parts::File {
                        url: b.url.clone(),
                        path: b.staged.clone(),
                        size: b.entry.size,
                    };
                    crate::pull_parts::discard(&file, crate::pull_parts::PIECE);
                    download_retrying(&b.url, blobs_dir, out, &b.entry)
                        .map_err(|message| err(format!("  {}: {}", b.entry.path, message)))?;
                    link_into_snapshot(snapshot_dir, &b.entry.path, &b.entry.blob)
                        .map_err(err)?;
                }
                return Ok(true);
            }
            crate::pull_parts::FetchError::Failed(message) => {
                return Err(err(format!(
                    "  the download failed ({}); pull again to fetch only the missing pieces",
                    message
                )));
            }
        }
    }
    for b in &big {
        let file = crate::pull_parts::File {
            url: b.url.clone(),
            path: b.staged.clone(),
            size: b.entry.size,
        };
        // Each matches its sha256 before it becomes a blob, else it comes
        // once more (pull.zig:140-150)
        let matches = match crate::pull_parts::digest(&b.staged, b.entry.size) {
            Ok(got) => b
                .entry
                .sha256
                .as_ref()
                .and_then(|want| hex_to_bytes(want))
                .is_some_and(|want| got == want),
            Err(_) => false,
        };
        if !matches {
            crate::pull_parts::discard(&file, crate::pull_parts::PIECE);
            let one = vec![file.clone()];
            crate::pull_parts::fetch(one, crate::pull_parts::PIECE, Some(out))
                .map_err(|problem| match problem {
                    crate::pull_parts::FetchError::RangeUnsupported => err(format!(
                        "  {}: the hub stopped answering ranges",
                        b.entry.path
                    )),
                    crate::pull_parts::FetchError::Failed(message) => {
                        err(format!("  {}: {}", b.entry.path, message))
                    }
                })?;
            let again = match crate::pull_parts::digest(&b.staged, b.entry.size) {
                Ok(got) => b
                    .entry
                    .sha256
                    .as_ref()
                    .and_then(|want| hex_to_bytes(want))
                    .is_some_and(|want| got == want),
                Err(_) => false,
            };
            if !again {
                crate::pull_parts::discard(&file, crate::pull_parts::PIECE);
                return Err(err(format!(
                    "  {}: the bytes do not match the hub's sha256 twice; refusing",
                    b.entry.path
                )));
            }
        }
        // The staged whole becomes the blob (pull.zig:152's rename)
        let blob_path = blobs_dir.join(&b.entry.blob);
        fs::rename(&b.staged, &blob_path)
            .map_err(|e| err(format!("  {}: {}", b.entry.path, e)))?;
        let _ = writeln!(
            out,
            "  {}: {:.2} MiB in {} pieces",
            b.entry.path,
            b.entry.size as f64 / (1 << 20) as f64,
            crate::pull_parts::count(b.entry.size, crate::pull_parts::PIECE)
        );
        link_into_snapshot(snapshot_dir, &b.entry.path, &b.entry.blob).map_err(err)?;
    }
    Ok(true)
}

/// The hub's hex sha256 as raw bytes (the digest's own shape).
fn hex_to_bytes(hex: &str) -> Option<[u8; 32]> {
    let mut out = [0u8; 32];
    let bytes = hex.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = (bytes[i * 2] as char).to_digit(16)?;
        let lo = (bytes[i * 2 + 1] as char).to_digit(16)?;
        *slot = (hi * 16 + lo) as u8;
    }
    Some(out)
}

/// The single-stream download with upstream's retry shape (pull_file.zig:26):
/// five attempts with the wait doubling, a sha mismatch restarting once.
fn download_retrying(
    url: &str,
    blobs_dir: &Path,
    out: &mut dyn Write,
    entry: &TreeEntry,
) -> Result<(), String> {
    let attempts = 5;
    let mut wait_ms: u64 = 250;
    let mut restarted = false;
    let mut attempt = 1;
    let partial = blobs_dir.join(format!("{}.incomplete", entry.blob));
    loop {
        match download_one(url, blobs_dir, out, entry) {
            Ok(()) => return Ok(()),
            Err(message) => {
                let mismatch = message.contains("sha256 mismatch");
                if attempt == attempts || (mismatch && restarted) {
                    return Err(message);
                }
                if mismatch {
                    restarted = true;
                    let _ = fs::remove_file(&partial);
                }
                std::thread::sleep(std::time::Duration::from_millis(wait_ms));
                wait_ms *= 2;
                attempt += 1;
            }
        }
    }
}

/// Resolve `revision` (a branch, tag or commit) to the commit sha.
/// Upstream: pull.zig's `resolveRevision`.
fn resolve_revision(endpoint: &str, repo: &str, revision: &str) -> Result<String, String> {
    let url = format!("{}/api/models/{}/revision/{}", endpoint, repo, revision);
    let body = http_get(&url)?;
    let value: serde_json::Value =
        serde_json::from_str(&body).map_err(|_| "the hub's reply is not JSON".to_string())?;
    value
        .get("sha")
        .and_then(serde_json::Value::as_str)
        .filter(|sha| hub::hex_sha(sha))
        .map(str::to_string)
        .ok_or_else(|| {
            format!(
                "{}@{}: not a revision Hugging Face names (a branch, tag or 40-hex commit)",
                repo, revision
            )
        })
}

/// The hub's file tree. Upstream: pull.zig's `listTree` (the paginated
/// tree API).
fn list_tree(endpoint: &str, repo: &str, sha: &str) -> Result<Vec<TreeEntry>, String> {
    // The hub's tree API: one page covers every checkpoint the port serves
    // today (the model trees are within a page); upstream paginates via the
    // Link header (pull.zig's listTree), a TODO when the tree grows past one
    // page.
    let url = format!("{}/api/models/{}/tree/{}?recursive=true", endpoint, repo, sha);
    let body = http_get(&url)?;
    let value: serde_json::Value =
        serde_json::from_str(&body).map_err(|_| "the hub's tree is not JSON".to_string())?;
    let list = value
        .as_array()
        .ok_or_else(|| "the hub's tree is not a list".to_string())?;
    let mut entries = Vec::with_capacity(list.len());
    for file in list {
        let Some(path) = file.get("path").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let size = file.get("size").and_then(serde_json::Value::as_u64).unwrap_or(0);
        let lfs = file.get("lfs");
        let (blob, sha256) = match lfs {
            Some(lfs) => (
                lfs.get("oid").and_then(serde_json::Value::as_str),
                lfs.get("oid").and_then(serde_json::Value::as_str),
            ),
            None => (file.get("oid").and_then(serde_json::Value::as_str), None),
        };
        let Some(blob) = blob else { continue };
        entries.push(TreeEntry {
            path: path.to_string(),
            size,
            blob: blob.to_string(),
            sha256: sha256.map(str::to_string),
        });
    }
    if entries.is_empty() {
        return Err("the hub tree is empty".into());
    }
    Ok(entries)
}

/// Download to `path` and verify the file's hash before it becomes a blob.
/// Upstream: pull_file.zig's `downloadRetrying` — every file checks out or the
/// download fails; nothing unverified is ever linked.
fn fetch_bytes(endpoint: &str, repo: &str, sha: &str, path: &str) -> Result<Vec<u8>, String> {
    let url = format!("{}/{}/resolve/{}/{}", endpoint, repo, sha, path);
    let mut buf = Vec::new();
    let mut sink = |chunk: &[u8]| -> Result<(), String> {
        buf.extend_from_slice(chunk);
        Ok(())
    };
    http_get_into(&url, &mut sink)?;
    Ok(buf)
}

/// One file from the hub into ``blobs`` with HTTP range resume, size and
/// sha256 checks. Upstream: `download` (pull_file.zig:36) — the LFS files
/// resume into their `.incomplete` file (the on-disk prefix feeding the
/// same digest), a hub that ignores the range restarting from zero; the
/// small files check git's blob sha1 (``sha1("blob SIZE\0" + bytes)``).
fn download_one(
    url: &str,
    blobs_dir: &Path,
    out: &mut dyn Write,
    entry: &TreeEntry,
) -> Result<(), String> {
    use std::os::unix::fs::FileExt;
    let final_path = blobs_dir.join(&entry.blob);
    let partial_path = blobs_dir.join(format!("{}.incomplete", entry.blob));
    let mut resume_from: u64 = 0;
    if entry.sha256.is_some() {
        if let Ok(meta) = fs::metadata(&partial_path) {
            resume_from = meta.len();
        }
    }
    if resume_from >= entry.size {
        resume_from = 0;
    }
    let mut request = ureq::get(url).set("Accept-Encoding", "identity");
    if resume_from > 0 {
        request = request.set("Range", &format!("bytes={resume_from}-"));
    }
    let response = request
        .call()
        .map_err(|e| format!("the download failed ({e})"))?;
    let status = response.status();
    if status != 200 && status != 206 {
        return Err(format!("the hub answered {status}"));
    }
    let mut restart = false;
    if resume_from > 0 && status != 206 {
        // The hub ignored the range; start over (pull_file.zig:58)
        restart = true;
        resume_from = 0;
    }
    // Read access too: a resumed blob feeds its on-disk prefix into the
    // same digest (pull_file.zig:62)
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(restart)
        .open(&partial_path)
        .map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let check_git = entry.sha256.is_none();
    let mut git = sha1::Sha1::new();
    if check_git {
        git.update(format!("blob {}\0", entry.size).as_bytes());
    }
    if resume_from > 0 {
        // The prefix on disk feeds the same digest in chunks: the check
        // spans the whole file, nothing allocated whole (pull_file.zig:70)
        let mut prefix = [0u8; 64 << 10];
        let mut at: u64 = 0;
        while at < resume_from {
            let want = (prefix.len() as u64).min(resume_from - at) as usize;
            let read = file.read_at(&mut prefix[..want], at).unwrap_or(0);
            if read == 0 {
                break;
            }
            hasher.update(&prefix[..read]);
            at += read as u64;
        }
    }
    let mut reader = response.into_reader();
    let mut chunk = [0u8; 32 << 10];
    let mut offset = resume_from;
    loop {
        let n = reader.read(&mut chunk).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&chunk[..n]);
        if check_git {
            git.update(&chunk[..n]);
        }
        file.write_all_at(&chunk[..n], offset)
            .map_err(|e| e.to_string())?;
        offset += n as u64;
    }
    let total = offset;
    if total != entry.size {
        return Err(format!(
            "size mismatch: expected {} bytes, got {}",
            entry.size, total
        ));
    }
    let digest: [u8; 32] = hasher.finalize().into();
    if let Some(want) = &entry.sha256 {
        if let Some(want) = hex_to_bytes(want) {
            if digest != want {
                return Err("sha256 mismatch".into());
            }
        } else {
            return Err("the hub's sha256 is not hex".into());
        }
    } else if let Some(want) = hex_sha1_to_bytes(&entry.blob) {
        if git.finalize()[..] != want[..] {
            return Err("sha256 mismatch".into());
        }
    }
    file.set_len(total).map_err(|e| e.to_string())?;
    let _ = file.sync_all();
    drop(file);
    fs::rename(&partial_path, &final_path).map_err(|e| e.to_string())?;
    let _ = writeln!(
        out,
        "  {}: {:.2} MiB",
        entry.path,
        total as f64 / (1 << 20) as f64
    );
    Ok(())
}

/// The hub's 40-hex git sha1 as raw bytes (the non-LFS check's own shape).
fn hex_sha1_to_bytes(hex: &str) -> Option<[u8; 20]> {
    let bytes = hex.as_bytes();
    if bytes.len() != 40 {
        return None;
    }
    let mut out = [0u8; 20];
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = (bytes[i * 2] as char).to_digit(16)?;
        let lo = (bytes[i * 2 + 1] as char).to_digit(16)?;
        *slot = (hi * 16 + lo) as u8;
    }
    Some(out)
}

/// A snapshot's path a hub tree names: relative, and no traversal
/// segments. Upstream: `snapshotPathOk` (pull_file.zig:118).
fn snapshot_path_ok(path: &str) -> bool {
    if path.is_empty() || path.starts_with('/') {
        return false;
    }
    !path
        .split('/')
        .any(|c| c == "." || c == "..")
}

/// The snapshot's path points at the blob, the way huggingface_hub's
/// cache links them: a relative symlink into blobs. Upstream:
/// `linkIntoSnapshot` (pull_file.zig:127).
fn link_into_snapshot(snapshot_dir: &Path, path: &str, blob_name: &str) -> Result<(), String> {
    if !snapshot_path_ok(path) {
        return Err(format!("{path}: not a path the cache links"));
    }
    let target = snapshot_dir.join(path);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {}", parent.display(), e))?;
    }
    let _ = fs::remove_file(&target);
    let link_target = format!("../../blobs/{blob_name}");
    #[cfg(unix)]
    let linked: Result<(), String> = std::os::unix::fs::symlink(&link_target, &target)
        .map_err(|e| format!("cannot link {}: {}", target.display(), e));
    #[cfg(not(unix))]
    let linked: Result<(), String> = fs::copy(
        snapshot_dir.join("../blobs").join(blob_name),
        &target,
    )
    .map(|_| ())
    .map_err(|e| format!("cannot copy {}: {}", target.display(), e));
    linked?;
    Ok(())
}

fn http_get(url: &str) -> Result<String, String> {
    let mut buf = Vec::new();
    let mut sink = |chunk: &[u8]| -> Result<(), String> {
        buf.extend_from_slice(chunk);
        Ok(())
    };
    http_get_into(url, &mut sink)?;
    String::from_utf8(buf).map_err(|_| "the hub's reply is not UTF-8".into())
}

fn http_get_into(url: &str, sink: &mut dyn FnMut(&[u8]) -> Result<(), String>) -> Result<(), String> {
    let response = ureq::get(url).call().map_err(|e| e.to_string())?;
    let mut reader = response.into_reader();
    let mut buffer = [0u8; 256 << 10];
    loop {
        let n = reader.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        sink(&buffer[..n])?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::{TcpListener, TcpStream};

    /// A small HTTP server over an in-memory file: a range answered 206
    /// (or 200 when `range_ok` is false), the requests recorded — the
    /// single-stream download's live source.
    struct Server {
        thread: Option<std::thread::JoinHandle<()>>,
        port: u16,
        data: Vec<u8>,
        requests: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    impl Drop for Server {
        fn drop(&mut self) {
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    fn server(data: Vec<u8>, range_ok: bool) -> Server {
        let served_data = std::sync::Arc::new(data.clone());
        let requests = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen = std::sync::Arc::clone(&requests);
        let thread = std::thread::spawn(move || {
            let mut idle_ms: u32 = 0;
            let mut served = 0;
            'serve: loop {
                match listener.accept() {
                    Ok((stream, _)) => {
                        idle_ms = 0;
                        served += 1;
                        let data = std::sync::Arc::clone(&served_data);
                        let seen = std::sync::Arc::clone(&seen);
                        std::thread::spawn(move || serve(stream, &data, &seen, range_ok));
                        if served > 32 {
                            break 'serve;
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        idle_ms += 10;
                        if served > 0 && idle_ms > 800 {
                            break 'serve;
                        }
                        if idle_ms > 10_000 {
                            break 'serve;
                        }
                    }
                    Err(_) => break 'serve,
                }
            }
        });
        Server {
            thread: Some(thread),
            port,
            data,
            requests,
        }
    }

    fn serve(
        mut stream: TcpStream,
        data: &[u8],
        seen: &std::sync::Mutex<Vec<String>>,
        range_ok: bool,
    ) {
        use std::io::Write as _;
        let mut reader = BufReader::new(&mut stream);
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).is_err() {
            return;
        }
        let mut range: Option<(u64, Option<u64>)> = None;
        loop {
            let mut header = String::new();
            match reader.read_line(&mut header) {
                Ok(0) => return,
                Ok(_) if header.trim().is_empty() => break,
                Ok(_) => {
                    if let Some(value) = header
                        .strip_prefix("Range: ")
                        .or_else(|| header.strip_prefix("range: "))
                    {
                        let value = value.trim().strip_prefix("bytes=").unwrap_or(value.trim());
                        let mut parts = value.splitn(2, '-');
                        let from: u64 = parts.next().unwrap_or("0").parse().unwrap_or(0);
                        let to = parts
                            .next()
                            .and_then(|t| t.trim().parse::<u64>().ok());
                        range = Some((from, to));
                    }
                }
                Err(_) => return,
            }
        }
        seen.lock()
            .unwrap()
            .push(range.map(|(f, _)| f.to_string()).unwrap_or_default());
        let (from, slice): (u64, &[u8]) = match range {
            Some((from, _)) if range_ok => {
                let at = from.min(data.len() as u64) as usize;
                (from, &data[at..])
            }
            _ => (0, data),
        };
        let head = if range.is_some() && range_ok {
            format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {}-{}/{}\r\nContent-Length: {}\r\n\r\n",
                from,
                from + slice.len() as u64 - 1,
                data.len(),
                slice.len()
            )
        } else {
            format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                slice.len()
            )
        };
        let _ = stream.write_all(head.as_bytes());
        let _ = stream.write_all(slice);
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("tf-pull-file-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A tree entry with the LFS sha256 over `data`.
    fn lfs_entry(path: &str, blob: &str, data: &[u8]) -> TreeEntry {
        TreeEntry {
            path: path.into(),
            size: data.len() as u64,
            blob: blob.into(),
            sha256: Some(format!("{:x}", Sha256::digest(data))),
        }
    }

    /// The whole download: the file verified by sha256, renamed into
    /// blobs, no `.incomplete` left.
    #[test]
    fn the_whole_download_verifies_and_renames() {
        let data: Vec<u8> = (0..=255u8).cycle().take(100_000).collect();
        let server = server(data.clone(), true);
        let dir = scratch("whole");
        let blobs = dir.join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        let entry = lfs_entry("model.safetensors", "aaaa", &data);
        let url = format!("http://127.0.0.1:{}/model", server.port);
        let mut out = Vec::new();
        download_one(&url, &blobs, &mut out, &entry).expect("the download");
        let blob = blobs.join("aaaa");
        assert_eq!(fs::read(&blob).unwrap(), data, "the blob's bytes");
        assert!(
            !blobs.join("aaaa.incomplete").exists(),
            "the staging file renamed away"
        );
        assert!(
            String::from_utf8_lossy(&out).contains("model.safetensors: 0.10 MiB"),
            "the MiB line: {}",
            String::from_utf8_lossy(&out)
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// The range resume: a `.incomplete` file's prefix feeds the same
    /// digest, one request with `Range: bytes=N-`, and the bytes after it
    /// complete the file (pull_file.zig:50-76).
    #[test]
    fn the_incomplete_prefix_resumes() {
        let data: Vec<u8> = (0..=255u8).cycle().take(100_000).collect();
        let server = server(data.clone(), true);
        let dir = scratch("resume");
        let blobs = dir.join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        let entry = lfs_entry("model.safetensors", "bbbb", &data);
        // The interrupted pull left 37,000 whole bytes behind
        fs::write(blobs.join("bbbb.incomplete"), &data[..37_000]).unwrap();
        let url = format!("http://127.0.0.1:{}/model", server.port);
        let mut out = Vec::new();
        download_one(&url, &blobs, &mut out, &entry).expect("the resumed download");
        assert_eq!(
            server.requests.lock().unwrap().as_slice(),
            &["37000".to_string()],
            "one request with the range: {:?}",
            server.requests.lock().unwrap()
        );
        assert_eq!(fs::read(blobs.join("bbbb")).unwrap(), data);
        let _ = fs::remove_dir_all(&dir);
    }

    /// A hub that ignores the range: the download starts over
    /// (pull_file.zig:58's restart).
    #[test]
    fn a_rangeless_hub_restarts() {
        let data: Vec<u8> = (0..=255u8).cycle().take(100_000).collect();
        let server = server(data.clone(), false);
        let dir = scratch("restart");
        let blobs = dir.join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        let entry = lfs_entry("model.safetensors", "cccc", &data);
        fs::write(blobs.join("cccc.incomplete"), &data[..40_000]).unwrap();
        let url = format!("http://127.0.0.1:{}/model", server.port);
        let mut out = Vec::new();
        download_one(&url, &blobs, &mut out, &entry).expect("the restarted download");
        assert_eq!(fs::read(blobs.join("cccc")).unwrap(), data);
        let _ = fs::remove_dir_all(&dir);
    }

    /// The small files check git's blob sha1 (pull_file.zig:84-90: sha1 of
    /// "blob SIZE\0" and the bytes).
    #[test]
    fn the_small_file_checks_the_git_blob_sha1() {
        let data = b"hello world\n".to_vec();
        let server = server(data.clone(), true);
        let dir = scratch("git");
        let blobs = dir.join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        let mut git = sha1::Sha1::new();
        git.update(format!("blob {}\0", data.len()).as_bytes());
        git.update(&data);
        let blob = format!("{:x}", git.finalize());
        let entry = TreeEntry {
            path: "config.json".into(),
            size: data.len() as u64,
            blob: blob.clone(),
            sha256: None,
        };
        let url = format!("http://127.0.0.1:{}/config", server.port);
        let mut out = Vec::new();
        download_one(&url, &blobs, &mut out, &entry).expect("the git-verified download");
        assert_eq!(fs::read(blobs.join(&blob)).unwrap(), data);
        // A wrong sha refuses
        let wrong = TreeEntry {
            path: "config.json".into(),
            size: data.len() as u64,
            blob: "0".repeat(40),
            sha256: None,
        };
        let err = download_one(&url, &blobs, &mut out, &wrong).unwrap_err();
        assert_eq!(err, "sha256 mismatch");
        let _ = fs::remove_dir_all(&dir);
    }

    /// A size mismatch refuses (pull_file.zig:82).
    #[test]
    fn a_short_body_refuses() {
        let data = b"abc".to_vec();
        let server = server(data.clone(), true);
        let dir = scratch("short");
        let blobs = dir.join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        let entry = TreeEntry {
            path: "x".into(),
            size: 100,
            blob: "dddd".into(),
            sha256: Some("e".repeat(64)),
        };
        let url = format!("http://127.0.0.1:{}/x", server.port);
        let mut out = Vec::new();
        let err = download_one(&url, &blobs, &mut out, &entry).unwrap_err();
        assert!(err.starts_with("size mismatch"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }

    /// Upstream's table (pull_file.zig:150): a hub tree path stays inside
    /// the snapshot — relative, without `.` or `..` segments.
    #[test]
    fn a_hub_tree_path_stays_inside_the_snapshot() {
        for p in ["config.json", "deep/nested/model.1-of-2.safetensors", "a..b/c"] {
            assert!(snapshot_path_ok(p), "{p}");
        }
        for p in ["", "/etc/passwd", "../outside", "a/../b", "./a", "a/."] {
            assert!(!snapshot_path_ok(p), "{p}");
        }
    }

    /// The snapshot link is the relative one huggingface_hub's cache uses
    /// (pull_file.zig:135): `../../blobs/NAME`, working wherever the
    /// cache moves.
    #[test]
    fn the_snapshot_link_is_relative() {
        let dir = scratch("link");
        let snapshot = dir.join("snapshots").join("abc");
        fs::create_dir_all(&snapshot).unwrap();
        link_into_snapshot(&snapshot, "model/model.safetensors", "deadbeef").expect("the link");
        let link = snapshot.join("model/model.safetensors");
        assert_eq!(
            std::fs::read_link(&link).unwrap(),
            std::path::Path::new("../../blobs/deadbeef"),
            "the relative target"
        );
        // A traversal path refuses
        assert!(link_into_snapshot(&snapshot, "../outside", "x").is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
