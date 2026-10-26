//! Large files as fixed pieces on one shared queue; a finished piece
//! leaves a marker, so a restart fetches the rest.
//!
//! Port of `zig/src/cli/pull_parts.zig` (182 lines): files at least
//! `THRESHOLD` come down in `PIECE`-sized ranges over up to `WORKERS`
//! connections — one piece resuming within itself after a reset or stall
//! (five attempts), its marker going down only once every byte is
//! written; a hub that answers a range with the whole file is
//! `RangeUnsupported`, the caller's single-stream fallback; the
//! piecewise bytes verified by the digest before they become a blob.

use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Files at least this large come down in pieces (pull_parts.zig:4).
pub const THRESHOLD: u64 = 64 << 20;
/// The bytes one request asks for (pull_parts.zig:5).
pub const PIECE: u64 = 32 << 20;
/// The connections at most (pull_parts.zig:6).
pub const WORKERS: usize = 16;
/// One piece's attempts (pull_parts.zig:7).
const ATTEMPTS: usize = 5;

/// One file: where its bytes come from, where they land, and how many
/// there are. Upstream: `File` (pull_parts.zig:10).
#[derive(Clone)]
pub struct File {
    pub url: String,
    pub path: String,
    pub size: u64,
}

struct Piece {
    file: usize,
    first: u64,
    end: u64,
    marker: String,
}

/// How many pieces of `bytes` each a file of `size` takes. Upstream:
/// `count` (pull_parts.zig:16).
pub fn count(size: u64, bytes: u64) -> u64 {
    size.div_ceil(bytes).max(1)
}

/// Why a piecewise fetch failed: `RangeUnsupported` hands the caller its
/// single-stream fallback (pull.zig:122).
#[derive(Debug)]
pub enum FetchError {
    RangeUnsupported,
    Failed(String),
}

struct Shared {
    files: Vec<File>,
    handles: Vec<std::sync::Mutex<fs_guard::FileAt>>,
    todo: Vec<Piece>,
    next: AtomicUsize,
    bytes: AtomicU64,
    running: AtomicU32,
    stop: AtomicBool,
}

/// The positional-write face the workers share (one handle a file, the
/// offset with each write).
mod fs_guard {
    use std::fs::File;
    use std::os::unix::fs::FileExt;
    pub struct FileAt(pub File);
    impl FileAt {
        pub fn write_at(&self, buf: &[u8], at: u64) -> std::io::Result<()> {
            self.0.write_all_at(buf, at)
        }
    }
}

fn marker(path: &str, i: u64) -> String {
    format!("{path}.part{i}")
}

/// `files` as `bytes`-sized pieces over up to 16 connections; a hub that
/// answers a range with the whole file gives `RangeUnsupported`. Upstream:
/// `fetch` (pull_parts.zig:31).
pub fn fetch(
    files: Vec<File>,
    bytes: u64,
    mut progress: Option<&mut dyn Write>,
) -> Result<(), FetchError> {
    use std::fs::OpenOptions;
    let mut handles = Vec::with_capacity(files.len());
    let mut todo: Vec<Piece> = Vec::new();
    let mut left: u64 = 0;
    for (k, f) in files.iter().enumerate() {
        // The file at its full size: the pieces land at their positions
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&f.path)
            .map_err(|e| FetchError::Failed(e.to_string()))?;
        file.set_len(f.size)
            .map_err(|e| FetchError::Failed(e.to_string()))?;
        handles.push(std::sync::Mutex::new(fs_guard::FileAt(file)));
        for i in 0..count(f.size, bytes) {
            let m = marker(&f.path, i);
            // A marker left by an earlier run: the piece is already whole
            if Path::new(&m).exists() {
                continue;
            }
            let end = (f.size).min((i + 1) * bytes);
            left += end - i * bytes;
            todo.push(Piece {
                file: k,
                first: i * bytes,
                end,
                marker: m,
            });
        }
    }
    let shared = Arc::new(Shared {
        files,
        handles,
        todo,
        next: AtomicUsize::new(0),
        bytes: AtomicU64::new(0),
        running: AtomicU32::new(0),
        stop: AtomicBool::new(false),
    });
    let n = WORKERS.min(shared.todo.len());
    shared.running.store(n as u32, Ordering::Release);
    let mut threads = Vec::with_capacity(n);
    let failures: Arc<Vec<std::sync::Mutex<Option<FetchError>>>> = Arc::new(
        (0..n)
            .map(|_| std::sync::Mutex::new(None))
            .collect(),
    );
    for i in 0..n {
        let shared = Arc::clone(&shared);
        let failures = Arc::clone(&failures);
        let thread = std::thread::Builder::new()
            .spawn(move || worker(shared, i, failures))
            .map_err(|e| FetchError::Failed(e.to_string()))?;
        threads.push(thread);
    }
    let start = std::time::Instant::now();
    let mut shown = start;
    while shared.running.load(Ordering::Acquire) > 0 {
        std::thread::sleep(std::time::Duration::from_millis(200));
        let now = std::time::Instant::now();
        if progress.is_none() || now.duration_since(shown).as_secs() < 10 {
            continue;
        }
        shown = now;
        let got = shared.bytes.load(Ordering::Relaxed) as f64;
        let secs = now.duration_since(start).as_secs_f64();
        let _ = writeln!(
            progress.as_mut().unwrap(),
            "  {:.2} of {:.2} GiB, {:.1} MiB/s",
            got / (1 << 30) as f64,
            left as f64 / (1 << 30) as f64,
            got / (1 << 20) as f64 / secs
        );
        let _ = progress.as_mut().unwrap().flush();
    }
    for thread in threads {
        let _ = thread.join();
    }
    for failure in failures.iter() {
        if let Some(err) = failure.lock().unwrap().take() {
            return Err(err);
        }
    }
    // The markers: every piece whole, the file whole
    for f in &shared.files {
        for i in 0..count(f.size, bytes) {
            let _ = std::fs::remove_file(marker(&f.path, i));
        }
    }
    Ok(())
}

/// Removes a file's bytes and piece markers, so the next fetch starts it
/// over. Upstream: `discard` (pull_parts.zig:75).
pub fn discard(file: &File, bytes: u64) {
    let _ = std::fs::remove_file(&file.path);
    for i in 0..count(file.size, bytes) {
        let _ = std::fs::remove_file(marker(&file.path, i));
    }
}

fn worker(
    s: Arc<Shared>,
    index: usize,
    failures: Arc<Vec<std::sync::Mutex<Option<FetchError>>>>,
) {
    let body = |s: &Arc<Shared>| -> Result<(), FetchError> {
        let agent = ureq::AgentBuilder::new().redirects(5).build();
        while !s.stop.load(Ordering::Acquire) {
            let i = s.next.fetch_add(1, Ordering::Relaxed);
            if i >= s.todo.len() {
                return Ok(());
            }
            get(s, &agent, &s.todo[i])?;
        }
        Ok(())
    };
    let result = body(&s);
    if let Err(e) = result {
        *failures[index].lock().unwrap() = Some(e);
        s.stop.store(true, Ordering::Release);
    }
    s.running.fetch_sub(1, Ordering::AcqRel);
}

/// One piece, resuming within it after a reset or stall; the marker goes
/// down once every byte is written. Upstream: `get` (pull_parts.zig:85).
fn get(s: &Shared, agent: &ureq::Agent, p: &Piece) -> Result<(), FetchError> {
    let mut done: u64 = 0;
    let mut last = FetchError::Failed("the piece never ran".into());
    for _ in 0..ATTEMPTS {
        match range(s, agent, p, &mut done) {
            Ok(()) => {
                std::fs::write(&p.marker, b"")
                    .map_err(|e| FetchError::Failed(e.to_string()))?;
                return Ok(());
            }
            Err(FetchError::RangeUnsupported) => return Err(FetchError::RangeUnsupported),
            Err(e) => last = e,
        }
    }
    Err(last)
}

/// One piece's remaining bytes as a range, written at their position.
/// Upstream: `range` (pull_parts.zig:95).
fn range(s: &Shared, agent: &ureq::Agent, p: &Piece, done: &mut u64) -> Result<(), FetchError> {
    let from = p.first + *done;
    if from >= p.end {
        return Ok(());
    }
    let response = agent
        .get(&s.files[p.file].url)
        .set("Range", &format!("bytes={}-{}", from, p.end - 1))
        .set("Accept-Encoding", "identity")
        .call()
        .map_err(|e| FetchError::Failed(e.to_string()))?;
    if response.status() != 206 {
        return Err(FetchError::RangeUnsupported);
    }
    // The first byte a 206 says it carries: Content-Range: bytes
    // FIRST-LAST/SIZE (pull_parts.zig:167's startOf)
    let start_of = response
        .header("Content-Range")
        .and_then(|v| v.trim().strip_prefix("bytes "))
        .and_then(|v| v.find('-').map(|dash| v[..dash].to_string()))
        .and_then(|first| first.parse::<u64>().ok());
    if start_of != Some(from) {
        return Err(FetchError::RangeUnsupported);
    }
    let mut reader = response.into_reader();
    let mut chunk = [0u8; 256 << 10];
    while p.first + *done < p.end {
        let want = (chunk.len() as u64).min(p.end - p.first - *done) as usize;
        let n = reader
            .read(&mut chunk[..want])
            .map_err(|_| FetchError::Failed("the read failed".into()))?;
        if n == 0 {
            return Err(FetchError::Failed("the range came up short".into()));
        }
        s.handles[p.file]
            .lock()
            .unwrap()
            .write_at(&chunk[..n], p.first + *done)
            .map_err(|e| FetchError::Failed(e.to_string()))?;
        *done += n as u64;
        s.bytes.fetch_add(n as u64, Ordering::Relaxed);
    }
    Ok(())
}

/// The sha256 of `path`'s first `size` bytes. Upstream: `digest`
/// (pull_parts.zig:172).
pub fn digest(path: &str, size: u64) -> Result<[u8; 32], String> {
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; 4 << 20];
    let mut hash = Sha256::new();
    let mut at: u64 = 0;
    while at < size {
        use std::io::Seek;
        use std::io::SeekFrom;
        file.seek(SeekFrom::Start(at)).map_err(|e| e.to_string())?;
        let want = (buf.len() as u64).min(size - at) as usize;
        let n = file.read(&mut buf[..want]).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("the file came up short".into());
        }
        hash.update(&buf[..n]);
        at += n as u64;
    }
    Ok(hash.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read as _, Write as _};
    use std::net::{TcpListener, TcpStream};

    /// A small HTTP server over an in-memory file: a range answered 206
    /// with Content-Range (the hub's behavior), or the whole file 200
    /// (the `RangeUnsupported` fallback's). The requests recorded so the
    /// tests can see what a restart fetched.
    struct RangeServer {
        thread: Option<std::thread::JoinHandle<()>>,
        port: u16,
        data: Arc<Vec<u8>>,
        requests: Arc<std::sync::Mutex<Vec<(u64, u64)>>>,
    }

    impl Drop for RangeServer {
        fn drop(&mut self) {
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    fn range_server(supports: bool) -> RangeServer {
        let data = Arc::new((0..=255u8).cycle().take(250_000).collect::<Vec<u8>>());
        let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener
            .set_nonblocking(true)
            .expect("the listener configures");
        let port = listener.local_addr().unwrap().port();
        let server_data = Arc::clone(&data);
        let seen = Arc::clone(&requests);
        let thread = std::thread::spawn(move || {
            let mut idle_ms: u32 = 0;
            let mut served = 0;
            'serve: loop {
                match listener.accept() {
                    Ok((stream, _)) => {
                        idle_ms = 0;
                        served += 1;
                        let data = Arc::clone(&server_data);
                        let seen = Arc::clone(&seen);
                        std::thread::spawn(move || serve(stream, &data, &seen, supports));
                        if served > 64 {
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
        RangeServer {
            thread: Some(thread),
            port,
            data,
            requests,
        }
    }

    fn serve(
        mut stream: TcpStream,
        data: &[u8],
        seen: &std::sync::Mutex<Vec<(u64, u64)>>,
        supports: bool,
    ) {
        let mut reader = BufReader::new(&mut stream);
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).is_err() {
            return;
        }
        let mut range: Option<(u64, u64)> = None;
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
                        let to: u64 = parts
                            .next()
                            .unwrap_or("")
                            .trim()
                            .parse()
                            .unwrap_or(data.len() as u64 - 1);
                        range = Some((from, to));
                    }
                }
                Err(_) => return,
            }
        }
        let Some((from, to)) = range else {
            // A plain GET: the whole file
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\n\r\n",
                data.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(data);
            return;
        };
        seen.lock().unwrap().push((from, to));
        if !supports {
            // The hub that ignores ranges: the whole file back
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\n\r\n",
                data.len()
            );
            let _ = stream.write_all(head.as_bytes());
            let _ = stream.write_all(data);
            return;
        }
        let first = from.min(data.len() as u64) as usize;
        let last_inclusive = to.min(data.len() as u64 - 1) as usize;
        let slice = &data[first..=last_inclusive];
        let head = format!(
            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {}-{}/{}\r\nContent-Length: {}\r\n\r\n",
            from,
            last_inclusive as u64,
            data.len(),
            slice.len()
        );
        let _ = stream.write_all(head.as_bytes());
        let _ = stream.write_all(slice);
    }

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("tf-pull-parts-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The pieces cover every byte once (pull_parts.zig:181's table).
    #[test]
    fn pieces_cover_every_byte_once() {
        assert_eq!(count(1, PIECE), 1);
        assert_eq!(count(128 << 20, PIECE), 4);
        assert_eq!(count((128 << 20) + 1, PIECE), 5);
        assert_eq!(count(5000, 1250), 4);
    }

    /// The piecewise fetch over the local range server: the file assembles
    /// byte-identical, the markers clean, and a second run with a marker
    /// standing fetches only the missing pieces.
    #[test]
    fn the_pieces_assemble_and_a_marker_skips_its_piece() {
        let server = range_server(true);
        let dir = scratch("assemble");
        let path = dir.join("big.bin");
        let file = File {
            url: format!("http://127.0.0.1:{}/big.bin", server.port),
            path: path.to_str().unwrap().to_string(),
            size: server.data.len() as u64,
        };
        let bytes = 50_000u64; // 5 pieces
        fetch(vec![file], bytes, None).expect("the piecewise fetch");
        let got = std::fs::read(&path).expect("the assembled file");
        assert_eq!(&got, server.data.as_slice(), "byte-identical");
        for i in 0..count(server.data.len() as u64, bytes) {
            assert!(
                !path.with_file_name(format!("big.bin.part{i}")).exists(),
                "marker {i} cleaned"
            );
        }
        // The digest over the assembled file
        let d = digest(path.to_str().unwrap(), server.data.len() as u64).unwrap();
        let want: [u8; 32] = Sha256::digest(server.data.as_slice()).into();
        assert_eq!(d, want);
        let first_run = server.requests.lock().unwrap().len();
        assert_eq!(first_run, 5, "one request a piece");

        // A restart with one piece already marked: only the rest fetch —
        // the file itself stays (upstream never truncates it; a restart
        // continues where the markers say the pieces stood)
        std::fs::write(path.with_file_name("big.bin.part2"), b"").unwrap();
        let file = File {
            url: format!("http://127.0.0.1:{}/big.bin", server.port),
            path: path.to_str().unwrap().to_string(),
            size: server.data.len() as u64,
        };
        server.requests.lock().unwrap().clear();
        fetch(vec![file], bytes, None).expect("the resuming fetch");
        let requests = server.requests.lock().unwrap().clone();
        assert_eq!(requests.len(), 4, "the marked piece skipped: {requests:?}");
        assert!(
            !requests.iter().any(|(from, _)| *from == 100_000),
            "piece 2 (at {} bytes) skipped",
            100_000
        );
        let got = std::fs::read(&path).expect("the resumed file");
        assert_eq!(&got, server.data.as_slice(), "the resumed file whole");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A hub that answers a range with the whole file: RangeUnsupported,
    /// the caller's single-stream fallback (pull.zig:122's arm).
    #[test]
    fn a_rangeless_hub_refuses() {
        let server = range_server(false);
        let dir = scratch("rangeless");
        let path = dir.join("big.bin");
        let file = File {
            url: format!("http://127.0.0.1:{}/big.bin", server.port),
            path: path.to_str().unwrap().to_string(),
            size: server.data.len() as u64,
        };
        let err = fetch(vec![file], 50_000, None).unwrap_err();
        assert!(matches!(err, FetchError::RangeUnsupported));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `discard` removes the file and its markers, so the next fetch
    /// starts over (pull_parts.zig:75).
    #[test]
    fn discard_restarts_the_file() {
        let dir = scratch("discard");
        let path = dir.join("big.bin");
        std::fs::write(&path, b"partial").unwrap();
        std::fs::write(path.with_file_name("big.bin.part0"), b"").unwrap();
        std::fs::write(path.with_file_name("big.bin.part1"), b"").unwrap();
        let file = File {
            url: "http://127.0.0.1/none".into(),
            path: path.to_str().unwrap().to_string(),
            size: 75_000,
        };
        discard(&file, 50_000);
        assert!(!path.exists());
        assert!(!path.with_file_name("big.bin.part0").exists());
        assert!(!path.with_file_name("big.bin.part1").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The digest of a short file refuses: nothing unverified ever becomes
    /// a blob (pull_parts.zig:172's ShortFile).
    #[test]
    fn a_short_file_refuses_its_digest() {
        let dir = scratch("short");
        let path = dir.join("short.bin");
        std::fs::write(&path, b"abc").unwrap();
        let err = digest(path.to_str().unwrap(), 4).unwrap_err();
        assert!(err.contains("short"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
