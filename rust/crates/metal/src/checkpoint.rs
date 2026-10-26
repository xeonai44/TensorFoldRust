//! The shared safetensors index read into Metal buffers: one shared buffer a shard,
//! each tensor at its data offset.
//!
//! Port of `zig/src/core/checkpoint_metal.zig`. Weights live in GPU-visible memory
//! from load on: each shard's selected data region is `pread` into one Metal buffer
//! on 8 threads (upstream `readParallel`), and every tensor is a view
//! (buffer, offset, bytes, dtype, shape) — no repacking, exactly like upstream.
//! A family selects one tensor namespace before interpreting shapes (upstream
//! `addFileSelected`'s `selected`), so a text engine never admits a vision tower's
//! layouts.

use crate::device::{Buffer, Gpu};
use std::collections::HashMap;
use std::path::Path;
use tensorfold_safetensors::{parse_header_prefix, DType, Error as StError, MAX_RANK};

#[derive(Debug)]
pub enum Error {
    /// The shard file cannot open.
    OpenFailed(std::path::PathBuf),
    /// The header is not shaped as expected.
    BadSafetensors,
    UnsupportedDType,
    RankTooHigh,
    BadShape,
    BadOffsets,
    BadHeaderLength,
    NotUtf8,
    /// A shard names no tensors in the selected namespace: nothing to load.
    NoSelectedTensors,
    /// The same tensor name arrived twice.
    DuplicateTensor,
    /// A short read from the file.
    ShortRead,
    /// The checkpoint holds no tensor of that name.
    MissingTensor(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::OpenFailed(path) => write!(f, "cannot open {}", path.display()),
            Error::BadSafetensors => StError::BadSafetensors.fmt(f),
            Error::UnsupportedDType => StError::UnsupportedDType.fmt(f),
            Error::RankTooHigh => StError::RankTooHigh.fmt(f),
            Error::BadShape => StError::BadShape.fmt(f),
            Error::BadOffsets => StError::BadOffsets.fmt(f),
            Error::BadHeaderLength => StError::BadHeaderLength.fmt(f),
            Error::NotUtf8 => StError::NotUtf8.fmt(f),
            Error::NoSelectedTensors => {
                f.write_str("the shard holds no tensors of the selected namespace")
            }
            Error::DuplicateTensor => f.write_str("the same tensor name arrived twice"),
            Error::ShortRead => f.write_str("a short read from the shard file"),
            Error::MissingTensor(name) => write!(f, "checkpoint has no tensor {}", name),
        }
    }
}

impl std::error::Error for Error {}

impl From<StError> for Error {
    fn from(err: StError) -> Error {
        match err {
            StError::BadSafetensors => Error::BadSafetensors,
            StError::UnsupportedDType => Error::UnsupportedDType,
            StError::RankTooHigh => Error::RankTooHigh,
            StError::BadShape => Error::BadShape,
            StError::BadOffsets => Error::BadOffsets,
            StError::BadHeaderLength => Error::BadHeaderLength,
            StError::NotUtf8 => Error::NotUtf8,
            StError::Io => Error::ShortRead,
        }
    }
}

/// One tensor: a view into a shard buffer.
///
/// Upstream: `checkpoint_metal.zig:7  pub const Tensor = struct { buffer, offset,
/// bytes, dtype, shape, rank }`. Cloning retains the buffer.
#[derive(Clone)]
pub struct Tensor {
    pub buffer: Buffer,
    pub offset: usize,
    pub bytes: usize,
    pub dtype: DType,
    pub shape: [usize; MAX_RANK],
    pub rank: usize,
}

impl Tensor {
    /// Upstream: `count` — the element count.
    pub fn count(&self) -> usize {
        self.bytes / self.dtype.size()
    }

    /// The tensor's bytes as host memory (shared storage).
    ///
    /// Upstream: `host(self, comptime T) []const T`.
    ///
    /// Safety: the caller must not write through the returned slice while a command
    /// buffer that reads the tensor is in flight; the forwards wait for completion.
    pub fn host<T: Copy>(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                (self.buffer.as_ptr().add(self.offset)) as *const T,
                self.bytes / std::mem::size_of::<T>(),
            )
        }
    }
}

struct Shard {
    /// The shard's storage, held for ownership: every tensor view keeps its own
    /// retain, and the shard keeps the original alive until the checkpoint drops.
    #[allow(dead_code)]
    buffer: Buffer,
    bytes: usize,
}

/// Safetensors read into one buffer a shard from its data start.
///
/// Upstream: `checkpoint_metal.zig:20  pub const Checkpoint`.
pub struct Checkpoint {
    shards: Vec<Shard>,
    tensors: HashMap<String, Tensor>,
}

impl Default for Checkpoint {
    fn default() -> Self {
        Checkpoint::new()
    }
}

impl Checkpoint {
    pub fn new() -> Checkpoint {
        Checkpoint {
            shards: Vec::new(),
            tensors: HashMap::new(),
        }
    }

    /// Read `path`'s tensors, each named `prefix ++ name` (upstream `addFile`).
    pub fn add_file(&mut self, gpu: &Gpu, path: &Path, prefix: &str) -> Result<(), Error> {
        self.add_file_selected(gpu, path, prefix, None)
    }

    /// Load only `selected`'s tensor namespace and its contiguous data region; names
    /// retain the caller's prefix.
    ///
    /// Upstream: `checkpoint_metal.zig:38  addFileSelected` — the region is the
    /// selected tensors' [min begin, max end) span, read from the file at
    /// `data_start + begin` on 8 pread threads.
    pub fn add_file_selected(
        &mut self,
        gpu: &Gpu,
        path: &Path,
        prefix: &str,
        selected: Option<&str>,
    ) -> Result<(), Error> {
        let file = std::fs::File::open(path).map_err(|_| Error::OpenFailed(path.to_path_buf()))?;
        let file_len = file
            .metadata()
            .map_err(|_| Error::OpenFailed(path.to_path_buf()))?
            .len() as usize;

        let mut head = [0u8; 8];
        use std::os::unix::fs::FileExt;
        file.read_exact_at(&mut head, 0)
            .map_err(|_| Error::ShortRead)?;
        let header_len = u64::from_le_bytes(head) as usize;
        if 8 + header_len > file_len {
            return Err(Error::BadHeaderLength);
        }
        let mut header = vec![0u8; header_len];
        file.read_exact_at(&mut header, 8)
            .map_err(|_| Error::ShortRead)?;
        let header = String::from_utf8(header).map_err(|_| Error::NotUtf8)?;
        let data_start = 8 + header_len;
        let data_len = file_len - data_start;

        let entries = parse_header_prefix(&header, data_len, selected)?;
        if entries.is_empty() {
            return Ok(()); // upstream: "if (entries.count() == 0) return"
        }
        let mut begin = if selected.is_some() { data_len } else { 0 };
        let mut limit = if selected.is_some() { 0 } else { data_len };
        for (_, e) in &entries {
            begin = begin.min(e.start);
            limit = limit.max(e.end);
        }
        let bytes = limit - begin;
        let mut buffer = gpu.buffer_with_options(bytes.max(16), Gpu::shared_untracked());
        read_parallel(&file, buffer.as_mut_ptr_bytes(), (data_start + begin) as u64)?;
        self.shards.push(Shard { buffer: buffer.clone(), bytes });
        for (name, e) in &entries {
            let mut shape = [1usize; MAX_RANK];
            shape[..e.shape.len().min(MAX_RANK)].copy_from_slice(&e.shape);
            let tensor = Tensor {
                buffer: buffer.clone(),
                offset: e.start - begin,
                bytes: e.end - e.start,
                dtype: e.dtype,
                shape,
                rank: e.shape.len(),
            };
            let name = format!("{}{}", prefix, name);
            if self.tensors.insert(name.clone(), tensor).is_some() {
                return Err(Error::DuplicateTensor);
            }
        }
        Ok(())
    }

    /// Upstream: `get` — a tensor by name; missing names are an error naming it.
    pub fn get(&self, name: &str) -> Result<Tensor, Error> {
        self.tensors
            .get(name)
            .cloned()
            .ok_or_else(|| Error::MissingTensor(name.to_string()))
    }

    /// Upstream: `has`.
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Hold a buffer made after loading as a shard: the checkpoint frees it.
    /// Upstream: `own`.
    pub fn own(&mut self, buffer: Buffer, bytes: usize) {
        self.shards.push(Shard { buffer: buffer.clone(), bytes });
    }

    /// Bytes held in shard buffers. Upstream: `residentBytes`.
    pub fn resident_bytes(&self) -> usize {
        self.shards.iter().map(|s| s.bytes).sum()
    }
}

/// pread `dest.len` bytes from `at` on 8 threads (page cache or disk, the copy is the
/// load's main cost).
///
/// Upstream: `checkpoint_metal.zig:168  readParallel`.
fn read_parallel(file: &std::fs::File, dest: &mut [u8], at: u64) -> Result<(), Error> {
    use std::os::unix::fs::FileExt;
    const N_THREADS: usize = 8;
    let chunk = dest.len().div_ceil(N_THREADS);
    let mut failed = false;
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut rest: &mut [u8] = dest;
        let mut base: usize = 0;
        for _ in 0..N_THREADS {
            let take = chunk.min(rest.len());
            let (head, tail) = rest.split_at_mut(take);
            rest = tail;
            let lo = base;
            base += take;
            if take == 0 {
                continue;
            }
            handles.push(scope.spawn(move || {
                // read_exact_at is pread-until-exact, upstream's readAll loop.
                file.read_exact_at(head, at + lo as u64).is_ok()
            }));
        }
        for handle in handles {
            if !handle.join().unwrap_or(false) {
                failed = true;
            }
        }
    });
    if failed {
        Err(Error::ShortRead)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors checkpoint_metal.zig:184: full and selected views retain data,
    /// ownership and relative offsets.
    #[test]
    fn full_and_selected_views_retain_data_and_offsets() {
        let Some(gpu) = Gpu::open() else {
            eprintln!("skipping: no Metal device");
            return;
        };
        let dir = std::env::temp_dir().join("tf-checkpoint-metal-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let header = r#"{"padding":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},
 "text.weight":{"dtype":"U32","shape":[2,3],"data_offsets":[4,28]},
 "text.scales":{"dtype":"BF16","shape":[1],"data_offsets":[28,30]}}"#;
        let mut data = Vec::with_capacity(8 + header.len() + 30);
        data.extend_from_slice(&(header.len() as u64).to_le_bytes());
        data.extend_from_slice(header.as_bytes());
        for i in 0..30u8 {
            data.push(i);
        }
        let path = dir.join("fixture.safetensors");
        std::fs::write(&path, &data).unwrap();

        let mut full = Checkpoint::new();
        full.add_file(&gpu, &path, "").unwrap();
        let mut selected = Checkpoint::new();
        selected.add_file_selected(&gpu, &path, "", Some("text.")).unwrap();

        assert_eq!(full.resident_bytes(), 30);
        assert_eq!(selected.resident_bytes(), 26);
        let a = full.get("text.weight").unwrap();
        let b = selected.get("text.weight").unwrap();
        assert_eq!(a.offset, 4);
        assert_eq!(b.offset, 0);
        assert_eq!(a.host::<u32>(), b.host::<u32>());
        assert_eq!(
            full.get("text.scales").unwrap().host::<u16>(),
            selected.get("text.scales").unwrap().host::<u16>()
        );
        assert!(!selected.has("padding"));

        // The same namespace twice is a duplicate; an absent namespace is fine.
        selected
            .add_file_selected(&gpu, &path, "", Some("text."))
            .unwrap_err();
        selected
            .add_file_selected(&gpu, &path, "", Some("absent."))
            .unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }
}
