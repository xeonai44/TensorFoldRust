//! A model folder's safetensors as one name space: shard files, tensors mapped,
//! leftovers refused.
//!
//! Port of `zig/src/core/checkpoint.zig` (the reading side; the `U8 image` writer at
//! checkpoint.zig:124 is a test helper there and returns with the engines that use it).

use tensorfold_safetensors::{DType, Error as StError, SafeTensors, Tensor};
use memmap2::Mmap;
use serde_json::Value;
use std::cell::RefCell;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum OpenError {
    /// The checkpoint directory has no safetensors to serve.
    NoShards,
    /// The index names a shard file that is not in the folder.
    MissingShard(String),
    /// A model*.safetensors file the index does not name (a leftover, refused).
    LeftoverShard(String),
    /// The index file exists but is not the expected shape.
    BadIndex,
    /// A shard failed to open or parse.
    Shard(PathBuf, StError),
    Io(PathBuf, std::io::Error),
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenError::NoShards => f.write_str("the model folder has no safetensors files"),
            OpenError::MissingShard(name) => write!(f, "the weight map names {} but the folder does not hold it", name),
            OpenError::LeftoverShard(name) => write!(f, "{} is a model safetensors file the index does not name", name),
            OpenError::BadIndex => f.write_str("model.safetensors.index.json is not the expected shape"),
            OpenError::Shard(path, err) => write!(f, "{}: {}", path.display(), err),
            OpenError::Io(path, err) => write!(f, "{}: {}", path.display(), err),
        }
    }
}

impl std::error::Error for OpenError {}

/// Upstream: `get`'s refusal (checkpoint.zig:93-100).
#[derive(Debug)]
pub struct MissingTensor(pub String);

impl std::fmt::Display for MissingTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "checkpoint has no tensor {}", self.0)
    }
}

impl std::error::Error for MissingTensor {}

/// Upstream: `expect`'s refusal (checkpoint.zig:103-110).
#[derive(Debug)]
pub struct UnexpectedTensor(pub String);

impl std::fmt::Display for UnexpectedTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: the dtype or shape differs from the expected", self.0)
    }
}

impl std::error::Error for UnexpectedTensor {}

/// One shard: its path, its mmap, and its parsed header. The mmap is kept so the
/// parsed borrows stay valid; it is never read directly.
struct Shard {
    path: PathBuf,
    _map: Mmap,
    st: SafeTensors<'static>,
}

impl std::fmt::Debug for Shard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shard")
            .field("path", &self.path)
            .field("tensors", &self.st.names().count())
            .finish()
    }
}

impl Shard {
    /// The shard's whole mapped span (the CUDA source stages reads
    /// beside it). Upstream: the file's `map.memory` (checkpoint.zig:60).
    fn map(&self) -> &[u8] {
        // SAFETY: the map outlives the borrow; this helper exposes the
        // same bytes SafeTensors already borrows.
        unsafe { std::slice::from_raw_parts(self._map.as_ptr(), self._map.len()) }
    }
}

/// One shard opened and mapped: the piece `open` and `add` share.
fn shard_at(path: &Path) -> Result<Shard, OpenError> {
    let file = fs::File::open(path).map_err(|e| OpenError::Io(path.to_path_buf(), e))?;
    let map = unsafe { Mmap::map(&file) }.map_err(|e| OpenError::Io(path.to_path_buf(), e))?;
    // Safety: the mmap is owned by the Shard; SafeTensors borrows its bytes and
    // the Shard keeps both for exactly as long.
    let st = {
        let bytes: &'static [u8] = unsafe { std::slice::from_raw_parts(map.as_ptr(), map.len()) };
        SafeTensors::parse(bytes).map_err(|e| OpenError::Shard(path.to_path_buf(), e))?
    };
    Ok(Shard {
        path: path.to_path_buf(),
        _map: map,
        st,
    })
}

/// The checkpoint: shards in name order, tensors by name.
///
/// Upstream: `checkpoint.zig:10  pub const Checkpoint = struct { ... }` — "The
/// checkpoint's shard paths in name order: model.safetensors.index.json's weight map,
/// else every model*.safetensors."
#[derive(Debug)]
pub struct Checkpoint {
    dir: PathBuf,
    shards: Vec<Shard>,
    /// Names a `get` took (checkpoint.zig:61): the loader refuses a
    /// checkpoint with leftovers. Interior so `get` borrows shared —
    /// upstream's get takes no exclusivity a reader can observe, and
    /// several mapped tensors read at once.
    used: RefCell<HashSet<String>>,
}

/// What `info`-style reads report for one tensor: where it lives and what it is.
#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub shard: usize,
}

impl Checkpoint {
    /// The empty namespace `add` grows (upstream constructs the struct
    /// directly at checkpoint.zig:156 for the MTP head).
    pub fn empty() -> Checkpoint {
        Checkpoint {
            dir: PathBuf::new(),
            shards: Vec::new(),
            used: RefCell::new(HashSet::new()),
        }
    }

    pub fn open(dir: &Path) -> Result<Checkpoint, OpenError> {
        let shard_paths = shard_files(dir)?;
        let mut shards = Vec::with_capacity(shard_paths.len());
        for path in shard_paths {
            shards.push(shard_at(&path)?);
        }
        Ok(Checkpoint {
            dir: dir.to_path_buf(),
            shards,
            used: RefCell::new(HashSet::new()),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The shard paths, in name order.
    pub fn shard_paths(&self) -> impl Iterator<Item = &Path> + '_ {
        self.shards.iter().map(|s| s.path.as_path())
    }

    /// The whole namespace's tensor names, shard by shard, each shard in name order.
    pub fn names(&self) -> impl Iterator<Item = &str> + '_ {
        self.shards.iter().flat_map(|s| s.st.names())
    }

    /// The tensor named `name`, marked as used; a missing name is an
    /// error naming it. Upstream: `get` (checkpoint.zig:93).
    pub fn get(&self, name: &str) -> Result<Tensor<'_>, MissingTensor> {
        for s in &self.shards {
            if let Some(t) = s.st.get(name) {
                self.used.borrow_mut().insert(name.to_string());
                return Ok(t);
            }
        }
        eprintln!("checkpoint has no tensor {name}");
        Err(MissingTensor(name.to_string()))
    }

    /// `get` with the dtype and shape checked. Upstream: `expect`
    /// (checkpoint.zig:103).
    pub fn expect(
        &self,
        name: &str,
        dtype: DType,
        shape: &[usize],
    ) -> Result<Tensor<'_>, UnexpectedTensor> {
        let t = match self.get(name) {
            Ok(t) => t,
            Err(e) => {
                return Err(UnexpectedTensor(e.0));
            }
        };
        if !t.is(dtype, shape) {
            eprintln!(
                "{name}: {:?} {:?}, expected {dtype:?} {shape:?}",
                t.dtype,
                t.shape
            );
            return Err(UnexpectedTensor(name.to_string()));
        }
        Ok(t)
    }

    /// One more file (an MTP head beside the model's files). Upstream:
    /// `add` (checkpoint.zig:78).
    pub fn add(&mut self, dir: &Path, name: &str) -> Result<(), OpenError> {
        let path = dir.join(name);
        let shard = shard_at(&path)?;
        self.shards.push(shard);
        Ok(())
    }

    /// Names no `get` took (the loader refuses a checkpoint with
    /// leftovers). Upstream: `unused` (checkpoint.zig:113).
    pub fn unused(&self) -> usize {
        let mut n: usize = 0;
        for s in &self.shards {
            for k in s.st.names() {
                if self.used.borrow().contains(k) {
                    continue;
                }
                if n < 5 {
                    eprintln!("unused checkpoint tensor {k}");
                }
                n += 1;
            }
        }
        n
    }

    /// The shard files with their mapped bytes, for readers that stage
    /// checkpoint reads beside the mappings (the CUDA source's direct
    /// slots). Upstream: the `files` field (checkpoint.zig:60).
    pub fn files(&self) -> impl Iterator<Item = (&Path, &[u8])> + '_ {
        self.shards.iter().map(|s| (s.path.as_path(), s.map()))
    }

    /// Where a tensor lives, for `info`-style listing without holding the borrow.
    pub fn info(&self, name: &str) -> Option<TensorInfo> {
        for (i, s) in self.shards.iter().enumerate() {
            if let Some(t) = s.st.get(name) {
                return Some(TensorInfo {
                    dtype: t.dtype,
                    shape: t.shape.to_vec(),
                    shard: i,
                });
            }
        }
        None
    }

    pub fn contains(&self, name: &str) -> bool {
        self.shards.iter().any(|s| s.st.names().any(|n| n == name))
    }
}

/// The checkpoint's shard paths in name order: `model.safetensors.index.json`'s weight
/// map, else every `model*.safetensors` in the folder.
///
/// Upstream: `checkpoint.zig` `shardFiles` — the file list every backend's load walks
/// (qwen3_5's model.zig:31 calls it before its selected Metal reads).
pub fn shard_files(dir: &Path) -> Result<Vec<PathBuf>, OpenError> {
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        // The weight map's files, in name order (checkpoint.zig:17).
        let bytes = fs::read_to_string(&index).map_err(|e| OpenError::Io(index.clone(), e))?;
        let value: Value = serde_json::from_str(&bytes).map_err(|_| OpenError::BadIndex)?;
        let names = value
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or(OpenError::BadIndex)?;
        let mut files: Vec<String> = names
            .values()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
        files.sort();
        files.dedup();
        for name in &files {
            let path = dir.join(name);
            if !path.is_file() {
                return Err(OpenError::MissingShard(name.clone()));
            }
        }
        // Leftovers refused: every model*.safetensors in the folder must be named.
        for entry in fs::read_dir(dir).map_err(|e| OpenError::Io(dir.to_path_buf(), e))? {
            let entry = entry.map_err(|e| OpenError::Io(dir.to_path_buf(), e))?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("model")
                && name.ends_with(".safetensors")
                && !files.contains(&name.to_string())
            {
                return Err(OpenError::LeftoverShard(name.to_string()));
            }
        }
        Ok(files.into_iter().map(|name| dir.join(name)).collect())
    } else {
        // Else every model*.safetensors (checkpoint.zig:38).
        let mut files: Vec<PathBuf> = fs::read_dir(dir)
            .map_err(|e| OpenError::Io(dir.to_path_buf(), e))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .map(|n| {
                        let n = n.to_string_lossy();
                        n.starts_with("model") && n.ends_with(".safetensors")
                    })
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(OpenError::NoShards);
        }
        Ok(files)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writes one safetensors file the way the Zig fixtures do.
    fn shard(dir: &Path, name: &str, header: &str, data: &[u8]) {
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        fs::write(dir.join(name), out).unwrap();
    }

    /// Mirrors checkpoint.zig:158-170: an index's two shards map as one namespace.
    #[test]
    fn opens_an_indexed_checkpoint() {
        let dir = std::env::temp_dir().join("tf-checkpoint-index-test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        shard(
            &dir,
            "model-00001-of-00002.safetensors",
            r#"{"a.weight":{"dtype":"U32","shape":[2,3],"data_offsets":[0,24]}}"#,
            &[1u8; 24],
        );
        shard(
            &dir,
            "model-00002-of-00002.safetensors",
            r#"{"b.bias":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}}"#,
            &[2u8; 6],
        );
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"metadata":{"total_size":30},"weight_map":{"a.weight":"model-00001-of-00002.safetensors","b.bias":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let ckpt = Checkpoint::open(&dir).unwrap();
        assert_eq!(ckpt.shard_paths().count(), 2);
        assert!(ckpt.get("a.weight").unwrap().is(DType::U32, &[2, 3]));
        assert!(ckpt.get("b.bias").unwrap().is(DType::BF16, &[3]));
        assert!(ckpt.contains("a.weight"));
        assert!(!ckpt.contains("c.weight"));
        let _ = fs::remove_dir_all(&dir);
    }

    /// The index names a shard the folder does not hold (checkpoint.zig's weight-map
    /// rule), and a model shard the index does not name is a leftover, refused.
    #[test]
    fn refuses_missing_then_leftover_shards() {
        let dir = std::env::temp_dir().join("tf-checkpoint-leftover-test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        shard(
            &dir,
            "model-00001-of-00002.safetensors",
            r#"{"a.weight":{"dtype":"U32","shape":[2,3],"data_offsets":[0,24]}}"#,
            &[1u8; 24],
        );
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"metadata":{"total_size":30},"weight_map":{"a.weight":"model-00001-of-00002.safetensors","b.bias":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        // The weight map names shard 2; the folder holds only shard 1.
        match Checkpoint::open(&dir) {
            Err(OpenError::MissingShard(n)) => assert_eq!(n, "model-00002-of-00002.safetensors"),
            other => panic!("expected missing shard, got open: {}", other.is_ok()),
        }
        // With both shards the index names, the namespace opens.
        shard(
            &dir,
            "model-00002-of-00002.safetensors",
            r#"{"b.bias":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}}"#,
            &[2u8; 6],
        );
        Checkpoint::open(&dir).expect("the named shards open");
        // An extra model shard the index does not name is a leftover, refused.
        shard(
            &dir,
            "model-00009-of-00009.safetensors",
            r#"{"z":{"dtype":"U32","shape":[1],"data_offsets":[0,4]}}"#,
            &[0u8; 4],
        );
        match Checkpoint::open(&dir) {
            Err(OpenError::LeftoverShard(n)) => {
                assert_eq!(n, "model-00009-of-00009.safetensors")
            }
            other => panic!("expected leftover refusal, got open: {}", other.is_ok()),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    /// No index: every model*.safetensors serves (checkpoint.zig:38).
    #[test]
    fn opens_unindexed_folder() {
        let dir = std::env::temp_dir().join("tf-checkpoint-plain-test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        shard(
            &dir,
            "model-00001.safetensors",
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            &[0u8; 8],
        );
        let ckpt = Checkpoint::open(&dir).unwrap();
        assert!(ckpt.get("a").unwrap().is(DType::F32, &[2]));
        // A non-model safetensors file is not a shard.
        shard(
            &dir,
            "tokenizer.safetensors",
            r#"{"t":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
            &[7],
        );
        let ckpt = Checkpoint::open(&dir).unwrap();
        assert_eq!(ckpt.shard_paths().count(), 1);
        let _ = fs::remove_dir_all(&dir);
    }
}
