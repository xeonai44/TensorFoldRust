//! Raw dumps of an eager forward (each block norm's h, y, xs, then
//! logits and tokens) named as the oracle names them.
//!
//! Port of `zig/src/families/nemotron/cuda_dump.zig` (35 lines).

use crate::cuda_kernels::Ops;
use crate::error::Error;

/// Upstream: `Dump` (cuda_dump.zig:6) — held behind a const Forward
/// (a mutable pointer upstream), so the counter is a Cell.
pub struct Dump<'a, 'd> {
    pub dir: String,
    ops: &'a Ops<'a, 'd>,
    pub i: std::cell::Cell<usize>,
}

impl Dump<'_, '_> {
    pub fn new<'a, 'd>(dir: &str, ops: &'a Ops<'a, 'd>) -> Dump<'a, 'd> {
        Dump {
            dir: dir.to_string(),
            ops,
            i: std::cell::Cell::new(0),
        }
    }

    fn write(&self, name: &str, ptr: u64, bytes: usize) -> Result<(), Error> {
        let mut host = vec![0u8; bytes];
        self.ops.download(&mut host, ptr)?;
        self.ops.s.synchronize()?;
        let path = format!("{}/{}.bin", self.dir, name);
        std::fs::write(&path, &host).map_err(|_| Error::FileNotFound)?;
        Ok(())
    }

    /// Upstream: `norm` (cuda_dump.zig:22).
    pub fn norm(&self, h: u64, y: u64, xs: u64, rows: usize, hidden: usize) -> Result<(), Error> {
        let n = rows * hidden;
        let i = self.i.get();
        self.write(&format!("{i:03}_h"), h, n * 2)?;
        self.write(&format!("{i:03}_y"), y, n * 2)?;
        self.write(&format!("{i:03}_xs"), xs, n / 64 * 4)?;
        self.i.set(i + 1);
        Ok(())
    }

    /// Upstream: `tail` (cuda_dump.zig:31).
    pub fn tail(&self, logits: u64, sampled: u64, rows: usize, vocab: usize) -> Result<(), Error> {
        self.write("logits", logits, rows * vocab * 2)?;
        self.write("sampled", sampled, rows * 4)
    }
}
