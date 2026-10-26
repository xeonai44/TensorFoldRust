//! Kernel launches: argument values packed where cuLaunchKernel reads
//! them, plus the extended launch attributes.
//!
//! Port of `zig/src/cuda/launch.zig` (143 lines).

use crate::abi;
use crate::driver::Error;
use crate::module::Function;
use crate::stream::Stream;

pub use abi::Dim3;

/// Argument values in kernel order; each must have exactly the C type of
/// its kernel parameter. Upstream: ``Args`` (launch.zig:12) — the values
/// packed at their C alignments, the pointer array rebuilt on each
/// `pointers()` so a moved Args never hands out stale addresses.
pub struct Args {
    storage: [u8; Args::MAX_BYTES],
    offsets: [u16; Args::MAX_ARGS],
    sizes: [u16; Args::MAX_ARGS],
    ptrs: [*mut core::ffi::c_void; Args::MAX_ARGS],
    used: usize,
    count: usize,
}

impl Args {
    pub const MAX_ARGS: usize = 48;
    pub const MAX_BYTES: usize = 2048;

    /// Upstream: the `= undefined` initializer.
    pub fn new() -> Args {
        Args {
            storage: [0; Args::MAX_BYTES],
            offsets: [0; Args::MAX_ARGS],
            sizes: [0; Args::MAX_ARGS],
            ptrs: [std::ptr::null_mut(); Args::MAX_ARGS],
            used: 0,
            count: 0,
        }
    }

    /// Copies `value` (a pointer-sized address, scalar, or extern struct
    /// passed by value) at its C alignment. Upstream: `add`
    /// (launch.zig:23).
    pub fn add<T: Copy>(&mut self, value: T) {
        const {
            assert!(std::mem::size_of::<T>() > 0);
        }
        let align = std::mem::align_of::<T>();
        let at = (self.used + align - 1) & !(align - 1);
        debug_assert!(self.count < Args::MAX_ARGS && at + std::mem::size_of::<T>() <= Args::MAX_BYTES);
        // SAFETY: the value copied into its aligned span of the storage.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&value as *const T).cast::<u8>(),
                self.storage.as_mut_ptr().add(at),
                std::mem::size_of::<T>(),
            );
        }
        self.offsets[self.count] = at as u16;
        self.sizes[self.count] = std::mem::size_of::<T>() as u16;
        self.count += 1;
        self.used = at + std::mem::size_of::<T>();
    }

    /// Argument `i` read back as an unsigned integer (4- and 8-byte
    /// values only). Upstream: `integer` (launch.zig:35).
    pub fn integer(&self, i: usize) -> Option<u64> {
        if i >= self.count {
            return None;
        }
        let at = &self.storage[self.offsets[i] as usize..];
        match self.sizes[i] {
            4 => Some(u32::from_le_bytes(at[0..4].try_into().unwrap()) as u64),
            8 => Some(u64::from_le_bytes(at[0..8].try_into().unwrap())),
            _ => None,
        }
    }

    /// The address array, rebuilt here so a moved Args never hands out
    /// stale addresses. Upstream: `pointers` (launch.zig:44).
    pub fn pointers(&mut self) -> *const *mut core::ffi::c_void {
        if self.count == 0 {
            return std::ptr::null();
        }
        for i in 0..self.count {
            // SAFETY: the storage's address at each offset — the rebuilt
            // array the launch reads.
            self.ptrs[i] = unsafe { self.storage.as_mut_ptr().add(self.offsets[i] as usize).cast() };
        }
        self.ptrs.as_ptr()
    }
}

impl Default for Args {
    fn default() -> Self {
        Args::new()
    }
}

/// Upstream: ``Config`` (launch.zig:55).
#[derive(Clone, Copy, Default)]
pub struct Config {
    pub grid: Dim3,
    pub block: Dim3,
    pub shared: u32,
    pub cluster: Option<Dim3>,
    pub cluster_spread: bool,
    pub pdl: bool,
    pub cooperative: bool,
}

impl Config {
    /// The plain-launch shape: no cluster, PDL or cooperative
    /// attribute (launch.zig:60's plain).
    pub fn plain(&self) -> bool {
        self.cluster.is_none() && !self.pdl && !self.cooperative
    }

    /// Upstream: `validate` (launch.zig:63) — every dimension non-zero,
    /// a block at most 1024 threads.
    pub fn validate(&self) -> Result<(), Error> {
        let (g, b) = (self.grid, self.block);
        if g.x == 0 || g.y == 0 || g.z == 0 || b.x == 0 || b.y == 0 || b.z == 0 {
            return Err(Error::Invalid);
        }
        if b.x as u64 * b.y as u64 * b.z as u64 > 1024 {
            return Err(Error::Invalid);
        }
        Ok(())
    }
}

/// cuLaunchKernel for plain launches, cuLaunchKernelEx when a cluster,
/// PDL or cooperative attribute is set. Upstream: `launch`
/// (launch.zig:71).
pub fn launch(f: &Function, cfg: Config, stream: &Stream, args: &mut Args) -> Result<(), Error> {
    cfg.validate()?;
    let d = f.d;
    let (g, b) = (cfg.grid, cfg.block);
    if cfg.plain() {
        // SAFETY: the plain launch with the packed arguments.
        let res = unsafe {
            (d.api.cuLaunchKernel.unwrap_unchecked())(
                f.handle,
                g.x, g.y, g.z, b.x, b.y, b.z,
                cfg.shared,
                stream.handle,
                args.pointers(),
                std::ptr::null(),
            )
        };
        return d.check(res, "cuLaunchKernel");
    }
    let mut attrs = [abi::LaunchAttribute::new(0); 4];
    let n = fill_attributes(cfg, &mut attrs);
    let lc = abi::LaunchConfig {
        grid_x: g.x,
        grid_y: g.y,
        grid_z: g.z,
        block_x: b.x,
        block_y: b.y,
        block_z: b.z,
        shared_bytes: cfg.shared,
        stream: stream.handle,
        attrs: attrs.as_ptr(),
        num_attrs: n as std::ffi::c_uint,
    };
    // SAFETY: the extended launch with its attributes.
    let res = unsafe {
        (d.api.cuLaunchKernelEx.unwrap_unchecked())(&lc, f.handle, args.pointers(), std::ptr::null())
    };
    d.check(res, "cuLaunchKernelEx")
}

fn attribute(id: std::ffi::c_uint) -> abi::LaunchAttribute {
    abi::LaunchAttribute::new(id)
}

/// Upstream: `fillAttributes` (launch.zig:94) — PDL, cooperative, the
/// cluster dimension and its spread preference, in the ABI's order.
fn fill_attributes(cfg: Config, out: &mut [abi::LaunchAttribute; 4]) -> usize {
    let mut n: usize = 0;
    if cfg.pdl {
        out[n] = attribute(abi::LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION);
        // SAFETY: the int field of the union, a plain write.
        out[n].value.int = 1;
        n += 1;
    }
    if cfg.cooperative {
        out[n] = attribute(abi::LAUNCH_ATTRIBUTE_COOPERATIVE);
        out[n].value.int = 1;
        n += 1;
    }
    if let Some(c) = cfg.cluster {
        out[n] = attribute(abi::LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION);
        // SAFETY: the cluster_dim field, a plain array write.
        out[n].value.cluster_dim = [c.x, c.y, c.z];
        n += 1;
        if cfg.cluster_spread {
            out[n] = attribute(abi::LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE);
            out[n].value.int = abi::CLUSTER_SCHEDULING_SPREAD;
            n += 1;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `args keep C alignment and sizes` (launch.zig:112):
    /// the offsets at their C alignments, the values read back through
    /// the packed pointers.
    #[test]
    fn args_keep_c_alignment_and_sizes() {
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct S {
            p: u64,
            n: i32,
            s: f32,
        }
        let mut a = Args::new();
        a.add(0x1122334455667788u64);
        a.add(-3i32);
        a.add(true);
        a.add(2.5f64);
        a.add(S { p: 7, n: 8, s: 9.0 });
        assert_eq!(&a.offsets[..a.count], &[0u16, 8, 12, 16, 24]);
        let ptrs = a.pointers();
        // SAFETY: the pointers into the storage at the offsets — the
        // same values read back through them.
        unsafe {
            let one = ptrs.add(1).read() as *const i32;
            let three = ptrs.add(3).read() as *const f64;
            assert_eq!(std::ptr::read_unaligned(one), -3);
            assert_eq!(std::ptr::read_unaligned(three), 2.5);
        }
    }

    /// `integer` reads the packed 4- and 8-byte values; other sizes and
    /// out-of-range give None.
    #[test]
    fn integer_reads_the_packed_values() {
        let mut a = Args::new();
        a.add(5u32);
        a.add(0x1122334455667788u64);
        assert_eq!(a.integer(0), Some(5));
        assert_eq!(a.integer(1), Some(0x1122334455667788));
        assert_eq!(a.integer(2), None);
        let mut b = Args::new();
        b.add(1u16);
        assert_eq!(b.integer(0), None, "a 2-byte value is not an integer read");
    }

    /// `Config::validate` (launch.zig:63): every dimension non-zero, a
    /// block at most 1024 threads.
    #[test]
    fn configs_validate() {
        let ok = Config {
            grid: Dim3 { x: 8, y: 1, z: 1 },
            block: Dim3 { x: 32, y: 1, z: 1 },
            ..Default::default()
        };
        assert!(ok.validate().is_ok());
        let big_block = Config {
            grid: Dim3 { x: 1, y: 1, z: 1 },
            block: Dim3 { x: 1025, y: 1, z: 1 },
            ..Default::default()
        };
        assert!(big_block.validate().is_err());
        let zero_grid = Config {
            grid: Dim3 { x: 0, y: 1, z: 1 },
            block: Dim3::DEFAULT,
            ..Default::default()
        };
        assert!(zero_grid.validate().is_err());
    }
}
