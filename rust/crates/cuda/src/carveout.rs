//! GB10 display memory as device memory: a DRM dumb buffer, mapped and
//! registered with CUDA, carved into cache spans.
//!
//! Port of `zig/src/cuda/carveout.zig` (232 lines): the pure placement
//! (whole framebuffer rows, 256-byte aligned spans, first fit) is
//! portable with upstream's own tests; the DRM ioctls and the CUDA
//! registration are Linux-only (cfg'd), the driver-facing shapes
//! asserted where they are compiled.

pub const WIDTH: u32 = 4096; // pixels a framebuffer row
pub const BPP: u32 = 32;
/// The unit a carveout's size rounds down to.
pub const ROW_BYTES: usize = (WIDTH * BPP / 8) as usize;
/// What GB10's display reservation holds with room to spare.
pub const DEFAULT_BYTES: usize = 1792 << 20;
pub const DEFAULT_CARD: &str = "/dev/dri/card0";
/// cuMemAlloc's own.
pub const ALIGNMENT: usize = 256;

#[cfg(target_os = "linux")]
mod drm {

    /// struct drm_mode_create_dumb (drm_mode.h): height, width and bpp
    /// in; handle, pitch and size out.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct CreateDumb {
        pub height: u32,
        pub width: u32,
        pub bpp: u32,
        pub flags: u32,
        pub handle: u32,
        pub pitch: u32,
        pub size: u64,
    }

    /// struct drm_mode_map_dumb: the handle in, the offset to mmap the
    /// card at out.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct MapDumb {
        pub handle: u32,
        pub pad: u32,
        pub offset: u64,
    }

    /// struct drm_mode_destroy_dumb.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct DestroyDumb {
        pub handle: u32,
    }

    /// The ioctl numbers, drm_mode.h's own: 'd' 0xB2/B3/B4 IOWR.
    pub const IOCTL_CREATE_DUMB: u32 = 0xC020_64B2;
    pub const IOCTL_MAP_DUMB: u32 = 0xC010_64B3;
    pub const IOCTL_DESTROY_DUMB: u32 = 0xC004_64B4;

    // pub: the linux_carveout module resolves these through the glob
    unsafe extern "C" {
        pub fn ioctl(fd: i32, request: std::ffi::c_ulong, ...) -> i32;
        pub fn open(path: *const std::ffi::c_char, flags: i32, ...) -> i32;
        pub fn close(fd: i32) -> i32;
    }
}

/// `bytes` in whole framebuffer rows; zero below one row. Upstream:
/// `roundBytes` (carveout.zig:31).
pub fn round_bytes(bytes: usize) -> usize {
    bytes / ROW_BYTES * ROW_BYTES
}

/// The size the options ask for: `flag` or `on` = "1" turns it on, `mib`
/// sizes it; None means off. Upstream: `requested` (carveout.zig:36) —
/// the invalid size the error's own (CarveoutTooSmall named in the map).
pub fn requested(flag: bool, on: Option<&str>, mib: Option<&str>) -> Result<Option<usize>, String> {
    let wanted = flag || on == Some("1");
    if !wanted {
        return Ok(None);
    }
    let Some(m) = mib else {
        return Ok(Some(DEFAULT_BYTES));
    };
    if m.is_empty() {
        return Ok(Some(DEFAULT_BYTES));
    }
    let n: usize = m
        .parse()
        .map_err(|_| format!("--carveout-mib: an integer, not {m:?}"))?;
    let bytes = n
        .checked_mul(1 << 20)
        .ok_or_else(|| format!("--carveout-mib {m} is too large"))?;
    let rounded = round_bytes(bytes);
    if rounded == 0 {
        return Err(format!(
            "--carveout-mib {m} is under one framebuffer row ({ROW_BYTES} bytes)"
        ));
    }
    Ok(Some(rounded))
}

/// First-fit placement of 256-byte aligned spans in `size` bytes; with
/// nothing given back it is a bump allocator. Upstream: `Spans`
/// (carveout.zig:46).
pub struct Spans {
    pub size: usize,
    /// Sorted by offset.
    live: Vec<(usize, usize)>,
}

impl Spans {
    pub const MAX: usize = 64;

    pub fn new(size: usize) -> Spans {
        Spans {
            size,
            live: Vec::with_capacity(Spans::MAX),
        }
    }

    /// The offset of a new span of `len` bytes, or None when no gap (or
    /// no slot) holds it. Upstream: `take` (carveout.zig:56).
    pub fn take(&mut self, len: usize) -> Option<usize> {
        if len == 0 || self.live.len() >= Spans::MAX {
            return None;
        }
        let mut at: usize = 0;
        for i in 0..=self.live.len() {
            let end = if i < self.live.len() {
                self.live[i].0
            } else {
                self.size
            };
            if at <= end && end - at >= len {
                self.live.insert(i, (at, len));
                return Some(at);
            }
            if i < self.live.len() {
                at = (self.live[i].0 + self.live[i].1).div_ceil(ALIGNMENT) * ALIGNMENT;
            }
        }
        None
    }

    /// Gives back the span at `at`; an offset no span starts at is
    /// ignored. Upstream: `release` (carveout.zig:71).
    pub fn release(&mut self, at: usize) {
        if let Some(i) = self.live.iter().position(|r| r.0 == at) {
            self.live.remove(i);
        }
    }

    /// Bytes no live span holds (alignment gaps count as free).
    /// Upstream: `free` (carveout.zig:79).
    pub fn free(&self) -> usize {
        let used: usize = self.live.iter().map(|r| r.1).sum();
        self.size - used
    }
}

#[cfg(target_os = "linux")]
mod linux_carveout {
    use super::*;
    use crate::abi;
    use crate::driver::Driver;
    use crate::memory::DeviceBuffer;

    #[derive(Debug)]
    pub enum Error {
        Driver(crate::driver::Error),
        CardUnavailable,
        DumbBufferRefused,
        MapRefused,
        RegisterRefused,
    }

    impl From<crate::driver::Error> for Error {
        fn from(e: crate::driver::Error) -> Error {
            Error::Driver(e)
        }
    }

    impl std::fmt::Display for Error {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Error::Driver(e) => write!(f, "{e}"),
                Error::CardUnavailable => write!(f, "the DRM card cannot be opened"),
                Error::DumbBufferRefused => write!(f, "the dumb buffer was refused"),
                Error::MapRefused => write!(f, "the dumb buffer cannot be mapped"),
                Error::RegisterRefused => write!(f, "the mapped buffer cannot be registered"),
            }
        }
    }

    impl std::error::Error for Error {}

    /// The registered framebuffer: owns the card's fd, the dumb buffer,
    /// its mapping and the CUDA registration. Upstream: `Carveout`
    /// (carveout.zig:88).
    pub struct Carveout<'d> {
        pub d: &'d Driver,
        fd: i32,
        handle: u32,
        pub host: *mut u8,
        pub host_len: usize,
        pub dev: abi::DevicePtr,
        /// The registration flags the driver took.
        pub flags: std::ffi::c_uint,
        /// The spans behind interior mutability: upstream's Carveout is
        /// a shared allocator (the engine's buffers and its sequences
        /// take and give through one *Carveout); the port's take and
        /// give run over a shared borrow.
        pub spans: std::cell::RefCell<Spans>,
    }

    impl<'d> Carveout<'d> {
        /// Maps `want` bytes (whole rows) of `card`'s display memory
        /// into the current context. Upstream: `open` (carveout.zig:99).
        pub fn open(d: &'d Driver, card: &str, want: usize) -> Result<Carveout<'d>, Error> {
            use super::drm::*;
            let size = round_bytes(want);
            if size == 0 {
                return Err(Error::Driver(crate::driver::Error::Invalid));
            }
            let path = std::ffi::CString::new(card).map_err(|_| Error::CardUnavailable)?;
            // O_RDWR | O_CLOEXEC = 2 | 0x80000
            // SAFETY: the card opened read-write, close-on-exec.
            let fd = unsafe { open(path.as_ptr(), 2 | 0x80000) };
            if fd < 0 {
                return Err(Error::CardUnavailable);
            }
            let mut create = CreateDumb {
                height: (size / ROW_BYTES) as u32,
                width: WIDTH,
                bpp: BPP,
                ..Default::default()
            };
            // SAFETY: the create-dumb ioctl on the card's fd.
            let rc = unsafe { ioctl(fd, IOCTL_CREATE_DUMB as std::ffi::c_ulong, &mut create) };
            if rc != 0 {
                // SAFETY: the fd closed on the failure path.
                unsafe { close(fd) };
                return Err(Error::DumbBufferRefused);
            }
            if create.pitch as usize != ROW_BYTES || (create.size as usize) < size {
                let mut x = DestroyDumb { handle: create.handle };
                // SAFETY: the buffer destroyed on the failure path.
                unsafe { ioctl(fd, IOCTL_DESTROY_DUMB as std::ffi::c_ulong, &mut x) };
                // SAFETY: the fd closed.
                unsafe { close(fd) };
                return Err(Error::DumbBufferRefused);
            }
            let mut map = MapDumb {
                handle: create.handle,
                ..Default::default()
            };
            // SAFETY: the map-dumb ioctl for the mmap offset.
            let rc = unsafe { ioctl(fd, IOCTL_MAP_DUMB as std::ffi::c_ulong, &mut map) };
            if rc != 0 {
                let mut x = DestroyDumb { handle: create.handle };
                // SAFETY: the buffer destroyed on the failure path.
                unsafe { ioctl(fd, IOCTL_DESTROY_DUMB as std::ffi::c_ulong, &mut x) };
                // SAFETY: the fd closed.
                unsafe { close(fd) };
                return Err(Error::MapRefused);
            }
            // mmap RW SHARED = PROT_READ|PROT_WRITE (3), MAP_SHARED (1)
            unsafe extern "C" {
                fn mmap(
                    addr: *mut core::ffi::c_void,
                    len: usize,
                    prot: i32,
                    flags: i32,
                    fd: i32,
                    offset: i64,
                ) -> *mut core::ffi::c_void;
                fn munmap(addr: *mut core::ffi::c_void, len: usize) -> i32;
            }
            // SAFETY: the dumb buffer mapped shared read-write at its offset.
            let base = unsafe { mmap(std::ptr::null_mut(), size, 3, 1, fd, map.offset as i64) };
            if base == libc_null_mut() {
                let mut x = DestroyDumb { handle: create.handle };
                // SAFETY: the failure path destroys and closes.
                unsafe {
                    ioctl(fd, IOCTL_DESTROY_DUMB as std::ffi::c_ulong, &mut x);
                    close(fd);
                }
                return Err(Error::MapRefused);
            }
            // DEVICEMAP first; a driver that sees the range as I/O memory
            // wants IOMEMORY said too (carveout.zig:126).
            // SAFETY: the mapped range registered with the driver.
            let plain = abi::HOST_REGISTER_DEVICEMAP;
            let mut flags = plain;
            let rc = unsafe {
                (d.api.cuMemHostRegister_v2.unwrap_unchecked())(base, size, plain)
            };
            if rc != abi::SUCCESS {
                let io = plain | abi::HOST_REGISTER_IOMEMORY;
                // SAFETY: the I/O-memory retry.
                let res = unsafe {
                    (d.api.cuMemHostRegister_v2.unwrap_unchecked())(base, size, io)
                };
                if d.check(res, "cuMemHostRegister").is_err() {
                    // SAFETY: the failure path unmaps, destroys, closes.
                    unsafe {
                        munmap(base, size);
                        let mut x = DestroyDumb { handle: create.handle };
                        ioctl(fd, IOCTL_DESTROY_DUMB as std::ffi::c_ulong, &mut x);
                        close(fd);
                    }
                    return Err(Error::RegisterRefused);
                }
                flags = io;
            }
            let mut dev: abi::DevicePtr = 0;
            // SAFETY: the mapped range's device address.
            let res =
                unsafe { (d.api.cuMemHostGetDevicePointer_v2.unwrap_unchecked())(&mut dev, base, 0) };
            if let Err(e) = d.check(res, "cuMemHostGetDevicePointer") {
                // SAFETY: the failure path unregisters, unmaps, destroys, closes.
                unsafe {
                    (d.api.cuMemHostUnregister.unwrap_unchecked())(base);
                    munmap(base, size);
                    let mut x = DestroyDumb { handle: create.handle };
                    ioctl(fd, IOCTL_DESTROY_DUMB as std::ffi::c_ulong, &mut x);
                    close(fd);
                }
                return Err(Error::Driver(e));
            }
            Ok(Carveout {
                d,
                fd,
                handle: create.handle,
                host: base.cast(),
                host_len: size,
                dev,
                flags,
                spans: std::cell::RefCell::new(Spans::new(size)),
            })
        }

        /// A zeroed span of `len` bytes, 256-byte aligned; None when it
        /// does not fit (the caller falls back). Upstream: `take`
        /// (carveout.zig:148).
        pub fn take(&self, len: usize) -> Result<Option<DeviceBuffer<'d>>, Error> {
            let Some(at) = self.spans.borrow_mut().take(len) else {
                return Ok(None);
            };
            let b = DeviceBuffer {
                d: self.d,
                ptr: self.dev + at as u64,
                len,
                borrowed: true,
            };
            if let Err(e) = b.fill8(0, None) {
                self.spans.borrow_mut().release(at);
                return Err(Error::Driver(e));
            }
            Ok(Some(b))
        }

        /// Returns a span `take` gave out. Upstream: `give`
        /// (carveout.zig:156).
        pub fn give(&self, b: &DeviceBuffer) {
            self.spans.borrow_mut().release((b.ptr - self.dev) as usize);
        }

        pub fn bytes(&self) -> usize {
            self.host_len
        }

        pub fn free_bytes(&self) -> usize {
            self.spans.borrow().free()
        }
    }

    impl Drop for Carveout<'_> {
        /// Releases in reverse order of open; every span taken from it
        /// must be out of use. Upstream: `close` (carveout.zig:142).
        fn drop(&mut self) {
            use super::drm::*;
            unsafe extern "C" {
                fn munmap(addr: *mut core::ffi::c_void, len: usize) -> i32;
            }
            // SAFETY: the drop path, in open's reverse.
            unsafe {
                (self.d.api.cuMemHostUnregister.unwrap_unchecked())(self.host.cast());
                munmap(self.host.cast(), self.host_len);
                let mut x = DestroyDumb { handle: self.handle };
                ioctl(self.fd, IOCTL_DESTROY_DUMB as std::ffi::c_ulong, &mut x);
                close(self.fd);
            }
        }
    }

    fn libc_null_mut() -> *mut core::ffi::c_void {
        std::ptr::null_mut()
    }
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "linux")]
pub use linux_carveout::{Carveout, Error as CarveoutError};

#[cfg(not(target_os = "linux"))]
pub use host_only::Carveout;

#[cfg(not(target_os = "linux"))]
pub type CarveoutError = crate::driver::Error;

/// The Carveout's host-only form: upstream's Carveout compiles only on
/// Linux (its DRM shapes live in std.os.linux, and Zig's laziness never
/// analyzes the type on other hosts); the family's state layer names
/// the type everywhere, so the port gives it a shape that exists off
/// Linux — never constructed, every call the refusal or the empty
/// answer the Linux side's guards give.
#[cfg(not(target_os = "linux"))]
mod host_only {
    use crate::driver::{Driver, Error};
    use crate::memory::DeviceBuffer;

    pub struct Carveout<'d> {
        pub(crate) _d: &'d Driver,
    }

    impl<'d> Carveout<'d> {
        pub fn open(_d: &'d Driver, _card: &str, _want: usize) -> Result<Carveout<'d>, Error> {
            unreachable!("the host-only Carveout is never constructed; open refuses before")
        }

        pub fn take(&self, _len: usize) -> Result<Option<DeviceBuffer<'d>>, Error> {
            Ok(None)
        }

        pub fn give(&self, _b: &DeviceBuffer) {}

        pub fn close(&mut self) {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `spans are 256-byte aligned and refused once full`
    /// (carveout.zig:187).
    #[test]
    fn spans_are_256_byte_aligned_and_refused_once_full() {
        let mut s = Spans::new(4096);
        assert_eq!(s.take(100), Some(0));
        assert_eq!(s.take(1000), Some(256));
        assert_eq!(s.take(2816), Some(1280));
        assert_eq!(s.free(), 4096 - 3916);
        assert_eq!(s.take(1), None);
        assert_eq!(s.take(0), None);
        let mut big = Spans::new(1 << 20);
        assert_eq!(big.take((1 << 20) + 1), None);
        assert_eq!(big.take(1 << 20), Some(0));
    }

    /// Upstream's `a given-back span's room is taken again, first fit`
    /// (carveout.zig:200).
    #[test]
    fn a_given_back_span_s_room_is_taken_again_first_fit() {
        let mut s = Spans::new(4096);
        s.take(512).unwrap();
        let mid = s.take(1024).unwrap();
        s.take(512).unwrap();
        s.release(mid);
        s.release(12345); // no span starts there
        assert_eq!(s.take(768), Some(512));
        assert_eq!(s.take(512), Some(2048), "1280..1536 is too small");
        assert_eq!(s.take(256), Some(1280));
        assert_eq!(s.free(), 1536);
    }

    /// Upstream's `a span list holds at most max spans`
    /// (carveout.zig:213).
    #[test]
    fn a_span_list_holds_at_most_max_spans() {
        let mut s = Spans::new(1 << 20);
        for _ in 0..Spans::MAX {
            s.take(1).unwrap();
        }
        assert_eq!(s.take(1), None);
    }

    /// Upstream's `the carveout is opt-in, sized in whole rows`
    /// (carveout.zig:219).
    #[test]
    fn the_carveout_is_opt_in_sized_in_whole_rows() {
        assert_eq!(requested(false, None, None).unwrap(), None);
        assert_eq!(requested(false, Some("0"), Some("64")).unwrap(), None);
        assert_eq!(requested(true, None, None).unwrap(), Some(DEFAULT_BYTES));
        assert_eq!(requested(false, Some("1"), Some("")).unwrap(), Some(DEFAULT_BYTES));
        assert_eq!(requested(false, Some("1"), Some("64")).unwrap(), Some(64 << 20));
        assert!(requested(true, None, Some("lots")).unwrap_err().contains("integer"));
        assert_eq!(round_bytes(ROW_BYTES - 1), 0);
        assert_eq!(round_bytes(ROW_BYTES + 1), ROW_BYTES);
    }

    /// Upstream's `DRM ioctl numbers are drm_mode.h's`
    /// (carveout.zig:171) and `dumb buffer structs have drm_mode.h's
    /// layout` (carveout.zig:177) — asserted where the DRM shapes are
    /// compiled (Linux), the row constant everywhere.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_drm_shapes() {
        use super::drm::*;
        // The externs referenced: the ioctl shapes assert the constants
        assert_eq!(IOCTL_CREATE_DUMB, 0xC02064B2);
        assert_eq!(IOCTL_MAP_DUMB, 0xC01064B3);
        assert_eq!(IOCTL_DESTROY_DUMB, 0xC00464B4);
        assert_eq!(std::mem::size_of::<CreateDumb>(), 32);
        assert_eq!(std::mem::offset_of!(CreateDumb, handle), 16);
        assert_eq!(std::mem::offset_of!(CreateDumb, pitch), 20);
        assert_eq!(std::mem::offset_of!(CreateDumb, size), 24);
        assert_eq!(std::mem::size_of::<MapDumb>(), 16);
        assert_eq!(std::mem::offset_of!(MapDumb, offset), 8);
        assert_eq!(std::mem::size_of::<DestroyDumb>(), 4);
        // The row constant (everywhere)
        assert_eq!(ROW_BYTES, 4096 * 32 / 8);
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn the_row_constant() {
        assert_eq!(ROW_BYTES, 4096 * 32 / 8);
    }
}
