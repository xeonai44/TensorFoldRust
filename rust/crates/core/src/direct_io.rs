//! Positional reads that skip the page cache (O_DIRECT on Linux):
//! 4 KiB-aligned spans into aligned buffers.
//!
//! Port of `zig/src/core/direct_io.zig` (151 lines) — the pread the
//! read-at loop runs is a trait so the retry test can script it, as
//! upstream's injected `pread` fn does.

use std::ffi::c_char;

/// Offset, length and buffer alignment a direct read needs (the
/// logical block size of every NVMe we run on). Upstream: `alignment`
/// (direct_io.zig:7).
pub const ALIGNMENT: usize = 4096;

/// Upstream: `Buffer` (direct_io.zig:9) — a slice the caller has
/// placed at `ALIGNMENT` (the port's slices carry no alignment in the
/// type; the loader allocates them aligned and the short-read retry
/// below keeps every pread aligned).
pub type Buffer<'a> = &'a mut [u8];

/// Upstream: the error set (direct_io.zig:18's uses).
#[derive(Debug)]
pub enum ReadError {
    FileNotFound,
    BufferTooSmall,
    ReadFailed,
    EndOfFile,
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadError::FileNotFound => write!(f, "the file does not open"),
            ReadError::BufferTooSmall => write!(f, "the buffer is too small for the read"),
            ReadError::ReadFailed => write!(f, "the read failed"),
            ReadError::EndOfFile => write!(f, "the read ran past the end of the file"),
        }
    }
}

impl std::error::Error for ReadError {}

/// Bytes a read of `len` bytes at `offset` occupies once widened to
/// aligned ends. Upstream: `span` (direct_io.zig:12).
pub fn span(offset: u64, len: usize) -> usize {
    let lo = offset & !(ALIGNMENT as u64 - 1);
    (offset + len as u64).next_multiple_of(ALIGNMENT as u64) as usize - lo as usize
}

/// The largest read whose span always fits a buffer of `bytes`
/// (whatever its offset). Upstream: `fits` (direct_io.zig:18).
pub fn fits(bytes: usize) -> usize {
    bytes - 2 * ALIGNMENT
}

/// The positional read the loop runs — a trait so tests script it.
/// Upstream: the `pread` fn parameter (direct_io.zig:53).
pub trait Pread {
    /// Returns the bytes read, or a negative errno.
    fn pread(&mut self, fd: i32, buf: *mut u8, len: usize, offset: u64) -> isize;
}

/// The platform's pread. Upstream: `sysPread` (direct_io.zig:49).
pub struct SysPread;

impl Pread for SysPread {
    fn pread(&mut self, fd: i32, buf: *mut u8, len: usize, offset: u64) -> isize {
        // SAFETY: the platform's own pread over the caller's buffer.
        unsafe { pread(fd, buf.cast(), len, offset as i64) }
    }
}

unsafe extern "C" {
    fn open(path: *const c_char, oflag: i32, ...) -> i32;
    fn close(fd: i32) -> i32;
    fn pread(fd: i32, buf: *mut core::ffi::c_void, nbyte: usize, offset: i64) -> isize;
}

/// Linux: O_DIRECT skips the page cache; CLOEXEC everywhere.
const O_RDONLY: i32 = 0;
const O_CLOEXEC: i32 = 0o2000000;
#[cfg(target_os = "linux")]
const O_DIRECT: i32 = 0o40000; // 0x4000, the Linux value (macOS's 0x400000 is O_DIRECTORY there)

/// Upstream: `File` (direct_io.zig:22).
pub struct File {
    pub fd: i32,
    pub direct: bool,
}

impl File {
    /// The aligned-span read against raw fields — the loader's queued
    /// reads carry the descriptor, not the owning File (upstream's
    /// Pending and Job copy the File value; the port's File owns its
    /// close, so its copies carry `fd` and `direct` instead).
    pub fn read_fd(
        fd: i32,
        direct: bool,
        buf: &mut [u8],
        offset: u64,
        len: usize,
    ) -> Result<usize, ReadError> {
        let f = File { fd, direct };
        let got = read_at(&f, buf, offset, len, &mut SysPread)?;
        Ok(got.len())
    }

    /// `path` read-only, direct when its file system allows it, else
    /// through the page cache. Upstream: `open` (direct_io.zig:27).
    pub fn open(path: &std::ffi::CStr) -> Result<File, ReadError> {
        #[cfg(target_os = "linux")]
        {
            // SAFETY: the platform's open over the NUL-terminated path.
            let fd = unsafe { open(path.as_ptr(), O_RDONLY | O_CLOEXEC | O_DIRECT) };
            if fd >= 0 {
                return Ok(File { fd, direct: true });
            }
        }
        // SAFETY: the platform's open over the NUL-terminated path.
        let fd = unsafe { open(path.as_ptr(), O_RDONLY | O_CLOEXEC) };
        if fd < 0 {
            return Err(ReadError::FileNotFound);
        }
        Ok(File { fd, direct: false })
    }

    /// Upstream: `close` (direct_io.zig:37).
    pub fn close(f: &mut File) {
        // SAFETY: the descriptor opened at construction, closed once.
        unsafe { close(f.fd) };
        f.fd = -1;
    }

    /// `len` bytes at `offset`, read as their aligned span into `buf`;
    /// the result is the requested bytes within it. Upstream: `read`
    /// (direct_io.zig:43).
    pub fn read<'a>(&self, buf: &'a mut [u8], offset: u64, len: usize) -> Result<&'a [u8], ReadError> {
        read_at(self, buf, offset, len, &mut SysPread)
    }
}

impl Drop for File {
    fn drop(&mut self) {
        if self.fd >= 0 {
            File::close(self);
        }
    }
}

/// Upstream: `readAt` (direct_io.zig:53) — the aligned span read, a
/// short O_DIRECT read retried from the last 4 KiB boundary so the
/// next pread stays aligned.
fn read_at<'a>(
    f: &File,
    buf: &'a mut [u8],
    offset: u64,
    len: usize,
    pread: &mut dyn Pread,
) -> Result<&'a [u8], ReadError> {
    let lo = offset & !(ALIGNMENT as u64 - 1);
    let want: usize = (offset + len as u64 - lo) as usize;
    let whole = span(offset, len);
    if whole > buf.len() {
        return Err(ReadError::BufferTooSmall);
    }
    let mut got: usize = 0;
    while got < want {
        // A short O_DIRECT read is retried from the last 4 KiB boundary
        // so the next pread stays aligned.
        let at: usize = if f.direct {
            got & !(ALIGNMENT - 1)
        } else {
            got
        };
        let ask = whole - at;
        let n = pread.pread(f.fd, buf.as_mut_ptr().wrapping_add(at), ask, lo + at as u64);
        if n < 0 {
            if std::io::Error::last_os_error().raw_os_error() == Some(libc_eintr()) {
                continue;
            }
            eprintln!("read of {ask} bytes at {} failed", lo + at as u64);
            return Err(ReadError::ReadFailed);
        }
        if n == 0 {
            return Err(ReadError::EndOfFile);
        }
        if n > ask as isize {
            return Err(ReadError::ReadFailed);
        }
        let next = at + n as usize;
        if next <= got {
            return Err(ReadError::ReadFailed);
        }
        got = next;
    }
    Ok(&buf[(offset - lo) as usize..(offset - lo) as usize + len])
}

/// EINTR, the errno spelling the platform shares.
const fn eintr() -> i32 {
    4
}

fn libc_eintr() -> i32 {
    eintr()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream's `span and fits` (direct_io.zig:78).
    #[test]
    fn span_and_fits() {
        assert_eq!(4096, span(0, 1));
        assert_eq!(8192, span(4095, 2));
        assert_eq!(4096, span(4096, 4096));
        assert_eq!(3 * 4096, span(100, 2 * 4096));
        assert!(span(4095, fits(1 << 20)) <= 1 << 20);
    }

    /// Upstream's `direct reads return the same bytes as buffered
    /// reads` (direct_io.zig:86) — Linux only (upstream skips elsewhere).
    #[cfg(target_os = "linux")]
    #[test]
    fn direct_reads_return_the_same_bytes_as_buffered_reads() {
        let mut f = File::open(c"/proc/self/exe").unwrap();
        // The reference uses the page cache, so its unaligned reads do
        // not inherit O_DIRECT.
        let mut plain = File {
            fd: -1,
            direct: false,
        };
        // SAFETY: the platform's open over the NUL-terminated path.
        plain.fd = unsafe { open(c"/proc/self/exe".as_ptr(), O_RDONLY | O_CLOEXEC) };
        assert!(plain.fd >= 0);
        let mut buf = vec![0u8; 1 << 20];
        let mut want = vec![0u8; 1 << 19];
        for c in [[0usize, 1usize], [1, 4095], [4095, 2], [12345, 100000], [8192, 1 << 19]] {
            // SAFETY: the platform's pread into the wanted span.
            let n = unsafe {
                pread(
                    plain.fd,
                    want.as_mut_ptr().cast(),
                    c[1],
                    c[0] as i64,
                )
            };
            assert_eq!(c[1] as isize, n);
            let got = f.read(&mut buf, c[0] as u64, c[1]).unwrap();
            assert_eq!(&want[..c[1]], got);
        }
        assert!(matches!(
            f.read(&mut buf, 1, 1 << 20),
            Err(ReadError::BufferTooSmall)
        ));
        File::close(&mut plain);
        File::close(&mut f);
    }

    /// Upstream's `Script` (direct_io.zig:107): a scripted pread that
    /// fails alignment and checks the read-at loop's retries.
    struct Script {
        plan: Vec<usize>,
        offs: [u64; 8],
        i: usize,
        calls: usize,
        bad: bool,
    }

    impl Pread for Script {
        fn pread(&mut self, _fd: i32, buf: *mut u8, len: usize, offset: u64) -> isize {
            if self.calls < self.offs.len() {
                self.offs[self.calls] = offset;
            }
            self.calls += 1;
            if offset % ALIGNMENT as u64 != 0
                || len % ALIGNMENT != 0
                || buf as usize % ALIGNMENT != 0
            {
                self.bad = true;
            }
            let step = if self.i < self.plan.len() {
                self.plan[self.i]
            } else {
                len
            };
            if self.i < self.plan.len() {
                self.i += 1;
            }
            let n = step.min(len);
            // SAFETY: the test's own buffer span.
            unsafe {
                for k in 0..n {
                    *buf.add(k) = ((offset + k as u64) % 251) as u8;
                }
            }
            n as isize
        }
    }

    fn direct(script: &mut Script, out: &mut [u8], offset: u64) -> Result<(), ReadError> {
        let mut storage = AlignedStorage::new();
        let f = File { fd: -1, direct: true };
        let got = read_at(&f, storage.buf(), offset, out.len(), script)?;
        out.copy_from_slice(got);
        Ok(())
    }

    /// An 8 KiB buffer at the alignment, as the loader's slots are.
    struct AlignedStorage {
        raw: *mut u8,
        len: usize,
    }

    impl AlignedStorage {
        fn new() -> AlignedStorage {
            let layout = std::alloc::Layout::from_size_align(8192, ALIGNMENT).unwrap();
            // SAFETY: an aligned allocation of the layout above.
            let raw = unsafe { std::alloc::alloc(layout) };
            AlignedStorage { raw, len: 8192 }
        }
        fn buf(&mut self) -> &mut [u8] {
            // SAFETY: the allocation's own span.
            unsafe { std::slice::from_raw_parts_mut(self.raw, self.len) }
        }
    }

    impl Drop for AlignedStorage {
        fn drop(&mut self) {
            // SAFETY: the allocation made in new, freed once.
            unsafe {
                std::alloc::dealloc(
                    self.raw,
                    std::alloc::Layout::from_size_align(self.len, ALIGNMENT).unwrap(),
                );
            }
        }
    }

    /// Upstream's `a short direct read is retried from the aligned
    /// boundary` (direct_io.zig:135).
    #[test]
    fn a_short_direct_read_is_retried_from_the_aligned_boundary() {
        let mut script = Script {
            plan: vec![100, 4096],
            offs: [0; 8],
            i: 0,
            calls: 0,
            bad: false,
        };
        let mut got = [0u8; 4096];
        direct(&mut script, &mut got, 0).unwrap();
        assert!(!script.bad);
        assert_eq!(2, script.calls);
        assert_eq!(0, script.offs[0]);
        assert_eq!(0, script.offs[1]);
        for (i, b) in got.iter().enumerate() {
            assert_eq!((i % 251) as u8, *b);
        }
        let mut boundary = Script {
            plan: vec![4096, 4096],
            offs: [0; 8],
            i: 0,
            calls: 0,
            bad: false,
        };
        let mut wide = [0u8; 8192];
        direct(&mut boundary, &mut wide, 0).unwrap();
        assert!(!boundary.bad);
        assert_eq!(0, boundary.offs[0]);
        assert_eq!(4096, boundary.offs[1]);
        assert_eq!((4096 % 251) as u8, wide[4096]);
    }
}
