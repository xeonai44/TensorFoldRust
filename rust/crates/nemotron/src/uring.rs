//! The io_uring the CUDA source stages checkpoint reads on — upstream
//! imports it from Zig's std (`std.os.linux.IoUring`); the port
//! hand-rolls the same ABI, every layout from the kernel's own
//! `linux/io_uring.h`, and carries the ring test std's own does
//! upstream (verified live on the box).
//!
//! Linux only, like the std piece; elsewhere the type is a name the
//! source code names and never constructs.

#[cfg(target_os = "linux")]
mod ring {
    use core::sync::atomic::{AtomicU32, Ordering};

    unsafe extern "C" {
        fn syscall(num: core::ffi::c_long, ...) -> core::ffi::c_long;
        fn mmap(
            addr: *mut core::ffi::c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut core::ffi::c_void;
        fn munmap(addr: *mut core::ffi::c_void, len: usize) -> i32;
        fn close(fd: i32) -> i32;
    }

    /// x86_64's syscall numbers (the box's asm/unistd_64.h: 425, 426).
    const IO_URING_SETUP: core::ffi::c_long = 425;
    const IO_URING_ENTER: core::ffi::c_long = 426;

    /// The mmap offsets (io_uring.h).
    const OFF_SQ_RING: i64 = 0;
    const OFF_CQ_RING: i64 = 0x8000000;
    const OFF_SQES: i64 = 0x10000000;

    const PROT_READ: i32 = 1;
    const PROT_WRITE: i32 = 2;
    const MAP_SHARED: i32 = 1;

    /// IORING_OP_READ (io_uring.h's enum, 22).
    const OP_READ: u8 = 22;
    /// IORING_ENTER_GETEVENTS.
    const ENTER_GETEVENTS: u32 = 1;

    /// struct io_uring_sqe (io_uring.h): 64 bytes.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Sqe {
        pub opcode: u8,
        pub flags: u8,
        pub ioprio: u16,
        pub fd: i32,
        pub off: u64,
        pub addr: u64,
        pub len: u32,
        pub rw_flags: u32,
        pub user_data: u64,
        pub buf_index: u16,
        pub personality: u16,
        pub splice_fd_in: i32,
        pub pad: [u64; 2],
    }

    /// struct io_uring_cqe: 16 bytes.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct Cqe {
        pub user_data: u64,
        pub res: i32,
        pub flags: u32,
    }

    /// struct io_sqring_offsets: 40 bytes.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct SqOffsets {
        head: u32,
        tail: u32,
        ring_mask: u32,
        ring_entries: u32,
        flags: u32,
        dropped: u32,
        array: u32,
        resv1: u32,
        user_addr: u64,
    }

    /// struct io_cqring_offsets: 40 bytes.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct CqOffsets {
        head: u32,
        tail: u32,
        ring_mask: u32,
        ring_entries: u32,
        overflow: u32,
        cqes: u32,
        flags: u32,
        resv1: u32,
        user_addr: u64,
    }

    /// struct io_uring_params: 120 bytes.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Params {
        sq_entries: u32,
        cq_entries: u32,
        flags: u32,
        sq_thread_cpu: u32,
        sq_thread_idle: u32,
        features: u32,
        wq_fd: u32,
        resv: [u32; 3],
        sq_off: SqOffsets,
        cq_off: CqOffsets,
    }

    // The layouts, held to the header's own sizes.
    const _: () = assert!(std::mem::size_of::<Sqe>() == 64);
    const _: () = assert!(std::mem::size_of::<Cqe>() == 16);
    const _: () = assert!(std::mem::size_of::<SqOffsets>() == 40);
    const _: () = assert!(std::mem::size_of::<CqOffsets>() == 40);
    const _: () = assert!(std::mem::size_of::<Params>() == 120);

    /// Why a ring did not set up.
    #[derive(Debug)]
    pub struct InitError(pub i32);

    impl std::fmt::Display for InitError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "io_uring_setup failed ({})", self.0)
        }
    }

    impl std::error::Error for InitError {}

    /// Why a submit or a wait failed.
    #[derive(Debug)]
    pub struct EnterError(pub i32);

    impl std::fmt::Display for EnterError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "io_uring_enter failed ({})", self.0)
        }
    }

    impl std::error::Error for EnterError {}

    /// The submission and completion rings around the setup fd —
    /// upstream: `std.os.linux.IoUring`.
    pub struct Uring {
        fd: i32,
        /// The three mappings, for the close.
        sq_base: *mut u8,
        sq_len: usize,
        sqes_base: *mut Sqe,
        sqes_len: usize,
        cq_base: *mut u8,
        cq_len: usize,
        /// The SQ ring's head, tail and flags, and the array of SQE
        /// indices, mapped together.
        sq_head: *const AtomicU32,
        sq_tail: *const AtomicU32,
        sq_mask: u32,
        sq_array: *mut u32,
        /// Our submission tail (the kernel's head is read at submit).
        tail: u32,
        /// The CQ ring's head and tail, the ring's entries.
        cq_head: *const AtomicU32,
        cq_tail: *const AtomicU32,
        cq_mask: u32,
        cqes: *mut Cqe,
    }

    impl Uring {
        /// Upstream: `IoUring.init(entries, flags)` — the setup syscall
        /// and the three mmaps its params point to.
        pub fn init(entries: u32, flags: u32) -> Result<Uring, InitError> {
            let mut params = Params {
                flags,
                ..Default::default()
            };
            // SAFETY: the io_uring_setup syscall over the params.
            let fd = unsafe { syscall(IO_URING_SETUP, entries as usize, &mut params as *mut _) }
                as i32;
            if fd < 0 {
                return Err(InitError(std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)));
            }
            let sq = map(
                fd,
                OFF_SQ_RING,
                params.sq_off.array as usize + params.sq_entries as usize * 4,
            );
            let sqes = map(
                fd,
                OFF_SQES,
                params.sq_entries as usize * std::mem::size_of::<Sqe>(),
            );
            let cq = map(
                fd,
                OFF_CQ_RING,
                params.cq_off.cqes as usize + params.cq_entries as usize * std::mem::size_of::<Cqe>(),
            );
            let sq_bytes = sq.0 as *const u8;
            let cq_bytes = cq.0 as *const u8;

            // SAFETY: the mapped rings; the setup fd holds them.
            unsafe {
                Ok(Uring {
                    fd,
                    sq_base: sq.0 as *mut u8,
                    sq_len: sq.1,
                    sqes_base: sqes.0 as *mut Sqe,
                    sqes_len: sqes.1,
                    cq_base: cq.0 as *mut u8,
                    cq_len: cq.1,
                    sq_head: sq_bytes.add(params.sq_off.head as usize) as *const AtomicU32,
                    sq_tail: sq_bytes.add(params.sq_off.tail as usize) as *const AtomicU32,
                    sq_mask: std::ptr::read(
                        sq_bytes.add(params.sq_off.ring_mask as usize) as *const u32,
                    ),
                    sq_array: sq_bytes.add(params.sq_off.array as usize) as *mut u32,
                    tail: 0,
                    cq_head: cq_bytes.add(params.cq_off.head as usize) as *const AtomicU32,
                    cq_tail: cq_bytes.add(params.cq_off.tail as usize) as *const AtomicU32,
                    cq_mask: std::ptr::read(
                        cq_bytes.add(params.cq_off.ring_mask as usize) as *const u32,
                    ),
                    cqes: cq_bytes.add(params.cq_off.cqes as usize) as *mut Cqe,
                })
            }
        }

        /// One read prepared: `user_data` comes back with its
        /// completion. Upstream: `IoUring.read`.
        pub fn read(&mut self, user_data: u64, fd: i32, buf: &mut [u8], offset: u64) {
            let at = (self.tail & self.sq_mask) as usize;
            // SAFETY: the SQE slot at our tail, within the mapped array.
            unsafe {
                let sqe = self.sqes_base.add(at);
                std::ptr::write(
                    sqe,
                    Sqe {
                        opcode: OP_READ,
                        flags: 0,
                        ioprio: 0,
                        fd,
                        off: offset,
                        addr: buf.as_mut_ptr() as u64,
                        len: buf.len() as u32,
                        rw_flags: 0,
                        user_data,
                        buf_index: 0,
                        personality: 0,
                        splice_fd_in: 0,
                        pad: [0; 2],
                    },
                );
                *self.sq_array.add(at) = at as u32;
            }
            self.tail = self.tail.wrapping_add(1);
            // SAFETY: our tail, released to the kernel.
            unsafe { (*self.sq_tail).store(self.tail, Ordering::Release) };
        }

        /// Upstream: `IoUring.submit` — the pending SQEs entered.
        pub fn submit(&mut self) -> Result<u32, EnterError> {
            let head = unsafe { (*self.sq_head).load(Ordering::Acquire) };
            let pending = self.tail.wrapping_sub(head);
            if pending > 0 {
                self.enter(pending, 0, 0)?;
            }
            Ok(pending)
        }

        /// Completions copied, up to `cqes.len()`, waiting for at least
        /// `wait` when asked. Upstream: `IoUring.copy_cqes`.
        pub fn copy_cqes(&mut self, cqes: &mut [Cqe], wait: u32) -> Result<usize, EnterError> {
            if wait > 0 {
                self.enter(0, wait, ENTER_GETEVENTS)?;
            }
            // SAFETY: the CQ ring's head and tail, acquired.
            let head = unsafe { (*self.cq_head).load(Ordering::Acquire) };
            let tail = unsafe { (*self.cq_tail).load(Ordering::Acquire) };
            let ready = tail.wrapping_sub(head).min(cqes.len() as u32);
            for (i, out) in cqes.iter_mut().enumerate().take(ready as usize) {
                // SAFETY: the i-th completion since head, within the ring.
                let cqe = unsafe { self.cqes.add((head.wrapping_add(i as u32) & self.cq_mask) as usize) };
                *out = unsafe { std::ptr::read(cqe) };
            }
            if ready > 0 {
                // SAFETY: the head advanced past what we copied.
                unsafe { (*self.cq_head).store(head.wrapping_add(ready), Ordering::Release) };
            }
            Ok(ready as usize)
        }

        /// The raw entry — upstream: `IoUring.enter`.
        fn enter(&mut self, to_submit: u32, min_complete: u32, flags: u32) -> Result<(), EnterError> {
            // SAFETY: the io_uring_enter syscall.
            let r = unsafe {
                syscall(
                    IO_URING_ENTER,
                    self.fd as usize,
                    to_submit as usize,
                    min_complete as usize,
                    flags as usize,
                    0usize,
                    0usize,
                )
            };
            if r < 0 {
                return Err(EnterError(
                    std::io::Error::last_os_error().raw_os_error().unwrap_or(-1),
                ));
            }
            Ok(())
        }
    }

    impl Drop for Uring {
        fn drop(&mut self) {
            // SAFETY: the mappings and the fd made at init, unmade once.
            unsafe {
                munmap(self.sq_base as *mut core::ffi::c_void, self.sq_len);
                munmap(self.sqes_base as *mut core::ffi::c_void, self.sqes_len);
                munmap(self.cq_base as *mut core::ffi::c_void, self.cq_len);
                close(self.fd);
            }
        }
    }

    /// One mapping: (addr, len).
    fn map(fd: i32, offset: i64, len: usize) -> (*mut core::ffi::c_void, usize) {
        // SAFETY: the ring mapping at its own offset.
        let addr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                offset,
            )
        };
        (addr, len)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The port's hand-rolled ABI carries the ring test std's own
        /// IoUring carries upstream: two reads submitted, both
        /// completed, the user_data tags and the bytes intact.
        #[test]
        fn reads_submit_and_complete() {
            use std::io::Read;
            use std::os::fd::AsRawFd;
            // the draft vocabulary: a file the crate always carries,
            // comfortably past two 4 KiB reads
            let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/draft_ids.txt");
            let mut file = std::fs::File::open(path).expect("this crate's own source");
            let mut whole = Vec::new();
            file.read_to_end(&mut whole).expect("the file reads");
            assert!(whole.len() > 8192);
            let mut ring = Uring::init(4, 0).expect("io_uring on this kernel");
            let mut a = vec![0u8; 4096];
            let mut b = vec![0u8; 4096];
            ring.read(7, file.as_raw_fd(), &mut a, 0);
            ring.read(11, file.as_raw_fd(), &mut b, 4096);
            assert_eq!(2, ring.submit().unwrap());
            let mut cqes = [Cqe { user_data: 0, res: 0, flags: 0 }; 4];
            let mut got = 0;
            while got < 2 {
                got += ring.copy_cqes(&mut cqes, 1).unwrap();
            }
            let mut seen = [(0u64, 0i32); 2];
            for c in &cqes[..2] {
                seen[if c.user_data == 7 { 0 } else { 1 }] = (c.user_data, c.res);
            }
            assert_eq!((7, 4096), seen[0]);
            assert_eq!((11, 4096), seen[1]);
            assert_eq!(&whole[..4096], &a);
            assert_eq!(&whole[4096..8192], &b);
        }
    }
}

#[cfg(target_os = "linux")]
pub use ring::{Cqe, EnterError, InitError, Sqe, Uring};

#[cfg(not(target_os = "linux"))]
pub use absent::{Cqe, EnterError, InitError, Sqe, Uring};

#[cfg(not(target_os = "linux"))]
mod absent {
    /// The ring's host-only names: upstream's IoUring lives in
    /// std.os.linux and the source catches its absence — the port's
    /// source code names the type everywhere, so these stand in off
    /// Linux, never constructed.
    #[derive(Debug)]
    pub struct InitError(pub i32);
    impl std::fmt::Display for InitError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "io_uring is unavailable on this platform")
        }
    }
    impl std::error::Error for InitError {}

    #[derive(Debug)]
    pub struct EnterError(pub i32);
    impl std::fmt::Display for EnterError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "io_uring is unavailable on this platform")
        }
    }
    impl std::error::Error for EnterError {}

    #[derive(Clone, Copy)]
    pub struct Sqe;

    #[derive(Clone, Copy)]
    pub struct Cqe {
        pub user_data: u64,
        pub res: i32,
        pub flags: u32,
    }

    pub struct Uring;

    impl Uring {
        pub fn init(_: u32, _: u32) -> Result<Uring, InitError> {
            Err(InitError(-1))
        }
        pub fn read(&mut self, _: u64, _: i32, _: &mut [u8], _: u64) {}
        pub fn submit(&mut self) -> Result<u32, EnterError> {
            Err(EnterError(-1))
        }
        pub fn copy_cqes(&mut self, _: &mut [Cqe], _: u32) -> Result<usize, EnterError> {
            Err(EnterError(-1))
        }
    }
}
