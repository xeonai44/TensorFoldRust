//! Checkpoint reads use io_uring or threads, overlapping allocations
//! and copies through O_DIRECT slots.
//!
//! Port of `zig/src/families/nemotron/cuda_source.zig` (241 lines) —
//! the queued reads carry the descriptor (upstream copies the File
//! value; the port's File owns its close, so `Pending` and `Job` hold
//! `fd` and `direct` instead), and the ring is the port's own
//! `uring` module where upstream imports std's.

use tensorfold_core::direct_io;
use tensorfold_cuda::{Event, HostBuffer};

use crate::cuda_kernels::Ops;
use crate::error::Error;
use crate::uring::{Cqe, Uring};
use direct_io::ALIGNMENT;

/// Bytes a slot holds: one read's span; a larger tensor goes as
/// several reads. Upstream: `slot_bytes` (cuda_source.zig:11).
pub const SLOT_BYTES: usize = 64 << 20;

/// Slots, and so reads in flight at once: enough to cover the loader's
/// largest allocation. Upstream: `slot_count` (cuda_source.zig:13).
pub const SLOT_COUNT: usize = 4;

/// The checkpoint's mapped files, each opened beside its mapping.
struct Mapped {
    base: usize,
    len: usize,
    fd: i32,
    direct: bool,
}

/// A slot's read in flight: `len` bytes from `offset`, landing `at`
/// into the slot, for the GPU at `dst`. Upstream: `Pending`
/// (cuda_source.zig:17).
#[derive(Clone, Copy)]
struct Pending {
    dst: u64,
    fd: i32,
    direct: bool,
    offset: u64,
    len: usize,
    at: usize,
}

/// A slot's read on its own thread where io_uring is unavailable: its
/// length or error once `done`. Upstream: `Job` (cuda_source.zig:21).
struct Job {
    fd: i32,
    direct: bool,
    buf: *mut u8,
    cap: usize,
    offset: u64,
    len: usize,
    got: std::cell::UnsafeCell<(i32, usize)>, // (verdict, n): zero the read's length
    done: std::sync::atomic::AtomicBool,
    thread: Option<std::thread::JoinHandle<()>>,
}

// The job's result crosses threads under its own done flag.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

/// The thread's own pointer to its job, the one the spawn moved (a
/// whole-value argument so the closure captures the wrapper, not the
/// raw field it wraps).
struct JobPtr(*mut Job);
unsafe impl Send for JobPtr {}

/// The thread's body: its job's read.
fn job_run(ptr: JobPtr) {
    // SAFETY: the one pointer the thread owns, until done.
    unsafe {
        (*ptr.0).run();
    }
}

impl Job {
    fn new() -> Job {
        Job {
            fd: -1,
            direct: false,
            buf: std::ptr::null_mut(),
            cap: 0,
            offset: 0,
            len: 0,
            got: std::cell::UnsafeCell::new((0, 0)),
            done: std::sync::atomic::AtomicBool::new(false),
            thread: None,
        }
    }

    /// Upstream: `Job.run` (cuda_source.zig:30).
    fn run(&mut self) {
        // SAFETY: the slot's span, ours until done.
        let buf = unsafe { std::slice::from_raw_parts_mut(self.buf, self.cap) };
        let r = direct_io::File::read_fd(self.fd, self.direct, buf, self.offset, self.len);
        let into = match r {
            Ok(n) => (0, n),
            Err(direct_io::ReadError::FileNotFound) => (-1, 0),
            Err(direct_io::ReadError::BufferTooSmall) => (-2, 0),
            Err(direct_io::ReadError::ReadFailed) => (-3, 0),
            Err(direct_io::ReadError::EndOfFile) => (-4, 0),
        };
        // SAFETY: our write, released by the flag below.
        unsafe { *self.got.get() = into };
        self.done
            .store(true, std::sync::atomic::Ordering::Release);
    }
}

/// Upstream: `Source` (cuda_source.zig:36) — the port's slots are an
/// aligned allocation (the alignment carried by the layout, not the
/// slice's type).
pub struct Source<'a, 'd> {
    ops: &'a Ops<'a, 'd>,
    files: Vec<Mapped>,
    slots: Vec<Slot>,
    /// A discrete card's slots, page-locked: copies run at the link's
    /// speed.
    pinned: Vec<Option<HostBuffer<'d>>>,
    /// A page-locked slot's last copy: its next read waits for that,
    /// not the stream.
    copied: Vec<Option<Event<'d>>>,
    pending: Vec<Option<Pending>>,
    in_flight: usize,
    /// None where io_uring is unavailable: each read then runs on its
    /// own thread.
    ring: Option<Uring>,
    jobs: Vec<Job>,
    /// The order reads were asked in: copies follow it without a ring.
    asked: Vec<u64>,
    next: u64,
}

/// One slot: its aligned storage, or the page-locked buffer's span.
enum Slot {
    Heap {
        raw: *mut u8,
        len: usize,
    },
    Pinned {
        at: *mut u8,
        len: usize,
    },
}

impl Slot {
    fn span(&mut self) -> &mut [u8] {
        match self {
            // SAFETY: the allocation's own span.
            Slot::Heap { raw, len } => unsafe { std::slice::from_raw_parts_mut(*raw, *len) },
            // SAFETY: the page-locked buffer's own span.
            Slot::Pinned { at, len } => unsafe { std::slice::from_raw_parts_mut(*at, *len) },
        }
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        if let Slot::Heap { raw, len } = *self {
            // SAFETY: the aligned allocation made at init, freed once.
            unsafe {
                std::alloc::dealloc(
                    raw,
                    std::alloc::Layout::from_size_align(len, ALIGNMENT).unwrap(),
                );
            }
        }
    }
}

impl<'a, 'd> Source<'a, 'd> {
    /// Page-aligned slots; on a discrete card page-locked, each with an
    /// event its copies record. Upstream: `init` (cuda_source.zig:51).
    pub fn init(ops: &'a Ops<'a, 'd>) -> Result<Source<'a, 'd>, Error> {
        let mut s = Source {
            ops,
            files: Vec::new(),
            slots: Vec::with_capacity(SLOT_COUNT),
            pinned: {
                let mut v = Vec::with_capacity(SLOT_COUNT);
                for _ in 0..SLOT_COUNT {
                    v.push(None);
                }
                v
            },
            copied: {
                let mut v = Vec::with_capacity(SLOT_COUNT);
                for _ in 0..SLOT_COUNT {
                    v.push(None);
                }
                v
            },
            pending: {
                let mut v = Vec::with_capacity(SLOT_COUNT);
                for _ in 0..SLOT_COUNT {
                    v.push(None);
                }
                v
            },
            in_flight: 0,
            ring: None,
            jobs: (0..SLOT_COUNT).map(|_| Job::new()).collect(),
            asked: vec![0; SLOT_COUNT],
            next: 0,
        };
        for _ in 0..SLOT_COUNT {
            if !ops.k.discrete {
                let layout =
                    std::alloc::Layout::from_size_align(SLOT_BYTES, ALIGNMENT).unwrap();
                // SAFETY: the aligned slot allocation.
                let raw = unsafe { std::alloc::alloc(layout) };
                if raw.is_null() {
                    return Err(Error::Cuda(
                        tensorfold_cuda::Error::OutOfDeviceMemory,
                    ));
                }
                s.slots.push(Slot::Heap {
                    raw,
                    len: SLOT_BYTES,
                });
                continue;
            }
            let h = HostBuffer::alloc(ops.k.d, SLOT_BYTES)?;
            let (at, len) = {
                let bytes = h.slice();
                (bytes.as_ptr() as *mut u8, bytes.len())
            };
            if at as usize % ALIGNMENT != 0 {
                return Err(Error::UnalignedPinnedSlot);
            }
            let event = match Event::init(ops.k.d, false) {
                Ok(e) => e,
                Err(e) => return Err(Error::Cuda(e)),
            };
            let made = s.slots.len();
            s.copied[made] = Some(event);
            s.pinned[made] = Some(h);
            s.slots.push(Slot::Pinned { at, len });
        }
        s.ring = match Uring::init(SLOT_COUNT as u32, 0) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("io_uring unavailable ({e}): checkpoint reads run on threads");
                None
            }
        };
        Ok(s)
    }

    /// Waits for reads still in flight, then frees the ring and slots
    /// and closes the files. Upstream: `deinit` (cuda_source.zig:80) —
    /// the port's Drop runs the same unwind at scope exit.
    pub fn deinit(&mut self) {
        while self.in_flight > 0 {
            if self.reap(1).is_err() {
                break;
            }
        }
        if self.ring.is_none() {
            self.join_all();
        }
        self.ring = None;
        self.slots.clear();
        self.copied.clear();
        self.pinned.clear();
        for f in &self.files {
            // SAFETY: the descriptor opened at add, closed once.
            unsafe { close_fd(f.fd) };
        }
        self.files.clear();
        self.in_flight = 0;
    }

    /// Threads still reading after a failed copy finish before their
    /// slots are freed. Upstream: `joinAll` (cuda_source.zig:90).
    fn join_all(&mut self) {
        for (j, p) in self.jobs.iter_mut().zip(self.pending.iter()) {
            if p.is_some() {
                if let Some(t) = j.thread.take() {
                    t.join().expect("the read thread");
                }
            }
        }
    }

    /// Slot k's last copy has read it, so new bytes can land there.
    /// Upstream: `settle` (cuda_source.zig:104).
    fn settle(&self, k: usize) -> Result<(), Error> {
        if let Some(e) = &self.copied[k] {
            e.synchronize()?;
        }
        Ok(())
    }

    /// Every file of `ck`, opened for direct reads beside its mapping
    /// (copies from the mapping crawl on GB10). Upstream: `add`
    /// (cuda_source.zig:110).
    pub fn add(&mut self, ck: &tensorfold_core::checkpoint::Checkpoint) -> Result<(), Error> {
        for (path, bytes) in ck.files() {
            let cpath = std::ffi::CString::new(path.to_string_lossy().as_bytes())
                .map_err(|_| Error::FileNotFound)?;
            let mut file = direct_io::File::open(&cpath)?;
            self.files.push(Mapped {
                base: bytes.as_ptr() as usize,
                len: bytes.len(),
                fd: file.fd,
                direct: file.direct,
            });
            // The Mapped owns the descriptor from here; the File's own
            // close would double it.
            file.fd = -1;
        }
        Ok(())
    }

    /// The file and offset of mapped bytes; None for bytes that live
    /// elsewhere (host staging). Upstream: `locate`
    /// (cuda_source.zig:119).
    fn locate(&self, bytes: &[u8]) -> Option<(&Mapped, u64)> {
        let p = bytes.as_ptr() as usize;
        for m in &self.files {
            if p >= m.base && p + bytes.len() <= m.base + m.len {
                return Some((m, (p - m.base) as u64));
            }
        }
        None
    }

    /// Mapped bytes read for the GPU at `dst`: their reads are in
    /// flight on return; other bytes go now, after a flush. Upstream:
    /// `upload` (cuda_source.zig:126).
    pub fn upload(&mut self, dst: u64, bytes: &[u8]) -> Result<(), Error> {
        let at = match self.locate(bytes) {
            Some((m, offset)) => (m.fd, m.direct, offset),
            None => {
                self.flush()?;
                return self.ops.upload(dst, bytes);
            }
        };
        let mut done: usize = 0;
        while done < bytes.len() {
            let n = (bytes.len() - done).min(direct_io::fits(SLOT_BYTES));
            let k = self.free_slot()?;
            self.start(
                k,
                Pending {
                    dst: dst + done as u64,
                    fd: at.0,
                    direct: at.1,
                    offset: at.2 + done as u64,
                    len: n,
                    at: 0,
                },
            )?;
            done += n;
        }
        Ok(())
    }

    /// Waits for every read in flight and queues its copy: stream work
    /// after this runs after them. Upstream: `flush`
    /// (cuda_source.zig:140).
    pub fn flush(&mut self) -> Result<(), Error> {
        while self.in_flight > 0 {
            self.reap(1)?;
        }
        Ok(())
    }

    /// A slot with no read in flight, once finished reads have been
    /// reaped (waiting for one when all are busy). Upstream: `free`
    /// (cuda_source.zig:145).
    fn free_slot(&mut self) -> Result<usize, Error> {
        self.reap(0)?;
        loop {
            for (k, p) in self.pending.iter().enumerate() {
                if p.is_none() {
                    return Ok(k);
                }
            }
            self.reap(1)?;
        }
    }

    /// Puts slot `k`'s read in flight: on the ring, else on its own
    /// thread (or here when no thread can start). Upstream: `start`
    /// (cuda_source.zig:154).
    fn start(&mut self, k: usize, p: Pending) -> Result<(), Error> {
        self.settle(k)?;
        let lo = p.offset & !(ALIGNMENT as u64 - 1);
        let q = Pending {
            at: (p.offset - lo) as usize,
            ..p
        };
        let span = direct_io::span(q.offset, q.len);
        let buf = self.slots[k].span();
        if let Some(ring) = &mut self.ring {
            ring.read(k as u64, q.fd, &mut buf[..span], lo);
            ring.submit()?;
        } else {
            let j = &mut self.jobs[k];
            j.fd = q.fd;
            j.direct = q.direct;
            j.buf = buf.as_mut_ptr();
            j.cap = buf.len();
            j.offset = q.offset;
            j.len = q.len;
            j.done
                .store(false, std::sync::atomic::Ordering::Release);
            let ptr = JobPtr(j as *mut Job);
            // SAFETY: the job is ours until the thread finishes; the
            // done flag orders the result.
            let spawned = std::thread::spawn(move || job_run(ptr));
            j.thread = Some(spawned);
        }
        self.pending[k] = Some(q);
        self.asked[k] = self.next;
        self.next += 1;
        self.in_flight += 1;
        Ok(())
    }

    /// Finished thread reads, oldest first, copied on; waits for at
    /// least `wait` of them. Upstream: `reapThreads`
    /// (cuda_source.zig:175).
    fn reap_threads(&mut self, wait: u32) -> Result<(), Error> {
        let mut reaped: u32 = 0;
        while self.in_flight > 0 {
            let mut k = SLOT_COUNT;
            for (i, p) in self.pending.iter().enumerate() {
                if p.is_some() && (k == SLOT_COUNT || self.asked[i] < self.asked[k]) {
                    k = i;
                }
            }
            let j = &mut self.jobs[k];
            if reaped >= wait
                && !j
                    .done
                    .load(std::sync::atomic::Ordering::Acquire)
            {
                return Ok(());
            }
            if let Some(t) = j.thread.take() {
                t.join().expect("the read thread");
            }
            // SAFETY: the thread's result, acquired by the flag.
            let (verdict, n) = unsafe { *j.got.get() };
            let p = self.pending[k].take().unwrap();
            self.in_flight -= 1;
            match verdict {
                0 => {
                    let base = self.slots[k].span().as_mut_ptr();
                    // SAFETY: the slot's read span, ended before the
                    // copy.
                    let bytes = unsafe { std::slice::from_raw_parts(base.add(p.at), n) };
                    self.finish(k, &p, bytes)?;
                }
                -1 => return Err(Error::FileNotFound),
                -2 => return Err(Error::BufferTooSmall),
                -3 => return Err(Error::ReadFailed),
                _ => return Err(Error::EndOfFile),
            }
            reaped += 1;
        }
        Ok(())
    }

    /// Finished reads, copied on; waits for at least `wait` of them.
    /// Upstream: `reap` (cuda_source.zig:194).
    fn reap(&mut self, wait: u32) -> Result<(), Error> {
        if self.ring.is_none() {
            return self.reap_threads(wait);
        }
        let mut cqes = [Cqe {
            user_data: 0,
            res: 0,
            flags: 0,
        }; SLOT_COUNT];
        let n = match &mut self.ring {
            Some(ring) => ring.copy_cqes(&mut cqes, wait)?,
            None => unreachable!(),
        };
        for c in &cqes[..n] {
            let k = c.user_data as usize;
            let p = self.pending[k].take().unwrap();
            self.in_flight -= 1;
            // a short or failed read is finished by a plain read of the
            // same span
            let base = self.slots[k].span().as_mut_ptr();
            if c.res < (p.at + p.len) as i32 {
                if c.res < 0 {
                    eprintln!("io_uring read at {} failed ({}); reading again", p.offset, c.res);
                }
                let buf = self.slots[k].span();
                direct_io::File::read_fd(p.fd, p.direct, buf, p.offset, p.len)?;
            }
            // SAFETY: the slot's requested span, its borrow ended.
            let bytes = unsafe { std::slice::from_raw_parts(base.add(p.at), p.len) };
            self.finish(k, &p, bytes)?;
        }
        Ok(())
    }

    /// Page-locked slots record their copy event; pageable slots wait
    /// for the driver to stage their bytes. Upstream: `finish`
    /// (cuda_source.zig:213).
    fn finish(&self, k: usize, p: &Pending, got: &[u8]) -> Result<(), Error> {
        if got.is_empty() {
            return Ok(());
        }
        // SAFETY: the staged copy on this stream.
        let res = unsafe {
            (self.ops.k.d.api.cuMemcpyHtoDAsync_v2.unwrap_unchecked())(
                p.dst,
                got.as_ptr().cast(),
                got.len(),
                self.ops.s.handle,
            )
        };
        self.ops.k.d.check(res, "cuMemcpyHtoDAsync")?;
        if let Some(e) = &self.copied[k] {
            e.record(self.ops.s)?;
            return Ok(());
        }
        self.ops.s.synchronize()?;
        Ok(())
    }

    /// Mapped bytes copied into host memory `out` (its length is
    /// theirs). Upstream: `read` (cuda_source.zig:221).
    pub fn read(&mut self, out: &mut [u8], bytes: &[u8]) -> Result<(), Error> {
        let at = match self.locate(bytes) {
            Some((m, offset)) => (m.fd, m.direct, offset),
            None => {
                out.copy_from_slice(bytes);
                return Ok(());
            }
        };
        self.flush()?;
        self.settle(0)?;
        let mut done: usize = 0;
        while done < bytes.len() {
            let n = (bytes.len() - done).min(direct_io::fits(SLOT_BYTES));
            let buf = self.slots[0].span();
            direct_io::File::read_fd(at.0, at.1, buf, at.2 + done as u64, n)?;
            let buf = self.slots[0].span();
            out[done..done + n].copy_from_slice(&buf[..n]);
            done += n;
        }
        Ok(())
    }

    /// Small mapped bytes as a view into a slot, valid until the next
    /// call; larger than a slot is refused. Upstream: `view`
    /// (cuda_source.zig:234).
    pub fn view<'s>(&'s mut self, bytes: &'s [u8]) -> Result<&'s [u8], Error> {
        let at = match self.locate(bytes) {
            Some((m, offset)) => (m.fd, m.direct, offset),
            None => return Ok(bytes),
        };
        if bytes.len() > direct_io::fits(SLOT_BYTES) {
            return Err(Error::TensorLargerThanSlot);
        }
        self.flush()?;
        self.settle(0)?;
        let n = direct_io::File::read_fd(at.0, at.1, self.slots[0].span(), at.2, bytes.len())?;
        let buf = self.slots[0].span();
        Ok(&buf[..n])
    }
}

impl Drop for Source<'_, '_> {
    fn drop(&mut self) {
        self.deinit();
    }
}

/// The platform's close, the way direct_io's own File does it.
unsafe fn close_fd(fd: i32) {
    unsafe extern "C" {
        fn close(fd: i32) -> i32;
    }
    // SAFETY: the descriptor opened at add.
    unsafe { close(fd) };
}
