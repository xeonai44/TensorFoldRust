//! Device memory and pinned host memory, each owned by one value; copies
//! and fills, blocking or on a stream.
//!
//! Port of `zig/src/cuda/memory.zig` (253 lines): the usage counters
//! (device, host, the peak since the last reset) with their lock, the
//! device buffer (its spans refused out of range, its fills whole-word),
//! and the pinned host buffer with its mapped variant.

use crate::abi;
use crate::driver::{Driver, Error};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Bytes this process holds in DeviceBuffers and HostBuffers, and the
/// device peak since the last reset. Upstream: `Usage` (memory.zig:10).
pub struct Usage {
    pub device: u64,
    pub host: u64,
    pub peak: u64,
}

static DEVICE_BYTES: AtomicU64 = AtomicU64::new(0);
static HOST_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);
static COUNTS: Mutex<()> = Mutex::new(());

/// The counts now; `reset_peak` starts a new peak at the current device
/// bytes. Safe from any thread. Upstream: `usage` (memory.zig:16).
pub fn usage(reset_peak: bool) -> Usage {
    usage_with_hook(reset_peak, &mut |_read| {})
}

/// Upstream: `usageWithHook` (memory.zig:22) — a hook runs between the
/// read and the store, so the peak reset can be proven atomic against a
/// concurrent allocation (its third test).
fn usage_with_hook(reset_peak: bool, hook: &mut dyn FnMut(u64)) -> Usage {
    let _guard = COUNTS.lock().unwrap();
    let now = DEVICE_BYTES.load(Ordering::Relaxed);
    hook(now);
    if reset_peak {
        PEAK_BYTES.store(now, Ordering::Relaxed);
    }
    Usage {
        device: now,
        host: HOST_BYTES.load(Ordering::Relaxed),
        peak: now.max(PEAK_BYTES.load(Ordering::Relaxed)),
    }
}

fn held(counter: &AtomicU64, n: usize) {
    let is_device = std::ptr::eq(counter, &DEVICE_BYTES);
    let _guard = is_device.then(|| COUNTS.lock().unwrap());
    let now = counter.fetch_add(n as u64, Ordering::Relaxed) + n as u64;
    if is_device {
        PEAK_BYTES.fetch_max(now, Ordering::Relaxed);
    }
}

fn freed(counter: &AtomicU64, n: usize) {
    let is_device = std::ptr::eq(counter, &DEVICE_BYTES);
    let _guard = is_device.then(|| COUNTS.lock().unwrap());
    counter.fetch_sub(n as u64, Ordering::Relaxed);
}

/// Upstream: `DeviceBuffer` (memory.zig:44).
pub struct DeviceBuffer<'d> {
    pub d: &'d Driver,
    pub ptr: abi::DevicePtr,
    pub len: usize,
    /// A span of memory another value owns (a carveout's): free leaves
    /// it alone.
    pub borrowed: bool,
}

impl<'d> DeviceBuffer<'d> {
    /// `len` bytes, 256-byte aligned by the driver; zero bytes allocate
    /// nothing and hold address 0. Upstream: `alloc` (memory.zig:48).
    pub fn alloc(d: &'d Driver, len: usize) -> Result<DeviceBuffer<'d>, Error> {
        let mut p: abi::DevicePtr = 0;
        if len > 0 {
            // SAFETY: the device allocation.
            let res = unsafe { (d.api.cuMemAlloc_v2.unwrap_unchecked())(&mut p, len) };
            d.check(res, "cuMemAlloc")?;
        }
        held(&DEVICE_BYTES, len);
        Ok(DeviceBuffer { d, ptr: p, len, borrowed: false })
    }

    /// Allocates and fills from host bytes in one call. Upstream:
    /// `fromHost` (memory.zig:55).
    pub fn from_host(d: &'d Driver, bytes: &[u8]) -> Result<DeviceBuffer<'d>, Error> {
        let mut b = DeviceBuffer::alloc(d, bytes.len())?;
        if let Err(e) = b.upload(0, bytes) {
            b.free();
            return Err(e);
        }
        Ok(b)
    }

    pub fn free(&mut self) {
        if !self.borrowed {
            if self.ptr != 0 {
                // SAFETY: the device allocation freed once.
                unsafe { (self.d.api.cuMemFree_v2.unwrap_unchecked())(self.ptr) };
            }
            freed(&DEVICE_BYTES, self.len);
        }
    }

    /// The device address `offset` bytes in; out of range is refused.
    /// Upstream: `at` (memory.zig:66).
    pub fn at(&self, offset: usize) -> Result<abi::DevicePtr, Error> {
        if offset > self.len {
            return Err(Error::Invalid);
        }
        Ok(self.ptr + offset as u64)
    }

    fn span(&self, offset: usize, n: usize) -> Result<abi::DevicePtr, Error> {
        if offset > self.len || n > self.len - offset {
            return Err(Error::Invalid);
        }
        Ok(self.ptr + offset as u64)
    }

    /// Returns once the bytes have landed: a pageable copy can return
    /// early, and non-blocking streams won't wait. Upstream: `upload`
    /// (memory.zig:75).
    pub fn upload(&self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        let dst = self.span(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        // SAFETY: the host bytes copied into the span.
        let res = unsafe {
            (self.d.api.cuMemcpyHtoD_v2.unwrap_unchecked())(
                dst,
                bytes.as_ptr().cast(),
                bytes.len(),
            )
        };
        self.d.check(res, "cuMemcpyHtoD")?;
        // SAFETY: the null stream's work drained (the copy's own sync).
        let res = unsafe {
            (self.d.api.cuStreamSynchronize.unwrap_unchecked())(std::ptr::null_mut())
        };
        self.d.check(res, "cuStreamSynchronize")
    }

    pub fn download(&self, offset: usize, out: &mut [u8]) -> Result<(), Error> {
        let src = self.span(offset, out.len())?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: the span copied into the host bytes.
        let res = unsafe {
            (self.d.api.cuMemcpyDtoH_v2.unwrap_unchecked())(
                out.as_mut_ptr().cast(),
                src,
                out.len(),
            )
        };
        self.d.check(res, "cuMemcpyDtoH")
    }

    /// Asynchronous only when `bytes` is pinned (HostBuffer); pageable
    /// memory makes the driver stage it. Upstream: `uploadAsync`
    /// (memory.zig:86).
    /// The stream is an opaque driver handle, never dereferenced here.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn upload_async(
        &self,
        offset: usize,
        bytes: &[u8],
        stream: abi::Stream,
    ) -> Result<(), Error> {
        let dst = self.span(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        // SAFETY: the async host-to-device copy on the stream.
        let res = unsafe {
            (self.d.api.cuMemcpyHtoDAsync_v2.unwrap_unchecked())(
                dst,
                bytes.as_ptr().cast(),
                bytes.len(),
                stream,
            )
        };
        self.d.check(res, "cuMemcpyHtoDAsync")
    }

    /// The stream is an opaque driver handle, never dereferenced here.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn download_async(
        &self,
        offset: usize,
        out: &mut [u8],
        stream: abi::Stream,
    ) -> Result<(), Error> {
        let src = self.span(offset, out.len())?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: the async device-to-host copy on the stream.
        let res = unsafe {
            (self.d.api.cuMemcpyDtoHAsync_v2.unwrap_unchecked())(
                out.as_mut_ptr().cast(),
                src,
                out.len(),
                stream,
            )
        };
        self.d.check(res, "cuMemcpyDtoHAsync")
    }

    pub fn copy_from(
        &self,
        offset: usize,
        src: abi::DevicePtr,
        n: usize,
        stream: Option<abi::Stream>,
    ) -> Result<(), Error> {
        let dst = self.span(offset, n)?;
        if n == 0 {
            return Ok(());
        }
        match stream {
            // SAFETY: the async device-to-device copy on the stream.
            Some(s) => {
                let res = unsafe {
                    (self.d.api.cuMemcpyDtoDAsync_v2.unwrap_unchecked())(dst, src, n, s)
                };
                self.d.check(res, "cuMemcpyDtoDAsync")
            }
            // SAFETY: the blocking device-to-device copy.
            None => {
                let res = unsafe { (self.d.api.cuMemcpyDtoD_v2.unwrap_unchecked())(dst, src, n) };
                self.d.check(res, "cuMemcpyDtoD")
            }
        }
    }

    pub fn fill8(&self, value: u8, stream: Option<abi::Stream>) -> Result<(), Error> {
        if self.len == 0 {
            return Ok(());
        }
        match stream {
            // SAFETY: the async byte fill on the stream.
            Some(s) => {
                let res = unsafe {
                    (self.d.api.cuMemsetD8Async.unwrap_unchecked())(self.ptr, value, self.len, s)
                };
                self.d.check(res, "cuMemsetD8Async")
            }
            // SAFETY: the blocking byte fill.
            None => {
                let res =
                    unsafe { (self.d.api.cuMemsetD8_v2.unwrap_unchecked())(self.ptr, value, self.len) };
                self.d.check(res, "cuMemsetD8")
            }
        }
    }

    /// Fills whole 32-bit words; the length must be a multiple of four.
    /// Upstream: `fill32` (memory.zig:109).
    pub fn fill32(&self, value: u32, stream: Option<abi::Stream>) -> Result<(), Error> {
        if self.len % 4 != 0 {
            return Err(Error::Invalid);
        }
        if self.len == 0 {
            return Ok(());
        }
        match stream {
            // SAFETY: the async word fill on the stream.
            Some(s) => {
                let res = unsafe {
                    (self.d.api.cuMemsetD32Async.unwrap_unchecked())(
                        self.ptr,
                        value,
                        self.len / 4,
                        s,
                    )
                };
                self.d.check(res, "cuMemsetD32Async")
            }
            // SAFETY: the blocking word fill.
            None => {
                let res = unsafe {
                    (self.d.api.cuMemsetD32_v2.unwrap_unchecked())(self.ptr, value, self.len / 4)
                };
                self.d.check(res, "cuMemsetD32")
            }
        }
    }
}

impl Drop for DeviceBuffer<'_> {
    fn drop(&mut self) {
        self.free();
    }
}

/// Page-locked host memory: the only host memory an asynchronous copy
/// can read or write without staging. Upstream: `HostBuffer`
/// (memory.zig:128).
pub struct HostBuffer<'d> {
    pub d: &'d Driver,
    pub bytes: *mut u8,
    pub len: usize,
}

impl<'d> HostBuffer<'d> {
    pub fn alloc(d: &'d Driver, len: usize) -> Result<HostBuffer<'d>, Error> {
        HostBuffer::alloc_flags(d, len, abi::HOST_ALLOC_PORTABLE)
    }

    /// Pinned memory kernels write directly (mapped into the device's
    /// address space): no copy reads it back. Upstream: `allocMapped`
    /// (memory.zig:140).
    pub fn alloc_mapped(d: &'d Driver, len: usize) -> Result<HostBuffer<'d>, Error> {
        HostBuffer::alloc_flags(d, len, abi::HOST_ALLOC_PORTABLE | abi::HOST_ALLOC_DEVICEMAP)
    }

    /// The device address of a mapped buffer. Upstream: `device`
    /// (memory.zig:145).
    pub fn device(&self) -> Result<abi::DevicePtr, Error> {
        let mut p: abi::DevicePtr = 0;
        // SAFETY: the mapped buffer's device address read.
        let res = unsafe {
            (self.d.api.cuMemHostGetDevicePointer_v2.unwrap_unchecked())(
                &mut p,
                self.bytes.cast(),
                0,
            )
        };
        self.d.check(res, "cuMemHostGetDevicePointer")?;
        Ok(p)
    }

    fn alloc_flags(d: &'d Driver, len: usize, flags: std::ffi::c_uint) -> Result<HostBuffer<'d>, Error> {
        if len == 0 {
            return Err(Error::Invalid);
        }
        let mut p: *mut core::ffi::c_void = std::ptr::null_mut();
        // SAFETY: the pinned allocation with its flags.
        let res = unsafe { (d.api.cuMemHostAlloc.unwrap_unchecked())(&mut p, len, flags) };
        d.check(res, "cuMemHostAlloc")?;
        held(&HOST_BYTES, len);
        Ok(HostBuffer { d, bytes: p.cast(), len })
    }

    /// The pinned bytes as a slice.
    pub fn slice(&self) -> &[u8] {
        // SAFETY: the pinned allocation holds `len` readable bytes for
        // the buffer's lifetime.
        unsafe { std::slice::from_raw_parts(self.bytes, self.len) }
    }

    /// The pinned bytes as a mutable slice.
    pub fn slice_mut(&mut self) -> &mut [u8] {
        // SAFETY: the pinned allocation holds `len` writable bytes for
        // the buffer's lifetime, uniquely borrowed here.
        unsafe { std::slice::from_raw_parts_mut(self.bytes, self.len) }
    }
}

impl Drop for HostBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: the pinned allocation freed once.
        unsafe { (self.d.api.cuMemFreeHost.unwrap_unchecked())(self.bytes.cast()) };
        freed(&HOST_BYTES, self.len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The counters are global: the tests that read and move them run one
    // at a time
    static COUNTERS: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Upstream's `usage counts held bytes and the device peak`
    /// (memory.zig:167).
    #[test]
    fn usage_counts_held_bytes_and_the_device_peak() {
        let _guard = COUNTERS.lock().unwrap();
        let before = usage(true);
        held(&DEVICE_BYTES, 1000);
        held(&HOST_BYTES, 24);
        freed(&DEVICE_BYTES, 1000);
        let after = usage(false);
        assert_eq!(before.device, after.device);
        assert_eq!(before.host + 24, after.host);
        assert_eq!(before.device + 1000, after.peak);
        assert_eq!(before.device, usage(true).peak);
        freed(&HOST_BYTES, 24);
    }

    /// Upstream's `freeing a borrowed span leaves the driver and the
    /// counts alone` (memory.zig:180): a borrowed span's free never
    /// calls the driver — the stub driver proves it — and the counts
    /// stand.
    #[test]
    fn freeing_a_borrowed_span_leaves_the_driver_and_the_counts_alone() {
        let _guard = COUNTERS.lock().unwrap();
        let before = usage(false);
        let d = crate::driver::stub::stub();
        let mut b = DeviceBuffer {
            d,
            ptr: 0x1000,
            len: 4096,
            borrowed: true,
        };
        b.free();
        assert_eq!(before.device, usage(false).device, "no count moved");
    }

    /// Upstream's `peak reset preserves an allocation between its read
    /// and store` (memory.zig:185): a concurrent allocation tries the
    /// counts lock during the reset's read-store pair — the lock holds
    /// through the hook, the peak sees the concurrent bytes.
    #[test]
    fn peak_reset_preserves_an_allocation_between_its_read_and_store() {
        let _guard = COUNTERS.lock().unwrap();
        use std::sync::mpsc;
        let before = usage(true);
        held(&DEVICE_BYTES, 100);
        // The thread's signals: (tried-the-lock, whether it blocked),
        // then the hook's release
        let (tried_tx, tried_rx) = mpsc::channel::<bool>();
        let (free_tx, free_rx) = mpsc::channel::<()>();
        let thread = std::thread::spawn(move || {
            let acquired = COUNTS.try_lock();
            let blocked = acquired.is_err();
            let _ = tried_tx.send(blocked);
            if !blocked {
                let _guard = acquired;
                held(&DEVICE_BYTES, 300);
            } else {
                held(&DEVICE_BYTES, 300);
            }
            let _ = free_rx.recv();
            freed(&DEVICE_BYTES, 300);
        });
        let mut hook = move |_now: u64| {
            let blocked = tried_rx.recv().unwrap();
            if !blocked {
                // the allocation finished inside its guard; nothing to
                // wait for
            }
        };
        usage_with_hook(true, &mut hook);
        let _ = free_tx.send(());
        thread.join().unwrap();
        let after = usage(false);
        assert_eq!(before.device + 100, after.device);
        assert_eq!(before.device + 400, after.peak);
        freed(&DEVICE_BYTES, 100);
    }
}
