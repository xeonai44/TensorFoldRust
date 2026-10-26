//! The device, its command queue, and buffers.
//!
//! Port of the pieces of `zig/src/metal/device.zig`, `command.zig` and `types.zig` the
//! Phase 0 runtime needs. All GPU objects are created inside an autorelease pool, the
//! way upstream's `mtl.objc.Pool.push()` does in its `main`s; here the pool wraps each
//! entry point.

use ::objc::{msg_send, sel, sel_impl};
use metal::{CommandQueue, MTLResourceOptions, MTLSize};

use crate::objc::Id;

/// The system's default GPU, or None where none serves.
pub struct Device(pub(crate) metal::Device);

/// One GPU: a device and its queue together, what an engine opens.
///
/// Upstream keeps a device and a queue on its backend struct the same way
/// (`families/nemotron/backend.zig` holds `mtl.Device` + queue).
pub struct Gpu {
    device: Device,
    queue: CommandQueue,
}

impl Gpu {
    /// Open the default GPU. The queue is the engine's own, like upstream.
    pub fn open() -> Option<Gpu> {
        objc::rc::autoreleasepool(|| {
            let device = metal::Device::system_default()?;
            let queue = device.new_command_queue();
            Some(Gpu {
                device: Device(device),
                queue,
            })
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The queue's command-buffer source, for `encoder::ComputePass`.
    pub(crate) fn queue_ref(&self) -> &metal::CommandQueueRef {
        &self.queue
    }

    /// A new buffer filled with `bytes` (upstream `new_buffer_with_bytes`).
    pub fn buffer_with_data(&self, bytes: &[u8]) -> Buffer {
        objc::rc::autoreleasepool(|| {
            let inner = self.device.0.new_buffer_with_data(
                bytes.as_ptr() as *const std::ffi::c_void,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            Buffer { inner, len: bytes.len() }
        })
    }

    /// A new zeroed buffer of `len` bytes.
    pub fn buffer_with_len(&self, len: usize) -> Buffer {
        self.buffer_with_options(len, MTLResourceOptions::StorageModeShared)
    }

    /// A new zeroed buffer with explicit resource options — the checkpoint's shards
    /// ask for shared+untracked (upstream `state.zig:5` and `checkpoint_metal.zig`).
    pub fn buffer_with_options(&self, len: usize, options: MTLResourceOptions) -> Buffer {
        objc::rc::autoreleasepool(|| {
            let inner = self.device.0.new_buffer(len as u64, options);
            Buffer { inner, len }
        })
    }

    /// Upstream's `shared | untracked` resource options, what every resident weight
    /// and cache buffer uses.
    pub fn shared_untracked() -> MTLResourceOptions {
        MTLResourceOptions::StorageModeShared | MTLResourceOptions::HazardTrackingModeUntracked
    }

    /// Run `pipeline` over `buffers` and wait for it — Phase 0's synchronous dispatch,
    /// the shape of upstream's test helpers (`tests/common.zig  fn buffer` and the
    /// runtime tests' round-trips). Grid and group sizes are thread counts per axis.
    pub fn dispatch(
        &self,
        pipeline: &super::pipeline::Pipeline,
        buffers: &[&Buffer],
        grid: (u64, u64, u64),
        group: (u64, u64, u64),
    ) {
        objc::rc::autoreleasepool(|| {
            let command = self.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline.0);
            for (i, b) in buffers.iter().enumerate() {
                encoder.set_buffer(i as u64, Some(&b.inner), 0);
            }
            encoder.dispatch_threads(
                MTLSize {
                    width: grid.0,
                    height: grid.1,
                    depth: grid.2,
                },
                MTLSize {
                    width: group.0,
                    height: group.1,
                    depth: group.2,
                },
            );
            encoder.end_encoding();
            command.commit();
            command.wait_until_completed();
        })
    }
}

/// A Metal buffer the host can read and write. Cloning retains, like upstream's
/// `mtl.Buffer` value copies.
#[derive(Clone)]
pub struct Buffer {
    pub(crate) inner: metal::Buffer,
    len: usize,
}

impl Buffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The buffer's base pointer (the MTLBuffer contents), for views at offsets.
    ///
    /// Safety: valid as long as the buffer lives; the phase-0 forwards always wait
    /// for completion before the host reads.
    pub fn as_ptr(&self) -> *mut u8 {
        self.inner.contents() as *mut u8
    }

    /// The buffer's bytes, mutable, for parallel loads (the checkpoint's read threads).
    ///
    /// Safety: valid as long as the buffer lives and no GPU work reads it
    /// concurrently; loads happen before any dispatch.
    pub fn as_mut_ptr_bytes(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.inner.contents() as *mut u8, self.len) }
    }

    /// A host view of `bytes` at `at`, the shape of upstream's
    /// `(buffer.contents() + at)[0..bytes]` slice on shared storage.
    ///
    /// Safety: the view must be read before any new GPU work is encoded; the
    /// forwards wait for completion before their host phases, so no in-flight
    /// command buffer writes the bytes.
    pub fn host_bytes(&self, at: usize, bytes: usize) -> &[u8] {
        debug_assert!(at + bytes <= self.len);
        unsafe { std::slice::from_raw_parts(self.as_ptr().add(at), bytes) }
    }

    /// The buffer's bytes, shared with the GPU (unsafe to write while a command buffer
    /// that reads it is in flight; the Phase 0 dispatches all wait).
    pub fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.inner.contents() as *const u8, self.len) }
    }

    /// The buffer's bytes, mutable (same rule as `bytes`).
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.inner.contents() as *mut u8, self.len) }
    }
}

impl Device {
    /// The N of the GPU's applegpu_gN architecture (M1 is 13, M5 17), 0 when it
    /// names none.
    ///
    /// Upstream: `device.zig:55  generation` — `respondsToSelector:` guards the
    /// Metal 4 `architecture` selector, then the architecture's name parses.
    pub fn generation(&self) -> u32 {
        unsafe {
            let responds: bool = msg_send![self.id(), respondsToSelector: sel!(architecture)];
            if !responds {
                return 0;
            }
            let arch: Id = msg_send![self.id(), architecture];
            if arch.is_null() {
                return 0;
            }
            let name: Id = msg_send![arch, name];
            generation_of(&crate::objc::nsstring_text(name))
        }
    }

    /// Metal 4 tensor units (applegpu_g17, the M5, and later).
    ///
    /// Upstream: `device.zig:63  tensorUnits`.
    pub fn tensor_units(&self) -> bool {
        self.generation() >= 17
    }

    /// The device's name (device.zig:51).
    pub fn name(&self) -> String {
        self.0.name().to_string()
    }

    /// The recommended working set's byte cap (device.zig:67's maxWorkingSet —
    /// the recommended cap is the fact Apple reports).
    pub fn max_working_set(&self) -> u64 {
        self.0.recommended_max_working_set_size()
    }

    /// The kernels the build embedded, as a library on this device.
    pub fn library_from_builtin(&self) -> Result<metal::Library, String> {
        objc::rc::autoreleasepool(|| self.0.new_library_with_data(super::METALLIB))
    }

    /// One function as a runnable pipeline on this device.
    pub fn compute_pipeline_state(
        &self,
        function: &metal::FunctionRef,
    ) -> Result<metal::ComputePipelineState, String> {
        objc::rc::autoreleasepool(|| self.0.new_compute_pipeline_state_with_function(function))
    }
}

/// The N of an "applegpu_gN..." architecture name ("applegpu_g15d": 15), 0 for any
/// other.
///
/// Upstream: `device.zig:120  generationOf`.
pub fn generation_of(arch: &str) -> u32 {
    const PREFIX: &str = "applegpu_g";
    let Some(rest) = arch.strip_prefix(PREFIX) else {
        return 0;
    };
    let end = rest
        .bytes()
        .take_while(|b| b.is_ascii_digit())
        .count();
    rest[..end].parse::<u32>().unwrap_or(0)
}
