//! The compute encoder: one command buffer's compute pass, the port of
//! `zig/src/metal/command.zig`'s ComputeEncoder half.
//!
//! Upstream's encoder is a raw id over `setBuffer:offset:atIndex:`,
//! `setBytes:length:atIndex:` and `dispatchThreads:threadsPerThreadgroup:`; the port
//! wraps the `metal` crate's encoder (same selectors) behind the same method shapes,
//! so `forward.zig`'s calls translate line for line.

use crate::device::Gpu;
use crate::pipeline::Pipeline;
use metal::{MTLCommandBufferStatus, MTLSize};
use ::objc::{msg_send, sel, sel_impl};

/// One GPU: a command buffer with its compute encoder, the unit a forward encodes
/// into. Borrowed from the `Gpu` it came from; `end()` closes the encoder and
/// `commit()` hands the buffer to the GPU.
pub struct ComputePass<'a> {
    command: &'a metal::CommandBufferRef,
    encoder: &'a metal::ComputeCommandEncoderRef,
}

impl Gpu {
    /// A serial compute pass (upstream `command.zig:45  compute(.serial)`).
    pub fn compute_pass(&self) -> ComputePass<'_> {
        let command = self.queue_ref().new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        ComputePass { command, encoder }
    }
}

impl<'a> ComputePass<'a> {
    /// The encoder this pass exposes; its lifetime is the pass's.
    pub fn encoder(&self) -> Encoder<'a> {
        Encoder { encoder: self.encoder }
    }

    /// End the encoder and hand the buffer to the GPU, then wait for it.
    ///
    /// Upstream: `forward.zig:107-109  e.end(); cb.commit(); cb.wait()`, with the
    /// failure read after the wait.
    pub fn submit(self) -> Result<(), String> {
        self.encoder.end_encoding();
        self.command.commit();
        self.command.wait_until_completed();
        if self.command.status() != MTLCommandBufferStatus::Completed {
            let message = unsafe {
                let error: *mut ::objc::runtime::Object = msg_send![self.command, error];
                if error.is_null() {
                    "command buffer failed".to_string()
                } else {
                    let desc: *mut ::objc::runtime::Object =
                        msg_send![error, localizedDescription];
                    crate::objc::nsstring_text(desc)
                }
            };
            return Err(message);
        }
        Ok(())
    }
}

/// The compute encoder: pipelines, buffers, inline values and dispatches.
///
/// Upstream: `command.zig` `ComputeEncoder` — `setPipeline`, `setBuffer`,
/// `setValue`/`setBytes`, `dispatchThreads`, `end`.
pub struct Encoder<'a> {
    encoder: &'a metal::ComputeCommandEncoderRef,
}

impl<'a> Encoder<'a> {
    /// Upstream: `command.zig` `setPipeline`.
    pub fn set_pipeline(&self, pipeline: &Pipeline) {
        self.encoder.set_compute_pipeline_state(&pipeline.0);
    }

    /// Upstream: `command.zig:137  setBuffer(buffer, offset, index)`.
    pub fn set_buffer(&self, buffer: &crate::device::Buffer, offset: usize, index: usize) {
        self.encoder
            .set_buffer(index as u64, Some(&buffer.inner), offset as u64);
    }

    /// Upstream: `command.zig:142  setBytes(bytes, index)`.
    pub fn set_bytes(&self, index: usize, bytes: &[u8]) {
        self.encoder
            .set_bytes(index as u64, bytes.len() as u64, bytes.as_ptr() as _);
    }

    /// Upstream: `command.zig:146  setValue(value, index)` — any fixed-size value as
    /// inline bytes.
    pub fn set_value<T: Copy>(&self, value: &T, index: usize) {
        let bytes = unsafe {
            std::slice::from_raw_parts(std::ptr::from_ref(value).cast::<u8>(), std::mem::size_of::<T>())
        };
        self.set_bytes(index, bytes);
    }

    /// Upstream: `command.zig:155  dispatchGroups(groups, threads)` — whole-group
    /// dispatch: the grid counts threadgroups, not threads.
    pub fn dispatch_groups(&self, grid: (u64, u64, u64), group: (u64, u64, u64)) {
        self.encoder.dispatch_thread_groups(
            MTLSize { width: grid.0, height: grid.1, depth: grid.2 },
            MTLSize { width: group.0, height: group.1, depth: group.2 },
        );
    }

    /// Upstream: `command.zig:165  dispatchThreads(grid, threads)` — whole-thread
    /// dispatch, grids need not divide into groups.
    pub fn dispatch(&self, grid: (u64, u64, u64), group: (u64, u64, u64)) {
        self.encoder.dispatch_threads(
            MTLSize { width: grid.0, height: grid.1, depth: grid.2 },
            MTLSize { width: group.0, height: group.1, depth: group.2 },
        );
    }

    /// Upstream: `command.zig` `end`.
    pub fn end(&self) {
        self.encoder.end_encoding();
    }
}
