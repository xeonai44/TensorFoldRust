//! Streams and events: ordering, waits and GPU-side timing.
//!
//! Port of `zig/src/cuda/stream.zig` (79 lines).

use crate::abi;
use crate::driver::{Driver, Error};

/// Upstream: ``Stream`` (stream.zig:7).
pub struct Stream<'d> {
    pub d: &'d Driver,
    pub handle: abi::Stream,
}

impl<'d> Stream<'d> {
    /// A non-blocking stream never waits on the legacy default stream
    /// (what every engine stream should be). Upstream: `init`
    /// (stream.zig:12).
    pub fn init(d: &'d Driver, non_blocking: bool) -> Result<Stream<'d>, Error> {
        let mut s: abi::Stream = std::ptr::null_mut();
        let flags = if non_blocking { abi::STREAM_NON_BLOCKING } else { 0 };
        // SAFETY: the stream created with its flags.
        let res = unsafe { (d.api.cuStreamCreate.unwrap_unchecked())(&mut s, flags) };
        d.check(res, "cuStreamCreate")?;
        Ok(Stream { d, handle: s })
    }

    pub fn synchronize(&self) -> Result<(), Error> {
        // SAFETY: the stream's work drained.
        let res = unsafe { (self.d.api.cuStreamSynchronize.unwrap_unchecked())(self.handle) };
        self.d.check(res, "cuStreamSynchronize")
    }

    /// True once every queued item has finished. Upstream: `done`
    /// (stream.zig:29).
    pub fn done(&self) -> Result<bool, Error> {
        // SAFETY: the stream's queue queried.
        let res = unsafe { (self.d.api.cuStreamQuery.unwrap_unchecked())(self.handle) };
        match self.d.check(res, "cuStreamQuery") {
            Ok(()) => Ok(true),
            Err(Error::NotReady) => Ok(false),
            Err(e) => Err(e),
        }
    }

    pub fn wait(&self, event: &Event) -> Result<(), Error> {
        // SAFETY: the stream made to wait on the event.
        let res = unsafe {
            (self.d.api.cuStreamWaitEvent.unwrap_unchecked())(self.handle, event.handle, 0)
        };
        self.d.check(res, "cuStreamWaitEvent")
    }
}

impl Drop for Stream<'_> {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return; // the test stub: no handle at all
        }
        // SAFETY: the stream destroyed once.
        unsafe { (self.d.api.cuStreamDestroy_v2.unwrap_unchecked())(self.handle) };
    }
}

/// Upstream: ``Event`` (stream.zig:44).
pub struct Event<'d> {
    pub d: &'d Driver,
    pub handle: abi::Event,
}

impl<'d> Event<'d> {
    /// Timing events cost a little more to record; ordering-only events
    /// skip the timestamp. Upstream: `init` (stream.zig:49).
    pub fn init(d: &'d Driver, timing: bool) -> Result<Event<'d>, Error> {
        let mut e: abi::Event = std::ptr::null_mut();
        let flags = if timing { 0 } else { abi::EVENT_DISABLE_TIMING };
        // SAFETY: the event created with its flags.
        let res = unsafe { (d.api.cuEventCreate.unwrap_unchecked())(&mut e, flags) };
        d.check(res, "cuEventCreate")?;
        Ok(Event { d, handle: e })
    }

    pub fn record(&self, stream: &Stream) -> Result<(), Error> {
        // SAFETY: the event recorded on the stream.
        let res = unsafe {
            (self.d.api.cuEventRecord.unwrap_unchecked())(self.handle, stream.handle)
        };
        self.d.check(res, "cuEventRecord")
    }

    pub fn synchronize(&self) -> Result<(), Error> {
        // SAFETY: the event's work drained.
        let res = unsafe { (self.d.api.cuEventSynchronize.unwrap_unchecked())(self.handle) };
        self.d.check(res, "cuEventSynchronize")
    }

    /// True once the event has fired. Upstream: `done` (stream.zig:67).
    pub fn done(&self) -> Result<bool, Error> {
        // SAFETY: the event queried.
        let res = unsafe { (self.d.api.cuEventQuery.unwrap_unchecked())(self.handle) };
        match self.d.check(res, "cuEventQuery") {
            Ok(()) => Ok(true),
            Err(Error::NotReady) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Milliseconds between two recorded timing events (about 0.5 us
    /// resolution). Upstream: `elapsedMs` (stream.zig:74).
    pub fn elapsed_ms(start: &Event, end: &Event) -> Result<f32, Error> {
        let mut ms: f32 = 0.0;
        // SAFETY: the elapsed time between the two events.
        let res = unsafe {
            (start.d.api.cuEventElapsedTime.unwrap_unchecked())(
                &mut ms,
                start.handle,
                end.handle,
            )
        };
        start.d.check(res, "cuEventElapsedTime")?;
        Ok(ms)
    }
}

impl Drop for Event<'_> {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return; // the test stub: no handle at all
        }
        // SAFETY: the event destroyed once.
        unsafe { (self.d.api.cuEventDestroy_v2.unwrap_unchecked())(self.handle) };
    }
}
