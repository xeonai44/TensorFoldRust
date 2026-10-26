//! What the round loop asks a backend (Metal, CUDA or the fake) to compute; drawn
//! tokens stay on the GPU behind handles.
//!
//! Port of `zig/src/core/lanes/backend.zig`. The vtable becomes a trait: required
//! slots stay required, the optional ones become default methods that refuse, so a
//! backend grows into them exactly the way upstream's null vtable slots do.

use crate::shape::Shape;
use crate::stream::{Held, Stream};
use crate::Error;
use tensorfold_engine::LogprobRow;

/// A token to feed: one the backend drew and holds (a handle), or a host value
/// (forced or pending). Upstream: `backend.zig:9  pub const Feed`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feed {
    Handle(u64),
    Value(u32),
}

/// One stream's verify window: its pending row, then its drafts, each row drawn at
/// its own keyed position. Upstream: `backend.zig:12  pub const Window`.
///
/// Safety: `stream` points at a live, boxed stream under the crate's invariants
/// (lib.rs); the backend reads it, never frees it.
#[derive(Clone, Copy)]
pub struct Window<'a> {
    pub stream: *mut Stream<'a>,
    pub pending: u32,
    /// Rows after the pending one taken from the drafts the backend holds for the
    /// stream.
    pub held: u32,
    /// Drafts known on the host (forced, copied, tree); empty when held > 0.
    pub tokens: &'a [u32],
    /// Each row's parent row (row 0: -1) for a tree; None for a chain.
    pub parents: Option<&'a [i32]>,
    /// The absolute position each row's draw is keyed at.
    pub positions: &'a [u64],
    /// Also draft each row's first head draft behind the verify (speculate_early).
    pub early: bool,
}

impl Window<'_> {
    /// Upstream: `rows` — 1 + held + tokens.
    pub fn rows(&self) -> usize {
        1 + self.held as usize + self.tokens.len()
    }
}

/// A drafted level's best tokens by the head (the first is the held draft) and the
/// head's probability for each. Upstream: `backend.zig:27  Alternative`.
#[derive(Clone, Copy)]
pub struct Alternative {
    pub tokens: [u32; 4],
    pub probs: [f64; 4],
}

/// A window's results: each row's drawn token, the held drafts the forward verified,
/// and logprob rows when asked. Upstream: `backend.zig:30  Verified`.
pub struct Verified<'a> {
    pub sampled: &'a mut [u32],
    pub drafts: &'a mut [u32],
    pub rows: &'a mut [LogprobRow],
}

/// Tapped cache rows for a drafter: the prompt's until the first verify, then each
/// verify's kept rows until the next. Upstream: `backend.zig:33  Features`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Features {
    pub space: Space,
    pub dtype: Dtype,
    /// A host pointer, or a device pointer on `device`.
    pub buffer: u64,
    /// Bytes into `buffer` where the first row starts.
    pub offset: u64,
    /// Packed rows, row-major.
    pub rows: u32,
    /// A row is the drafter's taps in its order, each the hidden width of dtype.
    pub row_bytes: u32,
    /// The drafter must run on this device.
    pub device: i32,
    /// The fence to wait on before reading (None: written).
    pub ready: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    Host,
    Device,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F16,
    F32,
    U32,
}

/// Absorb a stream's kept rows into its draft head and hold `depth` drafts for its
/// next round. Upstream: `backend.zig:48  DraftRequest`.
///
/// Safety: `stream` per the crate invariants; `lanes`, when set, points at a Shape
/// the caller keeps alive until the next `draft`.
#[derive(Clone, Copy)]
pub struct DraftRequest<'a, 's> {
    pub stream: *mut Stream<'s>,
    /// The token after each kept row (the last is the stream's pending token).
    pub follow: &'a [u32],
    /// The prompt's first token when it is still on the device (follow is then
    /// empty).
    pub first: Option<Feed>,
    /// The kept rows of the last verify; None: the prompt's last row.
    pub rows: Option<&'a [u32]>,
    /// The cache length the last verify started from.
    pub start: u64,
    /// Where the first draft lands (the new cache length + 1).
    pub position: u64,
    /// Drafts to hold (0: absorb only).
    pub depth: u32,
    /// Settle the drafts the verify began (speculate_early) instead of drafting
    /// late.
    pub early: bool,
    /// A tree of drafts to hold, each lane its parent's rank-r head token (depth
    /// unused).
    pub lanes: Option<&'a Shape>,
    /// Also keep each chained level's best tokens (alternatives(), for grafts).
    pub ranks: bool,
}

impl<'a, 's> DraftRequest<'a, 's> {
    /// Upstream's field defaults (backend.zig:48).
    pub fn new(stream: *mut Stream<'s>, follow: &'a [u32], start: u64, position: u64, depth: u32) -> Self {
        DraftRequest {
            stream,
            follow,
            first: None,
            rows: None,
            start,
            position,
            depth,
            early: false,
            lanes: None,
            ranks: false,
        }
    }
}

/// What the round loop asks a backend to compute. Upstream: `backend.zig:61
/// pub const Backend` — the vtable.
///
/// Safety: every `*mut Stream` the loop hands a backend is a live, boxed stream
/// under the crate's invariants; backends may read and mutate them and must not
/// free them.
pub trait Backend {
    /// Prefill the stream's prompt into its caches (and its draft head's).
    fn prefill(&mut self, s: *mut Stream<'_>) -> Result<(), Error>;
    /// Draw the first token from the prompt's last row at `position`; a handle to
    /// it.
    fn first(&mut self, s: *mut Stream<'_>, position: u64) -> Result<u64, Error>;
    /// Feed one token and queue the draw of the next at `position` (a one-token
    /// round); a handle to it.
    fn queue(&mut self, s: *mut Stream<'_>, feed: Feed, position: u64) -> Result<u64, Error>;
    /// The value of a drawn token (waits for the GPU).
    fn read(&mut self, handle: u64) -> Result<u32, Error>;
    /// One forward over every window, each row drawn with its stream's sampling at
    /// its position.
    fn verify(&mut self, windows: &[Window<'_>], out: &mut [Verified<'_>]) -> Result<(), Error>;
    /// Roll each stream's caches back to its kept rows (a prefix count, or a tree's
    /// path).
    fn keep(&mut self, windows: &[Window<'_>], paths: &[&[u32]]) -> Result<(), Error>;
    /// Absorb kept rows and hold drafts for the next round (one batch for a shared
    /// round).
    fn draft(&mut self, requests: &[DraftRequest<'_, '_>]) -> Result<(), Error>;

    /// Undo an early speculation whose round was cut (the drafts come from a late
    /// request instead). Upstream: optional `unspeculate`.
    fn unspeculate(&mut self, _s: *mut Stream<'_>) -> Result<(), Error> {
        Err(Error::Backend("the backend does not unspeculate".into()))
    }
    /// Each held draft's chance of landing, for heads that give them; false when it
    /// has none. Upstream: optional `probabilities`.
    fn probabilities(&mut self, _s: *mut Stream<'_>, _out: &mut [f64]) -> Result<bool, Error> {
        Ok(false)
    }
    /// A tree head's held drafts as host tokens and parents, read where Python
    /// reads them. Upstream: optional `tree`.
    fn tree(&mut self, _s: *mut Stream<'_>) -> Result<Option<Held>, Error> {
        Ok(None)
    }
    /// The head's best tokens and probabilities at each level it drafted for the
    /// stream (waits for the drafts). Upstream: optional `alternatives`.
    fn alternatives(&mut self, _s: *mut Stream<'_>, _out: &mut [Alternative]) -> Result<usize, Error> {
        Ok(0)
    }
    /// The drafter's layers, told once before any forward so the target keeps only
    /// those. Upstream: optional `prepare_features`.
    fn prepare_features(&mut self, _taps: &[u32]) -> Result<(), Error> {
        Ok(())
    }
    /// Cache rows [start, start + count) of the prepared taps, within what
    /// `Features` says is held. Upstream: optional `features`.
    fn features(
        &mut self,
        _s: *mut Stream<'_>,
        _taps: &[u32],
        _start: u64,
        _count: u32,
    ) -> Result<Features, Error> {
        Err(Error::NoFeatures)
    }
    /// The first token's logprob row (the prompt's last row) for a stream with
    /// `logprobs`. Upstream: optional `first_row`.
    fn first_row(&mut self, _s: *mut Stream<'_>) -> Result<LogprobRow, Error> {
        Err(Error::LogprobsUnsupported)
    }
    /// The stream left the rounds: free its caches and held drafts.
    fn release(&mut self, s: *mut Stream<'_>);

    /// A driver of its own starts a stream: the prompt pass and its first token,
    /// committed; a cancel or failure releases it. Upstream: `opening`
    /// (backend.zig:132).
    ///
    /// # Safety
    ///
    /// As the trait's stream pointers: `s` must be a live, boxed stream under
    /// the crate's invariants.
    unsafe fn opening(&mut self, s: *mut Stream<'_>) -> Result<u32, Error> {
        let result = (|| -> Result<u32, Error> {
            self.prefill(s)?;
            // Safety: the invariant-checked stream pointer.
            let handle = {
                let stream = unsafe { &mut *s };
                let position = stream.prompt_len as u64;
                self.first(s, position)?
            };
            let token = self.read(handle)?;
            // Safety: commit takes the stream's own state.
            unsafe { &mut *s }.commit(&[token], &[])?;
            Ok(token)
        })();
        if result.is_err() {
            self.release(s);
        }
        result
    }
}

/// Where the round loop reads time: the depth rule's costs and a shared round's
/// overhead. Upstream: `backend.zig:142  Mark` + `Clock` (a vtable).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    Cost,
    Overhead,
}

pub trait Clock {
    fn start(&mut self);
    fn elapsed_ms(&mut self, mark: Mark) -> f64;
}

/// The monotonic clock production rounds read (Python time.perf_counter:
/// CLOCK_UPTIME_RAW on macOS). Upstream: `backend.zig:162  WallClock`.
#[derive(Default)]
pub struct WallClock {
    started: Option<std::time::Instant>,
}



impl Clock for WallClock {
    fn start(&mut self) {
        self.started = Some(std::time::Instant::now());
    }

    fn elapsed_ms(&mut self, _mark: Mark) -> f64 {
        match self.started {
            Some(t) => t.elapsed().as_secs_f64() * 1e3,
            None => 0.0,
        }
    }
}

/// A clock the tests control: fixed costs, so depth decisions repeat.
/// Upstream: `fake.zig:230  FixedClock` (the clock half lives there; it belongs to
/// every backend).
#[derive(Default)]
pub struct FixedClock {
    pub calls: u64,
}



impl Clock for FixedClock {
    fn start(&mut self) {}

    fn elapsed_ms(&mut self, _mark: Mark) -> f64 {
        self.calls += 1;
        6.0 + (self.calls % 7) as f64 * 0.5
    }
}
